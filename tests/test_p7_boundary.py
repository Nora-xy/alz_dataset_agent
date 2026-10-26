"""P7 tests: deployment boundary (transport, suppression re-check guard, serializer).

Covers:
  1. ToolCallRequest dispatch for every tool, on FixtureDataBackend.
  2. The suppression re-check guard, including a *real* discovered leak:
     FixtureDataBackend.followup_distribution's evidence.aggregate_ref names a
     suppressed bucket's raw pre-suppression count even though the structured
     value already suppresses it (None). The boundary must redact that text.
  3. A row-level-data guard that fails loudly (BoundaryViolation) rather than
     silently redacting.
  4. Serialization round-trips for both ToolCallRequest and ToolResult.
  5. A LocalDataBackend smoke test behind the same real-data file-existence
     guard used in test_p2_backend.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.backend.local_backend import LocalDataBackend
from alz_agent.boundary.transport import (
    BoundaryViolation,
    ToolCallRequest,
    _check_and_redact_value,
    _enforce_boundary,
    deserialize_request,
    deserialize_tool_result,
    execute_tool_call,
    serialize_request,
    serialize_tool_result,
)
from alz_agent.constants import SUPPRESSION_THRESHOLD
from alz_agent.schemas import (
    CatalogEntry,
    CoverageStats,
    DataEvidence,
    FilterSpec,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
    ToolResult,
)
from alz_agent.tools.deterministic import ModelingDesignResult, VariableLookupResult

PROJECT_ROOT = Path(__file__).parent.parent
PATHS_YAML = PROJECT_ROOT / "config" / "paths.yaml"

# ---------------------------------------------------------------------------
# Shared fixtures (same Space A catalog as test_p5_agent.py / test_p6_cohort_map.py)
# ---------------------------------------------------------------------------


def _make_catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            variable_id="NACC.NACCUDSD",
            concept="nacc_unified_diagnostic_summary",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCUDSD",
            release_version="nacc70",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            leakage_reasons=["Primary outcome label"],
        ),
        CatalogEntry(
            variable_id="NACC.MMSE30",
            concept="mmse_total_score",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="MMSE30",
            release_version="nacc70",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
        CatalogEntry(
            variable_id="NACC.MRI_VOL",
            concept="mri_brain_volume",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="MRI_VOL",
            release_version="nacc70",
            modality=Modality.MRI,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
    ]


@pytest.fixture()
def catalog() -> list[CatalogEntry]:
    return _make_catalog()


@pytest.fixture()
def backend():
    return build_fixture_backend(
        n_normal=50,
        n_mci=40,
        n_dementia=30,
        n_other=20,
        n_mci_long=25,
        n_mci_short=8,
        n_single_visit=15,
        n_suppressed_group=5,
        n_mci_ad=15,
        n_dementia_ad=15,
        modality_fractions={
            "cognition": 0.9,
            "mri": 0.15,
            "pet": 0.04,
            "fluid_biomarker": 0.05,
            "vascular": 0.8,
            "neuropath": 0.0,
        },
        seed=42,
    )


MCI_SPEC = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A)


# ---------------------------------------------------------------------------
# Dispatch: one tool call per tool_name
# ---------------------------------------------------------------------------


class TestDispatchEveryTool:
    def test_compute_cohort_size(self, catalog, backend):
        req = ToolCallRequest(tool_name="compute_cohort_size", filter_spec=MCI_SPEC)
        result = execute_tool_call(req, catalog, backend)
        assert result.grounded
        assert isinstance(result.value, int)

    def test_lookup_variable(self, catalog, backend):
        req = ToolCallRequest(tool_name="lookup_variable", query="NACC.MMSE30")
        result = execute_tool_call(req, catalog, backend)
        assert result.grounded
        assert isinstance(result.value, VariableLookupResult)

    def test_assess_modality_overlap(self, catalog, backend):
        req = ToolCallRequest(
            tool_name="assess_modality_overlap",
            filter_spec=MCI_SPEC,
            modalities=["cognition", "mri"],
        )
        result = execute_tool_call(req, catalog, backend)
        assert result.grounded
        assert "cognition" in result.value

    def test_assess_longitudinal_feasibility(self, catalog, backend):
        req = ToolCallRequest(
            tool_name="assess_longitudinal_feasibility",
            filter_spec=MCI_SPEC,
            horizon=3,
        )
        result = execute_tool_call(req, catalog, backend)
        assert result.grounded
        assert set(result.value.keys()) >= {"sufficient", "insufficient", "censored"}

    def test_check_leakage(self, catalog, backend):
        req = ToolCallRequest(
            tool_name="check_leakage",
            feature_set=["NACC.NACCUDSD", "NACC.MMSE30"],
            label_def="conversion to dementia within 3 years",
        )
        result = execute_tool_call(req, catalog, backend)
        assert result.grounded
        assert "NACC.NACCUDSD" in [f.variable_id for f in result.value.flagged]

    def test_suggest_modeling_design(self, catalog, backend):
        req = ToolCallRequest(
            tool_name="suggest_modeling_design", filter_spec=MCI_SPEC, horizon=3
        )
        result = execute_tool_call(req, catalog, backend)
        assert result.grounded
        assert isinstance(result.value, ModelingDesignResult)

    def test_missing_required_param_abstains_not_raises(self, catalog, backend):
        req = ToolCallRequest(tool_name="compute_cohort_size", filter_spec=None)
        result = execute_tool_call(req, catalog, backend)
        assert not result.grounded
        assert "filter_spec" in result.abstain_reason


# ---------------------------------------------------------------------------
# Suppression re-check guard — the real discovered leak
# ---------------------------------------------------------------------------
#
# FixtureDataBackend.followup_distribution's evidence.aggregate_ref always
# embeds the raw pre-suppression sufficient/insufficient/censored counts, even
# for a bucket whose structured value the tool itself already suppressed to
# None. The boundary must redact that text before it can cross.


class TestSuppressionReCheckGuard:
    @pytest.fixture()
    def small_censored_backend(self):
        # n_single_visit=3 -> "censored" bucket = exactly 3 (below threshold),
        # while sufficient/insufficient buckets stay well above threshold.
        return build_fixture_backend(
            n_normal=0, n_dementia=0, n_other=0,
            n_mci=0, n_mci_long=20, n_mci_short=5, n_single_visit=3,
            n_suppressed_group=0, n_mci_ad=0, n_dementia_ad=0,
            modality_fractions={"cognition": 0.9},
            seed=7,
        )

    def test_value_already_suppressed_by_tool(self, catalog, small_censored_backend):
        req = ToolCallRequest(
            tool_name="assess_longitudinal_feasibility",
            filter_spec=MCI_SPEC,
            horizon=3,
        )
        result = execute_tool_call(req, catalog, small_censored_backend)
        assert result.grounded
        assert result.value["censored"] is None

    def test_evidence_text_leak_is_redacted_by_boundary(self, catalog, small_censored_backend):
        req = ToolCallRequest(
            tool_name="assess_longitudinal_feasibility",
            filter_spec=MCI_SPEC,
            horizon=3,
        )
        result = execute_tool_call(req, catalog, small_censored_backend)
        ref = result.evidence[0].aggregate_ref
        assert "censored=3" not in ref, f"Raw suppressed count leaked through evidence: {ref!r}"
        assert "censored=[REDACTED]" in ref

    def test_raw_backend_call_demonstrates_the_underlying_leak(self, small_censored_backend):
        """Sanity check: the leak exists in the raw backend output (pre-boundary)."""
        raw = small_censored_backend.followup_distribution(3, MCI_SPEC)
        assert raw.value["censored"] is None  # tool-level suppression: correct
        assert "censored=3" in raw.evidence[0].aggregate_ref  # text-level leak: the bug


class TestSuppressionReCheckUnitLevel:
    """White-box tests of _check_and_redact_value / _enforce_boundary."""

    def test_dict_with_subthreshold_value_is_redacted(self):
        redacted, notes = _check_and_redact_value({"cognition": 50, "pet": 4})
        assert redacted == {"cognition": 50, "pet": None}
        assert notes

    def test_dict_with_zero_is_not_redacted(self):
        redacted, notes = _check_and_redact_value({"neuropath": 0})
        assert redacted == {"neuropath": 0}
        assert not notes

    def test_dict_at_threshold_is_not_redacted(self):
        redacted, notes = _check_and_redact_value({"mri": SUPPRESSION_THRESHOLD})
        assert redacted == {"mri": SUPPRESSION_THRESHOLD}
        assert not notes

    def test_top_level_int_violation_forces_abstain(self):
        leaky = ToolResult.ok(
            7,
            evidence=[DataEvidence(query_spec={}, aggregate_ref="count_eligible=7")],
        )
        result = _enforce_boundary(leaky)
        assert not result.grounded
        assert "suppression" in result.abstain_reason.lower()

    def test_grounded_value_above_threshold_passes_through_unchanged(self):
        clean = ToolResult.ok(
            42,
            evidence=[DataEvidence(query_spec={}, aggregate_ref="count_eligible=42")],
        )
        result = _enforce_boundary(clean)
        assert result.grounded
        assert result.value == 42
        assert result.evidence[0].aggregate_ref == "count_eligible=42"

    def test_abstain_result_passes_through_unchanged(self):
        abstained = ToolResult.abstain("no data")
        result = _enforce_boundary(abstained)
        assert result is abstained

    def test_variable_lookup_coverage_redacted(self):
        leaky = ToolResult.ok(
            VariableLookupResult(
                variable_id="NACC.RARE_VAR",
                concept="rare_variable",
                source_dataset="NACC",
                raw_code="RARE_VAR",
                modality="other",
                coverage=CoverageStats(
                    n_available=4, longitudinal_availability=2, missingness_rate=0.99
                ),
            )
        )
        result = _enforce_boundary(leaky)
        assert result.grounded
        assert result.value.coverage is None

    def test_modeling_design_eligible_n_and_feasibility_redacted(self):
        leaky = ToolResult.ok(
            ModelingDesignResult(
                index_visit_def="x", horizon_years=3, outcome_label_def="y", id_space="A",
                eligible_n=5,
                feasibility={"sufficient": 20, "insufficient": 3, "censored": 0},
            )
        )
        result = _enforce_boundary(leaky)
        assert result.grounded
        assert result.value.eligible_n is None
        assert result.value.feasibility == {"sufficient": 20, "insufficient": None, "censored": 0}


# ---------------------------------------------------------------------------
# Row-level data guard — fails loudly, never silently redacts
# ---------------------------------------------------------------------------


class TestRowLevelDataGuardFailsLoudly:
    def test_naccid_in_query_spec_raises(self):
        leaky = ToolResult.ok(
            10,
            evidence=[DataEvidence(query_spec={"naccid": "NACC000123"}, aggregate_ref="n=10")],
        )
        with pytest.raises(BoundaryViolation):
            _enforce_boundary(leaky)

    def test_rid_in_aggregate_ref_raises(self):
        leaky = ToolResult.ok(
            10,
            evidence=[DataEvidence(query_spec={}, aggregate_ref="leaked rid=42 in debug text")],
        )
        with pytest.raises(BoundaryViolation):
            _enforce_boundary(leaky)

    def test_clean_payload_does_not_raise(self, catalog, backend):
        req = ToolCallRequest(tool_name="compute_cohort_size", filter_spec=MCI_SPEC)
        # Should not raise — real tool outputs never carry subject identifiers.
        execute_tool_call(req, catalog, backend)


# ---------------------------------------------------------------------------
# Serialization round-trips ("wire" transfer)
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_request_round_trips(self):
        req = ToolCallRequest(
            tool_name="assess_longitudinal_feasibility", filter_spec=MCI_SPEC, horizon=3
        )
        wire = serialize_request(req)
        restored = deserialize_request(wire)
        assert restored == req

    def test_request_wire_payload_is_json_safe(self):
        req = ToolCallRequest(tool_name="compute_cohort_size", filter_spec=MCI_SPEC)
        wire = serialize_request(req)
        import json
        json.dumps(wire)  # must not raise

    def test_result_round_trips_for_dict_value(self, catalog, backend):
        req = ToolCallRequest(
            tool_name="assess_modality_overlap",
            filter_spec=MCI_SPEC,
            modalities=["cognition", "mri"],
        )
        result = execute_tool_call(req, catalog, backend)
        wire = serialize_tool_result(result)
        restored = deserialize_tool_result(wire)
        assert restored.grounded == result.grounded
        assert restored.value == result.value

    def test_result_round_trips_for_modeling_design(self, catalog, backend):
        """ToolResult[Any] reconstructs the nested model as a plain dict — the
        cloud side narrates from JSON-safe values, it never reconstructs
        rich local-only Python classes."""
        req = ToolCallRequest(
            tool_name="suggest_modeling_design", filter_spec=MCI_SPEC, horizon=3
        )
        result = execute_tool_call(req, catalog, backend)
        wire = serialize_tool_result(result)
        restored = deserialize_tool_result(wire)
        assert restored.grounded
        assert isinstance(restored.value, dict)
        assert restored.value["horizon_years"] == 3
        assert "feature_whitelist" in restored.value

    def test_full_hybrid_round_trip_worked_target(self, catalog, backend):
        """End-to-end demonstration of the cloud<->local split for the P5
        worked-target question, entirely through serialized envelopes."""
        cloud_request = ToolCallRequest(
            tool_name="suggest_modeling_design", filter_spec=MCI_SPEC, horizon=3
        )
        wire_request = serialize_request(cloud_request)  # cloud -> wire

        local_request = deserialize_request(wire_request)  # wire -> local
        local_result = execute_tool_call(local_request, catalog, backend)
        wire_result = serialize_tool_result(local_result)  # local -> wire

        cloud_result = deserialize_tool_result(wire_result)  # wire -> cloud
        assert cloud_result.grounded
        assert cloud_result.value["eligible_n"] is not None
        assert cloud_result.value["eligible_n"] > 0


# ---------------------------------------------------------------------------
# Real-data smoke test — same file-existence guard pattern as test_p2_backend.py
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PROJECT_ROOT / "NACC" / "investigator_nacc70.csv").exists(),
    reason="Real NACC data not mounted",
)
class TestBoundaryWithRealLocalBackend:
    @pytest.fixture(scope="class")
    def real_backend(self):
        with open(PATHS_YAML, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        return LocalDataBackend(config, PROJECT_ROOT)

    def test_compute_cohort_size_crosses_boundary_cleanly(self, real_backend):
        req = ToolCallRequest(
            tool_name="compute_cohort_size",
            filter_spec=FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A),
        )
        result = execute_tool_call(req, [], real_backend)
        assert result.grounded
        assert result.value >= 10_000  # MCI baseline cohort is far above threshold

    def test_result_serializes_cleanly(self, real_backend):
        req = ToolCallRequest(
            tool_name="compute_cohort_size",
            filter_spec=FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A),
        )
        result = execute_tool_call(req, [], real_backend)
        wire = serialize_tool_result(result)
        import json
        json.dumps(wire)  # must not raise
