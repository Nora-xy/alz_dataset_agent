"""P0 tests: schema construction, suppression edge cases, ToolResult invariants."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from alz_agent.constants import SUPPRESSED, SUPPRESSION_THRESHOLD, suppress
from alz_agent.schemas import (
    CatalogEntry,
    CohortMap,
    CohortRole,
    CoverageStats,
    DataEvidence,
    DocEvidence,
    FilterSpec,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
    Tier,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 10, tzinfo=timezone.utc)


def _make_coverage(**kwargs) -> CoverageStats:
    defaults = dict(n_available=100, longitudinal_availability=60, missingness_rate=0.1)
    defaults.update(kwargs)
    return CoverageStats(**defaults)


def _make_catalog_entry(**kwargs) -> CatalogEntry:
    defaults = dict(
        variable_id="CDRSUM",
        concept="cdr_sum_of_boxes",
        source_dataset=SourceDataset.NACC,
        id_space=IdSpace.A,
        raw_code="CDRSUM",
        release_version="nacc70",
        modality=Modality.COGNITION,
        leakage_tags=[LeakageTag.NONE],
    )
    defaults.update(kwargs)
    return CatalogEntry(**defaults)


# ---------------------------------------------------------------------------
# suppress() — edge cases
# ---------------------------------------------------------------------------


class TestSuppress:
    def test_zero_passes_through(self):
        assert suppress(0) == 0

    def test_one_is_suppressed(self):
        assert suppress(1) is SUPPRESSED

    def test_threshold_minus_one_is_suppressed(self):
        assert suppress(SUPPRESSION_THRESHOLD - 1) is SUPPRESSED

    def test_threshold_passes_through(self):
        assert suppress(SUPPRESSION_THRESHOLD) == SUPPRESSION_THRESHOLD

    def test_threshold_plus_one_passes_through(self):
        assert suppress(SUPPRESSION_THRESHOLD + 1) == SUPPRESSION_THRESHOLD + 1

    def test_large_count_passes_through(self):
        assert suppress(50_000) == 50_000

    def test_suppressed_singleton(self):
        # SUPPRESSED is a singleton
        from alz_agent.constants import _Suppressed

        assert suppress(1) is suppress(5) is _Suppressed()

    def test_suppressed_repr(self):
        assert repr(SUPPRESSED) == "SUPPRESSED"
        assert str(SUPPRESSED) == "SUPPRESSED"

    def test_threshold_value(self):
        assert SUPPRESSION_THRESHOLD == 10


# ---------------------------------------------------------------------------
# CoverageStats
# ---------------------------------------------------------------------------


class TestCoverageStats:
    def test_construction(self):
        cs = _make_coverage()
        assert cs.n_available == 100
        assert cs.longitudinal_availability == 60
        assert cs.missingness_rate == 0.1
        assert cs.last_computed is None

    def test_with_timestamp(self):
        cs = _make_coverage(last_computed=_NOW)
        assert cs.last_computed == _NOW

    def test_immutability(self):
        cs = _make_coverage()
        with pytest.raises((TypeError, ValidationError)):
            cs.n_available = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CatalogEntry
# ---------------------------------------------------------------------------


class TestCatalogEntry:
    def test_minimal_construction(self):
        entry = _make_catalog_entry()
        assert entry.variable_id == "CDRSUM"
        assert entry.source_dataset == SourceDataset.NACC
        assert entry.id_space == IdSpace.A
        assert entry.modality == Modality.COGNITION
        assert entry.coverage is None

    def test_with_coverage(self):
        cs = _make_coverage()
        entry = _make_catalog_entry(coverage=cs)
        assert entry.coverage is not None
        assert entry.coverage.n_available == 100

    def test_leakage_tags_diagnosis_defining(self):
        entry = _make_catalog_entry(
            variable_id="NACCUDSD",
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
        )
        assert LeakageTag.DIAGNOSIS_DEFINING in entry.leakage_tags

    def test_leakage_tags_multiple(self):
        tags = [LeakageTag.DIAGNOSIS_DEFINING, LeakageTag.CIRCULAR]
        entry = _make_catalog_entry(leakage_tags=tags)
        assert len(entry.leakage_tags) == 2

    def test_harmonization_transform_nullable(self):
        entry = _make_catalog_entry(harmonization_transform=None)
        assert entry.harmonization_transform is None

    def test_id_space_b(self):
        entry = _make_catalog_entry(
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
        )
        assert entry.id_space == IdSpace.B

    def test_modality_defaults_to_other(self):
        entry = _make_catalog_entry(modality=Modality.OTHER)
        assert entry.modality == Modality.OTHER

    def test_value_coding_stored(self):
        coding = {"0": "Normal", "0.5": "MCI", "1": "MCI", "2": "Dementia"}
        entry = _make_catalog_entry(value_coding=coding)
        assert entry.value_coding["0.5"] == "MCI"

    def test_immutability(self):
        entry = _make_catalog_entry()
        with pytest.raises((TypeError, ValidationError)):
            entry.variable_id = "OTHER"  # type: ignore[misc]

    def test_family_and_variant_default_none(self):
        entry = _make_catalog_entry()
        assert entry.family is None
        assert entry.variant is None

    def test_family_and_variant_settable(self):
        entry = _make_catalog_entry(family="ptau", variant="plasma_harmonized_217")
        assert entry.family == "ptau"
        assert entry.variant == "plasma_harmonized_217"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_data_evidence(self):
        ev = DataEvidence(
            query_spec={"filter": "baseline_dx=MCI", "id_space": "A"},
            aggregate_ref="count_eligible => 1234",
        )
        assert ev.kind == "DATA"
        assert "filter" in ev.query_spec

    def test_doc_evidence(self):
        ev = DocEvidence(
            source="NACC/uds3-rdd.pdf",
            locator="Table B3, variable CDRSUM",
            snippet="CDR Sum of Boxes: sum of ratings in six domains.",
        )
        assert ev.kind == "DOC"
        assert "uds3-rdd.pdf" in ev.source

    def test_data_evidence_immutable(self):
        ev = DataEvidence(query_spec={}, aggregate_ref="x")
        with pytest.raises((TypeError, ValidationError)):
            ev.aggregate_ref = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_grounded_result(self):
        ev = DataEvidence(query_spec={"id_space": "A"}, aggregate_ref="n=500")
        result: ToolResult[int] = ToolResult.ok(500, evidence=[ev])
        assert result.grounded is True
        assert result.value == 500
        assert result.abstain_reason is None
        assert len(result.evidence) == 1

    def test_abstain_result(self):
        result: ToolResult[int] = ToolResult.abstain("R4 not available locally")
        assert result.grounded is False
        assert result.value is None
        assert result.abstain_reason == "R4 not available locally"
        assert result.evidence == []

    def test_grounded_false_requires_abstain_reason(self):
        with pytest.raises(ValidationError, match="abstain_reason"):
            ToolResult(grounded=False, value=None, evidence=[])

    def test_grounded_false_requires_value_none(self):
        with pytest.raises(ValidationError, match="value=None"):
            ToolResult(grounded=False, value=42, evidence=[], abstain_reason="reason")

    def test_grounded_true_with_none_value_is_allowed(self):
        # grounded=True + value=None is legal (e.g. result exists but is empty)
        result: ToolResult[None] = ToolResult(grounded=True, value=None, evidence=[])
        assert result.grounded is True

    def test_abstain_factory(self):
        r = ToolResult.abstain("cross-cohort linkage unverified — crosswalk not built")
        assert not r.grounded
        assert "crosswalk" in r.abstain_reason

    def test_ok_factory(self):
        r = ToolResult.ok({"sufficient": 800, "censored": 50})
        assert r.grounded
        assert r.value["sufficient"] == 800

    def test_immutability(self):
        r = ToolResult.ok(1)
        with pytest.raises((TypeError, ValidationError)):
            r.value = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FilterSpec
# ---------------------------------------------------------------------------


class TestFilterSpec:
    def test_defaults(self):
        fs = FilterSpec()
        assert fs.id_space == IdSpace.A
        assert fs.baseline_diagnosis is None
        assert fs.required_modalities is None

    def test_full_construction(self):
        fs = FilterSpec(
            baseline_diagnosis=["MCI", "Normal"],
            visit_window=(1, 5),
            required_modalities=["cognition", "mri"],
            label_certainty="definite",
            id_space=IdSpace.A,
        )
        assert fs.baseline_diagnosis == ["MCI", "Normal"]
        assert fs.visit_window == (1, 5)

    def test_space_b(self):
        fs = FilterSpec(id_space=IdSpace.B)
        assert fs.id_space == IdSpace.B

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError, match="Unknown FilterSpec fields"):
            FilterSpec(unknown_field="oops")  # type: ignore[call-arg]

    def test_immutability(self):
        fs = FilterSpec()
        with pytest.raises((TypeError, ValidationError)):
            fs.id_space = IdSpace.B  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CohortMap
# ---------------------------------------------------------------------------


class TestCohortMap:
    def _make_map(self, **kwargs) -> CohortMap:
        defaults = dict(
            id_space=IdSpace.A,
            tier=Tier.TIER1,
            cohort_role=CohortRole.TRAINING,
            eligible_n=1500,
            index_visit_def="first visit with MCI diagnosis",
            prediction_horizon=3,
            outcome_label_def="conversion to AD at 3 years",
            suppression_threshold=SUPPRESSION_THRESHOLD,
            generated_at=_NOW,
        )
        defaults.update(kwargs)
        return CohortMap(**defaults)

    def test_construction(self):
        cm = self._make_map()
        assert cm.id_space == IdSpace.A
        assert cm.tier == Tier.TIER1
        assert cm.eligible_n == 1500
        assert cm.prediction_horizon == 3

    def test_eligible_n_nullable_for_suppressed(self):
        cm = self._make_map(eligible_n=None)
        assert cm.eligible_n is None

    def test_tier3_external_validation(self):
        cm = self._make_map(
            id_space=IdSpace.B,
            tier=Tier.TIER3,
            cohort_role=CohortRole.EXTERNAL_VALIDATION_CANDIDATE,
            limitations=["NACC<->ADNI subject overlap undetectable without crosswalk"],
        )
        assert cm.tier == Tier.TIER3
        assert any("crosswalk" in lim for lim in cm.limitations)

    def test_modality_coverage_dict(self):
        cm = self._make_map(
            modality_coverage={"cognition": 1400, "mri": 300, "fluid_biomarker": 80}
        )
        assert cm.modality_coverage["cognition"] == 1400

    def test_feature_and_leakage_lists(self):
        cm = self._make_map(
            feature_whitelist=["CDRSUM", "MMSE"],
            leakage_blacklist=["NACCUDSD", "DXCONFID"],
        )
        assert "CDRSUM" in cm.feature_whitelist
        assert "NACCUDSD" in cm.leakage_blacklist

    def test_release_versions(self):
        cm = self._make_map(
            release_versions={"NACC": "nacc70", "ADSP_PHC": "R3_Dec2024"}
        )
        assert cm.release_versions["ADSP_PHC"] == "R3_Dec2024"

    def test_immutability(self):
        cm = self._make_map()
        with pytest.raises((TypeError, ValidationError)):
            cm.eligible_n = 999  # type: ignore[misc]
