"""P6 tests: cohort map generator + serializer.

Golden test target (same worked-target question as test_p5_agent.py):
  "Among MCI participants at an index visit, can we predict 3-year conversion
   to AD using only pre-index information, and how many subjects survive that
   definition?"

All tests run on FixtureDataBackend — no real data or API key required.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.cohort_map.generator import (
    CohortMapGenerationReport,
    deserialize_cohort_map,
    generate_cohort_map,
    serialize_cohort_map,
)
from alz_agent.constants import SUPPRESSION_THRESHOLD
from alz_agent.schemas import (
    CatalogEntry,
    CohortMap,
    CohortRole,
    FilterSpec,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
    Tier,
)

# ---------------------------------------------------------------------------
# Worked-target filter_spec (mirrors test_p5_agent.py's WORKED_TARGET question)
# ---------------------------------------------------------------------------

WORKED_TARGET_FILTER_SPEC = FilterSpec(
    baseline_diagnosis=["mci"],
    id_space=IdSpace.A,
)
WORKED_TARGET_HORIZON = 3


def _make_space_a_catalog() -> list[CatalogEntry]:
    """Same Space A catalog as test_p5_agent.py's worked-target scenario."""
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
            leakage_reasons=["Primary outcome label — defines conversion endpoint"],
        ),
        CatalogEntry(
            variable_id="NACC.CDRSUM",
            concept="cdr_sum_of_boxes",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="CDRSUM",
            release_version="nacc70",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.CIRCULAR],
            leakage_reasons=["CDR Sum of Boxes — collinear with outcome"],
        ),
        CatalogEntry(
            variable_id="ADSP_PHC.Cognition.PHC_MEM",
            concept="phc_memory_composite",
            source_dataset=SourceDataset.ADSP_PHC,
            id_space=IdSpace.A,
            raw_code="PHC_MEM",
            release_version="R3",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.HARMONIZATION_FIT_SCOPE],
            leakage_reasons=["GMM fit on full cohort including outcome visits"],
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
        CatalogEntry(
            variable_id="NACC.PET_AMYLOID",
            concept="pet_amyloid_suvr",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="PET_AMYLOID",
            release_version="nacc70",
            modality=Modality.PET,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
    ]


def _make_space_b_catalog() -> list[CatalogEntry]:
    """Minimal Space B (ADNI) catalog for external-validation tests."""
    return [
        CatalogEntry(
            variable_id="ADNI.DXSUM.DIAGNOSIS",
            concept="adni_diagnosis",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="DIAGNOSIS",
            release_version="ADNI_Jun2026",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            leakage_reasons=["Primary outcome label"],
        ),
        CatalogEntry(
            variable_id="ADNI.MMSE.MMSCORE",
            concept="adni_mmse_total_score",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="MMSCORE",
            release_version="ADNI_Jun2026",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
    ]


@pytest.fixture()
def catalog() -> list[CatalogEntry]:
    return _make_space_a_catalog()


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


# ---------------------------------------------------------------------------
# Golden test: worked-target filter_spec produces a fully-populated CohortMap
# ---------------------------------------------------------------------------


class TestGoldenWorkedTarget:
    def test_returns_generation_report(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        assert isinstance(report, CohortMapGenerationReport)
        assert isinstance(report.cohort_map, CohortMap)

    def test_id_space_and_tier_default_to_space_a_training(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        cm = report.cohort_map
        assert cm.id_space == IdSpace.A
        assert cm.tier == Tier.TIER1
        assert cm.cohort_role == CohortRole.TRAINING

    def test_eligible_n_populated_not_suppressed(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        assert report.cohort_map.eligible_n is not None
        assert report.cohort_map.eligible_n > 0
        assert report.provenance["eligible_n"].available
        assert report.provenance["eligible_n"].evidence

    def test_prediction_horizon_matches_input(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        assert report.cohort_map.prediction_horizon == 3

    def test_index_visit_and_outcome_label_populated(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        cm = report.cohort_map
        assert cm.index_visit_def != "not available"
        assert cm.outcome_label_def != "not available"
        assert "mci" in cm.index_visit_def.lower()

    def test_follow_up_availability_populated(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        cm = report.cohort_map
        assert set(cm.follow_up_availability.keys()) >= {"sufficient", "insufficient", "censored"}

    def test_feature_whitelist_excludes_leakage_blacklist(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        cm = report.cohort_map
        assert "NACC.MMSE30" in cm.feature_whitelist
        assert "NACC.NACCUDSD" in cm.leakage_blacklist
        assert "NACC.CDRSUM" in cm.leakage_blacklist
        assert "ADSP_PHC.Cognition.PHC_MEM" in cm.leakage_blacklist
        assert not (set(cm.feature_whitelist) & set(cm.leakage_blacklist))

    def test_modality_coverage_and_combo_sample_sizes_populated(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        cm = report.cohort_map
        # cognition is abundant (0.9 fraction) so its singleton count should survive
        # suppression and appear as a non-comma key.
        assert "cognition" in cm.modality_coverage
        assert all("," not in k for k in cm.modality_coverage)
        assert all("," in k for k in cm.sample_size_by_modality_combo)

    def test_release_versions_populated_from_catalog(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        cm = report.cohort_map
        assert cm.release_versions.get("NACC") == "nacc70"
        assert cm.release_versions.get("ADSP_PHC") == "R3"
        # Space B's ADNI release must never appear in a Space A map (invariant 8).
        assert "ADNI" not in cm.release_versions

    def test_suppression_threshold_matches_constant(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        assert report.cohort_map.suppression_threshold == SUPPRESSION_THRESHOLD

    def test_generated_at_is_recent_utc(self, catalog, backend):
        before = datetime.now(timezone.utc)
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        after = datetime.now(timezone.utc)
        assert before <= report.cohort_map.generated_at <= after

    def test_no_space_b_caveat_for_space_a_map(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        assert not any("crosswalk" in lim.lower() for lim in report.cohort_map.limitations)

    def test_round_trips_through_serialization(self, catalog, backend, tmp_path):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        path = serialize_cohort_map(report.cohort_map, tmp_path / "worked_target.json")
        loaded = deserialize_cohort_map(path)
        assert loaded == report.cohort_map


# ---------------------------------------------------------------------------
# Suppression: when eligible_n / modality combos are suppressed, the map
# records "not available" rather than a fabricated number.
# ---------------------------------------------------------------------------


class TestSuppressionMarksUnavailable:
    def test_suppressed_visit_window_yields_none_eligible_n(self, catalog, backend):
        spec = FilterSpec(
            baseline_diagnosis=["mci"],
            visit_window=(15, 16),
            id_space=IdSpace.A,
        )
        report = generate_cohort_map(spec, WORKED_TARGET_HORIZON, catalog, backend)
        cm = report.cohort_map
        assert cm.eligible_n is None
        assert not report.provenance["eligible_n"].available
        assert "suppressed" in (report.provenance["eligible_n"].reason or "").lower()

    def test_pet_modality_suppressed_omitted_not_zero(self, catalog, backend):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC,
            WORKED_TARGET_HORIZON,
            catalog,
            backend,
            modalities=["cognition", "pet"],
        )
        cm = report.cohort_map
        # pet is rare (0.04 fraction) — its singleton count should be suppressed
        # and therefore absent rather than present as 0.
        assert cm.modality_coverage.get("pet") != 0
        assert cm.sample_size_by_modality_combo.get("cognition,pet") != 0
        # The omission must be recorded as a limitation, not silently dropped.
        assert any("suppress" in lim.lower() for lim in cm.limitations)

    def test_unknown_target_population_abstains_design_fields(self, catalog, backend):
        spec = FilterSpec(
            baseline_diagnosis=["not_a_real_diagnosis_xyz"],
            id_space=IdSpace.A,
        )
        report = generate_cohort_map(spec, WORKED_TARGET_HORIZON, catalog, backend)
        cm = report.cohort_map
        assert cm.eligible_n is None
        assert cm.index_visit_def == "not available"
        assert cm.outcome_label_def == "not available"
        assert cm.feature_whitelist == []
        assert cm.leakage_blacklist == []
        assert not report.provenance["eligible_n"].available
        assert not report.provenance["modeling_design"].available


# ---------------------------------------------------------------------------
# Space B / Tier 3 external-validation candidate
# ---------------------------------------------------------------------------


class TestSpaceBExternalValidation:
    @pytest.fixture()
    def space_b_catalog(self) -> list[CatalogEntry]:
        return _make_space_b_catalog()

    @pytest.fixture()
    def space_b_backend(self):
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
            modality_fractions={"cognition": 0.9, "vascular": 0.0, "fluid_biomarker": 0.0,
                                 "mri": 0.0, "pet": 0.0, "neuropath": 0.0},
            id_space=IdSpace.B,
            seed=42,
        )

    def test_space_b_forces_tier3_external_validation(
        self, space_b_catalog, space_b_backend
    ):
        spec = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.B)
        report = generate_cohort_map(spec, 3, space_b_catalog, space_b_backend)
        cm = report.cohort_map
        assert cm.tier == Tier.TIER3
        assert cm.cohort_role == CohortRole.EXTERNAL_VALIDATION_CANDIDATE

    def test_space_b_override_attempt_is_ignored(self, space_b_catalog, space_b_backend):
        spec = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.B)
        report = generate_cohort_map(
            spec, 3, space_b_catalog, space_b_backend,
            tier=Tier.TIER1, cohort_role=CohortRole.TRAINING,
        )
        # Space B is always Tier 3 / external_validation_candidate regardless
        # of caller-supplied overrides.
        assert report.cohort_map.tier == Tier.TIER3
        assert report.cohort_map.cohort_role == CohortRole.EXTERNAL_VALIDATION_CANDIDATE

    def test_space_b_carries_crosswalk_caveat(self, space_b_catalog, space_b_backend):
        spec = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.B)
        report = generate_cohort_map(spec, 3, space_b_catalog, space_b_backend)
        assert any("crosswalk" in lim.lower() for lim in report.cohort_map.limitations)

    def test_space_b_release_versions_only_adni(self, space_b_catalog, space_b_backend):
        spec = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.B)
        report = generate_cohort_map(spec, 3, space_b_catalog, space_b_backend)
        cm = report.cohort_map
        assert cm.release_versions.get("ADNI") == "ADNI_Jun2026"
        assert "NACC" not in cm.release_versions
        assert "ADSP_PHC" not in cm.release_versions

    def test_space_b_round_trips_through_serialization(
        self, space_b_catalog, space_b_backend, tmp_path
    ):
        spec = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.B)
        report = generate_cohort_map(spec, 3, space_b_catalog, space_b_backend)
        path = serialize_cohort_map(report.cohort_map, tmp_path / "space_b.json")
        loaded = deserialize_cohort_map(path)
        assert loaded == report.cohort_map


# ---------------------------------------------------------------------------
# Serializer unit tests
# ---------------------------------------------------------------------------


class TestSerializer:
    def test_serialize_creates_parent_dirs(self, catalog, backend, tmp_path):
        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        nested = tmp_path / "a" / "b" / "c.json"
        path = serialize_cohort_map(report.cohort_map, nested)
        assert path.exists()

    def test_deserialize_rejects_unknown_schema_version(self, catalog, backend, tmp_path):
        import json

        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        path = serialize_cohort_map(report.cohort_map, tmp_path / "m.json")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["schema_version"] = "99.9"
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(ValueError):
            deserialize_cohort_map(path)

    def test_serialized_file_is_valid_json_with_envelope_keys(
        self, catalog, backend, tmp_path
    ):
        import json

        report = generate_cohort_map(
            WORKED_TARGET_FILTER_SPEC, WORKED_TARGET_HORIZON, catalog, backend
        )
        path = serialize_cohort_map(report.cohort_map, tmp_path / "m.json")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" in envelope
        assert "cohort_map" in envelope
