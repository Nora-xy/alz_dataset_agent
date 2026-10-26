"""P3 tests: seven deterministic tools.

Covers per-tool:
  - At least one abstain path
  - Correct grounded path with evidence
  - Leakage flags fire correctly
  - Modality combo that collapses (suppressed)
  - Every returned number traces to a matching Evidence entry
"""

from __future__ import annotations

from typing import Any

import pytest

from alz_agent.backend.fixture_backend import FixtureDataBackend, build_fixture_backend
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
from alz_agent.tools.deterministic import (
    CatalogMatch,
    LeakageCheckResult,
    LeakageFlag,
    ModelingDesignResult,
    StatResult,
    VariableFamilyResult,
    VariableLookupResult,
    assess_longitudinal_feasibility,
    assess_modality_overlap,
    check_leakage,
    compute_cohort_size,
    compute_variable_stat,
    describe_variable_family,
    explore_catalog,
    lookup_variable,
    suggest_modeling_design,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_catalog() -> list[CatalogEntry]:
    """Minimal Space A + Space B catalog for P3 tests."""
    return [
        # DIAGNOSIS_DEFINING — Space A
        CatalogEntry(
            variable_id="NACC.NACCUDSD",
            concept="nacc_unified_diagnostic_summary",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCUDSD",
            release_version="nacc70",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            leakage_reasons=["NACC unified diagnostic summary — primary outcome label"],
        ),
        # CIRCULAR — Space A
        CatalogEntry(
            variable_id="NACC.CDRSUM",
            concept="cdr_sum_of_boxes",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="CDRSUM",
            release_version="nacc70",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.CIRCULAR],
            leakage_reasons=["CDR Sum of Boxes — highly collinear with AD/MCI outcome"],
        ),
        # HARMONIZATION_FIT_SCOPE — Space A
        CatalogEntry(
            variable_id="ADSP_PHC.Cognition.PHC_MEM",
            concept="phc_memory_composite",
            source_dataset=SourceDataset.ADSP_PHC,
            id_space=IdSpace.A,
            raw_code="PHC_MEM",
            release_version="R3",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.HARMONIZATION_FIT_SCOPE],
            leakage_reasons=["PHC Memory composite: GMM fit on full cohort"],
        ),
        # NONE (safe, cognition) — Space A
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
        # NONE (safe, mri) — Space A
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
        # NONE (safe, pet) — Space A  — very low prevalence in fixture → suppression tests
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
        # DIAGNOSIS_DEFINING — Space B
        CatalogEntry(
            variable_id="ADNI.DXCURREN",
            concept="adni_current_diagnosis",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="DXCURREN",
            release_version="ADNI",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            leakage_reasons=["ADNI current diagnosis — primary outcome label"],
        ),
    ]


@pytest.fixture()
def catalog() -> list[CatalogEntry]:
    return _make_catalog()


@pytest.fixture()
def backend() -> FixtureDataBackend:
    # Build with modality fractions that match the catalog modalities above.
    # pet fraction 0.04 on ~183 total ≈ 7–8 participants → below suppression threshold.
    return build_fixture_backend(
        n_normal=50,
        n_mci=40,
        n_dementia=30,
        n_other=20,
        n_mci_long=25,
        n_mci_short=8,
        n_single_visit=15,
        n_suppressed_group=5,
        modality_fractions={
            "cognition": 0.9,
            "vascular": 0.8,
            "mri": 0.15,
            "pet": 0.04,   # deliberately low → suppressed intersection
            "fluid_biomarker": 0.05,
            "neuropath": 0.0,
        },
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tool 1: lookup_variable
# ---------------------------------------------------------------------------


class TestLookupVariable:
    def test_found_by_variable_id(self, catalog):
        result = lookup_variable("NACC.MMSE30", catalog)
        assert result.grounded
        assert result.value is not None
        assert result.value.variable_id == "NACC.MMSE30"
        assert result.value.modality == "cognition"
        assert result.value.leakage_tags == ["NONE"]
        assert len(result.evidence) >= 1

    def test_found_by_variable_id_case_insensitive(self, catalog):
        result = lookup_variable("nacc.mmse30", catalog)
        assert result.grounded
        assert result.value.variable_id == "NACC.MMSE30"

    def test_found_by_concept(self, catalog):
        result = lookup_variable("mmse_total_score", catalog)
        assert result.grounded
        assert result.value.variable_id == "NACC.MMSE30"

    def test_found_by_raw_code(self, catalog):
        result = lookup_variable("MMSE30", catalog)
        assert result.grounded
        assert result.value.variable_id == "NACC.MMSE30"

    def test_abstains_when_not_found(self, catalog):
        result = lookup_variable("NONEXISTENT_VAR_XYZ", catalog)
        assert not result.grounded
        assert result.value is None
        assert result.abstain_reason is not None

    def test_evidence_references_variable_id(self, catalog):
        result = lookup_variable("NACC.MMSE30", catalog)
        ev = result.evidence[0]
        assert isinstance(ev, DataEvidence)
        assert "NACC.MMSE30" in ev.aggregate_ref

    def test_leakage_tags_and_reasons_populated(self, catalog):
        result = lookup_variable("NACC.CDRSUM", catalog)
        assert result.grounded
        assert "CIRCULAR" in result.value.leakage_tags
        assert len(result.value.leakage_reasons) >= 1
        assert "collinear" in result.value.leakage_reasons[0].lower()


class TestLookupVariableDescriptionSearch:
    """Tests for passes 4 (substring) and 5 (keyword) of lookup_variable."""

    @pytest.fixture()
    def enriched_catalog(self):
        """Catalog entries with description populated (simulates RDD enrichment)."""
        return [
            CatalogEntry(
                variable_id="NACC.NACCAGE",
                concept="age of subject at time of uds visit",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="NACCAGE",
                release_version="nacc70",
                modality=Modality.DEMOGRAPHIC,
                leakage_tags=[LeakageTag.NONE],
                description="Age of subject at time of UDS visit, computed from reported date of birth and visit date.",
            ),
            CatalogEntry(
                variable_id="NACC.SEX",
                concept="sex",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="SEX",
                release_version="nacc70",
                modality=Modality.DEMOGRAPHIC,
                leakage_tags=[LeakageTag.NONE],
                description="Sex of the subject.",
            ),
            CatalogEntry(
                variable_id="NACC.CVHATT",
                concept="heart attack/cardiac arrest",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="CVHATT",
                release_version="nacc70",
                modality=Modality.VASCULAR,
                leakage_tags=[LeakageTag.NONE],
                description="Heart attack or cardiac arrest history.",
                value_coding={"0": "Absent", "1": "Recent/Active", "2": "Remote/Inactive"},
            ),
        ]

    def test_pass4_substring_in_concept(self, enriched_catalog):
        # "age of subject" is a substring of the concept field
        result = lookup_variable("age of subject", enriched_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_pass4_substring_in_description(self, enriched_catalog):
        result = lookup_variable("computed from reported date of birth", enriched_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_pass5_keyword_match(self, enriched_catalog):
        # "age visit" — both words in concept/description
        result = lookup_variable("age visit", enriched_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_pass5_cardiac_keyword(self, enriched_catalog):
        result = lookup_variable("cardiac history", enriched_catalog)
        assert result.grounded
        assert result.value.raw_code == "CVHATT"

    def test_pass1_exact_still_wins(self, enriched_catalog):
        # Exact variable_id match should still work (pass 1)
        result = lookup_variable("NACC.SEX", enriched_catalog)
        assert result.grounded
        assert result.value.raw_code == "SEX"

    def test_pass2_raw_code_still_wins(self, enriched_catalog):
        result = lookup_variable("CVHATT", enriched_catalog)
        assert result.grounded
        assert result.value.raw_code == "CVHATT"

    def test_no_match_still_abstains(self, enriched_catalog):
        result = lookup_variable("tau pet suvr centiloid", enriched_catalog)
        assert not result.grounded

    def test_short_query_does_not_match_noise(self, enriched_catalog):
        # Single-character or 2-char queries don't trigger passes 4/5
        result = lookup_variable("ag", enriched_catalog)
        assert not result.grounded

    def test_value_coding_returned_when_enriched(self, enriched_catalog):
        result = lookup_variable("CVHATT", enriched_catalog)
        assert result.grounded
        assert result.value.value_coding.get("0") == "Absent"


class TestLookupVariableAliasMap:
    """Tests for Pass 0 alias resolution and short-token handling."""

    @pytest.fixture()
    def age_catalog(self):
        """Catalog with NACCAGE, SEX, EDUC and a distracting entry."""
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
                description="All-cause dementia diagnosis summary.",
            ),
            CatalogEntry(
                variable_id="NACC.NACCAGE",
                concept="subject's age at visit",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="NACCAGE",
                release_version="nacc70",
                modality=Modality.DEMOGRAPHIC,
                leakage_tags=[LeakageTag.NONE],
                leakage_reasons=[],
                description="Subject's age at visit. NACC derived variable.",
            ),
            CatalogEntry(
                variable_id="NACC.SEX",
                concept="sex",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="SEX",
                release_version="nacc70",
                modality=Modality.DEMOGRAPHIC,
                leakage_tags=[LeakageTag.NONE],
                leakage_reasons=[],
                description="Sex of the subject.",
            ),
            CatalogEntry(
                variable_id="NACC.EDUC",
                concept="education years",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="EDUC",
                release_version="nacc70",
                modality=Modality.DEMOGRAPHIC,
                leakage_tags=[LeakageTag.NONE],
                leakage_reasons=[],
                description="Years of education completed.",
            ),
        ]

    def test_alias_age_resolves_to_naccage(self, age_catalog):
        """'age' alias must resolve to NACCAGE even though NACCUDSD comes first."""
        result = lookup_variable("age", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_alias_age_at_visit_resolves_to_naccage(self, age_catalog):
        result = lookup_variable("age at visit", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_alias_subject_age_resolves_to_naccage(self, age_catalog):
        result = lookup_variable("subject age", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_alias_baseline_age_resolves_to_naccage(self, age_catalog):
        result = lookup_variable("baseline age", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_alias_age_when_diagnosed_resolves_to_naccage(self, age_catalog):
        result = lookup_variable("age when diagnosed", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_alias_age_at_diagnosis_resolves_to_naccage(self, age_catalog):
        result = lookup_variable("age at diagnosis", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_alias_sex_resolves_to_sex(self, age_catalog):
        result = lookup_variable("sex", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "SEX"

    def test_alias_gender_resolves_to_sex(self, age_catalog):
        result = lookup_variable("gender", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "SEX"

    def test_alias_education_resolves_to_educ(self, age_catalog):
        result = lookup_variable("education", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "EDUC"

    def test_short_token_age_not_filtered_in_pass5(self, age_catalog):
        """'age' (3 chars) must survive Pass 5 keyword filtering."""
        # Query that only works via keyword "age" in concept+description
        result = lookup_variable("participant age", age_catalog)
        assert result.grounded
        assert result.value.raw_code == "NACCAGE"

    def test_alias_missing_from_catalog_abstains_gracefully(self):
        """Alias for NACCAGE but catalog has no NACCAGE → abstain (not KeyError)."""
        empty_catalog: list[CatalogEntry] = []
        result = lookup_variable("age", empty_catalog)
        assert not result.grounded

    def test_alias_naccage_not_in_catalog_falls_through(self):
        """When alias target is absent, later passes may still find via substring."""
        catalog_no_naccage = [
            CatalogEntry(
                variable_id="NACC.AGEVAR",
                concept="age at baseline",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="AGEVAR",
                release_version="nacc70",
                modality=Modality.DEMOGRAPHIC,
                leakage_tags=[LeakageTag.NONE],
                leakage_reasons=[],
                description="Age at baseline visit.",
            ),
        ]
        # "age" alias targets NACCAGE which is absent; Pass 4 finds AGEVAR via substring
        result = lookup_variable("age", catalog_no_naccage)
        assert result.grounded
        assert result.value.raw_code == "AGEVAR"

    def test_debug_flag_does_not_raise(self, age_catalog):
        """debug=True must not raise even when match is found."""
        result = lookup_variable("age", age_catalog, debug=True)
        assert result.grounded

    def test_debug_flag_unresolved_does_not_raise(self, age_catalog):
        """debug=True must not raise when query is unresolved."""
        result = lookup_variable("NONEXISTENT_CONCEPT_XYZ", age_catalog, debug=True)
        assert not result.grounded


# ---------------------------------------------------------------------------
# Tool 2: compute_cohort_size
# ---------------------------------------------------------------------------


class TestComputeCohortSize:
    def test_grounded_with_mci_filter(self, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = compute_cohort_size(spec, backend)
        assert result.grounded
        assert isinstance(result.value, int)
        assert result.value >= 10
        assert len(result.evidence) >= 1

    def test_evidence_contains_count(self, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = compute_cohort_size(spec, backend)
        ev = result.evidence[0]
        assert isinstance(ev, DataEvidence)
        assert str(result.value) in ev.aggregate_ref

    def test_abstains_on_suppressed_small_group(self, backend):
        # vnum [15,16] filter returns exactly 5 MCI participants → suppressed
        spec = FilterSpec(baseline_diagnosis=["mci"], visit_window=(15, 16))
        result = compute_cohort_size(spec, backend)
        assert not result.grounded
        assert result.abstain_reason is not None
        assert "suppressed" in result.abstain_reason.lower()

    def test_abstains_on_unknown_label(self, backend):
        spec = FilterSpec(baseline_diagnosis=["unknown_dx"])
        result = compute_cohort_size(spec, backend)
        assert not result.grounded

    def test_no_filter_returns_all(self, backend):
        spec = FilterSpec()
        result = compute_cohort_size(spec, backend)
        assert result.grounded
        assert result.value >= 100


# ---------------------------------------------------------------------------
# Tool 3: assess_modality_overlap
# ---------------------------------------------------------------------------


class TestAssessModalityOverlap:
    def test_grounded_single_modality(self, backend, catalog):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = assess_modality_overlap(["cognition"], spec, backend, catalog)
        assert result.grounded
        assert "cognition" in result.value
        assert len(result.evidence) >= 1

    def test_grounded_two_modalities_returns_all_combos(self, backend, catalog):
        spec = FilterSpec()
        result = assess_modality_overlap(["cognition", "mri"], spec, backend, catalog)
        assert result.grounded
        val = result.value
        # Should have single-modality counts and the intersection
        assert "cognition" in val
        assert "mri" in val
        assert "cognition,mri" in val

    def test_combo_collapses_pet_suppressed(self, backend, catalog):
        # pet fraction 0.04 → ~7 participants → suppressed intersection
        spec = FilterSpec()
        result = assess_modality_overlap(["cognition", "pet"], spec, backend, catalog)
        assert result.grounded
        val = result.value
        # The pet or cognition+pet combo count should be suppressed (None)
        # pet alone will be suppressed; cognition alone will not
        assert val["cognition"] is not None
        # pet alone should be suppressed (count below threshold)
        assert val["pet"] is None

    def test_abstains_on_unknown_modality_name(self, backend, catalog):
        spec = FilterSpec()
        result = assess_modality_overlap(["INVALID_MOD"], spec, backend, catalog)
        assert not result.grounded
        assert "Unknown modality" in result.abstain_reason

    def test_abstains_when_modality_absent_from_catalog(self, backend, catalog):
        # "neuropath" is not in the Space A catalog entries above
        spec = FilterSpec()
        result = assess_modality_overlap(["neuropath"], spec, backend, catalog)
        assert not result.grounded
        assert "absent from catalog" in result.abstain_reason

    def test_abstains_when_modality_absent_from_catalog_space_b(self, backend, catalog):
        # mri IS in the Space A catalog but NOT in Space B catalog entries
        spec = FilterSpec(id_space=IdSpace.B)
        result = assess_modality_overlap(["mri"], spec, backend, catalog)
        assert not result.grounded
        assert "absent from catalog" in result.abstain_reason

    def test_evidence_non_empty_on_grounded(self, backend, catalog):
        spec = FilterSpec()
        result = assess_modality_overlap(["cognition"], spec, backend, catalog)
        assert result.grounded
        assert len(result.evidence) >= 1


# ---------------------------------------------------------------------------
# Tool 4: assess_longitudinal_feasibility
# ---------------------------------------------------------------------------


class TestAssessLongitudinalFeasibility:
    def test_grounded_mci_3yr_horizon(self, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = assess_longitudinal_feasibility(3, spec, backend)
        assert result.grounded
        val = result.value
        assert "sufficient" in val
        assert "censored" in val
        assert "insufficient" in val
        assert len(result.evidence) >= 1

    def test_sufficient_count_positive(self, backend):
        # n_mci_long=25 participants have >=3yr follow-up
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = assess_longitudinal_feasibility(3, spec, backend)
        assert result.grounded
        # sufficient should be a positive count (25 long + some mci_ad with multiple visits)
        assert result.value["sufficient"] is not None
        assert result.value["sufficient"] >= 10

    def test_abstains_on_zero_horizon(self, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = assess_longitudinal_feasibility(0, spec, backend)
        assert not result.grounded
        assert result.abstain_reason is not None

    def test_abstains_on_negative_horizon(self, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = assess_longitudinal_feasibility(-1, spec, backend)
        assert not result.grounded

    def test_evidence_references_horizon(self, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = assess_longitudinal_feasibility(3, spec, backend)
        ev = result.evidence[0]
        assert isinstance(ev, DataEvidence)
        assert "3" in str(ev.query_spec)  # horizon in query_spec


# ---------------------------------------------------------------------------
# Tool 5: check_leakage
# ---------------------------------------------------------------------------


class TestCheckLeakage:
    def test_diagnosis_defining_flagged(self, catalog):
        result = check_leakage(["NACC.NACCUDSD"], "ad_conversion", catalog)
        assert result.grounded
        flagged = result.value.flagged
        assert len(flagged) == 1
        assert flagged[0].variable_id == "NACC.NACCUDSD"
        assert "DIAGNOSIS_DEFINING" in flagged[0].tags

    def test_circular_flagged(self, catalog):
        result = check_leakage(["NACC.CDRSUM"], "ad_conversion", catalog)
        assert result.grounded
        flagged = result.value.flagged
        assert len(flagged) == 1
        assert "CIRCULAR" in flagged[0].tags
        assert len(flagged[0].reasons) >= 1

    def test_harmonization_fit_scope_flagged(self, catalog):
        result = check_leakage(["ADSP_PHC.Cognition.PHC_MEM"], "ad_conversion", catalog)
        assert result.grounded
        flagged = result.value.flagged
        assert len(flagged) == 1
        assert "HARMONIZATION_FIT_SCOPE" in flagged[0].tags

    def test_none_tag_goes_to_clean(self, catalog):
        result = check_leakage(["NACC.MMSE30"], "ad_conversion", catalog)
        assert result.grounded
        assert result.value.flagged == []
        assert "NACC.MMSE30" in result.value.clean

    def test_unknown_variable_goes_to_unknown(self, catalog):
        result = check_leakage(["GHOST.VAR_NOT_IN_CATALOG"], "ad_conversion", catalog)
        assert result.grounded
        assert "GHOST.VAR_NOT_IN_CATALOG" in result.value.unknown
        assert result.value.flagged == []

    def test_mixed_set(self, catalog):
        features = [
            "NACC.NACCUDSD",       # DIAGNOSIS_DEFINING → flagged
            "NACC.CDRSUM",         # CIRCULAR → flagged
            "NACC.MMSE30",         # NONE → clean
            "GHOST.MISSING",       # absent → unknown
        ]
        result = check_leakage(features, "ad_conversion_3yr", catalog)
        assert result.grounded
        val = result.value
        assert len(val.flagged) == 2
        assert len(val.clean) == 1
        assert len(val.unknown) == 1
        assert val.label_def == "ad_conversion_3yr"

    def test_evidence_counts_match_result(self, catalog):
        features = ["NACC.NACCUDSD", "NACC.MMSE30", "GHOST.X"]
        result = check_leakage(features, "test_label", catalog)
        ev = result.evidence[0]
        assert isinstance(ev, DataEvidence)
        # Evidence aggregate_ref references flagged/clean/unknown counts
        assert "1 flagged" in ev.aggregate_ref
        assert "1 clean" in ev.aggregate_ref
        assert "1 unknown" in ev.aggregate_ref

    def test_empty_feature_set(self, catalog):
        result = check_leakage([], "ad_conversion", catalog)
        assert result.grounded
        assert result.value.flagged == []
        assert result.value.clean == []
        assert result.value.unknown == []

    def test_always_grounded(self, catalog):
        # check_leakage must never abstain
        result = check_leakage(["NACC.NACCUDSD", "GHOST.X"], "any_label", catalog)
        assert result.grounded


# ---------------------------------------------------------------------------
# Tool 6: suggest_modeling_design
# ---------------------------------------------------------------------------


class TestSuggestModelingDesign:
    def test_grounded_basic(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        assert result.grounded
        val = result.value
        assert isinstance(val, ModelingDesignResult)
        assert val.horizon_years == 3
        assert val.id_space == "A"
        assert len(result.evidence) >= 1

    def test_leakage_blacklist_excludes_flagged(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        assert result.grounded
        val = result.value
        # DIAGNOSIS_DEFINING, CIRCULAR, HARMONIZATION_FIT_SCOPE vars must be blacklisted
        assert "NACC.NACCUDSD" in val.leakage_blacklist
        assert "NACC.CDRSUM" in val.leakage_blacklist
        assert "ADSP_PHC.Cognition.PHC_MEM" in val.leakage_blacklist

    def test_feature_whitelist_excludes_leaky(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        val = result.value
        for var_id in val.leakage_blacklist:
            assert var_id not in val.feature_whitelist

    def test_feature_whitelist_includes_safe_vars(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        val = result.value
        assert "NACC.MMSE30" in val.feature_whitelist

    def test_eligible_n_is_int(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        assert result.grounded
        assert isinstance(result.value.eligible_n, int)
        assert result.value.eligible_n >= 10

    def test_feasibility_populated(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        feas = result.value.feasibility
        assert "sufficient" in feas
        assert "censored" in feas

    def test_index_visit_def_contains_diagnosis(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        assert "mci" in result.value.index_visit_def.lower()

    def test_outcome_label_def_references_horizon(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        assert "3" in result.value.outcome_label_def

    def test_abstains_on_zero_horizon(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 0, catalog, backend)
        assert not result.grounded
        assert result.abstain_reason is not None

    def test_abstains_on_negative_horizon(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, -5, catalog, backend)
        assert not result.grounded

    def test_abstains_on_invalid_filter(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["not_a_real_diagnosis"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        assert not result.grounded
        assert result.abstain_reason is not None

    def test_space_b_filter_uses_space_b_catalog(self, catalog, backend):
        # Space B backend won't match Space A participants → empty / suppressed
        from alz_agent.backend.fixture_backend import FixtureDataBackend
        b_backend = build_fixture_backend(id_space=IdSpace.B, n_mci=50, n_normal=30)
        spec = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.B)
        result = suggest_modeling_design(spec, 3, catalog, b_backend)
        assert result.grounded
        # Only Space B catalog entries should appear in whitelist/blacklist
        for var_id in result.value.feature_whitelist + result.value.leakage_blacklist:
            entry = next(e for e in catalog if e.variable_id == var_id)
            assert entry.id_space == IdSpace.B

    def test_evidence_non_empty(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        assert len(result.evidence) >= 2  # at least size + leakage evidence

    def test_eligible_n_is_in_evidence(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = suggest_modeling_design(spec, 3, catalog, backend)
        eligible_n = result.value.eligible_n
        assert eligible_n is not None
        # The count must appear somewhere in the evidence aggregate refs
        ev_refs = [e.aggregate_ref for e in result.evidence if isinstance(e, DataEvidence)]
        assert any(str(eligible_n) in ref for ref in ev_refs)


# ---------------------------------------------------------------------------
# Cross-tool: every number has evidence
# ---------------------------------------------------------------------------


class TestEveryNumberHasEvidence:
    """Verify that every numeric value returned by a grounded tool result
    appears in at least one Evidence entry (the number-tracing guarantee)."""

    def _all_int_values(self, result: ToolResult) -> list[int]:
        if not result.grounded or result.value is None:
            return []
        val = result.value
        if isinstance(val, int):
            return [val]
        if isinstance(val, dict):
            return [v for v in val.values() if isinstance(v, int)]
        return []

    def _ev_text(self, result: ToolResult) -> str:
        return " ".join(
            e.aggregate_ref for e in result.evidence if isinstance(e, DataEvidence)
        )

    def test_compute_cohort_size_number_in_evidence(self, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = compute_cohort_size(spec, backend)
        assert result.grounded
        assert str(result.value) in self._ev_text(result)

    def test_assess_longitudinal_feasibility_numbers_in_evidence(self, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        result = assess_longitudinal_feasibility(3, spec, backend)
        assert result.grounded
        ev_text = self._ev_text(result)
        for v in self._all_int_values(result):
            assert str(v) in ev_text

    def test_check_leakage_counts_in_evidence(self, catalog):
        features = ["NACC.NACCUDSD", "NACC.MMSE30", "GHOST.X"]
        result = check_leakage(features, "test", catalog)
        assert result.grounded
        ev_text = self._ev_text(result)
        # flagged=1, clean=1, unknown=1 must all appear
        assert "1 flagged" in ev_text
        assert "1 clean" in ev_text
        assert "1 unknown" in ev_text


# ---------------------------------------------------------------------------
# Tool 7: compute_variable_stat
# ---------------------------------------------------------------------------


def _make_age_catalog() -> list[CatalogEntry]:
    """Minimal catalog containing NACCAGE for stat tests."""
    return [
        CatalogEntry(
            variable_id="NACC.NACCAGE",
            concept="age of subject at time of uds visit",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCAGE",
            release_version="nacc70",
            modality=Modality.DEMOGRAPHIC,
            leakage_tags=[LeakageTag.NONE],
            description="Age of subject at time of UDS visit.",
        ),
    ]


class TestComputeVariableStat:
    @pytest.fixture()
    def age_catalog(self) -> list[CatalogEntry]:
        return _make_age_catalog()

    @pytest.fixture()
    def backend_with_ages(self) -> FixtureDataBackend:
        return build_fixture_backend(
            n_dementia=40,
            n_dementia_ad=20,
            n_mci=30,
            n_normal=20,
            n_other=10,
            n_mci_long=15,
            n_mci_short=5,
            n_single_visit=10,
            n_suppressed_group=5,
            n_mci_ad=10,
            seed=99,
        )

    def test_grounded_mean_age_dementia(self, age_catalog, backend_with_ages):
        spec = FilterSpec(baseline_diagnosis=["dementia"])
        result = compute_variable_stat("NACC.NACCAGE", "mean", spec, backend_with_ages)
        assert result.grounded
        assert isinstance(result.value, StatResult)
        assert result.value.stat_type == "mean"
        assert result.value.n > 0
        assert isinstance(result.value.value, float)
        assert 40.0 <= result.value.value <= 100.0  # plausible age range

    def test_grounded_median_age(self, age_catalog, backend_with_ages):
        spec = FilterSpec(baseline_diagnosis=["dementia"])
        result = compute_variable_stat("NACC.NACCAGE", "median", spec, backend_with_ages)
        assert result.grounded
        assert result.value.stat_type == "median"

    def test_grounded_ad_etiology_filter(self, age_catalog, backend_with_ages):
        spec = FilterSpec(baseline_diagnosis=["dementia"], baseline_etiology=["ad"])
        result = compute_variable_stat("NACC.NACCAGE", "mean", spec, backend_with_ages)
        assert result.grounded
        assert result.value.n > 0

    def test_abstains_on_unknown_variable(self, backend_with_ages):
        spec = FilterSpec(baseline_diagnosis=["dementia"])
        result = compute_variable_stat("NACC.GHOST_VAR_99", "mean", spec, backend_with_ages)
        assert not result.grounded
        assert result.abstain_reason is not None

    def test_abstains_on_invalid_stat_type(self, backend_with_ages):
        spec = FilterSpec(baseline_diagnosis=["dementia"])
        result = compute_variable_stat("NACC.NACCAGE", "variance", spec, backend_with_ages)
        assert not result.grounded
        assert "stat_type" in (result.abstain_reason or "").lower()

    def test_abstains_on_space_b(self, backend_with_ages):
        spec = FilterSpec(baseline_diagnosis=["dementia"], id_space=IdSpace.B)
        result = compute_variable_stat("NACC.NACCAGE", "mean", spec, backend_with_ages)
        assert not result.grounded

    def test_evidence_contains_stat_value_and_n(self, backend_with_ages):
        spec = FilterSpec(baseline_diagnosis=["dementia"])
        result = compute_variable_stat("NACC.NACCAGE", "mean", spec, backend_with_ages)
        assert result.grounded
        ev_text = " ".join(
            e.aggregate_ref for e in result.evidence if isinstance(e, DataEvidence)
        )
        # stat value and n must appear in evidence
        assert str(result.value.n) in ev_text
        assert str(result.value.value) in ev_text

    def test_hallucination_guard_passes_for_stat(self, backend_with_ages):
        from alz_agent.agent.orchestrator import _check_hallucination
        spec = FilterSpec(baseline_diagnosis=["dementia"])
        result = compute_variable_stat("NACC.NACCAGE", "mean", spec, backend_with_ages)
        assert result.grounded
        answer = (
            f"The mean age is {result.value.value} years "
            f"(n={result.value.n} participants)."
        )
        assert _check_hallucination(answer, [result])


# ---------------------------------------------------------------------------
# Tool 10: explore_catalog / Tool 11: describe_variable_family
# ---------------------------------------------------------------------------


def _make_biomarker_catalog() -> list[CatalogEntry]:
    """Space-B biomarker catalog with family/variant tags, mirroring the real
    ADSP_PHC_CSF/PLASMA header-fallback parsing (family='ptau217'/'ptau181').
    """
    return [
        CatalogEntry(
            variable_id="ADNI.ADSP_PHC_PLASMA.PHC_pTau217",
            concept="phc_ptau217",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="PHC_pTau217",
            release_version="ADNI_Jun2026",
            description="Harmonized Z-score for pTau217 biomarker levels",
            modality=Modality.FLUID_BIOMARKER,
            family="ptau217",
            variant="plasma_harmonized",
        ),
        CatalogEntry(
            variable_id="ADNI.ADSP_PHC_PLASMA.pTau217_RAW",
            concept="ptau217_raw",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="pTau217_RAW",
            release_version="ADNI_Jun2026",
            description="Raw pTau217 biomarker levels",
            modality=Modality.FLUID_BIOMARKER,
            family="ptau217",
            variant="plasma_raw",
        ),
        CatalogEntry(
            variable_id="ADNI.ADSP_PHC_CSF.PHC_pTau181",
            concept="phc_ptau181",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="PHC_pTau181",
            release_version="ADNI_Jun2026",
            description="Harmonized Z-score for pTau181 biomarker levels",
            modality=Modality.FLUID_BIOMARKER,
            family="ptau181",
            variant="csf_harmonized",
        ),
    ]


class TestExploreCatalog:
    @pytest.fixture()
    def biomarker_catalog(self) -> list[CatalogEntry]:
        return _make_biomarker_catalog()

    def test_finds_all_ptau217_variants(self, biomarker_catalog):
        result = explore_catalog("ptau217", biomarker_catalog)
        assert result.grounded
        assert {m.variable_id for m in result.value} == {
            "ADNI.ADSP_PHC_PLASMA.PHC_pTau217",
            "ADNI.ADSP_PHC_PLASMA.pTau217_RAW",
        }

    def test_does_not_match_ptau181_when_searching_ptau217(self, biomarker_catalog):
        result = explore_catalog("ptau217", biomarker_catalog)
        assert result.grounded
        assert all("181" not in m.variable_id for m in result.value)

    def test_filters_by_id_space(self, biomarker_catalog):
        # Catalog is entirely Space B — searching within Space A finds nothing.
        result = explore_catalog("ptau", biomarker_catalog, id_space=IdSpace.A)
        assert not result.grounded

    def test_abstains_on_no_match(self, biomarker_catalog):
        result = explore_catalog("nonexistent_keyword_xyz", biomarker_catalog)
        assert not result.grounded
        assert result.abstain_reason is not None

    def test_abstains_on_empty_keyword(self, biomarker_catalog):
        result = explore_catalog("   ", biomarker_catalog)
        assert not result.grounded

    def test_result_includes_family_and_variant(self, biomarker_catalog):
        result = explore_catalog("PHC_pTau217", biomarker_catalog)
        assert result.grounded
        match = result.value[0]
        assert match.family == "ptau217"
        assert match.variant == "plasma_harmonized"

    def test_evidence_present(self, biomarker_catalog):
        result = explore_catalog("ptau217", biomarker_catalog)
        assert result.grounded
        assert len(result.evidence) >= 1


class TestDescribeVariableFamily:
    @pytest.fixture()
    def biomarker_catalog(self) -> list[CatalogEntry]:
        return _make_biomarker_catalog()

    @pytest.fixture()
    def biomarker_backend(self) -> FixtureDataBackend:
        return build_fixture_backend(
            n_normal=50, n_mci=40, n_dementia=30, n_other=20,
            n_mci_long=25, n_mci_short=8, n_single_visit=15,
            n_suppressed_group=5,
            id_space=IdSpace.B,
            seed=42,
        )

    def test_grounded_ptau217_has_both_variants(self, biomarker_catalog, biomarker_backend):
        result = describe_variable_family(
            "ptau217", IdSpace.B, biomarker_catalog, biomarker_backend
        )
        assert result.grounded
        assert isinstance(result.value, VariableFamilyResult)
        variants = {v.variant for v in result.value.variants}
        assert variants == {"plasma_harmonized", "plasma_raw"}
        for v in result.value.variants:
            assert v.n is None or v.n > 0

    def test_abstains_on_unknown_family(self, biomarker_catalog, biomarker_backend):
        result = describe_variable_family(
            "nonexistent_family", IdSpace.B, biomarker_catalog, biomarker_backend
        )
        assert not result.grounded
        assert "not available" in result.abstain_reason.lower()

    def test_abstains_when_backend_cannot_ground_family(self, biomarker_catalog, biomarker_backend):
        """Catalog has ptau181 (CSF) but the fixture backend only synthesizes
        ptau217 values — variant_subject_counts abstains, and the tool must
        propagate that rather than fabricate a count.
        """
        result = describe_variable_family(
            "ptau181", IdSpace.B, biomarker_catalog, biomarker_backend
        )
        assert not result.grounded

    def test_abstains_on_id_space_mismatch(self, biomarker_catalog, biomarker_backend):
        result = describe_variable_family(
            "ptau217", IdSpace.B, biomarker_catalog, biomarker_backend,
            filter_spec=FilterSpec(id_space=IdSpace.A),
        )
        assert not result.grounded
        assert "mismatch" in result.abstain_reason.lower()

    def test_evidence_present_on_grounded(self, biomarker_catalog, biomarker_backend):
        result = describe_variable_family(
            "ptau217", IdSpace.B, biomarker_catalog, biomarker_backend
        )
        assert result.grounded
        assert len(result.evidence) >= 1
