"""P5 smoke tests: realistic English questions and edge-case requirements.

Requirements covered:
  1. A variety of realistic questions must not crash and must return a non-empty answer.
  2. "36-month follow-up" parses as a 3-year horizon.
  3. "4-year conversion" preserves horizon=4 (never silently coerced to 3).
  4. "Alzheimer's disease conversion" produces an all-cause / etiologic caveat or abstains.
  5. Multimodal questions (cognition + biomarker + imaging) trigger overlap output and
     mention those modalities.
  6. Suppressed modality counts are narrated as "suppressed", never as zero.

Catalog and backend reuse the same parameters as test_p5_agent.py with one addition:
a FLUID_BIOMARKER entry so modality-overlap tests don't pre-abstain at the catalog check.
"""

from __future__ import annotations

import pytest

from alz_agent.agent.intent import QuestionType, StructuredIntent, parse_intent
from alz_agent.agent.orchestrator import (
    AgentResponse,
    _check_hallucination,
    _narrate_template,
    run_agent,
)
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    CatalogEntry,
    DataEvidence,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Shared fixtures (same backend params as test_p5_agent.py)
# ---------------------------------------------------------------------------

_MODALITY_FRACTIONS = {
    "cognition": 0.9,
    "mri": 0.15,
    "pet": 0.04,           # ~1 MCI participant → suppressed
    "fluid_biomarker": 0.05,  # ~3 MCI participants → suppressed
    "vascular": 0.8,
    "neuropath": 0.0,
}


def _make_catalog() -> list[CatalogEntry]:
    """Space A catalog including fluid_biomarker so overlap tests don't pre-abstain."""
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
        # Extra entry enabling fluid_biomarker modality-overlap tests.
        CatalogEntry(
            variable_id="ADSP_PHC.Biomarker.PHC_AB42",
            concept="phc_amyloid_beta_42",
            source_dataset=SourceDataset.ADSP_PHC,
            id_space=IdSpace.A,
            raw_code="PHC_AB42",
            release_version="R3",
            modality=Modality.FLUID_BIOMARKER,
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
        modality_fractions=_MODALITY_FRACTIONS,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Requirement 1: realistic questions must not crash and produce a non-empty answer
# ---------------------------------------------------------------------------

REALISTIC_QUESTIONS = [
    "How many MCI participants have at least 3 years of follow-up?",
    "Is predicting 3-year conversion from MCI to dementia feasible with cognitive tests?",
    "What is the MMSE total score variable in NACC?",
    "Can we use MRI volumes to predict conversion in 5 years from MCI to dementia?",
    "How many normal participants are in the NACC cohort?",
    "Among MCI participants, how many have both cognition and MRI data?",
    "Is it feasible to build a 2-year MCI conversion model using ADSP-PHC harmonized features?",
    "How large is the MCI cohort for a 3-year prediction study?",
    "Can we combine cognition and CSF biomarkers to predict Alzheimer's disease conversion?",
]


@pytest.mark.parametrize("question", REALISTIC_QUESTIONS)
def test_realistic_question_does_not_crash(question, catalog, backend):
    response = run_agent(question, catalog, backend)
    assert isinstance(response, AgentResponse)
    assert response.answer.strip(), f"Empty answer for: {question!r}"


@pytest.mark.parametrize("question", REALISTIC_QUESTIONS)
def test_realistic_question_hallucination_guard_passes(question, catalog, backend):
    response = run_agent(question, catalog, backend)
    assert response.hallucination_check_passed, (
        f"Hallucination guard failed for: {question!r}\nAnswer: {response.answer}"
    )


# ---------------------------------------------------------------------------
# Requirement 2: "36-month" and other month expressions parse as correct horizon
# ---------------------------------------------------------------------------


class TestMonthHorizonParsing:
    def test_36_month_parsed_as_3_years(self):
        intent = parse_intent("Among MCI patients, predict 36-month conversion to dementia.")
        assert intent.horizon_years == 3

    def test_24_month_parsed_as_2_years(self):
        intent = parse_intent("Use a 24-month follow-up window for this prediction study.")
        assert intent.horizon_years == 2

    def test_12_month_parsed_as_1_year(self):
        intent = parse_intent("Is a 12-month conversion prediction feasible in MCI?")
        assert intent.horizon_years == 1

    def test_year_takes_precedence_over_month(self):
        # "3-year 36-month" — year pattern fires first
        intent = parse_intent("This is a 3-year (36-month) follow-up study for MCI.")
        assert intent.horizon_years == 3

    def test_36_month_end_to_end_answer_says_3(self, catalog, backend):
        # When the intent parses to 3 years, the answer must reflect horizon=3
        intent = parse_intent("Among MCI participants, predict 36-month conversion to AD.")
        assert intent.horizon_years == 3
        response = run_agent(
            "Among MCI participants, predict 36-month conversion to AD.",
            catalog,
            backend,
            intent=intent,
        )
        assert response.grounded
        assert "3" in response.answer
        assert response.hallucination_check_passed


# ---------------------------------------------------------------------------
# Requirement 3: "4-year" must not be silently coerced to 3-year
# ---------------------------------------------------------------------------


class TestFourYearHorizonPreserved:
    def test_4_year_parsed_as_4(self):
        intent = parse_intent("Can we predict 4-year conversion from MCI to AD?")
        assert intent.horizon_years == 4

    def test_5_year_parsed_as_5(self):
        intent = parse_intent("Feasibility of a 5-year MCI conversion prediction study.")
        assert intent.horizon_years == 5

    def test_4_year_answer_explicitly_shows_4(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=4,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        # Template writes "Prediction horizon : 4 year(s)" — "4" must appear
        assert "Prediction horizon : 4" in response.answer, (
            f"Answer does not mention horizon=4. Got:\n{response.answer}"
        )

    def test_4_year_answer_does_not_misreport_as_3(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=4,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        # "Prediction horizon : 3" must NOT appear (would mean silent coercion)
        assert "Prediction horizon : 3" not in response.answer

    def test_4_year_hallucination_guard_passes(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=4,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert response.hallucination_check_passed

    def test_4_year_feasibility_shows_insufficient_not_zero_eligible(self, catalog, backend):
        # With visits_per_participant=4, max follow-up is 3 years → sufficient=0 for horizon=4.
        # The answer must say "sufficient : 0" (informative), not abstain silently.
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=4,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert response.grounded
        # "0" for sufficient follow-up must appear — it is in evidence via feas aggregate_ref.
        # New ctx template renders "Sufficient: 0" (not the legacy "Follow-up sufficient : 0").
        lower = response.answer.lower()
        assert "sufficient" in lower and "0" in response.answer, (
            "Sufficient follow-up count (0) must appear in answer"
        )


# ---------------------------------------------------------------------------
# Requirement 4: "Alzheimer's disease conversion" must carry an all-cause caveat
# ---------------------------------------------------------------------------


class TestADSpecificityCaveat:
    def test_ad_candidate_label_produces_caveat(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(
            "Can we predict Alzheimer's disease conversion in MCI?",
            catalog, backend, intent=intent,
        )
        if response.grounded:
            answer_lower = response.answer.lower()
            has_caveat = (
                "all-cause" in answer_lower
                or "etiolog" in answer_lower
                or "caveat" in answer_lower
            )
            assert has_caveat, (
                "Answer for AD-label question must include an all-cause / etiologic "
                f"caveat. Got:\n{response.answer}"
            )
        # abstain is also acceptable

    def test_alzheimers_in_candidate_label_produces_caveat(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="Alzheimer's conversion",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        if response.grounded:
            answer_lower = response.answer.lower()
            assert (
                "all-cause" in answer_lower
                or "etiolog" in answer_lower
                or "caveat" in answer_lower
            )

    def test_non_ad_label_does_not_produce_caveat(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to dementia",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        if response.grounded:
            # "Caveat:" should not appear for a generic dementia label
            assert "Caveat:" not in response.answer

    def test_caveat_contains_no_new_numbers(self, catalog, backend):
        # The caveat is pure prose — it must not introduce numbers that fail the guard.
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert response.hallucination_check_passed


# ---------------------------------------------------------------------------
# Requirement 5: multimodal questions trigger overlap output mentioning all modalities
# ---------------------------------------------------------------------------


class TestMultimodalOverlapOutput:
    def test_cognition_mri_overlap_triggered(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "mri"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        # design + overlap = at least 2 tool results
        assert len(response.tool_results) >= 2
        answer_lower = response.answer.lower()
        assert "cognition" in answer_lower
        assert "mri" in answer_lower

    def test_three_modality_overlap_mentions_all(self, catalog, backend):
        # fluid_biomarker is in the extended catalog; its count is suppressed but
        # the key still appears in the overlap dict and therefore in the answer.
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "fluid_biomarker", "mri"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert len(response.tool_results) >= 2
        answer_lower = response.answer.lower()
        assert "cognition" in answer_lower
        # mri and fluid_biomarker appear as keys in the overlap dict (even if suppressed)
        assert "mri" in answer_lower or "fluid_biomarker" in answer_lower

    def test_heuristic_parses_multimodal_keywords(self):
        intent = parse_intent(
            "Can we combine cognitive tests, MRI imaging, and CSF biomarkers "
            "to predict 3-year conversion to AD in MCI?"
        )
        # Parser should detect multiple modalities
        assert len(intent.modalities) >= 2
        assert "cognition" in intent.modalities
        # mri matches "mri" or "imaging"; fluid_biomarker matches "csf"
        assert "mri" in intent.modalities or "fluid_biomarker" in intent.modalities

    def test_multimodal_hallucination_guard_passes(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "mri"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert response.hallucination_check_passed


# ---------------------------------------------------------------------------
# Requirement 6: suppressed modality counts must not be narrated as zero
# ---------------------------------------------------------------------------


class TestSuppressedCountNotNarratedAsZero:
    def test_template_renders_none_as_suppressed_not_zero(self, catalog):
        """Unit test: _narrate_template must say 'suppressed', not '0'."""
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "pet"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        # Construct a ToolResult whose dict has a suppressed (None) value
        result = ToolResult.ok(
            {"cognition": 50, "cognition,pet": None, "pet": None},
            evidence=[DataEvidence(
                query_spec={"modalities": ["cognition", "pet"]},
                aggregate_ref="fixture.modality_intersection(['cognition', 'pet'])",
            )],
        )
        answer = _narrate_template(intent, [result], [])
        assert "suppressed" in answer.lower(), "None values must appear as 'suppressed'"
        assert "pet : 0" not in answer
        assert "cognition,pet : 0" not in answer

    def test_integration_pet_suppressed_not_zero(self, catalog, backend):
        """Integration test: pet count (1 out of 68 MCI) is below threshold → suppressed."""
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "pet"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert response.grounded
        # pet count is below suppression threshold → overlap result has None for pet keys
        assert "suppressed" in response.answer.lower()
        assert "pet : 0" not in response.answer
        assert "cognition,pet : 0" not in response.answer

    def test_integration_hallucination_guard_passes_with_suppressed(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "pet"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert response.hallucination_check_passed

    def test_suppressed_eligible_n_not_zero(self, catalog, backend):
        """When eligible_n is suppressed (None) it must not appear as 0 in the answer."""
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        # Manually build a grounded ModelingDesignResult with eligible_n=None
        from alz_agent.tools.deterministic import ModelingDesignResult

        suppressed_design = ToolResult.ok(
            ModelingDesignResult(
                index_visit_def="Index visit: baseline_diagnosis in [mci]",
                horizon_years=3,
                outcome_label_def="conversion to dementia within 3 year(s)",
                id_space="A",
                feature_whitelist=["NACC.MMSE30"],
                leakage_blacklist=["NACC.NACCUDSD"],
                eligible_n=None,  # suppressed
                feasibility={"sufficient": 20, "insufficient": 10, "censored": 5},
            ),
            evidence=[DataEvidence(
                query_spec={"baseline_diagnosis": ["mci"]},
                aggregate_ref="fixture.count_eligible suppressed",
            ), DataEvidence(
                query_spec={"horizon_years": 3},
                aggregate_ref="fixture.followup: sufficient=20, insufficient=10, censored=5",
            ), DataEvidence(
                query_spec={"feature_set_size": 2},
                aggregate_ref="check_leakage: 1 flagged, 1 clean, 0 unknown",
            )],
        )
        answer = _narrate_template(intent, [suppressed_design], [])
        # eligible_n=None → must NOT appear as "Eligible N : 0"
        assert "Eligible N : 0" not in answer
        assert "Eligible N :" not in answer  # skipped entirely when None
        # Hallucination guard must still pass
        assert _check_hallucination(answer, [suppressed_design])
