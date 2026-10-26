"""P5 tests: intent parser + orchestrator + hallucination guard.

Integration test target:
  "Among MCI participants at an index visit, can we predict 3-year conversion
   to AD using only pre-index information, and how many subjects survive that
   definition?"

All tests run on FixtureDataBackend — no real data or API key required.
"""

from __future__ import annotations

import re

import pytest

from alz_agent.agent.intent import QuestionType, StructuredIntent, parse_intent
from alz_agent.agent.orchestrator import (
    AgentResponse,
    _check_hallucination,
    _collect_evidence_numbers,
    _narrate_template,
    run_agent,
)
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    CatalogEntry,
    DataEvidence,
    FilterSpec,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
    ToolResult,
)
from alz_agent.tools.deterministic import (
    compute_cohort_size,
    suggest_modeling_design,
)

# ---------------------------------------------------------------------------
# Worked-target question (from AGENTS.md prompt)
# ---------------------------------------------------------------------------

WORKED_TARGET = (
    "Among MCI participants at an index visit, can we predict 3-year conversion "
    "to AD using only pre-index information, and how many subjects survive that "
    "definition?"
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_catalog() -> list[CatalogEntry]:
    """Minimal Space A catalog covering the worked target scenario."""
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


# ---------------------------------------------------------------------------
# Intent parser tests
# ---------------------------------------------------------------------------


class TestParseIntentHeuristic:
    def test_worked_target_population(self):
        intent = parse_intent(WORKED_TARGET)
        assert "mci" in intent.target_population

    def test_worked_target_horizon(self):
        intent = parse_intent(WORKED_TARGET)
        assert intent.horizon_years == 3

    def test_worked_target_question_type(self):
        intent = parse_intent(WORKED_TARGET)
        # Asks both "predict" and "how many subjects" → MODELING_DESIGN
        assert intent.question_type == QuestionType.MODELING_DESIGN

    def test_worked_target_candidate_label(self):
        intent = parse_intent(WORKED_TARGET)
        assert "ad" in intent.candidate_label.lower() or "conversion" in intent.candidate_label.lower()

    def test_worked_target_pre_index(self):
        intent = parse_intent(WORKED_TARGET)
        assert intent.pre_index_only is True

    def test_worked_target_id_space_default_A(self):
        intent = parse_intent(WORKED_TARGET)
        assert intent.id_space == IdSpace.A

    def test_simple_cohort_size_question(self):
        intent = parse_intent("How many MCI subjects are in the dataset?")
        assert intent.question_type in (QuestionType.COHORT_SIZE, QuestionType.MODELING_DESIGN)
        assert "mci" in intent.target_population

    def test_lookup_question(self):
        intent = parse_intent("What is the definition of MMSE30?")
        assert intent.question_type == QuestionType.LOOKUP

    def test_horizon_extraction(self):
        intent = parse_intent("Predict 5-year conversion to dementia in MCI.")
        assert intent.horizon_years == 5

    def test_default_horizon_when_absent(self):
        intent = parse_intent("Can we predict conversion in MCI?")
        assert intent.horizon_years == 3

    def test_schema_is_frozen(self):
        intent = parse_intent(WORKED_TARGET)
        with pytest.raises(Exception):
            intent.horizon_years = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration test: worked target end-to-end
# ---------------------------------------------------------------------------


class TestWorkedTargetEndToEnd:
    def test_grounded(self, catalog, backend):
        response = run_agent(WORKED_TARGET, catalog, backend)
        assert response.grounded, f"Expected grounded; got: {response.abstain_reason}"

    def test_hallucination_guard_passes(self, catalog, backend):
        response = run_agent(WORKED_TARGET, catalog, backend)
        assert response.hallucination_check_passed

    def test_tool_results_non_empty(self, catalog, backend):
        response = run_agent(WORKED_TARGET, catalog, backend)
        assert len(response.tool_results) >= 1

    def test_at_least_one_grounded_tool_result(self, catalog, backend):
        response = run_agent(WORKED_TARGET, catalog, backend)
        assert any(tr.grounded for tr in response.tool_results)

    def test_answer_mentions_eligible_count(self, catalog, backend):
        response = run_agent(WORKED_TARGET, catalog, backend)
        # Answer must contain a number (the eligible N)
        assert re.search(r"\b\d+\b", response.answer), "Answer contains no numbers"

    def test_answer_mentions_horizon(self, catalog, backend):
        response = run_agent(WORKED_TARGET, catalog, backend)
        assert "3" in response.answer

    def test_answer_not_empty(self, catalog, backend):
        response = run_agent(WORKED_TARGET, catalog, backend)
        assert len(response.answer.strip()) > 0

    def test_evidence_non_empty(self, catalog, backend):
        response = run_agent(WORKED_TARGET, catalog, backend)
        grounded_results = [tr for tr in response.tool_results if tr.grounded]
        total_evidence = sum(len(tr.evidence) for tr in grounded_results)
        assert total_evidence >= 1

    def test_with_explicit_intent(self, catalog, backend):
        """Bypassing the parser with an explicit intent still produces a grounded answer."""
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(WORKED_TARGET, catalog, backend, intent=intent)
        assert response.grounded
        assert response.hallucination_check_passed

    def test_modeling_design_result_in_tool_results(self, catalog, backend):
        from alz_agent.tools.deterministic import ModelingDesignResult
        response = run_agent(WORKED_TARGET, catalog, backend)
        has_design = any(
            isinstance(tr.value, ModelingDesignResult)
            for tr in response.tool_results
            if tr.grounded
        )
        assert has_design

    def test_feasibility_numbers_in_answer(self, catalog, backend):
        response = run_agent(WORKED_TARGET, catalog, backend)
        # sufficient / insufficient / censored counts should appear
        assert "sufficient" in response.answer.lower() or re.search(r"\bfollow.up\b", response.answer.lower())


# ---------------------------------------------------------------------------
# Abstain propagation
# ---------------------------------------------------------------------------


class TestAbstainPropagation:
    def test_invalid_filter_abstains(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["not_a_real_diagnosis_xyz"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert not response.grounded
        assert response.abstain_reason is not None
        assert len(response.abstain_reason) > 0

    def test_abstain_answer_is_informative(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["bad_dx_label"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert not response.grounded
        # Answer must explain why, not silently fail
        assert "not" in response.answer.lower() or "abstain" in response.answer.lower()

    def test_suppressed_cohort_abstains(self, catalog, backend):
        # visit_window=[15,16] + mci → 5 participants → suppressed
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        # Override filter spec by injecting a cohort_size intent manually:
        # We need FilterSpec with visit_window — build the ToolResult directly
        from alz_agent.agent.orchestrator import _build_filter_spec
        filter_spec = FilterSpec(
            baseline_diagnosis=["mci"],
            visit_window=(15, 16),
            id_space=IdSpace.A,
        )
        result = compute_cohort_size(filter_spec, backend)
        assert not result.grounded  # suppressed
        assert result.abstain_reason is not None
        assert "suppressed" in result.abstain_reason.lower()

    def test_hallucination_guard_passes_on_all_abstain_response(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["nonexistent_dx"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        assert not response.grounded
        # Abstain response must always pass hallucination check
        assert response.hallucination_check_passed


# ---------------------------------------------------------------------------
# Hallucination guard unit tests
# ---------------------------------------------------------------------------


class TestHallucinationGuard:
    def _make_count_result(self, backend) -> ToolResult:
        spec = FilterSpec(baseline_diagnosis=["mci"])
        return compute_cohort_size(spec, backend)

    def test_guard_passes_for_real_count(self, backend):
        tr = self._make_count_result(backend)
        assert tr.grounded
        answer = f"There are {tr.value} eligible MCI participants."
        assert _check_hallucination(answer, [tr])

    def test_guard_fails_for_invented_count(self, backend):
        tr = self._make_count_result(backend)
        assert tr.grounded
        invented = tr.value + 99999
        answer = f"There are {invented} eligible participants."
        assert not _check_hallucination(answer, [tr])

    def test_guard_passes_for_empty_answer(self, backend):
        tr = self._make_count_result(backend)
        assert _check_hallucination("", [tr])

    def test_guard_passes_for_answer_with_no_numbers(self, backend):
        tr = self._make_count_result(backend)
        assert _check_hallucination("Data is available.", [tr])

    def test_guard_passes_for_abstained_tool_result(self):
        tr = ToolResult.abstain("No data available")
        assert _check_hallucination("Data is not available.", [tr])

    def test_guard_fails_when_number_not_in_evidence(self, backend):
        tr = self._make_count_result(backend)
        # "42" is extremely unlikely to be the actual count
        assert not _check_hallucination("There are 999999999 participants.", [tr])

    def test_collect_evidence_numbers_includes_value(self, backend):
        tr = self._make_count_result(backend)
        assert tr.grounded
        nums = _collect_evidence_numbers([tr])
        assert str(tr.value) in nums

    def test_collect_evidence_numbers_includes_horizon_from_feasibility(self, catalog, backend):
        from alz_agent.tools.deterministic import assess_longitudinal_feasibility
        spec = FilterSpec(baseline_diagnosis=["mci"])
        tr = assess_longitudinal_feasibility(3, spec, backend)
        nums = _collect_evidence_numbers([tr])
        assert "3" in nums  # horizon appears in query_spec

    def test_guard_passes_on_template_narration(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        spec = FilterSpec(baseline_diagnosis=["mci"])
        tr = suggest_modeling_design(spec, 3, catalog, backend)
        assert tr.grounded
        answer = _narrate_template(intent, [tr], [])
        assert _check_hallucination(answer, [tr]), (
            "Template narration must always pass hallucination guard.\n"
            f"Answer:\n{answer}\n\n"
            f"Evidence nums: {_collect_evidence_numbers([tr])}\n"
            f"Answer ints: {set(re.findall(r'\\b\\d+\\b', answer))}"
        )


# ---------------------------------------------------------------------------
# Narration template unit tests
# ---------------------------------------------------------------------------


class TestNarrationTemplate:
    def test_includes_horizon(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        spec = FilterSpec(baseline_diagnosis=["mci"])
        tr = suggest_modeling_design(spec, 3, catalog, backend)
        answer = _narrate_template(intent, [tr], [])
        assert "3" in answer

    def test_includes_eligible_n(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        spec = FilterSpec(baseline_diagnosis=["mci"])
        tr = suggest_modeling_design(spec, 3, catalog, backend)
        assert tr.grounded and tr.value.eligible_n is not None
        answer = _narrate_template(intent, [tr], [])
        assert str(tr.value.eligible_n) in answer

    def test_abstain_result_marked_clearly(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        abstained = ToolResult.abstain("test abstain reason")
        answer = _narrate_template(intent, [abstained], [])
        assert "NOT ANSWERABLE" in answer or "not answerable" in answer.lower()
        assert "test abstain reason" in answer

    def test_doc_evidence_section_present_when_non_empty(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        from alz_agent.schemas import DocEvidence
        doc_ev = [DocEvidence(source="NACC/uds3-rdd.pdf", locator="p.42", snippet="MCI definition...")]
        tr = ToolResult.abstain("no data")
        answer = _narrate_template(intent, [tr], doc_ev)
        assert "Documentation evidence" in answer


# ---------------------------------------------------------------------------
# Cohort-size question type
# ---------------------------------------------------------------------------


class TestCohortSizeQuestionType:
    def test_cohort_size_question_returns_grounded(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=False,
            id_space=IdSpace.A,
        )
        response = run_agent("How many MCI subjects?", catalog, backend, intent=intent)
        assert response.grounded
        assert response.hallucination_check_passed


# ---------------------------------------------------------------------------
# Modality overlap integration
# ---------------------------------------------------------------------------


class TestModalityOverlapInToolLoop:
    def test_modality_overlap_included_when_modalities_present(self, catalog, backend):
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
        # Should have >=2 tool results: design + overlap
        assert len(response.tool_results) >= 2

    def test_invalid_modality_abstains_gracefully(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["INVALID_MODALITY"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("...", catalog, backend, intent=intent)
        # Design result should still be grounded; overlap abstains
        grounded = [tr for tr in response.tool_results if tr.grounded]
        abstained = [tr for tr in response.tool_results if not tr.grounded]
        assert len(grounded) >= 1  # design result
        assert len(abstained) >= 1  # modality overlap


# ---------------------------------------------------------------------------
# Regression tests: horizon_years nullable + DeepSeek 0-placeholder fix
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _make_mock_tool_client(plan_dict: dict):
    """Mock anthropic client that returns plan_dict via the plan_analysis tool call."""
    block = SimpleNamespace(
        type="tool_use",
        name="plan_analysis",
        input=plan_dict,
    )
    response = SimpleNamespace(content=[block])
    messages = SimpleNamespace(create=lambda **kw: response)
    return SimpleNamespace(messages=messages)


class TestHorizonYearsNullable:
    """Unit tests for the StructuredIntent schema change (Optional horizon)."""

    def test_none_horizon_is_valid(self):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="participant count",
            horizon_years=None,
            modalities=["cognition"],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        assert intent.horizon_years is None

    def test_positive_horizon_is_valid(self):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.FEASIBILITY,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        assert intent.horizon_years == 3

    def test_zero_horizon_is_rejected(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError, match="horizon_years"):
            StructuredIntent(
                target_population=["mci"],
                candidate_label="conversion to AD",
                horizon_years=0,
                modalities=[],
                question_type=QuestionType.FEASIBILITY,
                pre_index_only=True,
                id_space=IdSpace.A,
            )

    def test_negative_horizon_is_rejected(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError, match="horizon_years"):
            StructuredIntent(
                target_population=["mci"],
                candidate_label="conversion to AD",
                horizon_years=-1,
                modalities=[],
                question_type=QuestionType.FEASIBILITY,
                pre_index_only=True,
                id_space=IdSpace.A,
            )


class TestNormalizeLlmOutput:
    """Unit tests for the _normalize_llm_output normalization function."""

    def setup_method(self):
        from alz_agent.agent.intent import _normalize_llm_output
        self._normalize = _normalize_llm_output

    def test_zero_horizon_non_longitudinal_becomes_none(self):
        data = {"horizon_years": 0, "question_type": "cohort_size"}
        result = self._normalize(data, "How many participants have cognition?")
        assert result["horizon_years"] is None

    def test_null_horizon_unchanged(self):
        data = {"horizon_years": None, "question_type": "cohort_size"}
        result = self._normalize(data, "How many participants?")
        assert result["horizon_years"] is None

    def test_positive_horizon_unchanged(self):
        data = {"horizon_years": 3, "question_type": "feasibility"}
        result = self._normalize(data, "Predict 3-year conversion in MCI.")
        assert result["horizon_years"] == 3

    def test_explicit_zero_horizon_in_question_not_normalized(self):
        # The user literally asked for a "0-year" horizon → preserve (let ge=1 reject it)
        data = {"horizon_years": 0, "question_type": "feasibility"}
        result = self._normalize(data, "Can we predict 0-year conversion to AD?")
        assert result["horizon_years"] == 0

    def test_imaging_alias_expanded(self):
        data = {"horizon_years": None, "modalities": ["cognition", "imaging"]}
        result = self._normalize(data, "How many have cognition and imaging?")
        assert "mri" in result["modalities"]
        assert "pet" in result["modalities"]
        assert "cognition" in result["modalities"]
        assert "imaging" not in result["modalities"]

    def test_any_imaging_alias_expanded(self):
        data = {"horizon_years": None, "modalities": ["cognition", "any_imaging"]}
        result = self._normalize(data, "How many have cognition and any imaging?")
        assert "mri" in result["modalities"]
        assert "pet" in result["modalities"]

    def test_no_duplicate_modalities_after_expansion(self):
        data = {"horizon_years": None, "modalities": ["mri", "imaging"]}
        result = self._normalize(data, "How many have mri and imaging?")
        assert result["modalities"].count("mri") == 1

    def test_original_data_not_mutated(self):
        original = {"horizon_years": 0, "modalities": ["imaging"]}
        self._normalize(original, "How many?")
        assert original["horizon_years"] == 0
        assert original["modalities"] == ["imaging"]

    # -- null string coercion (small-model robustness) ----------------------

    def test_null_candidate_label_derived_from_question(self):
        """None candidate_label is replaced with a truncated version of the question."""
        data = {"candidate_label": None}
        result = self._normalize(data, "Count the number of participants with baseline MCI.")
        assert isinstance(result["candidate_label"], str)
        assert len(result["candidate_label"]) > 0
        assert "MCI" in result["candidate_label"]

    def test_null_candidate_label_truncated_to_80_chars(self):
        long_q = "A" * 200
        result = self._normalize({"candidate_label": None}, long_q)
        assert result["candidate_label"] == "A" * 80

    def test_non_null_candidate_label_preserved(self):
        data = {"candidate_label": "conversion to AD"}
        result = self._normalize(data, "Some question.")
        assert result["candidate_label"] == "conversion to AD"

    def test_null_stat_type_defaults_to_mean(self):
        """None stat_type (bypasses field default in Pydantic) is coerced to 'mean'."""
        result = self._normalize({"stat_type": None}, "Count participants.")
        assert result["stat_type"] == "mean"

    def test_non_null_stat_type_preserved(self):
        result = self._normalize({"stat_type": "median"}, "Median MMSE score?")
        assert result["stat_type"] == "median"

    def test_null_question_type_defaults_to_cohort_size(self):
        """None question_type (no field default) is coerced to 'cohort_size'."""
        result = self._normalize({"question_type": None}, "How many participants?")
        assert result["question_type"] == "cohort_size"

    def test_non_null_question_type_preserved(self):
        result = self._normalize({"question_type": "longitudinal_feasibility"}, "3-year conversion?")
        assert result["question_type"] == "longitudinal_feasibility"


class TestLlmParseIntentWithMockClient:
    """Regression tests for LLM intent parsing — mock LLM, no live API needed."""

    def test_zero_horizon_non_longitudinal_normalized(self):
        """LLM returns horizon_years=0 for a non-longitudinal question → None after normalization."""
        payload = {
            "analysis_type": "cohort_size",
            "research_intent_summary": "Count MCI participants with cognition and MRI.",
            "requires_longitudinal_data": False,
            "horizon_years": 0,  # LLM placeholder — must be normalized to None
            "candidate_label": "participant count",
            "id_space": "A",
        }
        client = _make_mock_tool_client(payload)
        intent = parse_intent(
            "How many participants have cognition and imaging?",
            client=client,
        )
        assert intent.horizon_years is None
        assert intent.question_type == QuestionType.COHORT_SIZE

    def test_null_horizon_parses_successfully(self):
        """LLM returns horizon_years=null → parsing succeeds."""
        payload = {
            "analysis_type": "cohort_size",
            "research_intent_summary": "Count MCI participants with cognition.",
            "requires_longitudinal_data": False,
            "horizon_years": None,
            "candidate_label": "participant count",
            "id_space": "A",
        }
        client = _make_mock_tool_client(payload)
        intent = parse_intent("How many participants have cognition?", client=client)
        assert intent.horizon_years is None

    def test_three_year_horizon_preserved(self):
        """LLM returns horizon_years=3 → preserved as-is."""
        payload = {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "3-year conversion to AD feasibility.",
            "requires_longitudinal_data": True,
            "horizon_years": 3,
            "candidate_label": "conversion to AD",
            "id_space": "A",
        }
        client = _make_mock_tool_client(payload)
        intent = parse_intent("Can we predict 3-year conversion to AD?", client=client)
        assert intent.horizon_years == 3

    def test_explicit_zero_horizon_in_question_rejected(self):
        """LLM returns 0 AND question says '0-year' → NOT normalized, ge=1 rejects it."""
        import pydantic
        payload = {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "0-year conversion to AD.",
            "requires_longitudinal_data": False,
            "horizon_years": 0,
            "candidate_label": "conversion to AD",
            "id_space": "A",
        }
        client = _make_mock_tool_client(payload)
        with pytest.raises(pydantic.ValidationError, match="horizon_years"):
            parse_intent("Can we predict 0-year conversion to AD?", client=client)

    def test_plan_path_parses_cohort_size_successfully(self):
        """Plan-format payload parses to COHORT_SIZE question type."""
        payload = {
            "analysis_type": "cohort_size",
            "research_intent_summary": "Count MCI participants with cognition and imaging modalities.",
            "requires_longitudinal_data": False,
            "horizon_years": None,
            "candidate_label": "participant count",
            "id_space": "A",
        }
        client = _make_mock_tool_client(payload)
        intent = parse_intent(
            "How many participants have cognition and imaging?",
            client=client,
        )
        assert intent.question_type == QuestionType.COHORT_SIZE
        assert intent.horizon_years is None


class TestCohortSizeWithModalities:
    """When COHORT_SIZE has modalities, the overlap tool must also run."""

    def test_modality_overlap_called_for_cohort_size_with_modalities(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="participant count",
            horizon_years=None,
            modalities=["cognition", "mri"],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(
            "How many MCI participants have cognition and MRI?",
            catalog, backend, intent=intent,
        )
        # Should have ≥2 tool results: cohort_size + overlap
        assert len(response.tool_results) >= 2
        assert response.hallucination_check_passed

    def test_cohort_size_without_modalities_unchanged(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="participant count",
            horizon_years=None,
            modalities=[],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(
            "How many MCI participants are there?",
            catalog, backend, intent=intent,
        )
        # Only cohort_size tool, no overlap
        assert len(response.tool_results) == 1
        assert response.grounded

    def test_none_horizon_feasibility_uses_default(self, catalog, backend):
        """FEASIBILITY intent with horizon_years=None still produces a grounded design."""
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=None,
            modalities=[],
            question_type=QuestionType.FEASIBILITY,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("Can we predict conversion in MCI?", catalog, backend, intent=intent)
        assert response.grounded
        assert response.hallucination_check_passed


class TestHeuristicHorizonNone:
    """Heuristic parser returns None for non-longitudinal questions."""

    def test_cohort_size_no_explicit_horizon_returns_none(self):
        intent = parse_intent("How many participants have cognition and imaging?")
        assert intent.horizon_years is None
        assert intent.question_type == QuestionType.COHORT_SIZE

    def test_lookup_question_returns_none_horizon(self):
        intent = parse_intent("What is the definition of MMSE30?")
        assert intent.horizon_years is None
        assert intent.question_type == QuestionType.LOOKUP

    def test_feasibility_no_explicit_horizon_defaults_to_3(self):
        intent = parse_intent("Can we predict conversion in MCI?")
        assert intent.horizon_years == 3
        assert intent.question_type == QuestionType.FEASIBILITY

    def test_modeling_design_no_explicit_horizon_defaults_to_3(self):
        intent = parse_intent(
            "Among MCI subjects, how many can we predict will convert?"
        )
        assert intent.horizon_years == 3
        assert intent.question_type == QuestionType.MODELING_DESIGN

    def test_cohort_size_explicit_horizon_preserved(self):
        intent = parse_intent("How many 3-year follow-up MCI subjects are there?")
        # An explicit "3-year" mention → horizon is captured even in a COHORT_SIZE question
        assert intent.horizon_years == 3


# ---------------------------------------------------------------------------
# Value-variable resolver: alias map + fallback tests
# ---------------------------------------------------------------------------


def _make_naccage_catalog() -> list[CatalogEntry]:
    """Minimal catalog containing NACCAGE (and NACCALZD for event anchoring)."""
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
            description="All-cause dementia diagnosis.",
            value_coding={"1": "Normal", "2": "Impaired-not-MCI", "3": "MCI", "4": "Dementia"},
        ),
        CatalogEntry(
            variable_id="NACC.NACCALZD",
            concept="etiologic diagnosis of alzheimer's disease",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCALZD",
            release_version="nacc70",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            leakage_reasons=["Etiologic AD attribution"],
            description="Presumptive etiologic diagnosis of Alzheimer's disease.",
            value_coding={"0": "No", "1": "Yes"},
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
    ]


class TestValueVariableFallbackResolver:
    """Tests for the role-aware fallback in _plan_aggregate_stat.

    The fallback fires when the LLM planner returns selected_raw_code=null for
    value_variable but provides a researcher_label.  lookup_variable is called
    on the label and the alias map ensures common terms like 'age' find NACCAGE.
    """

    @pytest.fixture()
    def age_catalog(self):
        return _make_naccage_catalog()

    @pytest.fixture()
    def age_backend(self):
        return build_fixture_backend(
            n_normal=50, n_mci=40, n_dementia=30, n_other=20,
            n_mci_long=25, n_mci_short=8, n_single_visit=15, n_suppressed_group=5,
            modality_fractions={"cognition": 0.9, "vascular": 0.8},
            seed=42,
        )

    def _make_plan_with_unresolved_value(
        self,
        researcher_label: str,
        analysis_type: str = "event_anchored_stat",
        visit_anchor: str | None = "first_matching_event_visit",
    ):
        """Build an AnalysisPlan whose value_variable has selected_raw_code=None."""
        from alz_agent.schemas import AnalysisPlan, AnalysisRole
        return AnalysisPlan(
            original_question="avg age at first AD diagnosis",
            analysis_type=analysis_type,
            research_intent_summary=f"Compute {researcher_label}",
            candidate_label=f"average {researcher_label} at first AD diagnosis",
            id_space="A",
            requires_longitudinal_data=True,
            value_variable=AnalysisRole(
                role="value_variable",
                researcher_label=researcher_label,
                semantic_definition=f"The {researcher_label} of the participant",
                selected_raw_code=None,  # LLM did not resolve
                resolution_status="unresolved",
            ),
            event=AnalysisRole(
                role="event",
                researcher_label="first AD diagnosis",
                semantic_definition="First visit where NACCALZD==1",
                selected_raw_code="NACCALZD",
                filter_expression={"NACCALZD": 1},
                resolution_status="resolved",
            ),
            visit_anchor=visit_anchor,
            statistic="mean",
        )

    def test_fallback_resolves_age_to_naccage(self, age_catalog, age_backend):
        """'age' with selected_raw_code=null → fallback resolves to NACCAGE."""
        from alz_agent.agent.orchestrator import _plan_aggregate_stat
        from alz_agent.schemas import FilterSpec, IdSpace
        plan = self._make_plan_with_unresolved_value("age")
        filter_spec = FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A)
        results = _plan_aggregate_stat(plan, age_catalog, age_backend, filter_spec)
        assert len(results) == 1
        assert results[0].grounded, (
            f"Expected grounded; got: {results[0].abstain_reason}"
        )

    def test_fallback_resolves_age_at_visit_to_naccage(self, age_catalog, age_backend):
        from alz_agent.agent.orchestrator import _plan_aggregate_stat
        from alz_agent.schemas import FilterSpec, IdSpace
        plan = self._make_plan_with_unresolved_value("age at visit")
        filter_spec = FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A)
        results = _plan_aggregate_stat(plan, age_catalog, age_backend, filter_spec)
        assert results[0].grounded, results[0].abstain_reason

    def test_fallback_compound_phrase_still_abstains(self, age_catalog, age_backend):
        """Compound phrases must still be rejected (decomposition error) even with fallback."""
        from alz_agent.agent.orchestrator import _plan_aggregate_stat
        from alz_agent.schemas import FilterSpec, IdSpace
        plan = self._make_plan_with_unresolved_value("age at first AD diagnosis")
        filter_spec = FilterSpec(id_space=IdSpace.A)
        results = _plan_aggregate_stat(plan, age_catalog, age_backend, filter_spec)
        assert not results[0].grounded
        assert "compound" in results[0].abstain_reason.lower() or "decompose" in results[0].abstain_reason.lower()

    def test_fallback_unknown_label_abstains_gracefully(self, age_catalog, age_backend):
        """Unknown label with no alias → abstain (not crash)."""
        from alz_agent.agent.orchestrator import _plan_aggregate_stat
        from alz_agent.schemas import FilterSpec, IdSpace
        plan = self._make_plan_with_unresolved_value("tau suvr centiloid ratio")
        filter_spec = FilterSpec(id_space=IdSpace.A)
        results = _plan_aggregate_stat(plan, age_catalog, age_backend, filter_spec)
        assert not results[0].grounded
        assert results[0].abstain_reason is not None

    def test_no_value_variable_abstains(self, age_catalog, age_backend):
        from alz_agent.agent.orchestrator import _plan_aggregate_stat
        from alz_agent.schemas import AnalysisPlan, FilterSpec, IdSpace
        plan = AnalysisPlan(
            original_question="test",
            analysis_type="aggregate_stat",
            research_intent_summary="no value var",
            id_space="A",
            requires_longitudinal_data=False,
        )
        filter_spec = FilterSpec(id_space=IdSpace.A)
        results = _plan_aggregate_stat(plan, age_catalog, age_backend, filter_spec)
        assert not results[0].grounded
        assert "no value_variable" in results[0].abstain_reason

    def test_resolved_raw_code_naccage_works_directly(self, age_catalog, age_backend):
        """When LLM correctly returns selected_raw_code='NACCAGE', no fallback needed."""
        from alz_agent.agent.orchestrator import _plan_aggregate_stat
        from alz_agent.schemas import AnalysisPlan, AnalysisRole, FilterSpec, IdSpace
        plan = AnalysisPlan(
            original_question="avg age at first AD diagnosis",
            analysis_type="event_anchored_stat",
            research_intent_summary="Compute age at first AD diagnosis",
            candidate_label="average age at first AD diagnosis",
            id_space="A",
            requires_longitudinal_data=True,
            value_variable=AnalysisRole(
                role="value_variable",
                researcher_label="age",
                semantic_definition="Subject's age at visit",
                selected_raw_code="NACCAGE",  # LLM resolved correctly
                resolution_status="resolved",
                validation_status="valid",
            ),
            visit_anchor="first_matching_event_visit",
            statistic="mean",
        )
        filter_spec = FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A)
        results = _plan_aggregate_stat(plan, age_catalog, age_backend, filter_spec)
        assert results[0].grounded, results[0].abstain_reason


class TestEventAnchoredDecomposition:
    """Tests that compound phrases are decomposed by the heuristic parser and the
    LLM plan schema — not treated as a single variable name."""

    def test_heuristic_event_anchor_set_for_first_diagnosis(self):
        """Heuristic parser sets event_anchor='first_diagnosis' for diagnosis-anchored questions."""
        intent = parse_intent(
            "What is the average age when MCI participants are first diagnosed with AD?"
        )
        assert intent.event_anchor == "first_diagnosis"

    def test_heuristic_stat_variable_is_simple_for_age(self):
        """Heuristic parser produces 'age' (simple) not 'age at first AD diagnosis' (compound)."""
        intent = parse_intent(
            "What is the average age when MCI participants are first diagnosed with AD?"
        )
        assert intent.stat_variable == "age"
        if intent.stat_variable:
            assert "diagnosis" not in intent.stat_variable.lower()
            assert "at first" not in intent.stat_variable.lower()

    def test_compound_value_label_is_rejected_by_schema_validator(self):
        """StructuredIntent rejects value_variable_label containing compound event phrases."""
        import pydantic
        with pytest.raises(pydantic.ValidationError, match="compound|event-anchor"):
            StructuredIntent(
                target_population=[],
                candidate_label="average age at first AD diagnosis",
                modalities=[],
                question_type=QuestionType.AGGREGATE_STAT,
                pre_index_only=True,
                id_space=IdSpace.A,
                value_variable_label="age at first diagnosis",
            )


# ---------------------------------------------------------------------------
# Analysis trace tests
# ---------------------------------------------------------------------------


class TestAnalysisTrace:
    """Tests for the structured audit trace populated by run_agent().

    The trace is deterministic (no LLM calls), populated from:
    - StructuredIntent (parsing_mode, leakage_assessment, parse_debug)
    - AnalysisPlan roles (variable_resolutions, plan_steps)
    - FilterSpec (filter_trace)
    - ToolResult list (tool_calls with output_summary)
    """

    @pytest.fixture()
    def trace_catalog(self):
        return _make_catalog()

    @pytest.fixture()
    def trace_backend(self):
        return build_fixture_backend(
            n_normal=50, n_mci=40, n_dementia=30, n_other=20,
            n_mci_long=25, n_mci_short=8, n_single_visit=15, n_suppressed_group=5,
            modality_fractions={"cognition": 0.9},
            seed=42,
        )

    def _cohort_size_intent(self):
        return StructuredIntent(
            target_population=["mci"],
            candidate_label="participant count",
            modalities=[],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=True,
            id_space=IdSpace.A,
        )

    # ── 1. trace is always present ──────────────────────────────────────────

    def test_trace_present_on_grounded_response(self, trace_catalog, trace_backend):
        from alz_agent.schemas import AnalysisTrace
        resp = run_agent(
            "How many MCI participants?", trace_catalog, trace_backend,
            intent=self._cohort_size_intent(),
        )
        assert resp.trace is not None
        assert isinstance(resp.trace, AnalysisTrace)

    def test_trace_present_on_abstained_response(self, trace_catalog, trace_backend):
        """Trace is populated even when the entire answer is abstained."""
        from alz_agent.schemas import AnalysisTrace
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="stat",
            modalities=[],
            question_type=QuestionType.AGGREGATE_STAT,
            pre_index_only=True,
            id_space=IdSpace.A,
            stat_variable=None,  # missing → tool abstains
        )
        resp = run_agent("What is the average?", trace_catalog, trace_backend, intent=intent)
        assert resp.trace is not None
        assert isinstance(resp.trace, AnalysisTrace)

    # ── 2. question_type mirrors intent ─────────────────────────────────────

    def test_trace_question_type_matches_intent(self, trace_catalog, trace_backend):
        resp = run_agent(
            "How many MCI participants?", trace_catalog, trace_backend,
            intent=self._cohort_size_intent(),
        )
        assert resp.trace.question_type == "cohort_size"

    # ── 3. tool_calls count mirrors tool_results ────────────────────────────

    def test_trace_tool_calls_count_matches_tool_results(self, trace_catalog, trace_backend):
        resp = run_agent(
            "How many MCI participants?", trace_catalog, trace_backend,
            intent=self._cohort_size_intent(),
        )
        assert len(resp.trace.tool_calls) == len(resp.tool_results)

    # ── 4. numeric provenance: output_summary traces back to tool result ────

    def test_trace_cohort_size_numeric_provenance(self, trace_catalog, trace_backend):
        """trace.tool_calls[0].output_summary['n'] == resp.tool_results[0].value."""
        resp = run_agent(
            "How many MCI participants?", trace_catalog, trace_backend,
            intent=self._cohort_size_intent(),
        )
        grounded = [tr for tr in resp.tool_results if tr.grounded]
        if not grounded:
            pytest.skip("No grounded result — nothing to verify")
        actual_n = grounded[0].value
        trace_n = resp.trace.tool_calls[0].output_summary.get("n")
        assert trace_n == actual_n

    # ── 5. no raw chain-of-thought ──────────────────────────────────────────

    def test_trace_no_raw_chain_of_thought(self, trace_catalog, trace_backend):
        """Trace text fields must not contain LLM reasoning markers."""
        resp = run_agent(
            "How many MCI participants?", trace_catalog, trace_backend,
            intent=self._cohort_size_intent(),
        )
        trace = resp.trace
        _COT_MARKERS = {
            "let me think", "i think", "i'll reason", "step by step",
            "reasoning:", "thinking:", "chain of thought",
        }
        all_text = " ".join([
            trace.original_question,
            trace.parsing_mode,
            trace.question_type,
            trace.analysis_type or "",
            trace.research_intent_summary or "",
            " ".join(trace.plan_steps),
            " ".join(trace.caveats),
            " ".join(trace.validation_errors),
            trace.failure_mode or "",
        ]).lower()
        for marker in _COT_MARKERS:
            assert marker not in all_text, f"Chain-of-thought marker in trace: {marker!r}"

    # ── 6. failure_mode and validation_errors from parse_debug ──────────────

    def test_trace_failure_mode_from_parse_debug(self, trace_catalog, trace_backend):
        from alz_agent.agent.intent import ParseDebugInfo
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="participant count",
            modalities=[],
            question_type=QuestionType.AGGREGATE_STAT,
            pre_index_only=True,
            id_space=IdSpace.A,
            parsing_mode="llm",
            parse_debug=ParseDebugInfo(
                failure_mode="variable_resolution",
                validation_errors=["DECOMPOSITION_ERROR: value_variable compound phrase"],
            ),
        )
        resp = run_agent("What is the average?", trace_catalog, trace_backend, intent=intent)
        assert resp.trace.failure_mode == "variable_resolution"
        assert any("DECOMPOSITION_ERROR" in e for e in resp.trace.validation_errors)

    def test_trace_no_failure_mode_when_parse_ok(self, trace_catalog, trace_backend):
        resp = run_agent(
            "How many MCI participants?", trace_catalog, trace_backend,
            intent=self._cohort_size_intent(),
        )
        assert resp.trace.failure_mode is None
        assert resp.trace.validation_errors == []

    # ── 7. leakage findings propagated to trace ─────────────────────────────

    def test_trace_leakage_findings_populated(self, trace_catalog, trace_backend):
        from alz_agent.schemas import LeakageFinding
        finding = LeakageFinding(
            raw_code="NACCUDSD",
            concept="nacc_unified_diagnostic_summary",
            role="cohort_filter",
            leakage_tag="DIAGNOSIS_DEFINING",
            severity="allow_with_caveat",
            reason="NACCUDSD is diagnosis-defining; allowed only as cohort filter.",
            action_taken="allowed_for_cohort_definition",
        )
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="participant count",
            modalities=[],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=True,
            id_space=IdSpace.A,
            leakage_assessment=[finding],
        )
        resp = run_agent("How many MCI participants?", trace_catalog, trace_backend, intent=intent)
        codes = [f.raw_code for f in resp.trace.leakage_findings]
        assert "NACCUDSD" in codes

    # ── 8. variable_resolutions from analysis_plan ───────────────────────────

    def test_trace_variable_resolutions_from_plan(self, trace_catalog, trace_backend):
        """When an AnalysisPlan has value_variable + event, both appear in trace."""
        from alz_agent.schemas import AnalysisPlan, AnalysisRole
        age_catalog = _make_naccage_catalog()
        age_backend = build_fixture_backend(
            n_normal=50, n_mci=40, n_dementia=30, n_other=20,
            n_mci_long=25, n_mci_short=8, n_single_visit=15, n_suppressed_group=5,
            modality_fractions={"cognition": 0.9}, seed=42,
        )
        plan = AnalysisPlan(
            original_question="avg age at first AD diagnosis",
            analysis_type="event_anchored_stat",
            research_intent_summary="Compute average age at first AD etiologic diagnosis.",
            candidate_label="average age at first AD diagnosis",
            id_space="A",
            requires_longitudinal_data=True,
            value_variable=AnalysisRole(
                role="value_variable",
                researcher_label="age",
                semantic_definition="Subject's age at visit",
                selected_raw_code="NACCAGE",
                resolution_status="resolved",
                validation_status="valid",
            ),
            event=AnalysisRole(
                role="event",
                researcher_label="first AD diagnosis",
                semantic_definition="First visit with NACCALZD==1",
                selected_raw_code="NACCALZD",
                filter_expression={"NACCALZD": 1},
                resolution_status="resolved",
            ),
            visit_anchor="first_matching_event_visit",
            statistic="mean",
        )
        intent = StructuredIntent(
            target_population=[],
            target_etiology=["ad"],
            candidate_label="average age at first AD diagnosis",
            modalities=[],
            question_type=QuestionType.AGGREGATE_STAT,
            pre_index_only=True,
            id_space=IdSpace.A,
            parsing_mode="llm",
            analysis_type="event_anchored_stat",
            event_anchor="first_diagnosis",
            value_variable_label="age",
            value_column="NACCAGE",
            event_column="NACCALZD",
            event_filter={"NACCALZD": 1},
            analysis_plan=plan,
        )
        resp = run_agent(
            "Among these who were diagnosed as AD, what is the average age when they were first diagnosed?",
            age_catalog, age_backend, intent=intent,
        )
        assert resp.trace is not None
        roles = {vr.role for vr in resp.trace.variable_resolutions}
        assert "value_variable" in roles
        assert "event" in roles
        vv = next(vr for vr in resp.trace.variable_resolutions if vr.role == "value_variable")
        assert vv.selected_raw_code == "NACCAGE"
        assert vv.resolution_status == "resolved"

    def test_trace_validation_status_shown_for_rejected_variable(self, trace_catalog, trace_backend):
        """When validation_status='invalid_blocked', trace records it alongside selected_raw_code=None."""
        from alz_agent.schemas import AnalysisPlan, AnalysisRole
        plan = AnalysisPlan(
            original_question="test blocked",
            analysis_type="event_anchored_stat",
            research_intent_summary="Test blocked variable trace.",
            id_space="A",
            requires_longitudinal_data=False,
            value_variable=AnalysisRole(
                role="value_variable",
                researcher_label="diagnosis",
                semantic_definition="Diagnostic category",
                selected_raw_code=None,          # already rejected by validator
                resolution_status="unresolved",
                validation_status="invalid_blocked",
            ),
        )
        intent = StructuredIntent(
            target_population=[],
            candidate_label="test",
            modalities=[],
            question_type=QuestionType.AGGREGATE_STAT,
            pre_index_only=True,
            id_space=IdSpace.A,
            parsing_mode="llm",
            analysis_plan=plan,
        )
        resp = run_agent("test", trace_catalog, trace_backend, intent=intent)
        vv_resolutions = [vr for vr in resp.trace.variable_resolutions if vr.role == "value_variable"]
        assert vv_resolutions, "value_variable resolution trace should exist"
        vv = vv_resolutions[0]
        # selected_raw_code should be None and validation_status should record the rejection
        assert vv.selected_raw_code is None
        assert vv.validation_status == "invalid_blocked"

    # ── 9. tool_name inference unit tests ───────────────────────────────────

    def test_infer_tool_name_cohort_size(self):
        from alz_agent.agent.orchestrator import _infer_tool_name
        tr = ToolResult.ok(42)
        assert _infer_tool_name(tr) == "compute_cohort_size"

    def test_infer_tool_name_abstain_is_unknown(self):
        from alz_agent.agent.orchestrator import _infer_tool_name
        tr = ToolResult.abstain("not available")
        assert _infer_tool_name(tr) == "unknown"

    def test_infer_tool_name_stat_result(self):
        from alz_agent.agent.orchestrator import _infer_tool_name
        from alz_agent.tools.deterministic import StatResult
        sr = StatResult(variable_id="NACC.NACCAGE", stat_type="mean", value=72.3, n=100)
        tr = ToolResult.ok(sr)
        assert _infer_tool_name(tr) == "compute_variable_stat"

    # ── 10. filter_trace reflects FilterSpec ────────────────────────────────

    def test_trace_filter_trace_id_space(self, trace_catalog, trace_backend):
        resp = run_agent(
            "How many MCI participants?", trace_catalog, trace_backend,
            intent=self._cohort_size_intent(),
        )
        assert resp.trace.filter_trace["id_space"] == "A"

    def test_trace_filter_trace_baseline_diagnosis(self, trace_catalog, trace_backend):
        resp = run_agent(
            "How many MCI participants?", trace_catalog, trace_backend,
            intent=self._cohort_size_intent(),
        )
        assert resp.trace.filter_trace["baseline_diagnosis"] == ["mci"]

    # ── 11. AD caveats appear when etiology is 'ad' ─────────────────────────

    def test_trace_ad_caveat_when_etiology_ad(self, trace_catalog, trace_backend):
        """When target_etiology=['ad'], trace.caveats includes NACCALZD semantics caveat."""
        intent = StructuredIntent(
            target_population=[],
            target_etiology=["ad"],
            candidate_label="average age at AD diagnosis",
            modalities=[],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        resp = run_agent("How many AD participants?", trace_catalog, trace_backend, intent=intent)
        joined = " ".join(resp.trace.caveats).lower()
        assert "naccalzd" in joined or "etiologic" in joined

    # ── 12. parsing_mode preserved ──────────────────────────────────────────

    def test_trace_parsing_mode_heuristic(self, trace_catalog, trace_backend):
        resp = run_agent(
            "How many MCI participants?", trace_catalog, trace_backend,
            intent=self._cohort_size_intent(),  # parsing_mode defaults to "heuristic"
        )
        assert resp.trace.parsing_mode == "heuristic"

    def test_trace_parsing_mode_llm(self, trace_catalog, trace_backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="participant count",
            modalities=[],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=True,
            id_space=IdSpace.A,
            parsing_mode="llm",
        )
        resp = run_agent("How many MCI participants?", trace_catalog, trace_backend, intent=intent)
        assert resp.trace.parsing_mode == "llm"


# ---------------------------------------------------------------------------
# Semantic planner end-to-end integration test (P5 + semantic gate)
# ---------------------------------------------------------------------------


class TestRunAgentSemanticPathEndToEnd:
    """End-to-end test for the semantic gate wired into run_agent.

    Uses a fixture backend (no real data) and a fake PlannerClient (no real LLM).
    Verifies that the semantic gate, validation, compile, execution and abstain
    paths all work correctly through the full run_agent() call.
    """

    def _make_catalog(self):
        return [
            CatalogEntry(
                variable_id="nacc_age",
                concept="age at visit",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="NACCAGE",
                release_version="v70",
                modality=Modality.DEMOGRAPHIC,
            ),
            CatalogEntry(
                variable_id="nacc_alz",
                concept="etiologic alzheimer diagnosis",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="NACCALZD",
                release_version="v70",
                modality=Modality.COGNITION,
            ),
        ]

    def _make_fake_client(self, invoke_tool_returns):
        from alz_agent.agent.providers import PlannerClient

        class _Fake(PlannerClient):
            def __init__(self, returns):
                self._returns = list(returns)
                self._i = 0

            def invoke_tool(self, messages, tool_schema, max_tokens=1500):
                if self._i < len(self._returns):
                    val = self._returns[self._i]
                    self._i += 1
                    return val
                return None

            def complete_text(self, messages, max_tokens=1024):
                return ""

        return _Fake(invoke_tool_returns)

    def _correct_plan_data(self):
        return {
            "population_unit": "participant",
            "requires_unique_participants": True,
            "cohort_scope": "first_event",
            "baseline_filters": {},
            "event_filters": {"NACCALZD": 1},
            "event_anchor": "first_matching_event_visit",
            "time_ordering_variable": "NACCVNUM",
            "outcome_definition": "NACCALZD=1",
            "etiology_definition": "etiologic_ad",
            "diagnosis_variable": "NACCALZD",
            "statistic_variable": "NACCAGE",
            "statistic_type": "mean",
            "denominator_definition": "unique NACC participants with NACCALZD=1 on any visit",
            "allowed_backend_tools": ["variable_stat"],
            "known_invalid_interpretations": ["NOT baseline-only filter"],
            "caveats": [],
            "abstain_if_missing": [],
            "id_space": "A",
        }

    def test_run_agent_semantic_path_end_to_end_fixture(self):
        """Fixture backend + fake planner client, etiologic-AD question.
        Asserts that semantic validation hard errors are empty and the
        response is coherent (grounded or abstains with a backend reason,
        not a semantic validation reason)."""
        nl = (
            "Among participants with etiologic Alzheimer's disease in NACC, "
            "what is the mean age at first diagnosis?"
        )
        catalog = self._make_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        client = self._make_fake_client([self._correct_plan_data()])

        resp = run_agent(
            nl,
            catalog,
            backend,
            client=client,
            use_enriched_planner=True,
            use_semantic_planner=True,
        )

        # Semantic gate passed — no semantic validation hard errors
        assert resp.validation_hard_errors == [], (
            f"Expected no semantic validation hard errors; got: {resp.validation_hard_errors}"
        )

        # Response should not be a semantic abstention
        if resp.abstain_reason:
            assert "not answerable with the current backend semantics" not in resp.abstain_reason

        # enriched_plan_used should be True (went through enriched path)
        assert resp.enriched_plan_used is True

    def test_run_agent_semantic_gate_abstains_on_unresolvable(self):
        """If the LLM returns an invalid plan on both initial + repair calls,
        run_agent should return grounded=False with the semantic abstain message."""
        invalid_data = {
            "population_unit": "participant",
            "requires_unique_participants": True,
            "cohort_scope": "unclear",  # triggers C6 hard error
            "baseline_filters": {},
            "event_filters": {},
            "event_anchor": None,
            "etiology_definition": "etiologic_ad",
            "diagnosis_variable": "NACCUDSD",  # wrong (C3)
            "statistic_type": "mean",
            "denominator_definition": "",  # missing (C4)
            "id_space": "A",
        }
        nl = "Among participants with etiologic AD, what is the mean age?"
        catalog = self._make_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        # Both initial plan and repair return invalid data
        client = self._make_fake_client([invalid_data, invalid_data])

        resp = run_agent(
            nl,
            catalog,
            backend,
            client=client,
            use_enriched_planner=True,
            use_semantic_planner=True,
        )

        assert resp.grounded is False
        assert "not answerable with the current backend semantics" in resp.answer
        assert len(resp.validation_hard_errors) > 0
