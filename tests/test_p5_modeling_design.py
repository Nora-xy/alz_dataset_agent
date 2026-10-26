"""P5 tests — LLM-led evidence planning for modeling_design questions.

Three test scenarios from the task specification:

1. "How should we design a model for MCI-to-AD conversion?"
   - question_type == modeling_design (not regression)
   - heuristic path: calls suggest_modeling_design + modality overlap tools
   - answer includes grounded counts when available

2. "Can we predict 3-year MCI-to-AD conversion using cognition and MRI?"
   - horizon_years == 3
   - modalities include cognition and MRI
   - overlap tool is called

3. Missing evidence is clearly separated from computed facts
   - grounded=False tool results appear as NOT ANSWERABLE in template narration
   - grounded=True results are still reported

Mock-LLM tests verify the evidence planning path end-to-end without a live API key.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from alz_agent.agent.intent import QuestionType, StructuredIntent, parse_intent
from alz_agent.agent.orchestrator import (
    AgentResponse,
    _check_hallucination,
    _execute_evidence_plan,
    _narrate_template,
    run_agent,
)
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    CatalogEntry,
    EvidenceNeedItem,
    FilterSpec,
    IdSpace,
    LeakageTag,
    ModelingDesignPlan,
    Modality,
    SourceDataset,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Shared fixtures
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
            leakage_reasons=["AD etiologic attribution"],
            value_coding={"0": "No", "1": "Yes"},
        ),
        CatalogEntry(
            variable_id="NACC.NACCMMSE",
            concept="mmse total score",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCMMSE",
            release_version="nacc70",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
        CatalogEntry(
            variable_id="NACC.MRI_VOL",
            concept="mri brain volume",
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
            concept="pet amyloid suvr",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="PET_AMYLOID",
            release_version="nacc70",
            modality=Modality.PET,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
        CatalogEntry(
            variable_id="NACC.CSFABETA",
            concept="csf abeta42",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="CSFABETA",
            release_version="nacc70",
            modality=Modality.FLUID_BIOMARKER,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
    ]


@pytest.fixture()
def catalog():
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
# Helpers for mock LLM client
# ---------------------------------------------------------------------------

# Minimal plan_analysis response that classifies any question as modeling_design.
_PLAN_ANALYSIS_MODELING_DESIGN: dict[str, Any] = {
    "analysis_type": "modeling_design",
    "research_intent_summary": "Design a model for MCI-to-AD conversion",
    "requires_longitudinal_data": True,
    "horizon_years": 3,
    "candidate_label": "conversion to AD",
    "id_space": "A",
    "cohort": {
        "role": "cohort",
        "researcher_label": "baseline MCI",
        "semantic_definition": "Participants with MCI at baseline (NACCUDSD==3)",
        "selected_raw_code": "NACCUDSD",
        "filter_expression": {"NACCUDSD": 3},
        "resolution_status": "resolved",
    },
}


def _make_mock_combined_client(plan_dict: dict[str, Any], narration_text: str) -> Any:
    """Mock anthropic client that dispatches by tool_choice.name:

    - plan_analysis          → plan_analysis tool_use (modeling_design intent)
    - plan_modeling_evidence → plan_modeling_evidence tool_use with plan_dict
    - anything else          → text narration_text

    run_agent() makes three sequential calls:
      1. parse_intent     → plan_analysis
      2. evidence planner → plan_modeling_evidence
      3. narration        → text
    """

    def _create(**kw):
        tool_choice = kw.get("tool_choice", {})
        tool_name = tool_choice.get("name") if isinstance(tool_choice, dict) else None

        if tool_name == "plan_analysis":
            block = SimpleNamespace(
                type="tool_use",
                name="plan_analysis",
                input=_PLAN_ANALYSIS_MODELING_DESIGN,
            )
            return SimpleNamespace(content=[block])

        if tool_name == "plan_modeling_evidence":
            block = SimpleNamespace(
                type="tool_use",
                name="plan_modeling_evidence",
                input=plan_dict,
            )
            return SimpleNamespace(content=[block])

        # Narration (no tool_choice)
        text_block = SimpleNamespace(type="text", text=narration_text)
        return SimpleNamespace(content=[text_block])

    messages = SimpleNamespace(create=_create)
    return SimpleNamespace(messages=messages)


def _make_planner_failure_client(narration_text: str) -> Any:
    """Mock client where the evidence planner returns no tool_use block.

    run_agent() call sequence:
      1. parse_intent     → plan_analysis (modeling_design)
      2. evidence planner → text only (no plan_modeling_evidence block → planner returns None)
      3. narration        → text narration_text
    """

    def _create(**kw):
        tool_choice = kw.get("tool_choice", {})
        tool_name = tool_choice.get("name") if isinstance(tool_choice, dict) else None

        if tool_name == "plan_analysis":
            block = SimpleNamespace(
                type="tool_use",
                name="plan_analysis",
                input=_PLAN_ANALYSIS_MODELING_DESIGN,
            )
            return SimpleNamespace(content=[block])

        if tool_name == "plan_modeling_evidence":
            # Return a text block, not a tool_use block → _call_modeling_design_planner returns None
            text_block = SimpleNamespace(type="text", text="I cannot plan this.")
            return SimpleNamespace(content=[text_block])

        text_block = SimpleNamespace(type="text", text=narration_text)
        return SimpleNamespace(content=[text_block])

    messages = SimpleNamespace(create=_create)
    return SimpleNamespace(messages=messages)


# ---------------------------------------------------------------------------
# Scenario 1: "How should we design a model for MCI-to-AD conversion?"
# ---------------------------------------------------------------------------


class TestScenario1MciToAdDesign:
    """Heuristic-path tests for the canonical MCI-to-AD question."""

    Q = "How should we design a model for MCI-to-AD conversion?"

    def test_heuristic_classifies_as_modeling_design(self):
        intent = parse_intent(self.Q)
        assert intent.question_type == QuestionType.MODELING_DESIGN, (
            f"Expected MODELING_DESIGN, got {intent.question_type}"
        )

    def test_heuristic_not_regression(self):
        intent = parse_intent(self.Q)
        assert intent.question_type != QuestionType.STATISTICAL_ANALYSIS
        assert intent.analysis_type != "regression"

    def test_heuristic_target_population_mci(self):
        intent = parse_intent(self.Q)
        assert "mci" in intent.target_population

    def test_heuristic_target_etiology_ad(self):
        intent = parse_intent(self.Q)
        assert intent.target_etiology is not None
        assert "ad" in intent.target_etiology

    def test_agent_returns_grounded_response(self, catalog, backend):
        response = run_agent(self.Q, catalog, backend)
        assert response.grounded, f"Expected grounded; got: {response.abstain_reason}"

    def test_agent_hallucination_guard_passes(self, catalog, backend):
        response = run_agent(self.Q, catalog, backend)
        assert response.hallucination_check_passed

    def test_agent_calls_suggest_modeling_design(self, catalog, backend):
        from alz_agent.tools.deterministic import ModelingDesignResult
        response = run_agent(self.Q, catalog, backend)
        has_design = any(
            isinstance(tr.value, ModelingDesignResult)
            for tr in response.tool_results if tr.grounded
        )
        assert has_design, "Expected suggest_modeling_design in tool results"

    def test_agent_answer_contains_counts(self, catalog, backend):
        response = run_agent(self.Q, catalog, backend)
        assert re.search(r"\b\d+\b", response.answer), "Answer must contain at least one integer"

    def test_trace_question_type_modeling_design(self, catalog, backend):
        response = run_agent(self.Q, catalog, backend)
        assert response.trace is not None
        assert response.trace.question_type == "modeling_design"

    def test_trace_no_modeling_plan_without_client(self, catalog, backend):
        """Heuristic path (no LLM client) must not populate modeling_design_plan."""
        response = run_agent(self.Q, catalog, backend)
        assert response.trace.modeling_design_plan is None


# ---------------------------------------------------------------------------
# Scenario 2: "Can we predict 3-year MCI-to-AD conversion using cognition and MRI?"
# ---------------------------------------------------------------------------


class TestScenario2HorizonAndModalities:
    """Tests that horizon and modalities from the question drive the tool calls."""

    Q = "Can we predict 3-year MCI-to-AD conversion using cognition and MRI?"

    def test_heuristic_horizon_is_3(self):
        intent = parse_intent(self.Q)
        assert intent.horizon_years == 3

    def test_heuristic_modalities_include_cognition_and_mri(self):
        intent = parse_intent(self.Q)
        assert "cognition" in intent.modalities
        assert "mri" in intent.modalities

    def test_agent_grounded(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            target_etiology=["ad"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "mri"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(self.Q, catalog, backend, intent=intent)
        assert response.grounded

    def test_agent_calls_modality_overlap(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            target_etiology=["ad"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "mri"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(self.Q, catalog, backend, intent=intent)
        # At least one dict result from assess_modality_overlap or assess_longitudinal_feasibility
        dict_results = [tr for tr in response.tool_results if tr.grounded and isinstance(tr.value, dict)]
        assert dict_results, "Expected at least one dict result (overlap or feasibility)"

    def test_agent_answer_contains_3(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            target_etiology=["ad"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "mri"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(self.Q, catalog, backend, intent=intent)
        assert "3" in response.answer

    def test_hallucination_guard_passes(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            target_etiology=["ad"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "mri"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(self.Q, catalog, backend, intent=intent)
        assert response.hallucination_check_passed

    def test_at_least_two_tool_results(self, catalog, backend):
        """design result + overlap = at least 2 results."""
        intent = StructuredIntent(
            target_population=["mci"],
            target_etiology=["ad"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "mri"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(self.Q, catalog, backend, intent=intent)
        assert len(response.tool_results) >= 2


# ---------------------------------------------------------------------------
# Scenario 3: Missing evidence is clearly separated from computed facts
# ---------------------------------------------------------------------------


class TestScenario3MissingEvidence:
    """When some evidence items cannot be computed, grounded=False results must
    appear explicitly in the answer and not cause hallucination failures."""

    def _intent_bad_modality(self) -> StructuredIntent:
        return StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=["cognition", "neuropath"],  # neuropath has 0 overlap in fixture
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )

    def test_mixed_results_produce_grounded_response(self, catalog, backend):
        """When the design result is grounded but overlap abstains, still grounded."""
        response = run_agent("...", catalog, backend, intent=self._intent_bad_modality())
        # At least the design result should be grounded
        grounded = [tr for tr in response.tool_results if tr.grounded]
        assert grounded, "At least one tool result must be grounded"

    def test_abstained_results_appear_in_template_answer(self, catalog, backend):
        response = run_agent("...", catalog, backend, intent=self._intent_bad_modality())
        # If any abstained results exist, the template must label them
        abstained = [tr for tr in response.tool_results if not tr.grounded]
        if abstained:
            assert "NOT ANSWERABLE" in response.answer or "not answerable" in response.answer.lower()

    def test_hallucination_guard_passes_on_mixed_results(self, catalog, backend):
        response = run_agent("...", catalog, backend, intent=self._intent_bad_modality())
        assert response.hallucination_check_passed

    def test_fully_abstained_cohort_abstains_cleanly(self, catalog, backend):
        """Invalid filter → all tools abstain → response.grounded == False."""
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
        assert response.hallucination_check_passed


# ---------------------------------------------------------------------------
# Evidence plan schema unit tests
# ---------------------------------------------------------------------------


class TestEvidencePlanSchemas:
    """Validate that the new schema types construct and freeze correctly."""

    def test_evidence_need_item_constructs(self):
        item = EvidenceNeedItem(
            item_id="baseline_mci_n",
            description="Baseline MCI cohort size",
            tool_name="compute_cohort_size",
            baseline_diagnosis=["mci"],
            priority="required",
        )
        assert item.item_id == "baseline_mci_n"
        assert item.tool_name == "compute_cohort_size"
        assert item.modalities is None

    def test_evidence_need_item_is_frozen(self):
        item = EvidenceNeedItem(
            item_id="test", description="test", tool_name="compute_cohort_size"
        )
        with pytest.raises(Exception):
            item.item_id = "changed"  # type: ignore[misc]

    def test_modeling_design_plan_constructs(self):
        plan = ModelingDesignPlan(
            original_question="How to model MCI-to-AD?",
            modeling_goal="predict MCI-to-AD conversion",
            target_population="baseline MCI (NACCUDSD==3)",
            prediction_horizon_years=3,
            candidate_modalities=["cognition", "mri"],
            label_definition_options=[
                "NACCALZD==1 (AD etiologic)",
                "NACCUDSD==4 (all-cause dementia)",
            ],
            evidence_items=[
                EvidenceNeedItem(
                    item_id="baseline_n",
                    description="Baseline MCI cohort size",
                    tool_name="compute_cohort_size",
                    baseline_diagnosis=["mci"],
                ),
            ],
        )
        assert plan.prediction_horizon_years == 3
        assert len(plan.evidence_items) == 1
        assert plan.evidence_items[0].item_id == "baseline_n"

    def test_modeling_design_plan_is_frozen(self):
        plan = ModelingDesignPlan(
            original_question="test",
            modeling_goal="test",
            target_population="mci",
        )
        with pytest.raises(Exception):
            plan.modeling_goal = "changed"  # type: ignore[misc]

    def test_analysis_trace_has_modeling_design_plan_field(self, catalog, backend):
        from alz_agent.schemas import AnalysisTrace
        # The field must exist on the schema (even if None)
        fields = AnalysisTrace.model_fields
        assert "modeling_design_plan" in fields

    def test_trace_modeling_design_plan_is_none_by_default(self, catalog, backend):
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.COHORT_SIZE,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent("How many MCI?", catalog, backend, intent=intent)
        assert response.trace is not None
        assert response.trace.modeling_design_plan is None


# ---------------------------------------------------------------------------
# _execute_evidence_plan unit tests (heuristic path — no LLM)
# ---------------------------------------------------------------------------


class TestExecuteEvidencePlan:
    """Test that _execute_evidence_plan calls the right tools from plan items."""

    def _base_intent(self) -> StructuredIntent:
        return StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )

    def test_compute_cohort_size_item(self, catalog, backend):
        plan = ModelingDesignPlan(
            original_question="test",
            modeling_goal="test",
            target_population="mci",
            evidence_items=[
                EvidenceNeedItem(
                    item_id="n",
                    description="cohort size",
                    tool_name="compute_cohort_size",
                    baseline_diagnosis=["mci"],
                ),
            ],
        )
        results = _execute_evidence_plan(plan, self._base_intent(), catalog, backend)
        assert len(results) == 1
        assert results[0].grounded
        assert isinstance(results[0].value, int)

    def test_assess_longitudinal_feasibility_item(self, catalog, backend):
        plan = ModelingDesignPlan(
            original_question="test",
            modeling_goal="test",
            target_population="mci",
            prediction_horizon_years=3,
            evidence_items=[
                EvidenceNeedItem(
                    item_id="feas",
                    description="longitudinal feasibility",
                    tool_name="assess_longitudinal_feasibility",
                    baseline_diagnosis=["mci"],
                    horizon_years=3,
                ),
            ],
        )
        results = _execute_evidence_plan(plan, self._base_intent(), catalog, backend)
        assert len(results) == 1
        assert results[0].grounded
        assert isinstance(results[0].value, dict)

    def test_suggest_modeling_design_item(self, catalog, backend):
        from alz_agent.tools.deterministic import ModelingDesignResult
        plan = ModelingDesignPlan(
            original_question="test",
            modeling_goal="test",
            target_population="mci",
            prediction_horizon_years=3,
            evidence_items=[
                EvidenceNeedItem(
                    item_id="design",
                    description="full modeling design",
                    tool_name="suggest_modeling_design",
                    baseline_diagnosis=["mci"],
                ),
            ],
        )
        results = _execute_evidence_plan(plan, self._base_intent(), catalog, backend)
        assert len(results) == 1
        assert results[0].grounded
        assert isinstance(results[0].value, ModelingDesignResult)

    def test_assess_modality_overlap_item(self, catalog, backend):
        plan = ModelingDesignPlan(
            original_question="test",
            modeling_goal="test",
            target_population="mci",
            evidence_items=[
                EvidenceNeedItem(
                    item_id="cog_mri",
                    description="cognition+MRI overlap",
                    tool_name="assess_modality_overlap",
                    baseline_diagnosis=["mci"],
                    modalities=["cognition", "mri"],
                ),
            ],
        )
        results = _execute_evidence_plan(plan, self._base_intent(), catalog, backend)
        assert len(results) == 1
        # May be grounded or abstained depending on catalog/backend; either is valid
        assert results[0].grounded or results[0].abstain_reason is not None

    def test_overlap_without_modalities_abstains(self, catalog, backend):
        plan = ModelingDesignPlan(
            original_question="test",
            modeling_goal="test",
            target_population="mci",
            evidence_items=[
                EvidenceNeedItem(
                    item_id="bad",
                    description="overlap with no modalities specified",
                    tool_name="assess_modality_overlap",
                    # modalities intentionally omitted
                ),
            ],
        )
        results = _execute_evidence_plan(plan, self._base_intent(), catalog, backend)
        assert len(results) == 1
        assert not results[0].grounded
        assert results[0].abstain_reason is not None

    def test_unknown_tool_abstains(self, catalog, backend):
        plan = ModelingDesignPlan(
            original_question="test",
            modeling_goal="test",
            target_population="mci",
            evidence_items=[
                EvidenceNeedItem(
                    item_id="bad_tool",
                    description="unknown tool",
                    tool_name="nonexistent_tool_xyz",
                ),
            ],
        )
        results = _execute_evidence_plan(plan, self._base_intent(), catalog, backend)
        assert len(results) == 1
        assert not results[0].grounded

    def test_multiple_items_produce_multiple_results(self, catalog, backend):
        plan = ModelingDesignPlan(
            original_question="test",
            modeling_goal="test",
            target_population="mci",
            prediction_horizon_years=3,
            evidence_items=[
                EvidenceNeedItem(
                    item_id="n",
                    description="cohort size",
                    tool_name="compute_cohort_size",
                    baseline_diagnosis=["mci"],
                ),
                EvidenceNeedItem(
                    item_id="feas",
                    description="feasibility",
                    tool_name="assess_longitudinal_feasibility",
                    baseline_diagnosis=["mci"],
                    horizon_years=3,
                ),
            ],
        )
        results = _execute_evidence_plan(plan, self._base_intent(), catalog, backend)
        assert len(results) == 2

    def test_hallucination_guard_passes_on_plan_results(self, catalog, backend):
        """Results from evidence plan execution must pass the hallucination guard."""
        intent = self._base_intent()
        plan = ModelingDesignPlan(
            original_question="How should we design a model for MCI-to-AD?",
            modeling_goal="predict MCI-to-AD conversion",
            target_population="mci",
            prediction_horizon_years=3,
            evidence_items=[
                EvidenceNeedItem(
                    item_id="n",
                    description="cohort size",
                    tool_name="compute_cohort_size",
                    baseline_diagnosis=["mci"],
                ),
                EvidenceNeedItem(
                    item_id="design",
                    description="full design",
                    tool_name="suggest_modeling_design",
                    baseline_diagnosis=["mci"],
                ),
            ],
        )
        results = _execute_evidence_plan(plan, intent, catalog, backend)
        answer = _narrate_template(intent, results, [])
        assert _check_hallucination(answer, results), (
            "Template narration over evidence plan results must pass hallucination guard"
        )


# ---------------------------------------------------------------------------
# Mock-LLM end-to-end evidence planning test
# ---------------------------------------------------------------------------


class TestMockLlmEvidencePlanning:
    """Verify the full evidence-planning pipeline with a mock LLM client.

    The mock client simulates:
      - First call: plan_modeling_evidence tool_use (evidence planner)
      - Second call: text narration
    No live API key required.
    """

    def _minimal_plan_dict(self) -> dict:
        return {
            "modeling_goal": "predict MCI-to-AD conversion within 3 years",
            "target_population": "baseline MCI (NACCUDSD==3)",
            "target_population_filter": {"NACCUDSD": 3},
            "target_etiology": "ad",
            "label_definition_options": [
                "NACCALZD==1 (AD etiologic attribution)",
                "NACCUDSD==4 (all-cause dementia)",
            ],
            "prediction_horizon_years": 3,
            "candidate_modalities": ["cognition", "mri"],
            "candidate_feature_groups": ["demographics", "cognitive scores", "MRI volumes"],
            "evidence_items": [
                {
                    "item_id": "baseline_mci_n",
                    "description": "Baseline MCI cohort size",
                    "tool_name": "compute_cohort_size",
                    "baseline_diagnosis": ["mci"],
                    "priority": "required",
                },
                {
                    "item_id": "longitudinal_feas",
                    "description": "3-year follow-up availability for MCI cohort",
                    "tool_name": "assess_longitudinal_feasibility",
                    "baseline_diagnosis": ["mci"],
                    "horizon_years": 3,
                    "priority": "required",
                },
                {
                    "item_id": "design_overview",
                    "description": "Full modeling design with leakage blacklist",
                    "tool_name": "suggest_modeling_design",
                    "baseline_diagnosis": ["mci"],
                    "priority": "required",
                },
                {
                    "item_id": "cognition_only",
                    "description": "Cognition-only participant count",
                    "tool_name": "assess_modality_overlap",
                    "baseline_diagnosis": ["mci"],
                    "modalities": ["cognition"],
                    "priority": "required",
                },
                {
                    "item_id": "cog_mri",
                    "description": "Cognition+MRI overlap for multimodal modeling",
                    "tool_name": "assess_modality_overlap",
                    "baseline_diagnosis": ["mci"],
                    "modalities": ["cognition", "mri"],
                    "priority": "optional",
                },
            ],
            "leakage_checks": ["NACCUDSD", "NACCALZD"],
            "documentation_queries": [
                "MCI NACCUDSD diagnostic criteria",
                "NACCALZD etiologic AD attribution",
            ],
            "assumptions": ["Using Space A (NACC/ADSP-PHC). ADNI not joined."],
            "caveats": [
                "NACCALZD==1 is etiologic attribution, not confirmed neuropathology.",
                "NACCUDSD==4 is all-cause dementia.",
            ],
            "unresolved_items": [],
        }

    def test_mock_client_produces_modeling_design_plan(self, catalog, backend):
        """With a mock client, the evidence planning path populates trace.modeling_design_plan."""
        plan_dict = self._minimal_plan_dict()
        # Narration text uses only integers from tool results (3 = horizon)
        narration = (
            "## Cohort and Label Definition\n"
            "Baseline MCI participants are eligible.\n"
            "## Prediction Horizon\n"
            "3-year prediction horizon.\n"
            "## Recommended Feature Sets\n"
            "Cognition features are recommended.\n"
            "## Leakage Exclusions\n"
            "Diagnosis-defining variables are excluded.\n"
            "## Evaluation Design\n"
            "Participant-level split recommended.\n"
            "## Modeling Approach\n"
            "Logistic regression baseline.\n"
            "## Key Caveats\n"
            "NACCALZD==1 is etiologic AD attribution.\n"
        )
        client = _make_mock_combined_client(plan_dict, narration)

        response = run_agent(
            "How should we design a model for MCI-to-AD conversion?",
            catalog, backend,
            client=client,
        )

        assert response.trace is not None
        assert response.trace.modeling_design_plan is not None
        md_plan = response.trace.modeling_design_plan
        assert md_plan.prediction_horizon_years == 3
        assert "cognition" in md_plan.candidate_modalities
        assert len(md_plan.evidence_items) == 5

    def test_mock_client_calls_all_evidence_items(self, catalog, backend):
        """All 5 evidence items in the plan are executed, producing 5 tool results."""
        plan_dict = self._minimal_plan_dict()
        narration = "Baseline: 3 years. Cognition available."
        client = _make_mock_combined_client(plan_dict, narration)

        response = run_agent(
            "How should we design a model for MCI-to-AD conversion?",
            catalog, backend,
            client=client,
        )
        assert len(response.tool_results) == 5

    def test_mock_client_trace_plan_steps_from_evidence_items(self, catalog, backend):
        """trace.plan_steps must be populated from evidence item descriptions."""
        plan_dict = self._minimal_plan_dict()
        narration = "Cohort: 3 years."
        client = _make_mock_combined_client(plan_dict, narration)

        response = run_agent(
            "How should we design a model for MCI-to-AD conversion?",
            catalog, backend,
            client=client,
        )
        assert response.trace is not None
        assert len(response.trace.plan_steps) == 5
        # Each step should contain the item_id
        ids_in_steps = " ".join(response.trace.plan_steps)
        assert "baseline_mci_n" in ids_in_steps

    def test_mock_client_grounded_when_evidence_available(self, catalog, backend):
        plan_dict = self._minimal_plan_dict()
        narration = "Cohort size available. 3-year horizon."
        client = _make_mock_combined_client(plan_dict, narration)

        response = run_agent(
            "How should we design a model for MCI-to-AD conversion?",
            catalog, backend,
            client=client,
        )
        assert response.grounded

    def test_mock_client_hallucination_guard_passes(self, catalog, backend):
        """The mock narration uses only 3 (= horizon) which is in evidence."""
        plan_dict = self._minimal_plan_dict()
        narration = "Prediction horizon: 3 years. Cohort is eligible."
        client = _make_mock_combined_client(plan_dict, narration)

        response = run_agent(
            "How should we design a model for MCI-to-AD conversion?",
            catalog, backend,
            client=client,
        )
        assert response.hallucination_check_passed, (
            f"Hallucination guard failed.\nAnswer: {response.answer}\n"
            f"Tool results: {response.tool_results}"
        )

    def test_mock_client_modeling_plan_label_options(self, catalog, backend):
        """Label definition options from the plan are recorded in the trace."""
        plan_dict = self._minimal_plan_dict()
        narration = "Label: NACCALZD. Horizon: 3."
        client = _make_mock_combined_client(plan_dict, narration)

        response = run_agent(
            "How should we design a model for MCI-to-AD conversion?",
            catalog, backend,
            client=client,
        )
        md_plan = response.trace.modeling_design_plan
        assert md_plan is not None
        assert any("NACCALZD" in opt for opt in md_plan.label_definition_options)

    def test_planner_failure_falls_back_to_heuristic(self, catalog, backend):
        """If the evidence planner returns no tool_use block, fall back to
        heuristic suggest_modeling_design (modeling_design_plan stays None)."""
        from alz_agent.tools.deterministic import ModelingDesignResult

        client = _make_planner_failure_client("Cohort: 3 years.")

        response = run_agent(
            "How should we design a model for MCI-to-AD conversion?",
            catalog, backend,
            client=client,
        )

        # Heuristic fallback should still produce a grounded result
        assert response.grounded
        has_design = any(
            isinstance(tr.value, ModelingDesignResult)
            for tr in response.tool_results if tr.grounded
        )
        assert has_design, "Heuristic fallback must call suggest_modeling_design"
        # No modeling_design_plan in trace when the planner call failed
        assert response.trace.modeling_design_plan is None
