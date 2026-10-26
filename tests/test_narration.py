"""Tests for the narration layer (P5).

Covers:
  - NarrationFact building from ExplanationContext
  - validate_narration: explanation_coverage, analysis_section_present,
    scope_validation, high_importance_coverage
  - narrate_with_llm: llm mode, fallback, retry
  - run_agent integration: final_narration_mode, final_narration_error
  - No disease/modality hardcoding in narration.py source
"""

from __future__ import annotations

import inspect
import re

import pytest

import alz_agent.agent.narration as narration_module
from alz_agent.agent.explanation import build_explanation_context
from alz_agent.agent.intent import QuestionType, StructuredIntent
from alz_agent.agent.narration import (
    build_narration_facts,
    narrate_with_llm,
    validate_narration,
)
from alz_agent.agent.orchestrator import run_agent
from alz_agent.agent.providers import PlannerClient
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    CatalogEntry,
    FilterSpec,
    IdSpace,
    LeakageTag,
    Modality,
    NarrationFact,
    SourceDataset,
    ToolResult,
)
from alz_agent.tools.deterministic import suggest_modeling_design

# ---------------------------------------------------------------------------
# Good analytical answer — ~150+ words, ≥2 keywords from _ANALYSIS_KEYWORDS,
# no integers (→ always passes the hallucination guard regardless of fixture values).
# ---------------------------------------------------------------------------

_GOOD_ANALYTICAL_ANSWER = (
    "The analysis of the cohort indicates sufficient data for the prediction task. "
    "Based on the feasibility assessment, the study design is viable given the available "
    "follow-up window. I recommend beginning with a regularized logistic regression model "
    "because the sample characteristics suggest that overly complex architectures would "
    "overfit. However, several important caveats apply: the leakage policy must be strictly "
    "respected, as certain variables directly define the outcome and must be excluded. "
    "The prediction horizon should be carefully considered in the study design. "
    "The outcome label definition represents all-cause dementia, which is an important "
    "caveat that researchers should note as a limitation. Consider documenting this "
    "distinction in any publication. The modality availability assessment shows which "
    "data sources are present for feature engineering. The strategy recommended here "
    "focuses on pre-index information only, which aligns with the stated objective. "
    "This approach is appropriate and indicates a well-grounded feasibility analysis. "
    "The leakage blacklist variables are excluded because they are directly collinear "
    "with the outcome or otherwise define it. This is a fundamental insight for design."
)

_STUB_KV_ANSWER = (
    "eligible_participants: 40\n"
    "prediction_horizon: years\n"
    "feasibility: sufficient\n"
    "leakage_blacklist: excluded\n"
)


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------


class MockLLMClient(PlannerClient):
    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [_GOOD_ANALYTICAL_ANSWER])
        self._call_count = 0
        self.calls: list[dict] = []

    def complete_text(self, messages, max_tokens=1024):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        idx = min(self._call_count, len(self._responses) - 1)
        text = self._responses[idx]
        self._call_count += 1
        return text

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        # Return None → intent parser falls back to heuristic; narration uses complete_text only.
        return None

    @property
    def call_count(self):
        return self._call_count


# ---------------------------------------------------------------------------
# Shared catalog / backend / fixtures
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


@pytest.fixture()
def mci_filter():
    return FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A)


@pytest.fixture()
def mci_intent():
    return StructuredIntent(
        question_type=QuestionType.MODELING_DESIGN,
        target_population=["mci"],
        candidate_label="conversion to dementia",
        horizon_years=3,
    )


@pytest.fixture()
def design_result(catalog, backend, mci_filter):
    return suggest_modeling_design(mci_filter, 3, catalog, backend)


@pytest.fixture()
def ctx(catalog, design_result, mci_filter):
    return build_explanation_context(
        [design_result],
        filter_spec=mci_filter,
        horizon=3,
        horizon_source="default",
        catalog=catalog,
    )


# ---------------------------------------------------------------------------
# T1: run_agent with LLM client → final_narration_mode == "llm"
# ---------------------------------------------------------------------------


class TestNarrationModeField:
    def test_llm_mode_sets_narration_mode_llm(self, catalog, backend, mci_intent):
        client = MockLLMClient([_GOOD_ANALYTICAL_ANSWER])
        response = run_agent(
            "How many MCI participants can we use to predict conversion over 3 years?",
            catalog,
            backend,
            intent=mci_intent,
            client=client,
        )
        assert response.final_narration_mode == "llm"

    # T2
    def test_no_client_sets_deterministic_fallback(self, catalog, backend, mci_intent):
        response = run_agent(
            "How many MCI participants can we use to predict conversion over 3 years?",
            catalog,
            backend,
            intent=mci_intent,
        )
        assert response.final_narration_mode == "deterministic_fallback"
        assert response.final_narration_error is not None
        assert "no_llm_client" in response.final_narration_error


# ---------------------------------------------------------------------------
# T3: LLM answer is not a key:value dump
# ---------------------------------------------------------------------------


class TestAnswerNotKVDump:
    def test_llm_answer_is_not_kv_dump(self, catalog, backend, mci_intent):
        client = MockLLMClient([_GOOD_ANALYTICAL_ANSWER])
        response = run_agent(
            "Suggest a modeling design for 3-year MCI-to-dementia prediction.",
            catalog,
            backend,
            intent=mci_intent,
            client=client,
        )
        answer = response.answer
        kv_matches = len(re.findall(r"^\s*[\w_]+\s*:\s*\S", answer, re.MULTILINE))
        total_lines = max(len(answer.splitlines()), 1)
        kv_ratio = kv_matches / total_lines
        assert kv_ratio <= 0.4, (
            f"Answer looks like a key:value dump (kv_ratio={kv_ratio:.2f})"
        )


# ---------------------------------------------------------------------------
# T4: Count facts produced by build_narration_facts have scope_label
# ---------------------------------------------------------------------------


class TestCountFactsHaveScopeLabel:
    def test_count_facts_have_scope_label(self, ctx, mci_filter):
        facts = build_narration_facts(ctx, filter_spec=mci_filter, horizon=3, horizon_source="default")
        count_facts = [f for f in facts if f.fact_type == "count" and f.value is not None]
        # At least some count facts should exist
        if count_facts:
            for f in count_facts:
                assert f.scope_label is not None, (
                    f"Count fact '{f.fact_id}' has value={f.value} but no scope_label"
                )


# ---------------------------------------------------------------------------
# T5: Feasibility subset count facts have denominator
# ---------------------------------------------------------------------------


class TestFeasibilityCountFactsHaveDenominator:
    def test_feasibility_count_facts_have_denominator(self, mci_filter):
        """Feasibility subset facts carry a narration-layer denominator derived from bucket sum.

        Uses a backend where all three follow-up buckets are above the suppression
        threshold (n_mci_short=15 >= 10), so the bucket sum is unambiguous and
        the narration layer can safely derive total_eligible = sufficient+insufficient+censored.
        The denominator is computed entirely inside the narration layer — no backend
        numeric fields are changed.
        """
        backend_full = build_fixture_backend(
            n_normal=20, n_mci=60,
            n_mci_long=20,     # sufficient=20  (>= threshold=10)
            n_mci_short=15,    # insufficient=15 (>= threshold=10)
            n_single_visit=15, # censored=15     (>= threshold=10)
            n_dementia=10, n_other=5,
            n_suppressed_group=0, n_mci_ad=5, n_dementia_ad=5,
            seed=7,
        )
        catalog = _make_catalog()
        design_result = suggest_modeling_design(mci_filter, 3, catalog, backend_full)
        ctx_full = build_explanation_context(
            [design_result],
            filter_spec=mci_filter,
            horizon=3,
            horizon_source="default",
            catalog=catalog,
        )
        facts = build_narration_facts(ctx_full, filter_spec=mci_filter, horizon=3, horizon_source="default")
        feasibility_subset = [
            f for f in facts
            if f.fact_id.startswith("feasibility_")
            and f.fact_id != "feasibility_total_eligible"
            and f.value is not None
        ]
        assert feasibility_subset, "Expected at least one non-suppressed feasibility subset fact"
        for f in feasibility_subset:
            assert f.denominator_value is not None, (
                f"Feasibility fact '{f.fact_id}' has value={f.value} but no denominator_value"
            )
            assert f.denominator_source == "derived_from_followup_bucket_sum", (
                f"Expected denominator_source='derived_from_followup_bucket_sum', "
                f"got {f.denominator_source!r}"
            )


# ---------------------------------------------------------------------------
# T6: scope_validation catches impossible subset (value > denominator)
# ---------------------------------------------------------------------------


class TestScopeValidation:
    def test_scope_validation_catches_impossible_subset(self):
        impossible_fact = NarrationFact(
            fact_id="test_count",
            fact_type="count",
            label="Test subset count",
            value=999,
            unit="participants",
            denominator_value=100,
            denominator_label="total eligible",
            source_tool="test",
        )
        result = validate_narration(
            "The test subset count is participants.",
            [impossible_fact],
            "feasibility",
        )
        assert result.scope_validation_passed is False
        assert any("scope_violation" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# T7: high_importance_coverage == 1.0 when both high-importance facts are mentioned
# ---------------------------------------------------------------------------


class TestHighImportanceCoverage:
    def test_high_importance_coverage_computed(self):
        f1 = NarrationFact(
            fact_id="f1", fact_type="metadata", label="Prediction horizon",
            value=3, unit="years", source_tool="test", importance="high",
        )
        f2 = NarrationFact(
            fact_id="f2", fact_type="metadata", label="Eligible participants",
            value=40, unit="participants", source_tool="test", importance="high",
        )
        # Answer mentions both labels
        answer = (
            "The prediction horizon is set to years. "
            "The eligible participants count meets the threshold."
        )
        result = validate_narration(answer, [f1, f2], "feasibility")
        assert result.high_importance_coverage == 1.0


# ---------------------------------------------------------------------------
# T8: Flat single-line answer fails analysis_section_present for modeling_design
# ---------------------------------------------------------------------------


class TestAnalysisSectionPresent:
    def test_analytical_type_requires_analysis_keywords(self):
        flat_answer = "The eligible cohort has participants available."
        f = NarrationFact(
            fact_id="eligible_n", fact_type="count", label="Eligible participants",
            value=40, unit="participants", source_tool="test",
        )
        result = validate_narration(flat_answer, [f], "modeling_design")
        assert result.analysis_section_present is False

    # T9
    def test_stub_kv_answer_fails_analysis_check(self):
        f = NarrationFact(
            fact_id="eligible_n", fact_type="count", label="Eligible participants",
            value=40, unit="participants", source_tool="test",
        )
        result = validate_narration(_STUB_KV_ANSWER, [f], "modeling_design")
        assert result.analysis_section_present is False


# ---------------------------------------------------------------------------
# T10: Retry attempted when first answer fails validation
# ---------------------------------------------------------------------------


class TestRetryOnValidationFailure:
    def test_retry_attempted_on_validation_failure(self, catalog, backend, mci_intent, ctx):
        # First call returns stub KV dump (fails analysis_section_present for modeling_design).
        # Second call returns good analytical answer.
        client = MockLLMClient([_STUB_KV_ANSWER, _GOOD_ANALYTICAL_ANSWER])
        facts = build_narration_facts(ctx, horizon=3, horizon_source="default")

        result = narrate_with_llm(
            question="Suggest a modeling design for 3-year MCI-to-dementia prediction.",
            intent=mci_intent,
            facts=facts,
            ctx=ctx,
            doc_evidence=[],
            client=client,
        )
        assert result.retry_attempted is True
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# T11: narration.py has no disease/modality hardcoding in non-comment code
# ---------------------------------------------------------------------------


class TestNoDiseaseHardcoding:
    def test_narration_module_has_no_disease_hardcoding(self):
        source = inspect.getsource(narration_module)
        # Strip comment lines
        non_comment_lines = [
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        ]
        code_body = "\n".join(non_comment_lines).lower()

        hardcoded_terms = ["mci", "alzheimer", "dementia", "cognition", "mri", "pet"]
        violations = [term for term in hardcoded_terms if term in code_body]
        assert not violations, (
            f"narration.py hardcodes disease/modality terms in non-comment code: {violations}"
        )


# ---------------------------------------------------------------------------
# T12: Leakage facts from modeling design carry fact_type == "leakage_warning"
# ---------------------------------------------------------------------------


class TestLeakageFactsType:
    def test_leakage_facts_carry_leakage_warning_type(self, ctx, mci_filter):
        facts = build_narration_facts(ctx, filter_spec=mci_filter, horizon=3, horizon_source="default")
        leakage_facts = [f for f in facts if f.fact_type == "leakage_warning"]
        assert leakage_facts, "Expected at least one leakage_warning fact from a modeling design ctx"
        for f in leakage_facts:
            assert isinstance(f.value, str), (
                f"Leakage fact '{f.fact_id}' should carry a string value (raw_code), got {type(f.value)}"
            )


# ---------------------------------------------------------------------------
# T13: Not-answerable ToolResult produces not_answerable fact
# ---------------------------------------------------------------------------


class TestNotAnswerableFacts:
    def test_not_answerable_facts_carry_correct_type(self, catalog):
        """A ctx with an abstained ToolResult should produce at least one not_answerable fact."""
        abstain_result = ToolResult.abstain("requested modality not present in catalog")
        backend = build_fixture_backend(
            n_mci=40, n_normal=10, n_dementia=10, n_other=5,
            n_mci_long=20, n_mci_short=5, n_single_visit=5,
            n_suppressed_group=0, n_mci_ad=5, n_dementia_ad=5,
            seed=99,
        )
        filter_spec = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A)
        ctx = build_explanation_context(
            [abstain_result],
            filter_spec=filter_spec,
            horizon=3,
            horizon_source="default",
            catalog=catalog,
        )
        facts = build_narration_facts(ctx, filter_spec=filter_spec, horizon=3, horizon_source="default")
        na_facts = [f for f in facts if f.fact_type == "not_answerable"]
        assert na_facts, (
            "Expected at least one not_answerable fact when ctx contains an abstain result; "
            f"got fact types: {[f.fact_type for f in facts]}"
        )


# ---------------------------------------------------------------------------
# T14–T21: Longitudinal Feasibility Narration — definitions, percentages, insights
# ---------------------------------------------------------------------------

# A good answer for longitudinal feasibility questions: defines all three categories,
# includes percentages, explains censored as horizon-independent, gives interpretation.
_GOOD_LONGITUDINAL_ANSWER = (
    "Among the 12,260 participants with baseline MCI, follow-up sufficiency "
    "varies substantially across the three horizons requested. "
    "Sufficient follow-up means a participant has enough observed time (follow-up span "
    "meeting or exceeding the horizon) to determine whether conversion occurred. "
    "Insufficient follow-up means the participant has multiple visits but the "
    "follow-up span is shorter than the horizon. "
    "Censored refers to participants with only a single visit; because no follow-up "
    "is recorded, conversion status cannot be determined for any horizon. "
    "This count is horizon-independent: the same single-visit participants are "
    "censored regardless of the specified horizon length.\n\n"
    "At 2 years, 6,632 (54.1%) participants have sufficient follow-up, "
    "1,686 (13.8%) are insufficient, and 3,942 (32.2%) are censored. "
    "At 3 years, sufficient drops to 5,162 (42.1%), insufficient rises to 3,156 (25.7%), "
    "and censored remains 3,942 (32.2%). "
    "At 5 years, only 3,057 (24.9%) have sufficient follow-up, while 5,261 (42.9%) "
    "are insufficient and 3,942 (32.2%) remain censored.\n\n"
    "These results suggest that a 2-year conversion label is the most feasible among "
    "the three horizons, with 6,632 participants having sufficient follow-up. "
    "I recommend prioritizing the 2-year horizon as the primary modeling target, "
    "with the 3-year horizon as a secondary option. The 5-year horizon is less "
    "practical because only 3,057 participants have sufficient follow-up, making "
    "it more susceptible to follow-up bias and smaller effective training size."
)

_RAW_ONLY_ANSWER = (
    "Computed results:\n"
    "baseline_mci_count: 12260\n"
    "mci_followup_by_horizon:\n"
    "  2yr: sufficient=6632, insufficient=1686, censored=3942\n"
    "  3yr: sufficient=5162, insufficient=3156, censored=3942\n"
    "  5yr: sufficient=3057, insufficient=5261, censored=3942\n"
)


def _make_minimal_fanout_execution_ctx():
    """Return a minimal ExecutionContext with a cohort count + fan-out result."""
    from alz_agent.agent.executor import DeterministicExecutor  # noqa: F401 — import check
    from alz_agent.backend.local_backend import (
        CENSORING_HORIZON_DEPENDENT,
        FOLLOWUP_STATUS_DEFINITIONS,
    )
    from alz_agent.schemas import DataEvidence, ExecutionContext, ToolResult

    ctx = ExecutionContext()
    ctx.results["baseline_mci_count"] = ToolResult.ok(
        12260,
        evidence=[DataEvidence(
            query_spec={"baseline_diagnosis": ["mci"]},
            aggregate_ref="compute_cohort_size: 12260 participants",
        )],
    )
    ctx.results["mci_followup_by_horizon"] = ToolResult.ok(
        {
            "2yr": {"sufficient": 6632, "insufficient": 1686, "censored": 3942},
            "3yr": {"sufficient": 5162, "insufficient": 3156, "censored": 3942},
            "5yr": {"sufficient": 3057, "insufficient": 5261, "censored": 3942},
            "_status_metadata": {
                "status_definitions": FOLLOWUP_STATUS_DEFINITIONS,
                "censoring_horizon_dependent": CENSORING_HORIZON_DEPENDENT,
            },
        },
        evidence=[DataEvidence(
            query_spec={"horizons": [2, 3, 5]},
            aggregate_ref="fan_out_longitudinal: all 3 horizons succeeded",
        )],
    )
    ctx.operations_run = ["baseline_mci_count", "mci_followup_by_horizon"]
    return ctx


class TestLongitudinalFeasibilityNarration:
    """Tests for longitudinal feasibility synthesis: definitions, percentages, insights, metadata."""

    # T14: fan-out result includes _status_metadata with all three definitions
    def test_fan_out_result_includes_status_metadata(self):
        from alz_agent.agent.executor import DeterministicExecutor
        from alz_agent.backend.fixture_backend import build_fixture_backend
        from alz_agent.schemas import FilterSpec, IdSpace

        backend = build_fixture_backend(
            n_mci=80, n_normal=20, n_dementia=20, n_other=10,
            n_mci_long=30, n_mci_short=20, n_single_visit=15,
            n_suppressed_group=0, n_mci_ad=10, n_dementia_ad=10, seed=1,
        )
        executor = DeterministicExecutor(_make_catalog(), backend)
        filter_spec = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A)

        result = executor._fan_out_longitudinal(
            {"horizons": [2, 3, 5]}, filter_spec
        )
        assert result.grounded
        assert "_status_metadata" in result.value
        meta = result.value["_status_metadata"]
        assert "status_definitions" in meta
        defs = meta["status_definitions"]
        assert "sufficient" in defs
        assert "insufficient" in defs
        assert "censored" in defs
        assert "censoring_horizon_dependent" in meta
        assert meta["censoring_horizon_dependent"] is False

    # T15: censored definitions explain horizon-independence
    def test_censored_definition_horizon_independent(self):
        from alz_agent.backend.local_backend import FOLLOWUP_STATUS_DEFINITIONS
        defn = FOLLOWUP_STATUS_DEFINITIONS["censored"].lower()
        assert "single" in defn or "one" in defn or "1 visit" in defn
        assert "horizon-independent" in defn or "independent" in defn

    # T16: synthesis user prompt contains STATUS DEFINITIONS and COMPUTED PERCENTAGES blocks
    def test_synthesis_user_prompt_contains_computed_blocks(self):
        from unittest.mock import MagicMock

        from alz_agent.agent.narration import _build_synthesis_user_prompt

        exec_ctx = _make_minimal_fanout_execution_ctx()
        plan = MagicMock()
        plan.researcher_audit_trace = ["Interpreted baseline as MCI (NACCUDSD=3)"]
        plan.requested_horizons = [2, 3, 5]
        validation = MagicMock()
        validation.soft_warnings = []

        prompt = _build_synthesis_user_prompt(
            "How many MCI participants have sufficient follow-up for 2, 3, 5 year horizons?",
            plan, validation, exec_ctx, [],
        )
        assert "STATUS DEFINITIONS" in prompt
        assert "sufficient" in prompt
        assert "insufficient" in prompt
        assert "censored" in prompt
        assert "horizon-independent" in prompt.lower() or "censoring_horizon_dependent=False" in prompt
        assert "COMPUTED PERCENTAGES" in prompt
        assert "%" in prompt
        assert "12,260" in prompt or "12260" in prompt

    # T17: _status_metadata is not printed as a raw key in the evidence dump
    def test_status_metadata_not_in_raw_dump(self):
        from unittest.mock import MagicMock

        from alz_agent.agent.narration import _build_synthesis_user_prompt

        exec_ctx = _make_minimal_fanout_execution_ctx()
        plan = MagicMock()
        plan.researcher_audit_trace = []
        plan.requested_horizons = [2, 3, 5]
        validation = MagicMock()
        validation.soft_warnings = []

        prompt = _build_synthesis_user_prompt(
            "Follow-up question", plan, validation, exec_ctx, [],
        )
        # _status_metadata should not appear as a raw key=value line
        assert "_status_metadata:" not in prompt

    # T18: status_definitions_present=True for a good longitudinal answer
    def test_status_definitions_present_for_good_answer(self):
        result = validate_narration(
            _GOOD_LONGITUDINAL_ANSWER, [], "feasibility"
        )
        assert result.status_definitions_present is True

    # T19: percentages_present=True for a good longitudinal answer
    def test_percentages_present_for_good_answer(self):
        result = validate_narration(
            _GOOD_LONGITUDINAL_ANSWER, [], "feasibility"
        )
        assert result.percentages_present is True

    # T20: insights_section_present=True for a good longitudinal answer
    def test_insights_section_present_for_good_answer(self):
        result = validate_narration(
            _GOOD_LONGITUDINAL_ANSWER, [], "feasibility"
        )
        assert result.insights_section_present is True

    # T21: censoring_definition_present=True for a good longitudinal answer
    def test_censoring_definition_present_for_good_answer(self):
        result = validate_narration(
            _GOOD_LONGITUDINAL_ANSWER, [], "feasibility"
        )
        assert result.censoring_definition_present is True

    # T22: raw_only answer fails validation with raw_only_answer issue
    def test_raw_only_answer_fails_validation(self):
        result = validate_narration(
            _RAW_ONLY_ANSWER, [], "feasibility"
        )
        assert result.passed is False
        assert any("raw_only_answer" in issue for issue in result.issues)

    # T23: good answer does NOT fail due to raw_only issue
    def test_good_answer_passes_raw_only_check(self):
        result = validate_narration(
            _GOOD_LONGITUDINAL_ANSWER, [], "feasibility"
        )
        assert not any("raw_only_answer" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# External literature claim guard tests
# ---------------------------------------------------------------------------


class TestCheckExternalClaims:
    """Tests for _check_external_claims() in hallucination_guard.py."""

    def _check(self, text: str, doc_evidence_count: int = 0):
        from alz_agent.agent.hallucination_guard import _check_external_claims
        return _check_external_claims(text, doc_evidence_count)

    def test_clean_answer_passes(self):
        """Answer with no external comparison claims passes."""
        text = "The mean age at first NACCALZD==1 visit was 74.81 years (n=25,639)."
        passes, violations = self._check(text)
        assert passes
        assert violations == []

    def test_aligns_with_literature_detected(self):
        """'aligns with ... literature' triggers violation."""
        text = "This result aligns with typical late-onset Alzheimer's disease literature."
        passes, violations = self._check(text)
        assert not passes
        assert len(violations) >= 1
        assert any("aligns with" in v.lower() for v in violations)

    def test_consistent_with_prior_studies_detected(self):
        """'consistent with prior studies' triggers violation."""
        text = "The finding is consistent with prior studies of AD onset age."
        passes, violations = self._check(text)
        assert not passes
        assert any("consistent with" in v.lower() for v in violations)

    def test_typical_in_the_literature_detected(self):
        """'typical in the literature' triggers violation."""
        text = "A mean onset age of 74 years is typical in the literature for late-onset AD."
        passes, violations = self._check(text)
        assert not passes

    def test_previously_reported_detected(self):
        """'previously reported' triggers violation."""
        text = "This age distribution was previously reported in the NACC cohort."
        passes, violations = self._check(text)
        assert not passes
        assert any("previously reported" in v.lower() for v in violations)

    def test_in_line_with_prior_findings_detected(self):
        """'in line with prior findings' triggers violation."""
        text = "The estimate is in line with prior findings from longitudinal AD cohorts."
        passes, violations = self._check(text)
        assert not passes

    def test_passes_when_doc_evidence_present(self):
        """External claims pass unconditionally when doc_evidence_count > 0."""
        text = "This result aligns with typical late-onset Alzheimer's disease literature."
        passes, violations = self._check(text, doc_evidence_count=1)
        assert passes
        assert violations == []

    def test_violations_list_populated_with_phrase(self):
        """Violation list contains the matched phrase (not empty)."""
        text = "Mean age of 74.81 years, consistent with prior published AD studies."
        passes, violations = self._check(text)
        assert not passes
        assert len(violations) == 1
        assert "consistent with" in violations[0].lower()

    def test_data_grounded_statement_passes(self):
        """'consistent with the data/filter' doesn't trigger (no knowledge-source keyword)."""
        text = "The result is consistent with the applied filter criteria."
        passes, violations = self._check(text)
        assert passes

    def test_multiple_violations_returned(self):
        """Two violations in the same text are both detected."""
        text = (
            "This aligns with the literature on AD. "
            "It was previously reported in large cohort studies."
        )
        passes, violations = self._check(text)
        assert not passes
        assert len(violations) >= 2

    def test_violation_phrase_capped_at_120_chars(self):
        """Each violation phrase is at most 120 characters."""
        long_clause = "aligns with " + "typical " * 30 + "literature"
        passes, violations = self._check(long_clause)
        assert not passes
        for v in violations:
            assert len(v) <= 120


class TestValidateNarrationExternalClaims:
    """validate_narration correctly applies check #6 (external claims)."""

    def test_external_claim_without_evidence_fails_validation(self):
        """External claim + no doc evidence → passed=False, external_claim_violation=True."""
        answer = (
            "The mean age at first AD diagnosis is 74.81 years. "
            "This aligns with typical late-onset Alzheimer's disease literature."
        )
        result = validate_narration(answer, [], "aggregate_stat", doc_evidence_count=0)
        assert not result.passed
        assert result.external_claim_violation
        assert any("external_claim_violation" in i for i in result.issues)

    def test_external_claim_with_evidence_passes(self):
        """External claim + doc_evidence_count > 0 → external_claim_violation=False."""
        answer = (
            "The mean age at first AD diagnosis is 74.81 years. "
            "This aligns with typical late-onset Alzheimer's disease literature."
        )
        result = validate_narration(answer, [], "aggregate_stat", doc_evidence_count=2)
        assert not result.external_claim_violation
        assert result.external_claim_phrases == []

    def test_clean_answer_no_external_claim_violation(self):
        """Clean answer → external_claim_violation=False, phrases empty."""
        answer = (
            "The mean age at first NACCALZD==1 visit was 74.81 years (n=25,639). "
            "No external literature comparison was performed in this run."
        )
        result = validate_narration(answer, [], "aggregate_stat", doc_evidence_count=0)
        assert not result.external_claim_violation
        assert result.external_claim_phrases == []

    def test_external_claim_phrases_populated(self):
        """external_claim_phrases carries the matched strings."""
        answer = "The result is consistent with prior published studies on AD onset."
        result = validate_narration(answer, [], "aggregate_stat", doc_evidence_count=0)
        assert result.external_claim_violation
        assert len(result.external_claim_phrases) >= 1

    def test_retry_prompt_contains_violation_phrase(self):
        """Correction prompt (used in retry) contains the violation phrase."""
        from alz_agent.agent.narration import _build_synthesis_external_claim_correction_prompt
        violations = ["aligns with typical late-onset Alzheimer's disease literature"]
        prompt = _build_synthesis_external_claim_correction_prompt("original answer", violations)
        assert "aligns with typical" in prompt
        assert "No external literature comparison was performed in this run" in prompt

    def test_narrate_with_llm_retries_on_external_claim_violation(self):
        """narrate_with_llm triggers a retry when first answer has an external claim."""
        from alz_agent.agent.intent import QuestionType, StructuredIntent

        bad_answer = (
            "The mean age is 74 years. "
            "This result aligns with typical late-onset Alzheimer's disease literature."
        )
        clean_answer = (
            "The mean age is 74 years (n=25639). "
            "No external literature comparison was performed in this run."
        )

        client = MockLLMClient(responses=[bad_answer, clean_answer])
        intent = StructuredIntent(
            question_type=QuestionType.AGGREGATE_STAT,
            target_population=[],
            candidate_label="mean age",
            id_space=IdSpace.A,
            stat_type="mean",
        )
        facts = [
            NarrationFact(
                fact_id="mean_age",
                fact_type="count",
                label="mean age",
                value=74,
                source_tool="compute_variable_stat",
            ),
            NarrationFact(
                fact_id="n",
                fact_type="count",
                label="n participants",
                value=25639,
                source_tool="compute_cohort_size",
            ),
        ]

        result = narrate_with_llm(
            question="Mean age at first AD diagnosis?",
            intent=intent,
            facts=facts,
            ctx=None,
            doc_evidence=[],  # no evidence → external claims prohibited
            client=client,
            model=None,
        )

        # Should have attempted a retry
        assert result.retry_attempted
        # After retry, external_claim_violation should be False (clean answer provided)
        if result.validation:
            assert not result.validation.external_claim_violation
