"""P8 tests: eval harness wrapper.

The harness itself (alz_agent.eval.harness) is runnable standalone as
``python -m alz_agent.eval.harness``; this file makes it part of the regular
pytest suite (invariant 7) and parametrizes each category's checks so a
regression in any single check shows up as its own failing test, not just an
aggregate report line.
"""

from __future__ import annotations

import pytest

from alz_agent.eval.golden_questions import GOLDEN_QUESTIONS
from alz_agent.eval.harness import (
    EvalReport,
    default_backend,
    default_catalog,
    run_eval,
)


@pytest.fixture(scope="module")
def report() -> EvalReport:
    return run_eval(default_catalog(), default_backend())


def test_report_has_all_four_categories(report):
    names = {c.category for c in report.categories}
    assert names == {
        "golden_questions",
        "number_traceability",
        "abstention",
        "leakage",
    }


def test_overall_report_passes(report):
    assert report.passed, f"\n{report.summary()}"


class TestGoldenQuestionsCategory:
    @pytest.mark.parametrize("gq", GOLDEN_QUESTIONS, ids=lambda gq: gq.id)
    def test_each_golden_question(self, report, gq):
        cat = report.category("golden_questions")
        check = next(c for c in cat.checks if c.name == gq.id)
        assert check.passed, check.detail

    def test_covers_feasible_infeasible_and_lookup_variants(self):
        ids = {gq.id for gq in GOLDEN_QUESTIONS}
        assert any("worked_target" in i for i in ids)
        assert any(not gq.expected_grounded for gq in GOLDEN_QUESTIONS)  # infeasible
        assert any(gq.expected_grounded for gq in GOLDEN_QUESTIONS)  # feasible
        assert any("lookup" in i for i in ids)


class TestNumberTraceabilityCategory:
    @pytest.mark.parametrize("gq", GOLDEN_QUESTIONS, ids=lambda gq: gq.id)
    def test_every_golden_answer_is_traceable(self, report, gq):
        cat = report.category("number_traceability")
        check = next(c for c in cat.checks if c.name == gq.id)
        assert check.passed, check.detail


class TestAbstentionCategory:
    @pytest.mark.parametrize(
        "name",
        ["absent_modality_combo", "uncovered_release", "suppressed_cells", "unverified_mapping"],
    )
    def test_each_abstention_scenario(self, report, name):
        cat = report.category("abstention")
        check = next(c for c in cat.checks if c.name == name)
        assert check.passed, check.detail


class TestLeakageCategory:
    def test_design_grounded(self, report):
        cat = report.category("leakage")
        check = next(c for c in cat.checks if c.name == "suggest_modeling_design_grounded")
        assert check.passed

    @pytest.mark.parametrize(
        "var_id",
        ["NACC.NACCUDSD", "NACC.CDRSUM", "ADSP_PHC.Cognition.PHC_MEM"],
    )
    def test_known_leaky_var_blacklisted_and_excluded(self, report, var_id):
        cat = report.category("leakage")
        check = next(c for c in cat.checks if c.name == f"blacklisted:{var_id}")
        assert check.passed, check.detail

    def test_clean_var_not_blacklisted(self, report):
        cat = report.category("leakage")
        check = next(c for c in cat.checks if c.name == "check_leakage_keeps_clean_var_clean")
        assert check.passed, check.detail


# ---------------------------------------------------------------------------
# Regression test for the gap this harness discovered on first run: the
# all-abstained orchestrator path hardcoded hallucination_check_passed=True
# without verifying it, and _collect_evidence_numbers ignored abstain_reason
# text entirely — so a backend's "Valid: ['1','2','3','4']" abstain message
# would have failed an offline re-check despite being tool-emitted, not
# model-generated.
# ---------------------------------------------------------------------------


class TestAbstainReasonDigitsAreTraceable:
    def test_all_abstained_response_passes_offline_recheck(self):
        from alz_agent.agent.intent import QuestionType, StructuredIntent
        from alz_agent.agent.orchestrator import _check_hallucination, run_agent
        from alz_agent.schemas import IdSpace

        intent = StructuredIntent(
            target_population=["not_a_real_diagnosis_xyz"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        response = run_agent(
            "...", default_catalog(), default_backend(), intent=intent
        )
        assert not response.grounded
        assert response.hallucination_check_passed
        # Re-verify offline rather than trusting the field — this is exactly
        # what the harness's number_traceability category does for golden
        # questions.
        assert _check_hallucination(response.answer, response.tool_results)
