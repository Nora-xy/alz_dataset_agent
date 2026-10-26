"""Regression tests for the CN/MCI-collapse follow-up fixes.

Root cause (reported live): DeepSeek's compiled semantic plan gave the CN and MCI
GroupDefinitions both an EMPTY baseline_filters — structurally indistinguishable —
so both compiled to an unfiltered count_eligible op and silently returned the entire
ADNI cohort (eligible_n=3711) for both "distinct" groups, with identical downstream
stats. grounded=True the whole time; nothing caught it.

Sections:
  A. End-to-end: the exact observed bug pattern (empty group filters) now aborts
     via run_agent(), not just at the isolated validator level.
  B. C14/C15 direct validator checks (unit-level, precise error content).
  C. Post-execution mutually-exclusive-group-collision invariant (defense in depth
     for filters that differ structurally but still collide at the backend).
  D. Mixed-diagnosis-basis caveat injection (clinical-status vs etiologic-AD groups
     must never be narrated as one ordered gradient).
  E. External-claims guard: unsupported "adequate for modeling" / speculative
     biological-explanation claims are rejected the same way literature claims are.
  F. Existing single-cohort / two-distinct-group behavior is unaffected.
"""

from __future__ import annotations

import pytest

from alz_agent.agent.hallucination_guard import _check_external_claims
from alz_agent.agent.orchestrator import (
    _append_caveat,
    _check_mutually_exclusive_group_collision,
    _mixed_diagnostic_basis_caveat,
)
from alz_agent.agent.plan_validator import PlanValidator
from alz_agent.agent.providers import PlannerClient
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.eval.harness import default_catalog
from alz_agent.schemas import (
    CohortScope,
    EtiologyDefinition,
    ExecutionContext,
    GroupDefinition,
    PopulationUnit,
    SemanticAnalysisPlan,
    SynthesisReport,
    SynthesisSection,
    ToolResult,
)


class _FixedPlanClient(PlannerClient):
    def __init__(self, plan_data: dict):
        self._plan_data = plan_data

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        if tool_schema.get("name") == "plan_semantic_analysis":
            return dict(self._plan_data)
        return None

    def complete_text(self, messages, max_tokens=1024):
        return ""


# ---------------------------------------------------------------------------
# A. End-to-end: the exact observed empty-group-filter bug now aborts
# ---------------------------------------------------------------------------


class TestExactReportedBugPatternNowAborts:
    def test_cn_mci_with_empty_baseline_filters_aborts_via_run_agent(self):
        """Reproduces DeepSeek's actual output shape: CN and MCI groups with no
        baseline_filters at all (the real cause of eligible_n=3711 for both).
        Must abstain (C14/C15), never reach grounded=True with a collapsed cohort."""
        from alz_agent.agent.orchestrator import run_agent

        bad_plan_data = {
            "population_unit": "participant",
            "requires_unique_participants": True,
            "cohort_scope": "baseline",
            "etiology_definition": "unspecified",
            "statistic_type": "mean",
            "denominator_definition": "unique ADNI RIDs with a baseline measurement",
            "id_space": "B",
            "groups": [
                {"label": "Cognitively Normal (CN)", "cohort_scope": "baseline"},
                {"label": "Mild Cognitive Impairment (MCI)", "cohort_scope": "baseline"},
            ],
            "statistic_variable": "",
        }
        client = _FixedPlanClient(bad_plan_data)
        catalog = default_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)

        resp = run_agent(
            "In ADNI, what are CN and MCI levels at baseline?",
            catalog, backend, client=client,
            use_enriched_planner=True, use_semantic_planner=True,
        )

        assert resp.grounded is False
        assert resp.planner_path == "semantic_enriched"
        assert resp.semantic_grounding_passed is False
        assert resp.enriched_plan_used is False  # aborted before compile/execute
        combined_errors = " ".join(resp.validation_hard_errors)
        assert "C14" in combined_errors or "C15" in combined_errors


# ---------------------------------------------------------------------------
# B. C14/C15 direct validator checks
# ---------------------------------------------------------------------------


class TestC14C15DirectChecks:
    def setup_method(self):
        self.v = PlanValidator([])

    def _plan(self, groups: list[GroupDefinition]) -> SemanticAnalysisPlan:
        return SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            denominator_definition="x",
            id_space="B",
            groups=groups,
        )

    def test_two_empty_filter_groups_fail_c14_and_c15(self):
        sem = self._plan([
            GroupDefinition(label="CN"),
            GroupDefinition(label="MCI"),
        ])
        res = self.v.validate_semantic(sem)
        assert res.is_executable is False
        text = " ".join(res.hard_errors)
        assert "C14" in text
        assert "C15" in text

    def test_distinct_nonempty_filters_pass(self):
        sem = self._plan([
            GroupDefinition(label="CN", baseline_filters={"diagnosis": "normal"}),
            GroupDefinition(label="MCI", baseline_filters={"diagnosis": "mci"}),
        ])
        res = self.v.validate_semantic(sem)
        assert not any(e.startswith("C14") or e.startswith("C15") for e in res.hard_errors)

    def test_single_group_never_triggers_c14_c15(self):
        """A lone group (len < 2) is never a 'multi-group comparison' — no check applies."""
        sem = self._plan([GroupDefinition(label="Everyone")])
        res = self.v.validate_semantic(sem)
        assert not any(e.startswith("C14") or e.startswith("C15") for e in res.hard_errors)

    def test_three_groups_two_identical_one_distinct(self):
        """Only the colliding pair is reported; the distinct third group is fine."""
        sem = self._plan([
            GroupDefinition(label="CN", baseline_filters={"diagnosis": "normal"}),
            GroupDefinition(label="CN duplicate", baseline_filters={"diagnosis": "normal"}),
            GroupDefinition(label="MCI", baseline_filters={"diagnosis": "mci"}),
        ])
        res = self.v.validate_semantic(sem)
        assert res.is_executable is False
        assert any("C14" in e and "CN" in e and "CN duplicate" in e for e in res.hard_errors)


# ---------------------------------------------------------------------------
# C. Post-execution mutually-exclusive-group-collision invariant
# ---------------------------------------------------------------------------


class TestPostExecutionCollisionInvariant:
    def _sem(self, groups: list[GroupDefinition]) -> SemanticAnalysisPlan:
        return SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            denominator_definition="x",
            id_space="B",
            groups=groups,
        )

    def test_identical_eligible_n_across_clinical_groups_is_flagged(self):
        sem = self._sem([
            GroupDefinition(label="CN", baseline_filters={"diagnosis": "normal"}),
            GroupDefinition(label="MCI", baseline_filters={"diagnosis": "mci"}),
        ])
        ctx = ExecutionContext()
        ctx.results["eligible_n_cn"] = ToolResult.ok(100)
        ctx.results["eligible_n_mci"] = ToolResult.ok(100)  # same value -> collision

        violations = _check_mutually_exclusive_group_collision(sem, ctx)
        assert len(violations) == 1
        assert "CN" in violations[0] and "MCI" in violations[0]

    def test_distinct_eligible_n_passes(self):
        sem = self._sem([
            GroupDefinition(label="CN", baseline_filters={"diagnosis": "normal"}),
            GroupDefinition(label="MCI", baseline_filters={"diagnosis": "mci"}),
        ])
        ctx = ExecutionContext()
        ctx.results["eligible_n_cn"] = ToolResult.ok(100)
        ctx.results["eligible_n_mci"] = ToolResult.ok(150)

        assert _check_mutually_exclusive_group_collision(sem, ctx) == []

    def test_etiologic_ad_group_excluded_from_collision_check(self):
        """An etiologic-AD group is explicitly allowed to overlap/collide with a
        clinical-status group's count -- it must never be flagged."""
        sem = self._sem([
            GroupDefinition(label="MCI", baseline_filters={"diagnosis": "mci"}),
            GroupDefinition(
                label="AD (etiologic AD)", baseline_filters={"etiology": "ad"},
                etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            ),
        ])
        ctx = ExecutionContext()
        ctx.results["eligible_n_mci"] = ToolResult.ok(100)
        ctx.results["eligible_n_ad_etiologic_ad"] = ToolResult.ok(100)  # same, but exempt basis

        assert _check_mutually_exclusive_group_collision(sem, ctx) == []

    def test_none_semantic_plan_returns_no_violations(self):
        ctx = ExecutionContext()
        assert _check_mutually_exclusive_group_collision(None, ctx) == []


# ---------------------------------------------------------------------------
# D. Mixed-diagnosis-basis caveat injection
# ---------------------------------------------------------------------------


class TestMixedBasisCaveatInjection:
    def _sem(self, groups: list[GroupDefinition]) -> SemanticAnalysisPlan:
        return SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            denominator_definition="x",
            id_space="B",
            groups=groups,
        )

    def test_mixed_basis_triggers_caveat(self):
        sem = self._sem([
            GroupDefinition(label="MCI", baseline_filters={"diagnosis": "mci"}),
            GroupDefinition(
                label="AD (etiologic AD)", baseline_filters={"etiology": "ad"},
                etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            ),
        ])
        caveat = _mixed_diagnostic_basis_caveat(sem)
        assert caveat is not None
        assert "not mutually exclusive" in caveat.lower()
        assert "gradient" in caveat.lower()

    def test_all_clinical_status_groups_no_caveat(self):
        sem = self._sem([
            GroupDefinition(label="CN", baseline_filters={"diagnosis": "normal"}),
            GroupDefinition(label="MCI", baseline_filters={"diagnosis": "mci"}),
            GroupDefinition(
                label="Dementia", baseline_filters={"diagnosis": "dementia"},
                etiology_definition=EtiologyDefinition.ALL_CAUSE_DEMENTIA,
            ),
        ])
        assert _mixed_diagnostic_basis_caveat(sem) is None

    def test_all_etiologic_groups_no_caveat(self):
        sem = self._sem([
            GroupDefinition(
                label="A", baseline_filters={"etiology": "ad"},
                etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            ),
            GroupDefinition(
                label="B", event_filters={"NACCALZD": 1},
                etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            ),
        ])
        assert _mixed_diagnostic_basis_caveat(sem) is None

    def test_append_caveat_creates_section_when_absent(self):
        report = SynthesisReport(question="q", sections=[], grounded=True)
        updated = _append_caveat(report, "test caveat text")
        assert len(updated.sections) == 1
        assert updated.sections[0].section_id == "caveats"
        assert "test caveat text" in updated.sections[0].content

    def test_append_caveat_appends_to_existing_section(self):
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(section_id="caveats", title="Caveats", content="Existing.")],
            grounded=True,
        )
        updated = _append_caveat(report, "new caveat")
        assert len(updated.sections) == 1
        assert "Existing." in updated.sections[0].content
        assert "new caveat" in updated.sections[0].content


# ---------------------------------------------------------------------------
# E. External-claims guard: unsupported adequacy / biological-explanation claims
# ---------------------------------------------------------------------------


class TestExternalClaimsGuardExtension:
    def test_adequate_for_modeling_rejected_with_no_doc_evidence(self):
        text = "The sample sizes are adequate for modeling, with 1,245 participants."
        passed, violations = _check_external_claims(text, doc_evidence_count=0)
        assert passed is False
        assert violations

    def test_sufficient_for_modeling_rejected(self):
        text = "This sample is sufficient for modeling purposes."
        passed, _ = _check_external_claims(text, doc_evidence_count=0)
        assert passed is False

    def test_may_reflect_speculative_explanation_rejected(self):
        text = (
            "The syndromic dementia group shows higher p-tau181 than the etiologic "
            "AD group, which may reflect the inclusion of more advanced cases."
        )
        passed, violations = _check_external_claims(text, doc_evidence_count=0)
        assert passed is False
        assert any("may reflect" in v.lower() for v in violations)

    def test_same_claims_pass_when_doc_evidence_present(self):
        text = "The sample sizes are adequate for modeling."
        passed, _ = _check_external_claims(text, doc_evidence_count=2)
        assert passed is True

    def test_plain_grounded_prose_passes(self):
        text = "For the CN group, mean CSF p-tau181 was 25.58 pg/mL (n=118)."
        passed, violations = _check_external_claims(text, doc_evidence_count=0)
        assert passed is True
        assert violations == []


# ---------------------------------------------------------------------------
# F. Existing behavior unaffected
# ---------------------------------------------------------------------------


class TestExistingBehaviorUnaffected:
    def test_single_cohort_plan_unaffected_by_c14_c15(self):
        """No groups at all (single-cohort path) — C14/C15 must never fire."""
        sem = SemanticAnalysisPlan(
            original_question="mean age at baseline",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            statistic_variable="NACCAGE",
            statistic_type="mean",
            denominator_definition="unique NACC participants at baseline",
        )
        v = PlanValidator([])
        res = v.validate_semantic(sem)
        assert not any(e.startswith("C14") or e.startswith("C15") for e in res.hard_errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
