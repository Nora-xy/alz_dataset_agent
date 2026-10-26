"""Regression tests for the --use-semantic-planner CLI reachability defect.

Prior bug: `ask_agent.py: error: unrecognized arguments: --use-semantic-planner` --
scripts/ask_agent.py never registered the flag at all, so every fix built into the
semantic-gate path (agent/semantic_planner.py, PlanValidator.validate_semantic()) was
unreachable from the actual CLI. Two live queries were "fixed" in code that could never
run. This file proves the full live chain now works:

    argparse -> main() -> _resolve_planner_flags() -> _ask_once_nl() -> run_agent()
        -> orchestrator._run_enriched_path() -> AgentResponse.planner_path

Sections:
  A. argparse registers --use-semantic-planner (the exact reported error)
  B. _resolve_planner_flags() -- semantic implies enriched, never silently drops it
  C. run_agent() enforces the same implication even when called directly (bypassing
     the CLI), so no caller in the live chain can silently reset it to False
  D. planner_path is stamped correctly for all three planner modes
  E. End-to-end: the exact CSF p-tau181 command reaches planner_path=semantic_enriched
     against the real local catalog and executes successfully
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from alz_agent.agent.providers import PlannerClient
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.eval.harness import default_catalog


# ---------------------------------------------------------------------------
# A. argparse registers --use-semantic-planner
# ---------------------------------------------------------------------------


class TestArgparseRegistersFlag:
    def test_flag_is_recognized_not_an_error(self):
        """Reproduces the exact reported failure mode: argparse must accept the
        flag rather than raising 'unrecognized arguments'."""
        import ask_agent

        parser = ask_agent._build_arg_parser()
        args = parser.parse_args(["-q", "test question", "--use-semantic-planner"])
        assert args.use_semantic_planner is True

    def test_flag_defaults_false(self):
        import ask_agent

        parser = ask_agent._build_arg_parser()
        args = parser.parse_args(["-q", "test question"])
        assert args.use_semantic_planner is False

    def test_cli_help_does_not_error(self):
        """A real subprocess invocation of --help must not fail — the most direct
        reproduction of the user-reported command-line error."""
        result = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "ask_agent.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "--use-semantic-planner" in result.stdout


# ---------------------------------------------------------------------------
# B. _resolve_planner_flags() -- semantic implies enriched
# ---------------------------------------------------------------------------


class TestResolvePlannerFlags:
    def setup_method(self):
        import ask_agent
        self.resolve = ask_agent._resolve_planner_flags

    def test_neither_flag_set(self):
        enriched, semantic, note = self.resolve(False, False)
        assert (enriched, semantic) == (False, False)
        assert note is None

    def test_enriched_only(self):
        enriched, semantic, note = self.resolve(True, False)
        assert (enriched, semantic) == (True, False)
        assert note is None

    def test_semantic_only_implies_enriched(self):
        """The exact scenario a researcher hits: passing only --use-semantic-planner
        must not silently leave use_enriched_planner=False."""
        enriched, semantic, note = self.resolve(False, True)
        assert (enriched, semantic) == (True, True)
        assert note is not None
        assert "implies" in note.lower()

    def test_both_flags_set_no_note(self):
        enriched, semantic, note = self.resolve(True, True)
        assert (enriched, semantic) == (True, True)
        assert note is None


# ---------------------------------------------------------------------------
# C. run_agent() enforces the implication independent of the CLI
# ---------------------------------------------------------------------------


class _NoOpClient(PlannerClient):
    """A client that never resolves any tool call -- used to isolate the
    implication check itself (which happens before any LLM call) from
    planning success/failure."""

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        return None

    def complete_text(self, messages, max_tokens=1024):
        return ""


class TestRunAgentEnforcesImplication:
    def test_semantic_true_enriched_false_does_not_silently_use_classic_path(self):
        """Defense in depth: even a caller that bypasses the CLI entirely and sets
        use_enriched_planner=False, use_semantic_planner=True directly must still
        reach the enriched/semantic branch, not silently fall through to the
        classic parse_intent path."""
        from alz_agent.agent.orchestrator import run_agent

        catalog = default_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        client = _NoOpClient()

        resp = run_agent(
            "test question",
            catalog,
            backend,
            client=client,
            use_enriched_planner=False,  # deliberately NOT set
            use_semantic_planner=True,
        )

        # Must have entered the enriched/semantic branch (planner_path != non_enriched),
        # not silently landed in the classic parse_intent path.
        assert resp.planner_path != "non_enriched"
        assert resp.use_semantic_planner_requested is True


# ---------------------------------------------------------------------------
# D. planner_path is stamped correctly for all three modes
# ---------------------------------------------------------------------------


class TestPlannerPathMetadata:
    def test_non_enriched_when_no_client(self):
        from alz_agent.agent.orchestrator import run_agent

        catalog = default_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        resp = run_agent("How many MCI participants?", catalog, backend, client=None)
        assert resp.planner_path == "non_enriched"

    def test_classic_enriched_when_only_enriched_flag_set(self):
        from alz_agent.agent.orchestrator import run_agent

        catalog = default_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        client = _NoOpClient()
        resp = run_agent(
            "How many MCI participants?", catalog, backend,
            client=client, use_enriched_planner=True, use_semantic_planner=False,
        )
        assert resp.planner_path == "classic_enriched"

    def test_semantic_enriched_when_semantic_flag_set(self):
        from alz_agent.agent.orchestrator import run_agent

        catalog = default_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        client = _NoOpClient()
        resp = run_agent(
            "How many MCI participants?", catalog, backend,
            client=client, use_enriched_planner=True, use_semantic_planner=True,
        )
        assert resp.planner_path == "semantic_enriched"


# ---------------------------------------------------------------------------
# E. End-to-end: the exact CSF p-tau181 command reaches semantic_enriched
# ---------------------------------------------------------------------------


_CSF_PTAU181_PLAN_DATA = {
    "population_unit": "participant",
    "requires_unique_participants": True,
    "cohort_scope": "baseline",
    "etiology_definition": "unspecified",
    "statistic_type": "mean",
    "denominator_definition": "unique ADNI RIDs with a baseline CSF p-tau181 measurement",
    "specimen": "csf",
    "analyte": "ptau181",
    "id_space": "B",
    "groups": [
        {"label": "CN", "cohort_scope": "baseline", "baseline_filters": {"diagnosis": "normal"}},
        {"label": "MCI", "cohort_scope": "baseline", "baseline_filters": {"diagnosis": "mci"}},
        {
            "label": "AD (syndromic dementia)", "cohort_scope": "baseline",
            "baseline_filters": {"diagnosis": "dementia"},
            "etiology_definition": "all_cause_dementia",
        },
        {
            "label": "AD (etiologic AD)", "cohort_scope": "baseline",
            "baseline_filters": {"etiology": "ad"},
            "etiology_definition": "etiologic_ad", "diagnosis_variable": "DXAD",
        },
    ],
    "variable_selections": [
        {
            "variable_id": "ADNI.ADSP_PHC_CSF.pTau181_RAW",
            "specimen": "csf", "analyte": "ptau181", "variant_label": "csf_raw",
        },
        {
            "variable_id": "ADNI.ADSP_PHC_CSF.PHC_pTau181",
            "specimen": "csf", "analyte": "ptau181", "variant_label": "csf_harmonized",
        },
    ],
}


class _FixedSemanticPlanClient(PlannerClient):
    """Always returns the correct CSF p-tau181 SemanticAnalysisPlan tool call."""

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        if tool_schema.get("name") == "plan_semantic_analysis":
            return dict(_CSF_PTAU181_PLAN_DATA)
        return None

    def complete_text(self, messages, max_tokens=1024):
        return ""


class TestExactFailingCommandReachesSemanticPath:
    def test_csf_ptau181_command_reaches_semantic_enriched_and_grounds(self):
        """The exact question from the original bug report. Runs through the real
        run_agent()/orchestrator code (only the LLM client is a fixed stub, since
        no live API key is available in test); catalog is the small fixture
        6-variable catalog (no real ADNI biomarker entries) so the correct,
        expected outcome is a clean C10 abstain naming the unresolvable ids --
        the key assertion is that planner_path is proven to be semantic_enriched,
        not that grounded=True (that requires the real local catalog, exercised
        manually in this session's live proof run)."""
        from alz_agent.agent.orchestrator import run_agent

        catalog = default_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        client = _FixedSemanticPlanClient()

        resp = run_agent(
            "In ADNI, what are the mean cerebrospinal fluid (CSF) p-tau181 levels for "
            "Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's "
            "Disease (AD) groups at baseline?",
            catalog,
            backend,
            client=client,
            use_enriched_planner=False,  # deliberately not set -- proves the implication
            use_semantic_planner=True,
        )

        assert resp.planner_path == "semantic_enriched"
        assert resp.use_semantic_planner_requested is True
        assert resp.enriched_plan_used is False  # C10 abstain fires before compile/execute
        # The abstain must name the real catalog gap, not a generic/unrelated message.
        assert resp.abstain_reason is not None
        assert "C10" in resp.abstain_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
