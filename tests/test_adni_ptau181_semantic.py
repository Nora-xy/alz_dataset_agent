"""Regression tests for the ADNI CSF p-tau181 semantic-planner fix.

Covers the architecture correction: the LLM planner resolves semantic ambiguity into
explicit slots (groups, variable_selections, specimen, analyte); the validator only
verifies those slots exist / are mutually consistent / satisfy the question; the
executor and compile_to_enriched remain interpretation-free.

Sections:
  A. catalog_context — variable_id is now visible (root cause of the original bug)
  B. compile_to_enriched — groups x variable_selections fan-out
  C. PlanValidator.validate_semantic — C10 (catalog existence), C11 (specimen/analyte
     consistency), C13 (diagnosis-domain: "ad" is an etiology, not a diagnosis label)
  D. End-to-end via run_agent with a fake LLM client
  E. Executor discovery-abstention (explore_catalog abstains -> downstream ops blocked)
  F. Regression: existing NACC/ADNI single-cohort behavior unchanged
"""

from __future__ import annotations

import pytest

from alz_agent.agent.catalog_context import _build_catalog_context
from alz_agent.agent.executor import DeterministicExecutor
from alz_agent.agent.plan_validator import PlanValidator
from alz_agent.agent.providers import PlannerClient
from alz_agent.agent.semantic_planner import compile_to_enriched
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    CatalogEntry,
    CohortScope,
    EnrichedAnalysisPlan,
    EtiologyDefinition,
    GroupDefinition,
    IdSpace,
    Modality,
    PlanOperation,
    PopulationUnit,
    SemanticAnalysisPlan,
    SourceDataset,
    VariableSelection,
)


# ---------------------------------------------------------------------------
# Shared fixture: realistic ADNI CSF/plasma p-tau181 catalog (mirrors the real
# adni_parser tagging: family='ptau181', variant in {csf_raw, csf_harmonized,
# plasma_raw, plasma_harmonized}). Same field name ("pTau181_RAW") deliberately
# appears in two different tables to reproduce the real ambiguity.
# ---------------------------------------------------------------------------


def _make_ptau181_catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            variable_id="ADNI.ADSP_PHC_CSF.pTau181_RAW",
            concept="ptau181_raw",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="pTau181_RAW",
            release_version="R3",
            modality=Modality.FLUID_BIOMARKER,
            family="ptau181",
            variant="csf_raw",
        ),
        CatalogEntry(
            variable_id="ADNI.ADSP_PHC_CSF.PHC_pTau181",
            concept="phc_ptau181",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="PHC_pTau181",
            release_version="R3",
            modality=Modality.FLUID_BIOMARKER,
            family="ptau181",
            variant="csf_harmonized",
        ),
        CatalogEntry(
            variable_id="ADNI.ADSP_PHC_PLASMA.pTau181_RAW",
            concept="ptau181_raw",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="pTau181_RAW",
            release_version="R3",
            modality=Modality.FLUID_BIOMARKER,
            family="ptau181",
            variant="plasma_raw",
        ),
        CatalogEntry(
            variable_id="ADNI.ADSP_PHC_PLASMA.PHC_pTau181",
            concept="phc_ptau181",
            source_dataset=SourceDataset.ADNI,
            id_space=IdSpace.B,
            raw_code="PHC_pTau181",
            release_version="R3",
            modality=Modality.FLUID_BIOMARKER,
            family="ptau181",
            variant="plasma_harmonized",
        ),
    ]


def _correct_csf_ptau181_semantic_plan() -> SemanticAnalysisPlan:
    """The plan a correctly-instructed LLM should produce for the failing query.

    Two AD groups (syndromic + etiologic) per the "report both" decision; two CSF
    variants (raw + harmonized) kept separate; plasma variants NOT selected.
    """
    return SemanticAnalysisPlan(
        original_question=(
            "In ADNI, what are the mean cerebrospinal fluid (CSF) p-tau181 levels for "
            "Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's "
            "Disease (AD) groups at baseline?"
        ),
        population_unit=PopulationUnit.PARTICIPANT,
        requires_unique_participants=True,
        cohort_scope=CohortScope.BASELINE,
        etiology_definition=EtiologyDefinition.UNSPECIFIED,
        statistic_type="mean",
        denominator_definition="unique ADNI RIDs with a baseline CSF p-tau181 measurement",
        specimen="csf",
        analyte="ptau181",
        id_space="B",
        groups=[
            GroupDefinition(
                label="CN",
                cohort_scope=CohortScope.BASELINE,
                baseline_filters={"diagnosis": "normal"},
                etiology_definition=EtiologyDefinition.UNSPECIFIED,
            ),
            GroupDefinition(
                label="MCI",
                cohort_scope=CohortScope.BASELINE,
                baseline_filters={"diagnosis": "mci"},
                etiology_definition=EtiologyDefinition.UNSPECIFIED,
            ),
            GroupDefinition(
                label="AD (syndromic dementia)",
                cohort_scope=CohortScope.BASELINE,
                baseline_filters={"diagnosis": "dementia"},
                etiology_definition=EtiologyDefinition.ALL_CAUSE_DEMENTIA,
            ),
            GroupDefinition(
                label="AD (etiologic AD)",
                cohort_scope=CohortScope.BASELINE,
                baseline_filters={"etiology": "ad"},
                etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
                diagnosis_variable="DXAD",
            ),
        ],
        variable_selections=[
            VariableSelection(
                variable_id="ADNI.ADSP_PHC_CSF.pTau181_RAW",
                specimen="csf",
                analyte="ptau181",
                variant_label="csf_raw",
            ),
            VariableSelection(
                variable_id="ADNI.ADSP_PHC_CSF.PHC_pTau181",
                specimen="csf",
                analyte="ptau181",
                variant_label="csf_harmonized",
            ),
        ],
    )


class _FakePlannerClient(PlannerClient):
    """Returns canned tool-invocation results from a queue, like the semantic-planner tests."""

    def __init__(self, invoke_tool_returns: list[dict | None]) -> None:
        self._returns = list(invoke_tool_returns)
        self._call_count = 0

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        if self._call_count < len(self._returns):
            val = self._returns[self._call_count]
            self._call_count += 1
            return val
        return None

    def complete_text(self, messages, max_tokens=1024):
        return ""


# ---------------------------------------------------------------------------
# A. catalog_context — variable_id is now visible (the actual root cause fix)
# ---------------------------------------------------------------------------


class TestCatalogContextShowsVariableId:
    def test_context_table_header_includes_variable_id_and_raw_code(self):
        ctx = _build_catalog_context("csf ptau181", _make_ptau181_catalog())
        header = ctx.splitlines()[0]
        assert "variable_id" in header
        assert "raw_code" in header

    def test_context_rows_show_fully_qualified_adni_id(self):
        """Before the fix, only the bare field name was shown — the LLM could not
        distinguish the CSF table from the PLASMA table for the same field name."""
        ctx = _build_catalog_context("csf ptau181", _make_ptau181_catalog())
        assert "ADNI.ADSP_PHC_CSF.pTau181_RAW" in ctx
        assert "ADNI.ADSP_PHC_PLASMA.pTau181_RAW" in ctx
        # A fabricated table name must not appear anywhere the LLM could copy from
        assert "UPENNBIOMK" not in ctx or "ADNI.UPENNBIOMK" not in ctx


# ---------------------------------------------------------------------------
# B. compile_to_enriched — groups x variable_selections fan-out
# ---------------------------------------------------------------------------


class TestGroupsAndVariableSelectionsFanOut:
    def test_fan_out_produces_group_times_variant_ops(self):
        sem = _correct_csf_ptau181_semantic_plan()
        enriched = compile_to_enriched(sem, sem.original_question)

        stat_ops = [op for op in enriched.operations if op.op_type == "variable_stat"]
        count_ops = [op for op in enriched.operations if op.op_type == "count_eligible"]

        # 4 groups x 2 variants = 8 stat ops; 4 groups = 4 count ops
        assert len(stat_ops) == 8
        assert len(count_ops) == 4

    def test_no_variable_id_is_shortened_or_invented(self):
        """Every compiled variable_id must be exactly one of the two selected ids —
        never truncated to a bare field name, never carrying an invented table."""
        sem = _correct_csf_ptau181_semantic_plan()
        enriched = compile_to_enriched(sem, sem.original_question)
        stat_ops = [op for op in enriched.operations if op.op_type == "variable_stat"]
        seen_ids = {op.parameters["variable_id"] for op in stat_ops}
        assert seen_ids == {"ADNI.ADSP_PHC_CSF.pTau181_RAW", "ADNI.ADSP_PHC_CSF.PHC_pTau181"}

    def test_csf_variants_never_pooled_with_plasma(self):
        """Only the two CSF variable_selections were chosen — plasma ids never appear,
        even though they share the same catalog family/raw_code."""
        sem = _correct_csf_ptau181_semantic_plan()
        enriched = compile_to_enriched(sem, sem.original_question)
        all_var_ids = {
            op.parameters.get("variable_id")
            for op in enriched.operations
            if op.op_type == "variable_stat"
        }
        assert not any("PLASMA" in vid for vid in all_var_ids)

    def test_ad_syndromic_and_etiologic_reported_as_separate_groups(self):
        sem = _correct_csf_ptau181_semantic_plan()
        enriched = compile_to_enriched(sem, sem.original_question)
        count_ops = [op for op in enriched.operations if op.op_type == "count_eligible"]
        labels_in_descriptions = " ".join(op.description for op in count_ops)
        assert "AD (syndromic dementia)" in labels_in_descriptions
        assert "AD (etiologic AD)" in labels_in_descriptions
        # And their filters route to genuinely different bases
        syndromic_op = next(op for op in count_ops if "syndromic" in op.description)
        etiologic_op = next(op for op in count_ops if "etiologic" in op.description)
        assert syndromic_op.parameters["filter"].get("diagnosis") == "dementia"
        assert etiologic_op.parameters["filter"].get("etiology") == "ad"

    def test_exact_failing_query_produces_no_age_at_event_operation(self):
        """Required regression: the exact failing query's compiled plan must contain
        no age-related operation — confirms no contamination from a prior query."""
        sem = _correct_csf_ptau181_semantic_plan()
        enriched = compile_to_enriched(sem, sem.original_question)
        for op in enriched.operations:
            assert op.parameters.get("variable_id") != "ADNI.DERIVED.age_at_event"
        assert "age" not in " ".join(op.description.lower() for op in enriched.operations)

    def test_empty_groups_and_selections_preserve_single_cohort_behavior(self):
        """Backward compatibility: when groups/variable_selections are empty, behavior
        is byte-identical to the original single-cohort/single-variable compiler."""
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
        enriched = compile_to_enriched(sem, sem.original_question)
        stat_ops = [op for op in enriched.operations if op.op_type == "variable_stat"]
        count_ops = [op for op in enriched.operations if op.op_type == "count_eligible"]
        assert len(stat_ops) == 1
        assert len(count_ops) == 1
        assert stat_ops[0].output_name == "mean_naccage"
        assert count_ops[0].output_name == "eligible_n"


# ---------------------------------------------------------------------------
# C. PlanValidator.validate_semantic — C10, C11, C13
# ---------------------------------------------------------------------------


class TestCatalogExistenceAndConsistencyChecks:
    def setup_method(self):
        self.v = PlanValidator(_make_ptau181_catalog())

    def test_correct_plan_passes_all_new_checks(self):
        sem = _correct_csf_ptau181_semantic_plan()
        res = self.v.validate_semantic(sem)
        assert res.is_executable is True, res.hard_errors

    def test_invented_table_name_is_blocked_without_shortening(self):
        """C10: 'ADNI.UPENNBIOMK.PTAU181_RAW' does not exist in the catalog (the real
        tables are UPENNBIOMK_MASTER / UPENNBIOMK_ROCHE_ELECSYS) — must be rejected,
        and the rejection must name the exact invented id, not a shortened one."""
        sem = _correct_csf_ptau181_semantic_plan().model_copy(update={
            "variable_selections": [
                VariableSelection(
                    variable_id="ADNI.UPENNBIOMK.PTAU181_RAW",
                    specimen="csf",
                    analyte="ptau181",
                    variant_label="upenn",
                ),
            ],
        })
        res = self.v.validate_semantic(sem)
        assert res.is_executable is False
        assert any("C10" in e and "ADNI.UPENNBIOMK.PTAU181_RAW" in e for e in res.hard_errors)

    def test_bare_field_name_ambiguous_across_tables_is_blocked(self):
        """C10: a bare 'ADNI.pTau181_RAW'-shaped id (field only, no real table) must not
        silently resolve to either the CSF or PLASMA table — it simply isn't in the
        catalog verbatim, so it is rejected rather than guessed."""
        sem = _correct_csf_ptau181_semantic_plan().model_copy(update={
            "variable_selections": [
                VariableSelection(variable_id="ADNI.pTau181_RAW", variant_label="ambiguous"),
            ],
        })
        res = self.v.validate_semantic(sem)
        assert res.is_executable is False
        assert any("C10" in e for e in res.hard_errors)

    def test_plasma_cannot_satisfy_csf_declared_specimen(self):
        """C11: declaring specimen='csf' for a variable whose catalog variant is
        'plasma_raw' must be rejected — the LLM's claim is checked against catalog
        metadata, not trusted."""
        sem = _correct_csf_ptau181_semantic_plan().model_copy(update={
            "variable_selections": [
                VariableSelection(
                    variable_id="ADNI.ADSP_PHC_PLASMA.pTau181_RAW",
                    specimen="csf",  # WRONG — this variable's variant is plasma_raw
                    analyte="ptau181",
                    variant_label="plasma_raw",
                ),
            ],
        })
        res = self.v.validate_semantic(sem)
        assert res.is_executable is False
        assert any("C11" in e for e in res.hard_errors)

    def test_diagnosis_domain_rejects_ad_as_a_diagnosis_label(self):
        """C13: 'ad' must never appear as a 'diagnosis' filter value — that is an
        etiology attribution, not a cognitive-status label. This is the generic rule
        that prevents raw 'ad' from ever reaching the backend's diagnosis-status map."""
        bad_group = GroupDefinition(
            label="AD",
            cohort_scope=CohortScope.BASELINE,
            baseline_filters={"diagnosis": "ad"},  # WRONG
            etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            diagnosis_variable="DXAD",
        )
        sem = _correct_csf_ptau181_semantic_plan().model_copy(update={"groups": [bad_group]})
        res = self.v.validate_semantic(sem)
        assert res.is_executable is False
        assert any("C13" in e and "'ad'" in e for e in res.hard_errors)

    def test_diagnosis_domain_check_applies_to_top_level_plan_too(self):
        """C13 also fires for the non-group (single-cohort) path."""
        sem = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            baseline_filters={"diagnosis": "ad"},
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
        )
        res = self.v.validate_semantic(sem)
        assert any("C13" in e for e in res.hard_errors)

    def test_catalog_existence_skipped_when_catalog_empty(self):
        """C10/C11 must not hard-fail unit tests that validate without a full catalog —
        preserves existing test_semantic_planner.py behavior."""
        v = PlanValidator([])
        sem = _correct_csf_ptau181_semantic_plan()
        res = v.validate_semantic(sem)
        assert not any(e.startswith("C10") or e.startswith("C11") for e in res.hard_errors)

    def test_group_first_event_coherence_checked_per_group(self):
        """C1/C2b apply per-group when groups is populated — a group with
        cohort_scope=first_event and empty event_filters must fail C1."""
        bad_group = GroupDefinition(
            label="AD (bad)",
            cohort_scope=CohortScope.FIRST_EVENT,
            event_filters={},  # empty -> C1 violation
            event_anchor=None,
        )
        sem = _correct_csf_ptau181_semantic_plan().model_copy(update={"groups": [bad_group]})
        res = self.v.validate_semantic(sem)
        assert res.is_executable is False
        assert any("C1" in e and "AD (bad)" in e for e in res.hard_errors)


# ---------------------------------------------------------------------------
# D. End-to-end via run_agent with a fake LLM client
# ---------------------------------------------------------------------------


class TestEndToEndSemanticPlannerPath:
    def test_invented_variable_id_produces_clean_abstain_not_table_omitted(self):
        """The full pipeline: an LLM that invents a table name must produce a clean,
        specific abstain message naming the bad id — never a generic backend
        'table omitted' message, since C10 blocks it before execution ever runs."""
        from alz_agent.agent.orchestrator import run_agent

        bad_plan_data = {
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
            ],
            "variable_selections": [
                {"variable_id": "ADNI.UPENNBIOMK.PTAU181_RAW", "specimen": "csf", "analyte": "ptau181"},
            ],
        }
        client = _FakePlannerClient([bad_plan_data, bad_plan_data])  # same bad plan on repair
        catalog = _make_ptau181_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)

        resp = run_agent(
            "In ADNI, what are the mean CSF p-tau181 levels for CN, MCI, and AD at baseline?",
            catalog,
            backend,
            client=client,
            use_enriched_planner=True,
            use_semantic_planner=True,
        )

        assert resp.grounded is False
        assert resp.question_answered is False
        assert resp.semantic_grounding_passed is False
        assert resp.requested_measure_resolved is False
        assert "ADNI.UPENNBIOMK.PTAU181_RAW" in resp.answer
        assert "table" not in resp.answer.lower() or "omitted" not in resp.answer.lower()


# ---------------------------------------------------------------------------
# E. Executor discovery-abstention (explore_catalog abstains -> ops blocked)
# ---------------------------------------------------------------------------


class TestExecutorBlocksOnFailedDiscovery:
    def test_explore_catalog_abstention_blocks_downstream_ops(self):
        """explore_catalog abstains (grounded=False) when nothing matches — this must
        block all downstream ops rather than let them silently substitute an unrelated
        result. Regression for the dead-code bug: the executor previously keyed off
        grounded=True+empty-list, which explore_catalog never actually returns."""
        catalog = _make_ptau181_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        executor = DeterministicExecutor(catalog, backend)

        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="explore_catalog",
                    parameters={"keyword": "totally_nonexistent_concept_zzz"},
                    output_name="discover",
                ),
                PlanOperation(
                    op_type="count_eligible",
                    parameters={"filter": {"diagnosis": "normal"}},
                    output_name="fallback_count",
                ),
            ],
            id_space="A",
        )
        ctx = executor.execute(plan)

        discover_result = ctx.results.get("discover")
        assert discover_result is not None
        assert discover_result.grounded is False  # explore_catalog abstains, not empty-grounded

        assert "fallback_count" not in ctx.operations_run
        assert any("No catalog entries matched" in e for e in ctx.errors)


# ---------------------------------------------------------------------------
# F. Regression: existing NACC single-cohort behavior fully unchanged
# ---------------------------------------------------------------------------


class TestExistingBehaviorUnchanged:
    def test_nacc_first_event_plan_still_compiles_identically(self):
        """The canonical 'mean age at first etiologic AD diagnosis' plan from
        test_semantic_planner.py must compile exactly as before."""
        sem = SemanticAnalysisPlan(
            original_question="mean age at first etiologic AD diagnosis",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.FIRST_EVENT,
            baseline_filters={},
            event_filters={"NACCALZD": 1},
            event_anchor="first_matching_event_visit",
            etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            diagnosis_variable="NACCALZD",
            statistic_variable="NACCAGE",
            statistic_type="mean",
            denominator_definition="unique NACC participants with NACCALZD=1 on any visit",
        )
        enriched = compile_to_enriched(sem, sem.original_question)
        stat_ops = [op for op in enriched.operations if op.op_type == "variable_stat"]
        assert len(stat_ops) == 1
        assert stat_ops[0].parameters["variable_id"] == "NACC.NACCAGE"
        assert stat_ops[0].parameters["filter"]["event_etiology"] == "ad"

    def test_naccage_resolves_via_raw_code_fallback_in_space_a(self):
        """C10 must accept the classic 'NACCAGE' bare raw_code (no dataset prefix) for
        Space A, since NACC columns are inherently unambiguous — this preserves every
        existing single-cohort NACC test that never used a fully-qualified variable_id."""
        catalog = [
            CatalogEntry(
                variable_id="nacc_age",
                concept="age at visit",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="NACCAGE",
                release_version="v70",
                modality=Modality.DEMOGRAPHIC,
            ),
        ]
        v = PlanValidator(catalog)
        sem = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            statistic_variable="NACCAGE",
            statistic_type="mean",
            denominator_definition="unique NACC participants at baseline",
        )
        res = v.validate_semantic(sem)
        assert not any(e.startswith("C10") for e in res.hard_errors)


# ---------------------------------------------------------------------------
# G. group_comparison misuse for 3+ named groups (classic enriched-planner path)
#
# Live-observed failure: "op 'ptau181_group_comparison' (group_comparison): missing
# required parameter(s): ['group_variable_id', 'outcome_variable_id']." — the classic
# (non-semantic) enriched planner chose op_type='group_comparison' (a binary Welch
# t-test template) for a CN/MCI/AD (3-group) comparison. The repair pass correctly
# refused to force a nonsensical binary group_variable_id, so validation legitimately
# failed — but the terminal message gave no explanation of *why*. Root fix: the tool
# schema had zero guidance text for group_comparison at all (a bare enum entry), so
# the LLM had no way to know it was the wrong tool. Fixed by (1) adding explicit
# op_type-choice guidance in both the tool-use schema and the JSON-fallback prompt,
# and (2) appending a generic, dataset-agnostic clarification to the abstain message
# whenever a group_comparison op is still missing its ids after the repair attempt.
# ---------------------------------------------------------------------------


class TestGroupComparisonMisuseForMultiGroupQuestions:
    def test_tool_schema_warns_against_group_comparison_for_3plus_groups(self):
        """The tool-use schema's op_type enum description must explain the binary-only
        constraint and point the LLM at the count_eligible+variable_stat alternative."""
        from alz_agent.agent.llm_intent import _ENRICHED_PLAN_TOOL

        op_type_desc = (
            _ENRICHED_PLAN_TOOL["input_schema"]["properties"]["operations"]
            ["items"]["properties"]["op_type"]["description"]
        )
        assert "group_comparison" in op_type_desc
        assert "EXACTLY TWO" in op_type_desc
        assert "THREE OR MORE" in op_type_desc or "3+" in op_type_desc

    def test_json_fallback_prompt_warns_against_group_comparison_misuse(self):
        """The JSON-fallback prompt (used when tool-use isn't supported) carries the
        same warning — both LLM entry points must be covered, not just one."""
        from alz_agent.agent.llm_intent import _plan_enriched_json_fallback

        prompt_source = _plan_enriched_json_fallback.__code__.co_consts
        prompt_text = "".join(c for c in prompt_source if isinstance(c, str))
        assert "group_comparison" in prompt_text
        assert "EXACTLY TWO" in prompt_text

    def test_unrepaired_group_comparison_error_gets_actionable_explanation(self):
        """End-to-end: an LLM that emits a group_comparison op for a 3-group ADNI
        biomarker comparison (missing ids, unresolvable since there's no valid binary
        group_variable_id) must abstain with a clear explanation of the mismatch,
        not just the bare 'missing required parameter' message."""
        from alz_agent.agent.orchestrator import run_agent

        bad_plan_data = {
            "research_intent_summary": "compare CSF p-tau181 across CN/MCI/AD",
            "researcher_audit_trace": ["Interpreted as a 3-group comparison."],
            "operations": [
                {
                    "op_type": "group_comparison",
                    "parameters": {},
                    "output_name": "ptau181_group_comparison",
                    "description": "Compare p-tau181 across CN, MCI, AD.",
                }
            ],
            "requires_backend_data": True,
            "requested_horizons": [],
            "baseline_definition": {},
            "outcome_definition": "",
            "id_space": "B",
        }
        client = _FakePlannerClient([bad_plan_data])
        catalog = _make_ptau181_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)

        resp = run_agent(
            "In ADNI, compare mean CSF p-tau181 across CN, MCI, and AD at baseline.",
            catalog,
            backend,
            client=client,
            use_enriched_planner=True,
            use_semantic_planner=False,
        )

        assert resp.grounded is False
        assert resp.question_answered is False
        assert resp.semantic_grounding_passed is False
        assert "group_comparison" in resp.answer
        assert "EXACTLY TWO" in resp.answer or "exactly TWO" in resp.answer
        assert "CN vs. MCI vs. AD" in resp.answer or "three or more" in resp.answer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
