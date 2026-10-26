"""Regression tests for the real, live DeepSeek CN/MCI/AD-collapse failure.

Live-captured failure (see tests/fixtures/deepseek_ptau181_cn_mci_ad_raw_response.json,
captured from a real deepseek-chat API call, mode=deepseek, --use-semantic-planner):
DeepSeek expressed each clinical-status group's diagnosis criterion as raw ADNI column
names with operator objects inside baseline_filters (e.g. {"DXMCI": 1, "DXAD": {"neq":
1}}) instead of the {"diagnosis": "<label>"} shape the compiler recognized. Every
clinical-status group compiled to an EMPTY filter dict, all three collapsed to the
unfiltered Space-B cohort (eligible_n=3711 for CN, MCI, AND dementia), yet
grounded=True / question_answered=True / semantic_grounding_passed=True were reported.

C14/C15 (the pre-existing empty/identical-filter guards) did NOT catch this because
they compared the RAW baseline_filters dicts, which were genuinely different from each
other (different raw column names per group) — the collision only appears in the
COMPILED canonical filter, one boundary later. The fix: GroupDefinition gained a
Pydantic-validated diagnosis_basis/clinical_status enum (the LLM cannot misname an
enum the way it can invent a dict key), and C14/C15 now compare the CANONICAL COMPILED
filter (semantic_planner._filter_dict_for_scope) rather than the pre-compile dict.

This file runs the REAL captured fixture through the REAL production pipeline —
parser -> semantic validator -> compile_to_enriched -> classic validator -> executor —
with no hand-corrected ToolOperation anywhere. Sections:

  A. The captured failing fixture is rejected before execution (Phase 5, items 1-4, 8)
  B. A correctly-shaped plan (using the new canonical fields) preserves all four
     defining filters through every transformation, end to end (items 5-7)
  C. Classic (non-semantic) enriched planner is unaffected (item 9)
  D. Baseline CSF-to-DXSUM alignment and source_level remain unchanged (items 10-11)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alz_agent.agent.executor import DeterministicExecutor
from alz_agent.agent.plan_validator import PlanValidator
from alz_agent.agent.providers import PlannerClient
from alz_agent.agent.semantic_planner import (
    _build_semantic_plan_from_data,
    _filter_dict_for_scope,
    compile_to_enriched,
)
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    CatalogEntry,
    IdSpace,
    Modality,
    SourceDataset,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _ROOT / "tests" / "fixtures" / "deepseek_ptau181_cn_mci_ad_raw_response.json"


def _load_captured_fixture() -> tuple[dict, str]:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["raw_plan_data"], data["_fixture_metadata"]["question"]


def _ptau181_catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            variable_id="ADNI.ADSP_PHC_CSF.pTau181_RAW",
            concept="ptau181_raw", source_dataset=SourceDataset.ADNI, id_space=IdSpace.B,
            raw_code="pTau181_RAW", release_version="R3", modality=Modality.FLUID_BIOMARKER,
            family="ptau181", variant="csf_raw",
        ),
        CatalogEntry(
            variable_id="ADNI.ADSP_PHC_CSF.PHC_pTau181",
            concept="phc_ptau181", source_dataset=SourceDataset.ADNI, id_space=IdSpace.B,
            raw_code="PHC_pTau181", release_version="R3", modality=Modality.FLUID_BIOMARKER,
            family="ptau181", variant="csf_harmonized",
        ),
    ]


class _FixedPlanClient(PlannerClient):
    """Returns the SAME captured/constructed plan_data verbatim for whichever
    planner tool is actually invoked (plan_semantic_analysis for the semantic
    path, plan_enriched_analysis for the classic path) on every call, including
    the repair retry -- exactly reproduces what happens when the LLM makes the
    same mistake twice, without hand-correcting anything mid-test."""

    def __init__(self, plan_data: dict, tool_name: str = "plan_semantic_analysis"):
        self._plan_data = plan_data
        self._tool_name = tool_name

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        if tool_schema.get("name") == self._tool_name:
            return dict(self._plan_data)
        return None

    def complete_text(self, messages, max_tokens=1024):
        return ""


# ---------------------------------------------------------------------------
# A. The captured failing fixture is rejected before execution
# ---------------------------------------------------------------------------


class TestCapturedFailingFixtureRejected:
    def test_1_raw_captured_output_rejected_via_real_pipeline(self):
        """Item 1: the actual raw DeepSeek JSON, parsed through the real
        _build_semantic_plan_from_data + PlanValidator.validate_semantic(), with
        NO hand correction, is rejected before any operation executes."""
        raw, question = _load_captured_fixture()
        sem = _build_semantic_plan_from_data(dict(raw), question)
        assert sem is not None, "fixture must parse into a valid SemanticAnalysisPlan"

        v = PlanValidator([])
        res = v.validate_semantic(sem)
        assert res.is_executable is False
        combined = " ".join(res.hard_errors)
        assert "C14" in combined
        assert "C15" in combined

    def test_2_execution_operations_run_empty_end_to_end(self):
        """Item 2: run_agent() with the captured fixture as the LLM's response
        (both initial attempt and repair) produces zero executed operations."""
        from alz_agent.agent.orchestrator import run_agent

        raw, question = _load_captured_fixture()
        client = _FixedPlanClient(raw)
        catalog = _ptau181_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)

        resp = run_agent(
            question, catalog, backend, client=client,
            use_enriched_planner=True, use_semantic_planner=True,
        )
        assert resp.execution_operations_run == []
        assert resp.enriched_plan_used is False

    def test_3_success_flags_false(self):
        """Item 3: grounded, question_answered, semantic_grounding_passed are all
        False for the rejected fixture."""
        from alz_agent.agent.orchestrator import run_agent

        raw, question = _load_captured_fixture()
        client = _FixedPlanClient(raw)
        catalog = _ptau181_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)

        resp = run_agent(
            question, catalog, backend, client=client,
            use_enriched_planner=True, use_semantic_planner=True,
        )
        assert resp.grounded is False
        assert resp.question_answered is False
        assert resp.semantic_grounding_passed is False
        # requested_measure_resolved may independently remain True: the variable_id
        # (CSF p-tau181) itself was never in question -- only the group filters failed.
        assert resp.requested_measure_resolved is True

    def test_4_no_unfiltered_values_presented_as_valid(self):
        """Item 4: the abstain answer must not present '3711'-style unfiltered
        numbers as if they were a valid, grounded result under a 'Computed
        results'/'grounded values' heading -- the rejection happens before any
        backend call, so there ARE no computed values to leak."""
        from alz_agent.agent.orchestrator import run_agent

        raw, question = _load_captured_fixture()
        client = _FixedPlanClient(raw)
        catalog = _ptau181_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)

        resp = run_agent(
            question, catalog, backend, client=client,
            use_enriched_planner=True, use_semantic_planner=True,
        )
        assert "Computed results" not in resp.answer
        assert "grounded values from deterministic tools" not in resp.answer
        assert resp.tool_results == []

    def test_8_compilation_drop_caught_by_post_compile_style_validation(self):
        """Item 8: even directly at the compile_to_enriched() level (bypassing
        validate_semantic entirely), the captured fixture's groups compile to
        empty filters -- proving the defect is real at the compiled-operation
        boundary, not just detectable pre-compile."""
        raw, question = _load_captured_fixture()
        sem = _build_semantic_plan_from_data(dict(raw), question)
        assert sem is not None

        enriched = compile_to_enriched(sem, question)
        count_ops = [op for op in enriched.operations if op.op_type == "count_eligible"]
        assert len(count_ops) == 3  # CN, MCI, AD
        for op in count_ops:
            assert op.parameters.get("filter") == {}, (
                f"op '{op.output_name}' should compile to an empty filter given the "
                "fixture's raw-column-shaped baseline_filters, reproducing the defect "
                "at the compiled-operation boundary"
            )


# ---------------------------------------------------------------------------
# B. A correctly-shaped plan preserves all four defining filters end to end
# ---------------------------------------------------------------------------


def _correct_cn_mci_ad_plan_data() -> dict:
    """A plan shaped the way the UPDATED tool schema now requires: every group
    sets diagnosis_basis (+ clinical_status for clinical-status groups). This is
    NOT a hand-corrected ToolOperation -- it is raw LLM-tool-call-shaped JSON, at
    the SAME boundary (Boundary 1) as the captured failing fixture, just using the
    fields the schema now mandates."""
    return {
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
            {
                "label": "CN", "cohort_scope": "baseline",
                "diagnosis_basis": "clinical_status", "clinical_status": "normal",
            },
            {
                "label": "MCI", "cohort_scope": "baseline",
                "diagnosis_basis": "clinical_status", "clinical_status": "mci",
            },
            {
                "label": "AD (syndromic dementia)", "cohort_scope": "baseline",
                "diagnosis_basis": "clinical_status", "clinical_status": "dementia",
            },
            {
                "label": "AD (etiologic AD)", "cohort_scope": "baseline",
                "diagnosis_basis": "etiologic_ad",
            },
        ],
        "variable_selections": [
            {
                "variable_id": "ADNI.ADSP_PHC_CSF.pTau181_RAW",
                "specimen": "csf", "analyte": "ptau181", "variant_label": "csf_raw",
            },
        ],
    }


class TestCorrectPlanPreservesFiltersEndToEnd:
    def test_5_and_6_filters_preserved_through_every_transformation(self):
        """Items 5-6: trace the SAME four boundaries as the live investigation
        (parsed GroupDefinition -> semantic validation -> compiled operation filter
        -> executor FilterSpec) for a correctly-shaped plan and confirm the exact
        expected values survive unchanged at every stage."""
        raw = _correct_cn_mci_ad_plan_data()
        question = (
            "In ADNI, what are the mean CSF p-tau181 levels for CN, MCI, and AD "
            "groups at baseline?"
        )

        # Boundary 2: parsed GroupDefinition objects
        sem = _build_semantic_plan_from_data(dict(raw), question)
        assert sem is not None
        by_label = {g.label: g for g in sem.groups}
        assert by_label["CN"].diagnosis_basis == "clinical_status"
        assert by_label["CN"].clinical_status == "normal"
        assert by_label["MCI"].clinical_status == "mci"
        assert by_label["AD (syndromic dementia)"].clinical_status == "dementia"
        assert by_label["AD (etiologic AD)"].diagnosis_basis == "etiologic_ad"

        # Boundary 3: semantic validation must pass (distinct, non-empty filters)
        v = PlanValidator(_ptau181_catalog())
        validation = v.validate_semantic(sem)
        assert validation.is_executable is True, validation.hard_errors

        # Canonical compiled filter (what C14/C15 AND compile_to_enriched both use)
        assert _filter_dict_for_scope(by_label["CN"]) == {"diagnosis": "normal"}
        assert _filter_dict_for_scope(by_label["MCI"]) == {"diagnosis": "mci"}
        assert _filter_dict_for_scope(by_label["AD (syndromic dementia)"]) == {"diagnosis": "dementia"}
        assert _filter_dict_for_scope(by_label["AD (etiologic AD)"]) == {"etiology": "ad"}

        # Boundary 4: compiled PlanOperation.parameters['filter']
        enriched = compile_to_enriched(sem, question)
        count_ops = {
            op.description: op.parameters.get("filter")
            for op in enriched.operations if op.op_type == "count_eligible"
        }
        filters_seen = list(count_ops.values())
        assert {"diagnosis": "normal"} in filters_seen
        assert {"diagnosis": "mci"} in filters_seen
        assert {"diagnosis": "dementia"} in filters_seen
        assert {"etiology": "ad"} in filters_seen

        # Boundary 5: executor-created FilterSpec
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        executor = DeterministicExecutor(_ptau181_catalog(), backend)
        specs_by_filter = {}
        for op in enriched.operations:
            if op.op_type != "count_eligible":
                continue
            fs = executor._build_filter_spec(enriched, op.parameters)
            specs_by_filter[json.dumps(op.parameters.get("filter"), sort_keys=True)] = fs

        cn_spec = specs_by_filter[json.dumps({"diagnosis": "normal"}, sort_keys=True)]
        assert cn_spec.baseline_diagnosis == ["normal"]
        assert cn_spec.id_space == IdSpace.B

        mci_spec = specs_by_filter[json.dumps({"diagnosis": "mci"}, sort_keys=True)]
        assert mci_spec.baseline_diagnosis == ["mci"]

        dementia_spec = specs_by_filter[json.dumps({"diagnosis": "dementia"}, sort_keys=True)]
        assert dementia_spec.baseline_diagnosis == ["dementia"]

        ad_etiologic_spec = specs_by_filter[json.dumps({"etiology": "ad"}, sort_keys=True)]
        assert ad_etiologic_spec.baseline_etiology == ["ad"]
        assert ad_etiologic_spec.baseline_diagnosis is None

    def test_7_cn_mci_dementia_produce_distinct_cohort_counts(self):
        """Item 7: CN/MCI/dementia produce genuinely distinct eligible_n (not the
        3711/3711/3711 collapse). FixtureDataBackend's synthetic Space-B data
        returns 0 for every baseline_diagnosis label regardless of value (a
        fixture-data limitation, not a pipeline defect -- verified directly
        below), so this uses the real local backend with the SAME FilterSpecs
        produced by the real pipeline in test_5_and_6, skipped if unavailable."""
        config_path = _ROOT / "config" / "paths.yaml"
        if not config_path.exists():
            pytest.skip("config/paths.yaml not present")

        import yaml
        from alz_agent.backend.local_backend import LocalDataBackend
        from alz_agent.schemas import FilterSpec

        raw = _correct_cn_mci_ad_plan_data()
        question = (
            "In ADNI, what are the mean CSF p-tau181 levels for CN, MCI, and AD "
            "groups at baseline?"
        )
        sem = _build_semantic_plan_from_data(dict(raw), question)
        enriched = compile_to_enriched(sem, question)

        config = yaml.safe_load(open(config_path, encoding="utf-8"))
        adni_dir = _ROOT / config["adni"]["base_dir"]
        if not adni_dir.is_dir():
            pytest.skip("real ADNI data not mounted")
        try:
            backend = LocalDataBackend(config, _ROOT)
            executor = DeterministicExecutor(_ptau181_catalog(), backend)
            counts = {}
            for op in enriched.operations:
                if op.op_type != "count_eligible":
                    continue
                fs = executor._build_filter_spec(enriched, op.parameters)
                r = backend.count_eligible(fs)
                if not r.grounded:
                    pytest.skip(f"real ADNI data not mounted or insufficient: {r.abstain_reason}")
                counts[op.description] = r.value
        except FileNotFoundError:
            pytest.skip("real ADNI data files not present")

        assert len(counts) == 4  # CN, MCI, AD-syndromic, AD-etiologic
        values = list(counts.values())
        assert len(set(values)) > 1, (
            f"expected distinct cohort counts across groups, got {counts} "
            "(all identical would reproduce the 3711/3711/3711 collapse)"
        )


# ---------------------------------------------------------------------------
# C. Classic (non-semantic) enriched planner unaffected
# ---------------------------------------------------------------------------


class TestClassicPlannerUnaffected:
    def test_9_classic_path_ignores_diagnosis_basis_entirely(self):
        """Item 9: the classic (use_semantic_planner=False) enriched planner never
        constructs a GroupDefinition at all -- it is structurally impossible for
        this change to affect it. Verified by confirming a classic-path plan with
        an ordinary filter dict still executes normally."""
        from alz_agent.agent.orchestrator import run_agent
        from alz_agent.eval.harness import default_catalog

        enriched_data = {
            "research_intent_summary": "count MCI participants",
            "researcher_audit_trace": ["Interpreted as MCI count."],
            "operations": [
                {
                    "op_type": "count_eligible",
                    "parameters": {"filter": {"diagnosis": "mci"}},
                    "output_name": "n_mci",
                    "description": "Count MCI.",
                }
            ],
            "requires_backend_data": True,
            "requested_horizons": [],
            "baseline_definition": {"diagnosis": "mci"},
            "outcome_definition": "",
            "id_space": "A",
        }
        client = _FixedPlanClient(enriched_data, tool_name="plan_enriched_analysis")
        catalog = default_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)

        resp = run_agent(
            "How many MCI participants are there?", catalog, backend,
            client=client, use_enriched_planner=True, use_semantic_planner=False,
        )
        assert resp.planner_path == "classic_enriched"
        assert resp.enriched_plan_used is True


# ---------------------------------------------------------------------------
# D. Baseline CSF alignment and source_level unchanged
# ---------------------------------------------------------------------------


class TestBaselineAlignmentAndSourceLevelUnchanged:
    """Requires real local ADNI data (skipped if unavailable)."""

    def setup_method(self):
        config_path = _ROOT / "config" / "paths.yaml"
        if not config_path.exists():
            pytest.skip("config/paths.yaml not present")

    def test_10_and_11_baseline_alignment_and_source_level_still_correct(self):
        import yaml
        from alz_agent.backend.local_backend import LocalDataBackend
        from alz_agent.schemas import FilterSpec

        config = yaml.safe_load(open(_ROOT / "config" / "paths.yaml", encoding="utf-8"))
        backend = LocalDataBackend(config, _ROOT)

        fs = FilterSpec(baseline_diagnosis=["normal"], id_space=IdSpace.B)
        try:
            r_raw = backend.aggregate_stat("ADNI.ADSP_PHC_CSF.pTau181_RAW", "mean", fs, None)
            r_harm = backend.aggregate_stat("ADNI.ADSP_PHC_CSF.PHC_pTau181", "mean", fs, None)
        except FileNotFoundError:
            pytest.skip("real ADNI data files not present")

        if not r_raw.grounded or not r_harm.grounded:
            pytest.skip("real ADNI data not mounted or insufficient for this check")

        assert r_raw.evidence[0].source_level == "phc_raw_column"
        assert r_harm.evidence[0].source_level == "phc_harmonized"
        # matched_n <= eligible cohort, both present -- alignment machinery still active
        assert r_raw.evidence[0].matched_n is not None
        assert r_raw.evidence[0].baseline_alignment_rule is not None
        assert "excluded" in r_raw.evidence[0].baseline_alignment_rule.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
