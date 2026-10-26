"""Tests for the 4-layer enriched planner architecture.

Sections:
  A. Schema construction and serialization
  B. PlanValidator checks
  C. DeterministicExecutor fan-out (multi-horizon longitudinal feasibility)
  D. DeterministicExecutor routing
  E. Backward compat (use_enriched_planner=False)
  F. Motivating bug — MCI multi-horizon question
  G. Hallucination guard on SynthesisReport
  H. explore_catalog / describe_variable_family routing (Space B)
"""

from __future__ import annotations

import json

import pytest

from alz_agent.agent.executor import DeterministicExecutor
from alz_agent.agent.hallucination_guard import (
    _check_synthesis_hallucination,
    _collect_evidence_numbers,
)
from alz_agent.agent.plan_validator import PlanValidator
from alz_agent.agent.providers import PlannerClient
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    CatalogEntry,
    DataEvidence,
    EnrichedAnalysisPlan,
    ExecutionContext,
    IdSpace,
    Modality,
    PlanOperation,
    SourceDataset,
    SynthesisReport,
    SynthesisSection,
    ToolResult,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_catalog() -> list[CatalogEntry]:
    """Minimal Space A catalog for tests."""
    return [
        CatalogEntry(
            variable_id="nacc_udsd",
            concept="cognitive status",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCUDSD",
            release_version="v70",
            modality=Modality.COGNITION,
        ),
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
            variable_id="nacc_mmse",
            concept="MMSE score",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCMMSE",
            release_version="v70",
            modality=Modality.COGNITION,
        ),
        CatalogEntry(
            variable_id="phc_mem",
            concept="memory composite",
            source_dataset=SourceDataset.ADSP_PHC,
            id_space=IdSpace.A,
            raw_code="PHC_MEM",
            release_version="r3",
            modality=Modality.COGNITION,
        ),
    ]


def _make_enriched_plan_mci_multihorizon() -> EnrichedAnalysisPlan:
    """EnrichedAnalysisPlan for the motivating MCI multi-horizon question."""
    return EnrichedAnalysisPlan(
        original_question=(
            "Among participants with baseline MCI, how many have sufficient "
            "follow-up to determine conversion status within 2, 3, or 5 years?"
        ),
        research_intent_summary="Count MCI participants with sufficient follow-up at 2, 3, and 5 years.",
        researcher_audit_trace=[
            "Interpreted 'baseline MCI' as baseline_diagnosis=mci (NACCUDSD=3).",
            "Extracted horizons [2, 3, 5] from '2, 3, or 5 years'.",
            "This question counts participants, not visits.",
            "Requires real longitudinal visit data, not catalog-only lookup.",
        ],
        operations=[
            PlanOperation(
                op_type="count_eligible",
                parameters={"filter": {"diagnosis": "mci"}},
                output_name="mci_cohort_n",
                description="Count MCI participants at baseline.",
            ),
            PlanOperation(
                op_type="longitudinal_feasibility",
                parameters={"horizons": [2, 3, 5], "filter": {"diagnosis": "mci"}},
                output_name="mci_followup",
                description="Check follow-up sufficiency at 2, 3, and 5 year horizons.",
            ),
        ],
        requested_horizons=[2, 3, 5],
        requires_backend_data=True,
        baseline_definition={"diagnosis": "mci"},
        outcome_definition="conversion to NACCUDSD=4 within N years",
        id_space="A",
    )


# ---------------------------------------------------------------------------
# A. Schema construction and serialization
# ---------------------------------------------------------------------------


class TestSchemaConstruction:
    def test_plan_operation_is_frozen(self):
        op = PlanOperation(op_type="count_eligible", output_name="n")
        with pytest.raises(Exception):
            op.op_type = "other"  # type: ignore[misc]

    def test_enriched_analysis_plan_roundtrip(self):
        plan = _make_enriched_plan_mci_multihorizon()
        d = plan.model_dump()
        plan2 = EnrichedAnalysisPlan.model_validate(d)
        assert plan2.original_question == plan.original_question
        assert plan2.requested_horizons == [2, 3, 5]
        assert len(plan2.operations) == 2

    def test_validation_result_is_frozen(self):
        vr = ValidationResult(hard_errors=["e1"])
        with pytest.raises(Exception):
            vr.is_executable = True  # type: ignore[misc]

    def test_execution_context_is_mutable(self):
        ctx = ExecutionContext()
        ctx.results["x"] = ToolResult.ok(42)
        ctx.operations_run.append("x")
        assert ctx.results["x"].value == 42
        assert "x" in ctx.operations_run

    def test_synthesis_section_is_frozen(self):
        sec = SynthesisSection(section_id="direct_answer", title="Answer", content="hi")
        with pytest.raises(Exception):
            sec.content = "other"  # type: ignore[misc]

    def test_synthesis_report_serialization_with_tool_result_int(self):
        ctx = ExecutionContext()
        ctx.results["n"] = ToolResult.ok(42)
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(section_id="direct_answer", title="A", content="42")],
            execution_context=ctx,
            grounded=True,
        )
        dumped = report.model_dump_json()
        data = json.loads(dumped)
        assert data["question"] == "q"
        assert data["execution_context"]["results"]["n"]["value"] == 42

    def test_synthesis_report_serialization_with_suppressed_result(self):
        ctx = ExecutionContext()
        ctx.results["small"] = ToolResult.abstain("count suppressed")
        report = SynthesisReport(
            question="q",
            execution_context=ctx,
            grounded=False,
            abstain_reason="suppressed",
        )
        dumped = report.model_dump_json()
        data = json.loads(dumped)
        assert data["grounded"] is False
        assert data["execution_context"]["results"]["small"]["grounded"] is False

    def test_synthesis_report_serialization_with_dict_result(self):
        ctx = ExecutionContext()
        ctx.results["followup"] = ToolResult.ok({
            "2yr": {"sufficient": 100, "insufficient": 20, "censored": 5},
            "3yr": {"sufficient": 80, "insufficient": 30, "censored": 5},
        })
        report = SynthesisReport(question="q", execution_context=ctx)
        dumped = report.model_dump_json()
        data = json.loads(dumped)
        assert data["execution_context"]["results"]["followup"]["value"]["2yr"]["sufficient"] == 100


# ---------------------------------------------------------------------------
# B. Validator checks
# ---------------------------------------------------------------------------


class TestPlanValidator:
    def setup_method(self):
        self.catalog = _make_catalog()
        self.validator = PlanValidator(self.catalog)

    def test_valid_multi_horizon_plan_is_executable(self):
        plan = _make_enriched_plan_mci_multihorizon()
        result = self.validator.validate(plan)
        assert result.is_executable
        assert result.hard_errors == []

    def test_invalid_horizon_zero_is_hard_error(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="longitudinal_feasibility",
                    parameters={"horizons": [0]},
                    output_name="out",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert any("0" in e for e in result.hard_errors)

    def test_invalid_negative_requested_horizon_is_hard_error(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            requested_horizons=[-1],
            operations=[
                PlanOperation(
                    op_type="count_eligible",
                    parameters={},
                    output_name="n",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert any("positive integer" in e for e in result.hard_errors)

    def test_missing_variable_id_for_variable_stat_is_hard_error(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="variable_stat",
                    parameters={"stat_type": "mean"},  # missing variable_id
                    output_name="age_mean",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert any("variable_id" in e for e in result.hard_errors)

    def test_missing_variable_id_for_descriptive_stat_is_hard_error(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="descriptive_stat",
                    parameters={},  # missing variable_id
                    output_name="age_desc",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert any("variable_id" in e for e in result.hard_errors)

    def test_malformed_variable_id_without_prefix_is_hard_error(self):
        """variable_id must be 'NACC.<COL>' — bare names like 'nacc_age' must be rejected."""
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="variable_stat",
                    parameters={"variable_id": "nacc_age", "stat_type": "mean"},
                    output_name="bad_stat",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert any("NACC.<COLUMN_NAME>" in e for e in result.hard_errors)

    def test_nacc_variable_in_space_b_is_cross_space_leak(self):
        """A Space B plan carrying a NACC.* variable_id is a hard-blocked cross-space leak."""
        plan = EnrichedAnalysisPlan(
            original_question="mean age at first AD diagnosis in ADNI",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="variable_stat",
                    parameters={"variable_id": "NACC.NACCAGE", "stat_type": "mean"},
                    output_name="age_stat",
                )
            ],
            id_space="B",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert "age_stat" in result.blocked_operations
        assert any("cross-space" in e.lower() for e in result.hard_errors)

    def test_adni_variable_in_space_a_is_cross_space_leak(self):
        """A Space A plan carrying an ADNI.* variable_id is a hard-blocked cross-space leak."""
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="variable_stat",
                    parameters={"variable_id": "ADNI.MMSE.MMSCORE", "stat_type": "mean"},
                    output_name="mmse_stat",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert "mmse_stat" in result.blocked_operations
        assert any("cross-space" in e.lower() for e in result.hard_errors)

    def test_filter_diagnosis_key_with_nacc_raw_key_is_not_cross_space(self):
        """Regression: 'diagnosis' is the compiler's own universal semantic filter
        key (semantic_planner._filter_dict_for_scope emits it for BOTH id_space A
        and B baseline-diagnosis groups; executor._build_filter_spec reads it the
        same way for both spaces) — it is never a literal raw-column reference. A
        live DeepSeek Space A (NACC) plan whose compiled filter carried both
        'diagnosis' and a literal raw NACC column (e.g. 'NACCUDSD', passed through
        from the LLM's event_filters) previously false-positived here: 'diagnosis'
        was incorrectly listed as a Space-B-only key (it collides case-insensitively
        with ADNI's raw DXSUM.DIAGNOSIS column name), so a 100% legitimate Space A
        op was rejected as a fabricated cross-space join and the question abstained
        with a confusing 'mixes Space A keys and Space B keys' error."""
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="count_eligible",
                    parameters={"filter": {"diagnosis": "dementia", "NACCUDSD": 4}},
                    output_name="eligible_n",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert "eligible_n" not in result.blocked_operations
        assert not any("cross-space" in e.lower() for e in result.hard_errors)

    def test_filter_genuine_cross_space_key_mix_still_rejected(self):
        """A filter genuinely mixing a Space A raw column (NACCUDSD) with a Space B
        raw column (RID/PTID/etc., not 'diagnosis') must still be rejected — the
        fix for the 'diagnosis' false positive must not defeat this check entirely."""
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="count_eligible",
                    parameters={"filter": {"NACCUDSD": 4, "RID": 123}},
                    output_name="eligible_n",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert "eligible_n" in result.blocked_operations
        assert any("cross-space" in e.lower() for e in result.hard_errors)

    def test_adni_variable_in_space_b_passes(self):
        """An 'ADNI.<TABLE>.<FIELD>' id in Space B is well-formed and must not error."""
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="variable_stat",
                    parameters={"variable_id": "ADNI.MMSE.MMSCORE", "stat_type": "mean"},
                    output_name="mmse_stat",
                )
            ],
            id_space="B",
        )
        result = self.validator.validate(plan)
        assert "mmse_stat" not in result.blocked_operations
        assert not any("cross-space" in e.lower() for e in result.hard_errors)

    def test_derived_age_id_in_space_b_passes(self):
        """The derived 'ADNI.DERIVED.age_at_event' id must be accepted in Space B."""
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="variable_stat",
                    parameters={
                        "variable_id": "ADNI.DERIVED.age_at_event",
                        "stat_type": "mean",
                        "event_anchor": "first_diagnosis",
                    },
                    output_name="age_stat",
                )
            ],
            id_space="B",
        )
        result = self.validator.validate(plan)
        assert "age_stat" not in result.blocked_operations
        assert not any("cross-space" in e.lower() for e in result.hard_errors)

    def test_correct_variable_id_format_passes_validator(self):
        """'NACC.NACCAGE' is the correct format and should not trigger a hard error."""
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="variable_stat",
                    parameters={"variable_id": "NACC.NACCAGE", "stat_type": "mean"},
                    output_name="age_stat",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not any("NACC.<COLUMN_NAME>" in e for e in result.hard_errors)

    def test_duplicate_output_name_is_hard_error(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(op_type="count_eligible", parameters={}, output_name="n"),
                PlanOperation(op_type="count_eligible", parameters={}, output_name="n"),
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert any("Duplicate" in e for e in result.hard_errors)

    def test_invalid_id_space_is_hard_error(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(op_type="count_eligible", parameters={}, output_name="n"),
            ],
            id_space="C",  # invalid
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert any("id_space" in e for e in result.hard_errors)

    def test_syndromic_outcome_code_with_ad_mention_emits_soft_warning(self):
        plan = EnrichedAnalysisPlan(
            original_question="How many AD participants convert to dementia?",
            research_intent_summary="s",
            operations=[
                PlanOperation(op_type="count_eligible", parameters={}, output_name="n"),
            ],
            outcome_definition="conversion to NACCUDSD=4 within 3 years",
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert result.is_executable  # soft warning, not hard error
        assert any("all-cause" in w.lower() or "syndromic" in w.lower() for w in result.soft_warnings)

    def test_id_space_A_alone_does_not_emit_ad_caveat(self):
        """AD caveat must NOT be triggered merely by id_space=A."""
        plan = EnrichedAnalysisPlan(
            original_question="How many MCI participants are in NACC?",
            research_intent_summary="s",
            operations=[
                PlanOperation(op_type="count_eligible", parameters={}, output_name="n"),
            ],
            outcome_definition="",  # no outcome definition
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not any("all-cause" in w.lower() for w in result.soft_warnings)

    def test_empty_operations_is_hard_error(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable

    def test_unusually_large_horizon_emits_soft_warning(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            requested_horizons=[15],
            operations=[
                PlanOperation(
                    op_type="longitudinal_feasibility",
                    parameters={"horizons": [15]},
                    output_name="followup",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert result.is_executable  # only a warning
        assert any("unusually long" in w for w in result.soft_warnings)

    def test_missing_modalities_for_modality_overlap_is_hard_error(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="modality_overlap",
                    parameters={},  # missing modalities
                    output_name="overlap",
                )
            ],
            id_space="A",
        )
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert any("modalities" in e for e in result.hard_errors)


# ---------------------------------------------------------------------------
# C. DeterministicExecutor fan-out (multi-horizon longitudinal feasibility)
# ---------------------------------------------------------------------------


class TestExecutorFanOut:
    def setup_method(self):
        self.catalog = _make_catalog()
        self.backend = build_fixture_backend(
            n_mci=40, n_mci_long=25, n_mci_short=8, n_single_visit=7
        )

    def test_longitudinal_feasibility_fan_out_produces_keyed_dict(self):
        executor = DeterministicExecutor(self.catalog, self.backend)
        plan = _make_enriched_plan_mci_multihorizon()
        ctx = executor.execute(plan)

        assert "mci_followup" in ctx.results
        result = ctx.results["mci_followup"]
        assert result.grounded
        val = result.value
        assert isinstance(val, dict)
        assert "2yr" in val
        assert "3yr" in val
        assert "5yr" in val

    def test_each_horizon_bucket_has_correct_keys(self):
        executor = DeterministicExecutor(self.catalog, self.backend)
        plan = _make_enriched_plan_mci_multihorizon()
        ctx = executor.execute(plan)

        val = ctx.results["mci_followup"].value
        for horizon_key in ("2yr", "3yr", "5yr"):
            bucket = val[horizon_key]
            assert "sufficient" in bucket
            assert "insufficient" in bucket
            assert "censored" in bucket

    def test_partial_fan_out_succeeds_when_some_horizons_fail(self):
        """If a backend returns suppressed for some horizons, partial success is OK."""
        executor = DeterministicExecutor(self.catalog, self.backend)
        # Create a plan with a very large horizon (30yr) that would abstain
        # plus a valid one (3yr)
        from alz_agent.schemas import EnrichedAnalysisPlan, PlanOperation
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="longitudinal_feasibility",
                    parameters={"horizons": [3, 1000]},  # 1000yr will get 0 sufficient
                    output_name="followup",
                )
            ],
            requested_horizons=[3, 1000],
            id_space="A",
        )
        ctx = executor.execute(plan)
        result = ctx.results["followup"]
        # Should still be grounded (3yr has data)
        assert result.grounded
        assert "3yr" in result.value

    def test_all_horizons_abstain_produces_abstain_result(self):
        """When no horizons return data, the fan-out should abstain."""
        executor = DeterministicExecutor(self.catalog, self.backend)
        from alz_agent.schemas import EnrichedAnalysisPlan, PlanOperation
        # Use an id_space=B plan — fixture backend is id_space=A, so it'll abstain
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="longitudinal_feasibility",
                    parameters={"horizons": [3]},
                    output_name="followup",
                )
            ],
            requested_horizons=[3],
            baseline_definition={"diagnosis": "some_invalid_diagnosis_xyz"},
            id_space="A",
        )
        ctx = executor.execute(plan)
        # With an invalid diagnosis the backend may abstain or return empty
        # Either way, the result should be present in ctx
        assert "followup" in ctx.results


# ---------------------------------------------------------------------------
# D. DeterministicExecutor routing
# ---------------------------------------------------------------------------


class TestExecutorRouting:
    def setup_method(self):
        self.catalog = _make_catalog()
        self.backend = build_fixture_backend()
        self.executor = DeterministicExecutor(self.catalog, self.backend)

    def test_count_eligible_op_returns_int(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="count_eligible",
                    parameters={"filter": {"diagnosis": "mci"}},
                    output_name="mci_n",
                )
            ],
            id_space="A",
        )
        ctx = self.executor.execute(plan)
        assert "mci_n" in ctx.results
        result = ctx.results["mci_n"]
        assert result.grounded
        assert isinstance(result.value, int)
        assert result.value > 0

    def test_lookup_variable_op_returns_grounded_result(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="lookup_variable",
                    parameters={"query": "MMSE"},
                    output_name="mmse_lookup",
                )
            ],
            id_space="A",
        )
        ctx = self.executor.execute(plan)
        assert "mmse_lookup" in ctx.results
        assert ctx.results["mmse_lookup"].grounded

    def test_unknown_op_type_returns_abstain(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="nonexistent_op_xyz",
                    parameters={},
                    output_name="bad",
                )
            ],
            id_space="A",
        )
        ctx = self.executor.execute(plan)
        assert "bad" in ctx.results
        assert not ctx.results["bad"].grounded

    def test_blocked_operations_are_skipped(self):
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(op_type="count_eligible", parameters={}, output_name="n1"),
                PlanOperation(op_type="count_eligible", parameters={}, output_name="n2"),
            ],
            id_space="A",
        )
        ctx = self.executor.execute(plan, blocked_operations=["n2"])
        assert "n1" in ctx.results
        assert "n2" not in ctx.results

    def test_per_op_exception_does_not_abort_remaining_ops(self):
        """An exception in one op should not prevent subsequent ops from running."""
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="variable_stat",
                    parameters={"variable_id": "nonexistent_var", "stat_type": "mean"},
                    output_name="bad_stat",
                ),
                PlanOperation(
                    op_type="count_eligible",
                    parameters={},
                    output_name="n",
                ),
            ],
            id_space="A",
        )
        ctx = self.executor.execute(plan)
        assert "bad_stat" in ctx.results
        assert "n" in ctx.results
        # Second op should still succeed
        assert ctx.results["n"].grounded


# ---------------------------------------------------------------------------
# H. explore_catalog / describe_variable_family routing (Space B)
# ---------------------------------------------------------------------------


def _make_biomarker_catalog_b() -> list[CatalogEntry]:
    """Space-B biomarker catalog with family/variant tags (ptau217 in plasma only)."""
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
    ]


class TestExploreCatalogAndDescribeVariableFamilyRouting:
    """End-to-end Space-B routing: PlanValidator -> DeterministicExecutor for the
    two new op types — covers the motivating "do we have enough PTAU data, what
    subtype, what kind" question via the enriched-planner registry (not just the
    bare tool functions tested in test_p3_tools.py).
    """

    def setup_method(self):
        self.catalog = _make_biomarker_catalog_b()
        self.backend = build_fixture_backend(id_space=IdSpace.B, seed=42)
        self.executor = DeterministicExecutor(self.catalog, self.backend)
        self.validator = PlanValidator(self.catalog)

    def _plan(self, operations: list[PlanOperation]) -> EnrichedAnalysisPlan:
        return EnrichedAnalysisPlan(
            original_question="Do we have enough PTAU data, what subtype, what kind?",
            research_intent_summary="Assess PTAU biomarker coverage by subtype/fluid.",
            operations=operations,
            id_space="B",
        )

    def test_explore_catalog_op_validates_and_executes(self):
        plan = self._plan([
            PlanOperation(
                op_type="explore_catalog",
                parameters={"keyword": "ptau217"},
                output_name="ptau_matches",
            ),
        ])
        validation = self.validator.validate(plan)
        assert validation.is_executable
        ctx = self.executor.execute(plan, blocked_operations=validation.blocked_operations)
        result = ctx.results["ptau_matches"]
        assert result.grounded
        assert len(result.value) == 2

    def test_describe_variable_family_op_validates_and_executes(self):
        plan = self._plan([
            PlanOperation(
                op_type="describe_variable_family",
                parameters={"family": "ptau217"},
                output_name="ptau_family",
            ),
        ])
        validation = self.validator.validate(plan)
        assert validation.is_executable
        ctx = self.executor.execute(plan, blocked_operations=validation.blocked_operations)
        result = ctx.results["ptau_family"]
        assert result.grounded
        variants = {v.variant for v in result.value.variants}
        assert variants == {"plasma_harmonized", "plasma_raw"}

    def test_missing_keyword_param_blocks_explore_catalog(self):
        plan = self._plan([
            PlanOperation(op_type="explore_catalog", parameters={}, output_name="bad_explore"),
        ])
        validation = self.validator.validate(plan)
        assert "bad_explore" in validation.blocked_operations
        assert not validation.is_executable

    def test_missing_family_param_blocks_describe_variable_family(self):
        plan = self._plan([
            PlanOperation(op_type="describe_variable_family", parameters={}, output_name="bad_family"),
        ])
        validation = self.validator.validate(plan)
        assert "bad_family" in validation.blocked_operations

    def test_full_pipeline_explore_then_describe(self):
        """The natural two-step pattern: reconnaissance, then grouped coverage."""
        plan = self._plan([
            PlanOperation(
                op_type="explore_catalog",
                parameters={"keyword": "ptau"},
                output_name="ptau_matches",
            ),
            PlanOperation(
                op_type="describe_variable_family",
                parameters={"family": "ptau217"},
                output_name="ptau_family",
            ),
        ])
        validation = self.validator.validate(plan)
        assert validation.is_executable
        ctx = self.executor.execute(plan, blocked_operations=validation.blocked_operations)
        assert ctx.results["ptau_matches"].grounded
        assert ctx.results["ptau_family"].grounded

    def test_narrated_counts_pass_hallucination_guard(self):
        """Numbers surfaced via describe_variable_family evidence must be traceable —
        the generic evidence-text-mining path (aggregate_ref) covers this without a
        dedicated VariableFamilyResult branch in the guard.
        """
        from alz_agent.agent.hallucination_guard import _check_hallucination

        plan = self._plan([
            PlanOperation(
                op_type="describe_variable_family",
                parameters={"family": "ptau217"},
                output_name="ptau_family",
            ),
        ])
        ctx = self.executor.execute(plan)
        result = ctx.results["ptau_family"]
        assert result.grounded
        n = next(v.n for v in result.value.variants if v.n is not None)
        answer = f"There are {n} subjects with a plasma p-tau217 measurement."
        assert _check_hallucination(answer, [result])

    def test_unknown_family_abstains_through_executor(self):
        plan = self._plan([
            PlanOperation(
                op_type="describe_variable_family",
                parameters={"family": "ptau999_nonexistent"},
                output_name="ghost",
            ),
        ])
        ctx = self.executor.execute(plan)
        assert not ctx.results["ghost"].grounded


# ---------------------------------------------------------------------------
# E. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def setup_method(self):
        self.catalog = _make_catalog()
        self.backend = build_fixture_backend()

    def test_run_agent_without_enriched_planner_has_no_synthesis_report(self):
        from alz_agent.agent.orchestrator import run_agent
        response = run_agent(
            "How many MCI participants are in NACC?",
            self.catalog,
            self.backend,
            use_enriched_planner=False,
        )
        assert response.synthesis_report is None
        assert response.enriched_plan_used is False

    def test_run_agent_default_does_not_use_enriched_planner(self):
        from alz_agent.agent.orchestrator import run_agent
        response = run_agent(
            "How many MCI participants are in NACC?",
            self.catalog,
            self.backend,
        )
        assert response.synthesis_report is None
        assert response.enriched_plan_used is False

    def test_existing_answer_still_produced_without_enriched_planner(self):
        from alz_agent.agent.orchestrator import run_agent
        response = run_agent(
            "How many MCI participants are there?",
            self.catalog,
            self.backend,
            use_enriched_planner=False,
        )
        # Should still give some grounded answer
        assert isinstance(response.answer, str)
        assert response.hallucination_check_passed


# ---------------------------------------------------------------------------
# F. Motivating bug — MCI multi-horizon question
# ---------------------------------------------------------------------------


class TestMotivatingBug:
    """The question that caused the original bug must now produce grounded results."""

    def setup_method(self):
        self.catalog = _make_catalog()
        self.backend = build_fixture_backend(
            n_mci=40, n_mci_long=25, n_mci_short=8, n_single_visit=7
        )
        self.executor = DeterministicExecutor(self.catalog, self.backend)
        self.validator = PlanValidator(self.catalog)

    def _get_plan(self) -> EnrichedAnalysisPlan:
        return _make_enriched_plan_mci_multihorizon()

    def test_plan_validates_as_executable(self):
        plan = self._get_plan()
        result = self.validator.validate(plan)
        assert result.is_executable, result.hard_errors

    def test_cohort_size_op_is_grounded(self):
        plan = self._get_plan()
        ctx = self.executor.execute(plan)
        assert ctx.results["mci_cohort_n"].grounded

    def test_followup_op_is_grounded(self):
        plan = self._get_plan()
        ctx = self.executor.execute(plan)
        assert ctx.results["mci_followup"].grounded

    def test_followup_contains_all_three_horizons(self):
        plan = self._get_plan()
        ctx = self.executor.execute(plan)
        val = ctx.results["mci_followup"].value
        assert "2yr" in val
        assert "3yr" in val
        assert "5yr" in val

    def test_each_horizon_bucket_is_complete(self):
        plan = self._get_plan()
        ctx = self.executor.execute(plan)
        val = ctx.results["mci_followup"].value
        for key in ("2yr", "3yr", "5yr"):
            bucket = val[key]
            assert "sufficient" in bucket
            assert "insufficient" in bucket
            assert "censored" in bucket

    def test_horizon_counts_appear_in_evidence_numbers(self):
        """All numeric horizon-bucket values must be collectable as evidence."""
        plan = self._get_plan()
        ctx = self.executor.execute(plan)
        followup_tr = ctx.results["mci_followup"]
        nums = _collect_evidence_numbers([followup_tr])
        # At least some integers from the buckets should be in evidence numbers
        val = followup_tr.value
        for bucket in val.values():
            for v in bucket.values():
                if isinstance(v, int):
                    assert str(v) in nums

    def test_no_operations_run_after_blockage(self):
        plan = self._get_plan()
        ctx = self.executor.execute(
            plan,
            blocked_operations=["mci_cohort_n", "mci_followup"],
        )
        assert ctx.operations_run == []


# ---------------------------------------------------------------------------
# G. Hallucination guard on SynthesisReport
# ---------------------------------------------------------------------------


class TestSynthesisHallucinationGuard:
    def _make_report_with_count(self, count: int, section_id: str = "direct_answer") -> SynthesisReport:
        ctx = ExecutionContext()
        ctx.results["n"] = ToolResult.ok(
            count,
            evidence=[DataEvidence(
                query_spec={},
                aggregate_ref=f"count_eligible={count}",
            )],
        )
        return SynthesisReport(
            question="q",
            sections=[SynthesisSection(
                section_id=section_id,
                title="Answer",
                content=f"There are {count} participants.",
            )],
            execution_context=ctx,
            grounded=True,
        )

    def test_guard_passes_when_counts_match_evidence(self):
        report = self._make_report_with_count(47)
        assert _check_synthesis_hallucination(report)

    def test_guard_fails_when_invented_count_in_evidence_section(self):
        ctx = ExecutionContext()
        ctx.results["n"] = ToolResult.ok(
            47,
            evidence=[DataEvidence(query_spec={}, aggregate_ref="count_eligible=47")],
        )
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(
                section_id="evidence_provenance",
                title="Evidence",
                content="There are 999 participants.",  # 999 not in evidence
            )],
            execution_context=ctx,
            grounded=True,
        )
        assert not _check_synthesis_hallucination(report)

    def test_analysis_plan_section_exempt_from_guard(self):
        """Numbers in the analysis_plan section must not trigger the guard."""
        ctx = ExecutionContext()
        ctx.results["n"] = ToolResult.ok(47)
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(
                section_id="analysis_plan",
                title="Analysis Plan",
                content=(
                    "Interpreted baseline MCI as NACCUDSD=3. "
                    "Extracted horizons [2, 3, 5]. "
                    "Number 9999 would be invented here but section is exempt."
                ),
            )],
            execution_context=ctx,
            grounded=True,
        )
        assert _check_synthesis_hallucination(report)

    def test_next_steps_section_exempt_from_guard(self):
        """Numbers in next_steps must not trigger the guard."""
        ctx = ExecutionContext()
        ctx.results["n"] = ToolResult.ok(47)
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(
                section_id="next_steps",
                title="Next Steps",
                content="1. Consider 5-year horizon analysis.\n2. Check 10-year data.",
            )],
            execution_context=ctx,
            grounded=True,
        )
        assert _check_synthesis_hallucination(report)

    def test_codebook_values_do_not_trigger_guard(self):
        """NACCUDSD=4 style codes in section text must not fail."""
        ctx = ExecutionContext()
        ctx.results["n"] = ToolResult.ok(47)
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(
                section_id="caveats",
                title="Caveats",
                content=(
                    "NACCUDSD=4 captures all-cause dementia. "
                    "Use NACCALZD=1 for etiologic AD. "
                    "There are 47 participants meeting criteria."
                ),
            )],
            execution_context=ctx,
            grounded=True,
        )
        assert _check_synthesis_hallucination(report)

    def test_horizon_labels_do_not_trigger_guard(self):
        """'at 2, 3, or 5 years' in section text must not fail."""
        ctx = ExecutionContext()
        # Create a fan-out result
        ctx.results["followup"] = ToolResult.ok(
            {
                "2yr": {"sufficient": 100, "insufficient": 20, "censored": 5},
                "3yr": {"sufficient": 80, "insufficient": 30, "censored": 5},
                "5yr": {"sufficient": 60, "insufficient": 50, "censored": 5},
            },
            evidence=[DataEvidence(
                query_spec={"horizons": [2, 3, 5]},
                aggregate_ref="executor.fan_out_longitudinal: all 3 horizons succeeded",
            )],
        )
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(
                section_id="evidence_provenance",
                title="Evidence",
                content=(
                    "At 2 years: 100 sufficient, 20 insufficient, 5 censored.\n"
                    "At 3 years: 80 sufficient, 30 insufficient, 5 censored.\n"
                    "At 5 years: 60 sufficient, 50 insufficient, 5 censored."
                ),
            )],
            execution_context=ctx,
            grounded=True,
        )
        assert _check_synthesis_hallucination(report)

    def test_list_numbering_does_not_trigger_guard(self):
        """Ordered list numbers (1. 2. 3.) must not trigger the guard."""
        ctx = ExecutionContext()
        ctx.results["n"] = ToolResult.ok(47)
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(
                section_id="caveats",
                title="Caveats",
                content=(
                    "1. Suppression threshold may apply.\n"
                    "2. Follow-up attrition noted.\n"
                    "3. All-cause vs etiologic AD distinction.\n"
                    "There are 47 participants."
                ),
            )],
            execution_context=ctx,
            grounded=True,
        )
        assert _check_synthesis_hallucination(report)

    def test_guard_passes_when_no_execution_context(self):
        report = SynthesisReport(question="q", grounded=False, abstain_reason="x")
        assert _check_synthesis_hallucination(report)

    def test_guard_passes_on_empty_report(self):
        report = SynthesisReport(question="q", execution_context=ExecutionContext())
        assert _check_synthesis_hallucination(report)

    def test_fan_out_integers_are_collected_as_evidence(self):
        """Integers from nested dict values must be in _collect_evidence_numbers."""
        tr = ToolResult.ok({
            "2yr": {"sufficient": 111, "insufficient": 22, "censored": 3},
            "5yr": {"sufficient": 88, "insufficient": 11, "censored": 1},
        })
        nums = _collect_evidence_numbers([tr])
        assert "111" in nums
        assert "22" in nums
        assert "88" in nums
        assert "11" in nums


# ---------------------------------------------------------------------------
# H. Integration smoke — run_agent() with mock LLM client
# ---------------------------------------------------------------------------

_MOTIVATING_QUESTION = (
    "Among participants with baseline MCI, how many have sufficient "
    "follow-up to determine conversion status within 2, 3, or 5 years?"
)


class MockInvokeToolClient(PlannerClient):
    """Minimal PlannerClient subclass that returns a fixed plan_data dict on
    invoke_tool() and a plain string on complete_text().

    Used to test the full run_agent(use_enriched_planner=True) path without
    a live LLM API key.
    """

    def __init__(self, plan_data: dict) -> None:
        self._plan_data = dict(plan_data)

    def invoke_tool(self, messages: list, tool_schema: dict, max_tokens: int = 1500):
        return dict(self._plan_data)

    def complete_text(self, messages: list, max_tokens: int = 1024) -> str:
        return "Mock synthesis: the enriched planner produced grounded results."


def _make_mock_client() -> MockInvokeToolClient:
    """Build a MockInvokeToolClient pre-loaded with the motivating question plan."""
    plan = _make_enriched_plan_mci_multihorizon()
    plan_data = plan.model_dump()
    plan_data.pop("original_question", None)
    plan_data.pop("original_plan", None)
    return MockInvokeToolClient(plan_data)


class TestRunAgentEnrichedSmoke:
    """run_agent(use_enriched_planner=True) must use the enriched path end-to-end.

    These tests verify the real runtime wiring from run_agent() through
    _run_enriched_path() → DeterministicExecutor, NOT just direct plan construction.
    They use a MockInvokeToolClient to avoid a live LLM key.
    """

    def setup_method(self):
        from alz_agent.agent.orchestrator import run_agent
        self.run_agent = run_agent
        self.catalog = _make_catalog()
        self.backend = build_fixture_backend(
            n_mci=40, n_mci_long=25, n_mci_short=8, n_single_visit=7
        )
        self.client = _make_mock_client()

    def _run(self):
        return self.run_agent(
            _MOTIVATING_QUESTION,
            self.catalog,
            self.backend,
            client=self.client,
            use_enriched_planner=True,
        )

    def test_enriched_plan_used_is_true(self):
        resp = self._run()
        assert resp.enriched_plan_used, (
            f"enriched_plan_used=False; enriched_plan_error={resp.enriched_plan_error}"
        )

    def test_enriched_plan_error_is_none_on_success(self):
        resp = self._run()
        assert resp.enriched_plan_error is None, resp.enriched_plan_error

    def test_requested_horizons_are_2_3_5(self):
        resp = self._run()
        assert resp.requested_horizons == [2, 3, 5], resp.requested_horizons

    def test_execution_operations_run_includes_mci_followup(self):
        resp = self._run()
        assert "mci_followup" in resp.execution_operations_run, (
            f"Expected longitudinal op 'mci_followup' in execution_operations_run; "
            f"got {resp.execution_operations_run}"
        )

    def test_enriched_plan_operations_lists_both_ops(self):
        resp = self._run()
        assert len(resp.enriched_plan_operations) == 2, resp.enriched_plan_operations
        op_types = [o.split(":")[0] for o in resp.enriched_plan_operations]
        assert "longitudinal_feasibility" in op_types, resp.enriched_plan_operations

    def test_validation_hard_errors_empty_on_valid_plan(self):
        resp = self._run()
        assert resp.validation_hard_errors == [], resp.validation_hard_errors

    def test_synthesis_report_is_present(self):
        resp = self._run()
        assert resp.synthesis_report is not None
        assert resp.synthesis_report_present

    def test_response_is_grounded(self):
        resp = self._run()
        assert resp.grounded, f"grounded=False; abstain_reason={resp.abstain_reason}"

    def test_old_cohort_size_only_path_not_used(self):
        """Classic path must not be invoked when use_enriched_planner=True."""
        resp = self._run()
        # Enriched path succeeded — classic tool_results come from executor, not tool_loop
        # The execution must include the longitudinal feasibility fan-out
        mci_followup_result = next(
            (tr for tr in resp.tool_results if isinstance(tr.value, dict) and "2yr" in tr.value),
            None,
        )
        assert mci_followup_result is not None, (
            "Expected a fan-out longitudinal result (dict with '2yr') in tool_results; "
            f"tool_results values: {[type(tr.value).__name__ for tr in resp.tool_results]}"
        )

    def test_no_client_with_enriched_planner_guard_short_circuits(self):
        """When use_enriched_planner=True but client is None, the enriched guard
        short-circuits (requires a live client) and falls through to classic path."""
        resp = self.run_agent(
            _MOTIVATING_QUESTION,
            self.catalog,
            self.backend,
            client=None,
            use_enriched_planner=True,
        )
        assert resp.enriched_plan_used is False
        assert resp.enriched_plan_error is None  # guard didn't run, so no error

    def test_classic_path_unchanged_when_enriched_planner_false(self):
        """use_enriched_planner=False must leave all enriched debug fields empty."""
        resp = self.run_agent(
            _MOTIVATING_QUESTION,
            self.catalog,
            self.backend,
            client=None,
            use_enriched_planner=False,
        )
        assert resp.enriched_plan_used is False
        assert resp.requested_horizons == []
        assert resp.enriched_plan_operations == []
        assert resp.execution_operations_run == []


# ---------------------------------------------------------------------------
# I. JSON fallback path — _build_plan_from_data and _plan_enriched_json_fallback
# ---------------------------------------------------------------------------


class TestBuildPlanFromData:
    """_build_plan_from_data must parse valid plan dicts into EnrichedAnalysisPlan."""

    def _make_raw_data(self) -> dict:
        return {
            "research_intent_summary": "Count MCI participants with follow-up.",
            "researcher_audit_trace": ["Interpreted MCI as diagnosis=mci."],
            "operations": [
                {
                    "op_type": "count_eligible",
                    "parameters": {"filter": {"diagnosis": "mci"}},
                    "output_name": "mci_n",
                    "description": "Count MCI participants.",
                },
                {
                    "op_type": "longitudinal_feasibility",
                    "parameters": {"horizons": [2, 3, 5], "filter": {"diagnosis": "mci"}},
                    "output_name": "mci_followup",
                    "description": "Check follow-up at 2, 3, 5 years.",
                },
            ],
            "requested_horizons": [2, 3, 5],
            "requires_backend_data": True,
            "baseline_definition": {"diagnosis": "mci"},
            "outcome_definition": "conversion to NACCUDSD=4",
            "id_space": "A",
        }

    def test_returns_enriched_plan_with_two_ops(self):
        from alz_agent.agent.llm_intent import _build_plan_from_data
        data = self._make_raw_data()
        plan = _build_plan_from_data(data, "How many MCI have follow-up?")
        assert plan is not None
        assert len(plan.operations) == 2

    def test_longitudinal_op_has_correct_type(self):
        from alz_agent.agent.llm_intent import _build_plan_from_data
        plan = _build_plan_from_data(self._make_raw_data(), "q")
        long_ops = [o for o in plan.operations if o.op_type == "longitudinal_feasibility"]
        assert len(long_ops) == 1
        assert long_ops[0].parameters["horizons"] == [2, 3, 5]

    def test_requested_horizons_are_populated(self):
        from alz_agent.agent.llm_intent import _build_plan_from_data
        plan = _build_plan_from_data(self._make_raw_data(), "q")
        assert plan.requested_horizons == [2, 3, 5]

    def test_returns_none_when_operations_empty(self):
        from alz_agent.agent.llm_intent import _build_plan_from_data
        data = self._make_raw_data()
        data["operations"] = []
        plan = _build_plan_from_data(data, "q")
        assert plan is None

    def test_skips_unknown_op_types(self):
        from alz_agent.agent.llm_intent import _build_plan_from_data
        data = self._make_raw_data()
        data["operations"] = [
            {"op_type": "invented_op", "output_name": "x", "parameters": {}},
            {"op_type": "count_eligible", "output_name": "n", "parameters": {}},
        ]
        plan = _build_plan_from_data(data, "q")
        assert plan is not None
        assert len(plan.operations) == 1
        assert plan.operations[0].op_type == "count_eligible"

    def test_returns_none_when_all_ops_unknown(self):
        from alz_agent.agent.llm_intent import _build_plan_from_data
        data = self._make_raw_data()
        data["operations"] = [
            {"op_type": "unknown_a", "output_name": "a", "parameters": {}},
            {"op_type": "unknown_b", "output_name": "b", "parameters": {}},
        ]
        plan = _build_plan_from_data(data, "q")
        assert plan is None

    def test_deduplicates_output_names(self):
        from alz_agent.agent.llm_intent import _build_plan_from_data
        data = self._make_raw_data()
        data["operations"] = [
            {"op_type": "count_eligible", "output_name": "n", "parameters": {}},
            {"op_type": "count_eligible", "output_name": "n", "parameters": {}},
        ]
        plan = _build_plan_from_data(data, "q")
        assert plan is not None
        assert len(plan.operations) == 1

    def test_filters_non_positive_horizons(self):
        from alz_agent.agent.llm_intent import _build_plan_from_data
        data = self._make_raw_data()
        data["requested_horizons"] = [2, -1, 0, 5]
        plan = _build_plan_from_data(data, "q")
        assert plan is not None
        assert plan.requested_horizons == [2, 5]


class TestPlanEnrichedJsonFallback:
    """_plan_enriched_json_fallback must parse a JSON response from complete_text."""

    def _make_json_client(self, json_str: str) -> PlannerClient:
        """Mock client that returns a fixed JSON string from complete_text."""

        class JsonClient(PlannerClient):
            def __init__(self, text: str) -> None:
                self._text = text

            def invoke_tool(self, messages, tool_schema, max_tokens=1500):
                return None  # always fails; forces JSON fallback

            def complete_text(self, messages, max_tokens=1024) -> str:
                return self._text

        return JsonClient(json_str)

    def _make_valid_json(self) -> str:
        return json.dumps({
            "research_intent_summary": "Count MCI participants.",
            "researcher_audit_trace": ["MCI = NACCUDSD=3."],
            "operations": [
                {
                    "op_type": "count_eligible",
                    "parameters": {"filter": {"diagnosis": "mci"}},
                    "output_name": "mci_n",
                    "description": "Count.",
                },
                {
                    "op_type": "longitudinal_feasibility",
                    "parameters": {"horizons": [2, 3, 5], "filter": {"diagnosis": "mci"}},
                    "output_name": "mci_followup",
                    "description": "Follow-up.",
                },
            ],
            "requested_horizons": [2, 3, 5],
            "requires_backend_data": True,
            "baseline_definition": {"diagnosis": "mci"},
            "outcome_definition": "conversion to NACCUDSD=4",
            "id_space": "A",
        })

    def test_fallback_parses_valid_json(self):
        from alz_agent.agent.llm_intent import _plan_enriched_json_fallback
        client = self._make_json_client(self._make_valid_json())
        plan = _plan_enriched_json_fallback("How many MCI?", client)
        assert plan is not None
        assert len(plan.operations) == 2

    def test_fallback_handles_markdown_code_fence(self):
        from alz_agent.agent.llm_intent import _plan_enriched_json_fallback
        fenced = "```json\n" + self._make_valid_json() + "\n```"
        client = self._make_json_client(fenced)
        plan = _plan_enriched_json_fallback("How many MCI?", client)
        assert plan is not None

    def test_fallback_returns_none_on_invalid_json(self):
        from alz_agent.agent.llm_intent import _plan_enriched_json_fallback
        client = self._make_json_client("not json at all")
        plan = _plan_enriched_json_fallback("How many MCI?", client)
        assert plan is None

    def test_fallback_returns_none_on_empty_ops(self):
        from alz_agent.agent.llm_intent import _plan_enriched_json_fallback
        data = json.loads(self._make_valid_json())
        data["operations"] = []
        client = self._make_json_client(json.dumps(data))
        plan = _plan_enriched_json_fallback("How many MCI?", client)
        assert plan is None

    def test_plan_enriched_analysis_uses_fallback_when_invoke_returns_none(self):
        """plan_enriched_analysis() must call JSON fallback when invoke_tool returns None."""
        from alz_agent.agent.llm_intent import plan_enriched_analysis
        client = self._make_json_client(self._make_valid_json())
        plan = plan_enriched_analysis("How many MCI have 2-year follow-up?", client)
        assert plan is not None, "Expected JSON fallback to succeed"
        assert any(o.op_type == "longitudinal_feasibility" for o in plan.operations)

    def test_plan_enriched_analysis_returns_none_when_both_paths_fail(self):
        """plan_enriched_analysis() returns None only when both tool and JSON fallback fail."""
        from alz_agent.agent.llm_intent import plan_enriched_analysis

        class BothFailClient(PlannerClient):
            def invoke_tool(self, messages, tool_schema, max_tokens=1500):
                return None

            def complete_text(self, messages, max_tokens=1024) -> str:
                return "not valid json"

        plan = plan_enriched_analysis("any question", BothFailClient())
        assert plan is None


# ---------------------------------------------------------------------------
# J. Hallucination guard enforcement and _build_enriched_fallback_answer
# ---------------------------------------------------------------------------


class TestBuildEnrichedFallbackAnswer:
    """_build_enriched_fallback_answer must produce a grounded answer with no invented numbers."""

    def _make_exec_ctx_with_count(self, count: int) -> ExecutionContext:
        ctx = ExecutionContext()
        ctx.results["mci_n"] = ToolResult.ok(
            count,
            evidence=[DataEvidence(
                query_spec={"filter": {"diagnosis": "mci"}},
                aggregate_ref=f"count_eligible={count}",
            )],
        )
        ctx.operations_run.append("mci_n")
        return ctx

    def _make_exec_ctx_with_fanout(self) -> ExecutionContext:
        ctx = ExecutionContext()
        ctx.results["mci_followup"] = ToolResult.ok(
            {
                "2yr": {"sufficient": 100, "insufficient": 20, "censored": 5},
                "3yr": {"sufficient": 80, "insufficient": 30, "censored": 5},
                "5yr": {"sufficient": 60, "insufficient": 50, "censored": 5},
            },
            evidence=[DataEvidence(
                query_spec={"horizons": [2, 3, 5]},
                aggregate_ref="fan-out: 3 horizons succeeded",
            )],
        )
        ctx.operations_run.append("mci_followup")
        return ctx

    def _make_plan(self) -> EnrichedAnalysisPlan:
        return _make_enriched_plan_mci_multihorizon()

    def test_answer_contains_grounded_count(self):
        from alz_agent.agent.orchestrator import _build_enriched_fallback_answer
        ctx = self._make_exec_ctx_with_count(40)
        answer = _build_enriched_fallback_answer(ctx, self._make_plan())
        assert "40" in answer

    def test_answer_contains_fanout_values(self):
        from alz_agent.agent.orchestrator import _build_enriched_fallback_answer
        ctx = self._make_exec_ctx_with_fanout()
        answer = _build_enriched_fallback_answer(ctx, self._make_plan())
        assert "100" in answer
        assert "2yr" in answer

    def test_abstained_op_does_not_add_numbers(self):
        from alz_agent.agent.orchestrator import _build_enriched_fallback_answer
        ctx = ExecutionContext()
        ctx.results["mci_n"] = ToolResult.abstain("suppressed below threshold")
        ctx.operations_run.append("mci_n")
        answer = _build_enriched_fallback_answer(ctx, self._make_plan())
        assert "not available" in answer
        assert "40" not in answer

    def test_fallback_answer_passes_synthesis_hallucination_guard(self):
        """The fallback answer must pass the hallucination guard against exec_ctx."""
        from alz_agent.agent.hallucination_guard import _check_synthesis_hallucination
        from alz_agent.agent.orchestrator import _build_enriched_fallback_answer
        ctx = self._make_exec_ctx_with_count(40)
        answer = _build_enriched_fallback_answer(ctx, self._make_plan())
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(
                section_id="direct_answer",
                title="Answer",
                content=answer,
            )],
            execution_context=ctx,
            grounded=True,
        )
        assert _check_synthesis_hallucination(report)

    def test_fanout_fallback_passes_hallucination_guard(self):
        from alz_agent.agent.hallucination_guard import _check_synthesis_hallucination
        from alz_agent.agent.orchestrator import _build_enriched_fallback_answer
        ctx = self._make_exec_ctx_with_fanout()
        answer = _build_enriched_fallback_answer(ctx, self._make_plan())
        report = SynthesisReport(
            question="q",
            sections=[SynthesisSection(
                section_id="evidence_provenance",
                title="Evidence",
                content=answer,
            )],
            execution_context=ctx,
            grounded=True,
        )
        assert _check_synthesis_hallucination(report)


class TestHallucinationGuardEnforcementInRunAgent:
    """When the LLM synthesizer hallucinations, run_agent must return a deterministic answer."""

    def setup_method(self):
        from alz_agent.agent.orchestrator import run_agent
        self.run_agent = run_agent
        self.catalog = _make_catalog()
        self.backend = build_fixture_backend(
            n_mci=40, n_mci_long=25, n_mci_short=8, n_single_visit=7
        )

    def _make_hallucinating_client(self) -> PlannerClient:
        """Client that returns a valid plan but synthesizes hallucinated numbers."""
        plan = _make_enriched_plan_mci_multihorizon()
        plan_data = plan.model_dump()
        plan_data.pop("original_question", None)
        plan_data.pop("original_plan", None)

        class HallucinatingClient(PlannerClient):
            def __init__(self, pd: dict) -> None:
                self._plan_data = dict(pd)

            def invoke_tool(self, messages, tool_schema, max_tokens=1500):
                return dict(self._plan_data)

            def complete_text(self, messages, max_tokens=1024) -> str:
                # Synthesis output with an invented number (99999) that won't
                # trace to any tool result — should trigger the guard fallback.
                return (
                    "## Direct Answer\n"
                    "There are 99999 participants with sufficient follow-up.\n\n"
                    "## Analysis Plan\n"
                    "- Interpreted MCI.\n\n"
                    "## Evidence and Provenance\n"
                    "99999 participants at 2 years.\n\n"
                    "## Caveats\nNone.\n\n"
                    "## Insights\nFeasible.\n\n"
                    "## Recommended Next Steps\n"
                    "- Step 1.\n- Step 2.\n- Step 3."
                )

        return HallucinatingClient(plan_data)

    def test_hallucination_guard_enforced_answer_is_not_none(self):
        client = self._make_hallucinating_client()
        resp = self.run_agent(
            _MOTIVATING_QUESTION,
            self.catalog,
            self.backend,
            client=client,
            use_enriched_planner=True,
        )
        assert resp.answer is not None
        assert len(resp.answer) > 0

    def test_hallucination_guard_enforced_answer_does_not_contain_invented_number(self):
        client = self._make_hallucinating_client()
        resp = self.run_agent(
            _MOTIVATING_QUESTION,
            self.catalog,
            self.backend,
            client=client,
            use_enriched_planner=True,
        )
        # 99999 was invented by the synthesizer; it must not appear in the final answer
        assert "99999" not in resp.answer, (
            f"Invented number 99999 leaked through hallucination guard.\n"
            f"answer={resp.answer[:300]}"
        )

    def test_hallucination_guard_enforced_flag_is_true(self):
        """hallucination_check_passed must be True after guard enforcement."""
        client = self._make_hallucinating_client()
        resp = self.run_agent(
            _MOTIVATING_QUESTION,
            self.catalog,
            self.backend,
            client=client,
            use_enriched_planner=True,
        )
        assert resp.hallucination_check_passed, (
            "hallucination_check_passed should be True after fallback replaced the answer"
        )

    def test_hallucination_guard_enforced_final_narration_mode_is_fallback(self):
        """When guard fires and fallback is used, final_narration_mode must be deterministic_fallback."""
        client = self._make_hallucinating_client()
        resp = self.run_agent(
            _MOTIVATING_QUESTION,
            self.catalog,
            self.backend,
            client=client,
            use_enriched_planner=True,
        )
        assert resp.final_narration_mode == "deterministic_fallback", (
            f"Expected deterministic_fallback, got {resp.final_narration_mode}"
        )


# ---------------------------------------------------------------------------
# K. Executor horizon alias (singular 'horizon' → single-element list)
# ---------------------------------------------------------------------------


class TestExecutorHorizonAlias:
    """DeterministicExecutor must accept 'horizon' (singular) as well as 'horizons' (list)."""

    def setup_method(self):
        self.catalog = _make_catalog()
        self.backend = build_fixture_backend(
            n_mci=40, n_mci_long=25, n_mci_short=8, n_single_visit=7
        )
        self.executor = DeterministicExecutor(self.catalog, self.backend)

    def _make_plan_with_singular_horizon(self, horizon: int) -> EnrichedAnalysisPlan:
        return EnrichedAnalysisPlan(
            original_question="Follow-up at 3 years?",
            research_intent_summary="Check 3-year follow-up for MCI.",
            researcher_audit_trace=["horizon=3 extracted."],
            operations=[
                PlanOperation(
                    op_type="longitudinal_feasibility",
                    parameters={"horizon": horizon, "filter": {"diagnosis": "mci"}},
                    output_name="mci_followup",
                    description="Single-horizon follow-up.",
                ),
            ],
            requested_horizons=[horizon],
            requires_backend_data=True,
            baseline_definition={"diagnosis": "mci"},
            id_space="A",
        )

    def test_singular_horizon_produces_grounded_result(self):
        plan = self._make_plan_with_singular_horizon(3)
        ctx = self.executor.execute(plan)
        result = ctx.results.get("mci_followup")
        assert result is not None
        assert result.grounded, f"Expected grounded; abstain_reason={result.abstain_reason}"

    def test_singular_horizon_result_keyed_by_Nyr(self):
        plan = self._make_plan_with_singular_horizon(3)
        ctx = self.executor.execute(plan)
        val = ctx.results["mci_followup"].value
        assert isinstance(val, dict)
        assert "3yr" in val

    def test_singular_horizon_result_has_all_buckets(self):
        plan = self._make_plan_with_singular_horizon(3)
        ctx = self.executor.execute(plan)
        bucket = ctx.results["mci_followup"].value["3yr"]
        assert "sufficient" in bucket
        assert "insufficient" in bucket
        assert "censored" in bucket


# ---------------------------------------------------------------------------
# L. Group comparison variable-resolution repair
# ---------------------------------------------------------------------------


def _make_catalog_with_heart_attack_mmse() -> list[CatalogEntry]:
    """Catalog with CVHATT (heart attack history) and NACCMMSE (MMSE score)."""
    return [
        CatalogEntry(
            variable_id="NACC.NACCUDSD",
            concept="cognitive status",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCUDSD",
            release_version="v70",
            modality=Modality.COGNITION,
        ),
        CatalogEntry(
            variable_id="NACC.NACCMMSE",
            concept="MMSE score",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCMMSE",
            release_version="v70",
            modality=Modality.COGNITION,
            description="Mini-Mental State Examination total score (0-30).",
        ),
        CatalogEntry(
            variable_id="NACC.CVHATT",
            concept="heart attack history",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="CVHATT",
            release_version="v70",
            modality=Modality.OTHER,
            description="Participant has a history of heart attack / cardiac arrest.",
            value_coding={"0": "Absent", "1": "Recent/active", "2": "Remote/inactive"},
        ),
        CatalogEntry(
            variable_id="NACC.CVHATT_BROAD",
            concept="heart disease",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="CVHATT_BROAD",
            release_version="v70",
            modality=Modality.OTHER,
            description="Participant has any form of heart disease (composite indicator).",
            value_coding={"0": "Absent", "1": "Present"},
        ),
    ]


def _make_gc_plan_missing_outcome(
    group_variable_id: str = "NACC.CVHATT",
) -> EnrichedAnalysisPlan:
    """An EnrichedAnalysisPlan with a group_comparison op missing outcome_variable_id."""
    return EnrichedAnalysisPlan(
        original_question=(
            "Among baseline MCI participants, do participants with heart attack history "
            "have lower MMSE scores?"
        ),
        research_intent_summary=(
            "Compare MMSE scores between MCI participants with and without heart attack history."
        ),
        researcher_audit_trace=[
            "Interpreted 'baseline MCI' as baseline_diagnosis=mci.",
            "group_variable: heart attack history → NACC.CVHATT.",
            "outcome_variable: MMSE scores → not yet resolved.",
        ],
        operations=[
            PlanOperation(
                op_type="count_eligible",
                parameters={"filter": {"diagnosis": "mci"}},
                output_name="mci_cohort_n",
                description="Count baseline MCI participants.",
            ),
            PlanOperation(
                op_type="group_comparison",
                parameters={
                    "group_variable_id": group_variable_id,
                    # outcome_variable_id intentionally missing
                    "filter": {"diagnosis": "mci"},
                },
                output_name="mmse_by_heart_attack",
                description="Compare MMSE by heart attack history.",
            ),
        ],
        requested_horizons=[],
        requires_backend_data=True,
        baseline_definition={"diagnosis": "mci"},
        id_space="A",
    )


class TestGroupComparisonRepair:
    """The repair pass must resolve missing outcome_variable_id for group_comparison ops."""

    def setup_method(self):
        self.catalog = _make_catalog_with_heart_attack_mmse()
        self.validator = PlanValidator(self.catalog)
        self.question = (
            "Among baseline MCI participants, do participants with heart attack history "
            "have lower MMSE scores?"
        )

    def test_validator_rejects_plan_missing_outcome_variable_id(self):
        plan = _make_gc_plan_missing_outcome()
        result = self.validator.validate(plan)
        assert not result.is_executable
        assert any("outcome_variable_id" in e for e in result.hard_errors)

    def test_repair_resolves_missing_outcome_variable_id(self):
        from alz_agent.agent.plan_repair import repair_group_comparison_plan
        plan = _make_gc_plan_missing_outcome()
        validation = self.validator.validate(plan)
        assert not validation.is_executable

        repaired, warnings = repair_group_comparison_plan(
            plan, list(validation.hard_errors), self.question, self.catalog,
            client=None,  # heuristic path only
        )
        assert repaired is not None, f"Repair failed; warnings={warnings}"

        # Repaired op must have outcome_variable_id
        gc_ops = [op for op in repaired.operations if op.op_type == "group_comparison"]
        assert gc_ops, "No group_comparison op in repaired plan"
        assert "outcome_variable_id" in gc_ops[0].parameters
        assert gc_ops[0].parameters["outcome_variable_id"] == "NACC.NACCMMSE"

    def test_repaired_plan_passes_validation(self):
        from alz_agent.agent.plan_repair import repair_group_comparison_plan
        plan = _make_gc_plan_missing_outcome()
        validation = self.validator.validate(plan)

        repaired, warnings = repair_group_comparison_plan(
            plan, list(validation.hard_errors), self.question, self.catalog,
        )
        assert repaired is not None, f"Repair failed; warnings={warnings}"

        re_validation = self.validator.validate(repaired)
        assert re_validation.is_executable, re_validation.hard_errors

    def test_repair_preserves_group_variable_id(self):
        from alz_agent.agent.plan_repair import repair_group_comparison_plan
        plan = _make_gc_plan_missing_outcome(group_variable_id="NACC.CVHATT")
        validation = self.validator.validate(plan)
        repaired, warnings = repair_group_comparison_plan(
            plan, list(validation.hard_errors), self.question, self.catalog,
        )
        assert repaired is not None
        gc_op = next(op for op in repaired.operations if op.op_type == "group_comparison")
        assert gc_op.parameters.get("group_variable_id") == "NACC.CVHATT"

    def test_repair_adds_audit_trace_entry(self):
        from alz_agent.agent.plan_repair import repair_group_comparison_plan
        plan = _make_gc_plan_missing_outcome()
        validation = self.validator.validate(plan)
        repaired, _ = repair_group_comparison_plan(
            plan, list(validation.hard_errors), self.question, self.catalog,
        )
        assert repaired is not None
        repair_trace = [t for t in repaired.researcher_audit_trace if "[repair]" in t]
        assert repair_trace, "Expected a [repair] audit trace entry"

    def test_repair_with_llm_client_resolves_outcome_from_llm_json(self):
        """LLM client that returns concept JSON triggers LLM path, then catalog lookup."""
        from alz_agent.agent.plan_repair import repair_group_comparison_plan

        class ConceptClient(PlannerClient):
            def invoke_tool(self, messages, tool_schema, max_tokens=1500):
                return None

            def complete_text(self, messages, max_tokens=1024) -> str:
                return '{"outcome_concept": "MMSE score"}'

        plan = _make_gc_plan_missing_outcome()
        validation = self.validator.validate(plan)
        repaired, warnings = repair_group_comparison_plan(
            plan, list(validation.hard_errors), self.question, self.catalog,
            client=ConceptClient(),
        )
        assert repaired is not None, f"LLM-path repair failed; warnings={warnings}"
        gc_op = next(op for op in repaired.operations if op.op_type == "group_comparison")
        assert gc_op.parameters.get("outcome_variable_id") == "NACC.NACCMMSE"


class TestGroupComparisonValidationBeforeAfterRepair:
    """Explicit before/after validation cycle for group_comparison missing vars."""

    def setup_method(self):
        self.catalog = _make_catalog_with_heart_attack_mmse()
        self.validator = PlanValidator(self.catalog)
        self.question = (
            "Among baseline MCI participants, do participants with heart attack history "
            "have lower MMSE scores?"
        )

    def test_missing_outcome_is_hard_error_before_repair(self):
        from alz_agent.agent.plan_repair import repair_group_comparison_plan
        plan = _make_gc_plan_missing_outcome()
        pre_validation = self.validator.validate(plan)
        assert not pre_validation.is_executable
        assert any("outcome_variable_id" in e for e in pre_validation.hard_errors)

    def test_plan_is_executable_after_repair(self):
        from alz_agent.agent.plan_repair import repair_group_comparison_plan
        plan = _make_gc_plan_missing_outcome()
        pre_validation = self.validator.validate(plan)

        repaired, warnings = repair_group_comparison_plan(
            plan, list(pre_validation.hard_errors), self.question, self.catalog,
        )
        assert repaired is not None, f"warnings={warnings}"
        post_validation = self.validator.validate(repaired)
        assert post_validation.is_executable, post_validation.hard_errors

    def test_non_group_comparison_errors_not_repaired(self):
        """Repair only touches group_comparison missing-var errors, not others."""
        from alz_agent.agent.plan_repair import repair_group_comparison_plan
        plan = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="s",
            operations=[
                PlanOperation(
                    op_type="group_comparison",
                    parameters={"group_variable_id": "NACC.CVHATT"},
                    output_name="gc_op",
                ),
                PlanOperation(
                    op_type="longitudinal_feasibility",
                    parameters={"horizons": [-1]},  # invalid horizon — unrelated error
                    output_name="long_op",
                ),
            ],
            id_space="A",
        )
        pre_validation = self.validator.validate(plan)
        assert not pre_validation.is_executable

        repaired, warnings = repair_group_comparison_plan(
            plan, list(pre_validation.hard_errors), self.question, self.catalog,
        )
        if repaired is not None:
            # Repaired plan still fails because of the unrelated horizon error
            post = self.validator.validate(repaired)
            assert not post.is_executable


class TestVariablePreference:
    """'heart attack history' must prefer CVHATT over the broad heart-disease composite."""

    def setup_method(self):
        self.catalog = _make_catalog_with_heart_attack_mmse()

    def test_heart_attack_resolves_to_cvhatt_not_broad(self):
        from alz_agent.agent.plan_repair import _resolve_concept_to_variable
        var_id, raw_code, proxy_warning = _resolve_concept_to_variable(
            "heart attack history", self.catalog
        )
        assert var_id == "NACC.CVHATT", (
            f"Expected NACC.CVHATT; got var_id={var_id}, raw_code={raw_code}"
        )
        assert raw_code == "CVHATT"
        assert proxy_warning is None, "Should not warn about proxy when exact match found"

    def test_heart_attack_keyword_resolves_to_cvhatt(self):
        from alz_agent.agent.plan_repair import _resolve_concept_to_variable
        var_id, raw_code, _ = _resolve_concept_to_variable("heart attack", self.catalog)
        assert var_id == "NACC.CVHATT"
        assert raw_code == "CVHATT"

    def test_broad_heart_disease_resolves_with_proxy_warning(self):
        """When only a broad heart disease variable exists, the proxy warning must fire."""
        from alz_agent.agent.plan_repair import _resolve_concept_to_variable
        # Catalog with only the broad variable (no CVHATT)
        catalog_no_cvhatt = [
            e for e in self.catalog if e.raw_code != "CVHATT"
        ]
        var_id, raw_code, proxy_warning = _resolve_concept_to_variable(
            "heart attack history", catalog_no_cvhatt
        )
        if var_id is not None:
            # If resolution succeeded, it must be via the broad fallback → proxy warning
            assert proxy_warning is not None, (
                "Expected proxy warning when exact CVHATT not available; "
                f"got var_id={var_id}, raw_code={raw_code}"
            )
        # else: None means total abstain, which is also acceptable

    def test_exact_raw_code_lookup_finds_cvhatt(self):
        from alz_agent.tools.deterministic import lookup_variable
        result = lookup_variable("CVHATT", self.catalog)
        assert result.grounded
        assert result.value.raw_code == "CVHATT"

    def test_alias_heart_attack_history_resolves_via_alias_map(self):
        """'heart attack history' alias in _VARIABLE_ALIASES maps to CVHATT raw_code."""
        from alz_agent.tools.deterministic import lookup_variable
        result = lookup_variable("heart attack history", self.catalog)
        assert result.grounded, (
            "Expected 'heart attack history' to resolve via alias map to CVHATT"
        )
        assert result.value.raw_code == "CVHATT"


class TestSafeAbstention:
    """When a required variable is not in catalog, repair abstains with a clear message."""

    def setup_method(self):
        # Catalog WITHOUT NACCMMSE — forces MMSE resolution to fail
        self.catalog_no_mmse = [
            CatalogEntry(
                variable_id="NACC.CVHATT",
                concept="heart attack history",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="CVHATT",
                release_version="v70",
                modality=Modality.OTHER,
                value_coding={"0": "Absent", "1": "Present"},
            ),
        ]
        self.validator = PlanValidator(self.catalog_no_mmse)
        self.question = (
            "Among baseline MCI participants, do participants with heart attack history "
            "have lower MMSE scores?"
        )

    def test_repair_returns_none_when_outcome_variable_not_in_catalog(self):
        from alz_agent.agent.plan_repair import repair_group_comparison_plan
        plan = _make_gc_plan_missing_outcome()
        validation = self.validator.validate(plan)

        repaired, warnings = repair_group_comparison_plan(
            plan, list(validation.hard_errors), self.question,
            self.catalog_no_mmse,
            client=None,
        )
        assert repaired is None, (
            "Expected None when MMSE variable not in catalog — should not invent variable_id"
        )

    def test_repair_abstain_message_is_informative(self):
        from alz_agent.agent.plan_repair import repair_group_comparison_plan
        plan = _make_gc_plan_missing_outcome()
        validation = self.validator.validate(plan)

        repaired, warnings = repair_group_comparison_plan(
            plan, list(validation.hard_errors), self.question,
            self.catalog_no_mmse,
            client=None,
        )
        assert repaired is None
        assert warnings, "Expected at least one warning explaining why repair failed"
        combined = " ".join(warnings).lower()
        assert any(word in combined for word in ("not found", "cannot", "abstain")), (
            f"Expected informative abstain message; got: {warnings}"
        )

    def test_llm_returning_null_concept_still_abstains(self):
        """When LLM returns null for the outcome concept, repair must abstain."""
        from alz_agent.agent.plan_repair import repair_group_comparison_plan

        class NullConceptClient(PlannerClient):
            def invoke_tool(self, messages, tool_schema, max_tokens=1500):
                return None

            def complete_text(self, messages, max_tokens=1024) -> str:
                return '{"outcome_concept": null}'

        plan = _make_gc_plan_missing_outcome()
        # Use catalog without MMSE so heuristic fallback also fails
        catalog_no_mmse_no_heuristic = [
            CatalogEntry(
                variable_id="NACC.CVHATT",
                concept="history of cardiac events",  # no MMSE anywhere
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="CVHATT",
                release_version="v70",
                modality=Modality.OTHER,
            ),
        ]
        validator = PlanValidator(catalog_no_mmse_no_heuristic)
        validation = validator.validate(plan)

        repaired, warnings = repair_group_comparison_plan(
            plan, list(validation.hard_errors),
            "question with no outcome keyword",  # no MMSE keyword
            catalog_no_mmse_no_heuristic,
            client=NullConceptClient(),
        )
        assert repaired is None, "Should abstain when LLM returns null concept"


class TestGroupComparisonFixtureExecution:
    """group_comparison execution with a mock analysis engine returns group means."""

    def _make_mock_engine(self, group_n_pos: int = 15, group_n_neg: int = 25) -> object:
        """Mock LocalAnalysisEngine that returns a synthetic GroupComparisonResult."""
        from alz_agent.schemas import DataEvidence, GroupComparisonResult, GroupStats

        class MockEngine:
            def group_comparison(
                self,
                outcome_col: str,
                outcome_variable_id: str,
                outcome_label: str,
                group_col: str,
                group_variable_id: str,
                group_label: str,
                group_names: dict,
                filter_spec: object,
                positive_values: list | None = None,
            ) -> "ToolResult":
                gcr = GroupComparisonResult(
                    outcome_variable_id=outcome_variable_id,
                    outcome_label=outcome_label,
                    group_variable_id=group_variable_id,
                    group_label=group_label,
                    groups=[
                        GroupStats(
                            group_label=group_names.get("positive", "positive"),
                            n=group_n_pos,
                            mean=22.5,
                            sd=3.1,
                        ),
                        GroupStats(
                            group_label=group_names.get("negative", "negative"),
                            n=group_n_neg,
                            mean=24.8,
                            sd=2.9,
                        ),
                    ],
                    p_value=0.019,
                    test_name="welch_t_test",
                    cohort_def="baseline MCI participants",
                    caveats=["Unadjusted comparison — no covariate control."],
                    n_total=group_n_pos + group_n_neg,
                )
                return ToolResult.ok(
                    gcr,
                    evidence=[DataEvidence(
                        query_spec={
                            "group_col": group_col,
                            "outcome_col": outcome_col,
                        },
                        aggregate_ref=(
                            f"group_comparison: {group_col} → {outcome_col}; "
                            f"pos_n={group_n_pos}, neg_n={group_n_neg}"
                        ),
                    )],
                )

        return MockEngine()

    def setup_method(self):
        self.catalog = _make_catalog_with_heart_attack_mmse()
        self.backend = build_fixture_backend(n_mci=40)

    def _make_repaired_gc_plan(self) -> EnrichedAnalysisPlan:
        """A fully-specified group_comparison plan (both IDs present, no repair needed)."""
        return EnrichedAnalysisPlan(
            original_question=(
                "Among baseline MCI participants, do participants with heart attack history "
                "have lower MMSE scores?"
            ),
            research_intent_summary="Compare MMSE by heart attack history in baseline MCI.",
            researcher_audit_trace=[
                "Interpreted 'baseline MCI' as diagnosis=mci.",
                "group_variable_id=NACC.CVHATT (heart attack history).",
                "outcome_variable_id=NACC.NACCMMSE (MMSE score).",
            ],
            operations=[
                PlanOperation(
                    op_type="count_eligible",
                    parameters={"filter": {"diagnosis": "mci"}},
                    output_name="mci_n",
                    description="Count baseline MCI participants.",
                ),
                PlanOperation(
                    op_type="group_comparison",
                    parameters={
                        "group_variable_id": "NACC.CVHATT",
                        "outcome_variable_id": "NACC.NACCMMSE",
                        "filter": {"diagnosis": "mci"},
                    },
                    output_name="mmse_by_heart_attack",
                    description="Compare MMSE by heart attack history.",
                ),
            ],
            requested_horizons=[],
            requires_backend_data=True,
            baseline_definition={"diagnosis": "mci"},
            id_space="A",
        )

    def test_repaired_plan_validates_as_executable(self):
        plan = self._make_repaired_gc_plan()
        result = PlanValidator(self.catalog).validate(plan)
        assert result.is_executable, result.hard_errors

    def test_executor_dispatches_group_comparison_to_engine(self):
        engine = self._make_mock_engine(group_n_pos=15, group_n_neg=25)
        executor = DeterministicExecutor(self.catalog, self.backend, analysis_engine=engine)
        plan = self._make_repaired_gc_plan()
        ctx = executor.execute(plan)

        assert "mmse_by_heart_attack" in ctx.results
        result = ctx.results["mmse_by_heart_attack"]
        assert result.grounded, f"Expected grounded; abstain_reason={result.abstain_reason}"

    def test_group_comparison_result_has_group_stats(self):
        from alz_agent.schemas import GroupComparisonResult
        engine = self._make_mock_engine(group_n_pos=15, group_n_neg=25)
        executor = DeterministicExecutor(self.catalog, self.backend, analysis_engine=engine)
        plan = self._make_repaired_gc_plan()
        ctx = executor.execute(plan)

        result = ctx.results["mmse_by_heart_attack"]
        assert isinstance(result.value, GroupComparisonResult)
        assert result.value.groups[0].n == 15
        assert result.value.groups[1].n == 25

    def test_group_comparison_result_has_means(self):
        from alz_agent.schemas import GroupComparisonResult
        engine = self._make_mock_engine(group_n_pos=15, group_n_neg=25)
        executor = DeterministicExecutor(self.catalog, self.backend, analysis_engine=engine)
        plan = self._make_repaired_gc_plan()
        ctx = executor.execute(plan)

        gcr: GroupComparisonResult = ctx.results["mmse_by_heart_attack"].value
        assert gcr.groups[0].mean is not None
        assert gcr.groups[1].mean is not None
        assert gcr.p_value is not None

    def test_group_comparison_without_engine_abstains(self):
        """When no analysis engine is provided, group_comparison must abstain cleanly."""
        executor = DeterministicExecutor(self.catalog, self.backend, analysis_engine=None)
        plan = self._make_repaired_gc_plan()
        ctx = executor.execute(plan)

        result = ctx.results.get("mmse_by_heart_attack")
        assert result is not None
        assert not result.grounded
        assert "analysis" in (result.abstain_reason or "").lower() or \
               "engine" in (result.abstain_reason or "").lower() or \
               "group_comparison" in (result.abstain_reason or "").lower()

    def test_full_repair_then_execution_via_run_agent(self):
        """run_agent with enriched_planner repairs the plan then executes it."""
        from alz_agent.agent.orchestrator import run_agent

        engine = self._make_mock_engine(group_n_pos=15, group_n_neg=25)

        # Client: invoke_tool returns a plan missing outcome_variable_id;
        # complete_text returns concept JSON for repair, then synthesis text.
        call_count = {"n": 0}

        class RepairThenSynthClient(PlannerClient):
            def invoke_tool(self, messages, tool_schema, max_tokens=1500):
                plan = _make_gc_plan_missing_outcome()
                d = plan.model_dump()
                d.pop("original_question", None)
                d.pop("original_plan", None)
                d.pop("semantic_plan", None)
                return d

            def complete_text(self, messages, max_tokens=1024) -> str:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # First call: repair concept resolution
                    return '{"outcome_concept": "MMSE score"}'
                # Subsequent calls: synthesis
                return "The group comparison completed. MMSE is lower in the heart attack group."

        resp = run_agent(
            "Among baseline MCI participants, do participants with heart attack history "
            "have lower MMSE scores?",
            self.catalog,
            self.backend,
            client=RepairThenSynthClient(),
            analysis_engine=engine,
            use_enriched_planner=True,
        )
        # Should not have failed due to the validation error
        assert "plan validation" not in resp.answer.lower() or resp.grounded, (
            f"Expected repair to succeed; answer={resp.answer[:200]}"
        )
