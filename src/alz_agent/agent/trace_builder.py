"""P5 analysis trace builder — deterministic audit trail, no LLM calls.

Builds a structured AnalysisTrace from pipeline execution objects.
Called after every run_agent() invocation.

Public function:
  _build_analysis_trace(nl_question, intent, filter_spec, tool_results, *, modeling_design_plan) -> AnalysisTrace
"""

from __future__ import annotations

from typing import Any

from alz_agent.agent.intent_schema import StructuredIntent
from alz_agent.schemas import (
    AnalysisPlan,
    AnalysisTrace,
    DataEvidence,
    DescriptiveStatResult,
    FilterSpec,
    GroupComparisonResult,
    LeakageFinding,
    ModelingDesignPlan,
    RegressionResult,
    ToolCallTrace,
    ToolResult,
    VariableResolutionTrace,
)
from alz_agent.tools.deterministic import ModelingDesignResult, StatResult

try:
    from alz_agent.schemas import GeneratedAnalysisResult
except ImportError:
    GeneratedAnalysisResult = None  # type: ignore[assignment,misc]


def _infer_tool_name(tr: ToolResult) -> str:
    """Infer which deterministic tool produced this ToolResult from its value type."""
    if not tr.grounded:
        return "unknown"
    val = tr.value
    if isinstance(val, int):
        return "compute_cohort_size"
    if isinstance(val, StatResult):
        return "compute_variable_stat"
    if isinstance(val, DescriptiveStatResult):
        return "compute_descriptive_stat"
    if isinstance(val, GroupComparisonResult):
        return "group_comparison"
    if isinstance(val, ModelingDesignResult):
        return "suggest_modeling_design"
    if GeneratedAnalysisResult is not None and isinstance(val, GeneratedAnalysisResult):
        return "run_generated_analysis"
    if isinstance(val, RegressionResult):
        return "run_adjusted_regression"
    if isinstance(val, dict):
        # dict is returned by assess_modality_overlap / assess_longitudinal_feasibility
        if any(k in val for k in ("sufficient", "insufficient", "censored")):
            return "assess_longitudinal_feasibility"
        return "assess_modality_overlap"
    return "lookup_variable"  # VariableLookupResult or any other structured result


def _build_output_summary(tr: ToolResult) -> dict[str, Any] | None:
    """Extract numeric-provenance summary from a grounded ToolResult."""
    if not tr.grounded or tr.value is None:
        return None
    val = tr.value
    summary: dict[str, Any] = {}

    if isinstance(val, int):
        summary["n"] = val
    elif isinstance(val, StatResult):
        summary["variable_id"] = val.variable_id
        summary["statistic"] = val.stat_type
        summary["value"] = val.value
        summary["n"] = val.n
        if val.event_anchor:
            summary["event_anchor"] = val.event_anchor
    elif isinstance(val, DescriptiveStatResult):
        summary["variable_id"] = val.variable_id
        summary["count"] = val.count
        summary["mean"] = val.mean
        summary["median"] = val.median
        summary["sd"] = val.sd
        summary["n_missing"] = val.n_missing
    elif isinstance(val, GroupComparisonResult):
        summary["n_total"] = val.n_total
        summary["outcome_variable_id"] = val.outcome_variable_id
        summary["group_variable_id"] = val.group_variable_id
        summary["p_value"] = val.p_value
        summary["groups"] = [
            {"label": g.group_label, "n": g.n, "mean": g.mean, "sd": g.sd}
            for g in val.groups
        ]
    elif isinstance(val, ModelingDesignResult):
        summary["eligible_n"] = val.eligible_n
        summary["horizon_years"] = val.horizon_years
        summary["feasibility"] = dict(val.feasibility)
        summary["feature_whitelist_count"] = len(val.feature_whitelist)
        summary["leakage_blacklist_count"] = len(val.leakage_blacklist)
    elif GeneratedAnalysisResult is not None and isinstance(val, GeneratedAnalysisResult):
        summary["n_analyzed"] = val.n_analyzed
        summary["results"] = dict(val.results)
    elif isinstance(val, RegressionResult):
        summary["n"] = val.n
        summary["outcome_label"] = val.outcome_label
        summary["regression_type"] = val.regression_type
        summary["coefficients"] = [
            {"variable": c.variable, "coefficient": c.coefficient, "p_value": c.p_value}
            for c in val.coefficients
        ]
    elif isinstance(val, dict):
        summary = {k: v for k, v in val.items()}
    else:
        summary["repr"] = repr(val)[:200]

    # Attach evidence aggregate_refs for full provenance
    evidence_refs = [ev.aggregate_ref for ev in tr.evidence if isinstance(ev, DataEvidence)]
    if evidence_refs:
        summary["aggregate_refs"] = evidence_refs

    return summary


def _infer_input_summary(tr: ToolResult) -> dict[str, Any]:
    """Reconstruct a minimal input summary from the tool result value."""
    if not tr.grounded or tr.value is None:
        return {}
    val = tr.value
    if isinstance(val, StatResult):
        return {
            "variable_id": val.variable_id,
            "stat_type": val.stat_type,
            "event_anchor": val.event_anchor,
        }
    if isinstance(val, DescriptiveStatResult):
        return {"variable_id": val.variable_id}
    if isinstance(val, GroupComparisonResult):
        return {
            "outcome_variable_id": val.outcome_variable_id,
            "group_variable_id": val.group_variable_id,
        }
    if isinstance(val, ModelingDesignResult):
        return {"horizon_years": val.horizon_years}
    if isinstance(val, RegressionResult):
        return {
            "outcome_variable_id": val.outcome_variable_id,
            "regression_type": val.regression_type,
        }
    return {}


def _build_tool_call_traces(tool_results: list[ToolResult]) -> list[ToolCallTrace]:
    traces: list[ToolCallTrace] = []
    for tr in tool_results:
        traces.append(ToolCallTrace(
            tool_name=_infer_tool_name(tr),
            input_summary=_infer_input_summary(tr),
            output_summary=_build_output_summary(tr),
            grounded=tr.grounded,
            abstain_reason=tr.abstain_reason,
        ))
    return traces


def _build_variable_resolutions(plan: AnalysisPlan | None) -> list[VariableResolutionTrace]:
    """Build variable resolution traces from all roles in an AnalysisPlan."""
    if plan is None:
        return []

    # Index leakage findings by raw_code for fast lookup
    leakage_by_code: dict[str, str] = {
        f.raw_code: f.reason for f in plan.leakage_assessment
    }

    resolutions: list[VariableResolutionTrace] = []
    named_roles = [
        ("cohort", plan.cohort),
        ("event", plan.event),
        ("value_variable", plan.value_variable),
        ("group_variable", plan.group_variable),
        ("outcome_variable", plan.outcome_variable),
    ]
    for role_name, role in named_roles:
        if role is None:
            continue
        evidence_str = (
            leakage_by_code.get(role.selected_raw_code or "")
            if role.selected_raw_code
            else None
        )
        resolutions.append(VariableResolutionTrace(
            role=role_name,
            researcher_label=role.researcher_label,
            selected_raw_code=role.selected_raw_code,
            semantic_definition=role.semantic_definition,
            filter_expression=role.filter_expression,
            resolution_status=role.resolution_status,
            validation_status=role.validation_status,
            evidence=evidence_str,
        ))

    for cov in plan.covariates:
        resolutions.append(VariableResolutionTrace(
            role="covariate",
            researcher_label=cov.researcher_label,
            selected_raw_code=cov.selected_raw_code,
            semantic_definition=cov.semantic_definition,
            filter_expression=cov.filter_expression,
            resolution_status=cov.resolution_status,
            validation_status=cov.validation_status,
        ))

    return resolutions


def _derive_plan_steps(intent: StructuredIntent) -> list[str]:
    """Return plan_steps: from AnalysisPlan.execution_steps when present, else derived."""
    if intent.analysis_plan and intent.analysis_plan.execution_steps:
        return list(intent.analysis_plan.execution_steps)

    # Heuristic derivation for non-plan / heuristic-mode runs
    steps: list[str] = []
    steps.append(f"Classify question as {intent.question_type.value}.")
    if intent.target_population:
        steps.append(f"Apply cohort filter: {', '.join(intent.target_population)}.")
    if intent.target_etiology:
        steps.append(f"Apply etiology filter: {', '.join(intent.target_etiology)}.")
    if intent.analysis_type:
        steps.append(f"Analysis type: {intent.analysis_type}.")
    if intent.value_column:
        steps.append(
            f"Measure {intent.value_variable_label!r} using column {intent.value_column}."
        )
    if intent.event_column:
        steps.append(
            f"Define event using {intent.event_column} "
            f"with filter {intent.event_filter}."
        )
    if intent.event_anchor:
        steps.append(f"Visit anchor: {intent.event_anchor}.")
    return steps


def _build_caveats(
    intent: StructuredIntent,
    leakage_findings: list[LeakageFinding],
) -> list[str]:
    """Collect caveats from leakage findings, AD semantics, and backend limitations."""
    caveats: list[str] = []

    # Leakage findings that warrant researcher attention
    for f in leakage_findings:
        if f.severity in ("allow_with_caveat", "warn"):
            caveats.append(f"[{f.raw_code}/{f.role}] {f.reason}")

    # AD-specificity semantics caveat
    if intent.target_etiology and "ad" in intent.target_etiology:
        caveats.append(
            "NACCALZD==1 means cognitively impaired with AD etiologic diagnosis, "
            "not necessarily AD dementia. AD dementia requires NACCALZD==1 AND NACCUDSD==4."
        )
        caveats.append(
            "UDS v1/v2 and v3 use different AD etiologic criteria — "
            "compare across protocol versions with care."
        )

    # Event-anchored stat: warn only when using baseline_etiology (not event_etiology)
    # for a first-diagnosis anchor, which silently drops incident AD cases.
    if (
        intent.event_column == "NACCALZD"
        and intent.analysis_type in ("event_anchored_stat", "aggregate_stat")
        and intent.analysis_plan is not None
    ):
        filter_spec = getattr(intent.analysis_plan, "_resolved_filter_spec", None)
        uses_event_etiology = (
            filter_spec is not None
            and getattr(filter_spec, "event_etiology", None) is not None
        )
        if not uses_event_etiology:
            caveats.append(
                "Event filter uses baseline_etiology (NACCALZD==1 at first visit only); "
                "incident AD cases (first diagnosis after enrollment) are excluded. "
                "Use event_etiology filter for all-participant first-AD-diagnosis stats."
            )

    # Unresolved variables
    if intent.analysis_plan and intent.analysis_plan.unresolved_items:
        for item in intent.analysis_plan.unresolved_items:
            caveats.append(f"Unresolved variable: {item}")

    # General unresolved fields (heuristic path)
    for field in (intent.unresolved_fields or []):
        caveats.append(f"Unresolved field: {field}")

    return caveats


def _build_analysis_trace(
    nl_question: str,
    intent: StructuredIntent,
    filter_spec: FilterSpec,
    tool_results: list[ToolResult],
    *,
    modeling_design_plan: ModelingDesignPlan | None = None,
) -> AnalysisTrace:
    """Build a structured audit trace from pipeline execution objects.

    Called after every run_agent() invocation.  All fields are derived
    deterministically — no LLM calls, no generated numbers.
    """
    plan = intent.analysis_plan

    # For modeling_design evidence plans, derive plan_steps from evidence items.
    if modeling_design_plan is not None:
        plan_steps = [
            f"[{item.item_id}] {item.description}"
            for item in modeling_design_plan.evidence_items
        ]
    else:
        plan_steps = _derive_plan_steps(intent)
    variable_resolutions = _build_variable_resolutions(plan)

    filter_trace: dict[str, Any] = {
        "id_space": filter_spec.id_space.value,
        "baseline_diagnosis": filter_spec.baseline_diagnosis,
        "baseline_etiology": filter_spec.baseline_etiology,
        "event_etiology": filter_spec.event_etiology,
        "visit_window": list(filter_spec.visit_window) if filter_spec.visit_window else None,
        "required_modalities": filter_spec.required_modalities,
    }

    leakage_findings: list[LeakageFinding] = list(intent.leakage_assessment)
    tool_calls = _build_tool_call_traces(tool_results)
    caveats = _build_caveats(intent, leakage_findings)

    validation_errors: list[str] = []
    failure_mode: str | None = None
    if intent.parse_debug:
        validation_errors = list(intent.parse_debug.validation_errors)
        failure_mode = intent.parse_debug.failure_mode

    return AnalysisTrace(
        original_question=nl_question,
        parsing_mode=intent.parsing_mode,
        analysis_type=intent.analysis_type,
        question_type=intent.question_type.value,
        research_intent_summary=(
            modeling_design_plan.modeling_goal if modeling_design_plan
            else (plan.research_intent_summary if plan else None)
        ),
        unit_of_analysis=intent.unit_of_analysis,
        statistic=(plan.statistic if plan else intent.stat_type) or None,
        visit_anchor=intent.event_anchor,
        plan_steps=plan_steps,
        variable_resolutions=variable_resolutions,
        filter_trace=filter_trace,
        leakage_findings=leakage_findings,
        tool_calls=tool_calls,
        caveats=caveats,
        validation_errors=validation_errors,
        failure_mode=failure_mode,
        modeling_design_plan=modeling_design_plan,
    )
