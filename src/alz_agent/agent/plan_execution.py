"""AnalysisPlan execution — route a validated plan to backend tools.

All functions here read from an AnalysisPlan (already validated) and call
deterministic tool functions.  No LLM calls, no number generation.

Public functions:
  _build_filter_spec_from_plan(plan, id_space)                       -> FilterSpec
  _run_plan(plan, catalog, backend, filter_spec, *, analysis_engine) -> list[ToolResult]
  _plan_aggregate_stat(...)    — aggregate_stat / event_anchored_stat
  _plan_group_comparison(...)  — group_comparison via analysis engine
  _plan_descriptive_stat(...)  — descriptive_stat
  _plan_regression(...)        — regression via analysis engine
  _infer_positive_values(...)  — coding → positive integer codes
"""

from __future__ import annotations

import sys
from typing import Any

from alz_agent.agent.event_decomposition import _is_compound_event_phrase
from alz_agent.schemas import (
    AnalysisPlan,
    CatalogEntry,
    FilterSpec,
    IdSpace,
    ToolResult,
)
from alz_agent.tools.deterministic import (
    compute_descriptive_stat,
    compute_variable_stat,
    lookup_variable,
    run_adjusted_regression,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Analysis types handled by the plan router.  Others fall through to the
# old question_type-based routing using the derived StructuredIntent fields.
_PLAN_SUPPORTED_TYPES = frozenset({
    "group_comparison", "event_anchored_stat", "aggregate_stat",
    "descriptive_stat", "regression",
})

_VISIT_ANCHOR_TO_BACKEND: dict[str, str] = {
    "baseline": "baseline",
    "first_matching_event_visit": "first_diagnosis",
    "last_visit": "last_visit",
    "first_conversion_visit": "first_conversion_visit",
}

_NACCUDSD_CODE_TO_LABEL: dict[int, str] = {
    1: "normal", 2: "impaired-not-mci", 3: "mci", 4: "dementia",
}


# ---------------------------------------------------------------------------
# FilterSpec builder from plan
# ---------------------------------------------------------------------------


def _build_filter_spec_from_plan(plan: AnalysisPlan, id_space: IdSpace) -> FilterSpec:
    """Derive a FilterSpec from the plan's cohort and event filter expressions.

    Cohort filter → baseline_diagnosis / baseline_etiology (first-visit filters).
    Event filter  → event_etiology when NACCALZD==1 (any-visit longitudinal filter).

    Using event_etiology instead of baseline_etiology for the event prevents the
    prior bug where participants who first received NACCALZD==1 after visit 1
    (incident AD) were silently excluded from "mean age at first AD diagnosis" queries.
    """
    baseline_diag: list[str] | None = None
    baseline_etio: list[str] | None = None
    event_etio: list[str] | None = None

    cohort_fx = (plan.cohort.filter_expression or {}) if plan.cohort else {}
    event_fx = (plan.event.filter_expression or {}) if plan.event else {}

    if "NACCUDSD" in cohort_fx:
        lbl = _NACCUDSD_CODE_TO_LABEL.get(int(cohort_fx["NACCUDSD"]))
        if lbl:
            baseline_diag = [lbl]
    if "NACCALZD" in cohort_fx and cohort_fx["NACCALZD"] == 1:
        # Cohort filter = AD etiology defines WHO is in the cohort at baseline.
        baseline_etio = ["ad"]

    if "NACCALZD" in event_fx and event_fx["NACCALZD"] == 1:
        # Event filter = AD etiology defines the EVENT VISIT to anchor on,
        # so use event_etiology (any-visit) rather than baseline_etiology.
        event_etio = ["ad"]
    if "NACCUDSD" in event_fx and baseline_diag is None:
        lbl = _NACCUDSD_CODE_TO_LABEL.get(int(event_fx["NACCUDSD"]))
        if lbl:
            baseline_diag = [lbl]

    return FilterSpec(
        baseline_diagnosis=baseline_diag,
        baseline_etiology=baseline_etio,
        event_etiology=event_etio,
        id_space=id_space,
    )


# ---------------------------------------------------------------------------
# Plan router
# ---------------------------------------------------------------------------


def _run_plan(
    plan: AnalysisPlan,
    catalog: list[CatalogEntry],
    backend: Any,
    filter_spec: FilterSpec,
    *,
    analysis_engine: Any = None,
) -> list[ToolResult]:
    """Dispatch an AnalysisPlan to the appropriate backend / tool."""
    at = plan.analysis_type

    if at in ("aggregate_stat", "event_anchored_stat"):
        return _plan_aggregate_stat(plan, catalog, backend, filter_spec)

    if at == "group_comparison":
        if analysis_engine is None:
            return [ToolResult.abstain(
                "group_comparison requires a LocalAnalysisEngine "
                "(not available with the current backend configuration)"
            )]
        return _plan_group_comparison(plan, catalog, filter_spec, analysis_engine)

    if at == "descriptive_stat":
        return _plan_descriptive_stat(plan, catalog, backend, filter_spec)

    if at == "regression":
        if analysis_engine is None:
            return [ToolResult.abstain(
                "regression analysis requires a LocalAnalysisEngine "
                "(not available with the current backend configuration)"
            )]
        return _plan_regression(plan, catalog, filter_spec, analysis_engine)

    return [ToolResult.abstain(
        f"Analysis type '{at}' is recognised in the plan but execution is "
        "not yet implemented.  The analysis plan was successfully created."
    )]


# ---------------------------------------------------------------------------
# Individual plan executors
# ---------------------------------------------------------------------------


def _plan_aggregate_stat(
    plan: AnalysisPlan,
    catalog: list[CatalogEntry],
    backend: Any,
    filter_spec: FilterSpec,
) -> list[ToolResult]:
    """Execute aggregate_stat or event_anchored_stat via plan.value_variable.

    When the LLM planner returns selected_raw_code=null for the value_variable,
    a role-aware fallback resolver tries lookup_variable(researcher_label, catalog)
    before failing.  This handles the common case where the LLM correctly labels
    the variable ('age') but omits the catalog mapping ('NACCAGE').
    """
    vv = plan.value_variable
    if vv is None:
        return [ToolResult.abstain(
            "analysis plan has no value_variable — cannot compute aggregate statistic"
        )]

    resolved_code: str | None = vv.selected_raw_code

    if resolved_code is None:
        # Distinguish compound-phrase decomposition failure from unknown variable.
        if vv.researcher_label and _is_compound_event_phrase(vv.researcher_label):
            return [ToolResult.abstain(
                f"value_variable label {vv.researcher_label!r} is a compound event phrase "
                "and cannot be resolved to a single catalog column. "
                "Decompose: value_variable = simple attribute (e.g. 'age' → NACCAGE), "
                "event = anchor condition (e.g. NACCALZD==1), "
                "visit_anchor = first_matching_event_visit."
            )]

        # Role-aware fallback: try lookup_variable on the researcher label.
        if vv.researcher_label:
            fallback = lookup_variable(vv.researcher_label, catalog, debug=True)
            if fallback.grounded and fallback.value:
                resolved_code = fallback.value.raw_code
                print(
                    f"[resolver-fallback] value_variable '{vv.researcher_label}' "
                    f"→ {resolved_code!r} via lookup_variable fallback",
                    file=sys.stderr,
                )
            else:
                # Debug: list what was considered and why it failed.
                sample_candidates = [
                    f"{e.raw_code}|{e.concept}|{e.description[:50]}"
                    for e in catalog[:15]
                ]
                print(
                    f"[resolver-debug] value_variable '{vv.researcher_label}' unresolved.\n"
                    f"  LLM resolution_status: {vv.resolution_status}\n"
                    f"  Catalog sample (first 15): {sample_candidates}",
                    file=sys.stderr,
                )
                return [ToolResult.abstain(
                    f"value_variable '{vv.researcher_label}' could not be resolved to a "
                    f"catalog column (status: {vv.resolution_status})"
                )]
        else:
            return [ToolResult.abstain(
                "value_variable has no researcher_label — cannot attempt fallback resolution"
            )]

    lookup = lookup_variable(resolved_code, catalog)
    if not lookup.grounded:
        return [ToolResult.abstain(
            f"value_variable column '{resolved_code}' not found in catalog"
        )]

    var_id = lookup.value.variable_id
    stat = plan.statistic or "mean"
    event_anchor = _VISIT_ANCHOR_TO_BACKEND.get(plan.visit_anchor or "") if plan.visit_anchor else None

    return [compute_variable_stat(var_id, stat, filter_spec, backend, event_anchor=event_anchor)]


def _infer_positive_values(coding: dict[str, str]) -> list[int] | None:
    """Infer which integer codes count as the 'positive' group from value_coding.

    Excludes 0 (always negative), negative keys (e.g. -4 = not applicable),
    and keys whose label suggests unknown/not-assessed status.
    Returns None when the coding is empty (caller should use default: all nonzero).
    """
    _EXCLUDE_LABEL_TOKENS = frozenset({
        "unknown", "n/a", "not", "applicable", "missing", "assessed", "refused",
    })
    positive: list[int] = []
    for k, label in coding.items():
        try:
            k_int = int(k)
        except (ValueError, TypeError):
            continue
        if k_int <= 0:
            continue
        label_tokens = set(label.lower().split())
        if label_tokens & _EXCLUDE_LABEL_TOKENS:
            continue
        positive.append(k_int)
    return positive if positive else None


def _plan_group_comparison(
    plan: AnalysisPlan,
    catalog: list[CatalogEntry],
    filter_spec: FilterSpec,
    analysis_engine: Any,
) -> list[ToolResult]:
    """Execute group_comparison via plan.group_variable and plan.outcome_variable."""
    gv = plan.group_variable
    ov = plan.outcome_variable
    if gv is None or gv.selected_raw_code is None:
        return [ToolResult.abstain(
            "group_comparison requires a resolved group_variable in the analysis plan"
        )]
    if ov is None or ov.selected_raw_code is None:
        return [ToolResult.abstain(
            "group_comparison requires a resolved outcome_variable in the analysis plan"
        )]

    group_lookup = lookup_variable(gv.selected_raw_code, catalog)
    if not group_lookup.grounded:
        return [ToolResult.abstain(
            f"group_variable '{gv.selected_raw_code}' not found in catalog"
        )]
    outcome_lookup = lookup_variable(ov.selected_raw_code, catalog)
    if not outcome_lookup.grounded:
        return [ToolResult.abstain(
            f"outcome_variable '{ov.selected_raw_code}' not found in catalog"
        )]

    group_entry = group_lookup.value
    outcome_entry = outcome_lookup.value

    coding = group_entry.value_coding
    neg_label = coding.get("0", f"no {group_entry.concept}")
    pos_values = _infer_positive_values(coding)
    pos_label_candidates = [
        v for k, v in coding.items()
        if k.lstrip("-").isdigit() and int(k) > 0
        and v.lower() not in ("unknown", "n/a", "not applicable")
    ]
    pos_label = pos_label_candidates[0] if pos_label_candidates else group_entry.concept
    group_names = {"positive": pos_label, "negative": neg_label}

    result = analysis_engine.group_comparison(
        outcome_col=outcome_entry.raw_code,
        outcome_variable_id=outcome_entry.variable_id,
        outcome_label=outcome_entry.concept,
        group_col=group_entry.raw_code,
        group_variable_id=group_entry.variable_id,
        group_label=group_entry.concept,
        group_names=group_names,
        filter_spec=filter_spec,
        positive_values=pos_values,
    )
    return [result]


def _plan_descriptive_stat(
    plan: AnalysisPlan,
    catalog: list[CatalogEntry],
    backend: Any,
    filter_spec: FilterSpec,
) -> list[ToolResult]:
    """Execute descriptive_stat via plan.value_variable."""
    vv = plan.value_variable
    if vv is None or vv.selected_raw_code is None:
        return [ToolResult.abstain(
            "descriptive_stat requires a resolved value_variable in the analysis plan"
        )]
    lookup = lookup_variable(vv.selected_raw_code, catalog)
    if not lookup.grounded:
        return [ToolResult.abstain(
            f"value_variable '{vv.selected_raw_code}' not found in catalog"
        )]
    return [compute_descriptive_stat(lookup.value.variable_id, filter_spec, backend)]


def _plan_regression(
    plan: AnalysisPlan,
    catalog: list[CatalogEntry],
    filter_spec: FilterSpec,
    analysis_engine: Any,
) -> list[ToolResult]:
    """Execute regression via plan.outcome_variable + plan.covariates."""
    ov = plan.outcome_variable
    if ov is None or ov.selected_raw_code is None:
        return [ToolResult.abstain(
            "regression requires a resolved outcome_variable in the analysis plan"
        )]
    outcome_lookup = lookup_variable(ov.selected_raw_code, catalog)
    if not outcome_lookup.grounded:
        return [ToolResult.abstain(
            f"outcome_variable '{ov.selected_raw_code}' not found in catalog"
        )]
    outcome_entry = outcome_lookup.value

    covariate_var_ids: list[str] = []
    covariate_cols: list[str] = []
    for cov in plan.covariates:
        if cov.selected_raw_code:
            lk = lookup_variable(cov.selected_raw_code, catalog)
            if lk.grounded and lk.value:
                covariate_var_ids.append(lk.value.variable_id)
                covariate_cols.append(lk.value.raw_code)

    if not covariate_cols:
        return [ToolResult.abstain(
            "regression requires at least one resolved covariate in the analysis plan"
        )]

    reg_type = "logistic" if "logistic" in plan.analysis_type else "linear"

    return [run_adjusted_regression(
        outcome_variable_id=outcome_entry.variable_id,
        outcome_col=outcome_entry.raw_code,
        outcome_label=outcome_entry.concept,
        covariate_variable_ids=covariate_var_ids,
        covariate_cols=covariate_cols,
        filter_spec=filter_spec,
        analysis_engine=analysis_engine,
        regression_type=reg_type,
    )]
