"""Statistical analysis dispatch — route STATISTICAL_ANALYSIS intents to execution.

Handles pre-built templates (group_comparison, regression) and the LLM
code-generation fallback for novel analyses.

Public functions:
  _run_statistical_analysis(...)  — main dispatcher
  _run_group_comparison(...)      — pre-built group comparison template
  _run_regression(...)            — pre-built adjusted regression template
  _run_generated_analysis(...)    — LLM code-generation path
"""

from __future__ import annotations

from typing import Any

from alz_agent.agent.intent_schema import StructuredIntent
from alz_agent.agent.plan_execution import _infer_positive_values
from alz_agent.backend.interface import DataBackend
from alz_agent.schemas import CatalogEntry, FilterSpec, ToolResult
from alz_agent.tools.deterministic import lookup_variable, run_adjusted_regression


def _run_statistical_analysis(
    intent: StructuredIntent,
    catalog: list[CatalogEntry],
    backend: DataBackend,
    filter_spec: FilterSpec,
    *,
    analysis_engine: Any,
    nl_question: str,
    client: Any,
    model: str | None,
) -> list[ToolResult]:
    """Dispatch a STATISTICAL_ANALYSIS intent to the analysis engine."""
    if analysis_engine is None:
        return [ToolResult.abstain(
            "Statistical analysis requires a LocalAnalysisEngine "
            "(not available with the current backend configuration)"
        )]

    analysis_type = intent.analysis_type or "group_comparison"

    if analysis_type == "group_comparison":
        return _run_group_comparison(intent, catalog, filter_spec, analysis_engine)

    if analysis_type in ("regression", "linear_regression", "logistic_regression"):
        return _run_regression(intent, catalog, filter_spec, analysis_engine, analysis_type)

    # Novel analysis — delegate to LLM code generation
    if client is None:
        return [ToolResult.abstain(
            f"Analysis type '{analysis_type}' has no pre-built template. "
            "An LLM client is required for code generation."
        )]
    return _run_generated_analysis(intent, catalog, filter_spec, analysis_engine, nl_question, client, model)


def _run_group_comparison(
    intent: StructuredIntent,
    catalog: list[CatalogEntry],
    filter_spec: FilterSpec,
    analysis_engine: Any,
) -> list[ToolResult]:
    """Pre-built group comparison template dispatch.

    Variable resolution priority:
      1. intent.group_column / intent.outcome_column  — validated raw_code from
         LLM-mode catalog-grounded parsing (exact raw_code lookup, always succeeds).
      2. intent.group_variable / intent.outcome_variable — NL string from heuristic
         mode (keyword lookup, may fail on sparse catalogs).
    """
    # Prefer LLM-resolved raw columns; fall back to NL query strings.
    group_query = intent.group_column or intent.group_variable
    outcome_query = intent.outcome_column or intent.outcome_variable

    if not group_query:
        return [ToolResult.abstain(
            "group_comparison requires a group variable (group_column or group_variable)"
        )]
    if not outcome_query:
        return [ToolResult.abstain(
            "group_comparison requires an outcome variable (outcome_column or outcome_variable)"
        )]

    group_lookup = lookup_variable(group_query, catalog)
    if not group_lookup.grounded:
        return [ToolResult.abstain(
            f"Group variable '{group_query}' not found in catalog: "
            f"{group_lookup.abstain_reason}"
        )]

    outcome_lookup = lookup_variable(outcome_query, catalog)
    if not outcome_lookup.grounded:
        return [ToolResult.abstain(
            f"Outcome variable '{outcome_query}' not found in catalog: "
            f"{outcome_lookup.abstain_reason}"
        )]

    group_entry = group_lookup.value
    outcome_entry = outcome_lookup.value

    # Infer group names and positive_values from value_coding
    coding = group_entry.value_coding  # e.g. {"0": "Absent", "1": "Recent/Active"}
    neg_label = coding.get("0", f"no {group_entry.concept}")
    pos_values = _infer_positive_values(coding)
    # Label: use the first valid positive code's label, or the concept name
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
        positive_values=pos_values,  # None = any nonzero; list = explicit valid codes
    )
    return [result]


def _run_regression(
    intent: StructuredIntent,
    catalog: list[CatalogEntry],
    filter_spec: FilterSpec,
    analysis_engine: Any,
    analysis_type: str,
) -> list[ToolResult]:
    """Pre-built adjusted regression template dispatch.

    Prefers LLM-resolved raw columns (outcome_column, covariate_columns) over
    NL fallback strings.
    """
    outcome_query = intent.outcome_column or intent.outcome_variable
    if not outcome_query:
        return [ToolResult.abstain(
            "regression analysis requires an outcome variable"
        )]

    outcome_lookup = lookup_variable(outcome_query, catalog)
    if not outcome_lookup.grounded:
        return [ToolResult.abstain(
            f"Outcome variable '{outcome_query}' not found in catalog: "
            f"{outcome_lookup.abstain_reason}"
        )]
    outcome_entry = outcome_lookup.value

    # Covariates: prefer covariate_columns (LLM-resolved), else use group_column/group_variable
    covariate_var_ids: list[str] = []
    covariate_cols: list[str] = []
    predictor_queries: list[str] = list(intent.covariate_columns) or (
        [intent.group_column] if intent.group_column
        else ([intent.group_variable] if intent.group_variable else [])
    )
    for q in predictor_queries:
        if not q:
            continue
        lk = lookup_variable(q, catalog)
        if lk.grounded and lk.value:
            covariate_var_ids.append(lk.value.variable_id)
            covariate_cols.append(lk.value.raw_code)

    if not covariate_cols:
        return [ToolResult.abstain(
            "regression analysis requires at least one predictor variable "
            "(provide group_variable or a predictor list)"
        )]

    reg_type = "logistic" if "logistic" in analysis_type else "linear"

    result = run_adjusted_regression(
        outcome_variable_id=outcome_entry.variable_id,
        outcome_col=outcome_entry.raw_code,
        outcome_label=outcome_entry.concept,
        covariate_variable_ids=covariate_var_ids,
        covariate_cols=covariate_cols,
        filter_spec=filter_spec,
        analysis_engine=analysis_engine,
        regression_type=reg_type,
    )
    return [result]


def _run_generated_analysis(
    intent: StructuredIntent,
    catalog: list[CatalogEntry],
    filter_spec: FilterSpec,
    analysis_engine: Any,
    nl_question: str,
    client: Any,
    model: str | None,
) -> list[ToolResult]:
    """LLM code generation path for novel analyses."""
    from alz_agent.analysis.engine import generate_analysis_code

    # Build column schema from relevant catalog entries (limit to id_space A for now)
    relevant = [
        e for e in catalog
        if e.id_space == filter_spec.id_space
        and e.source_dataset.value == "NACC"
    ][:40]  # cap at 40 to stay within LLM context

    col_schema_lines = [
        f"{e.variable_id} | {e.raw_code} | {e.concept} | {dict(list(e.value_coding.items())[:4])}"
        for e in relevant
    ]
    col_schema = "\n".join(col_schema_lines)

    intent_summary = (
        f"question_type: {intent.question_type.value}\n"
        f"analysis_type: {intent.analysis_type}\n"
        f"population: {intent.target_population}\n"
        f"etiology: {intent.target_etiology}\n"
        f"group_variable: {intent.group_variable}\n"
        f"outcome_variable: {intent.outcome_variable}\n"
    )

    try:
        code, required_cols, description = generate_analysis_code(
            nl_question, intent_summary, col_schema, client, model
        )
    except (RuntimeError, ValueError) as exc:
        return [ToolResult.abstain(f"Analysis code generation failed: {exc}")]

    result = analysis_engine.run_generated_code(
        code=code,
        filter_spec=filter_spec,
        required_cols=required_cols,
        analysis_description=description,
    )
    return [result]
