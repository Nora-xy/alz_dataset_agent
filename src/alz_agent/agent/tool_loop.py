"""P5 tool loop — select and execute deterministic tools based on parsed intent.

Invariant: NO LLM calls inside the loop body.  LLM calls are only used before
(for modeling_design planning) and after (for narration) this module.

Public functions:
  _run_tool_loop(intent, catalog, backend, corpus_dir, *, ...) -> tuple
  _build_filter_spec(intent) -> FilterSpec
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alz_agent.agent.intent_schema import QuestionType, StructuredIntent
from alz_agent.agent.modeling_evidence_planner import (
    _call_modeling_design_planner,
    _execute_evidence_plan,
)
from alz_agent.agent.plan_execution import (
    _PLAN_SUPPORTED_TYPES,
    _build_filter_spec_from_plan,
    _run_plan,
)
from alz_agent.agent.statistical_dispatch import _run_statistical_analysis
from alz_agent.backend.interface import DataBackend
from alz_agent.retrieval.doc_retriever import retrieve_doc_evidence
from alz_agent.schemas import (
    CatalogEntry,
    DocEvidence,
    FilterSpec,
    ModelingDesignPlan,
    ToolResult,
)
from alz_agent.tools.deterministic import (
    assess_modality_overlap,
    compute_cohort_size,
    compute_variable_stat,
    lookup_variable,
    suggest_modeling_design,
)

# When intent.horizon_years is None (non-longitudinal question) but the tool loop
# still needs a horizon (FEASIBILITY / MODELING_DESIGN paths), fall back to this.
_DEFAULT_HORIZON = 3


def _build_filter_spec(intent: StructuredIntent) -> FilterSpec:
    return FilterSpec(
        baseline_diagnosis=intent.target_population if intent.target_population else None,
        baseline_etiology=intent.target_etiology if intent.target_etiology else None,
        id_space=intent.id_space,
    )


def _run_tool_loop(
    intent: StructuredIntent,
    catalog: list[CatalogEntry],
    backend: DataBackend,
    corpus_dir: Path | str | None,
    *,
    analysis_engine: Any = None,
    nl_question: str = "",
    client: Any = None,
    model: str | None = None,
) -> tuple[list[ToolResult], list[DocEvidence], FilterSpec, ModelingDesignPlan | None]:
    """Select and execute deterministic tools based on intent.question_type.

    Returns (tool_results, doc_evidence, effective_filter_spec, modeling_design_plan).
    The effective_filter_spec is plan-derived when an AnalysisPlan was used,
    otherwise it is derived from the StructuredIntent directly.
    The modeling_design_plan is non-None only for MODELING_DESIGN questions when
    an LLM client is available and evidence planning succeeds.
    """
    tool_results: list[ToolResult] = []
    filter_spec = _build_filter_spec(intent)
    effective_filter_spec: FilterSpec = filter_spec  # updated below if plan-based
    effective_modeling_plan: ModelingDesignPlan | None = None

    # Effective horizon for tools that require a positive integer.
    # Use the explicit value when present; fall back to the default for longitudinal
    # question types where no horizon was stated.
    effective_horizon = (
        intent.horizon_years if intent.horizon_years is not None else _DEFAULT_HORIZON
    )

    # Plan-based dispatch: when an AnalysisPlan is attached (LLM planner mode),
    # use it for supported analysis types.  This avoids sending compound phrases
    # like 'age at first AD diagnosis' to lookup_variable — the plan already
    # decomposes them into value_variable + event + visit_anchor.
    if (
        intent.analysis_plan is not None
        and intent.analysis_plan.analysis_type in _PLAN_SUPPORTED_TYPES
    ):
        plan_filter_spec = _build_filter_spec_from_plan(intent.analysis_plan, intent.id_space)
        effective_filter_spec = plan_filter_spec
        tool_results.extend(
            _run_plan(
                intent.analysis_plan, catalog, backend, plan_filter_spec,
                analysis_engine=analysis_engine,
            )
        )

    elif intent.question_type == QuestionType.STATISTICAL_ANALYSIS:
        tool_results.extend(
            _run_statistical_analysis(
                intent, catalog, backend, filter_spec,
                analysis_engine=analysis_engine,
                nl_question=nl_question,
                client=client,
                model=model,
            )
        )

    elif intent.question_type == QuestionType.AGGREGATE_STAT:
        stat_var = intent.stat_variable
        if not stat_var:
            tool_results.append(
                ToolResult.abstain("stat_variable not specified for AGGREGATE_STAT question")
            )
        else:
            # Resolve the stat_variable concept to a catalog entry to get the variable_id
            lookup = lookup_variable(stat_var, catalog)
            if not lookup.grounded:
                tool_results.append(lookup)
            else:
                var_id = lookup.value.variable_id
                tool_results.append(
                    compute_variable_stat(
                        var_id,
                        intent.stat_type or "mean",
                        filter_spec,
                        backend,
                        event_anchor=intent.event_anchor,
                    )
                )

    elif intent.question_type == QuestionType.LOOKUP:
        # Variable / concept lookup for each modality keyword (or the candidate label)
        targets = intent.modalities if intent.modalities else [intent.candidate_label]
        for concept in targets:
            tool_results.append(lookup_variable(concept, catalog))

    elif intent.question_type == QuestionType.COHORT_SIZE:
        tool_results.append(compute_cohort_size(filter_spec, backend))
        # When modalities are specified alongside a size question (e.g. "how many
        # participants have both cognition and imaging?"), also compute the overlap.
        if intent.modalities:
            overlap = assess_modality_overlap(
                intent.modalities, filter_spec, backend, catalog
            )
            tool_results.append(overlap)

    elif intent.question_type in (QuestionType.FEASIBILITY, QuestionType.MODELING_DESIGN):
        if client is not None and intent.question_type == QuestionType.MODELING_DESIGN:
            # LLM evidence planning path: ask LLM what to compute, then compute it
            md_plan = _call_modeling_design_planner(
                nl_question, intent, catalog, client, model
            )
            if md_plan is not None:
                effective_modeling_plan = md_plan
                tool_results.extend(
                    _execute_evidence_plan(md_plan, intent, catalog, backend)
                )
            else:
                # LLM failed to produce plan — fall back to heuristic
                design_result = suggest_modeling_design(
                    filter_spec, effective_horizon, catalog, backend
                )
                tool_results.append(design_result)
                if intent.modalities and design_result.grounded:
                    tool_results.append(
                        assess_modality_overlap(intent.modalities, filter_spec, backend, catalog)
                    )
        else:
            # Heuristic path: FEASIBILITY or MODELING_DESIGN without LLM client
            design_result = suggest_modeling_design(
                filter_spec, effective_horizon, catalog, backend
            )
            tool_results.append(design_result)
            if intent.modalities and design_result.grounded:
                tool_results.append(
                    assess_modality_overlap(intent.modalities, filter_spec, backend, catalog)
                )

    else:
        # Unknown / unhandled type — default to comprehensive design
        tool_results.append(
            suggest_modeling_design(filter_spec, effective_horizon, catalog, backend)
        )

    # P4: documentation evidence retrieval (gracefully returns [] when unavailable)
    # For modeling_design plans, also retrieve docs for each documentation_query.
    if effective_modeling_plan is not None and effective_modeling_plan.documentation_queries:
        doc_queries = effective_modeling_plan.documentation_queries
    else:
        horizon_str = (
            f" {intent.horizon_years} year" if intent.horizon_years is not None else ""
        )
        doc_queries = [
            (
                f"{' '.join(intent.target_population)} "
                f"{intent.candidate_label}"
                f"{horizon_str}"
            ).strip()
        ]

    all_doc_ev: list[DocEvidence] = []
    seen_locators: set[str] = set()
    for q in doc_queries[:4]:  # cap at 4 queries to limit context
        for ev in retrieve_doc_evidence(query=q, k=3, corpus_dir=corpus_dir):
            key = f"{ev.source}:{ev.locator}"
            if key not in seen_locators:
                seen_locators.add(key)
                all_doc_ev.append(ev)
    doc_ev = all_doc_ev

    return tool_results, doc_ev, effective_filter_spec, effective_modeling_plan
