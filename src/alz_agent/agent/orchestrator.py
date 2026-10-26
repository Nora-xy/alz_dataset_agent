"""P5 — Agent orchestrator: tool loop + narration + hallucination guard.

Public surface (thin facade — implementation lives in submodules):
  AgentResponse   — the final output of the agent pipeline
  run_agent(...)  — the full pipeline entry point

Execution flow:
  run_agent(nl_question, catalog, backend)
    │
    ├─ 1. parse_intent()                         [intent.py]
    │      └─ LLM or heuristic → StructuredIntent
    │
    ├─ 2. _run_tool_loop()                        [tool_loop.py]
    │      ├─ plan-based: _run_plan()             [plan_execution.py]
    │      ├─ statistical: _run_statistical_analysis() [statistical_dispatch.py]
    │      ├─ modeling design: _call_modeling_design_planner() + _execute_evidence_plan()
    │      │                                       [modeling_evidence_planner.py]
    │      └─ doc retrieval (P4)
    │
    ├─ 3. build_explanation_context()             [explanation.py]
    │
    ├─ 4. narrate (LLM or template)
    │      ├─ narrate_with_llm()                  [narration.py]
    │      └─ _narrate_template()                 [template_narration.py]
    │
    ├─ 5. _check_hallucination()                  [hallucination_guard.py]
    │      └─ on failure: fall back to template or abstain
    │
    └─ 6. _build_analysis_trace()                 [trace_builder.py]
           → AgentResponse

Invariant: no number in the final answer may originate from model generation;
it must trace to a tool result evidence entry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

_log = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from alz_agent.agent.explanation import build_explanation_context
from alz_agent.agent.hf_local_synthesis import (
    hf_local_extra_evidence_digits,
    synthesize_hf_local_answer,
)
from alz_agent.agent.intent import QuestionType, StructuredIntent, parse_intent
from alz_agent.agent.narration import build_narration_facts, narrate_with_llm
from alz_agent.agent.providers import LocalHFPlannerClient, PlannerClient
from alz_agent.backend.interface import DataBackend
from alz_agent.schemas import (
    AnalysisTrace,
    CatalogEntry,
    DataEvidence,
    DescriptiveStatResult,
    DocEvidence,
    EtiologyDefinition,
    ExplanationContext,
    ExplanationMetadata,
    FilterSpec,
    GeneratedAnalysisResult,
    GroupComparisonResult,
    ModelingDesignPlan,
    NarrationResult,
    RegressionResult,
    SemanticAnalysisPlan,
    ToolResult,
)
from alz_agent.tools.deterministic import (
    ModelingDesignResult,
    StatResult,
)

# ---------------------------------------------------------------------------
# Import submodule implementations
# ---------------------------------------------------------------------------

from alz_agent.agent.hallucination_guard import (  # noqa: F401
    _check_external_claims,
    _check_hallucination,
    _check_synthesis_hallucination,
    _check_synthesis_variable_ids,
    _collect_evidence_numbers,
)
from alz_agent.agent.template_narration import (  # noqa: F401
    _narrate_template,
    _narrate_template_from_ctx,
    _narrate_template_legacy,
    _strip_integers_from_definition,
)
from alz_agent.agent.trace_builder import (  # noqa: F401
    _build_analysis_trace,
    _build_caveats,
    _build_output_summary,
    _build_tool_call_traces,
    _build_variable_resolutions,
    _derive_plan_steps,
    _infer_input_summary,
    _infer_tool_name,
)
from alz_agent.agent.modeling_evidence_planner import (  # noqa: F401
    _MODELING_DESIGN_PLAN_TOOL,
    _call_modeling_design_planner,
    _execute_evidence_plan,
)
from alz_agent.agent.plan_execution import (  # noqa: F401
    _NACCUDSD_CODE_TO_LABEL,
    _PLAN_SUPPORTED_TYPES,
    _VISIT_ANCHOR_TO_BACKEND,
    _build_filter_spec_from_plan,
    _infer_positive_values,
    _plan_aggregate_stat,
    _plan_descriptive_stat,
    _plan_group_comparison,
    _plan_regression,
    _run_plan,
)
from alz_agent.agent.statistical_dispatch import (  # noqa: F401
    _run_generated_analysis,
    _run_group_comparison,
    _run_regression,
    _run_statistical_analysis,
)
from alz_agent.agent.tool_loop import (  # noqa: F401
    _DEFAULT_HORIZON,
    _build_filter_spec,
    _run_tool_loop,
)


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """The final output of the agent pipeline for a single question."""

    grounded: bool
    answer: str
    tool_results: list[ToolResult]
    doc_evidence: list[DocEvidence] = Field(default_factory=list)
    hallucination_check_passed: bool
    abstain_reason: str | None = None
    trace: AnalysisTrace | None = None  # structured audit trail (always populated)
    effective_horizon: int = 3  # horizon (years) used to produce this answer

    # Narration metadata
    final_narration_mode: Literal["llm", "deterministic_fallback"] = "deterministic_fallback"
    final_narration_model: str | None = None
    final_narration_error: str | None = None
    narration_validation_passed: bool = True
    explanation_coverage: float = 1.0
    analysis_section_present: bool = True
    scope_validation_passed: bool = True

    # Enriched planner path (None when use_enriched_planner=False)
    synthesis_report: Any = None   # SynthesisReport | None
    enriched_plan_used: bool = False

    # Debug fields — populated on both success and failure of the enriched path
    enriched_plan_error: str | None = None
    enriched_plan_operations: list[str] = Field(default_factory=list)
    requested_horizons: list[int] = Field(default_factory=list)
    validation_hard_errors: list[str] = Field(default_factory=list)
    validation_soft_warnings: list[str] = Field(default_factory=list)
    execution_operations_run: list[str] = Field(default_factory=list)
    synthesis_report_present: bool = False

    # Semantic success metadata (semantic-planner path only; True by default elsewhere —
    # see run_agent()/`_run_enriched_path` docstrings). `grounded=True` alone means "some
    # tool returned a real number" — it does NOT mean the number answers what was asked.
    # These three flags are computed deterministically from the SemanticAnalysisPlan +
    # ExecutionContext, never asserted by narration text:
    #   requested_measure_resolved — every variable_id the plan selected passed the
    #     validator's catalog-existence check (C10) and, when declared, the
    #     specimen/analyte consistency check (C11).
    #   question_answered — every operation the compiled plan emitted produced a
    #     grounded ToolResult (no op silently abstained/errored).
    #   semantic_grounding_passed — both of the above AND grounded=True. A deterministic
    #     but semantically-irrelevant result (e.g. the wrong variable) fails this even
    #     though grounded=True, because requested_measure_resolved/question_answered catch it.
    question_answered: bool = True
    requested_measure_resolved: bool = True
    semantic_grounding_passed: bool = True

    # Planner-path proof (CLI reachability regression — see scripts/ask_agent.py
    # --use-semantic-planner). Computed once, at the single point in run_agent()
    # that dispatches to _run_enriched_path(), from the actual use_enriched_planner/
    # use_semantic_planner values that were passed in — not re-derived downstream,
    # so no intermediate layer can silently reset it back to a different value.
    #   "non_enriched"     — use_enriched_planner=False (or no client): classic
    #                        parse_intent/StructuredIntent path.
    #   "classic_enriched"  — use_enriched_planner=True, use_semantic_planner=False:
    #                        plan_enriched_analysis() path.
    #   "semantic_enriched" — use_enriched_planner=True, use_semantic_planner=True:
    #                        plan_semantic_analysis() -> validate_semantic() ->
    #                        compile_to_enriched() path.
    planner_path: Literal["non_enriched", "classic_enriched", "semantic_enriched"] = "non_enriched"
    use_semantic_planner_requested: bool = False

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_agent(
    nl_question: str,
    catalog: list[CatalogEntry],
    backend: DataBackend,
    *,
    corpus_dir: Path | str | None = None,
    intent: StructuredIntent | None = None,
    client: Any = None,
    model: str | None = None,
    analysis_engine: Any = None,  # LocalAnalysisEngine | None; avoids circular import
    use_enriched_planner: bool = False,
    use_semantic_planner: bool = False,
) -> AgentResponse:
    """Run the full agent pipeline on a natural-language research question.

    Args:
        nl_question:          User's research question in plain English.
        catalog:              Parsed CatalogEntry list (P1 output).
        backend:              DataBackend implementation (P2).
        corpus_dir:           Optional path to documentation corpus for P4 retrieval.
        intent:               Pre-parsed StructuredIntent (bypasses parsing; for tests).
        client:               Optional Anthropic/OpenAI client for LLM steps.
                              When None, heuristic parsing and template narration are used.
        model:                Optional model name override for LLM steps.
        analysis_engine:      LocalAnalysisEngine | None — for group_comparison/regression.
        use_enriched_planner: When True and client is not None, uses the 4-layer enriched
                              planner path (plan_enriched_analysis → PlanValidator →
                              DeterministicExecutor → synthesize_research_report).
                              Falls through to the original path if enriched planning fails.
        use_semantic_planner: When True, ALWAYS implies use_enriched_planner=True (enforced
                              below, regardless of the use_enriched_planner argument's literal
                              value) — this guarantees a caller can never silently land in the
                              classic parse_intent path while believing the semantic gate is
                              active. When True and client is not None, prepends a semantic
                              gate: LLM generates SemanticAnalysisPlan,
                              PlanValidator.validate_semantic() checks invariants, one repair
                              pass if invalid, then abstain if still invalid.  The validated
                              plan is compiled deterministically to EnrichedAnalysisPlan.

    Returns:
        AgentResponse — grounded when at least one tool returned a value.
    """
    # -----------------------------------------------------------------------
    # Enriched planner path (opt-in; when True, no silent fallback to classic)
    # -----------------------------------------------------------------------
    # use_semantic_planner=True always implies enriched mode — enforced here, not just
    # at the CLI layer, so no caller (library or CLI) can ever end up silently in the
    # classic parse_intent path while believing the semantic gate was requested.
    use_enriched_planner = use_enriched_planner or use_semantic_planner

    # Note: intent is intentionally ignored when use_enriched_planner=True.
    # The enriched path calls plan_enriched_analysis() from the raw question;
    # any pre-parsed StructuredIntent is not the right first step here.
    if use_enriched_planner and client is not None:
        _planner_path: Literal["classic_enriched", "semantic_enriched"] = (
            "semantic_enriched" if use_semantic_planner else "classic_enriched"
        )
        enriched_response, enriched_error = _run_enriched_path(
            nl_question, catalog, backend,
            client=client, model=model, analysis_engine=analysis_engine,
            use_semantic_planner=use_semantic_planner,
        )
        if enriched_response is not None:
            return enriched_response.model_copy(update={
                "planner_path": _planner_path,
                "use_semantic_planner_requested": use_semantic_planner,
            })
        # Explicit error — no silent fallback when use_enriched_planner=True
        return AgentResponse(
            grounded=False,
            answer=(
                "Enriched planner failed to produce a plan.\n"
                f"Error: {enriched_error or 'unknown'}\n"
                "To use the classic parsing path, omit --use-enriched-planner."
            ),
            tool_results=[],
            doc_evidence=[],
            hallucination_check_passed=True,
            abstain_reason=enriched_error or "enriched_planner_failed",
            effective_horizon=3,
            enriched_plan_used=False,
            enriched_plan_error=enriched_error,
            final_narration_mode="deterministic_fallback",
            final_narration_error=enriched_error or "enriched_planner_failed",
            planner_path=_planner_path,
            use_semantic_planner_requested=use_semantic_planner,
        )

    # Step 1: parse intent — pass catalog so LLM mode can do grounded resolution
    parsed_intent = (
        intent if intent is not None
        else parse_intent(nl_question, client, model=model, catalog=catalog)
    )

    # Step 2: run tool loop (P3 tools + P4 retriever + analysis engine)
    tool_results, doc_ev, effective_filter_spec, modeling_plan = _run_tool_loop(
        parsed_intent, catalog, backend, corpus_dir,
        analysis_engine=analysis_engine, nl_question=nl_question, client=client, model=model,
    )

    # Determine effective horizon and its source for explanation context
    effective_horizon = (
        parsed_intent.horizon_years if parsed_intent.horizon_years is not None
        else _DEFAULT_HORIZON
    )
    horizon_source = "user" if parsed_intent.horizon_years is not None else "default"

    # Build explanation context (deterministic — no LLM)
    ctx = build_explanation_context(
        tool_results,
        filter_spec=effective_filter_spec,
        horizon=effective_horizon,
        horizon_source=horizon_source,
        default_horizon=_DEFAULT_HORIZON,
        catalog=catalog,
        target_etiology=list(parsed_intent.target_etiology) if parsed_intent.target_etiology else None,
        backend_name=type(backend).__name__,
    )

    # Step 3: check for full abstention
    all_abstained = bool(tool_results) and all(not tr.grounded for tr in tool_results)
    if all_abstained:
        reasons = [tr.abstain_reason or "unknown reason" for tr in tool_results]
        combined = "; ".join(dict.fromkeys(reasons))  # deduplicate while preserving order
        abstain_answer = (
            "This question is not currently answerable with available data.\n"
            f"Reason(s): {combined}\n"
            "Consider: verifying the filter parameters, checking whether the "
            "requested modality or release is present, or whether the cohort "
            "is above the suppression threshold."
        )
        trace = _build_analysis_trace(
            nl_question, parsed_intent, effective_filter_spec, tool_results,
            modeling_design_plan=modeling_plan,
        )
        return AgentResponse(
            grounded=False,
            answer=abstain_answer,
            tool_results=tool_results,
            doc_evidence=doc_ev,
            hallucination_check_passed=_check_hallucination(
                abstain_answer, tool_results, extra={str(effective_horizon)}
            ),
            abstain_reason=combined,
            effective_horizon=effective_horizon,
            trace=trace,
            final_narration_mode="deterministic_fallback",
            final_narration_error="all_tools_abstained",
        )

    # Step 4: build narration facts (deterministic) → narrate (LLM or template)
    narration_facts = build_narration_facts(
        ctx,
        filter_spec=effective_filter_spec,
        horizon=effective_horizon,
        horizon_source=horizon_source,
    )

    narration_result: NarrationResult
    if isinstance(client, LocalHFPlannerClient):
        # hf-local: skip free-form LLM narration entirely. A small local
        # model narrating structured tool results is prone to (a) verbose
        # prose instead of a short grounded number, and (b) picking the
        # wrong number for a modality-overlap union (e.g. the unrestricted
        # total cohort instead of the requested intersection/union). Use a
        # deterministic template instead — see hf_local_synthesis.py.
        # Safe/backward-compatible: this branch only ever triggers for
        # LocalHFPlannerClient instances; every other client (Anthropic,
        # DeepSeek-via-Anthropic-SDK) still takes the narrate_with_llm branch
        # below, unchanged.
        hf_answer = synthesize_hf_local_answer(
            parsed_intent, tool_results, backend_name=type(backend).__name__,
        )
        narration_result = NarrationResult(
            answer=hf_answer,
            mode="deterministic_fallback",
            error=None,
            facts_used=len(narration_facts),
        )
    elif client is not None:
        narration_result = narrate_with_llm(
            nl_question, parsed_intent, narration_facts, ctx, doc_ev,
            client, model=model, modeling_plan=modeling_plan,
        )
    else:
        template_answer = _narrate_template(parsed_intent, tool_results, doc_ev, ctx=ctx)
        narration_result = NarrationResult(
            answer=template_answer,
            mode="deterministic_fallback",
            error="no_llm_client_configured",
            facts_used=len(narration_facts),
        )

    answer = narration_result.answer
    # If LLM returned empty string (rare), fall back to template
    if not answer.strip():
        template_answer = _narrate_template(parsed_intent, tool_results, doc_ev, ctx=ctx)
        narration_result = NarrationResult(
            answer=template_answer,
            mode="deterministic_fallback",
            error=narration_result.error or "llm_returned_empty_response",
            facts_used=len(narration_facts),
        )
        answer = template_answer

    # Step 5: hallucination guard
    _extra_evidence = {str(effective_horizon)}
    if isinstance(client, LocalHFPlannerClient):
        # Whitelist hf-local's deterministic inclusion-exclusion derivations
        # (e.g. a "cognition AND (MRI OR PET)" union) — these are correct
        # computations over tool evidence, just not literally present as a
        # single tool value. Empty set (no-op) whenever no such derivation
        # applies. Does not affect any other client's hallucination check.
        _extra_evidence |= hf_local_extra_evidence_digits(parsed_intent, tool_results)
    passed = _check_hallucination(answer, tool_results, extra=_extra_evidence)
    if not passed:
        answer = _narrate_template(parsed_intent, tool_results, doc_ev, ctx=ctx)
        narration_result = NarrationResult(
            answer=answer,
            mode="deterministic_fallback",
            error=f"hallucination_guard_failed_on_{narration_result.mode}_answer",
            facts_used=len(narration_facts),
        )
        passed = _check_hallucination(answer, tool_results, extra=_extra_evidence)
        if not passed:
            trace = _build_analysis_trace(
                nl_question, parsed_intent, effective_filter_spec, tool_results,
                modeling_design_plan=modeling_plan,
            )
            return AgentResponse(
                grounded=False,
                answer=(
                    "Narration aborted: hallucination guard detected numbers not "
                    "traceable to tool results. Raw tool values are available in "
                    "tool_results."
                ),
                tool_results=tool_results,
                doc_evidence=doc_ev,
                hallucination_check_passed=False,
                abstain_reason="hallucination_guard_failed",
                trace=trace,
                effective_horizon=effective_horizon,
                final_narration_mode="deterministic_fallback",
                final_narration_error="hallucination_guard_failed",
                narration_validation_passed=False,
            )

    trace = _build_analysis_trace(
        nl_question, parsed_intent, effective_filter_spec, tool_results,
        modeling_design_plan=modeling_plan,
    )
    _val = narration_result.validation
    return AgentResponse(
        grounded=True,
        answer=answer,
        tool_results=tool_results,
        doc_evidence=doc_ev,
        hallucination_check_passed=passed,
        trace=trace,
        effective_horizon=effective_horizon,
        final_narration_mode=narration_result.mode,
        final_narration_model=narration_result.model,
        final_narration_error=narration_result.error,
        narration_validation_passed=_val.passed if _val is not None else True,
        explanation_coverage=_val.explanation_coverage if _val is not None else 1.0,
        analysis_section_present=_val.analysis_section_present if _val is not None else True,
        scope_validation_passed=_val.scope_validation_passed if _val is not None else True,
    )


# ---------------------------------------------------------------------------
# Narration helpers (kept here — narration.py handles the LLM side;
# the functions below are the modeling-design-specific LLM narration path
# that was already in orchestrator.py and is not part of the extracted modules)
# ---------------------------------------------------------------------------


def _narrate_modeling_design_with_llm(
    plan: ModelingDesignPlan,
    intent: StructuredIntent,
    tool_results: list[ToolResult],
    doc_evidence: list[DocEvidence],
    client: Any,
    model: str | None = None,
    ctx: ExplanationContext | None = None,
) -> str:
    """Synthesise a grounded modeling recommendation from the structured ExplanationContext.

    The LLM receives structured ExplainableItems (with definitions and computation notes)
    and writes explanatory text.  It may use general ML knowledge for qualitative guidance
    but must use ONLY numbers that appear verbatim in item.value fields.
    """
    # Build structured context block from ExplanationContext (preferred) or fall back
    # to the old flat serialization if ctx is unavailable.
    if ctx is not None:
        context = _format_ctx_for_llm(ctx, plan, doc_evidence)
    else:
        # Legacy path (ctx not yet built — should not happen in normal flow)
        evidence_lines: list[str] = []
        for i, tr in enumerate(tool_results, 1):
            if tr.grounded:
                evidence_lines.append(f"  [{i}] grounded=True  value={_serialize_value(tr.value)}")
                for ev in tr.evidence:
                    if isinstance(ev, DataEvidence):
                        evidence_lines.append(f"       evidence: {ev.aggregate_ref}")
            else:
                evidence_lines.append(f"  [{i}] grounded=False — {tr.abstain_reason}")
        context = "\n".join(evidence_lines) + "\n\nGenerate a grounded modeling recommendation."

    system_prompt = _build_narration_system_prompt(plan)

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]
        if isinstance(client, PlannerClient):
            return client.complete_text(messages, max_tokens=2500)
        response = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=2500,
            system=system_prompt,
            messages=[{"role": "user", "content": context}],
        )
        text_block = next((b for b in response.content if b.type == "text"), None)
        if text_block is None:
            raise RuntimeError("LLM returned no text block.")
        return text_block.text
    except Exception as exc:
        import sys
        print(f"[modeling_design_narration] LLM narration failed: {exc}", file=sys.stderr)
        return _narrate_template(intent, tool_results, doc_evidence, ctx=ctx)


def _build_narration_system_prompt(plan: ModelingDesignPlan | None = None) -> str:
    """Build the system prompt for final-answer LLM narration."""
    sections_instruction = (
        "## Parsed question / modeling target\n"
        "## Cohort definition\n"
        "## Outcome and follow-up feasibility\n"
        "## Modality availability\n"
        "## Feature eligibility and leakage policy\n"
        "## Recommended modeling design\n"
        "## Caveats and limitations\n"
        "## Metadata (audit)\n"
    )
    plan_context = ""
    if plan is not None:
        plan_context = (
            f"\nEvidence plan summary:\n"
            f"  Goal: {plan.modeling_goal}\n"
            f"  Target: {plan.target_population}\n"
            f"  Label options: {'; '.join(plan.label_definition_options) or 'not specified'}\n"
            f"  Candidate modalities: {', '.join(plan.candidate_modalities) or 'not specified'}\n"
        )

    return (
        "You are a research design advisor for an Alzheimer's disease machine learning study.\n"
        "You are writing a modeling-design report for a researcher.\n"
        f"{plan_context}\n"
        "PERMISSION AND RESPONSIBILITY:\n"
        "You are allowed and expected to explain the meaning of structured tool outputs. "
        "Explain important counts, statuses, modality combinations, abstentions, "
        "feature-policy results, and caveats. Use the 'definition' and 'computation_note' "
        "fields provided for each item to guide your explanation.\n\n"
        "STRICT RULES:\n"
        "1. Use ONLY integers and counts that appear verbatim in item.value fields below.\n"
        "   Do NOT invent, estimate, extrapolate, or copy digit sequences from definition/note text.\n"
        "2. For every item marked explanation_required=True, write at least one sentence "
        "   explaining what that number or value means in context.\n"
        "3. For NOT ANSWERABLE items: explain what was requested, why the tool abstained, "
        "   and what the user should do next.\n"
        "4. Leakage blacklist items: name each blacklisted variable explicitly with its "
        "   raw_code, leakage tag, and reason.\n"
        "5. If a tool result is grounded=False (not available), state that fact is missing "
        "   and do not invent a substitute number.\n"
        "6. If an explanation cannot be grounded in the supplied context, say "
        "   'definition unavailable from current metadata' — do not hallucinate.\n"
        "7. Horizon: if the horizon is an 'example default', explicitly state this.\n"
        "8. Whitelist: explain it as candidate predictors (pre-index, post-leakage-exclusion, "
        "   not final features — still need missingness/encoding).\n"
        "9. AD-specificity: NACCALZD==1 = etiologic AD attribution (NOT confirmed neuropath). "
        "   NACCUDSD==4 = all-cause dementia (NOT AD-specific). Always note which is used.\n\n"
        "Structure your answer with exactly these sections:\n"
        f"{sections_instruction}\n"
        "Keep the Metadata section compact — just backend, tools invoked, and grounding status.\n"
        "Every integer you write must come verbatim from an item.value field."
    )


def _format_ctx_for_llm(
    ctx: ExplanationContext,
    plan: ModelingDesignPlan | None,
    doc_evidence: list[DocEvidence],
) -> str:
    """Render ExplanationContext as a structured prompt block for the narration LLM."""
    lines: list[str] = [
        "=== STRUCTURED TOOL RESULTS ===",
        f"Horizon: {ctx.metadata.horizon_label}",
        f"Horizon source: {ctx.metadata.horizon_source} "
        f"({'user specified' if ctx.metadata.horizon_source == 'user' else 'system default — user did not specify'})",
        "",
    ]

    # Group items by section
    item_by_id = {i.item_id: i for i in ctx.items}
    for section in ctx.sections:
        if section.section_id == "metadata":
            continue  # rendered at the bottom
        if not section.item_ids:
            continue
        lines.append(f"--- SECTION: {section.title} ---")
        for iid in section.item_ids:
            item = item_by_id.get(iid)
            if item is None:
                continue
            lines.append(f"  item_id: {item.item_id}")
            lines.append(f"  display_name: {item.display_name}")
            lines.append(f"  value: {item.value}")
            if item.unit:
                lines.append(f"  unit: {item.unit}")
            if item.denominator is not None:
                lines.append(f"  denominator: {item.denominator}")
            if item.filter_scope:
                lines.append(f"  filter_scope: {item.filter_scope}")
            if item.definition:
                lines.append(f"  definition: {item.definition}")
            if item.computation_note:
                lines.append(f"  computation_note: {item.computation_note}")
            if item.caveats:
                lines.append(f"  caveats: {'; '.join(item.caveats)}")
            if item.is_not_answerable:
                lines.append(f"  NOT_ANSWERABLE: True")
            lines.append(f"  explanation_required: {item.explanation_required}")
            lines.append(f"  importance: {item.importance}")
            lines.append("")

    # Structured caveats
    if ctx.caveats:
        lines.append("--- STRUCTURED CAVEATS (explain each; do not invent new numbers) ---")
        for i, cav in enumerate(ctx.caveats, 1):
            lines.append(f"  [{i}] parsed_target_etiology: {cav.parsed_target_etiology}")
            lines.append(f"       actual_endpoint: {cav.actual_endpoint}")
            lines.append(f"       missing_concept: {cav.missing_concept}")
            lines.append(f"       implication: {cav.implication}")
            lines.append(f"       required_next_step: {cav.required_next_step}")
            lines.append("")

    # Documentation evidence
    if doc_evidence:
        lines.append("--- DOCUMENTATION EVIDENCE (cite if relevant) ---")
        for ev in doc_evidence:
            lines.append(f"  [{ev.source}/{ev.locator}]: {ev.snippet[:200]}")
        lines.append("")

    lines.append("=== END OF STRUCTURED TOOL RESULTS ===")
    lines.append("")
    lines.append("Write a modeling-design report from the structured items above.")
    lines.append(
        "For every item with explanation_required=True, include at least one explanatory sentence."
    )
    lines.append(
        "Append a compact Metadata section listing backend, tools invoked, and explanation coverage."
    )
    return "\n".join(lines)


def _retry_narration_with_missing(
    original_answer: str,
    missing_display_names: list[str],
    ctx: ExplanationContext,
    doc_evidence: list[DocEvidence],
    client: Any,
    model: str | None = None,
) -> str:
    """Re-prompt the LLM to add explanations for items that were missed."""
    if not missing_display_names:
        return original_answer

    missing_block = "\n".join(f"  - {name}" for name in missing_display_names)
    retry_prompt = (
        f"The following required items were not explained in your previous answer:\n"
        f"{missing_block}\n\n"
        f"Original answer:\n{original_answer}\n\n"
        f"Please revise the answer to include explanations for the listed items. "
        f"Use only values from the structured tool results. "
        f"Do not invent new numbers."
    )
    system_prompt = _build_narration_system_prompt()
    try:
        messages = [{"role": "user", "content": retry_prompt}]
        if isinstance(client, PlannerClient):
            return client.complete_text(
                [{"role": "system", "content": system_prompt}] + messages,
                max_tokens=2500,
            )
        response = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=2500,
            system=system_prompt,
            messages=messages,
        )
        text_block = next((b for b in response.content if b.type == "text"), None)
        return text_block.text if text_block else original_answer
    except Exception:
        return original_answer


def _narrate_with_llm(
    intent: StructuredIntent,
    tool_results: list[ToolResult],
    doc_evidence: list[DocEvidence],
    client: Any,
    model: str | None = None,
    ctx: ExplanationContext | None = None,
) -> str:
    context = _format_results_for_llm(intent, tool_results, doc_evidence, ctx=ctx)

    # Use a richer researcher-facing prompt for statistical analysis questions
    if intent.question_type == QuestionType.STATISTICAL_ANALYSIS:
        system_prompt = (
            "You are a research analysis assistant summarizing statistical results for an "
            "Alzheimer's disease researcher. Rules:\n"
            "1. Use ONLY the numbers from the tool results below — never invent or extrapolate.\n"
            "2. Write a researcher-facing paragraph that covers:\n"
            "   a) The cohort and analysis design (who was included, what was compared).\n"
            "   b) Key numeric results: group sizes (n), means, SDs, and p-value verbatim.\n"
            "   c) A brief statistical interpretation (significant/not significant; direction).\n"
            "   d) Important caveats listed in the results (confounders, limitations).\n"
            "   e) A suggested next step (e.g. adjusted regression, subgroup analysis).\n"
            "3. Do NOT add clinical interpretations beyond what the data supports.\n"
            "4. Every number you write must appear verbatim in the tool results."
        )
    else:
        system_prompt = (
            "You are narrating Alzheimer's dataset query results for a researcher. "
            "Rules:\n"
            "1. Restate ONLY numbers and facts that appear in the tool results below.\n"
            "2. Do NOT invent, estimate, or extrapolate any numbers.\n"
            "3. If a tool returned an abstain result, state the question is not "
            "answerable and why.\n"
            "4. Keep the answer concise and evidence-backed.\n"
            "5. Every numeric token you write must come verbatim from a tool result."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
    ]
    if isinstance(client, PlannerClient):
        return client.complete_text(messages, max_tokens=1024)
    response = client.messages.create(
        model=model or "claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": context}],
    )
    # Thinking/reasoning models prepend a reasoning block before the text block;
    # find the text block explicitly rather than assuming content[0] is text.
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise RuntimeError("LLM returned no text content in narration response.")
    return text_block.text


def _format_results_for_llm(
    intent: StructuredIntent,
    tool_results: list[ToolResult],
    doc_evidence: list[DocEvidence],
    ctx: ExplanationContext | None = None,
) -> str:
    # When a structured context is available, use it for richer LLM input
    if ctx is not None:
        return _format_ctx_for_llm(ctx, None, doc_evidence)

    horizon_display = (
        f"{intent.horizon_years}yr" if intent.horizon_years is not None else "not specified"
    )
    lines = [
        f"Question type: {intent.question_type.value}",
        f"Target: {intent.target_population}, horizon: {horizon_display}, "
        f"label: {intent.candidate_label}",
        "",
        "Tool results (narrate these verbatim):",
    ]
    for i, tr in enumerate(tool_results, 1):
        if tr.grounded:
            lines.append(f"  [{i}] grounded=True")
            lines.append(f"       value = {_serialize_value(tr.value)}")
            for ev in tr.evidence:
                if isinstance(ev, DataEvidence):
                    lines.append(f"       evidence: {ev.aggregate_ref}")
        else:
            lines.append(f"  [{i}] grounded=False — {tr.abstain_reason}")

    if doc_evidence:
        lines.append("")
        lines.append("Documentation evidence (cite if relevant):")
        for ev in doc_evidence:
            lines.append(f"  [{ev.source}/{ev.locator}]: {ev.snippet[:200]}")

    lines.append("")
    lines.append("Generate a concise, evidence-backed narration for the researcher.")
    return "\n".join(lines)


def _serialize_value(val: Any) -> str:
    if isinstance(val, int):
        return str(val)
    if isinstance(val, dict):
        return str(val)
    if isinstance(val, StatResult):
        return (
            f"stat_type={val.stat_type}, value={val.value}, n={val.n}, "
            f"variable_id={val.variable_id}, event_anchor={val.event_anchor}"
        )
    if isinstance(val, GroupComparisonResult):
        parts = [
            f"outcome={val.outcome_label}, group={val.group_label}, "
            f"n_total={val.n_total}, p_value={val.p_value}"
        ]
        for g in val.groups:
            parts.append(f"{g.group_label}: n={g.n} mean={g.mean} sd={g.sd}")
        return "; ".join(parts)
    if isinstance(val, GeneratedAnalysisResult):
        return (
            f"analysis={val.analysis_description}, n={val.n_analyzed}, "
            + ", ".join(f"{k}={v}" for k, v in val.results.items())
        )
    if isinstance(val, ModelingDesignResult):
        return (
            f"eligible_n={val.eligible_n}, horizon_years={val.horizon_years}, "
            f"feasibility={val.feasibility}, "
            f"whitelist_size={len(val.feature_whitelist)}, "
            f"blacklist_size={len(val.leakage_blacklist)}"
        )
    if isinstance(val, DescriptiveStatResult):
        return (
            f"variable={val.variable_id}, n={val.count}, n_missing={val.n_missing}, "
            f"mean={val.mean}, median={val.median}, sd={val.sd}, "
            f"q1={val.q1}, q3={val.q3}, iqr={val.iqr}, "
            f"min={val.min_val}, max={val.max_val}, proportion={val.proportion}, "
            f"event_anchor={val.event_anchor}"
        )
    if isinstance(val, RegressionResult):
        coef_str = "; ".join(
            f"{c.variable}={c.coefficient}(p={c.p_value})"
            for c in val.coefficients
        )
        r2 = val.r_squared if val.r_squared is not None else val.pseudo_r2
        return (
            f"type={val.regression_type}, outcome={val.outcome_label}, "
            f"n={val.n}, r2={r2}, coefficients=[{coef_str}]"
        )
    return repr(val)


# ---------------------------------------------------------------------------
# Enriched planner path (4-layer architecture)
# ---------------------------------------------------------------------------


# Stratifier concepts for gated next-steps. Each key maps to substrings matched
# against a catalog entry's raw_code/concept to decide whether that stratifier is
# actually available for the id_space (never assumed).
_STRATIFIER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sex":       ("sex", "gender", "ptgender", "phc_sex"),
    "apoe":      ("apoe", "apoerores", "apgen", "apoe4", "apoe_genotype"),
    "education": ("educ", "phc_education"),
}


def _verified_stratifiers(catalog: Any, id_space_str: str) -> list[str]:
    """Return catalog variable_ids for stratifiers (sex/APOE/education) actually
    present for the given id_space. Empty when none are catalogued — the
    synthesizer then must not propose stratifying by them (gated next-steps).
    """
    from alz_agent.schemas import IdSpace

    try:
        target = IdSpace(id_space_str)
    except ValueError:
        target = IdSpace.A

    out: list[str] = []
    seen: set[str] = set()
    for e in (catalog or []):
        if getattr(e, "id_space", None) != target:
            continue
        hay = f"{getattr(e, 'raw_code', '')} {getattr(e, 'concept', '')}".lower()
        if any(k in hay for keys in _STRATIFIER_KEYWORDS.values() for k in keys):
            vid = getattr(e, "variable_id", None)
            if vid and vid not in seen:
                seen.add(vid)
                out.append(vid)
    return out[:12]


def _build_enriched_fallback_answer(
    exec_ctx: Any,
    plan: Any,
) -> str:
    """Build a deterministic, grounded answer from execution context.

    Called when the LLM synthesizer produces a hallucination-check failure.
    Only emits integers that appear verbatim in exec_ctx.results, so the
    synthesis hallucination guard will pass on the returned string.

    This function never calls an LLM. Its output is deterministic given the
    same execution context.
    """
    lines: list[str] = [
        "Computed results (grounded values from deterministic tools):",
    ]

    for output_name, result in exec_ctx.results.items():
        if result.grounded and result.value is not None:
            val = result.value
            if isinstance(val, int):
                lines.append(f"  {output_name}: {val}")
            elif isinstance(val, dict):
                lines.append(f"  {output_name}:")
                for k, v in val.items():
                    if k.startswith("_"):
                        continue  # skip internal metadata keys (e.g. _status_metadata)
                    if isinstance(v, dict):
                        inner = ", ".join(
                            f"{ik}={iv}" for ik, iv in v.items() if iv is not None
                        )
                        lines.append(f"    {k}: {inner}")
                    elif v is not None:
                        lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {output_name}: {val}")
        else:
            reason = result.abstain_reason or "not available in current data"
            lines.append(f"  {output_name}: not available ({reason})")

    return "\n".join(lines)


def _check_mutually_exclusive_group_collision(
    semantic_plan: "SemanticAnalysisPlan | None",
    exec_ctx: "ExecutionContext",  # noqa: F821 — imported locally by the caller
) -> list[str]:
    """Post-execution invariant: mutually-exclusive baseline clinical-status groups
    must not resolve to identical eligible_n.

    Compares ONLY groups whose CANONICAL COMPILED filter (semantic_planner.
    _filter_dict_for_scope — the same function C14/C15 and compile_to_enriched use)
    carries a 'diagnosis' key — i.e. groups differentiated purely by clinical/
    syndromic status (CN/MCI/dementia), which are mutually exclusive by construction
    (a single categorical diagnosis field). Etiologic-AD-basis groups are
    deliberately EXCLUDED: etiologic AD may legitimately overlap MCI or dementia, so
    an identical or overlapping count there is expected, not a defect (see
    narration guardrail in the same module).

    Filters on the COMPILED result, not group.etiology_definition/baseline_filters
    directly — a group using the canonical diagnosis_basis='clinical_status' field
    may leave etiology_definition at its UNSPECIFIED default and baseline_filters
    empty (the diagnosis intent lives in clinical_status instead), so checking the
    raw pre-compile attributes would silently exclude exactly the groups this
    check needs to cover.

    Aggregate-count-only comparison (compares integers already surfaced by
    count_eligible ToolResults) — never touches subject-level data, respecting
    Invariant 4 (no raw participant IDs leave the backend). This is a defense-in-depth
    safety net: PlanValidator's C14/C15 already reject the known root cause (identical
    or empty group filters) before execution; this check catches the residual case
    where filters differ structurally but the backend nonetheless returns the same
    count for both — which should never legitimately happen for real, distinct
    baseline diagnosis codes.

    Returns a list of violation messages (empty if no collision detected, or if
    semantic_plan is None / has fewer than 2 comparable groups).
    """
    if semantic_plan is None or len(semantic_plan.groups) < 2:
        return []

    from alz_agent.agent.semantic_planner import _filter_dict_for_scope, _slug

    clinical_groups = [
        g for g in semantic_plan.groups
        if "diagnosis" in _filter_dict_for_scope(g)
    ]
    if len(clinical_groups) < 2:
        return []

    violations: list[str] = []
    seen: dict[Any, str] = {}
    for g in clinical_groups:
        output_name = f"eligible_n_{_slug(g.label)}"
        tr = exec_ctx.results.get(output_name)
        if tr is None or not tr.grounded or not isinstance(tr.value, int):
            continue
        if tr.value in seen:
            violations.append(
                f"Mutually-exclusive-group invariant violated: groups '{seen[tr.value]}' "
                f"and '{g.label}' both resolved to eligible_n={tr.value}. Distinct "
                "baseline clinical-status categories (a single categorical diagnosis "
                "field) cannot legitimately share an identical cohort size at this "
                "scale — treating this as a failed grounding invariant rather than "
                "reporting the numbers as trustworthy."
            )
        else:
            seen[tr.value] = g.label
    return violations


def _mixed_diagnostic_basis_caveat(
    semantic_plan: "SemanticAnalysisPlan | None",
) -> str | None:
    """Return a corrective caveat when the plan mixes clinical-status-basis groups
    with an etiologic-AD-basis group, else None.

    Clinical-status groups (diagnosis_basis='clinical_status', or the legacy
    etiology_definition UNSPECIFIED/ALL_CAUSE_DEMENTIA — CN/MCI/dementia by baseline
    diagnosis) and an etiologic-AD group (diagnosis_basis='etiologic_ad', or legacy
    ETIOLOGIC_AD) are NOT mutually exclusive: an MCI or dementia participant may
    also carry an AD attribution. Narrating them as points on one ordered
    disease-stage/severity gradient is a category error. This purely structural
    check (reads only the already-validated GroupDefinition list, no NL/backend
    access) decides WHETHER the caveat is needed; the wording itself is fixed and
    deterministic — never LLM-authored — so it cannot be skipped by a synthesis
    that doesn't follow the system prompt.

    Checks the canonical diagnosis_basis field first; falls back to the legacy
    etiology_definition enum for plans that predate diagnosis_basis (diagnosis_basis
    is None) — a group can express its basis through either channel.
    """
    if semantic_plan is None or len(semantic_plan.groups) < 2:
        return None

    def _basis(g: "GroupDefinition") -> str:  # noqa: F821 — resolved by caller's import
        if g.diagnosis_basis is not None:
            return g.diagnosis_basis
        if g.etiology_definition == EtiologyDefinition.ETIOLOGIC_AD:
            return "etiologic_ad"
        return "clinical_status"

    bases = {_basis(g) for g in semantic_plan.groups}
    has_clinical = "clinical_status" in bases
    has_etiologic = "etiologic_ad" in bases
    if not (has_clinical and has_etiologic):
        return None
    return (
        "This report includes both clinical-status groups (defined by baseline "
        "cognitive/syndromic diagnosis) and an etiologic-AD group (defined by AD "
        "attribution). These are NOT mutually exclusive — a participant classified "
        "as MCI or dementia may ALSO carry an etiologic AD attribution, so the same "
        "person can appear in more than one group. Do not interpret the reported "
        "values as points on a single ordered disease-stage or severity gradient "
        "across these groups."
    )


def _unaddressed_constraints_caveat(
    semantic_plan: "SemanticAnalysisPlan | None",
) -> str | None:
    """Surface semantic_plan.abstain_if_missing as a mandatory caveat, else None.

    compile_to_enriched() performs zero interpretation — it only ever translates
    already-resolved groups/variable_selections/filters into PlanOperations. Any
    aspect of the question the LLM flagged in abstain_if_missing (e.g. "no MRI
    variable in catalog", "no visit-count variable") is, by construction, NOT
    represented in the compiled operations: there is no mechanism by which such
    a condition could have been silently satisfied. Previously this list was
    folded into ValidationResult.assumptions by PlanValidator._sc9_abstain_
    assumptions(), but nothing ever read that field once validation passed
    (is_executable=True) — the condition was recorded and then discarded, so a
    question like "...and MRI images for at least two visits" would silently
    return a plain baseline cohort count with no indication that the MRI/visit
    part of the question was never applied. This is deterministic, not
    LLM-authored, so it cannot be dropped by a synthesis that ignores the system
    prompt — same pattern as _mixed_diagnostic_basis_caveat.
    """
    if semantic_plan is None or not semantic_plan.abstain_if_missing:
        return None
    items = "; ".join(semantic_plan.abstain_if_missing)
    return (
        "The following aspect(s) of the question could not be grounded in the "
        "current catalog and were NOT applied to the reported figures — treat "
        f"the answer below as partial, not a full answer to the question as asked: {items}."
    )


def _append_caveat(synthesis: "SynthesisReport", caveat_text: str) -> "SynthesisReport":
    """Append caveat_text to the 'caveats' SynthesisSection, creating one if absent.

    SynthesisReport/SynthesisSection are frozen — returns a new SynthesisReport
    with the updated sections list rather than mutating in place.
    """
    from alz_agent.schemas import SynthesisSection

    sections = list(synthesis.sections)
    for i, s in enumerate(sections):
        if s.section_id == "caveats":
            new_content = f"{s.content}\n\n{caveat_text}" if s.content.strip() else caveat_text
            sections[i] = s.model_copy(update={"content": new_content})
            return synthesis.model_copy(update={"sections": sections})

    sections.append(SynthesisSection(section_id="caveats", title="Caveats", content=caveat_text))
    return synthesis.model_copy(update={"sections": sections})


def _build_semantic_abstain_response(
    nl_question: str,
    reason: str,
    vresult: "ValidationResult",
) -> "AgentResponse":
    """Build an AgentResponse that abstains due to unresolvable semantic plan errors."""
    from alz_agent.schemas import SynthesisReport, SynthesisSection

    # requested_measure_resolved=False specifically when the failure is a C10 (catalog
    # existence) or C11 (specimen/analyte consistency) error — i.e. the requested
    # measure itself could not be resolved to a real, matching catalog variable.
    # Other hard-error causes (e.g. an incoherent cohort scope) don't implicate the
    # measure, so they leave this flag at its default True.
    requested_measure_resolved = not any(
        e.startswith("C10:") or e.startswith("C11:") for e in vresult.hard_errors
    )
    report = SynthesisReport(
        question=nl_question,
        sections=[SynthesisSection(
            section_id="direct_answer",
            title="Answer",
            content=(
                "This exact question is not answerable with the current backend "
                f"semantics because: {reason}"
            ),
        )],
        grounded=False,
        abstain_reason=reason,
    )
    return AgentResponse(
        grounded=False,
        answer=(
            "This exact question is not answerable with the current backend "
            f"semantics because: {reason}"
        ),
        tool_results=[],
        doc_evidence=[],
        hallucination_check_passed=True,
        abstain_reason=reason,
        effective_horizon=3,
        synthesis_report=report,
        enriched_plan_used=False,
        enriched_plan_operations=[],
        requested_horizons=[],
        validation_hard_errors=list(vresult.hard_errors),
        validation_soft_warnings=list(vresult.soft_warnings),
        execution_operations_run=[],
        synthesis_report_present=True,
        final_narration_mode="deterministic_fallback",
        final_narration_error="semantic_plan_unrepresentable",
        question_answered=False,
        requested_measure_resolved=requested_measure_resolved,
        semantic_grounding_passed=False,
    )


def _run_enriched_path(
    nl_question: str,
    catalog: list[CatalogEntry],
    backend: DataBackend,
    *,
    client: Any,
    model: str | None = None,
    analysis_engine: Any = None,
    use_semantic_planner: bool = False,
) -> tuple["AgentResponse | None", str | None]:
    """Execute the 4-layer enriched planner pipeline.

    Returns (AgentResponse, None) on success, or (None, error_str) if Layer 1
    planning failed.  The caller must not fall through silently when the second
    element is non-None.

    Layers:
      0. (opt-in) semantic_planner.plan_semantic_analysis() → SemanticAnalysisPlan
         PlanValidator.validate_semantic() → ValidationResult (+ repair loop)
         compile_to_enriched() → EnrichedAnalysisPlan
      1. plan_enriched_analysis() → EnrichedAnalysisPlan   (when use_semantic_planner=False)
      2. PlanValidator.validate()  → ValidationResult
      3. DeterministicExecutor.execute() → ExecutionContext
      4. synthesize_research_report()   → SynthesisReport
    """
    from alz_agent.agent.executor import DeterministicExecutor
    from alz_agent.agent.llm_intent import plan_enriched_analysis
    from alz_agent.agent.narration import synthesize_research_report
    from alz_agent.agent.plan_validator import PlanValidator
    from alz_agent.schemas import (
        EnrichedAnalysisPlan,
        ExecutionContext,
        SynthesisReport,
        SynthesisSection,
        ValidationResult,
    )

    # ---- Layer 0 (optional): Semantic gate ----
    if use_semantic_planner:
        from alz_agent.agent.semantic_planner import (
            compile_to_enriched,
            plan_semantic_analysis,
            repair_semantic_plan,
        )

        sem = plan_semantic_analysis(nl_question, client, model=model, catalog=catalog)
        if sem is None:
            return (None, "semantic_planner_returned_none")

        sem_validator = PlanValidator(catalog)
        sem_validation = sem_validator.validate_semantic(sem)

        if not sem_validation.is_executable:
            # Single repair pass
            sem2 = repair_semantic_plan(
                nl_question, sem, sem_validation, client, model=model, catalog=catalog
            )
            if sem2 is not None:
                sem = sem2
                sem_validation = sem_validator.validate_semantic(sem)

            if not sem_validation.is_executable:
                reason = "; ".join(sem_validation.hard_errors)
                return (_build_semantic_abstain_response(nl_question, reason, sem_validation), None)

        enriched_plan: EnrichedAnalysisPlan = compile_to_enriched(sem, nl_question)
    else:
        # ---- Layer 1: LLM Planner (standard enriched path) ----
        enriched_plan = plan_enriched_analysis(
            nl_question, client, model=model, catalog=catalog
        )
        if enriched_plan is None:
            return (None, "plan_enriched_analysis_returned_none")

    plan_op_names = [f"{op.op_type}:{op.output_name}" for op in enriched_plan.operations]

    # ---- Layer 2: Plan Validator ----
    validator = PlanValidator(catalog)
    validation: ValidationResult = validator.validate(enriched_plan)

    # ---- Layer 2b: Repair pass for group_comparison missing variable IDs ----
    # When the only failures are group_comparison ops missing group_variable_id
    # or outcome_variable_id, attempt one catalog-grounded repair before abstaining.
    if not validation.is_executable:
        gc_var_errors = [
            e for e in validation.hard_errors
            if "group_comparison" in e
            and "missing required parameter" in e
            and ("group_variable_id" in e or "outcome_variable_id" in e)
        ]
        if gc_var_errors:
            from alz_agent.agent.plan_repair import repair_group_comparison_plan
            repaired, repair_warnings = repair_group_comparison_plan(
                enriched_plan,
                list(validation.hard_errors),
                nl_question,
                catalog,
                client=client,
                model=model,
            )
            if repaired is not None:
                re_validation = validator.validate(repaired)
                if re_validation.is_executable:
                    # Repair succeeded — proceed with the repaired plan
                    enriched_plan = repaired
                    plan_op_names = [
                        f"{op.op_type}:{op.output_name}" for op in enriched_plan.operations
                    ]
                    validation = re_validation
                    _log.info(
                        "enriched_path: group_comparison repair succeeded "
                        "(warnings=%d)", len(repair_warnings)
                    )

    if not validation.is_executable:
        hard_err_str = "; ".join(validation.hard_errors)
        abstain_answer = (
            "This question cannot be answered due to plan validation errors.\n"
            f"Errors: {hard_err_str}"
        )
        # Generic clarification (not tied to any specific dataset/variable): a
        # group_comparison op still missing group_variable_id/outcome_variable_id after
        # the repair attempt usually means group_comparison (a binary Welch t-test
        # template) was the wrong tool for the question — most often a comparison across
        # THREE OR MORE named categories (e.g. CN/MCI/AD), for which there is no valid
        # binary grouping variable to resolve. Surface that explanation instead of
        # leaving the researcher to guess why the "missing parameter" occurred.
        unrepaired_gc_errors = [
            e for e in validation.hard_errors
            if "group_comparison" in e
            and "missing required parameter" in e
            and ("group_variable_id" in e or "outcome_variable_id" in e)
        ]
        if unrepaired_gc_errors:
            abstain_answer += (
                "\n\nLikely cause: group_comparison compares exactly TWO groups via a "
                "binary variable (Welch t-test) — it cannot express a comparison across "
                "three or more named categories/diagnostic groups. If this question asks "
                "to compare a measure across 3+ groups (e.g. CN vs. MCI vs. AD), rephrase "
                "or retry — the planner should instead compute one count + one statistic "
                "per group, not a single group_comparison op."
            )
        report = SynthesisReport(
            question=nl_question,
            sections=[SynthesisSection(
                section_id="direct_answer",
                title="Answer",
                content=abstain_answer,
            )],
            validation_result=validation,
            grounded=False,
            abstain_reason=hard_err_str,
        )
        return (AgentResponse(
            grounded=False,
            answer=abstain_answer,
            tool_results=[],
            doc_evidence=[],
            hallucination_check_passed=True,
            abstain_reason=hard_err_str,
            effective_horizon=enriched_plan.requested_horizons[0] if enriched_plan.requested_horizons else 3,
            synthesis_report=report,
            enriched_plan_used=True,
            enriched_plan_operations=plan_op_names,
            requested_horizons=list(enriched_plan.requested_horizons),
            validation_hard_errors=list(validation.hard_errors),
            validation_soft_warnings=list(validation.soft_warnings),
            execution_operations_run=[],
            synthesis_report_present=True,
            final_narration_mode="deterministic_fallback",
            final_narration_error="plan_validation_failed",
            question_answered=False,
            semantic_grounding_passed=False,
        ), None)

    # ---- Layer 3: Deterministic Executor ----
    executor = DeterministicExecutor(catalog, backend, analysis_engine=analysis_engine)
    exec_ctx: ExecutionContext = executor.execute(
        enriched_plan,
        blocked_operations=validation.blocked_operations,
    )

    all_tool_results = list(exec_ctx.results.values())
    all_abstained = bool(all_tool_results) and all(not tr.grounded for tr in all_tool_results)

    # Build ExplanationContext for fallback narration compatibility.
    # The enriched path may produce fan-out (nested dict) results that
    # build_explanation_context cannot handle — wrap in try/except and fall
    # back to a minimal context so the synthesizer still runs.
    effective_horizon = enriched_plan.requested_horizons[0] if enriched_plan.requested_horizons else 3
    horizon_source = "user" if enriched_plan.requested_horizons else "default"
    try:
        ctx = build_explanation_context(
            all_tool_results,
            filter_spec=None,
            horizon=effective_horizon,
            horizon_source=horizon_source,
            catalog=catalog,
            backend_name=type(backend).__name__,
        )
    except Exception:
        ctx = ExplanationContext(
            metadata=ExplanationMetadata(
                horizon_years=effective_horizon,
                horizon_source=horizon_source,
                horizon_label=f"{effective_horizon} year(s)",
                backend=type(backend).__name__,
            )
        )

    if all_abstained:
        reasons = [tr.abstain_reason or "unknown" for tr in all_tool_results]
        combined = "; ".join(dict.fromkeys(reasons))
        abstain_answer = (
            "This question is not currently answerable with available data.\n"
            f"Reason(s): {combined}"
        )
        report = SynthesisReport(
            question=nl_question,
            sections=[SynthesisSection(
                section_id="direct_answer",
                title="Answer",
                content=abstain_answer,
            )],
            validation_result=validation,
            execution_context=exec_ctx,
            grounded=False,
            abstain_reason=combined,
        )
        return (AgentResponse(
            grounded=False,
            answer=abstain_answer,
            tool_results=all_tool_results,
            doc_evidence=[],
            hallucination_check_passed=True,
            abstain_reason=combined,
            effective_horizon=effective_horizon,
            synthesis_report=report,
            enriched_plan_used=True,
            enriched_plan_operations=plan_op_names,
            requested_horizons=list(enriched_plan.requested_horizons),
            validation_hard_errors=[],
            validation_soft_warnings=list(validation.soft_warnings),
            execution_operations_run=list(exec_ctx.operations_run),
            synthesis_report_present=True,
            final_narration_mode="deterministic_fallback",
            final_narration_error="all_executor_ops_abstained",
            question_answered=False,
            semantic_grounding_passed=False,
        ), None)

    # ---- Layer 4: LLM Synthesizer ----
    _log.debug(
        "enriched_path: calling LLM synthesizer (model=%s, ops=%s)",
        model, list(exec_ctx.results.keys()),
    )
    allowed_next_steps = _verified_stratifiers(catalog, enriched_plan.id_space)
    synthesis: SynthesisReport = synthesize_research_report(
        nl_question, enriched_plan, validation, exec_ctx, ctx,
        doc_evidence=[], client=client, model=model,
        allowed_next_step_variables=allowed_next_steps,
    )
    _log.debug(
        "enriched_path: synthesis complete — sections=%d, grounded=%s, "
        "narration_mode=%s, error=%s",
        len(synthesis.sections),
        synthesis.grounded,
        getattr(synthesis.narration_result, "mode", "llm"),
        getattr(synthesis.narration_result, "error", None),
    )

    # Deterministic caveat injection (not reliant on the LLM following the system
    # prompt): when the semantic plan mixes clinical-status-basis groups (CN/MCI/
    # dementia) with an etiologic-AD-basis group, guarantee a corrective statement
    # appears — the two bases are not mutually exclusive, so any ordering/gradient
    # language spanning both is a category error. This is injected as a plan fact,
    # identically for every narration mode (LLM or deterministic fallback), rather
    # than relying on prompt compliance alone (system prompt rule 19 in
    # narration.py is the first line of defense; this is the guaranteed backstop).
    mixed_basis_caveat = _mixed_diagnostic_basis_caveat(
        getattr(enriched_plan, "semantic_plan", None)
    )
    if mixed_basis_caveat:
        synthesis = _append_caveat(synthesis, mixed_basis_caveat)

    unaddressed_caveat = _unaddressed_constraints_caveat(
        getattr(enriched_plan, "semantic_plan", None)
    )
    if unaddressed_caveat:
        synthesis = _append_caveat(synthesis, unaddressed_caveat)

    # Assemble answer from synthesis sections
    answer = "\n\n".join(
        f"## {s.title}\n{s.content}"
        for s in synthesis.sections
    ).strip()
    if not answer:
        answer = "No answer could be generated."

    # Hallucination guard on synthesis.
    # When the synthesized answer contains numbers that cannot be traced to tool
    # evidence, discard it and fall back to a deterministic grounded template.
    # This enforces the invariant that no number in the final answer may come from
    # LLM generation — every integer must trace to an executor result.
    hc_passed = _check_synthesis_hallucination(synthesis)
    # Variable-identifier guard: reject fabricated / cross-space variable ids
    # (e.g. an invented "ADNI.OTHER.NACCAGE" or a NACC.* id in a Space B report).
    varid_passed, varid_violations = _check_synthesis_variable_ids(
        synthesis, catalog=catalog, plan=enriched_plan
    )
    if not varid_passed:
        _log.warning(
            "enriched_path: variable-id guard FAILED (violations=%s) — "
            "switching to deterministic fallback", varid_violations[:5],
        )
    # External-claim guard: synthesize_research_report() already retries once
    # internally (narration.py) when the LLM writes an unsupported external-
    # literature/adequacy/biological-explanation claim with no doc evidence to
    # back it. synthesis_external_claim_violation is True only if THAT retry
    # still failed to remove it — previously this flag was set but never
    # checked anywhere, so a persistent violation could still reach the final
    # answer. Route it through the same deterministic-fallback gate as the
    # other two guards, closing that gap.
    ext_claim_passed = not synthesis.synthesis_external_claim_violation
    if not ext_claim_passed:
        _log.warning(
            "enriched_path: external-claim guard FAILED after retry — "
            "switching to deterministic fallback"
        )
    _log.debug(
        "enriched_path: hallucination_check=%s, variable_id_check=%s, "
        "external_claim_check=%s, answer_prefix=%r",
        hc_passed, varid_passed, ext_claim_passed, answer[:80],
    )

    if not hc_passed or not varid_passed or not ext_claim_passed:
        _log.warning(
            "enriched_path: synthesis guard FAILED — switching to deterministic fallback"
        )
        answer = _build_enriched_fallback_answer(exec_ctx, enriched_plan)
        # The deterministic fallback rebuilds the answer from raw exec_ctx values only
        # — it does not carry synthesis.sections, so the mixed-basis/unaddressed-
        # constraints caveats injected above would otherwise be silently dropped
        # here. Re-append them directly.
        if mixed_basis_caveat:
            answer = f"{answer}\n\nCaveat: {mixed_basis_caveat}"
        if unaddressed_caveat:
            answer = f"{answer}\n\nCaveat: {unaddressed_caveat}"
        narration_mode = "deterministic_fallback"
        # The fallback answer contains only integers from exec_ctx, so the guard
        # will pass when applied to the new content.
        hc_passed = True
    else:
        narration_mode = "llm"

    # Safety net: if the answer still starts with "Computed results" (i.e. the
    # deterministic fallback was used), try one LLM synthesis retry before giving up.
    # This covers guard failures that the primary fix did not catch.
    if answer.lstrip().startswith("Computed results") and client is not None:
        _log.warning(
            "enriched_path: final answer is raw fallback ('Computed results...'); "
            "attempting one LLM synthesis retry"
        )
        try:
            synthesis_retry: SynthesisReport = synthesize_research_report(
                nl_question, enriched_plan, validation, exec_ctx, ctx,
                doc_evidence=[], client=client, model=model,
                allowed_next_step_variables=allowed_next_steps,
            )
            # Re-apply the same deterministic caveat injections to the retry — it
            # carries a fresh synthesis.sections that would otherwise silently drop
            # the mandatory caveats if this retry is accepted below.
            if mixed_basis_caveat:
                synthesis_retry = _append_caveat(synthesis_retry, mixed_basis_caveat)
            if unaddressed_caveat:
                synthesis_retry = _append_caveat(synthesis_retry, unaddressed_caveat)
            retry_answer = "\n\n".join(
                f"## {s.title}\n{s.content}"
                for s in synthesis_retry.sections
            ).strip()
            retry_hc = _check_synthesis_hallucination(synthesis_retry)
            # Also re-check the external-claims guard — synthesize_research_report()
            # already retries this internally once, but a persistent violation must
            # not be silently accepted here either (this is the same gap fixed above
            # for the primary synthesis, applied consistently to the safety-net retry).
            retry_ext_passed = not synthesis_retry.synthesis_external_claim_violation
            _log.debug(
                "enriched_path: retry answer_prefix=%r, hc=%s, ext_claim_passed=%s",
                retry_answer[:80], retry_hc, retry_ext_passed,
            )
            if (
                retry_answer
                and not retry_answer.lstrip().startswith("Computed results")
                and retry_hc
                and retry_ext_passed
            ):
                answer = retry_answer
                synthesis = synthesis_retry
                narration_mode = "llm"
                hc_passed = True
                _log.debug("enriched_path: synthesis retry succeeded")
            else:
                _log.warning(
                    "enriched_path: synthesis retry still unsuitable (hc=%s, prefix=%r); "
                    "keeping deterministic fallback", retry_hc, retry_answer[:60],
                )
        except Exception as exc:
            _log.warning("enriched_path: synthesis retry failed with exception: %s", exc)

    # question_answered: every operation the compiled plan actually ran produced a
    # grounded result, and none were blocked pre-execution by the validator. A plan
    # that ran but had some ops abstain/error, or had ops blocked outright, did not
    # fully answer what was asked — even if grounded=True overall from the ops that
    # did succeed. Generic check: works for any op_type / dataset, not measure-specific.
    # Also False when the semantic plan itself flagged unaddressable aspects of the
    # question (abstain_if_missing) — compile_to_enriched cannot represent those in
    # the compiled operations, so a fully-grounded execution still only answers part
    # of what was asked (see _unaddressed_constraints_caveat).
    question_answered = (
        not validation.blocked_operations
        and bool(exec_ctx.results)
        and all(tr.grounded for tr in exec_ctx.results.values())
        and unaddressed_caveat is None
    )

    # Post-execution invariant (defense in depth, independent of C14/C15 which catch
    # the same defect pre-execution): mutually exclusive baseline clinical-status
    # groups (differentiated purely by baseline_filters diagnosis — CN/MCI/dementia,
    # never etiologic-AD groups, which are explicitly allowed to overlap) must not
    # resolve to identical eligible_n. Aggregate-count-only comparison — never touches
    # subject-level data, respecting Invariant 4 (no raw IDs leave the backend).
    group_collision_violations = _check_mutually_exclusive_group_collision(
        getattr(enriched_plan, "semantic_plan", None), exec_ctx,
    )
    if group_collision_violations:
        _log.warning(
            "enriched_path: mutually-exclusive group collision detected — %s",
            group_collision_violations,
        )

    # requested_measure_resolved: guaranteed True here for the semantic-planner path —
    # a C10/C11 failure would have aborted at the hard-error gate before execution ever
    # ran (see _build_semantic_abstain_response). True by default for the classic path,
    # which does not run C10/C11 checks.
    semantic_grounding_passed = (
        synthesis.grounded and question_answered and not group_collision_violations
    )
    if group_collision_violations:
        question_answered = False

    return (AgentResponse(
        grounded=synthesis.grounded,
        answer=answer,
        tool_results=all_tool_results,
        doc_evidence=[],
        hallucination_check_passed=hc_passed,
        effective_horizon=effective_horizon,
        synthesis_report=synthesis,
        enriched_plan_used=True,
        enriched_plan_operations=plan_op_names,
        requested_horizons=list(enriched_plan.requested_horizons),
        validation_hard_errors=[],
        validation_soft_warnings=list(validation.soft_warnings) + group_collision_violations,
        execution_operations_run=list(exec_ctx.operations_run),
        synthesis_report_present=True,
        final_narration_mode=narration_mode,
        final_narration_model=model if narration_mode == "llm" else None,
        question_answered=question_answered,
        semantic_grounding_passed=semantic_grounding_passed,
    ), None)
