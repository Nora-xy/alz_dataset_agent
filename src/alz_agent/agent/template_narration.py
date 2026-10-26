"""Deterministic template narration — used for tests and as LLM fallback.

INVARIANT: every integer written here must come directly from a ToolResult
value or evidence aggregate_ref / query_spec — never from derived expressions
that introduce new numbers.  The hallucination guard validates this.

Public functions:
  _narrate_template(intent, tool_results, doc_evidence, ctx=None) -> str
  _narrate_template_from_ctx(ctx, doc_evidence)                   -> str
  _narrate_template_legacy(intent, tool_results, doc_evidence)    -> str
  _strip_integers_from_definition(text)                           -> str
"""

from __future__ import annotations

import re
from typing import Any

from alz_agent.agent.intent_schema import StructuredIntent
from alz_agent.schemas import (
    DocEvidence,
    ExplanationContext,
    GroupComparisonResult,
    ToolResult,
)
from alz_agent.tools.deterministic import ModelingDesignResult, StatResult

try:
    from alz_agent.schemas import GeneratedAnalysisResult
except ImportError:
    GeneratedAnalysisResult = None  # type: ignore[assignment,misc]


def _narrate_template(
    intent: StructuredIntent,
    tool_results: list[ToolResult],
    doc_evidence: list[DocEvidence],
    ctx: ExplanationContext | None = None,
) -> str:
    """Deterministic fallback narration — used when no LLM client is available.

    When ctx is provided, renders a structured, deduplicated section-by-section answer.
    When ctx is unavailable, falls back to the legacy flat key-value format.
    Clearly marked as a template (no-LLM) output.
    """
    if ctx is not None:
        return _narrate_template_from_ctx(ctx, doc_evidence)
    return _narrate_template_legacy(intent, tool_results, doc_evidence)


def _narrate_template_from_ctx(
    ctx: ExplanationContext,
    doc_evidence: list[DocEvidence],
) -> str:
    """Structured template narration using ExplanationContext (deduplicated, sectioned)."""
    lines: list[str] = [
        "Modeling-design summary (deterministic template — no LLM narration active).",
        "",
    ]

    item_by_id = {i.item_id: i for i in ctx.items}

    for section in ctx.sections:
        if section.section_id == "metadata":
            continue  # rendered at the bottom
        if not section.item_ids:
            continue
        lines.append(f"## {section.title}")
        for iid in section.item_ids:
            item = item_by_id.get(iid)
            if item is None:
                continue
            if item.is_not_answerable:
                lines.append(
                    f"- [NOT ANSWERABLE] {item.display_name}: {item.definition}"
                )
                if item.caveats:
                    lines.append(f"  Next step: {item.caveats[0]}")
                continue
            if item.value is None:
                lines.append(f"- {item.display_name}: suppressed (below reporting threshold)")
            elif isinstance(item.value, (int, float)):
                # Numeric values are narrated verbatim — they are in the evidence set.
                unit_str = f" {item.unit}" if item.unit else ""
                denom_str = (
                    f" / {item.denominator}" if item.denominator is not None else ""
                )
                lines.append(f"- {item.display_name}: {item.value}{unit_str}{denom_str}")
                defn = _strip_integers_from_definition(item.definition)
                if defn:
                    lines.append(f"  ({defn})")
                if item.computation_note:
                    note = _strip_integers_from_definition(item.computation_note)
                    if note:
                        lines.append(f"  Note: {note}")
            else:
                # String values may contain arbitrary digits — strip them before rendering.
                # The display_name is the narrated label; the value is metadata context.
                safe_val = _strip_integers_from_definition(str(item.value))
                lines.append(f"- {item.display_name}: {safe_val}")
                defn = _strip_integers_from_definition(item.definition)
                if defn:
                    lines.append(f"  ({defn})")
        lines.append("")

    # Caveats section — strip integers from all caveat text since these are
    # explanatory metadata, not narrated numeric facts.
    if ctx.caveats:
        lines.append("## Caveats and limitations")
        for cav in ctx.caveats:
            if cav.parsed_target_etiology:
                lines.append(f"- Target etiology requested: {cav.parsed_target_etiology}")
            if cav.actual_endpoint:
                lines.append(
                    f"  Available endpoint: "
                    f"{_strip_integers_from_definition(cav.actual_endpoint)}"
                )
            if cav.implication:
                lines.append(
                    f"  Implication: "
                    f"{_strip_integers_from_definition(cav.implication)}"
                )
            if cav.required_next_step:
                lines.append(
                    f"  Next step: "
                    f"{_strip_integers_from_definition(cav.required_next_step)}"
                )
        lines.append("")

    # Documentation evidence
    if doc_evidence:
        lines.append("## Documentation evidence")
        for ev in doc_evidence:
            lines.append(f"[{ev.source} / {ev.locator}] {ev.snippet[:120]}")
        lines.append("")

    # Metadata / audit block.  horizon_years is a deterministic value whitelisted
    # into evidence by run_agent, so it can be rendered as a literal integer.
    horizon_source_note = (
        " (example default — user did not specify)"
        if ctx.metadata.horizon_source == "default"
        else ""
    )
    lines.append("## Metadata (audit)")
    lines.append(
        f"- Prediction horizon : {ctx.metadata.horizon_years} year(s){horizon_source_note}"
    )
    lines.append(f"- Horizon source: {ctx.metadata.horizon_source}")
    if ctx.metadata.backend:
        lines.append(f"- Backend: {ctx.metadata.backend}")
    if ctx.metadata.tools_invoked:
        lines.append(f"- Tools invoked: {', '.join(ctx.metadata.tools_invoked)}")
    lines.append(f"- Grounding: deterministic (template mode)")

    return "\n".join(lines)


def _strip_integers_from_definition(text: str) -> str:
    """Remove bare digit sequences from definition text to keep them out of the answer.

    This prevents the hallucination guard from flagging definitions that happen to
    contain digits (e.g. threshold values, NACCUDSD codes) that are not in evidence.
    The definition is metadata context for the LLM, not a narrated fact.
    """
    # Replace digit-only tokens with a placeholder
    cleaned = re.sub(r"\b\d+\b", "#", text)
    return cleaned.strip()


def _narrate_template_legacy(
    intent: StructuredIntent,
    tool_results: list[ToolResult],
    doc_evidence: list[DocEvidence],
) -> str:
    """Legacy flat key-value template (used when ctx is unavailable)."""
    lines: list[str] = [
        f"Question type : {intent.question_type.value}",
        f"Target population : {', '.join(intent.target_population)}",
        f"Candidate label : {intent.candidate_label}",
        "",
        "--- Tool results ---",
    ]

    for tr in tool_results:
        if not tr.grounded:
            lines.append(f"[NOT ANSWERABLE] {tr.abstain_reason}")
            continue

        val = tr.value

        if isinstance(val, int):
            lines.append(f"Eligible participants : {val}")

        elif isinstance(val, dict):
            for k, v in sorted(val.items()):
                if v is not None:
                    lines.append(f"{k} : {v}")
                else:
                    lines.append(f"{k} : suppressed (below reporting threshold)")

        elif isinstance(val, StatResult):
            anchor_note = f" (at {val.event_anchor})" if val.event_anchor else ""
            lines.append(
                f"{val.stat_type.capitalize()} {val.variable_id}{anchor_note} : "
                f"{val.value} (n={val.n})"
            )

        elif isinstance(val, GroupComparisonResult):
            lines.append(f"Analysis : {val.analysis_type}")
            lines.append(f"Cohort   : {val.cohort_def}")
            lines.append(f"Outcome  : {val.outcome_label} ({val.outcome_variable_id})")
            lines.append(f"Grouping : {val.group_label} ({val.group_variable_id})")
            for g in val.groups:
                lines.append(
                    f"  {g.group_label} — n={g.n}, mean={g.mean}, SD={g.sd}"
                )
            if val.p_value is not None:
                lines.append(f"P-value ({val.test_name}) : {val.p_value}")
            else:
                lines.append("P-value : not computed (insufficient observations)")
            for c in val.caveats:
                lines.append(f"Caveat: {c}")

        elif GeneratedAnalysisResult is not None and isinstance(val, GeneratedAnalysisResult):
            lines.append(f"Analysis : {val.analysis_description}")
            lines.append(f"Cohort   : {val.cohort_def}")
            lines.append(f"N analyzed: {val.n_analyzed}")
            for k, v in val.results.items():
                if k != "n_analyzed":
                    lines.append(f"  {k} : {v}")

        elif isinstance(val, ModelingDesignResult):
            if val.eligible_n is not None:
                lines.append(f"Eligible N : {val.eligible_n}")
            lines.append(f"Prediction horizon : {val.horizon_years} year(s)")
            for k, v in sorted(val.feasibility.items()):
                if v is not None:
                    lines.append(f"Follow-up {k} : {v}")
                else:
                    lines.append(f"Follow-up {k} : suppressed (below reporting threshold)")
            lines.append(
                f"Pre-index feature whitelist : {len(val.feature_whitelist)} variable(s)"
            )
            lines.append(
                f"Leakage blacklist : {len(val.leakage_blacklist)} variable(s) excluded"
            )
            for rec in val.leakage_records:
                lines.append(
                    f"  Excluded: {rec.raw_code} ({rec.display_name}) "
                    f"[tag: {rec.tag}] — {rec.reason}"
                )

        else:
            lines.append(str(val))

    # AD-specificity caveat: when the candidate label names AD or Alzheimer's
    if re.search(r"\bad\b|\balzheimer", intent.candidate_label.lower()):
        lines.append("")
        lines.append(
            "Caveat: NACC and ADSP-PHC encode all-cause dementia as the conversion "
            "endpoint. AD etiologic attribution is a separate concept and is not "
            "separately modeled here — treat this result as an upper bound on "
            "AD-specific conversion feasibility."
        )

    if doc_evidence:
        lines.append("")
        lines.append("--- Documentation evidence ---")
        for ev in doc_evidence:
            lines.append(f"[{ev.source} / {ev.locator}] {ev.snippet[:120]}")

    return "\n".join(lines)
