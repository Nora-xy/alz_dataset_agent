"""hf-local deterministic answer synthesis — no LLM narration.

hf-local's compact schema only ever produces COHORT_SIZE or AGGREGATE_STAT
questions (see hf_local_intent.py). Routing these through the general LLM
narration path (`narrate_with_llm`) risks two failure modes observed with a
small local model (Qwen2.5-3B, fixture backend):

  1. Long, LLM-authored explanatory paragraphs for what should be a single
     grounded number.
  2. Picking the WRONG number entirely for a modality-overlap union — e.g.
     narrating the unrestricted total-cohort count as if it were the answer
     to "cognition AND (MRI OR PET)", because free-form narration has no
     obligation to derive the union correctly from the combo dict.

This module replaces narrate_with_llm for hf-local's narrow question shapes
with a short, deterministic template. Every number it states is either:
  (a) a single ToolResult's own value (e.g. compute_cohort_size's int), or
  (b) a documented inclusion-exclusion combination of
      assess_modality_overlap's per-combo counts (resolve_modality_union) —
      never a number invented by generation.

When the requested combination cannot be derived exactly (suppressed or
missing combo cells, or an any_of shape this module does not know how to
combine), it abstains with an explicit "cannot determine" message rather
than falling back to a broader-but-wrong number such as the total cohort.

Public:
  resolve_modality_union(intent, overlap_counts) -> int | None
  synthesize_hf_local_answer(intent, tool_results, backend_name) -> str
  hf_local_extra_evidence_digits(intent, tool_results) -> set[str]
      Digit strings for numbers this module *derives* (not verbatim tool
      values) — e.g. an inclusion-exclusion union — so the hallucination
      guard's `extra` allowlist can accept them without weakening the guard
      for anything else.

Only used when the orchestrator's client is a LocalHFPlannerClient — see the
isinstance guard in orchestrator.run_agent(). DeepSeek/Anthropic narration
(narrate_with_llm) is untouched.
"""

from __future__ import annotations

from alz_agent.agent.intent_schema import QuestionType, StructuredIntent
from alz_agent.schemas import ToolResult
from alz_agent.tools.deterministic import StatResult


def _is_fixture_backend(backend_name: str) -> bool:
    return "fixture" in backend_name.lower()


def _fixture_caveat(backend_name: str) -> str:
    if _is_fixture_backend(backend_name):
        return " This is synthetic fixture data, not a real NACC/ADSP-PHC/ADNI count."
    return ""


def _backend_label(backend_name: str) -> str:
    return "Fixture backend" if _is_fixture_backend(backend_name) else "Backend"


def _find_cohort_size_result(tool_results: list[ToolResult]) -> ToolResult | None:
    for tr in tool_results:
        if isinstance(tr.value, int) and not isinstance(tr.value, bool):
            return tr
    return None


def _find_modality_overlap_result(tool_results: list[ToolResult]) -> ToolResult | None:
    for tr in tool_results:
        if isinstance(tr.value, dict):
            return tr
    return None


def _find_stat_result(tool_results: list[ToolResult]) -> ToolResult | None:
    for tr in tool_results:
        if isinstance(tr.value, StatResult):
            return tr
    return None


def resolve_modality_union(
    intent: StructuredIntent, overlap_counts: dict
) -> int | None:
    """Deterministically resolve the requested modality count from a
    modality_intersection combo dict (keys are comma-joined sorted modality
    names, e.g. "cognition,mri"), honoring any_of_modalities (OR semantics)
    instead of assuming a plain AND across every requested modality.

    Returns None when the exact requested combination cannot be derived
    (missing/suppressed cells, or a combo shape this module doesn't
    recognize) — callers must abstain rather than guess in that case.
    """
    modalities = intent.modalities
    if not modalities:
        return None
    any_of = intent.any_of_modalities

    if not any_of:
        key = ",".join(sorted(modalities))
        return overlap_counts.get(key)

    required = [m for m in modalities if m not in any_of]
    # Only the shapes hf-local's adapter can produce today: a two-way OR
    # (any_imaging -> mri/pet) combined with zero or one required modality.
    # Any other shape is intentionally unsupported here — abstain rather
    # than derive an inclusion-exclusion formula that hasn't been validated
    # for it.
    if len(any_of) != 2 or len(required) > 1:
        return None
    a, b = sorted(any_of)

    if not required:
        # |a ∪ b| = |a| + |b| - |a ∩ b|
        key_a, key_b, key_ab = a, b, ",".join(sorted([a, b]))
        v_a, v_b, v_ab = (
            overlap_counts.get(key_a), overlap_counts.get(key_b), overlap_counts.get(key_ab),
        )
        if v_a is None or v_b is None or v_ab is None:
            return None
        return v_a + v_b - v_ab

    r = required[0]
    key_ra = ",".join(sorted([r, a]))
    key_rb = ",".join(sorted([r, b]))
    key_rab = ",".join(sorted([r, a, b]))
    v_ra = overlap_counts.get(key_ra)
    v_rb = overlap_counts.get(key_rb)
    v_rab = overlap_counts.get(key_rab)
    if v_ra is None or v_rb is None or v_rab is None:
        return None
    # |required ∩ (a ∪ b)| = |required ∩ a| + |required ∩ b| - |required ∩ a ∩ b|
    return v_ra + v_rb - v_rab


def hf_local_extra_evidence_digits(
    intent: StructuredIntent, tool_results: list[ToolResult]
) -> set[str]:
    """Digit strings for numbers this module derives (not verbatim tool
    values), for the hallucination guard's `extra` allowlist. Empty set when
    no derivation applies (plain-AND modality counts and cohort-size/stat
    values are already verbatim in tool evidence and need no whitelisting)."""
    if intent.question_type != QuestionType.COHORT_SIZE or not intent.any_of_modalities:
        return set()
    overlap_result = _find_modality_overlap_result(tool_results)
    if overlap_result is None or not overlap_result.grounded:
        return set()
    value = resolve_modality_union(intent, overlap_result.value)
    if value is None:
        return set()
    return {str(value)}


def synthesize_hf_local_answer(
    intent: StructuredIntent,
    tool_results: list[ToolResult],
    backend_name: str,
) -> str:
    """Deterministic, short answer for hf-local's narrow question set (count,
    modality_overlap, aggregate_stat). Never invents a number: every value
    stated is either a literal ToolResult value or the documented
    inclusion-exclusion combination in resolve_modality_union()."""
    caveat = _fixture_caveat(backend_name)
    label = _backend_label(backend_name)

    if intent.question_type == QuestionType.COHORT_SIZE:
        if intent.modalities:
            overlap_result = _find_modality_overlap_result(tool_results)
            if overlap_result is None or not overlap_result.grounded:
                reason = (
                    overlap_result.abstain_reason
                    if overlap_result is not None
                    else "no modality-overlap tool result was returned"
                )
                return (
                    "I cannot determine the exact count from the available "
                    f"tool result. Reason: {reason}"
                )
            value = resolve_modality_union(intent, overlap_result.value)
            if value is None:
                return (
                    "I cannot determine the exact count from the available "
                    f"tool result. The available result only shows partial "
                    f"overlap counts {overlap_result.value}, not the full "
                    f"combination requested by '{intent.candidate_label}'."
                )
            return f"{label} returned N={value} for {intent.candidate_label}.{caveat}"

        cohort_result = _find_cohort_size_result(tool_results)
        if cohort_result is None or not cohort_result.grounded:
            reason = (
                cohort_result.abstain_reason
                if cohort_result is not None
                else "no cohort-size tool result was returned"
            )
            return (
                "This question is not currently answerable with available "
                f"data. Reason: {reason}"
            )
        return f"{label} returned N={cohort_result.value} for {intent.candidate_label}.{caveat}"

    if intent.question_type == QuestionType.AGGREGATE_STAT:
        stat_result = _find_stat_result(tool_results)
        if stat_result is None or not stat_result.grounded:
            reason = (
                stat_result.abstain_reason
                if stat_result is not None
                else "no aggregate-stat tool result was returned"
            )
            return (
                "This question is not currently answerable with available "
                f"data. Reason: {reason}"
            )
        val: StatResult = stat_result.value
        return (
            f"{label} returned {val.stat_type}={val.value} for "
            f"{intent.candidate_label} (n={val.n}).{caveat}"
        )

    # hf-local's compact schema never produces any other question_type
    # (see hf_local_intent.compact_intent_to_structured_intent) — this is an
    # unreachable defensive fallback, not a supported path.
    return "This question is outside hf-local's supported simple-task set."
