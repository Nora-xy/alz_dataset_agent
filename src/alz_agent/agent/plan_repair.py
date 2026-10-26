"""plan_repair.py — narrow repair pass for group_comparison operations missing variable IDs.

Called from _run_enriched_path after PlanValidator returns is_executable=False specifically
because a group_comparison op is missing group_variable_id or outcome_variable_id.

Repair steps:
1. Identify which variable IDs are missing (group_variable_id and/or outcome_variable_id).
2. Call the LLM (if client is available) to extract concept phrases from the question.
3. Use lookup_variable() against the catalog to resolve each concept phrase.
   - Pass 0 alias map catches "mmse score" → NACCMMSE, "heart attack history" → CVHATT.
   - Passes 1-5 catch raw_code, concept, substring, and keyword matches.
4. Apply specificity preference: "heart attack history" resolves to the exact heart-attack
   variable before any broader heart-disease composite.
5. Return a repaired EnrichedAnalysisPlan, or (None, [reason]) if repair fails.

No LLM-generated numbers here. All variable_ids trace to lookup_variable catalog results.
"""

from __future__ import annotations

import json
import re
from typing import Any

from alz_agent.agent.catalog_context import _build_catalog_context
from alz_agent.agent.providers import PlannerClient
from alz_agent.schemas import CatalogEntry, EnrichedAnalysisPlan, PlanOperation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered specificity chains for common group-variable concepts.
# Each tuple is tried in order via lookup_variable(); first catalog hit wins.
# "heart attack history" should match CVHATT before any broader heart-disease composite.
#
# NOTE: do NOT include raw_codes like "CVHATT" here — they would substring-match
# broader variables such as "CVHATT_BROAD" via Pass 4.  The alias map already handles
# "heart attack history" → CVHATT (exact raw_code) for any catalog that has CVHATT.
_HEART_ATTACK_CONCEPTS: tuple[str, ...] = (
    "heart attack history",        # alias map (Pass 0) → CVHATT
    "heart attack",                # alias map (Pass 0) → CVHATT
    "history of heart attack",     # alias map (Pass 0) → CVHATT
    "cardiac arrest history",      # alias map (Pass 0) → CVHATT
    "cardiac arrest",              # alias map (Pass 0) → CVHATT
    "myocardial infarction",       # alias map (Pass 0) → CVHATT
)

# Broad fallback — only used when the specific heart-attack chain fails entirely
_HEART_DISEASE_BROAD_CONCEPTS: tuple[str, ...] = (
    "heart disease",
    "cardiovascular disease",
    "cardiovascular history",
    "cardiac history",
    "CVD",
)

# Ordered specificity chain for MMSE outcome
_MMSE_CONCEPTS: tuple[str, ...] = (
    "NACCMMSE",        # exact raw_code
    "mmse score",      # alias map
    "mmse",            # alias map
    "mini mental",     # alias map
    "MMSE30",          # alternative NACC raw_code (baseline MMSE)
    "MMSE score",      # concept substring / keyword
    "mental status examination",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_concepts_chain(
    concepts: tuple[str, ...],
    catalog: list[CatalogEntry],
) -> tuple[str | None, str | None]:
    """Try a sequence of concept phrases; return (variable_id, raw_code) of first catalog hit."""
    from alz_agent.tools.deterministic import lookup_variable
    for concept in concepts:
        result = lookup_variable(concept, catalog)
        if result.grounded and result.value:
            return result.value.variable_id, result.value.raw_code
    return None, None


def _classify_concept_to_chain(
    phrase: str,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Map a concept phrase to (preferred_chain, broad_fallback_chain | None).

    Returns (preferred, broad_fallback) where:
    - preferred is tried first; first hit wins without any warning.
    - broad_fallback is tried only if preferred fails; triggers a proxy warning.
    - When concept is unrecognized, returns ((phrase,), None) — direct lookup only.
    """
    phrase_lower = phrase.lower()
    if any(kw in phrase_lower for kw in (
        "heart attack", "cardiac arrest", "myocardial infarction",
    )):
        return _HEART_ATTACK_CONCEPTS, _HEART_DISEASE_BROAD_CONCEPTS
    if any(kw in phrase_lower for kw in (
        "mmse", "mini mental", "mini-mental", "mental status",
    )):
        return _MMSE_CONCEPTS, None
    # Unknown concept: try the phrase itself via all 5 lookup_variable passes.
    return (phrase,), None


def _resolve_concept_to_variable(
    concept: str,
    catalog: list[CatalogEntry],
) -> tuple[str | None, str | None, str | None]:
    """Resolve a plain-language concept phrase to a catalog variable.

    Returns (variable_id, raw_code, proxy_warning).
    - variable_id and raw_code are None when resolution fails entirely.
    - proxy_warning is a non-None string when a broad proxy is used instead of
      the exact variable requested (e.g. broad heart disease instead of CVHATT).
    """
    preferred, broad_fallback = _classify_concept_to_chain(concept)

    # Try preferred chain first
    var_id, raw_code = _try_concepts_chain(preferred, catalog)
    if var_id:
        return var_id, raw_code, None

    # Try the concept phrase directly (5-pass lookup_variable search)
    from alz_agent.tools.deterministic import lookup_variable
    result = lookup_variable(concept, catalog)
    if result.grounded and result.value:
        return result.value.variable_id, result.value.raw_code, None

    # Try broad fallback — this triggers a proxy warning
    if broad_fallback:
        var_id, raw_code = _try_concepts_chain(broad_fallback, catalog)
        if var_id:
            proxy_warning = (
                f"Exact variable for '{concept}' not found in catalog. "
                f"Using broader variable '{raw_code}' as a proxy. "
                "This may not precisely capture the requested concept — "
                "results should be interpreted accordingly."
            )
            return var_id, raw_code, proxy_warning

    return None, None, None


def _is_group_comparison_missing_vars_error(error: str) -> bool:
    """Return True if the error is a group_comparison missing required variable ID."""
    return (
        "group_comparison" in error
        and "missing required parameter" in error
        and ("group_variable_id" in error or "outcome_variable_id" in error)
    )


def _missing_params_from_error(error: str) -> list[str]:
    """Extract which variable ID params are missing from a validation error string."""
    missing: list[str] = []
    if "group_variable_id" in error:
        missing.append("group_variable_id")
    if "outcome_variable_id" in error:
        missing.append("outcome_variable_id")
    return missing


# ---------------------------------------------------------------------------
# LLM-based concept extraction
# ---------------------------------------------------------------------------


def _llm_extract_concepts(
    nl_question: str,
    missing_params: list[str],
    op_output_name: str,
    catalog: list[CatalogEntry],
    client: Any,
    model: str | None,
) -> dict[str, str | None] | None:
    """Ask the LLM to identify concept phrases for missing group_comparison variables.

    Returns a dict with some subset of:
        {"group_concept": "...", "outcome_concept": "..."}
    where values are plain-language concept phrases (NOT raw_codes).
    Returns None if the LLM call fails.
    """
    catalog_context = _build_catalog_context(nl_question, catalog)

    missing_descs: list[str] = []
    if "group_variable_id" in missing_params:
        missing_descs.append(
            '"group_concept": the binary predictor / grouping variable '
            '(e.g. "heart attack history", "hypertension", "sex")'
        )
    if "outcome_variable_id" in missing_params:
        missing_descs.append(
            '"outcome_concept": the continuous outcome variable to compare between groups '
            '(e.g. "MMSE score", "CDR sum of boxes", "age")'
        )

    missing_fields_block = "\n".join(f"  {d}," for d in missing_descs)

    prompt = (
        "The group comparison operation is missing required variable IDs.\n"
        "Resolve the missing outcome and/or group variable from the question and catalog.\n"
        "Do not invent variable IDs. Use lookup_variable results or catalog aliases.\n"
        "If no catalog match is found, abstain with a clear message.\n\n"
        f"QUESTION: {nl_question}\n\n"
        f"FAILING OPERATION: {op_output_name} (group_comparison)\n"
        f"MISSING PARAMETERS: {missing_params}\n\n"
        f"CATALOG CONTEXT (variable raw_codes and concepts available):\n{catalog_context}\n\n"
        "Return ONLY a JSON object with the following keys "
        "(use plain English researcher phrases, NOT raw_codes):\n"
        "{\n"
        f"{missing_fields_block}\n"
        "}\n\n"
        "RULES:\n"
        "- Use the exact researcher-language concept from the question "
        '  (e.g. "heart attack history", not "CVHATT").\n'
        "- 'heart attack history' is more specific than 'heart disease' — prefer the specific term.\n"
        "- Do NOT use raw_codes like CVHATT or NACCMMSE — use researcher concept phrases.\n"
        "- If you cannot identify a concept from the question, set that key to null.\n"
        "- Respond with ONLY the JSON object — no markdown, no prose.\n"
    )

    try:
        if isinstance(client, PlannerClient):
            raw = client.complete_text(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
            )
        else:
            response = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text_block = next(
                (b for b in response.content if getattr(b, "type", None) == "text"),
                None,
            )
            if text_block is None:
                return None
            raw = text_block.text

        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return data if isinstance(data, dict) else None

    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Heuristic concept extraction (no LLM — keyword scan of the question)
# ---------------------------------------------------------------------------


def _heuristic_outcome_concept(nl_question: str) -> str | None:
    """Extract outcome concept from question text using simple keyword matching."""
    q = nl_question.lower()
    if "mmse" in q:
        return "MMSE score"
    if "cdr" in q:
        return "CDR sum of boxes"
    if "age" in q and "older" in q:
        return "age"
    return None


def _heuristic_group_concept(nl_question: str) -> str | None:
    """Extract group (predictor) concept from question text using keyword matching."""
    q = nl_question.lower()
    if "heart attack" in q:
        return "heart attack history"
    if "cardiac arrest" in q:
        return "cardiac arrest history"
    if "myocardial infarction" in q:
        return "myocardial infarction"
    if "hypertension" in q or "hyperten" in q:
        return "hypertension"
    if "diabetes" in q:
        return "diabetes"
    if "stroke" in q:
        return "stroke"
    if "sex" in q or "gender" in q:
        return "sex"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def repair_group_comparison_plan(
    plan: EnrichedAnalysisPlan,
    validation_errors: list[str],
    nl_question: str,
    catalog: list[CatalogEntry],
    client: Any = None,
    model: str | None = None,
) -> tuple[EnrichedAnalysisPlan | None, list[str]]:
    """One repair pass for group_comparison ops with missing variable IDs.

    Returns (repaired_plan, warnings) on success.
    Returns (None, [reason]) on failure — reason explains why repair was impossible.

    Only repairs ops where the failure is a missing group_variable_id or
    outcome_variable_id in a group_comparison operation.  All other errors in
    validation_errors are preserved (they will still cause is_executable=False
    after the caller re-validates).
    """
    # Find which errors are group_comparison missing-var errors
    gc_errors = [
        e for e in validation_errors
        if _is_group_comparison_missing_vars_error(e)
    ]
    if not gc_errors:
        return None, ["No group_comparison variable-ID errors to repair."]

    # Build map: output_name → [missing_params]
    op_repair_map: dict[str, list[str]] = {}
    for error in gc_errors:
        m = re.search(r"op '([^']+)' \(group_comparison\)", error)
        if not m:
            continue
        op_name = m.group(1)
        if op_name not in op_repair_map:
            op_repair_map[op_name] = []
        for p in _missing_params_from_error(error):
            if p not in op_repair_map[op_name]:
                op_repair_map[op_name].append(p)

    if not op_repair_map:
        return None, ["Could not extract operation names from validation error messages."]

    # Repair each group_comparison op that needs it
    repaired_operations: list[PlanOperation] = []
    warnings: list[str] = []

    for op in plan.operations:
        if op.op_type != "group_comparison" or op.output_name not in op_repair_map:
            repaired_operations.append(op)
            continue

        missing_params = op_repair_map[op.output_name]

        # Step 1: LLM concept extraction (one call covering all missing params)
        llm_concepts: dict[str, Any] | None = None
        if client is not None:
            llm_concepts = _llm_extract_concepts(
                nl_question, missing_params, op.output_name,
                catalog, client, model,
            )

        # Step 2: Resolve each missing variable
        new_params = dict(op.parameters)
        repair_failed = False

        for param in missing_params:
            if new_params.get(param):
                continue  # already present — edge case safety

            # Determine the concept to resolve
            llm_key = "outcome_concept" if param == "outcome_variable_id" else "group_concept"
            concept_phrase: str | None = (
                llm_concepts.get(llm_key) if llm_concepts else None
            )

            # Heuristic fallback when LLM didn't extract the concept
            if not concept_phrase:
                if param == "outcome_variable_id":
                    concept_phrase = _heuristic_outcome_concept(nl_question)
                else:
                    concept_phrase = _heuristic_group_concept(nl_question)

            if not concept_phrase:
                warnings.append(
                    f"Could not identify concept for '{param}' from question or LLM. "
                    "Abstaining from group comparison — provide clearer variable reference."
                )
                repair_failed = True
                break

            # Catalog resolution
            var_id, raw_code, proxy_warning = _resolve_concept_to_variable(
                concept_phrase, catalog
            )

            if not var_id:
                warnings.append(
                    f"Variable for '{concept_phrase}' (needed as {param}) "
                    "not found in catalog. "
                    "The group comparison cannot proceed without this variable."
                )
                repair_failed = True
                break

            if proxy_warning:
                warnings.append(proxy_warning)

            new_params[param] = var_id

        if repair_failed:
            return None, warnings

        # Build the repaired operation (PlanOperation is frozen — create new)
        repaired_operations.append(PlanOperation(
            op_type=op.op_type,
            parameters=new_params,
            output_name=op.output_name,
            description=op.description,
        ))

    # Build repaired plan (EnrichedAnalysisPlan is also frozen — create new)
    repaired_plan = EnrichedAnalysisPlan(
        original_question=plan.original_question,
        research_intent_summary=plan.research_intent_summary,
        researcher_audit_trace=list(plan.researcher_audit_trace) + [
            f"[repair] Resolved missing variable IDs for group_comparison "
            f"op(s): {sorted(op_repair_map.keys())}."
        ],
        operations=repaired_operations,
        requested_horizons=list(plan.requested_horizons),
        requires_backend_data=plan.requires_backend_data,
        baseline_definition=dict(plan.baseline_definition),
        outcome_definition=plan.outcome_definition,
        id_space=plan.id_space,
        original_plan=plan.original_plan,
        semantic_plan=plan.semantic_plan,
    )

    return repaired_plan, warnings
