"""Catalog context builder for LLM variable resolution.

Builds a compact text table of candidate catalog variables, scored by
relevance to the NL query.  The context is passed to the LLM so it can
map researcher phrases to raw catalog column names.

Always includes high-priority reference variables (NACCUDSD, NACCAGE, …)
and caps the output at max_entries to fit within model context limits.
"""

from __future__ import annotations

import re

from alz_agent.schemas import CatalogEntry

# Tokens so common they never discriminate between catalog entries.
_CONTEXT_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "are", "have", "from",
    "among", "between", "within", "does", "what", "how", "any", "all",
    "participants", "participant", "patient", "patients", "study",
})

# Always include these reference variables if present in catalog.
_ALWAYS_INCLUDE_CODES = frozenset({
    "NACCUDSD", "NACCALZD", "NACCID", "NACCAGE", "NACCMMSE",
    "NACCVNUM", "SEX", "EDUC", "NACCETPR",
})


def _build_catalog_context(
    nl_query: str,
    catalog: list[CatalogEntry],
    max_entries: int = 70,
) -> str:
    """Return a compact text table of candidate catalog variables.

    Strategy:
    - Score each entry: count NL query tokens appearing in
      raw_code + concept + description (lowercased).
    - Sort by score desc; always include reference variables.
    - Cap at max_entries (70 fits comfortably in Haiku's context).
    - If the catalog is small enough (≤ max_entries), return everything.
    """
    if not catalog:
        return "(catalog is empty — LLM must resolve variables from background knowledge only)"

    if len(catalog) <= max_entries:
        selected = list(catalog)
    else:
        # Include 2-char tokens that are meaningful in this domain (ad, mr, mr).
        # The base regex uses {2,} so "ad", "mr" etc. score correctly.
        _SHORT_DOMAIN_TOKENS = frozenset({"ad", "mr"})
        nl_tokens = (
            set(re.findall(r"\b\w{3,}\b", nl_query.lower())) - _CONTEXT_STOPWORDS
            | (set(re.findall(r"\b\w{2}\b", nl_query.lower())) & _SHORT_DOMAIN_TOKENS)
        )

        scored: list[tuple[int, CatalogEntry]] = []
        for entry in catalog:
            haystack = (
                f"{entry.raw_code} {entry.concept} {entry.description} "
                f"{entry.family or ''} {entry.variant or ''}"
            ).lower()
            score = sum(1 for t in nl_tokens if t in haystack)
            scored.append((score, entry))

        scored.sort(key=lambda x: (-x[0], x[1].raw_code))

        always = [e for _, e in scored if e.raw_code in _ALWAYS_INCLUDE_CODES]
        always_codes = {e.raw_code for e in always}
        rest = [e for _, e in scored if e.raw_code not in always_codes]

        selected = always + rest[: max(0, max_entries - len(always))]

    # Both variable_id (fully-qualified, e.g. ADNI.<TABLE>.<FIELD>) and raw_code (the bare
    # column name) are shown. Space A (NACC) columns are inherently unambiguous — one
    # raw_code per column — so consumers that only need a bare code (e.g. the classic
    # AnalysisPlan/fill_intent path, whose leakage_validation.py catalog index is keyed by
    # raw_code) may use that column. Space B (ADNI) fields are NOT inherently unambiguous —
    # the same field name can exist in multiple tables (e.g. "pTau181_RAW" in both
    # ADSP_PHC_CSF and ADSP_PHC_PLASMA) — so any consumer resolving a Space B variable MUST
    # copy variable_id verbatim; raw_code alone cannot disambiguate the table.
    lines = [
        "variable_id | raw_code | concept | value_coding | leakage_tag | family | variant | description"
    ]
    for entry in selected:
        vc_parts = list(entry.value_coding.items())[:5]
        vc_str = " ".join(f"{k}={v}" for k, v in vc_parts) if vc_parts else ""
        tags = (
            ",".join(t.value for t in entry.leakage_tags)
            if entry.leakage_tags
            else "NONE"
        )
        desc = (entry.description or "")[:100]
        family = entry.family or ""
        variant = entry.variant or ""
        lines.append(
            f"{entry.variable_id} | {entry.raw_code} | {entry.concept} | {vc_str} | {tags} | "
            f"{family} | {variant} | {desc}"
        )

    derived_note = (
        "\n\nSUPPORTED DERIVED VARIABLES (Space B / id_space='B' only):\n"
        "  ADNI.DERIVED.age_at_event — participant age (years) at an event-anchored "
        "visit, computed from PTDEMOG.PTDOB (month/year) and the anchored DXSUM.EXAMDATE. "
        "Use this (NOT NACC.NACCAGE, which is Space A) for 'age at first AD diagnosis' in "
        "ADNI, with event_anchor='first_diagnosis' and filter {\"event_etiology\": \"ad\"}. "
        "Do NOT invent other ADNI derived ids; only this one is supported."
    )

    note = (
        "\n\nNOTE: entries with a non-empty 'family' value are part of a variant "
        "group (e.g. family='ptau217' covers both plasma raw and harmonized "
        "columns). When a question asks what variants/kinds of a concept exist "
        "(e.g. 'what PTAU data do we have'), use op_type=explore_catalog to see "
        "every match, then op_type=describe_variable_family with the family "
        "value you observed to get per-variant subject coverage — never invent "
        "a family value that isn't shown here."
    )
    return "\n".join(lines) + derived_note + note
