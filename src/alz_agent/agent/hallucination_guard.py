"""Hallucination guard — verify every integer in the narrated answer is traceable.

The guard extracts all digit sequences from the answer text and checks that
each appears in the union of aggregate_refs, query_specs, and direct values
from all grounded ToolResults.  Extra whitelisted strings (e.g. the effective
horizon) can be passed via the ``extra`` parameter.

Empty-answer and zero-number cases always pass.
"""

from __future__ import annotations

import re
from typing import Any

from alz_agent.schemas import (
    DataEvidence,
    DescriptiveStatResult,
    GroupComparisonResult,
    RegressionResult,
    ToolResult,
)
from alz_agent.tools.deterministic import ModelingDesignResult, StatResult

try:
    from alz_agent.schemas import GeneratedAnalysisResult
except ImportError:
    GeneratedAnalysisResult = None  # type: ignore[assignment,misc]


def _check_hallucination(
    answer: str,
    tool_results: list[ToolResult],
    extra: set[str] | None = None,
) -> bool:
    """Return True iff every integer token in ``answer`` appears in tool evidence.

    Extracts all digit sequences from the narrated text and checks each is
    present in the union of aggregate_refs, query_specs, and direct values
    from all grounded ToolResults.  Pass ``extra`` to whitelist deterministic
    values (e.g. the effective horizon) that are not derived from any single
    tool result but are known to be correct.

    Empty-answer and zero-number cases always pass.
    """
    answer_ints: set[str] = set(re.findall(r"\b\d+\b", answer))
    if not answer_ints:
        return True

    evidence_nums = _collect_evidence_numbers(tool_results, extra=extra)
    missing = answer_ints - evidence_nums
    return len(missing) == 0


def _collect_evidence_numbers(
    tool_results: list[ToolResult],
    extra: set[str] | None = None,
) -> set[str]:
    """Collect all digit strings traceable to tool result values and evidence."""
    nums: set[str] = set()
    for tr in tool_results:
        if not tr.grounded:
            # abstain_reason is emitted deterministically by the tool/backend
            # (e.g. "Valid: ['1','2','3','4']", "below threshold (10)") — not
            # model generation — so its digits are legitimately traceable.
            if tr.abstain_reason:
                nums.update(re.findall(r"\d+", tr.abstain_reason))
            continue

        # Direct value integers
        val = tr.value
        if isinstance(val, int):
            nums.add(str(val))
        elif isinstance(val, dict):
            # Handles both flat dicts and fan-out dicts (dict[str, dict[str, int]])
            for v in val.values():
                if isinstance(v, int):
                    nums.add(str(v))
                elif isinstance(v, dict):
                    for deepest in v.values():
                        if isinstance(deepest, int):
                            nums.add(str(deepest))
        elif isinstance(val, StatResult):
            nums.add(str(val.n))
            # Float stat value: extract digit sequences (e.g. 72.35 → "72", "35")
            nums.update(re.findall(r"\d+", str(val.value)))

        elif isinstance(val, GroupComparisonResult):
            nums.add(str(val.n_total))
            if val.p_value is not None:
                nums.update(re.findall(r"\d+", str(val.p_value)))
            for g in val.groups:
                nums.add(str(g.n))
                nums.update(re.findall(r"\d+", str(g.mean)))
                nums.update(re.findall(r"\d+", str(g.sd)))
            # Caveats may include counts (e.g. "N participants excluded")
            for c in val.caveats:
                nums.update(re.findall(r"\d+", c))

        elif GeneratedAnalysisResult is not None and isinstance(val, GeneratedAnalysisResult):
            nums.add(str(val.n_analyzed))
            for v in val.results.values():
                nums.update(re.findall(r"\d+", str(v)))

        elif isinstance(val, ModelingDesignResult):
            if val.eligible_n is not None:
                nums.add(str(val.eligible_n))
            nums.add(str(val.horizon_years))
            nums.add(str(len(val.feature_whitelist)))
            nums.add(str(len(val.leakage_blacklist)))
            for v in val.feasibility.values():
                if isinstance(v, int):
                    nums.add(str(v))
            # Digit sequences within leakage record raw_codes (e.g. "ADAS13" → "13")
            # are narrated in the leakage section; include them so the guard passes.
            for rec in val.leakage_records:
                nums.update(re.findall(r"\d+", rec.raw_code))

        elif isinstance(val, DescriptiveStatResult):
            nums.add(str(val.n_total))
            nums.add(str(val.n_missing))
            nums.add(str(val.count))
            for attr in ("mean", "median", "sd", "q1", "q3", "iqr", "min_val", "max_val", "proportion"):
                v = getattr(val, attr, None)
                if v is not None:
                    nums.update(re.findall(r"\d+", str(v)))

        elif isinstance(val, RegressionResult):
            nums.add(str(val.n))
            if val.r_squared is not None:
                nums.update(re.findall(r"\d+", str(val.r_squared)))
            if val.pseudo_r2 is not None:
                nums.update(re.findall(r"\d+", str(val.pseudo_r2)))
            for coef in val.coefficients:
                nums.update(re.findall(r"\d+", str(coef.coefficient)))
                if coef.p_value is not None:
                    nums.update(re.findall(r"\d+", str(coef.p_value)))
            for c in val.caveats:
                nums.update(re.findall(r"\d+", c))

        # Evidence text mining
        for ev in tr.evidence:
            if isinstance(ev, DataEvidence):
                nums.update(re.findall(r"\d+", ev.aggregate_ref))
                nums.update(re.findall(r"\d+", str(ev.query_spec)))

    if extra:
        nums.update(extra)
    return nums


# ---------------------------------------------------------------------------
# Fan-out percentage derivation (for synthesis guard)
# ---------------------------------------------------------------------------

_SYNTHESIS_HORIZON_RE = re.compile(r"^\d+yr$")


def _compute_fanout_percentage_digits(tool_results: list[ToolResult]) -> set[str]:
    """Compute digit sequences that are valid percentages derived from fan-out + baseline N.

    When execution context has a fan-out longitudinal result (keys like "2yr", "3yr")
    and a baseline integer count, the synthesizer produces values like "54.1%".
    Those digits are fully deterministic (round(count / baseline_n * 100, 1)) and must
    not be flagged as hallucinations.  This function computes the digit sequences of
    every such percentage and returns them as an extra allowed set.

    Only operates when exactly one grounded integer result and at least one fan-out
    result are present; returns empty set otherwise.
    """
    baseline_n: int | None = None
    fanout_inner_counts: list[dict] = []

    for tr in tool_results:
        if not tr.grounded or tr.value is None:
            continue
        val = tr.value
        # Baseline: the first grounded positive integer (bool subclasses int, exclude it)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            if baseline_n is None:
                baseline_n = val
        elif isinstance(val, dict):
            for k, inner in val.items():
                if _SYNTHESIS_HORIZON_RE.match(k) and isinstance(inner, dict):
                    fanout_inner_counts.append(inner)

    if not baseline_n or not fanout_inner_counts:
        return set()

    allowed: set[str] = set()
    for counts in fanout_inner_counts:
        for cat_val in counts.values():
            if isinstance(cat_val, int) and not isinstance(cat_val, bool):
                pct = round(cat_val / baseline_n * 100, 1)
                allowed.update(re.findall(r"\d+", str(pct)))
    return allowed


# ---------------------------------------------------------------------------
# Synthesis hallucination guard (for SynthesisReport from enriched planner)
# ---------------------------------------------------------------------------

# Regex for codebook expressions: {VARNAME}={N} or {VARNAME}={N} (handles NACCUDSD=4 etc.)
_CODEBOOK_EXPR_RE = re.compile(
    r"\b[A-Z][A-Z0-9_]{2,}\s*[=><!]+\s*(\d+)",
    re.IGNORECASE,
)

# Regex for list/section numbering: digit(s) followed by a period or closing paren
# at a position that strongly suggests list ordering (start of line or after whitespace)
_LIST_NUMBER_RE = re.compile(
    r"(?:^|\s)(\d{1,2})[.)]\s",
    re.MULTILINE,
)

# Regex for suppression-threshold mentions: "threshold of N", "below N", "(N)"
_THRESHOLD_CONTEXT_RE = re.compile(
    r"threshold\s+(?:of\s+)?(\d+)|below\s+(?:reporting\s+)?threshold\s+\(?(\d+)",
    re.IGNORECASE,
)

# Sections that are never checked (contain interpretive/structural numbers, not counts)
_SKIP_SECTIONS = frozenset({"analysis_plan", "next_steps"})

# Matches thousands-separator commas: digit(s), comma, exactly 3 digits.
# Used to normalize "6,632" → "6632" before digit extraction.
_THOUSANDS_SEP_RE = re.compile(r"(\d{1,3}),(\d{3})(?!\d)")


def _collect_safe_numbers_from_text(
    text: str,
    requested_horizons: list[int] | None = None,
) -> set[str]:
    """Collect integers that are safe to appear in synthesis text without tracing to evidence.

    Safe contexts:
      - Codebook expressions: NACCUDSD=4, DIAGNOSIS=3, NACCALZD=1
      - List/section numbering: "1. ...", "2. ...", "3)"
      - Suppression threshold constants in context phrases
      - Explicitly requested horizons (e.g. [2, 3, 5])

    NOT safe: standalone participant counts (e.g. "47 participants").
    """
    safe: set[str] = set()

    # Codebook expression numbers
    for m in _CODEBOOK_EXPR_RE.finditer(text):
        safe.add(m.group(1))

    # List/section numbering
    for m in _LIST_NUMBER_RE.finditer(text):
        safe.add(m.group(1))

    # Suppression threshold mentions
    for m in _THRESHOLD_CONTEXT_RE.finditer(text):
        for g in m.groups():
            if g:
                safe.add(g)

    # Explicitly requested horizons
    for h in (requested_horizons or []):
        safe.add(str(h))

    return safe


# ---------------------------------------------------------------------------
# External literature claim guard
#
# Broader than the name suggests: covers every claim that requires supporting
# evidence the synthesizer was not actually given — external literature
# comparisons, statistical/modeling "adequacy" assessments (which require a
# power analysis or prior literature basis), and speculative biological/
# mechanistic explanations for an observed difference between groups (e.g.
# "may reflect more advanced cases"). All are gated the same way: prohibited
# when doc_evidence_count == 0, always allowed when real documentation
# evidence was supplied.
# ---------------------------------------------------------------------------

# Patterns matching external comparison / unsupported-adequacy / speculative-
# biological-explanation claims. Each uses [^.]* to stay within a single
# sentence (period = sentence boundary) where a co-occurring keyword is needed
# to confirm intent (e.g. "consistent with" + "literature"); a few phrases are
# specific enough on their own (e.g. "may reflect") to match without one.
_EXTERNAL_CLAIM_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\baligns?\s+with\b[^.]*?\b(?:literature|studies|research|prior|published|findings)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bconsistent\s+with\b[^.]*?\b(?:literature|studies|research|prior|published|findings)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btypical\s+(?:of|in|for)\b[^.]*?\b(?:literature|studies|alzheimer|late[- ]onset|onset)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bin\s+line\s+with\b[^.]*?\b(?:literature|studies|prior|findings|evidence)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcorroborates?\b[^.]*?\b(?:literature|studies|findings|evidence)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bpreviously\s+(?:reported|published|established|documented)\b", re.IGNORECASE),
    re.compile(
        r"\bas\s+expected\s+from\b[^.]*?\b(?:the\s+)?(?:literature|studies|prior)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwell[- ]established\b[^.]*?\b(?:range|pattern|norm|criterion|threshold|benchmark)\b",
        re.IGNORECASE,
    ),
    # Unsupported statistical/modeling adequacy assessments.
    re.compile(
        r"\b(?:adequate|sufficient)\b[^.]*?\bfor\s+modeling\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:adequate|sufficient)\s+sample\s+size\b",
        re.IGNORECASE,
    ),
    # Speculative biological/mechanistic explanations for a group difference —
    # specific enough to match standalone, no co-occurring keyword required.
    re.compile(r"\bmay\s+reflect\b", re.IGNORECASE),
    re.compile(r"\bsuggests?\s+(?:that\s+)?(?:the\s+)?underlying\b", re.IGNORECASE),
    re.compile(r"\breflects?\s+(?:more\s+)?(?:advanced|severe)\s+(?:cases|disease|pathology)\b", re.IGNORECASE),
]


def _check_external_claims(
    text: str,
    doc_evidence_count: int,
) -> tuple[bool, list[str]]:
    """Check whether text contains unsupported claims that require evidence not
    actually supplied: external literature comparisons, statistical/modeling
    adequacy assessments, or speculative biological/mechanistic explanations.

    Returns (passes, violations) where violations is a list of matched phrases
    (capped at 120 characters each for readability in correction prompts).

    Rule: when doc_evidence_count == 0, all such claims (aligns with literature,
    consistent with prior studies, previously reported, adequate for modeling,
    may reflect underlying pathology, etc.) are prohibited — there is no
    supplied evidence to support them. When doc_evidence_count > 0, always
    passes (the LLM has license to draw on the supplied documentation).

    Allowed replacement phrase when no doc evidence:
      "The estimate describes [what the number means].
       No external literature comparison was performed in this run."
    """
    if doc_evidence_count > 0:
        return True, []

    violations: list[str] = []
    for pattern in _EXTERNAL_CLAIM_PATTERNS:
        for m in pattern.finditer(text):
            violations.append(m.group(0)[:120].strip())

    return len(violations) == 0, violations


# Matches variable-identifier tokens the synthesizer might emit, e.g.
# "NACC.NACCAGE", "ADNI.MMSE.MMSCORE", "ADNI.DERIVED.age_at_event", or a
# fabricated "ADNI.OTHER.NACCAGE". Case-insensitive so lowercase field names match.
_SYNTHESIS_VAR_ID_RE = re.compile(r"\b(?:NACC|ADNI)\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")

# Sections skipped by the variable-identifier guard: analysis_plan copies the
# researcher_audit_trace verbatim (deterministic plan content, not model invention).
_VAR_ID_SKIP_SECTIONS = frozenset({"analysis_plan"})


def _collect_allowed_variable_ids(
    tool_results: list[ToolResult],
    catalog: list | None = None,
    plan: Any = None,
) -> set[str]:
    """Variable identifiers the synthesizer is allowed to reference (uppercased).

    A variable id is legitimate only if it traces to real grounding:
      - a catalog entry's variable_id,
      - a variable_id that appears in a grounded (or abstained) tool result's
        evidence / abstain reason,
      - a variable_id named by a plan operation,
      - the one supported Space B derived id.
    """
    allowed: set[str] = {"ADNI.DERIVED.AGE_AT_EVENT"}

    for e in (catalog or []):
        vid = getattr(e, "variable_id", None)
        if vid:
            allowed.add(vid.upper())

    if plan is not None:
        for op in getattr(plan, "operations", []) or []:
            vid = (op.parameters or {}).get("variable_id")
            if isinstance(vid, str) and vid:
                allowed.add(vid.upper())

    for tr in tool_results:
        texts: list[str] = []
        if not tr.grounded:
            if tr.abstain_reason:
                texts.append(tr.abstain_reason)
        else:
            for ev in tr.evidence:
                if isinstance(ev, DataEvidence):
                    texts.append(ev.aggregate_ref)
                    texts.append(str(ev.query_spec))
                    for attr in ("source_table", "source_field"):
                        v = getattr(ev, attr, None)
                        if v:
                            texts.append(str(v))
        for blob in texts:
            for m in _SYNTHESIS_VAR_ID_RE.finditer(blob):
                allowed.add(m.group(0).upper())

    return allowed


def _check_synthesis_variable_ids(
    report: "SynthesisReport",  # type: ignore[name-defined]
    catalog: list | None = None,
    plan: Any = None,
) -> tuple[bool, list[str]]:
    """Flag fabricated / cross-space variable identifiers in synthesized text.

    Every 'NACC.*' / 'ADNI.*' token in a checked section must appear in the
    allowed set (catalog, grounded evidence, plan ops, or the supported derived
    id). Catches inventions like 'ADNI.OTHER.NACCAGE' and cross-space leaks like a
    'NACC.*' id in a Space B report. Returns (passes, violations).
    """
    if report.execution_context is None:
        return True, []

    tool_results = list(report.execution_context.results.values())
    allowed = _collect_allowed_variable_ids(tool_results, catalog=catalog, plan=plan)

    violations: list[str] = []
    for section in report.sections:
        if section.section_id in _VAR_ID_SKIP_SECTIONS:
            continue
        for m in _SYNTHESIS_VAR_ID_RE.finditer(section.content):
            token = m.group(0)
            if token.upper() not in allowed:
                violations.append(token)

    return len(violations) == 0, violations


def _check_synthesis_hallucination(
    report: "SynthesisReport",  # type: ignore[name-defined]
    extra: set[str] | None = None,
) -> bool:
    """Hallucination guard for SynthesisReport sections.

    Checks that every participant count / numeric claim in checked sections traces
    back to values in report.execution_context.results.

    Sections SKIPPED (contain interpretive/structural content):
      analysis_plan, next_steps

    Safe number contexts (not flagged even if not in evidence):
      - Codebook expressions: e.g. "NACCUDSD=4", "NACCALZD=1"
      - List/section numbering: "1.", "2)", etc.
      - Suppression threshold mentions: "below threshold of 10"
      - Requested horizon values from the plan

    Returns True if all checked sections pass; False on first violation.
    """
    if report.execution_context is None:
        return True

    tool_results = list(report.execution_context.results.values())
    evidence_nums = _collect_evidence_numbers(tool_results, extra=extra)

    # Also allow digit sequences that are deterministically computable as percentages
    # from fan-out counts + baseline N (e.g. "54" from "54.1%").  These are grounded
    # values — they just aren't stored verbatim in any single aggregate_ref.
    evidence_nums |= _compute_fanout_percentage_digits(tool_results)

    # Derive requested horizons from plan if available
    requested_horizons: list[int] = []
    if report.execution_context:
        # Extract horizon keys from fan-out results (e.g. "2yr" → 2)
        for tr in tool_results:
            if tr.grounded and isinstance(tr.value, dict):
                for k in tr.value:
                    m = re.match(r"^(\d+)yr$", k)
                    if m:
                        requested_horizons.append(int(m.group(1)))

    for section in report.sections:
        if section.section_id in _SKIP_SECTIONS:
            continue

        text = section.content
        safe_in_section = _collect_safe_numbers_from_text(text, requested_horizons)

        # Normalize thousands separators before extracting digit tokens.
        # "6,632" → "6632" so the full number is checked rather than split sub-tokens
        # like "6" and "632" (which are too small to be uniquely traceable).
        text_norm = _THOUSANDS_SEP_RE.sub(r"\1\2", text)
        # Apply again for numbers with multiple commas (e.g. "1,234,567" → two passes)
        text_norm = _THOUSANDS_SEP_RE.sub(r"\1\2", text_norm)

        # All digit tokens in this (normalized) section
        candidate_ints = set(re.findall(r"\b\d+\b", text_norm))

        # Remove safe ones, then check the remainder against evidence
        unchecked = candidate_ints - safe_in_section
        untraceable = unchecked - evidence_nums

        if untraceable:
            return False

    return True
