"""P7 — Transport boundary between the cloud agent (P5) and the local execution
backend (P2/P3, runs on DCC where the real data lives).

Wire protocol (invariant 4 — subject-level data never leaves the local backend):
  cloud -> local : ToolCallRequest  (tool name + structured filter_spec/params only)
  local -> cloud : ToolResult       (aggregates + evidence only, suppression re-checked)

``execute_tool_call`` is the only function the local side exposes across the
boundary. It dispatches to the P3 deterministic tools (which already apply
suppression once), then re-applies an independent suppression check on the
result before it is allowed to cross — invariant 4's "AND re-checked at the
cloud/local transfer boundary." If a backend bug lets a sub-threshold count or
a row-level identifier slip through, this layer redacts it (numeric cells) or
raises ``BoundaryViolation`` (row-level data) rather than letting it cross.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel

from alz_agent.backend.interface import DataBackend
from alz_agent.constants import SUPPRESSION_THRESHOLD
from alz_agent.schemas import CatalogEntry, DataEvidence, Evidence, FilterSpec, ToolResult
from alz_agent.tools.deterministic import (
    ModelingDesignResult,
    VariableLookupResult,
    assess_longitudinal_feasibility,
    assess_modality_overlap,
    check_leakage,
    compute_cohort_size,
    lookup_variable,
    suggest_modeling_design,
)

logger = logging.getLogger(__name__)


class BoundaryViolation(RuntimeError):
    """Raised when a payload crossing the boundary contains row-level data.

    Unlike a sub-threshold numeric cell (which is redacted in place — the
    underlying request may still be answerable in aggregate), suspected
    row-level identifiers indicate a backend bug serious enough that the
    result must not cross at all.
    """


# ---------------------------------------------------------------------------
# ToolCallRequest — the only thing the cloud side sends
# ---------------------------------------------------------------------------


_TOOL_NAMES = (
    "lookup_variable",
    "compute_cohort_size",
    "assess_modality_overlap",
    "assess_longitudinal_feasibility",
    "check_leakage",
    "suggest_modeling_design",
)


class ToolCallRequest(BaseModel):
    """Structured tool call: a tool name plus the params that tool needs.

    Only FilterSpec and primitive params cross the wire — never a catalog or
    a backend handle. The local side supplies catalog/backend itself.
    """

    tool_name: Literal[
        "lookup_variable",
        "compute_cohort_size",
        "assess_modality_overlap",
        "assess_longitudinal_feasibility",
        "check_leakage",
        "suggest_modeling_design",
    ]
    filter_spec: FilterSpec | None = None
    horizon: int | None = None
    modalities: list[str] | None = None
    query: str | None = None
    feature_set: list[str] | None = None
    label_def: str | None = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Local-side dispatch (runs only on the machine holding the real data)
# ---------------------------------------------------------------------------


def _dispatch(
    request: ToolCallRequest,
    catalog: list[CatalogEntry],
    backend: DataBackend,
) -> ToolResult[Any]:
    """Call the requested P3 tool. Missing required params abstain (never raise)."""
    name = request.tool_name

    if name == "lookup_variable":
        if request.query is None:
            return ToolResult.abstain("lookup_variable requires 'query'")
        return lookup_variable(request.query, catalog)

    if name == "compute_cohort_size":
        if request.filter_spec is None:
            return ToolResult.abstain("compute_cohort_size requires 'filter_spec'")
        return compute_cohort_size(request.filter_spec, backend)

    if name == "assess_modality_overlap":
        if request.filter_spec is None or request.modalities is None:
            return ToolResult.abstain(
                "assess_modality_overlap requires 'filter_spec' and 'modalities'"
            )
        return assess_modality_overlap(request.modalities, request.filter_spec, backend, catalog)

    if name == "assess_longitudinal_feasibility":
        if request.filter_spec is None or request.horizon is None:
            return ToolResult.abstain(
                "assess_longitudinal_feasibility requires 'filter_spec' and 'horizon'"
            )
        return assess_longitudinal_feasibility(request.horizon, request.filter_spec, backend)

    if name == "check_leakage":
        if request.feature_set is None or request.label_def is None:
            return ToolResult.abstain("check_leakage requires 'feature_set' and 'label_def'")
        return check_leakage(request.feature_set, request.label_def, catalog)

    if name == "suggest_modeling_design":
        if request.filter_spec is None or request.horizon is None:
            return ToolResult.abstain(
                "suggest_modeling_design requires 'filter_spec' and 'horizon'"
            )
        return suggest_modeling_design(request.filter_spec, request.horizon, catalog, backend)

    return ToolResult.abstain(f"Unknown tool_name '{name}'")  # pragma: no cover — Literal-guarded


def execute_tool_call(
    request: ToolCallRequest,
    catalog: list[CatalogEntry],
    backend: DataBackend,
) -> ToolResult[Any]:
    """Run a ToolCallRequest locally and return a boundary-checked ToolResult.

    This is the only function the local side needs to expose to the cloud
    agent. The returned ToolResult has already passed the suppression
    re-check (_enforce_boundary) — it is safe to serialize and send.
    """
    raw_result = _dispatch(request, catalog, backend)
    return _enforce_boundary(raw_result)


# ---------------------------------------------------------------------------
# Suppression re-check guard (invariant 4: re-checked at the transfer boundary)
# ---------------------------------------------------------------------------

_SUBJECT_LEVEL_KEY_HINTS: frozenset[str] = frozenset({
    "naccid", "rid", "ptid", "subjid", "subject_id", "participant_id", "pid",
})

# Matches "key=123" style fragments inside human-readable aggregate_ref strings,
# e.g. "fixture.followup: sufficient=7, insufficient=40, censored=3".
_KV_PATTERN = re.compile(r"\b(\w+)\s*=\s*(\d+)\b")


def _is_unsuppressed_small_count(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and 0 < n < SUPPRESSION_THRESHOLD


def _check_and_redact_value(value: Any) -> tuple[Any, list[str]]:
    """Re-check `value` against SUPPRESSION_THRESHOLD; redact violations.

    Returns (redacted_value, notes). A top-level int violation redacts to
    None (the caller must then convert the whole ToolResult to an abstain,
    since ToolResult forbids grounded=True with value=None). Dict / nested
    model violations are redacted field-by-field, matching how the tools
    already represent suppression (None for a suppressed cell).
    """
    notes: list[str] = []

    if isinstance(value, dict):
        redacted: dict[str, Any] = dict(value)
        for k, v in value.items():
            if _is_unsuppressed_small_count(v):
                redacted[k] = None
                notes.append(
                    f"dict key {k!r}={v} violated suppression threshold "
                    f"({SUPPRESSION_THRESHOLD}); redacted to None"
                )
        return redacted, notes

    if isinstance(value, VariableLookupResult):
        cov = value.coverage
        if cov is not None and (
            _is_unsuppressed_small_count(cov.n_available)
            or _is_unsuppressed_small_count(cov.longitudinal_availability)
        ):
            notes.append(
                "VariableLookupResult.coverage redacted: sub-threshold cell(s) "
                f"(n_available={cov.n_available}, "
                f"longitudinal_availability={cov.longitudinal_availability})"
            )
            return value.model_copy(update={"coverage": None}), notes
        return value, notes

    if isinstance(value, ModelingDesignResult):
        updates: dict[str, Any] = {}
        if _is_unsuppressed_small_count(value.eligible_n):
            updates["eligible_n"] = None
            notes.append(
                f"ModelingDesignResult.eligible_n={value.eligible_n} violated "
                f"suppression threshold ({SUPPRESSION_THRESHOLD}); redacted to None"
            )
        if value.feasibility:
            new_feas = dict(value.feasibility)
            for k, v in value.feasibility.items():
                if _is_unsuppressed_small_count(v):
                    new_feas[k] = None
                    notes.append(
                        f"ModelingDesignResult.feasibility[{k!r}]={v} violated "
                        f"suppression threshold ({SUPPRESSION_THRESHOLD}); redacted to None"
                    )
            if new_feas != value.feasibility:
                updates["feasibility"] = new_feas
        if updates:
            return value.model_copy(update=updates), notes
        return value, notes

    if _is_unsuppressed_small_count(value):
        notes.append(
            f"top-level int value={value} violated suppression threshold "
            f"({SUPPRESSION_THRESHOLD}); redacted"
        )
        return None, notes

    return value, notes


def _collect_suppressed_keys(value: Any) -> set[str]:
    """Field names that are (now) suppressed in `value` — for evidence-text redaction.

    A backend may correctly suppress a cell in the structured value (set it to
    None) while its evidence aggregate_ref string still names that same key
    with the raw pre-suppression number — that text is the leak this catches.
    """
    keys: set[str] = set()
    if isinstance(value, dict):
        keys |= {k for k, v in value.items() if v is None}
    if isinstance(value, ModelingDesignResult):
        if value.eligible_n is None:
            keys |= {"eligible_n", "count_eligible"}
        keys |= {k for k, v in (value.feasibility or {}).items() if v is None}
    return keys


def _redact_text(text: str, suppressed_keys: set[str]) -> tuple[str, bool]:
    if not suppressed_keys:
        return text, False
    changed = False

    def _sub(match: re.Match[str]) -> str:
        nonlocal changed
        key = match.group(1)
        if key in suppressed_keys:
            changed = True
            return f"{key}=[REDACTED]"
        return match.group(0)

    return _KV_PATTERN.sub(_sub, text), changed


def _redact_evidence(
    evidence: list[Evidence],
    suppressed_keys: set[str],
) -> tuple[list[Evidence], list[str]]:
    notes: list[str] = []
    if not suppressed_keys:
        return list(evidence), notes

    redacted: list[Evidence] = []
    for ev in evidence:
        if isinstance(ev, DataEvidence):
            new_ref, changed = _redact_text(ev.aggregate_ref, suppressed_keys)
            if changed:
                notes.append(
                    f"evidence.aggregate_ref redacted for suppressed key(s) "
                    f"{sorted(suppressed_keys)}: {ev.aggregate_ref!r} -> {new_ref!r}"
                )
                ev = ev.model_copy(update={"aggregate_ref": new_ref})
        redacted.append(ev)
    return redacted, notes


def _check_no_row_level_data(evidence: list[Evidence]) -> list[str]:
    """Return violation messages if any evidence payload looks like row-level data."""
    violations: list[str] = []
    for ev in evidence:
        if not isinstance(ev, DataEvidence):
            continue
        blob = (json.dumps(ev.query_spec, default=str) + " " + ev.aggregate_ref).lower()
        for hint in _SUBJECT_LEVEL_KEY_HINTS:
            if hint in blob:
                violations.append(
                    f"possible row-level identifier key {hint!r} found in evidence "
                    f"(aggregate_ref={ev.aggregate_ref!r})"
                )
    return violations


def _enforce_boundary(result: ToolResult[Any]) -> ToolResult[Any]:
    """Independently re-check a ToolResult before it is allowed to cross.

    Abstain results carry no value/evidence (per the ToolResult contract) and
    pass through untouched. Grounded results are checked and, if necessary,
    redacted (numeric suppression) or rejected (row-level data).
    """
    if not result.grounded:
        return result

    violations = _check_no_row_level_data(result.evidence)
    if violations:
        raise BoundaryViolation(
            "Refusing to cross boundary — row-level data suspected: "
            + "; ".join(violations)
        )

    redacted_value, value_notes = _check_and_redact_value(result.value)
    suppressed_keys = _collect_suppressed_keys(redacted_value)
    redacted_evidence, evidence_notes = _redact_evidence(result.evidence, suppressed_keys)

    notes = value_notes + evidence_notes
    if notes:
        logger.warning("Boundary suppression re-check redacted payload: %s", "; ".join(notes))

    if redacted_value is None and result.value is not None:
        # Whole-value redaction (top-level int) — ToolResult forbids
        # grounded=True with value=None, so this must become an abstain.
        return ToolResult.abstain(
            "boundary suppression re-check redacted the result: " + "; ".join(notes)
        )

    return ToolResult(grounded=True, value=redacted_value, evidence=redacted_evidence)


# ---------------------------------------------------------------------------
# Serialization — JSON-safe envelopes for the actual wire transfer
# ---------------------------------------------------------------------------


def serialize_request(request: ToolCallRequest) -> dict[str, Any]:
    return json.loads(request.model_dump_json())


def deserialize_request(data: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest.model_validate(data)


def serialize_tool_result(result: ToolResult[Any]) -> dict[str, Any]:
    """Serialize a ToolResult to a JSON-safe dict — the only thing sent to the cloud.

    Call this only on a result that has already passed _enforce_boundary
    (i.e., came back from execute_tool_call).
    """
    return json.loads(result.model_dump_json())


def deserialize_tool_result(data: dict[str, Any]) -> ToolResult[Any]:
    """Reconstruct a ToolResult on the cloud side.

    Reconstructed as ToolResult[Any]: the cloud side narrates aggregate
    values and evidence — it has no need (and is not entitled) to reconstruct
    rich local-only types.
    """
    return ToolResult[Any].model_validate(data)
