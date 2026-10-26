"""P5 — Deterministic structured-explanation layer.

Converts raw ToolResult lists into an ExplanationContext whose items carry
display_name, definition, computation_note, and section assignments.  No LLM
calls are made here — the context is consumed by the LLM narration layer and
the template fallback.

Invariant: no number in any ExplainableItem.definition or .computation_note
may be narrated by the LLM unless it also appears in the ToolResult value /
evidence set (enforced by the hallucination guard).  Definitions are metadata
for LLM guidance, not for verbatim output.
"""

from __future__ import annotations

import re
from typing import Any

from alz_agent.schemas import (
    CatalogEntry,
    ExplainableItem,
    ExplanationContext,
    ExplanationMetadata,
    ExplanationSection,
    FilterSpec,
    LeakageRecord,
    StructuredCaveat,
    ToolResult,
)
from alz_agent.tools.deterministic import ModelingDesignResult, StatResult


# ---------------------------------------------------------------------------
# Section ordering (fixed)
# ---------------------------------------------------------------------------

_SECTION_ORDER: list[tuple[str, str]] = [
    ("parsed_target",       "Parsed question / modeling target"),
    ("cohort",              "Cohort definition"),
    ("outcome_feasibility", "Outcome and follow-up feasibility"),
    ("modality",            "Modality availability"),
    ("feature_leakage",     "Feature eligibility and leakage policy"),
    ("recommended_design",  "Recommended modeling design"),
    ("caveats",             "Caveats and limitations"),
    ("metadata",            "Metadata (audit)"),
]

_SECTION_TITLES: dict[str, str] = dict(_SECTION_ORDER)


# ---------------------------------------------------------------------------
# Friendly modality label mapping
# ---------------------------------------------------------------------------

_MODALITY_FRIENDLY: dict[str, str] = {
    "cognition":      "cognition / neuropsychological testing",
    "mri":            "MRI neuroimaging",
    "pet":            "PET imaging",
    "fluid_biomarker": "fluid biomarker (CSF/plasma)",
    "vascular":       "vascular risk factors",
    "neuropath":      "neuropathology",
    "demographic":    "demographics",
}


def _friendly_modality(key: str) -> str:
    parts = [k.strip() for k in key.split(",")]
    return " + ".join(_MODALITY_FRIENDLY.get(p, p) for p in parts)


# ---------------------------------------------------------------------------
# Cohort label helper
# ---------------------------------------------------------------------------

def _cohort_label(filter_spec: FilterSpec | None) -> str:
    if filter_spec is None:
        return "eligible"
    parts: list[str] = []
    if filter_spec.baseline_diagnosis:
        parts.extend(d.upper() for d in filter_spec.baseline_diagnosis)
    if filter_spec.baseline_etiology:
        parts.extend(f"AD-etiology-at-baseline ({e})" for e in filter_spec.baseline_etiology)
    if filter_spec.event_etiology:
        parts.extend(f"AD-etiology-ever ({e})" for e in filter_spec.event_etiology)
    return ", ".join(parts) if parts else "eligible"


# ---------------------------------------------------------------------------
# Per-tool normalizers
# ---------------------------------------------------------------------------


def _explain_modeling_design(
    result: ModelingDesignResult,
    filter_spec: FilterSpec | None,
) -> list[ExplainableItem]:
    items: list[ExplainableItem] = []
    cohort_lbl = _cohort_label(filter_spec)

    # Eligible N
    if result.eligible_n is not None:
        items.append(ExplainableItem(
            item_id="eligible_n",
            display_name="Eligible participants",
            raw_key="eligible_n",
            value=result.eligible_n,
            unit="participants",
            source_tool="suggest_modeling_design",
            filter_scope=cohort_lbl,
            definition=(
                "Participants meeting all cohort criteria at their index visit "
                "(baseline diagnosis filter applied)."
            ),
            importance="high",
            explanation_required=True,
            section_id="cohort",
        ))

    # Prediction horizon (in metadata + feasibility section)
    items.append(ExplainableItem(
        item_id="horizon_years",
        display_name="Prediction horizon",
        raw_key="horizon_years",
        value=result.horizon_years,
        unit="years",
        source_tool="suggest_modeling_design",
        definition=(
            "The time window after the index visit within which outcome events "
            "are counted for label assignment."
        ),
        importance="high",
        explanation_required=True,
        section_id="outcome_feasibility",
    ))

    # Outcome label definition
    items.append(ExplainableItem(
        item_id="outcome_label_def",
        display_name="Outcome label definition",
        raw_key="outcome_label_def",
        value=result.outcome_label_def,
        source_tool="suggest_modeling_design",
        definition="How the prediction target is operationally defined from the data.",
        importance="high",
        explanation_required=True,
        section_id="recommended_design",
    ))

    # Index visit definition
    items.append(ExplainableItem(
        item_id="index_visit_def",
        display_name="Index visit definition",
        raw_key="index_visit_def",
        value=result.index_visit_def,
        source_tool="suggest_modeling_design",
        definition="The visit used as time-zero for prediction (baseline).",
        importance="medium",
        section_id="recommended_design",
    ))

    # Feasibility breakdown
    for key in ("sufficient", "insufficient", "censored", "total_eligible"):
        v = result.feasibility.get(key)
        if v is None:
            continue
        definitions = {
            "sufficient": (
                "Participants who have adequate follow-up duration to observe "
                "the outcome within the prediction horizon."
            ),
            "insufficient": (
                "Participants whose follow-up window is shorter than the prediction "
                "horizon, so outcome status cannot be determined."
            ),
            "censored": (
                "Participants with only a single recorded visit; no follow-up is "
                "available so conversion status cannot be determined for any horizon. "
                "This count is horizon-independent: the same participants are censored "
                "regardless of the horizon length."
            ),
            "total_eligible": (
                "Total participants entering the follow-up feasibility calculation."
            ),
        }
        total = result.feasibility.get("total_eligible")
        items.append(ExplainableItem(
            item_id=f"feasibility_{key}",
            display_name=key.replace("_", " ").capitalize(),
            raw_key=key,
            value=v,
            unit="participants",
            source_tool="suggest_modeling_design",
            denominator=total if key != "total_eligible" else None,
            definition=definitions.get(key, ""),
            importance="high" if key in ("sufficient", "censored", "insufficient") else "medium",
            explanation_required=key in ("sufficient", "censored", "insufficient"),
            section_id="outcome_feasibility",
        ))

    # Feature whitelist (count + policy)
    wl_count = len(result.feature_whitelist)
    items.append(ExplainableItem(
        item_id="feature_whitelist_count",
        display_name="Eligible candidate predictors",
        raw_key="feature_whitelist",
        value=wl_count,
        unit="variables",
        source_tool="suggest_modeling_design",
        definition=(
            "Variables from the catalog allowed as candidate predictors. "
            "Pre-index only — collected before the index visit. "
            "Leakage variables are excluded. "
            "This is not a final feature set: missingness checks, encoding, "
            "and model-specific preprocessing are still required."
        ),
        evidence=[f"whitelist_variable_ids: {len(result.feature_whitelist)} items"],
        computation_note=(
            "Count = total catalog variables for this id_space minus leakage-blacklisted variables."
        ),
        importance="high",
        explanation_required=True,
        section_id="feature_leakage",
    ))

    # Leakage blacklist count + per-variable records
    bl_count = len(result.leakage_blacklist)
    items.append(ExplainableItem(
        item_id="leakage_blacklist_count",
        display_name="Leakage blacklist (total excluded)",
        raw_key="leakage_blacklist_count",
        value=bl_count,
        unit="variables",
        source_tool="suggest_modeling_design",
        definition=(
            "Variables excluded from the candidate predictor set due to leakage risk: "
            "they either directly define the outcome, are collinear with it, or were "
            "fit on cohort data that includes outcome-period observations."
        ),
        importance="high",
        explanation_required=True,
        section_id="feature_leakage",
    ))

    # Per-variable leakage records
    for rec in result.leakage_records:
        items.append(ExplainableItem(
            item_id=f"leakage_var_{rec.variable_id}",
            display_name=f"Blacklisted: {rec.display_name} ({rec.raw_code})",
            raw_key=rec.variable_id,
            value=rec.raw_code,
            source_tool="suggest_modeling_design",
            dataset=rec.source_dataset,
            definition=rec.description[:200] if rec.description else f"Variable {rec.raw_code}.",
            computation_note=(
                f"Tag: {rec.tag}. "
                f"Reason: {rec.reason}. "
                f"Excluded from: {rec.excluded_from}."
            ),
            importance="high",
            explanation_required=True,
            section_id="feature_leakage",
        ))

    return items


def _explain_feasibility_standalone(
    result: dict[str, Any],
    horizon: int,
    filter_spec: FilterSpec | None,
) -> list[ExplainableItem]:
    """For standalone assess_longitudinal_feasibility results (not inside ModelingDesignResult)."""
    items: list[ExplainableItem] = []
    total = result.get("total_eligible")
    definitions = {
        "sufficient": "Participants with enough follow-up to observe the outcome within the horizon.",
        "insufficient": "Participants with follow-up shorter than the prediction horizon.",
        "censored": (
            "Participants with only a single recorded visit; no follow-up is available "
            "so conversion status cannot be determined for any horizon. "
            "This count is horizon-independent: the same participants are censored "
            "regardless of the specified horizon length."
        ),
        "total_eligible": "Total participants entering the feasibility calculation.",
    }
    _censored_caveats = [
        "Censored count is horizon-independent (single-visit participants). "
        "Identical censored counts across horizons are expected behavior, not a bug."
    ]
    _censored_note = "Backend logic: n_cen = (grp_sz == 1).sum() — participants with exactly 1 visit."
    for key, v in result.items():
        if v is None:
            continue
        extra: dict = {}
        if key == "censored":
            extra["computation_note"] = _censored_note
            extra["caveats"] = _censored_caveats
        items.append(ExplainableItem(
            item_id=f"feas_{key}",
            display_name=key.replace("_", " ").capitalize(),
            raw_key=key,
            value=v,
            unit="participants",
            source_tool="assess_longitudinal_feasibility",
            denominator=total if key != "total_eligible" else None,
            definition=definitions.get(key, ""),
            importance="high" if key in ("sufficient", "censored", "insufficient") else "medium",
            explanation_required=key in ("sufficient", "censored", "insufficient"),
            section_id="outcome_feasibility",
            **extra,
        ))
    return items


def _explain_cohort_size(value: int, filter_spec: FilterSpec | None) -> ExplainableItem:
    cohort_lbl = _cohort_label(filter_spec)
    return ExplainableItem(
        item_id="cohort_size",
        display_name="Cohort size",
        raw_key="cohort_size",
        value=value,
        unit="participants",
        source_tool="compute_cohort_size",
        filter_scope=cohort_lbl,
        definition=(
            "Total number of unique participants meeting the baseline cohort filter."
        ),
        importance="high",
        explanation_required=True,
        section_id="cohort",
    )


def _explain_modality(
    result: dict[str, Any],
    filter_spec: FilterSpec | None,
) -> list[ExplainableItem]:
    cohort_lbl = _cohort_label(filter_spec)
    items: list[ExplainableItem] = []
    for key, v in sorted(result.items()):
        friendly = _friendly_modality(key)
        if "," in key:
            definition = (
                f"Number of {cohort_lbl} participants with usable data for "
                f"ALL of the following modalities: {friendly}. "
                "Based on the current modality map/catalog."
            )
            section = "modality"
        else:
            definition = (
                f"Number of {cohort_lbl} participants with usable "
                f"{friendly} data, based on the current modality map/catalog."
            )
            section = "modality"

        if v is None:
            items.append(ExplainableItem(
                item_id=f"modality_{key.replace(',', '_')}",
                display_name=f"{friendly} (suppressed)",
                raw_key=key,
                value=None,
                unit="participants",
                source_tool="assess_modality_overlap",
                definition=definition,
                computation_note="Count suppressed below reporting threshold.",
                importance="low",
                section_id=section,
            ))
        else:
            items.append(ExplainableItem(
                item_id=f"modality_{key.replace(',', '_')}",
                display_name=friendly,
                raw_key=key,
                value=v,
                unit="participants",
                source_tool="assess_modality_overlap",
                filter_scope=cohort_lbl,
                definition=definition,
                importance="high" if "," not in key else "medium",
                explanation_required=True,
                section_id=section,
            ))
    return items


def _explain_not_answerable(tr: ToolResult, tool_index: int) -> ExplainableItem:
    reason = tr.abstain_reason or "unknown reason"
    # Try to extract what was requested from the abstain reason
    what_requested = reason
    why_abstained = reason
    next_step = "Verify that the requested modality or variable exists in the current catalog/backend."

    # Pattern-match known abstain reasons for richer explanations
    if "modali" in reason.lower() and "absent" in reason.lower():
        what_requested = "Modality overlap count"
        next_step = (
            "Check that the modality is present in the ADSP-PHC release and the catalog. "
            "If it is not in the current backend, this is a known gap — not a data absence."
        )
    elif "suppress" in reason.lower():
        what_requested = "Participant count"
        why_abstained = "Count is below the suppression threshold and cannot be reported."
        next_step = "Use a broader cohort filter to increase the cell count above the threshold."
    elif "not found in catalog" in reason.lower():
        what_requested = "Variable / concept lookup"
        next_step = "Check the catalog for the correct variable name or raw_code."
    elif "demographic" in reason.lower():
        what_requested = "Demographic modality count"
        why_abstained = (
            "The current ADSP-PHC backend/modality map does not expose 'demographic' as a "
            "countable modality."
        )
        next_step = (
            "This does not mean no demographic variables exist — they may be available as "
            "individual variables (e.g. NACCAGE, SEX, EDUC) in the NACC dataset. "
            "Query those variables individually via a lookup or cohort-stat tool."
        )

    return ExplainableItem(
        item_id=f"not_answerable_{tool_index}",
        display_name=f"Not answerable: {what_requested[:80]}",
        raw_key=f"abstain_{tool_index}",
        value=None,
        source_tool="(abstained)",
        definition=why_abstained,
        computation_note=f"Raw reason: {reason}",
        caveats=[next_step],
        importance="high",
        explanation_required=True,
        section_id="caveats",
        is_not_answerable=True,
    )


# ---------------------------------------------------------------------------
# Structured caveat builder
# ---------------------------------------------------------------------------


def _build_structured_caveats(
    target_etiology: list[str] | None,
    outcome_label_def: str | None,
    catalog: list[CatalogEntry] | None = None,
) -> list[StructuredCaveat]:
    caveats: list[StructuredCaveat] = []
    if target_etiology and "ad" in target_etiology:
        # Check if a confirmed neuropath AD variable exists in catalog
        neuropath_available = False
        if catalog:
            neuropath_available = any(
                "neuropath" in e.modality.value.lower() and "amyloid" in (e.description + e.concept).lower()
                for e in catalog
            )
        caveats.append(StructuredCaveat(
            parsed_target_etiology="AD-specific progression (target etiology: AD)",
            actual_endpoint=(
                outcome_label_def or
                "All-cause dementia conversion (NACCUDSD=4) — not AD-etiologically confirmed"
            ),
            missing_concept=(
                "Etiologic AD attribution at the outcome visit. "
                "NACCALZD==1 is available at BASELINE (index visit) but not separately "
                "at the conversion visit without additional longitudinal joins. "
                + ("Neuropathological confirmation is available in catalog." if neuropath_available else
                   "Neuropathological confirmation is not available in the current catalog.")
            ),
            implication=(
                "The feasibility estimates may overstate AD-specific conversion feasibility, "
                "since the counted endpoint is all-cause dementia, which includes non-AD dementias."
            ),
            required_next_step=(
                "To narrow to AD-specific conversion: add NACCALZD==1 filter at the conversion "
                "visit if longitudinally available, or use a neuropathologically-confirmed "
                "AD endpoint if neuropathology data is present in the cohort."
            ),
        ))
    return caveats


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _deduplicate_items(items: list[ExplainableItem]) -> list[ExplainableItem]:
    """Remove exact duplicates by (display_name, value) and suppress inferior items.

    When two items share the same (value, section_id) and semantically equivalent
    display_names (e.g. 'Eligible participants' from cohort_size vs from
    suggest_modeling_design), keep the one with higher importance or richer definition.
    """
    seen: dict[tuple[Any, Any], int] = {}  # key -> index of kept item
    result: list[ExplainableItem] = []

    for item in items:
        dedup_key = (item.display_name.lower().strip(), item.value)
        if dedup_key in seen:
            existing_idx = seen[dedup_key]
            existing = result[existing_idx]
            # Keep the richer one (more definition, higher importance)
            importance_rank = {"high": 2, "medium": 1, "low": 0}
            if importance_rank.get(item.importance, 0) > importance_rank.get(existing.importance, 0):
                result[existing_idx] = item
                seen[dedup_key] = existing_idx
            elif len(item.definition) > len(existing.definition) and item.importance == existing.importance:
                result[existing_idx] = item
        else:
            seen[dedup_key] = len(result)
            result.append(item)

    return result


# ---------------------------------------------------------------------------
# Section assembly
# ---------------------------------------------------------------------------


def _assemble_sections(items: list[ExplainableItem]) -> list[ExplanationSection]:
    """Group items into ordered sections, emitting only non-empty sections."""
    by_section: dict[str, list[str]] = {sid: [] for sid, _ in _SECTION_ORDER}
    for item in items:
        sid = item.section_id
        if sid not in by_section:
            by_section.setdefault(sid, []).append(item.item_id)
        else:
            by_section[sid].append(item.item_id)

    sections: list[ExplanationSection] = []
    for sid, title in _SECTION_ORDER:
        ids = by_section.get(sid, [])
        if ids or sid == "metadata":  # always include metadata
            sections.append(ExplanationSection(
                section_id=sid,
                title=_SECTION_TITLES.get(sid, title),
                item_ids=ids,
            ))
    return sections


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_explanation_context(
    tool_results: list[ToolResult],
    *,
    filter_spec: FilterSpec | None = None,
    horizon: int,
    horizon_source: str,          # "user" | "default"
    default_horizon: int = 3,
    catalog: list[CatalogEntry] | None = None,
    target_etiology: list[str] | None = None,
    backend_name: str | None = None,
    catalog_release: str | None = None,
) -> ExplanationContext:
    """Build a structured ExplanationContext from tool results — deterministic, no LLM calls."""
    from alz_agent.tools.deterministic import ModelingDesignResult, VariableLookupResult  # avoid circular at module level

    all_items: list[ExplainableItem] = []
    not_answerable: list[ExplainableItem] = []
    tools_invoked: list[str] = []
    outcome_label_def: str | None = None

    for i, tr in enumerate(tool_results):
        if not tr.grounded:
            na_item = _explain_not_answerable(tr, i)
            all_items.append(na_item)
            not_answerable.append(na_item)
            tools_invoked.append("(abstained)")
            continue

        val = tr.value

        if isinstance(val, int):
            # compute_cohort_size
            tools_invoked.append("compute_cohort_size")
            all_items.append(_explain_cohort_size(val, filter_spec))

        elif isinstance(val, ModelingDesignResult):
            tools_invoked.append("suggest_modeling_design")
            outcome_label_def = val.outcome_label_def
            all_items.extend(_explain_modeling_design(val, filter_spec))

        elif isinstance(val, dict):
            # Distinguish feasibility vs modality by key presence
            feas_keys = {"sufficient", "insufficient", "censored", "total_eligible"}
            if val and any(k in val for k in feas_keys):
                tools_invoked.append("assess_longitudinal_feasibility")
                all_items.extend(_explain_feasibility_standalone(val, horizon, filter_spec))
            else:
                tools_invoked.append("assess_modality_overlap")
                all_items.extend(_explain_modality(val, filter_spec))

        elif isinstance(val, VariableLookupResult):
            tools_invoked.append("lookup_variable")
            leakage_note = (
                f"Leakage tags: {', '.join(val.leakage_tags)}"
                if val.leakage_tags and val.leakage_tags != ["NONE"]
                else None
            )
            all_items.append(ExplainableItem(
                item_id=f"lookup_{val.variable_id}",
                display_name=f"Variable: {val.concept}",
                raw_key=val.variable_id,
                value=val.raw_code,
                unit=None,
                source_tool="lookup_variable",
                dataset=val.source_dataset,
                definition=(
                    f"{val.concept} — raw column: {val.raw_code}, "
                    f"dataset: {val.source_dataset}, modality: {val.modality}"
                ),
                computation_note=leakage_note,
                importance="high",
                explanation_required=True,
                section_id="parsed_target",
            ))

        elif isinstance(val, StatResult):
            tools_invoked.append("compute_variable_stat")
            all_items.append(ExplainableItem(
                item_id=f"stat_{val.variable_id}",
                display_name=f"{val.stat_type.capitalize()} of {val.variable_id}",
                raw_key=val.variable_id,
                value=val.value,
                source_tool="compute_variable_stat",
                definition=f"Computed {val.stat_type} of {val.variable_id} over n={val.n} participants.",
                importance="high",
                explanation_required=True,
                section_id="cohort",
            ))

        else:
            tools_invoked.append("unknown_tool")
            all_items.append(ExplainableItem(
                item_id=f"result_{i}",
                display_name=f"Tool result {i}",
                raw_key=f"result_{i}",
                value=repr(val)[:100],
                source_tool="unknown",
                importance="low",
                section_id="metadata",
            ))

    # Deduplicate
    deduped = _deduplicate_items(all_items)

    # Structured caveats
    caveats = _build_structured_caveats(
        target_etiology=target_etiology,
        outcome_label_def=outcome_label_def,
        catalog=catalog,
    )

    # Sections
    sections = _assemble_sections(deduped)

    # Metadata
    horizon_label = (
        f"{horizon} years (example default — user did not specify a horizon)"
        if horizon_source == "default"
        else f"{horizon} years"
    )
    metadata = ExplanationMetadata(
        horizon_years=horizon,
        horizon_source=horizon_source,  # type: ignore[arg-type]
        horizon_label=horizon_label,
        backend=backend_name,
        catalog_release=catalog_release,
        tools_invoked=list(dict.fromkeys(tools_invoked)),
    )

    return ExplanationContext(
        items=deduped,
        sections=sections,
        caveats=caveats,
        metadata=metadata,
        not_answerable=not_answerable,
    )


# ---------------------------------------------------------------------------
# Explanation coverage validator
# ---------------------------------------------------------------------------

_EXPLANATION_COVERAGE_THRESHOLD = 0.8


def compute_explanation_coverage(ctx: ExplanationContext, answer_text: str) -> tuple[float, list[str]]:
    """Return (coverage_ratio, list_of_missing_display_names).

    An item is 'explained' if its display_name (or raw_key) appears in the
    answer text (case-insensitive).  For leakage items, also checks the raw_code
    (the value field when source_tool == 'suggest_modeling_design' and raw_key
    starts with a variable_id pattern).
    """
    required = ctx.required_items()
    if not required:
        return 1.0, []

    answer_lower = answer_text.lower()
    explained: list[str] = []
    missing: list[str] = []

    for item in required:
        # Check display_name
        if item.display_name.lower() in answer_lower:
            explained.append(item.item_id)
            continue
        # Check raw_key
        if item.raw_key.lower() in answer_lower:
            explained.append(item.item_id)
            continue
        # For leakage records: check the value (raw_code)
        if isinstance(item.value, str) and item.value.lower() in answer_lower:
            explained.append(item.item_id)
            continue
        # Check individual words from display_name (≥5 chars, unambiguous)
        words = [w for w in re.sub(r"[:()/]", " ", item.display_name.lower()).split() if len(w) >= 5]
        if words and any(w in answer_lower for w in words):
            explained.append(item.item_id)
            continue
        missing.append(item.display_name)

    ratio = len(explained) / max(1, len(required))
    return ratio, missing
