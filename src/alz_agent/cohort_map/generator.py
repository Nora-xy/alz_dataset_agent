"""P6 — Cohort map generator + serializer.

generate_cohort_map(filter_spec, horizon, catalog, backend) composes P3 tool
outputs into a populated CohortMap — the handoff artifact Stage 2 consumes.
It calls compute_cohort_size, assess_longitudinal_feasibility,
suggest_modeling_design (which internally calls check_leakage), and
assess_modality_overlap. It never recomputes a number itself; every numeric
field traces to a ToolResult, and that ToolResult's evidence is recorded in
the returned report's provenance map (invariant 3).

Invariant 8 (id-space partition): every tool call below is driven by a
single filter_spec.id_space; the generator never mixes Space A and Space B
counts. Space B (ADNI) cohorts are always Tier 3 / external_validation_candidate
per the ID-SPACE & LINKAGE MODEL in AGENTS.md, and carry the NACC<->ADNI
subject-overlap caveat as a limitation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alz_agent.backend.interface import DataBackend
from alz_agent.constants import SUPPRESSION_THRESHOLD
from alz_agent.schemas import (
    CatalogEntry,
    CohortMap,
    CohortRole,
    Evidence,
    FilterSpec,
    IdSpace,
    Modality,
    Tier,
)
from alz_agent.tools.deterministic import (
    ModelingDesignResult,
    assess_longitudinal_feasibility,
    assess_modality_overlap,
    compute_cohort_size,
    suggest_modeling_design,
)

# ---------------------------------------------------------------------------
# Limitation text (verbatim caveat from AGENTS.md's LEAKAGE CAVEAT section)
# ---------------------------------------------------------------------------

_LIMITATION_CROSSWALK = (
    "NACC<->ADNI subject overlap undetectable without an RID<->NACCID crosswalk; "
    "this external-validation cohort (Tier 3 / Space B) may share subjects with "
    "Tier 1 training data."
)

_NOT_AVAILABLE = "not available"

_NUMERIC_FIELD_NAMES = (
    "eligible_n",
    "follow_up_availability",
    "modality_coverage",
    "sample_size_by_modality_combo",
)


# ---------------------------------------------------------------------------
# Generation report (provenance sidecar — CohortMap itself stays P0-frozen)
# ---------------------------------------------------------------------------


@dataclass
class FieldProvenance:
    """Evidence trail for one CohortMap field, or the abstain reason if absent."""

    available: bool
    evidence: list[Evidence] = field(default_factory=list)
    reason: str | None = None  # set iff available is False


@dataclass
class CohortMapGenerationReport:
    """The populated CohortMap plus per-field provenance.

    Stage 2 consumes ``cohort_map`` (via the serializer). ``provenance`` is the
    audit trail satisfying invariant 3: every number in cohort_map traces back
    to one of these evidence lists, or is recorded here as unavailable with a
    reason instead of being fabricated.
    """

    cohort_map: CohortMap
    provenance: dict[str, FieldProvenance] = field(default_factory=dict)

    @property
    def unavailable_fields(self) -> list[str]:
        return [name for name, p in self.provenance.items() if not p.available]

    def summary(self) -> str:
        lines = [
            "=== Cohort Map Generation Report ===",
            f"id_space={self.cohort_map.id_space.value} "
            f"tier={self.cohort_map.tier.value} "
            f"cohort_role={self.cohort_map.cohort_role.value}",
            "",
        ]
        for name, p in self.provenance.items():
            status = "OK" if p.available else f"NOT AVAILABLE ({p.reason})"
            lines.append(f"  {name}: {status}")
        if self.cohort_map.limitations:
            lines.append("")
            lines.append("Limitations:")
            lines.extend(f"  - {lim}" for lim in self.cohort_map.limitations)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_cohort_map(
    filter_spec: FilterSpec,
    horizon: int,
    catalog: list[CatalogEntry],
    backend: DataBackend,
    *,
    cohort_role: CohortRole | None = None,
    tier: Tier | None = None,
    modalities: list[str] | None = None,
) -> CohortMapGenerationReport:
    """Compose deterministic tool outputs into a CohortMap.

    Args:
        filter_spec: Cohort eligibility filter (carries id_space; invariant 8).
        horizon:     Prediction horizon in years.
        catalog:     Parsed CatalogEntry list (P1 output).
        backend:     DataBackend implementation (P2).
        cohort_role: Override for Space A cohorts only (default TRAINING).
                     Ignored for Space B — always EXTERNAL_VALIDATION_CANDIDATE.
        tier:        Override for Space A cohorts only (default TIER1).
                     Ignored for Space B — always TIER3.
        modalities:  Modalities to report coverage/overlap for. Defaults to
                     every modality present in the catalog for this id_space
                     (excluding Modality.OTHER).

    Returns:
        CohortMapGenerationReport(cohort_map=..., provenance=...).
    """
    id_space = filter_spec.id_space

    if id_space == IdSpace.B:
        # Space B is always the external-validation tier — see AGENTS.md
        # DATA TIERS / ID-SPACE & LINKAGE MODEL. Any caller override is ignored.
        resolved_tier = Tier.TIER3
        resolved_role = CohortRole.EXTERNAL_VALIDATION_CANDIDATE
    else:
        resolved_tier = tier or Tier.TIER1
        resolved_role = cohort_role or CohortRole.TRAINING

    provenance: dict[str, FieldProvenance] = {}
    limitations: list[str] = []

    # ---- eligible_n (compute_cohort_size) ----------------------------------
    size_result = compute_cohort_size(filter_spec, backend)
    eligible_n: int | None = size_result.value if size_result.grounded else None
    provenance["eligible_n"] = FieldProvenance(
        available=size_result.grounded,
        evidence=list(size_result.evidence),
        reason=None if size_result.grounded else size_result.abstain_reason,
    )

    # ---- follow_up_availability (assess_longitudinal_feasibility) ----------
    feas_result = assess_longitudinal_feasibility(horizon, filter_spec, backend)
    follow_up_availability: dict[str, Any] = (
        dict(feas_result.value) if feas_result.grounded and feas_result.value is not None else {}
    )
    provenance["follow_up_availability"] = FieldProvenance(
        available=feas_result.grounded,
        evidence=list(feas_result.evidence),
        reason=None if feas_result.grounded else feas_result.abstain_reason,
    )

    # ---- index_visit_def / outcome_label_def / feature & leakage lists ------
    # suggest_modeling_design calls compute_cohort_size + assess_longitudinal_
    # feasibility + check_leakage internally; we don't re-derive any of its
    # outputs, only read them.
    design_result = suggest_modeling_design(filter_spec, horizon, catalog, backend)
    design: ModelingDesignResult | None = (
        design_result.value if design_result.grounded else None
    )
    index_visit_def = design.index_visit_def if design is not None else _NOT_AVAILABLE
    outcome_label_def = design.outcome_label_def if design is not None else _NOT_AVAILABLE
    feature_whitelist = list(design.feature_whitelist) if design is not None else []
    leakage_blacklist = list(design.leakage_blacklist) if design is not None else []
    provenance["modeling_design"] = FieldProvenance(
        available=design_result.grounded,
        evidence=list(design_result.evidence),
        reason=None if design_result.grounded else design_result.abstain_reason,
    )

    # ---- modality_coverage + sample_size_by_modality_combo ------------------
    # (assess_modality_overlap across the relevant combinations)
    target_modalities = (
        modalities
        if modalities is not None
        else sorted({
            e.modality.value
            for e in catalog
            if e.id_space == id_space and e.modality != Modality.OTHER
        })
    )

    modality_coverage: dict[str, int] = {}
    sample_size_by_modality_combo: dict[str, int] = {}

    if not target_modalities:
        provenance["modality_coverage"] = FieldProvenance(
            available=False,
            reason=f"no modalities present in catalog for id_space={id_space.value}",
        )
    else:
        overlap_result = assess_modality_overlap(
            target_modalities, filter_spec, backend, catalog
        )
        provenance["modality_coverage"] = FieldProvenance(
            available=overlap_result.grounded,
            evidence=list(overlap_result.evidence),
            reason=None if overlap_result.grounded else overlap_result.abstain_reason,
        )
        if overlap_result.grounded and overlap_result.value is not None:
            suppressed_combos: list[str] = []
            for combo_key, count in overlap_result.value.items():
                if count is None:
                    suppressed_combos.append(combo_key)
                    continue
                if "," in combo_key:
                    sample_size_by_modality_combo[combo_key] = count
                else:
                    modality_coverage[combo_key] = count
            if suppressed_combos:
                limitations.append(
                    "Modality combination(s) below suppression threshold "
                    f"({SUPPRESSION_THRESHOLD}) omitted from coverage/overlap: "
                    f"{', '.join(sorted(suppressed_combos))}"
                )

    # ---- release_versions (read directly from catalog, not a tool call) -----
    versions_by_source: dict[str, set[str]] = {}
    for entry in catalog:
        if entry.id_space != id_space:
            continue
        versions_by_source.setdefault(entry.source_dataset.value, set()).add(
            entry.release_version
        )
    release_versions = {
        source: "+".join(sorted(versions)) for source, versions in versions_by_source.items()
    }

    # ---- Tier 3 / Space B caveat --------------------------------------------
    if resolved_role == CohortRole.EXTERNAL_VALIDATION_CANDIDATE:
        limitations.append(_LIMITATION_CROSSWALK)

    cohort_map = CohortMap(
        id_space=id_space,
        tier=resolved_tier,
        cohort_role=resolved_role,
        eligible_n=eligible_n,
        index_visit_def=index_visit_def,
        prediction_horizon=horizon,
        outcome_label_def=outcome_label_def,
        follow_up_availability=follow_up_availability,
        modality_coverage=modality_coverage,
        feature_whitelist=feature_whitelist,
        leakage_blacklist=leakage_blacklist,
        sample_size_by_modality_combo=sample_size_by_modality_combo,
        release_versions=release_versions,
        suppression_threshold=SUPPRESSION_THRESHOLD,
        limitations=limitations,
        generated_at=datetime.now(timezone.utc),
    )

    return CohortMapGenerationReport(cohort_map=cohort_map, provenance=provenance)


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

# Bump when the on-disk envelope shape or CohortMap schema changes incompatibly.
SCHEMA_VERSION = "1.0"


def serialize_cohort_map(cohort_map: CohortMap, output_path: Path | str) -> Path:
    """Write a CohortMap to a stable, versioned JSON envelope.

    Envelope shape: {"schema_version": ..., "cohort_map": {...}}.
    Stage 2 should load via deserialize_cohort_map rather than parsing JSON directly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "cohort_map": json.loads(cohort_map.model_dump_json()),
    }
    output_path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
    return output_path


def deserialize_cohort_map(input_path: Path | str) -> CohortMap:
    """Load a CohortMap previously written by serialize_cohort_map.

    Raises ValueError if the envelope's schema_version is not supported.
    """
    input_path = Path(input_path)
    envelope = json.loads(input_path.read_text(encoding="utf-8"))
    version = envelope.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cohort map schema_version {version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    return CohortMap.model_validate(envelope["cohort_map"])
