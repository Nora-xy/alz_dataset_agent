"""Plan Validator — deterministic pre-execution checks for EnrichedAnalysisPlan.

No LLM calls.  Returns ValidationResult with hard_errors, soft_warnings,
assumptions, blocked_operations, and is_executable flag.

Checks performed:
  1. id_space integrity — plan.id_space must be "A" or "B"
  2. Horizon validity — all requested_horizons and per-op horizons are positive ints
  3. Parameter completeness — each op_type has its required parameters
  4. output_name uniqueness — no duplicate output_names across ops
  5. Cross-space join prevention — no filter mixing Space A and Space B fields
  6. AD/all-cause caveat — syndromic outcome codes (NACCUDSD) mentioned alongside
     AD etiology intent → soft warning (not triggered by id_space alone)
  7. Horizon sanity — horizons > 10 years flagged as possible misparse
"""

from __future__ import annotations

import json
import re
from typing import Any

from alz_agent.schemas import (
    CatalogEntry,
    CohortScope,
    EnrichedAnalysisPlan,
    EtiologyDefinition,
    GroupDefinition,
    PopulationUnit,
    SemanticAnalysisPlan,
    ValidationResult,
    VariableSelection,
)

# Values that belong in etiology_definition (AD attribution), never in a baseline_filters/
# event_filters "diagnosis" slot (which encodes cognitive STATUS: normal/mci/dementia, not
# etiology). A literal "ad" here would reach the backend's diagnosis-status map, which has
# no such code, and fail with a confusing backend error instead of a clear semantic one.
_AD_ETIOLOGY_VALUES = frozenset({"ad", "alzheimer", "alzheimers", "alzheimer's disease", "alzheimer disease"})

# op_types that require specific parameters to be present
_REQUIRED_PARAMS: dict[str, list[str]] = {
    "variable_stat": ["variable_id", "stat_type"],
    "descriptive_stat": ["variable_id"],
    "modality_overlap": ["modalities"],
    "group_comparison": ["group_variable_id", "outcome_variable_id"],
    "explore_catalog": ["keyword"],
    "describe_variable_family": ["family"],
}

# Regex to detect syndromic (all-cause) dementia codes in outcome_definition.
# Catches NACCUDSD=4 (NACC), DIAGNOSIS=3 (ADNI), explicit "all-cause dementia",
# and generic "syndromic dementia" phrasing.  Intentionally broad — the etiologic
# pattern below is used to suppress the caveat when etiologic codes ARE present.
_SYNDROMIC_CODE_PATTERN = re.compile(
    r"NACCUDSD\s*=\s*\d+"           # NACC all-cause dementia code (e.g., NACCUDSD=4)
    r"|DIAGNOSIS\s*=\s*3"            # ADNI dementia code
    r"|all.cause\s+dementia"         # explicit all-cause mention
    r"|\bsyndromic\s+dementia\b",    # explicit syndromic mention
    re.IGNORECASE,
)

# Regex to detect etiologic AD codes
_ETIOLOGIC_CODE_PATTERN = re.compile(
    r"NACCALZD\s*=\s*1|etiologic\s+AD|etiologic\s+Alzheimer",
    re.IGNORECASE,
)

_AD_INTENT_PATTERN = re.compile(
    r"\bAD\b|alzheimer",
    re.IGNORECASE,
)

_VALID_ID_SPACES = {"A", "B"}

_VALID_OP_TYPES = frozenset({
    "count_eligible",
    "longitudinal_feasibility",
    "modality_overlap",
    "variable_stat",
    "descriptive_stat",
    "group_comparison",
    "lookup_variable",
    "leakage_check",
    "modeling_design",
    "explore_catalog",
    "describe_variable_family",
})

# op_types that are catalog-only (no backend/participant data needed).
_CATALOG_ONLY_OP_TYPES = frozenset({"lookup_variable", "leakage_check", "explore_catalog"})


# ---------------------------------------------------------------------------
# Scope-agnostic coherence checks (C1/C2b/C3/C5) — shared by the top-level
# SemanticAnalysisPlan (single-cohort questions) and each GroupDefinition
# (multi-group comparison questions). One rule set, applied wherever a cohort
# scope is defined, instead of duplicating the logic per call site.
# ---------------------------------------------------------------------------


def _lbl(label: str) -> str:
    return f"[{label}] " if label else ""


def _check_first_event_coherence(
    *,
    cohort_scope: CohortScope,
    event_filters: dict[str, Any],
    event_anchor: str | None,
    hard_errors: list[str],
    label: str = "",
) -> None:
    """C1: first_event scope requires non-empty event_filters and correct event_anchor."""
    if cohort_scope != CohortScope.FIRST_EVENT:
        return
    p = _lbl(label)
    if not event_filters:
        hard_errors.append(
            f"{p}C1: cohort_scope=first_event but event_filters is empty. "
            "Specify the event criterion (e.g. event_filters={NACCALZD: 1})."
        )
    valid_anchors = {"first_matching_event_visit", "first_conversion_visit"}
    if event_anchor not in valid_anchors:
        hard_errors.append(
            f"{p}C1: cohort_scope=first_event requires event_anchor in "
            f"{valid_anchors}, got '{event_anchor}'. "
            "Set event_anchor=first_matching_event_visit for first-diagnosis questions."
        )


def _check_participant_anchor_required(
    *,
    cohort_scope: CohortScope,
    event_anchor: str | None,
    hard_errors: list[str],
    label: str = "",
) -> None:
    """C2b: multi-visit scope without an anchor → rows not deduplicated per participant."""
    multi_visit_scopes = {
        CohortScope.ANY_VISIT,
        CohortScope.FIRST_EVENT,
        CohortScope.FOLLOWUP_WINDOW,
    }
    if cohort_scope in multi_visit_scopes and event_anchor is None:
        p = _lbl(label)
        hard_errors.append(
            f"{p}C2b: population_unit=participant with a multi-visit cohort_scope "
            f"({cohort_scope.value}) requires a non-null event_anchor to "
            "reduce each participant to one row before computing the statistic. "
            "Set event_anchor (e.g. first_matching_event_visit)."
        )


# Per-id_space etiologic-AD vs. syndromic-dementia marker columns (BACKEND_CONTRACT.md):
# Space A (NACC): NACCALZD=1 is etiologic AD; NACCUDSD=4 is all-cause dementia.
# Space B (ADNI): DXAD/DXMDUE/DXDDUE==1 is etiologic AD; DIAGNOSIS==3 is all-cause dementia.
# Generic per-space lookup, not a single NACC-only literal, so this check applies
# uniformly to any dataset/id_space rather than being hardcoded to one variable name.
_ETIOLOGIC_MARKERS_BY_SPACE: dict[str, tuple[str, ...]] = {
    "A": ("NACCALZD",),
    "B": ("DXAD", "DXMDUE", "DXDDUE"),
}
_SYNDROMIC_MARKERS_BY_SPACE: dict[str, tuple[str, ...]] = {
    "A": ("NACCUDSD",),
    "B": ("DIAGNOSIS",),
}


def _check_etiology_vs_allcause(
    *,
    etiology_definition: EtiologyDefinition,
    diagnosis_variable: str,
    hard_errors: list[str],
    label: str = "",
    id_space: str = "A",
) -> None:
    """C3: etiologic_ad etiology requires an etiologic marker in diagnosis_variable,
    for whichever id_space the plan/group targets (see marker tables above)."""
    if etiology_definition != EtiologyDefinition.ETIOLOGIC_AD:
        return
    p = _lbl(label)
    dv = diagnosis_variable.upper()
    etiologic_markers = _ETIOLOGIC_MARKERS_BY_SPACE.get(id_space, _ETIOLOGIC_MARKERS_BY_SPACE["A"])
    syndromic_markers = _SYNDROMIC_MARKERS_BY_SPACE.get(id_space, _SYNDROMIC_MARKERS_BY_SPACE["A"])
    has_etiologic = any(m in dv for m in etiologic_markers)
    has_syndromic = any(m in dv for m in syndromic_markers)

    if has_etiologic:
        return
    if has_syndromic:
        hard_errors.append(
            f"{p}C3: diagnosis_variable ('{diagnosis_variable}') contains a syndromic/"
            f"all-cause marker but no etiologic-AD marker for id_space={id_space}, yet "
            "etiology_definition=etiologic_ad. Add an etiologic marker "
            f"(one of {etiologic_markers}) to diagnosis_variable."
        )
    else:
        hard_errors.append(
            f"{p}C3: etiology_definition=etiologic_ad but diagnosis_variable "
            f"(got '{diagnosis_variable}') references no etiologic-AD marker for "
            f"id_space={id_space} (expected one of {etiologic_markers}). "
            f"{etiologic_markers[0]}=1 is the etiologic AD flag; a syndromic/cognitive-"
            "status marker alone is all-cause dementia, not AD attribution."
        )


def _check_baseline_event_etiology_coherence(
    *,
    cohort_scope: CohortScope,
    etiology_definition: EtiologyDefinition,
    baseline_filters: dict[str, Any],
    event_filters: dict[str, Any],
    hard_errors: list[str],
    label: str = "",
) -> None:
    """C5/C5b: AD etiology must live in the correct filter bucket for the scope."""
    if etiology_definition != EtiologyDefinition.ETIOLOGIC_AD:
        return
    p = _lbl(label)

    ad_in_baseline = any(
        str(v).lower() in ("ad", "alzheimer", "alzheimers")
        for k, v in baseline_filters.items()
        if k.lower() in ("etiology", "baseline_etiology")
    )
    ad_in_event = any(
        str(v).lower() in ("ad", "alzheimer", "alzheimers")
        or str(k).upper() == "NACCALZD"
        for k, v in event_filters.items()
    )

    if cohort_scope == CohortScope.BASELINE:
        if ad_in_event and not ad_in_baseline:
            hard_errors.append(
                f"{p}C5: cohort_scope=baseline with etiology_definition=etiologic_ad but "
                "AD is only in event_filters (any-visit). "
                "Move AD to baseline_filters (e.g. baseline_filters={etiology: 'ad'}) "
                "to use baseline_etiology (first visit only)."
            )
    else:
        if ad_in_baseline and not ad_in_event:
            hard_errors.append(
                f"{p}C5b: cohort_scope={cohort_scope.value} with "
                "etiology_definition=etiologic_ad but AD is only in baseline_filters "
                "(first-visit-only). Move AD to event_filters so that any-visit / "
                "first-event semantics are applied correctly "
                "(e.g. event_filters={NACCALZD: 1})."
            )


def _check_diagnosis_allcause_collision(
    *,
    etiology_definition: EtiologyDefinition,
    baseline_filters: dict[str, Any],
    hard_errors: list[str],
    label: str = "",
) -> None:
    """C16: an explicit baseline_filters['diagnosis'] must not collide with
    etiology_definition=all_cause_dementia on the same compiled filter key.

    _filter_dict_for_scope() (semantic_planner.py) compiles a cohort's diagnosis
    intent onto ONE 'diagnosis' key: baseline_filters['diagnosis'] sets it first,
    but the legacy all_cause_dementia branch then unconditionally OVERWRITES it
    to 'dementia' — silently discarding whatever baseline_filters had asked for
    (e.g. a 'baseline MCI who progress to dementia' plan compiles to plain
    'baseline dementia', dropping the MCI-at-baseline requirement entirely, with
    no error and no caveat). There is currently no FilterSpec primitive for an
    event-level all-cause-dementia condition distinct from baseline_diagnosis, so
    this combination cannot be correctly compiled — it must be rejected here
    rather than silently mis-answered.
    """
    if etiology_definition != EtiologyDefinition.ALL_CAUSE_DEMENTIA:
        return
    existing = baseline_filters.get("diagnosis")
    if existing and str(existing).lower() != "dementia":
        p = _lbl(label)
        hard_errors.append(
            f"{p}C16: baseline_filters sets diagnosis='{existing}' but "
            "etiology_definition=all_cause_dementia also compiles onto the same "
            "'diagnosis' filter key as 'dementia' — the two collide and the "
            f"'{existing}' baseline requirement would be silently discarded. "
            "There is no backend primitive for 'baseline X who later reach "
            "all-cause dementia' — if this is a conversion/transition question, "
            "list it in abstain_if_missing instead of setting a conflicting "
            "etiology_definition."
        )


class PlanValidator:
    """Stateless validator — instantiate once, call validate() per plan.

    The catalog argument is accepted for future variable-existence checks but
    is not required for the current deterministic checks.
    """

    def __init__(self, catalog: list[CatalogEntry] | None = None) -> None:
        self._catalog = catalog or []

    def validate(self, plan: EnrichedAnalysisPlan) -> ValidationResult:
        """Run all checks and return a ValidationResult."""
        hard_errors: list[str] = []
        soft_warnings: list[str] = []
        assumptions: list[str] = []
        blocked_ops: list[str] = []

        self._check_id_space(plan, hard_errors)
        self._check_horizons(plan, hard_errors, soft_warnings)
        self._check_operations(plan, hard_errors, soft_warnings, blocked_ops, assumptions)
        self._check_ad_caveat(plan, soft_warnings)

        is_executable = (
            len(hard_errors) == 0
            and len(plan.operations) > len(blocked_ops)
        )

        return ValidationResult(
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            assumptions=assumptions,
            is_executable=is_executable,
            blocked_operations=blocked_ops,
        )

    # ------------------------------------------------------------------
    # Private checks
    # ------------------------------------------------------------------

    def _check_id_space(
        self,
        plan: EnrichedAnalysisPlan,
        hard_errors: list[str],
    ) -> None:
        if plan.id_space not in _VALID_ID_SPACES:
            hard_errors.append(
                f"id_space '{plan.id_space}' is not valid; must be 'A' (NACC/ADSP-PHC) "
                "or 'B' (ADNI_partial)."
            )

    def _check_horizons(
        self,
        plan: EnrichedAnalysisPlan,
        hard_errors: list[str],
        soft_warnings: list[str],
    ) -> None:
        for h in plan.requested_horizons:
            if not isinstance(h, int) or h <= 0:
                hard_errors.append(
                    f"requested_horizons: {h!r} is not a positive integer."
                )
            elif h > 10:
                soft_warnings.append(
                    f"requested_horizons: horizon {h} years is unusually long — "
                    "verify this is not a misparse."
                )

        for op in plan.operations:
            if op.op_type != "longitudinal_feasibility":
                continue
            horizons = op.parameters.get("horizons", [])
            if not horizons:
                hard_errors.append(
                    f"op '{op.output_name}' (longitudinal_feasibility): "
                    "missing required parameter 'horizons' (must be a non-empty list)."
                )
                continue
            if not isinstance(horizons, list):
                hard_errors.append(
                    f"op '{op.output_name}': 'horizons' must be a list, "
                    f"got {type(horizons).__name__}."
                )
                continue
            for h in horizons:
                if not isinstance(h, int) or h <= 0:
                    hard_errors.append(
                        f"op '{op.output_name}': horizon {h!r} must be a positive integer."
                    )
                elif h > 10:
                    soft_warnings.append(
                        f"op '{op.output_name}': horizon {h} years is unusually long — "
                        "verify this is not a misparse."
                    )

    def _check_operations(
        self,
        plan: EnrichedAnalysisPlan,
        hard_errors: list[str],
        soft_warnings: list[str],
        blocked_ops: list[str],
        assumptions: list[str],
    ) -> None:
        seen_output_names: set[str] = set()

        for op in plan.operations:
            # Check op_type is known
            if op.op_type not in _VALID_OP_TYPES:
                hard_errors.append(
                    f"op '{op.output_name}': unknown op_type '{op.op_type}'. "
                    f"Valid types: {sorted(_VALID_OP_TYPES)}."
                )
                blocked_ops.append(op.output_name)
                continue

            # Check output_name uniqueness
            if op.output_name in seen_output_names:
                hard_errors.append(
                    f"Duplicate output_name '{op.output_name}' — "
                    "each operation must have a unique output_name."
                )
                # Don't add to blocked_ops for duplicates; both will try to run
            seen_output_names.add(op.output_name)

            # Check required parameters
            required = _REQUIRED_PARAMS.get(op.op_type, [])
            missing = [p for p in required if p not in op.parameters]
            if missing:
                hard_errors.append(
                    f"op '{op.output_name}' ({op.op_type}): "
                    f"missing required parameter(s): {missing}."
                )
                blocked_ops.append(op.output_name)

            # Check variable_id format for variable_stat / descriptive_stat
            if op.op_type in ("variable_stat", "descriptive_stat"):
                self._check_variable_id_space(op, plan.id_space, hard_errors, blocked_ops)

            # Cross-space check: filter dict must not reference both NACC and ADNI keys
            self._check_filter_cross_space(op, plan.id_space, hard_errors, blocked_ops)

        # If no operations at all, that's a hard error
        if not plan.operations:
            hard_errors.append(
                "EnrichedAnalysisPlan has no operations — nothing to execute."
            )

        # Record assumptions about backend data
        if plan.requires_backend_data:
            assumptions.append(
                "This plan requires real backend data (real NACC/ADNI files must be mounted)."
            )
        else:
            # Verify no ops actually need the backend
            backend_ops = [
                op.output_name for op in plan.operations
                if op.op_type not in _CATALOG_ONLY_OP_TYPES
            ]
            if backend_ops:
                soft_warnings.append(
                    f"Plan sets requires_backend_data=False but ops {backend_ops} "
                    "will call the DataBackend — marking as requires_backend_data=True."
                )

    def _check_variable_id_space(
        self,
        op: "PlanOperation",
        plan_id_space: str,
        hard_errors: list[str],
        blocked_ops: list[str],
    ) -> None:
        """Reject variable_ids whose dataset prefix contradicts the plan's id_space.

        Space A (NACC/ADSP-PHC) variables must be 'NACC.<COLUMN>'.
        Space B (ADNI) variables must be 'ADNI.<TABLE>.<FIELD>' or a supported
        derived id 'ADNI.DERIVED.<name>' (e.g. 'ADNI.DERIVED.age_at_event').
        A mismatched prefix is a cross-space variable leak — a hard error, and the
        op is blocked so it never reaches the backend.
        """
        vid = op.parameters.get("variable_id", "")
        if not vid:
            return

        has_known_prefix = vid.startswith("NACC.") or vid.startswith("ADNI.")
        if not has_known_prefix:
            hard_errors.append(
                f"op '{op.output_name}' ({op.op_type}): variable_id '{vid}' "
                "must carry a dataset prefix — 'NACC.<COLUMN_NAME>' for Space A "
                "(e.g. 'NACC.NACCAGE') or 'ADNI.<TABLE>.<FIELD>' for Space B "
                "(e.g. 'ADNI.MMSE.MMSCORE')."
            )
            if op.output_name not in blocked_ops:
                blocked_ops.append(op.output_name)
            return

        if plan_id_space == "B" and vid.startswith("NACC."):
            hard_errors.append(
                f"op '{op.output_name}' ({op.op_type}): id_space is 'B' (ADNI) but "
                f"variable_id '{vid}' references a Space A (NACC) variable — "
                "cross-space variable leak. Use an 'ADNI.<TABLE>.<FIELD>' variable "
                "(e.g. 'ADNI.MMSE.MMSCORE') or the derived 'ADNI.DERIVED.age_at_event'."
            )
            if op.output_name not in blocked_ops:
                blocked_ops.append(op.output_name)
        elif plan_id_space == "A" and vid.startswith("ADNI."):
            hard_errors.append(
                f"op '{op.output_name}' ({op.op_type}): id_space is 'A' (NACC/ADSP-PHC) "
                f"but variable_id '{vid}' references a Space B (ADNI) variable — "
                "cross-space variable leak. Use a 'NACC.<COLUMN>' variable "
                "(e.g. 'NACC.NACCAGE')."
            )
            if op.output_name not in blocked_ops:
                blocked_ops.append(op.output_name)

    def _check_filter_cross_space(
        self,
        op: "PlanOperation",
        plan_id_space: str,
        hard_errors: list[str],
        blocked_ops: list[str],
    ) -> None:
        filter_dict: dict[str, Any] = op.parameters.get("filter", {})
        if not filter_dict:
            return

        # Space A keys
        space_a_keys = {"naccid", "naccvnum", "naccudsd", "naccalzd"}
        # Space B keys. NOTE: "diagnosis" is deliberately NOT included here — it is
        # the compiler's own universal semantic filter key (set by
        # semantic_planner._filter_dict_for_scope for baseline diagnosis, and read by
        # executor._build_filter_spec) for BOTH id_space A and B, never a literal raw
        # column reference. It happens to collide (case-insensitively) with ADNI's raw
        # DXSUM.DIAGNOSIS column name, which is what led this check to originally
        # misclassify it as Space-B-only — every legitimate Space A (NACC) plan whose
        # compiled filter also carried a raw NACC column (e.g. a literal "NACCUDSD" key
        # passed through from event_filters) would then false-positive here and abstain.
        space_b_keys = {"rid", "ptid", "subjid", "viscode", "dxcurren"}

        keys_lower = {k.lower() for k in filter_dict}
        has_a = bool(keys_lower & space_a_keys)
        has_b = bool(keys_lower & space_b_keys)

        if has_a and has_b:
            hard_errors.append(
                f"op '{op.output_name}': filter mixes Space A keys "
                f"({keys_lower & space_a_keys}) and Space B keys "
                f"({keys_lower & space_b_keys}) — cross-space join not allowed."
            )
            if op.output_name not in blocked_ops:
                blocked_ops.append(op.output_name)

    def _check_ad_caveat(
        self,
        plan: EnrichedAnalysisPlan,
        soft_warnings: list[str],
    ) -> None:
        """Emit AD/all-cause caveat when outcome uses syndromic codes AND AD is mentioned.

        Rule: trigger when outcome_definition references NACCUDSD (syndromic dementia)
        AND the question or outcome text mentions AD/Alzheimer's intent.
        Do NOT trigger based on id_space alone.
        """
        outcome = plan.outcome_definition
        if not outcome:
            return

        uses_syndromic = bool(_SYNDROMIC_CODE_PATTERN.search(outcome))
        uses_etiologic = bool(_ETIOLOGIC_CODE_PATTERN.search(outcome))

        if uses_syndromic and not uses_etiologic:
            # Check if question mentions AD intent
            combined_text = plan.original_question + " " + outcome
            if _AD_INTENT_PATTERN.search(combined_text):
                soft_warnings.append(
                    "outcome_definition uses a syndromic dementia code (e.g., NACCUDSD=4 or "
                    "DIAGNOSIS=3), which captures all-cause dementia — not etiologic AD. "
                    "If AD-specific conversion is required, use an etiologic code such as "
                    "NACCALZD=1 instead. This distinction matters for label definition in "
                    "modeling studies."
                )

    # ------------------------------------------------------------------
    # Semantic plan validation (upstream gate)
    # ------------------------------------------------------------------

    def validate_semantic(self, sem: SemanticAnalysisPlan) -> ValidationResult:
        """Validate a SemanticAnalysisPlan before compile_to_enriched() runs.

        Checks C1-C9 as defined in the plan, plus:
          C10 catalog existence  — every selected variable_id must exist in the catalog.
          C11 specimen/analyte consistency — declared specimen/analyte must be supported
              by the resolved catalog entry's own tags, not just asserted by the LLM.
          C13 diagnosis-domain — "diagnosis" filters may only hold cognitive-status labels
              (normal/mci/dementia/...); AD attribution belongs in etiology_definition.
          C14 distinct group filters — no two groups may share an identical defining
              filter (would silently collapse to the same cohort under two labels).
          C15 non-empty group filters — every group in a multi-group plan must carry
              a real defining filter, never an implicit "entire cohort."

        When sem.groups is non-empty, C1/C2b/C3/C5 (the per-cohort coherence checks) run
        once per GroupDefinition instead of once on the top-level plan — the top-level
        cohort_scope/baseline_filters/event_filters/etiology_definition are superseded by
        the per-group definitions in that case. Returns a ValidationResult with
        is_executable=False if any hard errors are found.
        """
        hard_errors: list[str] = []
        soft_warnings: list[str] = []
        assumptions: list[str] = []

        if sem.groups:
            for group in sem.groups:
                self._check_group_coherence(group, hard_errors, id_space=sem.id_space)
        else:
            self._sc1_first_event_coherence(sem, hard_errors)
            self._sc3_etiology_vs_allcause(sem, hard_errors)
            self._sc5_baseline_event_etiology_coherence(sem, hard_errors)
            _check_diagnosis_allcause_collision(
                etiology_definition=sem.etiology_definition,
                baseline_filters=sem.baseline_filters,
                hard_errors=hard_errors,
            )

        self._sc2_participant_uniqueness(sem, hard_errors)
        self._sc4_denominator_required(sem, hard_errors)
        self._sc6_unclear_scope(sem, hard_errors)
        self._sc7_id_space(sem, hard_errors)
        self._sc8_anchor_scope_alignment(sem, soft_warnings)
        self._sc9_abstain_assumptions(sem, assumptions)
        self._sc10_catalog_existence(sem, hard_errors)
        self._sc11_specimen_analyte_consistency(sem, hard_errors)
        self._sc13_diagnosis_domain(sem, hard_errors)
        self._sc14_groups_have_distinct_defining_filters(sem, hard_errors)
        self._sc15_groups_require_nonempty_filters(sem, hard_errors)

        return ValidationResult(
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            assumptions=assumptions,
            is_executable=len(hard_errors) == 0,
            blocked_operations=[],
        )

    def _sc1_first_event_coherence(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C1: first_event scope requires non-empty event_filters and correct event_anchor."""
        _check_first_event_coherence(
            cohort_scope=sem.cohort_scope,
            event_filters=sem.event_filters,
            event_anchor=sem.event_anchor,
            hard_errors=hard_errors,
        )

    def _sc2_participant_uniqueness(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C2a+C2b: participant population_unit requires unique flag and an event anchor.

        C2b is skipped here when sem.groups is non-empty — it is checked per-group by
        _check_group_coherence instead, since each group may have its own cohort_scope.
        """
        if sem.population_unit != PopulationUnit.PARTICIPANT:
            return
        # C2a
        if not sem.requires_unique_participants:
            hard_errors.append(
                "C2a: population_unit=participant but requires_unique_participants=False. "
                "Raw row count is never a valid participant count. Set to True."
            )
        if sem.groups:
            return
        # C2b: multi-visit scope without an anchor → rows not deduplicated per participant
        _check_participant_anchor_required(
            cohort_scope=sem.cohort_scope,
            event_anchor=sem.event_anchor,
            hard_errors=hard_errors,
        )

    def _sc3_etiology_vs_allcause(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C3: etiologic_ad etiology requires an etiologic marker in diagnosis_variable."""
        _check_etiology_vs_allcause(
            etiology_definition=sem.etiology_definition,
            diagnosis_variable=sem.diagnosis_variable,
            hard_errors=hard_errors,
            id_space=sem.id_space,
        )

    def _sc4_denominator_required(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C4: statistics require a non-empty denominator definition."""
        stat_types_requiring_denom = {"mean", "median", "count", "proportion"}
        if sem.statistic_type in stat_types_requiring_denom:
            if not sem.denominator_definition.strip():
                hard_errors.append(
                    f"C4: statistic_type='{sem.statistic_type}' requires a non-empty "
                    "denominator_definition. State the exact denominator in plain language "
                    "(e.g. 'unique NACC participants with NACCALZD=1 on any visit')."
                )

    def _sc5_baseline_event_etiology_coherence(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C5/C5b: AD etiology must live in the correct filter bucket for the scope."""
        _check_baseline_event_etiology_coherence(
            cohort_scope=sem.cohort_scope,
            etiology_definition=sem.etiology_definition,
            baseline_filters=sem.baseline_filters,
            event_filters=sem.event_filters,
            hard_errors=hard_errors,
        )

    def _check_group_coherence(
        self, group: GroupDefinition, hard_errors: list[str], *, id_space: str = "A"
    ) -> None:
        """Apply C1/C2b/C3/C5 to one GroupDefinition (multi-group comparison path).

        Reuses the exact same scope-agnostic helpers as the top-level single-cohort
        checks, so a group is held to the same coherence bar as the whole-plan case —
        no separate rule set to keep in sync. id_space is inherited from the parent
        plan (GroupDefinition has no id_space of its own — every group in one plan
        shares the same dataset).
        """
        label = group.label or "(unlabeled group)"
        _check_first_event_coherence(
            cohort_scope=group.cohort_scope,
            event_filters=group.event_filters,
            event_anchor=group.event_anchor,
            hard_errors=hard_errors,
            label=label,
        )
        _check_participant_anchor_required(
            cohort_scope=group.cohort_scope,
            event_anchor=group.event_anchor,
            hard_errors=hard_errors,
            label=label,
        )
        _check_etiology_vs_allcause(
            etiology_definition=group.etiology_definition,
            diagnosis_variable=group.diagnosis_variable,
            hard_errors=hard_errors,
            label=label,
            id_space=id_space,
        )
        _check_baseline_event_etiology_coherence(
            cohort_scope=group.cohort_scope,
            etiology_definition=group.etiology_definition,
            baseline_filters=group.baseline_filters,
            event_filters=group.event_filters,
            hard_errors=hard_errors,
            label=label,
        )
        _check_diagnosis_allcause_collision(
            etiology_definition=group.etiology_definition,
            baseline_filters=group.baseline_filters,
            hard_errors=hard_errors,
            label=label,
        )

    def _sc6_unclear_scope(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C6: cohort_scope=unclear is never executable — force repair."""
        if sem.cohort_scope == CohortScope.UNCLEAR:
            hard_errors.append(
                "C6: cohort_scope=unclear — the LLM could not determine the cohort scope. "
                "This plan is not executable. The repair pass must resolve it to a "
                "specific scope (baseline / any_visit / first_event / etc.)."
            )

    def _sc7_id_space(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C7: id_space must be 'A' or 'B'."""
        if sem.id_space not in _VALID_ID_SPACES:
            hard_errors.append(
                f"C7: id_space '{sem.id_space}' is not valid; must be 'A' (NACC/ADSP-PHC) "
                "or 'B' (ADNI)."
            )

    def _sc8_anchor_scope_alignment(
        self, sem: SemanticAnalysisPlan, soft_warnings: list[str]
    ) -> None:
        """C8 (SOFT): baseline scope with non-baseline event_anchor is redundant."""
        if (
            sem.cohort_scope == CohortScope.BASELINE
            and sem.event_anchor is not None
            and sem.event_anchor != "baseline"
        ):
            soft_warnings.append(
                f"C8: cohort_scope=baseline but event_anchor='{sem.event_anchor}'. "
                "For baseline scope, the anchor is implicitly 'baseline' (first visit). "
                "The specified event_anchor will be ignored."
            )

    def _sc9_abstain_assumptions(
        self, sem: SemanticAnalysisPlan, assumptions: list[str]
    ) -> None:
        """C9: surface abstain_if_missing conditions as assumptions."""
        for condition in sem.abstain_if_missing:
            assumptions.append(f"Agent must abstain if missing: {condition}")

    # ------------------------------------------------------------------
    # Generic, catalog-driven checks — apply uniformly to any dataset/
    # concept, no per-concept or per-dataset hardcoding.
    # ------------------------------------------------------------------

    def _resolve_variable_ids(self, sem: SemanticAnalysisPlan) -> list[tuple[str, "VariableSelection | None"]]:
        """Collect (variable_id, source_selection) pairs the plan actually selects.

        source_selection is the VariableSelection the id came from, or None when it
        came from the legacy top-level statistic_variable (no specimen/analyte claim
        to cross-check in that case). Returns [] when the plan makes no variable
        selection at all (a pure count/cohort question).
        """
        if sem.variable_selections:
            return [(vs.variable_id, vs) for vs in sem.variable_selections if vs.variable_id]
        if sem.statistic_variable:
            return [(sem.statistic_variable, None)]
        return []

    def _catalog_lookup(self, variable_id: str, id_space: str) -> CatalogEntry | None:
        """Resolve one selected id against the active catalog.

        Space B (ADNI) requires an exact, case-insensitive match against a real
        entry's fully-qualified variable_id — the field name alone is ambiguous
        across tables, so no fallback to a bare-code match is offered here.
        Space A (NACC) additionally accepts a bare raw_code match (NACC columns
        are inherently unambiguous — one raw_code per column), and also accepts
        the id with or without the leading "NACC." prefix, for backward
        compatibility with callers that pass either form.
        """
        vid_upper = variable_id.upper()
        by_variable_id = {e.variable_id.upper(): e for e in self._catalog}
        if vid_upper in by_variable_id:
            return by_variable_id[vid_upper]

        if id_space == "A":
            bare = vid_upper[len("NACC."):] if vid_upper.startswith("NACC.") else vid_upper
            for e in self._catalog:
                if e.id_space.value == "A" and e.raw_code.upper() == bare:
                    return e
        return None

    def _sc10_catalog_existence(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C10: every selected variable_id must exist, verbatim, in the active catalog.

        Generic membership check — no biomarker/dataset-specific branches. Catches
        invented identifiers (e.g. a guessed ADNI table name) and identifiers shortened
        to just a field name where the field is ambiguous across multiple tables.
        Skipped when the catalog is empty (nothing to validate against — preserves
        unit tests that exercise validate_semantic() without building a full catalog).
        """
        if not self._catalog:
            return
        for variable_id, selection in self._resolve_variable_ids(sem):
            entry = self._catalog_lookup(variable_id, sem.id_space)
            if entry is None:
                label = f" (variant '{selection.variant_label}')" if selection and selection.variant_label else ""
                hard_errors.append(
                    f"C10: variable_id '{variable_id}'{label} was not found in the active "
                    "catalog. It must be copied verbatim from the catalog context — never "
                    "constructed or shortened. Use explore_catalog to find the real "
                    "variable_id if the requested measure exists under a different name."
                )

    def _sc11_specimen_analyte_consistency(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C11: declared specimen/analyte must be supported by the resolved entry's own tags.

        Generic substring cross-check against whatever family/variant/concept tags the
        catalog builder already assigned — not a hardcoded list of specimen or analyte
        names. This is what prevents e.g. a plasma variable from silently satisfying a
        CSF request: the LLM's claim is checked against catalog metadata, not trusted.
        Skipped when the catalog is empty, same rationale as C10.
        """
        if not self._catalog:
            return
        for variable_id, selection in self._resolve_variable_ids(sem):
            specimen = (selection.specimen if selection else None) or sem.specimen
            analyte = (selection.analyte if selection else None) or sem.analyte
            if not specimen and not analyte:
                continue
            entry = self._catalog_lookup(variable_id, sem.id_space)
            if entry is None:
                continue  # already reported by C10
            haystack = " ".join(
                part for part in (entry.family, entry.variant, entry.concept, entry.description)
                if part
            ).lower()
            if specimen and specimen.lower() not in haystack:
                hard_errors.append(
                    f"C11: variable_id '{variable_id}' was declared as specimen='{specimen}' "
                    f"but the catalog entry's tags (family='{entry.family}', "
                    f"variant='{entry.variant}', concept='{entry.concept}') do not support "
                    "that. Do not assert a specimen the catalog metadata does not confirm."
                )
            if analyte and analyte.lower() not in haystack:
                hard_errors.append(
                    f"C11: variable_id '{variable_id}' was declared as analyte='{analyte}' "
                    f"but the catalog entry's tags (family='{entry.family}', "
                    f"concept='{entry.concept}') do not support that."
                )

    def _sc13_diagnosis_domain(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C13: "diagnosis" filters may only hold cognitive-status labels.

        AD attribution is an etiology, not a cognitive-status diagnosis — it belongs in
        etiology_definition (+ the etiology/event_etiology filter keys), never in a
        "diagnosis" key. A literal "ad" reaching the backend's diagnosis-status map would
        fail with a confusing low-level error instead of a clear semantic one; this check
        catches it at the plan layer, for the top-level plan and every group.
        """
        def _check(filters: dict[str, Any], label: str) -> None:
            val = filters.get("diagnosis") or filters.get("baseline_diagnosis")
            if val is None:
                return
            values = [val] if isinstance(val, str) else list(val)
            for v in values:
                if str(v).strip().lower() in _AD_ETIOLOGY_VALUES:
                    p = _lbl(label)
                    hard_errors.append(
                        f"{p}C13: 'diagnosis' filter contains '{v}', which is an AD "
                        "etiology attribution, not a cognitive-status label. Cognitive "
                        "status is normal/mci/dementia/impaired-not-mci; use "
                        "etiology_definition=etiologic_ad (+ etiology/event_etiology filter "
                        "keys) to express AD attribution instead."
                    )

        if sem.groups:
            for group in sem.groups:
                _check(group.baseline_filters, group.label)
        else:
            _check(sem.baseline_filters, "")

    def _sc14_groups_have_distinct_defining_filters(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C14: no two groups in a multi-group plan may compile to an identical
        canonical filter.

        Compares the CANONICAL COMPILED filter (semantic_planner._filter_dict_for_scope,
        the exact function compile_to_enriched() calls) — NOT the raw baseline_filters
        dict. This distinction is load-bearing: a live DeepSeek run produced three
        groups whose RAW baseline_filters dicts were all different from each other
        (e.g. {"DXMCI": 1, "DXAD": {"neq": 1}} vs {"DXAD": 1}) — comparing raw dicts
        found no collision — yet all three compiled to the SAME empty filter {}
        because none of them used the "diagnosis" key the compiler actually reads.
        Comparing the compiled output catches this; comparing the raw input does not.
        """
        if len(sem.groups) < 2:
            return
        from alz_agent.agent.semantic_planner import _filter_dict_for_scope

        seen: dict[str, str] = {}
        for group in sem.groups:
            compiled = _filter_dict_for_scope(group)
            key = json.dumps(compiled, sort_keys=True)
            if key in seen:
                hard_errors.append(
                    f"C14: groups '{seen[key]}' and '{group.label}' compile to an "
                    f"IDENTICAL canonical filter ({compiled!r}) — they would resolve to "
                    "the exact same cohort, silently answering one group's question "
                    "twice under two different labels. Set diagnosis_basis + "
                    "clinical_status to distinct values (e.g. clinical_status='normal' "
                    "for CN vs. clinical_status='mci' for MCI)."
                )
            else:
                seen[key] = group.label

    def _sc15_groups_require_nonempty_filters(
        self, sem: SemanticAnalysisPlan, hard_errors: list[str]
    ) -> None:
        """C15: in a multi-group plan, every group's CANONICAL COMPILED filter must
        be non-empty (contain a diagnosis/etiology-defining key: diagnosis, etiology,
        or event_etiology).

        Checks the same canonical compiled output as C14, for the same reason: a
        group's raw baseline_filters dict can be non-empty (e.g. containing raw ADNI
        column names the compiler doesn't recognize) while the ACTUAL executable
        filter it produces is empty. Checking bool(group.baseline_filters) — the
        pre-compile representation — is exactly the check that let the live DeepSeek
        failure through; checking the compiled result is what catches it.
        """
        if len(sem.groups) < 2:
            return
        from alz_agent.agent.semantic_planner import _filter_dict_for_scope

        _DEFINING_KEYS = {"diagnosis", "etiology", "event_etiology"}
        for group in sem.groups:
            compiled = _filter_dict_for_scope(group)
            if not (_DEFINING_KEYS & compiled.keys()):
                hard_errors.append(
                    f"C15: group '{group.label}' compiles to a filter "
                    f"({compiled!r}) with no diagnosis/etiology-defining key — as part "
                    "of a named multi-group comparison this would silently resolve to "
                    "the entire unfiltered cohort. Set diagnosis_basis='clinical_status' "
                    "+ clinical_status=<value>, or diagnosis_basis='etiologic_ad'. "
                    f"(raw baseline_filters was {group.baseline_filters!r} — if this "
                    "looks non-empty, it likely uses a key/shape the compiler does not "
                    "recognize; diagnosis_basis/clinical_status are the only fields "
                    "read for this purpose.)"
                )
