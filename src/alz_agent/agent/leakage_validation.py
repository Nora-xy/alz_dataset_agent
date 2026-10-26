"""Deterministic leakage computation and post-LLM validation.

All leakage assessment is computed from catalog tags — never by the LLM.

Public functions:
  _compute_leakage_finding   — single variable + role → LeakageFinding
  _validate_resolved_intent  — validates a StructuredIntent dict after LLM parsing
  _validate_analysis_plan    — validates an AnalysisPlan dict after LLM planning
"""

from __future__ import annotations

from typing import Any

from alz_agent.schemas import CatalogEntry, LeakageFinding, LeakageTag

# ---------------------------------------------------------------------------
# Role constants
# ---------------------------------------------------------------------------

# Leakage roles that produce a block for diagnosis-defining variables used
# in a predictor context.
_PREDICTOR_ROLES = frozenset({"predictor", "covariate"})
_LABEL_ROLES = frozenset({"label", "event_definition"})
_COHORT_ROLES = frozenset({"cohort_filter"})
_OUTCOME_ROLES = frozenset({"outcome", "value_variable"})

# Map AnalysisPlan field names → leakage role strings used by _compute_leakage_finding.
# This ensures a plan's "event" role gets evaluated as "event_definition" (allowed for
# DIAGNOSIS_DEFINING) rather than the raw "event" string (falls through to warn).
_PLAN_ROLE_TO_LEAKAGE_ROLE: dict[str, str] = {
    "cohort": "cohort_filter",
    "event": "event_definition",
    "value_variable": "value_variable",
    "group_variable": "grouping_variable",
    "outcome_variable": "outcome",
    "covariate": "covariate",
}

# Mapping from StructuredIntent field name → (role string, is_list)
_COLUMN_FIELD_ROLES: list[tuple[str, str, bool]] = [
    ("cohort_column",    "cohort_filter",     False),
    ("group_column",     "grouping_variable", False),
    ("outcome_column",   "outcome",           False),
    ("label_column",     "label",             False),
    ("event_column",     "event_definition",  False),
    ("covariate_columns","covariate",         True),
]


# ---------------------------------------------------------------------------
# Core leakage computation
# ---------------------------------------------------------------------------


def _compute_leakage_finding(
    raw_code: str,
    role: str,
    entry: CatalogEntry | None,
) -> LeakageFinding:
    """Compute a deterministic LeakageFinding from catalog tags and role.

    Always returns a finding — callers can check severity == 'block' to decide
    whether to reject the variable.
    """
    concept = entry.concept if entry else raw_code
    tags = set(entry.leakage_tags) if entry else set()

    # Row-identifier: block in all roles.
    if LeakageTag.ROW_IDENTIFIER in tags:
        return LeakageFinding(
            raw_code=raw_code, concept=concept, role=role,
            leakage_tag="ROW_IDENTIFIER", severity="block",
            reason=(
                f"{raw_code} is a row-identifying variable and cannot appear "
                "in any analytical role."
            ),
            action_taken="caused_abstain",
        )

    if LeakageTag.DIAGNOSIS_DEFINING in tags:
        if role in _COHORT_ROLES:
            return LeakageFinding(
                raw_code=raw_code, concept=concept, role=role,
                leakage_tag="DIAGNOSIS_DEFINING", severity="allow_with_caveat",
                reason=(
                    f"{raw_code} is diagnosis-defining. Using it only to define "
                    "the cohort baseline — not as a predictor."
                ),
                action_taken="allowed_for_cohort_definition",
            )
        if role in _LABEL_ROLES:
            return LeakageFinding(
                raw_code=raw_code, concept=concept, role=role,
                leakage_tag="DIAGNOSIS_DEFINING", severity="allow_with_caveat",
                reason=(
                    f"{raw_code} is diagnosis-defining. Allowed as the prediction "
                    "label or event anchor — this is its intended role."
                ),
                action_taken="allowed_for_cohort_definition",
            )
        if role in _PREDICTOR_ROLES:
            return LeakageFinding(
                raw_code=raw_code, concept=concept, role=role,
                leakage_tag="DIAGNOSIS_DEFINING", severity="block",
                reason=(
                    f"{raw_code} is diagnosis-defining. Blocked as a predictor or "
                    "covariate for diagnosis/conversion tasks — would cause leakage."
                ),
                action_taken="blocked_as_predictor",
            )
        if role in _OUTCOME_ROLES:
            return LeakageFinding(
                raw_code=raw_code, concept=concept, role=role,
                leakage_tag="DIAGNOSIS_DEFINING", severity="allow_with_caveat",
                reason=(
                    f"{raw_code} is diagnosis-defining. Allowed as the outcome for "
                    "descriptive / group-comparison analysis (not prediction)."
                ),
                action_taken="allowed_for_descriptive_analysis",
            )
        # grouping_variable or unknown role
        return LeakageFinding(
            raw_code=raw_code, concept=concept, role=role,
            leakage_tag="DIAGNOSIS_DEFINING", severity="warn",
            reason=(
                f"{raw_code} is diagnosis-defining. Review its use as '{role}'. "
                "Acceptable for stratification; not acceptable as a predictor."
            ),
            action_taken="allowed_for_descriptive_analysis",
        )

    if LeakageTag.CIRCULAR in tags:
        if role in _PREDICTOR_ROLES:
            return LeakageFinding(
                raw_code=raw_code, concept=concept, role=role,
                leakage_tag="CIRCULAR", severity="block",
                reason=(
                    f"{raw_code} is circular (outcome-proximal). Blocked as a predictor."
                ),
                action_taken="blocked_as_predictor",
            )
        if role in _OUTCOME_ROLES:
            return LeakageFinding(
                raw_code=raw_code, concept=concept, role=role,
                leakage_tag="CIRCULAR", severity="warn",
                reason=(
                    f"{raw_code} is circular. Verify it is not a proxy for the "
                    "prediction label before using as an outcome."
                ),
                action_taken="allowed_as_outcome",
            )
        return LeakageFinding(
            raw_code=raw_code, concept=concept, role=role,
            leakage_tag="CIRCULAR", severity="warn",
            reason=f"{raw_code} is circular. Review its use in role '{role}'.",
            action_taken="allowed_for_descriptive_analysis",
        )

    if LeakageTag.HARMONIZATION_FIT_SCOPE in tags:
        return LeakageFinding(
            raw_code=raw_code, concept=concept, role=role,
            leakage_tag="HARMONIZATION_FIT_SCOPE", severity="allow_with_caveat",
            reason=(
                f"{raw_code} was fit on the full ADSP-PHC cohort including outcome "
                "visits. Evaluation performance on the same data will be inflated."
            ),
            action_taken="allowed_for_descriptive_analysis",
        )

    # No leakage concerns.
    action = (
        "allowed_as_outcome" if role in _OUTCOME_ROLES
        else "allowed_for_cohort_definition" if role in _COHORT_ROLES | _LABEL_ROLES
        else "allowed_for_descriptive_analysis"
    )
    return LeakageFinding(
        raw_code=raw_code, concept=concept, role=role,
        leakage_tag="NONE", severity="allowed",
        reason=f"{raw_code} has no leakage concerns for role '{role}'.",
        action_taken=action,
    )


# ---------------------------------------------------------------------------
# Post-LLM validation — StructuredIntent
# ---------------------------------------------------------------------------


def _validate_resolved_intent(
    data: dict[str, Any],
    catalog: list[CatalogEntry],
) -> dict[str, Any]:
    """Validate every LLM-resolved raw_code and compute leakage_assessment.

    Mutates and returns a copy of ``data``. For each resolved column field:
    - Checks the raw_code exists in the catalog (case-insensitive on raw_code).
    - Computes a deterministic LeakageFinding.
    - If severity == 'block': sets field to None, records to unresolved_fields.
    - Otherwise: keeps field, includes finding in leakage_assessment.

    Columns not in the catalog are rejected (marked unresolved) but never
    silently substituted with a different variable.
    """
    data = dict(data)

    # Build case-normalised index: UPPER raw_code → CatalogEntry
    catalog_index: dict[str, CatalogEntry] = {
        e.raw_code.upper(): e for e in catalog
    }

    unresolved: list[str] = list(data.get("unresolved_fields") or [])
    findings: list[dict[str, Any]] = []

    def _process_one(field_name: str, raw_code: str, role: str) -> str | None:
        """Return validated raw_code or None if rejected."""
        if not raw_code or raw_code.upper() == "NONE":
            return None
        entry = catalog_index.get(raw_code.upper())
        if entry is None:
            unresolved.append(f"{field_name}:{raw_code}:not_in_catalog")
            return None
        # Normalise raw_code to canonical casing from catalog
        canonical = entry.raw_code
        finding = _compute_leakage_finding(canonical, role, entry)
        findings.append(finding.model_dump())
        if finding.severity == "block":
            unresolved.append(f"{field_name}:{canonical}:{finding.action_taken}")
            return None
        return canonical

    for field_name, role, is_list in _COLUMN_FIELD_ROLES:
        raw_val = data.get(field_name)
        if is_list:
            validated: list[str] = []
            for item in (raw_val or []):
                result = _process_one(field_name, str(item), role)
                if result:
                    validated.append(result)
            data[field_name] = validated
        else:
            data[field_name] = _process_one(field_name, str(raw_val) if raw_val else "", role)

    data["unresolved_fields"] = unresolved
    data["leakage_assessment"] = findings
    return data


# ---------------------------------------------------------------------------
# Post-LLM validation — AnalysisPlan
# ---------------------------------------------------------------------------


def _validate_analysis_plan(
    plan_data: dict[str, Any],
    catalog: list[CatalogEntry],
) -> dict[str, Any]:
    """Validate every selected_raw_code in an AnalysisPlan dict.

    Works on a plain dict (before Pydantic construction) so nested
    AnalysisRole dicts can be mutated safely.  Returns an updated copy with:
    - validation_status set on each AnalysisRole dict
    - resolution_status corrected when validation fails
    - leakage_assessment populated with LeakageFinding dicts
    - unresolved_items updated for blocked / missing required fields

    The leakage role used for _compute_leakage_finding is derived from
    _PLAN_ROLE_TO_LEAKAGE_ROLE (keyed by the plan field name, e.g. "event")
    so that DIAGNOSIS_DEFINING variables are correctly allowed as event anchors
    but blocked as covariates.
    """
    plan_data = dict(plan_data)
    catalog_index: dict[str, CatalogEntry] = {e.raw_code.upper(): e for e in catalog}
    findings: list[dict[str, Any]] = []
    unresolved: list[str] = list(plan_data.get("unresolved_items") or [])

    def _validate_role(role_dict: dict | None, plan_field: str) -> dict | None:
        if role_dict is None:
            return None
        role_dict = dict(role_dict)
        raw_code = role_dict.get("selected_raw_code")

        if not raw_code or str(raw_code).upper() == "NONE":
            role_dict["selected_raw_code"] = None
            role_dict["validation_status"] = None
            if role_dict.get("required", True):
                role_dict["resolution_status"] = "unresolved"
                label = role_dict.get("researcher_label", plan_field)
                unresolved.append(f"{plan_field}:{label}:not_resolved")
            else:
                role_dict["resolution_status"] = "not_applicable"
            return role_dict

        entry = catalog_index.get(str(raw_code).upper())
        if entry is None:
            role_dict["selected_raw_code"] = None
            role_dict["validation_status"] = "invalid_not_in_catalog"
            role_dict["resolution_status"] = "unresolved"
            unresolved.append(f"{plan_field}:{raw_code}:not_in_catalog")
            return role_dict

        canonical = entry.raw_code
        role_dict["selected_raw_code"] = canonical
        leakage_role = _PLAN_ROLE_TO_LEAKAGE_ROLE.get(plan_field, role_dict.get("role", plan_field))
        finding = _compute_leakage_finding(canonical, leakage_role, entry)
        findings.append(finding.model_dump())

        if finding.severity == "block":
            role_dict["selected_raw_code"] = None
            role_dict["validation_status"] = "invalid_blocked"
            role_dict["resolution_status"] = "unresolved"
            unresolved.append(f"{plan_field}:{canonical}:{finding.action_taken}")
        else:
            role_dict["validation_status"] = "valid"
            role_dict["resolution_status"] = "resolved"

        return role_dict

    for field in ("cohort", "event", "value_variable", "group_variable", "outcome_variable"):
        plan_data[field] = _validate_role(plan_data.get(field), field)

    validated_covariates: list[dict] = []
    for cov in (plan_data.get("covariates") or []):
        result = _validate_role(cov, "covariate")
        if result is not None:
            validated_covariates.append(result)
    plan_data["covariates"] = validated_covariates

    plan_data["unresolved_items"] = unresolved
    plan_data["leakage_assessment"] = findings
    return plan_data
