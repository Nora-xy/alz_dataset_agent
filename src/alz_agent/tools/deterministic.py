"""P3 — Six deterministic tools over (catalog, DataBackend).

No tool may call an LLM. Every returned number must trace to a ToolResult
evidence entry. Tools abstain (grounded=False) when ungrounded.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from alz_agent.backend.interface import DataBackend
from alz_agent.schemas import (
    CatalogEntry,
    CoverageStats,
    DataEvidence,
    DescriptiveStatResult,
    FilterSpec,
    IdSpace,
    LeakageRecord,
    LeakageTag,
    Modality,
    RegressionResult,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Value types for tool results
# ---------------------------------------------------------------------------


class VariableLookupResult(BaseModel):
    variable_id: str
    concept: str
    source_dataset: str
    raw_code: str
    modality: str
    value_coding: dict[str, Any] = Field(default_factory=dict)
    harmonization_transform: str | None = None
    leakage_tags: list[str] = Field(default_factory=list)
    leakage_reasons: list[str] = Field(default_factory=list)
    coverage: CoverageStats | None = None


class LeakageFlag(BaseModel):
    variable_id: str
    tags: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class LeakageCheckResult(BaseModel):
    label_def: str
    flagged: list[LeakageFlag] = Field(default_factory=list)
    clean: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)  # variable_ids absent from catalog


class ModelingDesignResult(BaseModel):
    index_visit_def: str
    horizon_years: int
    outcome_label_def: str
    id_space: str
    feature_whitelist: list[str] = Field(default_factory=list)
    leakage_blacklist: list[str] = Field(default_factory=list)
    leakage_records: list[LeakageRecord] = Field(default_factory=list)
    eligible_n: int | None = None  # None when suppressed
    feasibility: dict[str, Any] = Field(default_factory=dict)


class StatResult(BaseModel):
    variable_id: str
    stat_type: str
    value: float
    n: int
    event_anchor: str | None = None


# ---------------------------------------------------------------------------
# Alias map — common researcher phrases to known raw_codes
# ---------------------------------------------------------------------------
# These aliases ensure that short, unambiguous terms like "age" resolve to the
# correct NACC variable even when the LLM returns selected_raw_code=null.
# Only add aliases that are semantically unambiguous (one obvious target).

_VARIABLE_ALIASES: dict[str, str] = {
    # NACCAGE — visit-level age (NACC derived variable)
    "age":                  "NACCAGE",
    "age at visit":         "NACCAGE",
    "subject age":          "NACCAGE",
    "participant age":      "NACCAGE",
    "baseline age":         "NACCAGE",
    "age when diagnosed":   "NACCAGE",
    "age at diagnosis":     "NACCAGE",
    "visit age":            "NACCAGE",
    # NACCMMSE — NACC-derived MMSE total
    "mmse":                 "NACCMMSE",
    "mmse score":           "NACCMMSE",
    "mini mental":          "NACCMMSE",
    "mini-mental":          "NACCMMSE",
    # SEX
    "sex":                  "SEX",
    "gender":               "SEX",
    # Education
    "education":            "EDUC",
    "years of education":   "EDUC",
    "years education":      "EDUC",
    # CVHATT — history of heart attack / cardiac arrest (NACC UDS vascular history)
    "heart attack":                 "CVHATT",
    "heart attack history":         "CVHATT",
    "history of heart attack":      "CVHATT",
    "cardiac arrest":               "CVHATT",
    "cardiac arrest history":       "CVHATT",
    "history of cardiac arrest":    "CVHATT",
    "myocardial infarction":        "CVHATT",
}

# Short but important tokens that must never be dropped from keyword filtering.
# The standard threshold is len >= 3; these override it for 2-char terms.
_SHORT_TOKEN_EXCEPTIONS: frozenset[str] = frozenset({
    "ad", "mr",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_index(catalog: list[CatalogEntry]) -> dict[str, CatalogEntry]:
    return {e.variable_id: e for e in catalog}


def _modalities_for_id_space(catalog: list[CatalogEntry], id_space: IdSpace) -> set[str]:
    """Modality values present in the catalog for a given id_space."""
    return {e.modality.value for e in catalog if e.id_space == id_space}


_LEAKY_TAGS: frozenset[LeakageTag] = frozenset({
    LeakageTag.DIAGNOSIS_DEFINING,
    LeakageTag.CIRCULAR,
    LeakageTag.HARMONIZATION_FIT_SCOPE,
})


# ---------------------------------------------------------------------------
# Tool 1: lookup_variable
# ---------------------------------------------------------------------------


def lookup_variable(
    query: str,
    catalog: list[CatalogEntry],
    *,
    debug: bool = False,
) -> ToolResult[VariableLookupResult]:
    """Return definition, coding, provenance, and leakage tags for a variable or concept.

    Search order (stops at first hit):
      0. Alias map — common researcher phrases mapped to known raw_codes
      1. Exact match on variable_id  (e.g. "NACC.NACCAGE")
      2. Exact match on raw_code     (e.g. "NACCAGE", "naccage")
      3. Exact match on concept      (e.g. "age of subject at time of uds visit")
      4. Substring: query appears verbatim in concept or description
      5. Keyword: every non-stopword query token appears in concept + description
         (tokens >= 3 chars, plus _SHORT_TOKEN_EXCEPTIONS for 2-char terms)

    Abstains when no pass finds a match.
    When debug=True, prints candidate list and match trace to stderr.
    """
    import sys
    q = query.strip().lower()
    match: CatalogEntry | None = None
    pass_hit: str = "none"

    # Pass 0: alias map — resolve common researcher phrases directly
    alias_code = _VARIABLE_ALIASES.get(q)
    if alias_code:
        alias_upper = alias_code.upper()
        for entry in catalog:
            if entry.raw_code.upper() == alias_upper:
                match = entry
                pass_hit = "alias"
                break
        if debug and match is None:
            print(
                f"[lookup_variable debug] Pass 0 alias: '{q}' → {alias_code!r} "
                f"but {alias_code!r} not found in catalog",
                file=sys.stderr,
            )

    # Pass 1: exact variable_id
    if match is None:
        for entry in catalog:
            if entry.variable_id.lower() == q:
                match = entry
                pass_hit = "variable_id"
                break

    # Pass 2: exact raw_code
    if match is None:
        for entry in catalog:
            if entry.raw_code.lower() == q:
                match = entry
                pass_hit = "raw_code"
                break

    # Pass 3: exact concept
    if match is None:
        for entry in catalog:
            if entry.concept.lower() == q:
                match = entry
                pass_hit = "concept"
                break

    # Pass 4: substring — query phrase appears in concept or description
    if match is None and len(q) >= 3:
        for entry in catalog:
            haystack = f"{entry.raw_code} {entry.concept} {entry.description}".lower()
            if q in haystack:
                match = entry
                pass_hit = "substring"
                break

    # Pass 5: keyword — all non-trivial query words appear in concept + description
    # Includes _SHORT_TOKEN_EXCEPTIONS so "ad", "mr" etc. are never dropped.
    if match is None and len(q) >= 2:
        _STOPWORDS = frozenset({"a", "an", "the", "of", "at", "in", "on", "for",
                                 "to", "is", "or", "and", "by", "from", "with"})
        tokens = [
            t for t in q.split()
            if t not in _STOPWORDS and (len(t) >= 3 or t in _SHORT_TOKEN_EXCEPTIONS)
        ]
        if tokens:
            for entry in catalog:
                haystack = f"{entry.raw_code} {entry.concept} {entry.description}".lower()
                if all(tok in haystack for tok in tokens):
                    match = entry
                    pass_hit = "keyword"
                    break

    if debug:
        if match is not None:
            print(
                f"[lookup_variable debug] query={q!r} → {match.raw_code!r} "
                f"via pass={pass_hit} | concept={match.concept!r} | "
                f"desc={match.description[:80]!r}",
                file=sys.stderr,
            )
        else:
            sample = [
                f"{e.raw_code}({e.concept!r},{e.description[:40]!r})"
                for e in catalog[:10]
            ]
            print(
                f"[lookup_variable debug] query={q!r} UNRESOLVED. "
                f"Candidates (first 10): {sample}",
                file=sys.stderr,
            )

    if match is None:
        return ToolResult.abstain(
            f"Variable or concept '{query}' not found in catalog"
        )

    return ToolResult.ok(
        VariableLookupResult(
            variable_id=match.variable_id,
            concept=match.concept,
            source_dataset=match.source_dataset.value,
            raw_code=match.raw_code,
            modality=match.modality.value,
            value_coding=dict(match.value_coding),
            harmonization_transform=match.harmonization_transform,
            leakage_tags=[t.value for t in match.leakage_tags],
            leakage_reasons=list(match.leakage_reasons),
            coverage=match.coverage,
        ),
        evidence=[DataEvidence(
            query_spec={"query": query},
            aggregate_ref=f"catalog.lookup:{match.variable_id}",
        )],
    )


# ---------------------------------------------------------------------------
# Tool 2: compute_cohort_size
# ---------------------------------------------------------------------------


def compute_cohort_size(
    filter_spec: FilterSpec,
    backend: DataBackend,
) -> ToolResult[int]:
    """Return the eligible participant count via backend.count_eligible."""
    return backend.count_eligible(filter_spec)


# ---------------------------------------------------------------------------
# Tool 3: assess_modality_overlap
# ---------------------------------------------------------------------------


def assess_modality_overlap(
    modalities: list[str],
    filter_spec: FilterSpec,
    backend: DataBackend,
    catalog: list[CatalogEntry],
) -> ToolResult[dict[str, int | None]]:
    """Return intersection counts for all modality combinations.

    Abstains if any requested modality name is invalid or absent from the catalog
    for the given id_space (marking a release gap).
    """
    id_space = filter_spec.id_space
    valid_values = {m.value for m in Modality}
    invalid = [m for m in modalities if m not in valid_values]
    if invalid:
        return ToolResult.abstain(
            f"Unknown modality name(s): {invalid}. Valid: {sorted(valid_values)}"
        )

    catalog_modalities = _modalities_for_id_space(catalog, id_space)
    absent = [m for m in modalities if m not in catalog_modalities]
    if absent:
        return ToolResult.abstain(
            f"Modalities absent from catalog for id_space={id_space.value}: {absent}"
        )

    return backend.modality_intersection(modalities, filter_spec)


# ---------------------------------------------------------------------------
# Tool 4: assess_longitudinal_feasibility
# ---------------------------------------------------------------------------


def assess_longitudinal_feasibility(
    horizon: int,
    filter_spec: FilterSpec,
    backend: DataBackend,
) -> ToolResult[dict[str, int | None]]:
    """Return follow-up distribution (sufficient/censored/insufficient) for the horizon."""
    if horizon <= 0:
        return ToolResult.abstain("horizon must be a positive integer (years)")
    return backend.followup_distribution(horizon, filter_spec)


# ---------------------------------------------------------------------------
# Tool 5: check_leakage
# ---------------------------------------------------------------------------


def check_leakage(
    feature_set: list[str],
    label_def: str,
    catalog: list[CatalogEntry],
) -> ToolResult[LeakageCheckResult]:
    """Audit feature_set for leakage risk relative to label_def.

    Always grounded. Variables absent from the catalog are reported in 'unknown'.
    """
    index = _build_index(catalog)
    flagged: list[LeakageFlag] = []
    clean: list[str] = []
    unknown: list[str] = []

    for var_id in feature_set:
        entry = index.get(var_id)
        if entry is None:
            unknown.append(var_id)
            continue
        risky = [t for t in entry.leakage_tags if t in _LEAKY_TAGS]
        if risky:
            flagged.append(LeakageFlag(
                variable_id=var_id,
                tags=[t.value for t in risky],
                reasons=list(entry.leakage_reasons),
            ))
        else:
            clean.append(var_id)

    result = LeakageCheckResult(
        label_def=label_def,
        flagged=flagged,
        clean=clean,
        unknown=unknown,
    )
    return ToolResult.ok(
        result,
        evidence=[DataEvidence(
            query_spec={"feature_set_size": len(feature_set), "label_def": label_def},
            aggregate_ref=(
                f"check_leakage: {len(flagged)} flagged, "
                f"{len(clean)} clean, {len(unknown)} unknown"
            ),
        )],
    )


# ---------------------------------------------------------------------------
# Tool 6: suggest_modeling_design
# ---------------------------------------------------------------------------


def suggest_modeling_design(
    filter_spec: FilterSpec,
    horizon: int,
    catalog: list[CatalogEntry],
    backend: DataBackend,
) -> ToolResult[ModelingDesignResult]:
    """Propose index visit, horizon, label, and feature whitelist.

    Calls the other tools; does not re-derive any numbers itself.
    Abstains if the filter_spec is fundamentally invalid (not just suppressed).
    """
    if horizon <= 0:
        return ToolResult.abstain("horizon must be a positive integer (years)")

    # Step 1: cohort size — propagate only non-suppression abstains
    size_result = compute_cohort_size(filter_spec, backend)
    if not size_result.grounded:
        reason = size_result.abstain_reason or ""
        if "suppressed" not in reason.lower():
            return ToolResult.abstain(
                f"Cannot suggest design: cohort size error — {reason}"
            )
    eligible_n = size_result.value  # None if suppressed

    # Step 2: longitudinal feasibility
    feas_result = assess_longitudinal_feasibility(horizon, filter_spec, backend)
    feasibility: dict[str, Any] = {}
    if feas_result.grounded and feas_result.value is not None:
        feasibility = dict(feas_result.value)

    # Step 3: leakage check on all catalog vars for this id_space
    id_space = filter_spec.id_space
    all_vars = [e.variable_id for e in catalog if e.id_space == id_space]
    label_def = _derive_label_def(filter_spec, horizon)
    leakage_result = check_leakage(all_vars, label_def, catalog)
    leakage_val = leakage_result.value  # always grounded

    feature_whitelist = list(leakage_val.clean) if leakage_val else []
    leakage_blacklist = [f.variable_id for f in leakage_val.flagged] if leakage_val else []

    # Build full leakage records (raw_code, concept, tags, reason, source, excluded_from)
    catalog_index = {e.variable_id: e for e in catalog}
    leakage_records: list[LeakageRecord] = []
    for flag in (leakage_val.flagged if leakage_val else []):
        entry = catalog_index.get(flag.variable_id)
        raw_code = entry.raw_code if entry else flag.variable_id
        display_name = entry.concept if entry else flag.variable_id
        description = entry.description if entry else ""
        source_dataset = entry.source_dataset.value if entry else ""
        tags = list(flag.tags)
        tag = ", ".join(tags) if tags else "unknown"
        reason = "; ".join(flag.reasons) if flag.reasons else "see leakage rules"
        # DIAGNOSIS_DEFINING / CIRCULAR → label leakage risk; HARMONIZATION_FIT_SCOPE → feature
        label_tags = {"DIAGNOSIS_DEFINING", "CIRCULAR"}
        feature_tags = {"HARMONIZATION_FIT_SCOPE"}
        has_label = bool(set(tags) & label_tags)
        has_feature = bool(set(tags) & feature_tags)
        if has_label and has_feature:
            excluded_from = "both"
        elif has_label:
            excluded_from = "both"  # these vars are excluded entirely (they are the outcome)
        else:
            excluded_from = "features"
        leakage_records.append(LeakageRecord(
            variable_id=flag.variable_id,
            raw_code=raw_code,
            display_name=display_name,
            description=description,
            tag=tag,
            tags=tags,
            reason=reason,
            source_dataset=source_dataset,
            excluded_from=excluded_from,
        ))

    design = ModelingDesignResult(
        index_visit_def=_derive_index_visit_def(filter_spec),
        horizon_years=horizon,
        outcome_label_def=label_def,
        id_space=id_space.value,
        feature_whitelist=feature_whitelist,
        leakage_blacklist=leakage_blacklist,
        leakage_records=leakage_records,
        eligible_n=eligible_n,
        feasibility=feasibility,
    )

    evidence: list[Any] = []
    evidence.extend(size_result.evidence)
    if feas_result.grounded:
        evidence.extend(feas_result.evidence)
    evidence.extend(leakage_result.evidence)

    return ToolResult.ok(design, evidence=evidence)


# ---------------------------------------------------------------------------
# Tool 7: compute_variable_stat
# ---------------------------------------------------------------------------


def compute_variable_stat(
    variable_id: str,
    stat_type: str,
    filter_spec: FilterSpec,
    backend: DataBackend,
    event_anchor: str | None = None,
) -> ToolResult[StatResult]:
    """Compute mean/median/std/min/max of a catalog variable over a filtered cohort.

    Resolves the aggregate by delegating to backend.aggregate_stat, then wraps the
    result in a StatResult for downstream narration and hallucination-guard accounting.

    Abstains when:
    - stat_type is not in {mean, median, std, min, max}
    - variable_id not found in the backend's data files
    - eligible n < SUPPRESSION_THRESHOLD
    - backend / id_space combination is unsupported
    """
    raw = backend.aggregate_stat(variable_id, stat_type, filter_spec, event_anchor)
    if not raw.grounded:
        return ToolResult.abstain(raw.abstain_reason or "aggregate_stat abstained")

    d = raw.value
    return ToolResult.ok(
        StatResult(
            variable_id=variable_id,
            stat_type=d["stat_type"],
            value=d["value"],
            n=d["n"],
            event_anchor=event_anchor,
        ),
        evidence=raw.evidence,
    )


# ---------------------------------------------------------------------------
# Tool 8: compute_descriptive_stat
# ---------------------------------------------------------------------------


def compute_descriptive_stat(
    variable_id: str,
    filter_spec: FilterSpec,
    backend: DataBackend,
    event_anchor: str | None = None,
) -> ToolResult[DescriptiveStatResult]:
    """Compute full descriptive statistics (mean, median, SD, IQR, count, proportion).

    Supported event_anchor values:
      None / 'baseline'            — first visit per participant
      'first_diagnosis'            — first visit matching the filter's diagnosis condition
      'last_visit'                 — last visit per participant
      'first_conversion_visit'     — first visit where NACCUDSD >= 4

    Abstains when:
    - n eligible < SUPPRESSION_THRESHOLD
    - variable_id column not found in the backend's data files
    - event_anchor / id_space combo is unsupported
    """
    raw = backend.descriptive_stats(variable_id, filter_spec, event_anchor)
    if not raw.grounded:
        return ToolResult.abstain(raw.abstain_reason or "descriptive_stats abstained")

    d = raw.value
    return ToolResult.ok(
        DescriptiveStatResult(
            variable_id=variable_id,
            n_total=d["n_total"],
            n_missing=d["n_missing"],
            count=d["count"],
            mean=d.get("mean"),
            median=d.get("median"),
            sd=d.get("sd"),
            q1=d.get("q1"),
            q3=d.get("q3"),
            iqr=d.get("iqr"),
            min_val=d.get("min_val"),
            max_val=d.get("max_val"),
            proportion=d.get("proportion"),
            event_anchor=event_anchor,
        ),
        evidence=raw.evidence,
    )


# ---------------------------------------------------------------------------
# Tool 9: run_adjusted_regression
# ---------------------------------------------------------------------------


def run_adjusted_regression(
    outcome_variable_id: str,
    outcome_col: str,
    outcome_label: str,
    covariate_variable_ids: list[str],
    covariate_cols: list[str],
    filter_spec: FilterSpec,
    analysis_engine: Any,
    regression_type: str = "linear",
) -> ToolResult[RegressionResult]:
    """Run OLS or logistic regression, returning aggregate coefficients only.

    Delegates entirely to LocalAnalysisEngine.adjusted_regression.
    Abstains when analysis_engine is None (fixture / non-local backend).

    Args:
        outcome_variable_id:    Catalog id of the outcome variable.
        outcome_col:            Raw NACC column name for outcome.
        outcome_label:          Human-readable outcome name.
        covariate_variable_ids: Catalog ids of predictor variables (parallel with covariate_cols).
        covariate_cols:         Raw NACC column names for predictors.
        filter_spec:            Cohort filter.
        analysis_engine:        LocalAnalysisEngine instance or None.
        regression_type:        'linear' (OLS) or 'logistic'.
    """
    if analysis_engine is None:
        return ToolResult.abstain(
            "run_adjusted_regression requires a LocalAnalysisEngine (not available for fixture/CI backend)"
        )
    return analysis_engine.adjusted_regression(
        outcome_col=outcome_col,
        outcome_variable_id=outcome_variable_id,
        outcome_label=outcome_label,
        covariate_cols=covariate_cols,
        covariate_variable_ids=covariate_variable_ids,
        filter_spec=filter_spec,
        regression_type=regression_type,
    )


# ---------------------------------------------------------------------------
# Tool 10: explore_catalog
# ---------------------------------------------------------------------------


class CatalogMatch(BaseModel):
    variable_id: str
    concept: str
    description: str
    modality: str
    family: str | None = None
    variant: str | None = None


def explore_catalog(
    keyword: str,
    catalog: list[CatalogEntry],
    id_space: IdSpace | None = None,
) -> ToolResult[list[CatalogMatch]]:
    """Return ALL catalog entries matching `keyword`, not just the single best hit.

    Unlike lookup_variable (which stops at the first match across several
    precision passes), this is reconnaissance: a researcher navigating a
    fragmented, multi-table dataset like ADNI needs to see every variant that
    touches a concept (e.g. "ptau") across every table before picking one to
    query with compute_variable_stat/describe_variable_family.

    Matching is a case-insensitive substring search over raw_code, concept,
    description, family, and variant. Abstains if nothing matches.
    """
    kw = keyword.strip().lower()
    if not kw:
        return ToolResult.abstain("keyword must be a non-empty string")

    matches: list[CatalogEntry] = []
    for entry in catalog:
        if id_space is not None and entry.id_space != id_space:
            continue
        haystack = " ".join(
            part for part in (
                entry.raw_code, entry.concept, entry.description,
                entry.family, entry.variant,
            ) if part
        ).lower()
        if kw in haystack:
            matches.append(entry)

    if not matches:
        scope = f" for id_space={id_space.value}" if id_space is not None else ""
        return ToolResult.abstain(f"No catalog entries match '{keyword}'{scope}")

    matches.sort(key=lambda e: e.variable_id)
    results = [
        CatalogMatch(
            variable_id=e.variable_id,
            concept=e.concept,
            description=e.description,
            modality=e.modality.value,
            family=e.family,
            variant=e.variant,
        )
        for e in matches
    ]
    return ToolResult.ok(
        results,
        evidence=[DataEvidence(
            query_spec={"keyword": keyword, "id_space": id_space.value if id_space else None},
            aggregate_ref=f"catalog.explore('{keyword}'): {len(results)} matches",
        )],
    )


# ---------------------------------------------------------------------------
# Tool 11: describe_variable_family
# ---------------------------------------------------------------------------


class VariantCoverage(BaseModel):
    variant: str
    variable_id: str
    n: int | None  # None means suppressed (0 < n < SUPPRESSION_THRESHOLD)


class VariableFamilyResult(BaseModel):
    family: str
    id_space: str
    variants: list[VariantCoverage] = Field(default_factory=list)


def describe_variable_family(
    family: str,
    id_space: IdSpace,
    catalog: list[CatalogEntry],
    backend: DataBackend,
    filter_spec: FilterSpec | None = None,
) -> ToolResult[VariableFamilyResult]:
    """Group catalog entries sharing a `family` tag by `variant` and report
    per-variant distinct-subject coverage.

    This generalizes questions like "do we have enough PTAU data, what subtype
    (181/217), what kind (CSF/plasma)" to any tagged family — the grouping is
    driven entirely by whatever family/variant tags the catalog parsers have
    assigned (see CatalogEntry.family / .variant), not a hardcoded biomarker
    list. Abstains when no catalog entry carries the requested family tag for
    id_space (e.g. family='ptau217' has no CSF variant locally — that absence
    is surfaced by simply not appearing, never guessed).
    """
    fs = filter_spec or FilterSpec(id_space=id_space)
    if fs.id_space != id_space:
        return ToolResult.abstain(
            f"id_space mismatch: requested id_space={id_space.value} but "
            f"filter_spec.id_space={fs.id_space.value}"
        )

    matches = [e for e in catalog if e.family == family and e.id_space == id_space]
    if not matches:
        return ToolResult.abstain(
            f"No catalog entries tagged family='{family}' for id_space={id_space.value} "
            "— not available in local data"
        )

    variable_ids = [e.variable_id for e in matches]
    counts = backend.variant_subject_counts(variable_ids, fs)
    if not counts.grounded:
        return ToolResult.abstain(
            counts.abstain_reason or "variant_subject_counts abstained"
        )

    by_variable = {e.variable_id: e for e in matches}
    variants = [
        VariantCoverage(
            variant=by_variable[vid].variant or vid,
            variable_id=vid,
            n=n,
        )
        for vid, n in counts.value.items()
    ]
    variants.sort(key=lambda v: v.variant)

    return ToolResult.ok(
        VariableFamilyResult(family=family, id_space=id_space.value, variants=variants),
        evidence=counts.evidence,
    )


def _derive_label_def(filter_spec: FilterSpec, horizon: int) -> str:
    space_label = "NACCUDSD=4" if filter_spec.id_space == IdSpace.A else "DIAGNOSIS=3"
    return (
        f"conversion to all-cause dementia ({space_label}) "
        f"within {horizon} year(s) of index visit"
    )


def _derive_index_visit_def(filter_spec: FilterSpec) -> str:
    parts: list[str] = []
    if filter_spec.baseline_diagnosis:
        parts.append(f"baseline_diagnosis in [{'/'.join(filter_spec.baseline_diagnosis)}]")
    if filter_spec.baseline_etiology:
        parts.append(f"baseline_etiology in [{'/'.join(filter_spec.baseline_etiology)}]")
    if filter_spec.visit_window:
        lo, hi = filter_spec.visit_window
        parts.append(f"visit_window=[{lo},{hi}]")
    return "Index visit: " + ("; ".join(parts) if parts else "all participants")
