"""Typed schemas for the Dataset Expert Agent (P0 contracts)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class LeakageTag(str, Enum):
    DIAGNOSIS_DEFINING = "DIAGNOSIS_DEFINING"
    CIRCULAR = "CIRCULAR"
    HARMONIZATION_FIT_SCOPE = "HARMONIZATION_FIT_SCOPE"
    ROW_IDENTIFIER = "ROW_IDENTIFIER"
    NONE = "NONE"


class Modality(str, Enum):
    COGNITION = "cognition"
    VASCULAR = "vascular"
    MRI = "mri"
    PET = "pet"
    FLUID_BIOMARKER = "fluid_biomarker"
    NEUROPATH = "neuropath"
    DEMOGRAPHIC = "demographic"
    OTHER = "other"


class IdSpace(str, Enum):
    A = "A"  # NACCID
    B = "B"  # RID / PTID


class Tier(str, Enum):
    TIER1 = "TIER1"
    TIER2 = "TIER2"
    TIER3 = "TIER3"


class CohortRole(str, Enum):
    TRAINING = "training"
    HARMONIZED_FEATURE_SOURCE = "harmonized_feature_source"
    EXTERNAL_VALIDATION_CANDIDATE = "external_validation_candidate"


class SourceDataset(str, Enum):
    NACC = "NACC"
    ADNI = "ADNI"
    ADSP_PHC = "ADSP_PHC"


class PopulationUnit(str, Enum):
    PARTICIPANT = "participant"
    VISIT_ROW = "visit_row"
    VARIABLE = "variable"
    DATASET = "dataset"


class CohortScope(str, Enum):
    BASELINE = "baseline"
    ANY_VISIT = "any_visit"
    FIRST_EVENT = "first_event"
    FOLLOWUP_WINDOW = "followup_window"
    PRE_INDEX = "pre_index"
    UNCLEAR = "unclear"


class EtiologyDefinition(str, Enum):
    ETIOLOGIC_AD = "etiologic_ad"
    ALL_CAUSE_DEMENTIA = "all_cause_dementia"
    ANY_DIAGNOSIS = "any_diagnosis"
    UNSPECIFIED = "unspecified"


# ---------------------------------------------------------------------------
# CoverageStats
# ---------------------------------------------------------------------------


class CoverageStats(BaseModel):
    n_available: int
    longitudinal_availability: int  # n with >= 2 visits
    missingness_rate: float  # fraction in [0.0, 1.0]
    last_computed: datetime | None = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# CatalogEntry
# ---------------------------------------------------------------------------


class CatalogEntry(BaseModel):
    variable_id: str
    concept: str  # harmonized name (lowercased raw_code for NACC/ADNI; short descriptor after RDD enrichment)
    source_dataset: SourceDataset
    id_space: IdSpace
    raw_code: str
    release_version: str
    harmonization_transform: str | None = None
    value_coding: dict[str, Any] = Field(default_factory=dict)
    description: str = ""  # human-readable text from source dictionary (RDD / DATADIC / DD xlsx)
    modality: Modality = Modality.OTHER
    leakage_tags: list[LeakageTag] = Field(default_factory=list)
    leakage_reasons: list[str] = Field(default_factory=list)
    coverage: CoverageStats | None = None  # populated by P2, not P1
    family: str | None = None  # groups variants measuring "the same thing", e.g. "ptau"
    variant: str | None = None  # what distinguishes entries within a family, e.g. "plasma_raw_217"

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class DataEvidence(BaseModel):
    kind: Literal["DATA"] = "DATA"
    query_spec: dict[str, Any]
    aggregate_ref: str  # human-readable reference to the aggregate result

    # Provenance (optional; defaulted so existing callers are unaffected).
    # Names the concrete data origin of the aggregate so narration can report it
    # transparently instead of implying an unqualified "diagnosis" source.
    source_table: str | None = None   # e.g. "PTDEMOG", "DXSUM", "ADSP_PHC_DEMODX"
    source_field: str | None = None   # e.g. "PTDOB", "PHC_Age_Diagnosis"
    # "phc_raw_column"/"phc_harmonized" are the ADSP-PHC-specific labels (a raw
    # per-phase harmonization-input column vs. the cross-phase harmonized derived
    # column) — more precise than the generic "raw"/"harmonized" for that release,
    # which is kept for other sources (e.g. the age-at-event derived-vs-raw cross-check).
    source_level: Literal["raw", "harmonized", "phc_raw_column", "phc_harmonized"] | None = None
    # Basis for a diagnosis anchor, e.g. "DXAD/DXMDUE/DXDDUE etiologic AD" vs
    # "PHC_Diagnosis all-cause dementia".  None when the evidence is not
    # diagnosis-anchored.
    diagnosis_basis: str | None = None

    # Baseline/event measurement-alignment provenance (Space B biomarker stats).
    # baseline_alignment_rule: plain-language description of how the measure row was
    # aligned to the requested visit (e.g. "measure row aligned to each RID's own
    # baseline DXSUM visit via matching VISCODE; a participant with no measure row at
    # that visit is excluded, never substituted with their first available draw").
    # matched_n: participants in the eligible cohort who HAD a measure-table row at the
    # aligned visit (regardless of whether the specific field was non-null there).
    # unmatched_n: eligible_n - matched_n — participants excluded because they had no
    # measure-table row at the aligned visit at all. All None when not anchor-aligned
    # (e.g. Space A, or Space B non-baseline/non-diagnosis anchors).
    baseline_alignment_rule: str | None = None
    matched_n: int | None = None
    unmatched_n: int | None = None

    model_config = {"frozen": True}


class DocEvidence(BaseModel):
    kind: Literal["DOC"] = "DOC"
    source: str      # e.g. "NACC/uds3-rdd.pdf"
    locator: str     # page, section, table, or variable name
    snippet: str     # short verbatim excerpt

    model_config = {"frozen": True}


Evidence = Annotated[DataEvidence | DocEvidence, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# ToolResult[T]
# ---------------------------------------------------------------------------

T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    grounded: bool
    value: T | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    abstain_reason: str | None = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_grounded_consistency(self) -> "ToolResult[T]":
        if not self.grounded:
            if self.value is not None:
                raise ValueError(
                    "A grounded=False ToolResult must have value=None; "
                    f"got value={self.value!r}"
                )
            if self.abstain_reason is None:
                raise ValueError(
                    "A grounded=False ToolResult must supply abstain_reason"
                )
        return self

    @classmethod
    def abstain(cls, reason: str) -> "ToolResult[T]":
        return cls(grounded=False, value=None, evidence=[], abstain_reason=reason)

    @classmethod
    def ok(cls, value: T, evidence: list[Evidence] | None = None) -> "ToolResult[T]":
        return cls(grounded=True, value=value, evidence=evidence or [])


# ---------------------------------------------------------------------------
# FilterSpec  (used by DataBackend — defined here for shared import)
# ---------------------------------------------------------------------------


class FilterSpec(BaseModel):
    """Structured query specification for DataBackend methods.

    baseline_diagnosis — cognitive status only (not disease etiology).
        Space A (NACC) uses NACCUDSD codes:
            "normal" / "nc" / "nl" / "1"            → NACCUDSD = 1 (Normal cognition)
            "impaired not mci" / "impaired-not-mci"
             / "impaired_not_mci" / "2"              → NACCUDSD = 2 (Impaired-not-MCI)
            "mci" / "3"                              → NACCUDSD = 3 (MCI)
            "dementia" / "4"                         → NACCUDSD = 4 (All-cause dementia)
        Space B (ADNI) uses DXSUM.DIAGNOSIS codes:
            "normal" / "cn" / "nc" / "nl" / "1"     → DIAGNOSIS = 1 (CN)
            "mci" / "2"                              → DIAGNOSIS = 2 (MCI)
            "dementia" / "3"                         → DIAGNOSIS = 3 (Dementia)
        "ad" is NOT accepted as a baseline_diagnosis label in either space.
        NACCUDSD=4 and DIAGNOSIS=3 are all-cause dementia; AD etiologic
        attribution is a distinct concept captured by baseline_etiology.

    baseline_etiology — disease attribution / etiology at the FIRST (baseline) visit.
        Supported labels (all mean AD etiologic attribution):
            "ad" / "alzheimers" / "alzheimer" / "alzheimer's disease"
        Space A (NACC): uses NACCALZD==1 (presumptive etiologic diagnosis
            of Alzheimer's disease) on the participant's first NACCVNUM visit.
        Space B (ADNI): uses DXMDUE==1 OR DXDDUE==1 OR DXAD==1 on the
            baseline DXSUM row (earliest EXAMDATE, or VISCODE fallback).
        Unknown labels abstain deterministically.
        Use this for "participants with AD etiology at enrollment / baseline."

    event_etiology — disease attribution / etiology at ANY visit (longitudinal).
        Same accepted labels as baseline_etiology.
        Space A (NACC): participant is eligible if NACCALZD==1 appears on ANY
            NACCVNUM visit, not just the first. Combined with
            event_anchor="first_diagnosis" in aggregate_stat / descriptive_stats,
            this selects the FIRST visit where NACCALZD==1 per participant as the
            measurement row — giving the correct "age at first etiologic AD
            diagnosis" over all participants who ever received the designation.
        Distinct from baseline_etiology: baseline filters to prevalent-at-enrollment
        AD (n=19,576 in NACC); event_etiology also includes incident cases diagnosed
        at later visits (n=25,639 in NACC, mean age at first AD visit = 74.81).
        Do not use baseline_etiology to answer "among participants who ever had AD"
        or "mean age at first AD diagnosis" — use event_etiology instead.

    label_certainty is reserved for future use; passing a non-None value
    currently causes the backend to abstain.
    """

    baseline_diagnosis: list[str] | None = None
    baseline_etiology: list[str] | None = None  # AD etiology at first visit only
    event_etiology: list[str] | None = None      # AD etiology at any visit (longitudinal)
    visit_window: tuple[int, int] | None = None  # (min_vnum, max_vnum)
    required_modalities: list[str] | None = None
    label_certainty: str | None = None  # reserved; non-None → backend abstains
    id_space: IdSpace = IdSpace.A  # default to Space A

    model_config = {"frozen": True}

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = set(cls.model_fields)
            unknown = set(data) - known
            if unknown:
                raise ValueError(f"Unknown FilterSpec fields: {sorted(unknown)}")
        return data


# ---------------------------------------------------------------------------
# Analysis result types (P3+ analysis layer)
# ---------------------------------------------------------------------------


class GroupStats(BaseModel):
    """Aggregate statistics for one group in a comparison."""
    group_label: str
    group_values: list[Any] = Field(default_factory=list)  # raw column values for this group
    n: int
    mean: float
    sd: float

    model_config = {"frozen": True}


class GroupComparisonResult(BaseModel):
    """Result of a two-group statistical comparison analysis."""
    analysis_type: str = "group_comparison"
    outcome_variable_id: str   # catalog variable_id of the outcome
    outcome_label: str         # human-readable outcome name
    group_variable_id: str     # catalog variable_id of the grouping variable
    group_label: str           # human-readable grouping variable name
    groups: list[GroupStats]
    p_value: float | None = None   # None when suppressed or test not run
    test_name: str = "welch_t_test"
    cohort_def: str
    caveats: list[str] = Field(default_factory=list)
    n_total: int

    model_config = {"frozen": True}


class GeneratedAnalysisResult(BaseModel):
    """Result from LLM-generated analysis code executed in the local sandbox."""
    analysis_description: str
    results: dict[str, Any]  # only scalar values (validated before construction)
    n_analyzed: int
    cohort_def: str
    code_used: str  # the generated code, for auditability

    model_config = {"frozen": True}


class DescriptiveStatResult(BaseModel):
    """Full descriptive statistics for one variable over a filtered cohort."""
    variable_id: str
    n_total: int           # total eligible participants
    n_missing: int         # missing / null values in this variable
    count: int             # n_total - n_missing (non-null observations)
    mean: float | None     # None for non-numeric variables
    median: float | None
    sd: float | None
    q1: float | None       # 25th percentile
    q3: float | None       # 75th percentile
    iqr: float | None      # q3 - q1
    min_val: float | None
    max_val: float | None
    proportion: float | None  # fraction with value > 0 (binary variable summary)
    event_anchor: str | None = None

    model_config = {"frozen": True}


class RegressionCoef(BaseModel):
    """One coefficient row in a regression result."""
    variable: str
    coefficient: float
    std_error: float | None = None
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None

    model_config = {"frozen": True}


class RegressionResult(BaseModel):
    """Aggregate-only regression output (coefficients, fit stats — no subject rows)."""
    regression_type: str    # "linear" or "logistic"
    outcome_variable_id: str
    outcome_label: str
    n: int
    r_squared: float | None = None     # for linear; None for logistic
    pseudo_r2: float | None = None     # McFadden for logistic
    coefficients: list[RegressionCoef] = []
    cohort_def: str
    caveats: list[str] = []

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# LeakageFinding  (intent-layer leakage assessment per resolved variable)
# ---------------------------------------------------------------------------


class LeakageFinding(BaseModel):
    """Role-specific leakage assessment for one resolved catalog variable.

    Produced deterministically by the intent-validation layer after the LLM
    resolves NL concepts to raw catalog columns.  Never generated by the LLM
    itself — always computed from catalog leakage_tags + role context.
    """

    raw_code: str
    concept: str
    role: str
    # role values: cohort_filter | grouping_variable | outcome | predictor |
    #              covariate | label | event_definition
    leakage_tag: str
    # leakage_tag values: NONE | DIAGNOSIS_DEFINING | CIRCULAR |
    #                     HARMONIZATION_FIT_SCOPE | ROW_IDENTIFIER
    severity: str
    # severity values: allowed | allow_with_caveat | warn | block
    reason: str
    action_taken: str
    # action_taken values: allowed_for_cohort_definition |
    #   allowed_for_descriptive_analysis | allowed_as_outcome |
    #   blocked_as_predictor | excluded | caused_abstain

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# AnalysisTrace  (P5 auditable decision log)
# ---------------------------------------------------------------------------


class VariableResolutionTrace(BaseModel):
    """Role-variable resolution trace from the analysis plan.

    Populated deterministically from AnalysisRole fields after LLM planning
    and post-validation.  No raw chain-of-thought appears here.
    """

    role: str
    researcher_label: str
    selected_raw_code: str | None = None
    semantic_definition: str | None = None
    filter_expression: dict[str, Any] | None = None
    resolution_status: str = "unresolved"
    validation_status: str | None = None
    evidence: str | None = None  # leakage reason or catalog entry description

    model_config = {"frozen": True}


class ToolCallTrace(BaseModel):
    """Trace of one deterministic tool call in the pipeline.

    input_summary and output_summary are derived from the ToolResult value;
    no LLM generation is involved.  Numeric values in output_summary must
    trace back to the corresponding ToolResult.value.
    """

    tool_name: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any] | None = None
    grounded: bool
    abstain_reason: str | None = None

    model_config = {"frozen": True}


class AnalysisTrace(BaseModel):
    """Structured audit trail for one agent pipeline run.

    Populated entirely from deterministic pipeline outputs: StructuredIntent,
    AnalysisPlan, FilterSpec, ToolResult list.  No field here may contain
    raw LLM chain-of-thought or model-generated numbers.

    Numeric values in tool_calls[*].output_summary must trace back to the
    corresponding ToolResult.value — the hallucination guard enforces this
    for the narrated answer; the trace makes it inspectable.
    """

    original_question: str
    parsing_mode: str                        # "heuristic" or "llm"
    analysis_type: str | None = None
    question_type: str
    research_intent_summary: str | None = None
    unit_of_analysis: str | None = None
    statistic: str | None = None
    visit_anchor: str | None = None

    plan_steps: list[str] = Field(default_factory=list)
    variable_resolutions: list[VariableResolutionTrace] = Field(default_factory=list)
    filter_trace: dict[str, Any] = Field(default_factory=dict)
    leakage_findings: list[LeakageFinding] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    failure_mode: str | None = None
    modeling_design_plan: ModelingDesignPlan | None = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# AnalysisRole / AnalysisPlan  (P5 planner output)
# ---------------------------------------------------------------------------


class AnalysisRole(BaseModel):
    """One logical role in an AnalysisPlan with its resolved catalog variable.

    Produced by the LLM planner; raw_codes are validated deterministically
    after the LLM call.  Fields set to None when validation fails.
    """

    role: str
    # role values: cohort | event | value_variable | group_variable |
    #              outcome_variable | covariate
    researcher_label: str
    semantic_definition: str
    selected_raw_code: str | None = None
    filter_expression: dict[str, Any] | None = None
    required: bool = True
    resolution_status: str = "unresolved"
    # resolution_status values: resolved | unresolved | not_applicable
    validation_status: str | None = None
    # validation_status values: valid | invalid_not_in_catalog | invalid_blocked | None

    model_config = {"frozen": True}


class AnalysisPlan(BaseModel):
    """LLM-produced analysis plan: decomposed roles + resolved catalog columns.

    The planner decomposes compound phrases (e.g. 'age at first AD diagnosis')
    into distinct logical roles before any catalog lookup.  Deterministic
    _validate_analysis_plan then checks each selected_raw_code and populates
    leakage_assessment and unresolved_items.

    Key design: never treat a compound analysis phrase as a variable name.
    The value_variable, event, cohort, and visit_anchor are always separate roles.
    """

    original_question: str
    analysis_type: str
    # analysis_type values: group_comparison | event_anchored_stat | aggregate_stat |
    #   descriptive_stat | regression | cohort_size | lookup
    research_intent_summary: str
    candidate_label: str = ""
    horizon_years: int | None = None
    id_space: str = "A"
    unit_of_analysis: str | None = None
    requires_longitudinal_data: bool = False
    execution_steps: list[str] = Field(default_factory=list)
    statistical_method_hint: str | None = None

    cohort: AnalysisRole | None = None
    event: AnalysisRole | None = None
    value_variable: AnalysisRole | None = None
    group_variable: AnalysisRole | None = None
    outcome_variable: AnalysisRole | None = None
    covariates: list[AnalysisRole] = Field(default_factory=list)

    statistic: str | None = None
    visit_anchor: str | None = None
    # visit_anchor values: baseline | first_matching_event_visit |
    #   last_visit | first_conversion_visit | None
    unresolved_items: list[str] = Field(default_factory=list)
    leakage_assessment: list[LeakageFinding] = Field(default_factory=list)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# EvidenceNeedItem / ModelingDesignPlan  (P5 evidence planning for modeling_design)
# ---------------------------------------------------------------------------


class EvidenceNeedItem(BaseModel):
    """One evidence item requested by the LLM evidence planner.

    The LLM specifies which deterministic tool to call and the cohort/modality
    parameters needed.  The orchestrator executes the tool and records the result.
    No numbers may be generated by the LLM — they must come from tool output.
    """

    item_id: str
    description: str
    tool_name: str
    # Supported values: "compute_cohort_size" | "assess_longitudinal_feasibility" |
    #                   "suggest_modeling_design" | "assess_modality_overlap"
    baseline_diagnosis: list[str] | None = None
    baseline_etiology: list[str] | None = None
    modalities: list[str] | None = None   # for assess_modality_overlap
    horizon_years: int | None = None      # for assess_longitudinal_feasibility
    priority: str = "required"            # "required" | "optional"

    model_config = {"frozen": True}


class ModelingDesignPlan(BaseModel):
    """LLM-produced evidence plan for a modeling_design question.

    The LLM decides WHAT to compute (evidence_items); the orchestrator calls the
    deterministic tools and returns numbers.  The LLM then synthesises a
    recommendation from those grounded values.

    Invariant: no numeric fact in this object is LLM-generated.  Only strings
    describing what is needed (modeling_goal, label options, etc.) come from the LLM.
    """

    original_question: str
    modeling_goal: str
    target_population: str
    target_population_filter: dict[str, Any] = Field(default_factory=dict)
    target_etiology: str | None = None
    label_definition_options: list[str] = Field(default_factory=list)
    prediction_horizon_years: int = 3
    candidate_modalities: list[str] = Field(default_factory=list)
    candidate_feature_groups: list[str] = Field(default_factory=list)
    evidence_items: list[EvidenceNeedItem] = Field(default_factory=list)
    leakage_checks: list[str] = Field(default_factory=list)
    documentation_queries: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# ExplanationContext types  (P5 structured-explanation layer)
# ---------------------------------------------------------------------------


class LeakageRecord(BaseModel):
    """Full leakage detail for one variable in the modeling design blacklist.

    Constructed deterministically from CatalogEntry + LeakageFlag; never generated
    by the LLM.
    """

    variable_id: str
    raw_code: str
    display_name: str
    description: str = ""
    tag: str               # primary / joined leakage tags, e.g. "DIAGNOSIS_DEFINING"
    tags: list[str] = Field(default_factory=list)
    reason: str
    source_dataset: str = ""
    excluded_from: Literal["features", "labels", "both"] = "features"


class StructuredCaveat(BaseModel):
    """Structured caveat passed to the LLM narration layer.

    All fields are deterministically derived from intent/catalog metadata; the LLM
    writes a contextualised sentence from them but cannot invent new facts.
    """

    parsed_target_etiology: str | None = None
    actual_endpoint: str | None = None
    missing_concept: str | None = None
    implication: str | None = None
    required_next_step: str | None = None


class ExplainableItem(BaseModel):
    """One semantically meaningful unit extracted from a tool result.

    Constructed deterministically; no LLM involvement.  The LLM receives these
    items as structured context and writes explanatory sentences from them.
    """

    item_id: str
    display_name: str
    raw_key: str
    value: Union[int, float, str, None] = None
    unit: str | None = None                    # "participants", "variables", "years"
    source_tool: str
    dataset: str | None = None
    id_space: str | None = None
    filter_scope: str | None = None
    denominator: int | None = None
    definition: str = ""
    computation_note: str | None = None
    caveats: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    importance: Literal["high", "medium", "low"] = "medium"
    explanation_required: bool = False
    section_id: str = "metadata"               # which output section this item belongs to
    is_not_answerable: bool = False


class ExplanationSection(BaseModel):
    """One semantic section in the structured explanation output."""

    section_id: str
    title: str
    item_ids: list[str] = Field(default_factory=list)


class ExplanationMetadata(BaseModel):
    """Static metadata about the agent run — surfaced in the audit section."""

    horizon_years: int
    horizon_source: Literal["user", "default"]
    horizon_label: str          # e.g. "3 years (example default)"
    backend: str | None = None
    catalog_release: str | None = None
    tools_invoked: list[str] = Field(default_factory=list)


class ExplanationContext(BaseModel):
    """Container for all structured explanation material for one agent run.

    Built deterministically from tool results; consumed by LLM narration and
    the template fallback.  Never contains LLM-generated numbers.
    """

    items: list[ExplainableItem] = Field(default_factory=list)
    sections: list[ExplanationSection] = Field(default_factory=list)
    caveats: list[StructuredCaveat] = Field(default_factory=list)
    metadata: ExplanationMetadata
    not_answerable: list[ExplainableItem] = Field(default_factory=list)

    def required_items(self) -> list[ExplainableItem]:
        return [i for i in self.items if i.explanation_required]


# ---------------------------------------------------------------------------
# Narration layer types  (P5 LLM narration layer)
# ---------------------------------------------------------------------------


class NarrationFact(BaseModel):
    fact_id: str
    fact_type: Literal[
        "count", "percentage", "cohort_definition", "outcome_definition",
        "modality_availability", "variable_policy", "leakage_warning",
        "not_answerable", "caveat", "recommendation_seed", "metadata"
    ]
    label: str
    value: Any = None
    unit: str | None = None
    scope_label: str | None = None
    scope_filter: dict[str, Any] | None = None
    denominator_value: int | None = None
    denominator_label: str | None = None
    denominator_source: str | None = None  # e.g. "derived_from_followup_bucket_sum"
    source_tool: str
    evidence: list[str] = Field(default_factory=list)
    deterministic_definition: str | None = None
    caveats: list[str] = Field(default_factory=list)
    importance: Literal["high", "medium", "low"] = "medium"
    requires_explanation: bool = False


class NarrationValidationResult(BaseModel):
    passed: bool
    explanation_coverage: float
    analysis_section_present: bool
    scope_validation_passed: bool
    high_importance_coverage: float
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Longitudinal feasibility metadata (populated for feasibility / modeling_design types)
    status_definitions_present: bool = False
    insights_section_present: bool = False
    percentages_present: bool = False
    censoring_definition_present: bool = False
    censoring_horizon_dependent: bool | None = None  # None = not applicable
    # External literature claim guard (check #6)
    external_claim_violation: bool = False
    external_claim_phrases: list[str] = Field(default_factory=list)


class NarrationResult(BaseModel):
    answer: str
    mode: Literal["llm", "deterministic_fallback"]
    model: str | None = None
    error: str | None = None
    validation: NarrationValidationResult | None = None
    retry_attempted: bool = False
    facts_used: int = 0


# ---------------------------------------------------------------------------
# CohortMap
# ---------------------------------------------------------------------------


class CohortMap(BaseModel):
    id_space: IdSpace
    tier: Tier
    cohort_role: CohortRole
    eligible_n: int | None = None  # None when suppressed or not yet computed
    index_visit_def: str
    prediction_horizon: int  # years
    outcome_label_def: str
    follow_up_availability: dict[str, Any] = Field(default_factory=dict)
    modality_coverage: dict[str, int] = Field(default_factory=dict)
    feature_whitelist: list[str] = Field(default_factory=list)
    leakage_blacklist: list[str] = Field(default_factory=list)
    # key is sorted comma-joined modality names; value is intersection count
    sample_size_by_modality_combo: dict[str, int] = Field(default_factory=dict)
    release_versions: dict[str, str] = Field(default_factory=dict)
    suppression_threshold: int
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# 4-Layer Enriched Planner types  (P5+ architectural refactor)
# ---------------------------------------------------------------------------


class PlanOperation(BaseModel):
    """One composable operation in an EnrichedAnalysisPlan.

    op_type values are v0 adapter families over the existing deterministic tools,
    not a final atomic operation graph.  Future v1 work should decompose these into
    lower-level primitives (filter_cohort, derive_event_status, aggregate_count, etc.).

    Supported op_type values:
      count_eligible | longitudinal_feasibility | modality_overlap |
      variable_stat  | descriptive_stat          | group_comparison |
      lookup_variable | leakage_check            | modeling_design
    """

    op_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    output_name: str   # unique key in ExecutionContext.results
    description: str = ""

    model_config = {"frozen": True}


class GroupDefinition(BaseModel):
    """One named cohort in a multi-group comparison (e.g. CN vs. MCI vs. AD).

    Generalizes the single-cohort baseline_filters/event_filters/etiology_definition
    slots on SemanticAnalysisPlan into a list, for questions that compare several
    named groups side by side. Each group is independently subject to the same C1/C3/C5
    coherence rules as the top-level plan (see PlanValidator._check_group_coherence).

    When a group label is ambiguous between two distinct diagnostic bases (e.g. "AD" can
    mean syndromic all-cause dementia OR etiologic AD attribution), the LLM should emit
    TWO GroupDefinitions with distinct labels (e.g. "AD (syndromic)", "AD (etiologic)")
    rather than silently picking one — this keeps the disambiguation decision in the LLM's
    semantic territory instead of a hardcoded executor branch.

    diagnosis_basis / clinical_status (canonical contract, added after a live DeepSeek
    failure where a model expressed a clinical-status filter as raw ADNI column names
    with operator objects, e.g. {"DXMCI": 1, "DXAD": {"neq": 1}}, inside baseline_filters
    — a shape the compiler had no way to recognize, so it silently compiled to an empty
    filter for every clinical-status group while leaving their raw (but unrecognized)
    baseline_filters dicts non-empty and mutually distinct, defeating the duplicate/empty
    detection that compared those raw dicts). These two fields are the ONE constrained,
    Pydantic-validated channel for diagnosis-defining intent — an enum the LLM cannot
    misname the way it can invent an arbitrary dict key. When diagnosis_basis is set, it
    takes priority over baseline_filters for compiling the executable filter (see
    semantic_planner._filter_dict_for_scope, the single canonical normalization step).
    baseline_filters/event_filters remain available for non-diagnosis criteria and for
    backward compatibility with plans that predate this field (diagnosis_basis=None).
    """

    label: str
    cohort_scope: CohortScope = CohortScope.BASELINE
    diagnosis_basis: Literal["clinical_status", "etiologic_ad"] | None = None
    clinical_status: Literal["normal", "mci", "dementia", "impaired-not-mci"] | None = None
    baseline_filters: dict[str, Any] = Field(default_factory=dict)
    event_filters: dict[str, Any] = Field(default_factory=dict)
    event_anchor: str | None = None
    etiology_definition: EtiologyDefinition = EtiologyDefinition.UNSPECIFIED
    diagnosis_variable: str = ""
    required_modalities: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


class VariableSelection(BaseModel):
    """One exact catalog variable chosen by the LLM to satisfy a requested measure.

    variable_id MUST be copied verbatim from the catalog context (never constructed or
    shortened by the LLM) — PlanValidator enforces exact catalog membership. specimen and
    analyte are free-text semantic tags the LLM believes the variable satisfies (e.g.
    specimen="csf", analyte="ptau181"); PlanValidator cross-checks them against the
    resolved catalog entry's family/variant/concept metadata rather than trusting the LLM
    claim blindly. variant_label is a researcher-facing name for this specific selection
    (e.g. "csf_raw", "csf_harmonized") used to keep multiple assay variants of the same
    analyte separate in output_name / reporting — they are never pooled by default.
    """

    variable_id: str
    specimen: str | None = None
    analyte: str | None = None
    variant_label: str = ""

    model_config = {"frozen": True}


class SemanticAnalysisPlan(BaseModel):
    """Upstream semantic gate — produced by the LLM before any EnrichedAnalysisPlan is built.

    Captures every semantic commitment the LLM must resolve explicitly to prevent silent
    mismatches (baseline vs. event-level, row vs. participant, all-cause vs. etiologic AD,
    first-diagnosis vs. first-visit, missing denominator).

    Validated by PlanValidator.validate_semantic() before compile_to_enriched() runs.

    groups and variable_selections are OPTIONAL generalizations for multi-group /
    multi-variant questions (e.g. "CSF p-tau181 for CN/MCI/AD"):
      - groups: when non-empty, compile_to_enriched() fans out over each GroupDefinition
        instead of the single top-level baseline_filters/event_filters/etiology_definition.
        Empty (default) preserves the original single-cohort behavior exactly.
      - variable_selections: when non-empty, compile_to_enriched() fans out over each
        VariableSelection instead of the single statistic_variable. Empty (default)
        preserves the original single-variable behavior exactly.
    Both lists default empty so every existing single-cohort/single-variable plan is
    unaffected — these fields are pure additive generalization.
    """

    original_question: str

    population_unit: PopulationUnit
    requires_unique_participants: bool

    cohort_scope: CohortScope
    baseline_filters: dict[str, Any] = Field(default_factory=dict)
    event_filters: dict[str, Any] = Field(default_factory=dict)

    event_anchor: str | None = None
    time_ordering_variable: str = "NACCVNUM"

    outcome_definition: str = ""
    etiology_definition: EtiologyDefinition = EtiologyDefinition.UNSPECIFIED
    diagnosis_variable: str = ""

    statistic_variable: str = ""
    statistic_type: str | None = None
    denominator_definition: str = ""

    # Multi-group / multi-variant generalization (both default empty — see docstring).
    groups: list[GroupDefinition] = Field(default_factory=list)
    specimen: str | None = None
    analyte: str | None = None
    variable_selections: list[VariableSelection] = Field(default_factory=list)
    required_modalities: list[str] = Field(default_factory=list)

    allowed_backend_tools: list[str] = Field(default_factory=list)
    known_invalid_interpretations: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    abstain_if_missing: list[str] = Field(default_factory=list)

    id_space: str = "A"

    model_config = {"frozen": True}


class EnrichedAnalysisPlan(BaseModel):
    """Extended analysis plan with explicit composable operations and multi-horizon support.

    Layer 1 planner output.  The LLM produces this directly from the raw user question
    (bypassing StructuredIntent as the primary semantic gate).

    researcher_audit_trace is a concise, researcher-visible audit of key interpretation
    decisions — NOT chain-of-thought.  Example entries:
      "Interpreted 'baseline MCI' as baseline_diagnosis=mci (NACCUDSD=3)."
      "Extracted horizons [2, 3, 5] from '2, 3, or 5 years'."
      "This question counts participants, not visits."
      "Requires real longitudinal visit data, not catalog-only lookup."

    original_plan carries the legacy AnalysisPlan for code paths that still dispatch
    on analysis_type.  None when the enriched planner runs in standalone mode.
    """

    original_question: str
    research_intent_summary: str
    researcher_audit_trace: list[str] = Field(default_factory=list)
    operations: list[PlanOperation] = Field(default_factory=list)
    requested_horizons: list[int] = Field(default_factory=list)  # e.g. [2, 3, 5]
    requires_backend_data: bool = True
    baseline_definition: dict[str, Any] = Field(default_factory=dict)  # {"diagnosis": "mci"}
    outcome_definition: str = ""
    id_space: str = "A"
    original_plan: AnalysisPlan | None = None  # backward-compat carrier
    semantic_plan: SemanticAnalysisPlan | None = None  # upstream semantic gate carrier

    model_config = {"frozen": True}


class ValidationResult(BaseModel):
    """Output of PlanValidator — go/no-go plus structured warnings and assumptions.

    is_executable=False blocks the DeterministicExecutor from running.
    hard_errors are shown to the researcher.
    soft_warnings appear in the caveats section of the SynthesisReport.
    blocked_operations lists output_names of individual ops that were blocked;
    the executor skips them and runs the remaining ops.
    """

    hard_errors: list[str] = Field(default_factory=list)
    soft_warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    is_executable: bool = True
    blocked_operations: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


class ExecutionContext(BaseModel):
    """Outputs of DeterministicExecutor.execute() — one ToolResult per output_name.

    Not frozen: the executor builds this incrementally during the operation loop.

    results maps output_name → ToolResult (typed as Any to avoid Generic[T] issues).
    errors are non-blocking execution problems.
    operations_run records output_names of ops that were actually executed.

    Invariant: no value in results may be LLM-generated — all must trace to
    deterministic tool functions.
    """

    results: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    operations_run: list[str] = Field(default_factory=list)
    # NOT frozen — executor mutates this during the operation loop


class SynthesisSection(BaseModel):
    """One structured section of a SynthesisReport.

    Standard section_id values:
      direct_answer       — concise answer to the literal question
      analysis_plan       — researcher_audit_trace + operation descriptions
      evidence_provenance — per-operation tool results (integers checked by guard)
      caveats             — validator warnings + data caveats
      insights            — interpretation of results for modeling feasibility
      next_steps          — ≤3 recommended follow-on questions
    """

    section_id: str
    title: str
    content: str

    model_config = {"frozen": True}


class SynthesisReport(BaseModel):
    """Structured research report from the LLM Synthesizer (Layer 4).

    All quantitative claims in sections must trace back to tool results in
    execution_context (enforced by _check_synthesis_hallucination).

    narration_result carries the legacy NarrationResult for callers that use
    the old narrate_with_llm path.  None in the enriched path.

    synthesis_external_claim_violation: True when the LLM synthesis contained
    external literature comparison claims (aligns with literature, consistent with
    prior studies, etc.) without any doc_evidence supplied.  When True,
    synthesis_external_claim_retried indicates whether a correction prompt was
    attempted before returning this report.
    """

    question: str
    sections: list[SynthesisSection] = Field(default_factory=list)
    validation_result: ValidationResult | None = None
    execution_context: ExecutionContext | None = None
    grounded: bool = True
    abstain_reason: str | None = None
    narration_result: NarrationResult | None = None  # backward-compat
    # External literature claim guard (synthesis path)
    synthesis_external_claim_violation: bool = False
    synthesis_external_claim_retried: bool = False

    model_config = {"frozen": True}
