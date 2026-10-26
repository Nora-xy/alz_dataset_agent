"""Pydantic schemas for parsed research-question intent.

Public types:
  QuestionType      — enum of supported question categories
  ParseDebugInfo    — diagnostic payload from the LLM planner
  StructuredIntent  — fully parsed research question (LLM or heuristic)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from alz_agent.agent.event_decomposition import _is_compound_event_phrase
from alz_agent.schemas import AnalysisPlan, IdSpace, LeakageFinding


class QuestionType(str, Enum):
    FEASIBILITY = "feasibility"
    COHORT_SIZE = "cohort_size"
    LOOKUP = "lookup"
    MODELING_DESIGN = "modeling_design"
    AGGREGATE_STAT = "aggregate_stat"
    STATISTICAL_ANALYSIS = "statistical_analysis"


class ParseDebugInfo(BaseModel):
    """Diagnostic output from the LLM planner — for tracing decomposition failures.

    Failure modes:
      parsing            — LLM didn't invoke the tool or returned bad JSON
      variable_resolution — compound phrase or unknown variable in value_variable
      unsupported_executor — analysis_type not yet implemented
      missing_data       — required modality/variable absent from catalog
    """

    raw_llm_plan_json: dict[str, Any] = Field(default_factory=dict)
    normalized_plan: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    failure_mode: str | None = None

    model_config = {"frozen": True}


class StructuredIntent(BaseModel):
    """Parsed research question — NL labels + validated raw columns + leakage.

    Filled by the LLM (catalog-grounded tool use) or heuristic regex fallback.
    After LLM parsing, deterministic validation checks every resolved raw_code
    against the active catalog and computes leakage_assessment.

    Legacy NL fields (group_variable, outcome_variable, stat_variable) are
    kept for the heuristic path and as human-readable labels in the LLM path.
    Resolved raw columns (group_column, outcome_column, …) are set only in
    LLM mode and take precedence in the orchestrator.
    """

    # ---- Core classification fields (both paths) -------------------------
    target_population: list[str] = Field(
        description=(
            "Baseline cognitive-status labels. "
            "Valid: 'mci', 'normal', 'dementia', 'impaired-not-mci'. "
            "Do NOT put 'ad' here — use target_etiology for AD attribution."
        )
    )
    target_etiology: list[str] | None = Field(
        default=None,
        description=(
            "Etiologic attribution filter. ['ad'] when question refers to "
            "AD-diagnosed patients. null otherwise."
        ),
    )
    candidate_label: str = Field(
        description="Outcome / label of interest (e.g. 'conversion to AD')."
    )
    horizon_years: int | None = Field(
        default=None,
        description=(
            "Prediction horizon in integer years (>= 1). null for non-longitudinal."
        ),
    )
    modalities: list[str] = Field(
        default_factory=list,
        description=(
            "Modalities of interest. Valid: 'cognition','mri','pet',"
            "'fluid_biomarker','vascular','neuropath','demographic'."
        ),
    )
    any_of_modalities: list[str] | None = Field(
        default=None,
        description=(
            "Subset of `modalities` that should be OR-combined rather than "
            "AND-intersected (e.g. ['mri','pet'] for a question about 'any "
            "imaging'). The remaining entries in `modalities` (modalities "
            "minus this list) are still AND-required. None (the default) "
            "means every entry in `modalities` is a plain AND/intersection — "
            "unchanged behavior for every existing parsing path. Additive "
            "field; only ever populated by agent/hf_local_intent.py today, "
            "to preserve 'any imaging' OR semantics through the adapter "
            "without collapsing it into a plain intersection list."
        ),
    )
    question_type: QuestionType = Field(
        description="What the user is actually asking."
    )
    pre_index_only: bool = Field(default=True)
    id_space: IdSpace = Field(default=IdSpace.A)

    # ---- Aggregate stat fields -------------------------------------------
    stat_variable: str | None = Field(
        default=None,
        description=(
            "NL variable/concept for AGGREGATE_STAT — heuristic path only. "
            "In plan path use value_variable_label + value_column instead. "
            "Never populated with compound phrases like 'age at first AD diagnosis'."
        ),
    )
    stat_type: str | None = Field(
        default="mean",
        description=(
            "Stat to compute: 'mean', 'median', 'std', 'min', 'max'. "
            "Default remains 'mean' when unset — widened to allow explicit "
            "None so a non-AGGREGATE_STAT question_type (e.g. COHORT_SIZE) "
            "can signal 'not applicable' instead of carrying a meaningless "
            "leftover 'mean'. Every existing caller that doesn't pass this "
            "field explicitly is unaffected."
        ),
    )
    event_anchor: str | None = Field(
        default=None,
        description=(
            "'first_diagnosis' / 'last_visit' / 'first_conversion_visit' / "
            "'baseline' / null."
        ),
    )

    # ---- Structured value-variable fields (plan path only) ---------------
    value_variable_label: str | None = Field(
        default=None,
        description=(
            "NL label for the quantity being measured in aggregate/event-anchored "
            "stats (e.g. 'age', 'MMSE score'). Must be a simple atomic attribute — "
            "never a compound phrase like 'age at first AD diagnosis'. "
            "When the event is implicit in the phrase, decompose: place the attribute "
            "here and the event condition in event_filter + event_column."
        ),
    )
    value_column: str | None = Field(
        default=None,
        description=(
            "Validated raw catalog column for value_variable (e.g. 'NACCAGE'). "
            "Populated from AnalysisPlan.value_variable.selected_raw_code."
        ),
    )
    event_filter: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Column-value filter dict for the event condition "
            "(e.g. {'NACCALZD': 1}). Companion to event_column. "
            "Populated from AnalysisPlan.event.filter_expression."
        ),
    )
    unit_of_analysis: str | None = Field(
        default=None,
        description="Unit of analysis (e.g. 'participant', 'visit').",
    )
    parse_debug: ParseDebugInfo | None = Field(
        default=None,
        description=(
            "Diagnostic output from the LLM planner. Populated only in LLM mode. "
            "Contains raw plan JSON, normalized plan, validation errors, and failure mode."
        ),
    )

    # ---- Statistical analysis fields (NL labels — both paths) -----------
    analysis_type: str | None = Field(
        default=None,
        description=(
            "For STATISTICAL_ANALYSIS: 'group_comparison', 'correlation', "
            "'regression', 'descriptive_stat', 'event_anchored_stat'. null otherwise."
        ),
    )
    group_variable: str | None = Field(
        default=None,
        description=(
            "NL label for grouping variable (e.g. 'heart attack history'). "
            "Used as lookup query in heuristic mode; overridden by group_column in LLM mode."
        ),
    )
    outcome_variable: str | None = Field(
        default=None,
        description=(
            "NL label for outcome variable (e.g. 'MMSE', 'age'). "
            "Used as lookup query in heuristic mode; overridden by outcome_column in LLM mode."
        ),
    )

    # ---- Resolved raw columns (LLM mode only — validated against catalog) -
    cohort_label: str | None = Field(
        default=None,
        description="Human-readable cohort definition (e.g. 'baseline MCI').",
    )
    cohort_column: str | None = Field(
        default=None,
        description=(
            "Validated raw catalog column that defines the cohort "
            "(e.g. 'NACCUDSD'). null when not resolved."
        ),
    )
    cohort_filter: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Column-value filter dict (e.g. {'NACCUDSD': 3}). "
            "Values are numeric catalog codes."
        ),
    )
    group_variable_label: str | None = Field(
        default=None,
        description="Human-readable label for grouping variable (LLM mode).",
    )
    group_column: str | None = Field(
        default=None,
        description=(
            "Validated raw column for binary grouping (e.g. 'CVHATT'). "
            "Takes precedence over group_variable in orchestrator."
        ),
    )
    outcome_variable_label: str | None = Field(
        default=None,
        description="Human-readable label for outcome variable (LLM mode).",
    )
    outcome_column: str | None = Field(
        default=None,
        description=(
            "Validated raw column for outcome (e.g. 'NACCMMSE'). "
            "Takes precedence over outcome_variable in orchestrator."
        ),
    )
    covariate_labels: list[str] = Field(
        default_factory=list,
        description="NL labels for covariate variables.",
    )
    covariate_columns: list[str] = Field(
        default_factory=list,
        description="Validated raw columns for covariates.",
    )
    label_variable_label: str | None = Field(
        default=None,
        description="NL label for prediction label variable.",
    )
    label_column: str | None = Field(
        default=None,
        description="Validated raw column for label (e.g. 'NACCUDSD' for conversion).",
    )
    event_label: str | None = Field(
        default=None,
        description="NL label for the event variable (e.g. 'first AD diagnosis').",
    )
    event_column: str | None = Field(
        default=None,
        description="Validated raw column for event (e.g. 'NACCALZD').",
    )
    statistical_method_hint: str | None = Field(
        default=None,
        description="Suggested statistical method (e.g. 'Welch two-sample t-test').",
    )
    unresolved_fields: list[str] = Field(
        default_factory=list,
        description="Fields where catalog variable resolution failed.",
    )
    leakage_assessment: list[LeakageFinding] = Field(
        default_factory=list,
        description=(
            "Deterministic role-specific leakage findings for all resolved variables. "
            "Computed after LLM parsing, not by the LLM."
        ),
    )
    parsing_mode: str = Field(
        default="heuristic",
        description="'heuristic' (regex fallback) or 'llm' (catalog-grounded).",
    )
    analysis_plan: AnalysisPlan | None = Field(
        default=None,
        description=(
            "Validated AnalysisPlan from the LLM planner (LLM mode only). "
            "When set, the orchestrator dispatches via plan roles rather than "
            "NL-string slot filling."
        ),
    )

    @field_validator("horizon_years")
    @classmethod
    def _horizon_must_be_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("horizon_years must be >= 1 when provided")
        return v

    @field_validator("value_variable_label")
    @classmethod
    def _reject_compound_value_label(cls, v: str | None) -> str | None:
        if v is not None and _is_compound_event_phrase(v):
            raise ValueError(
                f"value_variable_label {v!r} contains an event-anchor phrase. "
                "Decompose: place the attribute in value_variable_label "
                "and the event condition in event_filter + event_column + visit_anchor."
            )
        return v

    model_config = {"frozen": True}
