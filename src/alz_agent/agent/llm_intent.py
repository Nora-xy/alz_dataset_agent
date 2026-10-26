"""LLM-based intent parsing — catalog-grounded tool-use and JSON fallback paths.

Two parsing paths:
  Primary:  plan_analysis tool → AnalysisPlan → derived StructuredIntent.
            The planner decomposes compound phrases before catalog lookup.
  Fallback: JSON-prompt fill_intent for models that don't support tool_choice.

Public functions:
  plan_analysis(nl, client, model, catalog) → AnalysisPlan
  _parse_with_llm(...)  → StructuredIntent   (primary entry for parse_intent)
  _plan_analysis_internal(...)  → (AnalysisPlan, ParseDebugInfo)
  _parse_with_json_prompt(...)  → StructuredIntent
"""

from __future__ import annotations

import json
import re
from typing import Any

from alz_agent.agent.catalog_context import _build_catalog_context
from alz_agent.agent.event_decomposition import _is_compound_event_phrase
from alz_agent.agent.intent_schema import (
    ParseDebugInfo,
    QuestionType,
    StructuredIntent,
)
from alz_agent.agent.leakage_validation import (
    _validate_analysis_plan,
    _validate_resolved_intent,
)
from alz_agent.agent.providers import PlannerClient
from alz_agent.schemas import AnalysisPlan, CatalogEntry, IdSpace, LeakageFinding

# ---------------------------------------------------------------------------
# LLM output normalisation
# ---------------------------------------------------------------------------

_EXPLICIT_ZERO_HORIZON_RE = re.compile(
    r"\b0\s*[-–]?\s*year|\bzero\s+year", re.IGNORECASE
)

_LLM_MODALITY_ALIASES: dict[str, list[str]] = {
    "imaging":     ["mri", "pet"],
    "any_imaging": ["mri", "pet"],
    "any imaging": ["mri", "pet"],
}


def _normalize_llm_output(data: dict[str, Any], nl_question: str) -> dict[str, Any]:
    """Normalize known LLM placeholder patterns before Pydantic validation."""
    data = dict(data)

    horizon = data.get("horizon_years")
    if isinstance(horizon, (int, float)) and int(horizon) <= 0:
        if not _EXPLICIT_ZERO_HORIZON_RE.search(nl_question):
            data["horizon_years"] = None

    if isinstance(data.get("modalities"), list):
        expanded: list[str] = []
        for mod in data["modalities"]:
            if mod in _LLM_MODALITY_ALIASES:
                for canonical in _LLM_MODALITY_ALIASES[mod]:
                    if canonical not in expanded:
                        expanded.append(canonical)
            elif mod not in expanded:
                expanded.append(mod)
        data["modalities"] = expanded

    # Ensure list fields are lists (LLM sometimes returns null for empty lists)
    for list_field in ("covariate_labels", "covariate_columns", "unresolved_fields", "modalities"):
        if data.get(list_field) is None:
            data[list_field] = []

    # Coerce required string fields that small models sometimes return as null.
    # candidate_label: no default in StructuredIntent — derive from question as fallback.
    if data.get("candidate_label") is None:
        data["candidate_label"] = nl_question[:80]
    # stat_type: declared default is "mean"; null bypasses the default because Pydantic
    # validates the explicit null before applying the field default.
    if data.get("stat_type") is None:
        data["stat_type"] = "mean"
    # question_type: no default — fall back to cohort_size (safe for count-style questions).
    if data.get("question_type") is None:
        data["question_type"] = "cohort_size"

    # parsing_mode: always llm when coming from this path
    data["parsing_mode"] = "llm"

    return data


# ---------------------------------------------------------------------------
# LLM planner — catalog-grounded AnalysisPlan tool-use path
# ---------------------------------------------------------------------------

# _ANALYSIS_PLAN_TOOL: the primary LLM tool for plan_analysis().
# Inline role sub-schemas (no $ref) for broadest model compatibility.
_ROLE_SUBSCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {
            "type": "string",
            "description": "Plan field name: cohort|event|value_variable|group_variable|outcome_variable|covariate",
        },
        "researcher_label": {
            "type": "string",
            "description": "Researcher's phrase (e.g. 'heart attack history', 'age', 'first AD diagnosis')",
        },
        "semantic_definition": {
            "type": "string",
            "description": "Precise definition of this concept, including which visit or condition",
        },
        "selected_raw_code": {
            "type": ["string", "null"],
            "description": "Exact raw_code from catalog. null when unresolved.",
        },
        "filter_expression": {
            "type": ["object", "null"],
            "description": "Column-value inclusion filter, e.g. {'NACCALZD': 1} or {'NACCUDSD': 3}",
        },
        "required": {"type": "boolean"},
        "resolution_status": {
            "type": "string",
            "enum": ["resolved", "unresolved", "not_applicable"],
        },
    },
    "required": ["role", "researcher_label", "semantic_definition", "resolution_status"],
}

_ANALYSIS_PLAN_TOOL: dict[str, Any] = {
    "name": "plan_analysis",
    "description": (
        "Decompose an Alzheimer's dataset research question into a structured analysis plan.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. NEVER treat a compound phrase like 'age at first AD diagnosis' as a single variable.\n"
        "   Decompose it: value_variable=NACCAGE, event.filter_expression={'NACCALZD':1},\n"
        "   visit_anchor='first_matching_event_visit'.\n"
        "2. For selected_raw_code, use ONLY raw_codes from the CATALOG CONTEXT.\n"
        "   Return null if no matching catalog variable exists.\n"
        "3. AD SEMANTICS — do NOT conflate:\n"
        "   'AD' / 'Alzheimer\\'s disease' → selected_raw_code='NACCALZD', filter={'NACCALZD':1}\n"
        "   'dementia' (all-cause)          → selected_raw_code='NACCUDSD', filter={'NACCUDSD':4}\n"
        "   'AD dementia'                   → both: NACCUDSD=4 AND NACCALZD=1\n"
        "4. visit_anchor: 'baseline' | 'first_matching_event_visit' | 'last_visit' | "
        "'first_conversion_visit'\n"
        "5. analysis_type: 'group_comparison' | 'event_anchored_stat' | 'aggregate_stat' | "
        "'descriptive_stat' | 'regression' | 'cohort_size' | 'lookup'\n"
        "6. For group_comparison: group_variable = binary predictor, outcome_variable = metric.\n"
        "7. For event_anchored_stat: value_variable = thing being measured (e.g. NACCAGE),\n"
        "   event = anchor condition (e.g. NACCALZD==1), visit_anchor = when to measure.\n"
        "8. researcher_label must be a SIMPLE ATOMIC concept ('age', 'MMSE score', 'sex').\n"
        "   NEVER embed event context in researcher_label:\n"
        "   BAD: 'age at first AD diagnosis', 'MMSE when diagnosed', 'age at conversion'\n"
        "   GOOD: value_variable.researcher_label='age',\n"
        "         event.researcher_label='first AD diagnosis', event.filter_expression={'NACCALZD':1}\n"
        "9. EXAMPLE — 'average age when first diagnosed with AD':\n"
        "   analysis_type='event_anchored_stat', statistic='mean',\n"
        "   value_variable={role='value_variable', researcher_label='age', selected_raw_code='NACCAGE'},\n"
        "   event={role='event', researcher_label='first AD diagnosis',\n"
        "          selected_raw_code='NACCALZD', filter_expression={'NACCALZD':1}},\n"
        "   visit_anchor='first_matching_event_visit',\n"
        "   cohort=null (no MCI default — leave null unless question specifies a cohort),\n"
        "   candidate_label='average age at first AD diagnosis'"
    ),
    "input_schema": {
        "type": "object",
        "required": ["analysis_type", "research_intent_summary", "requires_longitudinal_data"],
        "properties": {
            "analysis_type": {
                "type": "string",
                "enum": [
                    "group_comparison", "event_anchored_stat", "aggregate_stat",
                    "descriptive_stat", "regression", "cohort_size", "lookup",
                    "modeling_design",
                ],
            },
            "research_intent_summary": {"type": "string"},
            "candidate_label": {"type": "string"},
            "horizon_years": {"type": ["integer", "null"], "minimum": 1},
            "id_space": {"type": "string", "enum": ["A", "B"]},
            "unit_of_analysis": {"type": ["string", "null"]},
            "requires_longitudinal_data": {"type": "boolean"},
            "execution_steps": {"type": "array", "items": {"type": "string"}},
            "statistic": {
                "type": ["string", "null"],
                "enum": ["mean", "median", "std", "min", "max", "count", None],
            },
            "visit_anchor": {
                "type": ["string", "null"],
                "enum": [
                    "baseline", "first_matching_event_visit",
                    "last_visit", "first_conversion_visit", None,
                ],
            },
            "statistical_method_hint": {"type": ["string", "null"]},
            "unresolved_items": {"type": "array", "items": {"type": "string"}},
            "cohort": {**_ROLE_SUBSCHEMA, "type": ["object", "null"]},
            "event": {**_ROLE_SUBSCHEMA, "type": ["object", "null"]},
            "value_variable": {**_ROLE_SUBSCHEMA, "type": ["object", "null"]},
            "group_variable": {**_ROLE_SUBSCHEMA, "type": ["object", "null"]},
            "outcome_variable": {**_ROLE_SUBSCHEMA, "type": ["object", "null"]},
            "covariates": {"type": "array", "items": _ROLE_SUBSCHEMA},
        },
    },
}

# ---- Mapping constants for _derive_structured_intent_from_plan ----

_ANALYSIS_TYPE_TO_QUESTION_TYPE: dict[str, QuestionType] = {
    "group_comparison": QuestionType.STATISTICAL_ANALYSIS,
    "event_anchored_stat": QuestionType.AGGREGATE_STAT,
    "aggregate_stat": QuestionType.AGGREGATE_STAT,
    "descriptive_stat": QuestionType.STATISTICAL_ANALYSIS,
    "regression": QuestionType.STATISTICAL_ANALYSIS,
    "cohort_size": QuestionType.COHORT_SIZE,
    "lookup": QuestionType.LOOKUP,
    "modeling_design": QuestionType.MODELING_DESIGN,
}

_NACCUDSD_CODE_TO_LABEL: dict[int, str] = {
    1: "normal", 2: "impaired-not-mci", 3: "mci", 4: "dementia",
}

_VISIT_ANCHOR_MAP: dict[str, str] = {
    "baseline": "baseline",
    "first_matching_event_visit": "first_diagnosis",
    "last_visit": "last_visit",
    "first_conversion_visit": "first_conversion_visit",
}


def _derive_structured_intent_from_plan(
    plan: AnalysisPlan,
    nl: str,
    parse_debug: ParseDebugInfo | None = None,
) -> StructuredIntent:
    """Build a StructuredIntent from a validated AnalysisPlan.

    Maps each AnalysisPlan role to the corresponding StructuredIntent fields so
    that existing orchestrator routing, narration, and hallucination-guard code
    continues to work.  When a plan role has no selected_raw_code (validation
    rejected it), the corresponding column field is set to None.

    parse_debug — carries raw LLM plan JSON + normalized plan from _plan_analysis_internal.
    """
    q_type = _ANALYSIS_TYPE_TO_QUESTION_TYPE.get(
        plan.analysis_type, QuestionType.STATISTICAL_ANALYSIS
    )

    # Derive target_population and target_etiology from cohort + event filter expressions.
    # No default prior: leave target_population empty when the question doesn't specify one.
    target_population: list[str] = []
    target_etiology: list[str] | None = None

    cohort_fx: dict[str, Any] = (plan.cohort.filter_expression or {}) if plan.cohort else {}
    event_fx: dict[str, Any] = (plan.event.filter_expression or {}) if plan.event else {}

    if "NACCUDSD" in cohort_fx:
        lbl = _NACCUDSD_CODE_TO_LABEL.get(int(cohort_fx["NACCUDSD"]))
        if lbl:
            target_population = [lbl]
    if "NACCALZD" in cohort_fx and cohort_fx["NACCALZD"] == 1:
        target_etiology = ["ad"]

    if "NACCALZD" in event_fx and event_fx["NACCALZD"] == 1:
        target_etiology = ["ad"]
    if "NACCUDSD" in event_fx and not target_population:
        lbl = _NACCUDSD_CODE_TO_LABEL.get(int(event_fx["NACCUDSD"]))
        if lbl:
            target_population = [lbl]

    # No MCI default — only set when the question explicitly specifies MCI/normal/dementia.

    event_anchor = _VISIT_ANCHOR_MAP.get(plan.visit_anchor or "") if plan.visit_anchor else None

    # Normalize horizon: LLM sometimes emits 0 as a placeholder for non-longitudinal questions.
    # Only keep 0 when the question text literally says "0-year" / "zero year".
    raw_horizon = plan.horizon_years
    if isinstance(raw_horizon, int) and raw_horizon <= 0:
        if not _EXPLICIT_ZERO_HORIZON_RE.search(nl):
            raw_horizon = None

    # For event_anchored_stat, map value_variable → outcome_column so existing
    # orchestrator fallback code can still reference it if needed.
    outcome_col = (
        plan.outcome_variable.selected_raw_code if plan.outcome_variable else None
    ) or (
        plan.value_variable.selected_raw_code if plan.value_variable else None
    )

    # --- value_variable_label: must be a simple atomic attribute.
    # Pre-check for compound phrases here before the Pydantic validator fires,
    # so we can record the failure mode in parse_debug rather than raise.
    debug_errors: list[str] = list(parse_debug.validation_errors) if parse_debug else []
    failure_mode: str | None = parse_debug.failure_mode if parse_debug else None

    raw_vv_label = plan.value_variable.researcher_label if plan.value_variable else None
    if raw_vv_label and _is_compound_event_phrase(raw_vv_label):
        debug_errors.append(
            f"DECOMPOSITION_ERROR: value_variable.researcher_label {raw_vv_label!r} "
            "is a compound event phrase. Decompose: value_variable = simple attribute, "
            "event = anchor condition, visit_anchor = when to measure."
        )
        failure_mode = "variable_resolution"
        value_variable_label = None
        value_column = None
    else:
        value_variable_label = raw_vv_label
        value_column = plan.value_variable.selected_raw_code if plan.value_variable else None

    final_debug = ParseDebugInfo(
        raw_llm_plan_json=parse_debug.raw_llm_plan_json if parse_debug else {},
        normalized_plan=parse_debug.normalized_plan if parse_debug else {},
        validation_errors=debug_errors,
        failure_mode=failure_mode,
    )

    return StructuredIntent(
        target_population=target_population,
        target_etiology=target_etiology,
        candidate_label=plan.candidate_label or plan.research_intent_summary[:100],
        horizon_years=raw_horizon,
        modalities=[],
        question_type=q_type,
        pre_index_only=True,
        id_space=IdSpace(plan.id_space),
        stat_variable=None,  # not populated from plan; use value_variable_label instead
        stat_type=plan.statistic or "mean",
        event_anchor=event_anchor,
        analysis_type=plan.analysis_type,
        group_variable=(plan.group_variable.researcher_label if plan.group_variable else None),
        outcome_variable=(plan.outcome_variable.researcher_label if plan.outcome_variable else None),
        statistical_method_hint=plan.statistical_method_hint,
        # Resolved columns
        cohort_column=(plan.cohort.selected_raw_code if plan.cohort else None),
        cohort_filter=(plan.cohort.filter_expression if plan.cohort else None),
        group_column=(plan.group_variable.selected_raw_code if plan.group_variable else None),
        group_variable_label=(plan.group_variable.researcher_label if plan.group_variable else None),
        outcome_column=outcome_col,
        outcome_variable_label=(plan.outcome_variable.researcher_label if plan.outcome_variable else None),
        event_column=(plan.event.selected_raw_code if plan.event else None),
        event_label=(plan.event.researcher_label if plan.event else None),
        # Structured value-variable fields (plan path)
        value_variable_label=value_variable_label,
        value_column=value_column,
        event_filter=(plan.event.filter_expression if plan.event else None),
        unit_of_analysis=plan.unit_of_analysis,
        covariate_columns=[c.selected_raw_code for c in plan.covariates if c.selected_raw_code],
        leakage_assessment=list(plan.leakage_assessment),
        unresolved_fields=list(plan.unresolved_items),
        parsing_mode="llm",
        analysis_plan=plan,
        parse_debug=final_debug,
    )


def _plan_analysis_internal(
    nl: str,
    client: Any,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
) -> tuple[AnalysisPlan, ParseDebugInfo]:
    """Internal variant — returns (AnalysisPlan, ParseDebugInfo) for _parse_with_llm.

    The ParseDebugInfo carries raw_llm_plan_json + normalized_plan for model-debug
    output and A/B harness inspection.  The public plan_analysis() just drops debug.
    """
    catalog_context = _build_catalog_context(nl, catalog or [])

    user_message = (
        "Decompose this research question into a structured analysis plan.\n\n"
        f"QUESTION: {nl}\n\n"
        "CATALOG CONTEXT (use only raw_codes from this list for selected_raw_code fields):\n"
        f"{catalog_context}\n\n"
        "CRITICAL DECOMPOSITION RULES:\n"
        "1. researcher_label = SIMPLE attribute ('age', 'MMSE', 'sex'). "
        "NEVER 'age at first AD diagnosis' or 'MMSE when diagnosed'.\n"
        "2. For 'average age when first diagnosed with AD':\n"
        "   value_variable={researcher_label='age', selected_raw_code='NACCAGE'}\n"
        "   event={researcher_label='first AD diagnosis', selected_raw_code='NACCALZD', "
        "filter_expression={'NACCALZD':1}}\n"
        "   visit_anchor='first_matching_event_visit', statistic='mean'\n"
        "   candidate_label='average age at first AD diagnosis'\n"
        "   cohort=null  ← do NOT add MCI/dementia cohort unless question specifies one\n"
        "3. 'AD'/'Alzheimer\\'s disease' → NACCALZD==1 (etiologic). "
        "'dementia' (all-cause) → NACCUDSD==4.\n"
        "4. candidate_label must match the specific question — not 'conversion to dementia' "
        "unless the question is literally about dementia conversion."
    )

    messages = [{"role": "user", "content": user_message}]

    # Provider-neutral dispatch: use invoke_tool() when the client is a real
    # PlannerClient subclass (AnthropicPlannerClient or HFOpenAIPlannerClient);
    # otherwise fall back to the legacy anthropic.Anthropic SDK path.
    # isinstance is used instead of hasattr to avoid MagicMock false-positives.
    if isinstance(client, PlannerClient):
        plan_data = client.invoke_tool(messages, _ANALYSIS_PLAN_TOOL, max_tokens=1500)
        if plan_data is None:
            raise RuntimeError("LLM did not invoke plan_analysis tool.")
    else:
        response = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=1500,
            tools=[_ANALYSIS_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "plan_analysis"},
            messages=messages,
        )
        plan_data = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "plan_analysis":
                plan_data = dict(block.input)
                break
        if plan_data is None:
            raise RuntimeError("LLM did not invoke plan_analysis tool.")

    plan_data["original_question"] = nl
    plan_data.setdefault("candidate_label", "")
    plan_data.setdefault("id_space", "A")
    plan_data.setdefault("covariates", [])
    plan_data.setdefault("unresolved_items", [])
    # Some LLMs (e.g. deepseek-chat) invoke the tool but omit required fields.
    # Provide safe defaults so Pydantic validation can proceed with partial data.
    plan_data.setdefault("analysis_type", "modeling_design")
    plan_data.setdefault("research_intent_summary", nl[:200])
    plan_data.setdefault("requires_longitudinal_data", False)
    raw_plan_json = {k: v for k, v in plan_data.items() if k != "original_question"}
    validated = _validate_analysis_plan(plan_data, catalog or [])
    debug = ParseDebugInfo(
        raw_llm_plan_json=raw_plan_json,
        normalized_plan={k: v for k, v in validated.items() if k != "original_question"},
    )
    return AnalysisPlan.model_validate(validated), debug


def plan_analysis(
    nl: str,
    client: Any,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
) -> AnalysisPlan:
    """Decompose a research question into a validated AnalysisPlan.

    Public API: returns AnalysisPlan only.  ParseDebugInfo is available via
    _plan_analysis_internal() for internal use and the A/B harness.

    Raises RuntimeError when the LLM doesn't invoke the plan_analysis tool.
    The caller is responsible for fallbacks.
    """
    plan, _ = _plan_analysis_internal(nl, client, model=model, catalog=catalog)
    return plan


# ---------------------------------------------------------------------------
# LLM parser — catalog-grounded tool-use path  (fill_intent, kept for compat)
# ---------------------------------------------------------------------------

# Manually crafted tool schema — avoids auto-generating a complex nested
# schema from StructuredIntent (which now includes LeakageFinding lists).
_FILL_INTENT_TOOL: dict[str, Any] = {
    "name": "fill_intent",
    "description": (
        "Parse an Alzheimer's dataset research question into structured analytical intent.\n"
        "CRITICAL RULES:\n"
        "1. For raw column fields (cohort_column, group_column, outcome_column, etc.), "
        "use ONLY raw_codes from the CATALOG CONTEXT in the user message. "
        "Return null or omit the field if you cannot find a matching variable.\n"
        "2. Do NOT invent column names not in the catalog.\n"
        "3. NACCUDSD=4 is ALL-CAUSE dementia — not AD-specific. "
        "NACCALZD=1 is etiologic Alzheimer's disease. "
        "Do not conflate them.\n"
        "4. For cohort_filter: use numeric values matching the catalog coding "
        "(e.g. {\"NACCUDSD\": 3} for MCI, {\"NACCUDSD\": 4} for all-cause dementia).\n"
        "5. The statistical_method_hint should name the appropriate test "
        "(e.g. 'Welch two-sample t-test', 'Mann-Whitney U', 'Pearson correlation').\n"
        "6. If a variable name is mentioned explicitly (e.g. 'CVHATT'), look it up "
        "in the catalog first; only use it if found there."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "question_type", "target_population", "candidate_label",
            "id_space", "stat_type",
        ],
        "properties": {
            # Core classification
            "question_type": {
                "type": "string",
                "enum": [
                    "feasibility", "cohort_size", "lookup", "modeling_design",
                    "aggregate_stat", "statistical_analysis",
                ],
            },
            "target_population": {
                "type": "array", "items": {"type": "string"},
                "description": "Cognitive status: 'mci','normal','dementia','impaired-not-mci'. NOT 'ad'.",
            },
            "target_etiology": {
                "type": ["array", "null"], "items": {"type": "string"},
                "description": "['ad'] for AD-specific questions; null otherwise.",
            },
            "candidate_label": {"type": "string"},
            "horizon_years": {"type": ["integer", "null"], "minimum": 1},
            "modalities": {
                "type": "array", "items": {"type": "string"},
                "description": "cognition | mri | pet | fluid_biomarker | vascular | neuropath | demographic",
            },
            "pre_index_only": {"type": "boolean", "default": True},
            "id_space": {"type": "string", "enum": ["A", "B"]},
            # Aggregate stat
            "stat_type": {"type": "string", "enum": ["mean", "median", "std", "min", "max"]},
            "stat_variable": {"type": ["string", "null"]},
            "event_anchor": {
                "type": ["string", "null"],
                "enum": ["first_diagnosis", "last_visit", "first_conversion_visit", "baseline", None],
            },
            # Analysis sub-type
            "analysis_type": {
                "type": ["string", "null"],
                "enum": [
                    "group_comparison", "correlation", "regression",
                    "descriptive_stat", "event_anchored_stat", None,
                ],
            },
            "statistical_method_hint": {"type": ["string", "null"]},
            # Cohort definition
            "cohort_label": {"type": ["string", "null"]},
            "cohort_column": {
                "type": ["string", "null"],
                "description": "Raw catalog column defining cohort. From catalog only. E.g. 'NACCUDSD'.",
            },
            "cohort_filter": {
                "type": ["object", "null"],
                "description": "E.g. {\"NACCUDSD\": 3} for MCI. Integer values only.",
            },
            # Variable resolution
            "group_variable_label": {"type": ["string", "null"]},
            "group_column": {
                "type": ["string", "null"],
                "description": "Raw column for binary grouping. From catalog only.",
            },
            "outcome_variable_label": {"type": ["string", "null"]},
            "outcome_column": {
                "type": ["string", "null"],
                "description": "Raw column for outcome. From catalog only.",
            },
            "covariate_labels": {"type": "array", "items": {"type": "string"}},
            "covariate_columns": {
                "type": "array", "items": {"type": "string"},
                "description": "Raw columns for covariates. From catalog only.",
            },
            "label_variable_label": {"type": ["string", "null"]},
            "label_column": {
                "type": ["string", "null"],
                "description": "Raw column for prediction label. E.g. 'NACCUDSD'.",
            },
            "event_label": {"type": ["string", "null"]},
            "event_column": {
                "type": ["string", "null"],
                "description": "Raw column for event anchor. E.g. 'NACCALZD'.",
            },
            "unresolved_fields": {
                "type": "array", "items": {"type": "string"},
                "description": "Fields you could not map to a catalog variable.",
            },
            # NL fallback fields (kept for backward compat with orchestrator)
            "group_variable": {"type": ["string", "null"]},
            "outcome_variable": {"type": ["string", "null"]},
        },
    },
}


def _parse_with_llm(
    nl: str,
    client: Any,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
) -> StructuredIntent:
    """Catalog-grounded LLM analysis planning with deterministic post-validation.

    Primary path: plan_analysis tool → AnalysisPlan → derived StructuredIntent.
    The planner decomposes compound phrases (e.g. 'age at first AD diagnosis')
    into value_variable + event + visit_anchor before any catalog lookup.

    Fallback: JSON-prompt fill_intent for models that don't support tool_choice.
    """
    try:
        plan, debug = _plan_analysis_internal(nl, client, model=model, catalog=catalog or [])
        return _derive_structured_intent_from_plan(plan, nl, parse_debug=debug)
    except Exception as exc:
        err = str(exc).lower()
        if (
            "tool_choice" in err or "thinking mode" in err or "tool use" in err
            # Pydantic ValidationError when LLM returns a tool call with missing/invalid
            # AnalysisPlan fields (e.g. deepseek-chat omitting required fields).
            # Match only AnalysisPlan validation failures — not StructuredIntent ones.
            or "validation error for analysisplan" in err
            # HF / provider fallback: model did not invoke the tool (returned None).
            # Retry with JSON-only prompt so the same normalization pipeline applies.
            or "did not invoke" in err
        ):
            return _parse_with_json_prompt(nl, client, model, catalog)
        raise


def _parse_with_json_prompt(
    nl: str,
    client: Any,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
) -> StructuredIntent:
    """JSON-prompt fallback for models that don't support tool_choice."""
    catalog_context = _build_catalog_context(nl, catalog or [])

    prompt = (
        "Parse the Alzheimer's research question below. Return ONLY valid JSON.\n\n"
        f"QUESTION: {nl}\n\n"
        f"CATALOG (use only these raw_codes for column fields):\n{catalog_context}\n\n"
        'Required JSON fields:\n'
        '  "question_type": "feasibility"|"cohort_size"|"lookup"|"modeling_design"|"aggregate_stat"|"statistical_analysis"\n'
        '  "target_population": list — "mci","normal","dementia","impaired-not-mci" (NOT "ad")\n'
        '  "target_etiology": ["ad"] for AD-specific; null otherwise\n'
        '  "candidate_label": string\n'
        '  "horizon_years": integer >= 1 or null\n'
        '  "modalities": list — cognition|mri|pet|fluid_biomarker|vascular|neuropath|demographic\n'
        '  "question_type": as above\n'
        '  "stat_type": "mean"|"median"|"std"|"min"|"max"\n'
        '  "analysis_type": "group_comparison"|"correlation"|"regression"|"descriptive_stat"|null\n'
        '  "cohort_label": string or null (e.g. "baseline MCI")\n'
        '  "cohort_column": raw_code from catalog or null (e.g. "NACCUDSD")\n'
        '  "cohort_filter": dict e.g. {"NACCUDSD": 3} or null\n'
        '  "group_variable_label": NL label or null\n'
        '  "group_column": raw_code from catalog or null\n'
        '  "outcome_variable_label": NL label or null\n'
        '  "outcome_column": raw_code from catalog or null\n'
        '  "covariate_labels": list of strings\n'
        '  "covariate_columns": list of raw_codes from catalog\n'
        '  "label_column": raw_code for prediction label or null\n'
        '  "event_column": raw_code for event or null\n'
        '  "statistical_method_hint": string or null\n'
        '  "unresolved_fields": list of field names you could not resolve\n'
        '  "id_space": "A" or "B"\n'
        '  "pre_index_only": true/false\n'
        '  "group_variable": NL string or null\n'
        '  "outcome_variable": NL string or null\n\n'
        "REMINDER: NACCUDSD=4 is all-cause dementia, NOT AD. NACCALZD=1 is etiologic AD.\n\n"
        "JSON:"
    )

    # Provider-neutral dispatch for text completion.
    if isinstance(client, PlannerClient):
        raw = client.complete_text(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        ).strip()
    else:
        response = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text_block = next((b for b in response.content if b.type == "text"), None)
        if text_block is None:
            raise RuntimeError("LLM returned no text content in JSON-prompt fallback.")
        raw = text_block.text.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        raw = "\n".join(inner[:-1] if inner and inner[-1].strip() == "```" else inner)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned invalid JSON: {exc}\nRaw (first 400 chars): {raw[:400]}"
        ) from exc

    normalized = _normalize_llm_output(data, nl)
    validated = _validate_resolved_intent(normalized, catalog or [])
    return StructuredIntent.model_validate(validated)


# ---------------------------------------------------------------------------
# Enriched planner — produces EnrichedAnalysisPlan with composable operations
# ---------------------------------------------------------------------------

_VALID_ENRICHED_OP_TYPES: frozenset[str] = frozenset({
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

_ENRICHED_PLAN_TOOL: dict[str, Any] = {
    "name": "plan_enriched_analysis",
    "description": (
        "Decompose an Alzheimer's dataset research question into an enriched analysis plan "
        "with explicit composable operations.\n\n"
        "CRITICAL RULES:\n"
        "1. researcher_audit_trace: write 3-6 concise statements describing how you interpreted "
        "   the question. These are RESEARCHER-VISIBLE — not hidden reasoning. Examples:\n"
        "   'Interpreted baseline MCI as baseline_diagnosis=mci (NACCUDSD=3).'\n"
        "   'Extracted horizons [2, 3, 5] from 2, 3, or 5 years.'\n"
        "   'This question counts participants, not visits.'\n"
        "   'Requires real longitudinal visit data, not catalog-only lookup.'\n"
        "2. For questions with multiple horizons (e.g. '2, 3, or 5 years'), emit ONE "
        "   longitudinal_feasibility operation with parameters.horizons=[2,3,5].\n"
        "3. Operations are ordered: count_eligible first, then longitudinal_feasibility, "
        "   then modality, then stat operations.\n"
        "4. outcome_definition must use variable codes when the outcome is syndromic "
        "   (e.g. 'NACCUDSD=4') or etiologic (e.g. 'NACCALZD=1').\n"
        "5. AD SEMANTICS: 'AD' / 'Alzheimer's disease' → NACCALZD=1 (etiologic). "
        "   'dementia' (all-cause) → NACCUDSD=4. Do not conflate them.\n"
        "6. ETIOLOGY FILTER SEMANTICS (critical for correct participant counts):\n"
        "   a. 'diagnosed as AD' / 'ever had AD' / 'first diagnosed with AD' (any-visit):\n"
        "      use filter={\"event_etiology\": \"ad\"} in BOTH count_eligible AND variable_stat ops.\n"
        "      event_etiology counts/anchors on ANY visit with NACCALZD==1 (incident+prevalent).\n"
        "   b. 'AD at baseline' / 'AD at enrollment' / 'AD at first visit' (baseline only):\n"
        "      use filter={\"etiology\": \"ad\"} (maps to baseline_etiology, first visit only).\n"
        "   NOTE: variable_stat with event_anchor='first_diagnosis' auto-promotes etiology to\n"
        "   event_etiology — but count_eligible REQUIRES explicit event_etiology in the filter.\n"
        "7. baseline_definition: structured dict, e.g. {\"diagnosis\": \"mci\"} not prose.\n"
        "8. id_space: 'A' for NACC/ADSP-PHC questions, 'B' for ADNI-only questions.\n"
        "9. requires_backend_data=false ONLY for pure lookup/leakage-check questions.\n"
        "10. VARIABLE-ID DISCIPLINE (critical): every variable_id must be COPIED VERBATIM from "
        "    the catalog context's variable_id column — NEVER construct, shorten, or guess one, "
        "    even if you know the field name. For Space B (ADNI) the field name ALONE is NOT "
        "    sufficient: the same field can exist in multiple tables (e.g. a raw biomarker code "
        "    can appear in both a CSF table and a PLASMA table), so you must copy the FULL "
        "    'ADNI.<TABLE>.<FIELD>' string exactly as shown — do not assume or invent a table "
        "    name. The one supported derived id is 'ADNI.DERIVED.age_at_event' (Space B). NEVER "
        "    invent an id (no 'ADNI.OTHER.*', no 'ADNI.<TABLE>.NACCAGE'), and for id_space='B' "
        "    NEVER emit a NACC.* id. If no catalog row's variable_id matches what the question "
        "    asks for, use explore_catalog to search rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "research_intent_summary",
            "researcher_audit_trace",
            "operations",
            "requires_backend_data",
        ],
        "properties": {
            "research_intent_summary": {
                "type": "string",
                "description": "One-sentence summary of what the researcher is asking.",
            },
            "researcher_audit_trace": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Concise, researcher-visible audit of key interpretation decisions. "
                    "3-6 statements. Not chain-of-thought."
                ),
            },
            "operations": {
                "type": "array",
                "description": "Ordered list of composable operations to execute.",
                "items": {
                    "type": "object",
                    "required": ["op_type", "output_name"],
                    "properties": {
                        "op_type": {
                            "type": "string",
                            "enum": sorted(_VALID_ENRICHED_OP_TYPES),
                            "description": (
                                "count_eligible: count participants matching a filter. "
                                "longitudinal_feasibility: check follow-up availability "
                                "  (use parameters.horizons=[list of ints]). "
                                "modality_overlap: count participants with multiple modalities "
                                "  (use parameters.modalities=[list]). "
                                "variable_stat: compute mean/median/etc of a variable "
                                "  (Space A id_space='A': variable_id must be "
                                "  \"NACC.<COLUMN_NAME>\", e.g. \"NACC.NACCAGE\". "
                                "  Space B id_space='B': variable_id must be "
                                "  \"ADNI.<TABLE>.<FIELD>\", e.g. \"ADNI.MMSE.MMSCORE\", "
                                "  or the derived \"ADNI.DERIVED.age_at_event\" for age at an "
                                "  event visit; NEVER a NACC.* id in Space B. "
                                "  use event_anchor for event-specific rows). "
                                "descriptive_stat: full descriptive stats for a variable. "
                                "lookup_variable: find a variable in the catalog "
                                "  (use parameters.query). "
                                "leakage_check: audit features for leakage risk. "
                                "modeling_design: suggest a complete modeling design. "
                                "explore_catalog: list ALL catalog entries matching a keyword "
                                "  (use parameters.keyword), unlike lookup_variable which "
                                "  returns only one best match — use this FIRST when a question "
                                "  asks what variants/kinds/subtypes of something exist "
                                "  (e.g. 'what PTAU data do we have') before running a stat op. "
                                "describe_variable_family: report per-variant subject coverage "
                                "  for a tagged variable family (use parameters.family, e.g. "
                                "  \"ptau217\") — answers 'do we have enough X data, broken down "
                                "  by subtype/kind' by grouping catalog entries that share a "
                                "  family tag (fluid type, assay level, etc.) and counting "
                                "  distinct subjects per variant. "
                                "group_comparison: Welch t-test of a continuous outcome between "
                                "  EXACTLY TWO groups defined by a BINARY variable "
                                "  (group_variable_id, outcome_variable_id both required). "
                                "  DO NOT use this for comparing a measure across THREE OR MORE "
                                "  named categories/diagnostic groups (e.g. 'CN vs MCI vs AD', "
                                "  'compare p-tau181 across CN/MCI/AD') — there is no valid "
                                "  binary group_variable_id for a 3+-way comparison, and forcing "
                                "  one would silently answer a different, narrower question. "
                                "  For 3+ groups, instead emit ONE count_eligible + ONE "
                                "  variable_stat op PER group, each with its own filter and a "
                                "  distinct output_name (e.g. 'cn_ptau181', 'mci_ptau181', "
                                "  'ad_ptau181') — the same per-group pattern used for any "
                                "  multi-cohort question in this schema."
                            ),
                        },
                        "parameters": {
                            "type": "object",
                            "description": (
                                "Op-specific parameters. "
                                "For longitudinal_feasibility: {\"horizons\": [2,3,5], "
                                "  \"filter\": {\"diagnosis\": \"mci\"}}. "
                                "For count_eligible: {\"filter\": {\"diagnosis\": \"mci\"}}. "
                                "For variable_stat: variable_id MUST use the format "
                                "  \"NACC.<COLUMN_NAME>\" where COLUMN_NAME is the exact "
                                "  uppercase NACC column (e.g. \"NACC.NACCAGE\" for age at "
                                "  visit, \"NACC.MMSE\" for MMSE score). "
                                "  For age at first AD diagnosis in Space A (NACC) use: "
                                "  {\"variable_id\": \"NACC.NACCAGE\", \"stat_type\": \"mean\", "
                                "   \"event_anchor\": \"first_diagnosis\", "
                                "   \"filter\": {\"event_etiology\": \"ad\"}}. "
                                "  For age at first AD diagnosis in Space B (ADNI) use the "
                                "  derived variable: {\"variable_id\": \"ADNI.DERIVED.age_at_event\", "
                                "   \"stat_type\": \"mean\", \"event_anchor\": \"first_diagnosis\", "
                                "   \"filter\": {\"event_etiology\": \"ad\"}}. In ADNI, etiologic AD "
                                "  and all-cause dementia are DIFFERENT: emit a SEPARATE op with "
                                "  filter {\"diagnosis\": \"dementia\"} + event_anchor "
                                "  \"first_conversion_visit\" if the all-cause figure is also wanted. "
                                "  (Use event_etiology not etiology when anchoring to a "
                                "  first-diagnosis event — etiology only checks the first visit.) "
                                "  event_anchor options: null (baseline/first visit), "
                                "  \"first_diagnosis\" (first visit matching etiology/dx filter), "
                                "  \"first_conversion_visit\" (first dementia visit), "
                                "  \"last_visit\". "
                                "For modality_overlap: {\"modalities\": [\"mri\", \"pet\"]}. "
                                "For lookup_variable: {\"query\": \"MMSE score\"}. "
                                "For explore_catalog: {\"keyword\": \"ptau\"}. "
                                "For describe_variable_family: {\"family\": \"ptau217\"} "
                                "  (family value must come from a family/variant explore_catalog "
                                "  match — do not invent one). "
                                "For group_comparison (TWO groups only — see op_type description "
                                "  above for why 3+ named groups must NOT use this): "
                                "  {\"group_variable_id\": \"NACC.<BINARY_COLUMN>\", "
                                "   \"outcome_variable_id\": \"NACC.<CONTINUOUS_COLUMN>\"}."
                            ),
                        },
                        "output_name": {
                            "type": "string",
                            "description": (
                                "Unique snake_case key for this result, e.g. 'mci_cohort_n', "
                                "'mci_followup_by_horizon', 'mri_pet_overlap'."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": "One-sentence description of what this op computes.",
                        },
                    },
                },
            },
            "requested_horizons": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "description": "All horizons (in years) explicitly mentioned in the question.",
            },
            "requires_backend_data": {
                "type": "boolean",
                "description": (
                    "True when ops need real participant-level data. "
                    "False only for pure lookup or leakage-check questions."
                ),
            },
            "baseline_definition": {
                "type": "object",
                "description": (
                    "Structured cohort definition. "
                    "e.g. {\"diagnosis\": \"mci\"} or {\"diagnosis\": \"mci\", \"etiology\": \"ad\"}."
                ),
            },
            "outcome_definition": {
                "type": "string",
                "description": (
                    "Outcome variable and code, e.g. 'conversion to NACCUDSD=4 within N years' "
                    "or 'etiologic AD (NACCALZD=1)'."
                ),
            },
            "id_space": {
                "type": "string",
                "enum": ["A", "B"],
                "description": "A = NACC/ADSP-PHC (NACCID), B = ADNI (RID/PTID).",
            },
        },
    },
}


def _build_plan_from_data(
    plan_data: dict[str, Any],
    nl: str,
    original_plan: AnalysisPlan | None = None,
) -> "EnrichedAnalysisPlan | None":
    """Parse and validate a raw plan_data dict into an EnrichedAnalysisPlan.

    Shared by the tool-use path and the JSON-fallback path so validation logic
    is not duplicated.  Returns None when the operations list is empty after
    filtering invalid op_types.
    """
    from alz_agent.schemas import EnrichedAnalysisPlan, PlanOperation

    plan_data = dict(plan_data)
    plan_data["original_question"] = nl
    plan_data.setdefault("research_intent_summary", nl[:200])
    plan_data.setdefault("researcher_audit_trace", [])
    plan_data.setdefault("operations", [])
    plan_data.setdefault("requested_horizons", [])
    plan_data.setdefault("requires_backend_data", True)
    plan_data.setdefault("baseline_definition", {})
    plan_data.setdefault("outcome_definition", "")
    plan_data.setdefault("id_space", "A")

    ops_data: list[dict[str, Any]] = plan_data.get("operations", [])
    if not ops_data:
        return None

    seen_names: set[str] = set()
    operations: list[PlanOperation] = []
    for op_d in ops_data:
        op_type = op_d.get("op_type", "")
        output_name = op_d.get("output_name", "")
        if op_type not in _VALID_ENRICHED_OP_TYPES:
            continue
        if not output_name or output_name in seen_names:
            continue
        seen_names.add(output_name)
        operations.append(
            PlanOperation(
                op_type=op_type,
                parameters=op_d.get("parameters") or {},
                output_name=output_name,
                description=op_d.get("description") or "",
            )
        )

    if not operations:
        return None

    requested_horizons: list[int] = []
    for h in plan_data.get("requested_horizons", []):
        if isinstance(h, int) and h > 0:
            requested_horizons.append(h)

    return EnrichedAnalysisPlan(
        original_question=nl,
        research_intent_summary=str(plan_data.get("research_intent_summary", "")),
        researcher_audit_trace=[str(s) for s in plan_data.get("researcher_audit_trace", [])],
        operations=operations,
        requested_horizons=requested_horizons,
        requires_backend_data=bool(plan_data.get("requires_backend_data", True)),
        baseline_definition=dict(plan_data.get("baseline_definition") or {}),
        outcome_definition=str(plan_data.get("outcome_definition") or ""),
        id_space=str(plan_data.get("id_space") or "A"),
        original_plan=original_plan,
    )


def _plan_enriched_json_fallback(
    nl: str,
    client: Any,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
    *,
    original_plan: AnalysisPlan | None = None,
) -> "EnrichedAnalysisPlan | None":
    """JSON-prompt fallback for plan_enriched_analysis when tool-use path fails.

    Sends a free-text JSON prompt and parses the response.  Used when:
    - invoke_tool() returns None (LLM didn't invoke the tool)
    - tool_choice is not supported by the underlying provider
    - any exception in the primary tool-use path

    Returns None on any failure (the caller falls back to the classic parse path).
    """
    catalog_context = _build_catalog_context(nl, catalog or [])
    valid_op_types = sorted(_VALID_ENRICHED_OP_TYPES)

    prompt = (
        "You are planning a research computation pipeline for an Alzheimer's dataset.\n\n"
        f"QUESTION: {nl}\n\n"
        f"CATALOG CONTEXT (copy the variable_id column verbatim when referencing variables — "
        f"for Space B/ADNI the field name alone is ambiguous across tables, so the full "
        f"'ADNI.<TABLE>.<FIELD>' string is required; never construct or shorten one):\n"
        f"{catalog_context}\n\n"
        "TASK: Produce a JSON computation plan. Return ONLY valid JSON — no markdown, no prose.\n\n"
        "REQUIRED JSON FORMAT:\n"
        "{\n"
        '  "research_intent_summary": "one-sentence summary of what the question asks",\n'
        '  "researcher_audit_trace": [\n'
        '    "Interpretation decision 1 (e.g. Interpreted baseline MCI as diagnosis=mci)",\n'
        '    "Interpretation decision 2 (e.g. Extracted horizons [2,3,5] from text)",\n'
        '    "..."\n'
        '  ],\n'
        '  "operations": [\n'
        '    {\n'
        '      "op_type": "<one of the valid op_type values below>",\n'
        '      "parameters": { ... },\n'
        '      "output_name": "unique_snake_case_key",\n'
        '      "description": "one sentence describing what this op computes"\n'
        '    }\n'
        '  ],\n'
        '  "requested_horizons": [2, 3, 5],\n'
        '  "requires_backend_data": true,\n'
        '  "baseline_definition": {"diagnosis": "mci"},\n'
        '  "outcome_definition": "conversion to NACCUDSD=4 within N years",\n'
        '  "id_space": "A"\n'
        "}\n\n"
        f"VALID op_type VALUES: {valid_op_types}\n\n"
        "PLANNING RULES (read carefully before writing operations):\n"
        "1. For questions counting eligible participants: use op_type='count_eligible'.\n"
        "   parameters: {\"filter\": {\"diagnosis\": \"mci\"}}  ← diagnosis = mci|normal|dementia|impaired-not-mci\n"
        "   AD etiology at baseline (first visit only): {\"filter\": {\"etiology\": \"ad\"}}.\n"
        "   AD etiology any-visit / 'diagnosed as AD' / 'ever had AD': {\"filter\": {\"event_etiology\": \"ad\"}}.\n"
        "   CRITICAL: 'diagnosed as AD' / 'first diagnosed with AD' requires event_etiology (not etiology).\n"
        "2. For longitudinal follow-up availability: use op_type='longitudinal_feasibility'.\n"
        "   IMPORTANT: for multiple horizons (e.g. '2, 3, or 5 years'), use ONE op with\n"
        "   parameters: {\"horizons\": [2, 3, 5], \"filter\": {\"diagnosis\": \"mci\"}}\n"
        "   Do NOT emit separate ops for each horizon.\n"
        "3. For computing a variable statistic (mean, median, etc.): use op_type='variable_stat'.\n"
        "   Space A (id_space='A'): variable_id MUST be \"NACC.<COLUMN_NAME>\" "
        "   (e.g. \"NACC.NACCAGE\" for age at visit).\n"
        "   Space B (id_space='B'): variable_id MUST be \"ADNI.<TABLE>.<FIELD>\" "
        "   (e.g. \"ADNI.MMSE.MMSCORE\") or the derived \"ADNI.DERIVED.age_at_event\". "
        "   NEVER use a NACC.* id in Space B and NEVER invent an id (no \"ADNI.OTHER.*\").\n"
        "   parameters: {\"variable_id\": \"NACC.NACCAGE\", \"stat_type\": \"mean\", \"filter\": {...}}\n"
        "   For event-anchored stats (e.g. 'age at first AD diagnosis'): add\n"
        "   \"event_anchor\": \"first_diagnosis\" and use filter {\"event_etiology\": \"ad\"} for AD.\n"
        "   In Space B, use variable_id \"ADNI.DERIVED.age_at_event\" for that question; etiologic AD\n"
        "   and all-cause dementia are different — emit a separate op with filter\n"
        "   {\"diagnosis\": \"dementia\"} + event_anchor \"first_conversion_visit\" for all-cause.\n"
        "   (Use event_etiology when anchoring to first-diagnosis — etiology only checks visit 1.)\n"
        "   event_anchor options: null/omit (baseline), \"first_diagnosis\", "
        "   \"first_conversion_visit\", \"last_visit\".\n"
        "4. For modality counts: use op_type='modality_overlap'.\n"
        "   parameters: {\"modalities\": [\"cognition\", \"mri\"]}\n"
        "5. id_space: 'A' for NACC/ADSP-PHC questions (NACCID), 'B' for ADNI-only (RID/PTID).\n"
        "6. output_name must be unique across all operations (snake_case, no spaces).\n"
        "7. Order operations: count_eligible first, then longitudinal/stat, then synthesis.\n"
        "8. For 'what variants/kinds/subtypes of X do we have' questions (e.g. 'what PTAU data\n"
        "   do we have, CSF or plasma, 181 or 217'): use op_type='explore_catalog' FIRST\n"
        "   with parameters: {\"keyword\": \"ptau\"} to see what's catalogued, THEN\n"
        "   op_type='describe_variable_family' with parameters: {\"family\": \"ptau217\"}\n"
        "   (family value must be one you saw in the explore_catalog result — never invent one)\n"
        "   to get per-variant (fluid type / assay level) distinct-subject coverage counts.\n"
        "   This pattern generalizes beyond biomarkers to any tagged family (imaging tracer,\n"
        "   pipeline version, etc.) — do not hardcode a fixed list of biomarker names.\n"
        "9. group_comparison compares EXACTLY TWO groups via a binary group_variable_id — it is\n"
        "   a Welch t-test template, not a general multi-group comparison tool. For a question\n"
        "   comparing a measure across THREE OR MORE named categories/diagnostic groups (e.g.\n"
        "   'compare p-tau181 across CN, MCI, and AD'), do NOT use op_type='group_comparison' —\n"
        "   there is no valid binary group_variable_id for a 3+-way split, and forcing one would\n"
        "   silently answer a narrower, different question. Instead emit ONE count_eligible + ONE\n"
        "   variable_stat op PER named group, each with its own filter (e.g. {\"diagnosis\": \"mci\"})\n"
        "   and a distinct output_name (e.g. 'cn_ptau181', 'mci_ptau181', 'ad_ptau181').\n\n"
        "AD SEMANTICS (critical):\n"
        "  - 'MCI' / 'mild cognitive impairment' → diagnosis='mci' (NACCUDSD=3)\n"
        "  - 'dementia' (all-cause) → diagnosis='dementia' (NACCUDSD=4)\n"
        "  - 'AD at baseline / first visit' → etiology='ad' in filter (NACCALZD=1 at visit 1 only)\n"
        "  - 'diagnosed as AD' / 'ever had AD' / 'first diagnosed with AD' (any-visit):\n"
        "    → event_etiology='ad' in filter (NACCALZD=1 at ANY visit; n=25,639 not 19,576)\n"
        "  - Do NOT use diagnosis='ad' — that is invalid and will cause abstain.\n\n"
        "JSON:"
    )

    try:
        if isinstance(client, PlannerClient):
            raw = client.complete_text(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            ).strip()
        else:
            response = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text_block = next((b for b in response.content if b.type == "text"), None)
            if text_block is None:
                return None
            raw = text_block.text.strip()

        # Strip markdown code fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            inner = lines[1:] if lines[0].startswith("```") else lines
            raw = "\n".join(inner[:-1] if inner and inner[-1].strip() == "```" else inner)

        data = json.loads(raw)
        return _build_plan_from_data(data, nl, original_plan=original_plan)

    except Exception:  # noqa: BLE001
        return None


def plan_enriched_analysis(
    nl: str,
    client: Any,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
    *,
    original_plan: AnalysisPlan | None = None,
) -> "EnrichedAnalysisPlan | None":
    """Call the enriched planner tool and return an EnrichedAnalysisPlan.

    Primary path: invoke the plan_enriched_analysis tool via tool_choice.
    Fallback path: JSON-prompt completion when tool invocation fails or returns None.

    Returns None when both paths fail; the caller falls back to the classic parse path.

    Validates the LLM response for:
    - output_name uniqueness across operations
    - all op_type values in the allowed set
    - requested_horizons are positive integers
    - operations list is non-empty

    original_plan: if provided, embedded as EnrichedAnalysisPlan.original_plan for
    backward-compat callers that still read AnalysisPlan.analysis_type.
    """
    catalog_context = _build_catalog_context(nl, catalog or [])

    user_message = (
        "Decompose this research question into an enriched analysis plan with composable operations.\n\n"
        f"QUESTION: {nl}\n\n"
        "CATALOG CONTEXT (use raw_codes from this list when referencing variables):\n"
        f"{catalog_context}\n\n"
        "DECOMPOSITION RULES:\n"
        "- Extract all time horizons mentioned (e.g. '2, 3, or 5 years' → requested_horizons=[2,3,5]).\n"
        "- For multi-horizon questions, use ONE longitudinal_feasibility op with horizons=[2,3,5].\n"
        "- baseline_definition: {\"diagnosis\": \"mci\"} not prose text.\n"
        "- AD etiology filter: use {\"etiology\": \"ad\"} not {\"diagnosis\": \"ad\"}.\n"
        "- outcome_definition: use variable codes (e.g. 'NACCUDSD=4' for all-cause dementia).\n"
        "- researcher_audit_trace: 3-6 concise statements, researcher-visible, not reasoning."
    )

    messages = [{"role": "user", "content": user_message}]

    # ---- Primary path: forced tool invocation ----
    plan_data: dict[str, Any] | None = None
    _tool_error: Exception | None = None
    try:
        if isinstance(client, PlannerClient):
            plan_data = client.invoke_tool(
                messages, _ENRICHED_PLAN_TOOL, max_tokens=2000
            )
        else:
            response = client.messages.create(
                model=model or "claude-sonnet-4-6",
                max_tokens=2000,
                tools=[_ENRICHED_PLAN_TOOL],
                tool_choice={"type": "tool", "name": "plan_enriched_analysis"},
                messages=messages,
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "plan_enriched_analysis":
                    plan_data = dict(block.input)
                    break
    except Exception as exc:  # noqa: BLE001
        _tool_error = exc
        plan_data = None

    if plan_data is not None:
        result = _build_plan_from_data(plan_data, nl, original_plan=original_plan)
        if result is not None:
            return result
        # plan_data was not None but produced no valid operations — fall through to fallback

    # ---- Fallback path: JSON-prompt completion ----
    # Triggered when: tool not invoked (None), tool returned empty ops, or any exception.
    return _plan_enriched_json_fallback(
        nl, client, model=model, catalog=catalog, original_plan=original_plan
    )
