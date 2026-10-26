"""Modeling-design evidence planning — LLM decides what to compute; tools compute it.

The LLM is asked to enumerate required evidence items (tool calls and parameters).
The orchestrator then executes those deterministic tools and collects the actual numbers.

Invariant: this module contains NO numeric generation. The LLM only names tools and
parameters; _execute_evidence_plan() calls the actual backend tools.

Public functions:
  _call_modeling_design_planner(...) -> ModelingDesignPlan | None
  _execute_evidence_plan(...)        -> list[ToolResult]
"""

from __future__ import annotations

from typing import Any

from alz_agent.agent.catalog_context import _build_catalog_context
from alz_agent.agent.intent_schema import StructuredIntent
from alz_agent.agent.providers import PlannerClient
from alz_agent.backend.interface import DataBackend
from alz_agent.schemas import (
    CatalogEntry,
    EvidenceNeedItem,
    FilterSpec,
    IdSpace,
    ModelingDesignPlan,
    ToolResult,
)

# ---------------------------------------------------------------------------
# LLM tool schema — used to ask the LLM to enumerate required evidence items.
# The LLM specifies WHAT to compute; the orchestrator executes deterministic tools.
# ---------------------------------------------------------------------------

_MODELING_DESIGN_PLAN_TOOL: dict[str, Any] = {
    "name": "plan_modeling_evidence",
    "description": (
        "Plan what dataset evidence is needed to answer a modeling design question.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. You MUST NOT generate any numeric facts yourself.\n"
        "2. Specify evidence_items: each names a deterministic tool to call with its parameters.\n"
        "   The orchestrator calls those tools and computes the actual numbers.\n"
        "3. AD SEMANTICS:\n"
        "   NACCUDSD==3 → MCI (cognitive status, baseline_diagnosis='mci')\n"
        "   NACCALZD==1 → AD etiologic attribution (baseline_etiology='ad')\n"
        "   NACCUDSD==4 → all-cause dementia (NOT AD-specific)\n"
        "4. For MCI-to-AD conversion, request evidence for:\n"
        "   baseline cohort size, longitudinal feasibility, modality overlaps,\n"
        "   and a full modeling design assessment.\n"
        "5. candidate_modalities must use: cognition|mri|pet|fluid_biomarker|"
        "vascular|neuropath|demographic\n"
        "6. prediction_horizon_years must be >= 1 (default 3 when not specified).\n"
        "7. label_definition_options should list distinct candidate label definitions,\n"
        "   e.g. 'NACCALZD==1 (AD etiologic)', 'NACCUDSD==4 (all-cause dementia)',\n"
        "   'NACCALZD==1 AND NACCUDSD==4 (AD dementia)'."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "modeling_goal", "target_population", "prediction_horizon_years",
            "evidence_items",
        ],
        "properties": {
            "modeling_goal": {
                "type": "string",
                "description": "Short phrase describing the ML prediction goal.",
            },
            "target_population": {
                "type": "string",
                "description": "Human-readable cohort description, e.g. 'baseline MCI (NACCUDSD==3)'.",
            },
            "target_population_filter": {
                "type": "object",
                "description": "e.g. {'NACCUDSD': 3}",
            },
            "target_etiology": {
                "type": ["string", "null"],
                "description": "'ad' when question asks about AD-specific conversion; null otherwise.",
            },
            "label_definition_options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Candidate label definitions, from least to most specific.",
            },
            "prediction_horizon_years": {
                "type": "integer",
                "minimum": 1,
                "description": "Prediction horizon in years. Default 3.",
            },
            "candidate_modalities": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "cognition", "mri", "pet", "fluid_biomarker",
                        "vascular", "neuropath", "demographic",
                    ],
                },
            },
            "candidate_feature_groups": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Feature groups to propose (e.g. 'demographics', 'cognitive scores').",
            },
            "evidence_items": {
                "type": "array",
                "description": "Ordered list of evidence items to compute.",
                "items": {
                    "type": "object",
                    "required": ["item_id", "description", "tool_name"],
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "description": "Short unique identifier, e.g. 'baseline_mci_n'.",
                        },
                        "description": {
                            "type": "string",
                            "description": "What this evidence tells us.",
                        },
                        "tool_name": {
                            "type": "string",
                            "enum": [
                                "compute_cohort_size",
                                "assess_longitudinal_feasibility",
                                "suggest_modeling_design",
                                "assess_modality_overlap",
                            ],
                        },
                        "baseline_diagnosis": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "description": "e.g. ['mci'] for MCI cohort filter.",
                        },
                        "baseline_etiology": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "description": "e.g. ['ad'] for AD etiologic filter.",
                        },
                        "modalities": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "description": "Required for assess_modality_overlap.",
                        },
                        "horizon_years": {
                            "type": ["integer", "null"],
                            "description": "Required for assess_longitudinal_feasibility.",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["required", "optional"],
                        },
                    },
                },
            },
            "leakage_checks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Variable names to flag for leakage review (informational).",
            },
            "documentation_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short queries for the doc retriever (definitions, criteria).",
            },
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string"},
            },
            "unresolved_items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Items that could not be planned (missing data, unknown variables).",
            },
        },
    },
}


def _call_modeling_design_planner(
    nl_question: str,
    intent: StructuredIntent,
    catalog: list[CatalogEntry],
    client: Any,
    model: str | None = None,
) -> ModelingDesignPlan | None:
    """Ask the LLM to produce an evidence plan for a modeling_design question.

    Returns a ModelingDesignPlan (what evidence to compute) or None if the
    LLM fails to invoke the tool (caller falls back to heuristic path).

    Invariant: the plan contains only strings describing WHAT is needed.
    No numeric facts are generated here — those come from tool execution.
    """
    catalog_context = _build_catalog_context(nl_question, catalog)

    # Summarise what modalities are present in the catalog (Space A only)
    avail_modalities = sorted({
        e.modality.value for e in catalog
        if e.id_space == IdSpace.A and e.modality.value != "other"
    })

    user_message = (
        f"Modeling design question: {nl_question}\n\n"
        f"Parsed intent:\n"
        f"  target_population: {intent.target_population}\n"
        f"  target_etiology: {intent.target_etiology}\n"
        f"  horizon_years: {intent.horizon_years} (None → use 3-year default)\n"
        f"  modalities mentioned: {intent.modalities}\n\n"
        f"Available modalities in catalog (Space A): {avail_modalities}\n\n"
        f"CATALOG CONTEXT (for understanding variable names — do not generate numbers):\n"
        f"{catalog_context}\n\n"
        "Plan the evidence items needed to answer this question.\n"
        "For MCI-to-AD conversion, always include:\n"
        "  1. baseline cohort size (compute_cohort_size)\n"
        "  2. longitudinal feasibility (assess_longitudinal_feasibility)\n"
        "  3. full modeling design (suggest_modeling_design)\n"
        "  4. cognition-only overlap (assess_modality_overlap, modalities=['cognition'])\n"
        "  5. cognition+MRI overlap if MRI is available\n"
        "  6. cognition+PET overlap if PET is available\n"
        "  7. cognition+fluid_biomarker overlap if fluid_biomarker is available\n"
        "  8. any modalities explicitly named in the question\n"
    )

    system_prompt = (
        "You are planning what dataset evidence is needed to answer an Alzheimer's ML "
        "modeling design question. You MUST NOT generate any numeric facts yourself. "
        "Specify evidence_items listing which deterministic tools to call. "
        "The orchestrator will execute those tools and compute the actual numbers."
    )

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        if isinstance(client, PlannerClient):
            data = client.invoke_tool(
                messages=messages,
                tool_schema=_MODELING_DESIGN_PLAN_TOOL,
                max_tokens=2000,
            )
            if data is None:
                return None
        else:
            response = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
                max_tokens=2000,
                system=system_prompt,
                tools=[_MODELING_DESIGN_PLAN_TOOL],
                tool_choice={"type": "tool", "name": "plan_modeling_evidence"},
                messages=[{"role": "user", "content": user_message}],
            )
            data = None
            for block in response.content:
                if block.type == "tool_use" and block.name == "plan_modeling_evidence":
                    data = dict(block.input)
                    break
            if data is None:
                return None
    except Exception:
        return None

    # Build EvidenceNeedItem list from the raw dicts
    items: list[EvidenceNeedItem] = []
    for raw in (data.get("evidence_items") or []):
        if not isinstance(raw, dict):
            continue
        known = {k: v for k, v in raw.items() if k in EvidenceNeedItem.model_fields}
        try:
            items.append(EvidenceNeedItem(**known))
        except Exception:
            continue  # skip malformed items

    horizon = max(1, int(data.get("prediction_horizon_years") or 3))

    return ModelingDesignPlan(
        original_question=nl_question,
        modeling_goal=str(data.get("modeling_goal") or ""),
        target_population=str(data.get("target_population") or ""),
        target_population_filter=dict(data.get("target_population_filter") or {}),
        target_etiology=data.get("target_etiology"),
        label_definition_options=list(data.get("label_definition_options") or []),
        prediction_horizon_years=horizon,
        candidate_modalities=list(data.get("candidate_modalities") or []),
        candidate_feature_groups=list(data.get("candidate_feature_groups") or []),
        evidence_items=items,
        leakage_checks=list(data.get("leakage_checks") or []),
        documentation_queries=list(data.get("documentation_queries") or []),
        assumptions=list(data.get("assumptions") or []),
        caveats=list(data.get("caveats") or []),
        unresolved_items=list(data.get("unresolved_items") or []),
    )


def _execute_evidence_plan(
    plan: ModelingDesignPlan,
    intent: StructuredIntent,
    catalog: list[CatalogEntry],
    backend: DataBackend,
) -> list[ToolResult]:
    """Execute each evidence item in a ModelingDesignPlan using deterministic tools.

    Invariant: every number in the returned ToolResults comes from the backend
    or catalog — no LLM-generated values.
    """
    from alz_agent.tools.deterministic import (
        assess_longitudinal_feasibility,
        assess_modality_overlap,
        compute_cohort_size,
        suggest_modeling_design,
    )

    results: list[ToolResult] = []
    horizon_default = max(1, plan.prediction_horizon_years)

    for item in plan.evidence_items:
        # Build FilterSpec from item params (fall back to intent values when absent)
        diag = item.baseline_diagnosis if item.baseline_diagnosis is not None else (
            intent.target_population if intent.target_population else None
        )
        etio = item.baseline_etiology if item.baseline_etiology is not None else (
            intent.target_etiology if intent.target_etiology else None
        )
        filter_spec = FilterSpec(
            baseline_diagnosis=diag or None,
            baseline_etiology=etio or None,
            id_space=intent.id_space,
        )
        horizon = item.horizon_years if item.horizon_years is not None else horizon_default

        if item.tool_name == "compute_cohort_size":
            results.append(compute_cohort_size(filter_spec, backend))

        elif item.tool_name == "assess_longitudinal_feasibility":
            results.append(assess_longitudinal_feasibility(horizon, filter_spec, backend))

        elif item.tool_name == "suggest_modeling_design":
            results.append(suggest_modeling_design(filter_spec, horizon, catalog, backend))

        elif item.tool_name == "assess_modality_overlap":
            modalities = item.modalities
            if not modalities:
                results.append(ToolResult.abstain(
                    f"evidence item '{item.item_id}': assess_modality_overlap requires modalities"
                ))
            else:
                results.append(
                    assess_modality_overlap(modalities, filter_spec, backend, catalog)
                )

        else:
            results.append(ToolResult.abstain(
                f"evidence item '{item.item_id}': unknown tool '{item.tool_name}'"
            ))

    return results
