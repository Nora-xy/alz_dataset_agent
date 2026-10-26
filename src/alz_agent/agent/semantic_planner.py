"""Semantic planning layer — upstream gate before EnrichedAnalysisPlan execution.

Produces a SemanticAnalysisPlan that explicitly commits to semantic distinctions
(population_unit, cohort_scope, etiology vs. all-cause, event anchor, denominator)
before any backend tools run.  The plan is validated by PlanValidator.validate_semantic()
and compiled deterministically to an EnrichedAnalysisPlan by compile_to_enriched().
"""

from __future__ import annotations

import json
import re
from typing import Any

from alz_agent.agent.catalog_context import _build_catalog_context
from alz_agent.agent.llm_intent import _VISIT_ANCHOR_MAP
from alz_agent.agent.providers import PlannerClient
from alz_agent.schemas import (
    CatalogEntry,
    CohortScope,
    EnrichedAnalysisPlan,
    EtiologyDefinition,
    GroupDefinition,
    PlanOperation,
    PopulationUnit,
    SemanticAnalysisPlan,
    ValidationResult,
    VariableSelection,
)

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

_SEMANTIC_PLAN_TOOL: dict[str, Any] = {
    "name": "plan_semantic_analysis",
    "description": (
        "Capture the SEMANTIC commitments of an Alzheimer's dataset research question "
        "BEFORE planning composable operations.  You must resolve every ambiguity "
        "explicitly — leaving cohort_scope='unclear' forces a repair pass.\n\n"
        "CRITICAL SEMANTIC DISTINCTIONS YOU MUST RESOLVE:\n\n"
        "1. population_unit: does the question count PARTICIPANTS (unique people), "
        "   VISIT_ROWs, VARIABLEs, or DATASETs?\n"
        "   - 'how many participants' / 'mean age' / 'n with AD' → participant.\n"
        "   - Raw row count is NEVER a valid participant count.\n"
        "   - requires_unique_participants MUST be true when population_unit=participant.\n\n"
        "2. cohort_scope: which visits define the target cohort?\n"
        "   - 'baseline' = first study visit only.\n"
        "   - 'any_visit' = participant ever had the condition across all visits.\n"
        "   - 'first_event' = anchor on the FIRST visit meeting an event criterion.\n"
        "     Use this when question says 'first diagnosis', 'age at onset', etc.\n"
        "   - 'followup_window' = some post-index follow-up period.\n"
        "   - 'pre_index' = visits before an index event.\n"
        "   - 'unclear' = you cannot resolve — this forces a repair pass.\n\n"
        "3. etiology_definition: 'etiologic_ad' means NACCALZD=1 (AD attribution by "
        "   clinician).  'all_cause_dementia' means NACCUDSD=4 (any dementia, NOT "
        "   AD-specific).  NEVER conflate them.\n"
        "   - 'Alzheimer's disease' / 'etiologic AD' / 'AD diagnosis' → etiologic_ad.\n"
        "   - 'dementia' without qualifier → all_cause_dementia.\n"
        "   - When etiology_definition=etiologic_ad, diagnosis_variable MUST include "
        "     NACCALZD.  NACCUDSD alone is wrong.\n\n"
        "4. first diagnosis vs. first visit:\n"
        "   - 'age at first AD diagnosis' → cohort_scope=first_event, "
        "     event_anchor=first_matching_event_visit, event_filters={NACCALZD:1}.\n"
        "   - 'age at enrollment' → baseline_filters, event_anchor=baseline.\n\n"
        "5. event_filters: used for any-visit / first-event conditions (e.g. NACCALZD=1).\n"
        "   baseline_filters: used for first-visit-only conditions (e.g. diagnosis=mci).\n"
        "   When cohort_scope=first_event + etiology_definition=etiologic_ad:\n"
        "   event_filters MUST contain NACCALZD (not baseline_filters).\n\n"
        "6. denominator_definition: for ANY mean/median/count/proportion, state in "
        "   plain language the exact denominator (e.g. 'unique NACC participants with "
        "   NACCALZD=1 on any visit after event reduction').\n\n"
        "7. known_invalid_interpretations: list 2-4 alternative readings you are "
        "   explicitly ruling out.  Example: 'NOT baseline-only NACCALZD=1 prevalence', "
        "   'NOT visit-row count', 'NOT NACCUDSD=4 (all-cause dementia)'.\n\n"
        "8. VARIABLE-ID DISCIPLINE (critical): statistic_variable and every "
        "   variable_selections[*].variable_id must be COPIED VERBATIM from the "
        "   catalog context's variable_id column — never construct, shorten, or guess "
        "   one. For Space B (ADNI) the bare field name is NOT sufficient to identify a "
        "   variable: the same field name can exist in more than one table (e.g. a raw "
        "   biomarker code can appear in both a CSF table and a PLASMA table), so you "
        "   must copy the FULL 'ADNI.<TABLE>.<FIELD>' string exactly as shown in the "
        "   context. If nothing in the context matches what the question asks for, leave "
        "   the field unresolved rather than inventing an id — the validator will reject "
        "   any id it cannot find in the catalog.\n\n"
        "9. MULTI-GROUP COMPARISONS (e.g. 'compare CN vs. MCI vs. AD'): populate the "
        "   'groups' list with one GroupDefinition per named group instead of using the "
        "   single top-level baseline_filters/event_filters/etiology_definition. Each "
        "   group has its own label, cohort_scope, filters, and etiology_definition — the "
        "   SAME rules from points 1-6 above apply independently to EACH group.\n"
        "   DIAGNOSIS-DEFINING FILTER (CRITICAL — this is the ONLY channel the executor "
        "   reads for a group's diagnosis/clinical-status criterion): every group MUST set "
        "   diagnosis_basis to exactly 'clinical_status' or 'etiologic_ad'.\n"
        "     - diagnosis_basis='clinical_status' → ALSO set clinical_status to exactly "
        "       one of 'normal' | 'mci' | 'dementia' | 'impaired-not-mci'. NEVER express "
        "       this as a raw column name or operator object inside baseline_filters (e.g. "
        "       {'DXMCI': 1}, {'DXAD': {'neq': 1}}) — those are NOT read by the compiler and "
        "       will silently produce an UNFILTERED cohort for that group. If you know the "
        "       underlying raw column (e.g. ADNI DXSUM.DXMCI), that knowledge belongs in "
        "       diagnosis_variable (free text, for the researcher's reference only) — it "
        "       does not replace clinical_status.\n"
        "     - diagnosis_basis='etiologic_ad' → the executor derives the filter "
        "       automatically from this basis; clinical_status is not used.\n"
        "   IMPORTANT: a group label like 'AD' is often ambiguous between two distinct "
        "   diagnostic bases — syndromic all-cause dementia (clinical_status='dementia') "
        "   and etiologic AD attribution (diagnosis_basis='etiologic_ad'). When the "
        "   question does not explicitly disambiguate, do NOT silently pick one: emit TWO "
        "   GroupDefinitions with distinct labels (e.g. 'AD (syndromic dementia)' and "
        "   'AD (etiologic AD)'), one on each basis, so the researcher sees both reported "
        "   separately. Leave 'groups' empty for single-cohort questions — the top-level "
        "   fields are used unchanged.\n\n"
        "10. MULTI-VARIANT MEASURES (e.g. 'CSF p-tau181' when several assay variants "
        "    exist — raw vs. harmonized, or different assay platforms): populate "
        "    'variable_selections' with one VariableSelection per matching catalog "
        "    variable_id instead of the single top-level statistic_variable. Set "
        "    'specimen' and 'analyte' (top-level and/or per-selection) to the semantic "
        "    concept you believe each variable_id satisfies (e.g. specimen='csf', "
        "    analyte='ptau181') — the validator will cross-check these claims against "
        "    the catalog entry's own family/variant/concept tags, so do not guess; only "
        "    set them when the catalog context's family/variant columns support the "
        "    claim. Give each selection a distinct variant_label (e.g. 'csf_raw', "
        "    'csf_harmonized') so variants are reported separately — never pool distinct "
        "    assay variants into one number unless the question explicitly asks for a "
        "    single pooled figure. Leave 'variable_selections' empty for single-variable "
        "    questions — the top-level statistic_variable is used unchanged.\n\n"
        "11. NO TREND/TRAJECTORY CAPABILITY: the compiler can only emit a participant "
        "    count per group and/or one single-timepoint statistic per group anchored on "
        "    ONE visit (baseline / first-matching-event / last visit). There is NO "
        "    operation for a trend, trajectory, slope, rate-of-change, or 'how does X "
        "    change over time/visits'. If the question asks for that in addition to a "
        "    snapshot statistic, still resolve the snapshot part normally, but ALSO list "
        "    the unaddressed trend request in abstain_if_missing verbatim — 'caveats' is "
        "    never surfaced to the researcher, so a note placed only there is silently "
        "    lost and the researcher will wrongly believe the trend was reported.\n\n"
        "12. NO BOUNDED TIME-WINDOW CAPABILITY: cohort_scope/event_anchor select WHICH "
        "    visit anchors a filter or statistic, but there is no way to bound how much "
        "    time may elapse between two visits (e.g. 'converted WITHIN 3 years', "
        "    'within 18 months of baseline'). event_filters/event_anchor can express "
        "    'ever reaches state X at some later visit' (unbounded), never 'reaches X "
        "    within N years'. If the question includes such a time bound, still resolve "
        "    the unbounded version of the condition normally, but list the dropped time "
        "    bound in abstain_if_missing verbatim (e.g. 'the 3-year window could not be "
        "    enforced — this counts conversion at ANY follow-up visit, not within 3 "
        "    years specifically').\n\n"
        "13. DATA-AVAILABILITY / MODALITY-OVERLAP QUESTIONS (e.g. 'how many participants "
        "    have both MRI and CSF biomarker data', 'both amyloid PET and cognition "
        "    data'): this is NOT a plain participant count — set 'required_modalities' "
        "    (top-level, or per-group in groups[*].required_modalities) to the list of "
        "    modalities the question asks about. Valid values: cognition, mri, pet, "
        "    fluid_biomarker, vascular, neuropath, demographic. This compiles to a "
        "    modality_overlap operation that computes the actual intersection count — "
        "    do NOT leave required_modalities empty and let the question fall through to "
        "    a plain baseline count, which silently ignores the modality requirement "
        "    entirely."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "population_unit",
            "requires_unique_participants",
            "cohort_scope",
            "etiology_definition",
            "denominator_definition",
        ],
        "properties": {
            "population_unit": {
                "type": "string",
                "enum": ["participant", "visit_row", "variable", "dataset"],
            },
            "requires_unique_participants": {"type": "boolean"},
            "cohort_scope": {
                "type": "string",
                "enum": [
                    "baseline",
                    "any_visit",
                    "first_event",
                    "followup_window",
                    "pre_index",
                    "unclear",
                ],
            },
            "baseline_filters": {
                "type": "object",
                "description": (
                    "First-visit-only conditions.  E.g. {\"diagnosis\": \"mci\"} "
                    "or {\"etiology\": \"ad\"} for baseline-AD questions."
                ),
            },
            "event_filters": {
                "type": "object",
                "description": (
                    "Any-visit or first-event conditions.  E.g. {\"NACCALZD\": 1} "
                    "when cohort_scope=first_event + etiology_definition=etiologic_ad."
                ),
            },
            "event_anchor": {
                "type": ["string", "null"],
                "enum": [
                    "baseline",
                    "first_matching_event_visit",
                    "first_conversion_visit",
                    "last_visit",
                    None,
                ],
                "description": (
                    "Which visit row to anchor the statistic on.  "
                    "first_matching_event_visit = first visit with NACCALZD=1 (or other event). "
                    "first_conversion_visit = first visit with NACCUDSD>=4. "
                    "Required when cohort_scope=first_event and statistic_variable is set."
                ),
            },
            "time_ordering_variable": {
                "type": "string",
                "description": "Column used to order visits.  Default: NACCVNUM.",
            },
            "outcome_definition": {
                "type": "string",
                "description": "Variable code expression.  E.g. 'NACCALZD=1' or 'NACCUDSD=4'.",
            },
            "etiology_definition": {
                "type": "string",
                "enum": [
                    "etiologic_ad",
                    "all_cause_dementia",
                    "any_diagnosis",
                    "unspecified",
                ],
            },
            "diagnosis_variable": {
                "type": "string",
                "description": (
                    "Column(s) encoding the diagnosis/etiology condition.  "
                    "Must include NACCALZD when etiology_definition=etiologic_ad. "
                    "Comma-separate multiple columns: 'NACCALZD,NACCUDSD'."
                ),
            },
            "statistic_variable": {
                "type": "string",
                "description": (
                    "Column on which to compute the statistic (e.g. NACCAGE).  "
                    "Omit if the question only asks for a count."
                ),
            },
            "statistic_type": {
                "type": ["string", "null"],
                "enum": ["mean", "median", "count", "proportion", "std", "min", "max", None],
            },
            "denominator_definition": {
                "type": "string",
                "description": "Plain-language denominator statement (required for any statistic).",
            },
            "allowed_backend_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Which backend tools are valid for this question.",
            },
            "known_invalid_interpretations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-4 alternative misreadings explicitly ruled out.",
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string"},
            },
            "abstain_if_missing": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Any part of the QUESTION that the available operations cannot "
                    "express, phrased as what is missing (e.g. 'no MRI-per-visit "
                    "variable in catalog', 'no trend/trajectory operation exists — "
                    "only single-timepoint group counts and stats are computable'). "
                    "compile_to_enriched() can ONLY emit: a participant count per "
                    "group, and/or one single-timepoint statistic (mean/median/std/"
                    "min/max/count) per group anchored on ONE visit (baseline / "
                    "first-matching-event / last visit). There is NO operation for "
                    "a trend, trajectory, slope, rate-of-change, or 'how X changes "
                    "over time/visits' — if the question asks for that, put it here "
                    "verbatim rather than silently answering only the cross-sectional "
                    "part and calling it complete. This list is surfaced verbatim to "
                    "the researcher as a caveat on the final answer — 'caveats' is NOT "
                    "surfaced anywhere downstream, so any genuinely unaddressable part "
                    "of the question belongs here, never only in 'caveats'."
                ),
            },
            "id_space": {
                "type": "string",
                "enum": ["A", "B"],
                "description": "'A' for NACC/ADSP-PHC, 'B' for ADNI.",
            },
            "specimen": {
                "type": ["string", "null"],
                "description": (
                    "Free-text specimen/sample type the question asks about (e.g. 'csf', "
                    "'plasma'). Only set when the catalog context's family/variant tags "
                    "support the claim for the selected variable(s) — the validator "
                    "cross-checks this against catalog metadata."
                ),
            },
            "analyte": {
                "type": ["string", "null"],
                "description": (
                    "Free-text analyte/concept the question asks about (e.g. 'ptau181'). "
                    "Only set when the catalog context's family tag supports the claim."
                ),
            },
            "required_modalities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Set ONLY when the question asks how many participants have MULTIPLE "
                    "data types available (e.g. 'have both MRI and CSF biomarker data', "
                    "'both amyloid PET and cognition data'). This compiles to a "
                    "modality_overlap operation (intersection counts) instead of a plain "
                    "participant count. Valid values: cognition, mri, pet, fluid_biomarker, "
                    "vascular, neuropath, demographic. Leave empty for single-cohort/single-"
                    "variable questions. For multi-group questions, set this per-group in "
                    "'groups[*].required_modalities' instead."
                ),
            },
            "groups": {
                "type": "array",
                "description": (
                    "One GroupDefinition per named cohort for multi-group comparison "
                    "questions (e.g. CN vs. MCI vs. AD). Leave empty for single-cohort "
                    "questions — the top-level baseline_filters/event_filters/"
                    "etiology_definition are used instead. "
                    "CRITICAL: every group MUST set diagnosis_basis (and clinical_status "
                    "when diagnosis_basis='clinical_status') — see field descriptions below. "
                    "These are the ONLY channel the executor recognizes for a group's "
                    "diagnosis-defining filter; baseline_filters is NOT read for this purpose "
                    "and any diagnosis criterion placed there instead will silently produce "
                    "an UNFILTERED cohort for that group."
                ),
                "items": {
                    "type": "object",
                    "required": ["label", "diagnosis_basis"],
                    "properties": {
                        "label": {"type": "string"},
                        "diagnosis_basis": {
                            "type": "string",
                            "enum": ["clinical_status", "etiologic_ad"],
                            "description": (
                                "REQUIRED for every group. 'clinical_status': the group is "
                                "defined by baseline cognitive/syndromic diagnosis (CN, MCI, "
                                "dementia, impaired-not-mci) — also set clinical_status to the "
                                "specific value. 'etiologic_ad': the group is defined by AD "
                                "etiologic attribution (DXAD/DXMDUE/DXDDUE for ADNI, NACCALZD "
                                "for NACC) — clinical_status is not used for this basis."
                            ),
                        },
                        "clinical_status": {
                            "type": ["string", "null"],
                            "enum": ["normal", "mci", "dementia", "impaired-not-mci", None],
                            "description": (
                                "REQUIRED when diagnosis_basis='clinical_status'. Must be "
                                "exactly one of these four values — never a raw column name "
                                "(e.g. 'DXMCI', 'DXAD') and never an operator expression "
                                "(e.g. {'neq': 1}). null when diagnosis_basis='etiologic_ad'."
                            ),
                        },
                        "cohort_scope": {
                            "type": "string",
                            "enum": [
                                "baseline", "any_visit", "first_event",
                                "followup_window", "pre_index", "unclear",
                            ],
                        },
                        "baseline_filters": {
                            "type": "object",
                            "description": (
                                "For NON-diagnosis criteria ONLY (e.g. a visit-window or "
                                "modality requirement). NEVER put a diagnosis/clinical-status/"
                                "etiology condition here — use diagnosis_basis + clinical_status "
                                "instead; this dict's contents are not read for that purpose."
                            ),
                        },
                        "event_filters": {"type": "object"},
                        "event_anchor": {
                            "type": ["string", "null"],
                            "enum": [
                                "baseline", "first_matching_event_visit",
                                "first_conversion_visit", "last_visit", None,
                            ],
                        },
                        "etiology_definition": {
                            "type": "string",
                            "enum": [
                                "etiologic_ad", "all_cause_dementia",
                                "any_diagnosis", "unspecified",
                            ],
                        },
                        "diagnosis_variable": {"type": "string"},
                        "required_modalities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Set ONLY when this group's question asks how many "
                                "participants have MULTIPLE data types available (e.g. "
                                "'have both amyloid PET and CSF biomarker data'). Valid "
                                "values: cognition, mri, pet, fluid_biomarker, vascular, "
                                "neuropath, demographic. Leave empty for questions that "
                                "are not about data/modality availability."
                            ),
                        },
                    },
                },
            },
            "variable_selections": {
                "type": "array",
                "description": (
                    "One VariableSelection per matching catalog variable_id for "
                    "multi-variant measures (e.g. several CSF p-tau181 assay variants). "
                    "Leave empty for single-variable questions — the top-level "
                    "statistic_variable is used instead."
                ),
                "items": {
                    "type": "object",
                    "required": ["variable_id"],
                    "properties": {
                        "variable_id": {
                            "type": "string",
                            "description": (
                                "Copied verbatim from the catalog context's variable_id "
                                "column. Never construct or shorten one."
                            ),
                        },
                        "specimen": {"type": ["string", "null"]},
                        "analyte": {"type": ["string", "null"]},
                        "variant_label": {"type": "string"},
                    },
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_semantic_plan_from_data(
    data: dict[str, Any], nl: str
) -> SemanticAnalysisPlan | None:
    """Normalize and validate raw LLM output into a SemanticAnalysisPlan."""
    try:
        # Normalize missing optional fields
        data.setdefault("baseline_filters", {})
        data.setdefault("event_filters", {})
        data.setdefault("event_anchor", None)
        data.setdefault("time_ordering_variable", "NACCVNUM")
        data.setdefault("outcome_definition", "")
        data.setdefault("diagnosis_variable", "")
        data.setdefault("statistic_variable", "")
        data.setdefault("statistic_type", None)
        data.setdefault("denominator_definition", "")
        data.setdefault("allowed_backend_tools", [])
        data.setdefault("known_invalid_interpretations", [])
        data.setdefault("caveats", [])
        data.setdefault("abstain_if_missing", [])
        data.setdefault("id_space", "A")
        data.setdefault("specimen", None)
        data.setdefault("analyte", None)
        data.setdefault("groups", [])
        data.setdefault("variable_selections", [])
        data.setdefault("required_modalities", [])

        return SemanticAnalysisPlan(
            original_question=nl,
            **{k: v for k, v in data.items() if k != "original_question"},
        )
    except Exception:  # noqa: BLE001
        return None


def _plan_semantic_json_fallback(
    nl: str,
    client: Any,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
) -> SemanticAnalysisPlan | None:
    """JSON-prompt fallback when tool-use fails.  Returns None on any failure."""
    catalog_context = _build_catalog_context(nl, catalog or [])

    prompt = (
        "You are resolving the semantic commitments of an Alzheimer's dataset research question.\n\n"
        f"QUESTION: {nl}\n\n"
        f"CATALOG CONTEXT:\n{catalog_context}\n\n"
        "TASK: Produce a JSON object with the semantic commitments below.  "
        "Return ONLY valid JSON — no markdown, no prose.\n\n"
        "REQUIRED JSON FORMAT:\n"
        "{\n"
        '  "population_unit": "participant",\n'
        '  "requires_unique_participants": true,\n'
        '  "cohort_scope": "first_event",\n'
        '  "baseline_filters": {},\n'
        '  "event_filters": {"NACCALZD": 1},\n'
        '  "event_anchor": "first_matching_event_visit",\n'
        '  "time_ordering_variable": "NACCVNUM",\n'
        '  "outcome_definition": "NACCALZD=1",\n'
        '  "etiology_definition": "etiologic_ad",\n'
        '  "diagnosis_variable": "NACCALZD",\n'
        '  "statistic_variable": "NACCAGE",\n'
        '  "statistic_type": "mean",\n'
        '  "denominator_definition": "unique NACC participants with NACCALZD=1 on any visit",\n'
        '  "allowed_backend_tools": ["variable_stat"],\n'
        '  "known_invalid_interpretations": ["NOT baseline-only filter", "NOT visit-row count"],\n'
        '  "caveats": [],\n'
        '  "abstain_if_missing": [],\n'
        '  "id_space": "A",\n'
        '  "specimen": null,\n'
        '  "analyte": null,\n'
        '  "groups": [],\n'
        '  "variable_selections": [],\n'
        '  "required_modalities": []\n'
        "}\n\n"
        "KEY RULES:\n"
        "- etiology_definition=etiologic_ad requires NACCALZD in diagnosis_variable and event_filters.\n"
        "- cohort_scope=first_event requires event_filters non-empty and event_anchor set.\n"
        "- requires_unique_participants=true when population_unit=participant.\n"
        "- denominator_definition must be non-empty for any mean/median/count/proportion.\n"
        "- statistic_variable / variable_selections[*].variable_id must be copied VERBATIM "
        "from the catalog context's variable_id column — never construct or shorten one. "
        "For Space B (ADNI), the bare field name is ambiguous across tables; copy the full "
        "'ADNI.<TABLE>.<FIELD>' string exactly as shown.\n"
        "- For multi-group comparisons (e.g. CN vs MCI vs AD), populate 'groups' with one "
        "object per named group instead of the top-level filters; if a label like 'AD' is "
        "ambiguous between syndromic dementia and etiologic AD, emit two groups rather than "
        "picking one. EVERY group object MUST include \"diagnosis_basis\": \"clinical_status\" "
        "or \"diagnosis_basis\": \"etiologic_ad\". For clinical_status, ALSO include "
        "\"clinical_status\": one of \"normal\"|\"mci\"|\"dementia\"|\"impaired-not-mci\" "
        "— e.g. {\"label\": \"CN\", \"diagnosis_basis\": \"clinical_status\", "
        "\"clinical_status\": \"normal\"}. NEVER express a group's diagnosis criterion as a "
        "raw column name or operator object inside baseline_filters (e.g. {\"DXMCI\": 1} or "
        "{\"DXAD\": {\"neq\": 1}}) — baseline_filters is NOT read for diagnosis intent and "
        "doing so silently produces an UNFILTERED group.\n"
        "- For multi-variant measures (e.g. several CSF p-tau181 assay variants), populate "
        "'variable_selections' with one object per matching variable_id instead of the "
        "top-level statistic_variable; give each a distinct variant_label and never pool "
        "distinct assay variants into one figure.\n"
        "- required_modalities: set this (top-level, or per-group in groups[*]) when the "
        "question asks how many participants have MULTIPLE data types available (e.g. "
        "'have both MRI and CSF biomarker data'). Valid values: cognition, mri, pet, "
        "fluid_biomarker, vascular, neuropath, demographic. This compiles to an actual "
        "intersection-count operation — do not leave it empty and let such a question "
        "fall through to a plain participant count that ignores the modality requirement.\n"
        "- abstain_if_missing: the compiled plan can ONLY produce a participant count per "
        "group and/or one single-timepoint statistic per group anchored on ONE visit. There "
        "is NO operation for a trend, trajectory, slope, or 'how X changes over time/visits', "
        "and NO way to bound elapsed time between two visits (e.g. 'converted WITHIN 3 "
        "years') — event_filters/event_anchor can only express an unbounded 'ever reaches "
        "state X at some later visit'. If the question asks for either, resolve the "
        "unbounded/cross-sectional part normally but list the dropped part here verbatim "
        "(not just in 'caveats', which is never surfaced to the researcher) so the gap is "
        "disclosed rather than silently answering only part of the question.\n"
    )

    try:
        if isinstance(client, PlannerClient):
            raw = client.complete_text(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )
        else:
            response = client.messages.create(
                model=model or "claude-sonnet-4-6",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text if response.content else ""

        raw = raw.strip()
        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return _build_semantic_plan_from_data(data, nl)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_semantic_analysis(
    nl: str,
    client: Any,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
) -> SemanticAnalysisPlan | None:
    """Generate a SemanticAnalysisPlan from a natural-language question.

    Primary path: forced tool use (plan_semantic_analysis).
    Fallback: JSON-prompt completion when tool invocation fails.
    Returns None when both paths fail; caller should abstain.
    """
    catalog_context = _build_catalog_context(nl, catalog or [])

    user_message = (
        "Resolve the semantic commitments of this Alzheimer's dataset research question.\n\n"
        f"QUESTION: {nl}\n\n"
        "CATALOG CONTEXT (copy the variable_id column verbatim when referencing variables — "
        "for Space B/ADNI the field name alone is ambiguous across tables, so the full "
        "'ADNI.<TABLE>.<FIELD>' string is required; never construct or shorten one):\n"
        f"{catalog_context}\n\n"
        "Carefully decide:\n"
        "- population_unit: are we counting participants or visit rows?\n"
        "- cohort_scope: baseline / any_visit / first_event / followup_window / unclear?\n"
        "- etiology_definition: etiologic_ad (NACCALZD=1) vs all_cause_dementia (NACCUDSD=4)?\n"
        "- event_anchor: which visit row defines the statistic (if statistic_variable is set)?\n"
        "- denominator_definition: what is the exact denominator in plain language?\n"
        "- known_invalid_interpretations: explicitly rule out misreadings.\n"
        "- If the question compares several named groups (e.g. CN/MCI/AD), populate "
        "'groups' instead of the top-level filters — and if a group label is ambiguous "
        "between diagnostic bases (e.g. 'AD' could mean syndromic dementia or etiologic "
        "AD), emit both as separate groups rather than choosing one.\n"
        "- If the question's measure has multiple matching catalog variants (e.g. several "
        "assay platforms for the same analyte), populate 'variable_selections' with one "
        "entry per real variable_id instead of the top-level statistic_variable."
    )

    messages = [{"role": "user", "content": user_message}]

    # ---- Primary path: forced tool invocation ----
    plan_data: dict[str, Any] | None = None
    try:
        if isinstance(client, PlannerClient):
            plan_data = client.invoke_tool(messages, _SEMANTIC_PLAN_TOOL, max_tokens=1500)
        else:
            response = client.messages.create(
                model=model or "claude-sonnet-4-6",
                max_tokens=1500,
                tools=[_SEMANTIC_PLAN_TOOL],
                tool_choice={"type": "tool", "name": "plan_semantic_analysis"},
                messages=messages,
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "plan_semantic_analysis":
                    plan_data = dict(block.input)
                    break
    except Exception:  # noqa: BLE001
        plan_data = None

    if plan_data is not None:
        result = _build_semantic_plan_from_data(plan_data, nl)
        if result is not None:
            return result

    # ---- Fallback: JSON-prompt completion ----
    return _plan_semantic_json_fallback(nl, client, model=model, catalog=catalog)


def repair_semantic_plan(
    nl: str,
    prior: SemanticAnalysisPlan,
    validation: ValidationResult,
    client: Any,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
) -> SemanticAnalysisPlan | None:
    """Single repair pass: re-invoke LLM with validation errors to get a corrected plan.

    Returns None if the repair invocation itself fails.
    The caller re-validates and abstains if the repaired plan is still invalid.
    """
    errors_text = "\n".join(f"- {e}" for e in validation.hard_errors)
    prior_json = prior.model_dump_json(indent=2)

    user_message = (
        "Your previous semantic plan for this question was REJECTED by the validator.\n\n"
        f"QUESTION: {nl}\n\n"
        f"PRIOR PLAN (JSON):\n{prior_json}\n\n"
        f"HARD ERRORS (ALL must be fixed):\n{errors_text}\n\n"
        "Produce a corrected plan via the plan_semantic_analysis tool.  "
        "Address every error.  Do not introduce new ambiguity.  "
        "cohort_scope must NOT be 'unclear'.  "
        "If etiology_definition=etiologic_ad, diagnosis_variable and event_filters "
        "MUST include NACCALZD.  "
        "If cohort_scope=first_event, event_anchor and event_filters MUST be set.  "
        "If statistic_type is mean/median/count/proportion, denominator_definition MUST be non-empty."
    )

    messages = [{"role": "user", "content": user_message}]

    plan_data: dict[str, Any] | None = None
    try:
        if isinstance(client, PlannerClient):
            plan_data = client.invoke_tool(messages, _SEMANTIC_PLAN_TOOL, max_tokens=1500)
        else:
            response = client.messages.create(
                model=model or "claude-sonnet-4-6",
                max_tokens=1500,
                tools=[_SEMANTIC_PLAN_TOOL],
                tool_choice={"type": "tool", "name": "plan_semantic_analysis"},
                messages=messages,
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "plan_semantic_analysis":
                    plan_data = dict(block.input)
                    break
    except Exception:  # noqa: BLE001
        return None

    if plan_data is None:
        return None
    return _build_semantic_plan_from_data(plan_data, nl)


def _filter_dict_for_scope(group: "GroupDefinition") -> dict[str, Any]:
    """THE single canonical normalization step: GroupDefinition -> executable filter dict.

    This is the ONLY function in the codebase that maps a group's diagnosis-defining
    intent to a backend filter dict. compile_to_enriched(), PlanValidator's C14/C15,
    and the audit-trace logger all call THIS function (directly or by construction) so
    there is exactly one answer to "what filter does this group compile to" — never a
    second, drifted implementation under a different name.

    Diagnosis-defining intent priority:
      1. group.diagnosis_basis (Pydantic-validated enum — the LLM cannot misname this
         the way it can invent an arbitrary dict key). "clinical_status" requires
         group.clinical_status to be one of the four fixed enum values; "etiologic_ad"
         always produces {"etiology": "ad"} (baseline) or {"event_etiology": "ad"}
         (any-visit/first-event).
      2. Legacy fallback when diagnosis_basis is None (plans predating this field,
         single-cohort questions that never set it): the historical
         baseline_filters['diagnosis'] dict-read, and etiology_definition for
         etiologic-AD/all-cause-dementia. This path is why a group whose LLM-authored
         baseline_filters uses a DIFFERENT key (e.g. raw ADNI column names like
         {"DXMCI": 1}) or an unrecognized shape compiles to an EMPTY filter — that
         emptiness is exactly what C14/C15 (which call this same function, not the
         raw dict) now detect and reject before any operation executes.

    For any-visit / first-event etiologic AD: uses event_etiology key explicitly so
    count_eligible also applies any-visit semantics (not just the stat op). For
    baseline etiologic AD: uses etiology key (routes to baseline_etiology).
    """
    filter_dict: dict[str, Any] = {}

    if group.diagnosis_basis == "clinical_status":
        if group.clinical_status:
            filter_dict["diagnosis"] = group.clinical_status
    elif group.diagnosis_basis == "etiologic_ad":
        if group.cohort_scope == CohortScope.BASELINE:
            filter_dict["etiology"] = "ad"
        else:
            filter_dict["event_etiology"] = "ad"
    else:
        # Legacy path — no diagnosis_basis asserted.
        if group.baseline_filters.get("diagnosis"):
            filter_dict["diagnosis"] = group.baseline_filters["diagnosis"]

        if group.etiology_definition == EtiologyDefinition.ETIOLOGIC_AD:
            if group.cohort_scope == CohortScope.BASELINE:
                filter_dict["etiology"] = "ad"
            else:
                # any_visit, first_event, followup_window — explicit event_etiology key
                filter_dict["event_etiology"] = "ad"
        elif group.etiology_definition == EtiologyDefinition.ALL_CAUSE_DEMENTIA:
            filter_dict["diagnosis"] = "dementia"

    # Additional event_filters (e.g. literal NACCALZD key from LLM)
    # Exclude NACCALZD/etiology keys already handled above to avoid duplication
    for k, v in group.event_filters.items():
        if k.lower() not in ("naccalzd", "etiology", "event_etiology", "baseline_etiology"):
            filter_dict[k] = v

    return filter_dict


def _normalize_variable_id(var_id: str, id_space: str) -> str:
    """Prefix NACC. for id_space A if the id has no dataset prefix yet."""
    if id_space == "A" and not var_id.startswith("NACC.") and not var_id.startswith("ADNI."):
        return f"NACC.{var_id}"
    return var_id


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "x"


def compile_to_enriched(
    sem: SemanticAnalysisPlan,
    nl: str,
) -> EnrichedAnalysisPlan:
    """Deterministically compile a validated SemanticAnalysisPlan -> EnrichedAnalysisPlan.

    No LLM calls — this function performs zero interpretation of the question. Every
    semantic decision (which groups, which variable variants, which diagnostic basis)
    was already resolved and validated upstream (plan_semantic_analysis + validate_semantic).
    This function's only job is mechanical translation of those already-resolved slots
    into PlanOperations; the DeterministicExecutor that runs them is equally
    interpretation-free (see executor.py) — it only ever sees variable_id/filter dicts.

    Fan-out: when sem.groups is non-empty, one count_eligible + one stat op is emitted
    per (group x variable_selection) pair instead of a single cohort/variable — the cross
    product mechanically expresses "N groups compared on M variants" without any
    per-concept branching. Both lists default empty, which collapses this exactly to the
    original single-cohort/single-variable behavior (verified by TestCompileToEnriched).

    The executor's _build_filter_spec() processes the operation filters; we emit filter
    keys that route correctly through it.
    """
    operations: list[PlanOperation] = []

    # ---- Scopes to iterate: sem.groups, or one implicit scope from the top-level plan ----
    if sem.groups:
        scopes: list[GroupDefinition] = list(sem.groups)
    else:
        scopes = [
            GroupDefinition(
                label="",
                cohort_scope=sem.cohort_scope,
                baseline_filters=sem.baseline_filters,
                event_filters=sem.event_filters,
                event_anchor=sem.event_anchor,
                etiology_definition=sem.etiology_definition,
                diagnosis_variable=sem.diagnosis_variable,
                required_modalities=sem.required_modalities,
            )
        ]

    # ---- Variable selections to iterate: sem.variable_selections, or one implicit selection ----
    if sem.variable_selections:
        selections: list[VariableSelection] = list(sem.variable_selections)
    elif sem.statistic_variable:
        selections = [
            VariableSelection(
                variable_id=sem.statistic_variable,
                specimen=sem.specimen,
                analyte=sem.analyte,
                variant_label="",
            )
        ]
    else:
        selections = []

    # Records the ACTUAL compiled filter_dict per group/scope — the exact dict that
    # reaches count_eligible/variable_stat and, downstream, _build_filter_spec() /
    # the backend. Logged into the audit trace below so "what filter did this group
    # actually get" is directly visible in the synthesized answer and --show-debug
    # output, not just the LLM's pre-compile intent (which can silently diverge from
    # what was actually applied).
    group_filter_log: list[str] = []

    for group in scopes:
        filter_dict = _filter_dict_for_scope(group)
        group_filter_log.append(
            f"Group '{group.label or '(single cohort)'}': cohort_scope="
            f"{group.cohort_scope.value}, validated filter={filter_dict!r}, "
            f"etiology_definition={group.etiology_definition.value}, "
            f"event_anchor={group.event_anchor}."
        )
        backend_anchor = (
            _VISIT_ANCHOR_MAP.get(group.event_anchor or "") if group.event_anchor else None
        )
        group_suffix = f"_{_slug(group.label)}" if group.label else ""

        if sem.population_unit == PopulationUnit.PARTICIPANT:
            operations.append(
                PlanOperation(
                    op_type="count_eligible",
                    parameters={"filter": filter_dict},
                    output_name=f"eligible_n{group_suffix}",
                    description=(
                        f"Count unique participants meeting the "
                        f"{group.label or 'semantic'} cohort definition."
                    ),
                )
            )

        if group.required_modalities:
            operations.append(
                PlanOperation(
                    op_type="modality_overlap",
                    parameters={
                        "modalities": list(group.required_modalities),
                        "filter": filter_dict,
                    },
                    output_name=f"modality_overlap{group_suffix}",
                    description=(
                        f"Intersection counts for modalities "
                        f"{list(group.required_modalities)} in the "
                        f"{group.label or 'semantic'} cohort."
                    ),
                )
            )

        for sel in selections:
            var_id = _normalize_variable_id(sel.variable_id, sem.id_space)
            variant_suffix = f"_{_slug(sel.variant_label)}" if sel.variant_label else ""
            base_name = _slug(sel.variable_id)
            cohort_label = group.label or "semantic"

            if sem.statistic_type not in (None, "count", "proportion"):
                stat_params: dict[str, Any] = {
                    "variable_id": var_id,
                    "stat_type": sem.statistic_type,
                    "filter": filter_dict,
                }
                if backend_anchor:
                    stat_params["event_anchor"] = backend_anchor

                operations.append(
                    PlanOperation(
                        op_type="variable_stat",
                        parameters=stat_params,
                        output_name=f"{sem.statistic_type}_{base_name}{variant_suffix}{group_suffix}",
                        description=(
                            f"Compute {sem.statistic_type} of {sel.variable_id} "
                            f"anchored on {group.event_anchor or 'baseline'} "
                            f"for the {cohort_label} cohort."
                        ),
                    )
                )
            else:
                desc_params: dict[str, Any] = {
                    "variable_id": var_id,
                    "filter": filter_dict,
                }
                if backend_anchor:
                    desc_params["event_anchor"] = backend_anchor

                operations.append(
                    PlanOperation(
                        op_type="descriptive_stat",
                        parameters=desc_params,
                        output_name=f"desc_{base_name}{variant_suffix}{group_suffix}",
                        description=(
                            f"Full descriptive stats for {sel.variable_id} "
                            f"in the {cohort_label} cohort."
                        ),
                    )
                )

    # ---- Build baseline_definition (single-cohort summary; empty for multi-group plans,
    # where the per-group filters are already fully represented in each operation) ----
    baseline_def: dict[str, Any] = {}
    if not sem.groups:
        if sem.baseline_filters.get("diagnosis"):
            baseline_def["diagnosis"] = sem.baseline_filters["diagnosis"]
        if (
            sem.etiology_definition == EtiologyDefinition.ETIOLOGIC_AD
            and sem.cohort_scope == CohortScope.BASELINE
        ):
            baseline_def.setdefault("etiology", "ad")

    # ---- Researcher audit trace from semantic commitments ----
    # Group lines use group_filter_log (built above, inside the compile loop) so the
    # VALIDATED, actually-applied filter dict is what's reported — never the LLM's
    # raw pre-compile intent, which can silently diverge (see the CN/MCI collapse
    # regression: two groups both had empty baseline_filters, invisible in the old
    # audit-line format that only printed cohort_scope/etiology_definition/event_anchor).
    audit: list[str] = [
        f"population_unit={sem.population_unit.value}, requires_unique_participants={sem.requires_unique_participants}.",
        *group_filter_log,
    ]
    if not sem.groups:
        audit.append(
            f"etiology_definition={sem.etiology_definition.value}, "
            f"diagnosis_variable='{sem.diagnosis_variable}'."
        )
    if sem.variable_selections:
        for vs in sem.variable_selections:
            tags = "".join(
                f", {k}={v}" for k, v in (("specimen", vs.specimen), ("analyte", vs.analyte)) if v
            )
            audit.append(
                f"Variable selection '{vs.variant_label or vs.variable_id}': "
                f"variable_id={vs.variable_id}{tags}."
            )
    if sem.denominator_definition:
        audit.append(f"Denominator: {sem.denominator_definition}")
    if sem.known_invalid_interpretations:
        audit.append("Ruling out: " + "; ".join(sem.known_invalid_interpretations[:3]))

    # ---- Research intent summary ----
    if sem.groups:
        group_desc = f"across {len(sem.groups)} groups ({', '.join(g.label for g in sem.groups)})"
    else:
        group_desc = f"for {sem.cohort_scope.value} cohort ({sem.etiology_definition.value})"
    if sem.variable_selections:
        var_desc = f"{len(sem.variable_selections)} variable variant(s)"
    else:
        var_desc = sem.statistic_variable or "participants"

    return EnrichedAnalysisPlan(
        original_question=nl,
        research_intent_summary=f"{sem.statistic_type or 'count'} of {var_desc} {group_desc}",
        researcher_audit_trace=audit,
        operations=operations,
        requested_horizons=[],
        requires_backend_data=bool(operations),
        baseline_definition=baseline_def,
        outcome_definition=sem.outcome_definition,
        id_space=sem.id_space,
        semantic_plan=sem,
    )
