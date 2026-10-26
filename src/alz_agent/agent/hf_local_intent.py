"""hf-local intent parsing — strict, narrow compact schema for small local models.

Small open-source models (e.g. Qwen2.5-3B run via LocalHFPlannerClient) reliably
fail to fill the full ~30-field fill_intent / plan_analysis schemas used by the
DeepSeek and Anthropic paths (see docs/HF_LOCAL_TESTING_NOTE.md): outputs get
truncated, fields duplicate, and the model sometimes ignores the compact
contract entirely and free-associates full-schema-shaped keys it has seen
elsewhere (question_type, cohort_filter, NACCUDSD, ...).

This module supports four simple hf-local task shapes: a participant count
(optionally combined with a modality-overlap intersection), a modality-overlap
query, and a plain (non-event-anchored) aggregate statistic. Anything outside
this — multi-horizon follow-up sufficiency, event-anchored statistics such as
"age at first AD diagnosis", statistical-analysis / group-comparison questions
— is explicitly out of scope for hf-local and must resolve to "unsupported".
All dataset-specific mapping (e.g. "baseline_mci" -> NACCUDSD semantics) is
performed deterministically by compact_intent_to_structured_intent(), never
by the LLM.

Defense in depth, in three layers, all of which must pass before any tool
executes:
  1. validate_hf_local_compact_intent() — shape/enum validation. Rejects any
     compact object containing forbidden full-schema keys (contamination from
     the model's training distribution), any key outside the compact schema,
     or an unsupported enum value.
  2. _semantic_mismatch_reason() — cross-checks the validated compact object
     against the ORIGINAL natural-language question with deterministic
     keyword rules (e.g. "the question says 'dementia' but population was
     parsed as baseline_mci" -> reject). This is what stops a
     schema-VALID-but-semantically-WRONG compact object (the model picking
     the one population it has seen most often, regardless of what the
     question actually asked) from silently reaching tool execution — the
     original failure mode this module was rewritten to close.
  3. compact_intent_to_structured_intent() — raises immediately (no retry) if
     the model's own "intent" field is "unsupported": an honest "I can't
     answer this" is accepted at face value, not retried.

Public:
  HFLocalPlannerAbstention        — raised when compact JSON is still invalid
                                     after one repair/retry pass, or when the
                                     model (or a semantic guard) classifies the
                                     question as unsupported. No tools are
                                     executed in either case.
  HFLocalCompactValidationError   — raised by validate_hf_local_compact_intent
                                     for any invalid compact object, and by the
                                     semantic-mismatch guard. A ValueError
                                     subtype so it composes with the generic
                                     repair/retry catch in _parse_hf_local_compact.
  CompactIntent                   — validated compact schema instance.
  build_hf_local_compact_prompt(query)      -> str
  parse_hf_local_compact_json(raw)          -> Any (JSON-decoded; may not be a dict)
  validate_hf_local_compact_intent(obj)     -> CompactIntent
  compact_intent_to_structured_intent(ci)   -> StructuredIntent
  _parse_hf_local_compact(nl, client, model, catalog, debug_sink) -> StructuredIntent
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from alz_agent.agent.intent_schema import QuestionType, StructuredIntent
from alz_agent.schemas import IdSpace

logger = logging.getLogger("alz_agent.agent.hf_local_intent")

_ROUTE_NAME = "hf_local_compact_simple_tasks_v2"
_COMPACT_MAX_NEW_TOKENS = 220

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HFLocalCompactValidationError(ValueError):
    """Raised for any invalid compact object or semantic mismatch.

    A ValueError subtype so _parse_hf_local_compact's generic
    (json.JSONDecodeError, ValueError, ValidationError) catch composes and
    triggers the one-shot repair/retry pass.
    """


class HFLocalPlannerAbstention(RuntimeError):
    """Raised when hf-local cannot produce a trustworthy StructuredIntent.

    Raised when:
      - compact JSON is still invalid after one repair/retry pass, or
      - the model classified the question as intent="unsupported" (accepted
        at face value — no retry burned on an honest "I can't do this"), or
      - a semantic-mismatch guard rejected the parse on both attempts.

    Callers must not execute tools when this is raised — no StructuredIntent
    was produced, so there is nothing grounded to dispatch.
    """


# ---------------------------------------------------------------------------
# Compact schema — intentionally narrow. Supports exactly four task shapes:
# count, modality_overlap, aggregate_stat (non-event-anchored only), and the
# explicit unsupported escape hatch. Do not broaden without a corresponding
# backend-execution need; it is better to abstain than to infer an
# unsupported structure.
# ---------------------------------------------------------------------------

_SUPPORTED_INTENTS: frozenset[str] = frozenset({
    "count", "modality_overlap", "aggregate_stat", "unsupported",
})
_SUPPORTED_POPULATIONS: frozenset[str] = frozenset({
    "baseline_mci", "baseline_normal", "baseline_dementia",
    "baseline_impaired_not_mci", "unspecified",
})
_SUPPORTED_ETIOLOGIES: frozenset[str] = frozenset({"none", "ad"})
_SUPPORTED_MODALITIES: frozenset[str] = frozenset({
    "cognition", "any_imaging", "mri", "pet", "fluid_biomarker",
    "vascular", "neuropath", "demographic",
})
_SUPPORTED_STAT_VARIABLES: frozenset[str] = frozenset({
    "age", "education", "mmse", "cdr", "none",
})
_SUPPORTED_STAT_TYPES: frozenset[str] = frozenset({
    "mean", "median", "std", "min", "max",
})
_SUPPORTED_DATASETS: frozenset[str] = frozenset({"nacc", "adni", "unspecified"})

_ALLOWED_COMPACT_KEYS: frozenset[str] = frozenset({
    "intent", "population", "etiology", "modalities",
    "stat_variable", "stat_type", "dataset",
})

# Keys that indicate the model emitted (or drifted toward) a full-StructuredIntent
# / backend-tool-shaped object instead of the compact schema. Presence of any of
# these is treated as contamination, not as "extra fields to ignore" — reject.
_FORBIDDEN_COMPACT_KEYS: frozenset[str] = frozenset(k.lower() for k in (
    "question_type", "target_population", "target_etiology", "candidate_label",
    "horizon_years", "stat_type_raw", "analysis_type", "cohort_label",
    "cohort_column", "cohort_filter", "group_variable_label", "group_column",
    "outcome_variable_id", "outcome_column", "variable_id", "variable_column",
    "tool", "operation", "event_anchor", "id_space",
    # dataset-specific contamination guards (never legal in compact mode)
    "naccudsd", "naccalzd",
))

# The only dataset-specific mapping in this module. The LLM never sees this —
# it only ever emits these enumerated tokens.
_POPULATION_TO_TARGET_POPULATION: dict[str, list[str]] = {
    "baseline_mci": ["mci"],
    "baseline_normal": ["normal"],
    "baseline_dementia": ["dementia"],
    "baseline_impaired_not_mci": ["impaired-not-mci"],
    "unspecified": [],
}
_POPULATION_LABEL: dict[str, str] = {
    "baseline_mci": "baseline MCI",
    "baseline_normal": "baseline cognitively normal",
    "baseline_dementia": "baseline all-cause dementia",
    "baseline_impaired_not_mci": "baseline impaired-not-MCI",
    "unspecified": "all",
}
_MODALITY_ALIASES: dict[str, list[str]] = {
    "any_imaging": ["mri", "pet"],
}
_DATASET_TO_ID_SPACE: dict[str, IdSpace] = {
    "nacc": IdSpace.A,
    "adni": IdSpace.B,
    "unspecified": IdSpace.A,
}


@dataclass(frozen=True)
class CompactIntent:
    """Validated hf-local compact intent — the only shape the LLM may produce."""

    intent: str
    population: str
    etiology: str
    modalities: tuple[str, ...] = field(default_factory=tuple)
    stat_variable: str = "none"
    stat_type: str = "mean"
    dataset: str = "unspecified"


# ---------------------------------------------------------------------------
# Prompt construction
#
# Must never mention: StructuredIntent, question_type, target_population,
# target_etiology, cohort_filter, cohort_column, group_column,
# outcome_variable_id, NACCUDSD, or any other dataset column name / numeric
# code / backend tool parameter. Controlled enumerations only.
# ---------------------------------------------------------------------------

_COMPACT_SCHEMA_INSTRUCTIONS = (
    "Return ONLY one JSON object with EXACTLY these seven fields — no other "
    "fields, no markdown fences, no explanation, no duplicate keys:\n"
    "{\n"
    '  "intent": "count" | "modality_overlap" | "aggregate_stat" | "unsupported",\n'
    '  "population": "baseline_mci" | "baseline_normal" | "baseline_dementia" '
    '| "baseline_impaired_not_mci" | "unspecified",\n'
    '  "etiology": "none" | "ad",\n'
    '  "modalities": [],\n'
    '  "stat_variable": "age" | "education" | "mmse" | "cdr" | "none",\n'
    '  "stat_type": "mean" | "median" | "std" | "min" | "max",\n'
    '  "dataset": "nacc" | "adni" | "unspecified"\n'
    "}\n\n"
    'intent="count" for a plain participant count. intent="modality_overlap" '
    "when the question asks how many participants have two or more kinds of "
    "data at once. intent=\"aggregate_stat\" for a plain average/median/etc. "
    "of one variable (never for a value measured 'at first diagnosis' or "
    "'at onset' — use \"unsupported\" for those). Use intent=\"unsupported\" "
    "whenever the question does not clearly match one of the other three, "
    "or requires timing a value to a specific visit/event.\n"
    "population is the cognitive-status group being counted; use "
    '"unspecified" when the question does not restrict to one group.\n'
    'etiology="ad" only when the question is specifically about an '
    "Alzheimer's / etiologic-AD diagnosis, not just all-cause dementia; "
    'otherwise "none". If the question explicitly says not to infer or '
    'assume a cause, use "none".\n'
    "modalities is a list of zero or more of: cognition, any_imaging, mri, "
    "pet, fluid_biomarker, vascular, neuropath, demographic. Leave it [] "
    "unless the question asks about combined data types.\n"
    'stat_variable and stat_type only matter when intent="aggregate_stat"; '
    'otherwise use "none" and "mean".\n'
    'dataset="adni" only when the question explicitly names ADNI; '
    'otherwise "nacc".'
)


def build_hf_local_compact_prompt(query: str) -> str:
    """Build the hf-local compact-schema prompt. Never mentions full-schema
    field names, dataset column names, or dataset-specific numeric codes."""
    return (
        "You are extracting a small structured summary from a research question.\n\n"
        f"{_COMPACT_SCHEMA_INSTRUCTIONS}\n\n"
        f"QUESTION: {query}\n\n"
        'EXAMPLE for "Count participants with baseline MCI. Do not infer AD etiology.":\n'
        '{"intent": "count", "population": "baseline_mci", "etiology": "none", '
        '"modalities": [], "stat_variable": "none", "stat_type": "mean", '
        '"dataset": "nacc"}\n\n'
        'EXAMPLE for "How many participants have both cognition and any imaging data?":\n'
        '{"intent": "modality_overlap", "population": "unspecified", "etiology": "none", '
        '"modalities": ["cognition", "any_imaging"], "stat_variable": "none", '
        '"stat_type": "mean", "dataset": "nacc"}\n\n'
        "JSON:"
    )


def _build_compact_repair_prompt(query: str, broken_raw: str, error: Exception) -> str:
    return (
        "Your previous response was not a valid compact JSON object and was rejected.\n"
        f"REJECTION REASON: {error}\n"
        f"YOUR PREVIOUS OUTPUT:\n{broken_raw[:500]}\n\n"
        "Return ONE corrected, complete JSON object only — no markdown, no duplicate "
        "keys, no extra fields, no text before or after the JSON. If you cannot "
        'confidently fix this, set "intent": "unsupported".\n\n'
        f"{_COMPACT_SCHEMA_INSTRUCTIONS}\n\n"
        f"QUESTION: {query}\n\n"
        "JSON:"
    )


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    inner = lines[1:] if lines[0].startswith("```") else lines
    return "\n".join(inner[:-1] if inner and inner[-1].strip() == "```" else inner)


# ---------------------------------------------------------------------------
# Semantic mismatch guard — cross-checks the validated compact object against
# the ORIGINAL question text. This is defense against a schema-valid but
# semantically-wrong parse (e.g. the model always emitting the population it
# has seen most in training, regardless of what the question asked).
# ---------------------------------------------------------------------------

_AD_MENTION_RE = re.compile(r"\betiologic\b|\balzheimer|\bad\b", re.IGNORECASE)
_ETIOLOGY_NEGATION_RE = re.compile(
    r"do\s+not\s+infer\s+(?:ad|alzheimer)"
    r"|without\s+inferring\s+(?:ad|alzheimer)"
    r"|do\s+not\s+assume\s+(?:ad|alzheimer)"
    r"|regardless\s+of\s+(?:ad|alzheimer)(?:'s)?\s+etiology"
    r"|regardless\s+of\s+etiology",
    re.IGNORECASE,
)
_DEMENTIA_RE = re.compile(r"\bdementia\b", re.IGNORECASE)
_STAT_RE = re.compile(r"\baverage\b|\bmean\b|\bmedian\b|\bstandard deviation\b", re.IGNORECASE)
_MODALITY_KEYWORD_RE = re.compile(
    r"\bcognition\b|\bcognitive\b|\bimaging\b|\bmri\b|\bpet\b|\bbiomarker\b",
    re.IGNORECASE,
)
_ADNI_RE = re.compile(r"\badni\b", re.IGNORECASE)
_EVENT_ANCHOR_RE = re.compile(
    r"\bfirst diagno|\bwhen.*diagno|\bdiagno.*first|\bat.*diagno|\bonset\b",
    re.IGNORECASE,
)
# "any imaging" / "imaging data" phrasing means MRI OR PET (a union), not an
# AND/intersection of every imaging modality. Only the explicit AND phrases
# below license collapsing it into a plain ["mri", "pet"] intersection list.
_ANY_IMAGING_PHRASE_RE = re.compile(
    r"\bany imaging\b|\bany neuroimaging\b|\bany mri or pet\b|\bimaging data\b",
    re.IGNORECASE,
)
_EXPLICIT_ALL_MODALITY_RE = re.compile(
    r"\bmri and pet\b|\bboth mri and pet\b|\ball imaging modalities\b|\ball three\b",
    re.IGNORECASE,
)
# Explicit pairwise-OR language that already names the exact modalities being
# OR'd (e.g. "either MRI or PET", "MRI or PET", "MRI/PET", "any MRI or PET").
# Unlike the vaguer "any imaging"/"imaging data" phrasing above — which
# requires the model to emit the "any_imaging" alias token because the NL
# itself never says which modalities are meant — this language is
# self-describing, so a plain ["mri", "pet"] parse can be deterministically
# normalized into an OR group without demanding that token from the model.
_EXPLICIT_OR_MRI_PET_RE = re.compile(
    r"\beither\s+mri\s+or\s+pet\b|\beither\s+pet\s+or\s+mri\b"
    r"|\bmri\s+or\s+pet\b|\bpet\s+or\s+mri\b"
    r"|\bmri\s*/\s*pet\b|\bpet\s*/\s*mri\b",
    re.IGNORECASE,
)


def _explicit_or_pair(nl: str) -> list[str] | None:
    """Return ["mri", "pet"] if `nl` uses unambiguous pairwise-OR imaging
    language, else None. See _EXPLICIT_OR_MRI_PET_RE for the covered forms."""
    if _EXPLICIT_OR_MRI_PET_RE.search(nl):
        return ["mri", "pet"]
    return None


def _semantic_mismatch_reason(nl: str, compact: CompactIntent) -> str | None:
    """Return a human-readable mismatch reason, or None if the compact object
    is a plausible parse of nl. Deterministic keyword rules only — never an
    LLM call. Any single rule failing is enough to reject the whole parse."""
    ad_mentioned = bool(_AD_MENTION_RE.search(nl))
    etiology_negated = bool(_ETIOLOGY_NEGATION_RE.search(nl))

    if ad_mentioned and not etiology_negated and compact.etiology != "ad":
        return (
            "question mentions AD/Alzheimer/etiologic without an explicit "
            "etiology-negation phrase, but parsed etiology='none'"
        )

    if _DEMENTIA_RE.search(nl) and compact.population == "baseline_mci":
        return (
            "question mentions 'dementia' but parsed population='baseline_mci' "
            "(all-cause dementia is NACCUDSD=4, not MCI=3 — never conflate them)"
        )

    if _STAT_RE.search(nl) and compact.intent != "aggregate_stat":
        return (
            f"question asks for an average/mean/median but parsed intent="
            f"{compact.intent!r}, not 'aggregate_stat'"
        )

    if _MODALITY_KEYWORD_RE.search(nl) and not compact.modalities:
        return (
            "question mentions a modality (cognition/imaging/MRI/PET/biomarker) "
            "but no modalities were parsed"
        )

    if _ADNI_RE.search(nl) and compact.dataset != "adni":
        return (
            "question mentions ADNI but parsed dataset != 'adni' "
            "(id_space would incorrectly remain Space A)"
        )

    if compact.intent == "aggregate_stat" and _EVENT_ANCHOR_RE.search(nl):
        return (
            "question requires an event-anchored statistic (e.g. 'age at first "
            "diagnosis'), which the hf-local compact schema does not support — "
            "the correct answer requires event_etiology, not a plain baseline stat"
        )

    any_imaging_phrased = bool(_ANY_IMAGING_PHRASE_RE.search(nl))
    explicit_all_modality = bool(_EXPLICIT_ALL_MODALITY_RE.search(nl))
    explicit_or_pair = _explicit_or_pair(nl)
    has_mri = "mri" in compact.modalities
    has_pet = "pet" in compact.modalities
    has_any_imaging_token = "any_imaging" in compact.modalities

    if (
        any_imaging_phrased
        and not explicit_all_modality
        and not explicit_or_pair
        and has_mri
        and has_pet
        and not has_any_imaging_token
    ):
        return (
            "question uses 'any imaging'/'imaging data' (an OR of MRI/PET) "
            "without explicit all-of language, but parsed modalities list "
            "['mri', 'pet'] directly instead of the 'any_imaging' token — "
            "this would collapse an OR union into a plain AND/intersection"
        )

    if explicit_all_modality and has_any_imaging_token:
        return (
            "question explicitly asks for MRI and PET together (AND), but "
            "parsed modalities used the OR-token 'any_imaging' instead of "
            "listing 'mri' and 'pet' explicitly"
        )

    return None


# ---------------------------------------------------------------------------
# Parse -> validate -> adapt
# ---------------------------------------------------------------------------


def parse_hf_local_compact_json(raw: str) -> Any:
    """Parse raw hf-local text as JSON.

    Purely syntactic — raises json.JSONDecodeError on malformed input. Shape
    validation (must be an object containing only the compact schema's keys)
    is validate_hf_local_compact_intent's job, not this function's.
    """
    return json.loads(_strip_code_fences(raw))


def validate_hf_local_compact_intent(obj: Any) -> CompactIntent:
    """Validate a JSON-decoded compact object against the strict compact schema.

    Rejects (raises HFLocalCompactValidationError) when:
    - obj is not a JSON object
    - obj contains any forbidden full-schema / dataset-specific key
    - obj contains any key outside the compact schema
    - any field holds a value outside its controlled enumeration
    """
    if not isinstance(obj, dict):
        raise HFLocalCompactValidationError(
            f"hf-local compact output must be a JSON object, got {type(obj).__name__}"
        )

    present_forbidden = sorted(
        k for k in obj if str(k).strip().lower() in _FORBIDDEN_COMPACT_KEYS
    )
    if present_forbidden:
        raise HFLocalCompactValidationError(
            "hf-local compact mode rejected forbidden full-schema/dataset key(s) "
            f"{present_forbidden}: the model must only emit "
            f"{sorted(_ALLOWED_COMPACT_KEYS)}. This looks like full-StructuredIntent "
            "or backend-tool-parameter contamination, not the compact schema."
        )

    unknown = sorted(k for k in obj if k not in _ALLOWED_COMPACT_KEYS)
    if unknown:
        raise HFLocalCompactValidationError(
            f"hf-local compact mode rejected unrecognized key(s) {unknown}; "
            f"only {sorted(_ALLOWED_COMPACT_KEYS)} are permitted."
        )

    intent_val = obj.get("intent")
    if intent_val not in _SUPPORTED_INTENTS:
        raise HFLocalCompactValidationError(
            f"hf-local compact mode: unsupported intent {intent_val!r}; "
            f"only {sorted(_SUPPORTED_INTENTS)} is currently supported."
        )

    population_val = obj.get("population", "unspecified")
    if population_val not in _SUPPORTED_POPULATIONS:
        raise HFLocalCompactValidationError(
            f"hf-local compact mode: unsupported population {population_val!r}; "
            f"only {sorted(_SUPPORTED_POPULATIONS)} is currently supported."
        )

    etiology_val = obj.get("etiology", "none")
    if etiology_val not in _SUPPORTED_ETIOLOGIES:
        raise HFLocalCompactValidationError(
            f"hf-local compact mode: unsupported etiology {etiology_val!r}; "
            f"only {sorted(_SUPPORTED_ETIOLOGIES)} is currently supported."
        )

    modalities_val = obj.get("modalities", [])
    if not isinstance(modalities_val, list):
        raise HFLocalCompactValidationError(
            f"hf-local compact mode: modalities must be a list, got "
            f"{type(modalities_val).__name__}"
        )
    invalid_modalities = sorted(m for m in modalities_val if m not in _SUPPORTED_MODALITIES)
    if invalid_modalities:
        raise HFLocalCompactValidationError(
            f"hf-local compact mode: unsupported modality value(s) {invalid_modalities}; "
            f"only {sorted(_SUPPORTED_MODALITIES)} is currently supported."
        )

    stat_variable_val = obj.get("stat_variable", "none")
    if stat_variable_val not in _SUPPORTED_STAT_VARIABLES:
        raise HFLocalCompactValidationError(
            f"hf-local compact mode: unsupported stat_variable {stat_variable_val!r}; "
            f"only {sorted(_SUPPORTED_STAT_VARIABLES)} is currently supported."
        )

    stat_type_val = obj.get("stat_type", "mean")
    if stat_type_val not in _SUPPORTED_STAT_TYPES:
        raise HFLocalCompactValidationError(
            f"hf-local compact mode: unsupported stat_type {stat_type_val!r}; "
            f"only {sorted(_SUPPORTED_STAT_TYPES)} is currently supported."
        )

    dataset_val = obj.get("dataset", "unspecified")
    if dataset_val not in _SUPPORTED_DATASETS:
        raise HFLocalCompactValidationError(
            f"hf-local compact mode: unsupported dataset {dataset_val!r}; "
            f"only {sorted(_SUPPORTED_DATASETS)} is currently supported."
        )

    # intent-specific structural requirements — malformed combinations are
    # rejected here rather than silently coerced or executed.
    if intent_val == "modality_overlap" and not modalities_val:
        raise HFLocalCompactValidationError(
            "hf-local compact mode: intent='modality_overlap' requires a "
            "non-empty modalities list."
        )
    if intent_val == "aggregate_stat" and stat_variable_val == "none":
        raise HFLocalCompactValidationError(
            "hf-local compact mode: intent='aggregate_stat' requires a "
            "stat_variable other than 'none'."
        )

    return CompactIntent(
        intent=intent_val,
        population=population_val,
        etiology=etiology_val,
        modalities=tuple(modalities_val),
        stat_variable=stat_variable_val,
        stat_type=stat_type_val,
        dataset=dataset_val,
    )


def _expand_modalities_preserving_or(
    modalities: tuple[str, ...],
) -> tuple[list[str], list[str] | None]:
    """Expand alias tokens (currently only "any_imaging") into concrete
    Modality names for the deterministic tool call, while separately
    tracking which concrete names came from an OR-alias so the caller can
    preserve "any imaging means MRI or PET" instead of silently collapsing
    it into a plain ["mri", "pet"] AND/intersection.

    Returns (flat_modalities, any_of_modalities). any_of_modalities is None
    unless an OR-alias was present in `modalities`.
    """
    flat: list[str] = []
    any_of: list[str] = []
    for m in modalities:
        if m in _MODALITY_ALIASES:
            for canonical in _MODALITY_ALIASES[m]:
                if canonical not in flat:
                    flat.append(canonical)
                if canonical not in any_of:
                    any_of.append(canonical)
        elif m not in flat:
            flat.append(m)
    return flat, (any_of if any_of else None)


def _modality_combo_label(modalities: list[str], any_of_modalities: list[str] | None) -> str:
    """Build a human-readable modality phrase that preserves OR semantics
    (e.g. "cognition + any imaging (mri or pet)") instead of implying AND
    across every listed modality when an alias expansion was OR-based."""
    if not modalities:
        return ""
    if not any_of_modalities:
        return " + ".join(modalities)
    required = [m for m in modalities if m not in any_of_modalities]
    or_part = " or ".join(any_of_modalities)
    parts = list(required) + [f"any imaging ({or_part})"]
    return " + ".join(parts)


def compact_intent_to_structured_intent(
    compact: CompactIntent, nl: str | None = None
) -> StructuredIntent:
    """Deterministically map a validated CompactIntent to StructuredIntent.

    This is the ONLY place population/modality normalization / dataset-specific
    mapping happens. The LLM never sees or assigns cohort_filter, NACCUDSD, or
    any other backend-specific structure — it only ever emits the enumerated
    tokens validated by validate_hf_local_compact_intent(), which are mapped
    here.

    `nl` (the original question text) is optional and only used to recover
    OR semantics the model failed to encode via the "any_imaging" alias
    token: when the question uses explicit pairwise-OR language (see
    _explicit_or_pair) and the model emitted a plain ["mri", "pet"] list
    instead, this deterministically re-derives any_of_modalities from the
    question text rather than requiring a retry. Callers that don't have
    `nl` handy (e.g. direct unit tests) get the prior alias-only behavior.

    Raises HFLocalPlannerAbstention immediately (no retry) when the model's
    own "intent" field is "unsupported": that is an honest, valid answer, not
    a malformed one, so it does not go through the repair/retry loop.
    """
    if compact.intent == "unsupported":
        raise HFLocalPlannerAbstention(
            "hf-local classified this question as unsupported by the simple "
            "compact-schema task set (count / modality_overlap / "
            "aggregate_stat); abstaining without executing tools."
        )

    if compact.intent not in _SUPPORTED_INTENTS:
        raise HFLocalCompactValidationError(f"unsupported compact intent {compact.intent!r}")

    target_population = _POPULATION_TO_TARGET_POPULATION.get(compact.population)
    if target_population is None:
        raise HFLocalCompactValidationError(
            f"unsupported compact population {compact.population!r}"
        )
    target_population = list(target_population)

    target_etiology: list[str] = ["ad"] if compact.etiology == "ad" else []

    modalities, any_of_modalities = _expand_modalities_preserving_or(compact.modalities)

    if any_of_modalities is None and nl is not None:
        explicit_or_pair = _explicit_or_pair(nl)
        if explicit_or_pair and all(m in modalities for m in explicit_or_pair):
            any_of_modalities = explicit_or_pair

    id_space = _DATASET_TO_ID_SPACE.get(compact.dataset, IdSpace.A)

    pop_label = _POPULATION_LABEL.get(compact.population, compact.population)
    etiology_suffix = " with etiologic AD attribution" if target_etiology else ""

    if compact.intent in ("count", "modality_overlap"):
        combo_label = _modality_combo_label(modalities, any_of_modalities)
        modality_suffix = f" with {combo_label} data" if combo_label else ""
        candidate_label = f"count of {pop_label} participants{etiology_suffix}{modality_suffix}"
        return StructuredIntent(
            question_type=QuestionType.COHORT_SIZE,
            target_population=target_population,
            target_etiology=target_etiology,
            candidate_label=candidate_label,
            modalities=modalities,
            any_of_modalities=any_of_modalities,
            id_space=id_space,
            pre_index_only=True,
            # count / modality_overlap never compute a statistic — leave no
            # meaningless aggregate-stat leftovers (e.g. stat_type="mean") in
            # the adapted intent for a question that never asked for one.
            stat_variable=None,
            stat_type=None,
            parsing_mode="llm",
        )

    if compact.intent == "aggregate_stat":
        candidate_label = (
            f"{compact.stat_type} {compact.stat_variable} of {pop_label} "
            f"participants{etiology_suffix}"
        )
        return StructuredIntent(
            question_type=QuestionType.AGGREGATE_STAT,
            target_population=target_population,
            target_etiology=target_etiology,
            candidate_label=candidate_label,
            id_space=id_space,
            stat_variable=compact.stat_variable,
            stat_type=compact.stat_type,
            event_anchor=None,
            pre_index_only=True,
            parsing_mode="llm",
        )

    raise HFLocalCompactValidationError(f"unhandled compact intent {compact.intent!r}")


def _validate_and_adapt(raw: str, nl: str) -> StructuredIntent:
    """parse_hf_local_compact_json -> validate_hf_local_compact_intent ->
    semantic-mismatch guard -> compact_intent_to_structured_intent, in one
    call for the retry loop.

    Semantic-mismatch failures raise HFLocalCompactValidationError (a
    ValueError), so they compose with the same one-shot repair/retry pass as
    malformed JSON. An explicit intent="unsupported" raises
    HFLocalPlannerAbstention directly from compact_intent_to_structured_intent
    and is NOT retried.
    """
    data = parse_hf_local_compact_json(raw)
    logger.debug("hf-local compact parsed JSON: %r", data)
    compact = validate_hf_local_compact_intent(data)
    logger.info(
        "hf-local compact validation ok: intent=%s population=%s etiology=%s "
        "modalities=%s stat_variable=%s dataset=%s",
        compact.intent, compact.population, compact.etiology,
        compact.modalities, compact.stat_variable, compact.dataset,
    )

    # An honest "unsupported" is accepted at face value, before the semantic
    # guard runs — the guard only makes sense for outputs that would
    # otherwise execute a tool, and "unsupported" never does.
    if compact.intent == "unsupported":
        return compact_intent_to_structured_intent(compact)  # raises HFLocalPlannerAbstention

    mismatch = _semantic_mismatch_reason(nl, compact)
    if mismatch is not None:
        raise HFLocalCompactValidationError(
            f"hf-local compact mode: semantic mismatch — {mismatch}"
        )

    structured = compact_intent_to_structured_intent(compact, nl=nl)
    logger.info(
        "hf-local compact adapter output: question_type=%s target_population=%s "
        "target_etiology=%s modalities=%s",
        structured.question_type.value, structured.target_population,
        structured.target_etiology, structured.modalities,
    )
    return structured


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _parse_hf_local_compact(
    nl: str,
    client: Any,
    model: str | None = None,
    catalog: list[Any] | None = None,
    debug_sink: list[str] | None = None,
) -> StructuredIntent:
    """Parse a research question via the strict hf-local compact JSON schema.

    validate -> semantic-mismatch guard -> (on malformed JSON / forbidden
    keys / unsupported value / semantic mismatch) one repair/retry call ->
    validate again -> HFLocalPlannerAbstention if still invalid or still
    mismatched. No tool is ever executed on the abstain path — the caller
    never receives a StructuredIntent to dispatch. An explicit
    intent="unsupported" from the model abstains immediately without a retry.

    catalog is accepted only for call-signature parity with the other parsing
    paths (parse_intent() forwards it uniformly); the compact path never
    resolves catalog columns, so it is unused here by design.

    debug_sink, when provided, has the raw text of every model call appended
    to it in order (attempt 1, then the repair attempt if one was made) —
    regardless of whether parsing ultimately succeeds or abstains. Callers
    (e.g. the CLI's --show-raw-intent) use this to show exactly what the
    model produced without re-running inference.
    """
    del catalog  # unused by design — see docstring

    model_name = model or getattr(client, "_model_id", "unknown-model")
    logger.info(
        "hf-local compact route start: mode=hf-local model=%s route=%s query=%r",
        model_name, _ROUTE_NAME, nl,
    )

    prompt = build_hf_local_compact_prompt(nl)
    raw = client.complete_text(
        messages=[{"role": "user", "content": prompt}], max_tokens=_COMPACT_MAX_NEW_TOKENS
    )
    if debug_sink is not None:
        debug_sink.append(raw)
    logger.info("hf-local compact raw output (attempt 1): %r", raw)

    try:
        return _validate_and_adapt(raw, nl)
    except (json.JSONDecodeError, ValueError, ValidationError) as first_exc:
        logger.warning("hf-local compact validation failed (attempt 1): %s", first_exc)
        repair_prompt = _build_compact_repair_prompt(nl, raw, first_exc)
        try:
            repaired_raw = client.complete_text(
                messages=[{"role": "user", "content": repair_prompt}],
                max_tokens=_COMPACT_MAX_NEW_TOKENS,
            )
            if debug_sink is not None:
                debug_sink.append(repaired_raw)
            logger.info("hf-local compact raw output (attempt 2 / repair): %r", repaired_raw)
            return _validate_and_adapt(repaired_raw, nl)
        except (json.JSONDecodeError, ValueError, ValidationError) as second_exc:
            logger.error(
                "hf-local compact abstain — no tools executed. "
                "first_error=%s retry_error=%s",
                first_exc, second_exc,
            )
            raise HFLocalPlannerAbstention(
                "hf-local compact schema validation failed after one repair/retry "
                "pass; abstaining without executing tools.\n"
                f"first_error={first_exc!r}\nretry_error={second_exc!r}"
            ) from second_exc
