"""P5 — Intent parser: NL question -> StructuredIntent.

Public surface (thin facade — implementation lives in submodules):
  parse_intent(nl_question, client, model, catalog) -> StructuredIntent
  plan_analysis(nl, client, model, catalog)          -> AnalysisPlan

Two execution paths:
  LLM path  (client is provided): catalog-grounded variable resolution via
             structured tool use, followed by deterministic validation and
             leakage assessment.
  Heuristic (client is None):     regex fallback — used for tests/CI only.

Key invariant: the LLM selects variables; it never computes numbers.
Leakage assessment is computed deterministically from catalog tags, not by
the LLM.

Parse-flow diagram:
  NL question
    ↓
  parse_intent()
    ├─ client=None                     → _parse_heuristic()       [heuristic_intent.py]
    ├─ client is LocalHFPlannerClient   → _parse_hf_local_compact() [hf_local_intent.py]
    │                                     compact JSON, one repair/retry pass, then
    │                                     HFLocalPlannerAbstention (no tool execution)
    └─ client is any other PlannerClient → _parse_with_llm()       [llm_intent.py]
                       ├─ primary: _plan_analysis_internal()
                       │           → _validate_analysis_plan()   [leakage_validation.py]
                       │           → AnalysisPlan
                       │           → _derive_structured_intent_from_plan()
                       └─ fallback: _parse_with_json_prompt()
                                    → _normalize_llm_output()
                                    → _validate_resolved_intent() [leakage_validation.py]
                                    → StructuredIntent

All three branches converge on the same StructuredIntent type before tool
execution — backend queries, evidence, leakage checks, and cohort-map
generation are shared and do not know which branch produced the intent.
"""

from __future__ import annotations

from typing import Any

from alz_agent.schemas import AnalysisPlan, CatalogEntry

# ---------------------------------------------------------------------------
# Re-export all public and semi-public symbols so existing imports keep working.
# Tests and eval code import these names directly from alz_agent.agent.intent.
# ---------------------------------------------------------------------------

from alz_agent.agent.intent_schema import (  # noqa: F401
    ParseDebugInfo,
    QuestionType,
    StructuredIntent,
)
from alz_agent.agent.event_decomposition import (  # noqa: F401
    _COMPOUND_EVENT_PHRASES,
    _is_compound_event_phrase,
)
from alz_agent.agent.catalog_context import (  # noqa: F401
    _ALWAYS_INCLUDE_CODES,
    _CONTEXT_STOPWORDS,
    _build_catalog_context,
)
from alz_agent.agent.leakage_validation import (  # noqa: F401
    _COHORT_ROLES,
    _COLUMN_FIELD_ROLES,
    _LABEL_ROLES,
    _OUTCOME_ROLES,
    _PLAN_ROLE_TO_LEAKAGE_ROLE,
    _PREDICTOR_ROLES,
    _compute_leakage_finding,
    _validate_analysis_plan,
    _validate_resolved_intent,
)
from alz_agent.agent.heuristic_intent import (  # noqa: F401
    _HORIZON_RE,
    _MODALITY_PATTERNS,
    _MONTH_RE,
    _POPULATION_PATTERNS,
    _parse_heuristic,
)
from alz_agent.agent.hf_local_intent import (  # noqa: F401
    CompactIntent,
    HFLocalCompactValidationError,
    HFLocalPlannerAbstention,
    _build_compact_repair_prompt,
    _parse_hf_local_compact,
    _validate_and_adapt,
    build_hf_local_compact_prompt,
    compact_intent_to_structured_intent,
    parse_hf_local_compact_json,
    validate_hf_local_compact_intent,
)
from alz_agent.agent.providers import LocalHFPlannerClient
from alz_agent.agent.llm_intent import (  # noqa: F401
    _ANALYSIS_PLAN_TOOL,
    _ANALYSIS_TYPE_TO_QUESTION_TYPE,
    _EXPLICIT_ZERO_HORIZON_RE,
    _FILL_INTENT_TOOL,
    _LLM_MODALITY_ALIASES,
    _NACCUDSD_CODE_TO_LABEL,
    _ROLE_SUBSCHEMA,
    _VISIT_ANCHOR_MAP,
    _derive_structured_intent_from_plan,
    _normalize_llm_output,
    _parse_with_json_prompt,
    _parse_with_llm,
    _plan_analysis_internal,
    plan_analysis,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_intent(
    nl_question: str,
    client: Any = None,
    model: str | None = None,
    catalog: list[CatalogEntry] | None = None,
    debug_sink: list[str] | None = None,
) -> StructuredIntent:
    """Parse a natural-language research question into a StructuredIntent.

    Args:
        nl_question: The user's research question.
        client:      Optional ``anthropic.Anthropic`` instance, or a PlannerClient
                     (AnthropicPlannerClient / LocalHFPlannerClient / HFOpenAIPlannerClient).
                     When supplied, the LLM fills the schema via catalog-grounded
                     tool use, followed by deterministic validation.
                     When None, a keyword heuristic fills it (for tests/CI).
        model:       Model name override. Defaults to claude-haiku-4-5-20251001.
        catalog:     Active catalog entries for variable resolution.
                     Required for meaningful LLM-mode resolution.
                     When None in LLM mode, variable resolution fields will be null.
        debug_sink:  hf-local only. When client is a LocalHFPlannerClient and this
                     is provided, the raw compact-JSON text of every model call
                     (attempt + repair) is appended to it in order, regardless of
                     whether parsing succeeds or abstains. Ignored by every other
                     path (heuristic, DeepSeek/Anthropic) — does not change their
                     behavior.

    Raises:
        HFLocalPlannerAbstention: hf-local (LocalHFPlannerClient) produced invalid
            compact JSON even after one repair/retry pass, or the model/semantic
            guard classified the question as unsupported. No tools are executed
            in this case — the caller receives no StructuredIntent to dispatch.
    """
    if isinstance(client, LocalHFPlannerClient):
        return _parse_hf_local_compact(
            nl_question, client, model=model, catalog=catalog or [], debug_sink=debug_sink
        )
    if client is not None:
        return _parse_with_llm(nl_question, client, model=model, catalog=catalog or [])
    return _parse_heuristic(nl_question)
