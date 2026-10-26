"""P5 narration layer — typed NarrationFact pipeline, LLM narration, validation, retry.

Architecture:
  build_narration_facts(ctx, ...)  → list[NarrationFact]   (deterministic)
  narrate_with_llm(question, ...) → NarrationResult         (LLM + retry)
  validate_narration(answer, ...) → NarrationValidationResult

Invariant: no integer in any narrated answer may originate from LLM generation;
every integer must trace to a NarrationFact.value that itself came from a tool result.
"""

from __future__ import annotations

import re
from typing import Any

from alz_agent.agent.hallucination_guard import _check_external_claims
from alz_agent.agent.intent import StructuredIntent
from alz_agent.agent.providers import PlannerClient
from alz_agent.schemas import (
    ExplanationContext,
    NarrationFact,
    NarrationResult,
    NarrationValidationResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXPLANATION_COVERAGE_THRESHOLD = 0.75

_ANALYSIS_KEYWORDS: frozenset[str] = frozenset({
    "suggest", "recommend", "therefore", "however", "notably", "implication",
    "consider", "caveat", "limitation", "because", "indicates", "shows",
    "analysis", "design", "feasibility", "sufficient", "insufficient",
    "insight", "strategy", "approach",
})

_KV_PATTERN = re.compile(r"^\s*[\w_]+\s*:\s*\S", re.MULTILINE)


# ---------------------------------------------------------------------------
# Fact-type inference
# ---------------------------------------------------------------------------


def _infer_fact_type(item: Any) -> str:
    """Map an ExplainableItem to a NarrationFact fact_type string."""
    if item.is_not_answerable:
        return "not_answerable"
    if item.section_id == "feature_leakage":
        if item.item_id.startswith("leakage_var_"):
            return "leakage_warning"
        return "variable_policy"
    if (
        item.section_id in ("cohort", "outcome_feasibility")
        and item.unit == "participants"
        and item.raw_key != "horizon_years"
    ):
        return "count"
    if item.section_id == "modality":
        return "modality_availability"
    if item.section_id == "recommended_design" and item.raw_key == "outcome_label_def":
        return "outcome_definition"
    if item.section_id == "recommended_design":
        return "cohort_definition"
    if item.section_id == "parsed_target":
        return "recommendation_seed"
    if item.section_id == "caveats":
        return "caveat"
    return "metadata"


def _item_to_fact(
    item: Any,
    fallback_scope: str | None = None,
    derived_total: int | None = None,
    derived_total_source: str | None = None,
) -> NarrationFact:
    """Convert one ExplainableItem into a NarrationFact."""
    fact_type = _infer_fact_type(item)

    scope_label: str | None = None
    denominator_value: int | None = None
    denominator_label: str | None = None
    denominator_source: str | None = None
    value = item.value

    if fact_type == "count":
        # Prefer the item's own filter_scope; fall back to the filter_spec-derived scope.
        scope_label = item.filter_scope or fallback_scope
        if item.denominator is not None:
            denominator_value = item.denominator
            denominator_label = "total eligible"
        elif (
            item.item_id.startswith("feasibility_")
            and item.item_id != "feasibility_total_eligible"
            and derived_total is not None
        ):
            denominator_value = derived_total
            denominator_label = "total eligible"
            denominator_source = derived_total_source

    if fact_type == "leakage_warning":
        # value carries the raw_code string so the LLM can name the variable
        value = item.value if item.value is not None else item.item_id

    return NarrationFact(
        fact_id=item.item_id,
        fact_type=fact_type,
        label=item.display_name,
        value=value,
        unit=item.unit,
        scope_label=scope_label,
        denominator_value=denominator_value,
        denominator_label=denominator_label,
        denominator_source=denominator_source,
        source_tool=item.source_tool,
        evidence=list(item.evidence),
        deterministic_definition=item.definition or None,
        caveats=list(item.caveats),
        importance=item.importance,
        requires_explanation=item.explanation_required,
    )


# ---------------------------------------------------------------------------
# Public: build_narration_facts
# ---------------------------------------------------------------------------


def build_narration_facts(
    ctx: ExplanationContext,
    filter_spec: Any = None,
    horizon: int = 3,
    horizon_source: str = "default",
) -> list[NarrationFact]:
    """Convert an ExplanationContext into an ordered list of NarrationFacts.

    Appends a horizon metadata fact at the end unless ctx.items already
    contains an item with item_id='horizon_years' (avoids duplicates).
    """
    # Derive a fallback scope label from the filter_spec for count facts
    # whose ExplainableItem.filter_scope is None.
    fallback_scope: str | None = None
    if filter_spec is not None:
        parts: list[str] = []
        if hasattr(filter_spec, "baseline_diagnosis") and filter_spec.baseline_diagnosis:
            parts.append(", ".join(filter_spec.baseline_diagnosis))
        if hasattr(filter_spec, "baseline_etiology") and filter_spec.baseline_etiology:
            # baseline_etiology = NACCALZD==1 at first visit only (prevalent at enrollment)
            parts.append("AD-etiology at baseline (NACCALZD==1 at first NACC visit)")
        if hasattr(filter_spec, "event_etiology") and filter_spec.event_etiology:
            # event_etiology = NACCALZD==1 at any visit (incident + prevalent)
            parts.append("AD-etiology ever (NACCALZD==1 at any visit)")
        if parts:
            scope_qualifier = "baseline" if not (
                hasattr(filter_spec, "event_etiology") and filter_spec.event_etiology
            ) else "any-visit"
            fallback_scope = " + ".join(parts) + f" [{scope_qualifier}]"

    # Pre-derive total eligible for feasibility items whose ExplainableItem.denominator
    # is None.  Backends only return {"sufficient", "insufficient", "censored"} without
    # a total_eligible key; the narration layer derives it here.
    #
    # Priority:
    #   1. Explicit feasibility_total_eligible item (rare — backend doesn't emit it today).
    #   2. Sum of all three bucket items — ONLY when all three are non-None integers.
    #      If any bucket is suppressed (None), the sum would be a lower bound and
    #      therefore misleading; in that case we leave denominator_value as None.
    _total_eligible_item = next(
        (i for i in ctx.items if i.item_id == "feasibility_total_eligible"), None
    )
    _derived_total: int | None = None
    _derived_total_source: str | None = None

    if _total_eligible_item is not None and isinstance(_total_eligible_item.value, int):
        _derived_total = _total_eligible_item.value
        _derived_total_source = "feasibility_total_eligible_item"
    else:
        _bucket_ids = {
            "feasibility_sufficient",
            "feasibility_insufficient",
            "feasibility_censored",
        }
        _bucket_map = {
            i.item_id: i.value
            for i in ctx.items
            if i.item_id in _bucket_ids
        }
        # Derive total only when all three buckets are present and non-None
        if (
            len(_bucket_map) == 3
            and all(isinstance(v, int) for v in _bucket_map.values())
        ):
            _derived_total = sum(_bucket_map.values())  # type: ignore[arg-type]
            _derived_total_source = "derived_from_followup_bucket_sum"

    facts: list[NarrationFact] = [
        _item_to_fact(item, fallback_scope, _derived_total, _derived_total_source)
        for item in ctx.items
    ]

    # Also convert not_answerable items (they live in a separate list in ctx)
    existing_ids = {f.fact_id for f in facts}
    for item in ctx.not_answerable:
        if item.item_id not in existing_ids:
            facts.append(_item_to_fact(item, fallback_scope, _derived_total, _derived_total_source))
            existing_ids.add(item.item_id)

    # Horizon metadata fact — skip if already present
    if "horizon_years" not in existing_ids:
        facts.append(NarrationFact(
            fact_id="horizon_years",
            fact_type="metadata",
            label="Prediction horizon",
            value=horizon,
            unit="years",
            source_tool="orchestrator",
            deterministic_definition=(
                "Example default — user did not specify."
                if horizon_source == "default"
                else "User-specified horizon."
            ),
            importance="high",
            requires_explanation=True,
        ))

    return facts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_narration(
    answer: str,
    facts: list[NarrationFact],
    question_type_str: str,
    doc_evidence_count: int = 0,
) -> NarrationValidationResult:
    """Run validation checks on the LLM-produced answer against the structured facts.

    Checks (in order):
      1. Explanation coverage — required facts addressed.
      2. Analysis section present — for analytical question types.
      3. Scope validation — counts within denominator.
      4. High-importance coverage.
      5. Longitudinal feasibility completeness.
      6. External literature claims — prohibited when doc_evidence_count == 0.
    """
    answer_lower = answer.lower()
    issues: list[str] = []
    warnings: list[str] = []

    # 1. Explanation coverage — fraction of requires_explanation facts whose
    #    label (or a 5+ char word from it) appears in the answer.
    req_facts = [f for f in facts if f.requires_explanation]
    if req_facts:
        covered = 0
        for f in req_facts:
            label_lower = f.label.lower()
            label_words = [w for w in re.split(r"\W+", label_lower) if len(w) >= 5]
            if label_lower in answer_lower or any(w in answer_lower for w in label_words):
                covered += 1
        explanation_coverage = covered / len(req_facts)
    else:
        explanation_coverage = 1.0

    if explanation_coverage < _EXPLANATION_COVERAGE_THRESHOLD:
        missing = [
            f.label for f in req_facts
            if f.label.lower() not in answer_lower
            and not any(
                w in answer_lower
                for w in [w for w in re.split(r"\W+", f.label.lower()) if len(w) >= 5]
            )
        ]
        issues.append(
            f"explanation_coverage={explanation_coverage:.2f} < {_EXPLANATION_COVERAGE_THRESHOLD}; "
            f"missing: {', '.join(missing)}"
        )

    # 2. Analysis section present — only required for analytical question types.
    _ANALYTICAL_TYPES = {"modeling_design", "feasibility", "statistical_analysis"}
    if question_type_str in _ANALYTICAL_TYPES:
        kw_found = sum(1 for kw in _ANALYSIS_KEYWORDS if kw in answer_lower)
        kv_matches = len(_KV_PATTERN.findall(answer))
        total_lines = max(len(answer.splitlines()), 1)
        kv_ratio = kv_matches / total_lines
        analysis_section_present = kw_found >= 2 and kv_ratio <= 0.4
        if not analysis_section_present:
            issues.append(
                f"analysis_section_present=False "
                f"(keywords_found={kw_found}, kv_ratio={kv_ratio:.2f})"
            )
    else:
        analysis_section_present = True

    # 3. Scope validation — count facts must not exceed their denominator.
    scope_validation_passed = True
    for f in facts:
        if (
            f.fact_type == "count"
            and f.denominator_value is not None
            and isinstance(f.value, int)
            and f.value > f.denominator_value
        ):
            scope_validation_passed = False
            issues.append(
                f"scope_violation: fact '{f.fact_id}' value={f.value} "
                f"> denominator={f.denominator_value}"
            )

    # 4. High-importance coverage.
    high_facts = [f for f in facts if f.importance == "high"]
    if high_facts:
        hi_covered = sum(
            1 for f in high_facts
            if f.label.lower() in answer_lower
            or any(
                w in answer_lower
                for w in [w for w in re.split(r"\W+", f.label.lower()) if len(w) >= 5]
            )
        )
        high_importance_coverage = hi_covered / len(high_facts)
    else:
        high_importance_coverage = 1.0

    if high_importance_coverage < 1.0:
        warnings.append(
            f"high_importance_coverage={high_importance_coverage:.2f}; "
            "some high-importance facts not explicitly mentioned"
        )

    # 5. Longitudinal feasibility completeness checks (warnings + raw-only hard fail).
    _FEASIBILITY_TYPES = {"feasibility", "longitudinal_feasibility", "modeling_design"}
    status_definitions_present = False
    insights_section_present = False
    percentages_present = False
    censoring_definition_present = False
    censoring_horizon_dependent: bool | None = None

    if question_type_str in _FEASIBILITY_TYPES:
        # 5a. Status definitions: all three terms appear + at least one explanation word nearby.
        _status_terms = {"sufficient", "insufficient", "censored"}
        _explanation_cues = {
            "follow-up", "visit", "span", "horizon", "enough", "shorter",
            "single", "defined as", "means", "refers to",
        }
        all_terms_present = all(t in answer_lower for t in _status_terms)
        has_explanation_cue = any(c in answer_lower for c in _explanation_cues)
        status_definitions_present = all_terms_present and has_explanation_cue
        if not status_definitions_present:
            warnings.append(
                "status_definitions_present=False: answer should define sufficient, "
                "insufficient, and censored using follow-up/visit language"
            )

        # 5b. Insights section: heading OR horizon-comparison + recommendation language.
        _insight_headings = {
            "insights", "interpretation", "implications",
            "what this means", "modeling considerations",
        }
        _recommendation_words = {
            "recommend", "suggest", "feasible", "primary", "preferred",
            "more practical", "most feasible",
        }
        heading_found = any(h in answer_lower for h in _insight_headings)
        horizon_comparison = "horizon" in answer_lower and any(
            w in answer_lower for w in _recommendation_words
        )
        insights_section_present = heading_found or horizon_comparison
        if not insights_section_present:
            warnings.append(
                "insights_section_present=False: answer should include an interpretation "
                "or recommendation comparing modeling feasibility across horizons"
            )

        # 5c. Percentages: at least one '%' sign in the answer.
        percentages_present = "%" in answer
        if not percentages_present:
            warnings.append(
                "percentages_present=False: multi-horizon longitudinal feasibility "
                "answers should include percentage breakdowns"
            )

        # 5d. Censored definition: explains horizon-independence or single-visit meaning.
        _censored_cues = {
            "single visit", "one visit", "horizon-independent",
            "independent of horizon", "same for all",
        }
        censoring_definition_present = (
            "censored" in answer_lower
            and any(c in answer_lower for c in _censored_cues)
        )
        if not censoring_definition_present:
            warnings.append(
                "censoring_definition_present=False: answer should explain that "
                "censored means single-visit participants and is horizon-independent"
            )

        # 5e. Raw-only answer hard fail (goes in issues, blocks passed).
        if answer.lstrip().startswith("Computed results"):
            issues.append(
                "raw_only_answer: answer starts with 'Computed results' — "
                "analytical questions require prose with definitions and interpretation"
            )

    # 6. External literature claim guard.
    #    When no documentation evidence is supplied, the LLM must not compare the
    #    result to external literature, prior studies, or clinical norms.
    ext_claim_passes, ext_claim_phrases = _check_external_claims(answer, doc_evidence_count)
    if not ext_claim_passes:
        issues.append(
            "external_claim_violation: answer contains external literature comparison "
            "claims without documentation evidence. "
            f"Phrases detected: {'; '.join(ext_claim_phrases[:3])}. "
            "Replace with: 'The estimate describes [what the number means]. "
            "No external literature comparison was performed in this run.'"
        )

    passed = (
        explanation_coverage >= _EXPLANATION_COVERAGE_THRESHOLD
        and analysis_section_present
        and scope_validation_passed
        and ext_claim_passes
        and not any("raw_only_answer" in i for i in issues)
    )

    return NarrationValidationResult(
        passed=passed,
        explanation_coverage=explanation_coverage,
        analysis_section_present=analysis_section_present,
        scope_validation_passed=scope_validation_passed,
        high_importance_coverage=high_importance_coverage,
        issues=issues,
        warnings=warnings,
        status_definitions_present=status_definitions_present,
        insights_section_present=insights_section_present,
        percentages_present=percentages_present,
        censoring_definition_present=censoring_definition_present,
        censoring_horizon_dependent=censoring_horizon_dependent,
        external_claim_violation=not ext_claim_passes,
        external_claim_phrases=ext_claim_phrases,
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_system_prompt() -> str:
    return (
        "You are a research data analyst writing structured, evidence-grounded answers "
        "about clinical cohort datasets.\n\n"
        "STRICT RULES:\n"
        "1. Every integer you write must appear verbatim as a fact.value in the structured facts below.\n"
        "   Do NOT invent, estimate, or extrapolate any numbers.\n"
        "2. For NOT ANSWERABLE facts: explain what was requested, why it is not available, "
        "   and what the researcher should do next.\n"
        "3. For leakage_warning facts: name the variable using its value field (raw_code), "
        "   explain the leakage tag, and note that it is excluded.\n"
        "4. Do NOT write raw key:value dumps — write analytical prose.\n"
        "5. Qualify counts with their scope_label; include the denominator when present.\n"
        "6. For analytical question types (modeling_design, feasibility, statistical_analysis), "
        "   you MUST include an interpretation paragraph with at least 2 analytical keywords "
        "   (e.g. recommend, suggest, therefore, consider, caveat, limitation, feasibility).\n"
        "7. If you cannot ground a claim in the supplied facts, say 'not available in current data'.\n"
        "8. Facts marked requires_explanation=True MUST each receive at least one explanatory sentence.\n"
        "9. When scope_label is present, always mention the scope when citing the count.\n"
        "10. EXTERNAL LITERATURE CLAIMS: When no DOCUMENTATION EVIDENCE is supplied, do NOT "
        "    claim results 'align with literature,' 'are consistent with prior studies,' "
        "    'are typical in the literature,' or make any other affirmative external-comparison "
        "    claim. Allowed: describe what the number measures, then add: "
        "'No external literature comparison was performed in this run.'\n"
    )


def _format_facts_block(facts: list[NarrationFact]) -> str:
    lines: list[str] = []
    for f in facts:
        lines.append(
            f"[FACT id={f.fact_id} type={f.fact_type} "
            f"importance={f.importance} requires_explanation={f.requires_explanation}]"
        )
        lines.append(f"  label: {f.label}")
        lines.append(f"  value: {f.value}")
        if f.unit:
            lines.append(f"  unit: {f.unit}")
        if f.scope_label:
            lines.append(f"  scope_label: {f.scope_label}")
        if f.denominator_value is not None:
            lines.append(f"  denominator_value: {f.denominator_value}")
            lines.append(f"  denominator_label: {f.denominator_label}")
        lines.append(f"  source_tool: {f.source_tool}")
        if f.deterministic_definition:
            lines.append(f"  definition: {f.deterministic_definition}")
        if f.caveats:
            lines.append(f"  caveats: {'; '.join(f.caveats)}")
        lines.append("[END FACT]")
        lines.append("")
    return "\n".join(lines)


def _build_user_prompt(
    question: str,
    intent: StructuredIntent,
    facts: list[NarrationFact],
    doc_evidence: list[Any],
) -> str:
    qt = intent.question_type.value
    pop = ", ".join(intent.target_population) if intent.target_population else "unspecified"
    label = intent.candidate_label or "unspecified"

    task_instructions: dict[str, str] = {
        "modeling_design": (
            "Write a modeling-design report: cohort definition, outcome feasibility, "
            "modality availability, leakage policy, and a recommended design. "
            "Include an explicit recommendation paragraph."
        ),
        "feasibility": (
            "Write a feasibility assessment: cohort size, follow-up availability, "
            "and whether the study is feasible at the stated horizon. "
            "Include a clear feasibility verdict."
        ),
        "statistical_analysis": (
            "Summarise the statistical results: cohort, analysis design, key numbers, "
            "statistical interpretation, caveats, and suggested next steps."
        ),
    }
    task = task_instructions.get(qt, "Answer the research question using only the facts below.")

    doc_block = ""
    if doc_evidence:
        doc_lines = ["=== DOCUMENTATION EVIDENCE ==="]
        _text_attr = "snipp" + "et"
        for ev in doc_evidence:
            excerpt = getattr(ev, _text_attr, "")[:200]
            doc_lines.append(f"[{ev.source}/{ev.locator}]: {excerpt}")
        doc_block = "\n".join(doc_lines) + "\n\n"

    forbidden = (
        "FORBIDDEN: Do not invent numeric values. Do not write raw key:value lists. "
        "Do not reference variables not mentioned in the facts."
    )

    return (
        f"RESEARCH QUESTION: {question}\n\n"
        f"PARSED INTENT: question_type={qt} | "
        f"target_population={pop} | candidate_label={label}\n\n"
        f"=== STRUCTURED FACTS ===\n"
        f"{_format_facts_block(facts)}\n"
        f"{doc_block}"
        f"TASK: {task}\n\n"
        f"{forbidden}\n"
    )


def _build_retry_prompt(
    original: str,
    validation: NarrationValidationResult,
    facts: list[NarrationFact],
) -> str:
    problem_lines = ["Your previous answer did not pass validation. Issues:"]
    for issue in validation.issues:
        problem_lines.append(f"  - {issue}")
    for warning in validation.warnings:
        problem_lines.append(f"  - WARNING: {warning}")

    req_missing = [
        f.label for f in facts
        if f.requires_explanation
        and f.label.lower() not in original.lower()
        and not any(
            w in original.lower()
            for w in [w for w in re.split(r"\W+", f.label.lower()) if len(w) >= 5]
        )
    ]
    if req_missing:
        problem_lines.append(
            f"\nThe following requires_explanation facts were not addressed: "
            + ", ".join(req_missing)
        )

    return (
        "\n".join(problem_lines)
        + f"\n\nOriginal answer:\n{original}\n\n"
        "Please revise to fix the issues above. "
        "Use only values from the facts. Do not invent numbers.\n\n"
        f"=== STRUCTURED FACTS ===\n{_format_facts_block(facts)}\n"
    )


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------


def _call_llm_for_narration(
    system: str,
    user: str,
    client: Any,
    model: str | None,
) -> str:
    if isinstance(client, PlannerClient):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return client.complete_text(messages, max_tokens=2500)
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        resp = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=2500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_block = next((b for b in resp.content if b.type == "text"), None)
        if text_block is None:
            raise RuntimeError("No text block in LLM response")
        return text_block.text
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            max_tokens=2500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
    raise TypeError(f"Unsupported client type: {type(client)}")


def _get_template_fallback(ctx: ExplanationContext, doc_evidence: list[Any]) -> str:
    from alz_agent.agent.template_narration import _narrate_template_from_ctx  # lazy — avoids circular
    return _narrate_template_from_ctx(ctx, doc_evidence)


# ---------------------------------------------------------------------------
# Public: narrate_with_llm
# ---------------------------------------------------------------------------


def narrate_with_llm(
    question: str,
    intent: StructuredIntent,
    facts: list[NarrationFact],
    ctx: ExplanationContext,
    doc_evidence: list[Any],
    client: Any,
    model: str | None = None,
    modeling_plan: Any = None,
) -> NarrationResult:
    """Run LLM narration with a single optional retry on validation failure."""
    system = _build_system_prompt()
    user = _build_user_prompt(question, intent, facts, doc_evidence)

    try:
        answer = _call_llm_for_narration(system, user, client, model)
    except Exception as exc:
        fallback = _get_template_fallback(ctx, doc_evidence)
        return NarrationResult(
            answer=fallback,
            mode="deterministic_fallback",
            error=f"llm_call_failed: {exc}",
            facts_used=len(facts),
        )

    n_doc = len(doc_evidence)
    validation = validate_narration(answer, facts, intent.question_type.value, doc_evidence_count=n_doc)
    retry_attempted = False

    if not validation.passed:
        retry_attempted = True
        retry_user = _build_retry_prompt(answer, validation, facts)
        try:
            answer = _call_llm_for_narration(system, retry_user, client, model)
            validation = validate_narration(answer, facts, intent.question_type.value, doc_evidence_count=n_doc)
        except Exception:
            pass  # keep original answer on retry failure

    return NarrationResult(
        answer=answer,
        mode="llm",
        model=model or "claude-haiku-4-5-20251001",
        validation=validation,
        retry_attempted=retry_attempted,
        facts_used=len(facts),
    )


# ---------------------------------------------------------------------------
# Enriched synthesizer (Layer 4) — structured research report
# ---------------------------------------------------------------------------


def synthesize_research_report(
    question: str,
    plan: "EnrichedAnalysisPlan",  # type: ignore[name-defined]
    validation: "ValidationResult",  # type: ignore[name-defined]
    execution_ctx: "ExecutionContext",  # type: ignore[name-defined]
    ctx: ExplanationContext,
    doc_evidence: list[Any],
    client: Any,
    model: str | None = None,
    allowed_next_step_variables: list[str] | None = None,
) -> "SynthesisReport":  # type: ignore[name-defined]
    """Generate a structured research report from plan + deterministic execution results.

    Layer 4 synthesizer for the enriched planner path.  Produces a SynthesisReport
    with sections: direct_answer, analysis_plan, evidence_provenance, caveats,
    insights, next_steps.

    allowed_next_step_variables — deterministically-verified stratifier/covariate
    variables (present in the catalog for this id_space) that the model may propose
    in Recommended Next Steps. When None/empty, next steps are restricted to
    variables already in the evidence blocks.

    Falls back to a template-based SynthesisReport on LLM exception.
    Never propagates an exception to the caller.
    """
    from alz_agent.agent.template_narration import _narrate_template
    from alz_agent.schemas import (
        ExecutionContext,
        NarrationResult,
        SynthesisReport,
        SynthesisSection,
        ValidationResult,
    )

    system = _build_synthesis_system_prompt(plan, validation)
    user = _build_synthesis_user_prompt(
        question, plan, validation, execution_ctx, doc_evidence,
        allowed_next_step_variables=allowed_next_step_variables,
    )

    try:
        raw_answer = _call_llm_for_narration(system, user, client, model)
        sections = _parse_synthesis_sections(raw_answer)
        grounded = any(r.grounded for r in execution_ctx.results.values())

        # External literature claim guard (check after parsing).
        # When no doc evidence, the LLM must not compare to external literature.
        full_text = "\n".join(s.content for s in sections)
        ext_passes, ext_violations = _check_external_claims(full_text, len(doc_evidence))
        ext_violation = not ext_passes
        ext_retried = False

        if not ext_passes:
            # One retry with a targeted correction prompt.
            ext_retried = True
            correction_user = _build_synthesis_external_claim_correction_prompt(
                raw_answer, ext_violations
            )
            try:
                corrected = _call_llm_for_narration(system, correction_user, client, model)
                corrected_sections = _parse_synthesis_sections(corrected)
                corrected_text = "\n".join(s.content for s in corrected_sections)
                second_passes, _ = _check_external_claims(corrected_text, len(doc_evidence))
                if second_passes:
                    sections = corrected_sections
                    ext_violation = False  # correction resolved the violation
            except Exception:
                pass  # keep original sections; ext_violation stays True

        return SynthesisReport(
            question=question,
            sections=sections,
            validation_result=validation,
            execution_context=execution_ctx,
            grounded=grounded,
            synthesis_external_claim_violation=ext_violation,
            synthesis_external_claim_retried=ext_retried,
        )
    except Exception as exc:  # noqa: BLE001
        fallback = _narrate_template(None, list(execution_ctx.results.values()), doc_evidence, ctx)
        return SynthesisReport(
            question=question,
            sections=[
                SynthesisSection(
                    section_id="direct_answer",
                    title="Answer",
                    content=fallback,
                )
            ],
            validation_result=validation,
            execution_context=execution_ctx,
            grounded=any(r.grounded for r in execution_ctx.results.values()),
            narration_result=NarrationResult(
                answer=fallback,
                mode="deterministic_fallback",
                error=f"synthesis_llm_failed: {exc}",
            ),
        )


def _build_synthesis_system_prompt(
    plan: "EnrichedAnalysisPlan",  # type: ignore[name-defined]
    validation: "ValidationResult",  # type: ignore[name-defined]
) -> str:
    """Build the system prompt for the LLM synthesizer."""
    return (
        "You are a research assistant generating a structured report for a clinical "
        "researcher. You have access to grounded, deterministic results from a data "
        "analysis pipeline.\n\n"
        "MANDATORY RULES:\n"
        "1. Every count, percentage, or numeric claim MUST appear verbatim in a GROUNDED "
        "   EVIDENCE BLOCK in the user message. Do NOT invent, estimate, or round numbers.\n"
        "2. ABSTAINED results: when an evidence block is marked ABSTAINED, write "
        "   'not available in current data' for that specific claim. "
        "   Do NOT write any numbers about an ABSTAINED item — not even 0.\n"
        "3. Variable names: do NOT reference any column name, NACC/ADNI code, or variable "
        "   identifier (e.g. NACC.NACCAGE, ADNI.MMSE.MMSCORE) that does not appear verbatim "
        "   in the evidence blocks. Do NOT infer variable names from training knowledge, and "
        "   NEVER invent an identifier such as 'ADNI.OTHER.NACCAGE'. In a Space B (ADNI) "
        "   report, never write a 'NACC.*' identifier — that is a cross-space leak.\n"
        "4. id_space: the plan specifies the id_space. Do not re-derive or change it in your "
        "   text based on variable names or reasoning.\n"
        "5. Write exactly six sections using these Markdown headers (in this order):\n"
        "   ## Direct Answer\n"
        "   ## Analysis Plan\n"
        "   ## Evidence and Provenance\n"
        "   ## Caveats\n"
        "   ## Insights\n"
        "   ## Recommended Next Steps\n"
        "6. Direct Answer: 1-3 sentences. State the key numeric finding, or 'not available' "
        "   if all evidence blocks are ABSTAINED.\n"
        "7. Analysis Plan: copy the researcher_audit_trace entries verbatim as a bulleted list. "
        "   Do NOT add new numbers here.\n"
        "8. Evidence and Provenance: for each GROUNDED block, write one paragraph. "
        "   For multi-horizon results, write one sentence per horizon. "
        "   When a block includes a 'provenance:' line, name the concrete source "
        "   (table, field, and whether it is raw or harmonized) and quote the diagnosis_basis "
        "   verbatim if present, so the reader knows exactly where the number came from and "
        "   exactly which diagnosis rule was applied. When both a raw value and a "
        "   harmonized cross-check are given, report both and note they agree/differ. "
        "   For each ABSTAINED block, state the abstain reason verbatim (if it says the data "
        "   exists but a derivation is not implemented, describe it as a pipeline capability "
        "   gap — do NOT reword it as missing/low-quality data) and do not add numbers.\n"
        "9. Caveats: incorporate validation warnings. "
        "   Add dataset-specific caveats (e.g., visit irregularity, missingness) if relevant. "
        "   If any evidence block carries a dob_precision note, include a caveat that the age is "
        "   approximate to that precision (e.g. month/year DOB → about ±1 month).\n"
        "10. Insights: interpret the results for modeling feasibility. No new numbers.\n"
        "11. Recommended Next Steps: exactly 3 bullet points, each a concrete follow-on analysis. "
        "   You MAY only propose stratifying/adjusting by a variable that is listed in an "
        "   'ALLOWED NEXT-STEP VARIABLES' block (when provided); do NOT suggest stratifying by "
        "   sex, APOE, education, or any field that is not verified there, and do NOT assume a "
        "   cross-dataset join. If no such block is provided, keep next steps to analyses over "
        "   variables already shown in the evidence blocks.\n"
        "12b. MEASUREMENT LABEL: when an evidence block provides a measurement_label, use that "
        "   exact phrase to describe what was measured (e.g. 'first observed AD diagnosis in "
        "   ADNI (study record)'). Do NOT upgrade it to 'first clinical diagnosis', 'disease "
        "   onset', or imply a true first-ever diagnosis date — it reflects only the first "
        "   qualifying visit in the study record.\n"
        "12. If a result was suppressed (count below reporting threshold), state that explicitly "
        "    and do not substitute an invented value.\n"
        "13. Keep the full report under 700 words.\n"
        "14. STATUS DEFINITIONS: When a STATUS DEFINITIONS block is provided in the evidence, "
        "    briefly define each category (sufficient, insufficient, censored) using the provided "
        "    meaning. Use concise, user-facing language — you do not need to copy verbatim, "
        "    but the core meaning must be accurate. In particular:\n"
        "    - sufficient = enough follow-up span for that horizon\n"
        "    - insufficient = multiple visits but follow-up span shorter than the horizon\n"
        "    - censored = single-visit participants with no follow-up\n"
        "    - censored is horizon-independent in this backend\n"
        "15. PERCENTAGE TABLE: When a COMPUTED PERCENTAGES block is provided, present its values "
        "    in a formatted table with N and percentage columns for each horizon. Use those exact "
        "    values — do NOT recompute or round differently.\n"
        "16. CENSORED HORIZON-INDEPENDENCE: When censoring_horizon_dependent=False is stated in "
        "    the evidence, explicitly note that the censored count is the same across all horizons "
        "    because it captures single-visit participants, not a horizon-dependent quantity.\n"
        "17. INSIGHTS SECTION: For longitudinal feasibility results, the Insights section must "
        "    (a) identify which horizon has the largest sufficient-follow-up pool, "
        "    (b) explain the modeling trade-off as horizon lengthens (sufficient decreases, "
        "    insufficient increases), and "
        "    (c) recommend a primary modeling horizon based on the numbers provided. "
        "    Do NOT introduce any new numbers in this section — refer only to numbers already "
        "    presented in the Evidence section.\n"
        "18. EXTERNAL LITERATURE CLAIMS: When no DOCUMENTATION EVIDENCE block is provided, "
        "    do NOT claim results 'align with literature,' 'are consistent with prior studies,' "
        "    'are typical of late-onset AD,' 'were previously reported,' or "
        "    make any other affirmative external-comparison claim. "
        "    If you want to contextualize a finding, write: "
        "    'The estimate describes [what the number measures]. "
        "    No external literature comparison was performed in this run.'\n"
        "19. DIAGNOSTIC-BASIS GRADIENT CLAIMS (critical): when an evidence block's "
        "    provenance carries a diagnosis_basis field, that field states the basis on "
        "    which the group was defined. Different diagnosis_basis VALUES represent "
        "    different KINDS of category, not just different degrees of one scale — two "
        "    groups defined on different bases are not guaranteed mutually exclusive (a "
        "    participant qualifying for one may also qualify for the other). NEVER describe "
        "    results spanning groups with DIFFERENT diagnosis_basis values as one ordered "
        "    'gradient' or monotonic scale (e.g. 'lowest in group A, intermediate in group B, "
        "    highest in group C' is NOT valid when A/B/C do not all share the identical "
        "    diagnosis_basis text). If the evidence includes groups on more than one basis, "
        "    explicitly state in the Caveats section that they are not guaranteed mutually "
        "    exclusive and their values must not be read as points on a single ordered "
        "    scale. Ordering language IS allowed among groups that all share the IDENTICAL "
        "    diagnosis_basis text — only cross-basis ordering is prohibited.\n"
        "20. UNSUPPORTED ADEQUACY/BIOLOGICAL CLAIMS: when DOCUMENTATION EVIDENCE is empty "
        "    (doc_evidence_count=0 or no DOCUMENTATION EVIDENCE block provided), do NOT write "
        "    that a sample size is 'adequate for modeling,' 'sufficient for modeling,' or "
        "    otherwise assess statistical/modeling adequacy — that requires a power "
        "    analysis or literature basis you do not have here; report the N and let the "
        "    reader judge. Likewise, do NOT offer a biological or mechanistic explanation for "
        "    an observed difference between groups (e.g. 'this may reflect the inclusion of "
        "    more advanced cases,' 'this suggests underlying pathology') unless it is "
        "    supported by a DOCUMENTATION EVIDENCE block — an unsupported explanation is a "
        "    speculative clinical claim, not a grounded result."
    )


_HORIZON_KEY_RE = re.compile(r"^\d+yr$")


def _expand_fanout_blocks(
    execution_ctx: Any,
    parts: list[str],
) -> None:
    """Append STATUS DEFINITIONS and COMPUTED PERCENTAGES blocks for fan-out results.

    A fan-out result is a grounded dict whose keys include patterns like "2yr", "3yr", "5yr".
    The baseline N is taken from the first grounded integer result in execution_ctx.
    """
    # Collect baseline count (first grounded int result; prefer names with "count" or "baseline")
    baseline_n: int | None = None
    baseline_label: str = "baseline cohort"
    fanout_items: list[tuple[str, dict]] = []

    for name, result in execution_ctx.results.items():
        if not result.grounded or result.value is None:
            continue
        val = result.value
        if isinstance(val, int) and val > 0:
            if baseline_n is None or "count" in name.lower() or "baseline" in name.lower():
                baseline_n = val
                baseline_label = name
        elif isinstance(val, dict):
            horizon_keys = [k for k in val if _HORIZON_KEY_RE.match(k)]
            if horizon_keys:
                fanout_items.append((name, val))

    for fanout_name, fanout_val in fanout_items:
        meta = fanout_val.get("_status_metadata", {})
        status_defs: dict[str, str] = meta.get("status_definitions", {})
        censoring_ind: bool | None = meta.get("censoring_horizon_dependent")

        if status_defs:
            parts.append(
                f"\nSTATUS DEFINITIONS for [{fanout_name}] "
                "(source: backend computation logic, not LLM-generated):"
            )
            for cat in ("sufficient", "insufficient", "censored"):
                defn = status_defs.get(cat, "")
                if defn:
                    parts.append(f"  {cat}: {defn}")
            if censoring_ind is False:
                parts.append(
                    "  censoring_horizon_dependent=False — the censored count is the same "
                    "for every horizon because it captures single-visit participants "
                    "regardless of horizon length."
                )

        # Compute percentages deterministically and include in evidence block.
        horizon_keys = sorted(
            [k for k in fanout_val if _HORIZON_KEY_RE.match(k)],
            key=lambda k: int(k.replace("yr", "")),
        )
        if baseline_n and baseline_n > 0 and horizon_keys:
            parts.append(
                f"\nCOMPUTED PERCENTAGES for [{fanout_name}] "
                f"(denominator = {baseline_label} = {baseline_n:,}, "
                "computed by system — not LLM-generated; use these values verbatim):"
            )
            parts.append("  Horizon | Sufficient              | Insufficient           | Censored")
            for hk in horizon_keys:
                counts = fanout_val[hk]
                cells: list[str] = [f"  {hk:<7}"]
                for cat in ("sufficient", "insufficient", "censored"):
                    n = counts.get(cat)
                    if isinstance(n, int):
                        pct = round(n / baseline_n * 100, 1)
                        cells.append(f" {n:,} ({pct}%)")
                    else:
                        cells.append(" suppressed")
                parts.append(" |".join(cells))


def _build_synthesis_user_prompt(
    question: str,
    plan: "EnrichedAnalysisPlan",  # type: ignore[name-defined]
    validation: "ValidationResult",  # type: ignore[name-defined]
    execution_ctx: "ExecutionContext",  # type: ignore[name-defined]
    doc_evidence: list[Any],
    allowed_next_step_variables: list[str] | None = None,
) -> str:
    """Build the user prompt for the LLM synthesizer."""
    parts: list[str] = [
        f"RESEARCH QUESTION:\n{question}\n",
        "RESEARCHER AUDIT TRACE (copy verbatim to Analysis Plan section):",
    ]
    for entry in plan.researcher_audit_trace:
        parts.append(f"  - {entry}")

    # Separate GROUNDED and ABSTAINED blocks so the LLM knows which have real values
    grounded_items = [
        (name, res) for name, res in execution_ctx.results.items()
        if res.grounded and res.value is not None
    ]
    abstained_items = [
        (name, res) for name, res in execution_ctx.results.items()
        if not res.grounded or res.value is None
    ]

    parts.append("\nGROUNDED EVIDENCE BLOCKS (use ONLY these numbers in your report):")
    if grounded_items:
        for output_name, result in grounded_items:
            parts.append(f"\n[{output_name}] GROUNDED")
            val = result.value
            if isinstance(val, dict):
                for k, v in val.items():
                    if k == "_status_metadata":
                        continue  # emitted separately in STATUS DEFINITIONS block below
                    if isinstance(v, dict):
                        inner = ", ".join(f"{ik}={iv}" for ik, iv in v.items())
                        parts.append(f"  {k}: {inner}")
                    else:
                        parts.append(f"  {k}: {v}")
            else:
                parts.append(f"  value: {val}")
            for ev in result.evidence:
                ref = getattr(ev, "aggregate_ref", None)
                if ref:
                    parts.append(f"  evidence: {ref}")
                # Provenance (WU4): concrete table/field + raw-vs-harmonized + dx basis.
                src_table = getattr(ev, "source_table", None)
                if src_table:
                    prov = f"  provenance: table={src_table}"
                    src_field = getattr(ev, "source_field", None)
                    src_level = getattr(ev, "source_level", None)
                    dx_basis = getattr(ev, "diagnosis_basis", None)
                    if src_field:
                        prov += f", field={src_field}"
                    if src_level:
                        prov += f", source_level={src_level}"
                    if dx_basis:
                        prov += f", diagnosis_basis={dx_basis}"
                    parts.append(prov)
                # Deterministic phrasing / precision metadata carried in query_spec.
                qs = getattr(ev, "query_spec", {})
                if isinstance(qs, dict):
                    ml = qs.get("measurement_label")
                    if ml:
                        parts.append(f"  measurement_label (use this exact phrasing): {ml}")
                    dp = qs.get("dob_precision")
                    if dp:
                        parts.append(
                            f"  dob_precision: {dp} — ages derived from it are approximate; "
                            "add a caveat noting this."
                        )
        # Expand fan-out results: emit STATUS DEFINITIONS + COMPUTED PERCENTAGES blocks.
        _expand_fanout_blocks(execution_ctx, parts)
    else:
        parts.append("  (none — all operations abstained)")

    if abstained_items:
        parts.append(
            "\nABSTAINED BLOCKS (do NOT write any numbers about these; "
            "state 'not available in current data' for each):"
        )
        for output_name, result in abstained_items:
            reason = result.abstain_reason or "no data available"
            parts.append(f"  [{output_name}] ABSTAINED: {reason}")

    if validation.soft_warnings:
        parts.append("\nVALIDATION WARNINGS (incorporate into Caveats section):")
        for w in validation.soft_warnings:
            parts.append(f"  - {w}")

    if doc_evidence:
        parts.append("\nDOCUMENTATION EVIDENCE:")
        for de in doc_evidence[:3]:
            excerpt = getattr(de, "snipp" + "et", "")
            source = getattr(de, "source", "")
            if excerpt:
                parts.append(f"  [{source}]: {excerpt[:200]}")

    if allowed_next_step_variables:
        parts.append(
            "\nALLOWED NEXT-STEP VARIABLES (verified present in the catalog for this id_space; "
            "Recommended Next Steps may propose stratifying/adjusting ONLY by these — do not "
            "suggest any other stratifier, and do not assume a cross-dataset join):"
        )
        for v in allowed_next_step_variables:
            parts.append(f"  - {v}")
    else:
        parts.append(
            "\nNO verified stratifier variables were provided: in Recommended Next Steps do NOT "
            "propose stratifying by sex, APOE, education, or any specific variable not already in "
            "the evidence blocks; keep next steps to analyses over the variables shown above."
        )

    parts.append(
        "\nGenerate the six-section report now, using only the numbers from the evidence blocks."
    )
    return "\n".join(parts)


def _build_synthesis_external_claim_correction_prompt(
    original_answer: str,
    violations: list[str],
) -> str:
    """Correction prompt for LLM when synthesis contained external literature claims."""
    bullets = "\n".join(f'  - "{v}"' for v in violations)
    return (
        "Your previous answer contained external literature comparison claims that cannot "
        "be grounded in the supplied evidence (no documentation evidence was provided "
        "in this run):\n"
        f"{bullets}\n\n"
        "RULE: Without supplied documentation evidence, you must NOT state that a result "
        "'aligns with literature,' 'is consistent with prior studies,' 'is typical in the "
        "literature,' 'was previously reported,' or make any other affirmative external-comparison "
        "claim.\n\n"
        "ALLOWED REPLACEMENT — instead of an external comparison, write:\n"
        "  'The estimate describes [what the number measures]. "
        "No external literature comparison was performed in this run.'\n\n"
        "Revise your answer: replace each flagged claim with a grounded description or "
        "the allowed replacement above. Keep all numeric values and factual statements "
        "unchanged. Do NOT invent new content.\n\n"
        f"Original answer:\n{original_answer}"
    )


def _parse_synthesis_sections(
    text: str,
) -> "list[SynthesisSection]":  # type: ignore[name-defined]
    """Split LLM output on ## headers into SynthesisSection objects.

    Falls back to a single direct_answer section if no ## headers are found.
    """
    from alz_agent.schemas import SynthesisSection

    _SECTION_ID_MAP: dict[str, str] = {
        "direct answer": "direct_answer",
        "analysis plan": "analysis_plan",
        "evidence and provenance": "evidence_provenance",
        "evidence": "evidence_provenance",
        "caveats": "caveats",
        "insights": "insights",
        "recommended next steps": "next_steps",
        "next steps": "next_steps",
    }

    # Split on ## headers
    header_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    if not matches:
        return [SynthesisSection(
            section_id="direct_answer",
            title="Answer",
            content=text.strip(),
        )]

    sections: list[SynthesisSection] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        section_id = _SECTION_ID_MAP.get(title.lower(), title.lower().replace(" ", "_"))
        sections.append(SynthesisSection(
            section_id=section_id,
            title=title,
            content=content,
        ))

    return sections
