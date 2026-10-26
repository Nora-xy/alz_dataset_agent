"""P8 — Eval harness: protects the project's core guarantees.

Four independent categories:
  1. golden_questions     — NL questions run end-to-end via run_agent on a
                             FixtureDataBackend; checked against expected
                             grounded status + answer-shape substrings.
  2. number_traceability  — for every golden answer, every numeric token must
                             trace to a tool result (the hallucination guard,
                             re-asserted offline against the same responses).
  3. abstention           — four known-ungrounded request shapes must abstain,
                             not answer: absent modality combo, uncovered
                             release, suppressed cells, unverified/unknown
                             label mapping.
  4. leakage              — known diagnosis-defining / circular /
                             harmonization-fit-scope features must appear in
                             the leakage blacklist and never in the suggested
                             feature whitelist.

Runnable as one command:
    python -m alz_agent.eval.harness
Exit code is 0 iff every category passes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from alz_agent.agent.orchestrator import AgentResponse, _check_hallucination, run_agent
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.backend.interface import DataBackend
from alz_agent.eval.golden_questions import GOLDEN_QUESTIONS
from alz_agent.schemas import (
    CatalogEntry,
    FilterSpec,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
)
from alz_agent.tools.deterministic import (
    assess_modality_overlap,
    check_leakage,
    compute_cohort_size,
    lookup_variable,
    suggest_modeling_design,
)

# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CategoryResult:
    category: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)


@dataclass
class EvalReport:
    categories: list[CategoryResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.categories) and all(c.passed for c in self.categories)

    def category(self, name: str) -> CategoryResult:
        for c in self.categories:
            if c.category == name:
                return c
        raise KeyError(f"No category named {name!r} in this report")

    def summary(self) -> str:
        lines = ["=== P8 Eval Harness Report ==="]
        for cat in self.categories:
            status = "PASS" if cat.passed else "FAIL"
            lines.append(f"\n[{status}] {cat.category} ({cat.n_passed}/{len(cat.checks)} checks)")
            for c in cat.checks:
                mark = "ok  " if c.passed else "FAIL"
                detail = f" — {c.detail}" if c.detail else ""
                lines.append(f"  {mark} {c.name}{detail}")
        lines.append(f"\nOVERALL: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default fixture catalog + backend (worked-target scenario; same params as
# test_p5_agent.py / test_p6_cohort_map.py / test_p7_boundary.py)
# ---------------------------------------------------------------------------


def default_catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            variable_id="NACC.NACCUDSD",
            concept="nacc_unified_diagnostic_summary",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCUDSD",
            release_version="nacc70",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            leakage_reasons=["Primary outcome label — defines conversion endpoint"],
        ),
        CatalogEntry(
            variable_id="NACC.CDRSUM",
            concept="cdr_sum_of_boxes",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="CDRSUM",
            release_version="nacc70",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.CIRCULAR],
            leakage_reasons=["CDR Sum of Boxes — collinear with outcome"],
        ),
        CatalogEntry(
            variable_id="ADSP_PHC.Cognition.PHC_MEM",
            concept="phc_memory_composite",
            source_dataset=SourceDataset.ADSP_PHC,
            id_space=IdSpace.A,
            raw_code="PHC_MEM",
            release_version="R3",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.HARMONIZATION_FIT_SCOPE],
            leakage_reasons=["GMM fit on full cohort including outcome visits"],
        ),
        CatalogEntry(
            variable_id="NACC.MMSE30",
            concept="mmse_total_score",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="MMSE30",
            release_version="nacc70",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
        CatalogEntry(
            variable_id="NACC.MRI_VOL",
            concept="mri_brain_volume",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="MRI_VOL",
            release_version="nacc70",
            modality=Modality.MRI,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
        CatalogEntry(
            variable_id="NACC.PET_AMYLOID",
            concept="pet_amyloid_suvr",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="PET_AMYLOID",
            release_version="nacc70",
            modality=Modality.PET,
            leakage_tags=[LeakageTag.NONE],
            leakage_reasons=[],
        ),
    ]


def default_backend() -> DataBackend:
    return build_fixture_backend(
        n_normal=50, n_mci=40, n_dementia=30, n_other=20,
        n_mci_long=25, n_mci_short=8, n_single_visit=15,
        n_suppressed_group=5, n_mci_ad=15, n_dementia_ad=15,
        modality_fractions={
            "cognition": 0.9, "mri": 0.15, "pet": 0.04,
            "fluid_biomarker": 0.05, "vascular": 0.8, "neuropath": 0.0,
        },
        seed=42,
    )


# ---------------------------------------------------------------------------
# Category 1 + 2: golden questions / number traceability
# (run once, shared between both categories)
# ---------------------------------------------------------------------------


def _run_golden_questions(
    catalog: list[CatalogEntry], backend: DataBackend
) -> dict[str, AgentResponse]:
    return {
        gq.id: run_agent(gq.nl_question, catalog, backend, intent=gq.intent)
        for gq in GOLDEN_QUESTIONS
    }


def eval_golden_questions(responses: dict[str, AgentResponse]) -> CategoryResult:
    checks: list[CheckResult] = []
    for gq in GOLDEN_QUESTIONS:
        resp = responses[gq.id]
        if resp.grounded != gq.expected_grounded:
            checks.append(CheckResult(
                name=gq.id, passed=False,
                detail=f"expected grounded={gq.expected_grounded}, got={resp.grounded}",
            ))
            continue

        answer_lower = resp.answer.lower()
        missing = [s for s in gq.must_contain if s.lower() not in answer_lower]
        unexpected = [s for s in gq.must_not_contain if s.lower() in answer_lower]
        if missing or unexpected:
            detail_parts = []
            if missing:
                detail_parts.append(f"missing {missing}")
            if unexpected:
                detail_parts.append(f"unexpectedly contains {unexpected}")
            checks.append(CheckResult(name=gq.id, passed=False, detail="; ".join(detail_parts)))
            continue

        checks.append(CheckResult(name=gq.id, passed=True))
    return CategoryResult(category="golden_questions", checks=checks)


def eval_number_traceability(responses: dict[str, AgentResponse]) -> CategoryResult:
    checks: list[CheckResult] = []
    for gq in GOLDEN_QUESTIONS:
        resp = responses[gq.id]
        # Re-run the hallucination guard with the same extra-whitelist that run_agent used
        # (the effective horizon is a deterministic value, not LLM-generated).
        extra = {str(resp.effective_horizon)}
        passed = _check_hallucination(resp.answer, resp.tool_results, extra=extra)
        checks.append(CheckResult(
            name=gq.id, passed=passed,
            detail="" if passed else "untraceable numeric token found in narrated answer",
        ))
    return CategoryResult(category="number_traceability", checks=checks)


# ---------------------------------------------------------------------------
# Category 3: abstention
# ---------------------------------------------------------------------------


def eval_abstention(catalog: list[CatalogEntry], backend: DataBackend) -> CategoryResult:
    checks: list[CheckResult] = []

    # (a) Absent modality combo: 'neuropath' has no catalog entries for this
    # id_space in the fixture catalog above.
    r = assess_modality_overlap(
        ["neuropath"], FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A),
        backend, catalog,
    )
    checks.append(CheckResult(
        name="absent_modality_combo", passed=not r.grounded,
        detail=r.abstain_reason or "unexpectedly grounded",
    ))

    # (b) Uncovered release: a variable that would only exist in a release we
    # don't have locally (e.g. ADSP-PHC R4) is simply absent from the catalog.
    r = lookup_variable("R4_PLASMA_PTAU217", catalog)
    checks.append(CheckResult(
        name="uncovered_release", passed=not r.grounded,
        detail=r.abstain_reason or "unexpectedly grounded",
    ))

    # (c) Suppressed cells: visit_window=[15,16] + mci -> exactly the
    # n_suppressed_group-sized group (5, below SUPPRESSION_THRESHOLD).
    r = compute_cohort_size(
        FilterSpec(baseline_diagnosis=["mci"], visit_window=(15, 16), id_space=IdSpace.A),
        backend,
    )
    checks.append(CheckResult(
        name="suppressed_cells",
        passed=(not r.grounded) and "suppress" in (r.abstain_reason or "").lower(),
        detail=r.abstain_reason or "unexpectedly grounded",
    ))

    # (d) Unverified / unknown label mapping.
    r = compute_cohort_size(
        FilterSpec(baseline_diagnosis=["not_a_real_diagnosis_xyz"], id_space=IdSpace.A),
        backend,
    )
    checks.append(CheckResult(
        name="unverified_mapping", passed=not r.grounded,
        detail=r.abstain_reason or "unexpectedly grounded",
    ))

    return CategoryResult(category="abstention", checks=checks)


# ---------------------------------------------------------------------------
# Category 4: leakage
# ---------------------------------------------------------------------------

_KNOWN_LEAKY_VARS: dict[str, LeakageTag] = {
    "NACC.NACCUDSD": LeakageTag.DIAGNOSIS_DEFINING,
    "NACC.CDRSUM": LeakageTag.CIRCULAR,
    "ADSP_PHC.Cognition.PHC_MEM": LeakageTag.HARMONIZATION_FIT_SCOPE,
}


def eval_leakage(catalog: list[CatalogEntry], backend: DataBackend) -> CategoryResult:
    checks: list[CheckResult] = []
    spec = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A)

    design_result = suggest_modeling_design(spec, 3, catalog, backend)
    checks.append(CheckResult(
        name="suggest_modeling_design_grounded", passed=design_result.grounded,
        detail="" if design_result.grounded else (design_result.abstain_reason or ""),
    ))

    if design_result.grounded and design_result.value is not None:
        design = design_result.value
        for var_id, tag in _KNOWN_LEAKY_VARS.items():
            in_blacklist = var_id in design.leakage_blacklist
            in_whitelist = var_id in design.feature_whitelist
            checks.append(CheckResult(
                name=f"blacklisted:{var_id}",
                passed=in_blacklist and not in_whitelist,
                detail=f"tag={tag.value}, in_blacklist={in_blacklist}, in_whitelist={in_whitelist}",
            ))

    # Direct check_leakage call, decoupled from suggest_modeling_design.
    leak_result = check_leakage(
        [*_KNOWN_LEAKY_VARS.keys(), "NACC.MMSE30"],
        "conversion to dementia within 3 year(s)",
        catalog,
    )
    flagged_ids = (
        {f.variable_id for f in leak_result.value.flagged} if leak_result.grounded else set()
    )
    checks.append(CheckResult(
        name="check_leakage_flags_all_known_leaky_vars",
        passed=set(_KNOWN_LEAKY_VARS) <= flagged_ids,
        detail=f"flagged={sorted(flagged_ids)}",
    ))
    clean_ids = set(leak_result.value.clean) if leak_result.grounded else set()
    checks.append(CheckResult(
        name="check_leakage_keeps_clean_var_clean",
        passed="NACC.MMSE30" in clean_ids,
        detail=f"clean={sorted(clean_ids)}",
    ))

    return CategoryResult(category="leakage", checks=checks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_eval(
    catalog: list[CatalogEntry] | None = None,
    backend: DataBackend | None = None,
) -> EvalReport:
    """Run all four eval categories and return the combined report."""
    catalog = catalog if catalog is not None else default_catalog()
    backend = backend if backend is not None else default_backend()

    responses = _run_golden_questions(catalog, backend)

    return EvalReport(categories=[
        eval_golden_questions(responses),
        eval_number_traceability(responses),
        eval_abstention(catalog, backend),
        eval_leakage(catalog, backend),
    ])


def main() -> int:
    report = run_eval()
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Plan A/B harness — compare decomposition quality across providers
# ---------------------------------------------------------------------------


@dataclass
class PlanABRow:
    """One provider's plan decomposition for the target question."""

    provider: str
    analysis_type: str | None
    value_variable_label: str | None
    value_column: str | None
    event_label: str | None
    event_filter: dict | None
    visit_anchor: str | None
    statistic: str | None
    candidate_label: str | None
    target_population: list[str] = field(default_factory=list)
    decomposed_correctly: bool = False
    failure_mode: str | None = None
    raw_llm_plan: dict = field(default_factory=dict)
    normalized_plan: dict = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class PlanABResult:
    """Multi-provider comparison for one research question."""

    question: str
    rows: list[PlanABRow] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=== Plan A/B Decomposition Comparison ===", f"Q: {self.question}"]
        for r in self.rows:
            status = "OK  " if r.decomposed_correctly else "FAIL"
            lines.append(
                f"[{status}] {r.provider}\n"
                f"        analysis_type    = {r.analysis_type}\n"
                f"        value_label      = {r.value_variable_label!r}\n"
                f"        value_column     = {r.value_column}\n"
                f"        event_label      = {r.event_label!r}\n"
                f"        event_filter     = {r.event_filter}\n"
                f"        visit_anchor     = {r.visit_anchor}\n"
                f"        statistic        = {r.statistic}\n"
                f"        target_pop       = {r.target_population}\n"
                f"        candidate_label  = {r.candidate_label!r}"
            )
            if r.failure_mode:
                lines.append(f"        failure_mode     = {r.failure_mode}")
            if r.validation_errors:
                for e in r.validation_errors:
                    lines.append(f"        validation_error = {e}")
        overall = all(r.decomposed_correctly for r in self.rows)
        lines.append(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
        return "\n".join(lines)


def run_plan_ab(
    question: str,
    catalog: list[CatalogEntry],
    providers: list[tuple[str, Any]],
    *,
    expected_analysis_type: str | None = None,
    expected_value_column: str | None = None,
    expected_event_filter: dict | None = None,
    expected_visit_anchor: str | None = None,
    reject_mci_default: bool = True,
) -> PlanABResult:
    """Run plan_analysis across multiple providers and compare decomposition.

    Args:
        question: Research question to decompose (same for all providers).
        catalog:  Catalog for variable resolution (same for all providers).
        providers: List of (name, anthropic_client) pairs.
        expected_*: Optional expected values for automatic correctness check.
        reject_mci_default: When True, flags as incorrect if target_population=['mci']
            when the question doesn't specify MCI (catches the old default prior).

    Returns:
        PlanABResult with one row per provider and a summary() method.

    Usage::

        from anthropic import Anthropic
        result = run_plan_ab(
            "Among those diagnosed as AD, what is the average age at first diagnosis?",
            catalog=my_catalog,
            providers=[("claude-haiku", Anthropic())],
            expected_value_column="NACCAGE",
            expected_event_filter={"NACCALZD": 1},
        )
        print(result.summary())
    """
    from alz_agent.agent.intent import _is_compound_event_phrase, _plan_analysis_internal

    result = PlanABResult(question=question)
    for name, client in providers:
        try:
            plan, debug = _plan_analysis_internal(question, client, catalog=catalog)
            vv = plan.value_variable
            ev = plan.event

            # --- Build derived StructuredIntent just to get target_population ---
            from alz_agent.agent.intent import _derive_structured_intent_from_plan
            intent = _derive_structured_intent_from_plan(plan, question, parse_debug=debug)

            row_errors: list[str] = list(debug.validation_errors)
            correct = True
            failure: str | None = None

            # Compound-phrase check
            if vv and vv.researcher_label and _is_compound_event_phrase(vv.researcher_label):
                row_errors.append(
                    f"COMPOUND_PHRASE: value_variable.researcher_label={vv.researcher_label!r}"
                )
                correct = False
                failure = "variable_resolution"

            # Expected values checks
            if expected_analysis_type and plan.analysis_type != expected_analysis_type:
                row_errors.append(
                    f"analysis_type mismatch: got {plan.analysis_type!r}, "
                    f"expected {expected_analysis_type!r}"
                )
                correct = False
                failure = failure or "parsing"
            if expected_value_column and (not vv or vv.selected_raw_code != expected_value_column):
                got = vv.selected_raw_code if vv else None
                row_errors.append(
                    f"value_column mismatch: got {got!r}, expected {expected_value_column!r}"
                )
                correct = False
                failure = failure or "variable_resolution"
            if expected_event_filter and (not ev or ev.filter_expression != expected_event_filter):
                got = ev.filter_expression if ev else None
                row_errors.append(
                    f"event_filter mismatch: got {got!r}, expected {expected_event_filter!r}"
                )
                correct = False
                failure = failure or "variable_resolution"
            if expected_visit_anchor and plan.visit_anchor != expected_visit_anchor:
                row_errors.append(
                    f"visit_anchor mismatch: got {plan.visit_anchor!r}, "
                    f"expected {expected_visit_anchor!r}"
                )
                correct = False
                failure = failure or "parsing"

            # MCI-default check
            if (
                reject_mci_default
                and intent.target_population == ["mci"]
                and "mci" not in question.lower()
            ):
                row_errors.append(
                    "UNWANTED_DEFAULT: target_population defaulted to ['mci'] "
                    "but question does not specify MCI."
                )
                correct = False
                failure = failure or "parsing"

            result.rows.append(PlanABRow(
                provider=name,
                analysis_type=plan.analysis_type,
                value_variable_label=vv.researcher_label if vv else None,
                value_column=vv.selected_raw_code if vv else None,
                event_label=ev.researcher_label if ev else None,
                event_filter=ev.filter_expression if ev else None,
                visit_anchor=plan.visit_anchor,
                statistic=plan.statistic,
                candidate_label=plan.candidate_label,
                target_population=list(intent.target_population),
                decomposed_correctly=correct,
                failure_mode=failure,
                raw_llm_plan=dict(debug.raw_llm_plan_json),
                normalized_plan=dict(debug.normalized_plan),
                validation_errors=row_errors,
            ))

        except Exception as exc:
            result.rows.append(PlanABRow(
                provider=name,
                analysis_type=None,
                value_variable_label=None,
                value_column=None,
                event_label=None,
                event_filter=None,
                visit_anchor=None,
                statistic=None,
                candidate_label=None,
                decomposed_correctly=False,
                failure_mode="parsing",
                validation_errors=[f"EXCEPTION: {exc!s}"],
            ))
    return result
