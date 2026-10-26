"""End-to-end validation of the STATISTICAL_ANALYSIS and AGGREGATE_STAT pathways.

Tests three researcher-style questions and reports:
  - parsed intent (question_type, group_variable, outcome_variable, etc.)
  - variables/concepts resolved (via lookup_variable)
  - backend/analysis function called
  - fixture vs real local data
  - grounded / abstained
  - hallucination guard result
  - failure reason if abstained

Run:
    python scripts/validate_pathways.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import yaml

from alz_agent.agent.intent import parse_intent
from alz_agent.agent.orchestrator import run_agent
from alz_agent.analysis.engine import LocalAnalysisEngine
from alz_agent.backend.local_backend import LocalDataBackend
from alz_agent.schemas import (
    CatalogEntry,
    GroupComparisonResult,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
)
from alz_agent.tools.deterministic import lookup_variable

_SEP = "=" * 72


# ---------------------------------------------------------------------------
# Test catalog with proper human-readable concepts for the 4 key variables
# ---------------------------------------------------------------------------


def _make_analysis_catalog() -> list[CatalogEntry]:
    """Minimal catalog with properly-named entries for the 3 test questions.

    Concepts are set so lookup_variable() can find them by natural-language
    queries even without RDD enrichment (which takes 72s to parse).
    """
    return [
        # ── NACCAGE — used by Q2 (average age at first diagnosis) and Q3 (outcome) ──
        CatalogEntry(
            variable_id="NACC.NACCAGE",
            concept="age at visit",
            description="participant age in years at the time of the UDS visit",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCAGE",
            release_version="nacc70",
            modality=Modality.DEMOGRAPHIC,
            leakage_tags=[LeakageTag.NONE],
        ),
        # ── NACCMMSE — used by Q1 (MMSE outcome in group comparison) ──
        CatalogEntry(
            variable_id="NACC.NACCMMSE",
            concept="mmse total score",
            description=(
                "Mini-Mental State Examination total score (NACCMMSE). "
                "Range 0-30. Higher = better cognition."
            ),
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCMMSE",
            release_version="nacc70",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.NONE],
        ),
        # ── CVHATT — used by Q1 (heart attack history group variable) ──
        CatalogEntry(
            variable_id="NACC.CVHATT",
            concept="heart attack history",
            description=(
                "Cardiovascular history: heart attack / myocardial infarction (CVHATT). "
                "0=Absent, 1=Recent/Active, 2=Remote/inactive, 9=Unknown"
            ),
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="CVHATT",
            release_version="nacc70",
            modality=Modality.VASCULAR,
            leakage_tags=[LeakageTag.NONE],
            value_coding={
                "0": "Absent",
                "1": "Recent/Active",
                "2": "Remote/inactive",
                "9": "Unknown",
            },
        ),
        # ── NACCNMRI — used by Q3 (T1 MRI data presence as group variable) ──
        CatalogEntry(
            variable_id="NACC.NACCNMRI",
            concept="T1 MRI data available",
            description=(
                "Number of T1 MRI scans collected at this visit (NACCNMRI). "
                "0=No MRI data, >=1 = MRI data present. Binary group: 0 vs any."
            ),
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCNMRI",
            release_version="nacc70",
            modality=Modality.MRI,
            leakage_tags=[LeakageTag.NONE],
            value_coding={"0": "No MRI data"},
        ),
    ]


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _field(label: str, value) -> None:
    print(f"  {label:<30}: {value}")


def _indent(text: str, width: int = 4) -> None:
    for line in str(text).splitlines():
        print(" " * width + line)


def _check_lookup(label: str, query: str, catalog: list[CatalogEntry]) -> str | None:
    """Return raw_code if found, print result, return None if not found."""
    result = lookup_variable(query, catalog)
    if result.grounded:
        _field(f"  {label}", f"FOUND  {result.value.variable_id}  raw_code={result.value.raw_code}")
        return result.value.raw_code
    else:
        _field(f"  {label}", f"NOT FOUND — {result.abstain_reason}")
        return None


# ---------------------------------------------------------------------------
# Load real backend
# ---------------------------------------------------------------------------


def _load_local_backend():
    config_path = _ROOT / "config" / "paths.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    backend = LocalDataBackend(config, _ROOT)
    engine = LocalAnalysisEngine(backend)
    return backend, engine


# ---------------------------------------------------------------------------
# Question 1: heart attack vs MMSE (group comparison, MCI)
# ---------------------------------------------------------------------------

Q1 = (
    "Among participants with MCI at baseline, do people with heart attack history "
    "have lower baseline MMSE than those without heart attack? "
    "Report group means, p-value, and interpretation."
)


def run_q1(backend, engine, catalog):
    _banner("Q1 — Group comparison: MCI, heart attack history vs baseline MMSE")
    print(f"  Question: {textwrap.fill(Q1, 65, subsequent_indent='  ')}")
    print()

    # 1. Intent
    intent = parse_intent(Q1)
    _field("question_type", intent.question_type.value)
    _field("target_population", intent.target_population)
    _field("target_etiology", intent.target_etiology)
    _field("analysis_type", intent.analysis_type)
    _field("group_variable (heuristic)", intent.group_variable)
    _field("outcome_variable (heuristic)", intent.outcome_variable)
    print()

    # 2. Variable resolution
    print("  Variable resolution:")
    _check_lookup("group_variable", intent.group_variable or "heart attack history", catalog)
    _check_lookup("outcome_variable", intent.outcome_variable or "MMSE", catalog)
    print()

    # 3. Full pipeline
    print("  Running run_agent (local backend + LocalAnalysisEngine) ...")
    resp = run_agent(Q1, catalog, backend, analysis_engine=engine)
    print()

    _field("grounded", resp.grounded)
    _field("hallucination_guard", resp.hallucination_check_passed)
    if resp.abstain_reason:
        _field("abstain_reason", resp.abstain_reason)
    _field("tool_results", len(resp.tool_results))

    if resp.grounded and resp.tool_results:
        tr = resp.tool_results[0]
        if isinstance(tr.value, GroupComparisonResult):
            v = tr.value
            print(f"\n  GroupComparisonResult:")
            _field("    outcome", f"{v.outcome_label} ({v.outcome_variable_id})")
            _field("    group_variable", f"{v.group_label} ({v.group_variable_id})")
            _field("    n_total", v.n_total)
            for g in v.groups:
                _field(f"    {g.group_label}", f"n={g.n}  mean={g.mean}  SD={g.sd}")
            _field("    p_value", v.p_value)
            _field("    test", v.test_name)
            if v.caveats:
                print("    Caveats:")
                for c in v.caveats:
                    print(f"      - {textwrap.fill(c, 60, subsequent_indent='        ')}")

    print()
    print("  Template answer:")
    _indent(resp.answer)


# ---------------------------------------------------------------------------
# Question 2: average age at first AD diagnosis
# ---------------------------------------------------------------------------

Q2 = "Among participants diagnosed with AD, what is the average age when they were first diagnosed?"


def run_q2(backend, engine, catalog):
    _banner("Q2 — Aggregate stat: mean age at first AD diagnosis")
    print(f"  Question: {Q2}")
    print()

    # 1. Intent
    intent = parse_intent(Q2)
    _field("question_type", intent.question_type.value)
    _field("target_population", intent.target_population)
    _field("target_etiology", intent.target_etiology)
    _field("stat_variable", intent.stat_variable)
    _field("stat_type", intent.stat_type)
    _field("event_anchor", intent.event_anchor)
    print()

    # 2. Variable resolution
    print("  Variable resolution:")
    _check_lookup("stat_variable", intent.stat_variable or "age", catalog)
    print()

    # 3. Full pipeline
    print("  Running run_agent (local backend) ...")
    resp = run_agent(Q2, catalog, backend)
    print()

    _field("grounded", resp.grounded)
    _field("hallucination_guard", resp.hallucination_check_passed)
    if resp.abstain_reason:
        _field("abstain_reason", resp.abstain_reason)

    print()
    print("  Template answer:")
    _indent(resp.answer)


# ---------------------------------------------------------------------------
# Question 3: age by T1 MRI presence (MCI)
# ---------------------------------------------------------------------------

Q3 = (
    "Among baseline MCI participants, does baseline age differ between "
    "those with and without any T1 MRI data?"
)


def run_q3(backend, engine, catalog):
    _banner("Q3 — Group comparison: MCI, age difference by T1 MRI availability")
    print(f"  Question: {Q3}")
    print()

    # 1. Intent
    intent = parse_intent(Q3)
    _field("question_type", intent.question_type.value)
    _field("target_population", intent.target_population)
    _field("target_etiology", intent.target_etiology)
    _field("analysis_type", intent.analysis_type)
    _field("group_variable (heuristic)", intent.group_variable)
    _field("outcome_variable (heuristic)", intent.outcome_variable)
    print()

    # 2. Variable resolution
    print("  Variable resolution:")
    _check_lookup("group_variable", intent.group_variable or "T1 MRI data", catalog)
    _check_lookup("outcome_variable", intent.outcome_variable or "age", catalog)
    print()

    # 3. Full pipeline
    print("  Running run_agent (local backend + LocalAnalysisEngine) ...")
    resp = run_agent(Q3, catalog, backend, analysis_engine=engine)
    print()

    _field("grounded", resp.grounded)
    _field("hallucination_guard", resp.hallucination_check_passed)
    if resp.abstain_reason:
        _field("abstain_reason", resp.abstain_reason)
    _field("tool_results", len(resp.tool_results))

    if resp.grounded and resp.tool_results:
        tr = resp.tool_results[0]
        if isinstance(tr.value, GroupComparisonResult):
            v = tr.value
            print(f"\n  GroupComparisonResult:")
            _field("    outcome", f"{v.outcome_label} ({v.outcome_variable_id})")
            _field("    group_variable", f"{v.group_label} ({v.group_variable_id})")
            _field("    n_total", v.n_total)
            for g in v.groups:
                _field(f"    {g.group_label}", f"n={g.n}  mean={g.mean}  SD={g.sd}")
            _field("    p_value", v.p_value)
            _field("    test", v.test_name)
            if v.caveats:
                print("    Caveats:")
                for c in v.caveats:
                    print(f"      - {textwrap.fill(c, 60, subsequent_indent='        ')}")

    print()
    print("  Template answer:")
    _indent(resp.answer)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def run_summary(results: dict[str, dict]) -> None:
    _banner("VALIDATION SUMMARY")
    header = f"{'Q':<5} {'IntentOK':<10} {'VarsOK':<10} {'Grounded':<10} {'HallGuard':<12} {'Backend':<8} {'Note'}"
    print(f"  {header}")
    print("  " + "-" * 85)
    for qid, r in results.items():
        row = (
            f"{qid:<5} "
            f"{'YES' if r['intent_ok'] else 'NO':<10} "
            f"{'YES' if r['vars_ok'] else 'NO':<10} "
            f"{'YES' if r['grounded'] else 'NO':<10} "
            f"{'YES' if r['hallguard'] else 'NO':<12} "
            f"{'local':<8} "
            f"{r.get('note', '')}"
        )
        print(f"  {row}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("\nalz_dataset_agent — New pathway validation")
    print("Loading local backend...", end=" ", flush=True)
    backend, engine = _load_local_backend()
    print("done.")
    print("Building analysis catalog (4 variables)...", end=" ", flush=True)
    catalog = _make_analysis_catalog()
    print(f"done ({len(catalog)} entries).")

    results: dict[str, dict] = {}

    # ── Q1 ──────────────────────────────────────────────────────────────────
    try:
        run_q1(backend, engine, catalog)
        intent = parse_intent(Q1)
        resp = run_agent(Q1, catalog, backend, analysis_engine=engine)
        results["Q1"] = {
            "intent_ok": intent.question_type.value == "statistical_analysis",
            "vars_ok": intent.group_variable is not None and intent.outcome_variable is not None,
            "grounded": resp.grounded,
            "hallguard": resp.hallucination_check_passed,
            "note": resp.abstain_reason or ("group_comparison completed" if resp.grounded else ""),
        }
    except Exception as exc:
        results["Q1"] = {"intent_ok": False, "vars_ok": False, "grounded": False,
                         "hallguard": False, "note": f"ERROR: {exc}"}

    # ── Q2 ──────────────────────────────────────────────────────────────────
    try:
        run_q2(backend, engine, catalog)
        intent = parse_intent(Q2)
        resp = run_agent(Q2, catalog, backend)
        results["Q2"] = {
            "intent_ok": intent.question_type.value == "aggregate_stat",
            "vars_ok": intent.stat_variable is not None,
            "grounded": resp.grounded,
            "hallguard": resp.hallucination_check_passed,
            "note": resp.abstain_reason or ("aggregate_stat completed" if resp.grounded else ""),
        }
    except Exception as exc:
        results["Q2"] = {"intent_ok": False, "vars_ok": False, "grounded": False,
                         "hallguard": False, "note": f"ERROR: {exc}"}

    # ── Q3 ──────────────────────────────────────────────────────────────────
    try:
        run_q3(backend, engine, catalog)
        intent = parse_intent(Q3)
        resp = run_agent(Q3, catalog, backend, analysis_engine=engine)
        results["Q3"] = {
            "intent_ok": intent.question_type.value == "statistical_analysis",
            "vars_ok": intent.group_variable is not None,
            "grounded": resp.grounded,
            "hallguard": resp.hallucination_check_passed,
            "note": resp.abstain_reason or ("group_comparison completed" if resp.grounded else ""),
        }
    except Exception as exc:
        results["Q3"] = {"intent_ok": False, "vars_ok": False, "grounded": False,
                         "hallguard": False, "note": f"ERROR: {exc}"}

    run_summary(results)
