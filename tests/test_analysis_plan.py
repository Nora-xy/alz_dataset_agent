"""Tests for the LLM Planner layer: plan_analysis + orchestrator _run_plan.

Test cases A–F per specification:
  A. Event-anchored stat — 'average age at first AD diagnosis'
       → value_variable=NACCAGE, event.filter={NACCALZD:1}, anchor=first_diagnosis
  B. Unsupported analysis_type → clean abstain message ('not yet implemented')
  C. AD vs dementia semantic distinction (both directions)
  D. Leakage assessment per role (each role gets correct severity)
  E. Invalid column → validation rejects it (not_in_catalog)
  F. Row identifier blocked in all roles

All LLM responses are mocked via unittest.mock — no API key required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import pydantic

from alz_agent.agent.intent import (
    ParseDebugInfo,
    _derive_structured_intent_from_plan,
    _is_compound_event_phrase,
    _plan_analysis_internal,
    _validate_analysis_plan,
    plan_analysis,
)
from alz_agent.agent.orchestrator import (
    _build_filter_spec_from_plan,
    _plan_aggregate_stat,
    _plan_group_comparison,
    _run_plan,
)
from alz_agent.eval.harness import PlanABResult, run_plan_ab
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    AnalysisPlan,
    AnalysisRole,
    CatalogEntry,
    FilterSpec,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Shared catalog fixture
# ---------------------------------------------------------------------------

def _make_catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            variable_id="NACC.NACCUDSD",
            concept="cognitive status",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCUDSD",
            release_version="nacc70",
            description="UDS consensus diagnosis: 1=Normal 3=MCI 4=Dementia",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            value_coding={"1": "Normal", "3": "MCI", "4": "Dementia"},
        ),
        CatalogEntry(
            variable_id="NACC.NACCALZD",
            concept="etiologic ad diagnosis",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCALZD",
            release_version="nacc70",
            description="Etiologic AD: 1=Primary",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            value_coding={"1": "Primary", "2": "Contributing"},
        ),
        CatalogEntry(
            variable_id="NACC.NACCAGE",
            concept="age at visit",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCAGE",
            release_version="nacc70",
            description="Age of subject at time of UDS visit",
            modality=Modality.DEMOGRAPHIC,
            leakage_tags=[LeakageTag.NONE],
            value_coding={},
        ),
        CatalogEntry(
            variable_id="NACC.NACCMMSE",
            concept="mmse total score",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCMMSE",
            release_version="nacc70",
            description="Mini-Mental State Exam total score (0–30)",
            modality=Modality.COGNITION,
            leakage_tags=[LeakageTag.NONE],
            value_coding={},
        ),
        CatalogEntry(
            variable_id="NACC.CVHATT",
            concept="heart attack history",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="CVHATT",
            release_version="nacc70",
            description="History of heart attack / myocardial infarction",
            modality=Modality.VASCULAR,
            leakage_tags=[LeakageTag.NONE],
            value_coding={"0": "Absent", "1": "Recent/Active", "2": "Remote/inactive"},
        ),
        CatalogEntry(
            variable_id="NACC.NACCID",
            concept="participant id",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCID",
            release_version="nacc70",
            description="Unique participant identifier",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.ROW_IDENTIFIER],
            value_coding={},
        ),
    ]


@pytest.fixture()
def catalog() -> list[CatalogEntry]:
    return _make_catalog()


def _make_plan_client(tool_input: dict[str, Any]) -> MagicMock:
    """Mock client returning a single plan_analysis tool-use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "plan_analysis"
    block.input = tool_input

    response = MagicMock()
    response.content = [block]

    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Case A: event-anchored stat — 'average age at first AD diagnosis'
# ---------------------------------------------------------------------------

class TestCaseAEventAnchoredStat:
    """The key fix: 'age at first AD diagnosis' is decomposed into
    value_variable=NACCAGE + event.filter={'NACCALZD':1} + anchor.
    Never sent as a compound string to lookup_variable.
    """

    def _plan_input(self) -> dict[str, Any]:
        return {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "Mean participant age at their first AD diagnosis visit",
            "requires_longitudinal_data": True,
            "statistic": "mean",
            "visit_anchor": "first_matching_event_visit",
            "candidate_label": "etiologic AD diagnosis",
            "id_space": "A",
            "value_variable": {
                "role": "value_variable",
                "researcher_label": "age",
                "semantic_definition": "Participant age at the first AD diagnosis visit",
                "selected_raw_code": "NACCAGE",
                "filter_expression": None,
                "resolution_status": "resolved",
            },
            "event": {
                "role": "event",
                "researcher_label": "first AD diagnosis",
                "semantic_definition": "First visit where NACCALZD==1",
                "selected_raw_code": "NACCALZD",
                "filter_expression": {"NACCALZD": 1},
                "resolution_status": "resolved",
            },
            "covariates": [],
            "unresolved_items": [],
        }

    def test_plan_analysis_decomposes_compound_phrase(self, catalog):
        client = _make_plan_client(self._plan_input())
        plan = plan_analysis("average age at first AD diagnosis", client, catalog=catalog)
        assert plan.analysis_type == "event_anchored_stat"
        assert plan.value_variable is not None
        assert plan.value_variable.selected_raw_code == "NACCAGE"

    def test_event_column_is_naccalzd(self, catalog):
        client = _make_plan_client(self._plan_input())
        plan = plan_analysis("average age at first AD diagnosis", client, catalog=catalog)
        assert plan.event is not None
        assert plan.event.selected_raw_code == "NACCALZD"
        assert plan.event.filter_expression == {"NACCALZD": 1}

    def test_visit_anchor_maps_to_first_diagnosis(self, catalog):
        client = _make_plan_client(self._plan_input())
        plan = plan_analysis("...", client, catalog=catalog)
        assert plan.visit_anchor == "first_matching_event_visit"

    def test_naccalzd_leakage_finding_event_definition_role(self, catalog):
        client = _make_plan_client(self._plan_input())
        plan = plan_analysis("...", client, catalog=catalog)
        alz_findings = [f for f in plan.leakage_assessment if f.raw_code == "NACCALZD"]
        assert alz_findings, "NACCALZD must appear in leakage_assessment"
        assert all(f.role == "event_definition" for f in alz_findings)
        assert all(f.severity != "block" for f in alz_findings), (
            "NACCALZD as event anchor must not be blocked"
        )

    def test_filter_spec_from_plan_sets_etiology(self, catalog):
        # Event NACCALZD==1 → event_etiology (any-visit), NOT baseline_etiology (first-visit).
        # This ensures incident AD cases (NACCALZD first appears after visit 1) are included.
        client = _make_plan_client(self._plan_input())
        plan = plan_analysis("...", client, catalog=catalog)
        fs = _build_filter_spec_from_plan(plan, IdSpace.A)
        assert fs.event_etiology == ["ad"]
        assert fs.baseline_etiology is None

    def test_plan_aggregate_stat_returns_tool_result(self, catalog):
        """_plan_aggregate_stat should return a ToolResult (grounded or abstain)."""
        client = _make_plan_client(self._plan_input())
        plan = plan_analysis("...", client, catalog=catalog)

        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        fs = _build_filter_spec_from_plan(plan, IdSpace.A)
        results = _plan_aggregate_stat(plan, catalog, backend, fs)
        assert len(results) == 1
        assert isinstance(results[0], ToolResult)


# ---------------------------------------------------------------------------
# Case B: unsupported analysis type → clean abstain
# ---------------------------------------------------------------------------

class TestCaseBUnsupportedAnalysisType:
    """When plan.analysis_type is not in _PLAN_SUPPORTED_TYPES,
    _run_plan returns a clean abstain (not a KeyError or AttributeError).
    """

    def test_survival_analysis_abstains_with_not_implemented_message(self, catalog):
        plan = AnalysisPlan(
            original_question="Kaplan-Meier survival to AD",
            analysis_type="survival_analysis",  # not supported
            research_intent_summary="Kaplan-Meier survival curve to first AD diagnosis",
            requires_longitudinal_data=True,
        )

        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        fs = FilterSpec(id_space=IdSpace.A)
        results = _run_plan(plan, catalog, backend, fs)
        assert len(results) == 1
        assert not results[0].grounded
        assert "not yet implemented" in (results[0].abstain_reason or "").lower()

    def test_abstain_reason_names_analysis_type(self, catalog):
        plan = AnalysisPlan(
            original_question="...",
            analysis_type="causal_inference",
            research_intent_summary="...",
            requires_longitudinal_data=False,
        )
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        fs = FilterSpec(id_space=IdSpace.A)
        results = _run_plan(plan, catalog, backend, fs)
        assert "causal_inference" in (results[0].abstain_reason or "")

    def test_supported_types_do_not_hit_this_fallback(self, catalog):
        """Smoke-test: no supported type should return 'not yet implemented'."""
        from alz_agent.agent.orchestrator import _PLAN_SUPPORTED_TYPES
        assert "event_anchored_stat" in _PLAN_SUPPORTED_TYPES
        assert "group_comparison" in _PLAN_SUPPORTED_TYPES
        assert "regression" in _PLAN_SUPPORTED_TYPES


# ---------------------------------------------------------------------------
# Case C: AD vs dementia semantic distinction
# ---------------------------------------------------------------------------

class TestCaseCAdVsDementia:
    """'AD' and 'dementia' MUST map to different filter expressions.
    NACCALZD==1 is etiologic AD; NACCUDSD==4 is all-cause dementia.
    Never substitute one for the other.
    """

    def test_ad_question_maps_event_to_naccalzd(self, catalog):
        plan_input = {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "Age at first AD diagnosis",
            "requires_longitudinal_data": True,
            "statistic": "mean",
            "visit_anchor": "first_matching_event_visit",
            "event": {
                "role": "event", "researcher_label": "AD diagnosis",
                "semantic_definition": "Etiologic AD diagnosis (NACCALZD==1)",
                "selected_raw_code": "NACCALZD",
                "filter_expression": {"NACCALZD": 1},
                "resolution_status": "resolved",
            },
            "value_variable": {
                "role": "value_variable", "researcher_label": "age",
                "semantic_definition": "Age at the AD diagnosis visit",
                "selected_raw_code": "NACCAGE", "resolution_status": "resolved",
            },
            "unresolved_items": [],
        }
        client = _make_plan_client(plan_input)
        plan = plan_analysis("average age at first AD diagnosis", client, catalog=catalog)
        assert plan.event is not None
        assert plan.event.selected_raw_code == "NACCALZD"
        assert plan.event.filter_expression == {"NACCALZD": 1}
        # Must NOT use NACCUDSD for AD
        assert plan.event.selected_raw_code != "NACCUDSD"

    def test_dementia_question_maps_event_to_naccudsd(self, catalog):
        plan_input = {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "Age at first dementia diagnosis",
            "requires_longitudinal_data": True,
            "statistic": "mean",
            "visit_anchor": "first_matching_event_visit",
            "event": {
                "role": "event", "researcher_label": "dementia diagnosis",
                "semantic_definition": "All-cause dementia (NACCUDSD==4)",
                "selected_raw_code": "NACCUDSD",
                "filter_expression": {"NACCUDSD": 4},
                "resolution_status": "resolved",
            },
            "value_variable": {
                "role": "value_variable", "researcher_label": "age",
                "semantic_definition": "Age at the dementia diagnosis visit",
                "selected_raw_code": "NACCAGE", "resolution_status": "resolved",
            },
            "unresolved_items": [],
        }
        client = _make_plan_client(plan_input)
        plan = plan_analysis("average age at first dementia diagnosis", client, catalog=catalog)
        assert plan.event is not None
        assert plan.event.selected_raw_code == "NACCUDSD"
        assert plan.event.filter_expression == {"NACCUDSD": 4}
        # Must NOT use NACCALZD for all-cause dementia
        assert plan.event.selected_raw_code != "NACCALZD"

    def test_filter_spec_from_ad_plan_sets_etiology_not_diagnosis(self, catalog):
        client = _make_plan_client({
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "...",
            "requires_longitudinal_data": True,
            "event": {
                "role": "event", "researcher_label": "AD",
                "semantic_definition": "NACCALZD==1",
                "selected_raw_code": "NACCALZD",
                "filter_expression": {"NACCALZD": 1},
                "resolution_status": "resolved",
            },
            "value_variable": {
                "role": "value_variable", "researcher_label": "age",
                "semantic_definition": "...", "selected_raw_code": "NACCAGE",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        })
        plan = plan_analysis("...", client, catalog=catalog)
        fs = _build_filter_spec_from_plan(plan, IdSpace.A)
        # Event NACCALZD==1 → event_etiology (any-visit), not baseline_etiology (first-visit).
        assert fs.event_etiology == ["ad"]
        assert fs.baseline_etiology is None
        # baseline_diagnosis should NOT be set from the AD event
        assert fs.baseline_diagnosis is None

    def test_filter_spec_from_dementia_plan_sets_diagnosis_not_etiology(self, catalog):
        client = _make_plan_client({
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "...",
            "requires_longitudinal_data": True,
            "event": {
                "role": "event", "researcher_label": "dementia",
                "semantic_definition": "NACCUDSD==4",
                "selected_raw_code": "NACCUDSD",
                "filter_expression": {"NACCUDSD": 4},
                "resolution_status": "resolved",
            },
            "value_variable": {
                "role": "value_variable", "researcher_label": "age",
                "semantic_definition": "...", "selected_raw_code": "NACCAGE",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        })
        plan = plan_analysis("...", client, catalog=catalog)
        fs = _build_filter_spec_from_plan(plan, IdSpace.A)
        assert fs.baseline_diagnosis == ["dementia"]
        assert fs.baseline_etiology is None


# ---------------------------------------------------------------------------
# Case D: leakage assessment per role
# ---------------------------------------------------------------------------

class TestCaseDLeakagePerRole:
    """Each plan role should produce the correct leakage severity."""

    def _validate(self, plan_data: dict, catalog: list) -> dict:
        return _validate_analysis_plan(plan_data, catalog)

    def test_naccalzd_as_event_is_allowed(self, catalog):
        plan_data = {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "...",
            "requires_longitudinal_data": True,
            "event": {
                "role": "event", "researcher_label": "AD",
                "semantic_definition": "...", "selected_raw_code": "NACCALZD",
                "filter_expression": {"NACCALZD": 1}, "resolution_status": "resolved",
            },
            "unresolved_items": [],
        }
        result = self._validate(plan_data, catalog)
        alz_findings = [f for f in result["leakage_assessment"] if f["raw_code"] == "NACCALZD"]
        assert alz_findings
        assert alz_findings[0]["role"] == "event_definition"
        assert alz_findings[0]["severity"] != "block"

    def test_naccudsd_as_cohort_filter_is_allow_with_caveat(self, catalog):
        plan_data = {
            "analysis_type": "cohort_size",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "cohort": {
                "role": "cohort", "researcher_label": "MCI",
                "semantic_definition": "...", "selected_raw_code": "NACCUDSD",
                "filter_expression": {"NACCUDSD": 3}, "resolution_status": "resolved",
            },
            "unresolved_items": [],
        }
        result = self._validate(plan_data, catalog)
        udsd_findings = [f for f in result["leakage_assessment"] if f["raw_code"] == "NACCUDSD"]
        assert udsd_findings
        assert udsd_findings[0]["role"] == "cohort_filter"
        assert udsd_findings[0]["severity"] == "allow_with_caveat"

    def test_naccudsd_as_covariate_is_blocked(self, catalog):
        plan_data = {
            "analysis_type": "regression",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "covariates": [{
                "role": "covariate", "researcher_label": "cognitive status",
                "semantic_definition": "...", "selected_raw_code": "NACCUDSD",
                "resolution_status": "resolved",
            }],
            "unresolved_items": [],
        }
        result = self._validate(plan_data, catalog)
        udsd_findings = [f for f in result["leakage_assessment"] if f["raw_code"] == "NACCUDSD"]
        assert udsd_findings
        assert udsd_findings[0]["severity"] == "block"
        assert udsd_findings[0]["role"] == "covariate"
        # Column should be removed from covariates
        assert all(c.get("selected_raw_code") != "NACCUDSD" for c in result["covariates"])

    def test_naccage_as_value_variable_is_allowed(self, catalog):
        plan_data = {
            "analysis_type": "aggregate_stat",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "value_variable": {
                "role": "value_variable", "researcher_label": "age",
                "semantic_definition": "...", "selected_raw_code": "NACCAGE",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        }
        result = self._validate(plan_data, catalog)
        age_findings = [f for f in result["leakage_assessment"] if f["raw_code"] == "NACCAGE"]
        assert age_findings
        assert age_findings[0]["severity"] == "allowed"
        assert result["value_variable"]["selected_raw_code"] == "NACCAGE"


# ---------------------------------------------------------------------------
# Case E: invalid column rejected (not in catalog)
# ---------------------------------------------------------------------------

class TestCaseEInvalidColumnRejected:
    """LLM returns a column that doesn't exist in the catalog.
    _validate_analysis_plan must reject it and populate unresolved_items.
    """

    def test_nonexistent_value_variable_rejected(self, catalog):
        plan_data = {
            "analysis_type": "aggregate_stat",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "value_variable": {
                "role": "value_variable", "researcher_label": "invented metric",
                "semantic_definition": "...", "selected_raw_code": "NONEXISTENT_XYZ",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        }
        result = _validate_analysis_plan(plan_data, catalog)
        assert result["value_variable"]["selected_raw_code"] is None
        assert result["value_variable"]["validation_status"] == "invalid_not_in_catalog"
        assert any("NONEXISTENT_XYZ" in f for f in result["unresolved_items"])

    def test_nonexistent_group_variable_rejected(self, catalog):
        plan_data = {
            "analysis_type": "group_comparison",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "group_variable": {
                "role": "group_variable", "researcher_label": "mystery factor",
                "semantic_definition": "...", "selected_raw_code": "GHOST_COL",
                "resolution_status": "resolved",
            },
            "outcome_variable": {
                "role": "outcome_variable", "researcher_label": "MMSE",
                "semantic_definition": "...", "selected_raw_code": "NACCMMSE",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        }
        result = _validate_analysis_plan(plan_data, catalog)
        assert result["group_variable"]["selected_raw_code"] is None
        assert result["outcome_variable"]["selected_raw_code"] == "NACCMMSE"

    def test_run_plan_abstains_when_value_variable_missing(self, catalog):
        plan = AnalysisPlan(
            original_question="...",
            analysis_type="event_anchored_stat",
            research_intent_summary="...",
            requires_longitudinal_data=True,
            # value_variable is None — validation removed it
            value_variable=None,
            event=AnalysisRole(
                role="event", researcher_label="AD",
                semantic_definition="...", selected_raw_code="NACCALZD",
                filter_expression={"NACCALZD": 1}, resolution_status="resolved",
            ),
        )
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        fs = FilterSpec(id_space=IdSpace.A)
        results = _plan_aggregate_stat(plan, catalog, backend, fs)
        assert len(results) == 1
        assert not results[0].grounded
        assert "value_variable" in (results[0].abstain_reason or "").lower()


# ---------------------------------------------------------------------------
# Case F: row identifier blocked in all roles
# ---------------------------------------------------------------------------

class TestCaseFRowIdentifierBlocked:
    """NACCID has LeakageTag.ROW_IDENTIFIER.
    It must be blocked regardless of which role it appears in.
    """

    def _base(self, field: str, role: str) -> dict:
        return {
            "analysis_type": "group_comparison",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            field: {
                "role": role,
                "researcher_label": "participant ID",
                "semantic_definition": "Row-identifying variable",
                "selected_raw_code": "NACCID",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        }

    def test_naccid_blocked_as_outcome(self, catalog):
        result = _validate_analysis_plan(self._base("outcome_variable", "outcome_variable"), catalog)
        assert result["outcome_variable"]["selected_raw_code"] is None
        assert result["outcome_variable"]["validation_status"] == "invalid_blocked"

    def test_naccid_blocked_as_group_variable(self, catalog):
        result = _validate_analysis_plan(self._base("group_variable", "group_variable"), catalog)
        assert result["group_variable"]["selected_raw_code"] is None
        assert result["group_variable"]["validation_status"] == "invalid_blocked"

    def test_naccid_blocked_as_value_variable(self, catalog):
        result = _validate_analysis_plan(self._base("value_variable", "value_variable"), catalog)
        assert result["value_variable"]["selected_raw_code"] is None
        assert result["value_variable"]["validation_status"] == "invalid_blocked"

    def test_naccid_leakage_finding_severity_is_block(self, catalog):
        result = _validate_analysis_plan(self._base("value_variable", "value_variable"), catalog)
        id_findings = [f for f in result["leakage_assessment"] if f["raw_code"] == "NACCID"]
        assert id_findings
        assert id_findings[0]["severity"] == "block"
        assert id_findings[0]["leakage_tag"] == "ROW_IDENTIFIER"
        assert id_findings[0]["action_taken"] == "caused_abstain"

    def test_naccid_blocked_as_covariate(self, catalog):
        plan_data = {
            "analysis_type": "regression",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "covariates": [{
                "role": "covariate", "researcher_label": "participant ID",
                "semantic_definition": "...", "selected_raw_code": "NACCID",
                "resolution_status": "resolved",
            }],
            "unresolved_items": [],
        }
        result = _validate_analysis_plan(plan_data, catalog)
        assert all(c.get("selected_raw_code") != "NACCID" for c in result["covariates"])
        assert any("NACCID" in f for f in result["unresolved_items"])

    def test_run_plan_group_comparison_abstains_when_group_is_row_id(self, catalog):
        """_plan_group_comparison must abstain cleanly when group_variable is rejected."""
        plan = AnalysisPlan(
            original_question="...",
            analysis_type="group_comparison",
            research_intent_summary="...",
            requires_longitudinal_data=False,
            group_variable=None,  # NACCID was blocked → removed
            outcome_variable=AnalysisRole(
                role="outcome_variable", researcher_label="MMSE",
                semantic_definition="...", selected_raw_code="NACCMMSE",
                resolution_status="resolved",
            ),
        )
        fs = FilterSpec(id_space=IdSpace.A)
        results = _plan_group_comparison(plan, catalog, fs, analysis_engine=MagicMock())
        assert len(results) == 1
        assert not results[0].grounded
        assert "group_variable" in (results[0].abstain_reason or "").lower()


# ---------------------------------------------------------------------------
# Case G: compound-phrase validator (value_variable_label)
# ---------------------------------------------------------------------------

class TestCaseGCompoundPhraseValidator:
    """StructuredIntent.value_variable_label rejects compound event-anchor phrases.

    This makes 'age at first AD diagnosis' structurally impossible as a variable
    field — the validator fires before any catalog lookup is attempted.
    """

    def test_is_compound_event_phrase_detects_phrases(self):
        assert _is_compound_event_phrase("age at first AD diagnosis")
        assert _is_compound_event_phrase("MMSE when diagnosed")
        assert _is_compound_event_phrase("age at conversion")
        assert _is_compound_event_phrase("CDR at first diagnosis")
        assert _is_compound_event_phrase("age when they were first diagnosed")

    def test_is_compound_event_phrase_allows_simple_labels(self):
        assert not _is_compound_event_phrase("age")
        assert not _is_compound_event_phrase("MMSE score")
        assert not _is_compound_event_phrase("sex")
        assert not _is_compound_event_phrase("education years")
        assert not _is_compound_event_phrase("heart attack history")

    def test_structured_intent_rejects_compound_value_variable_label(self):
        from alz_agent.agent.intent import StructuredIntent, QuestionType
        from alz_agent.schemas import IdSpace
        with pytest.raises(pydantic.ValidationError, match="event-anchor phrase"):
            StructuredIntent(
                target_population=[],
                candidate_label="age at first AD diagnosis",
                modalities=[],
                question_type=QuestionType.AGGREGATE_STAT,
                id_space=IdSpace.A,
                value_variable_label="age at first AD diagnosis",
            )

    def test_structured_intent_accepts_simple_value_variable_label(self):
        from alz_agent.agent.intent import StructuredIntent, QuestionType
        from alz_agent.schemas import IdSpace
        intent = StructuredIntent(
            target_population=[],
            candidate_label="mean age",
            modalities=[],
            question_type=QuestionType.AGGREGATE_STAT,
            id_space=IdSpace.A,
            value_variable_label="age",
        )
        assert intent.value_variable_label == "age"

    def test_derive_silently_clears_compound_phrase(self, catalog):
        """When researcher_label is compound, _derive replaces it with None + logs debug."""
        plan = AnalysisPlan(
            original_question="average age when first diagnosed",
            analysis_type="event_anchored_stat",
            research_intent_summary="mean age at first AD diagnosis",
            requires_longitudinal_data=True,
            value_variable=AnalysisRole(
                role="value_variable",
                researcher_label="age at first AD diagnosis",  # compound phrase
                semantic_definition="...",
                selected_raw_code=None,
                resolution_status="unresolved",
            ),
        )
        intent = _derive_structured_intent_from_plan(plan, "average age when first diagnosed")
        assert intent.value_variable_label is None
        assert intent.value_column is None
        assert intent.parse_debug is not None
        assert intent.parse_debug.failure_mode == "variable_resolution"
        assert any("DECOMPOSITION_ERROR" in e for e in intent.parse_debug.validation_errors)

    def test_orchestrator_abstain_message_for_compound_phrase(self, catalog):
        """When value_variable has a compound researcher_label, abstain message is informative."""
        plan = AnalysisPlan(
            original_question="average age when first diagnosed",
            analysis_type="event_anchored_stat",
            research_intent_summary="...",
            requires_longitudinal_data=True,
            value_variable=AnalysisRole(
                role="value_variable",
                researcher_label="age at first AD diagnosis",
                semantic_definition="...",
                selected_raw_code=None,
                resolution_status="unresolved",
            ),
        )
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        fs = FilterSpec(id_space=IdSpace.A)
        results = _plan_aggregate_stat(plan, catalog, backend, fs)
        assert len(results) == 1
        assert not results[0].grounded
        reason = results[0].abstain_reason or ""
        assert "compound event phrase" in reason.lower() or "compound" in reason.lower()
        assert "decompose" in reason.lower()


# ---------------------------------------------------------------------------
# Case H: no default priors (target_population, candidate_label)
# ---------------------------------------------------------------------------

class TestCaseHNoDefaultPriors:
    """When the question doesn't specify MCI or conversion, no defaults are injected."""

    def _ad_age_plan(self) -> AnalysisPlan:
        return AnalysisPlan(
            original_question="Among those diagnosed as AD, what is the average age at first diagnosis?",
            analysis_type="event_anchored_stat",
            research_intent_summary="Mean age at first AD diagnosis",
            requires_longitudinal_data=True,
            candidate_label="average age at first AD diagnosis",
            statistic="mean",
            visit_anchor="first_matching_event_visit",
            value_variable=AnalysisRole(
                role="value_variable", researcher_label="age",
                semantic_definition="Age at first AD visit", selected_raw_code="NACCAGE",
                filter_expression=None, resolution_status="resolved",
            ),
            event=AnalysisRole(
                role="event", researcher_label="first AD diagnosis",
                semantic_definition="First visit with NACCALZD==1",
                selected_raw_code="NACCALZD",
                filter_expression={"NACCALZD": 1}, resolution_status="resolved",
            ),
        )

    def test_no_mci_default_when_cohort_not_specified(self):
        plan = self._ad_age_plan()
        intent = _derive_structured_intent_from_plan(plan, plan.original_question)
        assert intent.target_population != ["mci"], (
            "target_population must not default to ['mci'] when question says nothing about MCI"
        )

    def test_target_etiology_set_from_event_filter(self):
        plan = self._ad_age_plan()
        intent = _derive_structured_intent_from_plan(plan, plan.original_question)
        assert intent.target_etiology == ["ad"]

    def test_candidate_label_not_defaulted_to_conversion_to_dementia(self, catalog):
        """candidate_label must reflect the actual question, not a generic prior."""
        client = _make_plan_client({
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "Mean age at first AD diagnosis",
            "requires_longitudinal_data": True,
            "statistic": "mean",
            "visit_anchor": "first_matching_event_visit",
            "candidate_label": "average age at first AD diagnosis",
            "id_space": "A",
            "value_variable": {
                "role": "value_variable", "researcher_label": "age",
                "semantic_definition": "...", "selected_raw_code": "NACCAGE",
                "resolution_status": "resolved",
            },
            "event": {
                "role": "event", "researcher_label": "first AD diagnosis",
                "semantic_definition": "...", "selected_raw_code": "NACCALZD",
                "filter_expression": {"NACCALZD": 1}, "resolution_status": "resolved",
            },
            "covariates": [], "unresolved_items": [],
        })
        plan = plan_analysis(
            "Among those diagnosed as AD, what is the average age at first diagnosis?",
            client, catalog=catalog,
        )
        assert plan.candidate_label != "conversion to dementia"
        assert "age" in plan.candidate_label.lower() or "ad" in plan.candidate_label.lower()

    def test_stat_variable_is_none_in_plan_path(self):
        """stat_variable must NOT be populated from value_variable.researcher_label."""
        plan = self._ad_age_plan()
        intent = _derive_structured_intent_from_plan(plan, plan.original_question)
        assert intent.stat_variable is None

    def test_value_column_and_label_set_correctly(self):
        plan = self._ad_age_plan()
        intent = _derive_structured_intent_from_plan(plan, plan.original_question)
        assert intent.value_variable_label == "age"
        assert intent.value_column == "NACCAGE"

    def test_event_filter_and_column_set(self):
        plan = self._ad_age_plan()
        intent = _derive_structured_intent_from_plan(plan, plan.original_question)
        assert intent.event_column == "NACCALZD"
        assert intent.event_filter == {"NACCALZD": 1}


# ---------------------------------------------------------------------------
# Case I: parse_debug output
# ---------------------------------------------------------------------------

class TestCaseIParseDebugOutput:
    """parse_debug carries raw plan JSON + normalized plan + errors + failure mode."""

    def _correct_plan_input(self) -> dict[str, Any]:
        return {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "Mean age at first AD diagnosis",
            "requires_longitudinal_data": True,
            "statistic": "mean",
            "visit_anchor": "first_matching_event_visit",
            "candidate_label": "average age at first AD diagnosis",
            "id_space": "A",
            "value_variable": {
                "role": "value_variable", "researcher_label": "age",
                "semantic_definition": "...", "selected_raw_code": "NACCAGE",
                "resolution_status": "resolved",
            },
            "event": {
                "role": "event", "researcher_label": "first AD diagnosis",
                "semantic_definition": "...", "selected_raw_code": "NACCALZD",
                "filter_expression": {"NACCALZD": 1}, "resolution_status": "resolved",
            },
            "covariates": [], "unresolved_items": [],
        }

    def test_internal_returns_debug_info(self, catalog):
        client = _make_plan_client(self._correct_plan_input())
        plan, debug = _plan_analysis_internal(
            "average age at first AD diagnosis", client, catalog=catalog
        )
        assert isinstance(debug, ParseDebugInfo)

    def test_raw_llm_plan_json_captured(self, catalog):
        client = _make_plan_client(self._correct_plan_input())
        _, debug = _plan_analysis_internal(
            "average age at first AD diagnosis", client, catalog=catalog
        )
        assert debug.raw_llm_plan_json.get("analysis_type") == "event_anchored_stat"

    def test_normalized_plan_differs_from_raw(self, catalog):
        client = _make_plan_client(self._correct_plan_input())
        _, debug = _plan_analysis_internal(
            "average age at first AD diagnosis", client, catalog=catalog
        )
        assert "analysis_type" in debug.normalized_plan

    def test_parse_debug_attached_to_intent(self, catalog):
        client = _make_plan_client(self._correct_plan_input())
        plan, debug = _plan_analysis_internal(
            "average age at first AD diagnosis", client, catalog=catalog
        )
        intent = _derive_structured_intent_from_plan(plan, "average age at first AD diagnosis", parse_debug=debug)
        assert intent.parse_debug is not None
        assert isinstance(intent.parse_debug, ParseDebugInfo)
        assert intent.parse_debug.failure_mode is None  # no failures for correct plan

    def test_debug_failure_mode_set_for_compound_phrase(self):
        plan = AnalysisPlan(
            original_question="...",
            analysis_type="event_anchored_stat",
            research_intent_summary="...",
            requires_longitudinal_data=True,
            value_variable=AnalysisRole(
                role="value_variable",
                researcher_label="age at first AD diagnosis",
                semantic_definition="...",
                selected_raw_code=None,
                resolution_status="unresolved",
            ),
        )
        intent = _derive_structured_intent_from_plan(plan, "...")
        assert intent.parse_debug is not None
        assert intent.parse_debug.failure_mode == "variable_resolution"


# ---------------------------------------------------------------------------
# Case J: A/B harness (mocked)
# ---------------------------------------------------------------------------

class TestCaseJPlanABHarness:
    """run_plan_ab compares decomposition quality across providers."""

    def _correct_plan_input(self) -> dict[str, Any]:
        return {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "Mean age at first AD diagnosis",
            "requires_longitudinal_data": True,
            "statistic": "mean",
            "visit_anchor": "first_matching_event_visit",
            "candidate_label": "average age at first AD diagnosis",
            "id_space": "A",
            "value_variable": {
                "role": "value_variable", "researcher_label": "age",
                "semantic_definition": "...", "selected_raw_code": "NACCAGE",
                "resolution_status": "resolved",
            },
            "event": {
                "role": "event", "researcher_label": "first AD diagnosis",
                "semantic_definition": "...", "selected_raw_code": "NACCALZD",
                "filter_expression": {"NACCALZD": 1}, "resolution_status": "resolved",
            },
            "covariates": [], "unresolved_items": [],
        }

    def _bad_plan_input(self) -> dict[str, Any]:
        """Simulates the regression: compound phrase in researcher_label."""
        return {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "...",
            "requires_longitudinal_data": True,
            "statistic": "mean",
            "visit_anchor": "first_matching_event_visit",
            "candidate_label": "conversion to dementia",
            "id_space": "A",
            "value_variable": {
                "role": "value_variable",
                "researcher_label": "age at first AD diagnosis",
                "semantic_definition": "...",
                "selected_raw_code": None,
                "resolution_status": "unresolved",
            },
            "covariates": [], "unresolved_items": [],
        }

    def test_correct_provider_passes(self, catalog):
        client = _make_plan_client(self._correct_plan_input())
        result = run_plan_ab(
            "Among AD-diagnosed participants, what is the average age at first diagnosis?",
            catalog=catalog,
            providers=[("mock-correct", client)],
            expected_value_column="NACCAGE",
            expected_event_filter={"NACCALZD": 1},
            expected_analysis_type="event_anchored_stat",
        )
        assert isinstance(result, PlanABResult)
        assert len(result.rows) == 1
        assert result.rows[0].decomposed_correctly

    def test_bad_provider_fails(self, catalog):
        client = _make_plan_client(self._bad_plan_input())
        result = run_plan_ab(
            "Among AD-diagnosed participants, what is the average age at first diagnosis?",
            catalog=catalog,
            providers=[("mock-bad", client)],
            expected_value_column="NACCAGE",
            expected_event_filter={"NACCALZD": 1},
        )
        assert len(result.rows) == 1
        assert not result.rows[0].decomposed_correctly
        assert result.rows[0].failure_mode is not None

    def test_multi_provider_comparison(self, catalog):
        good = _make_plan_client(self._correct_plan_input())
        bad = _make_plan_client(self._bad_plan_input())
        result = run_plan_ab(
            "Average age at first AD diagnosis",
            catalog=catalog,
            providers=[("good-model", good), ("bad-model", bad)],
            expected_value_column="NACCAGE",
        )
        assert len(result.rows) == 2
        good_row = next(r for r in result.rows if r.provider == "good-model")
        bad_row = next(r for r in result.rows if r.provider == "bad-model")
        assert good_row.decomposed_correctly
        assert not bad_row.decomposed_correctly

    def test_summary_output_contains_provider_names(self, catalog):
        client = _make_plan_client(self._correct_plan_input())
        result = run_plan_ab("Q?", catalog=catalog, providers=[("mymodel", client)])
        summary = result.summary()
        assert "mymodel" in summary
        assert "A/B" in summary

    def test_mci_default_rejection(self, catalog):
        """A plan that injects MCI cohort for a non-MCI question is flagged."""
        mci_plan = {
            "analysis_type": "cohort_size",
            "research_intent_summary": "Count AD participants",
            "requires_longitudinal_data": False,
            "candidate_label": "conversion to dementia",
            "id_space": "A",
            "cohort": {
                "role": "cohort", "researcher_label": "MCI participants",
                "semantic_definition": "...", "selected_raw_code": "NACCUDSD",
                "filter_expression": {"NACCUDSD": 3}, "resolution_status": "resolved",
            },
            "covariates": [], "unresolved_items": [],
        }
        client = _make_plan_client(mci_plan)
        result = run_plan_ab(
            "How many AD-diagnosed participants are there?",
            catalog=catalog,
            providers=[("mci-defaulter", client)],
            reject_mci_default=True,
        )
        assert not result.rows[0].decomposed_correctly
        errors = result.rows[0].validation_errors
        assert any("mci" in e.lower() or "unwanted" in e.lower() for e in errors)
