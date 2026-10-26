"""Tests for catalog-grounded LLM intent parsing and deterministic validation.

Test cases (A–F) per specification:
  A. Valid group comparison — MCI, heart attack history vs MMSE
  B. AD event/diagnosis question — age at first AD diagnosis, NACCALZD handling
  C. Invalid raw_code — LLM returns a column not in catalog
  D. Row identifier — LLM returns NACCID as a variable
  E. Leakage-blocked predictor — NACCUDSD used as predictor
  F. Allowed diagnosis use — NACCUDSD allowed for cohort filter

All LLM responses are mocked via unittest.mock — no API key required.
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alz_agent.agent.intent import (
    QuestionType,
    StructuredIntent,
    _build_catalog_context,
    _compute_leakage_finding,
    _validate_resolved_intent,
    parse_intent,
)
from alz_agent.schemas import (
    CatalogEntry,
    IdSpace,
    LeakageFinding,
    LeakageTag,
    Modality,
    SourceDataset,
)


def test_heuristic_routes_explicit_adni_to_space_b() -> None:
    intent = parse_intent("For ADNI, how many participants have baseline MCI?")
    assert intent.id_space == IdSpace.B


def test_heuristic_rejects_cross_space_question() -> None:
    with pytest.raises(ValueError, match="cannot combine NACC and ADNI"):
        parse_intent("How many participants are in NACC and ADNI together?")


# ---------------------------------------------------------------------------
# Shared catalog fixture
# ---------------------------------------------------------------------------

def _make_catalog() -> list[CatalogEntry]:
    """Six-variable catalog covering all test cases."""
    return [
        CatalogEntry(
            variable_id="NACC.NACCUDSD",
            concept="cognitive status",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCUDSD",
            release_version="nacc70",
            description="UDS consensus diagnosis: 1=Normal 2=Impaired-not-MCI 3=MCI 4=Dementia",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            leakage_reasons=["Primary cognitive-status label"],
            value_coding={"1": "Normal", "2": "Impaired-not-MCI", "3": "MCI", "4": "Dementia"},
        ),
        CatalogEntry(
            variable_id="NACC.NACCALZD",
            concept="etiologic ad diagnosis",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCALZD",
            release_version="nacc70",
            description="Etiologic AD diagnosis: 1=Primary 2=Contributing 3=Not contributing",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.DIAGNOSIS_DEFINING],
            leakage_reasons=["Etiologic AD diagnosis"],
            value_coding={"1": "Primary", "2": "Contributing", "3": "Not contributing"},
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
            value_coding={"0": "Absent", "1": "Recent/Active", "2": "Remote/inactive", "9": "Unknown"},
        ),
        CatalogEntry(
            variable_id="NACC.NACCAGE",
            concept="age at visit",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCAGE",
            release_version="nacc70",
            description="Age of subject at time of UDS visit (years)",
            modality=Modality.DEMOGRAPHIC,
            leakage_tags=[LeakageTag.NONE],
            value_coding={},
        ),
        CatalogEntry(
            variable_id="NACC.NACCID",
            concept="participant id",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCID",
            release_version="nacc70",
            description="Unique participant identifier (row-identifying)",
            modality=Modality.OTHER,
            leakage_tags=[LeakageTag.ROW_IDENTIFIER],
            leakage_reasons=["Row-identifying — never use as a variable"],
            value_coding={},
        ),
    ]


@pytest.fixture()
def catalog() -> list[CatalogEntry]:
    return _make_catalog()


# ---------------------------------------------------------------------------
# Unit tests: _compute_leakage_finding
# ---------------------------------------------------------------------------

class TestComputeLeakageFinding:
    def _entry(self, raw_code: str, tags: list[LeakageTag]) -> CatalogEntry:
        return CatalogEntry(
            variable_id=f"NACC.{raw_code}", concept=raw_code.lower(),
            source_dataset=SourceDataset.NACC, id_space=IdSpace.A,
            raw_code=raw_code, release_version="nacc70",
            leakage_tags=tags,
        )

    def test_none_tag_outcome_is_allowed(self):
        e = self._entry("NACCMMSE", [LeakageTag.NONE])
        f = _compute_leakage_finding("NACCMMSE", "outcome", e)
        assert f.severity == "allowed"
        assert f.leakage_tag == "NONE"
        assert f.action_taken == "allowed_as_outcome"

    def test_diagnosis_defining_as_cohort_filter(self):
        e = self._entry("NACCUDSD", [LeakageTag.DIAGNOSIS_DEFINING])
        f = _compute_leakage_finding("NACCUDSD", "cohort_filter", e)
        assert f.severity == "allow_with_caveat"
        assert f.leakage_tag == "DIAGNOSIS_DEFINING"
        assert f.action_taken == "allowed_for_cohort_definition"

    def test_diagnosis_defining_as_label(self):
        e = self._entry("NACCUDSD", [LeakageTag.DIAGNOSIS_DEFINING])
        f = _compute_leakage_finding("NACCUDSD", "label", e)
        assert f.severity == "allow_with_caveat"
        assert f.action_taken == "allowed_for_cohort_definition"

    def test_diagnosis_defining_as_predictor_is_blocked(self):
        e = self._entry("NACCUDSD", [LeakageTag.DIAGNOSIS_DEFINING])
        f = _compute_leakage_finding("NACCUDSD", "predictor", e)
        assert f.severity == "block"
        assert f.action_taken == "blocked_as_predictor"

    def test_diagnosis_defining_as_covariate_is_blocked(self):
        e = self._entry("NACCUDSD", [LeakageTag.DIAGNOSIS_DEFINING])
        f = _compute_leakage_finding("NACCUDSD", "covariate", e)
        assert f.severity == "block"
        assert f.action_taken == "blocked_as_predictor"

    def test_diagnosis_defining_as_outcome_is_allow_with_caveat(self):
        e = self._entry("NACCUDSD", [LeakageTag.DIAGNOSIS_DEFINING])
        f = _compute_leakage_finding("NACCUDSD", "outcome", e)
        assert f.severity == "allow_with_caveat"
        assert f.action_taken == "allowed_for_descriptive_analysis"

    def test_circular_as_predictor_is_blocked(self):
        e = self._entry("CDRSUM", [LeakageTag.CIRCULAR])
        f = _compute_leakage_finding("CDRSUM", "predictor", e)
        assert f.severity == "block"
        assert f.action_taken == "blocked_as_predictor"

    def test_circular_as_outcome_is_warn(self):
        e = self._entry("CDRSUM", [LeakageTag.CIRCULAR])
        f = _compute_leakage_finding("CDRSUM", "outcome", e)
        assert f.severity == "warn"
        assert f.action_taken == "allowed_as_outcome"

    def test_row_identifier_always_blocked(self):
        e = self._entry("NACCID", [LeakageTag.ROW_IDENTIFIER])
        for role in ("outcome", "cohort_filter", "predictor", "grouping_variable"):
            f = _compute_leakage_finding("NACCID", role, e)
            assert f.severity == "block", f"Expected block for role={role}"
            assert f.action_taken == "caused_abstain"

    def test_harmonization_fit_scope_is_caveat(self):
        e = self._entry("PHC_MEM", [LeakageTag.HARMONIZATION_FIT_SCOPE])
        f = _compute_leakage_finding("PHC_MEM", "covariate", e)
        assert f.severity == "allow_with_caveat"
        assert f.leakage_tag == "HARMONIZATION_FIT_SCOPE"

    def test_none_entry_uses_raw_code_as_concept(self):
        f = _compute_leakage_finding("UNKNOWN_VAR", "outcome", None)
        assert f.concept == "UNKNOWN_VAR"
        assert f.severity == "allowed"


# ---------------------------------------------------------------------------
# Unit tests: _validate_resolved_intent
# ---------------------------------------------------------------------------

class TestValidateResolvedIntent:
    def _base(self, **overrides) -> dict[str, Any]:
        base = {
            "question_type": "statistical_analysis",
            "target_population": ["mci"],
            "candidate_label": "conversion to dementia",
            "id_space": "A",
            "stat_type": "mean",
            "cohort_column": None,
            "group_column": None,
            "outcome_column": None,
            "covariate_columns": [],
            "label_column": None,
            "event_column": None,
            "unresolved_fields": [],
            "parsing_mode": "llm",
        }
        base.update(overrides)
        return base

    def test_valid_columns_accepted(self, catalog):
        data = self._base(group_column="CVHATT", outcome_column="NACCMMSE")
        result = _validate_resolved_intent(data, catalog)
        assert result["group_column"] == "CVHATT"
        assert result["outcome_column"] == "NACCMMSE"
        assert result["unresolved_fields"] == []

    def test_unknown_column_rejected(self, catalog):
        data = self._base(group_column="INVENTED_COL")
        result = _validate_resolved_intent(data, catalog)
        assert result["group_column"] is None
        assert any("INVENTED_COL" in f for f in result["unresolved_fields"])

    def test_row_identifier_blocked(self, catalog):
        data = self._base(outcome_column="NACCID")
        result = _validate_resolved_intent(data, catalog)
        assert result["outcome_column"] is None
        assert any("NACCID" in f for f in result["unresolved_fields"])

    def test_diagnosis_defining_as_cohort_filter_kept(self, catalog):
        data = self._base(cohort_column="NACCUDSD")
        result = _validate_resolved_intent(data, catalog)
        assert result["cohort_column"] == "NACCUDSD"
        findings = result["leakage_assessment"]
        udsd_finding = next(f for f in findings if f["raw_code"] == "NACCUDSD")
        assert udsd_finding["severity"] == "allow_with_caveat"
        assert udsd_finding["action_taken"] == "allowed_for_cohort_definition"

    def test_diagnosis_defining_as_predictor_blocked(self, catalog):
        data = self._base(covariate_columns=["NACCUDSD"])
        result = _validate_resolved_intent(data, catalog)
        assert "NACCUDSD" not in result["covariate_columns"]
        assert any("NACCUDSD" in f for f in result["unresolved_fields"])

    def test_leakage_assessment_populated_for_all_accepted(self, catalog):
        data = self._base(
            cohort_column="NACCUDSD",
            group_column="CVHATT",
            outcome_column="NACCMMSE",
        )
        result = _validate_resolved_intent(data, catalog)
        codes = {f["raw_code"] for f in result["leakage_assessment"]}
        assert "NACCUDSD" in codes
        assert "CVHATT" in codes
        assert "NACCMMSE" in codes

    def test_none_fields_produce_no_findings(self, catalog):
        data = self._base()
        result = _validate_resolved_intent(data, catalog)
        assert result["leakage_assessment"] == []
        assert result["unresolved_fields"] == []

    def test_case_insensitive_raw_code_lookup(self, catalog):
        data = self._base(outcome_column="naccmmse")  # lowercase
        result = _validate_resolved_intent(data, catalog)
        assert result["outcome_column"] == "NACCMMSE"  # normalised to canonical

    def test_none_string_treated_as_null(self, catalog):
        data = self._base(group_column="NONE")
        result = _validate_resolved_intent(data, catalog)
        assert result["group_column"] is None
        assert result["unresolved_fields"] == []


# ---------------------------------------------------------------------------
# Unit tests: _build_catalog_context
# ---------------------------------------------------------------------------

class TestBuildCatalogContext:
    def test_returns_header_row(self, catalog):
        ctx = _build_catalog_context("heart attack MMSE", catalog)
        assert "raw_code" in ctx
        assert "value_coding" in ctx

    def test_includes_all_small_catalog(self, catalog):
        ctx = _build_catalog_context("anything", catalog)
        for entry in catalog:
            assert entry.raw_code in ctx

    def test_always_includes_naccudsd(self, catalog):
        ctx = _build_catalog_context("unrelated query xyz", catalog)
        assert "NACCUDSD" in ctx

    def test_leakage_tag_present_in_context(self, catalog):
        ctx = _build_catalog_context("heart attack", catalog)
        assert "DIAGNOSIS_DEFINING" in ctx

    def test_empty_catalog(self):
        ctx = _build_catalog_context("anything", [])
        assert "empty" in ctx.lower() or "catalog" in ctx.lower()


# ---------------------------------------------------------------------------
# Helper: build a fake Anthropic plan_analysis tool-use response
# ---------------------------------------------------------------------------

def _make_llm_client(tool_input: dict[str, Any]) -> MagicMock:
    """Return a mock Anthropic client that emits a single plan_analysis tool call."""
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
# Integration tests: parse_intent with mocked LLM (cases A–F)
#
# All LLM mocks return plan_analysis tool calls (AnalysisPlan format).
# Pure-function tests (_compute_leakage_finding, _validate_resolved_intent,
# _build_catalog_context) are above and unchanged.
# ---------------------------------------------------------------------------


class TestParseLLMGroundedCaseA:
    """Case A: valid group comparison — MCI, heart attack vs MMSE."""

    def _llm_output(self) -> dict[str, Any]:
        return {
            "analysis_type": "group_comparison",
            "research_intent_summary": (
                "Compare MMSE scores between MCI participants with vs without heart attack history"
            ),
            "requires_longitudinal_data": False,
            "statistic": "mean",
            "statistical_method_hint": "Welch two-sample t-test",
            "candidate_label": "conversion to dementia",
            "id_space": "A",
            "cohort": {
                "role": "cohort",
                "researcher_label": "baseline MCI",
                "semantic_definition": "Participants with MCI at baseline",
                "selected_raw_code": "NACCUDSD",
                "filter_expression": {"NACCUDSD": 3},
                "resolution_status": "resolved",
            },
            "group_variable": {
                "role": "group_variable",
                "researcher_label": "heart attack history",
                "semantic_definition": "History of myocardial infarction (CVHATT > 0)",
                "selected_raw_code": "CVHATT",
                "resolution_status": "resolved",
            },
            "outcome_variable": {
                "role": "outcome_variable",
                "researcher_label": "baseline MMSE",
                "semantic_definition": "Mini-Mental State Examination total score",
                "selected_raw_code": "NACCMMSE",
                "resolution_status": "resolved",
            },
            "covariates": [],
            "unresolved_items": [],
        }

    def test_parsing_mode_is_llm(self, catalog):
        client = _make_llm_client(self._llm_output())
        intent = parse_intent(
            "Among MCI participants, do people with heart attack history have lower MMSE?",
            client=client, catalog=catalog,
        )
        assert intent.parsing_mode == "llm"

    def test_question_type_and_analysis_type(self, catalog):
        client = _make_llm_client(self._llm_output())
        intent = parse_intent(
            "Among MCI participants, do people with heart attack history have lower MMSE?",
            client=client, catalog=catalog,
        )
        assert intent.question_type == QuestionType.STATISTICAL_ANALYSIS
        assert intent.analysis_type == "group_comparison"

    def test_resolved_raw_columns(self, catalog):
        client = _make_llm_client(self._llm_output())
        intent = parse_intent(
            "Among MCI participants, do people with heart attack history have lower MMSE?",
            client=client, catalog=catalog,
        )
        assert intent.cohort_column == "NACCUDSD"
        assert intent.group_column == "CVHATT"
        assert intent.outcome_column == "NACCMMSE"

    def test_cohort_filter_is_dict(self, catalog):
        client = _make_llm_client(self._llm_output())
        intent = parse_intent("...", client=client, catalog=catalog)
        assert intent.cohort_filter == {"NACCUDSD": 3}

    def test_statistical_method_hint(self, catalog):
        client = _make_llm_client(self._llm_output())
        intent = parse_intent("...", client=client, catalog=catalog)
        assert "t-test" in (intent.statistical_method_hint or "").lower()

    def test_leakage_assessment_contains_all_three_variables(self, catalog):
        client = _make_llm_client(self._llm_output())
        intent = parse_intent("...", client=client, catalog=catalog)
        codes = {f.raw_code for f in intent.leakage_assessment}
        assert "NACCUDSD" in codes, "cohort column must appear in leakage_assessment"
        assert "CVHATT" in codes, "group column must appear in leakage_assessment"
        assert "NACCMMSE" in codes, "outcome column must appear in leakage_assessment"

    def test_naccudsd_is_allow_with_caveat_for_cohort(self, catalog):
        client = _make_llm_client(self._llm_output())
        intent = parse_intent("...", client=client, catalog=catalog)
        udsd = next(f for f in intent.leakage_assessment if f.raw_code == "NACCUDSD")
        assert udsd.severity == "allow_with_caveat"
        assert udsd.action_taken == "allowed_for_cohort_definition"
        assert udsd.role == "cohort_filter"

    def test_no_variable_silently_omitted(self, catalog):
        client = _make_llm_client(self._llm_output())
        intent = parse_intent("...", client=client, catalog=catalog)
        assert intent.unresolved_fields == []

    def test_analysis_plan_attached(self, catalog):
        client = _make_llm_client(self._llm_output())
        intent = parse_intent("...", client=client, catalog=catalog)
        assert intent.analysis_plan is not None
        assert intent.analysis_plan.analysis_type == "group_comparison"


class TestParseLLMCaseB:
    """Case B: AD event/diagnosis — age at first AD diagnosis, NACCALZD handling.

    Key invariant: 'average age at first AD diagnosis' MUST NOT be sent as a
    single compound variable. The plan decomposes it into value_variable=NACCAGE
    and event.filter_expression={'NACCALZD':1}, visit_anchor=first_matching_event_visit.
    """

    def _llm_output_with_naccalzd(self) -> dict[str, Any]:
        return {
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "Mean participant age at the first AD diagnosis visit",
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
                "semantic_definition": "First visit where NACCALZD==1 (etiologic AD primary)",
                "selected_raw_code": "NACCALZD",
                "filter_expression": {"NACCALZD": 1},
                "resolution_status": "resolved",
            },
            "covariates": [],
            "unresolved_items": [],
        }

    def test_naccalzd_accepted_when_in_catalog(self, catalog):
        client = _make_llm_client(self._llm_output_with_naccalzd())
        intent = parse_intent(
            "What is the average age among participants first diagnosed with AD?",
            client=client, catalog=catalog,
        )
        assert intent.event_column == "NACCALZD"
        assert intent.outcome_column == "NACCAGE"  # value_variable falls through to outcome_col

    def test_naccalzd_leakage_allowed_as_event(self, catalog):
        client = _make_llm_client(self._llm_output_with_naccalzd())
        intent = parse_intent("...", client=client, catalog=catalog)
        event_findings = [f for f in intent.leakage_assessment if f.raw_code == "NACCALZD"]
        assert any(f.role == "event_definition" for f in event_findings), (
            "NACCALZD should appear in leakage_assessment as event_definition"
        )
        assert all(f.severity != "block" for f in event_findings), (
            "NACCALZD as event should not be blocked"
        )

    def test_nonexistent_naccalzd_column_rejected(self):
        small_catalog = [
            CatalogEntry(
                variable_id="NACC.NACCAGE", concept="age at visit",
                source_dataset=SourceDataset.NACC, id_space=IdSpace.A,
                raw_code="NACCAGE", release_version="nacc70",
                leakage_tags=[LeakageTag.NONE],
            )
        ]
        client = _make_llm_client({
            "analysis_type": "event_anchored_stat",
            "research_intent_summary": "Average age at first AD diagnosis",
            "requires_longitudinal_data": True,
            "value_variable": {
                "role": "value_variable", "researcher_label": "age",
                "semantic_definition": "Age at visit",
                "selected_raw_code": "NACCAGE", "resolution_status": "resolved",
            },
            "event": {
                "role": "event", "researcher_label": "first AD diagnosis",
                "semantic_definition": "First NACCALZD==1 visit",
                "selected_raw_code": "NACCALZD",  # not in small_catalog
                "filter_expression": {"NACCALZD": 1}, "resolution_status": "resolved",
            },
            "unresolved_items": [],
        })
        intent = parse_intent(
            "average age at first AD diagnosis", client=client, catalog=small_catalog
        )
        assert intent.event_column is None
        assert any("NACCALZD" in f for f in intent.unresolved_fields)


class TestParseLLMCaseC:
    """Case C: LLM returns a raw_code not in catalog — validation rejects it."""

    def test_invented_group_column_rejected(self, catalog):
        client = _make_llm_client({
            "analysis_type": "group_comparison",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "group_variable": {
                "role": "group_variable", "researcher_label": "invented factor",
                "semantic_definition": "...", "selected_raw_code": "MADE_UP_COLUMN_XYZ",
                "resolution_status": "resolved",
            },
            "outcome_variable": {
                "role": "outcome_variable", "researcher_label": "MMSE",
                "semantic_definition": "...", "selected_raw_code": "NACCMMSE",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        })
        intent = parse_intent("...", client=client, catalog=catalog)
        assert intent.group_column is None
        assert any("MADE_UP_COLUMN_XYZ" in f for f in intent.unresolved_fields)

    def test_valid_outcome_still_kept_when_group_fails(self, catalog):
        client = _make_llm_client({
            "analysis_type": "group_comparison",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "group_variable": {
                "role": "group_variable", "researcher_label": "nonexistent",
                "semantic_definition": "...", "selected_raw_code": "NONEXISTENT",
                "resolution_status": "resolved",
            },
            "outcome_variable": {
                "role": "outcome_variable", "researcher_label": "MMSE",
                "semantic_definition": "...", "selected_raw_code": "NACCMMSE",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        })
        intent = parse_intent("...", client=client, catalog=catalog)
        assert intent.outcome_column == "NACCMMSE"  # kept
        assert intent.group_column is None           # rejected

    def test_parsing_mode_still_llm_after_rejection(self, catalog):
        client = _make_llm_client({
            "analysis_type": "cohort_size",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "cohort": {
                "role": "cohort", "researcher_label": "ghost cohort",
                "semantic_definition": "...", "selected_raw_code": "GHOST_COLUMN",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        })
        intent = parse_intent("...", client=client, catalog=catalog)
        assert intent.parsing_mode == "llm"
        assert intent.cohort_column is None


class TestParseLLMCaseD:
    """Case D: LLM returns NACCID (row identifier) — validation rejects it."""

    def test_naccid_as_outcome_blocked(self, catalog):
        client = _make_llm_client({
            "analysis_type": "group_comparison",
            "research_intent_summary": "...",
            "requires_longitudinal_data": False,
            "outcome_variable": {
                "role": "outcome_variable", "researcher_label": "participant ID",
                "semantic_definition": "...", "selected_raw_code": "NACCID",
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        })
        intent = parse_intent("...", client=client, catalog=catalog)
        assert intent.outcome_column is None
        assert any("NACCID" in f for f in intent.unresolved_fields)

    def test_naccid_leakage_finding_is_row_identifier_block(self, catalog):
        """Leakage finding is still recorded even though the column is rejected."""
        data = {
            "question_type": "statistical_analysis",
            "target_population": ["mci"],
            "candidate_label": "conversion",
            "id_space": "A",
            "stat_type": "mean",
            "outcome_column": "NACCID",
            "group_column": "CVHATT",
            "unresolved_fields": [],
        }
        validated = _validate_resolved_intent(data, catalog)
        # NACCID should appear in leakage_assessment with block severity
        id_findings = [f for f in validated["leakage_assessment"] if f["raw_code"] == "NACCID"]
        assert id_findings, "NACCID must have a leakage finding even when rejected"
        assert id_findings[0]["severity"] == "block"
        assert id_findings[0]["leakage_tag"] == "ROW_IDENTIFIER"


class TestParseLLMCaseE:
    """Case E: diagnosis-defining variable used as predictor — blocked."""

    def test_naccudsd_as_covariate_blocked(self, catalog):
        client = _make_llm_client({
            "analysis_type": "regression",
            "research_intent_summary": "Regression with MMSE outcome",
            "requires_longitudinal_data": False,
            "outcome_variable": {
                "role": "outcome_variable", "researcher_label": "MMSE",
                "semantic_definition": "Mini-Mental State Examination score",
                "selected_raw_code": "NACCMMSE", "resolution_status": "resolved",
            },
            "covariates": [
                {
                    "role": "covariate", "researcher_label": "cognitive status",
                    "semantic_definition": "UDS consensus cognitive status",
                    "selected_raw_code": "NACCUDSD",  # should be blocked
                    "resolution_status": "resolved",
                },
                {
                    "role": "covariate", "researcher_label": "age",
                    "semantic_definition": "Participant age at visit",
                    "selected_raw_code": "NACCAGE", "resolution_status": "resolved",
                },
            ],
            "unresolved_items": [],
        })
        intent = parse_intent("...", client=client, catalog=catalog)
        assert "NACCUDSD" not in intent.covariate_columns
        assert "NACCAGE" in intent.covariate_columns

    def test_blocked_predictor_appears_in_unresolved_fields(self, catalog):
        # Tests _validate_resolved_intent directly (unchanged path)
        data = {
            "question_type": "statistical_analysis",
            "target_population": ["mci"],
            "candidate_label": "conversion",
            "id_space": "A",
            "stat_type": "mean",
            "covariate_columns": ["NACCUDSD"],
            "unresolved_fields": [],
        }
        validated = _validate_resolved_intent(data, catalog)
        assert any("NACCUDSD" in f for f in validated["unresolved_fields"])

    def test_blocked_predictor_leakage_finding_explains_reason(self, catalog):
        # Tests _validate_resolved_intent directly (unchanged path)
        data = {
            "question_type": "statistical_analysis",
            "target_population": ["mci"],
            "candidate_label": "conversion",
            "id_space": "A",
            "stat_type": "mean",
            "covariate_columns": ["NACCUDSD"],
            "unresolved_fields": [],
        }
        validated = _validate_resolved_intent(data, catalog)
        udsd_finding = next(
            f for f in validated["leakage_assessment"] if f["raw_code"] == "NACCUDSD"
        )
        assert udsd_finding["severity"] == "block"
        assert udsd_finding["action_taken"] == "blocked_as_predictor"
        assert "leakage" in udsd_finding["reason"].lower() or "blocked" in udsd_finding["reason"].lower()


class TestParseLLMCaseF:
    """Case F: NACCUDSD used as cohort filter — allowed with caveat."""

    def test_naccudsd_allowed_for_cohort_definition(self, catalog):
        client = _make_llm_client({
            "analysis_type": "cohort_size",
            "research_intent_summary": "Count MCI participants",
            "requires_longitudinal_data": False,
            "candidate_label": "cohort size",
            "id_space": "A",
            "cohort": {
                "role": "cohort", "researcher_label": "baseline MCI",
                "semantic_definition": "MCI participants at baseline",
                "selected_raw_code": "NACCUDSD",
                "filter_expression": {"NACCUDSD": 3},
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        })
        intent = parse_intent("How many MCI participants are there?", client=client, catalog=catalog)
        assert intent.cohort_column == "NACCUDSD"

    def test_leakage_finding_is_allow_with_caveat_not_block(self, catalog):
        client = _make_llm_client({
            "analysis_type": "cohort_size",
            "research_intent_summary": "Count MCI participants",
            "requires_longitudinal_data": False,
            "candidate_label": "cohort size",
            "id_space": "A",
            "cohort": {
                "role": "cohort", "researcher_label": "baseline MCI",
                "semantic_definition": "MCI participants at baseline",
                "selected_raw_code": "NACCUDSD",
                "filter_expression": {"NACCUDSD": 3},
                "resolution_status": "resolved",
            },
            "unresolved_items": [],
        })
        intent = parse_intent("...", client=client, catalog=catalog)
        udsd_f = next(f for f in intent.leakage_assessment if f.raw_code == "NACCUDSD")
        assert udsd_f.severity == "allow_with_caveat"
        assert udsd_f.action_taken == "allowed_for_cohort_definition"

    def test_leakage_assessment_still_reports_diagnosis_defining_tag(self, catalog):
        """NACCUDSD must not be silently omitted — even when allowed, report the tag."""
        data = {"cohort_column": "NACCUDSD", "unresolved_fields": []}
        validated = _validate_resolved_intent(data, catalog)
        udsd_findings = [f for f in validated["leakage_assessment"] if f["raw_code"] == "NACCUDSD"]
        assert udsd_findings, "NACCUDSD must appear in leakage_assessment"
        assert udsd_findings[0]["leakage_tag"] == "DIAGNOSIS_DEFINING"


# ---------------------------------------------------------------------------
# Heuristic fallback (no client) still produces a valid StructuredIntent
# ---------------------------------------------------------------------------

class TestHeuristicFallback:
    def test_heuristic_returns_structured_intent(self):
        intent = parse_intent(
            "Among MCI participants, do people with heart attack history have lower MMSE?",
        )
        assert isinstance(intent, StructuredIntent)
        assert intent.parsing_mode == "heuristic"

    def test_heuristic_has_empty_resolved_columns(self):
        intent = parse_intent("Do MCI patients with heart attacks have lower MMSE?")
        assert intent.group_column is None
        assert intent.outcome_column is None
        assert intent.leakage_assessment == []
