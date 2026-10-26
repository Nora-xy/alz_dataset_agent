"""Tests for the semantic planning layer.

Sections:
  A. SemanticAnalysisPlan schema
  B. PlanValidator.validate_semantic() — checks C1-C9
  C. compile_to_enriched() — deterministic compiler
  D. Orchestrator repair loop and abstention path
  E. Backward compatibility
"""

from __future__ import annotations

import pytest

from alz_agent.agent.executor import DeterministicExecutor
from alz_agent.agent.plan_validator import PlanValidator
from alz_agent.agent.providers import PlannerClient
from alz_agent.agent.semantic_planner import compile_to_enriched
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    CatalogEntry,
    CohortScope,
    EnrichedAnalysisPlan,
    EtiologyDefinition,
    IdSpace,
    Modality,
    PopulationUnit,
    SemanticAnalysisPlan,
    SourceDataset,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            variable_id="nacc_age",
            concept="age at visit",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCAGE",
            release_version="v70",
            modality=Modality.DEMOGRAPHIC,
        ),
        CatalogEntry(
            variable_id="nacc_alz",
            concept="etiologic alzheimer diagnosis",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCALZD",
            release_version="v70",
            modality=Modality.COGNITION,
        ),
        CatalogEntry(
            variable_id="nacc_udsd",
            concept="cognitive status",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCUDSD",
            release_version="v70",
            modality=Modality.COGNITION,
        ),
    ]


def _correct_first_event_plan(nl: str = "mean age at first etiologic AD diagnosis") -> SemanticAnalysisPlan:
    """The canonical correct SemanticAnalysisPlan for 'mean age at first etiologic AD diagnosis'."""
    return SemanticAnalysisPlan(
        original_question=nl,
        population_unit=PopulationUnit.PARTICIPANT,
        requires_unique_participants=True,
        cohort_scope=CohortScope.FIRST_EVENT,
        baseline_filters={},
        event_filters={"NACCALZD": 1},
        event_anchor="first_matching_event_visit",
        etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
        diagnosis_variable="NACCALZD",
        statistic_variable="NACCAGE",
        statistic_type="mean",
        denominator_definition="unique NACC participants with NACCALZD=1 on any visit",
        known_invalid_interpretations=["NOT baseline-only filter", "NOT visit-row count"],
    )


def _correct_first_event_plan_data() -> dict:
    """Dict suitable for invoke_tool return to produce the correct plan."""
    return {
        "population_unit": "participant",
        "requires_unique_participants": True,
        "cohort_scope": "first_event",
        "baseline_filters": {},
        "event_filters": {"NACCALZD": 1},
        "event_anchor": "first_matching_event_visit",
        "time_ordering_variable": "NACCVNUM",
        "outcome_definition": "NACCALZD=1",
        "etiology_definition": "etiologic_ad",
        "diagnosis_variable": "NACCALZD",
        "statistic_variable": "NACCAGE",
        "statistic_type": "mean",
        "denominator_definition": "unique NACC participants with NACCALZD=1 on any visit",
        "allowed_backend_tools": ["variable_stat"],
        "known_invalid_interpretations": ["NOT baseline-only filter"],
        "caveats": [],
        "abstain_if_missing": [],
        "id_space": "A",
    }


def _invalid_plan_data() -> dict:
    """Dict that produces a plan failing C1/C2b/C4/C5b (the regression scenario)."""
    return {
        "population_unit": "participant",
        "requires_unique_participants": True,
        "cohort_scope": "first_event",
        "baseline_filters": {"etiology": "ad"},  # AD wrongly placed here
        "event_filters": {},                       # empty — C1 + C5b
        "event_anchor": None,                      # missing — C1 + C2b
        "time_ordering_variable": "NACCVNUM",
        "outcome_definition": "NACCALZD=1",
        "etiology_definition": "etiologic_ad",
        "diagnosis_variable": "NACCALZD",
        "statistic_variable": "NACCAGE",
        "statistic_type": "mean",
        "denominator_definition": "",              # missing — C4
        "allowed_backend_tools": [],
        "known_invalid_interpretations": [],
        "caveats": [],
        "abstain_if_missing": [],
        "id_space": "A",
    }


class _FakePlannerClient(PlannerClient):
    """Test double for PlannerClient — returns canned responses from a queue."""

    def __init__(self, invoke_tool_returns: list[dict | None]) -> None:
        self._returns = list(invoke_tool_returns)
        self._call_count = 0
        self.invoke_tool_calls: list = []

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        self.invoke_tool_calls.append((messages, tool_schema))
        if self._call_count < len(self._returns):
            val = self._returns[self._call_count]
            self._call_count += 1
            return val
        return None

    def complete_text(self, messages, max_tokens=1024):
        return ""


# ---------------------------------------------------------------------------
# A. Schema construction and serialization
# ---------------------------------------------------------------------------


class TestSemanticPlanSchema:
    def test_semantic_plan_schema_roundtrip(self):
        plan = _correct_first_event_plan()
        data = plan.model_dump()
        plan2 = SemanticAnalysisPlan.model_validate(data)
        assert plan2.original_question == plan.original_question
        assert plan2.cohort_scope == CohortScope.FIRST_EVENT
        assert plan2.etiology_definition == EtiologyDefinition.ETIOLOGIC_AD
        assert plan2.population_unit == PopulationUnit.PARTICIPANT
        assert plan2.requires_unique_participants is True

    def test_semantic_plan_is_frozen(self):
        plan = _correct_first_event_plan()
        with pytest.raises(Exception):
            plan.cohort_scope = CohortScope.BASELINE  # type: ignore[misc]

    def test_enriched_plan_carries_semantic_plan(self):
        sem = _correct_first_event_plan()
        enriched = compile_to_enriched(sem, sem.original_question)
        assert enriched.semantic_plan is sem

    def test_enriched_plan_without_semantic_plan_defaults_none(self):
        enriched = EnrichedAnalysisPlan(
            original_question="q",
            research_intent_summary="r",
        )
        assert enriched.semantic_plan is None


# ---------------------------------------------------------------------------
# B. PlanValidator.validate_semantic() — checks C1–C9
# ---------------------------------------------------------------------------


class TestValidateSemantic:
    def setup_method(self):
        self.v = PlanValidator(_make_catalog())

    # --- Regression test ---

    def test_regression_etiologic_ad_first_diagnosis_baseline_only_rejected(self):
        """Primary regression: wrong plan for 'mean age at first etiologic AD diagnosis'
        should be rejected with C1, C2b, C4, and C5b errors."""
        bad = SemanticAnalysisPlan(
            original_question="mean age at first etiologic AD diagnosis",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.FIRST_EVENT,
            baseline_filters={"etiology": "ad"},  # AD wrongly here
            event_filters={},                       # empty — bad
            event_anchor=None,                      # missing — bad
            etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            diagnosis_variable="NACCALZD",
            statistic_variable="NACCAGE",
            statistic_type="mean",
            denominator_definition="",  # missing — bad
        )
        res = self.v.validate_semantic(bad)
        assert res.is_executable is False
        error_text = " ".join(res.hard_errors)
        assert "C1" in error_text
        assert "C2b" in error_text
        assert "C4" in error_text
        assert "C5b" in error_text

    def test_correct_etiologic_ad_first_diagnosis_plan_passes(self):
        good = _correct_first_event_plan()
        res = self.v.validate_semantic(good)
        assert res.is_executable is True
        assert res.hard_errors == []

    def test_c16_baseline_diagnosis_allcause_dementia_collision_rejected(self):
        """Regression: a live DeepSeek run for 'MCI participants who converted to AD
        dementia' produced baseline_filters={'diagnosis': 'mci'} together with
        etiology_definition=all_cause_dementia. _filter_dict_for_scope compiles both
        onto the SAME 'diagnosis' filter key, and the all_cause_dementia branch
        unconditionally overwrote 'mci' with 'dementia' — silently discarding the
        baseline-MCI requirement and returning the plain baseline-dementia count
        (a real, wrong, ungrounded number) with is_executable=True and no caveat.
        This combination must be rejected instead of silently mis-compiled."""
        plan = SemanticAnalysisPlan(
            original_question=(
                "how many MCI participants at baseline converted to AD dementia "
                "within 3 years of follow-up?"
            ),
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.FIRST_EVENT,
            baseline_filters={"diagnosis": "mci"},
            event_filters={"DXDDUE": 1},
            event_anchor="first_conversion_visit",
            etiology_definition=EtiologyDefinition.ALL_CAUSE_DEMENTIA,
            diagnosis_variable="ADNI.DXSUM.DXMCI,ADNI.DXSUM.DXDDUE",
            denominator_definition="unique ADNI participants with baseline MCI who convert",
            id_space="B",
        )
        res = self.v.validate_semantic(plan)
        assert res.is_executable is False
        assert any("C16" in e for e in res.hard_errors)

    def test_c16_allows_diagnosis_dementia_with_allcause_no_collision(self):
        """No collision (and no C16 error) when baseline_filters['diagnosis'] already
        agrees with all_cause_dementia's 'dementia' value, or is simply absent."""
        plan = SemanticAnalysisPlan(
            original_question="baseline dementia prevalence",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.ALL_CAUSE_DEMENTIA,
            diagnosis_variable="NACCUDSD",
            denominator_definition="unique NACC participants with baseline dementia",
        )
        res = self.v.validate_semantic(plan)
        assert not any("C16" in e for e in res.hard_errors)

    # --- C1: first_event coherence ---

    def test_c1_first_event_empty_event_filters(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.FIRST_EVENT,
            event_filters={},
            event_anchor="first_matching_event_visit",
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            denominator_definition="some denominator",
        )
        res = self.v.validate_semantic(plan)
        assert any("C1" in e and "event_filters" in e for e in res.hard_errors)

    def test_c1_first_event_wrong_anchor(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.FIRST_EVENT,
            event_filters={"NACCALZD": 1},
            event_anchor="baseline",  # wrong for first_event
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            denominator_definition="some denominator",
        )
        res = self.v.validate_semantic(plan)
        assert any("C1" in e and "event_anchor" in e for e in res.hard_errors)

    # --- C2: participant uniqueness ---

    def test_c2a_participant_unit_requires_unique_false_hard_error(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=False,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
        )
        res = self.v.validate_semantic(plan)
        assert any("C2a" in e for e in res.hard_errors)

    def test_c2b_participant_anyvisit_no_anchor_hard_error(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.ANY_VISIT,
            event_anchor=None,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
        )
        res = self.v.validate_semantic(plan)
        assert any("C2b" in e for e in res.hard_errors)

    def test_c2_baseline_scope_does_not_require_anchor(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            event_anchor=None,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
        )
        res = self.v.validate_semantic(plan)
        c2b_errors = [e for e in res.hard_errors if "C2b" in e]
        assert c2b_errors == []

    # --- C3: etiology vs all-cause ---

    def test_c3_etiologic_ad_naccudsd_only_hard_error(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.FIRST_EVENT,
            event_filters={"NACCALZD": 1},
            event_anchor="first_matching_event_visit",
            etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            diagnosis_variable="NACCUDSD",  # wrong — NACCALZD missing
            denominator_definition="some denominator",
        )
        res = self.v.validate_semantic(plan)
        assert any("C3" in e for e in res.hard_errors)

    def test_c3_etiologic_ad_with_naccalzd_passes(self):
        plan = _correct_first_event_plan()
        res = self.v.validate_semantic(plan)
        c3_errors = [e for e in res.hard_errors if "C3" in e]
        assert c3_errors == []

    # --- C4: denominator required ---

    def test_c4_mean_requires_denominator_hard_error(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.FIRST_EVENT,
            event_filters={"NACCALZD": 1},
            event_anchor="first_matching_event_visit",
            etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            diagnosis_variable="NACCALZD",
            statistic_variable="NACCAGE",
            statistic_type="mean",
            denominator_definition="",  # missing — should trigger C4
        )
        res = self.v.validate_semantic(plan)
        assert any("C4" in e for e in res.hard_errors)

    def test_c4_count_also_requires_denominator(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            statistic_type="count",
            denominator_definition="",
        )
        res = self.v.validate_semantic(plan)
        assert any("C4" in e for e in res.hard_errors)

    # --- C5/C5b: baseline/event etiology coherence ---

    def test_c5_baseline_scope_ad_in_event_filters_only_hard_error(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            baseline_filters={},
            event_filters={"event_etiology": "ad"},  # AD wrongly in event
            etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            diagnosis_variable="NACCALZD",
            denominator_definition="some denom",
        )
        res = self.v.validate_semantic(plan)
        assert any("C5" in e for e in res.hard_errors)

    def test_c5b_first_event_ad_in_baseline_filters_only_hard_error(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.FIRST_EVENT,
            baseline_filters={"etiology": "ad"},  # AD wrongly here
            event_filters={},
            event_anchor="first_matching_event_visit",
            etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            diagnosis_variable="NACCALZD",
            statistic_type="mean",
            denominator_definition="some denom",
        )
        res = self.v.validate_semantic(plan)
        assert any("C5b" in e for e in res.hard_errors)

    # --- C6: unclear scope ---

    def test_c6_unclear_cohort_scope_hard_error(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.UNCLEAR,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
        )
        res = self.v.validate_semantic(plan)
        assert any("C6" in e for e in res.hard_errors)

    # --- C8 (soft): anchor/scope alignment ---

    def test_c8_baseline_scope_nontrivial_anchor_soft_warning(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            event_anchor="last_visit",  # non-baseline anchor with baseline scope
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
        )
        res = self.v.validate_semantic(plan)
        assert res.is_executable is True  # soft only, not a hard error
        assert any("C8" in w for w in res.soft_warnings)

    # --- C9: abstain_if_missing as assumptions ---

    def test_c9_abstain_if_missing_surfaces_as_assumptions(self):
        plan = SemanticAnalysisPlan(
            original_question="q",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            abstain_if_missing=["NACCALZD not in catalog"],
        )
        res = self.v.validate_semantic(plan)
        assert any("NACCALZD" in a for a in res.assumptions)


# ---------------------------------------------------------------------------
# C. compile_to_enriched() — deterministic compiler
# ---------------------------------------------------------------------------


class TestCompileToEnriched:
    def test_compile_first_event_emits_event_etiology_and_anchor(self):
        """compile_to_enriched on a first_event etiologic_ad plan must emit
        event_etiology='ad' explicitly (not baseline_etiology) and event_anchor=first_diagnosis."""
        sem = _correct_first_event_plan()
        enriched = compile_to_enriched(sem, sem.original_question)

        # Check the variable_stat op
        stat_ops = [op for op in enriched.operations if op.op_type == "variable_stat"]
        assert len(stat_ops) == 1, "Expected exactly one variable_stat op"
        op = stat_ops[0]

        # event_anchor should be translated to backend name
        assert op.parameters.get("event_anchor") == "first_diagnosis"

        # Filter must use event_etiology (not etiology / baseline_etiology)
        op_filter = op.parameters.get("filter", {})
        assert "event_etiology" in op_filter, (
            f"event_etiology must be explicit in filter dict; got {op_filter}"
        )
        assert op_filter["event_etiology"] == "ad"
        assert "etiology" not in op_filter or op_filter.get("etiology") is None

        # Now verify via executor._build_filter_spec
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        executor = DeterministicExecutor(catalog=_make_catalog(), backend=backend)
        fs = executor._build_filter_spec(enriched, op.parameters)
        assert fs.event_etiology == ["ad"]
        assert fs.baseline_etiology is None

    def test_compile_baseline_scope_uses_baseline_etiology(self):
        """compile_to_enriched on a baseline etiologic_ad plan must use baseline_etiology."""
        sem = SemanticAnalysisPlan(
            original_question="baseline AD prevalence",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            baseline_filters={"etiology": "ad"},
            event_filters={},
            event_anchor=None,
            etiology_definition=EtiologyDefinition.ETIOLOGIC_AD,
            diagnosis_variable="NACCALZD",
            denominator_definition="unique NACC participants at baseline with NACCALZD=1",
        )
        enriched = compile_to_enriched(sem, sem.original_question)

        count_ops = [op for op in enriched.operations if op.op_type == "count_eligible"]
        assert len(count_ops) == 1
        op = count_ops[0]

        op_filter = op.parameters.get("filter", {})
        assert "etiology" in op_filter or "event_etiology" not in op_filter

        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        executor = DeterministicExecutor(catalog=_make_catalog(), backend=backend)
        fs = executor._build_filter_spec(enriched, op.parameters)
        # Baseline etiology should be used, not event_etiology
        assert fs.baseline_etiology == ["ad"]
        assert fs.event_etiology is None

    def test_compile_variable_id_prefixed(self):
        """compile_to_enriched should prefix NACCAGE → NACC.NACCAGE for id_space A."""
        sem = _correct_first_event_plan()
        enriched = compile_to_enriched(sem, sem.original_question)
        stat_ops = [op for op in enriched.operations if op.op_type == "variable_stat"]
        assert stat_ops[0].parameters["variable_id"] == "NACC.NACCAGE"

    def test_compile_count_eligible_op_added_for_participant_unit(self):
        """When population_unit=participant, a count_eligible op must be emitted."""
        sem = _correct_first_event_plan()
        enriched = compile_to_enriched(sem, sem.original_question)
        count_ops = [op for op in enriched.operations if op.op_type == "count_eligible"]
        assert len(count_ops) == 1

    def test_compile_semantic_plan_carried_on_result(self):
        sem = _correct_first_event_plan()
        enriched = compile_to_enriched(sem, sem.original_question)
        assert enriched.semantic_plan is sem

    def test_compile_required_modalities_emits_modality_overlap_op(self):
        """A question asking for data-availability across modalities (e.g. 'have both
        MRI and CSF biomarker data') must compile to a modality_overlap op, not just
        a plain count_eligible that silently ignores the modality requirement.

        Regression: compile_to_enriched previously had no channel for this at all —
        required_modalities did not exist on SemanticAnalysisPlan, so any such
        question silently fell through to a bare cohort-size count."""
        sem = SemanticAnalysisPlan(
            original_question="how many participants have both pet and fluid_biomarker data",
            population_unit=PopulationUnit.PARTICIPANT,
            requires_unique_participants=True,
            cohort_scope=CohortScope.BASELINE,
            etiology_definition=EtiologyDefinition.UNSPECIFIED,
            denominator_definition="unique participants with a baseline visit",
            required_modalities=["pet", "fluid_biomarker"],
            id_space="B",
        )
        enriched = compile_to_enriched(sem, sem.original_question)

        overlap_ops = [op for op in enriched.operations if op.op_type == "modality_overlap"]
        assert len(overlap_ops) == 1, "Expected exactly one modality_overlap op"
        assert overlap_ops[0].parameters["modalities"] == ["pet", "fluid_biomarker"]

        # count_eligible for the whole cohort should still be emitted alongside it.
        count_ops = [op for op in enriched.operations if op.op_type == "count_eligible"]
        assert len(count_ops) == 1

    def test_compile_no_required_modalities_emits_no_modality_overlap_op(self):
        """Backward compatibility: required_modalities defaults empty, so plans that
        predate this field compile exactly as before (no modality_overlap op)."""
        sem = _correct_first_event_plan()
        assert sem.required_modalities == []
        enriched = compile_to_enriched(sem, sem.original_question)
        assert not any(op.op_type == "modality_overlap" for op in enriched.operations)


# ---------------------------------------------------------------------------
# D. Orchestrator repair loop and abstention path
# ---------------------------------------------------------------------------


class TestOrchestratorRepairAndAbstain:
    """Integration tests for the semantic gate in _run_enriched_path.

    Uses a fake PlannerClient to control invoke_tool return values without
    hitting any real LLM or backend (fixture backend only).
    """

    def _run(self, client, use_semantic=True):
        from alz_agent.agent.orchestrator import run_agent
        catalog = _make_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        nl = "Among participants with etiologic Alzheimer's disease in NACC, what is the mean age at first diagnosis?"
        return run_agent(
            nl,
            catalog,
            backend,
            client=client,
            use_enriched_planner=True,
            use_semantic_planner=use_semantic,
        )

    def test_repair_loop_fixes_then_passes(self):
        """Fake client: first call returns invalid plan, second (repair) returns valid plan.
        The orchestrator should succeed after the repair."""
        client = _FakePlannerClient([
            _invalid_plan_data(),       # first call: invalid plan
            _correct_first_event_plan_data(),  # second call (repair): valid plan
            # Subsequent calls for plan_enriched_analysis / narration return None
            # so _run_enriched_path will get None from plan_enriched_analysis and short-circuit.
            # But with use_semantic_planner=True, enriched path takes the compiled plan, so
            # plan_enriched_analysis is not called.
        ])
        resp = self._run(client)
        # The semantic plan was repaired — we should not see an abstain about semantics
        # (we may still see an abstain from the executor if counts are suppressed, but
        # the semantic validation error should not be in abstain_reason)
        if resp.abstain_reason:
            assert "not answerable with the current backend semantics" not in resp.abstain_reason
        # Semantic validation hard errors should be empty (repair succeeded)
        assert resp.validation_hard_errors == []

    def test_repair_loop_exhausted_abstains(self):
        """Fake client always returns invalid plan data — repair also fails.
        The orchestrator must return grounded=False with the canonical abstain message."""
        client = _FakePlannerClient([
            _invalid_plan_data(),  # first call: invalid
            _invalid_plan_data(),  # second call (repair): still invalid
        ])
        resp = self._run(client)
        assert resp.grounded is False
        assert "not answerable with the current backend semantics" in resp.answer
        assert len(resp.validation_hard_errors) > 0

    def test_abstain_if_missing_surfaced_as_caveat_and_flips_question_answered(self):
        """Regression: PlanValidator._sc9_abstain_assumptions folded sem.abstain_if_missing
        into ValidationResult.assumptions, but nothing downstream of a successful
        validation ever read that field — a question like '...and MRI images for at
        least two visits' would silently return a plain grounded count with zero
        indication the MRI/visit-count part was dropped, and question_answered stayed
        True. Fixed by orchestrator._unaddressed_constraints_caveat(), which reads
        enriched_plan.semantic_plan.abstain_if_missing and injects it as a mandatory
        caveat on every answer path, and flips question_answered to False."""
        plan_data = dict(_correct_first_event_plan_data())
        plan_data["abstain_if_missing"] = [
            "no MRI-per-visit variable in catalog context"
        ]
        client = _FakePlannerClient([plan_data])
        resp = self._run(client)

        assert resp.grounded is True, "the executable part of the plan should still run"
        assert "no MRI-per-visit variable in catalog context" in resp.answer
        assert resp.question_answered is False

    def test_no_abstain_if_missing_leaves_question_answered_true(self):
        """Control: a plan with an empty abstain_if_missing list must not trip the
        new question_answered=False path — only a non-empty list should."""
        client = _FakePlannerClient([_correct_first_event_plan_data()])
        resp = self._run(client)

        assert resp.grounded is True
        assert resp.question_answered is True

    def test_backward_compat_enriched_planner_unchanged(self):
        """use_semantic_planner=False must not invoke any semantic planner code.
        The enriched path should still work via plan_enriched_analysis."""
        # Client returns valid enriched plan data on first call, None after
        enriched_data = {
            "research_intent_summary": "count MCI participants",
            "researcher_audit_trace": ["Interpreted as MCI count."],
            "operations": [
                {
                    "op_type": "count_eligible",
                    "parameters": {"filter": {"diagnosis": "mci"}},
                    "output_name": "n_mci",
                    "description": "Count MCI.",
                }
            ],
            "requires_backend_data": True,
            "requested_horizons": [],
            "baseline_definition": {"diagnosis": "mci"},
            "outcome_definition": "",
            "id_space": "A",
        }
        client = _FakePlannerClient([enriched_data])
        from alz_agent.agent.orchestrator import run_agent
        catalog = _make_catalog()
        backend = build_fixture_backend(n_mci=30, n_normal=20, seed=42)
        resp = run_agent(
            "How many MCI participants are there?",
            catalog,
            backend,
            client=client,
            use_enriched_planner=True,
            use_semantic_planner=False,  # explicitly off
        )
        # Should NOT return the "not answerable with the current backend semantics" message
        assert "not answerable with the current backend semantics" not in resp.answer
        # enriched_plan_used should be True (used the enriched path, not semantic)
        assert resp.enriched_plan_used is True
