"""Tests for the structured explanation layer (ExplanationContext, coverage gate, deduplication).

These tests verify:
  - ExplainableItem / LeakageRecord / StructuredCaveat construction
  - build_explanation_context() produces correct items
  - _deduplicate_items() suppresses duplicate (display_name, value) pairs
  - compute_explanation_coverage() correctly measures coverage
  - The template narration (_narrate_template_from_ctx) is deduplication-safe and
    hallucination-guard-safe
  - Leakage blacklist items include raw_code, tag, reason (not just count)
  - Horizon label carries "(example default)" when horizon_source == "default"
  - NOT ANSWERABLE items are flagged and explained
"""

from __future__ import annotations

import re

import pytest

from alz_agent.agent.explanation import (
    _EXPLANATION_COVERAGE_THRESHOLD,
    build_explanation_context,
    compute_explanation_coverage,
    _deduplicate_items,
)
from alz_agent.agent.orchestrator import (
    _check_hallucination,
    _collect_evidence_numbers,
    _narrate_template,
    run_agent,
)
from alz_agent.agent.intent import QuestionType, StructuredIntent
from alz_agent.backend.fixture_backend import build_fixture_backend
from alz_agent.schemas import (
    CatalogEntry,
    ExplainableItem,
    FilterSpec,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
    ToolResult,
)
from alz_agent.tools.deterministic import (
    ModelingDesignResult,
    suggest_modeling_design,
    compute_cohort_size,
    assess_modality_overlap,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_catalog() -> list[CatalogEntry]:
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


@pytest.fixture()
def catalog():
    return _make_catalog()


@pytest.fixture()
def backend():
    return build_fixture_backend(
        n_normal=50, n_mci=40, n_dementia=30, n_other=20,
        n_mci_long=25, n_mci_short=8, n_single_visit=15, n_suppressed_group=5,
        n_mci_ad=15, n_dementia_ad=15,
        modality_fractions={
            "cognition": 0.9, "mri": 0.15, "pet": 0.04,
            "fluid_biomarker": 0.05, "vascular": 0.8, "neuropath": 0.0,
        },
        seed=42,
    )


@pytest.fixture()
def mci_spec():
    return FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A)


@pytest.fixture()
def design_result(catalog, backend, mci_spec):
    tr = suggest_modeling_design(mci_spec, 3, catalog, backend)
    assert tr.grounded
    return tr


# ---------------------------------------------------------------------------
# Tests: LeakageRecord in ModelingDesignResult
# ---------------------------------------------------------------------------


class TestLeakageRecordsInDesignResult:
    def test_leakage_records_non_empty(self, design_result):
        val: ModelingDesignResult = design_result.value
        assert len(val.leakage_records) > 0, "Expect at least one blacklisted variable"

    def test_leakage_records_count_matches_blacklist(self, design_result):
        val: ModelingDesignResult = design_result.value
        assert len(val.leakage_records) == len(val.leakage_blacklist)

    def test_leakage_record_has_raw_code(self, design_result):
        for rec in design_result.value.leakage_records:
            assert rec.raw_code, f"raw_code missing for {rec.variable_id}"

    def test_leakage_record_has_tag(self, design_result):
        for rec in design_result.value.leakage_records:
            assert rec.tag, f"tag missing for {rec.variable_id}"

    def test_leakage_record_has_reason(self, design_result):
        for rec in design_result.value.leakage_records:
            assert rec.reason, f"reason missing for {rec.variable_id}"

    def test_leakage_record_has_display_name(self, design_result):
        for rec in design_result.value.leakage_records:
            assert rec.display_name, f"display_name missing for {rec.variable_id}"

    def test_leakage_record_excluded_from_valid(self, design_result):
        valid = {"features", "labels", "both"}
        for rec in design_result.value.leakage_records:
            assert rec.excluded_from in valid

    def test_naccudsd_tag_is_diagnosis_defining(self, design_result):
        naccudsd_recs = [r for r in design_result.value.leakage_records if r.raw_code == "NACCUDSD"]
        assert naccudsd_recs, "NACCUDSD should be in leakage records"
        assert "DIAGNOSIS_DEFINING" in naccudsd_recs[0].tags


# ---------------------------------------------------------------------------
# Tests: build_explanation_context
# ---------------------------------------------------------------------------


class TestBuildExplanationContext:
    def test_returns_explanation_context(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        assert ctx is not None

    def test_items_non_empty(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        assert len(ctx.items) > 0

    def test_eligible_n_item_present(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        eligible_items = [i for i in ctx.items if i.item_id == "eligible_n"]
        assert eligible_items, "eligible_n item should be present"

    def test_no_duplicate_eligible_n(self, catalog, backend, mci_spec, design_result):
        """Providing the same tool result twice must not produce duplicate items."""
        ctx = build_explanation_context(
            [design_result, design_result],  # same result submitted twice
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        dn = [i.display_name.lower() for i in ctx.items]
        # "Eligible participants" must appear at most once after deduplication
        assert dn.count("eligible participants") <= 1, (
            "Duplicate eligible_n items not deduped when same tool result given twice"
        )

    def test_leakage_var_items_present(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        leakage_var_items = [
            i for i in ctx.items if i.item_id.startswith("leakage_var_")
        ]
        assert len(leakage_var_items) == len(design_result.value.leakage_records)

    def test_leakage_var_items_have_raw_code_value(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        for item in ctx.items:
            if item.item_id.startswith("leakage_var_"):
                assert item.value is not None, f"{item.item_id} should have raw_code as value"
                assert item.computation_note, f"{item.item_id} should have computation_note with tag+reason"

    def test_feasibility_items_have_definitions(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        for key in ("censored", "insufficient", "sufficient"):
            items = [i for i in ctx.items if i.raw_key == key]
            if items:
                assert items[0].definition, f"{key} item must have a definition"
                assert items[0].explanation_required, f"{key} item must be explanation_required"

    def test_horizon_user_label(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        assert ctx.metadata.horizon_source == "user"
        assert "default" not in ctx.metadata.horizon_label.lower()

    def test_horizon_default_label(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="default",
            catalog=catalog,
        )
        assert ctx.metadata.horizon_source == "default"
        assert "default" in ctx.metadata.horizon_label.lower()

    def test_modality_abstain_creates_not_answerable_item(self, catalog, backend, mci_spec):
        abstained = ToolResult.abstain(
            "Modalities absent from catalog for id_space=A: ['demographic']"
        )
        ctx = build_explanation_context(
            [abstained],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        assert len(ctx.not_answerable) == 1
        na_item = ctx.not_answerable[0]
        assert na_item.is_not_answerable is True
        assert na_item.explanation_required is True

    def test_modality_items_have_definitions(self, catalog, backend, mci_spec):
        mod_tr = assess_modality_overlap(["cognition", "mri"], mci_spec, backend, catalog)
        if not mod_tr.grounded:
            pytest.skip("Modality overlap not grounded in this catalog")
        ctx = build_explanation_context(
            [mod_tr],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        for item in ctx.items:
            if item.source_tool == "assess_modality_overlap":
                assert item.definition, f"Modality item {item.item_id} must have a definition"


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def _make_item(self, item_id, display_name, value, importance="medium"):
        return ExplainableItem(
            item_id=item_id,
            display_name=display_name,
            raw_key=item_id,
            value=value,
            source_tool="test",
            importance=importance,
        )

    def test_exact_duplicates_removed(self):
        items = [
            self._make_item("a", "Eligible participants", 100),
            self._make_item("b", "Eligible participants", 100),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 1

    def test_different_values_kept(self):
        items = [
            self._make_item("a", "Eligible participants", 100),
            self._make_item("b", "Eligible participants", 200),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 2

    def test_higher_importance_wins(self):
        items = [
            self._make_item("a", "Eligible participants", 100, importance="low"),
            self._make_item("b", "Eligible participants", 100, importance="high"),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 1
        assert result[0].importance == "high"

    def test_distinct_items_preserved(self):
        items = [
            self._make_item("a", "Eligible participants", 100),
            self._make_item("b", "Sufficient follow-up", 60),
            self._make_item("c", "Insufficient follow-up", 25),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: compute_explanation_coverage
# ---------------------------------------------------------------------------


class TestExplanationCoverage:
    def test_perfect_coverage(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        # Build template answer which should cover all required items
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        answer = _narrate_template(intent, [design_result], [], ctx=ctx)
        ratio, missing = compute_explanation_coverage(ctx, answer)
        assert ratio >= _EXPLANATION_COVERAGE_THRESHOLD, (
            f"Template coverage {ratio:.0%} below threshold. Missing: {missing}"
        )

    def test_empty_answer_zero_coverage(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        required = ctx.required_items()
        if not required:
            pytest.skip("No required items in context")
        ratio, missing = compute_explanation_coverage(ctx, "")
        assert ratio == 0.0
        assert len(missing) == len(required)

    def test_full_answer_full_coverage(self, catalog, backend, mci_spec, design_result):
        ctx = build_explanation_context(
            [design_result],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        # Construct an answer that explicitly names every required item
        required = ctx.required_items()
        answer = " ".join(item.display_name for item in required)
        ratio, missing = compute_explanation_coverage(ctx, answer)
        assert ratio == 1.0
        assert missing == []

    def test_no_required_items_gives_full_coverage(self, catalog, backend, mci_spec):
        cohort_tr = compute_cohort_size(mci_spec, backend)
        ctx = build_explanation_context(
            [cohort_tr],
            filter_spec=mci_spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        # Manually clear explanation_required on all items
        from alz_agent.schemas import ExplanationMetadata, ExplanationContext
        empty_ctx = ExplanationContext(
            items=[],
            sections=ctx.sections,
            caveats=ctx.caveats,
            metadata=ctx.metadata,
        )
        ratio, missing = compute_explanation_coverage(empty_ctx, "any answer")
        assert ratio == 1.0


# ---------------------------------------------------------------------------
# Tests: Template narration with ctx
# ---------------------------------------------------------------------------


class TestTemplateNarrationWithContext:
    def test_no_raw_duplicate_eligible_lines(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        design_tr = suggest_modeling_design(spec, 3, catalog, backend)
        ctx = build_explanation_context(
            [design_tr],
            filter_spec=spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        answer = _narrate_template(intent, [design_tr], [], ctx=ctx)
        # Count occurrences of "eligible participants" (case-insensitive)
        count = answer.lower().count("eligible participants")
        assert count <= 1, f"'eligible participants' appears {count} times — deduplication failed"

    def test_feasibility_terms_defined(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        design_tr = suggest_modeling_design(spec, 3, catalog, backend)
        ctx = build_explanation_context(
            [design_tr],
            filter_spec=spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        answer = _narrate_template(intent, [design_tr], [], ctx=ctx)
        lower = answer.lower()
        # At least one feasibility term must be present with a definition note
        assert "censored" in lower or "insufficient" in lower or "sufficient" in lower

    def test_leakage_vars_named_not_just_count(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        design_tr = suggest_modeling_design(spec, 3, catalog, backend)
        ctx = build_explanation_context(
            [design_tr],
            filter_spec=spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        answer = _narrate_template(intent, [design_tr], [], ctx=ctx)
        # At least one concrete leakage variable raw_code must appear
        leakage_raw_codes = [r.raw_code for r in design_tr.value.leakage_records]
        found = any(code in answer for code in leakage_raw_codes)
        assert found, (
            f"No leakage raw_code found in answer. Codes: {leakage_raw_codes}\n"
            f"Answer snippet: {answer[:300]}"
        )

    def test_hallucination_guard_passes_on_ctx_template(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        design_tr = suggest_modeling_design(spec, 3, catalog, backend)
        ctx = build_explanation_context(
            [design_tr],
            filter_spec=spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        answer = _narrate_template(intent, [design_tr], [], ctx=ctx)
        assert _check_hallucination(answer, [design_tr]), (
            "Template with ctx must pass hallucination guard.\n"
            f"Answer ints: {sorted(set(re.findall(r'\\b\\d+\\b', answer)))}\n"
            f"Evidence nums: {sorted(_collect_evidence_numbers([design_tr]))}"
        )

    def test_default_horizon_label_in_ctx_template(self, catalog, backend):
        spec = FilterSpec(baseline_diagnosis=["mci"])
        design_tr = suggest_modeling_design(spec, 3, catalog, backend)
        ctx = build_explanation_context(
            [design_tr],
            filter_spec=spec,
            horizon=3,
            horizon_source="default",  # not user-specified
            catalog=catalog,
        )
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        answer = _narrate_template(intent, [design_tr], [], ctx=ctx)
        assert "default" in answer.lower(), (
            "Template must mention 'default' when horizon_source='default'"
        )

    def test_not_answerable_item_explained_in_template(self, catalog, backend):
        abstained = ToolResult.abstain("Modalities absent from catalog: ['demographic']")
        spec = FilterSpec(baseline_diagnosis=["mci"])
        ctx = build_explanation_context(
            [abstained],
            filter_spec=spec,
            horizon=3,
            horizon_source="user",
            catalog=catalog,
        )
        intent = StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to AD",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        )
        answer = _narrate_template(intent, [abstained], [], ctx=ctx)
        lower = answer.lower()
        assert "not answerable" in lower or "not_answerable" in lower or "abstain" in lower


# ---------------------------------------------------------------------------
# Tests: run_agent end-to-end with explanation context
# ---------------------------------------------------------------------------


class TestRunAgentExplanationIntegration:
    WORKED_TARGET = (
        "Among MCI participants at an index visit, can we predict 3-year conversion "
        "to AD using only pre-index information, and how many subjects survive that "
        "definition?"
    )

    def test_run_agent_produces_sectioned_answer(self, catalog, backend):
        response = run_agent(self.WORKED_TARGET, catalog, backend)
        assert response.grounded
        # Structured template should have section headers
        assert "##" in response.answer or "Cohort" in response.answer

    def test_run_agent_names_feasibility_terms(self, catalog, backend):
        response = run_agent(self.WORKED_TARGET, catalog, backend)
        lower = response.answer.lower()
        # At least one of the three feasibility terms must appear
        assert any(t in lower for t in ("censored", "insufficient", "sufficient")), (
            "Answer must mention at least one feasibility term"
        )

    def test_run_agent_names_leakage_vars(self, catalog, backend):
        response = run_agent(self.WORKED_TARGET, catalog, backend)
        # NACCUDSD is the primary leakage variable — must appear by name
        assert "NACCUDSD" in response.answer or "naccudsd" in response.answer.lower(), (
            "Leakage variable NACCUDSD must be named in the answer"
        )

    def test_run_agent_hallucination_guard_passes(self, catalog, backend):
        response = run_agent(self.WORKED_TARGET, catalog, backend)
        assert response.hallucination_check_passed

    def test_run_agent_grounded(self, catalog, backend):
        response = run_agent(self.WORKED_TARGET, catalog, backend)
        assert response.grounded

    def test_run_agent_no_duplicate_eligible_line(self, catalog, backend):
        response = run_agent(self.WORKED_TARGET, catalog, backend)
        count = response.answer.lower().count("eligible participant")
        assert count <= 1, (
            f"'eligible participant' repeated {count} times — dedup failed"
        )

    def test_run_agent_whitelist_explained(self, catalog, backend):
        response = run_agent(self.WORKED_TARGET, catalog, backend)
        lower = response.answer.lower()
        assert "predictor" in lower or "whitelist" in lower or "candidate" in lower, (
            "Whitelist must be mentioned in the answer"
        )
