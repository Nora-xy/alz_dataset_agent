"""Executor routing tests — DeterministicExecutor._build_filter_spec.

Verifies that etiology in an op filter is routed to the correct FilterSpec field:
  - filter {"etiology": "ad"} + event_anchor set (non-baseline) → event_etiology
  - filter {"event_etiology": "ad"} → event_etiology (explicit)
  - filter {"etiology": "ad"} without event_anchor → baseline_etiology (first-visit only)

These tests guard the fix for the routing bug where the enriched-planner path
always used baseline_etiology regardless of event_anchor, producing n=19,576
(prevalent-at-enrollment only) instead of the correct n=25,639 (any-visit).
"""

from __future__ import annotations

from pathlib import Path
import math
import pytest

from alz_agent.agent.executor import DeterministicExecutor
from alz_agent.backend.fixture_backend import (
    FixtureDataBackend,
    FixtureParticipant,
    FixtureVisit,
)
from alz_agent.schemas import (
    CatalogEntry,
    EnrichedAnalysisPlan,
    FilterSpec,
    IdSpace,
    LeakageTag,
    Modality,
    PlanOperation,
    SourceDataset,
)

PROJECT_ROOT = Path(__file__).parent.parent
NACC_CSV = PROJECT_ROOT / "NACC" / "investigator_nacc70.csv"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _minimal_plan(id_space: str = "A") -> EnrichedAnalysisPlan:
    return EnrichedAnalysisPlan(
        original_question="test",
        research_intent_summary="test plan",
        operations=[],
        id_space=id_space,
    )


def _naccage_catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            variable_id="NACC.NACCAGE",
            concept="age at visit",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCAGE",
            release_version="nacc70",
            modality=Modality.DEMOGRAPHIC,
            leakage_tags=[LeakageTag.NONE],
        ),
    ]


def _abc_backend() -> tuple[
    FixtureDataBackend,
    list[FixtureParticipant],
    list[FixtureParticipant],
    list[FixtureParticipant],
]:
    """Three groups for baseline vs. any-visit etiology testing.

    Group A (12): NACCALZD==1 at visit 1 (baseline AD), age=70.
    Group B (12): NACCALZD==0 at visit 1; NACCALZD==1 at visit 3 (incident AD), age at visit 3 = 67.
    Group C (12): never NACCALZD==1, excluded by any etiology filter.
    """
    n = 12
    group_a = [
        FixtureParticipant(
            pid=f"EA{i:02d}",
            visits=[FixtureVisit(vnum=1, year=2010, diagnosis=3)],
            modalities=frozenset({"cognition"}),
            baseline_etiology="ad",
            age_at_baseline=70,
            ad_etiology_first_vnum=1,
            age_at_first_ad_etiology=70,
        )
        for i in range(n)
    ]
    group_b = [
        FixtureParticipant(
            pid=f"EB{i:02d}",
            visits=[
                FixtureVisit(vnum=1, year=2010, diagnosis=3),
                FixtureVisit(vnum=2, year=2011, diagnosis=3),
                FixtureVisit(vnum=3, year=2012, diagnosis=4),
            ],
            modalities=frozenset({"cognition"}),
            baseline_etiology=None,
            age_at_baseline=65,
            ad_etiology_first_vnum=3,
            age_at_first_ad_etiology=67,
        )
        for i in range(n)
    ]
    group_c = [
        FixtureParticipant(
            pid=f"EC{i:02d}",
            visits=[FixtureVisit(vnum=1, year=2010, diagnosis=1)],
            modalities=frozenset(),
            baseline_etiology=None,
            age_at_baseline=72,
            ad_etiology_first_vnum=None,
            age_at_first_ad_etiology=None,
        )
        for i in range(n)
    ]
    backend = FixtureDataBackend(participants=group_a + group_b + group_c, id_space=IdSpace.A)
    return backend, group_a, group_b, group_c


# ---------------------------------------------------------------------------
# Unit tests: _build_filter_spec routing
# ---------------------------------------------------------------------------

class TestBuildFilterSpecRouting:
    """Unit tests for DeterministicExecutor._build_filter_spec.

    Tests that etiology is routed to the correct FilterSpec field based on
    whether an event_anchor is present in op_params.
    """

    def setup_method(self):
        # DeterministicExecutor only needs catalog and backend for _build_filter_spec
        backend, _, _, _ = _abc_backend()
        self._executor = DeterministicExecutor([], backend)
        self._plan = _minimal_plan()

    def test_event_anchor_first_diagnosis_promotes_etiology_to_event_etiology(self):
        """etiology + event_anchor='first_diagnosis' → event_etiology (auto-promotion)."""
        params = {
            "variable_id": "NACC.NACCAGE",
            "stat_type": "mean",
            "event_anchor": "first_diagnosis",
            "filter": {"etiology": "ad"},
        }
        fs = self._executor._build_filter_spec(self._plan, params)
        assert fs.event_etiology == ["ad"], "should auto-promote to event_etiology"
        assert fs.baseline_etiology is None, "baseline_etiology must be None after promotion"

    def test_explicit_event_etiology_key_in_filter(self):
        """filter={event_etiology: 'ad'} → event_etiology (explicit, no event_anchor needed)."""
        params = {"filter": {"event_etiology": "ad"}}
        fs = self._executor._build_filter_spec(self._plan, params)
        assert fs.event_etiology == ["ad"]
        assert fs.baseline_etiology is None

    def test_etiology_without_event_anchor_stays_baseline(self):
        """filter={etiology: 'ad'} + no event_anchor → baseline_etiology (unchanged behavior)."""
        params = {"filter": {"etiology": "ad"}}
        fs = self._executor._build_filter_spec(self._plan, params)
        assert fs.baseline_etiology == ["ad"]
        assert fs.event_etiology is None

    def test_event_anchor_baseline_does_not_promote(self):
        """event_anchor='baseline' with etiology → baseline_etiology (not promoted)."""
        params = {
            "variable_id": "NACC.NACCAGE",
            "stat_type": "mean",
            "event_anchor": "baseline",
            "filter": {"etiology": "ad"},
        }
        fs = self._executor._build_filter_spec(self._plan, params)
        assert fs.baseline_etiology == ["ad"]
        assert fs.event_etiology is None

    def test_event_anchor_last_visit_promotes_etiology(self):
        """event_anchor='last_visit' with etiology → event_etiology (any non-baseline anchor)."""
        params = {
            "event_anchor": "last_visit",
            "filter": {"etiology": "ad"},
        }
        fs = self._executor._build_filter_spec(self._plan, params)
        assert fs.event_etiology == ["ad"]
        assert fs.baseline_etiology is None

    def test_diagnosis_filter_unchanged(self):
        """diagnosis filter always maps to baseline_diagnosis regardless of event_anchor."""
        params = {
            "event_anchor": "first_diagnosis",
            "filter": {"diagnosis": "mci"},
        }
        fs = self._executor._build_filter_spec(self._plan, params)
        assert fs.baseline_diagnosis == ["mci"]
        assert fs.event_etiology is None
        assert fs.baseline_etiology is None

    def test_empty_filter_falls_back_to_plan_baseline_definition(self):
        """When op_params has no 'filter', plan.baseline_definition is used."""
        plan_with_def = EnrichedAnalysisPlan(
            original_question="test",
            research_intent_summary="test",
            operations=[],
            id_space="A",
            baseline_definition={"diagnosis": "mci"},
        )
        params = {}  # no filter key → falls back to plan.baseline_definition
        fs = self._executor._build_filter_spec(plan_with_def, params)
        assert fs.baseline_diagnosis == ["mci"]


# ---------------------------------------------------------------------------
# Integration tests: executor dispatch with fixture backend
# ---------------------------------------------------------------------------

class TestExecutorDispatchRouting:
    """Integration tests: executor dispatches count_eligible and variable_stat
    correctly for baseline_etiology vs event_etiology.
    """

    def setup_method(self):
        self._backend, self._ga, self._gb, self._gc = _abc_backend()
        self._catalog = _naccage_catalog()
        self._executor = DeterministicExecutor(self._catalog, self._backend)

    def _make_plan(self, ops: list[PlanOperation]) -> EnrichedAnalysisPlan:
        return EnrichedAnalysisPlan(
            original_question="test",
            research_intent_summary="test",
            operations=ops,
            id_space="A",
        )

    def test_count_eligible_explicit_event_etiology_returns_a_and_b(self):
        """count_eligible with filter={event_etiology:'ad'} counts A+B (24), excludes C."""
        op = PlanOperation(
            op_type="count_eligible",
            parameters={"filter": {"event_etiology": "ad"}},
            output_name="ad_count",
        )
        ctx = self._executor.execute(self._make_plan([op]))
        r = ctx.results["ad_count"]
        assert r.grounded, r.abstain_reason
        assert r.value == len(self._ga) + len(self._gb), (
            f"event_etiology count should be {len(self._ga) + len(self._gb)}, got {r.value}"
        )

    def test_count_eligible_baseline_etiology_returns_only_a(self):
        """count_eligible with filter={etiology:'ad'} (no event_anchor) returns only A (12)."""
        op = PlanOperation(
            op_type="count_eligible",
            parameters={"filter": {"etiology": "ad"}},
            output_name="ad_baseline_count",
        )
        ctx = self._executor.execute(self._make_plan([op]))
        r = ctx.results["ad_baseline_count"]
        assert r.grounded, r.abstain_reason
        assert r.value == len(self._ga), (
            f"baseline_etiology count should be {len(self._ga)}, got {r.value}"
        )

    def test_variable_stat_event_anchor_auto_promotes_etiology(self):
        """variable_stat NACCAGE with event_anchor='first_diagnosis'+etiology:'ad' uses
        auto-promoted event_etiology: includes A+B, ages taken at first NACCALZD==1 visit.

        Expected mean: (12*70 + 12*67) / 24 = 68.5
        """
        op = PlanOperation(
            op_type="variable_stat",
            parameters={
                "variable_id": "NACC.NACCAGE",
                "stat_type": "mean",
                "event_anchor": "first_diagnosis",
                "filter": {"etiology": "ad"},
            },
            output_name="mean_age_first_ad",
        )
        ctx = self._executor.execute(self._make_plan([op]))
        r = ctx.results["mean_age_first_ad"]
        assert r.grounded, r.abstain_reason
        stat = r.value  # StatResult
        assert stat is not None
        assert stat.n == len(self._ga) + len(self._gb), (
            f"n should be {len(self._ga) + len(self._gb)} (A+B), got {stat.n}"
        )
        expected_mean = (12 * 70 + 12 * 67) / 24
        assert abs(stat.value - expected_mean) < 0.01, (
            f"mean age should be {expected_mean:.2f}, got {stat.value}"
        )

    def test_event_etiology_count_exceeds_baseline_etiology_count(self):
        """Sanity: event_etiology always includes at least as many participants as baseline."""
        op_event = PlanOperation(
            op_type="count_eligible",
            parameters={"filter": {"event_etiology": "ad"}},
            output_name="n_event",
        )
        op_base = PlanOperation(
            op_type="count_eligible",
            parameters={"filter": {"etiology": "ad"}},
            output_name="n_base",
        )
        ctx = self._executor.execute(self._make_plan([op_event, op_base]))
        n_event = ctx.results["n_event"].value
        n_base = ctx.results["n_base"].value
        assert n_event > n_base, f"event({n_event}) must exceed baseline({n_base})"

    def test_n_is_unique_participant_count_not_row_count(self):
        """n from count_eligible must equal unique participant count, not row count."""
        op = PlanOperation(
            op_type="count_eligible",
            parameters={"filter": {"event_etiology": "ad"}},
            output_name="ad_n",
        )
        ctx = self._executor.execute(self._make_plan([op]))
        r = ctx.results["ad_n"]
        assert r.grounded
        # Group A has 1 visit each, group B has 3 visits each.
        # Row count would be 12*1 + 12*3 = 48; participant count = 24.
        assert r.value == 24, f"n must be participant count 24, not row count, got {r.value}"


# ---------------------------------------------------------------------------
# Real-data regression (skipped unless NACC CSV is present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not NACC_CSV.exists(),
    reason="NACC CSV not available in this environment",
)
class TestExecutorRoutingRealData:
    """Real-data regression for executor routing using the local NACC CSV.

    Guards the correct values from the audit:
      - baseline_etiology (first-visit only):  n=19,576, mean_age≈73.64
      - event_etiology (any-visit, first-diagnosis): n=25,639, mean_age≈74.81
    """

    def setup_method(self):
        import yaml
        from alz_agent.backend.local_backend import LocalDataBackend

        paths_yaml = PROJECT_ROOT / "config" / "paths.yaml"
        with open(paths_yaml, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self._backend = LocalDataBackend(config, PROJECT_ROOT)
        self._catalog = _naccage_catalog()
        self._executor = DeterministicExecutor(self._catalog, self._backend)

    def _make_plan(self, ops: list[PlanOperation]) -> EnrichedAnalysisPlan:
        return EnrichedAnalysisPlan(
            original_question="test",
            research_intent_summary="test",
            operations=ops,
            id_space="A",
        )

    def test_event_etiology_count_eligible_is_25639(self):
        """Any-visit NACCALZD==1 participant count must be 25,639."""
        op = PlanOperation(
            op_type="count_eligible",
            parameters={"filter": {"event_etiology": "ad"}},
            output_name="n_event",
        )
        ctx = self._executor.execute(self._make_plan([op]))
        r = ctx.results["n_event"]
        assert r.grounded, r.abstain_reason
        assert r.value == 25639, f"expected 25639, got {r.value}"

    def test_baseline_etiology_count_eligible_is_19576(self):
        """First-visit-only NACCALZD==1 count must be 19,576."""
        op = PlanOperation(
            op_type="count_eligible",
            parameters={"filter": {"etiology": "ad"}},
            output_name="n_base",
        )
        ctx = self._executor.execute(self._make_plan([op]))
        r = ctx.results["n_base"]
        assert r.grounded, r.abstain_reason
        assert r.value == 19576, f"expected 19576, got {r.value}"

    def test_mean_age_at_first_diagnosis_event_etiology_is_74_81(self):
        """Mean NACCAGE at first NACCALZD==1 visit per NACCID must be ≈74.81 (any-visit)."""
        op = PlanOperation(
            op_type="variable_stat",
            parameters={
                "variable_id": "NACC.NACCAGE",
                "stat_type": "mean",
                "event_anchor": "first_diagnosis",
                "filter": {"etiology": "ad"},  # auto-promoted by event_anchor
            },
            output_name="mean_age_first_dx",
        )
        ctx = self._executor.execute(self._make_plan([op]))
        r = ctx.results["mean_age_first_dx"]
        assert r.grounded, r.abstain_reason
        stat = r.value  # StatResult
        assert stat.n == 25639, f"n must be 25639 (any-visit), got {stat.n}"
        assert abs(stat.value - 74.81) < 0.05, f"mean age must be ≈74.81, got {stat.value}"

    def test_mean_age_at_first_diagnosis_baseline_etiology_is_73_64(self):
        """Mean NACCAGE at baseline for prevalent-AD participants must be ≈73.64."""
        op = PlanOperation(
            op_type="variable_stat",
            parameters={
                "variable_id": "NACC.NACCAGE",
                "stat_type": "mean",
                "event_anchor": "baseline",
                "filter": {"etiology": "ad"},
            },
            output_name="mean_age_baseline",
        )
        ctx = self._executor.execute(self._make_plan([op]))
        r = ctx.results["mean_age_baseline"]
        assert r.grounded, r.abstain_reason
        stat = r.value  # StatResult
        assert stat.n == 19576, f"n must be 19576 (baseline-only), got {stat.n}"
        assert abs(stat.value - 73.64) < 0.05, f"mean age must be ≈73.64, got {stat.value}"

    def test_n_is_participant_count_not_row_count(self):
        """n from event_etiology count must not equal total rows (204031) or visits."""
        op = PlanOperation(
            op_type="count_eligible",
            parameters={"filter": {"event_etiology": "ad"}},
            output_name="n",
        )
        ctx = self._executor.execute(self._make_plan([op]))
        r = ctx.results["n"]
        assert r.grounded
        assert r.value != 204031, "n must be participant count, not total row count"
        assert r.value != 19576, "event_etiology must not give prevalence-only count"
