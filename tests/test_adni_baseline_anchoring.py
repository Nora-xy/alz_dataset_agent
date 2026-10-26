"""Regression tests for Space-B baseline biomarker measurement anchoring.

Root cause fixed here: for event_anchor=None/"baseline", aggregate_stat/
descriptive_stats previously selected each RID's EARLIEST row in the measure table
itself (by DRAWDATE/VISDATE), independent of that RID's own baseline DXSUM visit.
On real ADNI data this is wrong for the majority of participants: only ~43% of
CSF-table RIDs have their earliest draw at their own baseline VISCODE (714 of 1249
RIDs' earliest CSF draw is at a LATER visit). The fix aligns the measure row to each
RID's own baseline DXSUM visit via VISCODE match (the same cross-table mechanism
already used for diagnosis-event anchors), excluding participants with no
measure-table row at that exact visit rather than substituting a later draw.

Sections:
  A. Adversarial: a participant whose first CSF draw is strictly after their
     baseline visit must be excluded, not substituted.
  B. matched_n / unmatched_n reporting and independent suppression.
  C. source_level derived from ADSP-PHC column naming (phc_raw_column vs
     phc_harmonized), not hardcoded "raw".
  D. baseline_alignment_rule text is present and anchor-appropriate.
  E. last_visit anchor is unaffected (legitimately does not need diagnosis-visit
     alignment).
  F. Live verification against the real local backend: CN/MCI/dementia baseline
     p-tau181 are now distinct (the originally reported bug).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from alz_agent.backend.local_backend import LocalDataBackend
from alz_agent.schemas import FilterSpec, IdSpace

_ROOT = Path(__file__).resolve().parents[1]


def _make_backend() -> LocalDataBackend:
    config = yaml.safe_load(open(_ROOT / "config" / "paths.yaml", encoding="utf-8"))
    return LocalDataBackend(config, _ROOT)


def _patch_synthetic_adni(backend: LocalDataBackend, dx_df: pd.DataFrame, csf_df: pd.DataFrame) -> None:
    """Monkeypatch DXSUM and ADSP_PHC_CSF to synthetic, fully-controlled data."""
    backend._adni_core = lambda: dx_df.copy()  # type: ignore[method-assign]

    orig_resolve = backend._resolve_adni_table
    orig_load = backend._load

    def _resolve(tbl: str):
        if tbl == "ADSP_PHC_CSF":
            return "SYNTHETIC_CSF.csv"
        return orig_resolve(tbl)

    def _load(rel, usecols=None, **kwargs):
        if rel == "SYNTHETIC_CSF.csv":
            return csf_df.copy()
        return orig_load(rel, usecols=usecols, **kwargs)

    backend._resolve_adni_table = _resolve  # type: ignore[method-assign]
    backend._load = _load  # type: ignore[method-assign]


def _synthetic_cohort(n_at_baseline: int, n_after_baseline: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a synthetic DXSUM + CSF pair.

    n_at_baseline RIDs have a CSF draw exactly at their own baseline VISCODE ('bl').
    n_after_baseline RIDs ONLY have a CSF draw at a LATER visit ('m12') -- their
    baseline diagnosis visit exists in DXSUM, but no CSF row matches it.
    All RIDs are baseline_diagnosis='normal' (DIAGNOSIS=1) so they form one cohort.
    """
    rids_matched = [f"M{i}" for i in range(n_at_baseline)]
    rids_unmatched = [f"U{i}" for i in range(n_after_baseline)]
    all_rids = rids_matched + rids_unmatched

    dx_rows = []
    for rid in all_rids:
        dx_rows.append({
            "RID": rid, "VISCODE": "bl", "DIAGNOSIS": 1,
            "EXAMDATE": "2020-01-01",
        })
    dx_df = pd.DataFrame(dx_rows)

    csf_rows = []
    for i, rid in enumerate(rids_matched):
        csf_rows.append({
            "RID": rid, "VISCODE": "bl", "VISCODE2": "bl",
            "DRAWDATE": "2020-01-01", "pTau181_RAW": 20.0 + i, "PHC_pTau181": -0.5 + i * 0.01,
        })
    for i, rid in enumerate(rids_unmatched):
        # Only a LATER visit -- their baseline diagnosis visit has no matching CSF row.
        csf_rows.append({
            "RID": rid, "VISCODE": "m12", "VISCODE2": "m12",
            "DRAWDATE": "2021-01-01", "pTau181_RAW": 999.0, "PHC_pTau181": 999.0,
        })
    csf_df = pd.DataFrame(csf_rows)
    return dx_df, csf_df


# ---------------------------------------------------------------------------
# A. Adversarial: post-baseline-only draw must be excluded, not substituted
# ---------------------------------------------------------------------------


class TestAdversarialPostBaselineDrawExcluded:
    def test_rid_with_only_post_baseline_draw_is_excluded_from_analytic_n(self):
        """12 RIDs have CSF at baseline (values 20-31); 3 RIDs ONLY have CSF at
        month-12 (value 999, would badly skew the mean if wrongly included). The
        fix must exclude the 3 post-baseline-only RIDs entirely."""
        backend = _make_backend()
        dx_df, csf_df = _synthetic_cohort(n_at_baseline=12, n_after_baseline=3)
        _patch_synthetic_adni(backend, dx_df, csf_df)

        fs = FilterSpec(baseline_diagnosis=["normal"], id_space=IdSpace.B)
        result = backend.aggregate_stat("ADNI.ADSP_PHC_CSF.pTau181_RAW", "mean", fs, None)

        assert result.grounded, result.abstain_reason
        assert result.value["n"] == 12, (
            "Analytic N must be exactly the 12 baseline-matched RIDs, excluding "
            "the 3 whose only draw was at month-12."
        )
        # If the 999.0 post-baseline values had leaked in, the mean would be >> 31.
        assert result.value["value"] < 40.0, (
            f"Mean {result.value['value']} suggests the post-baseline value (999.0) "
            "was wrongly included instead of excluded."
        )

    def test_matched_and_unmatched_n_reported_correctly(self):
        backend = _make_backend()
        # unmatched count (12) deliberately kept >= SUPPRESSION_THRESHOLD (10) so
        # this test asserts the reported VALUE, not suppression -- see the separate
        # TestMatchedUnmatchedSuppression class for the <10 suppression case.
        dx_df, csf_df = _synthetic_cohort(n_at_baseline=15, n_after_baseline=12)
        _patch_synthetic_adni(backend, dx_df, csf_df)

        fs = FilterSpec(baseline_diagnosis=["normal"], id_space=IdSpace.B)
        result = backend.aggregate_stat("ADNI.ADSP_PHC_CSF.pTau181_RAW", "mean", fs, None)

        assert result.grounded
        ev = result.evidence[0]
        assert ev.matched_n == 15
        # eligible cohort = 27 (15 matched + 12 unmatched); unmatched_n = 27 - 15 = 12
        assert ev.unmatched_n == 12


# ---------------------------------------------------------------------------
# B. Independent suppression of matched_n / unmatched_n
# ---------------------------------------------------------------------------


class TestMatchedUnmatchedSuppression:
    def test_small_unmatched_n_is_suppressed_to_none(self):
        """matched_n=15 (above threshold) but unmatched_n=3 (below threshold=10)
        must independently suppress unmatched_n to None while still reporting
        matched_n and the stat itself."""
        backend = _make_backend()
        dx_df, csf_df = _synthetic_cohort(n_at_baseline=15, n_after_baseline=3)
        _patch_synthetic_adni(backend, dx_df, csf_df)

        fs = FilterSpec(baseline_diagnosis=["normal"], id_space=IdSpace.B)
        result = backend.aggregate_stat("ADNI.ADSP_PHC_CSF.pTau181_RAW", "mean", fs, None)

        assert result.grounded
        ev = result.evidence[0]
        assert ev.matched_n == 15
        assert ev.unmatched_n is None  # 3 < SUPPRESSION_THRESHOLD (10)


# ---------------------------------------------------------------------------
# C. source_level derived from ADSP-PHC naming convention
# ---------------------------------------------------------------------------


class TestSourceLevelDerivation:
    def setup_method(self):
        self.backend = _make_backend()
        dx_df, csf_df = _synthetic_cohort(n_at_baseline=12, n_after_baseline=0)
        _patch_synthetic_adni(self.backend, dx_df, csf_df)
        self.fs = FilterSpec(baseline_diagnosis=["normal"], id_space=IdSpace.B)

    def test_raw_column_labeled_phc_raw_column(self):
        result = self.backend.aggregate_stat(
            "ADNI.ADSP_PHC_CSF.pTau181_RAW", "mean", self.fs, None
        )
        assert result.grounded
        assert result.evidence[0].source_level == "phc_raw_column"

    def test_harmonized_column_labeled_phc_harmonized(self):
        result = self.backend.aggregate_stat(
            "ADNI.ADSP_PHC_CSF.PHC_pTau181", "mean", self.fs, None
        )
        assert result.grounded
        assert result.evidence[0].source_level == "phc_harmonized"

    def test_neither_source_level_is_the_old_generic_raw_label(self):
        """Neither variant should silently fall back to the old, ambiguous
        hardcoded 'raw' label — that was the reported provenance defect."""
        r1 = self.backend.aggregate_stat("ADNI.ADSP_PHC_CSF.pTau181_RAW", "mean", self.fs, None)
        r2 = self.backend.aggregate_stat("ADNI.ADSP_PHC_CSF.PHC_pTau181", "mean", self.fs, None)
        assert r1.evidence[0].source_level != "raw"
        assert r2.evidence[0].source_level != "raw"


# ---------------------------------------------------------------------------
# D. baseline_alignment_rule text
# ---------------------------------------------------------------------------


class TestBaselineAlignmentRuleText:
    def test_alignment_rule_present_and_describes_exclusion_not_substitution(self):
        backend = _make_backend()
        dx_df, csf_df = _synthetic_cohort(n_at_baseline=12, n_after_baseline=3)
        _patch_synthetic_adni(backend, dx_df, csf_df)
        fs = FilterSpec(baseline_diagnosis=["normal"], id_space=IdSpace.B)
        result = backend.aggregate_stat("ADNI.ADSP_PHC_CSF.pTau181_RAW", "mean", fs, None)

        assert result.grounded
        rule = result.evidence[0].baseline_alignment_rule
        assert rule is not None
        assert "baseline DXSUM visit" in rule
        assert "excluded" in rule.lower()
        assert "never" in rule.lower() or "not" in rule.lower()


# ---------------------------------------------------------------------------
# E. last_visit anchor is unaffected
# ---------------------------------------------------------------------------


class TestLastVisitUnaffected:
    def test_last_visit_still_picks_latest_row_in_measure_table(self):
        """last_visit legitimately does NOT need diagnosis-visit alignment — it
        explicitly asks for the measure table's own latest row per RID. The
        post-baseline-only RIDs' month-12 draws ARE their last visit, so they
        should be included here (unlike the baseline case)."""
        backend = _make_backend()
        dx_df, csf_df = _synthetic_cohort(n_at_baseline=12, n_after_baseline=3)
        _patch_synthetic_adni(backend, dx_df, csf_df)
        fs = FilterSpec(baseline_diagnosis=["normal"], id_space=IdSpace.B)

        result = backend.aggregate_stat(
            "ADNI.ADSP_PHC_CSF.pTau181_RAW", "mean", fs, "last_visit"
        )
        assert result.grounded
        # All 15 RIDs have exactly one row each, so last_visit sees all 15.
        assert result.value["n"] == 15


# ---------------------------------------------------------------------------
# F. Live verification against the real local backend
# ---------------------------------------------------------------------------


class TestRealBackendGroupsAreDistinct:
    """Requires real local ADNI data (skipped if unavailable)."""

    def setup_method(self):
        config_path = _ROOT / "config" / "paths.yaml"
        if not config_path.exists():
            pytest.skip("config/paths.yaml not present")
        self.backend = _make_backend()

    def test_cn_mci_dementia_baseline_ptau181_are_distinct(self):
        try:
            results = {}
            for dx in ("normal", "mci", "dementia"):
                fs = FilterSpec(baseline_diagnosis=[dx], id_space=IdSpace.B)
                r = self.backend.aggregate_stat(
                    "ADNI.ADSP_PHC_CSF.pTau181_RAW", "mean", fs, None
                )
                if not r.grounded:
                    pytest.skip(f"real ADNI data not mounted or insufficient: {r.abstain_reason}")
                results[dx] = r.value["n"]
        except FileNotFoundError:
            pytest.skip("real ADNI data files not present")

        # The originally reported bug: CN and MCI both showed n=1245 (identical).
        # After the fix, they must be distinct real baseline-anchored counts.
        assert len(set(results.values())) == 3, (
            f"Expected 3 distinct analytic-N values for CN/MCI/dementia, got {results}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
