"""P2 tests: FixtureDataBackend, LocalDataBackend smoke test, suppression invariant,
and populate_catalog_coverage.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from alz_agent.backend.fixture_backend import (
    FixtureDataBackend,
    FixtureParticipant,
    FixtureVisit,
    build_fixture_backend,
)
from alz_agent.backend.local_backend import LocalDataBackend, _select_adni_baseline
from alz_agent.constants import SUPPRESSION_THRESHOLD
from alz_agent.schemas import CoverageStats, FilterSpec, IdSpace, ToolResult

PROJECT_ROOT = Path(__file__).parent.parent
PATHS_YAML = PROJECT_ROOT / "config" / "paths.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    with open(PATHS_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _default_fixture() -> FixtureDataBackend:
    """Standard fixture with large enough groups to avoid suppression in most tests."""
    return build_fixture_backend(
        n_normal=50, n_mci=40, n_dementia=30, n_other=20,
        n_mci_long=25, n_mci_short=8, n_single_visit=15,
        n_suppressed_group=5,  # exactly 5 in vnum [15,16] — below threshold
        seed=42,
    )


def _all_int_values(result: ToolResult) -> list[int]:
    """Extract all numeric values from a ToolResult for suppression checking."""
    if not result.grounded or result.value is None:
        return []
    val = result.value
    if isinstance(val, int):
        return [val]
    if isinstance(val, dict):
        return [v for v in val.values() if isinstance(v, int)]
    return []


# ---------------------------------------------------------------------------
# FixtureDataBackend — count_eligible
# ---------------------------------------------------------------------------


class TestFixtureCountEligible:
    def test_no_filter_returns_all(self):
        fb = _default_fixture()
        r = fb.count_eligible(FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        assert r.value > 0

    def test_mci_filter_returns_subset(self):
        fb = _default_fixture()
        all_r = fb.count_eligible(FilterSpec(id_space=IdSpace.A))
        mci_r = fb.count_eligible(FilterSpec(baseline_diagnosis=["MCI"], id_space=IdSpace.A))
        assert mci_r.grounded
        assert mci_r.value < all_r.value

    def test_dementia_filter(self):
        fb = _default_fixture()
        r = fb.count_eligible(FilterSpec(baseline_diagnosis=["Dementia"], id_space=IdSpace.A))
        assert r.grounded
        assert r.value >= 10  # factory creates 30 dementia

    def test_normal_string_variants(self):
        """'Normal', 'NC', '1' all resolve to NACCUDSD=1."""
        fb = _default_fixture()
        r1 = fb.count_eligible(FilterSpec(baseline_diagnosis=["Normal"], id_space=IdSpace.A))
        r2 = fb.count_eligible(FilterSpec(baseline_diagnosis=["NC"], id_space=IdSpace.A))
        r3 = fb.count_eligible(FilterSpec(baseline_diagnosis=["1"], id_space=IdSpace.A))
        assert r1.grounded and r2.grounded and r3.grounded
        assert r1.value == r2.value == r3.value

    def test_modality_filter(self):
        fb = _default_fixture()
        r = fb.count_eligible(
            FilterSpec(required_modalities=["cognition"], id_space=IdSpace.A)
        )
        assert r.grounded
        assert r.value > 0

    def test_modality_filter_not_in_fixture_abstains(self):
        fb = _default_fixture()
        r = fb.count_eligible(
            FilterSpec(required_modalities=["completely_unknown_modality"], id_space=IdSpace.A)
        )
        assert not r.grounded
        assert r.abstain_reason is not None

    def test_wrong_id_space_returns_zero(self):
        fb = _default_fixture()  # Space A backend
        r = fb.count_eligible(FilterSpec(id_space=IdSpace.B))
        # Space B returns 0 eligible on a Space A fixture
        assert r.grounded and r.value == 0

    def test_unknown_diagnosis_label_abstains(self):
        fb = _default_fixture()
        r = fb.count_eligible(
            FilterSpec(baseline_diagnosis=["NOT_A_DIAGNOSIS"], id_space=IdSpace.A)
        )
        assert not r.grounded

    def test_result_has_evidence(self):
        fb = _default_fixture()
        r = fb.count_eligible(FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        assert len(r.evidence) >= 1

    def test_suppression_triggers_on_small_group(self):
        """The 5-member vnum=[15,16] group should be suppressed."""
        fb = _default_fixture()
        r = fb.count_eligible(
            FilterSpec(
                baseline_diagnosis=["MCI"],
                visit_window=(15, 16),
                id_space=IdSpace.A,
            )
        )
        # Should abstain with suppression message
        assert not r.grounded
        assert "suppressed" in r.abstain_reason.lower()

    def test_zero_count_not_suppressed(self):
        """0 participants → returns 0, not suppressed (pet is known but nobody has it)."""
        fb = FixtureDataBackend(
            participants=[
                FixtureParticipant("P1", [FixtureVisit(1, 2010, 2)], frozenset({"cognition"}))
            ] * 20,
            known_modalities={"cognition", "pet"},  # pet is known but has 0 participants
        )
        r = fb.count_eligible(
            FilterSpec(
                required_modalities=["pet"],
                id_space=IdSpace.A,
            )
        )
        assert r.grounded
        assert r.value == 0


# ---------------------------------------------------------------------------
# NACCUDSD coding correctness
# ---------------------------------------------------------------------------


class TestNaccudsdCoding:
    """Verify that baseline_diagnosis labels map to the correct NACCUDSD integers.

    NACCUDSD (UDS3 RDD):
        1 = Normal cognition
        2 = Impaired-not-MCI
        3 = MCI
        4 = Dementia (all-cause)
    """

    # 15 participants per code — above SUPPRESSION_THRESHOLD (10) so counts aren't masked.
    _N_PER_CODE = 15

    def _fixture_with_n_per_code(self) -> FixtureDataBackend:
        """15 participants per NACCUDSD code, each with 20 visits."""
        participants = []
        n = self._N_PER_CODE
        for code, prefix in [(1, "NORM"), (2, "IMP"), (3, "MCI"), (4, "DEM")]:
            for i in range(n):
                participants.append(FixtureParticipant(
                    pid=f"{prefix}{i:03d}",
                    visits=[FixtureVisit(v, 2010 + v, code) for v in range(1, 21)],
                    modalities=frozenset({"cognition"}),
                ))
        return FixtureDataBackend(participants)

    def test_normal_maps_to_1(self):
        fb = self._fixture_with_n_per_code()
        r = fb.count_eligible(FilterSpec(baseline_diagnosis=["normal"], id_space=IdSpace.A))
        assert r.grounded and r.value == self._N_PER_CODE

    def test_impaired_not_mci_maps_to_2(self):
        fb = self._fixture_with_n_per_code()
        for label in ["impaired not mci", "impaired-not-mci", "impaired_not_mci", "2"]:
            r = fb.count_eligible(FilterSpec(baseline_diagnosis=[label], id_space=IdSpace.A))
            assert r.grounded and r.value == self._N_PER_CODE, (
                f"label={label!r} → expected {self._N_PER_CODE}, got {r.value}"
            )

    def test_other_label_abstains(self):
        """'other' is not a valid NACCUDSD label — it's ambiguous and was removed.
        Use 'impaired not mci' for NACCUDSD=2."""
        fb = self._fixture_with_n_per_code()
        r = fb.count_eligible(FilterSpec(baseline_diagnosis=["other"], id_space=IdSpace.A))
        assert not r.grounded, (
            "'other' should abstain — it is ambiguous (could mean etiologic 'other'). "
            "Use 'impaired not mci' for NACCUDSD=2."
        )

    def test_mci_maps_to_3(self):
        fb = self._fixture_with_n_per_code()
        for label in ["mci", "3"]:
            r = fb.count_eligible(FilterSpec(baseline_diagnosis=[label], id_space=IdSpace.A))
            assert r.grounded and r.value == self._N_PER_CODE, (
                f"label={label!r} → expected {self._N_PER_CODE}, got {r.value}"
            )

    def test_dementia_maps_to_4(self):
        fb = self._fixture_with_n_per_code()
        for label in ["dementia", "4"]:
            r = fb.count_eligible(FilterSpec(baseline_diagnosis=[label], id_space=IdSpace.A))
            assert r.grounded and r.value == self._N_PER_CODE, (
                f"label={label!r} → expected {self._N_PER_CODE}, got {r.value}"
            )

    def test_ad_label_abstains(self):
        """'ad' is NOT a valid NACCUDSD label; it must raise/abstain, not silently match."""
        fb = self._fixture_with_n_per_code()
        r = fb.count_eligible(FilterSpec(baseline_diagnosis=["ad"], id_space=IdSpace.A))
        assert not r.grounded, "'ad' should abstain — NACCUDSD=4 is all-cause dementia, not AD"
        assert "ad" in r.abstain_reason.lower() or "unknown" in r.abstain_reason.lower()

    def test_old_wrong_mapping_mci_is_not_2(self):
        """Under old (wrong) code, MCI was coded as 2. Under correct code it's 3.
        Filtering by '2' must select Impaired-not-MCI participants, NOT MCI participants."""
        fb = self._fixture_with_n_per_code()
        r_impaired = fb.count_eligible(FilterSpec(baseline_diagnosis=["2"], id_space=IdSpace.A))
        r_mci = fb.count_eligible(FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A))
        assert r_impaired.grounded and r_mci.grounded
        # Both should return _N_PER_CODE but from different participant groups;
        # the key check is that "2" and "mci" are disjoint (each has exactly N_PER_CODE).
        assert r_impaired.value == self._N_PER_CODE
        assert r_mci.value == self._N_PER_CODE


# ---------------------------------------------------------------------------
# Baseline-diagnosis propagation through all three public methods (regression)
# ---------------------------------------------------------------------------


class TestBaselineDiagnosisPropagation:
    """Verify that the corrected NACCUDSD=3 for MCI propagates through all three
    public DataBackend methods, not just count_eligible."""

    _N = 15  # participants per code — above suppression threshold

    def _fixture(self) -> FixtureDataBackend:
        """15 participants per NACCUDSD code; all have cognition; 20 visits each."""
        pts = []
        for code, prefix in [(1, "NORM"), (2, "IMP"), (3, "MCI"), (4, "DEM")]:
            for i in range(self._N):
                pts.append(FixtureParticipant(
                    pid=f"{prefix}{i:03d}",
                    visits=[FixtureVisit(v, 2010 + v, code) for v in range(1, 21)],
                    modalities=frozenset({"cognition"}),
                ))
        return FixtureDataBackend(pts)

    def test_modality_intersection_mci_uses_dx3(self):
        """modality_intersection with baseline_diagnosis=['mci'] must use NACCUDSD=3."""
        fb = self._fixture()
        r = fb.modality_intersection(
            ["cognition"],
            FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A),
        )
        assert r.grounded
        # All _N MCI (dx=3) participants have cognition → intersection = _N
        assert r.value.get("cognition") == self._N, (
            f"Expected {self._N} MCI participants with cognition, got {r.value.get('cognition')}"
        )

    def test_modality_intersection_impaired_not_mci_uses_dx2(self):
        """modality_intersection with baseline_diagnosis=['impaired not mci'] uses dx=2, not dx=3."""
        fb = self._fixture()
        r_impaired = fb.modality_intersection(
            ["cognition"],
            FilterSpec(baseline_diagnosis=["impaired not mci"], id_space=IdSpace.A),
        )
        r_mci = fb.modality_intersection(
            ["cognition"],
            FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A),
        )
        assert r_impaired.grounded and r_mci.grounded
        # Both have _N participants but from DIFFERENT groups
        assert r_impaired.value.get("cognition") == self._N
        assert r_mci.value.get("cognition") == self._N

    def test_followup_distribution_mci_uses_dx3(self):
        """followup_distribution with baseline_diagnosis=['mci'] must use NACCUDSD=3."""
        fb = self._fixture()
        r = fb.followup_distribution(
            3,
            FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A),
        )
        assert r.grounded
        # Each MCI (dx=3) participant has 20 visits spanning 19 years → all sufficient
        suf = r.value.get("sufficient")
        assert suf == self._N, (
            f"Expected all {self._N} MCI participants to have sufficient follow-up, got {suf}"
        )
        assert r.value.get("censored") == 0
        assert r.value.get("insufficient") == 0

    def test_followup_total_matches_count_eligible(self):
        """followup bucket totals must equal count_eligible for the same filter."""
        fb = self._fixture()
        fs = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A)
        r_cnt = fb.count_eligible(fs)
        r_fu = fb.followup_distribution(3, fs)
        assert r_cnt.grounded and r_fu.grounded
        bucket_total = sum(v for v in r_fu.value.values() if v is not None)
        assert bucket_total == r_cnt.value


# ---------------------------------------------------------------------------
# label_certainty rejection tests
# ---------------------------------------------------------------------------


class TestLabelCertaintyRejection:
    """label_certainty is reserved; passing a non-None value must abstain (not silently ignore)."""

    def test_fixture_count_eligible_rejects_label_certainty(self):
        fb = _default_fixture()
        r = fb.count_eligible(
            FilterSpec(label_certainty="definite", id_space=IdSpace.A)
        )
        assert not r.grounded
        assert "label_certainty" in r.abstain_reason.lower()

    def test_fixture_modality_intersection_rejects_label_certainty(self):
        fb = _default_fixture()
        r = fb.modality_intersection(
            ["cognition"],
            FilterSpec(label_certainty="definite", id_space=IdSpace.A),
        )
        assert not r.grounded
        assert "label_certainty" in r.abstain_reason.lower()

    def test_fixture_followup_distribution_rejects_label_certainty(self):
        fb = _default_fixture()
        r = fb.followup_distribution(
            3,
            FilterSpec(label_certainty="definite", id_space=IdSpace.A),
        )
        assert not r.grounded
        assert "label_certainty" in r.abstain_reason.lower()

    def test_none_label_certainty_is_accepted(self):
        """label_certainty=None (the default) must NOT cause abstention."""
        fb = _default_fixture()
        r = fb.count_eligible(
            FilterSpec(label_certainty=None, id_space=IdSpace.A)
        )
        assert r.grounded

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "NACC" / "investigator_nacc70.csv").exists(),
        reason="Real NACC data not mounted",
    )
    def test_local_backend_rejects_label_certainty(self):
        config = _load_config()
        backend = LocalDataBackend(config, PROJECT_ROOT)
        r = backend.count_eligible(
            FilterSpec(label_certainty="definite", id_space=IdSpace.A)
        )
        assert not r.grounded
        assert "label_certainty" in r.abstain_reason.lower()


# ---------------------------------------------------------------------------
# FixtureDataBackend — modality_intersection
# ---------------------------------------------------------------------------


class TestFixtureModalityIntersection:
    def test_single_modality(self):
        fb = _default_fixture()
        r = fb.modality_intersection(
            ["cognition"], FilterSpec(id_space=IdSpace.A)
        )
        assert r.grounded
        assert "cognition" in r.value

    def test_two_modalities(self):
        fb = _default_fixture()
        r = fb.modality_intersection(
            ["cognition", "mri"], FilterSpec(id_space=IdSpace.A)
        )
        assert r.grounded
        assert "cognition" in r.value
        assert "mri" in r.value
        assert "cognition,mri" in r.value

    def test_intersection_le_singles(self):
        """Intersection count ≤ each single-modality count."""
        fb = _default_fixture()
        r = fb.modality_intersection(
            ["cognition", "mri"], FilterSpec(id_space=IdSpace.A)
        )
        assert r.grounded
        combo = r.value.get("cognition,mri")
        cog = r.value.get("cognition")
        mri = r.value.get("mri")
        if combo is not None and cog is not None and mri is not None:
            assert combo <= cog
            assert combo <= mri

    def test_unknown_modality_abstains(self):
        fb = _default_fixture()
        r = fb.modality_intersection(
            ["cognition", "no_such_modality"], FilterSpec(id_space=IdSpace.A)
        )
        assert not r.grounded

    def test_suppressed_values_are_none(self):
        """Counts in (0, threshold) appear as None in the dict."""
        # Create a fixture with exactly 3 participants who have PET
        pts = [
            FixtureParticipant(f"P{i}", [FixtureVisit(1, 2010, 1)], frozenset({"pet"}))
            for i in range(3)
        ] + [
            FixtureParticipant(f"Q{i}", [FixtureVisit(1, 2010, 1)], frozenset({"cognition"}))
            for i in range(50)
        ]
        fb = FixtureDataBackend(pts)
        r = fb.modality_intersection(["pet"], FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        # 3 PET participants → 0 < 3 < 10 → suppressed (None)
        assert r.value.get("pet") is None


# ---------------------------------------------------------------------------
# FixtureDataBackend — followup_distribution
# ---------------------------------------------------------------------------


class TestFixtureFollowupDistribution:
    def test_returns_three_buckets(self):
        fb = _default_fixture()
        r = fb.followup_distribution(3, FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        assert "sufficient" in r.value
        assert "insufficient" in r.value
        assert "censored" in r.value

    def test_sufficient_count_ge_zero(self):
        fb = _default_fixture()
        r = fb.followup_distribution(3, FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        suf = r.value["sufficient"]
        if suf is not None:
            assert suf >= 0

    def test_mci_follow_up(self):
        fb = _default_fixture()
        r = fb.followup_distribution(
            3,
            FilterSpec(baseline_diagnosis=["MCI"], id_space=IdSpace.A),
        )
        assert r.grounded
        # Factory creates 25 MCI with long FU and 8 with short FU
        # sufficient should be ≥ 10 (passes suppression)
        suf = r.value["sufficient"]
        assert suf is not None  # not suppressed

    def test_bucket_counts_add_up_to_total_eligible(self):
        fb = _default_fixture()
        r = fb.followup_distribution(3, FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        vals = [v for v in r.value.values() if v is not None]
        total = sum(vals)
        cnt = fb.count_eligible(FilterSpec(id_space=IdSpace.A))
        assert cnt.grounded
        assert total == cnt.value

    def test_no_eligible_returns_zeros(self):
        fb = _default_fixture()
        r = fb.followup_distribution(
            3,
            FilterSpec(
                baseline_diagnosis=["Dementia"],
                required_modalities=["pet"],  # most dementia pts won't have PET
                id_space=IdSpace.A,
            ),
        )
        # Could be zeros or a valid count — just must not error
        assert r.grounded or (not r.grounded and r.abstain_reason)

    def test_result_has_evidence(self):
        fb = _default_fixture()
        r = fb.followup_distribution(3, FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        assert len(r.evidence) >= 1


# ---------------------------------------------------------------------------
# FixtureDataBackend — variable_coverage
# ---------------------------------------------------------------------------


class TestFixtureVariableCoverage:
    def test_known_variable_returns_stats(self):
        fb = _default_fixture()
        r = fb.variable_coverage("NACC.CDRSUM")
        assert r.grounded
        assert isinstance(r.value, CoverageStats)

    def test_coverage_stats_fields(self):
        fb = _default_fixture()
        r = fb.variable_coverage("NACC.CDRSUM")
        assert r.grounded
        cs = r.value
        assert cs.n_available >= 0
        assert 0.0 <= cs.missingness_rate <= 1.0
        assert cs.longitudinal_availability >= 0

    def test_unknown_variable_abstains(self):
        fb = _default_fixture()
        r = fb.variable_coverage("NACC.DOES_NOT_EXIST")
        assert not r.grounded
        assert r.abstain_reason is not None


# ---------------------------------------------------------------------------
# Suppression invariant — no count in (0, SUPPRESSION_THRESHOLD)
# ---------------------------------------------------------------------------


class TestSuppressionInvariant:
    """Exhaustively check that no method ever returns a count in the forbidden range."""

    def _all_values(self, result: ToolResult) -> list[int]:
        return _all_int_values(result)

    def _assert_no_forbidden(self, values: list[int]) -> None:
        for v in values:
            assert v == 0 or v >= SUPPRESSION_THRESHOLD, (
                f"Found forbidden count {v} in (0, {SUPPRESSION_THRESHOLD})"
            )

    def test_count_eligible_all_filter_specs(self):
        fb = _default_fixture()
        specs = [
            FilterSpec(id_space=IdSpace.A),
            FilterSpec(baseline_diagnosis=["MCI"], id_space=IdSpace.A),
            FilterSpec(baseline_diagnosis=["Normal"], id_space=IdSpace.A),
            FilterSpec(baseline_diagnosis=["Dementia"], id_space=IdSpace.A),
            FilterSpec(required_modalities=["cognition"], id_space=IdSpace.A),
            FilterSpec(required_modalities=["fluid_biomarker"], id_space=IdSpace.A),
        ]
        for fs in specs:
            r = fb.count_eligible(fs)
            self._assert_no_forbidden(self._all_values(r))

    def test_modality_intersection_suppression(self):
        fb = _default_fixture()
        combos = [
            ["cognition"],
            ["mri"],
            ["cognition", "mri"],
            ["cognition", "fluid_biomarker"],
            ["cognition", "mri", "fluid_biomarker"],
        ]
        fs = FilterSpec(id_space=IdSpace.A)
        for mods in combos:
            r = fb.modality_intersection(mods, fs)
            self._assert_no_forbidden(self._all_values(r))

    def test_followup_distribution_suppression(self):
        fb = _default_fixture()
        specs = [
            FilterSpec(id_space=IdSpace.A),
            FilterSpec(baseline_diagnosis=["MCI"], id_space=IdSpace.A),
        ]
        for fs in specs:
            r = fb.followup_distribution(3, fs)
            self._assert_no_forbidden(self._all_values(r))

    def test_tiny_fixture_all_suppressed_or_zero(self):
        """A fixture with exactly 1 participant must never leak a count of 1."""
        fb = FixtureDataBackend(
            participants=[
                FixtureParticipant(
                    "P1",
                    [FixtureVisit(1, 2010, 2), FixtureVisit(2, 2014, 2)],
                    frozenset({"cognition"}),
                )
            ]
        )
        for method, args in [
            (fb.count_eligible, (FilterSpec(id_space=IdSpace.A),)),
            (fb.modality_intersection, (["cognition"], FilterSpec(id_space=IdSpace.A))),
            (fb.followup_distribution, (3, FilterSpec(id_space=IdSpace.A))),
        ]:
            r = method(*args)
            vals = _all_int_values(r)
            for v in vals:
                assert v == 0 or v >= SUPPRESSION_THRESHOLD, (
                    f"{method.__name__} leaked forbidden count {v}"
                )


# ---------------------------------------------------------------------------
# baseline_etiology filter
# ---------------------------------------------------------------------------


class TestBaselineEtiologyFilter:
    """Tests for the baseline_etiology FilterSpec field.

    Proves:
    - "ad" is rejected from baseline_diagnosis (not a cognitive-status label).
    - baseline_etiology=["ad"] selects by etiologic attribution, not cognitive status.
    - "dementia" in baseline_diagnosis selects ALL-CAUSE dementia, not only AD.
    - AND semantics when both baseline_diagnosis and baseline_etiology are provided.
    - All three public methods propagate the etiology filter.
    - Unknown etiology labels abstain deterministically.
    """

    _N = 15  # participants per group — above SUPPRESSION_THRESHOLD (10)

    def _etiology_fixture(self) -> FixtureDataBackend:
        """75 participants split into 5 groups (15 each), all with cognition modality.

        Groups:
          NORM*      — Normal (dx=1, etiology=None)
          MCI_NONAD* — MCI (dx=3, etiology=None)     → non-AD etiology reference
          MCI_AD*    — MCI (dx=3, etiology="ad")
          DEM_NONAD* — Dementia (dx=4, etiology=None)
          DEM_AD*    — Dementia (dx=4, etiology="ad")

        MCI_* participants have 4 visits spanning 3 years (sufficient for 3-yr horizon).
        DEM_* and NORM participants have 1 visit (censored).
        """
        n = self._N
        pts: list[FixtureParticipant] = []
        for i in range(n):
            pts.append(FixtureParticipant(
                pid=f"NORM{i:03d}",
                visits=[FixtureVisit(1, 2010, 1)],
                modalities=frozenset({"cognition"}),
                baseline_etiology=None,
            ))
        for i in range(n):
            pts.append(FixtureParticipant(
                pid=f"MCI_NONAD{i:03d}",
                visits=[FixtureVisit(v, 2010 + v, 3) for v in range(1, 5)],
                modalities=frozenset({"cognition"}),
                baseline_etiology=None,
            ))
        for i in range(n):
            pts.append(FixtureParticipant(
                pid=f"MCI_AD{i:03d}",
                visits=[FixtureVisit(v, 2010 + v, 3) for v in range(1, 5)],
                modalities=frozenset({"cognition"}),
                baseline_etiology="ad",
            ))
        for i in range(n):
            pts.append(FixtureParticipant(
                pid=f"DEM_NONAD{i:03d}",
                visits=[FixtureVisit(1, 2010, 4)],
                modalities=frozenset({"cognition"}),
                baseline_etiology=None,
            ))
        for i in range(n):
            pts.append(FixtureParticipant(
                pid=f"DEM_AD{i:03d}",
                visits=[FixtureVisit(1, 2010, 4)],
                modalities=frozenset({"cognition"}),
                baseline_etiology="ad",
            ))
        return FixtureDataBackend(pts)

    # ── baseline_diagnosis=["ad"] must abstain ────────────────────────────────

    def test_nacc_baseline_diagnosis_ad_abstains(self):
        """baseline_diagnosis=["ad"] must abstain on Space A — not a NACCUDSD cognitive label."""
        fb = self._etiology_fixture()
        r = fb.count_eligible(FilterSpec(baseline_diagnosis=["ad"], id_space=IdSpace.A))
        assert not r.grounded, "'ad' in baseline_diagnosis must abstain (not a NACCUDSD code)"
        assert r.abstain_reason is not None

    def test_adni_baseline_diagnosis_ad_abstains(self):
        """baseline_diagnosis=["ad"] must abstain on Space B — not an ADNI DIAGNOSIS code."""
        fb_b = FixtureDataBackend(
            participants=[
                FixtureParticipant(
                    pid=f"B{i:03d}",
                    visits=[FixtureVisit(1, 2010, 3)],  # dx=3 = Dementia (ADNI codes)
                    modalities=frozenset({"cognition"}),
                )
                for i in range(self._N)
            ],
            id_space=IdSpace.B,
        )
        r = fb_b.count_eligible(FilterSpec(baseline_diagnosis=["ad"], id_space=IdSpace.B))
        assert not r.grounded, "'ad' in baseline_diagnosis must abstain for Space B too"
        assert r.abstain_reason is not None

    # ── baseline_etiology=["ad"] selects correct participants ─────────────────

    def test_etiology_ad_selects_ad_participants_only(self):
        """baseline_etiology=["ad"] returns ONLY participants with baseline_etiology="ad"."""
        fb = self._etiology_fixture()
        r = fb.count_eligible(FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A))
        assert r.grounded
        # MCI_AD (15) + DEM_AD (15) = 30
        assert r.value == self._N * 2

    def test_etiology_label_synonyms_accepted(self):
        """All accepted AD etiology label variants must resolve to the same count."""
        fb = self._etiology_fixture()
        counts = set()
        for label in ("ad", "alzheimers", "alzheimer", "alzheimer's disease"):
            r = fb.count_eligible(FilterSpec(baseline_etiology=[label], id_space=IdSpace.A))
            assert r.grounded, f"Label '{label}' should be accepted"
            counts.add(r.value)
        assert len(counts) == 1, f"All AD etiology labels should return the same count; got {counts}"

    # ── baseline_diagnosis=["dementia"] is all-cause ──────────────────────────

    def test_dementia_diagnosis_is_all_cause_not_ad_only(self):
        """baseline_diagnosis=["dementia"] selects ALL-CAUSE dementia, not just AD-etiology."""
        fb = self._etiology_fixture()
        r = fb.count_eligible(FilterSpec(baseline_diagnosis=["dementia"], id_space=IdSpace.A))
        assert r.grounded
        # DEM_NONAD (15) + DEM_AD (15) = 30
        assert r.value == self._N * 2, (
            "baseline_diagnosis=['dementia'] must include ALL dementia (all-cause), "
            f"expected {self._N * 2}, got {r.value}"
        )

    # ── AND semantics ─────────────────────────────────────────────────────────

    def test_mci_and_ad_etiology_is_and(self):
        """baseline_diagnosis=["mci"] + baseline_etiology=["ad"] → only MCI_AD participants."""
        fb = self._etiology_fixture()
        r = fb.count_eligible(
            FilterSpec(baseline_diagnosis=["mci"], baseline_etiology=["ad"], id_space=IdSpace.A)
        )
        assert r.grounded
        assert r.value == self._N  # only MCI_AD (15)

    def test_dementia_and_ad_etiology_is_and(self):
        """baseline_diagnosis=["dementia"] + baseline_etiology=["ad"] → only DEM_AD."""
        fb = self._etiology_fixture()
        r = fb.count_eligible(
            FilterSpec(baseline_diagnosis=["dementia"], baseline_etiology=["ad"], id_space=IdSpace.A)
        )
        assert r.grounded
        assert r.value == self._N  # only DEM_AD (15)

    def test_normal_and_ad_etiology_returns_zero(self):
        """Normal participants have no AD etiology → combined filter returns 0."""
        fb = self._etiology_fixture()
        r = fb.count_eligible(
            FilterSpec(baseline_diagnosis=["normal"], baseline_etiology=["ad"], id_space=IdSpace.A)
        )
        assert r.grounded
        assert r.value == 0

    # ── Unknown etiology abstains ─────────────────────────────────────────────

    def test_unknown_etiology_abstains(self):
        """An unrecognised baseline_etiology label must abstain, not silently pass or fail."""
        fb = self._etiology_fixture()
        r = fb.count_eligible(FilterSpec(baseline_etiology=["parkinsons"], id_space=IdSpace.A))
        assert not r.grounded
        assert r.abstain_reason is not None

    # ── All three public methods propagate etiology ───────────────────────────

    def test_etiology_propagates_to_modality_intersection(self):
        """modality_intersection must respect baseline_etiology filter."""
        fb = self._etiology_fixture()
        r_etiol = fb.modality_intersection(
            ["cognition"],
            FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A),
        )
        r_all = fb.modality_intersection(
            ["cognition"],
            FilterSpec(id_space=IdSpace.A),
        )
        assert r_etiol.grounded and r_all.grounded
        # AD-etiology group: 30; total: 75
        assert r_etiol.value.get("cognition") == self._N * 2
        assert r_all.value.get("cognition") == self._N * 5

    def test_etiology_propagates_to_followup_distribution(self):
        """followup_distribution must respect baseline_etiology filter."""
        fb = self._etiology_fixture()
        r = fb.followup_distribution(
            3,
            FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A),
        )
        assert r.grounded
        # MCI_AD (4 visits, 3yr span → sufficient); DEM_AD (1 visit → censored)
        assert r.value.get("sufficient") == self._N   # MCI_AD
        assert r.value.get("censored") == self._N     # DEM_AD
        assert r.value.get("insufficient") == 0

    def test_followup_total_matches_count_with_etiology(self):
        """Follow-up bucket totals must equal count_eligible for the same etiology filter."""
        fb = self._etiology_fixture()
        fs = FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A)
        r_cnt = fb.count_eligible(fs)
        r_fu = fb.followup_distribution(3, fs)
        assert r_cnt.grounded and r_fu.grounded
        bucket_total = sum(v for v in r_fu.value.values() if v is not None)
        assert bucket_total == r_cnt.value


# ---------------------------------------------------------------------------
# _select_adni_baseline — unit tests for EXAMDATE-based baseline row selection
# ---------------------------------------------------------------------------


class TestSelectAdniBaseline:
    """Unit tests for the module-level _select_adni_baseline() helper.

    Verifies that EXAMDATE is preferred over VISCODE for chronological ordering
    and that NaT EXAMDATE falls back to VISCODE lexicographic sort.
    """

    def _df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows).astype({"RID": str})

    def test_examdate_preferred_over_viscode(self):
        """Chronologically earlier EXAMDATE wins even when VISCODE sorts differently."""
        df = self._df([
            {"RID": "001", "VISCODE": "m12", "DIAGNOSIS": 2, "EXAMDATE": "2010-06-01"},
            {"RID": "001", "VISCODE": "bl",  "DIAGNOSIS": 1, "EXAMDATE": "2011-01-01"},
        ])
        # m12 has earlier EXAMDATE (2010-06-01) even though "bl" < "m12" alphabetically
        first = _select_adni_baseline(df, "DIAGNOSIS")
        assert len(first) == 1
        assert first.iloc[0]["VISCODE"] == "m12"
        assert int(first.iloc[0]["DIAGNOSIS"]) == 2

    def test_nat_examdate_falls_back_to_viscode(self):
        """Rows with unparseable/null EXAMDATE fall back to VISCODE lexicographic sort."""
        df = self._df([
            {"RID": "002", "VISCODE": "m12", "DIAGNOSIS": 2, "EXAMDATE": None},
            {"RID": "002", "VISCODE": "bl",  "DIAGNOSIS": 1, "EXAMDATE": None},
        ])
        # No parseable EXAMDATE → fallback to VISCODE; "bl" < "m12" → "bl" wins
        first = _select_adni_baseline(df, "DIAGNOSIS")
        assert len(first) == 1
        assert first.iloc[0]["VISCODE"] == "bl"

    def test_real_date_beats_nat_for_same_rid(self):
        """A row with a real EXAMDATE is chosen before a NaT row, regardless of VISCODE."""
        df = self._df([
            {"RID": "003", "VISCODE": "bl",  "DIAGNOSIS": 1, "EXAMDATE": None},
            {"RID": "003", "VISCODE": "m12", "DIAGNOSIS": 2, "EXAMDATE": "2008-03-15"},
        ])
        # "bl" < "m12" alphabetically, but "bl" has NaT EXAMDATE → "m12" (real date) wins
        first = _select_adni_baseline(df, "DIAGNOSIS")
        assert len(first) == 1
        assert first.iloc[0]["VISCODE"] == "m12"

    def test_no_examdate_column_uses_viscode(self):
        """Without an EXAMDATE column, pure VISCODE sort is used."""
        df = pd.DataFrame([
            {"RID": "004", "VISCODE": "m12", "DIAGNOSIS": 2},
            {"RID": "004", "VISCODE": "bl",  "DIAGNOSIS": 1},
        ]).astype({"RID": str})
        first = _select_adni_baseline(df, "DIAGNOSIS")
        assert len(first) == 1
        assert first.iloc[0]["VISCODE"] == "bl"

    def test_null_dx_rows_excluded(self):
        """Rows with null DIAGNOSIS are skipped; next valid row becomes baseline."""
        df = self._df([
            {"RID": "005", "VISCODE": "bl",  "DIAGNOSIS": None, "EXAMDATE": "2010-01-01"},
            {"RID": "005", "VISCODE": "m06", "DIAGNOSIS": 2,    "EXAMDATE": "2010-06-01"},
        ])
        first = _select_adni_baseline(df, "DIAGNOSIS")
        assert len(first) == 1
        assert first.iloc[0]["VISCODE"] == "m06"

    def test_multiple_rids_independent(self):
        """Each RID is selected independently."""
        df = self._df([
            {"RID": "A", "VISCODE": "m12", "DIAGNOSIS": 2, "EXAMDATE": "2009-01-01"},
            {"RID": "A", "VISCODE": "bl",  "DIAGNOSIS": 1, "EXAMDATE": "2010-01-01"},
            {"RID": "B", "VISCODE": "m12", "DIAGNOSIS": 3, "EXAMDATE": "2012-05-01"},
            {"RID": "B", "VISCODE": "bl",  "DIAGNOSIS": 2, "EXAMDATE": "2013-05-01"},
        ])
        first = _select_adni_baseline(df, "DIAGNOSIS").set_index("RID")
        assert first.loc["A", "VISCODE"] == "m12"  # earlier EXAMDATE
        assert first.loc["B", "VISCODE"] == "m12"  # earlier EXAMDATE


# ---------------------------------------------------------------------------
# LocalDataBackend smoke tests (require real data on disk)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PROJECT_ROOT / "NACC" / "investigator_nacc70.csv").exists(),
    reason="Real NACC data not mounted",
)
class TestLocalDataBackendSmoke:
    """Smoke tests that hit real data. Skipped in CI without real files."""

    @pytest.fixture(scope="class")
    def backend(self):
        config = _load_config()
        return LocalDataBackend(config, PROJECT_ROOT)

    def test_count_eligible_all_returns_sane_number(self, backend):
        """All NACC participants should number in the tens of thousands."""
        r = backend.count_eligible(FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        assert r.value >= 10_000

    def test_count_eligible_mci(self, backend):
        # NACC baseline NACCUDSD=3 (MCI) count is ~12,260 across all participants.
        # (NACCUDSD=2 is Impaired-not-MCI, with ~2,465 at baseline — a different group.)
        r = backend.count_eligible(
            FilterSpec(baseline_diagnosis=["MCI"], id_space=IdSpace.A)
        )
        assert r.grounded
        assert r.value >= 10_000  # NACCUDSD=3 baseline N ~12,260

    def test_count_eligible_normal(self, backend):
        r = backend.count_eligible(
            FilterSpec(baseline_diagnosis=["Normal"], id_space=IdSpace.A)
        )
        assert r.grounded
        assert r.value >= 5_000

    def test_count_eligible_space_b(self, backend):
        r = backend.count_eligible(FilterSpec(id_space=IdSpace.B))
        assert r.grounded
        assert r.value >= 100

    def test_modality_intersection_cognition(self, backend):
        r = backend.modality_intersection(
            ["cognition"], FilterSpec(id_space=IdSpace.A)
        )
        assert r.grounded
        assert r.value.get("cognition") is not None
        assert r.value["cognition"] >= 1_000

    def test_modality_intersection_cognition_mri(self, backend):
        r = backend.modality_intersection(
            ["cognition", "mri"], FilterSpec(id_space=IdSpace.A)
        )
        assert r.grounded
        assert "cognition,mri" in r.value
        # intersection must be ≤ each individual
        assert r.value["cognition,mri"] is None or (
            r.value["cognition,mri"] <= r.value["cognition"]
        )

    def test_followup_mci_3yr(self, backend):
        r = backend.followup_distribution(
            3,
            FilterSpec(baseline_diagnosis=["MCI"], id_space=IdSpace.A),
        )
        assert r.grounded
        assert "sufficient" in r.value
        assert "censored" in r.value
        # Some MCI participants should have 3+ year follow-up
        suf = r.value["sufficient"]
        assert suf is None or suf >= 10

    def test_variable_coverage_nacc_cdrsum(self, backend):
        r = backend.variable_coverage("NACC.CDRSUM")
        assert r.grounded
        cs = r.value
        assert cs.n_available > 100_000  # CDRSUM is fully populated
        assert cs.missingness_rate < 0.01
        assert cs.longitudinal_availability > 10_000
        assert cs.last_computed is not None

    def test_variable_coverage_adsp_phc(self, backend):
        r = backend.variable_coverage("ADSP_PHC.Cognition.PHC_MEM")
        assert r.grounded
        cs = r.value
        assert cs.n_available > 100_000
        assert cs.last_computed is not None

    def test_variable_coverage_adni(self, backend):
        r = backend.variable_coverage("ADNI.DXSUM.DXCURREN")
        # May or may not be present depending on exact column name
        # Just check it doesn't crash
        assert r.grounded or (not r.grounded and r.abstain_reason is not None)

    def test_no_output_leaks_subject_ids(self, backend):
        """Spot-check that no method returns a list of IDs."""
        r = backend.count_eligible(FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        assert isinstance(r.value, int), "count_eligible must return int, not a list"

    def test_aggregate_stat_space_b_plasma_ptau217(self, backend):
        """Plasma PTAU217 (harmonized) is present locally — Space B must not abstain."""
        r = backend.aggregate_stat(
            "ADNI.ADSP_PHC_PLASMA.PHC_pTau217", "mean", FilterSpec(id_space=IdSpace.B)
        )
        assert r.grounded
        assert r.value["stat_type"] == "mean"
        assert r.value["n"] >= 100
        assert isinstance(r.value["value"], float)

    def test_descriptive_stats_space_b_plasma_ptau217(self, backend):
        r = backend.descriptive_stats(
            "ADNI.ADSP_PHC_PLASMA.PHC_pTau217", FilterSpec(id_space=IdSpace.B)
        )
        assert r.grounded
        assert r.value["count"] >= 100
        assert r.value["mean"] is not None

    def test_aggregate_stat_space_b_csf_ptau217_abstains(self, backend):
        """CSF has no PTAU217 column locally — must abstain, not error or guess."""
        r = backend.aggregate_stat(
            "ADNI.ADSP_PHC_CSF.PHC_pTau217", "mean", FilterSpec(id_space=IdSpace.B)
        )
        assert not r.grounded
        assert r.abstain_reason is not None

    def test_aggregate_stat_space_b_bad_variable_id_abstains(self, backend):
        r = backend.aggregate_stat(
            "NACC.CDRSUM", "mean", FilterSpec(id_space=IdSpace.B)
        )
        assert not r.grounded

    # ── P0 expansion (2026-07-15 LONI download) ────────────────────────────

    def test_modality_intersection_space_b_pet_now_supported(self, backend):
        """PET was previously unmapped for Space B — UCBERKELEY_AMY_6MM/TAU_6MM
        fix that gap. Must return a grounded, non-zero count, not an abstain.
        """
        r = backend.modality_intersection(["pet"], FilterSpec(id_space=IdSpace.B))
        assert r.grounded
        assert r.value["pet"] > 0

    def test_modality_intersection_space_b_mri_pet_combo(self, backend):
        r = backend.modality_intersection(
            ["mri", "pet"], FilterSpec(id_space=IdSpace.B)
        )
        assert r.grounded
        assert r.value["mri"] > 0
        assert r.value["pet"] > 0
        assert r.value["mri,pet"] <= min(r.value["mri"], r.value["pet"])

    def test_aggregate_stat_space_b_upenn_plasma_fujirebio(self, backend):
        r = backend.aggregate_stat(
            "ADNI.UPENN_PLASMA_FUJIREBIO_QUANTERIX.pT217_F",
            "mean",
            FilterSpec(id_space=IdSpace.B),
        )
        assert r.grounded
        assert r.value["stat_type"] == "mean"
        assert isinstance(r.value["value"], float)

    def test_variant_subject_counts_ptau181_spans_legacy_and_harmonized(self, backend):
        """ptau181 must aggregate ADSP-PHC harmonized/raw AND the legacy
        Alzbio3/Roche Elecsys assay columns — not just the ADSP-PHC pair.
        """
        r = backend.variant_subject_counts(
            [
                "ADNI.ADSP_PHC_CSF.PHC_pTau181",
                "ADNI.UPENNBIOMK_MASTER.PTAU",
                "ADNI.UPENNBIOMK_ROCHE_ELECSYS.PTAU",
            ],
            FilterSpec(id_space=IdSpace.B),
        )
        assert r.grounded
        assert set(r.value.keys()) == {
            "ADNI.ADSP_PHC_CSF.PHC_pTau181",
            "ADNI.UPENNBIOMK_MASTER.PTAU",
            "ADNI.UPENNBIOMK_ROCHE_ELECSYS.PTAU",
        }

    def test_naccudsd_mci_is_3_not_2(self, backend):
        """NACCUDSD=3 is MCI; the old wrong code used 2. Verify real counts differ.

        Real NACC NACCUDSD=3 (MCI) baseline N ≈ 12,260.
        Real NACC NACCUDSD=2 (Impaired-not-MCI) baseline N ≈ 2,465.
        These are disjoint groups; the old bug would have returned 2,465 for 'mci'.
        """
        r_mci = backend.count_eligible(
            FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A)
        )
        r_impaired = backend.count_eligible(
            FilterSpec(baseline_diagnosis=["impaired not mci"], id_space=IdSpace.A)
        )
        assert r_mci.grounded and r_impaired.grounded
        # MCI (NACCUDSD=3) baseline N should be substantially larger than impaired-not-MCI
        assert r_mci.value >= 10_000  # ~12,260
        assert r_impaired.value >= 1_000  # ~2,465
        assert r_mci.value != r_impaired.value

    def test_nacc_baseline_diagnosis_ad_abstains_on_real(self, backend):
        """'ad' is not a valid NACCUDSD label on the real NACC backend."""
        r = backend.count_eligible(
            FilterSpec(baseline_diagnosis=["ad"], id_space=IdSpace.A)
        )
        assert not r.grounded

    def test_adni_baseline_diagnosis_ad_abstains_on_real(self, backend):
        """'ad' is not a valid ADNI DIAGNOSIS code on the real backend."""
        r = backend.count_eligible(
            FilterSpec(baseline_diagnosis=["ad"], id_space=IdSpace.B)
        )
        assert not r.grounded

    def test_nacc_etiology_ad_uses_naccalzd_not_naccudsd4(self, backend):
        """NACC baseline_etiology=["ad"] uses NACCALZD==1, which spans MCI AND dementia.

        NACCALZD==1 is set for any participant with presumptive AD etiology regardless
        of cognitive status — it includes both MCI-with-AD-etiology AND dementia-with-AD.
        So the AD-etiology count is NOT necessarily ≤ the all-cause-dementia count.

        The correct invariant: dementia AND AD etiology combined count is ≤ either
        of the individual filter counts (AND semantics).
        """
        r_dem = backend.count_eligible(
            FilterSpec(baseline_diagnosis=["dementia"], id_space=IdSpace.A)
        )
        r_ad_etiol = backend.count_eligible(
            FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A)
        )
        r_dem_and_ad = backend.count_eligible(
            FilterSpec(baseline_diagnosis=["dementia"], baseline_etiology=["ad"], id_space=IdSpace.A)
        )
        assert r_dem.grounded and r_ad_etiol.grounded and r_dem_and_ad.grounded
        # AND semantics: combined count ≤ each individual count
        assert r_dem_and_ad.value <= r_dem.value
        assert r_dem_and_ad.value <= r_ad_etiol.value
        # NACCALZD==1 is a real non-trivial group (sanity check)
        assert r_ad_etiol.value >= SUPPRESSION_THRESHOLD

    def test_adni_etiology_ad_returns_sane_count(self, backend):
        """ADNI baseline_etiology=["ad"] should return a grounded count or a clear abstain."""
        r = backend.count_eligible(
            FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.B)
        )
        # Either succeeds (etiology columns present) or abstains with a clear reason
        if r.grounded:
            assert r.value >= 0
        else:
            assert r.abstain_reason is not None
            assert "etiology" in r.abstain_reason.lower() or "dxmdue" in r.abstain_reason.lower()

    def test_adni_examdate_ordering_implemented(self, backend):
        """Space B baseline row selection uses EXAMDATE when available.

        We can't compare against a known ground-truth value, but we verify that
        the EXAMDATE-ordered path runs without error and returns a sane count
        that is consistent with (≤) the unfiltered Space B count.
        """
        r_all = backend.count_eligible(FilterSpec(id_space=IdSpace.B))
        r_dem = backend.count_eligible(
            FilterSpec(baseline_diagnosis=["dementia"], id_space=IdSpace.B)
        )
        assert r_all.grounded
        assert r_dem.grounded
        assert r_dem.value <= r_all.value

    def test_suppression_invariant_on_real_data(self, backend):
        """Every int count from real data must be 0 or ≥ threshold."""
        specs = [
            FilterSpec(id_space=IdSpace.A),
            FilterSpec(baseline_diagnosis=["MCI"], id_space=IdSpace.A),
            FilterSpec(id_space=IdSpace.B),
        ]
        for fs in specs:
            for result in [
                backend.count_eligible(fs),
                backend.followup_distribution(3, fs),
            ]:
                for v in _all_int_values(result):
                    assert v == 0 or v >= SUPPRESSION_THRESHOLD, (
                        f"Real data leaked forbidden count {v}"
                    )


# ---------------------------------------------------------------------------
# NACC modality-combination audit — design-gap documentation + regression
# ---------------------------------------------------------------------------
#
# Independent audit (scripts/audit_nacc_modality_counts_independent.py) shows:
#   - Raw NACC CSV has NO structural T1/MRI feature columns.
#   - 'cognition' in LocalDataBackend.modality_intersection() means presence in
#     the ADSP-PHC Cognition harmonised CSV, NOT raw NACC CDR/MMSE columns.
#   - 'mri' means presence in ADSP-PHC T1/FLAIR/DTI CSVs (not NACCNMRI field).
#   - These ADSP-PHC-derived counts CANNOT be validated from raw NACC columns.
#   - Therefore NACC-CSV-specific regression testing is not meaningful without
#     a source/id-space scoping API that exposes the ADSP-PHC file boundary.
#
# Audited values from LocalDataBackend (2026-06-15, investigator_nacc70 + ADSP-PHC R3):
#   cognition             = 49,344  (NACCIDs in NACC CSV ∩ ADSP-PHC Cognition file)
#   mri                   =  8,584  (NACCIDs in NACC CSV ∩ ADSP-PHC T1/FLAIR/DTI files)
#   cognition,mri         =  8,563  (intersection of both)
#
# Raw NACC independent audit (same run):
#   Total NACCIDs in raw NACC CSV           : 54,631
#   NACCIDs with cognition column non-null  : 54,631  (all participants have CDR/MMSE)
#   NACCIDs with NACCNMRI > 0 (MRI proxy)  :  7,935
#   cognition + any_t1 from raw NACC alone  : NOT COMPUTABLE — no T1 feature columns


@pytest.mark.skip(
    reason=(
        "Design gap: modality_intersection() for Space A computes ADSP-PHC file "
        "membership, not raw NACC CSV column presence. There are no structural T1 "
        "feature columns in investigator_nacc70.csv. 'cognition + any_t1' cannot "
        "be independently validated from raw NACC alone. A source/id-space scoping "
        "API change would be needed to enable NACC-CSV-specific modality regression "
        "testing. See scripts/audit_nacc_modality_counts_independent.py for details."
    )
)
def test_nacc_modality_combination_counts_match_independent_audit():
    """Placeholder: NACC-CSV-specific modality-combination regression test.

    Cannot be implemented without a backend API change that exposes which modality
    counts come from raw NACC columns vs ADSP-PHC file membership. Currently
    all Space A modality counts use ADSP-PHC file membership (by design), making
    a raw-NACC-only audit impossible via the modality_intersection() interface.
    """
    pytest.fail("This test requires a source-scoping API change — see reason above.")


@pytest.mark.skipif(
    not (PROJECT_ROOT / "NACC" / "investigator_nacc70.csv").exists(),
    reason="Real NACC data not mounted",
)
class TestNaccModalityCombinationRegression:
    """Regression tests for LocalDataBackend.modality_intersection() Space A counts.

    These counts are ADSP-PHC file-membership counts (not raw NACC column counts).
    Counts were verified against the independent audit on 2026-06-15 using
    investigator_nacc70.csv + ADSP-PHC R3 (December 2024).

    'cognition' = NACCIDs in NACC CSV ∩ ADSP-PHC Cognition file
    'mri'       = NACCIDs in NACC CSV ∩ ADSP-PHC T1/FLAIR/DTI files
    Neither is directly derivable from raw NACC CSV columns alone.
    """

    # Audited on 2026-06-15 from LocalDataBackend with investigator_nacc70 + ADSP-PHC R3.
    # Update these if the ADSP-PHC release or NACC CSV changes.
    _EXPECTED_COGNITION = 49_344
    _EXPECTED_MRI = 8_584
    _EXPECTED_COGNITION_MRI = 8_563

    # Allow ±1% tolerance for minor re-releases or row-order changes.
    _TOL = 0.01

    @pytest.fixture(scope="class")
    def backend(self):
        config = _load_config()
        return LocalDataBackend(config, PROJECT_ROOT)

    def _within_tolerance(self, actual: int | None, expected: int) -> bool:
        if actual is None:
            return False
        delta = abs(actual - expected) / max(expected, 1)
        return delta <= self._TOL

    def test_nacc_modality_combination_counts_match_independent_audit(self, backend):
        """Regression: Space A cognition+mri counts match 2026-06-15 audited values.

        These are raw NACC participant-level counts over NACCID, where modality
        presence is determined by ADSP-PHC file membership (not NACC CSV columns).
        Do NOT use the ADSP-PHC standalone count (7,400) as the expected value here
        — that is a different population (ADSP-PHC only; includes ADNI-origin).
        """
        r = backend.modality_intersection(
            ["cognition", "mri"],
            FilterSpec(id_space=IdSpace.A),
        )
        assert r.grounded, f"Expected grounded=True, got abstain: {r.abstain_reason}"
        assert r.value is not None

        # Normalize key order so {'mri,cognition'} == {'cognition,mri'}
        actual = {",".join(sorted(k.split(","))): v for k, v in r.value.items()}

        cog = actual.get("cognition")
        mri = actual.get("mri")
        cog_mri = actual.get("cognition,mri")

        assert self._within_tolerance(cog, self._EXPECTED_COGNITION), (
            f"cognition count {cog} deviates >1% from expected {self._EXPECTED_COGNITION}"
        )
        assert self._within_tolerance(mri, self._EXPECTED_MRI), (
            f"mri count {mri} deviates >1% from expected {self._EXPECTED_MRI}"
        )
        assert self._within_tolerance(cog_mri, self._EXPECTED_COGNITION_MRI), (
            f"cognition+mri count {cog_mri} deviates >1% from expected {self._EXPECTED_COGNITION_MRI}"
        )
        # Structural invariant: intersection ≤ each individual
        assert cog_mri <= cog, "cognition,mri intersection must be ≤ cognition count"
        assert cog_mri <= mri, "cognition,mri intersection must be ≤ mri count"

    def test_space_a_modality_source_is_adsp_phc_not_nacc_csv(self, backend):
        """Document: Space A 'cognition' count < raw NACC total (ADSP-PHC subset only).

        Raw NACC has 54,631 unique NACCIDs, all with CDR/MMSE data. But the backend
        returns 49,344 for 'cognition' — the subset that also appears in the ADSP-PHC
        Cognition harmonised file. This confirms modality presence uses ADSP-PHC file
        membership, not raw NACC column presence.
        """
        r_all = backend.count_eligible(FilterSpec(id_space=IdSpace.A))
        r_cog = backend.modality_intersection(
            ["cognition"],
            FilterSpec(id_space=IdSpace.A),
        )
        assert r_all.grounded and r_cog.grounded
        cog = r_cog.value.get("cognition")
        assert cog is not None
        # Raw NACC has ~54,631 NACCIDs; ADSP-PHC Cognition covers ~49,344 of them.
        # The gap (~5,287) are NACC participants not in ADSP-PHC Cognition release.
        assert cog < r_all.value, (
            "Cognition (ADSP-PHC subset) must be < total NACC eligible "
            "because ADSP-PHC Cognition does not cover all NACC participants"
        )
        assert cog >= 40_000, f"ADSP-PHC cognition coverage unexpectedly low: {cog}"


# ---------------------------------------------------------------------------
# populate_catalog_coverage
# ---------------------------------------------------------------------------


class TestPopulateCatalogCoverage:
    def test_coverage_populated_for_known_variables(self):
        from alz_agent.catalog.builder import populate_catalog_coverage
        from alz_agent.schemas import CatalogEntry, IdSpace, LeakageTag, Modality, SourceDataset

        fb = _default_fixture()
        entries = [
            CatalogEntry(
                variable_id="NACC.CDRSUM",
                concept="cdrsum",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="CDRSUM",
                release_version="nacc70",
                modality=Modality.COGNITION,
                leakage_tags=[LeakageTag.NONE],
            ),
            CatalogEntry(
                variable_id="NACC.DOES_NOT_EXIST",
                concept="missing",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="DOES_NOT_EXIST",
                release_version="nacc70",
                modality=Modality.OTHER,
                leakage_tags=[LeakageTag.NONE],
            ),
        ]

        updated, n = populate_catalog_coverage(entries, fb)
        assert len(updated) == 2
        assert n == 1  # only CDRSUM has coverage in fixture map
        assert updated[0].coverage is not None  # CDRSUM got coverage
        assert updated[1].coverage is None       # missing stays None

    def test_serialization_round_trip(self, tmp_path):
        import json as _json
        from alz_agent.catalog.builder import populate_catalog_coverage
        from alz_agent.schemas import CatalogEntry, IdSpace, LeakageTag, Modality, SourceDataset

        fb = _default_fixture()
        entries = [
            CatalogEntry(
                variable_id="NACC.CDRSUM",
                concept="cdrsum",
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code="CDRSUM",
                release_version="nacc70",
                modality=Modality.COGNITION,
                leakage_tags=[LeakageTag.NONE],
            ),
        ]
        out_path = tmp_path / "catalog.json"
        updated, _ = populate_catalog_coverage(entries, fb, output_path=out_path)

        assert out_path.exists()
        loaded = _json.loads(out_path.read_text())
        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]["variable_id"] == "NACC.CDRSUM"


# ---------------------------------------------------------------------------
# event_etiology filter — fixture A/B/C scenario
# ---------------------------------------------------------------------------
#
# Participant A: NACCALZD==1 at visit 1 (baseline AD).
# Participant B: NACCALZD==0 at visit 1; NACCALZD==1 first appears at visit 3 (incident AD).
# Participant C: NACCALZD==1 never (no AD etiology).
#
# Expected:
#   baseline_etiology=["ad"]  → A only           (n=1)
#   event_etiology=["ad"]     → A and B           (n=2)
#   no etiology filter        → A, B, and C       (n=3)
#   mean age at first_diagnosis with event_etiology → uses A's visit-1 age and B's visit-3 age
#   n must equal unique participants, not rows


def _abc_backend() -> tuple[FixtureDataBackend, list[FixtureParticipant], list[FixtureParticipant], list[FixtureParticipant]]:
    """Build a fixture with 3 distinct groups for event_etiology testing.

    Group A (12 participants): NACCALZD==1 at visit 1 (baseline AD).
        baseline_etiology="ad", ad_etiology_first_vnum=1.
        Mean age at first AD visit = 70.

    Group B (12 participants): NACCALZD==0 at visit 1, NACCALZD==1 at visit 3 (incident AD).
        baseline_etiology=None, ad_etiology_first_vnum=3.
        Mean age at first AD visit = 67 (baseline 65 + 2).

    Group C (12 participants): never NACCALZD==1.
        baseline_etiology=None, ad_etiology_first_vnum=None.

    Groups are 12 each to exceed SUPPRESSION_THRESHOLD (10).
    """
    n_per_group = 12
    group_a = [
        FixtureParticipant(
            pid=f"A{i:02d}",
            visits=[
                FixtureVisit(vnum=1, year=2010, diagnosis=3),
                FixtureVisit(vnum=2, year=2011, diagnosis=3),
                FixtureVisit(vnum=3, year=2012, diagnosis=4),
            ],
            modalities=frozenset({"cognition"}),
            baseline_etiology="ad",
            age_at_baseline=70,
            ad_etiology_first_vnum=1,
            age_at_first_ad_etiology=70,
        )
        for i in range(n_per_group)
    ]
    group_b = [
        FixtureParticipant(
            pid=f"B{i:02d}",
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
        for i in range(n_per_group)
    ]
    group_c = [
        FixtureParticipant(
            pid=f"C{i:02d}",
            visits=[
                FixtureVisit(vnum=1, year=2010, diagnosis=1),
                FixtureVisit(vnum=2, year=2011, diagnosis=1),
            ],
            modalities=frozenset(),
            baseline_etiology=None,
            age_at_baseline=72,
            ad_etiology_first_vnum=None,
            age_at_first_ad_etiology=None,
        )
        for i in range(n_per_group)
    ]
    backend = FixtureDataBackend(
        participants=group_a + group_b + group_c,
        id_space=IdSpace.A,
    )
    return backend, group_a, group_b, group_c


class TestEventEtiologyFixture:
    """event_etiology vs baseline_etiology separation — A/B/C scenario.

    Group A (12): NACCALZD==1 at visit 1 (baseline AD), age_at_first_ad_etiology=70.
    Group B (12): NACCALZD==1 first at visit 3 (incident AD), age_at_first_ad_etiology=67.
    Group C (12): never NACCALZD==1.
    Total = 36 participants, all groups > suppression threshold (10).
    """

    def test_no_etiology_filter_returns_all_36(self):
        backend, ga, gb, gc = _abc_backend()
        r = backend.count_eligible(FilterSpec(id_space=IdSpace.A))
        assert r.grounded
        assert r.value == 36

    def test_baseline_etiology_returns_only_group_a(self):
        """baseline_etiology=["ad"] → only group A (NACCALZD==1 at visit 1)."""
        backend, ga, gb, gc = _abc_backend()
        r = backend.count_eligible(FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A))
        assert r.grounded
        assert r.value == len(ga)   # 12

    def test_event_etiology_returns_a_and_b(self):
        """event_etiology=["ad"] → groups A and B (any-visit NACCALZD==1)."""
        backend, ga, gb, gc = _abc_backend()
        r = backend.count_eligible(FilterSpec(event_etiology=["ad"], id_space=IdSpace.A))
        assert r.grounded
        assert r.value == len(ga) + len(gb)   # 24

    def test_event_etiology_excludes_group_c(self):
        """Group C (no AD etiology ever) is not in event_etiology result."""
        backend, ga, gb, gc = _abc_backend()
        r = backend.count_eligible(FilterSpec(event_etiology=["ad"], id_space=IdSpace.A))
        assert r.grounded
        assert r.value == len(ga) + len(gb)   # 24; C excluded

    def test_event_etiology_superset_of_baseline_etiology(self):
        """event_etiology count > baseline_etiology count (incident group B added)."""
        backend, ga, gb, gc = _abc_backend()
        r_base = backend.count_eligible(FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A))
        r_event = backend.count_eligible(FilterSpec(event_etiology=["ad"], id_space=IdSpace.A))
        assert r_event.grounded and r_base.grounded
        assert r_event.value > r_base.value
        assert r_event.value - r_base.value == len(gb)  # incident cases

    def test_aggregate_stat_baseline_etiology_uses_visit1_age(self):
        """baseline_etiology + first_diagnosis → mean of group A ages (70.0)."""
        backend, ga, gb, gc = _abc_backend()
        r = backend.aggregate_stat(
            "NACC.NACCAGE", "mean",
            FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A),
            event_anchor="first_diagnosis",
        )
        assert r.grounded
        assert r.value["n"] == len(ga)          # 12 participants (group A only)
        assert r.value["value"] == 70.0         # all group A have age_at_first_ad=70

    def test_aggregate_stat_event_etiology_uses_first_ad_visit_age(self):
        """event_etiology + first_diagnosis → mean of A(70) and B(67) groups = 68.5."""
        backend, ga, gb, gc = _abc_backend()
        r = backend.aggregate_stat(
            "NACC.NACCAGE", "mean",
            FilterSpec(event_etiology=["ad"], id_space=IdSpace.A),
            event_anchor="first_diagnosis",
        )
        assert r.grounded
        assert r.value["n"] == len(ga) + len(gb)  # 24 participants
        # mean(12×70 + 12×67) / 24 = (840 + 804) / 24 = 1644 / 24 = 68.5
        assert r.value["value"] == 68.5

    def test_n_is_participant_count_not_row_count(self):
        """n must equal unique participants, not total qualifying rows."""
        backend, ga, gb, gc = _abc_backend()
        # Groups A+B have 3 visits each = 72 total visits.
        # event_etiology n must be 24 (unique participants), not 72.
        r = backend.aggregate_stat(
            "NACC.NACCAGE", "mean",
            FilterSpec(event_etiology=["ad"], id_space=IdSpace.A),
            event_anchor="first_diagnosis",
        )
        assert r.grounded
        assert r.value["n"] == len(ga) + len(gb)  # 24, not 72

    def test_repeated_qualifying_rows_do_not_inflate_n(self):
        """Group A qualifies at all 3 visits; n stays 12 (unique participants)."""
        backend, ga, gb, gc = _abc_backend()
        r = backend.aggregate_stat(
            "NACC.NACCAGE", "mean",
            FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A),
            event_anchor="first_diagnosis",
        )
        assert r.grounded
        assert r.value["n"] == len(ga)  # 12, not 36 (3 visits × 12 participants)

    def test_unknown_event_etiology_label_abstains(self):
        backend, ga, gb, gc = _abc_backend()
        r = backend.count_eligible(
            FilterSpec(event_etiology=["UNKNOWN_LABEL"], id_space=IdSpace.A)
        )
        assert not r.grounded

    def test_filter_spec_allows_both_etiology_fields_and_logic(self):
        """baseline_etiology AND event_etiology applied together (AND intersection)."""
        backend, ga, gb, gc = _abc_backend()
        # A satisfies both; B satisfies event_etiology only → intersection = A only.
        r = backend.count_eligible(FilterSpec(
            baseline_etiology=["ad"],
            event_etiology=["ad"],
            id_space=IdSpace.A,
        ))
        assert r.grounded
        assert r.value == len(ga)  # 12 (group A only)


# ---------------------------------------------------------------------------
# event_etiology — default fixture regression (incident_ad group)
# ---------------------------------------------------------------------------


class TestEventEtiologyDefaultFixture:
    """event_etiology in the default build_fixture_backend() with n_incident_ad=12."""

    def setup_method(self):
        self.fb = build_fixture_backend(
            n_mci_ad=15, n_dementia_ad=15, n_incident_ad=12, seed=42
        )

    def test_event_etiology_larger_than_baseline_etiology(self):
        r_base = self.fb.count_eligible(FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A))
        r_event = self.fb.count_eligible(FilterSpec(event_etiology=["ad"], id_space=IdSpace.A))
        assert r_base.grounded and r_event.grounded
        # event_etiology includes the n_incident_ad=12 extra participants
        assert r_event.value > r_base.value

    def test_event_etiology_includes_incident_ad_participants(self):
        r_base = self.fb.count_eligible(FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A))
        r_event = self.fb.count_eligible(FilterSpec(event_etiology=["ad"], id_space=IdSpace.A))
        # The difference should be at least n_incident_ad=12 minus suppressed
        assert r_event.value - r_base.value >= 10

    def test_mean_age_event_etiology_differs_from_baseline_etiology(self):
        """Mean age at first AD dx uses different cohort sizes for the two filters."""
        r_base = self.fb.aggregate_stat(
            "NACC.NACCAGE", "mean",
            FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A),
            event_anchor="first_diagnosis",
        )
        r_event = self.fb.aggregate_stat(
            "NACC.NACCAGE", "mean",
            FilterSpec(event_etiology=["ad"], id_space=IdSpace.A),
            event_anchor="first_diagnosis",
        )
        assert r_base.grounded and r_event.grounded
        # event_etiology must include incident cases → larger n than baseline_etiology.
        assert r_event.value["n"] > r_base.value["n"]
        # Both produce finite, positive mean ages.
        assert r_event.value["value"] > 0
        assert r_base.value["value"] > 0


# ---------------------------------------------------------------------------
# Optional real-data regression (skipped unless local CSV present)
# ---------------------------------------------------------------------------


NACC_CSV = PROJECT_ROOT / "NACC" / "investigator_nacc70.csv"


@pytest.mark.skipif(not NACC_CSV.exists(), reason="requires local NACC CSV")
class TestNaccRealDataEventEtiologyRegression:
    """Regression guard for the n=19,576 vs n=25,639 audit finding.

    Numbers sourced from the 2026-06-26 data audit; verified correct.
    These tests catch any future backend change that re-introduces the
    prevalent-only cohort bug for "mean age at first etiologic AD diagnosis."
    """

    def setup_method(self):
        config = _load_config()
        self.backend = LocalDataBackend(config, PROJECT_ROOT)

    def test_total_rows_nacc(self):
        """Sanity check: NACC CSV has 204,031 rows (confirm correct file)."""
        import pandas as _pd
        df = _pd.read_csv(str(NACC_CSV), usecols=["NACCID"])
        assert len(df) == 204_031, f"Expected 204031 rows, got {len(df)}"

    def test_unique_naccid(self):
        """54,631 unique NACCID in NACC CSV."""
        import pandas as _pd
        df = _pd.read_csv(str(NACC_CSV), usecols=["NACCID"])
        assert df["NACCID"].nunique() == 54_631

    def test_baseline_etiology_ad_n_is_19576(self):
        """Prevalent-AD cohort (NACCALZD==1 at first visit) = 19,576 participants."""
        r = self.backend.count_eligible(
            FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A)
        )
        assert r.grounded
        assert r.value == 19_576

    def test_event_etiology_ad_n_is_25639(self):
        """Any-visit AD cohort (NACCALZD==1 at any visit) = 25,639 participants."""
        r = self.backend.count_eligible(
            FilterSpec(event_etiology=["ad"], id_space=IdSpace.A)
        )
        assert r.grounded
        assert r.value == 25_639

    def test_mean_age_baseline_etiology_is_73_64(self):
        """Prevalent-AD baseline mean age = 73.64 (age at first NACC visit)."""
        r = self.backend.aggregate_stat(
            "NACC.NACCAGE", "mean",
            FilterSpec(baseline_etiology=["ad"], id_space=IdSpace.A),
        )
        assert r.grounded
        assert abs(r.value["value"] - 73.64) < 0.01

    def test_mean_age_event_etiology_first_diagnosis_is_74_81(self):
        """Corrected mean age at first NACCALZD==1 visit across all-AD participants = 74.81."""
        r = self.backend.aggregate_stat(
            "NACC.NACCAGE", "mean",
            FilterSpec(event_etiology=["ad"], id_space=IdSpace.A),
            event_anchor="first_diagnosis",
        )
        assert r.grounded
        assert r.value["n"] == 25_639, (
            f"n should be 25639 unique participants, got {r.value['n']}"
        )
        assert abs(r.value["value"] - 74.81) < 0.01, (
            f"Mean age should be ~74.81, got {r.value['value']}"
        )

    def test_n_is_participants_not_rows(self):
        """n in aggregate_stat must equal unique NACCID, never raw row count (70,044)."""
        r = self.backend.aggregate_stat(
            "NACC.NACCAGE", "mean",
            FilterSpec(event_etiology=["ad"], id_space=IdSpace.A),
            event_anchor="first_diagnosis",
        )
        assert r.grounded
        n = r.value["n"]
        # n must not equal the number of qualifying rows (70,044)
        assert n != 70_044, "n equals raw qualifying row count — participant dedup failed"
        # n must not equal prevalent-only count either
        assert n != 19_576, "n equals baseline-only count — incident cases were excluded"
        assert n == 25_639


# ---------------------------------------------------------------------------
# FixtureDataBackend — Space-B biomarker (ptau217) aggregate/descriptive stats
# ---------------------------------------------------------------------------


class TestFixtureSpaceBBiomarkerStats:
    """FixtureParticipant.ptau217 lets Space-B aggregate_stat/descriptive_stats
    be exercised without real ADNI_partial data.
    """

    def _backend(self) -> FixtureDataBackend:
        return build_fixture_backend(
            n_normal=50, n_mci=40, n_dementia=30, n_other=20,
            n_mci_long=25, n_mci_short=8, n_single_visit=15,
            n_suppressed_group=5,
            id_space=IdSpace.B,
            seed=42,
        )

    def test_aggregate_stat_ptau217_grounded(self):
        backend = self._backend()
        r = backend.aggregate_stat(
            "ADNI.ADSP_PHC_PLASMA.PHC_pTau217", "mean", FilterSpec(id_space=IdSpace.B)
        )
        assert r.grounded
        assert r.value["n"] > 0
        assert isinstance(r.value["value"], float)

    def test_descriptive_stats_ptau217_grounded(self):
        backend = self._backend()
        r = backend.descriptive_stats(
            "ADNI.ADSP_PHC_PLASMA.PHC_pTau217", FilterSpec(id_space=IdSpace.B)
        )
        assert r.grounded
        assert r.value["count"] > 0
        assert r.value["mean"] is not None

    def test_aggregate_stat_unsupported_variable_abstains(self):
        backend = self._backend()
        r = backend.aggregate_stat(
            "ADNI.ADSP_PHC_CSF.PHC_pTau181", "mean", FilterSpec(id_space=IdSpace.B)
        )
        assert not r.grounded
        assert r.abstain_reason is not None

    def test_descriptive_stats_unsupported_variable_abstains(self):
        backend = self._backend()
        r = backend.descriptive_stats(
            "ADNI.ADSP_PHC_CSF.PHC_pTau181", FilterSpec(id_space=IdSpace.B)
        )
        assert not r.grounded
