"""FixtureDataBackend (P2) — synthetic in-memory backend for tests/CI.

No real data files are needed. Invariant 4 still holds: only aggregates cross
any boundary from this backend to callers.
"""

from __future__ import annotations

import random
import statistics as _statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any

from alz_agent.backend.interface import DataBackend
from alz_agent.constants import SUPPRESSED, SUPPRESSION_THRESHOLD, suppress
from alz_agent.schemas import (
    CoverageStats,
    DataEvidence,
    FilterSpec,
    IdSpace,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Synthetic participant records
# ---------------------------------------------------------------------------


@dataclass
class FixtureVisit:
    vnum: int
    year: int
    diagnosis: int  # NACCUDSD: 1=Normal, 2=Impaired-not-MCI, 3=MCI, 4=Dementia (all-cause)


@dataclass
class FixtureParticipant:
    pid: str
    visits: list[FixtureVisit]
    modalities: frozenset[str] = field(default_factory=frozenset)
    baseline_etiology: str | None = None  # "ad" if NACCALZD==1 at FIRST visit; else None
    age_at_baseline: int | None = None    # synthetic age in years at first visit
    # Visit number (1-based) at which NACCALZD==1 first appears; None if never.
    # For baseline-AD participants this equals 1.
    # For incident-AD participants (NACCALZD==0 at visit 1) this is > 1.
    ad_etiology_first_vnum: int | None = None
    # Age at the first NACCALZD==1 visit; None if ad_etiology_first_vnum is None.
    age_at_first_ad_etiology: int | None = None
    # Synthetic Space-B (ADNI) plasma p-tau217 value, for testing Space-B
    # aggregate_stat/descriptive_stats without real ADNI_partial data.
    ptau217: float | None = None

    @property
    def baseline_dx(self) -> int | None:
        return self.visits[0].diagnosis if self.visits else None

    @property
    def first_year(self) -> int | None:
        return self.visits[0].year if self.visits else None

    @property
    def last_year(self) -> int | None:
        return self.visits[-1].year if self.visits else None

    @property
    def followup_years(self) -> int:
        if not self.visits:
            return 0
        return self.last_year - self.first_year  # type: ignore[operator]

    @property
    def has_any_ad_etiology(self) -> bool:
        return self.ad_etiology_first_vnum is not None


# ---------------------------------------------------------------------------
# FixtureDataBackend
# ---------------------------------------------------------------------------

# Must stay in sync with _NACCUDSD_MAP in local_backend.py.
# NACCUDSD: 1=Normal, 2=Impaired-not-MCI, 3=MCI, 4=Dementia (all-cause).
# "other" and "ad" are intentionally excluded.
_NACCUDSD_MAP: dict[str, int] = {
    "normal": 1, "nc": 1, "nl": 1, "1": 1,
    "impaired not mci": 2, "impaired-not-mci": 2, "impaired_not_mci": 2,
    "2": 2,
    "mci": 3, "3": 3,
    "dementia": 4, "4": 4,
}

# Space B (ADNI) cognitive-status codes — must stay in sync with _ADNI_DX_MAP.
# "ad" is intentionally excluded: use baseline_etiology=["ad"] instead.
_ADNI_DX_MAP: dict[str, int] = {
    "normal": 1, "cn": 1, "nc": 1, "nl": 1, "1": 1,
    "mci": 2, "2": 2,
    "dementia": 3, "3": 3,
}

# Accepted labels for AD etiologic attribution — must stay in sync with local_backend.py.
_AD_ETIOLOGY_LABELS: frozenset[str] = frozenset({
    "ad",
    "alzheimers",
    "alzheimer",
    "alzheimer's disease",
    "alzheimer disease",
})


class FixtureDataBackend(DataBackend):
    """In-memory backend backed by a list of FixtureParticipant records.

    All methods return aggregates with suppression enforced — same contract
    as LocalDataBackend.
    """

    def __init__(
        self,
        participants: list[FixtureParticipant],
        id_space: IdSpace = IdSpace.A,
        coverage_map: dict[str, CoverageStats] | None = None,
        known_modalities: set[str] | None = None,
        dx_map: dict[str, int] | None = None,
    ) -> None:
        self._participants = participants
        self._id_space = id_space
        self._coverage_map: dict[str, CoverageStats] = coverage_map or {}
        # Modalities the fixture "knows about" — unknown modalities trigger abstain.
        # Defaults to the union of all participants' modalities.
        self._known_modalities: set[str] = (
            known_modalities
            if known_modalities is not None
            else {m for p in participants for m in p.modalities}
        )
        # Diagnosis label map: defaults to NACCUDSD for Space A, ADNI codes for Space B.
        self._dx_map: dict[str, int] = (
            dx_map
            if dx_map is not None
            else (_NACCUDSD_MAP if id_space == IdSpace.A else _ADNI_DX_MAP)
        )

    # ── Eligibility helpers ───────────────────────────────────────────────────

    def _parse_dx(self, labels: list[str]) -> set[int]:
        result: set[int] = set()
        for lbl in labels:
            k = lbl.strip().lower()
            if k not in self._dx_map:
                valid = sorted(set(self._dx_map.keys()))
                raise ValueError(
                    f"Unknown cognitive status label '{lbl}'. Valid: {valid}"
                )
            result.add(self._dx_map[k])
        return result

    def _eligible(self, filter_spec: FilterSpec) -> list[FixtureParticipant]:
        if filter_spec.label_certainty is not None:
            raise ValueError(
                "label_certainty filter is not yet implemented. "
                "Pass label_certainty=None or omit the field."
            )

        if filter_spec.id_space != self._id_space:
            return []

        pts = list(self._participants)

        # 1. Baseline-diagnosis filter (cognitive status)
        if filter_spec.baseline_diagnosis:
            dx_ints = self._parse_dx(filter_spec.baseline_diagnosis)
            pts = [p for p in pts if p.baseline_dx in dx_ints]

        # 2. Baseline-etiology filter (NACCALZD==1 at first visit only)
        if filter_spec.baseline_etiology:
            for lbl in filter_spec.baseline_etiology:
                k = lbl.strip().lower()
                if k not in _AD_ETIOLOGY_LABELS:
                    raise ValueError(
                        f"Unknown baseline_etiology label '{lbl}'. "
                        f"Supported: {sorted(_AD_ETIOLOGY_LABELS)}"
                    )
            # All currently supported etiology labels denote AD.
            pts = [p for p in pts if p.baseline_etiology == "ad"]

        # 2b. Event-etiology filter (NACCALZD==1 at ANY visit — longitudinal)
        if filter_spec.event_etiology:
            for lbl in filter_spec.event_etiology:
                k = lbl.strip().lower()
                if k not in _AD_ETIOLOGY_LABELS:
                    raise ValueError(
                        f"Unknown event_etiology label '{lbl}'. "
                        f"Supported: {sorted(_AD_ETIOLOGY_LABELS)}"
                    )
            pts = [p for p in pts if p.has_any_ad_etiology]

        # 3. Visit-window filter
        if filter_spec.visit_window:
            lo, hi = filter_spec.visit_window
            pts = [
                p for p in pts
                if any(lo <= v.vnum <= hi for v in p.visits)
            ]

        # 4. Required-modalities filter
        if filter_spec.required_modalities:
            unknown = [m for m in filter_spec.required_modalities if m not in self._known_modalities]
            if unknown:
                raise ValueError(
                    f"Modalities not in fixture data: {unknown}"
                )
            for mod in filter_spec.required_modalities:
                pts = [p for p in pts if mod in p.modalities]

        return pts

    @staticmethod
    def _suppressed_int(n: int) -> int | None:
        s = suppress(n)
        return None if s is SUPPRESSED else int(s)

    # ── DataBackend interface ─────────────────────────────────────────────────

    def count_eligible(self, filter_spec: FilterSpec) -> ToolResult[int]:
        try:
            pts = self._eligible(filter_spec)
        except ValueError as exc:
            return ToolResult.abstain(str(exc))

        n = len(pts)
        s = suppress(n)
        if s is SUPPRESSED:
            return ToolResult.abstain(
                f"count suppressed — {n} is below reporting threshold ({SUPPRESSION_THRESHOLD})"
            )

        return ToolResult.ok(
            int(s),
            evidence=[DataEvidence(
                query_spec=filter_spec.model_dump(mode="json"),
                aggregate_ref=f"fixture.count_eligible={n}",
            )],
        )

    def modality_intersection(
        self,
        modalities: list[str],
        filter_spec: FilterSpec,
    ) -> ToolResult[dict[str, int | None]]:
        try:
            pts = self._eligible(filter_spec)
        except ValueError as exc:
            return ToolResult.abstain(str(exc))

        # Gather modality sets
        all_known = {m for p in self._participants for m in p.modalities}
        unknown = [m for m in modalities if m not in all_known]
        if unknown:
            return ToolResult.abstain(
                f"Modalities not in fixture data: {unknown}"
            )

        per_mod: dict[str, set[str]] = {
            m: {p.pid for p in pts if m in p.modalities} for m in modalities
        }

        counts: dict[str, int | None] = {}
        for r in range(1, len(modalities) + 1):
            for combo in combinations(sorted(modalities), r):
                key = ",".join(combo)
                inter = set.intersection(*(per_mod[m] for m in combo))
                counts[key] = self._suppressed_int(len(inter))

        return ToolResult.ok(
            counts,
            evidence=[DataEvidence(
                query_spec={**filter_spec.model_dump(mode="json"), "modalities": modalities},
                aggregate_ref=f"fixture.modality_intersection({modalities})",
            )],
        )

    def followup_distribution(
        self,
        horizon: int,
        filter_spec: FilterSpec,
    ) -> ToolResult[dict[str, int | None]]:
        try:
            pts = self._eligible(filter_spec)
        except ValueError as exc:
            return ToolResult.abstain(str(exc))

        if not pts:
            return ToolResult.ok({"sufficient": 0, "insufficient": 0, "censored": 0})

        n_suf = sum(1 for p in pts if p.followup_years >= horizon)
        n_cen = sum(1 for p in pts if len(p.visits) == 1)
        n_ins = sum(
            1 for p in pts
            if len(p.visits) > 1 and p.followup_years < horizon
        )

        ev = DataEvidence(
            query_spec={**filter_spec.model_dump(mode="json"), "horizon_years": horizon},
            aggregate_ref=(
                f"fixture.followup: sufficient={n_suf}, "
                f"insufficient={n_ins}, censored={n_cen}"
            ),
        )
        return ToolResult.ok(
            {
                "sufficient": self._suppressed_int(n_suf),
                "insufficient": self._suppressed_int(n_ins),
                "censored": self._suppressed_int(n_cen),
            },
            evidence=[ev],
        )

    def variable_coverage(self, variable_id: str) -> ToolResult[CoverageStats]:
        if variable_id in self._coverage_map:
            stats = self._coverage_map[variable_id]
            return ToolResult.ok(
                stats,
                evidence=[DataEvidence(
                    query_spec={"variable_id": variable_id},
                    aggregate_ref=f"fixture.coverage: n={stats.n_available}",
                )],
            )
        return ToolResult.abstain(
            f"Variable '{variable_id}' not in fixture coverage map"
        )

    def _extract_values(
        self,
        variable_id: str,
        pts: list[FixtureParticipant],
        event_anchor: str | None,
        filter_spec: FilterSpec,
    ) -> list[float] | None:
        """Map a variable_id to synthetic numeric values, or None if unsupported.

        Supports age-like variables (Space A) and 'ptau217' (Space B biomarker
        smoke-test field). Extend FixtureParticipant for other numeric fields.
        """
        vid = variable_id.lower()
        if "ptau217" in vid:
            return [p.ptau217 for p in pts if p.ptau217 is not None]
        if any(kw in vid for kw in ("age", "naccage")):
            # When event_etiology + first_diagnosis anchor: use age at first
            # AD-etiology visit. Otherwise: use age at baseline (first visit).
            use_event_age = (
                event_anchor == "first_diagnosis"
                and filter_spec.event_etiology is not None
            )
            if use_event_age:
                return [p.age_at_first_ad_etiology for p in pts if p.age_at_first_ad_etiology is not None]
            return [p.age_at_baseline for p in pts if p.age_at_baseline is not None]
        return None

    def aggregate_stat(
        self,
        variable_id: str,
        stat_type: str,
        filter_spec: FilterSpec,
        event_anchor: str | None = None,
    ) -> ToolResult[dict]:
        stat_type = stat_type.lower()
        if stat_type not in ("mean", "median", "std", "min", "max"):
            return ToolResult.abstain(
                f"Unknown stat_type '{stat_type}'. Valid: mean, median, std, min, max."
            )

        try:
            pts = self._eligible(filter_spec)
        except ValueError as exc:
            return ToolResult.abstain(str(exc))

        values = self._extract_values(variable_id, pts, event_anchor, filter_spec)
        if values is None:
            return ToolResult.abstain(
                f"Fixture aggregate_stat does not support variable '{variable_id}'. "
                "Extend FixtureParticipant to add other numeric fields as needed."
            )
        n = len(values)

        s = suppress(n)
        if s is SUPPRESSED:
            return ToolResult.abstain(
                f"count suppressed — {n} below threshold ({SUPPRESSION_THRESHOLD})"
            )
        if n == 0:
            return ToolResult.abstain(f"No values available for '{variable_id}' in eligible cohort")

        stat_map: dict[str, Any] = {
            "mean":   lambda v: round(_statistics.mean(v), 2),
            "median": lambda v: round(_statistics.median(v), 2),
            "std":    lambda v: round(_statistics.stdev(v), 2) if len(v) > 1 else 0.0,
            "min":    lambda v: float(min(v)),
            "max":    lambda v: float(max(v)),
        }
        stat_value = stat_map[stat_type](values)

        anchor_label = event_anchor or "baseline"
        return ToolResult.ok(
            {"stat_type": stat_type, "value": stat_value, "n": n},
            evidence=[DataEvidence(
                query_spec={
                    **filter_spec.model_dump(mode="json"),
                    "variable_id": variable_id,
                    "stat_type": stat_type,
                    "event_anchor": event_anchor,
                },
                aggregate_ref=(
                    f"fixture.aggregate_stat({variable_id},{stat_type},{anchor_label}): "
                    f"{stat_value} (n={n})"
                ),
            )],
        )

    def descriptive_stats(
        self,
        variable_id: str,
        filter_spec: FilterSpec,
        event_anchor: str | None = None,
    ) -> ToolResult[dict]:
        """Fixture descriptive_stats: supports age and ptau217 variables."""
        try:
            pts = self._eligible(filter_spec)
        except ValueError as exc:
            return ToolResult.abstain(str(exc))

        values = self._extract_values(variable_id, pts, event_anchor, filter_spec)
        if values is None:
            return ToolResult.abstain(
                f"Fixture descriptive_stats does not support variable '{variable_id}'."
            )
        n = len(values)
        n_total = len(pts)

        s = suppress(n)
        if s is SUPPRESSED:
            return ToolResult.abstain(
                f"count suppressed — {n} below threshold ({SUPPRESSION_THRESHOLD})"
            )
        if n == 0:
            return ToolResult.abstain(f"No values available for '{variable_id}' in eligible cohort")

        import statistics as _st
        sorted_values = sorted(values)
        q1 = float(sorted_values[len(sorted_values) // 4])
        q3 = float(sorted_values[3 * len(sorted_values) // 4])
        result = {
            "n_total": n_total,
            "n_missing": n_total - n,
            "count": n,
            "mean": round(_st.mean(values), 3),
            "median": round(_st.median(values), 3),
            "sd": round(_st.stdev(values), 3) if n > 1 else None,
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(q3 - q1, 3),
            "min_val": float(min(values)),
            "max_val": float(max(values)),
            "proportion": None,
        }
        anchor_label = event_anchor or "baseline"
        return ToolResult.ok(
            result,
            evidence=[DataEvidence(
                query_spec={
                    **filter_spec.model_dump(mode="json"),
                    "variable_id": variable_id,
                    "event_anchor": event_anchor,
                },
                aggregate_ref=(
                    f"fixture.descriptive_stats({variable_id},{anchor_label}): "
                    f"n={n} mean={result['mean']} median={result['median']} "
                    f"sd={result['sd']} iqr={result['iqr']}"
                ),
            )],
        )

    def variant_subject_counts(
        self,
        variable_ids: list[str],
        filter_spec: FilterSpec,
    ) -> ToolResult[dict[str, int | None]]:
        try:
            pts = self._eligible(filter_spec)
        except ValueError as exc:
            return ToolResult.abstain(str(exc))

        if not pts:
            return ToolResult.abstain("No eligible participants for the given filter")

        raw_counts: dict[str, int] = {}
        for vid in variable_ids:
            values = self._extract_values(vid, pts, None, filter_spec)
            if values is not None:
                raw_counts[vid] = len(values)

        if not raw_counts:
            return ToolResult.abstain(
                "None of the requested variable_ids were found in fixture data"
            )

        counts: dict[str, int | None] = {
            vid: self._suppressed_int(n) for vid, n in raw_counts.items()
        }
        return ToolResult.ok(
            counts,
            evidence=[DataEvidence(
                query_spec={
                    **filter_spec.model_dump(mode="json"),
                    "variable_ids": variable_ids,
                },
                aggregate_ref=f"fixture.variant_subject_counts: {raw_counts}",
            )],
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_fixture_backend(
    *,
    n_normal: int = 50,
    n_mci: int = 40,
    n_dementia: int = 30,
    n_other: int = 20,
    n_mci_long: int = 25,          # MCI with ≥3 years follow-up
    n_mci_short: int = 8,          # MCI with 1–2 years follow-up
    n_single_visit: int = 15,      # only 1 visit (censored)
    n_suppressed_group: int = 5,   # tiny group with vnum range [15–16] for suppression tests
    n_mci_ad: int = 15,            # MCI + AD etiology at BASELINE (baseline_etiology="ad")
    n_dementia_ad: int = 15,       # Dementia + AD etiology at BASELINE (baseline_etiology="ad")
    n_incident_ad: int = 12,       # Normal/MCI at baseline, AD etiology first appears at visit 3
    visits_per_participant: int = 4,
    visit_year_start: int = 2010,
    modality_fractions: dict[str, float] | None = None,
    id_space: IdSpace = IdSpace.A,
    seed: int = 42,
) -> FixtureDataBackend:
    """Build a synthetic FixtureDataBackend with controllable group sizes.

    Special group: `n_suppressed_group` MCI participants with NACCVNUM in [15, 16].
    Filtering on visit_window=(15, 16) + baseline_diagnosis=["MCI"] returns exactly
    n_suppressed_group participants, triggering suppression when 0 < n < threshold.

    Etiology groups:
        n_mci_ad      — MCI at baseline with AD etiology at visit 1 (baseline_etiology="ad").
        n_dementia_ad — Dementia at baseline with AD etiology at visit 1 (baseline_etiology="ad").
        n_incident_ad — Normal/MCI at baseline; NACCALZD first appears at visit 3
                        (baseline_etiology=None, ad_etiology_first_vnum=3).
                        Used to test event_etiology filtering vs baseline_etiology filtering:
                        baseline_etiology=["ad"] excludes incident cases;
                        event_etiology=["ad"]    includes them.
    The existing n_mci / n_dementia participants have baseline_etiology=None
    (non-AD or unknown etiology), serving as the "non-AD" comparison group.
    Normal and impaired-not-MCI participants always have baseline_etiology=None.
    """
    rng = random.Random(seed)
    age_rng = random.Random(seed + 1000)  # separate stream so age rolls don't shift modality RNG
    ptau217_rng = random.Random(seed + 2000)  # separate stream for synthetic biomarker values

    if modality_fractions is None:
        modality_fractions = {
            "cognition": 0.9,
            "vascular": 0.8,
            "fluid_biomarker": 0.05,
            "mri": 0.15,
            "pet": 0.04,
            "neuropath": 0.0,
        }

    all_modalities = list(modality_fractions.keys())
    pid_counter = [0]

    def _pid() -> str:
        pid_counter[0] += 1
        return f"FIXTURE{pid_counter[0]:05d}"

    def _modalities_for(fracs: dict[str, float]) -> frozenset[str]:
        return frozenset(m for m, f in fracs.items() if rng.random() < f)

    # Typical age ranges by baseline cognitive status
    _AGE_RANGES = {1: (60, 80), 2: (65, 82), 3: (65, 85), 4: (68, 90)}

    def _make_participant(
        dx: int,
        n_visits: int,
        first_year: int,
        vnum_start: int = 1,
        extra_modalities: frozenset[str] = frozenset(),
        etiology: str | None = None,
        incident_ad_vnum: int | None = None,
        incident_ad_age: int | None = None,
    ) -> FixtureParticipant:
        """Build a FixtureParticipant.

        etiology="ad" → NACCALZD==1 at visit 1 (baseline AD).
        incident_ad_vnum — if set, NACCALZD first appears at this visit number
            (vnum_start + offset); participant has no baseline etiology.
        incident_ad_age — age at the incident_ad_vnum visit.
        """
        visits = [
            FixtureVisit(
                vnum=vnum_start + i,
                year=first_year + i,
                diagnosis=dx,
            )
            for i in range(n_visits)
        ]
        mods = _modalities_for(modality_fractions) | extra_modalities
        # Always give cognition to MCI (dx=3) and Dementia (dx=4) participants
        if dx in (3, 4):
            mods = mods | {"cognition"}
        lo, hi = _AGE_RANGES.get(dx, (60, 85))
        age = age_rng.randint(lo, hi)  # use separate stream to keep main rng state stable

        # Populate AD-etiology visit fields
        if etiology == "ad":
            ad_first_vnum = vnum_start
            age_at_ad = age  # baseline age == first AD visit age
        elif incident_ad_vnum is not None:
            ad_first_vnum = incident_ad_vnum
            age_at_ad = incident_ad_age if incident_ad_age is not None else age + (incident_ad_vnum - vnum_start)
        else:
            ad_first_vnum = None
            age_at_ad = None

        return FixtureParticipant(
            pid=_pid(), visits=visits, modalities=mods,
            baseline_etiology=etiology, age_at_baseline=age,
            ad_etiology_first_vnum=ad_first_vnum,
            age_at_first_ad_etiology=age_at_ad,
            ptau217=round(ptau217_rng.gauss(0, 1), 3),
        )

    participants: list[FixtureParticipant] = []

    # Normal participants (NACCUDSD=1)
    for _ in range(n_normal):
        nv = rng.randint(1, visits_per_participant)
        yr = rng.randint(visit_year_start, visit_year_start + 5)
        participants.append(_make_participant(1, nv, yr))

    # MCI (NACCUDSD=3) with long follow-up (sufficient for 3-year horizon)
    for _ in range(n_mci_long):
        participants.append(_make_participant(3, 4, rng.randint(visit_year_start, visit_year_start + 3)))

    # MCI (NACCUDSD=3) with short follow-up (insufficient for 3-year horizon)
    for _ in range(n_mci_short):
        participants.append(_make_participant(3, 2, rng.randint(visit_year_start, visit_year_start + 5)))

    # Single-visit MCI (NACCUDSD=3, censored)
    for _ in range(n_single_visit):
        participants.append(_make_participant(3, 1, rng.randint(visit_year_start, visit_year_start + 5)))

    # Remaining regular MCI (NACCUDSD=3)
    remaining_mci = n_mci - n_mci_long - n_mci_short - n_single_visit
    for _ in range(max(0, remaining_mci)):
        nv = rng.randint(2, visits_per_participant)
        yr = rng.randint(visit_year_start, visit_year_start + 5)
        participants.append(_make_participant(3, nv, yr))

    # Dementia participants (NACCUDSD=4)
    # Dementia (NACCUDSD=4)
    for _ in range(n_dementia):
        nv = rng.randint(1, visits_per_participant)
        yr = rng.randint(visit_year_start, visit_year_start + 5)
        participants.append(_make_participant(4, nv, yr))

    # Impaired-not-MCI (NACCUDSD=2)
    for _ in range(n_other):
        nv = rng.randint(1, visits_per_participant)
        yr = rng.randint(visit_year_start, visit_year_start + 5)
        participants.append(_make_participant(2, nv, yr))

    # Suppression test group: MCI (NACCUDSD=3) with VNUM range [15, 16]
    for _ in range(n_suppressed_group):
        p = FixtureParticipant(
            pid=_pid(),
            visits=[
                FixtureVisit(vnum=15, year=visit_year_start + 15, diagnosis=3),
                FixtureVisit(vnum=16, year=visit_year_start + 16, diagnosis=3),
            ],
            modalities=frozenset({"cognition"}),
            age_at_baseline=age_rng.randint(65, 80),
            ptau217=round(ptau217_rng.gauss(0, 1), 3),
        )
        participants.append(p)

    # MCI + AD etiology (baseline_etiology="ad")
    for _ in range(n_mci_ad):
        nv = rng.randint(2, visits_per_participant)
        yr = rng.randint(visit_year_start, visit_year_start + 5)
        participants.append(_make_participant(3, nv, yr, etiology="ad"))

    # Dementia + AD etiology (baseline_etiology="ad")
    for _ in range(n_dementia_ad):
        nv = rng.randint(1, visits_per_participant)
        yr = rng.randint(visit_year_start, visit_year_start + 5)
        participants.append(_make_participant(4, nv, yr, etiology="ad"))

    # Incident AD: baseline_etiology=None (no AD at visit 1), NACCALZD first at visit 3.
    # These participants appear in event_etiology=["ad"] results but NOT baseline_etiology ones.
    # age_at_first_ad_etiology is derived as age_at_baseline + (3 - 1) = baseline_age + 2.
    for _ in range(n_incident_ad):
        yr = rng.randint(visit_year_start, visit_year_start + 3)
        participants.append(_make_participant(3, 4, yr, incident_ad_vnum=3))

    # Pre-compute coverage for common variables
    total = len(participants)
    cov_map = {
        "NACC.CDRSUM": CoverageStats(
            n_available=total,
            longitudinal_availability=int(total * 0.7),
            missingness_rate=0.0,
            last_computed=datetime.now(timezone.utc),
        ),
        "ADSP_PHC.Cognition.PHC_MEM": CoverageStats(
            n_available=int(total * 0.9),
            longitudinal_availability=int(total * 0.5),
            missingness_rate=0.1,
            last_computed=datetime.now(timezone.utc),
        ),
    }

    return FixtureDataBackend(
        participants=participants,
        id_space=id_space,
        coverage_map=cov_map,
    )
