"""LocalDataBackend (P2) — reads real local data; returns aggregates only.

Invariant 4 (AGENTS.md): subject-level rows NEVER leave this module.
Every public method returns aggregates and applies suppress() before returning.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from alz_agent.backend.interface import DataBackend
from alz_agent.catalog.adni_parser import resolve_latest_snapshot
from alz_agent.constants import SUPPRESSED, SUPPRESSION_THRESHOLD, suppress
from alz_agent.schemas import (
    CoverageStats,
    DataEvidence,
    FilterSpec,
    IdSpace,
    ToolResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnosis label → integer maps
# ---------------------------------------------------------------------------

# NACCUDSD cognitive-status codes (UDS3 RDD):
#   1 = Normal cognition
#   2 = Impaired-not-MCI
#   3 = MCI
#   4 = Dementia (all-cause; NOT AD-specific — do NOT map "ad" here)
#
# "other" is intentionally excluded: it is ambiguous (could mean other
# etiologic diagnosis) and does not have a stable, documented mapping.
# Use "impaired not mci" / "impaired-not-mci" / "impaired_not_mci" for NACCUDSD=2.
#
# AD etiologic diagnosis is a separate variable (NACCETPR / NACCELUC).
_NACCUDSD_MAP: dict[str, int] = {
    "normal": 1,
    "nc": 1,
    "1": 1,

    "impaired not mci": 2,
    "impaired-not-mci": 2,
    "impaired_not_mci": 2,
    "2": 2,

    "mci": 3,
    "3": 3,

    "dementia": 4,
    "4": 4,
}

_ADNI_DX_MAP: dict[str, int] = {
    "normal": 1, "cn": 1, "nc": 1, "nl": 1, "1": 1,
    "mci": 2, "2": 2,
    "dementia": 3, "3": 3,
    # "ad" intentionally excluded: AD is etiologic, not cognitive-status.
    # Use baseline_etiology=["ad"] instead.
}

# All accepted labels for AD etiologic attribution (baseline_etiology).
_AD_ETIOLOGY_LABELS: frozenset[str] = frozenset({
    "ad",
    "alzheimers",
    "alzheimer",
    "alzheimer's disease",
    "alzheimer disease",
})


def _parse_dx_labels(labels: list[str], dx_map: dict[str, int], source: str) -> list[int]:
    result: list[int] = []
    for lbl in labels:
        k = lbl.strip().lower()
        if k not in dx_map:
            valid = sorted(set(dx_map.keys()))
            raise ValueError(f"Unknown {source} diagnosis label '{lbl}'. Valid: {valid}")
        result.append(dx_map[k])
    return list(set(result))


def _validate_etiology_labels(labels: list[str]) -> None:
    """Raise ValueError for any unrecognised baseline_etiology label."""
    for lbl in labels:
        k = lbl.strip().lower()
        if k not in _AD_ETIOLOGY_LABELS:
            raise ValueError(
                f"Unknown baseline_etiology label '{lbl}'. "
                f"Supported: {sorted(_AD_ETIOLOGY_LABELS)}"
            )


def _select_adni_baseline(df: pd.DataFrame, dx_col: str) -> pd.DataFrame:
    """Select the earliest valid-dx row per RID.

    Ordering: prefers chronologically earliest EXAMDATE when the column is present
    and parseable; falls back to VISCODE lexicographic sort for rows where EXAMDATE
    is absent or cannot be parsed. Only rows with a non-null dx value are
    considered, preserving the behaviour of the original implementation.
    """
    valid = df[df[dx_col].notna()].copy()
    if valid.empty:
        return valid

    if "EXAMDATE" in valid.columns:
        valid["_parsed_date"] = pd.to_datetime(valid["EXAMDATE"], errors="coerce")
        # _has_date=0 for real dates (sorts first); 1 for NaT (sorts last within RID).
        # This avoids sentinel-date overflow issues with datetime64[ns] limits.
        valid["_has_date"] = valid["_parsed_date"].isna().astype(int)
        sorted_df = valid.sort_values(["RID", "_has_date", "_parsed_date", "VISCODE"])
        return sorted_df.drop_duplicates("RID", keep="first").drop(
            columns=["_parsed_date", "_has_date"]
        )

    return valid.sort_values(["RID", "VISCODE"]).drop_duplicates("RID", keep="first")


# Candidate date columns across ADNI tables, in priority order (DXSUM uses
# EXAMDATE, biomarker tables use DRAWDATE, most assessment tables use VISDATE,
# MOCA additionally has a plain DATE column).
_ADNI_DATE_COLUMN_CANDIDATES: tuple[str, ...] = ("EXAMDATE", "DRAWDATE", "VISDATE", "DATE")

# Alignment-rule text for DataEvidence.baseline_alignment_rule, keyed by event_anchor.
# Describes the deterministic rule actually applied — never "first available measurement"
# — so narration can report exactly how a measure row was matched to its anchor visit.
_ADNI_ALIGNMENT_RULE_BY_ANCHOR: dict[str | None, str] = {
    None: (
        "Measure row aligned to each participant's own baseline DXSUM visit "
        "(earliest valid-diagnosis row by EXAMDATE, VISCODE tiebreak) via matching "
        "VISCODE. A participant with no measure-table row at that exact visit is "
        "excluded from the analytic count — their first available measurement at a "
        "later visit is never silently substituted as baseline."
    ),
    "baseline": (
        "Measure row aligned to each participant's own baseline DXSUM visit "
        "(earliest valid-diagnosis row by EXAMDATE, VISCODE tiebreak) via matching "
        "VISCODE. A participant with no measure-table row at that exact visit is "
        "excluded from the analytic count — their first available measurement at a "
        "later visit is never silently substituted as baseline."
    ),
    "first_diagnosis": (
        "Measure row aligned to each participant's own first qualifying diagnosis "
        "visit (located in DXSUM) via matching VISCODE. A participant with no "
        "measure-table row at that exact visit is excluded."
    ),
    "first_conversion_visit": (
        "Measure row aligned to each participant's own first all-cause-dementia "
        "conversion visit (located in DXSUM) via matching VISCODE. A participant "
        "with no measure-table row at that exact visit is excluded."
    ),
}


def _adsp_phc_source_level(tblname: str, fldname: str) -> str:
    """Derive DataEvidence.source_level for an ADSP-PHC column from its own naming
    convention, instead of a hardcoded constant. ADSP-PHC harmonized (cross-phase)
    columns are prefixed 'PHC_'; everything else in an ADSP_PHC_* table is a raw,
    per-phase harmonization-input column. Falls back to the generic 'raw' label for
    tables outside the ADSP-PHC release, where this naming convention doesn't apply.
    """
    if not tblname.upper().startswith("ADSP_PHC"):
        return "raw"
    return "phc_harmonized" if fldname.startswith("PHC_") else "phc_raw_column"


def _select_adni_event_rows(sub: pd.DataFrame, event_anchor: str | None) -> pd.DataFrame:
    """Return one row per RID from an arbitrary ADNI table for the given event anchor.

    Supported anchors:
      None / 'baseline' — earliest row per RID (by date column if present, else VISCODE)
      'last_visit'       — latest row per RID

    Any other anchor (e.g. NACC-specific 'first_diagnosis') falls back to baseline —
    those anchors reference diagnosis-defining logic that doesn't generalize across
    ADNI's many tables the way it does for NACC's single wide file.
    """
    date_col = next((c for c in _ADNI_DATE_COLUMN_CANDIDATES if c in sub.columns), None)
    keep_last = event_anchor == "last_visit"

    if date_col:
        sub = sub.copy()
        sub["_parsed_date"] = pd.to_datetime(sub[date_col], errors="coerce")
        if keep_last:
            sorted_df = sub.sort_values(["RID", "_parsed_date"], na_position="first")
        else:
            sub["_has_date"] = sub["_parsed_date"].isna().astype(int)
            sort_cols = ["RID", "_has_date", "_parsed_date"]
            if "VISCODE" in sub.columns:
                sort_cols.append("VISCODE")
            sorted_df = sub.sort_values(sort_cols)
        result = sorted_df.drop_duplicates("RID", keep=("last" if keep_last else "first"))
        return result.drop(columns=[c for c in ("_parsed_date", "_has_date") if c in result.columns])

    if "VISCODE" in sub.columns:
        return sub.sort_values(["RID", "VISCODE"]).drop_duplicates(
            "RID", keep=("last" if keep_last else "first")
        )
    return sub.drop_duplicates("RID", keep=("last" if keep_last else "first"))


def _combo_key(mods: list[str] | tuple[str, ...]) -> str:
    return ",".join(sorted(mods))


class _MissingModalityError(RuntimeError):
    pass


class _AdniAnchorUnavailable(RuntimeError):
    """Raised when a Space B diagnosis-anchored event visit cannot be computed.

    Signals a *capability / data-availability gap* (missing DXSUM table, missing
    etiology columns, or no visit matching the anchor condition) — distinct from
    a suppressed or empty aggregate. Callers translate it into an abstain whose
    wording names the gap rather than implying the underlying data is absent.
    """


# Derived Space B variable ids (computed, not read directly from a single column).
_ADNI_DERIVED_AGE_AT_EVENT = "ADNI.DERIVED.age_at_event"

# ADNI PTDOB is recorded as month/year only (e.g. "04/1931"); day-of-month is not
# collected. Ages derived from it carry up to ~1 month of approximation error.
_ADNI_DOB_PRECISION = "month/year"


# ---------------------------------------------------------------------------
# Follow-up distribution status definitions (module-level constants)
# Derived directly from LocalDataBackend.followup_distribution() logic.
# Must stay in sync with n_suf / n_ins / n_cen computation below.
# ---------------------------------------------------------------------------

FOLLOWUP_STATUS_DEFINITIONS: dict[str, str] = {
    "sufficient": (
        "The participant has at least 2 visits and a follow-up span "
        "(last VISITYR minus first VISITYR) >= the specified horizon years. "
        "Enough observed time exists to determine whether conversion occurred."
    ),
    "insufficient": (
        "The participant has at least 2 visits but a follow-up span less than "
        "the specified horizon years. Some follow-up is observed but insufficient "
        "to confirm or rule out conversion within the horizon."
    ),
    "censored": (
        "The participant has only a single recorded visit (group size = 1). "
        "No follow-up visit was observed, so conversion status cannot be determined "
        "for any horizon. This category is HORIZON-INDEPENDENT: the same set of "
        "single-visit participants is censored regardless of the horizon length."
    ),
}

CENSORING_HORIZON_DEPENDENT: bool = False


# ---------------------------------------------------------------------------
# LocalDataBackend
# ---------------------------------------------------------------------------


class LocalDataBackend(DataBackend):
    """Reads real local data files; all public methods return aggregates with suppression.

    Lazy-loading: data files are read on first access and cached for subsequent calls.
    Cache is keyed by (file_relative_path, frozenset_of_columns) so the same file can
    be loaded with different column subsets without duplication.
    """

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self._config = config
        self._root = project_root
        # (rel_path, columns_frozenset | None) → DataFrame
        self._df_cache: dict[tuple[str, Any], pd.DataFrame] = {}
        self._modality_ids_a_cache: dict[str, frozenset[str]] | None = None
        self._modality_ids_b_cache: dict[str, frozenset[str]] | None = None
        self._adni_path_cache: dict[str, str | None] = {}

    # ── Public config accessors (used by LocalAnalysisEngine) ─────────────────

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def project_root(self) -> Path:
        return self._root

    def eligible_ids(self, filter_spec: "FilterSpec") -> frozenset[str]:
        """Return eligible participant IDs for filter_spec (public wrapper).

        Used by LocalAnalysisEngine to obtain the cohort before loading
        analysis-specific columns. Raises ValueError / _MissingModalityError
        on invalid filter parameters (callers must handle).
        """
        if filter_spec.id_space == IdSpace.B:
            return self._eligible_rid_set(filter_spec)
        return self._eligible_nacc_ids(filter_spec)

    # ── File I/O helpers ──────────────────────────────────────────────────────

    def _load(
        self,
        rel_path: str,
        usecols: list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load and cache a CSV or xlsx file (column-set aware cache)."""
        cols_key: Any = frozenset(usecols) if usecols is not None else None
        cache_key = (rel_path, cols_key)
        if cache_key not in self._df_cache:
            full = self._root / rel_path
            if not full.exists():
                raise FileNotFoundError(f"Data file not found: {full}")
            if full.suffix in (".xlsx", ".xls"):
                self._df_cache[cache_key] = pd.read_excel(full, usecols=usecols, **kwargs)
            else:
                self._df_cache[cache_key] = pd.read_csv(
                    full, usecols=usecols, low_memory=False, **kwargs
                )
        return self._df_cache[cache_key]

    def _resolve_adni_table(self, tbl: str) -> str | None:
        """Resolve `backend.adni_data_files[tbl]` (a glob pattern relative to
        `adni.base_dir`) to a root-relative path string for the newest
        matching snapshot, or None if the table isn't configured or no file
        matches. Cached — the file set on disk doesn't change mid-run.
        """
        if tbl in self._adni_path_cache:
            return self._adni_path_cache[tbl]
        pattern = self._config.get("backend", {}).get("adni_data_files", {}).get(tbl)
        rel: str | None = None
        if pattern:
            base_dir = self._root / self._config.get("adni", {}).get("base_dir", "")
            resolved = resolve_latest_snapshot(base_dir, pattern)
            if resolved is not None:
                rel = str(resolved.relative_to(self._root))
        self._adni_path_cache[tbl] = rel
        return rel

    # ── Core data loaders ─────────────────────────────────────────────────────

    def _nacc_core(self) -> pd.DataFrame:
        """NACC: NACCID, NACCVNUM, VISITYR, NACCUDSD, NACCALZD — minimal columns, cached.

        NACCALZD is included for baseline_etiology filtering (NACCALZD==1 → presumptive AD).
        """
        return self._load(
            self._config["nacc"]["data_csv"],
            usecols=["NACCID", "NACCVNUM", "VISITYR", "NACCUDSD", "NACCALZD"],
            dtype={"NACCID": str},
        )

    def _nacc_with_col(self, raw_code: str) -> pd.DataFrame:
        """Read NACCID + NACCVNUM + one extra column for coverage computation."""
        return self._load(
            self._config["nacc"]["data_csv"],
            usecols=["NACCID", "NACCVNUM", raw_code],
            dtype={"NACCID": str},
        )

    def _nacc_with_cols(self, extra_cols: list[str]) -> pd.DataFrame:
        """Read NACCID + NACCVNUM + multiple extra columns (column-set cached)."""
        cols = list(dict.fromkeys(["NACCID", "NACCVNUM"] + extra_cols))
        return self._load(
            self._config["nacc"]["data_csv"],
            usecols=cols,
            dtype={"NACCID": str},
        )

    def _modality_ids_a(self) -> dict[str, frozenset[str]]:
        """NACCIDs present in each ADSP-PHC modality file (Space A)."""
        if self._modality_ids_a_cache is not None:
            return self._modality_ids_a_cache

        result: dict[str, frozenset[str]] = {}
        data_files = self._config.get("backend", {}).get("adsp_phc_data_files", {})
        for modality, rel_paths in data_files.items():
            ids: set[str] = set()
            for rel in rel_paths:
                full = self._root / rel
                if not full.exists():
                    logger.warning("ADSP-PHC data file missing (modality=%s): %s", modality, full)
                    continue
                try:
                    if full.suffix in (".xlsx", ".xls"):
                        df = pd.read_excel(full, usecols=["NACCID"])
                    else:
                        df = pd.read_csv(full, usecols=["NACCID"])
                    ids.update(df["NACCID"].dropna().astype(str))
                except Exception as exc:
                    logger.warning("Could not load %s: %s", rel, exc)
            result[modality] = frozenset(ids)

        self._modality_ids_a_cache = result
        return result

    def _adni_core(self) -> pd.DataFrame:
        """ADNI DXSUM: RID, VISCODE, DIAGNOSIS/DXCURREN, plus optional EXAMDATE and
        etiology columns (DXMDUE, DXDDUE, DXAD) when present in the file.

        EXAMDATE enables chronological baseline-row selection (see _select_adni_baseline).
        DXMDUE/DXDDUE/DXAD are used for baseline_etiology filtering.
        """
        dxsum = self._resolve_adni_table("DXSUM") or ""
        if not dxsum:
            return pd.DataFrame(columns=["RID", "VISCODE", "DIAGNOSIS"])
        header = pd.read_csv(self._root / dxsum, nrows=0)
        avail = set(header.columns)
        dx_col = "DIAGNOSIS" if "DIAGNOSIS" in avail else "DXCURREN"
        usecols = ["RID", "VISCODE", dx_col]
        for col in ("EXAMDATE", "DXMDUE", "DXDDUE", "DXAD"):
            if col in avail:
                usecols.append(col)
        return self._load(dxsum, usecols=usecols, dtype={"RID": str})

    def _modality_ids_b(self) -> dict[str, frozenset[str]]:
        """RIDs present in each ADNI modality (Space B)."""
        if self._modality_ids_b_cache is not None:
            return self._modality_ids_b_cache

        # Keep in sync with catalog.adni_parser._TABLE_MODALITY — this dict is
        # what actually drives assess_modality_overlap for Space B.
        table_to_modality = {
            "CDR": "cognition", "MMSE": "cognition",
            "MOCA": "cognition", "ADAS": "cognition",
            "ADSP_PHC_COGN": "cognition",
            "ADSP_PHC_CSF": "fluid_biomarker",
            "ADSP_PHC_PLASMA": "fluid_biomarker",
            "ADSP_PHC_T1_FS": "mri",
            "FAMHXPAR": "demographic", "FAMHXSIB": "demographic",
            # P0 expansion (2026-07-15 LONI download)
            "PTDEMOG": "demographic",
            "FAQ": "cognition", "NEUROBAT": "cognition",
            "UCSFFSX7": "mri", "MRIMETA": "mri", "MRI3META": "mri",
            "AMYMETA": "pet", "AV45META": "pet", "TAUMETA": "pet",
            "UCBERKELEY_AMY_6MM": "pet", "UCBERKELEY_TAU_6MM": "pet",
            "UPENN_PLASMA_FUJIREBIO_QUANTERIX": "fluid_biomarker",
            "UPENNBIOMK_MASTER": "fluid_biomarker",
            "UPENNBIOMK_ROCHE_ELECSYS": "fluid_biomarker",
        }
        ids_by_mod: dict[str, set[str]] = {}
        for tbl, mod in table_to_modality.items():
            rel = self._resolve_adni_table(tbl)
            if not rel:
                continue
            try:
                df = self._load(rel, usecols=["RID"], dtype={"RID": str})
                ids_by_mod.setdefault(mod, set()).update(df["RID"].dropna().astype(str))
            except Exception as exc:
                logger.warning("Could not load ADNI table %s: %s", tbl, exc)

        self._modality_ids_b_cache = {k: frozenset(v) for k, v in ids_by_mod.items()}
        return self._modality_ids_b_cache

    # ── Eligibility helpers ───────────────────────────────────────────────────

    def _eligible_nacc_ids(self, filter_spec: FilterSpec) -> frozenset[str]:
        """Return eligible NACCIDs for Space A after all filters."""
        if filter_spec.label_certainty is not None:
            raise ValueError(
                "label_certainty filter is not yet implemented for Space A (NACC). "
                "Pass label_certainty=None or omit the field."
            )

        df = self._nacc_core()
        eligible: set[str] = set(df["NACCID"].unique())

        # Compute baseline rows once when either diagnosis or etiology filter is active.
        if filter_spec.baseline_diagnosis or filter_spec.baseline_etiology:
            first = (
                df.sort_values("NACCVNUM")
                .drop_duplicates("NACCID", keep="first")
                .copy()
            )
            first["NACCUDSD"] = pd.to_numeric(first["NACCUDSD"], errors="coerce").astype("Int64")
            first["NACCALZD"] = pd.to_numeric(first["NACCALZD"], errors="coerce").astype("Int64")

        # 1. Baseline-diagnosis filter (cognitive status; first visit per participant)
        if filter_spec.baseline_diagnosis:
            dx_ints = _parse_dx_labels(
                filter_spec.baseline_diagnosis, _NACCUDSD_MAP, "NACCUDSD cognitive status"
            )
            eligible &= set(first.loc[first["NACCUDSD"].isin(dx_ints), "NACCID"])

        # 2. Baseline-etiology filter (NACCALZD==1 at FIRST visit only → prevalent AD)
        if filter_spec.baseline_etiology:
            _validate_etiology_labels(filter_spec.baseline_etiology)
            eligible &= set(first.loc[first["NACCALZD"] == 1, "NACCID"])

        # 2b. Event-etiology filter (NACCALZD==1 at ANY visit → incident + prevalent AD)
        if filter_spec.event_etiology:
            _validate_etiology_labels(filter_spec.event_etiology)
            df["NACCALZD"] = pd.to_numeric(df["NACCALZD"], errors="coerce").astype("Int64")
            ever_alzd = set(df.loc[df["NACCALZD"] == 1, "NACCID"])
            eligible &= ever_alzd

        # 3. Visit-window filter (participants who have ≥1 visit within window)
        if filter_spec.visit_window:
            lo, hi = filter_spec.visit_window
            in_window = set(df.loc[(df["NACCVNUM"] >= lo) & (df["NACCVNUM"] <= hi), "NACCID"])
            eligible &= in_window

        # 4. Required-modalities filter
        if filter_spec.required_modalities:
            mod_ids = self._modality_ids_a()
            for mod in filter_spec.required_modalities:
                if mod not in mod_ids:
                    raise _MissingModalityError(
                        f"Modality '{mod}' not found in local ADSP-PHC data files"
                    )
                eligible &= mod_ids[mod]

        return frozenset(eligible)

    def _eligible_rid_set(self, filter_spec: FilterSpec) -> frozenset[str]:
        """Return eligible RIDs for Space B after all filters."""
        if filter_spec.label_certainty is not None:
            raise ValueError(
                "label_certainty filter is not yet implemented for Space B (ADNI). "
                "Pass label_certainty=None or omit the field."
            )

        df = self._adni_core()
        if df.empty:
            return frozenset()

        dx_col = "DIAGNOSIS" if "DIAGNOSIS" in df.columns else "DXCURREN"
        eligible: set[str] = set(df["RID"].unique())

        # Compute the baseline row per RID when either filter is active.
        # _select_adni_baseline: prefers EXAMDATE chronological order; falls back
        # to VISCODE lexicographic sort when EXAMDATE is absent or unparseable.
        if filter_spec.baseline_diagnosis or filter_spec.baseline_etiology:
            first = _select_adni_baseline(df, dx_col)

        # 1. Baseline-diagnosis filter (cognitive status only)
        if filter_spec.baseline_diagnosis:
            dx_ints = _parse_dx_labels(filter_spec.baseline_diagnosis, _ADNI_DX_MAP, "ADNI")
            first_dx = first.copy()
            first_dx[dx_col] = pd.to_numeric(first_dx[dx_col], errors="coerce").astype("Int64")
            eligible &= set(first_dx.loc[first_dx[dx_col].isin(dx_ints), "RID"])

        # 2. Baseline-etiology filter (DXMDUE==1 OR DXDDUE==1 OR DXAD==1) at FIRST visit
        if filter_spec.baseline_etiology:
            _validate_etiology_labels(filter_spec.baseline_etiology)
            etiol_cols = [c for c in ("DXMDUE", "DXDDUE", "DXAD") if c in first.columns]
            if not etiol_cols:
                raise ValueError(
                    "ADNI baseline_etiology filter requested but no etiology columns "
                    "(DXMDUE, DXDDUE, DXAD) found in DXSUM. Cannot filter by etiology."
                )
            ad_mask = pd.Series(False, index=first.index)
            for col in etiol_cols:
                ad_mask |= (pd.to_numeric(first[col], errors="coerce") == 1)
            eligible &= set(first.loc[ad_mask, "RID"])

        # 2b. Event-etiology filter (DXAD/DXMDUE/DXDDUE==1 at ANY visit → incident + prevalent AD).
        # Mirrors the NACC event_etiology path (NACCALZD==1 on any visit): required for
        # "first observed AD diagnosis in ADNI" questions, which must scan all visits, not
        # just the baseline row.
        if filter_spec.event_etiology:
            _validate_etiology_labels(filter_spec.event_etiology)
            etiol_cols = [c for c in ("DXMDUE", "DXDDUE", "DXAD") if c in df.columns]
            if not etiol_cols:
                raise ValueError(
                    "ADNI event_etiology filter requested but no etiology columns "
                    "(DXMDUE, DXDDUE, DXAD) found in DXSUM. Cannot filter by etiology."
                )
            ever_mask = pd.Series(False, index=df.index)
            for col in etiol_cols:
                ever_mask |= (pd.to_numeric(df[col], errors="coerce") == 1)
            eligible &= set(df.loc[ever_mask, "RID"])

        # 3. Required-modalities filter
        if filter_spec.required_modalities:
            mod_ids = self._modality_ids_b()
            for mod in filter_spec.required_modalities:
                if mod not in mod_ids:
                    raise _MissingModalityError(
                        f"Space B modality '{mod}' not found in ADNI data files"
                    )
                eligible &= mod_ids[mod]

        return frozenset(eligible)

    # ── Suppression helpers ───────────────────────────────────────────────────

    @staticmethod
    def _suppressed_int(n: int) -> int | None:
        """Apply suppress(); return None for SUPPRESSED, else int."""
        s = suppress(n)
        return None if s is SUPPRESSED else int(s)

    # ── DataBackend interface ─────────────────────────────────────────────────

    def count_eligible(self, filter_spec: FilterSpec) -> ToolResult[int]:
        try:
            if filter_spec.id_space == IdSpace.B:
                ids = self._eligible_rid_set(filter_spec)
            else:
                ids = self._eligible_nacc_ids(filter_spec)
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))

        n = len(ids)
        result = suppress(n)
        if result is SUPPRESSED:
            return ToolResult.abstain(
                f"count suppressed — {n} is below reporting threshold ({SUPPRESSION_THRESHOLD})"
            )

        return ToolResult.ok(
            int(result),
            evidence=[DataEvidence(
                query_spec=filter_spec.model_dump(mode="json"),
                aggregate_ref=f"count_eligible={n}",
            )],
        )

    def modality_intersection(
        self,
        modalities: list[str],
        filter_spec: FilterSpec,
    ) -> ToolResult[dict[str, int | None]]:
        """Return counts for all non-empty subsets of the requested modalities.

        None values in the dict indicate counts that were suppressed.
        """
        try:
            if filter_spec.id_space == IdSpace.B:
                eligible = self._eligible_rid_set(filter_spec)
                available = self._modality_ids_b()
            else:
                eligible = self._eligible_nacc_ids(filter_spec)
                available = self._modality_ids_a()
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))

        unknown = [m for m in modalities if m not in available]
        if unknown:
            source = "ADNI" if filter_spec.id_space == IdSpace.B else "ADSP-PHC"
            return ToolResult.abstain(
                f"Modalities {unknown} have no configured/loaded data tables in {source} "
                "for this deployment — a coverage/capability gap, not evidence that the "
                "participants lack those measurements."
            )

        per_mod: dict[str, frozenset[str]] = {
            m: eligible & available[m] for m in modalities
        }

        counts: dict[str, int | None] = {}
        suppressed_keys: list[str] = []

        for r in range(1, len(modalities) + 1):
            for combo in combinations(sorted(modalities), r):
                key = _combo_key(combo)
                intersection = frozenset.intersection(*(per_mod[m] for m in combo))
                v = self._suppressed_int(len(intersection))
                counts[key] = v
                if v is None:
                    suppressed_keys.append(key)

        ev_note = f"modality_intersection({modalities}), eligible_base={len(eligible)}"
        if suppressed_keys:
            ev_note += f"; suppressed: {suppressed_keys}"

        return ToolResult.ok(
            counts,
            evidence=[DataEvidence(
                query_spec={**filter_spec.model_dump(mode="json"), "modalities": modalities},
                aggregate_ref=ev_note,
            )],
        )

    def followup_distribution(
        self,
        horizon: int,
        filter_spec: FilterSpec,
    ) -> ToolResult[dict[str, int | None]]:
        """Classify eligible participants by follow-up length relative to horizon (years).

        Result keys:
          sufficient   — last visit ≥ horizon years after first visit
          insufficient — ≥2 visits but span < horizon years
          censored     — only 1 visit (no follow-up data; horizon-independent)
        None values indicate a suppressed bucket.
        """
        if filter_spec.id_space == IdSpace.B:
            return self._followup_b(horizon, filter_spec)

        try:
            eligible = self._eligible_nacc_ids(filter_spec)
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))

        if not eligible:
            return ToolResult.ok(
                {"sufficient": 0, "insufficient": 0, "censored": 0},
                evidence=[DataEvidence(
                    query_spec={**filter_spec.model_dump(mode="json"), "horizon_years": horizon},
                    aggregate_ref="followup_distribution: 0 eligible",
                )],
            )

        df = self._nacc_core()
        sub = df[df["NACCID"].isin(eligible)].dropna(subset=["VISITYR"])

        grp_yr = sub.groupby("NACCID")["VISITYR"]
        grp_sz = sub.groupby("NACCID").size()

        fu = grp_yr.max() - grp_yr.min()

        n_suf = int((fu >= horizon).sum())
        n_ins = int(((fu < horizon) & (grp_sz > 1)).sum())
        n_cen = int((grp_sz == 1).sum())

        ev = DataEvidence(
            query_spec={**filter_spec.model_dump(mode="json"), "horizon_years": horizon},
            aggregate_ref=(
                f"followup_distribution(horizon={horizon}): "
                f"sufficient={n_suf}, insufficient={n_ins}, censored={n_cen}"
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

    def _followup_b(self, horizon: int, filter_spec: FilterSpec) -> ToolResult[dict[str, int | None]]:
        """Follow-up distribution for Space B (ADNI), using CDR VISDATE."""
        try:
            eligible = self._eligible_rid_set(filter_spec)
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))

        if not eligible:
            return ToolResult.ok({"sufficient": 0, "insufficient": 0, "censored": 0})

        cdr_rel = self._resolve_adni_table("CDR") or ""
        if not cdr_rel:
            return ToolResult.abstain(
                "Space B follow-up uses the ADNI CDR table (RID, VISDATE), which is not "
                "configured/available in this deployment — a pipeline capability/coverage gap, "
                "not missing or low-quality data."
            )

        try:
            cdr = self._load(cdr_rel, usecols=["RID", "VISDATE"], dtype={"RID": str})
        except Exception as exc:
            return ToolResult.abstain(f"Could not load CDR for Space B follow-up: {exc}")

        sub = cdr[cdr["RID"].isin(eligible)].copy()
        sub["_yr"] = pd.to_datetime(sub["VISDATE"], errors="coerce").dt.year
        sub = sub.dropna(subset=["_yr"])

        grp_yr = sub.groupby("RID")["_yr"]
        grp_sz = sub.groupby("RID").size()
        fu = grp_yr.max() - grp_yr.min()

        n_suf = int((fu >= horizon).sum())
        n_ins = int(((fu < horizon) & (grp_sz > 1)).sum())
        n_cen = int((grp_sz == 1).sum())

        ev = DataEvidence(
            query_spec={**filter_spec.model_dump(mode="json"), "horizon_years": horizon},
            aggregate_ref=(
                f"ADNI followup(horizon={horizon}): "
                f"sufficient={n_suf}, insufficient={n_ins}, censored={n_cen}"
            ),
            source_table="CDR",
            source_field="VISDATE",
            source_level="raw",
        )
        return ToolResult.ok(
            {
                "sufficient": self._suppressed_int(n_suf),
                "insufficient": self._suppressed_int(n_ins),
                "censored": self._suppressed_int(n_cen),
            },
            evidence=[ev],
        )

    def aggregate_stat(
        self,
        variable_id: str,
        stat_type: str,
        filter_spec: FilterSpec,
        event_anchor: str | None = None,
    ) -> ToolResult[dict]:
        if filter_spec.id_space == IdSpace.B:
            if variable_id == _ADNI_DERIVED_AGE_AT_EVENT:
                return self._aggregate_age_at_event_adni(stat_type, filter_spec, event_anchor)
            return self._aggregate_stat_adni(variable_id, stat_type, filter_spec, event_anchor)

        parts = variable_id.split(".", maxsplit=1)
        if len(parts) < 2 or parts[0] != "NACC":
            return ToolResult.abstain(
                f"aggregate_stat only supports NACC variables for Space A (got '{variable_id}')"
            )
        raw_code = parts[1]

        stat_type = stat_type.lower()
        if stat_type not in ("mean", "median", "std", "min", "max"):
            return ToolResult.abstain(
                f"Unknown stat_type '{stat_type}'. Valid: mean, median, std, min, max."
            )

        # Get eligible participant IDs
        try:
            eligible = self._eligible_nacc_ids(filter_spec)
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))

        if not eligible:
            return ToolResult.abstain("No eligible participants for the given filter")

        # Determine which extra columns to load alongside the target variable
        extra: list[str] = [raw_code]
        if event_anchor == "first_diagnosis":
            if filter_spec.baseline_etiology or filter_spec.event_etiology:
                extra.append("NACCALZD")
            if filter_spec.baseline_diagnosis:
                extra.append("NACCUDSD")
        elif event_anchor == "first_conversion_visit":
            extra.append("NACCUDSD")

        try:
            df = self._nacc_with_cols(extra)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return ToolResult.abstain(f"Could not load NACC column '{raw_code}': {exc}")

        if raw_code not in df.columns:
            return ToolResult.abstain(
                f"Column '{raw_code}' not found in NACC data file"
            )

        sub = df[df["NACCID"].isin(eligible)].copy()

        # Determine the event row per participant
        event_rows = self._select_event_rows(sub, event_anchor, filter_spec)

        values = pd.to_numeric(event_rows[raw_code], errors="coerce").dropna()
        n = int(len(values))

        if n == 0:
            return ToolResult.abstain(
                f"No non-null '{raw_code}' values found for eligible cohort "
                f"(event_anchor={event_anchor!r})"
            )

        s = suppress(n)
        if s is SUPPRESSED:
            return ToolResult.abstain(
                f"count suppressed — {n} is below reporting threshold ({SUPPRESSION_THRESHOLD})"
            )

        stat_map = {
            "mean":   lambda v: round(float(v.mean()), 2),
            "median": lambda v: round(float(v.median()), 2),
            "std":    lambda v: round(float(v.std()), 2),
            "min":    lambda v: round(float(v.min()), 2),
            "max":    lambda v: round(float(v.max()), 2),
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
                    f"aggregate_stat({variable_id},{stat_type},{anchor_label}): "
                    f"{stat_value} (n={n})"
                ),
            )],
        )

    def _aggregate_stat_adni(
        self,
        variable_id: str,
        stat_type: str,
        filter_spec: FilterSpec,
        event_anchor: str | None,
    ) -> ToolResult[dict]:
        """Space-B (ADNI) aggregate_stat — mirrors the Space-A implementation above,
        keyed by RID and dispatched over the ADNI.<TABLE>.<FIELD> variable-id shape.
        """
        parts = variable_id.split(".", maxsplit=2)
        if len(parts) < 3 or parts[0] != "ADNI":
            return ToolResult.abstain(
                f"aggregate_stat for Space B requires an 'ADNI.<TABLE>.<FIELD>' "
                f"variable_id (got '{variable_id}')"
            )
        tblname, fldname = parts[1], parts[2]

        stat_type = stat_type.lower()
        if stat_type not in ("mean", "median", "std", "min", "max"):
            return ToolResult.abstain(
                f"Unknown stat_type '{stat_type}'. Valid: mean, median, std, min, max."
            )

        try:
            eligible = self._eligible_rid_set(filter_spec)
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))
        if not eligible:
            return ToolResult.abstain("No eligible participants for the given filter")

        rel = self._resolve_adni_table(tblname)
        if not rel:
            return ToolResult.abstain(f"ADNI table '{tblname}' is not a configured data file")

        try:
            df = self._load(rel, dtype={"RID": str})
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return ToolResult.abstain(f"Could not load ADNI table '{tblname}': {exc}")

        if fldname not in df.columns:
            return ToolResult.abstain(f"Column '{fldname}' not found in ADNI table '{tblname}'")

        sub = df[df["RID"].isin(eligible)].copy()
        try:
            event_rows, diagnosis_basis = self._select_adni_measure_rows(
                sub, event_anchor, filter_spec
            )
        except _AdniAnchorUnavailable as exc:
            return ToolResult.abstain(self._adni_capability_gap_msg(exc, tblname, fldname))

        # matched_n: participants who HAD a measure-table row at the aligned visit
        # (regardless of whether this specific field was non-null there).
        # unmatched_n: excluded because they had no measure-table row at that visit
        # at all. Both are suppressed independently (None if <10), same as any other
        # participant count leaving the backend (Invariant 4).
        matched_n_raw = int(len(event_rows))
        unmatched_n_raw = int(len(eligible)) - matched_n_raw
        matched_n = matched_n_raw if suppress(matched_n_raw) is not SUPPRESSED else None
        unmatched_n = unmatched_n_raw if suppress(unmatched_n_raw) is not SUPPRESSED else None

        values = pd.to_numeric(event_rows[fldname], errors="coerce").dropna()
        n = int(len(values))

        if n == 0:
            return ToolResult.abstain(
                f"No non-null '{fldname}' values found for eligible cohort "
                f"(event_anchor={event_anchor!r})"
            )

        s = suppress(n)
        if s is SUPPRESSED:
            return ToolResult.abstain(
                f"count suppressed — {n} is below reporting threshold ({SUPPRESSION_THRESHOLD})"
            )

        stat_map = {
            "mean":   lambda v: round(float(v.mean()), 2),
            "median": lambda v: round(float(v.median()), 2),
            "std":    lambda v: round(float(v.std()), 2),
            "min":    lambda v: round(float(v.min()), 2),
            "max":    lambda v: round(float(v.max()), 2),
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
                    f"aggregate_stat({variable_id},{stat_type},{anchor_label}): "
                    f"{stat_value} (n={n}, matched_n={matched_n}, unmatched_n={unmatched_n})"
                ),
                source_table=tblname,
                source_field=fldname,
                source_level=_adsp_phc_source_level(tblname, fldname),
                diagnosis_basis=diagnosis_basis,
                baseline_alignment_rule=_ADNI_ALIGNMENT_RULE_BY_ANCHOR.get(event_anchor),
                matched_n=matched_n,
                unmatched_n=unmatched_n,
            )],
        )

    # ── Event-anchor helpers ───────────────────────────────────────────────────

    def _select_adni_measure_rows(
        self,
        sub: "pd.DataFrame",
        event_anchor: str | None,
        filter_spec: "FilterSpec",
    ) -> tuple["pd.DataFrame", str | None]:
        """Return one row per RID from an ADNI measure table for the given anchor.

        For diagnosis anchors ('first_diagnosis', 'first_conversion_visit') the
        anchor visit is located in DXSUM (cross-table) and the measure row is the
        one sharing that (RID, VISCODE) — this is what makes 'MMSE at first AD
        diagnosis', 'age at first AD diagnosis', etc. all expressible through the
        same path. 'baseline'/None uses the SAME cross-table alignment mechanism,
        but against each RID's own baseline DXSUM visit rather than a
        diagnosis-matching visit — a measure row is only used as "baseline" when it
        shares that RID's baseline VISCODE; a participant whose first available
        measurement is at a LATER visit than their baseline diagnosis is correctly
        EXCLUDED, never silently substituted (empirically verified: only ~43% of
        CSF-table RIDs have their earliest draw at their own baseline VISCODE — see
        docs/BACKEND_CONTRACT.md). 'last_visit' is the one anchor that legitimately
        does NOT need diagnosis-visit alignment (it explicitly asks for the measure
        table's own latest row) and still delegates to the single-table selector.

        Returns (event_rows, diagnosis_basis); basis is None for non-diagnosis anchors.
        Raises _AdniAnchorUnavailable when the anchor visit cannot be computed.
        """
        if event_anchor in ("first_diagnosis", "first_conversion_visit"):
            anchor_df, basis = self._adni_event_anchor_visits(filter_spec, event_anchor)
            return self._align_measure_to_anchor_visits(sub, anchor_df), basis
        if event_anchor in (None, "baseline"):
            anchor_df = self._adni_baseline_diagnosis_visits(filter_spec)
            return self._align_measure_to_anchor_visits(sub, anchor_df), None
        return _select_adni_event_rows(sub, event_anchor), None

    @staticmethod
    def _align_measure_to_anchor_visits(
        sub: "pd.DataFrame", anchor_df: "pd.DataFrame"
    ) -> "pd.DataFrame":
        """Keep only measure-table rows whose VISCODE matches that RID's own anchor
        VISCODE (from anchor_df, columns RID/VISCODE). One row per RID (first match
        if duplicates). A RID with no matching-VISCODE row is simply absent from the
        result — this is what makes exclusion (rather than first-available
        substitution) the default outcome for a visit that isn't captured in the
        measure table. Shared by both the diagnosis-anchor and baseline-anchor paths
        in _select_adni_measure_rows() so the alignment rule is identical for both.

        Raises _AdniAnchorUnavailable if the measure table has no VISCODE column.
        """
        if "VISCODE" not in sub.columns:
            raise _AdniAnchorUnavailable(
                "measure table has no VISCODE column to align to the anchor visit"
            )
        rid_to_vc = dict(zip(anchor_df["RID"].astype(str), anchor_df["VISCODE"].astype(str)))
        s = sub.copy()
        s["_anchor_vc"] = s["RID"].astype(str).map(rid_to_vc)
        rows = s[s["_anchor_vc"] == s["VISCODE"].astype(str)].drop_duplicates(
            "RID", keep="first"
        )
        return rows.drop(columns=["_anchor_vc"])

    def _adni_baseline_diagnosis_visits(self, filter_spec: "FilterSpec") -> "pd.DataFrame":
        """Per-RID baseline DXSUM visit — RID's own earliest valid-diagnosis row,
        unconditional on any diagnosis VALUE (unlike _adni_event_anchor_visits, which
        requires a specific diagnosis/etiology match). Returns columns RID, VISCODE
        (used by _align_measure_to_anchor_visits for the cross-table join).

        Reuses _select_adni_baseline() — the SAME baseline-row selection already used
        to build the eligibility cohort in _eligible_rid_set() — so "baseline" means
        the identical visit for both cohort membership and measurement alignment.

        Raises _AdniAnchorUnavailable when DXSUM/VISCODE is unavailable.
        """
        dx = self._adni_core()
        if dx.empty:
            raise _AdniAnchorUnavailable(
                "ADNI DXSUM diagnosis table is not available, so the baseline visit "
                "cannot be located"
            )
        if "VISCODE" not in dx.columns:
            raise _AdniAnchorUnavailable(
                "ADNI DXSUM has no VISCODE column to align measure tables to the "
                "baseline visit"
            )
        dx_col = "DIAGNOSIS" if "DIAGNOSIS" in dx.columns else "DXCURREN"
        if dx_col not in dx.columns:
            raise _AdniAnchorUnavailable(
                f"ADNI DXSUM has no {dx_col} column to select a valid baseline row"
            )
        baseline = _select_adni_baseline(dx, dx_col)
        if baseline.empty:
            raise _AdniAnchorUnavailable("no valid baseline DXSUM row found for any RID")
        return baseline[["RID", "VISCODE"]].copy()

    def _adni_event_anchor_visits(
        self,
        filter_spec: "FilterSpec",
        event_anchor: str | None,
    ) -> tuple["pd.DataFrame", str]:
        """Per-RID first visit satisfying the diagnosis anchor, located in DXSUM.

        The diagnosis basis is derived from filter_spec (mirroring the NACC
        _select_event_rows convention), not a separate parameter:
          - event_anchor == 'first_conversion_visit' → first all-cause dementia visit
          - filter_spec.event_etiology / baseline_etiology set → first etiologic-AD
            visit (DXAD/DXMDUE/DXDDUE == 1)
          - otherwise → first all-cause dementia visit (DXSUM.DIAGNOSIS == 3)

        Returns (anchor_df, diagnosis_basis) where anchor_df has columns
        RID, VISCODE, and _anchor_date (parsed EXAMDATE, may be NaT). Raises
        _AdniAnchorUnavailable when the anchor cannot be computed.
        """
        dx = self._adni_core()
        if dx.empty:
            raise _AdniAnchorUnavailable(
                "ADNI DXSUM diagnosis table is not available, so a diagnosis-anchor "
                "visit cannot be located"
            )
        if "VISCODE" not in dx.columns:
            raise _AdniAnchorUnavailable(
                "ADNI DXSUM has no VISCODE column to align measure tables to the anchor visit"
            )
        dx = dx.copy()
        dx_col = "DIAGNOSIS" if "DIAGNOSIS" in dx.columns else "DXCURREN"
        etiol_cols = [c for c in ("DXMDUE", "DXDDUE", "DXAD") if c in dx.columns]

        use_etiologic = (
            event_anchor != "first_conversion_visit"
            and bool(filter_spec.event_etiology or filter_spec.baseline_etiology)
        )

        if use_etiologic:
            if not etiol_cols:
                raise _AdniAnchorUnavailable(
                    "etiologic-AD anchor requested but ADNI DXSUM has no "
                    "DXAD/DXMDUE/DXDDUE etiology columns"
                )
            mask = pd.Series(False, index=dx.index)
            for col in etiol_cols:
                mask |= (pd.to_numeric(dx[col], errors="coerce") == 1)
            basis = f"{'/'.join(etiol_cols)}=1 etiologic AD (raw DXSUM)"
        else:
            if dx_col not in dx.columns:
                raise _AdniAnchorUnavailable(
                    f"all-cause dementia anchor requested but ADNI DXSUM has no {dx_col} column"
                )
            mask = (pd.to_numeric(dx[dx_col], errors="coerce") == 3)
            basis = f"{dx_col}=3 all-cause dementia (raw DXSUM)"

        matching = dx[mask].copy()
        if matching.empty:
            raise _AdniAnchorUnavailable(
                f"no ADNI DXSUM visit matches the anchor condition ({basis})"
            )

        if "EXAMDATE" in matching.columns:
            matching["_anchor_date"] = pd.to_datetime(matching["EXAMDATE"], errors="coerce")
        else:
            matching["_anchor_date"] = pd.NaT
        matching["_no_date"] = matching["_anchor_date"].isna().astype(int)
        matching = matching.sort_values(["RID", "_no_date", "_anchor_date", "VISCODE"])
        anchor = matching.drop_duplicates("RID", keep="first")
        return anchor[["RID", "VISCODE", "_anchor_date"]].copy(), basis

    @staticmethod
    def _adni_capability_gap_msg(
        exc: "_AdniAnchorUnavailable", tblname: str, fldname: str
    ) -> str:
        """Abstain wording that names a capability/data gap rather than implying
        the underlying data is simply absent."""
        return (
            f"event-anchored value for ADNI.{tblname}.{fldname} is not computable: "
            f"{exc}. The requested table/field exists, but this specific "
            "event-anchored derivation is not available for the requested cohort — "
            "this is a pipeline capability/coverage gap, not a data-quality problem."
        )

    def _aggregate_age_at_event_adni(
        self,
        stat_type: str,
        filter_spec: "FilterSpec",
        event_anchor: str | None,
    ) -> ToolResult[dict]:
        """Derived Space B variable: age at the anchored diagnosis visit.

        Primary value = raw age from PTDEMOG.PTDOB (month/year) and the anchored
        DXSUM.EXAMDATE. Also computes a harmonized cross-check from
        ADSP_PHC_DEMODX.PHC_Age_Diagnosis at the same anchored visit; the
        cross-check value + provenance ride in the evidence so narration can report
        both with source_level. Both AD-etiologic and all-cause-dementia bases are
        selected via filter_spec (see _adni_event_anchor_visits) — callers emit one
        op per basis to report them separately.
        """
        stat_type = stat_type.lower()
        if stat_type not in ("mean", "median", "std", "min", "max"):
            return ToolResult.abstain(
                f"Unknown stat_type '{stat_type}'. Valid: mean, median, std, min, max."
            )

        anchor = event_anchor or "first_diagnosis"

        try:
            eligible = self._eligible_rid_set(filter_spec)
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))
        if not eligible:
            return ToolResult.abstain("No eligible participants for the given filter")

        try:
            anchor_df, basis = self._adni_event_anchor_visits(filter_spec, anchor)
        except _AdniAnchorUnavailable as exc:
            return ToolResult.abstain(
                f"event-anchored derived age is not computable: {exc}. The underlying "
                "ADNI data exists (PTDEMOG.PTDOB, DXSUM.EXAMDATE), but this event-anchored "
                "derivation is unavailable for the requested cohort — a pipeline "
                "capability/coverage gap, not a data-quality problem."
            )

        anchor_df = anchor_df[anchor_df["RID"].astype(str).isin(eligible)].copy()
        if anchor_df.empty:
            return ToolResult.abstain(
                "No eligible participants have an anchored diagnosis visit in DXSUM"
            )

        stat_fn = {
            "mean":   lambda v: round(float(v.mean()), 2),
            "median": lambda v: round(float(v.median()), 2),
            "std":    lambda v: round(float(v.std()), 2),
            "min":    lambda v: round(float(v.min()), 2),
            "max":    lambda v: round(float(v.max()), 2),
        }[stat_type]

        # ---- Raw age: PTDEMOG.PTDOB (month/year) vs anchored EXAMDATE ----
        raw_age = self._raw_age_at_anchor(anchor_df)
        raw_n = int(len(raw_age))
        if raw_n == 0:
            return ToolResult.abstain(
                "event-anchored derived age is not computable: PTDEMOG.PTDOB or the "
                "anchored DXSUM.EXAMDATE could not be parsed for any eligible participant "
                "(capability/coverage gap, not a data-quality problem)."
            )
        s = suppress(raw_n)
        if s is SUPPRESSED:
            return ToolResult.abstain(
                f"count suppressed — {raw_n} is below reporting threshold "
                f"({SUPPRESSION_THRESHOLD})"
            )
        raw_value = stat_fn(raw_age)

        # ---- Harmonized cross-check: ADSP_PHC_DEMODX.PHC_Age_Diagnosis ----
        harm_value: float | None = None
        harm_n = 0
        harm_series = self._harmonized_age_at_anchor(anchor_df)
        if harm_series is not None and len(harm_series) > 0:
            harm_n = int(len(harm_series))
            if suppress(harm_n) is not SUPPRESSED:
                harm_value = stat_fn(harm_series)

        anchor_label = anchor
        cross = (
            f"; harmonized cross-check (ADSP_PHC_DEMODX.PHC_Age_Diagnosis): "
            f"{harm_value} (n={harm_n})"
            if harm_value is not None
            else "; harmonized cross-check (PHC_Age_Diagnosis): not available"
        )
        return ToolResult.ok(
            {"stat_type": stat_type, "value": raw_value, "n": raw_n},
            evidence=[DataEvidence(
                query_spec={
                    **filter_spec.model_dump(mode="json"),
                    "variable_id": _ADNI_DERIVED_AGE_AT_EVENT,
                    "stat_type": stat_type,
                    "event_anchor": anchor,
                    "diagnosis_basis": basis,
                    "dob_precision": _ADNI_DOB_PRECISION,
                    "measurement_label": "first observed AD diagnosis in ADNI (study record)",
                    "harmonized_value": harm_value,
                    "harmonized_n": harm_n,
                },
                aggregate_ref=(
                    f"age_at_event(raw PTDOB+EXAMDATE, {stat_type}, {anchor_label}): "
                    f"{raw_value} (n={raw_n}){cross} "
                    f"[basis: {basis}; DOB precision: {_ADNI_DOB_PRECISION}]"
                ),
                source_table="PTDEMOG+DXSUM",
                source_field="PTDOB,EXAMDATE",
                source_level="raw",
                diagnosis_basis=basis,
            )],
        )

    def _raw_age_at_anchor(self, anchor_df: "pd.DataFrame") -> "pd.Series":
        """Age (years) at each RID's anchored visit from PTDEMOG.PTDOB.

        PTDOB is month/year (e.g. '04/1931'); parsed to first-of-month. Falls back
        to PTDOBYY (year, normalized to Jan 1) when PTDOB is missing/unparseable.
        Returns a Series of ages for RIDs where both DOB and anchor date parsed.
        """
        rel = self._resolve_adni_table("PTDEMOG")
        if not rel:
            return pd.Series(dtype=float)
        try:
            ptd = self._load(rel, dtype={"RID": str})
        except (ValueError, KeyError, FileNotFoundError):
            return pd.Series(dtype=float)
        if "PTDOB" not in ptd.columns and "PTDOBYY" not in ptd.columns:
            return pd.Series(dtype=float)

        dob = pd.Series(pd.NaT, index=ptd.index)
        if "PTDOB" in ptd.columns:
            dob = pd.to_datetime(ptd["PTDOB"], format="%m/%Y", errors="coerce")
        if "PTDOBYY" in ptd.columns:
            fallback = pd.to_datetime(ptd["PTDOBYY"], errors="coerce")
            dob = dob.fillna(fallback)
        dob_by_rid = (
            pd.DataFrame({"RID": ptd["RID"].astype(str), "_dob": dob})
            .dropna(subset=["_dob"])
            .drop_duplicates("RID", keep="first")
            .set_index("RID")["_dob"]
        )

        merged = anchor_df.copy()
        merged["RID"] = merged["RID"].astype(str)
        merged["_dob"] = merged["RID"].map(dob_by_rid)
        merged = merged.dropna(subset=["_dob", "_anchor_date"])
        if merged.empty:
            return pd.Series(dtype=float)
        age = (merged["_anchor_date"] - merged["_dob"]).dt.days / 365.25
        return age[age.notna()]

    def _harmonized_age_at_anchor(self, anchor_df: "pd.DataFrame") -> "pd.Series | None":
        """Harmonized age at the anchored visit from ADSP_PHC_DEMODX.PHC_Age_Diagnosis.

        Joined on (RID, VISCODE) to the anchor visit. Returns None when the table
        or field is unavailable (harmonized cross-check simply omitted).
        """
        rel = self._resolve_adni_table("ADSP_PHC_DEMODX")
        if not rel:
            return None
        try:
            demodx = self._load(rel, dtype={"RID": str})
        except (ValueError, KeyError, FileNotFoundError):
            return None
        if "PHC_Age_Diagnosis" not in demodx.columns or "VISCODE" not in demodx.columns:
            return None

        d = demodx[["RID", "VISCODE", "PHC_Age_Diagnosis"]].copy()
        d["RID"] = d["RID"].astype(str)
        d["VISCODE"] = d["VISCODE"].astype(str)
        anchor = anchor_df.copy()
        anchor["RID"] = anchor["RID"].astype(str)
        anchor["VISCODE"] = anchor["VISCODE"].astype(str)
        joined = anchor.merge(d, on=["RID", "VISCODE"], how="inner")
        vals = pd.to_numeric(joined["PHC_Age_Diagnosis"], errors="coerce").dropna()
        return vals if len(vals) > 0 else None

    # ── Event-anchor helper ────────────────────────────────────────────────────

    def _select_event_rows(
        self,
        sub: "pd.DataFrame",
        event_anchor: str | None,
        filter_spec: "FilterSpec",
    ) -> "pd.DataFrame":
        """Return one row per participant (NACCID) for the given event anchor.

        Supported anchors:
          None / 'baseline'            — first visit (min NACCVNUM)
          'first_diagnosis'            — first visit matching etiology or dx filter
          'last_visit'                 — last visit (max NACCVNUM)
          'first_conversion_visit'     — first visit where NACCUDSD >= 4 (any dementia)
        """
        import pandas as _pd

        if event_anchor == "first_diagnosis":
            if (
                (filter_spec.baseline_etiology or filter_spec.event_etiology)
                and "NACCALZD" in sub.columns
            ):
                sub = sub.copy()
                sub["NACCALZD"] = _pd.to_numeric(sub["NACCALZD"], errors="coerce").astype("Int64")
                event_sub = sub[sub["NACCALZD"] == 1]
            elif filter_spec.baseline_diagnosis and "NACCUDSD" in sub.columns:
                dx_ints = _parse_dx_labels(
                    filter_spec.baseline_diagnosis, _NACCUDSD_MAP, "NACCUDSD"
                )
                sub = sub.copy()
                sub["NACCUDSD"] = _pd.to_numeric(sub["NACCUDSD"], errors="coerce").astype("Int64")
                event_sub = sub[sub["NACCUDSD"].isin(dx_ints)]
            else:
                event_sub = sub
            return (
                event_sub.sort_values("NACCVNUM")
                .drop_duplicates("NACCID", keep="first")
            )

        if event_anchor == "last_visit":
            return (
                sub.sort_values("NACCVNUM")
                .drop_duplicates("NACCID", keep="last")
            )

        if event_anchor == "first_conversion_visit":
            if "NACCUDSD" not in sub.columns:
                return sub.sort_values("NACCVNUM").drop_duplicates("NACCID", keep="first")
            sub = sub.copy()
            sub["NACCUDSD"] = _pd.to_numeric(sub["NACCUDSD"], errors="coerce").astype("Int64")
            converted = sub[sub["NACCUDSD"] >= 4]
            return (
                converted.sort_values("NACCVNUM")
                .drop_duplicates("NACCID", keep="first")
            )

        # baseline / None
        return (
            sub.sort_values("NACCVNUM")
            .drop_duplicates("NACCID", keep="first")
        )

    def descriptive_stats(
        self,
        variable_id: str,
        filter_spec: "FilterSpec",
        event_anchor: str | None = None,
    ) -> "ToolResult[dict]":
        """Compute full descriptive statistics for a variable over a filtered cohort."""
        if filter_spec.id_space == IdSpace.B:
            return self._descriptive_stats_adni(variable_id, filter_spec, event_anchor)

        parts = variable_id.split(".", maxsplit=1)
        if len(parts) < 2 or parts[0] != "NACC":
            return ToolResult.abstain(
                f"descriptive_stats only supports NACC variables for Space A (got '{variable_id}')"
            )
        raw_code = parts[1]

        try:
            eligible = self._eligible_nacc_ids(filter_spec)
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))
        if not eligible:
            return ToolResult.abstain("No eligible participants for the given filter")

        extra: list[str] = [raw_code]
        if event_anchor == "first_diagnosis":
            if filter_spec.baseline_etiology or filter_spec.event_etiology:
                extra.append("NACCALZD")
            if filter_spec.baseline_diagnosis:
                extra.append("NACCUDSD")
        elif event_anchor == "first_conversion_visit":
            extra.append("NACCUDSD")

        try:
            df = self._nacc_with_cols(extra)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return ToolResult.abstain(f"Could not load NACC column '{raw_code}': {exc}")

        if raw_code not in df.columns:
            return ToolResult.abstain(
                f"Column '{raw_code}' not found in NACC data file"
            )

        sub = df[df["NACCID"].isin(eligible)].copy()
        event_rows = self._select_event_rows(sub, event_anchor, filter_spec)

        import numpy as np
        n_total = int(len(event_rows))
        series = pd.to_numeric(event_rows[raw_code], errors="coerce")
        n_missing = int(series.isna().sum())
        values = series.dropna()
        count = int(len(values))

        if count == 0:
            return ToolResult.abstain(
                f"No non-null '{raw_code}' values found for eligible cohort"
            )

        s = suppress(count)
        if s is SUPPRESSED:
            return ToolResult.abstain(
                f"count suppressed — {count} is below reporting threshold ({SUPPRESSION_THRESHOLD})"
            )

        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        result = {
            "n_total":    n_total,
            "n_missing":  n_missing,
            "count":      count,
            "mean":       round(float(values.mean()), 3),
            "median":     round(float(values.median()), 3),
            "sd":         round(float(values.std(ddof=1)), 3) if count > 1 else None,
            "q1":         round(q1, 3),
            "q3":         round(q3, 3),
            "iqr":        round(q3 - q1, 3),
            "min_val":    round(float(values.min()), 3),
            "max_val":    round(float(values.max()), 3),
            "proportion": round(float((values > 0).sum() / count), 4) if count > 0 else None,
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
                    f"descriptive_stats({variable_id},{anchor_label}): "
                    f"n={count} mean={result['mean']} median={result['median']} "
                    f"sd={result['sd']} iqr={result['iqr']}"
                ),
            )],
        )

    def _descriptive_stats_adni(
        self,
        variable_id: str,
        filter_spec: "FilterSpec",
        event_anchor: str | None,
    ) -> "ToolResult[dict]":
        """Space-B (ADNI) descriptive_stats — mirrors the Space-A implementation above."""
        if variable_id == _ADNI_DERIVED_AGE_AT_EVENT:
            return ToolResult.abstain(
                "ADNI.DERIVED.age_at_event is a computed scalar available via variable_stat "
                "(mean/median/etc.), not descriptive_stats. Request it with op_type "
                "'variable_stat'. This is a capability boundary, not missing data."
            )
        parts = variable_id.split(".", maxsplit=2)
        if len(parts) < 3 or parts[0] != "ADNI":
            return ToolResult.abstain(
                f"descriptive_stats for Space B requires an 'ADNI.<TABLE>.<FIELD>' "
                f"variable_id (got '{variable_id}')"
            )
        tblname, fldname = parts[1], parts[2]

        try:
            eligible = self._eligible_rid_set(filter_spec)
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))
        if not eligible:
            return ToolResult.abstain("No eligible participants for the given filter")

        rel = self._resolve_adni_table(tblname)
        if not rel:
            return ToolResult.abstain(f"ADNI table '{tblname}' is not a configured data file")

        try:
            df = self._load(rel, dtype={"RID": str})
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return ToolResult.abstain(f"Could not load ADNI table '{tblname}': {exc}")

        if fldname not in df.columns:
            return ToolResult.abstain(f"Column '{fldname}' not found in ADNI table '{tblname}'")

        sub = df[df["RID"].isin(eligible)].copy()
        try:
            event_rows, diagnosis_basis = self._select_adni_measure_rows(
                sub, event_anchor, filter_spec
            )
        except _AdniAnchorUnavailable as exc:
            return ToolResult.abstain(self._adni_capability_gap_msg(exc, tblname, fldname))

        n_total = int(len(event_rows))
        series = pd.to_numeric(event_rows[fldname], errors="coerce")
        n_missing = int(series.isna().sum())
        values = series.dropna()
        count = int(len(values))

        if count == 0:
            return ToolResult.abstain(
                f"No non-null '{fldname}' values found for eligible cohort"
            )

        s = suppress(count)
        if s is SUPPRESSED:
            return ToolResult.abstain(
                f"count suppressed — {count} is below reporting threshold ({SUPPRESSION_THRESHOLD})"
            )

        # matched_n == n_total (rows found at the aligned visit, before dropping nulls
        # for this specific field); unmatched_n = eligible cohort minus matched. Both
        # suppressed independently (None if <10), same as any participant count
        # leaving the backend (Invariant 4).
        unmatched_n_raw = int(len(eligible)) - n_total
        matched_n = n_total if suppress(n_total) is not SUPPRESSED else None
        unmatched_n = unmatched_n_raw if suppress(unmatched_n_raw) is not SUPPRESSED else None

        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        result = {
            "n_total":    n_total,
            "n_missing":  n_missing,
            "count":      count,
            "mean":       round(float(values.mean()), 3),
            "median":     round(float(values.median()), 3),
            "sd":         round(float(values.std(ddof=1)), 3) if count > 1 else None,
            "q1":         round(q1, 3),
            "q3":         round(q3, 3),
            "iqr":        round(q3 - q1, 3),
            "min_val":    round(float(values.min()), 3),
            "max_val":    round(float(values.max()), 3),
            "proportion": round(float((values > 0).sum() / count), 4) if count > 0 else None,
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
                    f"descriptive_stats({variable_id},{anchor_label}): "
                    f"n={count} mean={result['mean']} median={result['median']} "
                    f"sd={result['sd']} iqr={result['iqr']} "
                    f"(matched_n={matched_n}, unmatched_n={unmatched_n})"
                ),
                source_table=tblname,
                source_field=fldname,
                source_level=_adsp_phc_source_level(tblname, fldname),
                diagnosis_basis=diagnosis_basis,
                baseline_alignment_rule=_ADNI_ALIGNMENT_RULE_BY_ANCHOR.get(event_anchor),
                matched_n=matched_n,
                unmatched_n=unmatched_n,
            )],
        )

    def variant_subject_counts(
        self,
        variable_ids: list[str],
        filter_spec: FilterSpec,
    ) -> ToolResult[dict[str, int | None]]:
        try:
            if filter_spec.id_space == IdSpace.B:
                eligible = self._eligible_rid_set(filter_spec)
                id_col = "RID"
            else:
                eligible = self._eligible_nacc_ids(filter_spec)
                id_col = "NACCID"
        except (_MissingModalityError, ValueError) as exc:
            return ToolResult.abstain(str(exc))

        if not eligible:
            return ToolResult.abstain("No eligible participants for the given filter")

        raw_counts: dict[str, int] = {}
        for vid in variable_ids:
            n = self._distinct_subject_count(vid, eligible, id_col, filter_spec.id_space)
            if n is not None:
                raw_counts[vid] = n

        if not raw_counts:
            return ToolResult.abstain(
                "None of the requested variable_ids were found in local data"
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
                aggregate_ref=f"variant_subject_counts: {raw_counts}",
            )],
        )

    def _distinct_subject_count(
        self,
        variable_id: str,
        eligible: frozenset[str],
        id_col: str,
        id_space: IdSpace,
    ) -> int | None:
        """Distinct-subject count of non-null values for one variable_id, or
        None if the variable/column can't be resolved (never fabricated as 0).
        """
        parts = variable_id.split(".", maxsplit=2)
        source = parts[0] if parts else ""

        if id_space == IdSpace.B:
            if source != "ADNI" or len(parts) < 3:
                return None
            tblname, fldname = parts[1], parts[2]
            rel = self._resolve_adni_table(tblname)
            if not rel:
                return None
            try:
                df = self._load(rel, dtype={id_col: str})
            except Exception:
                return None
        else:
            if source != "NACC" or len(parts) < 2:
                return None
            fldname = parts[1]
            try:
                df = self._nacc_with_col(fldname)
            except Exception:
                return None

        if fldname not in df.columns or id_col not in df.columns:
            return None
        sub = df[df[id_col].isin(eligible)]
        return int(sub.loc[sub[fldname].notna(), id_col].nunique())

    def variable_coverage(self, variable_id: str) -> ToolResult[CoverageStats]:
        try:
            stats = self._compute_coverage(variable_id)
        except FileNotFoundError as exc:
            return ToolResult.abstain(str(exc))
        except Exception as exc:
            logger.warning("variable_coverage error for %s: %s", variable_id, exc)
            return ToolResult.abstain(
                f"variable_coverage failed for '{variable_id}': {exc}"
            )

        if stats is None:
            return ToolResult.abstain(
                f"Variable '{variable_id}' column not found in local data"
            )

        return ToolResult.ok(
            stats,
            evidence=[DataEvidence(
                query_spec={"variable_id": variable_id},
                aggregate_ref=(
                    f"coverage: n_available={stats.n_available}, "
                    f"longitudinal={stats.longitudinal_availability}, "
                    f"missingness={stats.missingness_rate:.4f}"
                ),
            )],
        )

    # ── Coverage computation internals ────────────────────────────────────────

    def _compute_coverage(self, variable_id: str) -> CoverageStats | None:
        parts = variable_id.split(".", maxsplit=2)
        source = parts[0] if parts else ""

        if source == "NACC" and len(parts) == 2:
            return self._cov_nacc(parts[1])
        if source == "ADSP_PHC" and len(parts) == 3:
            return self._cov_adsp_phc(subfolder=parts[1], raw_code=parts[2])
        if source == "ADNI" and len(parts) == 3:
            return self._cov_adni(tblname=parts[1], fldname=parts[2])
        return None

    def _cov_nacc(self, raw_code: str) -> CoverageStats | None:
        try:
            df = self._nacc_with_col(raw_code)
        except (ValueError, KeyError):
            return None

        if raw_code not in df.columns:
            return None
        return _stats_from_df(df, value_col=raw_code, id_col="NACCID")

    def _cov_adsp_phc(self, subfolder: str, raw_code: str) -> CoverageStats | None:
        base = self._config.get("adsp_phc", {}).get("base_dir", "")
        folder = self._root / base / subfolder
        if not folder.is_dir():
            return None

        # Scan data files in the subfolder (skip DD files)
        for fpath in sorted(folder.iterdir()):
            if "_DD_" in fpath.name:
                continue
            if fpath.suffix not in (".csv", ".xlsx", ".xls"):
                continue
            rel = str(fpath.relative_to(self._root))
            try:
                df = self._load(rel)
            except Exception:
                continue
            if raw_code not in df.columns:
                continue
            id_col = "NACCID" if "NACCID" in df.columns else None
            return _stats_from_df(df, value_col=raw_code, id_col=id_col)
        return None

    def _cov_adni(self, tblname: str, fldname: str) -> CoverageStats | None:
        rel = self._resolve_adni_table(tblname)
        if not rel:
            return None
        try:
            df = self._load(rel, dtype={"RID": str})
        except Exception:
            return None
        if fldname not in df.columns:
            return None
        id_col = "RID" if "RID" in df.columns else None
        return _stats_from_df(df, value_col=fldname, id_col=id_col)


def _stats_from_df(
    df: pd.DataFrame,
    value_col: str,
    id_col: str | None,
) -> CoverageStats:
    """Compute CoverageStats for a column in a DataFrame."""
    total = len(df)
    n_avail = int(df[value_col].notna().sum())
    miss = round(1.0 - n_avail / total, 4) if total > 0 else 1.0

    longitudinal = 0
    if id_col and id_col in df.columns:
        by_part = df[df[value_col].notna()].groupby(id_col).size()
        longitudinal = int((by_part >= 2).sum())

    return CoverageStats(
        n_available=n_avail,
        longitudinal_availability=longitudinal,
        missingness_rate=miss,
        last_computed=datetime.now(timezone.utc),
    )
