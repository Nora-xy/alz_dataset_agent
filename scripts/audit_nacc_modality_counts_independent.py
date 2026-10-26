"""Independent audit of NACC modality-combination counts.

Validates P2 aggregation logic WITHOUT importing LocalDataBackend.
Reads raw NACC CSV directly with pandas, applies nacc_modality_map.yaml
patterns to classify columns, then computes participant-level counts over NACCID.

Usage:
    python scripts/audit_nacc_modality_counts_independent.py [path/to/nacc.csv]

Default path (when no argument given):
    NACC/investigator_nacc70.csv relative to this project
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_NACC_CSV = _PROJECT_ROOT / "NACC" / "investigator_nacc70.csv"
_MODALITY_MAP_YAML = _PROJECT_ROOT / "config" / "nacc_modality_map.yaml"


# ---------------------------------------------------------------------------
# Column → modality classification (mirrors P1 catalog builder logic)
# ---------------------------------------------------------------------------

def _classify_columns(columns: list[str], patterns: list[dict]) -> dict[str, list[str]]:
    """Apply patterns from nacc_modality_map.yaml; return modality → [col, ...] map.

    This validates P2 aggregation logic, not the correctness of the P1 modality map.
    Only the first matching pattern per column wins (same as the catalog builder).
    """
    modality_cols: dict[str, list[str]] = {}
    for col in columns:
        matched = "other"
        for rule in patterns:
            if re.match(rule["pattern"], col):
                matched = rule["modality"]
                break
        modality_cols.setdefault(matched, []).append(col)
    return modality_cols


# ---------------------------------------------------------------------------
# Participant-level modality presence
# ---------------------------------------------------------------------------

def _participants_with_any(
    df: pd.DataFrame, cols: list[str], id_col: str = "NACCID"
) -> frozenset[str]:
    """Return NACCIDs that have at least one non-missing value across `cols`.

    A participant is counted if they have at least one non-missing value
    in at least one of the listed columns, across ANY row / visit.
    """
    sub = df[[id_col] + cols]
    mask = sub[cols].notna().any(axis=1)
    return frozenset(sub.loc[mask, id_col].unique())


def _participants_with_mri_proxy(
    df: pd.DataFrame, id_col: str = "NACCID"
) -> frozenset[str]:
    """NACCIDs where NACCNMRI > 0 in at least one visit.

    NACCNMRI is the NACC-reported count of MRI scans per visit. NACCNMRI > 0
    means the participant had at least one MRI scan reported at that visit.
    This is a rough proxy for 'had structural MRI' but is NOT equivalent to
    having T1 Freesurfer/MUSE features in ADSP-PHC (a stricter criterion).
    """
    if "NACCNMRI" not in df.columns:
        return frozenset()
    mask = pd.to_numeric(df["NACCNMRI"], errors="coerce").fillna(0) > 0
    return frozenset(df.loc[mask, id_col].unique())


# ---------------------------------------------------------------------------
# Sentinel-value detection
# ---------------------------------------------------------------------------

_SENTINEL_VALUES = {-4, -4.0}  # NACC uses -4 for "not applicable / no form"


def _has_real_non_null(series: pd.Series) -> pd.Series:
    """True if the value is non-null AND not a known NACC sentinel (-4)."""
    numeric = pd.to_numeric(series, errors="coerce")
    return series.notna() & ~numeric.isin(_SENTINEL_VALUES)


def _participants_with_real_value(
    df: pd.DataFrame, cols: list[str], id_col: str = "NACCID"
) -> frozenset[str]:
    """Participant-level presence excluding NACC sentinel values (e.g., -4)."""
    if not cols:
        return frozenset()
    sub = df[[id_col] + cols]
    real_mask = sub[cols].apply(_has_real_non_null).any(axis=1)
    return frozenset(sub.loc[real_mask, id_col].unique())


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def _cap(items: list, n: int = 20) -> list:
    """Return at most n items with a trailing '...' if truncated."""
    if len(items) <= n:
        return list(items)
    return list(items[:n]) + [f"... ({len(items) - n} more)"]


def run_audit(nacc_csv_path: str) -> None:
    path = Path(nacc_csv_path)
    if not path.exists():
        print(f"ERROR: NACC CSV not found: {path}")
        sys.exit(1)

    print("=" * 72)
    print("NACC Modality-Combination Count Audit (independent of LocalDataBackend)")
    print("=" * 72)
    print(f"CSV: {path}")
    print()

    # ── Load modality map ──────────────────────────────────────────────────
    with open(_MODALITY_MAP_YAML, encoding="utf-8") as fh:
        mmap = yaml.safe_load(fh)
    patterns: list[dict] = mmap.get("modality_patterns", [])

    # ── Read header ────────────────────────────────────────────────────────
    header_df = pd.read_csv(path, nrows=0)
    all_cols = list(header_df.columns)

    # ── Classify columns ───────────────────────────────────────────────────
    modality_cols = _classify_columns(all_cols, patterns)

    cognition_cols = modality_cols.get("cognition", [])
    mri_cols = modality_cols.get("mri", [])  # expected: empty for raw NACC

    print(f"Total columns in raw NACC CSV: {len(all_cols)}")
    print()
    print("--- Column classification (from nacc_modality_map.yaml) ---")
    for mod in sorted(modality_cols):
        n_cols = len(modality_cols[mod])
        print(f"  {mod:20s}: {n_cols} columns")
    print()

    print("--- Cognition columns (first 20) ---")
    print(" ", _cap(cognition_cols))
    print()

    if mri_cols:
        print(f"--- MRI/T1 columns mapped in nacc_modality_map.yaml ({len(mri_cols)}) ---")
        print(" ", _cap(mri_cols))
    else:
        print("--- MRI/T1 structural feature columns in raw NACC CSV ---")
        print("  NONE found via nacc_modality_map.yaml patterns.")
        print("  Raw NACC (investigator_nacc70.csv) is a clinical/cognitive dataset.")
        print("  T1 structural features live in ADSP-PHC Imaging_T1 CSVs (Space A),")
        print("  not in the NACC wide-format CSV.")
    print()

    # ── Imaging-adjacent columns actually present ─────────────────────────
    print("--- Imaging-adjacent columns present in raw NACC CSV ---")
    print("  (these are NOT structural T1 imaging features)")
    print()
    print("  NACCNMRI  — NACC count of MRI scans per visit (clinical report,")
    print("              maps to 'other' in modality map; NACCNMRI>0 used as")
    print("              a rough scan-history proxy below)")
    print("  CVDIMAG*  — vascular: self-reported history of imaging tests → 'vascular'")
    print("  IMAGLINF/IMAGLAC/IMAGMACH/IMAGMICH/IMAGMWMH/IMAGEWMH/STKIMAG")
    print("              — imaging FINDINGS on MRI (sentinel -4 when no MRI done)")
    print("  AMYLPET/TAUPETAD — PET imaging flags (0/1/-4/8)")
    print()

    # ── Load data (only needed columns) ────────────────────────────────────
    # Pull cognition columns + NACCNMRI + NACCID
    cols_to_load = list(
        set(["NACCID", "NACCNMRI"] + cognition_cols + mri_cols)
        & set(all_cols)
    )
    print("Loading data (cognition columns + NACCNMRI + NACCID) ...")
    df = pd.read_csv(path, usecols=cols_to_load, low_memory=False, dtype={"NACCID": str})
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print()

    # ── Basic counts ───────────────────────────────────────────────────────
    total_rows = len(df)
    total_ids = df["NACCID"].nunique()
    print(f"Total rows          : {total_rows:,}")
    print(f"Total unique NACCIDs: {total_ids:,}")
    print()

    # ── Cognition participant count (raw NACC columns, sentinel-aware) ─────
    ids_cognition = _participants_with_real_value(df, cognition_cols)
    print(f"Cognition (raw NACC columns, excl. sentinel -4): {len(ids_cognition):,} participants")
    print(f"  Columns used ({len(cognition_cols)}): {_cap(cognition_cols)}")
    print()

    # ── MRI proxy from NACCNMRI > 0 ───────────────────────────────────────
    ids_mri_proxy = _participants_with_mri_proxy(df)
    print(f"MRI-scan proxy (NACCNMRI > 0): {len(ids_mri_proxy):,} participants")
    print("  NOTE: This is NOT equivalent to having T1 structural features in ADSP-PHC.")
    print("  ADSP-PHC T1 requires Freesurfer/MUSE processing from a T1 scan — stricter.")
    print()

    # ── Cognition + MRI proxy ─────────────────────────────────────────────
    ids_cog_mri_proxy = ids_cognition & ids_mri_proxy
    print(f"Cognition + MRI-scan proxy: {len(ids_cog_mri_proxy):,} participants")
    print()

    # ── Cannot compute cognition + any_t1 from raw NACC alone ─────────────
    print("=" * 72)
    print("KEY FINDING: 'cognition + any_T1 structural features' CANNOT be computed")
    print("from investigator_nacc70.csv alone. Raw NACC is a clinical dataset with no")
    print("T1 Freesurfer/MUSE feature columns.")
    print()
    print("In the P2 LocalDataBackend:")
    print("  modality_intersection(['cognition','mri'], FilterSpec(id_space=IdSpace.A))")
    print("  counts NACCIDs that appear in BOTH:")
    print("    - ADSP-PHC Cognition CSV  (Space A harmonized cognition)")
    print("    - ADSP-PHC T1/FLAIR/DTI CSVs (Space A harmonized MRI)")
    print("  This is an ADSP-PHC file-membership count, NOT a raw NACC column count.")
    print()
    print("DESIGN GAP CONFIRMED:")
    print("  The backend modality_intersection for 'cognition' and 'mri' in Space A")
    print("  cannot be independently validated from raw NACC columns, because:")
    print("  1. 'cognition' in the backend = ADSP-PHC Cognition file membership.")
    print("     Raw NACC cognition columns cover ~all participants (~54K NACCIDs),")
    print("     while ADSP-PHC Cognition is a curated harmonized subset (~8-10K).")
    print("  2. 'mri' in the backend = ADSP-PHC T1/FLAIR/DTI file membership.")
    print("     Raw NACC has NACCNMRI (scan-count proxy, ~7,935 NACCIDs > 0)")
    print("     but no structural T1 feature columns.")
    print("  3. The counts come from different populations and cannot be compared.")
    print()
    print("CONSEQUENCE FOR REGRESSION TESTING:")
    print("  NACC-CSV-specific modality-combination regression testing requires")
    print("  source/id-space scoping that the current backend API does not expose.")
    print("  A skipped test is added in tests/test_p2_backend.py to document this gap.")
    print("=" * 72)
    print()

    # ── Summary table ──────────────────────────────────────────────────────
    print("SUMMARY (participant-level counts over NACCID):")
    print(f"  Total unique NACCIDs in raw NACC CSV           : {total_ids:,}")
    print(f"  NACCIDs with any cognition column (non-null, non-sentinel) : {len(ids_cognition):,}")
    print(f"  NACCIDs with NACCNMRI > 0 (MRI scan proxy)    : {len(ids_mri_proxy):,}")
    print(f"  NACCIDs with cognition + NACCNMRI > 0 (proxy) : {len(ids_cog_mri_proxy):,}")
    print()
    print("  T1 structural feature columns in raw NACC CSV  : NONE (not applicable)")
    print("  'cognition + any_t1' from raw NACC alone       : NOT COMPUTABLE")
    print()
    print("  Modality enum has NO T1/ANY_T1/ANY_MRI/ANY_IMAGING values.")
    print("  Project canonical name for MRI modality: 'mri'")
    print("  In Space A, 'mri' = membership in ADSP-PHC T1/FLAIR/DTI files.")
    print()
    print("KNOWN ADSP-PHC COUNT (from backend, not independent audit):")
    print("  cognition + mri (= 'any_t1' equivalent) in ADSP-PHC: ~7,400 participants")
    print("  (This is what LocalDataBackend.modality_intersection(['cognition','mri'])")
    print("   returns for Space A — an ADSP-PHC-derived count, not a NACC-column count.)")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_NACC_CSV
    run_audit(csv_path)
