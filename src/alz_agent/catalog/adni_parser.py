"""ADNI DATADIC catalog parser (P1).

Parses DATADIC_09Jun2026.csv for tables present in ADNI_partial/.
Every variable is id_space=B (keyed by RID/PTID — cannot join to Space A).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from alz_agent.schemas import (
    CatalogEntry,
    IdSpace,
    Modality,
    SourceDataset,
)

logger = logging.getLogger(__name__)

_SNAPSHOT_DATE_RE = re.compile(r"_(\d{2}[A-Za-z]{3}\d{4})\.\w+$")


def resolve_latest_snapshot(base_dir: Path, pattern: str) -> Path | None:
    """Resolve a glob `pattern` (relative to `base_dir`) to its newest match.

    ADNI download filenames carry a trailing `_DDMonYYYY` date token before
    the extension (e.g. `ADAS_15Jul2026.csv`). That token does not sort
    lexicographically in chronological order in general (e.g. `01Jan2027`
    sorts before `15Aug2026` as a string but is chronologically later), so
    matches are ranked by the *parsed* date, falling back to file mtime for
    any match whose name doesn't carry a parseable token.

    Returns None (never raises) if no file matches — callers already have
    an abstain/skip path for "table not configured".
    """
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    def _sort_key(p: Path) -> tuple[float, float]:
        m = _SNAPSHOT_DATE_RE.search(p.name)
        if m:
            try:
                return (datetime.strptime(m.group(1), "%d%b%Y").timestamp(), 0.0)
            except ValueError:
                pass
        return (0.0, p.stat().st_mtime)

    return max(matches, key=_sort_key)


# Table → modality mapping for tables we have locally.
# Tables not listed here default to Modality.OTHER.
_TABLE_MODALITY: dict[str, Modality] = {
    "ADAS":                              Modality.COGNITION,
    "CDR":                               Modality.COGNITION,
    "MMSE":                              Modality.COGNITION,
    "MOCA":                              Modality.COGNITION,
    "DXSUM":                             Modality.OTHER,        # diagnosis-defining
    "FAMHXPAR":                          Modality.DEMOGRAPHIC,
    "FAMHXSIB":                          Modality.DEMOGRAPHIC,
    "ADSP_PHC_COGN":                     Modality.COGNITION,
    "ADSP_PHC_CSF":                      Modality.FLUID_BIOMARKER,
    "ADSP_PHC_PLASMA":                   Modality.FLUID_BIOMARKER,
    "ADSP_PHC_T1_FS":                    Modality.MRI,

    # P0 expansion (2026-07-15 LONI download) — see docs/CURRENT_STATUS.md
    "PTDEMOG":                           Modality.DEMOGRAPHIC,  # core subject characteristics
    "FAQ":                               Modality.COGNITION,
    "NEUROBAT":                          Modality.COGNITION,
    "BLCHANGE":                          Modality.OTHER,        # diagnosis-adjacent, like DXSUM
    "UCSFFSX7":                          Modality.MRI,          # cross-sectional FreeSurfer 7.x
    "MRIMETA":                           Modality.MRI,          # 1.5T scan-occurred metadata
    "MRI3META":                          Modality.MRI,          # 3T scan-occurred metadata
    "AMYMETA":                           Modality.PET,          # amyloid PET scan-occurred metadata
    "AV45META":                          Modality.PET,          # AV-45 PET scan-occurred metadata
    "TAUMETA":                           Modality.PET,          # tau PET scan-occurred metadata
    "UCBERKELEY_AMY_6MM":                Modality.PET,          # amyloid PET SUVR/Centiloid
    "UCBERKELEY_TAU_6MM":                Modality.PET,          # tau PET SUVR
    "UPENN_PLASMA_FUJIREBIO_QUANTERIX":  Modality.FLUID_BIOMARKER,
    "UPENNBIOMK_MASTER":                 Modality.FLUID_BIOMARKER,  # CSF Alzbio3
    "UPENNBIOMK_ROCHE_ELECSYS":          Modality.FLUID_BIOMARKER,  # CSF Roche Elecsys
    # Catalog-searchable but not modality-defining (genotype / registry / lab
    # logistics — no existing Modality value fits; OTHER already holds DXSUM
    # for the same reason).
    "APOERES":                           Modality.OTHER,
    "ARM":                               Modality.OTHER,
    "VISITS":                            Modality.OTHER,
    "ROSTER":                            Modality.OTHER,
    "REGISTRY":                          Modality.OTHER,
    "ALIQUOT_COUNT":                     Modality.OTHER,
    "BIOMARK":                           Modality.OTHER,
    "LABDATA":                           Modality.OTHER,
    "LABTESTS":                          Modality.OTHER,
}

# Columns that are administrative / linkage keys — not substantive variables
_SKIP_FIELDS = frozenset({
    "PTID", "RID", "VISCODE", "VISCODE2", "EXAMDATE", "VISDATE",
    "SITEID", "SUBJECT_KEY", "USERDATE", "USERDATE2", "update_stamp",
})

# Fluid type per biomarker table, used to derive family/variant tags.
_TABLE_FLUID: dict[str, str] = {
    "ADSP_PHC_CSF": "csf",
    "ADSP_PHC_PLASMA": "plasma",
}

# Hand-authored descriptions for columns not present in DATADIC (ADSP_PHC_CSF/PLASMA
# have no DATADIC rows at all). Style/wording matches the Space-A ADSP-PHC Biomarker
# Data Dictionary (ADSP-PHC-122024-investigator/Biomarker/NACC_ADSP_PHC_Biomarker_DD_2024.xlsx),
# which documents the same harmonization pipeline applied here to ADNI-origin subjects,
# with the analyte suffix (181/217) made explicit since Space B distinguishes them.
_KNOWN_FIELD_DESCRIPTIONS: dict[str, str] = {
    "Platform": "Immunoassay platform used for biomarker measurement",
    "Tissue": "Biospecimen tissue/fluid type (e.g. plasma, CSF)",
    "DRAWDATE": "Date of biospecimen collection",
    "AB42_RAW": "Raw AB42 biomarker levels",
    "PHC_AB42": "Harmonized Z-score for AB42 biomarker levels",
    "AB40_RAW": "Raw AB40 biomarker levels",
    "PHC_AB40": "Harmonized Z-score for AB40 biomarker levels",
    "Tau_RAW": "Raw Tau biomarker levels",
    "PHC_Tau": "Harmonized Z-score for Tau biomarker levels",
    "pTau181_RAW": "Raw pTau181 biomarker levels",
    "PHC_pTau181": "Harmonized Z-score for pTau181 biomarker levels",
    "pTau217_RAW": "Raw pTau217 biomarker levels",
    "PHC_pTau217": "Harmonized Z-score for pTau217 biomarker levels",
    "AB42pTau181_Class": "A+/- and T+/- class as identified from CSF biomarker levels (AB42/pTau181 ratio)",
    "PHC_SCeNS_AB42_Score": "SymmetricCentipolar Normalized Score calculated from AB42 Z-score",
    "PHC_SCeNS_pTau181_Score": "SymmetricCentipolar Normalized Score calculated from pTau181 Z-score",
    "PHC_Age_Biomarker": "Age at biomarker measurement",
    "Left_Hippocampus": "Left hippocampal volume (raw FreeSurfer)",
    "Left_Hippocampus_combat": "Left hippocampal volume (ComBat-harmonized FreeSurfer)",
}

# Explicit (family, variant) tags for legacy ADNI-native biomarker assay tables
# whose column-naming conventions don't follow the ADSP-PHC RAW_/PHC_ pattern
# (each assay platform names its analyte columns differently, so a shared
# suffix heuristic doesn't generalize here — a direct lookup is clearer and
# less error-prone than trying to overfit one).
_LEGACY_BIOMARKER_TAGS: dict[str, dict[str, tuple[str, str]]] = {
    "UPENNBIOMK_MASTER": {  # CSF, Innogenetics/Alzbio3 Luminex multiplex assay
        # Both Alzbio3 and Roche Elecsys are documented p-tau181-specific
        # immunoassays (not a generic/undifferentiated "ptau") — tagged
        # "ptau181" so they aggregate with the ADSP-PHC ptau181 variants
        # under describe_variable_family("ptau181", ...), not a separate
        # untargeted "ptau" family that would otherwise hide them.
        "ABETA": ("ab42", "csf_alzbio3"),
        "ABETA_RAW": ("ab42", "csf_alzbio3_raw"),
        "TAU": ("tau", "csf_alzbio3"),
        "TAU_RAW": ("tau", "csf_alzbio3_raw"),
        "PTAU": ("ptau181", "csf_alzbio3"),
        "PTAU_RAW": ("ptau181", "csf_alzbio3_raw"),
    },
    "UPENNBIOMK_ROCHE_ELECSYS": {  # CSF, Roche Elecsys immunoassay
        "ABETA40": ("ab40", "csf_roche_elecsys"),
        "ABETA42": ("ab42", "csf_roche_elecsys"),
        "TAU": ("tau", "csf_roche_elecsys"),
        "PTAU": ("ptau181", "csf_roche_elecsys"),
    },
    "UPENN_PLASMA_FUJIREBIO_QUANTERIX": {  # plasma, dual assay platform
        "pT217_F": ("ptau217", "plasma_fujirebio"),
        "AB42_F": ("ab42", "plasma_fujirebio"),
        "AB40_F": ("ab40", "plasma_fujirebio"),
        "NfL_F": ("nfl", "plasma_fujirebio"),
        "GFAP_F": ("gfap", "plasma_fujirebio"),
        "NfL_Q": ("nfl", "plasma_quanterix"),
        "GFAP_Q": ("gfap", "plasma_quanterix"),
    },
}

# Explicit (family, variant) tags for a hand-picked, clinically-central
# imaging column per table — mirroring the "one representative column per
# analyte" biomarker pattern above, not exhaustive per-region tagging.
_IMAGING_TAGS: dict[str, dict[str, tuple[str, str]]] = {
    "ADSP_PHC_T1_FS": {
        "Left_Hippocampus_combat": ("hippocampal_volume", "freesurfer_t1fs_harmonized"),
        "Left_Hippocampus": ("hippocampal_volume", "freesurfer_t1fs_raw"),
    },
    "UCSFFSX7": {
        "ST29SV": ("hippocampal_volume", "freesurfer7_raw"),  # LeftHippocampus subcortical volume
    },
    "UCBERKELEY_AMY_6MM": {
        "SUMMARY_SUVR": ("amyloid_pet", "berkeley_6mm_raw"),
    },
    "UCBERKELEY_TAU_6MM": {
        "META_TEMPORAL_SUVR": ("tau_pet", "berkeley_6mm_raw"),
    },
}


def _infer_imaging_tags(tblname: str, fldname: str) -> tuple[str | None, str | None]:
    """Return (family, variant) tags for a hand-picked imaging column."""
    return _IMAGING_TAGS.get(tblname, {}).get(fldname, (None, None))


def _infer_biomarker_tags(tblname: str, fldname: str) -> tuple[str | None, str | None]:
    """Return (family, variant) tags for a plain analyte measurement column.

    `family` groups columns measuring "the same thing" (e.g. all ptau217 columns,
    CSF or plasma, raw or harmonized). `variant` distinguishes entries within a
    family by fluid type and assay level. Derived/composite columns (SCeNS scores,
    A/T classification) are intentionally left untagged — they are not a plain
    analyte value alongside the RAW/harmonized pair.
    """
    legacy = _LEGACY_BIOMARKER_TAGS.get(tblname, {}).get(fldname)
    if legacy is not None:
        return legacy

    fluid = _TABLE_FLUID.get(tblname)
    if fluid is None:
        return None, None

    name = fldname.lower()
    if "scens" in name or "class" in name:
        return None, None

    if name.endswith("_raw"):
        level = "raw"
    elif name.startswith("phc_"):
        level = "harmonized"
    else:
        return None, None

    for family in ("ptau217", "ptau181", "ab42", "ab40", "tau"):
        if family in name:
            return family, f"{fluid}_{level}"
    return None, None


def _parse_code_field(code_str: str | float) -> dict[str, str]:
    """Convert ADNI CODE string '1=NL;2=MCI;3=AD' → {1: NL, 2: MCI, 3: AD}."""
    if not isinstance(code_str, str) or not code_str.strip():
        return {}
    coding: dict[str, str] = {}
    for item in re.split(r"[;,]", code_str):
        item = item.strip()
        if "=" in item:
            k, _, v = item.partition("=")
            coding[k.strip()] = v.strip()
    return coding


def _infer_harmonization(tblname: str) -> str | None:
    """Return a harmonization note for ADSP-PHC tables."""
    if tblname.startswith("ADSP_PHC"):
        return "ADSP-PHC R3 harmonization"
    return None


def _parse_from_header(
    tblname: str,
    csv_path: Path,
    release_version: str,
) -> list[CatalogEntry]:
    """Build CatalogEntry rows directly from a CSV header.

    Used for local tables that have zero DATADIC dictionary rows (e.g.
    ADSP_PHC_CSF/ADSP_PHC_PLASMA, whose harmonized column names aren't
    documented under their own TBLNAME in DATADIC).
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    modality = _TABLE_MODALITY.get(tblname, Modality.OTHER)
    entries: list[CatalogEntry] = []
    for fldname in header:
        fldname = str(fldname).strip()
        if not fldname or fldname in _SKIP_FIELDS:
            continue
        family, variant = _infer_biomarker_tags(tblname, fldname)
        if family is None:
            family, variant = _infer_imaging_tags(tblname, fldname)
        entries.append(
            CatalogEntry(
                variable_id=f"ADNI.{tblname}.{fldname}",
                concept=fldname.lower(),
                source_dataset=SourceDataset.ADNI,
                id_space=IdSpace.B,
                raw_code=fldname,
                release_version=release_version,
                harmonization_transform=_infer_harmonization(tblname),
                value_coding={},
                description=_KNOWN_FIELD_DESCRIPTIONS.get(fldname, ""),
                modality=modality,
                leakage_tags=[],
                coverage=None,
                family=family,
                variant=variant,
            )
        )
    return entries


def parse_adni(
    datadic_csv: Path,
    local_tables: list[str],
    release_version: str = "ADNI_Jun2026",
    data_files: dict[str, Path] | None = None,
) -> list[CatalogEntry]:
    """Parse ADNI DATADIC and return CatalogEntry list for local tables only.

    Args:
        datadic_csv:    Path to DATADIC_09Jun2026.csv.
        local_tables:   Table names to include (others are skipped).
        release_version: Release tag for all entries.
        data_files:     table name -> absolute CSV path, used as a header-based
                        fallback for local tables with zero DATADIC rows (e.g.
                        ADSP_PHC_CSF/ADSP_PHC_PLASMA).

    Returns:
        Flat list of CatalogEntry (coverage=None; leakage_tags=[]).
    """
    df = pd.read_csv(datadic_csv, dtype=str)
    # Keep only rows for tables we have locally
    local_set = set(local_tables)
    matched = df[df["TBLNAME"].isin(local_set)].copy()

    entries: list[CatalogEntry] = []
    seen: set[str] = set()

    for _, row in matched.iterrows():
        tblname = str(row.get("TBLNAME", "")).strip()
        fldname = str(row.get("FLDNAME", "")).strip()
        if not fldname or fldname in _SKIP_FIELDS:
            continue

        variable_id = f"ADNI.{tblname}.{fldname}"
        # Deduplicate — DATADIC may have the same variable across multiple PHASE rows
        if variable_id in seen:
            continue
        seen.add(variable_id)

        text = str(row.get("TEXT", "")).strip()
        code_str = row.get("CODE", "")
        modality = _TABLE_MODALITY.get(tblname, Modality.OTHER)
        if modality == Modality.OTHER:
            logger.debug("ADNI variable %s (table %s) fell back to 'other'", fldname, tblname)
        family, variant = _infer_biomarker_tags(tblname, fldname)
        if family is None:
            family, variant = _infer_imaging_tags(tblname, fldname)

        entries.append(
            CatalogEntry(
                variable_id=variable_id,
                concept=fldname.lower(),
                source_dataset=SourceDataset.ADNI,
                id_space=IdSpace.B,
                raw_code=fldname,
                release_version=release_version,
                harmonization_transform=_infer_harmonization(tblname),
                value_coding=_parse_code_field(code_str),
                description=text,
                modality=modality,
                leakage_tags=[],
                coverage=None,
                family=family,
                variant=variant,
            )
        )

    # Header-based fallback for local tables that had zero DATADIC rows.
    tables_with_dd_rows: set[str] = set(matched["TBLNAME"].unique()) if not matched.empty else set()
    for tblname in local_set - tables_with_dd_rows:
        csv_path = (data_files or {}).get(tblname)
        if csv_path is None or not csv_path.exists():
            logger.warning(
                "ADNI parser: table %s has no DATADIC rows and no data_files path — skipped",
                tblname,
            )
            continue
        for entry in _parse_from_header(tblname, csv_path, release_version):
            if entry.variable_id in seen:
                continue
            seen.add(entry.variable_id)
            entries.append(entry)

    if not entries:
        logger.warning("ADNI parser: no entries produced for local_tables=%s", sorted(local_set))

    logger.info(
        "ADNI parser: produced %d entries from %d tables in %s",
        len(entries), len(local_set), datadic_csv.name,
    )
    return entries
