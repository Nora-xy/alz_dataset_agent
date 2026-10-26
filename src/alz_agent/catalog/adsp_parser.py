"""ADSP-PHC harmonized phenotype catalog parser (P1).

Reads all *_DD_*.xlsx files under ADSP-PHC-122024-investigator.
Every variable is id_space=A (keyed by NACCID).

The xlsx files have a 'Variable' column (occasionally 'Variables') and a 'Values'
column.  Extra unnamed columns in PET/Neuropath files carry overflow value codings
and are concatenated into a single value_coding dict.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from alz_agent.schemas import (
    CatalogEntry,
    IdSpace,
    Modality,
    SourceDataset,
)

logger = logging.getLogger(__name__)

# Columns that are linkage keys — not substantive variables
_SKIP_VARIABLES = frozenset({"NACCID", "NACCVNUM", "SUBJID", "SUBJECT_KEY"})

# Modality folder name → Modality enum
_FOLDER_MODALITY: dict[str, Modality] = {
    "cognition":      Modality.COGNITION,
    "imaging_t1":     Modality.MRI,
    "imaging_flair":  Modality.MRI,
    "imaging_dti":    Modality.MRI,
    "imaging_pet":    Modality.PET,
    "biomarker":      Modality.FLUID_BIOMARKER,
    "neuropath":      Modality.NEUROPATH,
    "vascular_risk":  Modality.VASCULAR,
}


def _folder_to_modality(folder_name: str) -> Modality:
    return _FOLDER_MODALITY.get(folder_name.lower(), Modality.OTHER)


def _parse_values_row(row: pd.Series) -> dict[str, str]:
    """Collect the Values column + any overflow Unnamed: N columns into a coding dict.

    Values entries look like:
      '0 = Does not meet PreciseFilter criteria'
      '1 = Meets PreciseFilter criteria'
    Sometimes multiple values are in the same cell separated by spaces/newlines,
    sometimes they overflow into adjacent unnamed columns.
    """
    coding: dict[str, str] = {}
    # Gather all value cells from this row
    value_cells: list[str] = []
    for col in row.index:
        if col == "Values" or (isinstance(col, str) and col.startswith("Unnamed:")):
            cell = row[col]
            if pd.notna(cell) and str(cell).strip():
                value_cells.append(str(cell).strip())

    for cell in value_cells:
        # Split on common delimiters between multiple code=label pairs
        for part in re.split(r"\s{2,}|\n|;", cell):
            part = part.strip()
            m = re.match(r"^(.+?)\s*=\s*(.+)$", part)
            if m:
                coding[m.group(1).strip()] = m.group(2).strip()
    return coding


def _infer_harmonization_transform(variable: str, description: str) -> str | None:
    desc_lower = description.lower()
    if "harmonized" in desc_lower or "z-score" in desc_lower or "composite" in desc_lower:
        return "ADSP-PHC R3 harmonization (GMM-based)"
    if variable.startswith("PHC_"):
        return "ADSP-PHC R3 harmonization"
    return None


def _read_dd_xlsx(
    xlsx_path: Path,
    modality: Modality,
    release_version: str,
    subfolder_tag: str,
) -> list[CatalogEntry]:
    """Parse a single DD xlsx file and return CatalogEntry list."""
    df = pd.read_excel(xlsx_path)

    # Normalize the variable column name ('Variables' → 'Variable' in DTI)
    if "Variables" in df.columns and "Variable" not in df.columns:
        df = df.rename(columns={"Variables": "Variable"})

    if "Variable" not in df.columns:
        logger.warning("No 'Variable' column in %s — skipping", xlsx_path.name)
        return []

    entries: list[CatalogEntry] = []
    for _, row in df.iterrows():
        raw_code = str(row.get("Variable", "")).strip()
        if not raw_code or raw_code in _SKIP_VARIABLES:
            continue
        # Skip rows that are headers repeated inside the sheet
        if raw_code.lower() in {"variable", "variables"}:
            continue

        description = str(row.get("Description", "")).strip()
        value_coding = _parse_values_row(row)
        harmonization = _infer_harmonization_transform(raw_code, description)

        if modality == Modality.OTHER:
            logger.debug(
                "ADSP-PHC variable %s in %s fell back to 'other'", raw_code, xlsx_path.name
            )

        entries.append(
            CatalogEntry(
                variable_id=f"ADSP_PHC.{subfolder_tag}.{raw_code}",
                concept=raw_code.lower(),
                source_dataset=SourceDataset.ADSP_PHC,
                id_space=IdSpace.A,
                raw_code=raw_code,
                release_version=release_version,
                harmonization_transform=harmonization,
                value_coding=value_coding,
                description=description,
                modality=modality,
                leakage_tags=[],
                coverage=None,
            )
        )

    return entries


def parse_adsp_phc(
    base_dir: Path,
    modality_dirs: dict[str, list[str]],
    release_version: str = "R3_Dec2024",
) -> list[CatalogEntry]:
    """Parse all ADSP-PHC DD xlsx files and return a flat CatalogEntry list.

    Args:
        base_dir:      Root of ADSP-PHC-122024-investigator/.
        modality_dirs: dict mapping modality name → list of subfolder names.
        release_version: Release tag for all entries.

    Returns:
        Flat list of CatalogEntry (coverage=None; leakage_tags=[]).
    """
    all_entries: list[CatalogEntry] = []
    seen_ids: set[str] = set()

    for modality_name, subfolders in modality_dirs.items():
        try:
            modality = Modality(modality_name)
        except ValueError:
            logger.warning("Unknown modality key '%s' in modality_dirs — using OTHER", modality_name)
            modality = Modality.OTHER
        for subfolder in subfolders:
            folder_path = base_dir / subfolder
            if not folder_path.is_dir():
                logger.warning("ADSP-PHC subfolder not found: %s", folder_path)
                continue

            # DD xlsx: *_DD_*.xlsx
            dd_files = sorted(folder_path.glob("*_DD_*.xlsx"))
            if not dd_files:
                logger.warning("No DD xlsx found in %s", folder_path)
                continue

            for xlsx_path in dd_files:
                entries = _read_dd_xlsx(
                    xlsx_path,
                    modality=modality,
                    release_version=release_version,
                    subfolder_tag=subfolder,
                )
                for entry in entries:
                    if entry.variable_id not in seen_ids:
                        seen_ids.add(entry.variable_id)
                        all_entries.append(entry)
                    else:
                        logger.debug("Duplicate variable_id skipped: %s", entry.variable_id)

    logger.info("ADSP-PHC parser: produced %d entries", len(all_entries))
    return all_entries
