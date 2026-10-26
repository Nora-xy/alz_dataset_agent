"""NACC UDS3 catalog parser (P1).

The NACC variable dictionary lives in uds3-rdd.pdf.  This parser reads the CSV
column headers as the authoritative variable list (guaranteed to match what is
actually in the data file) and optionally enriches each entry with human-readable
content from the RDD PDF via nacc_rdd_parser.parse_nacc_rdd().

Without RDD enrichment (rdd_pdf=None or parser import failure):
  concept       = raw_code.lower()   (cryptic, but safe)
  value_coding  = {}
  description   = ""

With RDD enrichment (rdd_pdf provided):
  concept       = RDD short_descriptor (lower-cased, e.g. "age of subject at time of uds visit")
  value_coding  = RDD allowable codes  (e.g. {"1": "Normal cognition", ...})
  description   = RDD derivation text
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import yaml

from alz_agent.schemas import (
    CatalogEntry,
    IdSpace,
    LeakageTag,
    Modality,
    SourceDataset,
)

logger = logging.getLogger(__name__)

# Admin/ID columns that should be skipped — not substantive variables
_SKIP_COLUMNS = frozenset({
    "NACCID", "NACCVNUM", "NACCADC", "PACKET", "FORMVER",
    "VISITMO", "VISITDAY", "VISITYR", "NACCAVST", "NACCNVST",
    "NACCDAYS", "NACCFDYS", "NACCCORE", "NACCREAS", "NACCREFR",
})


def _load_modality_map(map_path: Path) -> list[tuple[re.Pattern, Modality]]:
    """Load nacc_modality_map.yaml → compiled [(regex, Modality)] pairs."""
    with open(map_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    result: list[tuple[re.Pattern, Modality]] = []
    for entry in config.get("modality_patterns", []):
        try:
            modality = Modality(entry["modality"])
        except ValueError:
            raise ValueError(
                f"Unknown modality '{entry['modality']}' in {map_path}"
            )
        result.append((re.compile(entry["pattern"]), modality))
    return result


def _assign_modality(
    raw_code: str, patterns: list[tuple[re.Pattern, Modality]]
) -> Modality:
    for rx, modality in patterns:
        if rx.match(raw_code):
            return modality
    return Modality.OTHER


def parse_nacc(
    data_csv: Path,
    modality_map_path: Path,
    release_version: str = "nacc70",
    rdd_pdf: Path | None = None,
) -> list[CatalogEntry]:
    """Read NACC CSV column headers and return one CatalogEntry per variable.

    Args:
        data_csv:          Path to investigator_nacc70.csv (only the header is read).
        modality_map_path: Path to nacc_modality_map.yaml.
        release_version:   Release tag embedded in every entry.
        rdd_pdf:           Optional path to uds3-rdd.pdf.  When provided, each
                           entry is enriched with the RDD's short descriptor
                           (→ concept), allowable codes (→ value_coding), and
                           derivation text (→ description).

    Returns:
        Flat list of CatalogEntry (coverage=None; leakage_tags=[]; populated by builder).
    """
    modality_patterns = _load_modality_map(modality_map_path)

    # Optionally load RDD enrichment
    rdd_data: dict = {}
    if rdd_pdf is not None:
        try:
            from alz_agent.catalog.nacc_rdd_parser import parse_nacc_rdd
            rdd_data = parse_nacc_rdd(rdd_pdf)
            logger.info(
                "NACC RDD parser: loaded %d variable definitions from %s",
                len(rdd_data), rdd_pdf.name,
            )
        except Exception as exc:
            logger.warning(
                "NACC RDD enrichment skipped (%s: %s). "
                "Falling back to column-name-only catalog.",
                type(exc).__name__, exc,
            )

    with open(data_csv, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        columns = next(reader)

    entries: list[CatalogEntry] = []
    n_enriched = 0

    for raw_code in columns:
        raw_code = raw_code.strip()
        if not raw_code or raw_code in _SKIP_COLUMNS:
            continue

        modality = _assign_modality(raw_code, modality_patterns)
        if modality == Modality.OTHER:
            logger.debug("NACC variable %s fell back to 'other' modality", raw_code)

        rdd_entry = rdd_data.get(raw_code)
        if rdd_entry is not None:
            # RDD short_descriptor → concept (human-readable, lowercased)
            # Fall back to lowercased raw_code when descriptor is empty.
            concept = rdd_entry.short_descriptor.lower() if rdd_entry.short_descriptor else raw_code.lower()
            value_coding: dict = dict(rdd_entry.value_coding)
            description: str = rdd_entry.description
            n_enriched += 1
        else:
            concept = raw_code.lower()
            value_coding = {}
            description = ""

        entries.append(
            CatalogEntry(
                variable_id=f"NACC.{raw_code}",
                concept=concept,
                source_dataset=SourceDataset.NACC,
                id_space=IdSpace.A,
                raw_code=raw_code,
                release_version=release_version,
                harmonization_transform=None,
                value_coding=value_coding,
                description=description,
                modality=modality,
                leakage_tags=[],
                coverage=None,
            )
        )

    logger.info(
        "NACC parser: produced %d entries (%d RDD-enriched) from %s",
        len(entries), n_enriched, data_csv.name,
    )
    return entries
