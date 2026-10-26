"""Catalog builder (P1/P2): parsers + leakage rules + coverage population."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from alz_agent.catalog.adni_parser import parse_adni, resolve_latest_snapshot
from alz_agent.catalog.adsp_parser import parse_adsp_phc
from alz_agent.catalog.nacc_parser import parse_nacc
from alz_agent.schemas import CatalogEntry, LeakageTag, Modality, SourceDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BuildReport
# ---------------------------------------------------------------------------


@dataclass
class BuildReport:
    total: int
    by_source: dict[str, int] = field(default_factory=dict)
    by_modality: dict[str, int] = field(default_factory=dict)
    by_leakage_tag: dict[str, int] = field(default_factory=dict)
    other_modality_entries: list[str] = field(default_factory=list)
    no_leakage_rule_matched: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Catalog Build Report ===",
            f"Total entries: {self.total}",
            "",
            "By source:",
            *[f"  {k}: {v}" for k, v in sorted(self.by_source.items())],
            "",
            "By modality:",
            *[f"  {k}: {v}" for k, v in sorted(self.by_modality.items())],
            "",
            "By leakage tag:",
            *[f"  {k}: {v}" for k, v in sorted(self.by_leakage_tag.items())],
            "",
            f"'other' modality fallbacks ({len(self.other_modality_entries)}):",
            *[f"  {vid}" for vid in self.other_modality_entries[:50]],
            *(["  … (truncated)"] if len(self.other_modality_entries) > 50 else []),
            "",
            f"No leakage rule matched ({len(self.no_leakage_rule_matched)}):",
            *[f"  {vid}" for vid in self.no_leakage_rule_matched[:50]],
            *(["  … (truncated)"] if len(self.no_leakage_rule_matched) > 50 else []),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Leakage rule loader + applicator
# ---------------------------------------------------------------------------


@dataclass
class _LeakageRule:
    pattern: re.Pattern
    source_filter: set[str] | None  # None = match all sources
    tag: LeakageTag
    reason: str


def load_leakage_rules(rules_path: Path) -> list[_LeakageRule]:
    """Load leakage_rules.yaml and return compiled rule list."""
    with open(rules_path, encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    rules: list[_LeakageRule] = []
    for entry in config.get("rules", []):
        try:
            tag = LeakageTag(entry["tag"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid leakage rule tag in {rules_path}: {exc}") from exc

        source_filter: set[str] | None = None
        if "source_filter" in entry:
            source_filter = set(entry["source_filter"])

        rules.append(
            _LeakageRule(
                pattern=re.compile(entry["variable_pattern"]),
                source_filter=source_filter,
                tag=tag,
                reason=entry.get("reason", ""),
            )
        )
    return rules


def _apply_leakage_tags(
    entry: CatalogEntry,
    rules: list[_LeakageRule],
) -> tuple[CatalogEntry, bool]:
    """Return (updated_entry, rule_matched).

    Applies the first matching rule and sets leakage_tags.
    If no rule matches, sets [LeakageTag.NONE] and returns rule_matched=False.
    """
    source_name = entry.source_dataset.value  # e.g. "NACC", "ADNI", "ADSP_PHC"
    for rule in rules:
        if rule.source_filter and source_name not in rule.source_filter:
            continue
        if rule.pattern.match(entry.raw_code):
            updated = entry.model_copy(update={
                "leakage_tags": [rule.tag],
                "leakage_reasons": [rule.reason],
            })
            return updated, True

    # No rule matched — assign NONE
    updated = entry.model_copy(update={"leakage_tags": [LeakageTag.NONE], "leakage_reasons": []})
    return updated, False


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(paths_yaml: Path) -> dict[str, Any]:
    with open(paths_yaml, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_catalog(
    paths_yaml: Path,
    project_root: Path | None = None,
) -> tuple[list[CatalogEntry], BuildReport]:
    """Build the full catalog from all three sources and return (entries, report).

    Args:
        paths_yaml:    Path to config/paths.yaml (may be absolute or relative).
        project_root:  Directory that all relative paths in paths.yaml are relative to.
                       Defaults to cwd().

    Returns:
        (catalog_entries, build_report)
    """
    if project_root is None:
        project_root = Path.cwd()

    config = _load_config(paths_yaml)

    def _abs(rel: str) -> Path:
        return project_root / rel

    # ── NACC ────────────────────────────────────────────────────────────────
    nacc_cfg = config["nacc"]
    rdd_pdf_path: Path | None = None
    if "rdd_pdf" in nacc_cfg:
        candidate = _abs(nacc_cfg["rdd_pdf"])
        if candidate.exists():
            rdd_pdf_path = candidate
        else:
            logger.warning("rdd_pdf path not found, skipping RDD enrichment: %s", candidate)
    nacc_entries = parse_nacc(
        data_csv=_abs(nacc_cfg["data_csv"]),
        modality_map_path=_abs(nacc_cfg["modality_map"]),
        rdd_pdf=rdd_pdf_path,
    )

    # ── ADNI ────────────────────────────────────────────────────────────────
    adni_cfg = config["adni"]
    adni_base_dir = _abs(adni_cfg["base_dir"])
    adni_data_files: dict[str, Path] = {}
    for table, pattern in config.get("backend", {}).get("adni_data_files", {}).items():
        resolved = resolve_latest_snapshot(adni_base_dir, pattern)
        if resolved is None:
            logger.warning(
                "ADNI table %s: no file matched pattern %r under %s",
                table, pattern, adni_base_dir,
            )
            continue
        adni_data_files[table] = resolved

    datadic_csv = resolve_latest_snapshot(adni_base_dir, adni_cfg["datadic_glob"])
    if datadic_csv is None:
        raise FileNotFoundError(
            f"No ADNI DATADIC found matching {adni_cfg['datadic_glob']!r} under {adni_base_dir}"
        )

    adni_entries = parse_adni(
        datadic_csv=datadic_csv,
        local_tables=adni_cfg["local_tables"],
        data_files=adni_data_files,
    )

    # ── ADSP-PHC ────────────────────────────────────────────────────────────
    adsp_cfg = config["adsp_phc"]
    adsp_entries = parse_adsp_phc(
        base_dir=_abs(adsp_cfg["base_dir"]),
        modality_dirs={k: v for k, v in adsp_cfg["modality_dirs"].items()},
        release_version=adsp_cfg["release_version"],
    )

    all_entries: list[CatalogEntry] = nacc_entries + adni_entries + adsp_entries

    # ── Leakage tags ────────────────────────────────────────────────────────
    rules_path = _abs(config["catalog"]["leakage_rules"])
    rules = load_leakage_rules(rules_path)

    tagged: list[CatalogEntry] = []
    no_rule_matched: list[str] = []
    for entry in all_entries:
        updated, matched = _apply_leakage_tags(entry, rules)
        tagged.append(updated)
        if not matched:
            no_rule_matched.append(entry.variable_id)

    # ── Build report ────────────────────────────────────────────────────────
    by_source: dict[str, int] = defaultdict(int)
    by_modality: dict[str, int] = defaultdict(int)
    by_tag: dict[str, int] = defaultdict(int)
    other_modality: list[str] = []

    for entry in tagged:
        by_source[entry.source_dataset.value] += 1
        by_modality[entry.modality.value] += 1
        for tag in entry.leakage_tags:
            by_tag[tag.value] += 1
        if entry.modality == Modality.OTHER:
            other_modality.append(entry.variable_id)

    report = BuildReport(
        total=len(tagged),
        by_source=dict(by_source),
        by_modality=dict(by_modality),
        by_leakage_tag=dict(by_tag),
        other_modality_entries=other_modality,
        no_leakage_rule_matched=no_rule_matched,
    )

    logger.info(
        "Catalog build complete: %d entries, %d with no leakage rule",
        len(tagged), len(no_rule_matched),
    )
    return tagged, report


# ---------------------------------------------------------------------------
# Coverage population (P2 one-off pass)
# ---------------------------------------------------------------------------


def populate_catalog_coverage(
    catalog: list[CatalogEntry],
    backend: "DataBackend",  # type: ignore[name-defined]  # forward ref
    output_path: Path | None = None,
) -> tuple[list[CatalogEntry], int]:
    """Compute and attach CoverageStats to each CatalogEntry via the backend.

    Args:
        catalog:     List of CatalogEntry (coverage may be None).
        backend:     DataBackend implementation to query for coverage.
        output_path: If given, serialize the updated catalog as JSON here.

    Returns:
        (updated_catalog, n_populated) — n_populated = entries that received coverage.
    """
    from alz_agent.backend.interface import DataBackend  # local import avoids circular

    updated: list[CatalogEntry] = []
    n_populated = 0

    for entry in catalog:
        result = backend.variable_coverage(entry.variable_id)
        if result.grounded and result.value is not None:
            updated.append(entry.model_copy(update={"coverage": result.value}))
            n_populated += 1
        else:
            updated.append(entry)  # keep coverage=None

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [json.loads(e.model_dump_json()) for e in updated]
        output_path.write_text(
            json.dumps(serialized, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Catalog with coverage saved to %s (%d entries)", output_path, len(updated))

    logger.info(
        "populate_catalog_coverage: %d/%d entries populated",
        n_populated, len(catalog),
    )
    return updated, n_populated


# ---------------------------------------------------------------------------
# Retrieval corpus builder (P4 support)
# ---------------------------------------------------------------------------


def build_retrieval_corpus(
    catalog: list[CatalogEntry],
    output_dir: Path,
) -> int:
    """Write per-variable .txt files for the P4 BM25 doc retriever.

    Each file is written to:
        <output_dir>/<source_dataset>/<variable_id_stem>.txt

    The content is a structured snippet:
        VARIABLE_ID: <variable_id>
        LABEL: <concept>
        DESCRIPTION: <description>
        CODES: <value_coding as "code = label" lines>

    Args:
        catalog:    List of CatalogEntry (description field should be populated).
        output_dir: Root of the retrieval_notes/ directory.

    Returns:
        Number of files written (skips entries with no description).
    """
    output_dir = Path(output_dir)
    n_written = 0

    for entry in catalog:
        if not entry.description and not entry.value_coding:
            continue  # nothing useful to index

        source_dir = output_dir / entry.source_dataset.value
        source_dir.mkdir(parents=True, exist_ok=True)

        # Use variable_id as filename but replace dots to avoid path issues
        filename = entry.variable_id.replace(".", "_") + ".txt"
        lines = [
            f"VARIABLE_ID: {entry.variable_id}",
            f"LABEL: {entry.concept}",
        ]
        if entry.description:
            lines.append(f"DESCRIPTION: {entry.description}")
        if entry.value_coding:
            lines.append("CODES:")
            for code, label in sorted(entry.value_coding.items()):
                lines.append(f"  {code} = {label}")

        (source_dir / filename).write_text("\n".join(lines), encoding="utf-8")
        n_written += 1

    logger.info(
        "build_retrieval_corpus: wrote %d files to %s", n_written, output_dir
    )
    return n_written
