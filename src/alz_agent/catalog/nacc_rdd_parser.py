"""NACC UDS3 Researcher's Data Dictionary (RDD) PDF parser (P1 enrichment).

Public interface:
    parse_nacc_rdd(rdd_pdf: Path) -> dict[str, NaccRddEntry]

Returns a dict keyed by NACC raw_code (e.g. "NACCAGE", "SEX") with the
human-readable label, value coding, and full description text extracted from
the PDF.  Called by nacc_parser.parse_nacc() to enrich CatalogEntries.

The internal _parse_rdd_text() works on a plain string and is fully testable
without pdfplumber or the real PDF.

PDF structure (two formats co-exist in the document):

  Format A — Original UDS question (two-column layout):
    VARIABLE_NAME\n
    DATA SOURCE\n
    Variable type  Original UDS question\n
    Uniform Data Set (UDS)\n
    Short descriptor  <label>\n
    Form, version  v1.2  v2.0  v3.0/3.2\n
    Data type  Numeric longitudinal\n
    IV question no. ...   Allowable codes  <code> = <label>\n
    FV question no. ...   <code> = <label>\n
    ...
    Description/derivation  <text>\n

  Format B — NACC derived variable (single-column layout):
    VARIABLE_NAME\n
    Variable type  NACC derived variable\n
    Short descriptor  <label>\n
    Data type  Numeric longitudinal\n
    Allowable codes  <code> = <label>\n
    <code> = <label>\n
    ...
    Description/derivation  <text>\n

The two-column layout in Format A causes form-version table content (IV/FV/TI/TF
question no., Form version) to be interspersed with the allowable codes section
in the extracted text.  This parser filters those lines out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Public data type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NaccRddEntry:
    raw_code: str
    short_descriptor: str       # RDD "Short descriptor" — human-readable label
    value_coding: dict[str, str]  # {code_str: label_str} from "Allowable codes"
    description: str            # RDD "Description/derivation" text
    data_type: str              # "Numeric longitudinal", "Character", etc.


# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

# Marks the start of a variable block: an ALL-CAPS identifier on its own line
# followed immediately by "DATA SOURCE" (Format A) or "Variable type NACC/Original"
# (Format B).  Requires ≥ 3 characters to exclude form codes like A1, B9, C2T.
_BLOCK_START_RE = re.compile(
    r"\n([A-Z][A-Z0-9_]{2,})\n"
    r"(?=DATA SOURCE\n|Variable type (?:NACC|Original))",
    re.MULTILINE,
)

# Footer lines that mark the physical end of a page in the extracted text.
_FOOTER_RE = re.compile(
    r"NACC UDS Researcher.s Data Dictionary.*$", re.MULTILINE
)

# Form-version table noise interspersed by the two-column PDF layout.
_FORM_VERSION_NOISE_RE = re.compile(
    r"(?:IV|FV|TI|TF)\s+question\s+no\.\s+\S.*?(?=\n|$)"
    r"|DATA SOURCE\s*"
    r"|Uniform Data Set\s*\([^)]+\)\s*"
    r"|Form,\s+version\s+\S.*?(?=\n|$)",
    re.MULTILINE,
)

# Allowable code lines: optional leading whitespace, a numeric (possibly negative,
# possibly with an internal space like "- 4") or short alpha code, then " = " label.
# Also handles ranges written as "N – M" — those are skipped (no "=" sign).
_CODE_LINE_RE = re.compile(
    r"^[-\s]?\d[\d\s]{0,4}=\s*(.+)$"   # numeric code
    r"|^([A-Z]{1,3})\s*=\s*(.+)$",      # short alpha code (rare)
    re.MULTILINE,
)

# More precisely: extract (raw_code_text, label_text) pairs.
_CODE_PAIR_RE = re.compile(
    r"^(-?\s*\d[\d\s]*?)\s*=\s*(.+)$",
    re.MULTILINE,
)

# Trailing form-version junk that sometimes appears at the end of a code label line,
# e.g. "2 = Remote/Inactive TI question no. 2a"
_INLINE_NOISE_RE = re.compile(
    r"\s+(?:IV|FV|TI|TF)\s+question\s+no\.\s+.*$"
)

# "Short descriptor" field: text between "Short descriptor " and the next field key.
_SHORT_DESC_RE = re.compile(
    r"Short descriptor\s+(.+?)(?=\nData type\b|\nAllowable codes\b|\nDescription/derivation\b)",
    re.DOTALL,
)

# "Description/derivation" field: everything after the marker to end of block.
_DESC_RE = re.compile(
    r"Description/derivation[:\s]+(.+?)$",
    re.DOTALL,
)

# "Data type" field.
_DATA_TYPE_RE = re.compile(r"Data type\s+(\S[^\n]*)")

# Words to strip from the short descriptor when cleaning multi-line noise.
_DESC_NOISE_LINE_RE = re.compile(
    r"^(?:Data type\b|Allowable codes\b|Variable type\b|Form,\s+version\b"
    r"|Uniform Data Set\b|DATA SOURCE\b|(?:IV|FV|TI|TF)\s+question)",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def parse_nacc_rdd(rdd_pdf: Path) -> dict[str, NaccRddEntry]:
    """Parse uds3-rdd.pdf and return {raw_code: NaccRddEntry}.

    Args:
        rdd_pdf: Path to the NACC UDS3 Researcher's Data Dictionary PDF.

    Returns:
        Dict mapping NACC column names (e.g. "NACCAGE") to their RDD entries.
        Empty dict if pdfplumber is not installed or the PDF cannot be opened.

    Raises:
        FileNotFoundError: if rdd_pdf does not exist.
    """
    if not rdd_pdf.exists():
        raise FileNotFoundError(f"RDD PDF not found: {rdd_pdf}")

    try:
        import pdfplumber  # optional dependency
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for RDD parsing: pip install pdfplumber"
        ) from exc

    full_text = ""
    with pdfplumber.open(rdd_pdf) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"

    return _parse_rdd_text(full_text)


# ---------------------------------------------------------------------------
# Internal: text-based parser (no pdfplumber dependency — testable standalone)
# ---------------------------------------------------------------------------


def _parse_rdd_text(full_text: str) -> dict[str, NaccRddEntry]:
    """Parse raw extracted PDF text and return {raw_code: NaccRddEntry}.

    This is the core parsing logic, separated from PDF I/O so it can be
    unit-tested with synthetic text fixtures.
    """
    # Strip page footers so they don't bleed into descriptions.
    clean = _FOOTER_RE.sub("", full_text)

    # Split into variable blocks.  The regex captures the variable name and
    # uses a lookahead so the "DATA SOURCE" / "Variable type" line stays in the
    # block text.
    parts = _BLOCK_START_RE.split(clean)
    # parts layout after split: [preamble, varname1, block1_text, varname2, block2_text, ...]
    # parts[0] is preamble (table-of-contents / intro), skip it.

    result: dict[str, NaccRddEntry] = {}
    i = 1
    while i + 1 < len(parts):
        raw_code = parts[i].strip()
        block = parts[i + 1]
        i += 2
        entry = _parse_block(raw_code, block)
        if entry is not None:
            result[raw_code] = entry

    return result


def _parse_block(raw_code: str, block: str) -> NaccRddEntry | None:
    """Parse one variable block and return an NaccRddEntry, or None on failure."""
    # Remove form-version table noise to get a cleaner block.
    cleaned = _FORM_VERSION_NOISE_RE.sub("", block)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # collapse excessive blank lines

    # ── Short descriptor ──────────────────────────────────────────────────
    short_desc = _extract_short_descriptor(cleaned)

    # ── Data type ─────────────────────────────────────────────────────────
    m_dt = _DATA_TYPE_RE.search(cleaned)
    data_type = m_dt.group(1).strip() if m_dt else ""

    # ── Allowable codes ───────────────────────────────────────────────────
    value_coding = _extract_value_coding(block)  # use raw block — codes may have noise stripped

    # ── Description/derivation ────────────────────────────────────────────
    description = _extract_description(cleaned)

    return NaccRddEntry(
        raw_code=raw_code,
        short_descriptor=short_desc,
        value_coding=value_coding,
        description=description,
        data_type=data_type,
    )


def _extract_short_descriptor(cleaned_block: str) -> str:
    """Extract and clean the Short descriptor field."""
    m = _SHORT_DESC_RE.search(cleaned_block)
    if not m:
        return ""
    raw = m.group(1)
    # Remove any remaining noise lines (form-version artefacts not stripped earlier)
    lines = [ln for ln in raw.splitlines() if not _DESC_NOISE_LINE_RE.match(ln.strip())]
    desc = " ".join(ln.strip() for ln in lines if ln.strip())
    # Normalize internal whitespace
    desc = re.sub(r"\s+", " ", desc).strip()
    return desc


def _extract_value_coding(raw_block: str) -> dict[str, str]:
    """Extract {code: label} pairs from the Allowable codes section.

    Works on the raw (un-noise-stripped) block because some code labels are
    split across lines with noise interspersed; we filter noise line by line.
    """
    # Isolate the Allowable codes section.
    ac_match = re.search(
        r"Allowable codes\s+(.+?)(?=\nDescription/derivation|\Z)",
        raw_block,
        re.DOTALL,
    )
    if not ac_match:
        return {}

    section = ac_match.group(1)
    coding: dict[str, str] = {}

    for line in section.splitlines():
        # Skip form-version table lines
        if re.match(r"\s*(?:IV|FV|TI|TF)\s+question\s+no\.", line):
            continue
        if re.match(r"\s*(?:DATA SOURCE|Uniform Data Set|Form,\s+version)", line, re.IGNORECASE):
            continue

        # Try to match a code = label pair
        m = _CODE_PAIR_RE.match(line.strip())
        if not m:
            continue

        code_raw = m.group(1).strip().replace(" ", "")  # normalise "- 4" → "-4"
        label_raw = m.group(2).strip()
        # Strip trailing inline form-version noise (e.g. "TI question no. 2a")
        label = _INLINE_NOISE_RE.sub("", label_raw).strip()
        if code_raw and label:
            coding[code_raw] = label

    return coding


def _extract_description(cleaned_block: str) -> str:
    """Extract the Description/derivation field text."""
    m = _DESC_RE.search(cleaned_block)
    if not m:
        return ""
    raw = m.group(1).strip()
    # Remove page footers that leaked through
    raw = _FOOTER_RE.sub("", raw).strip()
    # Collapse whitespace runs
    return re.sub(r"\s+", " ", raw).strip()
