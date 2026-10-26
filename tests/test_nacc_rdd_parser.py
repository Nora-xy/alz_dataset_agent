"""Tests for nacc_rdd_parser._parse_rdd_text() — no PDF or pdfplumber required."""

from __future__ import annotations

import textwrap

import pytest

from alz_agent.catalog.nacc_rdd_parser import NaccRddEntry, _parse_rdd_text


# ---------------------------------------------------------------------------
# Synthetic RDD text fixtures (mimic real PDF extracted text)
# ---------------------------------------------------------------------------

# Format B: NACC derived variable (compact, single-column)
_FORMAT_B_TEXT = textwrap.dedent("""\

NACCCCBS
Variable type NACC derived variable
Short descriptor Reported current use of a calcium channel blocking agent
Data type Numeric longitudinal
Allowable codes 0 = Did not report use at visit
1 = Reported use at visit
-4 = Did not complete medications form
Description/derivation This variable indicates reported current use of a calcium channel blocking medication.

NACCADEP
Variable type NACC derived variable
Short descriptor Reported current use of an antidepressant
Data type Numeric longitudinal
Allowable codes 0 = Did not report use at visit
1 = Reported use at visit
-4 = Did not complete medications form
Description/derivation Antidepressant medication flag.
""")

# Format A: Original UDS question (two-column layout, with noise interspersed)
_FORMAT_A_TEXT = textwrap.dedent("""\

SEX
DATA SOURCE
Variable type Original UDS question
Uniform Data Set (UDS)
Short descriptor Sex
Form, version v1.2 v2.0 v3.0/3.2
Data type Numeric cross-sectional
IV question no. 4 4 3
Allowable codes 1 = Male
FV question no. Not applic Not applic Not applic
2 = Female
TI question no. 3
TF question no. Not applic Not applic
Description/derivation Sex of the subject.

RACE
DATA SOURCE
Variable type Original UDS question
Uniform Data Set (UDS)
Short descriptor Race
Form, version v1.2 v2.0 v3.0/3.2
Data type Numeric cross-sectional
IV question no. 10 10 9
Allowable codes 1 = White
FV question no. Not applic Not applic Not applic
2 = Black or African American
TI question no. 9
3 = American Indian or Alaska Native
4 = Native Hawaiian or Other Pacific TF question no. Not applic Not applic
Islander
5 = Asian
50 = Other (specify)
99 = Unknown
Description/derivation Note that although this variable is not collected at follow-up visits, the value from
the initial visit will be shown at all follow-up visits.
""")

# Mixed: one of each format in same text
_MIXED_TEXT = _FORMAT_B_TEXT + _FORMAT_A_TEXT

# Entry with multi-line short descriptor (form version table mixed in)
_MULTILINE_DESC_TEXT = textwrap.dedent("""\

COGATTN
DATA SOURCE
Variable type Original UDS question
Uniform Data Set (UDS)
Short descriptor Indicate whether the subject currently
Form, version v1.2 v2.0 v3.0/3.2
is meaningfully impaired, relative to
previously attained abilities, in attention or
concentration FV question no. 4e 4e 4f
Data type Numeric longitudinal TI question no. 4f
Allowable codes 0 = No
1 = Yes
9 = Unknown
Description/derivation Attention/concentration impairment indicator.
""")

# Entry with negative code using space ("- 4")
_NEGATIVE_CODE_TEXT = textwrap.dedent("""\

CVHATT
DATA SOURCE
Variable type Original UDS question
Uniform Data Set (UDS)
Short descriptor Heart attack/cardiac arrest
Form, version v1.2 v2.0 v3.0/3.2
Data type Numeric longitudinal
IV question no. 1a 1a 2a
Allowable codes 0 = Absent
FV question no. 1a 1a Not applic
1 = Recent/Active
2 = Remote/Inactive TI question no. 2a
9 = Unknown TF question no. 1a Not applic
- 4 = Not available: UDS form submitted
Description/derivation Heart attack history.
""")


# ---------------------------------------------------------------------------
# Tests: Format B (NACC derived variable)
# ---------------------------------------------------------------------------


class TestFormatB:
    def setup_method(self):
        self.entries = _parse_rdd_text(_FORMAT_B_TEXT)

    def test_finds_naccccbs(self):
        assert "NACCCCBS" in self.entries

    def test_finds_naccadep(self):
        assert "NACCADEP" in self.entries

    def test_short_descriptor(self):
        e = self.entries["NACCCCBS"]
        assert "calcium channel blocking" in e.short_descriptor.lower()

    def test_data_type(self):
        e = self.entries["NACCCCBS"]
        assert "longitudinal" in e.data_type.lower()

    def test_value_coding_present(self):
        e = self.entries["NACCCCBS"]
        assert "0" in e.value_coding
        assert "1" in e.value_coding

    def test_value_coding_label_correct(self):
        e = self.entries["NACCCCBS"]
        assert "did not report" in e.value_coding["0"].lower()
        assert "reported use" in e.value_coding["1"].lower()

    def test_negative_code_present(self):
        e = self.entries["NACCCCBS"]
        assert "-4" in e.value_coding

    def test_description_populated(self):
        e = self.entries["NACCCCBS"]
        assert "calcium channel blocking" in e.description.lower()

    def test_raw_code_preserved(self):
        e = self.entries["NACCCCBS"]
        assert e.raw_code == "NACCCCBS"


# ---------------------------------------------------------------------------
# Tests: Format A (Original UDS question with noise)
# ---------------------------------------------------------------------------


class TestFormatA:
    def setup_method(self):
        self.entries = _parse_rdd_text(_FORMAT_A_TEXT)

    def test_finds_sex(self):
        assert "SEX" in self.entries

    def test_finds_race(self):
        assert "RACE" in self.entries

    def test_sex_short_descriptor(self):
        e = self.entries["SEX"]
        assert e.short_descriptor.lower().strip() == "sex"

    def test_sex_value_coding_male_female(self):
        e = self.entries["SEX"]
        assert e.value_coding.get("1", "").lower() == "male"
        assert e.value_coding.get("2", "").lower() == "female"

    def test_race_value_coding_white(self):
        e = self.entries["RACE"]
        assert "white" in e.value_coding.get("1", "").lower()

    def test_race_value_coding_black(self):
        e = self.entries["RACE"]
        assert "black" in e.value_coding.get("2", "").lower()

    def test_race_value_coding_other(self):
        e = self.entries["RACE"]
        assert "50" in e.value_coding

    def test_noise_lines_not_in_coding(self):
        # Form version table entries should not appear as code keys
        e = self.entries["SEX"]
        for key in e.value_coding:
            assert "question" not in key.lower()
            assert "not applic" not in key.lower()

    def test_sex_description_populated(self):
        e = self.entries["SEX"]
        assert len(e.description) > 0

    def test_race_data_type(self):
        e = self.entries["RACE"]
        assert "cross-sectional" in e.data_type.lower()


# ---------------------------------------------------------------------------
# Tests: Mixed format
# ---------------------------------------------------------------------------


class TestMixed:
    def setup_method(self):
        self.entries = _parse_rdd_text(_MIXED_TEXT)

    def test_all_four_variables_found(self):
        assert {"NACCCCBS", "NACCADEP", "SEX", "RACE"} <= set(self.entries.keys())

    def test_total_count(self):
        assert len(self.entries) == 4


# ---------------------------------------------------------------------------
# Tests: Multi-line short descriptor
# ---------------------------------------------------------------------------


class TestMultiLineDescriptor:
    def setup_method(self):
        self.entries = _parse_rdd_text(_MULTILINE_DESC_TEXT)

    def test_finds_cogattn(self):
        assert "COGATTN" in self.entries

    def test_descriptor_contains_key_words(self):
        e = self.entries["COGATTN"]
        desc = e.short_descriptor.lower()
        assert "subject" in desc or "impaired" in desc or "attention" in desc

    def test_descriptor_not_empty(self):
        e = self.entries["COGATTN"]
        assert len(e.short_descriptor) > 0

    def test_value_coding_no_unknown(self):
        # "9 = Unknown" should be present
        e = self.entries["COGATTN"]
        assert "9" in e.value_coding


# ---------------------------------------------------------------------------
# Tests: Negative code with space ("- 4")
# ---------------------------------------------------------------------------


class TestNegativeCode:
    def setup_method(self):
        self.entries = _parse_rdd_text(_NEGATIVE_CODE_TEXT)

    def test_finds_cvhatt(self):
        assert "CVHATT" in self.entries

    def test_negative_code_normalised(self):
        e = self.entries["CVHATT"]
        # "- 4" should be normalised to "-4"
        assert "-4" in e.value_coding

    def test_inline_noise_stripped_from_label(self):
        # "2 = Remote/Inactive TI question no. 2a" — label should be clean
        e = self.entries["CVHATT"]
        label = e.value_coding.get("2", "")
        assert "question" not in label.lower()
        assert "remote" in label.lower() or "inactive" in label.lower()

    def test_cvhatt_descriptor(self):
        e = self.entries["CVHATT"]
        assert "heart" in e.short_descriptor.lower() or "cardiac" in e.short_descriptor.lower()


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_text_returns_empty_dict(self):
        result = _parse_rdd_text("")
        assert result == {}

    def test_preamble_only_returns_empty_dict(self):
        result = _parse_rdd_text("Some introduction text\nwith no variable blocks.\n")
        assert result == {}

    def test_returns_nacc_rdd_entry_type(self):
        entries = _parse_rdd_text(_FORMAT_B_TEXT)
        assert isinstance(entries["NACCCCBS"], NaccRddEntry)

    def test_value_coding_is_dict(self):
        entries = _parse_rdd_text(_FORMAT_B_TEXT)
        assert isinstance(entries["NACCCCBS"].value_coding, dict)

    def test_no_form_code_false_positives(self):
        # A1, B9, C2T etc. should not appear as variable entries
        text = "\nA1\nSEX\nDATA SOURCE\nVariable type Original UDS question\nShort descriptor Sex\nData type Numeric\nAllowable codes 1 = Male\n2 = Female\nDescription/derivation Sex of the subject.\n"
        entries = _parse_rdd_text(text)
        assert "A1" not in entries
        assert "SEX" in entries
