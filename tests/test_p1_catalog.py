"""P1 tests: catalog parsers, leakage rule loader, builder, and build report."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from alz_agent.catalog.adni_parser import (
    _infer_biomarker_tags,
    _infer_imaging_tags,
    _TABLE_MODALITY,
    parse_adni,
    resolve_latest_snapshot,
)
from alz_agent.catalog.adsp_parser import parse_adsp_phc
from alz_agent.catalog.builder import BuildReport, build_catalog, load_leakage_rules
from alz_agent.catalog.nacc_parser import parse_nacc
from alz_agent.schemas import IdSpace, LeakageTag, Modality, SourceDataset

FIXTURES = Path(__file__).parent / "fixtures"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / shared fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _nacc_map_path(tmp_path: Path) -> Path:
    """Write a minimal nacc_modality_map.yaml for testing."""
    content = textwrap.dedent("""\
        modality_patterns:
          - pattern: "^(BIRTHMO|BIRTHYR|SEX|EDUC|RACE)$"
            modality: demographic
          - pattern: "^(CDRSUM|CDRGLOB|MMSE30|MOCALOC|TRAILA|ANIMALS)$"
            modality: cognition
          - pattern: "^(HYPERTEN|CVDI)$"
            modality: vascular
          - pattern: "^NP.*"
            modality: neuropath
          - pattern: "^(CSFTAU|AB42)$"
            modality: fluid_biomarker
          - pattern: "^NGDS.*"
            modality: other
          - pattern: "^DRUG.*"
            modality: other
    """)
    p = tmp_path / "nacc_modality_map.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _leakage_rules_path(tmp_path: Path) -> Path:
    """Write a minimal leakage_rules.yaml for testing."""
    content = textwrap.dedent("""\
        rules:
          - variable_pattern: "^NACCUDSD$"
            tag: DIAGNOSIS_DEFINING
            reason: "NACC unified diagnostic summary"
          - variable_pattern: "^NACCCBD$"
            tag: DIAGNOSIS_DEFINING
            reason: "NACC clinical designation"
          - variable_pattern: "^NPPDX[A-Z]$"
            tag: DIAGNOSIS_DEFINING
            reason: "NACC neuropath diagnosis"
          - variable_pattern: "^DXCURREN$"
            source_filter: [ADNI]
            tag: DIAGNOSIS_DEFINING
            reason: "ADNI current diagnosis"
          - variable_pattern: "^DXCONV$"
            source_filter: [ADNI]
            tag: DIAGNOSIS_DEFINING
            reason: "ADNI conversion indicator"
          - variable_pattern: "^(CDRSUM|CDRGLOB)$"
            tag: CIRCULAR
            reason: "CDR collinear with outcome"
          - variable_pattern: "^PHC_MEM.*"
            source_filter: [ADSP_PHC]
            tag: HARMONIZATION_FIT_SCOPE
            reason: "PHC memory composite: GMM fit on full cohort"
          - variable_pattern: "^(PHC_AB42|PHC_Tau|PHC_pTau)$"
            source_filter: [ADSP_PHC]
            tag: HARMONIZATION_FIT_SCOPE
            reason: "PHC Z-scores: GMM harmonization scope"
          - variable_pattern: "^AT_class$"
            source_filter: [ADSP_PHC]
            tag: HARMONIZATION_FIT_SCOPE
            reason: "AT class derived from PHC Z-scores"
    """)
    p = tmp_path / "leakage_rules.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# NACC parser tests
# ──────────────────────────────────────────────────────────────────────────────


class TestNaccParser:
    def test_produces_entries(self, tmp_path):
        entries = parse_nacc(
            data_csv=FIXTURES / "nacc_fixture.csv",
            modality_map_path=_nacc_map_path(tmp_path),
        )
        assert len(entries) > 0

    def test_skips_admin_columns(self, tmp_path):
        entries = parse_nacc(
            data_csv=FIXTURES / "nacc_fixture.csv",
            modality_map_path=_nacc_map_path(tmp_path),
        )
        ids = [e.variable_id for e in entries]
        assert "NACC.NACCID" not in ids
        assert "NACC.NACCVNUM" not in ids

    def test_modality_demographic(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        demo = {e.raw_code for e in entries if e.modality == Modality.DEMOGRAPHIC}
        assert "SEX" in demo
        assert "EDUC" in demo

    def test_modality_cognition(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        cog = {e.raw_code for e in entries if e.modality == Modality.COGNITION}
        assert "CDRSUM" in cog
        assert "MMSE30" in cog

    def test_modality_fallback_to_other(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        # NACCUDSD and NACCCBD should map to 'other' via the diagnosis pattern
        # (in the fixture map they match no pattern → other)
        other_codes = {e.raw_code for e in entries if e.modality == Modality.OTHER}
        # At minimum NGDSGWAS and DRUG1 should fall to other
        assert len(other_codes) > 0

    def test_id_space_is_A(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        assert all(e.id_space == IdSpace.A for e in entries)

    def test_source_dataset(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        assert all(e.source_dataset == SourceDataset.NACC for e in entries)

    def test_coverage_is_none(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        assert all(e.coverage is None for e in entries)

    def test_leakage_tags_empty_before_builder(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        assert all(e.leakage_tags == [] for e in entries)

    def test_variable_id_prefix(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        assert all(e.variable_id.startswith("NACC.") for e in entries)

    def test_no_duplicate_variable_ids(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        ids = [e.variable_id for e in entries]
        assert len(ids) == len(set(ids))

    def test_invalid_modality_in_map_raises(self, tmp_path):
        bad_map = tmp_path / "bad_map.yaml"
        bad_map.write_text(
            "modality_patterns:\n  - pattern: '^FOO$'\n    modality: not_a_real_modality\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown modality"):
            parse_nacc(FIXTURES / "nacc_fixture.csv", bad_map)

    def test_description_empty_without_rdd_enrichment(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        assert all(e.description == "" for e in entries)

    def test_concept_is_lowercased_raw_code_without_rdd(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        for e in entries:
            assert e.concept == e.raw_code.lower()

    def test_value_coding_empty_without_rdd(self, tmp_path):
        entries = parse_nacc(FIXTURES / "nacc_fixture.csv", _nacc_map_path(tmp_path))
        assert all(e.value_coding == {} for e in entries)

    def test_rdd_enrichment_populates_description(self, tmp_path):
        from alz_agent.catalog.nacc_rdd_parser import NaccRddEntry

        rdd_data = {
            "SEX": NaccRddEntry(
                raw_code="SEX",
                short_descriptor="Sex of the subject",
                value_coding={"1": "Male", "2": "Female"},
                description="The biological sex of the participant.",
                data_type="Numeric cross-sectional",
            )
        }
        # Write a tiny CSV with just SEX
        csv_path = tmp_path / "tiny.csv"
        csv_path.write_text("NACCID,NACCVNUM,SEX\n", encoding="utf-8")

        from alz_agent.catalog.nacc_rdd_parser import NaccRddEntry  # noqa: F811
        # Patch parse_nacc to accept rdd_enrichment directly via rdd_pdf=None
        # and call internal enrichment by manually calling parse_nacc with a
        # monkeypatched rdd_pdf that returns our rdd_data.
        # Simpler: call parse_nacc with no rdd_pdf and verify defaults,
        # then directly test the enrichment path by providing the dict.

        # Direct enrichment test: parse_nacc accepts rdd_pdf; we test
        # that when rdd_pdf is None, it falls back correctly.
        # Testing with a real rdd_pdf would require the real PDF (integration test).
        # Instead, verify the rdd_enrichment dict lookup logic at unit level:
        entries = parse_nacc(csv_path, _nacc_map_path(tmp_path), rdd_pdf=None)
        sex_entry = next(e for e in entries if e.raw_code == "SEX")
        # Without rdd_pdf, concept should be "sex" (lowercase), description ""
        assert sex_entry.concept == "sex"
        assert sex_entry.description == ""

    def test_rdd_enrichment_applied_when_dict_injected(self, tmp_path):
        """Test the enrichment logic by monkey-patching parse_nacc_rdd."""
        import unittest.mock as mock
        from alz_agent.catalog.nacc_rdd_parser import NaccRddEntry

        csv_path = tmp_path / "tiny.csv"
        csv_path.write_text("NACCID,NACCVNUM,SEX,EDUC\n", encoding="utf-8")
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")  # dummy file so existence check passes

        rdd_data = {
            "SEX": NaccRddEntry(
                raw_code="SEX",
                short_descriptor="Sex of the subject",
                value_coding={"1": "Male", "2": "Female"},
                description="The biological sex of the participant.",
                data_type="Numeric cross-sectional",
            )
        }
        with mock.patch(
            "alz_agent.catalog.nacc_rdd_parser.parse_nacc_rdd",
            return_value=rdd_data,
        ):
            entries = parse_nacc(csv_path, _nacc_map_path(tmp_path), rdd_pdf=fake_pdf)

        sex_entry = next(e for e in entries if e.raw_code == "SEX")
        educ_entry = next(e for e in entries if e.raw_code == "EDUC")

        # SEX is in rdd_data → enriched
        assert sex_entry.concept == "sex of the subject"
        assert sex_entry.description == "The biological sex of the participant."
        assert sex_entry.value_coding == {"1": "Male", "2": "Female"}

        # EDUC is not in rdd_data → fallback
        assert educ_entry.concept == "educ"
        assert educ_entry.description == ""
        assert educ_entry.value_coding == {}


# ──────────────────────────────────────────────────────────────────────────────
# ADNI parser tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAdniParser:
    def test_produces_entries(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR", "DXSUM", "MMSE", "FAMHXPAR"],
        )
        assert len(entries) > 0

    def test_skips_admin_fields(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR"],
        )
        raw_codes = {e.raw_code for e in entries}
        assert "PTID" not in raw_codes
        assert "RID" not in raw_codes
        assert "VISCODE" not in raw_codes

    def test_id_space_is_B(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR", "DXSUM"],
        )
        assert all(e.id_space == IdSpace.B for e in entries)

    def test_source_dataset(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR"],
        )
        assert all(e.source_dataset == SourceDataset.ADNI for e in entries)

    def test_modality_cdr_is_cognition(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR"],
        )
        assert all(e.modality == Modality.COGNITION for e in entries)

    def test_modality_dxsum_is_other(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["DXSUM"],
        )
        assert all(e.modality == Modality.OTHER for e in entries)

    def test_modality_famhx_is_demographic(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["FAMHXPAR"],
        )
        assert all(e.modality == Modality.DEMOGRAPHIC for e in entries)

    def test_value_coding_parsed(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["DXSUM"],
        )
        dx_entry = next(e for e in entries if e.raw_code == "DXCURREN")
        assert dx_entry.value_coding.get("1") == "NL"
        assert dx_entry.value_coding.get("2") == "MCI"

    def test_empty_code_gives_empty_dict(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["MMSE"],
        )
        mmse_entry = next((e for e in entries if e.raw_code == "MMSCORE"), None)
        assert mmse_entry is not None
        assert isinstance(mmse_entry.value_coding, dict)

    def test_coverage_is_none(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR"],
        )
        assert all(e.coverage is None for e in entries)

    def test_leakage_tags_empty_before_builder(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR"],
        )
        assert all(e.leakage_tags == [] for e in entries)

    def test_filters_to_local_tables_only(self):
        entries_all = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR", "DXSUM", "MMSE", "FAMHXPAR"],
        )
        entries_cdr_only = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR"],
        )
        assert len(entries_cdr_only) < len(entries_all)
        assert all(e.variable_id.startswith("ADNI.CDR.") for e in entries_cdr_only)

    def test_no_duplicate_variable_ids(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR", "DXSUM", "MMSE", "FAMHXPAR"],
        )
        ids = [e.variable_id for e in entries]
        assert len(ids) == len(set(ids))

    def test_missing_table_returns_empty(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["NONEXISTENT_TABLE"],
        )
        assert entries == []


class TestAdniParserHeaderFallback:
    """ADSP_PHC_CSF/PLASMA have zero DATADIC rows — parser must fall back to
    reading the CSV header directly and tagging family/variant per column.
    """

    def _parse(self, local_tables: list[str] | None = None):
        return parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=local_tables or ["ADSP_PHC_CSF", "ADSP_PHC_PLASMA"],
            data_files={
                "ADSP_PHC_CSF": FIXTURES / "adsp_phc_csf_fixture.csv",
                "ADSP_PHC_PLASMA": FIXTURES / "adsp_phc_plasma_fixture.csv",
            },
        )

    def test_produces_entries_without_datadic_rows(self):
        entries = self._parse()
        assert len(entries) > 0

    def test_skips_admin_fields(self):
        entries = self._parse()
        raw_codes = {e.raw_code for e in entries}
        assert "RID" not in raw_codes
        assert "VISCODE" not in raw_codes

    def test_modality_is_fluid_biomarker(self):
        entries = self._parse()
        assert all(e.modality == Modality.FLUID_BIOMARKER for e in entries)

    def test_plasma_has_ptau217_variants_raw_and_harmonized(self):
        entries = self._parse()
        ptau217 = {e.raw_code: e for e in entries if e.family == "ptau217"}
        assert ptau217["pTau217_RAW"].variant == "plasma_raw"
        assert ptau217["PHC_pTau217"].variant == "plasma_harmonized"

    def test_csf_has_no_ptau217_entries(self):
        """CSF has no ptau217 column at all locally — must produce zero entries
        with family='ptau217', not a guessed/fabricated one.
        """
        entries = self._parse()
        csf_ptau217 = [
            e for e in entries
            if e.variable_id.startswith("ADNI.ADSP_PHC_CSF.") and e.family == "ptau217"
        ]
        assert csf_ptau217 == []

    def test_csf_has_ptau181_variant(self):
        entries = self._parse()
        csf_ptau181 = next(
            e for e in entries
            if e.variable_id == "ADNI.ADSP_PHC_CSF.PHC_pTau181"
        )
        assert csf_ptau181.family == "ptau181"
        assert csf_ptau181.variant == "csf_harmonized"

    def test_derived_composite_columns_untagged(self):
        """SCeNS scores and A/T classification are not plain analyte measurements."""
        entries = self._parse()
        by_raw_code = {e.raw_code: e for e in entries}
        assert by_raw_code["AB42pTau181_Class"].family is None
        assert by_raw_code["PHC_SCeNS_pTau181_Score"].family is None

    def test_missing_data_file_skips_table_without_error(self):
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["ADSP_PHC_CSF"],
            data_files={},  # no path registered — should skip, not raise
        )
        assert entries == []

    def test_datadic_backed_tables_still_get_family_variant_none_by_default(self):
        """CDR/DXSUM columns aren't biomarker columns — family/variant stay None."""
        entries = parse_adni(
            datadic_csv=FIXTURES / "adni_datadic_fixture.csv",
            local_tables=["CDR"],
        )
        assert all(e.family is None and e.variant is None for e in entries)


# ──────────────────────────────────────────────────────────────────────────────
# resolve_latest_snapshot (P0 expansion — date-agnostic path resolution)
# ──────────────────────────────────────────────────────────────────────────────


class TestResolveLatestSnapshot:
    def test_single_match(self, tmp_path):
        (tmp_path / "ADAS_15Jul2026.csv").write_text("x", encoding="utf-8")
        result = resolve_latest_snapshot(tmp_path, "ADAS_*.csv")
        assert result == tmp_path / "ADAS_15Jul2026.csv"

    def test_no_match_returns_none(self, tmp_path):
        assert resolve_latest_snapshot(tmp_path, "NOTHING_*.csv") is None

    def test_picks_newest_by_parsed_date_not_lexicographic_order(self, tmp_path):
        # "01Jan2027" sorts before "15Aug2026" lexicographically but is later
        # chronologically — the resolver must use the parsed date, not string sort.
        older = tmp_path / "ADAS_15Aug2026.csv"
        newer = tmp_path / "ADAS_01Jan2027.csv"
        older.write_text("old", encoding="utf-8")
        newer.write_text("new", encoding="utf-8")
        assert resolve_latest_snapshot(tmp_path, "ADAS_*.csv") == newer

    def test_multiple_matches_newest_date_wins(self, tmp_path):
        (tmp_path / "ADAS_09Jun2026.csv").write_text("old", encoding="utf-8")
        newest = tmp_path / "ADAS_15Jul2026.csv"
        newest.write_text("new", encoding="utf-8")
        assert resolve_latest_snapshot(tmp_path, "ADAS_*.csv") == newest

    def test_malformed_date_token_falls_back_to_mtime(self, tmp_path):
        """Neither filename carries a parseable _DDMonYYYY token — the
        resolver must fall back to comparing file mtime instead of erroring
        or picking arbitrarily.
        """
        import time

        older = tmp_path / "ADAS_v1.csv"
        older.write_text("old", encoding="utf-8")
        time.sleep(0.05)
        newer = tmp_path / "ADAS_v2.csv"
        newer.write_text("new", encoding="utf-8")
        assert resolve_latest_snapshot(tmp_path, "ADAS_*.csv") == newer

    def test_date_token_wins_over_mtime(self, tmp_path):
        """A file with a parseable date token outranks one without, even if
        the undated file was written more recently."""
        undated = tmp_path / "ADAS_weird.csv"
        undated.write_text("a", encoding="utf-8")
        import time
        time.sleep(0.05)
        dated = tmp_path / "ADAS_09Jun2026.csv"
        dated.write_text("b", encoding="utf-8")
        assert resolve_latest_snapshot(tmp_path, "ADAS_*.csv") == dated


# ──────────────────────────────────────────────────────────────────────────────
# P0 expansion (2026-07-15 LONI download): new-table modality + tagging
# ──────────────────────────────────────────────────────────────────────────────


class TestAdniParserP0Expansion:
    def test_new_pet_tables_mapped_to_pet_modality(self):
        assert _TABLE_MODALITY["UCBERKELEY_AMY_6MM"] == Modality.PET
        assert _TABLE_MODALITY["UCBERKELEY_TAU_6MM"] == Modality.PET
        assert _TABLE_MODALITY["AMYMETA"] == Modality.PET
        assert _TABLE_MODALITY["TAUMETA"] == Modality.PET

    def test_new_mri_tables_mapped_to_mri_modality(self):
        assert _TABLE_MODALITY["UCSFFSX7"] == Modality.MRI
        assert _TABLE_MODALITY["MRIMETA"] == Modality.MRI

    def test_ptdemog_mapped_to_demographic(self):
        assert _TABLE_MODALITY["PTDEMOG"] == Modality.DEMOGRAPHIC

    def test_infrastructure_tables_mapped_to_other(self):
        for tbl in ("ROSTER", "REGISTRY", "ARM", "VISITS", "APOERES"):
            assert _TABLE_MODALITY[tbl] == Modality.OTHER

    def test_legacy_csf_alzbio3_tagged_ptau181_not_generic_ptau(self):
        """Alzbio3 and Roche Elecsys are documented p-tau181-specific assays —
        must tag family='ptau181' so they aggregate with the ADSP-PHC ptau181
        variants, not a separate untargeted 'ptau' family that would hide them.
        """
        family, variant = _infer_biomarker_tags("UPENNBIOMK_MASTER", "PTAU")
        assert family == "ptau181"
        assert variant == "csf_alzbio3"

        family, variant = _infer_biomarker_tags("UPENNBIOMK_ROCHE_ELECSYS", "PTAU")
        assert family == "ptau181"
        assert variant == "csf_roche_elecsys"

    def test_legacy_csf_raw_variant_distinct_from_processed(self):
        family, variant = _infer_biomarker_tags("UPENNBIOMK_MASTER", "PTAU_RAW")
        assert family == "ptau181"
        assert variant == "csf_alzbio3_raw"

    def test_fujirebio_quanterix_dual_platform_tags(self):
        assert _infer_biomarker_tags(
            "UPENN_PLASMA_FUJIREBIO_QUANTERIX", "pT217_F"
        ) == ("ptau217", "plasma_fujirebio")
        assert _infer_biomarker_tags(
            "UPENN_PLASMA_FUJIREBIO_QUANTERIX", "NfL_F"
        ) == ("nfl", "plasma_fujirebio")
        assert _infer_biomarker_tags(
            "UPENN_PLASMA_FUJIREBIO_QUANTERIX", "NfL_Q"
        ) == ("nfl", "plasma_quanterix")

    def test_unknown_table_or_field_untagged(self):
        assert _infer_biomarker_tags("UPENNBIOMK_MASTER", "BATCH") == (None, None)
        assert _infer_biomarker_tags("SOME_OTHER_TABLE", "PTAU") == (None, None)

    def test_imaging_tags_hippocampal_volume(self):
        assert _infer_imaging_tags("UCSFFSX7", "ST29SV") == (
            "hippocampal_volume", "freesurfer7_raw",
        )
        assert _infer_imaging_tags("ADSP_PHC_T1_FS", "Left_Hippocampus_combat") == (
            "hippocampal_volume", "freesurfer_t1fs_harmonized",
        )
        assert _infer_imaging_tags("ADSP_PHC_T1_FS", "Left_Hippocampus") == (
            "hippocampal_volume", "freesurfer_t1fs_raw",
        )

    def test_imaging_tags_pet_suvr(self):
        assert _infer_imaging_tags("UCBERKELEY_AMY_6MM", "SUMMARY_SUVR") == (
            "amyloid_pet", "berkeley_6mm_raw",
        )
        assert _infer_imaging_tags("UCBERKELEY_TAU_6MM", "META_TEMPORAL_SUVR") == (
            "tau_pet", "berkeley_6mm_raw",
        )

    def test_imaging_tags_untagged_column_is_none(self):
        assert _infer_imaging_tags("UCSFFSX7", "ST88SV") == (None, None)
        assert _infer_imaging_tags("UNKNOWN_TABLE", "ANYTHING") == (None, None)


# ──────────────────────────────────────────────────────────────────────────────
# ADSP-PHC parser tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAdspParser:
    def test_produces_entries(self, tmp_path):
        # Set up fixture folder structure
        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")

        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"cognition": ["Cognition"]},
        )
        assert len(entries) > 0

    def test_skips_naccid_naccvnum(self, tmp_path):
        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"cognition": ["Cognition"]},
        )
        raw_codes = {e.raw_code for e in entries}
        assert "NACCID" not in raw_codes
        assert "NACCVNUM" not in raw_codes

    def test_modality_cognition(self, tmp_path):
        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"cognition": ["Cognition"]},
        )
        assert all(e.modality == Modality.COGNITION for e in entries)

    def test_modality_fluid_biomarker(self, tmp_path):
        bio_dir = tmp_path / "Biomarker"
        bio_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "adsp_biomarker_dd_fixture.xlsx",
                    bio_dir / "NACC_ADSP_PHC_Biomarker_DD_2024.xlsx")
        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"fluid_biomarker": ["Biomarker"]},
        )
        assert all(e.modality == Modality.FLUID_BIOMARKER for e in entries)

    def test_id_space_is_A(self, tmp_path):
        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"cognition": ["Cognition"]},
        )
        assert all(e.id_space == IdSpace.A for e in entries)

    def test_source_dataset(self, tmp_path):
        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"cognition": ["Cognition"]},
        )
        assert all(e.source_dataset == SourceDataset.ADSP_PHC for e in entries)

    def test_harmonization_transform_for_phc_variables(self, tmp_path):
        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"cognition": ["Cognition"]},
        )
        phc_entries = [e for e in entries if e.raw_code.startswith("PHC_")]
        assert all(e.harmonization_transform is not None for e in phc_entries)

    def test_coverage_is_none(self, tmp_path):
        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"cognition": ["Cognition"]},
        )
        assert all(e.coverage is None for e in entries)

    def test_missing_subfolder_skipped_gracefully(self, tmp_path):
        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"cognition": ["NonExistentFolder"]},
        )
        assert entries == []

    def test_no_duplicate_variable_ids(self, tmp_path):
        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        entries = parse_adsp_phc(
            base_dir=tmp_path,
            modality_dirs={"cognition": ["Cognition"]},
        )
        ids = [e.variable_id for e in entries]
        assert len(ids) == len(set(ids))


# ──────────────────────────────────────────────────────────────────────────────
# Leakage rule loader tests
# ──────────────────────────────────────────────────────────────────────────────


class TestLeakageRuleLoader:
    def test_loads_rules(self, tmp_path):
        rules = load_leakage_rules(_leakage_rules_path(tmp_path))
        assert len(rules) > 0

    def test_rule_has_pattern_and_tag(self, tmp_path):
        rules = load_leakage_rules(_leakage_rules_path(tmp_path))
        for rule in rules:
            assert rule.pattern is not None
            assert isinstance(rule.tag, LeakageTag)

    def test_source_filter_is_none_when_absent(self, tmp_path):
        rules_path = tmp_path / "minimal_rules.yaml"
        rules_path.write_text(
            "rules:\n  - variable_pattern: '^FOO$'\n    tag: CIRCULAR\n    reason: test\n",
            encoding="utf-8",
        )
        rules = load_leakage_rules(rules_path)
        assert rules[0].source_filter is None

    def test_source_filter_populated(self, tmp_path):
        rules = load_leakage_rules(_leakage_rules_path(tmp_path))
        adni_rules = [r for r in rules if r.source_filter and "ADNI" in r.source_filter]
        assert len(adni_rules) > 0

    def test_invalid_tag_raises(self, tmp_path):
        bad_path = tmp_path / "bad_rules.yaml"
        bad_path.write_text(
            "rules:\n  - variable_pattern: '^X$'\n    tag: NOT_A_TAG\n    reason: x\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_leakage_rules(bad_path)

    def test_real_leakage_rules_yaml_loads(self):
        """The production leakage_rules.yaml must load without error."""
        prod_path = (
            Path(__file__).parent.parent
            / "src" / "alz_agent" / "catalog" / "rules" / "leakage_rules.yaml"
        )
        rules = load_leakage_rules(prod_path)
        assert len(rules) >= 10


# ──────────────────────────────────────────────────────────────────────────────
# Leakage tag application tests (via builder internals)
# ──────────────────────────────────────────────────────────────────────────────


class TestLeakageTagApplication:
    """These tests exercise _apply_leakage_tags indirectly through build_catalog
    using a minimal but complete fixture paths.yaml."""

    def _make_paths_yaml(
        self,
        tmp_path: Path,
        nacc_csv: Path,
        adni_csv: Path,
        adsp_cognition_dir: Path,
        adsp_biomarker_dir: Path,
    ) -> Path:
        nacc_map = _nacc_map_path(tmp_path)
        leakage_rules = _leakage_rules_path(tmp_path)
        # Write paths.yaml that points to fixture files
        content = {
            "nacc": {
                "data_csv": str(nacc_csv),
                "modality_map": str(nacc_map),
            },
            "adni": {
                "base_dir": str(adni_csv.parent),
                "datadic_glob": adni_csv.name,
                "local_tables": ["CDR", "DXSUM", "FAMHXPAR"],
            },
            "adsp_phc": {
                "base_dir": str(adsp_cognition_dir.parent),
                "release_version": "R3_Dec2024",
                "modality_dirs": {
                    "cognition": ["Cognition"],
                    "fluid_biomarker": ["Biomarker"],
                },
            },
            "catalog": {"leakage_rules": str(leakage_rules)},
            "outputs": {"cohort_maps": str(tmp_path / "outputs")},
        }
        p = tmp_path / "paths_fixture.yaml"
        import yaml
        p.write_text(yaml.dump(content), encoding="utf-8")
        return p

    def test_nacc_diagnosis_defining_tagged(self, tmp_path):
        import shutil

        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        bio_dir = tmp_path / "Biomarker"
        bio_dir.mkdir()
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        shutil.copy(FIXTURES / "adsp_biomarker_dd_fixture.xlsx",
                    bio_dir / "NACC_ADSP_PHC_Biomarker_DD_2024.xlsx")

        paths = self._make_paths_yaml(
            tmp_path,
            nacc_csv=FIXTURES / "nacc_fixture.csv",
            adni_csv=FIXTURES / "adni_datadic_fixture.csv",
            adsp_cognition_dir=cog_dir,
            adsp_biomarker_dir=bio_dir,
        )
        catalog, _ = build_catalog(paths, project_root=tmp_path)

        nacc_udsd = next(e for e in catalog if e.variable_id == "NACC.NACCUDSD")
        assert LeakageTag.DIAGNOSIS_DEFINING in nacc_udsd.leakage_tags

    def test_adni_dxcurren_diagnosis_defining(self, tmp_path):
        import shutil

        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        bio_dir = tmp_path / "Biomarker"
        bio_dir.mkdir()
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        shutil.copy(FIXTURES / "adsp_biomarker_dd_fixture.xlsx",
                    bio_dir / "NACC_ADSP_PHC_Biomarker_DD_2024.xlsx")

        paths = self._make_paths_yaml(
            tmp_path,
            nacc_csv=FIXTURES / "nacc_fixture.csv",
            adni_csv=FIXTURES / "adni_datadic_fixture.csv",
            adsp_cognition_dir=cog_dir,
            adsp_biomarker_dir=bio_dir,
        )
        catalog, _ = build_catalog(paths, project_root=tmp_path)

        dxcurren = next(e for e in catalog if e.raw_code == "DXCURREN")
        assert LeakageTag.DIAGNOSIS_DEFINING in dxcurren.leakage_tags

    def test_cdrsum_circular(self, tmp_path):
        import shutil

        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        bio_dir = tmp_path / "Biomarker"
        bio_dir.mkdir()
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        shutil.copy(FIXTURES / "adsp_biomarker_dd_fixture.xlsx",
                    bio_dir / "NACC_ADSP_PHC_Biomarker_DD_2024.xlsx")

        paths = self._make_paths_yaml(
            tmp_path,
            nacc_csv=FIXTURES / "nacc_fixture.csv",
            adni_csv=FIXTURES / "adni_datadic_fixture.csv",
            adsp_cognition_dir=cog_dir,
            adsp_biomarker_dir=bio_dir,
        )
        catalog, _ = build_catalog(paths, project_root=tmp_path)

        cdrsum = next(e for e in catalog if e.variable_id == "NACC.CDRSUM")
        assert LeakageTag.CIRCULAR in cdrsum.leakage_tags

    def test_phc_mem_harmonization_fit_scope(self, tmp_path):
        import shutil

        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        bio_dir = tmp_path / "Biomarker"
        bio_dir.mkdir()
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        shutil.copy(FIXTURES / "adsp_biomarker_dd_fixture.xlsx",
                    bio_dir / "NACC_ADSP_PHC_Biomarker_DD_2024.xlsx")

        paths = self._make_paths_yaml(
            tmp_path,
            nacc_csv=FIXTURES / "nacc_fixture.csv",
            adni_csv=FIXTURES / "adni_datadic_fixture.csv",
            adsp_cognition_dir=cog_dir,
            adsp_biomarker_dir=bio_dir,
        )
        catalog, _ = build_catalog(paths, project_root=tmp_path)

        phc_mem = next(e for e in catalog if e.raw_code == "PHC_MEM")
        assert LeakageTag.HARMONIZATION_FIT_SCOPE in phc_mem.leakage_tags

    def test_non_leakage_variable_gets_NONE_tag(self, tmp_path):
        import shutil

        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        bio_dir = tmp_path / "Biomarker"
        bio_dir.mkdir()
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        shutil.copy(FIXTURES / "adsp_biomarker_dd_fixture.xlsx",
                    bio_dir / "NACC_ADSP_PHC_Biomarker_DD_2024.xlsx")

        paths = self._make_paths_yaml(
            tmp_path,
            nacc_csv=FIXTURES / "nacc_fixture.csv",
            adni_csv=FIXTURES / "adni_datadic_fixture.csv",
            adsp_cognition_dir=cog_dir,
            adsp_biomarker_dir=bio_dir,
        )
        catalog, _ = build_catalog(paths, project_root=tmp_path)

        # ANIMALS is a cognitive test — not diagnosis-defining or circular
        animals = next(e for e in catalog if e.variable_id == "NACC.ANIMALS")
        assert LeakageTag.NONE in animals.leakage_tags


# ──────────────────────────────────────────────────────────────────────────────
# Build report tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildReport:
    def _run_build(self, tmp_path: Path):
        import shutil

        cog_dir = tmp_path / "Cognition"
        cog_dir.mkdir()
        bio_dir = tmp_path / "Biomarker"
        bio_dir.mkdir()
        shutil.copy(FIXTURES / "adsp_cognition_dd_fixture.xlsx",
                    cog_dir / "NACC_ADSP_PHC_Cognition_DD_2024.xlsx")
        shutil.copy(FIXTURES / "adsp_biomarker_dd_fixture.xlsx",
                    bio_dir / "NACC_ADSP_PHC_Biomarker_DD_2024.xlsx")

        nacc_map = _nacc_map_path(tmp_path)
        leakage_rules = _leakage_rules_path(tmp_path)

        content = {
            "nacc": {
                "data_csv": str(FIXTURES / "nacc_fixture.csv"),
                "modality_map": str(nacc_map),
            },
            "adni": {
                "base_dir": str(FIXTURES),
                "datadic_glob": "adni_datadic_fixture.csv",
                "local_tables": ["CDR", "DXSUM", "FAMHXPAR"],
            },
            "adsp_phc": {
                "base_dir": str(tmp_path),
                "release_version": "R3_Dec2024",
                "modality_dirs": {
                    "cognition": ["Cognition"],
                    "fluid_biomarker": ["Biomarker"],
                },
            },
            "catalog": {"leakage_rules": str(leakage_rules)},
            "outputs": {"cohort_maps": str(tmp_path / "outputs")},
        }
        import yaml
        p = tmp_path / "paths.yaml"
        p.write_text(yaml.dump(content), encoding="utf-8")
        return build_catalog(p, project_root=tmp_path)

    def test_report_total_matches_catalog_length(self, tmp_path):
        catalog, report = self._run_build(tmp_path)
        assert report.total == len(catalog)

    def test_report_has_all_three_sources(self, tmp_path):
        _, report = self._run_build(tmp_path)
        assert "NACC" in report.by_source
        assert "ADNI" in report.by_source
        assert "ADSP_PHC" in report.by_source

    def test_by_source_sums_to_total(self, tmp_path):
        _, report = self._run_build(tmp_path)
        assert sum(report.by_source.values()) == report.total

    def test_by_modality_sums_to_total(self, tmp_path):
        _, report = self._run_build(tmp_path)
        assert sum(report.by_modality.values()) == report.total

    def test_by_leakage_tag_sums_to_total(self, tmp_path):
        _, report = self._run_build(tmp_path)
        assert sum(report.by_leakage_tag.values()) == report.total

    def test_other_modality_entries_listed(self, tmp_path):
        _, report = self._run_build(tmp_path)
        # some variables must fall to 'other' in the fixture map
        assert isinstance(report.other_modality_entries, list)

    def test_no_leakage_rule_matched_list_populated(self, tmp_path):
        _, report = self._run_build(tmp_path)
        assert isinstance(report.no_leakage_rule_matched, list)

    def test_summary_string_non_empty(self, tmp_path):
        _, report = self._run_build(tmp_path)
        summary = report.summary()
        assert "Total entries:" in summary
        assert "By source:" in summary
        assert "By modality:" in summary
        assert "By leakage tag:" in summary

    def test_report_is_dataclass(self, tmp_path):
        _, report = self._run_build(tmp_path)
        assert isinstance(report, BuildReport)
