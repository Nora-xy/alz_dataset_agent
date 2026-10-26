"""WU3/WU4/WU5 tests — Space B (ADNI) cross-table event anchoring, the derived
age_at_event variable, event vs baseline etiology eligibility, capability-gap
abstain wording, and the synthesizer variable-identifier guard.

All data is synthetic (written into a tmp ADNI layout); no real files are read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alz_agent.backend.local_backend import LocalDataBackend
from alz_agent.schemas import FilterSpec, IdSpace

# 16 participants keeps every reported cohort above the suppression threshold (10).
_RIDS = [str(i) for i in range(1, 17)]
_AD_AT_BASELINE = set(_RIDS[:6])   # RIDs 1..6 get AD at baseline
# All 16 have AD by m12 (RIDs 7..16 are incident at m12).


def _write_adni_fixture(base: Path, *, with_etiology_cols: bool = True) -> None:
    base.mkdir(parents=True, exist_ok=True)

    dx_rows = []
    ptd_rows = []
    mmse_rows = []
    demodx_rows = []
    for rid in _RIDS:
        ad_bl = rid in _AD_AT_BASELINE
        # DXSUM: bl (MCI), m12 (dementia + AD), plus AD at bl for the baseline group
        dx_rows.append({
            "RID": rid, "VISCODE": "bl", "EXAMDATE": "2010-06-15",
            "DIAGNOSIS": 3 if ad_bl else 2,
            "DXAD": 1 if ad_bl else "", "DXMDUE": "", "DXDDUE": "",
        })
        dx_rows.append({
            "RID": rid, "VISCODE": "m12", "EXAMDATE": "2011-06-15",
            "DIAGNOSIS": 3, "DXAD": 1, "DXMDUE": "", "DXDDUE": "",
        })
        # PTDEMOG: month/year DOB
        ptd_rows.append({"RID": rid, "VISCODE": "sc", "PTDOB": "01/1940", "PTDOBYY": "1940-01-01"})
        # MMSE: 28 at baseline, 24 at m12 — so anchoring to first AD visit changes the mean
        mmse_rows.append({"RID": rid, "VISCODE": "bl", "MMSCORE": 28})
        mmse_rows.append({"RID": rid, "VISCODE": "m12", "MMSCORE": 24})
        # Harmonized age at each visit (fixed so the harmonized cross-check mean is exact)
        demodx_rows.append({"RID": rid, "VISCODE": "bl", "PHC_Age_Diagnosis": 70.5})
        demodx_rows.append({"RID": rid, "VISCODE": "m12", "PHC_Age_Diagnosis": 71.5})

    dx = pd.DataFrame(dx_rows)
    if not with_etiology_cols:
        dx = dx.drop(columns=["DXAD", "DXMDUE", "DXDDUE"])

    dx.to_csv(base / "DXSUM_15Jul2026.csv", index=False)
    pd.DataFrame(ptd_rows).to_csv(base / "PTDEMOG_15Jul2026.csv", index=False)
    pd.DataFrame(mmse_rows).to_csv(base / "MMSE_15Jul2026.csv", index=False)
    pd.DataFrame(demodx_rows).to_csv(base / "ADSP_PHC_DEMODX_15Jul2026.csv", index=False)


def _make_backend(tmp_path: Path, *, with_etiology_cols: bool = True) -> LocalDataBackend:
    base = tmp_path / "ADNI"
    _write_adni_fixture(base, with_etiology_cols=with_etiology_cols)
    config = {
        "adni": {"base_dir": "ADNI"},
        "backend": {
            "adni_data_files": {
                "DXSUM": "DXSUM_*.csv",
                "PTDEMOG": "PTDEMOG_*.csv",
                "MMSE": "MMSE_*.csv",
                "ADSP_PHC_DEMODX": "ADSP_PHC_DEMODX_*.csv",
            }
        },
    }
    return LocalDataBackend(config, tmp_path)


# ---------------------------------------------------------------------------
# Eligibility — event_etiology (any visit) vs baseline_etiology (first visit)
# ---------------------------------------------------------------------------


class TestSpaceBEtiologyEligibility:
    def test_event_etiology_is_any_visit(self, tmp_path):
        be = _make_backend(tmp_path)
        fs = FilterSpec(id_space=IdSpace.B, event_etiology=["ad"])
        assert len(be.eligible_ids(fs)) == 16  # all participants ever have AD

    def test_baseline_etiology_is_first_visit_only(self, tmp_path):
        be = _make_backend(tmp_path)
        fs = FilterSpec(id_space=IdSpace.B, baseline_etiology=["ad"])
        assert be.eligible_ids(fs) == _AD_AT_BASELINE  # only the 6 with AD at bl

    def test_event_etiology_superset_of_baseline(self, tmp_path):
        be = _make_backend(tmp_path)
        ev = be.eligible_ids(FilterSpec(id_space=IdSpace.B, event_etiology=["ad"]))
        bl = be.eligible_ids(FilterSpec(id_space=IdSpace.B, baseline_etiology=["ad"]))
        assert bl < ev


# ---------------------------------------------------------------------------
# Derived age_at_event
# ---------------------------------------------------------------------------


class TestAgeAtEvent:
    def test_etiologic_ad_age_grounded_with_provenance(self, tmp_path):
        be = _make_backend(tmp_path)
        fs = FilterSpec(id_space=IdSpace.B, event_etiology=["ad"])
        r = be.aggregate_stat("ADNI.DERIVED.age_at_event", "mean", fs, "first_diagnosis")
        assert r.grounded, r.abstain_reason
        assert r.value["n"] == 16
        ev = r.evidence[0]
        assert ev.source_table == "PTDEMOG+DXSUM"
        assert ev.source_level == "raw"
        assert "etiologic AD" in ev.diagnosis_basis
        # Harmonized cross-check + DOB-precision note ride in the evidence text.
        assert "harmonized cross-check" in ev.aggregate_ref
        assert "month/year" in ev.aggregate_ref
        # Deterministic phrasing + precision metadata surface in query_spec.
        assert ev.query_spec["measurement_label"] == (
            "first observed AD diagnosis in ADNI (study record)"
        )
        assert ev.query_spec["dob_precision"] == "month/year"

    def test_harmonized_cross_check_value(self, tmp_path):
        be = _make_backend(tmp_path)
        fs = FilterSpec(id_space=IdSpace.B, event_etiology=["ad"])
        r = be.aggregate_stat("ADNI.DERIVED.age_at_event", "mean", fs, "first_diagnosis")
        # RIDs 1..6 anchor at bl (PHC 70.5); RIDs 7..16 anchor at m12 (PHC 71.5).
        expected = round((6 * 70.5 + 10 * 71.5) / 16, 2)
        assert r.evidence[0].query_spec["harmonized_value"] == expected
        assert r.evidence[0].query_spec["harmonized_n"] == 16

    def test_raw_age_uses_anchored_examdate(self, tmp_path):
        be = _make_backend(tmp_path)
        fs = FilterSpec(id_space=IdSpace.B, event_etiology=["ad"])
        r = be.aggregate_stat("ADNI.DERIVED.age_at_event", "mean", fs, "first_diagnosis")
        # DOB 1940-01-01; anchors at 2010-06-15 (bl, ~70.4y) or 2011-06-15 (m12, ~71.4y).
        # Mean must fall strictly between the two single-visit extremes.
        assert 70.0 < r.value["value"] < 72.0

    def test_capability_gap_when_no_etiology_columns(self, tmp_path):
        be = _make_backend(tmp_path, with_etiology_cols=False)
        fs = FilterSpec(id_space=IdSpace.B, event_etiology=["ad"])
        r = be.aggregate_stat("ADNI.DERIVED.age_at_event", "mean", fs, "first_diagnosis")
        # event_etiology eligibility itself fails first — but the message must name the
        # missing etiology columns, not imply the data is simply absent.
        assert not r.grounded
        assert "DXAD" in r.abstain_reason or "etiology" in r.abstain_reason.lower()


# ---------------------------------------------------------------------------
# Generalized anchoring — a non-age measure at the first-diagnosis visit
# ---------------------------------------------------------------------------


class TestGeneralizedAnchoring:
    def test_mmse_at_first_diagnosis_differs_from_baseline(self, tmp_path):
        be = _make_backend(tmp_path)
        fs = FilterSpec(id_space=IdSpace.B, event_etiology=["ad"])
        at_dx = be.aggregate_stat("ADNI.MMSE.MMSCORE", "mean", fs, "first_diagnosis")
        at_bl = be.aggregate_stat("ADNI.MMSE.MMSCORE", "mean", fs, None)
        assert at_dx.grounded and at_bl.grounded
        # RIDs 1..6 anchor at bl (MMSE 28); RIDs 7..16 at m12 (MMSE 24).
        assert at_dx.value["value"] == round((6 * 28 + 10 * 24) / 16, 2)
        assert at_bl.value["value"] == 28.0  # baseline row is always bl
        assert at_dx.value["value"] != at_bl.value["value"]

    def test_anchored_measure_carries_provenance(self, tmp_path):
        be = _make_backend(tmp_path)
        fs = FilterSpec(id_space=IdSpace.B, event_etiology=["ad"])
        r = be.aggregate_stat("ADNI.MMSE.MMSCORE", "mean", fs, "first_diagnosis")
        ev = r.evidence[0]
        assert ev.source_table == "MMSE"
        assert ev.source_field == "MMSCORE"
        assert ev.source_level == "raw"
        assert "etiologic AD" in ev.diagnosis_basis


# ---------------------------------------------------------------------------
# Synthesizer variable-identifier guard
# ---------------------------------------------------------------------------


def _report_with(text: str) -> "SynthesisReport":
    from alz_agent.schemas import (
        DataEvidence,
        ExecutionContext,
        SynthesisReport,
        SynthesisSection,
        ToolResult,
    )
    from alz_agent.tools.deterministic import StatResult

    tr = ToolResult.ok(
        StatResult(variable_id="ADNI.DERIVED.age_at_event", stat_type="mean",
                   value=73.7, n=16, event_anchor="first_diagnosis"),
        evidence=[DataEvidence(
            query_spec={"variable_id": "ADNI.DERIVED.age_at_event"},
            aggregate_ref="age_at_event(...): 73.7 (n=16)",
            source_table="PTDEMOG+DXSUM", source_field="PTDOB,EXAMDATE", source_level="raw",
        )],
    )
    ctx = ExecutionContext(results={"age_stat": tr})
    return SynthesisReport(
        question="q",
        sections=[SynthesisSection(section_id="direct_answer", title="Answer", content=text)],
        execution_context=ctx,
    )


class TestVariableIdGuard:
    def test_fabricated_adni_other_id_is_flagged(self):
        from alz_agent.agent.hallucination_guard import _check_synthesis_variable_ids

        report = _report_with("The mean age from ADNI.OTHER.NACCAGE was 73.7.")
        ok, violations = _check_synthesis_variable_ids(report)
        assert not ok
        assert "ADNI.OTHER.NACCAGE" in violations

    def test_nacc_id_in_space_b_report_is_flagged(self):
        from alz_agent.agent.hallucination_guard import _check_synthesis_variable_ids

        report = _report_with("Computed from NACC.NACCAGE at the diagnosis visit.")
        ok, violations = _check_synthesis_variable_ids(report)
        assert not ok
        assert "NACC.NACCAGE" in violations

    def test_grounded_derived_id_passes(self):
        from alz_agent.agent.hallucination_guard import _check_synthesis_variable_ids

        report = _report_with("The derived variable ADNI.DERIVED.age_at_event gave 73.7.")
        ok, violations = _check_synthesis_variable_ids(report)
        assert ok, violations
