"""Tests for the analysis engine (pure functions and LocalAnalysisEngine).

Coverage:
  - _binary_groups: positive/negative splits, NA exclusion, custom positive_values
  - group_comparison_impl: means, SDs, p-value, insufficient group guard
  - validate_aggregate_result: scalar acceptance, list/DataFrame rejection
  - _security_check: dangerous pattern rejection
  - LocalAnalysisEngine.group_comparison: full integration with synthetic backend
  - LocalAnalysisEngine.run_generated_code: sandbox success and security rejection
  - _run_ols / _run_logistic: pure regression helpers
  - LocalAnalysisEngine.adjusted_regression: linear and logistic, suppression, Space B abstain
"""

from __future__ import annotations

import math
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alz_agent.analysis.engine import (
    _binary_groups,
    _security_check,
    group_comparison_impl,
    validate_aggregate_result,
)
from alz_agent.schemas import (
    FilterSpec,
    GroupComparisonResult,
    IdSpace,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(**kwargs: list) -> pd.DataFrame:
    """Build a minimal DataFrame from keyword-argument lists."""
    return pd.DataFrame(kwargs)


def _make_engine(df_by_ids: dict[str, pd.DataFrame] | None = None):
    """Return a LocalAnalysisEngine backed by a mock backend.

    The mock backend's eligible_ids() always returns the frozenset of all
    NACCID values in df_by_ids (or an empty frozenset if None).
    The engine's _load_nacc_baseline is patched to return the supplied df.
    """
    from alz_agent.analysis.engine import LocalAnalysisEngine

    backend = MagicMock()
    backend.config = {"nacc": {"data_csv": "NACC/investigator_nacc70.csv"}}
    backend.project_root = MagicMock()

    if df_by_ids is not None:
        ids = frozenset(df_by_ids.get("NACCID", []))
        backend.eligible_ids.return_value = ids
        merged = pd.DataFrame(df_by_ids)

        engine = LocalAnalysisEngine.__new__(LocalAnalysisEngine)
        engine._backend = backend
        engine._config = backend.config
        engine._root = backend.project_root
        engine._load_nacc_baseline = MagicMock(return_value=merged)
        return engine
    else:
        backend.eligible_ids.return_value = frozenset()
        engine = LocalAnalysisEngine.__new__(LocalAnalysisEngine)
        engine._backend = backend
        engine._config = backend.config
        engine._root = backend.project_root
        engine._load_nacc_baseline = MagicMock(return_value=pd.DataFrame())
        return engine


# ---------------------------------------------------------------------------
# Tests: _binary_groups
# ---------------------------------------------------------------------------


class TestBinaryGroups:
    def test_default_splits_nonzero_positive(self):
        df = _df(grp=[0, 1, 2, 0, 1])
        masks = _binary_groups(df, "grp")
        assert masks["positive"].tolist() == [False, True, True, False, True]
        assert masks["negative"].tolist() == [True, False, False, True, False]

    def test_custom_positive_values(self):
        df = _df(grp=[1, 2, 3, 4])
        masks = _binary_groups(df, "grp", positive_values=[2, 4])
        assert masks["positive"].tolist() == [False, True, False, True]
        assert masks["negative"].tolist() == [False, False, False, False]

    def test_excludes_na_rows(self):
        df = _df(grp=[0, 1, None])
        masks = _binary_groups(df, "grp")
        assert not masks["positive"].iloc[2]
        assert not masks["negative"].iloc[2]

    def test_all_same_group(self):
        df = _df(grp=[1, 1, 1])
        masks = _binary_groups(df, "grp")
        assert masks["positive"].all()
        assert not masks["negative"].any()


# ---------------------------------------------------------------------------
# Tests: group_comparison_impl
# ---------------------------------------------------------------------------


class TestGroupComparisonImpl:
    def _base_df(self) -> pd.DataFrame:
        """30-person DataFrame: 15 with heart attack (grp=1), 15 without (grp=0).

        MMSE is higher in grp=0 by design so the t-test should be significant.
        """
        import random
        rng = random.Random(42)
        mmse_pos = [rng.gauss(22, 2) for _ in range(15)]  # lower MMSE
        mmse_neg = [rng.gauss(28, 2) for _ in range(15)]  # higher MMSE
        return _df(
            NACCID=[f"N{i:03d}" for i in range(30)],
            MMSE=mmse_pos + mmse_neg,
            HA=[1] * 15 + [0] * 15,
        )

    def test_returns_correct_group_sizes(self):
        df = self._base_df()
        res = group_comparison_impl(df, "MMSE", "HA", {"positive": "HA+", "negative": "HA-"})
        assert res["positive_n"] == 15
        assert res["negative_n"] == 15
        assert res["n_total"] == 30

    def test_means_in_expected_range(self):
        df = self._base_df()
        res = group_comparison_impl(df, "MMSE", "HA", {"positive": "HA+", "negative": "HA-"})
        # HA+ group should have lower mean MMSE (~22) than HA- (~28)
        assert res["positive_mean"] < res["negative_mean"]
        assert 18 < res["positive_mean"] < 26
        assert 24 < res["negative_mean"] < 32

    def test_p_value_significant(self):
        df = self._base_df()
        res = group_comparison_impl(df, "MMSE", "HA", {"positive": "HA+", "negative": "HA-"})
        assert res["p_value"] is not None
        assert res["p_value"] < 0.05

    def test_missing_outcome_col_raises(self):
        df = _df(HA=[0, 1])
        with pytest.raises(ValueError, match="Outcome column 'MMSE'"):
            group_comparison_impl(df, "MMSE", "HA", {})

    def test_missing_group_col_raises(self):
        df = _df(MMSE=[25, 28])
        with pytest.raises(ValueError, match="Group column 'HA'"):
            group_comparison_impl(df, "MMSE", "HA", {})

    def test_insufficient_positive_group_skips_test(self):
        df = _df(MMSE=[25, 28, 27, 26, 24], HA=[1, 0, 0, 0, 0])
        res = group_comparison_impl(df, "MMSE", "HA", {"positive": "HA+", "negative": "HA-"})
        assert res["p_value"] is None
        assert "insufficient" in res.get("test_skipped_reason", "").lower()

    def test_sd_is_nan_when_single_member(self):
        df = _df(MMSE=[25, 28, 27], HA=[1, 0, 0])
        res = group_comparison_impl(df, "MMSE", "HA", {"positive": "HA+", "negative": "HA-"})
        assert math.isnan(res["positive_sd"])

    def test_na_in_outcome_excluded(self):
        df = _df(MMSE=[25, None, 28, 27, 26, None], HA=[1, 1, 0, 0, 0, 0])
        res = group_comparison_impl(df, "MMSE", "HA", {"positive": "HA+", "negative": "HA-"})
        # Only 1 valid in positive group → test skipped; n_total should be 4
        assert res["positive_n"] == 1
        assert res["negative_n"] == 3


# ---------------------------------------------------------------------------
# Tests: validate_aggregate_result
# ---------------------------------------------------------------------------


class TestValidateAggregateResult:
    def test_accepts_scalar_dict(self):
        validate_aggregate_result({"n": 42, "mean": 3.14, "label": "ok", "flag": True})

    def test_accepts_nested_scalar_dict(self):
        validate_aggregate_result({"outer": {"inner": 5}})

    def test_rejects_list_value(self):
        with pytest.raises(ValueError, match="Non-scalar"):
            validate_aggregate_result({"n": 42, "bad": [1, 2, 3]})

    def test_rejects_dataframe_value(self):
        with pytest.raises(ValueError, match="Non-scalar"):
            validate_aggregate_result({"df": pd.DataFrame({"a": [1]})})

    def test_rejects_non_dict_result(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_aggregate_result([1, 2, 3])

    def test_accepts_none_value(self):
        validate_aggregate_result({"p_value": None})


# ---------------------------------------------------------------------------
# Tests: _security_check
# ---------------------------------------------------------------------------


class TestSecurityCheck:
    def test_rejects_import(self):
        with pytest.raises(ValueError, match="import"):
            _security_check("import os\nresult = {}")

    def test_rejects_open(self):
        with pytest.raises(ValueError, match="open"):
            _security_check("f = open('/etc/passwd'); result = {}")

    def test_rejects_os_dot(self):
        with pytest.raises(ValueError, match="os"):
            _security_check("result = {'x': os.getcwd()}")

    def test_rejects_subprocess(self):
        with pytest.raises(ValueError, match="subprocess"):
            _security_check("subprocess.run(['ls']); result = {}")

    def test_rejects_eval(self):
        with pytest.raises(ValueError, match="eval"):
            _security_check("result = eval('1+1')")

    def test_rejects_exec(self):
        with pytest.raises(ValueError, match="exec"):
            _security_check("exec('print(1)'); result = {}")

    def test_accepts_safe_code(self):
        _security_check(
            "outcome = df['MMSE'].dropna()\n"
            "result = {'n': int(len(outcome)), 'mean': float(outcome.mean()), 'n_analyzed': int(len(outcome))}"
        )

    def test_rejects_dunder_import(self):
        with pytest.raises(ValueError, match="__import__"):
            _security_check("os = __import__('os'); result = {}")


# ---------------------------------------------------------------------------
# Tests: LocalAnalysisEngine.group_comparison (integration)
# ---------------------------------------------------------------------------

FILTER_A = FilterSpec(baseline_diagnosis=["mci"], id_space=IdSpace.A)
FILTER_B = FilterSpec(id_space=IdSpace.B)


class TestEngineGroupComparison:
    def _make_df(self, n_pos: int = 20, n_neg: int = 20) -> dict[str, list]:
        """Synthetic data: positive group (HA=1) has lower MMSE."""
        import random
        rng = random.Random(99)
        ids = [f"N{i:04d}" for i in range(n_pos + n_neg)]
        mmse = [rng.gauss(22, 2) for _ in range(n_pos)] + [rng.gauss(28, 2) for _ in range(n_neg)]
        ha = [1] * n_pos + [0] * n_neg
        return {"NACCID": ids, "NACCVNUM": [1] * (n_pos + n_neg), "MMSE": mmse, "HXHYPER": ha}

    def test_returns_grounded_result(self):
        df_data = self._make_df()
        engine = _make_engine(df_data)
        res = engine.group_comparison(
            outcome_col="MMSE",
            outcome_variable_id="NACC.MMSE",
            outcome_label="MMSE",
            group_col="HXHYPER",
            group_variable_id="NACC.HXHYPER",
            group_label="History of hypertension",
            group_names={"positive": "hypertensive", "negative": "normotensive"},
            filter_spec=FILTER_A,
        )
        assert res.grounded
        assert isinstance(res.value, GroupComparisonResult)

    def test_group_sizes_match(self):
        df_data = self._make_df(n_pos=15, n_neg=18)
        engine = _make_engine(df_data)
        res = engine.group_comparison(
            outcome_col="MMSE",
            outcome_variable_id="NACC.MMSE",
            outcome_label="MMSE",
            group_col="HXHYPER",
            group_variable_id="NACC.HXHYPER",
            group_label="hypertension history",
            group_names={"positive": "Hypertensive", "negative": "Normotensive"},
            filter_spec=FILTER_A,
        )
        assert res.grounded
        groups = {g.group_label: g for g in res.value.groups}
        assert groups["Hypertensive"].n == 15
        assert groups["Normotensive"].n == 18
        assert res.value.n_total == 33

    def test_abstains_on_space_b(self):
        engine = _make_engine({})
        res = engine.group_comparison(
            outcome_col="MMSE",
            outcome_variable_id="NACC.MMSE",
            outcome_label="MMSE",
            group_col="HA",
            group_variable_id="NACC.HA",
            group_label="heart attack",
            group_names={"positive": "HA+", "negative": "HA-"},
            filter_spec=FILTER_B,
        )
        assert not res.grounded
        assert "Space A" in (res.abstain_reason or "")

    def test_abstains_on_empty_eligible(self):
        engine = _make_engine(None)
        res = engine.group_comparison(
            outcome_col="MMSE",
            outcome_variable_id="NACC.MMSE",
            outcome_label="MMSE",
            group_col="HA",
            group_variable_id="NACC.HA",
            group_label="heart attack",
            group_names={"positive": "HA+", "negative": "HA-"},
            filter_spec=FILTER_A,
        )
        assert not res.grounded

    def test_evidence_contains_numeric_values(self):
        df_data = self._make_df()
        engine = _make_engine(df_data)
        res = engine.group_comparison(
            outcome_col="MMSE",
            outcome_variable_id="NACC.MMSE",
            outcome_label="MMSE",
            group_col="HXHYPER",
            group_variable_id="NACC.HXHYPER",
            group_label="hypertension",
            group_names={"positive": "Hyp+", "negative": "Hyp-"},
            filter_spec=FILTER_A,
        )
        assert res.grounded
        assert res.evidence
        ev = res.evidence[0]
        from alz_agent.schemas import DataEvidence
        assert isinstance(ev, DataEvidence)
        assert "n=" in ev.aggregate_ref
        assert "mean=" in ev.aggregate_ref


# ---------------------------------------------------------------------------
# Tests: LocalAnalysisEngine.run_generated_code (sandbox)
# ---------------------------------------------------------------------------


class TestEngineGeneratedCode:
    def _make_engine_with_df(self, n: int = 30) -> Any:
        import random
        rng = random.Random(7)
        ids = [f"N{i:04d}" for i in range(n)]
        mmse = [rng.gauss(25, 3) for _ in range(n)]
        df_data = {
            "NACCID": ids,
            "NACCVNUM": [1] * n,
            "MMSE": mmse,
        }
        return _make_engine(df_data)

    def test_safe_code_returns_aggregate_result(self):
        from alz_agent.analysis.engine import LocalAnalysisEngine
        from alz_agent.schemas import GeneratedAnalysisResult

        engine = self._make_engine_with_df(30)
        code = (
            "vals = df['MMSE'].dropna()\n"
            "result = {'n_analyzed': int(len(vals)), 'mean_mmse': round(float(vals.mean()), 3)}"
        )
        res = engine.run_generated_code(
            code=code,
            filter_spec=FILTER_A,
            required_cols=["MMSE"],
            analysis_description="mean MMSE",
        )
        assert res.grounded
        assert isinstance(res.value, GeneratedAnalysisResult)
        assert res.value.n_analyzed == 30
        assert 20 < res.value.results["mean_mmse"] < 30

    def test_rejects_dangerous_import(self):
        engine = self._make_engine_with_df(30)
        code = "import os\nresult = {'n_analyzed': 30}"
        with pytest.raises(ValueError, match="import"):
            engine.run_generated_code(
                code=code,
                filter_spec=FILTER_A,
                required_cols=[],
                analysis_description="dangerous",
            )

    def test_abstains_when_no_result_variable(self):
        engine = self._make_engine_with_df(30)
        code = "x = 1  # forgot to set result"
        res = engine.run_generated_code(
            code=code,
            filter_spec=FILTER_A,
            required_cols=[],
            analysis_description="forgot result",
        )
        assert not res.grounded
        assert "result" in (res.abstain_reason or "").lower()

    def test_abstains_when_result_contains_list(self):
        engine = self._make_engine_with_df(30)
        code = "result = {'n_analyzed': 30, 'bad': [1, 2, 3]}"
        res = engine.run_generated_code(
            code=code,
            filter_spec=FILTER_A,
            required_cols=[],
            analysis_description="list result",
        )
        assert not res.grounded
        assert "non-aggregate" in (res.abstain_reason or "").lower()

    def test_abstains_on_space_b(self):
        engine = self._make_engine_with_df(5)
        res = engine.run_generated_code(
            code="result = {'n_analyzed': 5}",
            filter_spec=FILTER_B,
            required_cols=[],
            analysis_description="space B",
        )
        assert not res.grounded
        assert "Space A" in (res.abstain_reason or "")

    def test_evidence_has_aggregate_ref(self):
        engine = self._make_engine_with_df(30)
        code = (
            "vals = df['MMSE'].dropna()\n"
            "result = {'n_analyzed': int(len(vals)), 'mean_mmse': round(float(vals.mean()), 3)}"
        )
        res = engine.run_generated_code(
            code=code,
            filter_spec=FILTER_A,
            required_cols=["MMSE"],
            analysis_description="mean MMSE",
        )
        assert res.grounded
        from alz_agent.schemas import DataEvidence
        assert any(isinstance(ev, DataEvidence) for ev in res.evidence)


# ---------------------------------------------------------------------------
# Tests: _run_ols and _run_logistic (pure regression helpers)
# ---------------------------------------------------------------------------


class TestRunOls:
    def _linear_df(self, n: int = 50, seed: int = 0) -> pd.DataFrame:
        import random
        rng = random.Random(seed)
        x = [rng.gauss(5, 2) for _ in range(n)]
        y = [2.0 + 1.5 * xi + rng.gauss(0, 0.5) for xi in x]
        return pd.DataFrame({"y": y, "x": x})

    def test_returns_intercept_and_slope(self):
        from alz_agent.analysis.engine import _run_ols
        df = self._linear_df(100, seed=1)
        coef, r2, _ = _run_ols(df, "y", ["x"])
        # Intercept ~2.0, slope ~1.5
        assert abs(coef["coef"][0] - 2.0) < 0.3, f"intercept {coef['coef'][0]}"
        assert abs(coef["coef"][1] - 1.5) < 0.2, f"slope {coef['coef'][1]}"

    def test_r_squared_high_for_linear_data(self):
        from alz_agent.analysis.engine import _run_ols
        df = self._linear_df(100, seed=2)
        _, r2, _ = _run_ols(df, "y", ["x"])
        assert r2 is not None and r2 > 0.9

    def test_se_and_pvalues_present(self):
        from alz_agent.analysis.engine import _run_ols
        df = self._linear_df(50, seed=3)
        coef, _, _ = _run_ols(df, "y", ["x"])
        assert len(coef["se"]) == 2
        assert all(v >= 0 for v in coef["se"])
        assert len(coef["pval"]) == 2
        assert coef["pval"][1] < 0.05, "slope should be significant"

    def test_ci_bounds_bracket_true_slope(self):
        from alz_agent.analysis.engine import _run_ols
        df = self._linear_df(100, seed=4)
        coef, _, _ = _run_ols(df, "y", ["x"])
        lo, hi = coef["ci_lower"][1], coef["ci_upper"][1]
        assert lo < 1.5 < hi, f"CI [{lo}, {hi}] should contain true slope 1.5"

    def test_multiple_predictors(self):
        from alz_agent.analysis.engine import _run_ols
        import random
        rng = random.Random(5)
        n = 80
        x1 = [rng.gauss(3, 1) for _ in range(n)]
        x2 = [rng.gauss(1, 0.5) for _ in range(n)]
        y = [1.0 + 2.0 * a + 0.5 * b + rng.gauss(0, 0.3) for a, b in zip(x1, x2)]
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        coef, r2, _ = _run_ols(df, "y", ["x1", "x2"])
        assert len(coef["coef"]) == 3  # intercept + 2 predictors
        assert r2 is not None and r2 > 0.9


class TestRunLogistic:
    def _binary_df(self, n: int = 200, seed: int = 10) -> pd.DataFrame:
        import random
        rng = random.Random(seed)
        x = [rng.gauss(0, 1) for _ in range(n)]
        # P(y=1) = sigmoid(0.5 + 1.5*x)
        import math
        y = [1 if rng.random() < 1 / (1 + math.exp(-(0.5 + 1.5 * xi))) else 0 for xi in x]
        return pd.DataFrame({"y": y, "x": x})

    def test_converges_and_returns_coefficients(self):
        from alz_agent.analysis.engine import _run_logistic
        df = self._binary_df(200, seed=10)
        coef, _, pseudo_r2 = _run_logistic(df, "y", ["x"])
        assert len(coef["coef"]) == 2
        # True intercept ~0.5, slope ~1.5; tolerance loose due to MLE variance
        assert abs(coef["coef"][0] - 0.5) < 1.0
        assert coef["coef"][1] > 0.5  # positive slope direction

    def test_pseudo_r2_between_0_and_1(self):
        from alz_agent.analysis.engine import _run_logistic
        df = self._binary_df(200, seed=11)
        _, _, pseudo_r2 = _run_logistic(df, "y", ["x"])
        assert pseudo_r2 is not None and 0.0 < pseudo_r2 < 1.0

    def test_pvalue_significant_for_strong_predictor(self):
        from alz_agent.analysis.engine import _run_logistic
        df = self._binary_df(300, seed=12)
        coef, _, _ = _run_logistic(df, "y", ["x"])
        # p-value for slope should be < 0.05 with n=300 and true beta=1.5
        assert coef["pval"][1] < 0.05, f"pval {coef['pval'][1]}"


# ---------------------------------------------------------------------------
# Tests: LocalAnalysisEngine.adjusted_regression
# ---------------------------------------------------------------------------


def _make_regression_engine(n: int = 60, seed: int = 99) -> Any:
    """Return LocalAnalysisEngine with a synthetic linear dataset loaded."""
    import random
    rng = random.Random(seed)
    ids = [f"R{i:04d}" for i in range(n)]
    age = [rng.gauss(72, 8) for _ in range(n)]
    ha = [rng.choice([0, 1]) for _ in range(n)]
    # MMSE = 28 - 0.05*age - 1.5*ha + noise
    mmse = [28 - 0.05 * a - 1.5 * h + rng.gauss(0, 1) for a, h in zip(age, ha)]
    df_data = {
        "NACCID": ids,
        "NACCVNUM": [1] * n,
        "NACCMMSE": mmse,
        "NACCAGE": age,
        "CVHATT": ha,
    }
    return _make_engine(df_data)


FILTER_A_PLAIN = FilterSpec(
    id_space=IdSpace.A,
    baseline_diagnosis=None,
    baseline_etiology=None,
    visit_window=None,
)


class TestEngineAdjustedRegression:
    def test_linear_regression_grounded(self):
        engine = _make_regression_engine(60)
        res = engine.adjusted_regression(
            outcome_col="NACCMMSE",
            outcome_variable_id="NACC.NACCMMSE",
            outcome_label="MMSE score",
            covariate_cols=["NACCAGE", "CVHATT"],
            covariate_variable_ids=["NACC.NACCAGE", "NACC.CVHATT"],
            filter_spec=FILTER_A_PLAIN,
            regression_type="linear",
        )
        assert res.grounded, res.abstain_reason
        from alz_agent.schemas import RegressionResult
        assert isinstance(res.value, RegressionResult)
        assert res.value.regression_type == "linear"
        assert res.value.n == 60
        assert len(res.value.coefficients) == 3  # intercept + 2 covariates

    def test_returns_negative_ha_coefficient(self):
        """Heart attack (CVHATT) should have a negative coefficient on MMSE."""
        engine = _make_regression_engine(60)
        res = engine.adjusted_regression(
            outcome_col="NACCMMSE",
            outcome_variable_id="NACC.NACCMMSE",
            outcome_label="MMSE score",
            covariate_cols=["CVHATT"],
            covariate_variable_ids=["NACC.CVHATT"],
            filter_spec=FILTER_A_PLAIN,
            regression_type="linear",
        )
        assert res.grounded
        ha_coef = next(c for c in res.value.coefficients if c.variable == "NACC.CVHATT")
        assert ha_coef.coefficient < 0, f"Expected negative, got {ha_coef.coefficient}"

    def test_r_squared_is_populated(self):
        engine = _make_regression_engine(60)
        res = engine.adjusted_regression(
            outcome_col="NACCMMSE",
            outcome_variable_id="NACC.NACCMMSE",
            outcome_label="MMSE score",
            covariate_cols=["NACCAGE"],
            covariate_variable_ids=["NACC.NACCAGE"],
            filter_spec=FILTER_A_PLAIN,
            regression_type="linear",
        )
        assert res.grounded
        assert res.value.r_squared is not None

    def test_logistic_regression_grounded(self):
        import math
        import random
        rng = random.Random(77)
        n = 60
        ids = [f"L{i:04d}" for i in range(n)]
        age = [rng.gauss(72, 8) for _ in range(n)]
        # Binary outcome 0/1: probability of conversion increases with age
        ad = [1 if rng.random() < 1 / (1 + math.exp(-(0.01 * a - 0.7))) else 0 for a in age]
        df_data = {
            "NACCID": ids,
            "NACCVNUM": [1] * n,
            "AD_BINARY": ad,  # 0 or 1 — correct binary outcome for logistic
            "NACCAGE": age,
        }
        engine = _make_engine(df_data)
        res = engine.adjusted_regression(
            outcome_col="AD_BINARY",
            outcome_variable_id="NACC.AD_BINARY",
            outcome_label="dementia conversion",
            covariate_cols=["NACCAGE"],
            covariate_variable_ids=["NACC.NACCAGE"],
            filter_spec=FILTER_A_PLAIN,
            regression_type="logistic",
        )
        assert res.grounded, res.abstain_reason
        assert res.value.regression_type == "logistic"
        assert res.value.pseudo_r2 is not None

    def test_space_b_abstains(self):
        engine = _make_regression_engine(60)
        res = engine.adjusted_regression(
            outcome_col="NACCMMSE",
            outcome_variable_id="NACC.NACCMMSE",
            outcome_label="MMSE score",
            covariate_cols=["NACCAGE"],
            covariate_variable_ids=["NACC.NACCAGE"],
            filter_spec=FILTER_B,
            regression_type="linear",
        )
        assert not res.grounded
        assert "Space A" in (res.abstain_reason or "")

    def test_missing_column_abstains(self):
        engine = _make_regression_engine(60)
        res = engine.adjusted_regression(
            outcome_col="NONEXISTENT_COL",
            outcome_variable_id="NACC.NONEXISTENT",
            outcome_label="ghost",
            covariate_cols=["NACCAGE"],
            covariate_variable_ids=["NACC.NACCAGE"],
            filter_spec=FILTER_A_PLAIN,
            regression_type="linear",
        )
        assert not res.grounded

    def test_evidence_has_aggregate_ref(self):
        engine = _make_regression_engine(60)
        from alz_agent.schemas import DataEvidence
        res = engine.adjusted_regression(
            outcome_col="NACCMMSE",
            outcome_variable_id="NACC.NACCMMSE",
            outcome_label="MMSE score",
            covariate_cols=["NACCAGE"],
            covariate_variable_ids=["NACC.NACCAGE"],
            filter_spec=FILTER_A_PLAIN,
            regression_type="linear",
        )
        assert res.grounded
        assert any(isinstance(ev, DataEvidence) for ev in res.evidence)
