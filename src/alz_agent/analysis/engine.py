"""Analysis engine (P3+ analysis layer) — executes statistical analyses on local data.

Two execution paths:
  1. Pre-built templates (group_comparison, descriptive_stats) — always deterministic.
  2. LLM-generated code sandbox — for novel analyses not covered by templates.
     The LLM generates Python code; the engine executes it in a restricted namespace;
     only scalar aggregates may be returned.

Invariant (AGENTS.md §4): subject-level data NEVER leaves this module.
All public methods return ToolResult with aggregate-only values.
"""

from __future__ import annotations

import builtins
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from alz_agent.agent.providers import PlannerClient
from alz_agent.constants import SUPPRESSED, SUPPRESSION_THRESHOLD, suppress
from alz_agent.schemas import (
    DataEvidence,
    FilterSpec,
    GeneratedAnalysisResult,
    GroupComparisonResult,
    GroupStats,
    IdSpace,
    RegressionCoef,
    RegressionResult,
    ToolResult,
)

if TYPE_CHECKING:
    from alz_agent.backend.local_backend import LocalDataBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sandbox safety configuration
# ---------------------------------------------------------------------------

_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "frozenset", "hasattr", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "None", "print", "range", "repr", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "type", "zip",
    "True", "False", "ValueError", "KeyError", "RuntimeError",
})

_SAFE_BUILTINS: dict[str, Any] = {
    k: getattr(builtins, k)
    for k in _SAFE_BUILTIN_NAMES
    if hasattr(builtins, k)
}


# ---------------------------------------------------------------------------
# Pure analysis functions (testable without backend)
# ---------------------------------------------------------------------------


def _binary_groups(
    df: pd.DataFrame,
    group_col: str,
    positive_values: list[Any] | None = None,
) -> dict[str, pd.Series]:
    """Split a DataFrame mask series into two binary groups.

    If positive_values is None, uses (value != 0) as the positive group
    (works for most NACC history variables where 0 = Absent, 1/2 = Present).

    Returns {"positive": bool_mask, "negative": bool_mask}.
    """
    col = pd.to_numeric(df[group_col], errors="coerce")
    if positive_values is not None:
        pos_mask = col.isin(positive_values)
    else:
        pos_mask = col != 0
    neg_mask = col == 0
    # Exclude rows where group_col is NaN
    valid = col.notna()
    return {"positive": pos_mask & valid, "negative": neg_mask & valid}


def group_comparison_impl(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    group_names: dict[str, str],  # {"positive": "heart attack", "negative": "no heart attack"}
    positive_values: list[Any] | None = None,
) -> dict[str, Any]:
    """Core group-comparison computation — pure function, no I/O.

    Args:
        df:              One row per participant DataFrame (pre-filtered, pre-deduplicated).
        outcome_col:     Column name of the continuous outcome.
        group_col:       Column name of the binary grouping variable.
        group_names:     Human-readable labels for each side of the binary split.
        positive_values: Values in group_col treated as "positive" group.
                         None → any nonzero value (common NACC history pattern).

    Returns a dict with keys:
        positive_n, positive_mean, positive_sd,
        negative_n, negative_mean, negative_sd,
        p_value, test_name, n_total, n_excluded_na

    Raises ValueError when input is empty or outcome_col/group_col missing.
    """
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not in DataFrame")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not in DataFrame")

    outcome = pd.to_numeric(df[outcome_col], errors="coerce")
    masks = _binary_groups(df, group_col, positive_values)

    results: dict[str, Any] = {}
    group_data: dict[str, pd.Series] = {}

    for side in ("positive", "negative"):
        mask = masks[side]
        vals = outcome[mask].dropna()
        group_data[side] = vals
        label = group_names.get(side, side)
        results[f"{side}_n"] = int(len(vals))
        results[f"{side}_mean"] = round(float(vals.mean()), 3) if len(vals) > 0 else float("nan")
        results[f"{side}_sd"] = round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else float("nan")
        results[f"{side}_label"] = label

    # Welch two-sample t-test (unequal variances assumed)
    pos_vals = group_data["positive"]
    neg_vals = group_data["negative"]
    if len(pos_vals) >= 2 and len(neg_vals) >= 2:
        t_stat, p_val = scipy_stats.ttest_ind(pos_vals, neg_vals, equal_var=False)
        results["p_value"] = round(float(p_val), 4)
        results["t_statistic"] = round(float(t_stat), 3)
        results["test_name"] = "welch_t_test"
    else:
        results["p_value"] = None
        results["t_statistic"] = None
        results["test_name"] = "welch_t_test"
        results["test_skipped_reason"] = "insufficient observations in one group (n<2)"

    results["n_total"] = int(len(pos_vals) + len(neg_vals))
    # Participants excluded due to NA in either column
    n_na = int(len(df) - results["n_total"] - int(
        (~masks["positive"] & ~masks["negative"] & df[group_col].notna()).sum()
    ))
    results["n_excluded_na"] = max(0, len(df) - results["n_total"])
    return results


def validate_aggregate_result(result: Any) -> None:
    """Raise ValueError if result contains non-scalar or subject-level data.

    A 'scalar' is: int, float, str, bool, or None.
    Nested dicts of scalars are allowed.
    Any list, DataFrame, or other iterable is rejected.
    """
    if not isinstance(result, dict):
        raise ValueError(f"Analysis result must be a dict, got {type(result).__name__}")
    for k, v in result.items():
        if isinstance(v, dict):
            validate_aggregate_result(v)
        elif not isinstance(v, (int, float, str, bool, type(None))):
            raise ValueError(
                f"Non-scalar value in result['{k}']: {type(v).__name__}. "
                "Analysis code must return only aggregate scalars — never lists, "
                "DataFrames, or other row-level data."
            )


# ---------------------------------------------------------------------------
# Pure regression helpers (testable without backend)
# ---------------------------------------------------------------------------


def _run_ols(
    df: pd.DataFrame,
    outcome_col: str,
    covariate_cols: list[str],
) -> tuple[dict[str, Any], float | None, None]:
    """Ordinary least-squares regression via numpy lstsq.

    Returns (coef_dict, r_squared, None).
    coef_dict has keys 'coef', 'se', 'pval', 'ci_lower', 'ci_upper' — each a list
    of length len(covariate_cols)+1 (intercept first).
    """
    y = df[outcome_col].values.astype(float)
    X_cols = df[covariate_cols].values.astype(float)
    n = len(y)
    k = X_cols.shape[1] + 1  # +1 for intercept
    X = np.column_stack([np.ones(n), X_cols])

    # Solve via least squares
    betas, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    # R-squared
    ss_res = float(np.sum((y - X @ betas) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_sq: float | None = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    # Standard errors and t-stats via (X'X)^-1 * MSE
    dof = n - k
    if dof > 0:
        mse = ss_res / dof
        XtX_inv = np.linalg.pinv(X.T @ X)
        var_betas = mse * np.diag(XtX_inv)
        se = np.sqrt(np.maximum(var_betas, 0.0))
        t_stats = betas / np.where(se > 0, se, np.nan)
        pvals = [
            float(2 * scipy_stats.t.sf(abs(t), df=dof)) if not np.isnan(t) else float("nan")
            for t in t_stats
        ]
        t_crit = scipy_stats.t.ppf(0.975, df=dof)
        ci_lower = list(betas - t_crit * se)
        ci_upper = list(betas + t_crit * se)
    else:
        se = np.full(k, float("nan"))
        pvals = [float("nan")] * k
        ci_lower = [float("nan")] * k
        ci_upper = [float("nan")] * k

    coef_dict: dict[str, Any] = {
        "coef": list(betas),
        "se": list(se),
        "pval": pvals,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
    return coef_dict, r_sq, None


def _run_logistic(
    df: pd.DataFrame,
    outcome_col: str,
    covariate_cols: list[str],
) -> tuple[dict[str, Any], None, float | None]:
    """Logistic regression via scipy.optimize.minimize (L-BFGS-B).

    Returns (coef_dict, None, pseudo_r2).
    Uses McFadden pseudo-R² = 1 - LL_model / LL_null.
    """
    from scipy.optimize import minimize

    y = df[outcome_col].values.astype(float)
    X_cols = df[covariate_cols].values.astype(float)
    n = len(y)
    k = X_cols.shape[1] + 1
    X = np.column_stack([np.ones(n), X_cols])

    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _neg_log_lik(betas: np.ndarray) -> float:
        p = _sigmoid(X @ betas)
        eps = 1e-12
        return -float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def _grad(betas: np.ndarray) -> np.ndarray:
        p = _sigmoid(X @ betas)
        return -X.T @ (y - p)

    b0 = np.zeros(k)
    res = minimize(_neg_log_lik, b0, jac=_grad, method="L-BFGS-B",
                   options={"maxiter": 1000, "ftol": 1e-9})
    betas = res.x
    ll_model = -res.fun

    # Null log-likelihood (intercept only)
    p_null = float(y.mean())
    eps = 1e-12
    ll_null = n * (p_null * np.log(p_null + eps) + (1 - p_null) * np.log(1 - p_null + eps))
    pseudo_r2: float | None = float(1.0 - ll_model / ll_null) if abs(ll_null) > eps else None

    # Standard errors from observed information: H = X^T diag(p*(1-p)) X
    try:
        p_fit = _sigmoid(X @ betas)
        W = p_fit * (1.0 - p_fit)
        XtWX = X.T @ (W[:, None] * X)
        var_betas = np.diag(np.linalg.pinv(XtWX))
        se = np.sqrt(np.maximum(var_betas, 0.0))
    except Exception:
        se = np.full(k, float("nan"))

    z_stats = betas / np.where(se > 0, se, np.nan)
    pvals = [
        float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else float("nan")
        for z in z_stats
    ]
    z_crit = scipy_stats.norm.ppf(0.975)
    ci_lower = list(betas - z_crit * se)
    ci_upper = list(betas + z_crit * se)

    coef_dict: dict[str, Any] = {
        "coef": list(betas),
        "se": list(se),
        "pval": pvals,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
    return coef_dict, None, pseudo_r2


# ---------------------------------------------------------------------------
# LocalAnalysisEngine
# ---------------------------------------------------------------------------


class LocalAnalysisEngine:
    """Executes statistical analyses on local NACC/ADNI data.

    Reads data files using the same config paths as LocalDataBackend, but
    performs richer per-column computations for statistical analysis questions.

    Invariant: all public methods return ToolResult with scalar aggregates only.
    """

    def __init__(self, backend: "LocalDataBackend") -> None:
        self._backend = backend
        self._config = backend.config
        self._root = backend.project_root

    # ── Internal data loading ─────────────────────────────────────────────────

    def _load_nacc_baseline(
        self,
        eligible_ids: frozenset[str],
        extra_cols: list[str],
    ) -> pd.DataFrame:
        """Return one baseline row per eligible participant with requested columns.

        Baseline = row with minimum NACCVNUM per NACCID.
        Only eligible_ids are included; rows with no valid NACCVNUM are dropped.
        """
        cols = list(dict.fromkeys(["NACCID", "NACCVNUM"] + extra_cols))
        rel = self._config["nacc"]["data_csv"]
        full = self._root / rel
        if not full.exists():
            raise FileNotFoundError(f"NACC data file not found: {full}")

        # Read only needed columns; low_memory=False for mixed-type safety
        df = pd.read_csv(full, usecols=cols, low_memory=False, dtype={"NACCID": str})
        df = df[df["NACCID"].isin(eligible_ids)]
        df["NACCVNUM"] = pd.to_numeric(df["NACCVNUM"], errors="coerce")
        df = df.dropna(subset=["NACCVNUM"])
        # One row per participant at minimum NACCVNUM (baseline visit)
        return df.sort_values("NACCVNUM").drop_duplicates("NACCID", keep="first")

    # ── Pre-built template: group comparison ──────────────────────────────────

    def group_comparison(
        self,
        outcome_col: str,
        outcome_variable_id: str,
        outcome_label: str,
        group_col: str,
        group_variable_id: str,
        group_label: str,
        group_names: dict[str, str],
        filter_spec: FilterSpec,
        positive_values: list[Any] | None = None,
    ) -> ToolResult[GroupComparisonResult]:
        """Compare outcome_col across two groups defined by group_col.

        Uses one baseline row per participant. Applies suppression on each group's
        n separately: if either group has n < SUPPRESSION_THRESHOLD, abstains.

        Args:
            outcome_col:         Raw NACC column name for the continuous outcome.
            outcome_variable_id: Catalog variable_id (for evidence attribution).
            outcome_label:       Human-readable outcome name.
            group_col:           Raw NACC column name for the binary grouping variable.
            group_variable_id:   Catalog variable_id for the grouping variable.
            group_label:         Human-readable name of the grouping variable.
            group_names:         {"positive": "heart attack", "negative": "no heart attack"}
            filter_spec:         Cohort filter spec.
            positive_values:     Values in group_col treated as "positive" group.
        """
        if filter_spec.id_space != IdSpace.A:
            return ToolResult.abstain(
                "group_comparison analysis is currently only implemented for Space A (NACC)"
            )

        try:
            eligible = self._backend.eligible_ids(filter_spec)
        except (ValueError, RuntimeError) as exc:
            return ToolResult.abstain(str(exc))

        if not eligible:
            return ToolResult.abstain("No eligible participants for the given filter")

        try:
            df = self._load_nacc_baseline(eligible, [outcome_col, group_col])
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return ToolResult.abstain(f"Could not load analysis columns: {exc}")

        missing = [c for c in (outcome_col, group_col) if c not in df.columns]
        if missing:
            return ToolResult.abstain(
                f"Column(s) not found in NACC data: {missing}. "
                "Check raw_code in the catalog entry."
            )

        try:
            raw = group_comparison_impl(df, outcome_col, group_col, group_names, positive_values)
        except ValueError as exc:
            return ToolResult.abstain(str(exc))

        # Suppression check per group
        for side in ("positive", "negative"):
            n = raw.get(f"{side}_n", 0)
            s = suppress(n)
            if s is SUPPRESSED:
                return ToolResult.abstain(
                    f"Group '{group_names.get(side, side)}' count ({n}) is below "
                    f"suppression threshold ({SUPPRESSION_THRESHOLD}). Cannot report."
                )

        pos_n = int(raw["positive_n"])
        neg_n = int(raw["negative_n"])

        groups = [
            GroupStats(
                group_label=group_names.get("positive", "positive"),
                group_values=positive_values or ["≠0"],
                n=pos_n,
                mean=raw["positive_mean"],
                sd=raw["positive_sd"],
            ),
            GroupStats(
                group_label=group_names.get("negative", "negative"),
                group_values=[0],
                n=neg_n,
                mean=raw["negative_mean"],
                sd=raw["negative_sd"],
            ),
        ]

        caveats = [
            "Unadjusted comparison — confounders (age, sex, education, comorbidities) not controlled.",
            "Cross-sectional baseline snapshot; does not establish temporal precedence.",
        ]
        if raw.get("n_excluded_na", 0) > 0:
            caveats.append(
                f"{raw['n_excluded_na']} participant(s) excluded due to missing values "
                f"in '{outcome_col}' or '{group_col}'."
            )

        cohort_def = _build_cohort_def(filter_spec)
        n_total = int(raw["n_total"])
        p_value = raw.get("p_value")

        result = GroupComparisonResult(
            outcome_variable_id=outcome_variable_id,
            outcome_label=outcome_label,
            group_variable_id=group_variable_id,
            group_label=group_label,
            groups=groups,
            p_value=float(p_value) if p_value is not None else None,
            test_name=raw.get("test_name", "welch_t_test"),
            cohort_def=cohort_def,
            caveats=caveats,
            n_total=n_total,
        )

        # Build evidence aggregate_ref with all numbers for hallucination guard
        n_excluded = raw.get("n_excluded_na", 0)
        ev_nums = (
            f"group_comparison({outcome_col} ~ {group_col}): "
            f"{group_names.get('positive','pos')} n={pos_n} mean={raw['positive_mean']} sd={raw['positive_sd']} | "
            f"{group_names.get('negative','neg')} n={neg_n} mean={raw['negative_mean']} sd={raw['negative_sd']} | "
            f"p={p_value} n_total={n_total} n_excluded_na={n_excluded}"
        )
        return ToolResult.ok(
            result,
            evidence=[DataEvidence(
                query_spec={
                    **filter_spec.model_dump(mode="json"),
                    "outcome_col": outcome_col,
                    "group_col": group_col,
                    "analysis": "group_comparison",
                },
                aggregate_ref=ev_nums,
            )],
        )

    # ── Pre-built template: adjusted regression ───────────────────────────────

    def adjusted_regression(
        self,
        outcome_col: str,
        outcome_variable_id: str,
        outcome_label: str,
        covariate_cols: list[str],
        covariate_variable_ids: list[str],
        filter_spec: FilterSpec,
        regression_type: str = "linear",
    ) -> ToolResult[RegressionResult]:
        """Run OLS linear or logistic regression, returning aggregate coefficients only.

        Uses scipy/numpy for linear regression (no external ML library required).
        Logistic regression uses scipy optimize for maximum likelihood estimation.

        Args:
            outcome_col:           Raw NACC column name for the outcome variable.
            outcome_variable_id:   Catalog variable_id for the outcome.
            outcome_label:         Human-readable outcome name.
            covariate_cols:        Raw NACC column names for predictors (in order).
            covariate_variable_ids: Catalog variable_ids for predictors (parallel to covariate_cols).
            filter_spec:           Cohort filter.
            regression_type:       'linear' (OLS) or 'logistic' (MLE).

        Returns aggregate RegressionResult. Abstains if data is insufficient.
        """
        if filter_spec.id_space != IdSpace.A:
            return ToolResult.abstain(
                "adjusted_regression is currently only implemented for Space A (NACC)"
            )
        if regression_type not in ("linear", "logistic"):
            return ToolResult.abstain(
                f"Unknown regression_type '{regression_type}'. Valid: 'linear', 'logistic'."
            )

        try:
            eligible = self._backend.eligible_ids(filter_spec)
        except (ValueError, RuntimeError) as exc:
            return ToolResult.abstain(str(exc))
        if not eligible:
            return ToolResult.abstain("No eligible participants for the given filter")

        all_cols = [outcome_col] + covariate_cols
        try:
            df = self._load_nacc_baseline(eligible, all_cols)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return ToolResult.abstain(f"Could not load analysis columns: {exc}")

        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            return ToolResult.abstain(
                f"Column(s) not found in NACC data: {missing}"
            )

        # Drop rows with any missing value in outcome or covariates
        df_clean = df[all_cols].dropna()
        n = int(len(df_clean))
        n_dropped = int(len(df) - n)

        if n < max(SUPPRESSION_THRESHOLD, len(all_cols) + 5):
            return ToolResult.abstain(
                f"Too few observations after dropping missing values (n={n}) for regression "
                f"with {len(covariate_cols)} predictor(s). Need at least "
                f"{max(SUPPRESSION_THRESHOLD, len(all_cols) + 5)}."
            )

        for c in all_cols:
            df_clean = df_clean.copy()
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
        df_clean = df_clean.dropna()
        n = int(len(df_clean))

        try:
            if regression_type == "linear":
                coefficients, r_sq, pseudo_r2 = _run_ols(df_clean, outcome_col, covariate_cols)
            else:
                coefficients, r_sq, pseudo_r2 = _run_logistic(df_clean, outcome_col, covariate_cols)
        except Exception as exc:
            return ToolResult.abstain(f"Regression computation failed: {exc}")

        # Build coefficient objects aligned with covariate_variable_ids
        coef_names = ["(Intercept)"] + covariate_cols
        coef_ids = ["intercept"] + covariate_variable_ids
        def _safe_round(val: Any) -> float | None:
            try:
                v = float(val)
                return round(v, 4) if not np.isnan(v) else None
            except (TypeError, ValueError):
                return None

        coef_objects = [
            RegressionCoef(
                variable=coef_ids[i] if i < len(coef_ids) else coef_names[i],
                coefficient=round(float(coefficients["coef"][i]), 4),
                std_error=_safe_round(coefficients["se"][i]),
                p_value=_safe_round(coefficients["pval"][i]),
                ci_lower=_safe_round(coefficients["ci_lower"][i]),
                ci_upper=_safe_round(coefficients["ci_upper"][i]),
            )
            for i in range(len(coef_names))
        ]

        caveats = [
            f"Unadjusted for variables not included as covariates.",
            "Cross-sectional baseline — cannot infer causality.",
        ]
        if n_dropped > 0:
            caveats.append(f"{n_dropped} row(s) dropped due to missing values in outcome or covariates.")

        cohort_def = _build_cohort_def(filter_spec)
        result = RegressionResult(
            regression_type=regression_type,
            outcome_variable_id=outcome_variable_id,
            outcome_label=outcome_label,
            n=n,
            r_squared=round(float(r_sq), 4) if r_sq is not None else None,
            pseudo_r2=round(float(pseudo_r2), 4) if pseudo_r2 is not None else None,
            coefficients=coef_objects,
            cohort_def=cohort_def,
            caveats=caveats,
        )

        ev_ref = (
            f"regression({outcome_col} ~ {'+'.join(covariate_cols)},"
            f"{regression_type}): n={n} r2={r_sq} "
            + " ".join(
                f"coef[{c.variable}]={c.coefficient}"
                for c in coef_objects
            )
        )
        return ToolResult.ok(
            result,
            evidence=[DataEvidence(
                query_spec={
                    **filter_spec.model_dump(mode="json"),
                    "outcome_col": outcome_col,
                    "covariates": covariate_cols,
                    "regression_type": regression_type,
                },
                aggregate_ref=ev_ref,
            )],
        )

    # ── LLM code generation sandbox ───────────────────────────────────────────

    def run_generated_code(
        self,
        code: str,
        filter_spec: FilterSpec,
        required_cols: list[str],
        analysis_description: str,
    ) -> ToolResult[GeneratedAnalysisResult]:
        """Execute LLM-generated analysis code in a restricted sandbox.

        The generated code receives a pre-filtered, baseline-deduplicated DataFrame
        and must store its results in a variable called ``result`` (a dict of scalars).

        Args:
            code:                 Python code string defining the analysis.
            filter_spec:          Cohort filter; used to obtain eligible participants.
            required_cols:        NACC column names the code will access in the DataFrame.
            analysis_description: Human-readable description of the analysis being run.
        """
        if filter_spec.id_space != IdSpace.A:
            return ToolResult.abstain(
                "LLM-generated analysis is currently only implemented for Space A (NACC)"
            )

        # Security pre-check: reject obviously dangerous patterns
        _security_check(code)

        try:
            eligible = self._backend.eligible_ids(filter_spec)
        except (ValueError, RuntimeError) as exc:
            return ToolResult.abstain(str(exc))

        if not eligible:
            return ToolResult.abstain("No eligible participants for the given filter")

        try:
            df = self._load_nacc_baseline(eligible, required_cols)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return ToolResult.abstain(f"Could not load analysis columns: {exc}")

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return ToolResult.abstain(
                f"Column(s) not found in NACC data: {missing}"
            )

        namespace: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "stats": scipy_stats,
            "SUPPRESSION_THRESHOLD": SUPPRESSION_THRESHOLD,
        }

        try:
            exec(compile(code, "<llm_analysis>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            return ToolResult.abstain(f"Analysis code execution failed: {exc}")

        if "result" not in namespace:
            return ToolResult.abstain(
                "Generated analysis code must assign its output to a variable named 'result'."
            )

        raw_result = namespace["result"]
        try:
            validate_aggregate_result(raw_result)
        except ValueError as exc:
            return ToolResult.abstain(
                f"Analysis code returned non-aggregate data (invariant violation): {exc}"
            )

        n_analyzed = int(raw_result.get("n_analyzed", len(df)))
        s = suppress(n_analyzed)
        if s is SUPPRESSED:
            return ToolResult.abstain(
                f"Analyzed cohort ({n_analyzed}) below suppression threshold ({SUPPRESSION_THRESHOLD})"
            )

        cohort_def = _build_cohort_def(filter_spec)
        gen_result = GeneratedAnalysisResult(
            analysis_description=analysis_description,
            results=raw_result,
            n_analyzed=n_analyzed,
            cohort_def=cohort_def,
            code_used=code,
        )

        # Aggregate ref captures all numeric values for hallucination guard
        ev_ref = f"generated_analysis({analysis_description}): " + " ".join(
            f"{k}={v}" for k, v in raw_result.items()
            if isinstance(v, (int, float)) and v is not None
        )
        return ToolResult.ok(
            gen_result,
            evidence=[DataEvidence(
                query_spec={
                    **filter_spec.model_dump(mode="json"),
                    "analysis": analysis_description,
                    "required_cols": required_cols,
                },
                aggregate_ref=ev_ref,
            )],
        )


# ---------------------------------------------------------------------------
# LLM code generation
# ---------------------------------------------------------------------------


def generate_analysis_code(
    nl_question: str,
    intent_summary: str,
    col_schema: str,
    client: Any,
    model: str | None = None,
) -> tuple[str, list[str], str]:
    """Ask the LLM to generate a Python analysis function for a novel question.

    Returns:
        (code_str, required_cols, analysis_description)

    The generated code must:
    - Receive a DataFrame `df` (one baseline row per participant, columns per required_cols)
    - Store results in `result: dict` (only scalar values)
    - Include `result['n_analyzed'] = <int>` for suppression check
    """
    system = (
        "You are generating a Python analysis script for a local Alzheimer's dataset. "
        "Rules:\n"
        "1. You receive a pandas DataFrame `df` with one baseline row per eligible participant.\n"
        "2. Allowed imports are already in scope: pd, np, stats (scipy.stats), SUPPRESSION_THRESHOLD.\n"
        "3. Store ALL results in a dict called `result`. Include `result['n_analyzed']` (int).\n"
        "4. Return ONLY aggregate scalars (int, float, str) — no lists, no DataFrames, no subject IDs.\n"
        "5. Keep the code concise and correct. Prefer Welch t-test (stats.ttest_ind equal_var=False) "
        "for group comparisons.\n"
        "6. Output ONLY a JSON object with three keys:\n"
        '   "code": the complete Python snippet (string),\n'
        '   "required_cols": list of NACC column names the code references,\n'
        '   "description": one-sentence description of the analysis.'
    )
    user = (
        f"Research question: {nl_question}\n\n"
        f"Parsed intent:\n{intent_summary}\n\n"
        f"Available columns (variable_id | raw_code | label | value_coding):\n{col_schema}\n\n"
        "Generate the analysis code JSON:"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if isinstance(client, PlannerClient):
        raw = client.complete_text(messages, max_tokens=1024).strip()
    else:
        response = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_block = next((b for b in response.content if b.type == "text"), None)
        if text_block is None:
            raise RuntimeError("LLM returned no text for analysis code generation")
        raw = text_block.text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        raw = "\n".join(inner[:-1] if inner and inner[-1].strip() == "```" else inner)

    import json
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned invalid JSON for analysis code: {exc}\nRaw: {raw[:400]}"
        ) from exc

    code = str(parsed.get("code", ""))
    required_cols = list(parsed.get("required_cols", []))
    description = str(parsed.get("description", "LLM-generated analysis"))

    _security_check(code)
    return code, required_cols, description


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cohort_def(filter_spec: FilterSpec) -> str:
    parts: list[str] = []
    if filter_spec.baseline_diagnosis:
        parts.append(f"baseline_diagnosis={filter_spec.baseline_diagnosis}")
    if filter_spec.baseline_etiology:
        parts.append(f"baseline_etiology={filter_spec.baseline_etiology}")
    if filter_spec.visit_window:
        parts.append(f"visit_window={filter_spec.visit_window}")
    parts.append(f"id_space={filter_spec.id_space.value}")
    return "; ".join(parts) if parts else "all eligible participants"


_DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bimport\b",       # disallow new imports (safe ones are pre-injected)
        r"\bopen\b\s*\(",    # no file I/O
        r"\bos\b\.",         # no os module
        r"\bsubprocess\b",   # no subprocess
        r"\brequests?\b",    # no network
        r"__import__",       # no dynamic import
        r"\beval\b\s*\(",    # no eval
        r"\bexec\b\s*\(",    # no nested exec
        r"\bglobals\b\s*\(", # no globals()
        r"\blocals\b\s*\(",  # no locals()
    ]
]


def _security_check(code: str) -> None:
    """Raise ValueError if the code contains dangerous patterns."""
    for pat in _DANGEROUS_PATTERNS:
        m = pat.search(code)
        if m:
            raise ValueError(
                f"Generated analysis code contains disallowed pattern '{m.group()}'. "
                "The code may only use pre-injected pd, np, stats, and built-in functions."
            )
