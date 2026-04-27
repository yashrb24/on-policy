"""Statistical analysis utilities for DDCL toy-problem experiments.

All methods are explained in plain English in docs/STATS.md.

Functions
---------
bootstrap_ci           : bootstrap confidence interval for any statistic
iqm                    : interquartile mean
iqm_ci                 : IQM with bootstrap CI
paired_permutation_test: paired two-sided permutation test
wilcoxon_signed_rank   : Wilcoxon signed-rank test (requires scipy)
is_pareto_dominated    : check if a point is strictly dominated
pareto_frontier        : extract the Pareto frontier from a DataFrame
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a scalar statistic.

    Parameters
    ----------
    data    : 1-D array of observations (per-seed final performance values)
    stat_fn : statistic to bootstrap (default: mean)
    n_boot  : number of bootstrap resamples
    ci      : confidence level (default: 0.95 → 95% CI)
    rng     : optional NumPy random generator for reproducibility

    Returns
    -------
    (point_estimate, lower, upper)
    """
    data = np.asarray(data, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    point = stat_fn(data)
    n = len(data)
    boots = np.array([
        stat_fn(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------------------
# IQM (Interquartile Mean)
# ---------------------------------------------------------------------------

def iqm(data: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of values.

    More robust than the plain mean for heavy-tailed RL performance
    distributions (Agarwal et al., 2021).  Less sensitive to catastrophic
    seeds than median.
    """
    data = np.sort(np.asarray(data, dtype=float))
    n = len(data)
    lo = int(np.floor(n * 0.25))
    hi = int(np.ceil(n * 0.75))
    return float(data[lo:hi].mean())


def iqm_ci(
    data: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """IQM with stratified bootstrap CI (same interface as bootstrap_ci)."""
    return bootstrap_ci(data, stat_fn=iqm, n_boot=n_boot, ci=ci, rng=rng)


# ---------------------------------------------------------------------------
# Paired permutation test
# ---------------------------------------------------------------------------

def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Two-sided paired permutation test.

    Tests H₀: E[a] = E[b] against H₁: E[a] ≠ E[b].

    Parameters
    ----------
    a, b    : 1-D arrays of matched observations (same seeds / conditions)
    n_perm  : number of random sign-flip permutations
    rng     : optional generator for reproducibility

    Returns
    -------
    p-value (two-sided)

    Notes
    -----
    For ≤20 seeds the exact permutation distribution has 2^n cases; we use
    a Monte-Carlo approximation for simplicity. With 10K perms, the p-value
    has SE ≈ 0.003 at p=0.05.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), "a and b must have the same length (matched pairs)"
    if rng is None:
        rng = np.random.default_rng()

    diff = a - b
    observed = np.abs(diff.mean())

    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)), replace=True)
    null_means = np.abs((signs * diff).mean(axis=1))
    p_value = (null_means >= observed).mean()
    return float(p_value)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> float:
    """Wilcoxon signed-rank test p-value (requires scipy).

    Falls back to a warning + NaN if scipy is not installed.
    """
    try:
        from scipy.stats import wilcoxon  # type: ignore
        _, p = wilcoxon(a, b, alternative="two-sided")
        return float(p)
    except ImportError:
        warnings.warn(
            "scipy not installed; Wilcoxon test unavailable. "
            "Install with: pip install scipy",
            stacklevel=2,
        )
        return float("nan")


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def is_pareto_dominated(
    point: dict[str, float],
    population: list[dict[str, float]],
    better: dict[str, str],
) -> bool:
    """Return True if *point* is strictly dominated by any member of *population*.

    Parameters
    ----------
    point      : dict mapping metric_name → value for the candidate
    population : list of such dicts (including or excluding the candidate)
    better     : dict mapping metric_name → "higher" or "lower"

    A point X is strictly dominated by Y if Y is at least as good on all
    metrics and strictly better on at least one.
    """
    for other in population:
        if other is point:
            continue
        # Is `other` at least as good on every metric?
        at_least_as_good = all(
            (other[m] >= point[m] if d == "higher" else other[m] <= point[m])
            for m, d in better.items()
        )
        # Is `other` strictly better on at least one?
        strictly_better = any(
            (other[m] > point[m] if d == "higher" else other[m] < point[m])
            for m, d in better.items()
        )
        if at_least_as_good and strictly_better:
            return True
    return False


def pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_better: str = "higher",
    y_better: str = "lower",
) -> pd.DataFrame:
    """Extract the Pareto frontier from a DataFrame.

    Parameters
    ----------
    df       : DataFrame with one row per configuration
    x_col    : first objective column (e.g. 'success_rate')
    y_col    : second objective column (e.g. 'bits_per_msg')
    x_better : 'higher' or 'lower' for x
    y_better : 'higher' or 'lower' for y

    Returns
    -------
    DataFrame containing only the non-dominated rows, sorted by x_col.
    """
    better = {x_col: x_better, y_col: y_better}
    records = df[[x_col, y_col]].to_dict("records")
    mask = [
        not is_pareto_dominated(r, records, better)
        for r in records
    ]
    return df[mask].sort_values(x_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sweep winner comparison
# ---------------------------------------------------------------------------

def compare_configs(
    summary: pd.DataFrame,
    metric: str,
    group_col: str,
    higher_is_better: bool = True,
    n_perm: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Compare all configs pairwise on a metric using the permutation test.

    Parameters
    ----------
    summary  : per-seed final-metrics DataFrame (from load_runs.final_metrics)
    metric   : column to compare (e.g. 'success_rate')
    group_col: column identifying the config (e.g. 'channel', 'lambda_comms')
    higher_is_better: sign of preference
    n_perm   : permutations per pair
    alpha    : significance level

    Returns
    -------
    DataFrame with columns:
      config_a, config_b, delta_mean, p_value, significant
    Sorted by delta_mean descending (if higher_is_better) so the top row is
    the best config vs its nearest competitor.
    """
    configs = summary[group_col].unique()
    rows = []
    for i, ca in enumerate(configs):
        for j, cb in enumerate(configs):
            if j <= i:
                continue
            da = summary[summary[group_col] == ca]
            db = summary[summary[group_col] == cb]
            # Pair by seed when the column exists to avoid positional mismatches.
            if "seed" in summary.columns:
                shared = sorted(set(da["seed"]) & set(db["seed"]))
                if not shared:
                    continue
                va = da.set_index("seed").loc[shared, metric].values
                vb = db.set_index("seed").loc[shared, metric].values
            else:
                n = min(len(da), len(db))
                va = da[metric].values[:n]
                vb = db[metric].values[:n]
            p = paired_permutation_test(va, vb, n_perm=n_perm, rng=rng)
            rows.append({
                "config_a": ca,
                "config_b": cb,
                "mean_a": float(va.mean()),
                "mean_b": float(vb.mean()),
                "delta_mean": float(va.mean() - vb.mean()),
                "p_value": p,
                "significant": p < alpha,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        ascending = not higher_is_better
        result = result.sort_values("delta_mean", ascending=ascending).reset_index(drop=True)
    return result
