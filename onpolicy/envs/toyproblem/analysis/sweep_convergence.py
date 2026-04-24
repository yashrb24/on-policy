"""Automated sweep convergence gate (Phase 2 §2.3).

A sweep is considered DONE when all four criteria hold:

1. Interior optimum — winner is not at any axis boundary (else extend range).
2. Statistical stability — winner vs runner-up not significant at α=0.05
   (paired permutation test; they are "statistically indistinct" near the top).
   NOTE: for the convergence gate we actually want the WINNER to be
   distinguishable from all others — see docstrings below.
3. Seed stability — winner ranks first under mean, median, AND IQM.
4. Pareto non-dominance — winner not strictly dominated on
   (success_rate, bits/episode).

Usage
-----
    from analysis.sweep_convergence import check_convergence
    result = check_convergence(summary_df, sweep_axes=["lambda_comms", "delta"])
    print(result)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stats import (
    iqm,
    paired_permutation_test,
    pareto_frontier,
    is_pareto_dominated,
)


@dataclass
class ConvergenceResult:
    """Result of the 4-criterion convergence gate."""

    # Criterion flags (True = PASS)
    interior_optimum: bool = False
    statistical_stability: bool = False
    seed_stability: bool = False
    pareto_nondominance: bool = False

    # Diagnostics
    winner: str = ""
    runner_up: str = ""
    p_value_winner_vs_runnerup: float = float("nan")
    boundary_axes: list[str] = field(default_factory=list)
    rank_mean: int = -1
    rank_median: int = -1
    rank_iqm: int = -1
    notes: list[str] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return (
            self.interior_optimum
            and self.statistical_stability
            and self.seed_stability
            and self.pareto_nondominance
        )

    def __str__(self) -> str:
        status = "DONE" if self.all_pass else "NOT DONE"
        lines = [
            f"Convergence gate: {status}",
            f"  Winner: {self.winner}",
            f"  1. Interior optimum   : {'PASS' if self.interior_optimum else 'FAIL'}"
            + (f"  (boundary axes: {self.boundary_axes})" if self.boundary_axes else ""),
            f"  2. Statistical stability: {'PASS' if self.statistical_stability else 'FAIL'}"
            + f"  (p vs runner-up = {self.p_value_winner_vs_runnerup:.3f})",
            f"  3. Seed stability     : {'PASS' if self.seed_stability else 'FAIL'}"
            + f"  (rank by mean={self.rank_mean}, median={self.rank_median}, IQM={self.rank_iqm})",
            f"  4. Pareto non-dom.    : {'PASS' if self.pareto_nondominance else 'FAIL'}",
        ]
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


def check_convergence(
    summary: pd.DataFrame,
    sweep_axes: list[str],
    config_col: str = "exp_name",
    metric: str = "success_rate",
    bits_col: str = "bits_per_msg",
    higher_metric_is_better: bool = True,
    alpha: float = 0.05,
    n_perm: int = 10_000,
    rng: np.random.Generator | None = None,
) -> ConvergenceResult:
    """Run the 4-criterion convergence gate on a sweep summary.

    Parameters
    ----------
    summary     : per-seed final-metrics DataFrame. Must have columns:
                  config_col, metric, bits_col, seed, and all sweep_axes.
    sweep_axes  : list of hyperparameter column names that were swept
                  (e.g. ["lambda_comms", "delta"])
    config_col  : column that uniquely identifies each configuration
    metric      : primary performance metric (e.g. 'success_rate')
    bits_col    : communication cost metric (e.g. 'bits_per_msg')
    alpha       : significance level for criterion 2

    Returns
    -------
    ConvergenceResult with .all_pass indicating whether sweeping is complete.
    """
    result = ConvergenceResult()
    if rng is None:
        rng = np.random.default_rng(0)

    # Aggregate per-config: mean, median, IQM across seeds.
    configs = summary[config_col].unique()

    def agg(cfg: str) -> dict:
        vals = summary[summary[config_col] == cfg][metric].values
        bits = summary[summary[config_col] == cfg][bits_col].values
        return {
            "config": cfg,
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "iqm": iqm(vals),
            "bits_mean": float(bits.mean()),
            "n_seeds": len(vals),
        }

    agg_df = pd.DataFrame([agg(c) for c in configs])

    # Rank by each aggregator (rank 0 = best).
    ascending = not higher_metric_is_better
    for col in ("mean", "median", "iqm"):
        agg_df[f"rank_{col}"] = agg_df[col].rank(ascending=ascending, method="min").astype(int) - 1

    # Winner = top by mean.
    winner_row = agg_df.sort_values("mean", ascending=ascending).iloc[0]
    winner = winner_row["config"]
    runner_up_row = agg_df.sort_values("mean", ascending=ascending).iloc[1] if len(agg_df) > 1 else None
    runner_up = runner_up_row["config"] if runner_up_row is not None else ""

    result.winner = winner
    result.runner_up = runner_up
    result.rank_mean = int(winner_row["rank_mean"])
    result.rank_median = int(winner_row["rank_median"])
    result.rank_iqm = int(winner_row["rank_iqm"])

    # ------------------------------------------------------------------
    # Criterion 1: Interior optimum
    # ------------------------------------------------------------------
    # For each sweep axis, check whether the winner is at the boundary
    # of the axis's observed range.
    boundary_axes = []
    for axis in sweep_axes:
        if axis not in summary.columns:
            result.notes.append(f"sweep axis '{axis}' not in summary columns")
            continue
        winner_val = summary[summary[config_col] == winner][axis].iloc[0]
        axis_vals = summary[axis]
        # Skip categorical (non-numeric) axes — boundary concept doesn't apply.
        if not np.issubdtype(axis_vals.dtype, np.number):
            continue
        axis_min = axis_vals.min()
        axis_max = axis_vals.max()
        # Allow 1% tolerance for floating-point boundary.
        if np.isclose(winner_val, axis_min, rtol=1e-2) or np.isclose(winner_val, axis_max, rtol=1e-2):
            boundary_axes.append(f"{axis}={winner_val:.3g} (range [{axis_min:.3g},{axis_max:.3g}])")

    result.boundary_axes = boundary_axes
    result.interior_optimum = len(boundary_axes) == 0

    # ------------------------------------------------------------------
    # Criterion 2: Statistical stability (winner significantly better
    # than runner-up at α)
    # ------------------------------------------------------------------
    if runner_up:
        wa = summary[summary[config_col] == winner][metric].values
        wb = summary[summary[config_col] == runner_up][metric].values
        n = min(len(wa), len(wb))
        p = paired_permutation_test(wa[:n], wb[:n], n_perm=n_perm, rng=rng)
        result.p_value_winner_vs_runnerup = p
        # PASS: winner IS significantly better than runner-up (p < α).
        result.statistical_stability = p < alpha
    else:
        result.statistical_stability = True  # only one config

    # ------------------------------------------------------------------
    # Criterion 3: Seed stability
    # ------------------------------------------------------------------
    result.seed_stability = (
        result.rank_mean == 0
        and result.rank_median == 0
        and result.rank_iqm == 0
    )

    # ------------------------------------------------------------------
    # Criterion 4: Pareto non-dominance
    # ------------------------------------------------------------------
    # Winner must not be strictly dominated on (metric, bits_col).
    x_better = "higher" if higher_metric_is_better else "lower"
    pf = pareto_frontier(agg_df, x_col="mean", y_col="bits_mean",
                         x_better=x_better, y_better="lower")
    result.pareto_nondominance = winner in pf["config"].values

    return result


def report_convergence(result: ConvergenceResult) -> str:
    """Format a human-readable convergence report string."""
    return str(result)
