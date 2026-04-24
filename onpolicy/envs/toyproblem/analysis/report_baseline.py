"""Generate results/toyproblem/baseline.md from sweep run data.

Usage (after Stage A / B sweeps complete):

    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
        python -m onpolicy.envs.toyproblem.analysis.report_baseline \
        --sweep_dir runs/toyproblem/sweep_stage_a \
        --out_dir results/toyproblem/sweep_stage_a

Requires: analysis/load_runs.py, analysis/stats.py, analysis/plots.py
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from .load_runs import load_sweep, final_metrics, seed_aggregate
from .stats import bootstrap_ci, iqm_ci, pareto_frontier, compare_configs
from .plots import (
    plot_training_curves,
    plot_rate_distortion,
    plot_per_goal_bits,
    plot_channel_comparison,
    plot_bits_vs_entropy,
)
from .sweep_convergence import check_convergence

# Shannon entropy of the goal distribution — imported here for use in plots.
# Defined alongside the channel constants so there is one source of truth.
from onpolicy.envs.toyproblem.channels import H_GOAL_BITS, GOAL_OPTIMAL_BITS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ci(point: float, lo: float, hi: float, decimals: int = 3) -> str:
    """Format '0.923 [0.901, 0.941]'."""
    return f"{point:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _table_row(name: str, vals: np.ndarray) -> str:
    pt, lo, hi = bootstrap_ci(vals)
    iqm_pt, iqm_lo, iqm_hi = iqm_ci(vals)
    return (
        f"| {name} | {_fmt_ci(pt, lo, hi)} | {_fmt_ci(iqm_pt, iqm_lo, iqm_hi)} "
        f"| {len(vals)} |"
    )


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(
    sweep_dir: str | Path,
    out_dir: str | Path,
    metric: str = "success_rate",
    bits_col: str = "bits_per_msg",
    true_bits_col: str = "true_bits_per_msg",
    window: int = 20,
    sweep_axes: list[str] | None = None,
    n_perm: int = 10_000,
) -> None:
    """Load all runs from sweep_dir, run analysis, write baseline.md + plots."""
    sweep_dir = Path(sweep_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "figures"
    plots_dir.mkdir(exist_ok=True)

    if sweep_axes is None:
        sweep_axes = ["lambda_comms", "delta", "z_dim", "channel"]

    print(f"Loading runs from {sweep_dir} …")
    df = load_sweep(sweep_dir)
    print(f"  {len(df)} total rows, {df['seed'].nunique()} seeds")

    summary = final_metrics(df, window=window)
    group_cols = [c for c in ("exp_name", "channel", "lambda_comms", "delta", "z_dim")
                  if c in summary.columns]

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("Generating plots …")

    # 1. Training curves per channel
    if "channel" in df.columns:
        plot_training_curves(
            df, y_col="success_rate", group_col="channel",
            title="Success rate by channel type",
            save_path=plots_dir / "training_curves_success.png",
        )
        # Surrogate bits (used in training loss — always logged)
        plot_training_curves(
            df, y_col="bits_per_msg", group_col="channel",
            title="Surrogate bits/msg (Jensen UB on |z|/δ) by channel type",
            save_path=plots_dir / "training_curves_bits_surrogate.png",
        )
        # True transmission bits (float32 for none, log₂|m|+1 for quantized)
        if "true_bits_per_msg" in df.columns:
            plot_training_curves(
                df, y_col="true_bits_per_msg", group_col="channel",
                title="True transmission bits/msg by channel type\n"
                      "(none=32×z_dim, sd/nsd/additive=log₂|m|+1, ste=B×z_dim)",
                save_path=plots_dir / "training_curves_bits_true.png",
            )

    # 2. Rate–distortion frontier
    agg = seed_aggregate(summary, group_cols)
    grp_col = "channel" if "channel" in agg.columns else group_cols[0]

    # 2a. Surrogate bits (training objective perspective)
    pf = pareto_frontier(agg, x_col=f"{metric}_mean", y_col=f"{bits_col}_mean",
                         x_better="higher", y_better="lower")
    plot_rate_distortion(
        agg, x_col=f"{bits_col}_mean", y_col=f"{metric}_mean",
        group_col=grp_col, pareto_df=pf, h_goal=H_GOAL_BITS,
        title="Rate–distortion (surrogate bits: Jensen UB on |z|/δ)",
        save_path=plots_dir / "rate_distortion_surrogate.png",
    )

    # 2b. True transmission bits (deployment perspective) — requires true_bits_per_msg
    true_bits_col = true_bits_col if true_bits_col in agg.columns else f"{true_bits_col}_mean"
    true_bits_mean_col = f"{true_bits_col}_mean" if f"{true_bits_col}_mean" in agg.columns else None
    if true_bits_mean_col and true_bits_mean_col in agg.columns:
        pf_true = pareto_frontier(agg, x_col=f"{metric}_mean",
                                  y_col=true_bits_mean_col,
                                  x_better="higher", y_better="lower")
        plot_rate_distortion(
            agg, x_col=true_bits_mean_col, y_col=f"{metric}_mean",
            group_col=grp_col, pareto_df=pf_true, h_goal=H_GOAL_BITS,
            title="Rate–distortion (true transmission bits)\n"
                  "none=32×z_dim  |  sd/nsd/additive=log₂|m|+1  |  ste=B×z_dim",
            save_path=plots_dir / "rate_distortion_true.png",
        )

    # Keep legacy filename as symlink to surrogate for backward compat
    import shutil
    legacy = plots_dir / "rate_distortion.png"
    src = plots_dir / "rate_distortion_surrogate.png"
    if src.exists() and not legacy.exists():
        shutil.copy2(src, legacy)

    # 3. Per-goal bits vs optimal -log₂(p_i) allocation
    goal_cols = [c for c in df.columns if c.startswith("bits_goal_")]
    if goal_cols:
        # Compute total_updates from fixed Stage A config.
        _n_envs = int(df["n_envs"].iloc[0]) if "n_envs" in df.columns else 16
        _n_steps = int(df["n_steps"].iloc[0]) if "n_steps" in df.columns else 256
        _total_ts = int(df["total_timesteps"].iloc[0]) if "total_timesteps" in df.columns else 1_000_000
        _total_updates = _total_ts // (_n_envs * _n_steps)

        from onpolicy.envs.toyproblem.channels import _GOAL_PROBS
        plot_per_goal_bits(
            df, n_goals=len(goal_cols),
            group_col="channel" if "channel" in df.columns else group_cols[0],
            goal_optimal_bits=GOAL_OPTIMAL_BITS,
            goal_probs=_GOAL_PROBS,
            total_updates=_total_updates,
            show_uncertainty=True,
            save_path=plots_dir / "per_goal_bits.png",
        )

    # 4. Channel comparison bar chart
    if "channel" in summary.columns:
        plot_channel_comparison(
            summary, metrics=["success_rate", "bits_per_msg"],
            save_path=plots_dir / "channel_comparison.png",
        )

    # 5. Bits vs Shannon entropy (true transmission bits vs H(G))
    if true_bits_col in summary.columns and "channel" in summary.columns:
        plot_bits_vs_entropy(
            summary, h_goal=H_GOAL_BITS, bits_col=true_bits_col,
            save_path=plots_dir / "bits_vs_entropy.png",
        )

    # ------------------------------------------------------------------
    # Convergence gate
    # ------------------------------------------------------------------
    print("Running convergence gate …")
    valid_axes = [a for a in sweep_axes if a in summary.columns]
    conv = check_convergence(
        summary,
        sweep_axes=valid_axes,
        config_col=group_cols[0] if group_cols else "exp_name",
        metric=metric,
        bits_col=bits_col,
        n_perm=n_perm,
    )
    print(conv)

    # ------------------------------------------------------------------
    # Pairwise comparisons
    # ------------------------------------------------------------------
    comp_group = "channel" if "channel" in summary.columns else group_cols[0]
    comparisons = compare_configs(
        summary, metric=metric, group_col=comp_group, n_perm=n_perm
    )

    # ------------------------------------------------------------------
    # Write baseline.md
    # ------------------------------------------------------------------
    winner = conv.winner
    winner_vals = summary[summary[group_cols[0]] == winner][metric].values
    pt, lo, hi = bootstrap_ci(winner_vals)
    iqm_pt, iqm_lo, iqm_hi = iqm_ci(winner_vals)

    md_lines = [
        "# Baseline Results",
        "",
        f"> Generated from: `{sweep_dir}`",
        f"> Convergence gate: {'✅ DONE' if conv.all_pass else '🔲 NOT DONE'}",
        "",
        "## Summary",
        "",
        f"**Winner:** `{winner}`",
        f"  - {metric}: {_fmt_ci(pt, lo, hi)} (mean [95% bootstrap CI])",
        f"  - {metric}: {_fmt_ci(iqm_pt, iqm_lo, iqm_hi)} (IQM [95% CI])",
        "",
        "## Convergence Gate",
        "",
        "```",
        str(conv),
        "```",
        "",
        "## Per-Config Performance Table",
        "",
        f"| Config | {metric} mean [CI] | {metric} IQM [CI] | n_seeds |",
        "|--------|-------------------|------------------|---------|",
    ]

    for cfg in summary[group_cols[0]].unique():
        vals = summary[summary[group_cols[0]] == cfg][metric].values
        md_lines.append(_table_row(cfg, vals))

    md_lines += [
        "",
        "## Pairwise Statistical Comparisons",
        "",
        f"| Config A | Config B | Δ mean | p-value | Significant (α=0.05) |",
        "|----------|----------|--------|---------|----------------------|",
    ]
    for _, row in comparisons.iterrows():
        sig = "✅ YES" if row["significant"] else "NO"
        md_lines.append(
            f"| {row['config_a']} | {row['config_b']} "
            f"| {row['delta_mean']:+.4f} | {row['p_value']:.3f} | {sig} |"
        )

    md_lines += [
        "",
        "## Plots",
        "",
        "![Training curves](plots/training_curves_success.png)",
        "![Rate–distortion](plots/rate_distortion.png)",
        "![Channel comparison](plots/channel_comparison.png)",
        "![Per-goal bits](plots/per_goal_bits.png)",
    ]

    out_path = out_dir / "baseline.md"
    out_path.write_text("\n".join(md_lines))
    print(f"Report written to {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate baseline.md from sweep runs.")
    p.add_argument("--sweep_dir", type=str, required=True,
                   help="Root directory of sweep runs (contains <exp_name>/<seed>/ structure).")
    p.add_argument("--out_dir", type=str, default="results/toyproblem",
                   help="Directory to write baseline.md and plots/.")
    p.add_argument("--metric", type=str, default="success_rate")
    p.add_argument("--bits_col", type=str, default="bits_per_msg")
    p.add_argument("--window", type=int, default=20,
                   help="Number of final updates to average for summary metrics.")
    p.add_argument("--sweep_axes", type=str, nargs="+",
                   default=["lambda_comms", "delta", "z_dim", "channel"])
    p.add_argument("--n_perm", type=int, default=10_000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_report(
        sweep_dir=args.sweep_dir,
        out_dir=args.out_dir,
        metric=args.metric,
        bits_col=args.bits_col,
        window=args.window,
        sweep_axes=args.sweep_axes,
        n_perm=args.n_perm,
    )
