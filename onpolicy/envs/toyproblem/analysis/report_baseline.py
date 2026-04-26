"""Generate results/toyproblem/baseline.md from sweep run data.

Usage (after Stage A / B sweeps complete):

    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
        python -m onpolicy.envs.toyproblem.analysis.report_baseline \
        --sweep_dir runs/toyproblem/sweep_stage_a \
        --out_dir results/toyproblem/sweep_stage_a

Requires: analysis/load_runs.py, analysis/stats.py, analysis/paper_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .load_runs import load_sweep, final_metrics, seed_aggregate
from .stats import bootstrap_ci, iqm_ci, compare_configs
from .paper_figures import generate_sweep_figures
from .sweep_convergence import check_convergence


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
    agg = seed_aggregate(summary, group_cols)
    generate_sweep_figures(df, summary, agg, out_dir=plots_dir)

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
        "### Main paper figures",
        "![Rate–distortion frontier](figures/main/fig1_rate_distortion.png)",
        "![Channel comparison](figures/main/fig2_channel_comparison.png)",
        "",
        "### Appendix figures",
        "![λ sensitivity](figures/appendix/appA_lambda_sensitivity.png)",
        "![δ×λ heatmap](figures/appendix/appB_delta_lambda_heatmap.png)",
        "![Training dynamics](figures/appendix/appC_training_dynamics.png)",
        "![Per-goal allocation](figures/appendix/appD_per_goal_allocation.png)",
        "![Overhead above H(G)](figures/appendix/appE_overhead_above_hg.png)",
        "![Surrogate calibration](figures/appendix/appF_surrogate_calibration.png)",
        "![z_dim scaling](figures/appendix/appG_zdim_scaling.png)",
        "![SD vs NSD](figures/appendix/appH_sd_nsd_comparison.png)",
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
