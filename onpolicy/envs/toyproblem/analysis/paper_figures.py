"""Publication-quality figure functions for the DDCL toy-problem paper.

Each function is annotated with:
  - Hypothesis  : the scientific claim the figure is designed to support or refute
  - Analysis    : what the figure shows / how it is constructed
  - Conclusion  : what a reader should take away if the hypothesis holds

All figures follow the same calling convention:
  fig, ax = plot_paper_*(df_or_summary, ..., save_path=None)

Typical usage after a Stage A/B/C sweep:

    from onpolicy.envs.toyproblem.analysis.load_runs import load_sweep, final_metrics, seed_aggregate
    from onpolicy.envs.toyproblem.analysis.stats import pareto_frontier
    from onpolicy.envs.toyproblem.analysis.paper_figures import *
    from onpolicy.envs.toyproblem.channels import H_GOAL_BITS, GOAL_OPTIMAL_BITS, _GOAL_PROBS

    df = load_sweep("runs/toyproblem/sweep_stage_a")
    summary = final_metrics(df)
    agg = seed_aggregate(summary, group_cols=["channel", "lambda_comms", "delta", "z_dim"])

    plot_paper_rate_distortion(agg, save_path="results/toyproblem/sweep_stage_a/figures/fig_rate_distortion.pdf")
    plot_paper_per_goal_allocation(df, save_path="results/toyproblem/sweep_stage_a/figures/fig_per_goal_bits.pdf")
    ...

Dependency: matplotlib (required), scipy (optional, for Wilcoxon).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from onpolicy.envs.toyproblem.channels import (
    H_GOAL_BITS,
    GOAL_OPTIMAL_BITS,
    _GOAL_PROBS,
)
from onpolicy.envs.toyproblem.analysis.stats import pareto_frontier


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PAPER_RCPARAMS = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.8,
    "figure.dpi": 200,
}

_CHANNEL_COLORS = {
    "none":             "#888888",
    "sd":               "#1f77b4",
    "nsd":              "#ff7f0e",
    "additive_uniform": "#2ca02c",
    "gaussian":         "#d62728",
    "ste4":             "#9467bd",
    "ste8":             "#8c564b",
    "ste16":            "#e377c2",
}

_CHANNEL_LABELS = {
    "none":             "None (float32)",
    "sd":               "SD (DDCL subtractive)",
    "nsd":              "NSD (DDCL TPDF)",
    "additive_uniform": "Additive uniform",
    "gaussian":         "Gaussian (negative ctrl)",
    "ste4":             "STE-4 bit",
    "ste8":             "STE-8 bit",
    "ste16":            "STE-16 bit",
}


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")


def _paper_style() -> None:
    """Apply paper-quality rcParams."""
    plt.rcParams.update(_PAPER_RCPARAMS)


def _save(fig, save_path) -> None:
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure 1: Rate–Distortion Frontier (core result)
# ---------------------------------------------------------------------------

def plot_paper_rate_distortion(
    agg: pd.DataFrame,
    x_col: str = "true_bits_per_msg_mean",
    y_col: str = "success_rate_mean",
    group_col: str = "channel",
    title: str = "Rate–distortion frontier",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    DDCL channels (sd, nsd) lie on the Pareto frontier of task success rate vs
    true transmission cost. Fixed-rate baselines (STE) require far more bits for
    the same SR. The float passthrough (none) achieves perfect SR at the maximum
    float32 cost (32 × z_dim bits/element).

    Analysis
    --------
    Scatter plot of (true_bits_per_msg, success_rate) for every (config, λ, δ)
    combination. Each point is the seed-mean. The Pareto frontier (minimise bits,
    maximise SR) is overlaid as a black step line. A vertical dashed red line
    marks H(G) = 1.812 bits — the Shannon information-theoretic lower bound on
    the bits needed to communicate goal identity under the Zipf goal distribution.

    Conclusion
    ----------
    If DDCL achieves SR ≥ 0.95 at bits/msg ≈ H(G), the speaker has learned a
    near-optimal code. STE channels at 4–16 bits/msg with comparable SR confirm
    that quantisation overhead is the bottleneck, not task difficulty. The float
    baseline anchors the upper-right corner.
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    if group_col not in agg.columns:
        groups = [None]
    else:
        groups = list(agg[group_col].unique())

    for grp in groups:
        sub = agg if grp is None else agg[agg[group_col] == grp]
        color = _CHANNEL_COLORS.get(str(grp), "#333333")
        label = _CHANNEL_LABELS.get(str(grp), str(grp))
        marker = "o" if grp in ("sd", "nsd") else ("s" if grp == "none" else "^")
        ax.scatter(
            sub[x_col], sub[y_col],
            color=color, marker=marker, s=55, label=label,
            alpha=0.85, zorder=3,
        )

    # Pareto frontier
    if x_col in agg.columns and y_col in agg.columns:
        pf = pareto_frontier(agg, x_col=y_col, y_col=x_col,
                             x_better="higher", y_better="lower")
        # pf sorted by SR; re-sort by bits for step plot
        pf = pf.sort_values(x_col)
        ax.step(pf[x_col], pf[y_col], where="post",
                color="black", linewidth=2.0, linestyle="--",
                label="Pareto frontier", zorder=5)

    # Shannon entropy reference
    ax.axvline(H_GOAL_BITS, color="red", linewidth=1.5, linestyle=":",
               label=f"H(G) = {H_GOAL_BITS:.3f} bits", zorder=6)

    ax.set_xlabel("True transmission bits / message")
    ax.set_ylabel("Task success rate")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(framealpha=0.85, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 2: Training Curves by Channel
# ---------------------------------------------------------------------------

def plot_paper_training_curves(
    df: pd.DataFrame,
    y_col: str = "success_rate",
    x_col: str = "timestep",
    group_col: str = "channel",
    smooth: int = 15,
    title: str = "Learning curves by channel",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    All channels converge to comparable final SR when λ is small; λ > 0 slows
    initial convergence but reaches a lower plateau that uses fewer bits.

    Analysis
    --------
    Success rate (smoothed, window=15 updates) over timesteps, one curve per
    channel type. Shaded band shows ±1 std over seeds. A separate subplot shows
    true_bits_per_msg over time for DDCL channels to reveal how bit compression
    co-evolves with task learning.

    Conclusion
    ----------
    If DDCL curves converge to SR ≈ 1.0 under λ=0 and fall gracefully with
    increasing λ, the training dynamics are stable. A two-phase pattern (SR rises
    first; bits decrease after SR stabilises) supports the interpretation that
    communication efficiency is learned once the task is solved.
    """
    _require_mpl()
    _paper_style()

    has_bits = "bits_per_msg" in df.columns
    n_rows = 2 if has_bits else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    groups = df[group_col].unique() if group_col in df.columns else [None]
    for ax, metric in zip(axes, [y_col] + (["bits_per_msg"] if has_bits else [])):
        for grp in groups:
            sub = df if grp is None else df[df[group_col] == grp]
            color = _CHANNEL_COLORS.get(str(grp), None)
            label = _CHANNEL_LABELS.get(str(grp), str(grp))

            pivoted = sub.pivot_table(index=x_col, columns="seed", values=metric)
            if smooth > 1:
                pivoted = pivoted.rolling(smooth, min_periods=1).mean()
            xs = pivoted.index.values
            mean = pivoted.mean(axis=1).values
            std = pivoted.std(axis=1).fillna(0).values

            ax.plot(xs, mean, label=label, color=color)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.18, color=color)

        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.25)
        if metric == y_col:
            ax.set_ylim(-0.05, 1.08)
            ax.legend(framealpha=0.85, ncol=2)
            ax.set_title(title)

    axes[-1].set_xlabel("Environment timesteps")
    plt.tight_layout()
    _save(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# Figure 3: Per-Goal Bit Allocation with Uncertainty
# ---------------------------------------------------------------------------

def plot_paper_per_goal_allocation(
    df: pd.DataFrame,
    n_goals: int = 6,
    group_col: str = "channel",
    window: int = 20,
    total_updates: int | None = None,
    title: str = "Per-goal bit allocation vs Shannon-optimal",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    The speaker learns to allocate bits approximately proportional to -log₂(p_i),
    the Shannon-optimal code length, even without explicit supervision.  Frequent
    goals (goal 0, p=0.515) receive fewer bits; rare goals (goal 5, p=0.003)
    receive more.  The allocation emerges from the joint communication + task
    objective, not from a hard-coded entropy model.

    Analysis
    --------
    Bar chart of mean surrogate bits allocated to each goal (averaged over the
    last *window* updates and over seeds), grouped by channel.  Overlaid:
    (1) the theoretical optimal -log₂(p_i) as a dashed red line;
    (2) a shaded uncertainty envelope combining two independent sources:
       - Jensen gap (±0.20 bits, symmetric): the surrogate log₂(|z|/δ+1)
         overestimates true cost in mid-range operating regimes and
         underestimates near z=0.
       - RL sampling noise: scales as C_RL/sqrt(p_i × N_updates) with C_RL≈0.5 bits.
         Goal 5 (p=0.003) sees ≈13× higher gradient noise than goal 0 (p=0.515),
         so its allocation estimate is far less reliable.
    Effective sample counts n_i ≈ p_i × N_updates are annotated on the x-axis.

    Conclusion
    ----------
    DDCL approximately learns entropy-optimal allocation.  Deviations are
    consistent with the predicted uncertainty — not systematic failures.
    The shaded region ensures the reader does not over-interpret small
    deviations for rare goals, where the gradient signal is sparse.
    """
    _require_mpl()
    _paper_style()

    goal_cols = [f"bits_goal_{i}" for i in range(n_goals)]
    if any(c not in df.columns for c in goal_cols):
        raise ValueError(f"Missing goal columns in df: {[c for c in goal_cols if c not in df.columns]}")

    if total_updates is None:
        if "total_timesteps" in df.columns and "n_envs" in df.columns and "n_steps" in df.columns:
            total_updates = int(
                df["total_timesteps"].iloc[0]
                / (df["n_envs"].iloc[0] * df["n_steps"].iloc[0])
            )
        else:
            total_updates = 244  # default: 1M / (16 × 256)

    def _tail_mean(grp: pd.DataFrame) -> pd.Series:
        return grp.sort_values("update").tail(window)[goal_cols].mean()

    group_keys = [c for c in (group_col, "seed") if c in df.columns]
    per_seed = df.groupby(group_keys).apply(_tail_mean).reset_index()
    config_means = (
        per_seed.groupby(group_col)[goal_cols].mean()
        if group_col in per_seed.columns
        else per_seed[goal_cols].mean().to_frame().T
    )

    groups = config_means.index.tolist()
    n_groups = len(groups)
    x = np.arange(n_goals)
    width = 0.8 / max(n_groups, 1)
    colors = [_CHANNEL_COLORS.get(str(g), cm.tab10(i / 10)) for i, g in enumerate(groups)]
    x_center = x + width * (n_groups - 1) / 2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (grp, color) in enumerate(zip(groups, colors)):
        vals = config_means.loc[grp, goal_cols].values
        label = _CHANNEL_LABELS.get(str(grp), str(grp))
        ax.bar(x + i * width, vals, width=width, label=label,
               color=color, alpha=0.82)

    # Optimal allocation line
    opt = np.array(GOAL_OPTIMAL_BITS[:n_goals])
    ax.plot(x_center, opt, color="red", marker="x", linewidth=1.8,
            linestyle="--", markersize=8, label="Optimal −log₂(p_i)", zorder=5)

    # Uncertainty envelope
    probs = np.array(_GOAL_PROBS[:n_goals])
    sigma_rl = 0.5 / np.sqrt(probs * total_updates)
    sigma_jensen = 0.20
    sigma_total = np.sqrt(sigma_rl ** 2 + sigma_jensen ** 2)

    ax.fill_between(
        x_center, opt - sigma_total, opt + sigma_total,
        color="red", alpha=0.12, zorder=4,
        label=r"Uncertainty: $\sigma_{\rm Jensen}$=0.20b + $\sigma_{\rm RL}(i)$",
    )

    # Effective samples annotation
    for gi in range(n_goals):
        n_eff = probs[gi] * total_updates
        ax.text(x_center[gi], ax.get_ylim()[0] + 0.02,
                f"n≈{n_eff:.0f}", ha="center", va="bottom",
                fontsize=7, color="dimgrey")

    ax.set_xticks(x_center)
    ax.set_xticklabels(
        [f"Goal {i}\n(p={_GOAL_PROBS[i]:.3f})" for i in range(n_goals)],
        fontsize=8,
    )
    ax.set_ylabel("Mean surrogate bits allocated")
    ax.set_title(title)
    ax.legend(framealpha=0.85, fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    _save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 4: λ Sensitivity Analysis
# ---------------------------------------------------------------------------

def plot_paper_lambda_sensitivity(
    summary: pd.DataFrame,
    channel: str = "sd",
    delta: float | None = None,
    z_dim: int | None = None,
    sr_col: str = "success_rate",
    bits_col: str = "true_bits_per_msg",
    title: str = "λ sensitivity: rate–distortion tradeoff",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    There exists an optimal λ* such that:
      - λ < λ*: policy succeeds but wastes bits (no effective compression)
      - λ ≈ λ*: maximum SR at minimum bits (the Pareto knee)
      - λ > λ*: bits reduce further but SR drops significantly (over-compressed)
    The knee corresponds to the operating point a practitioner should deploy.

    Analysis
    --------
    Dual-axis line plot: success rate (left y-axis, blue) and true_bits_per_msg
    (right y-axis, orange) vs λ (log-scale x-axis).  Each λ value shows mean ±
    std over seeds at fixed (δ, z_dim) = (best values from Stage A).  A vertical
    dashed line marks the rate-distortion knee: argmin λ such that SR ≥ 0.95.

    Conclusion
    ----------
    The plot confirms that λ controls a smooth rate-distortion tradeoff.  The
    knee location identifies the minimum-overhead operating point.  If SR remains
    above 0.95 down to bits/msg ≈ H(G), DDCL achieves near-Shannon efficiency.
    """
    _require_mpl()
    _paper_style()

    sub = summary.copy()
    if "channel" in sub.columns:
        sub = sub[sub["channel"] == channel]
    if delta is not None and "delta" in sub.columns:
        sub = sub[sub["delta"] == delta]
    if z_dim is not None and "z_dim" in sub.columns:
        sub = sub[sub["z_dim"] == z_dim]

    if sub.empty:
        raise ValueError(f"No rows for channel={channel}, delta={delta}, z_dim={z_dim}")

    lambdas = sorted(sub["lambda_comms"].unique()) if "lambda_comms" in sub.columns else []
    sr_means, sr_stds, bits_means, bits_stds = [], [], [], []
    for lam in lambdas:
        vals_sr = sub[sub["lambda_comms"] == lam][sr_col].values
        vals_bits = sub[sub["lambda_comms"] == lam][bits_col].values
        sr_means.append(vals_sr.mean())
        sr_stds.append(vals_sr.std())
        bits_means.append(vals_bits.mean())
        bits_stds.append(vals_bits.std())

    lambdas_arr = np.array(lambdas, dtype=float)
    sr_means = np.array(sr_means)
    sr_stds = np.array(sr_stds)
    bits_means = np.array(bits_means)
    bits_stds = np.array(bits_stds)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.errorbar(lambdas_arr, sr_means, yerr=sr_stds,
                 color="steelblue", marker="o", capsize=4, label="Success rate")
    ax2.errorbar(lambdas_arr, bits_means, yerr=bits_stds,
                 color="darkorange", marker="s", capsize=4, label="Bits/msg (true)")

    # Annotate H(G) reference on bits axis
    ax2.axhline(H_GOAL_BITS, color="red", linewidth=1.2, linestyle=":",
                label=f"H(G) = {H_GOAL_BITS:.3f} bits")

    # Mark Pareto knee: smallest λ with SR ≥ 0.95
    knee_mask = sr_means >= 0.95
    if knee_mask.any():
        knee_lam = lambdas_arr[knee_mask][-1]  # largest λ still meeting SR threshold
        ax1.axvline(knee_lam, color="green", linewidth=1.5, linestyle="--",
                    label=f"λ* ≈ {knee_lam:.1e} (SR≥0.95 threshold)")

    ax1.set_xscale("symlog", linthresh=1e-6)
    ax1.set_xlabel("λ (communication penalty)")
    ax1.set_ylabel("Success rate", color="steelblue")
    ax2.set_ylabel("True bits / message", color="darkorange")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax1.set_title(f"{title}\n(channel={channel}, δ={delta}, z_dim={z_dim})")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.85)
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, save_path)
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Figure 5: Channel Efficiency (normalised to H(G))
# ---------------------------------------------------------------------------

def plot_paper_channel_efficiency(
    summary: pd.DataFrame,
    bits_col: str = "true_bits_per_msg",
    sr_col: str = "success_rate",
    group_col: str = "channel",
    title: str = "Communication efficiency relative to H(G)",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    DDCL channels use fewer bits per unit of task success than STE baselines.
    Normalising by H(G) makes the gap interpretable: a ratio of 1.0 means the
    channel transmits at exactly the Shannon entropy of the goal distribution.
    STE channels have fixed-rate cost (bits × z_dim) independent of goal content,
    giving high ratios.

    Analysis
    --------
    Horizontal bar chart with one bar per channel type.  The bar height is the
    *efficiency ratio*:  (mean true_bits_per_msg / H(G)) at the best (λ, δ, z_dim)
    configuration.  A vertical dashed line at ratio=1 marks the Shannon limit.
    Secondary bar or text overlay shows the corresponding mean success rate.

    Conclusion
    ----------
    Channels with ratio close to 1.0 at SR ≥ 0.95 are information-theoretically
    efficient.  The gap between DDCL and STE illustrates the quantitative advantage
    of variable-rate coding over fixed-rate coding.
    """
    _require_mpl()
    _paper_style()

    channels = summary[group_col].unique()
    ratios, srs = [], []
    for ch in channels:
        sub = summary[summary[group_col] == ch]
        # Best config for each channel: maximum SR, then minimum bits.
        best_idx = sub[sr_col].idxmax()
        ratios.append(sub.loc[best_idx, bits_col] / H_GOAL_BITS)
        srs.append(sub.loc[best_idx, sr_col])

    y = np.arange(len(channels))
    colors = [_CHANNEL_COLORS.get(str(c), "#888") for c in channels]
    labels = [_CHANNEL_LABELS.get(str(c), str(c)) for c in channels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(y, ratios, color=colors, alpha=0.85)

    # Annotate SR on each bar
    for bar, sr in zip(bars, srs):
        ax.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"SR={sr:.2f}", va="center", fontsize=8,
        )

    ax.axvline(1.0, color="red", linewidth=1.5, linestyle="--",
               label=f"Shannon limit (H(G) = {H_GOAL_BITS:.3f} bits)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Bits/msg ÷ H(G)")
    ax.set_title(title)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    _save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 6: SD vs NSD Head-to-Head
# ---------------------------------------------------------------------------

def plot_paper_sd_nsd_comparison(
    summary: pd.DataFrame,
    sr_col: str = "success_rate",
    bits_col: str = "true_bits_per_msg",
    lambda_col: str = "lambda_comms",
    title: str = "SD vs NSD: synchrony-free DDCL",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    NSD (TPDF dither, no shared PRNG between sender and receiver) performs
    comparably to SD (subtractive dither, requires shared PRNG) on the rate-
    distortion frontier.  The synchrony-free deployment advantage of NSD is
    effectively "free" in terms of task performance.

    Analysis
    --------
    Left panel: scatter of (true_bits_per_msg, success_rate) for SD (blue) and
    NSD (orange), at matched (δ, z_dim) and all λ values.  The Pareto frontiers
    of each channel are overlaid.  Right panel: paired scatter with SD on x-axis
    and NSD on y-axis for the same (λ, δ, z_dim, seed) tuples — points above
    the diagonal favour NSD; points below favour SD.

    Conclusion
    ----------
    If the Pareto frontiers overlap and the paired scatter is distributed around
    the diagonal, NSD ≈ SD with no systematic winner.  This validates NSD as a
    practical drop-in replacement that eliminates the synchronisation requirement.
    Any consistent deviation from the diagonal should be reported and investigated.
    """
    _require_mpl()
    _paper_style()

    sd = summary[summary["channel"] == "sd"] if "channel" in summary.columns else summary
    nsd = summary[summary["channel"] == "nsd"] if "channel" in summary.columns else summary

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: rate-distortion scatter
    ax = axes[0]
    for sub, label, color in [(sd, "SD", _CHANNEL_COLORS["sd"]),
                               (nsd, "NSD", _CHANNEL_COLORS["nsd"])]:
        ax.scatter(sub[bits_col], sub[sr_col], color=color, label=label,
                   alpha=0.75, s=40)
        if len(sub) >= 2:
            pf = pareto_frontier(sub, x_col=sr_col, y_col=bits_col,
                                 x_better="higher", y_better="lower")
            pf = pf.sort_values(bits_col)
            ax.step(pf[bits_col], pf[sr_col], where="post",
                    color=color, linewidth=2.0, linestyle="--")

    ax.axvline(H_GOAL_BITS, color="red", linewidth=1.2, linestyle=":",
               label=f"H(G)={H_GOAL_BITS:.2f} b")
    ax.set_xlabel("True bits/msg")
    ax.set_ylabel("Success rate")
    ax.set_title("Rate–distortion: SD vs NSD")
    ax.legend(framealpha=0.85)
    ax.grid(True, alpha=0.25)

    # Right: paired scatter (requires matching keys)
    ax = axes[1]
    merge_cols = [c for c in ("lambda_comms", "delta", "z_dim", "seed")
                  if c in sd.columns and c in nsd.columns]
    if merge_cols:
        paired = sd[merge_cols + [sr_col, bits_col]].merge(
            nsd[merge_cols + [sr_col, bits_col]],
            on=merge_cols, suffixes=("_sd", "_nsd"),
        )
        ax.scatter(paired[f"{sr_col}_sd"], paired[f"{sr_col}_nsd"],
                   alpha=0.6, s=30, color="purple")
        lo = min(paired[f"{sr_col}_sd"].min(), paired[f"{sr_col}_nsd"].min()) - 0.02
        hi = max(paired[f"{sr_col}_sd"].max(), paired[f"{sr_col}_nsd"].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="SD = NSD")
        ax.set_xlabel("SD success rate")
        ax.set_ylabel("NSD success rate")
        ax.set_title("Paired SR comparison (same config × seed)")
        ax.legend(framealpha=0.85)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Paired comparison requires\nmatching lambda/delta/z_dim/seed",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# Figure 7: Surrogate vs True Bits Calibration
# ---------------------------------------------------------------------------

def plot_paper_surrogate_calibration(
    df: pd.DataFrame,
    surrogate_col: str = "bits_per_msg",
    true_col: str = "true_bits_per_msg",
    group_col: str = "channel",
    title: str = "Surrogate calibration: Jensen UB vs true transmission bits",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    The surrogate bits_per_msg (Jensen upper bound: log₂(|z|/δ+1)) is a
    monotone proxy for true_bits_per_msg but systematically overestimates,
    especially at low z magnitudes (near z=0 where dither alone determines m)
    and at bin boundaries.  The gap shrinks as |z|/δ grows.

    Analysis
    --------
    Scatter plot of (bits_per_msg, true_bits_per_msg) sampled from training
    updates, one point per update × channel.  Reference line y=x (perfect
    calibration).  Points above the diagonal indicate overestimation; below
    indicate underestimation.  Inset shows the distribution of gaps
    (surrogate − true) per channel.

    Conclusion
    ----------
    The surrogate is a valid training objective (monotone, differentiable,
    non-negative) but true_bits_per_msg must be reported for fair cross-channel
    comparison.  Channels with high λ (z→0 regime) show larger under-estimation
    gaps near zero; this is expected from the Jensen gap analysis in MATH.md.
    """
    _require_mpl()
    _paper_style()

    if surrogate_col not in df.columns or true_col not in df.columns:
        raise ValueError(
            f"Need columns '{surrogate_col}' and '{true_col}' in df. "
            f"Available: {list(df.columns)}"
        )

    groups = df[group_col].unique() if group_col in df.columns else [None]
    fig, ax = plt.subplots(figsize=(6, 5))

    for grp in groups:
        sub = df if grp is None else df[df[group_col] == grp]
        color = _CHANNEL_COLORS.get(str(grp), None)
        label = _CHANNEL_LABELS.get(str(grp), str(grp))
        ax.scatter(sub[surrogate_col], sub[true_col],
                   color=color, label=label, alpha=0.4, s=8)

    all_vals = np.concatenate([df[surrogate_col].values, df[true_col].values])
    lo, hi = all_vals.min(), all_vals.max()
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="y = x (perfect)")

    ax.set_xlabel("Surrogate bits/msg  log₂(|z|/δ+1)  (Jensen UB)")
    ax.set_ylabel("True transmission bits/msg")
    ax.set_title(title)
    ax.legend(framealpha=0.85, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 8: z_dim Scaling
# ---------------------------------------------------------------------------

def plot_paper_zdim_scaling(
    summary: pd.DataFrame,
    sr_col: str = "success_rate",
    bits_col: str = "true_bits_per_msg",
    group_col: str = "channel",
    title: str = "Message dimension scaling",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    More message dimensions (larger z_dim) give the speaker greater capacity to
    encode the goal, improving SR — but also increase total bits.  An optimal
    z_dim exists where SR saturates and per-dimension bits decrease (the speaker
    learns factored encoding with individual dimensions specialising).

    Analysis
    --------
    Two panels: (left) success rate vs z_dim, grouped by channel; (right) true
    bits/msg vs z_dim.  Each channel shows mean ± std over seeds at the
    best (λ, δ) found in Stage A.

    Conclusion
    ----------
    If SR plateaus at z_dim=2 or 3 but bits continue to increase with z_dim,
    larger z_dim wastes capacity.  If per-dim bits decrease with z_dim (total
    bits sub-linear in z_dim), the speaker is efficiently using each dimension.
    """
    _require_mpl()
    _paper_style()

    if "z_dim" not in summary.columns:
        raise ValueError("'z_dim' column not in summary DataFrame")

    groups = summary[group_col].unique() if group_col in summary.columns else [None]
    zdims = sorted(summary["z_dim"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for grp in groups:
        sub = summary if grp is None else summary[summary[group_col] == grp]
        color = _CHANNEL_COLORS.get(str(grp), None)
        label = _CHANNEL_LABELS.get(str(grp), str(grp))

        sr_means, sr_stds, b_means, b_stds = [], [], [], []
        for z in zdims:
            vals_sr = sub[sub["z_dim"] == z][sr_col].values
            vals_b = sub[sub["z_dim"] == z][bits_col].values
            sr_means.append(vals_sr.mean() if len(vals_sr) else np.nan)
            sr_stds.append(vals_sr.std() if len(vals_sr) else np.nan)
            b_means.append(vals_b.mean() if len(vals_b) else np.nan)
            b_stds.append(vals_b.std() if len(vals_b) else np.nan)

        axes[0].errorbar(zdims, sr_means, yerr=sr_stds,
                         color=color, marker="o", capsize=3, label=label)
        axes[1].errorbar(zdims, b_means, yerr=b_stds,
                         color=color, marker="o", capsize=3, label=label)

    axes[0].set_xlabel("z_dim")
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("SR vs z_dim")
    axes[0].legend(framealpha=0.85, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].axhline(H_GOAL_BITS, color="red", linewidth=1.2, linestyle=":",
                    label=f"H(G)={H_GOAL_BITS:.2f} b")
    axes[1].set_xlabel("z_dim")
    axes[1].set_ylabel("True bits/msg")
    axes[1].set_title("True bits vs z_dim")
    axes[1].legend(framealpha=0.85, fontsize=8)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title)
    plt.tight_layout()
    _save(fig, save_path)
    return fig, axes


# ---------------------------------------------------------------------------
# Figure 9: Per-Goal Bit Dynamics over Training
# ---------------------------------------------------------------------------

def plot_paper_per_goal_dynamics(
    df: pd.DataFrame,
    n_goals: int = 6,
    channel_filter: str | None = "sd",
    smooth: int = 20,
    title: str = "Per-goal bit allocation over training",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    The speaker starts with near-uniform bit allocation across goals and
    gradually learns the entropy-optimal pattern as the policy matures.
    The learning of communication efficiency is correlated with task success:
    the policy first learns to navigate to the goal (SR rises), then refines
    its encoding efficiency (bits converge toward -log₂(p_i)).

    Analysis
    --------
    Training curves for bits_goal_0 through bits_goal_{n-1}, smoothed and
    shaded over seeds, on a single axis.  Horizontal dashed lines mark the
    optimal -log₂(p_i) allocation for each goal.  A secondary axis (right)
    shows the overall success rate for reference.

    Conclusion
    ----------
    If allocations converge toward optimal from a uniform starting point, this
    confirms emergent entropy coding.  A two-phase dynamic (SR plateau then
    bit refinement) supports the interpretation that task success is a
    prerequisite for efficient communication, not the reverse.
    """
    _require_mpl()
    _paper_style()

    goal_cols = [f"bits_goal_{i}" for i in range(n_goals)]
    available = [c for c in goal_cols if c in df.columns]
    if not available:
        raise ValueError("No bits_goal_* columns found in df")

    sub = df if channel_filter is None else df[df["channel"] == channel_filter]
    if sub.empty:
        raise ValueError(f"No rows for channel={channel_filter}")

    x_col = "timestep" if "timestep" in sub.columns else "update"
    colors = cm.viridis(np.linspace(0.0, 0.9, len(available)))
    opt = GOAL_OPTIMAL_BITS[:n_goals]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    for gc, color, opt_b in zip(available, colors, opt):
        pivoted = sub.pivot_table(index=x_col, columns="seed", values=gc)
        if smooth > 1:
            pivoted = pivoted.rolling(smooth, min_periods=1).mean()
        xs = pivoted.index.values
        mean = pivoted.mean(axis=1).values
        std = pivoted.std(axis=1).fillna(0).values

        goal_idx = int(gc.split("_")[-1])
        label = f"Goal {goal_idx} (p={_GOAL_PROBS[goal_idx]:.3f})"
        ax1.plot(xs, mean, color=color, label=label)
        ax1.fill_between(xs, mean - std, mean + std, alpha=0.15, color=color)
        ax1.axhline(opt_b, color=color, linewidth=0.9, linestyle="--", alpha=0.6)

    # Secondary axis: success rate
    if "success_rate" in sub.columns:
        ax2 = ax1.twinx()
        sr_piv = sub.pivot_table(index=x_col, columns="seed", values="success_rate")
        if smooth > 1:
            sr_piv = sr_piv.rolling(smooth, min_periods=1).mean()
        ax2.plot(sr_piv.index.values, sr_piv.mean(axis=1).values,
                 color="black", linewidth=1.8, linestyle="-.", label="Success rate")
        ax2.set_ylabel("Success rate", color="black")
        ax2.set_ylim(-0.05, 1.08)
        ax2.tick_params(axis="y", labelcolor="black")

    ax1.set_xlabel("Environment timesteps")
    ax1.set_ylabel("Surrogate bits allocated per goal")
    ax1.set_title(
        f"{title}\n(channel={channel_filter}; dashed lines = optimal −log₂(p_i))"
    )
    ax1.legend(
        loc="upper left", framealpha=0.85, fontsize=8,
        title="Goals (dashed = optimal)",
    )
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, save_path)
    return fig, ax1


# ---------------------------------------------------------------------------
# Figure 10: δ × λ Heatmap
# ---------------------------------------------------------------------------

def plot_paper_delta_lambda_heatmap(
    summary: pd.DataFrame,
    channel: str = "sd",
    z_dim: int = 2,
    metric: str = "success_rate",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    The δ-λ interaction is non-trivial: the jointly optimal configuration
    lies strictly inside the grid (not at any boundary), confirming that both
    hyperparameters need to be tuned.  The interior optimum also validates that
    Stage A convergence criterion 1 (interior optimum) will be satisfied.

    Analysis
    --------
    2-D heatmap of mean success_rate (or bits/msg) over the (δ, λ) grid at
    fixed z_dim and channel.  Colour scale from blue (low) to yellow (high).
    The optimal cell is annotated with a star.

    Conclusion
    ----------
    An interior maximum confirms there is a real tradeoff to be found, not just
    a degenerate "bigger is always better" situation.  The heatmap also serves as
    a sanity check that the sweep grid is wide enough to capture the optimum.
    """
    _require_mpl()
    _paper_style()

    sub = summary.copy()
    if "channel" in sub.columns:
        sub = sub[sub["channel"] == channel]
    if "z_dim" in sub.columns:
        sub = sub[sub["z_dim"] == z_dim]

    if sub.empty or "delta" not in sub.columns or "lambda_comms" not in sub.columns:
        raise ValueError(
            "Need 'delta' and 'lambda_comms' in summary for this channel/z_dim combination."
        )

    pivot = sub.groupby(["delta", "lambda_comms"])[metric].mean().unstack("lambda_comms")
    vmin, vmax = pivot.values.min(), pivot.values.max()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=metric.replace("_", " "))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [f"{v:.1e}" for v in pivot.columns], rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2g}" for v in pivot.index])
    ax.set_xlabel("λ (communication penalty)")
    ax.set_ylabel("δ (quantisation bin width)")

    # Annotate optimal cell
    best_idx = np.unravel_index(pivot.values.argmax(), pivot.values.shape)
    ax.plot(best_idx[1], best_idx[0], "*", color="white", markersize=14,
            label=f"Optimal: δ={pivot.index[best_idx[0]]:.2g}, λ={pivot.columns[best_idx[1]]:.1e}")
    ax.legend(framealpha=0.85, fontsize=8)
    ax.set_title(title or f"δ × λ heatmap  ({metric}, channel={channel}, z_dim={z_dim})")
    plt.tight_layout()
    _save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Batch generation helper
# ---------------------------------------------------------------------------

def generate_all_paper_figures(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    agg: pd.DataFrame,
    out_dir: str | Path = "results/toyproblem/channel_comparison/figures",
) -> None:
    """Generate all paper figures from sweep data and save to *out_dir*.

    Parameters
    ----------
    df      : full tidy DataFrame (all seeds, one row per update)
    summary : per-seed final-metrics DataFrame from load_runs.final_metrics
    agg     : per-config aggregated DataFrame from load_runs.seed_aggregate
    out_dir : directory to write PNG files — should be the sweep's figures/ dir,
              e.g. "results/toyproblem/sweep_stage_a/figures"

    Usage
    -----
    from onpolicy.envs.toyproblem.analysis.load_runs import (
        load_sweep, final_metrics, seed_aggregate
    )
    from onpolicy.envs.toyproblem.analysis.paper_figures import generate_all_paper_figures

    df = load_sweep("runs/toyproblem/sweep_stage_a")
    summary = final_metrics(df)
    agg = seed_aggregate(summary, group_cols=["channel","lambda_comms","delta","z_dim"])
    generate_all_paper_figures(df, summary, agg, out_dir="results/toyproblem/sweep_stage_a/figures")
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Rate–distortion frontier
    try:
        plot_paper_rate_distortion(
            agg,
            x_col="true_bits_per_msg_mean" if "true_bits_per_msg_mean" in agg.columns
                   else "bits_per_msg_mean",
            save_path=out_dir / "fig1_rate_distortion.png",
        )
        print("  fig1_rate_distortion.png — OK")
    except Exception as e:
        print(f"  fig1 FAILED: {e}")

    # Figure 2: Training curves
    try:
        plot_paper_training_curves(df, save_path=out_dir / "fig2_training_curves.png")
        print("  fig2_training_curves.png — OK")
    except Exception as e:
        print(f"  fig2 FAILED: {e}")

    # Figure 3: Per-goal bit allocation
    goal_cols = [c for c in df.columns if c.startswith("bits_goal_")]
    if goal_cols:
        try:
            plot_paper_per_goal_allocation(
                df, n_goals=len(goal_cols),
                save_path=out_dir / "fig3_per_goal_allocation.png",
            )
            print("  fig3_per_goal_allocation.png — OK")
        except Exception as e:
            print(f"  fig3 FAILED: {e}")

    # Figure 4: λ sensitivity (SD channel)
    if "lambda_comms" in summary.columns and "channel" in summary.columns:
        try:
            best_delta = None
            best_zdim = None
            sd_sub = summary[summary["channel"] == "sd"]
            if not sd_sub.empty:
                best_row = sd_sub.loc[sd_sub["success_rate"].idxmax()]
                best_delta = float(best_row.get("delta", 1.0))
                best_zdim = int(best_row.get("z_dim", 2))
            plot_paper_lambda_sensitivity(
                summary, channel="sd", delta=best_delta, z_dim=best_zdim,
                save_path=out_dir / "fig4_lambda_sensitivity.png",
            )
            print("  fig4_lambda_sensitivity.png — OK")
        except Exception as e:
            print(f"  fig4 FAILED: {e}")

    # Figure 5: Channel efficiency
    try:
        plot_paper_channel_efficiency(
            summary, save_path=out_dir / "fig5_channel_efficiency.png"
        )
        print("  fig5_channel_efficiency.png — OK")
    except Exception as e:
        print(f"  fig5 FAILED: {e}")

    # Figure 6: SD vs NSD
    try:
        plot_paper_sd_nsd_comparison(
            summary, save_path=out_dir / "fig6_sd_nsd_comparison.png"
        )
        print("  fig6_sd_nsd_comparison.png — OK")
    except Exception as e:
        print(f"  fig6 FAILED: {e}")

    # Figure 7: Surrogate calibration
    if "bits_per_msg" in df.columns and "true_bits_per_msg" in df.columns:
        try:
            plot_paper_surrogate_calibration(
                df, save_path=out_dir / "fig7_surrogate_calibration.png"
            )
            print("  fig7_surrogate_calibration.png — OK")
        except Exception as e:
            print(f"  fig7 FAILED: {e}")

    # Figure 8: z_dim scaling
    if "z_dim" in summary.columns:
        try:
            plot_paper_zdim_scaling(
                summary, save_path=out_dir / "fig8_zdim_scaling.png"
            )
            print("  fig8_zdim_scaling.png — OK")
        except Exception as e:
            print(f"  fig8 FAILED: {e}")

    # Figure 9: Per-goal dynamics
    if goal_cols:
        for ch in ("sd", "nsd"):
            if "channel" not in df.columns or ch not in df["channel"].values:
                continue
            try:
                plot_paper_per_goal_dynamics(
                    df, channel_filter=ch,
                    save_path=out_dir / f"fig9_per_goal_dynamics_{ch}.png",
                )
                print(f"  fig9_per_goal_dynamics_{ch}.png — OK")
            except Exception as e:
                print(f"  fig9 ({ch}) FAILED: {e}")

    # Figure 10: δ × λ heatmap
    if "delta" in summary.columns and "lambda_comms" in summary.columns:
        for ch in ("sd", "nsd"):
            for zdim in (2,):
                try:
                    plot_paper_delta_lambda_heatmap(
                        summary, channel=ch, z_dim=zdim,
                        save_path=out_dir / f"fig10_delta_lambda_{ch}_zdim{zdim}.png",
                    )
                    print(f"  fig10_delta_lambda_{ch}_zdim{zdim}.png — OK")
                except Exception as e:
                    print(f"  fig10 ({ch}, z={zdim}) FAILED: {e}")

    print(f"\nAll figures attempted. Check {out_dir}/")


# ---------------------------------------------------------------------------
# P2 Figures — Entropy Model
# ---------------------------------------------------------------------------

def plot_p2_entropy_rate_vs_lambda(
    agg: pd.DataFrame,
    K_values: Sequence[int] = (1, 3, 5, 10, 20),
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    Higher K (mixture components) in the DLM prior reduces entropy_rate (cross-entropy
    approaches H(m)) and improves success_rate at the same lambda_comms.

    Analysis
    --------
    Line plots of entropy_rate_mean vs lambda_comms for each K value (Context A,
    factored). Shaded band = bootstrap CI. A horizontal dashed line marks the
    baseline magnitude surrogate (bits_per_msg from baseline_magnitude run).

    Conclusion
    ----------
    If K=5 matches K=10/20, the mixture is expressive enough; K=1 (Gaussian)
    is the simplest useful special case.
    """
    _require_mpl()
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    cmap = cm.get_cmap("viridis", len(K_values))
    for i, K in enumerate(K_values):
        mask = agg["entropy_model_K"] == K
        sub = agg[mask].sort_values("lambda_comms")
        if sub.empty:
            continue
        axes[0].plot(
            sub["lambda_comms"], sub["entropy_rate_mean"],
            color=cmap(i), label=f"K={K}",
        )
        if "entropy_rate_ci_lo" in sub.columns:
            axes[0].fill_between(
                sub["lambda_comms"],
                sub["entropy_rate_ci_lo"], sub["entropy_rate_ci_hi"],
                color=cmap(i), alpha=0.15,
            )
        axes[1].plot(
            sub["lambda_comms"], sub["success_rate_mean"],
            color=cmap(i), label=f"K={K}",
        )

    for ax in axes:
        ax.set_xlabel("λ (lambda_comms)")
        ax.legend(title="Mixture K")
        ax.set_xscale("log")
    axes[0].set_ylabel("Entropy rate (bits/msg)")
    axes[0].set_title("Cross-entropy vs λ")
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("Task success vs λ")
    fig.suptitle("P2: DLM mixture components ablation (Context A, factored)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_p2_qphi_gap_training(
    df: pd.DataFrame,
    exp_names: Sequence[str] | None = None,
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    qphi_gap (cross-entropy − empirical H(m)) decreases over training as q_φ
    converges to the true message distribution. Wide-scale initialisation
    prevents early collapse.

    Analysis
    --------
    Training curves of qphi_gap over timesteps, one line per exp_name seed-mean.
    Shaded band = ±1 seed std. A dashed line at 0 marks perfect fit.

    Conclusion
    ----------
    If qphi_gap converges toward 0 and stays there, q_φ is tracking p(m) well.
    Spikes indicate warm-start distribution shift (see PILLAR_P2.md §9).
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    if exp_names is None:
        exp_names = df["exp_name"].unique().tolist() if "exp_name" in df.columns else []

    for exp in exp_names:
        sub = df[df["exp_name"] == exp] if "exp_name" in df.columns else df
        if "qphi_gap" not in sub.columns:
            continue
        grouped = sub.groupby("timestep")["qphi_gap"]
        mean = grouped.mean()
        std = grouped.std()
        ax.plot(mean.index, mean.values, label=exp)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)

    ax.axhline(0, linestyle="--", color="black", linewidth=1, label="perfect fit")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("qphi_gap (bits)")
    ax.set_title("P2: q_φ convergence diagnostic")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_p2_factored_vs_joint(
    agg: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    Joint autoregressive DLM achieves lower entropy_rate than factored DLM when
    message dimensions are correlated (tc_bits > 0), but at higher computational cost.

    Analysis
    --------
    Scatter of (entropy_rate_mean, success_rate_mean) for factored vs joint K=5,
    Context A. Annotated with tc_bits value to show when dimensions are correlated.

    Conclusion
    ----------
    If tc_bits ≈ 0, factored = joint. If tc_bits > 0, joint should have strictly
    lower entropy_rate (the improvement equals tc_bits by TC decomposition).
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {"factored": "#1f77b4", "joint": "#ff7f0e"}
    for model_type, color in colors.items():
        if "entropy_model_type" not in agg.columns:
            continue
        sub = agg[agg["entropy_model_type"] == model_type]
        if sub.empty:
            continue
        ax.scatter(
            sub["entropy_rate_mean"], sub["success_rate_mean"],
            color=color, label=model_type, s=60, alpha=0.8,
        )

    ax.set_xlabel("Entropy rate (bits/msg)")
    ax.set_ylabel("Success rate")
    ax.set_title("P2: Factored vs joint DLM (K=5, Context A)")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_p2_context_comparison(
    agg: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    entropy_rate_A ≥ entropy_rate_B ≥ H(m). The gap A−B quantifies I(z; m).

    Analysis
    --------
    Box plots (or bar plots with error bars) of entropy_rate for contexts A and B
    (K=5, factored). A horizontal dashed line at H_m_empirical mean shows the
    theoretical minimum.

    Conclusion
    ----------
    A small gap A−B means z carries little extra information about m beyond the
    marginal distribution — Context A is nearly optimal.
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    contexts = ["A", "B"]
    colors = {"A": "#1f77b4", "B": "#ff7f0e"}
    x_pos = list(range(len(contexts)))
    for i, ctx in enumerate(contexts):
        if "entropy_model_context" not in agg.columns:
            continue
        sub = agg[agg["entropy_model_context"] == ctx]
        if sub.empty:
            continue
        er = sub["entropy_rate_mean"]
        ci_lo = sub.get("entropy_rate_ci_lo", er)
        ci_hi = sub.get("entropy_rate_ci_hi", er)
        ax.bar(i, er.mean(), color=colors[ctx], label=f"Context {ctx}", alpha=0.8)
        ax.errorbar(i, er.mean(),
                    yerr=[[er.mean() - ci_lo.mean()], [ci_hi.mean() - er.mean()]],
                    color="black", capsize=5)

    # H(m) reference line
    if "H_m_empirical_mean" in agg.columns:
        h_emp = agg["H_m_empirical_mean"].mean()
        ax.axhline(h_emp, linestyle="--", color="black", label="H(m) empirical")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Context {c}" for c in contexts])
    ax.set_ylabel("Entropy rate (bits/msg)")
    ax.set_title("P2: Rate bound tightness by conditioning context")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax
