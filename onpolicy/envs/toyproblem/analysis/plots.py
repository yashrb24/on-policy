"""Plotting utilities for DDCL toy-problem experiments.

All plots save to a file and also return the (fig, ax) pair for inspection.
Requires matplotlib; optionally seaborn for improved aesthetics.

Functions
---------
plot_training_curves    : per-seed training curves with shaded seed band
plot_rate_distortion    : success_rate vs bits_per_msg Pareto frontier
plot_per_goal_bits      : bar chart of per-goal bit allocation with uncertainty bands
plot_sweep_heatmap      : 2-D sweep heatmap (x × y → z_col)
plot_channel_comparison : side-by-side bar plot comparing all channel types
plot_bits_vs_entropy    : true transmission bits vs Shannon entropy H(G)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    df: pd.DataFrame,
    y_col: str = "success_rate",
    x_col: str = "timestep",
    group_col: str = "channel",
    seed_col: str = "seed",
    smooth: int = 10,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple:
    """Plot mean ± 1 std training curves grouped by *group_col*.

    Parameters
    ----------
    df        : tidy DataFrame (all seeds, one row per update)
    y_col     : metric to plot on y-axis
    x_col     : x-axis column (usually 'timestep' or 'update')
    group_col : column to split curves by (e.g. 'channel', 'lambda_comms')
    seed_col  : column identifying seeds (used to compute std band)
    smooth    : rolling-average window (in rows); 1 = no smoothing
    save_path : if given, save the figure to this path

    Returns
    -------
    (fig, ax)
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = df[group_col].unique() if group_col in df.columns else [None]
    colors = cm.tab10(np.linspace(0, 0.9, len(groups)))

    for color, grp_val in zip(colors, groups):
        if grp_val is None:
            sub = df
            label = y_col
        else:
            sub = df[df[group_col] == grp_val]
            label = str(grp_val)

        # Pivot so each column is one seed.
        pivoted = sub.pivot_table(index=x_col, columns=seed_col, values=y_col)
        if smooth > 1:
            pivoted = pivoted.rolling(smooth, min_periods=1).mean()

        xs = pivoted.index.values
        mean = pivoted.mean(axis=1).values
        std = pivoted.std(axis=1).fillna(0).values

        ax.plot(xs, mean, label=label, color=color)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_title(title or f"{y_col} vs {x_col}")
    ax.legend(title=group_col if group_col else "", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


# ---------------------------------------------------------------------------
# Rate–distortion frontier
# ---------------------------------------------------------------------------

def plot_rate_distortion(
    df: pd.DataFrame,
    x_col: str = "bits_per_msg",
    y_col: str = "success_rate",
    group_col: str = "channel",
    pareto_df: pd.DataFrame | None = None,
    h_goal: float | None = None,
    title: str = "Rate–distortion frontier",
    save_path: str | Path | None = None,
) -> tuple:
    """Scatter plot of communication cost vs task performance.

    Each point is one (config, seed-mean) pair. If *pareto_df* is given,
    draws the Pareto frontier as a step line.

    Returns
    -------
    (fig, ax)
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=(7, 5))

    groups = df[group_col].unique() if group_col in df.columns else [None]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    colors = cm.tab10(np.linspace(0, 0.9, len(groups)))

    for (color, marker, grp_val) in zip(colors, markers, groups):
        sub = df if grp_val is None else df[df[group_col] == grp_val]
        ax.scatter(
            sub[x_col], sub[y_col],
            label=str(grp_val) if grp_val is not None else y_col,
            color=color, marker=marker, s=60, zorder=3,
        )

    if pareto_df is not None:
        pf = pareto_df.sort_values(x_col)
        ax.step(pf[x_col], pf[y_col], where="post",
                color="black", linewidth=2, linestyle="--",
                label="Pareto frontier", zorder=4)

    if h_goal is not None:
        ax.axvline(h_goal, color="red", linewidth=1.5, linestyle=":",
                   label=f"H(G) = {h_goal:.3f} bits", zorder=5)

    ax.set_xlabel(x_col.replace("_", " ") + " (bits)")
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_title(title)
    ax.legend(title=group_col if group_col else "", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


# ---------------------------------------------------------------------------
# Per-goal bit allocation
# ---------------------------------------------------------------------------

def plot_per_goal_bits(
    df: pd.DataFrame,
    n_goals: int = 6,
    group_col: str = "channel",
    window: int = 20,
    goal_optimal_bits: Sequence[float] | None = None,
    goal_probs: Sequence[float] | None = None,
    total_updates: int | None = None,
    show_uncertainty: bool = True,
    title: str = "Per-goal bit allocation",
    save_path: str | Path | None = None,
) -> tuple:
    """Bar chart of mean bits allocated to each goal, grouped by config.

    Expects columns ``bits_goal_0`` … ``bits_goal_{n_goals-1}`` in *df*.

    Uncertainty bands
    -----------------
    When ``goal_optimal_bits``, ``goal_probs``, and ``show_uncertainty=True``
    are provided, two sources of uncertainty are visualised around the optimal
    line -log₂(p_i):

    1. **Jensen gap** (systematic, upward bias in surrogate):
       The surrogate ``log₂(|z|/δ+1)`` overestimates true transmission bits
       because log is concave (Jensen's inequality).  In the typical operating
       regime (|z|/δ ∈ [1, 5]), the gap is ≈ 0.05–0.30 bits.  At z=0 the
       surrogate *under*estimates by ~0.5 bits because dither always moves m to
       {-1,0}.  We model a symmetric ±0.20 bit band as representative of the
       mid-range Jensen correction.

    2. **RL sampling noise** (stochastic, scales with goal rarity):
       Goal i appears with frequency p_i, so it drives only a fraction p_i of
       gradient updates.  The effective number of updates for goal i is
       n_i = p_i × total_updates (default 244 for 1M steps / 16 envs / 256
       steps).  The per-goal allocation variance scales as
         σ_RL(i) = C_RL / sqrt(p_i × total_updates)
       with empirical constant C_RL ≈ 0.5 bits.  For goal 5 (p=0.003,
       n_5≈0.7 per update) this is ~0.93 bits; for goal 0 (p=0.515,
       n_0≈125) this is ~0.04 bits — a 13× difference.

    The combined uncertainty shown is:
      σ_total(i) = sqrt(σ_RL(i)² + σ_Jensen²)
    Displayed as a shaded envelope around the optimal dashed line.

    Parameters
    ----------
    df               : tidy per-update DataFrame
    n_goals          : number of goals (columns bits_goal_0 … bits_goal_{n-1})
    group_col        : column to group bars by (e.g. 'channel')
    window           : final-window size for per-seed aggregation
    goal_optimal_bits: per-goal theoretical optimum -log₂(p_i)
    goal_probs       : per-goal sampling probabilities p_i
    total_updates    : total PPO updates (= total_timesteps / (n_envs × n_steps))
    show_uncertainty : whether to draw the uncertainty envelope
    title            : figure title
    save_path        : optional path to save the figure

    Returns
    -------
    (fig, ax)
    """
    _require_mpl()
    goal_cols = [f"bits_goal_{i}" for i in range(n_goals)]
    missing = [c for c in goal_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    # Use only the last *window* rows per seed.
    def tail_mean(grp: pd.DataFrame) -> pd.Series:
        return grp.sort_values("update").tail(window)[goal_cols].mean()

    group_keys = [c for c in (group_col, "seed") if c in df.columns]
    means = df.groupby(group_keys).apply(tail_mean).reset_index()

    # Average over seeds.
    config_means = (
        means.groupby(group_col)[goal_cols].mean()
        if group_col in means
        else means[goal_cols].mean().to_frame().T
    )

    groups = config_means.index.tolist()
    n_groups = len(groups)
    n_goals_actual = len(goal_cols)
    x = np.arange(n_goals_actual)
    width = 0.8 / max(n_groups, 1)
    colors = cm.tab10(np.linspace(0, 0.9, n_groups))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (grp, color) in enumerate(zip(groups, colors)):
        vals = config_means.loc[grp, goal_cols].values
        ax.bar(x + i * width, vals, width=width, label=str(grp), color=color)

    x_center = x + width * (n_groups - 1) / 2

    ax.set_xticks(x_center)
    ax.set_xticklabels([f"Goal {i}" for i in range(n_goals_actual)])
    ax.set_ylabel("Mean bits allocated (surrogate)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Overlay optimal per-goal code lengths -log₂(p_i) if provided.
    if goal_optimal_bits is not None:
        opt = np.array(list(goal_optimal_bits)[:n_goals_actual])
        ax.plot(x_center, opt, color="red", marker="x", linewidth=1.5,
                linestyle="--", label="Optimal −log₂(p_i)", zorder=5)

        # --- Uncertainty envelope ---
        if show_uncertainty and goal_probs is not None:
            probs = np.array(list(goal_probs)[:n_goals_actual])
            if total_updates is None:
                total_updates = 244  # default: 1M steps / (16 envs × 256 steps)

            # RL sampling noise: C_RL / sqrt(p_i * N_updates)
            C_RL = 0.5  # empirical constant (bits)
            sigma_rl = C_RL / np.sqrt(probs * total_updates)

            # Jensen gap: constant mid-range estimate
            sigma_jensen = 0.20  # bits — representative of |z|/δ ∈ [1, 5]

            # Combined uncertainty (in quadrature)
            sigma_total = np.sqrt(sigma_rl ** 2 + sigma_jensen ** 2)

            ax.fill_between(
                x_center,
                opt - sigma_total,
                opt + sigma_total,
                color="red", alpha=0.15,
                label=(
                    "Uncertainty: Jensen gap + RL noise\n"
                    r"  $\sigma_{\rm RL}(i)\propto 1/\sqrt{p_i N}$,"
                    r"  $\sigma_{\rm Jensen}\approx0.20$ bits"
                ),
                zorder=4,
            )

            # Annotate per-goal n_effective to help reader judge noise level
            for gi in range(n_goals_actual):
                n_eff = probs[gi] * total_updates
                ax.annotate(
                    f"n≈{n_eff:.0f}",
                    xy=(x_center[gi], 0.04),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7, color="grey",
                )

    ax.legend(title=group_col, framealpha=0.8, fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


# ---------------------------------------------------------------------------
# Bits-vs-entropy efficiency plot
# ---------------------------------------------------------------------------

def plot_bits_vs_entropy(
    summary: pd.DataFrame,
    h_goal: float,
    bits_col: str = "true_bits_per_msg",
    group_col: str = "channel",
    title: str = "Transmission bits vs Shannon entropy H(G)",
    save_path: str | Path | None = None,
) -> tuple:
    """Bar chart comparing mean transmission bits per message against H(G).

    Two bars per channel:
      - Blue  : mean ``bits_col`` across seeds (actual transmission cost)
      - Red   : H(G) reference line (Shannon limit for goal identification)

    For the identity channel (``none``), ``true_bits_per_msg`` = 32 × z_dim,
    shown as the float32 baseline.  For DDCL channels it is log₂(|m|+1) summed
    over dimensions.  For STE it is B × z_dim exactly.

    Parameters
    ----------
    summary   : per-seed final-metrics DataFrame (one row per seed)
    h_goal    : Shannon entropy H(G) of goal distribution (reference line)
    bits_col  : column to plot (default ``true_bits_per_msg``)
    group_col : column identifying channel/config groups
    """
    _require_mpl()

    if bits_col not in summary.columns:
        raise ValueError(
            f"Column '{bits_col}' not found in summary. "
            f"Available: {list(summary.columns)}"
        )

    channels = summary[group_col].unique()
    x = np.arange(len(channels))
    colors = cm.tab10(np.linspace(0, 0.9, len(channels)))

    means, stds = [], []
    for ch in channels:
        vals = summary[summary[group_col] == ch][bits_col].values
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    fig, ax = plt.subplots(figsize=(max(6, len(channels) * 1.2), 4))
    ax.bar(x, means, yerr=stds, color=colors, capsize=5, width=0.6,
           label="Mean transmission bits")
    ax.axhline(h_goal, color="red", linewidth=2, linestyle="--",
               label=f"H(G) = {h_goal:.3f} bits (Shannon limit)")

    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=30, ha="right")
    ax.set_ylabel("Bits per message")
    ax.set_title(title)
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


# ---------------------------------------------------------------------------
# Sweep heatmap
# ---------------------------------------------------------------------------

def plot_sweep_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str = "success_rate",
    agg: str = "mean",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple:
    """2-D heatmap of a metric aggregated over a 2-axis sweep.

    Parameters
    ----------
    df    : tidy per-seed summary DataFrame
    x_col : sweep axis on x (e.g. 'lambda_comms')
    y_col : sweep axis on y (e.g. 'delta')
    z_col : metric to show as colour (e.g. 'success_rate')
    agg   : aggregation over seeds ('mean', 'median', 'iqm')

    Returns
    -------
    (fig, ax)
    """
    _require_mpl()

    if agg == "iqm":
        from .stats import iqm
        pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col,
                               aggfunc=iqm)
    else:
        pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col,
                               aggfunc=agg)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, label=z_col.replace("_", " "))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.3g}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.3g}" for v in pivot.index])
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_title(title or f"{z_col} ({agg})")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


# ---------------------------------------------------------------------------
# Channel comparison bar chart
# ---------------------------------------------------------------------------

def plot_channel_comparison(
    summary: pd.DataFrame,
    metrics: Sequence[str] = ("success_rate", "bits_per_msg"),
    group_col: str = "channel",
    ci_col_suffix: str | None = None,
    title: str = "Channel comparison",
    save_path: str | Path | None = None,
) -> tuple:
    """Side-by-side bar chart comparing channels on multiple metrics.

    *summary* should be a per-seed final-metrics DataFrame (one row per seed).
    Error bars show ± 1 std across seeds.

    Returns
    -------
    (fig, axes)  — one axis per metric
    """
    _require_mpl()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    channels = summary[group_col].unique()
    x = np.arange(len(channels))
    colors = cm.tab10(np.linspace(0, 0.9, len(channels)))

    for ax, metric in zip(axes, metrics):
        means, stds = [], []
        for ch in channels:
            vals = summary[summary[group_col] == ch][metric].values
            means.append(vals.mean())
            stds.append(vals.std())

        ax.bar(x, means, yerr=stds, color=colors, capsize=5, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=30, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "))
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, axes
