"""Publication-quality figures for the DDCL toy-problem paper.

Structure
---------
MAIN PAPER (2 figures, minimal ink, standalone):
  plot_paper_rate_distortion    – Fig 1: core rate-distortion result
  plot_paper_channel_comparison – Fig 2: best-per-channel head-to-head

APPENDIX (8 figures, hyperparameter transparency):
  plot_appendix_lambda_sensitivity    – App A: λ sweep, SR + bits
  plot_appendix_delta_lambda_heatmap  – App B: δ×λ heatmap (pcolormesh)
  plot_appendix_training_dynamics     – App C: SR + bits over training
  plot_appendix_per_goal_allocation   – App D: per-goal bit allocation
  plot_appendix_overhead_above_hg     – App E: bits overhead above H(G)
  plot_appendix_surrogate_calibration – App F: surrogate vs true bits
  plot_appendix_zdim_scaling          – App G: z_dim effect on SR and bits
  plot_appendix_sd_nsd_comparison     – App H: SD vs NSD head-to-head

Batch entry point:
  generate_sweep_figures(df, summary, agg, out_dir)

Each function carries a docstring with:
  Category  : MAIN PAPER or APPENDIX, and which section
  Hypothesis: the scientific claim being tested
  How to read: what pattern confirms / refutes the hypothesis
  Why included: what decision or claim in the paper this backs

Calling convention:
  All functions accept either df (full tidy training data) or summary
  (per-seed final-metrics) or agg (per-config seed-aggregated), as
  documented per function.  All save as both .pdf and .png.
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
from onpolicy.envs.toyproblem.analysis.stats import pareto_frontier, bootstrap_ci


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_PAPER_RCPARAMS = {
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "lines.linewidth":  1.8,
    "figure.dpi":       200,
    "axes.spines.top":  False,
    "axes.spines.right": False,
}

# Consistent channel colours and markers across all figures.
_CHANNEL_COLORS = {
    "sd":               "#1f77b4",   # blue
    "nsd":              "#ff7f0e",   # orange
    "additive_uniform": "#2ca02c",   # green
    "none":             "#7f7f7f",   # grey
}
_CHANNEL_LABELS = {
    "sd":               "SD (subtractive dither)",
    "nsd":              "NSD (TPDF dither)",
    "additive_uniform": "Additive uniform",
    "none":             "Float32 passthrough",
}
_CHANNEL_MARKERS = {
    "sd":               "o",
    "nsd":              "s",
    "additive_uniform": "^",
    "none":             "D",
}

# Baseline-best config from Phase 2 Stage A sweep freeze.
_BASELINE_BEST = {"channel": "sd", "delta": 1.0, "lambda_comms": 5e-4, "z_dim": 2}

# True cost of sending one float32 value.
_FLOAT32_BITS_PER_DIM = 32


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError("matplotlib is required. pip install matplotlib")


def _paper_style() -> None:
    plt.rcParams.update(_PAPER_RCPARAMS)


def _save(fig: "plt.Figure", out_dir: Path, stem: str) -> None:
    """Save figure as both .pdf (paper) and .png (preview)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")


def _sem_ci(std: float, n: int, z: float = 1.96) -> float:
    """Approximate 95% CI half-width: z * std / sqrt(n)."""
    return z * std / math.sqrt(max(n, 1))


def _best_per_channel(
    agg: pd.DataFrame,
    sr_col: str = "success_rate_mean",
    bits_col: str = "true_bits_per_msg_mean",
    channel_col: str = "channel",
    channels: Sequence[str] = ("sd", "nsd", "additive_uniform"),
    z_dim: int | None = None,
) -> pd.DataFrame:
    """Return one row per channel: the Pareto-optimal config with highest SR
    then lowest bits.  Optionally filter to a fixed z_dim first.

    Returns a DataFrame indexed by channel with all agg columns present.
    """
    sub = agg.copy()
    if z_dim is not None and "z_dim" in sub.columns:
        sub = sub[sub["z_dim"] == z_dim]
    rows = []
    for ch in channels:
        ch_df = sub[sub[channel_col] == ch]
        if ch_df.empty:
            continue
        # Sort: highest SR first, then lowest bits as tiebreak.
        best = ch_df.sort_values(
            [sr_col, bits_col], ascending=[False, True]
        ).iloc[0]
        rows.append(best)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# MAIN PAPER — Figure 1: Rate-Distortion Frontier
# ---------------------------------------------------------------------------

def plot_paper_rate_distortion(
    agg: pd.DataFrame,
    sr_col: str = "success_rate_mean",
    bits_col: str = "true_bits_per_msg_mean",
    sr_std_col: str = "success_rate_std",
    bits_std_col: str = "true_bits_per_msg_std",
    n_col: str = "n_seeds",
    channel_col: str = "channel",
    title: str = "Rate–distortion frontier",
    save_path: str | Path | None = None,   # legacy; prefer out_dir+stem
    out_dir: str | Path | None = None,
    stem: str = "fig1_rate_distortion",
) -> tuple:
    """
    Category: MAIN PAPER — Figure 1 (Core result)

    Hypothesis
    ----------
    DDCL channels (SD, NSD) lie on the Pareto frontier of task success rate
    vs true transmission bits.  The additive-uniform channel achieves
    comparable SR but at higher bit cost (no second-order Schuchman property).
    The float32 passthrough (none) achieves SR≈1.0 but at 64 bits/msg —
    35× above H(G)=1.81 bits.

    How to read
    -----------
    Better operating points are in the upper-left (high SR, few bits).
    Grey background dots = all non-Pareto configs from the sweep (show
    the full search space).  Coloured markers = Pareto-optimal configs;
    their error bars are 95% CI across seeds.  The black dashed step line
    traces the Pareto frontier for DDCL + additive channels (none excluded).
    The vertical red dotted line = H(G): the Shannon lower bound for goal
    identification.  The vertical grey dashed line = float32 cost (64 bits
    at z_dim=2).

    Why included
    ------------
    This is the single most important figure in the paper: it directly
    shows that DDCL achieves near-Shannon efficiency while float32 wastes
    35× more bandwidth.  Pareto dominance by SD over additive-uniform
    demonstrates the value of the second-order Schuchman property.
    NSD not reaching SR=1.0 at z_dim=2 (it needs z_dim=3) is consistent
    with its 3× higher reconstruction noise variance (δ²/4 vs δ²/12).
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    # ── Separate channels ────────────────────────────────────────────────────
    ddcl_channels = ["sd", "nsd", "additive_uniform"]
    ddcl = agg[agg[channel_col].isin(ddcl_channels)].copy()
    none_df = agg[agg[channel_col] == "none"].copy()

    # ── Pareto frontier (DDCL + additive only; none excluded) ────────────────
    if not ddcl.empty and bits_col in ddcl.columns and sr_col in ddcl.columns:
        pf = pareto_frontier(
            ddcl, x_col=sr_col, y_col=bits_col,
            x_better="higher", y_better="lower",
        )
        pf_index = set(pf.index)
    else:
        pf = pd.DataFrame()
        pf_index = set()

    # ── Background: all non-Pareto DDCL configs ───────────────────────────────
    non_pf = ddcl[~ddcl.index.isin(pf_index)]
    if not non_pf.empty:
        ax.scatter(
            non_pf[bits_col], non_pf[sr_col],
            s=12, color="lightgrey", alpha=0.5, zorder=2,
            label="_nolegend_",
        )

    # ── Pareto configs, coloured by channel ───────────────────────────────────
    has_n = n_col in pf.columns if not pf.empty else False
    for ch in ddcl_channels:
        if pf.empty:
            break
        ch_pf = pf[pf[channel_col] == ch]
        if ch_pf.empty:
            continue
        color = _CHANNEL_COLORS[ch]
        marker = _CHANNEL_MARKERS[ch]
        label = _CHANNEL_LABELS[ch]

        # 95% CI half-widths
        if sr_std_col in ch_pf.columns and bits_std_col in ch_pf.columns:
            n_vals = ch_pf[n_col].values if has_n else np.full(len(ch_pf), 5)
            xerr = [_sem_ci(s, n) for s, n in zip(ch_pf[bits_std_col], n_vals)]
            yerr = [_sem_ci(s, n) for s, n in zip(ch_pf[sr_std_col], n_vals)]
        else:
            xerr, yerr = None, None

        ax.errorbar(
            ch_pf[bits_col], ch_pf[sr_col],
            xerr=xerr, yerr=yerr,
            fmt=marker, color=color, markersize=9,
            capsize=3, capthick=1.2, elinewidth=1.0,
            label=label, zorder=5, alpha=0.95,
        )

    # ── Pareto step line ──────────────────────────────────────────────────────
    if not pf.empty:
        pf_sorted = pf.sort_values(bits_col)
        ax.step(
            pf_sorted[bits_col], pf_sorted[sr_col],
            where="post", color="black", linewidth=1.8,
            linestyle="--", label="Pareto frontier", zorder=4, alpha=0.75,
        )

    # ── Float32 reference (none channel, z_dim=2) ─────────────────────────────
    float32_bits = None
    if not none_df.empty:
        none_z2 = none_df[none_df.get("z_dim", pd.Series(dtype=int)) == 2] \
            if "z_dim" in none_df.columns else none_df
        if none_z2.empty:
            none_z2 = none_df
        float32_bits = float(none_z2[bits_col].iloc[0]) if not none_z2.empty else None

    if float32_bits is not None:
        ax.axvline(
            float32_bits, color=_CHANNEL_COLORS["none"],
            linewidth=1.4, linestyle="-.",
            label=f"Float32 ({float32_bits:.0f} bits, z_dim=2)",
            zorder=3, alpha=0.7,
        )
        # Show the none point itself
        none_sr = float(none_z2[sr_col].iloc[0]) if not none_z2.empty else 1.0
        ax.scatter(
            float32_bits, none_sr,
            marker=_CHANNEL_MARKERS["none"],
            s=100, color=_CHANNEL_COLORS["none"],
            label=_CHANNEL_LABELS["none"], zorder=6,
            facecolors="none", edgecolors=_CHANNEL_COLORS["none"], linewidths=2.0,
        )

    # ── H(G) Shannon lower bound ──────────────────────────────────────────────
    ax.axvline(
        H_GOAL_BITS, color="red", linewidth=1.5, linestyle=":",
        label=f"H(G) = {H_GOAL_BITS:.2f} bits (Shannon limit)",
        zorder=6,
    )

    ax.set_xlabel("True transmission bits / message")
    ax.set_ylabel("Task success rate")
    ax.set_title(title)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlim(left=0)
    ax.legend(framealpha=0.88, fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.2, axis="both")
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# MAIN PAPER — Figure 2: Channel comparison at best config per channel
# ---------------------------------------------------------------------------

def plot_paper_channel_comparison(
    summary: pd.DataFrame,
    sr_col: str = "success_rate",
    bits_col: str = "true_bits_per_msg",
    z_dim_fixed: int = 2,
    title: str = "Best per-channel comparison (z_dim=2)",
    out_dir: str | Path | None = None,
    stem: str = "fig2_channel_comparison",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Category: MAIN PAPER — Figure 2 (Baseline comparison)

    Hypothesis
    ----------
    At the best (λ, δ) configuration for each channel (fixed z_dim=2):
    SD achieves the highest SR at the lowest bit cost.  Additive-uniform
    achieves comparable SR but at higher bit cost (lacks Schuchman property).
    NSD does not reach SR=1.0 at z_dim=2 due to its 3× higher noise
    variance (δ²/4 vs δ²/12) — needs z_dim=3 for parity with SD.
    Float32 uses 64 bits (35× over H(G)) while achieving SR=1.0.

    How to read
    -----------
    Left panel: success rate per channel (higher is better; H(G) is
    irrelevant here since SR is the task metric).
    Right panel: true_bits_per_msg per channel (lower is better).
    Red dashed line on right panel = H(G) theoretical minimum.
    Error bars = 95% CI across 5 seeds.
    Each channel is shown at the best (λ, δ) for z_dim=2 — not averaged
    over all hyperparameters.

    Why included
    ------------
    Provides the fair, fixed-capacity head-to-head comparison that the
    rate-distortion scatter cannot show cleanly.  The SR shortfall of NSD
    at z_dim=2 motivates the NSD vs SD analysis in the appendix (App H).
    """
    _require_mpl()
    _paper_style()

    channels = ["sd", "nsd", "additive_uniform", "none"]
    labels = [_CHANNEL_LABELS[c] for c in channels]
    colors = [_CHANNEL_COLORS[c] for c in channels]

    # For each channel, pick the (lambda, delta) config with highest mean SR
    # at z_dim=2.  For `none`, lambda=0 and delta=1.0 (only option).
    sub = summary[summary["z_dim"] == z_dim_fixed].copy() \
        if "z_dim" in summary.columns else summary.copy()

    best_rows: dict[str, pd.DataFrame] = {}
    for ch in channels:
        ch_df = sub[sub["channel"] == ch] if "channel" in sub.columns else sub
        if ch_df.empty:
            continue
        # Group by (delta, lambda) → pick max mean SR, then min mean bits.
        grp_cols = [c for c in ("delta", "lambda_comms") if c in ch_df.columns]
        if grp_cols:
            per_config = ch_df.groupby(grp_cols).agg(
                sr_mean=(sr_col, "mean"),
                bits_mean=(bits_col, "mean"),
                sr_std=(sr_col, "std"),
                bits_std=(bits_col, "std"),
                n=("seed" if "seed" in ch_df.columns else sr_col, "count"),
            ).reset_index()
            best_cfg = per_config.sort_values(
                ["sr_mean", "bits_mean"], ascending=[False, True]
            ).iloc[0]
            mask = ch_df[grp_cols[0]] == best_cfg[grp_cols[0]]
            for g in grp_cols[1:]:
                mask &= ch_df[g] == best_cfg[g]
            best_rows[ch] = ch_df[mask]
        else:
            best_rows[ch] = ch_df

    # ── Build arrays ──────────────────────────────────────────────────────────
    sr_means, sr_cis, bits_means, bits_cis = [], [], [], []
    annotations: list[str] = []  # config description for each bar

    for ch in channels:
        df_ch = best_rows.get(ch, pd.DataFrame())
        if df_ch.empty:
            sr_means.append(np.nan)
            sr_cis.append(0)
            bits_means.append(np.nan)
            bits_cis.append(0)
            annotations.append("")
            continue
        sr_vals = df_ch[sr_col].values
        bits_vals = df_ch[bits_col].values
        n = len(sr_vals)
        sr_means.append(float(np.mean(sr_vals)))
        sr_cis.append(_sem_ci(float(np.std(sr_vals)), n))
        bits_means.append(float(np.mean(bits_vals)))
        bits_cis.append(_sem_ci(float(np.std(bits_vals)), n))
        # Annotation: λ and δ of the chosen config
        if "lambda_comms" in df_ch.columns and "delta" in df_ch.columns:
            lam = df_ch["lambda_comms"].iloc[0]
            dlt = df_ch["delta"].iloc[0]
            annotations.append(f"δ={dlt:.1g}, λ={lam:.1g}")
        else:
            annotations.append(f"n={n}")

    x = np.arange(len(channels))
    width = 0.6
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, means, cis, ylabel, add_hg in [
        (axes[0], sr_means, sr_cis, "Task success rate", False),
        (axes[1], bits_means, bits_cis, "True bits / message", True),
    ]:
        bars = ax.bar(
            x, means, yerr=cis, color=colors,
            capsize=5, width=width, error_kw={"elinewidth": 1.4},
        )
        # Annotate bars with config info
        for bar, ann in zip(bars, annotations):
            if ann:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (cis[x[0]] if cis[x[0]] else 0) + 0.02,
                    ann, ha="center", va="bottom", fontsize=7, color="dimgrey",
                )
        if add_hg:
            ax.axhline(
                H_GOAL_BITS, color="red", linewidth=1.5, linestyle="--",
                label=f"H(G) = {H_GOAL_BITS:.2f} bits",
            )
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(title)
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


# ---------------------------------------------------------------------------
# APPENDIX A — Lambda sensitivity
# ---------------------------------------------------------------------------

def plot_appendix_lambda_sensitivity(
    summary: pd.DataFrame,
    channels: Sequence[str] = ("sd", "nsd"),
    delta_fixed: float = 1.0,
    z_dim_fixed: int = 2,
    sr_col: str = "success_rate",
    bits_col: str = "true_bits_per_msg",
    title: str = "λ sensitivity: SR and bits vs communication penalty",
    out_dir: str | Path | None = None,
    stem: str = "appA_lambda_sensitivity",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Category: APPENDIX A — Hyperparameter transparency

    Hypothesis
    ----------
    There exists a λ* (Pareto knee) such that λ < λ* wastes bits without
    SR cost, and λ > λ* causes SR to drop.  The knee identifies the optimal
    operating point.  SD reaches λ* at lower λ than NSD because SD's lower
    noise variance allows a more precise code at the same SR.

    How to read
    -----------
    Dual-axis plot: blue left axis = success rate, orange right axis =
    true bits/msg.  x-axis is λ on a symlog scale (0 shown as leftmost
    tick; log spacing for λ>0).  Green vertical dashed line = λ* (largest
    λ still maintaining SR ≥ 0.95).  Red horizontal dashed = H(G).
    Bands = ±1 std over seeds.  SD and NSD are on the same axes for
    direct comparison.

    Why included
    ------------
    Justifies the λ=5×10⁻⁴ choice for the baseline-best config.  Shows
    the reader the full rate-distortion tradeoff under λ variation, which
    is the primary design knob in DDCL.  Required for reproducibility.
    """
    _require_mpl()
    _paper_style()

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    has_data = False
    for ch in channels:
        sub = summary.copy()
        for col, val in [("channel", ch), ("delta", delta_fixed), ("z_dim", z_dim_fixed)]:
            if col in sub.columns:
                sub = sub[sub[col] == val]
        if sub.empty or "lambda_comms" not in sub.columns:
            continue
        has_data = True

        lambdas = sorted(sub["lambda_comms"].unique())
        sr_m, sr_s, b_m, b_s = [], [], [], []
        for lam in lambdas:
            g = sub[sub["lambda_comms"] == lam]
            sr_m.append(g[sr_col].mean()); sr_s.append(g[sr_col].std())
            b_m.append(g[bits_col].mean()); b_s.append(g[bits_col].std())

        lam_arr = np.array(lambdas, dtype=float)
        sr_m = np.array(sr_m); sr_s = np.array(sr_s)
        b_m = np.array(b_m);   b_s = np.array(b_s)
        color = _CHANNEL_COLORS.get(ch, "#333333")
        label = _CHANNEL_LABELS.get(ch, ch)

        ax1.plot(lam_arr, sr_m, color=color, marker=_CHANNEL_MARKERS.get(ch, "o"),
                 markersize=6, label=f"{label} SR")
        ax1.fill_between(lam_arr, sr_m - sr_s, sr_m + sr_s, alpha=0.12, color=color)

        ax2.plot(lam_arr, b_m, color=color, marker=_CHANNEL_MARKERS.get(ch, "o"),
                 markersize=6, linestyle="--", label=f"{label} bits")
        ax2.fill_between(lam_arr, b_m - b_s, b_m + b_s, alpha=0.10, color=color)

        # Pareto knee: largest λ where SR ≥ 0.95
        knee_mask = sr_m >= 0.95
        if knee_mask.any():
            knee_lam = lam_arr[knee_mask][-1]
            ax1.axvline(knee_lam, color=color, linewidth=1.2, linestyle=":",
                        alpha=0.6, label=f"{label} λ*={knee_lam:.1e}")

    if not has_data:
        ax1.text(0.5, 0.5, "No data for selected channels/config",
                 ha="center", va="center", transform=ax1.transAxes)
        return fig, (ax1, ax2)

    ax2.axhline(H_GOAL_BITS, color="red", linewidth=1.2, linestyle=":",
                label=f"H(G)={H_GOAL_BITS:.2f} b")
    ax1.set_xscale("symlog", linthresh=1e-6)
    ax1.set_xlabel(f"λ (communication penalty)  [δ={delta_fixed}, z_dim={z_dim_fixed}]")
    ax1.set_ylabel("Success rate", color="dimgrey")
    ax2.set_ylabel("True bits / message", color="dimgrey")
    ax1.set_ylim(-0.05, 1.08)

    # Combined legend
    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8, framealpha=0.85,
               loc="center right")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# APPENDIX B — Delta × Lambda heatmap
# ---------------------------------------------------------------------------

def plot_appendix_delta_lambda_heatmap(
    summary: pd.DataFrame,
    channel: str = "sd",
    z_dim_fixed: int = 2,
    metric: str = "success_rate",
    title: str | None = None,
    out_dir: str | Path | None = None,
    stem: str = "appB_delta_lambda_heatmap",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Category: APPENDIX B — Hyperparameter transparency

    Hypothesis
    ----------
    The δ-λ interaction produces an interior optimum: the jointly optimal
    (δ*, λ*) lies strictly inside the swept grid.  Coarse δ (large) with
    small λ uses few bits but fails the task; fine δ (small) with large λ
    over-compresses; the sweet spot is interior.

    How to read
    -----------
    Each cell = mean success_rate over seeds at (δ, λ).  Yellow = high
    (good); purple = low (poor).  White star = best cell.  λ axis is
    symlog so the λ=0 column and the log-spaced non-zero values are both
    visible with correct spacing.  Confirm the star is not at a grid edge
    (interior optimum criterion for the baseline freeze).

    Why included
    ------------
    Validates the Phase 2 convergence gate criterion 1 (interior optimum).
    Provides complete hyperparameter transparency for reproducibility.
    Fixing this figure in the appendix frees the main paper figure from
    showing all λ/δ points.
    """
    _require_mpl()
    _paper_style()

    sub = summary.copy()
    for col, val in [("channel", channel), ("z_dim", z_dim_fixed)]:
        if col in sub.columns:
            sub = sub[sub[col] == val]
    if sub.empty:
        raise ValueError(f"No data for channel={channel}, z_dim={z_dim_fixed}")

    pivot = sub.groupby(["delta", "lambda_comms"])[metric].mean().unstack("lambda_comms")
    deltas = pivot.index.values
    lambdas = pivot.columns.values

    fig, ax = plt.subplots(figsize=(9, 5))

    # pcolormesh needs edges; create them from midpoints on a symlog scale.
    # Use the actual numeric values on axes — not integer positions.
    d_edges = np.concatenate([[deltas[0] * 0.7],
                               (deltas[:-1] + deltas[1:]) / 2,
                               [deltas[-1] * 1.3]])
    # For lambda we mix 0 and log-spaced values; use ordinal x-ticks instead.
    x_pos = np.arange(len(lambdas))
    y_pos = np.arange(len(deltas))
    # pcolormesh with unit cells, manual tick labels
    X, Y = np.meshgrid(np.arange(len(lambdas) + 1) - 0.5,
                       np.arange(len(deltas) + 1) - 0.5)
    im = ax.pcolormesh(X, Y, pivot.values, cmap="viridis",
                       vmin=pivot.values.min(), vmax=pivot.values.max())
    plt.colorbar(im, ax=ax, label=metric.replace("_", " "))

    # Annotate cell values
    for yi, d in enumerate(deltas):
        for xi, l in enumerate(lambdas):
            val = pivot.loc[d, l]
            ax.text(xi, yi, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val < 0.7 else "black")

    # Best cell star
    best_idx = np.unravel_index(pivot.values.argmax(), pivot.values.shape)
    ax.plot(best_idx[1], best_idx[0], "*", color="white", markersize=18,
            label=f"Best: δ={deltas[best_idx[0]]:.2g}, λ={lambdas[best_idx[1]]:.1e}",
            zorder=5)

    lam_labels = [f"{v:.1e}" if v > 0 else "0" for v in lambdas]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lam_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{d:.2g}" for d in deltas])
    ax.set_xlabel("λ (communication penalty)")
    ax.set_ylabel("δ (quantisation bin width)")
    ax.legend(framealpha=0.85, fontsize=8, loc="upper left")
    ax.set_title(title or f"δ × λ heatmap — {metric}  (channel={channel}, z_dim={z_dim_fixed})")
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# APPENDIX C — Training dynamics at baseline-best config
# ---------------------------------------------------------------------------

def plot_appendix_training_dynamics(
    df: pd.DataFrame,
    channel: str = "sd",
    delta: float = 1.0,
    lambda_comms: float = 5e-4,
    z_dim: int = 2,
    x_col: str = "timestep",
    smooth: int = 10,
    title: str = "Training dynamics — baseline-best config",
    out_dir: str | Path | None = None,
    stem: str = "appC_training_dynamics",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Category: APPENDIX C — Training dynamics

    Hypothesis
    ----------
    Learning proceeds in two phases: (1) SR rises as the policy learns to
    navigate; (2) bits compress after SR plateaus as the speaker refines
    its code.  This two-phase dynamic confirms that task success is a
    prerequisite for communication efficiency — compression is a secondary
    effect of a well-trained policy.

    How to read
    -----------
    Top panel: success rate over training timesteps, mean ± std over seeds.
    Bottom panel: true_bits_per_msg (NOT surrogate) over training.  A two-
    phase pattern shows SR reaching plateau before bits begin to decrease.
    The red dashed line in the bits panel = H(G).  Grey dashed = float32
    reference (64 bits).

    Why included
    ------------
    Shows convergence stability of the baseline-best config.  The two-phase
    pattern supports the paper's claim that DDCL compression emerges
    naturally from the joint objective rather than requiring explicit
    supervision.
    """
    _require_mpl()
    _paper_style()

    sub = df.copy()
    filters = [("channel", channel), ("delta", delta),
               ("lambda_comms", lambda_comms), ("z_dim", z_dim)]
    for col, val in filters:
        if col in sub.columns:
            sub = sub[sub[col] == val]
    if sub.empty:
        raise ValueError(f"No training data for {filters}")

    bits_y = "true_bits_per_msg" if "true_bits_per_msg" in sub.columns else "bits_per_msg"
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for ax, ycol, ylabel, add_refs in [
        (axes[0], "success_rate", "Success rate", False),
        (axes[1], bits_y, "True bits / message", True),
    ]:
        if ycol not in sub.columns:
            continue
        piv = sub.pivot_table(index=x_col, columns="seed", values=ycol)
        if smooth > 1:
            piv = piv.rolling(smooth, min_periods=1).mean()
        xs = piv.index.values
        mean = piv.mean(axis=1).values
        std = piv.std(axis=1).fillna(0).values

        color = _CHANNEL_COLORS.get(channel, "#1f77b4")
        ax.plot(xs, mean, color=color,
                label=f"{_CHANNEL_LABELS.get(channel, channel)} (mean)")
        ax.fill_between(xs, mean - std, mean + std, alpha=0.18, color=color,
                        label="±1 std over seeds")

        if add_refs:
            ax.axhline(H_GOAL_BITS, color="red", linewidth=1.3, linestyle="--",
                       label=f"H(G) = {H_GOAL_BITS:.2f} bits")
            ax.axhline(_FLOAT32_BITS_PER_DIM * z_dim, color="grey",
                       linewidth=1.0, linestyle="-.",
                       label=f"Float32 = {_FLOAT32_BITS_PER_DIM * z_dim} bits")

        if not add_refs:
            ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Environment timesteps")
    axes[0].set_title(
        f"{title}\n(channel={channel}, δ={delta}, λ={lambda_comms:.1e}, z_dim={z_dim})"
    )
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


# ---------------------------------------------------------------------------
# APPENDIX D — Per-goal bit allocation
# ---------------------------------------------------------------------------

def plot_appendix_per_goal_allocation(
    df: pd.DataFrame,
    channels_to_show: Sequence[str] = ("sd", "additive_uniform"),
    delta: float = 1.0,
    lambda_comms: float = 5e-4,
    z_dim: int = 2,
    window: int = 20,
    n_goals: int = 6,
    title: str = "Per-goal bit allocation vs Shannon-optimal",
    out_dir: str | Path | None = None,
    stem: str = "appD_per_goal_allocation",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Category: APPENDIX D — Information allocation

    Hypothesis
    ----------
    The speaker learns to allocate bits approximately proportional to
    −log₂(p_i): rare goals (high −log₂(p_i)) receive more bits than
    frequent goals.  SD achieves tighter allocation to the Shannon-optimal
    pattern than additive-uniform, reflecting SD's superior code quality.

    How to read
    -----------
    Bars = mean surrogate bits allocated to each goal in the final
    training window, averaged over seeds.  Error bars = ±1 std.  Red
    dashed line = optimal −log₂(p_i) per goal (Shannon-optimal code
    length).  Red shading = combined uncertainty band: Jensen gap
    (systematic ~0.05–0.30 b upward bias from surrogate formula) plus
    RL sampling noise (∝ 1/√(p_i·N), larger for rare goals).

    ⚠ Unit note: bars are SURROGATE bits (log₂(|z|/δ+1)); the optimal
    line is TRUE bits (−log₂(p_i)).  The Jensen gap causes bars to sit
    systematically above the optimal line even for a perfect code — this
    is expected and does not indicate sub-optimality.  The uncertainty
    band accounts for this gap.

    Why included
    ------------
    Tests whether emergent communication is information-theoretically
    efficient.  An allocation matching the Zipf distribution would be
    near-optimal prefix-free coding without explicit supervision.
    `none` is excluded as its bits_goal_i = 0 (surrogate undefined).
    """
    _require_mpl()
    _paper_style()

    goal_cols = [f"bits_goal_{i}" for i in range(n_goals)]
    missing = [c for c in goal_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_goals)
    n_ch = len(channels_to_show)
    width = 0.7 / max(n_ch, 1)

    for ci, ch in enumerate(channels_to_show):
        sub = df.copy()
        for col, val in [("channel", ch), ("delta", delta),
                         ("lambda_comms", lambda_comms), ("z_dim", z_dim)]:
            if col in sub.columns:
                sub = sub[sub[col] == val]
        if sub.empty:
            continue

        def _tail(grp):
            return grp.sort_values("update").tail(window)[goal_cols].mean()

        grp_keys = [c for c in ("seed",) if c in sub.columns]
        if grp_keys:
            per_seed = sub.groupby(grp_keys).apply(_tail).reset_index()
        else:
            per_seed = _tail(sub).to_frame().T

        means = per_seed[goal_cols].mean().values
        stds = per_seed[goal_cols].std().fillna(0).values

        color = _CHANNEL_COLORS.get(ch, cm.tab10(ci / 10))
        label = _CHANNEL_LABELS.get(ch, ch)
        offset = x + ci * width
        ax.bar(offset, means, width=width, label=label, color=color,
               alpha=0.85, yerr=stds, capsize=3, error_kw={"elinewidth": 1.0})

    x_center = x + width * (n_ch - 1) / 2

    # Optimal allocation line
    opt = np.array(GOAL_OPTIMAL_BITS[:n_goals])
    ax.plot(x_center, opt, color="red", marker="x", markersize=8,
            linewidth=1.8, linestyle="--", label="Optimal −log₂(p_i)", zorder=5)

    # Uncertainty envelope (Jensen gap + RL sampling noise)
    probs = np.array(_GOAL_PROBS[:n_goals])
    total_updates = int(df["total_timesteps"].iloc[0] /
                        (df["n_envs"].iloc[0] * df["n_steps"].iloc[0])) \
        if {"total_timesteps", "n_envs", "n_steps"}.issubset(df.columns) \
        else 244
    sigma_rl = 0.5 / np.sqrt(probs * total_updates)
    sigma_total = np.sqrt(sigma_rl ** 2 + 0.20 ** 2)  # 0.20b = Jensen gap

    ax.fill_between(x_center, opt - sigma_total, opt + sigma_total,
                    color="red", alpha=0.10, zorder=4,
                    label="Uncertainty (Jensen gap + RL noise)")

    # Effective sample annotations
    for gi in range(n_goals):
        n_eff = probs[gi] * total_updates
        ax.annotate(f"n≈{n_eff:.0f}", xy=(x_center[gi], ax.get_ylim()[0] + 0.1),
                    ha="center", va="bottom", fontsize=7, color="dimgrey")

    ax.set_xticks(x_center)
    ax.set_xticklabels([f"Goal {i}\n(p={_GOAL_PROBS[i]:.3f})" for i in range(n_goals)],
                       fontsize=8)
    ax.set_ylabel("Mean surrogate bits allocated")
    ax.set_title(
        f"{title}\n(δ={delta}, λ={lambda_comms:.1e}, z_dim={z_dim})\n"
        "⚠ Bars = surrogate; optimal line = true bits; gap of ~0.05–0.30 b expected"
    )
    ax.legend(framealpha=0.85, fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# APPENDIX E — Information overhead above H(G)
# ---------------------------------------------------------------------------

def plot_appendix_overhead_above_hg(
    summary: pd.DataFrame,
    bits_col: str = "true_bits_per_msg",
    sr_col: str = "success_rate",
    z_dim_fixed: int = 2,
    title: str = "Bits overhead above Shannon limit H(G)",
    out_dir: str | Path | None = None,
    stem: str = "appE_overhead_above_hg",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Category: APPENDIX E — Efficiency quantification

    Hypothesis
    ----------
    DDCL channels use far fewer bits above H(G) than float32 passthrough.
    SD achieves the smallest overhead; NSD uses slightly more due to higher
    noise variance; additive-uniform is in between.

    How to read
    -----------
    Horizontal bar chart: each bar = (best-config true_bits − H(G)) for
    that channel at z_dim=2.  Lower is better.  Bars are sorted by
    overhead ascending (best channels leftmost).  Overhead = 0 would mean
    perfect Shannon-optimal coding.  Text annotations show the actual
    true_bits and success_rate.

    Why included
    ------------
    Provides the normalised efficiency number (e.g. "SD uses 2.9 bits
    above the Shannon limit") that the paper cites in the abstract and
    discussion.  Clarifies the gap between H(G), the best DDCL operating
    point, and float32.
    """
    _require_mpl()
    _paper_style()

    channels = ["sd", "nsd", "additive_uniform", "none"]
    sub = summary[summary["z_dim"] == z_dim_fixed].copy() \
        if "z_dim" in summary.columns else summary.copy()

    rows = []
    for ch in channels:
        ch_df = sub[sub["channel"] == ch] if "channel" in sub.columns else sub
        if ch_df.empty:
            continue
        grp_cols = [c for c in ("delta", "lambda_comms") if c in ch_df.columns]
        if grp_cols:
            per_cfg = ch_df.groupby(grp_cols).agg(
                sr_mean=(sr_col, "mean"), bits_mean=(bits_col, "mean"),
                bits_std=(bits_col, "std"),
                n=(sr_col, "count"),
            ).reset_index()
            best = per_cfg.sort_values(["sr_mean", "bits_mean"],
                                       ascending=[False, True]).iloc[0]
        else:
            best = pd.Series({
                "sr_mean": ch_df[sr_col].mean(),
                "bits_mean": ch_df[bits_col].mean(),
                "bits_std": ch_df[bits_col].std(),
                "n": len(ch_df),
            })
        rows.append({
            "channel": ch,
            "label": _CHANNEL_LABELS[ch],
            "overhead": float(best["bits_mean"]) - H_GOAL_BITS,
            "bits": float(best["bits_mean"]),
            "bits_std": float(best.get("bits_std", 0)),
            "n": int(best.get("n", 1)),
            "sr": float(best["sr_mean"]),
        })

    rows_df = pd.DataFrame(rows).sort_values("overhead")
    colors = [_CHANNEL_COLORS.get(r["channel"], "#888") for _, r in rows_df.iterrows()]

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(rows_df))
    cis = [_sem_ci(r["bits_std"], r["n"]) for _, r in rows_df.iterrows()]
    bars = ax.barh(y, rows_df["overhead"].values, xerr=cis,
                   color=colors, alpha=0.85, capsize=4)
    for bar, (_, row) in zip(bars, rows_df.iterrows()):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{row['bits']:.1f} bits  (SR={row['sr']:.2f})",
                va="center", fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(y)
    ax.set_yticklabels(rows_df["label"].values)
    ax.set_xlabel(f"true_bits_per_msg − H(G)  (H(G)={H_GOAL_BITS:.2f} bits)")
    ax.set_title(f"{title}  [z_dim={z_dim_fixed}]")
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# APPENDIX F — Surrogate vs true bits calibration
# ---------------------------------------------------------------------------

def plot_appendix_surrogate_calibration(
    summary: pd.DataFrame,
    surrogate_col: str = "bits_per_msg",
    true_col: str = "true_bits_per_msg",
    z_dim_fixed: int = 2,
    title: str = "Surrogate calibration: Jensen UB vs true transmission bits",
    out_dir: str | Path | None = None,
    stem: str = "appF_surrogate_calibration",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Category: APPENDIX F — Surrogate validity

    Hypothesis
    ----------
    The surrogate bits_per_msg = log₂(|z|/δ+1) is a monotone proxy for
    true_bits_per_msg but systematically overestimates it (Jensen gap).
    The gap is small (< 0.5 bits) in the operating regime |z|/δ ∈ [1, 5]
    and larger near z=0.  The `none` channel has surrogate=0 and
    true_bits=64 — a structural outlier that confirms the surrogate is
    only valid as a training proxy for quantised channels.

    How to read
    -----------
    Scatter of (surrogate, true_bits) per seed per channel.  Reference
    line y=x: perfect calibration.  Points above y=x: surrogate
    overestimates (expected for DDCL).  The `none` point at (0, 64) is
    annotated separately as a structural outlier.  The spread of points
    along the y=x direction indicates the range of operating regimes
    covered across (λ, δ) settings.

    Why included
    ------------
    Justifies use of `true_bits_per_msg` for all cross-channel comparisons.
    Confirms that the surrogate is a valid training objective (monotone,
    differentiable) while being quantitatively wrong for reporting purposes.
    """
    _require_mpl()
    _paper_style()

    if surrogate_col not in summary.columns or true_col not in summary.columns:
        raise ValueError(
            f"Need '{surrogate_col}' and '{true_col}' in summary. "
            f"Available: {list(summary.columns)}"
        )
    sub = summary[summary["z_dim"] == z_dim_fixed].copy() \
        if "z_dim" in summary.columns else summary.copy()

    fig, ax = plt.subplots(figsize=(6, 5))

    none_sub = sub[sub["channel"] == "none"] if "channel" in sub.columns else pd.DataFrame()
    ddcl_sub = sub[sub["channel"] != "none"] if "channel" in sub.columns else sub

    for ch in ["sd", "nsd", "additive_uniform"]:
        ch_df = ddcl_sub[ddcl_sub["channel"] == ch] if "channel" in ddcl_sub.columns else ddcl_sub
        if ch_df.empty:
            continue
        ax.scatter(ch_df[surrogate_col], ch_df[true_col],
                   color=_CHANNEL_COLORS.get(ch, "#333"),
                   marker=_CHANNEL_MARKERS.get(ch, "o"),
                   label=_CHANNEL_LABELS.get(ch, ch),
                   alpha=0.75, s=40, zorder=4)

    # y=x reference
    all_ddcl = ddcl_sub[[surrogate_col, true_col]].dropna()
    if not all_ddcl.empty:
        lo = all_ddcl.min().min()
        hi = all_ddcl.max().max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5,
                label="y = x  (perfect calibration)", zorder=5)

    # `none` as separate annotated marker
    if not none_sub.empty:
        ax.scatter(none_sub[surrogate_col].mean(), none_sub[true_col].mean(),
                   marker="D", s=120, color=_CHANNEL_COLORS["none"],
                   label=_CHANNEL_LABELS["none"], zorder=6,
                   facecolors="none", edgecolors=_CHANNEL_COLORS["none"], linewidths=2.0)
        ax.annotate(
            f"Float32\n(surr=0, true={none_sub[true_col].mean():.0f}b)",
            xy=(none_sub[surrogate_col].mean(), none_sub[true_col].mean()),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
        )

    ax.set_xlabel("Surrogate bits/msg   log₂(|z|/δ+1)   (Jensen UB, training proxy)")
    ax.set_ylabel("True transmission bits/msg")
    ax.set_title(f"{title}  [z_dim={z_dim_fixed}]")
    ax.legend(framealpha=0.85, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# APPENDIX G — z_dim scaling
# ---------------------------------------------------------------------------

def plot_appendix_zdim_scaling(
    summary: pd.DataFrame,
    channels: Sequence[str] = ("sd", "nsd", "additive_uniform"),
    sr_col: str = "success_rate",
    bits_col: str = "true_bits_per_msg",
    title: str = "Message dimension scaling",
    out_dir: str | Path | None = None,
    stem: str = "appG_zdim_scaling",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Category: APPENDIX G — Capacity analysis

    Hypothesis
    ----------
    z_dim=2 is sufficient for SD to reach SR=1.0 (H(G)=1.81 bits; 2
    dimensions give ample capacity).  NSD requires z_dim=3 due to its
    higher noise variance (δ²/4 vs δ²/12): the extra dimension compensates
    for noisier individual dimensions.  Larger z_dim increases total bits
    roughly linearly, so z_dim=2 is the Pareto-optimal choice for SD.

    How to read
    -----------
    Left panel: success rate vs z_dim.  Right panel: true bits vs z_dim.
    Each channel shown at its best (λ, δ) for each z_dim independently.
    If SR plateaus at z_dim=2 for SD but not NSD, this confirms the noise
    variance explanation.

    Why included
    ------------
    Explains why the baseline-best uses z_dim=2 and why NSD results should
    be compared at z_dim=3 rather than z_dim=2 for a fair head-to-head.
    """
    _require_mpl()
    _paper_style()

    if "z_dim" not in summary.columns:
        raise ValueError("'z_dim' column not in summary")

    zdims = sorted(summary["z_dim"].unique())
    grp_cols = [c for c in ("delta", "lambda_comms") if c in summary.columns]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ch in channels:
        ch_df = summary[summary["channel"] == ch] if "channel" in summary.columns \
            else summary
        sr_m, sr_s, b_m, b_s = [], [], [], []
        for z in zdims:
            z_df = ch_df[ch_df["z_dim"] == z]
            if z_df.empty:
                sr_m.append(np.nan); sr_s.append(0)
                b_m.append(np.nan);  b_s.append(0)
                continue
            # Best config at this z_dim
            if grp_cols:
                per_cfg = z_df.groupby(grp_cols).agg(
                    sr_=(sr_col, "mean"), bits_=(bits_col, "mean")
                ).reset_index()
                best = per_cfg.sort_values(["sr_", "bits_"],
                                           ascending=[False, True]).iloc[0]
                flt = z_df[grp_cols[0]] == best[grp_cols[0]]
                for g in grp_cols[1:]:
                    flt &= z_df[g] == best[g]
                z_df_best = z_df[flt]
            else:
                z_df_best = z_df
            sr_m.append(z_df_best[sr_col].mean())
            sr_s.append(z_df_best[sr_col].std())
            b_m.append(z_df_best[bits_col].mean())
            b_s.append(z_df_best[bits_col].std())

        color = _CHANNEL_COLORS.get(ch, None)
        label = _CHANNEL_LABELS.get(ch, ch)
        marker = _CHANNEL_MARKERS.get(ch, "o")
        axes[0].errorbar(zdims, sr_m, yerr=sr_s, color=color,
                         marker=marker, capsize=3, label=label)
        axes[1].errorbar(zdims, b_m, yerr=b_s, color=color,
                         marker=marker, capsize=3, label=label)

    axes[0].set_xlabel("z_dim"); axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].set_title("SR vs z_dim  (best λ, δ per z_dim)")
    axes[0].legend(framealpha=0.85, fontsize=8)
    axes[0].grid(True, alpha=0.2)

    axes[1].axhline(H_GOAL_BITS, color="red", linewidth=1.2, linestyle=":",
                    label=f"H(G)={H_GOAL_BITS:.2f} b")
    axes[1].set_xlabel("z_dim"); axes[1].set_ylabel("True bits / message")
    axes[1].set_title("True bits vs z_dim")
    axes[1].legend(framealpha=0.85, fontsize=8)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(title)
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


# ---------------------------------------------------------------------------
# APPENDIX H — SD vs NSD head-to-head
# ---------------------------------------------------------------------------

def plot_appendix_sd_nsd_comparison(
    summary: pd.DataFrame,
    delta_fixed: float = 1.0,
    z_dim_fixed: int = 2,
    sr_col: str = "success_rate",
    bits_col: str = "true_bits_per_msg",
    title: str = "SD vs NSD: synchrony-free dithering comparison",
    out_dir: str | Path | None = None,
    stem: str = "appH_sd_nsd_comparison",
    save_path: str | Path | None = None,
) -> tuple:
    """
    Category: APPENDIX H — DDCL variant comparison

    Hypothesis
    ----------
    NSD (no shared PRNG, TPDF dither, synchrony-free) performs comparably
    to SD at z_dim=3 but falls ~1% short at z_dim=2.  The performance
    gap is consistent with NSD's 3× higher reconstruction noise variance
    (δ²/4 vs δ²/12) requiring greater capacity.  The Pareto frontiers of
    both channels nearly coincide at matched z_dim.

    How to read
    -----------
    Left panel: scatter of (true_bits, SR) for SD (blue circles) and NSD
    (orange squares) across all λ at fixed δ=1.0, z_dim=2.  Both channels'
    Pareto frontiers are shown.  Right panel: paired SR scatter at matched
    (λ, δ, z_dim, seed) — points above diagonal = NSD better; below = SD
    better.  If NSD is systematically below diagonal, the noise variance
    penalty is real.

    Why included
    ------------
    NSD is the synchrony-free variant (no shared dither PRNG between
    speaker and listener) — a practically important property.  The trade-
    off between synchrony-free deployment and task performance needs to be
    quantified.  If the gap is small, NSD is a valid practical choice.
    ⚠ NSD variance derivation (δ²/4) should be verified against the TPDF
    specification in the paper before final publication.
    """
    _require_mpl()
    _paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sd = summary[(summary["channel"] == "sd") & (summary["z_dim"] == z_dim_fixed)
                 & (summary["delta"] == delta_fixed)].copy() \
        if all(c in summary.columns for c in ("channel", "z_dim", "delta")) \
        else pd.DataFrame()
    nsd = summary[(summary["channel"] == "nsd") & (summary["z_dim"] == z_dim_fixed)
                  & (summary["delta"] == delta_fixed)].copy() \
        if all(c in summary.columns for c in ("channel", "z_dim", "delta")) \
        else pd.DataFrame()

    # Left: rate-distortion scatter + per-channel Pareto
    ax = axes[0]
    for df_ch, label, color, marker in [
        (sd, "SD", _CHANNEL_COLORS["sd"], _CHANNEL_MARKERS["sd"]),
        (nsd, "NSD", _CHANNEL_COLORS["nsd"], _CHANNEL_MARKERS["nsd"]),
    ]:
        if df_ch.empty:
            continue
        ax.scatter(df_ch[bits_col], df_ch[sr_col], color=color,
                   marker=marker, alpha=0.7, s=50, label=label, zorder=3)
        if len(df_ch) >= 2:
            pf = pareto_frontier(df_ch, x_col=sr_col, y_col=bits_col,
                                 x_better="higher", y_better="lower")
            pf = pf.sort_values(bits_col)
            ax.step(pf[bits_col], pf[sr_col], where="post",
                    color=color, linewidth=2.0, linestyle="--", zorder=4)

    ax.axvline(H_GOAL_BITS, color="red", linewidth=1.2, linestyle=":",
               label=f"H(G)={H_GOAL_BITS:.2f} b")
    ax.set_xlabel("True bits / message")
    ax.set_ylabel("Success rate")
    ax.set_title(f"Rate–distortion: SD vs NSD  (δ={delta_fixed}, z_dim={z_dim_fixed})")
    ax.legend(framealpha=0.85, fontsize=8)
    ax.grid(True, alpha=0.2)

    # Right: paired scatter
    ax = axes[1]
    merge_cols = [c for c in ("lambda_comms", "seed")
                  if c in sd.columns and c in nsd.columns]
    if merge_cols and not sd.empty and not nsd.empty:
        paired = sd[merge_cols + [sr_col, bits_col]].merge(
            nsd[merge_cols + [sr_col, bits_col]],
            on=merge_cols, suffixes=("_sd", "_nsd"),
        )
        if not paired.empty:
            lo = min(paired[f"{sr_col}_sd"].min(), paired[f"{sr_col}_nsd"].min()) - 0.02
            hi = max(paired[f"{sr_col}_sd"].max(), paired[f"{sr_col}_nsd"].max()) + 0.02
            ax.scatter(paired[f"{sr_col}_sd"], paired[f"{sr_col}_nsd"],
                       alpha=0.65, s=35, color="mediumpurple", zorder=3)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2,
                    label="SD = NSD  (diagonal)")
            ax.set_xlabel("SD success rate")
            ax.set_ylabel("NSD success rate")
            ax.set_title("Paired SR comparison (same λ × seed)")
            ax.legend(framealpha=0.85, fontsize=8)
            ax.grid(True, alpha=0.2)
        else:
            ax.text(0.5, 0.5, "No matched pairs found",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Paired comparison requires\nlambda_comms and seed columns",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(title)
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_sweep_figures(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    agg: pd.DataFrame,
    out_dir: str | Path = "results/toyproblem/sweep_stage_a/figures",
) -> None:
    """Generate all main paper + appendix figures for the Stage A sweep.

    Output layout::

        <out_dir>/main/       — two paper-ready figures (fig1, fig2)
        <out_dir>/appendix/   — eight appendix figures (appA–appH)

    Each figure is saved as both .pdf (for the paper) and .png (preview).

    Parameters
    ----------
    df      : full tidy training DataFrame (all seeds, one row per update)
    summary : per-seed final-metrics DataFrame from load_runs.final_metrics
    agg     : per-config aggregated DataFrame from load_runs.seed_aggregate,
              grouped by ['channel', 'delta', 'lambda_comms', 'z_dim']
    out_dir : root figures directory
    """
    out = Path(out_dir)
    main_dir = out / "main"
    app_dir = out / "appendix"

    # ── MAIN PAPER ────────────────────────────────────────────────────────────
    print("Generating main paper figures …")
    try:
        plot_paper_rate_distortion(agg, out_dir=main_dir)
        print("  fig1_rate_distortion — OK")
    except Exception as e:
        print(f"  fig1 FAILED: {e}")

    try:
        plot_paper_channel_comparison(summary, out_dir=main_dir)
        print("  fig2_channel_comparison — OK")
    except Exception as e:
        print(f"  fig2 FAILED: {e}")

    # ── APPENDIX ──────────────────────────────────────────────────────────────
    print("Generating appendix figures …")

    # App A: λ sensitivity (sd and nsd, delta=1.0, z_dim=2)
    try:
        plot_appendix_lambda_sensitivity(
            summary, channels=["sd", "nsd"],
            delta_fixed=1.0, z_dim_fixed=2,
            out_dir=app_dir,
        )
        print("  appA_lambda_sensitivity — OK")
    except Exception as e:
        print(f"  appA FAILED: {e}")

    # App B: δ×λ heatmap (sd, z_dim=2)
    try:
        plot_appendix_delta_lambda_heatmap(
            summary, channel="sd", z_dim_fixed=2, out_dir=app_dir
        )
        print("  appB_delta_lambda_heatmap — OK")
    except Exception as e:
        print(f"  appB FAILED: {e}")

    # App C: training dynamics at baseline-best
    try:
        plot_appendix_training_dynamics(
            df, channel="sd", delta=1.0, lambda_comms=5e-4, z_dim=2,
            out_dir=app_dir,
        )
        print("  appC_training_dynamics — OK")
    except Exception as e:
        print(f"  appC FAILED: {e}")

    # App D: per-goal allocation (sd + additive_uniform)
    goal_cols = [c for c in df.columns if c.startswith("bits_goal_")]
    if goal_cols:
        try:
            plot_appendix_per_goal_allocation(
                df, channels_to_show=["sd", "additive_uniform"],
                delta=1.0, lambda_comms=5e-4, z_dim=2,
                out_dir=app_dir,
            )
            print("  appD_per_goal_allocation — OK")
        except Exception as e:
            print(f"  appD FAILED: {e}")
    else:
        print("  appD SKIPPED (no bits_goal_* columns)")

    # App E: overhead above H(G)
    try:
        plot_appendix_overhead_above_hg(summary, z_dim_fixed=2, out_dir=app_dir)
        print("  appE_overhead_above_hg — OK")
    except Exception as e:
        print(f"  appE FAILED: {e}")

    # App F: surrogate calibration
    if "bits_per_msg" in summary.columns and "true_bits_per_msg" in summary.columns:
        try:
            plot_appendix_surrogate_calibration(summary, z_dim_fixed=2, out_dir=app_dir)
            print("  appF_surrogate_calibration — OK")
        except Exception as e:
            print(f"  appF FAILED: {e}")
    else:
        print("  appF SKIPPED (missing bits columns)")

    # App G: z_dim scaling
    if "z_dim" in summary.columns:
        try:
            plot_appendix_zdim_scaling(summary, out_dir=app_dir)
            print("  appG_zdim_scaling — OK")
        except Exception as e:
            print(f"  appG FAILED: {e}")
    else:
        print("  appG SKIPPED (no z_dim column)")

    # App H: SD vs NSD
    if "channel" in summary.columns and "nsd" in summary["channel"].values:
        try:
            plot_appendix_sd_nsd_comparison(
                summary, delta_fixed=1.0, z_dim_fixed=2, out_dir=app_dir
            )
            print("  appH_sd_nsd_comparison — OK")
        except Exception as e:
            print(f"  appH FAILED: {e}")
    else:
        print("  appH SKIPPED (nsd not in data)")

    print(f"\nDone. Figures in:\n  main/     → {main_dir}\n  appendix/ → {app_dir}")
