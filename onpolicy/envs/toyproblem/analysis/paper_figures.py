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

P2 ANALYSIS (5 figures, require entropy_rate / H_dim_* columns):
  plot_p2_shannon_gap        – true_bits − H(G) over training
  plot_p2_qphi_gap           – q_φ model fit convergence vs DLM floor
  plot_p2_context_bounds     – Context A vs B vs H_m_empirical bounds
  plot_p2_per_dim_entropy    – Per-dimension H(m_k) bar chart at convergence
  plot_p2_gradient_balance   – Speaker grad norm + entropy loss magnitude

Batch entry point:
  generate_sweep_figures(df, summary, agg, out_dir)

Aesthetic conventions (applied consistently across all figures):
  - Float32 (none channel) is NEVER plotted on the data axes; it appears
    only as a text annotation.  This keeps axes focused on DDCL data.
  - H(G) = 1.81 bits appears as a vertical/horizontal dashed reference
    line on every bits axis.
  - Legends are always outside the data area (below or right).
  - Error bars show 95% CI (z * std / sqrt(n)) or ±1 std explicitly.
  - Channel colours and markers are consistent across all figures.
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
# Shared constants
# ---------------------------------------------------------------------------

_PAPER_RCPARAMS = {
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "lines.linewidth":   1.8,
    "figure.dpi":        200,
    "axes.spines.top":   False,
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


def _float32_annotation(ax, bits: float, sr: float) -> None:
    """Add Float32 as a text box at the top-right of the axes.

    Placed top-right (not bottom-right) so it never overlaps the legend,
    which is conventionally anchored to the lower-right for rate-distortion
    plots where the Pareto frontier rises to the upper-left.
    """
    ax.annotate(
        f"Float32 passthrough\n({bits:.0f} bits, SR={sr:.2f})",
        xy=(0.99, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        color=_CHANNEL_COLORS["none"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_CHANNEL_COLORS["none"],
                  alpha=0.85, lw=0.8),
    )


# ---------------------------------------------------------------------------
# MAIN PAPER — Figure 1: Rate-Distortion Frontier
# ---------------------------------------------------------------------------

def _monotone_frontier(
    ch_data: "pd.DataFrame",
    bits_col: str,
    sr_col: str,
) -> "pd.DataFrame":
    """Return the monotone non-dominated subset of one channel's configs.

    Sorts by bits ascending, then keeps only rows where SR is non-decreasing
    (i.e. each point must improve on everything to its left).  This produces
    a clean upward-staircase frontier even when mean-aggregated values are
    not perfectly Pareto-consistent due to seed noise.
    """
    if ch_data.empty:
        return ch_data
    s = ch_data.sort_values(bits_col).reset_index(drop=True)
    max_sr = -np.inf
    keep = []
    for i, row in s.iterrows():
        if row[sr_col] >= max_sr:
            max_sr = row[sr_col]
            keep.append(i)
    return s.loc[keep]


def _topk_per_delta(
    ch_data: "pd.DataFrame",
    bits_col: str,
    sr_col: str,
    delta_col: str,
    lambda_col: str,
    top_k: int,
) -> "pd.DataFrame":
    """Return the top-K Pareto-best rows per delta group for one channel.

    Within each (channel, δ) group the rows are ranked by Pareto dominance
    (high SR, low bits).  The K best are kept and sorted by bits so the
    returned subset can be drawn as a connected tradeoff line.
    """
    kept = []
    for d, grp in ch_data.groupby(delta_col):
        # Pareto rank: number of other points that dominate each point
        sr   = grp[sr_col].values
        bits = grp[bits_col].values
        # dominated[i] = True if some j has sr[j]>=sr[i] AND bits[j]<=bits[i]
        # with at least one strict — i.e., j strictly dominates i
        n = len(grp)
        rank = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                if sr[j] >= sr[i] and bits[j] <= bits[i] and (sr[j] > sr[i] or bits[j] < bits[i]):
                    rank[i] += 1
        # Keep top_k lowest-rank (best) rows; tie-break by SR desc
        order = np.lexsort((-sr, rank))          # primary: rank asc; secondary: SR desc
        sel_idx = order[:top_k]
        sel = grp.iloc[sel_idx].sort_values(bits_col)
        kept.append(sel)
    if not kept:
        return ch_data.iloc[0:0]
    return pd.concat(kept, ignore_index=True)


def plot_paper_rate_distortion(
    agg: "pd.DataFrame",
    sr_col: str = "success_rate_mean",
    bits_col: str = "true_bits_per_msg_mean",
    sr_std_col: str = "success_rate_std",
    bits_std_col: str = "true_bits_per_msg_std",
    n_col: str = "n_seeds",
    channel_col: str = "channel",
    lambda_col: str = "lambda_comms",
    delta_col: str = "delta",
    z_dim_fixed: int = 2,
    top_k: int = 4,
    title: str = "Rate–distortion frontier",
    save_path: "str | Path | None" = None,
    out_dir: "str | Path | None" = None,
    stem: str = "fig1_rate_distortion",
) -> tuple:
    """
    Category: MAIN PAPER — Figure 1 (Core result)

    Hypothesis
    ----------
    DDCL channels (SD, NSD) lie on the Pareto frontier of task success rate
    vs true transmission bits, achieving near-Shannon efficiency.  Additive-
    uniform achieves comparable SR at higher bit cost (lacks second-order
    Schuchman property).

    How to read
    -----------
    Upper-left is best (high SR, few bits).  Each faint line = one δ value
    for a channel, connecting the top-K Pareto-best configs (sorted by bits)
    to show the tradeoff shape without over-populating the plot.  Bold per-
    channel Pareto frontier (solid line + markers with 95% CI) shows the
    achievable envelope for each channel.  Inset zooms to SR > 0.85 / bits
    ≤ 5 where SD vs NSD competition is decided.  Vertical dotted red line =
    H(G) Shannon lower bound.  Float32 is NOT plotted on axes; see annotation.

    Why included
    ------------
    Core result figure.  The per-δ top-K tradeoff lines show that SD's
    dominance holds across the full sweep — not just the single best config
    — while keeping the plot uncluttered and readable.
    """
    _require_mpl()
    _paper_style()
    from matplotlib.lines import Line2D  # noqa: F401 — kept for proxy handles if needed

    fig, ax = plt.subplots(figsize=(7, 5))

    ddcl_channels = ["sd", "nsd", "additive_uniform"]

    # ── Filter to fixed z_dim ────────────────────────────────────────────────
    work = agg.copy()
    if "z_dim" in work.columns:
        work = work[work["z_dim"] == z_dim_fixed]

    ddcl    = work[work[channel_col].isin(ddcl_channels)].copy()
    none_df = work[work[channel_col] == "none"].copy()

    # ── Global Pareto frontier ───────────────────────────────────────────────
    if not ddcl.empty and bits_col in ddcl.columns and sr_col in ddcl.columns:
        pf = pareto_frontier(ddcl, x_col=sr_col, y_col=bits_col,
                             x_better="higher", y_better="lower")
    else:
        pf = pd.DataFrame()

    # ── Per-channel monotone frontier (achievable envelope) ─────────────────
    # One bold coloured line per channel.  Using _monotone_frontier (not raw
    # Pareto) so that mean-aggregated noise cannot produce V-shape artefacts.
    ch_pf_map: dict = {}
    for ch in ddcl_channels:
        ch_data = ddcl[ddcl[channel_col] == ch]
        if ch_data.empty:
            continue
        ch_pf_map[ch] = _monotone_frontier(ch_data, bits_col, sr_col)

    # ── Layer 1: top-K per-δ tradeoff lines (background, faint) ─────────────
    have_lambda = lambda_col in ddcl.columns
    have_delta  = delta_col  in ddcl.columns
    for ch in ddcl_channels:
        ch_data = ddcl[ddcl[channel_col] == ch]
        if ch_data.empty:
            continue
        color = _CHANNEL_COLORS[ch]
        marker = _CHANNEL_MARKERS[ch]

        if have_lambda and have_delta:
            subset = _topk_per_delta(ch_data, bits_col, sr_col,
                                     delta_col, lambda_col, top_k)
            # Draw one line per δ group, connecting the top-K points
            for _, grp in subset.groupby(delta_col):
                grp_sorted = grp.sort_values(bits_col)
                if len(grp_sorted) < 2:
                    ax.scatter(grp_sorted[bits_col], grp_sorted[sr_col],
                               s=14, color=color, marker=marker,
                               alpha=0.25, zorder=2)
                    continue
                ax.plot(grp_sorted[bits_col], grp_sorted[sr_col],
                        color=color, linewidth=0.9, alpha=0.25, zorder=2)
                ax.scatter(grp_sorted[bits_col], grp_sorted[sr_col],
                           s=14, color=color, marker=marker,
                           alpha=0.25, zorder=2)

    # ── Layer 2: per-channel frontier — bold line + CI band + sparse markers ──
    for ch in ddcl_channels:
        ch_pf = ch_pf_map.get(ch)
        if ch_pf is None or ch_pf.empty:
            continue
        color  = _CHANNEL_COLORS[ch]
        marker = _CHANNEL_MARKERS[ch]
        label  = _CHANNEL_LABELS[ch]
        ch_s   = ch_pf.sort_values(bits_col)

        has_n  = n_col in ch_s.columns
        has_ci = sr_std_col in ch_s.columns and bits_std_col in ch_s.columns
        if has_ci:
            n_vals = ch_s[n_col].values if has_n else np.full(len(ch_s), 5)
            yerr_arr = np.array([_sem_ci(s, n)
                                 for s, n in zip(ch_s[sr_std_col], n_vals)])
            xerr_arr = np.array([_sem_ci(s, n)
                                 for s, n in zip(ch_s[bits_std_col], n_vals)])
        else:
            yerr_arr = xerr_arr = None

        # Bold frontier line
        ax.plot(ch_s[bits_col], ch_s[sr_col],
                color=color, linewidth=2.0, alpha=0.9, zorder=4)

        # 95% CI shaded band on SR (visible at any density of frontier points)
        if has_ci and yerr_arr is not None:
            ax.fill_between(
                ch_s[bits_col],
                ch_s[sr_col] - yerr_arr,
                ch_s[sr_col] + yerr_arr,
                color=color, alpha=0.15, zorder=3,
            )

        # Sparse error-bar markers: first point, last point, and every 3rd in between
        idx_all = np.arange(len(ch_s))
        sparse  = sorted(set([0, len(ch_s) - 1] + list(idx_all[1:-1:3])))
        ch_sp   = ch_s.iloc[sparse]
        xe = xerr_arr[sparse] if xerr_arr is not None else None
        ye = yerr_arr[sparse] if yerr_arr is not None else None
        ax.errorbar(
            ch_sp[bits_col], ch_sp[sr_col],
            xerr=xe, yerr=ye,
            fmt=marker, color=color, markersize=7,
            capsize=3, capthick=1.1, elinewidth=0.9,
            label=label, zorder=5, alpha=0.95,
        )

    # ── H(G) Shannon lower bound ─────────────────────────────────────────────
    ax.axvline(H_GOAL_BITS, color="red", linewidth=1.5, linestyle=":",
               label=f"H(G) = {H_GOAL_BITS:.2f} bits", zorder=6)

    # ── Float32 annotation ───────────────────────────────────────────────────
    if not none_df.empty and bits_col in none_df.columns and sr_col in none_df.columns:
        _float32_annotation(ax, float(none_df[bits_col].mean()),
                            float(none_df[sr_col].mean()))

    # ── Axis limits, labels, legend ──────────────────────────────────────────
    if not ddcl.empty:
        ax.set_xlim(left=0, right=ddcl[bits_col].max() * 1.12)
    else:
        ax.set_xlim(left=0)
    if not ddcl.empty:
        x_max = ddcl[bits_col].max() * 1.12
    else:
        x_max = 10.0
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("True transmission bits / message")
    ax.set_ylabel("Task success rate")
    ax.set_title(title)
    # Legend in the lower-left void (bits < H(G), SR < 0.5 is always empty).
    ax.legend(framealpha=0.90, fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.2, axis="both")

    # ── Inset: zoom to high-SR competition zone (SR > 0.85, bits ≤ 5) ───────
    # Placed in the lower-right of the main axes where data is sparse.
    # Uses ax.inset_axes (axes-fraction coords) to stay inside the figure.
    zoom_sr_min, zoom_bits_max = 0.85, 5.0
    in_zone = ddcl[(ddcl[sr_col] >= zoom_sr_min) & (ddcl[bits_col] <= zoom_bits_max)]
    if in_zone[channel_col].nunique() >= 2:
        # [left, bottom, width, height] in axes-fraction coordinates
        axins = ax.inset_axes([0.50, 0.03, 0.48, 0.44])
        axins.set_facecolor("#f8f8f8")

        # Background: top-K per-δ tradeoff lines (same alpha as main)
        for ch in ddcl_channels:
            ch_data = ddcl[ddcl[channel_col] == ch]
            if ch_data.empty:
                continue
            color  = _CHANNEL_COLORS[ch]
            marker = _CHANNEL_MARKERS[ch]
            if have_lambda and have_delta:
                subset = _topk_per_delta(ch_data, bits_col, sr_col,
                                         delta_col, lambda_col, top_k)
                for _, grp in subset.groupby(delta_col):
                    grp_s = grp.sort_values(bits_col)
                    axins.plot(grp_s[bits_col], grp_s[sr_col],
                               color=color, linewidth=0.8, alpha=0.25)
                    axins.scatter(grp_s[bits_col], grp_s[sr_col],
                                  s=10, color=color, marker=marker, alpha=0.25)

        # Foreground: per-channel frontier lines + CI band + sparse markers
        for ch in ddcl_channels:
            ch_pf = ch_pf_map.get(ch)
            if ch_pf is None or ch_pf.empty:
                continue
            ch_s   = ch_pf.sort_values(bits_col)
            color  = _CHANNEL_COLORS[ch]
            has_n  = n_col in ch_s.columns
            if sr_std_col in ch_s.columns and bits_std_col in ch_s.columns:
                n_v   = ch_s[n_col].values if has_n else np.full(len(ch_s), 5)
                ye_i  = np.array([_sem_ci(s, n)
                                  for s, n in zip(ch_s[sr_std_col], n_v)])
                xe_i  = np.array([_sem_ci(s, n)
                                  for s, n in zip(ch_s[bits_std_col], n_v)])
            else:
                ye_i = xe_i = None

            axins.plot(ch_s[bits_col], ch_s[sr_col],
                       color=color, linewidth=1.6, alpha=0.9)
            if ye_i is not None:
                axins.fill_between(ch_s[bits_col],
                                   ch_s[sr_col] - ye_i,
                                   ch_s[sr_col] + ye_i,
                                   color=color, alpha=0.15)
            # Sparse markers in inset: first + last only (to avoid clutter)
            sp_idx = sorted({0, len(ch_s) - 1})
            ch_sp  = ch_s.iloc[sp_idx]
            axins.errorbar(
                ch_sp[bits_col], ch_sp[sr_col],
                xerr=xe_i[sp_idx] if xe_i is not None else None,
                yerr=ye_i[sp_idx] if ye_i is not None else None,
                fmt=_CHANNEL_MARKERS[ch], color=color,
                markersize=5, capsize=2, capthick=0.8, elinewidth=0.7,
                alpha=0.95, zorder=5,
            )

        axins.axvline(H_GOAL_BITS, color="red", linewidth=1.0,
                      linestyle=":", alpha=0.8)
        # Tight x-range: from just below the lowest bits in the zone
        zone_bits_min = in_zone[bits_col].min()
        axins.set_xlim(max(H_GOAL_BITS - 0.1, zone_bits_min - 0.15), zoom_bits_max)
        axins.set_ylim(zoom_sr_min - 0.01, 1.025)
        axins.tick_params(labelsize=6)
        axins.grid(True, alpha=0.15)
        axins.set_title("SR > 0.85 (zoom)", fontsize=7, pad=2)

        # Subtle zoom indicator: dashed box only, no diagonal connecting lines
        ax.indicate_inset_zoom(axins, edgecolor="grey", alpha=0.35,
                               linewidth=0.7)

    fig.tight_layout()

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
    SD achieves the highest SR at the lowest bit cost.  Float32 is excluded
    from the bars (64 bits distorts the scale) and shown as a text note.

    How to read
    -----------
    Left panel: SR per channel with 95% CI.  Right panel: true bits/msg
    with H(G) reference.  Each channel shown at its best config.

    Why included
    ------------
    Fair fixed-capacity head-to-head comparison.
    """
    _require_mpl()
    _paper_style()

    ddcl_channels = ["sd", "nsd", "additive_uniform"]
    labels = [_CHANNEL_LABELS[c] for c in ddcl_channels]
    colors = [_CHANNEL_COLORS[c] for c in ddcl_channels]

    sub = summary[summary["z_dim"] == z_dim_fixed].copy() \
        if "z_dim" in summary.columns else summary.copy()

    # For `none`, get SR and bits for the annotation
    none_df = sub[sub["channel"] == "none"] if "channel" in sub.columns else pd.DataFrame()
    none_sr = float(none_df[sr_col].mean()) if not none_df.empty else float("nan")
    none_bits = float(none_df[bits_col].mean()) if not none_df.empty else float("nan")

    best_rows: dict[str, pd.DataFrame] = {}
    for ch in ddcl_channels:
        ch_df = sub[sub["channel"] == ch] if "channel" in sub.columns else sub
        if ch_df.empty:
            continue
        grp_cols = [c for c in ("delta", "lambda_comms") if c in ch_df.columns]
        if grp_cols:
            per_config = ch_df.groupby(grp_cols).agg(
                sr_mean=(sr_col, "mean"),
                bits_mean=(bits_col, "mean"),
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

    sr_means, sr_cis, bits_means, bits_cis = [], [], [], []
    annotations: list[str] = []

    for ch in ddcl_channels:
        df_ch = best_rows.get(ch, pd.DataFrame())
        if df_ch.empty:
            sr_means.append(np.nan); sr_cis.append(0)
            bits_means.append(np.nan); bits_cis.append(0)
            annotations.append("")
            continue
        sr_vals = df_ch[sr_col].values
        bits_vals = df_ch[bits_col].values
        n = len(sr_vals)
        sr_means.append(float(np.mean(sr_vals)))
        sr_cis.append(_sem_ci(float(np.std(sr_vals)), n))
        bits_means.append(float(np.mean(bits_vals)))
        bits_cis.append(_sem_ci(float(np.std(bits_vals)), n))
        if "lambda_comms" in df_ch.columns and "delta" in df_ch.columns:
            lam = df_ch["lambda_comms"].iloc[0]
            dlt = df_ch["delta"].iloc[0]
            annotations.append(f"δ={dlt:.1g}, λ={lam:.1g}")
        else:
            annotations.append(f"n={n}")

    x = np.arange(len(ddcl_channels))
    width = 0.55
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, means, cis, ylabel, add_hg in [
        (axes[0], sr_means, sr_cis, "Task success rate", False),
        (axes[1], bits_means, bits_cis, "True bits / message", True),
    ]:
        bars = ax.bar(
            x, means, yerr=cis, color=colors,
            capsize=5, width=width, error_kw={"elinewidth": 1.4},
            alpha=0.85,
        )
        # Config labels above each bar
        for bar, ann, ci in zip(bars, annotations, cis):
            h = (bar.get_height() or 0) + (ci or 0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] > 0 else 1),
                ann, ha="center", va="bottom", fontsize=7, color="dimgrey",
            )
        if add_hg:
            ax.axhline(
                H_GOAL_BITS, color="red", linewidth=1.5, linestyle="--",
                label=f"H(G) = {H_GOAL_BITS:.2f} bits",
            )
            # Float32 annotation (text, not a bar)
            if not math.isnan(none_bits):
                ax.annotate(
                    f"Float32: {none_bits:.0f} bits",
                    xy=(1.0, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    color=_CHANNEL_COLORS["none"],
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=_CHANNEL_COLORS["none"], alpha=0.85, lw=0.8),
                )
            ax.legend(fontsize=8, loc="upper right")
        else:
            if not math.isnan(none_sr):
                ax.axhline(none_sr, color=_CHANNEL_COLORS["none"], linewidth=1.2,
                           linestyle=":", alpha=0.7, label="Float32 SR")
                ax.legend(fontsize=8, loc="lower right")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
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
    operating point.

    How to read
    -----------
    Two stacked panels, shared x-axis (λ on log scale; λ=0 is leftmost
    tick, log-spaced values follow).  Top = SR vs λ.  Bottom = true bits
    vs λ.  Shaded bands = ±1 std over seeds.  Dotted vertical lines = λ*
    (largest λ with SR ≥ 0.95).  Bottom panel has H(G) as red dashed
    reference.  Legend is outside below the plot.

    Why included
    ------------
    Justifies the λ=5×10⁻⁴ choice for the baseline-best config.
    """
    _require_mpl()
    _paper_style()

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_sr, ax_bits = axes

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

        # Use x-indices so λ=0 and log-spaced values plot at equal spacing
        x = np.arange(len(lam_arr))

        ax_sr.plot(x, sr_m, color=color, marker=_CHANNEL_MARKERS.get(ch, "o"),
                   markersize=6, label=label)
        ax_sr.fill_between(x, sr_m - sr_s, sr_m + sr_s, alpha=0.15, color=color)

        ax_bits.plot(x, b_m, color=color, marker=_CHANNEL_MARKERS.get(ch, "o"),
                     markersize=6, label=label)
        ax_bits.fill_between(x, b_m - b_s, b_m + b_s, alpha=0.12, color=color)

        # λ* — largest λ with SR ≥ 0.95
        knee_mask = sr_m >= 0.95
        if knee_mask.any():
            knee_idx = int(np.where(knee_mask)[0][-1])
            knee_lam = lam_arr[knee_idx]
            ax_sr.axvline(knee_idx, color=color, linewidth=1.0, linestyle=":",
                          alpha=0.7, label=f"λ*={knee_lam:.1e}")
            ax_bits.axvline(knee_idx, color=color, linewidth=1.0, linestyle=":",
                            alpha=0.7)

    if not has_data:
        ax_sr.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax_sr.transAxes)
        return fig, axes

    # H(G) reference on bits panel — label clarifies it is minimum bits for SR=1
    ax_bits.axhline(H_GOAL_BITS, color="red", linewidth=1.3, linestyle="--",
                    label=f"H(G) = {H_GOAL_BITS:.2f} bits  (min. bits for SR=1)")

    # X-axis: ordinal ticks with actual λ values as labels
    all_lambdas: list[float] = []
    for ch in channels:
        sub = summary.copy()
        for col, val in [("channel", ch), ("delta", delta_fixed), ("z_dim", z_dim_fixed)]:
            if col in sub.columns:
                sub = sub[sub[col] == val]
        if "lambda_comms" in sub.columns:
            all_lambdas.extend(sub["lambda_comms"].unique().tolist())
    all_lambdas = sorted(set(all_lambdas))
    tick_labels = [f"{v:.0e}" if v > 0 else "0" for v in all_lambdas]
    ax_bits.set_xticks(np.arange(len(all_lambdas)))
    ax_bits.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax_bits.set_xlabel(f"λ (communication penalty)  [δ={delta_fixed}, z_dim={z_dim_fixed}]")

    ax_sr.set_ylabel("Success rate")
    ax_sr.set_ylim(-0.05, 1.08)
    ax_sr.grid(True, alpha=0.2)

    ax_bits.set_ylabel("True bits / message")
    ax_bits.grid(True, alpha=0.2)

    # Legends below figure
    handles_sr, lbls_sr = ax_sr.get_legend_handles_labels()
    handles_bits, lbls_bits = ax_bits.get_legend_handles_labels()
    # Deduplicate
    seen: set[str] = set()
    all_h, all_l = [], []
    for h, l in zip(handles_sr + handles_bits, lbls_sr + lbls_bits):
        if l not in seen:
            all_h.append(h); all_l.append(l); seen.add(l)
    fig.legend(all_h, all_l, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=8, framealpha=0.85)

    axes[0].set_title(title)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


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
    (δ*, λ*) lies strictly inside the swept grid.

    How to read
    -----------
    Each cell = mean success_rate over seeds.  Yellow = high; purple = low.
    The best cell is highlighted with a bold white border.  Cell values
    are annotated in each cell.  λ axis uses ordinal spacing (labels show
    actual values including λ=0).  The legend is below the plot.

    Why included
    ------------
    Validates the Phase 2 convergence gate criterion 1 (interior optimum).
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

    fig, ax = plt.subplots(figsize=(10, 5))

    x_pos = np.arange(len(lambdas))
    y_pos = np.arange(len(deltas))
    X, Y = np.meshgrid(np.arange(len(lambdas) + 1) - 0.5,
                       np.arange(len(deltas) + 1) - 0.5)
    im = ax.pcolormesh(X, Y, pivot.values, cmap="viridis",
                       vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, label=metric.replace("_", " "))
    cbar.ax.tick_params(labelsize=8)

    # Cell value annotations — white text on dark cells, black on light
    for yi, d in enumerate(deltas):
        for xi, l in enumerate(lambdas):
            val = pivot.loc[d, l]
            text_color = "white" if val < 0.6 else "black"
            ax.text(xi, yi, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    # Best cell: bold border instead of a star marker
    best_idx = np.unravel_index(pivot.values.argmax(), pivot.values.shape)
    by, bx = best_idx
    rect = plt.Rectangle(
        (bx - 0.5, by - 0.5), 1.0, 1.0,
        fill=False, edgecolor="white", linewidth=2.5, zorder=6,
    )
    ax.add_patch(rect)
    # Small label outside the cell
    ax.text(bx, by + 0.45, "best", ha="center", va="bottom",
            fontsize=7, color="white", fontweight="bold")

    lam_labels = [f"{v:.1e}" if v > 0 else "0" for v in lambdas]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lam_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{d:.2g}" for d in deltas])
    ax.set_xlabel("λ (communication penalty)")
    ax.set_ylabel("δ (quantisation bin width)")
    ax.set_title(title or f"δ × λ heatmap — {metric}  (channel={channel}, z_dim={z_dim_fixed})")

    plt.tight_layout()
    # Reserve enough bottom space for rotated tick labels + annotation text
    fig.subplots_adjust(bottom=0.26)
    # Config of best cell shown in the bottom margin, safely below tick labels
    fig.text(0.5, 0.06,
             f"Best: δ={deltas[by]:.2g}, λ={lambdas[bx]:.1e}, "
             f"{metric}={pivot.values[by, bx]:.3f}",
             ha="center", fontsize=9, color="dimgrey")

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
    channels: Sequence[str] = ("sd", "nsd", "additive_uniform", "none"),
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
    SR rises first; bits compress after SR plateaus (two-phase dynamic).

    How to read
    -----------
    Top panel: SR over timesteps.  Bottom panel: true bits/msg (Float32
    excluded from bits axis; H(G) shown as red dashed).  All available
    channels shown.  Shaded bands = ±1 std over seeds.  The `none` channel
    shows only SR (bits undefined / 64).

    Why included
    ------------
    Shows convergence stability and the two-phase SR→bits pattern.
    """
    _require_mpl()
    _paper_style()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_sr, ax_bits = axes

    bits_y = "true_bits_per_msg" if "true_bits_per_msg" in df.columns else "bits_per_msg"

    any_bits_data = False
    for ch in channels:
        sub = df.copy()
        filters = [("channel", ch), ("delta", delta),
                   ("lambda_comms", lambda_comms), ("z_dim", z_dim)]
        # For `none`, only filter by channel and z_dim
        if ch == "none":
            filters = [("channel", ch), ("z_dim", z_dim)]
        for col, val in filters:
            if col in sub.columns:
                sub = sub[sub[col] == val]
        if sub.empty or x_col not in sub.columns:
            continue

        color = _CHANNEL_COLORS.get(ch, "#333333")
        label = _CHANNEL_LABELS.get(ch, ch)

        # SR panel — all channels
        if "success_rate" in sub.columns and "seed" in sub.columns:
            piv = sub.pivot_table(index=x_col, columns="seed", values="success_rate")
            if smooth > 1:
                piv = piv.rolling(smooth, min_periods=1).mean()
            xs = piv.index.values
            mean_sr = piv.mean(axis=1).values
            std_sr = piv.std(axis=1).fillna(0).values
            ax_sr.plot(xs, mean_sr, color=color, label=label, linewidth=1.8)
            ax_sr.fill_between(xs, mean_sr - std_sr, mean_sr + std_sr,
                               alpha=0.15, color=color)

        # Bits panel — skip `none` (64 bits off-scale)
        if ch != "none" and bits_y in sub.columns and "seed" in sub.columns:
            piv_b = sub.pivot_table(index=x_col, columns="seed", values=bits_y)
            if smooth > 1:
                piv_b = piv_b.rolling(smooth, min_periods=1).mean()
            xs_b = piv_b.index.values
            mean_b = piv_b.mean(axis=1).values
            std_b = piv_b.std(axis=1).fillna(0).values
            ax_bits.plot(xs_b, mean_b, color=color, label=label, linewidth=1.8)
            ax_bits.fill_between(xs_b, mean_b - std_b, mean_b + std_b,
                                 alpha=0.15, color=color)
            any_bits_data = True

    # H(G) reference on bits panel
    ax_bits.axhline(H_GOAL_BITS, color="red", linewidth=1.3, linestyle="--",
                    label=f"H(G) = {H_GOAL_BITS:.2f} bits")

    ax_sr.set_ylabel("Success rate")
    ax_sr.set_ylim(-0.05, 1.08)
    ax_sr.grid(True, alpha=0.2)
    ax_sr.legend(fontsize=8, framealpha=0.85, loc="lower right")

    ax_bits.set_ylabel("True bits / message")
    ax_bits.grid(True, alpha=0.2)
    if any_bits_data:
        ax_bits.legend(fontsize=8, framealpha=0.85, loc="upper right")

    axes[-1].set_xlabel("Environment timesteps")
    axes[0].set_title(
        f"{title}\n(δ={delta}, λ={lambda_comms:.1e}, z_dim={z_dim})"
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
    −log₂(p_i): rare goals receive more bits than frequent goals.  The
    gap above the Shannon-optimal line is evidence that the flat λ penalty
    does not achieve variable-rate coding — motivation for P2 (entropy model).

    How to read
    -----------
    Grouped bars (one group per goal, one bar per channel): actual mean
    bits in the final training window.  Red crosses = Shannon-optimal
    −log₂(p_i).  Goals are sorted by probability (most frequent left).
    Error bars = ±1 std over seeds.  The gap between bars and red crosses
    motivates P2 extensions.

    Why included
    ------------
    Tests whether emergent communication is information-theoretically
    efficient.  The residual gap motivates the P2 entropy model pillar.
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

        color = _CHANNEL_COLORS.get(ch, "#333333")
        label = _CHANNEL_LABELS.get(ch, ch)
        offset = x + ci * width - width * (n_ch - 1) / 2
        ax.bar(offset, means, width=width, label=label, color=color,
               alpha=0.85, yerr=stds, capsize=3, error_kw={"elinewidth": 1.0})

    # Shannon-optimal allocation as red cross markers
    opt = np.array(GOAL_OPTIMAL_BITS[:n_goals])
    ax.plot(x, opt, color="red", marker="x", markersize=10,
            linewidth=1.5, linestyle="--", label="Optimal −log₂(p_i)", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Goal {i}\np={_GOAL_PROBS[i]:.3f}" for i in range(n_goals)],
        fontsize=8,
    )
    ax.set_ylabel("Mean bits allocated per message")
    ax.set_title(f"{title}\n(δ={delta}, λ={lambda_comms:.1e}, z_dim={z_dim})")
    ax.legend(framealpha=0.85, fontsize=8, loc="upper right")
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

    How to read
    -----------
    Horizontal bar chart: bar = (best-config true_bits − H(G)) for each
    DDCL channel at z_dim=2.  Float32 is shown as a text annotation (its
    overhead of ~62 bits dwarfs the DDCL range and distorts the scale).
    Value labels are placed to the right of each bar.  x=0 = Shannon limit.

    Why included
    ------------
    Provides the normalised efficiency number (e.g. "SD uses 2.9 bits
    above H(G)") that the paper cites.
    """
    _require_mpl()
    _paper_style()

    ddcl_channels = ["sd", "nsd", "additive_uniform"]
    sub = summary[summary["z_dim"] == z_dim_fixed].copy() \
        if "z_dim" in summary.columns else summary.copy()

    # Float32 info for annotation
    none_df = sub[sub["channel"] == "none"] if "channel" in sub.columns else pd.DataFrame()
    none_bits = float(none_df[bits_col].mean()) if not none_df.empty else float("nan")
    none_sr = float(none_df[sr_col].mean()) if not none_df.empty else float("nan")

    rows = []
    for ch in ddcl_channels:
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

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(rows_df))
    cis = [_sem_ci(r["bits_std"], r["n"]) for _, r in rows_df.iterrows()]
    ax.barh(y, rows_df["overhead"].values, xerr=cis,
            color=colors, alpha=0.85, capsize=4)

    # Value labels to the right of bars (not on top, avoiding overlap)
    x_max = rows_df["overhead"].max() + max(cis, default=0)
    for i, (_, row) in enumerate(rows_df.iterrows()):
        ax.text(
            x_max * 1.02, i,
            f"{row['bits']:.2f} bits  (SR={row['sr']:.3f})",
            va="center", fontsize=8, color="dimgrey",
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(right=x_max * 1.45)
    ax.set_yticks(y)
    ax.set_yticklabels(rows_df["label"].values)
    ax.set_xlabel(f"true_bits/msg − H(G)    [H(G) = {H_GOAL_BITS:.2f} bits]")
    ax.set_title(f"{title}  [z_dim={z_dim_fixed}]")
    ax.grid(True, alpha=0.2, axis="x")

    # Float32 annotation
    if not math.isnan(none_bits):
        ax.annotate(
            f"Float32: {none_bits - H_GOAL_BITS:.1f} bits overhead\n"
            f"({none_bits:.0f} bits, SR={none_sr:.2f})",
            xy=(1.0, 0.02), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8,
            color=_CHANNEL_COLORS["none"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=_CHANNEL_COLORS["none"], alpha=0.85, lw=0.8),
        )

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
    The surrogate (log₂(|z|/δ+1)) is a monotone proxy for true bits but
    systematically overestimates (Jensen gap).  The gap is small in the
    operating regime |z|/δ ∈ [1, 5].

    How to read
    -----------
    Scatter of (surrogate, true) per seed per DDCL channel.  Float32
    (surrogate=0, true=64) is annotated separately but NOT plotted on axes
    (it compresses the DDCL region to the bottom 12%).  y=x = perfect
    calibration; DDCL points should lie above y=x (Jensen overestimate).

    Why included
    ------------
    Justifies using true_bits_per_msg for cross-channel comparisons and
    confirms the surrogate is a valid monotone training objective.
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

    none_sub = sub[sub["channel"] == "none"] if "channel" in sub.columns else pd.DataFrame()
    ddcl_sub = sub[sub["channel"] != "none"] if "channel" in sub.columns else sub

    fig, ax = plt.subplots(figsize=(6, 5))

    for ch in ["sd", "nsd", "additive_uniform"]:
        ch_df = ddcl_sub[ddcl_sub["channel"] == ch] if "channel" in ddcl_sub.columns \
            else ddcl_sub
        if ch_df.empty:
            continue
        ax.scatter(ch_df[surrogate_col], ch_df[true_col],
                   color=_CHANNEL_COLORS.get(ch, "#333"),
                   marker=_CHANNEL_MARKERS.get(ch, "o"),
                   label=_CHANNEL_LABELS.get(ch, ch),
                   alpha=0.75, s=40, zorder=4)

    # y=x reference — fitted to DDCL data range only
    all_ddcl = ddcl_sub[[surrogate_col, true_col]].dropna()
    if not all_ddcl.empty:
        lo = all_ddcl.min().min()
        hi = all_ddcl.max().max()
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", linewidth=1.5, label="y = x  (perfect calibration)", zorder=5)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)

    # Float32 as text annotation (not a data point)
    if not none_sub.empty:
        f32_surr = float(none_sub[surrogate_col].mean())
        f32_true = float(none_sub[true_col].mean())
        ax.annotate(
            f"Float32 passthrough\n(surr={f32_surr:.1f}, true={f32_true:.0f} bits)\n"
            f"excluded from axes",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8,
            color=_CHANNEL_COLORS["none"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=_CHANNEL_COLORS["none"], alpha=0.85, lw=0.8),
        )

    ax.set_xlabel("Surrogate bits/msg   log₂(|z|/δ+1)   (Jensen UB, training proxy)")
    ax.set_ylabel("True transmission bits/msg")
    ax.set_title(f"{title}  [z_dim={z_dim_fixed}]")
    ax.legend(framealpha=0.85, ncol=1, fontsize=8, loc="upper left")
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
    z_dim=2 is sufficient for SD (SR=1.0); NSD needs z_dim=3 due to higher
    noise variance (δ²/4 vs δ²/12).

    How to read
    -----------
    Left: SR vs z_dim (each channel at its best λ, δ per z_dim).
    Right: true bits vs z_dim.  Error bars = ±1 std.  H(G) on right panel.

    Why included
    ------------
    Explains why baseline uses z_dim=2 and when NSD reaches parity with SD.
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
                    label=f"H(G) = {H_GOAL_BITS:.2f} bits")
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
    NSD (synchrony-free, TPDF dither) performs comparably to SD at z_dim=3
    but ~1% short at z_dim=2.  The gap is consistent with NSD's 3× higher
    reconstruction noise variance (δ²/4 vs δ²/12).

    How to read
    -----------
    Left panel: paired dot plot — for each λ config, a dot for SD and one
    for NSD at the same λ, connected by a grey line.  Lines sloping up =
    NSD better; down = SD better.  Right panel: per-config Δ(SR) vs
    Δ(bits) scatter (SD − NSD), one point per (λ, z_dim) config.  Points
    in upper-left = SD better on both metrics.  Grey reference lines at
    Δ=0 show the no-difference baseline.

    Why included
    ------------
    Quantifies the trade-off between synchrony-free deployment and task
    performance.
    """
    _require_mpl()
    _paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Filter by delta and z_dim
    def _filter(ch):
        if not all(c in summary.columns for c in ("channel", "delta")):
            return pd.DataFrame()
        mask = (summary["channel"] == ch) & (summary["delta"] == delta_fixed)
        if "z_dim" in summary.columns:
            mask &= summary["z_dim"] == z_dim_fixed
        return summary[mask].copy()

    sd_all = _filter("sd")
    nsd_all = _filter("nsd")

    # ── Left panel: paired dot plot ───────────────────────────────────────────
    ax = axes[0]

    # Aggregate per-config mean across seeds for cleaner pairing
    grp_cols = [c for c in ("lambda_comms", "z_dim") if c in summary.columns]
    if grp_cols and not sd_all.empty and not nsd_all.empty:
        sd_agg = sd_all.groupby(grp_cols)[[sr_col, bits_col]].mean().reset_index()
        nsd_agg = nsd_all.groupby(grp_cols)[[sr_col, bits_col]].mean().reset_index()
        paired = sd_agg.merge(nsd_agg, on=grp_cols, suffixes=("_sd", "_nsd"))

        if not paired.empty:
            # Sort configs by SD SR for readable ordering
            paired = paired.sort_values(f"{sr_col}_sd")
            y_pos = np.arange(len(paired))

            # Connecting lines
            for i, (_, row) in enumerate(paired.iterrows()):
                ax.plot(
                    [row[f"{sr_col}_sd"], row[f"{sr_col}_nsd"]],
                    [i, i],
                    color="lightgrey", linewidth=1.0, zorder=1,
                )

            ax.scatter(paired[f"{sr_col}_sd"], y_pos,
                       color=_CHANNEL_COLORS["sd"], marker=_CHANNEL_MARKERS["sd"],
                       s=50, label="SD", zorder=3, alpha=0.85)
            ax.scatter(paired[f"{sr_col}_nsd"], y_pos,
                       color=_CHANNEL_COLORS["nsd"], marker=_CHANNEL_MARKERS["nsd"],
                       s=50, label="NSD", zorder=3, alpha=0.85)

            # y-axis: show λ values
            if "lambda_comms" in grp_cols:
                lam_vals = paired["lambda_comms"].values
                zlabels = [f"λ={v:.1e}" for v in lam_vals]
                if "z_dim" in grp_cols:
                    z_vals = paired["z_dim"].values
                    zlabels = [f"λ={l:.1e}, z={z}" for l, z in zip(lam_vals, z_vals)]
                ax.set_yticks(y_pos)
                ax.set_yticklabels(zlabels, fontsize=7)
            else:
                ax.set_yticks([])

            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Success rate")
            ax.set_title(
                f"Paired SR: SD vs NSD  (δ={delta_fixed})\n"
                "Each row = one config; lines connect matched configs"
            )
            ax.legend(fontsize=8, framealpha=0.85, loc="lower right")
            ax.grid(True, alpha=0.2, axis="x")
        else:
            ax.text(0.5, 0.5, "No matched pairs found",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Paired comparison requires\nlambda_comms column",
                ha="center", va="center", transform=ax.transAxes)

    # ── Right panel: Δ(SR) vs Δ(bits) per config ─────────────────────────────
    ax = axes[1]
    if not paired.empty if "paired" in dir() else False:
        pass
    elif grp_cols and not sd_all.empty and not nsd_all.empty:
        sd_agg = sd_all.groupby(grp_cols)[[sr_col, bits_col]].mean().reset_index()
        nsd_agg = nsd_all.groupby(grp_cols)[[sr_col, bits_col]].mean().reset_index()
        paired = sd_agg.merge(nsd_agg, on=grp_cols, suffixes=("_sd", "_nsd"))

    if "paired" in dir() and not paired.empty:
        delta_sr = paired[f"{sr_col}_sd"] - paired[f"{sr_col}_nsd"]
        delta_bits = paired[f"{bits_col}_sd"] - paired[f"{bits_col}_nsd"]

        # Color by z_dim if available
        if "z_dim" in paired.columns:
            for z in sorted(paired["z_dim"].unique()):
                mask = paired["z_dim"] == z
                ax.scatter(delta_bits[mask], delta_sr[mask],
                           s=60, alpha=0.8, label=f"z_dim={z}", zorder=3)
        else:
            ax.scatter(delta_bits, delta_sr,
                       color="mediumpurple", s=60, alpha=0.8, zorder=3)

        ax.axhline(0, color="grey", linewidth=1.0, linestyle="--")
        ax.axvline(0, color="grey", linewidth=1.0, linestyle="--")

        # Label quadrants
        ax.text(0.02, 0.98, "SD fewer bits\nSD higher SR",
                transform=ax.transAxes, fontsize=7, va="top",
                color="steelblue", alpha=0.7)
        ax.text(0.98, 0.02, "NSD fewer bits\nNSD higher SR",
                transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
                color="darkorange", alpha=0.7)

        ax.set_xlabel("Δ bits (SD − NSD)")
        ax.set_ylabel("Δ SR (SD − NSD)")
        ax.set_title(
            f"Per-config SD−NSD difference  (δ={delta_fixed})\n"
            "Upper-left = SD better on both metrics"
        )
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "Insufficient data for Δ plot",
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
    out_dir: str | Path = "results/figures",
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
    agg     : per-config aggregated DataFrame from load_runs.seed_aggregate
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

    try:
        plot_appendix_lambda_sensitivity(
            summary, channels=["sd", "nsd"],
            delta_fixed=1.0, z_dim_fixed=2,
            out_dir=app_dir,
        )
        print("  appA_lambda_sensitivity — OK")
    except Exception as e:
        print(f"  appA FAILED: {e}")

    try:
        plot_appendix_delta_lambda_heatmap(
            summary, channel="sd", z_dim_fixed=2, out_dir=app_dir
        )
        print("  appB_delta_lambda_heatmap — OK")
    except Exception as e:
        print(f"  appB FAILED: {e}")

    try:
        plot_appendix_training_dynamics(
            df, channels=["sd", "nsd", "additive_uniform", "none"],
            delta=1.0, lambda_comms=5e-4, z_dim=2,
            out_dir=app_dir,
        )
        print("  appC_training_dynamics — OK")
    except Exception as e:
        print(f"  appC FAILED: {e}")

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

    try:
        plot_appendix_overhead_above_hg(summary, z_dim_fixed=2, out_dir=app_dir)
        print("  appE_overhead_above_hg — OK")
    except Exception as e:
        print(f"  appE FAILED: {e}")

    if "bits_per_msg" in summary.columns and "true_bits_per_msg" in summary.columns:
        try:
            plot_appendix_surrogate_calibration(summary, z_dim_fixed=2, out_dir=app_dir)
            print("  appF_surrogate_calibration — OK")
        except Exception as e:
            print(f"  appF FAILED: {e}")
    else:
        print("  appF SKIPPED (missing bits columns)")

    if "z_dim" in summary.columns:
        try:
            plot_appendix_zdim_scaling(summary, out_dir=app_dir)
            print("  appG_zdim_scaling — OK")
        except Exception as e:
            print(f"  appG FAILED: {e}")
    else:
        print("  appG SKIPPED (no z_dim column)")

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

    # P2 analysis figures (only if P2 columns are present)
    p2_dir = Path(out_dir) / "p2"
    if "entropy_rate" in df.columns:
        for fn, key in [
            (plot_p2_shannon_gap,      "p2_shannon_gap"),
            (plot_p2_qphi_gap,         "p2_qphi_gap"),
            (plot_p2_context_bounds,   "p2_context_bounds"),
            (plot_p2_gradient_balance, "p2_gradient_balance"),
        ]:
            try:
                fig, _ = fn(df, out_dir=p2_dir)
                plt.close(fig)
                print(f"  {key} — OK")
            except Exception as e:
                print(f"  {key} FAILED: {e}")

    if "entropy_rate" in summary.columns and any(
        c.startswith("H_dim_") for c in summary.columns
    ):
        try:
            fig, _ = plot_p2_per_dim_entropy(summary, out_dir=p2_dir)
            plt.close(fig)
            print("  p2_per_dim_entropy — OK")
        except Exception as e:
            print(f"  p2_per_dim_entropy FAILED: {e}")

    print(f"\nDone. Figures in:\n  main/     → {main_dir}\n  appendix/ → {app_dir}\n  p2/       → {p2_dir}")


# ---------------------------------------------------------------------------
# P2 ANALYSIS — Shannon gap over training
# ---------------------------------------------------------------------------

def plot_p2_shannon_gap(
    df: pd.DataFrame,
    x_col: str = "timestep",
    bits_col: str = "true_bits_per_msg",
    channel_col: str = "channel",
    smooth: int = 10,
    title: str = "Shannon gap over training (true bits − H(G))",
    out_dir: str | Path | None = None,
    stem: str = "p2_shannon_gap",
    save_path: str | Path | None = None,
) -> tuple:
    """P2 ANALYSIS — Shannon gap (true_bits − H(G)) over training.

    Shows whether and how fast P2 closes the gap to the Shannon limit.
    One line per (channel, config variant); shaded band = ±1 std over seeds.
    H(G) is shown as y=0 (the x-axis represents overhead above Shannon).
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    _gap_col = "shannon_gap"
    if _gap_col not in df.columns:
        if bits_col in df.columns:
            df = df.copy()
            df[_gap_col] = df[bits_col] - H_GOAL_BITS
        else:
            ax.text(0.5, 0.5, "No bits data", ha="center", transform=ax.transAxes)
            return fig, ax

    ddcl_channels = ["sd", "nsd", "additive_uniform"]
    work = df[df[channel_col].isin(ddcl_channels)].copy() if channel_col in df.columns else df.copy()
    if work.empty:
        ax.text(0.5, 0.5, "No DDCL data", ha="center", transform=ax.transAxes)
        return fig, ax

    # Group by channel, average over seeds
    for ch in ddcl_channels:
        ch_df = work[work[channel_col] == ch] if channel_col in work.columns else work
        if ch_df.empty or x_col not in ch_df.columns:
            continue
        grouped = ch_df.groupby(x_col)[_gap_col]
        m, s = grouped.mean(), grouped.std().fillna(0)
        x = m.index.values
        if smooth > 1 and len(m) > smooth:
            kernel = np.ones(smooth) / smooth
            m_plot = np.convolve(m.values, kernel, mode="valid")
            s_plot = np.convolve(s.values, kernel, mode="valid")
            x_plot = x[smooth - 1:]
        else:
            m_plot, s_plot, x_plot = m.values, s.values, x
        color = _CHANNEL_COLORS[ch]
        ax.plot(x_plot, m_plot, color=color, label=_CHANNEL_LABELS[ch])
        ax.fill_between(x_plot, m_plot - s_plot, m_plot + s_plot, alpha=0.15, color=color)

    ax.axhline(0, color="red", linewidth=1.3, linestyle="--",
               label=f"H(G) = {H_GOAL_BITS:.2f} bits  (zero overhead)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("true_bits/msg − H(G)  (bits)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# P2 ANALYSIS — qphi_gap convergence
# ---------------------------------------------------------------------------

def plot_p2_qphi_gap(
    df: pd.DataFrame,
    x_col: str = "timestep",
    gap_col: str = "qphi_gap",
    z_dim: int = 2,
    smooth: int = 10,
    title: str = "q_φ model fit: qphi_gap over training",
    out_dir: str | Path | None = None,
    stem: str = "p2_qphi_gap",
    save_path: str | Path | None = None,
) -> tuple:
    """P2 ANALYSIS — qphi_gap = entropy_rate − H_m_empirical.

    A converged prior should reach qphi_gap ≈ 0.27 × z_dim bits (DLM floor).
    A gap persistently above the floor signals a training problem.
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    if gap_col not in df.columns or x_col not in df.columns:
        ax.text(0.5, 0.5, f"No {gap_col} data", ha="center", transform=ax.transAxes)
        return fig, ax

    grouped = df.groupby(x_col)[gap_col]
    m, s = grouped.mean(), grouped.std().fillna(0)
    x = m.index.values
    if smooth > 1 and len(m) > smooth:
        kernel = np.ones(smooth) / smooth
        m_plot = np.convolve(m.values, kernel, mode="valid")
        s_plot = np.convolve(s.values, kernel, mode="valid")
        x_plot = x[smooth - 1:]
    else:
        m_plot, s_plot, x_plot = m.values, s.values, x

    ax.plot(x_plot, m_plot, color=_CHANNEL_COLORS["sd"], label="qphi_gap (mean ±1 std)")
    ax.fill_between(x_plot, m_plot - s_plot, m_plot + s_plot, alpha=0.15,
                    color=_CHANNEL_COLORS["sd"])

    dlm_floor = 0.27 * z_dim
    ax.axhline(dlm_floor, color="orange", linewidth=1.3, linestyle="--",
               label=f"DLM floor ≈ {dlm_floor:.2f} bits  (irreducible, {z_dim} dims)")
    ax.axhline(0, color="red", linewidth=0.8, linestyle=":", alpha=0.5,
               label="Ideal (q_φ = p(m))")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("qphi_gap  (bits)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# P2 ANALYSIS — Context A vs B rate bounds
# ---------------------------------------------------------------------------

def plot_p2_context_bounds(
    df: pd.DataFrame,
    x_col: str = "timestep",
    smooth: int = 10,
    title: str = "Entropy rate bounds over training: A (marginal) vs B (oracle)",
    out_dir: str | Path | None = None,
    stem: str = "p2_context_bounds",
    save_path: str | Path | None = None,
) -> tuple:
    """P2 ANALYSIS — Context A vs B rate bounds over training.

    Three lines: entropy_rate (A, what we optimise), entropy_rate_B (context-B
    oracle bound), H_m_empirical (true entropy).  context_gap = A − B ≈ I(z;m).
    As training progresses, A should approach B (oracle).
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    series = {
        "entropy_rate":   ("Context A (marginal prior)", _CHANNEL_COLORS["sd"],    "-"),
        "entropy_rate_B": ("Context B oracle (H(m|z))", _CHANNEL_COLORS["nsd"],   "--"),
        "H_m_empirical":  ("H(m) empirical",             _CHANNEL_COLORS["additive_uniform"], ":"),
    }
    any_plotted = False
    for col, (label, color, ls) in series.items():
        if col not in df.columns or x_col not in df.columns:
            continue
        grouped = df.groupby(x_col)[col]
        m, s = grouped.mean(), grouped.std().fillna(0)
        x = m.index.values
        if smooth > 1 and len(m) > smooth:
            kernel = np.ones(smooth) / smooth
            m_plot = np.convolve(m.values, kernel, mode="valid")
            s_plot = np.convolve(s.values, kernel, mode="valid")
            x_plot = x[smooth - 1:]
        else:
            m_plot, s_plot, x_plot = m.values, s.values, x
        ax.plot(x_plot, m_plot, color=color, linestyle=ls, label=label)
        ax.fill_between(x_plot, m_plot - s_plot, m_plot + s_plot,
                        alpha=0.10, color=color)
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No entropy_rate data", ha="center", transform=ax.transAxes)
        return fig, ax

    ax.axhline(H_GOAL_BITS, color="red", linewidth=1.2, linestyle="--",
               label=f"H(G) = {H_GOAL_BITS:.2f} bits  (task entropy)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Bits / message element")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# P2 ANALYSIS — Per-dimension entropy bar chart
# ---------------------------------------------------------------------------

def plot_p2_per_dim_entropy(
    summary: pd.DataFrame,
    channel_col: str = "channel",
    z_dim_fixed: int = 2,
    title: str = "Per-dimension message entropy at convergence",
    out_dir: str | Path | None = None,
    stem: str = "p2_per_dim_entropy",
    save_path: str | Path | None = None,
) -> tuple:
    """P2 ANALYSIS — Per-dimension H(m_k) at convergence.

    Shows how information is distributed across message dimensions.  Uniform
    H_k suggests factored encoding; concentrated H_k may benefit P1 per-channel δ.
    """
    _require_mpl()
    _paper_style()

    h_cols = sorted(
        [c for c in summary.columns if c.startswith("H_dim_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not h_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No H_dim_* columns", ha="center", transform=ax.transAxes)
        return fig, ax

    work = summary.copy()
    if "z_dim" in work.columns:
        work = work[work["z_dim"] == z_dim_fixed]

    ddcl_channels = [c for c in ["sd", "nsd", "additive_uniform"]
                     if channel_col not in work.columns
                     or c in work[channel_col].values]

    n_dims = len(h_cols)
    x = np.arange(n_dims)
    width = 0.8 / max(len(ddcl_channels), 1)

    fig, ax = plt.subplots(figsize=(max(5, n_dims * 1.8), 4))
    for i, ch in enumerate(ddcl_channels):
        ch_df = work[work[channel_col] == ch] if channel_col in work.columns else work
        if ch_df.empty:
            continue
        means = [ch_df[c].mean() for c in h_cols]
        stds  = [ch_df[c].std() for c in h_cols]
        offset = (i - len(ddcl_channels) / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9, yerr=stds,
               color=_CHANNEL_COLORS[ch], alpha=0.85,
               label=_CHANNEL_LABELS[ch], capsize=4)

    ax.axhline(H_GOAL_BITS / n_dims, color="red", linewidth=1.2, linestyle="--",
               label=f"H(G)/{n_dims} = {H_GOAL_BITS/n_dims:.2f} bits  (equal split)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"dim {k}" for k in range(n_dims)])
    ax.set_ylabel("H(m_k)  (bits)")
    ax.set_title(f"{title}  [z_dim={z_dim_fixed}]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# P2 ANALYSIS — Gradient balance
# ---------------------------------------------------------------------------

def plot_p2_gradient_balance(
    df: pd.DataFrame,
    x_col: str = "timestep",
    grad_col: str = "speaker_grad_norm",
    ent_col: str = "entropy_loss_magnitude",
    smooth: int = 10,
    title: str = "Speaker gradient norm and entropy loss magnitude over training",
    out_dir: str | Path | None = None,
    stem: str = "p2_gradient_balance",
    save_path: str | Path | None = None,
) -> tuple:
    """P2 ANALYSIS — Gradient health diagnostic.

    Two-panel: top = speaker_grad_norm (total gradient reaching the speaker);
    bottom = entropy_loss_magnitude (λ × NLL, the P2 compression signal).
    If entropy_loss_magnitude is orders of magnitude below the grad norm,
    P2 is not influencing the speaker (gradient dead zone, PILLAR_P2.md §9).
    """
    _require_mpl()
    _paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax_grad, ax_ent = axes

    any_data = False
    for ax, col, label, color in [
        (ax_grad, grad_col, "Speaker grad norm (L2)", _CHANNEL_COLORS["sd"]),
        (ax_ent,  ent_col,  "Entropy loss magnitude (λ·NLL_bwd)", _CHANNEL_COLORS["nsd"]),
    ]:
        if col not in df.columns or x_col not in df.columns:
            ax.text(0.5, 0.5, f"No {col}", ha="center", transform=ax.transAxes)
            continue
        grouped = df.groupby(x_col)[col]
        m, s = grouped.mean(), grouped.std().fillna(0)
        x = m.index.values
        if smooth > 1 and len(m) > smooth:
            kernel = np.ones(smooth) / smooth
            m_plot = np.convolve(m.values, kernel, mode="valid")
            s_plot = np.convolve(s.values, kernel, mode="valid")
            x_plot = x[smooth - 1:]
        else:
            m_plot, s_plot, x_plot = m.values, s.values, x
        ax.plot(x_plot, m_plot, color=color, label=label)
        ax.fill_between(x_plot, m_plot - s_plot, m_plot + s_plot,
                        alpha=0.15, color=color)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        any_data = True

    if not any_data:
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No gradient data", ha="center", transform=ax.transAxes)
        return fig, ax

    ax_ent.set_xlabel("Environment steps")
    axes[0].set_title(title)
    plt.tight_layout()

    if out_dir is not None:
        _save(fig, Path(out_dir), stem)
    elif save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes
