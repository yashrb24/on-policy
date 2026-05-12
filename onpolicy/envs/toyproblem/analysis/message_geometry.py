"""message_geometry.py — Visualise how learned z-vectors map to goals and messages.

Produces a 4-panel figure for up to 3 checkpoints (Phase 1, Phase 2 stable, Phase 2
collapsed) showing:

  Panel A — z-space scatter (2D projections dim0 vs dim1), coloured by goal.
             Bin grid lines drawn at integer multiples of δ.

  Panel B — Message space: each integer tuple (m_0, m_1) scatter (marginalising m_2),
             coloured by goal. Shows how many distinct symbols each goal owns.

  Panel C — Frac distribution: histogram of frac(z_k / δ) across all dims and goals.
             Near-0 or near-1 means Phase 2 has converged; near-0.5 means it hasn't.

  Panel D — Message-goal confusion matrix: for each observed message tuple, what
             fraction of the time does it come from each goal? Diagonal = clean mapping;
             off-diagonal = bin-crossing / policy collapse.

Usage (from repo root):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.analysis.message_geometry \\
        --checkpoints \\
            runs/sc_ablation/sc_posthoc_mag/0/final.pt \\
            runs/sc_ablation/sc_twophase_dither1e-3/0/final.pt \\
            runs/sc_ablation/sc_twophase_dither1e-4/0/final.pt \\
        --labels "Phase 1" "Phase 2 (λ=1e-3, stable)" "Phase 2 (λ=1e-4, collapsed)" \\
        --n_samples 500 --delta 1.0 --z_dim 3 --out figures/message_geometry.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from onpolicy.envs.toyproblem.CommunicatingGoal_env import _DEFAULT_GOALS
from onpolicy.envs.toyproblem.channels import build_channel
from onpolicy.envs.toyproblem.network import SpeakerNetwork


# ── Colour palette (6 goals) ────────────────────────────────────────────────
_GOAL_COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
_GOALS = np.array(_DEFAULT_GOALS)  # (6, 2)


# ── Checkpoint loading ───────────────────────────────────────────────────────

def _load_speaker(ckpt_path: Path, z_dim: int, hidden: int = 64) -> SpeakerNetwork:
    state = torch.load(ckpt_path, map_location="cpu")
    speaker = SpeakerNetwork(obs_dim=2, z_dim=z_dim, hidden=hidden)
    if "speaker" in state:
        speaker.load_state_dict(state["speaker"])
    elif "state_dict" in state:
        prefix = "speaker."
        sd = {k[len(prefix):]: v for k, v in state["state_dict"].items()
              if k.startswith(prefix)}
        speaker.load_state_dict(sd)
    else:
        raise KeyError(f"Cannot find speaker weights; keys={list(state.keys())}")
    return speaker.eval()


# ── Sample collection ────────────────────────────────────────────────────────

@torch.no_grad()
def collect_per_goal(
    speaker: SpeakerNetwork,
    channel,
    delta: float,
    n_samples: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return z and m arrays per goal.

    Returns:
        z_per_goal : {goal_idx → (n_samples, z_dim) float array}
        m_per_goal : {goal_idx → (n_samples, z_dim) int array}
    """
    goals_t = torch.tensor(_GOALS, dtype=torch.float32)
    z_per_goal, m_per_goal = {}, {}

    for gid, goal_pos in enumerate(goals_t):
        g = goal_pos.unsqueeze(0).expand(n_samples, -1)
        z = speaker(g)
        _, info = channel(z)
        m = info["m"].long()

        # Add small noise so individual samples are visible in scatter
        z_jit = z + torch.randn_like(z) * 0.02

        z_per_goal[gid] = z_jit.numpy()
        m_per_goal[gid] = m.numpy()

    return z_per_goal, m_per_goal


# ── Panel helpers ────────────────────────────────────────────────────────────

def _draw_bin_grid(ax, xlim, ylim, delta):
    """Draw bin boundary grid lines at integer multiples of δ."""
    for x in np.arange(np.floor(xlim[0]), np.ceil(xlim[1]) + delta, delta):
        ax.axvline(x, color="gray", lw=0.4, alpha=0.4, zorder=0)
    for y in np.arange(np.floor(ylim[0]), np.ceil(ylim[1]) + delta, delta):
        ax.axhline(y, color="gray", lw=0.4, alpha=0.4, zorder=0)


def panel_z_scatter(ax, z_per_goal: dict, delta: float, dim_x: int = 0, dim_y: int = 1,
                    title: str = "", clip_percentile: float = 99.0):
    """Panel A: z-space scatter with bin grid.

    Clips axis to clip_percentile of the data range so outlier blowup
    doesn't collapse the scale; annotates if clipping occurred.
    """
    all_x, all_y = [], []
    for gid, z in z_per_goal.items():
        x, y = z[:, dim_x], z[:, dim_y]
        ax.scatter(x, y, s=8, alpha=0.5, color=_GOAL_COLOURS[gid], zorder=2)
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    all_x_np, all_y_np = np.array(all_x), np.array(all_y)
    pad = 0.5
    xlo = np.percentile(all_x_np, 100 - clip_percentile) - pad
    xhi = np.percentile(all_x_np, clip_percentile) + pad
    ylo = np.percentile(all_y_np, 100 - clip_percentile) - pad
    yhi = np.percentile(all_y_np, clip_percentile) + pad

    # Detect blowup: if true range is much larger than clipped range, annotate
    true_xrange = all_x_np.max() - all_x_np.min()
    clipped_xrange = xhi - xlo
    if true_xrange > clipped_xrange * 1.5:
        ax.text(0.97, 0.03, f"range clipped\n(true: [{all_x_np.min():.0f},{all_x_np.max():.0f}])",
                transform=ax.transAxes, fontsize=5, ha="right", va="bottom",
                color="red", alpha=0.8)

    xlim = (xlo, xhi)
    ylim = (ylo, yhi)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _draw_bin_grid(ax, xlim, ylim, delta)

    ax.set_xlabel(f"z_{dim_x}", fontsize=8)
    ax.set_ylabel(f"z_{dim_y}", fontsize=8)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=7)


def panel_message_scatter(ax, m_per_goal: dict, title: str = ""):
    """Panel B: message-space scatter (m_0 vs m_1), coloured by goal."""
    for gid, m in m_per_goal.items():
        # Jitter integer positions slightly for visibility
        jx = m[:, 0] + np.random.uniform(-0.15, 0.15, len(m))
        jy = m[:, 1] + np.random.uniform(-0.15, 0.15, len(m))
        ax.scatter(jx, jy, s=8, alpha=0.4, color=_GOAL_COLOURS[gid], zorder=2)

    ax.set_xlabel("m_0", fontsize=8)
    ax.set_ylabel("m_1", fontsize=8)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))


def panel_frac_histogram(ax, z_per_goal: dict, delta: float, title: str = ""):
    """Panel C: frac(z/δ) distribution across all goals and dims."""
    all_fracs = []
    for z in z_per_goal.values():
        frac = (z / delta) - np.floor(z / delta)
        all_fracs.append(frac.ravel())
    all_fracs = np.concatenate(all_fracs)

    ax.hist(all_fracs, bins=40, color="#444", edgecolor="none", alpha=0.8)
    ax.set_xlabel("frac(z / δ)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.axvline(0.5, color="red", lw=1, ls="--", alpha=0.7, label="frac=0.5 (max noise)")
    ax.axvspan(0.0, 0.1, alpha=0.12, color="green", label="low noise (frac<0.1)")
    ax.axvspan(0.9, 1.0, alpha=0.12, color="green")
    ax.legend(fontsize=6, loc="upper center")


def panel_confusion(ax, m_per_goal: dict, title: str = ""):
    """Panel D: confusion matrix — for each message tuple, which goals use it?

    Rows = observed joint tuples (sorted), cols = goals.
    Cell value = fraction of samples for that tuple coming from that goal.
    """
    n_goals = len(_GOALS)
    # Aggregate: tuple → goal → count
    from collections import defaultdict
    tuple_goal_counts: dict[tuple, list[int]] = defaultdict(lambda: [0] * n_goals)

    for gid, m in m_per_goal.items():
        for row in m:
            key = tuple(row.tolist())
            tuple_goal_counts[key][gid] += 1

    # Sort tuples and build matrix
    tuples = sorted(tuple_goal_counts.keys())
    matrix = np.array([tuple_goal_counts[t] for t in tuples], dtype=float)  # (K, n_goals)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix /= row_sums  # normalise rows to fractions

    im = ax.imshow(matrix.T, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xlabel("message tuple index", fontsize=8)
    ax.set_ylabel("goal index", fontsize=8)
    ax.set_yticks(range(n_goals))
    ax.set_yticklabels([f"g{i} {tuple(_GOALS[i].tolist())}" for i in range(n_goals)],
                       fontsize=6)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6)

    plt.colorbar(im, ax=ax, fraction=0.03, label="fraction", pad=0.02)
    n_tuples = len(tuples)
    ax.set_xticks(range(0, n_tuples, max(1, n_tuples // 8)))


# ── Main figure ──────────────────────────────────────────────────────────────

def make_figure(
    checkpoints: list[Path],
    labels: list[str],
    delta: float,
    z_dim: int,
    n_samples: int,
    out_path: Path,
) -> None:
    channel = build_channel("sd", delta, ste_clip=10.0)
    n_ckpts = len(checkpoints)

    # 4 rows × n_ckpts columns
    fig, axes = plt.subplots(4, n_ckpts, figsize=(5 * n_ckpts, 16),
                             constrained_layout=True)
    if n_ckpts == 1:
        axes = axes[:, np.newaxis]

    # Goal legend (shared)
    legend_patches = [
        mpatches.Patch(color=_GOAL_COLOURS[i],
                       label=f"goal {i}: {tuple(_GOALS[i].tolist())}")
        for i in range(len(_GOALS))
    ]

    for col, (ckpt, label) in enumerate(zip(checkpoints, labels)):
        print(f"[msg_geom] loading {ckpt.name} ({label})...")
        speaker = _load_speaker(ckpt, z_dim=z_dim)

        print(f"[msg_geom]   collecting {n_samples} samples per goal...")
        z_per_goal, m_per_goal = collect_per_goal(speaker, channel, delta, n_samples)

        panel_z_scatter(axes[0, col], z_per_goal, delta,
                        title=f"A. z-space (dim 0 vs 1)\n{label}")
        panel_message_scatter(axes[1, col], m_per_goal,
                              title=f"B. Message space (m_0 vs m_1)\n{label}")
        panel_frac_histogram(axes[2, col], z_per_goal, delta,
                             title=f"C. frac(z/δ) distribution\n{label}")
        panel_confusion(axes[3, col], m_per_goal,
                        title=f"D. Message-goal confusion\n{label}")

    # Add legend to top-left panel
    axes[0, 0].legend(handles=legend_patches, fontsize=6, loc="upper right",
                      framealpha=0.7)

    fig.suptitle(
        "Message geometry: z-space, message-space, dither noise, and goal confusion",
        fontsize=11, fontweight="bold", y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[msg_geom] saved → {out_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Message geometry visualisation.")
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--labels", nargs="+", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--z_dim", type=int, default=3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--out", type=Path,
                   default=Path("results/figures/message_geometry.pdf"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    labels = args.labels or [f"ckpt_{i}" for i in range(len(args.checkpoints))]
    if len(labels) != len(args.checkpoints):
        raise ValueError("--labels count must match --checkpoints count")

    make_figure(
        checkpoints=args.checkpoints,
        labels=labels,
        delta=args.delta,
        z_dim=args.z_dim,
        n_samples=args.n_samples,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
