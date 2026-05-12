"""geometry_phase2_figures.py — F6, F7 (geometry comparison) and F16 (Phase 2 curves).

  F6 — Per-goal z-space geometry: cosine-similarity Gram matrix between goal centroids
       across four configs (no_anchor, mag_only, dither_only_v2, both_v2). Shows how
       the magnitude anchor and dither loss reshape the z manifold.

  F7 — Rate decomposition per geometry config: stacked bar (H(G)|H_dither|TC|ε) for
       each config, highlighting how dither reduces both H_dither AND TC.

  F16 — Phase 2 training curves: hist_H_empirical and success_rate vs update for
        E44–E46 (sc_twophase_dither{1e-4,5e-4,1e-3}_v2). Shows hist_H decreasing
        after Phase 2 triggers (unlike the REINTERPRETED E12–E14 runs).

Usage (from repo root):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.analysis.geometry_phase2_figures \\
        --post_hoc_dir results/post_hoc \\
        --ckpt_dir runs/sc_ablation \\
        --runs_dir runs/sc_ablation \\
        --out_dir results/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from onpolicy.envs.toyproblem.channels import build_channel, H_GOAL_BITS
from onpolicy.envs.toyproblem.CommunicatingGoal_env import _DEFAULT_GOALS
from onpolicy.envs.toyproblem.network import SpeakerNetwork

_H_G = H_GOAL_BITS

_RC = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.8,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

_GEOM_CONFIGS = [
    ("geom_no_anchor",    "No anchor\n(RL only)"),
    ("geom_mag_only",     "Mag anchor\n(Phase 1)"),
    ("geom_dither_only_v2", "Dither only\n(corrected)"),
    ("geom_both_v2",      "Mag + Dither\n(both)"),
]

_PHASE2_CONFIGS = [
    ("sc_twophase_dither1e-4_v2", "λ=1e-4", "#2c7bb6"),
    ("sc_twophase_dither5e-4_v2", "λ=5e-4", "#fdae61"),
    ("sc_twophase_dither1e-3_v2", "λ=1e-3", "#d7191c"),
]


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _load_post_hoc(post_hoc_dir: Path, exp_name: str) -> list[dict]:
    seeds = sorted(post_hoc_dir.glob(f"{exp_name}/seed_*.json"))
    return [json.loads(p.read_text()) for p in seeds]


def _load_speaker(ckpt_path: Path, z_dim: int = 3, hidden: int = 64) -> SpeakerNetwork:
    state = torch.load(ckpt_path, map_location="cpu")
    speaker = SpeakerNetwork(obs_dim=2, z_dim=z_dim, hidden=hidden)
    if "speaker" in state:
        speaker.load_state_dict(state["speaker"])
    else:
        prefix = "speaker."
        sd = {k[len(prefix):]: v for k, v in state["state_dict"].items() if k.startswith(prefix)}
        speaker.load_state_dict(sd)
    return speaker.eval()


def _load_metrics(runs_dir: Path, exp_name: str, n_seeds: int = 5) -> list[dict[str, np.ndarray]]:
    seeds_data = []
    for seed in range(n_seeds):
        path = runs_dir / exp_name / str(seed) / "metrics.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        if not rows:
            continue
        cols: dict[str, np.ndarray] = {}
        for key in rows[0]:
            vals = []
            for r in rows:
                v = r[key]
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(float("nan"))
            cols[key] = np.array(vals)
        seeds_data.append(cols)
    return seeds_data


# ---------------------------------------------------------------------------
# F6 — z-space Gram matrix
# ---------------------------------------------------------------------------

def fig_f6_z_geometry(
    ckpt_dir: Path,
    out_dir: Path,
    seed: int = 0,
    z_dim: int = 3,
    hidden: int = 64,
    n_samples: int = 500,
) -> None:
    """F6: cosine-similarity Gram matrix between goal centroids for 4 geometry configs."""
    goals_t = torch.tensor(_DEFAULT_GOALS, dtype=torch.float32)
    n_goals = len(_DEFAULT_GOALS)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(_GEOM_CONFIGS), figsize=(14, 3.8))

        for ax, (exp_name, label) in zip(axes, _GEOM_CONFIGS):
            ckpt_path = ckpt_dir / exp_name / str(seed) / "final.pt"
            if not ckpt_path.exists():
                ax.set_visible(False)
                continue

            speaker = _load_speaker(ckpt_path, z_dim, hidden)
            with torch.no_grad():
                # Deterministic speaker: expand each goal to get a stable centroid
                centroids = []
                for gid in range(n_goals):
                    g = goals_t[gid].unsqueeze(0).expand(n_samples, -1)
                    z = speaker(g)  # (n_samples, z_dim) — all identical (deterministic)
                    centroids.append(z[0].numpy())  # speaker is deterministic

            # Pairwise cosine similarity matrix
            C = np.stack(centroids)   # (n_goals, z_dim)
            norms = np.linalg.norm(C, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            C_norm = C / norms
            gram = C_norm @ C_norm.T   # (n_goals, n_goals)

            im = ax.imshow(gram, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
            ax.set_xticks(range(n_goals))
            ax.set_yticks(range(n_goals))
            ax.set_xticklabels([f"G{i}" for i in range(n_goals)], fontsize=8)
            ax.set_yticklabels([f"G{i}" for i in range(n_goals)], fontsize=8)
            ax.set_title(label, fontsize=10)
            # Annotate cells
            for i in range(n_goals):
                for j in range(n_goals):
                    ax.text(j, i, f"{gram[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if abs(gram[i,j]) < 0.6 else "white")

        # Shared colourbar
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.02)
        cbar.set_label("Cosine similarity", fontsize=9)

        fig.suptitle("F6 — z-space goal geometry: cosine similarity between goal centroids\n"
                     "(dither + mag anchor push goals apart)",
                     fontsize=11, y=1.03)
        fig.tight_layout()
        _save(fig, out_dir, "F6_z_geometry")

    print(f"[F6] saved to {out_dir}/F6_z_geometry.{{pdf,png}}")


# ---------------------------------------------------------------------------
# F7 — Rate decomposition per geometry config
# ---------------------------------------------------------------------------

def fig_f7_geometry_decomposition(
    post_hoc_dir: Path,
    out_dir: Path,
) -> None:
    """F7: stacked bar rate decomposition across the 4 geometry configs."""
    configs = _GEOM_CONFIGS
    means: dict[str, dict[str, float]] = {}
    sems:  dict[str, dict[str, float]] = {}

    fields = ["H_G_bits", "H_dither_bits", "TC_bits", "eps_estimator_bits",
              "H_factored_bits", "H_joint_bits"]

    for exp_name, _ in configs:
        data = _load_post_hoc(post_hoc_dir, exp_name)
        if not data:
            continue
        n = len(data)
        means[exp_name] = {f: float(np.mean([d[f] for d in data])) for f in fields}
        sems[exp_name]  = {f: float(np.std([d[f] for d in data], ddof=1) / math.sqrt(n)) for f in fields}

    labels = [lbl for exp_name, lbl in configs if exp_name in means]
    exp_names = [exp_name for exp_name, _ in configs if exp_name in means]

    bar_components = [
        ("H(G)",           "H_G_bits",           "#2c7bb6"),
        ("H(m|goal)",      "H_dither_bits",       "#abd9e9"),
        ("TC(m)",          "TC_bits",             "#d7191c"),
        ("ε_estimator",    "eps_estimator_bits",  "#fdae61"),
    ]

    x = np.arange(len(labels))
    bar_w = 0.5

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

        # Panel A: stacked bars
        bottoms = np.zeros(len(labels))
        for comp_label, field, colour in bar_components:
            vals = np.array([means[e][field] for e in exp_names])
            errs = np.array([sems[e][field]  for e in exp_names])
            ax_a.bar(x, vals, bottom=bottoms, width=bar_w, color=colour,
                     label=comp_label, edgecolor="white", linewidth=0.5)
            # Label each segment if large enough
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 0.3:
                    ax_a.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                              fontsize=8, color="black")
            bottoms += vals

        # H_joint dots on top
        for xi, e in enumerate(exp_names):
            hj = means[e]["H_joint_bits"]
            ax_a.plot(xi, hj, "D", color="black", ms=7, zorder=5,
                      label="H_joint" if xi == 0 else None)
            ax_a.text(xi + 0.27, hj, f"{hj:.2f}", va="center", fontsize=8)

        ax_a.set_xticks(x)
        ax_a.set_xticklabels(labels, fontsize=9)
        ax_a.set_ylabel("Rate (bits per message)")
        ax_a.set_title("(A) Rate decomposition per geometry config")
        ax_a.legend(fontsize=8, loc="upper right")
        ax_a.axhline(_H_G, color="k", ls=":", lw=1.0, label="H(G)")

        # Panel B: TC and H_dither side-by-side (the key reduction metrics)
        w = 0.3
        tc_vals   = np.array([means[e]["TC_bits"]      for e in exp_names])
        hd_vals   = np.array([means[e]["H_dither_bits"] for e in exp_names])
        tc_errs   = np.array([sems[e]["TC_bits"]       for e in exp_names])
        hd_errs   = np.array([sems[e]["H_dither_bits"] for e in exp_names])

        ax_b.bar(x - w/2, tc_vals, width=w, color="#d7191c", label="TC(m)",
                 yerr=tc_errs, capsize=4)
        ax_b.bar(x + w/2, hd_vals, width=w, color="#abd9e9", label="H(m|goal)",
                 yerr=hd_errs, capsize=4)

        for xi, (tc, hd) in enumerate(zip(tc_vals, hd_vals)):
            ax_b.text(xi - w/2, tc + 0.12, f"{tc:.2f}", ha="center", fontsize=8)
            ax_b.text(xi + w/2, hd + 0.12, f"{hd:.2f}", ha="center", fontsize=8)

        ax_b.set_xticks(x)
        ax_b.set_xticklabels(labels, fontsize=9)
        ax_b.set_ylabel("Bits")
        ax_b.set_title("(B) TC and H(m|goal) per config\n(dither reduces both)")
        ax_b.legend(fontsize=9)

        fig.suptitle("F7 — Geometry ablation: dither loss reduces TC and H(m|goal); "
                     "magnitude + dither cooperate",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        _save(fig, out_dir, "F7_geometry_decomposition")

    print(f"[F7] saved to {out_dir}/F7_geometry_decomposition.{{pdf,png}}")
    for e, lbl in zip(exp_names, labels):
        print(f"     {lbl.replace(chr(10),' ')}: TC={means[e]['TC_bits']:.3f}  "
              f"H_dither={means[e]['H_dither_bits']:.3f}  H_joint={means[e]['H_joint_bits']:.3f}")


# ---------------------------------------------------------------------------
# F16 — Phase 2 training curves
# ---------------------------------------------------------------------------

def fig_f16_phase2_curves(
    runs_dir: Path,
    out_dir: Path,
    n_seeds: int = 5,
) -> None:
    """F16: hist_H_empirical and SR over training for E44-E46 v2 runs."""
    # Phase 1 baseline: use hist_H at update 54 (just before Phase 2 triggers)
    # from the v2 runs themselves, averaged over all λ and seeds.
    phase1_vals = []
    for exp_name, _, _ in _PHASE2_CONFIGS:
        sd = _load_metrics(runs_dir, exp_name, n_seeds)
        for s in sd:
            phases = s.get("training_phase", np.ones_like(s["update"]))
            # Last update still in Phase 1
            p1_mask = phases == 1.0
            if p1_mask.any():
                phase1_vals.append(s["hist_H_empirical"][p1_mask][-1])
    phase1_hist_H = float(np.nanmean(phase1_vals)) if phase1_vals else None

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

        for exp_name, label, colour in _PHASE2_CONFIGS:
            seeds_data = _load_metrics(runs_dir, exp_name, n_seeds)
            if not seeds_data:
                continue

            updates = seeds_data[0]["update"]

            # hist_H
            hh_list = [s["hist_H_empirical"] for s in seeds_data]
            hh = np.array(hh_list)
            mean_hh = np.nanmean(hh, axis=0)
            std_hh  = np.nanstd(hh,  axis=0)
            ax_a.plot(updates, mean_hh, color=colour, label=label)
            ax_a.fill_between(updates, mean_hh - std_hh, mean_hh + std_hh,
                              alpha=0.15, color=colour)

            # Success rate
            sr_list = [s["success_rate"] for s in seeds_data]
            sr = np.array(sr_list)
            mean_sr = np.nanmean(sr, axis=0)
            std_sr  = np.nanstd(sr,  axis=0)
            ax_b.plot(updates, mean_sr, color=colour, label=label)
            ax_b.fill_between(updates, mean_sr - std_sr, mean_sr + std_sr,
                              alpha=0.15, color=colour)

        # Phase 2 boundary (update 54 for all configs)
        phase2_update = 54
        for ax in (ax_a, ax_b):
            ax.axvline(phase2_update, color="black", ls="--", lw=1.2, label="Phase 2 start")

        # Phase 1 baseline on hist_H panel
        if phase1_hist_H is not None:
            ax_a.axhline(phase1_hist_H, color="#888888", ls=":", lw=1.2,
                         label=f"Phase 1 baseline ({phase1_hist_H:.2f} bits)")

        ax_a.axhline(_H_G, color="k", ls="-.", lw=1.0, label=f"H(G) = {_H_G:.2f} bits")
        ax_a.set_xlabel("Training update")
        ax_a.set_ylabel("H(m) empirical (bits)")
        ax_a.set_title("(A) H(m) decreases after Phase 2 triggers\n"
                       "(corrected dither loss: genuine H(m|goal) reduction)")
        ax_a.legend(fontsize=8)

        ax_b.set_xlabel("Training update")
        ax_b.set_ylabel("Success rate")
        ax_b.set_ylim(0, 1.05)
        ax_b.set_title("(B) SR maintained ≥ 0.98 throughout Phase 2\n"
                       "(E39 listener fix prevents collapse)")
        ax_b.legend(fontsize=8)

        fig.suptitle("F16 — Phase 2 (corrected dither loss + E39 fix): "
                     "H(m) reduced without SR collapse",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        _save(fig, out_dir, "F16_phase2_curves")

    print(f"[F16] saved to {out_dir}/F16_phase2_curves.{{pdf,png}}")

    # Print final values
    for exp_name, label, _ in _PHASE2_CONFIGS:
        seeds_data = _load_metrics(runs_dir, exp_name, n_seeds)
        if seeds_data:
            final_hh = float(np.nanmean([s["hist_H_empirical"][-1] for s in seeds_data]))
            final_sr = float(np.nanmean([s["success_rate"][-1]    for s in seeds_data]))
            print(f"     {label}: SR={final_sr:.3f}  hist_H={final_hh:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F6, F7, F16 geometry and Phase 2 figures.")
    p.add_argument("--post_hoc_dir", type=str, default="results/post_hoc")
    p.add_argument("--ckpt_dir",     type=str, default="runs/sc_ablation")
    p.add_argument("--runs_dir",     type=str, default="runs/sc_ablation")
    p.add_argument("--out_dir",      type=str, default="results/figures")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--figures",      type=str, default="F6,F7,F16")
    return p.parse_args()


def main() -> None:
    args = _parse()
    post_hoc_dir = Path(args.post_hoc_dir)
    ckpt_dir     = Path(args.ckpt_dir)
    runs_dir     = Path(args.runs_dir)
    out_dir      = Path(args.out_dir)
    figures      = {f.strip() for f in args.figures.split(",")}

    if "F6" in figures:
        print("\n=== F6: z-space geometry ===")
        fig_f6_z_geometry(ckpt_dir, out_dir, seed=args.seed)

    if "F7" in figures:
        print("\n=== F7: geometry decomposition ===")
        fig_f7_geometry_decomposition(post_hoc_dir, out_dir)

    if "F16" in figures:
        print("\n=== F16: Phase 2 training curves ===")
        fig_f16_phase2_curves(runs_dir, out_dir)


if __name__ == "__main__":
    main()
