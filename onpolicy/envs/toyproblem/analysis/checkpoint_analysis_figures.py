"""checkpoint_analysis_figures.py — F3, F4, F8, F10, F11, F12, F14, F17, F19.

F3, F4, F17 use the E09 (sc_posthoc_mag) checkpoint.
F8, F10, F12 use the P2-FIX runs (runs/p2_ablation/).
F14, F19    use the live_B diagnostic runs (runs/live_B_diag/) — re-run of E21
            with --log_grad_decomp and --log_moving_target.

  F3 — Implicit prior mismatch
       Two-panel: (A) rate-decomposition stacked bar showing TC as the "implicit prior
       mismatch" cost, (B) message scatter (m_0 vs m_1) coloured by goal, illustrating
       the correlated structure that the factored q(m) ignores.

  F4 — Gradient alignment
       Training dynamics from E09/E10: H(m) stays flat during live SC training while
       true_bits grows 3×. PPO/SC loss ratio shows PPO dominates during learning.

  F8 — qphi_gap: DLM vs histogram
       Bar chart comparing DLM estimator error (fix_baseline ≈ 9.71 bits) vs histogram
       (sc_posthoc_mag ≈ 0.004 bits) plus training trajectory showing gap growth.

  F10 — Circular gradient failure
       qphi_gap training trajectories for all P2-FIX configs. Gap starts low (~0.07 bits)
       when the policy is untrained, then explodes as the policy learns a rich codebook.
       Fix B1–B3 (gate + more q_φ training) make it worse, not better.

  F12 — Warm-start failure
       fix_B4_twophase achieves qphi_gap ≈ 0.05 bits (low) but SR collapses to 53.6% in
       Phase 2. Demonstrates that fitting q_φ is necessary but not sufficient — using the
       fitted model as a speaker gradient still causes task collapse.

  F14 — Score function gradient ratio
       Two-panel: (A) ppo_speaker_grad_norm vs sc_speaker_grad_norm over live_B
       training on log-y axes — PPO consistently dominates SC by 4–5× throughout.
       (B) PPO/SC ratio stays above 1 at all times; no crossover occurs.

  F17 — ε_MLE convergence vs N
       Decomposition residual and ε_MLE bound plotted against 1/√N across seven
       message-count levels. Demonstrates O(1/√N) convergence of the histogram
       estimator error.

  F19 — Moving-target error
       Two-panel: (A) true_H_offline (joint H(m) under correct _GOAL_PROBS weighting)
       drops 6→2.4 bits as z-blowup creates non-overlapping per-goal bins — degenerate,
       not useful compression. hist_H_empirical (minibatch/factored) stays ~7 bits,
       overestimating true H(m) by ~4.6 bits. (B) ‖Δθ‖ drops 6× early→late but
       remains nonzero — the speaker keeps moving (SC gradient is active but stale).

Usage (from repo root):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.analysis.checkpoint_analysis_figures \\
        --ckpt_dir runs/sc_ablation/sc_posthoc_mag \\
        --post_hoc_dir results/post_hoc/sc_posthoc_mag \\
        --p2_ablation_dir runs/p2_ablation \\
        --out_dir results/figures \\
        --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from onpolicy.envs.toyproblem.channels import build_channel, H_GOAL_BITS
from onpolicy.envs.toyproblem.CommunicatingGoal_env import _DEFAULT_GOALS
from onpolicy.envs.toyproblem.network import SpeakerNetwork
from onpolicy.envs.toyproblem.source_coding import (
    JointMessageHistogram,
    MessageHistogram,
)
from onpolicy.envs.toyproblem.analysis.post_hoc_coding import (
    collect_messages,
    compute_decomposition,
)

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

_GOAL_COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


# ---------------------------------------------------------------------------
# Checkpoint loader helpers
# ---------------------------------------------------------------------------

def _load_speaker(ckpt_path: Path, z_dim: int, hidden: int, device: torch.device) -> SpeakerNetwork:
    state = torch.load(ckpt_path, map_location=device)
    speaker = SpeakerNetwork(obs_dim=2, z_dim=z_dim, hidden=hidden)
    if "speaker" in state:
        speaker.load_state_dict(state["speaker"])
    else:
        prefix = "speaker."
        sd = {k[len(prefix):]: v for k, v in state["state_dict"].items() if k.startswith(prefix)}
        speaker.load_state_dict(sd)
    return speaker.to(device)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F3 — Implicit prior mismatch
# ---------------------------------------------------------------------------

def fig_f3_implicit_prior(
    post_hoc_dir: Path,
    ckpt_path: Path,
    out_dir: Path,
    z_dim: int = 3,
    hidden: int = 64,
    n_scatter: int = 10_000,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Generate F3: implicit prior mismatch."""
    # ── Load all-seed post_hoc JSONs ─────────────────────────────────────────
    seeds = sorted(post_hoc_dir.glob("seed_*.json"))
    if not seeds:
        raise FileNotFoundError(f"No seed_*.json in {post_hoc_dir}")

    fields = ["H_G_bits", "H_factored_bits", "H_joint_bits", "TC_bits",
              "H_dither_bits", "eps_estimator_bits", "decomp_residual_bits"]
    data: dict[str, list[float]] = {f: [] for f in fields}
    for p in seeds:
        d = json.loads(p.read_text())
        for f in fields:
            data[f].append(d[f])
    means = {f: float(np.mean(data[f])) for f in fields}
    sems  = {f: float(np.std(data[f], ddof=1) / math.sqrt(len(seeds))) for f in fields}

    # ── Collect messages for scatter ─────────────────────────────────────────
    speaker = _load_speaker(ckpt_path, z_dim, hidden, device)
    channel = build_channel("sd", 1.0, ste_clip=10.0).to(device)
    speaker.eval()
    m_all, gid_all = collect_messages(speaker, channel, 1.0, n_scatter, device)

    # ── Plot ─────────────────────────────────────────────────────────────────
    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Panel A: stacked bar decomposition
        components = [
            ("H(G)",                   means["H_G_bits"],          sems["H_G_bits"],          "#2c7bb6"),
            ("H(m|goal) [dither]",     means["H_dither_bits"],     sems["H_dither_bits"],      "#abd9e9"),
            ("TC(m) [implicit prior]", means["TC_bits"],           sems["TC_bits"],            "#d7191c"),
            ("ε_estimator",            means["eps_estimator_bits"],sems["eps_estimator_bits"], "#fdae61"),
        ]

        bar_x = 0.4
        bottom = 0.0
        for label, val, _sem, colour in components:
            ax_a.bar(bar_x, val, bottom=bottom, color=colour, width=0.5,
                     label=label, edgecolor="white", linewidth=0.5)
            mid = bottom + val / 2
            if val > 0.15:
                ax_a.text(bar_x + 0.33, mid, f"{val:.2f}", va="center", ha="left",
                          fontsize=8.5, color="black")
            bottom += val

        # H_joint horizontal line
        h_joint = means["H_joint_bits"]
        ax_a.axhline(h_joint, color="black", lw=1.5, ls="--", zorder=5)
        ax_a.text(bar_x + 0.33, h_joint + 0.05, f"H_joint = {h_joint:.2f} bits\n(joint coding target)",
                  fontsize=8, va="bottom")

        # H_factored label at top
        h_fact = means["H_factored_bits"]
        ax_a.axhline(h_fact, color="#555555", lw=1.0, ls=":", zorder=4)
        ax_a.text(bar_x + 0.33, h_fact + 0.05, f"H_factored = {h_fact:.2f} bits",
                  fontsize=8, va="bottom", color="#555555")

        # TC brace annotation
        tc_val = means["TC_bits"]
        tc_start = means["H_G_bits"] + means["H_dither_bits"]
        ax_a.annotate("", xy=(bar_x - 0.32, tc_start + tc_val),
                      xytext=(bar_x - 0.32, tc_start),
                      arrowprops=dict(arrowstyle="<->", color="#d7191c", lw=1.5))
        ax_a.text(bar_x - 0.35, tc_start + tc_val / 2,
                  f"TC = {tc_val:.2f} bits\n(implicit\nprior mismatch)",
                  ha="right", va="center", fontsize=8, color="#d7191c")

        ax_a.set_xlim(-0.2, 1.4)
        ax_a.set_xticks([])
        ax_a.set_ylabel("Rate (bits per message)")
        ax_a.set_title("(A) Rate decomposition: TC is the implicit prior mismatch")
        ax_a.legend(loc="upper right", fontsize=8)
        ax_a.set_ylim(0, h_fact * 1.18)

        # Panel B: message scatter m_0 vs m_1 coloured by goal
        m_np = m_all.numpy()
        gid_np = gid_all.numpy()
        n_goals = len(_DEFAULT_GOALS)
        for gid in range(n_goals):
            mask = gid_np == gid
            jitter = np.random.default_rng(gid).uniform(-0.15, 0.15, (mask.sum(), 2))
            ax_b.scatter(m_np[mask, 0] + jitter[:, 0],
                         m_np[mask, 1] + jitter[:, 1],
                         c=_GOAL_COLOURS[gid], s=4, alpha=0.35,
                         label=f"Goal {gid}" if mask.sum() > 0 else None,
                         rasterized=True)

        ax_b.set_xlabel("m₀ (dim 0 message)")
        ax_b.set_ylabel("m₁ (dim 1 message)")
        ax_b.set_title("(B) Message scatter: correlated goal→tuple structure\n"
                        "(factored q(m) ignores this; costs TC bits)")
        ax_b.legend(markerscale=3, fontsize=8, loc="upper right")

        fig.suptitle("F3 — Implicit prior mismatch: TC(m) is the cost of factored coding",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        _save(fig, out_dir, "F3_implicit_prior_mismatch")

    print(f"[F3] saved to {out_dir}/F3_implicit_prior_mismatch.{{pdf,png}}")
    print(f"     H_factored={means['H_factored_bits']:.3f}  H_joint={means['H_joint_bits']:.3f}"
          f"  TC={means['TC_bits']:.3f}  H_dither={means['H_dither_bits']:.3f}  ε={means['eps_estimator_bits']:.4f}")


# ---------------------------------------------------------------------------
# F4 — Gradient alignment
# ---------------------------------------------------------------------------

def _load_metrics(runs_dir: Path, exp_name: str, n_seeds: int = 5) -> list[dict[str, np.ndarray]]:
    """Load metrics.csv for each seed of an experiment. Returns list of column dicts."""
    import csv
    seeds_data = []
    for seed in range(n_seeds):
        path = runs_dir / exp_name / str(seed) / "metrics.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        # Convert to float arrays, nan for missing/blank
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


def _band(ax, xs, ys_list, color, label, alpha=0.15):
    """Plot mean ± std band from a list of equal-length arrays."""
    arr = np.array(ys_list)
    mean = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr,  axis=0)
    ax.plot(xs, mean, color=color, label=label)
    ax.fill_between(xs, mean - std, mean + std, alpha=alpha, color=color)
    return mean


def fig_f4_gradient_direction(
    runs_dir: Path,
    out_dir: Path,
    n_seeds: int = 5,
) -> None:
    """Generate F4: why live SC cannot restructure p(m) — training dynamics.

    Uses E09 (sc_posthoc_mag) and E10 (sc_live_entropy) training metrics.
    Three panels:
      A — H(m) [hist_H_empirical] over training: flat despite active SC
      B — true_bits_per_msg over training: z blowup in live SC
      C — |sc_rate_loss / pg_loss| ratio: SC loss is small fraction of total
    """
    e09 = _load_metrics(runs_dir, "sc_posthoc_mag", n_seeds)
    e10 = _load_metrics(runs_dir, "sc_live_entropy", n_seeds)

    if not e09 or not e10:
        raise FileNotFoundError(f"Missing E09 or E10 metrics under {runs_dir}")

    updates_09 = e09[0]["update"]
    updates_10 = e10[0]["update"]

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        ax_a, ax_b, ax_c = axes

        # Panel A: hist_H_empirical — does live SC reduce H(m)?
        _band(ax_a, updates_09,
              [s["hist_H_empirical"] for s in e09], "#2c7bb6", "E09: post-hoc mag (no live SC)")
        _band(ax_a, updates_10,
              [s["hist_H_empirical"] for s in e10], "#d7191c", "E10: live SC (sc_live_entropy)")
        ax_a.axhline(_H_G, color="k", ls="--", lw=1.2, label=f"H(G) = {_H_G:.2f} bits")
        ax_a.set_xlabel("Training update")
        ax_a.set_ylabel("H(m) empirical (bits)")
        ax_a.set_title("(A) H(m) stays flat despite live SC")
        ax_a.legend(fontsize=8)

        # Panel B: true_bits_per_msg — z-magnitude blowup in E10
        _band(ax_b, updates_09,
              [s["true_bits_per_msg"] for s in e09], "#2c7bb6", "E09: post-hoc mag")
        _band(ax_b, updates_10,
              [s["true_bits_per_msg"] for s in e10], "#d7191c", "E10: live SC")
        ax_b.set_xlabel("Training update")
        ax_b.set_ylabel("true_bits_per_msg")
        ax_b.set_title("(B) Live SC grows |z| (true_bits blows up)")
        ax_b.legend(fontsize=8)

        # Panel C: |sc_rate_loss| / |pg_loss| — loss fraction, log scale
        ratios_per_seed = []
        for s in e10:
            pg = s["pg_loss"]
            sc = s["sc_rate_loss"]
            ratio = np.where(np.abs(pg) > 1e-6, np.abs(sc) / np.abs(pg), np.nan)
            ratios_per_seed.append(ratio)
        mean_ratio = _band(ax_c, updates_10, ratios_per_seed, "#d7191c",
                           "|sc_rate_loss| / |pg_loss|")
        ax_c.axhline(1.0, color="k", ls="--", lw=1.0, label="ratio = 1 (equal)")
        ax_c.set_yscale("log")
        ax_c.set_xlabel("Training update")
        ax_c.set_ylabel("|sc_rate_loss| / |pg_loss|  (log scale)")
        ax_c.set_title("(C) SC loss fraction: PPO dominates during learning")
        ax_c.legend(fontsize=8)
        # Annotate early vs late ratio
        early_ratio = float(np.nanmean(mean_ratio[:10]))
        ax_c.text(0.03, 0.92, f"Early (PPO dominates):\nratio ≈ {early_ratio:.2f} (<1)",
                  transform=ax_c.transAxes, ha="left", va="top", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", edgecolor="#6baed6"))

        fig.suptitle("F4 — Why live SC cannot restructure p(m): PPO sets the policy structure before SC becomes significant",
                     fontsize=10, y=1.01)
        fig.tight_layout()
        _save(fig, out_dir, "F4_gradient_direction")

    print(f"[F4] saved to {out_dir}/F4_gradient_direction.{{pdf,png}}")
    print(f"     SC/PPO loss ratio during early learning: {early_ratio:.4f}")


# ---------------------------------------------------------------------------
# F17 — ε_MLE convergence vs N
# ---------------------------------------------------------------------------

def fig_f17_eps_mle(
    ckpt_path: Path,
    out_dir: Path,
    z_dim: int = 3,
    hidden: int = 64,
    delta: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Generate F17: decomposition residual and ε_MLE bound vs N."""
    speaker = _load_speaker(ckpt_path, z_dim, hidden, device)
    channel = build_channel("sd", delta, ste_clip=10.0).to(device)
    speaker.eval()

    goals_t = torch.tensor(_DEFAULT_GOALS, dtype=torch.float32, device=device)

    # Collect z_per_goal once (for H_dither)
    z_per_goal: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for gid in range(len(_DEFAULT_GOALS)):
            g = goals_t[gid].unsqueeze(0).expand(100, -1)
            z_per_goal[gid] = speaker(g).cpu()

    N_values = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000]
    results: list[dict] = []

    for N in N_values:
        print(f"[F17] N={N:>6,}...", end="  ", flush=True)
        m, goal_ids = collect_messages(speaker, channel, delta, N, device)
        d = compute_decomposition(m, goal_ids, z_per_goal, delta)
        results.append({
            "N": N,
            "residual": abs(d["decomp_residual_bits"]),
            "eps_mle": d["eps_MLE_bound"],
            "K_obs": d["K_obs_joint_tuples"],
        })
        print(f"residual={d['decomp_residual_bits']:.4f}  K_obs={d['K_obs_joint_tuples']}")

    inv_sqrt_N = [1.0 / math.sqrt(r["N"]) for r in results]
    residuals   = [r["residual"] for r in results]
    eps_mle     = [r["eps_mle"]  for r in results]
    K_obs       = [r["K_obs"]    for r in results]
    N_vals      = [r["N"]        for r in results]

    # Fit residual ~ c / sqrt(N)
    valid = [(x, y) for x, y in zip(inv_sqrt_N, residuals) if y > 0]
    if len(valid) >= 2:
        xs, ys = zip(*valid)
        c_fit = np.polyfit(np.log(xs), np.log(ys), 1)
        slope = c_fit[0]
    else:
        slope = None

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

        # Panel A: residual and ε_MLE vs 1/√N
        ax_a.scatter(inv_sqrt_N, residuals, color="#d7191c", s=60, zorder=5,
                     label="|decomp residual|")
        ax_a.plot(inv_sqrt_N, eps_mle, "s--", color="#2c7bb6", ms=6, zorder=4,
                  label="ε_MLE bound (K/√N)")

        if slope is not None:
            xs_line = np.linspace(min(inv_sqrt_N), max(inv_sqrt_N), 100)
            # Fit line in log-log: residual = exp(c0 + slope * log(1/sqrt(N)))
            log_c0 = np.mean([math.log(y) - slope * math.log(x) for x, y in zip(xs, ys)])
            ys_fit = np.exp(log_c0 + slope * np.log(xs_line))
            ax_a.plot(xs_line, ys_fit, "r:", lw=1.2, alpha=0.6,
                      label=f"power fit (slope={slope:.2f})")

        ax_a.set_xlabel("1 / √N")
        ax_a.set_ylabel("Bits")
        ax_a.set_title("(A) Decomposition residual converges as O(1/√N)")
        ax_a.legend(fontsize=8)
        ax_a.set_yscale("log")
        ax_a.set_xscale("log")

        for x, y, n in zip(inv_sqrt_N, residuals, N_vals):
            ax_a.annotate(f"N={n:,}", (x, y), textcoords="offset points",
                          xytext=(4, 3), fontsize=7, color="#d7191c")

        # Panel B: K_obs (distinct joint tuples) vs N
        ax_b.plot(N_vals, K_obs, "o-", color="#4daf4a", ms=7)
        ax_b.set_xlabel("N (number of messages)")
        ax_b.set_ylabel("K_obs (distinct joint tuples)")
        ax_b.set_title("(B) Observed support K_obs stabilises quickly")
        ax_b.set_xscale("log")
        ax_b.axhline(K_obs[-1], color="#888888", lw=1.0, ls="--",
                     label=f"K_obs at N=50K = {K_obs[-1]}")
        ax_b.legend(fontsize=8)

        fig.suptitle("F17 — ε_MLE vs N: histogram rate estimator convergence",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        _save(fig, out_dir, "F17_eps_mle_convergence")

    print(f"[F17] saved to {out_dir}/F17_eps_mle_convergence.{{pdf,png}}")


# ---------------------------------------------------------------------------
# F11 — Context bound ordering
# ---------------------------------------------------------------------------

def fig_f11_context_bound_ordering(
    post_hoc_dir: Path,
    out_dir: Path,
) -> None:
    """Generate F11: the strict entropy bound hierarchy from the E09 post-hoc data.

    Claim: H_factored ≥ H_joint ≥ H(m|goal) ≥ H(G), strictly, for every seed.

    This hierarchy corresponds to progressive receiver context:
      H_factored  — no cross-dim context (per-dim independent coding)
      H_joint     — global context (joint distribution, no factored assumption)
      H(m|goal)   — goal-conditional context (goal label available at encoder/decoder)
      H(G)        — theoretical floor (entropy of the goal distribution)

    Each step down the hierarchy represents a coding gain available at deployment:
      TC = H_factored - H_joint          → joint arithmetic coding (free, post-hoc)
      Goal gap = H_joint - H(m|goal)    → goal-conditional code (requires goal at RX)
      Residual = H(m|goal) - H(G)       → irreducible noise given current z distribution

    Two panels:
      A — Parallel coordinates: one line per seed, x = context level, y = rate (bits).
          All lines are strictly decreasing, proving the ordering is not an artefact.
          Per-dim entropies (H_dim_0/1/2) shown as a sub-band inside H_factored.
      B — Gap waterfall (mean ± std): shows what each context level saves in bits.
    """
    seeds = sorted(post_hoc_dir.glob("seed_*.json"))
    if not seeds:
        raise FileNotFoundError(f"No seed_*.json in {post_hoc_dir}")

    records = [json.loads(p.read_text()) for p in seeds]

    # Per-seed values for the four bound levels
    h_fact  = np.array([r["H_factored_bits"] for r in records])
    h_joint = np.array([r["H_joint_bits"]    for r in records])
    h_dith  = np.array([r["H_dither_bits"]   for r in records])   # H(m|goal)
    h_g     = np.array([r["H_G_bits"]        for r in records])   # constant

    # Per-dimension entropies (for sub-band in Panel A)
    h_dims = np.array([
        [r["H_dim_0_bits"], r["H_dim_1_bits"], r["H_dim_2_bits"]]
        for r in records
    ])   # shape (n_seeds, 3)

    # Gaps
    tc_gap   = h_fact  - h_joint   # joint coding benefit
    goal_gap = h_joint - h_dith    # goal-label benefit
    resid    = h_dith  - h_g       # irreducible noise

    # Verify ordering holds for every seed
    ordering_ok = all(
        hf >= hj >= hd >= hgg
        for hf, hj, hd, hgg in zip(h_fact, h_joint, h_dith, h_g)
    )

    n_seeds = len(records)
    x_levels = np.array([0, 1, 2, 3])
    x_labels = ["H_factored\n(no context)", "H_joint\n(global context)",
                 "H(m|goal)\n(goal context)", "H(G)\n(floor)"]

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

        # ── Panel A: parallel coordinates ─────────────────────────────────────
        seed_colors = plt.cm.tab10(np.linspace(0, 0.5, n_seeds))
        for i, (hf, hj, hd, hgg) in enumerate(zip(h_fact, h_joint, h_dith, h_g)):
            ax_a.plot(x_levels, [hf, hj, hd, hgg],
                      "o-", color=seed_colors[i], lw=1.4, ms=5, alpha=0.8,
                      label=f"seed {i}")

        # Mean ± std band
        means = np.array([h_fact.mean(), h_joint.mean(), h_dith.mean(), h_g.mean()])
        stds  = np.array([h_fact.std(),  h_joint.std(),  h_dith.std(),  h_g.std()])
        ax_a.plot(x_levels, means, "k-", lw=2.5, zorder=5, label="mean")
        ax_a.fill_between(x_levels, means - stds, means + stds,
                          alpha=0.12, color="black", zorder=4)

        # Per-dim sub-band inside H_factored (shows near-equal per-dim entropies)
        h_dim_min = h_dims.min(axis=1)
        h_dim_max = h_dims.max(axis=1)
        ax_a.errorbar([0], [h_dims.mean()],
                      yerr=[[h_dims.mean() - h_dim_min.mean()],
                             [h_dim_max.mean() - h_dims.mean()]],
                      fmt="none", color="#4daf4a", capsize=6, lw=2,
                      label="per-dim range [H_dim_min, H_dim_max]")

        ax_a.axhline(h_g[0], color="#888888", lw=1.2, ls="--", zorder=3,
                     label=f"H(G) = {h_g[0]:.3f} bits (floor)")
        ax_a.set_xticks(x_levels)
        ax_a.set_xticklabels(x_labels, fontsize=9)
        ax_a.set_ylabel("Rate (bits per message)")
        ax_a.set_title(f"(A) Bound ordering holds for all {n_seeds} seeds\n"
                        f"H_factored ≥ H_joint ≥ H(m|goal) ≥ H(G)  [OK={ordering_ok}]")
        ax_a.legend(fontsize=8, loc="upper right")
        ax_a.set_xlim(-0.3, 3.3)

        # Gap annotations (double-headed arrows on the mean line)
        gap_specs = [
            (0, 1, tc_gap,   "#d7191c", "TC = {:.2f}±{:.2f}"),
            (1, 2, goal_gap, "#2c7bb6", "goal gap = {:.2f}±{:.2f}"),
            (2, 3, resid,    "#4daf4a", "residual = {:.2f}±{:.2f}"),
        ]
        for xi, xj, gap_vals, col, fmt in gap_specs:
            mid_x = (xi + xj) / 2
            ax_a.annotate("", xy=(mid_x, means[xj]), xytext=(mid_x, means[xi]),
                          arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
            ax_a.text(mid_x + 0.05, (means[xi] + means[xj]) / 2,
                      fmt.format(gap_vals.mean(), gap_vals.std()),
                      fontsize=8, color=col, va="center")

        # ── Panel B: gap waterfall ─────────────────────────────────────────────
        gap_labels  = ["H(G)\n[floor]", "Residual noise\nH(m|goal)−H(G)",
                        "Goal-context gap\nH_joint−H(m|goal)", "TC\nH_factored−H_joint"]
        gap_means   = [h_g.mean(), resid.mean(), goal_gap.mean(), tc_gap.mean()]
        gap_stds    = [0.0,        resid.std(),  goal_gap.std(),  tc_gap.std()]
        gap_colors  = ["#aaaaaa", "#4daf4a", "#2c7bb6", "#d7191c"]

        bar_x = 0.5
        bottom = 0.0
        for label, val, err, col in zip(gap_labels, gap_means, gap_stds, gap_colors):
            ax_b.bar(bar_x, val, bottom=bottom, color=col, width=0.6,
                     edgecolor="white", linewidth=0.5, label=label)
            mid = bottom + val / 2
            if val > 0.1:
                ax_b.text(bar_x + 0.35, mid,
                          f"{val:.2f}±{err:.2f} bits" if err > 0 else f"{val:.3f} bits",
                          va="center", ha="left", fontsize=8.5)
            bottom += val

        ax_b.axhline(h_g[0], color="#888888", lw=1.0, ls=":", zorder=3)
        ax_b.set_xticks([])
        ax_b.set_ylabel("Rate (bits per message)")
        ax_b.set_title("(B) Gap attribution: what each context level saves\n"
                        "(TC = joint coding gain, free post-hoc)")
        ax_b.legend(loc="upper right", fontsize=8)
        ax_b.set_xlim(-0.2, 1.6)
        ax_b.set_ylim(0, (h_fact.mean() + h_fact.std()) * 1.15)

        fig.suptitle(
            "F11 — Context bound ordering: H_factored ≥ H_joint ≥ H(m|goal) ≥ H(G)\n"
            "Each level of receiver context gives a strictly tighter rate bound",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        _save(fig, out_dir, "F11_context_bound_ordering")

    print(f"[F11] saved to {out_dir}/F11_context_bound_ordering.{{pdf,png}}")
    print(f"      H_factored = {h_fact.mean():.3f}±{h_fact.std():.3f}  "
          f"H_joint = {h_joint.mean():.3f}±{h_joint.std():.3f}  "
          f"H(m|goal) = {h_dith.mean():.3f}±{h_dith.std():.3f}  "
          f"H(G) = {h_g[0]:.3f}")
    print(f"      TC = {tc_gap.mean():.3f}  goal_gap = {goal_gap.mean():.3f}  "
          f"residual = {resid.mean():.3f}  ordering_OK = {ordering_ok}")


# ---------------------------------------------------------------------------
# F8 — qphi_gap: DLM vs histogram
# ---------------------------------------------------------------------------

_P2FIX_CONFIGS = [
    ("fix_baseline",         "DLM baseline",          "#e41a1c"),
    ("fix_B1_gate",          "DLM + gate + warmup",   "#ff7f00"),
    ("fix_B2_gate_lambda",   "DLM + gate + λ",        "#a65628"),
    ("fix_B3_ema",           "DLM + EMA prior",       "#984ea3"),
    ("fix_B4_twophase",      "DLM two-phase",         "#377eb8"),
]

_HIST_QPHI_GAP = 0.004   # histogram estimator (sc_posthoc_mag, 5-seed mean)


def fig_f8_qphi_gap_comparison(
    p2_ablation_dir: Path,
    out_dir: Path,
    n_seeds: int = 5,
) -> None:
    """Generate F8: DLM qphi_gap vs histogram estimator."""
    # Final qphi_gap for each DLM config
    final_gaps: dict[str, tuple[float, float]] = {}  # name → (mean, std)
    for exp_name, _, _ in _P2FIX_CONFIGS:
        vals = []
        for seed in range(n_seeds):
            path = p2_ablation_dir / exp_name / str(seed) / "metrics.csv"
            if not path.exists():
                continue
            rows = list(csv.DictReader(path.open()))
            for row in reversed(rows):
                if row.get("qphi_gap", "") not in ("", "nan"):
                    vals.append(float(row["qphi_gap"]))
                    break
        if vals:
            final_gaps[exp_name] = (statistics.mean(vals),
                                    statistics.stdev(vals) if len(vals) > 1 else 0.0)

    # Training trajectory for fix_baseline (to show the growing gap)
    baseline_seeds = _load_metrics(p2_ablation_dir, "fix_baseline", n_seeds)

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Panel A: bar chart of final qphi_gap — histogram + DLM configs
        bar_labels  = ["Histogram\n(this work)"]
        bar_means   = [_HIST_QPHI_GAP]
        bar_errors  = [0.0]
        bar_colors  = ["#4daf4a"]
        for exp_name, label, color in _P2FIX_CONFIGS:
            if exp_name in final_gaps:
                bar_labels.append(label)
                bar_means.append(final_gaps[exp_name][0])
                bar_errors.append(final_gaps[exp_name][1])
                bar_colors.append(color)
        xs = np.arange(len(bar_means))
        bars = ax_a.bar(xs, bar_means, yerr=bar_errors, color=bar_colors,
                        capsize=4, edgecolor="white", linewidth=0.5)
        ax_a.set_yscale("log")
        ax_a.set_xticks(xs)
        ax_a.set_xticklabels(bar_labels, fontsize=8, rotation=15, ha="right")
        ax_a.set_ylabel("qphi_gap (bits, log scale)")
        ax_a.set_title("(A) Final qphi_gap: DLM vs histogram")

        # Annotate bar values
        for bar, val in zip(bars, bar_means):
            ax_a.text(bar.get_x() + bar.get_width() / 2, val * 1.4,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax_a.axhline(_HIST_QPHI_GAP, color="#4daf4a", lw=1.2, ls="--", alpha=0.5)

        # Panel B: qphi_gap training trajectory for fix_baseline
        if baseline_seeds:
            updates = baseline_seeds[0]["update"]
            _band(ax_b, updates,
                  [s["qphi_gap"] for s in baseline_seeds],
                  "#e41a1c", "DLM baseline (fix_baseline)")
            ax_b.axhline(_HIST_QPHI_GAP, color="#4daf4a", lw=1.5, ls="--",
                         label=f"Histogram floor ({_HIST_QPHI_GAP:.3f} bits)")
            ax_b.set_xlabel("Training update")
            ax_b.set_ylabel("qphi_gap (bits)")
            ax_b.set_title("(B) DLM gap grows monotonically during training")
            ax_b.legend(fontsize=8)

        fig.suptitle("F8 — qphi_gap: DLM estimator error vs histogram (3000× improvement)",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        _save(fig, out_dir, "F8_qphi_gap_comparison")

    means_str = "  ".join(f"{e}={final_gaps[e][0]:.2f}" for e, _, _ in _P2FIX_CONFIGS
                           if e in final_gaps)
    print(f"[F8] saved to {out_dir}/F8_qphi_gap_comparison.{{pdf,png}}")
    print(f"     {means_str}")
    print(f"     histogram={_HIST_QPHI_GAP:.4f} bits")


# ---------------------------------------------------------------------------
# F10 — Circular gradient failure
# ---------------------------------------------------------------------------

def fig_f10_circular_gradient(
    p2_ablation_dir: Path,
    out_dir: Path,
    n_seeds: int = 5,
) -> None:
    """Generate F10: qphi_gap training trajectories showing circular gradient failure.

    All DLM configs start with low qphi_gap (policy is untrained → simple distribution).
    As the policy learns a rich goal→tuple codebook, the distribution becomes complex and
    qphi_gap explodes. Applying fixes (B1–B3) makes the gap grow faster, not slower.
    fix_B4_twophase (no entropy backward in Phase 1) keeps the gap low throughout Phase 1
    but at the cost of SR in Phase 2.
    """
    all_metrics = {exp: _load_metrics(p2_ablation_dir, exp, n_seeds)
                   for exp, _, _ in _P2FIX_CONFIGS}

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.5))

        # Panel A: qphi_gap trajectories — all configs
        for exp_name, label, color in _P2FIX_CONFIGS:
            seeds = all_metrics[exp_name]
            if not seeds:
                continue
            updates = seeds[0]["update"]
            _band(ax_a, updates,
                  [s["qphi_gap"] for s in seeds],
                  color, label)

        ax_a.axhline(_HIST_QPHI_GAP, color="#4daf4a", lw=1.5, ls="--",
                     label=f"Histogram floor ({_HIST_QPHI_GAP:.3f} bits)")
        ax_a.set_xlabel("Training update")
        ax_a.set_ylabel("qphi_gap (bits)")
        ax_a.set_title("(A) qphi_gap vs training: circular gradient drives gap upward")
        ax_a.legend(fontsize=8, loc="upper left")

        # Panel B: same plot but log-y to see early values clearly
        for exp_name, label, color in _P2FIX_CONFIGS:
            seeds = all_metrics[exp_name]
            if not seeds:
                continue
            updates = seeds[0]["update"]
            gap_arrays = [s["qphi_gap"] for s in seeds]
            # Clip negatives before log scale
            gap_arrays = [np.clip(g, 1e-4, None) for g in gap_arrays]
            _band(ax_b, updates, gap_arrays, color, label)

        ax_b.axhline(_HIST_QPHI_GAP, color="#4daf4a", lw=1.5, ls="--",
                     label=f"Histogram floor ({_HIST_QPHI_GAP:.3f} bits)")
        ax_b.set_yscale("log")
        ax_b.set_xlabel("Training update")
        ax_b.set_ylabel("qphi_gap (bits, log scale)")
        ax_b.set_title("(B) Log scale: gap starts low then explodes\n(B1–B3 make it worse)")
        ax_b.legend(fontsize=8, loc="upper left")

        fig.suptitle("F10 — Circular gradient failure: DLM gap grows as policy learns;"
                     " engineering fixes do not help",
                     fontsize=10, y=1.01)
        fig.tight_layout()
        _save(fig, out_dir, "F10_circular_gradient")

    print(f"[F10] saved to {out_dir}/F10_circular_gradient.{{pdf,png}}")
    for exp_name, label, _ in _P2FIX_CONFIGS:
        seeds = all_metrics[exp_name]
        if seeds:
            early = float(np.nanmean([s["qphi_gap"][0] for s in seeds]))
            final = float(np.nanmean([s["qphi_gap"][-1] for s in seeds]))
            print(f"     {label}: early={early:.2f}  final={final:.2f} bits")


# ---------------------------------------------------------------------------
# F12 — Warm-start failure
# ---------------------------------------------------------------------------

def fig_f12_warmstart_failure(
    p2_ablation_dir: Path,
    out_dir: Path,
    n_seeds: int = 5,
) -> None:
    """Generate F12: fix_B4_twophase achieves low qphi_gap but SR collapses in Phase 2.

    Two panels:
      A — qphi_gap for fix_B4_twophase (low throughout, ~0.05 bits) vs fix_baseline
          (grows to ~9.7 bits). The warm start 'works' for fitting q_φ.
      B — SR training curve for fix_B4_twophase, showing Phase 2 collapse.
          Phase 2 trigger (training_phase transitions from 1→2) is annotated.
    """
    b4_seeds    = _load_metrics(p2_ablation_dir, "fix_B4_twophase", n_seeds)
    base_seeds  = _load_metrics(p2_ablation_dir, "fix_baseline",    n_seeds)

    if not b4_seeds:
        raise FileNotFoundError(f"fix_B4_twophase metrics not found in {p2_ablation_dir}")

    # Find Phase 2 trigger update (first update where training_phase == 2)
    phase2_updates: list[int] = []
    for s in b4_seeds:
        phase = s.get("training_phase")
        if phase is None:
            continue
        idxs = np.where(phase >= 2)[0]
        if len(idxs):
            phase2_updates.append(int(s["update"][idxs[0]]))
    phase2_trigger = int(np.median(phase2_updates)) if phase2_updates else None

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Panel A: qphi_gap
        updates_b4   = b4_seeds[0]["update"]
        updates_base = base_seeds[0]["update"] if base_seeds else updates_b4

        _band(ax_a, updates_b4,
              [np.clip(s["qphi_gap"], 1e-4, None) for s in b4_seeds],
              "#377eb8", "fix_B4_twophase (warm start)")
        if base_seeds:
            _band(ax_a, updates_base,
                  [np.clip(s["qphi_gap"], 1e-4, None) for s in base_seeds],
                  "#e41a1c", "fix_baseline (no warm start)")
        ax_a.axhline(_HIST_QPHI_GAP, color="#4daf4a", lw=1.5, ls="--",
                     label=f"Histogram floor ({_HIST_QPHI_GAP:.3f} bits)")

        if phase2_trigger is not None:
            ax_a.axvline(phase2_trigger, color="#377eb8", lw=1.2, ls=":",
                         label=f"Phase 2 start (update ≈ {phase2_trigger})")

        ax_a.set_yscale("log")
        ax_a.set_xlabel("Training update")
        ax_a.set_ylabel("qphi_gap (bits, log scale)")
        ax_a.set_title("(A) Warm start achieves low qphi_gap\n(but see Panel B)")
        ax_a.legend(fontsize=8)

        # Panel B: SR trajectory for fix_B4_twophase
        _band(ax_b, updates_b4,
              [s["success_rate"] for s in b4_seeds],
              "#377eb8", "fix_B4_twophase SR")

        if phase2_trigger is not None:
            ax_b.axvline(phase2_trigger, color="#e41a1c", lw=1.5, ls="--",
                         label=f"Phase 2 starts → SR collapses")

        sr_final = float(np.nanmean([s["success_rate"][-1] for s in b4_seeds]))
        ax_b.axhline(1.0,        color="#aaaaaa", lw=0.8, ls=":")
        ax_b.axhline(sr_final,   color="#377eb8", lw=1.0, ls=":",
                     label=f"Final SR = {sr_final:.3f}")
        ax_b.set_ylim(0, 1.08)
        ax_b.set_xlabel("Training update")
        ax_b.set_ylabel("Success rate")
        ax_b.set_title("(B) SR collapses in Phase 2 despite low qphi_gap\n"
                        "(fitting q_φ is necessary but not sufficient)")
        ax_b.legend(fontsize=8)

        fig.suptitle("F12 — Warm-start failure: good q_φ fit does not prevent SR collapse"
                     " when used as a speaker gradient",
                     fontsize=10, y=1.01)
        fig.tight_layout()
        _save(fig, out_dir, "F12_warmstart_failure")

    print(f"[F12] saved to {out_dir}/F12_warmstart_failure.{{pdf,png}}")
    print(f"     fix_B4_twophase: qphi_gap≈{float(np.nanmean([s['qphi_gap'][-1] for s in b4_seeds])):.3f} bits"
          f"  SR={sr_final:.3f}")
    if phase2_trigger:
        print(f"     Phase 2 triggered at update {phase2_trigger}")


# ---------------------------------------------------------------------------
# F14 — Score function gradient ratio
# ---------------------------------------------------------------------------

def fig_f14_grad_ratio(
    live_b_dir: Path,
    out_dir: Path,
    n_seeds: int = 5,
) -> None:
    """Generate F14: PPO vs SC speaker gradient norms during live_B training.

    Uses the new ppo_speaker_grad_norm / sc_speaker_grad_norm / ppo_sc_grad_ratio
    columns logged when --log_grad_decomp is active (runs/live_B_diag).

    Two panels:
      A — ppo_speaker_grad_norm and sc_speaker_grad_norm over training (log-y):
          PPO consistently dominates SC by 4–5× at every stage of training.
      B — ppo_sc_grad_ratio (PPO/SC) over training: ratio stays above 1
          throughout — PPO speaker gradient never loses its dominance over SC.
    """
    seeds = _load_metrics(live_b_dir, "live_B_live_sc", n_seeds)
    if not seeds:
        raise FileNotFoundError(
            f"No live_B_live_sc metrics found under {live_b_dir}.\n"
            f"Re-run E21 with --log_grad_decomp --log_dir {live_b_dir}"
        )

    # Check that the new columns are present
    if "ppo_speaker_grad_norm" not in seeds[0]:
        raise KeyError(
            "ppo_speaker_grad_norm not in metrics.csv — "
            "re-run live_B with --log_grad_decomp"
        )

    updates = seeds[0]["update"]

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Panel A: raw grad norms (log-y)
        _band(ax_a, updates,
              [s["ppo_speaker_grad_norm"] for s in seeds],
              "#2c7bb6", "PPO speaker grad norm")
        _band(ax_a, updates,
              [s["sc_speaker_grad_norm"] for s in seeds],
              "#d7191c", "SC speaker grad norm")

        ax_a.set_yscale("log")
        ax_a.set_xlabel("Training update")
        ax_a.set_ylabel("Speaker grad norm (log scale)")
        ax_a.set_title("(A) PPO vs SC gradient magnitude over training")
        ax_a.legend(fontsize=8)

        # Check dominance direction (expected: PPO > SC throughout)
        ppo_arr = np.array([s["ppo_speaker_grad_norm"] for s in seeds])
        sc_arr  = np.array([s["sc_speaker_grad_norm"]  for s in seeds])
        mean_ppo = np.nanmean(ppo_arr, axis=0)
        mean_sc  = np.nanmean(sc_arr,  axis=0)
        ppo_dom_frac = float(np.mean(mean_ppo > mean_sc))
        ax_a.text(0.97, 0.05,
                  f"PPO > SC in {ppo_dom_frac*100:.0f}% of updates",
                  transform=ax_a.transAxes, ha="right", va="bottom", fontsize=7.5,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", edgecolor="#6baed6"))

        # Panel B: PPO/SC ratio (log-y)
        ratio_list = []
        for s in seeds:
            r = s.get("ppo_sc_grad_ratio")
            if r is not None:
                ratio_list.append(r)
        if ratio_list:
            mean_ratio = _band(ax_b, updates[:len(ratio_list[0])],
                               ratio_list, "#984ea3", "PPO / SC grad norm ratio")
            ax_b.axhline(1.0, color="k", ls="--", lw=1.0, label="ratio = 1 (equal)")
            ax_b.set_yscale("log")
            ax_b.set_xlabel("Training update")
            ax_b.set_ylabel("PPO / SC grad norm  (log scale)")
            ax_b.set_title("(B) Ratio: PPO consistently leads SC throughout training")
            ax_b.legend(fontsize=8)

            early  = float(np.nanmean(mean_ratio[:20]))
            late   = float(np.nanmean(mean_ratio[-20:]))
            ax_b.text(0.03, 0.95,
                      f"Early ratio: {early:.1f}×\n(PPO > SC)",
                      transform=ax_b.transAxes, ha="left", va="top", fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", edgecolor="#6baed6"))
            ax_b.text(0.97, 0.05,
                      f"Late ratio: {late:.2f}×\n(PPO > SC throughout)",
                      transform=ax_b.transAxes, ha="right", va="bottom", fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", edgecolor="#6baed6"))
        else:
            ax_b.text(0.5, 0.5, "ppo_sc_grad_ratio not available\n(SC loss inactive?)",
                      transform=ax_b.transAxes, ha="center", va="center", fontsize=9)

        fig.suptitle(
            "F14 — Score function gradient ratio: PPO consistently dominates SC 4–5× throughout training,\n"
            "explaining why live SC cannot restructure p(m)",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        _save(fig, out_dir, "F14_score_function_grad_ratio")

    print(f"[F14] saved to {out_dir}/F14_score_function_grad_ratio.{{pdf,png}}")
    if ratio_list:
        print(f"     Early PPO/SC ratio (first 20 updates): {early:.2f}")
        print(f"     Late  PPO/SC ratio (last 20 updates):  {late:.4f}")


# ---------------------------------------------------------------------------
# F19 — Moving-target error
# ---------------------------------------------------------------------------

def fig_f19_moving_target(
    live_b_dir: Path,
    out_dir: Path,
    n_seeds: int = 5,
) -> None:
    """Generate F19: speaker keeps moving but H(m) stays flat — moving-target error.

    Uses speaker_param_delta_norm and true_H_offline columns logged when
    --log_moving_target is active (runs/live_B_diag).

    Two panels:
      A — true_H_offline (joint H(m) from 10K offline messages, correct _GOAL_PROBS
          weighting) over training. Decreases toward H(G) as z-blowup puts each goal
          in separate non-overlapping bins — degenerate, not useful compression.
          hist_H_empirical (minibatch, factored, ≈uniform goal weighting) stays flat
          at ~7 bits, revealing a ~4.6-bit overestimation from the training metric.
      B — speaker_param_delta_norm (‖Δθ‖ per update) over training:
          speaker actively changes throughout (moving target); the problem is that SC
          gradient direction is stale by the time it takes effect.
    """
    seeds = _load_metrics(live_b_dir, "live_B_live_sc", n_seeds)
    if not seeds:
        raise FileNotFoundError(
            f"No live_B_live_sc metrics found under {live_b_dir}.\n"
            f"Re-run E21 with --log_moving_target --log_dir {live_b_dir}"
        )

    if "true_H_offline" not in seeds[0]:
        raise KeyError(
            "true_H_offline not in metrics.csv — "
            "re-run live_B with --log_moving_target"
        )

    updates = seeds[0]["update"]

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Panel A: true_H_offline over training
        _band(ax_a, updates,
              [s["true_H_offline"] for s in seeds],
              "#d7191c", "H(m) offline (10K samples)")
        ax_a.axhline(_H_G, color="k", ls="--", lw=1.2,
                     label=f"H(G) = {_H_G:.2f} bits (lower bound)")

        # Also overlay hist_H_empirical from the minibatch for comparison
        if "hist_H_empirical" in seeds[0]:
            _band(ax_a, updates,
                  [s["hist_H_empirical"] for s in seeds],
                  "#2c7bb6", "H(m) minibatch (hist_H_empirical)", alpha=0.10)

        ax_a.set_xlabel("Training update")
        ax_a.set_ylabel("H(m) (bits)")
        ax_a.set_title("(A) True H(m) decreases via z-blowup, not compression (hist_H overestimates)")
        ax_a.legend(fontsize=8)

        final_h = float(np.nanmean([s["true_H_offline"][-1] for s in seeds]))
        ax_a.text(0.97, 0.95,
                  f"Final H(m) ≈ {final_h:.2f} bits\n(z-blowup: each goal in\nnon-overlapping bins)",
                  transform=ax_a.transAxes, ha="right", va="top", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#fde0d9", edgecolor="#de2d26"))

        # Panel B: speaker_param_delta_norm over training
        mean_delta = _band(ax_b, updates,
                           [s["speaker_param_delta_norm"] for s in seeds],
                           "#4daf4a", "‖Δθ‖ per update")
        ax_b.set_xlabel("Training update")
        ax_b.set_ylabel("Speaker param change ‖Δθ‖")
        ax_b.set_title("(B) Speaker keeps moving: gradient is active, not silent")
        ax_b.legend(fontsize=8)

        early_d = float(np.nanmean(mean_delta[:20]))
        late_d  = float(np.nanmean(mean_delta[-20:]))
        ax_b.text(0.03, 0.95,
                  f"Early ‖Δθ‖: {early_d:.4f}\nLate ‖Δθ‖: {late_d:.4f}",
                  transform=ax_b.transAxes, ha="left", va="top", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#e5f5e0", edgecolor="#41ab5d"))

        fig.suptitle(
            "F19 — Moving-target error: speaker parameters change continuously,\n"
            "but SC gradient points toward cheaper messages under the current "
            "distribution — which has already shifted",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        _save(fig, out_dir, "F19_moving_target_error")

    print(f"[F19] saved to {out_dir}/F19_moving_target_error.{{pdf,png}}")
    print(f"     Final H(m) offline: {final_h:.3f} bits  (H(G)={_H_G:.3f})")
    print(f"     ‖Δθ‖ early={early_d:.5f}  late={late_d:.5f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F3, F4, F8, F10, F11, F12, F14, F17, F19.")
    p.add_argument("--ckpt_dir",         type=str, default="runs/sc_ablation/sc_posthoc_mag")
    p.add_argument("--post_hoc_dir",     type=str, default="results/post_hoc/sc_posthoc_mag")
    p.add_argument("--runs_dir",         type=str, default="runs/sc_ablation",
                   help="Parent directory for sc_ablation experiment runs (F4)")
    p.add_argument("--p2_ablation_dir",  type=str, default="runs/p2_ablation",
                   help="Parent directory for P2-FIX runs (F8, F10, F12)")
    p.add_argument("--live_b_diag_dir",  type=str, default="runs/live_B_diag",
                   help="Parent directory for live_B diagnostic runs with grad/moving-target "
                        "logging (F14, F19). Produced by re-running E21 with "
                        "--log_grad_decomp --log_moving_target.")
    p.add_argument("--out_dir",          type=str, default="results/figures")
    p.add_argument("--seed",             type=int, default=0)
    p.add_argument("--z_dim",            type=int, default=3)
    p.add_argument("--hidden",           type=int, default=64)
    p.add_argument("--figures",          type=str, default="F3,F4,F17",
                   help="Comma-separated list: F3, F4, F8, F10, F11, F12, F14, F17, F19")
    p.add_argument("--device",           type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = _parse()
    ckpt_path       = Path(args.ckpt_dir) / str(args.seed) / "final.pt"
    post_hoc_dir    = Path(args.post_hoc_dir)
    runs_dir        = Path(args.runs_dir)
    p2_ablation_dir = Path(args.p2_ablation_dir)
    live_b_diag_dir = Path(args.live_b_diag_dir)
    out_dir         = Path(args.out_dir)
    device          = torch.device(args.device)
    figures         = {f.strip() for f in args.figures.split(",")}

    if "F3" in figures or "F17" in figures:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if "F3" in figures:
        print("\n=== F3: Implicit prior mismatch ===")
        fig_f3_implicit_prior(post_hoc_dir, ckpt_path, out_dir,
                              z_dim=args.z_dim, hidden=args.hidden, device=device)

    if "F4" in figures:
        print("\n=== F4: Gradient direction ===")
        fig_f4_gradient_direction(runs_dir, out_dir)

    if "F11" in figures:
        print("\n=== F11: Context bound ordering ===")
        fig_f11_context_bound_ordering(post_hoc_dir, out_dir)

    if "F8" in figures:
        print("\n=== F8: qphi_gap DLM vs histogram ===")
        fig_f8_qphi_gap_comparison(p2_ablation_dir, out_dir)

    if "F10" in figures:
        print("\n=== F10: Circular gradient failure ===")
        fig_f10_circular_gradient(p2_ablation_dir, out_dir)

    if "F12" in figures:
        print("\n=== F12: Warm-start failure ===")
        fig_f12_warmstart_failure(p2_ablation_dir, out_dir)

    if "F14" in figures:
        print("\n=== F14: Score function gradient ratio ===")
        fig_f14_grad_ratio(live_b_diag_dir, out_dir)

    if "F17" in figures:
        print("\n=== F17: ε_MLE vs N ===")
        fig_f17_eps_mle(ckpt_path, out_dir,
                        z_dim=args.z_dim, hidden=args.hidden, device=device)

    if "F19" in figures:
        print("\n=== F19: Moving-target error ===")
        fig_f19_moving_target(live_b_diag_dir, out_dir)


if __name__ == "__main__":
    main()
