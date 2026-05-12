"""post_hoc_coding.py — Post-hoc rate decomposition from a trained checkpoint.

Computes the full rate decomposition for a converged policy:

    R(m) = H(G) + H(m|goal) + TC(m) + ε_estimator + ε_MLE

where:
    H(G)         = 1.8094 bits  (Shannon entropy of 6-goal uniform distribution)
    H(m|goal)    = E_goal[Σ_k H_binary(|frac(z_k(goal)/δ) − 0.5|)]  — dither channel noise
                   (floor SD channel: minimum noise at frac=0.5, maximum at frac=0)
    TC(m)        = Σ_k H(m_k) - H_joint(m)                  — cross-dim correlation
    ε_estimator  = hist_qphi_gap ≈ 0.004 bits                — histogram fit quality
    ε_MLE        = O(K^D / √N)                               — finite-sample error bound

Usage (from repo root):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.analysis.post_hoc_coding \\
        --checkpoint runs/sc_option1/0/final.pt \\
        --n_rollout_msgs 50000 \\
        --delta 1.0 --z_dim 3

The script:
  1. Loads the checkpoint and collects a held-out rollout.
  2. Builds both a factored (per-dim) and a joint (D-dim) histogram.
  3. Computes H_joint, TC, H(m|goal), ε_MLE bound, and the decomposition residual.
  4. Prints a table and saves results to <checkpoint_dir>/post_hoc_decomposition.json.

Debug checklist:
    - If TC > 3 bits: cross-dim correlation high; joint coding gives big gain.
    - If H(m|goal) > 1 bit/dim: fracs near 0.5; Phase 2 L_dither not yet run.
    - If ε_MLE_bound > 0.1 bits: increase --n_rollout_msgs (more samples needed).
    - If residual > 0.5 bits: decomposition identity failing; check delta match.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from onpolicy.envs.toyproblem.channels import build_channel, H_GOAL_BITS, _GOAL_PROBS
from onpolicy.envs.toyproblem.CommunicatingGoal_env import _DEFAULT_GOALS
from onpolicy.envs.toyproblem.CommunicatingGoal_vec_env import CommunicatingGoalVecEnv
from onpolicy.envs.toyproblem.network import SpeakerNetwork
from onpolicy.envs.toyproblem.source_coding import (
    JointMessageHistogram,
    MessageHistogram,
    dither_channel_stats,
)


# ---------------------------------------------------------------------------
# Rollout collector
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_messages(
    speaker: SpeakerNetwork,
    channel,
    delta: float,
    n_msgs: int,
    device: torch.device,
    n_envs: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (m, goal_id) pairs from the trained speaker.

    Returns:
        m        : (N, z_dim) integer message tensor
        goal_ids : (N,) integer goal-index tensor
    """
    goals_t = torch.tensor(_DEFAULT_GOALS, dtype=torch.float32, device=device)
    n_goals = len(_DEFAULT_GOALS)

    all_m: list[torch.Tensor] = []
    all_gid: list[torch.Tensor] = []

    # Sample goals according to the actual training distribution (_GOAL_PROBS) so
    # the collected histogram reflects the true p(m) that the speaker produces.
    # Uniform cycling would give the wrong H(G) baseline (log2(6) ≈ 2.585 bits
    # instead of H_GOAL_BITS ≈ 1.81 bits), inflating the decomposition residual.
    goal_probs_t = torch.tensor(_GOAL_PROBS, dtype=torch.float32, device=device)

    collected = 0
    while collected < n_msgs:
        batch = min(n_envs, n_msgs - collected)
        gids = torch.multinomial(goal_probs_t, batch, replacement=True)
        goal = goals_t[gids]
        z = speaker(goal)
        # Quantise through channel; forward returns (z_hat, info_dict)
        _, info = channel(z)
        m = info["m"].long()  # (batch, z_dim) integer messages
        all_m.append(m.cpu())
        all_gid.append(gids.cpu())
        collected += batch

    return torch.cat(all_m, dim=0), torch.cat(all_gid, dim=0)


# ---------------------------------------------------------------------------
# Rate decomposition
# ---------------------------------------------------------------------------

def compute_decomposition(
    m: torch.Tensor,
    goal_ids: torch.Tensor,
    z_per_goal: dict[int, torch.Tensor],
    delta: float,
    smoothing: float = 0.5,
) -> dict[str, float]:
    """Compute the full rate decomposition from collected messages.

    Args:
        m         : (N, z_dim) integer messages
        goal_ids  : (N,) goal indices
        z_per_goal: dict mapping goal_idx → (K, z_dim) z-values from speaker
        delta     : quantisation bin width δ

    Returns dict with all decomposition components and diagnostics.
    """
    N, z_dim = m.shape

    # ── Joint histogram ──────────────────────────────────────────────────────
    joint_hist = JointMessageHistogram(z_dim=z_dim, smoothing=smoothing)
    joint_hist.update(m)
    H_joint = joint_hist.joint_entropy()
    H_marginals = joint_hist.marginal_entropies()
    H_factored = sum(H_marginals)
    TC = joint_hist.total_correlation()
    K_obs = joint_hist.n_distinct_tuples()

    # ── Factored histogram (per-dim) ─────────────────────────────────────────
    fact_hist = MessageHistogram(z_dim=z_dim, smoothing=smoothing)
    fact_hist.update(m)
    # Cross-check: factored entropy should equal H_marginals sum
    fact_H = sum(fact_hist.empirical_entropy())

    # ── H(m|goal): dither channel noise ─────────────────────────────────────
    # H(m|goal) = Σ_g P(g) · H(m|z_g) — probability-weighted over goals.
    # Simple averaging would give the wrong value when goals are non-uniform
    # (e.g. goal 0 has P=0.515 vs goal 5 has P=0.003).
    H_dither_per_goal: list[float] = []
    for gid, z_g in z_per_goal.items():
        stats = dither_channel_stats(z_g, delta)
        H_dither_per_goal.append(stats["H_dither_channel"])
    H_dither = float(sum(
        _GOAL_PROBS[gid] * h
        for gid, h in enumerate(H_dither_per_goal)
    ))

    # ── ε_estimator: histogram fit quality ───────────────────────────────────
    # For the factored histogram this is hist_qphi_gap ≈ 0.004 bits.
    # Compute directly: E[-log₂ q_hist(m)] - H(m)
    from onpolicy.envs.toyproblem.source_coding import histogram_rate_stats
    sc_stats = histogram_rate_stats(m, fact_hist)
    eps_estimator = sc_stats["hist_qphi_gap"]

    # ── ε_MLE: finite-sample bound ───────────────────────────────────────────
    # ε_MLE = O(K^D / √N).  Use K_obs as observed support size.
    # Constant = 1.0 gives conservative bound; real constant depends on estimator.
    eps_mle_bound = float(K_obs) / math.sqrt(N) if N > 0 else float("inf")
    # Convert to bits (rough): multiply by 1/ln(2) for nats→bits if needed.
    # Here K_obs is the number of distinct joint tuples observed.

    # ── Shannon gap from H(G) ────────────────────────────────────────────────
    H_G = H_GOAL_BITS  # 1.8094 bits
    gap_total = H_joint - H_G

    # ── Decomposition identity check ─────────────────────────────────────────
    # The factored rate decomposes as (PILLAR_P2_v2.md §6.1):
    #   H_factored = H(G) + H(m|goal) + TC(m) + ε_estimator
    # (ε_MLE is a bound on ε_estimator, not an independent additive term)
    #
    # The joint rate is:
    #   H_joint = H_factored - TC = H(G) + H(m|goal) + ε_estimator
    #
    # So decomp_sum should match H_factored; residual = H_factored - decomp_sum.
    decomp_sum = H_G + H_dither + TC + eps_estimator
    residual = H_factored - decomp_sum  # should be small when identity holds

    # Gap above H(G): factored gap (what you'd pay without joint coding or dither)
    gap_factored = H_factored - H_G
    # Gap above H(G): joint gap (after joint coding closes TC)
    gap_joint = H_joint - H_G

    return {
        "N_messages": N,
        "z_dim": z_dim,
        "delta": delta,
        "H_G_bits": H_G,
        "H_joint_bits": H_joint,
        "H_factored_bits": H_factored,
        "TC_bits": TC,
        "H_dither_bits": H_dither,
        "eps_estimator_bits": eps_estimator,
        "eps_MLE_bound": eps_mle_bound,
        "K_obs_joint_tuples": K_obs,
        "gap_factored_bits": gap_factored,
        "gap_joint_bits": gap_joint,
        "decomp_sum_bits": decomp_sum,
        "decomp_residual_bits": residual,
        # Per-dimension breakdown
        **{f"H_dim_{k}_bits": H_marginals[k] for k in range(z_dim)},
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_table(d: dict[str, float]) -> None:
    print("\n┌─────────────────────────────────────────────────┐")
    print("│         Post-hoc Rate Decomposition              │")
    print("├─────────────────────────────────────────────────┤")
    pairs = [
        ("N messages", f"{d['N_messages']:,}"),
        ("z_dim", str(d["z_dim"])),
        ("δ", str(d["delta"])),
        ("", ""),
        ("H(G)  [lower bound]", f"{d['H_G_bits']:.4f} bits"),
        ("H_joint(m) [joint code achieves]", f"{d['H_joint_bits']:.4f} bits"),
        ("H_factored(m) [independent code]", f"{d['H_factored_bits']:.4f} bits"),
        ("gap factored above H(G)", f"{d['gap_factored_bits']:.4f} bits"),
        ("gap joint   above H(G)", f"{d['gap_joint_bits']:.4f} bits"),
        ("", ""),
        ("─── Decomposition ───────────────────────────────", ""),
        ("  H(G)", f"{d['H_G_bits']:.4f} bits"),
        ("+ H(m|goal)  [dither noise]", f"{d['H_dither_bits']:.4f} bits"),
        ("+ TC(m)      [cross-dim corr]", f"{d['TC_bits']:.4f} bits"),
        ("+ ε_estimator [hist fit]", f"{d['eps_estimator_bits']:.4f} bits"),
        ("= decomp sum", f"{d['decomp_sum_bits']:.4f} bits"),
        ("  residual (should ≈ 0)", f"{d['decomp_residual_bits']:.4f} bits"),
        ("", ""),
        ("H_factored vs H_joint", f"{d['H_factored_bits']:.4f} vs {d['H_joint_bits']:.4f}"),
        ("ε_MLE bound  (O(K^D/√N))", f"{d['eps_MLE_bound']:.4f}"),
        ("K_obs (distinct tuples)", str(d["K_obs_joint_tuples"])),
    ]
    for k in range(d["z_dim"]):
        pairs.append((f"H(m_{k})", f"{d[f'H_dim_{k}_bits']:.4f} bits"))
    for label, val in pairs:
        if label == "":
            print("│" + " " * 51 + "│")
        elif label.startswith("─"):
            print(f"│ {label:<49} │")
        else:
            print(f"│  {label:<33} {val:>13} │")
    print("└─────────────────────────────────────────────────┘\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc rate decomposition from checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to final.pt saved by train.py")
    p.add_argument("--n_rollout_msgs", type=int, default=50_000,
                   help="Number of messages to collect for histogram estimation.")
    p.add_argument("--delta", type=float, default=1.0,
                   help="Quantisation bin width δ (must match training config).")
    p.add_argument("--z_dim", type=int, default=3,
                   help="Speaker output dimension z_dim.")
    p.add_argument("--hidden_size", type=int, default=64,
                   help="Speaker hidden size (must match checkpoint).")
    p.add_argument("--smoothing", type=float, default=0.5,
                   help="Laplace smoothing α for histograms.")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _select_device(req: str) -> torch.device:
    if req != "auto":
        return torch.device(req)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)
    ckpt_path = Path(args.checkpoint)

    print(f"[post_hoc] loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # Reconstruct speaker from saved state dict.
    # Supports two checkpoint formats:
    #   1. {"speaker": {...}}  — direct speaker state dict
    #   2. {"state_dict": {"speaker.network.0.weight": ..., ...}, ...}  — joint model
    speaker = SpeakerNetwork(obs_dim=2, z_dim=args.z_dim, hidden=args.hidden_size)
    if "speaker" in state:
        speaker.load_state_dict(state["speaker"])
    elif "state_dict" in state:
        prefix = "speaker."
        sd = {k[len(prefix):]: v for k, v in state["state_dict"].items()
              if k.startswith(prefix)}
        speaker.load_state_dict(sd)
    else:
        raise KeyError(f"Cannot find speaker weights in checkpoint keys: {list(state.keys())}")
    speaker.to(device).eval()

    # Channel (SD by default — must match training)
    channel = build_channel("sd", args.delta, ste_clip=10.0).to(device)

    print(f"[post_hoc] collecting {args.n_rollout_msgs:,} messages on {device}...")
    m, goal_ids = collect_messages(
        speaker, channel, args.delta, args.n_rollout_msgs, device
    )

    # Per-goal z samples (100 each; for dither stats)
    goals_t = torch.tensor(_DEFAULT_GOALS, dtype=torch.float32, device=device)
    z_per_goal: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for gid in range(len(_DEFAULT_GOALS)):
            g = goals_t[gid].unsqueeze(0).expand(100, -1)
            z_per_goal[gid] = speaker(g).detach().cpu()

    print("[post_hoc] computing rate decomposition...")
    results = compute_decomposition(
        m, goal_ids, z_per_goal, args.delta, args.smoothing
    )

    _print_table(results)

    # Infer exp_name and seed from checkpoint path convention:
    #   <log_dir>/<exp_name>/<seed>/final.pt
    # Write directly to the canonical results location so no manual copy is needed.
    exp_name = ckpt_path.parts[-3]
    seed = ckpt_path.parts[-2]
    results_dir = Path("results/post_hoc") / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = results_dir / f"seed_{seed}.json"
    with open(canonical_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[post_hoc] saved to {canonical_path}")


if __name__ == "__main__":
    main()
