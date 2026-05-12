"""run_sc_experiments.py — Histogram-era experiment runner (F1–F21+).

Generates all training runs required to produce empirical evidence for the
figures listed in PILLAR_P2_v2.md §16.  Each stage maps to one or more figures.

Stage map:
    BASELINE          → F2 (shannon gap curve), F5 (Pareto frontier denominator)
    SC_MAIN           → F2, F13 (post-hoc vs live)
    SC_TWOPHASE       → F16 (Phase 2 L_dither closes H(m|goal))
    PARETO            → F5 (rate-distortion Pareto frontier)
    GEOMETRY          → F6, F7 (magnitude loss as geometric anchor)
    LIVE              → F13, F14 (post-hoc vs live, gradient ratio)
    DLM_CMP           → F8, F9, F10 (DLM failure modes)
    DITHER_CORRECTED  → F6, F7, F16 (re-runs with corrected dither loss; supersedes
                        E12–E14 and E18–E19 which used the wrong H_binary(frac) formula)
    NSD_CHANNEL       → §11.8 channel interaction ablation (SD vs NSD generalisation)
    PHASE2_PARETO     → extended Phase 2 λ_dither sweep to trace full SR vs H_joint frontier

    P1_DELTA          → F22–F25 (per-channel δ: global vs. per-channel, + Phase 2 combined)
                        E53: global learned δ (1 param, ablation baseline for E54)
                        E54: per-channel δ (z_dim params, primary P1 result)
                        E55: per-channel δ + Phase 2 dither at best λ=5e-3 (P1+P2 combined)
    P4_RB             → F26–F28 (Rao-Blackwell gradient estimator vs STE)
                        E57: STE baseline (same Phase 1 config as E09; fair reference for E58)
                        E58: RB-joint (2 extra forward passes per minibatch)
                        E60: RB-joint + Phase 2 dither at λ=5e-3 (P2×P4 combined)
    P3_DEPLOY         → F29 (deployment consistency; checkpoint eval, not training)
                        Stub — prints instructions for the separate checkpoint eval script.

Post-hoc analysis (no extra training runs):
    TC_ZDIM      → F15 (TC grows with z_dim; uses BASELINE + SC_MAIN checkpoints)
    DECOMP       → F1 (rate decomposition bar chart; uses SC_MAIN checkpoint)

Usage (from repo root):
    # Dry run to see what will be launched:
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.experiments.run_sc_experiments \\
        --stage BASELINE --dry_run

    # Run a stage (5 seeds):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.experiments.run_sc_experiments \\
        --stage SC_MAIN --seeds 0 1 2 3 4 --log_dir runs/sc_ablation

    # Run all stages sequentially (use shell script for resource control):
        --stage all

Skip logic:
    A run is skipped if <log_dir>/<exp_name>/<seed>/metrics.csv already has
    >= MIN_ROWS rows, where MIN_ROWS = total_timesteps // (n_envs * n_steps) - 5.
    This makes the runner safe to restart after interruption.

Debug notes:
    - Each run saves config.json alongside metrics.csv — use load_runs.py to
      aggregate across seeds after completion.
    - Monitor runs/sc_ablation/<exp_name>/<seed>/metrics.csv during training.
    - The shell script run_p2_experiments.sh calls this file with resource throttling.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterator


_BASE_CMD = [sys.executable, "-m", "onpolicy.envs.toyproblem.train"]

# Shared fixed settings across all stages.
# Resource-conscious defaults for a laptop (16 envs, 256 steps, 64 hidden).
_SHARED = dict(
    channel="sd",
    total_timesteps="2000000",   # 2M steps for reliable convergence
    n_envs="16",
    n_steps="256",
    hidden_size="64",
    lr="3e-4",
    update_epochs="10",
    num_minibatches="4",
    z_dim="3",
    delta="1.0",
)

# Source coding defaults (histogram estimator)
_SC_DEFAULTS = dict(
    use_source_coding="",          # store_true flag
    source_coding_smoothing="0.5",
)

# Minimum completed rows to consider a run done
_MIN_ROWS_FRAC = 0.95  # run must have completed 95% of expected updates


def _is_done(log_dir: str, exp_name: str, seed: int, total_timesteps: int,
              n_envs: int, n_steps: int) -> bool:
    """Return True if this run already has sufficient logged updates."""
    csv = Path(log_dir) / exp_name / str(seed) / "metrics.csv"
    if not csv.exists():
        return False
    try:
        with open(csv) as f:
            n_rows = sum(1 for _ in f) - 1  # subtract header
        expected = total_timesteps // (n_envs * n_steps)
        return n_rows >= int(expected * _MIN_ROWS_FRAC)
    except Exception:
        return False


def _run(exp_name: str, seed: int, log_dir: str, extra: dict,
         dry_run: bool = False) -> None:
    """Construct and optionally execute one training run command."""
    ts = int(extra.get("total_timesteps", _SHARED["total_timesteps"]))
    ne = int(extra.get("n_envs", _SHARED["n_envs"]))
    ns = int(extra.get("n_steps", _SHARED["n_steps"]))

    if _is_done(log_dir, exp_name, seed, ts, ne, ns):
        print(f"  [skip] {exp_name} seed={seed}  (already done)")
        return

    cmd = _BASE_CMD + ["--exp_name", exp_name, "--seed", str(seed),
                       "--log_dir", log_dir]
    for k, v in extra.items():
        if v == "":          # store_true flags
            cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]

    if dry_run:
        print(f"  [dry]  {exp_name} seed={seed}")
        print(f"         {' '.join(cmd[2:])}")
    else:
        print(f"  [run]  {exp_name} seed={seed}")
        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

def _stage_baseline(args) -> Iterator[tuple[str, dict]]:
    """F2, F5: no-compression and magnitude-only baselines.

    Runs:
        no_comms         — no communication loss at all (H(m) ceiling)
        mag_lam{λ}       — magnitude loss at 7 λ values (Pareto denominator)
    """
    # No compression — establishes unconstrained H(m)
    yield "no_comms", {**_SHARED, "lambda_comms": "0.0", "loss_comms_mode": "magnitude"}

    # Magnitude baseline λ sweep
    for lam in ["1e-5", "1e-4", "5e-4", "1e-3", "4e-3", "1e-2", "3e-2"]:
        yield f"mag_lam{lam}", {**_SHARED,
                                "lambda_comms": lam,
                                "loss_comms_mode": "magnitude"}


def _stage_sc_main(args) -> Iterator[tuple[str, dict]]:
    """F2, F13, DECOMP: primary SC design — post-hoc histogram, no live training.

    The primary experiment: RL with magnitude anchor in Phase 1 only.
    Source coding applied post hoc to the converged checkpoint.
    """
    # Phase 1 only: magnitude anchor + no compression during training
    # Source coding is a post-hoc analysis step (run post_hoc_coding.py)
    yield "sc_posthoc_mag", {**_SHARED,
                              "lambda_comms": "5e-4",
                              "loss_comms_mode": "magnitude"}

    # Also run with entropy loss active during training (for comparison in F13)
    yield "sc_live_entropy", {**_SHARED, **_SC_DEFAULTS,
                               "lambda_comms": "5e-4",
                               "loss_comms_mode": "entropy"}

    # Both losses active during training
    yield "sc_live_both", {**_SHARED, **_SC_DEFAULTS,
                            "lambda_comms": "5e-4",
                            "loss_comms_mode": "both"}


def _stage_sc_twophase(args) -> Iterator[tuple[str, dict]]:
    """F16: Phase 2 L_dither reduces H(m|goal) without dropping SR.

    Phase 1: magnitude anchor until SR >= phase1_sr_threshold.
    Phase 2: dither entropy loss only (lambda_dither activates).
    """
    for lam_dither in ["1e-4", "5e-4", "1e-3"]:
        name = f"sc_twophase_dither{lam_dither}"
        yield name, {**_SHARED, **_SC_DEFAULTS,
                     "lambda_comms": "5e-4",
                     "loss_comms_mode": "magnitude",
                     "phase1_sr_threshold": "0.99",
                     "lambda_dither": lam_dither,
                     "total_timesteps": "3000000"}  # more steps for Phase 2


def _stage_pareto(args) -> Iterator[tuple[str, dict]]:
    """F5: rate-distortion Pareto frontier for both magnitude and SC.

    Both conditions use the same λ values so the two curves are comparable.
    """
    lambdas = ["1e-5", "1e-4", "5e-4", "1e-3", "4e-3", "1e-2", "3e-2"]
    for lam in lambdas:
        # Magnitude curve
        yield f"pareto_mag_lam{lam}", {**_SHARED,
                                        "lambda_comms": lam,
                                        "loss_comms_mode": "magnitude"}
        # SC curve (live entropy loss for fairest comparison)
        yield f"pareto_sc_lam{lam}", {**_SHARED, **_SC_DEFAULTS,
                                       "lambda_comms": lam,
                                       "loss_comms_mode": "entropy"}


def _stage_geometry(args) -> Iterator[tuple[str, dict]]:
    """F6, F7: magnitude loss as geometric anchor (neural collapse analogy).

    Four configs: no anchor, L_mag only, L_dither only (Phase 2), both.
    Post-hoc measures TC(m), H(m|goal), and Gram-matrix structure.
    """
    base = {**_SHARED, "total_timesteps": "2000000"}

    # (A) No anchor — pure RL
    yield "geom_no_anchor", {**base, "lambda_comms": "0.0",
                              "loss_comms_mode": "magnitude"}

    # (B) L_mag only (Phase 1) — the standard design
    yield "geom_mag_only", {**base, "lambda_comms": "5e-4",
                             "loss_comms_mode": "magnitude"}

    # (C) L_dither only (Phase 2, no Phase 1 anchor)
    # Phase 1 SR threshold still needed to detect when Phase 2 starts
    yield "geom_dither_only", {**base, **_SC_DEFAULTS,
                                "lambda_comms": "0.0",
                                "loss_comms_mode": "entropy",
                                "phase1_sr_threshold": "0.99",
                                "lambda_dither": "5e-4",
                                "total_timesteps": "3000000"}

    # (D) Both: L_mag in Phase 1, L_dither in Phase 2
    yield "geom_both", {**base, **_SC_DEFAULTS,
                         "lambda_comms": "5e-4",
                         "loss_comms_mode": "magnitude",
                         "phase1_sr_threshold": "0.99",
                         "lambda_dither": "5e-4",
                         "total_timesteps": "3000000"}


def _stage_live(args) -> Iterator[tuple[str, dict]]:
    """F13, F14: post-hoc vs live coding (three-way comparison, §8.4).

    (A) post-hoc — SC applied only after convergence (train.py run = Phase 1 only)
    (B) live from update 0 — histogram + score function active throughout
    (C) live + Phase 2 dither — config B with dither added in Phase 2
    """
    # Config A: same as sc_main / sc_posthoc_mag — skip if already run
    yield "live_A_posthoc", {**_SHARED,
                              "lambda_comms": "5e-4",
                              "loss_comms_mode": "magnitude"}

    # Config B: live histogram + score function from update 0
    yield "live_B_live_sc", {**_SHARED, **_SC_DEFAULTS,
                              "lambda_comms": "5e-4",
                              "loss_comms_mode": "entropy"}

    # Config C: live + Phase 2 dither
    yield "live_C_live_dither", {**_SHARED, **_SC_DEFAULTS,
                                  "lambda_comms": "5e-4",
                                  "loss_comms_mode": "entropy",
                                  "phase1_sr_threshold": "0.99",
                                  "lambda_dither": "5e-4",
                                  "total_timesteps": "3000000"}


def _stage_dlm_cmp(args) -> Iterator[tuple[str, dict]]:
    """F8, F9, F10: DLM comparison showing structural failure modes.

    Uses the existing P2-FIX configs (fix_baseline, fix_B1..fix_B4).
    These are DLM-based and should be compared against histogram metrics.
    Delegates to run_p2_ablation.py P2-FIX stage — this stage just emits
    a reminder message and the correct invocation command.
    """
    # This stage emits no runs directly; it points to the DLM runner.
    # Yielding zero items so the stage system handles it gracefully.
    print(
        "\n[DLM_CMP] This stage uses run_p2_ablation.py --stage P2-FIX.\n"
        "  Run: KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\\n"
        "       python -m onpolicy.envs.toyproblem.experiments.run_p2_ablation \\\n"
        "       --stage P2-FIX --seeds 0 1 2 3 4 --log_dir runs/dlm_cmp\n"
    )
    return
    yield  # make generator


def _stage_dither_corrected(args) -> Iterator[tuple[str, dict]]:
    """E42–E46: Re-runs with the corrected dither loss formula.

    Background: E12–E14 (sc_twophase) and E18–E19 (geom_dither_only / geom_both)
    used dither_channel_loss with the formula H_binary(frac) — correct for a
    Bernoulli or round quantiser but wrong for the floor SD channel used here.
    For floor(z/δ + U(-0.5,0.5)), the correct channel entropy per dimension is
    H_binary(|frac − 0.5|), which is minimised at frac = 0.5 (bin centres, zero
    noise) and maximised at frac = 0 (bin boundaries, 1 bit/dim).

    The old formula minimised at frac = 0, pushing z to integer multiples of δ
    (bin boundaries = maximum dither noise).  The loss was applied "backwards".

    Fix applied in source_coding.py (2026-05-05): both dither_channel_loss and
    dither_channel_stats now use H_binary(|frac − 0.5|) with the correct gradient
        dH/dz_k = log₂((1−g)/g) · sign(frac − 0.5) / δ,  g = |frac − 0.5|
    which pushes z toward frac = 0.5 (minimum noise locus).

    E42 (geom_dither_only_v2) and E43 (geom_both_v2) supersede E18 and E19.
    E44–E46 supersede E12–E14 for Phase 2 H(m|goal) reduction claims.

    Expected outcome: H(m|goal) → 0 as frac → 0.5; decomposition residual < 0.3
    bits; hist_H decreases below Phase 1 level (E44–E46 only).
    """
    # E42 — geom_dither_only with corrected loss (supersedes E18)
    # No magnitude anchor; only dither loss pushes frac → 0.5 after SR converges.
    yield "geom_dither_only_v2", {**_SHARED, **_SC_DEFAULTS,
                                   "lambda_comms": "0.0",
                                   "loss_comms_mode": "entropy",
                                   "phase1_sr_threshold": "0.99",
                                   "lambda_dither": "5e-4",
                                   "total_timesteps": "3000000"}

    # E43 — geom_both with corrected loss (supersedes E19)
    # L_mag in Phase 1 anchors geometry; corrected L_dither in Phase 2 reduces noise.
    yield "geom_both_v2", {**_SHARED, **_SC_DEFAULTS,
                            "lambda_comms": "5e-4",
                            "loss_comms_mode": "magnitude",
                            "phase1_sr_threshold": "0.99",
                            "lambda_dither": "5e-4",
                            "total_timesteps": "3000000"}

    # E44–E46 — sc_twophase with corrected loss, λ sweep (supersedes E12–E14)
    # Phase 1: magnitude anchor until SR ≥ 0.99.
    # Phase 2: corrected dither loss at three λ values to show H(m|goal) reduction.
    for lam_dither in ["1e-4", "5e-4", "1e-3"]:
        name = f"sc_twophase_dither{lam_dither}_v2"
        yield name, {**_SHARED, **_SC_DEFAULTS,
                     "lambda_comms": "5e-4",
                     "loss_comms_mode": "magnitude",
                     "phase1_sr_threshold": "0.99",
                     "lambda_dither": lam_dither,
                     "total_timesteps": "3000000"}


def _stage_zdim_sweep(args) -> Iterator[tuple[str, dict]]:
    """E37–E38: z_dim ∈ {1, 2} anchor points for F15 (TC grows with dimensionality).

    Uses the same config as sc_posthoc_mag (E09, z_dim=3) so the three points
    can be compared directly on a TC vs z_dim plot.  z_dim=3 already exists —
    only z_dim=1 and z_dim=2 need to be run here.

    F15 claim: TC(m) ≈ 0 at z_dim=1 (no cross-dim correlations possible),
    grows at z_dim=2, and reaches the E09 value at z_dim=3.
    """
    for z_dim in [1, 2]:
        yield f"sc_posthoc_mag_zdim{z_dim}", {
            **_SHARED,
            "z_dim": str(z_dim),
            "lambda_comms": "5e-4",
            "loss_comms_mode": "magnitude",
        }


def _stage_nsd_channel(args) -> Iterator[tuple[str, dict]]:
    """E47–E48: §11.8 SD vs NSD channel interaction ablation.

    Tests whether the histogram estimator and Phase 2 dither loss generalise
    automatically to the non-subtractive dithering (NSD) channel.  q_hist is
    built from actual m samples regardless of channel type, so generalisation
    is expected but must be confirmed empirically.

    Runs:
        nsd_posthoc_mag            (E47) — NSD baseline; same as sc_posthoc_mag
                                           but channel=nsd.  Post-hoc coding
                                           after training measures H_joint.
        nsd_twophase_dither5e-4_v2 (E48) — NSD + Phase 2 dither at λ=5e-4
                                           (matches E45 for apples-to-apples
                                           comparison).

    Expected: SR ≈ 1.0 and H_joint ≈ SD baseline on both runs, confirming
    the estimator is channel-agnostic.
    """
    # E47 — NSD magnitude anchor baseline (post-hoc SC applied offline)
    yield "nsd_posthoc_mag", {**_SHARED,
                               "channel": "nsd",
                               "lambda_comms": "5e-4",
                               "loss_comms_mode": "magnitude"}

    # E48 — NSD Phase 2 dither (corrected loss + E39 listener fix)
    yield "nsd_twophase_dither5e-4_v2", {**_SHARED, **_SC_DEFAULTS,
                                          "channel": "nsd",
                                          "lambda_comms": "5e-4",
                                          "loss_comms_mode": "magnitude",
                                          "phase1_sr_threshold": "0.99",
                                          "lambda_dither": "5e-4",
                                          "total_timesteps": "3000000"}


def _stage_phase2_pareto(args) -> Iterator[tuple[str, dict]]:
    """E49–E52: Phase 2 Pareto curve extension — additional λ_dither values.

    E44–E46 cover λ_dither ∈ {1e-4, 5e-4, 1e-3}, all reaching SR=1.000.
    This stage adds four more values to trace the full rate-distortion frontier:

        5e-5  (E49) — very gentle push; H_joint expected ≈ 2.5+ bits, SR=1.000
        2e-4  (E50) — fills gap between E44 and E45
        2e-3  (E51) — aggressive; may start trading SR for H_joint
        5e-3  (E52) — very aggressive; expected SR drop, revealing tradeoff

    Together with E44–E46, these 7 points trace a Phase 2 Pareto curve that can
    be overlaid on F5 to show Phase 2 dominates the magnitude and live-SC
    frontiers across the full operating range.
    """
    for lam_dither in ["5e-5", "2e-4", "2e-3", "5e-3"]:
        name = f"sc_twophase_dither{lam_dither}_v2"
        yield name, {**_SHARED, **_SC_DEFAULTS,
                     "lambda_comms": "5e-4",
                     "loss_comms_mode": "magnitude",
                     "phase1_sr_threshold": "0.99",
                     "lambda_dither": lam_dither,
                     "total_timesteps": "3000000"}


def _stage_p1_delta(args) -> Iterator[tuple[str, dict]]:
    """F22–F25: per-channel δ ablation (Pillar 1).

    Measures whether learnable per-channel bin widths δ_k = softplus(α_k) reduce
    H_joint below the global-δ baseline, and whether combining with Phase 2
    dither beats P2-alone (E52, H_joint=2.094 bits).

    Dependency: sc_posthoc_mag (E09) is the fixed-δ baseline — run SC_MAIN first.

    Runs:
        p1_global_delta     (E53) — 1-param global learned δ.  Isolates whether any
                                    δ adaptation helps at all vs fixed δ=1.0 (E09).
        p1_perchannel_delta (E54) — z_dim-param per-channel δ_k.  Primary P1 result:
                                    do dimensions specialise?
        p1_perchannel_p2_5e-3
                            (E55) — E54 config + Phase 2 dither at λ=5e-3 (best P2 λ
                                    from E52).  Headline P1+P2 combined result.
        p1_heuristic_delta  (E56) — Rule-based baseline: δ_k ∝ 1/H(m_k) at Phase 2
                                    onset, then frozen.  Compared to E54 to quantify
                                    the gap between heuristic and learned adaptation.
                                    Not a contribution — baseline only.

    Post-hoc coding must be run on E53/E54/E56 checkpoints after training to obtain H_joint.
    E55 post-hoc gives the P1+P2 vs P2-alone comparison for F25.
    """
    _p1_phase1_base = {
        **_SHARED,
        **_SC_DEFAULTS,
        "lambda_comms": "5e-4",
        "loss_comms_mode": "magnitude",
        "lr_delta": "1e-4",
    }

    # E53 — global learned δ (1 param); ablation baseline for E54
    yield "p1_global_delta", {
        **_p1_phase1_base,
        "learn_global_delta": "",    # store_true
    }

    # E54 — per-channel δ (z_dim params); primary P1 result
    yield "p1_perchannel_delta", {
        **_p1_phase1_base,
        "learn_delta": "",           # store_true
    }

    # E55 — per-channel δ + Phase 2 dither at best λ (P1+P2 combined)
    yield "p1_perchannel_p2_5e-3", {
        **_p1_phase1_base,
        "learn_delta": "",
        "phase1_sr_threshold": "0.99",
        "lambda_dither": "5e-3",
        "total_timesteps": "3000000",
    }

    # E56 — heuristic per-channel δ (baseline comparison only, not a contribution)
    # At Phase 2 onset, sets δ_k ∝ 1/H(m_k) from the empirical histogram, then freezes.
    # Compared against E54 to quantify the gap between rule-based and learned adaptation.
    yield "p1_heuristic_delta", {
        **_p1_phase1_base,
        "heuristic_delta": "",       # store_true
        "phase1_sr_threshold": "0.99",
        "total_timesteps": "3000000",
    }


def _stage_p4_rb(args) -> Iterator[tuple[str, dict]]:
    """F26–F28: Rao-Blackwell gradient estimator (Pillar 4).

    Tests whether replacing the STE speaker task gradient with the exact
    Rao-Blackwell finite-difference estimator (g_RB_k = [L_hi - L_lo] / δ_k)
    speeds up Phase 1 convergence and/or improves final H_joint.

    All runs use SD channel, magnitude loss in Phase 1 (same anchor as E09).
    The RB proxy replaces actor_loss → speaker; actor+critic still update the
    listener and critic via the detached z_hat path.

    Runs:
        p4_ste_baseline (E57) — STE with magnitude anchor; fair reference for E58.
                                 Config identical to sc_posthoc_mag (E09); may be
                                 skipped if E09 already exists in log_dir.
        p4_rb_joint     (E58) — RB-joint (2 extra listener forward passes per mb).
                                 Primary P4 result: does RB reach SR=0.99 faster?
        p4_rb_joint_p2  (E60) — RB-joint + Phase 2 dither λ=5e-3 (P2×P4 combined).
                                 Does RB change the Phase 2 H_joint floor?

    Note: RB-per-dim (E59) is deferred — per_dim mode needs full implementation.
    Note: P1×P4 conditional experiment (E61) is not included; run manually if
          E54 (p1_perchannel_delta) shows insufficient δ_k differentiation.
    """
    _p4_phase1_base = {
        **_SHARED,
        **_SC_DEFAULTS,
        "lambda_comms": "5e-4",
        "loss_comms_mode": "magnitude",
    }

    # E57 — STE reference (same as E09 sc_posthoc_mag; separate name for clean pairing)
    yield "p4_ste_baseline", {**_p4_phase1_base}

    # E58 — RB-joint: 2 extra listener forward passes per minibatch
    yield "p4_rb_joint", {
        **_p4_phase1_base,
        "use_rb_gradient": "",       # store_true
        "rb_mode": "joint",
    }

    # E60 — RB-joint + Phase 2 dither (P2×P4 combined headline claim)
    yield "p4_rb_joint_p2_5e-3", {
        **_p4_phase1_base,
        "use_rb_gradient": "",
        "rb_mode": "joint",
        "phase1_sr_threshold": "0.99",
        "lambda_dither": "5e-3",
        "total_timesteps": "3000000",
    }


def _stage_p3_deploy(args) -> Iterator[tuple[str, dict]]:
    """F29: deployment consistency evaluation (Pillar 3) — STUB.

    P3 evaluation requires loading trained checkpoints (final.pt) from E09 (SD)
    and E47 (NSD), then running rollout-only evaluation with deploy_eval=True.
    This cannot be done as a training run because --deploy_eval during training
    creates an old_log_probs / new_logp mismatch (rollout uses z_hat_deploy,
    update uses STE z_hat), which breaks the PPO importance weight.

    Required steps (run manually after E09 and E47 are done):

        # SD deployment evaluation (E62)
        python -m onpolicy.envs.toyproblem.eval_deploy \\
            --checkpoint runs/sc_ablation/sc_posthoc_mag/{seed}/final.pt \\
            --channel sd --n_eval_episodes 1000

        # NSD deployment evaluation (E63)
        python -m onpolicy.envs.toyproblem.eval_deploy \\
            --checkpoint runs/sc_ablation/nsd_posthoc_mag/{seed}/final.pt \\
            --channel nsd --n_eval_episodes 1000

    eval_deploy.py does not yet exist — add it when P3 experiments are ready.
    It should: load checkpoint, build trainer with deploy_eval=True, run 1000
    episodes, report SR_deploy vs SR_train (from metrics.csv final row).

    Expected: SD SR_deploy ≈ SR_train (Schuchman distributional equiv. sufficient);
              NSD SR_deploy = SR_train exactly (sample-consistent by construction).
    """
    print(
        "\n[P3_DEPLOY] Deployment evaluation requires a checkpoint eval script.\n"
        "\n"
        "  Prerequisites: sc_posthoc_mag (E09) and nsd_posthoc_mag (E47) must be done.\n"
        "  Run SC_MAIN and NSD_CHANNEL stages first.\n"
        "\n"
        "  Once eval_deploy.py exists, run per seed:\n"
        "    # E62 — SD deployment evaluation\n"
        "    for seed in 0 1 2 3 4; do\n"
        "      python -m onpolicy.envs.toyproblem.eval_deploy \\\n"
        "        --checkpoint runs/sc_ablation/sc_posthoc_mag/$seed/final.pt \\\n"
        "        --channel sd --n_eval_episodes 1000\n"
        "    done\n"
        "\n"
        "    # E63 — NSD deployment evaluation\n"
        "    for seed in 0 1 2 3 4; do\n"
        "      python -m onpolicy.envs.toyproblem.eval_deploy \\\n"
        "        --checkpoint runs/sc_ablation/nsd_posthoc_mag/$seed/final.pt \\\n"
        "        --channel nsd --n_eval_episodes 1000\n"
        "    done\n"
        "\n"
        "  See PILLAR_P3_v2.md §7 for the eval_deploy.py implementation spec.\n"
    )
    return
    yield  # make generator


def _stage_all_pillars(args) -> Iterator[tuple[str, dict]]:
    """F-ALL: All-Pillars Integration — SD (E64) and NSD (E65) variants.

    Combines all four pillars:
        P1: per-channel δ_k (learn_delta)
        P2: histogram rate loss + Phase 2 dither (use_source_coding, lambda_dither)
        P4: Rao-Blackwell gradient estimator (use_rb_gradient, rb_mode=joint)
        P3: NSD variant adds TPDF sample-consistent channel (channel=nsd)

    Gate: this stage yields zero configs unless --override_gate is passed.
    Run only after all individual pillar experiments (E53–E63) are validated.
    See docs/pillars/PILLAR_ALL_v2.md §4 for the gate rationale.

    Runs:
        all_pillars_sd  (E64) — P1+P2+P4, SD channel (distributional consistency)
        all_pillars_nsd (E65) — P1+P2+P4, NSD channel (sample consistency, P3 full)
    """
    if not getattr(args, "override_gate", False):
        print(
            "\n[ALL_PILLARS] Gate active — individual pillar experiments not yet validated.\n"
            "\n"
            "  Prerequisites (all must be DONE in EXPERIMENT_REGISTRY.md):\n"
            "    E53 (p1_global_delta), E54 (p1_perchannel_delta)  ← P1\n"
            "    E57 (p4_ste_baseline), E58 (p4_rb_joint)          ← P4\n"
            "    E52 (sc_twophase_dither_5e-3_v2)                  ← P2 (already DONE)\n"
            "    E47 (nsd_posthoc_mag), E48 (nsd_twophase_5e-3)    ← P3/NSD (already DONE)\n"
            "\n"
            "  Once all prerequisites are validated, re-run with --override_gate.\n"
        )
        return
        yield  # make generator

    _all_base = {
        **_SHARED,
        **_SC_DEFAULTS,
        "lambda_comms": "5e-4",
        "loss_comms_mode": "magnitude",
        "learn_delta": "",
        "lr_delta": "1e-4",
        "phase1_sr_threshold": "0.99",
        "lambda_dither": "5e-3",
        "use_rb_gradient": "",
        "rb_mode": "joint",
        "total_timesteps": "3000000",
    }

    # E64 — all pillars, SD channel
    yield "all_pillars_sd", {
        **_all_base,
        "channel": "sd",
    }

    # E65 — all pillars, NSD channel (P3 sample-consistency fully active)
    yield "all_pillars_nsd", {
        **_all_base,
        "channel": "nsd",
    }


_STAGE_FNS: dict[str, object] = {
    "BASELINE": _stage_baseline,
    "SC_MAIN": _stage_sc_main,
    "SC_TWOPHASE": _stage_sc_twophase,
    "PARETO": _stage_pareto,
    "GEOMETRY": _stage_geometry,
    "LIVE": _stage_live,
    "DLM_CMP": _stage_dlm_cmp,
    "DITHER_CORRECTED": _stage_dither_corrected,
    "ZDIM_SWEEP": _stage_zdim_sweep,
    "NSD_CHANNEL": _stage_nsd_channel,
    "PHASE2_PARETO": _stage_phase2_pareto,
    "P1_DELTA": _stage_p1_delta,
    "P4_RB": _stage_p4_rb,
    "P3_DEPLOY": _stage_p3_deploy,
    "ALL_PILLARS": _stage_all_pillars,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Histogram-era experiment runner for F1–F20 figures."
    )
    p.add_argument("--stage", type=str,
                   choices=list(_STAGE_FNS.keys()) + ["all"],
                   required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--log_dir", type=str, default="runs/sc_ablation")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--override_gate", action="store_true",
                   help="Skip sequencing gate checks (ALL_PILLARS stage). "
                        "Use only after all individual pillar experiments are validated.")
    args = p.parse_args()

    stages = list(_STAGE_FNS.keys()) if args.stage == "all" else [args.stage]

    for stage in stages:
        conditions = list(_STAGE_FNS[stage](args))
        n_total = len(conditions) * len(args.seeds)
        print(f"\n=== {stage}: {len(conditions)} configs × {len(args.seeds)} seeds = {n_total} runs ===")
        for exp_name, extra in conditions:
            for seed in args.seeds:
                _run(exp_name, seed, args.log_dir, extra, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
