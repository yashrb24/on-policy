"""P2 systematic ablation study — 7 staged experiments.

Stage ordering:
  P2-FIX Fix ladder        (25 runs) — must run FIRST after P2-A failure diagnosis.
                                       Tests Levels 1–5 of the fix ladder in isolation.
                                       See docs/pillars/PILLAR_P2.md §13. Run before P2-B.
  P2-A  Model selection   (115 runs) — K × model_type × loss_comms_mode for context A (trainable);
                                       context B measurement-only runs alongside (K=5, no speaker grad)
  P2-B  λ re-sweep         (35 runs) — P2 rate-distortion frontier; find optimal λ
  P2-C  z_dim interaction  (40 runs) — factored vs joint gap at z_dim={1,2,3}
  P2-D  δ interaction      (40 runs) — P2 gain vs quantisation width
  P2-E  Channel (sd/nsd)   (20 runs) — P2 compatibility with NSD dithering
  P2-F  Robustness OAT     (60 runs) — lr_qphi_mult, n_qphi_steps, n_warmup_steps

Usage (from repo root):
    # Stage P2-A (model selection):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.experiments.run_p2_ablation \\
        --stage P2-A --seeds 0 1 2 3 4 --log_dir runs/p2_ablation

    # Stage P2-B (after inspecting P2-A results):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.experiments.run_p2_ablation \\
        --stage P2-B --seeds 0 1 2 3 4 --log_dir runs/p2_ablation \\
        --best_K 5 --best_model_type factored --best_context A --best_loss_mode entropy

    # Stages P2-C through P2-F (after P2-B results):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.experiments.run_p2_ablation \\
        --stage P2-C --seeds 0 1 2 3 4 --log_dir runs/p2_ablation \\
        --best_K 5 --best_model_type factored --best_context A \\
        --best_loss_mode entropy --best_lambda 4e-3

    # Dry run any stage:
        ... --dry_run

Analysis after each stage:
    from onpolicy.envs.toyproblem.analysis.load_runs import load_sweep, final_metrics, seed_aggregate
    df = load_sweep("runs/p2_ablation")
    summary = final_metrics(df)
    # group by exp_name to compare conditions within each stage
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterator


_BASE_TRAIN_CMD = [sys.executable, "-m", "onpolicy.envs.toyproblem.train"]

# Fixed settings shared across all stages (overridden per-stage as needed).
# Populated from Phase 2 baseline_best.yaml defaults — edit if Phase 2 winner differs.
_SHARED = dict(
    channel="sd",
    total_timesteps="500000",
    n_envs="16",
    n_steps="256",
    hidden_size="64",
    lr="3e-4",
    update_epochs="10",
    num_minibatches="4",
)

# P2 training defaults (used unless overridden by robustness OAT in P2-F)
_P2_TRAIN_DEFAULTS = dict(
    n_warmup_steps="5000",
    n_qphi_steps="3",
    lr_qphi_mult="10.0",
)


def _run(exp_name: str, seed: int, log_dir: str, extra: dict) -> None:
    cmd = _BASE_TRAIN_CMD + ["--exp_name", exp_name, "--seed", str(seed),
                              "--log_dir", log_dir]
    for k, v in extra.items():
        if v == "":          # store_true flags (e.g. --use_entropy_model)
            cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    print(f"  [run] {exp_name} seed={seed}")
    subprocess.run(cmd, check=True)


def _stage_a(args) -> Iterator[tuple[str, dict]]:
    """Model selection — 115 runs total (23 configs × 5 seeds).

    Context A rows (20 configs): sweep K × model_type × loss_comms_mode.
      These are the TRAINABLE configs — q_φ drives the speaker backward loss.
    Context B rows (2 configs): K=5, loss_comms_mode=magnitude, measurement only.
      q_φ(m|z) trains on the forward loss to produce H(m|z) metrics; the
      speaker backward is disabled — context B cannot be an ablation winner.
      See docs/MATH.md §12 Fix 3 for the degeneracy derivation.
    Baseline (1 config): magnitude surrogate, no entropy model.

    Grid (23 configs):
      baseline_magnitude          : 1
      factored × A × {entropy,both} × K{1,3,5,10,20}: 10
      joint    × A × {entropy,both} × K{1,3,5,10,20}: 10
      factored × B × magnitude   × K=5              : 1
      joint    × B × magnitude   × K=5              : 1
    """
    base = {**_SHARED, "delta": args.phase2_delta, "z_dim": args.phase2_z_dim,
            "lambda_comms": args.phase2_lambda, **_P2_TRAIN_DEFAULTS}

    yield "baseline_magnitude", base

    # Context A: K × model_type × loss_comms_mode sweeps (trainable)
    K_values = [1, 3, 5, 10, 20]
    for K in K_values:
        for mode in ["entropy", "both"]:
            for model_type in ["factored", "joint"]:
                name = f"p2_A_{model_type}_K{K}_{mode}"
                yield name, {**base,
                              "use_entropy_model": "",
                              "entropy_model_K": str(K),
                              "entropy_model_type": model_type,
                              "entropy_model_context": "A",
                              "loss_comms_mode": mode}

    # Context B: measurement-only (K=5, magnitude loss — no speaker backward)
    for model_type in ["factored", "joint"]:
        name = f"p2_B_{model_type}_K5_measurement"
        yield name, {**base,
                      "use_entropy_model": "",
                      "entropy_model_K": "5",
                      "entropy_model_type": model_type,
                      "entropy_model_context": "B",
                      "loss_comms_mode": "magnitude"}


def _stage_b(args) -> Iterator[tuple[str, dict]]:
    """λ re-sweep for best P2 config; generates P2 rate-distortion frontier (35 new runs)."""
    base = {**_SHARED, "delta": args.phase2_delta, "z_dim": args.phase2_z_dim,
            **_P2_TRAIN_DEFAULTS}
    lambdas = ["1e-5", "1e-4", "5e-4", "1e-3", "4e-3", "1e-2", "3e-2"]
    for lam in lambdas:
        name = f"p2_{args.best_context}_{args.best_model_type}_K{args.best_K}_{args.best_loss_mode}_lam{lam}"
        yield name, {**base, "lambda_comms": lam,
                     "use_entropy_model": "",
                     "entropy_model_K": str(args.best_K),
                     "entropy_model_type": args.best_model_type,
                     "entropy_model_context": args.best_context,
                     "loss_comms_mode": args.best_loss_mode}


def _stage_c(args) -> Iterator[tuple[str, dict]]:
    """z_dim interaction: factored vs joint gap at z_dim={1,2,3} (40 runs)."""
    base = {**_SHARED, "delta": args.phase2_delta,
            "lambda_comms": args.best_lambda, **_P2_TRAIN_DEFAULTS}
    for z_dim in [1, 2, 3]:
        # Baseline
        yield f"baseline_magnitude_zdim{z_dim}", {**base, "z_dim": str(z_dim)}
        # Factored
        yield f"p2_A_factored_K{args.best_K}_zdim{z_dim}", {
            **base, "z_dim": str(z_dim),
            "use_entropy_model": "",
            "entropy_model_K": str(args.best_K),
            "entropy_model_type": "factored",
            "entropy_model_context": "A",
            "loss_comms_mode": args.best_loss_mode,
        }
        # Joint only at z_dim >= 2 (trivially equal to factored at z_dim=1)
        if z_dim >= 2:
            yield f"p2_A_joint_K{args.best_K}_zdim{z_dim}", {
                **base, "z_dim": str(z_dim),
                "use_entropy_model": "",
                "entropy_model_K": str(args.best_K),
                "entropy_model_type": "joint",
                "entropy_model_context": "A",
                "loss_comms_mode": args.best_loss_mode,
            }


def _stage_d(args) -> Iterator[tuple[str, dict]]:
    """δ interaction: P2 gain vs quantisation width (40 runs)."""
    base = {**_SHARED, "z_dim": args.best_z_dim, "lambda_comms": args.best_lambda,
            **_P2_TRAIN_DEFAULTS}
    for delta in ["0.5", "1.0", "5.0", "10.0"]:
        yield f"baseline_magnitude_delta{delta}", {**base, "delta": delta}
        yield f"p2_best_delta{delta}", {
            **base, "delta": delta,
            "use_entropy_model": "",
            "entropy_model_K": str(args.best_K),
            "entropy_model_type": args.best_model_type,
            "entropy_model_context": args.best_context,
            "loss_comms_mode": args.best_loss_mode,
        }


def _stage_e(args) -> Iterator[tuple[str, dict]]:
    """Channel interaction: P2 with sd vs nsd (20 runs)."""
    base = {**_SHARED, "z_dim": args.best_z_dim, "delta": args.best_delta,
            "lambda_comms": args.best_lambda, **_P2_TRAIN_DEFAULTS}
    for channel in ["sd", "nsd"]:
        yield f"baseline_magnitude_{channel}", {**base, "channel": channel}
        yield f"p2_best_{channel}", {
            **base, "channel": channel,
            "use_entropy_model": "",
            "entropy_model_K": str(args.best_K),
            "entropy_model_type": args.best_model_type,
            "entropy_model_context": args.best_context,
            "loss_comms_mode": args.best_loss_mode,
        }


def _stage_f(args) -> Iterator[tuple[str, dict]]:
    """Training robustness OAT: lr_qphi_mult, n_qphi_steps, n_warmup_steps (60 runs)."""
    base = {**_SHARED, "z_dim": args.best_z_dim, "delta": args.best_delta,
            "channel": "sd", "lambda_comms": args.best_lambda,
            "use_entropy_model": "",
            "entropy_model_K": str(args.best_K),
            "entropy_model_type": args.best_model_type,
            "entropy_model_context": args.best_context,
            "loss_comms_mode": args.best_loss_mode}

    for mult in [1, 5, 10, 50]:
        yield f"p2_lrqphi{mult}x", {**base, "n_warmup_steps": "5000",
                                      "n_qphi_steps": "3",
                                      "lr_qphi_mult": str(float(mult))}
    for n in [1, 3, 5, 10]:
        yield f"p2_nsteps{n}", {**base, "n_warmup_steps": "5000",
                                  "lr_qphi_mult": "10.0",
                                  "n_qphi_steps": str(n)}
    for ws in [0, 1000, 5000, 20000]:
        yield f"p2_warmup{ws}", {**base, "lr_qphi_mult": "10.0",
                                   "n_qphi_steps": "3",
                                   "n_warmup_steps": str(ws)}


def _stage_fix(args) -> Iterator[tuple[str, dict]]:
    """Fix-ladder for P2 gradient failures (25 runs = 5 configs × 5 seeds).

    Motivated by P2-A findings: q_φ poorly fitted (qphi_gap >> 0.54 bits DLM floor),
    entropy backward gradient only 1.5% of speaker gradient, and circular gradient
    attenuation once q_φ converges. Five configs target each fix level in order.

    Run order: B1 → B2 → B3 → B4 always.  B5 (two-phase) can run in parallel.

    Config map (see PILLAR_P2.md §13 for theory):
      fix_baseline   : current P2-A defaults for reference comparison
      fix_B1         : Level 1 — conditional backward gate (qphi_bwd_gate_threshold=2.0)
                       + better q_φ training (n_warmup=50000, n_qphi_steps=20)
      fix_B2         : Level 2 — same as B1 but λ=1e-2 (scale fix)
      fix_B3         : Level 3 — same as B2 + EMA prior (ema_prior_momentum=0.95)
      fix_B4         : Level 5 — two-phase training (phase1_sr_threshold=0.995,
                       λ=1e-2, n_warmup=50000, n_qphi_steps=20, gate enabled)
    """
    base = {
        **_SHARED,
        "delta": args.phase2_delta,
        "z_dim": args.phase2_z_dim,
        "use_entropy_model": "",
        "entropy_model_K": str(args.best_K),
        "entropy_model_type": args.best_model_type,
        "entropy_model_context": "A",
        "loss_comms_mode": "entropy",
    }
    # Reference config: P2-A defaults (no fixes applied)
    yield "fix_baseline", {**base,
                            "lambda_comms": args.phase2_lambda,
                            "n_warmup_steps": "5000",
                            "n_qphi_steps": "3",
                            "lr_qphi_mult": "10.0"}

    # Better q_φ training params shared across B1–B4
    _better_qphi = {"n_warmup_steps": "50000", "n_qphi_steps": "20", "lr_qphi_mult": "30.0"}

    # B1: Level 1 — backward gate + better q_φ training, λ unchanged
    yield "fix_B1_gate", {**base, **_better_qphi,
                           "lambda_comms": args.phase2_lambda,
                           "qphi_bwd_gate_threshold": "2.0"}

    # B2: Level 1+2 — gate + better q_φ + higher λ
    yield "fix_B2_gate_lambda", {**base, **_better_qphi,
                                  "lambda_comms": "1e-2",
                                  "qphi_bwd_gate_threshold": "2.0"}

    # B3: Level 1+2+3 — gate + better q_φ + higher λ + EMA prior
    yield "fix_B3_ema", {**base, **_better_qphi,
                          "lambda_comms": "1e-2",
                          "qphi_bwd_gate_threshold": "2.0",
                          "use_ema_prior": "",
                          "ema_prior_momentum": "0.95"}

    # B4: Level 1+2+5 — gate + better q_φ + higher λ + two-phase training
    # Phase 1: RL + magnitude until SR >= 0.995, then Phase 2: entropy only
    yield "fix_B4_twophase", {**base, **_better_qphi,
                               "lambda_comms": "1e-2",
                               "qphi_bwd_gate_threshold": "2.0",
                               "loss_comms_mode": "magnitude",  # Phase 1 uses magnitude
                               "phase1_sr_threshold": "0.995"}


_STAGE_FNS = {
    "P2-A": _stage_a,
    "P2-B": _stage_b,
    "P2-C": _stage_c,
    "P2-D": _stage_d,
    "P2-E": _stage_e,
    "P2-F": _stage_f,
    "P2-FIX": _stage_fix,
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="P2 systematic ablation runner. See PILLAR_P2.md §10 for stage ordering."
    )
    p.add_argument("--stage", type=str,
                   choices=list(_STAGE_FNS.keys()) + ["all"],
                   required=True, help="Which stage to run.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--log_dir", type=str, default="runs/p2_ablation")
    p.add_argument("--dry_run", action="store_true",
                   help="Print run list without executing.")
    # Phase 2 winners (defaults are sensible placeholders; update after Phase 2)
    p.add_argument("--phase2_lambda", type=str, default="1e-3")
    p.add_argument("--phase2_delta", type=str, default="1.0")
    p.add_argument("--phase2_z_dim", type=str, default="3")
    # P2-A winners (required for stages B-F)
    p.add_argument("--best_K", type=int, default=5)
    p.add_argument("--best_model_type", type=str, default="factored",
                   choices=["factored", "joint"])
    p.add_argument("--best_context", type=str, default="A", choices=["A", "B"])
    p.add_argument("--best_loss_mode", type=str, default="entropy",
                   choices=["entropy", "both"])
    # P2-B winners (required for stages C-F)
    p.add_argument("--best_lambda", type=str, default="1e-3")
    # P2-C winners (required for stages D-F)
    p.add_argument("--best_z_dim", type=str, default="3")
    # P2-D winners (required for stages E-F)
    p.add_argument("--best_delta", type=str, default="1.0")
    args = p.parse_args()

    stages = list(_STAGE_FNS.keys()) if args.stage == "all" else [args.stage]
    for stage in stages:
        conditions = list(_STAGE_FNS[stage](args))
        n_runs = len(conditions) * len(args.seeds)
        print(f"\n=== {stage}: {len(conditions)} conditions × {len(args.seeds)} seeds = {n_runs} runs ===")
        for i, (exp_name, extra) in enumerate(conditions):
            for seed in args.seeds:
                if args.dry_run:
                    print(f"  [{i}] {exp_name} seed={seed}")
                else:
                    _run(exp_name, seed, args.log_dir, extra)
    print("\nDone.")


if __name__ == "__main__":
    main()
