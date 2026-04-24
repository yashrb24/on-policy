"""Systematic channel comparison experiment.

Trains all channel types with identical hyperparameters across multiple seeds
and generates a comparison report. This is the primary tool for validating
DDCL vs baselines before running expensive full sweeps.

Channels compared
-----------------
  none             : float passthrough (upper bound on task performance)
  sd               : DDCL subtractive dither (our main method)
  nsd              : DDCL non-subtractive TPDF dither
  additive_uniform : 1st-order Schuchman dither (intermediate control)
  gaussian         : Gaussian dither (negative control — should be worst)
  ste4             : Fixed-rate STE 4-bit quantizer
  ste8             : Fixed-rate STE 8-bit quantizer
  ste16            : Fixed-rate STE 16-bit quantizer

For each channel × seed, runs train.py and saves results to:
    runs/toyproblem/channel_comparison/<channel>_lam<lambda>_d<delta>/<seed>/

Usage
-----
Quick validation (2 seeds, 200K steps):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
        python -m onpolicy.envs.toyproblem.experiments.run_channel_comparison \
        --seeds 0 1 --total_timesteps 200000 --quick

Full comparison (5 seeds, 1M steps):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
        python -m onpolicy.envs.toyproblem.experiments.run_channel_comparison \
        --seeds 0 1 2 3 4

After runs complete, generate the report:
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
        python -m onpolicy.envs.toyproblem.analysis.report_baseline \
        --sweep_dir runs/toyproblem/channel_comparison \
        --out_dir results/toyproblem/channel_comparison
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

# Channels to compare with their specific arguments.
# For DDCL channels we set a meaningful lambda_comms; for baselines we use 0
# (baselines don't optimize communication cost — they have it baked in).
CHANNEL_CONFIGS = [
    # (channel_name, lambda_comms, delta, ste_clip)
    ("none",             0.0,    1.0,  10.0),  # float passthrough
    ("sd",               4e-3,  10.0,  10.0),  # DDCL-SD (default λ from baseline_v0)
    ("nsd",              4e-3,  10.0,  10.0),  # DDCL-NSD
    ("additive_uniform", 4e-3,  10.0,  10.0),  # 1st-order Schuchman control
    ("gaussian",         4e-3,  10.0,  10.0),  # Gaussian control (biased)
    ("ste4",             0.0,    1.0,  10.0),  # Fixed-rate 4-bit
    ("ste8",             0.0,    1.0,  10.0),  # Fixed-rate 8-bit
    ("ste16",            0.0,    1.0,  10.0),  # Fixed-rate 16-bit
]


def _exp_name(channel: str, lam: float, delta: float) -> str:
    if channel.startswith("ste"):
        return f"{channel}"
    return f"{channel}_lam{lam:.0e}_d{int(delta)}"


def build_command(
    channel: str,
    lam: float,
    delta: float,
    ste_clip: float,
    seed: int,
    total_timesteps: int,
    log_dir: str,
    hidden_size: int = 64,
    z_dim: int = 3,
    n_envs: int = 16,
    n_steps: int = 256,
    lr: float = 3e-4,
    log_every: int = 10,
) -> list[str]:
    exp_name = _exp_name(channel, lam, delta)
    return [
        sys.executable, "-m", "onpolicy.envs.toyproblem.train",
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--channel", channel,
        "--lambda_comms", str(lam),
        "--delta", str(delta),
        "--ste_clip", str(ste_clip),
        "--hidden_size", str(hidden_size),
        "--z_dim", str(z_dim),
        "--n_envs", str(n_envs),
        "--n_steps", str(n_steps),
        "--total_timesteps", str(total_timesteps),
        "--lr", str(lr),
        "--log_dir", log_dir,
        "--log_every", str(log_every),
    ]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run systematic channel comparison experiment."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Seeds to run for each channel.")
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--log_dir", type=str, default="runs/toyproblem/channel_comparison",
                   help="Root directory for run outputs.")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=3)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 2 seeds, 200K steps, fewer envs.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing them.")
    p.add_argument("--channels_subset", type=str, nargs="*", default=None,
                   help="Restrict to a subset of channels (e.g. sd nsd none).")
    args = p.parse_args()

    if args.quick:
        args.seeds = args.seeds[:2]
        args.total_timesteps = 200_000
        args.n_envs = 8

    configs = CHANNEL_CONFIGS
    if args.channels_subset:
        configs = [c for c in configs if c[0] in args.channels_subset]

    total_runs = len(configs) * len(args.seeds)
    print(f"Channel comparison: {len(configs)} channels × {len(args.seeds)} seeds = {total_runs} runs")
    print(f"Total timesteps per run: {args.total_timesteps:,}")
    print(f"Output: {args.log_dir}/")
    print()

    failed = []
    for i, (channel, lam, delta, ste_clip) in enumerate(configs):
        for seed in args.seeds:
            exp_name = _exp_name(channel, lam, delta)
            cmd = build_command(
                channel=channel, lam=lam, delta=delta, ste_clip=ste_clip,
                seed=seed, total_timesteps=args.total_timesteps,
                log_dir=args.log_dir, hidden_size=args.hidden_size,
                z_dim=args.z_dim, n_envs=args.n_envs, n_steps=args.n_steps,
                lr=args.lr,
            )

            run_label = f"[{exp_name}/seed={seed}]"

            # Skip if output already exists.
            out_path = Path(args.log_dir) / exp_name / str(seed) / "metrics.csv"
            if out_path.exists():
                print(f"  SKIP  {run_label}  (already done)")
                continue

            print(f"  RUN   {run_label}  {'(dry)' if args.dry_run else ''}")
            if args.dry_run:
                print("    " + " ".join(cmd))
                continue

            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"  FAIL  {run_label}")
                failed.append(run_label)

    print()
    if args.dry_run:
        print("Dry run complete. Remove --dry_run to execute.")
    elif failed:
        print(f"FAILED runs ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
        raise SystemExit(1)
    else:
        print(f"All {total_runs} runs complete.")
        print()
        print("To generate comparison report:")
        print(
            f"  python -m onpolicy.envs.toyproblem.analysis.report_baseline "
            f"--sweep_dir {args.log_dir} --out_dir results/toyproblem/channel_comparison"
        )


if __name__ == "__main__":
    main()
