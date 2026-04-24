"""Offline sweep runner — no WandB required.

Reads a sweep YAML config (same format as WandB sweep configs) and runs
all grid combinations locally, sequentially or in parallel.

Usage
-----
# Dry run (print commands only):
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.scripts.sweeps.toyproblem.run_sweep \
    --config onpolicy/scripts/sweeps/toyproblem/configs/sweep_stage_a.yaml \
    --seeds 0 1 2 3 4 --dry_run

# Run all grid combinations (sequential):
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.scripts.sweeps.toyproblem.run_sweep \
    --config onpolicy/scripts/sweeps/toyproblem/configs/sweep_stage_a.yaml \
    --seeds 0 1 2 3 4 --log_dir runs/sweep_stage_a

Note: grid sweeps can be large (Stage A has 3×8×6×3 = 432 configs × 5 seeds
= 2160 runs). On a single machine, run a random subset with --max_runs N.
"""
from __future__ import annotations

import argparse
import itertools
import random
import subprocess
import sys
from pathlib import Path

import yaml


def _load_grid(config_path: str) -> list[dict]:
    """Expand a WandB-style sweep YAML into a list of hyperparameter dicts.

    config_path is resolved relative to the caller's CWD. If not found, also
    tries relative to this script file's location (useful when the module is
    invoked with `python -m` from a different directory).
    """
    p = Path(config_path)
    if not p.exists():
        # Try relative to the directory containing this script.
        alt = Path(__file__).parent / config_path
        if alt.exists():
            p = alt
        else:
            # Try stripping any leading path components up to "configs/".
            parts = Path(config_path).parts
            if "configs" in parts:
                idx = parts.index("configs")
                local = Path(__file__).parent / Path(*parts[idx:])
                if local.exists():
                    p = local
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path!r}\n"
            f"Run from the on-policy/ directory, or pass an absolute path."
        )
    with open(p) as f:
        cfg = yaml.safe_load(f)

    params = cfg.get("parameters", {})
    axes = {}
    for name, spec in params.items():
        if "values" in spec:
            axes[name] = spec["values"]
        elif "value" in spec:
            axes[name] = [spec["value"]]
        else:
            raise ValueError(f"Parameter {name!r} has neither 'value' nor 'values'")

    keys = list(axes.keys())
    combos = list(itertools.product(*[axes[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def _run_key(hparams: dict, seed: int) -> str:
    """Generate a stable string key for a (hparams, seed) combination.

    Only include parameters that actually vary across the sweep grid.
    Fixed hyper-parameters (PPO knobs, network size, etc.) are excluded so
    the key stays short and — critically — unique.  Including them in the key
    caused 80-char truncation to alias many distinct (lambda_comms, z_dim)
    combinations to the same directory name, silently skipping most runs.
    """
    _FIXED = {
        "n_envs", "n_steps", "update_epochs", "max_grad_norm", "gamma",
        "total_timesteps", "clip_eps", "entropy_coef", "gae_lambda",
        "hidden_size", "lr", "num_minibatches", "ste_clip",
    }
    items = sorted(hparams.items())
    parts = "_".join(f"{k}={v}" for k, v in items if k not in _FIXED)
    return f"seed{seed}_{parts}"


def _build_command(hparams: dict, seed: int, log_dir: str, exp_name: str) -> list[str]:
    cmd = [sys.executable, "-m", "onpolicy.envs.toyproblem.train",
           "--exp_name", exp_name, "--seed", str(seed), "--log_dir", log_dir]
    for k, v in hparams.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Offline grid sweep runner.")
    p.add_argument("--config", type=str, required=True,
                   help="Path to sweep YAML config.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--log_dir", type=str, default="runs/sweep",
                   help="Root directory for run outputs.")
    p.add_argument("--max_runs", type=int, default=None,
                   help="Maximum runs to execute (random subset of grid if exceeded).")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--shuffle", action="store_true",
                   help="Randomise order of runs (useful when max_runs is set).")
    p.add_argument("--resume", action="store_true",
                   help="Skip runs whose metrics.csv already exists.")
    args = p.parse_args()

    grid = _load_grid(args.config)
    runs = [(hp, seed) for hp in grid for seed in args.seeds]

    if args.shuffle:
        random.shuffle(runs)

    if args.max_runs is not None:
        runs = runs[:args.max_runs]

    print(f"Sweep: {len(grid)} configs × {len(args.seeds)} seeds = {len(grid)*len(args.seeds)} total")
    print(f"       (note: redundant `none` channel combos are auto-skipped at runtime)")
    print(f"Running: {len(runs)} runs max  (log_dir: {args.log_dir})")
    print()

    failed = []
    for i, (hparams, seed) in enumerate(runs):
        # Skip redundant `none` channel runs (lambda_comms and delta are no-ops for
        # IdentityChannel: comms_loss always returns 0 regardless). Only the canonical
        # combo (lambda=0, delta=1.0) differs meaningfully; all others are identical.
        if hparams.get("channel") == "none":
            lam = float(hparams.get("lambda_comms", 0.0))
            dlt = float(hparams.get("delta", 1.0))
            if lam != 0.0 or dlt != 1.0:
                print(
                    f"  [{i+1}/{len(runs)}] SKIP (none: λ={lam},δ={dlt} are no-ops; "
                    f"only λ=0,δ=1.0 canonical run executed)"
                )
                continue

        key = _run_key(hparams, seed)
        exp_name = key[:80]
        out_path = Path(args.log_dir) / exp_name / str(seed) / "metrics.csv"

        if args.resume and out_path.exists():
            print(f"  [{i+1}/{len(runs)}] SKIP  {exp_name}/seed={seed}")
            continue

        cmd = _build_command(hparams, seed, args.log_dir, exp_name)
        print(f"  [{i+1}/{len(runs)}] RUN   {exp_name}/seed={seed}"
              + ("  (dry)" if args.dry_run else ""))

        if args.dry_run:
            print("    " + " ".join(cmd))
            continue

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  FAIL  {exp_name}/seed={seed}")
            failed.append((exp_name, seed))

    print()
    if args.dry_run:
        print("Dry run complete.")
    elif failed:
        print(f"FAILED ({len(failed)} runs):")
        for name, seed in failed:
            print(f"  {name}/seed={seed}")
        raise SystemExit(1)
    else:
        print(f"All {len(runs)} runs complete.")
        print()
        print("Next: run convergence gate and report:")
        print(f"  python -m onpolicy.envs.toyproblem.analysis.report_baseline "
              f"--sweep_dir {args.log_dir} --out_dir docs/results/sweep")


if __name__ == "__main__":
    main()
