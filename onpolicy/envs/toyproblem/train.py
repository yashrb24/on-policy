from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from onpolicy.envs.toyproblem.buffer import RolloutBuffer
from onpolicy.envs.toyproblem.CommunicatingGoal_env import _DEFAULT_GOALS
from onpolicy.envs.toyproblem.CommunicatingGoal_vec_env import CommunicatingGoalVecEnv
from onpolicy.envs.toyproblem.trainer import MAPPOConfig, MAPPOTrainer

# Pre-compute lookup: position (x,y) -> goal index for the default 6 goals.
_GOAL_POS_TO_IDX: dict[tuple[int, int], int] = {
    (int(g[0]), int(g[1])): i for i, g in enumerate(_DEFAULT_GOALS)
}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int, device: torch.device) -> None:
    """Set all RNGs deterministically.

    Covers: Python random, NumPy global, PyTorch CPU, PyTorch CUDA (if used),
    and the PYTHONHASHSEED environment variable. Also enables deterministic
    CUDA ops where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DDCL toy problem (MAPPO on CommunicatingGoalEnv)."
    )
    # Experiment identity
    p.add_argument("--exp_name", type=str, default="debug",
                   help="Experiment name; determines runs/<exp_name>/<seed>/ log path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")

    # Rollout / training schedule
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=256)
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)

    # PPO hyperparameters
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--update_epochs", type=int, default=10)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--entropy_coef", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # Network architecture
    p.add_argument("--hidden_size", type=int, default=64,
                   help="Hidden units in speaker and listener MLPs.")
    p.add_argument("--z_dim", type=int, default=3,
                   help="Speaker output / message dimension. Use 1 for paper baseline.")

    # Communication channel
    p.add_argument("--channel", type=str, default="none",
                   choices=["none", "sd", "nsd", "additive_uniform",
                            "gaussian", "ste4", "ste8", "ste16"],
                   help="Quantisation channel. DDCL: sd, nsd. "
                        "Baselines: additive_uniform, gaussian, ste4/8/16.")
    p.add_argument("--delta", type=float, default=1.0,
                   help="Quantisation bin width δ (SD/NSD/additive_uniform/gaussian).")
    p.add_argument("--lambda_comms", type=float, default=0.0,
                   help="Communication loss weight λ.")
    p.add_argument("--ste_clip", type=float, default=10.0,
                   help="Clip bound for STE channels (ste4/ste8/ste16).")

    # Logging
    p.add_argument("--log_dir", type=str, default="runs",
                   help="Root log directory. Actual path: <log_dir>/<exp_name>/<seed>/")
    p.add_argument("--log_every", type=int, default=10,
                   help="Console print frequency (in updates).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception:
        return "unavailable"


def _setup_log_dir(args: argparse.Namespace) -> Path:
    run_dir = Path(args.log_dir) / args.exp_name / str(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot — everything needed to reproduce this run.
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Git SHA — identifies the exact code version.
    sha_path = run_dir / "git_sha.txt"
    sha_path.write_text(_git_sha())

    return run_dir


_N_GOALS = len(_DEFAULT_GOALS)  # 6

CSV_HEADER = [
    "update", "timestep", "mean_reward", "success_rate",
    "pg_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
    "comms_loss", "bits_per_msg", "true_bits_per_msg", "z_norm", "sps",
] + [f"bits_goal_{i}" for i in range(_N_GOALS)]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    set_seed(args.seed, device)

    # Logging setup (before any randomness is consumed by the env).
    run_dir = _setup_log_dir(args)
    csv_path = run_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)

    # Environment.
    env = CommunicatingGoalVecEnv(num_envs=args.n_envs)
    env.seed(args.seed)

    # Trainer.
    config = MAPPOConfig(
        z_dim=args.z_dim,
        hidden_size=args.hidden_size,
        lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        channel=args.channel,
        delta=args.delta,
        lambda_comms=args.lambda_comms,
        ste_clip=args.ste_clip,
    )
    trainer = MAPPOTrainer(config, device=device)
    buffer = RolloutBuffer(args.n_steps, args.n_envs, args.z_dim, device=device)

    n_updates = args.total_timesteps // (args.n_envs * args.n_steps)

    obs = env.reset()
    recent_rewards: deque[float] = deque(maxlen=200)
    recent_successes: deque[int] = deque(maxlen=200)

    start_time = time.time()
    for update in range(n_updates):
        buffer.reset()

        for _ in range(args.n_steps):
            goal_np, lp_np = obs
            goal = torch.from_numpy(goal_np).to(device)
            lp = torch.from_numpy(lp_np).to(device)

            action, log_prob, value = trainer.act_and_value(goal, lp)

            next_obs, reward, done, info = env.step(action.cpu().numpy())
            reward_shared = reward[:, 0, 0]
            done_shared = done[:, 0]

            # Map goal positions → goal indices for per-goal bit-allocation logging.
            goal_ids_np = np.array(
                [_GOAL_POS_TO_IDX.get((int(g[0]), int(g[1])), 0) for g in goal_np],
                dtype=np.int64,
            )

            buffer.insert(
                goal, lp, action, log_prob, value,
                torch.from_numpy(reward_shared).to(device),
                torch.from_numpy(done_shared.astype(np.float32)).to(device),
                goal_id=torch.from_numpy(goal_ids_np).to(device),
            )

            if done_shared.any():
                idx = np.nonzero(done_shared)[0]
                recent_rewards.extend(info["final_episode_reward"][idx].tolist())
                recent_successes.extend(info["success"][idx].tolist())

            obs = next_obs

        goal_np, lp_np = obs
        goal = torch.from_numpy(goal_np).to(device)
        lp = torch.from_numpy(lp_np).to(device)
        last_value = trainer.get_value(goal, lp)

        buffer.compute_returns_and_advantages(
            last_value, trainer.value_norm, args.gamma, args.gae_lambda
        )

        metrics = trainer.update(buffer)

        timestep = (update + 1) * args.n_envs * args.n_steps
        mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        success_rate = float(np.mean(recent_successes)) if recent_successes else 0.0
        sps = timestep / (time.time() - start_time)

        per_goal_bits = [
            metrics.get(f"bits_goal_{i}", float("nan")) for i in range(_N_GOALS)
        ]
        csv_writer.writerow([
            update, timestep, mean_reward, success_rate,
            metrics["pg_loss"], metrics["value_loss"], metrics["entropy"],
            metrics["approx_kl"], metrics["clip_frac"],
            metrics["comms_loss"], metrics["bits_per_msg"],
            metrics["true_bits_per_msg"], metrics["z_norm"], sps,
        ] + per_goal_bits)
        csv_file.flush()

        if update % args.log_every == 0 or update == n_updates - 1:
            print(
                f"[{update:4d}/{n_updates}] t={timestep:>8d} "
                f"reward={mean_reward:+.3f} success={success_rate:.2f} "
                f"pg={metrics['pg_loss']:+.4f} v={metrics['value_loss']:.4f} "
                f"H={metrics['entropy']:.3f} kl={metrics['approx_kl']:+.4f} "
                f"clip={metrics['clip_frac']:.2f} "
                f"bits_surr={metrics['bits_per_msg']:.2f} "
                f"bits_true={metrics['true_bits_per_msg']:.2f} "
                f"sps={sps:.0f}"
            )

    csv_file.close()

    ckpt_path = run_dir / "final.pt"
    torch.save({"state_dict": trainer.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Done. Logs: {run_dir}/  metrics: {csv_path}  ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
