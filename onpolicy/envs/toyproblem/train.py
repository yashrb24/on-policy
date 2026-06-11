from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from onpolicy.envs.toyproblem.buffer import RolloutBuffer
from onpolicy.envs.toyproblem.CommunicatingGoal_vec_env import CommunicatingGoalVecEnv
from onpolicy.envs.toyproblem.trainer import MAPPOConfig, MAPPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=256)
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--update_epochs", type=int, default=10)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--entropy_coef", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--z_dim", type=int, default=3)
    p.add_argument("--channel", type=str, default="none", choices=["none", "sd", "tpdf", "async_sd", "ad"])
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--delta_learnable", action="store_true")
    p.add_argument("--delta_global_learnable", action="store_true")
    p.add_argument("--lambda_comms", type=float, default=0.0)
    p.add_argument("--use_entropic_prior", action="store_true")
    p.add_argument("--gmm_structure", type=str, default="joint",
                   choices=["joint", "independent"])
    p.add_argument("--num_gmm_components", type=int, default=8)
    p.add_argument("--gmm_init_spread", type=float, default=1.0)
    p.add_argument("--gmm_lr", type=float, default=3e-4)
    p.add_argument("--beta_target", type=float, default=1e-2)
    p.add_argument("--beta_warmup", type=int, default=100_000)
    p.add_argument("--beta_anneal", type=int, default=300_000)
    p.add_argument("--gmm_tau", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log_dir", type=str, default="runs/toyproblem")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--grid_size", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--goals", type=str, default=None,
                   help="JSON array of goal coordinates, e.g. '[[0,0],[7,7]]'")
    p.add_argument("--goal_probs", type=str, default=None,
                   help="JSON array of probabilities, or 'zipf' for 1/k distribution")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    if args.goals is not None:
        args._parsed_goals = np.array(json.loads(args.goals), dtype=int)
    if args.goal_probs is not None:
        if args.goal_probs == "zipf":
            n = len(args._parsed_goals)
            p = 1.0 / np.arange(1, n + 1)
            args._parsed_goal_probs = p / p.sum()
        else:
            args._parsed_goal_probs = np.array(json.loads(args.goal_probs), dtype=np.float64)

    env = CommunicatingGoalVecEnv(num_envs=args.n_envs, args=args)
    env.seed(args.seed)

    config = MAPPOConfig(
        z_dim=args.z_dim,
        lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        channel=args.channel,
        delta=args.delta,
        delta_learnable=args.delta_learnable,
        delta_global_learnable=args.delta_global_learnable,
        lambda_comms=args.lambda_comms,
        use_entropic_prior=args.use_entropic_prior,
        gmm_structure=args.gmm_structure,
        num_gmm_components=args.num_gmm_components,
        gmm_init_spread=args.gmm_init_spread,
        gmm_lr=args.gmm_lr,
        beta_target=args.beta_target,
        beta_warmup=args.beta_warmup,
        beta_anneal=args.beta_anneal,
        gmm_tau=args.gmm_tau,
    )
    trainer = MAPPOTrainer(config, device=device)
    buffer = RolloutBuffer(args.n_steps, args.n_envs, args.z_dim, device=device)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "update", "timestep", "mean_reward", "success_rate",
        "pg_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
        "comms_loss", "bits_per_msg", "z_norm", "sps",
        "prior_nll", "gmm_entropy", "beta",
    ])

    diag_path = log_dir / "diagnostics.csv"
    diag_file = open(diag_path, "w", newline="")
    diag_writer = csv.writer(diag_file)
    diag_writer.writerow([
        "grad_step", "update", "timestep",
        "pg_loss", "actor_loss", "value_loss", "entropy", "comms_loss",
        "grad_norm_speaker", "grad_norm_listener", "grad_norm_critic",
        "grad_norm_channel", "grad_norm_total",
    ])
    grad_step = 0

    n_updates = args.total_timesteps // (args.n_envs * args.n_steps)

    obs = env.reset()  # [goal (N,2), lp (N,2)] numpy float32
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
            reward_shared = reward[:, 0, 0]  # (N,) float32 — both agents share
            done_shared = done[:, 0]  # (N,) bool

            buffer.insert(
                goal,
                lp,
                action,
                log_prob,
                value,
                torch.from_numpy(reward_shared).to(device),
                torch.from_numpy(done_shared.astype(np.float32)).to(device),
            )

            if done_shared.any():
                idx = np.nonzero(done_shared)[0]
                recent_rewards.extend(info["final_episode_reward"][idx].tolist())
                recent_successes.extend(info["success"][idx].tolist())

            obs = next_obs

        goal_np, lp_np = obs
        goal = torch.from_numpy(goal_np).to(device)
        lp = torch.from_numpy(lp_np).to(device)
        last_value = trainer.get_value(goal, lp)  # (N, 1) normalized

        buffer.compute_returns_and_advantages(
            last_value, trainer.value_norm, args.gamma, args.gae_lambda
        )

        timestep = (update + 1) * args.n_envs * args.n_steps
        metrics, step_diagnostics = trainer.update(buffer, timestep=timestep)

        mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        success_rate = float(np.mean(recent_successes)) if recent_successes else 0.0
        sps = timestep / (time.time() - start_time)

        csv_writer.writerow([
            update, timestep, mean_reward, success_rate,
            metrics["pg_loss"], metrics["value_loss"], metrics["entropy"],
            metrics["approx_kl"], metrics["clip_frac"],
            metrics["comms_loss"], metrics["bits_per_msg"], metrics["z_norm"], sps,
            metrics.get("prior_nll", 0.0), metrics.get("gmm_entropy", 0.0),
            metrics.get("beta", 0.0),
        ])
        csv_file.flush()

        for d in step_diagnostics:
            diag_writer.writerow([
                grad_step, update, timestep,
                d["pg_loss"], d["actor_loss"], d["value_loss"], d["entropy"], d["comms_loss"],
                d["grad_norm_speaker"], d["grad_norm_listener"], d["grad_norm_critic"],
                d["grad_norm_channel"], d["grad_norm_total"],
            ])
            grad_step += 1
        diag_file.flush()

        if update % args.log_every == 0 or update == n_updates - 1:
            print(
                f"[{update:4d}/{n_updates}] t={timestep:>8d} "
                f"reward={mean_reward:+.3f} success={success_rate:.2f} "
                f"pg={metrics['pg_loss']:+.4f} v={metrics['value_loss']:.4f} "
                f"H={metrics['entropy']:.3f} kl={metrics['approx_kl']:+.4f} "
                f"clip={metrics['clip_frac']:.2f} bits={metrics['bits_per_msg']:.2f} "
                f"sps={sps:.0f}"
            )

    csv_file.close()
    diag_file.close()

    # Post-training per-goal evaluation: one speaker pass per goal, no sampling.
    # Writes: goal_x, goal_y, goal_prob, z_norm, bits_channel (sd/tpdf/async_sd only),
    #         bits_prior (entropic only; = -log2 p(z) under GMM).
    goals_np = env.goals.astype(np.float32)
    probs_np = env.goal_probs.astype(np.float32)
    goals_t = torch.from_numpy(goals_np).to(device)
    with torch.no_grad():
        z_eval = trainer.speaker(goals_t)                          # (G, z_dim)
        bits_channel = trainer.channel.comms_loss(z_eval).sum(-1)  # (G,)
        z_norm_g = z_eval.norm(dim=-1)                             # (G,)
        bits_prior = torch.zeros(len(goals_np), device=device)
        if trainer.gmm_prior is not None:
            bits_prior = -trainer.gmm_prior.log_prob(z_eval) / math.log(2)

    per_goal_path = log_dir / "per_goal_bits.csv"
    with open(per_goal_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["goal_idx", "goal_x", "goal_y", "goal_prob",
                    "z_norm", "bits_channel", "bits_prior"])
        for i in range(len(goals_np)):
            w.writerow([
                i, int(goals_np[i, 0]), int(goals_np[i, 1]),
                float(probs_np[i]),
                float(z_norm_g[i].item()),
                float(bits_channel[i].item()),
                float(bits_prior[i].item()),
            ])

    ckpt_path = log_dir / "final.pt"
    torch.save({"state_dict": trainer.state_dict(), "args": vars(args)}, ckpt_path)

    print(f"Done. Logs at {csv_path}, ckpt at {ckpt_path}")


if __name__ == "__main__":
    main()
