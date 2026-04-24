"""Eval-only rollouts from saved checkpoints — count total bits per trajectory.

Re-instantiates env + trainer identically to `train.py`, loads `final.pt`, and
runs the policy stochastically for a fixed number of completed episodes. For
each episode records (goal_x, goal_y, trajectory_length, trajectory_bits,
success).

Per-step bits:
  • DDCL channels (sd, nsd): `channel.comms_loss(z).sum(-1)` — Jensen bound.
  • IdentityChannel (none, Exp D entropic): `32 * z_dim` — float32 uncompressed
    cost, since comms_loss returns zero for identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from onpolicy.envs.toyproblem.CommunicatingGoal_vec_env import CommunicatingGoalVecEnv
from onpolicy.envs.toyproblem.trainer import MAPPOConfig, MAPPOTrainer


def _args_from_ckpt(ckpt_args: dict) -> argparse.Namespace:
    ns = argparse.Namespace(**ckpt_args)
    if getattr(ns, "goals", None) is not None:
        ns._parsed_goals = np.array(json.loads(ns.goals), dtype=int)
    if getattr(ns, "goal_probs", None) is not None:
        if ns.goal_probs == "zipf":
            n = len(ns._parsed_goals)
            p = 1.0 / np.arange(1, n + 1)
            ns._parsed_goal_probs = p / p.sum()
        else:
            ns._parsed_goal_probs = np.array(json.loads(ns.goal_probs), dtype=np.float64)
    return ns


def _config_from_args(ns: argparse.Namespace) -> MAPPOConfig:
    return MAPPOConfig(
        z_dim=ns.z_dim,
        lr=ns.lr,
        clip_eps=ns.clip_eps,
        entropy_coef=ns.entropy_coef,
        max_grad_norm=ns.max_grad_norm,
        update_epochs=ns.update_epochs,
        num_minibatches=ns.num_minibatches,
        channel=ns.channel,
        delta=ns.delta,
        lambda_comms=ns.lambda_comms,
        use_entropic_prior=getattr(ns, "use_entropic_prior", False),
        gmm_structure=getattr(ns, "gmm_structure", "joint"),
        num_gmm_components=getattr(ns, "num_gmm_components", 8),
        gmm_init_spread=getattr(ns, "gmm_init_spread", 1.0),
        gmm_lr=getattr(ns, "gmm_lr", 3e-4),
        beta_target=getattr(ns, "beta_target", 1e-2),
        beta_warmup=getattr(ns, "beta_warmup", 100_000),
        beta_anneal=getattr(ns, "beta_anneal", 300_000),
        gmm_tau=getattr(ns, "gmm_tau", 0.0),
    )


def _bits_per_step_fn(channel_name: str, z_dim: int):
    if channel_name == "none":
        const = float(32 * z_dim)
        def f(_chan, z):
            return torch.full((z.shape[0],), const, dtype=torch.float32, device=z.device)
        return f
    def f(chan, z):
        return chan.comms_loss(z).sum(-1)
    return f


def rollout_trajectory_bits(
    run_dir: Path,
    num_episodes: int = 500,
    env_seed: int = 0,
    device: str = "cpu",
    n_envs: int = 16,
    arg_overrides: dict | None = None,
) -> np.ndarray:
    """Return (num_episodes, 5): [goal_x, goal_y, traj_len, traj_bits, success].

    `arg_overrides` patches the saved args before env/trainer construction.
    Needed for Exp C checkpoints, which predate max_steps/grid_size/goals being
    saved in the ckpt — the expC driver injects them from sweep.py.
    """
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    merged = dict(ckpt["args"])
    if arg_overrides:
        merged.update(arg_overrides)
    ns = _args_from_ckpt(merged)
    config = _config_from_args(ns)

    env = CommunicatingGoalVecEnv(num_envs=n_envs, args=ns)
    env.seed(env_seed)

    trainer = MAPPOTrainer(config, device=torch.device(device))
    trainer.load_state_dict(ckpt["state_dict"])
    trainer.eval()

    step_bits_fn = _bits_per_step_fn(ns.channel, ns.z_dim)

    obs = env.reset()
    inflight_steps = np.zeros(n_envs, dtype=np.int64)
    inflight_bits = np.zeros(n_envs, dtype=np.float64)

    records: list[tuple[int, int, int, float, int]] = []
    max_steps = getattr(ns, "max_steps", 50) or 50
    max_iters = max(1000, int(np.ceil(num_episodes * max_steps / n_envs)) * 2)

    with torch.no_grad():
        for _ in range(max_iters):
            if len(records) >= num_episodes:
                break

            goal_np, lp_np = obs
            goal_t = torch.from_numpy(goal_np).to(device)
            lp_t = torch.from_numpy(lp_np).to(device)

            z = trainer.speaker(goal_t)
            z_hat, _ = trainer.channel(z)
            dist = trainer.listener(torch.cat([lp_t, z_hat], dim=-1))
            action = dist.sample()

            step_bits = step_bits_fn(trainer.channel, z).cpu().numpy().astype(np.float64)
            inflight_bits += step_bits
            inflight_steps += 1

            next_obs, _reward, _done, info = env.step(action.cpu().numpy())
            done_arr = info["done"]
            success_arr = info["success"]

            if done_arr.any():
                for n in np.nonzero(done_arr)[0]:
                    if len(records) >= num_episodes:
                        break
                    records.append((
                        int(goal_np[n, 0]),
                        int(goal_np[n, 1]),
                        int(inflight_steps[n]),
                        float(inflight_bits[n]),
                        int(success_arr[n]),
                    ))
                    inflight_steps[n] = 0
                    inflight_bits[n] = 0.0

            obs = next_obs

    if len(records) < num_episodes:
        raise RuntimeError(
            f"Only collected {len(records)}/{num_episodes} episodes in {max_iters} iters "
            f"for {run_dir}"
        )

    return np.asarray(records, dtype=np.float64)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Roll out a saved toyproblem checkpoint.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--env_seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--out", type=Path, default=None,
                   help="CSV path; defaults to run_dir/trajectory_bits.csv")
    return p.parse_args()


def _write_csv(records: np.ndarray, out_path: Path) -> None:
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_idx", "goal_x", "goal_y", "trajectory_length",
                    "trajectory_bits", "success"])
        for i, row in enumerate(records):
            w.writerow([i, int(row[0]), int(row[1]), int(row[2]),
                        float(row[3]), int(row[4])])


def main() -> None:
    args = _parse_cli()
    records = rollout_trajectory_bits(
        args.run_dir, args.num_episodes, args.env_seed, args.device, args.n_envs
    )
    out = args.out or (args.run_dir / "trajectory_bits.csv")
    _write_csv(records, out)
    succ = records[:, 4].mean()
    mean_bits = records[:, 3].mean()
    mean_len = records[:, 2].mean()
    print(f"[rollout] {args.run_dir.name}: {len(records)} eps, "
          f"win_rate={succ:.3f}, mean_bits={mean_bits:.2f}, mean_len={mean_len:.1f} -> {out}")


if __name__ == "__main__":
    main()
