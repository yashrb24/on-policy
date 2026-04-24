"""WandB sweep wrapper for the DDCL toy problem.

Called by the WandB agent for each run. Reads hyperparameters from
wandb.config and calls train.main() directly (avoids subprocess overhead).

Usage (via WandB agent — do not call directly):
    wandb agent <sweep_id>

For offline (no WandB) use, see run_sweep.py instead.
"""
from __future__ import annotations

import sys

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def run() -> None:
    if not HAS_WANDB:
        raise RuntimeError(
            "wandb is not installed. For offline sweeps, use run_sweep.py instead.\n"
            "Install: pip install wandb"
        )

    with wandb.init() as run_obj:
        cfg = wandb.config

        # Build a sys.argv that train.parse_args() can consume.
        argv = [
            "train",
            "--exp_name", f"sweep_{run_obj.id}",
            "--seed", str(cfg.get("seed", 0)),
            "--channel", str(cfg.get("channel", "none")),
            "--lambda_comms", str(cfg.get("lambda_comms", 0.0)),
            "--delta", str(cfg.get("delta", 1.0)),
            "--z_dim", str(cfg.get("z_dim", 3)),
            "--hidden_size", str(cfg.get("hidden_size", 64)),
            "--lr", str(cfg.get("lr", 3e-4)),
            "--n_envs", str(cfg.get("n_envs", 16)),
            "--n_steps", str(cfg.get("n_steps", 256)),
            "--total_timesteps", str(cfg.get("total_timesteps", 1_000_000)),
            "--clip_eps", str(cfg.get("clip_eps", 0.2)),
            "--update_epochs", str(cfg.get("update_epochs", 10)),
            "--num_minibatches", str(cfg.get("num_minibatches", 4)),
            "--entropy_coef", str(cfg.get("entropy_coef", 0.03)),
            "--max_grad_norm", str(cfg.get("max_grad_norm", 0.5)),
            "--gamma", str(cfg.get("gamma", 0.99)),
            "--gae_lambda", str(cfg.get("gae_lambda", 0.95)),
            "--log_dir", "runs/wandb_sweep",
        ]

        old_argv = sys.argv
        sys.argv = argv
        try:
            from onpolicy.envs.toyproblem.train import main
            main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    run()
