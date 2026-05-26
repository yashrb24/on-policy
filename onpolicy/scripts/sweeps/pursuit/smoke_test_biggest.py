#!/usr/bin/env python3
"""Smoke test for the largest config in sweep_config_transformer_ac_vanilla_mappo.yaml.

Runs the biggest grid point (n_block=3, n_embd=64, n_head=4, ppo_epoch=8) at the
fixed num_mini_batch=2 for ~2 PPO updates, with wandb disabled. If this completes
on this GPU without CUDA OOM, every other point in the sweep grid will also fit
(they are all strictly smaller).

Verified on RTX A2000 12GB (clean GPU, no other CUDA processes):
    torch peak allocated = 7158 MiB
    torch peak reserved  = 8428 MiB
    nvidia-smi process   = 8562 MiB    (~3.3 GiB headroom out of 11.9 GiB free)

If you're hitting OOMs during the real sweep despite this smoke passing, the cause
is almost certainly leaked zombie processes from prior failed runs — every time a
run crashes inside train_pursuit.main() (e.g. CUDA OOM), the SubprocVecEnv worker
processes and the wandb run are not cleaned up, and they keep holding GPU memory.
Check with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before
launching the agent. Run `pkill -9 -f train_pursuit` to clear stragglers.

Usage:
    python smoke_test_biggest.py
"""

import sys
import time
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
TRAIN_PATH = REPO_ROOT / "onpolicy" / "scripts" / "train"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRAIN_PATH))


# Biggest config from the sweep grid. num_env_steps trimmed to ~2 PPO updates:
# 1 update = n_rollout_threads * episode_length = 48 * 500 = 24,000 env steps.
SMOKE_ARGS = [
    "--env_name", "Pursuit",
    "--algorithm_name", "rmappo",
    "--experiment_name", "smoke_biggest",
    "--seed", "8",
    "--num_env_steps", "50000",
    "--episode_length", "500",
    "--max_cycles", "500",
    "--n_pursuers", "20",
    "--n_evaders", "8",
    "--x_size", "40",
    "--y_size", "40",
    "--n_rollout_threads", "48",
    "--n_eval_rollout_threads", "4",
    "--save_interval", "999999",
    "--log_interval", "24000",
    "--eval_interval", "999999",
    "--eval_episodes", "32",
    "--use_transformer_base_actor",
    "--use_transformer_base_critic",
    "--hidden_size", "64",
    "--lr", "0.0003",
    "--critic_lr", "0.0003",
    "--ppo_epoch", "8",
    "--clip_param", "0.2",
    "--num_mini_batch", "2",
    "--entropy_coef", "0.01",
    "--value_loss_coef", "0.5",
    "--max_grad_norm", "1.0",
    "--data_chunk_length", "50",
    "--gamma", "0.99",
    "--gae_lambda", "0.95",
    "--n_block", "3",
    "--n_embd", "64",
    "--n_head", "4",
    "--user_name", "yashrb",
    "--wandb_name", "smoke",
    "--use_wandb",  # action='store_false' in config.py → passing this DISABLES wandb
]


def _report_peak_vram(label):
    if torch.cuda.is_available():
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_mib = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"[smoke] {label}: peak allocated={peak_mib:.0f} MiB, "
              f"peak reserved={reserved_mib:.0f} MiB")


def main():
    from train_pursuit import main as train_main

    print("[smoke] starting biggest-config smoke test")
    print(f"[smoke] n_block=3 n_embd=64 n_head=4 num_mini_batch=2 ppo_epoch=8")
    print(f"[smoke] expecting ~2 PPO updates over {50000 / 24000:.1f} rollouts")

    t0 = time.time()
    try:
        train_main(SMOKE_ARGS)
    except torch.cuda.OutOfMemoryError as exc:
        _report_peak_vram("at OOM")
        print(f"[smoke] FAILED with CUDA OOM after {time.time() - t0:.1f}s: {exc}")
        sys.exit(2)
    except Exception:
        _report_peak_vram("at exception")
        raise

    _report_peak_vram("at end")
    print(f"[smoke] PASSED in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
