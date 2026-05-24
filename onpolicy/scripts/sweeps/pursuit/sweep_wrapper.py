#!/usr/bin/env python3
"""WandB sweep wrapper for vanilla MAPPO/RMAPPO Pursuit transformer actor-critic runs."""

import sys
from pathlib import Path

import wandb


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
TRAIN_PATH = REPO_ROOT / "onpolicy" / "scripts" / "train"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRAIN_PATH))


PURSUIT_SCALES = {
    "20p8e": (20, 8, 40),
    "40p16e": (40, 16, 45),
    "60p24e": (60, 24, 50),
    "80p32e": (80, 32, 55),
    "100p40e": (100, 40, 60),
}


def validate_architecture(n_embd, n_head):
    return n_embd % n_head == 0


def add_bool_flag(args_list, flag_name, enabled):
    if enabled:
        args_list.append(flag_name)


def resolve_pursuit_scale(config):
    if hasattr(config, "pursuit_scale"):
        scale = str(config.pursuit_scale).lower()
        if scale not in PURSUIT_SCALES:
            raise ValueError(
                f"Unknown pursuit_scale={config.pursuit_scale}. "
                f"Expected one of {sorted(PURSUIT_SCALES)}."
            )
        n_pursuers, n_evaders, size = PURSUIT_SCALES[scale]
        return n_pursuers, n_evaders, size, size

    return (
        int(config.n_pursuers),
        int(config.n_evaders),
        int(config.x_size),
        int(config.y_size),
    )


def run_training():
    if wandb.run is None:
        wandb.init()

    config = wandb.config
    run_id = wandb.run.id

    if not validate_architecture(config.n_embd, config.n_head):
        print(f"Invalid architecture: n_embd={config.n_embd} not divisible by n_head={config.n_head}")
        wandb.run.summary["invalid_architecture"] = True
        return

    if (
        not getattr(config, "use_transformer_base_actor", True)
        or not getattr(config, "use_transformer_base_critic", True)
    ):
        raise ValueError("This Pursuit sweep requires both transformer actor and transformer critic enabled.")

    if getattr(config, "use_comms_channel", False) or getattr(config, "use_fake_quantization", False):
        raise ValueError(
            "This Pursuit sweep is vanilla MAPPO: comms channel and fake quantization must stay disabled."
        )

    n_pursuers, n_evaders, x_size, y_size = resolve_pursuit_scale(config)

    from train_pursuit import main

    args_list = [
        "--env_name", str(config.env_name),
        "--algorithm_name", str(config.algorithm_name),
        "--experiment_name", f"sweep_{run_id}",
        "--seed", str(config.seed),
        "--num_env_steps", str(config.num_env_steps),
        "--episode_length", str(config.episode_length),
        "--max_cycles", str(config.max_cycles),
        "--n_pursuers", str(n_pursuers),
        "--n_evaders", str(n_evaders),
        "--x_size", str(x_size),
        "--y_size", str(y_size),
        "--n_rollout_threads", str(config.n_rollout_threads),
        "--n_eval_rollout_threads", str(config.n_eval_rollout_threads),
        "--save_interval", str(config.save_interval),
        "--log_interval", str(config.log_interval),
        "--eval_interval", str(config.eval_interval),
        "--eval_episodes", str(config.eval_episodes),
        "--use_transformer_base_actor",
        "--use_transformer_base_critic",
        "--hidden_size", str(config.n_embd),
        "--lr", str(config.lr),
        "--critic_lr", str(config.critic_lr),
        "--ppo_epoch", str(config.ppo_epoch),
        "--clip_param", str(config.clip_param),
        "--num_mini_batch", str(config.num_mini_batch),
        "--entropy_coef", str(config.entropy_coef),
        "--value_loss_coef", str(config.value_loss_coef),
        "--max_grad_norm", str(config.max_grad_norm),
        "--data_chunk_length", str(config.data_chunk_length),
        "--gamma", str(config.gamma),
        "--gae_lambda", str(config.gae_lambda),
        "--n_block", str(config.n_block),
        "--n_embd", str(config.n_embd),
        "--n_head", str(config.n_head),
        "--user_name", str(config.user_name),
        "--wandb_name", str(config.wandb_name),
    ]

    add_bool_flag(args_list, "--use_eval", getattr(config, "use_eval", False))
    add_bool_flag(args_list, "--use_linear_lr_decay", getattr(config, "use_linear_lr_decay", False))
    add_bool_flag(args_list, "--use_active_masks_in_transformer", getattr(config, "use_active_masks_in_transformer", False))
    add_bool_flag(args_list, "--use_popart", getattr(config, "use_popart", False))

    if hasattr(config, "cuda") and not config.cuda:
        args_list.append("--cuda")

    if hasattr(config, "wandb_tags"):
        args_list.append("--wandb_tags")
        args_list.extend([str(tag) for tag in config.wandb_tags])

    print("Starting vanilla Pursuit transformer actor-critic sweep run:")
    print(f"  scale={n_pursuers}P-{n_evaders}E, grid={x_size}x{y_size}, steps={config.num_env_steps}")
    print(f"  transformer: n_block={config.n_block}, n_embd={config.n_embd}, n_head={config.n_head}")
    print(f"  lr={config.lr}, critic_lr={config.critic_lr}, ppo_epoch={config.ppo_epoch}")
    print(f"  num_mini_batch={config.num_mini_batch}, entropy_coef={config.entropy_coef}")

    try:
        main(args_list)
        print(f"Training completed successfully for run {run_id}")
    except Exception as exc:
        print(f"Training failed with error: {exc}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_training()
