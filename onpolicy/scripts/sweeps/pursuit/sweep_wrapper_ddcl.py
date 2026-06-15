#!/usr/bin/env python3
"""WandB sweep wrapper for Pursuit DDCL channel x granularization experiments.

Pairs with configs/sweep_config_ddcl_delta_channels.yaml. This is the DDCL
counterpart to the vanilla sweep_wrapper.py (which forbids comms channels): it
runs the proven Pursuit lambda0.90 shrunk-actor recipe (1-block transformer
actor + 3-block transformer critic, obs98) with a learned communication channel
enabled in the *actor only*, sweeping channel type, delta granularization,
comm_coeff (lambda), and seed.

delta_config encodes the 4 mutually-exclusive delta modes as a single string
(WandB grid cells must be flat). We follow the Football convention of setting
only --num_messages for the fixed cells so the quantizer step delta=1/M and the
rate-penalty M stay tied:

    fixed_10         -> --num_messages 10            (delta auto = 1/10)
    fixed_15         -> --num_messages 15            (delta auto = 1/15)
    learnable_global -> --num_messages 15 + --delta_global_learnable
    learnable_perdim -> --num_messages 15 + --delta_learnable

The crash-cleanup machinery (orphaned SubprocVecEnv workers, hung wandb.finish)
is carried over verbatim from sweep_wrapper.py so a failed cell does not stall
the agent's next run.
"""

import os
import signal
import subprocess
import sys
import threading
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


DELTA_CONFIG_MAP = {
    #                   num_messages, delta_learnable, delta_global_learnable
    "fixed_10":         (10, False, False),
    "fixed_15":         (15, False, False),
    "learnable_global": (15, False, True),
    "learnable_perdim": (15, True,  False),
}


def _force_kill_descendants():
    """SIGKILL every descendant of this process.

    On crash, train_pursuit.main() never reaches envs.close(), so the
    SubprocVecEnv worker processes are orphaned. Hard-killing the whole subtree
    here guarantees the wandb agent's next run starts on a clean GPU and CPU.
    """
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["pgrep", "-P", str(my_pid)], stderr=subprocess.DEVNULL
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    for pid_str in out.split():
        try:
            os.kill(int(pid_str), signal.SIGKILL)
        except (ProcessLookupError, ValueError, PermissionError):
            pass


def _arm_hard_exit_watchdog(seconds, exit_code):
    """If cleanup itself hangs (e.g. wandb.finish blocks), os._exit anyway."""
    timer = threading.Timer(seconds, lambda: os._exit(exit_code))
    timer.daemon = True
    timer.start()
    return timer


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
        raise ValueError("This Pursuit DDCL sweep requires both transformer actor and transformer critic enabled.")

    if not getattr(config, "use_comms_channel", False):
        raise ValueError("This is the Pursuit DDCL sweep: use_comms_channel must be enabled.")

    if str(config.channel) not in ("sd", "tpdf", "async_sd", "ad"):
        raise ValueError(f"Unexpected channel={config.channel}; expected one of sd/tpdf/async_sd/ad.")

    if str(config.delta_config) not in DELTA_CONFIG_MAP:
        raise ValueError(f"Unknown delta_config={config.delta_config}; expected one of {sorted(DELTA_CONFIG_MAP)}.")

    n_pursuers, n_evaders, x_size, y_size = resolve_pursuit_scale(config)

    from train_pursuit import main

    # Fixed lambda0.90 shrunk-actor recipe (mirrors the log-confirmed shrink_a1blk command verbatim;
    # value_loss_coef is left at the config.py default of 1.0 by passing it explicitly).
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
        "--catch_reward", str(config.catch_reward),
        "--tag_reward", str(config.tag_reward),
        "--urgency_reward", str(config.urgency_reward),
        "--n_rollout_threads", str(config.n_rollout_threads),
        "--n_training_threads", str(config.n_training_threads),
        "--lr", str(config.lr),
        "--critic_lr", str(config.critic_lr),
        "--entropy_coef", str(config.entropy_coef),
        "--ppo_epoch", str(config.ppo_epoch),
        "--num_mini_batch", str(config.num_mini_batch),
        "--clip_param", str(config.clip_param),
        "--gamma", str(config.gamma),
        "--gae_lambda", str(config.gae_lambda),
        "--max_grad_norm", str(config.max_grad_norm),
        "--value_loss_coef", str(config.value_loss_coef),
        "--n_head", str(config.n_head),
        "--n_block", str(config.n_block),
        "--actor_n_block", str(config.actor_n_block),
        "--critic_n_block", str(config.critic_n_block),
        "--n_embd", str(config.n_embd),
        "--hidden_size", str(config.hidden_size),
        "--data_chunk_length", str(config.data_chunk_length),
        "--user_name", str(config.user_name),
        "--wandb_name", str(config.wandb_name),
    ]

    add_bool_flag(args_list, "--use_transformer_base_actor", True)
    add_bool_flag(args_list, "--use_transformer_base_critic", True)
    add_bool_flag(args_list, "--pursuit_drop_walls_channel", getattr(config, "pursuit_drop_walls_channel", True))

    # DDCL: learned communication channel in the actor only.
    add_bool_flag(args_list, "--use_comms_channel", True)
    add_bool_flag(args_list, "--comms_channel_actor_only", getattr(config, "comms_channel_actor_only", True))
    args_list.extend(["--channel", str(config.channel)])
    args_list.extend(["--comm_coeff", str(config.comm_coeff)])

    num_messages, delta_learnable, delta_global_learnable = DELTA_CONFIG_MAP[str(config.delta_config)]
    args_list.extend(["--num_messages", str(num_messages)])
    add_bool_flag(args_list, "--delta_learnable", delta_learnable)
    add_bool_flag(args_list, "--delta_global_learnable", delta_global_learnable)

    # Periodic held-out evaluation (Done% / Capture% to wandb). eval_interval must be a multiple of
    # episode_length * n_rollout_threads or the exact-equality trigger never fires.
    if getattr(config, "use_eval", False):
        add_bool_flag(args_list, "--use_eval", True)
        args_list.extend(["--eval_interval", str(config.eval_interval)])
        args_list.extend(["--n_eval_rollout_threads", str(config.n_eval_rollout_threads)])
        # We want STOCHASTIC eval rollouts. --eval_deterministic is store_false (default True),
        # so passing the flag flips eval to stochastic.
        if getattr(config, "eval_stochastic", True):
            args_list.append("--eval_deterministic")

    if hasattr(config, "wandb_tags"):
        args_list.append("--wandb_tags")
        args_list.extend([str(tag) for tag in config.wandb_tags])

    print("Starting Pursuit DDCL delta-channels sweep run:")
    print(f"  scale={n_pursuers}P-{n_evaders}E, grid={x_size}x{y_size}, steps={config.num_env_steps}")
    print(f"  actor_n_block={config.actor_n_block}, critic_n_block={config.critic_n_block}, "
          f"n_embd={config.n_embd}, n_head={config.n_head}")
    print(f"  channel={config.channel}, delta_config={config.delta_config} "
          f"(num_messages={num_messages}, delta_learnable={delta_learnable}, "
          f"delta_global_learnable={delta_global_learnable}), comm_coeff={config.comm_coeff}, seed={config.seed}")

    crashed = False
    try:
        main(args_list)
        print(f"Training completed successfully for run {run_id}")
    except BaseException as exc:
        crashed = True
        print(f"Training failed with error: {exc}")
        import traceback
        traceback.print_exc()

    if crashed:
        # Absolute deadline: if any cleanup step below hangs, hard-exit anyway.
        _arm_hard_exit_watchdog(30, 1)

        _force_kill_descendants()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            if wandb.run is not None:
                wandb.finish(exit_code=1)
        except Exception:
            pass

        # Skip Python finalizers — torch CUDA destructors can deadlock here.
        os._exit(1)


if __name__ == "__main__":
    run_training()
