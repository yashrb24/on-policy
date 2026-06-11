#!/usr/bin/env python3
"""WandB sweep wrapper for the 3v1 delta-channels grid.

Pairs with configs/sweep_config_3v1_delta_channels.yaml. Kept separate from
the general sweep_wrapper.py so the delta_config -> CLI mapping lives next to
the only sweep that uses it.

delta_config encodes the 4 mutually-exclusive delta modes as a single string
(WandB grid-sweep cells must be flat, so we cannot let learnable flags and
fixed delta values combine arbitrarily):

    fixed_10         -> --num_messages 10           (delta auto = 1/10)
    fixed_15         -> --num_messages 15           (delta auto = 1/15)
    learnable_global -> --num_messages 15 + --delta_global_learnable
    learnable_perdim -> --num_messages 15 + --delta_learnable

Phase-1 PP convention: only --num_messages is set; --delta is omitted so the
model falls back to delta = 1/num_messages, keeping the quantizer step and
the rate-penalty M tied (matches sweep_config_ddcl_hard.yaml's original
single-dial sweep over num_messages).
"""

import sys
from pathlib import Path

# Add the training script directory to path
train_path = Path(__file__).parent / "../../train"
sys.path.insert(0, str(train_path.resolve()))

import wandb


DELTA_CONFIG_MAP = {
    #                   num_messages, delta_learnable, delta_global_learnable
    "fixed_10":         (10, False, False),
    "fixed_15":         (15, False, False),
    "learnable_global": (15, False, True),
    "learnable_perdim": (15, True,  False),
}


def run_training():
    if wandb.run is None:
        wandb.init()
    config = wandb.config

    if config.n_embd % config.n_head != 0:
        print(f"Invalid architecture: n_embd={config.n_embd} not divisible by n_head={config.n_head}")
        wandb.run.summary["invalid_architecture"] = True
        return

    args_list = [
        "--env_name",        str(config.env_name),
        "--scenario_name",   str(config.scenario_name),
        "--algorithm_name",  str(config.algorithm_name),
        "--experiment_name", f"sweep_{wandb.run.id}",
        "--seed",            str(config.seed),
        "--num_agents",      str(config.num_agents),
        "--num_env_steps",   str(config.num_env_steps),
        "--episode_length",  str(config.episode_length),
        "--representation",  str(config.representation),
        "--rewards",         str(config.rewards),
        "--n_rollout_threads", str(config.n_rollout_threads),
        "--save_interval",   str(config.save_interval),
        "--log_interval",    str(config.log_interval),
        "--use_transformer_base_actor",
        "--hidden_size",     str(config.hidden_size),
        "--lr",              str(config.lr),
        "--critic_lr",       str(config.critic_lr),
        "--ppo_epoch",       str(config.ppo_epoch),
        "--clip_param",      str(config.clip_param),
        "--num_mini_batch",  str(config.num_mini_batch),
        "--entropy_coef",    str(config.entropy_coef),
        "--max_grad_norm",   str(config.max_grad_norm),
        "--n_block",         str(config.n_block),
        "--n_embd",          str(config.n_embd),
        "--n_head",          str(config.n_head),
        "--user_name",       str(config.user_name),
        "--wandb_name",      str(config.wandb_name),
    ]

    # DDCL: this sweep always sets use_comms_channel=true and a channel value.
    if config.use_comms_channel:
        args_list.append("--use_comms_channel")
    args_list.extend(["--comm_coeff", str(config.comm_coeff)])
    args_list.extend(["--channel",    str(config.channel)])

    num_messages, delta_learnable, delta_global_learnable = DELTA_CONFIG_MAP[config.delta_config]
    args_list.extend(["--num_messages", str(num_messages)])
    if delta_learnable:
        args_list.append("--delta_learnable")
    if delta_global_learnable:
        args_list.append("--delta_global_learnable")

    print(
        f"Sweep config: channel={config.channel}, delta_config={config.delta_config} "
        f"(num_messages={num_messages}, delta_learnable={delta_learnable}, "
        f"delta_global_learnable={delta_global_learnable}), "
        f"comm_coeff={config.comm_coeff}, seed={config.seed}"
    )

    from train_football import main
    try:
        main(args_list)
        print(f"Training completed successfully for run {wandb.run.id}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_training()
