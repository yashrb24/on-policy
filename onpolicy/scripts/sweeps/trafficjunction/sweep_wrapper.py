#!/usr/bin/env python3
"""
WandB Sweep Wrapper for TrafficJunction Multi-Agent RL Training
"""

import sys
import os
from pathlib import Path

# Add the training script directory to path
train_path = Path(__file__).parent / "../../train"
sys.path.insert(0, str(train_path.resolve()))

import wandb

def validate_architecture(n_embd, n_head):
    """Validate that n_embd is divisible by n_head"""
    return n_embd % n_head == 0

def run_training():
    # Initialize wandb if not already done (when called by sweep agent, this connects to existing run)
    if wandb.run is None:
        wandb.init()

    # Now we can access config
    config = wandb.config

    # Validate architecture
    if not validate_architecture(config.n_embd, config.n_head):
        print(f"Invalid architecture: n_embd={config.n_embd} not divisible by n_head={config.n_head}")
        wandb.run.summary["invalid_architecture"] = True
        return

    # Import the training module
    from train_traffic_junction import main

    # Build arguments as a list (like command line args)
    args_list = [
        '--env_name', str(config.env_name),
        '--difficulty', str(config.difficulty),
        '--algorithm_name', str(config.algorithm_name),
        '--experiment_name', f"sweep_{wandb.run.id}",
        '--seed', str(config.seed),
        '--num_agents', str(config.num_agents),
        '--num_env_steps', str(config.num_env_steps),
        '--episode_length', str(config.episode_length),
        '--dim', str(config.dim),
        '--vision', str(config.vision),
        '--add_rate_min', str(config.add_rate_min),
        '--add_rate_max', str(config.add_rate_max),
        '--curr_start', str(config.curr_start),
        '--curr_end', str(config.curr_end),
        '--n_rollout_threads', str(config.n_rollout_threads),
        '--save_interval', str(config.save_interval),
        '--log_interval', str(config.log_interval),
        '--use_transformer_base_actor',
        '--hidden_size', str(config.n_embd),
        '--lr', str(config.lr),
        '--critic_lr', str(config.critic_lr),
        '--ppo_epoch', str(config.ppo_epoch),
        '--clip_param', str(config.clip_param),
        '--num_mini_batch', str(config.num_mini_batch),
        '--entropy_coef', str(config.entropy_coef),
        '--max_grad_norm', str(config.max_grad_norm),
        '--n_block', str(config.n_block),
        '--n_embd', str(config.n_embd),
        '--n_head', str(config.n_head),
        '--user_name', str(config.user_name),
        '--wandb_name', str(config.wandb_name),
    ]

    # Optional TJ-specific flags
    if getattr(config, 'use_active_masks_in_transformer', False):
        args_list.append('--use_active_masks_in_transformer')
    if getattr(config, 'use_fake_quantization', False):
        args_list.append('--use_fake_quantization')
        if hasattr(config, 'quant_bits'):
            args_list.extend(['--quant_bits', str(config.quant_bits)])

    # DDCL arguments (optional)
    if getattr(config, 'use_comms_channel', False):
        args_list.append('--use_comms_channel')
    if hasattr(config, 'comm_coeff'):
        args_list.extend(['--comm_coeff', str(config.comm_coeff)])
    if hasattr(config, 'num_messages'):
        args_list.extend(['--num_messages', str(config.num_messages)])
    if hasattr(config, 'ddcl_variation'):
        args_list.extend(['--ddcl_variation', str(config.ddcl_variation)])
    if hasattr(config, 'channel') and config.channel is not None:
        args_list.extend(['--channel', str(config.channel)])
    if hasattr(config, 'delta') and config.delta is not None:
        args_list.extend(['--delta', str(config.delta)])
    if getattr(config, 'delta_learnable', False):
        args_list.append('--delta_learnable')
    if getattr(config, 'delta_global_learnable', False):
        args_list.append('--delta_global_learnable')

    # Set CUDA device if needed
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print(f"Starting training with sweep config:")
    print(f"  lr={config.lr}, critic_lr={config.critic_lr}")
    print(f"  entropy_coef={config.entropy_coef}, clip_param={config.clip_param}")
    print(f"  n_block={config.n_block}, n_embd={config.n_embd}, n_head={config.n_head}")
    if getattr(config, 'use_comms_channel', False):
        print(f"  DDCL: comm_coeff={config.comm_coeff}, num_messages={config.num_messages}, variation={getattr(config, 'ddcl_variation', 'new')}")

    # Call the main function directly
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
