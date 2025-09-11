#!/usr/bin/env python3
"""
WandB Sweep Wrapper for Football Multi-Agent RL Training
This script handles architecture validation and parameter filtering
"""

import wandb
import subprocess
import sys
import os
from pathlib import Path

def validate_architecture(config):
    """
    Validate that n_embd is divisible by n_head
    """
    n_embd = config.get('n_embd', 128)
    n_head = config.get('n_head', 4)
    
    if n_embd % n_head != 0:
        print(f"Invalid architecture: n_embd={n_embd} not divisible by n_head={n_head}")
        return False
    return True

def run_training():
    """
    Main function to run training with WandB sweep
    """
    # Initialize wandb
    run = wandb.init()
    config = wandb.config
    
    # Validate architecture
    if not validate_architecture(config):
        # Mark this run as failed/skipped
        wandb.run.summary["invalid_architecture"] = True
        wandb.finish(exit_code=1)
        return
    
    # Ensure hidden_size matches n_embd
    config.update({'hidden_size': config.n_embd}, allow_val_change=True)
    
    # Build command
    cmd = [
        'python', '../train/train_football.py',
        '--env_name', str(config.env_name),
        '--scenario_name', str(config.scenario_name),
        '--algorithm_name', str(config.algorithm_name),
        '--experiment_name', f"sweep_{wandb.run.id}",
        '--seed', str(config.seed),
        '--num_agents', str(config.num_agents),
        '--num_env_steps', str(config.num_env_steps),
        '--episode_length', str(config.episode_length),
        '--representation', str(config.representation),
        '--rewards', str(config.rewards),
        '--n_rollout_threads', str(config.n_rollout_threads),
        '--save_interval', str(config.save_interval),
        '--log_interval', str(config.log_interval),
        '--use_transformer_base_actor',
        '--hidden_size', str(config.n_embd),  # Use n_embd for hidden_size
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
    
    # Set CUDA device (you can modify this for multi-GPU setups)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the training script
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        wandb.finish(exit_code=e.returncode)
        sys.exit(e.returncode)
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    run_training()
