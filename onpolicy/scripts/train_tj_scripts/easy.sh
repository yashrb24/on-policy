#!/bin/bash

# EXPERIMENT CONFIGURATION
exp="tj-exploratory-runs"
seed=1

# ENVIRONMENT CONFIGURATION
env="TrafficJunction"
difficulty="easy"
num_agents=5
dim=6
vision=1

# Curriculum settings
add_rate_min=0.3
add_rate_max=0.3
curr_start=0
curr_end=0

# TRAINING CONFIGURATION
algo="rmappo"
num_env_steps=40000
episode_length=20
n_rollout_threads=2

# PPO hyperparameters
ppo_epoch=10
num_mini_batch=1
lr=1e-3

# NETWORK ARCHITECTURE
hidden_size=128
n_head=4

# LOGGING & CHECKPOINTING
save_interval=200
log_interval=400

# WandB configuration - set use_wandb to True to enable
use_wandb=False  # Change to True to enable WandB
user_name="yashrb"
wandb_name="on-policy"

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================
CUDA_VISIBLE_DEVICES=0

# ============================================================================
# RUN TRAINING
# ============================================================================
# Build the command
cmd="python /Users/yashrb/Projects/on-policy/onpolicy/scripts/train/train_traffic_junction.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --seed ${seed} \
    --num_agents ${num_agents} \
    --num_env_steps ${num_env_steps} \
    --episode_length ${episode_length} \
    --save_interval ${save_interval} \
    --log_interval ${log_interval} \
    --dim ${dim} \
    --vision ${vision} \
    --add_rate_min ${add_rate_min} \
    --add_rate_max ${add_rate_max} \
    --curr_start ${curr_start} \
    --curr_end ${curr_end} \
    --difficulty ${difficulty} \
    --n_rollout_threads ${n_rollout_threads} \
    --ppo_epoch ${ppo_epoch} \
    --num_mini_batch ${num_mini_batch} \
    --n_head ${n_head} \
    --hidden_size ${hidden_size} \
    --lr ${lr} \
    --use_wandb ${use_wandb}"

# Add WandB parameters if enabled
if [ "${use_wandb}" = "True" ]; then
    cmd="${cmd} --user_name ${user_name} --wandb_name ${wandb_name}"
fi

# Execute the command
eval ${cmd}