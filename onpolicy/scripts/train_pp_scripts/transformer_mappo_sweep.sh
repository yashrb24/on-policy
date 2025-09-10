#!/bin/bash

# Focused hyperparameter sweep based on communication learning research
# Prioritizes parameters that have shown significant impact in multi-agent settings

# Base configuration
env="PredatorPrey"
scenario="medium"
algo="rmappo"
exp="pp-focused-sweep"
num_agents=5
num_env_steps=3000000
episode_length=40

# High-impact parameters (based on research insights)
# Learning rates: Critical for transformer training stability
lr_values=(1e-4 3e-4 5e-4 1e-3)
critic_lr_values=(1e-4 3e-4 5e-4 1e-3)

# PPO parameters: Essential for policy optimization
entropy_coef_values=(0.001 0.005 0.01 0.02 0.05)  # Higher entropy may help exploration
clip_param_values=(0.05 0.1 0.2)        # Affects policy update stability

# Transformer architecture: Core to your model
n_block_values=(1 2 3)                             # Deeper networks for complex coordination
n_embd_values=(64 128 256)                         # Larger embeddings for richer representations
n_head_values=(1 2 4)                           # Multi-head attention for different aspects

# Secondary importance parameters
ppo_epoch_values=(5 10 15)                          # Keep moderate values
num_mini_batch_values=(1)                       # Simpler values for stability
max_grad_norm_values=(0.5 5.0 10.0)                   # Conservative gradient clipping

echo "Starting focused hyperparameter sweep..."
echo "Targeting high-impact parameters based on multi-agent communication research"

# Function to validate transformer architecture
validate_architecture() {
    local n_embd=$1
    local n_head=$2
    
    # Check if n_embd is divisible by n_head
    if [ $((n_embd % n_head)) -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# Function to run experiment with logging
run_focused_experiment() {
    local params=("$@")
    local lr=${params[0]}
    local critic_lr=${params[1]}
    local entropy_coef=${params[2]}
    local clip_param=${params[3]}
    local n_block=${params[4]}
    local n_embd=${params[5]}
    local n_head=${params[6]}
    local ppo_epoch=${params[7]}
    local num_mini_batch=${params[8]}
    local max_grad_norm=${params[9]}
    local seed=${params[10]}
    
    # Validate architecture
    if ! validate_architecture $n_embd $n_head; then
        echo "Skipping invalid architecture: n_embd=${n_embd}, n_head=${n_head}"
        return
    fi
    
    local exp_name="${exp}_lr${lr}_ec${entropy_coef}_cp${clip_param}_nb${n_block}_ne${n_embd}_nh${n_head}_s${seed}"
    
    echo "=========================================="
    echo "Experiment: ${exp_name}"
    echo "Key parameters:"
    echo "  Learning: lr=${lr}, critic_lr=${critic_lr}"
    echo "  Policy: entropy_coef=${entropy_coef}, clip_param=${clip_param}"
    echo "  Architecture: n_block=${n_block}, n_embd=${n_embd}, n_head=${n_head}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 python ../train/train_predatorprey.py \
        --env_name ${env} \
        --scenario_name ${scenario} \
        --algorithm_name ${algo} \
        --experiment_name ${exp_name} \
        --seed ${seed} \
        --num_agents ${num_agents} \
        --num_env_steps ${num_env_steps} \
        --episode_length ${episode_length} \
        --dim 10 \
        --vision 1 \
        --n_rollout_threads 15 \
        --save_interval 200 \
        --log_interval 200 \
        --use_transformer_base_actor \
        --lr ${lr} \
        --critic_lr ${critic_lr} \
        --ppo_epoch ${ppo_epoch} \
        --clip_param ${clip_param} \
        --num_mini_batch ${num_mini_batch} \
        --entropy_coef ${entropy_coef} \
        --max_grad_norm ${max_grad_norm} \
        --n_block ${n_block} \
        --n_embd ${n_embd} \
        --n_head ${n_head} \
        --use_wandb True \
        --user_name "yashrb" \
        --wandb_name "on-policy" \
        2>&1 | tee "logs/${exp_name}.log"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "✓ SUCCESS: ${exp_name}"
    else
        echo "✗ FAILED: ${exp_name} (exit code: $exit_code)"
    fi
    echo ""
}

# Create logs directory
mkdir -p logs

# High-priority configurations based on research insights
echo "Phase 1: Testing high-impact learning rate and entropy combinations..."

# Test different learning rate combinations with varying entropy
for lr in "${lr_values[@]}"; do
    for entropy_coef in "${entropy_coef_values[@]}"; do
        # Use base architecture and moderate other parameters
        critic_lr=$lr
        clip_param=0.2
        n_block=2
        n_embd=128
        n_head=4
        ppo_epoch=15
        num_mini_batch=1
        max_grad_norm=10.0
        seed=1
        
        run_focused_experiment $lr $critic_lr $entropy_coef $clip_param $n_block $n_embd $n_head $ppo_epoch $num_mini_batch $max_grad_norm $seed
    done
done

# echo "Phase 2: Testing transformer architecture variations..."

# # Test architecture with best learning params from phase 1 (you may want to adjust these based on phase 1 results)
# best_lr=5e-4
# best_critic_lr=5e-4
# best_entropy_coef=0.01

# for n_block in "${n_block_values[@]}"; do
#     for n_embd in "${n_embd_values[@]}"; do
#         for n_head in "${n_head_values[@]}"; do
#             if validate_architecture $n_embd $n_head; then
#                 clip_param=0.2
#                 ppo_epoch=15
#                 num_mini_batch=1
#                 max_grad_norm=10.0
#                 seed=2
                
#                 run_focused_experiment $best_lr $best_critic_lr $best_entropy_coef $clip_param $n_block $n_embd $n_head $ppo_epoch $num_mini_batch $max_grad_norm $seed
#             fi
#         done
#     done
# done

# echo "Phase 3: Fine-tuning clip_param with best configurations..."

# # Test clip_param variations with best architecture (adjust based on phase 2 results)
# best_n_block=2
# best_n_embd=128
# best_n_head=4

# for clip_param in "${clip_param_values[@]}"; do
#     ppo_epoch=15
#     num_mini_batch=1
#     max_grad_norm=10.0
#     seed=3
    
#     run_focused_experiment $best_lr $best_critic_lr $best_entropy_coef $clip_param $best_n_block $best_n_embd $best_n_head $ppo_epoch $num_mini_batch $max_grad_norm $seed
# done

echo "=========================================="
echo "Focused hyperparameter sweep completed!"
echo "Check your wandb dashboard and logs/ directory for results."
echo "Recommended next steps:"
echo "1. Analyze phase 1 results to identify best learning rate and entropy combinations"
echo "2. Use phase 2 results to select optimal transformer architecture"
echo "3. Apply phase 3 findings for final clip_param tuning"
echo "=========================================="
