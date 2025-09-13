#!/bin/bash

# Configuration file for parallel hyperparameter sweep
# Edit this file to customize your experiments

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# CPU Management
TOTAL_CPU_CORES=$(nproc)
THREADS_PER_EXPERIMENT=15
# MAX_PARALLEL_JOBS will be calculated automatically based on available cores
# Override if you want to limit further:
# MAX_PARALLEL_JOBS=2

# GPU Management
AVAILABLE_GPUS=(0 1 2 3)  # List available GPU IDs
# Experiments will cycle through these GPUs

# =============================================================================
# EXPERIMENT BASE CONFIGURATION
# =============================================================================

ENV_NAME="TrafficJunction"
SCENARIO_NAME="medium"
ALGORITHM_NAME="rmappo"
EXPERIMENT_PREFIX="tj-exploratory-runs"
NUM_AGENTS=5
NUM_ENV_STEPS=3000000
EPISODE_LENGTH=40
DIM=10
VISION=1
N_ROLLOUT_THREADS=15
SAVE_INTERVAL=200
LOG_INTERVAL=200

# Wandb configuration
USE_WANDB=true
WANDB_USER="yashrb"
WANDB_PROJECT="on-policy"

# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

# Learning rates to test
LR_VALUES=(1e-4 3e-4 5e-4 1e-3)
CRITIC_LR_VALUES=(1e-4 3e-4 5e-4 1e-3)

# PPO parameters
PPO_EPOCH_VALUES=(5 10 15)
CLIP_PARAM_VALUES=(0.05 0.1 0.2)
NUM_MINI_BATCH_VALUES=(1)
ENTROPY_COEF_VALUES=(0.001 0.005 0.01)
MAX_GRAD_NORM_VALUES=(0.5 10.0)

# Transformer architecture parameters
N_BLOCK_VALUES=(1 2 3)
N_EMBD_VALUES=(64 128 256)
N_HEAD_VALUES=(1 2 4)

# =============================================================================
# EXPERIMENT GENERATION STRATEGY
# =============================================================================

# Choose strategy: "focused", "grid", "random"
STRATEGY="focused"

# For random strategy
RANDOM_EXPERIMENTS=50
RANDOM_SEED=42

# For grid strategy (warning: can generate many experiments!)
GRID_SAMPLE_RATE=0.1  # Sample 10% of all combinations

# =============================================================================
# FOCUSED STRATEGY CONFIGURATIONS
# =============================================================================

# Phase 1: Learning rate and entropy combinations (with baseline architecture)
generate_phase1_configs() {
    local configs=()
    local baseline_clip=0.2
    local baseline_n_block=2
    local baseline_n_embd=64
    local baseline_n_head=4
    local baseline_ppo_epoch=10
    local baseline_mini_batch=1
    local baseline_grad_norm=10.0
    
    local seed=1
    for lr in "${LR_VALUES[@]}"; do
        for entropy in "${ENTROPY_COEF_VALUES[@]}"; do
            configs+=("$lr $lr $entropy $baseline_clip $baseline_n_block $baseline_n_embd $baseline_n_head $baseline_ppo_epoch $baseline_mini_batch $baseline_grad_norm $seed")
            ((seed++))
        done
    done
    
    printf '%s\n' "${configs[@]}"
}

# Phase 2: Architecture exploration (with best learning params from phase 1)
# generate_phase2_configs() {
#     local configs=()
#     local best_lr=1e-4  # Update based on phase 1 results
#     local best_entropy=0.01  # Update based on phase 1 results
#     local baseline_clip=0.2
#     local baseline_ppo_epoch=15
#     local baseline_mini_batch=1
#     local baseline_grad_norm=10.0
    
#     local seed=100
#     for n_block in "${N_BLOCK_VALUES[@]}"; do
#         for n_embd in "${N_EMBD_VALUES[@]}"; do
#             for n_head in "${N_HEAD_VALUES[@]}"; do
#                 # Validate architecture
#                 if [ $((n_embd % n_head)) -eq 0 ]; then
#                     configs+=("$best_lr $best_lr $best_entropy $baseline_clip $n_block $n_embd $n_head $baseline_ppo_epoch $baseline_mini_batch $baseline_grad_norm $seed")
#                     ((seed++))
#                 fi
#             done
#         done
#     done
    
#     printf '%s\n' "${configs[@]}"
# }

# Phase 3: PPO parameter fine-tuning (with best architecture from phase 2)
# generate_phase3_configs() {
#     local configs=()
#     local best_lr=5e-4  # Update based on phase 1 results
#     local best_entropy=0.01  # Update based on phase 1 results
#     local best_n_block=2  # Update based on phase 2 results
#     local best_n_embd=128  # Update based on phase 2 results
#     local best_n_head=4  # Update based on phase 2 results
#     local baseline_mini_batch=1
#     local baseline_grad_norm=10.0
    
#     local seed=200
#     for clip in "${CLIP_PARAM_VALUES[@]}"; do
#         for ppo_epoch in "${PPO_EPOCH_VALUES[@]}"; do
#             configs+=("$best_lr $best_lr $best_entropy $clip $best_n_block $best_n_embd $best_n_head $ppo_epoch $baseline_mini_batch $baseline_grad_norm $seed")
#             ((seed++))
#         done
#     done
    
#     printf '%s\n' "${configs[@]}"
# }

# =============================================================================
# RANDOM STRATEGY CONFIGURATION
# =============================================================================

generate_random_configs() {
    local configs=()
    local seed=$RANDOM_SEED
    
    for i in $(seq 1 $RANDOM_EXPERIMENTS); do
        # Randomly sample parameters
        local lr=${LR_VALUES[$RANDOM % ${#LR_VALUES[@]}]}
        local critic_lr=${CRITIC_LR_VALUES[$RANDOM % ${#CRITIC_LR_VALUES[@]}]}
        local entropy=${ENTROPY_COEF_VALUES[$RANDOM % ${#ENTROPY_COEF_VALUES[@]}]}
        local clip=${CLIP_PARAM_VALUES[$RANDOM % ${#CLIP_PARAM_VALUES[@]}]}
        local n_block=${N_BLOCK_VALUES[$RANDOM % ${#N_BLOCK_VALUES[@]}]}
        local n_embd=${N_EMBD_VALUES[$RANDOM % ${#N_EMBD_VALUES[@]}]}
        local n_head=${N_HEAD_VALUES[$RANDOM % ${#N_HEAD_VALUES[@]}]}
        local ppo_epoch=${PPO_EPOCH_VALUES[$RANDOM % ${#PPO_EPOCH_VALUES[@]}]}
        local mini_batch=${NUM_MINI_BATCH_VALUES[$RANDOM % ${#NUM_MINI_BATCH_VALUES[@]}]}
        local grad_norm=${MAX_GRAD_NORM_VALUES[$RANDOM % ${#MAX_GRAD_NORM_VALUES[@]}]}
        
        # Ensure valid architecture
        while [ $((n_embd % n_head)) -ne 0 ] || [ $n_head -gt $n_embd ]; do
            n_head=${N_HEAD_VALUES[$RANDOM % ${#N_HEAD_VALUES[@]}]}
        done
        
        configs+=("$lr $critic_lr $entropy $clip $n_block $n_embd $n_head $ppo_epoch $mini_batch $grad_norm $seed")
        ((seed++))
    done
    
    printf '%s\n' "${configs[@]}"
}

# =============================================================================
# GRID STRATEGY CONFIGURATION
# =============================================================================

generate_grid_configs() {
    local configs=()
    local seed=1
    local total_combinations=0
    local sampled_combinations=0
    
    # Calculate total combinations
    total_combinations=$(( ${#LR_VALUES[@]} * ${#CRITIC_LR_VALUES[@]} * ${#ENTROPY_COEF_VALUES[@]} * ${#CLIP_PARAM_VALUES[@]} * ${#N_BLOCK_VALUES[@]} * ${#N_EMBD_VALUES[@]} * ${#N_HEAD_VALUES[@]} * ${#PPO_EPOCH_VALUES[@]} * ${#NUM_MINI_BATCH_VALUES[@]} * ${#MAX_GRAD_NORM_VALUES[@]} ))
    
    echo "Total possible combinations: $total_combinations"
    echo "Sampling rate: $GRID_SAMPLE_RATE"
    echo "Expected samples: $(echo "$total_combinations * $GRID_SAMPLE_RATE" | bc -l | cut -d. -f1)"
    
    for lr in "${LR_VALUES[@]}"; do
        for critic_lr in "${CRITIC_LR_VALUES[@]}"; do
            for entropy in "${ENTROPY_COEF_VALUES[@]}"; do
                for clip in "${CLIP_PARAM_VALUES[@]}"; do
                    for n_block in "${N_BLOCK_VALUES[@]}"; do
                        for n_embd in "${N_EMBD_VALUES[@]}"; do
                            for n_head in "${N_HEAD_VALUES[@]}"; do
                                for ppo_epoch in "${PPO_EPOCH_VALUES[@]}"; do
                                    for mini_batch in "${NUM_MINI_BATCH_VALUES[@]}"; do
                                        for grad_norm in "${MAX_GRAD_NORM_VALUES[@]}"; do
                                            # Validate architecture
                                            if [ $((n_embd % n_head)) -eq 0 ]; then
                                                # Sample with probability
                                                if (( $(echo "$RANDOM < $GRID_SAMPLE_RATE * 32767" | bc -l) )); then
                                                    configs+=("$lr $critic_lr $entropy $clip $n_block $n_embd $n_head $ppo_epoch $mini_batch $grad_norm $seed")
                                                    ((sampled_combinations++))
                                                    ((seed++))
                                                fi
                                            fi
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
    
    echo "Actually sampled: $sampled_combinations configurations"
    printf '%s\n' "${configs[@]}"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Function to validate configuration
validate_config() {
    local config="$1"
    IFS=' ' read -ra params <<< "$config"
    
    if [ ${#params[@]} -ne 11 ]; then
        echo "Error: Configuration must have exactly 11 parameters"
        echo "Got: ${#params[@]} parameters"
        return 1
    fi
    
    local n_embd=${params[5]}
    local n_head=${params[6]}
    
    if [ $((n_embd % n_head)) -ne 0 ]; then
        echo "Error: n_embd ($n_embd) must be divisible by n_head ($n_head)"
        return 1
    fi
    
    return 0
}

# Function to print configuration summary
print_config_summary() {
    echo "=== Hyperparameter Sweep Configuration ==="
    echo "Strategy: $STRATEGY"
    echo "Environment: $ENV_NAME ($SCENARIO_NAME)"
    echo "Algorithm: $ALGORITHM_NAME"
    echo "Agents: $NUM_AGENTS"
    echo "Steps: $NUM_ENV_STEPS"
    echo ""
    echo "System Configuration:"
    echo "  CPU Cores: $TOTAL_CPU_CORES"
    echo "  Threads per experiment: $THREADS_PER_EXPERIMENT"
    echo "  Max parallel jobs: $((TOTAL_CPU_CORES / THREADS_PER_EXPERIMENT))"
    echo "  Available GPUs: ${AVAILABLE_GPUS[*]}"
    echo ""
    echo "Parameter Ranges:"
    echo "  Learning rates: ${LR_VALUES[*]}"
    echo "  Entropy coefficients: ${ENTROPY_COEF_VALUES[*]}"
    echo "  Clip parameters: ${CLIP_PARAM_VALUES[*]}"
    echo "  Transformer blocks: ${N_BLOCK_VALUES[*]}"
    echo "  Embedding dimensions: ${N_EMBD_VALUES[*]}"
    echo "  Attention heads: ${N_HEAD_VALUES[*]}"
    echo "=========================================="
}

# Function to generate all configurations based on strategy
generate_all_configs() {
    case $STRATEGY in
        "focused")
            echo "# Phase 1: Learning rate and entropy exploration"
            generate_phase1_configs
            echo "# Phase 2: Architecture exploration" 
            generate_phase2_configs
            echo "# Phase 3: PPO parameter fine-tuning"
            generate_phase3_configs
            ;;
        "random")
            echo "# Random sampling strategy"
            generate_random_configs
            ;;
        "grid")
            echo "# Grid sampling strategy"
            generate_grid_configs
            ;;
        *)
            echo "Error: Unknown strategy '$STRATEGY'"
            echo "Available strategies: focused, random, grid"
            exit 1
            ;;
    esac
}

# Function to estimate runtime
estimate_runtime() {
    local num_configs=$1
    local max_parallel=${2:-$((TOTAL_CPU_CORES / THREADS_PER_EXPERIMENT))}
    
    # Rough estimates (adjust based on your system)
    local avg_experiment_time_hours=2.5  # hours per experiment
    local total_time_hours=$(echo "scale=1; $num_configs * $avg_experiment_time_hours / $max_parallel" | bc -l)
    
    echo "Runtime Estimation:"
    echo "  Total experiments: $num_configs"
    echo "  Parallel jobs: $max_parallel"
    echo "  Estimated time per experiment: ${avg_experiment_time_hours}h"
    echo "  Estimated total time: ${total_time_hours}h ($(echo "scale=1; $total_time_hours / 24" | bc -l) days)"
}

# =============================================================================
# MAIN CONFIGURATION VALIDATION
# =============================================================================

# Validate that all required arrays are defined
if [ ${#LR_VALUES[@]} -eq 0 ]; then
    echo "Error: LR_VALUES array is empty"
    exit 1
fi

if [ ${#N_EMBD_VALUES[@]} -eq 0 ]; then
    echo "Error: N_EMBD_VALUES array is empty"
    exit 1
fi

# Validate GPU configuration
if [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; then
    echo "Warning: No GPUs specified, using CPU only"
    AVAILABLE_GPUS=(0)  # Default to GPU 0
fi

# Export key variables for use in other scripts
export ENV_NAME SCENARIO_NAME ALGORITHM_NAME EXPERIMENT_PREFIX
export NUM_AGENTS NUM_ENV_STEPS EPISODE_LENGTH DIM VISION N_ROLLOUT_THREADS
export SAVE_INTERVAL LOG_INTERVAL USE_WANDB WANDB_USER WANDB_PROJECT
export TOTAL_CPU_CORES THREADS_PER_EXPERIMENT
export AVAILABLE_GPUS
