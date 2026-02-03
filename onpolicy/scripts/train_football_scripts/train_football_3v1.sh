#!/bin/bash

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base configuration
ENV_NAME="Football"
SCENARIO_NAME="academy_3_vs_1_with_keeper"
ALGORITHM_NAME="rmappo"
# "grf-3v1-best-config"/ "grf-3v1-ddcl-config"/ "grf-3v1-fakequant-config"/ "grf-3v1-delta-config"
EXPERIMENT_NAME="ddcl-grf-3v1-best-config"
NUM_AGENTS=3
NUM_ENV_STEPS=21000000
EPISODE_LENGTH=200
N_ROLLOUT_THREADS=15
PPO_EPOCH=10
NUM_MINI_BATCH=1
SAVE_INTERVAL=3000
LOG_INTERVAL=3000
WANDB_USER="yashrb"
WANDB_PROJECT="ddcl-applications"

# Best hyperparameters
BEST_LR="3e-4"
BEST_CRITIC_LR="3e-4"
BEST_ENTROPY_COEF="0.001"
BEST_CLIP_PARAM="0.05"
BEST_N_BLOCK="1"
BEST_N_EMBD="64"
BEST_N_HEAD="4"
BEST_HIDDEN_SIZE="64"  # Must match n_embd

# DDCL Configuration
COMM_COEFF="5e-4"  # Options: 1e-4/ 1e-3/ 1e-2
NUM_MESSAGES="15"  # Options: 10/15/20
# Uncomment these lines to enable DDCL:
# --use_comms_channel \
# --comm_coeff "$COMM_COEFF" \
# --num_messages "$NUM_MESSAGES" \

# Fake Quantization Configuration
# QUANT_BITS="8"  # Options: 4/8/16
# Uncomment these lines to enable fake quantization:
# --use_fake_quantization \
# --quant_bits "$QUANT_BITS" \

# Seed to run
SEED=41

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="run_${TIMESTAMP}"
mkdir -p "${RUN_DIR}/logs"

echo "=== Single Seed Execution Setup ==="
echo "Created run directory: $RUN_DIR"
echo "Seed: $SEED"
echo "===================================="
echo ""

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

EXP_NAME="${EXPERIMENT_NAME}_seed${SEED}"
LOG_FILE="${RUN_DIR}/logs/${EXP_NAME}.log"

echo "=== Starting Experiment ==="
echo "Experiment name: $EXP_NAME"
echo "Seed: $SEED"
echo "Log file: $LOG_FILE"
echo "WandB project: $WANDB_PROJECT"
echo ""

# Run the experiment
python ../train/train_football.py \
    --env_name "$ENV_NAME" \
    --scenario_name "$SCENARIO_NAME" \
    --algorithm_name "$ALGORITHM_NAME" \
    --experiment_name "$EXP_NAME" \
    --seed "$SEED" \
    --num_agents "$NUM_AGENTS" \
    --num_env_steps "$NUM_ENV_STEPS" \
    --episode_length "$EPISODE_LENGTH" \
    --dim "$DIM" \
    --vision "$VISION" \
    --n_rollout_threads "$N_ROLLOUT_THREADS" \
    --ppo_epoch "$PPO_EPOCH" \
    --num_mini_batch "$NUM_MINI_BATCH" \
    --save_interval "$SAVE_INTERVAL" \
    --log_interval "$LOG_INTERVAL" \
    --use_transformer_base_actor \
    --hidden_size "$BEST_HIDDEN_SIZE" \
    --lr "$BEST_LR" \
    --critic_lr "$BEST_CRITIC_LR" \
    --clip_param "$BEST_CLIP_PARAM" \
    --entropy_coef "$BEST_ENTROPY_COEF" \
    --n_block "$BEST_N_BLOCK" \
    --n_embd "$BEST_N_EMBD" \
    --n_head "$BEST_N_HEAD" \
    --use_comms_channel \
    --comm_coeff "$COMM_COEFF" \
    --num_messages "$NUM_MESSAGES" \
    --user_name "$WANDB_USER" \
    --wandb_name "$WANDB_PROJECT" \
    --use_comms_channel \
    --comm_coeff "$COMM_COEFF" \
    --num_messages "$NUM_MESSAGES" \
    2>&1 | tee "$LOG_FILE"

# =============================================================================
# EXPERIMENT COMPLETE
# =============================================================================

echo ""
echo "=== Experiment Complete ==="
echo "Results saved in: $RUN_DIR/"
echo "============================"