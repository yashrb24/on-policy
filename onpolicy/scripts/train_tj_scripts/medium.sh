#!/bin/bash

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base configuration
ENV_NAME="TrafficJunction"
DIFFICULTY="medium"
ALGORITHM_NAME="rmappo"
# "tj-medium-best-config"/ "tj-medium-ddcl-config"/ "tj-medium-fakequant-config"/ "tj-medium-delta-config"
EXPERIMENT_NAME="tj-medium-fakequant-config"
NUM_AGENTS=10
NUM_ENV_STEPS=120000
EPISODE_LENGTH=40
DIM=14
VISION=0
ADD_RATE_MIN=0.05
ADD_RATE_MAX=0.02
CURR_START=250
CURR_END=1250
N_ROLLOUT_THREADS=1
PPO_EPOCH=10
NUM_MINI_BATCH=1
SAVE_INTERVAL=200
LOG_INTERVAL=400
WANDB_USER="yashrb"
WANDB_PROJECT="on-policy"

# Best hyperparameters
BEST_LR="1e-3"
BEST_N_BLOCK="2"
BEST_N_EMBD="64"
BEST_N_HEAD="4"
BEST_HIDDEN_SIZE="64"  # Must match n_embd

# DDCL Configuration
COMM_COEFF="1e-4"  # Options: 1e-4/ 1e-3/ 1e-2
NUM_MESSAGES="15"  # Options: 10/15/20
# Uncomment these lines to enable DDCL:
# --use_comms_channel \
# --comm_coeff "$COMM_COEFF" \
# --num_messages "$NUM_MESSAGES" \

# Fake Quantization Configuration
# QUANT_BITS="8"  # Options: 4/8/16
# --use_fake_quantization is active below; comment it out to disable

# Seed to run
SEED=1
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

EXP_NAME="${EXPERIMENT_NAME}"
EXP_NAME_WITH_SEED="${EXPERIMENT_NAME}_seed${SEED}"
LOG_FILE="${RUN_DIR}/logs/${EXP_NAME_WITH_SEED}.log"

echo "=== Starting Experiment ==="
echo "Experiment name: $EXP_NAME_WITH_SEED"
echo "Seed: $SEED"
echo "Log file: $LOG_FILE"
echo "WandB project: $WANDB_PROJECT"
echo ""

# Run the experiment
python ../train/train_traffic_junction.py \
    --env_name "$ENV_NAME" \
    --difficulty "$DIFFICULTY" \
    --algorithm_name "$ALGORITHM_NAME" \
    --experiment_name "$EXP_NAME" \
    --seed "$SEED" \
    --num_agents "$NUM_AGENTS" \
    --num_env_steps "$NUM_ENV_STEPS" \
    --episode_length "$EPISODE_LENGTH" \
    --dim "$DIM" \
    --vision "$VISION" \
    --add_rate_min "$ADD_RATE_MIN" \
    --add_rate_max "$ADD_RATE_MAX" \
    --curr_start "$CURR_START" \
    --curr_end "$CURR_END" \
    --n_rollout_threads "$N_ROLLOUT_THREADS" \
    --ppo_epoch "$PPO_EPOCH" \
    --num_mini_batch "$NUM_MINI_BATCH" \
    --save_interval "$SAVE_INTERVAL" \
    --log_interval "$LOG_INTERVAL" \
    --use_transformer_base_actor \
    --use_fake_quantization \
    --hidden_size "$BEST_HIDDEN_SIZE" \
    --n_block "$BEST_N_BLOCK" \
    --n_embd "$BEST_N_EMBD" \
    --n_head "$BEST_N_HEAD" \
    --lr "$BEST_LR" \
    --user_name "$WANDB_USER" \
    --wandb_name "$WANDB_PROJECT" \
    2>&1 | tee "$LOG_FILE"

# =============================================================================
# EXPERIMENT COMPLETE
# =============================================================================

echo ""
echo "=== Experiment Complete ==="
echo "Results saved in: $RUN_DIR/"
echo "============================"
