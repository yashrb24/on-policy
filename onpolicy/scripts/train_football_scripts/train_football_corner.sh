# #!/bin/sh
# # exp param
# env="Football"
# scenario="academy_corner"
# algo="rmappo" # "mappo" "ippo"
# exp="check"
# seed=1

# # football param
# num_agents=10

# # train param
# num_env_steps=50000000
# episode_length=1000

# echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

# CUDA_VISIBLE_DEVICES=0 python ../train/train_football.py \
# --env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
# --num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
# --representation "simple115v2" --rewards "scoring" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
# --save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
# --user_name "yashrb" --wandb_name "on-policy" 


#!/bin/bash

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base configuration
ENV_NAME="Football"
SCENARIO_NAME="academy_corner"
ALGORITHM_NAME="rmappo"
EXPERIMENT_NAME="vanilla-mappo_og_params-grf-corner"
NUM_AGENTS=3
NUM_ENV_STEPS=50000000
EPISODE_LENGTH=1000
N_ROLLOUT_THREADS=50
PPO_EPOCH=15
NUM_MINI_BATCH=2
SAVE_INTERVAL=100000
LOG_INTERVAL=3000
WANDB_USER="yashrb"
WANDB_PROJECT="ddcl-applications"

# Best hyperparameters
BEST_LR="5e-4"
BEST_CRITIC_LR="5e-4"
BEST_ENTROPY_COEF="0.01"
BEST_CLIP_PARAM="0.2"
BEST_N_BLOCK="1"
BEST_N_EMBD="64"
BEST_N_HEAD="4"
BEST_HIDDEN_SIZE="64"  # Must match n_embd
BEST_MAX_GRAD_NORM="10"

# DDCL Configuration
# COMM_COEFF="1e-4"  # Options: 1e-4/ 1e-3/ 1e-2
# NUM_MESSAGES="15"  # Options: 10/15/20
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
SEED=18
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
python ../train/train_football.py \
    --env_name "$ENV_NAME" \
    --scenario_name "$SCENARIO_NAME" \
    --algorithm_name "$ALGORITHM_NAME" \
    --experiment_name "$EXP_NAME" \
    --seed "$SEED" \
    --num_agents "$NUM_AGENTS" \
    --num_env_steps "$NUM_ENV_STEPS" \
    --episode_length "$EPISODE_LENGTH" \
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
    --max_grad_norm "$BEST_MAX_GRAD_NORM" \
    --entropy_coef "$BEST_ENTROPY_COEF" \
    --n_block "$BEST_N_BLOCK" \
    --n_embd "$BEST_N_EMBD" \
    --n_head "$BEST_N_HEAD" \
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
