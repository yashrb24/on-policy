#!/usr/bin/env bash
# Train one distilled student. Usage:
#   run_student.sh <EXP> <N_EMBD> <N_BLOCK> <GPU> <DATA_PATH> [EPOCHS] [LR] [BATCH_EPS]
# n_head fixed at 4 (divides 64/48/32/24). hidden_size == n_embd (invariant).
set -euo pipefail
EXP=$1; N_EMBD=$2; N_BLOCK=$3; GPU=$4; DATA=$5
EPOCHS=${6:-40}; LR=${7:-5e-4}; BATCH=${8:-8}
cd "$(dirname "$0")/../../.."   # repo root
PY=${PYTHON:-/usr/bin/python3}   # anjuna3: PYTHON=.venv/bin/python (its /usr/bin/python3 lacks deps)
NH=${NHEAD:-4}                   # n_head must divide n_embd; for w6 use NHEAD=2/3, etc.
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD $PY onpolicy/scripts/distill/train_distill.py \
  --env_name Pursuit --algorithm_name rmappo --experiment_name "$EXP" \
  --n_pursuers 20 --n_evaders 8 --x_size 40 --y_size 40 --max_cycles 500 --episode_length 500 \
  --catch_reward 5.0 --tag_reward 0.0 --urgency_reward 0.0 \
  --n_head "$NH" --n_block "$N_BLOCK" --n_embd "$N_EMBD" --hidden_size "$N_EMBD" \
  --use_transformer_base_actor --pursuit_drop_walls_channel \
  --data_path "$DATA" \
  --distill_epochs "$EPOCHS" --distill_lr "$LR" --distill_batch_episodes "$BATCH" \
  --val_frac 0.1 --seed 1
