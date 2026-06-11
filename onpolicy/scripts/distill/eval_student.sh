#!/usr/bin/env bash
# Eval one actor with the SCoUT protocol (SAMPLING -- pass --eval_deterministic which
# inverts to stochastic; the teacher is ~100% sampled vs ~35% argmax). Usage:
#   eval_student.sh <EXP> <N_EMBD> <N_BLOCK> <GPU> <MODEL_DIR> [N_SEEDS]
set -euo pipefail
EXP=$1; N_EMBD=$2; N_BLOCK=$3; GPU=$4; MODEL_DIR=$5; NSEEDS=${6:-20}
cd "$(dirname "$0")/../../.."   # repo root
PY=${PYTHON:-/usr/bin/python3}
NH=${NHEAD:-4}                   # must match the n_head the student was trained with
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD $PY onpolicy/scripts/train/eval_pursuit_seeds.py \
  --env_name Pursuit --algorithm_name rmappo --experiment_name "$EXP" \
  --n_pursuers 20 --n_evaders 8 --x_size 40 --y_size 40 --max_cycles 500 --episode_length 500 \
  --n_head "$NH" --n_block "$N_BLOCK" --n_embd "$N_EMBD" --hidden_size "$N_EMBD" \
  --use_transformer_base_actor --pursuit_drop_walls_channel \
  --eval_deterministic --n_eval_rollout_threads "$NSEEDS" \
  --model_dir "$MODEL_DIR" 2>&1 | grep -E "^RESULT"
