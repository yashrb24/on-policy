#!/usr/bin/env bash
# Eval all distilled students + the s12 teacher with the SCoUT sampled protocol
# (eval_student.sh passes --eval_deterministic which INVERTS to stochastic sampling).
# Usage: eval_all.sh <GPU> [NSEEDS]   (default GPU 0, 20 seeds)
# Assumes every checkpoint is present locally (copy anjuna3 ones over first).
set -uo pipefail
GPU=${1:-0}; NSEEDS=${2:-20}
cd "$(dirname "$0")/../../.."
EV=onpolicy/scripts/distill/eval_student.sh
RES=onpolicy/scripts/results/Pursuit/rmappo
run () { echo -n "[$1 params?] "; bash "$EV" "$1" "$2" "$3" "$GPU" "$4" "$NSEEDS" || echo "RESULT $1 | EVAL_FAILED"; }

# exp        n_embd n_block  model_dir
run dst_w64b2 64 2 $RES/dst_w64b2/run1/models
run dst_w64b1 64 1 $RES/dst_w64b1/run1/models
run dst_w48b3 48 3 $RES/dst_w48b3/run1/models
run dst_w48b2 48 2 $RES/dst_w48b2/run1/models
run dst_w48b1 48 1 $RES/dst_w48b1/run1/models
run dst_w32b3 32 3 $RES/dst_w32b3/run1/models
run dst_w32b2 32 2 $RES/dst_w32b2/run1/models
run dst_w24b2 24 2 $RES/dst_w24b2/run1/models
# teacher reference (expect ~100%)
bash "$EV" teacher_s12 64 3 "$GPU" /home/pfs/notyash/on-policy/c1_lam90_s12_models "$NSEEDS" || echo "RESULT teacher_s12 | EVAL_FAILED"
