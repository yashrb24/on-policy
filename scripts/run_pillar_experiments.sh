#!/usr/bin/env bash
# run_pillar_experiments.sh — Launch Pillar experiments exactly MAX_PARALLEL at a time.
#
# Each "slot" is one individual training run (one config × one seed), not a stage batch.
# The script first collects all pending commands via --dry_run (which already skips done
# runs via _is_done()), then works through the list MAX_PARALLEL at a time.
#
# Usage:
#   cd /path/to/on-policy
#   bash scripts/run_pillar_experiments.sh
#
# Env overrides:
#   MAX_PARALLEL=2       concurrent training runs (default: 2)
#   STAGES="P1_DELTA P4_RB"   stages to run (default: P1_DELTA P4_RB)
#   LOG_DIR=runs/sc_ablation  output directory
#   CONDA_ENV=marl_comms
#   DRY_RUN=1            print commands only, do not execute
#
# Resume safety:
#   _is_done() in the Python runner checks metrics.csv row count. Interrupted runs
#   with < 95% of expected rows are re-queued automatically.

set -euo pipefail

MAX_PARALLEL="${MAX_PARALLEL:-2}"
STAGES="${STAGES:-P1_DELTA P4_RB}"
LOG_DIR="${LOG_DIR:-runs/sc_ablation}"
CONDA_ENV="${CONDA_ENV:-marl_comms}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$SCRIPT_DIR/pillar_experiments_${TIMESTAMP}.log"

# Locate conda
if command -v conda &>/dev/null; then
    CONDA_CMD="conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/anaconda3/bin/conda"
else
    echo "[ERROR] conda not found." >&2; exit 1
fi

CONDA_RUN="KMP_DUPLICATE_LIB_OK=TRUE $CONDA_CMD run -n $CONDA_ENV --no-capture-output"

# ── Step 1: collect all pending commands from dry-run ─────────────────────────
echo "Collecting pending experiment commands..." | tee -a "$MASTER_LOG"

CMDS=()
for stage in $STAGES; do
    while IFS= read -r line; do
        # The dry-run prints command lines indented with 9 spaces starting with "onpolicy"
        trimmed="${line#"${line%%[! ]*}"}"   # lstrip spaces
        if [[ "$trimmed" == onpolicy.envs.toyproblem.train* ]]; then
            CMDS+=("$trimmed")
        fi
    done < <(
        cd "$REPO_ROOT"
        eval "$CONDA_RUN python -m onpolicy.envs.toyproblem.experiments.run_sc_experiments \
            --stage $stage --dry_run --log_dir $LOG_DIR" 2>/dev/null
    )
done

TOTAL=${#CMDS[@]}
echo "Found $TOTAL pending runs across stages: $STAGES" | tee -a "$MASTER_LOG"
echo "Running $MAX_PARALLEL at a time.  Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "==========================================================" | tee -a "$MASTER_LOG"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to run — all experiments already done."
    exit 0
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] Would run the following $TOTAL commands:"
    for cmd in "${CMDS[@]}"; do
        echo "  python -m $cmd"
    done
    exit 0
fi

# ── Step 2: pool — run MAX_PARALLEL at a time ─────────────────────────────────
PIDS=()
RUN_IDX=0

reap_finished() {
    local alive=()
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        else
            wait "$pid" 2>/dev/null || true
        fi
    done
    PIDS=("${alive[@]:-}")
}

wait_for_slot() {
    while [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
        sleep 5
        reap_finished
    done
}

mkdir -p "$REPO_ROOT/$LOG_DIR"

for cmd in "${CMDS[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    wait_for_slot

    # Extract exp_name and seed from the command for the log filename
    exp_name=$(echo "$cmd" | grep -o '\-\-exp_name [^ ]*' | awk '{print $2}')
    seed=$(echo "$cmd" | grep -o '\-\-seed [^ ]*' | awk '{print $2}')
    run_log="$REPO_ROOT/$LOG_DIR/${exp_name}/${seed}/train.log"

    mkdir -p "$(dirname "$run_log")"

    echo "[$RUN_IDX/$TOTAL] launch  $exp_name  seed=$seed" | tee -a "$MASTER_LOG"
    (
        cd "$REPO_ROOT"
        eval "$CONDA_RUN python -m $cmd" >"$run_log" 2>&1
        echo "[$RUN_IDX/$TOTAL] done    $exp_name  seed=$seed  exit=$?" >> "$MASTER_LOG"
    ) &
    PIDS+=($!)
done

# Wait for remaining jobs
echo "" | tee -a "$MASTER_LOG"
echo "All $TOTAL commands dispatched. Waiting for final jobs..." | tee -a "$MASTER_LOG"
for pid in "${PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
done

echo "==========================================================" | tee -a "$MASTER_LOG"
echo "All done — $(date)" | tee -a "$MASTER_LOG"
echo "Logs: $REPO_ROOT/$LOG_DIR/<exp>/<seed>/train.log" | tee -a "$MASTER_LOG"
echo "Next: run aggregate_results.py then post_hoc_coding.py on E53–E58." | tee -a "$MASTER_LOG"
