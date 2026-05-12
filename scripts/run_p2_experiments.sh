#!/usr/bin/env bash
# run_p2_experiments.sh — Resource-aware experiment launcher for P2 ablations.
#
# Dynamically throttles parallelism based on current CPU load so your laptop
# remains usable while experiments run in the background.
#
# Resource policy (configurable via env vars):
#   MAX_PARALLEL   — hard ceiling on simultaneous Python processes (default: 2)
#   LOAD_PER_CORE  — target fraction of per-core utilisation (default: 0.50)
#   POLL_SECS      — seconds between load checks (default: 30)
#   CONDA_ENV      — conda environment name (default: marl_comms)
#   LOG_DIR        — root log directory for all runs (default: runs/sc_ablation)
#   SEEDS          — space-separated seed list (default: "0 1 2 3 4")
#   STAGES         — space-separated stage list (default: all stages)
#
# Usage:
#   cd /path/to/on-policy
#   bash scripts/run_p2_experiments.sh
#
#   # Run only specific stages:
#   STAGES="BASELINE SC_MAIN" bash scripts/run_p2_experiments.sh
#
#   # Dry run (print commands, don't execute):
#   DRY_RUN=1 bash scripts/run_p2_experiments.sh
#
#   # Aggressive — use up to 3 parallel slots (only if idle enough):
#   MAX_PARALLEL=3 bash scripts/run_p2_experiments.sh
#
# Skipping completed runs:
#   The Python runner (run_sc_experiments.py) skips any run whose
#   metrics.csv already has >= 95% of the expected update rows.
#   Safe to re-run this script after interruption.
#
# Log file:
#   All output is tee'd to scripts/experiment_log_<timestamp>.txt
#
# Debug tips:
#   - If a run hangs silently, check the per-run log:
#       tail -f runs/sc_ablation/<exp_name>/<seed>/train.log
#   - If KMP error appears, KMP_DUPLICATE_LIB_OK=TRUE is already set below.
#   - If "conda: command not found", set CONDA_CMD to the full conda path.

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOAD_PER_CORE="${LOAD_PER_CORE:-0.50}"
POLL_SECS="${POLL_SECS:-30}"
CONDA_ENV="${CONDA_ENV:-marl_comms}"
LOG_DIR="${LOG_DIR:-runs/sc_ablation}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DRY_RUN="${DRY_RUN:-0}"

# All stages in dependency order (BASELINE must precede PARETO for fair comparison)
ALL_STAGES="BASELINE SC_MAIN SC_TWOPHASE PARETO GEOMETRY LIVE DLM_CMP"
STAGES="${STAGES:-$ALL_STAGES}"

# Locate conda — try common paths on macOS
if command -v conda &>/dev/null; then
    CONDA_CMD="conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/anaconda3/bin/conda"
else
    echo "[ERROR] conda not found. Set CONDA_CMD= or add conda to PATH." >&2
    exit 1
fi

# ─── Derived paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$SCRIPT_DIR/experiment_log_${TIMESTAMP}.txt"

mkdir -p "$LOG_DIR"

# ─── Helpers ────────────────────────────────────────────────────────────────

n_physical_cores() {
    sysctl -n hw.physicalcpu 2>/dev/null || echo 4
}

# Returns 1-minute load average as an integer × 100 (avoids float in bash).
# e.g. load=1.87 → 187
load_avg_x100() {
    local raw
    raw=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    # Multiply by 100 and round to integer
    awk "BEGIN {printf \"%d\", $raw * 100}"
}

# Maximum concurrent slots given current load.
# slots = max(1, min(MAX_PARALLEL, floor((LOAD_PER_CORE * ncores - load_now) / load_per_job)))
# where load_per_job ≈ 1 core.  Returns 0 if system is already loaded.
available_slots() {
    local ncores load_x100 target_x100 free_x100
    ncores=$(n_physical_cores)
    load_x100=$(load_avg_x100)
    # target_load = LOAD_PER_CORE * ncores, in ×100 units
    target_x100=$(awk "BEGIN {printf \"%d\", $LOAD_PER_CORE * $ncores * 100}")
    free_x100=$(( target_x100 - load_x100 ))

    # Each job uses ~1 core worth of load.  free_x100 / 100 = free cores.
    local free_jobs=$(( free_x100 / 100 ))

    # Clamp to [0, MAX_PARALLEL]
    if [ "$free_jobs" -lt 0 ]; then free_jobs=0; fi
    if [ "$free_jobs" -gt "$MAX_PARALLEL" ]; then free_jobs=$MAX_PARALLEL; fi
    echo "$free_jobs"
}

# Count currently running experiment processes launched by this script.
count_running() {
    # Count background jobs still alive
    local n=0
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            n=$(( n + 1 ))
        fi
    done
    echo "$n"
}

# Remove finished PIDs from the array.
reap_finished() {
    local alive=()
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        else
            # Collect exit status (non-blocking)
            wait "$pid" 2>/dev/null || true
        fi
    done
    PIDS=("${alive[@]:-}")
}

# Wait until a slot is free, sleeping POLL_SECS between checks.
wait_for_slot() {
    while true; do
        reap_finished
        local running slots
        running=$(count_running)
        slots=$(available_slots)
        if [ "$running" -lt "$MAX_PARALLEL" ] && [ "$slots" -gt 0 ]; then
            return
        fi
        local load
        load=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
        echo "[throttle] load=$load  running=$running  max=$MAX_PARALLEL  slots=$slots — waiting ${POLL_SECS}s..."
        sleep "$POLL_SECS"
    done
}

# Launch one experiment stage (one Python call per stage×seed combination).
launch_stage_seed() {
    local stage="$1"
    local seed="$2"
    local run_log_path="$LOG_DIR/stage_${stage}_seed${seed}_${TIMESTAMP}.log"

    local cmd_prefix="KMP_DUPLICATE_LIB_OK=TRUE $CONDA_CMD run -n $CONDA_ENV --no-capture-output"
    local python_cmd="python -m onpolicy.envs.toyproblem.experiments.run_sc_experiments \
        --stage $stage \
        --seeds $seed \
        --log_dir $LOG_DIR"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry] $cmd_prefix $python_cmd"
        return
    fi

    echo "[launch] stage=$stage seed=$seed → $run_log_path"
    # Run in background; redirect both stdout and stderr to per-run log
    (
        cd "$REPO_ROOT"
        eval "$cmd_prefix $python_cmd" > "$run_log_path" 2>&1
    ) &
    PIDS+=($!)
}

# ─── Main ───────────────────────────────────────────────────────────────────

# Redirect all output to log file while also printing to terminal
exec > >(tee -a "$RUN_LOG") 2>&1

echo "========================================================"
echo " P2 Experiment Runner  —  $(date)"
echo "========================================================"
echo " Repo:        $REPO_ROOT"
echo " Log dir:     $LOG_DIR"
echo " Conda env:   $CONDA_ENV"
echo " Stages:      $STAGES"
echo " Seeds:       $SEEDS"
echo " Max parallel: $MAX_PARALLEL"
echo " Load target:  ${LOAD_PER_CORE} per core ($(n_physical_cores) cores)"
echo " Log file:    $RUN_LOG"
echo "========================================================"

PIDS=()

for stage in $STAGES; do
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " Stage: $stage"
    echo "──────────────────────────────────────────────────────"

    for seed in $SEEDS; do
        # Block until we have a free slot
        if [ "$DRY_RUN" != "1" ]; then
            wait_for_slot
        fi
        launch_stage_seed "$stage" "$seed"
    done
done

# Wait for all remaining background jobs
if [ "${#PIDS[@]}" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo ""
    echo "[wait] All stages launched. Waiting for ${#PIDS[@]} remaining jobs..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
fi

echo ""
echo "========================================================"
echo " All experiments complete — $(date)"
echo " Results: $LOG_DIR"
echo " Full log: $RUN_LOG"
echo "========================================================"
