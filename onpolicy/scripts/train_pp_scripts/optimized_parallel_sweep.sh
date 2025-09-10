#!/bin/bash

# Optimized parallel hyperparameter sweep script
# Uses configuration file for easy customization

# =============================================================================
# SETUP AND INITIALIZATION
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Source configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/sweep_config.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found at $CONFIG_FILE${NC}"
    echo "Please create sweep_config.sh in the same directory as this script"
    exit 1
fi

source "$CONFIG_FILE"

# Calculate system limits
if [ -z "$MAX_PARALLEL_JOBS" ]; then
    MAX_PARALLEL_JOBS=$((TOTAL_CPU_CORES / THREADS_PER_EXPERIMENT))
fi

if [ $MAX_PARALLEL_JOBS -eq 0 ]; then
    MAX_PARALLEL_JOBS=1
    echo -e "${YELLOW}Warning: System has fewer cores than threads per experiment. Running sequentially.${NC}"
fi

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

# Create necessary directories
mkdir -p logs pids results configs

# Save current configuration for reproducibility
cp "$CONFIG_FILE" "configs/sweep_config_$(date +%Y%m%d_%H%M%S).sh"

# Job management files
JOB_QUEUE="job_queue.txt"
ACTIVE_JOBS="active_jobs.txt"
COMPLETED_JOBS="completed_jobs.txt"
FAILED_JOBS="failed_jobs.txt"

# Clean up previous runs
rm -f "$JOB_QUEUE" "$ACTIVE_JOBS" "$COMPLETED_JOBS" "$FAILED_JOBS"
touch "$JOB_QUEUE" "$ACTIVE_JOBS" "$COMPLETED_JOBS" "$FAILED_JOBS"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${BLUE}[$timestamp INFO]${NC} $message" ;;
        "WARN")  echo -e "${YELLOW}[$timestamp WARN]${NC} $message" ;;
        "ERROR") echo -e "${RED}[$timestamp ERROR]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[$timestamp SUCCESS]${NC} $message" ;;
        *) echo "[$timestamp] $message" ;;
    esac
    
    # Also log to file
    echo "[$timestamp $level] $message" >> "sweep_execution.log"
}

# Function to get available GPU for job
get_gpu_for_job() {
    local job_id=$1
    local gpu_index=$((job_id % ${#AVAILABLE_GPUS[@]}))
    echo "${AVAILABLE_GPUS[$gpu_index]}"
}

# Function to validate architecture
validate_architecture() {
    local n_embd=$1
    local n_head=$2
    
    if [ $((n_embd % n_head)) -eq 0 ] && [ $n_head -le $n_embd ]; then
        return 0
    else
        return 1
    fi
}

# Function to check system resources
check_system_resources() {
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    local mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    local load_avg=$(uptime | awk '{print $10}' | sed 's/,//')
    
    log_message "INFO" "System status - CPU: ${cpu_usage}%, Memory: ${mem_usage}%, Load: ${load_avg}"
    
    # Warning thresholds
    if (( $(echo "$mem_usage > 90" | bc -l) )); then
        log_message "WARN" "High memory usage detected: ${mem_usage}%"
    fi
    
    if (( $(echo "$load_avg > $TOTAL_CPU_CORES" | bc -l) )); then
        log_message "WARN" "High CPU load detected: ${load_avg}"
    fi
}

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

# Function to run a single experiment
run_experiment() {
    local job_id=$1
    local config="$2"
    
    # Parse configuration
    IFS=' ' read -ra params <<< "$config"
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
        log_message "ERROR" "Job $job_id: Invalid architecture n_embd=$n_embd, n_head=$n_head"
        echo "$job_id" >> "$FAILED_JOBS"
        return 1
    fi
    
    local gpu_id=$(get_gpu_for_job $job_id)
    local exp_name="${EXPERIMENT_PREFIX}_j${job_id}_lr${lr}_ec${entropy_coef}_cp${clip_param}_nb${n_block}_ne${n_embd}_nh${n_head}_s${seed}"
    local log_file="logs/${exp_name}.log"
    local pid_file="pids/${job_id}.pid"
    local result_file="results/${exp_name}.result"
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    log_message "INFO" "Starting Job $job_id on GPU $gpu_id: $exp_name"
    
    # Create job metadata
    cat > "results/${exp_name}.meta" << EOF
job_id=$job_id
gpu_id=$gpu_id
start_time=$start_time
config=$config
lr=$lr
critic_lr=$critic_lr
entropy_coef=$entropy_coef
clip_param=$clip_param
n_block=$n_block
n_embd=$n_embd
n_head=$n_head
ppo_epoch=$ppo_epoch
num_mini_batch=$num_mini_batch
max_grad_norm=$max_grad_norm
seed=$seed
EOF
    
    # Run experiment in background
    (
        # Set process limits if needed
        # ulimit -v 8000000  # Limit virtual memory to 8GB
        
        # Handle WandB with fallback
        local wandb_args=""
        if [ "$USE_WANDB" = "true" ]; then
            # Test WandB connection first
            if python3 -c "import wandb; wandb.login()" &>/dev/null; then
                wandb_args="--user_name $WANDB_USER --wandb_name $WANDB_PROJECT"
                log_message "INFO" "Job $job_id: Using WandB online mode"
            elif [ "$WANDB_FALLBACK_TO_OFFLINE" = "true" ]; then
                wandb_args="--user_name $WANDB_USER --wandb_name $WANDB_PROJECT"
                export WANDB_MODE=offline
                log_message "WARN" "Job $job_id: Using WandB offline mode"
            else
                wandb_args="--use_wandb"
                log_message "WARN" "Job $job_id: Disabling WandB due to connection issues"
            fi
        else
            wandb_args="--use_wandb"
        fi

        CUDA_VISIBLE_DEVICES=$gpu_id python ../train/train_predatorprey.py \
            --env_name "$ENV_NAME" \
            --scenario_name "$SCENARIO_NAME" \
            --algorithm_name "$ALGORITHM_NAME" \
            --experiment_name "$exp_name" \
            --seed "$seed" \
            --num_agents "$NUM_AGENTS" \
            --num_env_steps "$NUM_ENV_STEPS" \
            --episode_length "$EPISODE_LENGTH" \
            --dim "$DIM" \
            --vision "$VISION" \
            --n_rollout_threads "$N_ROLLOUT_THREADS" \
            --save_interval "$SAVE_INTERVAL" \
            --log_interval "$LOG_INTERVAL" \
            --use_transformer_base_actor \
            --hidden_size "$n_embd" \
            --lr "$lr" \
            --critic_lr "$critic_lr" \
            --ppo_epoch "$ppo_epoch" \
            --clip_param "$clip_param" \
            --num_mini_batch "$num_mini_batch" \
            --entropy_coef "$entropy_coef" \
            --max_grad_norm "$max_grad_norm" \
            --n_block "$n_block" \
            --n_embd "$n_embd" \
            --n_head "$n_head" \
            $wandb_args \
            2>&1 | tee "$log_file"
        
        local exit_code=${PIPESTATUS[0]}
        local end_time=$(date '+%Y-%m-%d %H:%M:%S')
        
        # Calculate runtime
        local start_epoch=$(date -d "$start_time" +%s)
        local end_epoch=$(date -d "$end_time" +%s)
        local runtime=$((end_epoch - start_epoch))
        
        # Write result file
        cat > "$result_file" << EOF
job_id=$job_id
exit_code=$exit_code
start_time=$start_time
end_time=$end_time
runtime_seconds=$runtime
gpu_id=$gpu_id
wandb_project=$WANDB_PROJECT
wandb_run_name=$unique_run_name
config=$config
EOF
        
        # Clean up isolated WandB cache
        rm -rf "$wandb_cache_dir"
        
        # Update job tracking
        grep -v "^$job_id$" "$ACTIVE_JOBS" > "${ACTIVE_JOBS}.tmp" && mv "${ACTIVE_JOBS}.tmp" "$ACTIVE_JOBS"
        echo "$job_id" >> "$COMPLETED_JOBS"
        
        if [ $exit_code -eq 0 ]; then
            log_message "SUCCESS" "Job $job_id completed successfully (${runtime}s)"
        else
            log_message "ERROR" "Job $job_id failed with exit code $exit_code (${runtime}s)"
            echo "$job_id" >> "$FAILED_JOBS"
        fi
        
        # Clean up
        rm -f "$pid_file"
        
    ) &
    
    local pid=$!
    echo $pid > "$pid_file"
    echo "$job_id" >> "$ACTIVE_JOBS"
    
    return 0
}

# Function to wait for available job slot
wait_for_slot() {
    while [ $(wc -l < "$ACTIVE_JOBS") -ge $MAX_PARALLEL_JOBS ]; do
        log_message "INFO" "Waiting for job slot... ($(wc -l < "$ACTIVE_JOBS")/$MAX_PARALLEL_JOBS active)"
        sleep 10
        
        # Clean up completed jobs
        cleanup_completed_jobs
        
        # Check system resources periodically
        check_system_resources
    done
}

# Function to clean up completed jobs
cleanup_completed_jobs() {
    for pid_file in pids/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ! kill -0 $pid 2>/dev/null; then
                # Process has finished, cleanup will be handled by the background process
                continue
            fi
        fi
    done
}

# =============================================================================
# CONFIGURATION GENERATION AND VALIDATION
# =============================================================================

# Generate experiment configurations
log_message "INFO" "Generating experiment configurations using strategy: $STRATEGY"

# Show configuration summary
print_config_summary

# Generate configurations
mapfile -t all_configs < <(generate_all_configs | grep -v '^#')

# Filter out empty lines and validate
validated_configs=()
for config in "${all_configs[@]}"; do
    if [ -n "$config" ] && validate_config "$config"; then
        validated_configs+=("$config")
    fi
done

total_experiments=${#validated_configs[@]}

if [ $total_experiments -eq 0 ]; then
    log_message "ERROR" "No valid configurations generated!"
    exit 1
fi

log_message "INFO" "Generated $total_experiments valid experiment configurations"

# Show runtime estimation
estimate_runtime $total_experiments $MAX_PARALLEL_JOBS

# Ask for confirmation
echo ""
read -p "Do you want to proceed with $total_experiments experiments? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_message "INFO" "Sweep cancelled by user"
    exit 0
fi

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================

log_message "INFO" "Starting parallel hyperparameter sweep"
echo "Total experiments: $total_experiments"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
echo ""

# Populate job queue
job_id=1
for config in "${validated_configs[@]}"; do
    echo "$job_id:$config" >> "$JOB_QUEUE"
    ((job_id++))
done

# Start execution timer
sweep_start_time=$(date +%s)

# Process job queue
while IFS=':' read -r job_id config; do
    wait_for_slot
    
    log_message "INFO" "Queuing Job $job_id ($(wc -l < "$ACTIVE_JOBS")/$MAX_PARALLEL_JOBS active, $(wc -l < "$COMPLETED_JOBS") completed)"
    
    run_experiment "$job_id" "$config"
    
    # Brief pause between job starts to avoid overwhelming the system
    sleep 2
    
done < "$JOB_QUEUE"

# Wait for all remaining jobs to complete
log_message "INFO" "All jobs queued. Waiting for completion..."
while [ $(wc -l < "$ACTIVE_JOBS") -gt 0 ]; do
    local remaining=$(wc -l < "$ACTIVE_JOBS")
    local completed=$(wc -l < "$COMPLETED_JOBS")
    local failed=$(wc -l < "$FAILED_JOBS")
    
    log_message "INFO" "Progress: $completed/$total_experiments completed, $failed failed, $remaining active"
    sleep 30
    cleanup_completed_jobs
done

# =============================================================================
# FINAL SUMMARY AND CLEANUP
# =============================================================================

sweep_end_time=$(date +%s)
total_runtime=$((sweep_end_time - sweep_start_time))

log_message "SUCCESS" "Parallel hyperparameter sweep completed!"

# Generate final summary
final_completed=$(wc -l < "$COMPLETED_JOBS")
final_failed=$(wc -l < "$FAILED_JOBS")
final_success=$((final_completed - final_failed))

echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo "Total experiments: $total_experiments"
echo "Successful: $final_success"
echo "Failed: $final_failed"
echo "Total runtime: $(($total_runtime / 3600))h $(($total_runtime % 3600 / 60))m"
echo "Average time per experiment: $(($total_runtime / $final_completed))s"
echo ""
echo "Results saved in:"
echo "  - Logs: logs/"
echo "  - Results: results/"
echo "  - Execution log: sweep_execution.log"
echo ""
echo "Check your wandb dashboard for detailed results comparison"
echo "Use monitor script: ./monitor_parallel.sh summary"
echo "=========================================="

# Create final summary report
cat > "final_summary.txt" << EOF
Hyperparameter Sweep Summary
============================
Date: $(date)
Strategy: $STRATEGY
Total experiments: $total_experiments
Successful: $final_success
Failed: $final_failed
Success rate: $(echo "scale=2; $final_success * 100 / $total_experiments" | bc -l)%
Total runtime: $(($total_runtime / 3600))h $(($total_runtime % 3600 / 60))m
Average time per experiment: $(($total_runtime / $final_completed))s

System Configuration:
- CPU cores: $TOTAL_CPU_CORES
- Max parallel jobs: $MAX_PARALLEL_JOBS
- GPUs used: ${AVAILABLE_GPUS[*]}

Failed experiments (if any):
EOF

if [ $final_failed -gt 0 ]; then
    while read -r job_id; do
        echo "  Job $job_id" >> "final_summary.txt"
    done < "$FAILED_JOBS"
fi

log_message "INFO" "Summary saved to final_summary.txt"
