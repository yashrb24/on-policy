#!/bin/bash

# Parallel execution of best hyperparameter configuration with multiple seeds
# Maximizes GPU and CPU utilization

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base configuration from your setup
ENV_NAME="PredatorPrey"
SCENARIO_NAME="hard"
ALGORITHM_NAME="rmappo"
# "pp-hard-best-config"/ "pp-hard-ddcl-config"/ "pp-hard-fakequant-config"/ "pp-hard-delta-config"
EXPERIMENT_NAME="pp-hard-delta-config"
NUM_AGENTS=10
NUM_ENV_STEPS=3000000
EPISODE_LENGTH=40
DIM=10
VISION=1
N_ROLLOUT_THREADS=15
PPO_EPOCH=10
NUM_MINI_BATCH=1
SAVE_INTERVAL=200
LOG_INTERVAL=400
WANDB_USER="yashrb"
WANDB_PROJECT="on-policy"

# Best hyperparameters
BEST_LR="1e-4"
BEST_CRITIC_LR="1e-4"
BEST_ENTROPY_COEF="0.001"
BEST_CLIP_PARAM="0.2"
BEST_N_BLOCK="2"
BEST_N_EMBD="64"
BEST_N_HEAD="4"
BEST_HIDDEN_SIZE="64"  # Must match n_embd

# DDCL 1e-4/ 1e-3/ 1e-2
COMM_COEFF="1e-4"
# 10/15/20
NUM_MESSAGES="20"
# --use_comms_channel \
# --comm_coeff "$COMM_COEFF" \
# --num_messages "$NUM_MESSAGES" \

# FAKEQUANT 4/8/16
# QUANT_BITS="8"
# --use_fake_quantization \
# --quant_bits "$QUANT_BITS" \

# Seeds to run
SEEDS=(8 12 18 35 41)

# =============================================================================
# SYSTEM OPTIMIZATION
# =============================================================================

# Detect available resources
TOTAL_CPU_CORES=$(nproc)
AVAILABLE_GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null))
NUM_GPUS=${#AVAILABLE_GPUS[@]}

# Calculate optimal parallel jobs
THREADS_PER_EXPERIMENT=15
CPU_LIMITED_JOBS=$((TOTAL_CPU_CORES / THREADS_PER_EXPERIMENT))
GPU_LIMITED_JOBS=$NUM_GPUS

# Use the most restrictive limit
if [ $CPU_LIMITED_JOBS -le $GPU_LIMITED_JOBS ]; then
    MAX_PARALLEL_JOBS=$CPU_LIMITED_JOBS
    LIMITING_FACTOR="CPU"
else
    MAX_PARALLEL_JOBS=$GPU_LIMITED_JOBS
    LIMITING_FACTOR="GPU"
fi

# Ensure we don't exceed number of seeds
if [ $MAX_PARALLEL_JOBS -gt ${#SEEDS[@]} ]; then
    MAX_PARALLEL_JOBS=${#SEEDS[@]}
fi

echo "=== System Resource Analysis ==="
echo "Total CPU cores: $TOTAL_CPU_CORES"
echo "Threads per experiment: $THREADS_PER_EXPERIMENT"
echo "CPU-limited jobs: $CPU_LIMITED_JOBS"
echo "Available GPUs: ${AVAILABLE_GPUS[*]} (count: $NUM_GPUS)"
echo "GPU-limited jobs: $GPU_LIMITED_JOBS"
echo "Limiting factor: $LIMITING_FACTOR"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Seeds to run: ${SEEDS[*]}"
echo "================================="
echo ""

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="best_config_seeds_${TIMESTAMP}"
mkdir -p "${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}/pids"
mkdir -p "${RUN_DIR}/results"

echo "Created run directory: $RUN_DIR"
echo ""

# =============================================================================
# EXPERIMENT EXECUTION FUNCTIONS
# =============================================================================

# Function to get GPU for a job
get_gpu_for_job() {
    local job_index=$1
    local gpu_index=$((job_index % NUM_GPUS))
    echo "${AVAILABLE_GPUS[$gpu_index]}"
}

# Function to run a single seed experiment
run_seed_experiment() {
    local seed=$1
    local job_index=$2
    local gpu_id=$(get_gpu_for_job $job_index)
    
    local exp_name="${EXPERIMENT_NAME}_seed${seed}"
    local log_file="${RUN_DIR}/logs/${exp_name}.log"
    local pid_file="${RUN_DIR}/pids/seed_${seed}.pid"
    local result_file="${RUN_DIR}/results/${exp_name}.result"
    
    # Add staggered startup delay for WandB
    local startup_delay=$((job_index * 2))
    
    echo "Starting seed $seed on GPU $gpu_id (delay: ${startup_delay}s)"
    
    (
        # Startup delay for WandB coordination
        sleep $startup_delay
        
        # Set isolated WandB cache
        export WANDB_CACHE_DIR="/tmp/wandb_cache_seed_${seed}"
        export WANDB_DATA_DIR="/tmp/wandb_cache_seed_${seed}"
        mkdir -p "/tmp/wandb_cache_seed_${seed}"
        
        local start_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$start_time] Starting experiment for seed $seed on GPU $gpu_id" | tee -a "$log_file"
        
        # Run the experiment
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
            2>&1 | tee -a "$log_file"
        
        local exit_code=${PIPESTATUS[0]}
        local end_time=$(date '+%Y-%m-%d %H:%M:%S')
        
        # Calculate runtime
        local start_epoch=$(date -d "$start_time" +%s)
        local end_epoch=$(date -d "$end_time" +%s)
        local runtime=$((end_epoch - start_epoch))
        
        # Write result
        cat > "$result_file" << EOF
seed=$seed
gpu_id=$gpu_id
start_time=$start_time
end_time=$end_time
runtime_seconds=$runtime
exit_code=$exit_code
experiment_name=$exp_name
parameters=lr:$BEST_LR,critic_lr:$BEST_CRITIC_LR,entropy_coef:$BEST_ENTROPY_COEF,clip_param:$BEST_CLIP_PARAM,n_block:$BEST_N_BLOCK,n_embd:$BEST_N_EMBD,n_head:$BEST_N_HEAD
EOF
        
        # Clean up
        rm -rf "/tmp/wandb_cache_seed_${seed}"
        rm -f "$pid_file"
        
        if [ $exit_code -eq 0 ]; then
            echo "✓ Seed $seed completed successfully (${runtime}s)"
        else
            echo "✗ Seed $seed failed (exit code: $exit_code, ${runtime}s)"
        fi
        
    ) &
    
    # Save PID
    echo $! > "$pid_file"
    return 0
}

# Function to wait for available job slot
wait_for_available_slot() {
    local active_jobs=$(ls "${RUN_DIR}/pids"/*.pid 2>/dev/null | wc -l)
    
    while [ $active_jobs -ge $MAX_PARALLEL_JOBS ]; do
        echo "Waiting for job slot... ($active_jobs/$MAX_PARALLEL_JOBS active)"
        sleep 5
        
        # Clean up completed jobs
        for pid_file in "${RUN_DIR}/pids"/*.pid; do
            if [ -f "$pid_file" ]; then
                local pid=$(cat "$pid_file")
                if ! kill -0 $pid 2>/dev/null; then
                    rm -f "$pid_file"
                fi
            fi
        done
        
        active_jobs=$(ls "${RUN_DIR}/pids"/*.pid 2>/dev/null | wc -l)
    done
}

# Function to show real-time status
show_status() {
    while [ $(ls "${RUN_DIR}/pids"/*.pid 2>/dev/null | wc -l) -gt 0 ]; do
        local active_jobs=$(ls "${RUN_DIR}/pids"/*.pid 2>/dev/null | wc -l)
        local completed_jobs=$(ls "${RUN_DIR}/results"/*.result 2>/dev/null | wc -l)
        local total_seeds=${#SEEDS[@]}
        
        echo "Status: $completed_jobs/$total_seeds completed, $active_jobs active"
        
        # Show GPU utilization if available
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status:"
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
                awk '{printf "  GPU %s: %s%% util, %sMB/%sMB mem\n", $1, $2, $3, $4}'
        fi
        
        echo "---"
        sleep 30
    done
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "Starting parallel execution of best configuration with ${#SEEDS[@]} seeds"
echo "Configuration: lr=$BEST_LR, entropy_coef=$BEST_ENTROPY_COEF, clip_param=$BEST_CLIP_PARAM"
echo "Architecture: n_block=$BEST_N_BLOCK, n_embd=$BEST_N_EMBD, n_head=$BEST_N_HEAD"
echo ""

# Save configuration for reference
cat > "${RUN_DIR}/config.txt" << EOF
# Best Configuration Seed Runs
# Timestamp: $(date)
# 
# Base Configuration:
ENV_NAME=$ENV_NAME
SCENARIO_NAME=$SCENARIO_NAME
ALGORITHM_NAME=$ALGORITHM_NAME
NUM_AGENTS=$NUM_AGENTS
NUM_ENV_STEPS=$NUM_ENV_STEPS
EPISODE_LENGTH=$EPISODE_LENGTH

# Best Hyperparameters:
BEST_LR=$BEST_LR
BEST_CRITIC_LR=$BEST_CRITIC_LR
BEST_ENTROPY_COEF=$BEST_ENTROPY_COEF
BEST_CLIP_PARAM=$BEST_CLIP_PARAM
BEST_N_BLOCK=$BEST_N_BLOCK
BEST_N_EMBD=$BEST_N_EMBD
BEST_N_HEAD=$BEST_N_HEAD

# Seeds:
SEEDS=(${SEEDS[*]})

# System Resources:
TOTAL_CPU_CORES=$TOTAL_CPU_CORES
AVAILABLE_GPUS=(${AVAILABLE_GPUS[*]})
MAX_PARALLEL_JOBS=$MAX_PARALLEL_JOBS
LIMITING_FACTOR=$LIMITING_FACTOR
EOF

# Start experiments
experiment_start_time=$(date +%s)
job_index=0

for seed in "${SEEDS[@]}"; do
    wait_for_available_slot
    
    echo "Launching seed $seed (job $((job_index + 1))/${#SEEDS[@]})"
    run_seed_experiment $seed $job_index
    
    ((job_index++))
    sleep 1  # Brief pause between launches
done

echo ""
echo "All seeds launched. Monitoring progress..."

# Show status in background
show_status &
status_pid=$!

# Wait for all experiments to complete
while [ $(ls "${RUN_DIR}/pids"/*.pid 2>/dev/null | wc -l) -gt 0 ]; do
    sleep 10
    
    # Clean up completed jobs
    for pid_file in "${RUN_DIR}/pids"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ! kill -0 $pid 2>/dev/null; then
                rm -f "$pid_file"
            fi
        fi
    done
done

# Stop status monitoring
kill $status_pid 2>/dev/null

# =============================================================================
# FINAL SUMMARY
# =============================================================================

experiment_end_time=$(date +%s)
total_runtime=$((experiment_end_time - experiment_start_time))

echo ""
echo "=== FINAL RESULTS ==="
echo "Total runtime: $(($total_runtime / 3600))h $(($total_runtime % 3600 / 60))m"
echo ""

# Analyze results
successful_seeds=()
failed_seeds=()

for seed in "${SEEDS[@]}"; do
    result_file="${RUN_DIR}/results/${EXPERIMENT_NAME}_seed${seed}.result"
    if [ -f "$result_file" ]; then
        exit_code=$(grep "exit_code=" "$result_file" | cut -d= -f2)
        runtime=$(grep "runtime_seconds=" "$result_file" | cut -d= -f2)
        
        if [ "$exit_code" = "0" ]; then
            successful_seeds+=($seed)
            echo "✓ Seed $seed: SUCCESS (${runtime}s)"
        else
            failed_seeds+=($seed)
            echo "✗ Seed $seed: FAILED (exit code: $exit_code, ${runtime}s)"
        fi
    else
        failed_seeds+=($seed)
        echo "✗ Seed $seed: NO RESULT FILE"
    fi
done

echo ""
echo "Summary:"
echo "  Successful: ${#successful_seeds[@]}/${#SEEDS[@]} seeds"
echo "  Failed: ${#failed_seeds[@]}/${#SEEDS[@]} seeds"
echo "  Success rate: $(echo "scale=1; ${#successful_seeds[@]} * 100 / ${#SEEDS[@]}" | bc -l)%"

if [ ${#successful_seeds[@]} -gt 0 ]; then
    echo "  Successful seeds: ${successful_seeds[*]}"
fi

if [ ${#failed_seeds[@]} -gt 0 ]; then
    echo "  Failed seeds: ${failed_seeds[*]}"
    echo ""
    echo "Check failed seed logs:"
    for seed in "${failed_seeds[@]}"; do
        echo "  tail ${RUN_DIR}/logs/${EXPERIMENT_NAME}_seed${seed}.log"
    done
fi

echo ""
echo "All results saved in: $RUN_DIR/"
echo "WandB project: $WANDB_PROJECT"
echo "Monitor logs: tail -f ${RUN_DIR}/logs/*.log"
echo "==================="
