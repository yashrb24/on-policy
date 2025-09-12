#!/bin/bash

# Monitor script for parallel seed runs
# Usage: ./monitor_seed_runs.sh [run_directory]

RUN_DIR=${1:-$(ls -td best_config_seeds_* | head -1)}

if [ ! -d "$RUN_DIR" ]; then
    echo "Run directory not found: $RUN_DIR"
    echo "Usage: $0 [run_directory]"
    exit 1
fi

SEEDS=(8 12 18 35 41)

show_status() {
    clear
    echo "=== Seed Run Monitor ==="
    echo "Directory: $RUN_DIR"
    echo "Time: $(date)"
    echo ""
    
    # Overall progress
    local active_jobs=$(ls "${RUN_DIR}/pids"/*.pid 2>/dev/null | wc -l)
    local completed_jobs=$(ls "${RUN_DIR}/results"/*.result 2>/dev/null | wc -l)
    local total_seeds=${#SEEDS[@]}
    
    echo "Progress: $completed_jobs/$total_seeds completed, $active_jobs active"
    echo ""
    
    # Individual seed status
    echo "Seed Status:"
    for seed in "${SEEDS[@]}"; do
        local pid_file="${RUN_DIR}/pids/seed_${seed}.pid"
        local result_file="${RUN_DIR}/results/pp-medium-best-config_seed${seed}.result"
        local log_file="${RUN_DIR}/logs/pp-medium-best-config_seed${seed}.log"
        
        if [ -f "$result_file" ]; then
            local exit_code=$(grep "exit_code=" "$result_file" 2>/dev/null | cut -d= -f2)
            local runtime=$(grep "runtime_seconds=" "$result_file" 2>/dev/null | cut -d= -f2)
            
            if [ "$exit_code" = "0" ]; then
                echo "  Seed $seed: ✓ COMPLETED (${runtime}s)"
            else
                echo "  Seed $seed: ✗ FAILED (exit code: $exit_code)"
            fi
        elif [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                # Get runtime so far
                local start_time=$(ps -p $pid -o lstart= | xargs -I {} date -d "{}" +%s 2>/dev/null)
                local current_time=$(date +%s)
                local elapsed=$((current_time - start_time))
                echo "  Seed $seed: ⏳ RUNNING (${elapsed}s, PID: $pid)"
            else
                echo "  Seed $seed: ❓ PROCESS DIED"
                rm -f "$pid_file"
            fi
        elif [ -f "$log_file" ]; then
            echo "  Seed $seed: 📋 LOG EXISTS (check for errors)"
        else
            echo "  Seed $seed: ⚪ NOT STARTED"
        fi
    done
    
    # GPU status
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
            while IFS=, read -r idx name util mem_used mem_total; do
                printf "  GPU %s (%s): %s%% util, %sMB/%sMB mem\n" "$idx" "$(echo $name | cut -c1-12)" "$util" "$mem_used" "$mem_total"
            done
    fi
    
    # System load
    echo ""
    echo "System Load:"
    local load_avg=$(uptime | awk '{print $10 $11 $12}')
    local mem_usage=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')
    echo "  Load average: $load_avg"
    echo "  Memory usage: $mem_usage"
    
    # Recent log activity
    echo ""
    echo "Recent Activity (last 5 lines from active logs):"
    for seed in "${SEEDS[@]}"; do
        local log_file="${RUN_DIR}/logs/pp-medium-best-config_seed${seed}.log"
        local pid_file="${RUN_DIR}/pids/seed_${seed}.pid"
        
        if [ -f "$pid_file" ] && [ -f "$log_file" ]; then
            echo "  Seed $seed:"
            tail -2 "$log_file" 2>/dev/null | sed 's/^/    /' | grep -v '^[[:space:]]*$' | head -2
        fi
    done
    
    echo ""
    echo "Press Ctrl+C to exit monitor"
}

# Function to show detailed log for specific seed
show_seed_log() {
    local seed=$1
    local log_file="${RUN_DIR}/logs/pp-medium-best-config_seed${seed}.log"
    
    if [ -f "$log_file" ]; then
        echo "=== Log for Seed $seed ==="
        tail -30 "$log_file"
    else
        echo "Log file not found for seed $seed"
    fi
}

# Function to show summary
show_summary() {
    echo "=== Run Summary ==="
    echo "Directory: $RUN_DIR"
    echo ""
    
    local successful=0
    local failed=0
    
    for seed in "${SEEDS[@]}"; do
        local result_file="${RUN_DIR}/results/pp-medium-best-config_seed${seed}.result"
        
        if [ -f "$result_file" ]; then
            local exit_code=$(grep "exit_code=" "$result_file" | cut -d= -f2)
            local runtime=$(grep "runtime_seconds=" "$result_file" | cut -d= -f2)
            local gpu_id=$(grep "gpu_id=" "$result_file" | cut -d= -f2)
            
            if [ "$exit_code" = "0" ]; then
                echo "✓ Seed $seed: SUCCESS (${runtime}s on GPU $gpu_id)"
                ((successful++))
            else
                echo "✗ Seed $seed: FAILED (exit code: $exit_code, ${runtime}s on GPU $gpu_id)"
                ((failed++))
            fi
        else
            echo "❓ Seed $seed: NO RESULT"
            ((failed++))
        fi
    done
    
    echo ""
    echo "Total: $successful successful, $failed failed"
    echo "Success rate: $(echo "scale=1; $successful * 100 / ${#SEEDS[@]}" | bc -l)%"
}

# Main execution
case "${2:-monitor}" in
    "log")
        if [ -z "$3" ]; then
            echo "Usage: $0 $RUN_DIR log <seed_number>"
            exit 1
        fi
        show_seed_log "$3"
        ;;
    "summary")
        show_summary
        ;;
    "monitor"|*)
        # Continuous monitoring
        trap 'echo ""; echo "Monitor stopped"; exit 0' INT
        while true; do
            show_status
            sleep 10
        done
        ;;
esac
