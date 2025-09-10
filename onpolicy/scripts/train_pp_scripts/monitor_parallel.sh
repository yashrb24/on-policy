#!/bin/bash

# Monitor script for parallel hyperparameter sweep
# Provides real-time status of running experiments

REFRESH_INTERVAL=30  # seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display current status
show_status() {
    clear
    echo -e "${BLUE}=== Parallel Hyperparameter Sweep Monitor ===${NC}"
    echo "Last updated: $(date)"
    echo ""
    
    # System stats
    local total_cores=$(nproc)
    local load_avg=$(uptime | awk '{print $10}' | sed 's/,//')
    local mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    local disk_usage=$(df . | tail -1 | awk '{print $5}')
    
    echo -e "${BLUE}System Status:${NC}"
    echo "  CPU Cores: $total_cores"
    echo "  Load Average: $load_avg"
    echo "  Memory Usage: ${mem_usage}%"
    echo "  Disk Usage: $disk_usage"
    echo ""
    
    # Job statistics
    local total_jobs=0
    local active_jobs=0
    local completed_jobs=0
    local failed_jobs=0
    
    if [ -f "job_queue.txt" ]; then
        total_jobs=$(wc -l < job_queue.txt)
    fi
    
    if [ -f "active_jobs.txt" ]; then
        active_jobs=$(wc -l < active_jobs.txt)
    fi
    
    if [ -f "completed_jobs.txt" ]; then
        completed_jobs=$(wc -l < completed_jobs.txt)
    fi
    
    # Count failed jobs
    if [ -d "results" ]; then
        failed_jobs=$(grep -l "exit code: [^0]" results/*.result 2>/dev/null | wc -l)
    fi
    
    local success_jobs=$((completed_jobs - failed_jobs))
    
    echo -e "${BLUE}Job Statistics:${NC}"
    echo -e "  Total Jobs: $total_jobs"
    echo -e "  ${YELLOW}Active: $active_jobs${NC}"
    echo -e "  ${GREEN}Successful: $success_jobs${NC}"
    echo -e "  ${RED}Failed: $failed_jobs${NC}"
    echo -e "  Progress: $((completed_jobs * 100 / total_jobs))%"
    echo ""
    
    # Active jobs details
    echo -e "${BLUE}Active Jobs:${NC}"
    if [ $active_jobs -eq 0 ]; then
        echo "  No active jobs"
    else
        while read -r job_id; do
            if [ -f "pids/${job_id}.pid" ]; then
                local pid=$(cat "pids/${job_id}.pid")
                local cpu_usage=$(ps -p $pid -o %cpu --no-headers 2>/dev/null | tr -d ' ')
                local mem_usage_job=$(ps -p $pid -o %mem --no-headers 2>/dev/null | tr -d ' ')
                local runtime=$(ps -p $pid -o etime --no-headers 2>/dev/null | tr -d ' ')
                
                if [ -n "$cpu_usage" ]; then
                    echo -e "  Job $job_id (PID: $pid): CPU ${cpu_usage}%, MEM ${mem_usage_job}%, Runtime: $runtime"
                else
                    echo -e "  Job $job_id: Process not found (may have just finished)"
                fi
            fi
        done < active_jobs.txt
    fi
    echo ""
    
    # Recent completions
    echo -e "${BLUE}Recent Completions:${NC}"
    if [ -d "results" ]; then
        local recent_results=$(ls -t results/*.result 2>/dev/null | head -5)
        if [ -n "$recent_results" ]; then
            for result_file in $recent_results; do
                local job_name=$(basename "$result_file" .result)
                local exit_code=$(grep "exit code:" "$result_file" | awk '{print $NF}')
                local timestamp=$(grep "finished" "$result_file" | cut -d: -f1-3)
                
                if [ "$exit_code" = "0" ]; then
                    echo -e "  ${GREEN}✓${NC} $job_name ($timestamp)"
                else
                    echo -e "  ${RED}✗${NC} $job_name ($timestamp) - Exit code: $exit_code"
                fi
            done
        else
            echo "  No completed jobs yet"
        fi
    fi
    echo ""
    
    # GPU utilization (if nvidia-smi is available)
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BLUE}GPU Status:${NC}"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r idx name util mem_used mem_total; do
            echo "  GPU $idx ($name): ${util}% util, ${mem_used}MB/${mem_total}MB memory"
        done
        echo ""
    fi
    
    # Estimated completion time
    if [ $completed_jobs -gt 0 ] && [ $active_jobs -gt 0 ]; then
        local avg_job_time=$(find results -name "*.result" -exec grep "finished" {} \; | wc -l)
        # This is a simplified estimation - you could make it more sophisticated
        echo -e "${BLUE}Estimation:${NC}"
        echo "  Jobs remaining: $((total_jobs - completed_jobs))"
        echo "  (Completion time estimation requires more job history)"
        echo ""
    fi
    
    echo -e "${YELLOW}Press Ctrl+C to exit monitor${NC}"
}

# Function to show detailed log of a specific job
show_job_log() {
    local job_id=$1
    if [ -f "logs/pp-parallel-sweep_j${job_id}_*.log" ]; then
        echo "Showing log for job $job_id:"
        echo "========================="
        tail -20 logs/pp-parallel-sweep_j${job_id}_*.log
    else
        echo "No log file found for job $job_id"
    fi
}

# Function to kill a specific job
kill_job() {
    local job_id=$1
    if [ -f "pids/${job_id}.pid" ]; then
        local pid=$(cat "pids/${job_id}.pid")
        echo "Killing job $job_id (PID: $pid)..."
        kill -TERM $pid
        sleep 2
        if kill -0 $pid 2>/dev/null; then
            echo "Force killing job $job_id..."
            kill -KILL $pid
        fi
        rm -f "pids/${job_id}.pid"
        grep -v "^$job_id$" active_jobs.txt > active_jobs.txt.tmp && mv active_jobs.txt.tmp active_jobs.txt
        echo "Job $job_id terminated"
    else
        echo "Job $job_id is not active"
    fi
}

# Function to show summary statistics
show_summary() {
    echo "Experiment Summary:"
    echo "=================="
    
    if [ -d "results" ]; then
        local total=$(ls results/*.result 2>/dev/null | wc -l)
        local successful=$(grep -l "exit code: 0" results/*.result 2>/dev/null | wc -l)
        local failed=$((total - successful))
        
        echo "Total completed: $total"
        echo "Successful: $successful"
        echo "Failed: $failed"
        
        if [ $failed -gt 0 ]; then
            echo ""
            echo "Failed experiments:"
            for result_file in results/*.result; do
                if ! grep -q "exit code: 0" "$result_file"; then
                    local job_name=$(basename "$result_file" .result)
                    echo "  $job_name"
                fi
            done
        fi
    fi
}

# Main execution
case "${1:-monitor}" in
    "monitor")
        # Continuous monitoring
        trap 'echo ""; echo "Monitor stopped"; exit 0' INT
        while true; do
            show_status
            sleep $REFRESH_INTERVAL
        done
        ;;
    "log")
        if [ -z "$2" ]; then
            echo "Usage: $0 log <job_id>"
            exit 1
        fi
        show_job_log "$2"
        ;;
    "kill")
        if [ -z "$2" ]; then
            echo "Usage: $0 kill <job_id>"
            exit 1
        fi
        kill_job "$2"
        ;;
    "summary")
        show_summary
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  monitor          Show continuous status updates (default)"
        echo "  log <job_id>     Show recent log entries for specific job"
        echo "  kill <job_id>    Terminate a specific job"
        echo "  summary          Show completion summary"
        echo "  help             Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
