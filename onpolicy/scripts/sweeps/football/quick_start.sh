#!/bin/bash

# Quick Start Script for WandB Sweep
# This script sets up and starts the sweep with sensible defaults

echo "================================================"
echo "Football Multi-Agent RL - WandB Sweep Setup"
echo "================================================"

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

# Check if logged in to wandb
if ! wandb verify 2>/dev/null; then
    echo "Please log in to WandB:"
    wandb login
fi

# Default values
PROJECT_NAME="football-marl-sweep"
N_AGENTS=3
CONFIG_FILE="sweep_config_optimized.yaml"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --project) PROJECT_NAME="$2"; shift ;;
        --agents) N_AGENTS="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --gpus) GPUS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo ""
echo "Configuration:"
echo "  Project: $PROJECT_NAME"
echo "  Parallel agents: $N_AGENTS"
echo "  Config file: $CONFIG_FILE"
echo "  GPUs: ${GPUS:-0 (single GPU)}"
echo ""

# Make scripts executable
chmod +x sweep_wrapper.py run_sweep.py extract_best.py

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found!"
    exit 1
fi

# Check if training script exists
if [ ! -f "../train/train_football.py" ]; then
    echo "Warning: Training script not found at ../train/train_football.py"
    echo "Please ensure your training script is in the correct location."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting WandB sweep..."
echo ""

# Run the sweep
if [ -z "$GPUS" ]; then
    python run_sweep.py \
        --config "$CONFIG_FILE" \
        --project "$PROJECT_NAME" \
        --n-agents "$N_AGENTS"
else
    # Parse space-separated GPU list
    GPU_ARGS=""
    for gpu in $GPUS; do
        GPU_ARGS="$GPU_ARGS $gpu"
    done
    
    python run_sweep.py \
        --config "$CONFIG_FILE" \
        --project "$PROJECT_NAME" \
        --n-agents "$N_AGENTS" \
        --gpus $GPU_ARGS
fi
