#!/bin/bash
"""
WandB Sweep Runner Script
Initializes the sweep and runs 3 parallel agents
"""

import wandb
import subprocess
import time
import sys
import os
from pathlib import Path
import signal
import argparse

def initialize_sweep(config_file="sweep_config.yaml", project_name="ddcl-applications"):
    """
    Initialize a new sweep from the configuration file
    """
    with open(config_file, 'r') as f:
        import yaml
        sweep_config = yaml.safe_load(f)

    # Create the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name
    )

    print(f"Created sweep with ID: {sweep_id}")
    print(f"View sweep at: https://wandb.ai/{os.environ.get('WANDB_ENTITY', 'your-entity')}/{project_name}/sweeps/{sweep_id}")

    return sweep_id

def run_agent(sweep_id, project_name, gpu_id=0, count=None):
    """
    Run a single sweep agent
    """
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = ['wandb', 'agent']
    if count:
        cmd.extend(['--count', str(count)])
    cmd.append(f'{project_name}/{sweep_id}')

    return subprocess.Popen(cmd, env=env)

def run_parallel_sweep(sweep_id=None, project_name="ddcl-applications",
                      n_agents=3, runs_per_agent=None, gpus=None):
    """
    Run multiple sweep agents in parallel

    Args:
        sweep_id: WandB sweep ID (if None, creates new sweep)
        project_name: WandB project name
        n_agents: Number of parallel agents to run
        runs_per_agent: Number of runs each agent should complete (None for unlimited)
        gpus: List of GPU IDs to use (cycles through if fewer GPUs than agents)
    """
    if sweep_id is None:
        sweep_id = initialize_sweep(project_name=project_name)

    if gpus is None:
        gpus = [0]  # Default to single GPU

    processes = []

    def signal_handler(sig, frame):
        print("\nInterrupting sweep agents...")
        for p in processes:
            if p.poll() is None:
                p.terminate()
        time.sleep(2)
        for p in processes:
            if p.poll() is None:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"\nStarting {n_agents} parallel sweep agents...")
    print(f"Sweep ID: {sweep_id}")
    print(f"Project: {project_name}")
    if runs_per_agent:
        print(f"Each agent will run {runs_per_agent} experiments")

    # Start agents
    for i in range(n_agents):
        gpu_id = gpus[i % len(gpus)]
        print(f"Starting agent {i+1} on GPU {gpu_id}...")
        p = run_agent(sweep_id, project_name, gpu_id, count=runs_per_agent)
        processes.append(p)
        time.sleep(2)  # Small delay between starting agents

    print("\nAll agents started. Press Ctrl+C to stop.")
    print(f"Monitor progress at: https://wandb.ai/{os.environ.get('WANDB_ENTITY', 'your-entity')}/{project_name}/sweeps/{sweep_id}")

    # Wait for all processes to complete
    try:
        for i, p in enumerate(processes):
            p.wait()
            print(f"Agent {i+1} completed")
    except KeyboardInterrupt:
        signal_handler(None, None)

    print("\nSweep completed!")

def main():
    parser = argparse.ArgumentParser(description='Run WandB sweep for PredatorPrey MARL')
    parser.add_argument('--sweep-id', type=str, default=None,
                       help='Existing sweep ID to continue (creates new if not provided)')
    parser.add_argument('--project', type=str, default='ddcl-applications',
                       help='WandB project name')
    parser.add_argument('--n-agents', type=int, default=3,
                       help='Number of parallel agents to run')
    parser.add_argument('--runs-per-agent', type=int, default=None,
                       help='Number of runs each agent should complete')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                       help='GPU IDs to use (e.g., --gpus 0 1 2)')
    parser.add_argument('--config', type=str, default='configs/sweep_config_ddcl_hard.yaml',
                       help='Path to sweep configuration file')

    args = parser.parse_args()

    # Update config file path if needed
    if args.sweep_id is None and args.config != 'sweep_config.yaml':
        sweep_id = initialize_sweep(config_file=args.config, project_name=args.project)
    else:
        sweep_id = args.sweep_id

    run_parallel_sweep(
        sweep_id=sweep_id,
        project_name=args.project,
        n_agents=args.n_agents,
        runs_per_agent=args.runs_per_agent,
        gpus=args.gpus
    )

if __name__ == "__main__":
    main()
