#!/usr/bin/env python3
"""WandB sweep runner for Pursuit transformer actor-critic experiments."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import wandb
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = "configs/sweep_config_transformer_ac_vanilla_mappo.yaml"
DEFAULT_PROJECT = "ddcl-applications"


def _resolve_config_path(config_file):
    config_path = Path(config_file)
    if config_path.is_absolute() or config_path.exists():
        return config_path
    return SCRIPT_DIR / config_path


def initialize_sweep(config_file=DEFAULT_CONFIG, project_name=DEFAULT_PROJECT):
    """Initialize a new W&B sweep from the configuration file."""
    config_path = _resolve_config_path(config_file)
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name,
    )

    print(f"Created sweep with ID: {sweep_id}")
    print(
        "View sweep at: "
        f"https://wandb.ai/{os.environ.get('WANDB_ENTITY', 'your-entity')}/"
        f"{project_name}/sweeps/{sweep_id}"
    )

    return sweep_id


def run_agent(sweep_id, project_name, gpu_id=0, count=None):
    """Run a single W&B sweep agent."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = ["wandb", "agent"]
    if count:
        cmd.extend(["--count", str(count)])
    cmd.append(f"{project_name}/{sweep_id}")

    return subprocess.Popen(cmd, cwd=SCRIPT_DIR, env=env)


def run_parallel_sweep(
    sweep_id=None,
    project_name=DEFAULT_PROJECT,
    n_agents=1,
    runs_per_agent=None,
    gpus=None,
):
    """Run one or more W&B sweep agents, assigning each to a GPU."""
    if sweep_id is None:
        sweep_id = initialize_sweep(project_name=project_name)

    if gpus is None:
        gpus = [0]

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

    for i in range(n_agents):
        gpu_id = gpus[i % len(gpus)]
        print(f"Starting agent {i + 1} on GPU {gpu_id}...")
        process = run_agent(sweep_id, project_name, gpu_id, count=runs_per_agent)
        processes.append(process)
        time.sleep(2)

    print("\nAll agents started. Press Ctrl+C to stop.")
    print(
        "Monitor progress at: "
        f"https://wandb.ai/{os.environ.get('WANDB_ENTITY', 'your-entity')}/"
        f"{project_name}/sweeps/{sweep_id}"
    )

    try:
        for i, process in enumerate(processes):
            process.wait()
            print(f"Agent {i + 1} completed")
    except KeyboardInterrupt:
        signal_handler(None, None)

    print("\nSweep completed!")


def main():
    parser = argparse.ArgumentParser(description="Run WandB sweep for Pursuit MARL")
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing sweep ID to continue; creates a new sweep if omitted",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help="WandB project name",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=1,
        help="Number of parallel agents to run",
    )
    parser.add_argument(
        "--runs-per-agent",
        type=int,
        default=None,
        help="Number of runs each agent should complete",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="GPU IDs to use, e.g. --gpus 0 1 2",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to sweep configuration file",
    )

    args = parser.parse_args()

    if args.sweep_id is None:
        sweep_id = initialize_sweep(config_file=args.config, project_name=args.project)
    else:
        sweep_id = args.sweep_id

    run_parallel_sweep(
        sweep_id=sweep_id,
        project_name=args.project,
        n_agents=args.n_agents,
        runs_per_agent=args.runs_per_agent,
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
