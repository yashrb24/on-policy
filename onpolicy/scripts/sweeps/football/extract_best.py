#!/usr/bin/env python3
"""
Extract and use the best hyperparameters from a WandB sweep
"""

import wandb
import yaml
import argparse
import json
from pathlib import Path

def get_best_run(sweep_id, project_name="football-marl-sweep", metric="win_rate"):
    """
    Get the best run from a sweep based on a metric
    """
    api = wandb.Api()
    sweep = api.sweep(f"{api.default_entity}/{project_name}/{sweep_id}")
    
    best_run = sweep.best_run()
    
    if best_run is None:
        print("No completed runs found in sweep")
        return None
    
    print(f"\nBest run: {best_run.name}")
    print(f"ID: {best_run.id}")
    print(f"Best {metric}: {best_run.summary.get(metric, 'N/A')}")
    
    return best_run

def extract_hyperparameters(run):
    """
    Extract hyperparameters from the best run
    """
    config = dict(run.config)
    
    # Filter out fixed parameters
    fixed_params = [
        'env_name', 'scenario_name', 'algorithm_name', 'num_agents',
        'num_env_steps', 'episode_length', 'representation', 'rewards',
        'n_rollout_threads', 'save_interval', 'log_interval',
        'use_transformer_base_actor', 'user_name', 'wandb_name'
    ]
    
    hyperparams = {k: v for k, v in config.items() if k not in fixed_params}
    
    return hyperparams

def save_best_config(hyperparams, output_file="best_config.yaml"):
    """
    Save the best hyperparameters to a YAML file
    """
    with open(output_file, 'w') as f:
        yaml.dump(hyperparams, f, default_flow_style=False)
    
    print(f"\nBest configuration saved to: {output_file}")

def generate_training_script(hyperparams, base_script_path="../train/train_football.py"):
    """
    Generate a training script with the best hyperparameters
    """
    script = f"""#!/bin/bash
# Training script with best hyperparameters from WandB sweep
# Auto-generated - do not edit manually

# Best hyperparameters
echo "Running with optimized hyperparameters from sweep"
echo "=================================="
echo "Learning rates: lr={hyperparams.get('lr')}, critic_lr={hyperparams.get('critic_lr')}"
echo "PPO: entropy_coef={hyperparams.get('entropy_coef')}, clip_param={hyperparams.get('clip_param')}"
echo "Architecture: n_block={hyperparams.get('n_block')}, n_embd={hyperparams.get('n_embd')}, n_head={hyperparams.get('n_head')}"
echo "=================================="

CUDA_VISIBLE_DEVICES=0 python {base_script_path} \\
    --env_name Football \\
    --scenario_name academy_3_vs_1_with_keeper \\
    --algorithm_name rmappo \\
    --experiment_name best_sweep_config \\
    --seed {hyperparams.get('seed', 1)} \\
    --num_agents 3 \\
    --num_env_steps 15000000 \\
    --episode_length 200 \\
    --representation simple115v2 \\
    --rewards scoring \\
    --n_rollout_threads 50 \\
    --save_interval 20000 \\
    --log_interval 20000 \\
    --use_transformer_base_actor \\
    --hidden_size {hyperparams.get('n_embd')} \\
    --lr {hyperparams.get('lr')} \\
    --critic_lr {hyperparams.get('critic_lr')} \\
    --ppo_epoch {hyperparams.get('ppo_epoch')} \\
    --clip_param {hyperparams.get('clip_param')} \\
    --num_mini_batch {hyperparams.get('num_mini_batch', 1)} \\
    --entropy_coef {hyperparams.get('entropy_coef')} \\
    --max_grad_norm {hyperparams.get('max_grad_norm')} \\
    --n_block {hyperparams.get('n_block')} \\
    --n_embd {hyperparams.get('n_embd')} \\
    --n_head {hyperparams.get('n_head')} \\
    --user_name yashrb \\
    --wandb_name on-policy
"""
    
    output_file = "train_best.sh"
    with open(output_file, 'w') as f:
        f.write(script)
    
    Path(output_file).chmod(0o755)
    print(f"Training script generated: {output_file}")
    
    return output_file

def analyze_sweep_results(sweep_id, project_name="football-marl-sweep", top_k=5):
    """
    Analyze and display top performing runs from the sweep
    """
    api = wandb.Api()
    sweep = api.sweep(f"{api.default_entity}/{project_name}/{sweep_id}")
    
    # Get all runs sorted by metric
    runs = sorted(sweep.runs, 
                  key=lambda r: r.summary.get('win_rate', 0), 
                  reverse=True)
    
    print(f"\nTop {min(top_k, len(runs))} runs by win_rate:")
    print("-" * 80)
    
    for i, run in enumerate(runs[:top_k]):
        if run.state == "finished":
            print(f"\n{i+1}. Run: {run.name}")
            print(f"   Win Rate: {run.summary.get('win_rate', 'N/A'):.4f}")
            print(f"   Config: lr={run.config.get('lr')}, "
                  f"entropy={run.config.get('entropy_coef')}, "
                  f"n_embd={run.config.get('n_embd')}, "
                  f"n_head={run.config.get('n_head')}")
    
    # Statistical summary
    win_rates = [r.summary.get('win_rate', 0) for r in runs if r.state == "finished"]
    if win_rates:
        import numpy as np
        print(f"\n\nStatistics across {len(win_rates)} completed runs:")
        print(f"  Mean win rate: {np.mean(win_rates):.4f}")
        print(f"  Std deviation: {np.std(win_rates):.4f}")
        print(f"  Max win rate: {np.max(win_rates):.4f}")
        print(f"  Min win rate: {np.min(win_rates):.4f}")

def main():
    parser = argparse.ArgumentParser(description='Extract best hyperparameters from WandB sweep')
    parser.add_argument('sweep_id', type=str, help='WandB sweep ID')
    parser.add_argument('--project', type=str, default='football-marl-sweep',
                       help='WandB project name')
    parser.add_argument('--metric', type=str, default='win_rate',
                       help='Metric to optimize')
    parser.add_argument('--output', type=str, default='best_config.yaml',
                       help='Output file for best configuration')
    parser.add_argument('--analyze', action='store_true',
                       help='Show detailed analysis of sweep results')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top runs to display in analysis')
    parser.add_argument('--generate-script', action='store_true',
                       help='Generate a training script with best parameters')
    
    args = parser.parse_args()
    
    # Get best run
    best_run = get_best_run(args.sweep_id, args.project, args.metric)
    
    if best_run:
        # Extract hyperparameters
        hyperparams = extract_hyperparameters(best_run)
        
        print("\nBest hyperparameters:")
        print("-" * 40)
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
        
        # Save configuration
        save_best_config(hyperparams, args.output)
        
        # Generate training script if requested
        if args.generate_script:
            generate_training_script(hyperparams)
        
        # Show analysis if requested
        if args.analyze:
            analyze_sweep_results(args.sweep_id, args.project, args.top_k)
        
        print("\n✓ Success! You can now:")
        print(f"  1. Use the saved config: {args.output}")
        if args.generate_script:
            print(f"  2. Run the optimized training: ./train_best.sh")
        print(f"  3. View full results at: https://wandb.ai/{best_run.entity}/{args.project}/sweeps/{args.sweep_id}")

if __name__ == "__main__":
    main()
