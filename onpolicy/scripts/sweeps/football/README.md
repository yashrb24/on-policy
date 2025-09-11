# WandB Sweep for Football Multi-Agent RL

This setup converts your bash hyperparameter sweep script into a WandB sweep with parallel execution and early stopping based on win_rate.

## Files Created

1. **sweep_config.yaml** - Basic WandB sweep configuration
2. **sweep_config_optimized.yaml** - Optimized configuration with constraint handling
3. **sweep_wrapper.py** - Python wrapper that validates architecture constraints (n_embd divisible by n_head)
4. **run_sweep.py** - Main script to run parallel sweep agents

## Setup

1. Install required packages:
```bash
pip install wandb pyyaml
```

2. Login to WandB:
```bash
wandb login
```

3. Make scripts executable:
```bash
chmod +x sweep_wrapper.py run_sweep.py
```

## Usage

### Quick Start (3 parallel agents on single GPU)

```bash
python run_sweep.py --n-agents 3 --project football-marl-sweep
```

### Advanced Usage

#### Run on multiple GPUs (e.g., GPUs 0, 1, 2):
```bash
python run_sweep.py --n-agents 3 --gpus 0 1 2 --project football-marl-sweep
```

#### Continue an existing sweep:
```bash
python run_sweep.py --sweep-id YOUR_SWEEP_ID --n-agents 3 --project football-marl-sweep
```

#### Limit runs per agent (useful for testing):
```bash
python run_sweep.py --n-agents 3 --runs-per-agent 10 --project football-marl-sweep
```

#### Use optimized configuration:
```bash
python run_sweep.py --config sweep_config_optimized.yaml --n-agents 3 --project football-marl-sweep
```

### Manual Sweep Management

#### Create a sweep without running agents:

```python
import wandb
import yaml

with open('configs/sweep_config_optimized.yaml', 'r') as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, project="football-marl-sweep")
print(f"Sweep ID: {sweep_id}")
```

#### Run a single agent:
```bash
wandb agent football-marl-sweep/SWEEP_ID
```

## Key Features

### 1. Architecture Validation
The `sweep_wrapper.py` automatically validates that `n_embd` is divisible by `n_head`. Invalid configurations are skipped.

### 2. Early Stopping
The sweep uses Hyperband early stopping to terminate poorly performing runs based on `win_rate`. Configuration:
- **min_iter**: 3 (minimum iterations before considering stopping)
- **s**: 2 (maximum early stopping factor)
- **eta**: 3 (halving rate)

### 3. Bayesian Optimization
Uses Bayesian optimization to intelligently explore the hyperparameter space rather than random or grid search.

### 4. Parallel Execution
Run multiple agents in parallel to speed up the sweep. Each agent can run on a different GPU if available.

## Hyperparameters Being Swept

### High Priority (Based on research insights):
- **lr**: [1e-4, 3e-4, 5e-4, 1e-3]
- **critic_lr**: [1e-4, 3e-4, 5e-4, 1e-3]
- **entropy_coef**: [0.001, 0.005, 0.01, 0.02]
- **clip_param**: [0.05, 0.1, 0.2]

### Transformer Architecture:
- **n_block**: [1, 2, 3]
- **n_embd**: [64, 128, 256]
- **n_head**: [1, 2, 4]

### Secondary:
- **ppo_epoch**: [5, 10, 15]
- **max_grad_norm**: [0.5, 10.0]
- **seed**: [1-5] (random)

## Monitoring

1. **WandB Dashboard**: View real-time progress at:
   ```
   https://wandb.ai/YOUR_ENTITY/football-marl-sweep/sweeps/SWEEP_ID
   ```

2. **Local Logs**: Check console output for agent status

3. **Metrics**: Track win_rate, loss, and other metrics in WandB

## Tips for Optimization

1. **Start Small**: Test with `--runs-per-agent 5` to ensure everything works
2. **GPU Memory**: Adjust `n_rollout_threads` if you encounter OOM errors
3. **Early Stopping**: Tune early stopping parameters if too many/few runs are terminated
4. **Search Strategy**: Consider switching between `bayes`, `random`, or `grid` methods

## Troubleshooting

### Invalid Architecture Errors
- The wrapper automatically skips invalid n_embd/n_head combinations
- Check WandB logs for "invalid_architecture" flag

### CUDA Out of Memory
- Reduce `n_rollout_threads` in the configuration
- Use smaller `n_embd` values
- Run fewer parallel agents

### Sweep Not Starting
- Ensure you're logged into WandB: `wandb login`
- Check that the training script path is correct
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## Example Output

```
Created sweep with ID: abc123def
View sweep at: https://wandb.ai/your-entity/football-marl-sweep/sweeps/abc123def

Starting 3 parallel sweep agents...
Starting agent 1 on GPU 0...
Starting agent 2 on GPU 0...
Starting agent 3 on GPU 0...

All agents started. Press Ctrl+C to stop.
Monitor progress at: https://wandb.ai/your-entity/football-marl-sweep/sweeps/abc123def
```

## Best Practices

1. **Regular Checkpoints**: The configuration saves every 20000 steps
2. **Logging**: All runs log to WandB for easy comparison
3. **Reproducibility**: Seeds are tracked for each run
4. **Resource Management**: Agents automatically handle GPU allocation

## Next Steps

After the sweep completes:
1. Analyze results in WandB dashboard
2. Export best hyperparameters
3. Run longer training with best configuration
4. Consider second-stage fine-tuning sweep

## Notes

- The sweep prioritizes configurations based on multi-agent communication research
- Invalid transformer architectures are automatically filtered
- Early stopping helps focus compute on promising configurations
- Bayesian optimization learns from previous runs to suggest better parameters
