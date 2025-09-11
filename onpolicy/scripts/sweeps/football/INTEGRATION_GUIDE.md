# Integration Guide - WandB Sweep Setup

## Directory Structure

Your project should be organized as follows:

```
your_project/
│
├── train/
│   └── train_football.py          # Your existing training script
│
├── sweep/                          # New directory for sweep files
│   ├── sweep_config.yaml           # Basic sweep configuration
│   ├── sweep_config_optimized.yaml # Optimized configuration
│   ├── sweep_wrapper.py            # Wrapper script with validation
│   ├── run_sweep.py                # Main sweep runner
│   ├── extract_best.py             # Extract best parameters
│   ├── quick_start.sh              # Quick start script
│   └── README.md                   # Documentation
│
└── logs/                           # Directory for logs (created automatically)
```

## Integration Steps

### 1. Copy Files to Your Project

```bash
# Create sweep directory in your project
mkdir -p /path/to/your_project/sweep

# Copy all the generated files
cp sweep_config*.yaml sweep_wrapper.py run_sweep.py extract_best.py quick_start.sh README.md /path/to/your_project/sweep/

# Navigate to sweep directory
cd /path/to/your_project/sweep
```

### 2. Update Paths in Configuration

If your training script is not at `../train/train_football.py`, update the path in:
- `sweep_wrapper.py` (line ~43)
- `extract_best.py` (in `generate_training_script` function)

### 3. Quick Start

```bash
# Option 1: Use the quick start script (recommended)
./quick_start.sh

# Option 2: Manual start with custom settings
python run_sweep.py --n-agents 3 --project football-marl-sweep
```

### 4. Monitor Progress

```bash
# Watch the console output
# OR
# Open WandB dashboard in browser (URL will be printed)
```

### 5. Extract Best Configuration (after sweep)

```bash
# Get the sweep ID from the console output or WandB dashboard
python extract_best.py YOUR_SWEEP_ID --analyze --generate-script

# This will create:
# - best_config.yaml: Best hyperparameters
# - train_best.sh: Ready-to-run training script
```

### 6. Run Training with Best Parameters

```bash
./train_best.sh
```

## Customization Options

### Modify Hyperparameter Ranges

Edit `sweep_config_optimized.yaml`:

```yaml
parameters:
  lr:
    values: [0.00005, 0.0001, 0.0005, 0.001, 0.002]  # Add more values
  
  n_embd:
    values: [32, 64, 128, 256, 512]  # Extend range
```

### Change Early Stopping Behavior

Edit `sweep_config_optimized.yaml`:

```yaml
early_terminate:
  type: hyperband
  min_iter: 5      # Wait longer before stopping
  s: 3             # More aggressive stopping
  eta: 2           # Different halving rate
```

### Use Different GPUs

```bash
# Single GPU (GPU 1)
python run_sweep.py --gpus 1 --n-agents 3

# Multiple GPUs (0, 1, 2)
python run_sweep.py --gpus 0 1 2 --n-agents 3

# Or with quick start
./quick_start.sh --gpus "0 1 2" --agents 3
```

### Limit Number of Runs

```bash
# Each agent runs only 10 experiments
python run_sweep.py --n-agents 3 --runs-per-agent 10
```

## Troubleshooting

### Common Issues and Solutions

1. **"ModuleNotFoundError: No module named 'wandb'"**
   ```bash
   pip install wandb
   ```

2. **"Training script not found"**
   - Update the path in `sweep_wrapper.py`
   - Ensure you're running from the correct directory

3. **"CUDA out of memory"**
   - Reduce `n_rollout_threads` in sweep config
   - Use smaller `n_embd` values
   - Run fewer parallel agents

4. **"Invalid architecture" warnings**
   - This is expected - the wrapper automatically skips invalid n_embd/n_head combinations
   - No action needed

5. **Sweep not starting**
   ```bash
   # Check WandB login
   wandb login
   
   # Verify CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Advanced Features

### Resume Interrupted Sweep

```bash
# Get sweep ID from previous run
python run_sweep.py --sweep-id PREVIOUS_SWEEP_ID --n-agents 3
```

### Custom Metric for Early Stopping

Edit `sweep_config_optimized.yaml`:

```yaml
metric:
  name: custom_metric  # Change to your metric
  goal: maximize       # or minimize
```

### Conditional Parameter Sampling

For more complex parameter dependencies, modify `sweep_wrapper.py`:

```python
def validate_and_adjust_config(config):
    # Example: Scale learning rate with network size
    if config.n_embd > 128:
        config.lr = config.lr * 0.5
    return config
```

## Best Practices

1. **Start Small**: Test with 5-10 runs first
2. **Monitor Early**: Check first few runs for errors
3. **Save Checkpoints**: Keep `save_interval` reasonable
4. **Log Everything**: Use WandB logging extensively
5. **Document Results**: Export sweep results for future reference

## Export Results

After sweep completion:

```bash
# Export all results to CSV
wandb export YOUR_ENTITY/football-marl-sweep/SWEEP_ID --format csv

# Or use the API
python -c "
import wandb
api = wandb.Api()
sweep = api.sweep('YOUR_ENTITY/football-marl-sweep/SWEEP_ID')
df = sweep.to_dataframe()
df.to_csv('sweep_results.csv')
"
```

## Next Steps

1. **Analyze Results**: Use WandB dashboard for visualization
2. **Fine-tune**: Run targeted sweep on promising regions
3. **Scale Up**: Increase `num_env_steps` for best config
4. **Ablation Studies**: Test individual parameter impacts

## Support

- WandB Documentation: https://docs.wandb.ai/guides/sweeps
- Report Issues: Create issue in your project repository
- Community: WandB Community Forum

## Citation

If you use this sweep setup in your research:

```bibtex
@misc{football_marl_sweep,
  title={WandB Sweep Configuration for Football Multi-Agent RL},
  year={2024},
  author={Your Name},
  note={Hyperparameter optimization for transformer-based MARL}
}
```
