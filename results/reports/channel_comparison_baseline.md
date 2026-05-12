# Baseline Results

> Generated from: `onpolicy/envs/toyproblem/runs/channel_comparison`
> Convergence gate: 🔲 NOT DONE

## Summary

**Winner:** `none_lam0e+00_d1`
  - success_rate: 1.000 [1.000, 1.000] (mean [95% bootstrap CI])
  - success_rate: 1.000 [1.000, 1.000] (IQM [95% CI])

## Convergence Gate

```
Convergence gate: NOT DONE
  Winner: none_lam0e+00_d1
  1. Interior optimum   : FAIL  (boundary axes: ['lambda_comms=0 (range [0,0.004])', 'delta=1 (range [1,10])', 'z_dim=3 (range [3,3])'])
  2. Statistical stability: FAIL  (p vs runner-up = 1.000)
  3. Seed stability     : PASS  (rank by mean=0, median=0, IQM=0)
  4. Pareto non-dom.    : PASS
```

## Per-Config Performance Table

| Config | success_rate mean [CI] | success_rate IQM [CI] | n_seeds |
|--------|-------------------|------------------|---------|
| additive_uniform_lam4e-03_d10 | 0.556 [0.547, 0.565] | 0.555 [0.545, 0.568] | 5 |
| gaussian_lam4e-03_d10 | 0.552 [0.547, 0.558] | 0.552 [0.546, 0.559] | 5 |
| none_lam0e+00_d1 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 5 |
| nsd_lam4e-03_d10 | 0.555 [0.550, 0.559] | 0.554 [0.549, 0.561] | 5 |
| sd_lam4e-03_d10 | 0.548 [0.542, 0.554] | 0.548 [0.540, 0.557] | 5 |
| ste16 | 0.962 [0.885, 1.000] | 1.000 [0.872, 1.000] | 5 |
| ste4 | 0.882 [0.824, 0.941] | 0.875 [0.813, 0.967] | 5 |
| ste8 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 5 |

## Pairwise Statistical Comparisons

| Config A | Config B | Δ mean | p-value | Significant (α=0.05) |
|----------|----------|--------|---------|----------------------|
| none | sd | +0.4517 | 0.062 | NO |
| none | nsd | +0.4453 | 0.067 | NO |
| none | ste4 | +0.1181 | 0.061 | NO |
| ste16 | ste4 | +0.0796 | 0.188 | NO |
| none | ste16 | +0.0384 | 1.000 | NO |
| additive_uniform | sd | +0.0079 | 0.058 | NO |
| nsd | sd | +0.0064 | 0.249 | NO |
| gaussian | sd | +0.0040 | 0.316 | NO |
| additive_uniform | gaussian | +0.0038 | 0.446 | NO |
| additive_uniform | nsd | +0.0014 | 0.876 | NO |
| none | ste8 | +0.0000 | 1.000 | NO |
| gaussian | nsd | -0.0024 | 0.759 | NO |
| ste16 | ste8 | -0.0384 | 1.000 | NO |
| ste4 | ste8 | -0.1181 | 0.067 | NO |
| additive_uniform | ste4 | -0.3258 | 0.062 | NO |
| nsd | ste4 | -0.3272 | 0.062 | NO |
| gaussian | ste4 | -0.3296 | 0.059 | NO |
| sd | ste4 | -0.3337 | 0.056 | NO |
| additive_uniform | ste16 | -0.4054 | 0.060 | NO |
| nsd | ste16 | -0.4069 | 0.064 | NO |
| gaussian | ste16 | -0.4093 | 0.064 | NO |
| sd | ste16 | -0.4133 | 0.059 | NO |
| additive_uniform | none | -0.4439 | 0.061 | NO |
| additive_uniform | ste8 | -0.4439 | 0.065 | NO |
| nsd | ste8 | -0.4453 | 0.065 | NO |
| gaussian | none | -0.4477 | 0.065 | NO |
| gaussian | ste8 | -0.4477 | 0.064 | NO |
| sd | ste8 | -0.4517 | 0.062 | NO |

## Plots

![Training curves](plots/training_curves_success.png)
![Rate–distortion](plots/rate_distortion.png)
![Channel comparison](plots/channel_comparison.png)
![Per-goal bits](plots/per_goal_bits.png)