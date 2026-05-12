# DDCL Toy Problem — Statistical Methods Guide

> **Purpose:** Every statistical method used in this project is explained here in plain English. A reader should be able to understand what a p-value or confidence interval means in context without prior statistics knowledge. All methods are implemented in `analysis/stats.py`.

---

## 1. Why We Care About Statistics

A single RL training run is noisy. Two runs with the same hyperparameters but different random seeds can differ by 5–10% in final success rate. Without proper statistics:
- We might declare a method "better" when the difference is just random luck.
- We might report the best-of-5-seeds result, which is biased upward.

Our rules:
1. **Always report results across ≥5 seeds.**
2. **Always report uncertainty** (bootstrap CI or std) alongside means.
3. **Always test statistical significance** before claiming one method beats another.
4. **Never cherry-pick seeds** or hyperparameters post-hoc.

---

## 2. Bootstrap Confidence Interval (bootstrap CI)

### What it does
Estimates a confidence interval (CI) for a statistic (e.g., mean success rate) when we only have a small sample (e.g., 5 seeds).

### Plain English explanation
We have 5 numbers (one per seed). We want to know: if we ran 100 more seeds, where would the mean likely land? Bootstrapping answers this by *resampling with replacement* from our 5 numbers many times (typically 2000 rounds), computing the mean of each resample, and taking the 2.5th and 97.5th percentile of those means as the 95% CI.

### When to use it
Whenever you report a mean or IQM. Every number in results tables should have a CI.

### Interpretation
`0.923 [0.901, 0.941]` means: the point estimate is 0.923, and we are 95% confident the true mean lies between 0.901 and 0.941.

### Implementation
```python
from analysis.stats import bootstrap_ci
point, lo, hi = bootstrap_ci(per_seed_values, n_boot=2000, ci=0.95)
```

### Pitfalls
- With 5 seeds the CI is wide — this is honest, not a problem with the method.
- CIs can overlap even when methods differ significantly; overlapping CIs do NOT mean "no difference" — use a hypothesis test.

---

## 3. Interquartile Mean (IQM)

### What it does
Computes the mean of the middle 50% of values (drops bottom and top 25%).

### Plain English explanation
Imagine 10 seeds. Sort them. Drop the 2 worst and 2 best. Average the remaining 6. This is the IQM. It is more robust than the plain mean (not pulled by one catastrophic seed) and less conservative than the median (uses more data).

### When to use it
Always alongside the regular mean. IQM is our primary summary statistic for comparing methods (Agarwal et al., 2021 "Deep RL at the edge of the statistical precipice").

### Interpretation
If IQM ≈ mean → results are consistent across seeds. If IQM >> mean → there are catastrophic seeds dragging the mean down.

### Implementation
```python
from analysis.stats import iqm, iqm_ci
value = iqm(per_seed_values)
point, lo, hi = iqm_ci(per_seed_values, n_boot=2000)
```

---

## 4. Paired Permutation Test

### What it does
Tests whether two methods are statistically distinguishable, given matched per-seed measurements.

### Plain English explanation
We have method A and method B, each run on the same 5 seeds: `a = [0.91, 0.88, 0.93, 0.90, 0.89]`, `b = [0.86, 0.82, 0.87, 0.85, 0.83]`. The differences per seed are `d = a - b = [0.05, 0.06, 0.06, 0.05, 0.06]`.

The null hypothesis H₀ says: A and B are the same, so each sign of d is equally likely to be + or −. We randomly flip signs of d thousands of times and measure how often the (sign-flipped) mean difference is as large as what we observed. The fraction of times this happens is the p-value.

### When to use it
Whenever claiming "method A is better than method B". Also used in the sweep convergence gate.

### Interpretation
- p < 0.05: the difference is unlikely to be due to random chance alone → significant.
- p > 0.05: the difference could plausibly be random → not significant.
- **We report exact p-values, not just ✓/✗.** A p-value of 0.04 and 0.001 are both "significant" but represent very different strength of evidence.

### Implementation
```python
from analysis.stats import paired_permutation_test
p = paired_permutation_test(a_values, b_values, n_perm=10_000)
```

### Why not a t-test?
The t-test assumes normality. RL rewards have fat tails and clipping. The permutation test makes no distributional assumptions.

---

## 5. Wilcoxon Signed-Rank Test

### What it does
Non-parametric test for paired data that is more powerful than the sign test but less powerful than the permutation test when n is small.

### When to use it
As a secondary check alongside the permutation test. We report both when n_seeds < 10.

### Interpretation
Same as the permutation test: p < 0.05 → significant.

### Implementation
```python
from analysis.stats import wilcoxon_signed_rank
p = wilcoxon_signed_rank(a_values, b_values)  # requires scipy
```

### Note
This requires `scipy`. If not installed: `pip install scipy`.

---

## 6. Pareto Frontier Test

### What it does
Identifies which configurations are non-dominated on two objectives (success_rate and bits/episode).

### Plain English explanation
We want a method that is simultaneously: (a) high success rate, and (b) low communication cost. No single scalar metric captures this. A Pareto frontier is the set of configurations where you cannot improve one objective without worsening the other. A method that beats every other method on *both* objectives simultaneously is "dominant"; a method on the Pareto frontier is the best achievable at its communication cost.

### When to use it
In all rate–distortion plots and in the sweep convergence gate (criterion 4).

### Interpretation
If DDCL lies on the Pareto frontier and all baselines (STE, additive-uniform, Gaussian) do not, then DDCL is provably better for some communication budget.

### Implementation
```python
from analysis.stats import pareto_frontier
pf = pareto_frontier(df, x_col="success_rate", y_col="bits_per_msg",
                     x_better="higher", y_better="lower")
```

---

## 7. Sweep Convergence Gate (4 Criteria)

The Phase 2 sweep is declared **done** only when ALL four criteria pass simultaneously.

### Criterion 1: Interior optimum
The winning configuration must NOT be at the boundary of any swept axis. If the winner is at the boundary (e.g., largest λ tried), the sweep must be extended in that direction first.

**Why:** A boundary optimum means the true best hyperparameter might be outside the searched range.

### Criterion 2: Statistical stability
The winner must be statistically significantly better than the runner-up (p < 0.05, paired permutation).

**Why:** If the top two configs are statistically indistinguishable, we cannot reliably identify a winner and the sweep needs more seeds or a finer grid.

### Criterion 3: Seed stability
The winner must be #1 under ALL three aggregators: mean, median, and IQM.

**Why:** If a config wins under mean but not under IQM, it is probably driven by a single lucky seed and is not reliably best.

### Criterion 4: Pareto non-dominance
The winner must lie on the Pareto frontier (success_rate × bits/episode).

**Why:** A configuration with high success rate but 10× more communication than a simpler baseline is not a good baseline.

### Implementation
```python
from analysis.sweep_convergence import check_convergence
result = check_convergence(summary_df, sweep_axes=["lambda_comms", "delta"])
print(result)   # pass/fail for each criterion
```

---

## 8. Gradient-Variance Diagnostic (Phase 3.3)

### What it does
Compares the empirical per-parameter gradient variance of the Rao-Blackwell estimator (P4) to the single-sample estimator over many minibatches at a fixed checkpoint.

### Plain English explanation
For a stochastic gradient estimator, lower variance → faster convergence (lower SNR). We measure the sample variance of gradients over 256 minibatches at a fixed network checkpoint, for each parameter. The ratio `Var(g_RB) / Var(g_single)` tells us the variance reduction factor. Theorem 6 predicts this should be < 1.

### Interpretation
- Ratio < 1: P4 reduces variance → expected improvement in later training.
- Ratio ≈ 1: no practical variance reduction at this scale.
- We report the **log-ratio with bootstrap CIs** to detect if the effect is significant.

---

## 9. Reporting Standards

All results tables in `results/` follow this format:

| Config | success_rate mean [95% CI] | success_rate IQM [95% CI] | n_seeds |
|--------|---------------------------|---------------------------|---------|
| sd, λ=4e-3, δ=10.0 | 0.923 [0.901, 0.941] | 0.919 [0.898, 0.937] | 5 |

- **No WandB smoothing** in any numbers (raw unsmoothed CSV values only).
- **All random seeds reported**, not cherry-picked.
- **p-values reported exactly** (not just ✓/✗).
- **Git SHA** attached to every experiment run (in `runs/<exp>/<seed>/git_sha.txt`).

---

## References

- Agarwal, R. et al. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS 2021. [IQM justification]
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap.* [Bootstrap CI]
- Good, P. (2000). *Permutation Tests.* Springer. [Permutation test]
