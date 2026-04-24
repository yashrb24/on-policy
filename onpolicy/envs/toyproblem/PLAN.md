# DDCL Unleashed: Toy-Problem Verification, Sweeps & 4-Pillar Rollout

## Context

**Papers read.** The DDCL paper ("Learning What to Say and How Precisely", Kapoor et al.) generalises Differentiable Discrete Communication Learning to unbounded, signed signals and validates it on a toy `CommunicatingGoal` environment plus MARL benchmarks. The companion proposal ("Stochastic Quantisation via Dithering: A Plug-and-Play Layer…") proves DDCL is isomorphic to classical subtractive dithered quantisation and introduces four principled extensions:

- **P1 — Per-channel δ**: allocate quantisation width per message dimension. Natural progression: first heuristic (fixed per-channel scalars), then learned (`δ_k = softplus(α_k)`) — Thm 4.
- **P2 — Entropy model**: replace the magnitude surrogate with a learned `q_φ(m|h)` driving `L_ent = E[-log₂ q_φ]` to Shannon entropy (Eq. 2).
- **P3 — TPDF dithering**: `ε = ε₁ + ε₂` non-subtractive dither removing the shared-PRNG deployment constraint (Thm 5).
- **P4 — Rao-Blackwell gradient**: analytical two-bin gradient `g_RB = p_a ∇L_a + p_b ∇L_b` (Thm 6).

**Revised bit-cost formula.** The proposal writes `L_comms = Σ log₂(2|z|/δ + 1)`. We drop the constant `2` since it is a constant multiplier that does not affect optimisation. Throughout this project we use `L_comms = Σ log₂(|z|/δ + 1)` — documented in `docs/MATH.md`.

**Project goal.** Build a rigorous, well-documented, statistically honest toy-problem testbed that:

1. **Verifies DDCL** logically and numerically against the paper's **mathematical principles** (not hyperparameters — we redo hyperparameter selection from scratch).
2. Demonstrates that DDCL sits on the **Pareto frontier** of `success_rate` vs `communication_cost` relative to naive baselines — exact reproduction of paper numbers is **not** the goal.
3. Implements each of the 4 pillars as an independently-ablatable module with unit tests and a head-to-head study.
4. Shows scaling behaviour across **grid size, #goals, goal-distribution entropy, message dimension, and number of agents**.
5. Ships with reproducibility scripts and statistical analysis that does not rely on WandB's display-side smoothing.

**Scope:** toyproblem standalone only. **Pillar order:** P2 → P1 → P4 → P3 (highest-impact first). **Sweeps:** WandB + offline raw CSV/parquet archival. **Pre-existing repo issues** (PP step-limit, active_masks, transformer-DDCL parity): out of scope, noted in `docs/KNOWN_ISSUES.md`.

---

## Status Summary

| Phase | Status | pytest gate |
|-------|--------|-------------|
| 0 — Hygiene & principles | ✅ DONE | 28 passed, 23 skipped, 0 failed |
| 1 — Verify & tighten DDCL | ✅ DONE | 51 passed, 0 skipped, 0 failed |
| 2 — Hyperparameter sweeps | 🔶 INFRA DONE | 79 passed, 0 failed; sweeps pending compute |
| 3 — 4 Pillars (P2→P1→P4→P3) | 🔲 TODO | — |
| 4 — Combined + scaling | 🔲 TODO | — |
| 5 — Docs & reproducibility | 🔲 TODO | — |

---

## Phase 0 — Repository hygiene & principles ✅ DONE

- `docs/README.md` — reader-facing walkthrough of env, channels, networks, CLI flags, log layout.
- `docs/MATH.md` — symbol → code map; bit-cost deviation justified; per-pillar stubs.
- `docs/KNOWN_ISSUES.md` — 6 out-of-scope issues catalogued.
- `tests/` — pytest skeleton: `conftest.py`, `test_channels.py`, `test_env_determinism.py`, `test_trainer.py`. 28 passing, 23 skipped (stubs for Phase 1).
- `train.py` hardened: `set_seed()` (Python, NumPy, PyTorch, CUDA, PYTHONHASHSEED); `--exp_name`; `--hidden_size`; structured log dir `runs/<exp_name>/<seed>/` with `config.json` + `git_sha.txt`.
- `trainer.py`: `hidden_size` added to `MAPPOConfig`, wired through all three networks.
- Packages installed in `marl_comms` env: `pytest`, `absl-py`, `gym`.

---

## Phase 1 — Verify & tighten the DDCL toy-problem implementation ✅ DONE

**Gate:** `pytest` 51 passed, 0 skipped, 0 failed. Bit-cost formula fixed and unit-tested. `configs/baseline_v0.yaml` created. Key finding MATH-002: NSD variance = δ²/4 (not δ²/6).

Goal: produce a **logically correct, test-covered baseline** we trust before sweeping. Take only mathematical principles from the papers; let Phase 2 find good hyperparameters.

### 1.1 Environment audit (`CommunicatingGoal_env.py`, `CommunicatingGoal_vec_env.py`)

- Verify numerics: grid 8, episode 50, 2 agents, 5-action listener space, step penalty −0.01, success reward +1.
- Verify sampling weights `[0.515, 0.258, 0.129, 0.064, 0.031, 0.003]` and document the 6 goal positions.
- Verify observations are raw `(x, y)` ints without accidental normalisation.
- Fill in skipped tests: same seed → bit-identical trajectories; vec and non-vec envs agree.

### 1.2 Channel layer audit (`channels.py`)

Verify **mathematical principles** via Monte-Carlo unit tests (≥10⁵ samples per fixed `z`):

- `DDCL_SD`: `E[ẑ−z] ≈ 0`, `Var(ẑ−z) ≈ δ²/12`, `∂ẑ/∂z = 1`, `Corr(e, z) ≈ 0`.
- `DDCL_NSD` — the existing code already implements the proposal's Algorithm 1 (TPDF dual-path STE). Verify via Thm 5 moments: `E[e|z] ≈ 0`, `Var(e|z) ≈ δ²/6`, `Cov(e, z) ≈ 0`. Produce a side-by-side table of satisfied properties.
- **Fix bit-cost formula**: change `log₂(2|z|/δ + 1)` → `log₂(|z|/δ + 1)` in `channels.py:61,94`. Add unit test confirming the same ordering of configs.

### 1.3 Network & trainer audit (`network.py`, `trainer.py`, `train.py`)

- Confirm speaker output is unbounded (`z ∈ ℝ^{z_dim}`, no squashing).
- Confirm `λ · L_comms` enters **actor** loss only; critic has no comms path.
- Confirm fresh dither per minibatch (mock-channel test).
- Add runtime shape asserts and NaN/Inf guards on `z`.
- Add per-goal bit-allocation logging hook.

### 1.4 Deliverables

- All skipped tests turned green or explicitly justified as deferred.
- `configs/baseline_v0.yaml` — correct (not yet optimal) starting config.
- Short 5-seed smoke run + analysis pipeline emits plots (numbers not yet meaningful).

---

## Phase 2 — Hyperparameter sweep to fix the baseline 🔶 SWEEPS RUNNING

**Infrastructure complete. Stage A sweep active (~98/2175 runs done as of 2026-04-24).**

**What is done:**
- `analysis/` module: `load_runs.py`, `stats.py`, `plots.py`, `sweep_convergence.py`, `report_baseline.py`, `paper_figures.py` (12 figures)
- `docs/STATS.md`: plain-English guide to all 8 statistical methods
- `onpolicy/scripts/sweeps/toyproblem/`: `configs/sweep_stage_a.yaml`, `configs/sweep_stage_b.yaml`, `sweep_wrapper.py`, `run_sweep.py`
- Baseline channels in `channels.py`: `additive_uniform`, `gaussian`, `ste4/8/16`
- `experiments/`: `validate_environment.py` (5/5 pass), `validate_mappo.py` (6/6 pass), `run_channel_comparison.py`
- `tests/test_channels.py`: 28 new tests for baseline channels (79 total, all passing)
- ~~Run channel comparison~~ ✅ DONE — 40 runs (8 channels × 5 seeds); report + 12 paper figures in `results/toyproblem/channel_comparison/`
- Stage A sweep running — see `docs/README.md §5` for monitor/resume commands

**What remains after Stage A completes:**
1. Regenerate full Stage A report and 12 paper figures (run `report_baseline.py` + `generate_all_paper_figures`)
2. Check convergence gate (all 4 criteria); if FAIL, diagnose and extend grid
3. Run Stage B sweep (architecture + PPO hyperparameters)
4. Stage B convergence gate
5. (Optional) Stage C λ fine-sweep for clean rate–distortion frontier
6. Freeze `configs/baseline_best.yaml`

**Fixes applied (all resolved — see `docs/ISSUES_TRACKER.md` for details):**
- CODE-004 (2026-04-23): `true_bits_per_msg` now logged to CSV; both rate-distortion plots generated
- CODE-007 (2026-04-24): `additive_uniform` added to Stage A sweep config
- CODE-008 (2026-04-24): `none` channel redundancy auto-skipped in `run_sweep.py` (~1050 wasted runs saved)
- CODE-009 (2026-04-24): `_run_key()` truncation bug fixed; 95 existing runs recovered from checkpoint args

Goal: identify the best basic-DDCL configuration from scratch; freeze as `configs/baseline_best.yaml`.

### 2.1 Sweep infrastructure

New `onpolicy/scripts/sweeps/toyproblem/` mirroring `sweeps/predatorprey/`:
- `configs/sweep_baseline.yaml`, `sweep_wrapper.py`, `run_sweep.py`.
- Every run writes `runs/<sweep_id>/<run_id>/metrics.csv` (+ parquet once pyarrow installed).

**Staged coordinate descent:**
- **Stage A** — sweep `{channel, λ, δ, z_dim}` at fixed `hidden=64, lr=3e-4, n_envs=16, n_steps=256, 1M steps, seeds {0..4}`.
- **Stage B** — fix Stage A winner; sweep `{hidden_size, lr, batch_size, ppo_clip, gae_lambda, entropy_coef}`.
- **Stage C** — re-sweep `λ` over narrow range for clean rate–distortion frontier.

### 2.2 Sweep axes

**Stage A**

| Axis | Values |
|------|--------|
| `channel` | {sd, nsd, additive_uniform, none} — `none` deduplicated to 1 run per z_dim |
| `lambda_comms` | {0, 1e-5, 1e-4, 5e-4, 1e-3, 4e-3, 1e-2, 3e-2} |
| `delta` | {0.5, 1.0, 5.0, 10.0, 15.0, 20.0} |
| `z_dim` | {1, 2, 3} |

**Stage B**

| Axis | Values |
|------|--------|
| `hidden_size` | {32, 64, 128, 256} |
| `lr` | {1e-4, 3e-4, 5e-4, 1e-3, 3e-3} |
| `batch_size` | {64, 128, 256, 512} |
| `ppo_clip` | {0.1, 0.2, 0.3} |
| `gae_lambda` | {0.9, 0.95, 0.99} |
| `entropy_coef` | {0, 0.001, 0.01, 0.05} |

### 2.3 Sweep convergence criterion (`analysis/sweep_convergence.py`)

A sweep is **done** when ALL hold:
1. **Interior optimum** — winner is not at any axis boundary (else extend range).
2. **Statistical stability** — winner vs runner-up not significant at α=0.05 (paired permutation).
3. **Seed stability** — winner is first-ranked under mean, median, AND IQM.
4. **Pareto non-dominance** — winner not strictly dominated on `(success_rate, bits/episode)`.

### 2.4 Statistical analysis scripts (`analysis/`)

| Script | Purpose |
|--------|---------|
| `load_runs.py` | Tidy DataFrame from raw CSVs across seeds |
| `stats.py` | Bootstrap CI, paired permutation, Wilcoxon, IQM+stratified bootstrap, Pareto test, gradient-variance diagnostic |
| `plots.py` | Rate–distortion frontier, training curves with seed-bands, per-goal bit allocation |
| `sweep_convergence.py` | Automated convergence gate |
| `report_baseline.py` | Emit `results/toyproblem/baseline.md` |

All statistical methods explained in plain English in `docs/STATS.md`.

### 2.5 Deliverable

- `configs/baseline_best.yaml` frozen.
- `results/toyproblem/baseline.md` with tables + plots + convergence decision.
- `docs/STATS.md`.

### 2.6 Research Paper Figures (`analysis/paper_figures.py`)

Each figure function carries a docstring with three annotated fields:
- **Hypothesis** — the scientific claim being tested
- **Analysis** — what the plot shows and how it is constructed
- **Conclusion** — what a reader takes away if the hypothesis holds

| Figure | Function | Hypothesis |
|--------|----------|------------|
| Fig 1 | `plot_paper_rate_distortion` | DDCL lies on the Pareto frontier; STE uses far more bits for the same SR |
| Fig 2 | `plot_paper_training_curves` | All channels converge; λ>0 trades convergence speed for compression |
| Fig 3 | `plot_paper_per_goal_allocation` | Speaker learns -log₂(p_i) allocation without explicit supervision |
| Fig 4 | `plot_paper_lambda_sensitivity` | Smooth rate-distortion tradeoff; knee identifies optimal λ* |
| Fig 5 | `plot_paper_channel_efficiency` | DDCL bits/H(G) close to 1; STE far above 1 |
| Fig 6 | `plot_paper_sd_nsd_comparison` | NSD ≈ SD on Pareto frontier (synchrony-free is free in performance) |
| Fig 7 | `plot_paper_surrogate_calibration` | Surrogate is a monotone proxy; true bits must be reported |
| Fig 8 | `plot_paper_zdim_scaling` | SR plateaus at z_dim=2; per-dim bits decrease (factored encoding) |
| Fig 9 (×2) | `plot_paper_per_goal_dynamics` | Entropy-optimal allocation emerges from training, not hardcoded — plotted separately for SD and NSD |
| Fig 10 (×2) | `plot_paper_delta_lambda_heatmap` | Interior optimum exists in (δ,λ) grid; convergence gate criterion 1 — plotted separately for SD and NSD |

**Total: 12 output files** (Fig 9 and Fig 10 each produce one plot per channel type).
`generate_all_paper_figures(df, summary, agg, out_dir)` runs all 12 in one call; errors per figure are caught and logged without aborting the batch.

**Uncertainty model for per-goal allocation (Fig 3):**
Two independent sources of deviation from the optimal -log₂(p_i) line:
1. **Jensen gap**: the surrogate log₂(|z|/δ+1) overestimates true transmission bits
   by ≈0–0.3 bits in the typical operating regime. Modelled as ±0.20 bit symmetric band.
2. **RL sampling noise**: goal i drives only p_i × N_updates gradient updates.
   Expected deviation ≈ 0.5 / sqrt(p_i × N_updates) bits. For goal 5 (p=0.003)
   at N=244 updates this is ≈0.93 bits — 13× larger than for goal 0 (p=0.515).
Combined: σ_total(i) = sqrt(σ_RL² + σ_Jensen²), shown as shaded envelope.

**Generate all paper figures (after sweeps complete):**
```python
from onpolicy.envs.toyproblem.analysis.load_runs import load_sweep, final_metrics, seed_aggregate
from onpolicy.envs.toyproblem.analysis.paper_figures import generate_all_paper_figures

df = load_sweep("runs/toyproblem/sweep_stage_a")
summary = final_metrics(df)
agg = seed_aggregate(summary, group_cols=["channel","lambda_comms","delta","z_dim"])
generate_all_paper_figures(df, summary, agg, out_dir="results/toyproblem/sweep_stage_a/figures")
```

---

## Phase 3 — Roll out the 4 pillars (order: P2 → P1 → P4 → P3)

**Recipe for each pillar:** Derive loss → Implement (toggleable, baseline bit-identical when off) → Unit tests → Sanity run → Pillar-specific sweep (§2.3 gate) → `configs/pillarN_best.yaml` → `results/toyproblem/pillarN/baseline.md`.

### 3.1 P2 — Entropy-model communication cost

- New `entropy_model.py`: discretised logistic mixture, `K` components, MLP context `h`. Valid PMF output.
- `channels.py`: expose quantised integer `m` in forward info dict.
- `trainer.py`: `--loss_comms {magnitude, entropy, both}`.
- **Goal**: close the Shannon gap; confirm `L_ent → H(m)`.
- **Sweep axes**: `K ∈ {3,5,10,20}`, `h ∈ {32,64,128}`, `lr_qphi ∈ {1e-4,3e-4,1e-3}`, re-sweep `λ`.

### 3.2 P1 — Per-channel δ (4 modes)

- `channels.py`: `--delta_mode {fixed_scalar, fixed_per_channel_scalar, learned_scalar, learned_per_channel}`.
- `fixed_per_channel_scalar` = heuristic hand-set values; `learned_per_channel` = `δ_k = softplus(α_k)`.
- **Tests**: Thm 4 per-channel unbiasedness, `Var(e_k) = δ_k²/12`, gradient identity per channel.
- **Sweep axes**: `delta_mode ∈ {all 4}`, `z_dim ∈ {1,2,4,8}`, `alpha_init` variants.
- **Analysis**: plot learned `δ_k` trajectories vs per-channel mutual information with the goal.

### 3.3 P4 — Rao-Blackwell gradient estimator

- `channels.py`: `--grad_estimator {single, rao_blackwell}`. Two forward passes per step when enabled.
- **Caveat**: effect may be statistically null at 2-agent toy scale. Report honestly; re-test in Phase 4.2 at scale.
- **Gradient-variance diagnostic**: empirical per-parameter gradient variance over 256 minibatches at fixed checkpoints; report log-ratio with bootstrap CIs.
- **Tests**: `E[g_RB] ≈ E[g_single]` (unbiased) and `Var(g_RB) ≤ Var(g_single)` (Thm 6) over ≥10⁴ MC samples on a scalar objective.

### 3.4 P3 — TPDF / synchrony-free deployment

- The existing `DDCL_NSD` already faithfully implements Algorithm 1 of the proposal. Phase 3.4 formalises this, resolves the TPDF bit-cost formula question, and adds dither-method comparison.
- **Loss formula**: derive `E[|m||z]` under triangular noise; check if magnitude surrogate needs correction vs subtractive case. Document + MC test.
- **PRNG-desync robustness**: subtractive should collapse, TPDF invariant. Script: `analysis/prng_robustness.py`.
- **Dither comparison** — implement all four:
  - `subtractive` (SD) — requires shared PRNG; exact unbiasedness + independence.
  - `tpdf` (NSD) — no shared PRNG; unbiased + constant variance (Thm 5).
  - `additive_uniform` — rectangular, non-subtractive; unbiased but signal-dependent variance (Schuchman 1st order only).
  - `gaussian` — biased and signal-dependent; negative control.
- Compare on: training performance, deployment robustness, empirical moment conditions.

---

## Phase 4 — Combined framework & scaling laws

### 4.1 Combined "DDCL Unleashed"

- `configs/unleashed.yaml`: P1 (learned_per_channel) + P2 (entropy) + P4 (RB) + P3 (TPDF) at per-pillar best.
- Baselines: float passthrough, STE {4,8,16} bits, additive-uniform, Gaussian dither. ≥10 seeds.
- **Leave-one-out ablation**: confirm each pillar contributes marginally.

### 4.2 Scaling laws (including #agents)

- **Number of agents**: M-speakers/1-listener, 1-speaker/N-listeners, K-paired; vary M,N,K ∈ {2,4,8,16,32}.
- **Grid**: {8,16,32,64}; **#goals**: {6,12,24,48} Zipf-sampled; **entropy knob**; **z_dim** {1,2,4,8}.
- Fit `bits_per_episode` vs `H(X)`, rate–distortion frontier vs scale.
- Re-test P4 variance-reduction claim at scale.

### 4.3 Stress tests

- P3 PRNG-desync robustness at scale.
- Non-stationary goal distribution — does entropy model adapt faster than magnitude surrogate?

---

## Phase 5 — Documentation & reproducibility

- Finalise `docs/README.md`, `docs/MATH.md`, `docs/STATS.md`, `docs/RESULTS_INDEX.md`.
- `scripts/reproduce/` — one-line entry points for every canonical experiment.
- `REPRODUCE.md` — cold reader can regenerate every headline result end-to-end.

---

## Critical files

### Core implementation (all in `onpolicy/envs/toyproblem/`)

| File | Role |
|------|------|
| `CommunicatingGoal_env.py` | Single-env; goal distribution, reward, listener start sampling |
| `CommunicatingGoal_vec_env.py` | Vectorised env; auto-reset per slot |
| `channels.py` | `IdentityChannel`, `DDCL_SD`, `DDCL_NSD`; comms loss |
| `network.py` | `SpeakerNetwork`, `ListenerActor`, `Critic` |
| `trainer.py` | `MAPPOConfig`, `MAPPOTrainer`; PPO update loop |
| `buffer.py` | `RolloutBuffer`; GAE computation |
| `train.py` | Entry point; `set_seed`, `--exp_name`, structured log dir |

### To be created / status

| Path | Phase | Status | Purpose |
|------|-------|--------|---------|
| `entropy_model.py` | 3.1 | 🔲 TODO | P2: discretised logistic mixture entropy model |
| `analysis/` | 2 | ✅ DONE | `load_runs`, `stats`, `plots`, `sweep_convergence`, `report_baseline`, `paper_figures` (12 figs); `prng_robustness` and `scaling_plots` deferred to Phase 3–4 |
| `experiments/` | 2 | ✅ DONE | `validate_environment.py`, `validate_mappo.py`, `run_channel_comparison.py` |
| `configs/baseline_v0.yaml` | 1 | ✅ DONE | Correct (not yet optimal) starting config |
| `configs/baseline_best.yaml` | 2 | 🔲 PENDING | Freeze after Stage A/B/C sweeps pass convergence gate |
| `configs/pillar{1-4}_best.yaml` | 3 | 🔲 TODO | Per-pillar best configs |
| `configs/unleashed.yaml` | 4 | 🔲 TODO | All-pillars combined config |
| `scaling/` | 4.2 | 🔲 TODO | Multi-agent toy-problem variants |
| `results/toyproblem/` | 2–4 | 🔶 PARTIAL | `channel_comparison/` complete (40 runs + 12 paper figures); `sweep_stage_a/` pending full sweep |
| `docs/STATS.md` | 2 | ✅ DONE | Plain-English guide: 8 statistical methods |
| `docs/RESULTS_INDEX.md` | 5 | 🔲 TODO | Claim → result file → stat test pointers |
| `onpolicy/scripts/sweeps/toyproblem/` | 2 | ✅ DONE | `sweep_stage_a.yaml`, `sweep_stage_b.yaml`, `sweep_wrapper.py`, `run_sweep.py` |

---

## Verification gates

| Phase | Gate | Status |
|-------|------|--------|
| 0 | `pytest` 28 passed, 0 failed | ✅ DONE |
| 1 | All skipped MC/gradient tests green; bit-cost formula fixed and tested | ✅ DONE — 51 passed, 0 skipped, 0 failed |
| 2 (infra) | analysis/, sweep scripts, baseline channels, validation scripts | ✅ DONE — 79 passed, 0 failed |
| 2 (sweeps) | Convergence gate passes (§2.3 all 4 criteria); `baseline_best.yaml` frozen | 🔶 RUNNING — Stage A ~98/2175 done |
| 3 (per pillar) | Pillar-off = baseline bit-identical; pillar sweep passes §2.3 gate | 🔲 TODO |
| 4 | Unleashed on Pareto frontier vs all baselines; leave-one-out ablation passes | 🔲 TODO |
| 5 | Cold reader reproduces any headline result from `REPRODUCE.md` | 🔲 TODO |
