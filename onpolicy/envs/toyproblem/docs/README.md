# DDCL Toy Problem — Reader's Guide

> **Status:** Living document — updated with each project phase.  
> **Related papers:** "Learning What to Say and How Precisely" (Kapoor et al., 2025) and
> "Stochastic Quantisation via Dithering: A Plug-and-Play Layer for Communication-Efficient
> Multi-Agent Reinforcement Learning" (the proposal).

---

## Table of Contents

0. [Documentation Map](#0-documentation-map)
1. [Environment](#1-environment)
2. [Communication Channels](#2-communication-channels)
3. [Network Architecture](#3-network-architecture)
4. [Training: MAPPO](#4-training-mappo)
5. [Running Experiments — Full Procedure](#5-running-experiments--full-procedure)
6. [CLI Reference](#6-cli-reference)
7. [Offline Analysis](#7-offline-analysis)
8. [Project Phases](#8-project-phases)
9. [Known Deviations from the Papers](#9-known-deviations-from-the-papers)

---

## 0. Documentation Map

Every file in `docs/` and in the toyproblem root has a specific, non-overlapping purpose.
Read this section first to know where to look for any given type of information.

| File | Purpose |
|------|---------|
| **`docs/README.md`** *(this file)* | Reader-facing walkthrough: environment mechanics, channel math, network architecture, all CLI flags, log directory layout, step-by-step experiment procedure. Start here. |
| **`docs/MATH.md`** | All mathematical derivations, symbol-to-code mappings, proofs, and formula deviations from the papers. Covers bit-cost formula, SD/NSD theorem verification, Jensen gap analysis, two-quantity bit framework (surrogate vs true bits), per-pillar stubs. |
| **`docs/STATS.md`** | Plain-English guide to every statistical method used in this project: bootstrap CI, IQM + stratified bootstrap, paired permutation test, Wilcoxon signed-rank, Pareto frontier test, convergence gate, gradient-variance diagnostic, and reporting standards. |
| **`docs/ISSUES_TRACKER.md`** | Authoritative log of every bug, installation issue, test failure, math error, or reproducibility blocker encountered during development. Each entry has: symptom, root cause, fix, status, reproducibility impact. Also contains the master reproducibility checklist and a section for pre-existing upstream bugs. |
| **`CONTEXT.md`** | Living session log. Tracks the true current state of the project: phase status, what is fully done, immediate next tasks, key decisions, and a per-session log of what was done, found, and fixed. **Read this at the start of every session.** |
| **`PLAN.md`** | Full approved 5-phase project plan with detailed sub-tasks, success criteria, and gating conditions for each phase. Authoritative reference for scope and phase sequencing. |
| **`results/toyproblem/`** | Auto-generated result artefacts (repo root, gitignored). Each sub-directory corresponds to one experiment (e.g. `channel_comparison/`, `sweep_stage_a/`). Contains `baseline.md` (convergence gate report) and `figures/` with all plots. Do not edit by hand — regenerate by running `report_baseline.py` and `generate_all_paper_figures`. |
| **`runs/toyproblem/`** | Raw training output (repo root, gitignored). Each sub-directory corresponds to one experiment. Contains per-seed subdirectories with `metrics.csv`, `config.json`, `final.pt`. |

---

## 1. Environment

**File:** `CommunicatingGoal_env.py` (single env), `CommunicatingGoal_vec_env.py` (vectorised)

### Task

Two agents cooperate on an 8×8 grid:

| Agent | Role | Observation | Action |
|-------|------|-------------|--------|
| Speaker (idx 0) | Stationary | Goal `(x, y)` coordinates | None (no physical action) |
| Listener (idx 1) | Mobile | Own `(x, y)` coordinates | `{STAY, UP, DOWN, LEFT, RIGHT}` |

The listener cannot see the goal. The speaker must communicate the goal location through its
policy output (the communication vector `z`), which is passed through a quantisation channel
before the listener receives it. The environment itself has no communication machinery — that
is intentionally left to the policy.

### Goal Distribution

Six goal positions are sampled at the start of each episode with the following fixed probabilities:

| Goal | Position `(x, y)` | Probability |
|------|------------------|-------------|
| 0 | `(0, 0)` | 51.5 % |
| 1 | `(7, 7)` | 25.8 % |
| 2 | `(3, 4)` | 12.9 % |
| 3 | `(4, 3)` | 6.4 % |
| 4 | `(1, 6)` | 3.1 % |
| 5 | `(6, 1)` | 0.3 % |

The non-uniform distribution is deliberate: an optimal communication protocol should allocate
fewer bits to the most frequent goals (Huffman-analogue). The Shannon entropy of this
distribution is **H(X) ≈ 1.81 bits**.

### Reward

```
R(t) = +1.0   if listener reaches goal at step t
       -0.01  otherwise (step penalty)
```

Both agents receive the same scalar reward (fully cooperative). The episode ends when the
listener reaches the goal, or after `max_steps = 50` steps (truncation).

### Listener Start

At each episode reset, the listener is placed uniformly at random on any cell **other than the
goal cell**. This is achieved via a bijective shift: `flat_idx ∈ [0, G²-2]`, shifted past the
goal cell's linear index — no rejection sampling needed.

### Coordinate Convention

Positions are `(x, y)` where `x` is column (0 = left) and `y` is row (0 = top).
Actions: `UP=[0,-1]`, `DOWN=[0,+1]`, `LEFT=[-1,0]`, `RIGHT=[+1,0]`.
Linear index used internally: `flat = x * G + y`.

---

## 2. Communication Channels

**File:** `channels.py`

All channels share the interface:

```python
z_hat, info = channel(z)          # forward pass
loss_per_elem = channel.comms_loss(z)   # bit-cost tensor, shape = z.shape
```

### `IdentityChannel` — no quantisation

`z_hat = z`, `comms_loss = 0`. Used with `--channel none` to establish a no-communication-cost
baseline where the policy is free to pass any floating-point value.

### `DDCL_SD` — Subtractive Dithering

```
Training path:  ε ~ U(-δ/2, +δ/2)  →  z_hat = z + ε          (∂z_hat/∂z = 1)
Deployment path (detached, logging only):
    eps ~ U(-δ/2, +δ/2)
    m = floor((z + eps) / δ)          (discrete integer message)
    C(m) = (m + 0.5) · δ              (bin centre)
    z_hat_deploy = C(m) - eps
```

The training path is the distributional collapse of the full pipeline (Theorem A.1 /
Theorem 2 of the proposal): `z_hat_deploy ~ᵈ z + e` where `e ~ U(-δ/2, +δ/2)` and
`e ⊥⊥ z`. Therefore adding fresh uniform noise during training is an unbiased estimator of
the reconstructed value, giving `∂z_hat/∂z = 1` exactly.

The deployment path runs detached (no gradients) and is used only for bit-logging and to
verify deployment parity. In a real deployed system, sender and receiver share `eps` via a
synchronized PRNG.

### `DDCL_NSD` — Non-Subtractive / TPDF Dithering

```
ν = u₁ + u₂,  u₁,u₂ ~ U(-δ/2, +δ/2)   (triangular / TPDF noise)
m = floor((z + ν) / δ)                   (detached)
z_hat_true = (m + 0.5) · δ               (bin centre, no ε subtraction)
z_hat = z + sg(z_hat_true - z)           (STE: forward = z_hat_true, grad = 1)
```

`sg(·)` is the stop-gradient operator. By Schuchman's theorem (Theorem 5 of the proposal),
under TPDF noise the expected reconstruction `E[z_hat_true | z] = z`, so the STE is an
unbiased gradient estimator. The receiver does **not** need any shared PRNG — it outputs
`C(m)` directly. This removes the synchronization constraint present in `DDCL_SD`.

This implementation is faithful to Algorithm 1 of the proposal.

> **Key finding (MATH-002, Phase 1):** Reconstruction variance `Var(ẑ-z | z) = δ²/4`
> (not δ²/6 as the proposal stated). NSD noise is 3× larger than SD (δ²/4 vs δ²/12).
> See `docs/MATH.md §3.2` for the analytical derivation and Monte-Carlo verification.

### Baseline Channels (Phase 2, `channels.py`)

Three additional channels for systematic comparison against DDCL:

| Class | `--channel` | `comms_loss` | Role |
|-------|-------------|-------------|------|
| `AdditiveUniformChannel` | `additive_uniform` | `log₂(\|z\|/δ+1)` | 1st-order Schuchman — unbiased but signal-dependent variance. Intermediate control between SD/NSD and Gaussian. |
| `GaussianChannel` | `gaussian` | `log₂(\|z\|/δ+1)` | Additive Gaussian (σ=δ/√12, matched SD variance). Biased after quantization. Negative control — expected to be strictly worst. |
| `STEChannel` | `ste4`, `ste8`, `ste16` | constant `bits` | Fixed-rate straight-through quantizer. Clips z to `[-ste_clip, +ste_clip]`, rounds to 2^bits uniform levels. `comms_loss = bits` always. |

All three share the same `(z_hat, info) = channel(z)` + `comms_loss(z)` interface. Build with:
```python
from onpolicy.envs.toyproblem.channels import build_channel
ch = build_channel("ste8", delta=1.0, ste_clip=10.0)
```

### Communication Loss (Bit-Cost Surrogate)

`DDCL_SD`, `DDCL_NSD`, `additive_uniform`, and `gaussian` all use the same variable-rate
surrogate for expected message bit length:

```
L_comms[k] = log₂( |z[k]| / δ + 1 )   per element k
```

> **Deviation from the proposal.** The proposal writes `log₂(2|z|/δ + 1)`.
> We drop the constant `2` because it is a constant multiplier inside a `log(1 + cx)` term
> and does not change the argmin with respect to `z`. The full derivation is in
> `docs/MATH.md`.

---

## 3. Network Architecture

**File:** `network.py`

All networks use **orthogonal initialisation** (`std=√2` for hidden layers, `std=0.01` for
output layers) and **GELU** activation. This follows the CleanRL / MAPPO-on-policy convention.

| Network | Input | Hidden | Output |
|---------|-------|--------|--------|
| `SpeakerNetwork` | goal `(x,y)` → 2-dim | 1 × `hidden` GELU | `z_dim`-dim real (unbounded, no activation) |
| `ListenerActor` | `[listener_pos, z_hat]` → `2 + z_dim` | 1 × `hidden` GELU | logits for 5 actions via `Categorical` |
| `Critic` | `[listener_pos, goal]` → 4-dim | 1 × 32 GELU | scalar value |

> **Paper deviation.** The DDCL paper uses `[64, 64]` ReLU two-layer MLPs. The codebase
> uses a **single hidden layer** with configurable width (default `hidden=16`) and GELU.
> The paper's exact architecture will be matched once Phase 2 sweeps identify the best `hidden_size`.

The speaker output is **unbounded real-valued** — no sigmoid or tanh on `z`. This is
essential: the bit-cost formula `log₂(|z|/δ + 1)` encourages small `|z|` through the loss,
not through an architectural constraint. Squashing `z` would interfere with gradient flow.

The critic receives the **full state** `[listener_pos, goal]` (centralized training) but the
listener actor receives only `[listener_pos, z_hat]` (decentralized execution).

---

## 4. Training: MAPPO

**Files:** `trainer.py`, `buffer.py`

### Algorithm

Centralized Training with Decentralized Execution (CTDE) using Multi-Agent PPO:

```
for each update:
    collect n_steps × n_envs transitions via act_and_value (no_grad)
    bootstrap final value with get_value
    compute GAE advantages and returns (raw scale, via ValueNorm denormalization)
    for update_epochs:
        for each minibatch:
            recompute z = speaker(goal)          ← fresh forward (new ε resampled)
            z_hat = channel(z)
            dist  = listener([listener_pos, z_hat])
            PPO clip loss + critic MSE + λ · L_comms
            Adam step
```

### Key Design Decisions

**Fresh dither per minibatch.** The channel noise `ε` (or `ν` for NSD) is resampled on every
call to `channel(z)` during the update loop. This is required by Theorem A.1 / Theorem 2: the
unbiased gradient identity holds because the noise is independent of `z`. Reusing the rollout
noise would break the independence assumption.

**Communication loss on pre-quantization `z`.** `L_comms = channel.comms_loss(z_new)` is
evaluated on the speaker output `z`, not on `z_hat`. This is correct: the bit-cost formula
measures the magnitude of the signal the speaker is trying to transmit — it incentivises
the speaker to output small `|z|` for frequent goals.

**Critic uses full state.** The critic sees `[listener_pos, goal]` — it observes the true goal,
which the listener cannot. This is standard CTDE: the value function is centralised at training
time to reduce variance.

**ValueNorm.** Critic outputs are normalised via PopArt-style running mean/variance
(`onpolicy/utils/valuenorm.py`). GAE is computed on raw-scale values (denormalized once);
critic loss is on normalised returns.

**Single Adam optimiser.** All trainable parameters (speaker, listener, critic, any channel
parameters) share one Adam optimiser. This is the simplest setup; per-component learning rates
can be added in future phases if needed.

### GAE at Episode Boundaries

Episodes end mid-rollout. The vectorized env auto-resets done slots before returning the
next observation. The `done` flag in the buffer is set to 1.0 at terminal transitions,
which zeroes out the bootstrap term in the GAE delta: `delta = r + γ·V_next·(1-done) - V`.
This is correct: episode boundaries are handled purely by the mask, regardless of whether
`V_next` is the new-episode value or not.

---

## 5. Running Experiments — Full Procedure

This section is the single authoritative guide for running the full experiment pipeline from
a cold start. Follow the steps in order. All commands assume your working directory is
`on-policy/` unless otherwise stated.

> **macOS note.** Every `conda run` command must be prefixed with
> `KMP_DUPLICATE_LIB_OK=TRUE` to suppress an OpenMP library conflict (see INSTALL-005).
> This is shown in all commands below.

---

### Step 0 — One-time environment setup

```bash
# Create and activate the conda environment (if not already done)
conda activate marl_comms

# Install the onpolicy package in editable mode (required for all imports)
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms pip install -e .

# Install required packages
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms pip install pytest absl-py "gym==0.26.2"

# Optional but recommended: parquet output for faster sweep loading
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms pip install pyarrow
```

**Verify setup:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms python -c "import onpolicy; print('OK')"
```

---

### Step 1 — Run the test suite

Run the full pytest suite before any sweep. All 79 tests must pass.

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m pytest onpolicy/envs/toyproblem/tests/ -v
```

Expected: `79 passed, 0 failed`. If any test fails, stop and check `docs/ISSUES_TRACKER.md`.

---

### Step 2 — Validate the environment and MAPPO (optional but recommended)

These scripts check environment statistics and verify the MAPPO training loop before
committing to a multi-hour sweep.

```bash
# Environment: checks goal distribution, episode lengths, reward structure (5/5 checks)
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.experiments.validate_environment --n_episodes 5000

# MAPPO: checks value loss decrease, entropy control, lambda gating, etc. (6/6 checks)
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.experiments.validate_mappo --n_updates 200
```

Both should report all checks passing. Fix any failures before proceeding.

---

### Step 3 — (Optional) Single-run smoke test

Train one configuration for a short run to verify the full pipeline end-to-end.

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.train \
    --exp_name smoke_sd \
    --channel sd \
    --delta 5.0 \
    --lambda_comms 1e-3 \
    --z_dim 2 \
    --total_timesteps 100000 \
    --seed 0 \
    --log_dir runs/toyproblem/smoke
```

Check that `runs/toyproblem/smoke/smoke_sd/0/metrics.csv` was written and contains
`true_bits_per_msg` and `bits_goal_0`…`bits_goal_5` columns.

---

### Step 4 — Channel comparison experiment

This is a fixed single-config comparison across all 8 channel types (5 seeds × ~38s each ≈ 25 min).
It gives a quick qualitative picture before running the full sweep.

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.experiments.run_channel_comparison \
    --seeds 0 1 2 3 4 \
    --total_timesteps 1000000
```

Generate report and plots:
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.analysis.report_baseline \
    --sweep_dir runs/toyproblem/channel_comparison \
    --out_dir results/toyproblem/channel_comparison
```

Results: `results/toyproblem/channel_comparison/baseline.md` + `results/toyproblem/channel_comparison/figures/`.
The convergence gate will FAIL here (only one λ/δ point per channel — not a sweep).
That is expected. Use fig1 to check that `none` sits at high bits and `sd`/`nsd` are
somewhere on a lower-bits frontier.

---

### Step 5 — Stage A sweep (main sweep — ~23 hours on CPU)

The Stage A sweep covers the full grid:
`{sd, nsd, additive_uniform, none}` × `{8 λ values}` × `{6 δ values}` × `{3 z_dims}` × `5 seeds`
= ~2175 meaningful runs after `none` deduplication.

**Dry run first** (verify command list without running anything):
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.scripts.sweeps.toyproblem.run_sweep \
    --config onpolicy/scripts/sweeps/toyproblem/configs/sweep_stage_a.yaml \
    --seeds 0 1 2 3 4 \
    --log_dir runs/toyproblem/sweep_stage_a \
    --dry_run
```

**Launch sweep** (detached with nohup so it survives terminal closure):
```bash
nohup bash -c 'cd "$(pwd)" && KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.scripts.sweeps.toyproblem.run_sweep \
    --config onpolicy/scripts/sweeps/toyproblem/configs/sweep_stage_a.yaml \
    --seeds 0 1 2 3 4 \
    --log_dir runs/toyproblem/sweep_stage_a \
    --resume --shuffle' > /tmp/sweep_stage_a.log 2>&1 &
echo "Sweep PID=$!"
```

**Monitor progress:**
```bash
# Is it still running?
ps aux | grep run_sweep | grep -v grep

# How many runs are complete?
find runs/toyproblem/sweep_stage_a -name "metrics.csv" | wc -l

# Live log tail
tail -f /tmp/sweep_stage_a.log
```

**Resume after interruption** (the `--resume` flag skips already-complete runs):
```bash
nohup bash -c 'cd "$(pwd)" && KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.scripts.sweeps.toyproblem.run_sweep \
    --config onpolicy/scripts/sweeps/toyproblem/configs/sweep_stage_a.yaml \
    --seeds 0 1 2 3 4 \
    --log_dir runs/toyproblem/sweep_stage_a \
    --resume --shuffle' > /tmp/sweep_stage_a.log 2>&1 &
```

> **Important:** The `none` channel runs at most 3 z_dim × 5 seeds = 15 meaningful configs;
> all other λ/δ combos are auto-skipped. The sweep prints "SKIP (none: ...)" for each.

---

### Step 6 — Generate Stage A report and paper figures

All figures (diagnostic + publication-quality) go into a single `figures/` directory under the
sweep output. Run this at any point — even while the sweep is still running — and re-run after it
completes for the final analysis.

```bash
# Standard report + diagnostic plots → results/toyproblem/sweep_stage_a/figures/
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.analysis.report_baseline \
    --sweep_dir runs/toyproblem/sweep_stage_a \
    --out_dir results/toyproblem/sweep_stage_a
```

```bash
# Publication-quality figures (fig1–fig10) → same figures/ directory
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms python -c "
from onpolicy.envs.toyproblem.analysis.load_runs import load_sweep, final_metrics, seed_aggregate
from onpolicy.envs.toyproblem.analysis.paper_figures import generate_all_paper_figures
df = load_sweep('runs/toyproblem/sweep_stage_a')
summary = final_metrics(df)
agg = seed_aggregate(summary, group_cols=['channel','lambda_comms','delta','z_dim'])
generate_all_paper_figures(df, summary, agg,
    out_dir='results/toyproblem/sweep_stage_a/figures')
"
```

**Convergence gate criteria (all 4 must pass before Stage B):**
1. **Interior optimum** — winning λ and δ are not at the boundary of the sweep grid
2. **Statistical stability** — permutation test p < 0.05 vs runner-up
3. **Seed stability** — rank by mean, median, and IQM all agree
4. **Pareto non-dominance** — winner is not dominated on (success_rate, true_bits_per_msg)

If the gate fails, examine the `baseline.md` report for which criteria fail and why.

---

### Step 7 — Stage B sweep (after Stage A gate passes)

Stage B sweeps architecture and PPO hyperparameters while fixing the Stage A winner's
(channel, λ, δ, z_dim). Config: `configs/sweep_stage_b.yaml`.

```bash
nohup bash -c 'cd "$(pwd)" && KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.scripts.sweeps.toyproblem.run_sweep \
    --config onpolicy/scripts/sweeps/toyproblem/configs/sweep_stage_b.yaml \
    --seeds 0 1 2 3 4 \
    --log_dir runs/toyproblem/sweep_stage_b \
    --resume --shuffle' > /tmp/sweep_stage_b.log 2>&1 &
```

Analyse with the same `report_baseline.py` command (point `--sweep_dir` at `runs/toyproblem/sweep_stage_b`).

---

### Step 8 — Freeze baseline config

Once both Stage A and Stage B convergence gates pass, freeze the best config:

```bash
# Edit configs/baseline_best.yaml with the winning hyperparameters, then verify:
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.train \
    --exp_name baseline_best_verification \
    --seed 99 \
    $(cat onpolicy/envs/toyproblem/configs/baseline_best.yaml | \
      python -c "import sys,yaml; d=yaml.safe_load(sys.stdin); \
      print(' '.join(f'--{k} {v}' for k,v in d.items()))")
```

Commit `configs/baseline_best.yaml` and proceed to Phase 3 (pillar implementation).

---

### Quick reference — single runs

```bash
# No-communication baseline
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.train --exp_name none_baseline --channel none --seed 0

# Subtractive dithering (SD)
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.train \
    --exp_name sd_run --channel sd --delta 5.0 --lambda_comms 1e-3 --z_dim 2 --seed 0

# Non-subtractive TPDF dithering (NSD)
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.train \
    --exp_name nsd_run --channel nsd --delta 5.0 --lambda_comms 1e-3 --z_dim 2 --seed 0

# Fixed-rate STE baseline (8-bit)
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.train \
    --exp_name ste8_run --channel ste8 --ste_clip 10.0 --seed 0

# Parallel seeds (bash loop)
for SEED in 0 1 2 3 4; do
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
        python -m onpolicy.envs.toyproblem.train \
        --exp_name sd_multiseed --channel sd --delta 5.0 --lambda_comms 1e-3 \
        --seed $SEED &
done
wait
```

---

## 6. CLI Reference

All arguments are parsed by `train.py::parse_args()`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--exp_name` | str | `"debug"` | Experiment name; used to organise `runs/<exp_name>/<seed>/` |
| `--seed` | int | `42` | Master seed for NumPy, PyTorch, Python `random`, env |
| `--device` | str | `"cpu"` | `"cpu"` or `"cuda"` |
| `--n_envs` | int | `16` | Number of parallel environments |
| `--n_steps` | int | `256` | Rollout length per update |
| `--total_timesteps` | int | `1_000_000` | Total env steps to train for |
| `--gamma` | float | `0.99` | Discount factor |
| `--gae_lambda` | float | `0.95` | GAE λ |
| `--clip_eps` | float | `0.2` | PPO clip ε |
| `--lr` | float | `3e-4` | Adam learning rate |
| `--update_epochs` | int | `10` | PPO epochs per rollout |
| `--num_minibatches` | int | `4` | Minibatches per PPO epoch |
| `--entropy_coef` | float | `0.03` | Entropy bonus coefficient |
| `--max_grad_norm` | float | `0.5` | Gradient clip norm |
| `--hidden_size` | int | `64` | Hidden units in speaker and listener MLPs |
| `--z_dim` | int | `3` | Speaker output / message dimension |
| `--channel` | str | `"none"` | `{none, sd, nsd, additive_uniform, gaussian, ste4, ste8, ste16}` |
| `--delta` | float | `1.0` | Quantisation bin width δ (SD / NSD / additive_uniform / gaussian) |
| `--lambda_comms` | float | `0.0` | Communication loss weight λ |
| `--ste_clip` | float | `10.0` | Clip bound for STE channels (`ste4`, `ste8`, `ste16`) |
| `--log_every` | int | `10` | Console print frequency (in updates) |

**Log directory structure:**

```
runs/
  <exp_name>/
    <seed>/
      metrics.csv      ← per-update metrics (always written)
      config.json      ← full arg namespace snapshot
      git_sha.txt      ← current git commit (if available)
      final.pt         ← checkpoint at end of training
```

---

## 7. Offline Analysis

**Why not use WandB charts directly?**  
WandB applies smoothing to curves in the UI. For publication-quality comparisons we need
raw, unsmoothed values. Every run writes `metrics.csv` to disk; the analysis scripts in
`analysis/` load these directly.

> **Future:** Once `pyarrow` is installed, runs will additionally write `metrics.parquet`
> for faster loading when there are many seeds.

**Analysis scripts (`analysis/` — created Phase 2):**

| Script | Status | Purpose |
|--------|--------|---------|
| `analysis/load_runs.py` | ✅ Done | `load_run`, `load_exp`, `load_sweep`, `final_metrics`, `seed_aggregate` — loads raw CSVs into tidy DataFrames |
| `analysis/stats.py` | ✅ Done | `bootstrap_ci`, `iqm_ci`, `paired_permutation_test`, `wilcoxon_signed_rank`, `pareto_frontier` |
| `analysis/plots.py` | ✅ Done | `plot_training_curves`, `plot_rate_distortion`, `plot_per_goal_bits` (with Jensen+RL uncertainty bands), `plot_sweep_heatmap`, `plot_channel_comparison` |
| `analysis/sweep_convergence.py` | ✅ Done | `check_convergence()` — 4-criterion automated gate; returns `ConvergenceResult` with pass/fail per criterion |
| `analysis/report_baseline.py` | ✅ Done | Full pipeline: load sweep → convergence gate → diagnostic plots in `figures/` → `baseline.md` |
| `analysis/paper_figures.py` | ✅ Done | 12 publication-quality figures (fig1–fig10), each with `Hypothesis / Analysis / Conclusion` annotation; `generate_all_paper_figures()` batch runner |
| `analysis/prng_robustness.py` | 🔲 Phase 3.4 | PRNG desync robustness evaluation |
| `analysis/scaling_plots.py` | 🔲 Phase 4 | Scaling-law figures |

**Statistical methods** are explained in plain English in `docs/STATS.md`.

**CSV column reference** (`metrics.csv` written by `train.py`):

| Column | Description |
|--------|-------------|
| `update` | PPO update index (0-based) |
| `timestep` | Total environment steps elapsed |
| `mean_reward` | Rolling mean reward (last 200 episodes) |
| `success_rate` | Rolling fraction of successful episodes (last 200) |
| `pg_loss` | Mean policy gradient loss over PPO epochs |
| `value_loss` | Mean critic MSE over PPO epochs |
| `entropy` | Mean action entropy |
| `approx_kl` | Approximate KL divergence from old policy |
| `clip_frac` | Fraction of minibatch entries that hit the PPO clip |
| `comms_loss` | Mean λ·L_comms term (training surrogate contribution) |
| `bits_per_msg` | **Surrogate**: Σ log₂(\|z\|/δ+1) per message — differentiable Jensen upper bound used in training loss |
| `true_bits_per_msg` | **True transmission cost**: 32×z_dim for `none`; log₂(\|m\|+1) for quantized channels; B×z_dim for STE |
| `z_norm` | Mean L2 norm of speaker output z |
| `sps` | Steps per second |
| `bits_goal_0`…`bits_goal_5` | Per-goal mean surrogate bits (NaN if goal was not seen in this update) |

---

## 8. Project Phases

| Phase | Goal | Status |
|-------|------|--------|
| **0 — Hygiene** | Docs, tests skeleton, seeding, log structure | ✅ Done (`pytest` 28 pass) |
| **1 — Verify** | Unit-test channels/env, audit math against papers | ✅ Done (`pytest` 51 pass, MATH-002) |
| **2 — Infra** | analysis/, sweep scripts, baseline channels, validation | ✅ Done (`pytest` 79 pass) |
| **2 — Sweeps** | Stage A/B/C sweeps, baseline_best.yaml frozen | 🔲 Pending compute |
| **3.1 — P2** | Entropy-model communication cost | 🔲 Todo |
| **3.2 — P1** | Per-channel δ (heuristic → learned) | 🔲 Todo |
| **3.3 — P4** | Rao-Blackwell gradient estimator | 🔲 Todo |
| **3.4 — P3** | TPDF dithering + dither-method comparison | 🔲 Todo |
| **4 — Scale** | Combined "Unleashed" config + scaling laws | 🔲 Todo |
| **5 — Repro** | Reproducibility scripts, final docs | 🔲 Todo |

---

## 9. Known Deviations from the Papers

See `docs/KNOWN_ISSUES.md` for out-of-scope repository issues.

| Deviation | Paper says | This codebase | Rationale |
|-----------|-----------|---------------|-----------|
| Bit-cost constant | `log₂(2\|z\|/δ + 1)` | `log₂(\|z\|/δ + 1)` | Constant `2` does not affect argmin; see `MATH.md §2.3` |
| NSD variance | `Var = δ²/6` (proposal Thm 5) | `Var = δ²/4` | MATH-002 (Phase 1): analytically derived and MC-verified. See `MATH.md §3.2`. |
| Activation | ReLU | GELU | Standard in modern PPO implementations; swept in Phase 2 |
| Network depth | 2 hidden layers `[64, 64]` | 1 hidden layer, width configurable | To be swept in Phase 2 |
| Default `z_dim` | 1 | 3 (CLI default) | Paper result; use `--z_dim 1` for reproduction |
| Default `hidden` | 64 | 64 (after Phase 0 fix) | CLI default updated in Phase 0 |
| Baseline channels | Not in DDCL paper | `additive_uniform`, `gaussian`, `ste4/8/16` | Phase 2 addition for systematic comparison |
