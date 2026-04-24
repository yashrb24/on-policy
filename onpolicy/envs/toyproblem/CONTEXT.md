# DDCL Unleashed — Project Context

---

## MANDATORY UPDATE PROTOCOL

> **Every Claude session MUST follow this protocol — no exceptions.**
>
> 1. **At session start:** Read this entire file before touching any code.
> 2. **During the session:** Log every file you create or modify in the Session Log as you go.
> 3. **When any issue is encountered** (installation, test failure, unexpected behaviour, documentation gap): add it to `docs/ISSUES_TRACKER.md` immediately, with status and fix.
> 4. **At session end:** Before closing, update:
>    - "Phase Status" table — mark anything newly completed
>    - "What Is Fully Done" — add new entries for completed work
>    - "Immediate Next Tasks" — update to reflect what remains
>    - "Session Log" — add a new entry with: date, what was done, decisions made, issues found, what is next
>    - Keep entries concise (bullet form). Do NOT let the file grow beyond ~300 lines.
>
> **The file must always reflect the true current state of the project.**

---

## Project in One Paragraph

We are building a rigorous testbed for DDCL (Differentiable Discrete Communication Learning) on a toy 2-agent MARL problem (`CommunicatingGoal`). Goal is NOT to reproduce paper numbers — it is to: (1) verify mathematical correctness of the implementation, (2) find a strong baseline via systematic hyperparameter sweeps, (3) implement and ablate 4 principled extensions ("pillars") from a companion proposal, and (4) show scaling behaviour including number-of-agents scaling. All results must be statistically honest: no WandB smoothing, raw CSVs archived, bootstrap CIs, permutation tests, IQM.

**Full plan:** `PLAN.md` in this directory. **Issue tracker:** `docs/ISSUES_TRACKER.md`.

---

## Environment Setup

- **Conda env:** `marl_comms` (Python 3.10)
- **Run all commands as:** `KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms <cmd>`
  - The `KMP_DUPLICATE_LIB_OK=TRUE` prefix is required on macOS to suppress an OpenMP library conflict.
- **Editable install:** from `on-policy/` run `pip install -e .` (required so `import onpolicy` resolves)
- **Run tests:** `KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms python -m pytest onpolicy/envs/toyproblem/tests/ -v`
- **Installed packages (confirmed working):** torch, numpy, pytest, absl-py, gym==0.26.2
- **Pending installs:** pyarrow (needed for parquet output — currently falls back to CSV only)

---

## Repository Layout (toyproblem directory)

```
on-policy/onpolicy/envs/toyproblem/
├── CommunicatingGoal_env.py      # Single env: 8×8 grid, 2 agents, 6 goals, gym.Env
├── CommunicatingGoal_vec_env.py  # Vectorised N-slot env with per-slot auto-reset
├── channels.py                   # All channel classes + H_GOAL_BITS, GOAL_OPTIMAL_BITS constants
├── network.py                    # SpeakerNetwork, ListenerActor, Critic
├── buffer.py                     # RolloutBuffer with GAE + goal_ids tracking
├── trainer.py                    # MAPPOTrainer, MAPPOConfig; logs bits_per_msg + true_bits_per_msg
├── train.py                      # Entry point: set_seed, structured log dir, full CSV header
├── runs/                         # ← NEW: all toyproblem run outputs go here
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Fixtures: all 8 channels, envs
│   ├── test_channels.py          # 79 tests passing (SD/NSD/additive/gaussian/ste4/8/16 + build)
│   ├── test_env_determinism.py   # 20 tests passing
│   └── test_trainer.py           # 8 tests passing
├── analysis/
│   ├── __init__.py
│   ├── load_runs.py              # load_run, load_exp, load_sweep, final_metrics, seed_aggregate
│   ├── stats.py                  # bootstrap_ci, iqm_ci, paired_permutation_test, pareto_frontier
│   ├── plots.py                  # training curves, rate-distortion, per-goal bits (+ uncertainty bands)
│   ├── sweep_convergence.py      # 4-criterion convergence gate
│   ├── report_baseline.py        # Full report pipeline → baseline.md + plots
│   └── paper_figures.py          # ← NEW: 10 paper figures with hypothesis/analysis/conclusion
├── experiments/
│   ├── __init__.py
│   ├── validate_environment.py   # 5/5 pass
│   ├── validate_mappo.py         # 6/6 pass
│   └── run_channel_comparison.py # 8-channel × 5-seed comparison
├── configs/
│   └── baseline_v0.yaml          # Phase 1 verified baseline
├── docs/
│   ├── README.md
│   ├── MATH.md                   # §10 added: two-quantity bit framework, H(G), Jensen gap
│   ├── KNOWN_ISSUES.md
│   ├── STATS.md
│   ├── ISSUES_TRACKER.md         # CODE-001 through CODE-009 all resolved
│   └── results/
│       ├── channel_comparison/   # 40 runs complete; standard plots + 9 paper figures
│       └── sweep_stage_a/        # partial results (98 runs); updates as sweep runs
├── PLAN.md
└── CONTEXT.md                    # This file
```

**To be created (by phase):**
- `configs/` — baseline_v0.yaml (Phase 1), baseline_best.yaml (Phase 2), pillarN_best.yaml (Phase 3), unleashed.yaml (Phase 4)
- `analysis/` — load_runs.py, stats.py, plots.py, sweep_convergence.py, report_*.py (Phase 2)
- `entropy_model.py` — P2 discretised logistic mixture (Phase 3.1)
- `docs/STATS.md` — plain-English stats guide (Phase 2)
- `docs/results/` — per-phase result markdown files (Phases 2–4)
- `docs/RESULTS_INDEX.md` — claims → result files (Phase 5)
- `scaling/` — multi-agent toy-problem variants (Phase 4.2)
- `onpolicy/scripts/sweeps/toyproblem/` — WandB sweep infra (Phase 2)
- `scripts/reproduce/` — one-line entry points (Phase 5)
- `REPRODUCE.md` — cold-reader reproducibility guide (Phase 5)

---

## Key Decisions & Deviations from Papers

| Decision | Detail | Status |
|----------|--------|--------|
| Bit-cost formula | Use `log₂(\|z\|/δ+1)` not `log₂(2\|z\|/δ+1)` — constant `2` dropped (same argmin) | ✅ Fixed `channels.py:61,94`; unit tested |
| Pillar implementation order | P2 → P1 → P4 → P3 (highest-impact first) | Decided |
| NSD implementation | Existing `DDCL_NSD` is faithful to proposal Algorithm 1 (TPDF dual-path STE). Verify via Thm 5 MC tests in Phase 1. | Pending verification |
| Hyperparameters | NOT from paper — re-derived from scratch in Phase 2 sweeps | Decided |
| Scope | Toyproblem standalone only. Pre-existing bugs out of scope. | Decided |
| P4 null hypothesis | May be statistically null at 2-agent scale. Report honestly; re-test at scale. | Decided |
| Dither comparison | Compare 4 types: SD, TPDF, additive-uniform, Gaussian | Decided |
| P1 delta modes | 4 modes: fixed_scalar, fixed_per_channel_scalar, learned_scalar, learned_per_channel | Decided |
| Pareto goal | Show Pareto frontier vs naive baselines — NOT exact paper number reproduction | Decided |

---

## Phase Status

| Phase | Status | Gate |
|-------|--------|------|
| 0 — Hygiene & principles | ✅ DONE | `pytest` 28 passed, 23 skipped, 0 failed |
| 1 — Verify & tighten DDCL | ✅ DONE | `pytest` 51 passed, 0 skipped, 0 failed; bit-cost formula fixed; `configs/baseline_v0.yaml` created |
| 2 — Hyperparameter sweeps | 🔶 RUNNING | Infrastructure complete; CODE-001–009 fixed; Stage A sweep active (~98/2175 done) |
| 3 — 4 Pillars (P2→P1→P4→P3) | 🔲 TODO | Each pillar: off=baseline bit-identical; sweep passes gate |
| 4 — Combined + scaling laws | 🔲 TODO | Unleashed on Pareto frontier; leave-one-out ablation passes |
| 5 — Docs & reproducibility | 🔲 TODO | Cold reader reproduces headline result from REPRODUCE.md |

---

## What Is Fully Done (Phase 0)

### Files Created

| File | What it Contains |
|------|-----------------|
| `docs/README.md` | Full reader guide: env mechanics, channel layer math, network architecture, every CLI flag, log dir structure, how to run |
| `docs/MATH.md` | All symbols mapped to file:line; SD proof (Thm 2/A.1); NSD proof (Thm 5); bit-cost deviation justification; pillar stubs P1–P4 |
| `docs/KNOWN_ISSUES.md` | 6 pre-existing out-of-scope issues: PP step-limit, active_masks flag, dual DDCL code paths, fake_quantization gap, missing pyarrow, global numpy RNG pollution |
| `tests/__init__.py` | Empty (package marker) |
| `tests/conftest.py` | Fixtures: `device` (cpu, session-scoped), `delta` (parametrized [1,5,10,15,20]), `default_delta` (10.0), `identity_channel`, `sd_channel`, `nsd_channel`, `any_channel` (parametrized), `single_env`, `vec_env_small` (N=4), `seed_torch` (autouse, resets to 0) |
| `tests/test_channels.py` | 23 tests. Passing: IdentityChannel (3), SD shape/nonneg/zero-at-origin/info-keys (4), NSD shape/nonneg (2), bits-aggregation (1). Skipped: MC/gradient stubs (13) |
| `tests/test_env_determinism.py` | 20 tests. Passing: goal distribution (5), single-env correctness (8), vec-env shapes (2). Skipped: determinism, agreement, auto-reset, final_obs, independent reset (5) |
| `tests/test_trainer.py` | 8 tests. Passing: act_and_value shapes/range/no-grad (3). Skipped: fresh noise, comms loss wiring (5) |
| `PLAN.md` | Full approved 5-phase project plan |
| `CONTEXT.md` | This file |

### Files Modified

| File | What Changed |
|------|-------------|
| `train.py` | Added `set_seed()` (Python+NumPy+PyTorch+CUDA+PYTHONHASHSEED); added `--exp_name`, `--hidden_size` CLI args; structured log dir `runs/<exp_name>/<seed>/`; writes `config.json` + `git_sha.txt` per run |
| `trainer.py` | Added `hidden_size: int = 64` to `MAPPOConfig`; wired `hidden_size` to all 3 networks (SpeakerNetwork, ListenerActor, Critic) |

### What Has NOT Changed (original code, unmodified)

- `CommunicatingGoal_env.py` — original (audited Phase 1: all assertions pass)
- `CommunicatingGoal_vec_env.py` — original (audited Phase 1: all assertions pass)
- `network.py` — original; hardcoded `hidden=16` defaults overridden by trainer wiring

### Phase 1 — Newly Completed

| File | What Changed |
|------|-------------|
| `channels.py` | Bit-cost formula fixed: `log2(2*z.abs()/δ+1)` → `log2(z.abs()/δ+1)` (lines 61, 94) |
| `buffer.py` | Added `goal_ids` tensor (n_steps, n_envs); `insert()` accepts optional `goal_id`; `minibatches()` yields `goal_ids` |
| `trainer.py` | Per-goal bit logging in `update()`: `bits_goal_0`…`bits_goal_5` in returned metrics |
| `train.py` | Added `_GOAL_POS_TO_IDX` lookup; rollout computes `goal_ids_np` and passes to buffer; `CSV_HEADER` extended with per-goal columns; `_N_GOALS` constant |
| `tests/test_channels.py` | All 23 tests passing (was 10 pass, 13 skipped); MC tests added for SD and NSD; formula ordering test added |
| `tests/test_env_determinism.py` | All 20 tests passing (was 15 pass, 5 skipped); determinism, vec/single agreement, auto-reset, final_obs, independent reset |
| `tests/test_trainer.py` | All 8 tests passing (was 3 pass, 5 skipped); fresh noise, lambda gating, z vs z_hat, critic no comms path |
| `docs/MATH.md` | NSD variance corrected: δ²/6 → δ²/4 (MATH-002); Phase 1 verification status updated in table |
| `configs/baseline_v0.yaml` | New: correct (not yet optimal) starting config with Phase 1 verification notes |

---

## Immediate Next Tasks (Phase 2 — sweep running)

**Stage A sweep is active. See `docs/README.md §5` for complete step-by-step instructions.**

1. **Wait for Stage A sweep to complete** (~22 hrs remaining on CPU from 2026-04-24).
   - Check progress: `ps aux | grep run_sweep` — if not running, restart with `--resume`.
   - Restart command:
     ```bash
     cd on-policy && nohup bash -c 'KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
         python -m onpolicy.scripts.sweeps.toyproblem.run_sweep \
         --config onpolicy/scripts/sweeps/toyproblem/configs/sweep_stage_a.yaml \
         --seeds 0 1 2 3 4 \
         --log_dir onpolicy/envs/toyproblem/runs/sweep_stage_a \
         --resume --shuffle' > /tmp/sweep_stage_a.log 2>&1 &
     ```
2. **Generate Stage A report** (run any time; re-run after sweep completes):
   ```bash
   cd on-policy && KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
       python -m onpolicy.envs.toyproblem.analysis.report_baseline \
       --sweep_dir onpolicy/envs/toyproblem/runs/sweep_stage_a \
       --out_dir onpolicy/envs/toyproblem/docs/results/sweep_stage_a
   ```
3. **Generate all 12 paper figures** from Stage A data:
   ```bash
   cd on-policy && KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms python -c "
   from onpolicy.envs.toyproblem.analysis.load_runs import load_sweep, final_metrics, seed_aggregate
   from onpolicy.envs.toyproblem.analysis.paper_figures import generate_all_paper_figures
   df = load_sweep('onpolicy/envs/toyproblem/runs/sweep_stage_a')
   summary = final_metrics(df)
   agg = seed_aggregate(summary, group_cols=['channel','lambda_comms','delta','z_dim'])
   generate_all_paper_figures(df, summary, agg,
       out_dir='onpolicy/envs/toyproblem/docs/results/sweep_stage_a/paper_figures')
   "
   ```
4. **Check convergence gate** (all 4 criteria must pass before proceeding):
   - Gate is embedded in `report_baseline.py` output; check for `Convergence gate: DONE`.
5. **Run Stage B sweep** (architecture/PPO params) after Stage A winner identified.
6. **Freeze `configs/baseline_best.yaml`** after all convergence gates pass.

---

## Environment / Domain Specifics (quick reference)

- Grid: 8×8, max 50 steps per episode
- Rewards: step penalty −0.01, success +1.0
- Goal positions: `[[0,0],[7,7],[3,4],[4,3],[1,6],[6,1]]`
- Goal probs: `[0.515, 0.258, 0.129, 0.064, 0.031, 0.003]` (Zipf-like, strictly decreasing)
- Speaker obs = goal `(x,y)` ints; listener obs = listener position `(x,y)` ints (no normalisation)
- Listener action space: 5 discrete (0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT)
- Speaker is stationary; listener is mobile; both share reward (cooperative)

---

## Session Log

### Session 1 (2026-04-22)
- Read both DDCL papers; designed full 5-phase plan (approved by user in plan mode)
- Completed all of Phase 0:
  - Created docs/ (README.md, MATH.md, KNOWN_ISSUES.md)
  - Created tests/ skeleton (conftest.py, test_channels.py, test_env_determinism.py, test_trainer.py)
  - Hardened train.py: set_seed(), --exp_name, --hidden_size, structured log dir
  - Updated trainer.py: hidden_size added to MAPPOConfig, wired to all 3 networks
  - Installed required packages in marl_comms env: pytest, absl-py, gym
- Published plan as PLAN.md; created CONTEXT.md and ISSUES_TRACKER.md
- Issues encountered: 5 installation/runtime issues — all documented in ISSUES_TRACKER.md
- **Next:** Phase 1

### Session 2 (2026-04-22)
- Completed all of Phase 1 (51 tests passing, 0 skipped, 0 failed):
  - **channels.py**: fixed bit-cost formula (removed factor-of-2)
  - **test_channels.py**: filled all 13 MC stubs (SD: unbiasedness, variance δ²/12, grad identity, error independence, monotone loss, fresh noise; NSD: unbiasedness, variance δ²/4, zero covariance, grad identity, deploy no-grad, fresh noise; formula ordering)
  - **test_env_determinism.py**: filled all 5 env stubs (same-seed determinism, vec/single agreement, auto-reset, final_obs in info, independent slot reset)
  - **test_trainer.py**: filled all 5 trainer stubs (fresh noise, lambda gating, comms loss on z not z_hat, critic no comms path)
  - **buffer.py**: added goal_ids storage and per-goal tracking through minibatch pipeline
  - **trainer.py**: per-goal bit logging (bits_goal_0…bits_goal_5) in metrics
  - **train.py**: goal_id computation and per-goal CSV columns
  - **configs/baseline_v0.yaml**: created with Phase 1 verification notes
  - **docs/MATH.md**: corrected NSD variance formula δ²/6 → δ²/4 (MATH-002)
- **Key finding (MATH-002)**: NSD reconstruction variance is δ²/4 (not δ²/6 as proposal stated). Analytical derivation + MC verification (2M samples, all z values). The 2nd-order Schuchman property (constant variance) holds but the constant is δ²/4. NSD has 3× more noise variance than SD (δ²/4 vs δ²/12).
- **Next:** Phase 2 — install pyarrow, build WandB sweep infrastructure, run Stage A sweep

### Session 6 (2026-04-24)
- **CODE-009 (critical) discovered and fixed:** `_run_key()` in `run_sweep.py` truncated exp_name to 80 chars; `lambda_comms` and `z_dim` appear at chars 88+ and were excluded. All 24 combos per (channel, delta, seed) mapped to the same directory. Sweep falsely declared "all done" after 95/2175 runs.
  - Fix: `_run_key()` now excludes all fixed hyperparams (clip_eps, gae_lambda, hidden_size, lr, etc.) — only the 4 varying Stage A params (channel, delta, lambda_comms, z_dim) appear in the key. Keys are ~50 chars, fully unique, no truncation.
  - Recovery: loaded `args` from each `final.pt` checkpoint; renamed 95 directories from old 80-char prefix to new short format. Zero data lost.
  - Committed as `d34468e`.
- **Sweep restarted** (nohup-detached, correct naming confirmed from process list).
- **All Phase 2 infrastructure committed:** commits `a5a45c8` (analysis, sweeps, tests, channels, train/trainer) and `76eb272` (experiments) on `feat/ToyProblem`. Branch is now clean.
- **Channel comparison report and paper figures** (re)generated to `docs/results/channel_comparison/`.
- **Partial Stage A report and 12 paper figures** generated from 98 completed runs to `docs/results/sweep_stage_a/`.
- **Docs updated:** CONTEXT.md (this entry), ISSUES_TRACKER.md (CODE-009, checklist), docs/README.md (experiment instructions, documentation map).
- **Next:** Stage A sweep completes (~22 hrs); re-run report and paper figures on full data; check convergence gate.

### Session 5 (2026-04-24)
- **Jensen gap + RL uncertainty model for per-goal bit allocation:**
  - Added `goal_probs`, `total_updates`, `show_uncertainty` parameters to `plot_per_goal_bits` in `analysis/plots.py`
  - Two-source uncertainty model: (1) Jensen gap ≈ ±0.20 bits constant band, (2) RL sampling noise ∝ 1/sqrt(p_i × N_updates). Combined in quadrature.
  - Goal 5 (p=0.003) gets ≈13× wider band than goal 0 (p=0.515) — guides reader not to over-interpret rare-goal deviations.
  - Effective sample count n_i ≈ p_i × N_updates annotated on x-axis of every per-goal plot.
- **Pre-sweep comprehensive audit — all issues found and fixed:**
  - CODE-007: `additive_uniform` missing from Stage A sweep; added to `sweep_stage_a.yaml`
  - CODE-008: `none` channel running 8×6 redundant λ/δ combos; auto-deduplication added to `run_sweep.py`
  - Stage A sweep command updated: uses `--log_dir onpolicy/envs/toyproblem/runs/sweep_stage_a` (toyproblem-local)
- **Reorganised results path:** created `runs/` directory inside toyproblem; all future run outputs go there
- **Research paper figures:** created `analysis/paper_figures.py` with 10 figures, each with Hypothesis/Analysis/Conclusion docstring:
  - Fig 1: Rate-distortion frontier (core result; H(G) reference; Pareto frontier)
  - Fig 2: Training curves by channel
  - Fig 3: Per-goal bit allocation with Jensen+RL uncertainty envelope
  - Fig 4: λ sensitivity (smooth rate-distortion knee)
  - Fig 5: Channel efficiency relative to H(G)
  - Fig 6: SD vs NSD head-to-head (paired scatter)
  - Fig 7: Surrogate vs true bits calibration
  - Fig 8: z_dim scaling (SR and bits vs z_dim)
  - Fig 9: Per-goal bit dynamics over training
  - Fig 10: δ×λ heatmap (interior optimum check)
  - `generate_all_paper_figures()` batch runner included
- **Documentation updated:** PLAN.md (§2.6 paper figures), CONTEXT.md (this entry + layout), ISSUES_TRACKER.md (CODE-007, CODE-008, checklist)
- **Next:** run Stage A sweep — all infrastructure confirmed correct

### Session 3 (2026-04-22)
- Completed Phase 2 infrastructure + systematic experiment series design:
  - **channels.py**: Added 3 baseline channels: `AdditiveUniformChannel` (1st-order Schuchman), `GaussianChannel` (negative control, σ=δ/√12), `STEChannel` (fixed-rate, configurable bits). Updated `build_channel()` to support `additive_uniform`, `gaussian`, `ste4`, `ste8`, `ste16`. Added `_CHANNEL_NAMES` list.
  - **trainer.py**: Added `ste_clip: float = 10.0` to `MAPPOConfig`; wired through to `build_channel()`.
  - **train.py**: Added `--channel` choices for 5 new types; added `--ste_clip` arg; wired into `MAPPOConfig`.
  - **analysis/__init__.py**: Package marker with module docstring.
  - **analysis/load_runs.py**: `load_run()`, `load_exp()`, `load_sweep()`, `final_metrics()`, `seed_aggregate()`.
  - **analysis/stats.py**: `bootstrap_ci()`, `iqm()`, `iqm_ci()`, `paired_permutation_test()`, `wilcoxon_signed_rank()`, `is_pareto_dominated()`, `pareto_frontier()`, `compare_configs()`.
  - **analysis/plots.py**: `plot_training_curves()`, `plot_rate_distortion()`, `plot_per_goal_bits()`, `plot_sweep_heatmap()`, `plot_channel_comparison()`.
  - **analysis/sweep_convergence.py**: `check_convergence()` implementing all 4 Phase 2 §2.3 criteria; `ConvergenceResult` dataclass with pretty-print.
  - **analysis/report_baseline.py**: Full pipeline — load sweeps → run convergence gate → generate plots → write `docs/results/baseline.md`.
  - **docs/STATS.md**: Plain-English guide covering all 8 statistical methods (bootstrap CI, IQM, paired permutation, Wilcoxon, Pareto frontier, convergence gate, gradient-variance diagnostic, reporting standards).
  - **docs/results/**: Directory created.
  - **experiments/__init__.py**: Package marker.
  - **experiments/validate_environment.py**: 5 statistical validation checks (goal distribution χ², episode length, random policy SR, reward structure, listener start correctness). All pass with 2000 episodes.
  - **experiments/validate_mappo.py**: 6 MAPPO correctness checks (value loss decrease, entropy control, lambda gating, identity channel bits=0, advantage normalisation). All pass at 80 updates.
  - **experiments/run_channel_comparison.py**: Orchestrator for 8-channel × N-seed comparison. Dry-run verified; all 16 commands correct.
  - **tests/conftest.py**: Added fixtures for `additive_uniform_channel`, `gaussian_channel`, `ste_channel` (3-way parametrized). Updated `any_channel` to 8-way parametrized.
  - **tests/test_channels.py**: +28 new tests (AdditiveUniform: 5, Gaussian: 5, STE: 5, BuildChannel: 3 across all 8 channels). Total: 79 passing, 0 failing.
  - **onpolicy/scripts/sweeps/toyproblem/**: `configs/sweep_stage_a.yaml`, `configs/sweep_stage_b.yaml`, `sweep_wrapper.py`, `run_sweep.py`.
- **Gate**: `pytest` 79 passed, 0 failed; env validation 5/5 pass; MAPPO validation 6/6 pass.
- **Next:** Run sweeps (requires compute). See "Immediate Next Tasks" above.
