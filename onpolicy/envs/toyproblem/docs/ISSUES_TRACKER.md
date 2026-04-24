# DDCL Unleashed — Issues Tracker

> **Mandatory update rule:** Any time an issue is encountered — installation failure, test failure, unexpected behaviour, documentation gap, code bug, environment problem — add it here before fixing it. Record: what the issue is, when it appeared, what caused it, what the fix is, and current status. This file is the authoritative log for reproducibility.
>
> Format for new entries:
> ```
> ### [CATEGORY-NNN] Short title
> - **Phase discovered:** Phase N
> - **Date:** YYYY-MM-DD
> - **Symptom:** What went wrong / what was observed
> - **Root cause:** Why it happened
> - **Fix:** Exact command or code change that resolves it
> - **Status:** RESOLVED / PENDING / DEFERRED / WONT-FIX
> - **Reproducibility impact:** Does this affect someone trying to reproduce results from scratch?
> ```
>
> Categories: `INSTALL` (environment/packages), `CODE` (bugs in implementation), `TEST` (test failures or gaps), `DOCS` (documentation gaps), `MATH` (formula or derivation issues), `REPRO` (reproducibility blockers)

---

## Pre-existing / Out-of-scope Issues

These are upstream bugs in the shared `on-policy` repository. They were found during Phase 0/1 audits. They are **not** caused by this project and are **not** in scope for the DDCL toyproblem work. They are catalogued here so future work on other environments does not trip over them.

### [UPSTREAM-001] PredatorPrey: No step-count enforcement
- **File:** `onpolicy/envs/predator_prey/PredatorPrey_env.py`
- **Symptom:** Environment only sets `episode_over = True` on prey capture; it does not enforce `max_steps`. Episodes run indefinitely across rollout boundaries until capture, making win-rate logging at rollout boundaries incorrect.
- **Impact:** Affects PP-Medium and PP-Hard experiments only. Toyproblem is not affected.
- **Status:** WONT-FIX (out of scope) — fix when PP experiments are scheduled: add `self.step_count` tracker; set `episode_over = True` when `step_count >= max_steps`; set `bad_masks` at truncation boundaries in the runner.

---

### [UPSTREAM-002] `use_active_masks_in_transformer` flag inconsistency
- **Files:** `onpolicy/algorithms/utils/transformer_encoder.py`, `onpolicy/scripts/train_pp_scripts/`, `onpolicy/scripts/train_football_scripts/`
- **Symptom:** Flag implemented in transformer but removed from PP/TJ scripts; Football scripts still reference it. Using it produces inconsistent masking across environments.
- **Impact:** Multi-agent transformer experiments only. Toyproblem does not use the transformer stack.
- **Status:** WONT-FIX (out of scope) — fix when MAT experiments are scheduled: either remove flag entirely or consistently enable across all environments.

---

### [UPSTREAM-003] Two separate DDCL code paths never cross-validated
- **Files:** `onpolicy/algorithms/utils/transformer_encoder.py` vs `onpolicy/envs/toyproblem/channels.py`
- **Symptom:** Transformer DDCL wires into attention keys/values with `ddcl_variation ∈ {"old","new"}`; toyproblem uses standalone `DDCL_SD`/`DDCL_NSD` classes. Gradient mechanics, loss formulas, and noise-scaling conventions have not been verified to agree.
- **Impact:** PP/TJ/GRF results and toyproblem results may not be comparable if underlying DDCL maths differ.
- **Status:** DEFERRED — fix when cross-environment comparison is needed: unify on a single `channels.py` module or write a cross-implementation equivalence test.

---

### [UPSTREAM-004] `fake_quantization` only wired in transformer, not toyproblem
- **Files:** `onpolicy/algorithms/utils/transformer_encoder.py`, `onpolicy/config.py:350-352`
- **Symptom:** `--use_fake_quantization` and `--quant_bits` defined in global config but only consumed by the transformer encoder. STE-vs-DDCL comparisons cannot be run on the toyproblem without porting this feature.
- **Impact:** Phase 2 sweeps include STE as a baseline; this flag would need to be ported to `channels.py` (making it in-scope for Phase 2 if needed, not a pre-existing issue at that point).
- **Status:** DEFERRED — monitor during Phase 2 STE analysis; port to `channels.py` if STE sweep results are needed.

---

## Installation Issues

### [INSTALL-001] `onpolicy` package not on Python path
- **Phase discovered:** Phase 0
- **Date:** 2026-04-22
- **Symptom:** `ModuleNotFoundError: No module named 'onpolicy'` when running pytest from the toyproblem directory.
- **Root cause:** The `onpolicy` package lives in `on-policy/` but is not installed in editable mode by default. Python cannot find it without an explicit install.
- **Fix:** From the `on-policy/` directory, run:
  ```bash
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms pip install -e .
  ```
- **Status:** RESOLVED
- **Reproducibility impact:** YES — anyone reproducing from scratch must run this before any `import onpolicy` or pytest invocation.

---

### [INSTALL-002] `absl-py` not in `marl_comms` environment
- **Phase discovered:** Phase 0
- **Date:** 2026-04-22
- **Symptom:** `ModuleNotFoundError: No module named 'absl'` when importing onpolicy. Root is `onpolicy/__init__.py` eagerly importing all subpackages, including ones that depend on absl.
- **Root cause:** `absl-py` was not installed in the `marl_comms` conda environment.
- **Fix:**
  ```bash
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms pip install absl-py
  ```
- **Status:** RESOLVED
- **Reproducibility impact:** YES — required before any onpolicy import chain.

---

### [INSTALL-003] `gym` not in `marl_comms` environment
- **Phase discovered:** Phase 0
- **Date:** 2026-04-22
- **Symptom:** `ModuleNotFoundError: No module named 'gym'` when importing `CommunicatingGoal_env.py`.
- **Root cause:** `gym` was not installed in `marl_comms`.
- **Fix:**
  ```bash
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms pip install gym
  ```
  Installed version: `gym==0.26.2`. NumPy 2.x deprecation warning appears but is non-blocking.
- **Status:** RESOLVED
- **Reproducibility impact:** YES — required to import the environment.

---

### [INSTALL-004] `pytest` not in `marl_comms` environment
- **Phase discovered:** Phase 0
- **Date:** 2026-04-22
- **Symptom:** `pytest: command not found` when running tests inside the conda env.
- **Root cause:** pytest was not installed in `marl_comms`.
- **Fix:**
  ```bash
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms pip install pytest
  ```
- **Status:** RESOLVED
- **Reproducibility impact:** YES — required to run the test suite.

---

### [INSTALL-005] `OMP: Error #15` — OpenMP library conflict on macOS
- **Phase discovered:** Phase 0
- **Date:** 2026-04-22
- **Symptom:** `OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.` when running any conda command on macOS. Process may abort.
- **Root cause:** macOS has two OpenMP runtime libraries (from PyTorch and conda) that conflict on import.
- **Fix:** Prefix every `conda run` command with `KMP_DUPLICATE_LIB_OK=TRUE`:
  ```bash
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms <your command>
  ```
  This suppresses the conflict and allows the process to continue. It is a workaround, not a permanent fix.
- **Status:** RESOLVED (workaround)
- **Reproducibility impact:** YES — every shell command in this project must use this prefix on macOS.

---

### [INSTALL-006] `pyarrow` not in `marl_comms` environment
- **Phase discovered:** Phase 0
- **Date:** 2026-04-22
- **Symptom:** `import pyarrow` fails. Parquet output (`metrics.parquet`) cannot be written. All offline metrics fall back to CSV only.
- **Root cause:** `pyarrow` was not installed in `marl_comms`. It is needed for efficient columnar storage of sweep metrics.
- **Fix (pending):**
  ```bash
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms pip install pyarrow
  ```
  After installing, `train.py` should additionally write `metrics.parquet` alongside `metrics.csv`. Code change needed in `train.py` to enable parquet output.
- **Status:** PENDING — install not yet done; CSV fallback in use
- **Reproducibility impact:** MODERATE — CSV is a working fallback, but parquet is faster for sweep analysis scripts. Must be resolved before Phase 2 sweep infrastructure.

---

## Code Issues

### [CODE-001] Bit-cost formula uses factor-of-2 deviation from our chosen convention
- **Phase discovered:** Phase 0 (code audit)
- **Date:** 2026-04-22
- **Symptom:** `channels.py` lines 61 and 94 compute `log2(2 * z.abs() / self.delta + 1)`. Our project convention (documented in `docs/MATH.md`) is `log2(z.abs() / self.delta + 1)` — the factor `2` is dropped because it does not affect the argmin.
- **Root cause:** Original implementation followed the proposal's formula verbatim; we deliberately deviate.
- **Fix:** Applied Phase 1 — `channels.py:61` and `:94` updated. Unit test `TestCommsLossFormula.test_formula_vs_original_proposal` verifies same rank ordering.
- **Status:** ✅ RESOLVED (Phase 1, 2026-04-22)
- **Reproducibility impact:** YES — quantitative bit-cost numbers will be up to ~1 bit lower than the proposal. Intentional and documented in `docs/MATH.md §2.3`.

---

### [CODE-002] `network.py` hardcodes `hidden=16` defaults (bypassed by trainer wiring)
- **Phase discovered:** Phase 0 (code audit)
- **Date:** 2026-04-22
- **Symptom:** `SpeakerNetwork`, `ListenerActor` default `hidden=16`; `Critic` defaults `hidden=32`. These are tiny networks and were the original defaults.
- **Root cause:** Original code; not a bug per se — the trainer now passes `config.hidden_size` explicitly.
- **Fix:** `trainer.py` now correctly passes `hidden=config.hidden_size` to all three networks. The hardcoded defaults in `network.py` only apply if someone constructs networks directly without going through the trainer.
- **Status:** RESOLVED (via trainer wiring) — defaults in `network.py` still low but harmless
- **Reproducibility impact:** LOW — only matters if someone constructs networks outside the trainer.

---

### [CODE-003] `CommunicatingGoal_env.seed()` pollutes global NumPy RNG
- **Phase discovered:** Phase 0 (code audit)
- **Date:** 2026-04-22
- **Symptom:** `env.seed(seed)` calls both `self.np_random = np.random.RandomState(seed)` AND `np.random.seed(seed)` (the global RNG). The global seed is set as a side effect, which is surprising in a multi-env context.
- **Root cause:** Original implementation. The vec env's `env.seed()` does not have this issue (sets only instance RNG).
- **Fix:** `train.py`'s `set_seed()` sets `np.random.seed` before env construction, so the order is consistent in practice. The real fix (remove global side effect from `CommunicatingGoal_env.seed()`) is out of scope.
- **Status:** DEFERRED — low-priority; seeding order in `train.py` is correct in practice. Real fix: remove `np.random.seed(seed)` call from `CommunicatingGoal_env.seed()`.
- **Reproducibility impact:** LOW in practice (seeding order in `train.py` is correct), but could cause surprises in multi-process contexts.

---

## Test Issues

### [TEST-001] 23 tests skipped in `test_channels.py` (Phase 1 stubs)
- **Phase discovered:** Phase 0 (intentional stubs)
- **Date:** 2026-04-22
- **Fix:** All 13 skipped MC/gradient tests implemented in Phase 1. All 23 tests now pass.
- **Status:** ✅ RESOLVED (Phase 1, 2026-04-22)
- **Reproducibility impact:** RESOLVED — channel maths fully verified.

---

### [TEST-002] 5 tests skipped in `test_env_determinism.py` (Phase 1 stubs)
- **Phase discovered:** Phase 0 (intentional stubs)
- **Date:** 2026-04-22
- **Fix:** All 5 skipped tests implemented in Phase 1. All 20 tests now pass.
- **Status:** ✅ RESOLVED (Phase 1, 2026-04-22)
- **Reproducibility impact:** RESOLVED — determinism verified.

---

### [TEST-003] 5 tests skipped in `test_trainer.py` (Phase 1 stubs)
- **Phase discovered:** Phase 0 (intentional stubs)
- **Date:** 2026-04-22
- **Fix:** All 5 skipped tests implemented in Phase 1. All 8 tests now pass.
- **Status:** ✅ RESOLVED (Phase 1, 2026-04-22)
- **Reproducibility impact:** RESOLVED — loss wiring verified.

---

## Documentation Gaps

### [DOCS-001] `docs/STATS.md` not yet created
- **Phase discovered:** Phase 0 (planned)
- **Date:** 2026-04-22
- **Symptom:** File referenced in PLAN.md and CONTEXT.md but does not exist.
- **Root cause:** Intentional deferral — statistical methods are defined for Phase 2.
- **Fix:** Create in Phase 2, covering: bootstrap CI, paired permutation test, Wilcoxon, IQM + stratified bootstrap, Pareto-frontier test, gradient-variance diagnostic — each explained in plain English.
- **Status:** DEFERRED — Phase 2
- **Reproducibility impact:** YES — readers need this to understand any reported p-values or CIs.

---

### [DOCS-002] `configs/` directory does not exist
- **Phase discovered:** Phase 0 (planned)
- **Date:** 2026-04-22
- **Fix:** `configs/baseline_v0.yaml` created at end of Phase 1.
- **Status:** ✅ RESOLVED (Phase 1, 2026-04-22)

---

### [DOCS-003] Per-goal bit-allocation logging not implemented
- **Phase discovered:** Phase 0 (planned)
- **Date:** 2026-04-22
- **Fix:** `buffer.py` now stores `goal_ids`; `trainer.py` logs `bits_goal_0`…`bits_goal_5` per update; `train.py` computes `goal_ids_np` from position lookup and writes per-goal columns to CSV.
- **Status:** ✅ RESOLVED (Phase 1, 2026-04-22)
- **Reproducibility impact:** RESOLVED.

---

## Math Issues

### [MATH-002] NSD reconstruction variance formula incorrect in proposal and MATH.md
- **Phase discovered:** Phase 1 (MC verification of test_channels.py)
- **Date:** 2026-04-22
- **Symptom:** `docs/MATH.md` and the proposal (Thm 5) state `Var(ẑ_deploy - z | z) = δ²/6` for NSD. Empirical MC test (N=2M samples) gives `Var ≈ δ²/4 = 25.0` for δ=10, for all tested z values.
- **Root cause:** The NSD error `e = C(m) - z` takes full-bin jumps (e.g., −δ, 0, +δ), NOT the dither noise ν itself. The dither variance is δ²/6, but the reconstruction error variance is larger: analytically `E[(e/δ)²] = 1/4` for all z, giving `Var(e) = δ²/4`.  
  Verified for z ∈ {0, 1, 2.5, 5, 7.5, 9} — all give Var ≈ 25.0 at δ=10. The 2nd-order Schuchman property (constant variance, independent of z) holds; only the constant value was wrong.
- **Fix:** Updated `docs/MATH.md §3.2` to correct the formula to `δ²/4` with derivation. Updated `tests/test_channels.py::TestDDCL_NSD::test_reconstruction_variance` to expect `δ²/4`.
- **Status:** ✅ RESOLVED (Phase 1, 2026-04-22)
- **Reproducibility impact:** YES — the correct NSD noise level is 3× larger than SD (δ²/4 vs δ²/12). This affects relative performance comparison between SD and NSD channels; reported bit-cost comparisons in future results should note this.

---

### [MATH-001] TPDF bit-cost formula under NSD — derivation pending
- **Phase discovered:** Phase 0 (planned)
- **Date:** 2026-04-22
- **Symptom:** Under subtractive dither, `E[|m| | z]` reduces cleanly to give bit cost `log₂(|z|/δ+1)`. Under TPDF, `m = ⌊(z+ε₁+ε₂)/δ⌋` with triangular noise — it is not obvious whether the same formula applies or needs a correction.
- **Root cause:** Derivation not yet done.
- **Fix (Phase 3.4):** Derive `E[|m| | z]` under TPDF analytically; compare to SD case; add Monte-Carlo numerical test; document in `docs/MATH.md`; update `channels.py` comms_loss for NSD if a correction is needed.
- **Status:** DEFERRED — Phase 3.4
- **Reproducibility impact:** YES — if the NSD bit-cost formula is wrong, NSD bit-cost numbers are not comparable to SD numbers.

---

### [CODE-004] `true_bits_per_msg` not written to metrics.csv — `none` channel bits always 0
- **Phase discovered:** Phase 2 (sweep analysis, 2026-04-23)
- **Date:** 2026-04-23
- **Symptom:** `train.py` `CSV_HEADER` and the CSV row writer did not include `true_bits_per_msg`, even though `trainer.py` computed it. Separately, `load_runs.final_metrics` did not include `true_bits_per_msg` in `key_cols`. As a result: (1) `true_bits_per_msg` was never persisted; (2) the `none` channel showed `bits_per_msg = 0` in every row (since `comms_loss` correctly returns 0 for training, but this makes the rate-distortion plot put all `none` points at x=0 bits — factually wrong).
- **Root cause:** `true_bits_per_msg` was added to the trainer's `update()` metrics dict but never wired into `train.py`'s CSV output. The distinction between "training surrogate" (`bits_per_msg`) and "true transmission cost" (`true_bits_per_msg`) was not carried through to the logger.
- **Fix:** (2026-04-23)
  - `train.py`: Added `"true_bits_per_msg"` to `CSV_HEADER` and the writer row; updated console output to show both `bits_surr` and `bits_true`.
  - `analysis/load_runs.py`: Added `"true_bits_per_msg"` to `final_metrics` `key_cols`.
  - `analysis/report_baseline.py`: Added separate training-curve plot for `true_bits_per_msg`; added second rate-distortion plot using `true_bits_per_msg` (deployment perspective); legacy `rate_distortion.png` now copies from `rate_distortion_surrogate.png`.
  - Deleted the 90 corrupted Stage A runs (`runs/sweep_stage_a/`) so they re-run with correct logging.
- **Status:** ✅ RESOLVED (2026-04-23)
- **Reproducibility impact:** YES — all Stage A runs prior to this fix have `bits_per_msg = 0` for `none` channel and are missing `true_bits_per_msg`. Those 90 runs were deleted and must be re-run. New runs log both columns correctly.

---

### [CODE-005] Surrogate `comms_loss` underestimates at z=0, overestimates at bin boundaries
- **Phase discovered:** Phase 2 (bit-calculation validation, 2026-04-23)
- **Date:** 2026-04-23
- **Symptom:** MC validation (N=100K) showed: at z=0, surrogate=0 but empirical E[log₂(|m|+1)] ≈ 0.5 bits (dither always places m in {-1,0} with equal prob); at z=n·δ (bin boundaries), surrogate overestimates by up to 2× because m splits evenly between two bins.
- **Root cause:** The surrogate assumes E[|m||z] = |z|/δ, which equals 0 at z=0 but the true value is 0.5. At bin boundaries, m is equally likely to be in bin n-1 or n, giving E[|m|] ≈ n-0.5 < n = |z|/δ.
- **Fix:** Documented in `channels.py` (`DDCL_SD.transmission_bits_per_elem` docstring) and `docs/MATH.md §10.2` with the full MC validation table. No code change needed — the surrogate is intentionally a Jensen upper bound used for training; `true_bits_per_msg` captures the empirical cost.
- **Status:** ✅ RESOLVED (documented; no code fix required)
- **Reproducibility impact:** LOW — affects interpretation of `bits_per_msg` values near z=0 or bin boundaries; `true_bits_per_msg` is unaffected and correct.

---

### [CODE-006] GaussianChannel docstring incorrectly claimed empirical bits > surrogate
- **Phase discovered:** Phase 2 (bit-calculation validation, 2026-04-23)
- **Date:** 2026-04-23
- **Symptom:** The `GaussianChannel` class docstring said "empirical > surrogate (Gaussian tails)." MC validation showed the opposite: surrogate ≥ empirical at all signal levels because σ=δ/√12 ≈ 0.289δ is narrower than the uniform dither's half-width δ/2 = 0.5δ, so Gaussian concentrates m around the true bin more tightly.
- **Root cause:** Incorrect intuition in docstring; Gaussian σ is matched to SD variance (δ²/12), not SD half-width (δ/2).
- **Fix:** Updated `channels.py` `GaussianChannel` class docstring to correctly state "surrogate overestimates for Gaussian at all signal levels."
- **Status:** ✅ RESOLVED (2026-04-23)
- **Reproducibility impact:** NONE — docstring only; no functional change.

---

### [CODE-007] Stage A sweep included only {none, sd, nsd} — no additive_uniform baseline
- **Phase discovered:** Phase 2 (pre-sweep verification, 2026-04-24)
- **Date:** 2026-04-24
- **Symptom:** The original `sweep_stage_a.yaml` swept only `channel ∈ {none, sd, nsd}`. For a fair rate-distortion comparison, `additive_uniform` (1st-order Schuchman control) must sweep the same λ/δ/z_dim grid. Without this, its Pareto frontier cannot be plotted alongside DDCL, making the paper figures incomplete.
- **Root cause:** `additive_uniform` was added in Session 3 as a "baseline channel" and run in the channel comparison experiment, but not added to Stage A.
- **Fix:** Added `additive_uniform` to `channel.values` in `sweep_stage_a.yaml` (2026-04-24). The sweep grid grows by 8×6×3×5=720 runs (additive_uniform). Total unique DDCL/baseline runs: ~2175.
- **Status:** ✅ RESOLVED (2026-04-24)
- **Reproducibility impact:** YES — any Stage A runs completed before this fix do not include additive_uniform. Must restart from 0 (no runs existed yet).

---

### [CODE-009] `run_sweep.py` `_run_key` truncation causes exp_name collisions — 2080 runs silently skipped
- **Phase discovered:** Phase 2 (sweep execution, 2026-04-24)
- **Date:** 2026-04-24
- **Symptom:** After launching the Stage A sweep with `--resume --shuffle`, the runner printed "All 2880 runs complete" after only 95 real runs. The other ~2080 runs were silently skipped by `--resume`.
- **Root cause:** `_run_key()` sorted all hyperparameter names alphabetically then truncated to 120 chars, from which `exp_name = key[:80]` was derived. For `channel=additive_uniform`, the string `seed0_channel=additive_uniform_clip_eps=0.2_delta=0.5_entropy_coef=0.03_gae_lamb` fills exactly 80 chars — the parameters `lambda_comms` and `z_dim` appear at positions 88+ and are fully excluded from the exp_name. For `sd` and `nsd`, the same thing happens. This meant all 8 lambda_comms × 3 z_dim = 24 combinations per (channel, delta, seed) mapped to the same directory. Once one run wrote `metrics.csv`, `--resume` treated all 23 others as done.
- **Fix:** (2026-04-24) Rewrote `_run_key()` to exclude all fixed hypers (`clip_eps`, `entropy_coef`, `gae_lambda`, `hidden_size`, `lr`, `num_minibatches`, `ste_clip`) from the key entirely — only the 4 varying Stage A params (`channel`, `delta`, `lambda_comms`, `z_dim`) appear. Keys are now ~50 chars and fully unique. No truncation needed.
  - The 95 existing runs were recovered: loaded `args` from each `final.pt` checkpoint, renamed directories from old 80-char prefix to the new short format.
  - Committed as `d34468e` on `feat/ToyProblem`.
- **Status:** ✅ RESOLVED (2026-04-24)
- **Reproducibility impact:** YES — any Stage A results produced before this fix (before commit `d34468e`) will have only 1 run per (channel, delta, seed) instead of 24. Those directories must be renamed or re-run. The recovery script is documented in the Session 6 log in `CONTEXT.md`.

---

### [CODE-008] Stage A `none` channel redundancy — 45 of 48 runs per z_dim are no-ops
- **Phase discovered:** Phase 2 (pre-sweep verification, 2026-04-24)
- **Date:** 2026-04-24
- **Symptom:** The `IdentityChannel` ignores both `lambda_comms` (comms_loss always returns 0) and `delta`. Running the full 8×6 (λ×δ) grid for `none` wastes (8×6 − 1) × 3 z_dims × 5 seeds = 1050 identical runs.
- **Root cause:** The sweep config uses `method: grid` which generates all combinations unconditionally.
- **Fix:** (2026-04-24) Added deduplication logic to `run_sweep.py`: for `channel=none`, only the canonical combo (λ=0, δ=1.0) is executed per z_dim; all other (λ,δ) combinations are skipped at runtime with a clear log message. The `sweep_stage_a.yaml` comment was updated to explain this.
- **Status:** ✅ RESOLVED (2026-04-24)
- **Reproducibility impact:** LOW — no functional change; only compute efficiency improvement.

---

## Reproducibility Checklist

This checklist tracks what a cold reader needs to fully reproduce the project from scratch. Update as items are resolved.

| Item | Status | Notes |
|------|--------|-------|
| Conda environment specification | 🔲 PENDING | Need `environment.yaml` or `requirements.txt` with pinned versions |
| Editable install documented | ✅ DONE | In CONTEXT.md and README.md |
| `KMP_DUPLICATE_LIB_OK` workaround documented | ✅ DONE | In CONTEXT.md |
| All package installs documented | ✅ DONE | CONTEXT.md + this file |
| Bit-cost formula deviation documented | ✅ DONE | docs/MATH.md §2.3 |
| Bit-cost formula fixed in code | ✅ DONE | channels.py:61,94 fixed Phase 1; unit tested |
| All channel MC tests passing | ✅ DONE | 23/23 passing (Phase 1) |
| Env determinism tests passing | ✅ DONE | 20/20 passing (Phase 1) |
| Trainer loss-wiring tests passing | ✅ DONE | 8/8 passing (Phase 1) |
| `configs/baseline_v0.yaml` created | ✅ DONE | Phase 1 |
| NSD variance formula corrected | ✅ DONE | MATH-002: δ²/4 not δ²/6; MATH.md updated |
| Per-goal bit-allocation logging | ✅ DONE | buffer.py goal_ids; trainer bits_goal_0…5; CSV columns |
| `pyarrow` installed + parquet output enabled | 🔲 PENDING | Before Phase 2 sweeps (INSTALL-006) |
| WandB sweep scripts created | ✅ DONE | Phase 2 Session 3: `onpolicy/scripts/sweeps/toyproblem/` |
| `analysis/` module created | ✅ DONE | Phase 2 Session 3: load_runs, stats, plots, sweep_convergence, report_baseline |
| `docs/STATS.md` created | ✅ DONE | Phase 2 Session 3: 8 methods explained in plain English |
| Baseline channels implemented | ✅ DONE | Phase 2 Session 3: additive_uniform, gaussian, ste4/8/16 in channels.py |
| Env + MAPPO validation scripts | ✅ DONE | Phase 2 Session 3: experiments/validate_environment.py, validate_mappo.py |
| Channel comparison script | ✅ DONE | Phase 2 Session 3: experiments/run_channel_comparison.py |
| Channel comparison completed | ✅ DONE | 2026-04-23: 40 runs (8 channels × 5 seeds × 1M steps); results in `docs/results/channel_comparison/` |
| H(G) and per-channel bit formalization | ✅ DONE | 2026-04-23: `channels.py` H_GOAL_BITS, GOAL_OPTIMAL_BITS, true_bits_from_m, transmission_bits_per_elem; MATH.md §10 |
| Bit-calculation MC validation | ✅ DONE | 2026-04-23: all channels validated; surrogate properties documented in MATH.md §10.2 |
| `true_bits_per_msg` logging fixed | ✅ DONE | 2026-04-23: CODE-004; train.py, load_runs.py, report_baseline.py all updated |
| Stage A sweep 90 corrupted runs deleted | ✅ DONE | 2026-04-23: CODE-004; runs deleted; sweep must restart from 0 |
| Stage A config includes additive_uniform | ✅ DONE | 2026-04-24: CODE-007; additive_uniform added to sweep_stage_a.yaml |
| `none` channel redundancy fixed in run_sweep.py | ✅ DONE | 2026-04-24: CODE-008; auto-deduplication logic added |
| Jensen gap uncertainty bands in plots | ✅ DONE | 2026-04-24: plot_per_goal_bits enhanced; paper_figures.py created |
| Run logs moved to toyproblem-local directory | ✅ DONE | 2026-04-24: runs/ directory created inside toyproblem; PLAN.md/CONTEXT.md updated |
| Research paper figure scripts created | ✅ DONE | 2026-04-24: analysis/paper_figures.py — 10 figures with hypothesis/analysis/conclusion |
| run_sweep key collision fixed + 95 runs recovered | ✅ DONE | 2026-04-24: CODE-009; _run_key now uses only varying params; 95 existing runs renamed from checkpoint args |
| Stage A sweep re-launched (post CODE-009 fix) | ✅ DONE | 2026-04-24: sweep running; ~98 of ~2175 done as of session end |
| `configs/baseline_best.yaml` frozen | 🔲 PENDING | Phase 2 — after Stage A/B sweeps complete |
| Per-pillar unit tests passing | 🔲 PENDING | Phase 3 |
| `REPRODUCE.md` written | 🔲 PENDING | Phase 5 |
| `scripts/reproduce/` entry points created | 🔲 PENDING | Phase 5 |
