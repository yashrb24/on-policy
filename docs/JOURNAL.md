# Research Journal

Chronological log of experiments, decisions, and findings.  
Format: `## YYYY-MM-DD — Title` followed by context, what was done, and what was learned.

**Getting up to speed?** Start with `docs/CONCLUSIONS.md` — it has a recommended
reading-order guide at the top. Do not start with this file; the journal is a log,
not a summary.

---

## 2026-04-[early] — Histogram estimator replaces DLM; qphi_gap confirmed at 0.004 bits

**Context:** The DLM (deep latent model) source coding estimator had an irreducible
qphi_gap floor of 12–14 bits across all variants. This made it impossible to measure
the true Shannon gap H(m) − H(G), because the estimator error was larger than the
quantity we were trying to measure.

**What was done:** Replaced DLM with a simple Laplace-smoothed histogram estimator
(`source_coding.py: MessageHistogram`). Applied to the held-out message distribution
from a converged speaker.

**What was learned:** The histogram estimator achieves qphi_gap ≈ 0.004 bits from
update 1 — a 3,000× improvement over DLM. This was the key unblocking result for
all subsequent source coding analysis.

**Decision:** Commit to histogram + score function as the primary source coding
method. Abandon DLM as a training loss (keep only for comparison figures F8–F12).

---

## 2026-04-[mid] — Rate decomposition theory formalised

**Context:** With a reliable estimator, we needed a theoretical framework for
decomposing the total rate R(m) into interpretable components.

**What was done:** Derived the decomposition identity:
```
H_factored(m) = H(G) + H(m|goal) + TC(m) + ε_estimator
```
where:
- H(G) = 1.8128 bits (Shannon entropy of 6-goal distribution)
- H(m|goal) = E_g[Σ_k H_binary(frac(z_k/δ))] — within-goal randomness from dither
- TC(m) = Σ_k H(m_k) − H_joint(m) — cross-dimension correlation
- ε_estimator ≈ 0.004 bits — histogram fit quality

Implemented `JointMessageHistogram` for joint entropy and TC computation,
`dither_channel_loss` and `dither_channel_stats` for Phase 2, and `post_hoc_coding.py`
for offline rate decomposition from a checkpoint.

**What was learned:** The decomposition separates three distinct sources of overhead
above H(G): within-goal noise (reduced by Phase 2 dither), cross-dim correlation
(eliminated by joint coding post-hoc), and estimator noise (negligible with histogram).

---

## 2026-04-[late] — Figure list F1–F20 designed; experiment plan finalised

**Context:** Needed to translate theoretical claims into concrete empirical evidence.
Each claim in PILLAR_P2_v2.md needed a figure backed by training runs.

**What was done:** Designed F1–F20 figure list (PILLAR_P2_v2.md §16), mapping each
figure to the experiment stage that produces its data. Implemented the full experiment
runner: `experiments/run_sc_experiments.py` (7 stages: BASELINE, SC_MAIN, SC_TWOPHASE,
PARETO, GEOMETRY, LIVE, DLM_CMP) and `scripts/run_p2_experiments.sh` (resource-aware
launcher with dynamic CPU throttling).

**Decision:** Run all stages at 5 seeds, 2M steps (3M for two-phase runs), on local
MacOS laptop with MAX_PARALLEL=2 and LOAD_PER_CORE=0.50.

---

## 2026-05-01 — All P2 experiments completed (BASELINE through LIVE)

**Context:** Launched run_p2_experiments.sh covering stages BASELINE, SC_MAIN,
SC_TWOPHASE, PARETO, GEOMETRY, LIVE (DLM_CMP skipped — requires separate runner).

**What was done:** 35 distinct experiment configurations × 5 seeds = 175 training runs.
Total compute: ~350M environment steps across all seeds. All runs completed without
interruption.

**Key results at convergence (mean across 5 seeds):**

| Configuration | SR | true_bits | hist_H |
|---|---|---|---|
| no_comms (ceiling) | 1.000 | 8.74 | — |
| mag_lam1e-3 (best mag) | 1.000 | 6.52 | — |
| sc_live_entropy (live SC) | 1.000 | 20.06 | 7.05 |
| sc_twophase_dither1e-3 (Phase 2) | 0.986 | **5.14** | 7.51 |
| sc_twophase_dither1e-4 (collapsed) | 0.047 | 23.90 | 5.22 |

**Surprises:**
1. Live SC (`sc_live_entropy`) caused true_bits to blow up 3× above the ceiling while
   leaving hist_H = 7.05 flat. This revealed that the two metrics come apart: the score
   function is growing z magnitude to reach lower-indexed bins, not restructuring p(m).
2. Phase 2 at λ=1e-4 collapsed catastrophically (SR=0.047) while λ=1e-3 was stable
   (SR=0.986). The 25× difference in λ produced qualitatively different outcomes.
3. geom_no_anchor = no_comms exactly: RL alone never reduces communication rate.
   The magnitude anchor is strictly necessary.

---

## 2026-05-04 — Post-hoc rate decomposition run; §17 documented

**Context:** The `sc_posthoc_mag` checkpoint is the primary Phase 1 result. Running
`post_hoc_coding.py` on it gives the empirical rate decomposition for F1.

**What was done:**
- Fixed two bugs in `post_hoc_coding.py`: checkpoint used `state_dict` key with
  `speaker.` prefix (not bare `"speaker"`), and SD channel returns `(z_hat, info_dict)`
  not a 3-tuple. Both fixed.
- Ran on all 5 seeds (N=50K messages, δ=1.0, z_dim=3, CPU to avoid MPS buffer error).
- Documented all experiment results in PILLAR_P2_v2.md §17 (Sections 17.1–17.9).

**Key findings from decomposition:**

| Component | Mean | Std |
|---|---|---|
| H(G) | 1.8128 | 0 |
| H_factored | 9.48 | 0.29 |
| H_joint | 4.75 | 0.23 |
| TC | 4.73 | 0.32 |
| H(m\|goal) [dither] | 2.17 | 0.17 |
| decomp residual | 0.77 | 0.39 |

**Key insight:** TC = 4.73 bits (63% of the rate gap above H(G)) is the dominant term.
The policy encodes 6 goals using only 48 distinct joint tuples — joint arithmetic coding
cuts the cost from 9.48 → 4.75 bits for free.

**Residual caveat:** Seed 0 shows residual = 0.18 bits; seeds 1–4 show 0.57–1.11 bits.
The likely cause is noisy H(m|goal) estimation (100 z-samples per goal). Plan: increase
to 1000 samples per goal in a future post_hoc run.

---

## 2026-05-04 — Message geometry visualisation created

**Context:** The user proposed a visualisation: map learned messages m to goal positions
and show how Phase 2 and the collapsed policy differ geometrically.

**What was done:** Implemented `analysis/message_geometry.py` (4-panel figure):
- Panel A: z-space scatter coloured by goal (with bin grid)
- Panel B: message-space scatter (integer tuples, coloured by goal)
- Panel C: frac(z/δ) histogram (shows Phase 2 convergence)
- Panel D: message-goal confusion matrix (shows collapse)

Run on three checkpoints: Phase 1 (sc_posthoc_mag/0), Phase 2 stable
(sc_twophase_dither1e-3/0), Phase 2 collapsed (sc_twophase_dither1e-4/0).

**What was learned from the figure:**
- Phase 1: 6 tight clusters in z-space, clean one-to-one goal→tuple mapping in panel D.
- Phase 2 stable: z moves toward bin boundaries (frac distribution bimodal near 0/1);
  goal→tuple mapping remains clean.
- Phase 2 collapsed: z drifts to |z| ≈ 100–200 (score function blowup + dither failure);
  all goals map to the same extreme message tuples; panel D shows complete mixing.

Saved to `results/figures/message_geometry.{png,pdf}`.

---

## 2026-05-04 — Repository reorganisation; experiment registry and conclusions created

**Context:** Experimental outputs were scattered (runs/, figures/, figures_test/,
post_hoc JSONs in individual seed directories). No central tracking of experiment
dependencies or up-to-date conclusions.

**What was done:**
- Created `results/` directory with `aggregated/`, `post_hoc/`, `figures/` subdirs.
- Implemented `aggregate_results.py`: rebuilds `results/aggregated/summary.csv` and
  per-experiment `aggregate.json` for post_hoc decompositions. Run after every new
  experiment or analysis run.
- Created `docs/EXPERIMENT_REGISTRY.md`: tracks all 36 experiments (E01–E36) with
  status, figures fed, and invalidation dependencies.
- Created `docs/CONCLUSIONS.md`: 10 conclusions (C1–C10) with evidence, confidence
  levels, and open questions.
- Created `docs/JOURNAL.md` (this file).

**Pending:** DLM comparison (F8–F12), geometry post_hoc analysis (F6/F7), z_dim sweep
(F15), and Phase 2 listener gradient fix (E39).

---

## 2026-05-04 — Repository reorganisation: single media directory, CLAUDE.md, figure map

**Context:** Results were scattered across `figures/`, `figures_test/`, and
`results/toyproblem/`. No single place to find all figures. No project CLAUDE.md.
Python scripts had stale default output paths.

**What was done:**
- Consolidated all media into `results/figures/` as the single canonical location:
  - `results/figures/archive/` — all pre-registry figures (sweep_stage_a, p2_ablation,
    plus 2 standalone files p2_gradient_balance, p2_per_dim_entropy)
  - `results/figures/` — current figures (message_geometry.{png,pdf})
- Moved text reports to `results/reports/`.
- Deleted `figures_test/` and `results/toyproblem/` (all files accounted for above).
- Fixed `report_baseline.py` default `--out_dir` from `results/toyproblem` to `results`.
- Created `on-policy/CLAUDE.md` with: directory layout, canonical output paths,
  figure-to-experiment mapping table (F1–F20), and workflow checklists.
- Updated `results/README.md` to reflect new structure.

---

## Figure-to-experiment map (all F1–F20)

Complete mapping of paper figures to the experiment IDs (see `docs/EXPERIMENT_REGISTRY.md`)
and data files that produce them. Updated as figures are generated.

| Figure | Description | Status | Data | Experiments |
|---|---|---|---|---|
| **F1** | Rate decomposition bar chart | data ready, script needed | `results/post_hoc/sc_posthoc_mag/aggregate.json` | E09 + post_hoc_coding.py |
| **F2** | Shannon gap training curve | data ready, script needed | `results/aggregated/summary.csv` rows for no_comms, mag_lam5e-4, sc_live_entropy | E01, E04, E10 |
| **F3** | Implicit prior mismatch | analysis script needed | E09 checkpoint | E09 |
| **F4** | Gradient direction alignment | analysis script needed | E09 checkpoint | E09 |
| **F5** | (SR, H(m)) Pareto frontier | data ready, script needed | summary.csv pareto_mag_* and pareto_sc_* rows | E23–E36 |
| **F6** | Per-goal z geometry (Gram matrix) | E16/E17 data ready; E18/E19 INVALIDATED — use E42/E43 | geom_* checkpoints | E16, E17, E42, E43 |
| **F7** | TC and H_dither per geometry config | E16/E17 data ready; E18/E19 INVALIDATED — use E42/E43 | geom_* checkpoints | E16, E17, E42, E43 |
| **F8** | qphi_gap: DLM vs histogram | DLM runs needed | DLM runs + E09 metrics | E09, E41 |
| **F9** | DLM floor K sweep | DLM runs needed | run_p2_ablation P2-A | E41 |
| **F10** | DLM circular gradient | DLM runs needed | run_p2_ablation P2-FIX fix_B1_gate | E41 |
| **F11** | Context bound ordering | analysis needed | E09 + context model variants | E09 |
| **F12** | DLM warm-start failure | DLM runs needed | run_p2_ablation P2-FIX warmup sweep | E41 |
| **F13** | Three-way post-hoc vs live | data ready, script needed | summary.csv live_A/B/C rows | E20–E22 |
| **F14** | Score function gradient ratio | gradient logging needed | live_B training run with grad logging | E21 |
| **F15** | TC grows with z_dim | z_dim 1,2 runs needed | post_hoc at z_dim ∈ {1,2,3} | E09, E37, E38 |
| **F16** | Phase 2 closes H_dither_channel | E12–E14 REINTERPRETED; use E44–E46 for correct H(m\|goal) reduction | sc_twophase_* metrics.csv | E44–E46 |
| **F17** | ε_MLE converges at O(1/√N) | analysis needed | E09 checkpoint, N sweep | E09 |
| **F18** | SVD diagonalisation collapses TC | deferred | isotropic activation variant | future |
| **F19** | Moving-target error over training | checkpoint logging needed | live_B checkpoint series | E21 |
| **F20** | Three-way Pareto at SR=1 | data ready, script needed | summary.csv live_A/B/C + post_hoc H_joint | E20–E22 |

**Currently generated figures:**
- `results/figures/message_geometry.{png,pdf}` — z-space / message-space / frac /
  confusion for Phase 1, Phase 2 stable, Phase 2 collapsed. Generated by
  `analysis/message_geometry.py`. Not part of F1–F20 but illustrates C6, C8.

**Archive figures** (pre-registry era, in `results/figures/archive/`):
- `sweep_stage_a/main/` — fig1_rate_distortion, fig2_channel_comparison (early Pareto)
- `sweep_stage_a/appendix/` — appA–appH (hyperparameter transparency appendix)
- `sweep_stage_a/p2/` — p2_qphi_gap, p2_shannon_gap, p2_context_bounds (DLM era)
- `p2_ablation/` — p2a_overview, p2a_dynamics, p2a_entropy_diagnostics, etc.
- Standalone: p2_gradient_balance, p2_per_dim_entropy

---

## 2026-05-07 — E42–E46 completed; post_hoc on E42/E43; F6, F7, F16 generated

**Context:** DITHER_CORRECTED second batch (E42–E46, corrected dither loss + E39 listener
fix) completed. All 25 runs finished with SR=1.000 — confirming E39 fix prevents collapse.

**Post_hoc on E42/E43 (geom configs):**

| Config | H_dither (mean) | TC (mean) | H_joint (mean) | Residual range |
|---|---|---|---|---|
| geom_no_anchor | 2.27 | 5.01 | 4.64 | (prior run, valid) |
| geom_mag_only | 2.17 | 4.73 | 4.75 | (prior run, valid) |
| geom_dither_only_v2 | **0.78** | **3.42** | **2.60** | −0.008 to +0.017 bits |
| geom_both_v2 | **0.57** | **2.80** | **2.39** | −0.008 to +0.017 bits |

All residuals < 0.02 bits — decomposition identity holds. Corrected dither loss reduces
H_dither from 2.17 → 0.57–0.78 bits as expected (frac → 0.5).

**New finding:** The corrected dither loss also reduces TC (from 4.73 → 2.80–3.42 bits).
This was not anticipated: forcing z toward bin centres reduces cross-dimensional correlation
in addition to within-goal noise. The mechanism is likely that bin-centre alignment
reduces the spread of z values, which compresses the implicit codebook and reduces TC.
Both H_dither and TC contribute to the H_joint reduction (4.75 → 2.39–2.60 bits).

**Phase 2 results summary (E44–E46):**

| λ | SR | hist_H | Phase 2 trigger |
|---|---|---|---|
| 1e-4 | 1.000 | 4.88 | update 54 |
| 5e-4 | 1.000 | 4.59 | update 54 |
| 1e-3 | 1.000 | 4.65 | update 54 |

Compare to Phase 1 baseline: hist_H ≈ 7.0 bits. Phase 2 reduces H(m) by ~2.4 bits.
Compare to original E12 (λ=1e-4): SR=0.047 (collapsed). E39 fix was decisive.

**Figures generated:** F6, F7, F16 via `analysis/geometry_phase2_figures.py`.

---

## 2026-05-07 — F3, F4, F17 generated from E09 checkpoint

**What was done:** Implemented `analysis/checkpoint_analysis_figures.py` — a single script
generating three paper figures that require only the E09 (sc_posthoc_mag) checkpoint and
training metrics:

**F3 — Implicit prior mismatch** (`F3_implicit_prior_mismatch.{pdf,png}`):
- Panel A: Rate decomposition stacked bar (mean ± SEM across 5 seeds) showing
  H(G) | H_dither | TC | ε = H_factored. TC = 3.28 bits is labelled as the
  "implicit prior mismatch" — the cost of using a factored q(m) instead of joint.
  H_joint = 4.16 bits shown as the joint coding target.
- Panel B: Message scatter (m_0 vs m_1, coloured by goal) from E09 seed 0, illustrating
  the correlated goal→tuple structure that the factored histogram ignores.

**F4 — Gradient direction** (`F4_gradient_direction.{pdf,png}`):
- Revised from "gradient alignment at checkpoint" (misleading at SR=1.0 where
  task gradient ≈ 0) to training dynamics from E10 (sc_live_entropy, 5 seeds):
- Panel A: hist_H_empirical flat at ≈ 7.0 bits throughout training despite active SC.
- Panel B: true_bits_per_msg growing to 19.5 in E10 vs stable 7.0 in E09.
- Panel C: |sc_rate_loss| / |pg_loss| ratio (log scale). During early learning (SR < 0.5):
  ratio ≈ 0.13 (PPO dominates ~8:1). After convergence: SC dominates (ratio > 1).
  The policy structure is locked in by PPO before SC becomes significant.

**Key finding (F4):** "PPO dominates SC 100:1 during training" is an overstatement.
The measured ratio during learning is ~8-10:1, not 100:1. But the conclusion stands:
PPO sets the goal→tuple mapping during the SR-learning phase. By the time SC becomes
significant, the policy structure is fixed and SC can only reduce |z| (true_bits),
not restructure p(m). C5 updated accordingly.

**F17 — ε_MLE convergence** (`F17_eps_mle_convergence.{pdf,png}`):
- Panel A: |decomp residual| vs 1/√N (log-log). Residual oscillates around zero with
  amplitude O(1/√N) — consistent with histogram sampling noise.
- Panel B: K_obs (distinct joint tuples) stabilises at 45 by N = 20,000.
- ε_MLE bound = K_obs / √N drops from 0.43 (N=500) to 0.20 (N=50K).

**Figures generated:**
- `results/figures/F3_implicit_prior_mismatch.{pdf,png}`
- `results/figures/F4_gradient_direction.{pdf,png}`
- `results/figures/F17_eps_mle_convergence.{pdf,png}`

---

## 2026-05-07 — DLM comparison (P2-FIX) launched; stale log_dir fixed

**Context:** The DLM comparison runs (E41) were the last planned experiment batch for
Pillar 2. All histogram-era experiments (E01–E46) were done. Needed to run
`run_p2_ablation.py --stage P2-FIX` to produce data for F8, F10, F12 (DLM qphi_gap vs
histogram, circular gradient failure, warm-start failure).

**Pre-launch checks:**
1. Verified all P2-FIX flags are supported in `train.py`: `use_entropy_model`,
   `entropy_model_K`, `entropy_model_type`, `entropy_model_context`, `lr_qphi_mult`,
   `n_qphi_steps`, `n_warmup_steps`, `qphi_bwd_gate_threshold`, `use_ema_prior`,
   `ema_prior_momentum`, `phase1_sr_threshold` — all confirmed present.
2. Verified metrics needed for F8–F12 are logged in `trainer.py`:
   - `qphi_gap` (logged at line 578)
   - `entropy_loss_magnitude` (logged at line 609)
   - `speaker_grad_norm` (logged at line 504)
   - `warm_start_bits_final` (logged at line 556)
3. Fixed stale `log_dir` default in `run_p2_ablation.py`: was `"runs/toyproblem/p2_ablation"`,
   corrected to `"runs/p2_ablation"` (3 occurrences: argparse default + 2 docstring examples).

**What was launched:**
- P2-FIX stage: 5 configs × 5 seeds = 25 runs into `runs/p2_ablation/`
  - `fix_baseline`: DLM reference at P2-A defaults (n_warmup=5000, n_qphi_steps=3, lr_mult=10)
  - `fix_B1_gate`: + backward gate (threshold=2.0) + better q_φ (n_warmup=50000, nsteps=20)
  - `fix_B2_gate_lambda`: + λ=1e-2 (scale fix)
  - `fix_B3_ema`: + EMA prior (momentum=0.95)
  - `fix_B4_twophase`: Phase 1 magnitude → Phase 2 entropy at SR≥0.995
- Log: `runs/p2_ablation/p2_fix_run.log`

**Expected outcomes:**
- `fix_baseline` should reproduce the ~12–14 bit qphi_gap seen in early DLM experiments
- `fix_B1`–`fix_B3` should progressively close the gap (gate prevents circular gradient;
  better q_φ training reduces NLL; EMA prior stabilises)
- `fix_B4_twophase` may achieve lowest gap (q_φ trains passively in Phase 1 without
  circular gradient; Phase 2 starts with a pre-fitted model)

**What was not launched (deferred):**
- P2-A full model selection (115 runs) — superseded by histogram approach; only needed
  if DLM results reveal a tractable fix worth pursuing further

---

## 2026-05-08 — E41 (P2-FIX) complete; F8, F10, F12 generated; DLM story closed

**Context:** P2-FIX stage (25 runs: 5 configs × 5 seeds, `runs/p2_ablation/`) completed.
These runs exist to demonstrate *why* the histogram replaced the DLM, not to find a
working DLM config.

**Results summary:**

| Config | qphi_gap (final, mean) | SR (mean) | Interpretation |
|---|---|---|---|
| fix_baseline (DLM ref.) | 9.71 ± 0.62 bits | 0.977 | 2400× worse than histogram |
| fix_B1_gate (+gate +warmup) | 18.79 ± 3.79 bits | 0.993 | gap grows WORSE |
| fix_B2_gate_lambda (+λ=1e-2) | 21.97 ± 6.19 bits | 0.976 | gap grows even worse |
| fix_B3_ema (+EMA prior) | 20.33 ± 3.27 bits | 0.982 | no improvement |
| fix_B4_twophase (warm start) | **0.046 ± 0.000 bits** | **0.536** | low gap but SR collapse |
| Histogram (E09, reference) | 0.004 bits | 1.000 | baseline for comparison |

**Training dynamics (key insight for F10):** All DLM configs start with qphi_gap ≈ 0.07
bits (early in training, policy is untrained → message distribution is narrow and simple).
As the policy learns a rich goal→tuple codebook (48 distinct tuples), the distribution
becomes complex and the DLM cannot track it — gap grows monotonically. The gate (B1–B3)
is always CLOSED (gap >> 2.0 bit threshold), so more q_φ training steps simply fit the
distribution harder without fixing the gap. This IS the circular gradient: q_φ chases a
moving target driven by the magnitude-only speaker, and the gap grows as the target
becomes more complex.

**fix_B4_twophase story (F12):** The two-phase trick avoids the circular gradient in
Phase 1 (magnitude only → simple distribution → q_φ fits well → gap = 0.046 bits).
But when Phase 2 activates the entropy backward, SR collapses to 53.6%. Fitting q_φ is
necessary but not sufficient — the DLM gradient itself is destructive. The histogram
avoids this entirely: it is post-hoc and never drives the speaker.

**F9 deferred:** The K-sweep (P2-A) was not run. F8, F10, F12 together give the complete
DLM failure story; F9 (showing gap vs K) adds little beyond what F10 already shows.
Marked deferred in CLAUDE.md.

**Figures generated:**
- `results/figures/F8_qphi_gap_comparison.{pdf,png}` — bar chart + trajectory
- `results/figures/F10_circular_gradient.{pdf,png}` — gap trajectories, all 5 configs
- `results/figures/F12_warmstart_failure.{pdf,png}` — B4 qphi_gap (low) vs SR (collapsed)

**summary.csv rebuilt** via `aggregate_results.py`.

---

## 2026-05-08 — F11 generated; C3 corrected

**F11 — Context bound ordering** (`F11_context_bound_ordering.{pdf,png}`):

The figure reframes the DLM "context A/B/C" concept from PILLAR_P2_v2.md §7.2 in
terms of the histogram decomposition. Instead of comparing three DLM model variants
(which were never run in measurement mode), F11 uses the four bounds derivable from
E09 post_hoc data:

| Context level | Rate (mean±std) | What is available at the receiver |
|---|---|---|
| H_factored | 7.44±0.44 bits | per-dim frequencies only (factored coding) |
| H_joint | 4.16±0.34 bits | full joint frequency table (joint coding) |
| H(m\|goal) | 2.34±0.31 bits | goal label (goal-conditional coding) |
| H(G) | 1.813 bits | theoretical floor |

The ordering H_factored ≥ H_joint ≥ H(m|goal) ≥ H(G) holds strictly for **all 5 seeds**
with no exceptions. Gaps: TC = 3.28 bits (joint coding benefit, free post-hoc), goal
context = 1.82 bits (requires goal label at decoder), residual = 0.53 bits (irreducible
given current z distribution).

**C3 corrected:** The prior values (H_factored=9.48, TC=4.73) were from the pre-fix
`collect_messages` that cycled goals uniformly. The corrected values are:
H_factored=7.44±0.44, H_joint=4.16±0.34, TC=3.28±0.22. The 2-bit difference is
entirely explained by the goal-sampling fix (C4). TC at z_dim=3 (3.281 bits) now
matches C12 exactly.

---

## 2026-05-11 — §11.8 NSD channel ablation + Phase 2 Pareto extension (E47–E52)

**Context:** Two simultaneous ablation batches launched after Pillar 2 closure: (1) §11.8
channel interaction — NSD vs SD for histogram estimator and Phase 2; (2) Phase 2 Pareto
curve extension — 4 additional λ_dither values to trace the full SR vs H_joint frontier.

### E47–E48: NSD channel generalisation (§11.8)

10 runs (2 configs × 5 seeds). Post-hoc coding run on both.

| Config | Channel | H_joint (bits) | SR |
|---|---|---|---|
| E47 `nsd_posthoc_mag` | NSD | 4.137 ± 0.165 | 1.000 |
| E48 `nsd_twophase_dither5e-4_v2` | NSD | 2.413 ± 0.141 | 1.000 |
| E09 `sc_posthoc_mag` (SD ref) | SD | 4.156 ± 0.344 | 1.000 |
| E45 `sc_twophase_dither5e-4_v2` (SD ref) | SD | 2.385 ± 0.174 | 1.000 |

Differences of 0.019 and 0.028 bits — both within 1 std. **Channel agnosticism confirmed.**
The histogram builds q_hist from actual m samples regardless of channel type; the dither
loss operates on frac = fractional part of z/δ, which is also channel-agnostic. Added as C16.

### E49–E52: Phase 2 Pareto curve extension

20 runs (4 configs × 5 seeds). Post-hoc coding run on all.

Full 7-point λ_dither sweep (E44–E46 + E49–E52), all SR=1.000:

| λ_dither | Exp | H_joint (bits) | H(m\|goal) |
|---|---|---|---|
| 5e-5 | E49 | 2.491 ± 0.201 | 0.670 |
| 1e-4 | E44 | 2.539 ± 0.105 | 0.722 |
| 2e-4 | E50 | 2.546 ± 0.193 | 0.733 |
| 5e-4 | E45 | 2.385 ± 0.174 | 0.570 |
| 1e-3 | E46 | 2.286 ± 0.111 | 0.474 |
| 2e-3 | E51 | 2.232 ± 0.052 | 0.416 |
| **5e-3** | **E52** | **2.094 ± 0.047** | **0.283** |

**Key finding:** SR=1.000 at all 7 λ values — no degradation even at λ=5e-3. This is
stronger than predicted (E52 was expected to show SR tradeoff). Phase 2 is robust across
the full tested range. The non-monotonicity at low λ (5e-5, 1e-4, 2e-4 cluster ≈2.5 bits)
is within noise; monotone improvement visible for λ≥5e-4.

Best Phase 2 operating point: λ=5e-3, H_joint=2.094 bits (0.28 bits above H(G)=1.813).

**Implication for F5:** The Phase 2 Pareto "curve" is a horizontal band at SR=1.000 with
H_joint decreasing from ~2.55 to ~2.09 bits as λ increases. All 7 points strictly dominate
the entire Phase 1 mag / live SC Pareto frontier (best: H_joint≈4.16 at SR=1). F5 can
be updated to show Phase 2 as a third curve, with H_joint on x-axis (post_hoc metric).

### Docs updated

- CONCLUSIONS.md: C10 expanded with 7-point λ sweep table + NSD comparison; C16 added
- EXPERIMENT_REGISTRY.md: E47–E52 marked DONE
- PILLAR_P2_v2.md, on-policy/CLAUDE.md: pending update for F5 and §11.8

---

**Open threads (as of 2026-05-11):**

| Item | Status |
|---|---|
| F21 Phase 2 ablations figure | **DONE** — `results/figures/F21_phase2_ablations.{pdf,png}` |
| on-policy/CLAUDE.md F21 entry | **DONE** |
| PILLAR_P2_v2.md §11.8 status | **DONE** — results + F21 ref added |

---

## 2026-05-11 — Pillar 2 closure: post_hoc on E44–E46; F20 updated; PILLAR_P2_v2.md corrected

**Context:** Four remaining items required for Pillar 2 closure.

### 1 — post_hoc_coding.py on E44–E46 (sc_twophase_v2, 15 checkpoints)

Ran on all 3 configs × 5 seeds at N=50K messages. Key results:

| Config | H_joint (mean±std) | H(m\|goal) (mean) | SR |
|---|---|---|---|
| E44 λ=1e-4 | 2.54 ± 0.11 bits | 0.72 bits | 1.000 |
| E45 λ=5e-4 | 2.39 ± 0.17 bits | 0.57 bits | 1.000 |
| E46 λ=1e-3 | **2.29 ± 0.11 bits** | **0.47 bits** | 1.000 |

Phase 2 reduces H_joint from 4.16 (Phase 1) to 2.29 bits — 45% reduction — while
maintaining SR=1.000 at all three λ values. H(m|goal) drops from 2.34 → 0.47 bits;
the corrected dither loss (frac→0.5) is effective. Residuals all < 0.02 bits.

All 15 JSONs saved to `results/post_hoc/sc_twophase_dither{1e-4,5e-4,1e-3}_v2/`.

### 2 — F20 updated with Phase 2 cluster

`plot_f20_three_way_pareto` extended to include Phase 2 v2 configs (E44–E46) as a
fourth group. Phase 2 uses H_joint from post_hoc aggregate.json as x-axis; live A/B/C
continue to use hist_H_empirical. The x-axis label documents the metric difference.

Resulting figure: Phase 2 cluster sits at 2.3–2.5 bits vs live configs at 7–8 bits,
all at SR=1.000. Pareto dominance of Phase 2 over all live configurations is decisive.
Saved to `results/figures/F20_three_way_pareto.{pdf,png}`.

C10 updated with full Phase 2 post_hoc numbers.

### 3 — PILLAR_P2_v2.md corrections

- §16 figure-to-experiment map: completely rewritten from stale placeholder status to
  current DONE/deferred status for all 20 figures.
- §16 F14: "~100:1" corrected to "~5:1" with explicit revision note.
- §16 F7: claim reattributed from "magnitude anchor reduces TC" to "dither loss is the
  primary TC reduction driver; magnitude anchor provides geometric separation (F6)".
- §14 failure modes table: "100:1 RL dominance" corrected to "~5:1 measured".

### 4 — EXPERIMENT_REGISTRY.md: E15 SUPERSEDED

E15 (`sc_twophase_dither1e-3` fixed listener, PLANNED) marked SUPERSEDED by E46.

---

**Open threads (as of 2026-05-11):** None. Pillar 2 is closed.

| Figure | Status |
|---|---|
| F9 (DLM K sweep) | **Deferred** — not needed for core claims |
| F18 (SVD diagonalisation) | **Deferred** — future architectural work |
| All other F1–F20 | **DONE** |

---

## 2026-05-10 — F14/F19 figure review; annotation bugs fixed; C5 revised; C15 added

**Context:** Visual review of generated F14 and F19 figures revealed that the figure
code had been written with incorrect prior assumptions about what the data would show.

### F14 — annotation bugs (code written for wrong assumption)

The figure code assumed the PPO/SC gradient ratio would eventually cross 1.0 (SC
overtakes PPO), based on the F4 loss-scale finding ("SC dominates after convergence").
The actual data: PPO/SC grad ratio = 4.39 early, 4.96 late — PPO is always 4–5× SC.

**Bugs fixed:**
- Panel B title changed from "PPO leads early, SC dominates at convergence" → "PPO
  consistently leads SC throughout training"
- Late annotation changed from "(SC > PPO, but too late)" → "(PPO > SC throughout)"
- Crossover detection code (looking for mean_sc > mean_ppo, which never triggers)
  replaced with a dominance-fraction annotation ("PPO > SC in 100% of updates")
- Module docstring and suptitle updated to reflect the actual finding

**What F14 actually shows:** PPO speaker grad norm dominates SC by a stable 4–5× factor
at every stage of training. SC never gains leverage. This is a STRONGER result than F4
implied — it's not that PPO dominates only during learning and then loses, it's that PPO
never loses the gradient contest on the speaker.

**C5 revised:** The "SC dominates after convergence" claim from F4 was based on the
scalar loss ratio (sc_loss/pg_loss > 1 after convergence). F14 shows this is an artifact:
pg_loss → 0 as the task is learned, so any fixed SC loss looks numerically large. At the
actual gradient level, PPO's influence on the speaker never weakens.

### F19 — wrong assumption about H(m) shape

The figure code assumed `true_H_offline` would stay flat at ~7 bits (matching the
hist_H_empirical training metric). The actual data: true_H_offline drops from ~6 bits
early to 2.42 bits at convergence, while hist_H_empirical stays flat at ~7 bits.

**Panel A title fixed:** "True H(m) stays flat: SC gradient cannot reduce entropy" →
"True H(m) decreases via z-blowup, not compression (hist_H overestimates)"

**Module docstring updated** with the correct description of the two-metric divergence.

**What F19 actually shows:**
- hist_H_empirical (training metric) overestimates true H(m) by ~4.6 bits because: (a)
  minibatch goal sampling is approximately uniform (inflating H(G) toward 2.585 bits);
  (b) it estimates factored H (per-dim sum), not joint H (off by TC ≈ 3.28 bits).
- true_H_offline decreases toward H(G) as z-blowup puts each goal in separate
  non-overlapping integer bins — degenerate: H(m) → H(G) trivially when no two goals
  share a bin, but the channel cost is 20 bits to transmit the extreme bin index.
- The "moving target" (‖Δθ‖ dropping 6× but nonzero throughout) confirms SC gradient
  is active but the speaker keeps shifting, making each gradient step stale.

**C15 added:** Documents the hist_H overestimation mechanism, the degenerate H → H(G)
via z-blowup, and why this doesn't constitute useful compression.

**Figures regenerated:** Both F14 and F19 saved to results/figures/ with correct annotations.

---

## 2026-05-08 — F14/F19 logging implemented; E21 diagnostic re-run launched; CSV writer bug fixed

**Context:** F14 (score function gradient ratio) and F19 (moving-target error) required
per-minibatch gradient decomposition and per-update parameter-delta metrics that were not
previously logged. Both figures need a live_B re-run with the new logging enabled.

**What was done:**

### trainer.py additions

Three new `MAPPOConfig` fields added (all default `False` / `10_000`; no effect on
existing runs):
- `log_grad_decomp: bool` — if True, computes PPO speaker grad norm and SC speaker grad
  norm per minibatch via `torch.autograd.grad(..., retain_graph=True)` before
  `total_loss.backward()`.
- `log_moving_target: bool` — if True, snapshots speaker params before the epoch loop and
  computes ‖Δθ‖ after. Also runs an offline forward pass (N=10K goals sampled from
  `_GOAL_PROBS`) to compute `joint_entropy_bits`.
- `moving_target_n: int` — sample size for the offline H(m) estimate.

Internal rename: `sc_rate_loss → _sc_rate_loss_tensor` (tensor reference kept alive for
F14 grad computation; the CSV metric key `"sc_rate_loss"` is unchanged).

New metrics logged per-update: `ppo_speaker_grad_norm`, `sc_speaker_grad_norm`,
`ppo_sc_grad_ratio`, `speaker_param_delta_norm`, `true_H_offline`.

### train.py additions

Three new CLI flags: `--log_grad_decomp`, `--log_moving_target`, `--moving_target_n`.
Five new columns added to `_build_csv_header()` and five corresponding
`metrics.get(...)` entries in the `p2_base_vals` writer.

### CSV writer alignment bug (found and fixed)

After the first launch the new columns showed all NaN. Root cause: a 3-column offset
between the header and the writer. The header had `dither_loss`, `H_dither_channel`,
`mean_frac` between `sc_rate_loss` and the new F14 columns; the writer was missing
those three entries. This caused the writer to shift the new F14 values into the
dither columns and leave the F14/F19 headers empty. Fix: inserted the three missing
`metrics.get(...)` calls in the correct position. Third launch (after two aborted
attempts) shows all 57 columns correctly aligned with real non-NaN values.

### figure functions implemented

`analysis/checkpoint_analysis_figures.py` extended with:
- `fig_f14_grad_ratio(live_b_dir, out_dir, n_seeds=5)` — two panels: (A) PPO and SC
  speaker grad norms vs training step (log-y) with crossover annotation; (B) ratio
  `ppo_sc_grad_ratio` (log-y) with early/late text boxes. Saves
  `F14_score_function_grad_ratio.{pdf,png}`.
- `fig_f19_moving_target(live_b_dir, out_dir, n_seeds=5)` — two panels: (A)
  `true_H_offline` + H(G) dashed reference; (B) `speaker_param_delta_norm` (‖Δθ‖)
  with early/late annotations. Saves `F19_moving_target_error.{pdf,png}`.
- CLI updated: `--live_b_diag_dir` flag (default `runs/live_B_diag`); `--figures F14`
  and `--figures F19` dispatch to the new functions.

### F5 x-axis annotation verified

Inspected `sc_ablation_figures.py: plot_f5_pareto()` (lines 222–274). The function
already uses `true_bits_per_msg` for magnitude runs and `hist_H_empirical` for SC runs,
with x-axis label `"Rate (bits/message)  [metric differs by condition — see caption]"`
and legend entries that name the metric per condition. No changes needed.

**E21 diagnostic re-run launched:**
- Path: `runs/live_B_diag/live_B_live_sc/` (5 seeds, 2M steps each)
- Flags: `--log_grad_decomp --log_moving_target --moving_target_n 10000`
- Status at launch: all 5 seeds started, ~8% completion (41/488 updates each)
- All 5 new columns confirmed non-NaN after column-alignment fix

**E21 run completed. Figures generated:**

Key measurements from `runs/live_B_diag/live_B_live_sc/` (5 seeds, 2M steps):

| Metric | Early (first 20 updates) | Late (last 20 updates) |
|---|---|---|
| PPO/SC grad ratio | 4.39 | 4.96 |
| ‖Δθ‖ | 0.146 | 0.023 |
| H(m) offline | — | 2.421 bits |
| H(G) reference | 1.813 bits | — |

**Findings:**
- F14: The PPO/SC gradient ratio is ~5:1 throughout training — PPO consistently dominates the
  speaker gradient, meaning the score function cannot restructure p(m) even late in training.
  This directly supports C5 (PPO locks goal→tuple mapping before SC becomes significant).
- F19: ‖Δθ‖ drops 6× from early to late training (0.146 → 0.023), confirming the moving-target
  problem is severe early and modest late. The offline H(m) = 2.421 bits vs H(G) = 1.813 bits
  leaves a 0.61-bit gap — the remaining overhead that joint coding or goal-conditional coding
  could close (see F11).

**Figures saved:**
- `results/figures/F14_score_function_grad_ratio.{pdf,png}`
- `results/figures/F19_moving_target_error.{pdf,png}`

---

## Open threads (as of 2026-05-08)

| Thread | Priority | Status | Blocking |
|---|---|---|---|
| F14 — score function gradient ratio | Low | **DONE** 2026-05-08 — `results/figures/F14_score_function_grad_ratio.{pdf,png}` | — |
| F19 — moving-target error | Low | **DONE** 2026-05-08 — `results/figures/F19_moving_target_error.{pdf,png}` | — |
| F9 — DLM K sweep | Low | **deferred** — F8/F10/F12 sufficient for paper | — |
| F18 — isotropic activations | — | deferred | F18 |

**Completed since last update:**
- ~~post_hoc on E44–E46~~ — DONE 2026-05-11; H_joint=2.29–2.54 bits; SR=1.000 all configs
- ~~F20 Phase 2 data point~~ — DONE 2026-05-11; Phase 2 cluster clearly Pareto-dominates live
- ~~PILLAR_P2_v2.md corrections~~ — DONE 2026-05-11; §16 status table, F14/F7 claims fixed
- ~~E15 SUPERSEDED~~ — DONE 2026-05-11
- ~~F14 score function gradient ratio~~ — DONE 2026-05-08; PPO/SC ratio ~5:1 throughout; supports C5
- ~~F19 moving-target error~~ — DONE 2026-05-08; ‖Δθ‖ drops 6× early→late; offline H(m)=2.421 bits
- ~~F11 context bound ordering~~ — DONE 2026-05-08; ordering confirmed for all 5 seeds
- ~~C3 corrected~~ — DONE 2026-05-08; H_factored=7.44, TC=3.28 (was 9.48, 4.73 pre-fix)
- ~~E41 DLM comparison (P2-FIX)~~ — DONE 2026-05-08; circular gradient and warm-start failure confirmed
- ~~F8 qphi_gap DLM vs histogram~~ — DONE 2026-05-08
- ~~F10 circular gradient failure~~ — DONE 2026-05-08
- ~~F12 warm-start failure~~ — DONE 2026-05-08
- ~~E42–E46 second batch~~ — DONE 2026-05-07; all SR=1.000
- ~~post_hoc on E42/E43~~ — DONE 2026-05-07; residuals < 0.02 bits
- ~~F3 implicit prior mismatch~~ — DONE 2026-05-07
- ~~F4 gradient direction~~ — DONE 2026-05-07
- ~~F6 z-space geometry~~ — DONE 2026-05-07
- ~~F7 geometry decomposition~~ — DONE 2026-05-07
- ~~F16 Phase 2 training curves~~ — DONE 2026-05-07
- ~~F17 ε_MLE vs N~~ — DONE 2026-05-07

---

## 2026-05-06 — OQ5 fix, goal-sampling fix, E39 listener fix, E42–E46 relaunch, E37/E38 and F15

**Context:** Follow-on from 2026-05-05 dither formula fix. Four separate code corrections
were identified, tested, and applied on 2026-05-06 before relaunching the corrected experiments.

---

### Fix 1 — OQ5: secondary bin selection in `source_coding_rate_loss`

`source_coding_rate_loss` always picked the secondary bin as `m_hi = floor(z/δ) + 1`.
For the floor SD channel, when `frac < 0.5`, the two reachable bins are `{m_base−1, m_base}`,
not `{m_base, m_base+1}`. The gradient direction was still correct (same sign for monotone
rate) but the magnitude was biased for `f < 0.5`.

**Fix applied:** `source_coding.py` updated to:
```python
frac = (z / delta) - torch.floor(z / delta)
m_base = torch.floor(z / delta).long()
low_frac = frac < 0.5
m_lo = torch.where(low_frac, m_base - 1, m_base)
m_hi = torch.where(low_frac, m_base,     m_base + 1)
```
**No re-runs required.** All experiments using `source_coding_rate_loss` are Phase 1 magnitude
runs, and the gradient direction was already correct — results are valid.

---

### Fix 2 — Goal sampling distribution mismatch (root cause of large decomposition residuals)

The 2026-05-04 residuals (mean 0.77 bits, up to 1.11 bits) were traced to a systematic error
in `post_hoc_coding.py: collect_messages`. Goals were cycled uniformly (equal weight per goal)
but `H_GOAL_BITS = 1.81 bits` is the entropy of the *non-uniform* training distribution
(`_GOAL_PROBS`). Uniform cycling gave an empirical H(G) of `log2(6) ≈ 2.585 bits`, introducing
a systematic offset of `2.585 − 1.81 = 0.77 bits` — matching the observed mean residual exactly.

Additionally, H_dither was averaged equally over goals rather than weighted by `_GOAL_PROBS[gid]`,
adding a small secondary error for non-uniform distributions.

**Fix applied:**
- `collect_messages`: replaced uniform cycling with `torch.multinomial(goal_probs_t, ...)`.
- `compute_decomposition`: weighted H_dither average by `_GOAL_PROBS[gid]`.
- Also noted: increasing z-samples from 100 → 1000 would be a no-op — the speaker is
  deterministic, so 100 identical goal vectors yield 100 identical z values. The fix was
  always goal *distribution*, not sample count.

**Re-ran post_hoc on E09 (sc_posthoc_mag) with corrected code:**

| Seed | Residual before | Residual after |
|---|---|---|
| 0 | 0.182 | −0.001 |
| 1 | 0.572 | 0.006 |
| 2 | 0.909 | 0.013 |
| 3 | 0.860 | 0.003 |
| 4 | 1.111 | 0.005 |

Residuals collapsed to −0.001 to +0.013 bits. The decomposition identity now holds tightly
across all seeds. F1 regenerated with corrected data.

---

### Fix 3 — E39: listener keeps RL gradients in Phase 2

Phase 2 previously froze the listener by zeroing its gradients after `optimizer.step()`.
This caused SR to collapse whenever the speaker shifted its z distribution during Phase 2,
because the listener could not adapt to the new message distribution.

**Root cause confirmed (2026-05-06):** First batch of E42–E46 (5 experiments × 5 seeds = 25
runs) was launched on 2026-05-05 with the corrected dither loss but without the listener fix.
Phase 2 triggered at update ~22–54 in all 25 seeds. SR collapsed from ~1.0 to ~5% within
700 updates in every case. All 25 runs were deleted.

**Fix applied in `trainer.py`:**
- Phase 2 `total_loss` now includes RL losses (`actor_loss + critic_loss`), so the listener
  receives its usual PPO gradient.
- Channel-parameter gradients are zeroed *after* `optimizer.step()` — only the channel encoder
  is frozen, not the listener or critic.

**Test updated:** `test_phase2_zeros_non_speaker_grads` → `test_phase2_listener_keeps_adapting`.
Assertion inverted: listener weights MUST change after a Phase 2 update. All 186 tests pass.

---

### Fix 4 — Repository path audit

Full audit of all Python analysis scripts and docs confirmed no stale `runs/toyproblem/` or
`results/toyproblem/` references in Python source. `docs/README.md` contained stale paths in
the CLI examples table; all corrected to canonical `runs/` and `results/` paths.
`post_hoc_coding.py` was also updated to infer `exp_name` and `seed` from the checkpoint path
and write directly to the canonical `results/post_hoc/<exp_name>/seed_<seed>.json` — no manual
copy step required.

---

### E42–E46 second batch launched (2026-05-06)

Re-launched all 5 experiments × 5 seeds with both fixes applied (corrected dither loss + E39).
All are RUNNING. Expected outcome: frac → 0.5 per dimension; H(m|goal) → 0; decomposition
residual < 0.3 bits; SR stable ≥ 0.98.

---

### E37/E38 — z_dim sweep; F15 generated (2026-05-06)

Added `ZDIM_SWEEP` stage to `run_sc_experiments.py` (E37: z_dim=1, E38: z_dim=2; same
config as sc_posthoc_mag). Both completed in ~15 minutes on CPU.

Post_hoc_coding.py run on all 10 checkpoints. Key results:

| z_dim | TC (mean) | H_joint (mean) | SR (mean) |
|---|---|---|---|
| 1 | 0.000 | 1.826 | 0.966 |
| 2 | 1.752 | 3.501 | 0.940 |
| 3 | 3.281 | 4.751 | 1.000 |

TC = 0 trivially at z_dim=1 (no cross-dim correlation possible). TC grows with dimensionality
as expected: each additional dimension allows the policy to create correlated clusters.

F15 generated as two-panel figure: TC vs z_dim and H_joint/H_factored vs z_dim with H(G)
reference. Saved to `results/figures/F15_tc_vs_zdim.{pdf,png}`.

---

### Figures generated on 2026-05-06

| Figure | File |
|---|---|
| F1 (regenerated) | `results/figures/F1_rate_decomposition.{pdf,png}` |
| F2 | `results/figures/F2_shannon_gap.{pdf,png}` |
| F5 | `results/figures/F5_pareto_frontier.{pdf,png}` |
| F13 | `results/figures/F13_three_way_comparison.{pdf,png}` |
| F15 | `results/figures/F15_tc_vs_zdim.{pdf,png}` |
| F20 | `results/figures/F20_three_way_pareto.{pdf,png}` |

---

## Open threads (as of 2026-05-05)

| Thread | Priority | Blocking |
|---|---|---|
| Re-run E44–E46 (sc_twophase with corrected dither loss) | High | F16, C6 restatement |
| Re-run E42–E43 (geom_dither_only / geom_both corrected) | High | F6, F7, C11 |
| Phase 2 listener gradient fix (E39) | High | F16 validity, future Phase 2 runs |
| z_dim sweep (E37, E38) | Medium | F15, OQ2 |
| Investigate source_coding_rate_loss secondary bin for f < 0.5 (OQ5) | Medium | P1 unbiasedness claim |
| Increase post_hoc z-samples to 1000 | Low | C4 residual, OQ4 |
| DLM comparison (E41) | Low | F8–F12 |

---

## 2026-05-05 — Post-hoc on geom_* checkpoints; dither loss formula bug discovered and fixed

**Context:** E40 (post_hoc analysis on geom_* checkpoints) was executed. All 4 geometry
experiments × 5 seeds = 20 runs completed. Results copied to `results/post_hoc/geom_*/`.

**What was done:**
- Ran `post_hoc_coding.py` on `geom_no_anchor`, `geom_mag_only`, `geom_dither_only`,
  `geom_both` (δ=1.0, z_dim=3, CPU, 50K messages each).
- Compared decomposition residuals across configs.

**Key finding — dither loss formula error:**

The decomposition residual was ~2.7–3.3 bits for geom_dither_only and geom_both (which
used `lambda_dither > 0`), vs 0.18–0.59 bits for geom_mag_only and geom_no_anchor
(which did not). Investigation revealed the root cause: `dither_channel_stats` and
`dither_channel_loss` computed H_binary(frac) when the correct formula for the floor
SD channel is H_binary(|frac − 0.5|).

Simulation (100K samples) confirmed:

| frac | H_binary(frac) [old code] | H_binary(\|frac−0.5\|) [correct] |
|---|---|---|
| 0.01 | 0.081 bits | 1.000 bits |
| 0.25 | 0.811 bits | 0.811 bits |
| 0.50 | 1.000 bits | 0.000 bits |
| 0.99 | 0.081 bits | 1.000 bits |

The erroneous loss pushed frac → 0 (bin boundaries = maximum noise). The corrected
loss pushes frac → 0.5 (bin centres = minimum noise for the floor quantiser).

**Impact:**
- E18 (geom_dither_only), E19 (geom_both): INVALIDATED. The dither loss made noise
  worse, not better. H(m|goal) was ~3.1 bits (maximum) not ~0.37 bits as reported.
- E12–E14 (sc_twophase): REINTERPRETED. The true_bits reduction came from a magnitude
  side-effect (pushing z toward integer multiples of δ = slightly smaller |z|), not
  from reducing dither noise. H(m) was unchanged throughout.
- geom_no_anchor and geom_mag_only: valid. Residuals 0.59 and 0.18 bits confirm
  the rate decomposition identity holds for configs without the dither loss.

**Corrected formula:**
  H(m_k | z_k) = H_binary(|frac(z_k/δ) − 0.5|)
  dH/dz_k = log₂((1−g)/g) · sign(frac − 0.5) / δ   where g = |frac − 0.5|

The corrected loss is compatible with the magnitude anchor: their joint minimum is
z_k ≈ δ/2 per dimension (m = 0 deterministically, |z| = δ/2).

**Fix applied:** `dither_channel_loss` and `dither_channel_stats` in `source_coding.py`
updated. No changes to `channels.py` (floor quantisation is correct and paper-consistent).

**Also noted (OQ5):** `source_coding_rate_loss` uses m_hi = floor(z/δ)+1 for all f, but
the correct secondary bin is floor(z/δ)−1 for f < 0.5. Direction is likely still correct
for monotone rate functions, but magnitude is wrong. Needs separate investigation.

**Next steps:** Re-run E42–E46 with corrected loss. Run post_hoc on E42–E43 to verify
residuals < 0.3 bits and H(m|goal) → 0.

---

## 2026-05-12 — Pillar design docs; P1 experiments complete; P4 running; git setup

**Context:** With Pillar 2 closed (E01–E52 done, C1–C16 written), the session focused on
designing and launching the remaining three pillars (P1, P4, P3) and their integration
(ALL_PILLARS), followed by repository housekeeping before paper writing begins.

---

### Pillar design docs created

All four v2 pillar docs written and committed to `feat/ToyProblem` (merged into
`Dev_ToyProblem` on 2026-05-12):

| Doc | Coverage |
|---|---|
| `docs/pillars/PILLAR_P1_v2.md` | Per-channel δ — design, E53–E56 configs, hypotheses, cross-pillar interactions |
| `docs/pillars/PILLAR_P2_v2.md` | Phase 2 dither — updated §17 results, E44–E52 complete |
| `docs/pillars/PILLAR_P3_v2.md` | NSD deployment consistency — TPDF theory, eval_deploy design |
| `docs/pillars/PILLAR_P4_v2.md` | Rao-Blackwell gradient — RB formula derivation, E57–E61 plan |
| `docs/pillars/PILLAR_ALL_v2.md` | All-Pillars integration — SD (E64) and NSD (E65) variants, gate conditions |

v1 stubs (PILLAR_P1.md, PILLAR_P2.md, PILLAR_P3.md, PILLAR_P4.md) deleted.

---

### LARGE_SCALE_EXTENSION.md created

`docs/LARGE_SCALE_EXTENSION.md` maps each toy-problem design decision to its scale
assumption. The document distinguishes:
- **Assumptions that hold at scale:** Schuchman/NSD theorems (channel-level, independent
  of task structure), dither loss formula, histogram estimator mechanics.
- **Assumptions that break at scale:** Role separation (at scale each agent both sends
  and receives using shared parameters θ), gradient isolation (RB detach trick severs
  inter-agent credit assignment), Phase 2 gradient zeroing (impossible with shared params).

**Key section added after user feedback (§0.5 Role-Collapse Problem):** At scale, each
agent has one shared policy θ — there is no separate speaker θ_S and listener θ_L. This
has four consequences documented in §0.5.3:
- (A) Comms loss bleeds into action head via shared parameters
- (B) RB detach trick breaks inter-agent credit assignment
- (C) Phase 2 gradient zeroing impossible
- (D) N(N-1) gradient paths amplify but entangle the message encoder

Alternatives proposed: critic-based RB (V(s) instead of π for probe), additive RB proxy
(alongside PPO, not replacing STE), λ_rl < 1 in Phase 2 instead of gradient zeroing.

---

### E53–E56 launched and completed (P1_DELTA stage)

All 20 P1 runs (4 configs × 5 seeds) completed with exit=0:

| Experiment | Config | Seeds done |
|---|---|---|
| E53 | `p1_global_delta` — global δ, STE, Phase 1+2 | 5/5 |
| E54 | `p1_perchannel_delta` — learned per-channel δ | 5/5 |
| E55 | `p1_perchannel_p2_5e-3` — per-channel δ + Phase 2 | 5/5 |
| E56 | `p1_heuristic_delta` — heuristic δ ∝ 1/H(m_k), frozen at Phase 2 onset | 5/5 |

Post-hoc coding and aggregate_results.py not yet run — H_joint and per-dim δ_k values
for E53–E56 are pending.

---

### E57–E60 launched (P4_RB stage, partially complete)

15 P4 runs (3 configs × 5 seeds). Status at end of session:

| Experiment | Config | Seeds done |
|---|---|---|
| E57 | `p4_ste_baseline` — STE, no RB | 4/5 |
| E58 | `p4_rb_joint` — RB-joint mode | 0/5 (running) |
| E60 | `p4_rb_joint_p2_5e-3` — RB-joint + Phase 2 | 0/5 (queued) |

Launcher re-started on 2026-05-12; dry-run detects completed seeds automatically.
E59 (p4_rb_per_dim) not yet in runner — requires `rb_mode="per_dim"` support to be
verified and added separately.

---

### Repository and git housekeeping

- Branch incident: user switched from `feat/ToyProblem` to `Dev_ToyProblem`, reverting
  all uncommitted changes. Recovery: `git merge feat/ToyProblem` restored all 50 changed
  files (17,397 insertions) via a clean merge commit (`b96aa61`).
- `.gitignore` corrected: removed `*.md`, `*.csv`, `*.pdf`, `*.txt` (too broad); added
  `*.pt`, `*.pth`, `scripts/pillar_experiments_*.log`; added `**/results/` to exclude
  all generated outputs. Repo now tracks only source code and markdown docs.
- Previously tracked data files (`results/figures/*.pdf`, `results/post_hoc/*.json`,
  `results/aggregated/summary.csv`) removed from git index via `git rm --cached`.

---

### Open threads (as of 2026-05-12)

| Thread | Priority | Status | Blocking |
|---|---|---|---|
| E57 seed 4 completion | High | Running | P4 validation |
| E58 (p4_rb_joint) 5 seeds | High | Running | P4 primary claim |
| E60 (p4_rb_joint_p2_5e-3) 5 seeds | High | Queued | P4+P2 combo |
| post_hoc_coding.py on E53–E60 | High | PENDING (after runs finish) | P1+P4 conclusions |
| aggregate_results.py on E53–E60 | High | PENDING | summary.csv update |
| P1 conclusion (C17): does δ_k differentiate? | Medium | PENDING post_hoc | Informs E64/E65 config |
| P4 conclusion (C18): RB vs STE speed? | Medium | PENDING E58 done | Informs E64/E65 config |
| eval_deploy.py for P3 (E62/E63) | Medium | NOT STARTED | E62/E63, F29 |
| E59 (p4_rb_per_dim) add to runner | Low | PLANNED | completeness |
| E61 (p4_rb_p1_joint) conditional run | Low | PLANNED | P1×P4 interaction |
| ALL_PILLARS (E64/E65) | Low | BLOCKED on P1+P4 gate | headline claim |
| Paper §3 Method + §4 Toy (E01–E52) | Medium | Ready to start | — |
