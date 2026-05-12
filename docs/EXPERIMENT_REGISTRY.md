# Experiment Registry

Tracks every training run, its status, the analyses that depend on it, and which
paper figures it feeds. Use this document to determine what needs to be re-run when
a configuration changes.

**Getting up to speed?** Start with `docs/CONCLUSIONS.md` instead — it has a
recommended reading-order guide at the top. This file is a dependency tracker,
not a scientific summary.

**How to use:**
- Before re-running an experiment, check its `Invalidates` column. Everything listed
  there must be regenerated.
- Before adding a new experiment, add a row here first with status `PLANNED`.
- After a run completes, update status to `DONE` and run `aggregate_results.py`.

**Aggregation command** (run from repo root after any change):
```
python -m onpolicy.envs.toyproblem.analysis.aggregate_results --verbose
```

**Status codes:** `DONE` | `PENDING` | `PLANNED` | `BLOCKED` | `SUPERSEDED` | `INVALIDATED` | `REINTERPRETED`

- `INVALIDATED` — run completed but results are wrong due to a code bug; do not use for paper claims
- `REINTERPRETED` — run completed and data is valid, but the mechanism interpretation was incorrect; re-run supersedes it

---

## Experiment name glossary

Every experiment name encodes a configuration in shorthand. Use this table to decode any name
without having to read the full config.

### Prefix tokens

| Token | Meaning |
|---|---|
| `no_comms` | No communication penalty at all — establishes the unconstrained SR=1 ceiling |
| `mag` | Magnitude penalty only: L_mag = λ · ‖z‖ per message dimension |
| `sc` | Source coding active (histogram-based rate loss + score-function gradient) |
| `geom` | Geometry experiment — same training configs as sc/mag but the checkpoint is analysed for z-space structure (clusters, Gram matrix, per-goal geometry) |
| `pareto` | Pareto frontier sweep — same config as the `mag` or `sc` series but run at many λ values to trace the SR vs bits curve |
| `live` | Three-way comparison set (live_A / live_B / live_C) — compares post-hoc SC, live SC, and live dither side-by-side |
| `fix` | DLM fix-ladder experiment (P2-FIX stage) — tests engineering fixes to the Deep Latent Model estimator |

### Suffix / modifier tokens

| Token | Meaning |
|---|---|
| `lam<value>` | Penalty weight λ, e.g. `lam1e-3` = λ=0.001 |
| `posthoc` | Source coding applied post-hoc (after convergence) — no live gradient during training |
| `live_entropy` | Live score-function SC gradient applied throughout training |
| `live_both` | Both magnitude penalty AND live SC gradient during training |
| `twophase` | Two-phase training: Phase 1 = magnitude until SR=1, Phase 2 = dither/entropy loss |
| `dither<value>` | Dither loss weight λ_dither in Phase 2 |
| `no_anchor` | No magnitude penalty — z is unconstrained; used to study what happens without any geometric structure |
| `mag_only` | Magnitude penalty only, no dither — isolates the effect of the magnitude anchor on z-space geometry |
| `dither_only` | Dither loss only, no magnitude — isolates the effect of the corrected dither loss |
| `both` | Magnitude + dither together |
| `_v2` | Re-run with corrected `dither_channel_loss` formula (H_binary(|frac−0.5|)) + E39 listener fix; supersedes the non-v2 version |
| `zdim1` / `zdim2` | z-vector dimension set to 1 or 2 (default is 3); used to study how TC scales with dimensionality |

### DLM fix-ladder configs (E41)

| Name | What it tests |
|---|---|
| `fix_baseline` | Standard DLM with q_φ(m) trained alongside the policy; the reference point for all DLM comparisons |
| `fix_B1_gate` | Adds a backward gate: the DLM rate gradient only fires when q_φ is well-fitted (NLL gap < 2 bits); also adds a q_φ warm-up period before RL begins |
| `fix_B2_gate_lambda` | Same as B1 but with a higher penalty weight λ=0.01, testing whether stronger compression pressure helps when the gate is active |
| `fix_B3_ema` | Replaces the live q_φ with an exponential-moving-average shadow of q_φ for the backward pass; intended to break the circular gradient by using a lagged prior |
| `fix_B4_twophase` | Two-phase design with DLM: Phase 1 uses magnitude only (keeps the distribution simple so q_φ fits well), Phase 2 fires the DLM backward into the speaker |

### Full experiment name examples

| Experiment name | Plain English |
|---|---|
| `no_comms` | Train with no penalty; SR=1 at 8.74 bits — the unconstrained ceiling |
| `mag_lam1e-3` | Train with magnitude penalty λ=0.001; best SR=1 operating point (6.52 bits) |
| `sc_posthoc_mag` | Train with magnitude penalty to convergence, then analyse with post-hoc source coding (histogram decomposition); the primary rate-decomposition reference |
| `sc_live_entropy` | Train with live score-function SC gradient throughout; demonstrates z blowup failure mode |
| `sc_live_both` | Train with magnitude + live SC simultaneously |
| `sc_twophase_dither1e-3` | Two-phase: Phase 1 magnitude, Phase 2 dither at λ=0.001 (original, wrong formula — see E12–E14) |
| `sc_twophase_dither1e-3_v2` | Same two-phase design but with corrected dither formula and listener RL fix (E44–E46) |
| `geom_no_anchor` | No penalty; same as no_comms but run as part of the geometry comparison batch |
| `geom_mag_only` | Magnitude penalty only; establishes the z-space geometry baseline for Phase 1 |
| `geom_dither_only_v2` | Corrected dither loss, no magnitude anchor; isolates the dither effect on geometry and TC |
| `geom_both_v2` | Magnitude + corrected dither together; best geometry/TC result |
| `pareto_mag_lam1e-3` | Pareto curve data point at λ=0.001 with magnitude penalty |
| `pareto_sc_lam1e-3` | Pareto curve data point at λ=0.001 with source coding |
| `live_A_posthoc` | Three-way comparison: post-hoc SC (same config as sc_posthoc_mag) |
| `live_B_live_sc` | Three-way comparison: live SC gradient active |
| `live_C_live_dither` | Three-way comparison: live dither active |
| `sc_posthoc_mag_zdim1` | Post-hoc SC at z_dim=1; confirms TC=0 |
| `sc_posthoc_mag_zdim2` | Post-hoc SC at z_dim=2; TC=1.75 bits |
| `fix_baseline` | DLM baseline; reference for the fix-ladder sweep |
| `fix_B4_twophase` | DLM with two-phase trick; achieves low estimation gap but SR collapses in Phase 2 |

---

## Group 1 — Baseline (no compression ceiling + magnitude Pareto)

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E01 | `no_comms` | DONE | 5 | F2, F5 | summary.csv, F2, F5 |
| E02 | `mag_lam1e-5` | DONE | 5 | F2, F5 | summary.csv, F2, F5 |
| E03 | `mag_lam1e-4` | DONE | 5 | F2, F5 | summary.csv, F2, F5 |
| E04 | `mag_lam5e-4` | DONE | 5 | F2, F5 | summary.csv, F2, F5 |
| E05 | `mag_lam1e-3` | DONE | 5 | F2, F5 | summary.csv, F2, F5 |
| E06 | `mag_lam4e-3` | DONE | 5 | F2, F5 | summary.csv, F2, F5 |
| E07 | `mag_lam1e-2` | DONE | 5 | F2, F5 | summary.csv, F2, F5 |
| E08 | `mag_lam3e-2` | DONE | 5 | F2, F5 | summary.csv, F2, F5 |

**Note:** E01 and E04 are the primary anchors (ceiling and standard λ). All others
establish the shape of the magnitude Pareto curve.

---

## Group 2 — SC Main (post-hoc vs live source coding)

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E09 | `sc_posthoc_mag` | DONE | 5 | F1, F2, F3, F4, F7, F13 | post_hoc decomposition, message_geometry, F1–F4, F7, F13 |
| E10 | `sc_live_entropy` | DONE | 5 | F2, F13 | summary.csv, F2, F13 |
| E11 | `sc_live_both` | DONE | 5 | F13 | summary.csv, F13 |

**Downstream analyses for E09** (must be re-run if E09 changes):
- `post_hoc_coding.py` → `results/post_hoc/sc_posthoc_mag/`
- `aggregate_results.py` → `results/post_hoc/sc_posthoc_mag/aggregate.json`
- `message_geometry.py` (Panel A/B/D for Phase 1 column)
- F3 gradient alignment analysis (not yet implemented)
- F4 gradient direction analysis (not yet implemented)

---

## Group 3 — Two-Phase Dither (Phase 2 L_dither)

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E12 | `sc_twophase_dither1e-4` | REINTERPRETED | 5 | F16 | summary.csv, F16, message_geometry (col 3) |
| E13 | `sc_twophase_dither5e-4` | REINTERPRETED | 5 | F16 | summary.csv, F16 |
| E14 | `sc_twophase_dither1e-3` | REINTERPRETED | 5 | F16 | summary.csv, F16, message_geometry (col 2) |
| E15 | `sc_twophase_dither1e-3` (fixed listener) | SUPERSEDED | 5 | — | Superseded by E46 (`sc_twophase_dither1e-3_v2`), which includes both the corrected dither formula and E39 listener fix |

**REINTERPRETED (2026-05-05):** E12–E14 used `dither_channel_loss` with the wrong formula
(H_binary(frac) instead of H_binary(|frac−0.5|)). The loss pushed frac → 0 (bin boundaries
= maximum noise), not toward 0.5 (bin centres = minimum noise). The observed true_bits
reduction was a magnitude side-effect of pushing z toward integer multiples of δ, not
genuine H(m|goal) reduction. hist_H was unchanged (7.38–7.51), confirming H(m) was not
reduced. Must be re-run with the corrected loss (E44–E46).

**Status code:** `REINTERPRETED` — runs are valid; mechanism interpretation was wrong.
Re-runs with corrected loss will supersede these.

---

## Group 4 — Geometry (magnitude anchor geometric role)

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E16 | `geom_no_anchor` | DONE | 5 | F6, F7 | summary.csv, F6, F7 |
| E17 | `geom_mag_only` | DONE | 5 | F6, F7 | summary.csv, F6, F7, post_hoc (if run) |
| E18 | `geom_dither_only` | INVALIDATED | 5 | — | summary.csv, F6, F7 |
| E19 | `geom_both` | INVALIDATED | 5 | — | summary.csv, F6, F7 |

**INVALIDATED (2026-05-05):** E18 and E19 used `dither_channel_loss` with the wrong
formula. The erroneous loss maximised dither noise (frac → 0) rather than minimising it
(frac → 0.5). Post-hoc analysis confirmed empirical H(m|goal) ≈ 3.1 bits (maximum noise)
vs the code's reported 0.37 bits (formula error). The decomposition residuals for these
experiments (2.74 and 3.32 bits) are entirely explained by this discrepancy.
Re-run as E42 and E43 with the corrected loss.

**Post-hoc analysis completed (2026-05-05):** `post_hoc_coding.py` was run on all
E16–E19 checkpoints (5 seeds each). Results in `results/post_hoc/geom_*/`.
E16 and E17 results are valid (residuals 0.59 and 0.18 bits).
E18 and E19 results reflect the formula error and should not be used for F6/F7.

---

## Group 5 — Live coding comparison

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E20 | `live_A_posthoc` | DONE | 5 | F13, F20 | summary.csv, F13, F20 |
| E21 | `live_B_live_sc` | DONE | 5 | F13, F14, F19, F20 | summary.csv, F13, F14, F19, F20 |
| E22 | `live_C_live_dither` | DONE | 5 | F13, F20 | summary.csv, F13, F20 |

**Note:** E20 (`live_A_posthoc`) is identical to E09 (`sc_posthoc_mag`) — same config,
run twice. Results match exactly (SR=1.0, true_bits=7.04).

---

## Group 6 — Pareto frontier

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E23 | `pareto_mag_lam1e-5` | DONE | 5 | F5 | summary.csv, F5 |
| E24 | `pareto_mag_lam1e-4` | DONE | 5 | F5 | summary.csv, F5 |
| E25 | `pareto_mag_lam5e-4` | DONE | 5 | F5 | summary.csv, F5 |
| E26 | `pareto_mag_lam1e-3` | DONE | 5 | F5 | summary.csv, F5 |
| E27 | `pareto_mag_lam4e-3` | DONE | 5 | F5 | summary.csv, F5 |
| E28 | `pareto_mag_lam1e-2` | DONE | 5 | F5 | summary.csv, F5 |
| E29 | `pareto_mag_lam3e-2` | DONE | 5 | F5 | summary.csv, F5 |
| E30 | `pareto_sc_lam1e-5` | DONE | 5 | F5 | summary.csv, F5 |
| E31 | `pareto_sc_lam1e-4` | DONE | 5 | F5 | summary.csv, F5 |
| E32 | `pareto_sc_lam5e-4` | DONE | 5 | F5 | summary.csv, F5 |
| E33 | `pareto_sc_lam1e-3` | DONE | 5 | F5 | summary.csv, F5 |
| E34 | `pareto_sc_lam4e-3` | DONE | 5 | F5 | summary.csv, F5 |
| E35 | `pareto_sc_lam1e-2` | DONE | 5 | F5 | summary.csv, F5 |
| E36 | `pareto_sc_lam3e-2` | DONE | 5 | F5 | summary.csv, F5 |

**Important:** F5 must use `hist_H_empirical` as x-axis for SC runs and `true_bits_per_msg`
for magnitude runs. These measure different things; the figure must note this explicitly.

---

## Group 7 — Planned / future experiments

| ID | Experiment | Status | Blocks | Purpose |
|---|---|---|---|---|
| E37 | `sc_posthoc_mag_zdim1` | DONE (2026-05-06) | F15 | TC≈0 confirmed; SR=0.966±0.010 |
| E38 | `sc_posthoc_mag_zdim2` | DONE (2026-05-06) | F15 | TC≈1.7 bits; SR=0.940±0.134 |
| E39 | Phase 2 with fixed listener gradients | DONE (2026-05-06) — implemented in trainer.py; listener keeps RL gradients in Phase 2 | F16 | Fix SR degradation; E42–E46 re-run with this fix |
| E40 | `geom_*` post_hoc analysis | DONE (2026-05-05) | F6, F7 | TC and H_dither per geometry config |
| E41 | DLM comparison (P2-FIX) | DONE (2026-05-08) | F8, F10, F12 | fix_baseline qphi_gap=9.71 bits; B1–B3 gap grows to 18–22 bits (circular gradient confirmed); fix_B4_twophase qphi_gap=0.046 bits but SR=0.536 (warm-start failure). Histogram=0.004 bits, SR=1.000. |

---

## Group 8 — Re-runs with corrected dither loss (post formula fix 2026-05-05)

**First batch INVALIDATED (2026-05-06):** E42–E46 were first launched 2026-05-05 with the
corrected `dither_channel_loss` but without the E39 listener fix. Phase 2 triggered at ~update
22–54 in all seeds and SR collapsed immediately (1.0 → 0.05 within 700 updates) because the
listener was frozen while the speaker shifted its z distribution. All 25 runs were deleted.

**Second batch RUNNING (2026-05-06):** Re-launched with E39 fix (listener keeps RL gradients
in Phase 2). Stage: `DITHER_CORRECTED` in `run_sc_experiments.py`.

| ID | Experiment | Status | Seeds | Supersedes | Purpose |
|---|---|---|---|---|---|
| E42 | `geom_dither_only_v2` (corrected loss + E39) | DONE (2026-05-07) | 5 | E18 | SR=1.000, hist_H=5.37, H_dither=0.78, H_joint=2.60 |
| E43 | `geom_both_v2` (corrected loss + E39) | DONE (2026-05-07) | 5 | E19 | SR=1.000, hist_H=4.59, H_dither=0.57, H_joint=2.39 |
| E44 | `sc_twophase_dither1e-4_v2` (corrected loss + E39) | DONE (2026-05-07) | 5 | E12 | SR=1.000, hist_H=4.88 — no collapse (E39 fix confirmed) |
| E45 | `sc_twophase_dither5e-4_v2` (corrected loss + E39) | DONE (2026-05-07) | 5 | E13 | SR=1.000, hist_H=4.59 |
| E46 | `sc_twophase_dither1e-3_v2` (corrected loss + E39) | DONE (2026-05-07) | 5 | E14 | SR=1.000, hist_H=4.65; Phase 2 triggers at update 54 |

**Expected outcome for E42–E43:** frac converges to ~0.5 per dimension; H(m|goal) → 0;
decomposition residual < 0.3 bits; the dither loss and magnitude anchor cooperate.

**Expected outcome for E44–E46:** True H(m|goal) reduction (not just true_bits); hist_H
decreases below the Phase 1 level; the two-phase design achieves lower H(m), not lower
log2(|z|/δ+1) only.

---

## Group 9 — §11.8 Channel interaction ablation (SD vs NSD)

**Purpose:** Confirm the histogram estimator and Phase 2 dither loss generalise
automatically to the NSD channel (q_hist adapts to actual m samples regardless of
channel type).  Matches E09 and E45 configs exactly, with only `--channel nsd` changed.

Post-hoc coding must be run on E47 after training to measure H_joint.

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E47 | `nsd_posthoc_mag` | DONE (2026-05-11) | 5 | §11.8 table, C16 | summary.csv |
| E48 | `nsd_twophase_dither5e-4_v2` | DONE (2026-05-11) | 5 | §11.8 table, C16 | summary.csv |

**Results (2026-05-11):** SR=1.000 on both. E47 H_joint=4.137±0.165 (SD ref: 4.156±0.344).
E48 H_joint=2.413±0.141 (SD ref E45: 2.385±0.174). Difference <0.03 bits on both — channel agnosticism confirmed.

---

## Group 10 — Phase 2 Pareto curve extension

**Purpose:** Extend the λ_dither sweep beyond E44–E46 ({1e-4, 5e-4, 1e-3}) to trace
the full SR vs H_joint rate-distortion frontier for Phase 2.  Lower λ values show the
curve extends toward H(G)=1.81 bits at SR=1; higher λ values reveal the SR tradeoff.
Together with E44–E46, these 7 points enable a Phase 2 Pareto curve on F5.

Post-hoc coding must be run on all E49–E52 checkpoints after training.

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E49 | `sc_twophase_dither5e-5_v2` | DONE (2026-05-11) | 5 | F5 (Phase 2 curve) | summary.csv, F5 |
| E50 | `sc_twophase_dither2e-4_v2` | DONE (2026-05-11) | 5 | F5 (Phase 2 curve) | summary.csv, F5 |
| E51 | `sc_twophase_dither2e-3_v2` | DONE (2026-05-11) | 5 | F5 (Phase 2 curve) | summary.csv, F5 |
| E52 | `sc_twophase_dither5e-3_v2` | DONE (2026-05-11) | 5 | F5 (Phase 2 curve) | summary.csv, F5 |

**Results (2026-05-11):** All 4 new configs + E44–E46 = 7-point λ sweep, all SR=1.000.
H_joint monotone for λ≥5e-4; low-λ cluster (5e-5, 1e-4, 2e-4) near 2.5 bits within noise.
Best: E52 (λ=5e-3) H_joint=2.094±0.047 bits — 0.28 bits above H(G)=1.813. No SR degradation
at any λ tested; Phase 2 is more robust than anticipated. See C10 for full table, C16 for NSD.

---

## Group 11 — Pillar 1: Per-Channel δ (E53–E56)

**Purpose:** Test whether letting each message dimension k learn its own quantisation
step δ_k (via softplus(α_k)) allows asymmetric bit allocation across dimensions.
E56 provides a heuristic baseline (δ_k ∝ 1/H(m_k), set at Phase 2 onset, then frozen).

Post-hoc coding must be run on all checkpoints after training to measure per-dim H(m_k)
and whether δ_k values differentiate across dimensions.

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E53 | `p1_global_delta` | DONE (2026-05-12) | 5 | F22–F24 | summary.csv, F22–F24 |
| E54 | `p1_perchannel_delta` | DONE (2026-05-12) | 5 | F22–F24 | summary.csv, F22–F24 |
| E55 | `p1_perchannel_p2_5e-3` | DONE (2026-05-12) | 5 | F22–F24 | summary.csv, F22–F24 |
| E56 | `p1_heuristic_delta` | DONE (2026-05-12) | 5 | F22–F24 | summary.csv, F22–F24 |

**Primary comparison:** E54 (learned) vs E53 (global δ) on H_joint and per-dim δ_k spread.
**Baseline:** E56 (heuristic) is not a paper contribution — it contextualises how much
the optimizer does vs a rule-based assignment.
**Post-hoc pending:** aggregate_results.py and post_hoc_coding.py on all E53–E56 checkpoints.

---

## Group 12 — Pillar 4: Rao-Blackwell Gradient (E57–E61)

**Purpose:** Test whether replacing the straight-through estimator (STE) speaker task
gradient with the Rao-Blackwell finite-difference (g_RB_k = [L_hi − L_lo] / δ_k)
speeds Phase 1 convergence and/or improves final H_joint.

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E57 | `p4_ste_baseline` | PENDING (4/5 done) | 5 | F26–F28 | summary.csv, F26–F28 |
| E58 | `p4_rb_joint` | PENDING (running) | 5 | F26–F28 | summary.csv, F26–F28 |
| E59 | `p4_rb_per_dim` | PLANNED | 5 | F26–F28 | summary.csv, F26–F28 |
| E60 | `p4_rb_joint_p2_5e-3` | PENDING (running) | 5 | F28 | summary.csv, F28 |
| E61 | `p4_rb_p1_joint` | PLANNED (conditional) | 5 | F28 | summary.csv, F28 |

**Primary comparison:** E58 (RB-joint) vs E57 (STE) on steps-to-SR99.
**Conditional:** E61 (RB + per-channel δ) only if P1×P4 interaction is non-additive.
**E59** (per-dim RB mode) not yet added to runner; requires `rb_mode="per_dim"` support.

---

## Group 13 — Pillar 3: NSD Deployment Consistency (E62–E63)

**Purpose:** Measure the train/deploy SR gap for the SD and NSD channels.
NSD (TPDF dither) should have zero deploy gap by construction (Schuchman theorem).
SD has distributional consistency but not sample consistency.

Requires `eval_deploy.py` (checkpoint evaluation script, not yet implemented).

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E62 | `p3_sd_deploy_eval` | PLANNED | 5 | F29 | F29 |
| E63 | `p3_nsd_deploy_eval` | PLANNED | 5 | F29 | F29 |

**Blocked on:** `eval_deploy.py` implementation. Uses E09 (SD) and E47 (NSD) checkpoints;
no new training runs required. The deploy eval replaces dither with its expected value
(zero for SD; zero for NSD) and measures SR degradation.

---

## Group 14 — ALL_PILLARS: Full Integration (E64–E65)

**Purpose:** Combine all four pillars (P1+P2+P4+P3) to test whether benefits compose.
This is the headline experimental claim of the paper.

| ID | Experiment | Status | Seeds | Figures | Invalidates if re-run |
|---|---|---|---|---|---|
| E64 | `all_pillars_sd` | BLOCKED | 5 | F-ALL | summary.csv, F-ALL |
| E65 | `all_pillars_nsd` | BLOCKED | 5 | F-ALL | summary.csv, F-ALL |

**Gate:** Do not run until E53+E54 validated (P1 conclusion) AND E57+E58 validated
(P4 conclusion). Override with `--override_gate` flag once gate passes.
**Config:** P1 (learn_delta) + P2 (use_source_coding, lambda_dither=5e-3) +
P4 (use_rb_gradient, rb_mode=joint) + P3 (channel=sd or nsd).

---

## Dependency graph (which analyses must be re-run after each experiment)

```
E09 (sc_posthoc_mag)
  └─→ post_hoc_coding.py
        └─→ results/post_hoc/sc_posthoc_mag/seed_N.json
              └─→ aggregate_results.py → aggregate.json
                    └─→ F1, F7 (decomposition bar chart)
  └─→ message_geometry.py (col 1)
  └─→ F3 gradient alignment analysis
  └─→ F4 gradient direction analysis

E14 (sc_twophase_dither1e-3)
  └─→ message_geometry.py (col 2)

E12 (sc_twophase_dither1e-4)
  └─→ message_geometry.py (col 3)

ANY experiment in Group 1, 2, 5, 6
  └─→ aggregate_results.py → summary.csv
        └─→ F2, F5, F13, F20 (need updated numbers)
```

---

## Checklist: adding a new experiment

1. Add a row to this registry with status `PLANNED`.
2. Add the config to `experiments/run_sc_experiments.py` or create a new runner.
3. Run the experiment via `scripts/run_p2_experiments.sh` or manually.
4. Update status to `DONE`.
5. Run `python -m onpolicy.envs.toyproblem.analysis.aggregate_results --verbose`.
6. If the experiment produces a checkpoint used by post_hoc_coding.py, run it — it
   writes directly to `results/post_hoc/<exp_name>/seed_<seed>.json`. Then re-run step 5.
7. Add a dated entry to `docs/JOURNAL.md`.
8. Update any affected conclusions in `docs/CONCLUSIONS.md`.
