# Conclusions

Living document. Updated after each experiment batch completes.  
Last updated: 2026-05-11. Source data: `results/aggregated/summary.csv`,
`results/post_hoc/sc_posthoc_mag/aggregate.json`.

Each conclusion states: the claim, the evidence, the confidence level, and any caveats.

---

## How to get up to speed (read in this order)

If you are returning after a break and need to re-orient, read these documents in order:

1. **This file (`docs/CONCLUSIONS.md`)** — 10–15 min. The most compressed source of
   truth. Each conclusion (C1–C13) states the claim, evidence, confidence, and any
   corrections. Reading C1–C13 in order gives the full scientific picture without
   looking at any raw results. If a finding was revised, the conclusion says so and
   points to the fix.

2. **`docs/EXPERIMENT_REGISTRY.md`** — 5 min. Tells you which experiments exist, their
   current status (`DONE` / `RUNNING` / `INVALIDATED` / `REINTERPRETED`), which figures
   they feed, and what must be re-run if they change. The status column is the fastest
   way to see what the current experiment state is.

3. **`on-policy/CLAUDE.md`** — 5 min. Directory layout, canonical output paths, and the
   figure-to-experiment mapping table (F1–F20). If you want to know where a specific
   figure's data comes from, this is the table to check. Also has workflow checklists
   for adding experiments and generating figures.

4. **`docs/JOURNAL.md` (recent entries only)** — 10 min. Chronological log. Jump to the
   `## Open threads` section near the top of the most recent dated entry and read
   backwards through only the recent entries until you have covered the current batch.
   Each entry records what was done, what was found, and any corrections. The Open
   threads table is the concise punch list of what is still pending.

5. **`docs/pillars/PILLAR_P2_v2.md`** — reference only, not required for orientation.
   §16 has the figure list; §17 has the full results table. Consult only when you need
   the theoretical motivation for a specific design decision.

---

## C1 — RL alone never reduces the communication rate

**Claim:** Without an explicit compression penalty, a converged MARL policy uses the
same number of bits regardless of how long it trains.

**Evidence:**  
`no_comms`: SR=1.000, true_bits=8.74±0.30 (5 seeds, 2M steps).  
`geom_no_anchor`: SR=1.000, true_bits=8.74±0.30 — identical to no_comms despite
the policy being trained in the geometry stage with different random seeds.

**Confidence:** High. Replicated across all 5 seeds with zero variance in SR and
near-zero variance in true_bits.

**Implication for paper:** The unconstrained ceiling is a well-defined quantity (8.74
bits for this environment). Any compression result must be compared to this baseline.

---

## C2 — The magnitude anchor is necessary and sufficient for Phase 1 compression

**Claim:** L_mag (penalise ‖z‖) reduces the communication rate from the unconstrained
ceiling to a controllable operating point, at zero SR cost up to λ ≈ 1e-3.

**Evidence:**  
- `mag_lam5e-4`: SR=1.000, true_bits=7.04±0.87 (−19% vs ceiling)  
- `mag_lam1e-3`: SR=1.000, true_bits=6.52±0.37 (−25% vs ceiling) ← best SR=1 point  
- `mag_lam4e-3`: SR=0.856±0.207, true_bits=3.49±1.69 ← Pareto knee; SR degrades  
- `mag_lam1e-2`: SR=0.558±0.017, true_bits=1.50±0.00 ← over-constrained

**Confidence:** High. Pareto curve is smooth and consistent across seeds.

**Caveat:** `true_bits` is the SD channel's z-magnitude-based cost, not H(m). The
magnitude anchor reduces `true_bits` but does not reduce H(m) (Shannon entropy of the
discrete message distribution). See C7 for the metric distinction.

---

## C3 — Total correlation TC(m) dominates the rate gap

**Claim:** The gap between per-dimension independent coding (H_factored) and joint coding
(H_joint) is TC(m) ≈ 3.28 bits — the largest single component of the rate decomposition.

**Evidence** (post_hoc on sc_posthoc_mag, 5 seeds, 50K messages each, corrected goal
sampling — updated 2026-05-08):
- H_factored = 7.44±0.44 bits
- H_joint = 4.16±0.34 bits
- TC = H_factored − H_joint = 3.28±0.22 bits
- **46.4 distinct joint tuples** (m_0, m_1, m_2) observed on average across 50K messages
- H(G) = 1.8128 bits (entropy of 6-goal non-uniform training distribution)

**Prior (stale) numbers:** H_factored=9.48, H_joint=4.75, TC=4.73. These came from
`collect_messages` cycling goals uniformly (pre-2026-05-06 fix). With uniform cycling,
all goals are weighted equally regardless of the non-uniform training distribution,
inflating H_factored and TC. After fixing goal sampling to use `_GOAL_PROBS`, the
decomposition residual collapsed from 0.77 mean to [−0.001, +0.013] bits (see C4).

**Confidence:** High. TC is consistent across seeds (std=0.22). The ordering
H_factored ≥ H_joint holds strictly for all 5 seeds (verified in F11).

**Implication:** Joint arithmetic coding reduces transmission cost from 7.44 → 4.16 bits
(−44%) for free, exploiting the implicit combinatorial structure the policy has learned.
The full bound hierarchy (F11): H_factored (7.44) ≥ H_joint (4.16) ≥ H(m|goal) (2.34)
≥ H(G) (1.81). Each level of receiver context provides a strictly tighter rate bound.

---

## C4 — The rate decomposition identity holds empirically

**Claim:** H_factored = H(G) + H(m|goal) + TC(m) + ε_estimator

**Evidence** (sc_posthoc_mag, corrected post_hoc run 2026-05-06):  
- All 5 seeds: residual ∈ [−0.001, +0.013] bits  
- ε_estimator ≈ −0.000 bits (histogram fit is excellent)

**Confidence:** High. The identity holds tightly across all seeds after fixing the goal
sampling distribution in `collect_messages` (2026-05-06 fix).

**Root cause of prior high residuals (0.77 mean):** `collect_messages` cycled goals
uniformly (weight 1/6 each), but `H_GOAL_BITS = 1.81 bits` is the non-uniform training
distribution entropy. The systematic offset was `log2(6) − 1.81 ≈ 0.77 bits` — matching
the observed mean residual exactly. Fix: sample goals with `torch.multinomial(_GOAL_PROBS)`.
Increasing z-samples from 100 → 1000 would be a no-op: the speaker is deterministic, so
100 identical goal vectors give 100 identical z values.

---

## C5 — Live source coding is ineffective and causes z-magnitude blowup

**Claim:** Applying the score function gradient during training does not usefully reduce
H(m); it only grows z magnitude.

**Evidence:**  
- `sc_live_entropy`: SR=1.000, true_bits=20.06±0.51, **hist_H=7.05±0.35**  
- `sc_posthoc_mag` (no live SC): SR=1.000, true_bits=7.04, hist_H=N/A (same level)  
- **F14 (2026-05-10) — gradient norm decomposition from E21 re-run:**
  - PPO speaker grad norm / SC speaker grad norm = **4–5× throughout training**
  - Ratio is stable: 4.39 early (first 20 updates), 4.96 late (last 20 updates)
  - PPO grad dominance never weakens — SC never gains leverage on the speaker
- **F4 (2026-05-07) — loss-scale ratio from E10:**
  - During early learning (SR < 0.5): |sc_loss|/|pg_loss| ≈ 0.09–0.13 (PPO leads ~8–10:1)
  - After convergence: |sc_loss|/|pg_loss| > 1 — SC loss is numerically larger than PPO loss

**Confidence:** High. Replicated across 5 seeds and two live SC configurations.

**Mechanism (revised 2026-05-10 with F14):** F14 and F4 measure different things:

- F4's loss-scale ratio (SC/PPO > 1 after convergence) is an artifact: the PPO task loss
  pg_loss → 0 as the policy converges, so any fixed SC loss looks large by comparison.
  It does not mean SC's gradient influence on the speaker is larger.

- F14's gradient-norm ratio (PPO/SC ≈ 5× throughout) is the direct measure of each
  loss's influence on the speaker weights. PPO never loses dominance, even after
  convergence. The advantage-weighted policy gradient remains large because episode
  return variance persists regardless of task performance.

- Together: PPO locks in the goal→tuple mapping during the SR-learning phase (first
  ~50 updates) and continues to reinforce it at every subsequent update. SC can only
  grow |z| (reducing true_bits by reaching lower integer bins), not remap which goal
  uses which bin. The hist_H metric stays flat at 7 bits throughout. The z grows
  because SC pushes toward lower-indexed bins while PPO maintains the goal→tuple
  structure — requiring ever-larger absolute values to satisfy both objectives.

**Note on true_H_offline (F19):** The offline H(m) estimate under correct _GOAL_PROBS
weighting decreases from ~6 → 2.4 bits as training progresses. This is NOT useful
compression — see C15.

---

## C6 — Phase 2 dither (E12–E14) reduced true_bits via a magnitude side-effect, not via H(m|goal)

**Claim (revised 2026-05-05):** The `dither_channel_loss` used in E12–E14 contained
a formula error: it computed H_binary(frac) when the correct H(m|z) for the floor
SD channel is H_binary(|frac − 0.5|).  The erroneous loss pushed frac → 0 (bin
boundaries), which is the maximum-noise locus, not minimum.  The observed true_bits
reduction came from the loss accidentally reducing |z| (pushing z toward integer
multiples of δ has a small magnitude-reduction side-effect), not from reducing dither
noise.  H(m) was unchanged (hist_H ≈ 7.38–7.51 across all Phase 2 runs), confirming
that H(m|goal) was not actually reduced.

**Evidence:**  
- `sc_twophase_dither1e-3`: SR=0.986±0.016, true_bits=5.14±0.10 (−27% vs Phase 1)  
- hist_H for both: 7.38–7.51 — unchanged → H(m|goal) + TC(m) not reduced  
- Post-hoc on geom_dither_only (same loss): empirical H(m|goal) ≈ 3.1 bits (maximum
  noise, matching frac → 0) vs code-reported H_dither = 0.37 bits (formula error)  
- Decomposition residual for geom_dither_only: 2.74 bits (entirely explained by formula error)

**Fix applied (2026-05-05):** `dither_channel_loss` and `dither_channel_stats` in
`source_coding.py` corrected to use H_binary(|frac − 0.5|).  The corrected loss pushes
frac → 0.5 (bin centres), the true minimum-noise locus for the floor quantiser.
E12–E14 must be re-run with the corrected loss (see E44–E46).

**Confidence:** High. The formula error is confirmed analytically and by simulation.
See C11 for the full derivation and implications.

---

## C7 — The two communication cost metrics measure different things

**Claim:** `true_bits_per_msg` (SD channel cost, z-magnitude-dependent) and
`hist_H_empirical` (Shannon entropy of p(m)) are not interchangeable and diverge in
live SC runs.

**Evidence:**  
- `sc_live_entropy`: true_bits=20.06 vs hist_H=7.05 — 3× divergence  
- `sc_twophase_dither1e-3`: true_bits=5.14 vs hist_H=7.51 — dither reduces true_bits
  without changing H(m)

**Implication for paper:**  
- The Pareto frontier (F5) must specify which metric is on each axis.  
- For information-theoretic claims: use hist_H_empirical (or H_joint from post_hoc).  
- For channel cost claims: use true_bits_per_msg.  
- The two-phase design is the only configuration that reduces both.

---

## C8 — Phase 2 stability is λ-sensitive; λ=1e-4 causes catastrophic collapse

**Claim:** At λ_dither = 1e-4, the dither gradient is too weak to converge frac
before listener degradation accumulates, causing SR collapse.

**Evidence:**  
- `sc_twophase_dither1e-4`: SR=0.047±0.045 — near-random policy  
- `sc_twophase_dither1e-3`: SR=0.986±0.016 — stable  
- message_geometry.py Panel D (col 3): goals share message tuples in collapsed policy

**Mechanism:** Phase 2 previously zeroed listener gradients. At small λ, Phase 2 runs for
~1M additional steps while frac converges slowly. During this time, transient bin-crossing
events as z slowly migrates destroy the goal-to-message mapping, and the listener could not
adapt because its gradients were zeroed. Confirmed empirically: first batch of E42–E46
(25 runs with frozen listener) collapsed SR from 1.0 to ~5% within 700 updates at Phase 2.

**Fix applied (E39, 2026-05-06):** `trainer.py` updated — Phase 2 now includes RL losses
in `total_loss`, so the listener keeps receiving PPO gradients. Only channel-parameter
gradients are zeroed after `optimizer.step()`. E42–E46 relaunched with this fix.

**Confidence:** High for the phenomenon; the mechanism is confirmed by the collapse of the
first batch and the fix that resolved it.

---

## C9 — The magnitude anchor provides geometric structure in z-space

**Claim:** Without L_mag, goals occupy the same manifold in z-space; with it, goals
separate into distinct clusters with inter-goal distances growing with λ.

**Evidence:**  
- `geom_no_anchor` = `no_comms`: 8.74 bits — no compression, no structure  
- `geom_mag_only`: 7.04 bits — goals cluster into compact regions  
- visual: Panel A of message_geometry.py shows 6 distinct clusters for Phase 1

**Pending (F6/F7):** Per-goal Gram matrix and cosine similarity heatmaps need to be
computed from GEOMETRY checkpoints. TC per config also needs post_hoc_coding.py.

---

## C10 — Post-hoc + Phase 2 Pareto-dominates all live coding configurations; robust across channel types

**Claim:** Post-hoc source coding applied to a converged Phase 1 policy, followed by
Phase 2 dither training, achieves H_joint ≈ 2.1–2.5 bits at SR=1.000 — decisively better
than any live coding configuration (~7 bits, SR≈1.000) with no z-blowup. Phase 2 is
channel-agnostic: NSD and SD channels achieve identical H_joint within noise.

**Evidence:**

*Phase 1 post-hoc alone (live_A, E20):*
- SR=1.000, hist_H≈7.05 — same as no-live-SC runs; post-hoc = free decomposition

*Live SC (live_B, E21):*
- SR=1.000, hist_H=7.05, true_bits=20.06 — 3× z-blowup, no H(m) reduction

*Live+Dither (live_C, E22):*
- SR≈0.980, hist_H≈7.67 — SR cost with no rate improvement

*Phase 2 v2 corrected dither — full 7-point λ sweep (E44–E46, E49–E52, post_hoc 2026-05-11):*

| Config (λ_dither) | H_joint (bits) | H(m\|goal) | SR |
|---|---|---|---|
| E49 (5e-5) | 2.491 ± 0.201 | 0.670 | 1.000 |
| E44 (1e-4) | 2.539 ± 0.105 | 0.722 | 1.000 |
| E50 (2e-4) | 2.546 ± 0.193 | 0.733 | 1.000 |
| E45 (5e-4) | 2.385 ± 0.174 | 0.570 | 1.000 |
| E46 (1e-3) | 2.286 ± 0.111 | 0.474 | 1.000 |
| E51 (2e-3) | 2.232 ± 0.052 | 0.416 | 1.000 |
| E52 (5e-3) | **2.094 ± 0.047** | **0.283** | 1.000 |
| Phase 1 ref (E09) | 4.156 ± 0.344 | 2.34 | 1.000 |

Phase 2 at λ=5e-3 achieves H_joint=2.094 bits — only 0.28 bits above H(G)=1.81 bits.
All 7 λ values maintain SR=1.000; the E39 listener fix is the reason no SR collapse occurs.
The non-monotonicity at low λ (5e-5, 1e-4, 2e-4 cluster near 2.5 bits) is within noise (std≈0.1–0.2 bits).

*NSD channel generalisation (§11.8, E47–E48, post_hoc 2026-05-11):*

| Config | Channel | H_joint (bits) |
|---|---|---|
| Phase 1 mag (E09 / E47) | SD | 4.156 ± 0.344 |
| Phase 1 mag (E47) | NSD | 4.137 ± 0.165 |
| Phase 2 λ=5e-4 (E45) | SD | 2.385 ± 0.174 |
| Phase 2 λ=5e-4 (E48) | NSD | 2.413 ± 0.141 |

NSD ≈ SD at both Phase 1 and Phase 2 (differences within 1 std). Confirms the histogram
estimator is channel-agnostic: q_hist is built from actual m samples regardless of channel type.

**Confidence:** High. All Phase 2 points strictly dominate the entire Phase 1 mag / live SC
Pareto frontier at equal SR=1.000. The Pareto advantage is decisive across all tested λ values
and both channel types. See C16 for the NSD generalisation as a standalone conclusion.

---

## C11 — dither_channel_loss formula was wrong; correct target is frac → 0.5

**Claim:** For the floor SD channel `m = floor((z + ε)/δ)` with `ε ~ U(-δ/2, δ/2)`,
the true H(m|z) per dimension is H_binary(|frac(z/δ) − 0.5|), not H_binary(frac(z/δ)).

**Derivation:**  
Let z_d = n·δ + f·δ (n integer, f = frac). Then z_d + ε ~ U((n+f-0.5)δ, (n+f+0.5)δ).

  For f ∈ [0, 0.5]: m ∈ {n−1, n},  P(n) = 0.5 + f,  P(n−1) = 0.5 − f  
    → H(m|z) = H_binary(0.5 − f)

  For f ∈ [0.5, 1): m ∈ {n, n+1},  P(n) = 1.5 − f,  P(n+1) = f − 0.5  
    → H(m|z) = H_binary(f − 0.5)

  In both cases: H(m|z) = H_binary(|f − 0.5|)

  Minimum (0 bits) at f = 0.5: z is at the centre of a quantisation bin; dither never
  crosses a boundary. Maximum (1 bit) at f = 0: z is at a bin boundary; 50/50 split.

**Verified empirically:** Simulation at 100K samples across all frac values confirms
H_binary(|frac − 0.5|) matches the true simulated H(m|z) exactly. H_binary(frac) is
maximally wrong at frac = 0 (code says 0 bits; true is 1 bit) and at frac = 0.5
(code says 1 bit; true is 0 bits).

**Compatibility with magnitude anchor:**  
The joint minimum of magnitude cost + dither loss is z ≈ δ/2 per dimension:
m = 0 deterministically (zero noise), |z| = δ/2 (low magnitude cost). The two
losses are compatible — they are not antagonistic at their shared optimum.

**Fix applied (2026-05-05):** Both `dither_channel_loss` and `dither_channel_stats`
in `source_coding.py` updated to use H_binary(|frac − 0.5|) with corrected gradient.

**Impact on experiments:**
- E12–E14 (sc_twophase): used wrong loss; true_bits reduction was a magnitude side-effect.
  Must re-run as E44–E46.
- E18 (geom_dither_only), E19 (geom_both): INVALIDATED. Re-run as E42–E43.
- E16 (geom_no_anchor), E17 (geom_mag_only): unaffected (no dither loss).
- All Group 1, 2, 5, 6 experiments: unaffected (use magnitude anchor only).

---

## C12 — TC grows with z_dim; z_dim=1 gives TC=0 (confirmed)

**Claim:** Total correlation TC(m) = 0 when z_dim=1 (trivially — no cross-dim correlation
possible), and grows with z_dim as the policy encodes goals into correlated multi-dim clusters.

**Evidence** (E37/E38/E09, post_hoc on 5 seeds each):

| z_dim | TC (mean) | H_joint (mean) | SR (mean) |
|---|---|---|---|
| 1 | 0.000 | 1.826 | 0.966 |
| 2 | 1.752 | 3.501 | 0.940 |
| 3 | 3.281 | 4.751 | 1.000 |

**Confidence:** High. TC=0 at z_dim=1 is exact (no off-diagonal structure possible).
Growth at z_dim=2,3 is consistent with the policy learning goal-specific clusters in higher
dimensions. SR degradation at lower z_dim is expected: fewer dimensions limits expressive capacity.

**Implication for paper (F15):** The joint-coding gain (H_factored − H_joint = TC) scales
with dimensionality. For the 3-dim case, joint arithmetic coding saves 3.28 bits — the
largest single compression lever available post-hoc.

---

## Open questions / unresolved

## C13 — The corrected dither loss reduces both H(m|goal) and TC(m)

**Claim:** Applying `dither_channel_loss` with the correct formula (frac → 0.5) reduces
both the within-goal noise H(m|goal) and the cross-dim correlation TC(m).

**Evidence** (E42/E43, post_hoc on 5 seeds each, corrected loss + E39):

| Config | H_dither | TC | H_joint |
|---|---|---|---|
| geom_no_anchor (baseline) | 2.27 | 5.01 | 4.64 |
| geom_mag_only (Phase 1) | 2.17 | 4.73 | 4.75 |
| geom_dither_only_v2 | **0.78** | **3.42** | **2.60** |
| geom_both_v2 | **0.57** | **2.80** | **2.39** |

The dither loss reduces H_dither by ~1.4 bits and TC by ~1.9 bits, cutting H_joint from
4.75 → 2.39 bits (50% reduction). Magnitude + dither cooperate: `geom_both_v2` achieves
the lowest H_joint.

**Mechanism (inferred):** Bin-centre alignment (frac → 0.5) forces each z_k toward a
fixed offset from the nearest integer. This reduces the variance in z and compresses the
implicit z-space codebook — goals occupy tighter, more separated regions — which reduces
the cross-dim correlation as a side effect. The TC reduction was not anticipated from the
theoretical design, which only targets H(m|goal).

**Confidence:** High. Consistent across all 5 seeds; residuals < 0.02 bits confirm
the decomposition holds.

---

## C14 — The DLM estimator fails due to circular gradient; the histogram is 2400× better

**Claim:** The Deep Latent Model (DLM) q_φ(m|z) has an irreducible estimation gap of
9–22 bits due to the circular gradient problem. The histogram achieves 0.004 bits (fixed).

**Evidence** (E41, P2-FIX stage, 5 configs × 5 seeds):

| Config | Final qphi_gap | SR | Interpretation |
|---|---|---|---|
| fix_baseline (DLM ref.) | 9.71 ± 0.62 bits | 0.977 | 2400× worse than histogram |
| fix_B1_gate (+gate +warmup) | 18.79 ± 3.79 bits | 0.993 | engineering fix makes it worse |
| fix_B2_gate_lambda (+λ=1e-2) | 21.97 ± 6.19 bits | 0.976 | even worse |
| fix_B3_ema (+EMA prior) | 20.33 ± 3.27 bits | 0.982 | no improvement |
| fix_B4_twophase (warm start) | **0.046 ± 0.000 bits** | **0.536** | low gap but SR collapse |
| Histogram (E09, reference) | 0.004 bits | 1.000 | 3000× improvement |

**Mechanism — circular gradient (confirmed):** All DLM configs start with qphi_gap ≈ 0.07
bits (early training, simple distribution). As the policy learns a rich goal→tuple codebook,
the distribution becomes complex (48 distinct tuples) and the DLM cannot track it — gap
grows monotonically from 0.07 → 9.71 bits. The backward gate (B1–B3) is always CLOSED
(qphi_gap >> gate_threshold = 2.0 bits), so additional q_φ training steps simply fit the
current distribution more aggressively, driving the gap higher as the distribution
becomes more complex. The histogram has no training loop — it always fits the current
distribution exactly.

**Mechanism — warm-start failure (F12):** fix_B4_twophase avoids the circular gradient
in Phase 1 (magnitude only, no entropy backward → simple distribution → gap = 0.046 bits).
But when Phase 2 activates the DLM backward into the speaker, SR collapses to 53.6%.
A well-fitted q_φ is necessary but not sufficient: the DLM score-function gradient is
destructive regardless of fit quality.

**Confidence:** High. The gap trajectory (0.07 → 9.71) is consistent across all seeds
and all DLM configs. The warm-start result (low gap, high SR collapse) is consistent
across all 5 seeds of fix_B4_twophase.

**Implication for paper (F8, F10, F12):** The histogram replacement is not an incremental
improvement — it eliminates a fundamental failure mode. The DLM cannot be fixed by
engineering (B1–B3 make it worse). The two-phase trick (B4) achieves low qphi_gap at
the cost of task performance, demonstrating that fitting q_φ is not the bottleneck.

---

## C15 — hist_H_empirical overestimates true H(m) by ~4.6 bits; true H drops toward H(G) via z-blowup

**Claim (F19, 2026-05-10):** The training-time metric `hist_H_empirical` significantly
overestimates the true joint H(m) for the live_B run. As live SC training progresses,
the true joint H(m) decreases toward H(G) — but this is a degenerate effect of z-blowup,
not useful compression.

**Evidence:**  
- live_B run (`sc_live_entropy`): true_bits=20.06, hist_H_empirical=7.05±0.35  
- F19 offline estimate (`true_H_offline`): starts ~6 bits, converges to **2.42 bits**  
- H(G) = 1.813 bits (theoretical floor)

**Why hist_H_empirical overestimates:**  
1. **Goal sampling:** Minibatch goal sampling is approximately uniform over all 6 goals
   (the environment samples goals at episode resets with frequency proportional to
   _GOAL_PROBS ≈ [0.515, 0.258, ...]). Over a short minibatch window, goal frequencies
   are more equal than the training distribution, inflating the effective H(G) toward
   log2(6) ≈ 2.585 bits and therefore inflating H(m).  
2. **Factored vs joint:** `hist_H_empirical` estimates the sum of per-dimension entropies
   H(m_0) + H(m_1) + H(m_2) (factored), not the joint H(m_0, m_1, m_2). The TC gap
   (3.28 bits in Phase 1) contributes to this overestimate.  
   Combined bias: ≈ 0.77 bits (goal weighting) + 3.28 bits (factored vs joint) ≈ 4 bits.

**Why true_H_offline decreases toward H(G):**  
As z-blowup progresses (true_bits → 20), the z-values for each goal are enormous
(e.g., z_0 ≈ ±10,000). Each goal occupies a distinct, non-overlapping region in integer
message space — no two goals ever share a bin. With no inter-goal message overlap, the
joint H(m) → H(G) = 1.813 bits trivially, because the only source of entropy is which
goal is active.

**Why this is degenerate, not useful compression:**  
The channel cost to transmit integer 10,000 in the SD channel is enormous (true_bits =
magnitude-based cost ≈ 20 bits). H(m) → H(G) is achieved not by restructuring the
codebook into a cheaper one, but by growing the codebook into extreme integer bins
that happen not to overlap. A joint arithmetic coder for this distribution would still
need ~20 bits of precision to specify which extreme bin is in use.

**Implication:**  
The two relevant measures of compression quality are H_joint (from post_hoc, using
correct goal sampling) and true_bits (SD channel cost). For live_B: H_joint ≈ 2.4 bits
(degenerate, 20-bit cost), while post_hoc on Phase 1 gives H_joint = 4.16 bits (20×
lower channel cost). Post_hoc coding Pareto-dominates live SC on both metrics.

**Confidence:** High. The mechanism is analytically clear; the offline measurement
matches the predicted degenerate limit.

---

## C16 — Histogram estimator and Phase 2 are channel-agnostic (SD ≈ NSD)

**Claim:** The histogram source coding estimator generalises automatically to the
non-subtractive dithering (NSD) channel because q_hist is built from actual m samples
regardless of the quantisation channel used. Phase 2 dither training on NSD achieves
the same H_joint reduction as on SD.

**Evidence (E47–E48, post_hoc 2026-05-11):**

| Config | Channel | Phase | H_joint (bits) | SR |
|---|---|---|---|---|
| E09 `sc_posthoc_mag` | SD | Phase 1 | 4.156 ± 0.344 | 1.000 |
| E47 `nsd_posthoc_mag` | NSD | Phase 1 | 4.137 ± 0.165 | 1.000 |
| E45 `sc_twophase_dither5e-4_v2` | SD | Phase 2 | 2.385 ± 0.174 | 1.000 |
| E48 `nsd_twophase_dither5e-4_v2` | NSD | Phase 2 | 2.413 ± 0.141 | 1.000 |

Differences: 0.019 bits (Phase 1) and 0.028 bits (Phase 2) — both within 1 std. No
statistically meaningful difference between channels.

**Why:** NSD adds independent TPDF dither (uniform on [−δ, δ]), but the receiver still
observes integer-valued m. The histogram over m is identical in structure between SD and
NSD — only the within-bin noise distribution differs, and the histogram is insensitive
to this. The dither loss operates on frac = z/δ − floor(z/δ), which is also channel-agnostic.

**Confidence:** High. The two channels are mechanistically separated by design and
the empirical values match within noise.

---

**OQ1 (CLOSED 2026-05-06):** E39 implemented — listener keeps RL gradients in Phase 2.
Confirmed necessary: first batch of E42–E46 (frozen listener) collapsed SR from 1.0 to ~5%.
E42–E46 relaunched with fix; awaiting results.

**OQ2 (CLOSED 2026-05-06):** TC at z_dim ∈ {1,2} confirmed via E37/E38.
TC = 0 (z_dim=1), 1.752 (z_dim=2), 3.281 (z_dim=3). See C12.

**OQ3 (CLOSED 2026-05-08):** E42/E43 completed. From F7 post_hoc data:
geom_no_anchor TC=5.01, geom_mag_only TC=4.73 — the magnitude anchor reduces TC by only
0.28 bits. The dither loss (E42: geom_dither_only_v2) reduces TC from 4.73 → 3.42 bits.
The magnitude anchor primarily separates goals in z-space (F6) but does little to
decorrelate per-dim messages. TC reduction is mainly a dither effect (C13).

**OQ4 (CLOSED 2026-05-06):** Large sc_posthoc_mag residuals (0.77 mean) were caused by
uniform goal sampling in `collect_messages`, not insufficient z-samples. After fix, all
5 seed residuals are in [−0.001, +0.013] bits. Increasing z-samples 100→1000 is a no-op
for deterministic speakers. geom_dither_only/geom_both residuals (2.74/3.32) were caused
by the dither formula error; will be re-checked after E42–E43 complete.

**OQ5 (CLOSED 2026-05-06):** Secondary bin selection fixed in `source_coding_rate_loss`.
For frac < 0.5, secondary bin is now `m_base − 1` (not `m_base + 1`). No re-runs needed —
gradient direction was already correct; only magnitude was biased.

---

## Open questions — Pillars P1, P4, P3, ALL (as of 2026-05-12)

These questions will be resolved as E53–E65 complete and post-hoc analysis runs.

**OQ6 (OPEN):** Does learned per-channel δ (E54) differentiate across dimensions?
- Expected: δ_k values diverge across dimensions, reflecting unequal H(m_k) from Phase 1.
- Falsification: all δ_k converge to the same value as global δ (E53). This would mean
  the optimizer finds no signal to allocate bits asymmetrically.
- Resolves to: C17. Data: post_hoc on E53/E54 checkpoints.

**OQ7 (OPEN):** Does the Rao-Blackwell gradient (E58) reach SR=0.99 in fewer updates
than the STE baseline (E57)?
- Expected: RB-joint reduces variance on the speaker task gradient → faster Phase 1.
- Falsification: no statistically significant difference in steps-to-SR99. SR=0.99 may
  be achieved through the listener without requiring a good speaker task gradient.
- Resolves to: C18. Data: metrics.csv column `training_phase` transition timestamps.

**OQ8 (OPEN, conditional):** Do P1 and P4 interact non-additively?
- Run E61 (RB-joint + per-channel δ) only if OQ6 and OQ7 both show positive results.
  If effects are independent, E64/E65 can use the best single-pillar configs directly.
- Resolves to: C19 (if E61 runs). Data: E54, E58, E61 H_joint comparison.

**OQ9 (OPEN):** Do all four pillars compose additively (or super-additively) to achieve
H_joint below any single-pillar result?
- Expected: E64/E65 H_joint < E55 (P1+P2), E60 (P4+P2). SR ≥ 0.99.
- Falsification: H_joint(E64) > H_joint(E52 or E60). This would indicate destructive
  interference and must be reported even if it falsifies the paper's headline claim.
- Resolves to: C20. Data: post_hoc on E64/E65. Gate: OQ6+OQ7 both resolved.
