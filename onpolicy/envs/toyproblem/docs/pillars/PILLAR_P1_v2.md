# Pillar P1 — Per-Channel δ (Adaptive Quantisation Width)

**Status:** IMPLEMENTATION DONE — v2 plan drafted 2026-05-11; implementation complete 2026-05-11. Experiments (E53–E56) pending.
**Pillar order:** P2 → **P1** → P4 → P3
**Paper reference:** "Stochastic Quantisation via Dithering" §Thm 4
**Depends on:** P2 experiments complete (E01–E52, all figures F1–F21 done)

---

## 1. The Core Question: Can Unequal Bin Widths Reduce H(m)?

### 1.1 What global δ gets wrong

In all P2 experiments, every dimension k of the message vector uses the same bin
width δ (default 1.0). The magnitude loss penalises all dimensions equally:

```
L_mag = Σ_k log₂(|z_k|/δ + 1)
```

But the task does not use all dimensions equally. In the toyproblem, z ∈ ℝ³ must
encode one of 6 goals — an intrinsically 2D signal (x, y coordinates) into a 3D
message. With global δ, the speaker is forced to waste one dimension's bit budget
either on:
- Redundant information (repeating goal identity), or
- Near-zero signal (z_k ≈ 0, contributing ~0 bits but still occupying a slot)

Per-channel δ_k allows the speaker to declare one dimension "cheap" (large δ_k →
coarse bins → small m_k → few bits) and concentrate precision on the two task-relevant
dimensions (small δ_k → fine bins → large m_k → more bits, but encoding more goal info).

The claim is that the optimal per-channel allocation reduces total H(m) below what
is achievable with global δ, at equal or higher SR.

### 1.2 Theorem 4 — the theoretical foundation

For the SD channel with per-dimension dither ε_k ~ U(-δ_k/2, +δ_k/2) drawn
independently per dimension k:

```
m_k = floor((z_k + ε_k) / δ_k)       ← integer message per dimension
ẑ_k = z_k + e_k,  e_k ~ U(-δ_k/2, +δ_k/2),  e_k ⊥⊥ z_k   (Schuchman, per-dim)
∂ẑ_k/∂z_k = 1   (exact, not STE — same as global δ case)
```

Independence across dimensions is preserved because each ε_k is drawn independently.
The Schuchman collapse (Thm A.1 / Thm 2) applies dimension-wise; the STE gradient
is exact for each k separately. **No additional approximation is introduced by making
δ per-channel.**

### 1.3 The remaining gap from P2

Best P2 result (E52, λ_dither=5e-3, all SR=1.000):
```
H_joint = 2.094 bits    H(G) = 1.813 bits    Gap = 0.281 bits
```

This gap decomposes as:
```
H_joint − H(G) = H(m|goal)  ≈  0.28 bits  [average residual conditional entropy]
```

Two sources of the remaining gap:
1. **Intra-goal spread** — within a goal, the speaker does not always produce the
   same m (z is somewhat spread). The dither loss shrinks this but does not eliminate
   it at λ=5e-3 without SR degradation.
2. **Dimension waste** — one of the 3 dimensions may be carrying redundant or
   near-uniform information, inflating H_joint above the minimum.

P1 attacks source (2) directly: if the third dimension's δ_3 grows large (bins coarser),
m_3 concentrates on {0, ±1} → H(m_3) → 0. Combined with P2, this could bring
H_joint → H(G).

---

## 2. Per-Channel δ — Mathematical Formulation

### 2.1 Parameterisation

```
δ_k = softplus(α_k)    α_k ∈ ℝ   (learnable, initialised so δ_k = 1.0)
```

Softplus guarantees δ_k > 0 always. Initialisation: α_k = softplus_inv(1.0) ≈ 0.541
so δ_k = 1.0 at the start of training, identical to the global δ=1.0 baseline.

A lower clamp δ_min = 0.1 should be enforced to prevent δ_k → 0 (which would require
infinite bits per dimension and make m_k unbounded). Implementation:
```python
delta_k = F.softplus(self.log_alpha_k).clamp(min=0.1)
```

### 2.2 Magnitude loss gradient w.r.t. δ_k

```
L_mag_k = log₂(|z_k|/δ_k + 1)
```

Gradient descent on L_mag_k w.r.t. δ_k pushes δ_k **up** (coarser bins = fewer bits
in the surrogate):

```
∂L_mag_k/∂δ_k = −|z_k| / (δ_k · (|z_k| + δ_k) · ln 2)    < 0
```

This is always negative, so ∂L_mag_k/∂(−δ_k) > 0: gradient descent increases δ_k.
The task loss counteracts this (see §2.3), forcing an equilibrium.

### 2.3 Task loss gradient w.r.t. δ_k

In the SD training path:
```
e_k = (u_k − 0.5) · δ_k,   u_k ~ Uniform(0, 1)
ẑ_k = z_k + e_k
```

Here e_k is differentiable w.r.t. δ_k:
```
∂ẑ_k/∂δ_k = (u_k − 0.5)    [a scalar, random sign]
```

So the task gradient w.r.t. δ_k is:
```
∂L_task/∂δ_k = ∂L_task/∂ẑ_k · (u_k − 0.5)
```

This is an unbiased, zero-mean signal (E[u_k − 0.5] = 0) with variance proportional
to Var(∂L_task/∂ẑ_k). The task loss provides a **noise-regularising** gradient: when
larger δ_k hurts task performance (listener cannot identify goal under coarse bins),
L_task increases and the gradient pushes δ_k down. The equilibrium point is where
precision is just sufficient for task success.

**Important:** This gradient is noisy (u_k is random) and may be high-variance. A small
dedicated learning rate for α_k (e.g. 10× smaller than the speaker LR) is recommended
to prevent δ_k from oscillating.

### 2.4 Dither loss gradient w.r.t. δ_k

In Phase 2, the dither loss per dimension is:

```
H_dither_k = H_binary(|frac(z_k/δ_k) − 0.5|)
```

where frac(x) = x − floor(x). The gradient w.r.t. δ_k flows through frac:

```
∂frac(z_k/δ_k)/∂δ_k = −z_k/δ_k²   (STE through the floor)
```

So:
```
∂H_dither_k/∂δ_k = [dH_binary/d(frac)] · ∂frac/∂δ_k
                 = log₂((1−g)/g) · sign(frac − 0.5) · (−z_k/δ_k²)
```

where g = |frac − 0.5|. This gradient can push δ_k in either direction depending on
the current fractional part and z_k's sign/magnitude. It will cooperate with the dither
gradient w.r.t. z_k to jointly push (z_k, δ_k) toward the minimum-noise locus
frac(z_k/δ_k) = 0.5.

### 2.5 Rate decomposition under per-channel δ

The post-hoc rate decomposition extends directly:

```
H_factored(m) = Σ_k H(m_k)             [sum of per-dim entropies]
H_joint(m)    = H(m_0, …, m_{D−1})     [joint entropy]
TC(m)         = H_factored − H_joint   [total correlation]
H(m|goal)     = H_joint − H(G)         [residual after goal conditioning]
```

P1 changes the distribution of each m_k (coarser δ_k → smaller m_k range → lower H(m_k)),
but the decomposition identity remains exact. The key question is whether unequal δ_k
can reduce Σ_k H(m_k) and/or TC(m) below the P2 optimum.

---

## 3. Implementation Design

### 3.1 Where δ_k lives

**Option A: Inside the channel.**
`DDCL_SD_PerChannel` takes a `delta: torch.Tensor` of shape `(z_dim,)` instead of a
scalar. This keeps the channel self-contained and does not change the speaker network.
Pro: clean separation; channel knows its own bin widths.
Con: the channel's δ must be updated from outside (trainer must pass optimised δ values).

**Option B: Shared parameter module.**
A `PerChannelDelta` module holds `log_alpha` and exposes `delta()`. Both the channel
and the trainer reference this module; the optimiser updates it.
Pro: single source of truth; natural integration with PyTorch optimiser groups.
Con: slightly more wiring.

**Recommended: Option B.** The `PerChannelDelta` module is added to the trainer's
parameter groups with a dedicated (lower) learning rate. The channel receives a fresh
`delta` tensor each forward pass.

### 3.2 Channel modifications

`DDCL_SD` and `DDCL_NSD` currently take a scalar `delta: float`. Two changes:

1. `delta` becomes a `torch.Tensor` of shape `(z_dim,)` (or broadcast scalar for
   backward compatibility).
2. The magnitude loss, dither path, and m computation all vectorise over k:
   ```python
   m = torch.floor((z + eps) / self.delta)          # delta broadcasts over batch
   z_hat = z + (torch.rand_like(z) - 0.5) * delta  # training path
   ```
3. `comms_loss` returns per-element: `log₂(|z|/delta + 1)` — delta is now a vector,
   broadcasting over the batch dimension naturally.

### 3.3 Trainer modifications

1. Instantiate `PerChannelDelta(z_dim)` and add to optimiser with `lr_delta` (e.g. 1e-4,
   10× smaller than speaker LR 3e-4).
2. Each forward pass: call `channel.set_delta(per_channel_delta.delta())` or pass delta
   directly to the channel forward.
3. Log δ_k per dimension to `metrics.csv` each update (3 new columns: `delta_0`,
   `delta_1`, `delta_2`).
4. The Phase 2 dither loss must receive the current δ_k tensor from `PerChannelDelta`,
   not the global config `delta`.

### 3.4 Phase structure

Two natural integration points with the Phase 2 design:

**Design A: P1 in Phase 1 only.**
δ_k adapts during Phase 1 (magnitude loss). Once SR ≥ 0.99 threshold and Phase 2
begins, freeze δ_k and apply dither loss with the learned widths. Simpler; avoids
histogram staleness in Phase 2.

**Design B: P1 throughout both phases.**
δ_k continues to adapt in Phase 2. The histogram must be invalidated and refit whenever
δ_k changes by more than a threshold (e.g., any δ_k changes by > 5%). More complex
but potentially lower final H(m) since dither loss can jointly optimise (z_k, δ_k).

**Recommended: Design A first.** If Phase 1 alone does not produce useful δ_k
separation, revisit Design B.

### 3.5 Histogram interaction

`MessageHistogram` tracks m_k = floor((z_k + ε_k)/δ_k). If δ_k changes between epochs,
the same z values produce different m_k values — the histogram is stale. Two mitigations:

1. **Reset on large δ_k change:** if max(|Δδ_k|/δ_k) > 0.05 between epochs, clear
   histogram counts and re-accumulate on the next rollout. Cost: one epoch of inaccurate
   rate signal.
2. **Slow δ_k learning:** with `lr_delta` 10× smaller than speaker LR, δ_k moves slowly
   enough that histogram staleness is negligible within one EM cycle.

For the initial implementation, use mitigation 2 (slow learning) and monitor δ_k traces.

---

## 4. Expected Failure Modes

### 4.1 δ_k → 0 (collapse to infinite precision)

If the task gradient dominates and always pushes δ_k down (higher precision = easier
task), δ_k can approach the lower clamp (0.1). At δ_k = 0.1 with typical |z_k| ≈ 2,
m_k ≈ 20 — similar to the z-blowup failure mode seen in live SC (E21). The magnitude
loss counteracts this, but if λ_mag is too small, collapse can occur.

**Mitigation:** Maintain a minimum δ_k = 0.1 clamp and monitor `delta_k` traces in
metrics.csv. If any dimension collapses early in training, increase λ_mag.

### 4.2 No differentiation (all δ_k converge to the same value)

If the task is symmetric across all 3 z dimensions (which it is NOT — 3D z for a 2D
goal), all δ_k may converge to the same value and P1 provides no benefit over global δ.
In the toyproblem, the speaker is free to use any of the 3 dimensions as it chooses
(there is no structural prior on which z_k encodes what), so symmetry-breaking depends
on random initialisation and whether the magnitude gradient is strong enough to reward
coarsening one dimension.

**Mitigation:** Run 5 seeds. If no seed shows differentiation, the task may not provide
sufficient gradient signal to break the symmetry. In that case, consider a structured
initialisation where one α_k starts larger.

### 4.3 Histogram invalidity under changing δ_k

If δ_k changes substantially during Phase 1 (before histogram is used), the Phase 2
histogram may be initialised from a distribution that no longer matches the current
channel. This manifests as high initial qphi_gap or incorrect rate gradients in the
first epoch of Phase 2.

**Monitoring:** Log histogram entropy vs. true H(m_k) at Phase 2 onset. If discrepancy
> 0.5 bits, add an explicit histogram reset at Phase 2 transition.

### 4.4 Task gradient variance for δ_k

∂L_task/∂δ_k = ∂L_task/∂ẑ_k · (u_k − 0.5) has high variance because u_k is
independently sampled and (u_k − 0.5) has zero mean. With 16 environments and 256
steps per rollout = 4096 samples per update, the effective gradient for δ_k is
averaged over 4096 samples, which should be sufficient to reduce variance to manageable
levels. Monitor `delta_k` traces for oscillation — if present, reduce `lr_delta`.

---

## 5. Ablation Matrix

### 5.1 P1 alone — global vs per-channel δ (primary P1 result)

| Config | δ | Phase 1 loss | Post-hoc code | Primary metric |
|---|---|---|---|---|
| (A) Global δ baseline | δ = 1.0 (fixed) | magnitude | H_joint | H_joint at SR=1 |
| (B) Learned global δ | δ = softplus(α), 1 param | magnitude | H_joint | H_joint at SR=1 |
| (C) Learned per-channel δ | δ_k = softplus(α_k), 3 params | magnitude | H_joint | H_joint at SR=1 |

Prediction: (C) ≤ (B) ≤ (A) in H_joint at SR=1. If (B) < (A), a globally larger δ
is sufficient for H reduction. If (C) < (B), dimensional differentiation is happening.

### 5.2 P1 + P2 combined

Run Phase 2 dither loss on top of Phase 1 learned δ_k:

| Config | Phase 1 | Phase 2 |
|---|---|---|
| (A) P2 only (best, E52) | global δ=1.0, magnitude | λ_dither=5e-3, corrected loss |
| (B) P1 Phase 1 + P2 Phase 2 | per-channel δ, magnitude | λ_dither=5e-3, corrected loss |

Prediction: (B) ≤ (A) in H_joint. If the best P2 point already reaches near-Shannon
efficiency (2.094 bits vs H(G)=1.813), P1 may close most of the remaining 0.28-bit gap.

### 5.3 δ_k trace ablation

Run 5 seeds of Config C (per-channel δ, Phase 1) and measure at convergence:
- δ_0, δ_1, δ_2 values (are they differentiated?)
- Per-dim H(m_k) vs δ_k (does smaller δ → larger H(m_k)?)
- Total H_joint vs global δ baseline

### 5.4 Learning rate sensitivity for δ_k

| lr_delta | Notes |
|---|---|
| 3e-5 (100× slower) | Conservative; may not move δ meaningfully |
| 1e-4 (30× slower) | **Recommended default** |
| 3e-4 (same as speaker) | May cause δ oscillation |

### 5.5 Phase design ablation (Design A vs B)

| Config | δ_k in Phase 2? |
|---|---|
| Design A | Frozen at Phase 2 onset |
| Design B | Continues adapting in Phase 2 |

Only run Design B if Design A shows insufficient H_joint reduction.

---

## 6. Figure Plan (F22–F25)

| Figure | Description | Data source | Experiment(s) |
|---|---|---|---|
| F22 | δ_k per-dimension trajectory over training | `metrics.csv` delta_0/1/2 columns | P1 Phase 1 runs |
| F23 | P1 + P2 Pareto: H_joint vs SR | post_hoc aggregate.json | P1 baseline + P1+P2 configs |
| F24 | δ_k vs H(m_k) scatter at convergence (5 seeds) | post_hoc + metrics.csv | P1 Phase 1 runs |
| F25 | Rate decomposition bar: P2-only vs P1+P2 | post_hoc aggregate.json | E52 + P1+P2 best config |

**F22 — δ_k trajectory:** Three curves (one per dimension), possibly showing one
dimension's δ_k growing larger (coarser) over Phase 1 training while the others shrink
or stay near 1.0. Reference line at δ=1.0 (global baseline).

**F23 — P1+P2 Pareto:** Add P1+P2 operating points to the F21-style plot. If H_joint
drops below 2.094 bits (best P2), this is the headline P1 result.

**F24 — δ_k vs H(m_k):** Scatter plot with one point per (seed, dimension). If P1 works,
dimensions with small δ_k should have large H(m_k) and vice versa — confirming the
adaptive allocation hypothesis.

**F25 — Decomposition bar:** Side-by-side stacked bars for P2-only (E52) and P1+P2
(best config), showing H(G), H(m|goal), TC. P1 should reduce TC (coarsening unused
dimension eliminates cross-dim correlation) and possibly H(m|goal).

---

## 7. Key Open Questions (to resolve before implementation)

**Q1 — Does the task gradient to δ_k have the right sign?**
∂L_task/∂δ_k = ∂L_task/∂ẑ_k · (u_k − 0.5). In expectation this is zero (the task
gradient averages out because E[u_k − 0.5] = 0). The magnitude gradient always pushes
δ_k up. Does this mean δ_k only increases? No — the **variance** of the task gradient
w.r.t. δ_k is non-zero, and Adam's adaptive moment estimation can capture a direction.
But this needs empirical verification: does per-channel δ actually differentiate, or
does it just drift upward uniformly (same as learned global δ)?

*Resolution path:* Run 5 seeds of Config C. If δ_k differentiation is <0.1 across
dimensions at convergence, Q1 answer is "task gradient to δ_k is too noisy" and an
alternative gradient source is needed.

**Q2 — Alternative: use the magnitude gradient only?**
The magnitude gradient ∂L_mag_k/∂δ_k = −|z_k|/(δ_k(|z_k|+δ_k) ln2) is always
negative (pushes δ_k up), but its magnitude scales with |z_k|. Dimensions where the
speaker uses large |z_k| will see stronger upward pressure on δ_k — which is the
WRONG direction (large |z_k| = dimension is being used = needs fine precision).

This means the magnitude loss gradient alone cannot produce useful per-channel differentiation.
The task gradient (§2.3) is necessary for P1 to work. If it's too noisy, P1 may fail.

*Alternative approach:* Use the P2 histogram signal to drive δ_k: if H(m_k) for
dimension k is already near H(G) (dimension is well-used), leave δ_k alone; if H(m_k)
is near-zero (dimension is unused), increase δ_k. This is a rule-based heuristic rather
than an end-to-end gradient — cleaner but requires additional logic.

**Q3 — Does the STE ∂ẑ_k/∂δ_k hold?**
In the current SD implementation:
```python
e = (torch.rand_like(z) - 0.5) * d   # d is a scalar float, not a tensor
z_hat = z + e
```
If `d` becomes a tensor `delta_k`, autograd will track the graph through `e`, and
`∂z_hat/∂delta_k = ∂e/∂delta_k = (u_k − 0.5)`. This is exactly what §2.3 says.
Verify with a small unit test before committing to the full implementation.

**Q4 — Interaction with source coding rate loss w.r.t. δ_k**
The score function proxy loss is:
```
L_rate = λ · [(R_hi_k − R_lo_k).detach() / δ_k] · z_k
```
Here δ_k appears in the denominator. When δ_k is a learned parameter, this term has
a gradient w.r.t. δ_k:
```
∂L_rate/∂δ_k = −λ · [(R_hi_k − R_lo_k).detach() / δ_k²] · z_k
```
This is a second path for δ_k gradient in Phase 2. It pushes δ_k up when R_hi > R_lo
(current bin is cheaper) and down when R_hi < R_lo (upper bin is cheaper). This is
meaningful. **The dither loss and rate loss both provide δ_k gradients in Phase 2 —
this needs careful monitoring to avoid instability.**

---

## 8. Implementation Checklist

| Component | File | Status |
|---|---|---|
| `PerChannelDelta` module (α_k, softplus, clamp) | `network.py` or `channels.py` | TODO |
| `DDCL_SD` accept vector δ | `channels.py` | TODO |
| `DDCL_NSD` accept vector δ | `channels.py` | TODO |
| Trainer: instantiate PerChannelDelta, add to optimiser | `trainer.py` | TODO |
| Trainer: pass delta tensor to channel each forward | `trainer.py` | TODO |
| Trainer: log delta_0, delta_1, delta_2 to metrics.csv | `train.py` + `trainer.py` | TODO |
| CLI flag: `--learn_delta` (store_true) | `train.py` | TODO |
| CLI flag: `--lr_delta` (float, default 1e-4) | `train.py` | TODO |
| Source coding rate loss: δ in denominator as tensor | `source_coding.py` | TODO |
| Dither loss: δ passed as tensor for gradient | `source_coding.py` | TODO |
| Unit test: ∂z_hat/∂δ_k with vector delta | `tests/test_channels.py` | TODO |
| Unit test: PerChannelDelta clamp and softplus | `tests/test_channels.py` | TODO |
| F22 figure: δ_k trajectory plot | `analysis/` (new file or sc_ablation_figures.py) | TODO |
| F23–F25 figures | `analysis/sc_ablation_figures.py` | TODO |

---

## 9. Experiment Registry Entries (to add before running)

| ID | Experiment | Stage | Purpose |
|---|---|---|---|
| E53 | `p1_global_delta_learned` | P1_BASELINE | Learned scalar δ (1 param); isolates global δ benefit |
| E54 | `p1_perchannel_delta_v1` | P1_BASELINE | Learned per-channel δ_k; primary P1 result |
| E55 | `p1_perchannel_dither5e-3_v2` | P1_PHASE2 | P1 Phase 1 + P2 Phase 2 (λ=5e-3); headline combined result |
| E56 | `p1_perchannel_dither5e-3_v2_freeze` | P1_PHASE2 | Same but δ_k frozen at Phase 2 onset (Design A vs B ablation) |

Run 5 seeds each. E53 and E54 need post_hoc_coding after training. E55 and E56 need
post_hoc_coding + F23/F25 after completion.

---

## 10. Success Criteria

P1 is **successful** if, at SR=1.000:

1. E54 shows measurable differentiation across δ_k values (max/min ratio > 1.5 across
   dimensions in ≥3/5 seeds).
2. E54 achieves lower H_joint than the E09 global-δ baseline (4.156 bits), confirming
   P1 reduces bit cost even without P2.
3. E55 achieves H_joint < 2.094 bits (E52, best P2 alone), confirming P1+P2 > P2 alone.

P1 is **inconclusive** if δ_k values do not differentiate (max/min ratio < 1.2 across
all seeds). In this case, fallback: implement a heuristic rule based on per-dimension
post-hoc H(m_k) to set δ_k manually rather than learning it end-to-end.

P1 is **failed** if SR drops below 0.95 in E54 (per-channel δ destabilises training).

---

## 11. Pillar Sequencing and Dependencies

P1 should be implemented before P4 (Rao-Blackwell) because:
- P1 changes the channel interface (scalar → vector δ), which P4 also uses
- P4's probability computation `p_b = frac(z_k/δ_k)` needs the per-channel δ
- P1+P4 together are a stronger combined baseline for P3 comparison

P3 (TPDF deployment) is independent of P1 and can run in parallel if resources allow.

---

## 12. Reading Order and Orientation

1. **This document (§1–§5)** — core math, design decisions, failure modes, ablations
2. **`docs/MATH.md §6`** — Theorem 4 statement; update this section when P1 is implemented
3. **`channels.py`** — current DDCL_SD/NSD implementation to understand the delta parameter
4. **`trainer.py`** — current Phase 1/2 trainer structure; identify where PerChannelDelta plugs in
5. **`source_coding.py`** — score function gradient and dither loss; understand δ dependency
6. **`PILLAR_P2_v2.md §16`** — figure-to-experiment map; F22–F25 add to this table when done
