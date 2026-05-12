# Pillar P4 — Rao-Blackwell Gradient Estimator

**Status:** IN DESIGN — v2 plan drafted 2026-05-12; implementation not yet started.
**Pillar order:** P2 → P1 → **P4** → P3
**Paper reference:** "Stochastic Quantisation via Dithering" §Thm 6
**Depends on:** P2 closed (E01–E52 done); P1 implementation done (E53+ pending)

---

## 1. The Core Question: Does Replacing STE with RB Improve Training?

### 1.1 What the STE gets wrong

DDCL trains the speaker with the straight-through estimator (STE): the quantised
channel forward pass is `ẑ_k = z_k + e_k`, where `e_k ~ U(-δ/2, δ/2)` is fresh dither
noise. The gradient flows as `∂ẑ_k/∂z_k = 1`, ignoring the floor discontinuity.

The STE approximation introduces **gradient noise** from two sources:
1. **Dither noise:** `e_k` is a new random sample each forward pass. The gradient
   `∂L/∂ẑ_k` is evaluated at `ẑ_k = z_k + e_k`, which varies each call. The STE
   gradient is a noisy estimate of `∂E_m[L]/∂z_k`.
2. **Sign noise:** `∂L/∂z_k = ∂L/∂ẑ_k · 1` — the "1" is the STE approximation of
   a true gradient that would pass through the floor discontinuity. The floor
   quantiser has a SCORE FUNCTION gradient (not a reparameterisation one).

The Rao-Blackwell (RB) estimator addresses both: it computes the **exact expected
gradient** of the task loss w.r.t. z_k by marginalising over the two candidate bins
that z_k can land in, without sampling the dither noise.

### 1.2 Theorem 6 — the foundation

For the SD channel, given z_k, only two integer bins are reachable:
```
frac_k = frac(z_k/δ_k) ∈ [0, 1)

frac_k < 0.5:  m_lo = floor(z_k/δ_k) − 1,  m_hi = floor(z_k/δ_k)
               p_lo = 0.5 − frac_k,          p_hi = 0.5 + frac_k

frac_k ≥ 0.5:  m_lo = floor(z_k/δ_k),       m_hi = floor(z_k/δ_k) + 1
               p_lo = 1.5 − frac_k,           p_hi = frac_k − 0.5
```

The decoder outputs bin centres: `ẑ_lo_k = (m_lo + 0.5)·δ_k`,
`ẑ_hi_k = (m_hi + 0.5)·δ_k`.

The true expected gradient of `E_m[L(m)]` w.r.t. z_k, via the score function:

```
∂/∂z_k E_m[L(m)] = p_lo · L(ẑ_lo) · ∂log p_lo/∂z_k
                   + p_hi · L(ẑ_hi) · ∂log p_hi/∂z_k
                 = [L(ẑ_hi) − L(ẑ_lo)] / δ_k
```

The simplification to a finite difference divided by δ_k holds for all values of
frac_k (proved in §1.3 below). This is the **Rao-Blackwell gradient**.

### 1.3 Derivation of the finite-difference form

For frac_k ≥ 0.5 (m_lo = n, m_hi = n+1, p_hi = frac_k − 0.5, p_lo = 1.5 − frac_k):

```
∂log p_hi/∂z_k = +1/((frac_k − 0.5) · δ_k)
∂log p_lo/∂z_k = −1/((1.5 − frac_k) · δ_k)

g_RB_k = p_hi · L(ẑ_hi) · (1/((frac_k − 0.5) · δ_k))
        + p_lo · L(ẑ_lo) · (−1/((1.5 − frac_k) · δ_k))
       = L(ẑ_hi)/δ_k − L(ẑ_lo)/δ_k
       = [L(ẑ_hi) − L(ẑ_lo)] / δ_k  ✓
```

For frac_k < 0.5 (m_lo = n−1, m_hi = n, p_hi = 0.5 + frac_k, p_lo = 0.5 − frac_k):

```
∂log p_hi/∂z_k = +1/((0.5 + frac_k) · δ_k)
∂log p_lo/∂z_k = −1/((0.5 − frac_k) · δ_k)

g_RB_k = [L(ẑ_hi) − L(ẑ_lo)] / δ_k  ✓
```

Same formula. The RB gradient is a **signed finite difference** over the bin width δ_k,
with sign given by which of `L(ẑ_hi)` and `L(ẑ_lo)` is larger.

### 1.4 Key insight: source_coding_rate_loss IS already the RB gradient for rate

The existing `source_coding_rate_loss` computes:
```python
grad_scale = (R_hi − R_lo) / delta   # (B, z_dim)
loss = lambda_comms * (grad_scale * z).sum(dim=-1).mean()
```

This gives `∂loss/∂z_k = λ · (R_hi_k − R_lo_k) / δ_k`, which is exactly the
Rao-Blackwell gradient for `L = E_m[−log₂ q(m_k)]` (the histogram rate).

**P4 has already been implemented for the rate loss.** The remaining contribution is
applying the same principle to the **task loss** (L = L_actor + L_critic).

---

## 2. STE vs RB: Variance Analysis

### 2.1 Variance decomposition

Let L(ẑ) denote the task loss as a function of the listener's input ẑ.

**STE gradient** for one sample z_k (with dither sample e_k):
```
g_STE_k = ∂L(z_k + e_k)/∂(z_k + e_k) · 1   (STE: ∂ẑ/∂z = 1)
```

Variance (w.r.t. e_k): Var_e[g_STE_k] = Var_e[∂L(z_k + e_k)/∂ẑ_k] > 0

**RB gradient** for one sample z_k (analytical, no dither sampling):
```
g_RB_k = [L(ẑ_hi_k) − L(ẑ_lo_k)] / δ_k
```

Variance (w.r.t. dither): Var[g_RB_k | ẑ_hi_k, ẑ_lo_k fixed] = 0

The RB gradient **eliminates** the within-sample dither variance entirely for dimension k.
Remaining variance comes from:
1. Variance across minibatch samples (different z_b, irreducible)
2. Variance from other dimensions' dither (if those dims are not also Rao-Blackwellized)

### 2.2 Expected variance reduction

In the independent-dimensions approximation (dimensions treated separately):
- Each dimension's dither contributes independently to gradient variance
- Eliminating one dimension's dither reduces total STE variance by approximately 1/z_dim
- Full per-dim RB (all z_dim dims): reduces STE dither variance to zero

For z_dim=3: expected ~3× reduction in dither-noise component of gradient variance.

### 2.3 When variance reduction matters most

- **Phase 1 (learning phase):** High task-gradient variance → RB most useful
- **Phase 2 (compression phase):** SR ≈ 1, task gradient already small → RB less useful
- **Small minibatches:** higher variance → RB more beneficial
- **This experiment:** n_envs=16, n_steps=256, num_minibatches=4 → B=1024 per minibatch.
  At B=1024, across-sample variance dominates; within-sample dither variance is secondary.
  RB benefit may be **modest** in this specific setting but worth measuring.

---

## 3. Implementation Design

### 3.1 Two-path forward pass (recommended)

For each minibatch, compute ẑ_lo and ẑ_hi for all dimensions simultaneously:

```python
# Current STE path
z_hat_ste, ch_info = channel(z)          # z_hat_ste = z + e (random)
m = ch_info["m"]                         # floor((z + eps)/delta)

# RB: compute the two bin-centre outputs
with torch.no_grad():
    frac = (z / delta) - torch.floor(z / delta)
    low_frac = frac < 0.5
    m_lo = torch.where(low_frac, m - 1, m)
    m_hi = torch.where(low_frac, m,     m + 1)
    z_hat_lo = (m_lo + 0.5) * delta          # (B, z_dim)
    z_hat_hi = (m_hi + 0.5) * delta          # (B, z_dim)

# Two forward passes through listener
dist_lo = listener(cat([lp, z_hat_lo]))      # distribution at lower bin
dist_hi = listener(cat([lp, z_hat_hi]))      # distribution at upper bin
```

The RB task loss gradient w.r.t. z_k uses the finite-difference approximation:
```python
# Compute task losses at both bin outputs
L_lo = actor_loss(dist_lo) + critic_loss(new_value)   # loss at lower bin
L_hi = actor_loss(dist_hi) + critic_loss(new_value)   # loss at upper bin

# Scale trick: proxy loss that gives g_RB_k = (L_hi - L_lo) / delta via autograd
rb_scale = (L_hi - L_lo).detach() / delta             # (B, z_dim)
rb_proxy_loss = (rb_scale * z).sum(dim=-1).mean()
```

### 3.2 All-dims-joint approximation (fast, recommended first)

The two-path approach above moves ALL z_dim dimensions to their lo/hi bins
simultaneously. This gives an approximate RB gradient:

```
g_approx_RB_k ≈ [L(ẑ_hi_all) − L(ẑ_lo_all)] / δ_k
```

where `ẑ_hi_all` moves ALL dims to their hi bin, not just k. This conflates the
gradient from different dims but requires only 2 additional forward passes total
(vs. 2·z_dim for per-dim RB). For z_dim=3: 2 passes vs. 6.

**Recommended implementation order:**
1. All-dims-joint (2 passes): fast, approximate, sufficient for initial ablation
2. Per-dim (2·z_dim passes): exact, expensive, only if joint shows clear benefit

### 3.3 Integration with the training loop

The RB task gradient replaces the STE gradient for the SPEAKER only (listener and
critic still use STE z_hat via the existing code path). Pseudocode:

```python
if config.use_rb_gradient:
    # Two extra forward passes at lo/hi bin centres
    L_lo, L_hi = compute_task_loss_at_bin_centres(z_new, ...)
    rb_scale = (L_hi - L_lo).detach() / effective_delta   # (B, z_dim)
    rb_task_loss = (rb_scale * z_new).sum(dim=-1).mean()
    total_loss = rb_task_loss + sc_rate_loss + dither_loss
    # Note: listener and critic losses are NOT included in total_loss here
    # (their gradients flow via the standard STE path instead)
    # ... or: include RL losses for listener/critic but use RB only for speaker
else:
    # Standard STE path
    total_loss = actor_loss + critic_loss + sc_rate_loss + dither_loss
```

**Design choice:** Whether to use RB for the entire actor+critic loss or just the
actor loss (the part that directly affects the speaker). The critic loss gradient
w.r.t. z is zero (critic takes state = [lp, goal], not z_hat), so RB is irrelevant
for critic; only the actor loss matters.

### 3.4 MAPPOConfig additions

```python
# P4 — Rao-Blackwell gradient estimator
use_rb_gradient: bool = False    # replace STE task gradient with RB
rb_mode: str = "joint"           # "joint" (2 passes) | "per_dim" (2*z_dim passes)
```

CLI flags: `--use_rb_gradient`, `--rb_mode`.

---

## 4. Interaction with Per-Channel δ (P1)

The RB gradient for the task loss w.r.t. z_k is:
```
g_RB_k = [L(ẑ_hi_k) − L(ẑ_lo_k)] / δ_k
```

If δ_k is a learnable parameter (P1), the gradient of `rb_task_loss` w.r.t. δ_k flows
through `1/δ_k` in the scale. This is in addition to the existing gradient from the
magnitude loss and the STE noise path.

**Hypothesis:** RB may help P1 by providing a cleaner gradient signal for δ_k adaptation.
The STE task gradient to δ_k has high variance (∂L/∂ẑ_k · (u_k − 0.5), zero-mean noise).
The RB task gradient to δ_k is more structured: `−(L(ẑ_hi_k) − L(ẑ_lo_k)) / δ_k²`.

This is a potential P1×P4 synergy — but only worth testing if P1 standalone (E53) shows
insufficient δ_k differentiation (Q1 from PILLAR_P1_v2.md §7).

---

## 5. Interaction with Histogram Rate Loss (P2)

The existing `source_coding_rate_loss` is already an RB gradient for L = rate. Adding
RB for the task loss creates a fully-RB training loop for Phase 1:

```
total_gradient_k = λ · g_RB_rate_k + g_RB_task_k
                 = λ · (R_hi_k − R_lo_k) / δ_k + (L_task(ẑ_hi) − L_task(ẑ_lo)) / δ_k
```

Both terms share the same (L_hi − L_lo)/δ structure. This symmetry is clean and
theoretically satisfying. The two objectives compete: rate loss pushes z_k to
lower bins (reduce R), task loss pushes z_k to whichever bin the listener prefers.

**Phase 2 note:** In Phase 2 the dither loss is also present. The RB gradient for the
dither loss is `[H_dither(frac_hi) − H_dither(frac_lo)] / δ_k`. Since
`frac_hi − frac_lo = 1` (adjacent bins), this simplifies to the gradient of
H_binary across a full bin — deterministic, not random. The dither loss is already
effectively a "RB" computation (it uses the frac analytically, not a random sample).
So P4 mainly adds value in **Phase 1** via the task-loss RB.

---

## 6. Failure Modes

### 6.1 L(ẑ_hi) ≈ L(ẑ_lo) at convergence

When the policy has converged (SR ≈ 1), the task loss is small and both bin centres
give similar action distributions → L(ẑ_hi) ≈ L(ẑ_lo) → g_RB_task_k ≈ 0.

This is CORRECT behaviour: at convergence, the task gradient for the speaker should
be near zero (the speaker cannot further improve the already-optimal policy).
The compression signal then dominates. No fix needed.

### 6.2 Listener sensitivity when ẑ_hi − ẑ_lo = δ is large

If δ = 1.0 and a bin boundary is at z_k = 0.5·δ = 0.5, then ẑ_lo = 0 and ẑ_hi = δ.
The listener input changes by δ = 1.0, which may cause the listener to produce very
different action distributions → high (L_hi − L_lo). This large signal is CORRECT
(bins that are far apart produce very different actions) but may cause large gradient
steps if not clipped.

**Mitigation:** Apply `max_grad_norm` clipping to the RB proxy loss gradient (already
done via the global clip). Monitor `rb_task_loss` magnitude.

### 6.3 Two forward passes double the per-minibatch compute time

At B=1024, the listener forward pass is a 2-layer MLP with hidden=64. The cost is
negligible compared to the rollout collection time. Not a practical concern.

### 6.4 Bias if ẑ_lo/ẑ_hi use other-dims at STE values

In the all-dims-joint approximation, L(ẑ_lo_all) moves ALL dims to their lo bin
simultaneously. This conflates cross-dim interactions. The true per-dim RB should
fix other dims at their expected value (which is just z_k itself, as shown in §1.3 derivation
for the Schuchman property: E[ẑ_k] = z_k).

For the per-dim RB, other dims should be fixed at `z_j` (not at their ẑ_lo or ẑ_hi),
which means using the SAME listener input for other dims in both the lo and hi passes
for dim k. This requires 2·z_dim passes but is unbiased per-dimension.

---

## 7. Ablation Matrix

### 7.1 Primary: STE vs RB convergence speed

| Config | Estimator | Total timesteps | Primary metric |
|---|---|---|---|
| STE baseline (≈ E09) | STE | 2M | Steps to SR=0.99 |
| RB-joint | RB (all-dims-joint, 2 passes) | 2M | Steps to SR=0.99 |
| RB-per-dim | RB (per-dim, 2·z_dim passes) | 2M | Steps to SR=0.99 |

**Primary claim:** RB reaches SR=0.99 in fewer updates than STE. If the advantage
exists, it should appear in the Phase 1 convergence curve.

### 7.2 Final performance: does RB improve H_joint?

| Config | Estimator | Phase 2 | H_joint at SR=1 |
|---|---|---|---|
| STE + P2 (E52) | STE | λ_dither=5e-3 | 2.094 ± 0.047 |
| RB-joint + P2 | RB | λ_dither=5e-3 | TBD |

Prediction: H_joint ≈ same. RB reduces variance during Phase 1 but the final
code (z distribution at convergence) should be similar. P2 dither compression is
what drives H_joint reduction, not the gradient estimator.

### 7.3 P1×P4: RB gradient with per-channel δ (conditional)

Only run if P1 standalone (E53) fails to differentiate δ_k. If Q1 from PILLAR_P1_v2.md
is answered negatively (task gradient too noisy), RB may provide cleaner δ_k signal.

| Config | Estimator | δ | Primary metric |
|---|---|---|---|
| P1 STE (E53/E54) | STE | per-channel | δ_k differentiation |
| P1 RB-joint | RB | per-channel | δ_k differentiation |

---

## 8. Figure Plan (F26–F28)

| Figure | Description | Data source |
|---|---|---|
| F26 | Phase 1 convergence: SR vs timesteps for STE / RB-joint / RB-per-dim | metrics.csv SR column |
| F27 | Gradient variance per update: Var(g_k) STE vs RB (3 speaker dims) | custom variance logging |
| F28 | Final H_joint: STE+P2 vs RB+P2 (does RB change the compression floor?) | post_hoc aggregate.json |

**F26** is the headline P4 figure: does RB speed up learning?

**F27** requires logging the variance of the speaker gradient norm across minibatches
within an update (a new diagnostic, similar to `log_grad_decomp` in F14). This could
be expensive to implement; may be omitted if F26 is sufficient.

**F28** closes the claim that RB is purely a variance-reduction technique and does
not change the final operating point.

---

## 9. Open Questions

**Q1 — Does RB produce a correct gradient for PPO's ratio computation?**

The PPO actor loss uses `ratio = exp(new_logp − old_logp)`. In the RB path, we
compute actor loss at `ẑ_lo` and `ẑ_hi`, not at `ẑ_STE`. But `old_logp` was computed
at `ẑ_old` (from the rollout buffer, STE-sampled). The ratio therefore involves a
mismatch:

```
ratio_lo = π(a | ẑ_lo) / π_old(a | ẑ_old)
ratio_hi = π(a | ẑ_hi) / π_old(a | ẑ_old)
```

These are NOT standard PPO importance weights (which require `ẑ_new ≈ ẑ_old` to be
on-policy). For bins far from `ẑ_old`, the ratio can be large, causing gradient
instability.

**Resolution path:** The proxy-loss formulation (Scale-trick `rb_scale.detach() * z`)
bypasses this issue: the gradient w.r.t. z_k is `rb_scale_k`, not the ratio. The
`rb_scale = (L_hi − L_lo).detach() / δ` is computed without any PPO ratio — it is a
pure score function gradient. This avoids the ratio issue entirely.

**Q2 — Is the all-dims-joint RB a sufficient approximation?**

If dimensions are correlated in the listener (which they are — the listener sees
[lp, ẑ_0, ẑ_1, ẑ_2] as a single vector), moving all dims simultaneously gives a
gradient for dim k that is confounded with dim j's contribution. The per-dim RB
is unbiased but 3× more expensive.

**Resolution path:** Run both and compare convergence speed. If all-dims-joint
matches per-dim at similar training cost, use joint as the default.

**Q3 — Does RB interact with the advantage normalization in PPO?**

PPO normalizes advantages per minibatch. The RB loss uses raw `L_hi − L_lo` which
may not be normalized. Should `L_hi − L_lo` be normalized by a running advantage
estimate? Needs careful implementation.

---

## 10. Implementation Checklist

- [ ] Add `use_rb_gradient: bool` and `rb_mode: str` to `MAPPOConfig`
- [ ] Add `--use_rb_gradient`, `--rb_mode` to `train.py` CLI
- [ ] In `trainer.py` update(): implement two-path forward (z_hat_lo, z_hat_hi)
- [ ] Compute `rb_scale = (L_hi − L_lo).detach() / effective_delta`
- [ ] Add `rb_proxy_loss = (rb_scale * z_new).sum(dim=-1).mean()` to total_loss
- [ ] Add `rb_task_loss` to metrics.csv
- [ ] Add P4 experiment stages to `run_sc_experiments.py`
- [ ] Add unit test: g_RB_k ≈ g_STE_k in expectation (1000 samples, atol=0.1)
- [ ] Add unit test: Var(g_STE) > Var(g_RB) for fixed z (at least dim-wise)

---

## 11. Experiment Registry

| ID | Name | Status | Config | Seeds | Primary metric |
|---|---|---|---|---|---|
| E57 | p4_ste_baseline | PLANNED | STE, sd, sc, phase1_sr=0.99 | 5 | steps_to_sr99 |
| E58 | p4_rb_joint | PLANNED | RB-joint, sd, sc, phase1_sr=0.99 | 5 | steps_to_sr99 |
| E59 | p4_rb_per_dim | PLANNED | RB-per-dim, sd, sc, phase1_sr=0.99 | 5 | steps_to_sr99 |
| E60 | p4_rb_joint_p2 | PLANNED | RB-joint + λ_dither=5e-3 | 5 | H_joint at SR=1 |
| E61 | p4_rb_p1_joint | PLANNED (conditional) | RB-joint + per-channel δ | 5 | δ_k differentiation |

**E57 serves as the STE reference for P4;** it may overlap with E53 from P1 if configs
match. Consolidate at planning time.

---

## 12. Success Criteria

**Primary:** E58/E59 reaches SR=0.99 in fewer updates than E57, with statistical significance
(p<0.05, Wilcoxon rank-sum over 5 seeds).

**Secondary:** E60 H_joint ≤ 2.094 bits (P2 best). Even if RB does not improve final
H_joint, it should not make it worse.

**Minimum bar for paper:** F26 shows clear convergence acceleration for at least the
RB-joint variant, confirming Thm 6 empirically.

**Failure criterion:** If E58 does not converge faster than E57 (within 1-sigma), P4
is a null result for this environment. The theorem may still be correct but the variance
reduction is dominated by minibatch averaging at B=1024.
