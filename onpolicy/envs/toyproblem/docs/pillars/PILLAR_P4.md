# Pillar P4 — Rao-Blackwell Gradient Estimator

**Status:** STUB — design not yet started (implement after P1)  
**Pillar order:** P2 → P1 → **P4** → P3  
**Paper reference:** "Stochastic Quantisation via Dithering" §Thm 6

---

## Overview

The STE (straight-through estimator) used in baseline DDCL propagates gradients through quantisation by ignoring the step discontinuity. P4 replaces this with an analytical Rao-Blackwell gradient that marginalises over the two candidate quantisation bins:

```
g_RB = p_a · ∇L_a + p_b · ∇L_b
```

where `p_a, p_b` are the probabilities of landing in the lower vs upper bin given the dither, and `∇L_a, ∇L_b` are the losses at those two outcomes. This reduces variance compared to STE (Thm 6 of the proposal paper).

---

## Key Questions to Resolve at Design Time

- What is `p_a` and `p_b` as a function of `z`, `δ`, and the dither distribution?
- How does `g_RB` interact with PPO's probability ratio clipping? (PPO uses importance sampling — does the two-bin gradient change the ratio computation?)
- Is `g_RB` compatible with both SD and NSD dithering?
- What variance reduction does `g_RB` provide empirically vs. STE? (Primary ablation metric: `grad_variance_actor`.)
- Does lower gradient variance translate to faster convergence or better final performance in this environment?
- Interaction with P2: if the loss includes `-log₂ q_φ(m)`, does the Rao-Blackwell apply to this term as well?

---

## Expected Ablations

- STE vs Rao-Blackwell: convergence speed, final success rate, gradient variance
- Rao-Blackwell with SD vs NSD (P4 + P3 combined)
- Rao-Blackwell + entropy model (P4 + P2 combined): does lower variance improve q_φ convergence?

---

## Document Update Instructions

When implementation begins: replace this stub with a full design doc following the same structure as `PILLAR_P2.md` (§1–§12). Mark Status as IN PROGRESS at the top.
