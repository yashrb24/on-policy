# Pillar P3 — TPDF Non-Subtractive Dithering

**Status:** STUB — design not yet started (implement after P4)  
**Pillar order:** P2 → P1 → P4 → **P3**  
**Paper reference:** "Stochastic Quantisation via Dithering" §Thm 5

---

## Overview

DDCL-SD (subtractive dither) requires that the same PRNG seed is shared between sender and receiver at inference time — a deployment constraint. P3 implements TPDF non-subtractive dithering (ε = ε₁ + ε₂), which removes this constraint because the receiver does not need to know the dither noise: the quantisation error statistics are still well-behaved (Thm 5).

**Note:** The existing `DDCL_NSD` class already implements the dual-path STE version of non-subtractive dithering (Algorithm 1 of the proposal). P3 is thus partly done at the channel level. The pillar is about verifying the deployment-realistic setup: shared PRNG is not required, and the mathematical properties are preserved.

---

## Key Questions to Resolve at Design Time

- What exactly is different in the NSD forward pass at deployment vs. training?
- How do we test "no shared PRNG" in a unit test context?
- What is the variance formula for TPDF dither (Thm 5)? Verify MATH-002 (δ²/4 for NSD).
- Does P3 interact with P1 (per-channel δ) or P2 (entropy model) in any non-trivial way?
- Does removing the shared-PRNG constraint change the gradient estimator in any way?

---

## Expected Ablations

- SD vs NSD: rate-distortion frontier comparison
- SD vs NSD: evaluation under deployment constraint (no shared PRNG at test time)
- Interaction: P3 + P2 combined (does entropy model work equally well for NSD messages?)

---

## Document Update Instructions

When implementation begins: replace this stub with a full design doc following the same structure as `PILLAR_P2.md` (§1–§12). Mark Status as IN PROGRESS at the top.
