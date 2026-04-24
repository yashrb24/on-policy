# Pillar P1 — Per-Channel δ (Adaptive Quantisation Width)

**Status:** STUB — design not yet started (implement after P2)  
**Pillar order:** P2 → **P1** → P4 → P3  
**Paper reference:** "Stochastic Quantisation via Dithering" §Thm 4

---

## Overview

P1 replaces the global scalar `δ` with per-dimension quantisation widths `δ_k`, allowing the speaker to allocate precision unevenly across message dimensions.

**Two-stage implementation:**
1. **Heuristic:** fixed per-channel scalars `{δ_k}` as learnable parameters with no additional structure
2. **Learned:** `δ_k = softplus(α_k)` with gradient flowing through the channel (Thm 4 of the proposal paper)

---

## Key Questions to Resolve at Design Time

- What is the correct gradient for `δ_k` given the STE on `z/δ_k`?
- Does `δ_k` need to be bounded away from 0? What initialisation?
- How does `L_comms = Σ_k log₂(|z_k|/δ_k + 1)` change when δ_k is per-dimension? (The Jensen bound still applies dimension-wise; they sum.)
- Interaction with P2: if both are active simultaneously, which term drives `δ_k`?
- How does learned `δ_k` interact with Schuchman's theorem (dither must be TPDF on `[−δ/2, δ/2]`)?

---

## Expected Ablations

- Fixed vs learned δ_k
- Shared δ vs per-dimension δ
- P1 alone vs P1 + P2 (combined pillar)
- Rate–distortion frontier: does uneven allocation reach the Pareto frontier faster?

---

## Document Update Instructions

When implementation begins: replace this stub with a full design doc following the same structure as `PILLAR_P2.md` (§1–§12). Mark Status as IN PROGRESS at the top.
