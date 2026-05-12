# Pillar ALL — Full Integration (All Four Pillars Combined)

**Status:** PLANNED — design drafted 2026-05-12; gate: all individual pillar experiments (E53–E63) done.
**Pillar order:** P2 → P1 → P4 → P3 → **ALL**
**Paper reference:** "Stochastic Quantisation via Dithering" §Results (headline claim)
**Depends on:** All of P1 (E53–E56), P4 (E57–E60), P3 (E62–E63) complete and conclusions validated.

---

## 1. Motivation

Each pillar addresses a distinct limitation of the baseline (E09, fixed δ, STE, SD channel):

| Pillar | Mechanism | What it fixes |
|--------|-----------|---------------|
| P2 | Histogram rate + Phase 2 dither | Teaches speaker to minimise H(m) |
| P1 | Per-channel δ_k | Lets dimensions allocate bits asymmetrically |
| P4 | Rao-Blackwell gradient | Reduces STE variance for speaker task gradient |
| P3 (NSD variant) | TPDF dithering + NSD consistency | Eliminates train/deploy distribution gap |

The All-Pillars integration asks: do these benefits **compose**? Is the combined system
strictly better than any individual pillar, and does the combination surface interactions
that single-pillar studies cannot reveal?

This is the **headline experimental claim** of the paper. If P1+P2+P4 at SR=1 achieves
H_joint < 1.813 bits (the theoretical H(G) floor), the system is overcoding; if it
approaches 1.813 bits, we have near-optimal communication.

---

## 2. Two Variants

### 2.1 SD Variant (E64)

All four pillars, SD channel:

```
P1: learn_delta=True          # δ_k = softplus(α_k), z_dim free params
P2: use_source_coding=True    # histogram rate loss, Phase 2 dither
    loss_comms_mode="magnitude"
    phase1_sr_threshold=0.99
    lambda_dither=5e-3
P4: use_rb_gradient=True      # RB-joint speaker task gradient (Phase 1 only)
    rb_mode="joint"
P3: channel="sd"              # SD: distributional consistency (Schuchman A.1)
    deploy_eval=False         # No P3 eval flag during training (PPO ratio mismatch)
```

SD channel has distributional consistency (Schuchman Theorem A.1): the training
and deployment marginal distributions are the same, so the train/deploy gap does
not arise at the distributional level. Deploy_eval is recorded post-hoc via
`eval_deploy.py` (E62 approach), not during training.

**Config flags:**
```bash
--channel sd
--use_source_coding --loss_comms_mode magnitude --lambda_comms 5e-4
--learn_delta --lr_delta 1e-4
--phase1_sr_threshold 0.99 --lambda_dither 5e-3
--use_rb_gradient --rb_mode joint
--total_timesteps 3000000
```

### 2.2 NSD Variant (E65)

All four pillars, NSD channel (Pillar 3 fully active):

```
P1: learn_delta=True
P2: use_source_coding=True, loss_comms_mode="magnitude"
    phase1_sr_threshold=0.99, lambda_dither=5e-3
P4: use_rb_gradient=True, rb_mode="joint"
P3: channel="nsd"             # TPDF dither: sample-consistent at train and deploy
                              # z_hat = z_hat_true always (NSD construction)
```

The NSD channel uses triangular-PDF dither (ν_k ~ Tri(-δ_k, 0, δ_k)). By the NSD
construction (PILLAR_P3_v2.md §4), z_hat = z_hat_true at all times — training and
deployment are sample-consistent, not just distributionally consistent. This is the
strongest deployment guarantee any variant offers.

NSD changes the dither variance: Var(ẑ-z) = δ²/4 (vs δ²/3 for SD). The RB gradient
formula (P4) remains valid with the same `[L_hi - L_lo] / δ` structure, but the two
bin-centre neighbours differ in their marginal noise distribution. See PILLAR_P4_v2.md
§Cross-pillar for the TPDF caveat.

**Config flags:**
```bash
--channel nsd
--use_source_coding --loss_comms_mode magnitude --lambda_comms 5e-4
--learn_delta --lr_delta 1e-4
--phase1_sr_threshold 0.99 --lambda_dither 5e-3
--use_rb_gradient --rb_mode joint
--total_timesteps 3000000
```

---

## 3. Hypotheses

### H-ALL-1: Composition is additive or super-additive

If P1, P2, P4 each reduce H_joint independently by ΔH₁, ΔH₂, ΔH₄, the combined
system should achieve at least ΔH₁ + ΔH₂ + ΔH₄. Super-additivity is possible if,
for example, P1 creates a well-separated dimensional structure that makes the P2
dither loss more effective.

**Falsification threshold:** If H_joint(E64) > H_joint(best single-pillar), the pillars
interfere destructively — this must be reported even if it falsifies the paper claim.

### H-ALL-2: NSD matches or beats SD on H_joint

The NSD dither noise is slightly larger (Var = δ²/4 vs δ²/3), which could push z_k
slightly toward bin centres (lower H(m|goal)), but also increases the marginal spread
of m (higher H(m|goal) from more dispersed quantisation). The net effect on H_joint is
an empirical question.

**Expected outcome:** NSD ≈ SD on H_joint, with NSD strictly better on deploy SR
(measured separately via `eval_deploy.py`).

### H-ALL-3: SR degrades less than 1% from baseline

The compound system must not hurt task performance. We require SR ≥ 0.99 (same as
Phase 1 threshold used to trigger Phase 2).

---

## 4. Sequencing Gate

**Do not run E64/E65 until:**

1. E53 (global δ) and E54 (per-channel δ) complete → P1 conclusion validated
2. E57 (STE baseline) and E58 (RB-joint) complete → P4 conclusion validated
3. E52 (Phase 2 dither, best λ) already complete (DONE — used as P2 anchor)
4. E47/E48 (NSD training, C16) already complete (DONE)

The gate exists because E64/E65 configs are chosen based on the best hyperparameters
from individual pillar studies. Running before the pillars are validated risks choosing
suboptimal λ_dither, lr_delta, or rb_mode.

**Conditional on P1×P4 interaction (E61):** If E54 vs E58 shows that P1 and P4
interact (non-additive), run E61 (P1+P4, no P2 or P3) before E64 to isolate the
interaction term. E61 is not currently in the runner — add manually if needed.

---

## 5. Metrics to Report

For each variant (E64 SD, E65 NSD), report at training convergence (mean ± std, 5 seeds):

| Metric | Symbol | Source |
|--------|--------|--------|
| Task success rate | SR | `metrics.csv` |
| Joint message entropy | H_joint | `post_hoc_coding.py` |
| Shannon gap | H_joint − H(G) | derived |
| Per-dim entropies | H(m_0), H(m_1), H(m_2) | `post_hoc_coding.py` |
| Total correlation | TC = Σ H(m_k) − H_joint | `post_hoc_coding.py` |
| Dither entropy | H(m_k|goal) per k | `post_hoc_coding.py` |
| Deployment SR gap | SR_train − SR_deploy | `eval_deploy.py` (separate) |

The primary figure (F-ALL) is a side-by-side bar: H_joint for E09 (fixed δ, STE,
SD), E52 (P2 only), E55 (P1+P2), E64 (all SD), E65 (all NSD). Secondary panel:
deployment SR gap for SD vs NSD.

---

## 6. Experiment Runner Entry

The `ALL_PILLARS` stage is added to `run_sc_experiments.py` but **gated**. Before
the gate check passes, the stage prints a warning and yields zero configs.

```
python -m onpolicy.envs.toyproblem.experiments.run_sc_experiments --stage ALL_PILLARS
```

Gate flag override (for manual use after validation):
```
python -m onpolicy.envs.toyproblem.experiments.run_sc_experiments \
    --stage ALL_PILLARS --override_gate
```

Both SD (E64) and NSD (E65) variants run under this stage: 2 configs × 5 seeds = 10 runs.

---

## 7. Cross-Pillar Interactions

### P1 × P2

Per-channel δ_k changes the Phase 2 dither loss landscape. The dither entropy
term `h_binary(frac(z_k/δ_k))` now uses a different δ_k per dimension. The gradient
`∂L_dither/∂z_k = (2p_k − 1) / (δ_k · p_k · (1−p_k))` scales inversely with δ_k:
large δ_k dims (coarse bins) get weaker dither push. This is intentional — coarse
dims have fewer bits to compress. **No special handling needed.**

### P1 × P4

The RB gradient formula is `g_RB_k = [L_hi − L_lo] / δ_k`. With per-channel δ_k,
each dimension uses its own learned width in the denominator. The joint-mode
approximation computes z_hat_lo = (m − 0.5)·δ and z_hat_hi = (m + 0.5)·δ using
the full δ vector — all dims shift simultaneously. This is an approximation (not
per-dimension exact), but each dimension's own δ_k still scales its effective
gradient. **Likely interaction:** large-δ dims get smaller g_RB_k, reducing their
contribution to the total policy gradient, which is consistent with their reduced
information content.

### P1 × P3

When `channel="nsd"` and `learn_delta=True`, the NSD channel uses the per-channel
δ vector for both the TPDF dither (`ν_k ~ Tri(-δ_k, 0, δ_k)`) and the quantisation
step. The NSD consistency property (z_hat = z_hat_true) holds independently for each
dimension. **No coupling beyond what each pillar individually requires.**

### P2 × P4

RB gradient is Phase 1 only; dither loss is Phase 2 only. They do not overlap in
time. The only coupling is through the shared histogram: the RB gradient in Phase 1
changes how quickly z converges, which changes what the histogram records at Phase 2
onset. **No gradient-level interaction.**

### P3 × P4

TPDF dither changes the marginal distribution of m_k at a given z_k compared to
SD. The RB formula assumes `[L(ẑ_hi) − L(ẑ_lo)] / δ_k` is a good estimator of
`∂E_m[L] / ∂z_k`. For SD (uniform dither), this is exact. For NSD (triangular
dither), the two-neighbour approximation is still a valid finite-difference but
introduces a small TPDF-specific bias (the probability mass at the boundary differs
from the SD case). See PILLAR_P4_v2.md §Cross-pillar for the derivation.
**Expected effect:** small, sub-dominant compared to variance reduction.

---

## 8. Expected Results Summary

| Experiment | SR | H_joint (bits) | Shannon gap |
|------------|-----|----------------|-------------|
| E09 (baseline) | ~1.00 | ~3.58 | ~1.77 |
| E52 (P2 only) | 1.00 | ~2.09 | ~0.28 |
| E55 (P1+P2) | 1.00 | TBD | TBD |
| E64 (all SD) | ≥0.99 | **TBD** | **TBD** |
| E65 (all NSD) | ≥0.99 | **TBD** | **TBD** |
| H(G) floor | — | 1.813 | 0.000 |

The paper's primary claim is that E64 or E65 achieves the lowest Shannon gap among all
variants, at SR ≥ 0.99. If both achieve comparable H_joint, the NSD variant is preferred
because it additionally eliminates the deployment gap.

---

## 9. Failure Modes

**If E64/E65 SR < 0.99:** The combined comms penalties or RB gradient is destabilising
Phase 1. Diagnostic: ablate one pillar at a time starting from E52 (which achieves SR=1).

**If H_joint(E64) > H_joint(E52):** P1 or P4 is interfering with Phase 2 compression.
Diagnostic: run E61 (P1+P4 without P2) to check if the interference is P1×P2 or P4×P2.

**If H_joint(E65) >> H_joint(E64):** NSD noise is too large for the learned δ_k values.
Diagnostic: check whether NSD δ_k values converge to smaller values (compensating for
larger noise variance) or whether the optimizer fails to find them.
