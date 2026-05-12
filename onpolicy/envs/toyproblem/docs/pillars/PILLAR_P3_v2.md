# Pillar P3 — TPDF Non-Subtractive Dithering: Deployment Consistency

**Status:** IN DESIGN — v2 plan drafted 2026-05-12; implementation partially done (DDCL_NSD complete); experiments partially done (E47/E48 done; E62/E63 pending).
**Pillar order:** P2 → P1 → P4 → **P3**
**Paper reference:** "Stochastic Quantisation via Dithering" §Thm 5
**Depends on:** P2 closed (E01–E52 done); C16 confirmed NSD ≈ SD empirically (E47/E48)

---

## 1. The Core Question: Does NSD's Deployment Consistency Translate to a Measurable Advantage?

### 1.1 The deployment gap for subtractive dithering (SD)

The SD channel uses the same dither noise ε at sender and receiver ("shared PRNG"):
- Sender: `z' = z + ε;  m = floor(z'/δ);  transmits m`
- Receiver: `z_hat_deploy = (m + 0.5)·δ − ε`  (subtracts its copy of ε)

At **training** time, DDCL_SD uses the Schuchman equivalence (Thm A.1) and replaces the
real pipeline with a simpler STE: `z_hat_STE = z + e`, where `e ~ U(−δ/2, δ/2)` is a
fresh noise sample (not the shared ε from the real pipeline).

The listener during training therefore receives `z_hat_STE = z + e`, not the deployment
value `z_hat_deploy = (m+0.5)·δ − ε`. These are **distributionally equivalent** (same
marginal) but **not sample-consistent** — for any given rollout step, `z_hat_STE ≠
z_hat_deploy`.

The practical consequence: in a deployed system, the listener receives bin-center-minus-ε
values it has never seen exactly during training. Schuchman's theorem guarantees the
marginals match, so performance is preserved in expectation — but there is a philosophical
(and potentially practical) mismatch.

### 1.2 NSD eliminates the deployment gap

Non-subtractive dithering uses TPDF noise `ν = u₁ + u₂` (uᵢ ~ U(−δ/2, δ/2)) and the
receiver outputs the bare bin centre without subtracting ν:
- Sender: `z' = z + ν;  m = floor(z'/δ);  transmits m`
- Receiver: `z_hat_deploy = (m + 0.5)·δ`  (no shared ν required)

At **training** time, DDCL_NSD computes the same bin centre:
```python
# channels.py DDCL_NSD.forward()
m = torch.floor((z + nu) / d)          # real pipeline, under no_grad
z_hat_true = (m + 0.5) * d             # exact bin centre
z_hat = z + (z_hat_true - z).detach()  # STE: forward = z_hat_true, gradient passes through
```

The STE forward value `z_hat` equals `z_hat_true = (m+0.5)·δ` exactly.
The deployment value is also `(m+0.5)·δ`.

**NSD is sample-consistent: the listener sees the same value at training time and
deployment time, for every rollout step, with no shared PRNG needed.**

### 1.3 What Theorem 5 states

Theorem 5 (Schuchman/Widrow for TPDF dither) establishes the NSD quantisation error
statistics:

```
E[ẑ − z | z] = 0          (first-order Schuchman: unbiased)
Var(ẑ − z)  = δ²/4        (second-order: signal-independent variance)
```

For comparison, SD achieves:
```
E[ẑ − z | z] = 0          (unbiased via subtraction)
Var(ẑ − z)  = δ²/12       (uniform quantisation noise)
```

**NSD has 3× higher quantisation noise variance than SD.** The tradeoff: SD needs shared
PRNG (strong deployment assumption); NSD accepts higher noise variance in exchange for
deployment simplicity.

### 1.4 Derivation of NSD variance (Theorem 5)

The TPDF characteristic function satisfies Schuchman's second-order condition:
```
Φ_TPDF(2πk/δ) = sinc²(k) = 0   for all integer k ≠ 0
```
This ensures the quantisation error of `(z + ν)` is statistically independent of z.

The NSD error decomposes as:
```
ẑ − z  =  (m + 0.5)·δ − z
       =  ν  +  [(m + 0.5)·δ − (z + ν)]
       =  ν  +  e_q(z + ν)
```
where `e_q(z') = (m+0.5)·δ − z'` is the quantisation error of the dithered input.
By Schuchman's second-order condition, `e_q` is independent of z and uniform on
`[−δ/2, δ/2]`, so:

```
Var(ẑ − z) = Var(ν) + Var(e_q)  +  2·Cov(ν, e_q)
           = δ²/6  + δ²/12  +  0
           = 2δ²/12 + δ²/12
           = 3δ²/12 = δ²/4   ✓
```

The TPDF variance `Var(ν) = 2 × δ²/12 = δ²/6` adds to the SD noise floor `δ²/12`.

---

## 2. What Is Already Done

### 2.1 Implementation complete

`DDCL_NSD` in `channels.py` fully implements the NSD forward pass (§1.2 above):
- TPDF dither `ν = u₁ + u₂` sampled on every forward call
- Exact bin centre `z_hat_true = (m+0.5)·δ` computed under `no_grad`
- STE gradient path: `z_hat = z + (z_hat_true − z).detach()`
- `comms_loss`: same Jensen bound `log₂(|z|/δ + 1)` as SD (derivation carries over)
- `transmission_bits_per_elem`: `log₂(|m|+1)` from actual discrete m

Per-channel δ support (from P1) also already works for NSD: both `forward` and
`comms_loss` accept `delta: torch.Tensor | None = None`.

### 2.2 Empirical validation complete (C16)

**Conclusion C16:** NSD and SD achieve identical H_joint within noise (Δ < 0.03 bits).

| Experiment | Channel | Phase | H_joint (bits) | SR |
|---|---|---|---|---|
| E09 `sc_posthoc_mag` | SD | Phase 1 | 4.156 ± 0.344 | 1.000 |
| E47 `nsd_posthoc_mag` | NSD | Phase 1 | 4.137 ± 0.165 | 1.000 |
| E45 `sc_twophase_dither5e-4_v2` | SD | Phase 2 | 2.385 ± 0.174 | 1.000 |
| E48 `nsd_twophase_dither5e-4_v2` | NSD | Phase 2 | 2.413 ± 0.141 | 1.000 |

**C16 status:** Closed. Both phase 1 and phase 2 NSD match SD within 1 σ. The histogram
estimator and Phase 2 dither compression are channel-agnostic.

---

## 3. Why Does NSD ≈ SD Despite 3× Higher Noise? (C16 Explanation)

### 3.1 STE gradient paths are identical

Both SD and NSD use STE with slope 1:
- SD: `∂z_hat_SD/∂z = 1` (from `z_hat = z + e`)
- NSD: `∂z_hat_NSD/∂z = 1` (from `z_hat = z + (z_hat_true − z).detach()`)

The **speaker gradient** is identical in both channels. The NSD gradient noise is
the same as SD's — both have `∂L/∂z_k = ∂L/∂ẑ_k` with ẑ_k varying across rollout
steps due to the dither.

### 3.2 Listener input distributions differ but lead to same policy

SD listener receives `z_hat_SD ~ U(z − δ/2, z + δ/2)` (RPDF-shifted).
NSD listener receives `z_hat_NSD = (m + 0.5)·δ` for `m` determined by TPDF dither.
The conditional distribution of `z_hat_NSD | z` is NOT uniform — it is a mixture of two
point masses at the two reachable bin centres, with weights depending on `frac(z/δ)`.

Despite this difference in input distribution:
1. **Both are unbiased:** `E[z_hat] = z` for both (Schuchman 1st-order condition)
2. **Goal separation is large:** z_dim=3 goals are well-separated in z-space; the
   listener can tolerate δ²/4 noise as easily as δ²/12 noise when inter-goal distance
   ≫ δ
3. **Histogram is distribution-free:** `q_hist(m)` is built from actual m-counts
   regardless of whether m came from SD or NSD quantisation. Rate loss and Phase 2
   dither loss are identical in structure.

### 3.3 The gap NSD cannot close

NSD's 3× higher noise means the listener requires larger inter-goal z-separation to
achieve the same disambiguation as SD. In the current 6-goal environment, the speaker
learns sufficiently spread z values in Phase 1 (H_joint ≈ 4 bits > H(G) ≈ 1.8 bits),
so the noise difference is not binding. In a higher-z_dim or higher-entropy-goal
environment, NSD might require larger z magnitudes — and thus slightly higher rate cost.
This is not tested in the current experiment set.

---

## 4. The Remaining P3 Claim: Sample Consistency at Deployment

### 4.1 What "sample consistency" means

During a training rollout, each step `(z, m, ẑ)` is a triplet. The listener learns a
policy `π(a | ẑ)`. At deployment, for the same `(z, m)`, the listener receives:

| Channel | Training ẑ | Deployment ẑ | Sample-consistent? |
|---|---|---|---|
| SD | `z + e` (fresh STE noise) | `(m+0.5)·δ − ε` (shared dither) | **No** |
| NSD | `(m+0.5)·δ` (exact bin centre) | `(m+0.5)·δ` | **Yes** |

For SD, Schuchman (Thm A.1) guarantees the *marginals* of training ẑ and deployment ẑ
are the same, so the expected policy performance is preserved. But any individual
prediction at a specific z can use a different ẑ at test time than was seen in training.

For NSD, the listener is trained on bin centres — the exact values it will see at
deployment — with no approximation.

### 4.2 Predicted deployment evaluation result

| Experiment | Channel | Evaluation z_hat | Expected SR |
|---|---|---|---|
| E62 SD deployment | SD | `z_hat_deploy = (m+0.5)·δ − ε` | ≈ SR_train (Thm A.1) |
| E63 NSD deployment | NSD | `z_hat_true = (m+0.5)·δ` | = SR_train exactly (by construction) |

**For NSD (E63):** The deployment evaluation uses the same z_hat the listener saw
during training. SR should be identical to training SR (within rollout variance).
This is provable by inspection of the NSD forward pass.

**For SD (E62):** The deployment evaluation uses `z_hat_deploy = (m+0.5)·δ − ε`, which
differs from the STE value `z + e` used during training. Schuchman guarantees these
have the same marginal, so SR should not drop — but it cannot be equal by construction
(only by distributional equivalence). A small SR drop would indicate the listener is
overfitting to the STE input distribution.

**Prediction:** Both E62 and E63 will show SR ≈ SR_train, confirming C16's
channel-agnosticism. NSD provides a stronger (sample-level) guarantee.

---

## 5. Remaining Experiments

### 5.1 E62 — SD deployment evaluation (no new training)

Load checkpoints from E09 (`sc_posthoc_mag`, SD, Phase 1, 5 seeds). Evaluate
using deployment z_hat:

```python
# In act_and_value() or eval loop:
z_hat_train, ch_info = self.channel(z)          # as during training
z_hat_eval = ch_info["z_hat_deploy"]            # = (m+0.5)δ − ε (SD-specific)
dist_eval = self.listener(cat([listener_pos, z_hat_eval], dim=-1))
SR_deploy = eval_sr(dist_eval)
```

Report: `SR_train` vs. `SR_deploy`. The gap (if any) quantifies the SD deployment
inconsistency.

**Implementation note:** `ch_info["z_hat_deploy"]` is already computed by DDCL_SD.forward()
and stored in `info`. No changes to channels.py needed. A small flag `deploy_eval=True`
in `act_and_value()` to use `ch_info["z_hat_deploy"]` instead of `z_hat` is the only
addition required to `trainer.py`.

### 5.2 E63 — NSD deployment evaluation (no new training)

Load checkpoints from E47 (`nsd_posthoc_mag`, NSD, Phase 1, 5 seeds). Evaluate
using deployment z_hat:

```python
# In act_and_value() or eval loop:
z_hat_train, ch_info = self.channel(z)          # forward value = z_hat_true (NSD)
z_hat_eval = ch_info["z_hat_true"]              # = (m+0.5)δ (NSD-specific)
# Note: z_hat_train == z_hat_eval for NSD (by construction)
```

Report: `SR_train` vs. `SR_deploy`. Should be identical to within rollout noise.

**Why this is needed even if the result is obvious:** The deployment evaluation for NSD
provides the proof-of-concept that sample-consistent training (the DDCL_NSD design
goal) is empirically realised, not just theoretically guaranteed.

### 5.3 Why no new training experiments are needed for P3

P3's core claims are:
1. NSD is sample-consistent at deployment (provable from code, confirmed by E63)
2. NSD ≈ SD in task performance (confirmed by E47/E48, C16)
3. NSD does not require shared PRNG at deployment (provable by inspection of NSD.forward())

All three are either provable theoretically or already empirically confirmed. P3 is a
**theory + code verification pillar**, not a new training pillar. The experiments
(E62/E63) are lightweight evaluation runs on existing checkpoints.

---

## 6. Interaction with Other Pillars

### 6.1 P3 × P1 (per-channel δ for NSD)

Per-channel δ (Pillar 1) already works with NSD — both `DDCL_NSD.forward()` and
`DDCL_NSD.comms_loss()` accept `delta: torch.Tensor | None = None` and broadcast
correctly with a per-channel tensor. The deployment consistency argument still holds:
NSD with learnable per-channel `δ_k` outputs `z_hat_true_k = (m_k + 0.5)·δ_k` at
training time, and the deployment receiver outputs the same value.

**P3×P1 interaction: no new experiments required.** The per-channel δ capability is
transparent to P3's deployment claim.

### 6.2 P3 × P4 (RB gradient for NSD)

The Rao-Blackwell gradient formula (P4) was derived for the SD channel:
```
g_RB_k = [L(ẑ_hi_k) − L(ẑ_lo_k)] / δ_k
```
where `ẑ_lo_k = (m_lo + 0.5)·δ_k` and `ẑ_hi_k = (m_hi + 0.5)·δ_k` are bin centres.

For NSD, the listener **always** sees bin centres (it never sees STE noise). The RB
formula applies unchanged — the two candidate outputs are still `ẑ_lo` and `ẑ_hi` (the
only two bin centres reachable from `z_k`). The probability calculation uses the TPDF
probabilities (§6.2.1 below), but the finite-difference form `[L_hi − L_lo]/δ_k`
remains the correct gradient estimate for NSD as well.

**P3×P4 interaction:** If P4's RB gradient is implemented, it applies equally to SD and
NSD channels. No NSD-specific modification needed.

#### 6.2.1 TPDF bin probabilities for Rao-Blackwell

For NSD with TPDF dither, the probability that `z_k` maps to the two reachable bins
differs from SD (which uses RPDF). For frac_k = `frac(z_k/δ_k)`:

```
TPDF (NSD), frac_k < 0.5:
  p_lo = (0.5 − frac_k)²      p_hi = 1 − p_lo
TPDF (NSD), frac_k ≥ 0.5:
  p_hi = (frac_k − 0.5)²      p_lo = 1 − p_hi
```

(The TPDF density near a bin boundary is quadratic, not linear.) However, the
RB gradient derivation in PILLAR_P4_v2.md §1.3 shows the finite-difference form
`[L_hi − L_lo]/δ_k` arises from the RPDF score function. For NSD with TPDF, the
exact RB formula differs in the probability terms — but the proxy-loss scale trick
used in P4 is an approximation regardless (the all-dims-joint mode is already approximate).
The bias is small and can be accepted for the initial ablation.

### 6.3 P3 × P2 (histogram rate loss)

C16 directly confirms P3×P2 interaction: `q_hist` works identically for NSD messages.
The histogram is built from discrete message integers `m`; whether `m` came from SD or
NSD quantisation is irrelevant to `q_hist`. The Rate Phase 2 dither loss is also
channel-agnostic (it uses `frac(z/δ)`, not the channel noise). **Closed — no new
experiments.**

---

## 7. Implementation Plan

### 7.1 What is already implemented (no changes needed)

| Component | Status |
|---|---|
| `DDCL_NSD.forward()` | Done — outputs `z_hat_true = (m+0.5)·δ`, stored in `info["z_hat_true"]` |
| `DDCL_NSD.comms_loss()` | Done — same Jensen bound as SD |
| `DDCL_NSD.transmission_bits_per_elem()` | Done — `log₂(|m|+1)` |
| Per-channel δ support | Done — `delta: torch.Tensor | None = None` accepted |
| E47/E48 training runs | Done — 5 seeds each, SR=1.000 |

### 7.2 Addition needed: deployment eval flag in trainer.py

A single flag `deploy_eval: bool` in `act_and_value()` (or a separate `eval_with_deploy()` method) to route the listener input through `z_hat_deploy` (SD) or `z_hat_true` (NSD) instead of the STE `z_hat`:

```python
def act_and_value(
    self,
    listener_pos: torch.Tensor,
    speaker_pos: torch.Tensor,
    deploy_eval: bool = False,
) -> tuple[...]:
    ...
    z_hat, ch_info = self.channel(z, _act_delta)

    if deploy_eval and self.config.channel == "sd":
        listener_input = ch_info["z_hat_deploy"]
    elif deploy_eval and self.config.channel == "nsd":
        listener_input = ch_info["z_hat_true"]
    else:
        listener_input = z_hat   # training path (STE)

    dist = self.listener(torch.cat([listener_pos, listener_input], dim=-1))
    ...
```

This is the only code change required for E62/E63. The training loop is unchanged.

### 7.3 Eval runner addition

A small evaluation script (or a CLI flag `--deploy_eval` on `train.py`) to:
1. Load a checkpoint (`final.pt`)
2. Run 1000 eval episodes with `deploy_eval=True`
3. Report `SR_deploy` alongside the training `SR_train` from `metrics.csv`

Alternatively, extend `post_hoc_coding.py` with a `--deploy_eval` flag.

---

## 8. Figure Plan (F29)

| Figure | Description | Data source |
|---|---|---|
| F29 | SR at training vs. deployment for SD and NSD (4 bars: SD-train, SD-deploy, NSD-train, NSD-deploy) | E09/E47 checkpoints + E62/E63 eval runs |

**F29** is the headline P3 figure: deployment consistency visualised as SR gap (or lack thereof).

A supplementary panel can show the deployment quantisation noise distributions:
- SD: histogram of `ẑ_deploy − z` (should be approximately U(−δ/2, δ/2) by Thm A.1)
- NSD: scatter plot of `(z_k, z_hat_true_k)` — always a step function, never continuous

---

## 9. Open Questions

**Q1 — Is there a measurable SR gap between SD training and SD deployment?**

Schuchman (Thm A.1) guarantees no gap in expectation. But the SD listener is trained
on STE noise `e ~ U(−δ/2, δ/2)`, while deployment inputs are `(m+0.5)·δ − ε` — a
different realization from the same distribution. If the listener's policy is very
sensitive to the exact input distribution (e.g., has high curvature near bin boundaries),
a small gap is possible. E62 will measure this directly.

**Hypothesis:** SR gap < 0.01 (within rollout noise). If gap > 0.01, it would motivate
using NSD over SD even when shared PRNG is feasible.

**Q2 — Does the TPDF bin probability affect the RB gradient for NSD?**

As discussed in §6.2.1, the exact RB gradient for NSD uses quadratic TPDF bin
probabilities rather than the linear RPDF probabilities used in the SD derivation. For
the proxy-loss approximation used in P4 (all-dims-joint), this bias is accepted.
If the per-dim RB mode is implemented, the TPDF correction may need to be applied
for NSD to get an unbiased estimate.

**Resolution:** This is a P4 implementation detail, not a P3 experiment. Note it in
the P4 implementation checklist when `rb_mode="per_dim"` is built.

**Q3 — Does Phase 2 dither loss interact with the deployment gap?**

Phase 2 training pushes z to bin boundaries (`frac(z/δ) → 0.5`). Near boundaries,
the SD training input `z + e` spans two bins with equal probability — which is also
exactly what the deployment input `(m+0.5)·δ − ε` does. The distribution match
should be tightest at Phase 2 convergence, not worst. No interaction issue expected.

---

## 10. Implementation Checklist

- [x] `DDCL_NSD.forward()` — sample-consistent z_hat_true implementation
- [x] `DDCL_NSD.comms_loss()` — Jensen bound (same as SD)
- [x] `DDCL_NSD.transmission_bits_per_elem()` — empirical bits from m
- [x] Per-channel δ support in `DDCL_NSD` (P1 compat)
- [x] E47 `nsd_posthoc_mag` — NSD Phase 1 training (DONE)
- [x] E48 `nsd_twophase_dither5e-4_v2` — NSD Phase 2 training (DONE)
- [ ] Add `deploy_eval: bool` flag to `trainer.py` `act_and_value()`
- [ ] Implement E62: SD deployment evaluation on E09 checkpoints
- [ ] Implement E63: NSD deployment evaluation on E47 checkpoints
- [ ] Write F29: SR training vs. deployment bar chart

---

## 11. Experiment Registry

| ID | Name | Type | Status | Config | Seeds | Primary metric |
|---|---|---|---|---|---|---|
| E47 | `nsd_posthoc_mag` | Training | **DONE** | NSD, Phase 1, λ_mag | 5 | H_joint, SR |
| E48 | `nsd_twophase_dither5e-4_v2` | Training | **DONE** | NSD, Phase 2, λ_dither=5e-4 | 5 | H_joint, SR |
| E62 | `sd_deploy_eval` | Evaluation | PLANNED | SD, deploy_eval=True, E09 checkpoints | 5 | SR_deploy vs SR_train |
| E63 | `nsd_deploy_eval` | Evaluation | PLANNED | NSD, deploy_eval=True, E47 checkpoints | 5 | SR_deploy vs SR_train |

**E47/E48** are the P3 training experiments (already done, motivated C16).
**E62/E63** are lightweight eval-only experiments — load checkpoint, run 1000 eval
episodes with deploy_eval=True, report SR. No GPU training needed.

---

## 12. Success Criteria

**Primary:** E63 shows `SR_deploy = SR_train` to within rollout noise (± 0.01), confirming
sample-consistent training for NSD.

**Secondary:** E62 shows `SR_deploy ≥ SR_train − 0.01` for SD, confirming Schuchman
Thm A.1 holds empirically (distributional equivalence is sufficient).

**Failure criterion for NSD claim:** If `SR_deploy < SR_train − 0.02` for E63 (NSD),
the sample-consistency argument is violated — which would indicate a bug in
`DDCL_NSD.forward()` (the forward value does not match the deployment value). This
would be a code error, not a theoretical failure.

**Failure criterion for SD claim:** If `SR_deploy < SR_train − 0.05` for E62 (SD),
Schuchman distributional equivalence is insufficient for this environment, and using
NSD would provide a concrete task-performance advantage. This would strengthen the P3
argument for NSD as the preferred channel.

**Minimum bar for paper:** F29 shows no SR degradation at deployment for either channel,
confirming the DDCL quantisation scheme is deployment-realistic.

---

## 13. Cross-Pillar Dependency Summary

| Pillar pair | Interaction | Status | New experiments? |
|---|---|---|---|
| P3 × P2 | Histogram adapts to NSD messages identically | Closed by C16 | None |
| P3 × P1 | Per-channel δ transparent to NSD deployment claim | Closed by code | None |
| P3 × P4 | RB formula applies to NSD bin centres; TPDF probabilities differ slightly | Noted; P4 detail | None for P3; note in P4 |
| P3 standalone | Deployment eval on existing checkpoints | E62/E63 pending | Eval-only (lightweight) |

**P3 is the lightest pillar:** the theory is already established (Thm 5), the
implementation is done, and the empirical validation (C16) is complete. The remaining
work is E62/E63 (eval-only) and F29.
