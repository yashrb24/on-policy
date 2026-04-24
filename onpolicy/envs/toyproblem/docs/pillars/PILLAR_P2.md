# Pillar P2 — Entropy Model

**Status:** DESIGN COMPLETE — pending implementation (Phase 3)  
**Pillar order:** P2 → P1 → P4 → P3 (first to implement)  
**Paper reference:** "Stochastic Quantisation via Dithering" §Eq. 2

---

## 1. Motivation

The baseline DDCL communication cost uses a magnitude surrogate:

```
L_comms = Σ_k log₂(|z_k|/δ + 1)
```

This is a Jensen upper bound on true entropy (derived in `docs/MATH.md`). P2 replaces it with a learned prior `q_φ(m)` that tracks the true distribution of quantised messages `m = round(z/δ)`:

```
L_ent = E[-log₂ q_φ(m)]          # cross-entropy, upper-bounds H(m)
```

When `q_φ → p(m)` this equals the true Shannon entropy, giving a strictly tighter and adaptive rate signal.

**Why this matters:** The magnitude surrogate is biased (penalises large |z| whether or not the receiver can predict them), scale-sensitive (depends on δ), and loses information about the marginal distribution shape. A learned prior can discover that, e.g., a bimodal message distribution has lower entropy than its magnitude suggests.

---

## 2. Prior Family — Discretised Logistic Mixture (DLM)

### Choice of family

`q_φ` is a **Discretised Logistic Mixture (DLM)**:

```
q_φ_k(m_k) = CDF(m_k + 0.5) − CDF(m_k − 0.5)
CDF(x) = Σ_c π_c · σ((x − μ_c) / s_c)
```

where `{log π_c, μ_c, log s_c}` are the K mixture parameters per dimension k.

**Why DLM over alternatives:**
- **vs Laplace/Gaussian:** DLM is a universal approximator over ℤ with K components; a single Laplace is accurate only if the true distribution is unimodal and symmetric
- **vs Laplace (principled argument):** z near 0 implies m near 0 and Laplace would be fine at baseline, but once the speaker learns to use quantisation bins strategically the distribution can be multimodal or skewed; DLM handles this without model selection
- **vs neural density (autoregressive NN over ℤ):** much higher sample complexity for Context A; DLM with K≈5 captures anything realistic in z_dim ≤ 3
- **vs softmax over finite alphabet:** requires a hard codebook size; integers are unbounded (see §5)

### Parameters

| Symbol | Shape | Note |
|--------|-------|------|
| `log_pi` | `[z_dim, K]` | log mixing weights (normalised via log-softmax) |
| `mu` | `[z_dim, K]` | component means |
| `log_s` | `[z_dim, K]` | log scales (exp → s > 0) |

Total: `3 × K × z_dim` scalars. At K=5, z_dim=3: 45 parameters. Very small.

### Factored vs joint modelling

**Factored (primary):** `q_φ(m) = ∏_k q_φ_k(m_k)` — one DLM per dimension, trained and evaluated independently.

**Joint autoregressive (ablation):** `q_φ(m) = q_φ_1(m_1) · q_φ_2(m_2|m_1) · q_φ_3(m_3|m_1,m_2)`, where each conditional is a DLM whose parameters are produced by a small MLP taking prior dimensions as input.

**Looseness of factored (derivable bound):**

The factored approximation overestimates entropy by exactly the **total correlation**:

```
TC(m_1, ..., m_K) = Σ_k H(m_k) − H(m_1, ..., m_K)  ≥ 0
```

This is the KL divergence from the joint to the product of marginals. In code, TC is estimated empirically from a batch of collected `m` vectors:

```python
tc_bits = sum(marginal_entropy(m[:, k]) for k in range(K)) - joint_entropy(m)
```

where entropies are estimated from histogram counts. TC = 0 if and only if all dimensions are statistically independent. For the toy problem at z_dim=1 this is trivially 0; for z_dim=2,3 it can be non-zero if the speaker learns correlated messages.

**Implementation plan:** implement factored first. Implement joint as a separate `EntropyModelJoint` class that wraps the autoregressive MLP. Compare in ablation study.

---

## 3. Context Conditioning Options

Three variants, from most realistic to tightest bound:

### Context A — Marginal prior (PRIMARY, deployment-realistic)

`q_φ(m)` is unconditioned. It is trained as a running parametric estimate of the marginal distribution of `m` across all time steps and episodes. The receiver needs `q_φ` at deployment (it is fixed after training); no access to z or hidden state required.

**Training:** gradient on `q_φ` parameters via `-log₂ q_φ(m)` over a rollout buffer of collected `m` samples.

**This is the main contribution.** Context B and C are ablations for supplementary material.

### Context B — Condition on z (ablation)

`q_φ(m|z)` — DLM parameters produced by a small MLP from `z`. Tighter cross-entropy than A (can capture z-dependent shaping). Unrealistic at deployment because the receiver does not observe `z`. Useful as an oracle upper bound on how tight the rate signal can be.

### Context C — Condition on speaker hidden state (ablation)

`q_φ(m|h_speaker)` — DLM parameters from speaker hidden state. Even tighter than B (h contains history). Even less realistic. Useful to bound the information available in h that B misses.

**Comparison purpose:** rank the three by: `entropy_rate_A ≥ entropy_rate_B ≥ entropy_rate_C ≥ H(m)`. The gap A−B quantifies the mutual information `I(z; m)` (beyond what the prior knows). The gap B−C quantifies the extra information in h beyond z.

---

## 4. Gradient Path — Ballé-style Relaxation

`-log₂ q_φ(m)` is not differentiable with respect to z because m = round(z/δ) is discrete. We use a **two-term loss** following Ballé et al. (2018):

```
# Forward loss: trains q_φ on actual discrete messages
loss_ent_fwd = -log2(q_phi(m))              # discrete m, no grad to speaker

# Backward loss: differentiable proxy for speaker gradient
loss_ent_bwd = -log2(q_phi(z / delta))      # continuous relaxation, grads flow to speaker
```

Both terms are evaluated at each step:
- `loss_ent_fwd` is detached from the speaker; it trains `q_φ` on the true discrete distribution
- `loss_ent_bwd` propagates gradients to the speaker as if m were continuous; `q_φ` is detached here

Total pillar loss: `L_P2 = λ_ent · (loss_ent_fwd + loss_ent_bwd)` (both contribute to q_φ update; only bwd contributes to speaker).

**Why this preserves Schuchman's theorem:** The channel forward pass (dithering, quantisation, STE) is not touched. The only change is in the loss term. Quantisation error independence is preserved.

---

## 5. Handling Unbounded Support of m

`m ∈ ℤ` is unbounded. The DLM handles this naturally:

- The logistic CDF gives exponentially decaying probability in the tails: `q_φ(m) ~ exp(-|m|/s)` for large |m|
- Therefore `-log₂ q_φ(m) ~ |m|/s` linearly for large |m| — penalises extreme messages proportionally, which is the correct behaviour
- No clipping or truncation needed; the model self-regulates

**Initialisation to prevent early pathology:**
- `log_s` initialised to `≥ 1.0` (wide scales at init) so early messages receive finite, non-explosive costs
- `mu` initialised to 0 (centred)
- `log_pi` initialised to uniform (equal mixing weights)

**Gradient clipping:** already present in the trainer. Monitor `qphi_neg_log_max` metric (see §7) to detect tail spikes early.

---

## 6. The Moving Target Problem

As the speaker policy improves, `p(m)` changes. `q_φ` must track a non-stationary target.

**Mitigation strategies:**

| Strategy | Hyperparameter | Default | Notes |
|----------|---------------|---------|-------|
| Higher q_φ learning rate | `lr_qphi` | `10 × lr_actor` | q_φ adapts faster than speaker |
| Multiple q_φ updates per RL step | `n_qphi_steps` | 3 | Each PPO epoch does n_qphi_steps q_φ gradient steps |
| Warm-start | `n_warmup_steps` | 5000 | Pre-train q_φ on initial rollouts before RL begins |
| q_φ update frequency | always | always | q_φ updated every training step (not every N steps) |

**Future environments (non-trivial):** the same strategy applies but `n_qphi_steps` and `lr_qphi` may need to be larger. In large multi-agent environments with many speakers, `qphi_sharing` (see §8) reduces the number of independent priors, indirectly accelerating convergence of each.

**Warm-start distribution shift risk:** after warm-start, the RL policy may quickly move away from the warm-start distribution. Monitor `qphi_gap` (§7) to detect stale priors; if it grows post-warm-start, increase `n_qphi_steps`.

---

## 7. Metrics

All new metrics are logged to `metrics.csv` per step in addition to the existing columns.

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| `entropy_rate` | `E[-log₂ q_φ(m)]` over rollout batch | Cross-entropy; true bit rate upper bound (nats→bits via /log(2)) |
| `H_m_empirical` | Empirical entropy from histogram of m in batch | Ground-truth entropy estimate (converges with large batch) |
| `qphi_gap` | `entropy_rate − H_m_empirical` | Model fit quality; should → 0 as q_φ converges |
| `tc_bits` | `Σ_k H(m_k) − H(m)` (batch estimate) | Factored looseness; measures inter-dimension correlation |
| `qphi_neg_log_max` | `max(-log₂ q_φ(m))` over batch | Tail spike detector; should stay bounded |
| `bits_vs_magnitude` | `magnitude_surrogate_bits − entropy_rate` | Rate improvement of P2 over baseline surrogate |
| `entropy_rate_goal_{i}` | Per-goal `entropy_rate` | Adaptive rate allocation analysis |

**Per-goal entropy rate:** computed by masking the rollout buffer by `goal_id` (already logged in Phase 1 infrastructure).

---

## 8. Multi-Agent Sharing Strategy

`qphi_sharing` controls how many independent priors are maintained:

| Value | Behaviour | Use case |
|-------|-----------|----------|
| `per_agent` | One q_φ per speaker agent | Heterogeneous agents |
| `shared` | One q_φ shared across all speakers | Homogeneous agents |
| `per_role` | One q_φ per agent role class | Mixed teams |

For the toy problem (1 speaker), all three are equivalent. Implement `per_agent` and `shared`; `per_role` can be added when multi-role environments appear. Compare `per_agent` vs `shared` in ablation to establish whether sharing helps or hurts (for future environments: sharing reduces estimation variance but increases bias if agents' message distributions differ).

---

## 9. Failure Modes and Diagnostics

| Failure mode | Symptom | Diagnosis | Fix |
|-------------|---------|-----------|-----|
| **Prior collapse** | `entropy_rate` → 0, task success → 0 | Speaker pushes z→0 to minimise cross-entropy | Reduce `λ_ent`; check `qphi_gap` stays small |
| **Prior chase** | `qphi_gap` grows then oscillates | q_φ overfits to current batch; speaker chases moving optimum | Increase `n_qphi_steps`; reduce `lr_qphi`; increase rollout buffer |
| **Gradient explosion** | `qphi_neg_log_max` spikes; NaN in actor loss | Rare large m values; log_s too small | Widen scale init; clip gradients (already in place) |
| **Warm-start distribution shift** | `qphi_gap` low during warm-start, spikes after RL begins | q_φ stale relative to new policy | Increase `n_qphi_steps`; reduce warm-start length |
| **Dimension collapse** | All z_k → same value; `tc_bits` → 0 but `entropy_rate` not reduced | Speaker finds degenerate solution | Monitor per-dimension entropy; check z variance per dimension |

---

## 10. Ablation Matrix

All ablations produce 5-seed runs, compared on `(success_rate, entropy_rate, qphi_gap)` with bootstrap CI.

### Primary ablations (P2 core)

| Axis | Values |
|------|--------|
| `K` (mixture components) | 1, 3, 5, 10, 20 |
| `context` | A (primary), B (ablation), C (ablation) |
| `model` | factored, joint |
| `loss_comms_mode` | `entropy` (P2 only), `both` (P2 + magnitude surrogate) |

### Interaction ablations

| Axis | Values |
|------|--------|
| `qphi_sharing` | `per_agent`, `shared` |
| `lr_qphi` (as multiple of `lr_actor`) | 1×, 5×, 10×, 50× |
| `n_qphi_steps` | 1, 3, 5, 10 |
| `n_warmup_steps` | 0, 1000, 5000, 20000 |

### Comparison baselines

- Baseline DDCL (magnitude surrogate, `λ_comms = baseline_best`)
- P2-A factored K=5 (primary P2)
- P2-A joint K=5
- P2-B factored K=5 (oracle, supplementary)
- P2-C factored K=5 (oracle, supplementary)

---

## 11. Implementation Plan (overview)

1. `channels.py`: no changes — forward pass untouched
2. `network.py`: add `EntropyModelFactored` and `EntropyModelJoint` classes
3. `trainer.py`: add `loss_ent_fwd`, `loss_ent_bwd`; add q_φ optimizer; add n_qphi_steps loop; log P2 metrics to CSV
4. `train.py`: add CLI flags (`--use_entropy_model`, `--K`, `--context`, `--model_type`, `--lr_qphi_mult`, `--n_qphi_steps`, `--n_warmup_steps`, `--qphi_sharing`)
5. `tests/test_entropy_model.py`: unit tests for DLM eval, gradient path, warm-start, metric shapes
6. `experiments/run_p2_ablation.py`: orchestrate ablation matrix
7. `analysis/paper_figures.py`: add P2 figures (entropy_rate vs λ, qphi_gap over training, factored vs joint)
8. `docs/pillars/PILLAR_P2.md`: update status to IN PROGRESS / DONE as implementation proceeds

---

## 12. Document Update Instructions

**At session start (when working on P2):** Read this file fully. Check §1–§10 before writing any code.

**During implementation:** Update the Status line at the top of this file when implementation begins; update again when complete. Add findings (unexpected behaviour, empirical surprises) to the relevant section. Do NOT duplicate in MATH.md — cross-reference instead.

**Mathematical derivations** (e.g., TC looseness proof, Ballé relaxation justification) → `docs/MATH.md`, then link here.
