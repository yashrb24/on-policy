# Pillar P2 — Entropy Model

**Status:** IMPLEMENTATION COMPLETE — ablation runs pending (Phase 3)  
**Pillar order:** P2 → P1 → P4 → P3 (first to implement)  
**Paper reference:** "Stochastic Quantisation via Dithering" §Eq. 2

---

## 1. Motivation

The baseline DDCL communication cost uses a magnitude surrogate:

```
L_comms = Σ_k log₂(|z_k|/δ + 1)
```

This is a Jensen upper bound on true entropy (derived in `docs/MATH.md`). P2 replaces it with a learned prior `q_φ(m)` that tracks the true distribution of quantised messages `m = floor((z+noise)/δ)`:

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
q_φ_k(m_k) = CDF(m_k + 1) − CDF(m_k)
CDF(x) = Σ_c π_c · σ((x − μ_c) / s_c)
```

where `{log π_c, μ_c, log s_c}` are the K mixture parameters per dimension k.

**Bin semantics — floor, not round.** The quantiser computes `m = floor((z+noise)/δ)`, so integer m is assigned all continuous mass in the interval `[m, m+1)`. Using rounding bins `[m−0.5, m+0.5)` would be incorrect and would bias the learned μ by +0.5 relative to each bin centre. The floor formula `CDF(m+1) − CDF(m)` is correct. At convergence, the DLM component means μ settle near `n + 0.5` for integer bin n (the bin centre), not at n.

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

**V4 empirical confirmation (test_v4_tc_identity_factored_minus_joint):** Trained both factored and joint models on perfectly-correlated synthetic data (m₀ = m₁). Observed factored−joint NLL gap ≈ TC + ε_DLM, where ε_DLM ≈ 0.27 bits/dim is the irreducible DLM approximation error (see §5 and `docs/MATH.md §11`). The TC identity holds; the extra offset is a known DLM floor, not a bug. **Practical implication:** use `tc_bits` (empirical histogram estimate) rather than the factored−joint NLL gap for model selection — the NLL gap includes ε_DLM noise that is absent from `tc_bits`.

---

## 3. Context Conditioning Options

Three variants, from most realistic to tightest bound. Only Context A is a valid training loss source for the speaker. Contexts B and C are **MEASUREMENT-ONLY**: they train q_φ and produce rate metrics, but their backward loss to the speaker is disabled because it is structurally degenerate (see §4 and `docs/MATH.md §12` Fix 3 for the full derivation).

### Context A — Marginal prior (PRIMARY, deployment-realistic, LOSS SOURCE)

`q_φ(m)` is unconditioned. It is trained as a running parametric estimate of the marginal distribution of `m` across all time steps and episodes. The receiver needs `q_φ` at deployment (it is fixed after training); no access to z or hidden state required.

**Training:** gradient on `q_φ` parameters via `-log₂ q_φ(m)` over a rollout buffer of collected `m` samples.

**This is the main contribution.** Context B and C are supplementary measurements only.

**Why context A is the only valid loss source for the speaker.** When z increases in magnitude, x = z/δ moves into the tail of the marginal prior q_φ(m). Since q_φ does NOT condition on z, it cannot follow the current z — NLL grows, and the gradient correctly signals "this message is expensive." This is the desired incentive.

### Context B — Condition on z (MEASUREMENT ONLY — NOT an ablation winner candidate)

`q_φ(m|z)` — DLM parameters produced by a small MLP from `z`. Measures the conditional entropy H(m|z), which is tighter than the marginal H(m). Unrealistic at deployment (the receiver does not observe z; the whole point of the channel is to transmit z via m).

**Why context B CANNOT provide a valid speaker gradient.** In DDCL with dithering, m is not perfectly deterministic given z (there is dither-induced stochasticity), but after training q_φ(m|z) converges to the true conditional. At convergence, NLL → 0 bits and the gradient to z → 0. The speaker can increase |z| freely without changing NLL because q_φ always "knows" what m will be from z. See `docs/MATH.md §12` Fix 3 for the three-reason derivation.

**Correct use:** run as a parallel measurement-only model alongside context A. The metric `entropy_rate_B` ≈ H(m|z) serves as an oracle lower bound. The gap `entropy_rate_A − entropy_rate_B ≈ I(z; m)` (mutual information between the speaker output and the discrete message) is the key diagnostic: if large, the marginal prior is leaving information on the table; if near zero, context A is already near-optimal.

**Implementation:** `entropy_model_context="B"`, forward loss only, backward guarded off. This is not a training-mode ablation; it is a diagnostic configuration.

### Context C — Condition on speaker hidden state (MEASUREMENT ONLY — NOT an ablation winner candidate)

`q_φ(m|h_speaker)` — DLM parameters from the speaker's internal hidden state. Even tighter than B because h contains the full history compressed by the speaker network. Even less realistic (h is internal to the speaker and cannot be transmitted to the receiver).

**Why context C cannot provide a valid speaker gradient.** The same degeneracy as B, but more severe: h is more informative about m than z is (h is the representation from which z was computed). q_φ(m|h) converges even faster to near-zero NLL, and the backward gradient vanishes even earlier.

**Correct use:** metric `entropy_rate_C` ≈ H(m|h) is a tighter oracle bound than B. The gap `entropy_rate_B − entropy_rate_C` quantifies information in h beyond z — useful for understanding the speaker's computation, not for training it.

**Note:** Context C requires exposing `SpeakerNetwork` intermediate activations. Currently deferred; add `SpeakerNetwork.hidden` property when implementing.

**Comparison purpose:** rank by `entropy_rate_A ≥ entropy_rate_B ≥ entropy_rate_C ≥ H(m)`. These inequalities hold because conditioning can only reduce entropy. Context B and C cannot "win" an ablation (their backward is disabled), but they set bounds that context A should approach as training improves.

---

## 4. Gradient Path — Ballé-style Relaxation

`-log₂ q_φ(m)` is not differentiable with respect to z because m = floor((z+noise)/δ) is discrete. We use a **two-term loss** following Ballé et al. (2018):

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

**Context B backward is disabled (implementation fix):** When `entropy_model_context == "B"`, the backward loss is skipped entirely. Context B models `q_φ(m|z)`. Although m = floor((z+noise)/δ) has dither-induced stochasticity, q_φ(m|z) converges to the true conditional and NLL → 0 at the continuous evaluation point z/δ — the backward gradient to the speaker → 0. Evaluating `-log q_φ(z/δ | z_fixed)` with `z_fixed = z.detach()` creates an additional inconsistency: the gradient pushes z toward regions where the OLD z's DLM peaked, not toward genuinely lower-rate messages. **Context B is only valid as a rate measurement tool (forward loss + metrics). Only context A (unconditional marginal) provides a valid backward gradient to the speaker.** See `docs/MATH.md §12` (Fix 3) for the full derivation. Test: `test_v8_context_b_backward_disabled`.

**Update order fix:** The q_φ forward update now runs BEFORE the RL optimizer step. This ensures that when the backward entropy loss is computed (step 3 of each minibatch), q_φ has already tracked the current batch's message distribution. The previous order (q_φ update after RL step) trained q_φ on stale z values from a speaker that had just been modified. Timing: q_φ update → backward loss (frozen q_φ) → RL step.

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

**DLM approximation floor (V2 empirical finding):** DLM cannot represent bounded-support distributions exactly because it assigns non-zero probability mass to all integers. For a distribution with true support of cardinality N, DLM leaks ~9% of its probability mass outside that support, producing an irreducible NLL gap above H(P) of approximately **0.27 bits/dim**. This means `qphi_gap` (defined in §7) will never reach true zero — the expected floor is `≈ 0.27 × z_dim` bits. A `qphi_gap` persistently above this floor indicates a training problem; a gap near the floor indicates the model has converged. See `docs/MATH.md §11` (V2) for the derivation and tolerance analysis.

**Mixture prior (hardening fix):** All four entropy model classes now use a mixture prior `q̃_φ(x) = (1−α)·q_φ(x) + α·Laplace(x; 0, s_flat)` with α=0.01, s_flat=50. This guarantees `q̃_φ(x) > 0` for all x without requiring warm-start, and provides a small but non-zero gradient at all z/δ values. See `docs/MATH.md §12` (Fix 1). Code: `_FLAT_ALPHA`, `_FLAT_SCALE`, `_mix_with_flat` in `network.py`.

**Scale floor (hardening fix):** DLM scales are clamped to `s_eff ≥ exp(_S_LOG_MIN) = 0.1` before `exp()`. This prevents numerical collapse to a delta function if `log_s → −∞`, AND allows joint models to express sharp conditionals (P ≈ 0.987 per component at the mode). The previous s_min=0.5 would cap conditional probability at 0.46, which breaks the TC identity measured in V4. See `docs/MATH.md §12` (Fix 2). Code: `_S_LOG_MIN` in `network.py`.

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

**V3 empirical finding — warm-start is necessary, not just convenient:** A q_φ trained only on a narrow distribution (e.g., all-zeros messages) assigns probability < 1e-10 to messages outside that support. When the Ballé backward loss evaluates `q_φ(z/δ)` for z/δ values in the dead region, the probability hits the numerical clamp and the gradient is exactly zero — the speaker receives no learning signal. This is not a rounding issue; it is a fundamental consequence of the logistic mixture's very narrow tails after premature convergence. `n_warmup_steps ≥ 5000` ensures q_φ has seen a broad enough message distribution before RL begins, preventing gradient silence from day 1. The `n_warmup_steps=0` case is expected to fail silently (training appears to run, but speaker gradient is zero). See `docs/MATH.md §11` (V3) and §9 below for the failure mode entry.

---

## 7. Metrics

All new metrics are logged to `metrics.csv` per step in addition to the existing columns.

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| `entropy_rate` | `E[-log₂ q_φ(m)]` over rollout batch | Cross-entropy; true bit rate upper bound |
| `H_m_empirical` | Empirical entropy from histogram of m in batch | Ground-truth entropy estimate |
| `qphi_gap` | `entropy_rate − H_m_empirical` | Model fit (= KL divergence); should → DLM floor as q_φ converges |
| `tc_bits` | `Σ_k H(m_k) − H(m)` (batch estimate) | Factored looseness; inter-dimension correlation |
| `qphi_neg_log_max` | `max(-log₂ q_φ(m))` over batch | Tail spike detector; should stay bounded |
| `bits_vs_magnitude` | `magnitude_surrogate_bits − entropy_rate` | Rate improvement of P2 over baseline surrogate |
| `entropy_rate_B` | `E[-log₂ q_φ_B(m\|z)]` from companion context-B model | Oracle conditional rate bound; logged automatically alongside context A |
| `context_gap_bits` | `entropy_rate − entropy_rate_B` ≈ I(z; m) | Mutual information left on the table by marginal prior; target → 0 |
| `shannon_gap` | `true_bits_per_msg − H(G)` | Distance from Shannon limit; the ultimate P2 metric |
| `bits_to_hg_ratio` | `true_bits_per_msg / H(G)` | Normalised; target = 1.0 |
| `warm_start_bits_final` | `entropy_rate` at end of warm-start phase | Confirms warm-start was effective; NaN if warmup not run |
| `H_dim_{k}` | `H(m_k)` for each dimension k | Per-dimension entropy; informs P1 per-channel δ allocation |
| `entropy_loss_magnitude` | `λ · E[-log₂ q_φ(z/δ)]` (Ballé backward term value) | Compression pressure on speaker; if ≪ `speaker_grad_norm`, P2 is inert |
| `speaker_grad_norm` | L2 norm of speaker parameter gradients (pre-clip) | Total gradient reaching speaker; compare with `entropy_loss_magnitude` |
| `entropy_rate_goal_{i}` | Per-goal `entropy_rate` | Adaptive rate allocation analysis |

**Companion context-B model:** A `EntropyModelCondZ` (or `EntropyModelJointCondZ`) companion is automatically created alongside the context-A primary whenever `use_entropy_model=True` and `entropy_model_context="A"`. It always uses K=5 (sufficient for measurement; see §10 Stage P2-A). It is updated in forward-only mode (no backward to speaker). This gives `entropy_rate_B` and `context_gap_bits` in every context-A run without a separate experiment.

**Per-goal entropy rate:** computed by masking the rollout buffer by `goal_id` (already logged in Phase 1 infrastructure).

**qphi_gap floor:** `qphi_gap` is not expected to reach zero. The DLM approximation floor is ≈ 0.27 bits/dim (see §5). Expected minimum: `qphi_gap_floor ≈ 0.27 × z_dim`. A gap near this floor indicates convergence; a gap more than ~0.5 bits above this floor after warm-start indicates a training problem (stale prior, insufficient `n_qphi_steps`, or `lr_qphi_mult` too low). Do **not** use `qphi_gap` alone for factored-vs-joint model selection; use `tc_bits` instead (see §2).

**P2 analysis figures** (generated by `generate_sweep_figures` when entropy_rate columns are present):
- `p2_shannon_gap` — true_bits − H(G) over training
- `p2_qphi_gap` — model fit vs DLM floor
- `p2_context_bounds` — entropy_rate (A) vs entropy_rate_B (oracle) vs H_m_empirical
- `p2_per_dim_entropy` — H_dim_k bar chart at convergence
- `p2_gradient_balance` — speaker_grad_norm and entropy_loss_magnitude over training

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
| **Gradient dead zone** | Speaker entropy does not decrease; `entropy_rate` flat despite λ_ent > 0 | q_φ assigns probability < 1e-10 to current messages; Ballé backward gradient is exactly 0; happens when warm-start is skipped or too short | Ensure `n_warmup_steps ≥ 5000`; verify warm-start data covers the actual message support; diagnose by checking `qphi_neg_log_max` > 30 bits at training start |

---

## 10. Ablation Matrix — Staged Design

All ablations produce 5-seed runs, compared on `(success_rate, entropy_rate, qphi_gap)` with bootstrap CI.

**Execution order matters**: each stage fixes winners from prior stages. Run P2-A first; do not run P2-C through P2-F until P2-A and P2-B are complete.

**Total runs: ~395** across all stages.

---

### Stage P2-A — Model selection (165 runs)

**Fixed:** `channel=sd`, `delta=Phase2_best`, `z_dim=Phase2_best`, `lambda_comms=Phase2_winner`

**Purpose:** Identify best `(K, model_type, loss_comms_mode)` combination for context A (the only valid training context). Contexts B are run in MEASUREMENT-ONLY mode alongside to bound the rate gap. The question is: does joint beat factored? How many mixture components are needed? Does entropy-only or entropy+RL joint loss perform better?

**The two trainable model variants** (context A only):

| | Context A — marginal prior (LOSS SOURCE) |
|---|---|
| **Factored** | `EntropyModelFactored` — independent dimensions |
| **Joint (AR)** | `EntropyModelJoint` — autoregressive conditionals |

**Two measurement-only variants** (context B — NO speaker gradient, forward loss only):

| | Context B — conditional prior (MEASUREMENT ONLY) |
|---|---|
| **Factored** | `EntropyModelCondZ` — measures H(m\|z) factored |
| **Joint (AR)** | `EntropyModelJointCondZ` — measures H(m\|z) joint (tightest bound) |

Context B variants always run alongside their context A counterpart in the same experiment (same rollout, separate q_φ network). They cannot win the ablation because they do not affect the speaker. They serve as oracle bounds: a large gap `entropy_rate_A − entropy_rate_B` indicates the marginal prior is leaving information on the table; a small gap means context A has converged near-optimally.

**Why context B is NOT swept over K:** K only affects the expressiveness of q_φ(m|z). Since q_φ(m|z) is not used as a training signal, K for context B is irrelevant to RL performance. Context B always uses K=5 (sufficient to fit the dither-induced conditional). The K ablation is only meaningful for context A where q_φ's expressiveness directly affects the speaker's rate penalty signal.

**Grid:**

| Group | K | model_type | context | loss_comms_mode | Configs |
|-------|---|-----------|---------|----------------|---------|
| baseline | — | — | — | magnitude | 1 |
| factored × A | {1,3,5,10,20} | factored | A | {entropy, both} | 10 |
| joint × A | {1,3,5,10,20} | joint | A | {entropy, both} | 10 |
| factored × B (measurement) | {5} | factored | B | magnitude | 1 |
| joint × B (measurement) | {5} | joint | B | magnitude | 1 |

Note: Context B rows use `loss_comms_mode="magnitude"` (the baseline surrogate) for the speaker. The entropy model runs in parallel to produce measurement metrics only. This correctly separates the question "does context B improve training?" (no — it cannot) from "what does context B tell us about the rate?" (the H(m|z) bound).

Total configs: 1 + 10 + 10 + 1 + 1 = 23 configs × 5 seeds = **115 runs** (reduced from 165 in the original draft, which incorrectly included context B as a training loss source).

**Winners:** model_type*, K*, loss_comms_mode* from the context A rows only — these fix all subsequent stages. Context B metrics are reported in supplementary material.

---

### Stage P2-B — λ re-sweep (70 runs, 35 net-new)

**Fixed:** best config from P2-A, `z_dim=Phase2_best`, `channel=sd`, `delta=Phase2_best`

**Purpose:** The P2 entropy surrogate has a different scale than the magnitude surrogate, so the optimal `lambda_comms` changes. Without this sweep we cannot produce a P2 rate-distortion frontier or make the core claim "P2 lies on a better frontier than baseline." The baseline magnitude sweep (Phase 2 Stage A) already provides 7×5 = 35 comparison runs.

**Grid:**

| λ | baseline_magnitude | P2_best |
|---|---|---|
| {1e-5, 1e-4, 5e-4, 1e-3, 4e-3, 1e-2, 3e-2} | (from Phase 2 Stage A) | 35 new runs |

**Winners:** `lambda_best_p2` — fixes all subsequent stages.

---

### Stage P2-C — z_dim interaction (40 runs)

**Fixed:** `K=K*`, `model_type=model_type*`, `context=A`, `loss_comms_mode=entropy`, `lambda=lambda_best_p2`, `channel=sd`, `delta=Phase2_best`

**Purpose:** Total correlation TC is zero at z_dim=1 (factored = joint trivially). At z_dim=2,3 the speaker may learn correlated messages. This stage tests whether the joint model's advantage is larger at higher z_dim, as TC theory predicts. This is the key result for extrapolating to future environments where z_dim will be larger.

**Grid:**

| z_dim | baseline | P2_factored | P2_joint |
|-------|----------|-------------|---------|
| 1 | ✓ | ✓ | (= factored, skip) |
| 2 | ✓ | ✓ | ✓ |
| 3 | ✓ | ✓ | ✓ |

Total: 8 configs × 5 seeds = 40 runs.

**Expected result:** `tc_bits` at z_dim=1 ≈ 0 (joint ≈ factored); `tc_bits` grows with z_dim; joint closes more of the gap at higher z_dim.

---

### Stage P2-D — δ interaction (40 runs)

**Fixed:** `K=K*`, `model_type=model_type*`, `context=A`, `lambda=lambda_best_p2`, `z_dim=Phase2_best`, `channel=sd`

**Purpose:** δ controls the quantisation grid width. Small δ → large |m| values → complex m distribution → larger P2 gain (the magnitude surrogate `log₂(|z|/δ + 1)` is proportionally looser). Large δ → m ≈ 0 almost always → near-trivial prior → P2 gain vanishes. This is a testable and paper-worthy prediction.

**Grid:**

| δ | baseline | P2_best |
|---|----------|---------|
| {0.5, 1.0, 5.0, 10.0} | ✓ | ✓ |

Total: 8 configs × 5 seeds = 40 runs.

---

### Stage P2-E — Channel interaction (20 runs)

**Fixed:** `K=K*`, `model_type=model_type*`, `context=A`, `lambda=lambda_best_p2`, `z_dim=Phase2_best`, `delta=Phase2_best`

**Purpose:** NSD (non-subtractive) dither has a different quantisation error distribution (TPDF vs uniform). The message distribution `p(m)` under NSD differs from SD at the same δ. P2 should adapt automatically since `q_φ` learns from actual m samples. Confirms P2 is channel-agnostic.

**Grid:**

| channel | baseline | P2_best |
|---------|----------|---------|
| sd | ✓ | ✓ |
| nsd | ✓ | ✓ |

Total: 4 configs × 5 seeds = 20 runs.

---

### Stage P2-F — Training robustness (60 runs)

**Fixed:** best model from P2-A, `lambda=lambda_best_p2`, `z_dim=Phase2_best`, `delta=Phase2_best`, `channel=sd`

**Purpose:** Characterise sensitivity to training hyperparameters. The ranges found here directly inform what to use in future (more complex) environments, where the moving target problem is harder.

**One-at-a-time (OAT) around the P2 defaults:**

| Axis | Values tested | Other axes |
|------|--------------|------------|
| `lr_qphi_mult` | {1, 5, 10, 50} | n_qphi_steps=3, n_warmup=5000 |
| `n_qphi_steps` | {1, 3, 5, 10} | lr_qphi_mult=10, n_warmup=5000 |
| `n_warmup_steps` | {0, 1000, 5000, 20000} | lr_qphi_mult=10, n_qphi_steps=3 |

Total: 12 configs × 5 seeds = 60 runs.

**Key diagnostic:** `qphi_gap` over training — should converge to near-zero. Large `qphi_gap` at low `n_qphi_steps` or `lr_qphi_mult` identifies the moving-target boundary.

---

### Extrapolation to main environments

After all 6 stages, the following facts will be established for the toy problem:

| Finding | Extrapolation |
|---------|--------------|
| Best K (e.g., K=5 sufficient) | Use K=5 as default; increase only if `qphi_gap` remains large |
| factored vs joint threshold (e.g., joint helps at z_dim≥2 when TC>0.1 bits) | Use joint when z_dim>2 in new envs |
| Optimal `lr_qphi_mult` range (e.g., 5–20×) | Use same range in new envs; scale n_qphi_steps with env complexity |
| P2 gain vs δ (larger gain at small δ) | Tune δ to sit in the regime where P2 helps |
| P2 works for NSD as well as SD | Safe to combine P2 with P3 (TPDF dithering) |
| context A ≈ context B gap size | If gap is large, consider context B for environments where z is accessible at train time |

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
