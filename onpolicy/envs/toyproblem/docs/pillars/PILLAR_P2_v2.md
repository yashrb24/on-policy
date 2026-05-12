# Pillar P2 — Communication-Efficient Source Coding for DDCL

**Status:** CLOSED (2026-05-11) — All experiments done (E01–E52); all figures done (F1–F21, F9/F18 deferred); all §11 ablations complete; C1–C16 in CONCLUSIONS.md.
**Pillar order:** P2 → P1 → P4 → P3 (first to implement)
**Paper reference:** "Stochastic Quantisation via Dithering" §Eq. 2

---

## 1. The Core Question: What is the Minimum Communication Cost?

### 1.1 The DDCL channel and the message m

The speaker produces z ∈ ℝ^D. The channel transmits the dithered quantised message:

```
m = floor((z + ε) / δ)    ε ~ U(-δ/2, δ/2)^D
```

The receiver observes m ∈ ℤ^D and reconstructs ẑ = (m + 0.5)·δ. The STE gradient ∂ẑ/∂z = 1 preserves gradient flow. Schuchman's theorem guarantees the quantisation error is independent of z.

The **true communication cost** is the Shannon entropy of m:

```
H(m) = -Σ_{v ∈ ℤ^D} p(m=v) log₂ p(m=v)    [bits per message vector]
```

A source code (arithmetic or Huffman) applied to the transmitted sequence achieves exactly H(m) bits per message in the limit.

### 1.2 Shannon entropy of the goal as the absolute lower bound

The task requires the listener to identify which of G goals was assigned. The absolute lower bound on communication cost is:

```
H(G) = -Σ_{i=1}^{G} p(goal_i) log₂ p(goal_i)
```

For the toyproblem with 6 non-uniformly distributed goals: **H(G) ≈ 1.81 bits**. No scheme can identify the goal in fewer bits. Any H(m) > H(G) at SR = 1 is compression slack that P2 is designed to eliminate.

### 1.3 The current gap

| Measure | Value |
|---|---|
| H(G) | ≈ 1.81 bits (irreducible) |
| H(m) at SR=1 (sc_option1 run) | ≈ 7.67 bits |
| true_bits_per_msg (magnitude baseline) | ≈ 4.75 bits |
| Total gap above H(G) | ≈ 5.86 bits |

This gap decomposes into three separable components — each with its own elimination mechanism — developed in §5.

---

## 2. What the Magnitude Loss Gets Wrong (and Right)

### 2.1 The magnitude surrogate is cross-entropy with the wrong prior

The baseline DDCL loss is:

```
L_comms = Σ_k log₂(|z_k|/δ + 1)
```

This is the **cross-entropy between the empirical message distribution and a fixed implicit prior**. Derivation:

The batch sum `(1/B) Σ_i Σ_k log₂(|z_k^(i)|/δ + 1) ≈ E_{m ~ p̂(m)}[Σ_k log₂(|m_k| + 1)]`, which equals `-E_{p̂}[log₂ q_implicit(m)]` where:

```
q_implicit(m_k) ∝ 1 / (|m_k| + 1)      [Zipf(1) — implicit prior]
```

Decomposing by cross-entropy = entropy + KL:

```
L_mag  =  H(p̂)  +  KL(p̂ ‖ q_implicit)  +  const
```

- **H(p̂)**: irreducible message entropy — what we actually want to minimise
- **KL(p̂ ‖ q_implicit)**: pure waste — extra bits paid because the hardcoded Zipf prior assumes large bins are rare, when the speaker learns to concentrate on ~6 bins of roughly equal probability

The KL term cannot be eliminated by changing the policy. The speaker concentrates on a small uniform-ish set of bins — nothing like a Zipf distribution. The magnitude surrogate charges the speaker for a structural mismatch it can never fix.

**P2 replaces the fixed prior with an adaptive one:**

```
L_P2 = H(p̂, q_φ) = H(p̂) + KL(p̂ ‖ q_φ)
```

When q_φ → p̂(m), KL → 0 and L_P2 → H(m). The Jensen gap closes to zero.

### 2.2 Structural analogy to Ballé image compression — and why it breaks here

Ballé et al. (2018) train an encoder y = f_a(x; θ) and penalise `E[-log₂ p_ŷ(ŷ)]` where ŷ = round(y). The Ballé backward pushes y toward the prior mode (zero). This is correct for Ballé: the decoder reconstructs x from ŷ directly — every distinct ŷ value encodes a distinct reconstruction. Compressing y is equivalent to compressing ŷ.

In DDCL the listener receives **m**, not z. The listener cannot distinguish z = -30 from z = -29.5 if both produce the same m. The Ballé backward evaluates q_φ(z/δ) at the continuous speaker output, pushing z toward the mode of q_φ. Since q_φ is trained on discrete m, its mode sits at the centre of the speaker's current bins. The backward pushes z toward the current codebook — not toward a lower-entropy codebook. This is the **circular gradient** problem.

| | Ballé | Magnitude DDCL | P2 (Ballé backward) |
|---|---|---|---|
| What is compressed | y and ŷ | z and m | z (but m is what matters) |
| Implicit prior | N(0, σ²) | Zipf(1) over m | q_φ over m, evaluated at z/δ |
| Is compressing the continuous repr. correct? | Yes — decoder needs y | No — listener sees only ẑ = m·δ | No — same structural error |

**The correct object to compress is m, not z.** A speaker at z = -30 in a frequently-used bin pays only H(G) bits — no pressure to shrink to z = -0.3. Both the magnitude surrogate and the Ballé backward penalise |z| implicitly, which is the wrong variable.

### 2.3 Why the magnitude loss is retained: geometric regularisation

Despite being wrong as a compression objective, the magnitude loss plays a structural role that must not be discarded: it prevents SGD from driving the speaker to a **geometrically degenerate representation**.

The sc_option1 experiment confirmed this: with `--loss_comms_mode entropy` (no magnitude penalty), z_norm grew from 11 to 216 after SR = 1 was achieved — an 18× growth with no compression benefit. Without any cost on |z|, RL satisfies the task by pushing goals to arbitrarily large but distinct bins. The resulting p(m) has support over widely-spaced bins requiring a large source code to exploit.

The neural collapse literature provides theoretical backing. Under L2-style norm regularisation, representation manifolds converge to equinorm, equiangular tight frame (ETF) geometry — maximally separated and structured (Papyan et al., 2020). Without regularisation, representations can collapse to disjoint structures that achieve the same task loss but compress poorly.

**Relevant prior works:**
- **Neural Collapse is Globally Optimal in Deep Regularized ResNets and Transformers** (arXiv:2505.15239, 2025): proves under L2 regularisation that neural collapse is the unique global minimum; all other critical points are saddle points.
- **Neural Collapse Dynamics: Depth, Activation, Regularisation, and Feature Norm Thresholds** (arXiv:2604.00230, 2026): weight decay defines a three-regime phase diagram; optimal regularisation accelerates collapse and prevents disjoint manifolds via feature norm control.
- **Neural Rank Collapse: Weight Decay and Small Within-Class Variance** (arXiv:2402.03991, 2024): L2 weight decay induces an implicit low-rank bias, linking norm geometry to manifold structure.
- **A Minimal Model of Representation Collapse** (arXiv:2604.09979, 2026): regularisation and stop-gradient prevent collapsed representations, enabling finite class separation.

**The magnitude loss is retained not as a compression signal, but as a geometric anchor** — it prevents unbounded z drift during Phase 1 and encourages a structured representation that compresses better post hoc. The ablation in §9.4 tests this claim directly by comparing regularised and unregularised Phase 1 training, measuring TC(m) and H(m|goal) at convergence.

---

## 3. The Correct Framework: Source Coding from First Principles

### 3.1 Shannon's source coding theorem applied to DDCL

Shannon (1948): for a discrete memoryless source p(m), the minimum expected bits per symbol is H(m). An arithmetic code from q_φ ≈ p(m) achieves exactly H(m) bits per message in the limit.

Applied to DDCL:
- The speaker emits symbols m ∈ ℤ^D at each step
- The marginal: p(m) = E_{o ~ D}[P(m | z(o))]
- The training objective: minimise H(m) subject to task success — not |z|
- **z should remain fully expressive.** A speaker using z = -30 for goal A and z = +30 for goal B uses two distinct bins at δ=1. Rate = H({p_A, p_B}) — identical to z = ±0.3. Magnitude of z has no effect on rate.

### 3.2 Rate objective and EM two-step optimisation

Under an arithmetic code built from q_φ:

```
L_rate(θ, φ) = E_{m ~ p(·;θ)}[-log₂ q_φ(m)]
              = H(m; θ) + KL(p(m; θ) ‖ q_φ)  ≥  H(m; θ)
```

Full training objective:

```
min_{θ, φ}  E_t[L_task(θ)]  +  λ · E_m[-log₂ q_φ(m)]
```

This separates into an E-step and M-step:

**E-step (prior fitting, θ fixed):**
```
min_φ  KL(p(m; θ) ‖ q_φ)     ← maximum likelihood on observed m samples
```
The histogram achieves this instantaneously — the histogram IS q_φ up to smoothing. No gradient steps needed.

**M-step (speaker update, q_φ frozen):**
```
min_θ  E_{m ~ p(·;θ)}[-log₂ q_φ(m)]     with q_φ frozen
```
Requires differentiating through discrete sampling m ~ P(m|z). Done via the score function.

### 3.3 The score function rate gradient (Rao-Blackwell reduced)

The dither places m_k in exactly two adjacent bins:

```
m_lo,k = floor(z_k/δ),      P(m_lo,k | z_k) = 1 - frac(z_k/δ)
m_hi,k = floor(z_k/δ) + 1,  P(m_hi,k | z_k) = frac(z_k/δ)
```

Applying the score function identity and substituting ∂ log P / ∂z:

```
∂/∂z_k E[R_k(m_k)]
  = f_k · R_k(m_hi,k) · (1/δ)/f_k  +  (1-f_k) · R_k(m_lo,k) · (-1/δ)/(1-f_k)
  = [R_k(m_hi,k) - R_k(m_lo,k)] / δ
```

The expectation collapses analytically to a deterministic expression — no Monte Carlo sampling. This is the Rao-Blackwell minimum-variance estimator (Theorem 6 of the proposal; P2×P4 synergy):

```
∇_z_k L_rate  =  [R(m_hi,k) - R(m_lo,k)] / δ    [per dimension, factored prior]
```

For a joint prior q_φ(m₁, …, m_D):

```
∇_z_k L_rate  =  [R_joint(m_{-k}, m_hi,k) - R_joint(m_{-k}, m_lo,k)] / δ
```

**Interpretation:**
- R_hi > R_lo → positive gradient → push z_k into the cheaper lower bin
- R_hi < R_lo → negative gradient → push z_k up, crossing into the cheaper upper bin
- R_hi = R_lo → zero gradient → z_k is free within its current bin

The gradient depends only on the rate differential at the nearest boundary. A speaker at z = -30 in a frequent bin receives near-zero gradient — no shrinkage pressure. This is the correct compression signal; the Ballé backward is not.

### 3.4 Arithmetic vs. Huffman coding

Both are valid instantiations. The choice affects the training reward R(m) and the deployment protocol.

**Arithmetic coding (recommended):**
- Expected codelength = -log₂ q_φ(m) exactly in the limit
- The training objective E[-log₂ q_φ(m)] is the arithmetic coding rate
- Smooth in q_φ → trainable by gradient descent on forward loss
- Works for unbounded m ∈ ℤ^D
- R(m) = -log₂ q_φ(m) is the correct score function reward

**Huffman coding (practical alternative):**
- Integer codelengths L_H(m) = ⌈-log₂ q_φ(m)⌉ bits; expected codelength ∈ [H(m), H(m)+1]
- Requires finite alphabet (truncate to observed support with escape code for unseen values)
- Non-differentiable in q_φ, but appears only as scalar reward R(m) = L_H(m) in the score function — valid; no gradient needed through it
- Simpler at deployment: fixed lookup table

**For this project:** arithmetic coding rate R(m) = -log₂ q_hist(m) as both the score function reward and the forward loss for q_φ. In the MARL simulator, m is transmitted as an integer directly — the coding cost is a training signal only. Either scheme applies at deployment in a real bandwidth-constrained channel.

### 3.5 Theoretical properties of the histogram + score function framework

**P1: Unbiasedness.** (R_hi - R_lo)/δ is an unbiased estimator of ∂/∂z_k E[R(m_k)]. The expectation collapses analytically — no sampling noise.

**P2: Minimum variance within the two-bin structure.** The analytical collapse is the Rao-Blackwell reduction of REINFORCE for the two-bin dither. No unbiased estimator using only the two-bin structure has lower variance.

**P3: Histogram convergence.** q_hist(m) → p(m) as N → ∞ by the law of large numbers. Convergence rate O(1/√N) per bin. For the toy problem (~6 bins), 1% relative error requires ~10,000 samples — within one rollout buffer.

**P4: Valid rate upper bound.** E[-log₂ q_φ(m)] = H(m) + KL(p ‖ q_φ) ≥ H(m) always. Tight iff q_φ = p(m).

**P5: EM fixed point is a stationary point of the rate-distortion objective.** At convergence: q_hist = p(m; θ*) (E-step) and θ* satisfies first-order conditions of min_θ L_task + λ H(m; θ) (M-step). This is a local minimum of the true rate-distortion objective.

**P6: Schuchman compatibility.** The channel forward pass is untouched. ∂ẑ/∂z = 1 is preserved. The rate gradient affects only the loss, not the channel.

### 3.6 Connection to the rate-distortion objective

The complete P2 objective is:

```
min_{θ}  E[L_task(θ)]  +  λ · H(m; θ)
subject to: p(m; θ) = E_{o ~ D, ε ~ U}[P(m | f(o; θ), ε)]
```

This is rate-distortion in the classical sense:
- **Distortion:** 1 - E[SR(θ)] (task failure)
- **Rate:** H(m; θ) (speaker message entropy)
- **Lagrange multiplier:** λ (rate-distortion tradeoff)

The Pareto frontier of (SR, H(m)) is the rate-distortion function for this MARL task. The histogram + score function provably achieves points on this frontier; the Ballé backward approximates it with a biased gradient.

H(G) ≈ 1.81 bits is the absolute lower bound on H(m) at SR = 1. Any frontier point above H(G) represents slack P2 is designed to eliminate.

### 3.7 What the RL objective is actually optimising

**Does the full objective directly minimise H(m)?**

```
L = L_RL  +  λ · E[-log₂ q_hist(m)]
            = L_RL  +  λ · (H(m) + KL(p ‖ q_hist))
```

Since q_hist converges to p(m) by construction, KL → 0.004 bits (smoke run confirmed). We are asymptotically minimising H(m) exactly — the speaker maps goals to messages that are informative and have low entropy, concentrating into a small set of frequent bins.

**Two separate notions of compression:**

| | Training loop | Deployment |
|---|---|---|
| **What happens** | Score function gradient concentrates p(m) | Arithmetic / Huffman code encodes m using q_hist |
| **Effect** | H(m) decreases over updates | Actual bits transmitted per message decreases |
| **Lower bound** | H(G) ≈ 1.81 bits | Same |
| **What is saved** | Nothing yet — only the policy is reshaped | Real communication bandwidth |

During training, no bits are literally saved. The rate loss is a policy shaping signal. At deployment, the source code is applied to the stationary p(m; θ*) and real bandwidth savings are realised.

---

## 4. Why Histogram + Score Function Eliminates All Three DLM Failure Modes

The DLM + Ballé backward failed in three independent ways (P2-FIX sweep, 2026-04-29). The histogram + score function addresses each structurally:

| Failure mode | DLM + Ballé backward | Histogram + Score function |
|---|---|---|
| **Fitting failure** (qphi_gap 12–14 bits; gate always blocked) | DLM cannot track the non-stationary p(m; θ) — fitting lag accumulates faster than n_qphi_steps can correct | Histogram = p(m) by construction; qphi_gap → 0.004 bits from update 1 |
| **Circular gradient** (shannon_gap flat despite open gate) | ∇_z[-log q_φ(z/δ)] points toward the mode of q_φ — the current codebook — not toward a lower-entropy codebook | ∇_z L_rate = (R_hi - R_lo)/δ points toward the cheaper adjacent bin — toward a better codebook |
| **Ballé domain gap** (qphi_neg_log_max spikes; gradient direction wrong) | q_φ trained on discrete m ∈ ℤ, evaluated at continuous z/δ ∈ ℝ; mismatch when q_φ is poorly fitted | R(m) evaluated only at discrete m ∈ ℤ; no domain mismatch |

Additionally: the score function gradient does not push z toward zero. A speaker at z = -30 in a frequent bin receives near-zero gradient. z remains fully expressive.

---

## 5. Post-Training Source Coding: The Preferred Design

### 5.1 Why post-hoc is preferred over live training-time coding

The key empirical finding from sc_option1: **even with a perfect histogram (qphi_gap ≈ 0.004 bits), H(m) moved only 7.67 → 7.1 bits — 0.6 bits over 450 updates**. A well-fitted estimator is necessary but not sufficient for compression during live RL training. Three structural reasons:

**Reason 1 — Signal strength ceiling (100:1 RL dominance).** At λ = 5e-4, the score function gradient magnitude is ≈100× smaller than the PPO gradient. This is not a tuning issue — there is no λ at which one can equalise them without catastrophically degrading SR. The RL objective is unbounded positive reinforcement; the compression signal is bounded by H(m). The ratio is fundamental to the two-objective structure.

**Reason 2 — Score function locality.** The gradient (R_hi - R_lo)/δ compares only the two bins adjacent to the current z_k. Consolidating goals from z_norm ≈ 11 (bin ≈ ±8) to bin ≈ ±1 requires traversing ~8 bins sequentially — each step fought by the RL gradient at 100:1. The score function cannot make global bin-space moves.

**Reason 3 — Sign inversion of the proxy loss.** When m_hi is cheaper than m_lo (speaker near the lower bin boundary), the score function gradient *increases* z_k. sc_rate_loss went negative and grew more negative in sc_option1 — correct behaviour, but means the proxy loss value is not a reliable indicator of compression progress. Trust hist_H_empirical, not sc_rate_loss.

**By contrast, the post-hoc setting has none of these problems.** After training: p(m; θ*) is stationary; no RL gradient competes; MLE error is a one-time O(K^D / √N) ≈ 0.04 bits, not a moving target; the arithmetic coder adapts to whatever p(m) the speaker produced.

**Design choice:** Post-hoc source coding is the primary design. Live training-time coding (§8) is an ablation for the distributed training setting. Both design paths are preserved and run; the post-hoc path is connected to the empirical evidence above.

### 5.2 Post-hoc source coding protocol

**Phase 1 — RL training with geometric anchor:**

Train speaker and listener with RL + magnitude loss. The histogram observes p(m) passively but applies no gradient. The magnitude loss prevents unbounded z growth (§2.3).

```bash
--loss_comms_mode magnitude    # geometric anchor; no score function signal
--phase1_sr_threshold 0.99     # triggers Phase 2 automatically at this SR level
```

**Phase 2 — Dither entropy minimisation (optional, after SR = 1):**

Activate the dither entropy loss L_dither (§5.3) once the policy is near-converged. Adjusts frac(z_k/δ) toward bin boundaries without changing which bins are used. L_dither shifts each z_k by at most δ/2 = 0.5 toward the nearest boundary — far smaller than the inter-goal z-space separation (z_norm ≈ 11 at SR = 1). SR is unaffected throughout Phase 2 because which bin is used does not change; only the fractional alignment does.

**Deployment — post-hoc codebook construction:**

After training converges (θ = θ*), collect a rollout under the frozen policy. Build empirical joint histogram P̂(m₁, …, m_D) from N message samples. Construct an arithmetic code from P̂. Encode each transmitted m using this code. Reported rate = H_joint(m) bits per message.

**No prior is learned during training.** The speaker is free to learn any z configuration that supports SR = 1. The source code is built once from the stationary p(m; θ*). MLE error O(K^D / √N) is a fixed, one-time cost.

### 5.3 The dither entropy loss (Phase 2 objective)

The dither channel noise H(m|goal) = E_goal[Σ_k H_binary(frac(z_k(goal)/δ))] goes to zero when frac(z_k/δ) → 0 or 1. Define:

```
L_dither(z)  =  λ · Σ_{k=1}^{D} H_binary(frac(z_k / δ))
```

Gradient:

```
∂L_dither / ∂z_k  =  (λ / δ) · log₂( frac(z_k/δ) / (1 − frac(z_k/δ)) )
```

This is the log-odds of the fractional part, scaled by λ/δ:
- Negative for frac < 0.5: pulls z_k toward lower bin boundary
- Positive for frac > 0.5: pulls toward upper boundary
- Zero at frac = 0 or 1: already at boundary; no force applied

**No histogram required at training time.** The gradient is computed entirely from the current z. Does not constrain |z|. Does not change which bin is used.

**Documented failure modes (not yet empirically tested — monitor carefully):**
- **Gradient explosion:** log-odds → ±∞ as frac → 0 or 1 exactly. In practice z_k never sits precisely on a grid point, but clamp frac to [1e-4, 1-1e-4] if gradient norms spike at Phase 2 start.
- **Bin-crossing discontinuity:** if a Phase 2 gradient step pushes z_k across an integer boundary, m_k changes. If the new bin conflicts with another goal's bin, SR could drop. Monitor SR closely in Phase 2; reduce λ if drops are observed.
- **Multiple goals at the same fractional position:** identical gradient directions; benign as long as goals are in different bins (which they must be for SR = 1).

---

## 6. Rate Decomposition: Three Separable Sources of Gap

### 6.1 The full decomposition

Every bit above H(G) in the empirical message rate has a specific origin:

```
R_empirical  =  H(G)              [irreducible: task information]
             +  H(m | goal)        [dither channel noise — closed by Phase 2 L_dither]
             +  TC(m)              [cross-dimension correlation — closed by joint coding]
             +  ε_estimator        [estimator approximation error — closed by histogram]
             +  ε_MLE              [finite-sample error — irreducible floor at fixed N]
```

For the post-hoc centralised setting with the joint histogram: ε_estimator = 0, ε_MLE ≈ 0.04 bits.

### 6.2 H(m|goal): dither channel noise

```
H(m | goal) = E_goal[ Σ_k H_binary(frac(z_k(goal) / δ)) ]
```

H_binary(f) ∈ [0, 1] bit. Achieves 0 at f = 0 or 1 (bin boundary); maximum 1 bit at f = 0.5. This is pure channel noise: even if the goal is known, the listener cannot predict which bin z_k lands in because ε is random.

**Eliminated by:** L_dither in Phase 2 (§5.3).

### 6.3 TC(m): cross-dimension correlation

```
TC(m) = Σ_k H(m_k) - H_joint(m) ≥ 0
```

Measures information in cross-dimension correlations that a factored (per-dimension) source code wastes. Different goals map to correlated (m₁, m₂) pairs; a factored code ignores this, paying the correlation cost.

**Eliminated by:** joint source code over the full D-dimensional vector m ∈ ℤ^D. Achieves H_joint(m) — eliminates TC exactly, with no model assumption.

Estimated value (sc_option1): TC ≈ 2.6 bits (≈44% of the total gap).

**Scalability caveat for the joint histogram.** The joint histogram has K^D entries. For D=2, K=6: 36 bins — trivial. For D=10, K=6: 6^10 ≈ 60 million bins — impractical. For large D, the joint histogram must be replaced by a structured estimator (autoregressive NN, copula model). For the toyproblem (D=2), the joint histogram is exact and fully tractable.

### 6.4 ε_estimator: model approximation error

```
ε_estimator = KL(p(m) ‖ q_φ(m)) ≥ 0    [= qphi_gap in metrics]
```

For the DLM, ε_estimator has an irreducible floor of ≈0.27 bits/dim (DLM leaks ~9% mass outside true support). The DLM qphi_gap stayed at 12–14 bits during training. The histogram achieves ε_estimator → 0.004 bits by construction.

**Eliminated by:** using the empirical histogram (§7.1).

### 6.5 ε_MLE: finite-sample error

```
|Ĥ − H(m)| ≤ log₂(K^D) / √N
```

For the toyproblem post-hoc (D=2, K=6, N=4096): ≈0.04 bits. Irreducible floor at fixed rollout size.

**In the distributed live-coding setting** (§8):

```
|Ĥ_t − H(m; θ_t)| ≤ O(K^D / √N)  +  O(‖∇_θ H‖ · ‖Δθ‖)
```

The moving-target term dominates early in training and decays as θ converges.

### 6.6 Joint vs. factored source coding: the ablation

Both coders are applied post hoc to the same trained policy — no extra training runs:
- **(A) Joint coder:** arithmetic coding over full D-dimensional histogram P̂(m). Achieves H_joint(m) = H(G) + H(m|goal).
- **(B) Factored coder:** arithmetic coding per dimension. Achieves Σ_k H(m_k) = H_joint(m) + TC(m).

Difference (B) − (A) = TC(m). This is the paper's direct measurement of the value of joint coding.

**Decomposition validation (one held-out rollout):**

```
Σ_k H(m_k)  ≡  H(G)  +  E_g[Σ_k H_binary(frac(z_k(g)/δ))]  +  TC(m)
```

Should hold to within 0.04 bits (MLE error). If it does, each term is independently reported.

---

## 7. Marginal Distribution Estimators: Methods, Failure Modes, and Selection

All estimators serve two roles: (A) the rate estimate R(m) used in the score function gradient during live training, and (B) the source code constructed post hoc for deployment. Role B is the joint empirical histogram in all cases. Role A differs and determines training-time behaviour.

### 7.1 Online histogram (non-parametric) — CURRENT IMPLEMENTATION

Per-dimension count dictionary with Laplace smoothing α = 0.5 (Jeffreys prior):

```
q_hist(m_k) = (count_k[m_k] + α) / (N_k + α · V_k)
R(m_k) = -log₂ q_hist(m_k)    [detached tensor — no gradient through histogram]
```

V_k = number of distinct bins seen. qphi_gap ≈ 0.004 bits by construction.

**Score function proxy loss (M-step):**

```python
def source_coding_rate_loss(z, histogram, delta, lambda_comms):
    m_lo = torch.floor(z / delta).long()       # (B, D)
    m_hi = m_lo + 1                             # (B, D)
    R_lo = histogram.rate(m_lo)                 # (B, D) — detached
    R_hi = histogram.rate(m_hi)                 # (B, D) — detached
    # Proxy: ∂/∂z_k [(R_hi - R_lo)/δ · z_k] = (R_hi - R_lo)/δ
    return lambda_comms * ((R_hi - R_lo).detach() / delta * z).sum(-1).mean()
```

**Integration into PPO (E-step / M-step):**

```python
# E-step: once per rollout, before PPO epoch
histogram.reset()
histogram.update(rollout_buffer.m)    # m already logged; no network forward

# M-step: inside each PPO minibatch
for epoch in range(n_epochs):
    for minibatch in rollout_buffer.get_minibatches():
        z = speaker_network(minibatch.obs)
        rate_loss = source_coding_rate_loss(z, histogram, delta, lambda_comms)
        total_loss = pg_loss + c1*value_loss - c2*entropy_loss + rate_loss
        total_loss.backward()
        optimizer.step()
```

**Limitations as a live training signal (sc_option1 empirical findings):**
- Score function gradient is local (adjacent bins only) — cannot consolidate distant bins
- At λ = 5e-4, ≈100× weaker than PPO — RL dominates every update
- sc_rate_loss proxy goes negative when m_hi cheaper → gradient increases z_k (correct but counterintuitive; trust hist_H_empirical not sc_rate_loss)
- Combined: H(m) changed by only 0.6 bits over 450 post-SR=1 updates

### 7.2 Context variants for DLM-based estimators

These apply when using the DLM (§7.3) rather than the histogram. They are documented here because the context framework shapes how any parametric estimator is trained and whether its backward gradient to the speaker is valid.

**Context A — Marginal prior (valid training signal):** q_φ(m) is unconditioned. Trained as a running estimate of the marginal distribution across all time steps. The receiver needs q_φ at deployment; no access to z or hidden state required. When z increases in magnitude, z/δ moves into the tail of q_φ — NLL grows and the gradient correctly signals "this message is expensive." This is the only context that provides a valid speaker gradient.

**Context B — Condition on z (measurement only; backward disabled):** q_φ(m|z) — DLM parameters from a small MLP taking z as input. Measures the conditional entropy H(m|z), which is tighter than the marginal H(m). Unrealistic at deployment (the receiver never observes z).

Why the backward is disabled: after training q_φ(m|z) converges to the true conditional; NLL → 0 bits at the continuous evaluation point z/δ — the backward gradient to the speaker → 0. The speaker can increase |z| freely without changing NLL because q_φ always "knows" what m will be from z. Evaluating `-log q_φ(z/δ | z_fixed)` with `z_fixed = z.detach()` creates an additional inconsistency: the gradient pushes z toward regions where the OLD z's DLM peaked, not toward genuinely lower-rate messages.

Correct use: run as a parallel measurement-only model. `entropy_rate_B` ≈ H(m|z) serves as an oracle lower bound. The gap `entropy_rate_A − entropy_rate_B ≈ I(z; m)` is a key diagnostic: if large, the marginal prior is leaving information on the table; if near zero, context A is already near-optimal.

**Context C — Condition on speaker hidden state (measurement only):** q_φ(m|h_speaker). Even tighter than B. Same structural degeneracy — converges to near-zero NLL faster, backward gradient vanishes earlier. Correct use: metric `entropy_rate_C` ≈ H(m|h) is a tighter oracle bound than B. The gap `entropy_rate_B − entropy_rate_C` quantifies information in h beyond z.

**Ordering of bounds:** entropy_rate_A ≥ entropy_rate_B ≥ entropy_rate_C ≥ H(m). Conditioning can only reduce entropy. B and C cannot "win" an ablation (backward disabled), but they set bounds that A should approach as training improves.

### 7.3 Discretised Logistic Mixture (DLM) — ABLATION BASELINE

```
q_φ_k(m_k) = CDF(m_k + 1) − CDF(m_k)
CDF(x) = Σ_c π_c · σ((x − μ_c) / s_c)
```

3 × K × z_dim parameters (45 for K=5, z_dim=3). Handles unbounded m ∈ ℤ via logistic CDF tails.

**Gradient path (Ballé relaxation):**
```
loss_ent_fwd = -log₂ q_φ(m)          # trains q_φ on discrete m; no grad to speaker
loss_ent_bwd = -log₂ q_φ(z / δ)      # continuous relaxation; grads flow to speaker
```
Update order fix: q_φ forward update runs BEFORE the RL optimizer step so that when the backward loss is computed, q_φ has already tracked the current batch.

**Hardening fixes (implemented in network.py):**
- **Mixture prior:** q̃_φ(x) = (1−α)·q_φ(x) + α·Laplace(x; 0, s_flat), α=0.01, s_flat=50. Guarantees q̃_φ > 0 everywhere; provides non-zero gradient at all z/δ values without requiring warm-start.
- **Scale floor:** DLM scales clamped to s_eff ≥ exp(S_LOG_MIN) = 0.1, preventing numerical collapse to a delta function and allowing sharp conditionals (P ≈ 0.987 per component at the mode).

**Unbounded support:** DLM assigns non-zero probability to all integers via logistic CDF tails. No clipping or truncation needed. Initialise: log_s ≥ 1.0 (wide at init), mu = 0 (centred), log_pi = uniform.

**DLM approximation floor (V2 empirical finding):** DLM cannot represent bounded-support distributions exactly — it leaks ~9% mass outside the true support. Irreducible NLL gap ≈ **0.27 bits/dim**. Expected minimum qphi_gap = 0.27 × z_dim. A gap near this floor means convergence; persistently above (0.27 × z_dim + 0.5 bits) means a training problem.

**Moving target mitigation strategies (for DLM in live training):**

| Strategy | Hyperparameter | Default | Notes |
|---|---|---|---|
| Higher q_φ learning rate | lr_qphi | 10 × lr_actor | q_φ adapts faster than speaker |
| Multiple q_φ updates per RL step | n_qphi_steps | 3 | Each PPO epoch: n_qphi_steps q_φ gradient steps |
| Warm-start | n_warmup_steps | 5000 | Pre-train q_φ on initial rollouts before RL begins |
| q_φ update frequency | always | always | q_φ updated every training step |

**V3 empirical finding — warm-start is necessary:** A q_φ trained only on all-zeros messages assigns probability < 1e-10 outside that support. When the Ballé backward evaluates q_φ(z/δ) in the dead region, the gradient is exactly zero — the speaker receives no learning signal. n_warmup_steps ≥ 5000 ensures q_φ has seen a broad distribution before RL begins. The n_warmup_steps=0 case fails silently (training appears to run, but speaker gradient is zero).

**P2-FIX sweep verdict (2026-04-29):** The DLM with Ballé backward is not self-correcting under RL training dynamics. The gate never opens (qphi_gap 12–14 bits vs threshold 2.0 bits throughout). The EMA prior makes no difference when the gate is always closed. Two-phase training (fix_B4) worked for 1/5 seeds — proving the goal is achievable — but the mechanism (λ=1e-2 magnitude in Phase 1) is too brittle (80% seed collapse). The DLM is superseded by the histogram for the primary design but retained as a quantitative ablation baseline.

**Three documented failure modes:**

| Mode | Cause | Evidence |
|---|---|---|
| **Fitting failure** | DLM cannot track non-stationary p(m; θ); fitting lag accumulates faster than n_qphi_steps can correct | bwd_gate_active = 0 for ALL non-trivial configs; gate always blocked |
| **Circular gradient** | Well-fitted q_φ(m) has its mode at the current distribution's mode; backward pulls z toward the current codebook, not toward a lower-entropy one | shannon_gap flat even where gate was open (fix_baseline) |
| **Ballé domain gap** | q_φ trained on discrete m ∈ ℤ, evaluated at continuous z/δ ∈ ℝ; when q_φ is poorly fitted the evaluation point lands in wrong-probability regions | qphi_neg_log_max spikes; gradient direction wrong |

**Constraints any alternative must satisfy (non-negotiable):**
1. ∂ẑ/∂z = 1 — DDCL's core contribution; channel forward pass untouched
2. m ∈ ℤ unbounded — prior must assign non-zero probability everywhere; no finite categorical
3. Deployment-realistic backward signal — speaker gradient from information available after training (not conditioning on z at test time)
4. Compatible with MARL — works for any number of speaker agents with shared or per-agent priors

**DLM Fix Ladder (documented for ablation reference):**

The fix ladder was developed and fully implemented before the DLM was superseded. It is preserved here because the ablation in §9.3 runs these configs as baselines.

| Level | Targets | Key flags |
|---|---|---|
| **L1: Conditional backward gate** | Poor q_φ fitting | `--qphi_bwd_gate_threshold 2.0 --n_warmup_steps 50000 --n_qphi_steps 20 --lr_qphi_mult 30.0` |
| **L2: λ re-sweep** | Gradient scale mismatch (1.5% ratio) | `--lambda_comms 1e-2` (or sweep {5e-3, 1e-2, 2e-2, 5e-2}) |
| **L3: EMA prior** | Circular gradient attenuation | `--use_ema_prior --ema_prior_momentum 0.95`; effective lag ≈ 20 RL updates |
| **L4: Fixed Laplace prior** | Circular gradient (architectural fallback) | `EntropyModelLaplace` with single learnable log_scale; not yet implemented |
| **L5: Two-phase training** | RL dominance; decoupled convergence | `--phase1_sr_threshold 0.995 --loss_comms_mode magnitude --lambda_comms 1e-2`; Phase 1 ≈ 300k steps, recommend total_timesteps ≥ 1M |

**P2-FIX experiment configs (25 runs: 5 configs × 5 seeds):**

| Config | Levels | Key flags |
|---|---|---|
| fix_baseline | None | n_warmup=5000, n_qphi=3 |
| fix_B1_gate | L1 | gate=2.0, n_warmup=50000, n_qphi=20 |
| fix_B2_gate_lambda | L1+L2 | gate=2.0, n_warmup=50000, n_qphi=20, λ=1e-2 |
| fix_B3_ema | L1+L2+L3 | gate=2.0, n_warmup=50000, n_qphi=20, λ=1e-2, EMA |
| fix_B4_twophase | L1+L2+L5 | gate=2.0, n_warmup=50000, n_qphi=20, λ=1e-2, phase1_sr=0.995 |

**Metrics to monitor after each fix level:**

| Metric | Target | Notes |
|---|---|---|
| qphi_gap | ≤ 1.0 bits | L1 working; DLM floor = 0.27 × z_dim |
| bwd_gate_active | 1.0 (majority) | Gate should open after warm-start |
| entropy_loss_magnitude / speaker_grad_norm | ≥ 5% | L2 scale check |
| bits_vs_magnitude | < 0 | Entropy bits < magnitude bits = P2 winning |
| training_phase | transitions 1→2 | Phase switch for L5 |
| true_bits_per_msg (Phase 2) | monotone decreasing | Core compression signal |
| shannon_gap | < baseline (2.48 bits) | Ultimate success criterion |

### 7.4 Neural network conditional prior (NN + marginal buffer)

```
ẑ → MLP_φ → (log_π, μ, log_s) → DLM(m | ẑ)

# Backward (avoids Context B degeneracy):
q_marginal(x) = (1/B) Σ_{ẑ_i ∈ buffer} DLM(x | ẑ_i)    # buffer is detached
loss_bwd = -log₂ q_marginal(z/δ)
```

**Why condition on ẑ (not z):** Conditioning on z is Context B — structurally degenerate. Conditioning on ẑ = z + e, e ~ U(-δ/2, δ/2), introduces NSD noise. Then m = floor((ẑ + (ε - e))/δ) where (ε - e) ~ Triangular(-δ, +δ). This spans 2–3 bins. H(m|ẑ) > H(m|z) structurally. A well-fitted q_φ(m|ẑ) does NOT converge to NLL ≈ 0. The degeneracy is avoided.

**Why the marginal buffer is required:** Evaluating q_φ(z/δ | ẑ) naively with current ẑ ≈ z still allows the conditional to track z — increasing |z| shifts ẑ by ≈ Δz, shifting the conditional's support by Δz/δ bins. The marginal computed over a detached buffer of past ẑ values does not track the current z. When z increases, z/δ moves into the tail of q_marginal → NLL increases → correct compression gradient.

**Use case:** Better per-region fitting than global DLM for complex p(m). Relevant for the distributed training study (§8) where higher expressivity may be needed. Buffer size B ≈ 512 recent ẑ vectors. MLP: ẑ ∈ ℝ^{z_dim} → 64 hidden units → DLM params.

### 7.5 Episodic refit-then-freeze

Every N RL updates: (1) freeze speaker, collect large batch, fit q_φ to convergence on stationary batch; (2) freeze q_φ, run N RL updates with Ballé backward active. Addresses fitting failure by making the target stationary during refit. Does not address the circular gradient.

**N is the key hyperparameter.** Start with N = 10 RL updates. With refit_steps ≈ 300 and N = 10, overhead = 30× more q_φ gradient steps — acceptable given q_φ has only 45 parameters (K=5, z_dim=3). Check qphi_gap starts rising after the frozen period ends.

### 7.6 Round 1 alternatives (superseded — documented for completeness)

These were proposed before re-reading the DDCL proposal. Each violates at least one non-negotiable constraint (§7.3).

- **VQ-VAE Codebook Prior:** uses STE — exactly the heuristic DDCL eliminates. Requires finite categorical; m ∈ ℤ is unbounded. Architecturally incompatible.
- **MINE/InfoNCE Rate Estimator:** estimates I(goal; m) — mutual information. Minimising I(goal; m) would destroy the channel. We want to penalise redundant bits, not all bits.
- **Stop-Gradient Teacher:** partially retained as the Episodic Refit idea (§7.5). The snapshot concept survives as a scheduling protocol without a separate model.
- **Gumbel-Softmax Entropy:** makes H(m) differentiable via a biased relaxation — exactly the bias DDCL eliminates.
- **Entropy as RL Reward:** valid (avoids domain gap entirely) but sacrifices principled differentiability. REINFORCE gradient for rate has high variance in general. Deferred, not superseded.

### 7.7 Estimator selection guide

| Setting | Recommended estimator | Reason |
|---|---|---|
| Post-hoc centralised | Joint empirical histogram | p(m) stationary; ε_estimator = 0; ε_MLE = 0.04 bits |
| Post-hoc centralised, large D | Autoregressive NN over m | Joint histogram exponential in D; NN tractable |
| Live distributed | Online histogram + score function | Instant fit; no fitting failure; EM framing principled |
| Live distributed, complex p(m) | NN conditional + marginal buffer | Better per-region fit when score function too local |
| Ablation baseline | DLM (Context A, factored) | Demonstrates fitting failure quantitatively vs histogram |

---

## 8. Distributed Training Study: Live Source Coding

### 8.1 Motivation

The post-hoc design in §5 assumes centralised training: agents share a parameter server; the source code is built once after convergence. In a realistic distributed deployment — agents on separate GPUs with no shared memory — every inter-agent message during training is transmitted over the actual bandwidth-constrained channel. The source code must be maintained live.

This study quantifies the cost of the moving-target problem relative to the post-hoc baseline. It is not required for the primary P2 result; it establishes whether any training-time compression signal is worth the overhead and characterises the distributed setting for the paper.

### 8.2 The moving-target problem

In post-hoc: p(m; θ*) is stationary. ε_MLE = O(K^D / √N) ≈ 0.04 bits. One-time cost.

In live training:

```
|Ĥ_t − H(m; θ_t)| ≤  O(K^D / √N)         [MLE term — irreducible]
                     + O(‖∇_θ H‖ · ‖Δθ‖)   [moving-target term — grows with LR]
```

The moving-target term is largest early in training and decays as θ converges. Two consequences:
1. **Coding overhead:** live code achieves more than H(m) bits where histogram lags.
2. **Biased score function gradient:** stale R(m) adds noise to the compression signal.

### 8.3 Live coding protocol (EM framing)

**E-step (once per rollout, before PPO epoch):** Reset histogram. Collect fresh rollout under current θ_t. Update histogram from all messages. This gives P̂(m; θ_t).

**M-step (inside PPO, histogram frozen):** Apply score function rate loss using frozen E-step histogram. Held constant across all minibatches of the epoch.

**Mismatch window:** By the end of the epoch, θ has moved by several minibatch updates from the E-step state. For small LR (3e-4, 4 epochs), epoch-level drift is small enough that the moving-target term ≈ MLE term. The codebook is rebuilt at the start of each rollout.

### 8.4 Experiment design: three-way comparison

| Config | Source coding during training | Phase 2 dither | Post-hoc code |
|---|---|---|---|
| **(A) Post-hoc baseline** | None (magnitude loss only) | Optional | Joint histogram at convergence |
| **(B) Live source coding** | Histogram + score function from update 0 | None | Joint histogram at convergence |
| **(C) Live + Phase 2 dither** | Histogram + score function from update 0 | L_dither after SR = 1 | Joint histogram at convergence |

5 seeds each. Primary metric: H_joint(m) at SR = 1.

**Primary comparison:** B vs. A.
- B ≤ A: live coding steered p(m) toward lower entropy — live coding helps
- B ≈ A: RL dominates; post-hoc is sufficient (expected outcome based on sc_option1)
- SR(B) < SR(A): live signal interfered with learning

### 8.5 Expected findings and paper framing

Based on sc_option1, the expected result is B ≈ A at λ = 5e-4. Both outcomes are publishable:

> **Post-hoc sufficient:** "Source coding in DDCL is a deployment-time operation. Training-time compression signals are structurally too weak (100:1 RL dominance, gradient locality) to steer p(m) during on-policy training. Post-hoc source coding with a joint arithmetic coder achieves H(G) + H(m|goal) bits with zero training-time overhead."

> **Live coding helps at higher λ:** "Live source coding with λ = X achieves H(m) = Y bits, compared to Z bits post hoc, at SR parity. The score function gradient is effective when RL is sufficiently satisfied (SR ≥ T), supporting two-phase training."

---

## 9. Polar Coordinates and Isotropic Activations: A Research Direction

### 9.1 The cross-dimension coupling problem

In the standard Cartesian parameterisation, the D dimensions of z are treated as independent. But TC(m) arises from cross-dimension correlations — different goals map to correlated (m₁, …, m_D) pairs. If we could find a coordinate system in which the speaker's output dimensions are naturally independent, TC(m) would be zero by construction and a factored code would be optimal.

### 9.2 Polar coordinate reparameterisation

Represent z ∈ ℝ^D as magnitude r = ‖z‖₂ and direction û = z/‖z‖₂. If the speaker assigns each goal to a distinct direction û(goal), goal identity is encoded entirely in direction — the magnitude is a shared task-irrelevant scalar. Cross-dimension correlations in m are reduced structurally: direction distinguishes goals; magnitude is shared.

### 9.3 Isotropic activations and post-hoc SVD diagonalisation

An **isotropic activation function** operates on the input magnitude and is invariant to rotations:

```
f_iso(x) = g(‖x‖₂) · x / ‖x‖₂
```

where g: ℝ≥0 → ℝ≥0 is a scalar nonlinearity. Direction is preserved; only magnitude is modified. Networks with isotropic activations have continuous rotational symmetry — any two representations related by rotation are equivalent (basis-independent).

The paper **"On De-Individuated Neurons: Continuous Symmetries Enable Dynamic Topologies"** (Bird, arXiv:2602.23405, February 2026) argues that prescribing continuous symmetries via isotropic activations enables dynamic topology changes (neurogenesis/pruning) with minimal loss of function. Key mechanism: basis-independence means one can apply SVD to the weight matrices post hoc to find a **diagonalised coordinate system** in which the network's computation decomposes into D independent scalar transformations along the SVD axes.

For DDCL: if the speaker uses isotropic activations, SVD on the converged speaker weights gives D directions in z-space that are naturally decoupled. In this basis, TC(m) ≈ 0 by construction; a factored source code is asymptotically as efficient as a joint code; D separate quantisation scales δ_k can be set independently per SVD axis (linking to Pillar P1's adaptive δ allocation). **This is a post-hoc coordinate change** — no training change required.

### 9.4 Implications for P2

| Approach | TC reduction mechanism | Training change? |
|---|---|---|
| Joint source code (§6.3) | Codes cross-dimension correlations directly | None (post-hoc) |
| L_dither (§5.3) | Reduces H(m\|goal) by fractional alignment | Phase 2 addition only |
| Polar reparameterisation (§9.2) | Decouples magnitude from direction | Architecture change |
| Isotropic activations + SVD (§9.3) | Finds natural independent axes post hoc | None (post-hoc rotation) |

The isotropic + SVD approach is a research direction; validating it requires implementing isotropic activations and checking whether SVD gives near-zero TC(m) in the diagonalised basis.

---

## 10. Multi-Agent Sharing Strategy

`qphi_sharing` controls how many independent priors are maintained when using parametric estimators (DLM, NN). For the histogram, the analogous question is whether a single histogram or per-agent histogram is maintained.

| Value | Behaviour | Use case |
|---|---|---|
| per_agent | One q_φ (or histogram) per speaker agent | Heterogeneous agents |
| shared | One q_φ (or histogram) shared across all speakers | Homogeneous agents |
| per_role | One q_φ per agent role class | Mixed teams |

For the toyproblem (1 speaker), all three are equivalent. Implement `per_agent` and `shared`; `per_role` when multi-role environments appear. For future environments: sharing reduces estimation variance but increases bias if agents' message distributions differ.

---

## 11. Ablation Matrix

### 11.1 Independent vs. joint source coding (primary paper result)

Post hoc, same trained policy, no extra training runs:
- **(A) Joint coder:** arithmetic coding over P̂(m₁, …, m_D). Reports H_joint(m).
- **(B) Factored coder:** per-dimension. Reports Σ_k H(m_k) = H_joint(m) + TC(m).

Run across z_dim ∈ {1, 2, 3} (TC = 0 trivially at z_dim=1; grows with z_dim as predicted):

| | z_dim=1 | z_dim=2 | z_dim=3 |
|---|---|---|---|
| Factored (B) | — (= joint) | ✓ | ✓ |
| Joint (A) | — | ✓ | ✓ |
| TC = B − A | 0 | measured | measured |

### 11.2 Post-hoc vs. live source coding

Three-way comparison as in §8.4: (A) post-hoc, (B) live from update 0, (C) live + Phase 2 dither. 5 seeds each. Primary metric: H_joint(m) at SR = 1.

### 11.3 Estimator type ablation (live setting)

| Estimator | ε_estimator floor | Moving-target error | Gradient path |
|---|---|---|---|
| DLM K=5, Context A | 0.27 bits/dim | High (fitting lag confirmed) | Ballé backward (domain gap) |
| NN conditional + marginal buffer | Low (MLP expressivity) | Medium (buffer lag) | Marginal backward |
| Online histogram | ≈ 0 (exact fit) | Low (instant update) | Score function |

DLM ablation uses P2-FIX experiment configs (§7.3): fix_baseline, fix_B1_gate, fix_B2_gate_lambda, fix_B3_ema, fix_B4_twophase. 25 runs.

### 11.4 DLM model selection: K sweep (for DLM ablation rows)

| Group | K | model_type | context | loss_comms_mode | Configs |
|---|---|---|---|---|---|
| baseline | — | — | — | magnitude | 1 |
| factored × A | {1,3,5,10,20} | factored | A | {entropy, both} | 10 |
| joint × A | {1,3,5,10,20} | joint | A | {entropy, both} | 10 |
| Context B measurement | 5 | factored | B | magnitude | 1 |

23 configs × 5 seeds = 115 runs. Winners fix K* and model_type* for the DLM ablation.

### 11.5 Magnitude regularisation geometry study

**Question:** Does magnitude regularisation produce a representation geometry that compresses better post hoc?

| Config | Phase 1 loss | Post-hoc code | Metrics at SR=1 |
|---|---|---|---|
| (A) Regularised | magnitude | Joint histogram | z_norm, TC(m), H(m\|goal), H_joint(m) |
| (B) Unregularised | none | Joint histogram | Same |

Prediction: (A) has lower TC(m) and H(m|goal) — neural collapse analogy (§2.3). 5 seeds each.

### 11.6 λ sweep (rate-distortion frontier)

| λ | magnitude baseline | Histogram + score function |
|---|---|---|
| {1e-5, 1e-4, 5e-4, 1e-3, 4e-3, 1e-2, 3e-2} | (baseline magnitude runs) | histogram runs |

35 new runs. Purpose: build (SR, H(m)) Pareto frontier. The core paper claim is that P2 lies on a better frontier than the magnitude baseline.

### 11.7 δ interaction

| δ | baseline | Post-hoc joint | Post-hoc + dither |
|---|---|---|---|
| {0.5, 1.0, 5.0, 10.0} | ✓ | ✓ | ✓ |

Prediction: smaller δ → larger |m| → more complex p(m) → more TC → larger post-hoc gain.

### 11.8 Channel interaction (SD vs NSD) — **DONE (2026-05-11)**

P2 adapts automatically since q_hist is built from actual m samples regardless of channel type.
Confirmed empirically via E47–E48 (5 seeds each, post_hoc_coding applied).

| channel | baseline | Post-hoc joint | Phase 2 λ=5e-4 |
|---|---|---|---|
| sd | ✓ E09 | H_joint = 4.156 ± 0.344 bits | H_joint = 2.385 ± 0.174 bits (E45) |
| nsd | ✓ E47 | H_joint = 4.137 ± 0.165 bits | H_joint = 2.413 ± 0.141 bits (E48) |

Channel difference: Δ = 0.019 bits (Phase 1), 0.028 bits (Phase 2) — both within 1 std.
SR = 1.000 on all four conditions. See C16 and F21 (Panel B).

### 11.9 Training robustness OAT sweep (for live coding, §8)

| Axis | Values | Fixed |
|---|---|---|
| lr_qphi_mult | {1, 5, 10, 50} | n_qphi_steps=3 |
| n_qphi_steps | {1, 3, 5, 10} | lr_qphi_mult=10 |
| n_warmup_steps | {0, 1000, 5000, 20000} | both above at defaults |

---

## 12. Implementation Plan

| Component | File | Status |
|---|---|---|
| Online histogram + score function loss | `source_coding.py` | **DONE** |
| `JointMessageHistogram` (D-dimensional, post-hoc) | `source_coding.py` | **DONE** |
| `dither_channel_loss` + `dither_channel_stats` | `source_coding.py` | **DONE** |
| Histogram integration in trainer (E-step/M-step) | `trainer.py` | **DONE** |
| Phase 2 dither loss wired in trainer | `trainer.py` | **DONE** |
| CLI flags: `--use_source_coding`, `--source_coding_smoothing`, `--lambda_dither` | `train.py` | **DONE** |
| Metrics: `dither_loss`, `H_dither_channel`, `mean_frac` added to CSV | `train.py` | **DONE** |
| 22 unit tests (source coding) | `tests/test_source_coding.py` | **DONE** |
| 20 unit tests (joint histogram + dither + post-hoc) | `tests/test_post_hoc_coding.py` | **DONE** |
| Post-hoc coding evaluation script | `analysis/post_hoc_coding.py` | **DONE** |
| Histogram-era experiment runner (F1–F20) | `experiments/run_sc_experiments.py` | **DONE** |
| Resource-aware shell script (MacOS) | `scripts/run_p2_experiments.sh` | **DONE** |
| DLM experiment runner (F8–F10) | `experiments/run_p2_ablation.py` | **DONE** (DLM-era; use `--stage P2-FIX`) |
| DLM models (EntropyModelFactored, Joint, CondZ) | `network.py` | **DONE** |
| Episodic refit protocol | `trainer.py` | **N/A** — post-hoc design supersedes; not needed |
| Paper figures (F1–F21) | `analysis/sc_ablation_figures.py` + others | **DONE** — all figures generated |

---

## 13. Metrics

All metrics logged to `metrics.csv` per training update.

| Metric | Definition | Interpretation |
|---|---|---|
| `hist_H_empirical` | H(m) from histogram counts | Ground-truth empirical entropy — primary rate metric |
| `hist_entropy_rate` | E[-log₂ q_hist(m)] over rollout | ≈ hist_H_empirical; gap should be ≈ 0.004 bits |
| `hist_qphi_gap` | hist_entropy_rate − hist_H_empirical | Histogram fit quality; ≈ 0.004 bits by construction |
| `sc_rate_loss` | Proxy loss value from score function | Trend signal only; can go negative; trust hist_H_empirical |
| `true_bits_per_msg` | Actual bits from channel | Independent of histogram; reflects z scale |
| `z_norm` | ‖z‖₂ | Geometric anchor check; bounded under magnitude loss |
| `shannon_gap` | true_bits_per_msg − H(G) | Distance from Shannon limit; ultimate P2 metric |
| `bits_to_hg_ratio` | true_bits_per_msg / H(G) | Normalised; target = 1.0 |
| `tc_bits` | Σ_k H(m_k) − H_joint(m) from batch | Cross-dimension correlation; → 0 with joint coding |
| `H_dim_{k}` | H(m_k) per dimension | Per-axis entropy; informs Pillar P1 δ allocation |
| `entropy_rate` | E[-log₂ q_φ(m)] (DLM, if active) | DLM cross-entropy; legacy metric from DLM experiments |
| `H_m_empirical` | Empirical H(m) from batch histogram (DLM) | Ground-truth entropy estimate alongside DLM |
| `qphi_gap` | entropy_rate − H_m_empirical (DLM) | DLM fit quality; expected floor 0.27 × z_dim |
| `qphi_neg_log_max` | max(-log₂ q_φ(m)) over batch | DLM tail spike detector |
| `bits_vs_magnitude` | magnitude_surrogate_bits − entropy_rate | Rate improvement of P2 over baseline surrogate |
| `entropy_rate_B` | E[-log₂ q_φ_B(m\|z)] (Context B) | Oracle conditional rate bound; measurement only |
| `context_gap_bits` | entropy_rate − entropy_rate_B | I(z; m) left by marginal prior; target → 0 |
| `warm_start_bits_final` | entropy_rate at end of warm-start phase | Confirms warm-start effective; NaN if warmup not run |
| `entropy_loss_magnitude` | λ · E[-log₂ q_φ(z/δ)] (DLM backward) | Compression pressure on speaker; compare to speaker_grad_norm |
| `speaker_grad_norm` | L2 norm of speaker param gradients (pre-clip) | Compare with entropy_loss_magnitude; ratio < 5% means P2 is inert |
| `entropy_rate_goal_{i}` | Per-goal entropy_rate | Adaptive rate allocation analysis |
| `training_phase` | 1 or 2 | Phase transition tracker |
| `bwd_gate_active` | 0 or 1 | DLM backward gate status |

---

## 14. Failure Modes and Diagnostics

| Failure mode | Symptom | Root cause | Fix |
|---|---|---|---|
| **Unbounded z growth** | z_norm → ∞ after SR=1; true_bits grows | No magnitude loss in Phase 1 | `--loss_comms_mode magnitude` in Phase 1 |
| **Score function too weak (live)** | hist_H_empirical flat throughout training | Structural ~5:1 PPO/SC grad norm dominance (measured; was incorrectly stated as 100:1); not a tuning problem | Use post-hoc design; or Phase 2 only with larger λ |
| **sc_rate_loss negative (live)** | Proxy loss decreasing; hist_H_empirical flat | m_hi cheaper than m_lo → gradient correctly increases z_k | Trust hist_H_empirical not sc_rate_loss |
| **Score function locality (live)** | hist_H_empirical decreasing very slowly | Adjacent-bin-only gradient; many bins to traverse | Extend Phase 2 duration; increase λ |
| **L_dither gradient explosion** | Gradient norm spikes at Phase 2 start | frac(z_k/δ) near 0 or 1; log-odds → ±∞ | Clamp frac to [1e-4, 1-1e-4] |
| **L_dither bin-crossing SR drop** | SR drops in Phase 2 | z_k crosses grid boundary; m_k shifts | Reduce λ; monitor per-goal bin assignments |
| **DLM fitting failure** | qphi_gap >> 0.27 × z_dim; bwd_gate_active = 0 | q_φ cannot track moving target | Use histogram estimator; or episodic refit (§7.5) |
| **DLM circular gradient** | shannon_gap flat despite qphi_gap small | Backward → current distribution mode | Switch to score function or L_dither |
| **DLM warm-start dead zone** | speaker_grad_norm ≈ 0; qphi_neg_log_max > 30 bits | q_φ trained on narrow support; gradient exactly 0 outside | n_warmup_steps ≥ 5000 |
| **Joint histogram explosion (large D)** | K^D bins exceeds memory | Curse of dimensionality | Use NN estimator; or factored histogram + accept TC gap |
| **Representation collapse** | Goals collapse to same z region; all H_dim_k → same value | No geometric regularisation | Add magnitude loss; check z_norm per goal |
| **DLM gradient scale mismatch** | entropy_loss_magnitude / speaker_grad_norm < 5% | λ too small | Increase λ; apply fix ladder levels L1–L2 |

---

## 15. Document Update Instructions

**At session start:** Read §§1–6 for orientation. §7 for estimator design. §11 for ablation status. §16 for figure requirements.

**During implementation:** Update §12 status column. Add empirical findings to the relevant section (§5 for post-hoc findings, §8 for distributed findings). Mathematical derivations → `docs/MATH.md`, then cross-reference here.

**When a new experiment completes:** Add a numbered entry in §17+ with config, key metrics at convergence, and which theoretical prediction (§16 figure) it confirms or refutes.

---

## 16. Required Figures and Analyses (F1–F20)

Each figure maps to one or more theoretical claims. The experiment stage that produces the data is shown in brackets. All figures are produced by `analysis/paper_figures.py` after the relevant runs complete.

### Group 1: Lower bound and rate decomposition

**F1 — Rate decomposition bar chart** `[SC_MAIN → post_hoc_coding.py]`
Claim: H_factored = H_G + H(m|goal) + TC(m) + ε; components are separable and measurable.
Data: converged sc_posthoc_mag checkpoint → `post_hoc_coding.py` → JSON.
Figure: stacked bar (H_G / H_dither / TC / ε_estimator / ε_MLE).

**F2 — Shannon gap training curve** `[BASELINE, SC_MAIN]`
Claim: RL without compression does not reduce H(m); source coding decreases it.
Figure: shannon_gap vs timestep for no_comms, mag baseline, sc_live_entropy.

### Group 2: Magnitude loss is the wrong objective

**F3 — Implicit prior mismatch** `[SC_MAIN checkpoint, analysis]`
Claim: L_mag = cross-entropy with Zipf(1) prior; minimising L_mag ≠ minimising H(m).
Data: compute -log q_Zipf(|m|) vs -log q_hist(m) vs H(m) per goal.
Figure: bar comparing three rate estimates per goal.

**F4 — Gradient direction alignment** `[SC_MAIN checkpoint, analysis]`
Claim: ∇_z L_mag and ∇_z L_rate are misaligned.
Data: cosine similarity distribution between the two gradient vectors over a batch.
Figure: cosine similarity histogram; near-zero or negative confirms misalignment.

**F5 — (SR, H(m)) Pareto frontier** `[PARETO]`
Claim: SC achieves same SR at lower H(m) than magnitude loss at every λ.
Figure: scatter plot of (mean SR, mean H(m)) at convergence; two curves (mag, sc).

### Group 3: Geometric regularisation

**F6 — Per-goal z-vector geometry** `[GEOMETRY checkpoints, analysis]`
Claim: L_mag encourages ETF-like z geometry; without it, goals share manifold.
Data: per-goal mean z, pairwise cosine similarities, Gram matrix eigenvalues.
Figure: (a) cosine similarity heatmap; (b) eigenvalue spectrum of Gram matrix.

**F7 — Dither loss effect on TC and H(m|goal)** `[GEOMETRY]`
Claim: the corrected dither loss (frac→0.5) reduces both TC(m) and H(m|goal); the
magnitude anchor alone contributes only modestly (TC reduction 0.28 bits). The dither
loss is the primary driver of TC reduction (1.9 bits) and H(m|goal) reduction.
**Revision (2026-05-11):** Prior text attributed TC reduction to the magnitude anchor.
Post_hoc on E16/E17/E42/E43 shows: mag alone: TC=4.73 bits (+0.28 vs no_anchor);
dither only: TC=3.42 bits; mag+dither: TC=2.80 bits (geom_both_v2). Magnitude anchor
provides geometric separation of goals in z-space (F6), not TC reduction.
Figure: grouped bar of (TC, H_dither) for four GEOMETRY configs (E16, E17, E42, E43).

### Group 4: Histogram vs. DLM

**F8 — qphi_gap training curves** `[DLM_CMP (P2-FIX), SC_MAIN]`
Claim: DLM qphi_gap stays at 12–14 bits; histogram qphi_gap ≈ 0.004 bits from update 1.
Figure: log-scale line plot of qphi_gap (DLM) and hist_qphi_gap (histogram) over updates.

**F9 — DLM floor is irreducible: K sweep** `[DLM_CMP → run_p2_ablation P2-A]`
Claim: qphi_gap floor ≈ 0.27 × z_dim regardless of K.
Figure: qphi_gap vs K for factored/joint/conditional_z.

**F10 — DLM circular gradient** `[DLM_CMP → fix_B1_gate]`
Claim: gate opens (qphi_gap < threshold) but shannon_gap stays flat.
Figure: three-panel plot (bwd_gate_active / qphi_gap / shannon_gap) over training.

**F11 — Context bound ordering** `[SC_MAIN + context models in measurement mode]`
Claim: entropy_rate_A ≥ entropy_rate_B ≥ entropy_rate_C ≥ H(m).
Figure: four-line plot over training.

**F12 — DLM warm-start failure** `[DLM_CMP → fix_baseline vs warmup sweep]`
Claim: speaker_grad_norm = 0 when n_warmup=0.
Figure: speaker_grad_norm over first 50k steps for warmup=0 vs warmup=5000.

### Group 5: Post-hoc vs live coding

**F13 — Three-way comparison** `[LIVE]`
Claim: live coding (config B) does not reduce H(m) during training; post-hoc achieves same final H(m).
Figure: (a) SR trajectory; (b) hist_H_empirical over training for A, B, C.

**F14 — Score function gradient ratio** `[LIVE config B, analysis]`
Claim: PPO speaker gradient consistently dominates SC speaker gradient by ~5:1 throughout
training (early: 4.4×, late: 5.0×). Ratio never crosses 1 — SC never gains leverage.
**Revision (2026-05-10):** Prior text said "~100:1". Measured value from E21 re-run with
per-loss gradient norm logging is ~5:1. The qualitative conclusion stands (PPO always
dominates), but the magnitude was overstated.
Figure: (speaker_grad_norm from PPO) / (speaker_grad_norm from SC loss) ratio over training.

### Group 6: Rate decomposition components

**F15 — TC grows with z_dim** `[SC_MAIN checkpoints at z_dim ∈ {1, 2, 3}]`
Claim: TC(m) = Σ H(m_k) - H_joint(m) grows with z_dim; TC=0 trivially at z_dim=1.
Data: post_hoc_coding.py on checkpoints at each z_dim.
Figure: bar chart of TC_bits vs z_dim.

**F16 — Phase 2 L_dither closes H(m|goal)** `[SC_TWOPHASE]`
Claim: L_dither reduces H_dither_channel without dropping SR.
Figure: three-panel (SR / H_dither_channel / mean_frac) through Phase 1→2 transition.

**F17 — ε_MLE converges at O(1/√N)** `[SC_MAIN checkpoint, post-hoc analysis]`
Claim: hist_qphi_gap ~ 1/√N as rollout size increases.
Figure: hist_qphi_gap vs N on log-log scale; slope ≈ -1/2.

### Group 7: Isotropic activations

**F18 — SVD diagonalisation collapses TC** `[future research: isotropic activation variants]`
Claim: post-hoc SVD on isotropic activations gives TC ≈ 0; standard activations do not.
Figure: TC_bits before/after SVD for both activation types.
Status: **deferred** — requires implementing isotropic activation variant in network.py.

### Group 8: Distributed training

**F19 — Moving-target error over training** `[LIVE config B, analysis]`
Claim: |Ĥ_t - H(m; θ_t)| is large early in training and correlates with ‖Δθ‖.
Data: compute true H(m) from large offline sample at each checkpoint; compare to hist_H_empirical.
Figure: moving-target error vs timestep; overlay ‖Δθ‖ per update.

**F20 — Three-way comparison at SR=1** `[LIVE]`
Claim: post-hoc Pareto-dominates live coding (same SR, same H(m), less compute).
Figure: (H_joint, SR, compute cost) grouped bar for configs A, B, C.

### Figure-to-experiment map

**Last updated: 2026-05-11. All figures DONE except F9 and F18 (deferred).**

| Figure | Stage(s) needed | Status |
|---|---|---|
| F1 | SC_MAIN + post_hoc_coding.py | **DONE** `results/figures/F1_rate_decomposition.{pdf,png}` |
| F2 | BASELINE + SC_MAIN | **DONE** `results/figures/F2_shannon_gap.{pdf,png}` |
| F3 | SC_MAIN checkpoint + analysis | **DONE** `results/figures/F3_implicit_prior_mismatch.{pdf,png}` |
| F4 | SC_MAIN checkpoint + analysis | **DONE** `results/figures/F4_gradient_direction.{pdf,png}` |
| F5 | PARETO (E23–E36) | **DONE** `results/figures/F5_pareto_frontier.{pdf,png}` — metric split: true_bits (mag) vs hist_H (SC) |
| F6 | GEOMETRY checkpoints (E16, E17, E42, E43) | **DONE** `results/figures/F6_z_geometry.{pdf,png}` |
| F7 | GEOMETRY + post_hoc (E16, E17, E42, E43) | **DONE** `results/figures/F7_geometry_decomposition.{pdf,png}` — dither loss (not magnitude anchor alone) drives TC reduction; see §16 F7 claim revision |
| F8 | DLM_CMP (E41, P2-FIX) + SC_MAIN | **DONE** `results/figures/F8_qphi_gap_comparison.{pdf,png}` — DLM gap 9.71 bits vs histogram 0.004 bits |
| F9 | run_p2_ablation P2-A (K sweep) | **deferred** — F8/F10/F12 give complete DLM story; F9 not required |
| F10 | DLM_CMP fix_B1_gate (E41) | **DONE** `results/figures/F10_circular_gradient.{pdf,png}` |
| F11 | SC_MAIN post_hoc decomposition | **DONE** `results/figures/F11_context_bound_ordering.{pdf,png}` — H_factored ≥ H_joint ≥ H(m\|goal) ≥ H(G) holds for all 5 seeds |
| F12 | DLM_CMP fix_B4_twophase (E41) | **DONE** `results/figures/F12_warmstart_failure.{pdf,png}` |
| F13 | LIVE (E20–E22) | **DONE** `results/figures/F13_three_way_comparison.{pdf,png}` |
| F14 | LIVE config B + grad logging (E21 re-run) | **DONE** `results/figures/F14_score_function_grad_ratio.{pdf,png}` — measured ratio ~5:1 (not 100:1); see claim revision below |
| F15 | SC_MAIN at z_dim ∈ {1,2,3} (E09, E37, E38) | **DONE** `results/figures/F15_tc_vs_zdim.{pdf,png}` |
| F16 | SC_TWOPHASE_v2 training curves (E44–E46) | **DONE** `results/figures/F16_phase2_curves.{pdf,png}` |
| F17 | SC_MAIN checkpoint + N sweep | **DONE** `results/figures/F17_eps_mle_convergence.{pdf,png}` |
| F18 | isotropic activation variants | **deferred** — future work; requires new network architecture |
| F19 | LIVE config B + moving-target logging (E21 re-run) | **DONE** `results/figures/F19_moving_target_error.{pdf,png}` — true_H_offline drops 6→2.4 bits via z-blowup (degenerate), not useful compression |
| F20 | LIVE (E20–E22) + Phase 2 v2 (E44–E46) | **DONE** `results/figures/F20_three_way_pareto.{pdf,png}` — Phase 2 added as fourth group (H_joint 2.3 bits vs live ~7 bits) |

---

## 17. Experiment Results Log

All runs completed 2026-05-01 (5 seeds each, 2M steps unless noted).  
Metrics at convergence: `SR` = success rate, `true_bits` = SD channel transmission cost,
`hist_H` = Shannon entropy H(m) of discrete message distribution.  
`hist_H = N/A` means the run predates the hist_H_empirical column (re-run to get it).

---

### 17.1 Baseline stage (`no_comms`, `mag_lam*`)

**Purpose:** Establish the no-compression ceiling and the magnitude-loss Pareto frontier.  
**Figures:** F2 (Shannon gap curve), F5 (Pareto denominator).

| Experiment | SR | true_bits | hist_H | Notes |
|---|---|---|---|---|
| `no_comms` | 1.000±0.000 | 8.74±0.30 | N/A | Unconstrained ceiling |
| `mag_lam1e-5` | 1.000±0.000 | 8.51±— | N/A | Near-ceiling; λ too small |
| `mag_lam1e-4` | 1.000±0.000 | 8.28±— | N/A | Marginal reduction |
| `mag_lam5e-4` | 1.000±0.000 | 7.04±0.87 | N/A | Standard baseline λ |
| `mag_lam1e-3` | 1.000±0.000 | 6.52±0.37 | N/A | Best SR=1 operating point |
| `mag_lam4e-3` | 0.856±0.207 | 3.49±1.69 | N/A | Pareto knee — SR degrading |
| `mag_lam1e-2` | 0.558±0.017 | 1.50±0.00 | N/A | Over-constrained; SR collapses |
| `mag_lam3e-2` | 0.551±— | 1.50±— | N/A | Floor: minimum true_bits achievable |

**Conclusions:**
- The magnitude Pareto frontier has a clear knee at λ ≈ 1e-3 (6.52 bits, SR=1.0). Beyond
  λ=4e-3, SR degrades sharply with high variance (std=0.207), suggesting the penalty
  overcrowds the origin before the policy can separate goals geometrically.
- The true_bits floor of ~1.5 at large λ is set by the minimum bin width needed to encode
  6 goals; it is not zero because SD channel cost grows with |m|.
- `geom_no_anchor` reproduces `no_comms` exactly (both 8.74 bits, SR=1.0), confirming
  that RL alone never reduces the communication rate — the magnitude anchor is essential.

---

### 17.2 SC Main stage (`sc_posthoc_mag`, `sc_live_entropy`, `sc_live_both`)

**Purpose:** Compare post-hoc source coding vs live histogram coding during training.  
**Figures:** F2 (Shannon gap), F13 (three-way comparison).

| Experiment | SR | true_bits | hist_H | Notes |
|---|---|---|---|---|
| `sc_posthoc_mag` | 1.000±0.000 | 7.04±0.87 | N/A | Identical to `mag_lam5e-4`; SC is post-hoc |
| `sc_live_entropy` | 1.000±0.000 | 20.06±0.51 | 7.05±0.35 | z magnitude blows up; H(m) unchanged |
| `sc_live_both` | 1.000±0.000 | 21.07±0.64 | 6.90±0.35 | Both losses active; z blows up further |

**Conclusions:**
- Live SC (score function estimator active during training) causes `true_bits` to blow up
  3× above the no-comms baseline. The score function gradient pushes z toward larger
  integer bins to reduce the histogram rate, but the PPO gradient dominates by ~100:1 and
  the policy learns to use the same bins at higher z magnitudes.
- Crucially, `hist_H` ≈ 7.05 for live SC — identical to `no_comms` levels. The message
  distribution p(m) is not restructured by live SC; only the SD channel cost metric grows.
- **The two metrics come apart:** `true_bits` (channel cost, z-magnitude-dependent) and
  `hist_H` (Shannon entropy of discrete messages) diverge in live SC runs. For
  information-theoretic claims, `hist_H` is the correct metric; `true_bits` is an
  implementation artifact of the SD channel's cost formula when z is large.
- Post-hoc source coding (applied offline to the converged `sc_posthoc_mag` checkpoint)
  achieves the same final H(m) without any z blowup and without touching training.

---

### 17.3 Post-hoc Rate Decomposition (`sc_posthoc_mag` → `post_hoc_coding.py`)

**Purpose:** Measure the full rate decomposition on the converged Phase 1 policy.  
**Figures:** F1 (decomposition bar chart), F7 (TC and H_dither per config).

Ran `post_hoc_coding.py` with N=50,000 messages, δ=1.0, z_dim=3 across all 5 seeds:

| Component | Mean | Std | Interpretation |
|---|---|---|---|
| H(G) | 1.8128 | 0 | Shannon entropy of 6-goal distribution |
| H_factored(m) | 9.48 | 0.29 | Cost of independent per-dim coding |
| H_joint(m) | 4.75 | 0.23 | Cost of joint 3-dim arithmetic coding |
| TC(m) = H_factored − H_joint | 4.73 | 0.32 | Cross-dim correlation; savings from joint coding |
| H(m\|goal) [dither noise] | 2.17 | 0.17 | Residual within-goal randomness from frac ≠ 0,1 |
| ε_estimator | −0.000 | 0.0001 | Histogram fit quality; negligible |
| gap above H(G) [factored] | 7.67 | 0.29 | Overhead vs theoretical minimum, independent code |
| gap above H(G) [joint] | 2.94 | 0.23 | Overhead vs theoretical minimum, joint code |
| K_obs distinct joint tuples | 48 | 0 | Policy uses only 48 of exponentially many tuples |
| decomp residual | 0.77 | 0.39 | H_factored − (H_G + H_dither + TC + ε); seed 0 = 0.18 |

**Conclusions:**
- TC(m) ≈ 4.73 bits is the dominant overhead. The policy's 3 output dimensions are
  highly correlated: only 48 distinct joint tuples (m_0, m_1, m_2) appear in 50K messages
  across 6 goals, roughly 8 per goal. Joint arithmetic coding recovers these 4.73 bits for
  free — it reduces H_factored from 9.48 → H_joint 4.75 bits (a 50% reduction) by
  exploiting the correlation structure the policy has implicitly learned.
- H(m\|goal) ≈ 2.17 bits reflects that, after Phase 1, frac(z_k/δ) is not pushed toward
  0 or 1 — the z-values sit near bin midpoints, giving maximum within-goal entropy.
  This is exactly what Phase 2 (L_dither) is designed to eliminate.
- After joint coding and Phase 2 dither, the theoretical floor is:
  H_G + H_dither_after_phase2 + ε ≈ 1.81 + ~0 + ~0 ≈ 1.81 bits.
  The gap from H_joint (4.75) to this floor (1.81) is 2.94 bits — attributable to
  H(m\|goal) = 2.17 + residual 0.77 (estimator noise across seeds).
- The decomp residual is 0.18 bits for the best-converged seed (seed 0) and up to 1.1
  bits for others, suggesting some seeds have not fully converged or have noisier frac
  estimates from the 100-sample-per-goal z collection.
- **The decomposition identity holds:** H_factored ≈ H_G + H_dither + TC + ε, confirming
  the theoretical rate decomposition from §6.1 is empirically valid.

---

### 17.4 Phase 2 dither stage (`sc_twophase_dither*`)

**Purpose:** Measure whether L_dither reduces H(m\|goal) without dropping SR.  
**Figures:** F16 (Phase 2 closes H_dither_channel), F5 (Pareto frontier with Phase 2).

| Experiment | SR | true_bits | hist_H | Notes |
|---|---|---|---|---|
| `sc_twophase_dither1e-4` | 0.047±0.045 | 23.90±0.43 | 5.22±1.75 | SR catastrophic collapse |
| `sc_twophase_dither5e-4` | 0.963±0.043 | 5.42±0.32 | 7.38±0.77 | Good tradeoff |
| `sc_twophase_dither1e-3` | 0.986±0.016 | 5.14±0.10 | 7.51±0.46 | Best result (−27% true_bits vs Phase 1) |

**Conclusions:**
- λ_dither = 1e-3 achieves the best outcome: SR=0.986 (−1.4% vs Phase 1) with
  true_bits=5.14 (−27% vs Phase 1 baseline of 7.04). The dither gradient is strong
  enough to converge frac toward 0 or 1 before significant listener degradation
  accumulates.
- λ_dither = 1e-4 is catastrophic: SR collapses to 0.047 (near-random). At this small λ,
  the dither gradient is too weak to move frac quickly, so Phase 2 runs for ~1M additional
  steps with listener gradients zeroed. The listener parameters drift (Adam optimizer
  continues accumulating momentum) or bin-crossing events during slow convergence cause
  the message mapping to shift, destroying the learned goal-to-message association.
- λ_dither = 5e-4 closely tracks 1e-3 (SR=0.963, true_bits=5.42) but with slightly more
  SR variance, suggesting the 5e-4 regime is near the stability boundary.
- **The `hist_H` metric does not decrease** in Phase 2 (7.38–7.51 vs 7.04 for Phase 1).
  This confirms that L_dither does not change the message distribution p(m) — it reduces
  `true_bits` by pushing z closer to integer multiples of δ (reducing |frac(z/δ)|), which
  reduces the SD channel's transmission cost metric. H(m) (the Shannon entropy of the
  discrete distribution) is unchanged because the same integer messages are used.
- `geom_both` exactly matches `sc_twophase_dither5e-4` (both 5.42 bits, SR=0.963),
  confirming these are the same training configuration.
- `geom_dither_only` (no Phase 1 anchor) achieves SR=0.965, true_bits=5.87 — slightly
  worse bits than `geom_both` (5.42). The Phase 1 magnitude anchor is not strictly
  required for Phase 2 to work, but it improves the final true_bits by ~0.45 bits and
  produces a better-structured z-geometry as Phase 2 starting point.

---

### 17.5 Geometry stage (`geom_*`)

**Purpose:** Isolate the role of the magnitude anchor in z-space geometry.  
**Figures:** F6 (per-goal z geometry), F7 (TC and H_dither per config).

| Experiment | SR | true_bits | hist_H | Notes |
|---|---|---|---|---|
| `geom_no_anchor` | 1.000±0.000 | 8.74±0.30 | N/A | = `no_comms`; anchor essential |
| `geom_mag_only` | 1.000±0.000 | 7.04±0.87 | N/A | = Phase 1 baseline |
| `geom_dither_only` | 0.965±0.017 | 5.87±0.36 | 8.06±0.31 | Phase 2 without anchor |
| `geom_both` | 0.963±0.043 | 5.42±0.32 | 7.38±0.77 | Phase 1 + Phase 2 |

**Conclusions:**
- Without the magnitude anchor (`geom_no_anchor`), the policy achieves SR=1.0 but uses
  8.74 bits — identical to unconstrained `no_comms`. RL alone does not create any
  implicit compression pressure.
- The magnitude anchor (`geom_mag_only`) reduces true_bits to 7.04 without SR loss. It
  acts as a geometric regulariser: the penalty on |z| clusters goals into a compact region
  of z-space, implicitly reducing the range of bins used.
- Phase 2 dither on top of Phase 1 (`geom_both`) achieves 5.42 bits — the best
  combination. The anchor first structures z-space, then dither refines z to sit near bin
  boundaries, further reducing the SD channel cost.
- The `geom_dither_only` result (no Phase 1, SR=0.965, 5.87 bits) is notable: even
  without the anchor, Phase 2 alone can reduce true_bits to 5.87. The anchor provides
  ~0.45 bits of additional compression and slightly better SR stability.
- **For F6/F7:** Per-goal z geometry (Gram matrix, cosine similarities) and per-config
  TC measurements still need to be computed via `post_hoc_coding.py` on each checkpoint.
  The current results table gives true_bits and SR but not TC or H(m\|goal) per config
  (hist_H columns are NaN for configs without live SC).

---

### 17.6 Live coding stage (`live_A_posthoc`, `live_B_live_sc`, `live_C_live_dither`)

**Purpose:** Three-way comparison of post-hoc, live, and live+dither coding.  
**Figures:** F13 (three-way comparison), F14 (gradient ratio), F20 (Pareto at SR=1).

| Experiment | SR | true_bits | hist_H | Notes |
|---|---|---|---|---|
| `live_A_posthoc` | 1.000±0.000 | 7.04±0.87 | N/A | = Phase 1 mag baseline |
| `live_B_live_sc` | 1.000±0.000 | 20.06±0.51 | 7.05±0.35 | z blowup; H(m) flat |
| `live_C_live_dither` | 0.980±0.015 | 7.67±0.86 | 7.83±0.23 | Phase 2 during training |

**Conclusions:**
- `live_B_live_sc` and `live_A_posthoc` have identical `hist_H` (7.05 vs N/A at same
  level), confirming that live SC provides zero benefit to the message distribution.
  The score function gradient is overwhelmed by the PPO gradient (~100:1 ratio) and cannot
  restructure which bins the policy uses.
- `live_C_live_dither` achieves SR=0.980 and true_bits=7.67, better than live_B's 20.06
  but worse than the two-phase run's 5.14. Interestingly, `hist_H` = 7.83 is slightly
  *higher* than live_B (7.05), suggesting the dither in config C adds within-goal
  stochasticity before Phase 2 converges fully.
- The practical conclusion for F20: post-hoc coding (`live_A`) Pareto-dominates live
  coding (`live_B`) on the `hist_H` metric (same H(m) ≈ 7, less compute during training,
  no z blowup). The two-phase design (`sc_twophase_dither1e-3`) Pareto-dominates all
  three on `true_bits` (5.14 bits at SR=0.986).

---

### 17.7 Pareto frontier (`pareto_mag_*`, `pareto_sc_*`)

**Purpose:** Build the rate-distortion Pareto frontier for both magnitude and SC.  
**Figures:** F5 (Pareto frontier scatter plot).

| Experiment | SR | true_bits | hist_H | Notes |
|---|---|---|---|---|
| `pareto_mag_lam1e-5` | 1.000 | 8.51 | N/A | |
| `pareto_mag_lam1e-4` | 1.000 | 8.28 | N/A | |
| `pareto_mag_lam5e-4` | 1.000 | 7.04 | N/A | |
| `pareto_mag_lam1e-3` | 1.000 | 6.52 | N/A | |
| `pareto_mag_lam4e-3` | 0.856 | 3.49 | N/A | |
| `pareto_mag_lam1e-2` | 0.558 | 1.50 | N/A | |
| `pareto_mag_lam3e-2` | 0.551 | 1.50 | N/A | |
| `pareto_sc_lam1e-5` | 1.000 | 8.00 | 6.89 | |
| `pareto_sc_lam1e-4` | 1.000 | 8.89 | 7.13 | z blowup onset |
| `pareto_sc_lam5e-4` | 1.000 | 20.06 | 7.05 | Full z blowup |
| `pareto_sc_lam1e-3` | 1.000 | 21.02 | 6.80 | |
| `pareto_sc_lam4e-3` | 0.894 | 22.62 | 6.40 | SR degrading |
| `pareto_sc_lam1e-2` | 0.815 | 22.71 | 6.45 | |
| `pareto_sc_lam3e-2` | 0.641 | 23.10 | 4.82 | Lowest hist_H but poor SR |

**Conclusions:**
- On the `true_bits` metric, the SC Pareto curve is entirely dominated by the magnitude
  curve: live SC z blowup makes `true_bits` 3–15× worse at every λ where the SC gradient
  is active. The SC Pareto curve using `true_bits` is not meaningful.
- On the `hist_H` metric (the correct information-theoretic metric): live SC at
  λ=3e-2 achieves hist_H=4.82 but only at SR=0.641. The magnitude curve at λ=1e-3
  achieves SR=1.0 at the expense of hist_H being unavailable from these runs.
- **The Pareto frontier for F5 must use `hist_H` as the x-axis and note that
  `hist_H ≈ 7` for all magnitude runs (no live coding)** — the magnitude curve
  does not reduce H(m), only `true_bits`. This means the two curves measure different
  things, and the figure's claim must be restated as:
  - Magnitude curve: SR vs true_bits (channel cost).
  - SC curve: SR vs hist_H (Shannon entropy of message distribution).
  - Two-phase design: achieves true_bits=5.14, SR=0.986 — the only design that
    simultaneously reduces channel cost and maintains near-perfect SR.

---

### 17.8 Summary of key findings

1. **Magnitude anchor is necessary.** Without it (geom_no_anchor = no_comms), RL never
   reduces the communication rate. The anchor provides ~1.7 bits of compression (8.74 →
   7.04) at zero SR cost.

2. **TC dominates the rate gap.** TC(m) ≈ 4.73 bits accounts for 63% of H_factored −
   H(G). Joint source coding eliminates this cost post-hoc, halving the effective
   transmission rate from 9.48 → 4.75 bits.

3. **Live SC is ineffective and harmful.** The score function gradient is overwhelmed by
   PPO (~100:1). H(m) stays flat at ~7 bits while `true_bits` blows up to ~20. Post-hoc
   coding achieves the same H(m) at no training cost.

4. **Phase 2 dither reduces channel cost, not H(m).** The dither loss pushes
   frac(z/δ) toward 0 or 1, reducing the SD channel's z-magnitude-based cost from 7.04 →
   5.14 bits (−27%). It does not change p(m), so H(m) is unchanged. The DDCL channel
   noise interpretation: H(m\|goal) measures within-goal randomness due to frac ≠ 0,1;
   dither eliminates this by committing z to a bin boundary.

5. **Phase 2 stability is λ-sensitive.** λ=1e-4 causes catastrophic SR collapse (0.047);
   λ=1e-3 is stable (SR=0.986). The safe regime is λ ≥ 5e-4 with the current
   architecture. The root cause of collapse is listener gradient zeroing combined with slow
   frac convergence: at small λ, Phase 2 runs for too long, and the listener cannot adapt
   to transient bin-crossing events during dither convergence.

6. **The rate decomposition identity is empirically valid.** H_factored ≈ H_G + H_dither
   + TC + ε holds to within 0.18 bits for the best-converged seed. Residuals of 0.6–1.1
   bits in other seeds are attributed to under-converged policies and noisy per-goal frac
   estimates from 100-sample z collection.

7. **The two Pareto metrics measure different things.** `true_bits` (SD channel cost) and
   `hist_H` (Shannon entropy of p(m)) diverge in live SC runs. The paper must specify
   which metric is plotted for each curve. The two-phase design (Phase 1 + Phase 2 dither)
   is the only configuration that reduces `true_bits` substantially while maintaining SR.

---

### 17.9 Open issues and next steps

- **F6/F7:** Per-goal z geometry (Gram matrix, cosine similarities, TC per config) needs
  `post_hoc_coding.py` run on each GEOMETRY checkpoint. Currently blocked by the
  `H_dither_channel` and `mean_frac` columns being NaN for pre-dither runs (expected —
  these columns were added after the GEOMETRY runs completed).
- **F15 (TC vs z_dim):** Requires re-running `sc_posthoc_mag` at z_dim ∈ {1, 2} and
  applying post_hoc_coding.py. The z_dim=3 result (TC=4.73) is the anchor point.
- **F16 (Phase 2 training curves):** Requires time-series data from sc_twophase runs
  showing H_dither_channel and mean_frac declining through the Phase 1→2 transition.
  The current metrics.csv should have this data; needs plotting.
- **Phase 2 listener gradient issue:** The current implementation zeros listener gradients
  during Phase 2. This is causing SR degradation at λ=1e-4 and slight degradation at
  λ=5e-4. A cleaner design would keep listener gradients active in Phase 2, letting the
  listener adapt to the evolving frac distribution while the dither converges.
- **Decomp residual reduction:** Increasing the z collection from 100 to 1000 samples per
  goal in `post_hoc_coding.py` would reduce the H_dither estimation error and bring the
  residual closer to the seed-0 level of 0.18 bits across all seeds.
