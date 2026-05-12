# Large-Scale Environment Extension

**Status:** SPECULATIVE — written before pillar experiments conclude (2026-05-12).
Assumptions and alternatives are ranked by confidence; update each section as toy
problem conclusions (C-numbers) arrive.

**Purpose:** Map every design decision in the toy problem to its corresponding assumption,
determine which assumptions survive scaling, and propose concrete alternatives for those
that do not. This is not an implementation plan — it is the design vocabulary for
the next phase.

**Reading order:** Read alongside the pillar docs. Cross-references use §P1/P2/P3/P4
for the toy-problem pillars and §L1–LN for sections of this document.

---

## 0. What "Large Scale" Means Here

We assume the target environment class is characterised by:

- **N agents:** N in the range 4–20 (cooperative MARL, e.g. SMAC, MPE, GRF)
- **Continuous partial observations:** each agent's obs ∈ ℝ^d_obs, d_obs ≈ 50–300
- **Continuous or large-discrete actions:** action ∈ ℝ^d_act or |A| >> 5
- **Multi-timestep episodes:** T ≈ 100–400 steps; communication occurs every step
- **CTDE training:** centralised critic, decentralised execution (standard MAPPO)
- **Heterogeneous importance:** not all agents need to communicate at all timesteps

The toy problem has N=2 (one speaker, one listener), discrete obs (6 goals in 2D),
discrete actions (5 directions), one-shot communication per episode, and a known
theoretical minimum rate H(G) = 1.813 bits. All of these differ at scale.

---

## 0.5 The Role-Collapse Problem: Agents Are Both Speakers and Listeners

This is the most structurally consequential difference between the toy problem and
large-scale MARL. Everything downstream of it needs to be re-examined.

### 0.5.1 Toy-problem architecture

The toy problem has two completely separate networks with disjoint parameter sets:

```
Speaker network θ_S:  obs_speaker → z → [channel] → z_hat
Listener network θ_L: [obs_listener, z_hat] → action → task loss

Gradient flows:
  Task loss → θ_L (via action)
  Task loss → θ_S (via STE: z_hat → z → θ_S)
  Comms loss → θ_S only (magnitude, rate, dither)
```

The speaker receives task gradients ONLY through the message path. The listener
receives task gradients ONLY through its action. Comms loss has zero gradient to
the listener's action parameters. This clean separation is the foundation of every
implementation choice in the pillar docs.

### 0.5.2 Large-scale architecture

At scale, each agent i maintains ONE policy network (or a weight-tied shared network
across all agents). This network simultaneously:
- Encodes its own observation obs_i and outgoing message z_{i→j}
- Receives and decodes incoming messages z_hat_{j→i}
- Outputs both an action and a message

```
Agent i's policy π_θ:  [obs_i, {z_hat_{j→i}}] → (action_i, z_{i→j})

Gradient flows to θ (the shared parameter):
  1. Own task loss (L_i): L_i → action_i → θ  [via own observation+action path]
  2. Receiver task loss (L_j): L_j → z_hat_{i→j} → z_{i→j} → θ  [via STE, across agents]
     (because z_{i→j} = message_head(obs_i; θ) and θ is shared)
  3. Comms loss on z_{i→j}: → θ  [same parameters, includes action head backbone]
```

All three gradient streams flow into the same parameter vector θ. There is no
separation between "speaker parameters" and "listener parameters."

### 0.5.3 Implications that cascade through all pillars

The following consequences must be addressed in each pillar's scale extension:

**A. Comms loss now directly competes with action quality.**
In the toy problem, the magnitude/rate/dither loss only affects the speaker, which
has no direct action output. At scale, comms loss on z_{i→j} flows through the
policy backbone — the same backbone that produces actions. A large comms penalty
can degrade actions even if messages are perfectly compressed.

**B. The RB detach trick does not translate directly.**
In the toy problem, the P4 implementation DETACHES z_hat from the listener so that
the speaker's task gradient flows only via the RB proxy (not via actor_loss). At
scale, "detaching received messages from the listener" would block the credit
assignment gradient that flows from agent j's loss through received messages back
to agent i's message encoder (via weight sharing). This gradient is load-bearing.

**C. Phase 2 listener-gradient zeroing is impossible.**
The toy problem's Phase 2 design zeroes listener and critic gradients to prevent
listener drift while the speaker compresses. At scale, the "listener" IS the
policy network. Zeroing listener gradients would freeze the entire agent, halting
all learning.

**D. The task gradient to the message encoder is amplified but noisier.**
With N agents and weight sharing, the message encoder receives task gradients from
all N(N-1) (receiver, sender) pairs simultaneously. This is more signal than the
toy problem's single path, but it is also more entangled with other agents'
non-stationary policies.

---

## 1. The Channel (SD / NSD) — What Holds and What Changes

### 1.1 What holds

**Schuchman dither linearisation (Theorem 2):**
> ẑ_k = z_k + e_k,  e_k ~ U(-δ/2, δ/2),  e_k ⊥⊥ z_k  ⟹  E[ẑ_k] = z_k,  ∂ẑ_k/∂z_k = 1

This is a channel property, not a task property. It holds for any speaker network and
any z dimension, as long as:
- The dither ε is drawn uniformly and independently per dimension per forward pass
- δ > 0 is finite

Both conditions are implementation-level guarantees. **Theorem 2 holds at any scale.**

**NSD sample consistency (Theorem 5 / §P3):**
> z_hat = z_hat_true = (m + 0.5)·δ at all times

This is also a channel implementation property. It holds regardless of the speaker
architecture, z_dim, N agents, or task structure. **Theorem 5 holds at any scale.**

**STE gradient exactness:**
The STE gradient ∂ẑ/∂z = 1 is exact under Schuchman (not an approximation). This
holds for any differentiable speaker producing z, regardless of network depth or width.
**The STE gradient holds at any scale.**

### 1.2 What changes

**δ scaling with z_dim:**
In the toy problem z_dim = 3 and δ = 1.0 works well. At scale, z_dim may be 16–256.
The range of z_k values produced by a deep speaker network may be much larger (e.g.,
|z_k| ≈ 5–20 vs. ≈2 in the toy problem) — so a fixed δ = 1.0 that worked for 6
discrete goals may create too many bins (fine-grained messages that listener cannot
distinguish) or too few (coarse messages that lose goal information).

**Alternative:** Per-channel δ (P1, §L2) becomes even more important at scale, since
different z_k dimensions will have very different natural scales. Auto-scaling δ to the
empirical std of z_k per dimension (e.g., δ_k = β · std(z_k) where β ∈ [0.5, 2]) is
a practical initialisation heuristic that the toy problem does not need.

**Message length (z_dim) selection:**
In the toy problem z_dim = 3 is chosen to slightly over-specify the task (6 goals need
≈2.58 bits, z_dim=3 gives ≤3 bits). At scale the information content of obs_i that is
relevant to agent j is task-dependent and unknown. The toy problem's clean H(G) oracle
does not exist.

**Alternative:** z_dim as a hyperparameter swept over {8, 16, 32, 64}. The post-hoc
rate decomposition (H_joint, TC) serves as the signal: if TC >> 0 at convergence,
z_dim is too large (dimensions are wasted on correlated info); if SR drops when z_dim
is reduced, increase it.

**Communication topology:**
In the toy problem there is one speaker and one listener (1:1, unidirectional). At scale
with N agents there are N(N-1) directed pairs (fully connected) or a subset (partial
graph). The channel operates independently on each (i→j) message. Total bandwidth
is O(N² · z_dim) bits per step.

**Alternative:** The dithering channel applies identically to each (i→j) message vector
independently. No architectural change needed — just instantiate N(N-1) independent
channels (or one shared channel module parameterised by sender/receiver embeddings if
weight sharing is desired). Per-pair δ_ij tensors would add O(N²·z_dim) parameters,
which is tractable for N ≤ 20, z_dim ≤ 64.

---

## 2. Pillar P1 — Per-Channel δ at Scale

### 2.1 Toy-problem assumption

A single speaker produces z ∈ ℝ³. Experiments (E53–E56) test whether the 3 learned
δ_k values differentiate. The task structure (2D goal → 3D message) provides a clear
reason to expect one wasted dimension.

### 2.2 What holds

The `PerChannelDelta` module (softplus, clamp, separate lr) is architecturally
independent of the task. It plugs in for any z_dim and any channel type.

### 2.3 What changes

**Gradient signal for δ_k becomes weaker with large z_dim:**
The task gradient ∂L/∂δ_k = ∂L/∂ẑ_k · (u_k − 0.5) is zero in expectation.
With z_dim=3, three δ_k parameters each receive B=1024 samples per update — sufficient
to estimate direction. With z_dim=64, the same budget gives each δ_k only B/64 ≈ 16
effective samples, which is much noisier. Adam's per-parameter moments may still
converge, but more slowly.

**Alternative A — Structured δ initialisation:**
Instead of softplus_inv(1.0) for all k, initialise δ_k based on the empirical std of
z_k from an initial rollout without channel: δ_k(0) = std(z_k). This makes the initial
message distribution roughly uniform per dimension, regardless of speaker scale.

**Alternative B — Grouped δ (fewer parameters):**
Rather than z_dim independent δ_k, learn G < z_dim group widths with each group
covering z_dim/G dimensions. For z_dim=64 and G=8, this is 8 parameters — similar
to the toy problem's 3. Grouping can be structured (e.g., by attention head if the
speaker is a transformer) or data-driven (cluster z_k dims by empirical entropy).

**Alternative C — Entropy-driven heuristic δ (E56 approach scaled up):**
The E56 heuristic (δ_k ∝ 1/H(m_k)) can be applied at Phase 2 onset at any z_dim.
Since it requires only a histogram call (O(z_dim · n_bins)), it scales cheaply. This
avoids the noisy gradient problem entirely. The cost is that it does not adapt
continuously — it is a one-shot assignment.

**Interaction with N agents and role collapse (§0.5D):**
With weight-tied agents and N(N-1) message channels, the task gradient to δ_k is
the sum of N(N-1) independent ∂L_j/∂δ_k terms (one per receiver j of agent i's
messages). This amplifies the gradient signal relative to the toy problem's single
path — potentially making P1 δ_k differentiation *easier* at scale, not harder.
However, the comms loss gradient to δ_k now competes with N(N-1) task gradients
simultaneously, which requires careful tuning of λ_comms relative to task scale.

A single shared PerChannelDelta across all agents is the right default for weight-tied
MARL (SMAC, MPE). Heterogeneous agents require per-agent δ vectors.

---

## 3. Pillar P2 — Histogram Rate Loss at Scale

### 3.1 Toy-problem assumption

`MessageHistogram` tracks per-dimension message counts in a dict (sparse bins, exact).
It is updated once per rollout from the full buffer. The key assumption is that the
histogram is a good estimator of p(m_k) — this requires:
1. Enough samples per bin (toy: 4096 per rollout, ~6 goals, so ~680 samples/goal/dim)
2. Stationarity: p(m_k) does not change within a rollout

Both hold tightly in the toy problem.

### 3.2 What holds

**The score-function proxy loss formula** (source_coding_rate_loss) is exact regardless
of z_dim or N. The structural form [R_hi − R_lo]/δ is the RB gradient for any integer
channel — this extends directly.

**The Phase 2 dither loss** (H_binary(|frac − 0.5|)) is per-dimension and holds for
any z_dim with per-channel δ.

### 3.3 What changes

**Histogram sample complexity with large z_dim:**
At z_dim=64, each dimension's histogram requires many more samples to estimate p(m_k)
well. With 4096 rollout samples split across potentially hundreds of bins, the per-bin
count drops below useful levels. The Laplace smoothing (α=0.5) partially mitigates
this, but high-entropy dimensions will have unreliable histograms.

**Alternative A — Factored Gaussian rate approximation:**
Replace the histogram with a running estimate of mean and std of z_k per dimension.
Treat m_k ~ Rounded-Gaussian. The rate becomes approximately:
```
H(m_k) ≈ log₂(std(z_k)/δ_k · √(2πe))   [Gaussian differential entropy bound]
```
This is a continuous approximation — less tight than the histogram but requires only
2 statistics per dimension (mean, std), scales to any z_dim, and is always well-defined.

**Alternative B — Joint histogram via learned entropy model (DLM, revisited):**
The DLM (PILLAR_P2_v2.md, Option 2) failed for the toy problem due to the circular
gradient and estimation gap. At scale, the key barrier was that H(m) was too high for
a 5-component Gaussian mixture to fit well. With a large speaker network and rich
observations, the message distribution may be smoother (more Gaussian-like), making
the DLM viable. The fix-ladder insights (backward gate, EMA prior, two-phase training)
from E41 should be applied.

**Alternative C — Vector-quantised (VQ) discrete message:**
Replace the dither quantiser with a learned codebook (VQ-VAE style). The rate becomes
log₂(|codebook|) bits — fixed, trivially computable. The dither channel's continuous
relaxation is replaced by straight-through vector quantisation. This loses the
Schuchman exactness guarantee but is common practice for discrete communication.
**This is an alternative to the whole approach, not a pillar extension.**

**Non-stationarity in MARL:**
With N simultaneously learning agents, the message distribution p(m_k) from agent i
shifts as agent i's speaker policy changes. The histogram from rollout t may be stale
at rollout t+1. This is the moving-target problem (C-F19 in the toy problem), but
amplified by the non-stationarity of N policies instead of 1.

**Alternative:** Use a shorter histogram window (e.g., last K steps rather than full
rollout) or an exponential moving average over histogram counts. The toy problem's
histogram resets each rollout; at scale, a shorter EMA window would make the rate
estimate faster-reacting.

**Role-collapse consequence — comms loss bleeds into action quality (§0.5A):**
In the toy problem, the magnitude and dither losses only affect the speaker's message
output, with no path to the listener's action head. At scale, both losses flow through
the shared policy backbone — directly modifying the representations used for action
selection. A comms penalty that is too large will degrade task performance not because
the listener receives worse messages, but because the backbone itself is being pushed
toward message compression at the expense of action-relevant features.

**Alternative — architectural separation:** Add a dedicated message head that branches
off from a shared observation encoder (backbone → [action head | message head]).
Apply comms loss ONLY to the message head and stop its gradient at the backbone (or
use a small λ to allow backbone sharing). This partially restores the toy problem's
separation. The trade-off is that the message cannot exploit the full representational
power of the backbone if the gradient is stopped. A compromise: apply comms loss to
the message head with full gradient, but scale λ_comms to be much smaller than in
the toy problem (e.g., 5× smaller) to prevent backbone domination.

---

## 4. Pillar P3 — Deployment Consistency at Scale

### 4.1 What holds

The NSD sample-consistency property (z_hat = z_hat_true) is purely a channel property.
It holds at any scale with no modification.

The SD distributional consistency (Schuchman Theorem A.1: marginal distribution of ẑ
is the same at train and deploy) also holds at any scale.

### 4.2 What changes

**Train/deploy gap may be more severe at scale:**
In the toy problem with SR=1, the gap between SD train-time ẑ and deploy-time ẑ_deploy
has a small effect because the listener has near-perfect adaptation. At scale with
partial observability and complex listeners, the distributional mismatch between
ẑ_STE (train) and ẑ_true (deploy) could cause larger SR degradation.

Hypothesis: NSD's stronger guarantee (sample-consistent, not just distributionally
consistent) becomes more valuable as listener complexity increases. The toy problem
may underestimate the deploy gap because its listener is a 2-layer MLP over a clean
6-class input.

**No sample-by-sample consistency with TPDF and multiple agents:**
In the toy problem, each (speaker, listener) pair has a private dither draw. At scale
with N agents, agent j's action at step t depends on messages from agents 1..N-1 ≠ j.
During deploy, all these messages are transmitted via their respective channels (no
dither). During training, each is dithered independently. The joint action distribution
at deploy is thus a product of N independently channel-shifted inputs — the NSD
consistency property holds per-channel, not for the joint action.

**Alternative:** No structural change needed — NSD still eliminates each per-channel
train/deploy gap independently. The residual joint mismatch (from correlated multi-agent
dithering) is a separate problem, addressed by CTDE (centralised critics already
account for the joint multi-agent distribution at training time).

---

## 5. Pillar P4 — Rao-Blackwell Gradient at Scale

### 5.1 What holds

The finite-difference formula `g_RB_k = [L(ẑ_hi) − L(ẑ_lo)] / δ_k` is algebraically
derived from the score-function gradient for any differentiable loss. It holds for:
- Any speaker architecture (MLP, GNN, transformer)
- Any listener architecture
- Any differentiable task loss (actor loss, value loss, shaped reward)
- Any number of agents, as long as L is differentiable w.r.t. the message vector z

The RB formula itself is task- and architecture-agnostic. It will give an unbiased
estimate of the score-function gradient for z regardless of how the policy is structured.

### 5.2 What changes — the detach trick breaks down (§0.5B)

**The toy-problem detach trick:**
In the toy problem, when `use_rb_gradient=True`, z_hat is DETACHED from the listener
before computing actor_loss. This ensures the speaker's task gradient flows only via
the RB proxy, not via the standard STE actor_loss path. The listener's parameters
still update from actor_loss because z_hat is fed to it after detach.

```python
# Toy problem:
_z_hat_for_listener = z_hat.detach()   # speaker gets NO gradient from actor_loss
dist = listener(cat([obs, _z_hat_for_listener]))
actor_loss = ppo_loss(dist, ...)
rb_proxy   = (rb_scale * z_speaker).sum(-1).mean()
total_loss = rb_proxy + actor_loss + critic_loss
# actor_loss gives gradient to listener θ_L; rb_proxy gives gradient to speaker θ_S
```

This works because θ_L and θ_S are disjoint. Detaching z_hat cleanly separates the
two gradient paths.

**At scale, this separation collapses:**
Agent i's message encoder and action head share parameters θ. The "task gradient
through the speaker" from agent j's loss is:

```
L_j → z_hat_{i→j} → z_{i→j} → message_head(obs_i; θ) → θ
```

If you detach z_hat_{i→j} from agent j's computation graph before computing L_j,
you sever this gradient path — blocking the credit-assignment signal from receiver j
back to sender i's encoder. This is exactly the gradient you need to estimate with RB;
blocking it and separately computing an RB proxy would be double-counting.

**The correct formulation at scale:**
Instead of detaching z_hat and replacing the gradient entirely with rb_proxy, apply
the RB proxy as a CORRECTION to the existing STE gradient:

```
total_loss_θ = task_loss (via STE, all agents, standard MAPPO)
             + rb_proxy_{i→j}  ← adds the RB correction on top, not a replacement
             + comms_loss
```

The STE gradient from `z_hat_{i→j}` gives `[L_j(ẑ) · ∂log p(m|z) / ∂z]` via the
score function — which is ALREADY an unbiased but high-variance estimate. The RB
proxy replaces only the dither-noise variance component, treating the multi-agent
credit assignment as fixed. In practice: compute rb_proxy without detaching z_hat
from the task loss, so both gradients flow simultaneously. The RB proxy adds a
lower-variance correction; the standard STE gradient handles the inter-agent
credit assignment path.

This is a weaker separation than the toy problem achieves, but it is implementable
and still provides the variance-reduction benefit of P4.

**Forward-pass cost at scale:**
In the toy problem: 2 extra listener forward passes per minibatch (cheap, z_dim=3
MLP listener). At scale with N agents and fully connected topology: to compute
L_j(z_hat_hi_{i→j}) and L_j(z_hat_lo_{i→j}) for all (i,j) pairs requires
2·N(N-1) extra forward passes per minibatch. For N=10: 180 extra passes. This is
prohibitive.

**Alternative A — Sender-side RB only (recommended at scale):**
For each sending agent i, compute the RB proxy using only i's OWN task loss L_i
(not the receiver's loss L_j). This approximates the credit signal with only 2 extra
passes per agent (2N total), at the cost of missing the cross-agent information gain.
The approximation is tightest when agents have similar task structures (homogeneous
MARL like SMAC).

**Alternative B — Sampled receiver:**
For each sender i, randomly sample ONE receiver j and compute the RB proxy using L_j
only. Cost: 2N extra passes. Unbiased in expectation (averaged over random sampling).
Doubles the per-update cost vs. standard, regardless of N.

**Alternative C — Message-value surrogate:**
Train a lightweight surrogate model V_msg(z_{i→j}) ≈ ∂L_j/∂z_{i→j} offline from
replay buffer data. Use V_msg as a proxy for the RB scale without extra forward
passes. Similar in spirit to a value function for communication.

**Credit assignment in MARL:**
The MAPPO central critic computes a joint value V(s), so ∂V/∂z_{i→j} in principle
captures the full multi-agent effect. The RB formula applied to V rather than
actor_loss would be:

```
rb_scale = (V(z_hat_hi_{i→j}) − V(z_hat_lo_{i→j})) / δ_k
```

Since V is a small centralised MLP (not the full policy), the two extra passes are
cheap. This is the most tractable form of P4 at scale — use the critic, not the
actor, for the RB computation.

**Continuous action spaces:**
For continuous actions (Gaussian policy), L_actor = −log N(a; μ(ẑ), σ²). The RB
formula applies without change — the proxy loss `(L_hi − L_lo) · z / δ` gives the
correct score-function gradient through z regardless of action space.

**Multi-step communication:**
At scale, z is produced at every timestep t. The RB gradient applies per-step with
GAE/PPO's advantage handling temporal credit assignment. No structural change needed
for multi-step; the role-collapse considerations above are the dominant concern.

---

## 6. Phase Training Structure at Scale

### 6.1 Toy-problem assumption

Two phases:
- Phase 1: RL + magnitude anchor until SR ≥ 0.99
- Phase 2: entropy compression (dither loss) with RL maintained

The SR threshold is a clean trigger because SR ∈ {0, 1} effectively (the task either
succeeds or fails per episode).

### 6.2 What changes

**SR is not a sharp threshold at scale:**
In complex tasks, SR is a continuous quantity that may plateau at different levels
depending on credit assignment and observation quality. A fixed threshold like 0.99
may never be reached in some environments even with a well-trained policy.

**Alternative A — Entropy-triggered Phase 2:**
Instead of SR ≥ threshold, trigger Phase 2 when the magnitude loss plateau is detected:
`|∂L_mag/∂t| < ε` for K consecutive updates. This is policy-agnostic and works even
if the best achievable SR is 0.75.

**Alternative B — Curriculum Phase 2:**
Gradually ramp λ_dither from 0 to λ_max over a warmup window (e.g., 100K steps after
Phase 2 onset), rather than switching it on at full strength. Reduces the risk of
Phase 2 SR collapse in environments where the listener is more sensitive to message
distribution shifts.

**Alternative C — No phase switching:**
Run magnitude + dither losses simultaneously from the start. The Phase 2 design was
motivated by the circular gradient problem in the toy problem's DLM. With the histogram
estimator (Option 1), the circular gradient is already resolved. The two-phase design
may be unnecessary at scale if the listener naturally adapts to changing message
distributions under joint training.

**Phase 2 listener-gradient zeroing is impossible at scale (§0.5C):**
In the toy problem, Phase 2 zeros the listener's parameter gradients to prevent the
listener from drifting while the speaker compresses its messages. This works because
the listener and speaker are separate networks. At scale, the "listener" — the part
of the policy that processes received messages and outputs actions — IS the shared
policy network. Zeroing its gradients would freeze the entire agent's RL learning,
which is unacceptable.

**Alternative for Phase 2 at scale:**
Rather than freezing listener parameters, reduce the task loss weight during Phase 2
compression:

```
Phase 2 total_loss = λ_rl · task_loss + λ_dither · dither_loss + λ_rate · rate_loss
```

With `λ_rl < 1` during Phase 2 (e.g., 0.1–0.3), the policy is still learning but
the compression objective dominates. This prevents both listener freezing and
unconstrained policy drift. The specific λ_rl schedule should be validated empirically.

Alternatively, apply dither loss ONLY to the dedicated message head (if architectural
separation is used per §3's Alternative), and maintain normal RL gradients on the
action head. This is the cleanest solution but requires the architectural separation
to already be in place.

**Freeze design for δ_k (Design A in P1):**
The toy problem freezes δ_k at Phase 2 onset (Design A). At scale, the optimal δ_k
may continue to evolve in Phase 2 as the message distribution changes under dither
pressure. Design B (continue adapting δ_k in Phase 2) may be necessary if the Phase 1
δ_k values are a poor initialisation for Phase 2 compression.

---

## 7. Rate Decomposition — Loss of the H(G) Oracle

### 7.1 Toy-problem measurement

In the toy problem, H(G) = 1.813 bits is analytically known (uniform distribution over
6 discrete goals). The Shannon gap `H_joint − H(G)` is the primary metric: how far
above the theoretical minimum are we?

### 7.2 At scale

There is no known H(G) for complex tasks (SMAC, MPE, GRF). The theoretical minimum
communication rate depends on:
- The task (what information the listener actually needs)
- The observation structure (how much speaker obs is already known to listener)
- The episode dynamics (how information becomes relevant over time)

**Alternative metrics for scale:**
1. **Communication efficiency ratio:** H_joint / H_factored — measures how much
   TC (total correlation, wasted redundancy between message dims) is present. Should
   approach 1 at optimum. Does not require knowledge of H(G).

2. **Rate-distortion curve:** Pareto frontier of (SR, H_joint) at varying λ_comms.
   The shape of the curve — specifically how steeply SR drops when H_joint is forced
   below a threshold — reveals the task's intrinsic communication requirement.

3. **Ablation rate:** Compare H_joint with and without communication (zero-shot
   listener performance). The difference bounds the minimum rate needed for the
   current SR level.

4. **Mutual information I(obs_speaker; message):** Measures how much of the speaker's
   observation is actually transmitted. Ideally, I(obs; m) equals the task-relevant
   information in obs — computable via MINE or CLUB estimators at scale.

---

## 8. Multi-Agent Topology Extensions

### 8.1 Broadcast vs. selective communication

**Toy problem:** speaker sends to one listener (1→1).
**Scale:** N agents × N-1 receivers = fully connected. Total bits = N(N-1)·H_joint.

For N=10, z_dim=16: 10×9×16 = 1440 floats per step before quantisation. At H_joint=4
bits per dim (rough estimate), this is 57.6 kbits per step. This may be prohibitive
in real deployments (e.g., bandwidth-constrained swarm robotics).

**Alternative — Attention-gated selective communication:**
Each agent learns a gate g_{i→j} ∈ {0,1} deciding whether to send to agent j.
The channel only operates on selected (i,j) pairs. The rate loss is augmented with
a sparsity penalty: λ_gate · Σ_{i,j} g_{i→j}.

The dither channel and all four pillars apply identically to each selected (i,j) message.
The gate mechanism is a separate architectural decision orthogonal to the channel design.

### 8.2 Shared vs. per-agent message encoders

**Toy problem:** one shared speaker network.
**Scale:** N homogeneous agents share one speaker encoder (weight-tied MAPPO).

This is the standard assumption in SMAC and MPE. All four pillars apply directly: one
shared PerChannelDelta, one shared histogram, one shared phase-switch trigger.

If agents are heterogeneous (different obs dimensions, different roles), per-agent
encoders are needed. Each agent has its own δ vector and histogram. Phase switching
can be per-agent (trigger when agent i's SR ≥ threshold) or global (trigger when
mean SR ≥ threshold).

### 8.3 Recurrent speakers (LSTM/GRU)

At scale, speakers may be recurrent (past observation history integrated into hidden
state). The message z is a function of h_t (hidden state) rather than a single obs.

The channel, pillars, and rate loss all apply to z regardless of how z was computed
from obs. No architectural change to the channel design is required. However:
- The histogram must be indexed by the current z/m values, not by obs
- The RB gradient `[L_hi − L_lo] / δ` applies to the loss at step t, which may
  depend on future steps via the recurrent rollout. PPO's clipped surrogate handles
  this via importance sampling — the RB proxy is applied per-step identically.

---

## 9. Assumptions Summary Table

| Assumption | Toy problem | Holds at scale? | Alternative if not |
|---|---|---|---|
| Schuchman linearisation (Thm 2) | Always | **Yes, always** | N/A |
| NSD sample consistency (Thm 5) | Always | **Yes, always** | N/A |
| STE gradient exactness | Always | **Yes, always** | N/A |
| RB formula validity | Any loss | **Yes, always** | N/A |
| **Speaker ≠ listener (separate params)** | **By design** | **No** | Dedicated message head; additive RB proxy; critic-based RB |
| Comms loss isolated from action head | Separate networks | **No** | Architectural separation; small λ_comms |
| RB detach trick applicable | Separate networks | **No** | Additive proxy (not replacement); critic-based RB |
| Phase 2 can zero listener gradients | Separate networks | **No** | λ_rl < 1 in Phase 2; message-head-only dither loss |
| Fixed δ = 1.0 | z ∈ ℝ³, small | **No** | Per-channel δ (P1), empirical std init |
| z_dim = 3 sufficient | 6 discrete goals | **No** | Sweep z_dim; use TC as signal |
| H(G) oracle known | 6 goals | **No** | Rate-distortion curve, I(obs;m) |
| Histogram sufficient for p(m) | Small z_dim, large B | **Probably not** | Gaussian approx, EMA window |
| SR ≥ 0.99 triggers Phase 2 | Task succeeds sharply | **Maybe not** | Entropy plateau trigger, λ ramp |
| 1:1 communication (N=2) | By design | **No** | Per-pair channels, gated comm |
| Discrete 5-action listener | By design | **No** | Gaussian policy — RB still works |
| Short episode (T=1) | By design | **No** | Per-step RB, GAE handles credit |
| Homogeneous agents | By design | **Usually** | Per-agent δ, per-agent histogram |
| No listener non-stationarity | N=2, joint training | **Soft** | EMA histogram, shorter window |

---

## 10. Recommended Implementation Order at Scale

These are listed in dependency order, not confidence order. The role-collapse problem
(§0.5) must be resolved architecturally FIRST — it determines whether the remaining
pillar implementations are straightforward ports or require fundamental redesign.

1. **Architectural decision: separate message head or shared backbone?**
   Choose before implementing any pillar. Options:
   - **Fully separate** (backbone → [action head | message head]): closest to toy
     problem; comms loss targets message head only; RB detach trick works as-is.
   - **Shared backbone, separate heads** (recommended): backbone gradient receives
     both task and comms signals; apply small λ_comms to prevent backbone domination.
   - **Single network** (fully coupled): simplest, but comms loss most aggressively
     interferes with actions; requires smallest λ_comms and most careful tuning.

2. **SD / NSD channel with per-channel δ (P1+P3):** Drop-in replacement for the
   message head output. Replace `z_msg` → `z_hat = channel(z_msg, δ_k)`. Cost: one
   PerChannelDelta module (shared across agents). Zero risk to the action head.

3. **Histogram rate loss + Phase 2 dither (P2):** Add after baseline SR plateaus.
   Phase 2 trigger: L_mag plateau detector (not SR ≥ 0.99, which may never be
   reached at scale). In Phase 2: reduce λ_rl (e.g., 0.1) instead of zeroing listener
   gradients. Start with EMA histogram over last 5 rollouts (non-stationarity risk).

4. **RB gradient for task loss (P4):** Use critic-based RB (2 extra centralised critic
   forward passes, cheap) rather than actor-based RB (2N extra policy forward passes,
   expensive). Apply additively (not as a replacement) if shared backbone is used.
   `rb_scale = (V(z_hat_hi) − V(z_hat_lo)) / δ`; proxy added to total_loss alongside
   standard PPO loss.

5. **Selective communication gate:** Only if N > 6 and bandwidth is a constraint.
   Independent of all four pillars — add after pillars are validated.

6. **VQ codebook:** Only if the soft-quantisation approach (all four pillars) does
   not achieve the required compression ratio in the deployment environment.

---

## 11. Open Questions (to resolve during large-scale experiments)

**Q-LS-1:** Does per-channel δ (P1) differentiate across z_dim=64 dimensions with
a deep speaker network, or does the gradient signal become too noisy?

**Q-LS-2:** Is the two-phase trigger (SR ≥ 0.99) appropriate for SMAC, where the
best-known SR on hard maps is ~0.85? Should Phase 2 be triggered by L_mag plateau
instead?

**Q-LS-3:** Does the NSD deployment gap (SR_train vs SR_deploy) grow larger with
complex listeners, motivating stronger P3 guarantees at scale?

**Q-LS-4:** Is the histogram rate loss (source_coding_rate_loss) stable with N > 2
agents learning simultaneously? Does the moving-target problem (C-F19) worsen with
joint non-stationarity?

**Q-LS-5:** Does the RB gradient acceleration (P4, measured by convergence speed)
generalise from the 6-goal toy problem to environments with reward sparsity and
delayed credit assignment?

**Q-LS-6:** What is the right z_dim for communication in SMAC 3m (obs_dim=8),
SMAC 6h_vs_8z (obs_dim=80), and MPE Simple Spread (obs_dim=18)? Is H_joint /
H_factored (TC efficiency) a reliable guide for z_dim selection without an H(G) oracle?

**Q-LS-7 (role collapse — architecture):** Does comms loss on a shared backbone
meaningfully degrade action quality at practical λ_comms values (e.g., 5e-4)? Or is
the backbone large enough that comms gradients do not measurably shift the action head?
This determines whether architectural separation is necessary or merely optional.

**Q-LS-8 (role collapse — P4):** Is critic-based RB (`(V(z_hi) − V(z_lo)) / δ`)
a sufficient variance-reduction signal for the message encoder, or does the critic
value function smooth over too much task-relevant structure? How does its variance
compare to actor-based RB when measured on a toy environment where both are tractable?

**Q-LS-9 (role collapse — Phase 2):** What λ_rl reduction in Phase 2 preserves
task performance while allowing compression to proceed? Is there a principled way
to schedule λ_rl (e.g., tied to the rate of H_joint reduction per update) rather
than setting it as a fixed hyperparameter?

**Q-LS-10 (role collapse — gradient entanglement):** In weight-tied MARL, does the
summed task gradient over N(N-1) message channels amplify or dilute the δ_k
differentiation signal compared to the toy problem's single gradient path? Does
P1 differentiate more cleanly at scale precisely because there is more signal, or
does the entanglement with other agents' non-stationary policies introduce enough
noise to wash out the per-channel structure?
