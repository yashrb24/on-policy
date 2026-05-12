from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from onpolicy.envs.toyproblem.buffer import RolloutBuffer
from onpolicy.envs.toyproblem.channels import build_channel, H_GOAL_BITS, _GOAL_PROBS
from onpolicy.envs.toyproblem.CommunicatingGoal_env import _DEFAULT_GOALS as _ENV_DEFAULT_GOALS
from onpolicy.envs.toyproblem.source_coding import (
    MessageHistogram,
    dither_channel_loss,
    dither_channel_stats,
    histogram_rate_stats,
    source_coding_rate_loss,
)
from onpolicy.envs.toyproblem.network import (
    Critic, ListenerActor, SpeakerNetwork, PerChannelDelta,
    EntropyModelFactored, EntropyModelJoint,
    EntropyModelCondZ, EntropyModelJointCondZ,
    joint_entropy_bits, marginal_entropies_bits, total_correlation_bits,
)
from onpolicy.utils.valuenorm import ValueNorm


@dataclass
class MAPPOConfig:
    z_dim: int = 3
    hidden_size: int = 64
    lr: float = 3e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.03
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    num_minibatches: int = 4
    adam_eps: float = 1e-5
    channel: str = "none"
    delta: float = 1.0
    lambda_comms: float = 0.0
    ste_clip: float = 10.0

    # P2 — Entropy model
    use_entropy_model: bool = False
    entropy_model_K: int = 5
    entropy_model_type: str = "factored"   # "factored" | "joint"
    entropy_model_context: str = "A"       # "A" (marginal) | "B" (conditioned on z)
    lr_qphi_mult: float = 10.0             # q_φ lr = lr_qphi_mult × lr
    n_qphi_steps: int = 3                  # q_φ gradient steps per RL minibatch
    n_warmup_steps: int = 5000             # q_φ warm-start steps before RL
    loss_comms_mode: str = "magnitude"     # "magnitude" | "entropy" | "both"

    # P2 — Fix-ladder options (see docs/pillars/PILLAR_P2.md §13)
    # Level 1: backward gate — only allow Ballé backward once q_φ is well-fitted.
    # Default inf disables the gate (backward always active, current behaviour).
    qphi_bwd_gate_threshold: float = float("inf")

    # Level 3: EMA prior — use an exponential-moving-average shadow of q_φ for the
    # backward loss.  Breaks the circular gradient problem: the EMA model lags behind
    # p(m) so the gradient always points toward genuinely cheaper messages.
    use_ema_prior: bool = False
    ema_prior_momentum: float = 0.95       # higher = slower EMA (more lag, more stable)

    # Level 5: Two-phase training — jointly train RL + magnitude in Phase 1 until the
    # policy converges, then switch to entropy-only loss in Phase 2 to compress the
    # already-learned code without corrupting task performance.
    # 0.0 = disabled (no phase switching).
    phase1_sr_threshold: float = 0.0

    # P2 Option 1 — Source coding via Online Histogram + Score Function (§15)
    # Replaces the DLM prior with an empirical frequency table, eliminating the
    # qphi_gap bottleneck that caused the fix-ladder sweep to fail (gap 12-14 bits).
    #
    # use_source_coding:      enable the histogram rate loss (replaces DLM backward)
    # source_coding_smoothing: Laplace α for histogram (Jeffreys prior: 0.5)
    #
    # When use_source_coding=True:
    #   - loss_comms_mode="magnitude" → adds magnitude + score-function losses
    #   - loss_comms_mode="entropy"   → score-function loss only (no magnitude)
    #   - loss_comms_mode="both"      → magnitude + score-function losses
    #   The existing use_entropy_model / DLM path is independent and can coexist
    #   for measurement (forward pass only, no backward to speaker).
    use_source_coding: bool = False
    source_coding_smoothing: float = 0.5
    # Phase 2 dither entropy loss — reduces H(m|goal) by pushing frac(z_k/δ) toward 0.5.
    # For the floor SD channel, minimum dither noise is at frac = 0.5 (bin centres),
    # not at frac = 0 (bin boundaries).  See source_coding.dither_channel_loss for
    # the full derivation.  0.0 = disabled.  Only applied when training_phase == 2.
    lambda_dither: float = 0.0

    # F14/F19 diagnostic logging (off by default — adds compute overhead)
    log_grad_decomp: bool = False    # F14: PPO vs SC speaker grad norms (per minibatch)
    log_moving_target: bool = False  # F19: ‖Δθ‖ + offline H(m) (per update)
    moving_target_n: int = 10_000   # F19: offline sample size for H(m) estimate

    # P1 — Per-channel learnable δ_k (Pillar 1)
    # learn_delta: replace global scalar δ with z_dim independent widths δ_k = softplus(α_k).
    # learn_global_delta: learn a single global δ (1 param, broadcasts). E53 baseline for E54.
    # lr_delta: dedicated Adam LR for α_k (10× smaller than speaker LR to reduce oscillation).
    # Only active for SD / NSD channels. Design A: δ_k frozen at Phase 2 onset.
    learn_delta: bool = False
    learn_global_delta: bool = False
    lr_delta: float = 1e-4
    # heuristic_delta: at Phase 2 onset, set δ_k ∝ 1/H(m_k) from empirical histogram,
    # then freeze. Baseline comparison for learned per-channel δ (E56). No optimizer.
    heuristic_delta: bool = False

    # P4 — Rao-Blackwell gradient estimator for task loss (Phase 1 only)
    # use_rb_gradient: replace STE speaker task gradient with RB finite-difference estimator.
    # rb_mode: "joint" (2 passes, all dims move together) | "per_dim" (2*z_dim passes, exact)
    use_rb_gradient: bool = False
    rb_mode: str = "joint"

    # P3 — Deployment evaluation: route listener input through z_hat_true/z_hat_deploy
    # instead of the STE z_hat used during training. Applies only to SD and NSD channels.
    deploy_eval: bool = False


class MAPPOTrainer(nn.Module):
    """Speaker + listener + centralized critic with a single Adam over all params.

    Rollout: `act_and_value` (no_grad) produces (action, logp, value_norm).
    Update:  `update(buffer)` runs epochs × minibatches of PPO + MSE.
    Bootstrap: `get_value` for the GAE tail value.
    """

    def __init__(self, config: MAPPOConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.speaker = SpeakerNetwork(
            obs_dim=2, z_dim=config.z_dim, hidden=config.hidden_size
        ).to(device)
        self.listener = ListenerActor(
            obs_dim=2 + config.z_dim, action_dim=5, hidden=config.hidden_size
        ).to(device)
        self.critic = Critic(state_dim=4, hidden=config.hidden_size).to(device)
        self.channel = build_channel(
            config.channel, config.delta, ste_clip=config.ste_clip
        ).to(device)
        self.value_norm = ValueNorm(input_shape=1, device=device)

        # P2 — Entropy model and separate q_φ optimizer
        self.entropy_model = None
        self.optim_qphi = None
        # Companion context-B model: always created alongside context-A primary
        # for oracle bound measurement (forward-only, no backward to speaker).
        # Context B always uses K=5 (expressiveness beyond 5 components adds no
        # value for measurement; see PILLAR_P2.md §10 Stage P2-A).
        self.entropy_model_B = None
        self.optim_qphi_B = None
        # EMA shadow of entropy model (Level 3 fix — only created when use_ema_prior=True)
        self._ema_entropy_model = None
        # Warm-start final loss (NaN until warm-start runs)
        self._warmup_bits_final: float = float("nan")
        # Training phase: 1 = RL + comms (joint), 2 = entropy compression only
        self._training_phase: int = 1

        # P2 Option 1 — Online histogram for source-coding rate loss.
        # Reset each update() call; populated from the full rollout before PPO epochs.
        self.histogram: MessageHistogram | None = None
        if config.use_source_coding:
            self.histogram = MessageHistogram(
                z_dim=config.z_dim,
                smoothing=config.source_coding_smoothing,
            )

        if config.use_entropy_model:
            ctx = config.entropy_model_context
            typ = config.entropy_model_type
            K = config.entropy_model_K
            if ctx == "A" and typ == "factored":
                self.entropy_model = EntropyModelFactored(config.z_dim, K).to(device)
            elif ctx == "A" and typ == "joint":
                self.entropy_model = EntropyModelJoint(config.z_dim, K).to(device)
            elif ctx == "B" and typ == "factored":
                self.entropy_model = EntropyModelCondZ(config.z_dim, K).to(device)
            elif ctx == "B" and typ == "joint":
                self.entropy_model = EntropyModelJointCondZ(config.z_dim, K).to(device)
            else:
                raise ValueError(
                    f"Unsupported entropy_model_context={ctx!r}, type={typ!r}. "
                    f"Supported: (A, factored), (A, joint), (B, factored), (B, joint)."
                )
            self.optim_qphi = torch.optim.Adam(
                self.entropy_model.parameters(),
                lr=config.lr * config.lr_qphi_mult,
                eps=config.adam_eps,
            )
            # Companion context-B: factored uses CondZ, joint uses JointCondZ
            if ctx == "A":
                _B_cls = (EntropyModelCondZ if typ == "factored"
                          else EntropyModelJointCondZ)
                self.entropy_model_B = _B_cls(config.z_dim, K=5).to(device)
                self.optim_qphi_B = torch.optim.Adam(
                    self.entropy_model_B.parameters(),
                    lr=config.lr * config.lr_qphi_mult,
                    eps=config.adam_eps,
                )
                # Level 3: EMA shadow for backward loss (context A only)
                if config.use_ema_prior and ctx == "A":
                    self._ema_entropy_model = copy.deepcopy(self.entropy_model).to(device)
                    for p in self._ema_entropy_model.parameters():
                        p.requires_grad_(False)

        self._trainable = (
            list(self.speaker.parameters())
            + list(self.listener.parameters())
            + list(self.critic.parameters())
            + list(self.channel.parameters())
        )
        self.optim = torch.optim.Adam(
            self._trainable, lr=config.lr, eps=config.adam_eps
        )

        # P1 — Per-channel δ_k (Pillar 1). Only for dithering channels.
        self.per_channel_delta: PerChannelDelta | None = None
        self.optim_delta: torch.optim.Adam | None = None
        if (config.learn_delta or config.learn_global_delta) and config.channel in ("sd", "nsd"):
            n_d = 1 if config.learn_global_delta else config.z_dim
            self.per_channel_delta = PerChannelDelta(
                n_d, delta_init=config.delta
            ).to(device)
            self.optim_delta = torch.optim.Adam(
                self.per_channel_delta.parameters(),
                lr=config.lr_delta,
                eps=config.adam_eps,
            )
        elif config.heuristic_delta and config.channel in ("sd", "nsd"):
            # Heuristic δ: starts at global delta_init, replaced at Phase 2 onset.
            # No optimizer — purely rule-based assignment, then frozen.
            self.per_channel_delta = PerChannelDelta(
                config.z_dim, delta_init=config.delta
            ).to(device)
            self.per_channel_delta.log_alpha.requires_grad_(False)

    @torch.no_grad()
    def act_and_value(
        self, goal: torch.Tensor, listener_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.speaker(goal)
        _act_delta = self.per_channel_delta.delta() if self.per_channel_delta is not None else None
        z_hat, ch_info = (
            self.channel(z) if _act_delta is None else self.channel(z, _act_delta)
        )
        # P3: deployment eval routes listener through actual quantised output, not STE z_hat.
        if self.config.deploy_eval and self.config.channel == "sd":
            listener_input = ch_info["z_hat_deploy"]
        elif self.config.deploy_eval and self.config.channel == "nsd":
            listener_input = ch_info["z_hat_true"]
        else:
            listener_input = z_hat
        dist = self.listener(torch.cat([listener_pos, listener_input], dim=-1))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        state = torch.cat([listener_pos, goal], dim=-1)
        value = self.critic(state)  # (N, 1) normalized
        return action, log_prob, value

    @torch.no_grad()
    def get_value(
        self, goal: torch.Tensor, listener_pos: torch.Tensor
    ) -> torch.Tensor:
        state = torch.cat([listener_pos, goal], dim=-1)
        return self.critic(state)

    def notify_success_rate(self, sr: float) -> bool:
        """Inform the trainer of the current rolling success rate.

        Call from train.py after each update. Returns True exactly once, when
        the training phase transitions from 1 → 2 (i.e. the first time SR
        crosses `phase1_sr_threshold`).  After that, Phase 2 stays active for
        the rest of training.

        Phase 2 semantics (Level 5 fix — two-phase training):
          - Phase 1: standard RL + magnitude/entropy comms loss.
          - Phase 2: entropy compression only.  RL losses are computed but not
            added to total_loss; only the Ballé backward entropy term drives the
            speaker.  Listener and critic gradients are zeroed before the
            optimizer step so they do not drift away from their Phase 1 solution.
        """
        if (self._training_phase == 1
                and self.config.phase1_sr_threshold > 0.0
                and sr >= self.config.phase1_sr_threshold):
            self._training_phase = 2
            # Heuristic δ: set δ_k ∝ 1/H(m_k) before freezing.
            if self.config.heuristic_delta and self.per_channel_delta is not None and self.histogram is not None:
                H = self.histogram.empirical_entropy()  # list[float], per dim, in bits
                H_max = max(H) if max(H) > 0.0 else 1.0
                eps = 1e-3
                import math as _math
                with torch.no_grad():
                    for k in range(self.config.z_dim):
                        # Coarser bin (larger δ) for low-entropy dims; finer for high-entropy.
                        delta_k = float(
                            self.config.delta * H_max / max(H[k], eps)
                        )
                        delta_k = max(self.per_channel_delta.delta_min,
                                      min(delta_k, self.per_channel_delta.delta_max))
                        # Write back via softplus_inv so delta() returns delta_k exactly.
                        self.per_channel_delta.log_alpha[k] = _math.log(
                            _math.exp(delta_k) - 1.0
                        )
            # Design A: freeze δ_k at Phase 2 onset (learn_delta and heuristic_delta).
            if self.per_channel_delta is not None:
                self.per_channel_delta.log_alpha.requires_grad_(False)
            return True
        return False

    def _update_ema_prior(self) -> None:
        """Update EMA shadow of entropy model (Level 3 — call after each q_φ step).

        The EMA model lags behind the live q_φ by one-minus-momentum generations,
        breaking the circular gradient: the backward loss uses a prior that does
        not perfectly track the current p(m), so ∇_z[-log q_ema(z/δ)] always
        points toward genuinely cheaper messages rather than toward the current
        distribution's mode.
        """
        if self._ema_entropy_model is None:
            return
        mom = self.config.ema_prior_momentum
        with torch.no_grad():
            for p_live, p_ema in zip(
                self.entropy_model.parameters(),
                self._ema_entropy_model.parameters(),
            ):
                p_ema.data.mul_(mom).add_(p_live.data, alpha=1.0 - mom)

    def warmup_entropy_model(self, buffer: RolloutBuffer, n_steps: int) -> float:
        """Pre-train q_φ for n_steps gradient steps before RL begins.

        Uses the current rollout buffer to sample minibatches. The speaker is
        run in no-grad mode to collect discrete messages m. Only q_φ (optim_qphi)
        is updated — RL parameters are untouched.

        Returns the final warm-start loss value (0.0 if entropy model is off).
        """
        if self.entropy_model is None:
            return 0.0
        step = 0
        final_loss = float("nan")
        while step < n_steps:
            for mb in buffer.minibatches(self.config.num_minibatches):
                if step >= n_steps:
                    break
                with torch.no_grad():
                    z = self.speaker(mb["goals"])
                    _wm_d = self.per_channel_delta.delta() if self.per_channel_delta is not None else None
                    _, ch_info = self.channel(z) if _wm_d is None else self.channel(z, _wm_d)
                m = ch_info.get("m")
                if m is None:
                    return 0.0  # IdentityChannel — no discrete messages
                m_float = m.float()
                if self.config.entropy_model_context == "A":
                    nll = self.entropy_model.nll_bits(m_float)
                else:
                    nll = self.entropy_model.nll_bits(m_float, z.detach())
                loss = nll.mean()
                self.optim_qphi.zero_grad(set_to_none=True)
                loss.backward()
                self.optim_qphi.step()
                # Also warm-start companion context-B model
                if self.entropy_model_B is not None:
                    nll_B = self.entropy_model_B.nll_bits(m_float.detach(), z.detach())
                    loss_B = nll_B.mean()
                    self.optim_qphi_B.zero_grad(set_to_none=True)
                    loss_B.backward()
                    self.optim_qphi_B.step()
                final_loss = loss.item()
                step += 1
        self._warmup_bits_final = final_loss
        return final_loss

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        # Update ValueNorm running statistics once with all returns in this rollout.
        # Calling update() inside the minibatch loop would shift the normalization
        # statistics 40× (10 epochs × 4 mbs), causing inconsistent returns_norm
        # across minibatches and epochs within the same update.
        all_returns = buffer.returns.reshape(-1, 1)  # (T*N, 1)
        self.value_norm.update(all_returns)

        metrics: dict[str, list[float]] = defaultdict(list)

        # ── Histogram E-step: populate counts from the full rollout ───────────
        # m is not stored in the buffer; we recompute it via a no_grad forward pass
        # over all minibatches before the PPO epoch loop.  This is the EM E-step:
        # q_hist ← empirical_freq(m | current_policy), then freeze for M-step below.
        _delta_estep = (
            self.per_channel_delta.delta().detach()
            if self.per_channel_delta is not None else None
        )
        if self.histogram is not None:
            self.histogram.reset()
            with torch.no_grad():
                for mb_sc in buffer.minibatches(self.config.num_minibatches):
                    z_sc = self.speaker(mb_sc["goals"])
                    _, ch_info_sc = (
                        self.channel(z_sc) if _delta_estep is None
                        else self.channel(z_sc, _delta_estep)
                    )
                    m_sc = ch_info_sc.get("m")
                    if m_sc is not None:
                        self.histogram.update(m_sc)

        # ── F19: snapshot speaker params before epoch loop ────────────────────
        _speaker_params_before: list[torch.Tensor] | None = None
        if self.config.log_moving_target:
            _speaker_params_before = [
                p.detach().clone() for p in self.speaker.parameters()
            ]

        for _ in range(self.config.update_epochs):
            for mb in buffer.minibatches(self.config.num_minibatches):
                # P1: get current per-channel δ (gradient-connected for Phase 1 updates).
                _delta = (
                    self.per_channel_delta.delta()
                    if self.per_channel_delta is not None else None
                )
                z_new = self.speaker(mb["goals"])
                z_hat, ch_info = (
                    self.channel(z_new) if _delta is None
                    else self.channel(z_new, _delta)
                )
                m = ch_info.get("m")

                # ── Step 1: q_φ forward update BEFORE the RL step ─────────────────
                # Rationale: updating q_φ first ensures the backward entropy loss
                # (step 3) uses a prior that has already tracked the current batch's
                # message distribution, giving a fresher rate signal. Using z_new
                # from the current minibatch avoids the stale-z problem that arises
                # when q_φ is updated after the speaker has already been changed.
                _qphi_gap_now: float | None = None   # used by backward gate in Step 3
                if self.entropy_model is not None and m is not None:
                    m_float = m.float().detach()
                    for _ in range(self.config.n_qphi_steps):
                        if self.config.entropy_model_context == "A":
                            nll_fwd = self.entropy_model.nll_bits(m_float)
                        else:  # context B
                            nll_fwd = self.entropy_model.nll_bits(m_float, z_new.detach())
                        loss_q = nll_fwd.mean()
                        self.optim_qphi.zero_grad(set_to_none=True)
                        loss_q.backward()
                        self.optim_qphi.step()
                    # Level 3: update EMA shadow after q_φ gradient steps
                    self._update_ema_prior()
                    # Level 1: compute qphi_gap for backward gate check in Step 3
                    if self.config.qphi_bwd_gate_threshold < float("inf"):
                        with torch.no_grad():
                            _nll_check = self.entropy_model.nll_bits(m_float)
                            _h_check = joint_entropy_bits(m.long())
                            _qphi_gap_now = _nll_check.sum(dim=-1).mean().item() - _h_check
                    # Companion context-B forward update (measurement only)
                    if self.entropy_model_B is not None:
                        nll_B_fwd = self.entropy_model_B.nll_bits(
                            m_float, z_new.detach()
                        )
                        self.optim_qphi_B.zero_grad(set_to_none=True)
                        nll_B_fwd.mean().backward()
                        self.optim_qphi_B.step()

                # ── Step 2: RL losses ──────────────────────────────────────────────
                # P4: when RB gradient is active, detach z_hat from the speaker so
                # actor_loss gradient does NOT flow to speaker params; the RB proxy
                # loss handles the speaker update instead.
                _z_hat_for_listener = (
                    z_hat.detach() if self.config.use_rb_gradient else z_hat
                )
                dist = self.listener(torch.cat([mb["listener_pos"], _z_hat_for_listener], dim=-1))
                new_logp = dist.log_prob(mb["actions"])
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - mb["old_log_probs"])
                # Normalize per-minibatch so advantages have zero mean and unit
                # variance within each update step, regardless of epoch number.
                adv_mb = mb["advantages"]
                adv = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                pg_loss = -torch.min(surr1, surr2).mean()
                entropy_mean = entropy.mean()
                actor_loss = pg_loss - self.config.entropy_coef * entropy_mean

                returns_mb = mb["returns"].unsqueeze(-1)
                returns_norm = self.value_norm.normalize(returns_mb)

                state_mb = torch.cat([mb["listener_pos"], mb["goals"]], dim=-1)
                new_value = self.critic(state_mb)
                critic_loss = 0.5 * (new_value - returns_norm).pow(2).mean()

                comms_per_elem = (
                    self.channel.comms_loss(z_new) if _delta is None
                    else self.channel.comms_loss(z_new, _delta)
                )
                comms_mean = comms_per_elem.mean()

                # Phase 1: RL + magnitude comms.  Phase 2: entropy only (RL terms
                # are computed above for logging but not added to total_loss).
                if self._training_phase == 1:
                    total_loss = actor_loss + critic_loss
                    if self.config.lambda_comms > 0.0:
                        if self.config.loss_comms_mode in ("magnitude", "both"):
                            total_loss = total_loss + self.config.lambda_comms * comms_mean
                else:
                    # Phase 2 — include RL losses so the listener keeps adapting as
                    # the speaker shifts its z distribution toward bin centres.
                    # Without RL, the frozen listener cannot decode the new messages,
                    # causing SR to degrade.  At SR=1 the RL pressure on the speaker
                    # is near-zero, so entropy loss still dominates speaker updates.
                    total_loss = actor_loss + critic_loss

                # ── Step 2a: P4 Rao-Blackwell proxy loss (Phase 1 only, joint mode) ─
                # Replaces the STE speaker task gradient with the finite-difference RB
                # estimator: g_RB_k = (L(ẑ_hi) - L(ẑ_lo)) / δ_k.
                # z_hat was detached above so actor_loss does NOT flow to speaker;
                # rb_proxy_loss is the speaker's sole task gradient path.
                _rb_task_loss: float = 0.0
                if self.config.use_rb_gradient and self._training_phase == 1:
                    _eff_d = _delta if _delta is not None else self.config.delta
                    with torch.no_grad():
                        frac = (z_new / _eff_d) - torch.floor(z_new / _eff_d)
                        lo_frac = frac < 0.5
                        m_floor = torch.floor(z_new / _eff_d)
                        m_lo = torch.where(lo_frac, m_floor - 1, m_floor)
                        m_hi = torch.where(lo_frac, m_floor, m_floor + 1)
                        z_hat_lo = (m_lo + 0.5) * _eff_d   # (B, z_dim)
                        z_hat_hi = (m_hi + 0.5) * _eff_d   # (B, z_dim)

                    dist_lo = self.listener(
                        torch.cat([mb["listener_pos"], z_hat_lo], dim=-1)
                    )
                    dist_hi = self.listener(
                        torch.cat([mb["listener_pos"], z_hat_hi], dim=-1)
                    )
                    logp_lo = dist_lo.log_prob(mb["actions"])   # (B,)
                    logp_hi = dist_hi.log_prob(mb["actions"])   # (B,)
                    r_lo = torch.exp(logp_lo - mb["old_log_probs"])
                    r_hi = torch.exp(logp_hi - mb["old_log_probs"])
                    # Per-sample task loss at lo/hi bin centres (no PPO clipping)
                    L_lo = -(r_lo * adv) - self.config.entropy_coef * dist_lo.entropy()
                    L_hi = -(r_hi * adv) - self.config.entropy_coef * dist_hi.entropy()
                    # Proxy loss: ∂/∂z_k = (L_hi - L_lo).detach() / δ_k (RB gradient)
                    rb_scale = (L_hi - L_lo).detach().unsqueeze(-1) / _eff_d  # (B, z_dim)
                    rb_proxy = (rb_scale * z_new).sum(dim=-1).mean()
                    total_loss = total_loss + rb_proxy
                    _rb_task_loss = (L_hi - L_lo).detach().mean().item()

                # ── Step 2b: source-coding score function loss (Option 1 / §15) ──
                # Uses histogram populated in the E-step above (frozen during M-step).
                # Active in Phase 1 when loss_comms_mode is "entropy" or "both", and
                # always active in Phase 2 (replaces DLM backward entirely).
                _sc_loss_active = (
                    self.histogram is not None and m is not None
                    and self.config.lambda_comms > 0.0
                    and (self.config.loss_comms_mode in ("entropy", "both")
                         or self._training_phase == 2)
                )
                _sc_rate_loss_tensor: torch.Tensor | None = None
                if _sc_loss_active:
                    _sc_delta = _delta if _delta is not None else self.config.delta
                    _sc_rate_loss_tensor = source_coding_rate_loss(
                        z_new, self.histogram, _sc_delta, self.config.lambda_comms
                    )
                    total_loss = total_loss + _sc_rate_loss_tensor
                    metrics["sc_rate_loss"].append(_sc_rate_loss_tensor.item())
                else:
                    metrics["sc_rate_loss"].append(0.0)

                # ── Step 2c: dither channel entropy loss (Phase 2 only) ──────────
                # Reduces H(m|goal) by pushing frac(z_k/δ) toward 0.5 (bin centres).
                # For the floor SD channel, frac = 0.5 gives zero dither noise;
                # frac = 0 gives maximum noise (1 bit/dim).  The corrected loss
                # targets the true minimum, unlike the old H_binary(frac) formula
                # which pushed toward frac = 0 (the maximum-noise locus).
                # Only active in Phase 2 (task already converged) to avoid
                # disrupting the SR=1 representation learned in Phase 1.
                if (self._training_phase == 2
                        and self.config.lambda_dither > 0.0):
                    _dith_delta = _delta if _delta is not None else self.config.delta
                    d_loss = dither_channel_loss(
                        z_new, _dith_delta, self.config.lambda_dither
                    )
                    total_loss = total_loss + d_loss
                    metrics["dither_loss"].append(d_loss.item())
                else:
                    metrics["dither_loss"].append(0.0)

                # ── Step 3: entropy backward loss — context A ONLY ────────────────
                # Context B backward is DISABLED: q_φ(m|z) conditions on the very z
                # that deterministically produces m = round(z/δ), so the model can
                # trivially achieve NLL ≈ 0 bits without the speaker changing at all.
                # The backward loss becomes a no-op and provides zero compression
                # pressure. Context B is valid as a MEASUREMENT tool (forward loss +
                # metrics) but must never be used for the speaker gradient path.
                # See docs/pillars/PILLAR_P2.md §4 for the full derivation.
                #
                # Level 1 gate: skip backward if q_φ is poorly fitted.
                # Level 3 EMA: use shadow model so gradient points away from current mode.
                # Level 5 Phase 2: entropy backward is the ONLY loss in Phase 2.
                _bwd_gate_active = False
                _entropy_bwd_active = (
                    self.entropy_model is not None and m is not None
                    and self.config.entropy_model_context == "A"
                    and (self.config.loss_comms_mode in ("entropy", "both")
                         or self._training_phase == 2)
                )
                if _entropy_bwd_active:
                    # Level 1: gate — only fire if q_φ is well-fitted (or gate off)
                    _gate_threshold = self.config.qphi_bwd_gate_threshold
                    _gate_ok = (
                        _gate_threshold == float("inf")   # gate disabled
                        or _qphi_gap_now is None           # could not compute (gate off)
                        or _qphi_gap_now <= _gate_threshold
                    )
                    if _gate_ok:
                        _bwd_gate_active = True
                        z_over_delta = z_new / self.config.delta
                        # Level 3: use EMA shadow if configured; otherwise freeze live q_φ
                        if self._ema_entropy_model is not None:
                            # EMA params already have requires_grad=False
                            nll_bwd = self._ema_entropy_model.nll_bits(z_over_delta)
                        else:
                            for p in self.entropy_model.parameters():
                                p.requires_grad_(False)
                            nll_bwd = self.entropy_model.nll_bits(z_over_delta)
                            for p in self.entropy_model.parameters():
                                p.requires_grad_(True)
                        # Use joint NLL (sum over dims) so λ scale is dim-independent
                        total_loss = total_loss + self.config.lambda_comms * nll_bwd.sum(dim=-1).mean()

                # ── Step 3b: gradient decomposition logging (F14) ─────────────────
                # Computes separate speaker grad norms for PPO (task) vs SC (rate) loss
                # using autograd.grad with retain_graph=True before backward consumes
                # the graph. Gated on log_grad_decomp to avoid overhead in production.
                if self.config.log_grad_decomp and total_loss.grad_fn is not None:
                    _sp = [p for p in self.speaker.parameters() if p.requires_grad]
                    try:
                        _ppo_g = torch.autograd.grad(
                            actor_loss + critic_loss, _sp,
                            retain_graph=True, allow_unused=True,
                        )
                        _ppo_norm = float(sum(
                            g.norm() ** 2 for g in _ppo_g if g is not None
                        ) ** 0.5)
                        metrics["ppo_speaker_grad_norm"].append(_ppo_norm)
                        if _sc_rate_loss_tensor is not None:
                            _sc_g = torch.autograd.grad(
                                _sc_rate_loss_tensor, _sp,
                                retain_graph=True, allow_unused=True,
                            )
                            _sc_norm = float(sum(
                                g.norm() ** 2 for g in _sc_g if g is not None
                            ) ** 0.5)
                            metrics["sc_speaker_grad_norm"].append(_sc_norm)
                            if _sc_norm > 1e-9:
                                metrics["ppo_sc_grad_ratio"].append(_ppo_norm / _sc_norm)
                    except RuntimeError:
                        pass  # graph freed early (e.g., no_grad context)

                # ── Step 4: optimizer step ────────────────────────────────────────
                speaker_grad_norm = 0.0
                if total_loss.grad_fn is not None:
                    self.optim.zero_grad(set_to_none=True)
                    if self.optim_delta is not None:
                        self.optim_delta.zero_grad(set_to_none=True)
                    total_loss.backward()
                    # Phase 2: zero only channel gradients (channel params should not
                    # change during compression).  Listener and critic keep their RL
                    # gradients so they can adapt to the speaker's new message distribution.
                    # Design A: δ_k frozen at Phase 2 (requires_grad already False).
                    if self._training_phase == 2:
                        for p in self.channel.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                    # Capture speaker gradient norm BEFORE clipping (raw signal strength)
                    speaker_grad_norm = float(sum(
                        p.grad.detach().norm().item() ** 2
                        for p in self.speaker.parameters()
                        if p.grad is not None
                    ) ** 0.5)
                    nn.utils.clip_grad_norm_(self._trainable, self.config.max_grad_norm)
                    self.optim.step()
                    if self.optim_delta is not None:
                        self.optim_delta.step()
                metrics["speaker_grad_norm"].append(speaker_grad_norm)
                # P1: log per-channel δ_k values
                if self.per_channel_delta is not None:
                    with torch.no_grad():
                        _dv = self.per_channel_delta.delta()
                        for _k in range(_dv.shape[0]):
                            metrics[f"delta_{_k}"].append(_dv[_k].item())
                metrics["training_phase"].append(float(self._training_phase))
                metrics["bwd_gate_active"].append(float(_bwd_gate_active))

                # ── Metrics (no_grad) ──────────────────────────────────────────────
                with torch.no_grad():
                    approx_kl = (mb["old_log_probs"] - new_logp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()
                    true_bits_per_elem = self.channel.transmission_bits_per_elem(z_new, ch_info)
                    true_bits_per_msg = true_bits_per_elem.sum(dim=-1).mean().item()
                    z_norm = z_new.norm(dim=-1).mean().item()

                    # Shannon gap: distance from theoretical minimum
                    shannon_gap = true_bits_per_msg - H_GOAL_BITS
                    bits_to_hg_ratio = true_bits_per_msg / H_GOAL_BITS

                    # Magnitude surrogate — always logged as baseline comparison.
                    # shape: (mb, z_dim); sum over dims gives per-message cost.
                    mag_bits_per_elem = comms_per_elem
                    mag_bits_per_msg = mag_bits_per_elem.sum(dim=-1).mean().item()

                    # P2 metrics: compute nll_log here so it can set the canonical
                    # bits_per_msg and per-goal bits when the entropy model is active.
                    nll_log = None
                    if self.entropy_model is not None and m is not None:
                        m_float_log = m.float().detach()
                        if self.config.entropy_model_context == "A":
                            nll_log = self.entropy_model.nll_bits(m_float_log)
                        else:
                            nll_log = self.entropy_model.nll_bits(m_float_log, z_new.detach())

                    # Canonical bits_per_msg: prior-based when P2 active (honest rate
                    # estimate under the learned code), magnitude otherwise.
                    if nll_log is not None:
                        canonical_bits_per_elem = nll_log          # (mb, z_dim)
                        bits_per_msg = nll_log.sum(dim=-1).mean().item()
                    else:
                        canonical_bits_per_elem = mag_bits_per_elem
                        bits_per_msg = mag_bits_per_msg

                metrics["pg_loss"].append(pg_loss.item())
                metrics["value_loss"].append(critic_loss.item())
                metrics["entropy"].append(entropy_mean.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_frac"].append(clip_frac)
                metrics["comms_loss"].append(comms_mean.item())
                metrics["bits_per_msg"].append(bits_per_msg)
                metrics["mag_bits_per_msg"].append(mag_bits_per_msg)
                metrics["true_bits_per_msg"].append(true_bits_per_msg)
                metrics["z_norm"].append(z_norm)
                metrics["shannon_gap"].append(shannon_gap)
                metrics["bits_to_hg_ratio"].append(bits_to_hg_ratio)
                metrics["warm_start_bits_final"].append(self._warmup_bits_final)
                metrics["rb_task_loss"].append(_rb_task_loss)

                with torch.no_grad():
                    # Per-goal bits: uses canonical source (prior-based or magnitude).
                    per_msg_bits = canonical_bits_per_elem.sum(dim=-1)  # (mb,)
                    for g_idx in mb["goal_ids"].unique():
                        mask = mb["goal_ids"] == g_idx
                        key = f"bits_goal_{g_idx.item()}"
                        metrics[key].append(per_msg_bits[mask].mean().item())

                # Remaining P2 metrics (nll_log already computed above)
                if nll_log is not None:
                    with torch.no_grad():
                        entropy_rate = nll_log.sum(dim=-1).mean().item()  # joint NLL per msg
                        qphi_neg_log_max = nll_log.sum(dim=-1).max().item()
                        h_emp = joint_entropy_bits(m.long())
                        tc = total_correlation_bits(m.long())
                        qphi_gap = entropy_rate - h_emp  # NLL_joint - H_joint ≥ 0
                        bits_vs_mag = mag_bits_per_msg - bits_per_msg

                        metrics["entropy_rate"].append(entropy_rate)
                        metrics["H_m_empirical"].append(h_emp)
                        metrics["qphi_gap"].append(qphi_gap)
                        metrics["tc_bits"].append(tc)
                        metrics["qphi_neg_log_max"].append(qphi_neg_log_max)
                        metrics["bits_vs_magnitude"].append(bits_vs_mag)

                        # Per-dimension marginal entropies H(m_k)
                        h_dims = marginal_entropies_bits(m.long())
                        for k, h_k in enumerate(h_dims):
                            metrics[f"H_dim_{k}"].append(h_k)

                        # Companion context-B oracle bound
                        if self.entropy_model_B is not None:
                            nll_B = self.entropy_model_B.nll_bits(
                                m.float().detach(), z_new.detach()
                            )
                            entropy_rate_B = nll_B.sum(dim=-1).mean().item()  # joint NLL per msg
                            metrics["entropy_rate_B"].append(entropy_rate_B)
                            metrics["context_gap_bits"].append(
                                entropy_rate - entropy_rate_B
                            )

                        # Entropy backward loss magnitude (how strongly P2 nudges speaker)
                        if (self.config.entropy_model_context == "A"
                                and (self.config.loss_comms_mode in ("entropy", "both")
                                     or self._training_phase == 2)):
                            ent_loss_mag = (
                                self.config.lambda_comms
                                * self.entropy_model.nll_bits(
                                    z_new.detach() / self.config.delta
                                ).sum(dim=-1).mean()  # joint NLL — matches backward loss
                            ).item()
                            metrics["entropy_loss_magnitude"].append(ent_loss_mag)

                        # Per-goal model NLL (joint NLL per message, grouped by goal)
                        nll_per_msg = nll_log.sum(dim=-1)  # (mb,) — joint NLL per msg
                        for g_idx in mb["goal_ids"].unique():
                            mask = mb["goal_ids"] == g_idx
                            key = f"nll_goal_{g_idx.item()}"
                            metrics[key].append(nll_per_msg[mask].mean().item())

                # ── Histogram metrics (Option 1 / §15) ───────────────────────────
                # Logged every minibatch regardless of sc_loss_active so we can
                # monitor histogram quality even in magnitude-only mode.
                if self.histogram is not None and m is not None:
                    with torch.no_grad():
                        sc_stats = histogram_rate_stats(m, self.histogram)
                    metrics["hist_entropy_rate"].append(sc_stats["hist_entropy_rate"])
                    metrics["hist_H_empirical"].append(sc_stats["hist_H_empirical"])
                    metrics["hist_qphi_gap"].append(sc_stats["hist_qphi_gap"])

                # ── Dither channel metrics ─────────────────────────────────────────
                # Always logged when channel is quantised so we can track H(m|goal)
                # even when lambda_dither=0 (Phase 1 monitoring).
                if m is not None:
                    with torch.no_grad():
                        _stat_delta = (
                            _delta.detach() if _delta is not None else self.config.delta
                        )
                        d_stats = dither_channel_stats(z_new, _stat_delta)
                    metrics["H_dither_channel"].append(d_stats["H_dither_channel"])
                    metrics["mean_frac"].append(d_stats["mean_frac"])

        # ── F19: moving-target metrics (computed once per update) ─────────────
        if self.config.log_moving_target and _speaker_params_before is not None:
            with torch.no_grad():
                _delta_sq = sum(
                    (p - p0).norm() ** 2
                    for p, p0 in zip(self.speaker.parameters(), _speaker_params_before)
                )
                metrics["speaker_param_delta_norm"] = [float(_delta_sq ** 0.5)]

                # true H(m) from a large offline sample using correct goal distribution
                _n = self.config.moving_target_n
                _g_idx = np.random.choice(len(_GOAL_PROBS), size=_n, p=list(_GOAL_PROBS))
                _goals_arr = np.array(_ENV_DEFAULT_GOALS, dtype=np.float32)[_g_idx]
                _goals_t = torch.tensor(_goals_arr, device=self.device)
                _z_off = self.speaker(_goals_t)
                _, _ch_off = self.channel(_z_off)
                _m_off = _ch_off.get("m")
                if _m_off is not None:
                    metrics["true_H_offline"] = [joint_entropy_bits(_m_off.long())]

        return {k: float(np.mean(v)) for k, v in metrics.items()}
