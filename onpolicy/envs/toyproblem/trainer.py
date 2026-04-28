from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from onpolicy.envs.toyproblem.buffer import RolloutBuffer
from onpolicy.envs.toyproblem.channels import build_channel, H_GOAL_BITS
from onpolicy.envs.toyproblem.network import (
    Critic, ListenerActor, SpeakerNetwork,
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
        # Warm-start final loss (NaN until warm-start runs)
        self._warmup_bits_final: float = float("nan")
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

        self._trainable = (
            list(self.speaker.parameters())
            + list(self.listener.parameters())
            + list(self.critic.parameters())
            + list(self.channel.parameters())
        )
        self.optim = torch.optim.Adam(
            self._trainable, lr=config.lr, eps=config.adam_eps
        )

    @torch.no_grad()
    def act_and_value(
        self, goal: torch.Tensor, listener_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.speaker(goal)
        z_hat, _ = self.channel(z)
        dist = self.listener(torch.cat([listener_pos, z_hat], dim=-1))
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
                    _, ch_info = self.channel(z)
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

        for _ in range(self.config.update_epochs):
            for mb in buffer.minibatches(self.config.num_minibatches):
                z_new = self.speaker(mb["goals"])
                z_hat, ch_info = self.channel(z_new)
                m = ch_info.get("m")

                # ── Step 1: q_φ forward update BEFORE the RL step ─────────────────
                # Rationale: updating q_φ first ensures the backward entropy loss
                # (step 3) uses a prior that has already tracked the current batch's
                # message distribution, giving a fresher rate signal. Using z_new
                # from the current minibatch avoids the stale-z problem that arises
                # when q_φ is updated after the speaker has already been changed.
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
                    # Companion context-B forward update (measurement only)
                    if self.entropy_model_B is not None:
                        nll_B_fwd = self.entropy_model_B.nll_bits(
                            m_float, z_new.detach()
                        )
                        self.optim_qphi_B.zero_grad(set_to_none=True)
                        nll_B_fwd.mean().backward()
                        self.optim_qphi_B.step()

                # ── Step 2: RL losses ──────────────────────────────────────────────
                dist = self.listener(torch.cat([mb["listener_pos"], z_hat], dim=-1))
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

                comms_per_elem = self.channel.comms_loss(z_new)
                comms_mean = comms_per_elem.mean()

                total_loss = actor_loss + critic_loss
                if self.config.lambda_comms > 0.0:
                    if self.config.loss_comms_mode in ("magnitude", "both"):
                        total_loss = total_loss + self.config.lambda_comms * comms_mean

                # ── Step 3: entropy backward loss — context A ONLY ────────────────
                # Context B backward is DISABLED: q_φ(m|z) conditions on the very z
                # that deterministically produces m = round(z/δ), so the model can
                # trivially achieve NLL ≈ 0 bits without the speaker changing at all.
                # The backward loss becomes a no-op and provides zero compression
                # pressure. Context B is valid as a MEASUREMENT tool (forward loss +
                # metrics) but must never be used for the speaker gradient path.
                # See docs/pillars/PILLAR_P2.md §4 for the full derivation.
                if (self.entropy_model is not None and m is not None
                        and self.config.loss_comms_mode in ("entropy", "both")
                        and self.config.entropy_model_context == "A"):
                    z_over_delta = z_new / self.config.delta
                    # Freeze q_φ: grad flows to z (speaker), not to q_φ params
                    for p in self.entropy_model.parameters():
                        p.requires_grad_(False)
                    nll_bwd = self.entropy_model.nll_bits(z_over_delta)
                    for p in self.entropy_model.parameters():
                        p.requires_grad_(True)
                    total_loss = total_loss + self.config.lambda_comms * nll_bwd.mean()

                # ── Step 4: RL optimizer step ──────────────────────────────────────
                self.optim.zero_grad(set_to_none=True)
                total_loss.backward()
                # Capture speaker gradient norm BEFORE clipping (raw signal strength)
                speaker_grad_norm = float(sum(
                    p.grad.detach().norm().item() ** 2
                    for p in self.speaker.parameters()
                    if p.grad is not None
                ) ** 0.5)
                nn.utils.clip_grad_norm_(self._trainable, self.config.max_grad_norm)
                self.optim.step()
                metrics["speaker_grad_norm"].append(speaker_grad_norm)

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
                        if (self.config.loss_comms_mode in ("entropy", "both")
                                and self.config.entropy_model_context == "A"):
                            ent_loss_mag = (
                                self.config.lambda_comms
                                * self.entropy_model.nll_bits(
                                    z_new.detach() / self.config.delta
                                ).mean()
                            ).item()
                            metrics["entropy_loss_magnitude"].append(ent_loss_mag)

                        # Per-goal model NLL (joint NLL per message, grouped by goal)
                        nll_per_msg = nll_log.sum(dim=-1)  # (mb,) — joint NLL per msg
                        for g_idx in mb["goal_ids"].unique():
                            mask = mb["goal_ids"] == g_idx
                            key = f"nll_goal_{g_idx.item()}"
                            metrics[key].append(nll_per_msg[mask].mean().item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}
