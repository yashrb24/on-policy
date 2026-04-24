from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from onpolicy.envs.toyproblem.buffer import RolloutBuffer
from onpolicy.envs.toyproblem.channels import build_channel
from onpolicy.envs.toyproblem.network import Critic, ListenerActor, SpeakerNetwork
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

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        # Per-batch advantage normalization (computed once, applied to all mbs).
        adv_flat = buffer.advantages.flatten()
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std()

        metrics: dict[str, list[float]] = defaultdict(list)

        for _ in range(self.config.update_epochs):
            for mb in buffer.minibatches(self.config.num_minibatches):
                # Fresh forward — speaker weights change across minibatches;
                # channel dither is resampled each pass (unbiased per Thm A.1 / Schuchman).
                z_new = self.speaker(mb["goals"])
                z_hat, ch_info = self.channel(z_new)
                dist = self.listener(torch.cat([mb["listener_pos"], z_hat], dim=-1))
                new_logp = dist.log_prob(mb["actions"])
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - mb["old_log_probs"])
                adv = (mb["advantages"] - adv_mean) / (adv_std + 1e-8)

                # Actor loss: PPO clip + entropy bonus. The `- coef * H`
                # on a minimization target is a positive entropy bonus
                # (encourages high-entropy policies).
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                pg_loss = -torch.min(surr1, surr2).mean()
                entropy_mean = entropy.mean()
                actor_loss = pg_loss - self.config.entropy_coef * entropy_mean

                # Critic loss in normalized space. Update stats on raw returns,
                # then normalize returns for MSE.
                returns_mb = mb["returns"].unsqueeze(-1)  # (mb, 1)
                self.value_norm.update(returns_mb)
                returns_norm = self.value_norm.normalize(returns_mb)

                state_mb = torch.cat([mb["listener_pos"], mb["goals"]], dim=-1)
                new_value = self.critic(state_mb)  # (mb, 1)
                critic_loss = 0.5 * (new_value - returns_norm).pow(2).mean()

                # Communication cost: always compute for logging; only add to
                # the loss when lambda_comms > 0 (IdentityChannel returns zeros anyway).
                comms_per_elem = self.channel.comms_loss(z_new)  # (mb, z_dim)
                comms_mean = comms_per_elem.mean()

                total_loss = actor_loss + critic_loss
                if self.config.lambda_comms > 0.0:
                    total_loss = total_loss + self.config.lambda_comms * comms_mean

                self.optim.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self._trainable, self.config.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    approx_kl = (mb["old_log_probs"] - new_logp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()
                    # Surrogate: differentiable Jensen UB used in training loss.
                    bits_per_msg = comms_per_elem.sum(dim=-1).mean().item()
                    # True transmission cost: float32 for none, log₂(|m|+1) for
                    # quantized channels, fixed B for STE.
                    true_bits_per_elem = self.channel.transmission_bits_per_elem(
                        z_new, ch_info
                    )
                    true_bits_per_msg = true_bits_per_elem.sum(dim=-1).mean().item()
                    z_norm = z_new.norm(dim=-1).mean().item()

                metrics["pg_loss"].append(pg_loss.item())
                metrics["value_loss"].append(critic_loss.item())
                metrics["entropy"].append(entropy_mean.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_frac"].append(clip_frac)
                metrics["comms_loss"].append(comms_mean.item())
                metrics["bits_per_msg"].append(bits_per_msg)
                metrics["true_bits_per_msg"].append(true_bits_per_msg)
                metrics["z_norm"].append(z_norm)

                # Per-goal bit allocation: mean bits used per goal index (0–5).
                with torch.no_grad():
                    bits_per_elem = comms_per_elem.sum(dim=-1)  # (mb,)
                    for g_idx in mb["goal_ids"].unique():
                        mask = mb["goal_ids"] == g_idx
                        key = f"bits_goal_{g_idx.item()}"
                        metrics[key].append(bits_per_elem[mask].mean().item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}
