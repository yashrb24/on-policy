from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from onpolicy.envs.toyproblem.buffer import RolloutBuffer
from onpolicy.envs.toyproblem.channels import build_channel
from onpolicy.envs.toyproblem.entropic import GMMPrior, beta_schedule
from onpolicy.envs.toyproblem.network import Critic, ListenerActor, SpeakerNetwork
from onpolicy.utils.valuenorm import ValueNorm


@dataclass
class MAPPOConfig:
    z_dim: int = 3
    lr: float = 3e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.03
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    num_minibatches: int = 4
    adam_eps: float = 1e-5
    channel: str = "none"
    delta: float = 1.0
    delta_learnable: bool = False
    delta_global_learnable: bool = False
    lambda_comms: float = 0.0
    # --- Entropic GMM Prior ---
    use_entropic_prior: bool = False
    gmm_structure: str = "joint"        # "joint" or "independent"
    num_gmm_components: int = 8
    gmm_init_spread: float = 1.0
    gmm_lr: float = 3e-4
    beta_target: float = 1e-2
    beta_warmup: int = 100_000          # env timesteps before prior kicks in
    beta_anneal: int = 300_000          # env timesteps over which beta ramps
    gmm_tau: float = 0.0               # entropy bonus coefficient for GMM weights


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

        self.speaker = SpeakerNetwork(obs_dim=2, z_dim=config.z_dim).to(device)
        self.listener = ListenerActor(
            obs_dim=2 + config.z_dim, action_dim=5
        ).to(device)
        self.critic = Critic(state_dim=4).to(device)
        self.channel = build_channel(
            config.channel, config.delta, config.delta_learnable,
            config.delta_global_learnable, config.z_dim
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

        # --- Entropic GMM Prior & Independent Optimizer ---
        self.gmm_prior: GMMPrior | None = None
        self.gmm_optim: torch.optim.Optimizer | None = None
        if config.use_entropic_prior:
            self.gmm_prior = GMMPrior(
                num_components=config.num_gmm_components,
                signal_dim=config.z_dim,
                init_spread=config.gmm_init_spread,
                structure=config.gmm_structure,
            ).to(device)
            self.gmm_optim = torch.optim.Adam(
                self.gmm_prior.parameters(),
                lr=config.gmm_lr,
                eps=config.adam_eps,
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

    @staticmethod
    def _param_grad_norm(params) -> float:
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def update(
        self, buffer: RolloutBuffer, timestep: int = 0
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        # Per-batch advantage normalization (computed once, applied to all mbs).
        adv_flat = buffer.advantages.flatten()
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std()

        # Beta for the entropic prior: constant across this update call.
        beta = 0.0
        if self.gmm_prior is not None:
            beta = beta_schedule(
                timestep,
                self.config.beta_warmup,
                self.config.beta_anneal,
                self.config.beta_target,
            )

        metrics: dict[str, list[float]] = defaultdict(list)
        diagnostics: list[dict[str, float]] = []

        for _ in range(self.config.update_epochs):
            for mb in buffer.minibatches(self.config.num_minibatches):
                # Fresh forward — speaker weights change across minibatches;
                # channel dither is resampled each pass (unbiased per Thm A.1 / Schuchman).
                z_new = self.speaker(mb["goals"])
                z_hat, _ = self.channel(z_new)
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

                # Entropic prior contribution. Speaker path: GMM params frozen
                # (grad flows only through z → speaker). GMM path: z detached
                # (grad flows only to GMM). Both summed into one backward.
                prior_nll_val = 0.0
                gmm_entropy_val = 0.0
                if self.gmm_prior is not None:
                    for p in self.gmm_prior.parameters():
                        p.requires_grad_(False)
                    prior_for_speaker = -self.gmm_prior.log_prob(z_new).mean()
                    for p in self.gmm_prior.parameters():
                        p.requires_grad_(True)

                    gmm_mle = -self.gmm_prior.log_prob(z_new.detach()).mean()

                    gmm_repulsion = self.gmm_prior.mean_repulsion()
                    gmm_ent = self.gmm_prior.weights_entropy()
                    total_loss = (
                        total_loss + beta * prior_for_speaker + gmm_mle
                        + self.config.gmm_tau * gmm_repulsion
                    )
                    prior_nll_val = prior_for_speaker.item()
                    gmm_entropy_val = gmm_ent.item()

                self.optim.zero_grad(set_to_none=True)
                if self.gmm_optim is not None:
                    self.gmm_optim.zero_grad(set_to_none=True)
                total_loss.backward()

                # Compute per-component grad norms before clipping.
                gn_speaker  = self._param_grad_norm(self.speaker.parameters())
                gn_listener = self._param_grad_norm(self.listener.parameters())
                gn_critic   = self._param_grad_norm(self.critic.parameters())
                gn_channel  = self._param_grad_norm(self.channel.parameters())

                gn_total = nn.utils.clip_grad_norm_(self._trainable, self.config.max_grad_norm)
                if self.gmm_optim is not None:
                    nn.utils.clip_grad_norm_(
                        self.gmm_prior.parameters(), self.config.max_grad_norm
                    )
                self.optim.step()
                if self.gmm_optim is not None:
                    self.gmm_optim.step()

                with torch.no_grad():
                    approx_kl = (mb["old_log_probs"] - new_logp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()
                    bits_per_msg = comms_per_elem.sum(dim=-1).mean().item()
                    z_norm = z_new.norm(dim=-1).mean().item()

                pg_loss_val   = pg_loss.item()
                actor_loss_val = actor_loss.item()
                value_loss_val = critic_loss.item()
                entropy_val   = entropy_mean.item()
                comms_val     = comms_mean.item()

                metrics["pg_loss"].append(pg_loss_val)
                metrics["value_loss"].append(value_loss_val)
                metrics["entropy"].append(entropy_val)
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_frac"].append(clip_frac)
                metrics["comms_loss"].append(comms_val)
                metrics["bits_per_msg"].append(bits_per_msg)
                metrics["z_norm"].append(z_norm)
                metrics["prior_nll"].append(prior_nll_val)
                metrics["gmm_entropy"].append(gmm_entropy_val)
                metrics["beta"].append(beta)

                diagnostics.append({
                    "pg_loss":          pg_loss_val,
                    "actor_loss":       actor_loss_val,
                    "value_loss":       value_loss_val,
                    "entropy":          entropy_val,
                    "comms_loss":       comms_val,
                    "grad_norm_speaker":  gn_speaker,
                    "grad_norm_listener": gn_listener,
                    "grad_norm_critic":   gn_critic,
                    "grad_norm_channel":  gn_channel,
                    "grad_norm_total":    float(gn_total),
                })

        return {k: float(np.mean(v)) for k, v in metrics.items()}, diagnostics
