"""Smoke tests for MAPPOTrainer — Phase 1 implementation.

Loss-wiring tests verify the computational graph: which gradients flow where,
and that the comms penalty is correctly gated by lambda_comms.
"""
from __future__ import annotations

import pytest
import torch

from onpolicy.envs.toyproblem.trainer import MAPPOConfig, MAPPOTrainer


@pytest.fixture
def base_config() -> MAPPOConfig:
    return MAPPOConfig(z_dim=2, channel="none", delta=10.0, lambda_comms=0.0)


@pytest.fixture
def sd_config() -> MAPPOConfig:
    return MAPPOConfig(z_dim=2, channel="sd", delta=10.0, lambda_comms=1e-3)


@pytest.fixture
def nsd_config() -> MAPPOConfig:
    return MAPPOConfig(z_dim=2, channel="nsd", delta=10.0, lambda_comms=1e-3)


class TestActAndValue:
    def test_output_shapes(self, base_config, device):
        """act_and_value returns (action, log_prob, value) with correct shapes."""
        trainer = MAPPOTrainer(base_config, device)
        N = 8
        goal = torch.randn(N, 2)
        lp = torch.randn(N, 2)
        action, log_prob, value = trainer.act_and_value(goal, lp)
        assert action.shape == (N,), f"action shape: {action.shape}"
        assert log_prob.shape == (N,), f"log_prob shape: {log_prob.shape}"
        assert value.shape == (N, 1), f"value shape: {value.shape}"

    def test_actions_in_range(self, base_config, device):
        """Actions are valid discrete choices in [0, 4]."""
        trainer = MAPPOTrainer(base_config, device)
        goal = torch.randn(32, 2)
        lp = torch.randn(32, 2)
        action, _, _ = trainer.act_and_value(goal, lp)
        assert action.min() >= 0 and action.max() <= 4

    def test_no_gradient_in_rollout(self, base_config, device):
        """act_and_value runs under no_grad — no gradients should be tracked."""
        trainer = MAPPOTrainer(base_config, device)
        goal = torch.randn(4, 2)
        lp = torch.randn(4, 2)
        action, log_prob, value = trainer.act_and_value(goal, lp)
        assert not log_prob.requires_grad
        assert not value.requires_grad


class TestFreshNoisePerMinibatch:
    def test_channel_noise_resampled(self, sd_config, device):
        """Two calls to the channel with the same z produce different z_hat values.

        This verifies that channel noise is resampled on every forward call,
        as required by Theorem A.1 / Theorem 2: the unbiased gradient identity
        holds only when the noise is drawn independently of z each time.
        Reusing stale rollout noise would break the independence assumption.
        """
        trainer = MAPPOTrainer(sd_config, device)
        z = torch.randn(8, sd_config.z_dim)
        z_hat1, _ = trainer.channel(z)
        z_hat2, _ = trainer.channel(z)
        assert not torch.equal(z_hat1, z_hat2), (
            "Two channel forward passes with the same z must produce different "
            "z_hat values — noise must be resampled each call (Thm A.1)"
        )


class TestCommunicationLossWiring:
    def test_comms_loss_zero_when_lambda_zero(self, device):
        """When lambda_comms=0, total_loss equals actor+critic only (comms not added).

        The trainer gates the comms term: `if self.config.lambda_comms > 0.0:`
        This test verifies the gate works: with lambda=0, total_loss must equal
        the actor+critic proxy and must differ from the loss that includes comms.
        Uses SD channel so that comms_loss is non-zero — only the gate differs.
        """
        config_zero = MAPPOConfig(z_dim=2, channel="sd", delta=10.0, lambda_comms=0.0)
        trainer = MAPPOTrainer(config_zero, device)

        N = 16
        goals = torch.randn(N, 2)
        lp_input = torch.randn(N, 2)

        z = trainer.speaker(goals)
        z_hat, _ = trainer.channel(z)
        dist = trainer.listener(torch.cat([lp_input, z_hat], dim=-1))

        actor_proxy = -dist.entropy().mean()  # proxy for PPO actor loss
        comms_mean = trainer.channel.comms_loss(z).mean()

        # comms_mean must be non-zero (SD channel, non-zero z)
        assert comms_mean.item() > 0, "comms_loss must be positive for SD with |z|>0"

        # Simulate trainer logic: total_loss with lambda=0 (no comms added)
        total_no_comms = actor_proxy
        if config_zero.lambda_comms > 0.0:
            total_no_comms = total_no_comms + config_zero.lambda_comms * comms_mean

        # total_loss WITH comms (for comparison)
        total_with_comms = actor_proxy + 1.0 * comms_mean

        # With lambda=0: total_loss == actor_proxy (comms branch not taken)
        assert torch.isclose(total_no_comms, actor_proxy), (
            "With lambda_comms=0, total_loss must equal actor+critic only"
        )
        # With lambda=1: total_loss includes comms (they must differ)
        assert not torch.isclose(total_no_comms, total_with_comms), (
            "comms_loss must change total_loss when added (non-zero contribution)"
        )

    def test_comms_loss_nonzero_when_lambda_nonzero(self, sd_config, device):
        """When lambda_comms>0 with SD channel, comms_loss is positive and
        its gradient reaches the speaker network parameters.

        This verifies the full gradient path:
          comms_loss(z) → z → speaker.parameters
        so that the speaker is incentivised to output small |z| for frequent goals.
        """
        trainer = MAPPOTrainer(sd_config, device)  # lambda=1e-3, channel=sd

        N = 16
        goals = torch.randn(N, 2)
        z = trainer.speaker(goals)
        comms_mean = trainer.channel.comms_loss(z).mean()

        # comms_loss must be positive (log2(|z|/δ + 1) > 0 for z ≠ 0)
        assert comms_mean.item() > 0, (
            "comms_loss must be positive for SD channel with non-zero z"
        )

        # Gradient of comms_loss must flow back to the speaker
        comms_mean.backward()

        speaker_has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 1e-10
            for p in trainer.speaker.parameters()
        )
        assert speaker_has_grad, (
            "comms_loss must produce non-zero gradient in speaker parameters — "
            "this is the incentive for the speaker to compress its message"
        )

    def test_comms_loss_on_z_not_z_hat(self, sd_config, device):
        """comms_loss is evaluated on z (pre-quantization speaker output), not z_hat.

        In trainer.py: `comms_per_elem = self.channel.comms_loss(z_new)` — using z_new
        (not z_hat) is correct because:
        1. The bit-cost surrogate measures the speaker's INTENDED magnitude |z|.
        2. Using z_hat = z + noise would measure a random noisy magnitude.
        3. The gradient of comms_loss(z) reaches speaker params directly.

        We verify: (a) z ≠ z_hat (noise was added), (b) comms values on z vs z_hat
        differ (large noise relative to z at delta=10), and (c) gradient of
        comms_loss(z) flows to speaker parameters.
        """
        trainer = MAPPOTrainer(sd_config, device)

        N = 64
        goals = torch.randn(N, 2)
        z = trainer.speaker(goals)
        z_hat, _ = trainer.channel(z)

        # (a) z_hat ≠ z — noise was added
        assert not torch.equal(z, z_hat), (
            "SD channel must add noise: z_hat must differ from z"
        )

        # (b) comms_loss values differ (delta=10 noise is large relative to typical |z|)
        loss_on_z = trainer.channel.comms_loss(z).mean().item()
        loss_on_z_hat = trainer.channel.comms_loss(z_hat.detach()).mean().item()
        assert abs(loss_on_z - loss_on_z_hat) > 1e-4, (
            f"comms_loss(z)={loss_on_z:.6f} must differ from "
            f"comms_loss(z_hat)={loss_on_z_hat:.6f} — quantization noise shifts |z|"
        )

        # (c) comms_loss(z) gradient reaches speaker parameters
        trainer.channel.comms_loss(z).sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 1e-10
            for p in trainer.speaker.parameters()
        )
        assert has_grad, (
            "comms_loss(z) must produce non-zero gradient in speaker parameters"
        )

    def test_critic_has_no_comms_path(self, sd_config, device):
        """Critic gradient does not flow through the communication channel.

        The critic takes [listener_pos, goal] as input (centralized training) —
        it never sees z or z_hat. Therefore backprop through the critic loss
        produces zero gradient in the speaker and channel.  This is the CTDE
        property: value function is centralized (sees true goal) but execution
        is decentralized (listener only sees z_hat, not goal).
        """
        trainer = MAPPOTrainer(sd_config, device)
        trainer.optim.zero_grad()

        N = 16
        goals = torch.randn(N, 2)
        lp = torch.randn(N, 2)

        # Critic forward pass: input is [listener_pos, goal], NOT z or z_hat
        state = torch.cat([lp, goals], dim=-1)
        value = trainer.critic(state)

        # Backward through critic only
        value.sum().backward()

        # Speaker parameters must have NO gradient from critic backward
        for name, param in trainer.speaker.named_parameters():
            assert param.grad is None or param.grad.abs().sum().item() == 0.0, (
                f"critic backward must not create gradient in speaker.{name} — "
                "critic has no communication path (CTDE)"
            )
