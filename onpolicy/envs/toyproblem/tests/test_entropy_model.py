"""Tests for Pillar P2 entropy model classes and helpers."""
from __future__ import annotations
import math
import pytest
import torch
from onpolicy.envs.toyproblem.network import EntropyModelFactored


class TestEntropyModelFactored:
    def test_output_shape_1d(self):
        model = EntropyModelFactored(z_dim=1, K=3)
        m = torch.zeros(32, 1)
        nll = model.nll_bits(m)
        assert nll.shape == (32, 1)

    def test_output_shape_3d(self):
        model = EntropyModelFactored(z_dim=3, K=5)
        m = torch.zeros(64, 3)
        nll = model.nll_bits(m)
        assert nll.shape == (64, 3)

    def test_nll_positive(self):
        """NLL in bits must be non-negative (probability ≤ 1)."""
        torch.manual_seed(0)
        model = EntropyModelFactored(z_dim=3, K=5)
        m = torch.randn(100, 3).round()  # random integers
        nll = model.nll_bits(m)
        assert (nll >= 0).all(), f"negative NLL values: {nll[nll < 0]}"

    def test_probabilities_sum_to_one(self):
        """Σ_m q(m) ≈ 1 over a reasonable integer range."""
        model = EntropyModelFactored(z_dim=1, K=3)
        ms = torch.arange(-50, 51, dtype=torch.float32).unsqueeze(-1)  # (101, 1)
        log_probs = model.log_prob(ms)  # (101, 1)
        total = log_probs.exp().sum().item()
        assert abs(total - 1.0) < 0.01, f"probabilities sum to {total:.4f}"

    def test_grad_flows_to_model_not_input(self):
        """Forward pass (m.float() detached): grad to q_φ, not to m."""
        model = EntropyModelFactored(z_dim=2, K=3)
        m = torch.tensor([[1.0, -1.0], [2.0, 0.0]], requires_grad=False)
        nll = model.nll_bits(m)
        nll.mean().backward()
        assert model.log_pi.grad is not None
        assert model.mu.grad is not None

    def test_grad_flows_to_input_when_frozen(self):
        """Frozen q_φ: grad to z/delta, not to model parameters."""
        model = EntropyModelFactored(z_dim=2, K=3)
        z_over_delta = torch.randn(8, 2, requires_grad=True)
        # freeze
        for p in model.parameters():
            p.requires_grad_(False)
        nll = model.nll_bits(z_over_delta)
        nll.mean().backward()
        assert z_over_delta.grad is not None
        for p in model.parameters():
            assert p.grad is None
        # restore
        for p in model.parameters():
            p.requires_grad_(True)

    def test_wide_init_no_inf(self):
        """Wide log_s init must not produce inf NLL for m in [-10, 10]."""
        model = EntropyModelFactored(z_dim=3, K=5)
        ms = torch.randint(-10, 11, (200, 3)).float()
        nll = model.nll_bits(ms)
        assert torch.isfinite(nll).all()
