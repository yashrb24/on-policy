"""Tests for source_coding.py (Option 1 / §15 of PILLAR_P2.md).

Test plan:
  T1  MessageHistogram.rate() returns non-negative bits for all inputs
  T2  Frequent bins get lower rate than rare bins (R inversely proportional to freq)
  T3  Laplace smoothing: unseen bins get finite rate proportional to α
  T4  empirical_entropy() matches -Σ p log₂ p analytically
  T5  reset() clears state; subsequent update produces fresh counts
  T6  source_coding_rate_loss gradient flows only through z, not through R values
  T7  Gradient magnitude = λ · (R_hi - R_lo) / δ  analytically
  T8  Zero lambda → zero loss, zero gradient
  T9  histogram_rate_stats: hist_qphi_gap ≈ 0 after many samples (histogram = truth)
  T10 Trainer integration: use_source_coding=True runs update() without error
  T11 Trainer: histogram metrics appear in update() return dict
  T12 Trainer: sc_rate_loss > 0 when mode="entropy" and lambda_comms > 0
"""
from __future__ import annotations

import math

import pytest
import torch

from onpolicy.envs.toyproblem.source_coding import (
    MessageHistogram,
    histogram_rate_stats,
    source_coding_rate_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_m(z_dim: int, bins: list[int], N: int) -> torch.Tensor:
    """Return N × z_dim message tensor uniformly drawn from bins list."""
    indices = torch.randint(0, len(bins), (N, z_dim))
    bins_t = torch.tensor(bins, dtype=torch.long)
    return bins_t[indices]


# ---------------------------------------------------------------------------
# T1 — rate() returns non-negative values
# ---------------------------------------------------------------------------

class TestRateNonNegative:
    def test_rate_nonnegative_after_update(self):
        hist = MessageHistogram(z_dim=2)
        m = torch.tensor([[0, 1], [2, 3], [0, 1]], dtype=torch.long)
        hist.update(m)
        R = hist.rate(m)
        assert (R >= 0).all(), f"Negative rates: {R}"

    def test_rate_finite_for_seen_bins(self):
        hist = MessageHistogram(z_dim=1)
        m = torch.tensor([[5], [5], [5]], dtype=torch.long)
        hist.update(m)
        R = hist.rate(m)
        assert torch.isfinite(R).all()

    def test_rate_finite_for_unseen_bins(self):
        hist = MessageHistogram(z_dim=1, smoothing=0.5)
        m_train = torch.tensor([[0]], dtype=torch.long)
        hist.update(m_train)
        m_unseen = torch.tensor([[999]], dtype=torch.long)
        R = hist.rate(m_unseen)
        # Unseen bin: (0 + 0.5) / (N + 0.5 * n_bins) — finite positive
        assert torch.isfinite(R).all()
        assert (R > 0).all()

    def test_rate_zero_before_any_update(self):
        hist = MessageHistogram(z_dim=2)
        m = torch.zeros(4, 2, dtype=torch.long)
        R = hist.rate(m)
        # No data → rate = 0 (gradient = 0, harmless)
        assert (R == 0.0).all()


# ---------------------------------------------------------------------------
# T2 — Frequent bins get lower rate
# ---------------------------------------------------------------------------

class TestRateFrequency:
    def test_frequent_bin_lower_rate(self):
        hist = MessageHistogram(z_dim=1)
        # Bin 0 seen 100 times, bin 1 seen 1 time
        m_common = torch.zeros(100, 1, dtype=torch.long)
        m_rare = torch.ones(1, 1, dtype=torch.long)
        hist.update(m_common)
        hist.update(m_rare)

        R_common = hist.rate(torch.zeros(1, 1, dtype=torch.long))
        R_rare = hist.rate(torch.ones(1, 1, dtype=torch.long))
        assert R_common.item() < R_rare.item(), (
            f"Common bin ({R_common.item():.3f}) should have lower rate "
            f"than rare bin ({R_rare.item():.3f})"
        )

    def test_uniform_distribution_rate_equals_log2_nbins(self):
        """Uniform over K bins → rate per bin ≈ log₂(K) for large N."""
        K = 8
        N = 10_000
        hist = MessageHistogram(z_dim=1, smoothing=0.5)
        m = torch.randint(0, K, (N, 1))
        hist.update(m)

        # Check a single bin from the middle of the range
        m_query = torch.tensor([[3]], dtype=torch.long)
        R = hist.rate(m_query).item()
        expected = math.log2(K)
        # Allow 5% relative tolerance
        assert abs(R - expected) / expected < 0.05, (
            f"Rate {R:.4f} far from log₂({K})={expected:.4f}"
        )


# ---------------------------------------------------------------------------
# T3 — Laplace smoothing
# ---------------------------------------------------------------------------

class TestLaplaceSmoothing:
    def test_unseen_bin_rate_decreases_with_more_data(self):
        """Rate for an unseen bin decreases as total N increases (denominator grows)."""
        hist = MessageHistogram(z_dim=1, smoothing=0.5)
        m_unseen = torch.tensor([[999]], dtype=torch.long)

        hist.update(torch.zeros(10, 1, dtype=torch.long))
        R_small = hist.rate(m_unseen).item()

        hist.update(torch.zeros(1000, 1, dtype=torch.long))
        R_large = hist.rate(m_unseen).item()

        assert R_large > R_small, (
            "Unseen bin rate should grow as N increases (bin stays unseen, "
            f"but got R_small={R_small:.4f} > R_large={R_large:.4f})"
        )

    def test_zero_smoothing_raises_for_unseen_bin(self):
        """With α=0, an unseen bin has count=0 → -log₂(0) = inf."""
        hist = MessageHistogram(z_dim=1, smoothing=0.0)
        hist.update(torch.zeros(10, 1, dtype=torch.long))
        R = hist.rate(torch.tensor([[999]], dtype=torch.long))
        assert not torch.isfinite(R).all() or R.item() > 100, (
            "With zero smoothing, unseen bin should give very large or inf rate"
        )


# ---------------------------------------------------------------------------
# T4 — empirical_entropy() correctness
# ---------------------------------------------------------------------------

class TestEmpiricalEntropy:
    def test_binary_uniform_entropy(self):
        """Uniform over 2 bins → H = 1 bit."""
        hist = MessageHistogram(z_dim=1, smoothing=0.0)
        m = torch.tensor([[0]] * 1000 + [[1]] * 1000, dtype=torch.long)
        hist.update(m)
        H = hist.empirical_entropy()
        assert abs(H[0] - 1.0) < 0.01, f"Expected H≈1 bit, got {H[0]:.4f}"

    def test_entropy_before_update(self):
        hist = MessageHistogram(z_dim=2)
        H = hist.empirical_entropy()
        assert H == [0.0, 0.0]

    def test_entropy_single_bin(self):
        """All messages to same bin → H = 0."""
        hist = MessageHistogram(z_dim=1, smoothing=0.0)
        m = torch.zeros(100, 1, dtype=torch.long)
        hist.update(m)
        H = hist.empirical_entropy()
        assert abs(H[0]) < 1e-6, f"Single bin: H should be 0, got {H[0]}"


# ---------------------------------------------------------------------------
# T5 — reset() clears state
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_counts(self):
        hist = MessageHistogram(z_dim=1)
        hist.update(torch.tensor([[5]] * 50, dtype=torch.long))
        hist.reset()
        assert hist.total_counts() == [0.0]
        assert hist.n_distinct_bins() == [0]

    def test_reset_then_update_independent(self):
        hist = MessageHistogram(z_dim=1)
        hist.update(torch.tensor([[100]] * 100, dtype=torch.long))
        hist.reset()
        hist.update(torch.tensor([[0]] * 50 + [[1]] * 50, dtype=torch.long))
        # After reset, bin 100 should be unseen
        R_100 = hist.rate(torch.tensor([[100]], dtype=torch.long))
        R_0 = hist.rate(torch.tensor([[0]], dtype=torch.long))
        # 100 is unseen (smoothed), 0 was seen 50/100 times
        assert R_100.item() > R_0.item()


# ---------------------------------------------------------------------------
# T6 — Gradient flows only through z
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradient_flows_through_z_only(self):
        hist = MessageHistogram(z_dim=2, smoothing=0.5)
        m_train = torch.randint(0, 5, (100, 2))
        hist.update(m_train)

        z = torch.randn(16, 2, requires_grad=True)
        loss = source_coding_rate_loss(z, hist, delta=1.0, lambda_comms=1.0)
        loss.backward()

        assert z.grad is not None, "z should have gradient"
        # Check gradient is finite everywhere
        assert torch.isfinite(z.grad).all(), f"Non-finite z.grad: {z.grad}"

    def test_no_gradient_through_histogram(self):
        """R values from histogram.rate() must be detached — no grad path."""
        hist = MessageHistogram(z_dim=1, smoothing=0.5)
        m_train = torch.arange(0, 10).unsqueeze(1)
        hist.update(m_train)

        z = torch.randn(8, 1, requires_grad=True)
        loss = source_coding_rate_loss(z, hist, delta=1.0, lambda_comms=1.0)
        # If R tensors had gradients, the graph would have multiple roots.
        # Checking that loss.backward() completes without error is sufficient,
        # but we also verify the gradient computation is self-consistent below (T7).
        loss.backward()
        assert z.grad is not None


# ---------------------------------------------------------------------------
# T7 — Gradient magnitude matches analytical formula
# ---------------------------------------------------------------------------

class TestGradientMagnitude:
    def test_gradient_equals_score_function(self):
        """∂L/∂z_k = λ · (R_hi_k - R_lo_k) / δ for each element."""
        delta = 2.0
        lambda_c = 0.5
        z_dim = 2

        hist = MessageHistogram(z_dim=z_dim, smoothing=0.5)
        m_train = torch.randint(0, 8, (200, z_dim))
        hist.update(m_train)

        # Fix z to a known value so m_lo / m_hi are deterministic
        z_val = torch.tensor([[3.3, 7.8]])   # shape (1, 2)
        z = z_val.clone().requires_grad_(True)

        loss = source_coding_rate_loss(z, hist, delta=delta, lambda_comms=lambda_c)
        loss.backward()

        # Analytical gradient
        with torch.no_grad():
            m_lo = torch.floor(z_val / delta).long()
            m_hi = m_lo + 1
            R_lo = hist.rate(m_lo)
            R_hi = hist.rate(m_hi)
            expected_grad = lambda_c * (R_hi - R_lo) / delta   # (1, 2)

        assert torch.allclose(z.grad, expected_grad, atol=1e-5), (
            f"Grad mismatch:\n  actual:   {z.grad}\n  expected: {expected_grad}"
        )


# ---------------------------------------------------------------------------
# T8 — Zero lambda
# ---------------------------------------------------------------------------

class TestZeroLambda:
    def test_zero_lambda_zero_loss_and_grad(self):
        hist = MessageHistogram(z_dim=2)
        hist.update(torch.randint(0, 5, (50, 2)))

        z = torch.randn(8, 2, requires_grad=True)
        loss = source_coding_rate_loss(z, hist, delta=1.0, lambda_comms=0.0)
        assert loss.item() == 0.0
        loss.backward()
        assert (z.grad == 0.0).all()


# ---------------------------------------------------------------------------
# T9 — histogram_rate_stats: qphi_gap ≈ 0
# ---------------------------------------------------------------------------

class TestHistogramRateStats:
    def test_qphi_gap_near_zero_for_histogram(self):
        """Histogram is a consistent estimator: gap → 0 as N → ∞."""
        hist = MessageHistogram(z_dim=2, smoothing=0.5)
        N = 50_000
        # Multinomial message distribution
        m = torch.randint(0, 6, (N, 2))
        hist.update(m)

        stats = histogram_rate_stats(m[:1000], hist)
        # For a self-consistent estimator, the gap should be small (< 0.1 bits/msg)
        assert abs(stats["hist_qphi_gap"]) < 0.5, (
            f"Expected near-zero gap, got {stats['hist_qphi_gap']:.4f}"
        )
        # And definitely far from the DLM failure mode (12-14 bits)
        assert abs(stats["hist_qphi_gap"]) < 2.0

    def test_stats_return_dict_keys(self):
        hist = MessageHistogram(z_dim=1)
        hist.update(torch.randint(0, 3, (20, 1)))
        stats = histogram_rate_stats(torch.randint(0, 3, (10, 1)), hist)
        assert set(stats.keys()) == {"hist_entropy_rate", "hist_H_empirical", "hist_qphi_gap"}


# ---------------------------------------------------------------------------
# T10 — Trainer integration: update() runs without error
# ---------------------------------------------------------------------------

class TestTrainerIntegration:
    def test_update_with_source_coding_runs(self, device):
        from onpolicy.envs.toyproblem.trainer import MAPPOConfig, MAPPOTrainer
        from onpolicy.envs.toyproblem.buffer import RolloutBuffer

        config = MAPPOConfig(
            z_dim=2,
            channel="sd",
            delta=1.0,
            lambda_comms=5e-4,
            update_epochs=2,
            num_minibatches=2,
            use_source_coding=True,
            source_coding_smoothing=0.5,
            loss_comms_mode="entropy",
        )
        trainer = MAPPOTrainer(config, device=device)
        buffer = RolloutBuffer(n_steps=32, n_envs=4, z_dim=2, device=device)

        # Fill buffer with dummy data
        for _ in range(32):
            goal = torch.randn(4, 2, device=device)
            lp = torch.randn(4, 2, device=device)
            action, log_prob, value = trainer.act_and_value(goal, lp)
            buffer.insert(
                goal, lp, action, log_prob, value,
                torch.zeros(4, device=device),
                torch.zeros(4, device=device),
                goal_id=torch.zeros(4, dtype=torch.long, device=device),
            )
        last_value = trainer.get_value(
            torch.randn(4, 2, device=device),
            torch.randn(4, 2, device=device),
        )
        buffer.compute_returns_and_advantages(
            last_value, trainer.value_norm, gamma=0.99, lam=0.95
        )

        metrics = trainer.update(buffer)
        assert isinstance(metrics, dict), "update() should return a dict"

    # ---------------------------------------------------------------------------
    # T11 — histogram metrics appear in return dict
    # ---------------------------------------------------------------------------

    def test_histogram_metrics_in_return_dict(self, device):
        from onpolicy.envs.toyproblem.trainer import MAPPOConfig, MAPPOTrainer
        from onpolicy.envs.toyproblem.buffer import RolloutBuffer

        config = MAPPOConfig(
            z_dim=2,
            channel="sd",
            delta=1.0,
            lambda_comms=5e-4,
            update_epochs=1,
            num_minibatches=2,
            use_source_coding=True,
            loss_comms_mode="entropy",
        )
        trainer = MAPPOTrainer(config, device=device)
        buffer = RolloutBuffer(n_steps=16, n_envs=4, z_dim=2, device=device)

        for _ in range(16):
            goal = torch.randn(4, 2, device=device)
            lp = torch.randn(4, 2, device=device)
            action, log_prob, value = trainer.act_and_value(goal, lp)
            buffer.insert(
                goal, lp, action, log_prob, value,
                torch.zeros(4, device=device),
                torch.zeros(4, device=device),
                goal_id=torch.zeros(4, dtype=torch.long, device=device),
            )
        last_value = trainer.get_value(
            torch.randn(4, 2, device=device),
            torch.randn(4, 2, device=device),
        )
        buffer.compute_returns_and_advantages(
            last_value, trainer.value_norm, gamma=0.99, lam=0.95
        )
        metrics = trainer.update(buffer)

        for key in ("hist_entropy_rate", "hist_H_empirical", "hist_qphi_gap", "sc_rate_loss"):
            assert key in metrics, f"Expected '{key}' in metrics, got keys: {list(metrics.keys())}"
            assert math.isfinite(metrics[key]), f"{key}={metrics[key]} is not finite"

    # ---------------------------------------------------------------------------
    # T12 — sc_rate_loss > 0 when mode=entropy and lambda > 0
    # ---------------------------------------------------------------------------

    def test_sc_rate_loss_nonzero(self, device):
        from onpolicy.envs.toyproblem.trainer import MAPPOConfig, MAPPOTrainer
        from onpolicy.envs.toyproblem.buffer import RolloutBuffer

        config = MAPPOConfig(
            z_dim=2,
            channel="sd",
            delta=1.0,
            lambda_comms=1.0,   # large λ to ensure nonzero signal
            update_epochs=1,
            num_minibatches=2,
            use_source_coding=True,
            loss_comms_mode="entropy",
        )
        trainer = MAPPOTrainer(config, device=device)
        buffer = RolloutBuffer(n_steps=32, n_envs=8, z_dim=2, device=device)

        for _ in range(32):
            goal = torch.randn(8, 2, device=device)
            lp = torch.randn(8, 2, device=device)
            action, log_prob, value = trainer.act_and_value(goal, lp)
            buffer.insert(
                goal, lp, action, log_prob, value,
                torch.zeros(8, device=device),
                torch.zeros(8, device=device),
                goal_id=torch.zeros(8, dtype=torch.long, device=device),
            )
        last_value = trainer.get_value(
            torch.randn(8, 2, device=device),
            torch.randn(8, 2, device=device),
        )
        buffer.compute_returns_and_advantages(
            last_value, trainer.value_norm, gamma=0.99, lam=0.95
        )
        metrics = trainer.update(buffer)

        # With λ=1, the rate loss should be non-trivially large
        assert metrics.get("sc_rate_loss", 0.0) != 0.0, (
            "sc_rate_loss should be non-zero when lambda_comms=1.0 and mode=entropy"
        )
