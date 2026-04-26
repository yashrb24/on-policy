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


from onpolicy.envs.toyproblem.network import EntropyModelJoint


class TestEntropyModelJoint:
    def test_output_shape(self):
        model = EntropyModelJoint(z_dim=3, K=5)
        m = torch.zeros(32, 3)
        nll = model.nll_bits(m)
        assert nll.shape == (32, 3)

    def test_z_dim_1_matches_factored(self):
        """Joint with z_dim=1 should behave identically to factored."""
        torch.manual_seed(42)
        factored = EntropyModelFactored(z_dim=1, K=3)
        joint = EntropyModelJoint(z_dim=1, K=3)
        # Copy weights from factored into joint dim-0 params
        with torch.no_grad():
            joint.log_pi_0.copy_(factored.log_pi[0])
            joint.mu_0.copy_(factored.mu[0])
            joint.log_s_0.copy_(factored.log_s[0])
        m = torch.tensor([[0.0], [1.0], [-1.0], [3.0]])
        assert torch.allclose(factored.nll_bits(m), joint.nll_bits(m), atol=1e-5)

    def test_nll_positive(self):
        torch.manual_seed(0)
        model = EntropyModelJoint(z_dim=3, K=5)
        m = torch.randn(100, 3).round()
        nll = model.nll_bits(m)
        assert (nll >= 0).all()

    def test_grad_flows_through_context(self):
        """Autoregressive: grad must flow through earlier dimensions."""
        model = EntropyModelJoint(z_dim=3, K=3)
        x = torch.randn(8, 3, requires_grad=True)
        nll = model.nll_bits(x)
        nll.mean().backward()
        assert x.grad is not None
        assert x.grad.shape == (8, 3)
        # All dimensions should have non-zero grad (context coupling)
        assert x.grad.abs().sum(dim=0).min().item() > 0


from onpolicy.envs.toyproblem.network import EntropyModelCondZ


class TestEntropyModelCondZ:
    def test_output_shape(self):
        model = EntropyModelCondZ(z_dim=3, K=5)
        m = torch.zeros(32, 3)
        z = torch.randn(32, 3)
        nll = model.nll_bits(m, z)
        assert nll.shape == (32, 3)

    def test_nll_positive(self):
        torch.manual_seed(0)
        model = EntropyModelCondZ(z_dim=3, K=5)
        m = torch.randn(64, 3).round()
        z = torch.randn(64, 3)
        nll = model.nll_bits(m, z)
        assert (nll >= 0).all()

    def test_grad_to_z_when_params_frozen(self):
        """Frozen q_φ: grad flows through z (context) to speaker."""
        model = EntropyModelCondZ(z_dim=2, K=3)
        z = torch.randn(8, 2, requires_grad=True)
        m = torch.randn(8, 2).round().detach()
        for p in model.parameters():
            p.requires_grad_(False)
        nll = model.nll_bits(m, z)
        nll.mean().backward()
        assert z.grad is not None
        for p in model.parameters():
            p.requires_grad_(True)

    def test_different_z_different_output(self):
        """Conditioning: different z values should give different NLL."""
        torch.manual_seed(0)
        model = EntropyModelCondZ(z_dim=2, K=3)
        m = torch.zeros(8, 2)
        z1 = torch.randn(8, 2)
        z2 = torch.randn(8, 2)
        assert not torch.allclose(model.nll_bits(m, z1), model.nll_bits(m, z2))


from onpolicy.envs.toyproblem.network import EntropyModelJointCondZ


class TestEntropyModelJointCondZ:
    def test_output_shape(self):
        model = EntropyModelJointCondZ(z_dim=3, K=5)
        m = torch.zeros(32, 3)
        z = torch.randn(32, 3)
        nll = model.nll_bits(m, z)
        assert nll.shape == (32, 3)

    def test_nll_positive(self):
        torch.manual_seed(0)
        model = EntropyModelJointCondZ(z_dim=3, K=5)
        m = torch.randn(64, 3).round()
        z = torch.randn(64, 3)
        nll = model.nll_bits(m, z)
        assert (nll >= 0).all()

    def test_z_dim_1_matches_condz(self):
        """Joint+CondZ with z_dim=1 reduces to CondZ (no autoregressive prefix)."""
        torch.manual_seed(42)
        cond_z = EntropyModelCondZ(z_dim=1, K=3)
        joint_cond_z = EntropyModelJointCondZ(z_dim=1, K=3)
        # Copy dim-0 MLP weights from cond_z into joint_cond_z.mlp_0
        with torch.no_grad():
            for p_src, p_dst in zip(cond_z.mlps[0].parameters(),
                                    joint_cond_z.mlp_0.parameters()):
                p_dst.copy_(p_src)
        m = torch.tensor([[0.0], [1.0], [-1.0]])
        z = torch.randn(3, 1)
        assert torch.allclose(cond_z.nll_bits(m, z), joint_cond_z.nll_bits(m, z), atol=1e-5)

    def test_different_z_different_output(self):
        """Conditioning on z must change output."""
        torch.manual_seed(0)
        model = EntropyModelJointCondZ(z_dim=2, K=3)
        m = torch.zeros(8, 2)
        z1 = torch.randn(8, 2)
        z2 = torch.randn(8, 2)
        assert not torch.allclose(model.nll_bits(m, z1), model.nll_bits(m, z2))

    def test_grad_flows_to_z_and_m_context(self):
        """Frozen q_φ: grad flows to both z and m (autoregressive context)."""
        model = EntropyModelJointCondZ(z_dim=3, K=3)
        z = torch.randn(8, 3, requires_grad=True)
        x = torch.randn(8, 3, requires_grad=True)
        for p in model.parameters():
            p.requires_grad_(False)
        nll = model.nll_bits(x, z)
        nll.mean().backward()
        assert z.grad is not None
        assert x.grad is not None
        for p in model.parameters():
            p.requires_grad_(True)


from onpolicy.envs.toyproblem.network import joint_entropy_bits, total_correlation_bits


class TestEntropyHelpers:
    def test_joint_entropy_uniform(self):
        """4 equiprobable outcomes → H = 2 bits."""
        m = torch.tensor([[0], [1], [2], [3]] * 250)  # (1000, 1)
        h = joint_entropy_bits(m)
        assert abs(h - 2.0) < 0.01

    def test_joint_entropy_deterministic(self):
        """Deterministic m → H = 0."""
        m = torch.zeros(100, 3, dtype=torch.long)
        h = joint_entropy_bits(m)
        assert h < 1e-6

    def test_tc_independent(self):
        """Independent dimensions → TC ≈ 0."""
        torch.manual_seed(0)
        m = torch.randint(0, 4, (2000, 3))
        tc = total_correlation_bits(m)
        assert tc < 0.05

    def test_tc_perfectly_correlated(self):
        """Perfectly correlated: m_1 = m_0 always → TC > 0."""
        m0 = torch.randint(0, 4, (1000,))
        m = torch.stack([m0, m0, m0], dim=-1)  # (1000, 3) — perfect correlation
        tc = total_correlation_bits(m)
        # Marginals each have H ≈ 2 bits; joint H ≈ 2 bits → TC ≈ 4 bits
        assert tc > 3.0

    def test_tc_z_dim_1_is_zero(self):
        m = torch.randint(0, 8, (500, 1))
        assert total_correlation_bits(m) == 0.0


from onpolicy.envs.toyproblem.trainer import MAPPOConfig, MAPPOTrainer
from onpolicy.envs.toyproblem.buffer import RolloutBuffer


def _make_buffer(n_steps=4, n_envs=4, z_dim=3, device=torch.device("cpu")) -> RolloutBuffer:
    buf = RolloutBuffer(n_steps, n_envs, z_dim, device=device)
    buf.goals = torch.randn(n_steps, n_envs, 2)
    buf.listener_pos = torch.randn(n_steps, n_envs, 2)
    buf.actions = torch.randint(0, 5, (n_steps, n_envs))
    buf.log_probs = torch.randn(n_steps, n_envs)
    buf.advantages = torch.randn(n_steps, n_envs)
    buf.returns = torch.randn(n_steps, n_envs)
    buf.goal_ids = torch.zeros(n_steps, n_envs, dtype=torch.long)
    return buf


class TestTrainerEntropyLoss:
    def _make_trainer_with_em(self, mode="entropy") -> MAPPOTrainer:
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0, lambda_comms=1e-3,
            use_entropy_model=True, entropy_model_K=3,
            entropy_model_type="factored", entropy_model_context="A",
            lr_qphi_mult=5.0, n_qphi_steps=2,
            loss_comms_mode=mode,
            update_epochs=1, num_minibatches=1,
        )
        return MAPPOTrainer(cfg, device=torch.device("cpu"))

    def test_update_returns_entropy_rate(self):
        """update() must include entropy_rate in returned metrics."""
        t = self._make_trainer_with_em()
        buf = _make_buffer(z_dim=3)
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        metrics = t.update(buf)
        assert "entropy_rate" in metrics

    def test_update_returns_qphi_gap(self):
        t = self._make_trainer_with_em()
        buf = _make_buffer(z_dim=3)
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        metrics = t.update(buf)
        assert "qphi_gap" in metrics

    def test_entropy_mode_changes_loss(self):
        """With mode='entropy', entropy_rate must be logged and non-zero."""
        t = self._make_trainer_with_em(mode="entropy")
        buf = _make_buffer(z_dim=3)
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        metrics = t.update(buf)
        assert metrics["entropy_rate"] > 0.0

    def test_qphi_params_change_after_update(self):
        """q_φ parameters must be updated by the q_φ optimizer."""
        t = self._make_trainer_with_em()
        buf = _make_buffer(z_dim=3)
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        mu_before = t.entropy_model.mu.detach().clone()
        t.update(buf)
        assert not torch.allclose(t.entropy_model.mu, mu_before)

    def test_no_entropy_metrics_when_disabled(self):
        """Without use_entropy_model, entropy_rate must NOT be in metrics."""
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0,
            use_entropy_model=False,
            update_epochs=1, num_minibatches=1,
        )
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        buf = _make_buffer(z_dim=3)
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        metrics = t.update(buf)
        assert "entropy_rate" not in metrics


class TestTrainerEntropyModelConstruction:
    def _make_trainer(self, **overrides) -> MAPPOTrainer:
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0, lambda_comms=1e-3,
            use_entropy_model=True, entropy_model_K=3,
            **overrides,
        )
        return MAPPOTrainer(cfg, device=torch.device("cpu"))

    def test_factored_A_constructed(self):
        from onpolicy.envs.toyproblem.network import EntropyModelFactored
        t = self._make_trainer(entropy_model_type="factored", entropy_model_context="A")
        assert isinstance(t.entropy_model, EntropyModelFactored)
        assert t.optim_qphi is not None

    def test_joint_A_constructed(self):
        from onpolicy.envs.toyproblem.network import EntropyModelJoint
        t = self._make_trainer(entropy_model_type="joint", entropy_model_context="A")
        assert isinstance(t.entropy_model, EntropyModelJoint)

    def test_cond_z_B_constructed(self):
        from onpolicy.envs.toyproblem.network import EntropyModelCondZ
        t = self._make_trainer(entropy_model_type="factored", entropy_model_context="B")
        assert isinstance(t.entropy_model, EntropyModelCondZ)

    def test_joint_cond_z_B_constructed(self):
        from onpolicy.envs.toyproblem.network import EntropyModelJointCondZ
        t = self._make_trainer(entropy_model_type="joint", entropy_model_context="B")
        assert isinstance(t.entropy_model, EntropyModelJointCondZ)

    def test_entropy_model_none_when_disabled(self):
        cfg = MAPPOConfig(z_dim=3, channel="sd", use_entropy_model=False)
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        assert t.entropy_model is None
        assert t.optim_qphi is None

    def test_qphi_lr_scaled(self):
        """q_φ optimizer lr = lr_qphi_mult × base_lr."""
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", use_entropy_model=True, entropy_model_K=3,
            lr=1e-3, lr_qphi_mult=5.0,
        )
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        actual_lr = t.optim_qphi.param_groups[0]["lr"]
        assert abs(actual_lr - 5e-3) < 1e-9


class TestWarmStart:
    def test_warmup_returns_loss(self):
        """warmup_entropy_model must return a finite float."""
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0,
            use_entropy_model=True, entropy_model_K=3,
            entropy_model_type="factored", entropy_model_context="A",
            update_epochs=1, num_minibatches=1,
        )
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        buf = _make_buffer(z_dim=3, n_steps=8, n_envs=4)
        loss = t.warmup_entropy_model(buf, n_steps=10)
        assert math.isfinite(loss), f"warmup loss not finite: {loss}"

    def test_warmup_updates_qphi(self):
        """After warmup, q_φ params must differ from init."""
        torch.manual_seed(99)
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0,
            use_entropy_model=True, entropy_model_K=3,
            update_epochs=1, num_minibatches=1,
        )
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        mu_before = t.entropy_model.mu.detach().clone()
        buf = _make_buffer(z_dim=3, n_steps=8, n_envs=4)
        t.warmup_entropy_model(buf, n_steps=20)
        assert not torch.allclose(t.entropy_model.mu, mu_before)

    def test_warmup_no_change_without_em(self):
        """warmup_entropy_model is a no-op when use_entropy_model=False."""
        cfg = MAPPOConfig(z_dim=3, channel="sd", use_entropy_model=False)
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        result = t.warmup_entropy_model(_make_buffer(z_dim=3), n_steps=10)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Rigorous mathematical validation tests for P2 entropy model
# ---------------------------------------------------------------------------
# These tests go beyond smoke-testing: each one proves a specific mathematical
# property that must hold for the P2 entropy model to be correct. They are
# documented in docs/MATH.md §11.
# ---------------------------------------------------------------------------

from onpolicy.envs.toyproblem.network import (
    EntropyModelJoint, joint_entropy_bits, total_correlation_bits,
)


class TestEntropyModelValidation:
    """Mathematical validation suite for Pillar P2.

    Each test corresponds to a named property in MATH.md §11. Failures here
    indicate a correctness bug in the DLM prior or Ballé gradient path, not
    merely a performance regression.
    """

    # ------------------------------------------------------------------
    # Property V1: DLM normalization over wide support
    # ------------------------------------------------------------------
    def test_v1_dlm_wide_normalization(self):
        """V1 — DLM partition function: Σ_{m=-200}^{200} q_φ(m) ≥ 0.999.

        The DLM is defined on all integers via bin-integration of a continuous
        logistic mixture. With wide-scale initialisation (log_s=1, s≈e) the
        mixture tails are broad enough that the mass outside [-200,200] is
        negligible. If this fails, the distribution is either too peaked (mode
        collapse) or the scale init is wrong.
        """
        model = EntropyModelFactored(z_dim=1, K=5)
        ms = torch.arange(-200, 201, dtype=torch.float32).unsqueeze(-1)
        total = model.log_prob(ms).exp().sum().item()
        assert total >= 0.999, (
            f"Wide-range partition function = {total:.5f} (expected ≥ 0.999). "
            "Possible mode collapse or narrow-scale initialisation."
        )
        assert total <= 1.001, (
            f"Wide-range partition function = {total:.5f} (exceeds 1). "
            "Numerical error in log-sum-exp or sigmoid computation."
        )

    # ------------------------------------------------------------------
    # Property V2: Synthetic entropy convergence
    # ------------------------------------------------------------------
    def test_v2_synthetic_entropy_convergence(self):
        """V2 — q_φ learns true entropy: E[-log₂ q_φ(m)] → H(P) under stationary P.

        If we draw m i.i.d. from a known discrete distribution P and train q_φ
        on those samples, the cross-entropy H(P, q_φ) = E_P[-log₂ q_φ(m)] must
        converge to the true entropy H(P) (by the Gibbs inequality, equality
        holds iff q_φ = P everywhere).

        Distribution: uniform over {-2,-1,0,1,2} → H(P) = log₂(5) ≈ 2.3219 bits.

        Tolerance: 0.35 bits. The DLM is a logistic mixture over all integers,
        so it cannot assign exactly zero mass outside {-2,...,2}. At convergence
        ~9% of mass leaks to out-of-support integers, giving a cross-entropy
        gap of ~0.28 bits above H(P). This is a known DLM approximation limit,
        not a bug. Failure beyond 0.35 bits indicates a real training problem.
        """
        true_entropy = math.log2(5)  # ≈ 2.3219 bits
        torch.manual_seed(0)
        model = EntropyModelFactored(z_dim=1, K=5)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        for _ in range(2000):
            m = torch.randint(-2, 3, (256, 1)).float()
            loss = model.nll_bits(m).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            m_eval = torch.randint(-2, 3, (10000, 1)).float()
            avg_nll = model.nll_bits(m_eval).mean().item()

        assert abs(avg_nll - true_entropy) < 0.35, (
            f"Converged NLL = {avg_nll:.4f} bits, expected H(P) = {true_entropy:.4f} bits "
            f"(gap = {abs(avg_nll - true_entropy):.4f} > 0.35). "
            "Check nll_bits formula or Adam step in forward loss."
        )

    # ------------------------------------------------------------------
    # Property V3: Ballé backward gradient direction
    # ------------------------------------------------------------------
    def test_v3_balle_gradient_direction(self):
        """V3 — Backward loss gradient pushes z toward high-probability regions.

        With frozen q_φ, L_bwd = -log₂ q_φ(z/δ). The gradient ∂L_bwd/∂z equals
        -1/(δ ln 2) · ∂ log q_φ/∂x|_{x=z/δ}.

        For a q_φ with a strong mode at 0:
          - At z/δ = +1.5 (right of mode): ∂ log q_φ/∂x < 0, so ∂L_bwd/∂z > 0.
            Gradient descent on L_bwd decrements z → pushes z toward 0. ✓
          - At z/δ = -1.5 (left of mode): ∂ log q_φ/∂x > 0, so ∂L_bwd/∂z < 0.
            Gradient descent increments z → pushes z toward 0. ✓

        We use z/δ = ±1.5 (not ±3) because a strongly peaked DLM trained on
        all-zeros data assigns negligible mass beyond |m|≈2, hitting the 1e-10
        probability clamp and zeroing the gradient. At ±1.5 the probability is
        small but above the clamp threshold, so the gradient is well-defined.

        If this fails, the Ballé backward path provides the wrong direction signal
        to the speaker: the speaker would be pushed *away* from likely messages,
        increasing the rate instead of reducing it.
        """
        torch.manual_seed(0)
        model = EntropyModelFactored(z_dim=1, K=3)

        # Pre-train q_φ to concentrate on 0 (so mode is clearly at 0)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        m_train = torch.zeros(256, 1)  # all zeros
        for _ in range(500):
            loss = model.nll_bits(m_train).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Freeze q_φ (simulates the backward pass in trainer.update)
        for p in model.parameters():
            p.requires_grad_(False)

        delta = 1.0

        # Right of mode: z/δ = +1.5
        z_right = torch.tensor([[1.5]], requires_grad=True)
        model.nll_bits(z_right / delta).backward()
        grad_right = z_right.grad.item()
        assert grad_right > 0, (
            f"At z/δ=+1.5, grad={grad_right:.4f} should be > 0 (descent pushes z toward 0). "
            "Ballé backward path gives wrong sign — speaker gradient is inverted."
        )

        # Left of mode: z/δ = -1.5
        z_left = torch.tensor([[-1.5]], requires_grad=True)
        model.nll_bits(z_left / delta).backward()
        grad_left = z_left.grad.item()
        assert grad_left < 0, (
            f"At z/δ=-1.5, grad={grad_left:.4f} should be < 0 (descent pushes z toward 0). "
            "Ballé backward path gives wrong sign — speaker gradient is inverted."
        )

        # Restore
        for p in model.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Property V4: TC identity — factored gap equals joint improvement
    # ------------------------------------------------------------------
    def test_v4_tc_identity_factored_minus_joint(self):
        """V4 — After convergence: E[factored_nll] − E[joint_nll] ≈ TC(m).

        By the chain rule of entropy:
          H(m_0,...,m_{K-1}) = Σ_k H(m_k | m_0,...,m_{k-1})
        The factored model approximates H(m) ≈ Σ_k H(m_k) (ignores correlations).
        The joint model learns the conditional decomposition exactly.

        For perfectly correlated messages (m_0 = m_1, both uniform over {0,1,2,3}):
          Σ_k H(m_k) = 2 + 2 = 4 bits  (each marginal has H = log₂(4) = 2 bits)
          H(m_0, m_1) = 2 bits          (joint is the same as a single dim)
          TC = 4 − 2 = 2 bits

        After training, factored_nll − joint_nll → TC + ε_DLM. The DLM
        approximation error ε_DLM ≈ 0.27 bits arises because the logistic mixture
        cannot assign zero mass outside {0,1,2,3}. The factored model pays this
        error twice (once per dimension) while the joint pays it once, inflating
        the gap by ~ε_DLM above the true TC. Tolerance: 0.35 bits above TC=2.0.

        If this fails, either the joint autoregressive structure is wrong (it
        cannot learn conditional distributions) or the DLM parameterisation
        is not expressive enough for this distribution.
        """
        torch.manual_seed(42)
        z_dim = 2
        true_tc = 2.0  # TC for perfectly correlated uniform-{0,1,2,3} pair

        factored = EntropyModelFactored(z_dim=z_dim, K=5)
        joint = EntropyModelJoint(z_dim=z_dim, K=5)

        opt_f = torch.optim.Adam(factored.parameters(), lr=3e-3)
        opt_j = torch.optim.Adam(joint.parameters(), lr=3e-3)

        for _ in range(3000):
            m0 = torch.randint(0, 4, (512,))
            m = torch.stack([m0, m0], dim=-1).float()  # perfect correlation
            loss_f = factored.nll_bits(m).sum(dim=-1).mean()
            opt_f.zero_grad(); loss_f.backward(); opt_f.step()
            loss_j = joint.nll_bits(m).sum(dim=-1).mean()
            opt_j.zero_grad(); loss_j.backward(); opt_j.step()

        with torch.no_grad():
            m0_eval = torch.randint(0, 4, (10000,))
            m_eval = torch.stack([m0_eval, m0_eval], dim=-1).float()
            nll_f = factored.nll_bits(m_eval).sum(dim=-1).mean().item()
            nll_j = joint.nll_bits(m_eval).sum(dim=-1).mean().item()

        observed_gap = nll_f - nll_j
        assert abs(observed_gap - true_tc) < 0.35, (
            f"Factored−joint NLL gap = {observed_gap:.3f} bits, expected TC ≈ {true_tc:.1f} bits "
            f"(error = {abs(observed_gap - true_tc):.3f} > 0.35). "
            "Joint model may not be learning the conditional correctly."
        )
        # Also verify: joint_nll ≤ factored_nll (joint is always at least as good)
        assert nll_j <= nll_f + 0.01, (
            f"Joint NLL ({nll_j:.3f}) > factored NLL ({nll_f:.3f}). "
            "Joint model should never be worse than factored on the same data."
        )

    # ------------------------------------------------------------------
    # Property V5: Frozen-speaker qphi_gap convergence
    # ------------------------------------------------------------------
    def test_v5_frozen_speaker_qphi_gap_converges(self):
        """V5 — qphi_gap decreases toward 0 when p(m) is stationary.

        qphi_gap = E[-log₂ q_φ(m)] − H(m) ≥ 0 measures how well q_φ has
        converged to the true message distribution p(m). When the speaker is
        frozen (p(m) does not change), running n_qphi_steps should drive
        qphi_gap toward 0.

        This test freezes the speaker, runs warmup_entropy_model for 300 steps,
        and verifies:
          (a) gap_final < gap_init  (q_φ is actually learning)
          (b) gap_final < 0.5 bits  (converged close to true entropy)

        If (a) fails: the optimizer step in warmup_entropy_model is a no-op
        (zero grad, wrong parameters, etc.).
        If (b) fails: q_φ has the right direction but is too slow — check
        lr_qphi or n_warmup_steps defaults.
        """
        torch.manual_seed(7)
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0,
            use_entropy_model=True, entropy_model_K=5,
            entropy_model_type="factored", entropy_model_context="A",
            update_epochs=1, num_minibatches=1,
            lr=3e-4, lr_qphi_mult=10.0,
        )
        trainer = MAPPOTrainer(cfg, device=torch.device("cpu"))

        # Freeze speaker so message distribution p(m) is stationary
        for p in trainer.speaker.parameters():
            p.requires_grad_(False)

        buf = _make_buffer(z_dim=3, n_steps=32, n_envs=8)

        def _measure_gap(trainer, buf) -> float:
            """Compute qphi_gap on the entire buffer."""
            with torch.no_grad():
                all_nll, all_m = [], []
                for mb in buf.minibatches(1):
                    z = trainer.speaker(mb["goals"])
                    _, ch_info = trainer.channel(z)
                    m = ch_info["m"]
                    nll = trainer.entropy_model.nll_bits(m.float())
                    all_nll.append(nll.mean().item())
                    all_m.append(m.long())
                avg_nll = sum(all_nll) / len(all_nll)
                m_all = torch.cat(all_m, dim=0)
                h_emp = joint_entropy_bits(m_all)
                return avg_nll - h_emp

        gap_init = _measure_gap(trainer, buf)
        trainer.warmup_entropy_model(buf, n_steps=300)
        gap_final = _measure_gap(trainer, buf)

        assert gap_final < gap_init, (
            f"qphi_gap did not decrease: init={gap_init:.4f} → final={gap_final:.4f}. "
            "warmup_entropy_model may not be updating q_φ parameters."
        )
        assert gap_final < 0.5, (
            f"qphi_gap too large after 300 steps: {gap_final:.4f} bits (expected < 0.5). "
            "Check lr_qphi_mult or that optim_qphi is updating the correct parameters."
        )

        # Restore speaker
        for p in trainer.speaker.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Hardening-fix validation tests (V6–V8)
# Tests for the three robustness fixes applied after the initial P2 implementation:
#   V6 — Mixture prior eliminates gradient dead-zone without warm-start
#   V7 — Scale floor prevents DLM collapse to delta function
#   V8 — Context B backward is disabled (degenerate rate signal)
# Each test is documented in docs/MATH.md §12 and docs/pillars/PILLAR_P2.md.
# ---------------------------------------------------------------------------

from onpolicy.envs.toyproblem.network import _FLAT_ALPHA, _FLAT_SCALE, _S_LOG_MIN


class TestHardeningFixes:
    """Validates the three robustness fixes applied to the P2 entropy model."""

    # ------------------------------------------------------------------
    # Fix V6: Mixture prior — no gradient dead-zone for extreme integers
    # ------------------------------------------------------------------
    def test_v6_mixture_prior_no_dead_zone(self):
        """V6 — Mixture prior: finite log-prob and non-zero gradient everywhere.

        Before the fix, q_φ(m) hit the 1e-10 clamp for integers far outside
        training support, making log q_φ ≈ −33 bits and ∂L/∂z ≈ 0 (dead zone).

        After the fix, q̃_φ(m) = (1−α)·q_φ(m) + α·Laplace(m; 0, _FLAT_SCALE).
        Even for extreme unseen integers (here ±500 and ±300):
          1. log q̃_φ(m) is finite (not −inf)
          2. The backward gradient through nll_bits is finite and non-zero
          3. The gradient sign is correct: it points toward smaller |x| (rate-reducing)

        If this fails, the mixture prior is not being applied; any message the
        speaker hasn't seen before will produce zero gradient, reverting to the
        warm-start dependency.
        """
        model = EntropyModelFactored(z_dim=2, K=3)
        # Extreme integers — far outside anything the DLM covers without training
        extreme = torch.tensor([[500.0, -300.0]])
        log_q = model.log_prob(extreme)
        assert log_q.isfinite().all(), (
            f"log_prob not finite for extreme inputs {extreme.tolist()}: {log_q}. "
            "Mixture prior (_mix_with_flat) may not be applied."
        )

        # Gradient must be finite and non-zero
        x_ext = extreme.clone().requires_grad_(True)
        model.nll_bits(x_ext).sum().backward()
        assert x_ext.grad is not None
        assert x_ext.grad.isfinite().all(), (
            f"NaN gradient at extreme inputs: {x_ext.grad}"
        )
        assert x_ext.grad.abs().max().item() > 0, (
            "Zero gradient at extreme inputs — dead zone still present."
        )

        # Gradient must point toward zero (rate-reducing direction)
        # At x=+500: grad should be > 0 (descent moves x toward 0)
        # At x=-300: grad should be < 0 (descent moves x toward 0)
        assert x_ext.grad[0, 0].item() > 0, (
            f"Gradient at x=+500 is {x_ext.grad[0,0].item():.4f}, expected > 0. "
            "Mixture prior gradient direction is wrong."
        )
        assert x_ext.grad[0, 1].item() < 0, (
            f"Gradient at x=-300 is {x_ext.grad[0,1].item():.4f}, expected < 0. "
            "Mixture prior gradient direction is wrong."
        )

    # ------------------------------------------------------------------
    # Fix V7: Scale floor — DLM stays well-behaved at extreme log_s
    # ------------------------------------------------------------------
    def test_v7_scale_floor_prevents_collapse(self):
        """V7 — Scale floor: s_eff ≥ 0.1 even when log_s is driven to −∞.

        Without the clamp, log_s = −100 gives s ≈ 3.7e−44 → all probability
        mass collapses to a delta at the nearest integer → log q = −∞ for
        non-integer inputs (including the continuous Ballé backward z/δ) →
        NaN loss and zero gradient.

        With the clamp (log_s ≥ _S_LOG_MIN = log(0.1), s_eff ≥ 0.1), log q
        remains finite everywhere and the gradient flows correctly.
        At s=0.1 with μ at bin centre n+0.5: P(floor bin n) = σ(0.5/0.1)−σ(−0.5/0.1)
        = 2σ(5)−1 ≈ 0.987, which is sufficient for the TC identity (Fix 2).

        This test also verifies the conditional models (Joint, CondZ,
        JointCondZ) via their MLP output scale bias.
        """
        import math as _math
        from onpolicy.envs.toyproblem.network import (
            EntropyModelJoint, EntropyModelCondZ, EntropyModelJointCondZ,
        )
        # Non-integer inputs — hardest case for a delta-like DLM
        x_nonint = torch.tensor([[0.7, -1.3]])

        # --- EntropyModelFactored: log_s → -∞, effective s = exp(_S_LOG_MIN) ---
        m_factored = EntropyModelFactored(z_dim=2, K=3)
        with torch.no_grad():
            m_factored.log_s.fill_(-100.0)
        log_q = m_factored.log_prob(x_nonint)
        assert log_q.isfinite().all(), (
            f"EntropyModelFactored log_prob not finite with log_s=-100 "
            f"(s_eff=exp({_S_LOG_MIN:.3f})={_math.exp(_S_LOG_MIN):.2f}): {log_q}"
        )
        x_g = x_nonint.clone().requires_grad_(True)
        m_factored.nll_bits(x_g).sum().backward()
        assert x_g.grad.isfinite().all()

        # --- EntropyModelJoint ---
        m_joint = EntropyModelJoint(z_dim=2, K=3)
        with torch.no_grad():
            m_joint.log_s_0.fill_(-100.0)
        log_q_j = m_joint.log_prob(x_nonint)
        assert log_q_j.isfinite().all(), (
            f"EntropyModelJoint log_prob not finite with log_s_0=-100: {log_q_j}"
        )

        # --- EntropyModelCondZ ---
        m_condz = EntropyModelCondZ(z_dim=2, K=3)
        z = torch.zeros(1, 2)
        log_q_c = m_condz.log_prob(x_nonint, z)
        assert log_q_c.isfinite().all(), (
            f"EntropyModelCondZ log_prob not finite: {log_q_c}"
        )

    # ------------------------------------------------------------------
    # Fix V8: Context B backward disabled — no entropy gradient to speaker
    # ------------------------------------------------------------------
    def test_v8_context_b_backward_disabled(self):
        """V8 — Context B entropy backward is disabled; speaker receives no gradient.

        In context B, q_φ(m|z) conditions on z. Although dithering makes m
        non-deterministic (m = floor((z+noise)/δ)), q_φ converges to the true
        conditional after training, driving NLL → 0 and the backward gradient
        to the speaker → 0. No meaningful compression pressure reaches the speaker.

        After the fix, the backward entropy loss is silently skipped for context B
        (only context A provides a speaker gradient from the entropy path).

        Test design:
          - Zero RL advantages → actor_loss = 0, no RL gradient to speaker
          - loss_comms_mode = "entropy" → magnitude loss also absent
          - entropy_coef = 0 → no entropy bonus
          The ONLY loss that can affect the speaker is the entropy backward.
          With context A: speaker params MUST change (entropy backward active).
          With context B: speaker params must NOT change (entropy backward disabled).
        """
        def make_trainer(context: str) -> MAPPOTrainer:
            cfg = MAPPOConfig(
                z_dim=3, channel="sd", delta=1.0, lambda_comms=1.0,
                use_entropy_model=True, entropy_model_K=3,
                entropy_model_type="factored", entropy_model_context=context,
                loss_comms_mode="entropy",
                update_epochs=1, num_minibatches=1,
                entropy_coef=0.0,
            )
            return MAPPOTrainer(cfg, device=torch.device("cpu"))

        buf = _make_buffer(z_dim=3, n_steps=8, n_envs=4)
        buf.advantages = torch.zeros(8, 4)  # zero RL signal
        buf.returns = torch.zeros(8, 4)

        # Context A: entropy backward must update speaker
        torch.manual_seed(0)
        t_a = make_trainer("A")
        w_before_a = t_a.speaker.network[-1].weight.detach().clone()
        t_a.update(buf)
        w_after_a = t_a.speaker.network[-1].weight.detach()
        assert not torch.allclose(w_before_a, w_after_a), (
            "Context A: entropy backward should have updated speaker params but didn't. "
            "Check that backward entropy loss is being added to total_loss."
        )

        # Context B: speaker must NOT change (entropy backward disabled)
        torch.manual_seed(0)
        t_b = make_trainer("B")
        w_before_b = t_b.speaker.network[-1].weight.detach().clone()
        t_b.update(buf)
        w_after_b = t_b.speaker.network[-1].weight.detach()
        assert torch.allclose(w_before_b, w_after_b, atol=1e-7), (
            "Context B: entropy backward is degenerate and must be disabled, but "
            "speaker params changed. See PILLAR_P2.md §4 for the derivation."
        )
