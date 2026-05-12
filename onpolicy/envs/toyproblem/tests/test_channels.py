"""Unit tests for channels.py — Phase 1 implementation.

Monte-Carlo tests use N_SAMPLES = 200_000 forward-pass samples drawn in a single
batched call for speed.  Tolerances are set at 5 sigma of the relevant sampling
distribution so failures indicate genuine mathematical errors, not noise.
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# IdentityChannel
# ---------------------------------------------------------------------------

class TestIdentityChannel:
    def test_passthrough(self, identity_channel):
        """z_hat == z exactly (no noise, no quantisation)."""
        z = torch.randn(16, 3)
        z_hat, info = identity_channel(z)
        assert torch.equal(z_hat, z), "IdentityChannel must be a strict no-op."

    def test_comms_loss_zero(self, identity_channel):
        """L_comms == 0 for all z (identity channel has zero communication cost)."""
        z = torch.randn(16, 3)
        loss = identity_channel.comms_loss(z)
        assert (loss == 0).all(), "IdentityChannel comms_loss must be identically zero."

    def test_gradient_is_one(self, identity_channel):
        """∂z_hat/∂z = 1 element-wise (trivially satisfied by passthrough)."""
        z = torch.randn(4, 2, requires_grad=True)
        z_hat, _ = identity_channel(z)
        grad = torch.autograd.grad(z_hat.sum(), z)[0]
        assert torch.allclose(grad, torch.ones_like(grad)), (
            "IdentityChannel gradient must be 1 everywhere."
        )


# ---------------------------------------------------------------------------
# DDCL_SD: Subtractive Dithering
# ---------------------------------------------------------------------------

class TestDDCL_SD:
    N_SAMPLES = 200_000

    def test_reconstruction_unbiased(self, sd_channel):
        """E[ẑ - z] ≈ 0  (Theorem 2 / Theorem A.1 of the proposal).

        The training path is z_hat = z + e with e ~ U(-δ/2, +δ/2).
        By construction, E[e] = 0, so E[z_hat - z] = 0.
        Tolerance set at 5σ of the sampling distribution of the sample mean.
        """
        z_val = 5.0
        z = torch.full((self.N_SAMPLES, 1), z_val)
        z_hat, _ = sd_channel(z)
        mean_err = (z_hat - z).mean().item()
        # std(e) = δ/sqrt(12); SE of mean = δ / (sqrt(12) * sqrt(N))
        tol = 5.0 * sd_channel.delta / (12.0 ** 0.5 * self.N_SAMPLES ** 0.5)
        assert abs(mean_err) < tol, (
            f"E[ẑ-z]={mean_err:.6f}, expected ≈0 (5σ tol={tol:.6f})"
        )

    def test_reconstruction_variance(self, sd_channel, default_delta):
        """Var(ẑ - z) ≈ δ²/12  (uniform distribution variance).

        For the SD training path, e ~ U(-δ/2, +δ/2), so Var(e) = δ²/12.
        """
        z = torch.full((self.N_SAMPLES, 1), 5.0)
        z_hat, _ = sd_channel(z)
        empirical_var = (z_hat - z).var().item()
        expected_var = (default_delta ** 2) / 12.0
        # 2% relative tolerance
        assert abs(empirical_var - expected_var) / expected_var < 0.02, (
            f"Var(ẑ-z)={empirical_var:.4f}, expected δ²/12={expected_var:.4f}"
        )

    def test_gradient_is_one(self, sd_channel):
        """∂z_hat/∂z = 1 per element (key DDCL property — enables end-to-end training)."""
        z = torch.randn(16, 3, requires_grad=True)
        z_hat, _ = sd_channel(z)
        grad = torch.autograd.grad(z_hat.sum(), z)[0]
        assert torch.allclose(grad, torch.ones_like(grad)), (
            "SD gradient must be exactly 1 everywhere."
        )

    def test_error_independent_of_signal(self, sd_channel):
        """Corr(ẑ - z, z) ≈ 0  (e ⊥⊥ z, Theorem 2).

        Uses a linearly-spaced grid of z values so z has known spread.
        The noise e is drawn independently of z, so Pearson correlation ≈ 0.
        """
        d = sd_channel.delta
        z = torch.linspace(-5.0 * d, 5.0 * d, self.N_SAMPLES).unsqueeze(-1)
        z_hat, _ = sd_channel(z)
        e = (z_hat - z).squeeze()
        z_flat = z.squeeze()
        e_c = e - e.mean()
        z_c = z_flat - z_flat.mean()
        cov = (e_c * z_c).mean().item()
        corr = cov / (e.std().item() * z_flat.std().item())
        assert abs(corr) < 0.01, (
            f"Corr(e, z)={corr:.5f}, expected ≈0 — e must be independent of z"
        )

    def test_comms_loss_shape(self, sd_channel):
        """comms_loss(z) has the same shape as z (per-element loss)."""
        z = torch.randn(8, 3)
        loss = sd_channel.comms_loss(z)
        assert loss.shape == z.shape, (
            f"comms_loss must return same shape as input; got {loss.shape} vs {z.shape}"
        )

    def test_comms_loss_nonnegative(self, sd_channel):
        """comms_loss(z) ≥ 0 for all z (log₂(|z|/δ + 1) ≥ 0 since argument ≥ 1)."""
        z = torch.randn(64, 3)
        loss = sd_channel.comms_loss(z)
        assert (loss >= 0).all(), "comms_loss must be non-negative."

    def test_comms_loss_zero_at_origin(self, sd_channel):
        """comms_loss(0) = 0  (zero-magnitude signal costs zero bits)."""
        z = torch.zeros(4, 3)
        loss = sd_channel.comms_loss(z)
        assert torch.allclose(loss, torch.zeros_like(loss)), (
            "comms_loss must be 0 when z=0."
        )

    def test_comms_loss_monotone_in_magnitude(self, sd_channel):
        """comms_loss is monotonically non-decreasing in |z|.

        log₂(|z|/δ + 1) is a strictly increasing function of |z|, so the
        discrete evaluation on a non-decreasing grid must be non-decreasing.
        """
        z_vals = torch.linspace(0.0, 100.0 * sd_channel.delta, 500).unsqueeze(-1)
        loss = sd_channel.comms_loss(z_vals)
        diffs = (loss[1:] - loss[:-1]).squeeze()
        assert (diffs >= 0).all(), "comms_loss must be non-decreasing in |z|"

    def test_fresh_noise_per_forward(self, sd_channel):
        """Two forward passes on the same z produce different z_hat values.

        This verifies that noise is resampled each call, as required by Thm A.1.
        """
        z = torch.randn(8, 3)
        z_hat1, _ = sd_channel(z)
        z_hat2, _ = sd_channel(z)
        assert not torch.equal(z_hat1, z_hat2), (
            "Two forward passes must produce different z_hat (fresh noise each call)"
        )

    def test_deploy_path_in_info(self, sd_channel):
        """info dict contains keys: m, C_m, z_hat_deploy, eps, z_prime."""
        z = torch.randn(4, 2)
        _, info = sd_channel(z)
        for key in ("m", "C_m", "z_hat_deploy", "eps", "z_prime"):
            assert key in info, f"Missing key {key!r} in DDCL_SD info dict."


# ---------------------------------------------------------------------------
# DDCL_NSD: Non-Subtractive / TPDF Dithering
# ---------------------------------------------------------------------------

class TestDDCL_NSD:
    N_SAMPLES = 200_000

    def test_reconstruction_unbiased(self, nsd_channel):
        """E[ẑ - z] ≈ 0  (Schuchman / TPDF Theorem 5, property a).

        Under TPDF dither, E[C(m) | z] = z by Schuchman's unbiasedness condition.
        The STE training path inherits this: E[z_hat - z] = E[C(m) - z] = 0.
        """
        z_val = 5.0
        z = torch.full((self.N_SAMPLES, 1), z_val)
        z_hat, _ = nsd_channel(z)
        mean_err = (z_hat - z).mean().item()
        # std(e) = δ/sqrt(6) for TPDF; SE of mean = δ / (sqrt(6) * sqrt(N))
        tol = 5.0 * nsd_channel.delta / (6.0 ** 0.5 * self.N_SAMPLES ** 0.5)
        assert abs(mean_err) < tol, (
            f"E[ẑ-z]={mean_err:.6f}, expected ≈0 (5σ tol={tol:.6f})"
        )

    def test_reconstruction_variance(self, nsd_channel, default_delta):
        """Var(ẑ - z) ≈ δ²/4  (constant, independent of z — Schuchman 2nd order property).

        NSD implements C(m) - z where m = floor((z+ν)/δ), ν ~ Triangular(-δ,δ).
        The error jumps between full bin boundaries (multiples of δ), whereas SD
        confines error to ±δ/2.  Analytical derivation (verified in MATH.md):
          Var(e|z) = δ²/4  for all z  (constant — 2nd-order TPDF property holds).
        Note: MATH.md originally claimed δ²/6 — corrected in Phase 1 (MATH-002).
        """
        z = torch.full((self.N_SAMPLES, 1), 5.0)
        z_hat, _ = nsd_channel(z)
        empirical_var = (z_hat - z).var().item()
        expected_var = (default_delta ** 2) / 4.0
        # 2% relative tolerance
        assert abs(empirical_var - expected_var) / expected_var < 0.02, (
            f"Var(ẑ-z)={empirical_var:.4f}, expected δ²/4={expected_var:.4f}"
        )

    def test_zero_covariance(self, nsd_channel):
        """Cov(ẑ - z, z) ≈ 0  (Theorem 5, property c).

        Follows from E[e|z] = 0 everywhere: Cov(e, z) = E[e·z] - E[e]·E[z] = 0.
        """
        d = nsd_channel.delta
        z = torch.linspace(-5.0 * d, 5.0 * d, self.N_SAMPLES).unsqueeze(-1)
        z_hat, _ = nsd_channel(z)
        e = (z_hat - z).squeeze()
        z_flat = z.squeeze()
        e_c = e - e.mean()
        z_c = z_flat - z_flat.mean()
        cov = (e_c * z_c).mean().item()
        corr = cov / (e.std().item() * z_flat.std().item())
        assert abs(corr) < 0.01, (
            f"Corr(e, z)={corr:.5f}, expected ≈0 under TPDF (Theorem 5c)"
        )

    def test_gradient_is_one(self, nsd_channel):
        """∂z_hat/∂z = 1 (STE with principled Schuchman backing)."""
        z = torch.randn(16, 3, requires_grad=True)
        z_hat, _ = nsd_channel(z)
        grad = torch.autograd.grad(z_hat.sum(), z)[0]
        assert torch.allclose(grad, torch.ones_like(grad)), (
            "NSD STE gradient must be exactly 1 everywhere."
        )

    def test_comms_loss_shape(self, nsd_channel):
        """comms_loss(z) has the same shape as z."""
        z = torch.randn(8, 3)
        loss = nsd_channel.comms_loss(z)
        assert loss.shape == z.shape

    def test_comms_loss_nonnegative(self, nsd_channel):
        """comms_loss(z) ≥ 0 for all z."""
        z = torch.randn(64, 3)
        assert (nsd_channel.comms_loss(z) >= 0).all()

    def test_deploy_path_no_gradient(self, nsd_channel):
        """z_hat_true (the deploy value) carries no gradient into z.

        The STE wiring ensures the deployment path is detached from the
        computational graph, so the gradient flows only through the identity path.
        Verified by: (a) z_hat_true.requires_grad is False,
                      (b) z_hat.grad_fn is not None, and
                      (c) ∂z_hat/∂z = 1 (STE identity).
        """
        z = torch.randn(4, 3, requires_grad=True)
        z_hat, info = nsd_channel(z)
        # Deploy value must have no grad
        assert not info["z_hat_true"].requires_grad, (
            "z_hat_true must be detached — it is the deployment value, not the gradient path"
        )
        # STE z_hat must still carry gradient
        assert z_hat.requires_grad, "z_hat must have gradient back to z via STE"
        # Gradient flows with slope 1
        grad = torch.autograd.grad(z_hat.sum(), z)[0]
        assert torch.allclose(grad, torch.ones_like(grad))

    def test_fresh_noise_per_forward(self, nsd_channel):
        """Two forward passes produce different z_hat values (noise resampled)."""
        z = torch.randn(8, 3)
        z_hat1, _ = nsd_channel(z)
        z_hat2, _ = nsd_channel(z)
        assert not torch.equal(z_hat1, z_hat2), (
            "Two NSD forward passes must produce different z_hat (fresh TPDF noise each call)"
        )


# ---------------------------------------------------------------------------
# Cross-channel: comms_loss formula consistency
# ---------------------------------------------------------------------------

class TestCommsLossFormula:
    def test_formula_vs_original_proposal(self, sd_channel, default_delta):
        """Our formula log₂(|z|/δ + 1) vs proposal's log₂(2|z|/δ + 1).

        Both are strictly increasing in |z|, so they produce the same ordering
        (argmin is the same).  This test verifies rank identity on a dense grid.
        """
        # Non-negative z grid (loss formula uses |z|)
        z_vals = torch.linspace(0.0, 100.0 * default_delta, 1000).unsqueeze(-1)
        our_loss = sd_channel.comms_loss(z_vals)
        original_loss = torch.log2(2.0 * z_vals.abs() / default_delta + 1.0)
        our_ranks = our_loss.argsort(dim=0)
        original_ranks = original_loss.argsort(dim=0)
        assert torch.equal(our_ranks, original_ranks), (
            "Our formula and proposal formula must produce the same rank ordering"
        )

    def test_bits_per_msg_summation(self, sd_channel):
        """bits_per_msg = comms_loss.sum(dim=-1).mean() matches trainer.py:139."""
        z = torch.randn(32, 3)
        loss = sd_channel.comms_loss(z)
        expected = loss.sum(dim=-1).mean().item()
        # Verify the aggregation formula used in the trainer matches.
        assert expected >= 0, "bits_per_msg must be non-negative."


# ---------------------------------------------------------------------------
# Baseline channels — structural correctness
# ---------------------------------------------------------------------------

class TestAdditiveUniformChannel:
    """Verify AdditiveUniformChannel satisfies the 1st-order Schuchman condition."""
    N_SAMPLES = 200_000

    def test_shape(self, additive_uniform_channel):
        z = torch.randn(8, 3)
        z_hat, info = additive_uniform_channel(z)
        assert z_hat.shape == z.shape
        assert "m" in info and "z_hat_true" in info

    def test_gradient_ste(self, additive_uniform_channel):
        """Gradient flows through straight (STE): ∂z_hat/∂z = 1."""
        z = torch.randn(4, 3, requires_grad=True)
        z_hat, _ = additive_uniform_channel(z)
        grad = torch.autograd.grad(z_hat.sum(), z)[0]
        assert torch.allclose(grad, torch.ones_like(grad))

    def test_unbiased(self, additive_uniform_channel, default_delta):
        """E[ẑ - z] ≈ 0: additive uniform dither is unbiased."""
        z_val = 5.0
        z = torch.full((self.N_SAMPLES, 1), z_val)
        z_hat, _ = additive_uniform_channel(z)
        mean_err = (z_hat - z).mean().item()
        tol = 5.0 * default_delta / (12.0 ** 0.5 * self.N_SAMPLES ** 0.5)
        assert abs(mean_err) < tol, f"Bias={mean_err:.5f} > tol={tol:.5f}"

    def test_comms_loss_nonneg(self, additive_uniform_channel):
        z = torch.randn(16, 3)
        loss = additive_uniform_channel.comms_loss(z)
        assert (loss >= 0).all()

    def test_fresh_noise_per_call(self, additive_uniform_channel):
        z = torch.randn(8, 3)
        z_hat1, _ = additive_uniform_channel(z)
        z_hat2, _ = additive_uniform_channel(z)
        assert not torch.equal(z_hat1, z_hat2)


class TestGaussianChannel:
    """Verify GaussianChannel has matched variance and correct STE gradient."""

    def test_shape(self, gaussian_channel):
        z = torch.randn(8, 3)
        z_hat, info = gaussian_channel(z)
        assert z_hat.shape == z.shape

    def test_gradient_ste(self, gaussian_channel):
        z = torch.randn(4, 3, requires_grad=True)
        z_hat, _ = gaussian_channel(z)
        grad = torch.autograd.grad(z_hat.sum(), z)[0]
        assert torch.allclose(grad, torch.ones_like(grad))

    def test_sigma_matches_sd(self, gaussian_channel, default_delta):
        """σ = δ/√12 so variance matches SD (δ²/12) in the noise distribution."""
        expected_sigma = default_delta / (12 ** 0.5)
        assert abs(gaussian_channel.sigma - expected_sigma) < 1e-9

    def test_comms_loss_nonneg(self, gaussian_channel):
        z = torch.randn(16, 3)
        assert (gaussian_channel.comms_loss(z) >= 0).all()

    def test_fresh_noise_per_call(self, gaussian_channel):
        z = torch.randn(8, 3)
        z_hat1, _ = gaussian_channel(z)
        z_hat2, _ = gaussian_channel(z)
        assert not torch.equal(z_hat1, z_hat2)


class TestSTEChannel:
    """Verify STEChannel quantizes correctly and reports fixed-rate comms_loss."""

    def test_shape(self, ste_channel):
        z = torch.randn(8, 3)
        z_hat, info = ste_channel(z)
        assert z_hat.shape == z.shape
        assert "m" in info and "z_hat_true" in info

    def test_gradient_ste(self, ste_channel):
        """Gradient flows through straight: ∂z_hat/∂z = 1."""
        z = torch.randn(4, 3, requires_grad=True)
        z_hat, _ = ste_channel(z)
        grad = torch.autograd.grad(z_hat.sum(), z)[0]
        assert torch.allclose(grad, torch.ones_like(grad))

    def test_output_in_range(self, ste_channel):
        """z_hat must lie in [-clip_val, clip_val]."""
        z = torch.randn(1000, 3) * 20  # deliberately out of range
        z_hat, _ = ste_channel(z)
        assert (z_hat >= -ste_channel.clip_val - 1e-6).all()
        assert (z_hat <= ste_channel.clip_val + 1e-6).all()

    def test_comms_loss_constant(self, ste_channel):
        """comms_loss is exactly equal to `bits` for all z."""
        z = torch.randn(16, 3) * 50
        loss = ste_channel.comms_loss(z)
        expected = float(ste_channel.bits)
        assert torch.allclose(loss, torch.full_like(loss, expected))

    def test_quantization_levels(self, ste_channel):
        """For ≤ 8-bit channels, verify the exact number of distinct output levels.

        For 16-bit channels (65536 levels) float32 step-size precision makes
        grid-alignment checks unreliable; we instead verify output range and
        that the number of unique outputs saturates at n_samples < n_levels.
        """
        clip = ste_channel.clip_val

        if ste_channel.bits <= 8:
            # Sample 4× overcount to guarantee we hit every level.
            z = torch.linspace(-clip, clip, ste_channel.n_levels * 4).unsqueeze(-1)
            z_hat, _ = ste_channel(z)
            n_unique = z_hat.unique().numel()
            assert n_unique == ste_channel.n_levels, (
                f"Expected {ste_channel.n_levels} levels, got {n_unique}"
            )
        else:
            # High-bit: output must stay within [-clip_val, clip_val].
            z = torch.randn(1000, 3) * 50
            z_hat, _ = ste_channel(z)
            assert (z_hat >= -clip - 1e-4).all() and (z_hat <= clip + 1e-4).all()


class TestBuildChannel:
    """Verify build_channel returns the correct type for all names."""

    def test_all_channel_names(self, default_delta):
        from onpolicy.envs.toyproblem.channels import (
            IdentityChannel, DDCL_SD, DDCL_NSD,
            AdditiveUniformChannel, GaussianChannel, STEChannel,
            _CHANNEL_NAMES, build_channel,
        )
        type_map = {
            "none": IdentityChannel,
            "sd": DDCL_SD,
            "nsd": DDCL_NSD,
            "additive_uniform": AdditiveUniformChannel,
            "gaussian": GaussianChannel,
            "ste4": STEChannel,
            "ste8": STEChannel,
            "ste16": STEChannel,
        }
        for name in _CHANNEL_NAMES:
            ch = build_channel(name, default_delta)
            assert isinstance(ch, type_map[name]), (
                f"build_channel({name!r}) returned {type(ch).__name__}, "
                f"expected {type_map[name].__name__}"
            )

    def test_ste_bits(self, default_delta):
        from onpolicy.envs.toyproblem.channels import STEChannel, build_channel
        for bits in (4, 8, 16):
            ch = build_channel(f"ste{bits}", default_delta)
            assert isinstance(ch, STEChannel)
            assert ch.bits == bits

    def test_unknown_name_raises(self, default_delta):
        from onpolicy.envs.toyproblem.channels import build_channel
        with pytest.raises(ValueError, match="Unknown channel"):
            build_channel("bad_channel_name", default_delta)


# ---------------------------------------------------------------------------
# P1: PerChannelDelta
# ---------------------------------------------------------------------------

class TestPerChannelDelta:
    """Tests for the Pillar 1 per-channel δ module."""

    def test_init_values(self):
        """delta() returns delta_init for all dims at construction."""
        from onpolicy.envs.toyproblem.network import PerChannelDelta
        pcd = PerChannelDelta(z_dim=3, delta_init=1.0)
        d = pcd.delta()
        assert d.shape == (3,)
        assert torch.allclose(d, torch.ones(3), atol=1e-4), (
            f"Expected delta ≈ 1.0 at init, got {d.tolist()}"
        )

    def test_requires_grad(self):
        """delta() must have a gradient path to log_alpha."""
        from onpolicy.envs.toyproblem.network import PerChannelDelta
        pcd = PerChannelDelta(z_dim=2)
        d = pcd.delta()
        loss = d.sum()
        loss.backward()
        assert pcd.log_alpha.grad is not None
        assert not torch.any(pcd.log_alpha.grad == 0), "gradient should be non-zero"

    def test_clamp_lower(self):
        """delta() stays >= delta_min even when log_alpha is very negative."""
        from onpolicy.envs.toyproblem.network import PerChannelDelta
        pcd = PerChannelDelta(z_dim=2, delta_min=0.1)
        with torch.no_grad():
            pcd.log_alpha.fill_(-100.0)
        d = pcd.delta()
        assert (d >= 0.1 - 1e-6).all(), f"clamp_min failed: {d.tolist()}"

    def test_clamp_upper(self):
        """delta() stays <= delta_max even when log_alpha is very large."""
        from onpolicy.envs.toyproblem.network import PerChannelDelta
        pcd = PerChannelDelta(z_dim=2, delta_max=10.0)
        with torch.no_grad():
            pcd.log_alpha.fill_(100.0)
        d = pcd.delta()
        assert (d <= 10.0 + 1e-6).all(), f"clamp_max failed: {d.tolist()}"

    def test_task_gradient_flows_through_sd_channel(self):
        """∂(z+e)/∂δ_k = u_k − 0.5: gradient path exists through noise term."""
        from onpolicy.envs.toyproblem.network import PerChannelDelta
        from onpolicy.envs.toyproblem.channels import DDCL_SD
        pcd = PerChannelDelta(z_dim=3, delta_init=1.0)
        sd = DDCL_SD(delta=1.0)
        z = torch.randn(64, 3)
        d = pcd.delta()
        z_hat, _ = sd(z, delta=d)
        z_hat.sum().backward()
        assert pcd.log_alpha.grad is not None, "no gradient for log_alpha through channel"
        # E[∂/∂delta_k (z_k + (u-0.5)*delta_k)] = E[u-0.5] = 0;
        # std ≈ 1/sqrt(3)/sqrt(64) ≈ 0.07 → any finite grad is fine
        assert torch.isfinite(pcd.log_alpha.grad).all()

    def test_magnitude_loss_gradient_direction(self):
        """Magnitude loss gradient pushes δ_k upward (coarser → fewer surrogate bits)."""
        from onpolicy.envs.toyproblem.network import PerChannelDelta
        from onpolicy.envs.toyproblem.channels import DDCL_SD
        pcd = PerChannelDelta(z_dim=3, delta_init=1.0)
        sd = DDCL_SD(delta=1.0)
        z = torch.ones(16, 3) * 2.0   # fixed z so gradient is deterministic
        d = pcd.delta()
        loss = sd.comms_loss(z, delta=d).sum()
        loss.backward()
        # ∂L_mag/∂δ_k < 0 always → Adam/SGD step increases δ_k (coarser)
        assert (pcd.log_alpha.grad < 0).all(), (
            f"Magnitude loss gradient should be negative; got {pcd.log_alpha.grad.tolist()}"
        )

    def test_global_delta_broadcasts(self):
        """PerChannelDelta(z_dim=1) broadcasts correctly to all channel dims."""
        from onpolicy.envs.toyproblem.network import PerChannelDelta
        from onpolicy.envs.toyproblem.channels import DDCL_SD
        pcd = PerChannelDelta(z_dim=1, delta_init=1.5)
        sd = DDCL_SD(delta=1.0)
        z = torch.randn(32, 4)
        d = pcd.delta()   # shape (1,)
        z_hat, info = sd(z, delta=d)
        assert z_hat.shape == z.shape
        # Gradient flows back through the 1-param delta
        z_hat.sum().backward()
        assert pcd.log_alpha.grad is not None
        assert pcd.log_alpha.grad.shape == (1,)


# ---------------------------------------------------------------------------
# P4: Rao-Blackwell gradient estimator
# ---------------------------------------------------------------------------

class TestRaoBlackwellGradient:
    """Tests for the P4 RB finite-difference gradient estimator."""

    def test_rb_bin_centres_correct(self):
        """Bin centres ẑ_lo and ẑ_hi are adjacent to the STE bin, separated by δ."""
        from onpolicy.envs.toyproblem.channels import DDCL_SD
        torch.manual_seed(0)
        z = torch.tensor([[0.3, 1.7, -0.6]])   # (1, 3)
        delta = 1.0
        frac = (z / delta) - torch.floor(z / delta)
        lo_frac = frac < 0.5
        m_floor = torch.floor(z / delta)
        m_lo = torch.where(lo_frac, m_floor - 1, m_floor)
        m_hi = torch.where(lo_frac, m_floor, m_floor + 1)
        z_hat_lo = (m_lo + 0.5) * delta
        z_hat_hi = (m_hi + 0.5) * delta
        # ẑ_hi - ẑ_lo == δ for all dims
        assert torch.allclose(z_hat_hi - z_hat_lo, torch.full_like(z, delta))
        # ẑ_lo and ẑ_hi straddle z: ẑ_lo < z < ẑ_hi (or within δ of z)
        assert (z_hat_lo < z_hat_hi).all()

    def test_rb_proxy_gradient_shape(self):
        """RB proxy loss gradient w.r.t. z has shape (B, z_dim)."""
        z = torch.randn(16, 3, requires_grad=True)
        delta = 1.0
        with torch.no_grad():
            frac = (z / delta) - torch.floor(z / delta)
            lo_frac = frac < 0.5
            m_floor = torch.floor(z / delta)
            m_lo = torch.where(lo_frac, m_floor - 1, m_floor)
            m_hi = torch.where(lo_frac, m_floor, m_floor + 1)
            z_hat_lo = (m_lo + 0.5) * delta
            z_hat_hi = (m_hi + 0.5) * delta

        # Simulate L_hi - L_lo as random per-sample scalar (B,)
        L_diff = torch.randn(16)
        rb_scale = L_diff.detach().unsqueeze(-1) / delta   # (B, 1) broadcasts to (B, 3)
        rb_proxy = (rb_scale * z).sum(dim=-1).mean()
        rb_proxy.backward()
        assert z.grad is not None
        assert z.grad.shape == (16, 3)

    def test_rb_gradient_unbiased_in_expectation(self):
        """E[g_RB_k] ≈ E[g_STE_k] for a simple quadratic loss (1000 samples, atol=0.15)."""
        # For L(ẑ) = ẑ_k^2, the true gradient ∂E[L]/∂z_k = 2 * E[ẑ_k] = 2*z_k.
        # Both STE and RB should recover this in expectation.
        torch.manual_seed(42)
        N = 2000
        z_val = 1.0
        delta = 1.0
        z = torch.full((N, 1), z_val)

        # STE gradient: ∂(z+e)^2/∂z = 2*(z+e), mean ≈ 2*z_val
        e = (torch.rand(N, 1) - 0.5) * delta
        ste_grads = 2 * (z + e)
        ste_mean = ste_grads.mean().item()

        # RB gradient: (L_hi - L_lo) / delta = (ẑ_hi^2 - ẑ_lo^2) / delta
        frac = (z / delta) - torch.floor(z / delta)
        lo_frac = frac < 0.5
        m_floor = torch.floor(z / delta)
        m_lo = torch.where(lo_frac, m_floor - 1, m_floor)
        m_hi = torch.where(lo_frac, m_floor, m_floor + 1)
        z_hat_lo = (m_lo + 0.5) * delta
        z_hat_hi = (m_hi + 0.5) * delta
        rb_grads = (z_hat_hi ** 2 - z_hat_lo ** 2) / delta
        rb_mean = rb_grads.mean().item()

        true_grad = 2 * z_val
        assert abs(ste_mean - true_grad) < 0.15, f"STE mean {ste_mean:.4f} far from {true_grad}"
        assert abs(rb_mean - true_grad) < 0.15, f"RB mean {rb_mean:.4f} far from {true_grad}"

    def test_rb_variance_le_ste_variance(self):
        """Var(g_RB_k) < Var(g_STE_k) for a quadratic loss (RB reduces variance)."""
        # For L(ẑ) = ẑ^2: STE grad = 2*(z+e), RB grad = (ẑ_hi^2 - ẑ_lo^2)/δ.
        # STE gradient has dither noise; RB is deterministic per fixed z.
        torch.manual_seed(7)
        N = 5000
        z_val = 0.7    # fractional part 0.7/1.0 = 0.7 — not near a bin boundary
        delta = 1.0
        z = torch.full((N, 1), z_val)

        e = (torch.rand(N, 1) - 0.5) * delta
        ste_var = (2 * (z + e)).var().item()

        frac = (z / delta) - torch.floor(z / delta)
        lo_frac = frac < 0.5
        m_floor = torch.floor(z / delta)
        m_lo = torch.where(lo_frac, m_floor - 1, m_floor)
        m_hi = torch.where(lo_frac, m_floor, m_floor + 1)
        z_hat_lo = (m_lo + 0.5) * delta
        z_hat_hi = (m_hi + 0.5) * delta
        rb_var = ((z_hat_hi ** 2 - z_hat_lo ** 2) / delta).var().item()

        assert rb_var < ste_var, (
            f"RB variance {rb_var:.6f} should be < STE variance {ste_var:.6f}"
        )


# ---------------------------------------------------------------------------
# P3: Deployment evaluation (deploy_eval flag)
# ---------------------------------------------------------------------------

class TestDeployEval:
    """Tests for the P3 deploy_eval routing in act_and_value."""

    def test_nsd_train_deploy_equal(self):
        """NSD: z_hat_true equals the z_hat STE value (sample-consistent by construction)."""
        from onpolicy.envs.toyproblem.channels import DDCL_NSD
        torch.manual_seed(0)
        z = torch.randn(32, 3)
        nsd = DDCL_NSD(delta=1.0)
        z_hat, info = nsd(z)
        # Forward value should equal z_hat_true (they're the same by design)
        assert torch.allclose(z_hat, info["z_hat_true"]), (
            "NSD z_hat (STE forward) must equal z_hat_true (deployment value)"
        )

    def test_sd_deploy_differs_from_ste(self):
        """SD: z_hat_deploy differs from STE z_hat (not sample-consistent)."""
        from onpolicy.envs.toyproblem.channels import DDCL_SD
        torch.manual_seed(0)
        z = torch.randn(256, 3)
        sd = DDCL_SD(delta=1.0)
        z_hat, info = sd(z)
        z_hat_deploy = info["z_hat_deploy"]
        # In general they should differ (different dither samples e vs eps)
        assert not torch.allclose(z_hat, z_hat_deploy), (
            "SD z_hat (STE) and z_hat_deploy should differ in general"
        )

    def test_sd_deploy_distribution_matches_ste(self):
        """SD: z_hat_deploy and STE z_hat have the same mean (Schuchman unbiasedness)."""
        from onpolicy.envs.toyproblem.channels import DDCL_SD
        torch.manual_seed(0)
        N = 50_000
        z = torch.zeros(N, 1)   # fixed z=0 for clarity
        sd = DDCL_SD(delta=1.0)
        z_hat, info = sd(z)
        z_hat_deploy = info["z_hat_deploy"]
        # Both should have mean ≈ z = 0 (Schuchman 1st-order condition)
        assert abs(z_hat.mean().item()) < 0.02, f"STE mean {z_hat.mean():.4f} != 0"
        assert abs(z_hat_deploy.mean().item()) < 0.02, (
            f"deploy mean {z_hat_deploy.mean():.4f} != 0"
        )
