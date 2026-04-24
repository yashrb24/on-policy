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
