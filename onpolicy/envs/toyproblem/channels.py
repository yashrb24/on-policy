from __future__ import annotations

import math

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Shannon entropy of the goal distribution
# ---------------------------------------------------------------------------
# Goal probabilities from CommunicatingGoal_env._DEFAULT_GOAL_PROBS.
# H(G) = -Σ p_i log₂(p_i) ≈ 1.812 bits — the theoretical minimum bits to
# communicate goal identity using an optimal entropy code.
_GOAL_PROBS = (0.515, 0.258, 0.129, 0.064, 0.031, 0.003)
H_GOAL_BITS: float = -sum(p * math.log2(p) for p in _GOAL_PROBS)  # ≈ 1.812

# Per-goal optimal code length: -log₂(p_i) bits.
# Index i corresponds to goal i in _DEFAULT_GOALS.
GOAL_OPTIMAL_BITS: tuple[float, ...] = tuple(-math.log2(p) for p in _GOAL_PROBS)
# ≈ (0.957, 1.954, 2.954, 3.967, 5.011, 8.382)


def true_bits_from_m(m: torch.Tensor) -> torch.Tensor:
    """Empirical per-element bit cost from actual discrete message integers.

    Uses our project convention (no factor-of-2):
        b_k = log₂(|m_k| + 1)

    This is the non-differentiable empirical counterpart to ``comms_loss``,
    which uses ``log₂(|z_k|/δ + 1)`` as a differentiable surrogate.  The two
    agree in expectation when ``E[|m_k| | z_k] = |z_k|/δ`` (exact for SD).

    Parameters
    ----------
    m : Tensor  — integer messages, any shape

    Returns
    -------
    Tensor of same shape, dtype float32, with non-negative values.
    """
    return torch.log2(m.abs().float() + 1.0)


class IdentityChannel(nn.Module):
    """No-op passthrough so `--channel none` shares the same call shape as SD/NSD."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return z, {}

    def comms_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Returns zeros — no quantization penalty during training."""
        return torch.zeros_like(z)

    def effective_bits(self, z: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Information-content surrogate at *delta* resolution.

        Computes what it *would* cost to transmit z if quantized at bin width
        *delta* (default 1.0), using the DDCL Jensen surrogate:

            B_ref(z) = log₂(|z_k| / delta + 1)   per element k

        This measures the *information content* of the speaker's output, not
        what is actually transmitted (which is ``transmission_bits_per_elem``).
        Used for rate-distortion analysis only — not part of the training loss.
        """
        return torch.log2(z.abs() / delta + 1.0)

    def transmission_bits_per_elem(
        self, z: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Actual per-element transmission cost for the identity channel.

        The identity channel passes z as raw float32, so every element costs
        exactly 32 bits regardless of its magnitude:

            B_transmitted = 32 bits × z_dim

        This is in contrast to the DDCL variable-rate channels, which transmit
        a discrete integer m whose size depends on |z|/δ.
        """
        return torch.full_like(z, 32.0)


class DDCL_SD(nn.Module):
    """Subtractive-dithering DDCL (user's Implementation 1).

    Theorem A.1 collapses the real pipeline
        eps ~ U(-δ/2, δ/2);  z' = z + eps;  m = floor(z'/δ);
        C(m) = (m + 1/2)·δ;  z_hat_deploy = C(m) - eps
    into the distributional equivalent  z_hat = z + e,  e ~ U(-δ/2, δ/2),
    so for gradients we just add fresh uniform noise (∂ẑ/∂z = 1). The real
    pipeline runs detached alongside it for bitrate logging and deployment
    parity (in a real system sender and receiver share eps via a common RNG).
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        d = self.delta

        e = (torch.rand_like(z) - 0.5) * d
        z_hat = z + e

        with torch.no_grad():
            eps = (torch.rand_like(z) - 0.5) * d
            z_prime = z + eps
            m = torch.floor(z_prime / d)
            C_m = (m + 0.5) * d
            z_hat_deploy = C_m - eps

        info = {
            "m": m,
            "C_m": C_m,
            "z_prime": z_prime,
            "z_hat_deploy": z_hat_deploy,
            "eps": eps,
        }
        return z_hat, info

    def comms_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Per-element Jensen upper bound on expected bit length: log₂(|z|/δ + 1)."""
        return torch.log2(z.abs() / self.delta + 1)

    def transmission_bits_per_elem(
        self, z: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Empirical per-element bits from the actual discrete message m.

        Uses log₂(|m_k| + 1) per element (our signed-integer convention).

        Relationship to the surrogate ``comms_loss(z) = log₂(|z_k|/δ + 1)``:
        - For |z| ≫ δ: surrogate ≥ empirical (Jensen's inequality is tight).
        - At z = 0: surrogate = 0, but empirical ≈ 0.5 bits because dither
          always lands m in {-1, 0} with equal probability. The surrogate
          underestimates here due to E[|m||z=0] = 0.5 ≠ |z|/δ = 0.
        - At z = n·δ (bin boundary): surrogate overestimates because m splits
          evenly between bins n-1 and n, giving E[|m|] ≈ n - 0.5 < n = |z|/δ.
        """
        return true_bits_from_m(info["m"])


class DDCL_NSD(nn.Module):
    """Non-subtractive-dithering DDCL (user's Implementation 2).

    Triangular dither ν = u₁ + u₂, uᵢ ~ U(-δ/2, δ/2); receiver outputs the
    bin center (no ε subtraction, no shared RNG). Schuchman's theorem gives
    E[ẑ|z] = z exactly, so straight-through with slope 1 is an unbiased
    estimator of the expected gradient.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        d = self.delta

        u1 = (torch.rand_like(z) - 0.5) * d
        u2 = (torch.rand_like(z) - 0.5) * d
        nu = u1 + u2
        with torch.no_grad():
            m = torch.floor((z + nu) / d)
            z_hat_true = (m + 0.5) * d

        z_hat = z + (z_hat_true - z).detach()

        info = {"m": m, "nu": nu, "z_hat_true": z_hat_true}
        return z_hat, info

    def comms_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Same Jensen bound as SD — derivation carries over under NSD."""
        return torch.log2(z.abs() / self.delta + 1)

    def transmission_bits_per_elem(
        self, z: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Empirical per-element bits from the actual discrete message m."""
        return true_bits_from_m(info["m"])


class AdditiveUniformChannel(nn.Module):
    """Additive (non-subtractive) rectangular dither — 1st-order Schuchman scheme.

    ε ~ U(-δ/2, δ/2) is added before quantization; the receiver outputs the
    bin center WITHOUT subtracting ε (no shared RNG needed). This differs from
    DDCL_SD in that only the 1st-order Schuchman condition holds:
      • E[ẑ|z] = z   (unbiased)   ← same as SD / NSD
      • Var(ẑ-z)     is signal-dependent  ← worse than SD (δ²/12) and NSD (δ²/4)

    Role in experiments: intermediate control — same bit cost as SD, same
    deployment simplicity as NSD, but weaker theoretical guarantees.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        d = self.delta
        eps = (torch.rand_like(z) - 0.5) * d
        with torch.no_grad():
            m = torch.floor((z + eps) / d)
            z_hat_true = (m + 0.5) * d
        # STE: forward value is quantized; gradient flows through z unchanged.
        z_hat = z + (z_hat_true - z).detach()
        return z_hat, {"m": m, "eps": eps, "z_hat_true": z_hat_true}

    def comms_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log2(z.abs() / self.delta + 1)

    def transmission_bits_per_elem(
        self, z: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Empirical per-element bits from the actual discrete message m."""
        return true_bits_from_m(info["m"])


class GaussianChannel(nn.Module):
    """Additive Gaussian noise — negative control.

    σ = δ/√12 so variance matches DDCL_SD at the same δ.
    Gaussian dither does NOT satisfy Schuchman's conditions:
      • E[ẑ|z] ≠ z   (biased — tails wrap across bins asymmetrically)
      • Var(ẑ-z) is signal-dependent

    Role in experiments: should perform strictly worse than SD/NSD; if it
    does not, something is wrong with the dithering implementation.

    Bit-cost note: σ = δ/√12 ≈ 0.289δ is narrower than the uniform dither's
    half-width of δ/2 = 0.5δ.  Consequently the empirical ``true_bits_per_msg``
    (from actual m values) is typically *lower* than the surrogate
    ``comms_loss(z) = log₂(|z|/δ + 1)``, because Gaussian noise concentrates m
    near the true bin more than uniform noise does.  The surrogate overestimates
    for Gaussian at all signal levels.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta
        self.sigma = delta / (12 ** 0.5)  # match SD variance δ²/12

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        d = self.delta
        eps = torch.randn_like(z) * self.sigma
        with torch.no_grad():
            m = torch.floor((z + eps) / d)
            z_hat_true = (m + 0.5) * d
        z_hat = z + (z_hat_true - z).detach()
        return z_hat, {"m": m, "eps": eps, "z_hat_true": z_hat_true}

    def comms_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log2(z.abs() / self.delta + 1)

    def transmission_bits_per_elem(
        self, z: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Empirical per-element bits from the actual discrete message m."""
        return true_bits_from_m(info["m"])


class STEChannel(nn.Module):
    """Fixed-rate straight-through estimator quantizer.

    Clips z to [-clip_val, clip_val], quantizes uniformly to 2^bits levels,
    and routes gradients straight through (∂ẑ/∂z = 1 everywhere).

    comms_loss = constant `bits` per element — fixed-rate code, unlike DDCL's
    variable-rate magnitude surrogate. This is the most common non-learned
    baseline in discrete-communication MARL papers.

    Typical usage: STEChannel(bits=4), STEChannel(bits=8), STEChannel(bits=16).
    """

    def __init__(self, bits: int = 8, clip_val: float = 10.0) -> None:
        super().__init__()
        self.bits = bits
        self.clip_val = clip_val
        self.n_levels = 2 ** bits
        self.step = 2.0 * clip_val / (self.n_levels - 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        z_clipped = z.clamp(-self.clip_val, self.clip_val)
        with torch.no_grad():
            m = torch.round((z_clipped + self.clip_val) / self.step).clamp(
                0, self.n_levels - 1
            )
            z_hat_true = m * self.step - self.clip_val
        z_hat = z + (z_hat_true - z).detach()
        return z_hat, {"m": m, "z_hat_true": z_hat_true}

    def comms_loss(self, z: torch.Tensor) -> torch.Tensor:
        # Fixed-rate: always costs `bits` per element regardless of magnitude.
        return torch.full_like(z, float(self.bits))

    def transmission_bits_per_elem(
        self, z: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Fixed-rate: always costs ``self.bits`` per element."""
        return torch.full_like(z, float(self.bits))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CHANNEL_NAMES = ["none", "sd", "nsd", "additive_uniform", "gaussian",
                  "ste4", "ste8", "ste16"]


def build_channel(
    name: str, delta: float, ste_clip: float = 10.0
) -> nn.Module:
    """Instantiate a channel by name.

    Channel names
    -------------
    none             : IdentityChannel (float passthrough)
    sd               : DDCL_SD (subtractive dither)
    nsd              : DDCL_NSD (non-subtractive TPDF dither)
    additive_uniform : AdditiveUniformChannel (1st-order Schuchman)
    gaussian         : GaussianChannel (negative control)
    ste4 / ste8 / ste16 : STEChannel with 4 / 8 / 16 bits

    Parameters
    ----------
    name     : channel identifier (see above)
    delta    : bin width δ (used by SD / NSD / additive_uniform / gaussian)
    ste_clip : clip bound for STE channels (default 10.0)
    """
    if name == "none":
        return IdentityChannel(delta)
    if name == "sd":
        return DDCL_SD(delta)
    if name == "nsd":
        return DDCL_NSD(delta)
    if name == "additive_uniform":
        return AdditiveUniformChannel(delta)
    if name == "gaussian":
        return GaussianChannel(delta)
    if name in ("ste4", "ste8", "ste16"):
        bits = int(name[3:])
        return STEChannel(bits=bits, clip_val=ste_clip)
    raise ValueError(
        f"Unknown channel {name!r}; expected one of {_CHANNEL_NAMES}"
    )
