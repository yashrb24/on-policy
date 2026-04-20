from __future__ import annotations

import torch
from torch import nn


class IdentityChannel(nn.Module):
    """No-op passthrough so `--channel none` shares the same call shape as SD/NSD."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return z, {}

    def comms_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(z)


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
        """Per-element Jensen upper bound on expected bit length."""
        return torch.log2(2 * z.abs() / self.delta + 1)


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
        return torch.log2(2 * z.abs() / self.delta + 1)


def build_channel(name: str, delta: float) -> nn.Module:
    table = {"none": IdentityChannel, "sd": DDCL_SD, "nsd": DDCL_NSD}
    if name not in table:
        raise ValueError(f"Unknown channel {name!r}; expected one of {list(table)}")
    return table[name](delta)
