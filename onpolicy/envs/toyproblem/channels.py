from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IdentityChannel(nn.Module):
    """No-op passthrough so `--channel none` shares the same call shape as SD/NSD."""

    def __init__(
        self,
        delta: float = 1.0,
        delta_learnable: bool = False,
        delta_global_learnable: bool = False,
        zdim: int = 3,
    ) -> None:
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

    delta_learnable=True         → per-dim softplus(raw_delta), shape (zdim,)
    delta_global_learnable=True  → shared scalar softplus(raw_delta), shape (1,)
    both False                   → fixed scalar self._delta
    """

    def __init__(
        self,
        delta: float = 1.0,
        delta_learnable: bool = False,
        delta_global_learnable: bool = False,
        zdim: int = 1,
    ) -> None:
        super().__init__()
        self._delta_learnable = delta_learnable
        self._delta_global_learnable = delta_global_learnable
        if delta_learnable:
            # softplus(0) = ln(2) ≈ 0.693, a reasonable starting δ
            self.raw_delta = nn.Parameter(torch.zeros(zdim))
        elif delta_global_learnable:
            self.raw_delta = nn.Parameter(torch.zeros(1))
        else:
            self._delta = delta

    @property
    def delta(self) -> torch.Tensor | float:
        if self._delta_learnable or self._delta_global_learnable:
            return F.softplus(self.raw_delta)
        return self._delta

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
        return torch.log2(z.abs() / self.delta + 1)


class DDCL_NSD(nn.Module):
    """Non-subtractive-dithering DDCL (user's Implementation 2).

    Triangular dither ν = u₁ + u₂, uᵢ ~ U(-δ/2, δ/2); receiver outputs the
    bin center (no ε subtraction, no shared RNG). Schuchman's theorem gives
    E[ẑ|z] = z exactly, so straight-through with slope 1 is an unbiased
    estimator of the expected gradient.

    delta_learnable=True         → per-dim softplus(raw_delta), shape (zdim,)
    delta_global_learnable=True  → shared scalar softplus(raw_delta), shape (1,)
    both False                   → fixed scalar self._delta
    """

    def __init__(
        self,
        delta: float = 1.0,
        delta_learnable: bool = False,
        delta_global_learnable: bool = False,
        zdim: int = 1,
    ) -> None:
        super().__init__()
        self._delta_learnable = delta_learnable
        self._delta_global_learnable = delta_global_learnable
        if delta_learnable:
            self.raw_delta = nn.Parameter(torch.zeros(zdim))
        elif delta_global_learnable:
            self.raw_delta = nn.Parameter(torch.zeros(1))
        else:
            self._delta = delta

    @property
    def delta(self) -> torch.Tensor | float:
        if self._delta_learnable or self._delta_global_learnable:
            return F.softplus(self.raw_delta)
        return self._delta

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


class DDCL_Async_SD(nn.Module):
    """
    Modified version of DDCL_SD where instead of subtracting the same noise at receiver's end, 
    we subtract a new noise term that is randomly sampled from the same distribution. This breaks
    our theoretical guarantees, but allows the receiver to work independent of the receiver. 
    That's why "async". 
    """

    def __init__(
        self,
        delta: float = 1.0,
        delta_learnable: bool = False,
        delta_global_learnable: bool = False,
        zdim: int = 1,
    ) -> None:
        super().__init__()
        self._delta_learnable = delta_learnable
        self._delta_global_learnable = delta_global_learnable
        if delta_learnable:
            # softplus(0) = ln(2) ≈ 0.693, a reasonable starting δ
            self.raw_delta = nn.Parameter(torch.zeros(zdim))
        elif delta_global_learnable:
            self.raw_delta = nn.Parameter(torch.zeros(1))
        else:
            self._delta = delta

    @property
    def delta(self) -> torch.Tensor | float:
        if self._delta_learnable or self._delta_global_learnable:
            return F.softplus(self.raw_delta)
        return self._delta

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        
        d = self.delta

        eps1 = (torch.rand_like(z) - 0.5) * d
        z_prime = z + eps1
        m = torch.floor(z_prime / d)
        C_m = (m + 0.5) * d
        eps2 = (torch.rand_like(z) - 0.5) * d
        z_hat_deploy = C_m - eps2
        z_hat = z + (z_hat_deploy - z).detach()

        info = {
            "m": m,
            "C_m": C_m,
            "z_prime": z_prime,
            "z_hat_deploy": z_hat_deploy,
            "eps1": eps1,
            "eps2": eps2,
        }

        return z_hat, info

    def comms_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Per-element Jensen upper bound on expected bit length."""
        return torch.log2(z.abs() / self.delta + 1)


def build_channel(
    name: str,
    delta: float,
    delta_learnable: bool = False,
    delta_global_learnable: bool = False,
    zdim: int = 1,
) -> nn.Module:
    table = {"none": IdentityChannel, "sd": DDCL_SD, "nsd": DDCL_NSD, "async_sd": DDCL_Async_SD}
    if name not in table:
        raise ValueError(f"Unknown channel {name!r}; expected one of {list(table)}")
    return table[name](
        delta=delta,
        delta_learnable=delta_learnable,
        delta_global_learnable=delta_global_learnable,
        zdim=zdim,
    )
