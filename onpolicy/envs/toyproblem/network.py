from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(
    layer: nn.Linear,
    std: float = math.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SpeakerNetwork(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, hidden: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.GELU(),
            layer_init(nn.Linear(hidden, z_dim), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ListenerActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.GELU(),
        )
        self.logits = layer_init(nn.Linear(hidden, action_dim), std=0.01)

    def forward(self, x: torch.Tensor) -> Categorical:
        features = self.network(x)
        return Categorical(logits=self.logits(features))


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)),
            nn.GELU(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EntropyModelFactored(nn.Module):
    """Per-dimension Discretised Logistic Mixture prior.

    q_φ(m) = ∏_k q_φ_k(m_k)
    q_φ_k(m_k) = Σ_c π_c · [σ((m_k+0.5−μ_c)/s_c) − σ((m_k−0.5−μ_c)/s_c)]

    Works for both discrete m.float() (forward loss: trains q_φ) and continuous
    z/δ (backward loss: grads flow to speaker via Ballé relaxation).
    """

    def __init__(self, z_dim: int, K: int = 5) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.K = K
        # (z_dim, K) — uniform mixture, centred, wide scales
        self.log_pi = nn.Parameter(torch.zeros(z_dim, K))
        self.mu = nn.Parameter(torch.zeros(z_dim, K))
        self.log_s = nn.Parameter(torch.ones(z_dim, K))  # s = e ≈ 2.72 at init

    @staticmethod
    def _dlm_log_prob(
        x: torch.Tensor,       # (..., z_dim)
        log_pi: torch.Tensor,  # (z_dim, K)  or  (..., z_dim, K)
        mu: torch.Tensor,      # same shape as log_pi
        s: torch.Tensor,       # same shape as log_pi (positive)
    ) -> torch.Tensor:         # (..., z_dim)
        """DLM log-probability per dimension."""
        x_e = x.unsqueeze(-1)                                         # (..., z_dim, 1)
        upper = torch.sigmoid((x_e + 0.5 - mu) / s)                  # (..., z_dim, K)
        lower = torch.sigmoid((x_e - 0.5 - mu) / s)                  # (..., z_dim, K)
        log_pi_n = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)
        log_p_k = log_pi_n + (upper - lower).clamp(min=1e-10).log()  # (..., z_dim, K)
        return torch.logsumexp(log_p_k, dim=-1)                       # (..., z_dim)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log q_φ(x) per dimension. x: (..., z_dim)."""
        return self._dlm_log_prob(x, self.log_pi, self.mu, self.log_s.exp())

    def nll_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood in bits per element: −log₂ q_φ(x)."""
        return -self.log_prob(x) / math.log(2)


class EntropyModelJoint(nn.Module):
    """Autoregressive DLM prior.

    q_φ(m) = q_φ_0(m_0) · ∏_{k=1}^{K-1} q_φ_k(m_k | m_0,...,m_{k-1})

    Each conditional q_φ_k is a DLM whose parameters are produced by a
    small MLP taking the previous k dimensions as context.
    For z_dim=1 this reduces to EntropyModelFactored.
    """

    def __init__(self, z_dim: int, K: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.K = K
        # Dimension 0: marginal prior (no conditioning)
        self.log_pi_0 = nn.Parameter(torch.zeros(K))
        self.mu_0 = nn.Parameter(torch.zeros(K))
        self.log_s_0 = nn.Parameter(torch.ones(K))  # wide init
        # Conditional MLPs: dim k conditioned on dims 0..k-1
        self.cond_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(k, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3 * K),
            )
            for k in range(1, z_dim)
        ])

    @staticmethod
    def _dlm_log_prob_1d(
        x: torch.Tensor,       # (...)
        log_pi: torch.Tensor,  # (..., K)  or  (K,)
        mu: torch.Tensor,      # (..., K)  or  (K,)
        s: torch.Tensor,       # (..., K)  or  (K,)  — positive
    ) -> torch.Tensor:         # (...)
        x_e = x.unsqueeze(-1)
        upper = torch.sigmoid((x_e + 0.5 - mu) / s)
        lower = torch.sigmoid((x_e - 0.5 - mu) / s)
        log_pi_n = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)
        log_p_k = log_pi_n + (upper - lower).clamp(min=1e-10).log()
        return torch.logsumexp(log_p_k, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log q_φ(x) per dimension. x: (..., z_dim)."""
        lp = [self._dlm_log_prob_1d(
            x[..., 0], self.log_pi_0, self.mu_0, self.log_s_0.exp()
        )]
        for k, mlp in enumerate(self.cond_mlps, start=1):
            params = mlp(x[..., :k])           # (..., 3K)
            log_pi_k = params[..., :self.K]
            mu_k = params[..., self.K:2 * self.K]
            # bias log_s output toward wide init (add 1.0 before exp)
            s_k = (params[..., 2 * self.K:] + 1.0).exp()
            lp.append(self._dlm_log_prob_1d(x[..., k], log_pi_k, mu_k, s_k))
        return torch.stack(lp, dim=-1)          # (..., z_dim)

    def nll_bits(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x) / math.log(2)


class EntropyModelCondZ(nn.Module):
    """Context-B DLM: q_φ(m | z). Per-dimension MLP(z) → DLM params.

    Unrealistic at deployment (receiver does not observe z), but useful as an
    oracle upper bound on rate reduction achievable with z-side information.
    """

    def __init__(self, z_dim: int, K: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.K = K
        # One MLP per output dimension: full z → DLM params for that dimension
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3 * K),
            )
            for _ in range(z_dim)
        ])

    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Log q_φ(x|z) per dimension. x, z: (..., z_dim)."""
        lp = []
        for k, mlp in enumerate(self.mlps):
            params = mlp(z)                           # (..., 3K)
            log_pi_k = params[..., :self.K]
            mu_k = params[..., self.K:2 * self.K]
            s_k = (params[..., 2 * self.K:] + 1.0).exp()  # wide init bias
            x_e = x[..., k].unsqueeze(-1)             # (..., 1)
            upper = torch.sigmoid((x_e + 0.5 - mu_k) / s_k)
            lower = torch.sigmoid((x_e - 0.5 - mu_k) / s_k)
            log_pi_n = log_pi_k - torch.logsumexp(log_pi_k, dim=-1, keepdim=True)
            log_p_k = log_pi_n + (upper - lower).clamp(min=1e-10).log()
            lp.append(torch.logsumexp(log_p_k, dim=-1))  # (...)
        return torch.stack(lp, dim=-1)                # (..., z_dim)

    def nll_bits(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, z) / math.log(2)


class EntropyModelJointCondZ(nn.Module):
    """Context-B autoregressive DLM: q_φ(m | z).

    q(m|z) = q_0(m_0|z) · ∏_{k≥1} q_k(m_k | m_0,...,m_{k-1}, z)

    Closes the 2×2 of (factored/joint) × (context A/B). Each conditional MLP
    takes [m_{<k}, z] as context, so z informs every conditional directly.
    For z_dim=1 this reduces to EntropyModelCondZ.
    """

    def __init__(self, z_dim: int, K: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.K = K
        # Dim 0: conditioned on z only  (input size = z_dim)
        self.mlp_0 = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * K),
        )
        # Dims 1..z_dim-1: conditioned on [m_{<k}, z]  (input size = k + z_dim)
        self.cond_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(k + z_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3 * K),
            )
            for k in range(1, z_dim)
        ])

    @staticmethod
    def _eval_dlm_1d(
        x_k: torch.Tensor,     # (...)
        params: torch.Tensor,  # (..., 3K)
        K: int,
    ) -> torch.Tensor:         # (...)
        log_pi = params[..., :K]
        mu = params[..., K:2 * K]
        s = (params[..., 2 * K:] + 1.0).exp()  # wide init bias
        x_e = x_k.unsqueeze(-1)                 # (..., 1)
        upper = torch.sigmoid((x_e + 0.5 - mu) / s)
        lower = torch.sigmoid((x_e - 0.5 - mu) / s)
        log_pi_n = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)
        return torch.logsumexp(
            log_pi_n + (upper - lower).clamp(min=1e-10).log(), dim=-1
        )

    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Log q_φ(x|z) per dimension. x, z: (..., z_dim)."""
        lp = [self._eval_dlm_1d(x[..., 0], self.mlp_0(z), self.K)]
        for k, mlp in enumerate(self.cond_mlps, start=1):
            context = torch.cat([x[..., :k], z], dim=-1)  # (..., k + z_dim)
            lp.append(self._eval_dlm_1d(x[..., k], mlp(context), self.K))
        return torch.stack(lp, dim=-1)  # (..., z_dim)

    def nll_bits(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, z) / math.log(2)


# ---------------------------------------------------------------------------
# Empirical entropy utilities (no-grad, batch-level estimates)
# ---------------------------------------------------------------------------

def _marginal_entropy_bits_1d(m_col: torch.Tensor) -> float:
    """Empirical H(m_k) in bits from a 1D integer tensor."""
    m_np = m_col.detach().cpu().numpy().astype(int).ravel()
    _, counts = np.unique(m_np, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def joint_entropy_bits(m: torch.Tensor) -> float:
    """Empirical joint entropy H(m_1,...,m_K) in bits.

    Parameters
    ----------
    m : Tensor of shape (batch, z_dim) — integer-valued
    """
    if m.ndim == 1 or m.shape[-1] == 1:
        return _marginal_entropy_bits_1d(m)
    m_np = m.detach().cpu().numpy().astype(int)
    _, counts = np.unique(m_np, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def total_correlation_bits(m: torch.Tensor) -> float:
    """Empirical total correlation TC = Σ_k H(m_k) − H(m) in bits.

    Non-negative; equals 0 iff all dimensions are mutually independent.
    For z_dim=1 always returns 0.0.

    Parameters
    ----------
    m : Tensor of shape (batch, z_dim) — integer-valued
    """
    if m.ndim == 1 or m.shape[-1] == 1:
        return 0.0
    marginal_sum = float(sum(
        _marginal_entropy_bits_1d(m[:, k]) for k in range(m.shape[-1])
    ))
    return max(0.0, marginal_sum - joint_entropy_bits(m))
