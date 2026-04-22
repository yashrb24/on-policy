from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMPrior(nn.Module):
    """GMM over R^D, configurable as joint or per-dimension independent.

    structure="joint":
        K joint components in R^D with diagonal covariance.
        pi is (K,). Good for holistic, correlated codewords.

    structure="independent":
        D independent 1-D GMMs. Each dimension clusters to its own K modes.
        pi is (K, D). Good for orthogonal, disentangled channels.
    """

    def __init__(
        self,
        num_components: int,
        signal_dim: int,
        init_spread: float = 1.0,
        eps_min: float = 1e-3,
        structure: str = "joint",
    ) -> None:
        super().__init__()
        self.K, self.D, self.eps_min = num_components, signal_dim, eps_min
        self.structure = structure

        if structure == "joint":
            self.logits_w = nn.Parameter(torch.zeros(num_components))
        elif structure == "independent":
            self.logits_w = nn.Parameter(torch.zeros(num_components, signal_dim))
        else:
            raise ValueError(f"Unknown GMM structure: {structure}")

        init_mu = torch.linspace(-init_spread, init_spread, num_components)
        self.mu = nn.Parameter(init_mu.unsqueeze(-1).repeat(1, signal_dim))  # (K, D)
        self.v = nn.Parameter(torch.zeros(num_components, signal_dim))       # (K, D)

    @property
    def pi(self) -> torch.Tensor:
        # Softmax over the component dimension (dim=0)
        return F.softmax(self.logits_w, dim=0)

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.v) + self.eps_min

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluates the log probability of z under the chosen structure."""
        z_b = z.unsqueeze(-2)                                       # (..., 1, D)
        sig = self.sigma                                            # (K, D)

        # Log probability of the Gaussian components
        log_n = (
            -0.5 * ((z_b - self.mu) / sig) ** 2
            - torch.log(sig)
            - 0.5 * math.log(2 * math.pi)
        )                                                           # (..., K, D)

        log_pi = torch.log(self.pi + 1e-12)                         # (K,) or (K, D)

        if self.structure == "joint":
            # 1. Sum log probabilities across D for the joint Gaussian
            log_joint = log_n.sum(dim=-1)                           # (..., K)
            # 2. Add component log weights and marginalize over K
            return torch.logsumexp(log_joint + log_pi, dim=-1)      # (...,)

        else:  # independent
            # 1. Add component log weights to the independent 1D Gaussians
            mix_log_probs = log_n + log_pi                          # (..., K, D)
            # 2. Marginalize over K *per dimension*
            dim_log_probs = torch.logsumexp(mix_log_probs, dim=-2)  # (..., D)
            # 3. Sum the independent dimension log probabilities
            return dim_log_probs.sum(dim=-1)                        # (...,)

    def weights_entropy(self) -> torch.Tensor:
        """
        Total entropy of the categorical distribution(s).
        For "joint", p is (K,). For "independent", p is (K, D).
        Summing captures total system categorical entropy in both cases.
        """
        p = self.pi
        return -(p * torch.log(p + 1e-12)).sum()


def beta_schedule(
    step: int, warmup: int, anneal: int, beta_max: float
) -> float:
    if step < warmup:
        return 0.0
    if step < warmup + anneal:
        return beta_max * (step - warmup) / max(anneal, 1)
    return beta_max
