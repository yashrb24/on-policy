from __future__ import annotations

import math

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
