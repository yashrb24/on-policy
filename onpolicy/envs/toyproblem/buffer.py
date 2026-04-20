from __future__ import annotations

from typing import Iterator

import torch


class RolloutBuffer:
    """Fixed-size tensor buffer for n_steps * n_envs transitions on one device.

    Values are stored in NORMALIZED space (critic outputs go here directly);
    GAE denormalizes once via the provided ValueNorm. Advantages/returns are
    in RAW scale after `compute_returns_and_advantages`.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        z_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device

        # Rollout-time storage.
        self.goals = torch.zeros(n_steps, n_envs, 2, device=device)
        self.listener_pos = torch.zeros(n_steps, n_envs, 2, device=device)
        self.actions = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, 1, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)

        # Filled by compute_returns_and_advantages.
        self.advantages = torch.zeros(n_steps, n_envs, device=device)
        self.returns = torch.zeros(n_steps, n_envs, device=device)

        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def insert(
        self,
        goal: torch.Tensor,
        listener_pos: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        t = self.step
        self.goals[t] = goal
        self.listener_pos[t] = listener_pos
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        value_norm,
        gamma: float,
        lam: float,
    ) -> None:
        """GAE over raw-scale values. `last_value` is (n_envs, 1) NORMALIZED."""
        # Denormalize once; ValueNorm returns numpy, bring back to tensor.
        values_raw = torch.as_tensor(
            value_norm.denormalize(self.values), device=self.device
        ).squeeze(-1)  # (T, N)
        last_value_raw = torch.as_tensor(
            value_norm.denormalize(last_value), device=self.device
        ).squeeze(-1)  # (N,)

        advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value_raw if t == self.n_steps - 1 else values_raw[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - values_raw[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns = advantages + values_raw

    def minibatches(
        self, num_minibatches: int, shuffle: bool = True
    ) -> Iterator[dict[str, torch.Tensor]]:
        batch_size = self.n_steps * self.n_envs
        assert batch_size % num_minibatches == 0, (
            f"batch_size {batch_size} not divisible by num_minibatches {num_minibatches}"
        )
        mb_size = batch_size // num_minibatches

        goals = self.goals.reshape(batch_size, 2)
        listener_pos = self.listener_pos.reshape(batch_size, 2)
        actions = self.actions.reshape(batch_size)
        log_probs = self.log_probs.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        returns = self.returns.reshape(batch_size)

        if shuffle:
            idx = torch.randperm(batch_size, device=self.device)
        else:
            idx = torch.arange(batch_size, device=self.device)

        for start in range(0, batch_size, mb_size):
            mb = idx[start : start + mb_size]
            yield {
                "goals": goals[mb],
                "listener_pos": listener_pos[mb],
                "actions": actions[mb],
                "old_log_probs": log_probs[mb],
                "advantages": advantages[mb],
                "returns": returns[mb],
            }
