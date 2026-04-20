from __future__ import annotations

import gym
import numpy as np
from gym import spaces

from onpolicy.envs.toyproblem.CommunicatingGoal_env import (
    _ACTION_DELTAS,
    _DEFAULT_GOALS,
    _DEFAULT_GOAL_PROBS,
)


class CommunicatingGoalVecEnv(gym.Env):
    """Native-vectorized CommunicatingGoalEnv operating on (N, ...) tensors.

    Semantics match the single-env variant: speaker stationary observing the
    goal, mobile listener observing only its own position, shared reward
    (+1 reach / -0.01 step), weighted 6-goal sampling. Episodes terminate
    independently per slot; done slots auto-reset in place before the next
    step. The pre-reset observation is exposed in ``info['final_obs_*']`` so
    value-function bootstrapping stays correct at episode boundaries.
    """

    SPEAKER_IDX = 0
    LISTENER_IDX = 1

    def __init__(self, num_envs: int, args=None) -> None:
        self.N = num_envs
        self.num_envs = num_envs
        self.grid_size = getattr(args, "grid_size", 8)
        self.G = self.grid_size
        self.max_steps = getattr(args, "max_steps", 50)
        self.step_penalty = np.float32(getattr(args, "step_penalty", -0.01))
        self.goal_reward = np.float32(getattr(args, "goal_reward", 1.0))

        self.n_agents = 2
        self.naction = 5

        self.rng = np.random.RandomState(None)

        self.goals = _DEFAULT_GOALS.copy()
        self.goal_probs = _DEFAULT_GOAL_PROBS / _DEFAULT_GOAL_PROBS.sum()

        single_box = spaces.Box(
            low=0.0, high=float(self.grid_size - 1),
            shape=(2,), dtype=np.float32,
        )
        self.single_observation_space = [single_box, single_box]
        self.single_action_space = spaces.MultiDiscrete([self.naction])

        batched_box = spaces.Box(
            low=0.0, high=float(self.grid_size - 1),
            shape=(self.N, 2), dtype=np.float32,
        )
        self.observation_space = [batched_box, batched_box]
        self.action_space = spaces.MultiDiscrete([self.naction])

        self.goal_pos = np.zeros((self.N, 2), dtype=int)
        self.listener_pos = np.zeros((self.N, 2), dtype=int)
        self.step_count = np.zeros(self.N, dtype=int)
        self.episode_reward = np.zeros(self.N, dtype=np.float32)

    def seed(self, seed=None) -> list[int | None]:
        self.rng = np.random.RandomState(seed)
        return [seed]

    def _sample_goals(self, n: int) -> np.ndarray:
        idx = self.rng.choice(len(self.goals), size=n, p=self.goal_probs)
        return self.goals[idx].copy()

    def _sample_listener_starts(self, goal_pos: np.ndarray) -> np.ndarray:
        n = goal_pos.shape[0]
        total = self.G * self.G
        goal_flat = goal_pos[:, 0] * self.G + goal_pos[:, 1]
        idx = self.rng.randint(0, total - 1, size=n)
        # Shift past goal_flat element-wise — bijection onto non-goal cells.
        idx = np.where(idx >= goal_flat, idx + 1, idx)
        return np.stack([idx // self.G, idx % self.G], axis=1).astype(int)

    def _reset_slots(self, slots: np.ndarray) -> None:
        if slots.size == 0:
            return
        new_goals = self._sample_goals(slots.size)
        new_listener = self._sample_listener_starts(new_goals)
        self.goal_pos[slots] = new_goals
        self.listener_pos[slots] = new_listener
        self.step_count[slots] = 0
        self.episode_reward[slots] = 0.0

    def reset(self) -> list[np.ndarray]:
        self._reset_slots(np.arange(self.N))
        return self._get_obs()

    def _get_obs(self) -> list[np.ndarray]:
        return [
            self.goal_pos.astype(np.float32),
            self.listener_pos.astype(np.float32),
        ]

    def step(
        self, actions
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict]:
        actions = np.asarray(actions)
        if actions.ndim == 2:
            listener_acts = actions[:, self.LISTENER_IDX].astype(int)
        elif actions.ndim == 1:
            listener_acts = actions.astype(int)
        else:
            raise ValueError(
                f"actions must be shape (N,) or (N, 2); got {actions.shape}"
            )
        assert listener_acts.shape[0] == self.N, (
            f"Expected {self.N} actions, got {listener_acts.shape[0]}"
        )
        assert np.all((listener_acts >= 0) & (listener_acts < self.naction)), (
            f"Actions must be in [0, {self.naction}); got {listener_acts}"
        )

        self.listener_pos = np.clip(
            self.listener_pos + _ACTION_DELTAS[listener_acts],
            0, self.G - 1,
        )
        self.step_count += 1
        reached = (self.listener_pos == self.goal_pos).all(axis=1)
        truncated = self.step_count >= self.max_steps
        done = reached | truncated

        reward_scalar = np.where(
            reached, self.goal_reward, self.step_penalty
        ).astype(np.float32)
        self.episode_reward += reward_scalar

        # Snapshot terminal bookkeeping BEFORE resetting done slots.
        final_obs = self._get_obs()
        final_ep_reward = self.episode_reward.copy()
        final_success = reached.astype(np.int64)

        if done.any():
            self._reset_slots(np.where(done)[0])

        info = {
            "final_obs_speaker": final_obs[0],
            "final_obs_listener": final_obs[1],
            "final_episode_reward": final_ep_reward,
            "success": final_success,
            "done": done.copy(),
        }
        reward = np.broadcast_to(
            reward_scalar[:, None, None], (self.N, self.n_agents, 1)
        ).copy()
        dones = np.broadcast_to(
            done[:, None], (self.N, self.n_agents)
        ).copy()
        return self._get_obs(), reward, dones, info

    def close(self) -> None:
        return
