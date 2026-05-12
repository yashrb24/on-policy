from __future__ import annotations

import gym
import numpy as np
from gym import spaces


_ACTION_DELTAS = np.array(
    [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]],  # STAY, UP, DOWN, LEFT, RIGHT (coords are (x, y))
    dtype=int,
)

_DEFAULT_GOALS = np.array(
    [[0, 0], [7, 7], [3, 4], [4, 3], [1, 6], [6, 1]], dtype=int
)
_DEFAULT_GOAL_PROBS = np.array(
    [0.515, 0.258, 0.129, 0.064, 0.031, 0.003], dtype=np.float64
)


class CommunicatingGoalEnv(gym.Env):
    """CommunicatingGoalEnv from Kapoor, Bhisikar et al. (2025) §5.1.

    Two agents share a cooperative reward on an 8x8 grid. The speaker (idx 0)
    is stationary and observes the goal coordinates; the listener (idx 1)
    observes only its own coordinates and moves one cell per step. The
    listener must reach the goal — which it cannot see — forcing the speaker
    to communicate through whatever channel the policy provides. The env
    itself carries no message-passing machinery; that is deliberately left
    to the policy so a differentiable channel (DDCL) can be inserted.
    """

    SPEAKER_IDX = 0
    LISTENER_IDX = 1

    def __init__(self, args=None) -> None:
        self.grid_size = getattr(args, "grid_size", 8)
        self.max_steps = getattr(args, "max_steps", 50)
        self.goal_mode = getattr(args, "goal_mode", "weighted")
        self.step_penalty = getattr(args, "step_penalty", -0.01)
        self.goal_reward = getattr(args, "goal_reward", 1.0)

        self.n_agents = 2
        self.naction = 5

        self.np_random = None

        self.goals = _DEFAULT_GOALS.copy()
        self.goal_probs = _DEFAULT_GOAL_PROBS / _DEFAULT_GOAL_PROBS.sum()

        obs_box = spaces.Box(
            low=0.0,
            high=float(self.grid_size - 1),
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = [obs_box, obs_box]
        self.action_space = spaces.MultiDiscrete([self.naction])
        share_box = spaces.Box(
            low=0.0,
            high=float(self.grid_size - 1),
            shape=(2 * self.n_agents,),
            dtype=np.float32,
        )
        self.share_observation_space = [share_box, share_box]

        self.goal_pos = None
        self.listener_pos = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_over = False

    def seed(self, seed=None) -> list[int | None]:
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _rng(self):
        return self.np_random if self.np_random is not None else np.random

    def _sample_goal(self) -> np.ndarray:
        rng = self._rng()
        if self.goal_mode == "weighted":
            idx = rng.choice(len(self.goals), p=self.goal_probs)
            return self.goals[idx].copy()
        if self.goal_mode == "uniform":
            return rng.randint(0, self.grid_size, size=2)
        raise ValueError(f"Unknown goal_mode {self.goal_mode!r}")

    def _sample_listener_start(self, goal_pos) -> np.ndarray:
        rng = self._rng()
        total = self.grid_size * self.grid_size
        goal_flat = int(goal_pos[0]) * self.grid_size + int(goal_pos[1])
        # Draw from [0, total-1), then shift past goal_flat — uniform over non-goal cells in one draw.
        idx = int(rng.randint(0, total - 1))
        if idx >= goal_flat:
            idx += 1
        x, y = divmod(idx, self.grid_size) # divmod returns a tuple of (x // y, x % y)
        return np.array([x, y], dtype=int)

    def reset(self) -> list[np.ndarray]:
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_over = False
        self.goal_pos = np.asarray(self._sample_goal(), dtype=int)
        self.listener_pos = np.asarray(self._sample_listener_start(self.goal_pos), dtype=int)
        return self._get_obs()

    def _get_obs(self) -> list[np.ndarray]:
        return [
            self.goal_pos.astype(np.float32),
            self.listener_pos.astype(np.float32),
        ]

    def step(self, action) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict]:
        if self.episode_over:
            raise RuntimeError("Episode is done; call reset() before step().")

        action = np.atleast_1d(np.asarray(action).squeeze()).astype(int)
        assert action.shape[0] >= self.n_agents, (
            f"Expected at least {self.n_agents} actions, got {action.shape[0]}"
        )
        assert np.all((action >= 0) & (action < self.naction)), (
            f"Actions must be in [0, {self.naction}); got {action}"
        )

        listener_action = int(action[self.LISTENER_IDX])
        delta = _ACTION_DELTAS[listener_action]
        self.listener_pos = np.clip(
            self.listener_pos + delta, 0, self.grid_size - 1
        )

        self.current_step += 1
        reached = bool(np.array_equal(self.listener_pos, self.goal_pos))
        truncated = self.current_step >= self.max_steps
        self.episode_over = reached or truncated

        reward_scalar = self.goal_reward if reached else self.step_penalty
        reward = np.full((self.n_agents, 1), reward_scalar, dtype=np.float32)
        self.episode_reward += float(reward_scalar)

        dones = np.array([self.episode_over] * self.n_agents, dtype=bool)

        info = {
            "success": int(reached),
            "episode_reward": self.episode_reward,
            "goal": tuple(int(v) for v in self.goal_pos),
            "listener_pos": tuple(int(v) for v in self.listener_pos),
            "step": self.current_step,
        }
        return self._get_obs(), reward, dones, info

    def render(self, mode: str = "human") -> str:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        if self.goal_pos is not None:
            gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
            grid[gy][gx] = "G"
        if self.listener_pos is not None:
            lx, ly = int(self.listener_pos[0]), int(self.listener_pos[1])
            grid[ly][lx] = "*" if grid[ly][lx] == "G" else "L"
        return "\n".join("".join(row) for row in grid)

    def close(self) -> None:
        return
