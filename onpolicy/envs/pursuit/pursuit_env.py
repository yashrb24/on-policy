"""MAPPO-compatible wrapper around PettingZoo SISL pursuit_v4.

Exposes the same gym.Env-list-spaces contract used by predator_prey /
traffic_junction so it slots into env_wrappers.{Dummy,Subproc}VecEnv and the
shared Runner without bespoke runner/buffer changes.
"""

import os
import types

import gym
import numpy as np
from gym.spaces import Box
from pettingzoo.sisl import pursuit_v4


def _fast_reward(self):
    """Faster drop-in replacement for PettingZoo's PursuitBase.reward() that
    returns identical rewards.

    The stock reward() is recomputed on every sub-step (so O(n_pursuers**2) per
    environment cycle, since the parallel API steps agents one at a time) and is
    the dominant cost of rollout collection. It only adds, per pursuer,
    tag_reward * (number of evaders inside the surround window).

    - tag_reward == 0 (the training recipe): the result is all zeros, so we skip
      the whole computation. Identical by definition (0 * anything == 0).
    - tag_reward != 0: the same per-pursuer window sums, computed in one vectorized
      numpy step instead of a Python loop. Same values.
    """
    if self.tag_reward == 0:
        return np.zeros(self.n_pursuers)
    es = self.evader_layer.get_state_matrix()
    pos = np.array(
        [self.pursuer_layer.get_position(i) for i in range(self.n_pursuers)]
    )  # (N, 2)
    xs = np.clip(pos[:, 0][:, None] + self.surround_mask[:, 0][None, :], 0, self.x_size - 1)
    ys = np.clip(pos[:, 1][:, None] + self.surround_mask[:, 1][None, :], 0, self.y_size - 1)
    return self.tag_reward * es[xs, ys].sum(axis=1)


class PursuitEnv(gym.Env):
    def __init__(self, args):
        self.n_pursuers = args.n_pursuers
        self.n_evaders = args.n_evaders
        self.x_size = args.x_size
        self.y_size = args.y_size
        self.max_cycles = args.max_cycles

        self.env = pursuit_v4.parallel_env(
            n_pursuers=self.n_pursuers,
            n_evaders=self.n_evaders,
            x_size=self.x_size,
            y_size=self.y_size,
            max_cycles=self.max_cycles,
            catch_reward=args.catch_reward,
            tag_reward=args.tag_reward,
            urgency_reward=args.urgency_reward,
        )
        # Trigger one reset so possible_agents and per-agent spaces are populated.
        self.env.reset()

        # Swap in the faster reward() (see _fast_reward) for this env instance only.
        # Guarded so we fall back to the stock method if PettingZoo's internals change.
        # Set PURSUIT_DISABLE_FAST_REWARD=1 to force the original method.
        if not os.environ.get("PURSUIT_DISABLE_FAST_REWARD"):
            try:
                sim = self.env.unwrapped.env
                sim.reward = types.MethodType(_fast_reward, sim)
            except AttributeError:
                pass

        self.possible_agents = list(self.env.possible_agents)
        self.num_agents = len(self.possible_agents)

        sample_obs = self.env.observation_space(self.possible_agents[0])
        self._raw_obs_shape = tuple(sample_obs.shape)
        # Optional SCoUT-style representation: drop the walls/boundary plane (channel 0 of the
        # 7x7x3 window) -> 7x7x2 = 98-dim. Default off (147-dim).
        self._drop_walls = bool(args.pursuit_drop_walls_channel) \
            and len(self._raw_obs_shape) == 3 and self._raw_obs_shape[2] >= 2
        if self._drop_walls:
            self.obs_dim = int(self._raw_obs_shape[0] * self._raw_obs_shape[1] * (self._raw_obs_shape[2] - 1))
        else:
            self.obs_dim = int(np.prod(sample_obs.shape))
        self._obs_dtype = sample_obs.dtype
        single_obs_box = Box(
            low=float(sample_obs.low.min()),
            high=float(sample_obs.high.max()),
            shape=(self.obs_dim,),
            dtype=self._obs_dtype,
        )
        global_share_obs_box = Box(
            low=float(sample_obs.low.min()),
            high=float(sample_obs.high.max()),
            shape=(self.obs_dim * self.num_agents,),
            dtype=self._obs_dtype,
        )
        share_obs_box = single_obs_box if getattr(args, "use_transformer_base_critic", False) else global_share_obs_box
        single_act = self.env.action_space(self.possible_agents[0])

        self.observation_space = [single_obs_box for _ in range(self.num_agents)]
        self.share_observation_space = [share_obs_box for _ in range(self.num_agents)]
        self.action_space = [single_act for _ in range(self.num_agents)]

        self._seed = None

    def _dict_to_obs(self, obs_dict):
        rows = []
        for ag in self.possible_agents:
            v = obs_dict.get(ag)
            if v is None:
                rows.append(np.zeros(self.obs_dim, dtype=np.float32))
            else:
                arr = np.asarray(v, dtype=np.float32)
                if self._drop_walls:
                    arr = arr.reshape(self._raw_obs_shape)[:, :, 1:]
                rows.append(arr.reshape(-1))
        return np.stack(rows, axis=0)

    def reset(self):
        if self._seed is not None:
            obs_dict, _ = self.env.reset(seed=self._seed)
            self._seed = None
        else:
            obs_dict, _ = self.env.reset()
        return self._dict_to_obs(obs_dict)

    def step(self, action):
        action = np.asarray(action).flatten()
        actions_dict = {
            ag: int(action[i])
            for i, ag in enumerate(self.possible_agents)
            if ag in self.env.agents
        }
        obs_dict, rew_dict, term_dict, trunc_dict, _ = self.env.step(actions_dict)

        obs = self._dict_to_obs(obs_dict)
        rewards = np.array(
            [rew_dict.get(ag, 0.0) for ag in self.possible_agents],
            dtype=np.float32,
        ).reshape(self.num_agents, 1)

        all_term = all(term_dict.get(ag, False) for ag in self.possible_agents)
        all_trunc = all(trunc_dict.get(ag, False) for ag in self.possible_agents)
        dones = np.array(
            [
                term_dict.get(ag, all_term) or trunc_dict.get(ag, all_trunc)
                for ag in self.possible_agents
            ],
            dtype=bool,
        )

        info = {
            "success": float(all_term),
            "truncated": float(all_trunc),
            "team_step_reward": float(rewards.sum()),
        }

        sim = self.env.unwrapped.env
        n_captures = int(np.sum(sim.evaders_gone))
        info["n_captures_so_far"] = n_captures
        info["n_evaders_remaining"] = self.n_evaders - n_captures

        return obs, rewards, dones, info

    def seed(self, seed=None):
        self._seed = seed

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render()
