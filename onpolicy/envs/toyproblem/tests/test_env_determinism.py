"""Determinism and correctness tests for the CommunicatingGoal environments.

Phase 1 implementation: all 5 determinism stubs filled in.
"""
from __future__ import annotations

import numpy as np
import pytest

from onpolicy.envs.toyproblem.CommunicatingGoal_env import (
    CommunicatingGoalEnv,
    _DEFAULT_GOAL_PROBS,
    _DEFAULT_GOALS,
)
from onpolicy.envs.toyproblem.CommunicatingGoal_vec_env import CommunicatingGoalVecEnv


# ---------------------------------------------------------------------------
# Goal distribution constants
# ---------------------------------------------------------------------------

class TestGoalDistribution:
    def test_probs_sum_to_one(self):
        """Goal probabilities must form a valid probability distribution."""
        assert abs(_DEFAULT_GOAL_PROBS.sum() - 1.0) < 1e-6, (
            "Goal probabilities must sum to 1."
        )

    def test_six_goals(self):
        """Paper specifies exactly 6 goal positions."""
        assert len(_DEFAULT_GOALS) == 6, f"Expected 6 goals, got {len(_DEFAULT_GOALS)}"
        assert len(_DEFAULT_GOAL_PROBS) == 6

    def test_goals_in_grid(self):
        """All goal positions must lie within the 8×8 grid."""
        assert (_DEFAULT_GOALS >= 0).all() and (_DEFAULT_GOALS <= 7).all(), (
            "All goal positions must be in [0, 7]²."
        )

    def test_prob_ordering(self):
        """Probabilities must be strictly decreasing (most → least frequent).

        This is required for the Shannon-gap analysis and Huffman-analogue claim.
        """
        probs = _DEFAULT_GOAL_PROBS
        assert all(probs[i] > probs[i + 1] for i in range(len(probs) - 1)), (
            "Goal probabilities must be strictly decreasing."
        )

    def test_goals_unique(self):
        """No two goals share the same position."""
        seen = set()
        for g in _DEFAULT_GOALS:
            key = tuple(g)
            assert key not in seen, f"Duplicate goal position: {key}"
            seen.add(key)


# ---------------------------------------------------------------------------
# Single environment
# ---------------------------------------------------------------------------

class TestSingleEnv:
    def test_reset_obs_shape(self, single_env):
        """Reset returns two observations each of shape (2,)."""
        obs = single_env.reset()
        assert len(obs) == 2
        assert obs[0].shape == (2,), f"Speaker obs shape: {obs[0].shape}"
        assert obs[1].shape == (2,), f"Listener obs shape: {obs[1].shape}"

    def test_obs_in_grid(self, single_env):
        """All observations lie within [0, grid_size-1]."""
        obs = single_env.reset()
        for o in obs:
            assert (o >= 0).all() and (o <= 7).all(), f"Obs out of grid range: {o}"

    def test_listener_not_on_goal_at_start(self, single_env):
        """Listener does not start on the goal cell."""
        for _ in range(50):
            obs = single_env.reset()
            goal, listener = obs
            assert not np.array_equal(goal.astype(int), listener.astype(int)), (
                "Listener must not start on the goal cell."
            )

    def test_step_returns_correct_shapes(self, single_env):
        """Step returns (obs_list, reward, dones, info) with correct shapes."""
        single_env.reset()
        obs, reward, dones, info = single_env.step([0, 0])  # both STAY
        assert len(obs) == 2
        assert reward.shape == (2, 1)
        assert dones.shape == (2,)

    def test_step_penalty_when_not_reached(self, single_env):
        """Step reward is -0.01 (step penalty) when the listener has not reached the goal."""
        single_env.reset()
        # Force listener far from goal to ensure no accidental reach.
        single_env.goal_pos = np.array([0, 0], dtype=int)
        single_env.listener_pos = np.array([7, 7], dtype=int)
        _, reward, dones, info = single_env.step([0, 0])  # STAY
        assert not info["success"], "Listener should not reach goal with STAY from (7,7)."
        assert abs(float(reward[0, 0]) - (-0.01)) < 1e-6

    def test_goal_reward_on_reach(self, single_env):
        """Step reward is +1.0 when the listener reaches the goal."""
        single_env.reset()
        single_env.goal_pos = np.array([3, 3], dtype=int)
        single_env.listener_pos = np.array([3, 3], dtype=int)  # already on goal
        # Take any action — listener is already on goal so 'reached' should be True.
        _, reward, dones, info = single_env.step([0, 0])
        assert info["success"], "Listener on goal must trigger success."
        assert abs(float(reward[0, 0]) - 1.0) < 1e-6

    def test_episode_truncates_at_max_steps(self, single_env):
        """Episode terminates after max_steps even without reaching the goal."""
        single_env.reset()
        # Force listener into a corner far from goal.
        single_env.goal_pos = np.array([0, 0], dtype=int)
        single_env.listener_pos = np.array([7, 7], dtype=int)
        done = False
        steps = 0
        while not done:
            _, _, dones, _ = single_env.step([0, 0])  # STAY forever
            done = dones[0]
            steps += 1
        assert steps == single_env.max_steps, (
            f"Episode should truncate at {single_env.max_steps} steps; got {steps}"
        )

    def test_wall_clamping(self, single_env):
        """Listener cannot move outside the grid (wall clamping)."""
        single_env.reset()
        single_env.listener_pos = np.array([0, 0], dtype=int)
        # UP = [0, -1]: trying to go above y=0
        single_env.step([0, 1])  # action 1 = UP
        assert (single_env.listener_pos >= 0).all()
        assert (single_env.listener_pos <= 7).all()

    def test_determinism_same_seed(self):
        """Two envs with the same seed produce bit-identical trajectories.

        Both envs are seeded identically and driven with the same action sequence.
        Observations, rewards, and done flags must agree at every step.  Three full
        episodes are run (using STAY so truncation drives episode ends deterministically).
        """
        N_EPISODES = 3
        seed = 7

        env1 = CommunicatingGoalEnv()
        env2 = CommunicatingGoalEnv()
        env1.seed(seed)
        env2.seed(seed)

        for ep in range(N_EPISODES):
            o1 = env1.reset()
            o2 = env2.reset()
            assert np.array_equal(o1[0], o2[0]), f"Episode {ep}: speaker obs mismatch at reset"
            assert np.array_equal(o1[1], o2[1]), f"Episode {ep}: listener obs mismatch at reset"

            done = False
            step = 0
            while not done:
                action = [0, 0]  # STAY
                o1, r1, d1, _ = env1.step(action)
                o2, r2, d2, _ = env2.step(action)
                assert np.array_equal(o1[0], o2[0]), f"Ep {ep} step {step}: speaker obs mismatch"
                assert np.array_equal(o1[1], o2[1]), f"Ep {ep} step {step}: listener obs mismatch"
                assert np.array_equal(r1, r2), f"Ep {ep} step {step}: reward mismatch"
                assert np.array_equal(d1, d2), f"Ep {ep} step {step}: done mismatch"
                done = bool(d1[0])
                step += 1

    def test_vec_and_single_env_agree(self):
        """Single env and 1-slot vec env produce identical step outputs from the same state.

        Rather than matching RNG sequences (which differ due to batching),
        we manually force both envs to the same state and verify that step()
        returns identical observations, rewards, and done flags.
        """
        single = CommunicatingGoalEnv()
        single.seed(0)
        single.reset()

        vec = CommunicatingGoalVecEnv(num_envs=1)
        vec.seed(0)
        vec.reset()

        # Force both envs to the same known state
        goal = np.array([3, 4], dtype=int)
        listener = np.array([1, 1], dtype=int)

        single.goal_pos = goal.copy()
        single.listener_pos = listener.copy()
        single.current_step = 0
        single.episode_over = False

        vec.goal_pos[0] = goal.copy()
        vec.listener_pos[0] = listener.copy()
        vec.step_count[0] = 0

        # Take the same sequence of actions and verify agreement
        for action in [2, 4, 2, 4, 3, 1, 0]:  # DOWN, RIGHT, DOWN, RIGHT, LEFT, UP, STAY
            o_s, r_s, d_s, _ = single.step([0, action])
            o_v, r_v, d_v, _ = vec.step(np.array([[0, action]]))

            assert np.array_equal(o_s[0].astype(int), o_v[0][0].astype(int)), (
                f"action={action}: speaker obs mismatch: {o_s[0]} vs {o_v[0][0]}"
            )
            assert np.array_equal(o_s[1].astype(int), o_v[1][0].astype(int)), (
                f"action={action}: listener obs mismatch: {o_s[1]} vs {o_v[1][0]}"
            )
            assert np.isclose(float(r_s[0, 0]), float(r_v[0, 0, 0])), (
                f"action={action}: reward mismatch: {r_s[0,0]} vs {r_v[0,0,0]}"
            )
            if d_s[0] or d_v[0, 0]:
                break  # episode ended; don't compare further


# ---------------------------------------------------------------------------
# Vectorized environment
# ---------------------------------------------------------------------------

class TestVecEnv:
    def test_reset_obs_shapes(self, vec_env_small):
        """VecEnv reset returns (N, 2) observations for both agents."""
        obs = vec_env_small.reset()
        N = vec_env_small.N
        assert obs[0].shape == (N, 2), f"Speaker obs shape: {obs[0].shape}"
        assert obs[1].shape == (N, 2), f"Listener obs shape: {obs[1].shape}"

    def test_step_shapes(self, vec_env_small):
        """VecEnv step returns correctly-shaped outputs."""
        N = vec_env_small.N
        vec_env_small.reset()
        actions = np.zeros((N, 2), dtype=int)
        obs, reward, dones, info = vec_env_small.step(actions)
        assert obs[0].shape == (N, 2)
        assert reward.shape == (N, 2, 1)
        assert dones.shape == (N, 2)

    def test_auto_reset_on_done(self, vec_env_small):
        """Done slots are auto-reset: post-done step_count is 0 and obs are valid.

        Force all slots to one step before truncation, take a STAY action to
        trigger done, then verify that all slots have been auto-reset.
        """
        N = vec_env_small.N
        vec_env_small.reset()

        # Force all slots to be one step before truncation
        vec_env_small.step_count[:] = vec_env_small.max_steps - 1

        actions = np.zeros((N, 2), dtype=int)  # all STAY
        obs, _, dones, _ = vec_env_small.step(actions)

        # All slots must be done at this step
        assert dones[:, 0].all(), "All slots must be done after max_steps"

        # After auto-reset: step_count back to 0, obs must be in-grid
        assert (vec_env_small.step_count == 0).all(), (
            "step_count must be 0 after auto-reset"
        )
        for o in obs:
            assert (o >= 0).all() and (o <= 7).all(), (
                f"Post-reset obs out of grid range: {o}"
            )

    def test_final_obs_in_info(self, vec_env_small):
        """info dict exposes 'final_obs_speaker' and 'final_obs_listener' on done steps.

        These hold the TERMINAL observation (before auto-reset) so that the
        value function can bootstrap correctly at episode boundaries.
        """
        N = vec_env_small.N
        vec_env_small.reset()

        # Record the pre-step positions
        goal_before = vec_env_small.goal_pos.copy()
        listener_before = vec_env_small.listener_pos.copy()

        # Force all slots to be one step before truncation
        vec_env_small.step_count[:] = vec_env_small.max_steps - 1

        actions = np.zeros((N, 2), dtype=int)  # all STAY — listener does not move
        _, _, dones, info = vec_env_small.step(actions)

        assert dones[:, 0].all(), "All slots must be done"
        assert "final_obs_speaker" in info, "info must contain 'final_obs_speaker'"
        assert "final_obs_listener" in info, "info must contain 'final_obs_listener'"

        # STAY action leaves positions unchanged → terminal obs == pre-step obs
        np.testing.assert_array_equal(
            info["final_obs_speaker"].astype(int), goal_before,
            err_msg="final_obs_speaker must equal terminal goal position",
        )
        np.testing.assert_array_equal(
            info["final_obs_listener"].astype(int), listener_before,
            err_msg="final_obs_listener must equal terminal listener position",
        )

    def test_independent_episode_reset(self, vec_env_small):
        """Done slots reset independently without affecting other slots.

        Only slot 0 is forced to truncate; slots 1, 2, 3 remain in-progress.
        After the step, slot 0 must be at step_count=0 while the other slots
        increment normally, and their goal/listener state is unchanged.
        """
        N = vec_env_small.N
        vec_env_small.reset()

        # Record state of slots 1+ before the step
        goal_others = vec_env_small.goal_pos[1:].copy()
        listener_others = vec_env_small.listener_pos[1:].copy()

        # Force only slot 0 to trigger truncation
        vec_env_small.step_count[0] = vec_env_small.max_steps - 1
        # Slots 1+ start at step 0

        actions = np.zeros((N, 2), dtype=int)  # all STAY
        _, _, dones, _ = vec_env_small.step(actions)

        # Slot 0 done; others not done
        assert dones[0, 0], "Slot 0 must be done after max_steps"
        for i in range(1, N):
            assert not dones[i, 0], f"Slot {i} must not be done"

        # Slot 0 auto-reset: step_count back to 0
        assert vec_env_small.step_count[0] == 0, "Done slot must reset step_count to 0"

        # Slots 1+ incremented normally (step 0 → step 1)
        for i in range(1, N):
            assert vec_env_small.step_count[i] == 1, (
                f"Non-done slot {i} step_count must be 1, got {vec_env_small.step_count[i]}"
            )

        # Goal and listener of non-done slots must be unchanged
        np.testing.assert_array_equal(
            vec_env_small.goal_pos[1:], goal_others,
            err_msg="Non-done slots must not have their goal reset",
        )
        np.testing.assert_array_equal(
            vec_env_small.listener_pos[1:], listener_others,
            err_msg="Non-done slots must not have their listener position reset",
        )
