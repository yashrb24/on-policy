#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of the Predator-Prey Environment.

This file demonstrates various ways to use the environment including:
- Basic usage
- Different game modes
- Multi-agent scenarios
- Rendering
- Custom configurations
"""

import argparse
import time

import numpy as np

from PredatorPrey_env import PredatorPreyEnv


def basic_usage_example():
    """Basic usage without any arguments."""
    print("=" * 50)
    print("BASIC USAGE EXAMPLE")
    print("=" * 50)

    # Create a simple argument namespace for initialization
    args = argparse.Namespace(
        nenemies=1,  # Number of prey
        nfriendly=2,  # Number of predators
        dim=5,
        vision=2,
        moving_prey=False,
        no_stay=False,
        mode='mixed',
        enemy_comm=False
    )

    # Create environment with default settings
    env = PredatorPreyEnv(args)

    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Number of predators: {env.npredator}")
    print(f"Number of prey: {env.nprey}")
    print(f"Grid dimensions: {env.dims}")

    # Run a few steps with random actions
    for step in range(5):
        # Random actions for each predator
        actions = np.random.randint(0, env.naction, size=env.npredator)
        obs, reward, done, info = env.step(actions)

        print(f"\nStep {step + 1}:")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {reward}")
        print(f"  Done: {done}")
        print(f"  Predator locations: {info['predator_locs']}")
        print(f"  Prey locations: {info['prey_locs']}")

        if done.any():
            print("  Episode completed!")
            break


def cooperative_mode_example():
    """Example with cooperative mode where predators work together."""
    print("\n" + "=" * 50)
    print("COOPERATIVE MODE EXAMPLE")
    print("=" * 50)

    args = argparse.Namespace(
        nenemies=1,
        nfriendly=3,  # 3 predators
        dim=7,  # Larger grid
        vision=2,
        moving_prey=False,
        no_stay=False,
        mode='cooperative',  # Cooperative mode
        enemy_comm=False
    )

    env = PredatorPreyEnv(args)
    obs = env.reset()

    print(f"Mode: Cooperative")
    print(f"Grid size: {args.dim}x{args.dim}")
    print(f"Number of predators: {args.nfriendly}")
    print("In cooperative mode, rewards increase with more predators on prey")

    # Run episode
    total_rewards = np.zeros(env.npredator)
    steps = 0
    max_steps = 20

    while steps < max_steps:
        actions = np.random.randint(0, env.naction, size=env.npredator)
        obs, reward, done, info = env.step(actions)
        total_rewards += reward[:env.npredator]
        steps += 1

        # Check if any predator reached prey
        if np.any(reward[:env.npredator] > 0):
            print(f"\nStep {steps}: Predator(s) reached prey!")
            print(f"  Rewards this step: {reward[:env.npredator]}")
            print(f"  Number of predators on prey: {np.sum(reward[:env.npredator] > 0)}")

        if done.any():
            break

    print(f"\nEpisode finished after {steps} steps")
    print(f"Total rewards per predator: {total_rewards}")


def competitive_mode_example():
    """Example with competitive mode where predators compete."""
    print("\n" + "=" * 50)
    print("COMPETITIVE MODE EXAMPLE")
    print("=" * 50)

    args = argparse.Namespace(
        nenemies=1,
        nfriendly=3,
        dim=6,
        vision=2,
        moving_prey=False,
        no_stay=False,
        mode='competitive',  # Competitive mode
        enemy_comm=False
    )

    env = PredatorPreyEnv(args)
    obs = env.reset()

    print(f"Mode: Competitive")
    print(f"Number of predators: {args.nfriendly}")
    print("In competitive mode, rewards are divided among predators on prey")

    # Run episode
    total_rewards = np.zeros(env.npredator)
    steps = 0
    max_steps = 20

    while steps < max_steps:
        actions = np.random.randint(0, env.naction, size=env.npredator)
        obs, reward, done, info = env.step(actions)
        total_rewards += reward[:env.npredator]
        steps += 1

        if np.any(reward[:env.npredator] > 0):
            print(f"\nStep {steps}: Predator(s) reached prey!")
            print(f"  Individual rewards: {reward[:env.npredator]}")
            n_on_prey = np.sum(reward[:env.npredator] > 0)
            if n_on_prey > 0:
                print(f"  Reward split among {n_on_prey} predator(s)")

        if done.any():
            break

    print(f"\nEpisode finished after {steps} steps")
    print(f"Total rewards per predator: {total_rewards}")


def enemy_communication_example():
    """Example with enemy communication enabled."""
    print("\n" + "=" * 50)
    print("ENEMY COMMUNICATION EXAMPLE")
    print("=" * 50)

    args = argparse.Namespace(
        nenemies=2,  # 2 prey
        nfriendly=2,  # 2 predators
        dim=8,
        vision=2,
        moving_prey=False,
        no_stay=False,
        mode='mixed',
        enemy_comm=True  # Enable enemy communication
    )
    env = PredatorPreyEnv(args)

    obs = env.reset()

    print(f"Enemy communication: Enabled")
    print(f"Number of predators: {args.nfriendly}")
    print(f"Number of prey: {args.nenemies}")
    print(f"Observation shape: {obs.shape}")
    print("With enemy_comm, prey also get observations and rewards")

    # Run a few steps
    for step in range(5):
        # Actions for predators only (prey don't move in this version)
        actions = np.random.randint(0, env.naction, size=env.npredator)
        obs, reward, done, info = env.step(actions)

        print(f"\nStep {step + 1}:")
        print(f"  Predator rewards: {reward[:env.npredator]}")
        if env.enemy_comm:
            print(f"  Prey rewards: {reward[env.npredator:]}")

        if done.any():
            break


def rendering_example():
    """Example with rendering enabled (requires curses support)."""
    print("\n" + "=" * 50)
    print("RENDERING EXAMPLE (if curses available)")
    print("=" * 50)

    args = argparse.Namespace(
        nenemies=1,
        nfriendly=2,
        dim=5,
        vision=2,
        moving_prey=False,
        no_stay=False,
        mode='mixed',
        enemy_comm=False
    )

    env = PredatorPreyEnv(args)

    try:
        # Initialize curses for rendering
        env.init_curses()
        obs = env.reset()

        print("Rendering enabled. Running episode with visualization...")
        print("X = Predator, P = Prey")
        print("Press Ctrl+C to stop")

        # Run episode with rendering
        for step in range(20):
            env.render()
            time.sleep(0.5)  # Pause to see the visualization

            actions = np.random.randint(0, env.naction, size=env.npredator)
            obs, reward, done, info = env.step(actions)

            if done.any():
                print("\nEpisode completed!")
                break

        env.exit_render()

    except Exception as e:
        print(f"Rendering not available: {e}")
        print("Curses might not be supported on this system")


def smart_agent_example():
    """Example with a simple smart agent that moves toward prey."""
    print("\n" + "=" * 50)
    print("SMART AGENT EXAMPLE")
    print("=" * 50)

    args = argparse.Namespace(
        nenemies=1,
        nfriendly=2,
        dim=7,
        vision=3,  # Larger vision
        moving_prey=False,
        no_stay=False,
        mode='mixed',
        enemy_comm=False
    )

    env = PredatorPreyEnv(args)
    obs = env.reset()

    print("Smart agents that move toward prey when visible")
    print(f"Vision range: {args.vision}")

    def get_smart_action(observation, agent_idx):
        """Simple policy: move toward prey if visible, else random."""
        # Check if prey is visible in observation
        # The observation is a 3D array: [vocab_size, 2*vision+1, 2*vision+1]

        # Get prey channel
        prey_channel = observation[env.PREY_CLASS]

        # Find prey position if visible
        prey_positions = np.where(prey_channel > 0)

        if len(prey_positions[0]) > 0:
            # Prey is visible, move toward it
            prey_y, prey_x = prey_positions[0][0], prey_positions[1][0]
            center = args.vision  # Agent is at center of observation

            # Determine direction to move
            if prey_y < center:
                return 0  # UP
            elif prey_y > center:
                return 2  # DOWN
            elif prey_x < center:
                return 3  # LEFT
            elif prey_x > center:
                return 1  # RIGHT
            else:
                return 4  # STAY (on prey)
        else:
            # Prey not visible, move randomly
            return np.random.randint(0, env.naction)

    # Run episode with smart agents
    steps = 0
    max_steps = 30

    while steps < max_steps:
        # Get smart actions for each predator
        actions = []
        for i in range(env.npredator):
            action = get_smart_action(obs[i], i)
            actions.append(action)

        obs, reward, done, info = env.step(actions)
        steps += 1

        if np.any(reward[:env.npredator] > 0):
            print(f"Step {steps}: Smart predator caught prey!")
            print(f"  Final positions - Predators: {info['predator_locs']}, Prey: {info['prey_locs']}")
            break

        if done.any():
            print(f"Step {steps}: Episode ended")
            break

        if steps % 10 == 0:
            print(f"Step {steps}: Still hunting...")
            print(f"  Predator positions: {info['predator_locs']}")
            print(f"  Prey position: {info['prey_locs']}")

    if steps >= max_steps:
        print(f"Episode timeout after {max_steps} steps")


def statistics_example():
    """Example showing how to collect statistics over multiple episodes."""
    print("\n" + "=" * 50)
    print("STATISTICS COLLECTION EXAMPLE")
    print("=" * 50)

    args = argparse.Namespace(
        nenemies=1,
        nfriendly=2,
        dim=5,
        vision=2,
        moving_prey=False,
        no_stay=False,
        mode='mixed',
        enemy_comm=False
    )

    env = PredatorPreyEnv(args)

    # Run multiple episodes and collect statistics
    n_episodes = 10
    episode_lengths = []
    episode_rewards = []
    success_count = 0

    print(f"Running {n_episodes} episodes to collect statistics...")

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 50

        while steps < max_steps:
            actions = np.random.randint(0, env.naction, size=env.npredator)
            obs, reward, done, info = env.step(actions)
            episode_reward += np.sum(reward[:env.npredator])
            steps += 1

            if done.any() or np.any(reward[:env.npredator] > 0):
                success_count += 1
                break

        episode_lengths.append(steps)
        episode_rewards.append(episode_reward)

    # Print statistics
    print(f"\nStatistics over {n_episodes} episodes:")
    print(f"  Success rate: {success_count}/{n_episodes} ({100 * success_count / n_episodes:.1f}%)")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Average total reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min/Max episode length: {np.min(episode_lengths)}/{np.max(episode_lengths)}")


def main():
    """Run all examples."""
    print("PREDATOR-PREY ENVIRONMENT EXAMPLES")
    print("===================================\n")

    # Run different examples
    basic_usage_example()
    # cooperative_mode_example()
    # competitive_mode_example()
    # enemy_communication_example()
    # smart_agent_example()
    statistics_example()

    # Rendering example (optional, might not work on all systems)
    try:
        rendering_example()
    except:
        print("\nSkipping rendering example (curses not available)")

    print("\n" + "=" * 50)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
