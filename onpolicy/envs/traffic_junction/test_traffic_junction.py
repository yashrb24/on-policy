#!/usr/bin/env python
"""
Traffic Junction Environment Test Script

A comprehensive test script to demonstrate the Traffic Junction environment's
rendering capabilities and basic training functionality.

Features:
- Basic environment operations testing
- Curses-based visual rendering
- Minimal training loop with metrics tracking
- Configurable parameters for quick testing

Usage:
    python test_traffic_junction.py [options]
"""

import argparse
import curses
import time
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from onpolicy.envs.traffic_junction.TrafficJunction_Env import TrafficJunctionEnv


class SimpleArgs:
    """Simple configuration class mimicking argparse results."""
    
    def __init__(self, **kwargs):
        # Traffic Junction specific parameters
        self.num_agents = kwargs.get('num_agents', 3)
        self.nagents = self.num_agents  # Alias for compatibility
        self.dim = kwargs.get('dim', 6)
        self.vision = kwargs.get('vision', 1)
        self.difficulty = kwargs.get('difficulty', 'easy')
        self.vocab_type = kwargs.get('vocab_type', 'bool')
        
        # Curriculum parameters
        self.add_rate_min = kwargs.get('add_rate_min', 0.05)
        self.add_rate_max = kwargs.get('add_rate_max', 0.2)
        self.curr_start = kwargs.get('curr_start', 0)
        self.curr_end = kwargs.get('curr_end', 0)
        
        # Training parameters (not used in basic testing)
        self.seed = kwargs.get('seed', 42)
        self.episode_length = kwargs.get('episode_length', 100)


def test_environment_basics():
    """Test basic environment operations without rendering."""
    print("=== Testing Basic Environment Operations ===")
    
    args = SimpleArgs(num_agents=2, dim=6, difficulty='easy')
    env = TrafficJunctionEnv(args)
    
    print(f"Environment created with {args.num_agents} agents on {args.dim}x{args.dim} grid")
    print(f"Difficulty: {args.difficulty}")
    print(f"Action space: {[space.n for space in env.action_space]}")
    print(f"Number of possible paths: {env.npath}")
    
    # Test reset
    obs = env.reset()
    print(f"Environment reset. Observation shape: {len(obs)} agents")
    print(f"Observation sample shape for agent 0: {[np.array(o).shape for o in obs[0]]}")
    
    # Test a few random steps
    print("\nTesting random actions for 10 steps...")
    total_reward = np.zeros(args.num_agents)
    
    for step in range(10):
        # Random actions: 0 = GAS, 1 = BRAKE
        actions = [np.random.choice([0, 1]) for _ in range(args.num_agents)]
        
        obs, rewards, done, info = env.step(actions)
        total_reward += rewards.mean(axis=-1)
        
        print(f"Step {step + 1}: Actions={actions}, Rewards={rewards}, "
              f"Cars in system={info.get('cars_in_sys', 0)}")
        
        if done:
            print("Episode finished!")
            break
    
    print(f"Final total rewards: {total_reward}")
    print(f"Episode statistics: Success={info.get('success', 'N/A')}, "
          f"Add rate={info.get('add_rate', 'N/A')}")
    
    return env


def test_rendering(duration=30):
    """Test the curses-based rendering system."""
    print(f"\n=== Testing Rendering (running for {duration} seconds) ===")
    print("Press 'q' to quit early, or wait for automatic timeout")
    
    args = SimpleArgs(num_agents=3, dim=8, difficulty='medium', vision=1)
    env = TrafficJunctionEnv(args)
    
    try:
        # Initialize curses
        env.init_curses()
        env.stdscr.timeout(100)  # Non-blocking input with 100ms timeout
        env.stdscr.nodelay(True)
        
        obs = env.reset()
        
        start_time = time.time()
        step_count = 0
        episode_count = 1
        
        # Statistics tracking
        total_cars_completed = 0
        total_crashes = 0
        episode_rewards = np.zeros(args.num_agents)
        
        # Per-episode tracking
        episode_has_crashed = False
        episode_wait_times = []
        
        while time.time() - start_time < duration:
            # Check for user input
            try:
                key = env.stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    break
            except:
                pass
            
            # Take random actions with some intelligence
            # 70% chance of GAS, 30% chance of BRAKE for more movement
            actions = [np.random.choice([0, 1], p=[0.7, 0.3]) for _ in range(args.num_agents)]
            
            obs, rewards, done, info = env.step(actions)
            episode_rewards += rewards
            
            # Track cars completed (per step)
            if info.get('is_completed') is not None:
                total_cars_completed += np.sum(info['is_completed'])
            
            # Calculate total cars spawned as: completed + currently active
            # This works because: spawned = completed + still_in_system
            cars_in_sys = info.get('cars_in_sys', 0)
            total_cars_spawned = total_cars_completed + cars_in_sys
            
            # Track crashes (only once per episode when first crash occurs)
            if not episode_has_crashed and info.get('success') == 0:
                total_crashes += 1
                episode_has_crashed = True
            
            # Track wait times for active cars
            if 'wait' in info and 'alive_mask' in info:
                active_wait = info['wait'][info['alive_mask'] > 0]
                if len(active_wait) > 0:
                    episode_wait_times.extend(active_wait)
            
            # Render the environment
            env.render()
            
            # Display statistics on screen
            stats_y = env.dim + 2
            completion_rate = (total_cars_completed / total_cars_spawned * 100) if total_cars_spawned > 0 else 0
            # Ensure completion rate doesn't exceed 100%
            completion_rate = min(completion_rate, 100.0)
            avg_wait = np.mean(episode_wait_times) if episode_wait_times else 0
            
            env.stdscr.addstr(stats_y, 0, f"Episode: {episode_count} | Step: {step_count} | "
                                         f"Cars in system: {cars_in_sys}")
            env.stdscr.addstr(stats_y + 1, 0, f"Spawned: {total_cars_spawned} | "
                                             f"Completed: {total_cars_completed} | "
                                             f"Completion: {completion_rate:.1f}%")
            env.stdscr.addstr(stats_y + 2, 0, f"Crashes: {total_crashes} | "
                                             f"Avg wait: {avg_wait:.1f} | "
                                             f"Add rate: {info.get('add_rate', 0):.3f}")
            env.stdscr.addstr(stats_y + 3, 0, f"Episode rewards: {episode_rewards}")
            env.stdscr.addstr(stats_y + 4, 0, f"Time remaining: {duration - (time.time() - start_time):.1f}s | "
                                             f"Press 'q' to quit")
            
            env.stdscr.refresh()
            
            step_count += 1
            
            if done:
                # Reset for next episode
                obs = env.reset()
                episode_count += 1
                episode_rewards = np.zeros(args.num_agents)
                step_count = 0
                # Reset episode-specific tracking
                episode_has_crashed = False
                episode_wait_times = []
            
            time.sleep(0.1)  # Small delay for better visualization
        
        # Final statistics
        final_completion_rate = (total_cars_completed / total_cars_spawned * 100) if total_cars_spawned > 0 else 0
        final_completion_rate = min(final_completion_rate, 100.0)
        env.stdscr.addstr(stats_y + 6, 0, f"Final Stats - Episodes: {episode_count} | "
                                         f"Spawned: {total_cars_spawned} | "
                                         f"Completed: {total_cars_completed}")
        env.stdscr.addstr(stats_y + 7, 0, f"Completion Rate: {final_completion_rate:.1f}% | "
                                         f"Episode Crashes: {total_crashes}")
        env.stdscr.addstr(stats_y + 8, 0, "Press any key to continue...")
        env.stdscr.nodelay(False)
        env.stdscr.getch()
        
    except Exception as e:
        print(f"Rendering error: {e}")
    finally:
        try:
            env.exit_render()
        except:
            pass
    
    print("Rendering test completed!")


def test_training(num_steps=1000):
    """Test a minimal training loop with metrics tracking."""
    print(f"\n=== Testing Training Loop ({num_steps} steps) ===")
    
    args = SimpleArgs(num_agents=3, dim=6, difficulty='easy', episode_length=50)
    env = TrafficJunctionEnv(args)
    
    # Training statistics
    episode_rewards = []
    success_rates = []
    cars_completed = []
    cars_spawned_list = []
    completion_rates = []
    episode_lengths = []
    wait_times = []
    
    # Current episode tracking
    current_episode_reward = np.zeros(args.num_agents)
    current_episode_length = 0
    current_episode_cars_completed = 0
    current_episode_wait_times = []
    episodes_completed = 0
    
    # Per-step tracking
    episode_has_crashed = False
    
    obs = env.reset()
    
    print("Starting training loop...")
    print("Step | Episode | Spawned | Completed | Success | Avg Reward | Completion %")
    print("-" * 75)
    
    for step in range(num_steps):
        # Simple policy: mostly GAS with some BRAKE
        actions = [np.random.choice([0, 1], p=[0.8, 0.2]) for _ in range(args.num_agents)]
        
        obs, rewards, done, info = env.step(actions)
        
        current_episode_reward += rewards
        current_episode_length += 1
        
        # Track cars completed during this step
        if info.get('is_completed') is not None:
            current_episode_cars_completed += np.sum(info['is_completed'])
        
        # Track wait times for active cars
        if 'wait' in info and 'alive_mask' in info:
            active_wait = info['wait'][info['alive_mask'] > 0]
            if len(active_wait) > 0:
                current_episode_wait_times.extend(active_wait)
        
        # Track crashes (only once per episode)
        if not episode_has_crashed and info.get('success') == 0:
            episode_has_crashed = True
        
        if done or current_episode_length >= args.episode_length:
            # Episode completed
            episodes_completed += 1
            
            # Record episode statistics
            episode_rewards.append(np.mean(current_episode_reward))
            episode_lengths.append(current_episode_length)
            
            # Traffic-specific metrics
            success = info.get('success', 0)
            success_rates.append(success)
            
            # Record cars completed during this episode
            cars_completed.append(current_episode_cars_completed)
            
            # Calculate total cars spawned as: completed + currently active
            cars_in_sys = info.get('cars_in_sys', 0)
            current_episode_cars_spawned = current_episode_cars_completed + cars_in_sys
            cars_spawned_list.append(current_episode_cars_spawned)
            
            # Calculate completion rate (with validation)
            completion_rate = (current_episode_cars_completed / current_episode_cars_spawned * 100) if current_episode_cars_spawned > 0 else 0
            # Ensure it doesn't exceed 100% due to any edge cases
            completion_rate = min(completion_rate, 100.0)
            completion_rates.append(completion_rate)
            
            # Record wait times
            avg_wait = np.mean(current_episode_wait_times) if current_episode_wait_times else 0
            wait_times.append(avg_wait)
            
            # Print progress every 10 episodes
            if episodes_completed % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                avg_success = np.mean(success_rates[-10:]) if len(success_rates) >= 10 else np.mean(success_rates)
                avg_cars_spawned = np.mean(cars_spawned_list[-10:]) if len(cars_spawned_list) >= 10 else np.mean(cars_spawned_list)
                avg_cars_completed = np.mean(cars_completed[-10:]) if len(cars_completed) >= 10 else np.mean(cars_completed)
                avg_completion_rate = np.mean(completion_rates[-10:]) if len(completion_rates) >= 10 else np.mean(completion_rates)
                
                print(f"{step:4d} | {episodes_completed:7d} | {avg_cars_spawned:7.1f} | "
                      f"{avg_cars_completed:9.1f} | {avg_success:7.1%} | "
                      f"{avg_reward:10.3f} | {avg_completion_rate:11.1f}%")
            
            # Reset environment
            obs = env.reset()
            current_episode_reward = np.zeros(args.num_agents)
            current_episode_length = 0
            # Reset episode-specific tracking
            current_episode_cars_completed = 0
            current_episode_wait_times = []
            episode_has_crashed = False
    
    # Final statistics
    print("\n=== Training Summary ===")
    print(f"Episodes completed: {episodes_completed}")
    print(f"Average episode reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Success rate: {np.mean(success_rates):.2%}")
    print(f"Average cars spawned per episode: {np.mean(cars_spawned_list):.2f}")
    print(f"Average cars completed per episode: {np.mean(cars_completed):.2f}")
    print(f"Average completion rate: {np.mean(completion_rates):.1f}%")
    print(f"Average wait time: {np.mean(wait_times):.1f} steps")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Throughput: {np.sum(cars_completed) / num_steps:.4f} cars/step")
    print(f"Total cars spawned: {np.sum(cars_spawned_list)}")
    print(f"Total cars completed: {np.sum(cars_completed)}")
    print(f"Overall completion rate: {np.sum(cars_completed) / np.sum(cars_spawned_list) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Traffic Junction Environment Test Script")
    parser.add_argument('--test', choices=['basic', 'render', 'train', 'all'], 
                       default='all', help='Which test to run')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--dim', type=int, default=6, help='Grid dimension')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], 
                       default='easy', help='Environment difficulty')
    parser.add_argument('--render-time', type=int, default=30, 
                       help='Rendering test duration in seconds')
    parser.add_argument('--train-steps', type=int, default=1000, 
                       help='Number of training steps')
    
    args = parser.parse_args()
    
    print("Traffic Junction Environment Test Script")
    print("=" * 50)
    print(f"Configuration: {args.agents} agents, {args.dim}x{args.dim} grid, {args.difficulty} difficulty")
    
    try:
        if args.test in ['basic', 'all']:
            test_environment_basics()
        
        if args.test in ['render', 'all']:
            test_rendering(duration=args.render_time)
        
        if args.test in ['train', 'all']:
            test_training(num_steps=args.train_steps)
            
        print("\nAll tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure curses is properly cleaned up
        try:
            curses.endwin()
        except:
            pass


if __name__ == "__main__":
    main()