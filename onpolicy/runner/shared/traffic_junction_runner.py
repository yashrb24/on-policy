import time
from collections import defaultdict

import imageio
import numpy as np
import torch
import wandb

from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class TrafficJunctionRunner(Runner):
    """Runner class for Traffic Junction environment with comprehensive traffic-specific metrics tracking."""
    
    def __init__(self, config):
        super(TrafficJunctionRunner, self).__init__(config)
        self.env_infos = defaultdict(list)
        
        # Initialize episode trackers for each parallel environment
        self.episode_rewards = np.zeros(self.n_rollout_threads)
        self.episode_steps = np.zeros(self.n_rollout_threads, dtype=int)
        self.episodes_completed = 0  # Track total episodes completed
        
        # Traffic-specific trackers
        self.episode_cars_spawned = np.zeros(self.n_rollout_threads, dtype=int)
        self.episode_cars_completed = np.zeros(self.n_rollout_threads, dtype=int)
        self.episode_total_wait_time = np.zeros(self.n_rollout_threads)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Observe reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)
                
                # Check for timeout episodes at the last step
                if step == self.episode_length - 1:
                    # Handle episodes that didn't complete naturally
                    dones_env = np.all(dones, axis=-1)
                    for i in range(self.n_rollout_threads):
                        if not dones_env[i] and self.episode_steps[i] > 0:
                            # Episode timed out - record statistics
                            self.env_infos["episode_rewards"].append(self.episode_rewards[i] + np.sum(rewards[i]))
                            self.env_infos["episode_length"].append(self.episode_steps[i] + 1)
                            
                            # Extract traffic junction specific info
                            if len(infos) > i and infos[i]:
                                info = infos[i]  # This is the debug dict from environment
                                
                                # Note: Traffic Junction env needs to be modified to include stat info
                                # For now, we'll work with available debug info
                                
                                # Track cars completed
                                cars_completed = np.sum(info.get('is_completed', []))
                                self.env_infos["cars_completed"].append(cars_completed)
                                
                                # Calculate completion rate if we know how many cars spawned
                                if self.episode_cars_spawned[i] > 0:
                                    completion_rate = cars_completed / self.episode_cars_spawned[i]
                                    self.env_infos["completion_rate"].append(completion_rate)
                                
                                # Track waiting times
                                if 'wait' in info and 'alive_mask' in info:
                                    avg_wait = np.mean(info['wait'][info['alive_mask'] > 0]) if np.sum(info['alive_mask']) > 0 else 0
                                    self.env_infos["average_wait_time"].append(avg_wait)
                                
                                # Check success/crash from environment stats
                                success = info.get('success', 0)
                                self.env_infos["success_rate"].append(float(success))
                                self.env_infos["crash_rate"].append(1.0 - float(success))
                                
                                # Track add rate if available
                                if 'add_rate' in info:
                                    self.env_infos["current_add_rate"].append(info['add_rate'])
                            
                            # Reset trackers
                            self.episode_rewards[i] = 0.0
                            self.episode_steps[i] = 0
                            self.episode_cars_spawned[i] = 0
                            self.episode_cars_completed[i] = 0
                            self.episode_total_wait_time[i] = 0.0
                            self.episodes_completed += 1

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\\n"
                      .format(self.env_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # Log comprehensive traffic junction metrics
                if len(self.env_infos["episode_rewards"]) > 0:
                    train_infos["average_episode_rewards"] = np.mean(self.env_infos["episode_rewards"])
                    train_infos["average_episode_length"] = np.mean(self.env_infos["episode_length"])
                    train_infos["episodes_completed"] = self.episodes_completed
                    
                    # Success and crash metrics
                    if len(self.env_infos["success_rate"]) > 0:
                        train_infos["success_rate"] = np.mean(self.env_infos["success_rate"])
                        train_infos["crash_rate"] = np.mean(self.env_infos["crash_rate"])
                        train_infos["num_successes"] = np.sum(self.env_infos["success_rate"])
                    
                    # Traffic efficiency metrics
                    if len(self.env_infos["cars_completed"]) > 0:
                        train_infos["average_cars_completed"] = np.mean(self.env_infos["cars_completed"])
                        train_infos["total_cars_completed"] = np.sum(self.env_infos["cars_completed"])
                    
                    if len(self.env_infos["completion_rate"]) > 0:
                        train_infos["completion_rate"] = np.mean(self.env_infos["completion_rate"])
                    
                    if len(self.env_infos["average_wait_time"]) > 0:
                        train_infos["average_wait_time"] = np.mean(self.env_infos["average_wait_time"])
                    
                    # Curriculum tracking
                    if len(self.env_infos["current_add_rate"]) > 0:
                        train_infos["current_add_rate"] = np.mean(self.env_infos["current_add_rate"])
                    
                    # Calculate throughput (cars completed per timestep)
                    if len(self.env_infos["cars_completed"]) > 0 and len(self.env_infos["episode_length"]) > 0:
                        total_cars = np.sum(self.env_infos["cars_completed"])
                        total_steps = np.sum(self.env_infos["episode_length"])
                        train_infos["throughput"] = total_cars / total_steps if total_steps > 0 else 0.0
                else:
                    # No episodes completed in this logging interval
                    train_infos["average_episode_rewards"] = 0
                    train_infos["success_rate"] = 0
                    print("No episodes completed in this interval")
                
                # Print key metrics
                print(f"Episodes completed: {self.episodes_completed}")
                print(f"Success rate: {train_infos.get('success_rate', 0):.2%}")
                print(f"Average episode rewards: {train_infos.get('average_episode_rewards', 0):.3f}")
                print(f"Average cars completed: {train_infos.get('average_cars_completed', 0):.1f}")
                if 'completion_rate' in train_infos:
                    print(f"Completion rate: {train_infos['completion_rate']:.2%}")
                if 'throughput' in train_infos:
                    print(f"Throughput (cars/step): {train_infos['throughput']:.4f}")
                if 'current_add_rate' in train_infos:
                    print(f"Current add rate: {train_infos['current_add_rate']:.3f}")
                
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        # insert obs to buffer
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # get environment-level dones
        dones_env = np.all(dones, axis=-1)
        
        # Accumulate rewards for each environment and track episode completion
        for i in range(self.n_rollout_threads):
            # Add current step rewards
            self.episode_rewards[i] += np.sum(rewards[i])
            self.episode_steps[i] += 1
            
            # Track cars in system and other traffic metrics
            if len(infos) > i and infos[i]:
                info = infos[i]
                
                # Track cars spawned (increment when new cars enter)
                cars_in_sys = info.get('cars_in_sys', 0)
                if hasattr(self, '_prev_cars_in_sys'):
                    if cars_in_sys > self._prev_cars_in_sys[i]:
                        self.episode_cars_spawned[i] += (cars_in_sys - self._prev_cars_in_sys[i])
                    self._prev_cars_in_sys[i] = cars_in_sys
                else:
                    self._prev_cars_in_sys = np.zeros(self.n_rollout_threads, dtype=int)
                    self._prev_cars_in_sys[i] = cars_in_sys
                
                # Track wait times
                if 'wait' in info and 'alive_mask' in info:
                    active_wait_times = info['wait'][info['alive_mask'] > 0]
                    if len(active_wait_times) > 0:
                        self.episode_total_wait_time[i] += np.sum(active_wait_times)
            
            # Check if episode ended
            if dones_env[i]:
                # Record episode statistics when episode completes
                self.env_infos["episode_rewards"].append(self.episode_rewards[i])
                self.env_infos["episode_length"].append(self.episode_steps[i])
                
                # Extract traffic junction specific info  
                if len(infos) > i and infos[i]:
                    info = infos[i]  # This is the debug dict from environment
                    
                    # Cars completed tracking
                    cars_completed = np.sum(info.get('is_completed', []))
                    self.env_infos["cars_completed"].append(cars_completed)
                    self.episode_cars_completed[i] = cars_completed
                    
                    # Completion rate
                    if self.episode_cars_spawned[i] > 0:
                        completion_rate = cars_completed / self.episode_cars_spawned[i]
                        self.env_infos["completion_rate"].append(completion_rate)
                    
                    # Average wait time for this episode
                    if self.episode_cars_spawned[i] > 0:
                        avg_wait = self.episode_total_wait_time[i] / self.episode_cars_spawned[i]
                        self.env_infos["average_wait_time"].append(avg_wait)
                    
                    # Success/crash tracking using environment stats
                    success = info.get('success', 0)
                    self.env_infos["success_rate"].append(float(success))
                    self.env_infos["crash_rate"].append(1.0 - float(success))
                    
                    # Track add rate if available
                    if 'add_rate' in info:
                        self.env_infos["current_add_rate"].append(info['add_rate'])
                
                # Reset trackers for next episode
                self.episode_rewards[i] = 0.0
                self.episode_steps[i] = 0
                self.episode_cars_spawned[i] = 0
                self.episode_cars_completed[i] = 0
                self.episode_total_wait_time[i] = 0.0
                self.episodes_completed += 1

        # reset rnn and mask args for done envs
        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # prepare shared obs
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            rnn_states_actor=rnn_states,
            rnn_states_critic=rnn_states_critic,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards,
            masks=masks
        )

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # init eval metrics
        num_done = 0
        eval_success_rates = np.zeros(self.all_args.eval_episodes)
        eval_episode_rewards = np.zeros(self.all_args.eval_episodes)
        eval_cars_completed = np.zeros(self.all_args.eval_episodes)
        
        step = 0
        quo = self.all_args.eval_episodes // self.n_eval_rollout_threads
        rem = self.all_args.eval_episodes % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

        # loop until enough episodes
        while num_done < self.all_args.eval_episodes and step < self.episode_length:
            # get actions
            self.trainer.prep_rollout()

            # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=self.all_args.eval_deterministic
            )

            # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_actions_env = [eval_actions[idx, :, 0] for idx in range(self.n_eval_rollout_threads)]

            # step
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            # update goals if done
            eval_dones_env = np.all(eval_dones, axis=-1)
            eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
            if np.any(eval_dones_unfinished_env):
                for idx_env in range(self.n_eval_rollout_threads):
                    if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                        # Record evaluation metrics
                        if len(eval_infos) > idx_env and eval_infos[idx_env]:
                            info = eval_infos[idx_env]
                            eval_success_rates[num_done] = float(info.get('success', 0))
                            eval_cars_completed[num_done] = np.sum(info.get('is_completed', []))
                        
                        eval_episode_rewards[num_done] = np.sum(eval_rewards[idx_env])
                        
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

            # reset rnn and masks for done envs
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)
            step += 1

        # calculate evaluation metrics
        eval_success_rate = np.mean(eval_success_rates)
        eval_avg_reward = np.mean(eval_episode_rewards)
        eval_avg_cars_completed = np.mean(eval_cars_completed)

        # log and print
        print(f"Evaluation over {self.all_args.eval_episodes} episodes:")
        print(f"  Success rate: {eval_success_rate:.2%}")
        print(f"  Average reward: {eval_avg_reward:.3f}")
        print(f"  Average cars completed: {eval_avg_cars_completed:.1f}")

        if self.use_wandb:
            wandb.log({"eval_success_rate": eval_success_rate}, step=total_num_steps)
            wandb.log({"eval_average_reward": eval_avg_reward}, step=total_num_steps)
            wandb.log({"eval_cars_completed": eval_avg_cars_completed}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_success_rate", {"eval_success_rate": eval_success_rate}, total_num_steps)
            self.writter.add_scalars("eval_average_reward", {"eval_average_reward": eval_avg_reward}, total_num_steps)
            self.writter.add_scalars("eval_cars_completed", {"eval_cars_completed": eval_avg_cars_completed}, total_num_steps)

    @torch.no_grad()
    def render(self):
        # reset envs and init rnn and mask
        render_env = self.envs

        # init goal
        render_rewards = np.zeros(self.all_args.render_episodes)
        for i_episode in range(self.all_args.render_episodes):
            render_obs = render_env.reset()
            render_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                         dtype=np.float32)
            render_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            if self.all_args.save_gifs:
                frames = []
                # Traffic junction doesn't have frame observation, so we'll skip gif saving
                print("GIF saving not supported for Traffic Junction environment")

            render_dones = False
            step_count = 0
            while not np.any(render_dones) and step_count < self.episode_length:
                self.trainer.prep_rollout()
                render_actions, render_rnn_states = self.trainer.policy.act(
                    np.concatenate(render_obs),
                    np.concatenate(render_rnn_states),
                    np.concatenate(render_masks),
                    deterministic=True
                )

                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                render_actions = np.array(np.split(_t2n(render_actions), self.n_rollout_threads))
                render_rnn_states = np.array(np.split(_t2n(render_rnn_states), self.n_rollout_threads))

                render_actions_env = [render_actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

                # step
                render_obs, render_rewards, render_dones, render_infos = render_env.step(render_actions_env)
                render_dones = np.all(render_dones, axis=-1)
                step_count += 1

            # print episode statistics
            total_reward = np.sum(render_rewards[0])
            success = render_infos[0].get('success', 0) if render_infos and render_infos[0] else 0
            cars_completed = np.sum(render_infos[0].get('is_completed', [])) if render_infos and render_infos[0] else 0
            
            print(f"Episode {i_episode}: Reward={total_reward:.3f}, Success={success}, Cars completed={cars_completed}")
            render_rewards[i_episode] = total_reward

        print(f"Average reward over {self.all_args.render_episodes} episodes: {np.mean(render_rewards):.3f}")