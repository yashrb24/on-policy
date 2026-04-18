import time
from collections import defaultdict

import imageio
import numpy as np
import torch
import wandb

from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class PredatorPreyRunner(Runner):
    def __init__(self, config):
        super(PredatorPreyRunner, self).__init__(config)
        self.env_infos = defaultdict(list)
        
        # Initialize episode trackers for each parallel environment
        self.episode_rewards = np.zeros(self.n_rollout_threads)
        self.episode_steps = np.zeros(self.n_rollout_threads, dtype=int)
        self.episodes_completed = 0  # Track total episodes completed

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

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)
                
                # Check for timeout episodes at the last step
                if step == self.episode_length - 1:
                    # Handle episodes that didn't complete naturally
                    dones_env = np.all(dones, axis=-1)
                    for i in range(self.n_rollout_threads):
                        # Only count as failure if episode has run for full duration (episode_steps will be 39 after 40 steps: 0-39)
                        if not dones_env[i] and self.episode_steps[i] >= self.episode_length - 1:
                            # Episode timed out without success after running full duration - this is a failure in mixed mode
                            self.env_infos["episode_rewards"].append(self.episode_rewards[i] + np.sum(rewards[i]))  # Add final step reward
                            self.env_infos["episode_length"].append(self.episode_steps[i] + 1)  # Include final step (will be 40)
                            self.env_infos["win_rate"].append(0.0)  # Failed episode
                            
                            # Track partial progress
                            if 'predators_on_prey' in infos[i]:
                                partial = infos[i]['predators_on_prey'] / self.num_agents
                                self.env_infos["partial_success_rate"].append(partial)
                            
                            # Reset trackers
                            self.episode_rewards[i] = 0.0
                            self.episode_steps[i] = 0
                            self.episodes_completed += 1
                        # Episodes that haven't run full duration are ignored (not counted as failures)

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
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.env_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # Use properly tracked episode rewards
                if len(self.env_infos["episode_rewards"]) > 0:
                    episodes_in_interval = len(self.env_infos["episode_rewards"])
                    train_infos["average_episode_rewards"] = np.mean(self.env_infos["episode_rewards"])
                    train_infos["average_episode_length"] = np.mean(self.env_infos["episode_length"])
                    train_infos["episodes_completed"] = self.episodes_completed
                    train_infos["episodes_in_interval"] = episodes_in_interval
                    
                    # Win rate is now correctly calculated from completed episodes only
                    if len(self.env_infos["win_rate"]) > 0:
                        train_infos["win_rate"] = np.mean(self.env_infos["win_rate"])
                        train_infos["num_wins"] = np.sum(self.env_infos["win_rate"])
                        
                    if len(self.env_infos["success_steps"]) > 0:
                        train_infos["average_win_steps"] = np.mean(self.env_infos["success_steps"])
                    
                    if len(self.env_infos["partial_success_rate"]) > 0:
                        train_infos["partial_success_rate"] = np.mean(self.env_infos["partial_success_rate"])
                else:
                    # No episodes completed in this logging interval
                    episodes_in_interval = 0
                    train_infos["average_episode_rewards"] = 0
                    train_infos["win_rate"] = 0
                    train_infos["episodes_in_interval"] = episodes_in_interval
                    print("No episodes completed in this interval")
                
                print(f"Episodes completed (total): {self.episodes_completed}")
                if episodes_in_interval > 0:
                    print(f"Episodes in this interval: {episodes_in_interval}")
                    print(f"Win rate: {train_infos.get('win_rate', 0):.2%} (based on {episodes_in_interval} episodes)")
                    print(f"Average episode rewards: {train_infos.get('average_episode_rewards', 0):.3f} (based on {episodes_in_interval} episodes)")
                    if 'average_win_steps' in train_infos:
                        print(f"Average steps to win: {train_infos['average_win_steps']:.1f} (based on {len(self.env_infos['success_steps'])} winning episodes)")
                
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
            self.episode_rewards[i] += np.mean(rewards[i]) # TODO: only works for mixed case
            self.episode_steps[i] += 1
            
            # Check if episode ended
            if dones_env[i]:
                # Record episode statistics ONLY when episode completes
                self.env_infos["episode_rewards"].append(self.episode_rewards[i])
                self.env_infos["episode_length"].append(self.episode_steps[i])
                
                # Record win rate based on mode
                if self.all_args.mode == 'mixed':
                    # Mixed mode: episode ends only on success
                    self.env_infos["win_rate"].append(1.0)
                    self.env_infos["success_steps"].append(self.episode_steps[i])
                else:
                    # Other modes: check info dict for success
                    success = infos[i].get('success', 0)
                    self.env_infos["win_rate"].append(float(success))
                    if success:
                        self.env_infos["success_steps"].append(self.episode_steps[i])
                
                # Track partial success if available
                if 'predators_on_prey' in infos[i]:
                    partial = infos[i]['predators_on_prey'] / self.num_agents
                    self.env_infos["partial_success_rate"].append(partial)
                
                # Reset trackers for next episode
                self.episode_rewards[i] = 0.0
                self.episode_steps[i] = 0
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

        # init eval goals
        num_done = 0
        eval_win_rates = np.zeros(self.all_args.eval_episodes)
        eval_steps = np.zeros(self.all_args.eval_episodes)
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
                        # eval_win_rates[num_done] = int(eval_dones_env[idx_env])
                        # eval_steps[num_done] = eval_infos[idx_env]["max_steps"] - eval_infos[idx_env]["steps_left"]
                        # print("episode {:>2d} done by env {:>2d}: {}".format(num_done, idx_env, eval_infos[idx_env]["score_reward"]))
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

            # reset rnn and masks for done envs
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)
            step += 1

        # get expected goal
        eval_win_rate = np.mean(eval_win_rates)
        eval_step = np.mean(eval_steps)

        # log and print
        if self.use_wandb:
            wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
            wandb.log({"eval_step": eval_step}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
            self.writter.add_scalars("eval_step", {"expected_step": eval_step}, total_num_steps)

    @torch.no_grad()
    def render(self):
        # reset envs and init rnn and mask
        render_env = self.envs

        # init goal
        render_goals = np.zeros(self.all_args.render_episodes)
        for i_episode in range(self.all_args.render_episodes):
            render_obs = render_env.reset()
            render_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                         dtype=np.float32)
            render_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            if self.all_args.save_gifs:
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            render_dones = False
            while not np.any(render_dones):
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

                # append frame
                if self.all_args.save_gifs:
                    image = render_infos[0]["frame"]
                    frames.append(image)

            # print goal
            render_goals[i_episode] = render_rewards[0, 0]
            print("goal in episode {}: {}".format(i_episode, render_rewards[0, 0]))

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )

        print("expected goal: {}".format(np.mean(render_goals)))
