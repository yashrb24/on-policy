import time
from collections import defaultdict

import numpy as np
import torch
import wandb

from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class PursuitRunner(Runner):
    def __init__(self, config):
        super(PursuitRunner, self).__init__(config)
        self.env_infos = defaultdict(list)
        self.use_transformer_base_critic = getattr(self.all_args, "use_transformer_base_critic", False)

        # Per-rollout-thread episode trackers
        self.episode_rewards = np.zeros(self.n_rollout_threads)
        self.episode_steps = np.zeros(self.n_rollout_threads, dtype=int)
        self.episodes_completed = 0

    def _build_share_obs(self, obs):
        if self.use_centralized_V and not self.use_transformer_base_critic:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            return np.broadcast_to(
                np.expand_dims(share_obs, 1),
                (self.n_rollout_threads, self.num_agents, share_obs.shape[-1]))
        return obs

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

                # Step envs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log
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

                if len(self.env_infos["episode_rewards"]) > 0:
                    episodes_in_interval = len(self.env_infos["episode_rewards"])
                    train_infos["average_episode_rewards"] = np.mean(self.env_infos["episode_rewards"])
                    train_infos["average_episode_length"] = np.mean(self.env_infos["episode_length"])
                    train_infos["episodes_completed"] = self.episodes_completed
                    train_infos["episodes_in_interval"] = episodes_in_interval

                    if len(self.env_infos["success"]) > 0:
                        train_infos["success"] = np.mean(self.env_infos["success"])
                    if len(self.env_infos["truncated"]) > 0:
                        train_infos["truncated"] = np.mean(self.env_infos["truncated"])
                    if len(self.env_infos["n_captures"]) > 0:
                        train_infos["n_captures"] = np.mean(self.env_infos["n_captures"])
                        train_infos["capture_rate"] = np.mean(self.env_infos["capture_rate"])
                else:
                    episodes_in_interval = 0
                    train_infos["average_episode_rewards"] = 0
                    train_infos["success"] = 0
                    train_infos["episodes_in_interval"] = episodes_in_interval
                    print("No episodes completed in this interval")

                print(f"Episodes completed (total): {self.episodes_completed}")
                if episodes_in_interval > 0:
                    print(f"Episodes in this interval: {episodes_in_interval}")
                    print(f"Success rate (all evaders captured): "
                          f"{train_infos.get('success', 0):.2%}")
                    print(f"Average episode rewards: "
                          f"{train_infos.get('average_episode_rewards', 0):.3f}")
                    if 'n_captures' in train_infos:
                        print(f"Average captures per episode: "
                              f"{train_infos['n_captures']:.2f} / {self.all_args.n_evaders} "
                              f"({train_infos['capture_rate']:.2%})")

                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs = self.envs.reset()

        share_obs = self._build_share_obs(obs)

        self.buffer.share_obs[0] = share_obs
        self.buffer.obs[0] = obs

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

        values = _t2n(values)
        actions = _t2n(actions)
        action_log_probs = _t2n(action_log_probs)
        rnn_states = _t2n(rnn_states)
        rnn_states_critic = _t2n(rnn_states_critic)
        values = values.reshape(self.n_rollout_threads, -1, *values.shape[1:])
        actions = actions.reshape(self.n_rollout_threads, -1, *actions.shape[1:])
        action_log_probs = action_log_probs.reshape(self.n_rollout_threads, -1, *action_log_probs.shape[1:])
        rnn_states = rnn_states.reshape(self.n_rollout_threads, -1, *rnn_states.shape[1:])
        rnn_states_critic = rnn_states_critic.reshape(self.n_rollout_threads, -1, *rnn_states_critic.shape[1:])

        actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=-1)

        self.episode_rewards += rewards.mean(axis=(1, 2))
        self.episode_steps += 1
        for i in np.flatnonzero(dones_env):
            self.env_infos["episode_rewards"].append(self.episode_rewards[i])
            self.env_infos["episode_length"].append(self.episode_steps[i])
            self.env_infos["success"].append(float(infos[i].get('success', 0.0)))
            self.env_infos["truncated"].append(float(infos[i].get('truncated', 0.0)))
            if 'n_captures_so_far' in infos[i]:
                n_cap = int(infos[i]['n_captures_so_far'])
                self.env_infos["n_captures"].append(n_cap)
                self.env_infos["capture_rate"].append(
                    n_cap / max(1, self.all_args.n_evaders))
            self.episode_rewards[i] = 0.0
            self.episode_steps[i] = 0
            self.episodes_completed += 1

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        share_obs = self._build_share_obs(obs)

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
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_episode_rewards = np.zeros(self.n_eval_rollout_threads)
        eval_finished_rewards = []
        eval_finished_success = []
        eval_finished_captures = []

        step = 0
        while step < self.episode_length:
            self.trainer.prep_rollout()

            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=self.all_args.eval_deterministic
            )

            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_actions_env = [eval_actions[idx, :, 0] for idx in range(self.n_eval_rollout_threads)]

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards += eval_rewards.mean(axis=(1, 2))

            eval_dones_env = np.all(eval_dones, axis=-1)
            for i in np.flatnonzero(eval_dones_env):
                eval_finished_rewards.append(eval_episode_rewards[i])
                eval_finished_success.append(float(eval_infos[i].get('success', 0.0)))
                if 'n_captures_so_far' in eval_infos[i]:
                    eval_finished_captures.append(int(eval_infos[i]['n_captures_so_far']))
                eval_episode_rewards[i] = 0.0

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            step += 1

        eval_reward = float(np.mean(eval_finished_rewards)) if eval_finished_rewards else 0.0
        eval_success = float(np.mean(eval_finished_success)) if eval_finished_success else 0.0
        eval_captures = float(np.mean(eval_finished_captures)) if eval_finished_captures else 0.0
        eval_capture_rate = eval_captures / max(1, self.all_args.n_evaders)

        # eval_success is the Done% (fraction of episodes capturing all evaders); log a clearly-named
        # alias eval_done_rate alongside it. eval_capture_rate is the Capture%.
        if self.use_wandb:
            wandb.log({"eval_average_episode_rewards": eval_reward}, step=total_num_steps)
            wandb.log({"eval_success": eval_success}, step=total_num_steps)
            wandb.log({"eval_done_rate": eval_success}, step=total_num_steps)
            wandb.log({"eval_n_captures": eval_captures}, step=total_num_steps)
            wandb.log({"eval_capture_rate": eval_capture_rate}, step=total_num_steps)
            wandb.log({"eval_episodes": len(eval_finished_captures)}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_average_episode_rewards",
                                     {"eval_average_episode_rewards": eval_reward}, total_num_steps)
            self.writter.add_scalars("eval_success", {"eval_success": eval_success}, total_num_steps)
            self.writter.add_scalars("eval_done_rate", {"eval_done_rate": eval_success}, total_num_steps)
            self.writter.add_scalars("eval_n_captures", {"eval_n_captures": eval_captures}, total_num_steps)
            self.writter.add_scalars("eval_capture_rate", {"eval_capture_rate": eval_capture_rate}, total_num_steps)
            self.writter.add_scalars("eval_episodes", {"eval_episodes": len(eval_finished_captures)}, total_num_steps)
