import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)
        
        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], self.num_agents, device = self.device)
        else:
            self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device = self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)
        else:
            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        # value_normalizer lives on the trainer; restore() runs before the trainer exists.
        if self.model_dir is not None:
            self.restore_value_normalizer(self.model_dir)

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

        # Per-update DDCL delta metrics; populated by train(), consumed by log_train().
        self._delta_info = {}

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        # Critic explained variance on this rollout, measured BEFORE trainer.train()
        # shifts the valuenorm stats. value_preds are stored normalized; returns are
        # raw (denormalized in compute_returns), so denormalize preds to match.
        vn = getattr(self.trainer, "value_normalizer", None)
        y_pred = self.buffer.value_preds[:-1]
        if vn is not None:
            y_pred = vn.denormalize(y_pred)
        y_true = self.buffer.returns[:-1]
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else float(1.0 - np.var(y_true - y_pred) / var_y)
        result = self.trainer.train(self.buffer)
        train_infos, self._delta_info = result
        train_infos["explained_variance"] = explained_var
        self.buffer.after_update()
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(self.save_dir, episode)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
            # Persist optimizer + value-normalizer state so a run can be resumed cleanly
            torch.save(self.trainer.policy.actor_optimizer.state_dict(), str(self.save_dir) + "/actor_optimizer.pt")
            torch.save(self.trainer.policy.critic_optimizer.state_dict(), str(self.save_dir) + "/critic_optimizer.pt")
            if self.all_args.use_valuenorm and not self.all_args.use_popart \
                    and getattr(self.trainer, "value_normalizer", None) is not None:
                torch.save(self.trainer.value_normalizer.state_dict(), str(self.save_dir) + "/valuenorm.pt")

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(model_dir)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
                self.policy.critic.load_state_dict(policy_critic_state_dict)
                # Restore optimizer state if present 
                actor_opt_path = str(self.model_dir) + '/actor_optimizer.pt'
                critic_opt_path = str(self.model_dir) + '/critic_optimizer.pt'
                if os.path.exists(actor_opt_path):
                    self.policy.actor_optimizer.load_state_dict(torch.load(actor_opt_path))
                if os.path.exists(critic_opt_path):
                    self.policy.critic_optimizer.load_state_dict(torch.load(critic_opt_path))

    def restore_value_normalizer(self, model_dir):
        """Restore the standalone ValueNorm running statistics, if saved.

        Called after the trainer is built (the value_normalizer lives there).
        and a no-op under PopArt (its normalizer is restored via critic.pt).
        """
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            return
        vn_path = str(model_dir) + '/valuenorm.pt'
        if getattr(self.trainer, "value_normalizer", None) is not None and os.path.exists(vn_path):
            self.trainer.value_normalizer.load_state_dict(torch.load(vn_path))

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

        # Log learnable delta values: one grouped chart per channel (key / out), per block
        for k, v in self._delta_info.items():
            if isinstance(v, torch.Tensor) and v.numel() > 1:
                dim_dict = {f"d{i}": float(v[i]) for i in range(v.numel())}
            else:
                dim_dict = {"d0": float(v)}
            if self.use_wandb:
                wandb.log({k: dim_dict}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, dim_dict, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
