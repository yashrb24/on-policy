import copy
import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.transformer_encoder import TransformerEncoderBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

"""
Transformer Reshaping Logic:

The transformer-based base requires special handling of the agent dimension:
1. TransformerEncoderBase expects input shape: (batch, n_agents, obs_dim)
2. Other archs flatten all dimensions: (batch * n_agents, obs_dim)

Data Flow:
- During training with transformer:
  * SharedBuffer.recurrent_generator_agent_preserved() provides data as [L*N, M, ...]
  * Where L=chunk_length, N=mini_batch_size, M=num_agents
  * This preserves the agent dimension for transformer processing

- During inference with transformer:
  * Input arrives as [batch*agents, obs_dim] and needs reshaping
  * Reshape to [batch, agents, obs_dim] for transformer
  * Flatten back to [batch*agents, hidden_dim] for downstream processing

- The flag `use_transformer_base_actor` tells whether the transformer should be used for the actor or not. 
- The flat `use_transformer_base_critic` tells whether the transformer should be used for the critic or not.
"""


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.use_transformer_base_actor = args.use_transformer_base_actor
        self.use_active_masks_in_transformer = args.use_active_masks_in_transformer
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        if self.use_transformer_base_actor:
            # Actor always calculates communication metrics if communication channel is enabled.
            # Optional asymmetric depth 
            actor_args = args
            if args.actor_n_block is not None and args.actor_n_block != args.n_block:
                actor_args = copy.copy(args)
                actor_args.n_block = args.actor_n_block
            self.base = TransformerEncoderBase(actor_args, obs_shape, calc_comm_metrics=True)
        else:
            base = CNNBase if len(obs_shape) == 3 else MLPBase
            self.base = base(args, obs_shape)

        self.num_agents = getattr(args, 'num_agents', 1)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, active_masks=None):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.
        :param active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # Reshape for transformer if needed
        if self.use_transformer_base_actor:
            batch_size = obs.shape[0]
            # Reshape from (batch*agents, obs_dim) to (batch, agents, obs_dim)
            obs_reshaped = obs.reshape(batch_size // self.num_agents, self.num_agents, -1)

            # Prepare active_masks for transformer (needs shape [batch, agents, 1])
            if active_masks is not None:
                # Currently shape is flattened, reshape to [batch, agents, 1]
                active_masks = active_masks.reshape(
                    batch_size // self.num_agents, self.num_agents, -1
                )

            if self.use_active_masks_in_transformer:
                base_output = self.base(obs_reshaped, active_masks=active_masks)
            else:
                base_output = self.base(obs_reshaped)

            # Handle communication metrics if transformer returns them
            if isinstance(base_output, tuple):
                actor_features, _ = base_output # the ignored return value is comm_metrics
            else:
                actor_features = base_output

            # Reshape back from (batch, agents, hidden_dim) to (batch*agents, hidden_dim)
            actor_features = actor_features.reshape(batch_size, -1)
        else:
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def forward_logits(self, obs, rnn_states, masks, available_actions=None):
        """Distillation hook: same data-flow as forward() but returns the full
        categorical log-probs (FixedCategorical.logits) instead of a sampled action.
        Discrete action space only; the RL path (forward/evaluate_actions) is untouched.

        Handles both single-step rollout and whole-episode BPTT — RNNLayer dispatches
        on whether obs.size(0) == rnn.size(0):
          * single-step:  obs (B*M, obs_dim),   rnn (B*M, recN, H), masks (B*M, 1)
          * sequence:     obs (T*B*M, obs_dim), rnn (B*M, recN, H), masks (T*B*M, 1)

        :return: (logits (rows, action_dim) log-softmax, updated rnn_states)
        """
        if getattr(self.act, 'multi_discrete', False) or getattr(self.act, 'mixed_action', False):
            raise NotImplementedError("forward_logits supports single Discrete action heads only.")

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.use_transformer_base_actor:
            batch_size = obs.shape[0]
            obs_reshaped = obs.reshape(batch_size // self.num_agents, self.num_agents, -1)
            base_output = self.base(obs_reshaped)
            if isinstance(base_output, tuple):
                actor_features, _ = base_output
            else:
                actor_features = base_output
            actor_features = actor_features.reshape(batch_size, -1)
        else:
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_logits = self.act.action_out(actor_features, available_actions)
        return action_logits.logits, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.use_transformer_base_actor:
            action = action.reshape(-1, action.shape[-1])
            rnn_states = rnn_states.reshape(-1, rnn_states.shape[-2], rnn_states.shape[-1])  # num_rollout_threads * num_agents, num_recurrent_layers, hidden_size
            masks = masks.reshape(-1, masks.shape[-1])  # num_rollout_threads * num_agents, 1

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            if self.use_transformer_base_actor:
                available_actions = available_actions.reshape(-1, available_actions.shape[-1])

        comm_metrics = None
        if self.use_transformer_base_actor:

            if self.use_active_masks_in_transformer:
                base_output = self.base(obs, active_masks)
            else:
                base_output = self.base(obs)

            # Handle communication metrics if transformer returns them
            if isinstance(base_output, tuple):
                actor_features, comm_metrics = base_output
            else:
                actor_features = base_output
                comm_metrics = None

            actor_features = actor_features.reshape(-1, actor_features.shape[-1])  # num_rollout_threads * num_agents, action_feature_dim
        else:
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if self.use_transformer_base_actor:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            else:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
            if self.use_transformer_base_actor:
                active_masks = active_masks.reshape(-1, active_masks.shape[-1])

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

        return action_log_probs, dist_entropy, comm_metrics


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.num_agents = getattr(args, 'num_agents', 1)
        self.use_transformer_base_critic = args.use_transformer_base_critic
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if self.use_transformer_base_critic:
            # Critic never calculates communication metrics, and never uses the DDCL
            # channel: it only ever lives in the actor (the comm-rate penalty comes from
            # the actor alone), so the centralized value estimates stay undithered.
            critic_args = copy.copy(args)
            if args.critic_n_block is not None and args.critic_n_block != args.n_block:
                critic_args.n_block = args.critic_n_block   # optional asymmetric depth
            critic_args.use_comms_channel = False
            self.base = TransformerEncoderBase(critic_args, cent_obs_shape, calc_comm_metrics=False)
        else:
            base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
            self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.use_transformer_base_critic:
            if cent_obs.dim() == 2:
                batch_size = cent_obs.shape[0]
                if batch_size % self.num_agents != 0:
                    raise ValueError(
                        f"Transformer critic expected flattened batch divisible by num_agents={self.num_agents}, "
                        f"got {batch_size} rows.")
                cent_obs = cent_obs.reshape(batch_size // self.num_agents, self.num_agents, -1)
            elif cent_obs.dim() != 3:
                raise ValueError(
                    f"Transformer critic expected 2D flattened or 3D agent-preserved input, got {cent_obs.dim()}D.")

            critic_features = self.base(cent_obs)
            critic_features = critic_features.reshape(-1, critic_features.shape[-1])
        else:
            critic_features = self.base(cent_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:

            if self.use_transformer_base_critic:
                if masks.dim() == 3:
                    masks = masks.reshape(-1, masks.shape[-1])
                if rnn_states.dim() == 4:
                    rnn_states = rnn_states.reshape(-1, rnn_states.shape[-2], rnn_states.shape[-1])
            elif self.training:
                masks = masks.reshape(-1, masks.shape[-1]) # num_rollout_threads * num_agents, 1

            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states
