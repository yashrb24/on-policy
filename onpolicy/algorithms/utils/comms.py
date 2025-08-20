import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.utils.util import dict_merge
from algorithms.utils.attn import NoisySoftmaxAttention


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CommsTransformer(nn.Module):
    def __init__(self, dim, depth, heads, qk_dim, v_dim, dropout, use_comms_channel):
        super(CommsTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.bits_mat = 0
        self.use_comms_channel = use_comms_channel
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

        self.embedding = nn.Sequential(
            init_(nn.Linear(dim, dim * 2, bias=True), activate=True),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            init_(nn.Linear(dim * 2, dim, bias=True), activate=True),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        # AttentionLayer = get_attention_layer(attention)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, NoisySoftmaxAttention(dim=dim, qk_dim=qk_dim, v_dim=v_dim, heads=heads,
                                                           dropout=dropout,
                                                           use_comms_channel=self.use_comms_channel), ),
                        PreNorm(dim, FeedForward(dim, dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, dist=False, mask=None):
        x = self.fc(x)
        batch, timesteps, n_agents, s_dim = x.shape
        x = x.reshape(-1, n_agents, s_dim)

        infos = []
        comms_loss, comms_bits = 0, 0
        for attn, ff in self.layers:
            x_, info, (c_loss, c_bits) = attn(x, dist=dist, mask=mask)
            x = x_ + x
            x = ff(x) + x
            infos.append(info)

            if 'bits_mat' in info:
                self.bits_mat = info['bits_mat'].cpu().detach().numpy()
            comms_loss += c_loss
            comms_bits += c_bits

        info = dict_merge(infos, mode="mean")

        x = x.reshape(batch, timesteps, n_agents, s_dim)

        return x, info, (comms_loss, comms_bits)


class CommsMLP(nn.Module):
    def __init__(self, obs_input_dim, num_agents) -> None:
        super(CommsMLP, self).__init__()
        self.num_agents = num_agents
        self.fc = nn.Sequential(nn.Linear(obs_input_dim * self.num_agents, obs_input_dim * self.num_agents), nn.GELU())

        self.mlp = nn.Sequential(
            init_(nn.Linear(obs_input_dim * self.num_agents, 2 * obs_input_dim * self.num_agents), activate=True),
            nn.GELU(),
            init_(nn.Linear(2 * obs_input_dim * self.num_agents, obs_input_dim * self.num_agents), activate=True),
            nn.GELU()
        )

        self.embedding = nn.Sequential(
            init_(nn.Linear(obs_input_dim * self.num_agents, obs_input_dim * self.num_agents * 2, bias=True),
                  activate=True),
            nn.GELU(),
            nn.LayerNorm(obs_input_dim * self.num_agents * 2),
            init_(nn.Linear(obs_input_dim * self.num_agents * 2, obs_input_dim * self.num_agents, bias=True),
                  activate=True),
            nn.GELU(),
            nn.LayerNorm(obs_input_dim * self.num_agents),
        )

        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x, mask=None):
        batch, timestep, n_agents, obs_dim = x.shape

        # flatten x to [batch, timestep, n_agents * obs_dim]
        x = self.flatten(x)
        x = self.fc(x)
        x = self.mlp(x)
        x = self.embedding(x)

        # reshape x back to [batch, timestep, n_agents, obs_dim]
        x = x.reshape(batch, timestep, n_agents, obs_dim)

        return x
