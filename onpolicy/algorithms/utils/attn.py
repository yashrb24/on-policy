import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum

from algorithms.utils.util import torch_uniform_like
from algorithms.utils.comms_loss import (
    NOISE_WIDTH, component_log_loss, compute_num_bits_used
)


class BaseAttention(nn.Module):
    def __init__(
            self, *,
            dim=512,
            qk_dim=64,
            v_dim=64,
            heads=8,
            dropout=0.0,
            attention_config=None,
            critic_truncate_len=0,
            use_comms_channel=False
    ):
        super().__init__()
        inner_qk_dim = qk_dim * heads
        inner_v_dim = v_dim * heads

        self.heads = heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.critic_truncate_len = critic_truncate_len
        self.use_comms_channel = use_comms_channel
        self.key_mat = 0

        if not critic_truncate_len:
            self.to_qk = nn.Linear(dim, inner_qk_dim * 2, bias=False)
            self.to_v = nn.Linear(dim, inner_v_dim, bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_qk_dim, bias=False)
            self.to_k = nn.Linear(dim - critic_truncate_len, inner_qk_dim, bias=False)
            self.to_v = nn.Linear(dim - critic_truncate_len, inner_v_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_v_dim, dim - critic_truncate_len), nn.Dropout(dropout))

    def attend(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, dist=False, mask=None):
        # print("input to base attention", x.shape)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        b, n, _, h = *x.shape, self.heads

        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qk)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=h)

        # print("k q v shapes", k.shape, q.shape, v.shape)
        # if dist:

        if mask is not None:
            n_agents = mask.shape[-1]
            mask = mask.repeat(self.heads, 1).reshape(-1, self.heads, n_agents)

        out, info, (comms_loss, comms_bits) = self.dist_attend(q, k, v, mask=mask)

        self.key_mat = 0
        # else:
        #     out, info = self.attend(q, k, v)

        # print("after dist atted", out.shape)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out, info, (comms_loss, comms_bits)

    def dist_attend(self, *args, **kwargs):
        raise NotImplementedError


class NoisySoftmaxAttention(BaseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = self.qk_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def dist_attend(self, q, k, v, mask=None):
        b, h, n, _ = q.shape
        info = {}
        comms_loss = 0
        comms_bits = 0

        info["k_reg_loss"], k_mat = component_log_loss(k, 2 * NOISE_WIDTH, mask=mask.unsqueeze(-1))

        # print(k_mat.shape, "important key")
        # print(k[0,0,0,:5])
        if self.use_comms_channel:
            comms_loss += component_log_loss(k, NOISE_WIDTH, mask=mask.unsqueeze(-1))[0]
            comms_bits += compute_num_bits_used(k, mask.unsqueeze(-1))
            k = k + torch_uniform_like(k, NOISE_WIDTH)
        else:
            comms_bits += compute_num_bits_used(k, mask.unsqueeze(-1))
            comms_loss += compute_num_bits_used(k, mask.unsqueeze(-1))

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # setting the diagonal elements to a very small value as an agent's communication output should not depend on itself.
        dots = dots.diagonal_scatter(torch.ones(b, h, n).to(dots.device) * torch.finfo(torch.float).min, 0, dim1=-2,
                                     dim2=-1)

        weights = self.softmax(dots)

        # setting the diagonal elements to 1
        weights = weights.diagonal_scatter(torch.ones(b, h, n).to(weights.device), 0, dim1=-2, dim2=-1)

        wv = einsum("b h i j, b h j d -> b h i j d", weights, v)
        # print(wv[0,0,0,0,:5])
        n_agents = mask.shape[-1]
        mask = mask.repeat(1, 1, n_agents).reshape(b, h, n_agents, n_agents)
        info["wv_reg_loss"], wv_mat = component_log_loss(wv, 2 * NOISE_WIDTH, mask=mask.unsqueeze(-1))
        # print(info['wv_mat'].shape, "important wv")

        if self.use_comms_channel:
            comms_loss += component_log_loss(wv, NOISE_WIDTH, mask=mask)[0]
            comms_bits += compute_num_bits_used(wv, mask.unsqueeze(-1))
            wv = wv + torch_uniform_like(wv, NOISE_WIDTH)
        else:
            comms_bits += compute_num_bits_used(wv, mask.unsqueeze(-1))
            comms_loss += compute_num_bits_used(wv, mask.unsqueeze(-1))

        repeat_shape = len(k_mat.shape)
        if repeat_shape == 2:
            rep = [1, n, 1]
        else:
            rep = [n, 1]
        # info['bits_mat'] = (k_mat.unsqueeze(-2).repeat(*rep) + wv_mat)#.cpu().detach().numpy()

        wv = wv + torch_uniform_like(wv, NOISE_WIDTH)
        out = wv.sum(-2)

        return out, info, (comms_loss, comms_bits)
