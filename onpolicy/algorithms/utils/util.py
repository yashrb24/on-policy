import copy
import numpy as np

import torch
import torch.nn as nn

from collections import defaultdict

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def dict_merge(list_of_dict, dim=0, mode="concat", backend="torch"):
    ret = defaultdict(list)
    for d in list_of_dict:
        for k, v in d.items():
            ret[k].append(v)
    if mode == "concat":
        if backend == "torch":
            ret = {k: torch.cat(v, dim) for k, v in ret.items()}
        else:
            ret = {k: np.concatenate(v, axis=dim) for k, v in ret.items()}
    elif mode == "stack":
        if backend == "torch":
            ret = {k: torch.stack(v, dim) for k, v in ret.items()}
        else:
            ret = {k: np.stack(v, axis=dim) for k, v in ret.items()}
    elif mode == "mean":
        if backend == "torch":
            ret = {k: torch.mean(torch.stack(v, 0), 0) for k, v in ret.items()}
        else:
            ret = {k: np.mean(v) for k, v in ret.items()}
    return ret

def torch_uniform_like(x, width):
    """
    Generate tensor of same shape as `x` from random uniform distribution
    between -`width` and `width`

    This operation is differentiable with respect to `width`
    """
    unit_sample = torch.rand_like(x)
    sample = (unit_sample - 0.5) * 2 * width
    return sample


def torch_layer_norm(x, norm=1, ord=2, eps=1e-5):
    """
    Scale tensor so that each vector in the last dimension will have norm `norm`
    """
    return x / (torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True) + eps) * norm


def topk_softmax(x, k):
    topk_indices = torch.topk(x, k, dim=-1).indices
    m = torch.full_like(x, -float("inf"))
    m.scatter_(-1, topk_indices, 0)
    return torch.softmax(x+m, -1)
