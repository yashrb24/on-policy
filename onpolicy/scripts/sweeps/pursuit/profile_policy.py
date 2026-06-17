"""Standalone micro-benchmark for the pursuit ROLLOUT policy forward.

Isolates `R_MAPPOPolicy.get_actions` (actor + critic forward at the exact batch
shape used during rollout collection: B = n_rollout_threads * num_agents) so we
can measure the levers that actually touch the policy:

  * baseline      : eager, no_grad, eval()                (== current rollout path)
  * compile        : torch.compile(actor.base, critic.base) default mode
  * compile-RO     : torch.compile(..., mode="reduce-overhead")  (CUDA graphs)
  * bf16           : torch.autocast(bfloat16) around get_actions
  * sdpa           : monkeypatch attention -> F.scaled_dot_product_attention
                     (fused/flash kernel, O(L) attn memory instead of O(L^2))

It also reports peak CUDA memory per variant (relevant to the 60p+ OOM) and can
dump a torch.profiler op breakdown.

NOTE ON HARDWARE: torch.compile's real win (Triton fusion + CUDA-graph launch
collapse) and bf16 only show their true numbers on the CUDA cluster nodes. Run
this on a GPU node; CPU/MPS numbers here are only a compatibility / structure
check and DO NOT transfer.

Usage (GPU node):
  python profile_policy.py --device cuda --num_agents 40 --n_rollout_threads 48
  python profile_policy.py --device cuda --num_agents 60 --n_rollout_threads 48 --profile
"""
import argparse
import time

import numpy as np
import torch

from gym.spaces import Box, Discrete

from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


def build_args(num_agents, n_embd, n_block, n_head):
    """Reuse the repo arg parser for defaults, then apply the pursuit sweep overrides."""
    args = get_config().parse_args([])
    args.algorithm_name = "rmappo"
    args.use_transformer_base_actor = True
    args.use_transformer_base_critic = True
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    args.recurrent_N = 1
    args.n_embd = n_embd
    args.hidden_size = n_embd
    args.n_block = n_block
    args.actor_n_block = n_block
    args.critic_n_block = n_block
    args.n_head = n_head
    args.num_agents = num_agents
    args.use_active_masks_in_transformer = False
    args.use_comms_channel = False
    return args


def make_inputs(B, obs_dim, recurrent_N, hidden, device):
    rng = np.random.default_rng(0)
    obs = rng.random((B, obs_dim), dtype=np.float32)
    return dict(
        cent_obs=obs.copy(),                                           # transformer critic = per-agent obs
        obs=obs,
        rnn_states_actor=np.zeros((B, recurrent_N, hidden), np.float32),
        rnn_states_critic=np.zeros((B, recurrent_N, hidden), np.float32),
        masks=np.ones((B, 1), np.float32),
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timeit(fn, device, iters, warmup):
    for _ in range(warmup):
        fn()
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - t0) / iters * 1e3  # ms / forward


def peak_mem_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return float("nan")


def patch_sdpa():
    """Swap the hand-rolled QK^T-softmax-V for the fused F.scaled_dot_product_attention.
    Valid only for the vanilla sweep (no active masks, no comms channel)."""
    import torch.nn.functional as F
    from onpolicy.algorithms.utils import transformer_encoder as te

    def forward(self, key, value, query, active_masks=None):
        B, L, D = query.size()
        self.comm_loss = 0
        self.comm_bits = 0
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)        # fused, O(L) memory
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)

    te.SelfAttention.forward = forward


def run_variant(name, policy, inp, device, iters, warmup, autocast=False, profile=False):
    policy.actor.eval()
    policy.critic.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def call():
        with torch.no_grad(), (torch.autocast(device.type, dtype=torch.bfloat16) if autocast else _null()):
            policy.get_actions(**inp)

    ms = timeit(call, device, iters, warmup)
    mem = peak_mem_mb(device)
    print(f"  {name:<16} {ms:8.3f} ms/fwd   peak {mem:8.1f} MB")

    if profile:
        with torch.no_grad(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
            + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
            record_shapes=True,
        ) as prof:
            for _ in range(10):
                policy.get_actions(**inp)
            synchronize(device)
        sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
        trace = f"trace_{name}_{policy.actor.num_agents}p.json"
        prof.export_chrome_trace(trace)
        print(f"  chrome trace -> {trace}")
    return ms


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_agents", type=int, default=40)
    p.add_argument("--n_rollout_threads", type=int, default=48)
    p.add_argument("--obs_dim", type=int, default=147)   # pursuit obs_range=7, 3 channels -> 7*7*3
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--n_block", type=int, default=3)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--variants", default="baseline,compile,compile-ro,bf16,sdpa")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    B = args.n_rollout_threads * args.num_agents
    print(f"\n== pursuit policy forward | {args.num_agents} agents | "
          f"B = {args.n_rollout_threads} threads x {args.num_agents} = {B} rows | device={device} ==")

    obs_space = Box(low=0.0, high=1.0, shape=(args.obs_dim,), dtype=np.float32)
    act_space = Discrete(5)
    want = [v.strip() for v in args.variants.split(",") if v.strip()]

    for variant in want:
        torch.manual_seed(0)
        if variant == "sdpa":
            patch_sdpa()
        a = build_args(args.num_agents, args.n_embd, args.n_block, args.n_head)
        policy = R_MAPPOPolicy(a, obs_space, obs_space, act_space, device)
        inp = make_inputs(B, args.obs_dim, a.recurrent_N, a.hidden_size, device)

        if variant in ("compile", "compile-ro"):
            mode = "reduce-overhead" if variant == "compile-ro" else "default"
            try:
                policy.actor.base = torch.compile(policy.actor.base, mode=mode)
                policy.critic.base = torch.compile(policy.critic.base, mode=mode)
            except Exception as e:
                print(f"  {variant:<16} compile setup FAILED: {type(e).__name__}: {e}")
                continue

        try:
            run_variant(variant, policy, inp, device, args.iters, args.warmup,
                        autocast=(variant == "bf16"), profile=args.profile)
        except Exception as e:
            print(f"  {variant:<16} RUN FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
