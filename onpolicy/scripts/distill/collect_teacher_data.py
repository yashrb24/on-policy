#!/usr/bin/env python3
"""Collect an off-policy distillation dataset from a trained Pursuit teacher actor.

Rolls the teacher (stochastically by default, for state coverage) through the obs98
Pursuit env and stores COMPLETE episodes of (obs, teacher_log_probs). The teacher's
full categorical distribution (log-probs over the 5 discrete actions) is the
distillation target -- this carries the "dark knowledge" that hard argmax labels lose.

Output: a torch .pt file holding a list of episodes, each a tuple
    (obs   : float32 (L, n_agents, obs_dim),
     logp  : float16 (L, n_agents, action_dim))   # teacher log-softmax
plus a small meta dict. Episodes are variable length (terminate on all-captured or
max_cycles). Reusable across student architectures -- collect once, train many.

Teacher arch flags (must match the checkpoint) are passed on the CLI exactly like a
normal pursuit run, e.g. --n_embd 64 --hidden_size 64 --n_block 3 --n_head 4
--use_transformer_base_actor --pursuit_drop_walls_channel ...
"""
import os
import sys
import time
import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.scripts.train.train_pursuit import parse_args, make_train_env
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.utils.distributions import FixedCategorical


def main(argv):
    parser = get_config()
    # ---- distillation-collection-specific flags ----
    parser.add_argument("--teacher_model_dir", type=str, required=True,
                        help="dir containing the teacher actor.pt")
    parser.add_argument("--num_episodes", type=int, default=500,
                        help="number of COMPLETE episodes to collect")
    parser.add_argument("--out_path", type=str, required=True,
                        help="output .pt path for the dataset")
    parser.add_argument("--collect_deterministic", action="store_true", default=False,
                        help="if set, roll teacher greedily (argmax); default samples (better coverage)")

    all_args = parse_args(argv, parser)
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False

    device = torch.device("cuda:0" if (all_args.cuda and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(all_args.n_training_threads)
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    N = all_args.n_rollout_threads
    M = all_args.n_pursuers
    recN = all_args.recurrent_N
    H = all_args.hidden_size

    envs = make_train_env(all_args)
    obs_space = envs.observation_space[0]
    act_space = envs.action_space[0]
    A = act_space.n
    obs_dim = obs_space.shape[0]
    print(f"[collect] N_threads={N} n_agents={M} obs_dim={obs_dim} action_dim={A} "
          f"hidden={H} recN={recN} device={device}")

    teacher = R_Actor(all_args, obs_space, act_space, device=device)
    sd = torch.load(os.path.join(all_args.teacher_model_dir, "actor.pt"), map_location=device, weights_only=False)
    teacher.load_state_dict(sd)
    teacher.eval()

    obs = envs.reset()                                   # (N, M, obs_dim)
    rnn = np.zeros((N, M, recN, H), dtype=np.float32)
    masks = np.ones((N, M, 1), dtype=np.float32)

    ep_obs = [[] for _ in range(N)]
    ep_logp = [[] for _ in range(N)]
    episodes = []
    lengths = []

    t0 = time.time()
    step = 0
    while len(episodes) < all_args.num_episodes:
        with torch.no_grad():
            logits, new_rnn = teacher.forward_logits(
                np.concatenate(obs), np.concatenate(rnn), np.concatenate(masks))
            dist = FixedCategorical(logits=logits)
            action = dist.mode() if all_args.collect_deterministic else dist.sample()  # (N*M,1)

        logp_np = logits.detach().cpu().numpy().reshape(N, M, A)     # teacher log-softmax target
        action_np = action.detach().cpu().numpy().reshape(N, M, 1)
        new_rnn = new_rnn.detach().cpu().numpy().reshape(N, M, recN, H)

        for i in range(N):
            ep_obs[i].append(np.asarray(obs[i], dtype=np.float32))   # (M, obs_dim)  state teacher acted on
            ep_logp[i].append(logp_np[i])                            # (M, A)        teacher target

        actions_env = [action_np[i, :, 0] for i in range(N)]
        obs, rewards, dones, infos = envs.step(actions_env)
        dones_env = np.all(dones, axis=-1)                           # (N,)

        # reset hidden/masks for finished threads (next step begins a fresh episode)
        rnn = new_rnn
        rnn[dones_env == True] = 0.0
        masks = np.ones((N, M, 1), dtype=np.float32)
        masks[dones_env == True] = 0.0

        for i in np.flatnonzero(dones_env):
            if len(ep_obs[i]) > 0 and len(episodes) < all_args.num_episodes:
                o = np.stack(ep_obs[i]).astype(np.float32)           # (L, M, obs_dim)
                lp = np.stack(ep_logp[i]).astype(np.float16)         # (L, M, A)
                episodes.append((o, lp))
                lengths.append(o.shape[0])
            ep_obs[i] = []
            ep_logp[i] = []

        step += 1
        if step % 50 == 0:
            rate = len(episodes) / max(1e-9, time.time() - t0)
            print(f"[collect] step={step} episodes={len(episodes)}/{all_args.num_episodes} "
                  f"(~{rate:.1f} ep/s) mean_len={np.mean(lengths) if lengths else 0:.0f}", flush=True)

    envs.close()

    lengths = np.array(lengths)
    total_agent_steps = int(sum(l * M for l in lengths))
    meta = dict(obs_dim=obs_dim, action_dim=A, n_agents=M, recN=recN, hidden=H,
                n_episodes=len(episodes), mean_len=float(lengths.mean()),
                median_len=float(np.median(lengths)), min_len=int(lengths.min()),
                max_len=int(lengths.max()), total_agent_steps=total_agent_steps,
                teacher_dir=all_args.teacher_model_dir,
                deterministic=all_args.collect_deterministic, seed=all_args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(all_args.out_path)), exist_ok=True)
    torch.save({"episodes": episodes, "meta": meta}, all_args.out_path)
    print(f"[collect] DONE -> {all_args.out_path}")
    print(f"[collect] meta = {meta}")
    print(f"[collect] elapsed = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
