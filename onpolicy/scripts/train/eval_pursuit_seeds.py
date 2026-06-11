#!/usr/bin/env python3
"""Standalone SCoUT-protocol eval: load a trained actor, run exactly N eval episodes
across N distinct env seeds (one per rollout thread), report Catch% and Done% (all-8)
as mean +/- std over those episodes. Pure eval, no training.

Catch%  = captures/n_evaders per episode (= SCoUT Catch%).
Done%   = fraction of episodes that captured ALL evaders (= SCoUT Done% / win).
"""
import sys
import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.scripts.train.train_pursuit import parse_args, make_eval_env
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


def _t2n(x):
    return x.detach().cpu().numpy()


def main(argv):
    parser = get_config()
    all_args = parse_args(argv, parser)

    n_threads = all_args.n_eval_rollout_threads
    n_agents = all_args.n_pursuers
    n_evaders = all_args.n_evaders
    hidden = all_args.hidden_size
    rec_N = all_args.recurrent_N

    device = torch.device("cuda:0" if (all_args.cuda and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_eval_env(all_args)
    obs_space = envs.observation_space[0]
    share_obs_space = envs.share_observation_space[0] if all_args.use_centralized_V else obs_space
    act_space = envs.action_space[0]

    policy = R_MAPPOPolicy(all_args, obs_space, share_obs_space, act_space, device=device)
    actor_sd = torch.load(all_args.model_dir + "/actor.pt", map_location=device)
    policy.actor.load_state_dict(actor_sd)
    policy.actor.eval()

    obs = envs.reset()
    rnn = np.zeros((n_threads, n_agents, rec_N, hidden), dtype=np.float32)
    masks = np.ones((n_threads, n_agents, 1), dtype=np.float32)

    # first-finish-per-thread capture: one episode per distinct seed
    ep_success = [None] * n_threads
    ep_capture = [None] * n_threads

    step = 0
    MAX_STEPS = all_args.episode_length + 20  # allow truncation at max_cycles to register
    while step < MAX_STEPS and any(s is None for s in ep_success):
        with torch.no_grad():
            actions, rnn = policy.act(
                np.concatenate(obs),
                np.concatenate(rnn),
                np.concatenate(masks),
                deterministic=all_args.eval_deterministic,
            )
        actions = np.array(np.split(_t2n(actions), n_threads))
        rnn = np.array(np.split(_t2n(rnn), n_threads))
        actions_env = [actions[i, :, 0] for i in range(n_threads)]

        obs, rewards, dones, infos = envs.step(actions_env)
        dones_env = np.all(dones, axis=-1)
        for i in np.flatnonzero(dones_env):
            if ep_success[i] is None:  # record only the FIRST finished episode per thread
                ep_success[i] = float(infos[i].get('success', 0.0))
                ep_capture[i] = int(infos[i].get('n_captures_so_far', 0))
        # reset rnn/masks for threads that just finished (vec env auto-resets obs)
        rnn[dones_env == True] = 0.0
        masks = np.ones((n_threads, n_agents, 1), dtype=np.float32)
        masks[dones_env == True] = 0.0
        step += 1

    envs.close()

    succ = np.array([s for s in ep_success if s is not None], dtype=np.float64)
    caps = np.array([c for c in ep_capture if c is not None], dtype=np.float64)
    catch = caps / max(1, n_evaders)  # per-episode catch fraction

    n = len(succ)
    catch_mean, catch_std = catch.mean() * 100, catch.std(ddof=1) * 100
    done_mean = succ.mean() * 100
    done_std = succ.std(ddof=1) * 100

    tag = all_args.experiment_name
    print(f"RESULT {tag} | n_ep={n} | "
          f"Catch%={catch_mean:.1f}+/-{catch_std:.1f} | "
          f"Done%(all{n_evaders})={done_mean:.1f}+/-{done_std:.1f} | "
          f"per_ep_caps={caps.astype(int).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1:])
