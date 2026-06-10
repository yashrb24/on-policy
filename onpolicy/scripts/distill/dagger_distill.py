#!/usr/bin/env python3
"""Stage B -- on-policy (DAgger-style) distillation for Pursuit.

Offline BC (train_distill.py) trains on the TEACHER's state distribution, so the
student can drift into states the teacher never showed it and compound errors over a
500-step horizon. DAgger fixes this: the STUDENT rolls out (its own state
distribution), the TEACHER relabels every visited state with its action distribution,
and we minimize KL(teacher||student) on the student's trajectories. Iterate.

Recommended use: warm-start from a BC student (--student_init_dir) -- BC gets ~90%
for free, DAgger closes the distribution-shift tail. Pure on-policy (no teacher
mixing, beta=0) by default; set --beta_start/--beta_end for classic DAgger action
mixing (act with the teacher w.p. beta, decaying over rounds).

Each round: (1) student rolls episodes_per_round episodes, teacher relabels;
(2) episodes added to a capped aggregation buffer; (3) KL training for
train_epochs_per_round epochs (whole-episode BPTT, same as Stage A). The artifact is
the student actor.pt, evaluated with the sampled SCoUT protocol.
"""
import os
import sys
import time
import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.scripts.train.train_pursuit import parse_args, make_train_env, make_eval_env
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.utils.distributions import FixedCategorical


def build_batch(chunk, M, obs_dim, A, device):
    B = len(chunk)
    Lmax = max(o.shape[0] for o, _ in chunk)
    obs_b = torch.zeros(Lmax, B, M, obs_dim, dtype=torch.float32)
    tgt_b = torch.zeros(Lmax, B, M, A, dtype=torch.float32)
    lmask = torch.zeros(Lmax, B, M, dtype=torch.float32)
    for b, (o, lp) in enumerate(chunk):
        L = o.shape[0]
        obs_b[:L, b] = torch.from_numpy(o)
        tgt_b[:L, b] = torch.from_numpy(lp.astype(np.float32))
        lmask[:L, b] = 1.0
    return (obs_b.reshape(Lmax * B * M, obs_dim).to(device),
            tgt_b.reshape(Lmax * B * M, A).to(device),
            lmask.reshape(Lmax * B * M).to(device), B)


def make_batches(episodes, batch_eps, rng):
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    idx = sorted(idx, key=lambda i: episodes[i][0].shape[0])
    for s in range(0, len(idx), batch_eps):
        yield [episodes[i] for i in idx[s:s + batch_eps]]


@torch.no_grad()
def rollout_relabel(student, teacher, envs, n_ep, M, recN, H, A, device, beta=0.0, rng=None):
    """Student acts (its own state distribution); teacher relabels each visited state.
    Returns a list of (obs float32 (L,M,obs), teacher_logp float16 (L,M,A)) episodes,
    plus mean teacher catch fraction over finished episodes (a free sampled-eval signal)."""
    N = envs.num_envs if hasattr(envs, "num_envs") else None
    obs = envs.reset()
    N = obs.shape[0]
    s_rnn = np.zeros((N, M, recN, H), dtype=np.float32)
    t_rnn = np.zeros((N, M, recN, H), dtype=np.float32)
    masks = np.ones((N, M, 1), dtype=np.float32)
    ep_obs = [[] for _ in range(N)]
    ep_lp = [[] for _ in range(N)]
    episodes, caps = [], []
    while len(episodes) < n_ep:
        s_logits, s_rnn_new = student.forward_logits(np.concatenate(obs), np.concatenate(s_rnn), np.concatenate(masks))
        t_logits, t_rnn_new = teacher.forward_logits(np.concatenate(obs), np.concatenate(t_rnn), np.concatenate(masks))
        s_act = FixedCategorical(logits=s_logits).sample()           # student's own action (state coverage)
        if beta > 0.0:
            t_act = FixedCategorical(logits=t_logits).sample()
            use_t = torch.from_numpy((rng.random(s_act.shape[0]) < beta).astype(np.float32)).to(s_act.device).long().unsqueeze(-1)
            act = torch.where(use_t.bool(), t_act, s_act)
        else:
            act = s_act
        t_lp = t_logits.detach().cpu().numpy().reshape(N, M, A)       # teacher target on student-visited state
        act_np = act.detach().cpu().numpy().reshape(N, M, 1)
        s_rnn = s_rnn_new.detach().cpu().numpy().reshape(N, M, recN, H)
        t_rnn = t_rnn_new.detach().cpu().numpy().reshape(N, M, recN, H)
        for i in range(N):
            ep_obs[i].append(np.asarray(obs[i], dtype=np.float32))
            ep_lp[i].append(t_lp[i])
        obs, rewards, dones, infos = envs.step([act_np[i, :, 0] for i in range(N)])
        dones_env = np.all(dones, axis=-1)
        s_rnn[dones_env == True] = 0.0
        t_rnn[dones_env == True] = 0.0
        masks = np.ones((N, M, 1), dtype=np.float32)
        masks[dones_env == True] = 0.0
        for i in np.flatnonzero(dones_env):
            if len(ep_obs[i]) > 0 and len(episodes) < n_ep:
                episodes.append((np.stack(ep_obs[i]).astype(np.float32),
                                 np.stack(ep_lp[i]).astype(np.float16)))
                caps.append(int(infos[i].get('n_captures_so_far', 0)))
            ep_obs[i] = []
            ep_lp[i] = []
    return episodes, (np.mean(caps) / 8.0 if caps else 0.0)


def main(argv):
    parser = get_config()
    parser.add_argument("--teacher_model_dir", type=str, required=True)
    parser.add_argument("--student_init_dir", type=str, default=None, help="BC checkpoint dir to warm-start from")
    parser.add_argument("--dagger_rounds", type=int, default=10)
    parser.add_argument("--episodes_per_round", type=int, default=48)
    parser.add_argument("--train_epochs_per_round", type=int, default=8)
    parser.add_argument("--distill_lr", type=float, default=5e-4)
    parser.add_argument("--distill_batch_episodes", type=int, default=8)
    parser.add_argument("--aggregate_cap", type=int, default=600)
    parser.add_argument("--beta_start", type=float, default=0.0)
    parser.add_argument("--beta_end", type=float, default=0.0)
    parser.add_argument("--distill_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default=None)

    all_args = parse_args(argv, parser)
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False

    device = torch.device("cuda:0" if (all_args.cuda and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(all_args.n_training_threads)
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
    rng = np.random.RandomState(all_args.seed)

    M, recN, H = all_args.n_pursuers, all_args.recurrent_N, all_args.hidden_size
    envs = make_train_env(all_args)
    obs_space, act_space = envs.observation_space[0], envs.action_space[0]
    A, obs_dim = act_space.n, obs_space.shape[0]

    teacher = R_Actor(all_args, obs_space, act_space, device=device)
    teacher.load_state_dict(torch.load(os.path.join(all_args.teacher_model_dir, "actor.pt"), map_location=device, weights_only=False))
    teacher.eval()

    student = R_Actor(all_args, obs_space, act_space, device=device)
    if all_args.student_init_dir:
        student.load_state_dict(torch.load(os.path.join(all_args.student_init_dir, "actor.pt"), map_location=device, weights_only=False))
        print(f"[dagger] warm-started student from {all_args.student_init_dir}")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[dagger] student params={n_params:,} n_embd={all_args.n_embd} n_block={all_args.n_block} "
          f"rounds={all_args.dagger_rounds} ep/round={all_args.episodes_per_round} device={device}")

    opt = torch.optim.Adam(student.parameters(), lr=all_args.distill_lr)

    if all_args.save_dir:
        save_dir = all_args.save_dir
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(os.path.abspath(os.path.join(here, "..")), "results", "Pursuit", "rmappo",
                                all_args.experiment_name, "run1", "models")
    os.makedirs(save_dir, exist_ok=True)

    buffer = []
    t0 = time.time()
    for r in range(1, all_args.dagger_rounds + 1):
        beta = all_args.beta_start + (all_args.beta_end - all_args.beta_start) * (r - 1) / max(1, all_args.dagger_rounds - 1)
        student.eval()
        fresh, catch = rollout_relabel(student, teacher, envs, all_args.episodes_per_round,
                                       M, recN, H, A, device, beta=beta, rng=rng)
        buffer.extend(fresh)
        if len(buffer) > all_args.aggregate_cap:
            buffer = buffer[-all_args.aggregate_cap:]

        student.train()
        last_kl = 0.0
        for e in range(all_args.train_epochs_per_round):
            ek, es = 0.0, 0
            for chunk in make_batches(buffer, all_args.distill_batch_episodes, rng):
                obs_flat, tgt_flat, mask_flat, B = build_batch(chunk, M, obs_dim, A, device)
                rnn0 = torch.zeros(B * M, recN, H, device=device)
                Lmax = obs_flat.shape[0] // (B * M)
                gru_masks = torch.ones(Lmax * B * M, 1, device=device)
                log_s, _ = student.forward_logits(obs_flat, rnn0, gru_masks)
                p_t = tgt_flat.exp()
                kl = (p_t * (tgt_flat - log_s)).sum(-1)
                loss = (kl * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), all_args.distill_max_grad_norm)
                opt.step()
                ek += (kl * mask_flat).sum().item() / mask_flat.sum().clamp(min=1.0).item(); es += 1
            last_kl = ek / max(1, es)

        torch.save(student.state_dict(), os.path.join(save_dir, "actor.pt"))
        print(f"[dagger] round {r:2d}/{all_args.dagger_rounds} beta={beta:.2f} "
              f"rollout_teacher_catch={catch*100:.1f}% buffer={len(buffer)} train_KL={last_kl:.4f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    envs.close()
    print(f"[dagger] DONE exp={all_args.experiment_name} params={n_params:,} "
          f"actor.pt -> {save_dir}/actor.pt total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
