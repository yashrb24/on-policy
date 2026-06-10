#!/usr/bin/env python3
"""Stage A -- offline behavioral cloning / policy distillation for Pursuit.

Trains a SMALLER transformer actor (R_Actor with reduced n_embd==hidden_size and/or
n_block) to imitate the teacher's full action distribution, via forward
KL(teacher || student) on a fixed dataset of teacher rollouts (collected by
collect_teacher_data.py). No critic, no RL, no PPO -- pure supervised learning. The
deployed artifact is just the student actor, evaluated with eval_pursuit_seeds.py.

Recurrence is handled with WHOLE-EPISODE BPTT from a zero initial hidden state: each
stored episode is one training sequence (hxs=0 at t=0, GRU masks=1 throughout). This
sidesteps the "student hidden state at chunk start is unknown" problem that arises if
you chunk mid-episode. The transformer is non-recurrent over time, so padding short
episodes to the batch max is free (padded time-steps are isolated and loss-masked).

Student arch is set with the usual flags, e.g. --n_embd 48 --hidden_size 48
--n_block 2 --n_head 4 --use_transformer_base_actor --pursuit_drop_walls_channel.
The student actor.pt is written to the standard results path so eval_pursuit_seeds.py
can load it via --model_dir.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from onpolicy.config import get_config
from onpolicy.scripts.train.train_pursuit import parse_args, make_train_env
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_batches(indices, episodes, batch_eps):
    """Yield lists of (obs, logp) sorted by length and chunked, to minimize padding."""
    order = sorted(indices, key=lambda i: episodes[i][0].shape[0])
    for s in range(0, len(order), batch_eps):
        yield [episodes[i] for i in order[s:s + batch_eps]]


def build_batch(chunk, M, obs_dim, A, device):
    B = len(chunk)
    Lmax = max(o.shape[0] for o, _ in chunk)
    obs_b = torch.zeros(Lmax, B, M, obs_dim, dtype=torch.float32)
    tgt_b = torch.zeros(Lmax, B, M, A, dtype=torch.float32)          # teacher log-probs
    lmask = torch.zeros(Lmax, B, M, dtype=torch.float32)
    for b, (o, lp) in enumerate(chunk):
        L = o.shape[0]
        obs_b[:L, b] = torch.from_numpy(o)
        tgt_b[:L, b] = torch.from_numpy(lp.astype(np.float32))
        lmask[:L, b] = 1.0
    obs_flat = obs_b.reshape(Lmax * B * M, obs_dim).to(device)
    tgt_flat = tgt_b.reshape(Lmax * B * M, A).to(device)
    mask_flat = lmask.reshape(Lmax * B * M).to(device)
    return obs_flat, tgt_flat, mask_flat, B


def kl_loss(student_logq, teacher_logp, mask_flat, temp=1.0):
    """forward KL(teacher || student), masked-mean over valid agent-steps.
    teacher_logp / student_logq are log-softmax. With temperature T>1 both are
    softened and the loss is scaled by T^2 (standard distillation)."""
    if temp != 1.0:
        t_logp = F.log_softmax(teacher_logp / temp, dim=-1)
        s_logq = F.log_softmax(student_logq / temp, dim=-1)
    else:
        t_logp = teacher_logp
        s_logq = student_logq
    p_t = t_logp.exp()
    kl = (p_t * (t_logp - s_logq)).sum(-1)                          # (rows,)
    denom = mask_flat.sum().clamp(min=1.0)
    loss = (kl * mask_flat).sum() / denom * (temp ** 2)
    return loss, (kl * mask_flat).sum().item() / denom.item()       # (loss tensor, raw mean KL)


@torch.no_grad()
def evaluate_split(student, episodes, idxs, M, obs_dim, A, device, batch_eps):
    student.eval()
    tot_kl, tot_agree, tot_n = 0.0, 0.0, 0.0
    for chunk in make_batches(idxs, episodes, batch_eps):
        obs_flat, tgt_flat, mask_flat, B = build_batch(chunk, M, obs_dim, A, device)
        rnn0 = torch.zeros(B * M, student._recurrent_N, student.hidden_size, device=device)
        Lmax = obs_flat.shape[0] // (B * M)
        gru_masks = torch.ones(Lmax * B * M, 1, device=device)
        log_s, _ = student.forward_logits(obs_flat, rnn0, gru_masks)
        p_t = tgt_flat.exp()
        kl = (p_t * (tgt_flat - log_s)).sum(-1)
        agree = (log_s.argmax(-1) == tgt_flat.argmax(-1)).float()
        n = mask_flat.sum().item()
        tot_kl += (kl * mask_flat).sum().item()
        tot_agree += (agree * mask_flat).sum().item()
        tot_n += n
    student.train()
    return tot_kl / max(1.0, tot_n), tot_agree / max(1.0, tot_n)


def main(argv):
    parser = get_config()
    parser.add_argument("--data_path", type=str, required=True, help="dataset .pt from collect_teacher_data.py")
    parser.add_argument("--distill_epochs", type=int, default=40)
    parser.add_argument("--distill_lr", type=float, default=5e-4)
    parser.add_argument("--distill_batch_episodes", type=int, default=8)
    parser.add_argument("--distill_temp", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--distill_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="where to write actor.pt; default results/Pursuit/rmappo/<exp>/run1/models")

    all_args = parse_args(argv, parser)
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False

    device = torch.device("cuda:0" if (all_args.cuda and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(all_args.n_training_threads)
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # spaces (tiny throwaway env)
    all_args_env = all_args
    n_rollout_saved = all_args.n_rollout_threads
    all_args.n_rollout_threads = 1
    envs = make_train_env(all_args)
    obs_space = envs.observation_space[0]
    act_space = envs.action_space[0]
    envs.close()
    all_args.n_rollout_threads = n_rollout_saved
    M = all_args.n_pursuers
    A = act_space.n
    obs_dim = obs_space.shape[0]

    # student
    student = R_Actor(all_args, obs_space, act_space, device=device)
    student.train()
    n_params = count_params(student)
    print(f"[distill] STUDENT n_embd={all_args.n_embd} hidden={all_args.hidden_size} "
          f"n_block={all_args.n_block} n_head={all_args.n_head} -> params={n_params:,} "
          f"(teacher ref ~107,721; SCoUT ~39,195)")
    print(f"[distill] obs_dim={obs_dim} action_dim={A} n_agents={M} device={device}")

    # data
    blob = torch.load(all_args.data_path, map_location="cpu", weights_only=False)
    episodes = blob["episodes"]
    meta = blob.get("meta", {})
    print(f"[distill] dataset: {len(episodes)} episodes  meta={meta}")
    assert meta.get("obs_dim", obs_dim) == obs_dim, "dataset/student obs_dim mismatch"
    assert meta.get("action_dim", A) == A, "dataset/student action_dim mismatch"

    n = len(episodes)
    rng = np.random.RandomState(all_args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * all_args.val_frac))
    val_idx = list(perm[:n_val])
    train_idx = list(perm[n_val:])
    print(f"[distill] train={len(train_idx)} val={len(val_idx)} batch_episodes={all_args.distill_batch_episodes}")

    opt = torch.optim.Adam(student.parameters(), lr=all_args.distill_lr)

    # save path
    if all_args.save_dir is not None:
        save_dir = all_args.save_dir
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_scripts = os.path.abspath(os.path.join(here, ".."))
        save_dir = os.path.join(repo_scripts, "results", "Pursuit", "rmappo",
                                all_args.experiment_name, "run1", "models")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[distill] save_dir={save_dir}")

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, all_args.distill_epochs + 1):
        rng.shuffle(train_idx)
        ep_kl, ep_steps = 0.0, 0
        for chunk in make_batches(train_idx, episodes, all_args.distill_batch_episodes):
            obs_flat, tgt_flat, mask_flat, B = build_batch(chunk, M, obs_dim, A, device)
            rnn0 = torch.zeros(B * M, student._recurrent_N, student.hidden_size, device=device)
            Lmax = obs_flat.shape[0] // (B * M)
            gru_masks = torch.ones(Lmax * B * M, 1, device=device)

            log_s, _ = student.forward_logits(obs_flat, rnn0, gru_masks)
            loss, raw_kl = kl_loss(log_s, tgt_flat, mask_flat, temp=all_args.distill_temp)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), all_args.distill_max_grad_norm)
            opt.step()
            ep_kl += raw_kl
            ep_steps += 1

        train_kl = ep_kl / max(1, ep_steps)
        if epoch % 2 == 0 or epoch == 1 or epoch == all_args.distill_epochs:
            val_kl, val_agree = evaluate_split(student, episodes, val_idx, M, obs_dim, A,
                                               device, all_args.distill_batch_episodes)
            tr_kl, tr_agree = evaluate_split(student, episodes, train_idx[:max(1, len(val_idx))],
                                             M, obs_dim, A, device, all_args.distill_batch_episodes)
            print(f"[distill] epoch {epoch:3d}/{all_args.distill_epochs} "
                  f"train_KL={train_kl:.4f} val_KL={val_kl:.4f} val_agree={val_agree*100:.1f}% "
                  f"(trcheck_agree={tr_agree*100:.1f}%) elapsed={time.time()-t0:.0f}s", flush=True)
            if val_kl < best_val:
                best_val = val_kl
                torch.save(student.state_dict(), os.path.join(save_dir, "actor.pt"))
                with open(os.path.join(save_dir, "distill_best.txt"), "w") as f:
                    f.write(f"epoch={epoch} val_KL={val_kl:.5f} val_agree={val_agree:.4f} "
                            f"params={n_params} n_embd={all_args.n_embd} n_block={all_args.n_block}\n")

    # always keep best-val checkpoint as actor.pt (already saved). Write final summary.
    val_kl, val_agree = evaluate_split(student, episodes, val_idx, M, obs_dim, A,
                                       device, all_args.distill_batch_episodes)
    print(f"[distill] DONE exp={all_args.experiment_name} params={n_params:,} "
          f"best_val_KL={best_val:.5f} final_val_KL={val_kl:.5f} final_val_agree={val_agree*100:.1f}% "
          f"actor.pt -> {save_dir}/actor.pt  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
