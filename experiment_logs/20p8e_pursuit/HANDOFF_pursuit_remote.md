# Handoff — run Pursuit MAPPO experiments on colva1 / colva4

This is a self-contained brief for a fresh agent session to launch a few training
runs on the dashlab GPU nodes **colva1** and **colva4** (reachable from the laptop
via the `campnet → lab → colvaX` ssh proxyjump). One run per node (single GPU each).

## 1. What we're doing & why these runs

We're benchmarking this repo's MAPPO Pursuit baseline against the **SCoUT** paper at
**20 pursuers / 8 evaders, 40×40 grid, max_cycles 500**. Metric = capture rate
(`captures / 8` = SCoUT "Catch%").

The winning recipe (call it **combo2**) reached **63.1% @ 4M steps** — the first time
we matched SCoUT's range (their weakest Pursuit row "w/o grouping" = 57%). It is:

> transformer actor + transformer critic, **paper-sparse reward** (catch 5.0, tag 0.0,
> urgency 0.0), **entropy_coef 0.001**, **num_mini_batch 1**, **ppo_epoch 8**,
> lr 3e-4, gamma 0.99, gae_lambda 0.95, n_embd/hidden 64, n_block 3, n_head 4.

Hard-won lessons (don't re-litigate these):
- **ppo_epoch 8 >> 5** once entropy 0.001 + mb1 are on (ppo5 throttled it to 28.9%).
- **mb1 is a slow starter** (loses early, wins late) — judge only at >=3M steps.
- **Width 128 hurts** (stalls ~7%); keep width 64. No bigger nets.
- Paper-sparse reward beats PZ defaults. MLP critic underperforms the transformer critic.

### Runs to launch (recommended)
The single-knob probes (ppo10, gamma 0.995, entropy 0.0005) and the 8M ceiling run are
already in flight on other machines. **The highest-value use of two fresh nodes is
seed-replication of combo2** — confirming 63% isn't a lucky seed (seed 8) before we
claim it. Launch:

| node   | experiment_name | change from combo2 | steps |
|--------|-----------------|--------------------|-------|
| colva1 | `combo2_seed1`  | `--seed 1`         | 4M    |
| colva4 | `combo2_seed2`  | `--seed 2`         | 4M    |

(If the user prefers new probes instead, the command template below makes any variant a
one-line change — e.g. `--ppo_epoch 10`, `--clip_param 0.1`, `--gae_lambda 0.97`.)

## 2. Get the code onto each node

Repo lives at `/home/pfs/yash/on-policy` (it's usually on `main`, remote = upstream
marlbenchmark, so it does NOT have our branch). Bring in the branch:

```bash
cd /home/pfs/yash/on-policy
git remote get-url fork 2>/dev/null || git remote add fork https://github.com/yashrb24/on-policy.git
git fetch fork feat/pursuit-checkpoint-resume
git checkout feat/pursuit-checkpoint-resume   # or: git checkout -B feat/pursuit-checkpoint-resume fork/feat/pursuit-checkpoint-resume
```

If a node has no GitHub access, **rsync the `onpolicy/` tree** from a machine that has
the branch checked out instead:
```bash
rsync -az --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pt' \
  --exclude='scripts/results/' --exclude='wandb/' \
  <src>/on-policy/onpolicy/  <node>:/home/pfs/yash/on-policy/onpolicy/
```

This branch contains: paper-reward kwargs (`--catch_reward/--tag_reward/--urgency_reward`)
and a **clean-resume checkpoint patch** (saves actor/critic optimizer + ValueNorm stats,
so a run can be resumed via `--model_dir <run>/models` without a warm-start dip).

## 3. Environment — conda `grf`

```bash
source /home/pfs/miniconda3/etc/profile.d/conda.sh && conda activate grf
python -c "import torch,pettingzoo.sisl,gym,wandb,setproctitle,tensorboardX; print(torch.cuda.is_available())"  # expect True
```
`grf` has torch+CUDA and all Pursuit deps (verified on colva2). Do NOT use `.venv` or
system python.

## 4. Launch (one run per node, detached so it survives ssh logout)

```bash
source /home/pfs/miniconda3/etc/profile.d/conda.sh && conda activate grf
cd /home/pfs/yash/on-policy
export PYTHONPATH=/home/pfs/yash/on-policy PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
NAME=combo2_seed1   # set per node; SEED below must match
setsid nohup python onpolicy/scripts/train/train_pursuit.py \
  --env_name Pursuit --algorithm_name rmappo --experiment_name $NAME \
  --n_pursuers 20 --n_evaders 8 --x_size 40 --y_size 40 --max_cycles 500 --episode_length 500 \
  --catch_reward 5.0 --tag_reward 0.0 --urgency_reward 0.0 \
  --lr 3e-4 --critic_lr 3e-4 --entropy_coef 0.001 --ppo_epoch 8 --num_mini_batch 1 \
  --clip_param 0.2 --gamma 0.99 --gae_lambda 0.95 --max_grad_norm 1.0 \
  --n_head 4 --n_block 3 --n_embd 64 --hidden_size 64 --data_chunk_length 50 \
  --use_transformer_base_actor --use_transformer_base_critic \
  --n_rollout_threads 24 --n_training_threads 1 --num_env_steps 4000000 --seed 1 --use_wandb \
  </dev/null > /home/pfs/yash/$NAME.log 2>&1 &
```
Notes:
- `--use_wandb` here **disables** wandb (repo quirk: the flag is `store_false` for
  `use_wandb`; passing it sets use_wandb=False → local logging only). Keep it.
- A transformer-critic run uses **~8.5 GB** on a 12 GB A2000 at 24 threads → fits, FPS ~600.
  If a node has a smaller/occupied GPU and OOMs, drop to `--num_mini_batch 4` ONLY as a
  last resort (it changes the recipe; prefer freeing the GPU).
- Set `CUDA_VISIBLE_DEVICES` to a free GPU index; check with `nvidia-smi` first.

## 5. Monitor

```bash
tail -f /home/pfs/yash/$NAME.log
# capture rate over last 20 readings:
grep -oE '\(([0-9.]+)%\)' /home/pfs/yash/$NAME.log | tr -d '()%' | tail -20 \
  | awk '{s+=$1;n++} END{printf "mean(last%d)=%.1f%%\n",n,s/n}'
```
The proctitle becomes `Pursuit-rmappo-<NAME>`; checkpoints land in
`onpolicy/scripts/results/Pursuit/rmappo/<NAME>/run1/models/`. Each run ~2.2 h to 4M.
Judge combo2-family runs only near the end (mb1 is a slow starter).

## 6. Report back
For each run: final `mean(last20)` capture%, the quartile trajectory (0-1/1-2/2-3/3-4M),
and whether it matched combo2's ~63%. Seed spread across seed 1/2/8 is the headline.
