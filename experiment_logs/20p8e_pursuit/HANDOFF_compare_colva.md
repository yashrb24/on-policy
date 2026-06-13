# Handoff — verify colva1 / colva4 run the SAME code as anjuna2

Purpose: the runs on **colva1** and **colva4** are seed-replications of the **combo2**
recipe (which hit **63% capture @4M** on **anjuna2**, the current main node). Before
trusting that those seeds are comparable, confirm the *training code* on colva1/colva4 is
**identical** to anjuna2's. This brief lets you detect any drift (e.g. if a node was
rsync'd onto `main` instead of checked out on the branch).

## Reference (source of truth = anjuna2)
- Branch: **`feat/pursuit-checkpoint-resume`** on `github.com/yashrb24/on-policy`
- Code commit: **`61dba21`** (`7d700a6` adds docs/tooling only — no code change)
- The 3 files that differ from stock on-policy (sha256):
  - `onpolicy/envs/pursuit/pursuit_env.py`     `23f990cb…`
  - `onpolicy/runner/shared/base_runner.py`    `41148926…`
  - `onpolicy/scripts/train/train_pursuit.py`  `15d9cfb9…`

## Procedure — run on colva1 AND colva4
```bash
cd /home/pfs/yash/on-policy
# make sure the branch + compare tooling are present:
git remote get-url fork 2>/dev/null || git remote add fork https://github.com/yashrb24/on-policy.git
git fetch fork feat/pursuit-checkpoint-resume
git checkout feat/pursuit-checkpoint-resume    # cleanest: puts you exactly on the reference
# if you must keep the node's current checkout, at least pull the two tool files:
#   git show fork/feat/pursuit-checkpoint-resume:tools/manifest_anjuna2.sha256 > tools/manifest_anjuna2.sha256
#   git show fork/feat/pursuit-checkpoint-resume:tools/compare_to_anjuna2.sh   > tools/compare_to_anjuna2.sh

bash tools/compare_to_anjuna2.sh
```
Interpreting output:
- **`RESULT: IDENTICAL`** → the node runs the exact same `onpolicy/` code as anjuna2. ✅
- **`DIFFER  <file>`** → that file's content differs — investigate with
  `git fetch fork && git diff fork/feat/pursuit-checkpoint-resume -- <file>`.
- **`MISSING <file>`** → reference file absent on the node (incomplete copy).
- **`EXTRA  <file>`** → harmless leftover .py not in the reference (typical after rsync-over-main).

## Also compare the *experiment config* (the "set of changes" includes hyperparameters)
The only intended differences between nodes are `--experiment_name` and `--seed`. Confirm
each node's launch command matches this canonical combo2 recipe (everything else identical):
```
--env_name Pursuit --algorithm_name rmappo
--n_pursuers 20 --n_evaders 8 --x_size 40 --y_size 40 --max_cycles 500 --episode_length 500
--catch_reward 5.0 --tag_reward 0.0 --urgency_reward 0.0
--lr 3e-4 --critic_lr 3e-4 --entropy_coef 0.001 --ppo_epoch 8 --num_mini_batch 1
--clip_param 0.2 --gamma 0.99 --gae_lambda 0.95 --max_grad_norm 1.0
--n_head 4 --n_block 3 --n_embd 64 --hidden_size 64 --data_chunk_length 50
--use_transformer_base_actor --use_transformer_base_critic
--n_rollout_threads 24 --n_training_threads 1 --num_env_steps 4000000 --use_wandb
--seed <1|2>        # the ONLY value that should differ between colva1 and colva4
```
Grep the actual flags a node launched with from its log header (the arg dump near the top),
or from the launch script, and diff against the above.

## Optional cross-check against anjuna2 directly
anjuna2 is reachable from the laptop via the same proxyjump (`ProxyJump lab`). If you want a
live diff instead of trusting the manifest, pull both trees and `diff -r`, or run a
checksum dry-run:
```
rsync -nci --checksum -e "ssh -J lab" pfs@anjuna2:/home/pfs/notyash/on-policy/onpolicy/ \
   /home/pfs/yash/on-policy/onpolicy/    # lists only files that differ by content
```

## Report back
Per node (colva1, colva4): `git HEAD`, the compare RESULT line (+ any DIFFER/MISSING), and
the launch `--seed`/flags. Bottom line wanted: "colva1 and colva4 are running code identical
to anjuna2, seeds 1 and 2."
