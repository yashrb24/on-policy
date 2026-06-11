# Pursuit obs98 + λ0.90 — Final Working Setup

**The recipe that matches/beats SCoUT on SISL Pursuit 20P-8E at 4M steps.** This is the reference for taking the work forward — exact config, paths, checkpoints, results, and next steps. (Not a history log; see `PURSUIT_RUNS_SUMMARY.md` for the full campaign.)

## Code provenance
- **Branch:** `feat/pursuit-checkpoint-resume`
- **Code commit (training runs ran on this):** `174bf44` — `feat(pursuit): --pursuit_drop_walls_channel for SCoUT-style 98-dim obs`
- **Remote:** `origin` = https://github.com/yashrb24/on-policy.git
- **Model:** decentralized **rMAPPO** (`R_MAPPO` + `R_MAPPOPolicy`), actor & critic each = `TransformerEncoderBase` (self-attention over the 20-agent dim) + GRU + linear head. **NOT MAT.** Params @obs98: actor 107,721 / critic 107,461 / total 215,182 (vs SCoUT actor 39,195 / total 156,677 — we are ~2.75× the actor; capacity is *not* matched).

## The two decisive levers
1. **obs98** — `--pursuit_drop_walls_channel` drops the all-zeros walls channel → obs 7×7×2 = **98** (matches SCoUT) instead of 147 = 7×7×3.
2. **λ0.90** — `--gae_lambda 0.90` (down from 0.95). Rescues the low/mid seeds obs98 alone leaves behind; erases the low tail.

## Exact launch command (per seed `$S`)
```bash
python onpolicy/scripts/train/train_pursuit.py \
  --env_name Pursuit --algorithm_name rmappo --experiment_name cX_lam90_s$S \
  --n_pursuers 20 --n_evaders 8 --x_size 40 --y_size 40 --max_cycles 500 --episode_length 500 \
  --catch_reward 5.0 --tag_reward 0.0 --urgency_reward 0.0 \
  --lr 3e-4 --critic_lr 3e-4 --entropy_coef 0.001 --ppo_epoch 8 --num_mini_batch 1 \
  --clip_param 0.2 --gamma 0.99 --gae_lambda 0.90 --max_grad_norm 1.0 \
  --n_head 4 --n_block 3 --n_embd 64 --hidden_size 64 --data_chunk_length 50 \
  --use_transformer_base_actor --use_transformer_base_critic --pursuit_drop_walls_channel \
  --n_rollout_threads 24 --n_training_threads 1 --num_env_steps 4000000 --seed $S --use_wandb
```
Reward = paper-sparse (catch 5, tag 0, urgency 0). Only `--seed` varies across the 20 runs; everything else is identical.

## Checkpoint locations (per-node split, 20 seeds)
Path pattern: `…/onpolicy/scripts/results/Pursuit/rmappo/<EXP>/run1/models/{actor,critic,actor_optimizer,critic_optimizer,valuenorm}.pt` (clean-resume capable). Only latest snapshot kept per run (save() overwrites fixed filenames).

| seeds | node | EXP prefix | repo root on node |
|---|---|---|---|
| s1–s7 | colva2 | `c2_lam90_s{S}` | `/home/pfs/yash/on-policy-pursuit/` |
| s8–s14 | colva1 | `c1_lam90_s{S}` | `/home/pfs/yash/on-policy/` |
| s15–s20 | colva4 | `c4_lam90_s{S}` | (colva4 — path unverified, tunnel down) |

Eval checkpoints staged on anjuna2 at `/tmp/evalck/<EXP>/actor.pt` (s1–s13 pulled; s14–s20 pending colva4/full pull).

## Results — 20P-8E, 4M, horizon 500. SCoUT ref = Catch% 94±10, Done% 70.

**Headline (SCoUT-apples-to-apples = 1 policy × 20 eval seeds, deterministic):** our best policies beat SCoUT on both metrics at lower variance.

| metric | λ0.90 **s12** | λ0.90 **s8** | SCoUT (full) |
|---|---|---|---|
| Catch% (mean±std, 20 eval seeds) | **100.0 ± 0.0** | **99.4 ± 2.8** | 94 ± 10 |
| Done% (all-8 captured) | **100** | **95** | 70 |

**Reliability (robustness across training seeds — NOT the SCoUT comparison):** deterministic-eval Catch% λ0.90 **92.2 ± 9.4** (n=13) vs plain obs98 **71.2 ± 22.6** (n=12). Difference significant: Mann-Whitney U **p=0.0023** (rank-biserial 0.67); paired Wilcoxon on 7 overlap seeds p=0.039 (underpowered). Done% mean 66.5 vs 26.7. Complete 20/20 *training* trail-10 sweep agrees: mean **92.67 vs 76.25** (+16.4), bands HIGH16/MID4/LOW0, zero downward moves.

**Per-seed table** — training Catch% (trail-10, from logs) and deterministic 20-eval-seed Catch%±std / Done%:
| seed | train Catch% | eval Catch%±std (Done%) | | seed | train Catch% | eval Catch%±std (Done%) |
|---|---|---|---|---|---|---|
| s1 | 76.4 | 74.4±14.3 (5) | | s11 | 81.0 | 80.6±14.9 (15) |
| s2 | 99.4 | 97.5±6.5 (85) | | s12 | 99.4 | 100.0±0.0 (100) |
| s3 | 99.0 | 98.8±5.6 (95) | | s13 | 96.0 | 96.9±5.6 (75) |
| s4 | 98.0 | 95.6±14.8 (90) | | s14 | 99.7 | *(colva4-side; pull pending)* |
| s5 | 95.1 | 93.1±11.1 (65) | | s15 | *colva4* | — |
| s6 | 81.5 | 74.4±14.9 (5) | | s16 | *colva4* | — |
| s7 | 98.6 | 97.5±6.5 (85) | | s17 | *colva4 ~98* | — |
| s8 | 98.2 | 99.4±2.8 (95) | | s18 | *colva4* | — |
| s9 | 95.6 | 91.9±12.4 (60) | | s19 | *colva4* | — |
| s10 | 99.0 | 98.1±6.1 (90) | | s20 | *colva4* | — |

s1–s14 train Catch% verified from logs; s15–s20 on colva4 (mean reflected in the 92.67 sweep figure; exact per-seed pending tunnel/colva4 access). Biggest rescue: s14 plain 39.1 → λ0.90 99.7.

## Reproduce the SCoUT-protocol eval
```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD python onpolicy/scripts/train/eval_pursuit_seeds.py \
  --env_name Pursuit --algorithm_name rmappo --experiment_name <tag> \
  --n_pursuers 20 --n_evaders 8 --x_size 40 --y_size 40 --max_cycles 500 --episode_length 500 \
  --n_head 4 --n_block 3 --n_embd 64 --hidden_size 64 \
  --use_transformer_base_actor --use_transformer_base_critic --pursuit_drop_walls_channel \
  --use_eval --eval_deterministic --n_eval_rollout_threads 20 --model_dir <…/run1/models>
```
Runs one policy on 20 fixed eval seeds (`seed=1×50000+rank×10000`), prints Catch% mean±std + Done%.

## SCoUT comparison caveat (code-vs-paper)
SCoUT's `/tmp/scout` clone confirms the eval *mechanism* (single-seed training; one policy looped over `episode_seeds`) but contains **no Pursuit Catch%/Done% eval** (only Battle `win_rate`) and **no literal 20** (`eval_episodes` default = 5). The "20 eval seeds" + Pursuit numbers come from the paper; runners are excluded per their README. We reproduce the mechanism; we did not verify their exact Pursuit eval script.

## Next steps (from here)
1. **Complete the eval to 20/20 paired** — stage colva4 `c4_lam90_*` + the missing plain `c4_obs98_*` checkpoints, re-run the harness → lifts the paired Wilcoxon out of underpower; fills s14–s20 above.
2. **Capacity-matched control** — shrink actor/critic toward SCoUT's ~39k actor (fewer blocks / narrower n_embd) and rerun obs98+λ0.90 on ~3 seeds (4M) to show the win isn't just parameters.
3. **Pick the reporting policy** — decide best-seed vs median-seed as the headline cell (currently best = s12).
4. Constraints in force: **4M only, no 8M extensions** ([[experiment-cadence]]); don't test DDCL or group-aware critics.
