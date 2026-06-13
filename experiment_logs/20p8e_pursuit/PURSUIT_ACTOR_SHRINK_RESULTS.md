# Pursuit Actor-Shrink + Entropy — Results

SCoUT benchmark: **20 pursuers / 8 evaders, 40×40 grid, max_cycles 500.** All runs 4M steps on the
λ0.90 recipe. Metrics from held-out evaluation (`onpolicy/scripts/train/eval_pursuit_seeds.py`):
**Catch% = captures / 8** per episode (= SCoUT Catch%); **Done% = fraction of episodes capturing all 8**.
Eval = 20 distinct held-out env seeds (`--seed 1`, one episode per rollout thread), **stochastic actions**
(the policy is intrinsically stochastic — deterministic argmax collapses every model to ~30–45% Catch).

## Reporting numbers (headline)

**Actor depth-1 (57k params, ~47% smaller than the 107k 3-block baseline):**
- **Best trained seed (s19): Catch% 100.0 ± 0.0, Done% 100.0 ± 0.0** (8/8 caught in all 20 held-out
  episodes) — identical to the full-size original. Seeds s8 and s6 also reach 100.0 / 100.0.
- **6-seed mean (robustness): Catch% 93.2, Done% 69.2** vs full-depth baseline **84.1 / 40.0**
  (depth-1 is +9.1 Catch / +29.2 Done ahead; depth-1 ≥ depth-3 on 5 of 6 seeds).
- Recommendation: lead with the 6-seed mean; cite best-seed 100/100 as the architecture's ceiling.

## Track 1 — actor depth shrink (held-out, per seed)

Actor transformer blocks cut 3 → 1, **critic held fixed at 3 blocks**. Shared recipe otherwise.

| seed | depth-1 (57k) Catch / Done | depth-3 (107k) Catch / Done | Δ Catch |
|------|----------------------------|-----------------------------|---------|
| s8  | 100.0 / 100 | 99.4 / 95  | +0.6  |
| s11 | 90.0  / 45  | 80.6 / 15  | +9.4  |
| s12 | 92.5  / 55  | 100.0 / 100 | −7.5 |
| s19 | 100.0 / 100 | 75.6 / 20  | +24.4 |
| s1  | 76.9  / 15  | 74.4 / 5   | +2.5  |
| s6  | 100.0 / 100 | 74.4 / 5   | +25.6 |
| **mean** | **93.2 / 69.2** | **84.1 / 40.0** | **+9.1** |

Depth-2 (82k) held-out: s8 = 91.9 / 65, s12 = 99.4 / 95. The per-seed depth ladder is non-monotonic
(s8: depth-1 100.0 / depth-2 91.9 / depth-3 99.4), so the differences are seed noise, not a capacity trend.

**Conclusion:** the transformer actor is over-provisioned in depth. Cutting it from 3 blocks to 1
(~47% fewer actor params) does not reduce held-out capture rate — it is slightly net-positive across the
full difficulty range. The held-out advantage (+9.1) exceeds the training-rollout advantage (+7.0).

## Track 2 — entropy coefficient (held-out)

`entropy_coef` 0.001 → 0.02, full depth-3 actor, on two weak seeds:

| seed | ent 0.02 Catch / Done | ent 0.001 baseline Catch / Done | Δ Catch / Δ Done |
|------|-----------------------|---------------------------------|------------------|
| s1 | 85.0 / 35 | 74.4 / 5 | +10.6 / +30 |
| s6 | 88.8 / 55 | 74.4 / 5 | +14.4 / +50 |

**Conclusion:** on held-out seeds the entropy-0.02 lift is large (+10–14 Catch%, all-8 win rate 5% → 35–55%),
much bigger than the training-rollout gap (+2.8) implied — entropy regularization improves generalization.

## λ0.90 recipe

`rmappo`, transformer actor+critic, obs98 (`--pursuit_drop_walls_channel`), catch 5.0 / tag 0 / urgency 0,
`entropy_coef 0.001` (0.02 for Track 2), `ppo_epoch 8`, `num_mini_batch 1`, `lr=critic_lr 3e-4`,
`gamma 0.99`, `gae_lambda 0.90`, `clip 0.2`, `max_grad_norm 1.0`, `n_embd=hidden_size 64`, `n_block 3`,
`n_head 4`, `data_chunk_length 50`, `n_rollout_threads 24`, `num_env_steps 4_000_000`. Actor depth set via
`--actor_n_block {1,2}` with `--critic_n_block 3` (default `None` → both fall back to `n_block`).

## Next lever (not yet run)

Depth is exhausted (depth-1 holds). The remaining shrink axis is actor **width** (`n_embd` / `hidden_size`),
but those are currently **shared** between actor and critic — probing a narrower actor needs a
width-decoupling code change.
