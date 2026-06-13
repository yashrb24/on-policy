# trained_seeds_actor_shrink — RL actor-depth shrink (critic held at depth-3)

Same λ0.90 recipe and obs98 as `../trained_seeds_lam90_big`, but the **actor** transformer depth is
cut via `--actor_n_block {1,2}` while `--critic_n_block 3` is held fixed. Trained from scratch (4M,
not warm-started). Tests whether the full-size actor is over-provisioned in depth.

- **depth-1** (`--actor_n_block 1`): actor ≈ **57k params** (~47% smaller than the 107k depth-3).
- **depth-2** (`--actor_n_block 2`): actor ≈ **82k params**.

Each dir is a full checkpoint set. Eval = 20 held-out env seeds, stochastic (same protocol as the big
sweep). Numbers below are **held-out** Catch% / Done% (per `PURSUIT_ACTOR_SHRINK_RESULTS.md`).

| dir | source exp | node | actor params | held-out Catch / Done |
|-----|------------|------|-------------:|-----------------------|
| depth1_s8  | shrink_a1blk_s8  | colva1 | 57k | 100.0 / 100 |
| depth1_s11 | shrink_a1blk_s11 | colva1 | 57k | 90.0 / 45 |
| depth1_s12 | shrink_a1blk_s12 | colva2 | 57k | 92.5 / 55 |
| depth1_s19 | shrink_a1blk_s19 | colva2 | 57k | **100.0 / 100 (best trained seed)** |
| depth1_s1  | shrink_a1blk_s1  | colva4 | 57k | 76.9 / 15 |
| depth1_s6  | shrink_a1blk_s6  | colva4 | 57k | 100.0 / 100 |
| depth2_s8  | shrink_a2blk_s8  | colva1 | 82k | 91.9 / 65 |
| depth2_s12 | shrink_a2blk_s12 | colva2 | 82k | 99.4 / 95 |

**Verdict:** depth-1 mean (6 seeds) = **93.2 Catch / 69.2 Done** vs depth-3 baseline **84.1 / 40.0** —
the actor is over-provisioned in depth; cutting 3→1 block (~47% fewer actor params) is net-positive.
Depth was the exhausted lever; width needs the decoupling refactor (see distillation track, which
sidesteps it). Full report: `../PURSUIT_ACTOR_SHRINK_RESULTS.md`.
