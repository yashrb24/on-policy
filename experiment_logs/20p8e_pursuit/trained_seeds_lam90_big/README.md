# trained_seeds_lam90_big — λ0.90 full-size (depth-3) Pursuit policies

20-pursuer / 8-evader, 40×40, max_cycles 500. The **big-model** seed sweep: full transformer
actor+critic, `n_embd=hidden 64`, **`n_block 3`**, `n_head 4`, GRU 64, Discrete(5), **obs98**
(`--pursuit_drop_walls_channel`), λ0.90 recipe, 4M steps. **Actor = 107,721 params.** This is the
**c12 lineage** — `s12` is the policy we shrank from and the **teacher** for the distillation track.

Each `sN/` is a full checkpoint dir (`actor.pt`, `critic.pt`, `actor_optimizer.pt`,
`critic_optimizer.pt`, `valuenorm.pt`) — load with `--model_dir <path>`. For eval only `actor.pt` is
needed. Eval protocol: 20 held-out env seeds, **stochastic** actions (argmax collapses every model
to ~30–45%). Catch% = captures/8; Done% = fraction of episodes catching all 8.

`trail-10` below = final training trailing-10 capture %. Source exp = the per-node run dir these were
copied from (`results/Pursuit/rmappo/<exp>/run1/models`).

| seed | node | source exp | trail-10 % | band |
|------|------|------------|-----------:|------|
| s1  | colva2 | c2_lam90_s1  | 78.44 | rescued→mid |
| s2  | colva2 | c2_lam90_s2  | 98.76 | high |
| s3  | colva2 | c2_lam90_s3  | 97.95 | high |
| s4  | colva2 | c2_lam90_s4  | 95.75 | high |
| s5  | colva2 | c2_lam90_s5  | 94.67 | high |
| s6  | colva2 | c2_lam90_s6  | 79.75 | mid |
| s7  | colva2 | c2_lam90_s7  | 97.07 | high |
| s8  | colva1 | c1_lam90_s8  | 98.84 | high |
| s9  | colva1 | c1_lam90_s9  | 94.39 | high |
| s10 | colva1 | c1_lam90_s10 | 97.18 | rescued→high |
| s11 | colva1 | c1_lam90_s11 | 80.29 | mid |
| **s12** | colva1 | c1_lam90_s12 | **99.41** | **high — TEACHER / shrink-from; held-out 100.0±0.0 / 100** |
| s13 | colva1 | c1_lam90_s13 | 97.96 | high |
| s14 | colva1 | c1_lam90_s14 | 99.29 | high (largest rescue, +60 vs plain) |
| s15 | colva4 | c4_lam90_s15 | 87.62 | high |
| s16 | colva4 | c4_lam90_s16 | 98.52 | high |
| s17 | colva4 | c4_lam90_s17 | 98.40 | high |
| s18 | colva4 | c4_lam90_s18 | 94.81 | high |
| s19 | colva4 | c4_lam90_s19 | 78.30 | mid (slow-climber) |
| s20 | colva4 | c4_lam90_s20 | 86.06 | mid→high |

Sweep mean trail-10 ≈ 93.2%. Equal-compute (4M) λ0.90 beats plain obs98 by ~+11 pts.
Full sweep narrative: `../../../pursuit_campaign_log.md` (Mac) / `PURSUIT_LAMBDA090_SETUP.md`.
