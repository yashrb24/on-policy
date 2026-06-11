# Trained seeds — 20p/8e Pursuit (index)

Checkpoints archived from the colva/anjuna2 nodes. SCoUT benchmark target: 20 pursuers / 8 evaders,
40×40, max_cycles 500; SCoUT actor 39,195 params, Catch 94±10 / Done 70. All our policies use obs98
(`--pursuit_drop_walls_channel`), λ0.90 recipe, 4M steps. Eval = sampled (stochastic) actions.

| group | runs | what | best result |
|-------|-----:|------|-------------|
| [`trained_seeds_lam90_big/`](trained_seeds_lam90_big/README.md) | 20 | full depth-3 actor (107k), the c12 lineage + distill teacher (s12) | sweep mean trail-10 ~93%; s12 held-out 100/100 |
| [`trained_seeds_actor_shrink/`](trained_seeds_actor_shrink/README.md) | 8 | RL actor-depth shrink (depth-1 57k, depth-2 82k; critic fixed depth-3) | depth-1 6-seed mean 93.2/69.2; best s19 & s6 100/100 |

**Layout:** each run dir is a checkpoint (`actor.pt` always present; RL runs also have
`critic.pt` + optimizers + `valuenorm.pt`). Load via `--model_dir <dir>`; eval needs only `actor.pt`.

**Note:** the `.pt` binaries are gitignored (~71MB) — only these `.md` manifests are tracked. To
re-pull binaries, the source paths are in each group's README. Reports:
`PURSUIT_ACTOR_SHRINK_RESULTS.md`, `PURSUIT_LAMBDA090_SETUP.md` (here).
