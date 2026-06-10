# Pursuit Actor Distillation — Results & Pipeline

**Question:** can the strong obs98+λ0.90 Pursuit teacher (decentralized rMAPPO transformer
actor, 107,721 params, 100% capture) be distilled into a **much smaller** transformer actor —
at or below SCoUT's ~39k-param actor — without losing capture performance?

**Answer (decisive): yes, drastically.** Plain offline behavioral cloning (KL distillation on
the teacher's action distribution) produces students that **match the teacher and beat SCoUT
(94±10 Catch / 70 Done) all the way down to ~2,400 params — ~1/16 of SCoUT's actor and ~1/44 of
the teacher.** No distribution-shift collapse. This means **the architecture's capacity is not
the binding constraint at SCoUT scale**; SCoUT's lower from-scratch numbers reflect an
optimization/exploration/algorithm gap, not a representational wall.

Branch: `feat/pursuit-actor-distillation` (off `origin/feat/pursuit-checkpoint-resume`).

---

## Pipeline (`onpolicy/scripts/distill/`)

| file | role |
|---|---|
| `collect_teacher_data.py` | roll the teacher **stochastically** in obs98 Pursuit, save complete episodes of `(obs, teacher_log_probs)`. Reusable across all student sizes (collect once). |
| `train_distill.py` | **Stage A — offline BC.** Build a smaller `R_Actor` (reduced `n_embd==hidden_size` / `n_block`), whole-episode BPTT from `hxs=0`, forward `KL(teacher‖student)` at T=1 on the full 5-way distribution. No critic, no RL. |
| `dagger_distill.py` | **Stage B — on-policy DAgger.** Student rolls (its own state distribution), teacher relabels each visited state, KL on the student's trajectories. Warm-starts from a BC student. |
| `run_student.sh` / `eval_student.sh` / `eval_all.sh` | launchers (`PYTHON=`/`NHEAD=` overrides). |

Core enabler: additive `R_Actor.forward_logits()` returns the full categorical logits (Discrete
only); serves both teacher data-collection and student BPTT training; **does not touch the RL
forward/evaluate_actions path.** The student is a standalone `R_Actor` — no critic, no shared
buffer — so its width/depth are free (this sidesteps the `n_embd==hidden_size` width-decoupling
refactor that blocked the RL shrink approach).

## CRITICAL: evaluate with SAMPLING, distill the FULL distribution

The teacher is **100% with action sampling but only ~35% with argmax** — greedy actions collapse
the 20-agent coordination into a symmetric deadlock; sampling breaks symmetry. The eval flag
`--eval_deterministic` is `store_false`, so **passing it selects stochastic sampling** (the
documented "deterministic" eval is actually sampled). Consequences, both load-bearing here:
- All eval below uses **sampling** (`eval_student.sh` passes `--eval_deterministic`).
- Distillation targets the **full softmax distribution (KL)**, not hard argmax labels — argmax
  BC would discard the very stochasticity the policy relies on.

## Teacher & dataset

- Teacher: `c1_lam90_s12_models/` (s12, the top λ0.90 policy; delivered to anjuna2). obs98,
  n_embd 64, n_block 3, n_head 4, GRU 64, Discrete(5). Sampled eval: **100.0±0.0 / 100**.
- Dataset: 600 stochastic teacher episodes, **3.07M agent-steps**, mean len 255
  (`data/s12_obs98_stoch_600.pt`, gitignored). One dataset trains every student.

---

## Results — sampled 20-seed eval (SCoUT protocol). SCoUT ref: actor 39,195, Catch 94±10, Done 70.

`val_agree` = student-vs-teacher argmax agreement on held-out episodes; `val_KL` = distillation
KL. All students distilled from the **same** 600-episode dataset (offline BC, 40–60 epochs).

| student | n_embd/n_block | **params** | val_agree | val_KL | **Catch%** | **Done%** |
|---|---|---|---|---|---|---|
| teacher | 64 / 3 | 107,721 | — | — | **100.0±0.0** | 100 |
| dst_w64b2 | 64 / 2 | 82,505 | 96.4% | 0.027 | 100.0±0.0 | 100 |
| dst_w48b3 | 48 / 3 | 62,409 | 95.9% | 0.033 | 98.1±6.1 | 90 |
| dst_w64b1 | 64 / 1 | 57,289 | 95.9% | 0.031 | 99.4±2.8 | 95 |
| dst_w48b2 | 48 / 2 | 48,105 | 95.8% | 0.033 | 100.0±0.0 | 100 |
| **dst_w48b1** | 48 / 1 | **33,801** | 95.5% | 0.036 | 99.4±2.8 | 95 | ← ≈ SCoUT params |
| dst_w32b3 | 32 / 3 | 29,385 | 94.8% | 0.047 | 100.0±0.0 | 100 |
| dst_w32b2 | 32 / 2 | 22,921 | 94.9% | 0.046 | 100.0±0.0 | 100 |
| dst_w24b2 | 24 / 2 | 13,785 | 94.1% | 0.059 | 99.4±2.8 | 95 |
| dst_w24b1 | 24 / 1 | 10,089 | 93.8% | 0.061 | 100.0±0.0 | 100 |
| dst_w16b3 | 16 / 3 | 8,649 | 93.1% | 0.074 | 99.4±2.8 | 95 |
| dst_w16b2 | 16 / 2 | 6,953 | 92.8% | 0.076 | 100.0±0.0 | 100 |
| dst_w12b3 | 12 / 3 | 5,385 | 91.9% | 0.093 | 100.0±0.0 | 100 |
| dst_w16b1 | 16 / 1 | 5,257 | 92.6% | 0.080 | 99.4±2.8 | 95 |
| dst_w12b2 | 12 / 2 | 4,401 | 91.4% | 0.094 | 100.0±0.0 | 100 |
| dst_w8b3 | 8 / 3 | 2,889 | 89.8% | 0.126 | 99.4±2.8 | 95 |
| **dst_w8b2** | 8 / 2 | **2,425** | 89.7% | 0.134 | **100.0±0.0** | 100 | ← 6.2% of SCoUT's actor |

Every student matches the teacher and beats SCoUT. `val_agree`/`val_KL` degrade smoothly with
size, but sampled capture stays ~99–100% throughout — the policy is extremely compressible.

### Capacity floor (sub-2.4k probe, in progress)
Wave 3 (`w8b1`..`w4b1`, n_head 2) and wave 4 (`w3`/`w2`, n_head 1) push into the hundreds of
params to locate the actual wall. `w8b1` = **1,961 params** trains to 89.4% agree. *(Catch%
for these pending eval — appended below.)*

## Stage B — on-policy DAgger (validated)
`dagger_distill.py` validated end-to-end on anjuna2 (warm-start w24b2, 2 rounds, 29s, rollout
catch 100%, KL 0.063→0.060). Planned use: at the BC wall, warm-start + DAgger to test whether
on-policy data extends the compression frontier *below* where offline BC degrades. *(Results
appended below.)*

---

## Reproduce
```bash
# collect (once)
onpolicy/scripts/distill/collect_teacher_data.py --teacher_model_dir c1_lam90_s12_models \
  --num_episodes 600 --n_rollout_threads 24 --out_path .../s12_obs98_stoch_600.pt  <obs98 + arch flags>
# train a student (e.g. ~2.4k params)
bash onpolicy/scripts/distill/run_student.sh dst_w8b2 8 2 <gpu> <data.pt> 40
# eval (SAMPLED — note the inverted flag is handled inside eval_student.sh)
bash onpolicy/scripts/distill/eval_student.sh dst_w8b2 8 2 <gpu> <model_dir> 20
```
Infra: anjuna2 uses `/usr/bin/python3`; anjuna3's `/usr/bin/python3` lacks deps — use its
`.venv/bin/python` (set `PYTHON=`). Training a student = ~45–70s for 40 epochs on a 4060 Ti.

## Open questions / next steps
1. Locate the exact capacity wall (waves 3–4) and whether DAgger pushes below it.
2. Multi-seed confirmation of the smallest working size (rule out single-seed luck).
3. The real lever for matching SCoUT *from scratch* is optimization, not capacity — distillation
   shows a tiny actor CAN represent the policy; reaching it via RL is the open problem.
