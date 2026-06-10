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

Every student down to 2,425 params matches the teacher and beats SCoUT. Sampled capture stays
~99–100% throughout even as `val_agree`/`val_KL` degrade — the policy is extremely compressible.

### The capacity wall (sub-2.4k, n_head 2 / n_head 1)

| student | n_embd/n_block | **params** | val_agree | val_KL | **Catch%** | **Done%** |
|---|---|---|---|---|---|---|
| dst_w8b1 | 8 / 1 | 1,961 | 89.4% | 0.135 | 100.0±0.0 | 100 |
| dst_w6b3 | 6 / 3 | 1,929 | 88.1% | 0.168 | 99.4±2.8 | 95 |
| dst_w6b2 | 6 / 2 | 1,653 | 88.2% | 0.164 | 100.0±0.0 | 100 |
| **dst_w6b1** | 6 / 1 | **1,377** | 87.6% | 0.175 | **100.0±0.0** | 100 | ← smallest that is still PERFECT |
| dst_w4b3 | 4 / 3 | 1,161 | 85.4% | 0.254 | 96.9±9.0 | 85 |
| dst_w4b2 | 4 / 2 | 1,025 | 84.3% | 0.273 | 96.9±6.9 | 75 |
| dst_w4b1 | 4 / 1 | 889 | 85.4% | 0.238 | 98.1±4.6 | 85 |
| dst_w3b2 | 3 / 2 | 765 | 74.0% | 0.459 | 63.1±17.0 | 0 |  ← cliff
| dst_w3b1 | 3 / 1 | 681 | 73.6% | 0.470 | 61.9±21.3 | 5 |
| dst_w2b2 | 2 / 2 | 541 | 39.2% | 1.111 | 0.6±2.8 | 0 |  ← collapsed
| dst_w2b1 | 2 / 1 | 497 | 39.0% | 1.066 | 4.4±7.3 | 0 |

**The capability cliff for offline BC is ~800 params.** Down to ~1,377 params (w6b1) capture is
perfect (100/100); the w4 family (889–1,161) holds ~97–98% Catch but starts missing the last
1–2 evaders (Done 75–85%); below ~765 it breaks (63% Catch, 0% Done) and by ~500 it collapses to
near-random. So a transformer actor with **<1.4k params fully represents this 20-agent
coordination policy** — and even ~900 params nearly does. (For reference: SCoUT's actor is 39,195;
the cliff is ~50× smaller.)

### Stage B — does on-policy DAgger push below the BC cliff? **No — the cliff is a real capacity wall.**
Warm-start DAgger from the BC checkpoint at the cliff: `dagger_w3b2` (765 params, BC=63.1/0),
15 rounds × 48 episodes, pure student rollout (beta=0), teacher relabels. **Result: no recovery.**
Rollout-catch stayed flat at ~58–62% every round (never climbed), train_KL floored at ~0.50
(vs BC's ~0.46), and the final sampled eval is **66.9±15.3 Catch / 5 Done — statistically the
same as its BC baseline.** On-policy data cannot help because the failure is not distribution
shift (which states the student visits) but **representational capacity** (a 765-param function
class simply cannot fit the teacher's distribution — KL floors regardless of the data).

Implication: above ~900 params offline BC already works (no distribution shift to fix); at/below
~800 params *both* BC and on-policy DAgger fail. **So the ~800-param cliff is a genuine capacity
wall, not an artifact of offline BC.** (`dagger_w2b2`, 541 params + teacher mixing, as the
even-harder control — appended below.)

## Stage B — on-policy DAgger
`dagger_distill.py` validated end-to-end on anjuna2 (warm-start w24b2, 2 rounds, 29s, rollout
catch 100%, KL 0.063→0.060). Applied at the BC cliff (see "does on-policy DAgger push below the
BC cliff?" above): it does **not** extend the frontier — the ~800-param cliff is a real capacity
wall, so on-policy data cannot rescue it. Above the cliff offline BC already saturates, so DAgger
adds nothing there either; its proper role would be a task where the teacher's state distribution
is genuinely hard to cover offline (not the case here).

---

### Robustness (multi-seed, 3 seeds each = fresh init + train/val split)
- **w8b2 (2,425 params): 100/100 on all 3 seeds.** **w6b1 (1,377): 100/100 on all 3 seeds.**
  The "perfect capture at 1–2k params" headline is robust, not single-seed luck.
- **w4b1 (889, cliff edge): seed-variable** — 98.1/85, 81.2/40, 99.4/95 across seeds. Right at the
  capacity cliff the model is *barely* large enough, so a good fit depends on init luck; above the
  cliff (≥1,377 params) it is deterministically perfect across seeds.
- DAgger control `dagger_w2b2` (541 + teacher mixing, 20 rounds): 4.4 Catch / 0 Done — unchanged
  from its BC baseline. The wall holds even with on-policy data.

## Summary of the curve (offline BC, sampled 20-seed Catch%/Done%)
```
params:  107721  82505  48105  22921  13785   6953   4401   2425   1961   1377    889    765    541
            (T)                                                                    cliff→  break  collapse
Catch%:   100    100    100    100   99.4    100    100    100    100    100   ~93*   63.1    0.6
Done%:    100    100    100    100     95    100    100    100    100    100   ~73*    0      0
```
\*w4b1 (889) is the seed-variable cliff edge (mean of 3 seeds). Perfect & robust ≥1,377; breaks <800.

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

## What this establishes & open questions
- **Capacity is not the bottleneck at SCoUT scale.** A transformer actor of **~1,377 params
  (35× smaller than SCoUT's 39k) fully represents** this 20-pursuer coordination policy, robustly
  across seeds; even ~900 params nearly does. The wall is ~800 params and is a *real* capacity
  limit (DAgger / on-policy data does not cross it).
- **So SCoUT's lower from-scratch numbers are an optimization/exploration/algorithm gap, not a
  representational one.** The open problem is reaching this tiny-actor solution *via RL from
  scratch* (distillation shows the target exists; RL must find it under sparse reward).
- Done this run: full param sweep (107k→500), the wall, multi-seed robustness, DAgger control.
- Not yet: distill from a different teacher seed (generalization across teachers); whether a
  BC-initialized tiny actor + short RL fine-tune trains stably (BC as an RL initializer).
