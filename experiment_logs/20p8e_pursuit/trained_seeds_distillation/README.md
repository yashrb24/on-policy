# trained_seeds_distillation — distilled students from teacher s12

37 runs distilled from the **teacher = `../trained_seeds_lam90_big/s12`** (107,721-param λ0.90 obs98
policy, sampled eval 100.0±0.0 / 100). Branch `feat/pursuit-actor-distillation`. Pipeline:
`onpolicy/scripts/distill/` (offline behavioral cloning on the teacher's full action distribution,
forward KL at T=1, whole-episode BPTT; DAgger variant for the on-policy probe).

- Naming: `dst_w<n_embd>b<n_block>` (= student width / depth). `_s2`/`_s3` = seed replicas.
  `dagger_*` = on-policy DAgger probe. `smoke_student_*` = smoke test (ignore).
- One dataset trains every student: 600 stochastic teacher episodes, 3.07M agent-steps
  (`data/s12_obs98_stoch_600.pt`, gitignored — regenerate with `collect_teacher_data.py`).
- Each dir: `actor.pt` (standalone `R_Actor`, no critic) + `distill_best.txt` (epoch/val_KL/val_agree/params).
- Eval = sampled 20-seed (SCoUT protocol). **SCoUT ref: actor 39,195 params, Catch 94±10, Done 70.**

**Headline: BC matches the teacher and beats SCoUT down to ~2,400 params (1/16 of SCoUT, 1/44 of the
teacher). Capability cliff ≈ 800 params.** Capacity, not algorithm, is far from the binding constraint.

### Offline BC students (sampled eval)
| dir | n_embd/n_block | params | val_agree | val_KL | Catch% | Done% |
|---|---|---:|---|---|---|---|
| dst_w64b2 | 64/2 | 82,505 | 96.4% | 0.027 | 100.0±0.0 | 100 |
| dst_w48b3 | 48/3 | 62,409 | 95.9% | 0.033 | 98.1±6.1 | 90 |
| dst_w64b1 | 64/1 | 57,289 | 95.9% | 0.031 | 99.4±2.8 | 95 |
| dst_w48b2 | 48/2 | 48,105 | 95.8% | 0.033 | 100.0±0.0 | 100 |
| **dst_w48b1** | 48/1 | **33,801** | 95.5% | 0.036 | 99.4±2.8 | 95 | ← ≈ SCoUT params |
| dst_w32b3 | 32/3 | 29,385 | 94.8% | 0.047 | 100.0±0.0 | 100 |
| dst_w32b2 | 32/2 | 22,921 | 94.9% | 0.046 | 100.0±0.0 | 100 |
| dst_w24b2 | 24/2 | 13,785 | 94.1% | 0.059 | 99.4±2.8 | 95 |
| dst_w24b1 | 24/1 | 10,089 | 93.8% | 0.061 | 100.0±0.0 | 100 |
| dst_w16b3 | 16/3 | 8,649 | 93.1% | 0.074 | 99.4±2.8 | 95 |
| dst_w16b2 | 16/2 | 6,953 | 92.8% | 0.076 | 100.0±0.0 | 100 |
| dst_w12b3 | 12/3 | 5,385 | 91.9% | 0.093 | 100.0±0.0 | 100 |
| dst_w16b1 | 16/1 | 5,257 | 92.6% | 0.080 | 99.4±2.8 | 95 |
| dst_w12b2 | 12/2 | 4,401 | 91.4% | 0.094 | 100.0±0.0 | 100 |
| dst_w8b3 | 8/3 | 2,889 | 89.8% | 0.126 | 99.4±2.8 | 95 |
| **dst_w8b2** | 8/2 | **2,425** | 89.7% | 0.134 | 100.0±0.0 | 100 | ← 6.2% of SCoUT's actor |

### Capacity wall (sub-2.4k)
| dir | n_embd/n_block | params | val_agree | val_KL | Catch% | Done% |
|---|---|---:|---|---|---|---|
| dst_w8b1 | 8/1 | 1,961 | 89.4% | 0.135 | 100.0±0.0 | 100 |
| dst_w6b3 | 6/3 | 1,929 | 88.1% | 0.168 | 99.4±2.8 | 95 |
| dst_w6b2 | 6/2 | 1,653 | 88.2% | 0.164 | 100.0±0.0 | 100 |
| **dst_w6b1** | 6/1 | **1,377** | 87.6% | 0.175 | 100.0±0.0 | 100 | ← smallest still PERFECT |
| dst_w4b3 | 4/3 | 1,161 | 85.4% | 0.254 | 96.9±9.0 | 85 |
| dst_w4b2 | 4/2 | 1,025 | 84.3% | 0.273 | 96.9±6.9 | 75 |
| dst_w4b1 | 4/1 | 889 | 85.4% | 0.238 | 98.1±4.6 | 85 |
| dst_w3b2 | 3/2 | 765 | 74.0% | 0.459 | 63.1±17.0 | 0 | ← cliff |
| dst_w3b1 | 3/1 | 681 | 73.6% | 0.470 | 61.9±21.3 | 5 |
| dst_w2b2 | 2/2 | 541 | 39.2% | 1.111 | 0.6±2.8 | 0 | ← collapsed |
| dst_w2b1 | 2/1 | 497 | 39.0% | 1.066 | 4.4±7.3 | 0 |

### Seed replicas (BC, alt init — see each dir's distill_best.txt)
`dst_w4b1_s2` (889, val_agree 80.4%), `dst_w4b1_s3` (889, 84.5%), `dst_w6b1_s2` (1377, 87.7%),
`dst_w6b1_s3` (1377, 86.0%), `dst_w8b2_s2` (2425, 90.1%), `dst_w8b2_s3` (2425, 89.7%).

### On-policy DAgger probe (does it push below the BC cliff? → No, capacity wall is real)
`dagger_w3b2` (765 params, warm-started from w3b2 BC=63.1/0): 15 rounds × 48 ep, β=0 → **66.9±15.3 /
5** — statistically unchanged. `dagger_w2b2` (541 + teacher mixing). `dagger_smoke_a2` = smoke test.
`smoke_student_w32b1` = smoke test (6 epochs, ignore). distill_best.txt is empty for the dagger runs
(different pipeline; metrics in the campaign/distillation log).

Full report: anjuna2 `PURSUIT_DISTILLATION.md`.
