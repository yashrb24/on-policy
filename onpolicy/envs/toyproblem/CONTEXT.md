# DDCL Unleashed — Project Context

## MANDATORY UPDATE PROTOCOL

Every session must follow these rules — no exceptions.

**At session start:** Read this file. Then read the relevant section of `PLAN.md` for the current phase. If working on a pillar (Phase 3), also read `docs/pillars/PILLAR_P<N>.md` for the pillar being worked on — it contains the full theoretical and implementation spec.

**During the session — update the right file immediately when something happens:**

| Event | File to update |
|-------|---------------|
| Bug found or fixed | `docs/ISSUES_TRACKER.md` — add entry before fixing |
| Code file created or changed | `PLAN.md` — mark task done; update phase status if phase completes |
| New sweep results generated | Re-run `report_baseline.py`; replace `results/toyproblem/<sweep>/` |
| Experiment procedure changes | `docs/README.md §5` |
| Mathematical finding or correction | `docs/MATH.md` |
| Statistical method added or changed | `docs/STATS.md` |
| Session ends | Add entry to Session Log below; update "Current State" block |

**Never duplicate information across files.** Each fact lives in exactly one place:
- What the code does → `docs/README.md`
- What is planned and what is done → `PLAN.md`
- What broke and how it was fixed → `docs/ISSUES_TRACKER.md`
- Mathematical derivations → `docs/MATH.md`
- Statistical methods → `docs/STATS.md`
- Pillar theory and implementation spec → `docs/pillars/PILLAR_P<N>.md`
- Current running state → this file (CONTEXT.md) only

---

## Project in One Paragraph

Rigorous testbed for DDCL (Differentiable Discrete Communication Learning) on a toy 2-agent MARL problem (`CommunicatingGoal`). Goals: (1) verify mathematical correctness, (2) find strong baseline via systematic sweeps, (3) implement and ablate 4 principled extensions ("pillars"), (4) show scaling behaviour. All results are statistically honest: raw CSVs, bootstrap CIs, permutation tests, IQM. Full plan: `PLAN.md`. Bug log: `docs/ISSUES_TRACKER.md`. How to run: `docs/README.md §5`.

**Environment:** `marl_comms` conda env (Python 3.10). Prefix every command: `KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms <cmd>` (macOS OpenMP conflict workaround — INSTALL-005).

---

## Current State

**Phase:** 2 → 3 transition — Stage A sweep complete; baseline frozen; P2-A ablation ready to launch  
**Running:** Nothing. Stage A sweep complete (2175 / 2175 runs). Data at `runs/toyproblem/sweep_stage_a/` (canonical path).  
**P2 status:** Implementation complete (47 tests, 0 failures). All 3 hardening fixes applied (mixture prior, scale floor, context B backward disabled). DLM floor CDF fix applied (bin [m,m+1) not [m-0.5,m+0.5)). Prior-based bit cost active when P2 entropy model is on. Device auto-selection (MPS > CUDA > CPU) in train.py.  
**Baseline:** FROZEN — `configs/baseline_best.yaml`: channel=sd, delta=1.0, lambda_comms=5e-4, z_dim=2. Success_rate=1.000 ± 0.000 across 5 seeds; true_bits=4.75; interior optimum (delta=1.0 ∈ (0.5,20)); Pareto-optimal (fewest bits at perfect SR).  
**paper_figures.py:** Fully rewritten (10 publication-quality figures). Fig 1 uses per-channel λ-sweep trade-off curves (one line per δ, sorted by λ) so SD's dominance is visible across the entire search space, not just the best config. H(G) labelled "min. bits for SR=1". AppB annotation repositioned below tick labels.  
**P2 metrics:** Extended — shannon_gap, bits_to_hg_ratio, warm_start_bits_final, H_dim_k, entropy_rate_B, context_gap_bits, entropy_loss_magnitude, speaker_grad_norm. Companion context-B model auto-runs alongside context-A. CSV header now dynamic via `_build_csv_header(z_dim)`.  
**P2-A result:** No P2 config beats baseline on shannon_gap (baseline=2.48, best P2=2.90). entropy_rate_B≈0.7 bits confirms m near-deterministic given z. Entropy backward gradient ~1% of speaker gradient at λ=5e-4 — too small. K(1→20) and loss_mode(both vs entropy) have negligible effect.  
**Immediate next action:** Re-run P2-A sweep after CODE-010/CODE-011 fixes (metrics were wrong). Then decide on P2-B λ re-sweep or redesign.

**Directory layout (canonical, from repo root):**
- Raw runs: `runs/toyproblem/<experiment>/` (gitignored)
- Analysis output: `results/toyproblem/<experiment>/` (gitignored)
- Committed docs: `onpolicy/envs/toyproblem/docs/` (no data here)

**Check sweep:** `ps aux | grep run_sweep | grep -v grep`  
**Completed runs:** `find onpolicy/envs/toyproblem/runs/sweep_stage_a -name metrics.csv | wc -l`

**After sweep finishes — migrate to canonical path:**
```bash
mkdir -p runs/toyproblem
mv onpolicy/envs/toyproblem/runs/sweep_stage_a runs/toyproblem/sweep_stage_a
```

**Restart if needed** (uses canonical path for new runs):
```bash
nohup bash -c 'cd "$(pwd)" && KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.scripts.sweeps.toyproblem.run_sweep \
    --config onpolicy/scripts/sweeps/toyproblem/configs/sweep_stage_a.yaml \
    --seeds 0 1 2 3 4 --log_dir runs/toyproblem/sweep_stage_a \
    --resume --shuffle' > /tmp/sweep_stage_a.log 2>&1 &
```

---

## Session Log

*Keep entries concise. One paragraph per session maximum.*

**Session 16 (2026-04-27):** CODE-010 + CODE-011 fixed. (1) CODE-010: `entropy_rate = nll_log.mean()` was per-element NLL (NLL_joint/z_dim) compared against H_joint — dimensional mismatch producing negative qphi_gap for z_dim>1. Fixed to `nll_log.sum(dim=-1).mean()` (joint NLL per message) in `trainer.py`. Same fix applied to `entropy_rate_B` and `qphi_neg_log_max`. Gibbs was always satisfied at joint level; bug was purely a units error. (2) CODE-011: per-goal column renamed `entropy_rate_goal_*` → `nll_goal_*` in `trainer.py` and `train.py` to accurately reflect it stores model NLL not empirical entropy. (3) Added `test_qphi_gap_gibbs_nonnegative` test covering z_dim ∈ {1,2,3} — now 127 tests, 0 failures. (4) P2-A ablation results (runs/toyproblem/p2_ablation) are invalid under old metrics; need re-run. paper_figures.py redesign complete with per-channel Pareto frontier + top-K tradeoff lines + CI bands + inset zoom.

**Session 15 (2026-04-27):** P2 metrics expansion and fig1 redesign. (1) fig1 rate-distortion: replaced background dots with per-channel λ-sweep trade-off curves (one line per δ, sorted by λ), Float32 annotation moved to top-right to avoid legend overlap, alpha 0.35. (2) New metrics in trainer.py: shannon_gap, bits_to_hg_ratio, warm_start_bits_final, H_dim_k (per-dimension empirical entropy), entropy_rate_B + context_gap_bits (companion context-B oracle), entropy_loss_magnitude, speaker_grad_norm. Added marginal_entropies_bits() to network.py. (3) Companion context-B model auto-created alongside context-A primary for oracle bound measurement in every run. (4) CSV header moved to _build_csv_header(z_dim) inside main() for dynamic H_dim_k columns. (5) Five new P2 analysis figures: p2_shannon_gap, p2_qphi_gap, p2_context_bounds, p2_per_dim_entropy, p2_gradient_balance. (6) PILLAR_P2.md §7 updated with full metrics table. 126 tests passing. P2-A sweep launching.

**Session 14 (2026-04-27):** Bug fixes and paper_figures.py rewrite. (1) Fixed 3 PPO training correctness bugs: ValueNorm.update() called once per rollout not once per minibatch (40× over-update); advantage normalised per-minibatch not once before all epochs; CSV file leak — try/finally in train.py main(). (2) Minor fixes: global RNG side-effect removed from CommunicatingGoal_env.seed(); stale comment corrected in network.py; fragile positional seed pairing fixed in stats.py compare_configs(). (3) paper_figures.py fully rewritten: 10 publication-quality figures (2 main + 8 appendix), Float32 as text annotation only, H(G) on all bits axes, legends outside data area. Fig 1 updated to per-channel λ-sweep trade-off curves (one line per δ sorted by λ) so SD's dominance is visible across the entire search space. AppA H(G) labelled "min. bits for SR=1". AppB "Best: δ=…" annotation repositioned to y=0.06 with bottom=0.26. All changes committed (dcef2e4, 93ccf29, latest).

**Session 13 (2026-04-25):** Completed Phase 2 analysis. (1) Fixed DLM floor CDF bug: all 4 `_dlm_log_prob` implementations used rounding bins [m-0.5,m+0.5) but DDCL channels use floor m=floor((z+noise)/delta), correct bins are [m,m+1) — fixed by upper=sigma((x+1-mu)/s), lower=sigma((x-mu)/s). (2) Prior-based bit cost: when P2 entropy model active, canonical bits_per_msg=-log2 q_phi(m) (learned-code rate); mag_bits_per_msg preserved for comparison. (3) Device auto-selection: select_device("auto") in train.py, MPS>CUDA>CPU. (4) Stage A sweep migration (1867->2175 runs to canonical path) + 15 figures generated (7 standard + 8 paper PDFs). (5) Baseline frozen: sd/delta=1.0/lambda=5e-4/z_dim=2, success=1.0 on 5/5 seeds, 4.75 true bits, interior optimum — saved to configs/baseline_best.yaml. Updated MATH.md §12, PILLAR_P2.md §3/§10, network.py, trainer.py, train.py, tests. All changes committed.

**Session 12 (2026-04-25):** Applied 3 hardening fixes to P2 entropy model. Fix 1: mixture prior q̃_φ = (1−α)q_φ + α·Laplace(0,50) added to all 4 entropy model classes — guarantees nonzero gradient everywhere, eliminating gradient dead-zone without warm-start dependency. Fix 2: DLM scale floor s≥0.1 (was s≥0.5, which capped joint conditional probability at 0.46 and broke V4). Fix 3: context B backward loss disabled — q_φ(m|z) conditioning on z is degenerate since m=round(z/δ) is deterministic; context B is now measurement-only. Update order also corrected: q_φ forward update runs before RL step (fresher rate signal). Added 3 new validation tests (V6/V7/V8). All 47 tests pass. Mathematical basis documented in MATH.md §12 and PILLAR_P2.md §4/§5.

**Session 11 (2026-04-25):** Implemented Pillar P2 (Entropy Model) end-to-end. Added EntropyModelFactored, EntropyModelJoint, EntropyModelCondZ, EntropyModelJointCondZ to network.py (closing 2×2 factored/joint × context A/B grid) plus joint_entropy_bits and total_correlation_bits helpers. Extended MAPPOConfig with 8 P2 fields; integrated Ballé two-term loss (fwd trains q_φ, bwd propagates to speaker with frozen q_φ) and warm-start into trainer.py. Added 8 CLI flags and P2 CSV columns to train.py. Smoke test confirmed entropy_rate finite and non-NaN. Added run_p2_ablation.py (6-stage, 395 runs) and 4 P2 paper figures. All 118 tests pass. Then added 5 rigorous mathematical validation tests (V1–V5) covering DLM normalisation, entropy convergence, Ballé gradient direction, TC identity, and warm-start convergence — now 123 tests total. Key empirical findings: (1) DLM has irreducible ~0.27 bits/dim approximation floor (qphi_gap floor ≈ 0.27 × z_dim); (2) warm-start is critical not optional — without it q_φ assigns probability < 1e-10 to unseen messages, killing Ballé backward gradient entirely; (3) TC identity confirmed empirically (factored−joint gap ≈ TC + ε_DLM); (4) use tc_bits not NLL gap for model selection. All findings documented in MATH.md §11, PILLAR_P2.md §2/§5/§6/§7/§9. Phase 2 Stage A sweep still running at ~86% complete (PID 11556) — do NOT migrate or analyse until complete.

**Session 10 (2026-04-24):** Designed P2 — Entropy Model pillar in full. Prior family: Discretised Logistic Mixture (DLM), K components, factored primary / joint autoregressive ablation. Context A (marginal prior) is primary; B (condition on z) and C (condition on h) are supplementary ablations. Gradient path: Ballé-style two-term loss (fwd trains q_φ on discrete m; bwd propagates to speaker via continuous relaxation). Metrics: entropy_rate, H_m_empirical, qphi_gap, tc_bits, qphi_neg_log_max. Unbounded m handled naturally by DLM tails. Moving target mitigated by lr_qphi, n_qphi_steps, warm-start. Wrote `docs/pillars/PILLAR_P2.md` (full spec), stubs `PILLAR_P1.md`, `PILLAR_P3.md`, `PILLAR_P4.md`. Updated CONTEXT.md mandatory protocol to reference pillar docs at session start.

**Session 9 (2026-04-24):** Unified data layout. Canonical paths: `runs/toyproblem/<exp>/` and `results/toyproblem/<exp>/` at repo root (both gitignored). Deleted stale top-level `docs/` and `runs/` (pre-consolidation artefacts). Moved `toyproblem/runs/channel_comparison` → `runs/toyproblem/channel_comparison`; `toyproblem/docs/results/channel_comparison` → `results/toyproblem/channel_comparison`. Active sweep_stage_a left in place (sweep still running). Removed `plots/` and `paper_figures/` split — all figures go to `figures/`. Updated all script defaults and docs. Committed `f2c7a6a` and follow-on.

**Session 8 (2026-04-24):** Completed 7→5 file consolidation: KNOWN_ISSUES.md content migrated into ISSUES_TRACKER.md as UPSTREAM-001–004; KNOWN_ISSUES.md deleted; CODE-003 status updated. CONTEXT.md committed in new slim format. Committed `81a9c59`.

**Session 7 (2026-04-24):** Consolidated documentation from 7 → 5 files. CONTEXT.md stripped to current-state-only; PLAN.md is now single source of truth for phase status and what-is-done; KNOWN_ISSUES.md deleted and content moved to ISSUES_TRACKER.md §Pre-existing Issues; config.json exp_name fixed in all 95 recovered runs; stale `docs/results/sweep_stage_a/` deleted. Committed `2be01be`, `921946a`.

**Session 6 (2026-04-24):** CODE-009 — `_run_key()` 80-char truncation caused 2080 runs silently skipped; fixed to use only varying params; 95 runs recovered from checkpoint args and renamed. Sweep restarted. All Phase 2 infrastructure committed. Channel comparison and partial Stage A plots generated. CONTEXT/ISSUES_TRACKER/README updated.

**Session 5 (2026-04-24):** Pre-sweep audit: CODE-007 (additive_uniform missing from Stage A), CODE-008 (none channel redundancy). Jensen gap + RL uncertainty model added to per-goal plot. `analysis/paper_figures.py` created (12 figures). Results path moved to toyproblem-local `runs/`. PLAN/CONTEXT/ISSUES_TRACKER updated.

**Session 3 (2026-04-22):** Phase 2 infrastructure complete. Analysis module, stats, plots, sweep convergence, report_baseline, paper_figures. Baseline channels (additive_uniform, gaussian, ste4/8/16). Experiments (validate_env, validate_mappo, run_channel_comparison). Sweep configs and run_sweep.py. 79 tests passing.

**Session 2 (2026-04-22):** Phase 1 complete. 51 tests passing. Bit-cost formula fixed. MATH-002: NSD variance = δ²/4 not δ²/6. Per-goal bit logging (buffer goal_ids, trainer metrics, train.py CSV columns). `configs/baseline_v0.yaml` created.

**Session 1 (2026-04-22):** Phase 0 complete. 5-phase plan designed and approved. docs/ (README, MATH, KNOWN_ISSUES), tests/ skeleton, train.py hardened (set_seed, --exp_name, log dir). INSTALL-001–005 resolved.
