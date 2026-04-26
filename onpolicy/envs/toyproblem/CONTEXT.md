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
**Immediate next action:** Launch P2-A ablation: `run_p2_ablation.py --stage P2-A` (115 runs, 23 configs × 5 seeds).

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
