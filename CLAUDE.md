# CLAUDE.md — on-policy project

Project-specific context for the DDCL Unleashed / Stochastic Quantisation toy-problem
experiments. Behavioural coding guidelines live in the parent-directory CLAUDE.md.

---

## Repository layout

```
on-policy/
  onpolicy/envs/toyproblem/     ← all toy-problem source code
    train.py                    ← entry point: python -m onpolicy.envs.toyproblem.train
    trainer.py                  ← PPO + MAPPO training loop with SC hooks
    network.py                  ← SpeakerNetwork, ListenerNetwork
    channels.py                 ← SD / NSD channel implementations
    source_coding.py            ← MessageHistogram, JointMessageHistogram,
                                   dither_channel_loss, dither_channel_stats
    CommunicatingGoal_env.py    ← 6-goal toy environment
    analysis/                   ← post-training analysis scripts
      aggregate_results.py      ← rebuilds results/aggregated/summary.csv (run after every new experiment)
      post_hoc_coding.py        ← rate decomposition from a checkpoint
      message_geometry.py       ← z-space / message-space / frac / confusion figure
      paper_figures.py          ← publication-quality figures (F1–F20)
      load_runs.py              ← load metrics.csv into DataFrames
    experiments/                ← experiment launchers
      run_sc_experiments.py     ← stages: BASELINE SC_MAIN SC_TWOPHASE PARETO GEOMETRY LIVE
      run_p2_ablation.py        ← DLM comparison stages (P2-FIX etc.)
    docs/                       ← theory documentation (pillars, math, issues)
      pillars/PILLAR_P2_v2.md   ← primary P2 design doc (includes §16 figure list, §17 results)
      MATH.md                   ← rate decomposition derivations
      ISSUES_TRACKER.md
    tests/                      ← unit tests

  runs/                         ← raw training outputs (metrics.csv, final.pt, config.json)
    sc_ablation/                ← current experiment batch (35 configs × 5 seeds)

  results/                      ← ALL processed outputs (single canonical location)
    aggregated/summary.csv      ← mean±std at convergence per experiment (auto-generated)
    post_hoc/<exp>/seed_N.json  ← rate decomposition JSONs; aggregate.json per experiment
    figures/                    ← ALL figures for the toyproblem (single media directory)
      message_geometry.{png,pdf}
      archive/                  ← pre-registry era figures
        sweep_stage_a/          ← main / appendix / p2 figures from early sweep
        p2_ablation/            ← DLM-era ablation figures
        p2_gradient_balance.*   ← standalone early figures
        p2_per_dim_entropy.*
    reports/                    ← text reports from pre-registry runs

  docs/                         ← project-level navigation documents
    JOURNAL.md                  ← chronological scientific log (update after every experiment batch)
    CONCLUSIONS.md              ← living conclusions (C1–C10); update when findings change
    EXPERIMENT_REGISTRY.md      ← all experiments with status, figures, dependencies

  scripts/
    run_p2_experiments.sh       ← resource-aware launcher (MAX_PARALLEL, LOAD_PER_CORE)
```

---

## Canonical output paths

| Script | Output |
|---|---|
| `aggregate_results.py` | `results/aggregated/summary.csv`, `results/post_hoc/*/aggregate.json` |
| `post_hoc_coding.py` | `results/post_hoc/<exp_name>/seed_<seed>.json` (inferred from checkpoint path) |
| `message_geometry.py` | `results/figures/message_geometry.{png,pdf}` |
| `paper_figures.py` | `results/figures/<figure_stem>.{png,pdf}` (pass `--out_dir results/figures`) |
| `report_baseline.py` | `results/figures/<stem>.{png,pdf}`, `results/reports/<stem>.md` |
| `checkpoint_analysis_figures.py` | `results/figures/F3_implicit_prior_mismatch.{png,pdf}`, `F4_gradient_direction.{png,pdf}`, `F8_qphi_gap_comparison.{png,pdf}`, `F10_circular_gradient.{png,pdf}`, `F11_context_bound_ordering.{png,pdf}`, `F12_warmstart_failure.{png,pdf}`, `F14_score_function_grad_ratio.{png,pdf}`, `F17_eps_mle_convergence.{png,pdf}`, `F19_moving_target_error.{png,pdf}` |
| `geometry_phase2_figures.py` | `results/figures/F6_z_geometry.{png,pdf}`, `F7_geometry_decomposition.{png,pdf}`, `F16_phase2_curves.{png,pdf}` |
| `sc_ablation_figures.py` | `results/figures/F21_phase2_ablations.{png,pdf}` |

---

## Figure-to-experiment mapping (F1–F21)

| Figure | Status | Data source | Experiment(s) |
|---|---|---|---|
| F1 rate decomposition bar | **DONE** `results/figures/F1_rate_decomposition.{pdf,png}` | `results/post_hoc/sc_posthoc_mag/` | E09 |
| F2 Shannon gap curve | **DONE** `results/figures/F2_shannon_gap.{pdf,png}` | `results/aggregated/summary.csv` | E01–E11 |
| F3 implicit prior mismatch | **DONE** `results/figures/F3_implicit_prior_mismatch.{pdf,png}` | `results/post_hoc/sc_posthoc_mag/` + E09 checkpoint | E09 |
| F4 gradient direction | **DONE** `results/figures/F4_gradient_direction.{pdf,png}` | E09 + E10 `metrics.csv` | E09, E10 |
| F5 Pareto frontier | **DONE** `results/figures/F5_pareto_frontier.{pdf,png}` | summary.csv | E23–E36 |
| F6 per-goal z geometry | **DONE** `results/figures/F6_z_geometry.{pdf,png}` | geom_*_v2 checkpoints | E16, E17, E42, E43 |
| F7 TC and H_dither per config | **DONE** `results/figures/F7_geometry_decomposition.{pdf,png}` | `results/post_hoc/geom_*/` | E16, E17, E42, E43 |
| F8 qphi_gap DLM vs hist | **DONE** `results/figures/F8_qphi_gap_comparison.{pdf,png}` | `runs/p2_ablation/` metrics | E41 |
| F9 DLM K sweep | deferred — P2-A not run; F8/F10/F12 sufficient for paper | — | — |
| F10 DLM circular gradient | **DONE** `results/figures/F10_circular_gradient.{pdf,png}` | `runs/p2_ablation/` metrics | E41 |
| F11 context bound ordering | **DONE** `results/figures/F11_context_bound_ordering.{pdf,png}` | `results/post_hoc/sc_posthoc_mag/` | E09 |
| F12 DLM warm-start failure | **DONE** `results/figures/F12_warmstart_failure.{pdf,png}` | `runs/p2_ablation/` metrics | E41 |
| F13 three-way comparison | **DONE** `results/figures/F13_three_way_comparison.{pdf,png}` | summary.csv | E20–E22 |
| F14 score function gradient ratio | **DONE** `results/figures/F14_score_function_grad_ratio.{pdf,png}` | `runs/live_B_diag/live_B_live_sc/` | E21 |
| F15 TC vs z_dim | **DONE** `results/figures/F15_tc_vs_zdim.{pdf,png}` | `results/post_hoc/sc_posthoc_mag_zdim{1,2}/` | E37, E38 |
| F16 Phase 2 training curves | **DONE** `results/figures/F16_phase2_curves.{pdf,png}` | sc_twophase_*_v2 metrics.csv | E44–E46 |
| F17 ε_MLE vs N | **DONE** `results/figures/F17_eps_mle_convergence.{pdf,png}` | E09 checkpoint (N sweep) | E09 |
| F18 isotropic activations | deferred | — | future |
| F19 moving-target error | **DONE** `results/figures/F19_moving_target_error.{pdf,png}` | `runs/live_B_diag/live_B_live_sc/` | E21 |
| F20 three-way Pareto at SR=1 | **DONE** `results/figures/F20_three_way_pareto.{pdf,png}` | summary.csv | E20–E22 |
| F21 Phase 2 ablations | **DONE** `results/figures/F21_phase2_ablations.{pdf,png}` | `results/post_hoc/sc_twophase_dither*_v2/`, `nsd_*/` | E44–E52 |

---

## Workflow: adding a new experiment

1. Add a row to `docs/EXPERIMENT_REGISTRY.md` (status: PLANNED).
2. Add config to `experiments/run_sc_experiments.py` or write a standalone runner.
3. Run via `scripts/run_p2_experiments.sh` or manually.
4. Update status to DONE in registry.
5. Run `python -m onpolicy.envs.toyproblem.analysis.aggregate_results --verbose`.
6. If checkpoint is used by post_hoc: run `post_hoc_coding.py --device cpu`; it writes
   directly to `results/post_hoc/<exp_name>/seed_<seed>.json`. Then re-run `aggregate_results.py`.
7. Add a dated entry to `docs/JOURNAL.md`.
8. Update any affected conclusion in `docs/CONCLUSIONS.md`.

## Workflow: generating a figure

```bash
# message_geometry (z-space / confusion)
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.analysis.message_geometry \
    --checkpoints <ckpt1> [<ckpt2> ...] --labels "..." \
    --out results/figures/message_geometry.png

# paper figures (F1–F20)
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.analysis.paper_figures \
    --runs_dir runs/sc_ablation --out_dir results/figures

# rebuild aggregated summary after any change
python -m onpolicy.envs.toyproblem.analysis.aggregate_results --verbose
```

## Run environment

- Conda env: `marl_comms`
- Always prefix with `KMP_DUPLICATE_LIB_OK=TRUE` on macOS
- Use `--device cpu` for post_hoc_coding.py (MPS has buffer issues with small batches)
- Training logs: `runs/sc_ablation/<exp>/<seed>/train.log` (if redirected by shell script)
