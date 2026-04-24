"""Analysis utilities for DDCL toy-problem experiments.

Modules
-------
load_runs       : load raw CSVs into tidy DataFrames
stats           : bootstrap CI, permutation test, IQM, Pareto frontier
plots           : training curves, rate-distortion frontier, per-goal bits
sweep_convergence : automated 4-criterion convergence gate (Phase 2 §2.3)
report_baseline : generate results/toyproblem/<sweep>/baseline.md
"""
