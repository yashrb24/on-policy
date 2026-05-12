"""aggregate_results.py — Rebuild results/aggregated/summary.csv and post_hoc aggregate.

Run this script after any new training run completes or after running post_hoc_coding.py
on a new checkpoint. It reads all metrics.csv files under runs/sc_ablation/ and all
post_hoc_decomposition.json files, then writes:

    results/aggregated/summary.csv        — mean±std across seeds for every experiment
    results/post_hoc/<exp>/aggregate.json — mean±std of decomposition components

Usage (from repo root):
    python -m onpolicy.envs.toyproblem.analysis.aggregate_results

Flags:
    --runs_dir     root of training outputs (default: runs/sc_ablation)
    --results_dir  root of processed outputs (default: results)
    --verbose      print a summary table to stdout
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _mean_std(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


# ── Training run aggregation ──────────────────────────────────────────────────

_METRICS = [
    "success_rate",
    "true_bits_per_msg",
    "hist_H_empirical",
    "hist_qphi_gap",
    "dither_loss",
    "H_dither_channel",
    "mean_frac",
    "policy_loss",
    "value_loss",
    "entropy_loss",
]


def aggregate_training_runs(runs_dir: Path) -> list[dict]:
    """Read every metrics.csv, return list of dicts (one per experiment)."""
    records = []

    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("stage_"):
            continue

        seed_rows: list[dict] = []
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            csv_path = seed_dir / "metrics.csv"
            if not csv_path.exists():
                continue
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                seed_rows.append(rows[-1])  # final row = convergence metrics

        if not seed_rows:
            continue

        n_seeds = len(seed_rows)
        row: dict = {"experiment": exp_dir.name, "n_seeds": n_seeds}

        for metric in _METRICS:
            vals = [_safe_float(r.get(metric)) for r in seed_rows]
            m, s = _mean_std([v for v in vals if v is not None])
            row[f"{metric}_mean"] = round(m, 4) if not math.isnan(m) else None
            row[f"{metric}_std"] = round(s, 4) if not math.isnan(s) else None

        # Total updates logged
        updates = [_safe_float(r.get("update")) for r in seed_rows]
        row["max_update_mean"] = round(statistics.mean([u for u in updates if u]), 1)

        records.append(row)

    return records


def write_training_summary(records: list[dict], out_path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"[aggregate] training summary → {out_path}  ({len(records)} experiments)")


# ── Post-hoc decomposition aggregation ───────────────────────────────────────

_DECOMP_KEYS = [
    "H_G_bits", "H_joint_bits", "H_factored_bits", "TC_bits",
    "H_dither_bits", "eps_estimator_bits", "eps_MLE_bound",
    "gap_factored_bits", "gap_joint_bits",
    "decomp_sum_bits", "decomp_residual_bits",
    "K_obs_joint_tuples", "N_messages",
]


def aggregate_post_hoc(post_hoc_dir: Path) -> None:
    """For each <exp> subfolder containing seed_N.json files, write aggregate.json."""
    if not post_hoc_dir.exists():
        return

    for exp_dir in sorted(post_hoc_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        seed_files = sorted(exp_dir.glob("seed_*.json"))
        if not seed_files:
            continue

        all_data: list[dict] = []
        for sf in seed_files:
            with open(sf) as f:
                all_data.append(json.load(f))

        aggregate: dict = {"n_seeds": len(all_data), "experiment": exp_dir.name}
        for key in _DECOMP_KEYS:
            vals = [_safe_float(d.get(key)) for d in all_data]
            vals = [v for v in vals if v is not None]
            if vals:
                aggregate[f"{key}_mean"] = round(statistics.mean(vals), 4)
                aggregate[f"{key}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
            else:
                aggregate[f"{key}_mean"] = None
                aggregate[f"{key}_std"] = None

        # Per-dim entropies
        z_dim = int(all_data[0].get("z_dim", 0))
        for k in range(z_dim):
            key = f"H_dim_{k}_bits"
            vals = [_safe_float(d.get(key)) for d in all_data]
            vals = [v for v in vals if v is not None]
            if vals:
                aggregate[f"{key}_mean"] = round(statistics.mean(vals), 4)
                aggregate[f"{key}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0

        out_path = exp_dir / "aggregate.json"
        with open(out_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"[aggregate] post_hoc {exp_dir.name} ({len(all_data)} seeds) → {out_path}")


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_summary_table(records: list[dict]) -> None:
    print(f"\n{'Experiment':<35} {'SR':<14} {'true_bits':<14} {'hist_H':<14} {'seeds'}")
    print("─" * 85)
    for r in records:
        sr = r.get("success_rate_mean")
        sr_s = r.get("success_rate_std")
        tb = r.get("true_bits_per_msg_mean")
        tb_s = r.get("true_bits_per_msg_std")
        hh = r.get("hist_H_empirical_mean")
        hh_s = r.get("hist_H_empirical_std")

        def fmt(m, s):
            if m is None:
                return "N/A"
            return f"{m:.3f}±{s:.3f}" if s else f"{m:.3f}"

        print(f"{r['experiment']:<35} {fmt(sr, sr_s):<14} {fmt(tb, tb_s):<14} {fmt(hh, hh_s):<14} {r['n_seeds']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate training and post-hoc results.")
    p.add_argument("--runs_dir", type=Path, default=Path("runs/sc_ablation"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Training summary
    records = aggregate_training_runs(args.runs_dir)
    write_training_summary(records, args.results_dir / "aggregated" / "summary.csv")

    # Post-hoc decomposition
    aggregate_post_hoc(args.results_dir / "post_hoc")

    if args.verbose:
        print_summary_table(records)

    print("\n[aggregate] done.")


if __name__ == "__main__":
    main()
