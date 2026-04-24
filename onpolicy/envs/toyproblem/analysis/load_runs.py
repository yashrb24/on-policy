"""Load experiment run outputs into tidy DataFrames.

Typical structure written by train.py:

    runs/<exp_name>/<seed>/
        config.json       -- all CLI args
        metrics.csv       -- one row per update
        metrics.parquet   -- same (if pyarrow installed)
        git_sha.txt
        final.pt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-run loader
# ---------------------------------------------------------------------------

def load_run(run_dir: str | Path) -> pd.DataFrame:
    """Load one run's metrics CSV and attach config columns.

    Returns a DataFrame with one row per logged update, plus columns for
    every config key (exp_name, seed, channel, delta, lambda_comms, …).
    """
    run_dir = Path(run_dir)

    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        # Try parquet first (faster), fall back to CSV.
        parquet_path = run_dir / "metrics.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"No metrics.csv or metrics.parquet in {run_dir}")
    else:
        df = pd.read_csv(csv_path)

    # Attach config metadata as columns.
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        for k, v in config.items():
            df[k] = v  # broadcast scalar to all rows

    # Record the path for debugging.
    df["run_dir"] = str(run_dir)
    return df


# ---------------------------------------------------------------------------
# Sweep / experiment loaders
# ---------------------------------------------------------------------------

def load_exp(exp_dir: str | Path) -> pd.DataFrame:
    """Load all seeds of one experiment (<exp_dir>/<seed>/metrics.csv).

    exp_dir is the directory containing numbered seed sub-directories, e.g.
    ``runs/baseline_v0/`` which contains ``0/``, ``1/``, ``2/``, …
    """
    exp_dir = Path(exp_dir)
    frames = []
    for seed_dir in sorted(exp_dir.iterdir()):
        if seed_dir.is_dir() and (seed_dir / "metrics.csv").exists():
            frames.append(load_run(seed_dir))
    if not frames:
        raise FileNotFoundError(f"No seed runs found under {exp_dir}")
    return pd.concat(frames, ignore_index=True)


def load_sweep(sweep_dir: str | Path, pattern: str = "*") -> pd.DataFrame:
    """Load all experiments matching *pattern* under sweep_dir.

    sweep_dir : root directory (e.g. ``runs/sweep_stage_a/``)
    pattern   : glob pattern applied to immediate children of sweep_dir
                (e.g. ``"sd_*"`` for all SD runs)

    Returns one tidy DataFrame with all runs concatenated.
    """
    sweep_dir = Path(sweep_dir)
    frames = []
    for exp_dir in sorted(sweep_dir.glob(pattern)):
        if exp_dir.is_dir():
            try:
                frames.append(load_exp(exp_dir))
            except FileNotFoundError:
                pass  # skip empty dirs silently
    if not frames:
        raise FileNotFoundError(
            f"No runs found under {sweep_dir} matching {pattern!r}"
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def final_metrics(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Aggregate the last *window* updates per run into per-seed summary stats.

    Groups by (exp_name, seed) and returns mean of the last `window` rows
    for key performance metrics.
    """
    key_cols = [
        "success_rate", "bits_per_msg", "true_bits_per_msg", "mean_reward",
        "pg_loss", "value_loss", "entropy", "comms_loss",
    ]
    key_cols = [c for c in key_cols if c in df.columns]

    group_cols = [c for c in ("exp_name", "seed", "channel", "lambda_comms",
                               "delta", "z_dim", "hidden_size") if c in df.columns]

    def tail_mean(grp: pd.DataFrame) -> pd.Series:
        tail = grp.sort_values("update").tail(window)
        return tail[key_cols].mean()

    return df.groupby(group_cols, as_index=False).apply(tail_mean).reset_index(drop=True)


def seed_aggregate(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Collapse per-seed summary into per-config mean ± std.

    *summary* is the output of ``final_metrics``.
    *group_cols* lists the hyperparameter columns to group by
    (everything except 'seed').

    Returns a DataFrame with columns <metric>_mean and <metric>_std for each
    numeric metric column.
    """
    numeric = summary.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in group_cols + ["seed"]]

    agg_dict = {c: ["mean", "std"] for c in numeric}
    agg = summary.groupby(group_cols).agg(agg_dict).reset_index()
    agg.columns = ["_".join(filter(None, c)) for c in agg.columns]
    return agg
