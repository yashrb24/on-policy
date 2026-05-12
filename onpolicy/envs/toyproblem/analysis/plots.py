"""Low-level plotting utilities shared by paper_figures.py.

All standalone analysis figures have been replaced by the paper-quality
functions in paper_figures.py.  This module retains only shared constants
and the matplotlib availability guard.
"""
from __future__ import annotations

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install: pip install matplotlib"
        )
