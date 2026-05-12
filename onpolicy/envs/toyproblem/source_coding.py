"""
source_coding.py — Online Histogram + Score Function rate loss (Option 1 / §15).

Background (see docs/pillars/PILLAR_P2.md §15):
  DDCL quantises the speaker output z ∈ ℝ^d to m = floor(z/δ) ∈ ℤ^d.
  The channel transmits m; the actual bit cost is H(m), the entropy of the
  discrete message distribution.  The goal of P2 is to minimise H(m) subject
  to task success.

  The DLM prior approach (original P2) failed because the parametric model
  could not track the non-stationary distribution of m — the qphi_gap never
  fell below ~12 bits, far above the 2-bit gate threshold.

This module replaces the DLM prior with two simpler components:

  1. MessageHistogram  — an online per-dimension frequency table.
       q_hist(m_k) = (count_k[m_k] + α) / total_k      (Laplace-smoothed)
       R(m_k) = -log₂ q_hist(m_k)                       (bits; detached)

     The histogram tracks any distribution by construction — fitting failure
     is impossible.  α = 0.5 (Jeffreys prior) is the default.

  2. source_coding_rate_loss  — score function proxy loss.
       ∂/∂z_k E[R_k(m_k)] = [R_k(m_hi,k) - R_k(m_lo,k)] / δ
     (derived in §15.3.4; m_hi = floor(z_k/δ)+1, m_lo = floor(z_k/δ))

     Proxy loss that realises this gradient:
       L_rate = λ · Σ_k [(R_hi_k − R_lo_k).detach() / δ · z_k]

     The detach means no gradient flows through the histogram values;
     the only gradient path is through z_k itself, which is correct.

EM analogy (§15.3.3):
  • E-step: update histogram from current rollout before each PPO epoch.
  • M-step: minimise L_rate w.r.t. speaker parameters with histogram frozen.

Key properties:
  P1  Unbiased gradient  — expectation of score function = true rate gradient.
  P2  No domain gap      — histogram evaluated only at integer m, never at z/δ.
  P3  No fitting failure — empirical counts converge to true p(m) as N→∞.
  P4  Compression target — gradient pushes z_k toward cheaper adjacent bin,
                           not toward the current distribution's mode.
  P5  Schuchman compatible — histogram is irrelevant to the DDCL STE; only
                              the gradient of L_rate w.r.t. z matters.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch


class MessageHistogram:
    """Online per-dimension frequency table for discrete message integers.

    One count dict per dimension k.  Laplace (add-α) smoothing prevents
    zero-probability bins — important for stability when the speaker explores
    rare messages early in training.

    Update semantics:
        Call reset() once per rollout (before the PPO epoch loop).
        Call update(m) on each minibatch of integer messages.
        rate(m) can then be queried inside the PPO epoch loop.

    Thread safety: not thread-safe.  Single-process training only.
    """

    def __init__(self, z_dim: int, smoothing: float = 0.5) -> None:
        """
        Args:
            z_dim:     number of message dimensions (= speaker z_dim)
            smoothing: Laplace smoothing parameter α (Jeffreys: 0.5)
        """
        self.z_dim = z_dim
        self.smoothing = smoothing
        # _counts[k]: maps integer bin → raw count (NOT including smoothing).
        # Smoothing is added at query time so that the total stays consistent.
        self._counts: list[dict[int, float]] = [{} for _ in range(z_dim)]
        # _totals[k]: sum of raw counts (N observations, not smoothed).
        self._totals: list[float] = [0.0] * z_dim
        # _n_bins[k]: number of distinct bins seen so far.
        self._n_bins: list[int] = [0] * z_dim

    def reset(self) -> None:
        """Clear all accumulated counts (call once per rollout)."""
        self._counts = [{} for _ in range(self.z_dim)]
        self._totals = [0.0] * self.z_dim
        self._n_bins = [0] * self.z_dim

    def update(self, m: torch.Tensor) -> None:
        """Accumulate integer message counts from a batch.

        Args:
            m: (..., z_dim) integer tensor of quantised bins.
               Any leading batch dimensions are flattened internally.
        """
        m_flat = m.detach().reshape(-1, self.z_dim).cpu()
        N = m_flat.shape[0]
        for k in range(self.z_dim):
            col = m_flat[:, k].tolist()
            counts_k = self._counts[k]
            for val in col:
                v = int(val)
                if v not in counts_k:
                    counts_k[v] = 0.0
                    self._n_bins[k] += 1
                counts_k[v] += 1.0
            self._totals[k] += float(N)

    def rate(self, m: torch.Tensor) -> torch.Tensor:
        """Compute per-element rate R(m_k) = -log₂ q_hist(m_k) in bits.

        Uses Laplace-smoothed probabilities:
            q_hist(m_k) = (count_k[m_k] + α) / (N_k + α · V_k)
        where V_k = n_bins[k] (number of distinct bins seen).

        If no data has been accumulated yet (total = 0), returns zero for all
        elements — the score function gradient will be zero, which is benign.

        The output tensor is on the same device as m and has no gradient.

        Args:
            m: (..., z_dim) integer tensor
        Returns:
            (..., z_dim) float32 tensor of rates in bits, detached
        """
        orig_shape = m.shape
        m_flat = m.detach().reshape(-1, self.z_dim).cpu()
        N = m_flat.shape[0]
        out = torch.zeros(N, self.z_dim, dtype=torch.float32)

        for k in range(self.z_dim):
            total_k = self._totals[k]
            if total_k <= 0.0:
                continue  # no data: rate = 0 (gradient = 0, harmless)

            alpha = self.smoothing
            n_bins_k = max(self._n_bins[k], 1)
            denom = total_k + alpha * n_bins_k
            counts_k = self._counts[k]

            for i in range(N):
                v = int(m_flat[i, k].item())
                raw = counts_k.get(v, 0.0)   # unseen bins have raw count 0
                prob = (raw + alpha) / denom
                out[i, k] = -math.log2(prob) if prob > 0.0 else float("inf")

        return out.reshape(orig_shape[:-1] + (self.z_dim,)).to(m.device)

    def empirical_entropy(self) -> list[float]:
        """Per-dimension empirical entropy H(m_k) in bits, from smoothed counts.

        Returns: list of z_dim floats.  Sum = total bits per message vector.
        """
        result = []
        for k in range(self.z_dim):
            total_k = self._totals[k]
            if total_k <= 0.0:
                result.append(0.0)
                continue
            alpha = self.smoothing
            n_bins_k = max(self._n_bins[k], 1)
            denom = total_k + alpha * n_bins_k
            h = 0.0
            for raw in self._counts[k].values():
                prob = (raw + alpha) / denom
                if prob > 0.0:
                    h -= prob * math.log2(prob)
            result.append(h)
        return result

    def n_distinct_bins(self) -> list[int]:
        """Number of distinct message bins seen per dimension."""
        return list(self._n_bins)

    def total_counts(self) -> list[float]:
        """Total observation count per dimension."""
        return list(self._totals)


class JointMessageHistogram:
    """D-dimensional joint frequency table for post-hoc source coding analysis.

    Stores counts for full message tuples (m_0, ..., m_{D-1}) ∈ ℤ^D.
    Used to compute H_joint(m) and TC(m) = Σ_k H(m_k) - H_joint(m).

    Scalability: K^D entries where K = observed bins per dim.  Practical for
    D ≤ 3 with K ≈ 6–8 (max ~512 entries).  For larger D use a NN estimator.

    Not intended for use during training — post-hoc analysis only.
    """

    def __init__(self, z_dim: int, smoothing: float = 0.5) -> None:
        self.z_dim = z_dim
        self.smoothing = smoothing
        self._counts: dict[tuple[int, ...], float] = {}
        self._total: float = 0.0
        # Per-dimension marginal counts (mirrors MessageHistogram logic)
        self._marginals: list[dict[int, float]] = [{} for _ in range(z_dim)]
        self._marginal_totals: list[float] = [0.0] * z_dim

    def reset(self) -> None:
        self._counts = {}
        self._total = 0.0
        self._marginals = [{} for _ in range(self.z_dim)]
        self._marginal_totals = [0.0] * self.z_dim

    def update(self, m: torch.Tensor) -> None:
        """Accumulate joint counts from a message batch.

        Args:
            m: (..., z_dim) integer tensor
        """
        m_flat = m.detach().reshape(-1, self.z_dim).cpu().tolist()
        N = len(m_flat)
        for row in m_flat:
            key = tuple(int(v) for v in row)
            self._counts[key] = self._counts.get(key, 0.0) + 1.0
            for k, v in enumerate(key):
                d = self._marginals[k]
                d[v] = d.get(v, 0.0) + 1.0
        self._total += float(N)
        for k in range(self.z_dim):
            self._marginal_totals[k] += float(N)

    def joint_entropy(self) -> float:
        """H_joint(m) in bits, Laplace-smoothed."""
        if self._total <= 0.0:
            return 0.0
        alpha = self.smoothing
        V = max(len(self._counts), 1)
        denom = self._total + alpha * V
        h = 0.0
        for raw in self._counts.values():
            p = (raw + alpha) / denom
            if p > 0.0:
                h -= p * math.log2(p)
        return h

    def marginal_entropies(self) -> list[float]:
        """H(m_k) per dimension in bits, Laplace-smoothed (matches MessageHistogram)."""
        result = []
        for k in range(self.z_dim):
            total_k = self._marginal_totals[k]
            if total_k <= 0.0:
                result.append(0.0)
                continue
            alpha = self.smoothing
            n_bins = max(len(self._marginals[k]), 1)
            denom = total_k + alpha * n_bins
            h = 0.0
            for raw in self._marginals[k].values():
                p = (raw + alpha) / denom
                if p > 0.0:
                    h -= p * math.log2(p)
            result.append(h)
        return result

    def total_correlation(self) -> float:
        """TC(m) = Σ_k H(m_k) - H_joint(m) ≥ 0 in bits."""
        return sum(self.marginal_entropies()) - self.joint_entropy()

    def n_distinct_tuples(self) -> int:
        return len(self._counts)


def source_coding_rate_loss(
    z: torch.Tensor,
    histogram: MessageHistogram,
    delta: float | torch.Tensor,
    lambda_comms: float,
) -> torch.Tensor:
    """Score function proxy loss for the expected histogram rate E[-log₂ q_hist(m)].

    Derivation (PILLAR_P2_v2.md §3 / score function gradient):
    For the SD channel m = floor((z + ε)/δ), ε ~ U(-δ/2, δ/2):

      Let n = floor(z/δ),  f = frac(z/δ) = z/δ − n ∈ [0, 1)

      f ∈ [0, 0.5):  P(m=n−1) = 0.5−f,  P(m=n) = 0.5+f   → bins {n−1, n}
      f ∈ [0.5, 1):  P(m=n)   = 1.5−f,  P(m=n+1) = f−0.5  → bins {n, n+1}

      In both cases the score function gradient is:
        ∂/∂z_k E[R_k(m_k)] = [R_k(m_hi,k) - R_k(m_lo,k)] / δ
      where {m_lo, m_hi} are the two reachable bins for the current frac.

    Proxy loss (scale trick):
      L_rate = λ · Σ_k [(R_hi_k - R_lo_k).detach() / δ · z_k]

    ∂L_rate/∂z_k = λ · (R_hi_k - R_lo_k) / δ  ✓ matches the score function.

    Note on gradient direction: R is monotone in |m| for a well-trained histogram,
    so R_hi > R_lo always, and the gradient pushes z toward lower-magnitude bins
    regardless of which frac half we are in. The bin selection (below) ensures the
    magnitude is also correct — previously m_hi was always m_lo+1, which was wrong
    for f < 0.5 (should be m_lo−1 and m_base instead of m_base and m_base+1).

    Args:
        z:             (B, z_dim) float tensor, requires_grad should be True
        histogram:     MessageHistogram populated from the current rollout
        delta:         quantisation bin width δ > 0
        lambda_comms:  communication loss weight λ ≥ 0

    Returns:
        scalar loss tensor; backprop through it gives the correct rate gradient
        w.r.t. z.  Returns zero tensor if histogram has no data.
    """
    with torch.no_grad():
        frac = (z / delta) - torch.floor(z / delta)   # ∈ [0, 1)
        m_base = torch.floor(z / delta).long()
        # Select the correct bin pair depending on which half of [0,1) frac is in.
        # f < 0.5: reachable bins are {m_base−1, m_base}
        # f ≥ 0.5: reachable bins are {m_base, m_base+1}
        low_frac = frac < 0.5
        m_lo = torch.where(low_frac, m_base - 1, m_base)
        m_hi = torch.where(low_frac, m_base,     m_base + 1)

    # histogram.rate() returns a detached tensor by construction.
    R_lo = histogram.rate(m_lo)   # (B, z_dim)
    R_hi = histogram.rate(m_hi)   # (B, z_dim)

    # grad_scale has no autograd path (both R tensors are detached).
    # The only gradient path is through z below.
    grad_scale = (R_hi - R_lo) / delta   # (B, z_dim)
    return lambda_comms * (grad_scale * z).sum(dim=-1).mean()


def dither_channel_loss(
    z: torch.Tensor,
    delta: float | torch.Tensor,
    lambda_dither: float,
    frac_eps: float = 1e-4,
) -> torch.Tensor:
    """Phase 2 loss that reduces H(m|goal) by pushing frac(z/δ) toward 0.5.

    Derivation of the correct formula (floor quantiser with Uniform dither):
    -----------------------------------------------------------------------
    The SD channel uses m = floor((z + ε)/δ) with ε ~ U(-δ/2, δ/2).
    For z_k = n·δ + f_k·δ  (n integer, f_k = frac(z_k/δ) ∈ [0,1)):

      For f_k ∈ [0, 0.5]:
        P(m = n-1) = 0.5 - f_k        P(m = n) = 0.5 + f_k
        H(m_k | z_k) = H_binary(0.5 - f_k)

      For f_k ∈ [0.5, 1):
        P(m = n)   = 1.5 - f_k        P(m = n+1) = f_k - 0.5
        H(m_k | z_k) = H_binary(f_k - 0.5)

    In both cases: H(m_k | z_k) = H_binary(|f_k - 0.5|)

    Key consequence:
      • f_k = 0.5  → H = 0 bits  (z at bin centre; dither never crosses boundary)
      • f_k = 0    → H = 1 bit   (z at bin boundary; dither always ambiguous)

    The loss therefore pushes f_k → 0.5, i.e. z_k toward half-integer multiples
    of δ (bin centres).  This is the MINIMUM-noise locus for the floor quantiser.

    This is compatible with the magnitude anchor: their joint minimum is near
    z_k ≈ δ/2, where m_k = 0 deterministically and |z_k| is small.

    Gradient derivation:
        g_k  = |f_k - 0.5|                        ∈ [0, 0.5)
        dg/dz_k = sign(f_k - 0.5) / δ
        dH_binary(g_k)/dg_k = log₂((1-g_k)/g_k)
        ∂H/∂z_k = log₂((1-g_k)/g_k) · sign(f_k - 0.5) / δ

    Proxy loss (scale trick so autograd gives the correct gradient):
        L_dither = λ · Σ_k [grad_scale_k.detach() · z_k]
        where grad_scale_k = log₂((1-g_k)/g_k) · sign(f_k - 0.5) / δ

    Singularity at g_k = 0 (f_k = 0.5): clamp g_k ≥ frac_eps.

    Args:
        z:             (B, z_dim) float tensor, requires_grad should be True
        delta:         quantisation bin width δ > 0
        lambda_dither: loss weight λ ≥ 0
        frac_eps:      clamp |f - 0.5| to [frac_eps, 0.5] for stability

    Returns:
        scalar loss tensor; backprop gives ∂H_binary(|frac-0.5|)/∂z per element.
    """
    with torch.no_grad():
        frac = (z / delta) - torch.floor(z / delta)   # (B, z_dim), ∈ [0, 1)
        g = (frac - 0.5).abs()                         # (B, z_dim), ∈ [0, 0.5)
        g = g.clamp(frac_eps, 0.5 - frac_eps)          # avoid singularity at g=0
        # gradient: log2((1-g)/g) * sign(frac - 0.5) / delta
        grad_scale = torch.log2((1.0 - g) / g) * torch.sign(frac - 0.5) / delta

    return lambda_dither * (grad_scale * z).sum(dim=-1).mean()


def dither_channel_stats(z: torch.Tensor, delta: float | torch.Tensor) -> dict[str, float]:
    """Compute H(m|goal) — per-element dither noise entropy for the floor SD channel.

    Correct formula for m = floor((z + ε)/δ), ε ~ U(-δ/2, δ/2):

        H(m_k | z_k) = H_binary(|frac(z_k/δ) - 0.5|)

    where H_binary is minimised (0 bits) at frac = 0.5 and maximised (1 bit)
    at frac = 0 or 1.  See dither_channel_loss docstring for derivation.

    Args:
        z:     (B, z_dim) float tensor
        delta: quantisation bin width δ > 0

    Returns dict with:
        H_dither_channel  — mean Σ_k H_binary(|frac(z_k/δ) - 0.5|) over batch (bits)
        mean_frac         — mean fractional part across all elements
        mean_g            — mean |frac - 0.5| (0 = at bin centre, 0.5 = at boundary)
    """
    with torch.no_grad():
        frac = (z / delta) - torch.floor(z / delta)   # ∈ [0, 1)
        g = (frac - 0.5).abs().clamp(1e-7, 0.5 - 1e-7)  # ∈ (0, 0.5)
        h_bin = -g * torch.log2(g) - (1.0 - g) * torch.log2(1.0 - g)
        return {
            "H_dither_channel": float(h_bin.sum(dim=-1).mean().item()),
            "mean_frac": float(frac.mean().item()),
            "mean_g": float(g.mean().item()),
        }


def histogram_rate_stats(
    m: torch.Tensor,
    histogram: MessageHistogram,
) -> dict[str, float]:
    """Diagnostic metrics for the current histogram against a message batch.

    Useful for monitoring rate quality and histogram convergence.

    Args:
        m:         (N, z_dim) integer tensor of message bins
        histogram: MessageHistogram populated this rollout

    Returns dict with:
        hist_entropy_rate  — E[-log₂ q_hist(m)] over batch m (bits/msg-vector)
        hist_H_empirical   — H(m) from histogram counts (bits/msg-vector)
        hist_qphi_gap      — hist_entropy_rate - hist_H_empirical
                             Should be ≈ 0 for a well-fitted histogram
                             (unlike DLM which had gaps of 12-14 bits)
    """
    with torch.no_grad():
        R = histogram.rate(m)
        hist_entropy_rate = float(R.sum(dim=-1).mean().item())
        hist_H_empirical = float(sum(histogram.empirical_entropy()))
        hist_qphi_gap = hist_entropy_rate - hist_H_empirical

    return {
        "hist_entropy_rate": hist_entropy_rate,
        "hist_H_empirical": hist_H_empirical,
        "hist_qphi_gap": hist_qphi_gap,
    }
