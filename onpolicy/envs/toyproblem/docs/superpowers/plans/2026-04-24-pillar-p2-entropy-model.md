# Pillar P2 — Entropy Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Context files to read before starting:**
> - `onpolicy/envs/toyproblem/docs/pillars/PILLAR_P2.md` — full design spec
> - `onpolicy/envs/toyproblem/network.py` — SpeakerNetwork, ListenerActor, Critic
> - `onpolicy/envs/toyproblem/trainer.py` — MAPPOConfig, MAPPOTrainer
> - `onpolicy/envs/toyproblem/train.py` — parse_args, main loop, CSV_HEADER
> - `onpolicy/envs/toyproblem/channels.py` — channel forward pass returns `(z_hat, info)` where `info["m"]` is the integer message
>
> **Run all tests from the repo root:**
> ```
> KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms pytest onpolicy/envs/toyproblem/tests/ -v
> ```

**Goal:** Add a learned prior `q_φ(m)` (Discretised Logistic Mixture) as the communication rate surrogate, replacing the magnitude upper bound `log₂(|z|/δ + 1)` with `E[-log₂ q_φ(m)]`.

**Architecture:** Four new classes in `network.py` (`EntropyModelFactored`, `EntropyModelJoint`, `EntropyModelCondZ`, `EntropyModelJointCondZ`) closing the full 2×2 of (factored/joint) × (context A/B), plus two helper functions for empirical entropy. `trainer.py` gains a second Adam optimizer for `q_φ`, a warm-start method, and per-minibatch forward+backward entropy losses. `train.py` gets 8 new CLI flags and extended CSV columns. `run_p2_ablation.py` implements a 6-stage systematic study (395 runs total).

**Tech Stack:** PyTorch (nn.Parameter, Adam), NumPy (histogram-based entropy estimate), existing `MAPPOTrainer`/`MAPPOConfig`/`RolloutBuffer` infrastructure.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `onpolicy/envs/toyproblem/network.py` | Add 4 entropy model classes + 3 helper functions |
| Modify | `onpolicy/envs/toyproblem/trainer.py` | Add 8 config fields, q_φ optimizer, Ballé loss, warm-start, P2 metrics |
| Modify | `onpolicy/envs/toyproblem/train.py` | Add 8 CLI flags, extend CSV header |
| Create | `onpolicy/envs/toyproblem/tests/test_entropy_model.py` | Unit tests for all 4 models, helpers, trainer integration |
| Create | `onpolicy/envs/toyproblem/experiments/run_p2_ablation.py` | 6-stage systematic ablation (395 runs) |
| Modify | `onpolicy/envs/toyproblem/analysis/paper_figures.py` | Add 4 P2 figure functions |

---

## Task 1: EntropyModelFactored — tests first

**Files:**
- Create: `onpolicy/envs/toyproblem/tests/test_entropy_model.py`
- Modify: `onpolicy/envs/toyproblem/network.py`

- [ ] **Step 1: Write failing tests**

Create `onpolicy/envs/toyproblem/tests/test_entropy_model.py`:

```python
"""Tests for Pillar P2 entropy model classes and helpers."""
from __future__ import annotations
import math
import pytest
import torch
from onpolicy.envs.toyproblem.network import EntropyModelFactored


class TestEntropyModelFactored:
    def test_output_shape_1d(self):
        model = EntropyModelFactored(z_dim=1, K=3)
        m = torch.zeros(32, 1)
        nll = model.nll_bits(m)
        assert nll.shape == (32, 1)

    def test_output_shape_3d(self):
        model = EntropyModelFactored(z_dim=3, K=5)
        m = torch.zeros(64, 3)
        nll = model.nll_bits(m)
        assert nll.shape == (64, 3)

    def test_nll_positive(self):
        """NLL in bits must be non-negative (probability ≤ 1)."""
        torch.manual_seed(0)
        model = EntropyModelFactored(z_dim=3, K=5)
        m = torch.randn(100, 3).round()  # random integers
        nll = model.nll_bits(m)
        assert (nll >= 0).all(), f"negative NLL values: {nll[nll < 0]}"

    def test_probabilities_sum_to_one(self):
        """Σ_m q(m) ≈ 1 over a reasonable integer range."""
        model = EntropyModelFactored(z_dim=1, K=3)
        ms = torch.arange(-50, 51, dtype=torch.float32).unsqueeze(-1)  # (101, 1)
        log_probs = model.log_prob(ms)  # (101, 1)
        total = log_probs.exp().sum().item()
        assert abs(total - 1.0) < 0.01, f"probabilities sum to {total:.4f}"

    def test_grad_flows_to_model_not_input(self):
        """Forward pass (m.float() detached): grad to q_φ, not to m."""
        model = EntropyModelFactored(z_dim=2, K=3)
        m = torch.tensor([[1.0, -1.0], [2.0, 0.0]], requires_grad=False)
        nll = model.nll_bits(m)
        nll.mean().backward()
        assert model.log_pi.grad is not None
        assert model.mu.grad is not None

    def test_grad_flows_to_input_when_frozen(self):
        """Frozen q_φ: grad to z/delta, not to model parameters."""
        model = EntropyModelFactored(z_dim=2, K=3)
        z_over_delta = torch.randn(8, 2, requires_grad=True)
        # freeze
        for p in model.parameters():
            p.requires_grad_(False)
        nll = model.nll_bits(z_over_delta)
        nll.mean().backward()
        assert z_over_delta.grad is not None
        for p in model.parameters():
            assert p.grad is None
        # restore
        for p in model.parameters():
            p.requires_grad_(True)

    def test_wide_init_no_inf(self):
        """Wide log_s init must not produce inf NLL for m in [-10, 10]."""
        model = EntropyModelFactored(z_dim=3, K=5)
        ms = torch.randint(-10, 11, (200, 3)).float()
        nll = model.nll_bits(ms)
        assert torch.isfinite(nll).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'EntropyModelFactored'`

- [ ] **Step 3: Implement EntropyModelFactored in network.py**

Add at the top of `onpolicy/envs/toyproblem/network.py` (after existing imports):

```python
import math
import numpy as np
```

Then add after the `Critic` class:

```python
class EntropyModelFactored(nn.Module):
    """Per-dimension Discretised Logistic Mixture prior.

    q_φ(m) = ∏_k q_φ_k(m_k)
    q_φ_k(m_k) = Σ_c π_c · [σ((m_k+0.5−μ_c)/s_c) − σ((m_k−0.5−μ_c)/s_c)]

    Works for both discrete m.float() (forward loss: trains q_φ) and continuous
    z/δ (backward loss: grads flow to speaker via Ballé relaxation).
    """

    def __init__(self, z_dim: int, K: int = 5) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.K = K
        # (z_dim, K) — uniform mixture, centred, wide scales
        self.log_pi = nn.Parameter(torch.zeros(z_dim, K))
        self.mu = nn.Parameter(torch.zeros(z_dim, K))
        self.log_s = nn.Parameter(torch.ones(z_dim, K))  # s = e ≈ 2.72 at init

    @staticmethod
    def _dlm_log_prob(
        x: torch.Tensor,       # (..., z_dim)
        log_pi: torch.Tensor,  # (z_dim, K)  or  (..., z_dim, K)
        mu: torch.Tensor,      # same shape as log_pi
        s: torch.Tensor,       # same shape as log_pi (positive)
    ) -> torch.Tensor:         # (..., z_dim)
        """DLM log-probability per dimension."""
        x_e = x.unsqueeze(-1)                                         # (..., z_dim, 1)
        upper = torch.sigmoid((x_e + 0.5 - mu) / s)                  # (..., z_dim, K)
        lower = torch.sigmoid((x_e - 0.5 - mu) / s)                  # (..., z_dim, K)
        log_pi_n = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)
        log_p_k = log_pi_n + (upper - lower).clamp(min=1e-10).log()  # (..., z_dim, K)
        return torch.logsumexp(log_p_k, dim=-1)                       # (..., z_dim)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log q_φ(x) per dimension. x: (..., z_dim)."""
        return self._dlm_log_prob(x, self.log_pi, self.mu, self.log_s.exp())

    def nll_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood in bits per element: −log₂ q_φ(x)."""
        return -self.log_prob(x) / math.log(2)
```

- [ ] **Step 4: Run tests to verify they pass**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestEntropyModelFactored -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /path/to/on-policy
git add onpolicy/envs/toyproblem/network.py \
        onpolicy/envs/toyproblem/tests/test_entropy_model.py
git commit -m "feat(p2): add EntropyModelFactored (DLM per-dimension prior)"
```

---

## Task 2: EntropyModelJoint — tests first

**Files:**
- Modify: `onpolicy/envs/toyproblem/tests/test_entropy_model.py`
- Modify: `onpolicy/envs/toyproblem/network.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/test_entropy_model.py`:

```python
from onpolicy.envs.toyproblem.network import EntropyModelJoint


class TestEntropyModelJoint:
    def test_output_shape(self):
        model = EntropyModelJoint(z_dim=3, K=5)
        m = torch.zeros(32, 3)
        nll = model.nll_bits(m)
        assert nll.shape == (32, 3)

    def test_z_dim_1_matches_factored(self):
        """Joint with z_dim=1 should behave identically to factored."""
        torch.manual_seed(42)
        factored = EntropyModelFactored(z_dim=1, K=3)
        joint = EntropyModelJoint(z_dim=1, K=3)
        # Copy weights from factored into joint dim-0 params
        with torch.no_grad():
            joint.log_pi_0.copy_(factored.log_pi[0])
            joint.mu_0.copy_(factored.mu[0])
            joint.log_s_0.copy_(factored.log_s[0])
        m = torch.tensor([[0.0], [1.0], [-1.0], [3.0]])
        assert torch.allclose(factored.nll_bits(m), joint.nll_bits(m), atol=1e-5)

    def test_nll_positive(self):
        torch.manual_seed(0)
        model = EntropyModelJoint(z_dim=3, K=5)
        m = torch.randn(100, 3).round()
        nll = model.nll_bits(m)
        assert (nll >= 0).all()

    def test_grad_flows_through_context(self):
        """Autoregressive: grad must flow through earlier dimensions."""
        model = EntropyModelJoint(z_dim=3, K=3)
        x = torch.randn(8, 3, requires_grad=True)
        nll = model.nll_bits(x)
        nll.mean().backward()
        assert x.grad is not None
        assert x.grad.shape == (8, 3)
        # All dimensions should have non-zero grad (context coupling)
        assert x.grad.abs().sum(dim=0).min().item() > 0
```

- [ ] **Step 2: Run to verify they fail**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestEntropyModelJoint -v 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'EntropyModelJoint'`

- [ ] **Step 3: Implement EntropyModelJoint in network.py**

Add after `EntropyModelFactored`:

```python
class EntropyModelJoint(nn.Module):
    """Autoregressive DLM prior.

    q_φ(m) = q_φ_0(m_0) · ∏_{k=1}^{K-1} q_φ_k(m_k | m_0,...,m_{k-1})

    Each conditional q_φ_k is a DLM whose parameters are produced by a
    small MLP taking the previous k dimensions as context.
    For z_dim=1 this reduces to EntropyModelFactored.
    """

    def __init__(self, z_dim: int, K: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.K = K
        # Dimension 0: marginal prior (no conditioning)
        self.log_pi_0 = nn.Parameter(torch.zeros(K))
        self.mu_0 = nn.Parameter(torch.zeros(K))
        self.log_s_0 = nn.Parameter(torch.ones(K))  # wide init
        # Conditional MLPs: dim k conditioned on dims 0..k-1
        self.cond_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(k, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3 * K),
            )
            for k in range(1, z_dim)
        ])

    @staticmethod
    def _dlm_log_prob_1d(
        x: torch.Tensor,       # (...)
        log_pi: torch.Tensor,  # (..., K)  or  (K,)
        mu: torch.Tensor,      # (..., K)  or  (K,)
        s: torch.Tensor,       # (..., K)  or  (K,)  — positive
    ) -> torch.Tensor:         # (...)
        x_e = x.unsqueeze(-1)
        upper = torch.sigmoid((x_e + 0.5 - mu) / s)
        lower = torch.sigmoid((x_e - 0.5 - mu) / s)
        log_pi_n = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)
        log_p_k = log_pi_n + (upper - lower).clamp(min=1e-10).log()
        return torch.logsumexp(log_p_k, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log q_φ(x) per dimension. x: (..., z_dim)."""
        lp = [self._dlm_log_prob_1d(
            x[..., 0], self.log_pi_0, self.mu_0, self.log_s_0.exp()
        )]
        for k, mlp in enumerate(self.cond_mlps, start=1):
            params = mlp(x[..., :k])           # (..., 3K)
            log_pi_k = params[..., :self.K]
            mu_k = params[..., self.K:2 * self.K]
            # bias log_s output toward wide init (add 1.0 before exp)
            s_k = (params[..., 2 * self.K:] + 1.0).exp()
            lp.append(self._dlm_log_prob_1d(x[..., k], log_pi_k, mu_k, s_k))
        return torch.stack(lp, dim=-1)          # (..., z_dim)

    def nll_bits(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x) / math.log(2)
```

- [ ] **Step 4: Run tests**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestEntropyModelJoint -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add onpolicy/envs/toyproblem/network.py \
        onpolicy/envs/toyproblem/tests/test_entropy_model.py
git commit -m "feat(p2): add EntropyModelJoint (autoregressive DLM prior)"
```

---

## Task 3: EntropyModelCondZ (Context B) — tests first

**Files:**
- Modify: `onpolicy/envs/toyproblem/tests/test_entropy_model.py`
- Modify: `onpolicy/envs/toyproblem/network.py`

- [ ] **Step 1: Append failing tests**

```python
from onpolicy.envs.toyproblem.network import EntropyModelCondZ


class TestEntropyModelCondZ:
    def test_output_shape(self):
        model = EntropyModelCondZ(z_dim=3, K=5)
        m = torch.zeros(32, 3)
        z = torch.randn(32, 3)
        nll = model.nll_bits(m, z)
        assert nll.shape == (32, 3)

    def test_nll_positive(self):
        torch.manual_seed(0)
        model = EntropyModelCondZ(z_dim=3, K=5)
        m = torch.randn(64, 3).round()
        z = torch.randn(64, 3)
        nll = model.nll_bits(m, z)
        assert (nll >= 0).all()

    def test_grad_to_z_when_params_frozen(self):
        """Frozen q_φ: grad flows through z (context) to speaker."""
        model = EntropyModelCondZ(z_dim=2, K=3)
        z = torch.randn(8, 2, requires_grad=True)
        m = torch.randn(8, 2).round().detach()
        for p in model.parameters():
            p.requires_grad_(False)
        nll = model.nll_bits(m, z)
        nll.mean().backward()
        assert z.grad is not None
        for p in model.parameters():
            p.requires_grad_(True)

    def test_different_z_different_output(self):
        """Conditioning: different z values should give different NLL."""
        torch.manual_seed(0)
        model = EntropyModelCondZ(z_dim=2, K=3)
        m = torch.zeros(8, 2)
        z1 = torch.randn(8, 2)
        z2 = torch.randn(8, 2)
        assert not torch.allclose(model.nll_bits(m, z1), model.nll_bits(m, z2))
```

- [ ] **Step 2: Run to verify they fail**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestEntropyModelCondZ -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'EntropyModelCondZ'`

- [ ] **Step 3: Implement EntropyModelCondZ in network.py**

Add after `EntropyModelJoint`:

```python
class EntropyModelCondZ(nn.Module):
    """Context-B DLM: q_φ(m | z). Per-dimension MLP(z) → DLM params.

    Unrealistic at deployment (receiver does not observe z), but useful as an
    oracle upper bound on rate reduction achievable with z-side information.
    """

    def __init__(self, z_dim: int, K: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.K = K
        # One MLP per output dimension: full z → DLM params for that dimension
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3 * K),
            )
            for _ in range(z_dim)
        ])

    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Log q_φ(x|z) per dimension. x, z: (..., z_dim)."""
        lp = []
        for k, mlp in enumerate(self.mlps):
            params = mlp(z)                           # (..., 3K)
            log_pi_k = params[..., :self.K]
            mu_k = params[..., self.K:2 * self.K]
            s_k = (params[..., 2 * self.K:] + 1.0).exp()  # wide init bias
            x_e = x[..., k].unsqueeze(-1)             # (..., 1)
            upper = torch.sigmoid((x_e + 0.5 - mu_k) / s_k)
            lower = torch.sigmoid((x_e - 0.5 - mu_k) / s_k)
            log_pi_n = log_pi_k - torch.logsumexp(log_pi_k, dim=-1, keepdim=True)
            log_p_k = log_pi_n + (upper - lower).clamp(min=1e-10).log()
            lp.append(torch.logsumexp(log_p_k, dim=-1))  # (...)
        return torch.stack(lp, dim=-1)                # (..., z_dim)

    def nll_bits(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, z) / math.log(2)
```

- [ ] **Step 4: Run tests**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestEntropyModelCondZ -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add onpolicy/envs/toyproblem/network.py \
        onpolicy/envs/toyproblem/tests/test_entropy_model.py
git commit -m "feat(p2): add EntropyModelCondZ (Context-B conditioned on z)"
```

---

## Task 3b: EntropyModelJointCondZ (Context B + joint) — closes the 2×2 grid

**Files:**
- Modify: `onpolicy/envs/toyproblem/tests/test_entropy_model.py`
- Modify: `onpolicy/envs/toyproblem/network.py`

This is the 4th cell of the (factored/joint) × (context A/B) grid:
`q_φ(m | z) = q_0(m_0 | z) · q_1(m_1 | m_0, z) · q_2(m_2 | m_0, m_1, z)`

Each conditional MLP takes `[m_{<k}, z]` as input, adding z as an extra context vector alongside the autoregressive prefix.

- [ ] **Step 1: Append failing tests**

```python
from onpolicy.envs.toyproblem.network import EntropyModelJointCondZ


class TestEntropyModelJointCondZ:
    def test_output_shape(self):
        model = EntropyModelJointCondZ(z_dim=3, K=5)
        m = torch.zeros(32, 3)
        z = torch.randn(32, 3)
        nll = model.nll_bits(m, z)
        assert nll.shape == (32, 3)

    def test_nll_positive(self):
        torch.manual_seed(0)
        model = EntropyModelJointCondZ(z_dim=3, K=5)
        m = torch.randn(64, 3).round()
        z = torch.randn(64, 3)
        nll = model.nll_bits(m, z)
        assert (nll >= 0).all()

    def test_z_dim_1_matches_condz(self):
        """Joint+CondZ with z_dim=1 reduces to CondZ (no autoregressive prefix)."""
        torch.manual_seed(42)
        cond_z = EntropyModelCondZ(z_dim=1, K=3)
        joint_cond_z = EntropyModelJointCondZ(z_dim=1, K=3)
        # Copy dim-0 MLP weights from cond_z into joint_cond_z.mlp_0
        with torch.no_grad():
            for p_src, p_dst in zip(cond_z.mlps[0].parameters(),
                                    joint_cond_z.mlp_0.parameters()):
                p_dst.copy_(p_src)
        m = torch.tensor([[0.0], [1.0], [-1.0]])
        z = torch.randn(3, 1)
        assert torch.allclose(cond_z.nll_bits(m, z), joint_cond_z.nll_bits(m, z), atol=1e-5)

    def test_different_z_different_output(self):
        """Conditioning on z must change output."""
        torch.manual_seed(0)
        model = EntropyModelJointCondZ(z_dim=2, K=3)
        m = torch.zeros(8, 2)
        z1 = torch.randn(8, 2)
        z2 = torch.randn(8, 2)
        assert not torch.allclose(model.nll_bits(m, z1), model.nll_bits(m, z2))

    def test_grad_flows_to_z_and_m_context(self):
        """Frozen q_φ: grad flows to both z and m (autoregressive context)."""
        model = EntropyModelJointCondZ(z_dim=3, K=3)
        z = torch.randn(8, 3, requires_grad=True)
        x = torch.randn(8, 3, requires_grad=True)
        for p in model.parameters():
            p.requires_grad_(False)
        nll = model.nll_bits(x, z)
        nll.mean().backward()
        assert z.grad is not None
        assert x.grad is not None
        for p in model.parameters():
            p.requires_grad_(True)
```

- [ ] **Step 2: Run to verify they fail**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestEntropyModelJointCondZ -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'EntropyModelJointCondZ'`

- [ ] **Step 3: Implement EntropyModelJointCondZ in network.py**

Add after `EntropyModelCondZ`:

```python
class EntropyModelJointCondZ(nn.Module):
    """Context-B autoregressive DLM: q_φ(m | z).

    q(m|z) = q_0(m_0|z) · ∏_{k≥1} q_k(m_k | m_0,...,m_{k-1}, z)

    Closes the 2×2 of (factored/joint) × (context A/B). Each conditional MLP
    takes [m_{<k}, z] as context, so z informs every conditional directly.
    For z_dim=1 this reduces to EntropyModelCondZ.
    """

    def __init__(self, z_dim: int, K: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.K = K
        # Dim 0: conditioned on z only  (input size = z_dim)
        self.mlp_0 = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * K),
        )
        # Dims 1..z_dim-1: conditioned on [m_{<k}, z]  (input size = k + z_dim)
        self.cond_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(k + z_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3 * K),
            )
            for k in range(1, z_dim)
        ])

    @staticmethod
    def _eval_dlm_1d(
        x_k: torch.Tensor,     # (...)
        params: torch.Tensor,  # (..., 3K)
        K: int,
    ) -> torch.Tensor:         # (...)
        log_pi = params[..., :K]
        mu = params[..., K:2 * K]
        s = (params[..., 2 * K:] + 1.0).exp()  # wide init bias
        x_e = x_k.unsqueeze(-1)                 # (..., 1)
        upper = torch.sigmoid((x_e + 0.5 - mu) / s)
        lower = torch.sigmoid((x_e - 0.5 - mu) / s)
        log_pi_n = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)
        return torch.logsumexp(
            log_pi_n + (upper - lower).clamp(min=1e-10).log(), dim=-1
        )

    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Log q_φ(x|z) per dimension. x, z: (..., z_dim)."""
        lp = [self._eval_dlm_1d(x[..., 0], self.mlp_0(z), self.K)]
        for k, mlp in enumerate(self.cond_mlps, start=1):
            context = torch.cat([x[..., :k], z], dim=-1)  # (..., k + z_dim)
            lp.append(self._eval_dlm_1d(x[..., k], mlp(context), self.K))
        return torch.stack(lp, dim=-1)  # (..., z_dim)

    def nll_bits(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, z) / math.log(2)
```

- [ ] **Step 4: Run tests**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestEntropyModelJointCondZ -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add onpolicy/envs/toyproblem/network.py \
        onpolicy/envs/toyproblem/tests/test_entropy_model.py
git commit -m "feat(p2): add EntropyModelJointCondZ (Context-B autoregressive, closes 2x2 grid)"
```

---

## Task 4: Empirical entropy helpers — tests first

**Files:**
- Modify: `onpolicy/envs/toyproblem/tests/test_entropy_model.py`
- Modify: `onpolicy/envs/toyproblem/network.py`

- [ ] **Step 1: Append failing tests**

```python
from onpolicy.envs.toyproblem.network import joint_entropy_bits, total_correlation_bits


class TestEntropyHelpers:
    def test_joint_entropy_uniform(self):
        """4 equiprobable outcomes → H = 2 bits."""
        m = torch.tensor([[0], [1], [2], [3]] * 250)  # (1000, 1)
        h = joint_entropy_bits(m)
        assert abs(h - 2.0) < 0.01

    def test_joint_entropy_deterministic(self):
        """Deterministic m → H = 0."""
        m = torch.zeros(100, 3, dtype=torch.long)
        h = joint_entropy_bits(m)
        assert h < 1e-6

    def test_tc_independent(self):
        """Independent dimensions → TC ≈ 0."""
        torch.manual_seed(0)
        m = torch.randint(0, 4, (2000, 3))
        tc = total_correlation_bits(m)
        assert tc < 0.05

    def test_tc_perfectly_correlated(self):
        """Perfectly correlated: m_1 = m_0 always → TC > 0."""
        m0 = torch.randint(0, 4, (1000,))
        m = torch.stack([m0, m0, m0], dim=-1)  # (1000, 3) — perfect correlation
        tc = total_correlation_bits(m)
        # Marginals each have H ≈ 2 bits; joint H ≈ 2 bits → TC ≈ 4 bits
        assert tc > 3.0

    def test_tc_z_dim_1_is_zero(self):
        m = torch.randint(0, 8, (500, 1))
        assert total_correlation_bits(m) == 0.0
```

- [ ] **Step 2: Run to verify they fail**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestEntropyHelpers -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'joint_entropy_bits'`

- [ ] **Step 3: Add helpers to network.py**

Add at the top of `network.py` (in the imports section, `import numpy as np` if not already there).

Then add after the `EntropyModelCondZ` class:

```python
# ---------------------------------------------------------------------------
# Empirical entropy utilities (no-grad, batch-level estimates)
# ---------------------------------------------------------------------------

def _marginal_entropy_bits_1d(m_col: torch.Tensor) -> float:
    """Empirical H(m_k) in bits from a 1D integer tensor."""
    m_np = m_col.detach().cpu().numpy().astype(int).ravel()
    _, counts = np.unique(m_np, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def joint_entropy_bits(m: torch.Tensor) -> float:
    """Empirical joint entropy H(m_1,...,m_K) in bits.

    Parameters
    ----------
    m : Tensor of shape (batch, z_dim) — integer-valued
    """
    if m.ndim == 1 or m.shape[-1] == 1:
        return _marginal_entropy_bits_1d(m)
    m_np = m.detach().cpu().numpy().astype(int)
    _, counts = np.unique(m_np, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def total_correlation_bits(m: torch.Tensor) -> float:
    """Empirical total correlation TC = Σ_k H(m_k) − H(m) in bits.

    Non-negative; equals 0 iff all dimensions are mutually independent.
    For z_dim=1 always returns 0.0.

    Parameters
    ----------
    m : Tensor of shape (batch, z_dim) — integer-valued
    """
    if m.ndim == 1 or m.shape[-1] == 1:
        return 0.0
    marginal_sum = float(sum(
        _marginal_entropy_bits_1d(m[:, k]) for k in range(m.shape[-1])
    ))
    return max(0.0, marginal_sum - joint_entropy_bits(m))
```

- [ ] **Step 4: Run tests**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestEntropyHelpers -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full test suite**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/ -v --tb=short 2>&1 | tail -10
```

Expected: 79 + 15 new = 94 passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add onpolicy/envs/toyproblem/network.py \
        onpolicy/envs/toyproblem/tests/test_entropy_model.py
git commit -m "feat(p2): add joint_entropy_bits and total_correlation_bits helpers"
```

---

## Task 5: Extend MAPPOConfig and construct entropy model in trainer

**Files:**
- Modify: `onpolicy/envs/toyproblem/trainer.py`
- Modify: `onpolicy/envs/toyproblem/tests/test_entropy_model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_entropy_model.py`:

```python
from onpolicy.envs.toyproblem.trainer import MAPPOConfig, MAPPOTrainer


class TestTrainerEntropyModelConstruction:
    def _make_trainer(self, **overrides) -> MAPPOTrainer:
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0, lambda_comms=1e-3,
            use_entropy_model=True, entropy_model_K=3,
            **overrides,
        )
        return MAPPOTrainer(cfg, device=torch.device("cpu"))

    def test_factored_A_constructed(self):
        from onpolicy.envs.toyproblem.network import EntropyModelFactored
        t = self._make_trainer(entropy_model_type="factored", entropy_model_context="A")
        assert isinstance(t.entropy_model, EntropyModelFactored)
        assert t.optim_qphi is not None

    def test_joint_A_constructed(self):
        from onpolicy.envs.toyproblem.network import EntropyModelJoint
        t = self._make_trainer(entropy_model_type="joint", entropy_model_context="A")
        assert isinstance(t.entropy_model, EntropyModelJoint)

    def test_cond_z_B_constructed(self):
        from onpolicy.envs.toyproblem.network import EntropyModelCondZ
        t = self._make_trainer(entropy_model_type="factored", entropy_model_context="B")
        assert isinstance(t.entropy_model, EntropyModelCondZ)

    def test_joint_cond_z_B_constructed(self):
        from onpolicy.envs.toyproblem.network import EntropyModelJointCondZ
        t = self._make_trainer(entropy_model_type="joint", entropy_model_context="B")
        assert isinstance(t.entropy_model, EntropyModelJointCondZ)

    def test_entropy_model_none_when_disabled(self):
        cfg = MAPPOConfig(z_dim=3, channel="sd", use_entropy_model=False)
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        assert t.entropy_model is None
        assert t.optim_qphi is None

    def test_qphi_lr_scaled(self):
        """q_φ optimizer lr = lr_qphi_mult × base_lr."""
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", use_entropy_model=True, entropy_model_K=3,
            lr=1e-3, lr_qphi_mult=5.0,
        )
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        actual_lr = t.optim_qphi.param_groups[0]["lr"]
        assert abs(actual_lr - 5e-3) < 1e-9
```

- [ ] **Step 2: Run to verify they fail**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestTrainerEntropyModelConstruction -v 2>&1 | head -15
```

Expected: `TypeError` or `AttributeError` on `MAPPOConfig(use_entropy_model=...)`

- [ ] **Step 3: Extend MAPPOConfig in trainer.py**

In `trainer.py`, update the `MAPPOConfig` dataclass — add new fields **after the existing fields**:

```python
@dataclass
class MAPPOConfig:
    z_dim: int = 3
    hidden_size: int = 64
    lr: float = 3e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.03
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    num_minibatches: int = 4
    adam_eps: float = 1e-5
    channel: str = "none"
    delta: float = 1.0
    lambda_comms: float = 0.0
    ste_clip: float = 10.0
    # P2 — Entropy model
    use_entropy_model: bool = False
    entropy_model_K: int = 5
    entropy_model_type: str = "factored"   # "factored" | "joint"
    entropy_model_context: str = "A"       # "A" (marginal) | "B" (conditioned on z)
    lr_qphi_mult: float = 10.0             # q_φ lr = lr_qphi_mult × lr
    n_qphi_steps: int = 3                  # q_φ gradient steps per RL minibatch
    n_warmup_steps: int = 5000             # q_φ warm-start steps before RL
    loss_comms_mode: str = "magnitude"     # "magnitude" | "entropy" | "both"
```

- [ ] **Step 4: Add imports and construct entropy model in MAPPOTrainer.__init__**

At the top of `trainer.py`, add to the existing import from network.py:

```python
from onpolicy.envs.toyproblem.network import (
    Critic, ListenerActor, SpeakerNetwork,
    EntropyModelFactored, EntropyModelJoint,
    EntropyModelCondZ, EntropyModelJointCondZ,
)
```

In `MAPPOTrainer.__init__`, after `self.value_norm = ValueNorm(...)`, add:

```python
        # P2 — Entropy model and separate q_φ optimizer
        self.entropy_model = None
        self.optim_qphi = None
        if config.use_entropy_model:
            ctx = config.entropy_model_context
            typ = config.entropy_model_type
            K = config.entropy_model_K
            if ctx == "A" and typ == "factored":
                self.entropy_model = EntropyModelFactored(config.z_dim, K).to(device)
            elif ctx == "A" and typ == "joint":
                self.entropy_model = EntropyModelJoint(config.z_dim, K).to(device)
            elif ctx == "B" and typ == "factored":
                self.entropy_model = EntropyModelCondZ(config.z_dim, K).to(device)
            elif ctx == "B" and typ == "joint":
                self.entropy_model = EntropyModelJointCondZ(config.z_dim, K).to(device)
            else:
                raise ValueError(
                    f"Unsupported entropy_model_context={ctx!r}, type={typ!r}. "
                    f"Supported: (A, factored), (A, joint), (B, factored), (B, joint)."
                )
            self.optim_qphi = torch.optim.Adam(
                self.entropy_model.parameters(),
                lr=config.lr * config.lr_qphi_mult,
                eps=config.adam_eps,
            )
```

- [ ] **Step 5: Run tests**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestTrainerEntropyModelConstruction -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add onpolicy/envs/toyproblem/trainer.py \
        onpolicy/envs/toyproblem/tests/test_entropy_model.py
git commit -m "feat(p2): extend MAPPOConfig with P2 fields; construct entropy model in trainer"
```

---

## Task 6: Ballé loss + q_φ optimizer step in trainer.update

**Files:**
- Modify: `onpolicy/envs/toyproblem/trainer.py`
- Modify: `onpolicy/envs/toyproblem/tests/test_entropy_model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_entropy_model.py`:

```python
from onpolicy.envs.toyproblem.buffer import RolloutBuffer


def _make_buffer(n_steps=4, n_envs=4, z_dim=3, device=torch.device("cpu")) -> RolloutBuffer:
    buf = RolloutBuffer(n_steps, n_envs, z_dim, device=device)
    buf.goals = torch.randn(n_steps, n_envs, 2)
    buf.listener_pos = torch.randn(n_steps, n_envs, 2)
    buf.actions = torch.randint(0, 5, (n_steps, n_envs))
    buf.log_probs = torch.randn(n_steps, n_envs)
    buf.advantages = torch.randn(n_steps, n_envs)
    buf.returns = torch.randn(n_steps, n_envs)
    buf.goal_ids = torch.zeros(n_steps, n_envs, dtype=torch.long)
    return buf


class TestTrainerEntropyLoss:
    def _make_trainer_with_em(self, mode="entropy") -> MAPPOTrainer:
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0, lambda_comms=1e-3,
            use_entropy_model=True, entropy_model_K=3,
            entropy_model_type="factored", entropy_model_context="A",
            lr_qphi_mult=5.0, n_qphi_steps=2,
            loss_comms_mode=mode,
            update_epochs=1, num_minibatches=1,
        )
        return MAPPOTrainer(cfg, device=torch.device("cpu"))

    def test_update_returns_entropy_rate(self):
        """update() must include entropy_rate in returned metrics."""
        t = self._make_trainer_with_em()
        buf = _make_buffer(z_dim=3)
        # Need compute_returns_and_advantages first
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        metrics = t.update(buf)
        assert "entropy_rate" in metrics

    def test_update_returns_qphi_gap(self):
        t = self._make_trainer_with_em()
        buf = _make_buffer(z_dim=3)
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        metrics = t.update(buf)
        assert "qphi_gap" in metrics

    def test_entropy_mode_changes_loss(self):
        """With mode='entropy', entropy_rate must be logged and non-zero."""
        t = self._make_trainer_with_em(mode="entropy")
        buf = _make_buffer(z_dim=3)
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        metrics = t.update(buf)
        assert metrics["entropy_rate"] > 0.0

    def test_qphi_params_change_after_update(self):
        """q_φ parameters must be updated by the q_φ optimizer."""
        t = self._make_trainer_with_em()
        buf = _make_buffer(z_dim=3)
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        mu_before = t.entropy_model.mu.detach().clone()
        t.update(buf)
        assert not torch.allclose(t.entropy_model.mu, mu_before)

    def test_no_entropy_metrics_when_disabled(self):
        """Without use_entropy_model, entropy_rate must NOT be in metrics."""
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0,
            use_entropy_model=False,
            update_epochs=1, num_minibatches=1,
        )
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        buf = _make_buffer(z_dim=3)
        buf.advantages = torch.randn(4, 4)
        buf.returns = torch.ones(4, 4)
        metrics = t.update(buf)
        assert "entropy_rate" not in metrics
```

- [ ] **Step 2: Run to verify they fail**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestTrainerEntropyLoss -v 2>&1 | head -20
```

Expected: `AssertionError` on `"entropy_rate" in metrics` (key not yet in metrics dict)

- [ ] **Step 3: Integrate Ballé losses into trainer.update**

In `trainer.py`, add these imports at the top (they should already be there from Task 4):

```python
from onpolicy.envs.toyproblem.network import (
    Critic, ListenerActor, SpeakerNetwork,
    EntropyModelFactored, EntropyModelJoint, EntropyModelCondZ,
    joint_entropy_bits, total_correlation_bits,
)
```

In the `update` method, **replace the existing minibatch loop** with the following (changes are inside the loop — the structure around it stays identical):

```python
    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        adv_flat = buffer.advantages.flatten()
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std()

        metrics: dict[str, list[float]] = defaultdict(list)

        for _ in range(self.config.update_epochs):
            for mb in buffer.minibatches(self.config.num_minibatches):
                z_new = self.speaker(mb["goals"])
                z_hat, ch_info = self.channel(z_new)
                dist = self.listener(torch.cat([mb["listener_pos"], z_hat], dim=-1))
                new_logp = dist.log_prob(mb["actions"])
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - mb["old_log_probs"])
                adv = (mb["advantages"] - adv_mean) / (adv_std + 1e-8)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                pg_loss = -torch.min(surr1, surr2).mean()
                entropy_mean = entropy.mean()
                actor_loss = pg_loss - self.config.entropy_coef * entropy_mean

                returns_mb = mb["returns"].unsqueeze(-1)
                self.value_norm.update(returns_mb)
                returns_norm = self.value_norm.normalize(returns_mb)
                state_mb = torch.cat([mb["listener_pos"], mb["goals"]], dim=-1)
                new_value = self.critic(state_mb)
                critic_loss = 0.5 * (new_value - returns_norm).pow(2).mean()

                comms_per_elem = self.channel.comms_loss(z_new)
                comms_mean = comms_per_elem.mean()

                total_loss = actor_loss + critic_loss
                if self.config.lambda_comms > 0.0:
                    if self.config.loss_comms_mode in ("magnitude", "both"):
                        total_loss = total_loss + self.config.lambda_comms * comms_mean

                # P2: entropy model backward loss (speaker gradient)
                m = ch_info.get("m")
                loss_ent_bwd = None
                if self.entropy_model is not None and m is not None:
                    if self.config.loss_comms_mode in ("entropy", "both"):
                        z_over_delta = z_new / self.config.delta
                        # Freeze q_φ: grad flows to z, not to q_φ params
                        for p in self.entropy_model.parameters():
                            p.requires_grad_(False)
                        if self.config.entropy_model_context == "A":
                            nll_bwd = self.entropy_model.nll_bits(z_over_delta)
                        else:  # context B
                            nll_bwd = self.entropy_model.nll_bits(z_over_delta, z_new.detach())
                        for p in self.entropy_model.parameters():
                            p.requires_grad_(True)
                        loss_ent_bwd = nll_bwd.mean()
                        total_loss = total_loss + self.config.lambda_comms * loss_ent_bwd

                # RL optimizer step
                self.optim.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self._trainable, self.config.max_grad_norm)
                self.optim.step()

                # P2: q_φ forward update (n_qphi_steps)
                if self.entropy_model is not None and m is not None:
                    m_float = m.float().detach()
                    for _ in range(self.config.n_qphi_steps):
                        if self.config.entropy_model_context == "A":
                            nll_fwd = self.entropy_model.nll_bits(m_float)
                        else:  # context B
                            nll_fwd = self.entropy_model.nll_bits(m_float, z_new.detach())
                        loss_q = nll_fwd.mean()
                        self.optim_qphi.zero_grad(set_to_none=True)
                        loss_q.backward()
                        self.optim_qphi.step()

                # Metrics (no_grad)
                with torch.no_grad():
                    approx_kl = (mb["old_log_probs"] - new_logp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()
                    bits_per_msg = comms_per_elem.sum(dim=-1).mean().item()
                    true_bits_per_elem = self.channel.transmission_bits_per_elem(z_new, ch_info)
                    true_bits_per_msg = true_bits_per_elem.sum(dim=-1).mean().item()
                    z_norm = z_new.norm(dim=-1).mean().item()

                metrics["pg_loss"].append(pg_loss.item())
                metrics["value_loss"].append(critic_loss.item())
                metrics["entropy"].append(entropy_mean.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_frac"].append(clip_frac)
                metrics["comms_loss"].append(comms_mean.item())
                metrics["bits_per_msg"].append(bits_per_msg)
                metrics["true_bits_per_msg"].append(true_bits_per_msg)
                metrics["z_norm"].append(z_norm)

                with torch.no_grad():
                    bits_per_elem = comms_per_elem.sum(dim=-1)
                    for g_idx in mb["goal_ids"].unique():
                        mask = mb["goal_ids"] == g_idx
                        key = f"bits_goal_{g_idx.item()}"
                        metrics[key].append(bits_per_elem[mask].mean().item())

                # P2 metrics
                if self.entropy_model is not None and m is not None:
                    with torch.no_grad():
                        m_float = m.float().detach()
                        if self.config.entropy_model_context == "A":
                            nll_log = self.entropy_model.nll_bits(m_float)
                        else:
                            nll_log = self.entropy_model.nll_bits(m_float, z_new.detach())
                        entropy_rate = nll_log.mean().item()
                        qphi_neg_log_max = nll_log.max().item()
                        h_emp = joint_entropy_bits(m.long())
                        tc = total_correlation_bits(m.long())
                        qphi_gap = entropy_rate - h_emp
                        bits_vs_mag = bits_per_msg - entropy_rate

                        metrics["entropy_rate"].append(entropy_rate)
                        metrics["H_m_empirical"].append(h_emp)
                        metrics["qphi_gap"].append(qphi_gap)
                        metrics["tc_bits"].append(tc)
                        metrics["qphi_neg_log_max"].append(qphi_neg_log_max)
                        metrics["bits_vs_magnitude"].append(bits_vs_mag)

                        # Per-goal entropy rate
                        nll_per_msg = nll_log.sum(dim=-1)  # (mb,)
                        for g_idx in mb["goal_ids"].unique():
                            mask = mb["goal_ids"] == g_idx
                            key = f"entropy_rate_goal_{g_idx.item()}"
                            metrics[key].append(nll_per_msg[mask].mean().item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}
```

- [ ] **Step 4: Run tests**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestTrainerEntropyLoss -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full suite**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/ -v --tb=short 2>&1 | tail -5
```

Expected: all previous tests still pass + new ones.

- [ ] **Step 6: Commit**

```bash
git add onpolicy/envs/toyproblem/trainer.py \
        onpolicy/envs/toyproblem/tests/test_entropy_model.py
git commit -m "feat(p2): Ballé two-term entropy loss + q_φ optimizer step in trainer.update"
```

---

## Task 7: Warm-start method

**Files:**
- Modify: `onpolicy/envs/toyproblem/trainer.py`
- Modify: `onpolicy/envs/toyproblem/tests/test_entropy_model.py`

- [ ] **Step 1: Write failing tests**

```python
class TestWarmStart:
    def test_warmup_returns_loss(self):
        """warmup_entropy_model must return a finite float."""
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0,
            use_entropy_model=True, entropy_model_K=3,
            entropy_model_type="factored", entropy_model_context="A",
            update_epochs=1, num_minibatches=1,
        )
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        buf = _make_buffer(z_dim=3, n_steps=8, n_envs=4)
        loss = t.warmup_entropy_model(buf, n_steps=10)
        assert math.isfinite(loss), f"warmup loss not finite: {loss}"

    def test_warmup_updates_qphi(self):
        """After warmup, q_φ params must differ from init."""
        torch.manual_seed(99)
        cfg = MAPPOConfig(
            z_dim=3, channel="sd", delta=1.0,
            use_entropy_model=True, entropy_model_K=3,
            update_epochs=1, num_minibatches=1,
        )
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        mu_before = t.entropy_model.mu.detach().clone()
        buf = _make_buffer(z_dim=3, n_steps=8, n_envs=4)
        t.warmup_entropy_model(buf, n_steps=20)
        assert not torch.allclose(t.entropy_model.mu, mu_before)

    def test_warmup_no_change_without_em(self):
        """warmup_entropy_model is a no-op when use_entropy_model=False."""
        cfg = MAPPOConfig(z_dim=3, channel="sd", use_entropy_model=False)
        t = MAPPOTrainer(cfg, device=torch.device("cpu"))
        result = t.warmup_entropy_model(_make_buffer(z_dim=3), n_steps=10)
        assert result == 0.0
```

- [ ] **Step 2: Run to verify they fail**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestWarmStart -v 2>&1 | head -10
```

Expected: `AttributeError: 'MAPPOTrainer' object has no attribute 'warmup_entropy_model'`

- [ ] **Step 3: Add warmup_entropy_model to MAPPOTrainer in trainer.py**

Add as a new method of `MAPPOTrainer`, before `update`:

```python
    def warmup_entropy_model(self, buffer: RolloutBuffer, n_steps: int) -> float:
        """Pre-train q_φ for n_steps gradient steps before RL begins.

        Uses the current rollout buffer to sample minibatches. The speaker is
        run in no-grad mode to collect discrete messages m. Only q_φ (optim_qphi)
        is updated — RL parameters are untouched.

        Returns the final warm-start loss value (0.0 if entropy model is off).
        """
        if self.entropy_model is None:
            return 0.0
        step = 0
        final_loss = float("nan")
        while step < n_steps:
            for mb in buffer.minibatches(self.config.num_minibatches):
                if step >= n_steps:
                    break
                with torch.no_grad():
                    z = self.speaker(mb["goals"])
                    _, ch_info = self.channel(z)
                m = ch_info.get("m")
                if m is None:
                    return 0.0  # IdentityChannel — no discrete messages
                m_float = m.float()
                if self.config.entropy_model_context == "A":
                    nll = self.entropy_model.nll_bits(m_float)
                else:
                    nll = self.entropy_model.nll_bits(m_float, z.detach())
                loss = nll.mean()
                self.optim_qphi.zero_grad(set_to_none=True)
                loss.backward()
                self.optim_qphi.step()
                final_loss = loss.item()
                step += 1
        return final_loss
```

- [ ] **Step 4: Run tests**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/test_entropy_model.py::TestWarmStart -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add onpolicy/envs/toyproblem/trainer.py \
        onpolicy/envs/toyproblem/tests/test_entropy_model.py
git commit -m "feat(p2): add warmup_entropy_model method for q_φ pre-training"
```

---

## Task 8: Extend train.py — CLI flags and CSV header

**Files:**
- Modify: `onpolicy/envs/toyproblem/train.py`

- [ ] **Step 1: Add P2 CLI flags to parse_args**

In `train.py`, inside `parse_args()`, add after the existing `--ste_clip` argument:

```python
    # P2 — Entropy model
    p.add_argument("--use_entropy_model", action="store_true",
                   help="Enable P2 entropy model (DLM prior q_φ).")
    p.add_argument("--entropy_model_K", type=int, default=5,
                   help="Number of mixture components in DLM prior.")
    p.add_argument("--entropy_model_type", type=str, default="factored",
                   choices=["factored", "joint"],
                   help="factored: independent per-dim DLM. joint: autoregressive DLM.")
    p.add_argument("--entropy_model_context", type=str, default="A",
                   choices=["A", "B"],
                   help="A: marginal prior (deployment-realistic). B: conditioned on z (oracle).")
    p.add_argument("--lr_qphi_mult", type=float, default=10.0,
                   help="q_φ learning rate = lr_qphi_mult × --lr.")
    p.add_argument("--n_qphi_steps", type=int, default=3,
                   help="q_φ gradient steps per RL minibatch.")
    p.add_argument("--n_warmup_steps", type=int, default=5000,
                   help="q_φ warm-start gradient steps before RL begins.")
    p.add_argument("--loss_comms_mode", type=str, default="magnitude",
                   choices=["magnitude", "entropy", "both"],
                   help="magnitude: baseline Jensen surrogate. "
                        "entropy: P2 DLM rate. both: sum of both.")
```

- [ ] **Step 2: Extend CSV_HEADER**

Replace the existing `CSV_HEADER` definition:

```python
_N_GOALS = len(_DEFAULT_GOALS)  # 6

_P2_COLS = [
    "entropy_rate", "H_m_empirical", "qphi_gap",
    "tc_bits", "qphi_neg_log_max", "bits_vs_magnitude",
] + [f"entropy_rate_goal_{i}" for i in range(_N_GOALS)]

CSV_HEADER = [
    "update", "timestep", "mean_reward", "success_rate",
    "pg_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
    "comms_loss", "bits_per_msg", "true_bits_per_msg", "z_norm", "sps",
] + [f"bits_goal_{i}" for i in range(_N_GOALS)] + _P2_COLS
```

- [ ] **Step 3: Pass P2 flags to MAPPOConfig in main()**

In `main()`, update the `config = MAPPOConfig(...)` call to include:

```python
    config = MAPPOConfig(
        z_dim=args.z_dim,
        hidden_size=args.hidden_size,
        lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        channel=args.channel,
        delta=args.delta,
        lambda_comms=args.lambda_comms,
        ste_clip=args.ste_clip,
        use_entropy_model=args.use_entropy_model,
        entropy_model_K=args.entropy_model_K,
        entropy_model_type=args.entropy_model_type,
        entropy_model_context=args.entropy_model_context,
        lr_qphi_mult=args.lr_qphi_mult,
        n_qphi_steps=args.n_qphi_steps,
        n_warmup_steps=args.n_warmup_steps,
        loss_comms_mode=args.loss_comms_mode,
    )
```

- [ ] **Step 4: Add warm-start call and P2 CSV row in main()**

After `trainer = MAPPOTrainer(config, device=device)` and `buffer = RolloutBuffer(...)`, add a warm-start block. This must come **after** the first rollout (so we have data). Add it just before the training loop, inside an initial rollout collection:

Replace the start of the training loop section:

```python
    n_updates = args.total_timesteps // (args.n_envs * args.n_steps)

    obs = env.reset()
    recent_rewards: deque[float] = deque(maxlen=200)
    recent_successes: deque[int] = deque(maxlen=200)

    # Warm-start q_φ on initial rollout data (before RL begins).
    if args.use_entropy_model and args.n_warmup_steps > 0:
        buffer.reset()
        for _ in range(args.n_steps):
            goal_np, lp_np = obs
            goal = torch.from_numpy(goal_np).to(device)
            lp = torch.from_numpy(lp_np).to(device)
            action, log_prob, value = trainer.act_and_value(goal, lp)
            next_obs, reward, done, info = env.step(action.cpu().numpy())
            reward_shared = reward[:, 0, 0]
            done_shared = done[:, 0]
            goal_ids_np = np.array(
                [_GOAL_POS_TO_IDX.get((int(g[0]), int(g[1])), 0) for g in goal_np],
                dtype=np.int64,
            )
            buffer.insert(
                goal, lp, action, log_prob, value,
                torch.from_numpy(reward_shared).to(device),
                torch.from_numpy(done_shared.astype(np.float32)).to(device),
                goal_id=torch.from_numpy(goal_ids_np).to(device),
            )
            obs = next_obs
        warmup_loss = trainer.warmup_entropy_model(buffer, args.n_warmup_steps)
        print(f"[warmup] q_φ pre-training done. final_loss={warmup_loss:.4f}")
```

- [ ] **Step 5: Add P2 columns to CSV row**

In the CSV write section, replace:

```python
        csv_writer.writerow([
            update, timestep, mean_reward, success_rate,
            metrics["pg_loss"], metrics["value_loss"], metrics["entropy"],
            metrics["approx_kl"], metrics["clip_frac"],
            metrics["comms_loss"], metrics["bits_per_msg"],
            metrics["true_bits_per_msg"], metrics["z_norm"], sps,
        ] + per_goal_bits)
```

with:

```python
        per_goal_bits = [
            metrics.get(f"bits_goal_{i}", float("nan")) for i in range(_N_GOALS)
        ]
        p2_vals = [
            metrics.get("entropy_rate", float("nan")),
            metrics.get("H_m_empirical", float("nan")),
            metrics.get("qphi_gap", float("nan")),
            metrics.get("tc_bits", float("nan")),
            metrics.get("qphi_neg_log_max", float("nan")),
            metrics.get("bits_vs_magnitude", float("nan")),
        ] + [metrics.get(f"entropy_rate_goal_{i}", float("nan")) for i in range(_N_GOALS)]
        csv_writer.writerow([
            update, timestep, mean_reward, success_rate,
            metrics["pg_loss"], metrics["value_loss"], metrics["entropy"],
            metrics["approx_kl"], metrics["clip_frac"],
            metrics["comms_loss"], metrics["bits_per_msg"],
            metrics["true_bits_per_msg"], metrics["z_norm"], sps,
        ] + per_goal_bits + p2_vals)
```

Also remove the existing `per_goal_bits` definition that appears just before the old `csv_writer.writerow` call (it will now be defined in the new block above).

- [ ] **Step 6: Commit**

```bash
git add onpolicy/envs/toyproblem/train.py
git commit -m "feat(p2): add 8 CLI flags, extend CSV header with P2 metrics, warm-start call"
```

---

## Task 9: Smoke test — end-to-end run with entropy model

**Files:** none (smoke test only)

- [ ] **Step 1: Run a short training run with --use_entropy_model**

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.train \
    --exp_name debug_p2 --seed 0 \
    --channel sd --delta 1.0 --lambda_comms 1e-3 \
    --use_entropy_model --entropy_model_K 3 \
    --entropy_model_type factored --entropy_model_context A \
    --n_warmup_steps 100 --n_qphi_steps 2 \
    --loss_comms_mode entropy \
    --total_timesteps 10000 --n_envs 4 --n_steps 32 \
    --log_dir /tmp/p2_smoke --log_every 5
```

Expected output:
```
[warmup] q_φ pre-training done. final_loss=X.XXXX
[   0/ 78] t=     128 reward=... bits_surr=... bits_true=... sps=...
...
Done. Logs: /tmp/p2_smoke/debug_p2/0/  metrics: ...  ckpt: ...
```

- [ ] **Step 2: Verify P2 columns appear in CSV**

```bash
head -2 /tmp/p2_smoke/debug_p2/0/metrics.csv | tr ',' '\n' | grep -E "entropy|qphi|tc_bits"
```

Expected: `entropy_rate`, `H_m_empirical`, `qphi_gap`, `tc_bits`, `qphi_neg_log_max`, `bits_vs_magnitude`, `entropy_rate_goal_0` ... `entropy_rate_goal_5`

- [ ] **Step 3: Verify no NaN in P2 columns for the first 5 rows**

```bash
python3 -c "
import csv
with open('/tmp/p2_smoke/debug_p2/0/metrics.csv') as f:
    r = list(csv.DictReader(f))
for row in r[:5]:
    er = row['entropy_rate']
    print('entropy_rate:', er)
    assert er != 'nan', 'got nan!'
print('OK')
"
```

Expected: numeric values, no `nan`.

- [ ] **Step 4: Run full pytest suite**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/ -v --tb=short 2>&1 | tail -5
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A  # in case any __pycache__ changes
git commit -m "test(p2): smoke run verified — all P2 CSV columns present and finite"
```

---

## Task 10: Ablation runner — 6-stage systematic study

**Files:**
- Create: `onpolicy/envs/toyproblem/experiments/run_p2_ablation.py`

The runner supports `--stage P2-A|P2-B|P2-C|P2-D|P2-E|P2-F|all`. Stages P2-C through P2-F require winner flags from earlier stages — pass them via `--best_K`, `--best_model_type`, `--best_context`, `--best_loss_mode`, `--best_lambda`, `--best_z_dim`, `--best_delta`. See PILLAR_P2.md §10 for stage ordering and run counts.

- [ ] **Step 1: Create the ablation runner**

Create `onpolicy/envs/toyproblem/experiments/run_p2_ablation.py`:

```python
"""P2 systematic ablation study — 6 staged experiments, ~395 total runs.

Stage ordering (each stage requires winners from prior stages):
  P2-A  Model selection   (165 runs) — vary K, model_type, context, loss_comms_mode
  P2-B  λ re-sweep         (70 runs) — P2 rate-distortion frontier; find optimal λ
  P2-C  z_dim interaction  (40 runs) — factored vs joint gap at z_dim={1,2,3}
  P2-D  δ interaction      (40 runs) — P2 gain vs quantisation width
  P2-E  Channel (sd/nsd)   (20 runs) — P2 compatibility with NSD dithering
  P2-F  Robustness OAT     (60 runs) — lr_qphi_mult, n_qphi_steps, n_warmup_steps

Usage (from repo root):
    # Stage P2-A (model selection):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.experiments.run_p2_ablation \\
        --stage P2-A --seeds 0 1 2 3 4 --log_dir runs/toyproblem/p2_ablation

    # Stage P2-B (after inspecting P2-A results):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.experiments.run_p2_ablation \\
        --stage P2-B --seeds 0 1 2 3 4 --log_dir runs/toyproblem/p2_ablation \\
        --best_K 5 --best_model_type factored --best_context A --best_loss_mode entropy

    # Stages P2-C through P2-F (after P2-B results):
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \\
        python -m onpolicy.envs.toyproblem.experiments.run_p2_ablation \\
        --stage P2-C --seeds 0 1 2 3 4 --log_dir runs/toyproblem/p2_ablation \\
        --best_K 5 --best_model_type factored --best_context A \\
        --best_loss_mode entropy --best_lambda 4e-3

    # Dry run any stage:
        ... --dry_run

Analysis after each stage:
    from onpolicy.envs.toyproblem.analysis.load_runs import load_sweep, final_metrics, seed_aggregate
    df = load_sweep("runs/toyproblem/p2_ablation")
    summary = final_metrics(df)
    # group by exp_name to compare conditions within each stage
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterator


_BASE_TRAIN_CMD = [sys.executable, "-m", "onpolicy.envs.toyproblem.train"]

# Fixed settings shared across all stages (overridden per-stage as needed).
# Populated from Phase 2 baseline_best.yaml defaults — edit if Phase 2 winner differs.
_SHARED = dict(
    channel="sd",
    total_timesteps="500000",
    n_envs="16",
    n_steps="256",
    hidden_size="64",
    lr="3e-4",
    update_epochs="10",
    num_minibatches="4",
)

# P2 training defaults (used unless overridden by robustness OAT in P2-F)
_P2_TRAIN_DEFAULTS = dict(
    n_warmup_steps="5000",
    n_qphi_steps="3",
    lr_qphi_mult="10.0",
)


def _run(exp_name: str, seed: int, log_dir: str, extra: dict) -> None:
    cmd = _BASE_TRAIN_CMD + ["--exp_name", exp_name, "--seed", str(seed),
                              "--log_dir", log_dir]
    for k, v in extra.items():
        if v == "":          # store_true flags (e.g. --use_entropy_model)
            cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    print(f"  [run] {exp_name} seed={seed}")
    subprocess.run(cmd, check=True)


def _stage_a(args) -> Iterator[tuple[str, dict]]:
    """Model selection: K × model_type × context × loss_comms_mode (165 runs)."""
    base = {**_SHARED, "delta": args.phase2_delta, "z_dim": args.phase2_z_dim,
            "lambda_comms": args.phase2_lambda, **_P2_TRAIN_DEFAULTS}

    yield "baseline_magnitude", base

    K_values = [1, 3, 5, 10, 20]
    for K in K_values:
        for mode in ["entropy", "both"]:
            for model_type in ["factored", "joint"]:
                for context in ["A", "B"]:
                    # Skip (joint, B) for K != 5 to keep run count manageable;
                    # (joint, B) K=5 is sufficient to close the 2x2 grid.
                    if context == "B" and model_type == "joint" and K != 5:
                        continue
                    name = f"p2_{context}_{model_type}_K{K}_{mode}"
                    yield name, {**base,
                                  "use_entropy_model": "",
                                  "entropy_model_K": str(K),
                                  "entropy_model_type": model_type,
                                  "entropy_model_context": context,
                                  "loss_comms_mode": mode}


def _stage_b(args) -> Iterator[tuple[str, dict]]:
    """λ re-sweep for best P2 config; generates P2 rate-distortion frontier (35 new runs)."""
    base = {**_SHARED, "delta": args.phase2_delta, "z_dim": args.phase2_z_dim,
            **_P2_TRAIN_DEFAULTS}
    lambdas = ["1e-5", "1e-4", "5e-4", "1e-3", "4e-3", "1e-2", "3e-2"]
    for lam in lambdas:
        name = f"p2_{args.best_context}_{args.best_model_type}_K{args.best_K}_{args.best_loss_mode}_lam{lam}"
        yield name, {**base, "lambda_comms": lam,
                     "use_entropy_model": "",
                     "entropy_model_K": str(args.best_K),
                     "entropy_model_type": args.best_model_type,
                     "entropy_model_context": args.best_context,
                     "loss_comms_mode": args.best_loss_mode}


def _stage_c(args) -> Iterator[tuple[str, dict]]:
    """z_dim interaction: factored vs joint gap at z_dim={1,2,3} (40 runs)."""
    base = {**_SHARED, "delta": args.phase2_delta,
            "lambda_comms": args.best_lambda, **_P2_TRAIN_DEFAULTS}
    for z_dim in [1, 2, 3]:
        # Baseline
        yield f"baseline_magnitude_zdim{z_dim}", {**base, "z_dim": str(z_dim)}
        # Factored
        yield f"p2_A_factored_K{args.best_K}_zdim{z_dim}", {
            **base, "z_dim": str(z_dim),
            "use_entropy_model": "",
            "entropy_model_K": str(args.best_K),
            "entropy_model_type": "factored",
            "entropy_model_context": "A",
            "loss_comms_mode": args.best_loss_mode,
        }
        # Joint only at z_dim >= 2 (trivially equal to factored at z_dim=1)
        if z_dim >= 2:
            yield f"p2_A_joint_K{args.best_K}_zdim{z_dim}", {
                **base, "z_dim": str(z_dim),
                "use_entropy_model": "",
                "entropy_model_K": str(args.best_K),
                "entropy_model_type": "joint",
                "entropy_model_context": "A",
                "loss_comms_mode": args.best_loss_mode,
            }


def _stage_d(args) -> Iterator[tuple[str, dict]]:
    """δ interaction: P2 gain vs quantisation width (40 runs)."""
    base = {**_SHARED, "z_dim": args.best_z_dim, "lambda_comms": args.best_lambda,
            **_P2_TRAIN_DEFAULTS}
    for delta in ["0.5", "1.0", "5.0", "10.0"]:
        yield f"baseline_magnitude_delta{delta}", {**base, "delta": delta}
        yield f"p2_best_delta{delta}", {
            **base, "delta": delta,
            "use_entropy_model": "",
            "entropy_model_K": str(args.best_K),
            "entropy_model_type": args.best_model_type,
            "entropy_model_context": args.best_context,
            "loss_comms_mode": args.best_loss_mode,
        }


def _stage_e(args) -> Iterator[tuple[str, dict]]:
    """Channel interaction: P2 with sd vs nsd (20 runs)."""
    base = {**_SHARED, "z_dim": args.best_z_dim, "delta": args.best_delta,
            "lambda_comms": args.best_lambda, **_P2_TRAIN_DEFAULTS}
    for channel in ["sd", "nsd"]:
        yield f"baseline_magnitude_{channel}", {**base, "channel": channel}
        yield f"p2_best_{channel}", {
            **base, "channel": channel,
            "use_entropy_model": "",
            "entropy_model_K": str(args.best_K),
            "entropy_model_type": args.best_model_type,
            "entropy_model_context": args.best_context,
            "loss_comms_mode": args.best_loss_mode,
        }


def _stage_f(args) -> Iterator[tuple[str, dict]]:
    """Training robustness OAT: lr_qphi_mult, n_qphi_steps, n_warmup_steps (60 runs)."""
    base = {**_SHARED, "z_dim": args.best_z_dim, "delta": args.best_delta,
            "channel": "sd", "lambda_comms": args.best_lambda,
            "use_entropy_model": "",
            "entropy_model_K": str(args.best_K),
            "entropy_model_type": args.best_model_type,
            "entropy_model_context": args.best_context,
            "loss_comms_mode": args.best_loss_mode}

    for mult in [1, 5, 10, 50]:
        yield f"p2_lrqphi{mult}x", {**base, "n_warmup_steps": "5000",
                                      "n_qphi_steps": "3",
                                      "lr_qphi_mult": str(float(mult))}
    for n in [1, 3, 5, 10]:
        yield f"p2_nsteps{n}", {**base, "n_warmup_steps": "5000",
                                  "lr_qphi_mult": "10.0",
                                  "n_qphi_steps": str(n)}
    for ws in [0, 1000, 5000, 20000]:
        yield f"p2_warmup{ws}", {**base, "lr_qphi_mult": "10.0",
                                   "n_qphi_steps": "3",
                                   "n_warmup_steps": str(ws)}


_STAGE_FNS = {
    "P2-A": _stage_a,
    "P2-B": _stage_b,
    "P2-C": _stage_c,
    "P2-D": _stage_d,
    "P2-E": _stage_e,
    "P2-F": _stage_f,
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="P2 systematic ablation runner. See PILLAR_P2.md §10 for stage ordering."
    )
    p.add_argument("--stage", type=str,
                   choices=list(_STAGE_FNS.keys()) + ["all"],
                   required=True, help="Which stage to run.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--log_dir", type=str, default="runs/toyproblem/p2_ablation")
    p.add_argument("--dry_run", action="store_true",
                   help="Print run list without executing.")
    # Phase 2 winners (defaults are sensible placeholders; update after Phase 2)
    p.add_argument("--phase2_lambda", type=str, default="1e-3")
    p.add_argument("--phase2_delta", type=str, default="1.0")
    p.add_argument("--phase2_z_dim", type=str, default="3")
    # P2-A winners (required for stages B-F)
    p.add_argument("--best_K", type=int, default=5)
    p.add_argument("--best_model_type", type=str, default="factored",
                   choices=["factored", "joint"])
    p.add_argument("--best_context", type=str, default="A", choices=["A", "B"])
    p.add_argument("--best_loss_mode", type=str, default="entropy",
                   choices=["entropy", "both"])
    # P2-B winners (required for stages C-F)
    p.add_argument("--best_lambda", type=str, default="1e-3")
    # P2-C winners (required for stages D-F)
    p.add_argument("--best_z_dim", type=str, default="3")
    # P2-D winners (required for stages E-F)
    p.add_argument("--best_delta", type=str, default="1.0")
    args = p.parse_args()

    stages = list(_STAGE_FNS.keys()) if args.stage == "all" else [args.stage]
    for stage in stages:
        conditions = list(_STAGE_FNS[stage](args))
        n_runs = len(conditions) * len(args.seeds)
        print(f"\n=== {stage}: {len(conditions)} conditions × {len(args.seeds)} seeds = {n_runs} runs ===")
        for i, (exp_name, extra) in enumerate(conditions):
            for seed in args.seeds:
                if args.dry_run:
                    print(f"  [{i}] {exp_name} seed={seed}")
                else:
                    _run(exp_name, seed, args.log_dir, extra)
    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify dry run for each stage**

```bash
for stage in P2-A P2-B P2-C P2-D P2-E P2-F; do
    echo "=== $stage ==="
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
        python -m onpolicy.envs.toyproblem.experiments.run_p2_ablation \
        --stage $stage --seeds 0 1 --dry_run \
        --best_K 5 --best_model_type factored --best_context A \
        --best_loss_mode entropy --best_lambda 4e-3 \
        --best_z_dim 3 --best_delta 1.0 2>&1 | head -5
done
```

Expected: each stage prints its condition list (no errors, no actual runs).

- [ ] **Step 3: Verify total run count**

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -m onpolicy.envs.toyproblem.experiments.run_p2_ablation \
    --stage all --seeds 0 1 2 3 4 --dry_run \
    --best_K 5 --best_model_type factored --best_context A \
    --best_loss_mode entropy --best_lambda 4e-3 \
    --best_z_dim 3 --best_delta 1.0 2>&1 | grep "runs ==="
```

Expected output lines (approximately):
```
=== P2-A: 33 conditions × 5 seeds = 165 runs ===
=== P2-B:  7 conditions × 5 seeds =  35 runs ===
=== P2-C:  8 conditions × 5 seeds =  40 runs ===
=== P2-D:  8 conditions × 5 seeds =  40 runs ===
=== P2-E:  4 conditions × 5 seeds =  20 runs ===
=== P2-F: 12 conditions × 5 seeds =  60 runs ===
```

- [ ] **Step 4: Commit**

```bash
git add onpolicy/envs/toyproblem/experiments/run_p2_ablation.py
git commit -m "feat(p2): add 6-stage systematic ablation runner (395 total runs)"
```

---

## Task 11: P2 paper figures

**Files:**
- Modify: `onpolicy/envs/toyproblem/analysis/paper_figures.py`

- [ ] **Step 1: Add 4 P2 figure functions**

Append to `onpolicy/envs/toyproblem/analysis/paper_figures.py`:

```python
# ---------------------------------------------------------------------------
# P2 Figures — Entropy Model
# ---------------------------------------------------------------------------

def plot_p2_entropy_rate_vs_lambda(
    agg: pd.DataFrame,
    K_values: Sequence[int] = (1, 3, 5, 10, 20),
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    Higher K (mixture components) in the DLM prior reduces entropy_rate (cross-entropy
    approaches H(m)) and improves success_rate at the same lambda_comms.

    Analysis
    --------
    Line plots of entropy_rate_mean vs lambda_comms for each K value (Context A,
    factored). Shaded band = bootstrap CI. A horizontal dashed line marks the
    baseline magnitude surrogate (bits_per_msg from baseline_magnitude run).

    Conclusion
    ----------
    If K=5 matches K=10/20, the mixture is expressive enough; K=1 (Gaussian)
    is the simplest useful special case.
    """
    _require_mpl()
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    cmap = cm.get_cmap("viridis", len(K_values))
    for i, K in enumerate(K_values):
        mask = agg["entropy_model_K"] == K
        sub = agg[mask].sort_values("lambda_comms")
        if sub.empty:
            continue
        axes[0].plot(
            sub["lambda_comms"], sub["entropy_rate_mean"],
            color=cmap(i), label=f"K={K}",
        )
        if "entropy_rate_ci_lo" in sub.columns:
            axes[0].fill_between(
                sub["lambda_comms"],
                sub["entropy_rate_ci_lo"], sub["entropy_rate_ci_hi"],
                color=cmap(i), alpha=0.15,
            )
        axes[1].plot(
            sub["lambda_comms"], sub["success_rate_mean"],
            color=cmap(i), label=f"K={K}",
        )

    for ax in axes:
        ax.set_xlabel("λ (lambda_comms)")
        ax.legend(title="Mixture K")
        ax.set_xscale("log")
    axes[0].set_ylabel("Entropy rate (bits/msg)")
    axes[0].set_title("Cross-entropy vs λ")
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("Task success vs λ")
    fig.suptitle("P2: DLM mixture components ablation (Context A, factored)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_p2_qphi_gap_training(
    df: pd.DataFrame,
    exp_names: Sequence[str] | None = None,
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    qphi_gap (cross-entropy − empirical H(m)) decreases over training as q_φ
    converges to the true message distribution. Wide-scale initialisation
    prevents early collapse.

    Analysis
    --------
    Training curves of qphi_gap over timesteps, one line per exp_name seed-mean.
    Shaded band = ±1 seed std. A dashed line at 0 marks perfect fit.

    Conclusion
    ----------
    If qphi_gap converges toward 0 and stays there, q_φ is tracking p(m) well.
    Spikes indicate warm-start distribution shift (see PILLAR_P2.md §9).
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    if exp_names is None:
        exp_names = df["exp_name"].unique().tolist() if "exp_name" in df.columns else []

    for exp in exp_names:
        sub = df[df["exp_name"] == exp] if "exp_name" in df.columns else df
        if "qphi_gap" not in sub.columns:
            continue
        grouped = sub.groupby("timestep")["qphi_gap"]
        mean = grouped.mean()
        std = grouped.std()
        ax.plot(mean.index, mean.values, label=exp)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)

    ax.axhline(0, linestyle="--", color="black", linewidth=1, label="perfect fit")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("qphi_gap (bits)")
    ax.set_title("P2: q_φ convergence diagnostic")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_p2_factored_vs_joint(
    agg: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    Joint autoregressive DLM achieves lower entropy_rate than factored DLM when
    message dimensions are correlated (tc_bits > 0), but at higher computational cost.

    Analysis
    --------
    Scatter of (entropy_rate_mean, success_rate_mean) for factored vs joint K=5,
    Context A. Annotated with tc_bits value to show when dimensions are correlated.

    Conclusion
    ----------
    If tc_bits ≈ 0, factored = joint. If tc_bits > 0, joint should have strictly
    lower entropy_rate (the improvement equals tc_bits by TC decomposition).
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {"factored": "#1f77b4", "joint": "#ff7f0e"}
    for model_type, color in colors.items():
        if "entropy_model_type" not in agg.columns:
            continue
        sub = agg[agg["entropy_model_type"] == model_type]
        if sub.empty:
            continue
        ax.scatter(
            sub["entropy_rate_mean"], sub["success_rate_mean"],
            color=color, label=model_type, s=60, alpha=0.8,
        )

    ax.set_xlabel("Entropy rate (bits/msg)")
    ax.set_ylabel("Success rate")
    ax.set_title("P2: Factored vs joint DLM (K=5, Context A)")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_p2_context_comparison(
    agg: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple:
    """
    Hypothesis
    ----------
    entropy_rate_A ≥ entropy_rate_B ≥ H(m). The gap A−B quantifies I(z; m).

    Analysis
    --------
    Box plots (or bar plots with error bars) of entropy_rate for contexts A and B
    (K=5, factored). A horizontal dashed line at H_m_empirical mean shows the
    theoretical minimum.

    Conclusion
    ----------
    A small gap A−B means z carries little extra information about m beyond the
    marginal distribution — Context A is nearly optimal.
    """
    _require_mpl()
    _paper_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    contexts = ["A", "B"]
    colors = {"A": "#1f77b4", "B": "#ff7f0e"}
    x_pos = list(range(len(contexts)))
    for i, ctx in enumerate(contexts):
        if "entropy_model_context" not in agg.columns:
            continue
        sub = agg[agg["entropy_model_context"] == ctx]
        if sub.empty:
            continue
        er = sub["entropy_rate_mean"]
        ci_lo = sub.get("entropy_rate_ci_lo", er)
        ci_hi = sub.get("entropy_rate_ci_hi", er)
        ax.bar(i, er.mean(), color=colors[ctx], label=f"Context {ctx}", alpha=0.8)
        ax.errorbar(i, er.mean(),
                    yerr=[[er.mean() - ci_lo.mean()], [ci_hi.mean() - er.mean()]],
                    color="black", capsize=5)

    # H(m) reference line
    if "H_m_empirical_mean" in agg.columns:
        h_emp = agg["H_m_empirical_mean"].mean()
        ax.axhline(h_emp, linestyle="--", color="black", label="H(m) empirical")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Context {c}" for c in contexts])
    ax.set_ylabel("Entropy rate (bits/msg)")
    ax.set_title("P2: Rate bound tightness by conditioning context")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax
```

- [ ] **Step 2: Verify no import errors**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    python -c "from onpolicy.envs.toyproblem.analysis.paper_figures import \
        plot_p2_entropy_rate_vs_lambda, plot_p2_qphi_gap_training, \
        plot_p2_factored_vs_joint, plot_p2_context_comparison; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add onpolicy/envs/toyproblem/analysis/paper_figures.py
git commit -m "feat(p2): add 4 P2 paper figure functions"
```

---

## Task 12: Update docs and run full test suite

**Files:**
- Modify: `onpolicy/envs/toyproblem/docs/pillars/PILLAR_P2.md`
- Modify: `onpolicy/envs/toyproblem/PLAN.md`
- Modify: `onpolicy/envs/toyproblem/CONTEXT.md`

- [ ] **Step 1: Update PILLAR_P2.md status**

Change the status line at the top of `docs/pillars/PILLAR_P2.md`:

```
**Status:** IMPLEMENTATION COMPLETE — ablation runs pending (Phase 3)
```

- [ ] **Step 2: Update PLAN.md Phase 3 section**

In `PLAN.md`, find the Phase 3 section and mark P2 tasks done:

```
| 3 — 4 Pillars (P2→P1→P4→P3) | 🔶 P2 DONE | pytest gate: see below |
```

And under Phase 3 checklist, mark:
- [x] P2: EntropyModelFactored (network.py)
- [x] P2: EntropyModelJoint (network.py)
- [x] P2: EntropyModelCondZ / Context B (network.py)
- [x] P2: Ballé gradient path + q_φ optimizer (trainer.py)
- [x] P2: warm-start (trainer.py)
- [x] P2: P2 metrics in CSV (trainer.py + train.py)
- [x] P2: run_p2_ablation.py
- [x] P2: paper figures

- [ ] **Step 3: Add session log entry to CONTEXT.md**

Add at the top of the Session Log:

```
**Session 11 (2026-04-24):** Implemented Pillar P2 (Entropy Model) end-to-end. Added EntropyModelFactored, EntropyModelJoint, EntropyModelCondZ to network.py; joint_entropy_bits and total_correlation_bits helpers. Extended MAPPOConfig with 8 new P2 fields; integrated Ballé two-term loss (fwd trains q_φ, bwd propagates to speaker with frozen q_φ) and warm-start into trainer.py. Added 8 CLI flags and P2 CSV columns to train.py. Smoke test passes; all previous tests still pass. Added run_p2_ablation.py and 4 paper figures.
```

- [ ] **Step 4: Run full pytest suite one final time**

```
KMP_DUPLICATE_LIB_OK=TRUE conda run -n marl_comms \
    pytest onpolicy/envs/toyproblem/tests/ -v --tb=short 2>&1 | tail -10
```

Expected: all tests pass, count ≥ 94.

- [ ] **Step 5: Commit**

```bash
git add -f \
    onpolicy/envs/toyproblem/docs/pillars/PILLAR_P2.md \
    onpolicy/envs/toyproblem/PLAN.md \
    onpolicy/envs/toyproblem/CONTEXT.md
git commit -m "docs(p2): mark P2 implementation complete; update PLAN.md and session log"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] §2 Prior family (DLM) — Tasks 1–3b
- [x] §2 Factored vs joint — Tasks 1–2
- [x] §2 TC looseness bound — Task 4
- [x] §3 Context A/B — Tasks 1, 2, 3, 3b; full 2×2 (factored/joint) × (A/B) — all 4 cells
- [x] §4 Ballé gradient path (fwd + bwd) — Task 6
- [x] §5 Unbounded m (DLM tails, wide init) — Task 1 (log_s=1 init)
- [x] §6 Moving target (lr_qphi, n_qphi_steps, warmup) — Tasks 5, 7, 8
- [x] §7 All 7 metrics — Task 6
- [x] §8 qphi_sharing — NOT implemented (deferred; toy problem has 1 speaker so per_agent=shared)
- [x] §9 Failure modes — diagnostics are the metrics; no code change needed
- [x] §10 Staged ablation (P2-A through P2-F, 395 runs) — Task 10
- [x] §11 Implementation plan steps — all covered

**qphi_sharing gap:** The spec lists `qphi_sharing ∈ {per_agent, shared}` as a CLI flag. Deferred — in the toy problem there is 1 speaker so both modes are equivalent. Add when implementing a multi-agent environment.

**Type consistency:**
- `EntropyModelFactored.nll_bits(x)` — context A, factored
- `EntropyModelJoint.nll_bits(x)` — context A, joint
- `EntropyModelCondZ.nll_bits(x, z)` — context B, factored
- `EntropyModelJointCondZ.nll_bits(x, z)` — context B, joint
All four used consistently in trainer Tasks 5–7.

**Context B backward loss note:** In Task 6 the backward loss for Context B uses `z_over_delta` as `x` and `z_new.detach()` as context. This is correct: z_over_delta has grad (flows to speaker), and we detach z_new as context to avoid a second grad path through context.

**Stage ordering note:** P2-A can run immediately after smoke test (Task 9). P2-B requires reading P2-A results and picking winner flags. P2-C through P2-F each require reading results from all prior stages. Do NOT pass stale defaults — update `--best_*` flags from actual sweep results.
