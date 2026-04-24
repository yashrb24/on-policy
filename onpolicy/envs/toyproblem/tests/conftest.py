"""Shared fixtures for the toyproblem test suite."""
from __future__ import annotations

import pytest
import torch

from onpolicy.envs.toyproblem.channels import (
    DDCL_NSD, DDCL_SD, IdentityChannel,
    AdditiveUniformChannel, GaussianChannel, STEChannel,
    build_channel,
)
from onpolicy.envs.toyproblem.CommunicatingGoal_env import CommunicatingGoalEnv
from onpolicy.envs.toyproblem.CommunicatingGoal_vec_env import CommunicatingGoalVecEnv


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Default quantisation parameters
# ---------------------------------------------------------------------------

CANONICAL_DELTAS = [1.0, 5.0, 10.0, 15.0, 20.0]

@pytest.fixture(params=CANONICAL_DELTAS, ids=[f"delta={d}" for d in CANONICAL_DELTAS])
def delta(request) -> float:
    return request.param


@pytest.fixture
def default_delta() -> float:
    return 10.0


# ---------------------------------------------------------------------------
# Channel fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_channel() -> IdentityChannel:
    return IdentityChannel(delta=10.0)


@pytest.fixture
def sd_channel(default_delta) -> DDCL_SD:
    return DDCL_SD(delta=default_delta)


@pytest.fixture
def nsd_channel(default_delta) -> DDCL_NSD:
    return DDCL_NSD(delta=default_delta)


@pytest.fixture(params=["none", "sd", "nsd", "additive_uniform", "gaussian",
                        "ste4", "ste8", "ste16"],
                ids=["identity", "DDCL_SD", "DDCL_NSD",
                     "additive_uniform", "gaussian", "ste4", "ste8", "ste16"])
def any_channel(request, default_delta):
    """Parametrized fixture that runs tests against all channel types."""
    return build_channel(request.param, default_delta)


@pytest.fixture
def additive_uniform_channel(default_delta) -> AdditiveUniformChannel:
    return AdditiveUniformChannel(delta=default_delta)


@pytest.fixture
def gaussian_channel(default_delta) -> GaussianChannel:
    return GaussianChannel(delta=default_delta)


@pytest.fixture(params=[4, 8, 16], ids=["ste4", "ste8", "ste16"])
def ste_channel(request) -> STEChannel:
    return STEChannel(bits=request.param, clip_val=10.0)


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_env() -> CommunicatingGoalEnv:
    env = CommunicatingGoalEnv()
    env.seed(0)
    return env


@pytest.fixture
def vec_env_small() -> CommunicatingGoalVecEnv:
    """4-env vectorized env for fast integration tests."""
    env = CommunicatingGoalVecEnv(num_envs=4)
    env.seed(0)
    return env


# ---------------------------------------------------------------------------
# Seeding helper
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed_torch():
    """Reset PyTorch RNG before each test for reproducibility."""
    torch.manual_seed(0)
    yield
