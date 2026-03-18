"""Tests for the gymnasium wrapper and environment registration."""

import gymnasium
import numpy as np
import pytest

from mara_robosim import get_all_env_ids, make, register_all_environments
from mara_robosim.config import PyBulletConfig
from mara_robosim.structs import State

# ---------------------------------------------------------------------------
# Module-scoped fixture -- the Ants env is expensive to create, so we
# share one instance across all tests in this module.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ants_env():
    """Create a mara/Ants-v0 gymnasium environment once for the module."""
    config = PyBulletConfig(seed=42, num_train_tasks=1, num_test_tasks=1)
    env = make("mara/Ants-v0", config=config)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_register_all_environments_count():
    """register_all_environments() registers all 15 envs."""
    register_all_environments()
    mara_ids = {eid for eid in gymnasium.registry if eid.startswith("mara/")}
    assert len(mara_ids) == 15


def test_get_all_env_ids_returns_15():
    """get_all_env_ids() returns exactly 15 ids."""
    env_ids = get_all_env_ids()
    assert len(env_ids) == 15


def test_get_all_env_ids_prefix():
    """Every id returned by get_all_env_ids() starts with 'mara/'."""
    for eid in get_all_env_ids():
        assert eid.startswith("mara/"), f"{eid} does not start with 'mara/'"


# ---------------------------------------------------------------------------
# Environment creation tests
# ---------------------------------------------------------------------------


def test_make_creates_env(ants_env):
    """make('mara/Ants-v0') creates a working gymnasium env."""
    assert ants_env is not None
    # The wrapper should be a gymnasium.Env
    assert isinstance(ants_env, gymnasium.Env)


def test_observation_space(ants_env):
    """The env has a proper Box observation space with finite bounds."""
    obs_space = ants_env.observation_space
    assert isinstance(obs_space, gymnasium.spaces.Box)
    assert len(obs_space.shape) == 1
    assert obs_space.shape[0] > 0
    assert obs_space.dtype == np.float32


def test_action_space(ants_env):
    """The env has a proper Box action space."""
    act_space = ants_env.action_space
    assert isinstance(act_space, gymnasium.spaces.Box)
    assert len(act_space.shape) >= 1
    assert act_space.shape[0] > 0


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------


def test_reset_returns_tuple(ants_env):
    """reset() returns (obs, info) with correct shapes."""
    result = ants_env.reset()
    assert isinstance(result, tuple)
    assert len(result) == 2

    obs, info = result
    assert isinstance(obs, np.ndarray)
    assert obs.shape == ants_env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# step() tests
# ---------------------------------------------------------------------------


def test_step_returns_five_tuple(ants_env):
    """step() returns (obs, reward, terminated, truncated, info)."""
    ants_env.reset()
    action = ants_env.action_space.sample()
    result = ants_env.step(action)

    assert isinstance(result, tuple)
    assert len(result) == 5

    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert obs.shape == ants_env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# info dict tests
# ---------------------------------------------------------------------------


def test_info_contains_state_key(ants_env):
    """info from reset() contains a 'state' key holding a State."""
    _, info = ants_env.reset()
    assert "state" in info
    assert isinstance(info["state"], State)


def test_info_contains_goal_reached_key(ants_env):
    """info from reset() contains a 'goal_reached' key (bool)."""
    _, info = ants_env.reset()
    assert "goal_reached" in info
    assert isinstance(info["goal_reached"], bool)


def test_step_info_contains_required_keys(ants_env):
    """info from step() also contains 'state' and 'goal_reached'."""
    ants_env.reset()
    action = ants_env.action_space.sample()
    _, _, _, _, info = ants_env.step(action)
    assert "state" in info
    assert "goal_reached" in info
    assert isinstance(info["state"], State)
    assert isinstance(info["goal_reached"], bool)
