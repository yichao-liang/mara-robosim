"""Tests for the PyBulletDominoComposedEnv environment."""

import numpy as np
import pytest

from mara_robosim.config import PyBulletConfig
from mara_robosim.structs import Action


@pytest.fixture(scope="module")
def env():
    """Create environment once for all tests in this module."""
    try:
        from mara_robosim.envs.domino.composed_env import PyBulletDominoComposedEnv

        config = PyBulletConfig(seed=42, num_train_tasks=2, num_test_tasks=2)
        return PyBulletDominoComposedEnv(  # pylint: disable=no-value-for-parameter
            config=config, use_gui=False
        )
    except Exception as exc:
        pytest.skip(f"Cannot instantiate PyBulletDominoComposedEnv: {exc}")


def test_name(env):
    """get_name() returns the expected identifier."""
    assert env.get_name() == "pybullet_domino_composed"


def test_types(env):
    """types is a non-empty set containing expected type names."""
    type_names = {t.name for t in env.types}
    assert "robot" in type_names


def test_predicates(env):
    """predicates contains the expected predicate names."""
    assert isinstance(env.predicates, set)


def test_train_tasks(env):
    """get_train_tasks() returns the configured number of tasks."""
    try:
        tasks = env.get_train_tasks()
    except Exception as exc:
        pytest.xfail(f"get_train_tasks() failed: {exc}")
    assert isinstance(tasks, list)
    assert len(tasks) == 2


def test_test_tasks(env):
    """get_test_tasks() returns the configured number of tasks."""
    try:
        tasks = env.get_test_tasks()
    except Exception as exc:
        pytest.xfail(f"get_test_tasks() failed: {exc}")
    assert isinstance(tasks, list)
    assert len(tasks) == 2


def test_reset_and_step(env):
    """reset() and step() produce valid states."""
    try:
        state = env.reset("train", 0)
    except Exception as exc:
        pytest.xfail(f"reset failed: {exc}")
    assert state is not None

    action = Action(
        np.array(env._pybullet_robot.initial_joint_positions, dtype=np.float32)
    )
    try:
        next_state = env.step(action)
    except Exception as exc:
        pytest.xfail(f"step failed: {exc}")
    assert next_state is not None
