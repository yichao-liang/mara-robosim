"""PyBullet Domino Environment Package.

This package provides a modular, component-based domino environment for PyBullet.

Example usage:

    from mara_robosim.envs.domino import PyBulletDominoEnv, PyBulletDominoFanEnv

    env = PyBulletDominoEnv(use_gui=True)
    # or
    env = PyBulletDominoFanEnv(use_gui=True)
"""

from mara_robosim.envs.domino.composed_env import (
    PyBulletDominoEnvNew,
    PyBulletDominoFanEnvNew,
)

# Backward-compatible aliases
PyBulletDominoEnv = PyBulletDominoEnvNew
PyBulletDominoFanEnv = PyBulletDominoFanEnvNew

__all__ = [
    "PyBulletDominoEnv",
    "PyBulletDominoFanEnv",
]
