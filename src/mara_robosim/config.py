"""Frozen dataclass configurations for PyBullet simulation.

Replaces the predicators CFG.pybullet_* global mutable settings with
explicit, immutable configuration objects that are passed through the
call stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

import numpy as np

# Type alias for an orientation quaternion (x, y, z, w).
Quaternion = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Default end-effector orientations
# ---------------------------------------------------------------------------

# Gripper down, parallel to x-axis -- the common default for Fetch/Panda.
DEFAULT_EE_ORNS: Dict[str, Quaternion] = {
    "fetch": (0.5, -0.5, -0.5, -0.5),
    "mobile_fetch": (0.5, -0.5, -0.5, -0.5),
    "panda": (0.7071, 0.7071, 0.0, 0.0),
}

# Per-environment overrides (e.g. blocks/balance use gripper straight down
# because the objects are thin enough that the 90-degree rotation is not
# necessary).
BLOCKS_EE_ORNS: Dict[str, Quaternion] = {
    "fetch": (0.7071, 0.0, -0.7071, 0.0),
    "mobile_fetch": (0.7071, 0.0, -0.7071, 0.0),
    "panda": (0.7071, 0.7071, 0.0, 0.0),
}


def _default_ee_orns() -> Dict[str, Quaternion]:
    """Return a fresh copy of the default end-effector orientations."""
    return dict(DEFAULT_EE_ORNS)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiRRTConfig:
    """Parameters for the Bi-directional RRT motion planner."""

    num_attempts: int = 10
    num_iters: int = 100
    smooth_amt: int = 50
    extend_num_interp: int = 10
    path_subsample_ratio: int = 1


@dataclass(frozen=True)
class IKFastConfig:
    """Parameters for IKFast inverse-kinematics solver."""

    max_time: float = 0.05
    max_candidates: int = 100
    max_attempts: float = np.inf
    max_distance: float = np.inf
    norm: float = np.inf  # ord parameter for np.linalg.norm


@dataclass(frozen=True)
class PyBulletConfig:
    """Base configuration for all PyBullet environments.

    This frozen dataclass replaces the ``CFG.pybullet_*`` global settings
    from *predicators*.  Every field mirrors one of those settings (with
    the ``pybullet_`` prefix stripped).

    Instances are immutable; use ``dataclasses.replace(cfg, field=val)``
    to derive a modified copy.
    """

    # -- Robot ----------------------------------------------------------
    robot: str = "fetch"

    # -- Debug / GUI ----------------------------------------------------
    draw_debug: bool = False
    use_gui: bool = False

    # -- Camera / rendering ---------------------------------------------
    camera_width: int = 335  # for high quality, use 1674
    camera_height: int = 180  # for high quality, use 900
    rgb_observation: bool = False

    # -- Simulation -----------------------------------------------------
    sim_steps_per_action: int = 20

    # -- Inverse kinematics ---------------------------------------------
    max_ik_iters: int = 100
    ik_tol: float = 1e-3
    ik_validate: bool = True

    # -- Control --------------------------------------------------------
    control_mode: str = "position"  # "position" or "reset"
    max_vel_norm: float = 0.05

    # -- End-effector orientations --------------------------------------
    # Mapping from robot name to desired end-effector quaternion.
    # Each environment can supply its own; this is the fallback default.
    ee_orns: Dict[str, Quaternion] = field(default_factory=_default_ee_orns)

    # -- Motion planning (BiRRT) ----------------------------------------
    birrt: BiRRTConfig = field(default_factory=BiRRTConfig)

    # -- IKFast ----------------------------------------------------------
    ikfast: IKFastConfig = field(default_factory=IKFastConfig)

    # -- Task generation ------------------------------------------------
    num_train_tasks: int = 50
    num_test_tasks: int = 50

    # -- Random seed ----------------------------------------------------
    seed: int = 0

    # -- Convenience helpers --------------------------------------------

    def get_ee_orn(self, robot: str | None = None) -> Quaternion:
        """Return the end-effector orientation for the given (or current)
        robot.

        Raises ``KeyError`` if the robot name is not found in
        ``self.ee_orns``.
        """
        robot = robot or self.robot
        return self.ee_orns[robot]
