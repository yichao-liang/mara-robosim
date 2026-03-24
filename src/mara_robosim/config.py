"""Frozen dataclass configurations for PyBullet simulation.

Replaces the predicators CFG.pybullet_* global mutable settings with
explicit, immutable configuration objects that are passed through the
call stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Mapping, Tuple
from typing import Type as TypingType
from typing import TypeVar

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

    @classmethod
    def _upgrade(cls, config: PyBulletConfig) -> PyBulletConfig:
        """Upgrade a base PyBulletConfig to this subclass.

        Copies all base fields from ``config`` and uses defaults for any
        subclass-specific fields.  If ``config`` is already the correct
        type, it is returned unchanged.
        """
        if isinstance(config, cls):
            return config
        base_field_names = {f.name for f in fields(PyBulletConfig)}
        base_kwargs = {
            f.name: getattr(config, f.name)
            for f in fields(config)
            if f.name in base_field_names
        }
        return cls(**base_kwargs)

    def get_ee_orn(self, robot: str | None = None) -> Quaternion:
        """Return the end-effector orientation for the given (or current)
        robot.

        Raises ``KeyError`` if the robot name is not found in
        ``self.ee_orns``.
        """
        robot = robot or self.robot
        return self.ee_orns[robot]


# ---------------------------------------------------------------------------
# Per-domain configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AntsConfig(PyBulletConfig):
    """Configuration for the Ants environment."""

    ants_attracted_to_points: bool = False


@dataclass(frozen=True)
class BalanceConfig(PyBulletConfig):
    """Configuration for the Balance environment."""

    block_size: float = 0.045
    num_blocks_train: Tuple[int, ...] = (2, 4)
    num_blocks_test: Tuple[int, ...] = (4, 6)
    holding_goals: bool = False
    weird_balance: bool = False


def _blocks_ee_orns() -> Dict[str, Quaternion]:
    return dict(BLOCKS_EE_ORNS)


@dataclass(frozen=True)
class BlocksConfig(PyBulletConfig):
    """Configuration for the Blocks environment."""

    block_size: float = 0.045
    num_blocks_train: Tuple[int, ...] = (3, 4)
    num_blocks_test: Tuple[int, ...] = (5, 6)
    holding_goals: bool = False
    high_towers_are_unstable: bool = False
    ee_orns: Dict[str, Quaternion] = field(default_factory=_blocks_ee_orns)


@dataclass(frozen=True)
class BoilConfig(PyBulletConfig):
    """Configuration for the Boil environment."""

    boil_goal: str = "simple"
    boil_goal_simple_human_happy: bool = False
    boil_use_derived_predicates: bool = True
    boil_require_jug_full_to_heatup: bool = False
    boil_goal_require_burner_off: bool = True
    boil_add_jug_reached_capacity_predicate: bool = False
    boil_num_jugs_train: Tuple[int, ...] = (1,)
    boil_num_jugs_test: Tuple[int, ...] = (1, 2)
    boil_num_burner_train: Tuple[int, ...] = (1,)
    boil_num_burner_test: Tuple[int, ...] = (1,)
    boil_water_fill_speed: float = 0.002
    boil_use_skill_factories: bool = True
    boil_use_constant_delay: bool = False
    boil_use_normal_delay: bool = True
    boil_use_cmp_delay: bool = False


@dataclass(frozen=True)
class CircuitConfig(PyBulletConfig):
    """Configuration for the Circuit environment."""

    circuit_light_doesnt_need_battery: bool = False
    circuit_battery_in_box: bool = False


@dataclass(frozen=True)
class CoffeeConfig(PyBulletConfig):
    """Configuration for the Coffee environment."""

    num_cups_train: Tuple[int, ...] = (1, 2)
    num_cups_test: Tuple[int, ...] = (2, 3)
    rotated_jug_ratio: float = 0.5
    use_pixelated_jug: bool = False
    jug_pickable_pred: bool = False
    simple_tasks: bool = False
    machine_have_light_bar: bool = True
    machine_has_plug: bool = False
    plug_break_after_plugged_in: bool = False
    fill_jug_gradually: bool = False
    render_grid_world: bool = False


@dataclass(frozen=True)
class CoverConfig(PyBulletConfig):
    """Configuration for the Cover environment."""

    cover_num_blocks: int = 2
    cover_num_targets: int = 2
    cover_block_widths: Tuple[float, ...] = (0.1, 0.07)
    cover_target_widths: Tuple[float, ...] = (0.05, 0.03)
    cover_initial_holding_prob: float = 0.75
    cover_blocks_change_color_when_cover: bool = False
    ee_orns: Dict[str, Quaternion] = field(default_factory=_blocks_ee_orns)


@dataclass(frozen=True)
class DominoConfig(PyBulletConfig):
    """Configuration for the Domino environment."""

    domino_debug_layout: bool = False
    domino_some_dominoes_are_connected: bool = False
    domino_initialize_at_finished_state: bool = True
    domino_use_domino_blocks_as_target: bool = True
    domino_use_grid: bool = False
    domino_include_connected_predicate: bool = False
    domino_has_glued_dominos: bool = True
    domino_prune_actions: bool = False
    domino_only_straight_sequence_in_training: bool = True
    domino_train_num_dominos: Tuple[int, ...] = (2,)
    domino_test_num_dominos: Tuple[int, ...] = (3,)
    domino_train_num_targets: Tuple[int, ...] = (1,)
    domino_test_num_targets: Tuple[int, ...] = (1, 2)
    domino_train_num_pivots: Tuple[int, ...] = (0,)
    domino_test_num_pivots: Tuple[int, ...] = (0,)
    domino_oracle_knows_glued_dominos: bool = False
    domino_use_continuous_place: bool = False
    domino_restricted_push: bool = False
    domino_use_skill_factories: bool = True


@dataclass(frozen=True)
class FanConfig(PyBulletConfig):
    """Configuration for the Fan environment."""

    fan_use_skill_factories: bool = True
    fan_fans_blow_opposite_direction: bool = False
    fan_known_controls_relation: bool = True
    fan_combine_switch_on_off: bool = False
    fan_use_kinematic: bool = False


@dataclass(frozen=True)
class FloatConfig(PyBulletConfig):
    """Configuration for the Float environment."""

    water_level_doesnt_raise: bool = False


@dataclass(frozen=True)
class GrowConfig(PyBulletConfig):
    """Configuration for the Grow environment."""

    grow_use_skill_factories: bool = True
    grow_num_cups_train: Tuple[int, ...] = (2,)
    grow_num_cups_test: Tuple[int, ...] = (2, 3)
    grow_num_jugs_train: Tuple[int, ...] = (2,)
    grow_num_jugs_test: Tuple[int, ...] = (2,)


@dataclass(frozen=True)
class LaserConfig(PyBulletConfig):
    """Configuration for the Laser environment."""

    laser_zero_reflection_angle: bool = False
    laser_use_debug_line_for_beams: bool = False
