"""PyBullet Coffee environment.

A robot must plug in a coffee machine, place a jug, brew coffee, then pour
into cups. Merges domain logic (types, predicates, task generation) from the
original ``CoffeeEnv`` with the PyBullet-specific simulation from
``PyBulletCoffeeEnv``.

x: cup <-> jug,
y: robot <-> machine
z: up <-> down
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from mara_robosim import utils
from mara_robosim.config import PyBulletConfig
from mara_robosim.envs.base_env import PyBulletEnv
from mara_robosim.pybullet_helpers.geometry import Pose3D, Quaternion
from mara_robosim.pybullet_helpers.objects import (
    create_object,
    sample_collision_free_2d_positions,
    update_object,
)
from mara_robosim.pybullet_helpers.robots import SingleArmPyBulletRobot
from mara_robosim.structs import (
    Action,
    EnvironmentTask,
    GroundAtom,
    Object,
    Predicate,
    State,
    Type,
)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class PyBulletCoffeeEnv(PyBulletEnv):
    """PyBullet Coffee domain.

    A robot must brew coffee and pour it into cups. Optionally, the
    coffee machine must be plugged in first.
    """

    # -----------------------------------------------------------------
    # Tolerances  (tighter than abstract CoffeeEnv for 3D physics)
    # -----------------------------------------------------------------
    grasp_finger_tol: ClassVar[float] = 1e-2
    grasp_position_tol: ClassVar[float] = 1e-2
    _finger_action_tol: ClassVar[float] = 1e-3
    dispense_tol: ClassVar[float] = 1e-2
    plugged_in_tol: ClassVar[float] = 1e-2
    pour_angle_tol: ClassVar[float] = 1e-1
    pour_pos_tol_factor: ClassVar[float] = 1.8
    pour_pos_tol: ClassVar[float] = 0.005 * pour_pos_tol_factor
    init_padding: ClassVar[float] = 0.05
    pick_jug_y_padding: ClassVar[float] = 0.05
    pick_jug_rot_tol: ClassVar[float] = 0.1
    safe_z_tol: ClassVar[float] = 1e-2
    place_jug_in_machine_tol: ClassVar[float] = 1e-3 / 2
    jug_twist_offset: ClassVar[float] = 0.025

    # -----------------------------------------------------------------
    # Table / workspace config
    # -----------------------------------------------------------------
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = tuple(
        p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2])
    )  # type: ignore[assignment]

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2

    # -----------------------------------------------------------------
    # Robot settings
    # -----------------------------------------------------------------
    robot_init_x: ClassVar[float] = (x_ub + x_lb) / 2.0
    robot_init_y: ClassVar[float] = (y_ub + y_lb) / 2.0
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = tuple(
        p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2])
    )  # type: ignore[assignment]
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2
    tilt_lb: ClassVar[float] = robot_init_tilt
    tilt_ub: ClassVar[float] = tilt_lb - np.pi / 4
    wrist_lb: ClassVar[float] = -np.pi
    wrist_ub: ClassVar[float] = np.pi
    max_position_vel: ClassVar[float] = 2.5
    max_angular_vel: ClassVar[float] = tilt_ub
    max_finger_vel: ClassVar[float] = 1.0

    # -----------------------------------------------------------------
    # Machine settings
    # -----------------------------------------------------------------
    machine_x_len: ClassVar[float] = 0.2 * (x_ub - x_lb)
    machine_y_len: ClassVar[float] = 0.15 * (y_ub - y_lb)
    machine_z_len: ClassVar[float] = 0.5 * (z_ub - z_lb)
    machine_top_y_len: ClassVar[float] = machine_y_len
    machine_x: ClassVar[float] = x_ub - machine_x_len / 2 - init_padding
    machine_y: ClassVar[float] = y_ub - machine_y_len / 2 - init_padding
    button_radius: ClassVar[float] = 0.6 * machine_y_len
    button_height = button_radius / 10
    button_x: ClassVar[float] = machine_x
    button_y: ClassVar[float] = (
        machine_y - machine_y_len / 2 - machine_top_y_len - button_height / 2
    )
    button_z: ClassVar[float] = z_lb + machine_z_len - button_radius
    button_press_threshold: ClassVar[float] = 3e-2
    machine_color: ClassVar[Tuple[float, float, float, float]] = (0.1, 0.1, 0.1, 1.0)
    button_color_on: ClassVar[Tuple[float, float, float, float]] = (0.2, 0.5, 0.2, 1.0)
    plate_color_on: ClassVar[Tuple[float, float, float, float]] = machine_color
    button_color_off: ClassVar[Tuple[float, float, float, float]] = (0.5, 0.2, 0.2, 1.0)
    button_color_power_off: ClassVar[Tuple[float, float, float, float]] = (
        0.25,
        0.25,
        0.25,
        1.0,
    )
    plate_color_off: ClassVar[Tuple[float, float, float, float]] = machine_color

    # -----------------------------------------------------------------
    # Jug settings
    # -----------------------------------------------------------------
    jug_radius: ClassVar[float] = 0.3 * machine_y_len
    jug_old_height: ClassVar[float] = 0.19 * (z_ub - z_lb)
    jug_new_height: ClassVar[float] = 0.12

    @classmethod
    def jug_height(cls) -> float:
        """Use class method to allow for dynamic changes."""
        if cls.use_pixelated_jug:
            return cls.jug_new_height
        return cls.jug_old_height

    jug_init_x_lb: ClassVar[float] = machine_x - machine_x_len / 2 + init_padding
    jug_init_x_ub: ClassVar[float] = machine_x + machine_x_len / 2 - init_padding
    jug_init_y_lb: ClassVar[float] = y_lb + 3 * jug_radius + init_padding + 0.02
    jug_init_y_ub: ClassVar[float] = (
        machine_y - machine_y_len - 4 * jug_radius - init_padding
    )
    jug_init_y_ub_og: ClassVar[float] = (
        machine_y - machine_y_len - 3 * jug_radius - init_padding
    )
    jug_handle_offset: ClassVar[float] = 3 * jug_radius
    jug_old_handle_height: ClassVar[float] = jug_old_height
    jug_new_handle_height: ClassVar[float] = 0.1

    @classmethod
    def jug_handle_height(cls) -> float:
        """Use class method to allow for dynamic changes."""
        if cls.use_pixelated_jug:
            return cls.jug_new_handle_height
        return cls.jug_old_handle_height

    jug_init_rot_lb: ClassVar[float] = -2 * np.pi / 3
    jug_init_rot_ub: ClassVar[float] = 2 * np.pi / 3
    jug_handle_radius: ClassVar[float] = 1e-1  # just for rendering
    jug_pickable_rot: ClassVar[float] = -np.pi / 2
    jug_color: ClassVar[Tuple[float, float, float, float]] = (1, 1, 1, 1)

    # -----------------------------------------------------------------
    # Dispense area settings
    # -----------------------------------------------------------------
    dispense_area_x: ClassVar[float] = machine_x
    dispense_area_y: ClassVar[float] = machine_y - 5 * jug_radius
    dispense_radius = 2 * jug_radius
    dispense_height = 0.0001

    # -----------------------------------------------------------------
    # Cup settings
    # -----------------------------------------------------------------
    cup_radius: ClassVar[float] = jug_radius
    cup_init_x_lb: ClassVar[float] = x_lb + cup_radius + init_padding
    cup_init_x_ub: ClassVar[float] = (
        machine_x - machine_x_len / 2 - cup_radius - init_padding
    )
    cup_init_y_lb: ClassVar[float] = jug_init_y_lb
    cup_init_y_ub: ClassVar[float] = cup_init_y_lb + init_padding
    cup_capacity_lb: ClassVar[float] = 0.075 * (z_ub - z_lb)
    cup_capacity_ub: ClassVar[float] = 0.15 * (z_ub - z_lb)
    cup_target_frac: ClassVar[float] = 0.75
    cup_colors: ClassVar[List[Tuple[float, float, float, float]]] = [
        (244 / 255, 27 / 255, 63 / 255, 1.0),
        (121 / 255, 37 / 255, 117 / 255, 1.0),
        (35 / 255, 100 / 255, 54 / 255, 1.0),
    ]

    # -----------------------------------------------------------------
    # Coffee filling settings
    # -----------------------------------------------------------------
    coffee_machine_fill_speed: ClassVar[float] = 0.03
    max_jug_coffee_capacity: ClassVar[float] = 1.0
    coffee_filled_threshold: ClassVar[float] = 1.0

    # -----------------------------------------------------------------
    # Powercord / Plug settings
    # -----------------------------------------------------------------
    num_cord_links = 10
    cord_link_length = 0.02
    cord_segment_gap = 0.00
    cord_start_x = machine_x - machine_x_len / 2 - 4 * cord_link_length
    cord_start_y = machine_y - machine_y_len
    cord_start_z = z_lb + cord_link_length / 2
    plug_x = (
        cord_start_x
        - (num_cord_links - 1) * cord_link_length
        - cord_segment_gap * (num_cord_links - 1)
    )
    plug_y = cord_start_y
    plug_z = cord_start_z

    # -----------------------------------------------------------------
    # Socket settings
    # -----------------------------------------------------------------
    socket_height: ClassVar[float] = 0.1
    socket_width: ClassVar[float] = 0.05
    socket_depth: ClassVar[float] = 0.01
    socket_x: ClassVar[float] = (x_lb + x_ub) / 2
    socket_y: ClassVar[float] = machine_y
    socket_z: ClassVar[float] = z_lb + socket_height * 2

    # -----------------------------------------------------------------
    # Pour settings
    # -----------------------------------------------------------------
    pour_x_offset: ClassVar[float] = cup_radius
    pour_y_offset: ClassVar[float] = -3 * (cup_radius + jug_radius)

    @classmethod
    def pour_z_offset(cls) -> float:
        """The z offset for pouring liquid into a cup."""
        return 2.5 * (cls.cup_capacity_ub + cls.jug_height() - cls.jug_handle_height())

    pour_velocity: ClassVar[float] = cup_capacity_ub / 10.0

    # -----------------------------------------------------------------
    # Camera parameters
    # -----------------------------------------------------------------
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -38
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # -----------------------------------------------------------------
    # Domain configuration flags (originally global settings)
    # -----------------------------------------------------------------
    num_cups_train: ClassVar[List[int]] = [1, 2]
    num_cups_test: ClassVar[List[int]] = [2, 3]
    rotated_jug_ratio: ClassVar[float] = 0.5
    use_pixelated_jug: ClassVar[bool] = False
    jug_pickable_pred: ClassVar[bool] = False
    simple_tasks: ClassVar[bool] = False
    machine_have_light_bar: ClassVar[bool] = True
    machine_has_plug: ClassVar[bool] = False
    plug_break_after_plugged_in: ClassVar[bool] = False
    fill_jug_gradually: ClassVar[bool] = False
    render_grid_world: ClassVar[bool] = False

    # =================================================================
    # Construction
    # =================================================================

    def __init__(
        self,
        config: Optional[PyBulletConfig] = None,
        use_gui: bool = True,
    ) -> None:
        # Camera overrides based on domain flags
        if self.render_grid_world:
            type(self)._camera_distance = 3
            type(self)._camera_yaw = 90
            type(self)._camera_pitch = 0
            type(self)._camera_target = (0.75, 1.33, 0.3)
        else:
            type(self)._camera_distance = 1.3
            if self.machine_has_plug:
                type(self)._camera_yaw = -60
            else:
                type(self)._camera_yaw = 70
            type(self)._camera_pitch = -38
            type(self)._camera_target = (0.75, 1.25, 0.42)

        # ------- Types -------
        self._table_type = Type("table", [])
        self._robot_type = Type("robot", ["x", "y", "z", "tilt", "wrist", "fingers"])
        if self.fill_jug_gradually:
            self._jug_type = Type(
                "jug", ["x", "y", "z", "rot", "is_held", "current_liquid"]
            )
        else:
            self._jug_type = Type("jug", ["x", "y", "z", "rot", "is_held", "is_filled"])
        self._machine_type = Type("coffee_machine", ["is_on"])
        self._cup_type = Type(
            "cup",
            [
                "x",
                "y",
                "z",
                "capacity_liquid",
                "target_liquid",
                "current_liquid",
            ],
        )
        self._plug_type = Type("plug", ["x", "y", "z", "plugged_in"])

        # ------- Static objects -------
        self._robot = Object("robby", self._robot_type)
        self._jug = Object("jug", self._jug_type)
        self._machine = Object("coffee_machine", self._machine_type)
        self._table = Object("table", self._table_type)

        max_cups = max(max(self.num_cups_train), max(self.num_cups_test))
        self._cups: List[Object] = [
            Object(f"cup{i}", self._cup_type) for i in range(max_cups)
        ]
        if self.machine_has_plug:
            self._plug = Object("plug", self._plug_type)

        # ------- Predicates -------
        self._PluggedIn = Predicate(
            "PluggedIn", [self._plug_type], self._PluggedIn_holds
        )
        self._CupFilled = Predicate(
            "CupFilled", [self._cup_type], self._CupFilled_holds
        )
        self._Holding = Predicate(
            "Holding", [self._robot_type, self._jug_type], self._Holding_holds
        )
        self._JugInMachine = Predicate(
            "JugInMachine",
            [self._jug_type, self._machine_type],
            self._JugInMachine_holds,
        )
        self._MachineOn = Predicate(
            "MachineOn", [self._machine_type], self._MachineOn_holds
        )
        self._OnTable = Predicate("OnTable", [self._jug_type], self._OnTable_holds)
        self._HandEmpty = Predicate(
            "HandEmpty", [self._robot_type], self._HandEmpty_holds
        )
        self._JugFilled = Predicate(
            "JugFilled", [self._jug_type], self._JugFilled_holds
        )
        self._RobotAboveCup = Predicate(
            "RobotAboveCup",
            [self._robot_type, self._cup_type],
            self._RobotAboveCup_holds,
        )
        self._JugAboveCup = Predicate(
            "JugAboveCup", [self._jug_type, self._cup_type], self._JugAboveCup_holds
        )
        self._NotAboveCup = Predicate(
            "NotAboveCup", [self._robot_type, self._jug_type], self._NotAboveCup_holds
        )
        self._Twisting = Predicate(
            "Twisting", [self._robot_type, self._jug_type], self._Twisting_holds
        )
        self._PressingButton = Predicate(
            "PressingButton",
            [self._robot_type, self._machine_type],
            self._PressingButton_holds,
        )
        self._NotSameCup = Predicate(
            "NotSameCup", [self._cup_type, self._cup_type], self._NotSameCup_holds
        )
        self._HandTilted = Predicate(
            "HandTilted", [self._robot_type], self._HandTilted_holds
        )
        self._JugPickable = Predicate(
            "JugPickable", [self._jug_type], self._JugPickable_holds
        )

        # ------- Call base __init__ (creates pybullet world) -------
        super().__init__(config, use_gui)

        # ------- PyBullet-specific bookkeeping -------
        self._cup_to_liquid_id: Dict[Object, Optional[int]] = {}
        self._cup_to_capacity: Dict[Object, float] = {}
        self._jug_filled = False
        self._jug_current_liquid = 0.0
        self._jug_liquid_id: Optional[int] = None
        self._cord_ids: Optional[List[int]] = None
        self._machine_plugged_in_id: Optional[int] = None

    # =================================================================
    # Name / metadata
    # =================================================================

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_coffee"

    @property
    def predicates(self) -> Set[Predicate]:
        preds = {
            self._CupFilled,
            self._JugInMachine,
            self._Holding,
            self._MachineOn,
            self._OnTable,
            self._HandEmpty,
            self._JugFilled,
            self._RobotAboveCup,
            self._JugAboveCup,
            self._NotAboveCup,
            self._PressingButton,
            self._Twisting,
            self._NotSameCup,
            self._HandTilted,
        }
        if self.jug_pickable_pred:
            preds.add(self._JugPickable)
        if self.machine_has_plug:
            preds.add(self._PluggedIn)
        return preds

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._CupFilled}

    @property
    def types(self) -> Set[Type]:
        return {
            self._cup_type,
            self._jug_type,
            self._machine_type,
            self._robot_type,
            self._plug_type,
        }

    # =================================================================
    # PyBullet Initialisation
    # =================================================================

    @classmethod
    def initialize_pybullet(
        cls,
        using_gui: bool,
        config: Optional[PyBulletConfig] = None,
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then create coffee-specific static bodies."""
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui, config=config
        )

        cls._add_pybullet_debug_lines(physics_client_id)

        table_id = cls._add_pybullet_table(physics_client_id)
        bodies["table_id"] = table_id

        machine_id = cls._add_pybullet_coffee_machine(physics_client_id)
        dispense_area_id = cls._add_pybullet_dispense_area(physics_client_id)
        button_id = cls._add_pybullet_machine_button(physics_client_id)
        bodies["machine_id"] = machine_id
        bodies["dispense_area_id"] = dispense_area_id
        bodies["button_id"] = button_id

        jug_id = cls._add_pybullet_jug(physics_client_id)
        bodies["jug_id"] = jug_id

        if cls.machine_has_plug:
            socket_id = cls._add_pybullet_socket(physics_client_id)
            bodies["socket_id"] = socket_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_ids = [pybullet_bodies["table_id"]]
        self._table.id = pybullet_bodies["table_id"]
        self._jug.id = pybullet_bodies["jug_id"]
        self._machine.id = pybullet_bodies["machine_id"]
        self._robot.id = self._pybullet_robot.robot_id
        self._dispense_area_id = pybullet_bodies["dispense_area_id"]
        self._button_id = pybullet_bodies["button_id"]
        if self.machine_has_plug:
            self._socket_id = pybullet_bodies["socket_id"]

    def _get_object_ids_for_held_check(self) -> List[int]:
        if self.machine_has_plug:
            assert self._plug.id is not None
            return [self._jug.id, self._plug.id]
        return [self._jug.id]

    # =================================================================
    # Task-specific object creation / teardown
    # =================================================================

    def _create_task_specific_objects(self, state: State) -> None:
        """Remove/rebuild cups, liquids, and cords so each new task can have
        different cups and states."""
        self._remake_jug_liquid(state)
        self._remake_cup_liquids(state)
        self._remake_cups(state)
        self._remake_cord()

    def _remake_cups(self, state: State) -> None:
        """Re-load cup URDFs with appropriate scaling and color."""
        for cup in self._cups:
            if cup.id is not None:
                p.removeBody(cup.id, physicsClientId=self._physics_client_id)

        cup_objs = state.get_objects(self._cup_type)
        self._cup_to_capacity.clear()
        for i, cup_obj in enumerate(cup_objs):
            cup_cap = state.get(cup_obj, "capacity_liquid")
            global_scale = 0.5 * cup_cap / self.cup_capacity_ub
            color = self._obj_colors[self._train_rng.choice(len(self._obj_colors))]
            if self.use_pixelated_jug:
                file = "urdf/pot-pixel.urdf"
                global_scale *= 0.5
            else:
                file = "urdf/cup.urdf"
            cup_id = create_object(
                file,
                color=color,
                scale=global_scale,
                use_fixed_base=True,
                physics_client_id=self._physics_client_id,
            )
            self._cup_to_capacity[cup_obj] = cup_cap
            cup_obj.id = cup_id

    def _remake_cup_liquids(self, state: State) -> None:
        """Re-create the visual liquid objects for the new cups."""
        for liquid_id in self._cup_to_liquid_id.values():
            if liquid_id is not None:
                p.removeBody(liquid_id, physicsClientId=self._physics_client_id)
        self._cup_to_liquid_id.clear()

        cup_objs = state.get_objects(self._cup_type)
        for cup in cup_objs:
            new_liquid_id = self._create_liquid_for_cup(cup, state)
            self._cup_to_liquid_id[cup] = new_liquid_id

    def _remake_jug_liquid(self, state: State) -> None:
        """Check jug's fill status and re-create liquid object if needed."""
        if self.fill_jug_gradually:
            self._jug_current_liquid = state.get(self._jug, "current_liquid")
            self._jug_filled = bool(
                self._jug_current_liquid >= self.coffee_filled_threshold
            )
        else:
            self._jug_filled = bool(state.get(self._jug, "is_filled") > 0.5)
        if self._jug_liquid_id is not None:
            p.removeBody(self._jug_liquid_id, physicsClientId=self._physics_client_id)
            self._jug_liquid_id = None
        if self._jug_filled:
            self._jug_liquid_id = self._create_liquid_for_jug()

    def _remake_cord(self) -> None:
        """If the machine uses a plug, rebuild the cord bodies and
        constraints."""
        if self.machine_has_plug:
            if self._cord_ids is not None:
                for part_id in self._cord_ids:
                    p.removeBody(part_id, physicsClientId=self._physics_client_id)
            if self._machine_plugged_in_id is not None:
                p.removeConstraint(
                    self._machine_plugged_in_id, physicsClientId=self._physics_client_id
                )
                self._machine_plugged_in_id = None
            self._cord_ids, self._cord_constraints = self._add_pybullet_cord(
                self._physics_client_id
            )
            self._plug.id = self._cord_ids[-1]

    # =================================================================
    # State reset
    # =================================================================

    def _reset_custom_env_state(self, state: State) -> None:
        """Handles extra coffee-specific reset steps: button color, etc."""
        if self._MachineOn_holds(state, [self._machine]) and self._JugInMachine_holds(
            state, [self._jug, self._machine]
        ):
            button_color = self.button_color_on
            plate_color = self.plate_color_on
        else:
            if self.machine_has_plug and self._PluggedIn_holds(state, [self._plug]):
                button_color = self.button_color_off
                plate_color = self.plate_color_off
            else:
                button_color = self.button_color_power_off
                plate_color = self.plate_color_off

        p.changeVisualShape(
            self._button_id,
            -1,
            rgbaColor=button_color,
            physicsClientId=self._physics_client_id,
        )
        p.changeVisualShape(
            self._button_id,
            0,
            rgbaColor=button_color,
            physicsClientId=self._physics_client_id,
        )
        p.changeVisualShape(
            self._dispense_area_id,
            -1,
            rgbaColor=plate_color,
            physicsClientId=self._physics_client_id,
        )

    # =================================================================
    # Feature extraction
    # =================================================================

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._jug_type:
            if feature == "is_filled":
                return float(self._jug_filled)
            elif feature == "current_liquid":
                return (
                    self._jug_current_liquid
                    if hasattr(self, "_jug_current_liquid")
                    else 0.0
                )
        elif obj.type == self._machine_type:
            if feature == "is_on":
                button_color = p.getVisualShapeData(
                    self._button_id, physicsClientId=self._physics_client_id
                )[0][-1]
                button_color_on_dist = sum(
                    np.subtract(button_color, self.button_color_on) ** 2
                )
                button_color_off_dist = sum(
                    np.subtract(button_color, self.button_color_off) ** 2
                )
                return float(button_color_on_dist < button_color_off_dist)
        elif obj.type == self._cup_type:
            if feature == "capacity_liquid":
                return self._cup_to_capacity[obj]
            elif feature == "current_liquid":
                liquid_id = self._cup_to_liquid_id.get(obj, None)
                if liquid_id is not None:
                    liquid_height = p.getVisualShapeData(
                        liquid_id,
                        physicsClientId=self._physics_client_id,
                    )[0][3][0]
                    return self._cup_liquid_height_to_liquid(
                        liquid_height, self._cup_to_capacity[obj]
                    )
                else:
                    return 0.0
            elif feature == "target_liquid":
                if self.use_pixelated_jug:
                    return self._cup_to_capacity[obj]
                else:
                    return self._cup_to_capacity[obj] * self.cup_target_frac
        elif obj.type == self._plug_type:
            if feature == "plugged_in":
                return float(self._machine_plugged_in_id is not None)
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    # =================================================================
    # Step / simulation
    # =================================================================

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Override to handle plug/pour/twist after the base step."""
        current_ee_rpy = self._pybullet_robot.forward_kinematics(
            self._pybullet_robot.get_joints()
        ).rpy
        state = super().step(action, render_obs=render_obs)

        if self.machine_has_plug:
            self._check_and_apply_plug_in_constraint(state)
        self._handle_machine_on_and_jug_filling(state)
        self._handle_pouring(state)
        self._handle_twisting(state, current_ee_rpy, action)

        self._current_observation = self._get_state(render_obs=False)
        state = self._current_observation.copy()
        return state

    def _update_jug_liquid_position(self) -> None:
        """If the jug is filled, move its liquid to match the jug's pose."""
        if self._jug_filled and self._jug_liquid_id is not None:
            pos, quat = p.getBasePositionAndOrientation(
                self._jug.id, physicsClientId=self._physics_client_id
            )

            if self.fill_jug_gradually:
                if not hasattr(self, "_last_jug_liquid_level"):
                    self._last_jug_liquid_level = self._jug_current_liquid
                if abs(self._jug_current_liquid - self._last_jug_liquid_level) > 0.01:
                    p.removeBody(
                        self._jug_liquid_id, physicsClientId=self._physics_client_id
                    )
                    self._jug_liquid_id = self._create_liquid_for_jug()
                    self._last_jug_liquid_level = self._jug_current_liquid
                    pos, quat = p.getBasePositionAndOrientation(
                        self._jug.id, physicsClientId=self._physics_client_id
                    )

            p.resetBasePositionAndOrientation(
                self._jug_liquid_id, pos, quat, physicsClientId=self._physics_client_id
            )

    def _check_and_apply_plug_in_constraint(self, state: State) -> None:
        """If the plug is 'plugged_in', create a fixed constraint."""
        if (
            self._PluggedIn_holds(state, [self._plug])
            and self._machine_plugged_in_id is None
        ):
            self._machine_plugged_in_id = p.createConstraint(
                parentBodyUniqueId=self._socket_id,
                parentLinkIndex=-1,
                childBodyUniqueId=self._plug.id,
                childLinkIndex=-1,
                jointAxis=[0, 0, 0],
                jointType=p.JOINT_FIXED,
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeVisualShape(
                self._button_id,
                -1,
                rgbaColor=self.button_color_off,
                physicsClientId=self._physics_client_id,
            )
            if self.plug_break_after_plugged_in:
                p.removeConstraint(
                    self._cord_constraints[2], physicsClientId=self._physics_client_id
                )

    def _handle_machine_on_and_jug_filling(self, state: State) -> None:
        """If the robot is pressing the button, turn on the machine and
        fill the jug if it is in the machine."""
        machine_on = state.get(self._machine, "is_on")
        if self._PressingButton_holds(state, [self._robot, self._machine]):
            p.changeVisualShape(
                self._button_id,
                -1,
                rgbaColor=self.button_color_on,
                physicsClientId=self._physics_client_id,
            )
            machine_on = True
        if machine_on:
            if self._JugInMachine_holds(state, [self._jug, self._machine]) and (
                not self.machine_has_plug or self._machine_plugged_in_id is not None
            ):
                if self.fill_jug_gradually:
                    if self._jug_current_liquid < self.max_jug_coffee_capacity:
                        self._jug_current_liquid = min(
                            self.max_jug_coffee_capacity,
                            self._jug_current_liquid + self.coffee_machine_fill_speed,
                        )
                        self._jug_liquid_id = self._create_liquid_for_jug()
                        if (
                            not self._jug_filled
                            and self._jug_current_liquid > self.coffee_filled_threshold
                        ):
                            self._jug_filled = True
                else:
                    self._jug_current_liquid = min(
                        self.max_jug_coffee_capacity,
                        self._jug_current_liquid + self.coffee_machine_fill_speed,
                    )
                    self._jug_liquid_id = self._create_liquid_for_jug()
                    if (
                        not self._jug_filled
                        and self._jug_current_liquid > self.coffee_filled_threshold
                    ):
                        self._jug_filled = True

    def _handle_pouring(self, state: State) -> None:
        """If the robot is tilted sufficiently to pour, increase liquid in
        the appropriate cup."""
        if abs(state.get(self._robot, "tilt") - self.tilt_ub) < self.pour_angle_tol:
            if not self._JugFilled_holds(state, [self._jug]):
                return
            cup = self._get_cup_to_pour(state)
            if cup is None:
                return
            current_liquid = state.get(cup, "current_liquid")
            cup_cap = state.get(cup, "capacity_liquid")
            new_liquid = min(current_liquid + self.pour_velocity, cup_cap + 0.01)
            state.set(cup, "current_liquid", new_liquid)

            old_liquid_id = self._cup_to_liquid_id.get(cup, None)
            if old_liquid_id is not None:
                p.removeBody(old_liquid_id, physicsClientId=self._physics_client_id)
            self._cup_to_liquid_id[cup] = self._create_liquid_for_cup(cup, state)

    def _handle_twisting(
        self, state: State, current_ee_rpy: Tuple[float, float, float], action: Action
    ) -> None:
        """If the robot is twisting the jug, update the jug's yaw."""
        if self._Twisting_holds(state, [self._robot, self._jug]):
            gripper_pose = self._pybullet_robot.forward_kinematics(action.arr.tolist())
            d_roll = gripper_pose.rpy[0] - current_ee_rpy[0]
            d_yaw = gripper_pose.rpy[2] - current_ee_rpy[2]

            if abs(d_yaw) > 0.2:
                if d_yaw < 0:
                    d_roll -= np.pi
                else:
                    d_roll += np.pi

            (jx, jy, _), jug_quat = p.getBasePositionAndOrientation(
                self._jug.id, physicsClientId=self._physics_client_id
            )
            jug_yaw = p.getEulerFromQuaternion(jug_quat)[2]
            new_jug_yaw = _wrap_angle(jug_yaw - d_roll)
            new_jug_quat = p.getQuaternionFromEuler([0.0, 0.0, new_jug_yaw])
            p.resetBasePositionAndOrientation(
                self._jug.id,
                [jx, jy, self.z_lb + self.jug_height() / 2],
                new_jug_quat,
                physicsClientId=self._physics_client_id,
            )

    # =================================================================
    # Gripper orientation helpers
    # =================================================================

    def _state_to_gripper_orn(self, state: State) -> Quaternion:
        wrist = state.get(self._robot, "wrist")
        tilt = state.get(self._robot, "tilt")
        return self.tilt_wrist_to_gripper_orn(tilt, wrist)

    @classmethod
    def tilt_wrist_to_gripper_orn(cls, tilt: float, wrist: float) -> Quaternion:
        """Public for oracle options."""
        return tuple(p.getQuaternionFromEuler([0.0, tilt, wrist]))  # type: ignore[return-value]

    def _gripper_orn_to_tilt_wrist(self, orn: Quaternion) -> Tuple[float, float]:
        _, tilt, wrist = p.getEulerFromQuaternion(orn)
        return (tilt, wrist)

    # =================================================================
    # Liquid management
    # =================================================================

    def _cup_liquid_to_liquid_height(self, liquid: float, capacity: float) -> float:
        scale = 0.5 * np.sqrt(capacity / self.cup_capacity_ub)
        return liquid * scale

    def _cup_liquid_height_to_liquid(self, height: float, capacity: float) -> float:
        scale = 0.5 * np.sqrt(capacity / self.cup_capacity_ub)
        return height / scale

    def _cup_to_liquid_radius(self, capacity: float) -> float:
        scale = 1.5 * np.sqrt(capacity / self.cup_capacity_ub)
        return self.cup_radius * scale

    def _create_liquid_for_cup(self, cup: Object, state: State) -> Optional[int]:
        current_liquid = state.get(cup, "current_liquid")
        cup_cap = state.get(cup, "capacity_liquid")
        liquid_height = self._cup_liquid_to_liquid_height(current_liquid, cup_cap)
        liquid_radius = self._cup_to_liquid_radius(cup_cap)
        if current_liquid == 0:
            return None
        cx = state.get(cup, "x")
        cy = state.get(cup, "y")
        cz = self.z_lb + current_liquid / 2 + 0.025

        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=liquid_radius,
            height=liquid_height,
            physicsClientId=self._physics_client_id,
        )
        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=liquid_radius,
            length=liquid_height,
            rgbaColor=(0.35, 0.1, 0.0, 1.0),
            physicsClientId=self._physics_client_id,
        )

        pose = (cx, cy, cz)
        orientation = self._default_orn
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=pose,
            baseOrientation=orientation,
            physicsClientId=self._physics_client_id,
        )

    def _create_liquid_for_jug(self) -> int:
        if self.use_pixelated_jug:
            liquid_radius = self.jug_radius * 1.3
        else:
            liquid_radius = self.jug_radius

        if self.fill_jug_gradually:
            liquid_fill_ratio = self._jug_current_liquid / self.max_jug_coffee_capacity
            if self.use_pixelated_jug:
                liquid_height = (self.jug_height() * 0.8) * liquid_fill_ratio
            else:
                liquid_height = (self.jug_height() * 0.6) * liquid_fill_ratio
        else:
            if self.use_pixelated_jug:
                liquid_height = self.jug_height() * 0.8
            else:
                liquid_height = self.jug_height() * 0.6

        if self._jug_liquid_id is not None:
            p.removeBody(self._jug_liquid_id, physicsClientId=self._physics_client_id)
            self._jug_liquid_id = None

        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[liquid_radius, liquid_radius, liquid_height / 2],
            physicsClientId=self._physics_client_id,
        )
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[liquid_radius, liquid_radius, liquid_height / 2],
            rgbaColor=(0.35, 0.1, 0.0, 1.0),
            physicsClientId=self._physics_client_id,
        )

        jug_pos, jug_orientation = p.getBasePositionAndOrientation(
            self._jug.id, physicsClientId=self._physics_client_id
        )
        liquid_pos = (jug_pos[0], jug_pos[1], self.z_lb + liquid_height / 2 + 0.02)

        return p.createMultiBody(
            baseMass=1e-5,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=liquid_pos,
            baseOrientation=jug_orientation,
            physicsClientId=self._physics_client_id,
        )

    # =================================================================
    # PyBullet static body creation (classmethods)
    # =================================================================

    @classmethod
    def _add_pybullet_coffee_machine(cls, physics_client_id: int) -> int:
        """Create the coffee machine body (base + top + dispense base)."""
        half_extents_base = (
            cls.machine_x_len,
            cls.machine_y_len / 2,
            cls.machine_z_len / 2,
        )
        collision_id_base = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents_base, physicsClientId=physics_client_id
        )
        visual_id_base = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_base,
            rgbaColor=cls.machine_color,
            physicsClientId=physics_client_id,
        )
        pose_base = (
            cls.machine_x,
            cls.machine_y,
            cls.z_lb + cls.machine_z_len / 2,
        )
        orientation_base = [0, 0, 0, 1]

        half_extents_top = (
            cls.machine_x_len * 5 / 6,
            cls.machine_top_y_len / 2,
            cls.machine_z_len / 6,
        )
        collision_id_top = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents_top, physicsClientId=physics_client_id
        )
        visual_id_top = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_top,
            rgbaColor=cls.machine_color,
            physicsClientId=physics_client_id,
        )
        pose_top = (
            -cls.machine_x_len / 6,
            -cls.machine_y_len / 2 - cls.machine_top_y_len / 2,
            cls.machine_z_len / 3,
        )
        orientation_top = cls._default_orn

        half_extents_dispense_base = (
            cls.machine_x_len,
            1.1 * cls.dispense_radius + cls.jug_radius + 0.003,
            cls.dispense_height,
        )
        collision_id_dispense_base = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=(0, 0, 0), physicsClientId=physics_client_id
        )
        visual_id_dispense_base = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_dispense_base,
            rgbaColor=cls.machine_color,
            physicsClientId=physics_client_id,
        )
        pose_dispense_base = (
            0,
            -cls.machine_y_len - cls.dispense_radius + 0.01,
            -cls.machine_z_len / 2,
        )
        orientation_dispense_base = cls._default_orn

        link_mass = 0
        link_inertial_frame_position = [0, 0, 0]
        link_inertial_frame_orientation = [0, 0, 0, 1]

        machine_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id_base,
            baseVisualShapeIndex=visual_id_base,
            basePosition=pose_base,
            baseOrientation=orientation_base,
            linkMasses=[link_mass, link_mass],
            linkCollisionShapeIndices=[collision_id_top, collision_id_dispense_base],
            linkVisualShapeIndices=[visual_id_top, visual_id_dispense_base],
            linkPositions=[pose_top, pose_dispense_base],
            linkOrientations=[orientation_top, orientation_dispense_base],
            linkInertialFramePositions=[
                link_inertial_frame_position,
                link_inertial_frame_position,
            ],
            linkInertialFrameOrientations=[
                link_inertial_frame_orientation,
                link_inertial_frame_orientation,
            ],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0], [0, 0, 0]],
            physicsClientId=physics_client_id,
        )
        return machine_id

    @classmethod
    def _add_pybullet_dispense_area(cls, physics_client_id: int) -> int:
        """Create the dispense area circle."""
        pose = (cls.dispense_area_x, cls.dispense_area_y, cls.z_lb)
        orientation = cls._default_orn

        collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=(0, 0, 0), physicsClientId=physics_client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=cls.dispense_radius + 0.8 * cls.jug_radius,
            length=cls.dispense_height,
            rgbaColor=cls.plate_color_off,
            physicsClientId=physics_client_id,
        )
        dispense_area_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=pose,
            baseOrientation=orientation,
            physicsClientId=physics_client_id,
        )
        return dispense_area_id

    @classmethod
    def _add_pybullet_machine_button(cls, physics_client_id: int) -> int:
        """Create the machine button (and optional light bar)."""
        button_position = (cls.button_x, cls.button_y, cls.button_z)
        button_orientation = p.getQuaternionFromEuler([0.0, np.pi / 2, np.pi / 2])

        collision_id_button = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[cls.button_radius, cls.button_radius, cls.button_height / 2],
            physicsClientId=physics_client_id,
        )

        initial_button_color = (
            cls.button_color_power_off if cls.machine_has_plug else cls.button_color_off
        )
        visual_id_button = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cls.button_radius, cls.button_radius, cls.button_height / 2],
            rgbaColor=initial_button_color,
            physicsClientId=physics_client_id,
        )

        if cls.machine_have_light_bar:
            half_extents_bar = (
                cls.machine_z_len / 6 - 0.01,
                cls.machine_x_len * 5 / 6,
                cls.machine_top_y_len / 2,
            )
            collision_id_light_bar = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents_bar,
                physicsClientId=physics_client_id,
            )
            visual_id_light_bar = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents_bar,
                rgbaColor=cls.button_color_off,
                physicsClientId=physics_client_id,
            )

            link_positions = [
                [
                    cls.machine_z_len / 6 - 0.017,
                    cls.machine_x_len / 6 - 0.001,
                    cls.machine_top_y_len / 2 - 0.001,
                ]
            ]
            link_orientations = [[0, 0, 0, 1]]

            button_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_id_button,
                baseVisualShapeIndex=visual_id_button,
                basePosition=button_position,
                baseOrientation=button_orientation,
                linkMasses=[0],
                linkCollisionShapeIndices=[collision_id_light_bar],
                linkVisualShapeIndices=[visual_id_light_bar],
                linkPositions=link_positions,
                linkOrientations=link_orientations,
                linkInertialFramePositions=[[0, 0, 0]],
                linkInertialFrameOrientations=[[0, 0, 0, 1]],
                linkParentIndices=[0],
                linkJointTypes=[p.JOINT_FIXED],
                linkJointAxis=[[0, 0, 0]],
                physicsClientId=physics_client_id,
            )
        else:
            button_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_id_button,
                baseVisualShapeIndex=visual_id_button,
                basePosition=button_position,
                baseOrientation=button_orientation,
                physicsClientId=physics_client_id,
            )
        return button_id

    @classmethod
    def _add_pybullet_jug(cls, physics_client_id: int) -> int:
        """Load the coffee jug URDF."""
        jug_loc = (0, 0, 0)
        jug_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        if cls.use_pixelated_jug:
            jug_id = p.loadURDF(
                utils.get_asset_path("urdf/jug-pixel.urdf"),
                globalScaling=0.2,
                useFixedBase=False,
                physicsClientId=physics_client_id,
            )
        else:
            jug_id = p.loadURDF(
                utils.get_asset_path("urdf/kettle.urdf"),
                globalScaling=0.09,
                useFixedBase=False,
                physicsClientId=physics_client_id,
            )
            p.changeVisualShape(
                jug_id, 0, rgbaColor=cls.jug_color, physicsClientId=physics_client_id
            )
            # Remove the lid.
            p.changeVisualShape(
                jug_id, 1, rgbaColor=[1, 1, 1, 0], physicsClientId=physics_client_id
            )

        p.changeDynamics(
            bodyUniqueId=jug_id,
            linkIndex=-1,
            mass=0.1,
            physicsClientId=physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            jug_id, jug_loc, jug_orientation, physicsClientId=physics_client_id
        )
        return jug_id

    @classmethod
    def _add_pybullet_table(cls, physics_client_id: int) -> int:
        """Load the table URDF."""
        table_id = p.loadURDF(
            utils.get_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            table_id, cls.table_pos, cls.table_orn, physicsClientId=physics_client_id
        )
        return table_id

    @classmethod
    def _add_pybullet_cord(
        cls,
        physics_client_id: int,
    ) -> Tuple[List[int], List[int]]:
        """Create the power cord chain. First segment is machine, last is
        the plug."""
        base_position = [cls.cord_start_x, cls.cord_start_y, cls.cord_start_z]
        segments: List[int] = []
        constraint_ids: List[int] = []

        for i in range(cls.num_cord_links):
            x_pos = base_position[0] - i * (cls.cord_link_length + cls.cord_segment_gap)
            y_pos = base_position[1]
            z_pos = base_position[2]
            link_pos = [x_pos, y_pos, z_pos]

            if i == 0:
                color = [0.0, 0.0, 0.0, 1.0]
            elif i == cls.num_cord_links - 1:
                color = [1.0, 0.0, 0.0, 1.0]
            else:
                color = [0.5, 0.0, 0.0, 1.0]

            if i == cls.num_cord_links - 1:
                col_x = cls.cord_link_length / 2
                col_y = cls.cord_link_length / 2
                col_z = cls.cord_link_length / 2
            else:
                col_x = cls.cord_link_length / 6
                col_y = cls.cord_link_length / 6
                col_z = cls.cord_link_length / 6

            segment = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[col_x, col_y, col_z],
                physicsClientId=physics_client_id,
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    cls.cord_link_length / 2,
                    cls.cord_link_length / 2,
                    cls.cord_link_length / 2,
                ],
                rgbaColor=color,
                physicsClientId=physics_client_id,
            )
            base_mass = 0 if i == 0 else 0.001
            segment_id = p.createMultiBody(
                baseMass=base_mass,
                baseCollisionShapeIndex=segment,
                baseVisualShapeIndex=visual_shape,
                basePosition=link_pos,
                physicsClientId=physics_client_id,
            )
            segments.append(segment_id)

        half_gap = cls.cord_segment_gap / 2
        for i in range(len(segments) - 1):
            constraint_id = p.createConstraint(
                parentBodyUniqueId=segments[i],
                parentLinkIndex=-1,
                childBodyUniqueId=segments[i + 1],
                childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=[-cls.cord_link_length / 2 - half_gap, 0, 0],
                childFramePosition=[cls.cord_link_length / 2 + half_gap, 0, 0],
            )
            constraint_ids.append(constraint_id)

        return segments, constraint_ids

    @classmethod
    def _add_pybullet_socket(cls, physics_client_id: int) -> int:
        """Create the wall socket for the plug."""
        socket_position = [cls.socket_x, cls.socket_y, cls.socket_z]
        socket_collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                cls.socket_width / 2,
                cls.socket_depth / 2,
                cls.socket_height / 2,
            ],
            physicsClientId=physics_client_id,
        )
        socket_visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                cls.socket_width / 2,
                cls.socket_depth / 2,
                cls.socket_height / 2,
            ],
            rgbaColor=[0, 0, 1, 1],
            physicsClientId=physics_client_id,
        )
        socket_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=socket_collision_shape,
            baseVisualShapeIndex=socket_visual_shape,
            basePosition=socket_position,
            physicsClientId=physics_client_id,
        )
        return socket_id

    @classmethod
    def _add_pybullet_debug_lines(cls, physics_client_id: int) -> None:
        """Draw workspace boundaries and sampling regions for debugging."""
        for z in [cls.z_lb, cls.z_ub]:
            p.addUserDebugLine(
                [cls.x_lb, cls.y_lb, z],
                [cls.x_ub, cls.y_lb, z],
                [1.0, 0.0, 0.0],
                lineWidth=5.0,
                physicsClientId=physics_client_id,
            )
            p.addUserDebugLine(
                [cls.x_lb, cls.y_ub, z],
                [cls.x_ub, cls.y_ub, z],
                [1.0, 0.0, 0.0],
                lineWidth=5.0,
                physicsClientId=physics_client_id,
            )
            p.addUserDebugLine(
                [cls.x_lb, cls.y_lb, z],
                [cls.x_lb, cls.y_ub, z],
                [1.0, 0.0, 0.0],
                lineWidth=5.0,
                physicsClientId=physics_client_id,
            )
            p.addUserDebugLine(
                [cls.x_ub, cls.y_lb, z],
                [cls.x_ub, cls.y_ub, z],
                [1.0, 0.0, 0.0],
                lineWidth=5.0,
                physicsClientId=physics_client_id,
            )

        # Jug sampling region.
        p.addUserDebugLine(
            [cls.jug_init_x_lb, cls.jug_init_y_lb, cls.z_lb],
            [cls.jug_init_x_ub, cls.jug_init_y_lb, cls.z_lb],
            [0.0, 0.0, 1.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugLine(
            [cls.jug_init_x_lb, cls.jug_init_y_ub, cls.z_lb],
            [cls.jug_init_x_ub, cls.jug_init_y_ub, cls.z_lb],
            [0.0, 0.0, 1.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugLine(
            [cls.jug_init_x_lb, cls.jug_init_y_lb, cls.z_lb],
            [cls.jug_init_x_lb, cls.jug_init_y_ub, cls.z_lb],
            [0.0, 0.0, 1.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugLine(
            [cls.jug_init_x_ub, cls.jug_init_y_lb, cls.z_lb],
            [cls.jug_init_x_ub, cls.jug_init_y_ub, cls.z_lb],
            [0.0, 0.0, 1.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )

        # Cup sampling region.
        p.addUserDebugLine(
            [cls.cup_init_x_lb, cls.cup_init_y_lb, cls.z_lb],
            [cls.cup_init_x_ub, cls.cup_init_y_lb, cls.z_lb],
            [0.0, 0.0, 1.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugLine(
            [cls.cup_init_x_lb, cls.cup_init_y_ub, cls.z_lb],
            [cls.cup_init_x_ub, cls.cup_init_y_ub, cls.z_lb],
            [0.0, 0.0, 1.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugLine(
            [cls.cup_init_x_lb, cls.cup_init_y_lb, cls.z_lb],
            [cls.cup_init_x_lb, cls.cup_init_y_ub, cls.z_lb],
            [0.0, 0.0, 1.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugLine(
            [cls.cup_init_x_ub, cls.cup_init_y_lb, cls.z_lb],
            [cls.cup_init_x_ub, cls.cup_init_y_ub, cls.z_lb],
            [0.0, 0.0, 1.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )

        # Coordinate frame labels.
        p.addUserDebugLine(
            [0, 0, 0],
            [0.25, 0, 0],
            [1.0, 0.0, 0.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugText(
            "x", [0.25, 0, 0], [0.0, 0.0, 0.0], physicsClientId=physics_client_id
        )
        p.addUserDebugLine(
            [0, 0, 0],
            [0.0, 0.25, 0],
            [1.0, 0.0, 0.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugText(
            "y", [0, 0.25, 0], [0.0, 0.0, 0.0], physicsClientId=physics_client_id
        )
        p.addUserDebugLine(
            [0, 0, 0],
            [0.0, 0, 0.25],
            [1.0, 0.0, 0.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugText(
            "z", [0, 0, 0.25], [0.0, 0.0, 0.0], physicsClientId=physics_client_id
        )

    # =================================================================
    # Predicates (merged from CoffeeEnv + PyBulletCoffeeEnv overrides)
    # =================================================================

    @staticmethod
    def _CupFilled_holds(state: State, objects: Sequence[Object]) -> bool:
        (cup,) = objects
        current = state.get(cup, "current_liquid")
        target = state.get(cup, "target_liquid")
        return current >= target

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, jug = objects
        return state.get(jug, "is_held") > 0.5

    def _PluggedIn_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (plug,) = objects
        plug_x = state.get(plug, "x")
        plug_y = state.get(plug, "y")
        plug_z = state.get(plug, "z")
        sq_dist = np.sum(
            np.subtract(
                (plug_x, plug_y, plug_z), (self.socket_x, self.socket_y, self.socket_z)
            )
            ** 2
        )
        return bool(sq_dist < self.plugged_in_tol)

    def _JugInMachine_holds(self, state: State, objects: Sequence[Object]) -> bool:
        jug, _ = objects
        if self._Holding_holds(state, [self._robot, jug]):
            return False
        dispense_pos = (self.dispense_area_x, self.dispense_area_y, self.z_lb)
        x = state.get(jug, "x")
        y = state.get(jug, "y")
        z = self._get_jug_z(state, jug)
        jug_pos = (x, y, z)
        sq_dist_to_dispense = np.sum(np.subtract(dispense_pos, jug_pos) ** 2)
        return sq_dist_to_dispense < self.dispense_tol

    @staticmethod
    def _MachineOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (machine,) = objects
        return state.get(machine, "is_on") > 0.5

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (jug,) = objects
        if self._Holding_holds(state, [self._robot, jug]):
            return False
        return not self._JugInMachine_holds(state, [jug, self._machine])

    def _Twisting_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The robot gripper is in the twisting position."""
        robot, jug = objects
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        z = state.get(robot, "z")
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        jug_top = (jug_x, jug_y, self.jug_height())
        handle_pos = self._get_jug_handle_grasp(state, jug)
        sq_dist_to_handle = np.sum(np.subtract(handle_pos, (x, y, z)) ** 2)
        sq_dist_to_jug_top = np.sum(np.subtract(jug_top, (x, y, z)) ** 2)
        if sq_dist_to_handle < sq_dist_to_jug_top:
            return False
        return sq_dist_to_jug_top < self.grasp_position_tol

    def _HandEmpty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (robot,) = objects
        if self._Twisting_holds(state, [robot, self._jug]):
            return False
        return not self._Holding_holds(state, [robot, self._jug])

    def _HandTilted_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (robot,) = objects
        tilt = np.abs(state.get(robot, "tilt") - self.tilt_ub)
        return tilt < 0.1

    def _JugFilled_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (jug,) = objects
        if self.fill_jug_gradually:
            return state.get(jug, "current_liquid") >= self.coffee_filled_threshold
        else:
            return state.get(jug, "is_filled") > 0.5

    def _RobotAboveCup_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, cup = objects
        assert robot == self._robot
        return self._robot_jug_above_cup(state, cup)

    def _JugAboveCup_holds(self, state: State, objects: Sequence[Object]) -> bool:
        jug, cup = objects
        if not self._Holding_holds(state, [self._robot, jug]):
            return False
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        jug_z = state.get(self._robot, "z") - self.jug_handle_height()
        jug_pos = (jug_x, jug_y, jug_z)

        closest_cup = None
        closest_cup_dist = float("inf")
        for cup_target in state.get_objects(self._cup_type):
            pour_pos = self._get_pour_position(state, cup_target)
            sq_dist_to_pour = np.sum(np.subtract(jug_pos, pour_pos) ** 2)
            if (
                sq_dist_to_pour < self.pour_pos_tol
                and sq_dist_to_pour < closest_cup_dist
            ):
                closest_cup = cup_target
                closest_cup_dist = sq_dist_to_pour
        if closest_cup is None or closest_cup != cup:
            return False
        return True

    def _NotAboveCup_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, jug = objects
        assert robot == self._robot
        assert jug == self._jug
        for cup in state.get_objects(self._cup_type):
            if self._robot_jug_above_cup(state, cup):
                return False
        return True

    def _PressingButton_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, _ = objects
        button_pos = (self.button_x, self.button_y, self.button_z)
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        z = state.get(robot, "z")
        dist_to_button = np.sqrt(np.sum(np.subtract(button_pos, (x, y, z)) ** 2))
        return dist_to_button < self.button_press_threshold

    @staticmethod
    def _NotSameCup_holds(state: State, objects: Sequence[Object]) -> bool:
        del state  # unused
        cup1, cup2 = objects
        return cup1 != cup2

    def _JugPickable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (jug,) = objects
        jug_rot = state.get(jug, "rot")
        return abs(jug_rot - self.jug_pickable_rot) <= self.pick_jug_rot_tol

    # =================================================================
    # Domain helper methods
    # =================================================================

    def _robot_jug_above_cup(self, state: State, cup: Object) -> bool:
        if not self._Holding_holds(state, [self._robot, self._jug]):
            return False
        jug_x = state.get(self._jug, "x")
        jug_y = state.get(self._jug, "y")
        jug_z = state.get(self._robot, "z") - self.jug_handle_height()
        jug_pos = (jug_x, jug_y, jug_z)
        pour_pos = self._get_pour_position(state, cup)
        sq_dist_to_pour = np.sum(np.subtract(jug_pos, pour_pos) ** 2)
        return sq_dist_to_pour < self.pour_pos_tol

    @classmethod
    def _get_jug_handle_grasp(
        cls,
        state: State,
        jug: Object,
    ) -> Tuple[float, float, float]:
        """Get the grasp position for the jug handle."""
        rot = state.get(jug, "rot")
        target_x = state.get(jug, "x") + np.cos(rot) * cls.jug_handle_offset
        target_y = state.get(jug, "y") + np.sin(rot) * cls.jug_handle_offset - 0.02
        if not cls.use_pixelated_jug:
            target_y += 0.02
        target_z = cls.z_lb + cls.jug_handle_height()
        return (target_x, target_y, target_z)

    @classmethod
    def _get_pour_position(
        cls,
        state: State,
        cup: Object,
    ) -> Tuple[float, float, float]:
        """Get the target position for pouring into a cup."""
        target_x = state.get(cup, "x") + cls.pour_x_offset
        target_y = state.get(cup, "y") + cls.pour_y_offset
        target_z = cls.z_lb + cls.pour_z_offset()
        return (target_x, target_y, target_z)

    def _get_cup_to_pour(self, state: State) -> Optional[Object]:
        """Find the closest cup that is in pour range."""
        jug_x = state.get(self._jug, "x")
        jug_y = state.get(self._jug, "y")
        jug_z = self._get_jug_z(state, self._jug)
        jug_pos = (jug_x, jug_y, jug_z)
        closest_cup = None
        closest_cup_dist = float("inf")
        for cup in state.get_objects(self._cup_type):
            target = self._get_pour_position(state, cup)
            sq_dist = np.sum(np.subtract(jug_pos, target) ** 2)
            if sq_dist < self.pour_pos_tol and sq_dist < closest_cup_dist:
                closest_cup = cup
                closest_cup_dist = sq_dist
        return closest_cup

    def _get_jug_z(self, state: State, jug: Object) -> float:
        if state.get(jug, "is_held") > 0.5:
            return state.get(self._robot, "z") - self.jug_handle_height()
        return self.z_lb

    # =================================================================
    # Task generation
    # =================================================================

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num=self._config.num_train_tasks,
            num_cups_lst=self.num_cups_train,
            rng=self._train_rng,
            is_train=True,
        )

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num=self._config.num_test_tasks,
            num_cups_lst=self.num_cups_test,
            rng=self._test_rng,
            is_train=False,
        )

    def _get_tasks(
        self,
        num: int,
        num_cups_lst: List[int],
        rng: np.random.Generator,
        is_train: bool = False,
    ) -> List[EnvironmentTask]:
        del is_train  # unused
        tasks: List[EnvironmentTask] = []

        common_state_dict: Dict[Object, Dict[str, float]] = {}
        common_state_dict[self._robot] = {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
            "fingers": self.open_fingers,
        }
        common_state_dict[self._machine] = {
            "is_on": 0.0,
        }

        for task_idx in range(num):
            state_dict = {k: v.copy() for k, v in common_state_dict.items()}

            if self.simple_tasks:
                num_cups = 0
            else:
                num_cups = num_cups_lst[rng.choice(len(num_cups_lst))]

            cups = self._cups[:num_cups]

            if self.simple_tasks:
                goal = {
                    GroundAtom(self._JugFilled, [self._jug]),
                    GroundAtom(self._JugInMachine, [self._jug, self._machine]),
                }
            else:
                goal = {GroundAtom(self._CupFilled, [c]) for c in cups}

            # Sample cup positions.
            radius = self.cup_radius + self.init_padding
            cup_state_dict: Dict[Object, Dict[str, float]] = {}
            cup_positions = sample_collision_free_2d_positions(
                num_cups,
                (self.cup_init_x_lb, self.cup_init_x_ub),
                (self.cup_init_y_lb, self.cup_init_y_ub),
                "circle",
                (radius,),
                rng=self._train_rng,
            )
            for cup, (cx, cy) in zip(self._cups, cup_positions):
                cap = rng.uniform(self.cup_capacity_lb, self.cup_capacity_ub)
                if self.use_pixelated_jug:
                    target_liquid = cap
                else:
                    target_liquid = cap * self.cup_target_frac
                cup_state_dict[cup] = {
                    "x": cx,
                    "y": cy,
                    "z": self.z_lb + cap / 2,
                    "capacity_liquid": cap,
                    "target_liquid": target_liquid,
                    "current_liquid": 0.0,
                }
            state_dict.update(cup_state_dict)

            # Sample jug position and rotation.
            x = rng.uniform(self.jug_init_x_lb, self.jug_init_x_ub)
            y = rng.uniform(self.jug_init_y_lb, self.jug_init_y_ub)

            p_rot = self.rotated_jug_ratio
            add_rotation = rng.choice([True, False], p=[p_rot, 1 - p_rot])
            if add_rotation:
                logging.info("Adding rotation to jug for task %d", task_idx)
                rot = self.jug_init_rot_ub
            else:
                epsilon = 1e-10
                rot = rng.uniform(-0.1 + epsilon, 0.1 - epsilon)
            rot -= np.pi / 2

            state_dict[self._jug] = {
                "x": x,
                "y": y,
                "z": self.z_lb + self.jug_height() / 2,
                "rot": rot,
                "is_held": 0.0,
            }
            if self.fill_jug_gradually:
                state_dict[self._jug]["current_liquid"] = 0.0
            else:
                state_dict[self._jug]["is_filled"] = 0.0

            if self.machine_has_plug:
                state_dict[self._plug] = {
                    "x": self.plug_x,
                    "y": self.plug_y,
                    "z": self.plug_z,
                    "plugged_in": 0.0,
                }

            init_state = utils.create_state_from_dict(state_dict)
            task = EnvironmentTask(init_state, goal)
            tasks.append(task)

        return self._add_pybullet_state_to_tasks(tasks)
