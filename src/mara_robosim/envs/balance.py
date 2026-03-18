"""PyBullet Balance environment.

A robot must balance blocks across two plates on a seesaw-like mechanism
and press a button when the plates are balanced.
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from mara_robosim import utils
from mara_robosim.config import PyBulletConfig
from mara_robosim.envs.base_env import PyBulletEnv, create_pybullet_block
from mara_robosim.pybullet_helpers.geometry import Pose3D, Quaternion
from mara_robosim.pybullet_helpers.robots import SingleArmPyBulletRobot
from mara_robosim.structs import (
    Action,
    Array,
    EnvironmentTask,
    GroundAtom,
    Object,
    Predicate,
    State,
    Type,
)


class PyBulletBalanceEnv(PyBulletEnv):
    """PyBullet Balance domain."""

    # -- Domain-specific defaults ----------------------------------------------
    _block_size: ClassVar[float] = 0.045
    _num_blocks_train: ClassVar[List[int]] = [2, 4]
    _num_blocks_test: ClassVar[List[int]] = [4, 6]
    _holding_goals: ClassVar[bool] = False
    _weird_balance: ClassVar[bool] = False

    # -- Table parameters ---------------------------------------------------
    _table_height: ClassVar[float] = 0.4
    _table2_pose: ClassVar[Pose3D] = (1.35, 0.75, _table_height / 2)
    _table_x, _table2_y, _table_z = _table2_pose
    _table_orientation: ClassVar[Quaternion] = (0.0, 0.0, 0.0, 1.0)
    _table_mid_w = 0.1
    _table_side_w = 0.3
    _table_gap = 0.05
    _table_mid_half_extents = (0.1, _table_mid_w / 2, _table_height / 2)

    # -- Plate --------------------------------------------------------------
    _plate_height: ClassVar[float] = 0.02
    _plate_z = _table_height - _plate_height * 3
    _plate1_pose: ClassVar[Pose3D] = (
        _table_x,
        _table2_y - _table_mid_w / 2 - _table_side_w / 2 - _table_gap,
        _plate_z,
    )
    _plate3_pose: ClassVar[Pose3D] = (
        _table_x,
        _table2_y + _table_mid_w / 2 + _table_side_w / 2 + _table_gap,
        _plate_z,
    )
    _plate_half_extents = (0.25, _table_side_w / 2, _plate_height)

    # -- Under-plate beams --------------------------------------------------
    _beam1_pose: ClassVar[Pose3D] = (
        _table_x,
        (_plate1_pose[1] + _table2_pose[1]) / 2,
        _plate_z - 4 * _plate_height,
    )
    _beam2_pose: ClassVar[Pose3D] = (
        _table_x,
        (_plate3_pose[1] + _table2_pose[1]) / 2,
        _plate_z - 4 * _plate_height,
    )
    _beam_half_extents = (0.01, 0.15, _plate_height / 2)

    # -- Button on table ----------------------------------------------------
    _button_radius = 0.04
    _button_color_off = (1, 0, 0, 1)
    _button_color_on = (0, 1, 0, 1)
    button_x, button_y, button_z = _table_x, _table2_y, _table_height
    button_press_threshold = 1e-3

    # -- Workspace parameters -----------------------------------------------
    x_lb: ClassVar[float] = 1.325
    x_ub: ClassVar[float] = 1.375
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    z_lb: ClassVar[float] = _table_height
    z_ub: ClassVar[float] = 0.75 + _table_height / 2
    y_plate1_ub: ClassVar[float] = _plate1_pose[1] + _table_side_w / 2 - 0.1
    y_plate3_lb: ClassVar[float] = _plate3_pose[1] - _table_side_w / 2 + 0.1

    # -- Robot parameters ---------------------------------------------------
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = z_ub - 0.1
    held_tol: ClassVar[float] = 0.5
    on_tol: ClassVar[float] = 0.01
    collision_padding: ClassVar[float] = 2.0

    # -- Camera -------------------------------------------------------------
    _camera_target: ClassVar[Pose3D] = (1.65, 0.75, 0.52)

    # -- Block physics ------------------------------------------------------
    _block_mass: ClassVar[float] = 1

    def __init__(
        self, config: Optional[PyBulletConfig] = None, use_gui: bool = True
    ) -> None:
        # Types
        self._block_type = Type(
            "block", ["x", "y", "z", "is_held", "color_r", "color_g", "color_b"]
        )
        self._robot_type = Type("robot", ["x", "y", "z", "fingers"])
        self._plate_type = Type("plate", ["z"])
        self._machine_type = Type("machine", ["is_on"])

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._plate1 = Object("plate1", self._plate_type)
        self._plate3 = Object("plate3", self._plate_type)
        self._machine = Object("mac", self._machine_type)
        self._blocks = [
            Object(f"block{i}", self._block_type)
            for i in range(max(self._num_blocks_train + self._num_blocks_test))
        ]

        super().__init__(config, use_gui)

        # Predicates
        self._DirectlyOn = Predicate(
            "DirectlyOn",
            [self._block_type, self._block_type],
            self._DirectlyOn_holds,
        )
        self._DirectlyOnPlate = Predicate(
            "DirectlyOnPlate",
            [self._block_type, self._plate_type],
            self._DirectlyOnPlate_holds,
        )
        self._GripperOpen = Predicate(
            "GripperOpen", [self._robot_type], self._GripperOpen_holds
        )
        self._Holding = Predicate("Holding", [self._block_type], self._Holding_holds)
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)
        self._MachineOn = Predicate(
            "MachineOn", [self._machine_type, self._robot_type], self._MachineOn_holds
        )
        self._Balanced = Predicate(
            "Balanced", [self._plate_type, self._plate_type], self._Balanced_holds
        )
        self._ClearPlate = Predicate(
            "ClearPlate", [self._plate_type], self._ClearPlate_holds
        )

        # We track the correspondence between PyBullet object IDs and Object
        # instances for blocks. This correspondence changes with the task.
        self._block_id_to_block: Dict[int, Object] = {}

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_balance"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._DirectlyOn,
            self._DirectlyOnPlate,
            self._GripperOpen,
            self._Holding,
            self._Clear,
            self._MachineOn,
            self._ClearPlate,
            self._Balanced,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        if self._holding_goals:
            return {self._Holding}
        return {self._DirectlyOn, self._DirectlyOnPlate}

    @property
    def types(self) -> Set[Type]:
        return {
            self._block_type,
            self._robot_type,
            self._plate_type,
            self._machine_type,
        }

    # -------------------------------------------------------------------------
    # Environment Setup

    @classmethod
    def initialize_pybullet(
        cls,
        using_gui: bool,
        config: Optional[PyBulletConfig] = None,
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle balance-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui, config=config
        )

        table2_id = create_pybullet_block(
            (0.9, 0.9, 0.9, 1),
            cls._table_mid_half_extents,
            0.0,  # mass
            1.0,  # friction
            cls._table2_pose,
            cls._table_orientation,
            physics_client_id,
        )

        plate3_id = create_pybullet_block(
            (0.9, 0.9, 0.9, 1),
            cls._plate_half_extents,
            0.0,
            1.0,
            cls._plate3_pose,
            cls._table_orientation,
            physics_client_id,
        )

        plate1_id = create_pybullet_block(
            (0.9, 0.9, 0.9, 1),
            cls._plate_half_extents,
            0.0,
            1.0,
            cls._plate1_pose,
            cls._table_orientation,
            physics_client_id,
        )
        bodies["table_ids"] = [plate1_id, plate3_id, table2_id]

        beam1_id = create_pybullet_block(
            (0.9, 0.9, 0.9, 1),
            cls._beam_half_extents,
            0.0,
            1.0,
            cls._beam1_pose,
            cls._table_orientation,
            physics_client_id,
        )
        beam2_id = create_pybullet_block(
            (0.9, 0.9, 0.9, 1),
            cls._beam_half_extents,
            0.0,
            1.0,
            cls._beam2_pose,
            cls._table_orientation,
            physics_client_id,
        )
        bodies["beam_ids"] = [beam1_id, beam2_id]

        button_id = create_pybullet_block(
            cls._button_color_off,
            (cls._button_radius, cls._button_radius, cls._button_radius / 2),
            0.0,
            1.0,
            (cls.button_x, cls.button_y, cls.button_z),
            cls._table_orientation,
            physics_client_id,
        )
        bodies["button_id"] = button_id

        # Create blocks. Note that we create the maximum number once, and then
        # later on, in reset_state(), we will remove blocks from the workspace
        # (teleporting them far away) based on which ones are in the state.
        num_blocks = max(max(cls._num_blocks_train), max(cls._num_blocks_test))
        block_ids = []
        block_size = cls._block_size
        for i in range(num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            half_extents = (block_size / 2.0, block_size / 2.0, block_size / 2.0)
            block_ids.append(
                create_pybullet_block(
                    color,
                    half_extents,
                    cls._block_mass,
                    cls._obj_friction,
                    physics_client_id=physics_client_id,
                )
            )
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._plate1.id = pybullet_bodies["table_ids"][0]
        self._plate3.id = pybullet_bodies["table_ids"][1]
        self._table_id = pybullet_bodies["table_ids"][2]
        self._machine.id = pybullet_bodies["button_id"]
        self._robot.id = self._pybullet_robot.robot_id
        for block, block_id in zip(self._blocks, pybullet_bodies["block_ids"]):
            block.id = block_id
        self._beam_ids = pybullet_bodies["beam_ids"]

    # -------------------------------------------------------------------------
    # State Management: Get, (Re)Set, Step

    def _create_task_specific_objects(self, state: State) -> None:
        pass

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._block_type:
            visual_data = p.getVisualShapeData(
                obj.id, physicsClientId=self._physics_client_id
            )[0]
            r, g, b, _ = visual_data[7]
            if feature == "color_r":
                return r
            elif feature == "color_g":
                return g
            elif feature == "color_b":
                return b
        elif obj.type == self._machine_type:
            if feature == "is_on":
                button_color = p.getVisualShapeData(
                    self._machine.id, physicsClientId=self._physics_client_id
                )[0][-1]
                button_color_on_dist = sum(
                    np.subtract(button_color, self._button_color_on) ** 2
                )
                button_color_off_dist = sum(
                    np.subtract(button_color, self._button_color_off) ** 2
                )
                return float(button_color_on_dist < button_color_off_dist)

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def step(self, action: Action, render_obs: bool = False) -> State:
        state = super().step(action, render_obs=render_obs)

        self._update_balance_beam(state)

        # Turn machine on
        if self._PressingButton_holds(state, [self._robot, self._machine]):
            if self._Balanced_holds(state, [self._plate1, self._plate3]):
                p.changeVisualShape(
                    self._machine.id,
                    -1,
                    rgbaColor=self._button_color_on,
                    physicsClientId=self._physics_client_id,
                )
            self._current_observation = self._get_state()
            state = self._current_observation.copy()

        return state

    def _reset_custom_env_state(self, state: State) -> None:
        """Domain-specific reset logic.

        The base ``_reset_state`` has already handled standard features
        for objects that appear in ``_get_all_objects()``, so here we just
        do custom domain-specific tasks: setting block colors, hiding unused
        blocks, updating button colour, and running the beam-balancing
        update.
        """
        # block objs in the state
        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block.clear()

        for i, block_obj in enumerate(block_objs):
            self._block_id_to_block[block_obj.id] = block_obj
            # Manually set color
            r = state.get(block_obj, "color_r")
            g = state.get(block_obj, "color_g")
            b = state.get(block_obj, "color_b")
            p.changeVisualShape(
                block_obj.id,
                linkIndex=-1,
                rgbaColor=(r, g, b, 1.0),
                physicsClientId=self._physics_client_id,
            )

        # For blocks beyond the number actually in the state, put them out of
        # view:
        h = self._block_size
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._blocks)):
            p.resetBasePositionAndOrientation(
                self._blocks[i].id,
                [oov_x, oov_y, i * h],
                self._default_orn,
                physicsClientId=self._physics_client_id,
            )

        self._prev_diff = 0  # reset difference
        self._update_balance_beam(state)

        # Update button color for whether the machine is on
        if self._MachineOn_holds(state, [self._machine, self._robot]):
            button_color = self._button_color_on
        else:
            button_color = self._button_color_off
        p.changeVisualShape(
            self._machine.id,
            -1,
            rgbaColor=button_color,
            physicsClientId=self._physics_client_id,
        )

    def _update_balance_beam(self, state: State) -> None:
        """Shift the plates, beams, and blocks on them to simulate a balance.

        Ensures rising sides move blocks first then plate, and dropping sides
        move plate first then blocks.
        """
        left_count = self.count_num_blocks(state, self._plate1)
        right_count = self.count_num_blocks(state, self._plate3)
        diff = left_count - right_count
        if self._weird_balance:
            diff *= -1
        if diff == self._prev_diff:
            return

        shift_per_block = 0.007
        shift_amount = abs(diff) * shift_per_block
        block_objs = state.get_objects(self._block_type)
        left_dropping = diff > 0

        def shift_blocks(is_left: bool, dropping: bool) -> None:
            """Shift blocks for one side, dropping or rising."""
            sign = -1 if dropping else 1
            midpoint_y = self._table2_y
            for block_obj in block_objs:
                # Skip out-of-view or held
                if state.get(block_obj, "z") < 0 or self._held_obj_id == block_obj.id:
                    continue
                by = state.get(block_obj, "y")
                belongs_to_side = (by < midpoint_y) if is_left else (by > midpoint_y)
                if belongs_to_side:
                    old_z = state.get(block_obj, "z")
                    padding = 0
                    new_z = old_z + (sign * shift_amount) + (sign * padding)
                    block_pos, block_orn = p.getBasePositionAndOrientation(
                        block_obj.id, physicsClientId=self._physics_client_id
                    )
                    p.resetBasePositionAndOrientation(
                        block_obj.id,
                        [block_pos[0], block_pos[1], new_z],
                        block_orn,
                        physicsClientId=self._physics_client_id,
                    )

        def shift_plate(is_left: bool, dropping: bool) -> None:
            """Shift plate & beam, dropping or rising."""
            sign = -1 if dropping else 1
            if is_left:
                plate_id, beam_id = self._plate1.id, self._beam_ids[0]
                base_plate_z, base_beam_z = self._plate1_pose[2], self._beam1_pose[2]
            else:
                plate_id, beam_id = self._plate3.id, self._beam_ids[1]
                base_plate_z, base_beam_z = self._plate3_pose[2], self._beam2_pose[2]

            new_plate_z = base_plate_z + (sign * shift_amount)
            new_beam_z = base_beam_z + (sign * shift_amount)

            plate_pos, plate_orn = p.getBasePositionAndOrientation(
                plate_id, physicsClientId=self._physics_client_id
            )
            p.resetBasePositionAndOrientation(
                plate_id,
                [plate_pos[0], plate_pos[1], new_plate_z],
                plate_orn,
                physicsClientId=self._physics_client_id,
            )

            beam_pos, beam_orn = p.getBasePositionAndOrientation(
                beam_id, physicsClientId=self._physics_client_id
            )
            p.resetBasePositionAndOrientation(
                beam_id,
                [beam_pos[0], beam_pos[1], new_beam_z],
                beam_orn,
                physicsClientId=self._physics_client_id,
            )

        # Left side update
        if left_dropping:
            shift_plate(is_left=True, dropping=True)
            shift_blocks(is_left=True, dropping=True)
            shift_blocks(is_left=False, dropping=False)
            shift_plate(is_left=False, dropping=False)
        else:
            shift_blocks(is_left=True, dropping=False)
            shift_plate(is_left=True, dropping=False)
            shift_plate(is_left=False, dropping=True)
            shift_blocks(is_left=False, dropping=True)

        self._prev_diff = diff

    # -------------------------------------------------------------------------
    # Predicates

    def _ClearPlate_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (plate,) = objects
        for block in state.get_objects(self._block_type):
            if self._DirectlyOnPlate_holds(state, [block, plate]):
                return False
        return True

    def _MachineOn_holds(self, state: State, objects: Sequence[Object]) -> bool:
        machine, _ = objects
        return state.get(machine, "is_on") > 0.5

    def _PressingButton_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, _ = objects
        button_pos = (self.button_x, self.button_y, self.button_z + self._button_radius)
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        z = state.get(robot, "z")
        sq_dist_to_button = np.sum(np.subtract(button_pos, (x, y, z)) ** 2)
        return bool(sq_dist_to_button < self.button_press_threshold)

    def count_num_blocks(self, state: State, table: Object) -> int:
        """Count the number of blocks in a tower on the given plate."""

        def count_recursive(base_obj: Object, count: int) -> int:
            for block in state.get_objects(self._block_type):
                if base_obj.type == self._block_type and self._DirectlyOn_holds(
                    state, [block, base_obj]
                ):
                    count = count_recursive(block, count + 1)
                elif base_obj.type == self._plate_type and self._DirectlyOnPlate_holds(
                    state, [block, base_obj]
                ):
                    count = count_recursive(block, count + 1)
            return count

        return count_recursive(table, 0)

    def _Balanced_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if the blocks are balanced on the table."""
        plate1, plate2 = objects
        if plate1 == plate2:
            return False
        height1 = self.count_num_blocks(state, plate1)
        height2 = self.count_num_blocks(state, plate2)
        return height1 == height2

    def _DirectlyOn_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        if (
            state.get(block1, "is_held") >= self.held_tol
            or state.get(block2, "is_held") >= self.held_tol
        ):
            return False
        x1 = state.get(block1, "x")
        y1 = state.get(block1, "y")
        z1 = state.get(block1, "z")
        x2 = state.get(block2, "x")
        y2 = state.get(block2, "y")
        z2 = state.get(block2, "z")
        return np.allclose(
            [x1, y1, z1], [x2, y2, z2 + self._block_size], atol=self.on_tol
        )

    def _DirectlyOnPlate_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, table = objects
        y = state.get(block, "y")
        z = state.get(block, "z")
        table_z = state.get(table, "z") + self._plate_height / 2
        desired_z = table_z + self._block_size * 0.5

        if (state.get(block, "is_held") < self.held_tol) and (
            desired_z - self.on_tol < z < desired_z + self.on_tol
        ):
            if table.name == "plate1":
                return y < self._table2_y
            elif table.name == "plate3":
                return y > self._table2_y
            else:
                raise ValueError("Invalid table name")
        else:
            return False

    def _GripperOpen_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (robot,) = objects
        rf = state.get(robot, "fingers")
        return rf > 0.03

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (block,) = objects
        return self._get_held_block(state) == block

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        if self._Holding_holds(state, objects):
            return False
        (block,) = objects
        for other_block in state:
            if other_block.type != self._block_type:
                continue
            if self._DirectlyOn_holds(state, [other_block, block]):
                return False
        return True

    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            if state.get(block, "is_held") >= self.held_tol:
                return block
        return None

    # -------------------------------------------------------------------------
    # Task Generation

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=self._config.num_train_tasks,
            possible_num_blocks=self._num_blocks_train,
            rng=self._train_rng,
        )

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=self._config.num_test_tasks,
            possible_num_blocks=self._num_blocks_test,
            rng=self._test_rng,
        )

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _force_grasp_object(self, block: Object) -> None:
        block_to_block_id = {b: i for i, b in self._block_id_to_block.items()}
        block_id = block_to_block_id[block]
        # The block should already be held. Otherwise, the position of the
        # block was wrong in the state.
        held_obj_id = self._detect_held_object()
        assert block_id == held_obj_id
        # Create the grasp constraint.
        self._held_obj_id = block_id
        self._create_grasp_constraint()

    def _make_tasks(
        self, num_tasks: int, possible_num_blocks: List[int], rng: np.random.Generator
    ) -> List[EnvironmentTask]:
        tasks = []
        for idx in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks, p=[0.3, 0.7])
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            if max(possible_num_blocks) == 4:
                goal = {
                    GroundAtom(self._MachineOn, [self._machine, self._robot]),
                    GroundAtom(self._DirectlyOn, [piles[1][2], piles[0][0]]),
                }
            else:
                goal = {
                    GroundAtom(self._MachineOn, [self._machine, self._robot]),
                    GroundAtom(self._DirectlyOn, [piles[1][4], piles[0][0]]),
                    GroundAtom(self._DirectlyOn, [piles[1][3], piles[1][4]]),
                }
            tasks.append(EnvironmentTask(init_state, goal))
        return self._add_pybullet_state_to_tasks(tasks)

    def _sample_initial_piles(
        self, num_blocks: int, rng: np.random.Generator
    ) -> List[List[Object]]:
        n_piles = 0
        piles: List[List[Object]] = []
        for block_num in range(num_blocks):
            block = self._blocks[block_num]
            # For generating a 1:5 pile
            if block_num == 0 or block_num == 1:
                n_piles += 1
                piles.append([])
            # Add block to pile
            piles[-1].append(block)
        return piles

    def _sample_state_from_piles(
        self, piles: List[List[Object]], rng: np.random.Generator
    ) -> State:
        data: Dict[Object, Array] = {}
        # Create objects
        block_to_pile_idx = {}
        for i, pile in enumerate(piles):
            for j, block in enumerate(pile):
                assert block not in block_to_pile_idx
                block_to_pile_idx[block] = (i, j)
        # Sample pile (x, y)s
        pile_to_xy: Dict[int, Tuple[float, float]] = {}
        for i in range(len(piles)):
            pile_to_xy[i] = self._sample_initial_pile_xy(rng, set(pile_to_xy.values()))
        # Create block states
        for block, pile_idx in block_to_pile_idx.items():
            pile_i, pile_j = pile_idx
            x, y = pile_to_xy[pile_i]
            z = self._plate_z + self._plate_height + self._block_size * (0.5 + pile_j)
            r, g, b = rng.uniform(size=3)
            # [x, y, z, held, color_r, color_g, color_b]
            data[block] = np.array([x, y, z, 0.0, r, g, b])
        # [x, y, z, fingers]
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = self.open_fingers  # fingers start out open
        data[self._robot] = np.array([rx, ry, rz, rf], dtype=np.float32)
        data[self._plate1] = np.array([self._plate1_pose[2]], dtype=np.float32)
        data[self._plate3] = np.array([self._plate3_pose[2]], dtype=np.float32)
        data[self._machine] = np.array([0.0], dtype=np.float32)
        return State(data)

    def _sample_goal_from_piles(
        self, num_blocks: int, piles: List[List[Object]], rng: np.random.Generator
    ) -> Set[GroundAtom]:
        """Sample a goal that involves holding a block on top of a pile, or
        rearranging piles."""
        if self._holding_goals:
            pile_idx = rng.choice(len(piles))
            top_block = piles[pile_idx][-1]
            return {GroundAtom(self._Holding, [top_block])}
        # Sample goal pile that is different from initial
        while True:
            goal_piles = self._sample_initial_piles(num_blocks, rng)
            if goal_piles != piles:
                break
        # Create goal from piles
        goal_atoms = set()
        for pile in goal_piles:
            goal_atoms.add(GroundAtom(self._DirectlyOnPlate, [pile[0]]))
            if len(pile) == 1:
                continue
            for block1, block2 in zip(pile[1:], pile[:-1]):
                goal_atoms.add(GroundAtom(self._DirectlyOn, [block1, block2]))
        return goal_atoms

    def _sample_initial_pile_xy(
        self, rng: np.random.Generator, existing_xys: Set[Tuple[float, float]]
    ) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.x_lb, self.x_ub)
            if rng.uniform(0, 1) < 0.5:
                # Table 1
                y = rng.uniform(self.y_lb, self.y_plate1_ub)
            else:
                # Table 3
                y = rng.uniform(self.y_plate3_lb, self.y_ub)

            if self._table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    def _table_xy_is_clear(
        self, x: float, y: float, existing_xys: Set[Tuple[float, float]]
    ) -> bool:
        if all(
            abs(x - other_x) > self.collision_padding * self._block_size
            for other_x, _ in existing_xys
        ):
            return True
        if all(
            abs(y - other_y) > self.collision_padding * self._block_size
            for _, other_y in existing_xys
        ):
            return True
        return False
