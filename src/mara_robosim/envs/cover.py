"""PyBullet Cover environment.

A robot arm picks up blocks and places them on targets to 'cover' them.
This is a merged version that combines domain logic from predicators'
CoverEnv with the PyBullet simulation from PyBulletCoverEnv.

x: robot -> table
y: table left -> right
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from mara_robosim import utils
from mara_robosim.config import CoverConfig, PyBulletConfig
from mara_robosim.envs.base_env import PyBulletEnv, create_pybullet_block
from mara_robosim.pybullet_helpers.geometry import Pose3D, Quaternion
from mara_robosim.pybullet_helpers.objects import update_object
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


class PyBulletCoverEnv(PyBulletEnv):
    """PyBullet Cover domain, a standalone merge of CoverEnv + PyBulletEnv.

    x: robot -> table
    y: table left -> right
    """

    # ------------------------------------------------------------------
    # Domain-logic class variables (merged from CoverEnv)
    # ------------------------------------------------------------------
    _allow_free_space_placing: ClassVar[bool] = False
    _initial_pick_offsets: ClassVar[List[float]] = []

    workspace_x: ClassVar[float] = 1.35
    workspace_z: ClassVar[float] = 0.65

    # Cover-specific defaults now live in CoverConfig (self._config)

    # ------------------------------------------------------------------
    # PyBullet-specific class variables
    # ------------------------------------------------------------------
    # Table parameters
    _table_height: ClassVar[float] = 0.4
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, _table_height / 2)
    _table_orientation: ClassVar[Quaternion] = (0.0, 0.0, 0.0, 1.0)

    _camera_target: ClassVar[Pose3D] = (1.65, 0.75, 0.62)

    # Object parameters
    _obj_len_hgt: ClassVar[float] = 0.045
    _max_obj_width: ClassVar[float] = 0.07  # highest width normalized to this
    _block_cover_color: ClassVar[Tuple[float, float, float, float]] = (
        1.0,
        1.0,
        1.0,
        1.0,
    )

    # Dimension and workspace parameters
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    robot_init_x: ClassVar[float] = workspace_x
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = workspace_z
    _offset: ClassVar[float] = 0.01
    pickplace_z: ClassVar[float] = _table_height + _obj_len_hgt * 0.5 + _offset
    _target_height: ClassVar[float] = 0.0001

    _obj_colors_bw: ClassVar[Sequence[Tuple[float, float, float, float]]] = [
        (0, 0, 0, 1.0),
        (1, 1, 1, 1.0),
    ]

    # ------------------------------------------------------------------
    # Types (merged from CoverEnv)
    # ------------------------------------------------------------------
    _block_type = Type("block", ["is_block", "is_target", "width", "pose", "grasp"])
    _target_type = Type("target", ["is_block", "is_target", "width", "pose"])
    _robot_type = Type("robot", ["hand", "pose_x", "pose_z"])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self, config: Optional[CoverConfig] = None, use_gui: bool = True
    ) -> None:
        config = CoverConfig._upgrade(config or CoverConfig())

        # Predicates (must be set before super().__init__ because
        # task generation may need them).
        self._IsBlock = Predicate("IsBlock", [self._block_type], self._IsBlock_holds)
        self._IsTarget = Predicate(
            "IsTarget", [self._target_type], self._IsTarget_holds
        )
        self._Covers = Predicate(
            "Covers", [self._block_type, self._target_type], self._Covers_holds
        )
        self._HandEmpty = Predicate("HandEmpty", [], self._HandEmpty_holds)
        self._Holding = Predicate("Holding", [self._block_type], self._Holding_holds)

        # Static objects
        self._robot = Object("robby", self._robot_type)
        self._blocks: List[Object] = []
        self._targets: List[Object] = []
        self._create_blocks_and_targets(config)

        # Call PyBulletEnv.__init__ (sets up physics, robot, etc.)
        super().__init__(config, use_gui)

        # Store table ID for reference
        self._table_id: int = -1

        # Forward-kinematics client for hand-constraint checking in step()
        fk_physics_id = p.connect(p.DIRECT)
        self._pybullet_robot_fk = self._create_pybullet_robot(
            fk_physics_id, config=self._config
        )

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_cover"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._IsBlock,
            self._IsTarget,
            self._Covers,
            self._HandEmpty,
            self._Holding,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Covers}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._target_type, self._robot_type}

    # ------------------------------------------------------------------
    # PyBullet initialisation
    # ------------------------------------------------------------------

    @classmethod
    def initialize_pybullet(
        cls,
        using_gui: bool,
        config: Optional[PyBulletConfig] = None,
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Create the world: plane, table, block IDs, etc."""
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui, config=config
        )

        # Load table
        table_id = p.loadURDF(
            utils.get_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=physics_client_id,
        )
        bodies["table_id"] = table_id
        p.resetBasePositionAndOrientation(
            table_id,
            cls._table_pose,
            cls._table_orientation,
            physicsClientId=physics_client_id,
        )

        # Create blocks
        cover_block_widths = getattr(config, "cover_block_widths", (0.1, 0.07))
        cover_target_widths = getattr(config, "cover_target_widths", (0.05, 0.03))
        cover_num_blocks = getattr(config, "cover_num_blocks", 2)
        cover_num_targets = getattr(config, "cover_num_targets", 2)
        max_width = max(max(cover_block_widths), max(cover_target_widths))
        block_ids = []
        for i in range(cover_num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            width = cover_block_widths[i] / max_width * cls._max_obj_width
            half_extents = (cls._obj_len_hgt / 2.0, width / 2.0, cls._obj_len_hgt / 2.0)
            block_id = create_pybullet_block(
                color=color,
                half_extents=half_extents,
                mass=cls._obj_mass,
                friction=cls._obj_friction,
                physics_client_id=physics_client_id,
            )
            block_ids.append(block_id)
        bodies["block_ids"] = block_ids

        # Create targets
        target_ids = []
        for i in range(cover_num_targets):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            color = (color[0], color[1], color[2], 0.5)  # semi-transparent
            width = cover_target_widths[i] / max_width * cls._max_obj_width
            half_extents = (cls._obj_len_hgt * 2, width / 2.0, cls._target_height / 2.0)
            target_id = create_pybullet_block(
                color=color,
                half_extents=half_extents,
                mass=cls._obj_mass,
                friction=cls._obj_friction,
                physics_client_id=physics_client_id,
            )
            target_ids.append(target_id)
        bodies["target_ids"] = target_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet IDs for environment assets."""
        self._table_id = pybullet_bodies["table_id"]
        blk_id: int
        for blk, blk_id in zip(self._blocks, pybullet_bodies["block_ids"]):
            blk.id = blk_id
        tgt_id: int
        for tgt, tgt_id in zip(self._targets, pybullet_bodies["target_ids"]):
            tgt.id = tgt_id

    def _create_task_specific_objects(self, state: State) -> None:
        """No domain-specific extra creation needed here."""

    def _get_object_ids_for_held_check(self) -> List[int]:
        """We only consider blocks for 'held' detection here."""
        return [blk.id for blk in self._blocks]

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        # Both fetch and panda have grippers parallel to x-axis
        return {
            self._pybullet_robot.left_finger_id: np.array([1.0, 0.0, 0.0]),
            self._pybullet_robot.right_finger_id: np.array([-1.0, 0.0, 0.0]),
        }

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _extract_robot_state(self, state: State) -> np.ndarray:
        """Convert from our domain's features (hand, pose_x, pose_z) into the.

        [x, y, z, qx, qy, qz, qw, fingers] array expected by the
        PyBullet robot.
        """
        # 1) Determine fingers (closed if any block is being held)
        is_holding_something = False
        for obj in state.get_objects(self._block_type):
            if state.get(obj, "grasp") != -1:
                is_holding_something = True
                break
        if is_holding_something:
            fingers = self._pybullet_robot.closed_fingers
        else:
            fingers = self._pybullet_robot.open_fingers

        # 2) The robot object
        robot_obj = state.get_objects(self._robot_type)[0]
        hand_norm = state.get(robot_obj, "hand")
        rx = state.get(robot_obj, "pose_x")
        rz = state.get(robot_obj, "pose_z")

        # De-normalize the hand => actual y coordinate
        ry = self.y_lb + (self.y_ub - self.y_lb) * hand_norm

        # 3) The orientation is fixed (pointing downward)
        qx, qy, qz, qw = self.get_robot_ee_home_orn(self._config)

        return np.array([rx, ry, rz, qx, qy, qz, qw, fingers], dtype=np.float32)

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Domain-specific feature extraction for blocks, targets, and the
        robot."""
        max_width = max(
            max(self._config.cover_block_widths), max(self._config.cover_target_widths)
        )

        # Block features
        if obj.type == self._block_type:
            block_id = obj.id
            if feature == "is_block":
                return 1.0
            if feature == "is_target":
                return 0.0
            if feature == "width":
                shape_data = p.getVisualShapeData(
                    block_id, physicsClientId=self._physics_client_id
                )[0]
                y_half = shape_data[3][1]
                width = (y_half * 2.0) / self._max_obj_width * max_width
                return width
            if feature == "pose":
                (bx, by, bz), _ = p.getBasePositionAndOrientation(
                    block_id, physicsClientId=self._physics_client_id
                )
                return (by - self.y_lb) / (self.y_ub - self.y_lb)
            if feature == "grasp":
                if (
                    block_id == self._held_obj_id
                    and self._held_constraint_id is not None
                ):
                    pivot_in_B = p.getConstraintInfo(
                        self._held_constraint_id,
                        physicsClientId=self._physics_client_id,
                    )[7]
                    grasp_unnorm = pivot_in_B[1]
                    return grasp_unnorm / (self.y_ub - self.y_lb)
                else:
                    return -1.0
            raise ValueError(f"Unknown block feature: {feature}")

        # Target features
        if obj.type == self._target_type:
            target_id = obj.id
            if feature == "is_block":
                return 0.0
            if feature == "is_target":
                return 1.0
            if feature == "width":
                shape_data = p.getVisualShapeData(
                    target_id, physicsClientId=self._physics_client_id
                )[0]
                y_half = shape_data[3][1]
                width = (y_half * 2.0) / self._max_obj_width * max_width
                return width
            if feature == "pose":
                (tx, ty, tz), _ = p.getBasePositionAndOrientation(
                    target_id, physicsClientId=self._physics_client_id
                )
                return (ty - self.y_lb) / (self.y_ub - self.y_lb)
            raise ValueError(f"Unknown target feature: {feature}")

        raise ValueError(f"Unknown object type or feature: {obj}, {feature}")

    def _reset_custom_env_state(self, state: State) -> None:
        """After the parent class has reset the robot, handle the block/target
        positions.

        Because our block objects do not have standard 'x','y','z'
        features, we do the custom placement here.
        """
        max_width = max(
            max(self._config.cover_block_widths), max(self._config.cover_target_widths)
        )

        # 1) Reset blocks
        block_objs = state.get_objects(self._block_type)
        for i, block_obj in enumerate(block_objs):
            width_unnorm = p.getVisualShapeData(
                block_obj.id, physicsClientId=self._physics_client_id
            )[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            assert np.isclose(
                width, state.get(block_obj, "width"), atol=1e-5
            ), "Mismatch in block width!"

            bx = self.workspace_x
            y_norm = state.get(block_obj, "pose")
            by = self.y_lb + (self.y_ub - self.y_lb) * y_norm

            grasp_val = state.get(block_obj, "grasp")
            if grasp_val != -1:
                bz = self.workspace_z - self._offset
            else:
                bz = self._table_height + self._obj_len_hgt * 0.5

            color = self._obj_colors[self._train_rng.choice(len(self._obj_colors))]
            update_object(
                block_obj.id,
                position=(bx, by, bz),
                color=color,
                physics_client_id=self._physics_client_id,
            )

            # If initially held, set up constraint
            if grasp_val != -1:
                self._held_obj_id = block_obj.id
                self._create_grasp_constraint()

        # Put any leftover blocks out of view
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._blocks)):
            oov_x2, oov_y2 = self._out_of_view_xy
            update_object(
                self._blocks[i].id,
                position=(oov_x2, oov_y2, 2.0),
                physics_client_id=self._physics_client_id,
            )

        # 2) Reset targets
        target_objs = state.get_objects(self._target_type)
        for i, target_obj in enumerate(target_objs):
            width_unnorm = p.getVisualShapeData(
                target_obj.id, physicsClientId=self._physics_client_id
            )[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            assert np.isclose(width, state.get(target_obj, "width"), atol=1e-5)

            y_norm = state.get(target_obj, "pose")
            ty = self.y_lb + (self.y_ub - self.y_lb) * y_norm
            tx = self.workspace_x
            tz = self._table_height + self._obj_len_hgt * 0.5

            color = self._obj_colors[self._train_rng.choice(len(self._obj_colors))]
            color = (color[0], color[1], color[2], 0.5)  # semi-transparent
            update_object(
                target_obj.id,
                position=(tx, ty, tz),
                color=color,
                physics_client_id=self._physics_client_id,
            )

        # 3) Optionally draw hand regions as debug lines
        if self._config.draw_debug:  # pragma: no cover
            assert self.using_gui, "use_gui must be True to use draw_debug."
            p.removeAllUserDebugItems(physicsClientId=self._physics_client_id)
            for hand_lb, hand_rb in self._get_hand_regions(state):
                y_lb_val = self.y_lb + (self.y_ub - self.y_lb) * hand_lb
                y_rb_val = self.y_lb + (self.y_ub - self.y_lb) * hand_rb
                p.addUserDebugLine(
                    [self.workspace_x, y_lb_val, self._table_height + 1e-4],
                    [self.workspace_x, y_rb_val, self._table_height + 1e-4],
                    [0.0, 0.0, 1.0],
                    lineWidth=5.0,
                    physicsClientId=self._physics_client_id,
                )

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Override to handle the Cover domain's 'hand region' constraint
        before calling the parent's step()."""
        if not self._satisfies_hand_constraints(action):
            return self._current_state.copy()

        next_state = super().step(action, render_obs=render_obs)

        if self._config.cover_blocks_change_color_when_cover:
            self._change_block_color_when_cover(next_state)
        return next_state

    def _change_block_color_when_cover(self, state: State) -> None:
        """If a block is now covering a target, change its color to
        self._block_cover_color."""
        for block_obj in state.get_objects(self._block_type):
            for target_obj in state.get_objects(self._target_type):
                if self._Covers_holds(state, [block_obj, target_obj]):
                    update_object(
                        block_obj.id,
                        color=self._block_cover_color,
                        physics_client_id=self._physics_client_id,
                    )
                    break

    def _satisfies_hand_constraints(self, action: Action) -> bool:
        joint_positions = action.arr.tolist()
        _, ry, rz = self._pybullet_robot_fk.forward_kinematics(joint_positions).position

        if self._is_below_z_threshold(rz):
            return self._is_in_valid_hand_region(ry)
        return True

    def _is_below_z_threshold(self, rz: float) -> bool:
        """Check if the z position is below the threshold."""
        z_thresh = (self.pickplace_z + self.workspace_z) / 2
        return rz < z_thresh

    def _is_in_valid_hand_region(self, ry: float) -> bool:
        """Check if the hand position is within any valid hand region."""
        hand = (ry - self.y_lb) / (self.y_ub - self.y_lb)
        hand_regions = self._get_hand_regions(self._current_state)
        return any(lb <= hand <= rb for lb, rb in hand_regions)

    # ------------------------------------------------------------------
    # Domain logic (merged from CoverEnv)
    # ------------------------------------------------------------------

    def _create_blocks_and_targets(self, config: CoverConfig) -> None:
        """Create block and target Object instances."""
        for i in range(config.cover_num_blocks):
            self._blocks.append(Object(f"block{i}", self._block_type))
        for i in range(config.cover_num_targets):
            self._targets.append(Object(f"target{i}", self._target_type))

    def _get_hand_regions(self, state: State) -> List[Tuple[float, float]]:
        """Return the list of (lb, ub) hand regions based on block and target
        positions."""
        hand_regions = []
        for block in state.get_objects(self._block_type):
            hand_regions.append(
                (
                    state.get(block, "pose") - state.get(block, "width") / 2,
                    state.get(block, "pose") + state.get(block, "width") / 2,
                )
            )
        for targ in state.get_objects(self._target_type):
            hand_regions.append(
                (
                    state.get(targ, "pose") - state.get(targ, "width") / 10,
                    state.get(targ, "pose") + state.get(targ, "width") / 10,
                )
            )
        return hand_regions

    def _any_intersection(
        self,
        pose: float,
        width: float,
        data: Dict[Object, Array],
        block_only: bool = False,
        larger_gap: bool = False,
        excluded_object: Optional[Object] = None,
    ) -> bool:
        """Check whether placing an object at (pose, width) would intersect any
        existing object in data."""
        mult = 1.5 if larger_gap else 0.5
        for other in data:
            if block_only and other.type != self._block_type:
                continue
            if excluded_object is not None and other == excluded_object:
                continue
            other_feats = data[other]
            distance = abs(other_feats[3] - pose)
            if distance <= (width + other_feats[2]) * mult:
                return True
        return False

    # ------------------------------------------------------------------
    # Predicates (merged from CoverEnv)
    # ------------------------------------------------------------------

    @staticmethod
    def _IsBlock_holds(state: State, objects: Sequence[Object]) -> bool:
        (block,) = objects
        return block in state

    @staticmethod
    def _IsTarget_holds(state: State, objects: Sequence[Object]) -> bool:
        (target,) = objects
        return target in state

    @staticmethod
    def _Covers_holds(state: State, objects: Sequence[Object]) -> bool:
        block, target = objects
        block_pose = state.get(block, "pose")
        block_width = state.get(block, "width")
        target_pose = state.get(target, "pose")
        target_width = state.get(target, "width")
        return (
            (block_pose - block_width / 2 <= target_pose - target_width / 2)
            and (block_pose + block_width / 2 >= target_pose + target_width / 2)
            and state.get(block, "grasp") == -1
        )

    def _HandEmpty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        assert not objects
        for obj in state:
            if obj.is_instance(self._block_type) and state.get(obj, "grasp") != -1:
                return False
        return True

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        (block,) = objects
        return state.get(block, "grasp") != -1

    # ------------------------------------------------------------------
    # Task generation (merged from CoverEnv)
    # ------------------------------------------------------------------

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=self._config.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=self._config.num_test_tasks, rng=self._test_rng)

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[EnvironmentTask]:
        """Generate cover tasks with goals involving Covers predicates."""
        tasks = []
        blocks, targets = self._blocks, self._targets

        # Create goals
        goal1 = {GroundAtom(self._Covers, [blocks[0], targets[0]])}
        goals = [goal1]
        if len(blocks) > 1 and len(targets) > 1:
            goal2 = {GroundAtom(self._Covers, [blocks[1], targets[1]])}
            goals.append(goal2)
            goal3 = {
                GroundAtom(self._Covers, [blocks[0], targets[0]]),
                GroundAtom(self._Covers, [blocks[1], targets[1]]),
            }
            goals.append(goal3)

        for i in range(num):
            init = self._create_initial_state(blocks, targets, rng)
            assert init.get_objects(self._block_type) == blocks
            assert init.get_objects(self._target_type) == targets
            tasks.append(EnvironmentTask(init, goals[i % len(goals)]))

        return self._add_pybullet_state_to_tasks(tasks)

    def _create_initial_state(
        self, blocks: List[Object], targets: List[Object], rng: np.random.Generator
    ) -> State:
        """Create a random initial state for the cover domain."""
        data: Dict[Object, Array] = {}

        assert len(self._config.cover_block_widths) == len(blocks)
        for block, width in zip(blocks, self._config.cover_block_widths):
            while True:
                pose = rng.uniform(width / 2, 1.0 - width / 2)
                if not self._any_intersection(pose, width, data):
                    break
            # [is_block, is_target, width, pose, grasp]
            data[block] = np.array([1.0, 0.0, width, pose, -1.0])

        assert len(self._config.cover_target_widths) == len(targets)
        for target, width in zip(targets, self._config.cover_target_widths):
            while True:
                pose = rng.uniform(width / 2, 1.0 - width / 2)
                if not self._any_intersection(pose, width, data, larger_gap=True):
                    break
            # [is_block, is_target, width, pose]
            data[target] = np.array([0.0, 1.0, width, pose])

        # [hand, pose_x, pose_z]
        data[self._robot] = np.array([0.5, self.workspace_x, self.workspace_z])

        state = State(data)

        # Allow some chance of holding a block in the initial state.
        if rng.uniform() < self._config.cover_initial_holding_prob:
            block = blocks[rng.choice(len(blocks))]
            block_pose = state.get(block, "pose")
            pick_pose = block_pose
            if self._initial_pick_offsets:
                offset = rng.choice(self._initial_pick_offsets)
                assert (
                    -1.0 < offset < 1.0
                ), "initial pick offset should be between -1 and 1"
                pick_pose += state.get(block, "width") * offset / 2.0
            state.set(self._robot, "hand", pick_pose)
            state.set(block, "grasp", pick_pose - block_pose)

        return state
