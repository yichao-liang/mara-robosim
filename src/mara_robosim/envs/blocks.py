"""PyBullet Blocks environment.

A robot manipulates blocks on a table -- picking, stacking, and placing
them.  This file merges the domain logic from the original
``predicators`` BlocksEnv (types, predicates, task generation, 2-D
transition helpers) with the PyBullet-specific code from
PyBulletBlocksEnv (initialisation, feature extraction, physics step).
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from mara_robosim.config import BlocksConfig, PyBulletConfig
from mara_robosim.envs.base_env import PyBulletEnv, create_pybullet_block
from mara_robosim.pybullet_helpers.geometry import Pose3D, Quaternion
from mara_robosim.pybullet_helpers.objects import create_object
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


class PyBulletBlocksEnv(PyBulletEnv):
    """PyBullet Blocks domain.

    Inherits only from :class:`PyBulletEnv` (the mara-robosim base). All
    domain logic that previously lived in ``BlocksEnv`` (types,
    predicates, task generation, 2-D simulate helpers) is merged
    directly into this class.
    """

    # ------------------------------------------------------------------
    # Domain-specific class variables (merged from BlocksEnv)
    # ------------------------------------------------------------------
    table_height: ClassVar[float] = 0.4

    # Workspace bounds (block-centre positions, not table edges).
    x_lb: ClassVar[float] = 1.325
    x_ub: ClassVar[float] = 1.375
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1

    pick_z: ClassVar[float] = 0.7
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = pick_z

    # Tolerances
    held_tol: ClassVar[float] = 0.5
    pick_tol: ClassVar[float] = 0.0001
    on_tol: ClassVar[float] = 0.01
    collision_padding: ClassVar[float] = 2.0

    # Domain-specific defaults are now in BlocksConfig (self._config).

    # ------------------------------------------------------------------
    # PyBullet-specific class variables
    # ------------------------------------------------------------------
    _camera_target: ClassVar[Pose3D] = (1.65, 0.75, 0.62)

    # Table geometry
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, table_height / 2)
    _table_orientation: ClassVar[Quaternion] = (0.0, 0.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self, config: Optional[BlocksConfig] = None, use_gui: bool = True
    ) -> None:
        # Resolve the config early so _create_blocks() can read it before
        # super().__init__() (which calls _store_pybullet_bodies and needs
        # self._blocks to exist).
        config = BlocksConfig._upgrade(config or BlocksConfig())
        self._config: BlocksConfig = config  # narrow type for mypy

        # Types (merged from BlocksEnv)
        self._block_type = Type(
            "block",
            ["pose_x", "pose_y", "pose_z", "held", "color_r", "color_g", "color_b"],
        )
        self._robot_type = Type("robot", ["pose_x", "pose_y", "pose_z", "fingers"])

        # Static objects
        self._robot = Object("robby", self._robot_type)
        self._blocks: List[Object] = []
        self._create_blocks()

        # Call the base PyBulletEnv constructor (sets up physics, robot, etc.)
        super().__init__(config=self._config, use_gui=use_gui)

        # Predicates (merged from BlocksEnv)
        self._On = Predicate("On", [self._block_type, self._block_type], self._On_holds)
        self._OnTable = Predicate("OnTable", [self._block_type], self._OnTable_holds)
        self._GripperOpen = Predicate(
            "GripperOpen", [self._robot_type], self._GripperOpen_holds
        )
        self._Holding = Predicate("Holding", [self._block_type], self._Holding_holds)
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)

        # PyBullet bookkeeping
        self._block_id_to_block: Dict[int, Object] = {}
        self._prev_held_obj_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_blocks"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._On, self._OnTable, self._GripperOpen, self._Holding, self._Clear}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        if self._config.holding_goals:
            return {self._Holding}
        return {self._On, self._OnTable}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._robot_type}

    # ------------------------------------------------------------------
    # Block creation helper (merged from BlocksEnv)
    # ------------------------------------------------------------------

    def _create_blocks(self) -> None:
        num_blocks = max(
            max(self._config.num_blocks_train), max(self._config.num_blocks_test)
        )
        for i in range(num_blocks):
            block = Object(f"block{i}", self._block_type)
            self._blocks.append(block)

    # ==================================================================
    # PyBullet Hooks
    # ==================================================================

    @classmethod
    def initialize_pybullet(
        cls,
        using_gui: bool,
        config: Optional[PyBulletConfig] = None,
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Create the plane, table, debug lines, and maximum number of
        blocks."""
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui, config=config
        )

        # Load the table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls._table_pose,
            orientation=cls._table_orientation,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id

        # Optional debug lines
        cfg = config or BlocksConfig()
        if cfg.draw_debug and using_gui:  # pragma: no cover
            cls._draw_table_workspace_debug_lines(physics_client_id)

        # Create the maximum number of blocks
        num_blocks_train = getattr(cfg, "num_blocks_train", (3, 4))
        num_blocks_test = getattr(cfg, "num_blocks_test", (5, 6))
        num_blocks = max(max(num_blocks_train), max(num_blocks_test))
        block_ids = []
        block_size = getattr(cfg, "block_size", 0.045)
        for i in range(num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            half_extents = (block_size / 2.0, block_size / 2.0, block_size / 2.0)
            block_id = create_pybullet_block(
                color=color,
                half_extents=half_extents,
                mass=cls._obj_mass,
                friction=cls._obj_friction,
                physics_client_id=physics_client_id,
            )
            block_ids.append(block_id)
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to table and block IDs."""
        self._table_id = pybullet_bodies["table_id"]
        self._block_ids: List[int] = pybullet_bodies["block_ids"]
        for blk, bid in zip(self._blocks, self._block_ids):
            blk.id = bid

    def _create_task_specific_objects(self, state: State) -> None:
        """No additional environment assets needed per-task."""

    def _reset_custom_env_state(self, state: State) -> None:
        """After the parent ``_reset_state()`` has reset the robot, set the
        block positions/colours and handle constraints for any 'held' block."""
        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block.clear()

        # Place the relevant blocks
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            self._block_id_to_block[block_id] = block_obj

            # Position/orientation from the state's block features
            bx = state.get(block_obj, "pose_x")
            by = state.get(block_obj, "pose_y")
            bz = state.get(block_obj, "pose_z")
            p.resetBasePositionAndOrientation(
                block_id,
                [bx, by, bz],
                self._default_orn,
                physicsClientId=self._physics_client_id,
            )

            # Update colour
            r = state.get(block_obj, "color_r")
            g = state.get(block_obj, "color_g")
            b = state.get(block_obj, "color_b")
            p.changeVisualShape(
                block_id,
                linkIndex=-1,
                rgbaColor=(r, g, b, 1.0),
                physicsClientId=self._physics_client_id,
            )

        # If there is a held block, create the constraint
        held_block = self._get_held_block(state)
        if held_block is not None:
            self._force_grasp_object(held_block)

        # Teleport any leftover blocks out of view
        block_size = self._config.block_size
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            p.resetBasePositionAndOrientation(
                block_id,
                [oov_x, oov_y, i * block_size],
                self._default_orn,
                physicsClientId=self._physics_client_id,
            )

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Called by the parent class when constructing the ``PyBulletState``.

        The block features use non-standard names (``pose_x`` instead of
        ``x``, ``held`` instead of ``is_held``, etc.) so they all flow
        through here rather than the generic extraction in
        ``_get_state()``.
        """
        if obj.type == self._block_type:
            block_id = None
            for bid, block_obj in self._block_id_to_block.items():
                if block_obj == obj:
                    block_id = bid
                    break
            if block_id is None:
                raise ValueError(f"Object {obj} not found in _block_id_to_block")

            # Position from PyBullet
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id
            )

            if feature == "pose_x":
                return bx
            if feature == "pose_y":
                return by
            if feature == "pose_z":
                return bz
            if feature == "held":
                return 1.0 if block_id == self._held_obj_id else 0.0
            if feature in ("color_r", "color_g", "color_b"):
                visual_data = p.getVisualShapeData(
                    block_id, physicsClientId=self._physics_client_id
                )[0]
                cr, cg, cb, _ = visual_data[7]
                if feature == "color_r":
                    return cr
                if feature == "color_g":
                    return cg
                return cb  # color_b

            raise ValueError(f"Unknown block feature: {feature}")

        raise ValueError(f"Unknown object type {obj.type} or feature {feature}")

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return the IDs of blocks for which we check 'held' contact."""
        return list(self._block_id_to_block.keys())

    # ------------------------------------------------------------------
    # step() override
    # ------------------------------------------------------------------

    def step(self, action: Action, render_obs: bool = False) -> State:
        self._prev_held_obj_id = self._held_obj_id
        next_state = super().step(action, render_obs=render_obs)

        if self._config.high_towers_are_unstable:
            self._apply_force_to_high_towers(next_state)
            next_state = self._get_state()
            self._current_observation = next_state

        return next_state

    # ==================================================================
    # Domain-Specific Logic
    # ==================================================================

    def _force_grasp_object(self, block: Object) -> None:
        """Manually create a fixed constraint for a block that is marked 'held'
        in the State.

        Called from ``_reset_custom_env_state()``.
        """
        block_id = None
        for bid, block_obj in self._block_id_to_block.items():
            if block_obj == block:
                block_id = bid
                break
        if block_id is None:
            return
        held_obj_id = self._detect_held_object()
        if held_obj_id != block_id:
            self._held_obj_id = block_id
        self._create_grasp_constraint()

    def _apply_force_to_high_towers(self, state: State) -> None:
        """Apply downward force to blocks that form towers of height >= 3."""
        just_released_obj = self._just_released_object(state)
        if just_released_obj is None:
            return
        if self._count_block_height(state, just_released_obj) >= 2:
            force = [0, -100, 0]
            pos = p.getBasePositionAndOrientation(
                just_released_obj.id, physicsClientId=self._physics_client_id
            )[0]
            p.applyExternalForce(
                just_released_obj.id,
                -1,
                force,
                pos,
                p.WORLD_FRAME,
                physicsClientId=self._physics_client_id,
            )

    def _just_released_object(self, state: State) -> Optional[Object]:
        """Check if we just released an object in this step."""
        if self._held_obj_id is None and self._prev_held_obj_id is not None:
            for block_obj in state.get_objects(self._block_type):
                if block_obj.id == self._prev_held_obj_id:
                    return block_obj
        return None

    # ==================================================================
    # Predicate Classifiers (merged from BlocksEnv)
    # ==================================================================

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        if (
            state.get(block1, "held") >= self.held_tol
            or state.get(block2, "held") >= self.held_tol
        ):
            return False
        x1 = state.get(block1, "pose_x")
        y1 = state.get(block1, "pose_y")
        z1 = state.get(block1, "pose_z")
        x2 = state.get(block2, "pose_x")
        y2 = state.get(block2, "pose_y")
        z2 = state.get(block2, "pose_z")
        return np.allclose(
            [x1, y1, z1], [x2, y2, z2 + self._config.block_size], atol=self.on_tol
        )

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (block,) = objects
        z = state.get(block, "pose_z")
        desired_z = self.table_height + self._config.block_size * 0.5
        return (state.get(block, "held") < self.held_tol) and (
            desired_z - self.on_tol < z < desired_z + self.on_tol
        )

    def _GripperOpen_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (robot,) = objects
        rf = state.get(robot, "fingers")
        return rf == self.open_fingers

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (block,) = objects
        held_block = self._get_held_block(state)
        if held_block is None:
            return False
        return held_block == block

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        if self._Holding_holds(state, objects):
            return False
        (block,) = objects
        for other_block in state:
            if other_block.type != self._block_type:
                continue
            if self._On_holds(state, [other_block, block]):
                return False
        return True

    # ==================================================================
    # Block Query Helpers (merged from BlocksEnv)
    # ==================================================================

    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            if state.get(block, "held") >= self.held_tol:
                return block
        return None

    def _block_is_clear(self, block: Object, state: State) -> bool:
        return self._Clear_holds(state, [block])

    def _get_block_at_xyz(
        self, state: State, x: float, y: float, z: float
    ) -> Optional[Object]:
        close_blocks = []
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            block_pose = np.array(
                [
                    state.get(block, "pose_x"),
                    state.get(block, "pose_y"),
                    state.get(block, "pose_z"),
                ]
            )
            if np.allclose([x, y, z], block_pose, atol=self.pick_tol):
                dist = float(np.linalg.norm(np.array([x, y, z]) - block_pose))
                close_blocks.append((block, dist))
        if not close_blocks:
            return None
        return min(close_blocks, key=lambda item: item[1])[0]

    def _get_highest_block_below(
        self, state: State, x: float, y: float, z: float
    ) -> Optional[Object]:
        blocks_here: List[Tuple[Object, float]] = []
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            block_pose = np.array(
                [
                    state.get(block, "pose_x"),
                    state.get(block, "pose_y"),
                ]
            )
            block_z = state.get(block, "pose_z")
            if (
                np.allclose([x, y], block_pose, atol=self.pick_tol)
                and block_z < z - self.pick_tol
            ):
                blocks_here.append((block, block_z))
        if not blocks_here:
            return None
        return max(blocks_here, key=lambda item: item[1])[0]

    def _count_block_height(self, state: State, block: Object) -> int:
        """Count the height of the block (number of blocks it is on)."""
        height = 0
        current_block = block
        blocks = state.get_objects(self._block_type)
        while True:
            below_blocks = [
                b for b in blocks if self._On_holds(state, [current_block, b])
            ]
            if not below_blocks:
                break
            current_block = below_blocks[0]
            height += 1
        return height

    def _table_xy_is_clear(
        self, x: float, y: float, existing_xys: Set[Tuple[float, float]]
    ) -> bool:
        if all(
            abs(x - other_x) > self.collision_padding * self._config.block_size
            for other_x, _ in existing_xys
        ):
            return True
        if all(
            abs(y - other_y) > self.collision_padding * self._config.block_size
            for _, other_y in existing_xys
        ):
            return True
        return False

    # ==================================================================
    # 2-D Simulate (merged from BlocksEnv)
    # ==================================================================
    # NOTE: The PyBullet base already has a concrete ``simulate()`` that
    # resets state and calls ``step()``.  The transitions below are kept
    # for any callers that need a fast, non-physics-based forward model
    # (e.g. planning with an abstract simulator).

    def simulate_abstract(self, state: State, action: Action) -> State:
        """Non-physics 2-D transition model (from BlocksEnv.simulate)."""
        x, y, z, fingers = action.arr
        if fingers < 0.5:
            return self._transition_pick(state, x, y, z)
        if z < self.table_height + self._config.block_size:
            return self._transition_putontable(state, x, y, z)
        return self._transition_stack(state, x, y, z)

    def _transition_pick(self, state: State, x: float, y: float, z: float) -> State:
        next_state = state.copy()
        if not self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_block_at_xyz(state, x, y, z)
        if block is None:
            return next_state
        if not self._block_is_clear(block, state):
            return next_state
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", self.pick_z)
        next_state.set(block, "held", 1.0)
        next_state.set(self._robot, "fingers", self.closed_fingers)
        return next_state

    def _transition_putontable(
        self, state: State, x: float, y: float, z: float
    ) -> State:
        next_state = state.copy()
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_held_block(state)
        assert block is not None
        poses = [
            [
                state.get(b, "pose_x"),
                state.get(b, "pose_y"),
                state.get(b, "pose_z"),
            ]
            for b in state
            if b.is_instance(self._block_type)
        ]
        existing_xys = {(float(pos[0]), float(pos[1])) for pos in poses}
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z)
        next_state.set(block, "held", 0.0)
        next_state.set(self._robot, "fingers", self.open_fingers)
        return next_state

    def _transition_stack(self, state: State, x: float, y: float, z: float) -> State:
        next_state = state.copy()
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_held_block(state)
        assert block is not None
        other_block = self._get_highest_block_below(state, x, y, z)
        if other_block is None:
            return next_state
        if block == other_block:
            return next_state
        if not self._block_is_clear(other_block, state):
            return next_state
        cur_x = state.get(other_block, "pose_x")
        cur_y = state.get(other_block, "pose_y")
        cur_z = state.get(other_block, "pose_z")
        next_state.set(block, "pose_x", cur_x)
        next_state.set(block, "pose_y", cur_y)
        next_state.set(block, "pose_z", cur_z + self._config.block_size)
        next_state.set(block, "held", 0.0)
        next_state.set(self._robot, "fingers", self.open_fingers)
        return next_state

    # ==================================================================
    # Task Generation (merged from BlocksEnv + PyBulletBlocksEnv)
    # ==================================================================

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num_tasks=self._config.num_train_tasks,
            possible_num_blocks=self._config.num_blocks_train,
            rng=self._train_rng,
        )

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num_tasks=self._config.num_test_tasks,
            possible_num_blocks=self._config.num_blocks_test,
            rng=self._test_rng,
        )

    def _get_tasks(
        self, num_tasks: int, possible_num_blocks: List[int], rng: np.random.Generator
    ) -> List[EnvironmentTask]:
        tasks: List[EnvironmentTask] = []
        for _ in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks)
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            while True:  # repeat until goal is not satisfied
                goal = self._sample_goal_from_piles(num_blocks, piles, rng)
                if not all(goal_atom.holds(init_state) for goal_atom in goal):
                    break
            tasks.append(EnvironmentTask(init_state, goal))
        return self._add_pybullet_state_to_tasks(tasks)

    def _sample_initial_piles(
        self, num_blocks: int, rng: np.random.Generator
    ) -> List[List[Object]]:
        piles: List[List[Object]] = []
        for block_num in range(num_blocks):
            block = self._blocks[block_num]
            if block_num == 0 or rng.uniform() < 0.2:
                piles.append([])
            piles[-1].append(block)
        return piles

    def _sample_state_from_piles(
        self, piles: List[List[Object]], rng: np.random.Generator
    ) -> State:
        data: Dict[Object, Array] = {}
        block_to_pile_idx: Dict[Object, Tuple[int, int]] = {}
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
            z = self.table_height + self._config.block_size * (0.5 + pile_j)
            cr, cg, cb = rng.uniform(size=3)
            # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b]
            data[block] = np.array([x, y, z, 0.0, cr, cg, cb])
        # Robot: [pose_x, pose_y, pose_z, fingers]
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = self.open_fingers
        data[self._robot] = np.array([rx, ry, rz, rf], dtype=np.float32)
        return State(data)

    def _sample_goal_from_piles(
        self, num_blocks: int, piles: List[List[Object]], rng: np.random.Generator
    ) -> Set[GroundAtom]:
        if self._config.holding_goals:
            pile_idx = rng.choice(len(piles))
            top_block = piles[pile_idx][-1]
            return {GroundAtom(self._Holding, [top_block])}
        # Sample goal pile that is different from initial
        while True:
            goal_piles = self._sample_initial_piles(num_blocks, rng)
            if goal_piles != piles:
                break
        goal_atoms: Set[GroundAtom] = set()
        for pile in goal_piles:
            goal_atoms.add(GroundAtom(self._OnTable, [pile[0]]))
            if len(pile) == 1:
                continue
            for block1, block2 in zip(pile[1:], pile[:-1]):
                goal_atoms.add(GroundAtom(self._On, [block1, block2]))
        return goal_atoms

    def _sample_initial_pile_xy(
        self, rng: np.random.Generator, existing_xys: Set[Tuple[float, float]]
    ) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.x_lb, self.x_ub)
            y = rng.uniform(self.y_lb, self.y_ub)
            if self._table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_table_workspace_debug_lines(
        physics_client_id: int,
    ) -> None:  # pragma: no cover
        """Draw red lines marking the workspace on the table."""
        x_lb = PyBulletBlocksEnv.x_lb
        x_ub = PyBulletBlocksEnv.x_ub
        y_lb = PyBulletBlocksEnv.y_lb
        y_ub = PyBulletBlocksEnv.y_ub
        z = PyBulletBlocksEnv.table_height

        p.addUserDebugLine(
            [x_lb, y_lb, z],
            [x_ub, y_lb, z],
            [1.0, 0.0, 0.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugLine(
            [x_lb, y_ub, z],
            [x_ub, y_ub, z],
            [1.0, 0.0, 0.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugLine(
            [x_lb, y_lb, z],
            [x_lb, y_ub, z],
            [1.0, 0.0, 0.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
        p.addUserDebugLine(
            [x_ub, y_lb, z],
            [x_ub, y_ub, z],
            [1.0, 0.0, 0.0],
            lineWidth=5.0,
            physicsClientId=physics_client_id,
        )
