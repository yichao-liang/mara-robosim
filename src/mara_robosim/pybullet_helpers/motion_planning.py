"""Motion Planning in PyBullet."""

from __future__ import annotations

from typing import Collection, Iterator, Optional, Sequence

import numpy as np
import pybullet as p
from numpy.typing import NDArray

from mara_robosim.pybullet_helpers.joint import JointPositions
from mara_robosim.pybullet_helpers.link import get_link_state
from mara_robosim.pybullet_helpers.robots import SingleArmPyBulletRobot
from mara_robosim.utils import BiRRT


def run_motion_planning(
    robot: SingleArmPyBulletRobot,
    initial_positions: JointPositions,
    target_positions: JointPositions,
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object: Optional[int] = None,
    base_link_to_held_obj: Optional[NDArray] = None,
    birrt_num_attempts: int = 10,
    birrt_num_iters: int = 100,
    birrt_smooth_amt: int = 50,
    birrt_extend_num_interp: int = 10,
    birrt_path_subsample_ratio: int = 1,
) -> Optional[Sequence[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Note that this function changes the state of the robot.

    Parameters
    ----------
    birrt_num_attempts : int
        Number of BiRRT attempts.
    birrt_num_iters : int
        Number of iterations per BiRRT attempt.
    birrt_smooth_amt : int
        Number of smoothing iterations.
    birrt_extend_num_interp : int
        Number of interpolation steps per extend.
    birrt_path_subsample_ratio : int
        Subsample ratio for the final path (1 = no subsampling).
    """
    rng = np.random.default_rng(seed)
    joint_space = robot.action_space
    joint_space.seed(seed)
    num_interp = birrt_extend_num_interp

    def _sample_fn(pt: JointPositions) -> JointPositions:
        new_pt: JointPositions = list(joint_space.sample())
        # Don't change the fingers.
        new_pt[robot.left_finger_joint_idx] = pt[robot.left_finger_joint_idx]
        new_pt[robot.right_finger_joint_idx] = pt[robot.right_finger_joint_idx]
        return new_pt

    def _set_state(pt: JointPositions) -> None:
        robot.set_joints(pt)
        if held_object is not None:
            assert base_link_to_held_obj is not None
            world_to_base_link = get_link_state(
                robot.robot_id,
                robot.end_effector_id,
                physics_client_id=physics_client_id,
            ).com_pose
            world_to_held_obj = p.multiplyTransforms(
                world_to_base_link[0],
                world_to_base_link[1],
                base_link_to_held_obj[0],
                base_link_to_held_obj[1],
            )
            p.resetBasePositionAndOrientation(
                held_object,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=physics_client_id,
            )

    def _extend_fn(
        pt1: JointPositions, pt2: JointPositions
    ) -> Iterator[JointPositions]:
        pt1_arr = np.array(pt1)
        pt2_arr = np.array(pt2)
        num = int(np.ceil(max(abs(pt1_arr - pt2_arr)))) * num_interp
        if num == 0:
            yield pt2
        for i in range(1, num + 1):
            yield list(pt1_arr * (1 - i / num) + pt2_arr * i / num)

    def _collision_fn(pt: JointPositions) -> bool:
        _set_state(pt)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        for body in collision_bodies:
            if p.getContactPoints(
                robot.robot_id, body, physicsClientId=physics_client_id
            ):
                return True
            if held_object is not None and p.getContactPoints(
                held_object, body, physicsClientId=physics_client_id
            ):
                return True
        return False

    def _distance_fn(from_pt: JointPositions, to_pt: JointPositions) -> float:
        # NOTE: only using positions to calculate distance. Should use
        # orientations as well in the near future.
        from_ee = robot.forward_kinematics(from_pt).position
        to_ee = robot.forward_kinematics(to_pt).position
        return sum(np.subtract(from_ee, to_ee) ** 2)

    birrt = BiRRT(
        _sample_fn,
        _extend_fn,
        _collision_fn,
        _distance_fn,
        rng,
        num_attempts=birrt_num_attempts,
        num_iters=birrt_num_iters,
        smooth_amt=birrt_smooth_amt,
    )

    path = birrt.query(initial_positions, target_positions)
    if path is not None and birrt_path_subsample_ratio > 1:
        ratio = birrt_path_subsample_ratio
        last = path[-1]
        path = [path[i] for i in range(0, len(path), ratio)]
        # Always include the final waypoint.
        if path[-1] is not last:
            path.append(last)
    return path
