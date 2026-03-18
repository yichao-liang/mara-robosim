"""Handles the creation of robots."""

from typing import Dict, Optional, Type

from mara_robosim.pybullet_helpers.geometry import Pose, Pose3D
from mara_robosim.pybullet_helpers.robots.fetch import FetchPyBulletRobot
from mara_robosim.pybullet_helpers.robots.mobile_fetch import MobileFetchPyBulletRobot
from mara_robosim.pybullet_helpers.robots.panda import PandaPyBulletRobot
from mara_robosim.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot

# Note: these are static base poses which suffice for the current environments.
_ROBOT_TO_BASE_POSE: Dict[str, Pose] = {
    "fetch": Pose(position=(0.75, 0.7441, 0.0)),
    "mobile_fetch": Pose(position=(0.75, 0.7441, 0.0)),
    "panda": Pose(position=(0.8, 0.7441, 0.195)),
}

_ROBOT_TO_CLS: Dict[str, Type[SingleArmPyBulletRobot]] = {
    "fetch": FetchPyBulletRobot,
    "mobile_fetch": MobileFetchPyBulletRobot,
    "panda": PandaPyBulletRobot,
}

# Used if home position is not specified during robot creation.
_DEFAULT_EE_HOME_POSITION: Pose3D = (1.35, 0.6, 0.7)


def create_single_arm_pybullet_robot(
    robot_name: str,
    physics_client_id: int,
    ee_home_pose: Optional[Pose] = None,
    base_pose: Optional[Pose] = None,
    ee_orientation: Optional[tuple] = None,
) -> SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot.

    Parameters
    ----------
    robot_name : str
        Name of the robot (e.g. "fetch", "panda", "mobile_fetch").
    physics_client_id : int
        PyBullet physics client ID.
    ee_home_pose : Optional[Pose]
        End-effector home pose. If None, a default is constructed using
        ``ee_orientation``.
    base_pose : Optional[Pose]
        Base pose for the robot. If None, a built-in default is used.
    ee_orientation : Optional[tuple]
        Quaternion (x, y, z, w) for the default end-effector orientation.
        Only used when ``ee_home_pose`` is None. If both ``ee_home_pose``
        and ``ee_orientation`` are None, a default identity orientation
        is used.
    """
    if robot_name not in _ROBOT_TO_CLS:
        raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")
    if ee_home_pose is None:
        if ee_orientation is None:
            ee_orientation = (0.0, 0.0, 0.0, 1.0)
        ee_home_pose = Pose(_DEFAULT_EE_HOME_POSITION, ee_orientation)
    if base_pose is None:
        assert (
            robot_name in _ROBOT_TO_BASE_POSE
        ), f"Base pose not specified for robot {robot_name}."
        base_pose = _ROBOT_TO_BASE_POSE[robot_name]
    cls = _ROBOT_TO_CLS[robot_name]
    return cls(ee_home_pose, physics_client_id, base_pose=base_pose)
