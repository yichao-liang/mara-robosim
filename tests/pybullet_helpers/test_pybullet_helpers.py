"""Tests for the pybullet_helpers module.

Focuses on pure-logic functions that do not require a running PyBullet
simulation. For code that depends on a live physics server, we verify
that the relevant symbols import successfully.
"""

import math

import numpy as np
import pybullet as p
import pytest

# ---------------------------------------------------------------------------
# Geometry: Pose, types, and utility functions
# ---------------------------------------------------------------------------


class TestPoseCreation:
    """Tests for Pose construction and field access."""

    def test_pose_with_default_orientation(self):
        """Pose created with only position should have identity quaternion."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        pose = Pose((1.0, 2.0, 3.0))
        assert pose.position == (1.0, 2.0, 3.0)
        assert pose.orientation == (0.0, 0.0, 0.0, 1.0)

    def test_pose_with_explicit_orientation(self):
        """Pose created with position and orientation should store both."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        pos = (0.5, -1.0, 0.0)
        orn = (0.0, 0.0, 0.7071068, 0.7071068)
        pose = Pose(pos, orn)
        assert pose.position == pos
        assert pose.orientation == orn

    def test_pose_is_named_tuple(self):
        """Pose supports integer indexing (NamedTuple behaviour)."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        pose = Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        assert pose[0] == (1.0, 2.0, 3.0)
        assert pose[1] == (0.0, 0.0, 0.0, 1.0)

    def test_pose_unpacking(self):
        """Pose can be unpacked into position and orientation."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        pose = Pose((4.0, 5.0, 6.0), (0.0, 0.0, 0.0, 1.0))
        pos, orn = pose
        assert pos == (4.0, 5.0, 6.0)
        assert orn == (0.0, 0.0, 0.0, 1.0)


class TestPoseIdentity:
    """Tests for Pose.identity()."""

    def test_identity_position_is_origin(self):
        from mara_robosim.pybullet_helpers.geometry import Pose

        identity = Pose.identity()
        assert identity.position == (0.0, 0.0, 0.0)

    def test_identity_orientation_is_unit_quaternion(self):
        from mara_robosim.pybullet_helpers.geometry import Pose

        identity = Pose.identity()
        assert identity.orientation == (0.0, 0.0, 0.0, 1.0)


class TestPoseFromRpy:
    """Tests for Pose.from_rpy()."""

    def test_from_rpy_zero_angles(self):
        """Zero RPY should yield identity quaternion."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        pose = Pose.from_rpy((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        assert pose.position == (1.0, 2.0, 3.0)
        np.testing.assert_allclose(pose.orientation, (0.0, 0.0, 0.0, 1.0), atol=1e-7)

    def test_from_rpy_ninety_degree_yaw(self):
        """A 90-degree yaw should produce a known quaternion."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        pose = Pose.from_rpy((0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2))
        # Expected quaternion for 90-degree yaw: (0, 0, sin(pi/4), cos(pi/4))
        expected_quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        np.testing.assert_allclose(pose.orientation, expected_quat, atol=1e-6)

    def test_from_rpy_roundtrip(self):
        """Converting from RPY and back should recover the original angles."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        rpy_in = (0.1, 0.2, 0.3)
        pose = Pose.from_rpy((0.0, 0.0, 0.0), rpy_in)
        rpy_out = pose.rpy
        np.testing.assert_allclose(rpy_out, rpy_in, atol=1e-6)


class TestPoseRpy:
    """Tests for the Pose.rpy property."""

    def test_identity_rpy_is_zero(self):
        from mara_robosim.pybullet_helpers.geometry import Pose

        rpy = Pose.identity().rpy
        np.testing.assert_allclose(rpy, (0.0, 0.0, 0.0), atol=1e-7)

    def test_rpy_known_orientation(self):
        """A pure 90-degree roll should report correctly."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        # Quaternion for 90-degree roll (rotation about X axis)
        quat = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))
        pose = Pose((0.0, 0.0, 0.0), quat)
        rpy = pose.rpy
        np.testing.assert_allclose(rpy[0], math.pi / 2, atol=1e-6)
        np.testing.assert_allclose(rpy[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(rpy[2], 0.0, atol=1e-6)


class TestPoseAllclose:
    """Tests for Pose.allclose()."""

    def test_identical_poses_are_close(self):
        from mara_robosim.pybullet_helpers.geometry import Pose

        p1 = Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        p2 = Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        assert p1.allclose(p2)

    def test_slightly_different_poses_are_close(self):
        from mara_robosim.pybullet_helpers.geometry import Pose

        p1 = Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        p2 = Pose((1.0 + 1e-8, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        assert p1.allclose(p2)

    def test_very_different_poses_are_not_close(self):
        from mara_robosim.pybullet_helpers.geometry import Pose

        p1 = Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        p2 = Pose((10.0, 20.0, 30.0), (0.0, 0.0, 0.0, 1.0))
        assert not p1.allclose(p2)

    def test_allclose_respects_atol(self):
        from mara_robosim.pybullet_helpers.geometry import Pose

        p1 = Pose((0.0, 0.0, 0.0))
        p2 = Pose((0.05, 0.0, 0.0))
        # Should be close with a large tolerance
        assert p1.allclose(p2, atol=0.1)
        # Should not be close with a tight tolerance
        assert not p1.allclose(p2, atol=0.01)

    def test_allclose_checks_orientation(self):
        from mara_robosim.pybullet_helpers.geometry import Pose

        p1 = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        p2 = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.5, 0.866))
        assert not p1.allclose(p2)


# ---------------------------------------------------------------------------
# Geometry: multiply_poses (uses PyBullet multiplyTransforms)
# ---------------------------------------------------------------------------


class TestMultiplyPoses:
    """Tests for multiply_poses and Pose.multiply()."""

    def test_multiply_identity(self):
        """Multiplying by identity should return the same pose."""
        from mara_robosim.pybullet_helpers.geometry import Pose, multiply_poses

        pose = Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        result = multiply_poses(pose, Pose.identity())
        assert result.allclose(pose)

    def test_multiply_identity_left(self):
        """Identity on the left should also return the same pose."""
        from mara_robosim.pybullet_helpers.geometry import Pose, multiply_poses

        pose = Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        result = multiply_poses(Pose.identity(), pose)
        assert result.allclose(pose)

    def test_multiply_two_translations(self):
        """Multiplying two pure translations should add positions."""
        from mara_robosim.pybullet_helpers.geometry import Pose, multiply_poses

        p1 = Pose((1.0, 0.0, 0.0))
        p2 = Pose((0.0, 2.0, 0.0))
        result = multiply_poses(p1, p2)
        np.testing.assert_allclose(result.position, (1.0, 2.0, 0.0), atol=1e-7)

    def test_multiply_chained(self):
        """Multiplying three poses should be associative."""
        from mara_robosim.pybullet_helpers.geometry import Pose, multiply_poses

        p1 = Pose((1.0, 0.0, 0.0))
        p2 = Pose((0.0, 1.0, 0.0))
        p3 = Pose((0.0, 0.0, 1.0))
        result = multiply_poses(p1, p2, p3)
        np.testing.assert_allclose(result.position, (1.0, 1.0, 1.0), atol=1e-7)

    def test_multiply_single_pose(self):
        """Multiplying a single pose should return that pose."""
        from mara_robosim.pybullet_helpers.geometry import Pose, multiply_poses

        pose = Pose((5.0, 6.0, 7.0), (0.0, 0.0, 0.0, 1.0))
        result = multiply_poses(pose)
        assert result.allclose(pose)

    def test_pose_multiply_method(self):
        """Pose.multiply() should delegate to multiply_poses."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        p1 = Pose((1.0, 0.0, 0.0))
        p2 = Pose((0.0, 2.0, 0.0))
        result = p1.multiply(p2)
        np.testing.assert_allclose(result.position, (1.0, 2.0, 0.0), atol=1e-7)


# ---------------------------------------------------------------------------
# Geometry: Pose.invert()
# ---------------------------------------------------------------------------


class TestPoseInvert:
    """Tests for Pose.invert()."""

    def test_invert_identity(self):
        """Inverting identity should return identity."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        identity = Pose.identity()
        inv = identity.invert()
        assert inv.allclose(identity)

    def test_invert_pure_translation(self):
        """Inverting a pure translation should negate the position."""
        from mara_robosim.pybullet_helpers.geometry import Pose

        pose = Pose((1.0, 2.0, 3.0))
        inv = pose.invert()
        np.testing.assert_allclose(inv.position, (-1.0, -2.0, -3.0), atol=1e-7)

    def test_invert_roundtrip(self):
        """pose * pose.invert() should yield identity."""
        from mara_robosim.pybullet_helpers.geometry import Pose, multiply_poses

        pose = Pose.from_rpy((1.0, 2.0, 3.0), (0.3, 0.2, 0.1))
        result = multiply_poses(pose, pose.invert())
        assert result.allclose(Pose.identity(), atol=1e-5)

    def test_invert_roundtrip_reverse_order(self):
        """pose.invert() * pose should also yield identity."""
        from mara_robosim.pybullet_helpers.geometry import Pose, multiply_poses

        pose = Pose.from_rpy((1.0, 2.0, 3.0), (0.3, 0.2, 0.1))
        result = multiply_poses(pose.invert(), pose)
        assert result.allclose(Pose.identity(), atol=1e-5)


# ---------------------------------------------------------------------------
# Geometry: matrix_from_quat
# ---------------------------------------------------------------------------


class TestMatrixFromQuat:
    """Tests for matrix_from_quat()."""

    def test_identity_quaternion_gives_identity_matrix(self):
        from mara_robosim.pybullet_helpers.geometry import matrix_from_quat

        mat = matrix_from_quat((0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(mat, np.eye(3), atol=1e-7)

    def test_output_shape(self):
        from mara_robosim.pybullet_helpers.geometry import matrix_from_quat

        mat = matrix_from_quat((0.0, 0.0, 0.7071068, 0.7071068))
        assert mat.shape == (3, 3)

    def test_rotation_matrix_is_orthogonal(self):
        """A rotation matrix R satisfies R^T R = I."""
        from mara_robosim.pybullet_helpers.geometry import matrix_from_quat

        quat = (0.1, 0.2, 0.3, 0.9274)  # approximately unit quaternion
        # Normalise to be safe
        quat_arr = np.array(quat)
        quat_arr = quat_arr / np.linalg.norm(quat_arr)
        mat = matrix_from_quat(tuple(quat_arr))
        np.testing.assert_allclose(mat.T @ mat, np.eye(3), atol=1e-6)

    def test_determinant_is_one(self):
        """A proper rotation matrix has determinant +1."""
        from mara_robosim.pybullet_helpers.geometry import matrix_from_quat

        mat = matrix_from_quat((0.0, 0.0, 0.7071068, 0.7071068))
        np.testing.assert_allclose(np.linalg.det(mat), 1.0, atol=1e-6)

    def test_90_degree_yaw_rotation(self):
        """A 90-degree yaw should rotate the x-axis onto the y-axis."""
        from mara_robosim.pybullet_helpers.geometry import matrix_from_quat

        quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        mat = matrix_from_quat(quat)
        x_axis = np.array([1.0, 0.0, 0.0])
        rotated = mat @ x_axis
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Geometry: type aliases
# ---------------------------------------------------------------------------


class TestTypeAliases:
    """Verify that the type aliases are importable and usable."""

    def test_pose3d_alias(self):
        from mara_robosim.pybullet_helpers.geometry import Pose3D

        pos: Pose3D = (1.0, 2.0, 3.0)
        assert len(pos) == 3

    def test_quaternion_alias(self):
        from mara_robosim.pybullet_helpers.geometry import Quaternion

        q: Quaternion = (0.0, 0.0, 0.0, 1.0)
        assert len(q) == 4

    def test_rollpitchyaw_alias(self):
        from mara_robosim.pybullet_helpers.geometry import RollPitchYaw

        rpy: RollPitchYaw = (0.0, 0.0, 0.0)
        assert len(rpy) == 3


# ---------------------------------------------------------------------------
# Joint: JointInfo properties (pure logic, no physics server)
# ---------------------------------------------------------------------------


class TestJointInfo:
    """Tests for JointInfo NamedTuple properties that are pure logic."""

    def _make_joint_info(
        self, joint_type=p.JOINT_REVOLUTE, lower_limit=0.0, upper_limit=1.0
    ):
        """Helper to create a JointInfo with sensible defaults."""
        from mara_robosim.pybullet_helpers.joint import JointInfo

        return JointInfo(
            jointIndex=0,
            jointName="test_joint",
            jointType=joint_type,
            qIndex=0,
            uIndex=0,
            flags=0,
            jointDamping=0.0,
            jointFriction=0.0,
            jointLowerLimit=lower_limit,
            jointUpperLimit=upper_limit,
            jointMaxForce=100.0,
            jointMaxVelocity=1.0,
            linkName="test_link",
            jointAxis=(0.0, 0.0, 1.0),
            parentFramePos=(0.0, 0.0, 0.0),
            parentFrameOrn=(0.0, 0.0, 0.0, 1.0),
            parentIndex=-1,
        )

    def test_is_fixed_true(self):
        ji = self._make_joint_info(joint_type=p.JOINT_FIXED)
        assert ji.is_fixed

    def test_is_fixed_false_for_revolute(self):
        ji = self._make_joint_info(joint_type=p.JOINT_REVOLUTE)
        assert not ji.is_fixed

    def test_is_movable_for_revolute(self):
        ji = self._make_joint_info(joint_type=p.JOINT_REVOLUTE)
        assert ji.is_movable

    def test_is_movable_false_for_fixed(self):
        ji = self._make_joint_info(joint_type=p.JOINT_FIXED)
        assert not ji.is_movable

    def test_is_circular_when_upper_less_than_lower(self):
        """PyBullet uses upper < lower to indicate circular joints."""
        ji = self._make_joint_info(
            joint_type=p.JOINT_REVOLUTE, lower_limit=0.0, upper_limit=-1.0
        )
        assert ji.is_circular

    def test_is_not_circular_for_normal_limits(self):
        ji = self._make_joint_info(lower_limit=-1.0, upper_limit=1.0)
        assert not ji.is_circular

    def test_is_not_circular_for_fixed_joint(self):
        """Fixed joints are never circular, even with weird limits."""
        ji = self._make_joint_info(
            joint_type=p.JOINT_FIXED, lower_limit=0.0, upper_limit=-1.0
        )
        assert not ji.is_circular

    def test_violates_limit_within_range(self):
        ji = self._make_joint_info(lower_limit=-1.0, upper_limit=1.0)
        assert not ji.violates_limit(0.0)

    def test_violates_limit_below_lower(self):
        ji = self._make_joint_info(lower_limit=-1.0, upper_limit=1.0)
        assert ji.violates_limit(-2.0)

    def test_violates_limit_above_upper(self):
        ji = self._make_joint_info(lower_limit=-1.0, upper_limit=1.0)
        assert ji.violates_limit(2.0)

    def test_violates_limit_circular_always_false(self):
        """Circular joints never violate limits."""
        ji = self._make_joint_info(
            joint_type=p.JOINT_REVOLUTE, lower_limit=0.0, upper_limit=-1.0
        )
        assert not ji.violates_limit(999.0)

    def test_violates_limit_with_tolerance(self):
        """The tolerance widens the violation zone: a value is considered.

        violating if ``lower > value - tol`` or ``value + tol > upper``.
        """
        ji = self._make_joint_info(lower_limit=0.0, upper_limit=1.0)
        # value=0.5 is well inside [0, 1] even with tol=0.1
        assert not ji.violates_limit(0.5, tol=0.1)
        # value=0.05 triggers lower > value - tol  (0 > -0.05) => True
        assert ji.violates_limit(0.05, tol=0.1)
        # value=0.95 triggers value + tol > upper  (1.05 > 1.0) => True
        assert ji.violates_limit(0.95, tol=0.1)
        # With a tiny tolerance the center value should be fine
        assert not ji.violates_limit(0.5, tol=0.001)

    def test_joint_info_integer_indexing(self):
        """JointInfo (NamedTuple) supports integer indexing."""
        ji = self._make_joint_info()
        assert ji[0] == 0  # jointIndex
        assert ji[1] == "test_joint"  # jointName


# ---------------------------------------------------------------------------
# Joint: JointState
# ---------------------------------------------------------------------------


class TestJointState:
    """Tests for JointState NamedTuple."""

    def test_joint_state_creation(self):
        from mara_robosim.pybullet_helpers.joint import JointState

        js = JointState(
            jointPosition=0.5,
            jointVelocity=0.1,
            jointReactionForces=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            appliedJointMotorTorque=0.0,
        )
        assert js.jointPosition == 0.5
        assert js.jointVelocity == 0.1

    def test_joint_state_integer_indexing(self):
        from mara_robosim.pybullet_helpers.joint import JointState

        js = JointState(1.0, 2.0, (0,) * 6, 3.0)
        assert js[0] == 1.0
        assert js[3] == 3.0


# ---------------------------------------------------------------------------
# Link: LinkState pure-logic properties
# ---------------------------------------------------------------------------


class TestLinkState:
    """Tests for LinkState NamedTuple properties."""

    def test_com_pose(self):
        from mara_robosim.pybullet_helpers.geometry import Pose
        from mara_robosim.pybullet_helpers.link import LinkState

        ls = LinkState(
            linkWorldPosition=(1.0, 2.0, 3.0),
            linkWorldOrientation=(0.0, 0.0, 0.0, 1.0),
            localInertialFramePosition=(0.0, 0.0, 0.0),
            localInertialFrameOrientation=(0.0, 0.0, 0.0, 1.0),
            worldLinkFramePosition=(4.0, 5.0, 6.0),
            worldLinkFrameOrientation=(0.0, 0.0, 0.0, 1.0),
        )
        com = ls.com_pose
        assert isinstance(com, Pose)
        assert com.position == (1.0, 2.0, 3.0)

    def test_pose_property(self):
        from mara_robosim.pybullet_helpers.geometry import Pose
        from mara_robosim.pybullet_helpers.link import LinkState

        ls = LinkState(
            linkWorldPosition=(1.0, 2.0, 3.0),
            linkWorldOrientation=(0.0, 0.0, 0.0, 1.0),
            localInertialFramePosition=(0.0, 0.0, 0.0),
            localInertialFrameOrientation=(0.0, 0.0, 0.0, 1.0),
            worldLinkFramePosition=(4.0, 5.0, 6.0),
            worldLinkFrameOrientation=(0.1, 0.0, 0.0, 0.995),
        )
        pose = ls.pose
        assert isinstance(pose, Pose)
        assert pose.position == (4.0, 5.0, 6.0)
        assert pose.orientation == (0.1, 0.0, 0.0, 0.995)

    def test_base_link_constant(self):
        from mara_robosim.pybullet_helpers.link import BASE_LINK

        assert BASE_LINK == -1


# ---------------------------------------------------------------------------
# IKFast: IKFastInfo
# ---------------------------------------------------------------------------


class TestIKFastInfo:
    """Tests for IKFastInfo NamedTuple."""

    def test_ikfast_info_creation(self):
        from mara_robosim.pybullet_helpers.ikfast import IKFastInfo

        info = IKFastInfo(
            module_dir="/some/path",
            module_name="ikfast_mod",
            base_link="base",
            ee_link="ee",
            free_joints=["j1", "j2"],
        )
        assert info.module_name == "ikfast_mod"
        assert info.free_joints == ["j1", "j2"]


# ---------------------------------------------------------------------------
# Robots __init__: configuration dictionaries and factory guard
# ---------------------------------------------------------------------------


class TestRobotsInit:
    """Tests for the robots __init__ module configuration and validation."""

    def test_robot_to_base_pose_keys(self):
        from mara_robosim.pybullet_helpers.robots import _ROBOT_TO_BASE_POSE

        assert "fetch" in _ROBOT_TO_BASE_POSE
        assert "panda" in _ROBOT_TO_BASE_POSE
        assert "mobile_fetch" in _ROBOT_TO_BASE_POSE

    def test_robot_to_cls_keys(self):
        from mara_robosim.pybullet_helpers.robots import _ROBOT_TO_CLS

        assert "fetch" in _ROBOT_TO_CLS
        assert "panda" in _ROBOT_TO_CLS
        assert "mobile_fetch" in _ROBOT_TO_CLS

    def test_default_ee_home_position(self):
        from mara_robosim.pybullet_helpers.robots import _DEFAULT_EE_HOME_POSITION

        assert len(_DEFAULT_EE_HOME_POSITION) == 3
        assert all(isinstance(v, (int, float)) for v in _DEFAULT_EE_HOME_POSITION)

    def test_create_single_arm_raises_for_unknown_robot(self):
        from mara_robosim.pybullet_helpers.robots import (
            create_single_arm_pybullet_robot,
        )

        with pytest.raises(NotImplementedError, match="Unrecognized robot name"):
            create_single_arm_pybullet_robot("nonexistent_robot", physics_client_id=0)

    def test_base_pose_values_are_poses(self):
        from mara_robosim.pybullet_helpers.geometry import Pose
        from mara_robosim.pybullet_helpers.robots import _ROBOT_TO_BASE_POSE

        for name, pose in _ROBOT_TO_BASE_POSE.items():
            assert isinstance(pose, Pose), f"Base pose for {name} is not a Pose"
            assert len(pose.position) == 3


# ---------------------------------------------------------------------------
# InverseKinematicsError
# ---------------------------------------------------------------------------


class TestInverseKinematicsError:
    """Tests for InverseKinematicsError exception class."""

    def test_is_value_error_subclass(self):
        from mara_robosim.pybullet_helpers.inverse_kinematics import (
            InverseKinematicsError,
        )

        assert issubclass(InverseKinematicsError, ValueError)

    def test_can_be_raised_and_caught(self):
        from mara_robosim.pybullet_helpers.inverse_kinematics import (
            InverseKinematicsError,
        )

        with pytest.raises(InverseKinematicsError, match="test message"):
            raise InverseKinematicsError("test message")


# ---------------------------------------------------------------------------
# Import smoke tests for all pybullet_helpers submodules
# ---------------------------------------------------------------------------


class TestSubmoduleImports:
    """Verify that every submodule in pybullet_helpers can be imported."""

    def test_import_geometry(self):
        from mara_robosim.pybullet_helpers import geometry  # noqa: F401

        assert hasattr(geometry, "Pose")
        assert hasattr(geometry, "multiply_poses")
        assert hasattr(geometry, "matrix_from_quat")
        assert hasattr(geometry, "get_pose")

    def test_import_joint(self):
        from mara_robosim.pybullet_helpers import joint  # noqa: F401

        assert hasattr(joint, "JointInfo")
        assert hasattr(joint, "JointState")
        assert hasattr(joint, "JointPositions")

    def test_import_link(self):
        from mara_robosim.pybullet_helpers import link  # noqa: F401

        assert hasattr(link, "LinkState")
        assert hasattr(link, "BASE_LINK")

    def test_import_camera(self):
        from mara_robosim.pybullet_helpers import camera  # noqa: F401

        assert hasattr(camera, "create_gui_connection")

    def test_import_objects(self):
        from mara_robosim.pybullet_helpers import objects  # noqa: F401

        assert hasattr(objects, "create_object")
        assert hasattr(objects, "update_object")

    def test_import_inverse_kinematics(self):
        from mara_robosim.pybullet_helpers import inverse_kinematics  # noqa: F401

        assert hasattr(inverse_kinematics, "pybullet_inverse_kinematics")
        assert hasattr(inverse_kinematics, "InverseKinematicsError")

    def test_import_motion_planning(self):
        from mara_robosim.pybullet_helpers import motion_planning  # noqa: F401

        assert hasattr(motion_planning, "run_motion_planning")

    def test_import_controllers(self):
        from mara_robosim.pybullet_helpers import controllers  # noqa: F401

        assert hasattr(controllers, "get_move_end_effector_to_pose_action")
        assert hasattr(controllers, "get_change_fingers_action")

    def test_import_robots_init(self):
        from mara_robosim.pybullet_helpers import robots  # noqa: F401

        assert hasattr(robots, "create_single_arm_pybullet_robot")

    def test_import_robots_single_arm(self):
        from mara_robosim.pybullet_helpers.robots import single_arm  # noqa: F401

        assert hasattr(single_arm, "SingleArmPyBulletRobot")

    def test_import_robots_fetch(self):
        from mara_robosim.pybullet_helpers.robots import fetch  # noqa: F401

        assert hasattr(fetch, "FetchPyBulletRobot")

    def test_import_robots_panda(self):
        from mara_robosim.pybullet_helpers.robots import panda  # noqa: F401

        assert hasattr(panda, "PandaPyBulletRobot")

    def test_import_robots_mobile_fetch(self):
        from mara_robosim.pybullet_helpers.robots import mobile_fetch  # noqa: F401

        assert hasattr(mobile_fetch, "MobileFetchPyBulletRobot")

    def test_import_ikfast_init(self):
        from mara_robosim.pybullet_helpers.ikfast import IKFastInfo  # noqa: F401

    def test_import_ikfast_utils(self):
        from mara_robosim.pybullet_helpers.ikfast import utils  # noqa: F401

    def test_import_ikfast_load(self):
        from mara_robosim.pybullet_helpers.ikfast import load  # noqa: F401

    def test_import_pybullet_helpers_package(self):
        import mara_robosim.pybullet_helpers  # noqa: F401
