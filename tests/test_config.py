"""Tests for mara_robosim.config."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from mara_robosim.config import (
    BLOCKS_EE_ORNS,
    DEFAULT_EE_ORNS,
    BiRRTConfig,
    IKFastConfig,
    PyBulletConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify the module-level orientation dictionaries exist and are sane."""

    def test_default_ee_orns_keys(self):
        assert "fetch" in DEFAULT_EE_ORNS
        assert "mobile_fetch" in DEFAULT_EE_ORNS
        assert "panda" in DEFAULT_EE_ORNS

    def test_blocks_ee_orns_keys(self):
        assert "fetch" in BLOCKS_EE_ORNS
        assert "mobile_fetch" in BLOCKS_EE_ORNS
        assert "panda" in BLOCKS_EE_ORNS

    def test_quaternion_length(self):
        for quat in DEFAULT_EE_ORNS.values():
            assert len(quat) == 4
        for quat in BLOCKS_EE_ORNS.values():
            assert len(quat) == 4


# ---------------------------------------------------------------------------
# BiRRTConfig
# ---------------------------------------------------------------------------


class TestBiRRTConfig:
    """Tests for BiRRTConfig frozen dataclass."""

    def test_defaults(self):
        cfg = BiRRTConfig()
        assert cfg.num_attempts == 10
        assert cfg.num_iters == 100
        assert cfg.smooth_amt == 50
        assert cfg.extend_num_interp == 10
        assert cfg.path_subsample_ratio == 1

    def test_custom_values(self):
        cfg = BiRRTConfig(num_attempts=5, num_iters=200, smooth_amt=25)
        assert cfg.num_attempts == 5
        assert cfg.num_iters == 200
        assert cfg.smooth_amt == 25
        # Unchanged defaults preserved.
        assert cfg.extend_num_interp == 10
        assert cfg.path_subsample_ratio == 1

    def test_frozen(self):
        cfg = BiRRTConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.num_attempts = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IKFastConfig
# ---------------------------------------------------------------------------


class TestIKFastConfig:
    """Tests for IKFastConfig frozen dataclass."""

    def test_defaults(self):
        cfg = IKFastConfig()
        assert cfg.max_time == 0.05
        assert cfg.max_candidates == 100
        assert cfg.max_attempts == np.inf
        assert cfg.max_distance == np.inf
        assert cfg.norm == np.inf

    def test_custom_values(self):
        cfg = IKFastConfig(max_time=0.1, max_candidates=50, norm=2.0)
        assert cfg.max_time == 0.1
        assert cfg.max_candidates == 50
        assert cfg.norm == 2.0
        # Unchanged defaults preserved.
        assert cfg.max_attempts == np.inf
        assert cfg.max_distance == np.inf

    def test_frozen(self):
        cfg = IKFastConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.max_time = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PyBulletConfig
# ---------------------------------------------------------------------------


class TestPyBulletConfig:
    """Tests for PyBulletConfig frozen dataclass."""

    # -- Default values --

    def test_default_robot(self):
        cfg = PyBulletConfig()
        assert cfg.robot == "fetch"

    def test_default_debug_flags(self):
        cfg = PyBulletConfig()
        assert cfg.draw_debug is False
        assert cfg.use_gui is False

    def test_default_camera(self):
        cfg = PyBulletConfig()
        assert cfg.camera_width == 335
        assert cfg.camera_height == 180
        assert cfg.rgb_observation is False

    def test_default_simulation(self):
        cfg = PyBulletConfig()
        assert cfg.sim_steps_per_action == 20

    def test_default_ik(self):
        cfg = PyBulletConfig()
        assert cfg.max_ik_iters == 100
        assert cfg.ik_tol == 1e-3
        assert cfg.ik_validate is True

    def test_default_control(self):
        cfg = PyBulletConfig()
        assert cfg.control_mode == "position"
        assert cfg.max_vel_norm == 0.05

    def test_default_ee_orns_is_copy_of_module_constant(self):
        cfg = PyBulletConfig()
        assert cfg.ee_orns == DEFAULT_EE_ORNS
        # Must be a *copy*, not the same dict object.
        assert cfg.ee_orns is not DEFAULT_EE_ORNS

    def test_default_tasks_and_seed(self):
        cfg = PyBulletConfig()
        assert cfg.num_train_tasks == 50
        assert cfg.num_test_tasks == 50
        assert cfg.seed == 0

    # -- Custom values --

    def test_custom_values(self):
        cfg = PyBulletConfig(
            robot="panda",
            use_gui=True,
            camera_width=1674,
            camera_height=900,
            seed=42,
        )
        assert cfg.robot == "panda"
        assert cfg.use_gui is True
        assert cfg.camera_width == 1674
        assert cfg.camera_height == 900
        assert cfg.seed == 42
        # Unchanged defaults preserved.
        assert cfg.draw_debug is False
        assert cfg.control_mode == "position"

    def test_custom_ee_orns(self):
        custom = {"my_robot": (0.0, 0.0, 0.0, 1.0)}
        cfg = PyBulletConfig(ee_orns=custom)
        assert cfg.ee_orns == custom

    # -- Frozen (immutability) --

    def test_frozen_scalar(self):
        cfg = PyBulletConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.robot = "panda"  # type: ignore[misc]

    def test_frozen_nested(self):
        cfg = PyBulletConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.birrt = BiRRTConfig(num_attempts=1)  # type: ignore[misc]

    def test_replace_creates_new_instance(self):
        cfg = PyBulletConfig()
        cfg2 = replace(cfg, robot="panda", seed=7)
        assert cfg2.robot == "panda"
        assert cfg2.seed == 7
        # Original unchanged.
        assert cfg.robot == "fetch"
        assert cfg.seed == 0

    # -- get_ee_orn helper --

    def test_get_ee_orn_uses_current_robot(self):
        cfg = PyBulletConfig(robot="fetch")
        assert cfg.get_ee_orn() == DEFAULT_EE_ORNS["fetch"]

    def test_get_ee_orn_explicit_robot(self):
        cfg = PyBulletConfig(robot="fetch")
        assert cfg.get_ee_orn("panda") == DEFAULT_EE_ORNS["panda"]

    def test_get_ee_orn_missing_robot_raises(self):
        cfg = PyBulletConfig()
        with pytest.raises(KeyError):
            cfg.get_ee_orn("nonexistent_robot")

    def test_get_ee_orn_with_custom_orns(self):
        custom = {"custom_bot": (0.1, 0.2, 0.3, 0.4)}
        cfg = PyBulletConfig(robot="custom_bot", ee_orns=custom)
        assert cfg.get_ee_orn() == (0.1, 0.2, 0.3, 0.4)

    # -- Nested sub-configs --

    def test_nested_birrt_default(self):
        cfg = PyBulletConfig()
        assert isinstance(cfg.birrt, BiRRTConfig)
        assert cfg.birrt.num_attempts == 10

    def test_nested_ikfast_default(self):
        cfg = PyBulletConfig()
        assert isinstance(cfg.ikfast, IKFastConfig)
        assert cfg.ikfast.max_time == 0.05

    def test_nested_custom_birrt(self):
        birrt = BiRRTConfig(num_attempts=3, num_iters=50)
        cfg = PyBulletConfig(birrt=birrt)
        assert cfg.birrt.num_attempts == 3
        assert cfg.birrt.num_iters == 50

    def test_nested_custom_ikfast(self):
        ikfast = IKFastConfig(max_time=0.2, max_candidates=10)
        cfg = PyBulletConfig(ikfast=ikfast)
        assert cfg.ikfast.max_time == 0.2
        assert cfg.ikfast.max_candidates == 10

    def test_separate_instances_have_independent_sub_configs(self):
        cfg1 = PyBulletConfig()
        cfg2 = PyBulletConfig()
        # Default sub-configs should be equal but distinct objects.
        assert cfg1.birrt == cfg2.birrt
        assert cfg1.birrt is not cfg2.birrt
        assert cfg1.ikfast == cfg2.ikfast
        assert cfg1.ikfast is not cfg2.ikfast
