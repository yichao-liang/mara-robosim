"""Tests for utils.py."""

from mara_robosim.structs import Object, State, Type
from mara_robosim.utils import create_state_from_dict, get_asset_path


def test_get_asset_path():
    """Test that get_asset_path resolves correctly."""
    path = get_asset_path("urdf", assert_exists=True)
    assert "assets/urdf" in path


def test_create_state_from_dict():
    """Test create_state_from_dict."""
    block_type = Type("block", ["x", "y"])
    b0 = Object("b0", block_type)
    state = create_state_from_dict({b0: {"x": 1.0, "y": 2.0}})
    assert isinstance(state, State)
    assert state.get(b0, "x") == 1.0
    assert state.get(b0, "y") == 2.0
