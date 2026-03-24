"""Comprehensive tests for mara_robosim.structs.

Tests cover Type, Object, Variable, State, Action, Predicate,
GroundAtom, LiftedAtom, EnvironmentTask, and PyBulletState data
structures. All tests are pure unit tests and do not require PyBullet or
any simulator.
"""

import copy

import numpy as np
import pytest

from mara_robosim.structs import (
    Action,
    EnvironmentTask,
    GroundAtom,
    LiftedAtom,
    Object,
    Predicate,
    PyBulletState,
    State,
    Task,
    Type,
    Variable,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def block_type():
    """A simple 2-feature type."""
    return Type("block", ["x", "y"])


@pytest.fixture()
def robot_type():
    """A robot type with three features and custom sim_features."""
    return Type("robot", ["x", "y", "theta"], sim_features=["id", "gripper_id"])


@pytest.fixture()
def block_objects(block_type):
    """Two block objects."""
    b0 = Object("b0", block_type)
    b1 = Object("b1", block_type)
    return b0, b1


@pytest.fixture()
def simple_state(block_type, block_objects):
    """A state with two blocks."""
    b0, b1 = block_objects
    return State(
        {
            b0: np.array([1.0, 2.0], dtype=np.float32),
            b1: np.array([3.0, 4.0], dtype=np.float32),
        }
    )


# ---------------------------------------------------------------------------
# Type tests
# ---------------------------------------------------------------------------


class TestType:
    """Tests for the Type dataclass."""

    def test_creation_basic(self, block_type):
        """Type stores name and feature_names correctly."""
        assert block_type.name == "block"
        assert list(block_type.feature_names) == ["x", "y"]

    def test_feature_names_preserved(self):
        """Feature names order is preserved."""
        t = Type("widget", ["alpha", "beta", "gamma"])
        assert list(t.feature_names) == ["alpha", "beta", "gamma"]

    def test_sim_features_default(self, block_type):
        """Default sim_features is ['id']."""
        assert list(block_type.sim_features) == ["id"]

    def test_sim_features_custom(self, robot_type):
        """Custom sim_features are stored."""
        assert list(robot_type.sim_features) == ["id", "gripper_id"]

    def test_dim(self, block_type):
        """dim returns the length of feature_names."""
        assert block_type.dim == 2

    def test_dim_zero(self):
        """A type with no features has dim 0."""
        t = Type("empty", [])
        assert t.dim == 0

    def test_parent_none_by_default(self, block_type):
        """Parent is None when not specified."""
        assert block_type.parent is None

    def test_parent_type(self, block_type):
        """A child type references its parent."""
        child = Type("small_block", ["x", "y"], parent=block_type)
        assert child.parent is block_type

    def test_get_ancestors_no_parent(self, block_type):
        """Ancestors of a root type contains only itself."""
        ancestors = block_type.get_ancestors()
        assert ancestors == {block_type}

    def test_get_ancestors_with_chain(self, block_type):
        """Ancestors traverses the full parent chain."""
        child = Type("small_block", ["x", "y"], parent=block_type)
        grandchild = Type("tiny_block", ["x", "y"], parent=child)
        ancestors = grandchild.get_ancestors()
        assert ancestors == {grandchild, child, block_type}

    def test_hash_consistent(self):
        """Two types with same name and feature_names hash equally."""
        t1 = Type("block", ["x", "y"])
        t2 = Type("block", ["x", "y"])
        assert hash(t1) == hash(t2)

    def test_hash_different_names(self):
        """Types with different names hash differently."""
        t1 = Type("block", ["x", "y"])
        t2 = Type("ball", ["x", "y"])
        assert hash(t1) != hash(t2)

    def test_equality(self):
        """Types compare equal when name and feature_names match."""
        t1 = Type("block", ["x", "y"])
        t2 = Type("block", ["x", "y"])
        assert t1 == t2

    def test_inequality_different_features(self):
        """Types with different feature names are not equal."""
        t1 = Type("block", ["x", "y"])
        t2 = Type("block", ["x", "z"])
        assert t1 != t2

    def test_ordering(self):
        """Types support ordering (frozen=True, order=True)."""
        t_a = Type("alpha", ["x"])
        t_b = Type("beta", ["x"])
        assert t_a < t_b
        assert sorted([t_b, t_a]) == [t_a, t_b]

    def test_call_creates_object(self, block_type):
        """Calling a type with a plain name creates an Object."""
        obj = block_type("b0")
        assert isinstance(obj, Object)
        assert obj.name == "b0"
        assert obj.type == block_type

    def test_call_creates_variable(self, block_type):
        """Calling a type with a '?'-prefixed name creates a Variable."""
        var = block_type("?block")
        assert isinstance(var, Variable)
        assert var.name == "?block"
        assert var.type == block_type

    def test_pretty_str(self, block_type):
        """pretty_str returns a human-readable representation."""
        s = block_type.pretty_str()
        assert "block" in s
        assert "'x'" in s
        assert "'y'" in s

    def test_python_definition_str(self, block_type):
        """python_definition_str returns a valid-looking Type instantiation."""
        s = block_type.python_definition_str()
        assert "Type('block'" in s
        assert "'x'" in s
        assert "'y'" in s


# ---------------------------------------------------------------------------
# Object tests
# ---------------------------------------------------------------------------


class TestObject:
    """Tests for the Object dataclass."""

    def test_creation(self, block_type):
        """Object is created with correct name and type."""
        obj = Object("b0", block_type)
        assert obj.name == "b0"
        assert obj.type == block_type

    def test_name_must_not_start_with_question_mark(self, block_type):
        """Object name starting with '?' triggers an assertion."""
        with pytest.raises(AssertionError):
            Object("?b0", block_type)

    def test_str_and_repr(self, block_type):
        """__str__ and __repr__ show name:type_name."""
        obj = Object("b0", block_type)
        assert str(obj) == "b0:block"
        assert repr(obj) == "b0:block"

    def test_equality(self, block_type):
        """Objects with same name and type are equal."""
        o1 = Object("b0", block_type)
        o2 = Object("b0", block_type)
        assert o1 == o2

    def test_inequality_different_name(self, block_type):
        """Objects with different names are not equal."""
        o1 = Object("b0", block_type)
        o2 = Object("b1", block_type)
        assert o1 != o2

    def test_inequality_different_type(self):
        """Objects with different types are not equal."""
        t1 = Type("block", ["x", "y"])
        t2 = Type("ball", ["x", "y"])
        o1 = Object("b0", t1)
        o2 = Object("b0", t2)
        assert o1 != o2

    def test_equality_not_object(self, block_type):
        """Comparing Object to a non-Object returns False."""
        obj = Object("b0", block_type)
        assert obj != "b0:block"

    def test_hash_consistency(self, block_type):
        """Equal objects hash the same."""
        o1 = Object("b0", block_type)
        o2 = Object("b0", block_type)
        assert hash(o1) == hash(o2)

    def test_usable_in_set(self, block_type):
        """Objects can be stored in a set, deduplicating equal ones."""
        o1 = Object("b0", block_type)
        o2 = Object("b0", block_type)
        o3 = Object("b1", block_type)
        s = {o1, o2, o3}
        assert len(s) == 2

    def test_ordering(self, block_type):
        """Objects support ordering."""
        o_a = Object("a0", block_type)
        o_b = Object("b0", block_type)
        assert o_a < o_b
        assert sorted([o_b, o_a]) == [o_a, o_b]

    def test_sim_data_initialized(self, block_type):
        """Object.sim_data is populated from the type's sim_features."""
        obj = Object("b0", block_type)
        assert "id" in obj.sim_data
        assert obj.sim_data["id"] is None

    def test_sim_data_custom_sim_features(self, robot_type):
        """sim_data reflects custom sim_features."""
        obj = Object("robot0", robot_type)
        assert "id" in obj.sim_data
        assert "gripper_id" in obj.sim_data

    def test_sim_data_attribute_access(self, block_type):
        """sim_data values accessible as attributes via __getattr__."""
        obj = Object("b0", block_type)
        obj.sim_data["id"] = 42
        assert obj.id == 42

    def test_sim_data_attribute_set(self, block_type):
        """sim_data values settable as attributes via __setattr__."""
        obj = Object("b0", block_type)
        obj.id = 99
        assert obj.sim_data["id"] == 99

    def test_unknown_attribute_set_raises(self, block_type):
        """Setting an unknown attribute raises AttributeError."""
        obj = Object("b0", block_type)
        with pytest.raises(AttributeError, match="Cannot set unknown attribute"):
            obj.unknown_attr = "value"

    def test_unknown_attribute_get_raises(self, block_type):
        """Getting an unknown attribute raises AttributeError."""
        obj = Object("b0", block_type)
        with pytest.raises(AttributeError):
            _ = obj.nonexistent

    def test_id_name(self, block_type):
        """id_name builds a string from type name and id."""
        obj = Object("b0", block_type)
        obj.id = 7
        assert obj.id_name == "block7"

    def test_id_name_requires_id(self, block_type):
        """id_name raises when id is None."""
        obj = Object("b0", block_type)
        with pytest.raises(AssertionError):
            _ = obj.id_name

    def test_is_instance_same_type(self, block_type):
        """is_instance returns True for the object's own type."""
        obj = Object("b0", block_type)
        assert obj.is_instance(block_type)

    def test_is_instance_parent_type(self, block_type):
        """is_instance returns True for a parent type."""
        child_type = Type("small_block", ["x", "y"], parent=block_type)
        obj = Object("sb0", child_type)
        assert obj.is_instance(block_type)
        assert obj.is_instance(child_type)

    def test_is_instance_unrelated_type(self):
        """is_instance returns False for an unrelated type."""
        t1 = Type("block", ["x", "y"])
        t2 = Type("ball", ["r"])
        obj = Object("b0", t1)
        assert not obj.is_instance(t2)

    def test_holds_predicate_with_type(self, block_type, simple_state):
        """A predicate over blocks can check the type via holds."""
        b0, b1 = list(simple_state)
        pred = Predicate(
            "On",
            [block_type, block_type],
            lambda s, objs: s.get(objs[0], "x") < s.get(objs[1], "x"),
        )
        assert pred.holds(simple_state, [b0, b1])
        assert not pred.holds(simple_state, [b1, b0])


# ---------------------------------------------------------------------------
# Variable tests
# ---------------------------------------------------------------------------


class TestVariable:
    """Tests for the Variable dataclass."""

    def test_creation(self, block_type):
        """Variable stores name (with '?') and type."""
        v = Variable("?b", block_type)
        assert v.name == "?b"
        assert v.type == block_type

    def test_name_must_start_with_question_mark(self, block_type):
        """Variable name without '?' triggers an assertion."""
        with pytest.raises(AssertionError):
            Variable("b", block_type)

    def test_str_and_repr(self, block_type):
        """__str__ and __repr__ show name:type_name."""
        v = Variable("?b", block_type)
        assert str(v) == "?b:block"
        assert repr(v) == "?b:block"

    def test_equality_via_hash(self, block_type):
        """Two variables with same name and type hash the same."""
        v1 = Variable("?b", block_type)
        v2 = Variable("?b", block_type)
        assert hash(v1) == hash(v2)

    def test_inequality_different_name(self, block_type):
        """Variables with different names have different hashes."""
        v1 = Variable("?a", block_type)
        v2 = Variable("?b", block_type)
        assert hash(v1) != hash(v2)

    def test_is_instance(self, block_type):
        """Variable.is_instance works like Object.is_instance."""
        child_type = Type("small_block", ["x", "y"], parent=block_type)
        v = Variable("?sb", child_type)
        assert v.is_instance(block_type)
        assert v.is_instance(child_type)

    def test_usable_in_set(self, block_type):
        """Variables can be placed in sets."""
        v1 = Variable("?a", block_type)
        v2 = Variable("?a", block_type)
        v3 = Variable("?b", block_type)
        assert len({v1, v2, v3}) == 2


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


class TestState:
    """Tests for the State dataclass."""

    def test_creation_from_dict(self, block_type, block_objects):
        """State created from {Object: Array} dict."""
        b0, b1 = block_objects
        state = State(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            }
        )
        assert len(list(state)) == 2

    def test_creation_empty(self):
        """An empty state is valid."""
        state = State({})
        assert list(state) == []

    def test_creation_wrong_dim_raises(self, block_type):
        """Feature vector with wrong dimension raises on construction."""
        obj = Object("b0", block_type)
        with pytest.raises(AssertionError):
            State({obj: np.array([1.0, 2.0, 3.0], dtype=np.float32)})

    def test_iter_sorted(self, block_type):
        """Iterating over state yields objects in sorted order."""
        b_z = Object("z0", block_type)
        b_a = Object("a0", block_type)
        state = State(
            {
                b_z: np.array([0.0, 0.0], dtype=np.float32),
                b_a: np.array([1.0, 1.0], dtype=np.float32),
            }
        )
        objects = list(state)
        assert objects == [b_a, b_z]

    def test_getitem(self, simple_state, block_objects):
        """state[obj] returns the feature array."""
        b0, b1 = block_objects
        np.testing.assert_array_equal(simple_state[b0], [1.0, 2.0])
        np.testing.assert_array_equal(simple_state[b1], [3.0, 4.0])

    def test_get_feature(self, simple_state, block_objects):
        """state.get(obj, feature_name) returns the correct value."""
        b0, _ = block_objects
        assert simple_state.get(b0, "x") == pytest.approx(1.0)
        assert simple_state.get(b0, "y") == pytest.approx(2.0)

    def test_set_feature(self, simple_state, block_objects):
        """state.set(obj, feature_name, val) mutates the feature."""
        b0, _ = block_objects
        simple_state.set(b0, "x", 10.0)
        assert simple_state.get(b0, "x") == pytest.approx(10.0)

    def test_copy_independence(self, simple_state, block_objects):
        """Modifying a copy does not affect the original."""
        b0, _ = block_objects
        state_copy = simple_state.copy()
        state_copy.set(b0, "x", 999.0)
        assert simple_state.get(b0, "x") == pytest.approx(1.0)
        assert state_copy.get(b0, "x") == pytest.approx(999.0)

    def test_copy_preserves_values(self, simple_state, block_objects):
        """A copy has the same feature values."""
        b0, b1 = block_objects
        state_copy = simple_state.copy()
        np.testing.assert_array_equal(state_copy[b0], simple_state[b0])
        np.testing.assert_array_equal(state_copy[b1], simple_state[b1])

    def test_copy_deep_copies_simulator_state(self, block_type, block_objects):
        """Simulator state is deep-copied."""
        b0, b1 = block_objects
        sim_state = {"key": [1, 2, 3]}
        state = State(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            },
            simulator_state=sim_state,
        )
        state_copy = state.copy()
        state_copy.simulator_state["key"].append(4)
        assert sim_state["key"] == [1, 2, 3]  # original unchanged

    def test_allclose_identical(self, simple_state):
        """allclose returns True for an identical copy."""
        state_copy = simple_state.copy()
        assert simple_state.allclose(state_copy)

    def test_allclose_within_tolerance(self, block_type, block_objects):
        """allclose returns True when values differ by less than atol."""
        b0, b1 = block_objects
        s1 = State(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            }
        )
        s2 = State(
            {
                b0: np.array([1.0005, 2.0005], dtype=np.float32),
                b1: np.array([3.0005, 4.0005], dtype=np.float32),
            }
        )
        assert s1.allclose(s2)

    def test_allclose_outside_tolerance(self, block_type, block_objects):
        """allclose returns False when values differ too much."""
        b0, b1 = block_objects
        s1 = State(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            }
        )
        s2 = State(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 5.0], dtype=np.float32),
            }
        )
        assert not s1.allclose(s2)

    def test_allclose_different_objects(self, block_type):
        """allclose returns False when object sets differ."""
        b0 = Object("b0", block_type)
        b1 = Object("b1", block_type)
        b2 = Object("b2", block_type)
        s1 = State({b0: np.array([1.0, 2.0], dtype=np.float32)})
        s2 = State({b2: np.array([1.0, 2.0], dtype=np.float32)})
        assert not s1.allclose(s2)

    def test_allclose_raises_when_sim_state_present(self, block_type):
        """allclose raises NotImplementedError if simulator_state is set."""
        b0 = Object("b0", block_type)
        s1 = State(
            {b0: np.array([1.0, 2.0], dtype=np.float32)},
            simulator_state="something",
        )
        s2 = State({b0: np.array([1.0, 2.0], dtype=np.float32)})
        with pytest.raises(NotImplementedError):
            s1.allclose(s2)

    def test_allclose_with_sim_state_allowed(self, block_type):
        """allclose succeeds when allow_sim_state_comparison=True."""
        b0 = Object("b0", block_type)
        s1 = State(
            {b0: np.array([1.0, 2.0], dtype=np.float32)},
            simulator_state="same",
        )
        s2 = State(
            {b0: np.array([1.0, 2.0], dtype=np.float32)},
            simulator_state="same",
        )
        assert s1.allclose(s2, allow_sim_state_comparison=True)

    def test_allclose_with_different_sim_state(self, block_type):
        """allclose returns False when simulator_states differ."""
        b0 = Object("b0", block_type)
        s1 = State(
            {b0: np.array([1.0, 2.0], dtype=np.float32)},
            simulator_state="one",
        )
        s2 = State(
            {b0: np.array([1.0, 2.0], dtype=np.float32)},
            simulator_state="two",
        )
        assert not s1.allclose(s2, allow_sim_state_comparison=True)

    def test_get_objects(self, simple_state, block_type, block_objects):
        """get_objects returns objects of the given type."""
        objects = simple_state.get_objects(block_type)
        assert len(objects) == 2
        b0, b1 = block_objects
        assert set(objects) == {b0, b1}

    def test_get_objects_with_parent_type(self, block_type):
        """get_objects finds child-type objects when querying parent type."""
        child_type = Type("small_block", ["x", "y"], parent=block_type)
        parent_obj = Object("b0", block_type)
        child_obj = Object("sb0", child_type)
        state = State(
            {
                parent_obj: np.array([1.0, 2.0], dtype=np.float32),
                child_obj: np.array([3.0, 4.0], dtype=np.float32),
            }
        )
        result = state.get_objects(block_type)
        assert parent_obj in result
        assert child_obj in result

    def test_get_objects_no_match(self, simple_state):
        """get_objects returns empty list for an unrelated type."""
        other_type = Type("ball", ["r"])
        assert simple_state.get_objects(other_type) == []

    def test_vec(self, simple_state, block_objects):
        """vec concatenates feature vectors for the given objects."""
        b0, b1 = block_objects
        v = simple_state.vec([b0, b1])
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0, 4.0])

    def test_vec_empty(self, simple_state):
        """vec with empty list returns a zero-length array."""
        v = simple_state.vec([])
        assert len(v) == 0

    def test_pretty_str(self, simple_state):
        """pretty_str returns a non-empty formatted string."""
        s = simple_state.pretty_str()
        assert "STATE" in s
        assert "block" in s

    def test_pretty_str_empty(self):
        """pretty_str of empty state returns a placeholder."""
        s = State({}).pretty_str()
        assert "EMPTY STATE" in s

    def test_hash(self, simple_state, block_type, block_objects):
        """State is hashable and consistent for equal states."""
        b0, b1 = block_objects
        s2 = State(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            }
        )
        assert hash(simple_state) == hash(s2)

    def test_dict_str(self, simple_state):
        """dict_str returns a dict-like string representation."""
        s = simple_state.dict_str()
        assert "{" in s
        assert "block" in s


# ---------------------------------------------------------------------------
# Action tests
# ---------------------------------------------------------------------------


class TestAction:
    """Tests for the Action dataclass."""

    def test_creation_with_numpy_array(self):
        """Action wraps a numpy array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action = Action(arr)
        np.testing.assert_array_equal(action.arr, arr)

    def test_arr_property(self):
        """The arr property returns the internal array."""
        arr = np.array([0.5, -1.0], dtype=np.float32)
        action = Action(arr)
        assert action.arr is arr

    def test_extra_info_default_none(self):
        """extra_info defaults to None."""
        action = Action(np.array([1.0], dtype=np.float32))
        assert action.extra_info is None

    def test_extra_info_stored(self):
        """extra_info can store arbitrary data."""
        info = {"grasp": True, "force": 10.0}
        action = Action(np.array([1.0], dtype=np.float32), extra_info=info)
        assert action.extra_info == info

    def test_arr_shape(self):
        """arr shape matches the input array shape."""
        arr = np.zeros((5,), dtype=np.float32)
        action = Action(arr)
        assert action.arr.shape == (5,)


# ---------------------------------------------------------------------------
# Predicate tests
# ---------------------------------------------------------------------------


class TestPredicate:
    """Tests for the Predicate dataclass."""

    def test_creation(self, block_type):
        """Predicate stores name, types, and classifier."""
        pred = Predicate("Clear", [block_type], lambda s, o: True)
        assert pred.name == "Clear"
        assert list(pred.types) == [block_type]
        assert pred.arity == 1

    def test_arity(self, block_type):
        """arity returns the number of types."""
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        assert pred.arity == 2

    def test_arity_zero(self):
        """A 0-arity predicate is valid."""
        pred = Predicate("Done", [], lambda s, o: True)
        assert pred.arity == 0

    def test_holds_true(self, block_type, simple_state, block_objects):
        """holds returns True when the classifier returns True."""
        b0, b1 = block_objects
        pred = Predicate(
            "On",
            [block_type, block_type],
            lambda s, objs: True,
        )
        assert pred.holds(simple_state, [b0, b1])

    def test_holds_false(self, block_type, simple_state, block_objects):
        """holds returns False when the classifier returns False."""
        b0, b1 = block_objects
        pred = Predicate(
            "On",
            [block_type, block_type],
            lambda s, objs: False,
        )
        assert not pred.holds(simple_state, [b0, b1])

    def test_holds_checks_types(self, block_type, simple_state, block_objects):
        """holds asserts that objects match the predicate's types."""
        ball_type = Type("ball", ["r"])
        ball = Object("ball0", ball_type)
        pred = Predicate("Clear", [block_type], lambda s, o: True)
        # Ball is not a block, so this should fail the assertion.
        with pytest.raises(AssertionError):
            pred.holds(
                State({ball: np.array([1.0], dtype=np.float32)}),
                [ball],
            )

    def test_holds_checks_arity(self, block_type, simple_state, block_objects):
        """holds asserts correct number of objects."""
        b0, b1 = block_objects
        pred = Predicate("Clear", [block_type], lambda s, o: True)
        with pytest.raises(AssertionError):
            pred.holds(simple_state, [b0, b1])

    def test_str(self, block_type):
        """__str__ returns the predicate name."""
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        assert str(pred) == "On"

    def test_equality(self, block_type):
        """Two predicates with same name and types are equal."""
        p1 = Predicate("On", [block_type, block_type], lambda s, o: True)
        p2 = Predicate("On", [block_type, block_type], lambda s, o: False)
        assert p1 == p2  # classifier is compare=False

    def test_inequality_different_name(self, block_type):
        """Predicates with different names are not equal."""
        p1 = Predicate("On", [block_type, block_type], lambda s, o: True)
        p2 = Predicate("Under", [block_type, block_type], lambda s, o: True)
        assert p1 != p2

    def test_inequality_different_types(self, block_type):
        """Predicates with different type lists are not equal."""
        other_type = Type("ball", ["r"])
        p1 = Predicate("Holds", [block_type], lambda s, o: True)
        p2 = Predicate("Holds", [other_type], lambda s, o: True)
        assert p1 != p2

    def test_hash_consistent(self, block_type):
        """Equal predicates hash the same."""
        p1 = Predicate("On", [block_type, block_type], lambda s, o: True)
        p2 = Predicate("On", [block_type, block_type], lambda s, o: False)
        assert hash(p1) == hash(p2)

    def test_ordering(self, block_type):
        """Predicates support < comparison."""
        p_a = Predicate("Alpha", [block_type], lambda s, o: True)
        p_b = Predicate("Beta", [block_type], lambda s, o: True)
        assert p_a < p_b

    def test_get_negation(self, block_type, simple_state, block_objects):
        """get_negation returns a predicate that inverts the classifier."""
        b0, _ = block_objects
        pred = Predicate("Clear", [block_type], lambda s, o: True)
        neg_pred = pred.get_negation()
        assert neg_pred.name == "NOT-Clear"
        assert neg_pred.holds(simple_state, [b0]) is False

    def test_call_creates_ground_atom(self, block_type, block_objects):
        """Calling a predicate with Objects creates a GroundAtom."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        atom = pred([b0, b1])
        assert isinstance(atom, GroundAtom)

    def test_call_creates_lifted_atom(self, block_type):
        """Calling a predicate with Variables creates a LiftedAtom."""
        v0 = Variable("?x", block_type)
        v1 = Variable("?y", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        atom = pred([v0, v1])
        assert isinstance(atom, LiftedAtom)

    def test_call_mixed_raises(self, block_type, block_objects):
        """Calling a predicate with mixed Objects/Variables raises."""
        b0, _ = block_objects
        v = Variable("?x", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        with pytest.raises(ValueError, match="mix"):
            pred([b0, v])

    def test_call_zero_arity_raises(self):
        """Calling a 0-arity predicate raises ValueError."""
        pred = Predicate("Done", [], lambda s, o: True)
        with pytest.raises(ValueError, match="0-arity"):
            pred([])

    def test_get_ground_atoms(self, block_type, block_objects, simple_state):
        """Predicate can be used to derive ground atoms and check them."""
        b0, b1 = block_objects
        pred = Predicate(
            "On",
            [block_type, block_type],
            lambda s, objs: (s.get(objs[0], "x") < s.get(objs[1], "x")),
        )
        # Create ground atoms for all pairs.
        atoms = set()
        for o1 in [b0, b1]:
            for o2 in [b0, b1]:
                ga = GroundAtom(pred, [o1, o2])
                if ga.holds(simple_state):
                    atoms.add(ga)
        # b0.x=1 < b1.x=3, so On(b0, b1) should hold.
        assert len(atoms) == 1
        atom = atoms.pop()
        assert atom.objects == [b0, b1]

    def test_pretty_str(self, block_type):
        """pretty_str returns a tuple of (vars, body) strings."""
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        vars_str, body_str = pred.pretty_str()
        assert "x:block" in vars_str
        assert "y:block" in vars_str
        assert "On(x, y)" == body_str

    def test_pddl_str(self, block_type):
        """pddl_str returns a PDDL-style string."""
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        s = pred.pddl_str()
        assert s.startswith("(On")
        assert "block" in s

    def test_pddl_str_zero_arity(self):
        """pddl_str for a 0-arity predicate."""
        pred = Predicate("Done", [], lambda s, o: True)
        assert pred.pddl_str() == "(Done)"

    def test_natural_language_assertion(self, block_type):
        """pretty_str_with_assertion includes the NL assertion."""
        pred = Predicate(
            "On",
            [block_type, block_type],
            lambda s, o: True,
            natural_language_assertion=lambda names: f"{names[0]} is on top of {names[1]}",
        )
        s = pred.pretty_str_with_assertion()
        assert "x is on top of y" in s


# ---------------------------------------------------------------------------
# GroundAtom tests
# ---------------------------------------------------------------------------


class TestGroundAtom:
    """Tests for the GroundAtom dataclass."""

    def test_creation(self, block_type, block_objects):
        """GroundAtom is created from predicate and objects."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        atom = GroundAtom(pred, [b0, b1])
        assert atom.predicate == pred
        assert atom.objects == [b0, b1]

    def test_str(self, block_type, block_objects):
        """__str__ shows predicate(obj1, obj2)."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        atom = GroundAtom(pred, [b0, b1])
        assert str(atom) == "On(b0:block, b1:block)"

    def test_holds_true(self, block_type, block_objects, simple_state):
        """holds returns True when the predicate holds."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        atom = GroundAtom(pred, [b0, b1])
        assert atom.holds(simple_state)

    def test_holds_false(self, block_type, block_objects, simple_state):
        """holds returns False when the predicate does not hold."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: False)
        atom = GroundAtom(pred, [b0, b1])
        assert not atom.holds(simple_state)

    def test_equality(self, block_type, block_objects):
        """Equal ground atoms compare equal."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        a1 = GroundAtom(pred, [b0, b1])
        a2 = GroundAtom(pred, [b0, b1])
        assert a1 == a2

    def test_inequality_different_objects(self, block_type, block_objects):
        """Ground atoms with different object order are not equal."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        a1 = GroundAtom(pred, [b0, b1])
        a2 = GroundAtom(pred, [b1, b0])
        assert a1 != a2

    def test_hash_consistent(self, block_type, block_objects):
        """Equal ground atoms hash the same."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        a1 = GroundAtom(pred, [b0, b1])
        a2 = GroundAtom(pred, [b0, b1])
        assert hash(a1) == hash(a2)

    def test_usable_in_set(self, block_type, block_objects):
        """Ground atoms work correctly in sets."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        a1 = GroundAtom(pred, [b0, b1])
        a2 = GroundAtom(pred, [b0, b1])
        a3 = GroundAtom(pred, [b1, b0])
        assert len({a1, a2, a3}) == 2

    def test_ordering(self, block_type, block_objects):
        """Ground atoms support < comparison for sorting."""
        b0, b1 = block_objects
        pred_a = Predicate("Alpha", [block_type], lambda s, o: True)
        pred_b = Predicate("Beta", [block_type], lambda s, o: True)
        a1 = GroundAtom(pred_a, [b0])
        a2 = GroundAtom(pred_b, [b0])
        assert a1 < a2
        assert sorted([a2, a1]) == [a1, a2]

    def test_wrong_arity_raises(self, block_type, block_objects):
        """Creating a GroundAtom with wrong number of objects raises."""
        b0, _ = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        with pytest.raises(AssertionError):
            GroundAtom(pred, [b0])

    def test_wrong_type_raises(self, block_type, block_objects):
        """Creating a GroundAtom with wrong object type raises."""
        ball_type = Type("ball", ["r"])
        ball = Object("ball0", ball_type)
        pred = Predicate("Clear", [block_type], lambda s, o: True)
        with pytest.raises(AssertionError):
            GroundAtom(pred, [ball])

    def test_lift(self, block_type, block_objects):
        """lift() creates a LiftedAtom from a substitution."""
        b0, b1 = block_objects
        v0 = Variable("?x", block_type)
        v1 = Variable("?y", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        ga = GroundAtom(pred, [b0, b1])
        la = ga.lift({b0: v0, b1: v1})
        assert isinstance(la, LiftedAtom)
        assert la.variables == [v0, v1]

    def test_get_negated_atom(self, block_type, block_objects, simple_state):
        """get_negated_atom returns a ground atom with inverted predicate."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        ga = GroundAtom(pred, [b0, b1])
        neg_ga = ga.get_negated_atom()
        assert neg_ga.predicate.name == "NOT-On"
        assert neg_ga.holds(simple_state) is False  # NOT-On when On is True

    def test_pddl_str(self, block_type, block_objects):
        """pddl_str returns PDDL representation."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        ga = GroundAtom(pred, [b0, b1])
        s = ga.pddl_str()
        assert s == "(On b0 b1)"


# ---------------------------------------------------------------------------
# LiftedAtom tests
# ---------------------------------------------------------------------------


class TestLiftedAtom:
    """Tests for the LiftedAtom dataclass."""

    def test_creation(self, block_type):
        """LiftedAtom created from predicate and variables."""
        v0 = Variable("?x", block_type)
        v1 = Variable("?y", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        atom = LiftedAtom(pred, [v0, v1])
        assert atom.predicate == pred
        assert atom.variables == [v0, v1]

    def test_str(self, block_type):
        """__str__ shows predicate(?var1:type, ?var2:type)."""
        v0 = Variable("?x", block_type)
        v1 = Variable("?y", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        atom = LiftedAtom(pred, [v0, v1])
        assert str(atom) == "On(?x:block, ?y:block)"

    def test_equality(self, block_type):
        """Equal lifted atoms compare equal."""
        v0 = Variable("?x", block_type)
        v1 = Variable("?y", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        a1 = LiftedAtom(pred, [v0, v1])
        a2 = LiftedAtom(pred, [v0, v1])
        assert a1 == a2

    def test_hash_consistent(self, block_type):
        """Equal lifted atoms hash the same."""
        v0 = Variable("?x", block_type)
        v1 = Variable("?y", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        a1 = LiftedAtom(pred, [v0, v1])
        a2 = LiftedAtom(pred, [v0, v1])
        assert hash(a1) == hash(a2)

    def test_ground(self, block_type, block_objects):
        """ground() creates a GroundAtom from a substitution."""
        b0, b1 = block_objects
        v0 = Variable("?x", block_type)
        v1 = Variable("?y", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        la = LiftedAtom(pred, [v0, v1])
        ga = la.ground({v0: b0, v1: b1})
        assert isinstance(ga, GroundAtom)
        assert ga.objects == [b0, b1]

    def test_substitute(self, block_type):
        """substitute() creates a new LiftedAtom with renamed variables."""
        v0 = Variable("?x", block_type)
        v1 = Variable("?y", block_type)
        v2 = Variable("?a", block_type)
        v3 = Variable("?b", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        la = LiftedAtom(pred, [v0, v1])
        new_la = la.substitute({v0: v2, v1: v3})
        assert isinstance(new_la, LiftedAtom)
        assert new_la.variables == [v2, v3]

    def test_pddl_str(self, block_type):
        """pddl_str returns PDDL representation with variable names."""
        v0 = Variable("?x", block_type)
        v1 = Variable("?y", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        la = LiftedAtom(pred, [v0, v1])
        assert la.pddl_str() == "(On ?x ?y)"

    def test_wrong_arity_raises(self, block_type):
        """Creating a LiftedAtom with wrong number of variables raises."""
        v0 = Variable("?x", block_type)
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        with pytest.raises(AssertionError):
            LiftedAtom(pred, [v0])

    def test_single_entity_not_sequence_raises(self, block_type):
        """Passing a single entity instead of a sequence raises."""
        v0 = Variable("?x", block_type)
        pred = Predicate("Clear", [block_type], lambda s, o: True)
        with pytest.raises(ValueError, match="sequence"):
            LiftedAtom(pred, v0)


# ---------------------------------------------------------------------------
# EnvironmentTask tests
# ---------------------------------------------------------------------------


class TestEnvironmentTask:
    """Tests for the EnvironmentTask dataclass."""

    def test_creation_with_state_and_goal(
        self, block_type, block_objects, simple_state
    ):
        """EnvironmentTask created with State init_obs and GroundAtom goal."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        goal = {GroundAtom(pred, [b0, b1])}
        env_task = EnvironmentTask(simple_state, goal)
        assert env_task.init_obs is simple_state
        assert env_task.goal_description == goal

    def test_init_property(self, simple_state):
        """init property returns the State when init_obs is a State."""
        env_task = EnvironmentTask(simple_state, set())
        assert env_task.init is simple_state

    def test_goal_property(self, block_type, block_objects, simple_state):
        """goal property returns the set of GroundAtoms."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        goal = {GroundAtom(pred, [b0, b1])}
        env_task = EnvironmentTask(simple_state, goal)
        assert env_task.goal == goal

    def test_task_property(self, block_type, block_objects, simple_state):
        """task property returns a Task with init and goal."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        goal = {GroundAtom(pred, [b0, b1])}
        env_task = EnvironmentTask(simple_state, goal)
        task = env_task.task
        assert isinstance(task, Task)
        assert task.init is simple_state
        assert task.goal == goal

    def test_goal_nl(self, simple_state):
        """goal_nl is passed through to the Task."""
        env_task = EnvironmentTask(simple_state, set(), goal_nl="Stack all blocks")
        task = env_task.task
        assert task.goal_nl == "Stack all blocks"

    def test_alt_goal_desc(self, block_type, block_objects, simple_state):
        """alt_goal_desc is stored and used in task."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        goal = {GroundAtom(pred, [b0, b1])}
        alt_goal = {GroundAtom(pred, [b1, b0])}
        env_task = EnvironmentTask(
            simple_state,
            goal,
            alt_goal_desc=alt_goal,
        )
        task = env_task.task
        assert task.alt_goal == alt_goal

    def test_replace_goal_with_alt_goal(self, block_type, block_objects, simple_state):
        """replace_goal_with_alt_goal creates a new EnvironmentTask."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        goal = {GroundAtom(pred, [b0, b1])}
        alt_goal = {GroundAtom(pred, [b1, b0])}
        env_task = EnvironmentTask(
            simple_state,
            goal,
            alt_goal_desc=alt_goal,
        )
        replaced = env_task.replace_goal_with_alt_goal()
        assert replaced.goal_description == alt_goal

    def test_replace_goal_without_alt_returns_self(self, simple_state):
        """replace_goal_with_alt_goal returns self if no alt_goal."""
        env_task = EnvironmentTask(simple_state, set())
        assert env_task.replace_goal_with_alt_goal() is env_task


# ---------------------------------------------------------------------------
# Task tests
# ---------------------------------------------------------------------------


class TestTask:
    """Tests for the Task dataclass."""

    def test_creation(self, block_type, block_objects, simple_state):
        """Task stores init state and goal set."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        goal = {GroundAtom(pred, [b0, b1])}
        task = Task(simple_state, goal)
        assert task.init is simple_state
        assert task.goal == goal

    def test_goal_holds(self, block_type, block_objects, simple_state):
        """goal_holds returns True when all goal atoms hold."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        goal = {GroundAtom(pred, [b0, b1])}
        task = Task(simple_state, goal)
        assert task.goal_holds(simple_state)

    def test_goal_does_not_hold(self, block_type, block_objects, simple_state):
        """goal_holds returns False when not all goal atoms hold."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: False)
        goal = {GroundAtom(pred, [b0, b1])}
        task = Task(simple_state, goal)
        assert not task.goal_holds(simple_state)

    def test_replace_goal_with_alt_goal(self, block_type, block_objects, simple_state):
        """replace_goal_with_alt_goal swaps the goal."""
        b0, b1 = block_objects
        pred = Predicate("On", [block_type, block_type], lambda s, o: True)
        goal = {GroundAtom(pred, [b0, b1])}
        alt = {GroundAtom(pred, [b1, b0])}
        task = Task(simple_state, goal, alt_goal=alt)
        replaced = task.replace_goal_with_alt_goal()
        assert replaced.goal == alt

    def test_replace_goal_without_alt_returns_self(self, simple_state):
        """replace_goal_with_alt_goal returns self when no alt."""
        task = Task(simple_state, set())
        assert task.replace_goal_with_alt_goal() is task


# ---------------------------------------------------------------------------
# PyBulletState tests
# ---------------------------------------------------------------------------


class TestPyBulletState:
    """Tests for the PyBulletState dataclass."""

    def test_creation(self, block_type, block_objects):
        """PyBulletState is created like State with simulator_state."""
        b0, b1 = block_objects
        joint_pos = [0.1, 0.2, 0.3]
        pbs = PyBulletState(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            },
            simulator_state=joint_pos,
        )
        assert len(list(pbs)) == 2

    def test_joint_positions_from_list(self, block_type, block_objects):
        """joint_positions returns simulator_state when it's a list."""
        b0, b1 = block_objects
        joint_pos = [0.1, 0.2, 0.3]
        pbs = PyBulletState(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            },
            simulator_state=joint_pos,
        )
        assert pbs.joint_positions == [0.1, 0.2, 0.3]

    def test_joint_positions_from_dict(self, block_type, block_objects):
        """joint_positions extracted from dict simulator_state."""
        b0, b1 = block_objects
        sim_state = {"joint_positions": [0.5, 1.0]}
        pbs = PyBulletState(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            },
            simulator_state=sim_state,
        )
        assert pbs.joint_positions == [0.5, 1.0]

    def test_copy_returns_pybullet_state(self, block_type, block_objects):
        """copy() returns a PyBulletState, not a plain State."""
        b0, b1 = block_objects
        pbs = PyBulletState(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            },
            simulator_state=[0.1, 0.2],
        )
        pbs_copy = pbs.copy()
        assert isinstance(pbs_copy, PyBulletState)

    def test_allclose_ignores_sim_state(self, block_type, block_objects):
        """allclose compares data only, ignoring simulator_state."""
        b0, b1 = block_objects
        pbs1 = PyBulletState(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            },
            simulator_state={"joint_positions": [0.1]},
        )
        pbs2 = PyBulletState(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            },
            simulator_state={"joint_positions": [999.0]},
        )
        # PyBulletState.allclose ignores sim_state, so should be True.
        assert pbs1.allclose(pbs2)

    def test_add_images_and_masks(self, block_type, block_objects):
        """add_images_and_masks stores image and mask data."""
        b0, b1 = block_objects
        sim_state = {"joint_positions": [0.1, 0.2]}
        pbs = PyBulletState(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            },
            simulator_state=sim_state,
        )
        fake_image = np.zeros((64, 64, 3), dtype=np.uint8)
        fake_mask_b0 = np.ones((64, 64), dtype=np.bool_)
        fake_mask_b1 = np.zeros((64, 64), dtype=np.bool_)
        masks = {b0: fake_mask_b0, b1: fake_mask_b1}

        pbs.add_images_and_masks(fake_image, masks)

        np.testing.assert_array_equal(pbs.state_image, fake_image)
        assert pbs.obj_mask_dict is not None
        assert b0 in pbs.obj_mask_dict
        assert b1 in pbs.obj_mask_dict
        np.testing.assert_array_equal(pbs.get_obj_mask(b0), fake_mask_b0)
        np.testing.assert_array_equal(pbs.get_obj_mask(b1), fake_mask_b1)

    def test_labeled_image(self, block_type, block_objects):
        """labeled_image property returns stored images or None."""
        b0, b1 = block_objects
        sim_state = {"joint_positions": [0.1], "images": None}
        pbs = PyBulletState(
            {
                b0: np.array([1.0, 2.0], dtype=np.float32),
                b1: np.array([3.0, 4.0], dtype=np.float32),
            },
            simulator_state=sim_state,
        )
        assert pbs.labeled_image is None

        # Now set images.
        fake_images = np.ones((64, 64, 3), dtype=np.uint8)
        sim_state["images"] = fake_images
        np.testing.assert_array_equal(pbs.labeled_image, fake_images)
