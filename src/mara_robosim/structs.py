"""Data structures for mara-robosim.

Extracted from predicators.structs with all predicators-specific dependencies
removed. Only numpy and PIL are required as external dependencies.
"""

from __future__ import annotations

import abc
import copy
import itertools
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Array = NDArray[np.float32]
Image = NDArray[np.uint8]
Video = List[Image]
Mask = NDArray[np.bool_]
JointPositions = List[float]


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Type:
    """Struct defining a type.

    sim_feature_names are features stored in an object, and usually
    won't change throughout and across tasks. An example is the object's
    pybullet id.
    This is convenient for variables that are not easily extractable from the
    sim state -- whether a food block attracts ants, or the joint id for a
    switch -- but are nonetheless for running the simulation.

    Why not store all features here instead of storing in the State object?
    They can only store one value per feature, so if we generate 10 tasks where
    the blocks are at different locations, it won't be able to store all 10
    locations. One might think they could reset any feature when reset is
    called. But this would require the information is first stored in the State
    object.
    """

    name: str
    feature_names: Sequence[str] = field(repr=False)
    parent: Optional[Type] = field(default=None, repr=False)
    sim_features: Sequence[str] = field(default_factory=lambda: ["id"], repr=False)

    @property
    def dim(self) -> int:
        """Dimensionality of the feature vector of this object type."""
        return len(self.feature_names)

    def get_ancestors(self) -> Set[Type]:
        """Get the set of all types that are ancestors (i.e. parents,
        grandparents, great-grandparents, etc.) of the current type."""
        curr_type: Optional[Type] = self
        ancestors_set: Set[Type] = set()
        while curr_type is not None:
            ancestors_set.add(curr_type)
            curr_type = curr_type.parent
        return ancestors_set

    def pretty_str(self) -> str:
        """Display the type in a nice human-readable format."""
        formatted_features = [f"'{name}'" for name in self.feature_names]
        return f"{self.name}: {{{', '.join(formatted_features)}}}"

    def python_definition_str(self) -> str:
        """Display in a format similar to how a type is instantiated."""
        formatted_features = [f"'{name}'" for name in self.feature_names]
        return (
            f"_{self.name}_type = Type('{self.name}', "
            f"[{', '.join(formatted_features)}])"
        )

    def __call__(self, name: str) -> _TypedEntity:
        """Convenience method for generating _TypedEntities."""
        if name.startswith("?"):
            return Variable(name, self)
        return Object(name, self)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.feature_names)))


# ---------------------------------------------------------------------------
# _TypedEntity / Object / Variable
# ---------------------------------------------------------------------------


@dataclass(frozen=False, order=True, repr=False)
class _TypedEntity:
    """Struct defining an entity with some type, either an object (e.g.,
    block3) or a variable (e.g., ?block).

    Should not be instantiated externally.
    """

    name: str
    type: Type

    @cached_property
    def _str(self) -> str:
        return f"{self.name}:{self.type.name}"

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str

    def is_instance(self, t: Type) -> bool:
        """Return whether this entity is an instance of the given type, taking
        hierarchical typing into account."""
        cur_type: Optional[Type] = self.type
        while cur_type is not None:
            if cur_type == t:
                return True
            cur_type = cur_type.parent
        return False


@dataclass(frozen=False, order=True, repr=False)
class Object(_TypedEntity):
    """Struct defining an Object, which is just a _TypedEntity whose name does
    not start with '?'."""

    sim_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert not self.name.startswith("?")
        # Initialize sim_data from the Type's sim_features
        for sim_feature in self.type.sim_features:
            self.sim_data[sim_feature] = None  # Default to None
        # Keep track of allowed attributes
        self._allowed_attributes = {"sim_data"}.union(self.sim_data.keys())

    def __getattr__(self, name: str) -> Any:
        # Bypass custom logic for internal attributes
        sim_data = object.__getattribute__(self, "sim_data")
        if name in sim_data:
            return sim_data[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        # Always allow the dataclass fields (e.g., "name", "type", "sim_data").
        if name in {"name", "type", "sim_data", "_allowed_attributes"}:
            super().__setattr__(name, value)
            return

        # For anything else, check _allowed_attributes.
        allowed_attrs = (
            object.__getattribute__(self, "_allowed_attributes")
            if object.__getattribute__(self, "__dict__").get("_allowed_attributes")
            else set()
        )
        if name in allowed_attrs:
            sim_data = object.__getattribute__(self, "sim_data")
            if name in sim_data:
                sim_data[name] = value
            else:
                super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot set unknown attribute '{name}'")

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Object):
            return False
        return self.name == other.name and self.type == other.type

    @cached_property
    def id_name(self) -> str:
        """Return a name based on the object's id sim_data field."""
        assert self.id is not None, "Object must have an id set to use id_name"
        return f"{self.type.name}{self.id}"


@dataclass(frozen=False, order=True, repr=False)
class Variable(_TypedEntity):
    """Struct defining a Variable, which is just a _TypedEntity whose name
    starts with '?'."""

    def __post_init__(self) -> None:
        assert self.name.startswith("?")

    def __hash__(self) -> int:
        return self._hash


# ---------------------------------------------------------------------------
# Substitution type aliases (depend on Object / Variable)
# ---------------------------------------------------------------------------

ObjToVarSub = Dict[Object, Variable]
VarToObjSub = Dict[Variable, Object]
VarToVarSub = Dict[Variable, Variable]
ObjToObjSub = Dict[Object, Object]
ObjectOrVariable = TypeVar("ObjectOrVariable", bound=_TypedEntity)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class State:
    """Struct defining the low-level state of the world."""

    data: Dict[Object, Array]
    # Some environments will need to store additional simulator state, so
    # this field is provided.
    simulator_state: Optional[Any] = None

    def __post_init__(self) -> None:
        # Check feature vector dimensions.
        for obj in self:
            assert len(self[obj]) == obj.type.dim

    def __hash__(self) -> int:
        items = []
        for obj in sorted(self.data.keys()):
            arr = self.data[obj]
            if hasattr(arr, "tobytes"):
                items.append((obj, hash(arr.tobytes())))
            else:
                items.append((obj, hash(tuple(arr))))
        return hash(tuple(items))

    def __iter__(self) -> Iterator[Object]:
        """An iterator over the state's objects, in sorted order."""
        return iter(sorted(self.data))

    def __getitem__(self, key: Object) -> Array:
        return self.data[key]

    def get(self, obj: Object, feature_name: str) -> Any:
        """Look up an object feature by name."""
        idx = obj.type.feature_names.index(feature_name)
        return self.data[obj][idx]

    def set(self, obj: Object, feature_name: str, feature_val: Any) -> None:
        """Set the value of an object feature by name."""
        idx = obj.type.feature_names.index(feature_name)
        self.data[obj][idx] = feature_val

    def get_objects(self, object_type: Type) -> List[Object]:
        """Return objects of the given type in the order of __iter__()."""
        return [o for o in self if o.is_instance(object_type)]

    def vec(self, objects: Sequence[Object]) -> Array:
        """Concatenated vector of features for each of the objects in the given
        ordered list."""
        feats: List[Array] = []
        if len(objects) == 0:
            return np.zeros(0, dtype=np.float32)
        for obj in objects:
            feats.append(self[obj])
        return np.hstack(feats)

    def copy(self) -> State:
        """Return a copy of this state.

        The simulator state is deep-copied.
        """
        new_data = {}
        for obj in self:
            new_data[obj] = self._copy_state_value(self.data[obj])
        return State(new_data, simulator_state=copy.deepcopy(self.simulator_state))

    def _copy_state_value(self, val: Any) -> Any:
        if val is None or isinstance(val, (float, bool, int, str)):
            return val
        if isinstance(val, (list, tuple, set)):
            return type(val)(self._copy_state_value(v) for v in val)
        assert hasattr(val, "copy")
        return val.copy()

    def allclose(
        self, other: State, *, allow_sim_state_comparison: bool = False
    ) -> bool:
        """Return whether this state is close enough to another one, i.e., its
        objects are the same, and the features are close.

        Parameters
        ----------
        allow_sim_state_comparison:
            When False (the default), raises NotImplementedError if either state
            has a non-None simulator_state.  Set to True to allow comparison
            even when simulator_state is present.
        """
        if self.simulator_state is not None or other.simulator_state is not None:
            if not allow_sim_state_comparison:
                raise NotImplementedError(
                    "Cannot use allclose when simulator_state is not None."
                )
            if self.simulator_state != other.simulator_state:
                return False
        if not sorted(self.data) == sorted(other.data):
            return False
        for obj in self.data:
            if not np.allclose(self.data[obj], other.data[obj], atol=1e-3):
                return False
        return True

    def pretty_str(self) -> str:
        """Display the state in a nice human-readable format.

        Uses plain string formatting (no tabulate dependency).
        """
        type_to_rows: Dict[Type, List[List[str]]] = {}
        for obj in self:
            if obj.type not in type_to_rows:
                type_to_rows[obj.type] = []
            type_to_rows[obj.type].append([obj.name] + list(map(str, self[obj])))

        table_strs: List[str] = []
        for t in sorted(type_to_rows):
            headers = ["type: " + t.name] + list(t.feature_names)
            rows = type_to_rows[t]
            # Compute column widths
            all_rows = [headers] + rows
            col_widths = [
                max(len(row[c]) for row in all_rows) for c in range(len(headers))
            ]
            fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
            lines = [fmt.format(*headers)]
            lines.append("  ".join("-" * w for w in col_widths))
            for row in rows:
                lines.append(fmt.format(*row))
            table_strs.append("\n".join(lines))

        if not table_strs:
            return "### EMPTY STATE ###\n"
        ll = max(len(line) for table in table_strs for line in table.split("\n"))
        prefix = "#" * (ll // 2 - 3) + " STATE " + "#" * (ll - ll // 2 - 4) + "\n"
        suffix = "\n" + "#" * ll + "\n"
        return prefix + "\n\n".join(table_strs) + suffix

    def dict_str(
        self,
        indent: int = 0,
        object_features: bool = True,
        num_decimal_points: int = 2,
        use_object_id: bool = False,
        ignored_features: Optional[List[str]] = None,
        excluded_object_types: Optional[List[str]] = None,
    ) -> str:
        """Return a dictionary-style string representation of the state.

        Parameters
        ----------
        excluded_object_types:
            List of type names to exclude from the output.  Defaults to no
            exclusions.
        ignored_features:
            Feature names to omit from the output.
        """
        if ignored_features is None:
            ignored_features = ["capacity_liquid", "target_liquid"]
        if excluded_object_types is None:
            excluded_object_types = []

        state_dict: Dict[str, Dict[str, Any]] = {}

        # Collect all unique types from objects in the state
        object_types: Set[Type] = set()
        for obj in self:
            object_types.add(obj.type)

        # Iterate through types and add all objects of each type
        for obj_type in sorted(object_types, key=lambda t: t.name):
            obj_type_name = obj_type.name
            if obj_type_name not in excluded_object_types:
                objects_of_type = self.get_objects(obj_type)
                for obj in objects_of_type:
                    obj_dict: Dict[str, Any] = {}
                    if obj_type_name == "robot" or object_features:
                        for attribute, value in zip(obj.type.feature_names, self[obj]):
                            if attribute not in ignored_features:
                                obj_dict[attribute] = value
                    if use_object_id:
                        obj_name = obj.id_name
                    else:
                        obj_name = obj.name
                    state_dict[f"{obj_name}:{obj.type.name}"] = obj_dict

        # Build string
        spaces = " " * indent
        dict_str = spaces + "{"
        n_keys = len(state_dict.keys())
        for i, (key, value) in enumerate(state_dict.items()):
            formatted_items = []
            for k, v in value.items():
                if isinstance(v, (float, np.floating)):
                    formatted_items.append(f"'{k}': {v:.{num_decimal_points}f}")
                else:
                    formatted_items.append(f"'{k}': {v}")
            value_str = ", ".join(formatted_items)

            if i == 0:
                dict_str += f"'{key}': {{{value_str}}},\n"
            elif i == n_keys - 1:
                dict_str += spaces + f" '{key}': {{{value_str}}}"
            else:
                dict_str += spaces + f" '{key}': {{{value_str}}},\n"
        dict_str += "}"
        return dict_str


DefaultState = State({})


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

# Default variable names used in pretty_str for predicates.
_PRETTY_STR_VAR_NAMES = list("xyzwvutsrq")


@dataclass(frozen=True, order=False, repr=False)
class Predicate:
    """Struct defining a predicate (a lifted classifier over states)."""

    name: str
    types: Sequence[Type]
    # The classifier takes in a complete state and a sequence of objects
    # representing the arguments. These objects should be the only ones
    # treated "specially" by the classifier.
    _classifier: Callable[[State, Sequence[Object]], bool] = field(compare=False)
    natural_language_assertion: Optional[Callable[[List[str]], str]] = field(
        default=None, compare=False
    )

    def __call__(self, entities: Sequence[_TypedEntity]) -> _Atom:
        """Convenience method for generating Atoms."""
        if self.arity == 0:
            raise ValueError(
                "Cannot use __call__ on a 0-arity predicate, "
                "since we can't determine whether it becomes a "
                "LiftedAtom or a GroundAtom. Use the LiftedAtom "
                "or GroundAtom constructors directly instead"
            )
        if all(isinstance(ent, Variable) for ent in entities):
            return LiftedAtom(self, entities)
        if all(isinstance(ent, Object) for ent in entities):
            return GroundAtom(self, entities)
        raise ValueError("Cannot instantiate Atom with mix of " "variables and objects")

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, Predicate)
        if self.name != other.name:
            return False
        if len(self.types) != len(other.types):
            return False
        for self_type, other_type in zip(self.types, other.types):
            if self_type != other_type:
                return False
        return True

    @cached_property
    def arity(self) -> int:
        """The arity of this predicate (number of arguments)."""
        return len(self.types)

    def holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Public method for calling the classifier.

        Performs type checking first.
        """
        assert len(objects) == self.arity
        for obj, pred_type in zip(objects, self.types):
            assert isinstance(obj, Object)
            assert obj.is_instance(pred_type)
        return self._classifier(state, objects)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def pretty_str(self) -> Tuple[str, str]:
        """Display the predicate in a nice human-readable format.

        Returns a tuple of (variables string, body string).
        """
        if hasattr(self._classifier, "pretty_str"):
            pretty_str_f = getattr(self._classifier, "pretty_str")
            return pretty_str_f()
        var_names = _PRETTY_STR_VAR_NAMES
        vars_str = ", ".join(
            f"{var_names[i]}:{t.name}" for i, t in enumerate(self.types)
        )
        vars_str_no_types = ", ".join(f"{var_names[i]}" for i in range(self.arity))
        body_str = f"{self.name}({vars_str_no_types})"
        return vars_str, body_str

    def pretty_str_with_assertion(self) -> str:
        """Display predicate with its natural-language assertion."""
        var_names_list: List[str] = []
        vars_str_parts: List[str] = []
        for i, t in enumerate(self.types):
            vn = _PRETTY_STR_VAR_NAMES[i]
            vars_str_parts.append(f"{vn}:{t.name}")
            var_names_list.append(vn)
        vars_str = ", ".join(vars_str_parts)
        body_str = f"{self.name}({vars_str})"
        if (
            hasattr(self, "natural_language_assertion")
            and self.natural_language_assertion is not None
        ):
            body_str += f": {self.natural_language_assertion(var_names_list)}"
        return body_str

    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to a PDDL
        file."""
        if self.arity == 0:
            return f"({self.name})"
        vars_str = " ".join(f"?x{i} - {t.name}" for i, t in enumerate(self.types))
        return f"({self.name} {vars_str})"

    def get_negation(self) -> Predicate:
        """Return a negated version of this predicate."""
        return Predicate("NOT-" + self.name, self.types, self._negated_classifier)

    def _negated_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        return not self._classifier(state, objects)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, Predicate)
        return str(self) < str(other)

    def __reduce__(self) -> Tuple:
        """Tell pickle/dill how to re-create a Predicate."""
        return (self.__class__, (self.name, tuple(self.types), self._classifier))


# ---------------------------------------------------------------------------
# _Atom / LiftedAtom / GroundAtom
# ---------------------------------------------------------------------------


@dataclass(frozen=True, repr=False, eq=False)
class _Atom:
    """Struct defining an atom (a predicate applied to either variables or
    objects).

    Should not be instantiated externally.
    """

    predicate: Predicate
    entities: Sequence[_TypedEntity]

    def __post_init__(self) -> None:
        if isinstance(self.entities, _TypedEntity):
            raise ValueError(
                "Atoms expect a sequence of entities, not a " "single entity."
            )
        assert len(self.entities) == self.predicate.arity
        for ent, pred_type in zip(self.entities, self.predicate.types):
            assert ent.is_instance(pred_type)

    @property
    def _str(self) -> str:
        raise NotImplementedError("Override me")

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to a PDDL
        file."""
        if not self.entities:
            return f"({self.predicate.name})"
        entities_str = " ".join(e.name for e in self.entities)
        return f"({self.predicate.name} {entities_str})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _Atom)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _Atom)
        return str(self) < str(other)

    def __reduce__(self) -> Tuple:
        """Return a pickling recipe."""
        return (self.__class__, (self.predicate, tuple(self.entities)))


@dataclass(frozen=True, repr=False, eq=False)
class LiftedAtom(_Atom):
    """Struct defining a lifted atom (a predicate applied to variables)."""

    @cached_property
    def variables(self) -> List[Variable]:
        """Arguments for this lifted atom.

        A list of ``Variable``s.
        """
        return list(cast(Variable, ent) for ent in self.entities)

    @cached_property
    def _str(self) -> str:
        return str(self.predicate) + "(" + ", ".join(map(str, self.variables)) + ")"

    def ground(self, sub: VarToObjSub) -> GroundAtom:
        """Create a GroundAtom with a given substitution."""
        assert set(self.variables).issubset(set(sub.keys()))
        return GroundAtom(self.predicate, [sub[v] for v in self.variables])

    def substitute(self, sub: VarToVarSub) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution."""
        assert set(self.variables).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[v] for v in self.variables])


@dataclass(frozen=True, repr=False, eq=False)
class GroundAtom(_Atom):
    """Struct defining a ground atom (a predicate applied to objects)."""

    @cached_property
    def objects(self) -> List[Object]:
        """Arguments for this ground atom.

        A list of ``Object``s.
        """
        return list(cast(Object, ent) for ent in self.entities)

    @cached_property
    def _str(self) -> str:
        return str(self.predicate) + "(" + ", ".join(map(str, self.objects)) + ")"

    def lift(self, sub: ObjToVarSub) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution."""
        assert set(self.objects).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[o] for o in self.objects])

    def holds(self, state: State) -> bool:
        """Check whether this ground atom holds in the given state."""
        return self.predicate.holds(state, self.objects)

    def get_negated_atom(self) -> GroundAtom:
        """Get the negated atom of this GroundAtom.

        Always uses ``Predicate.get_negation()`` to produce the negated
        predicate. If the predicate is already a negation (name starts with
        'NOT-'), the double negation is stripped to recover the original.
        """
        if self.predicate.name.startswith("NOT-"):
            # Already negated -- try to invert by stripping the NOT- prefix.
            # We cannot perfectly recover the original classifier, so we negate
            # the (already negated) classifier which gives back the original.
            return GroundAtom(self.predicate.get_negation(), self.objects)
        return GroundAtom(self.predicate.get_negation(), self.objects)


# ---------------------------------------------------------------------------
# Higher-order type aliases (depend on GroundAtom / State / Image)
# ---------------------------------------------------------------------------

Observation = Optional[Union[State, Image]]
GoalDescription = Optional[Union[Set[GroundAtom], str]]
LiftedOrGroundAtom = TypeVar("LiftedOrGroundAtom", LiftedAtom, GroundAtom, _Atom)


# ---------------------------------------------------------------------------
# Task / EnvironmentTask
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Task:
    """Struct defining a task, which is an initial state and goal."""

    init: State
    goal: Set[GroundAtom]
    # Sometimes we want the task presented to the agent to have goals described
    # in terms of predicates that are different than those describing the goal
    # of the task presented to the demonstrator. In these cases, we will store
    # an "alternative goal" in this field and replace the goal with the
    # alternative goal before giving the task to the agent.
    alt_goal: Optional[Set[GroundAtom]] = field(default_factory=set)
    # Optional natural language description of the goal.
    goal_nl: Optional[str] = None

    def __post_init__(self) -> None:
        for atom in self.goal:
            assert isinstance(atom, GroundAtom)

    def goal_holds(self, state: State) -> bool:
        """Return whether the goal of this task holds in the given state."""
        return all(atom.holds(state) for atom in self.goal)

    def replace_goal_with_alt_goal(self) -> Task:
        """Return a Task with the goal replaced with the alternative goal if it
        exists."""
        if self.alt_goal:
            return Task(self.init, goal=self.alt_goal, goal_nl=self.goal_nl)
        return self


DefaultTask = Task(DefaultState, set())


@dataclass(frozen=True, eq=False)
class EnvironmentTask:
    """An initial observation and goal description.

    Environments produce environment tasks and agents produce and solve
    tasks.

    In fully observed settings, the init_obs will be a State and the
    goal_description will be a Set[GroundAtom]. For convenience, we can
    convert an EnvironmentTask into a Task in those cases.
    """

    init_obs: Observation
    goal_description: GoalDescription
    # See Task.alt_goal for the reason for this field.
    alt_goal_desc: Optional[GoalDescription] = field(default=None)
    # Optional natural language goal description (passed through to Task).
    goal_nl: Optional[str] = None

    @cached_property
    def task(self) -> Task:
        """Convenience method for environment tasks that are fully observed."""
        if self.alt_goal_desc is None:
            return Task(self.init, self.goal, goal_nl=self.goal_nl)
        assert isinstance(self.alt_goal_desc, set)
        for atom in self.alt_goal_desc:
            assert isinstance(atom, GroundAtom)
        return Task(
            self.init, self.goal, alt_goal=self.alt_goal_desc, goal_nl=self.goal_nl
        )

    @cached_property
    def init(self) -> State:
        """Convenience method for environment tasks that are fully observed."""
        assert isinstance(self.init_obs, State)
        return self.init_obs

    @cached_property
    def goal(self) -> Set[GroundAtom]:
        """Convenience method for environment tasks that are fully observed."""
        assert isinstance(self.goal_description, set)
        assert not self.goal_description or isinstance(
            next(iter(self.goal_description)), GroundAtom
        )
        return self.goal_description

    def replace_goal_with_alt_goal(self) -> EnvironmentTask:
        """Return an EnvironmentTask with the goal description replaced with
        the alternative goal description if it exists."""
        if self.alt_goal_desc is not None:
            return EnvironmentTask(self.init_obs, goal_description=self.alt_goal_desc)
        return self


DefaultEnvironmentTask = EnvironmentTask(DefaultState, set())


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Action:
    """An action in an environment.

    This is a light wrapper around a numpy float array that can
    optionally store extra information.
    """

    _arr: Array
    # In rare cases, we want to associate additional information with an action
    # to control how it is executed in the environment. This is helpful if
    # actions are awkward to represent with continuous vectors.
    extra_info: Optional[Any] = None

    @property
    def arr(self) -> Array:
        """The array representation of this action."""
        return self._arr


# ---------------------------------------------------------------------------
# PyBulletState
# ---------------------------------------------------------------------------


class PyBulletState(State):
    """A PyBullet state that stores the robot joint positions in addition to
    the features that are exposed in the object-centric state."""

    @property
    def joint_positions(self) -> JointPositions:
        """Expose the current joints state in the simulator_state."""
        if isinstance(self.simulator_state, dict):
            jp = self.simulator_state["joint_positions"]
        else:
            jp = self.simulator_state
        return cast(JointPositions, jp)

    @property
    def state_image(self) -> Image:
        """Expose the current image state in the simulator_state."""
        assert isinstance(self.simulator_state, dict)
        return self.simulator_state["unlabeled_image"]

    @property
    def labeled_image(self) -> Optional[Image]:
        """Expose the current labeled image in the simulator_state."""
        assert isinstance(self.simulator_state, dict)
        return self.simulator_state.get("images")

    @property
    def obj_mask_dict(self) -> Optional[Dict[Object, Mask]]:
        """Expose the current object masks in the simulator_state."""
        assert isinstance(self.simulator_state, dict)
        return self.simulator_state.get("obj_mask_dict")

    def allclose(self, other: State, **kwargs: Any) -> bool:
        """Compare states ignoring simulator_state."""
        return State(self.data).allclose(State(other.data), **kwargs)

    def copy(self) -> PyBulletState:
        """Return a copy of this PyBulletState."""
        base_copy = super().copy()
        return PyBulletState(base_copy.data, base_copy.simulator_state)

    def get_obj_mask(self, obj: Object) -> Mask:
        """Return the mask for the object."""
        assert self.obj_mask_dict is not None
        mask = self.obj_mask_dict.get(obj)
        assert mask is not None
        return mask

    def add_images_and_masks(
        self, unlabeled_image: Any, masks: Dict[Object, Mask]
    ) -> None:
        """Add the unlabeled image and object masks to the simulator state."""
        assert isinstance(self.simulator_state, dict)
        self.simulator_state["unlabeled_image"] = unlabeled_image
        self.simulator_state["obj_mask_dict"] = masks
