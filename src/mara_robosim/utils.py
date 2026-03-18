"""Utility functions for mara-robosim PyBullet environments."""

from __future__ import annotations

import abc
import functools
import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray

from mara_robosim.structs import Object, State

# ---------------------------------------------------------------------------
# Asset path resolution
# ---------------------------------------------------------------------------


def get_asset_path(asset_name: str, assert_exists: bool = True) -> str:
    """Return the absolute path to a bundled asset file.

    Assets are located under ``src/mara_robosim/assets/``.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, "assets")
    path = os.path.join(asset_dir_path, asset_name)
    if assert_exists:
        assert os.path.exists(path), f"Asset not found: {asset_name}."
    return path


def get_third_party_path() -> str:
    """Return the absolute path to the third-party directory.

    Third-party assets (e.g. IKFast compiled modules) are located under
    ``src/mara_robosim/third_party/``.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "third_party")


# ---------------------------------------------------------------------------
# State construction helper
# ---------------------------------------------------------------------------


def create_state_from_dict(
    data: Dict[Object, Dict[str, float]],
    simulator_state: Optional[Any] = None,
) -> State:
    """Create a :class:`State` from a dictionary of per-object feature values.

    Parameters
    ----------
    data:
        Mapping from each :class:`Object` to a dict of
        ``{feature_name: value}`` entries.
    simulator_state:
        An optional opaque simulator state to attach to the returned
        :class:`State`.
    """
    state_dict: Dict[Object, np.ndarray] = {}
    for obj, obj_data in data.items():
        obj_vec = []
        for feat in obj.type.feature_names:
            obj_vec.append(obj_data[feat])
        state_dict[obj] = np.array(obj_vec)
    return State(state_dict, simulator_state)


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


class OptionExecutionFailure(Exception):
    """Raised when an option/controller policy fails during execution."""


# ---------------------------------------------------------------------------
# 2-D geometry primitives (used for collision-free sampling)
# ---------------------------------------------------------------------------


class _Geom2D(abc.ABC):
    """A 2D shape that contains some points."""

    @abc.abstractmethod
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is contained in the shape."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        """Sample a random point inside the 2D shape."""
        raise NotImplementedError("Override me!")

    def intersects(self, other: _Geom2D) -> bool:
        """Check if this shape intersects with another one."""
        return geom2ds_intersect(self, other)


@dataclass(frozen=True)
class LineSegment(_Geom2D):
    """A line segment defined by two endpoints."""

    x1: float
    y1: float
    x2: float
    y2: float

    def contains_point(self, x: float, y: float) -> bool:
        # https://stackoverflow.com/questions/328107
        a = (self.x1, self.y1)
        b = (self.x2, self.y2)
        c = (x, y)
        eps = 1e-6

        def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
            return float(np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2))

        return -eps < _dist(a, c) + _dist(c, b) - _dist(a, b) < eps

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        line_slope = (self.y2 - self.y1) / (self.x2 - self.x1)
        y_intercept = self.y2 - (line_slope * self.x2)
        random_x_point = rng.uniform(self.x1, self.x2)
        random_y_point_on_line = line_slope * random_x_point + y_intercept
        assert self.contains_point(random_x_point, random_y_point_on_line)
        return (random_x_point, random_y_point_on_line)


@dataclass(frozen=True)
class Circle(_Geom2D):
    """A circle defined by centre and radius."""

    x: float
    y: float
    radius: float

    def contains_point(self, x: float, y: float) -> bool:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius**2

    def contains_circle(self, other_circle: Circle) -> bool:
        """Check whether this circle wholly contains another one."""
        dist_between_centers = np.sqrt(
            (other_circle.x - self.x) ** 2 + (other_circle.y - self.y) ** 2
        )
        return (dist_between_centers + other_circle.radius) <= self.radius

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        rand_mag = rng.uniform(0, self.radius)
        rand_theta = rng.uniform(0, 2 * np.pi)
        x_point = self.x + rand_mag * np.cos(rand_theta)
        y_point = self.y + rand_mag * np.sin(rand_theta)
        assert self.contains_point(x_point, y_point)
        return (x_point, y_point)


@dataclass(frozen=True)
class Rectangle(_Geom2D):
    """A rectangle with origin at the bottom-left corner.

    Following the convention in ``matplotlib.patches.Rectangle``, the origin
    is at the bottom-left corner and *theta* is the anti-clockwise rotation
    (in **radians**, between ``-pi`` and ``pi``) about that point.
    """

    x: float
    y: float
    width: float
    height: float
    theta: float  # in radians, between -np.pi and np.pi

    def __post_init__(self) -> None:
        assert -np.pi <= self.theta <= np.pi, "Expecting angle in [-pi, pi]."

    @staticmethod
    def from_center(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        rotation_about_center: float,
    ) -> Rectangle:
        """Create a rectangle given an (x, y) for the centre, with *theta*
        rotating about that centre point."""
        x = center_x - width / 2
        y = center_y - height / 2
        norm_rect = Rectangle(x, y, width, height, 0.0)
        assert np.isclose(norm_rect.center[0], center_x)
        assert np.isclose(norm_rect.center[1], center_y)
        return norm_rect.rotate_about_point(center_x, center_y, rotation_about_center)

    @functools.cached_property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Get the rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def inverse_rotation_matrix(self) -> NDArray[np.float64]:
        """Get the inverse rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), np.sin(self.theta)],
                [-np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the four vertices of the rectangle."""
        scale_matrix = np.array(
            [
                [self.width, 0],
                [0, self.height],
            ]
        )
        translate_vector = np.array([self.x, self.y])
        vertices = np.array(
            [
                (0, 0),
                (0, 1),
                (1, 1),
                (1, 0),
            ]
        )
        vertices = vertices @ scale_matrix.T
        vertices = vertices @ self.rotation_matrix.T
        vertices = translate_vector + vertices
        return list(map(lambda p: (p[0], p[1]), vertices))

    @functools.cached_property
    def line_segments(self) -> List[LineSegment]:
        """Get the four line segments forming the rectangle's boundary."""
        vs = list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))
        line_segments = []
        for (x1, y1), (x2, y2) in vs:
            line_segments.append(LineSegment(x1, y1, x2, y2))
        return line_segments

    @functools.cached_property
    def center(self) -> Tuple[float, float]:
        """Get the point at the centre of the rectangle."""
        x, y = np.mean(self.vertices, axis=0)
        return (x, y)

    @functools.cached_property
    def circumscribed_circle(self) -> Circle:
        """Return the circumscribed circle (centre + radius)."""
        x, y = self.center
        radius = np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        return Circle(x, y, radius)

    def contains_point(self, x: float, y: float) -> bool:
        # First invert translation, then invert rotation.
        rx, ry = np.array([x - self.x, y - self.y]) @ self.inverse_rotation_matrix.T
        return 0 <= rx <= self.width and 0 <= ry <= self.height

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        rand_width = rng.uniform(0, self.width)
        rand_height = rng.uniform(0, self.height)
        # First rotate, then translate.
        rx, ry = np.array([rand_width, rand_height]) @ self.rotation_matrix.T
        x = rx + self.x
        y = ry + self.y
        assert self.contains_point(x, y)
        return (x, y)

    def rotate_about_point(self, x: float, y: float, rot: float) -> Rectangle:
        """Return a new rectangle rotated CCW by *rot* radians about (x, y)."""
        vertices = np.array(self.vertices)
        origin = np.array([x, y])
        vertices = vertices - origin
        rotate_matrix = np.array(
            [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
        )
        vertices = vertices @ rotate_matrix.T
        vertices = vertices + origin
        (lx, ly), _, _, (rx, ry) = vertices
        theta = np.arctan2(ry - ly, rx - lx)
        rect = Rectangle(lx, ly, self.width, self.height, theta)
        assert np.allclose(rect.vertices, vertices)
        return rect


# ---------------------------------------------------------------------------
# 2-D geometry intersection helpers
# ---------------------------------------------------------------------------


def line_segments_intersect(seg1: LineSegment, seg2: LineSegment) -> bool:
    """Check if two line segments intersect.

    Uses relative orientation; allows for collinearity, and only checks if
    each segment straddles the line containing the other.
    """

    def _subtract(
        a: Tuple[float, float], b: Tuple[float, float]
    ) -> Tuple[float, float]:
        return (a[0] - b[0]), (a[1] - b[1])

    def _cross_product(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return a[1] * b[0] - a[0] * b[1]

    def _direction(
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
    ) -> float:
        return _cross_product(_subtract(a, c), _subtract(a, b))

    p1 = (seg1.x1, seg1.y1)
    p2 = (seg1.x2, seg1.y2)
    p3 = (seg2.x1, seg2.y1)
    p4 = (seg2.x2, seg2.y2)
    d1 = _direction(p3, p4, p1)
    d2 = _direction(p3, p4, p2)
    d3 = _direction(p1, p2, p3)
    d4 = _direction(p1, p2, p4)

    return ((d2 < 0 < d1) or (d1 < 0 < d2)) and ((d4 < 0 < d3) or (d3 < 0 < d4))


def circles_intersect(circ1: Circle, circ2: Circle) -> bool:
    """Check if two circles intersect."""
    x1, y1, r1 = circ1.x, circ1.y, circ1.radius
    x2, y2, r2 = circ2.x, circ2.y, circ2.radius
    return (x1 - x2) ** 2 + (y1 - y2) ** 2 < (r1 + r2) ** 2


def rectangles_intersect(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Check if two rectangles intersect."""
    if not circles_intersect(rect1.circumscribed_circle, rect2.circumscribed_circle):
        return False
    # Case 1: line segments intersect.
    if any(
        line_segments_intersect(seg1, seg2)
        for seg1 in rect1.line_segments
        for seg2 in rect2.line_segments
    ):
        return True
    # Case 2: rect1 contains centre of rect2.
    if rect1.contains_point(rect2.center[0], rect2.center[1]):
        return True
    # Case 3: rect2 contains centre of rect1.
    if rect2.contains_point(rect1.center[0], rect1.center[1]):
        return True
    return False


def line_segment_intersects_circle(seg: LineSegment, circ: Circle) -> bool:
    """Check if a line segment intersects a circle."""
    if circ.contains_point(seg.x1, seg.y1):
        return True
    if circ.contains_point(seg.x2, seg.y2):
        return True
    c = (circ.x, circ.y)
    a = (seg.x1, seg.y1)
    b = (seg.x2, seg.y2)
    ba = np.subtract(b, a)
    ca = np.subtract(c, a)
    da = ba * np.dot(ca, ba) / np.dot(ba, ba)
    dx, dy = (a[0] + da[0], a[1] + da[1])
    if not seg.contains_point(dx, dy):
        return False
    return circ.contains_point(dx, dy)


def line_segment_intersects_rectangle(seg: LineSegment, rect: Rectangle) -> bool:
    """Check if a line segment intersects a rectangle."""
    if rect.contains_point(seg.x1, seg.y1) or rect.contains_point(seg.x2, seg.y2):
        return True
    return any(line_segments_intersect(s, seg) for s in rect.line_segments)


def rectangle_intersects_circle(rect: Rectangle, circ: Circle) -> bool:
    """Check if a rectangle intersects a circle."""
    if not circles_intersect(rect.circumscribed_circle, circ):
        return False
    if rect.contains_point(circ.x, circ.y):
        return True
    for seg in rect.line_segments:
        if line_segment_intersects_circle(seg, circ):
            return True
    return False


def geom2ds_intersect(geom1: _Geom2D, geom2: _Geom2D) -> bool:
    """Check if two 2D geometries intersect."""
    if isinstance(geom1, LineSegment) and isinstance(geom2, LineSegment):
        return line_segments_intersect(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, Circle):
        return line_segment_intersects_circle(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, Rectangle):
        return line_segment_intersects_rectangle(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, LineSegment):
        return line_segment_intersects_rectangle(geom2, geom1)
    if isinstance(geom1, Circle) and isinstance(geom2, LineSegment):
        return line_segment_intersects_circle(geom2, geom1)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Rectangle):
        return rectangles_intersect(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Circle):
        return rectangle_intersects_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, Rectangle):
        return rectangle_intersects_circle(geom2, geom1)
    if isinstance(geom1, Circle) and isinstance(geom2, Circle):
        return circles_intersect(geom1, geom2)
    raise NotImplementedError(
        f"Intersection not implemented for geoms {geom1} and {geom2}"
    )


# ---------------------------------------------------------------------------
# RRT / BiRRT motion planning
# ---------------------------------------------------------------------------

_RRTState = TypeVar("_RRTState")


class _RRTNode(Generic[_RRTState]):
    """A node in an RRT tree."""

    def __init__(
        self,
        data: _RRTState,
        parent: Optional[_RRTNode[_RRTState]] = None,
    ) -> None:
        self.data = data
        self.parent = parent

    def path_from_root(self) -> List[_RRTNode[_RRTState]]:
        """Return the path from the root to this node."""
        sequence: List[_RRTNode[_RRTState]] = []
        node: Optional[_RRTNode[_RRTState]] = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]


class RRT(Generic[_RRTState]):
    """Rapidly-exploring random tree."""

    def __init__(
        self,
        sample_fn: Callable[[_RRTState], _RRTState],
        extend_fn: Callable[[_RRTState, _RRTState], Iterator[_RRTState]],
        collision_fn: Callable[[_RRTState], bool],
        distance_fn: Callable[[_RRTState, _RRTState], float],
        rng: np.random.Generator,
        num_attempts: int,
        num_iters: int,
        smooth_amt: int,
    ):
        self._sample_fn = sample_fn
        self._extend_fn = extend_fn
        self._collision_fn = collision_fn
        self._distance_fn = distance_fn
        self._rng = rng
        self._num_attempts = num_attempts
        self._num_iters = num_iters
        self._smooth_amt = smooth_amt

    def query(
        self,
        pt1: _RRTState,
        pt2: _RRTState,
        sample_goal_eps: float = 0.0,
    ) -> Optional[List[_RRTState]]:
        """Query for a collision-free path from *pt1* to *pt2*.

        Returns ``None`` if no path is found.
        """
        if self._collision_fn(pt1) or self._collision_fn(pt2):
            return None
        direct_path = self._try_direct_path(pt1, pt2)
        if direct_path is not None:
            return direct_path
        for _ in range(self._num_attempts):
            path = self._rrt_connect(
                pt1,
                goal_sampler=lambda: pt2,
                sample_goal_eps=sample_goal_eps,
            )
            if path is not None:
                return self._smooth_path(path)
        return None

    def query_to_goal_fn(
        self,
        start: _RRTState,
        goal_sampler: Callable[[], _RRTState],
        goal_fn: Callable[[_RRTState], bool],
        sample_goal_eps: float = 0.0,
    ) -> Optional[List[_RRTState]]:
        """Query for a collision-free path from *start* to any state
        satisfying *goal_fn*.

        Uses *goal_sampler* to produce candidate targets.  Returns ``None``
        if no path is found.
        """
        if self._collision_fn(start):
            return None
        direct_path = self._try_direct_path(start, goal_sampler())
        if direct_path is not None:
            return direct_path
        for _ in range(self._num_attempts):
            path = self._rrt_connect(
                start, goal_sampler, goal_fn, sample_goal_eps=sample_goal_eps
            )
            if path is not None:
                return self._smooth_path(path)
        return None

    def _try_direct_path(
        self, pt1: _RRTState, pt2: _RRTState
    ) -> Optional[List[_RRTState]]:
        path = [pt1]
        for newpt in self._extend_fn(pt1, pt2):
            if self._collision_fn(newpt):
                return None
            path.append(newpt)
        return path

    def _rrt_connect(
        self,
        pt1: _RRTState,
        goal_sampler: Callable[[], _RRTState],
        goal_fn: Optional[Callable[[_RRTState], bool]] = None,
        sample_goal_eps: float = 0.0,
    ) -> Optional[List[_RRTState]]:
        root = _RRTNode(pt1)
        nodes = [root]

        for _ in range(self._num_iters):
            sample_goal = self._rng.random() < sample_goal_eps
            samp = goal_sampler() if sample_goal else self._sample_fn(pt1)
            min_key = functools.partial(self._get_pt_dist_to_node, samp)
            nearest = min(nodes, key=min_key)
            reached_goal = False
            for newpt in self._extend_fn(nearest.data, samp):
                if self._collision_fn(newpt):
                    break
                nearest = _RRTNode(newpt, parent=nearest)
                nodes.append(nearest)
            else:
                reached_goal = sample_goal
            if reached_goal or (goal_fn is not None and goal_fn(nearest.data)):
                path = nearest.path_from_root()
                return [node.data for node in path]
        return None

    def _get_pt_dist_to_node(self, pt: _RRTState, node: _RRTNode[_RRTState]) -> float:
        return self._distance_fn(pt, node.data)

    def _smooth_path(self, path: List[_RRTState]) -> List[_RRTState]:
        assert len(path) > 2
        for _ in range(self._smooth_amt):
            i = self._rng.integers(0, len(path) - 1)
            j = self._rng.integers(0, len(path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = list(self._extend_fn(path[i], path[j]))
            if len(shortcut) < j - i and all(
                not self._collision_fn(pt) for pt in shortcut
            ):
                path = path[: i + 1] + shortcut + path[j + 1 :]
        return path


class BiRRT(RRT[_RRTState]):
    """Bidirectional rapidly-exploring random tree."""

    def query_to_goal_fn(
        self,
        start: _RRTState,
        goal_sampler: Callable[[], _RRTState],
        goal_fn: Callable[[_RRTState], bool],
        sample_goal_eps: float = 0.0,
    ) -> Optional[List[_RRTState]]:
        raise NotImplementedError("Can't query to goal function using BiRRT")

    def _rrt_connect(
        self,
        pt1: _RRTState,
        goal_sampler: Callable[[], _RRTState],
        goal_fn: Optional[Callable[[_RRTState], bool]] = None,
        sample_goal_eps: float = 0.0,
    ) -> Optional[List[_RRTState]]:
        # goal_fn and sample_goal_eps are unused in BiRRT
        pt2 = goal_sampler()
        root1, root2 = _RRTNode(pt1), _RRTNode(pt2)
        nodes1: List[_RRTNode[_RRTState]] = [root1]
        nodes2: List[_RRTNode[_RRTState]] = [root2]

        for _ in range(self._num_iters):
            if len(nodes1) > len(nodes2):
                nodes1, nodes2 = nodes2, nodes1
            samp = self._sample_fn(pt1)
            min_key1 = functools.partial(self._get_pt_dist_to_node, samp)
            nearest1 = min(nodes1, key=min_key1)
            for newpt in self._extend_fn(nearest1.data, samp):
                if self._collision_fn(newpt):
                    break
                nearest1 = _RRTNode(newpt, parent=nearest1)
                nodes1.append(nearest1)
            min_key2 = functools.partial(self._get_pt_dist_to_node, nearest1.data)
            nearest2 = min(nodes2, key=min_key2)
            for newpt in self._extend_fn(nearest2.data, nearest1.data):
                if self._collision_fn(newpt):
                    break
                nearest2 = _RRTNode(newpt, parent=nearest2)
                nodes2.append(nearest2)
            else:
                path1 = nearest1.path_from_root()
                path2 = nearest2.path_from_root()
                if path1[0] != root1:
                    path1, path2 = path2, path1
                assert path1[0] == root1
                path = path1[:-1] + path2[::-1]
                return [node.data for node in path]
        return None
