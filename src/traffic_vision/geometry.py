"""Dependency-light 2-D geometry used by the line counter and the speed estimator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Point = tuple[float, float]


def side_of_line(point: Point, start: Point, end: Point) -> float:
    """Signed area of the triangle (start, end, point).

    Positive on one side of the directed line ``start -> end``, negative on the
    other and zero on it. In image coordinates (y grows downward) a positive
    value means the point lies on the right-hand side when facing from
    ``start`` towards ``end``.
    """
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def _within_bounds(p: Point, a: Point, b: Point) -> bool:
    """For a point already known to be collinear with a-b: does it lie on the segment?"""
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment p1-p2 meets segment q1-q2. Touching at an endpoint counts."""
    d1 = side_of_line(p1, q1, q2)
    d2 = side_of_line(p2, q1, q2)
    d3 = side_of_line(q1, p1, p2)
    d4 = side_of_line(q2, p1, p2)

    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return (
        (d1 == 0 and _within_bounds(p1, q1, q2))
        or (d2 == 0 and _within_bounds(p2, q1, q2))
        or (d3 == 0 and _within_bounds(q1, p1, p2))
        or (d4 == 0 and _within_bounds(q2, p1, p2))
    )


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd ray casting test. Points exactly on an edge may fall either way."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def homography_from_points(src: Sequence[Point], dst: Sequence[Point]) -> np.ndarray:
    """Solve the 3x3 perspective transform that maps four ``src`` points onto ``dst``.

    Direct linear transform with h33 fixed to 1: each correspondence
    (x, y) -> (u, v) contributes two rows of an 8x8 linear system.
    """
    src_arr = np.asarray(src, dtype=np.float64)
    dst_arr = np.asarray(dst, dtype=np.float64)
    if src_arr.shape != (4, 2) or dst_arr.shape != (4, 2):
        raise ValueError("homography_from_points needs exactly four (x, y) pairs on each side.")

    a = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((x, y), (u, v)) in enumerate(zip(src_arr, dst_arr, strict=True)):
        a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i] = u
        b[2 * i + 1] = v

    try:
        h = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Calibration points are degenerate (three or more are collinear)."
        ) from exc
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(h: np.ndarray, points: Sequence[Point]) -> np.ndarray:
    """Project (N, 2) points through ``h``; returns an (N, 2) array."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    projected = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return projected[:, :2] / projected[:, 2:3]
