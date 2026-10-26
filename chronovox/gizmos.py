"""Math behind the matrix-tool handles: mouse rays against axes and planes, and screen-space
picking of line handles. Pure numpy: no GL, UI or file imports.

Matrices are standard 4x4 row-major (translation in the last column), points are (x, y, z).
"""
from __future__ import annotations

import math
from enum import Enum

import numpy as np

AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
AXIS_COLORS = ((0.93, 0.33, 0.33), (0.45, 0.85, 0.40), (0.38, 0.58, 1.0))
HIGHLIGHT_COLOR = (1.0, 0.88, 0.25)


class Mode(Enum):
    VOXEL = "Voxel"
    MATRIX = "Matrix"


class MatrixTool(Enum):
    MOVE = "Move"
    ROTATE = "Rotate"
    RESIZE = "Resize"
    PIVOT = "Origin"          # moves the pivot while the matrix stays where it is


def closest_param_on_line(ray_origin, ray_direction, line_point, line_direction) -> float | None:
    """Parameter t of the point ``line_point + t * line_direction`` closest to the mouse ray.

    The ray is treated as an infinite line. Returns None when the two are (nearly) parallel.
    ``line_direction`` is normalized first, so t is a distance along it.
    """
    a = np.asarray(line_direction, float)
    a = a / np.linalg.norm(a)
    d = np.asarray(ray_direction, float)
    r = np.asarray(line_point, float) - np.asarray(ray_origin, float)
    A, B, C = a @ a, a @ d, d @ d
    D, E = a @ r, d @ r
    denom = A * C - B * B
    if abs(denom) < 1e-9 * max(A * C, 1e-30):
        return None
    return float((B * E - C * D) / denom)


def ray_plane_point(ray_origin, ray_direction, plane_point, plane_normal) -> np.ndarray | None:
    """Where the ray meets a plane, or None if it is parallel to it or the plane is behind the ray."""
    o, d = np.asarray(ray_origin, float), np.asarray(ray_direction, float)
    n = np.asarray(plane_normal, float)
    denom = d @ n
    if abs(denom) < 1e-9 * np.linalg.norm(d) * np.linalg.norm(n):
        return None
    t = ((np.asarray(plane_point, float) - o) @ n) / denom
    if t <= 0:
        return None
    return o + d * t


def project(view_proj, point, width: float, height: float) -> tuple[float, float] | None:
    """Window pixel (x right, y down) of a world point, or None if it is behind the camera."""
    clip = np.asarray(view_proj, float) @ np.array([point[0], point[1], point[2], 1.0])
    if clip[3] <= 1e-6:
        return None
    return ((clip[0] / clip[3] + 1) / 2 * width, (1 - clip[1] / clip[3]) / 2 * height)


def segment_distance_2d(p, a, b) -> float:
    """Distance from point p to the segment a-b, all in 2D."""
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    length2 = ab @ ab
    t = 0.0 if length2 < 1e-12 else min(max(((p - a) @ ab) / length2, 0.0), 1.0)
    return float(np.linalg.norm(p - (a + ab * t)))


def polyline_distance(view_proj, points, cursor, width: float, height: float) -> float:
    """Smallest on-screen distance in pixels from the cursor to a 3D polyline (inf if not visible)."""
    screen = [project(view_proj, p, width, height) for p in points]
    best = math.inf
    for a, b in zip(screen, screen[1:]):
        if a is not None and b is not None:
            best = min(best, segment_distance_2d(cursor, a, b))
    return best


def pick_handle(candidates, view_proj, cursor, width: float, height: float, threshold: float = 9.0):
    """Key of the handle closest to the cursor within ``threshold`` pixels, or None.

    ``candidates`` is a list of (key, polyline points). Ties go to the earlier candidate.
    """
    best_key, best = None, threshold
    for key, points in candidates:
        distance = polyline_distance(view_proj, points, cursor, width, height)
        if distance < best:
            best_key, best = key, distance
    return best_key


def ring_points(center, axis: int, radius: float, segments: int = 48) -> np.ndarray:
    """Closed circle of points around ``center`` in the plane perpendicular to a world axis."""
    angles = np.linspace(0.0, 2 * math.pi, segments + 1)
    u, v = [i for i in range(3) if i != axis]
    points = np.tile(np.asarray(center, float), (segments + 1, 1))
    points[:, u] += radius * np.cos(angles)
    points[:, v] += radius * np.sin(angles)
    return points
