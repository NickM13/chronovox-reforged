"""Quaternion and Euler helpers for matrix rotation. Pure math: no GL, UI or file imports.

Quaternions are (x, y, z, w) tuples, as stored in .cvox. Euler angles follow Unity:
degrees, applied Z first, then X, then Y, so ``q = qy * qx * qz``. The numbers in
the rotation formulas are the standard ones, which is also what Unity uses in its
left-handed space.
"""
from __future__ import annotations

import math

import numpy as np

Quat = tuple[float, float, float, float]
IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)


def multiply(a: Quat, b: Quat) -> Quat:
    """Hamilton product: the rotation ``b`` followed by ``a``."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def normalize(q: Quat) -> Quat:
    length = math.sqrt(sum(c * c for c in q))
    if length < 1e-12:
        return IDENTITY
    return tuple(c / length for c in q)


def from_axis_angle(axis, radians: float) -> Quat:
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return IDENTITY
    s = math.sin(radians / 2) / n
    return (x * s, y * s, z * s, math.cos(radians / 2))


def to_matrix(q: Quat) -> np.ndarray:
    """3x3 rotation matrix of a unit quaternion."""
    x, y, z, w = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotate(q: Quat, v) -> np.ndarray:
    return to_matrix(q) @ np.asarray(v, float)


def from_euler(x_deg: float, y_deg: float, z_deg: float) -> Quat:
    """Unity's Quaternion.Euler: Z, then X, then Y."""
    qx = from_axis_angle((1, 0, 0), math.radians(x_deg))
    qy = from_axis_angle((0, 1, 0), math.radians(y_deg))
    qz = from_axis_angle((0, 0, 1), math.radians(z_deg))
    return normalize(multiply(qy, multiply(qx, qz)))


def to_euler(q: Quat) -> tuple[float, float, float]:
    """Euler angles in degrees (x in [-90, 90], y and z in [-180, 180]) for ``from_euler``."""
    m = to_matrix(q)
    sx = -m[1, 2]
    if abs(sx) < 0.999999:
        x = math.asin(sx)
        y = math.atan2(m[0, 2], m[2, 2])
        z = math.atan2(m[1, 0], m[1, 1])
    else:  # looking straight up or down: Y and Z are the same rotation, so give it all to Y
        x = math.copysign(math.pi / 2, sx)
        y = math.atan2(-m[2, 0], m[0, 0])
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)
