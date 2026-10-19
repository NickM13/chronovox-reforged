import math

import numpy as np
import pytest

from chronovox import transform as t


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_identity_does_nothing():
    assert np.allclose(t.rotate(t.IDENTITY, (1, 2, 3)), (1, 2, 3))


def test_ninety_degrees_about_y_turns_forward_into_right_like_unity():
    q = t.from_euler(0, 90, 0)
    assert np.allclose(t.rotate(q, (0, 0, 1)), (1, 0, 0), atol=1e-9)


def test_multiply_applies_the_right_operand_first():
    a = t.from_axis_angle((0, 1, 0), 0.7)
    b = t.from_axis_angle((1, 0, 0), -0.4)
    v = (0.3, -1.2, 2.0)
    assert np.allclose(t.rotate(t.multiply(a, b), v), t.rotate(a, t.rotate(b, v)))


def test_euler_order_is_z_then_x_then_y_as_in_unity():
    x, y, z = 30, 45, 60
    expected = ry(math.radians(y)) @ rx(math.radians(x)) @ rz(math.radians(z))
    assert np.allclose(t.to_matrix(t.from_euler(x, y, z)), expected)


def test_euler_round_trip_over_the_whole_range():
    rng = np.random.default_rng(5)
    for _ in range(300):
        x = rng.uniform(-89, 89)
        y, z = rng.uniform(-179, 179, 2)
        back = t.to_euler(t.from_euler(x, y, z))
        assert np.allclose(back, (x, y, z), atol=1e-6)


def test_euler_round_trip_describes_the_same_rotation_at_the_poles():
    for x in (90, -90):
        q = t.from_euler(x, 33, 20)
        q2 = t.from_euler(*t.to_euler(q))
        assert np.allclose(t.to_matrix(q), t.to_matrix(q2), atol=1e-6)


def test_from_axis_angle_ignores_axis_length_and_rejects_zero():
    assert np.allclose(t.from_axis_angle((0, 5, 0), 1.0), t.from_axis_angle((0, 1, 0), 1.0))
    assert t.from_axis_angle((0, 0, 0), 1.0) == t.IDENTITY


def test_normalize_handles_a_degenerate_quaternion():
    assert t.normalize((0, 0, 0, 0)) == t.IDENTITY
    assert sum(c * c for c in t.normalize((1, 2, 3, 4))) == pytest.approx(1.0)
