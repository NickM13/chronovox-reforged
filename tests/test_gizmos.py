import math

import numpy as np
import pytest

from chronovox import gizmos as g


def test_closest_param_on_an_axis_follows_the_mouse_ray():
    assert g.closest_param_on_line((3, 2, 5), (0, 0, -1), (0, 0, 0), (1, 0, 0)) == pytest.approx(3.0)
    assert g.closest_param_on_line((3, 2, 5), (0, 0, -1), (10, 0, 0), (1, 0, 0)) == pytest.approx(-7.0)


def test_closest_param_matches_a_brute_force_search():
    rng = np.random.default_rng(1)
    for _ in range(50):
        o, d = rng.uniform(-5, 5, 3), rng.normal(size=3)
        p0, a = rng.uniform(-5, 5, 3), rng.normal(size=3)
        t = g.closest_param_on_line(o, d, p0, a)
        an = a / np.linalg.norm(a)

        def dist(tt):
            p = p0 + an * tt
            s = ((p - o) @ d) / (d @ d)
            return np.linalg.norm(p - (o + d * s))

        ts = np.linspace(t - 2, t + 2, 4001)
        assert abs(ts[np.argmin([dist(x) for x in ts])] - t) < 2e-3


def test_closest_param_is_none_for_a_parallel_ray():
    assert g.closest_param_on_line((0, 1, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0)) is None


def test_ray_plane_point():
    assert np.allclose(g.ray_plane_point((0, 0, 0), (1, 1, 1), (0, 0, 2), (0, 0, 1)), (2, 2, 2))
    assert g.ray_plane_point((0, 0, 0), (1, 0, 0), (0, 0, 2), (0, 0, 1)) is None       # parallel
    assert g.ray_plane_point((0, 0, 0), (0, 0, -1), (0, 0, 2), (0, 0, 1)) is None      # behind


def test_project_maps_clip_space_to_window_pixels_and_drops_points_behind():
    identity = np.eye(4)
    assert g.project(identity, (0, 0, 0), 200, 100) == pytest.approx((100, 50))
    assert g.project(identity, (1, 1, 0), 200, 100) == pytest.approx((200, 0))
    behind = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1.0]])
    assert g.project(behind, (0, 0, 0), 200, 100) is None


def test_segment_distance_2d():
    assert g.segment_distance_2d((5, 3), (0, 0), (10, 0)) == pytest.approx(3)
    assert g.segment_distance_2d((-4, 3), (0, 0), (10, 0)) == pytest.approx(5)   # past the end
    assert g.segment_distance_2d((1, 1), (2, 2), (2, 2)) == pytest.approx(math.sqrt(2))


def test_pick_handle_takes_the_nearest_within_the_threshold():
    identity = np.eye(4)
    x_axis = [(0, 0, 0), (0.5, 0, 0)]
    y_axis = [(0, 0, 0), (0, 0.5, 0)]
    candidates = [("x", x_axis), ("y", y_axis)]
    assert g.pick_handle(candidates, identity, (125, 51), 200, 100, threshold=5) == "x"
    assert g.pick_handle(candidates, identity, (100, 30), 200, 100, threshold=5) == "y"
    assert g.pick_handle(candidates, identity, (20, 90), 200, 100, threshold=5) is None


def test_ring_points_form_a_closed_circle_around_the_right_axis():
    pts = g.ring_points((1, 2, 3), 1, 4.0, 32)
    assert np.allclose(pts[0], pts[-1])
    assert np.allclose(pts[:, 1], 2)                                   # flat in the plane of axis y
    assert np.allclose(np.linalg.norm(pts - (1, 2, 3), axis=1), 4.0)   # all on the circle
