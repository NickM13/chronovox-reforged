from pyglm import glm
import pytest

from chronovox.camera import Camera


def ndc(cam: Camera, point) -> glm.vec3:
    clip = cam.projection(1.5) * cam.view() * glm.vec4(*point, 1.0)
    return clip.xyz / clip.w


def toward_camera(cam: Camera, amount: float = 1.0) -> glm.vec3:
    return cam.target + cam._offset() * amount


def test_orbit_right_moves_near_point_right():
    cam = Camera()
    p = toward_camera(cam)
    before = ndc(cam, p).x
    cam.orbit(30, 0)
    assert ndc(cam, p).x > before


def test_orbit_down_moves_near_point_down():
    cam = Camera()
    p = toward_camera(cam)
    before = ndc(cam, p).y
    cam.orbit(0, 30)
    assert ndc(cam, p).y < before


def test_orbit_keeps_pivot_and_distance():
    cam = Camera()
    cam.target = glm.vec3(3, 4, 5)
    cam.orbit(100, -40)
    assert glm.distance(cam.eye(), cam.target) == pytest.approx(cam.distance)
    assert tuple(cam.target) == (3, 4, 5)


def test_pitch_is_clamped():
    cam = Camera()
    cam.orbit(0, 100000)
    assert cam.pitch < 1.571
    cam.orbit(0, -100000)
    assert cam.pitch > -1.571


def test_pan_keeps_point_under_cursor():
    cam = Camera()
    height = 800.0
    p = glm.vec3(cam.target)
    start = ndc(cam, p)
    cam.pan(40, 0, height)
    moved = ndc(cam, p)
    # cursor moved 40px right of an 800px-high, 1.5-aspect viewport: ndc = 2 * px / width
    assert moved.x - start.x == pytest.approx(2 * 40 / (height * 1.5), rel=1e-3)
    cam.pan(-40, 0, height)
    cam.pan(0, 40, height)
    assert ndc(cam, p).y - start.y == pytest.approx(-2 * 40 / height, rel=1e-3)


def test_pan_moves_scene_with_mouse():
    cam = Camera()
    p = glm.vec3(cam.target)
    cam.pan(0, 30, 800)  # drag down
    assert ndc(cam, p).y < 0
    cam = Camera()
    cam.pan(30, 0, 800)  # drag right
    assert ndc(cam, p).x > 0


def test_zoom_drag_right_zooms_in_and_left_zooms_out():
    cam = Camera()
    d = cam.distance
    cam.zoom_drag(50, 0)
    assert cam.distance < d
    cam.zoom_drag(-100, 0)
    assert cam.distance > d


def test_scroll_forward_zooms_in():
    cam = Camera()
    d = cam.distance
    cam.scroll(1)
    assert cam.distance < d


def test_orbit_never_moves_the_focus():
    cam = Camera()
    cam.focus_on(glm.vec3(3, 4, 5))
    cam.orbit(80, 25)
    cam.orbit(-200, -90)
    assert tuple(cam.target) == (3, 4, 5)
    assert glm.distance(cam.eye(), cam.target) == pytest.approx(cam.distance)


def test_orbit_right_turns_the_view_right():
    cam = Camera()
    right, _, forward = cam.axes()
    cam.orbit(30, 0)
    _, _, new_forward = cam.axes()
    assert glm.dot(new_forward, right) > 0 and glm.dot(new_forward, forward) > 0


def test_orbit_down_turns_the_view_down():
    cam = Camera()
    _, up, _ = cam.axes()
    cam.orbit(0, 30)
    _, _, new_forward = cam.axes()
    assert glm.dot(new_forward, up) < 0


def test_scroll_changes_distance_but_not_the_focus():
    cam = Camera()
    focus = glm.vec3(cam.target)
    cam.scroll(5)
    cam.scroll(-12)
    assert tuple(cam.target) == tuple(focus)


def test_fly_forward_moves_toward_what_is_ahead():
    cam = Camera()
    ahead = glm.vec3(cam.target)
    d = glm.distance(cam.eye(), ahead)
    cam.fly_move(0, 0, 1, 0.1, fast=False)
    assert glm.distance(cam.eye(), ahead) < d


def test_fly_up_moves_along_world_y():
    cam = Camera()
    y = cam.eye().y
    cam.fly_move(0, 1, 0, 0.5, fast=True)
    assert cam.eye().y > y


def test_fly_right_moves_screen_right():
    cam = Camera()
    p = glm.vec3(cam.target)
    cam.fly_move(1, 0, 0, 0.5, fast=False)
    assert ndc(cam, p).x < 0  # we moved right, so the point slides left


def test_focus_on_moves_the_pivot_but_keeps_angle_and_distance():
    cam = Camera()
    cam.distance = 25.0
    yaw, pitch = cam.yaw, cam.pitch
    cam.focus_on(glm.vec3(7, 8, 9))
    assert tuple(cam.target) == (7, 8, 9)
    assert (cam.yaw, cam.pitch, cam.distance) == (yaw, pitch, 25.0)
    assert glm.distance(cam.eye(), cam.target) == pytest.approx(25.0)


def test_orbit_after_focus_pivots_around_the_new_focus():
    cam = Camera()
    cam.focus_on(glm.vec3(10, 0, 0))
    cam.orbit(120, 20)
    assert glm.distance(cam.eye(), glm.vec3(10, 0, 0)) == pytest.approx(cam.distance)
    centre = ndc(cam, (10, 0, 0))
    assert abs(centre.x) < 1e-5 and abs(centre.y) < 1e-5  # still dead centre of the view


def test_wasd_moves_the_focus_with_the_camera():
    cam = Camera()
    start = glm.vec3(cam.target)
    cam.fly_move(0, 0, 1, 1.0, fast=False)
    assert glm.distance(cam.target, start) == pytest.approx(cam.fly_speed)
    assert glm.distance(cam.eye(), cam.target) == pytest.approx(cam.distance)


def test_gizmo_shows_all_six_directions_far_to_near():
    entries = Camera().gizmo_axes()
    assert sorted((n, sg) for n, sg, *_ in entries) == [(a, s) for a in "XYZ" for s in (-1, 1)]
    depths = [e[4] for e in entries]
    assert depths == sorted(depths, reverse=True)


def test_gizmo_positive_y_points_up_and_opposites_mirror():
    cam = Camera()
    by = {(n, sg): (x, y, d) for n, sg, x, y, d in cam.gizmo_axes()}
    assert by[("Y", 1)][1] > 0
    for axis in "XYZ":
        assert by[(axis, 1)] == pytest.approx(tuple(-v for v in by[(axis, -1)]))


def test_gizmo_looking_straight_down_puts_positive_y_toward_the_viewer():
    cam = Camera()
    cam.pitch = 1.5707  # clamped just short of straight down
    cam.orbit(0, 1000)
    by = {(n, sg): (x, y, d) for n, sg, x, y, d in cam.gizmo_axes()}
    x, y, depth = by[("Y", 1)]
    assert depth < -0.99 and abs(x) < 0.05 and abs(y) < 0.05
