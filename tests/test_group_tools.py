"""Move and rotate with several matrices selected: they act as one rigid piece."""
import math

import numpy as np

from chronovox import transform
from chronovox.gizmos import MatrixTool
from chronovox.history import GroupEdit, TransformEdit
from chronovox.matrix_tools import MatrixTools, group_anchor, matrix_center, pivot_world, rotate_about
from chronovox.model import Matrix

S = 128


def mat(name, x, y=0, z=0, size=(2, 2, 2), rotation=transform.IDENTITY):
    m = Matrix(name, np.ones((size[2], size[1], size[0]), np.uint16), (x * S, y * S, z * S))
    m.rotation = rotation
    return m


def grab(tools, tool, key, ray, matrix, others, anchor):
    tools.tool, tools.hover = tool, key
    assert tools.begin_drag(matrix, ray, others, anchor)


def ray_along_z(x, y):
    return (np.array([x, y, -50.0]), np.array([0.0, 0.0, 1.0]))


def ring_ray(point):
    """A ray down the Y axis through a point of the XZ plane, for grabbing a Y ring."""
    return (np.array([point[0], 50.0, point[2]]), np.array([0.0, -1.0, 0.0]))


def test_the_group_anchor_is_the_average_of_the_centers():
    a, b = mat("a", 0), mat("b", 10, size=(4, 2, 2))
    assert np.allclose(group_anchor([a, b]), ((1 + 12) / 2, 1, 1))
    assert np.allclose(group_anchor([a]), matrix_center(a))


def test_move_carries_every_selected_matrix_by_the_same_amount():
    a, b, c = mat("a", 0), mat("b", 10, 3), mat("c", -4, 0, 7)
    tools, anchor = MatrixTools(), group_anchor([a, b, c])
    grab(tools, MatrixTool.MOVE, ("axis", 0), ray_along_z(anchor[0] + 1, anchor[1]), a, [b, c], anchor)
    assert tools.update_drag(ray_along_z(anchor[0] + 3, anchor[1]), snap=True)           # 2 voxels along x
    assert [m.position for m in (a, b, c)] == [(2 * S, 0, 0), (12 * S, 3 * S, 0), (-2 * S, 0, 7 * S)]
    edit, error = tools.end_drag(commit=True)
    assert error is None and isinstance(edit, GroupEdit) and len(edit.edits) == 3
    assert [m.position for m in (a, b, c)] == [(2 * S, 0, 0), (12 * S, 3 * S, 0), (-2 * S, 0, 7 * S)]
    edit.undo()
    assert [m.position for m in (a, b, c)] == [(0, 0, 0), (10 * S, 3 * S, 0), (-4 * S, 0, 7 * S)]
    edit.redo()
    assert a.position == (2 * S, 0, 0)


def test_cancelling_a_group_move_puts_everything_back():
    a, b = mat("a", 0), mat("b", 6)
    tools, anchor = MatrixTools(), group_anchor([a, b])
    grab(tools, MatrixTool.MOVE, ("axis", 0), ray_along_z(anchor[0], anchor[1]), a, [b], anchor)
    tools.update_drag(ray_along_z(anchor[0] + 5, anchor[1]), snap=False)
    assert tools.end_drag(commit=False) == (None, None)
    assert a.position == (0, 0, 0) and b.position == (6 * S, 0, 0)


def test_a_single_matrix_still_gives_a_plain_transform_edit():
    a = mat("a", 0)
    tools = MatrixTools()
    x = pivot_world(a)[0]
    grab(tools, MatrixTool.MOVE, ("axis", 0), ray_along_z(x, 0), a, [], None)
    tools.update_drag(ray_along_z(x + 2, 0), snap=True)
    edit, _ = tools.end_drag(commit=True)
    assert isinstance(edit, TransformEdit)


def test_rotate_swings_every_matrix_about_the_center_of_the_group():
    a, b = mat("a", 0), mat("b", 8)                  # centers (1, 1, 1) and (9, 1, 1): the anchor is (5, 1, 1)
    anchor = group_anchor([a, b])
    centers = [matrix_center(m) for m in (a, b)]
    tools = MatrixTools()
    start = anchor + np.array([3.0, 0.0, 0.0])
    grab(tools, MatrixTool.ROTATE, ("ring", 1), ring_ray(start), a, [b], anchor)
    # a quarter turn about Y: the point at +x moves to -z or +z depending on handedness; follow whatever it does
    target = anchor + np.array([0.0, 0.0, 3.0])
    assert tools.update_drag(ring_ray(target), snap=True)
    angle = math.atan2(float(np.array([0, 1.0, 0]) @ np.cross(start - anchor, target - anchor)),
                       float((start - anchor) @ (target - anchor)))
    assert abs(abs(angle) - math.pi / 2) < 1e-9
    turn = transform.from_axis_angle(np.array([0, 1.0, 0]), angle)
    for m, c in zip((a, b), centers):
        expected = anchor + transform.rotate(turn, c - anchor)
        assert np.allclose(matrix_center(m), expected, atol=1 / S)       # the center went where a rigid turn puts it
    assert np.allclose(group_anchor([a, b]), anchor, atol=1 / S)         # and the middle of the group did not move
    assert abs(np.linalg.norm(matrix_center(a) - matrix_center(b)) - 8) < 1e-2     # the pieces stay 8 apart
    edit, _ = tools.end_drag(commit=True)
    assert isinstance(edit, GroupEdit)
    edit.undo()
    assert a.position == (0, 0, 0) and b.position == (8 * S, 0, 0)
    assert a.rotation == transform.IDENTITY and b.rotation == transform.IDENTITY


def test_rotating_about_its_own_pivot_only_turns_the_matrix():
    m = mat("m", 3, 1, 2)
    state = (m.position, m.pivot, m.rotation)
    position, rotation = rotate_about(m, state, pivot_world(m), np.array([0, 1.0, 0]), 1.0)
    assert position == m.position and rotation != m.rotation


def test_rotate_about_keeps_the_pivot_point_on_its_circle_at_resolution_and_with_an_odd_pivot():
    m = mat("m", 4, 0, 0, size=(4, 4, 4))
    m.resolution = 3
    m.pivot = (77, 200, 5)
    anchor = np.array([0.0, 0.0, 0.0])
    state = (m.position, m.pivot, m.rotation)
    position, rotation = rotate_about(m, state, anchor, np.array([0, 1.0, 0]), math.radians(90))
    before = np.asarray(m.position, float) / S + m.cell_size * np.asarray(m.pivot, float) / S
    m.position, m.rotation = position, rotation
    after = pivot_world(m)
    assert abs(np.linalg.norm(after) - np.linalg.norm(before)) < 2 / S        # same distance from the anchor
    assert abs(after[1] - before[1]) < 2 / S                                   # still the same height


# --- moving in a plane ------------------------------------------------------------------------

from chronovox.matrix_tools import PLANE_FILL_ALPHA, PLANE_FILL_ALPHA_ACTIVE, PLANE_SIZE, plane_square


def ray_down(x, z):
    return (np.array([x, 50.0, z]), np.array([0.0, -1.0, 0.0]))


def test_move_has_a_plane_handle_between_each_pair_of_axes_after_the_axis_handles():
    tools = MatrixTools()
    keys = [key for key, _ in tools.candidates(mat("a", 0), (0, 0, -20))]
    assert keys == [("axis", 0), ("axis", 1), ("axis", 2), ("plane", 0), ("plane", 1), ("plane", 2)]


def test_a_plane_square_lies_in_its_plane_with_one_corner_on_the_origin():
    points = plane_square((1.0, 2.0, 3.0), 1, 10.0)               # normal Y: the square is in XZ
    assert np.allclose(points[:, 1], 2.0)
    assert np.allclose(points[0], (1.0, 2.0, 3.0))                 # the corner touches the origin
    assert np.allclose(points[:, 0].max(), 1.0 + PLANE_SIZE * 10) and np.allclose(points[:, 2].max(), 3.0 + PLANE_SIZE * 10)
    assert np.allclose(points[0], points[4])                       # the first five points are the closed outline


def test_hover_finds_the_plane_square_by_its_inside_and_not_only_its_edges():
    m = mat("a", 0)
    anchor = pivot_world(m)
    view_proj = np.diag([0.1, 0.1, 0.001, 1.0])                        # looking down Z: the XY plane faces the viewer
    length = MatrixTools.handle_length(anchor, (0, 0, -10))
    middle = plane_square(anchor, 2, length, (0, 0, -10))[:4].mean(axis=0)     # the middle of the Z-normal square
    cursor = (100 + middle[0] * 10, 100 - middle[1] * 10)
    tools = MatrixTools()
    tools.update_hover(m, (0, 0, -10), view_proj, cursor, 200, 200)
    assert tools.hover == ("plane", 2)


def test_dragging_a_plane_moves_along_both_of_its_axes_and_not_the_third():
    m = mat("a", 0, 0, 0)
    m.position = (2 * S, 3 * S, 4 * S)
    tools, anchor = MatrixTools(), pivot_world(m)
    grab(tools, MatrixTool.MOVE, ("plane", 1), ray_down(anchor[0] + 1, anchor[2] + 1), m, [], None)
    assert tools.update_drag(ray_down(anchor[0] + 4.0, anchor[2] - 2.0), snap=True)         # +3 in x, -3 in z
    assert m.position == (5 * S, 3 * S, 1 * S)
    tools.update_drag(ray_down(anchor[0] + 1.25, anchor[2] + 1.5), snap=False)              # fine steps: +0.25, +0.5
    assert m.position == (2 * S + 32, 3 * S, 4 * S + 64)
    edit, error = tools.end_drag(commit=True)
    assert error is None and isinstance(edit, TransformEdit) and edit.old[0] == (2 * S, 3 * S, 4 * S)
    edit.undo()
    assert m.position == (2 * S, 3 * S, 4 * S)


def test_dragging_a_plane_carries_every_selected_matrix():
    a, b = mat("a", 0), mat("b", 10, 5, 0)
    tools, anchor = MatrixTools(), group_anchor([a, b])
    grab(tools, MatrixTool.MOVE, ("plane", 2), ray_along_z(anchor[0], anchor[1]), a, [b], anchor)     # the XY plane
    tools.update_drag(ray_along_z(anchor[0] + 2, anchor[1] - 1), snap=True)
    assert a.position == (2 * S, -1 * S, 0) and b.position == (12 * S, 4 * S, 0)
    edit, _ = tools.end_drag(commit=True)
    assert isinstance(edit, GroupEdit) and len(edit.edits) == 2


def test_a_plane_seen_edge_on_cannot_be_grabbed():
    m = mat("a", 0)
    tools = MatrixTools()
    tools.tool, tools.hover = MatrixTool.MOVE, ("plane", 1)
    assert not tools.begin_drag(m, (np.array([0.0, 5.0, 0.0]), np.array([1.0, 0.0, 0.0])))   # a ray along the plane


def test_the_overlay_draws_only_the_outline_of_a_plane_square_and_fills_it_faintly():
    m = mat("a", 0)
    tools = MatrixTools()
    segments, _ = tools.overlay(m, (0, 0, -20))
    triangles, colors = tools.fills(m, (0, 0, -20))
    assert len(triangles) == 6 and np.allclose(colors[:, 3], PLANE_FILL_ALPHA)            # two triangles per plane
    plain = len(segments)
    # three planes with four edges each and no diagonals: outlines add 12 segments to 3 arrows (+ chevrons) + pivot cross
    tools.tool = MatrixTool.ROTATE
    assert len(tools.fills(m, (0, 0, -20))[0]) == 0                                       # no tint for other tools
    tools.tool = MatrixTool.MOVE
    tools.hover = ("plane", 1)
    _, hovered = tools.fills(m, (0, 0, -20))
    assert np.allclose(hovered[2:4, 3], PLANE_FILL_ALPHA_ACTIVE) and np.allclose(hovered[[0, 1, 4, 5], 3], PLANE_FILL_ALPHA)
    assert len(tools.overlay(m, (0, 0, -20))[0]) == plain


def test_a_plane_being_dragged_stays_tinted_more():
    m = mat("a", 0)
    tools, anchor = MatrixTools(), pivot_world(m)
    grab(tools, MatrixTool.MOVE, ("plane", 2), ray_along_z(anchor[0], anchor[1]), m, [], None)
    tools.hover = None
    _, colors = tools.fills(m, (0, 0, -20))
    assert np.allclose(colors[4:6, 3], PLANE_FILL_ALPHA_ACTIVE) and np.allclose(colors[:4, 3], PLANE_FILL_ALPHA)


def test_a_plane_square_sits_in_the_quadrant_the_camera_looks_into():
    anchor = np.array([5.0, 5.0, 5.0])
    for eye, signs in (((9.0, 9.0, 0.0), (1, 1)), ((1.0, 9.0, 0.0), (-1, 1)), ((9.0, 1.0, 0.0), (1, -1)),
                       ((1.0, 1.0, 0.0), (-1, -1))):
        points = plane_square(anchor, 2, 10.0, eye)                 # normal Z: the square spans X and Y
        assert np.allclose(points[0], anchor)                       # one corner always on the origin
        assert np.allclose(points[2] - anchor, np.array([signs[0], signs[1], 0.0]) * PLANE_SIZE * 10.0)
        assert np.allclose(points[:, 2], 5.0)


def test_each_plane_uses_its_own_two_axes_for_the_quadrant():
    anchor = np.array([0.0, 0.0, 0.0])
    eye = (-4.0, 7.0, -2.0)
    x_normal = plane_square(anchor, 0, 10.0, eye)[2]                # spans Y and Z: toward +y and -z
    y_normal = plane_square(anchor, 1, 10.0, eye)[2]                # spans X and Z: toward -x and -z
    assert np.allclose(x_normal, (0, PLANE_SIZE * 10, -PLANE_SIZE * 10))
    assert np.allclose(y_normal, (-PLANE_SIZE * 10, 0, -PLANE_SIZE * 10))


def test_the_tinted_fill_follows_the_same_quadrant_as_the_outline():
    m = mat("a", 0)
    tools = MatrixTools()
    eye = (-30.0, 40.0, -50.0)
    triangles, _ = tools.fills(m, eye)
    outline = dict((key, pts) for key, pts in tools.candidates(m, eye))[("plane", 2)][:4]
    assert np.allclose(np.sort(triangles[4:6].reshape(-1, 3), axis=0)[[0, -1]], np.sort(outline, axis=0)[[0, -1]])
