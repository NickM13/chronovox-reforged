import math

import numpy as np
import pytest

from chronovox import transform
from chronovox.gizmos import HIGHLIGHT_COLOR, MatrixTool
from chronovox.history import ResizeEdit, TransformEdit
from chronovox.matrix_tools import MatrixTools, model_point, pick_matrix, pivot_world, to_local
from chronovox.model import Matrix


def make(size=(4, 4, 4), position=(0, 0, 0), pivot=None, rotation=transform.IDENTITY):
    m = Matrix("m", np.zeros((size[2], size[1], size[0]), np.uint16), position)
    if pivot is not None:
        m.pivot = pivot
    m.rotation = rotation
    return m


def ray_along_z(x, y):
    return (np.array([x, y, -50.0]), np.array([0.0, 0.0, 1.0]))


def ray_down(x, z):
    return (np.array([x, 50.0, z]), np.array([0.0, -1.0, 0.0]))


def grab(tools, m, tool, key, ray):
    tools.tool = tool
    tools.hover = key
    assert tools.begin_drag(m, ray)


# --- coordinate helpers -----------------------------------------------------------

def test_model_point_and_to_local_are_inverses_even_rotated_with_an_odd_pivot():
    m = make((5, 4, 3), position=(300, -90, 12), pivot=(77, 200, 5), rotation=transform.from_euler(10, 50, -20))
    for p in [(0, 0, 0), (2.5, 1, 3), (-4, 9, 1.25)]:
        assert np.allclose(to_local(m, model_point(m.position, m.pivot, m.rotation, p)), p)


def test_pivot_world_is_where_the_pivot_point_lands_in_the_model():
    m = make(position=(256, 0, 128), pivot=(64, 64, 64), rotation=transform.from_euler(0, 90, 0))
    assert np.allclose(pivot_world(m), (2.5, 0.5, 1.5))
    assert np.allclose(model_point(m.position, m.pivot, m.rotation, (0.5, 0.5, 0.5)), pivot_world(m))


# --- hover ------------------------------------------------------------------------

def test_hover_finds_the_move_handle_under_the_cursor():
    m = make(pivot=(0, 0, 0))
    view_proj = np.diag([0.1, 0.1, 0.001, 1.0])
    tools = MatrixTools()
    tools.update_hover(m, (0, 0, -10), view_proj, (106, 100), 200, 200)   # on the +X handle
    assert tools.hover == ("axis", 0)
    tools.update_hover(m, (0, 0, -10), view_proj, (100, 94), 200, 200)    # on the +Y handle
    assert tools.hover == ("axis", 1)
    tools.update_hover(m, (0, 0, -10), view_proj, (10, 190), 200, 200)    # nowhere near
    assert tools.hover is None


def test_hover_is_frozen_while_dragging():
    m, tools = make(), MatrixTools()
    grab(tools, m, MatrixTool.MOVE, ("axis", 0), ray_along_z(2, 0))
    tools.update_hover(m, (0, 0, -10), np.diag([0.1, 0.1, 0.001, 1.0]), (10, 190), 200, 200)
    assert tools.hover == ("axis", 0)


# --- move -------------------------------------------------------------------------

def test_move_follows_the_mouse_along_the_axis_in_fine_steps():
    m, tools = make(position=(3 * 128, 0, 0)), MatrixTools()
    anchor_x = pivot_world(m)[0]
    grab(tools, m, MatrixTool.MOVE, ("axis", 0), ray_along_z(anchor_x + 1, 0))
    assert tools.update_drag(ray_along_z(anchor_x + 3.3, 0), snap=False)
    assert m.position == (round(3 * 128 + 2.3 * 128), 0, 0)       # 678: not a whole voxel
    assert m.position[0] % 128 != 0


def test_move_snaps_to_whole_voxels_when_asked():
    m, tools = make(position=(3 * 128, 0, 0)), MatrixTools()
    anchor_x = pivot_world(m)[0]
    grab(tools, m, MatrixTool.MOVE, ("axis", 0), ray_along_z(anchor_x + 1, 0))
    tools.update_drag(ray_along_z(anchor_x + 3.3, 0), snap=True)
    assert m.position == (5 * 128, 0, 0)
    tools.update_drag(ray_along_z(anchor_x + 4.6, 0), snap=True)   # 3.6 voxels from the start: 6.6, so 7
    assert m.position == (7 * 128, 0, 0)


def test_move_only_changes_the_dragged_axis():
    m, tools = make(position=(10, 20, 30)), MatrixTools()
    ax = pivot_world(m)
    grab(tools, m, MatrixTool.MOVE, ("axis", 1), (np.array([ax[0], ax[1] + 1, -50.0]), np.array([0, 0, 1.0])))
    tools.update_drag((np.array([ax[0] + 9, ax[1] + 3, -50.0]), np.array([0, 0, 1.0])), snap=False)
    assert m.position[0] == 10 and m.position[2] == 30 and m.position[1] == 20 + 2 * 128


def test_finishing_a_move_returns_one_undo_entry_with_the_old_and_new_position():
    m, tools = make(position=(128, 0, 0)), MatrixTools()
    x = pivot_world(m)[0]
    grab(tools, m, MatrixTool.MOVE, ("axis", 0), ray_along_z(x, 0))
    tools.update_drag(ray_along_z(x + 2, 0), snap=True)
    edit, error = tools.end_drag(commit=True)
    assert error is None and isinstance(edit, TransformEdit)
    assert edit.old[0] == (128, 0, 0) and edit.new[0] == (3 * 128, 0, 0)
    assert m.position == (3 * 128, 0, 0) and tools.drag is None
    edit.undo()
    assert m.position == (128, 0, 0)


def test_cancelling_a_move_puts_the_matrix_back_and_makes_no_entry():
    m, tools = make(position=(128, 0, 0)), MatrixTools()
    x = pivot_world(m)[0]
    grab(tools, m, MatrixTool.MOVE, ("axis", 0), ray_along_z(x, 0))
    tools.update_drag(ray_along_z(x + 5, 0), snap=False)
    assert tools.end_drag(commit=False) == (None, None)
    assert m.position == (128, 0, 0)


def test_a_drag_that_ends_where_it_started_makes_no_entry():
    m, tools = make(), MatrixTools()
    x = pivot_world(m)[0]
    grab(tools, m, MatrixTool.MOVE, ("axis", 0), ray_along_z(x, 0))
    tools.update_drag(ray_along_z(x + 3, 0), snap=False)
    tools.update_drag(ray_along_z(x, 0), snap=False)
    assert tools.end_drag(commit=True) == (None, None)


# --- rotate -----------------------------------------------------------------------

def test_rotate_turns_the_matrix_with_the_mouse_around_the_pivot():
    m, tools = make(), MatrixTools()
    c = pivot_world(m)
    grab(tools, m, MatrixTool.ROTATE, ("ring", 1), ray_down(c[0] + 5, c[2]))
    assert tools.update_drag(ray_down(c[0], c[2] + 5), snap=False)
    v0, v1 = np.array([5.0, 0, 0]), np.array([0, 0, 5.0])
    assert np.allclose(transform.rotate(m.rotation, v0), v1, atol=1e-9)    # the point under the cursor follows it
    assert np.allclose(pivot_world(m), c)                                  # and the pivot does not move


def test_rotate_composes_with_the_existing_rotation_about_the_world_axis():
    start = transform.from_euler(0, 30, 0)
    m, tools = make(rotation=start), MatrixTools()
    c = pivot_world(m)
    grab(tools, m, MatrixTool.ROTATE, ("ring", 1), ray_down(c[0] + 5, c[2]))
    tools.update_drag(ray_down(c[0], c[2] + 5), snap=False)                # -90 degrees about +Y
    assert np.allclose(transform.to_matrix(m.rotation),
                       transform.to_matrix(transform.from_euler(0, -60, 0)), atol=1e-9)


def test_rotate_snaps_to_fifteen_degree_steps():
    m, tools = make(), MatrixTools()
    c = pivot_world(m)
    grab(tools, m, MatrixTool.ROTATE, ("ring", 1), ray_down(c[0] + 5, c[2]))
    angle = math.radians(40)                          # toward +z is negative about +Y, so aim at -40
    tools.update_drag(ray_down(c[0] + 5 * math.cos(angle), c[2] + 5 * math.sin(angle)), snap=True)
    assert transform.to_euler(m.rotation)[1] == pytest.approx(-45.0, abs=1e-6)


def test_cancelling_a_rotation_restores_it():
    m, tools = make(), MatrixTools()
    c = pivot_world(m)
    grab(tools, m, MatrixTool.ROTATE, ("ring", 1), ray_down(c[0] + 5, c[2]))
    tools.update_drag(ray_down(c[0], c[2] + 5), snap=False)
    tools.end_drag(commit=False)
    assert m.rotation == transform.IDENTITY


# --- resize -----------------------------------------------------------------------

def face_center_x(m, sign):
    return model_point(m.position, m.pivot, m.rotation, (m.size[0] if sign > 0 else 0, m.size[1] / 2, m.size[2] / 2))


def test_dragging_the_maximum_face_out_adds_cells_there():
    m, tools = make((4, 3, 2)), MatrixTools()
    f = face_center_x(m, +1)
    grab(tools, m, MatrixTool.RESIZE, ("face", (0, 1)), ray_along_z(f[0], f[1]))
    tools.update_drag(ray_along_z(f[0] + 3.2, f[1]), snap=False)
    assert tools.drag.cells == 3 and m.size == (4, 3, 2)                  # only previewed so far
    edit, error = tools.end_drag(commit=True)
    assert error is None and isinstance(edit, ResizeEdit)
    assert m.size == (7, 3, 2) and m.position == (0, 0, 0)


def test_dragging_the_minimum_face_out_adds_cells_there_and_moves_the_origin():
    m, tools = make((4, 3, 2)), MatrixTools()
    f = face_center_x(m, -1)
    grab(tools, m, MatrixTool.RESIZE, ("face", (0, -1)), ray_along_z(f[0], f[1]))
    tools.update_drag(ray_along_z(f[0] - 2.1, f[1]), snap=False)
    tools.end_drag(commit=True)
    assert m.size == (6, 3, 2) and m.position == (-2 * 128, 0, 0)


def test_dragging_a_face_inward_crops_but_never_below_one_cell():
    m, tools = make((4, 3, 2)), MatrixTools()
    f = face_center_x(m, +1)
    grab(tools, m, MatrixTool.RESIZE, ("face", (0, 1)), ray_along_z(f[0], f[1]))
    tools.update_drag(ray_along_z(f[0] - 50, f[1]), snap=False)
    tools.end_drag(commit=True)
    assert m.size == (1, 3, 2)


def test_resize_along_a_rotated_matrix_uses_its_own_axis():
    m = make((4, 3, 2), rotation=transform.from_euler(0, 90, 0))
    tools = MatrixTools()
    face = model_point(m.position, m.pivot, m.rotation, (4, 1.5, 1))       # centre of the +X face
    axis = transform.rotate(m.rotation, (1, 0, 0))                          # local +X, now pointing along -z... in the model
    down = np.array([0.0, -1.0, 0.0])
    above = np.array([0.0, 40.0, 0.0])
    grab(tools, m, MatrixTool.RESIZE, ("face", (0, 1)), (face + axis * 2 + above, down))
    assert tools.drag.t0 == pytest.approx(2.0)
    tools.update_drag((face + axis * 4.1 + above, down), snap=False)
    assert tools.drag.cells == 2                                           # 4.1 - 2.0 = 2.1 along the rotated axis


def test_cancelling_a_resize_changes_nothing():
    m, tools = make((4, 3, 2)), MatrixTools()
    f = face_center_x(m, +1)
    grab(tools, m, MatrixTool.RESIZE, ("face", (0, 1)), ray_along_z(f[0], f[1]))
    tools.update_drag(ray_along_z(f[0] + 5, f[1]), snap=False)
    assert tools.end_drag(commit=False) == (None, None)
    assert m.size == (4, 3, 2)


def test_a_resize_that_cannot_happen_reports_an_error_and_changes_nothing(monkeypatch):
    import chronovox.matrix_ops as ops
    monkeypatch.setattr(ops, "MAX_MATRIX_VOXELS", 10)
    m, tools = make((2, 2, 2)), MatrixTools()
    f = face_center_x(m, +1)
    grab(tools, m, MatrixTool.RESIZE, ("face", (0, 1)), ray_along_z(f[0], f[1]))
    tools.update_drag(ray_along_z(f[0] + 5, f[1]), snap=False)
    edit, error = tools.end_drag(commit=True)
    assert edit is None and error and m.size == (2, 2, 2)


# --- drawing and grabbing ---------------------------------------------------------

def test_overlay_highlights_the_hovered_handle_and_previews_a_resize():
    m, tools = make((4, 3, 2)), MatrixTools()
    tools.tool = MatrixTool.MOVE
    tools.hover = ("axis", 2)
    segments, colors = tools.overlay(m, (0, 0, -20))
    assert segments.shape[1:] == (2, 3) and len(segments) == len(colors)
    assert any(np.allclose(c[:3], HIGHLIGHT_COLOR) for c in colors)

    tools.tool = MatrixTool.RESIZE
    tools.hover = ("face", (0, 1))
    f = face_center_x(m, +1)
    assert tools.begin_drag(m, ray_along_z(f[0], f[1]))
    plain = len(tools.overlay(m, (0, 0, -20))[0])
    tools.update_drag(ray_along_z(f[0] + 2, f[1]), snap=False)
    assert len(tools.overlay(m, (0, 0, -20))[0]) == plain + 12            # the 12 edges of the new box


def test_begin_drag_needs_a_hovered_handle_and_a_ray_that_can_grab_it():
    m, tools = make(), MatrixTools()
    assert not tools.begin_drag(m, ray_along_z(0, 0))
    tools.hover = ("axis", 0)
    parallel = (np.array([0.0, 5.0, 0.0]), np.array([1.0, 0.0, 0.0]))      # runs alongside the X axis
    assert not tools.begin_drag(m, parallel)
    assert tools.drag is None


def test_drag_summary_describes_the_current_drag():
    m, tools = make(), MatrixTools()
    assert tools.drag_summary() == ""
    grab(tools, m, MatrixTool.MOVE, ("axis", 0), ray_along_z(pivot_world(m)[0], 0))
    assert tools.drag_summary().startswith("Position")


# --- picking a matrix by clicking -------------------------------------------------

def test_clicking_a_voxel_picks_its_matrix_and_the_nearer_one_wins():
    near, far = make((2, 2, 2), position=(0, 0, 0)), make((2, 2, 2), position=(0, 0, 5 * 128))
    near.voxels[:] = 1
    far.voxels[:] = 1
    ray = (np.array([1.0, 1.0, -10.0]), np.array([0.0, 0.0, 1.0]))
    assert pick_matrix([far, near], ray) == 1
    assert pick_matrix([None, near], ray) == 1                       # hidden ones are skipped
    assert pick_matrix([near, far], (ray[0] + [9, 0, 0], ray[1])) is None


def test_an_empty_matrix_is_picked_by_its_bounds_but_voxels_elsewhere_win():
    empty, solid = make((8, 8, 8)), make((2, 2, 2), position=(3 * 128, 3 * 128, 3 * 128))
    solid.voxels[:] = 1
    miss = (np.array([0.5, 0.5, -10.0]), np.array([0.0, 0.0, 1.0]))   # only crosses the empty matrix
    assert pick_matrix([empty, solid], miss) == 0
    through = (np.array([4.0, 4.0, -10.0]), np.array([0.0, 0.0, 1.0]))  # also hits the solid one's voxels
    assert pick_matrix([empty, solid], through) == 1


def test_picking_works_on_rotated_matrices():
    m = make((4, 1, 1), rotation=transform.from_euler(0, 90, 0))      # lies along z after rotation about its pivot (center)
    m.voxels[:] = 1
    c = pivot_world(m)
    ray = (np.array([c[0], c[1], c[2] - 30]) + [0, 0, 0], np.array([0.0, 0.0, 1.0]))
    assert pick_matrix([m], ray) == 0


def test_a_small_empty_matrix_inside_a_big_empty_one_can_be_picked():
    big, small = make((32, 32, 32)), make((4, 4, 4), position=(14 * 128, 14 * 128, 14 * 128))
    ray = (np.array([16.0, 16.0, -20.0]), np.array([0.0, 0.0, 1.0]))
    assert pick_matrix([big, small], ray) == 1
    assert pick_matrix([small, big], ray) == 0
    outside_small = (np.array([2.0, 2.0, -20.0]), np.array([0.0, 0.0, 1.0]))
    assert pick_matrix([big, small], outside_small) == 0
