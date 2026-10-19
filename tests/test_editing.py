import numpy as np
import pytest

from chronovox.editing import Tool, apply_box, apply_tool, exit_cell, find_target, raycast
from chronovox.model import Matrix, Model

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def grid(x=4, y=4, z=4):
    return np.zeros((z, y, x), np.uint16)


def test_ray_hits_voxel_and_reports_entered_face():
    v = grid()
    v[2, 1, 3] = 1  # cell (x=3, y=1, z=2)
    hit = raycast(v, (-5, 1.5, 2.5), (1, 0, 0))
    assert hit.cell == (3, 1, 2) and hit.normal == (-1, 0, 0)
    hit = raycast(v, (9, 1.5, 2.5), (-1, 0, 0))
    assert hit.cell == (3, 1, 2) and hit.normal == (1, 0, 0)


def test_ray_from_above_and_below():
    v = grid()
    v[0, 2, 1] = 1
    assert raycast(v, (1.5, 10, 0.5), (0, -1, 0)).normal == (0, 1, 0)
    assert raycast(v, (1.5, -10, 0.5), (0, 1, 0)).normal == (0, -1, 0)


def test_ray_misses_empty_grid_and_outside_bounds():
    v = grid()
    v[0, 0, 0] = 1
    assert raycast(v, (-5, 3.5, 3.5), (1, 0, 0)) is None
    assert raycast(grid(), (-5, 1.5, 1.5), (1, 0, 0)) is None
    assert raycast(v, (-5, 9, 0.5), (1, 0, 0)) is None  # parallel and outside the slab


def test_ray_returns_nearest_of_several():
    v = grid()
    v[:, 1, 1] = 1
    v[:, 1, 3] = 1
    assert raycast(v, (-3, 1.5, 1.5), (1, 0, 0)).cell == (1, 1, 1)


def test_ray_diagonal_matches_brute_force():
    rng = np.random.default_rng(3)
    v = (rng.random((6, 6, 6)) < 0.15).astype(np.uint16)
    ts = np.linspace(0, 40, 16001)[:, None]
    for _ in range(200):
        o = rng.uniform(-4, 10, 3)
        d = rng.normal(size=3)
        hit = raycast(v, o, d)
        cells = np.floor(o + d * ts).astype(int)
        inside = ((cells >= 0) & (cells < 6)).all(axis=1)
        solid = np.zeros(len(cells), bool)
        solid[inside] = v[cells[inside, 2], cells[inside, 1], cells[inside, 0]] != 0
        if solid.any():
            first = cells[solid.argmax()]
            assert hit is not None and hit.cell == tuple(int(i) for i in first)
        elif hit is not None:  # sampling can step over a grazing hit
            assert v[hit.cell[2], hit.cell[1], hit.cell[0]] != 0


def test_ray_starting_inside_solid_has_no_normal():
    v = np.ones((2, 2, 2), np.uint16)
    hit = raycast(v, (0.5, 0.5, 0.5), (1, 0, 0))
    assert hit.cell == (0, 0, 0) and hit.normal == (0, 0, 0)


def test_exit_cell_floor_from_above():
    assert exit_cell((4, 4, 4), (1.5, 5, 2.5), (0, -1, 0)) == (1, 0, 2)
    assert exit_cell((4, 4, 4), (9.5, 5, 2.5), (0, -1, 0)) is None   # misses the matrix
    assert exit_cell((4, 4, 4), (1.5, 5, 2.5), (0, 1, 0)) is None    # points away from it


def test_exit_cell_on_every_back_face():
    size = (4, 5, 6)
    assert exit_cell(size, (-3, 2.5, 3.5), (1, 0, 0)) == (3, 2, 3)    # +x wall
    assert exit_cell(size, (9, 2.5, 3.5), (-1, 0, 0)) == (0, 2, 3)    # -x wall
    assert exit_cell(size, (1.5, 2.5, -3), (0, 0, 1)) == (1, 2, 5)    # +z wall
    assert exit_cell(size, (1.5, 2.5, 9), (0, 0, -1)) == (1, 2, 0)    # -z wall
    assert exit_cell(size, (1.5, -4, 2.5), (0, 1, 0)) == (1, 4, 2)    # ceiling
    assert exit_cell(size, (1.5, 2.5, 3.5), (0, 0, 1)) == (1, 2, 5)    # starting inside


def test_exit_cell_diagonal_ray_leaves_through_the_nearest_wall():
    # heading right and slightly down: leaves through the +x wall
    assert exit_cell((4, 4, 4), (-2, 3.5, 1.5), (1, -0.2, 0)) == (3, 2, 1)
    # steeper: leaves through the floor instead
    assert exit_cell((4, 4, 4), (-2, 3.5, 1.5), (1, -1, 0)) == (1, 0, 1)


def test_place_target_is_cell_next_to_hit_face():
    v = grid()
    v[0, 0, 0] = 1
    assert find_target(v, Tool.PLACE, (0.5, 5, 0.5), (0, -1, 0)) == (0, 1, 0)
    assert find_target(v, Tool.ERASE, (0.5, 5, 0.5), (0, -1, 0)) == (0, 0, 0)


def test_place_target_falls_back_to_floor():
    assert find_target(grid(), Tool.PLACE, (2.5, 8, 2.5), (0, -1, 0)) == (2, 0, 2)
    assert find_target(grid(), Tool.ERASE, (2.5, 8, 2.5), (0, -1, 0)) is None
    assert find_target(grid(), Tool.PAINT, (2.5, 8, 2.5), (0, -1, 0)) is None


def test_place_target_on_a_side_wall():
    assert find_target(grid(), Tool.PLACE, (-5, 1.5, 2.5), (1, 0, 0)) == (3, 1, 2)


def test_erase_and_paint_use_the_back_face_only_when_asked():
    for tool in (Tool.ERASE, Tool.PAINT):
        assert find_target(grid(), tool, (-5, 1.5, 2.5), (1, 0, 0)) is None
        assert find_target(grid(), tool, (-5, 1.5, 2.5), (1, 0, 0), back_face=True) == (3, 1, 2)


def test_voxel_in_front_wins_over_the_back_face():
    v = grid()
    v[2, 1, 1] = 1
    assert find_target(v, Tool.ERASE, (-5, 1.5, 2.5), (1, 0, 0), back_face=True) == (1, 1, 2)


def test_place_target_out_of_bounds_is_none():
    v = grid(y=1)
    v[0, 0, 0] = 1
    assert find_target(v, Tool.PLACE, (0.5, 5, 0.5), (0, -1, 0)) is None


def make():
    model = Model(np.zeros((0, 4), np.uint8), [Matrix("m", grid())])
    return model, model.matrices[0]


def test_place_adds_palette_color_once():
    model, m = make()
    assert apply_tool(model, m, Tool.PLACE, (1, 2, 3), RED)
    assert apply_tool(model, m, Tool.PLACE, (0, 0, 0), RED)
    assert m.voxels[3, 2, 1] == 1 and m.voxels[0, 0, 0] == 1
    assert len(model.palette) == 1
    assert apply_tool(model, m, Tool.PLACE, (2, 2, 2), BLUE)
    assert m.voxels[2, 2, 2] == 2 and len(model.palette) == 2


def test_erase_and_noop_cases():
    model, m = make()
    apply_tool(model, m, Tool.PLACE, (1, 1, 1), RED)
    assert not apply_tool(model, m, Tool.PLACE, (1, 1, 1), RED)  # same color: unchanged
    assert apply_tool(model, m, Tool.ERASE, (1, 1, 1), RED)
    assert m.voxels.sum() == 0
    assert not apply_tool(model, m, Tool.ERASE, (1, 1, 1), RED)


def test_paint_only_recolors_existing_voxels():
    model, m = make()
    assert not apply_tool(model, m, Tool.PAINT, (0, 0, 0), RED)
    assert len(model.palette) == 0  # no palette growth for a no-op
    apply_tool(model, m, Tool.PLACE, (0, 0, 0), RED)
    assert apply_tool(model, m, Tool.PAINT, (0, 0, 0), BLUE)
    assert m.voxels[0, 0, 0] == 2


def test_out_of_bounds_edit_is_rejected():
    model, m = make()
    assert not apply_tool(model, m, Tool.PLACE, (4, 0, 0), RED)
    assert not apply_tool(model, m, Tool.PLACE, (-1, 0, 0), RED)


def test_index_for_color_respects_palette_limit(monkeypatch):
    import chronovox.model as mod
    monkeypatch.setattr(mod, "MAX_PALETTE", 1)
    model, _ = make()
    model.index_for_color(RED)
    with pytest.raises(ValueError):
        model.index_for_color(BLUE)


# --- strokes pick against the pre-stroke copy ---------------------------------

def test_picking_against_a_snapshot_does_not_stack_or_dig():
    model, m = make()
    m.voxels[0, 0, 0] = model.index_for_color(RED)
    base = m.voxels.copy()
    ray = ((0.5, 9, 0.5), (0, -1, 0))
    for _ in range(5):  # holding the button still: five frames of the same stroke
        apply_tool(model, m, Tool.PLACE, find_target(base, Tool.PLACE, *ray), BLUE)
    assert int((m.voxels > 0).sum()) == 2  # one voxel placed, not a tower of five
    for _ in range(5):
        apply_tool(model, m, Tool.ERASE, find_target(base, Tool.ERASE, *ray), BLUE)
    assert m.voxels[0, 0, 0] == 0 and m.voxels[0, 1, 0] != 0  # only the aimed voxel went


# --- box fills ------------------------------------------------------------------

def test_box_place_fills_inclusive_box_with_corners_in_any_order():
    model, m = make()
    assert apply_box(model, m, Tool.PLACE, (2, 3, 1), (0, 1, 0), RED)
    assert (m.voxels[0:2, 1:4, 0:3] == 1).all()
    assert int((m.voxels > 0).sum()) == 3 * 3 * 2
    assert len(model.palette) == 1


def test_box_is_clipped_to_the_grid():
    model, m = make()
    assert apply_box(model, m, Tool.PLACE, (2, 2, 2), (9, 9, 9), RED)
    assert int((m.voxels > 0).sum()) == 2 * 2 * 2
    assert not apply_box(model, m, Tool.PLACE, (8, 8, 8), (9, 9, 9), RED)


def test_box_erase_clears_only_the_box():
    model, m = make()
    apply_box(model, m, Tool.PLACE, (0, 0, 0), (3, 3, 3), RED)
    assert apply_box(model, m, Tool.ERASE, (1, 1, 1), (2, 2, 2), RED)
    assert int((m.voxels > 0).sum()) == 64 - 8
    assert not apply_box(model, m, Tool.ERASE, (1, 1, 1), (2, 2, 2), RED)


def test_box_paint_recolors_existing_voxels_only():
    model, m = make()
    apply_box(model, m, Tool.PLACE, (0, 0, 0), (1, 0, 0), RED)
    assert apply_box(model, m, Tool.PAINT, (0, 0, 0), (3, 0, 0), BLUE)
    assert m.voxels[0, 0, :].tolist() == [2, 2, 0, 0]


def test_box_paint_over_nothing_leaves_palette_alone():
    model, m = make()
    assert not apply_box(model, m, Tool.PAINT, (0, 0, 0), (3, 3, 3), RED)
    assert len(model.palette) == 0


def test_box_same_color_place_reports_no_change():
    model, m = make()
    apply_box(model, m, Tool.PLACE, (0, 0, 0), (1, 1, 1), RED)
    assert not apply_box(model, m, Tool.PLACE, (0, 0, 0), (1, 1, 1), RED)


def test_box_can_span_from_a_floor_cell_to_a_back_wall_cell():
    model, m = make()
    start = find_target(m.voxels, Tool.PLACE, (1.5, 9, 1.5), (0, -1, 0))        # floor
    end = find_target(m.voxels, Tool.PLACE, (-5, 3.5, 1.5), (1, 0, 0))          # +x wall, top row
    assert apply_box(model, m, Tool.PLACE, start, end, RED)
    assert int((m.voxels > 0).sum()) == 3 * 4 * 1  # x 1..3, y 0..3, z 1


# --- paint box that only replaces one color ---------------------------------------------------

GREEN = (0, 255, 0, 255)

def mixed_row():
    model, m = make()
    put_color = lambda cells, color: [setattr_cell(m, c, model.index_for_color(color)) for c in cells]
    put_color([(0, 0, 0), (1, 0, 0), (3, 0, 0)], RED)
    put_color([(2, 0, 0)], BLUE)
    return model, m


def setattr_cell(m, cell, index):
    x, y, z = cell
    m.voxels[z, y, x] = index


def test_paint_box_with_only_color_recolors_just_that_color():
    model, m = mixed_row()
    red, blue = model.find_color(RED), model.find_color(BLUE)
    assert apply_box(model, m, Tool.PAINT, (0, 0, 0), (3, 0, 0), GREEN, only_color=red)
    green = model.find_color(GREEN)
    assert m.voxels[0, 0, :4].tolist() == [green, green, blue, green]       # the blue one is left alone


def test_paint_box_with_only_color_skips_empty_cells_and_other_colors():
    model, m = mixed_row()
    blue = model.find_color(BLUE)
    apply_box(model, m, Tool.PAINT, (0, 0, 0), (5, 3, 3), GREEN, only_color=blue)
    assert m.voxels[0, 0, 2] == model.find_color(GREEN)
    assert int((m.voxels > 0).sum()) == 4 and (m.voxels == model.find_color(RED)).sum() == 3


def test_paint_box_with_only_color_and_nothing_of_that_color_changes_nothing_and_adds_no_color():
    model, m = mixed_row()
    colors = len(model.palette)
    assert not apply_box(model, m, Tool.PAINT, (5, 0, 0), (5, 3, 3), GREEN, only_color=1)
    assert not apply_box(model, m, Tool.PAINT, (0, 0, 0), (3, 0, 0), GREEN, only_color=99)
    assert len(model.palette) == colors


def test_paint_box_with_only_color_to_the_same_color_changes_nothing():
    model, m = mixed_row()
    red = model.find_color(RED)
    before = m.voxels.copy()
    assert not apply_box(model, m, Tool.PAINT, (0, 0, 0), (3, 0, 0), RED, only_color=red)
    assert np.array_equal(m.voxels, before)


def test_without_only_color_paint_box_still_recolors_everything_solid():
    model, m = mixed_row()
    assert apply_box(model, m, Tool.PAINT, (0, 0, 0), (3, 0, 0), GREEN)
    assert (m.voxels[0, 0, :4] == model.find_color(GREEN)).all() and int((m.voxels > 0).sum()) == 4


def test_only_color_is_ignored_by_place_and_erase_boxes():
    model, m = mixed_row()
    assert apply_box(model, m, Tool.ERASE, (0, 0, 0), (3, 0, 0), RED, only_color=model.find_color(RED))
    assert not m.voxels.any()
    assert apply_box(model, m, Tool.PLACE, (0, 0, 0), (1, 0, 0), GREEN, only_color=99)
    assert int((m.voxels > 0).sum()) == 2


def test_paint_fill_already_changes_only_the_clicked_color():
    from chronovox.editing import FillTarget, apply_fill
    model, m = mixed_row()
    n = apply_fill(model, m, Tool.FILL_REPLACE, FillTarget((0, 0, 0), 1, 1), GREEN)
    green = model.find_color(GREEN)
    assert n == 2 and m.voxels[0, 0, :4].tolist() == [green, green, model.find_color(BLUE), model.find_color(RED)]


# --- place_face: the face a Place click builds against -------------------------------------------------------

def test_place_face_is_the_side_of_the_new_cell_that_touches_the_hit_voxel():
    from chronovox.editing import find_target, place_face
    voxels = np.zeros((3, 3, 3), np.uint16)
    voxels[1, 1, 1] = 1
    # From +x looking toward -x: the cell is (2, 1, 1) and the face is its low-x side (it touches the voxel).
    assert place_face(voxels, (5.0, 1.5, 1.5), (-1.0, 0.0, 0.0)) == ((2, 1, 1), 0, -1)
    # From above, looking down: the cell is above the voxel and the face is its low-y side.
    assert place_face(voxels, (1.5, 5.0, 1.5), (0.0, -1.0, 0.0)) == ((1, 2, 1), 1, -1)
    # From -z looking toward +z.
    assert place_face(voxels, (1.5, 1.5, -4.0), (0.0, 0.0, 1.0)) == ((1, 1, 0), 2, 1)
    for origin, direction in (((5.0, 1.5, 1.5), (-1.0, 0.0, 0.0)), ((1.5, 5.0, 1.5), (0.0, -1.0, 0.0))):
        assert place_face(voxels, origin, direction)[0] == find_target(voxels, Tool.PLACE, origin, direction)


def test_place_face_on_an_empty_matrix_is_the_back_wall():
    from chronovox.editing import find_target, place_face
    voxels = np.zeros((3, 3, 3), np.uint16)
    # Looking down into the matrix: the floor is the back wall, so the face is the low-y side of a floor cell.
    cell, axis, side = place_face(voxels, (1.5, 9.0, 1.5), (0.0, -1.0, 0.0))
    assert (cell, axis, side) == ((1, 0, 1), 1, -1)
    assert cell == find_target(voxels, Tool.PLACE, (1.5, 9.0, 1.5), (0.0, -1.0, 0.0))
    # Looking toward +x from outside: the wall at x = 3, the high-x side of cell x = 2.
    assert place_face(voxels, (-5.0, 1.5, 1.5), (1.0, 0.0, 0.0)) == ((2, 1, 1), 0, 1)


def test_place_face_is_none_when_the_ray_misses_or_the_new_cell_would_be_outside():
    from chronovox.editing import place_face
    voxels = np.zeros((3, 3, 3), np.uint16)
    voxels[2, 1, 1] = 1
    assert place_face(voxels, (50.0, 50.0, 50.0), (0.0, 1.0, 0.0)) is None
    assert place_face(voxels, (1.5, 1.5, 9.0), (0.0, 0.0, -1.0)) is None   # hit voxel is on the top layer edge: cell outside


# --- faces on the edge of the matrix: the cell beyond them ----------------------------------------------------

def test_expansion_says_which_side_to_grow_for_a_cell_beyond_the_edge():
    from chronovox.editing import cell_after_expansion, expansion_for
    assert expansion_for((1, 1, 1), (4, 4, 4)) is None
    assert expansion_for((4, 1, 1), (4, 4, 4)) == ((0, 0, 0), (1, 0, 0))                 # past the high x edge
    assert expansion_for((1, -1, 1), (4, 4, 4)) == ((0, 1, 0), (0, 0, 0))                # past the low y edge
    assert cell_after_expansion((1, -1, 1), (0, 1, 0)) == (1, 0, 1)                      # a low side shifts the numbering up
    assert cell_after_expansion((4, 1, 1), (0, 0, 0)) == (4, 1, 1)                       # a high side does not


def test_place_face_gives_the_cell_beyond_the_edge_only_when_asked_to():
    from chronovox.editing import place_face
    voxels = np.zeros((3, 3, 3), np.uint16)
    voxels[1, 1, 2] = 1                                                                  # a voxel on the high x edge
    voxels[1, 1, 0] = 1                                                                  # and one on the low x edge
    from_high = ((9.0, 1.5, 1.5), (-1.0, 0.0, 0.0))                                      # looking at the voxel at x = 2 from +x
    assert place_face(voxels, *from_high) is None
    assert place_face(voxels, *from_high, outside=True) == ((3, 1, 1), 0, -1)
    from_low = ((-9.0, 1.5, 1.5), (1.0, 0.0, 0.0))
    assert place_face(voxels, *from_low, outside=True) == ((-1, 1, 1), 0, 1)
    assert place_face(voxels, ((1.5, 9.0, 1.5)), (0.0, -1.0, 0.0), outside=True)[0] == (1, 0, 1)   # inside is unchanged


def test_fill_append_can_start_beyond_the_edge_when_asked():
    from chronovox.editing import Tool, pick_fill
    voxels = np.zeros((3, 3, 3), np.uint16)
    voxels[:, :, 2] = 1                                                                  # a wall at the high x edge
    ray = ((9.0, 1.5, 1.5), (-1.0, 0.0, 0.0))
    assert pick_fill(voxels, Tool.FILL_APPEND, *ray) is None
    target = pick_fill(voxels, Tool.FILL_APPEND, *ray, outside=True)
    assert target.cell == (3, 1, 1) and target.axis == 0 and target.sign == 1
    assert pick_fill(voxels, Tool.FILL_ERASE, *ray, outside=True).cell == (2, 1, 1)       # erase is not affected
