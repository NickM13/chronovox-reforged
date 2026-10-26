from collections import deque

import numpy as np
import pytest

from chronovox.editing import (FILL_TOOLS, FillTarget, Tool, apply_fill, exit_face, flood_region, flood_region_3d,
                               find_target, pick_fill)
from chronovox.model import Matrix, Model

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)


def grid(mask_rows):
    return np.array([[c == "#" for c in row] for row in mask_rows])


def brute_force(mask, seed):
    region = np.zeros(mask.shape, bool)
    if not mask[seed]:
        return region
    queue = deque([seed])
    region[seed] = True
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not region[nr, nc]:
                region[nr, nc] = True
                queue.append((nr, nc))
    return region


# --- flood_region -------------------------------------------------------------------

def test_flood_region_fills_a_connected_shape_and_nothing_else():
    mask = grid(["##..#",
                 ".#..#",
                 ".###.",
                 "....#"])
    region = flood_region(mask, (0, 0))
    assert region.sum() == 6 and region[2, 3] and not region[0, 4] and not region[3, 4]


def test_diagonal_neighbours_are_not_connected():
    mask = grid(["#.", ".#"])
    assert flood_region(mask, (0, 0)).sum() == 1


def test_a_seed_on_a_false_cell_or_outside_gives_an_empty_region():
    mask = grid(["#."])
    assert not flood_region(mask, (0, 1)).any()
    assert not flood_region(mask, (5, 5)).any()
    assert not flood_region(mask, (-1, 0)).any()


def test_regions_wind_around_obstacles_and_stay_apart_when_walled_off():
    mask = ~grid(["#####",      # '.' cells are the ones that can be filled
                  "#..##",
                  "#.#.#",
                  "#.#.#"])
    left = flood_region(mask, (3, 1))
    assert left.sum() == 4 and left[1, 2] and not left[2, 3]       # goes up the left channel and along the top
    right = flood_region(mask, (3, 3))
    assert right.sum() == 2 and not right[1, 1]                    # the right channel is walled off
    assert np.array_equal(left, brute_force(mask, (3, 1)))


def test_flood_region_matches_a_plain_flood_fill_on_random_masks():
    rng = np.random.default_rng(2)
    for density in (0.35, 0.55, 0.75):
        for _ in range(15):
            mask = rng.random((17, 23)) < density
            r, c = rng.integers(0, 17), rng.integers(0, 23)
            assert np.array_equal(flood_region(mask, (int(r), int(c))), brute_force(mask, (int(r), int(c))))


def test_flood_region_on_a_large_flat_area_is_fast_and_complete():
    mask = np.ones((200, 200), bool)
    mask[::7, 50:150] = False                                      # many short walls: a comb the fill must snake through
    region = flood_region(mask, (3, 3))
    assert np.array_equal(region, brute_force(mask, (3, 3)))


# --- setting up matrices -------------------------------------------------------------------

def make(shape=(4, 6, 6)):
    """A model with one empty matrix; shape is (z, y, x)."""
    model = Model(np.zeros((0, 4), np.uint8), [Matrix("m", np.zeros(shape, np.uint16))])
    return model, model.matrices[0]


def put(model, m, cells, color):
    index = model.index_for_color(color)
    for x, y, z in cells:
        m.voxels[z, y, x] = index
    return index


# --- append ---------------------------------------------------------------------------------

def test_append_fills_the_layer_above_solid_voxels_whatever_their_color():
    model, m = make()
    put(model, m, [(x, 0, 2) for x in range(0, 3)], RED)
    put(model, m, [(x, 0, 2) for x in range(3, 5)], BLUE)         # a second color in the same slab
    target = FillTarget((0, 1, 2), axis=1, sign=1)                 # the cell above the left end
    assert apply_fill(model, m, Tool.FILL_APPEND, target, GREEN) == 5
    layer = m.voxels[2, 1, :]
    assert (layer[:5] == model.find_color(GREEN)).all() and layer[5] == 0
    assert m.voxels[2, 2].sum() == 0 and m.voxels[2, 0].sum() > 0  # nothing above, slab untouched


def test_append_only_covers_cells_that_have_something_behind_them():
    model, m = make()
    put(model, m, [(0, 0, 0), (1, 0, 0), (4, 0, 0)], RED)
    n = apply_fill(model, m, Tool.FILL_APPEND, FillTarget((0, 1, 0), 1, 1), GREEN)
    assert n == 2                                                  # (4,1,0) is a different, unconnected patch
    assert m.voxels[0, 1, 4] == 0 and m.voxels[0, 1, 2] == 0


def test_append_follows_the_surface_around_a_step_only_within_one_layer():
    model, m = make()
    put(model, m, [(x, 0, 1) for x in range(6)], RED)               # floor slab
    put(model, m, [(3, 1, 1), (4, 1, 1), (5, 1, 1)], RED)           # a raised block on the right
    # the top of the raised block (layer y=2) only has the block behind it
    assert apply_fill(model, m, Tool.FILL_APPEND, FillTarget((3, 2, 1), 1, 1), GREEN) == 3
    # the layer next to the floor (y=1): x 0..2 are free, the block's own cells are solid and excluded
    assert apply_fill(model, m, Tool.FILL_APPEND, FillTarget((0, 1, 1), 1, 1), GREEN) == 3
    # what was just filled is solid too, so a new layer above it is available
    assert apply_fill(model, m, Tool.FILL_APPEND, FillTarget((0, 2, 1), 1, 1), BLUE) == 3


def test_append_on_a_side_face_fills_a_vertical_layer():
    model, m = make()
    put(model, m, [(2, y, z) for y in range(3) for z in range(2)], RED)     # a wall at x=2
    n = apply_fill(model, m, Tool.FILL_APPEND, FillTarget((3, 1, 0), axis=0, sign=1), GREEN)
    assert n == 6 and (m.voxels[0:2, 0:3, 3] == model.find_color(GREEN)).all()
    assert m.voxels[:, :, 4].sum() == 0


def test_append_against_the_back_wall_fills_the_whole_empty_layer_but_not_around_blockers():
    model, m = make((2, 3, 4))
    put(model, m, [(1, 0, 0)], RED)                                 # one floor cell already taken
    n = apply_fill(model, m, Tool.FILL_APPEND, FillTarget((0, 0, 0), axis=1, sign=1), GREEN)
    assert n == 4 * 2 - 1 and m.voxels[:, 1].sum() == 0              # the y=0 layer only, minus the solid cell


def test_append_against_the_back_wall_stops_at_a_complete_wall_of_solids():
    model, m = make((2, 3, 4))
    put(model, m, [(1, 0, 0), (1, 0, 1)], RED)                      # both cells across the width are taken
    n = apply_fill(model, m, Tool.FILL_APPEND, FillTarget((0, 0, 0), axis=1, sign=1), GREEN)
    assert n == 2                                                  # only the strip on the seed's side


def test_append_cannot_leak_into_a_pocket_cut_off_by_solid_cells():
    model, m = make((1, 1, 7))
    put(model, m, [(3, 0, 0)], RED)
    n = apply_fill(model, m, Tool.FILL_APPEND, FillTarget((0, 0, 0), axis=1, sign=1), GREEN)
    assert n == 3 and m.voxels[0, 0, 4:].sum() == 0


def test_append_adds_the_color_to_the_palette_once():
    model, m = make()
    put(model, m, [(0, 0, 0)], RED)
    apply_fill(model, m, Tool.FILL_APPEND, FillTarget((0, 1, 0), 1, 1), GREEN)
    assert len(model.palette) == 2


# --- 3D flood region -------------------------------------------------------------------------------

def brute_force_3d(mask, seed):
    """Plain breadth-first fill over face-adjacent voxels. mask is (z, y, x); seed is (x, y, z)."""
    sx, sy, sz = seed
    region = np.zeros(mask.shape, bool)
    if not mask[sz, sy, sx]:
        return region
    queue = deque([(sz, sy, sx)])
    region[sz, sy, sx] = True
    while queue:
        z, y, x = queue.popleft()
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nz, ny, nx = z + dz, y + dy, x + dx
            if (0 <= nz < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nx < mask.shape[2]
                    and mask[nz, ny, nx] and not region[nz, ny, nx]):
                region[nz, ny, nx] = True
                queue.append((nz, ny, nx))
    return region


def test_flood_region_3d_goes_around_corners_and_through_all_three_axes():
    mask = np.zeros((3, 3, 3), bool)
    for z, y, x in [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1), (2, 2, 1)]:   # a winding path
        mask[z, y, x] = True
    region = flood_region_3d(mask, (0, 0, 0))
    assert np.array_equal(region, mask)


def test_voxels_touching_only_along_an_edge_or_a_corner_are_not_connected():
    mask = np.zeros((2, 2, 2), bool)
    mask[0, 0, 0] = mask[1, 1, 1] = True          # corner to corner
    assert flood_region_3d(mask, (0, 0, 0)).sum() == 1
    mask[:] = False
    mask[0, 0, 0] = mask[0, 1, 1] = True          # edge to edge
    assert flood_region_3d(mask, (0, 0, 0)).sum() == 1


def test_flood_region_3d_with_a_bad_seed_is_empty():
    mask = np.zeros((2, 2, 2), bool)
    mask[0, 0, 0] = True
    assert not flood_region_3d(mask, (1, 1, 1)).any()      # a False cell
    assert not flood_region_3d(mask, (5, 0, 0)).any()      # outside
    assert not flood_region_3d(mask, (-1, 0, 0)).any()


def test_flood_region_3d_matches_a_plain_flood_fill_on_random_masks():
    rng = np.random.default_rng(8)
    for density in (0.2, 0.35, 0.5, 0.7):
        for _ in range(12):
            mask = rng.random((7, 9, 11)) < density
            seed = (int(rng.integers(0, 11)), int(rng.integers(0, 9)), int(rng.integers(0, 7)))
            assert np.array_equal(flood_region_3d(mask, seed), brute_force_3d(mask, seed))


def test_flood_region_3d_on_thin_shapes_and_single_rows():
    for shape in ((1, 1, 20), (1, 20, 1), (20, 1, 1), (1, 1, 1), (3, 1, 30)):
        mask = np.ones(shape, bool)
        assert flood_region_3d(mask, (0, 0, 0)).all()
    mask = np.ones((1, 1, 9), bool)
    mask[0, 0, 4] = False
    assert flood_region_3d(mask, (0, 0, 0)).sum() == 4 and flood_region_3d(mask, (8, 0, 0)).sum() == 4


def test_flood_region_3d_on_a_large_hollow_shape_is_complete():
    n = 40
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((x - 19.5) ** 2 + (y - 19.5) ** 2 + (z - 19.5) ** 2)
    shell = (r < 18) & (r > 15)                                  # a hollow ball
    inner = (r < 12)                                             # and a separate solid core
    mask = shell | inner
    shell_region = flood_region_3d(mask, (19, 19, 3))            # starts in the shell (radius 16.5)
    core_region = flood_region_3d(mask, (19, 19, 19))
    assert shell_region.sum() == shell.sum() and core_region.sum() == inner.sum()
    assert not (shell_region & core_region).any()


# --- erase fill (3D) ------------------------------------------------------------------------------

def test_erase_fill_removes_everything_of_that_color_connected_in_3d():
    model, m = make()
    put(model, m, [(1, y, 1) for y in range(5)], RED)               # a column through five layers
    put(model, m, [(2, 2, 1), (3, 2, 1)], RED)                      # with an arm sticking out
    n = apply_fill(model, m, Tool.FILL_ERASE, FillTarget((1, 4, 1), axis=1, sign=1), RED)
    assert n == 7 and not m.voxels.any()


def test_erase_fill_stops_at_other_colors_and_leaves_unconnected_patches():
    model, m = make()
    put(model, m, [(x, 2, 1) for x in range(0, 3)], RED)
    put(model, m, [(3, 2, 1)], BLUE)
    put(model, m, [(4, 2, 1), (5, 2, 1)], RED)                      # same color but cut off by the blue voxel
    put(model, m, [(0, 5, 3)], RED)                                 # and a lone red one elsewhere
    n = apply_fill(model, m, Tool.FILL_ERASE, FillTarget((0, 2, 1), axis=1, sign=1), RED)
    assert n == 3
    assert m.voxels[1, 2, 0:3].sum() == 0
    assert m.voxels[1, 2, 3] == model.find_color(BLUE)
    assert m.voxels[1, 2, 4] == model.find_color(RED) and m.voxels[3, 5, 0] == model.find_color(RED)


def test_erase_fill_removes_a_whole_solid_cube_in_one_click():
    model, m = make((6, 6, 6))
    put(model, m, [(x, y, z) for x in range(6) for y in range(6) for z in range(6)], RED)
    assert apply_fill(model, m, Tool.FILL_ERASE, FillTarget((0, 5, 3), axis=1, sign=1), RED) == 216
    assert not m.voxels.any()


def test_erase_fill_follows_the_color_around_a_corner_between_faces():
    model, m = make()
    put(model, m, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)], RED)  # turns through x, y and z
    assert apply_fill(model, m, Tool.FILL_ERASE, FillTarget((0, 0, 0), axis=0, sign=-1), RED) == 4


def test_erase_fill_on_empty_space_does_nothing():
    model, m = make()
    assert apply_fill(model, m, Tool.FILL_ERASE, FillTarget((0, 0, 0), 1, 1), RED) == 0


# --- replace fill (3D) --------------------------------------------------------------------------------

def test_replace_fill_recolors_the_connected_same_colored_region_in_3d_only():
    model, m = make()
    put(model, m, [(x, 0, 0) for x in range(3)], RED)
    put(model, m, [(1, 1, 0), (1, 2, 0), (1, 2, 1)], RED)           # leaves the row in y and z
    put(model, m, [(3, 0, 0)], BLUE)
    put(model, m, [(4, 0, 0)], RED)
    n = apply_fill(model, m, Tool.FILL_REPLACE, FillTarget((1, 0, 0), axis=1, sign=1), GREEN)
    green = model.find_color(GREEN)
    assert n == 6 and (m.voxels[0, 0, :3] == green).all() and m.voxels[1, 2, 1] == green
    assert m.voxels[0, 0, 3] == model.find_color(BLUE) and m.voxels[0, 0, 4] == model.find_color(RED)


def test_replace_with_the_same_color_changes_nothing_and_adds_nothing():
    model, m = make()
    put(model, m, [(0, 0, 0), (1, 0, 0)], RED)
    before = m.voxels.copy()
    assert apply_fill(model, m, Tool.FILL_REPLACE, FillTarget((0, 0, 0), 1, 1), RED) == 0
    assert np.array_equal(m.voxels, before) and len(model.palette) == 1


def test_replace_fill_on_empty_space_does_nothing_and_adds_no_color():
    model, m = make()
    assert apply_fill(model, m, Tool.FILL_REPLACE, FillTarget((2, 2, 2), 1, 1), GREEN) == 0
    assert len(model.palette) == 0


def test_replace_merges_into_a_neighbouring_color_without_growing_the_palette():
    model, m = make()
    put(model, m, [(0, 0, 0)], RED)
    put(model, m, [(1, 0, 0)], BLUE)
    assert apply_fill(model, m, Tool.FILL_REPLACE, FillTarget((0, 0, 0), 1, 1), BLUE) == 1
    assert len(model.palette) == 2 and m.voxels[0, 0, 0] == m.voxels[0, 0, 1]


def test_out_of_bounds_targets_and_non_fill_tools_are_ignored():
    model, m = make()
    put(model, m, [(0, 0, 0)], RED)
    assert apply_fill(model, m, Tool.FILL_ERASE, FillTarget((9, 0, 0), 1, 1), RED) == 0
    assert apply_fill(model, m, Tool.PLACE, FillTarget((0, 0, 0), 1, 1), RED) == 0
    assert m.voxels[0, 0, 0] != 0


# --- picking -----------------------------------------------------------------------------------

def test_pick_fill_for_append_starts_in_front_of_the_clicked_face():
    v = np.zeros((4, 4, 4), np.uint16)
    v[0, 0, 0] = 1
    target = pick_fill(v, Tool.FILL_APPEND, (0.5, 9.0, 0.5), (0, -1, 0))
    assert target == FillTarget((0, 1, 0), axis=1, sign=1)
    side = pick_fill(v, Tool.FILL_APPEND, (9.0, 0.5, 0.5), (-1, 0, 0))
    assert side == FillTarget((1, 0, 0), axis=0, sign=1)
    below = pick_fill(v, Tool.FILL_APPEND, (0.5, -9.0, 0.5), (0, 1, 0))
    assert below is None                                            # the cell below is outside the grid


def test_pick_fill_for_append_without_a_hit_uses_the_back_wall():
    v = np.zeros((4, 4, 4), np.uint16)
    assert pick_fill(v, Tool.FILL_APPEND, (2.5, 9.0, 2.5), (0, -1, 0)) == FillTarget((2, 0, 2), axis=1, sign=1)
    assert pick_fill(v, Tool.FILL_APPEND, (-9.0, 1.5, 2.5), (1, 0, 0)) == FillTarget((3, 1, 2), axis=0, sign=-1)
    assert pick_fill(v, Tool.FILL_APPEND, (20.0, 20.0, 20.0), (0, 1, 0)) is None


def test_pick_fill_for_erase_and_replace_needs_a_voxel():
    v = np.zeros((4, 4, 4), np.uint16)
    v[1, 1, 1] = 1
    for tool in (Tool.FILL_ERASE, Tool.FILL_REPLACE):
        assert pick_fill(v, tool, (1.5, 9.0, 1.5), (0, -1, 0)) == FillTarget((1, 1, 1), axis=1, sign=1)
        assert pick_fill(v, tool, (3.5, 9.0, 3.5), (0, -1, 0)) is None


def test_exit_face_reports_the_wall_and_the_inward_direction():
    assert exit_face((4, 4, 4), (2.5, 9, 2.5), (0, -1, 0)) == ((2, 0, 2), 1, 1)
    assert exit_face((4, 4, 4), (-9, 1.5, 1.5), (1, 0, 0)) == ((3, 1, 1), 0, -1)
    assert exit_face((4, 4, 4), (-9, 9, 9), (1, 0, 0)) is None


def test_the_fill_tools_are_the_three_modes_and_other_tools_are_unaffected():
    assert [t.value for t in FILL_TOOLS] == ["Fill: append", "Fill: erase", "Fill: replace"]
    v = np.zeros((2, 2, 2), np.uint16)
    assert find_target(v, Tool.PLACE, (0.5, 9, 0.5), (0, -1, 0)) == (0, 0, 0)


def _two_blocks():
    """A 5x1x1 row: red, red, empty, red, near-red; palette 1 = red, 2 = near red, 3 = blue."""
    import numpy as np
    from chronovox.model import Matrix, Model
    palette = np.array([[200, 0, 0, 255], [210, 10, 10, 255], [0, 0, 200, 255]], np.uint8)
    matrix = Matrix("m", np.array([[[1, 1, 0, 1, 2]]], np.uint16), (0, 0, 0))
    return Model(palette, [matrix]), matrix


def test_similar_mask_matches_close_colors_only_within_the_tolerance():
    import numpy as np
    from chronovox.editing import similar_mask
    model, matrix = _two_blocks()
    assert similar_mask(matrix.voxels, model.palette, 1, 0).tolist() == [[[True, True, False, True, False]]]
    assert similar_mask(matrix.voxels, model.palette, 1, 5).tolist() == [[[True, True, False, True, True]]]
    assert similar_mask(matrix.voxels, model.palette, 3, 5).sum() == 0


def test_replace_fill_can_leave_the_gap_or_ignore_it_and_widen_with_tolerance():
    from chronovox.editing import FillTarget, Tool, apply_fill
    green = (0, 200, 0, 255)
    model, matrix = _two_blocks()
    assert apply_fill(model, matrix, Tool.FILL_REPLACE, FillTarget((0, 0, 0), 0, 1), green) == 2     # as before
    model, matrix = _two_blocks()
    assert apply_fill(model, matrix, Tool.FILL_REPLACE, FillTarget((0, 0, 0), 0, 1), green, contiguous=False) == 3
    model, matrix = _two_blocks()
    assert apply_fill(model, matrix, Tool.FILL_REPLACE, FillTarget((0, 0, 0), 0, 1), green, True, 5) == 2
    model, matrix = _two_blocks()
    assert apply_fill(model, matrix, Tool.FILL_REPLACE, FillTarget((0, 0, 0), 0, 1), green, False, 5) == 4


def test_the_wand_follows_contiguous_and_tolerance():
    from chronovox import selection
    model, matrix = _two_blocks()
    assert selection.wand_mask(matrix.voxels, (0, 0, 0)).sum() == 2
    assert selection.wand_mask(matrix.voxels, (0, 0, 0), model.palette, False, 0).sum() == 3
    assert selection.wand_mask(matrix.voxels, (0, 0, 0), model.palette, False, 5).sum() == 4
