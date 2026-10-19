import numpy as np
import pytest

from chronovox import selection as sel


def grid(size=(6, 5, 4)):
    """An empty matrix of x, y, z cells as a (z, y, x) array."""
    return np.zeros((size[2], size[1], size[0]), np.uint16)


def put(voxels, *cells, value=1):
    for i, (x, y, z) in enumerate(cells):
        voxels[z, y, x] = value + i if value == "seq" else value
    return voxels


def cells_of(mask):
    """The selected cells as a set of (x, y, z)."""
    zs, ys, xs = np.nonzero(mask)
    return set(zip(xs.tolist(), ys.tolist(), zs.tolist()))


def colored(voxels):
    zs, ys, xs = np.nonzero(voxels)
    return {(int(x), int(y), int(z)): int(voxels[z, y, x]) for x, y, z in zip(xs, ys, zs)}


# --- making a selection -------------------------------------------------------------------------

def test_a_box_selects_only_the_solid_voxels_inside_it_in_any_corner_order():
    v = put(grid(), (1, 1, 1), (2, 1, 1), (5, 4, 3))
    assert cells_of(sel.box_mask(v, (0, 0, 0), (2, 2, 2))) == {(1, 1, 1), (2, 1, 1)}
    assert cells_of(sel.box_mask(v, (2, 2, 2), (0, 0, 0))) == {(1, 1, 1), (2, 1, 1)}
    assert cells_of(sel.box_mask(v, (3, 2, 2), (4, 3, 3))) == set()                    # only empty space
    assert cells_of(sel.box_mask(v, (-5, -5, -5), (50, 50, 50))) == {(1, 1, 1), (2, 1, 1), (5, 4, 3)}   # clipped
    assert cells_of(sel.box_mask(v, (9, 9, 9), (12, 12, 12))) == set()                 # wholly outside


def test_the_wand_takes_the_connected_voxels_of_one_color_through_faces_only():
    v = grid()
    put(v, (1, 1, 1), (2, 1, 1), (3, 1, 1), value=1)
    put(v, (4, 1, 1), value=2)                        # touching, but another color
    put(v, (1, 2, 2), value=1)                        # same color, touching only along an edge of (2,1,1)? no: far away
    put(v, (5, 4, 3), value=1)                        # same color, not connected
    assert cells_of(sel.wand_mask(v, (2, 1, 1))) == {(1, 1, 1), (2, 1, 1), (3, 1, 1)}
    assert cells_of(sel.wand_mask(v, (4, 1, 1))) == {(4, 1, 1)}
    assert cells_of(sel.wand_mask(v, (0, 0, 0))) == set()                              # an empty seed
    assert cells_of(sel.wand_mask(v, (99, 0, 0))) == set()                             # outside


def test_bounds_and_indices_describe_a_selection_and_rebuild_it():
    v = put(grid(), (1, 2, 0), (4, 1, 3))
    mask = sel.all_mask(v)
    assert sel.bounds(mask) == ((1, 1, 0), (4, 2, 3))
    assert sel.bounds(np.zeros_like(mask)) is None
    assert np.array_equal(sel.mask_from(sel.indices_of(mask), mask.shape), mask)


def test_selected_cells_that_have_since_been_emptied_are_not_counted_as_solid():
    v = put(grid(), (1, 1, 1), (2, 2, 2))
    mask = sel.all_mask(v)
    v[1, 1, 1] = 0
    assert cells_of(sel.solid(mask, v)) == {(2, 2, 2)}


# --- moving -------------------------------------------------------------------------------------

def test_move_carries_voxels_and_colors_and_the_selection_with_them():
    v = grid()
    v[1, 1, 1], v[1, 1, 2] = 3, 4
    mask = sel.all_mask(v)
    moved, new_mask, used = sel.move(v, mask, (2, 1, 1))
    assert used == (2, 1, 1)
    assert colored(moved) == {(3, 2, 2): 3, (4, 2, 2): 4}
    assert cells_of(new_mask) == {(3, 2, 2), (4, 2, 2)}
    assert colored(v) == {(1, 1, 1): 3, (2, 1, 1): 4}                     # the input is untouched


def test_moved_voxels_replace_what_they_land_on_and_leave_empty_space_behind():
    v = grid()
    v[1, 1, 1] = 1                                    # selected
    v[1, 1, 2] = 9                                    # not selected, in the way
    mask = np.zeros(v.shape, bool)
    mask[1, 1, 1] = True
    moved, _, _ = sel.move(v, mask, (1, 0, 0))
    assert colored(moved) == {(2, 1, 1): 1}                                # the 9 was replaced, the old place is empty


def test_a_selection_can_move_over_its_own_old_place():
    v = grid()
    put(v, (1, 1, 1), (2, 1, 1), (3, 1, 1))
    for i, x in enumerate((1, 2, 3)):
        v[1, 1, x] = 10 + i
    moved, _, _ = sel.move(v, sel.all_mask(v), (1, 0, 0))
    assert colored(moved) == {(2, 1, 1): 10, (3, 1, 1): 11, (4, 1, 1): 12}


def test_move_stops_at_the_edge_of_the_matrix_and_never_loses_voxels():
    v = put(grid((6, 5, 4)), (1, 1, 1), (3, 1, 1))
    mask = sel.all_mask(v)
    assert sel.clamp_offset(mask, (100, 0, 0)) == (2, 0, 0)               # the highest voxel may reach x = 5
    assert sel.clamp_offset(mask, (-100, -100, -100)) == (-1, -1, -1)
    moved, _, used = sel.move(v, mask, (100, 100, 100))
    assert used == (2, 3, 2) and len(colored(moved)) == 2
    assert sel.clamp_offset(np.zeros_like(mask), (3, 3, 3)) == (0, 0, 0)


# --- flipping -----------------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flip_mirrors_across_the_middle_of_the_selection_along_one_axis(axis):
    v = grid()
    for i, cell in enumerate([(1, 1, 1), (2, 1, 1), (4, 2, 1), (1, 3, 2)]):
        v[cell[2], cell[1], cell[0]] = i + 1
    mask = sel.all_mask(v)
    (x0, y0, z0), (x1, y1, z1) = sel.bounds(mask)
    lo, hi = (x0, y0, z0)[axis], (x1, y1, z1)[axis]
    flipped, new_mask = sel.flip(v, mask, axis)
    expected = {}
    for cell, value in colored(v).items():
        mirrored = list(cell)
        mirrored[axis] = lo + hi - cell[axis]
        expected[tuple(mirrored)] = value
    assert colored(flipped) == expected and cells_of(new_mask) == set(expected)


def test_flipping_twice_gives_back_the_original():
    v = put(grid(), (1, 1, 1), (3, 2, 1), (2, 3, 2), value=7)
    v[2, 3, 2] = 8
    mask = sel.all_mask(v)
    once, once_mask = sel.flip(v, mask, 0)
    twice, twice_mask = sel.flip(once, once_mask, 0)
    assert np.array_equal(twice, v) and np.array_equal(twice_mask, mask)


def test_flip_only_touches_the_selected_voxels():
    v = put(grid(), (1, 1, 1), (3, 1, 1))
    v[1, 1, 5] = 9                                    # not selected
    mask = np.zeros(v.shape, bool)
    mask[1, 1, 1] = mask[1, 1, 3] = True
    flipped, _ = sel.flip(v, mask, 0)
    assert colored(flipped) == {(1, 1, 1): 1, (3, 1, 1): 1, (5, 1, 1): 9}


# --- rotating -----------------------------------------------------------------------------------

def test_a_quarter_turn_takes_the_positive_axes_around_in_a_right_handed_way():
    """About x: +y goes to +z. About y: +z goes to +x. About z: +x goes to +y."""
    for axis, start, end in ((0, (2, 4, 2), (2, 2, 4)), (1, (2, 2, 4), (4, 2, 2)), (2, (4, 2, 2), (2, 4, 2))):
        v = grid((5, 5, 5))
        v[0, 0, 0] = 5                                # two opposite corners fix the box at 5 x 5 x 5, centered on (2, 2, 2)
        v[4, 4, 4] = 6
        v[start[2], start[1], start[0]] = 1           # a voxel on the middle of the +axis face
        turned, mask = sel.rotate(v, sel.all_mask(v), axis)
        assert [cell for cell, value in colored(turned).items() if value == 1] == [end], axis
        assert len(cells_of(mask)) == 3


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_four_quarter_turns_give_back_the_original_for_a_cube(axis):
    v = grid((5, 5, 5))
    rng = np.random.default_rng(4)
    v[1:4, 1:4, 1:4] = rng.integers(0, 4, (3, 3, 3))
    mask = sel.all_mask(v)
    turned, turned_mask = v, mask
    for _ in range(4):
        turned, turned_mask = sel.rotate(turned, turned_mask, axis)
    assert np.array_equal(turned, v) and np.array_equal(turned_mask, mask)


def test_turning_the_other_way_undoes_a_turn():
    v = grid((5, 5, 5))
    put(v, (1, 1, 1), (2, 1, 1), (3, 2, 1), (2, 2, 3), value=3)           # 3 x 2 x 3: the two swapping sides match
    mask = sel.all_mask(v)
    there, there_mask = sel.rotate(v, mask, 1)
    back, back_mask = sel.rotate(there, there_mask, 1, quarter_turns=-1)
    assert np.array_equal(back, v) and np.array_equal(back_mask, mask)


def test_a_turn_of_a_long_block_swaps_its_sides_about_its_middle():
    v = grid((7, 5, 7))
    for x in range(1, 6):                              # a bar along x, 5 long, at y = 2, z = 3
        v[3, 2, x] = 1
    turned, mask = sel.rotate(v, sel.all_mask(v), 1)   # about y: the bar now runs along z
    assert cells_of(mask) == {(3, 2, z) for z in range(1, 6)}


def test_a_turn_that_cannot_fit_in_the_matrix_is_refused_and_one_near_the_edge_is_pulled_in():
    v = grid((7, 3, 3))
    v[1, 1, 1:7] = 1                                   # 6 long on x, but the matrix is only 3 deep on z
    assert sel.rotate(v, sel.all_mask(v), 1) is None
    w = grid((5, 5, 5))
    w[0, 0, 0:4] = 1                                   # a bar in a corner turned about y: stays inside the matrix
    turned, mask = sel.rotate(w, sel.all_mask(w), 1)
    assert mask.sum() == 4 and sel.bounds(mask)[0][0] >= 0 and sel.bounds(mask)[0][2] >= 0


def test_a_turn_replaces_what_is_in_the_way_and_only_moves_the_selected_voxels():
    v = grid((5, 5, 5))
    v[2, 2, 1] = 1
    v[2, 2, 3] = 1                                     # selected bar along x
    v[3, 2, 2] = 9                                     # unselected, where the turned bar will cross
    mask = np.zeros(v.shape, bool)
    mask[2, 2, 1] = mask[2, 2, 3] = True
    turned, _ = sel.rotate(v, mask, 1)
    # the bar along x is now a bar along z through the same middle; one of its voxels landed on the 9 and replaced it
    assert colored(turned) == {(2, 2, 1): 1, (2, 2, 3): 1}


# --- delete, copy and paste ---------------------------------------------------------------------

def test_delete_removes_only_the_selected_voxels():
    v = put(grid(), (1, 1, 1), (2, 2, 2))
    mask = np.zeros(v.shape, bool)
    mask[1, 1, 1] = True
    assert colored(sel.delete(v, mask)) == {(2, 2, 2): 1} and colored(v) == {(1, 1, 1): 1, (2, 2, 2): 1}


PALETTE = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], np.uint8)


def test_extract_keeps_the_colors_and_the_shape_of_the_selection():
    v = grid()
    v[1, 1, 1], v[1, 1, 2], v[2, 3, 4] = 1, 3, 2
    mask = np.zeros(v.shape, bool)
    mask[1, 1, 1] = mask[1, 1, 2] = mask[2, 3, 4] = True
    clip = sel.extract(v, mask, PALETTE)
    assert clip.size == (4, 3, 2) and clip.count == 3
    assert tuple(clip.rgba[0, 0, 0]) == (255, 0, 0, 255)                   # (1,1,1) is the clip's corner
    assert tuple(clip.rgba[0, 0, 1]) == (0, 0, 255, 255)
    assert tuple(clip.rgba[1, 2, 3]) == (0, 255, 0, 255)
    assert clip.rgba[0, 1, 0, 3] == 0                                      # an empty cell inside the box
    assert sel.extract(v, np.zeros_like(mask), PALETTE) is None


def test_paste_writes_the_block_at_the_origin_with_colors_matched_by_value():
    v = grid()
    v[1, 1, 1], v[1, 1, 2] = 1, 3
    clip = sel.extract(v, sel.all_mask(v), PALETTE)
    target = grid()
    new_palette = [(0, 0, 255, 255)]                                       # the target already has blue, as index 1

    def index_for(rgba):
        if rgba not in new_palette:
            new_palette.append(rgba)
        return new_palette.index(rgba) + 1

    pasted, mask, dropped = sel.paste(target, clip, (3, 2, 1), index_for)
    assert dropped == 0 and cells_of(mask) == {(3, 2, 1), (4, 2, 1)}
    assert colored(pasted) == {(3, 2, 1): 2, (4, 2, 1): 1}                 # red was added as 2; blue was found as 1
    assert new_palette == [(0, 0, 255, 255), (255, 0, 0, 255)]
    assert not target.any()                                                # the target itself is not changed


def test_paste_drops_what_falls_outside_and_fit_origin_pulls_a_block_inside():
    v = grid()
    put(v, (1, 1, 1), (2, 1, 1), (3, 1, 1))
    clip = sel.extract(v, sel.all_mask(v), PALETTE)
    target = grid((6, 5, 4))
    pasted, mask, dropped = sel.paste(target, clip, (4, 0, 0), lambda rgba: 1)         # x = 4, 5, 6: one is outside
    assert dropped == 1 and cells_of(mask) == {(4, 0, 0), (5, 0, 0)}
    assert sel.fit_origin((4, 0, 0), clip.size, (6, 5, 4)) == (3, 0, 0)              # moved in so that all three fit
    assert sel.fit_origin((-3, 9, 2), (3, 1, 1), (6, 5, 4)) == (0, 4, 2)
    assert sel.fit_origin((2, 2, 2), (9, 1, 1), (6, 5, 4)) == (0, 2, 2)              # wider than the matrix: cut off


# --- the selection holder and its undo entry ------------------------------------------------------

from chronovox.history import History, SelectionEdit, VoxelEdit
from chronovox.model import Matrix


def test_the_holder_gives_the_selection_only_to_its_own_matrix_and_only_while_it_fits():
    a, b = Matrix("a", grid()), Matrix("b", grid())
    holder = sel.VoxelSelection()
    mask = np.zeros(a.voxels.shape, bool)
    mask[1, 1, 1] = True
    holder.set(a, mask)
    assert holder.mask_for(a) is mask and holder.mask_for(b) is None
    a.voxels = grid((7, 5, 4))                                          # the matrix was resized: the selection is stale
    assert holder.mask_for(a) is None
    before = holder.version
    holder.set(a, np.zeros(a.voxels.shape, bool))                       # an empty selection is no selection
    assert holder.mask_for(a) is None and holder.version > before


def test_undoing_a_selection_edit_puts_back_both_the_voxels_and_the_selection():
    matrix = Matrix("m", put(grid(), (1, 1, 1), (2, 1, 1), value=3))
    holder = sel.VoxelSelection()
    mask = sel.all_mask(matrix.voxels)
    holder.set(matrix, mask)
    before = matrix.voxels.copy()
    moved, moved_mask, _ = sel.move(matrix.voxels, mask, (0, 2, 0))
    matrix.voxels[...] = moved
    edit = SelectionEdit(VoxelEdit.between(matrix, before), holder, matrix, sel.indices_of(mask),
                         sel.indices_of(moved_mask), "move")
    holder.set(matrix, moved_mask)
    history = History()
    history.push(edit)
    assert cells_of(holder.mask) == {(1, 3, 1), (2, 3, 1)}
    history.undo()
    assert np.array_equal(matrix.voxels, before) and cells_of(holder.mask_for(matrix)) == {(1, 1, 1), (2, 1, 1)}
    history.redo()
    assert colored(matrix.voxels) == {(1, 3, 1): 3, (2, 3, 1): 3} and cells_of(holder.mask_for(matrix)) == {(1, 3, 1), (2, 3, 1)}


def test_undoing_a_paste_that_started_with_no_selection_leaves_none():
    matrix = Matrix("m", grid())
    holder = sel.VoxelSelection()
    before = matrix.voxels.copy()
    pasted = put(grid(), (2, 2, 2))
    matrix.voxels[...] = pasted
    mask = sel.all_mask(pasted)
    edit = SelectionEdit(VoxelEdit.between(matrix, before), holder, matrix, np.zeros(0, np.uint32), sel.indices_of(mask), "paste")
    holder.set(matrix, mask)
    edit.undo()
    assert not matrix.voxels.any() and holder.mask_for(matrix) is None
    edit.redo()
    assert cells_of(holder.mask_for(matrix)) == {(2, 2, 2)} and edit.nbytes > 0


# --- where a pasted block goes ---------------------------------------------------------------------

def test_a_block_set_on_a_floor_rises_from_it_and_is_centered_across_it():
    # the empty cell above a floor: the surface touches its low y face, so the block extends up from the cell
    assert sel.origin_against_surface((5, 3, 7), 1, -1, (3, 4, 5)) == (4, 3, 5)


def test_a_block_set_against_a_face_that_looks_the_other_way_extends_the_other_way():
    # the surface touches the cell's high z face, so the block reaches back along -z and ends on the cell
    assert sel.origin_against_surface((5, 3, 7), 2, +1, (3, 4, 5)) == (4, 1, 3)
    assert sel.origin_against_surface((5, 3, 7), 0, -1, (1, 1, 1)) == (5, 3, 7)        # a single voxel goes on the cell itself


def test_a_block_placed_this_way_is_pulled_into_the_matrix_when_it_would_stick_out():
    origin = sel.origin_against_surface((0, 0, 0), 1, -1, (4, 2, 4))
    assert origin == (-2, 0, -2)
    assert sel.fit_origin(origin, (4, 2, 4), (8, 8, 8)) == (0, 0, 0)


# --- a floating selection keeps what it covers --------------------------------------------------------

def two_pieces():
    """A selected voxel (colour 1) at x = 1 and an unselected one (colour 2) at x = 3."""
    v = grid()
    v[1, 1, 1], v[1, 1, 3] = 1, 2
    mask = np.zeros(v.shape, bool)
    mask[1, 1, 1] = True
    return v, mask


def test_a_selection_moved_over_another_voxel_uncovers_it_when_it_moves_on():
    v, mask = two_pieces()
    under = sel.VoxelSelection().under_for(Matrix("m", v), v, mask)
    over, over_mask, _ = sel.move(v, mask, (2, 0, 0), under)
    assert colored(over) == {(3, 1, 1): 1}                               # on top of it: the 2 is hidden, not gone
    on, _, _ = sel.move(over, over_mask, (1, 0, 0), under)
    assert colored(on) == {(3, 1, 1): 2, (4, 1, 1): 1}                   # moved on: the 2 is back
    back, _, _ = sel.move(over, over_mask, (-2, 0, 0), under)
    assert colored(back) == {(1, 1, 1): 1, (3, 1, 1): 2}                 # and moving back leaves everything as it was


def test_without_a_floating_layer_the_covered_voxel_is_lost_as_before():
    v, mask = two_pieces()
    over, over_mask, _ = sel.move(v, mask, (2, 0, 0))
    on, _, _ = sel.move(over, over_mask, (1, 0, 0))
    assert colored(on) == {(4, 1, 1): 1}


def test_flipping_and_deleting_a_floating_selection_also_uncover_what_was_beneath():
    v, mask = two_pieces()
    holder = sel.VoxelSelection()
    under = holder.under_for(Matrix("m", v), v, mask)
    over, over_mask, _ = sel.move(v, mask, (2, 0, 0), under)               # now covering the 2
    assert colored(sel.delete(over, over_mask, under)) == {(3, 1, 1): 2}   # deleting it reveals the 2
    wide = grid()
    wide[1, 1, 1], wide[1, 1, 2] = 1, 1                                    # a two-voxel selection...
    wide[1, 1, 3] = 2                                                      # ...and a voxel it will be flipped onto
    wide_mask = np.zeros(wide.shape, bool)
    wide_mask[1, 1, 1] = wide_mask[1, 1, 2] = True
    under = sel.VoxelSelection().under_for(Matrix("m", wide), wide, wide_mask)
    moved, moved_mask, _ = sel.move(wide, wide_mask, (1, 0, 0), under)     # selection now at x = 2, 3, covering the 2
    flipped, _ = sel.flip(moved, moved_mask, 0, under)
    assert colored(flipped) == {(2, 1, 1): 1, (3, 1, 1): 1}                # flipped in place, still covering it
    off, _, _ = sel.move(flipped, moved_mask, (-1, 0, 0), under)
    assert colored(off) == {(1, 1, 1): 1, (2, 1, 1): 1, (3, 1, 1): 2}


def test_the_holder_starts_floating_keeps_floating_and_stops_when_the_grid_changes_behind_its_back():
    v, mask = two_pieces()
    matrix = Matrix("m", v)
    holder = sel.VoxelSelection()
    holder.set(matrix, mask)
    first = holder.under_for(matrix, v, mask)
    assert colored(first) == {(3, 1, 1): 2}                              # the grid with the selection lifted off
    assert holder.under_for(matrix, v, mask) is first                    # asked again: the same floating layer
    over, over_mask, _ = sel.move(v, mask, (2, 0, 0), first)
    matrix.voxels = over
    holder.set(matrix, over_mask, keep_under=True)
    assert holder.under_for(matrix, over, over_mask) is first            # the selection moved: still floating over it
    over[3, 4, 5] = 9                                                    # something else drew on the grid
    assert holder.under_for(matrix, over, over_mask) is not first        # so the old layer is given up


def test_a_new_selection_or_clearing_ends_the_floating():
    v, mask = two_pieces()
    matrix = Matrix("m", v)
    holder = sel.VoxelSelection()
    holder.set(matrix, mask)
    holder.under_for(matrix, v, mask)
    holder.set(matrix, mask)                                              # selected afresh
    assert holder.under is None
    holder.under_for(matrix, v, mask)
    holder.clear()
    assert holder.under is None and holder.under_matrix is None


def test_undoing_and_redoing_a_move_keeps_the_selection_floating():
    v, mask = two_pieces()
    matrix = Matrix("m", v)
    holder = sel.VoxelSelection()
    holder.set(matrix, mask)
    under = holder.under_for(matrix, v, mask)
    before = v.copy()
    over, over_mask, _ = sel.move(v, mask, (2, 0, 0), under)
    matrix.voxels[...] = over
    edit = SelectionEdit(VoxelEdit.between(matrix, before), holder, matrix, sel.indices_of(mask),
                         sel.indices_of(over_mask), "move", floating=True)
    holder.set(matrix, over_mask, keep_under=True)
    edit.undo()
    assert holder.under is under and colored(matrix.voxels) == {(1, 1, 1): 1, (3, 1, 1): 2}
    edit.redo()
    assert holder.under is under and colored(matrix.voxels) == {(3, 1, 1): 1}
    paste_like = SelectionEdit(None, holder, matrix, sel.indices_of(over_mask), sel.indices_of(mask), "paste")   # not a move
    paste_like.undo()
    assert holder.under is None                                           # a paste or delete changes the selection itself


# --- which plane a selection is dragged in ---------------------------------------------------------------

def test_grabbing_a_face_drags_in_the_plane_of_that_face():
    # a top face (normal +y) of the voxel at y = 4 lies at y = 5; its bottom face (normal -y) at y = 4
    assert sel.grab_plane((2, 4, 3), (0, 1, 0), (0.3, -1.0, 0.2)) == (1, 5.0)
    assert sel.grab_plane((2, 4, 3), (0, -1, 0), (0.3, 1.0, 0.2)) == (1, 4.0)
    assert sel.grab_plane((2, 4, 3), (1, 0, 0), (-1.0, 0.2, 0.1)) == (0, 3.0)         # the +x face is at x = 3
    assert sel.grab_plane((2, 4, 3), (0, 0, -1), (0.1, 0.2, 1.0)) == (2, 3.0)         # the -z face is at z = 3


def test_the_plane_follows_the_face_and_not_the_camera_direction():
    # the ray runs mostly along x, but the face it entered is a top face: still the plane of constant y
    axis, plane = sel.grab_plane((0, 0, 0), (0, 1, 0), (-1.0, -0.3, 0.1))
    assert (axis, plane) == (1, 1.0)


def test_a_face_seen_edge_on_or_no_face_at_all_falls_back_to_the_plane_facing_the_viewer():
    assert sel.grab_plane((2, 4, 3), (0, 1, 0), (1.0, -0.05, 0.1))[0] == 0           # nearly edge-on: use the facing plane
    assert sel.grab_plane((2, 4, 3), (0, 0, 0), (0.1, 0.2, -1.0)) == (2, 3.5)         # started inside: no face was entered
