from chronovox.editing import FillTarget, mirror_boxes, mirror_cells, mirror_targets


def test_no_axes_gives_only_the_cell():
    assert mirror_cells((1, 2, 3), (8, 8, 8), (False, False, False)) == [(1, 2, 3)]


def test_single_axis_reflects_about_the_center():
    assert mirror_cells((1, 2, 3), (8, 8, 8), (True, False, False)) == [(1, 2, 3), (6, 2, 3)]


def test_all_axes_give_eight_cells_and_cells_on_the_plane_are_not_repeated():
    assert len(set(mirror_cells((1, 2, 3), (8, 8, 8), (True, True, True)))) == 8
    assert mirror_cells((1, 0, 0), (3, 4, 4), (True, False, False)) == [(1, 0, 0)]   # the middle column of 3


def test_box_reflects_both_corners():
    boxes = mirror_boxes((0, 0, 0), (1, 1, 1), (8, 8, 8), (True, False, False))
    assert boxes == [((0, 0, 0), (1, 1, 1)), ((7, 0, 0), (6, 1, 1))]


def test_fill_target_flips_direction_on_its_own_axis_only():
    t = FillTarget((1, 2, 3), 0, 1)
    assert mirror_targets(t, (8, 8, 8), (True, False, False)) == [t, FillTarget((6, 2, 3), 0, -1)]
    assert mirror_targets(t, (8, 8, 8), (False, True, False)) == [t, FillTarget((1, 5, 3), 0, 1)]


# --- radial symmetry ---
import numpy as np

from chronovox.editing import symmetric_box_cells, symmetric_cells, symmetric_targets


def test_four_copies_turn_a_cell_in_quarter_turns_around_the_axis():
    cells = symmetric_cells((6, 3, 4), (8, 8, 8), (False, False, False), (1, 4))
    assert len(set(cells)) == 4 and cells[0] == (6, 3, 4)
    assert all(c[1] == 3 for c in cells)                       # turning around y never changes y
    assert set(cells) == {(6, 3, 4), (3, 3, 6), (1, 3, 3), (4, 3, 1)}


def test_two_copies_are_the_opposite_cell():
    assert symmetric_cells((1, 2, 5), (8, 8, 8), (False, False, False), (1, 2)) == [(1, 2, 5), (6, 2, 2)]


def test_radial_and_mirror_combine():
    cells = symmetric_cells((6, 3, 4), (8, 8, 8), (False, True, False), (1, 4))
    assert len(set(cells)) == 8


def test_radial_box_covers_each_turn_of_the_box():
    cells = symmetric_box_cells((0, 0, 0), (0, 0, 0), (8, 8, 8), (False, False, False), (1, 4))
    assert {tuple(c) for c in cells} == {(0, 0, 0), (0, 0, 7), (7, 0, 7), (7, 0, 0)}


def test_fill_face_on_the_turning_axis_stays_and_a_side_face_follows_the_turn():
    top = FillTarget((1, 4, 2), 1, 1)
    assert all(t.axis == 1 and t.sign == 1 for t in symmetric_targets(top, (8, 8, 8), (False,) * 3, (1, 3)))
    side = FillTarget((6, 2, 3), 0, 1)
    turned = symmetric_targets(side, (8, 8, 8), (False,) * 3, (1, 4))
    assert len(turned) == 4 and {t.axis for t in turned} == {0, 2}
    assert len(symmetric_targets(side, (8, 8, 8), (False,) * 3, (1, 3))) == 1      # not a quarter turn: kept flat only


def test_growth_takes_in_every_copy_and_keeps_the_center_where_it_was():
    from chronovox.editing import symmetric_expansion
    none = (False, False, False)
    assert symmetric_expansion((-1, 2, 2), (8, 8, 8), none, None) == ((1, 0, 0), (0, 0, 0))     # as without symmetry
    assert symmetric_expansion((4, 2, 2), (8, 8, 8), none, None) is None
    assert symmetric_expansion((-1, 2, 2), (8, 8, 8), (True, False, False), None) == ((1, 0, 0), (1, 0, 0))
    assert symmetric_expansion((2, 8, 2), (8, 8, 8), (False, True, False), None) == ((0, 1, 0), (0, 1, 0))
    # turned around y, a cell past the x edge pushes the x and z extents out on both sides, and y is left alone
    low, high = symmetric_expansion((-1, 3, 4), (8, 8, 8), none, (1, 4))
    assert low == high and low[0] > 0 and low[2] > 0 and low[1] == 0
