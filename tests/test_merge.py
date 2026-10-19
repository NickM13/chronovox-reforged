import numpy as np
import pytest

from chronovox import transform
from chronovox.history import History, VoxelEdit
from chronovox.matrix_ops import merge_matrix, merge_offset, merge_problem, plan_merge
from chronovox.matrix_tools import model_point
from chronovox.model import POSITION_SCALE, Matrix, Model

S = POSITION_SCALE


def world_voxels(matrix):
    """Every solid voxel of a matrix as {model position of its center (rounded): palette index}."""
    out = {}
    for z, y, x in np.argwhere(matrix.voxels != 0):
        center = model_point(matrix.position, matrix.pivot, matrix.rotation, (x + .5, y + .5, z + .5))
        out[tuple(np.round(center, 4))] = int(matrix.voxels[z, y, x])
    return out


def make_model(*matrices):
    model = Model(np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], np.uint8))
    for m in matrices:
        model.matrices.append(m)
    return model


def mat(name, size, position=(0, 0, 0), cells=(), rotation=transform.IDENTITY):
    m = Matrix(name, np.zeros((size[2], size[1], size[0]), np.uint16), position)
    m.rotation = rotation
    for (x, y, z), index in cells:
        m.voxels[z, y, x] = index
    return m


def test_merging_keeps_every_voxel_exactly_where_it_was_in_the_model():
    a = mat("a", (3, 3, 3), (0, 0, 0), [((0, 0, 0), 1), ((2, 2, 2), 1)])
    b = mat("b", (2, 2, 2), (5 * S, 1 * S, 0), [((0, 0, 0), 2), ((1, 1, 1), 3)])   # beside it, on the +x side
    before = {**world_voxels(a), **world_voxels(b)}
    model = make_model(a, b)
    merge_matrix(model, b, a)
    assert world_voxels(a) == before and model.matrices == [a]


def test_the_target_grows_on_the_low_side_when_the_source_sits_below_and_behind_it():
    a = mat("a", (2, 2, 2), (4 * S, 4 * S, 4 * S), [((0, 0, 0), 1)])
    b = mat("b", (2, 2, 2), (1 * S, 3 * S, 0), [((0, 0, 0), 2)])          # at lower x, y and z
    before = {**world_voxels(a), **world_voxels(b)}
    model = make_model(a, b)
    merge_matrix(model, b, a)
    assert a.size == (5, 3, 6) and world_voxels(a) == before
    assert a.position == (1 * S, 3 * S, 0)                               # the origin corner moved to where b's was


def test_only_the_sources_occupied_cells_count_not_its_empty_margins():
    a = mat("a", (4, 4, 4), (0, 0, 0), [((0, 0, 0), 1)])
    b = mat("b", (20, 20, 20), (2 * S, 0, 0), [((1, 2, 3), 2)])         # one voxel in a big empty matrix
    model = make_model(a, b)
    merge_matrix(model, b, a)
    assert a.size == (4, 4, 4) and a.voxels[3, 2, 3] == 2                # fits inside, so no growth at all


def test_where_both_have_a_voxel_the_source_wins_and_its_empty_cells_change_nothing():
    a = mat("a", (2, 1, 1), (0, 0, 0), [((0, 0, 0), 1), ((1, 0, 0), 1)])
    b = mat("b", (2, 1, 1), (0, 0, 0), [((1, 0, 0), 3)])                # cell 0 is empty, cell 1 overlaps
    model = make_model(a, b)
    merge_matrix(model, b, a)
    assert a.voxels[0, 0].tolist() == [1, 3]


def test_merging_an_empty_matrix_just_removes_it():
    a = mat("a", (2, 2, 2), (0, 0, 0), [((0, 0, 0), 1)])
    b = mat("b", (9, 9, 9), (50 * S, 0, 0))
    model = make_model(a, b)
    before = a.voxels.copy()
    merge_matrix(model, b, a)
    assert np.array_equal(a.voxels, before) and model.matrices == [a]


def test_matrices_with_the_same_rotation_merge_along_their_own_axes():
    q = transform.from_euler(0, 90, 0)
    a = mat("a", (3, 1, 3), (0, 0, 0), [((0, 0, 0), 1)], rotation=q)
    pivot_a = np.array(a.pivot) / S
    origin_a = np.array(a.position) / S + pivot_a - transform.rotate(q, pivot_a)       # where a's cell (0, 0, 0) is
    b = mat("b", (2, 1, 3), (0, 0, 0), [((0, 0, 0), 2), ((1, 0, 2), 3)], rotation=q)
    pivot_b = np.array(b.pivot) / S
    wanted_origin_b = origin_a + transform.rotate(q, (3, 0, 0))                         # three cells along a's own x
    b.position = tuple(int(round(v * S)) for v in wanted_origin_b - pivot_b + transform.rotate(q, pivot_b))
    before = {**world_voxels(a), **world_voxels(b)}
    model = make_model(a, b)
    merge_matrix(model, b, a)
    assert world_voxels(a) == before and a.size == (5, 1, 3)


def test_the_cells_line_up_even_with_an_odd_pivot_on_either_matrix():
    a = mat("a", (3, 3, 3), (2 * S, 0, 0), [((0, 0, 0), 1)])
    b = mat("b", (2, 2, 2), (7 * S, 1 * S, 1 * S), [((1, 1, 1), 2)])
    a.pivot, b.pivot = (37, 200, 5), (90, 12, 333)                         # arbitrary, off-center
    before = {**world_voxels(a), **world_voxels(b)}
    merge_matrix(make_model(a, b), b, a)
    assert world_voxels(a) == before


# --- things that cannot be merged ---------------------------------------------------------------

def test_different_rotations_are_refused_with_a_reason():
    a = mat("Torso", (2, 2, 2), rotation=transform.from_euler(0, 30, 0))
    b = mat("Arm", (2, 2, 2))
    assert "rotated differently" in merge_problem(b, a)
    model = make_model(a, b)
    with pytest.raises(ValueError, match="rotated differently"):
        merge_matrix(model, b, a)
    assert model.matrices == [a, b]


def test_q_and_minus_q_are_the_same_rotation():
    q = transform.from_euler(10, 20, 30)
    a, b = mat("a", (1, 1, 1), rotation=q), mat("b", (1, 1, 1), rotation=tuple(-c for c in q))
    assert merge_problem(a, b) is None


def test_a_fractional_offset_between_the_grids_is_refused_and_a_whole_one_is_fine():
    a, b = mat("a", (2, 2, 2)), mat("b", (2, 2, 2), (S // 2, 0, 0))
    problem = merge_problem(b, a)
    assert "voxel grid" in problem and "0.5" in problem
    b.position = (3 * S, 0, 0)
    assert merge_problem(b, a) is None
    b.position = (3 * S + 1, 0, 0)                                          # off by 1/128
    assert merge_problem(b, a) is not None


def test_a_matrix_cannot_be_merged_into_itself():
    a = mat("a", (1, 1, 1))
    assert "itself" in merge_problem(a, a)


def test_a_result_that_would_be_too_big_is_refused_and_nothing_changes(monkeypatch):
    import chronovox.matrix_ops as ops
    a = mat("a", (2, 2, 2), (0, 0, 0), [((0, 0, 0), 1)])
    b = mat("b", (2, 2, 2), (50 * S, 0, 0), [((0, 0, 0), 2)])
    monkeypatch.setattr(ops, "MAX_MATRIX_VOXELS", 100)
    model = make_model(a, b)
    before = (a.voxels.copy(), a.position, a.pivot)
    with pytest.raises(ValueError):
        merge_matrix(model, b, a)
    assert model.matrices == [a, b] and np.array_equal(a.voxels, before[0]) and a.position == before[1]


def test_planning_changes_neither_matrix():
    a = mat("a", (2, 2, 2), (0, 0, 0), [((0, 0, 0), 1)])
    b = mat("b", (2, 2, 2), (4 * S, 0, 0), [((0, 0, 0), 2)])
    va, vb = a.voxels.copy(), b.voxels.copy()
    plan_merge(b, a)
    assert np.array_equal(a.voxels, va) and np.array_equal(b.voxels, vb) and a.size == (2, 2, 2)


def test_merge_offset_is_the_whole_voxel_difference_for_unrotated_matrices():
    a, b = mat("a", (1, 1, 1), (S, 2 * S, 3 * S)), mat("b", (1, 1, 1), (5 * S, 3 * S, 3 * S))
    assert np.allclose(merge_offset(b, a), (4, 1, 0))


# --- undo ---------------------------------------------------------------------------------------------

def test_a_merge_undoes_and_redoes_exactly_including_the_list_order():
    a = mat("a", (2, 2, 2), (4 * S, 4 * S, 4 * S), [((0, 0, 0), 1)])
    b = mat("b", (2, 2, 2), (1 * S, 3 * S, 0), [((0, 0, 0), 2)])
    c = mat("c", (1, 1, 1))
    model = make_model(a, b, c)
    history = History()
    before_a = (a.voxels.copy(), a.position, a.pivot, a.size)
    before_b = (b.voxels.copy(), b.position)
    history.push(merge_matrix(model, b, a))
    assert [m.name for m in model.matrices] == ["a", "c"] and a.size == (5, 3, 6)
    history.undo()
    assert [m.name for m in model.matrices] == ["a", "b", "c"] and model.matrices[1] is b
    assert np.array_equal(a.voxels, before_a[0]) and (a.position, a.pivot, a.size) == before_a[1:]
    assert np.array_equal(b.voxels, before_b[0]) and b.position == before_b[1]
    history.redo()
    assert [m.name for m in model.matrices] == ["a", "c"] and a.size == (5, 3, 6)


def test_voxel_edits_made_before_a_merge_still_undo_after_the_merge_is_undone():
    a = mat("a", (2, 2, 2), (0, 0, 0), [((0, 0, 0), 1)])
    b = mat("b", (2, 2, 2), (4 * S, 0, 0), [((0, 0, 0), 2)])
    model, history = make_model(a, b), History()
    before = a.voxels.copy()
    a.voxels[1, 1, 1] = 3
    history.push(VoxelEdit.between(a, before))
    history.push(merge_matrix(model, b, a))
    history.undo()                         # the merge
    history.undo()                         # the voxel edit, on a's original grid
    assert a.voxels[1, 1, 1] == 0 and a.voxels[0, 0, 0] == 1 and model.matrices == [a, b]


def test_the_history_budget_counts_the_grids_it_keeps():
    a = mat("a", (2, 2, 2), (0, 0, 0), [((0, 0, 0), 1)])
    b = mat("b", (2, 2, 2), (4 * S, 0, 0), [((0, 0, 0), 2)])
    edit = merge_matrix(make_model(a, b), b, a)
    assert edit.nbytes >= edit.old_voxels.nbytes + edit.new_voxels.nbytes + b.voxels.nbytes
