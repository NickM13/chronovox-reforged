import numpy as np
import pytest

from chronovox import transform
from chronovox.history import History, ReorientEdit
from chronovox.matrix_ops import flip_matrix, turn_matrix
from chronovox.matrix_tools import model_point
from chronovox.model import POSITION_SCALE, Matrix

S = POSITION_SCALE


def mat(size=(3, 2, 5), position=(7 * S, -3 * S, 5 * S // 2), pivot=None, rotation=transform.from_euler(12, 33, -20),
        resolution=3):
    rng = np.random.default_rng(5)
    voxels = rng.integers(1, 200, (size[2], size[1], size[0])).astype(np.uint16)
    m = Matrix("m", voxels, position, resolution=resolution)
    m.rotation = rotation
    if pivot is not None:
        m.pivot = pivot
    return m


def quarter(vector, axis):
    """One right-handed quarter turn of a vector (dx, dy, dz): about x +y -> +z, about y +z -> +x, about z +x -> +y."""
    dx, dy, dz = vector
    return {0: (dx, -dz, dy), 1: (dz, dy, -dx), 2: (-dy, dx, dz)}[axis]


def cells(m):
    zs, ys, xs = np.indices(m.voxels.shape)
    return zip(xs.ravel().tolist(), ys.ravel().tolist(), zs.ravel().tolist())


def expected_cell(cell, old_size, new_size, how):
    """Where a cell goes: its center is moved about the matrix's middle by ``how`` (a function on offsets)."""
    old_c = np.array(old_size) / 2
    new_c = np.array(new_size) / 2
    centre = np.array(cell) + 0.5
    moved = new_c + np.array(how(tuple(centre - old_c)))
    return tuple(int(np.floor(v)) for v in moved)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("turns", [1, 2, 3])
def test_a_turn_moves_every_voxel_as_the_right_handed_rule_says_and_swaps_the_sizes(axis, turns):
    m = mat()
    before, size = m.voxels.copy(), m.size

    def how(offset):
        for _ in range(turns):
            offset = quarter(offset, axis)
        return offset

    edit = turn_matrix(m, axis, turns)
    assert isinstance(edit, ReorientEdit)
    swapped = list(size)
    if turns % 2:
        u, v = [(1, 2), (2, 0), (0, 1)][axis]
        swapped[u], swapped[v] = size[v], size[u]
    assert m.size == tuple(swapped)
    for (x, y, z) in cells(Matrix("old", before)):
        nx, ny, nz = expected_cell((x, y, z), size, m.size, how)
        assert m.voxels[nz, ny, nx] == before[z, y, x]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_flip_mirrors_across_the_middle_and_keeps_the_size(axis):
    m = mat()
    before, size = m.voxels.copy(), m.size

    def how(offset):
        out = list(offset)
        out[axis] = -out[axis]
        return tuple(out)

    flip_matrix(m, axis)
    assert m.size == size
    for (x, y, z) in cells(Matrix("old", before)):
        nx, ny, nz = expected_cell((x, y, z), size, size, how)
        assert m.voxels[nz, ny, nx] == before[z, y, x]


@pytest.mark.parametrize("op", ["turn0", "turn1", "turn2", "flip0", "flip1", "flip2"])
def test_the_matrix_keeps_its_rotation_and_its_center_in_the_model(op):
    m = mat(pivot=(100, 40, 300))                                                  # an off-center pivot, and a rotation
    rotation = m.rotation
    before = model_point(m.position, m.pivot, m.rotation, np.array(m.size) / 2, m.cell_size)
    (turn_matrix(m, int(op[-1])) if op.startswith("turn") else flip_matrix(m, int(op[-1])))
    after = model_point(m.position, m.pivot, m.rotation, np.array(m.size) / 2, m.cell_size)
    assert m.rotation == rotation                                                   # the rotation value is not touched
    assert np.allclose(before, after, atol=2 / S)                                   # the middle of the matrix has not moved


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_voxel_ends_up_where_turning_the_matrix_about_its_center_puts_it(axis):
    m = mat()
    size, before = m.size, m.voxels.copy()
    old = Matrix("old", before, m.position, m.pivot, m.rotation, m.resolution)
    centre_world = model_point(old.position, old.pivot, old.rotation, np.array(size) / 2, old.cell_size)
    r = transform.to_matrix(m.rotation)
    turn_matrix(m, axis)
    for (x, y, z) in list(cells(old))[::7]:
        offset = quarter(tuple(np.array((x, y, z)) + 0.5 - np.array(size) / 2), axis)
        expected = centre_world + r @ (old.cell_size * np.array(offset))
        nx, ny, nz = expected_cell((x, y, z), size, m.size, lambda o: quarter(o, axis))
        actual = model_point(m.position, m.pivot, m.rotation, (nx + 0.5, ny + 0.5, nz + 0.5), m.cell_size)
        assert np.allclose(actual, expected, atol=2 / S)


def test_a_pivot_in_the_middle_stays_in_the_middle_and_an_unrotated_matrix_keeps_its_center():
    m = Matrix("m", np.ones((6, 2, 4), np.uint16), position=(5 * S, 0, 3 * S))        # 4 x 2 x 6, resolution 1, centered pivot
    assert m.pivot == m.center_pivot()
    turn_matrix(m, 1)                                                                  # about y: 6 x 2 x 4
    assert m.size == (6, 2, 4) and m.pivot == m.center_pivot()
    assert m.position == (4 * S, 0, 4 * S)                                             # x: 4 -> 6 so one unit left; z: 6 -> 4 so one unit in


def test_an_off_center_pivot_turns_and_flips_with_the_voxels():
    m = Matrix("m", np.ones((2, 2, 4), np.uint16), pivot=(0, 0, 0))                    # the pivot is at the corner of the content
    flip_matrix(m, 0)
    assert m.pivot == (4 * S, 0, 0)                                                    # the corner it was on is now at +x
    flip_matrix(m, 0)
    assert m.pivot == (0, 0, 0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_four_quarter_turns_and_two_flips_give_back_the_matrix(axis):
    m = mat(position=(8 * S, 4 * S, -2 * S), pivot=(64, 128, 192))
    voxels, position, pivot, size = m.voxels.copy(), m.position, m.pivot, m.size
    for _ in range(4):
        turn_matrix(m, axis)
    assert np.array_equal(m.voxels, voxels) and m.size == size and m.pivot == pivot
    assert all(abs(a - b) <= 1 for a, b in zip(m.position, position))
    flip_matrix(m, axis)
    flip_matrix(m, axis)
    assert np.array_equal(m.voxels, voxels) and m.pivot == pivot


def test_turning_the_other_way_undoes_a_turn_and_whole_turns_do_nothing():
    m = mat(size=(3, 2, 5))
    voxels = m.voxels.copy()
    assert turn_matrix(m, 1, 4) is None and turn_matrix(m, 1, 0) is None
    turn_matrix(m, 1, 1)
    turn_matrix(m, 1, -1)
    assert np.array_equal(m.voxels, voxels)


def test_a_flip_that_changes_nothing_makes_no_entry():
    m = Matrix("m", np.full((2, 2, 4), 3, np.uint16))                                  # the same everywhere, pivot in the middle
    assert flip_matrix(m, 0) is None


def test_undo_and_redo_restore_the_grid_the_position_and_the_pivot():
    m = mat(pivot=(100, 40, 300))
    before = (m.voxels.copy(), m.position, m.pivot, m.size)
    history = History()
    history.push(turn_matrix(m, 2))
    after = (m.voxels.copy(), m.position, m.pivot, m.size)
    assert after[3] != before[3]
    history.undo()
    assert np.array_equal(m.voxels, before[0]) and (m.position, m.pivot, m.size) == before[1:]
    history.redo()
    assert np.array_equal(m.voxels, after[0]) and (m.position, m.pivot, m.size) == after[1:]
    assert "turn about Z of m" == history._undo[-1].describe()
