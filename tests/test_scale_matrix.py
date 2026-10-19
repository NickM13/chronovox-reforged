import numpy as np
import pytest

from chronovox import transform
from chronovox.history import History
from chronovox.matrix_ops import scale_double, scale_half
from chronovox.matrix_tools import model_point
from chronovox.model import POSITION_SCALE, Matrix

S = POSITION_SCALE


def mat(size=(3, 2, 4), position=(0, 0, 0), rotation=transform.IDENTITY, resolution=1, seed=1):
    """A matrix full of distinct values, so any mix-up shows."""
    values = np.arange(1, size[0] * size[1] * size[2] + 1, dtype=np.uint16).reshape(size[2], size[1], size[0])
    return Matrix("m", values, position, rotation=rotation, resolution=resolution)


# --- double ------------------------------------------------------------------------------------------------

def test_double_makes_every_cell_a_two_by_two_by_two_block_of_the_same_color():
    m = mat((3, 2, 4))
    before = m.voxels.copy()
    scale_double(m)
    assert m.size == (6, 4, 8) and m.voxels.shape == (8, 4, 6)
    for z in range(4):
        for y in range(2):
            for x in range(3):
                assert (m.voxels[2 * z:2 * z + 2, 2 * y:2 * y + 2, 2 * x:2 * x + 2] == before[z, y, x]).all()


def test_double_multiplies_the_voxel_count_by_eight_and_keeps_empty_cells_empty():
    m = mat((2, 2, 2))
    m.voxels[0, 0, 0] = 0
    solid = int((m.voxels > 0).sum())
    scale_double(m)
    assert int((m.voxels > 0).sum()) == solid * 8 and not m.voxels[:2, :2, :2].any()


def test_double_scales_about_the_world_origin_doubling_position_and_pivot():
    m = mat((4, 2, 6), position=(5 * S, -3 * S, S // 2), rotation=transform.from_euler(10, 20, 30))
    assert m.pivot == m.center_pivot()
    scale_double(m)
    assert m.position == (10 * S, -6 * S, S) and m.rotation == transform.from_euler(10, 20, 30)
    assert m.pivot == m.center_pivot()                                  # still the middle of the (bigger) matrix


def test_double_doubles_every_point_of_the_matrix_away_from_the_origin():
    m = mat((4, 4, 4), position=(S, 2 * S, 3 * S), resolution=3, rotation=transform.from_euler(0, 40, 0))
    m.pivot = (S, 0, 3 * S)                                             # an off-center pivot
    spots = [(0, 0, 0), (4, 4, 4), (1, 2, 3)]
    before = [model_point(m.position, m.pivot, m.rotation, c, m.cell_size) for c in spots]
    scale_double(m)
    assert m.resolution == 3
    after = [model_point(m.position, m.pivot, m.rotation, (2 * c[0], 2 * c[1], 2 * c[2]), m.cell_size) for c in spots]
    assert np.allclose(after, 2 * np.array(before), atol=1 / S)


# --- half --------------------------------------------------------------------------------------------------

def test_half_keeps_the_corner_voxel_of_each_two_by_two_by_two_block():
    m = mat((4, 4, 4))
    before = m.voxels.copy()
    scale_half(m)
    assert m.size == (2, 2, 2)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                assert m.voxels[z, y, x] == before[2 * z, 2 * y, 2 * x]


def test_half_of_an_odd_side_keeps_its_last_lone_cell():
    m = mat((5, 3, 1))
    before = m.voxels.copy()
    scale_half(m)
    assert m.size == (3, 2, 1)                                          # (5 + 1) // 2, (3 + 1) // 2, (1 + 1) // 2
    assert np.array_equal(m.voxels, before[::2, ::2, ::2])
    assert m.voxels[0, 0, 2] == before[0, 0, 4]                         # the lone fifth cell survives


def test_half_ignores_the_other_cells_of_each_block():
    m = Matrix("m", np.zeros((2, 2, 2), np.uint16))
    m.voxels[1, 1, 1] = 5                                               # not the corner: it is dropped
    scale_half(m)
    assert m.size == (1, 1, 1) and m.voxels[0, 0, 0] == 0


def test_half_scales_about_the_world_origin_halving_position_and_pivot():
    m = mat((4, 6, 2), position=(5 * S, 0, -4 * S), rotation=transform.from_euler(15, 25, 35))
    scale_half(m)
    assert m.size == (2, 3, 1) and m.position == (int(2.5 * S), 0, -2 * S)
    assert m.pivot == m.center_pivot()                                  # the center of the old matrix is the new center


def test_half_rounds_an_odd_pivot_to_the_nearest_step():
    m = mat((4, 4, 4))
    m.pivot = (37, 38, 39)
    scale_half(m)
    assert m.pivot == (19, 19, 20)                                      # 18.5, 19, 19.5 rounded half up


def test_half_of_a_one_cell_matrix_is_refused_and_changes_nothing():
    m = Matrix("m", np.ones((1, 1, 1), np.uint16))
    pivot = m.pivot
    with pytest.raises(ValueError, match="as small"):
        scale_half(m)
    assert m.size == (1, 1, 1) and m.pivot == pivot


# --- both -----------------------------------------------------------------------------------------------------

def test_halving_a_doubled_matrix_gives_back_the_original_voxels_and_pivot():
    m = mat((3, 5, 2))
    voxels, pivot = m.voxels.copy(), m.pivot
    scale_double(m)
    scale_half(m)
    assert np.array_equal(m.voxels, voxels) and m.pivot == pivot


def test_doubling_a_halved_matrix_does_not_bring_back_what_was_dropped():
    m = mat((4, 4, 4))
    original = m.voxels.copy()
    scale_half(m)
    scale_double(m)
    assert m.size == (4, 4, 4) and not np.array_equal(m.voxels, original)


def test_double_and_half_undo_and_redo_exactly():
    m = mat((3, 3, 3), position=(S, S, S))
    start = (m.voxels.copy(), m.pivot, m.size)
    history = History()
    history.push(scale_double(m))
    history.push(scale_half(m))
    history.push(scale_half(m))
    for _ in range(3):
        history.undo()
    assert np.array_equal(m.voxels, start[0]) and (m.pivot, m.size) == start[1:] and m.position == (S, S, S)
    for _ in range(3):
        history.redo()
    assert m.size == (2, 2, 2)


def test_double_refuses_a_result_that_would_be_too_big_and_changes_nothing(monkeypatch):
    import chronovox.matrix_ops as ops
    m = mat((4, 4, 4))
    before = m.voxels.copy()
    monkeypatch.setattr(ops, "MAX_MATRIX_VOXELS", 100)                  # 4*4*4*8 = 512 > 100
    with pytest.raises(ValueError, match="voxels"):
        scale_double(m)
    assert np.array_equal(m.voxels, before) and m.size == (4, 4, 4)
    monkeypatch.setattr(ops, "MAX_MATRIX_SIZE", 6)
    monkeypatch.setattr(ops, "MAX_MATRIX_VOXELS", 10 ** 9)
    with pytest.raises(ValueError, match="cells on an axis"):
        scale_double(m)


def test_a_doubled_matrix_saves_and_loads():
    from chronovox import cvox
    from chronovox.model import Model
    m = mat((3, 2, 2))
    scale_double(m)
    model = Model(np.zeros((12, 4), np.uint8) + 255, [m])
    back = cvox.loads(cvox.dumps(model))
    assert back.matrices[0].size == (6, 4, 4) and np.array_equal(back.matrices[0].voxels, m.voxels)
