import numpy as np
import pytest

import chronovox.matrix_ops as ops
from chronovox import transform
from chronovox.history import GroupEdit
from chronovox.matrix_ops import refine
from chronovox.matrix_tools import model_point
from chronovox.model import POSITION_SCALE, Matrix
from chronovox.preferences import Preferences

S = POSITION_SCALE


def mat(size=(4, 2, 6), position=(5 * S, 3 * S, -2 * S), resolution=2, rotation=transform.IDENTITY):
    values = np.arange(1, size[0] * size[1] * size[2] + 1, dtype=np.uint16).reshape(size[2], size[1], size[0])
    m = Matrix("m", values, position, resolution=resolution)
    m.rotation = rotation
    return m


def box_corners(m):
    """Where the eight corners of the matrix are in the model."""
    return np.array([model_point(m.position, m.pivot, m.rotation, (x * m.size[0], y * m.size[1], z * m.size[2]), m.cell_size)
                     for x in (0, 1) for y in (0, 1) for z in (0, 1)])


def test_finer_detail_keeps_the_size_and_the_place_and_doubles_the_cells():
    m = mat(rotation=transform.from_euler(10, 20, 30))
    before_corners, before_voxels = box_corners(m), m.voxels.copy()
    refine(m, 1)
    assert m.size == (8, 4, 12) and m.resolution == 3
    assert np.allclose(box_corners(m), before_corners, atol=1 / S)              # the same box in the model
    assert (m.voxels[::2, ::2, ::2] == before_voxels).all()                      # every old cell is now a 2 x 2 x 2 block
    assert (m.voxels[1::2, 1::2, 1::2] == before_voxels).all()


def test_coarser_detail_keeps_the_size_and_halves_the_cells():
    m = mat(resolution=3)
    before_corners = box_corners(m)
    refine(m, -1)
    assert m.size == (2, 1, 3) and m.resolution == 2
    assert np.allclose(box_corners(m), before_corners, atol=1 / S)


def test_several_steps_go_all_the_way_in_one_undo_step():
    m = mat(resolution=1)
    before = (m.voxels.copy(), m.position, m.pivot, m.resolution)
    edit = refine(m, 2)
    assert isinstance(edit, GroupEdit) and m.size == (16, 8, 24) and m.resolution == 3
    edit.undo()
    assert np.array_equal(m.voxels, before[0]) and (m.position, m.pivot, m.resolution) == before[1:]
    edit.redo()
    assert m.size == (16, 8, 24) and m.resolution == 3


def test_finer_then_coarser_gives_back_the_matrix():
    m = mat(position=(6 * S, -4 * S, 2 * S))                                      # even positions survive halving exactly
    voxels, position, pivot = m.voxels.copy(), m.position, m.pivot
    refine(m, 1)
    refine(m, -1)
    assert np.array_equal(m.voxels, voxels) and (m.position, m.pivot, m.resolution) == (position, pivot, 2)


def test_no_steps_is_nothing_and_leaving_the_resolution_range_is_refused_without_a_change():
    m = mat(resolution=8)
    assert refine(m, 0) is None
    before = m.voxels.copy()
    with pytest.raises(ValueError, match="between 1 and 8"):
        refine(m, 1)
    low = mat(resolution=1)
    with pytest.raises(ValueError):
        refine(low, -1)
    with pytest.raises(ValueError):
        refine(mat(resolution=7), 2)
    assert np.array_equal(m.voxels, before) and m.resolution == 8 and low.resolution == 1


def test_if_the_matrix_would_get_too_big_part_way_everything_is_put_back(monkeypatch):
    m = mat(size=(4, 4, 4), resolution=1)
    before = (m.voxels.copy(), m.position, m.pivot, m.resolution)
    monkeypatch.setattr(ops, "MAX_MATRIX_VOXELS", 4 * 4 * 4 * 8 + 10)           # one step fits (512 cells), two do not
    with pytest.raises(ValueError, match="voxels"):
        refine(m, 2)
    assert np.array_equal(m.voxels, before[0]) and (m.position, m.pivot, m.resolution) == before[1:]


def test_an_old_preferences_file_that_still_has_the_removed_switch_loads_fine():
    prefs = Preferences.from_dict({"couple_scale": True, "bulk_resolution": 3})
    assert prefs.bulk_resolution == 3 and not hasattr(prefs, "couple_scale")
