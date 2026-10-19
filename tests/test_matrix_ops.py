import numpy as np
import pytest

from chronovox import transform
from chronovox.matrix_ops import plan_resize
from chronovox.model import POSITION_SCALE, Matrix, Model


def numbered(x, y, z):
    """A matrix whose every voxel has a distinct non-zero value."""
    values = np.arange(1, x * y * z + 1, dtype=np.uint16).reshape(z, y, x)
    return Matrix("m", values, position=(3 * 128, -2 * 128, 128))


def world_of_local(matrix, p, position=None, pivot=None, rotation=None):
    position = np.array(matrix.position if position is None else position, float) / POSITION_SCALE
    pivot = np.array(matrix.pivot if pivot is None else pivot, float) / POSITION_SCALE
    q = matrix.rotation if rotation is None else rotation
    return position + transform.rotate(q, np.asarray(p, float) - pivot) + pivot


def test_growing_the_high_side_keeps_everything_in_place():
    m = numbered(3, 2, 2)
    voxels, position, pivot = plan_resize(m, (0, 0, 0), (2, 1, 0))
    assert voxels.shape == (2, 3, 5)
    assert np.array_equal(voxels[:, :2, :3], m.voxels) and not voxels[:, 2:, :].any() and not voxels[:, :, 3:].any()
    assert position == m.position and pivot == m.pivot


def test_growing_the_low_side_shifts_cells_and_moves_the_origin():
    m = numbered(2, 2, 2)
    voxels, position, pivot = plan_resize(m, (1, 0, 2), (0, 0, 0))
    assert voxels.shape == (4, 2, 3)
    assert np.array_equal(voxels[2:, :, 1:], m.voxels) and not voxels[:2].any() and not voxels[:, :, 0].any()
    assert position == (m.position[0] - 128, m.position[1], m.position[2] - 2 * 128)
    assert pivot == (m.pivot[0] + 128, m.pivot[1], m.pivot[2] + 2 * 128)


def test_cropping_removes_cells_on_the_chosen_side():
    m = numbered(4, 3, 2)
    voxels, position, _ = plan_resize(m, (-1, 0, 0), (0, -2, 0))
    assert voxels.shape == (2, 1, 3)
    assert np.array_equal(voxels, m.voxels[:, :1, 1:])
    assert position[0] == m.position[0] + 128


def test_cropping_and_growing_at_once():
    m = numbered(4, 4, 1)
    voxels, _, _ = plan_resize(m, (-2, 0, 0), (3, 0, 0))
    assert voxels.shape == (1, 4, 5)
    assert np.array_equal(voxels[:, :, :2], m.voxels[:, :, 2:]) and not voxels[:, :, 2:].any()


def test_existing_voxels_stay_exactly_where_they_were_in_the_model_even_when_rotated():
    m = numbered(5, 4, 3)
    m.rotation = transform.from_euler(20, 70, -35)
    m.pivot = (2 * 128 + 37, 128, 5)   # an arbitrary off-center, fractional pivot
    low, high = (2, -1, 3), (1, 2, -1)
    voxels, position, pivot = plan_resize(m, low, high)
    for local in [(0, 0, 0), (4, 3, 2), (1, 2, 1)]:
        before = world_of_local(m, local)
        shifted = np.array(local) + np.array(low)
        after = world_of_local(m, shifted, position, pivot)
        assert np.allclose(before, after)


def test_resize_never_touches_the_original_matrix():
    m = numbered(2, 2, 2)
    snapshot = m.voxels.copy()
    plan_resize(m, (1, 1, 1), (1, 1, 1))
    assert np.array_equal(m.voxels, snapshot) and m.size == (2, 2, 2)


def test_bad_resizes_are_rejected():
    m = numbered(3, 3, 3)
    for low, high in [((-3, 0, 0), (0, 0, 0)), ((-2, 0, 0), (-1, 0, 0)), ((0, 0, 0), (0, 0, -3)),
                      ((70000, 0, 0), (0, 0, 0)), ((5000, 5000, 5000), (0, 0, 0))]:
        with pytest.raises(ValueError):
            plan_resize(m, low, high)


def test_resized_matrix_round_trips_through_the_model_like_any_other():
    model = Model(np.zeros((0, 4), np.uint8))
    m = model.new_matrix((2, 2, 2), "a")
    voxels, position, pivot = plan_resize(m, (1, 0, 0), (0, 3, 0))
    m.voxels, m.position, m.pivot = voxels, position, pivot
    assert m.size == (3, 5, 2) and m.position == (-128, 0, 0)


# --- applying edits ---------------------------------------------------------------

from chronovox.history import History  # noqa: E402
from chronovox.matrix_ops import move_pivot_in_place, resize_matrix, set_transform  # noqa: E402


def test_resize_matrix_applies_in_place_and_undoes():
    m, h = numbered(3, 3, 3), History()
    original = m.voxels.copy()
    h.push(resize_matrix(m, (1, 0, 0), (0, 2, 0)))
    assert m.size == (4, 5, 3) and m.position[0] == 2 * 128
    h.undo()
    assert m.size == (3, 3, 3) and np.array_equal(m.voxels, original) and m.position == (3 * 128, -2 * 128, 128)


def test_failed_resize_changes_nothing():
    m = numbered(3, 3, 3)
    before = (m.voxels, m.position, m.pivot)
    with pytest.raises(ValueError):
        resize_matrix(m, (-3, 0, 0), (0, 0, 0))
    assert (m.voxels, m.position, m.pivot) == before


def test_set_transform_changes_only_what_was_given_and_is_undoable():
    m, h = numbered(2, 2, 2), History()
    old = (m.position, m.pivot, m.rotation)
    edit = set_transform(m, position=(1, 2, 3), rotation=(0, 0, 0, 2))     # unnormalized on purpose
    h.push(edit)
    assert m.position == (1, 2, 3) and m.pivot == old[1] and m.rotation == (0, 0, 0, 1.0)
    h.undo()
    assert (m.position, m.pivot, m.rotation) == old


def test_set_transform_with_nothing_new_returns_none():
    m = numbered(2, 2, 2)
    assert set_transform(m) is None
    assert set_transform(m, position=m.position, rotation=m.rotation) is None


def test_set_transform_rejects_values_outside_int32():
    m = numbered(2, 2, 2)
    with pytest.raises(ValueError):
        set_transform(m, position=(2 ** 31, 0, 0))
    assert m.position == (3 * 128, -2 * 128, 128)


def test_move_pivot_in_place_keeps_an_unrotated_matrix_exactly_where_it_was():
    m = numbered(4, 4, 4)
    h = History()
    before = world_of_local(m, (1.5, 2, 3))
    h.push(move_pivot_in_place(m, (10, 20, 30)))
    assert m.pivot == (10, 20, 30) and np.allclose(world_of_local(m, (1.5, 2, 3)), before)
    h.undo()
    assert m.pivot == (2 * 128,) * 3 and m.position == (3 * 128, -2 * 128, 128)


def test_move_pivot_in_place_keeps_a_rotated_matrix_within_half_a_step():
    m = numbered(4, 4, 4)
    m.rotation = transform.from_euler(25, 80, -40)
    for p in [(0, 0, 0), (3.5, 1, 2)]:
        before = world_of_local(m, p)
        edit = move_pivot_in_place(m, (300, 17, -90))
        assert np.allclose(world_of_local(m, p), before, atol=1.0 / 128)
        edit.undo()


def test_move_pivot_in_place_to_the_same_pivot_is_nothing():
    m = numbered(2, 2, 2)
    assert move_pivot_in_place(m, m.pivot) is None
