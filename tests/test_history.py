import numpy as np

from chronovox.history import History, VoxelEdit
from chronovox.model import Matrix


def make(shape=(4, 4, 4)):
    return Matrix("m", np.zeros(shape, np.uint16))


def edit(history, matrix, changes):
    """Apply {(x, y, z): value} to the matrix and record it, like a stroke would."""
    before = matrix.voxels.copy()
    for (x, y, z), value in changes.items():
        matrix.voxels[z, y, x] = value
    recorded = VoxelEdit.between(matrix, before)
    if recorded:
        history.push(recorded)
    return recorded


def test_between_finds_only_the_changed_voxels():
    m = make()
    before = m.voxels.copy()
    m.voxels[1, 2, 3] = 5
    m.voxels[0, 0, 0] = 7
    e = VoxelEdit.between(m, before)
    assert len(e.indices) == 2 and sorted(e.new.tolist()) == [5, 7] and e.old.tolist() == [0, 0]
    assert VoxelEdit.between(m, m.voxels.copy()) is None


def test_undo_and_redo_restore_the_exact_voxels():
    m, h = make(), History()
    edit(h, m, {(0, 0, 0): 1, (3, 3, 3): 2})
    after = m.voxels.copy()
    assert h.undo() is not None and not m.voxels.any()
    assert h.redo() is not None and np.array_equal(m.voxels, after)


def test_undo_restores_overwritten_values_not_just_zeros():
    m, h = make(), History()
    edit(h, m, {(1, 1, 1): 3})
    edit(h, m, {(1, 1, 1): 9, (2, 2, 2): 4})   # recolor one voxel, add another
    h.undo()
    assert m.voxels[1, 1, 1] == 3 and m.voxels[2, 2, 2] == 0
    h.undo()
    assert not m.voxels.any()


def test_several_edits_undo_in_reverse_order_and_redo_forward():
    m, h = make(), History()
    for i in range(1, 4):
        edit(h, m, {(i, 0, 0): i})
    states = []
    while h.can_undo:
        h.undo()
        states.append(int((m.voxels > 0).sum()))
    assert states == [2, 1, 0]
    while h.can_redo:
        h.redo()
    assert m.voxels[0, 0, 1:4].tolist() == [1, 2, 3]


def test_nothing_to_undo_or_redo_returns_none():
    h = History()
    assert h.undo() is None and h.redo() is None
    assert not h.can_undo and not h.can_redo


def test_a_new_edit_after_undo_discards_the_redo_stack():
    m, h = make(), History()
    edit(h, m, {(0, 0, 0): 1})
    h.undo()
    assert h.can_redo
    edit(h, m, {(1, 1, 1): 2})
    assert not h.can_redo and h.redo() is None


def test_edits_on_different_matrices_stay_separate():
    a, b, h = make(), make(), History()
    edit(h, a, {(0, 0, 0): 1})
    edit(h, b, {(0, 0, 0): 2})
    assert h.undo().matrix is b
    assert a.voxels[0, 0, 0] == 1 and b.voxels[0, 0, 0] == 0


def test_works_on_non_cubic_matrices():
    m, h = make((5, 3, 2)), History()  # (x=5, y=3, z=2) as (z, y, x) shape (2, 3, 5)
    m = Matrix("m", np.zeros((2, 3, 5), np.uint16))
    edit(h, m, {(4, 2, 1): 8})
    h.undo()
    assert not m.voxels.any()
    h.redo()
    assert m.voxels[1, 2, 4] == 8


def test_the_memory_budget_drops_the_oldest_edits_but_keeps_the_latest():
    m = make((10, 10, 10))
    h = History(max_bytes=200)
    for i in range(30):
        edit(h, m, {(i % 10, i // 10, 0): i + 1})
    assert 1 <= len(h._undo) < 30
    assert h.undo() is not None
    assert m.voxels[0, 2, 9] == 0  # the newest edit really was the one undone


def test_clear_forgets_everything():
    m, h = make(), History()
    edit(h, m, {(0, 0, 0): 1})
    h.undo()
    h.clear()
    assert not h.can_undo and not h.can_redo


# --- matrix edits ---------------------------------------------------------------

from chronovox.history import ResizeEdit, TransformEdit  # noqa: E402
from chronovox.matrix_ops import plan_resize  # noqa: E402


def test_transform_edit_restores_position_pivot_and_rotation():
    m, h = make(), History()
    old = (m.position, m.pivot, m.rotation)
    m.position, m.pivot, m.rotation = (128, 0, -64), (1, 2, 3), (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)
    h.push(TransformEdit(m, old, (m.position, m.pivot, m.rotation)))
    h.undo()
    assert (m.position, m.pivot, m.rotation) == old
    h.redo()
    assert m.position == (128, 0, -64) and m.rotation[1] > 0.7


def test_resize_edit_undo_restores_a_cropped_grid_exactly():
    m, h = make(), History()
    edit(h, m, {(0, 0, 0): 1, (3, 3, 3): 2})
    original = m.voxels.copy()
    voxels, position, pivot = plan_resize(m, (-1, 0, 0), (0, -1, 0))   # crops away x=0 and y=3
    h.push(ResizeEdit(m, m.voxels, voxels, (m.position, m.pivot), (position, pivot)))
    m.voxels, m.position, m.pivot = voxels, position, pivot
    assert m.size == (3, 3, 4)
    h.undo()
    assert np.array_equal(m.voxels, original) and m.size == (4, 4, 4)
    h.redo()
    assert m.size == (3, 3, 4) and m.position == position


def test_voxel_edits_before_and_after_a_resize_undo_in_order():
    m, h = make(), History()
    edit(h, m, {(1, 1, 1): 5})                                    # old shape (4, 4, 4)
    voxels, position, pivot = plan_resize(m, (2, 0, 0), (0, 0, 0))
    h.push(ResizeEdit(m, m.voxels, voxels, (m.position, m.pivot), (position, pivot)))
    m.voxels, m.position, m.pivot = voxels, position, pivot
    edit(h, m, {(5, 0, 0): 9})                                    # new shape (4, 4, 6)
    h.undo()                       # the voxel edit on the new grid
    assert m.size == (6, 4, 4) and m.voxels[0, 0, 5] == 0 and m.voxels[1, 1, 3] == 5
    h.undo()                       # the resize
    assert m.size == (4, 4, 4) and m.voxels[1, 1, 1] == 5
    h.undo()                       # the first voxel edit, against the original grid
    assert not m.voxels.any()
    for _ in range(3):
        h.redo()
    assert m.size == (6, 4, 4) and m.voxels[0, 0, 5] == 9 and m.voxels[1, 1, 3] == 5


def test_every_edit_kind_can_describe_itself():
    m = make()
    assert "voxel change" in VoxelEdit(m, np.array([0], np.uint32), np.array([0], np.uint16), np.array([1], np.uint16)).describe()
    assert "transform" in TransformEdit(m, (), ()).describe()
    assert "size" in ResizeEdit(m, m.voxels, m.voxels, (), ()).describe()
