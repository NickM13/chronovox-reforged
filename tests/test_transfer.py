import numpy as np
import pytest

from chronovox import transfer
from chronovox.model import Matrix, Model

RED, GREEN, BLUE = (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)


def model(colors, *matrices):
    return Model(np.array(colors, np.uint8).reshape(-1, 4), list(matrices))


def mat(name, values, **kw):
    return Matrix(name, np.array(values, np.uint16).reshape(1, 1, -1), **kw)


def colors_of(m, matrix):
    return [None if v == 0 else tuple(int(c) for c in m.palette[v - 1]) for v in matrix.voxels.ravel()]


def test_copy_keeps_every_voxel_color_by_value_and_adds_missing_colors():
    source_matrix = mat("Arm", [1, 0, 2, 3], position=(5, 6, 7), resolution=3)
    source = model([RED, GREEN, BLUE], source_matrix)
    target = model([BLUE, RED])                                    # same colors, other order, no green
    copy, index = transfer.copy_matrix(source, source_matrix, target)
    assert index == 0 and target.matrices == [copy] and copy is not source_matrix
    assert colors_of(target, copy) == [RED, None, GREEN, BLUE]
    assert [tuple(c) for c in target.palette] == [BLUE, RED, GREEN]  # only green was added
    assert copy.position == (5, 6, 7) and copy.resolution == 3
    assert colors_of(source, source_matrix) == [RED, None, GREEN, BLUE]  # the source is untouched


def test_copy_is_independent_of_the_source_voxels():
    source_matrix = mat("Arm", [1, 1])
    source = model([RED], source_matrix)
    copy, _ = transfer.copy_matrix(source, source_matrix, model([]))
    copy.voxels[:] = 0
    assert source_matrix.voxels.all()


def test_copy_only_adds_colors_the_matrix_uses():
    source_matrix = mat("Arm", [2])
    source = model([RED, GREEN, BLUE], source_matrix)
    target = model([])
    transfer.copy_matrix(source, source_matrix, target)
    assert [tuple(c) for c in target.palette] == [GREEN]


def test_copy_renames_on_a_clash_keeping_the_game_suffix():
    source_matrix = mat("Arm-box", [1])
    source = model([RED], source_matrix)
    target = model([RED], mat("Arm-box", [1]))
    copy, index = transfer.copy_matrix(source, source_matrix, target)
    assert copy.name == "Arm2-box" and index == 1 and len(target.matrices) == 2


def test_copy_into_a_full_palette_changes_nothing(monkeypatch):
    monkeypatch.setattr(transfer, "MAX_PALETTE", 2)
    source_matrix = mat("Arm", [1, 2])
    source = model([GREEN, BLUE], source_matrix)
    target = model([RED, (1, 2, 3, 255)])
    with pytest.raises(ValueError, match="palette"):
        transfer.copy_matrix(source, source_matrix, target)
    assert len(target.palette) == 2 and not target.matrices


def test_import_colors_adds_used_missing_colors_once():
    source = model([RED, GREEN, BLUE], mat("a", [1, 3, 3]))        # green is in the palette but unused
    target = model([BLUE])
    assert transfer.import_colors(source, target) == 1
    assert [tuple(c) for c in target.palette] == [BLUE, RED]
    assert transfer.import_colors(source, target) == 0             # nothing new the second time


def test_import_colors_can_take_the_unused_ones_too():
    source = model([RED, GREEN], mat("a", [1]))
    target = model([])
    assert transfer.import_colors(source, target, used_only=False) == 2
    assert [tuple(c) for c in target.palette] == [RED, GREEN]


def test_centered_position_puts_the_middle_of_the_matrix_on_the_focus():
    from chronovox import transform
    from chronovox.matrix_tools import model_point
    from chronovox.model import POSITION_SCALE
    m = Matrix("a", np.ones((4, 6, 8), np.uint16), position=(1000, -50, 7), rotation=transform.from_euler(0, 90, 0))
    position = transfer.centered_position(m, (10, 3, -4))
    assert all((a - b) % POSITION_SCALE == 0 for a, b in zip(position, m.position))   # moved by whole units
    m.position = position
    middle = model_point(m.position, m.pivot, m.rotation, tuple(n / 2 for n in m.size), m.cell_size)
    assert np.allclose(middle, (10, 3, -4), atol=0.5)


def test_group_shift_puts_the_common_center_on_the_focus_in_whole_units():
    from chronovox.matrix_tools import group_anchor
    from chronovox.model import POSITION_SCALE
    a = Matrix("a", np.ones((2, 2, 2), np.uint16), position=(0, 0, 0))
    b = Matrix("b", np.ones((2, 2, 2), np.uint16), position=(10 * POSITION_SCALE, 0, 4 * POSITION_SCALE))
    shift = transfer.group_shift([a, b], (20, 5, 3))
    assert all(d % POSITION_SCALE == 0 for d in shift)
    for m in (a, b):
        m.position = tuple(p + d for p, d in zip(m.position, shift))
    assert np.allclose(group_anchor([a, b]), (20, 5, 3), atol=0.5)
    assert (b.position[0] - a.position[0], b.position[2] - a.position[2]) == (10 * POSITION_SCALE, 4 * POSITION_SCALE)
