import numpy as np
import pytest

from chronovox import cvox
from chronovox.model import MAX_MATRIX_VOXELS, Matrix, Model


def test_new_matrix_is_empty_with_the_right_shape_and_a_centered_pivot():
    model = Model()
    m = model.new_matrix((4, 5, 6), "box", (128, 0, -256))
    assert model.matrices == [m]
    assert m.voxels.shape == (6, 5, 4) and not m.voxels.any()
    assert m.size == (4, 5, 6)
    assert tuple(m.position) == (128, 0, -256)
    assert m.pivot == (4 * 64, 5 * 64, 6 * 64)


def test_names_are_made_unique_when_not_given():
    model = Model()
    assert [model.new_matrix((1, 1, 1)).name for _ in range(3)] == ["Matrix", "Matrix2", "Matrix3"]
    assert model.new_matrix((1, 1, 1), "  ").name == "Matrix4"


def test_duplicate_and_bad_names_are_rejected_without_adding_anything():
    model = Model()
    model.new_matrix((1, 1, 1), "a")
    for bad in ("a", "caf\u00e9", "x" * 256):
        with pytest.raises(ValueError):
            model.new_matrix((1, 1, 1), bad)
    assert len(model.matrices) == 1


def test_bad_sizes_are_rejected():
    model = Model()
    for size in ((0, 1, 1), (1, -3, 1), (65536, 1, 1)):
        with pytest.raises(ValueError):
            model.new_matrix(size)
    side = int(round(MAX_MATRIX_VOXELS ** (1 / 3))) + 1
    with pytest.raises(ValueError):
        model.new_matrix((side, side, side))
    assert model.matrices == []


def test_a_model_with_added_matrices_round_trips_through_cvox():
    model = Model(np.array([[1, 2, 3, 255]], np.uint8))
    first = model.new_matrix((3, 3, 3), "first")
    first.voxels[1, 1, 1] = 1
    model.new_matrix((2, 2, 2), "second", (128 * 5, 0, 0))
    back = cvox.loads(cvox.dumps(model))
    assert [m.name for m in back.matrices] == ["first", "second"]
    assert back.matrices[0].voxels[1, 1, 1] == 1
    assert tuple(back.matrices[1].position) == (640, 0, 0)


def test_automatic_names_never_contain_spaces_even_from_a_custom_base():
    model = Model()
    model.new_matrix((1, 1, 1), "LeftArm")
    assert model.unique_matrix_name("LeftArm") == "LeftArm2"
    assert " " not in model.unique_matrix_name()


def test_rename_changes_the_name_and_keeps_everything_else():
    model = Model()
    m = model.new_matrix((2, 2, 2), "Matrix")
    m.voxels[0, 0, 0] = 1
    assert model.rename_matrix(m, "  LeftArm-box ")
    assert m.name == "LeftArm-box" and m.voxels[0, 0, 0] == 1


def test_rename_to_the_same_name_is_not_a_change():
    model = Model()
    m = model.new_matrix((1, 1, 1), "Head")
    assert not model.rename_matrix(m, "Head")
    assert not model.rename_matrix(m, " Head ")


def test_rename_rejects_bad_names_and_leaves_the_old_one():
    model = Model()
    a = model.new_matrix((1, 1, 1), "A")
    model.new_matrix((1, 1, 1), "B")
    for bad in ("", "   ", "B", "caf\u00e9", "x" * 256):
        with pytest.raises(ValueError):
            model.rename_matrix(a, bad)
        assert a.name == "A"


def test_names_with_game_suffixes_are_allowed_and_survive_a_round_trip():
    model = Model()
    for name in ("Shield+", "Wing-", "Torso-box", "LeftArm"):
        model.new_matrix((1, 1, 1), name)
    assert [m.name for m in cvox.loads(cvox.dumps(model)).matrices] == \
        ["Shield+", "Wing-", "Torso-box", "LeftArm"]


# --- unused palette colors ----------------------------------------------------------------------

def painted_model():
    model = Model(np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [9, 9, 9, 255]], np.uint8))
    a = model.new_matrix((2, 2, 1), "a")
    b = model.new_matrix((2, 1, 1), "b")
    a.voxels[0, :, :] = [[1, 3], [0, 1]]       # uses colors 1 and 3
    b.voxels[0, 0, :] = [3, 0]                 # color 3 again; colors 2 and 4 are never used
    return model


def colors_of(model):
    """The color of every voxel of every matrix, as RGBA tuples (None for empty)."""
    return [[tuple(model.palette[i - 1]) if i else None for i in m.voxels.ravel()] for m in model.matrices]


def test_used_palette_mask_counts_every_matrix():
    model = painted_model()
    assert model.used_palette_mask().tolist() == [True, False, True, False]


def test_a_model_with_no_voxels_uses_no_colors_and_no_colors_is_fine():
    model = painted_model()
    for m in model.matrices:
        m.voxels[:] = 0
    assert not model.used_palette_mask().any()
    assert Model().used_palette_mask().tolist() == []


def test_compacting_drops_unused_colors_but_every_voxel_keeps_its_color():
    model = painted_model()
    before = colors_of(model)
    packed = model.compacted()
    assert [tuple(c) for c in packed.palette] == [(255, 0, 0, 255), (0, 0, 255, 255)]    # order kept
    assert colors_of(packed) == before
    assert [m.name for m in packed.matrices] == ["a", "b"] and packed is not model


def test_compacting_leaves_the_original_model_alone():
    model = painted_model()
    palette, voxels = model.palette.copy(), [m.voxels.copy() for m in model.matrices]
    model.compacted()
    assert np.array_equal(model.palette, palette)
    assert all(np.array_equal(m.voxels, v) for m, v in zip(model.matrices, voxels))


def test_compacting_a_model_that_uses_every_color_changes_nothing():
    model = painted_model()
    model.matrices[1].voxels[0, 0, 1] = 2
    model.matrices[1].voxels[0, 0, 0] = 4
    assert model.compacted() is model


def test_compacting_keeps_position_pivot_and_rotation():
    model = painted_model()
    a = model.matrices[0]
    a.position, a.pivot, a.rotation = (5, 6, 7), (1, 2, 3), (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)
    packed = model.compacted().matrices[0]
    assert (packed.position, packed.pivot, packed.rotation) == (a.position, a.pivot, a.rotation)


def test_everything_unused_compacts_to_an_empty_palette():
    model = painted_model()
    for m in model.matrices:
        m.voxels[:] = 0
    packed = model.compacted()
    assert len(packed.palette) == 0 and all(not m.voxels.any() for m in packed.matrices)
    assert cvox.loads(cvox.dumps(packed)).palette.shape == (0, 4)


def test_a_saved_file_has_no_unused_colors_and_loads_back_with_the_same_colors():
    model = painted_model()
    back = cvox.loads(cvox.dumps(model.compacted()))
    assert len(back.palette) == 2
    assert colors_of(back) == colors_of(model)


def test_compacting_can_bring_a_big_palette_back_under_the_one_byte_index_limit():
    palette = np.zeros((300, 4), np.uint8)
    palette[:, 0] = np.arange(300) % 256
    palette[:, 1] = np.arange(300) // 256
    palette[:, 3] = 255
    model = Model(palette)
    m = model.new_matrix((4, 1, 1), "m")
    m.voxels[0, 0, :] = [1, 150, 299, 300]
    wide = cvox.dumps(model)
    packed = model.compacted()
    assert len(packed.palette) == 4 and len(cvox.dumps(packed)) < len(wide)
    assert colors_of(cvox.loads(cvox.dumps(packed))) == colors_of(model)


# --- opaque palettes ----------------------------------------------------------------------------

def test_opaque_resets_alpha_and_keeps_every_voxel_color():
    palette = np.array([[10, 20, 30, 1], [40, 50, 60, 128], [70, 80, 90, 255]], np.uint8)
    model = Model(palette)
    m = model.new_matrix((3, 1, 1), "m")
    m.voxels[0, 0, :] = [3, 0, 1]
    clean = model.opaque()
    assert (clean.palette[:, 3] == 255).all() and clean.palette[:, :3].tolist() == palette[:, :3].tolist()
    assert clean.matrices[0].voxels[0, 0].tolist() == [3, 0, 1]


def test_opaque_merges_colors_that_only_differed_in_alpha_and_remaps_voxels():
    palette = np.array([[10, 20, 30, 1], [5, 5, 5, 1], [10, 20, 30, 255], [5, 5, 5, 7]], np.uint8)
    model = Model(palette)
    a = model.new_matrix((4, 1, 1), "a")
    a.voxels[0, 0, :] = [1, 2, 3, 4]
    clean = model.opaque()
    assert clean.palette.tolist() == [[10, 20, 30, 255], [5, 5, 5, 255]]          # first appearance order
    assert clean.matrices[0].voxels[0, 0].tolist() == [1, 2, 1, 2]                 # duplicates share an index


def test_opaque_returns_the_same_model_when_nothing_needs_changing():
    model = Model(np.array([[1, 2, 3, 255]], np.uint8))
    assert model.opaque() is model and Model().opaque().palette.shape == (0, 4)


def test_opaque_leaves_the_original_alone_and_keeps_transforms():
    palette = np.array([[1, 2, 3, 1]], np.uint8)
    model = Model(palette.copy())
    m = model.new_matrix((1, 1, 1), "m", (5, 6, 7))
    m.voxels[0, 0, 0] = 1
    m.rotation = (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)
    clean = model.opaque()
    assert model.palette.tolist() == palette.tolist() and clean is not model
    assert (clean.matrices[0].position, clean.matrices[0].rotation) == (m.position, m.rotation)


# --- reordering matrices -------------------------------------------------------------------------

def names(model):
    return [m.name for m in model.matrices]


def test_move_matrix_puts_it_at_the_new_place_and_reports_where_it_was():
    model = Model()
    ms = [model.new_matrix((1, 1, 1), n) for n in "abcd"]
    assert model.move_matrix(ms[0], 2) == 0 and names(model) == ["b", "c", "a", "d"]
    assert model.move_matrix(ms[3], 0) == 3 and names(model) == ["d", "b", "c", "a"]


def test_move_matrix_clamps_to_the_ends_and_keeps_every_matrix():
    model = Model()
    ms = [model.new_matrix((1, 1, 1), n) for n in "abc"]
    model.move_matrix(ms[1], 99)
    assert names(model) == ["a", "c", "b"]
    model.move_matrix(ms[1], -5)
    assert names(model) == ["b", "a", "c"] and len(model.matrices) == 3


def test_move_edit_undoes_and_redoes_the_order():
    from chronovox.history import History, MatrixMoveEdit
    model = Model()
    ms = [model.new_matrix((1, 1, 1), n) for n in "abcd"]
    history = History()
    old = model.move_matrix(ms[0], 3)
    history.push(MatrixMoveEdit(model, ms[0], old, 3))
    assert names(model) == ["b", "c", "d", "a"]
    history.undo()
    assert names(model) == ["a", "b", "c", "d"] and model.matrices[0] is ms[0]
    history.redo()
    assert names(model) == ["b", "c", "d", "a"]


def test_the_saved_order_follows_the_list_order():
    model = Model()
    ms = [model.new_matrix((1, 1, 1), n) for n in ("Head", "Torso-box", "Arm")]
    model.move_matrix(ms[2], 0)
    assert [m.name for m in cvox.loads(cvox.dumps(model)).matrices] == ["Arm", "Head", "Torso-box"]
