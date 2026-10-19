import numpy as np
import pytest

from chronovox import cvox
from chronovox.history import History, MatrixListEdit, VoxelEdit
from chronovox.model import Matrix, Model
from chronovox.naming import name_behaviour, split_suffix, variant_name


# --- names for copies ------------------------------------------------------------------------

def test_split_suffix_recognizes_the_game_suffixes():
    assert split_suffix("Torso-box") == ("Torso", "-box")
    assert split_suffix("Shield+") == ("Shield", "+")
    assert split_suffix("Wing-") == ("Wing", "-")
    assert split_suffix("LeftArm") == ("LeftArm", "")
    assert split_suffix("box") == ("box", "")


def test_a_copy_gets_the_next_free_number_with_no_spaces():
    assert variant_name("Arm", {"Arm"}) == "Arm2"
    assert variant_name("Arm", {"Arm", "Arm2", "Arm3"}) == "Arm4"
    assert variant_name("Arm2", {"Arm", "Arm2"}) == "Arm3"          # counts up from a numbered name
    assert variant_name("Arm7", {"Arm7"}) == "Arm2"                 # first free number, not "7 + 1"
    assert variant_name("Left Arm", {"Left Arm"}) == "Left Arm2"     # an existing space is left as it was, none is added


def test_the_number_goes_before_the_game_suffix_so_the_copy_behaves_the_same():
    assert variant_name("Torso-box", {"Torso-box"}) == "Torso2-box"
    assert variant_name("Shield+", {"Shield+"}) == "Shield2+"
    assert variant_name("Wing-", {"Wing-"}) == "Wing2-"
    assert variant_name("Arm2-box", {"Arm2-box"}) == "Arm3-box"
    for name in ("LeftArm", "Head+", "Cape-", "Torso-box", "Torso3-box"):
        copy = variant_name(name, {name})
        assert name_behaviour(copy) == name_behaviour(name) and copy != name


def test_a_copy_never_takes_a_name_that_is_in_use_in_any_form():
    taken = {"Torso-box", "Torso2-box", "Torso3-box", "Torso2"}
    assert variant_name("Torso-box", taken) == "Torso4-box"


# --- duplicating in the model --------------------------------------------------------------------

def make_model():
    model = Model(np.array([[1, 2, 3, 255], [4, 5, 6, 255]], np.uint8))
    a = model.new_matrix((3, 4, 5), "Arm", (128, 256, -384))
    a.voxels[1, 2, 0] = 1
    a.voxels[4, 3, 2] = 2
    a.pivot = (10, 20, 30)
    a.rotation = (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)
    model.new_matrix((2, 2, 2), "Head")
    return model, a


def test_duplicate_copies_everything_and_sits_right_after_the_original():
    model, arm = make_model()
    copy, index = model.duplicate_matrix(arm)
    assert index == 1 and [m.name for m in model.matrices] == ["Arm", "Arm2", "Head"]
    assert model.matrices[1] is copy and copy is not arm
    assert np.array_equal(copy.voxels, arm.voxels) and copy.size == arm.size
    assert (copy.position, copy.pivot, copy.rotation) == (arm.position, arm.pivot, arm.rotation)


def test_the_copy_is_independent_of_the_original():
    model, arm = make_model()
    copy, _ = model.duplicate_matrix(arm)
    copy.voxels[:] = 0
    copy.position = (0, 0, 0)
    assert arm.voxels.any() and arm.position == (128, 256, -384)
    arm.voxels[0, 0, 0] = 2
    assert copy.voxels[0, 0, 0] == 0


def test_duplicating_twice_gives_distinct_names():
    model, arm = make_model()
    first, _ = model.duplicate_matrix(arm)
    second, _ = model.duplicate_matrix(arm)
    assert len({m.name for m in model.matrices}) == 4 and {first.name, second.name} == {"Arm2", "Arm3"}


def test_duplicating_into_a_full_model_is_refused(monkeypatch):
    import chronovox.model as mod
    model, arm = make_model()
    monkeypatch.setattr(mod, "MAX_MATRICES", 2)
    with pytest.raises(ValueError):
        model.duplicate_matrix(arm)
    assert [m.name for m in model.matrices] == ["Arm", "Head"]


def test_remove_and_insert_by_identity():
    model, arm = make_model()
    twin = Matrix("Arm", arm.voxels.copy())                     # same name, different object
    model.insert_matrix(5, twin)                                # an index past the end just appends
    assert model.matrices[-1] is twin
    assert model.remove_matrix(arm) == 0 and twin in model.matrices
    with pytest.raises(ValueError):
        model.remove_matrix(arm)


def test_a_duplicated_model_saves_and_loads():
    model, arm = make_model()
    model.duplicate_matrix(arm)
    back = cvox.loads(cvox.dumps(model))
    assert [m.name for m in back.matrices] == ["Arm", "Arm2", "Head"]
    assert np.array_equal(back.matrices[1].voxels, arm.voxels)


# --- undoing structure changes ----------------------------------------------------------------------

def test_deleting_and_undoing_puts_the_same_matrix_back_in_the_same_place():
    model, arm = make_model()
    history = History()
    head = model.matrices[1]
    edit = MatrixListEdit(model, head, model.matrices.index(head), added=False)
    edit.redo()
    history.push(edit)
    assert model.matrices == [arm]
    history.undo()
    assert model.matrices[1] is head
    history.redo()
    assert model.matrices == [arm]


def test_a_delete_from_the_middle_returns_to_the_middle():
    model, arm = make_model()
    last = model.new_matrix((1, 1, 1), "Tail")
    history = History()
    edit = MatrixListEdit(model, arm, model.remove_matrix(arm), added=False)
    history.push(edit)
    assert [m.name for m in model.matrices] == ["Head", "Tail"]
    history.undo()
    assert [m.name for m in model.matrices] == ["Arm", "Head", "Tail"]


def test_undoing_an_add_removes_it_and_redo_restores_it():
    model, arm = make_model()
    history = History()
    copy, index = model.duplicate_matrix(arm)
    history.push(MatrixListEdit(model, copy, index, added=True))
    history.undo()
    assert [m.name for m in model.matrices] == ["Arm", "Head"]
    history.redo()
    assert model.matrices[1] is copy


def test_voxel_edits_made_before_a_delete_still_undo_correctly_after_the_delete_is_undone():
    model, arm = make_model()
    history = History()
    before = arm.voxels.copy()
    arm.voxels[0, 0, 0] = 2
    history.push(VoxelEdit.between(arm, before))
    edit = MatrixListEdit(model, arm, model.remove_matrix(arm), added=False)
    history.push(edit)
    history.undo()                       # the delete
    history.undo()                       # the voxel edit, on the very same matrix object
    assert model.matrices[0] is arm and arm.voxels[0, 0, 0] == 0 and arm.voxels[1, 2, 0] == 1


def test_a_removed_matrix_counts_toward_the_history_budget_and_a_present_one_does_not():
    model, arm = make_model()
    edit = MatrixListEdit(model, arm, 0, added=True)
    assert edit.nbytes == 0
    model.remove_matrix(arm)
    assert edit.nbytes == arm.voxels.nbytes


def test_descriptions_name_the_matrix():
    model, arm = make_model()
    assert "adding Arm" == MatrixListEdit(model, arm, 0, True).describe()
    assert "deleting Arm" == MatrixListEdit(model, arm, 0, False).describe()
