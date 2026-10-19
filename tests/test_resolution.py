import numpy as np
import pytest

from chronovox import transform
from chronovox.gizmos import MatrixTool
from chronovox.history import History
from chronovox.matrix_ops import (merge_matrix, merge_problem, move_pivot_in_place, plan_resize, resize_matrix,
                                  set_all_resolutions, set_resolution, set_transform)
from chronovox.matrix_tools import MatrixTools, model_point, pick_matrix, pivot_world, to_local
from chronovox.model import MAX_RESOLUTION, POSITION_SCALE, Matrix, Model

S = POSITION_SCALE


def mat(name="m", size=(4, 3, 2), position=(0, 0, 0), resolution=1, rotation=transform.IDENTITY, cells=()):
    m = Matrix(name, np.zeros((size[2], size[1], size[0]), np.uint16), position, rotation=rotation,
               resolution=resolution)
    for (x, y, z), index in cells:
        m.voxels[z, y, x] = index
    return m


def world(m, local):
    return model_point(m.position, m.pivot, m.rotation, local, m.cell_size)


def world_voxels(m):
    return {tuple(np.round(world(m, (x + .5, y + .5, z + .5)), 4)): int(m.voxels[z, y, x])
            for z, y, x in np.argwhere(m.voxels != 0)}


# --- what a resolution means ---------------------------------------------------------------------------

def test_a_higher_resolution_means_smaller_cells_halving_each_step():
    assert [mat(resolution=n).cell_size for n in (1, 2, 3, 4, 5, 8)] == [1, 0.5, 0.25, 0.125, 0.0625, 1 / 128]


def test_the_cell_size_in_position_units_is_a_whole_number_up_to_the_finest_resolution():
    assert [mat(resolution=n).cell_size_128 for n in range(1, MAX_RESOLUTION + 1)] == [128, 64, 32, 16, 8, 4, 2, 1]
    assert MAX_RESOLUTION == 8                       # one cell is 1/128 unit there, the finest a position can address


def test_the_default_resolution_is_one_and_a_copy_keeps_it():
    assert Matrix("m", np.zeros((1, 1, 1), np.uint16)).resolution == 1
    model = Model(np.zeros((0, 4), np.uint8), [mat("leaves", resolution=4)])
    copy, _ = model.duplicate_matrix(model.matrices[0])
    assert copy.resolution == 4


def test_compacting_and_making_opaque_keep_the_resolution():
    palette = np.array([[1, 2, 3, 1], [4, 5, 6, 255]], np.uint8)
    model = Model(palette, [mat("a", resolution=3, cells=[((0, 0, 0), 2)])])
    assert model.compacted().matrices[0].resolution == 3
    assert model.opaque().matrices[0].resolution == 3


# --- placement ---------------------------------------------------------------------------------------------

def test_a_finer_matrix_keeps_its_origin_corner_at_the_position_and_its_cells_are_small():
    m = mat(resolution=3, position=(5 * S, 6 * S, 7 * S))
    assert np.allclose(world(m, (0, 0, 0)), (5, 6, 7))
    assert np.allclose(world(m, (1, 0, 0)), (5.25, 6, 7))            # one cell is a quarter of a unit
    assert np.allclose(world(m, (4, 4, 4)), (6, 7, 8))


def test_cells_scale_about_the_pivot_so_the_pivot_point_sits_at_position_plus_f_pivot():
    m = mat(resolution=4, position=(S, 0, 0))
    m.pivot = (2 * S, S, S // 2)
    assert np.allclose(world(m, np.array(m.pivot) / S), pivot_world(m))
    assert np.allclose(pivot_world(m), np.array([1, 0, 0]) + np.array([2, 1, 0.5]) / 8)


def test_rotation_turns_a_fine_matrix_about_its_pivot_point():
    m = mat(resolution=3, position=(3 * S, 0, 0), rotation=transform.from_euler(0, 90, 0))
    pivot = pivot_world(m)
    assert np.allclose(world(m, np.array(m.pivot) / S), pivot)
    step = world(m, np.array(m.pivot) / S + (1, 0, 0)) - pivot       # one cell along local x
    assert np.allclose(step, 0.25 * transform.rotate(m.rotation, (1, 0, 0)))


def test_to_local_undoes_model_point_with_resolution_rotation_and_an_odd_pivot():
    m = mat(resolution=4, size=(5, 4, 3), position=(300, -90, 12), rotation=transform.from_euler(10, 50, -20))
    m.pivot = (77, 200, 5)
    for p in [(0, 0, 0), (2.5, 1, 3), (-4, 9, 1.25)]:
        assert np.allclose(to_local(m, world(m, p)), p)


# --- resizing keeps voxels where they are ----------------------------------------------------------------

def test_growing_the_low_side_of_a_fine_matrix_keeps_every_voxel_in_place():
    m = mat(size=(3, 3, 3), resolution=3, position=(4 * S, 0, S), rotation=transform.from_euler(15, 40, 5),
            cells=[((0, 0, 0), 1), ((2, 1, 2), 2)])
    m.pivot = (90, 250, 40)
    before = world_voxels(m)
    resize_matrix(m, (2, 1, 3), (1, 0, 0))
    assert m.size == (6, 4, 6) and world_voxels(m) == before


def test_the_origin_corner_moves_by_the_cell_size_times_the_cells_added():
    m = mat(size=(2, 2, 2), resolution=4, position=(0, 0, 0))
    _, position, pivot = plan_resize(m, (3, 0, 0), (0, 0, 0))
    assert position == (-3 * 16, 0, 0) and pivot[0] == m.pivot[0] + 3 * S


def test_positions_stay_whole_numbers_at_every_resolution():
    for n in range(1, MAX_RESOLUTION + 1):
        m = mat(resolution=n)
        _, position, _ = plan_resize(m, (3, 1, 2), (0, 0, 0))
        assert all(isinstance(p, int) for p in position)


# --- changing the resolution --------------------------------------------------------------------------------

def test_a_higher_resolution_shrinks_the_matrix_about_the_world_origin():
    m = mat(size=(4, 4, 4), resolution=1, position=(8 * S, 4 * S, -12 * S), cells=[((1, 2, 3), 1), ((0, 0, 0), 1)])
    before = world_voxels(m)
    edit = set_resolution(m, 3)                                       # cells go from 1 to 1/4 unit
    assert m.resolution == 3 and m.position == (2 * S, 1 * S, -3 * S)  # the position shrinks by the same factor
    assert world_voxels(m) == {tuple(np.round(np.array(k) / 4, 4)): v for k, v in before.items()}
    edit.undo()
    assert m.resolution == 1 and m.position == (8 * S, 4 * S, -12 * S)


def test_a_lower_resolution_grows_it_about_the_origin_again():
    m = mat(size=(4, 4, 4), resolution=3, position=(2 * S, S, -3 * S), cells=[((1, 2, 3), 1)])
    before = world_voxels(m)
    set_resolution(m, 1)
    assert m.position == (8 * S, 4 * S, -12 * S)
    assert world_voxels(m) == {tuple(np.round(np.array(k) * 4, 4)): v for k, v in before.items()}


def test_a_matrix_at_the_origin_stays_there_whatever_its_pivot():
    m = mat(size=(4, 4, 4), resolution=1, position=(0, 0, 0))
    m.pivot = (37, 200, 5)
    set_resolution(m, 5)
    assert m.position == (0, 0, 0) and m.pivot == (37, 200, 5)         # the pivot is in cells: it is not touched


def test_a_rotated_matrix_scales_about_the_origin_within_half_a_step():
    m = mat(size=(3, 5, 2), resolution=2, position=(5 * S + 3, 2 * S + 1, -3 * S), rotation=transform.from_euler(20, 70, -30),
            cells=[((1, 2, 1), 1)])
    m.pivot = (50, 300, 77)
    before = world_voxels(m)
    set_resolution(m, 5)                                                # cells 1/2 -> 1/16: a factor of 1/8
    for (x, y, z), value in before.items():
        assert any(np.allclose((x / 8, y / 8, z / 8), k, atol=1.0 / S) and value == v for k, v in world_voxels(m).items())


def test_an_assembly_of_matrices_keeps_its_shape_when_they_all_change_together():
    a = mat("a", size=(2, 2, 2), position=(16 * S, 0, 0), cells=[((0, 0, 0), 1)])
    b = mat("b", size=(2, 2, 2), position=(0, 16 * S, 32 * S), cells=[((1, 1, 1), 2)])
    model = Model(np.array([[1, 1, 1, 255], [2, 2, 2, 255]], np.uint8), [a, b])
    gap = np.array(list(world_voxels(a))[0]) - np.array(list(world_voxels(b))[0])
    set_all_resolutions(model, 5)
    new_gap = np.array(list(world_voxels(a))[0]) - np.array(list(world_voxels(b))[0])
    assert np.allclose(new_gap, gap / 16, atol=1e-3)          # the keys are rounded to 4 decimals


# --- all at once ------------------------------------------------------------------------------------------------

def test_set_all_gives_every_matrix_the_resolution_as_one_undo_step():
    model = Model(np.zeros((0, 4), np.uint8), [mat("a", resolution=1, position=(16 * S, 0, 0)),
                                               mat("b", resolution=3, position=(S, 0, 0)), mat("c", resolution=5)])
    history = History()
    edit = set_all_resolutions(model, 5)
    history.push(edit)
    assert [m.resolution for m in model.matrices] == [5, 5, 5]
    assert model.matrices[0].position == (S, 0, 0) and model.matrices[1].position == (S // 4, 0, 0)
    assert "2 matrices" in edit.describe() and "resolution 5" in edit.describe()   # c was already there
    history.undo()
    assert [m.resolution for m in model.matrices] == [1, 3, 5]
    assert model.matrices[0].position == (16 * S, 0, 0) and model.matrices[1].position == (S, 0, 0)
    history.redo()
    assert [m.resolution for m in model.matrices] == [5, 5, 5]


def test_set_all_with_nothing_to_change_returns_none_and_a_bad_level_is_refused():
    model = Model(np.zeros((0, 4), np.uint8), [mat("a", resolution=4), mat("b", resolution=4)])
    assert set_all_resolutions(model, 4) is None
    assert set_all_resolutions(Model(), 3) is None
    for bad in (0, MAX_RESOLUTION + 1):
        with pytest.raises(ValueError):
            set_all_resolutions(model, bad)
    assert [m.resolution for m in model.matrices] == [4, 4]


def test_a_group_edit_undoes_in_reverse_order_and_counts_its_parts():
    from chronovox.history import GroupEdit
    m = mat("m", resolution=1, position=(8 * S, 0, 0))
    first, second = set_resolution(m, 2), set_resolution(m, 4)
    group = GroupEdit("two changes", [first, second])
    group.undo()
    assert m.resolution == 1 and m.position == (8 * S, 0, 0)
    group.redo()
    assert m.resolution == 4 and group.nbytes == first.nbytes + second.nbytes


def test_resolution_changes_undo_and_redo_with_everything_else():
    m = mat(resolution=2)
    history = History()
    history.push(set_resolution(m, 5))
    history.push(set_transform(m, position=(S, 0, 0)))
    history.undo()
    history.undo()
    assert m.resolution == 2 and m.position == (0, 0, 0)
    history.redo()
    assert m.resolution == 5


def test_a_resolution_outside_the_range_is_refused_and_changes_nothing():
    m = mat(resolution=2)
    for bad in (0, MAX_RESOLUTION + 1, -3):
        with pytest.raises(ValueError):
            set_resolution(m, bad)
        with pytest.raises(ValueError):
            set_transform(m, resolution=bad)
    assert m.resolution == 2 and set_resolution(m, 2) is None


def test_moving_the_pivot_of_a_fine_matrix_keeps_it_in_place():
    m = mat(resolution=3, size=(4, 4, 4), position=(S, S, S), rotation=transform.from_euler(10, 30, 20),
            cells=[((1, 2, 3), 1)])
    before = world_voxels(m)
    move_pivot_in_place(m, (400, 20, 90))
    for key, value in before.items():
        assert any(np.allclose(key, k, atol=1.0 / S) and value == v for k, v in world_voxels(m).items())


# --- handles and clicking ------------------------------------------------------------------------------------

def test_a_resize_drag_counts_cells_so_finer_cells_take_less_distance():
    m = mat(size=(4, 3, 2), resolution=3)                              # a cell is a quarter of a unit
    tools = MatrixTools()
    tools.tool = MatrixTool.RESIZE
    tools.hover = ("face", (0, 1))
    face = world(m, (4, 1.5, 1))
    ray = lambda x: (np.array([x, face[1], -50.0]), np.array([0.0, 0.0, 1.0]))
    assert tools.begin_drag(m, ray(face[0]))
    tools.update_drag(ray(face[0] + 1.0), snap=False)                  # one unit = four cells
    assert tools.drag.cells == 4
    tools.update_drag(ray(face[0] + 0.3), snap=False)                  # 0.3 / 0.25 = 1.2 cells
    assert tools.drag.cells == 1


def test_the_move_handle_sits_at_the_pivot_and_moves_in_model_units():
    m = mat(size=(4, 4, 4), resolution=3, position=(2 * S, 0, 0))
    tools = MatrixTools()
    tools.tool = MatrixTool.MOVE
    tools.hover = ("axis", 0)
    anchor = pivot_world(m)
    ray = lambda x: (np.array([x, anchor[1], -50.0]), np.array([0.0, 0.0, 1.0]))
    assert tools.begin_drag(m, ray(anchor[0]))
    tools.update_drag(ray(anchor[0] + 3.0), snap=False)
    assert m.position[0] == 2 * S + 3 * S                              # three units, whatever the resolution


def test_clicking_picks_through_fine_voxels_and_prefers_the_nearer_matrix():
    fine = mat("fine", size=(8, 8, 8), resolution=3, position=(0, 0, 0))      # 2 units across
    fine.voxels[:] = 1
    coarse = mat("coarse", size=(2, 2, 2), resolution=1, position=(0, 0, -6 * S))
    coarse.voxels[:] = 1
    ray = (np.array([1.0, 1.0, -40.0]), np.array([0.0, 0.0, 1.0]))
    assert pick_matrix([fine, coarse], ray) == 1                       # the coarse one (z -6 to -4) is in front
    assert pick_matrix([fine], ray) == 0
    miss = (np.array([5.0, 1.0, -40.0]), np.array([0.0, 0.0, 1.0]))
    assert pick_matrix([fine], miss) is None


# --- merging ----------------------------------------------------------------------------------------------------

def test_matrices_with_different_resolutions_cannot_be_merged():
    a, b = mat("a", resolution=2), mat("b", resolution=3)
    problem = merge_problem(b, a)
    assert "different resolutions" in problem and "3 and 2" in problem
    assert merge_problem(mat("c", resolution=3), b) is None


def test_matrices_with_the_same_resolution_merge_on_their_fine_grid_without_moving_a_voxel():
    a = mat("a", size=(3, 3, 3), resolution=3, position=(0, 0, 0), cells=[((0, 0, 0), 1)])
    b = mat("b", size=(2, 2, 2), resolution=3, position=(5 * 32, 32, 0), cells=[((1, 1, 1), 2)])   # 5 cells along, 1 up
    before = {**world_voxels(a), **world_voxels(b)}
    model = Model(np.array([[1, 1, 1, 255], [2, 2, 2, 255]], np.uint8), [a, b])
    merge_matrix(model, b, a)
    assert world_voxels(a) == before and a.resolution == 3 and a.size == (7, 3, 3)


def test_a_half_cell_offset_between_fine_matrices_is_refused():
    a = mat("a", resolution=3)
    b = mat("b", resolution=3, position=(16, 0, 0))                    # half of a 32-unit-wide cell step
    assert "voxel grid" in merge_problem(b, a)
    b.position = (32, 0, 0)
    assert merge_problem(b, a) is None


def test_set_all_resolutions_can_be_limited_to_some_matrices():
    from chronovox.model import POSITION_SCALE
    model = Model(np.array([[1, 2, 3, 255]], np.uint8), [Matrix(n, np.ones((2, 2, 2), np.uint16), position=(i * 128, 0, 0))
                                                         for i, n in enumerate("ABC")])
    a, b, c = model.matrices
    edit = set_all_resolutions(model, 3, [a, c])
    assert (a.resolution, b.resolution, c.resolution) == (3, 1, 3)
    assert len(edit.edits) == 2
    edit.undo()
    assert (a.resolution, b.resolution, c.resolution) == (1, 1, 1)
    assert set_all_resolutions(model, 1, [a, b]) is None                  # the chosen ones already have it


def test_the_cell_count_per_unit_is_worded_for_each_resolution():
    import numpy as np
    from chronovox.model import Matrix
    from chronovox.viewer import cells_per_unit
    words = [cells_per_unit(Matrix("m", np.zeros((1, 1, 1), np.uint16), resolution=r)) for r in (1, 2, 3, 8)]
    assert words == ["1 cell", "2 cells", "4 cells", "128 cells"]


def test_a_new_matrix_can_start_at_a_finer_resolution():
    import pytest
    from chronovox.model import Model
    model = Model()
    matrix = model.new_matrix((4, 4, 4), "a", (0, 0, 0), resolution=3)
    assert matrix.resolution == 3 and matrix.cell_size == 0.25
    assert model.new_matrix((1, 1, 1)).resolution == 1
    with pytest.raises(ValueError):
        model.new_matrix((1, 1, 1), resolution=9)
