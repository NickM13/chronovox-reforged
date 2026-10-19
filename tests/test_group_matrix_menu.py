import tempfile
from pathlib import Path

import numpy as np
import pytest

from chronovox.model import POSITION_SCALE, Matrix, Model


@pytest.fixture
def viewer():
    from chronovox.viewer import Viewer
    v = Viewer(visible=False, preferences_path=Path(tempfile.mkdtemp()) / "p.json")
    model = Model()
    model.palette = np.array([[255, 0, 0, 255]], np.uint8)
    model.matrices += [Matrix("A", np.ones((2, 2, 2), np.uint16), pivot=(0, 0, 0)),
                       Matrix("B", np.ones((2, 2, 4), np.uint16), pivot=(0, 0, 0)),
                       Matrix("C", np.ones((2, 2, 2), np.uint16))]
    v.set_model(model)
    v.active, v.selection = 0, {id(model.matrices[0]), id(model.matrices[1])}
    yield v


def test_scale_double_changes_every_selected_matrix_as_one_step(viewer):
    a, b, c = viewer.model.matrices
    assert viewer.scale_selected(True)
    assert a.size == (4, 4, 4) and b.size == (8, 4, 4) and c.size == (2, 2, 2)
    viewer.undo()
    assert a.size == (2, 2, 2) and b.size == (4, 2, 2)


def test_scale_half_that_cannot_work_on_one_matrix_changes_none(viewer):
    a, b, c = viewer.model.matrices
    c_item = next(i for i, it in enumerate(viewer.items) if it.matrix is c)
    viewer.active, viewer.selection = c_item, {id(b), id(c)}
    viewer.model.matrices[2].voxels = np.ones((1, 1, 1), np.uint16)      # cannot be halved
    assert not viewer.scale_selected(False)
    assert b.size == (4, 2, 2)


def test_center_pivot_changes_every_selected_matrix_as_one_step(viewer):
    a, b, c = viewer.model.matrices
    assert viewer.center_selected_pivots()
    assert a.pivot == (POSITION_SCALE, POSITION_SCALE, POSITION_SCALE)
    assert b.pivot == (2 * POSITION_SCALE, POSITION_SCALE, POSITION_SCALE)
    viewer.undo()
    assert a.pivot == (0, 0, 0) and b.pivot == (0, 0, 0)


def test_ctrl_a_and_ctrl_d_choose_matrices_unless_a_select_tool_is_in_use(viewer):
    from chronovox.editing import VoxelTool
    from chronovox.gizmos import Mode
    c = viewer.model.matrices[2]
    viewer.set_mode(Mode.VOXEL)
    viewer.set_voxel_tool(VoxelTool.PLACE)
    viewer.select_all_smart()
    assert len(viewer.selected_indices()) == 3 and viewer.selection_mask() is None
    viewer.deselect_smart()
    assert viewer.selected_indices() == []
    viewer.active, viewer.selection = 0, set()
    viewer.set_voxel_tool(VoxelTool.SELECT_BOX)
    viewer.select_all_smart()                                    # a select tool: the voxels of the matrix
    assert viewer.selection_count() == 8 and len(viewer.selected_indices()) == 1
    viewer.deselect_smart()
    assert viewer.selection_mask() is None and viewer.selected_indices() == [0]
