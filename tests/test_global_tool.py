import tempfile
from pathlib import Path

from chronovox.editing import VoxelTool
from chronovox.gizmos import MatrixTool, Mode
from chronovox.viewer import Viewer


def test_the_mode_and_the_chosen_tool_stay_when_another_tab_is_shown():
    v = Viewer(visible=False, preferences_path=Path(tempfile.mkdtemp()) / "p.json")
    v.new_model()
    v._open_tab()
    v.new_model()
    v.set_voxel_tool(VoxelTool.ERASE)
    v.switch_tab(0)
    assert v.mode is Mode.VOXEL and v.voxel_tool is VoxelTool.ERASE
    v.set_mode(Mode.MATRIX)
    v.matrix_tools.tool = MatrixTool.ROTATE
    v.switch_tab(1)
    assert v.mode is Mode.MATRIX and v.matrix_tools.tool is MatrixTool.ROTATE
    v.switch_tab(0)
    assert v.mode is Mode.MATRIX and v.matrix_tools.tool is MatrixTool.ROTATE


def test_ctrl_v_pastes_matrices_in_place_and_the_focus_paste_moves_them_to_the_camera():
    from pyglm import glm
    v = Viewer(visible=False, preferences_path=Path(tempfile.mkdtemp()) / "p.json")
    v.new_model()
    original = v.model.matrices[0]
    v.active, v.selection = 0, set()
    assert v.copy_selected_matrix()
    v.camera.target = glm.vec3(40.0, 5.0, 40.0)
    assert v.paste(at_focus=False)
    assert v.model.matrices[1].position == original.position           # exactly where it was copied from
    assert v.paste(at_focus=True)
    assert v.model.matrices[2].position != original.position           # moved to the camera focus
