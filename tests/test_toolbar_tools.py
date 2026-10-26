from chronovox.editing import (ACTIONS, BRUSHES, TOOL_PARTS, VOXEL_TOOLS, Brush, Tool, VoxelTool, effective_tool,
                               voxel_tool_for)


def test_every_tool_maps_to_its_own_action_and_brush_and_back():
    assert len(set(TOOL_PARTS.values())) == len(VOXEL_TOOLS) == 7
    for tool, (action, brush) in TOOL_PARTS.items():
        assert voxel_tool_for(action, brush) is tool


def test_the_toolbox_order():
    assert [t.value for t in VOXEL_TOOLS] == ["Place", "Erase", "Replace", "Pick color", "Box select", "Wand select", "Fill"]


def test_a_box_brush_counts_as_the_single_voxel_tool():
    for action in ACTIONS:
        assert voxel_tool_for(action, Brush.BOX) is voxel_tool_for(action, Brush.VOXEL)


def test_fill_is_one_tool_whether_it_appends_or_replaces():
    assert voxel_tool_for(Tool.PLACE, Brush.FILL) is VoxelTool.FILL
    assert voxel_tool_for(Tool.PAINT, Brush.FILL) is VoxelTool.FILL
    assert effective_tool(Tool.PLACE, Brush.FILL) is Tool.FILL_APPEND
    assert effective_tool(Tool.PAINT, Brush.FILL) is Tool.FILL_REPLACE
    assert {b for _, b in TOOL_PARTS.values()} <= set(BRUSHES)
