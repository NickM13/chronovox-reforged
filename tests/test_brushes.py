from chronovox.editing import (ACTIONS, BRUSHES, FILL_FOR_ACTION, FILL_TOOLS, Brush, Tool, effective_tool)


def test_the_actions_and_brushes_are_the_ones_on_the_keys():
    assert [a.value for a in ACTIONS] == ["Place", "Erase", "Paint", "Pick color", "Select"]
    assert [b.value for b in BRUSHES] == ["Voxel", "Box", "Fill"]


def test_voxel_and_box_brushes_use_the_action_itself():
    for action in ACTIONS:
        assert effective_tool(action, Brush.VOXEL) is action
        assert effective_tool(action, Brush.BOX) is action


def test_the_fill_brush_picks_the_fill_tool_for_each_action():
    assert effective_tool(Tool.PLACE, Brush.FILL) is Tool.FILL_APPEND
    assert effective_tool(Tool.ERASE, Brush.FILL) is Tool.FILL_ERASE
    assert effective_tool(Tool.PAINT, Brush.FILL) is Tool.FILL_REPLACE


def test_pick_ignores_the_brush():
    for brush in BRUSHES:
        assert effective_tool(Tool.PICK, brush) is Tool.PICK


def test_every_fill_tool_is_reachable_and_nothing_else_maps_to_one():
    reachable = {effective_tool(a, b) for a in ACTIONS for b in BRUSHES} & set(FILL_TOOLS)
    assert reachable == set(FILL_TOOLS) == set(FILL_FOR_ACTION.values())
