import json

import pytest

from chronovox.preferences import Preferences, default_path


def test_defaults_when_there_is_no_file(tmp_path):
    assert Preferences.load(tmp_path / "missing.json") == Preferences()


def test_everything_survives_a_save_and_load(tmp_path):
    prefs = Preferences(window_width=1500, window_height=900, window_x=-1200, window_y=40, maximized=True,
                        ambient_occlusion=False, show_grid=False, show_inactive_outlines=False,
                        live_box_preview=False, model_panel=[10.0, 40.0, 300.0, 480.0],
                        tools_panel=[24.0, 36.0, 350.0, 620.0], brush_color=[1, 2, 3], last_dir="C:/models", bulk_resolution=5)
    path = tmp_path / "p.json"
    assert prefs.save(path)
    assert Preferences.load(path) == prefs


def test_save_creates_missing_folders_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "a" / "b" / "p.json"
    assert Preferences().save(path)
    assert [p.name for p in path.parent.iterdir()] == ["p.json"]


def test_saving_over_an_existing_file_replaces_it(tmp_path):
    path = tmp_path / "p.json"
    Preferences(maximized=False).save(path)
    Preferences(maximized=True).save(path)
    assert Preferences.load(path).maximized is True


def test_an_unwritable_location_reports_failure_instead_of_raising(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert Preferences().save(blocker / "p.json") is False


@pytest.mark.parametrize("text", ["", "not json", "[1, 2, 3]", "null", '"text"', "{", '{"window": 5}'])
def test_damaged_files_fall_back_to_defaults(tmp_path, text):
    path = tmp_path / "p.json"
    path.write_text(text, encoding="utf-8")
    assert Preferences.load(path) == Preferences()


def test_a_file_that_is_not_utf8_falls_back_to_defaults(tmp_path):
    path = tmp_path / "p.json"
    path.write_bytes(b"\xff\xfe\x00bad")
    assert Preferences.load(path) == Preferences()


def test_each_field_is_checked_on_its_own():
    prefs = Preferences.from_dict({
        "window": {"width": 1600, "height": "tall", "x": 100, "y": None, "maximized": "yes"},
        "view": {"show_grid": False, "ambient_occlusion": 1},
        "panels": {"model": [1, 2, 3], "tools": [1, 2, 3, 4]},
        "brush_color": [10, 20, 300],
        "last_dir": 5,
    })
    assert prefs.window_width == 1600                        # good value kept
    assert prefs.window_height == Preferences().window_height   # bad value replaced
    assert (prefs.window_x, prefs.window_y) == (None, None)  # a position needs both numbers
    assert prefs.maximized is False
    assert prefs.show_grid is False and prefs.ambient_occlusion is True
    assert prefs.model_panel is None and prefs.tools_panel == [1.0, 2.0, 3.0, 4.0]
    assert prefs.brush_color == Preferences().brush_color and prefs.last_dir == ""


def test_absurd_window_sizes_and_positions_are_rejected():
    prefs = Preferences.from_dict({"window": {"width": 10, "height": 10 ** 9, "x": 10 ** 9, "y": 0}})
    default = Preferences()
    assert prefs.window_width == default.window_width and prefs.window_height == default.window_height
    assert prefs.window_x is None


def test_booleans_are_not_accepted_as_numbers():
    prefs = Preferences.from_dict({"window": {"width": True, "height": False}, "brush_color": [True, 0, 0]})
    assert prefs.window_width == Preferences().window_width and prefs.brush_color == Preferences().brush_color


def test_nan_and_infinity_in_panel_lists_are_rejected():
    prefs = Preferences.from_dict(json.loads('{"panels": {"model": [1, 2, NaN, 4], "tools": [1, Infinity, 3, 4]}}'))
    assert prefs.model_panel is None and prefs.tools_panel is None


def test_unknown_fields_and_newer_versions_are_ignored_but_known_fields_still_read():
    prefs = Preferences.from_dict({"version": 99, "future": {"x": 1}, "view": {"show_grid": False}})
    assert prefs.show_grid is False


def test_the_default_path_is_a_file_in_a_chronovox_folder():
    path = default_path()
    assert path.name == "preferences.json" and "hronovox" in str(path.parent)


def test_the_set_all_resolution_is_remembered_and_checked():
    assert Preferences.from_dict({"bulk_resolution": 5}).bulk_resolution == 5
    for bad in (0, 9, "five", None, True, 2.5e9):
        assert Preferences.from_dict({"bulk_resolution": bad}).bulk_resolution == 1


def test_the_explorer_sidebar_is_wider_by_default_and_an_unchanged_old_width_follows():
    from chronovox.preferences import DEFAULT_EXPLORER_WIDTH, Preferences
    assert DEFAULT_EXPLORER_WIDTH == 390.0 and Preferences().explorer_width == 390.0
    assert Preferences.from_dict({"explorer_width": 260}).explorer_width == 390.0      # the old default: never resized
    assert Preferences.from_dict({"explorer_width": 300}).explorer_width == 300.0      # a width someone chose is kept
