from pathlib import Path

from chronovox.tabs import Tab, tab_label


def test_label_is_the_file_name_or_untitled_with_a_star_when_unsaved():
    assert tab_label(Path("C:/models/Dragon.cvox"), False) == "Dragon.cvox"
    assert tab_label(Path("C:/models/Dragon.cvox"), True) == "*Dragon.cvox"
    assert tab_label(None, False) == "untitled"
    assert tab_label(None, True) == "*untitled"


def test_an_imported_model_is_labelled_by_the_name_it_will_be_saved_as():
    assert tab_label(None, True, "dragon.cvox") == "*dragon"


def test_every_tab_has_its_own_history_camera_and_id():
    a, b = Tab(), Tab()
    assert a.history is not b.history and a.camera is not b.camera and a.uid != b.uid
