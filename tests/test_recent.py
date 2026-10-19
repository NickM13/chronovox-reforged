from chronovox.preferences import MAX_RECENT, Preferences
from chronovox.recent import add_recent, remove_recent


def test_the_newest_file_goes_first_and_a_reopened_one_moves_up_without_a_duplicate():
    recent = add_recent(add_recent([], "C:/m/a.cvox"), "C:/m/b.cvox")
    assert recent == ["C:/m/b.cvox", "C:/m/a.cvox"]
    assert add_recent(recent, "C:/m/a.cvox") == ["C:/m/a.cvox", "C:/m/b.cvox"]


def test_the_same_file_spelled_differently_counts_once():
    import os
    first = os.path.join("C:\\", "m", "a.cvox")
    assert len(add_recent([first], first.replace("\\", "/"))) == 1


def test_the_list_is_capped_and_drops_the_oldest():
    recent = []
    for i in range(MAX_RECENT + 5):
        recent = add_recent(recent, f"C:/m/{i}.cvox")
    assert len(recent) == MAX_RECENT and recent[0] == f"C:/m/{MAX_RECENT + 4}.cvox" and "C:/m/0.cvox" not in recent


def test_remove_recent():
    assert remove_recent(["C:/m/a.cvox", "C:/m/b.cvox"], "C:/m/a.cvox") == ["C:/m/b.cvox"]


def test_preferences_keep_the_recent_files_and_ignore_bad_entries():
    prefs = Preferences(recent_files=["C:/m/a.cvox", "C:/m/b.cvox"])
    assert Preferences.from_dict(prefs.to_dict()).recent_files == prefs.recent_files
    assert Preferences.from_dict({"recent_files": ["ok.cvox", 5, "", None]}).recent_files == ["ok.cvox"]
    assert Preferences.from_dict({"recent_files": "nope"}).recent_files == []
    assert Preferences.from_dict({}).recent_files == []
    assert len(Preferences.from_dict({"recent_files": [f"{i}.cvox" for i in range(MAX_RECENT + 50)]}).recent_files) == MAX_RECENT


def test_preferences_keep_how_the_recent_models_are_shown():
    assert Preferences().recent_view == "list"
    assert Preferences.from_dict(Preferences(recent_view="grid").to_dict()).recent_view == "grid"
    assert Preferences.from_dict({"recent_view": "tiles"}).recent_view == "list"
    assert Preferences.from_dict({"recent_view": 3}).recent_view == "list"
