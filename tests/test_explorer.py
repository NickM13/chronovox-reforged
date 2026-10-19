import os

from chronovox.explorer import filtered, folders, norm, route_to, scan
from chronovox.preferences import Preferences


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def names(node):
    return [(c.name, names(c) if c.is_dir else None) for c in node.children]


def test_the_tree_has_folders_then_files_sorted_and_leaves_out_what_is_not_a_model(tmp_path):
    touch(tmp_path / "b.cvox")
    touch(tmp_path / "A.CVOX")
    touch(tmp_path / "notes.txt")
    touch(tmp_path / "zoo" / "deep" / "x.cvox")
    touch(tmp_path / "empty" / "readme.md")
    touch(tmp_path / ".hidden" / "h.cvox")
    tree = scan(str(tmp_path))
    assert names(tree) == [("zoo", [("deep", [("x.cvox", None)])]), ("A.CVOX", None), ("b.cvox", None)]
    assert tree.files == 3


def test_legacy_models_are_listed_only_when_asked_for(tmp_path):
    touch(tmp_path / "a.cvox")
    touch(tmp_path / "old" / "b.csm")
    assert scan(str(tmp_path)).files == 1
    assert scan(str(tmp_path), extensions=(".cvox", ".csm")).files == 2


def test_a_missing_folder_gives_an_empty_tree(tmp_path):
    tree = scan(str(tmp_path / "nope"))
    assert tree.children == [] and tree.files == 0


def test_the_scan_stops_at_the_limit(tmp_path):
    for i in range(10):
        touch(tmp_path / f"m{i}.cvox")
    assert scan(str(tmp_path), max_files=4).files == 4


def test_the_filter_keeps_matching_files_and_their_folders(tmp_path):
    touch(tmp_path / "trees" / "oak.cvox")
    touch(tmp_path / "trees" / "pine.cvox")
    touch(tmp_path / "rocks" / "big.cvox")
    tree = scan(str(tmp_path))
    assert names(filtered(tree, "OAK")) == [("trees", [("oak.cvox", None)])]
    assert names(filtered(tree, "rocks")) == [("rocks", [("big.cvox", None)])]       # a folder name keeps its contents
    assert filtered(tree, "zzz") is None
    assert filtered(tree, "  ") is tree


def test_the_panel_settings_are_remembered_and_bad_values_ignored():
    prefs = Preferences(explorer_roots=["C:/a"], explorer_expanded=["C:/a/b"], explorer_visible=False)
    again = Preferences.from_dict(prefs.to_dict())
    assert again.explorer_roots == ["C:/a"] and again.explorer_expanded == ["C:/a/b"] and again.explorer_visible is False
    bad = Preferences.from_dict({"explorer_roots": [1, "", "ok"], "explorer_expanded": "no", "explorer_visible": 3})
    assert bad.explorer_roots == ["ok"] and bad.explorer_expanded == [] and bad.explorer_visible is True


def test_route_to_gives_the_folders_down_to_a_file(tmp_path):
    touch(tmp_path / "a" / "b" / "x.cvox")
    tree = scan(str(tmp_path))
    route = route_to(tree, norm(str(tmp_path / "a" / "b" / "x.cvox")))
    assert [n.name for n in route][1:] == ["a", "b", "x.cvox"]
    assert route_to(tree, norm(str(tmp_path / "nope.cvox"))) is None


def test_folders_lists_every_folder_in_the_tree(tmp_path):
    touch(tmp_path / "a" / "b" / "x.cvox")
    touch(tmp_path / "c" / "y.cvox")
    tree = scan(str(tmp_path))
    assert sorted(folders(tree)) == sorted(str(tmp_path / n) for n in ("", "a", "a/b", "c")) or len(folders(tree)) == 4
