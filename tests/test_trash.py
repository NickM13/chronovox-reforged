import sys

import pytest

from chronovox.trash import move_to_trash


def test_a_missing_file_is_reported_and_nothing_happens(tmp_path):
    assert "not there" in move_to_trash(tmp_path / "nope.cvox")


def test_without_a_known_trash_the_file_is_left_alone(tmp_path, monkeypatch):
    path = tmp_path / "keep.cvox"
    path.write_bytes(b"x")
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert "left alone" in move_to_trash(path, platform="linux")
    assert path.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="uses the Windows Recycle Bin")
def test_a_file_goes_to_the_recycle_bin(tmp_path):
    path = tmp_path / "recycle_me.cvox"
    path.write_bytes(b"x")
    assert move_to_trash(path) is None
    assert not path.exists()
