import os

import pytest

from chronovox import filemanager


@pytest.fixture
def calls(monkeypatch):
    """Record what would be launched instead of opening anything on the screen."""
    log = []
    monkeypatch.setattr(filemanager.subprocess, "Popen", lambda command, *a, **k: log.append(("run", command)))
    monkeypatch.setattr(filemanager.os, "startfile", lambda folder: log.append(("startfile", folder)), raising=False)
    return log


def test_the_menu_item_says_what_the_system_calls_its_file_manager():
    assert filemanager.label("win32") == "Show in File Explorer"
    assert filemanager.label("darwin") == "Reveal in Finder"
    assert filemanager.label("linux") == "Show in file manager"


def test_on_windows_explorer_is_asked_to_select_the_file_with_its_own_quoting(tmp_path, calls):
    model = tmp_path / "My Model.cvox"
    model.write_bytes(b"x")
    assert filemanager.reveal(model, "win32") is None
    assert calls == [("run", f'explorer /select,"{os.path.normpath(model)}"')]


def test_on_mac_and_linux_the_file_or_its_folder_is_opened(tmp_path, calls):
    model = tmp_path / "a.cvox"
    model.write_bytes(b"x")
    assert filemanager.reveal(model, "darwin") is None
    assert filemanager.reveal(model, "linux") is None
    assert calls == [("run", ["open", "-R", str(model)]), ("run", ["xdg-open", str(tmp_path)])]


def test_a_file_that_is_gone_opens_the_folder_it_was_in(tmp_path, calls):
    gone = tmp_path / "deleted.cvox"
    assert filemanager.reveal(gone, "win32") is None
    assert calls == [("startfile", str(tmp_path))]


def test_a_file_and_folder_that_are_both_gone_say_so_and_open_nothing(tmp_path, calls):
    message = filemanager.reveal(tmp_path / "nowhere" / "x.cvox", "win32")
    assert "not there any more" in message and calls == []


def test_a_failure_to_launch_becomes_a_message(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("no such program")
    monkeypatch.setattr(filemanager.subprocess, "Popen", broken)
    model = tmp_path / "a.cvox"
    model.write_bytes(b"x")
    assert "no such program" in filemanager.reveal(model, "linux")
