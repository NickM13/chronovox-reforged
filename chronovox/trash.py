"""Move a file to the system's trash or Recycle Bin instead of deleting it for good. Standard library only."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _windows(path: str) -> str | None:
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x4, 0x10, 0x40, 0x400
    operation = SHFILEOPSTRUCTW()
    operation.wFunc = FO_DELETE
    operation.pFrom = path + "\0"                       # a list of names, ended by an extra null
    operation.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT   # ALLOWUNDO: Recycle Bin
    code = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if code != 0 or operation.fAnyOperationsAborted:
        return f"Windows could not move it to the Recycle Bin (code {code})."
    return None


def move_to_trash(path: str | Path, platform: str | None = None) -> str | None:
    """Move a file to the trash. Returns None on success, or a short message saying why it could not be done.

    It never deletes for good: if the system has no trash this knows how to use, the file is left alone.
    """
    platform = platform or sys.platform
    target = os.path.abspath(os.path.normpath(str(path)))
    if not os.path.exists(target):
        return "That file is not there any more."
    try:
        if platform == "win32":
            return _windows(target)
        if platform == "darwin":
            script = f'tell application "Finder" to delete POSIX file "{target}"'
            done = subprocess.run(["osascript", "-e", script], capture_output=True, timeout=20)
            return None if done.returncode == 0 else "Finder could not move it to the Trash."
        for command in (["gio", "trash", target], ["trash-put", target]):
            if shutil.which(command[0]):
                done = subprocess.run(command, capture_output=True, timeout=20)
                return None if done.returncode == 0 else "The trash command failed."
        return "This system has no trash command that Chronovox knows how to use, so the file was left alone."
    except (OSError, subprocess.SubprocessError) as error:
        return f"Could not move it to the trash: {error}"
