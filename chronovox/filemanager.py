"""Show a file in the system's file manager. Standard library only: no GL or UI imports."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def label(platform: str | None = None) -> str:
    """What the menu item should say on this system."""
    platform = platform or sys.platform
    if platform == "win32":
        return "Show in File Explorer"
    return "Reveal in Finder" if platform == "darwin" else "Show in file manager"


def reveal(path: str | Path, platform: str | None = None) -> str | None:
    """Open the file manager with this file selected, or, if the file is gone, on the folder it was in.

    Returns None if something was opened, or a short message saying why nothing could be.
    """
    platform = platform or sys.platform
    target = Path(os.path.normpath(str(path)))
    folder = target.parent
    try:
        if target.exists():
            if platform == "win32":
                subprocess.Popen(f'explorer /select,"{target}"')          # one string: explorer wants its own quoting
            elif platform == "darwin":
                subprocess.Popen(["open", "-R", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
            return None
        if folder.is_dir():
            if platform == "win32":
                os.startfile(str(folder))                                  # noqa: S606 - opens the folder in Explorer
            else:
                subprocess.Popen(["open" if platform == "darwin" else "xdg-open", str(folder)])
            return None
    except (OSError, AttributeError) as error:
        return f"Could not open the file manager: {error}"
    return f"{target.name} and its folder are not there any more."
