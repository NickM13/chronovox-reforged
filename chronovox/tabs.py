"""One open document. The viewer keeps the current tab's state in its own fields and parks the others here."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .camera import Camera
from .gizmos import Mode
from .history import History
from .selection import VoxelSelection

_uids = itertools.count(1)


@dataclass
class Tab:
    model: Any = None                                    # chronovox.model.Model, or None for a tab with nothing in it yet
    items: list = field(default_factory=list)            # the viewer's DrawItems (they own GL buffers)
    path: Path | None = None
    suggested_name: str = "model.cvox"
    active: int = 0
    history: History = field(default_factory=History)
    dirty: bool = False
    camera: Camera = field(default_factory=Camera)
    mode: Mode = Mode.VOXEL
    selection: set = field(default_factory=set)          # ids of the matrices selected along with the active one
    voxel_selection: VoxelSelection = field(default_factory=VoxelSelection)   # the selected voxels of the active matrix
    uid: int = field(default_factory=lambda: next(_uids))   # never reused: a stable id for the tab bar


def tab_label(path: Path | None, dirty: bool, suggested_name: str = "model.cvox") -> str:
    """The text on a tab: the file name (or "untitled"), with a * in front while there are unsaved changes."""
    name = path.name if path else (Path(suggested_name).stem if suggested_name != "model.cvox" else "untitled")
    return ("*" if dirty else "") + name
