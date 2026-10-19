"""The list of recently opened models. Plain data: no GL, UI or file-format imports."""
from __future__ import annotations

import os

from .preferences import MAX_RECENT


def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def add_recent(recent: list[str], path: str, limit: int = MAX_RECENT) -> list[str]:
    """A new list with ``path`` first, and no other entry for the same file (case and slashes do not matter)."""
    return [str(path)] + [p for p in recent if not _same(p, path)][:limit - 1]


def remove_recent(recent: list[str], path: str) -> list[str]:
    return [p for p in recent if not _same(p, path)]
