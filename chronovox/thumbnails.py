"""A disk cache for the small preview images of the home page. Plain files and numpy: no GL or UI.

A preview is a square RGBA picture. It is stored as ``<hash of the model's path>_<modified time>_<size>.npy``, so
a model that has been saved again (a new modified time or size) no longer matches and is drawn afresh, and the
old file is removed when the new one is written.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

SIZE = 160          # pixels on a side of a stored preview


def file_key(path: str | Path) -> tuple[int, int] | None:
    """(modified time in ns, size in bytes) of the file, or None if it cannot be read."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return info.st_mtime_ns, info.st_size


def _stem(path: str | Path) -> str:
    return hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()[:16]


def cache_file(folder: Path, path: str | Path, key: tuple[int, int]) -> Path:
    return Path(folder) / f"{_stem(path)}_{key[0]}_{key[1]}.npy"


def load(folder: Path | None, path: str | Path, key: tuple[int, int]) -> np.ndarray | None:
    """The stored preview for this file as it is now, or None if there is none (or it cannot be used)."""
    if folder is None:
        return None
    try:
        pixels = np.load(cache_file(folder, path, key), allow_pickle=False)
    except (OSError, ValueError):
        return None
    return pixels if pixels.shape == (SIZE, SIZE, 4) and pixels.dtype == np.uint8 else None


def store(folder: Path | None, path: str | Path, key: tuple[int, int], pixels: np.ndarray) -> None:
    """Keep a preview, replacing any older one for the same file. Silent if the folder cannot be written."""
    if folder is None:
        return
    try:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob(f"{_stem(path)}_*.npy"):
            old.unlink()
        np.save(cache_file(folder, path, key), np.ascontiguousarray(pixels, np.uint8), allow_pickle=False)
    except OSError:
        pass


def prune(folder: Path | None, keep_paths) -> None:
    """Delete the stored previews of every file that is not in ``keep_paths``."""
    if folder is None or not Path(folder).is_dir():
        return
    keep = {_stem(p) for p in keep_paths}
    try:
        for entry in Path(folder).glob("*.npy"):
            if entry.name.split("_")[0] not in keep:
                entry.unlink()
    except OSError:
        pass
