"""The small decisions behind the workspace window that need no UI or network: file names, dates and link status."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from . import cvox
from .workspace_api import sha256_hex
from .workspace_store import Link, LinkStore

_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(name: str) -> str:
    """A model's name as a file name: no characters Windows refuses, no dots or spaces at the end."""
    cleaned = _BAD.sub("_", name).strip().rstrip(". ")
    return cleaned or "model"


def pretty_time(text: str) -> str:
    """The server's UTC time ("2026-10-16 18:40:02.858Z") as local "2026-10-16 11:40". Unreadable text comes back as is."""
    try:
        moment = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return text
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def default_folder() -> Path:
    """Where downloaded models go: Documents/Chronovox Workspace."""
    return Path.home() / "Documents" / "Chronovox Workspace"


def canonical(content: bytes) -> bytes:
    """The bytes the editor itself would write for this model: compressed, without unused colors.

    Two files with the same model in them give the same bytes, whoever wrote them, so "has anything changed" is a
    question about the model and not about how the file happened to be encoded. Anything that is not a .cvox file
    is returned as it is.
    """
    try:
        return cvox.dumps(cvox.loads(content).opaque().compacted(), compress=True)
    except (cvox.CvoxError, ValueError):
        return content


def file_sha(path: Path) -> str | None:
    """Checksum of the model in a file, by its canonical bytes."""
    try:
        return sha256_hex(canonical(Path(path).read_bytes()))
    except OSError:
        return None


def choose_download_path(folder: Path, model_name: str, number: int, is_latest: bool, links: LinkStore, server: str,
                         model_id: str) -> tuple[Path, bool]:
    """Where to put a downloaded version, as (path, already_there).

    Nothing is ever overwritten, so work in progress cannot be lost. The newest version is called ``Name.cvox`` and any
    other ``Name (v2).cvox``. If that file exists already it is used as it is when it is this very version and nobody
    has changed it since (``already_there``), and otherwise the next free name is taken: ``Name (v3).cvox``,
    ``Name (v3) (2).cvox`` and so on.
    """
    base = safe_name(model_name)
    names = [f"{base}.cvox"] if is_latest else []
    names.append(f"{base} (v{number}).cvox")
    names += [f"{base} (v{number}) ({n}).cvox" for n in range(2, 1000)]
    for name in names:
        path = Path(folder) / name
        if not path.exists():
            return path, False
        link = links.get(path, server)
        if link is not None and link.model_id == model_id and link.version == number and link.sha256 == file_sha(path):
            return path, True
    raise OSError("too many copies of this model in the folder")


def versions_to_drop(numbers: list[int], limit: int | None) -> list[int]:
    """Which version numbers the server removes when one more is uploaded and it keeps only ``limit`` per model.

    ``numbers`` are the versions that exist now, in any order. The oldest go first. Nothing is dropped without a limit.
    """
    if not limit:
        return []
    excess = len(numbers) + 1 - limit
    return sorted(numbers)[:excess] if excess > 0 else []


def link_status(link: Link | None, current_sha: str | None, latest: int | None) -> str:
    """One line about a file's link: whether it is up to date, changed here, or behind the server."""
    if link is None:
        return "Not shared yet"
    parts = [f"Linked to {link.model_name}, version {link.version}"]
    changed = current_sha is not None and current_sha != link.sha256
    behind = latest is not None and latest > link.version
    if changed:
        parts.append("changed here since then")
    if behind:
        parts.append(f"version {latest} is on the server")
    if not changed and not behind and latest is not None:
        parts.append("up to date")
    return ", ".join(parts)
