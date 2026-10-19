"""What the editor remembers about the workspace between runs: the saved login and which local file is which shared
model. Plain JSON files next to the preferences. No GL, UI or network imports.

The password is never stored, only the login token the server hands out (it stops working after a while, and the
program asks for the password again). Both files are forgiving like the preferences: damaged or missing means empty.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


def session_path(folder: Path) -> Path:
    return Path(folder) / "workspace_session.json"


def links_path(folder: Path) -> Path:
    return Path(folder) / "workspace_links.json"


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(path: Path, data) -> bool:
    """Replace the file in one step, so a crash cannot leave half of it."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
            json.dump(data, handle, indent=2)
            temp = Path(handle.name)
        os.replace(temp, path)
        return True
    except OSError:
        return False


@dataclass
class Session:
    url: str
    token: str
    user_id: str
    email: str
    name: str = ""


def load_session(path: Path | None) -> Session | None:
    """The saved login, or None. ``path`` None means nothing is kept on disk."""
    data = _read(path) if path else None
    if not isinstance(data, dict):
        return None
    try:
        values = {k: data[k] for k in ("url", "token", "user_id", "email")}
    except KeyError:
        return None
    if not all(isinstance(v, str) and v for v in values.values()):
        return None
    name = data.get("name")
    return Session(**values, name=name if isinstance(name, str) else "")


def save_session(session: Session, path: Path | None) -> bool:
    return bool(path) and _write(path, asdict(session))


def clear_session(path: Path | None) -> None:
    try:
        if path:
            Path(path).unlink()
    except OSError:
        pass


@dataclass
class Link:
    """A local file that came from, or was uploaded as, a version of a shared model."""
    server: str          # the server address it belongs to
    model_id: str
    model_name: str
    version: int         # the version this file was last downloaded as, or uploaded as
    sha256: str          # what the file looked like then, to notice later changes


def _key(path) -> str:
    return os.path.normcase(os.path.abspath(path))


class LinkStore:
    """The links, kept in a JSON file. With ``path`` None they only live as long as the program runs."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.links: dict[str, Link] = {}
        data = _read(self.path) if self.path else None
        if isinstance(data, dict):
            for key, value in data.items():
                try:
                    self.links[key] = Link(**value)
                except TypeError:
                    continue

    def _save(self) -> None:
        if self.path:
            _write(self.path, {k: asdict(v) for k, v in self.links.items()})

    def get(self, file_path, server: str) -> Link | None:
        """The link of a local file, if it belongs to this server."""
        link = self.links.get(_key(file_path))
        return link if link is not None and link.server == server else None

    def set(self, file_path, link: Link) -> None:
        self.links[_key(file_path)] = link
        self._save()

    def remove(self, file_path) -> None:
        if self.links.pop(_key(file_path), None) is not None:
            self._save()
