"""The folder tree behind the Files panel: the .cvox files under some folders. Plain data: no GL or UI imports."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

MAX_FILES = 5000     # a scan stops after this many models, so a huge folder cannot stall the editor
MAX_DEPTH = 12
MODEL_EXTENSIONS = (".cvox",)
LEGACY_EXTENSIONS = (".csm",)     # Chronovox Studio models: opened as an import


@dataclass
class Node:
    name: str
    path: str
    is_dir: bool
    children: list["Node"] = field(default_factory=list)
    files: int = 0           # how many models are in this folder and below it


def norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def scan(root: str, max_files: int = MAX_FILES, max_depth: int = MAX_DEPTH,
         extensions: tuple[str, ...] = MODEL_EXTENSIONS) -> Node:
    """The tree under ``root``: folders first, then files, each sorted by name. Folders without models are left out.

    Only files ending in one of ``extensions`` count as models.

    Hidden folders (starting with a dot) and links to folders are skipped. A missing root gives an empty node.
    """
    budget = [max_files]

    def walk(path: str, name: str, depth: int) -> Node:
        node = Node(name, path, True)
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name.lower())
        except OSError:
            return node
        folders, files = [], []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if depth < max_depth and not entry.name.startswith("."):
                        folders.append(entry)
                elif entry.name.lower().endswith(extensions) and entry.is_file():
                    files.append(entry)
            except OSError:
                continue
        for entry in folders:
            if budget[0] <= 0:
                break
            child = walk(entry.path, entry.name, depth + 1)
            if child.files:
                node.children.append(child)
        for entry in files:
            if budget[0] <= 0:
                break
            budget[0] -= 1
            node.children.append(Node(entry.name, entry.path, False, files=1))
        node.files = sum(c.files for c in node.children)
        return node

    root = os.path.normpath(root)
    return walk(root, os.path.basename(root) or root, 0)


def filtered(node: Node, text: str) -> Node | None:
    """A copy of the tree with only the models whose names contain ``text`` (any case), or None if there are none.

    A folder whose own name matches keeps everything inside it.
    """
    text = text.strip().lower()
    if not text:
        return node
    if not node.is_dir:
        return node if text in node.name.lower() else None
    if text in node.name.lower():
        return node
    kept = [c for c in (filtered(child, text) for child in node.children) if c is not None]
    if not kept:
        return None
    return Node(node.name, node.path, True, kept, sum(c.files for c in kept))


def route_to(node: Node, key: str) -> list[Node] | None:
    """The nodes from ``node`` down to the one whose normalized path is ``key`` (both ends included), or None."""
    if norm(node.path) == key:
        return [node]
    for child in node.children:
        found = route_to(child, key)
        if found is not None:
            return [node] + found
    return None


def folders(node: Node) -> list[str]:
    """The paths of a folder and every folder under it."""
    if not node.is_dir:
        return []
    return [node.path] + [path for child in node.children for path in folders(child)]
