"""Undo/redo for voxel edits. No GL, UI or file imports.

A voxel edit stores only the voxels that changed (flat index, old value, new value), so a
big matrix with a small edit costs a few bytes, not a copy of the grid. A stroke or
a box fill is one edit. Palette colors added by an edit are kept when it is undone:
voxel indices stay valid, and an unused color is harmless.

Matrix edits (move, rotate, resize) are entries too. History is a strict stack, so a
voxel edit made before a resize is only undone after the resize has been undone,
and its indices always refer to the grid shape that was current when it was made.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Matrix, Model

DEFAULT_MAX_BYTES = 256 * 1024 * 1024


@dataclass
class VoxelEdit:
    matrix: Matrix
    indices: np.ndarray  # flat C-order indices into matrix.voxels
    old: np.ndarray
    new: np.ndarray

    @classmethod
    def between(cls, matrix: Matrix, before: np.ndarray) -> "VoxelEdit | None":
        """The edit that turned ``before`` (a copy of the grid) into the matrix's voxels now."""
        after = matrix.voxels.reshape(-1)
        previous = before.reshape(-1)
        changed = np.flatnonzero(after != previous)
        if not len(changed):
            return None
        index_type = np.uint32 if after.size < 2 ** 32 else np.uint64
        return cls(matrix, changed.astype(index_type), previous[changed].copy(), after[changed].copy())

    @property
    def nbytes(self) -> int:
        return self.indices.nbytes + self.old.nbytes + self.new.nbytes

    def undo(self) -> None:
        self.matrix.voxels.flat[self.indices] = self.old

    def redo(self) -> None:
        self.matrix.voxels.flat[self.indices] = self.new

    def describe(self) -> str:
        return f"{len(self.indices):,} voxel change(s) in {self.matrix.name}"


@dataclass
class SelectionEdit:
    """A change to a matrix's voxels together with the selection that went with it (a move, flip, rotate or paste).

    Undoing puts the voxels and the selection back as they were. The selection is kept as the flat indices of its
    selected cells, which stays small however big the matrix is.
    """
    edit: "VoxelEdit | None"
    state: object                 # the VoxelSelection that holds the current selection
    matrix: Matrix
    old: np.ndarray               # selected cells before
    new: np.ndarray               # and after
    label: str
    floating: bool = False        # a move, flip or turn of the same selection: undoing it keeps it floating over what is beneath

    @property
    def nbytes(self) -> int:
        return (self.edit.nbytes if self.edit else 0) + self.old.nbytes + self.new.nbytes

    def _restore(self, indices: np.ndarray) -> None:
        from .selection import mask_from
        # After a move the same selection is back, so what it floats over is still right. A paste or a delete changes
        # which voxels are selected, so undoing it ends the floating.
        self.state.set(self.matrix, mask_from(indices, self.matrix.voxels.shape) if len(indices) else None,
                       keep_under=self.floating)

    def undo(self) -> None:
        if self.edit is not None:
            self.edit.undo()
        self._restore(self.old)

    def redo(self) -> None:
        if self.edit is not None:
            self.edit.redo()
        self._restore(self.new)

    def describe(self) -> str:
        return self.label


@dataclass
class TransformEdit:
    """A change of a matrix's position, pivot or rotation."""
    matrix: Matrix
    old: tuple  # (position, pivot, rotation, resolution)
    new: tuple

    @property
    def nbytes(self) -> int:
        return 200

    def _set(self, state: tuple) -> None:
        self.matrix.position, self.matrix.pivot, self.matrix.rotation = state[:3]
        if len(state) > 3:
            self.matrix.resolution = state[3]

    def undo(self) -> None:
        self._set(self.old)

    def redo(self) -> None:
        self._set(self.new)

    def describe(self) -> str:
        return f"transform of {self.matrix.name}"


@dataclass
class MatrixListEdit:
    """A matrix added to, or removed from, the model (clone, add and delete).

    The matrix object itself is kept, so voxel and transform edits made on it earlier in the
    history are still valid when this one is undone. ``added`` says which way the edit goes
    when it is done (and redone): True puts the matrix in at ``index``, False takes it out.
    """
    model: Model
    matrix: Matrix
    index: int
    added: bool

    @property
    def nbytes(self) -> int:
        in_model = any(m is self.matrix for m in self.model.matrices)
        return 0 if in_model else self.matrix.voxels.nbytes     # a removed matrix is kept alive only by this entry

    def _put_in(self) -> None:
        self.model.insert_matrix(self.index, self.matrix)

    def _take_out(self) -> None:
        self.index = self.model.remove_matrix(self.matrix)

    def undo(self) -> None:
        self._take_out() if self.added else self._put_in()

    def redo(self) -> None:
        self._put_in() if self.added else self._take_out()

    def describe(self) -> str:
        return f"{'adding' if self.added else 'deleting'} {self.matrix.name}"


@dataclass
class MatrixMoveEdit:
    """A matrix moved from one place in the model's list to another (the order they are saved in)."""
    model: Model
    matrix: Matrix
    old_index: int
    new_index: int

    nbytes = 100

    def undo(self) -> None:
        self.model.move_matrix(self.matrix, self.old_index)

    def redo(self) -> None:
        self.model.move_matrix(self.matrix, self.new_index)

    def describe(self) -> str:
        return f"moving {self.matrix.name} to position {self.new_index + 1}"


@dataclass
class MergeEdit:
    """One matrix merged into another: the target grew to hold it, and the source left the model.

    Keeps both of the target's grids and the source matrix itself, so either direction is exact and
    earlier edits to either matrix are still valid when this one is undone.
    """
    model: Model
    target: Matrix
    source: Matrix
    source_index: int
    old_voxels: np.ndarray
    new_voxels: np.ndarray
    old: tuple  # (position, pivot) of the target before
    new: tuple  # and after

    @property
    def matrix(self) -> Matrix:
        return self.target

    @property
    def nbytes(self) -> int:
        in_model = any(m is self.source for m in self.model.matrices)
        return self.old_voxels.nbytes + self.new_voxels.nbytes + (0 if in_model else self.source.voxels.nbytes)

    def undo(self) -> None:
        self.target.voxels = self.old_voxels
        self.target.position, self.target.pivot = self.old
        self.model.insert_matrix(self.source_index, self.source)

    def redo(self) -> None:
        self.target.voxels = self.new_voxels
        self.target.position, self.target.pivot = self.new
        self.model.remove_matrix(self.source)

    def describe(self) -> str:
        return f"merging {self.source.name} into {self.target.name}"


@dataclass
class GroupEdit:
    """Several edits undone and redone together as one step (for example a change to every matrix)."""
    label: str
    edits: list

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self.edits)

    def undo(self) -> None:
        for edit in reversed(self.edits):
            edit.undo()

    def redo(self) -> None:
        for edit in self.edits:
            edit.redo()

    def describe(self) -> str:
        return self.label


@dataclass
class ResizeEdit:
    """Cells added to or cropped from a matrix. Keeps both grids so either direction is exact."""
    matrix: Matrix
    old_voxels: np.ndarray
    new_voxels: np.ndarray
    old: tuple  # (position, pivot)
    new: tuple

    @property
    def nbytes(self) -> int:
        return self.old_voxels.nbytes + self.new_voxels.nbytes

    def _set(self, voxels: np.ndarray, state: tuple) -> None:
        self.matrix.voxels = voxels
        self.matrix.position, self.matrix.pivot = state

    def undo(self) -> None:
        self._set(self.old_voxels, self.old)

    def redo(self) -> None:
        self._set(self.new_voxels, self.new)

    def describe(self) -> str:
        x, y, z = self.matrix.size
        return f"size of {self.matrix.name} ({x}x{y}x{z})"


@dataclass
class ReorientEdit(ResizeEdit):
    """A matrix's voxels turned or flipped inside it. It changes the grid and the position and pivot, like a resize."""
    label: str = "reorientation"

    def describe(self) -> str:
        return f"{self.label} of {self.matrix.name}"


class History:
    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.max_bytes = max_bytes
        self._undo: list = []
        self._redo: list = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def push(self, edit) -> None:
        """Record an edit that has already been applied. Drops anything that could be redone."""
        self._redo.clear()
        self._undo.append(edit)
        # Forget the oldest edits when over budget, but always keep the latest one.
        while len(self._undo) > 1 and sum(e.nbytes for e in self._undo) > self.max_bytes:
            self._undo.pop(0)

    def undo(self):
        """Revert the latest edit and return it, or None if there is nothing to undo."""
        if not self._undo:
            return None
        edit = self._undo.pop()
        edit.undo()
        self._redo.append(edit)
        return edit

    def redo(self):
        """Re-apply the most recently undone edit and return it, or None."""
        if not self._redo:
            return None
        edit = self._redo.pop()
        edit.redo()
        self._undo.append(edit)
        return edit
