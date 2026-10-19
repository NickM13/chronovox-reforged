"""In-memory model. No file, rendering or UI imports belong here."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .naming import variant_name

# Positions and pivots are int32 in units of 1/128 voxel.
POSITION_SCALE = 128

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1

MAX_PALETTE = 65535
MAX_MATRICES = 65535
MAX_RESOLUTION = 8             # resolutions run 1..8: a cell is 1, 1/2, 1/4 ... 1/128 unit across, the finest a position can address
MAX_MATRIX_SIZE = 65535        # per axis, a limit of the .cvox format
MAX_MATRIX_VOXELS = 2 ** 26    # 64M voxels (128 MB of indices), a sanity limit for the editor
MAX_NAME_BYTES = 255

IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)


@dataclass
class Matrix:
    """One independent voxel grid.

    ``voxels`` has shape (size_z, size_y, size_x) and holds palette indices
    (0 = empty). ``position`` is (x, y, z) in 1/128 voxel and ``pivot`` is
    measured from the matrix origin corner in 1/128 cell. ``rotation`` is a unit
    quaternion (x, y, z, w). ``resolution`` says how fine the cells are: a cell is
    ``0.5 ** (resolution - 1)`` units across, so resolution 1 is the normal size and
    each step up halves it. A matrix is placed by ``position + R * f * (p - pivot) + f * pivot``
    for a local point p in cells, f being the cell size.
    """

    name: str
    voxels: np.ndarray
    position: tuple[int, int, int] = (0, 0, 0)
    pivot: tuple[int, int, int] | None = None
    rotation: tuple[float, float, float, float] = IDENTITY_ROTATION
    resolution: int = 1

    @property
    def cell_size(self) -> float:
        """How many units across a cell is: 1, 1/2, 1/4, 1/8 ... for resolutions 1, 2, 3, 4 ... (exact in floats)."""
        return 0.5 ** (self.resolution - 1)

    @property
    def cell_size_128(self) -> int:
        """The same in 1/128 units, the unit of ``position``: 128, 64, 32 ... a whole number up to resolution 8."""
        return POSITION_SCALE >> (self.resolution - 1)

    def __post_init__(self) -> None:
        if self.pivot is None:
            self.pivot = self.center_pivot()

    @property
    def size(self) -> tuple[int, int, int]:
        """Grid size as (x, y, z)."""
        z, y, x = self.voxels.shape
        return x, y, z

    def center_pivot(self) -> tuple[int, int, int]:
        """Matrix center in 1/128 units, i.e. size * 64 per axis."""
        sx, sy, sz = self.size
        half = POSITION_SCALE // 2
        return sx * half, sy * half, sz * half


@dataclass
class Model:
    """A set of matrices sharing one palette.

    ``palette`` is an (N, 4) uint8 array of RGBA. Row k is palette index k + 1;
    index 0 is reserved for empty and is not stored.
    """

    palette: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.uint8))
    matrices: list[Matrix] = field(default_factory=list)

    def used_palette_mask(self) -> np.ndarray:
        """Which palette entries any voxel in any matrix uses, as a bool array with one entry per color."""
        count = len(self.palette)
        uses = np.zeros(count + 1, np.int64)
        for matrix in self.matrices:
            uses += np.bincount(matrix.voxels.ravel(), minlength=count + 1)[:count + 1]
        return uses[1:] > 0

    def compacted(self) -> "Model":
        """A model without the palette colors nothing uses, for saving. Returns ``self`` if all are used.

        The colors that are used keep their order and the voxels are renumbered to match, so every voxel
        keeps its color. This model is left as it is: unused colors stay available for editing.
        """
        used = self.used_palette_mask()
        if used.all():
            return self
        lookup = np.zeros(len(self.palette) + 1, np.uint16)       # old index -> new index; 0 stays empty
        lookup[1:][used] = np.arange(1, int(used.sum()) + 1, dtype=np.uint16)
        matrices = [Matrix(m.name, lookup[m.voxels], m.position, m.pivot, m.rotation, m.resolution)
                    for m in self.matrices]
        return Model(self.palette[used].copy(), matrices)

    def opaque(self) -> "Model":
        """A model whose palette colors all have alpha 255, with colors that were the same but for alpha merged.

        Alpha is not used by the editor: a voxel is either there or not. Old files carry values such as 1 that
        only meant "solid". Returns ``self`` if every color is already opaque. Colors keep their order
        of first appearance and every voxel keeps its color.
        """
        if not len(self.palette) or (self.palette[:, 3] == 255).all():
            return self
        rgb = self.palette.copy()
        rgb[:, 3] = 255
        key = (rgb[:, 0].astype(np.uint32) | (rgb[:, 1].astype(np.uint32) << 8)
               | (rgb[:, 2].astype(np.uint32) << 16))
        _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        by_appearance = np.argsort(first)
        rank = np.empty(len(first), np.int64)
        rank[by_appearance] = np.arange(len(first))
        lookup = np.zeros(len(self.palette) + 1, np.uint16)       # old index -> new index; 0 stays empty
        lookup[1:] = rank[inverse] + 1
        matrices = [Matrix(m.name, lookup[m.voxels], m.position, m.pivot, m.rotation, m.resolution)
                    for m in self.matrices]
        return Model(rgb[first[by_appearance]], matrices)

    def move_matrix(self, matrix: Matrix, new_index: int) -> int:
        """Move a matrix to another place in the list (clamped to the ends). Returns the place it came from."""
        old_index = self.remove_matrix(matrix)
        self.insert_matrix(new_index, matrix)
        return old_index

    def find_color(self, rgba: tuple[int, int, int, int]) -> int:
        """Palette index of a color, or 0 if it is not in the palette."""
        matches = np.flatnonzero((self.palette == np.array(rgba, np.uint8)).all(axis=1))
        return int(matches[0]) + 1 if len(matches) else 0

    def index_for_color(self, rgba: tuple[int, int, int, int]) -> int:
        """Palette index of a color, appending it to the palette if it is new."""
        index = self.find_color(rgba)
        if index:
            return index
        if len(self.palette) >= MAX_PALETTE:
            raise ValueError(f"palette is full ({MAX_PALETTE} colors)")
        self.palette = np.vstack((self.palette.reshape(-1, 4), np.array(rgba, np.uint8)))
        return len(self.palette)

    def unique_matrix_name(self, base: str = "Matrix") -> str:
        """``base`` if it is free, otherwise ``base2``, ``base3`` and so on (no spaces)."""
        taken = {m.name for m in self.matrices}
        if base not in taken:
            return base
        n = 2
        while f"{base}{n}" in taken:
            n += 1
        return f"{base}{n}"

    def _checked_name(self, name: str, renaming: Matrix | None = None) -> str:
        """Strip and validate a matrix name; ValueError if it cannot be used."""
        name = name.strip()
        if not name:
            raise ValueError("a matrix needs a name")
        if not name.isascii():
            raise ValueError("matrix names must be ASCII")
        if len(name.encode("ascii")) > MAX_NAME_BYTES:
            raise ValueError(f"matrix names can be at most {MAX_NAME_BYTES} characters")
        if any(m.name == name and m is not renaming for m in self.matrices):
            raise ValueError(f"there is already a matrix called {name!r}")
        return name

    def rename_matrix(self, matrix: Matrix, new_name: str) -> bool:
        """Rename a matrix. Returns False if the (stripped) name is unchanged.

        Raises ValueError, leaving the matrix as it was, if the name is not allowed.
        """
        name = self._checked_name(new_name, renaming=matrix)
        if name == matrix.name:
            return False
        matrix.name = name
        return True

    def new_matrix(self, size: tuple[int, int, int], name: str | None = None,
                   position: tuple[int, int, int] = (0, 0, 0), resolution: int = 1) -> Matrix:
        """Append an empty matrix of ``size`` (x, y, z) cells at a resolution (1 to 8) and return it.

        An empty or missing name is replaced by a free "Matrix", "Matrix2", ... Raises
        ValueError for a bad size, a name that is not allowed, or too many matrices.
        """
        if len(self.matrices) >= MAX_MATRICES:
            raise ValueError(f"a model can hold at most {MAX_MATRICES} matrices")
        if not 1 <= int(resolution) <= MAX_RESOLUTION:
            raise ValueError(f"resolution must be 1 to {MAX_RESOLUTION}")
        sx, sy, sz = (int(v) for v in size)
        if not all(1 <= v <= MAX_MATRIX_SIZE for v in (sx, sy, sz)):
            raise ValueError(f"matrix size must be 1 to {MAX_MATRIX_SIZE} on every axis")
        if sx * sy * sz > MAX_MATRIX_VOXELS:
            raise ValueError(f"a matrix can hold at most {MAX_MATRIX_VOXELS:,} voxels")
        name = self._checked_name(name) if (name or "").strip() else self.unique_matrix_name()
        matrix = Matrix(name, np.zeros((sz, sy, sx), np.uint16), tuple(position), resolution=int(resolution))
        self.matrices.append(matrix)
        return matrix

    def insert_matrix(self, index: int, matrix: Matrix) -> None:
        """Put an existing matrix into the model at a position in the list."""
        self.matrices.insert(max(0, min(index, len(self.matrices))), matrix)

    def remove_matrix(self, matrix: Matrix) -> int:
        """Take a matrix out of the model (by identity) and return the position it had."""
        for index, existing in enumerate(self.matrices):
            if existing is matrix:
                del self.matrices[index]
                return index
        raise ValueError("that matrix is not in the model")

    def duplicate_matrix(self, matrix: Matrix) -> tuple[Matrix, int]:
        """Insert an independent copy of a matrix right after it and return (copy, its index).

        The copy has the same voxels, position, pivot, rotation and scale, and a free name that keeps the
        original's game suffix (``Arm`` -> ``Arm2``, ``Torso-box`` -> ``Torso2-box``).
        Raises ValueError if the model is full.
        """
        if len(self.matrices) >= MAX_MATRICES:
            raise ValueError(f"a model can hold at most {MAX_MATRICES} matrices")
        position = next((i for i, m in enumerate(self.matrices) if m is matrix), None)
        if position is None:
            raise ValueError("that matrix is not in the model")
        index = position + 1
        taken = {m.name for m in self.matrices}
        copy = Matrix(variant_name(matrix.name, taken), matrix.voxels.copy(), matrix.position,
                      matrix.pivot, matrix.rotation, matrix.resolution)
        self.insert_matrix(index, copy)
        return copy, index
