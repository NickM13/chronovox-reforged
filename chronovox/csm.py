"""Importer for legacy .csm (Chronovox Studio Model) version 4 files. Read-only; never writes .csm.

Pure file I/O: no rendering or UI imports. Run ``python -m chronovox.csm in.csm [out.cvox]`` to
convert a file and compare sizes.

Layout (little-endian): ``.CSM`` | int32 version (4) | int32 unused | int32 matrix count, then per
matrix: name and parent (each an int32 length plus that many characters), position and size
(int16 x 3 each; position is in 1/10 voxel), then runs of 5 bytes (count 1-255, R, G, B, A) until
the counts add up to the volume. Voxels run z fastest, then y, then x.

Conversion to the editor's model: the old loader mirrors X, so voxels are flipped along x and the
matrix position becomes ``-position_x - size_x`` (in voxels). Positions are converted to 1/128
voxel, the parent is dropped, rotation is identity and the pivot is the matrix center. One
shared palette is built from every distinct color. Alpha only ever meant "solid" (0 = no voxel), so
any non-zero alpha becomes fully opaque and two colors that differ only in alpha are the same color.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .model import MAX_MATRICES, MAX_NAME_BYTES, MAX_PALETTE, POSITION_SCALE, Matrix, Model

MAGIC = b".CSM"
SUPPORTED_VERSION = 4
RUN_BYTES = 5


class CsmError(ValueError):
    """Raised when a .csm file cannot be imported."""


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise CsmError("unexpected end of file")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def string(self, what: str, strict: bool) -> str:
        (length,) = self.unpack("<i")
        if length < 0:
            raise CsmError(f"negative {what} length")
        raw = self.take(length)
        if not strict:
            return ""  # the parent is read only to skip past it
        if not raw.isascii():
            raise CsmError(f"{what} {raw!r} is not ASCII")
        if length > MAX_NAME_BYTES:
            raise CsmError(f"{what} is longer than {MAX_NAME_BYTES} characters")
        return raw.decode("ascii")


def _position_to_cvox(position_tenths: int, size: int, axis_is_x: bool) -> int:
    """Convert a .csm position (1/10 voxel) to 1/128 voxel, mirroring X first.

    All arithmetic is exact integers, rounding half up, so the result is within 1/256 voxel.
    """
    if axis_is_x:
        position_tenths = -position_tenths - 10 * size      # mirror: -position_x - size_x, in tenths
    return (position_tenths * POSITION_SCALE + 5) // 10


def _read_runs(reader: _Reader, volume: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read RLE runs until they cover ``volume`` voxels. Returns (counts, rgba) with rgba shaped (runs, 4)."""
    available = (len(reader.data) - reader.pos) // RUN_BYTES
    rows = min(available, volume)               # a run covers at least one voxel
    block = np.frombuffer(reader.data, np.uint8, rows * RUN_BYTES, reader.pos).reshape(rows, RUN_BYTES)
    covered = np.cumsum(block[:, 0], dtype=np.int64)
    if rows == 0 or covered[-1] < volume:
        raise CsmError(f"matrix {name!r}: voxel data ends early")
    last = int(np.searchsorted(covered, volume))  # first run that reaches the volume
    if covered[last] != volume:
        raise CsmError(f"matrix {name!r}: run counts overshoot the matrix volume")
    runs = block[:last + 1]
    if (runs[:, 0] == 0).any():
        raise CsmError(f"matrix {name!r}: a run has a count of zero")
    reader.pos += (last + 1) * RUN_BYTES
    return runs[:, 0].astype(np.int64), runs[:, 1:5]


def loads(data: bytes) -> Model:
    """Parse .csm version 4 bytes into a Model, with one shared palette."""
    reader = _Reader(data)
    if reader.take(4) != MAGIC:
        raise CsmError("not a .csm file (bad format type)")
    version, _unused, count = reader.unpack("<3i")
    if version != SUPPORTED_VERSION:
        raise CsmError(f"only .csm version {SUPPORTED_VERSION} can be imported, this is version {version}")
    if not 0 <= count <= MAX_MATRICES:
        raise CsmError(f"bad matrix count {count}")

    parsed = []   # (name, position tenths, size, counts, packed colors)
    for _ in range(count):
        name = reader.string("matrix name", strict=True)
        reader.string("matrix parent", strict=False)
        position = reader.unpack("<3h")
        size = reader.unpack("<3h")
        if min(size) < 1:
            raise CsmError(f"matrix {name!r}: size {size} must be at least 1 on every axis")
        volume = size[0] * size[1] * size[2]
        counts, rgba = _read_runs(reader, volume, name)
        packed = (rgba[:, 0].astype(np.uint32) | (rgba[:, 1].astype(np.uint32) << 8)
                  | (rgba[:, 2].astype(np.uint32) << 16) | np.uint32(255 << 24))   # solid colors are opaque
        packed[rgba[:, 3] == 0] = 0                   # alpha 0 is "no voxel", whatever its RGB says
        parsed.append((name, position, size, counts, packed))

    # One palette from every distinct color, in order of first appearance. Index 0 stays "empty".
    all_runs = np.concatenate([p[4] for p in parsed]) if parsed else np.zeros(0, np.uint32)
    colors, first_seen = np.unique(all_runs[all_runs != 0], return_index=True)
    colors = colors[np.argsort(first_seen)]
    if len(colors) > MAX_PALETTE:
        raise CsmError(f"this model has {len(colors):,} distinct colors; the limit is {MAX_PALETTE:,}")
    palette = np.stack([(colors >> shift) & 0xFF for shift in (0, 8, 16, 24)], axis=1).astype(np.uint8)
    order = np.argsort(colors)
    sorted_colors = colors[order]

    matrices = []
    for name, position, (sx, sy, sz), counts, packed in parsed:
        # index into the palette: 0 for empty, else 1 + position of the color in `colors`
        index = np.zeros(len(packed), np.uint16)
        solid = packed != 0
        index[solid] = order[np.searchsorted(sorted_colors, packed[solid])] + 1
        grid = np.repeat(index, counts).reshape(sx, sy, sz)       # [x, y, z]: z varies fastest in the file
        voxels = np.ascontiguousarray(grid[::-1].transpose(2, 1, 0))   # mirror x, then to [z, y, x]
        cvox_position = (_position_to_cvox(position[0], sx, True),
                         _position_to_cvox(position[1], sy, False),
                         _position_to_cvox(position[2], sz, False))
        matrices.append(Matrix(name, voxels, cvox_position))       # identity rotation, centered pivot
    return Model(palette, matrices)


def load(path: str | Path) -> Model:
    return loads(Path(path).read_bytes())


def main() -> None:
    import argparse

    from . import cvox

    parser = argparse.ArgumentParser(prog="python -m chronovox.csm",
                                     description="Convert a .csm (v4) file to .cvox and compare sizes.")
    parser.add_argument("input", help=".csm file to read")
    parser.add_argument("output", nargs="?", help=".cvox file to write (default: next to the input)")
    args = parser.parse_args()

    source = Path(args.input)
    try:
        model = load(source)
    except (OSError, CsmError) as exc:
        raise SystemExit(f"Could not import {source}: {exc}")
    target = Path(args.output) if args.output else source.with_suffix(".cvox")
    cvox.save(model, target, compress=True)
    plain = len(cvox.dumps(model, compress=False))

    voxels = sum(int((m.voxels > 0).sum()) for m in model.matrices)
    print(f"{len(model.matrices)} matrices, {len(model.palette):,} colors, {voxels:,} solid voxels")
    for m in model.matrices:
        x, y, z = m.size
        print(f"  {m.name or '(unnamed)'}  {x}x{y}x{z}  at "
              f"({m.position[0] / POSITION_SCALE:g}, {m.position[1] / POSITION_SCALE:g}, {m.position[2] / POSITION_SCALE:g})")
    original = source.stat().st_size
    compressed = target.stat().st_size
    print(f".csm {original:,} bytes | .cvox {plain:,} bytes | .cvox compressed {compressed:,} bytes -> {target}")


if __name__ == "__main__":
    main()
