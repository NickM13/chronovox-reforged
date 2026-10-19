"""Reader and writer for the .cvox format, version 1.

Pure file I/O: no rendering or UI imports.
"""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np

from .model import MAX_RESOLUTION, Matrix, Model

MAGIC = b"CVOX"
VERSION = 1
FLAG_ZLIB = 0x0001
KNOWN_FLAGS = FLAG_ZLIB

MAX_PALETTE = 65535
MAX_MATRICES = 65535
MAX_NAME_BYTES = 255
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


class CvoxError(ValueError):
    """Raised for invalid .cvox data, on read or on write."""


# --- varint (unsigned LEB128) -------------------------------------------------

def _encode_varint(value: int, out: bytearray) -> None:
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


# --- voxel data ---------------------------------------------------------------

def encode_voxels(voxels: np.ndarray, index_width: int) -> bytes:
    """Run-length encode a (z, y, x) index array, merging equal neighbours."""
    flat = np.ascontiguousarray(voxels).ravel()
    if flat.size == 0:
        return b""
    starts = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1))
    counts = np.diff(np.concatenate((starts, [flat.size])))
    indices = flat[starts]

    out = bytearray()
    for count, index in zip(counts.tolist(), indices.tolist()):
        _encode_varint(count, out)
        out += index.to_bytes(index_width, "little")
    return bytes(out)


def decode_voxels(data: bytes, shape: tuple[int, int, int], palette_count: int,
                  index_width: int) -> np.ndarray:
    """Decode RLE data into a (z, y, x) uint16 array, validating as it goes."""
    total = shape[0] * shape[1] * shape[2]
    counts: list[int] = []
    indices: list[int] = []
    pos, end, summed = 0, len(data), 0
    while pos < end:
        count, shift = 0, 0
        while True:
            if pos >= end:
                raise CvoxError("truncated varint in voxel data")
            byte = data[pos]
            pos += 1
            count |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
            if shift > 63:
                raise CvoxError("varint too long in voxel data")
        if pos + index_width > end:
            raise CvoxError("truncated run in voxel data")
        index = int.from_bytes(data[pos:pos + index_width], "little")
        pos += index_width
        if count < 1:
            raise CvoxError("run count must be at least 1")
        if index > palette_count:
            raise CvoxError(f"palette index {index} exceeds palette count {palette_count}")
        summed += count
        if summed > total:
            raise CvoxError("run counts exceed matrix volume")
        counts.append(count)
        indices.append(index)
    if summed != total:
        raise CvoxError(f"run counts sum to {summed}, expected {total}")
    flat = np.repeat(np.array(indices, dtype=np.uint16), counts)
    return flat.reshape(shape)


# --- writing ------------------------------------------------------------------

def _pack_string(text: str) -> bytes:
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError:
        raise CvoxError(f"name {text!r} is not ASCII") from None
    if len(raw) > MAX_NAME_BYTES:
        raise CvoxError(f"name {text!r} exceeds {MAX_NAME_BYTES} bytes")
    return bytes([len(raw)]) + raw


def _pack_int32s(values, what: str) -> bytes:
    values = tuple(values)
    if len(values) != 3 or any(int(v) != v for v in values):
        raise CvoxError(f"{what} must be three integers (1/128 units)")
    if any(not INT32_MIN <= v <= INT32_MAX for v in values):
        raise CvoxError(f"{what} out of int32 range")
    return struct.pack("<3i", *(int(v) for v in values))


def dumps(model: Model, compress: bool = False) -> bytes:
    """Serialize a model to .cvox bytes."""
    palette = np.asarray(model.palette)
    if palette.size == 0:
        palette = palette.reshape(0, 4)
    if palette.ndim != 2 or palette.shape[1] != 4:
        raise CvoxError("palette must have shape (N, 4)")
    n = palette.shape[0]
    if n > MAX_PALETTE:
        raise CvoxError(f"palette has {n} colors, maximum is {MAX_PALETTE}")
    if len(model.matrices) > MAX_MATRICES:
        raise CvoxError(f"too many matrices (maximum {MAX_MATRICES})")
    if palette.size and (palette.min() < 0 or palette.max() > 255):
        raise CvoxError("palette values must be 0-255")
    index_width = 1 if n <= 255 else 2

    out = bytearray(MAGIC)
    out += struct.pack("<HHH", VERSION, FLAG_ZLIB if compress else 0, n)
    out += palette.astype(np.uint8).tobytes()
    out += struct.pack("<H", len(model.matrices))

    for matrix in model.matrices:
        vox = matrix.voxels
        if vox.ndim != 3 or min(vox.shape) < 1 or max(vox.shape) > 65535:
            raise CvoxError(f"matrix {matrix.name!r}: size must be 1-65535 per axis")
        if vox.min() < 0 or vox.max() > n:
            raise CvoxError(f"matrix {matrix.name!r}: palette index out of range")
        rotation = tuple(float(c) for c in matrix.rotation)
        if len(rotation) != 4 or not all(math.isfinite(c) for c in rotation):
            raise CvoxError(f"matrix {matrix.name!r}: rotation must be 4 finite floats")

        if not 1 <= matrix.resolution <= MAX_RESOLUTION:
            raise CvoxError(f"matrix {matrix.name!r}: resolution must be 1-{MAX_RESOLUTION}")

        data = encode_voxels(vox, index_width)
        if compress:
            data = zlib.compress(data)
        sz, sy, sx = vox.shape
        body = b"".join((
            struct.pack("<B", matrix.resolution),
            _pack_string(matrix.name),
            _pack_int32s(matrix.position, "position"),
            struct.pack("<3H", sx, sy, sz),
            _pack_int32s(matrix.pivot, "pivot"),
            struct.pack("<4f", *rotation),
            struct.pack("<I", len(data)),
            data,
        ))
        out += struct.pack("<I", len(body)) + body
    return bytes(out)


def save(model: Model, path: str | Path, compress: bool = False) -> None:
    Path(path).write_bytes(dumps(model, compress))


# --- reading ------------------------------------------------------------------

class _Cursor:
    def __init__(self, data: bytes, pos: int = 0, end: int | None = None):
        self.data, self.pos = data, pos
        self.end = len(data) if end is None else end

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > self.end:
            raise CvoxError("unexpected end of data")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def string(self) -> str:
        raw = self.take(self.unpack("<B")[0])
        try:
            return raw.decode("ascii")
        except UnicodeDecodeError:
            raise CvoxError("non-ASCII name") from None


def loads(data: bytes) -> Model:
    """Parse .cvox bytes into a Model, rejecting non-conforming files."""
    cur = _Cursor(data)
    if cur.take(4) != MAGIC:
        raise CvoxError("bad magic, not a .cvox file")
    version, flags, n = cur.unpack("<HHH")
    if version > VERSION:
        raise CvoxError(f"unsupported version {version}")
    if flags & ~KNOWN_FLAGS:
        raise CvoxError(f"unknown flag bits set: {flags:#06x}")
    compressed = bool(flags & FLAG_ZLIB)
    index_width = 1 if n <= 255 else 2

    palette = np.frombuffer(cur.take(4 * n), dtype=np.uint8).reshape(n, 4).copy()
    (matrix_count,) = cur.unpack("<H")

    matrices = []
    for _ in range(matrix_count):
        (block_length,) = cur.unpack("<I")
        if cur.pos + block_length > len(data):
            raise CvoxError("block length runs past end of file")
        block = _Cursor(data, cur.pos, cur.pos + block_length)
        cur.pos = block.end  # trailing bytes in a block are skipped

        (resolution,) = block.unpack("<B")
        if not 1 <= resolution <= MAX_RESOLUTION:
            raise CvoxError(f"resolution {resolution} is not 1-{MAX_RESOLUTION}")
        name = block.string()
        position = block.unpack("<3i")
        sx, sy, sz = block.unpack("<3H")
        if min(sx, sy, sz) < 1:
            raise CvoxError(f"matrix {name!r}: size must be at least 1")
        pivot = block.unpack("<3i")
        rotation = block.unpack("<4f")
        (data_length,) = block.unpack("<I")
        raw = block.take(data_length)
        if compressed:
            try:
                raw = zlib.decompress(raw)
            except zlib.error as exc:
                raise CvoxError(f"matrix {name!r}: bad zlib data ({exc})") from None
        voxels = decode_voxels(raw, (sz, sy, sx), n, index_width)
        matrices.append(Matrix(name, voxels, position, pivot, rotation, resolution))
    return Model(palette, matrices)


def load(path: str | Path) -> Model:
    return loads(Path(path).read_bytes())
