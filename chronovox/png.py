"""A small PNG reader and writer, and a resizer, for the window icon. Only numpy and zlib: no other imports.

The reader handles ordinary 8 bit, non-interlaced PNGs (grey, grey with alpha, RGB and RGBA) with every row filter.
The writer always writes RGBA with no row filter, which is the easiest thing for the reader to decode quickly.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngError(ValueError):
    """Raised for a PNG file this module cannot read."""


def _chunks(data: bytes):
    pos = len(SIGNATURE)
    while pos + 8 <= len(data):
        length, tag = struct.unpack(">I4s", data[pos:pos + 8])
        yield tag, data[pos + 8:pos + 8 + length]
        pos += 12 + length                                  # length, tag, body, crc


def _unfilter(raw: bytes, width: int, height: int, channels: int) -> np.ndarray:
    stride = width * channels
    if len(raw) != (stride + 1) * height:
        raise PngError("the image data is the wrong size")
    out = np.zeros((height, stride), np.uint8)
    previous = np.zeros(stride, np.uint8)
    for y in range(height):
        start = y * (stride + 1)
        kind = raw[start]
        row = np.frombuffer(raw, np.uint8, stride, start + 1).copy()
        if kind == 0:
            pass
        elif kind == 1:                                     # Sub: add the pixel to the left
            for c in range(channels):
                row[c::channels] = np.cumsum(row[c::channels], dtype=np.uint64) % 256
        elif kind == 2:                                     # Up: add the pixel above
            row = (row + previous).astype(np.uint8)
        elif kind in (3, 4):                                # Average and Paeth need the finished left neighbours
            cur, up = row.tolist(), previous.tolist()
            for i in range(stride):
                left = cur[i - channels] if i >= channels else 0
                if kind == 3:
                    predictor = (left + up[i]) >> 1
                else:
                    upper_left = up[i - channels] if i >= channels else 0
                    estimate = left + up[i] - upper_left
                    da, db, dc = abs(estimate - left), abs(estimate - up[i]), abs(estimate - upper_left)
                    predictor = left if da <= db and da <= dc else (up[i] if db <= dc else upper_left)
                cur[i] = (cur[i] + predictor) & 255
            row = np.array(cur, np.uint8)
        else:
            raise PngError(f"unknown row filter {kind}")
        out[y] = row
        previous = row
    return out.reshape(height, width, channels)


def decode_png(data: bytes) -> np.ndarray:
    """PNG bytes to an (height, width, 4) uint8 RGBA array."""
    if not data.startswith(SIGNATURE):
        raise PngError("not a PNG file")
    header, idat = None, []
    for tag, body in _chunks(data):
        if tag == b"IHDR":
            header = struct.unpack(">IIBBBBB", body[:13])
        elif tag == b"IDAT":
            idat.append(body)
    if header is None or not idat:
        raise PngError("the PNG file is incomplete")
    width, height, depth, color, _, _, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    if depth != 8 or channels is None or interlace != 0:
        raise PngError("only 8 bit, non-interlaced grey, RGB and RGBA PNG files are supported")
    pixels = _unfilter(zlib.decompress(b"".join(idat)), width, height, channels)
    rgba = np.empty((height, width, 4), np.uint8)
    if color == 0:
        rgba[..., :3], rgba[..., 3] = pixels, 255
    elif color == 4:
        rgba[..., :3], rgba[..., 3] = pixels[..., :1], pixels[..., 1]
    elif color == 2:
        rgba[..., :3], rgba[..., 3] = pixels, 255
    else:
        rgba[...] = pixels
    return rgba


def read_png(path: str | Path) -> np.ndarray:
    return decode_png(Path(path).read_bytes())


def encode_png(rgba: np.ndarray) -> bytes:
    """An (height, width, 4) uint8 array to PNG bytes (RGBA, no row filter)."""
    rgba = np.ascontiguousarray(rgba, np.uint8)
    height, width, _ = rgba.shape
    rows = np.concatenate((np.zeros((height, 1), np.uint8), rgba.reshape(height, width * 4)), axis=1)

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows.tobytes(), 9)) + chunk(b"IEND", b""))


def write_png(path: str | Path, rgba: np.ndarray) -> None:
    Path(path).write_bytes(encode_png(rgba))


def _area_weights(old: int, new: int) -> np.ndarray:
    """Matrix (new, old) that averages the old samples falling in each new sample's span, by how much of each they cover."""
    edges = np.linspace(0.0, old, new + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    cells = np.arange(old)[None, :]
    overlap = np.clip(np.minimum(hi, cells + 1) - np.maximum(lo, cells), 0.0, None)
    return overlap / (hi - lo)


def resize(rgba: np.ndarray, width: int, height: int | None = None) -> np.ndarray:
    """Resize an RGBA image by averaging (good for making small copies). Colors are weighted by alpha, so the
    soft transparent edges of a picture do not turn dark or fringed."""
    height = width if height is None else height
    old_h, old_w, _ = rgba.shape
    image = rgba.astype(np.float64)
    alpha = image[..., 3:4] / 255.0
    image[..., :3] *= alpha                                  # premultiplied, so averaging respects the transparency
    wy, wx = _area_weights(old_h, height), _area_weights(old_w, width)
    small = np.einsum("lk,ikc->ilc", wx, np.einsum("ij,jkc->ikc", wy, image))     # rows first, then columns
    alpha_out = small[..., 3:4] / 255.0
    color = np.divide(small[..., :3], alpha_out, out=np.zeros_like(small[..., :3]), where=alpha_out > 1e-9)
    out = np.concatenate((color, small[..., 3:4]), axis=2)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
