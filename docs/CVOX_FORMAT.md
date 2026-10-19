# The .cvox file format (version 1)

.cvox is the native model format of Chronovox Reforged. A file holds **one shared palette** and a list of
**matrices**. A matrix is a 3D grid of voxels (each cell is a palette index or empty) with a name, a position, a
pivot, a rotation and a resolution. Voxel data is run-length encoded and can be zlib-compressed.

This page is for anyone writing a reader or writer, such as the Unity importer. The reference implementation is
[chronovox/cvox.py](../chronovox/cvox.py), and the round-trip tests are in [tests/test_cvox.py](../tests/test_cvox.py).
The same specification, with the reasons behind it, is in the design doc ([DESIGN.md](DESIGN.md)); if the two
ever differ, the design doc wins.

## Conventions

- All multi-byte values are **little-endian**.
- **Coordinates** are Y-up and **left-handed**, the same as Unity. One voxel is one unit at resolution 1.
- **Positions and pivots** are fixed-point int32 values in units of 1/128. The value is the stored number divided by 128.
- **Strings** are a uint8 byte length followed by that many ASCII bytes, with no terminator. They are at most 255 bytes.
  Writers must reject non-ASCII text.
- **Quaternions** are four float32 values in the order (x, y, z, w). The identity is (0, 0, 0, 1).

## File layout

```
header
palette            N colors
matrix count       uint16
matrix block       repeated "matrix count" times
```

### Header

| Bytes | Field | Type | Meaning |
| --- | --- | --- | --- |
| 4 | Magic | char[4] | `CVOX` |
| 2 | Version | uint16 | 1 |
| 2 | Flags | uint16 | Bit 0: voxel data is zlib-compressed. All other bits must be 0. |
| 2 | Palette count N | uint16 | 0 to 65,535 |
| 4 x N | Palette | uint8 x 4 each | R, G, B, A. Entry k (from 0) is palette index k + 1. |
| 2 | Matrix count | uint16 | Number of matrix blocks that follow |

Palette index **0 means empty** and is not stored. The index width of every voxel is **1 byte if N is 255 or
less, otherwise 2 bytes**. There is no field for it, so derive it from N. The editor always writes alpha 255:
a voxel is either there or it is not, so palette alpha carries no meaning.

### Matrix block

| Bytes | Field | Type | Meaning |
| --- | --- | --- | --- |
| 4 | Block length | uint32 | Number of bytes in the rest of this block, so a reader can skip it |
| 1 | Resolution | uint8 | 1 to 8. A cell is 1 / 2^(resolution - 1) units across. |
| 1 + n | Name | string | |
| 12 | Position | int32 x 3 | Where the matrix origin is in the model, in 1/128 units (x, y, z) |
| 6 | Size | uint16 x 3 | Cells along x, y and z, each at least 1 |
| 12 | Pivot | int32 x 3 | The rotation pivot, in 1/128 **cells** from the origin corner |
| 16 | Rotation | float32 x 4 | Quaternion (x, y, z, w) |
| 4 | Data length | uint32 | Size of the voxel data as stored |
| Data length | Voxel data | bytes | Run-length data below, zlib-compressed if the flag is set |

Bytes left over at the end of a block, after the voxel data, must be skipped. Use the block length to find the
next block.

**Placement.** The matrix origin is the outer corner of cell (0, 0, 0). A point `p` of the matrix, measured in
cells from the origin, lands in the model at

```
model = position + R * f * (p - pivot) + f * pivot
```

where `R` is the rotation, `f` is the cell size `1 / 2^(resolution - 1)`, and `position` and `pivot` are the stored
values divided by 128. So with no rotation the origin corner sits exactly at `position`, and the cell at (x, y, z)
fills the box from (x, y, z) to (x + 1, y + 1, z + 1) in cells.

| Resolution | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cell size (units) | 1 | 1/2 | 1/4 | 1/8 | 1/16 | 1/32 | 1/64 | 1/128 |

Resolution 8 is the finest because positions are stored in 1/128 units.

### Voxel data

After decompression the data is a list of **runs** that together cover every cell of the matrix:

| Field | Type | Meaning |
| --- | --- | --- |
| Count | varint | How many cells in a row have this value, at least 1 |
| Index | uint8 or uint16 | The palette index, 0 for empty. The width comes from the palette count. |

A **varint** is an unsigned LEB128: 7 bits per byte, lowest bits first, the top bit set on every byte but the last.

The cells run **x fastest, then y, then z**: the cell at (x, y, z) is number `x + y * size_x + z * size_x * size_y`.
In numpy that is an array of shape `(size_z, size_y, size_x)` in C order. A run may cross the end of a row or a
layer. Writers should merge neighbouring runs that have the same index.

If the flag is set, each matrix's data is compressed **on its own** (a plain zlib stream), so one matrix can be
decoded without the rest.

## Reading a file

1. Check the magic is `CVOX`, the version is not above 1 and no unknown flag bit is set.
2. Read N, the palette and the matrix count. Work out the index width from N.
3. For each matrix: read the block length, then the fields above from that block. Decompress the data if the flag is
   set, and decode runs until you have exactly `size_x * size_y * size_z` cells. Then move on by the block length.

Reject the file when:

- a run has an index above N, or a count below 1;
- the run counts do not add up to exactly `size_x * size_y * size_z`, or go past it;
- a block length or data length runs past the end of the file;
- the resolution is not between 1 and 8, or a size is 0;
- a name is not ASCII.

## Limits

| What | Limit |
| --- | --- |
| Palette colors | 65,535 |
| Matrices per file | 65,535 |
| Cells along one axis | 65,535 |
| Name length | 255 bytes |
| Position and pivot | int32, so about 16.7 million cells either way |

The editor itself also refuses matrices of more than 2^26 (about 67 million) cells.

## Example

A model with one red color and one matrix called `Arm`: 3 x 1 x 1 cells (red, red, empty), resolution 2, position
(2, 0, -0.5), pivot (1.5, 0.5, 0) cells, no rotation, uncompressed. 79 bytes:

```
43 56 4f 58        magic "CVOX"
01 00              version 1
00 00              flags: not compressed
01 00              palette count 1
ff 00 00 ff        palette entry 1: red, alpha 255
01 00              matrix count 1
3b 00 00 00        block length 59
02                 resolution 2 (cells are 1/2 unit)
03 41 72 6d        name: length 3, "Arm"
00 01 00 00        position x = 256 / 128 = 2
00 00 00 00        position y = 0
c0 ff ff ff        position z = -64 / 128 = -0.5
03 00 01 00 01 00  size 3 x 1 x 1
c0 00 00 00        pivot x = 192 / 128 = 1.5
40 00 00 00        pivot y = 64 / 128 = 0.5
00 00 00 00        pivot z = 0
00 00 00 00        rotation x
00 00 00 00        rotation y
00 00 00 00        rotation z
00 00 80 3f        rotation w = 1.0
04 00 00 00        data length 4
02 01              run: 2 cells of palette index 1
01 00              run: 1 cell of index 0 (empty)
```
