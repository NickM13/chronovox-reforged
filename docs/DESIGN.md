# Chronovox Reforged — Master Design Doc

Oct 15, 2026 · @Nick

## Project overview

Chronovox Reforged is a Python rewrite of Chronovox Studio, Nick's voxel model editor, originally written in C++ over a decade ago. The rewrite targets stability: the original is slow to open, sometimes crashes, and leaks memory.

The editor builds voxel models as a set of named **matrices** (independent voxel grids) used purely for graphics. Models feed Nick's Unity game, Aetheria. Bones and animation rigs are set up later, outside this tool.

**Goals**

- A stable, fast-starting editor with no memory leaks.
- A new native format, **.cvox**, with finer positions, per-matrix rotation, a large palette, and better compression.
- Full import of legacy **.csm** (Chronovox Studio Model) v4 files.
- A codebase that is easy to modify and extend, in the spirit of Blender's Python layer.

**Non-goals**

- Bones, parenting, or animation. The .csm `Parent` field is dropped.
- Writing .csm files. Legacy support is read-only.
- Exporting to .vox or .obj.

**For agents using this doc:** treat the .cvox spec below as authoritative. If something is ambiguous or missing, raise it as an open question rather than guessing, and record any decision in the decisions log.

## Tech stack

Python does the orchestration while heavy work runs in numpy or on the GPU, so the editor stays responsive without native code.

| Component | Library | Role |
| --- | --- | --- |
| Language | Python 3 | Editor logic, tools, file I/O |
| Voxel storage | numpy | Each matrix's grid as one array; fast fills, mirrors, shifts |
| Rendering | moderngl | OpenGL rendering on the GPU |
| UI panels | imgui (Python bindings) | Tool panels, sliders, palette |
| Meshing speed-up | numba (optional) | Accelerates face culling and greedy meshing if numpy alone is too slow |
| Binary I/O | struct | Reading and writing headers and fields |
| Compression | zlib | Optional compression of .cvox voxel data |

Meshing (turning voxels into faces) is the main performance hot spot; profile it before adding numba. Some models, such as trees, exceed 200 voxels per axis (about 8 million voxels), so meshing speed matters from the first viewer. Target platform is Windows.

## .cvox format specification (version 1)

A standalone copy for people writing readers is in [CVOX_FORMAT.md](CVOX_FORMAT.md). Keep the two in step; this section is the authoritative one.

A .cvox file is a header, one shared palette, then a list of matrix blocks; voxel data is run-length encoded palette indices, optionally zlib-compressed.

### Conventions

- All multi-byte values are **little-endian**.
- **Fixed-point positions** use units of 1/128 voxel, stored as int32. Value = stored ÷ 128. Range is about ±16.7 million voxels.
- **Coordinate system:** Y-up, left-handed, matching Unity. One voxel = one unit.
- **Strings:** uint8 byte length, then that many ASCII bytes, no null terminator. Writers must reject non-ASCII names. Maximum 255 bytes.

### Header

| Size (bytes) | Name | Type | Description |
| --- | --- | --- | --- |
| 4 | Magic | char\[4\] | ASCII `CVOX` |
| 2 | Version | uint16 | Format version, currently 1 |
| 2 | Flags | uint16 | Bit 0: voxel data is zlib-compressed. All other bits must be 0. |
| 2 | Palette count | uint16 | Number of colors N, from 0 to 65,535 |
| 4 × N | Palette | uint8 × 4 each | R, G, B, A for indices 1 through N |
| 2 | Matrix count | uint16 | Number of matrix blocks that follow |

**Palette rules.** Index 0 is reserved for empty and is not stored. Stored entry k (counting from 0) is palette index k + 1. A file should share one palette across all its matrices.

**Index width.** If N ≤ 255, every voxel index is stored as 1 byte (uint8). Otherwise each index is 2 bytes (uint16). Readers derive this from N; there is no separate field.

### Matrix block

Repeated `Matrix count` times.

| Size (bytes) | Name | Type | Description |
| --- | --- | --- | --- |
| 4 | Block length | uint32 | Bytes remaining in this block after this field. Lets readers skip a matrix. |
| 1 | Resolution | uint8 | Cell fineness, 1 to 8. A cell is f = 1 / 2^(resolution − 1) units across: resolution 1 is 1 unit, 2 is 1/2, 3 is 1/4, and so on down to 1/128 at 8. |
| 1 + n | Name | string | Matrix name |
| 12 | Position | int32 × 3 | Model-space position of the matrix origin, 1/128 units (x, y, z) |
| 6 | Size | uint16 × 3 | Grid size in voxels (x, y, z), each ≥ 1 |
| 12 | Pivot | int32 × 3 | Rotation pivot in matrix-local space, measured from the origin corner, 1/128 units |
| 16 | Rotation | float32 × 4 | Unit quaternion (x, y, z, w). Identity is (0, 0, 0, 1). |
| 4 | Data length | uint32 | Byte length of the voxel data as stored |
| Data length | Voxel data | bytes | Run-length encoded voxels (below), zlib-compressed if flag bit 0 is set |

The **matrix origin** is the outer corner of voxel (0, 0, 0). A matrix-local point p, measured in cells, maps to model space as:

```latex
p_{\text{model}} = \text{position} + R\,f\,(p - \text{pivot}) + f\,\text{pivot}
```

where R is the rotation and f is the cell size, 1 / 2^(resolution − 1). The matrix scales about its pivot, and the pivot is measured in cells, so the pivot point stays put if the resolution changes. Position is in model units and is unaffected by the resolution. With identity rotation and resolution 1 this reduces to position + p; with identity rotation the origin corner is always at position. Resolution 8 (cells of 1/128 unit) is the finest because position is stored in 1/128 units.

### Voxel data encoding

After decompression (if flagged), voxel data is a sequence of **runs**, each:

| Field | Type | Description |
| --- | --- | --- |
| Count | varint | Number of consecutive voxels in this run, ≥ 1 |
| Index | uint8 or uint16 | Palette index (0 = empty), width set by the palette count |

**Varint** is unsigned LEB128: 7 bits of value per byte, least significant group first, high bit set on every byte except the last. Values up to 127 take 1 byte; up to 16,383 take 2 bytes.

**Iteration order:** x varies fastest, then y, then z. The linear voxel index is x + y × size\_x + z × size\_x × size\_y. In numpy this is an array of shape (size\_z, size\_y, size\_x) in C order.

Runs may cross row and slice boundaries. Writers should merge adjacent runs with the same index.

When compression is enabled, each matrix's voxel data is compressed independently, so a matrix can be skipped or decoded on its own.

### Reader validation

A conforming reader rejects a file when:

- the magic is not `CVOX`, or the version is higher than it supports;
- any unknown flag bit is set;
- a run's index exceeds the palette count;
- the run counts in a matrix do not sum exactly to size\_x × size\_y × size\_z;
- a block length or data length runs past the end of the file;
- a matrix's resolution is not between 1 and 8.

## Legacy .csm import (version 4)

Chronovox Reforged reads .csm files and converts them to the in-memory model; it never writes .csm. Only version 4 is supported; no v3 files need importing, so the importer rejects other versions. The layout below is confirmed from his Unity loader (`CreateModelObject` and `CreateMatrixObject` in C#).

### .csm v4 layout

Little-endian throughout.

**Header**

| Size (bytes) | Name | Type | Description |
| --- | --- | --- | --- |
| 4 | Format type | char\[4\] | `.CSM` |
| 4 | Version | int32 | 4 for current files |
| 4 | Unused | int32 | Ignored |
| 4 | Matrix count | int32 | Number of matrices |

**Body (per matrix)**

| Size (bytes) | Name | Type | Description |
| --- | --- | --- | --- |
| 4 + n | Matrix name | int32 length + n chars | 4-byte length prefix, then the characters |
| 4 + n | Matrix parent | int32 length + n chars | Same encoding as the name |
| 6 | Position | int16 × 3 | Signed, since positions can be negative. Divide by 10. |
| 6 | Size | int16 × 3 | Grid size in voxels |
| 5 per run | Voxel data | uint8 × 5 | Count (1 to 255), R, G, B, A. Runs repeat until counts sum to size\_x × size\_y × size\_z. |

**Iteration order:** z varies fastest, then y, then x (loops nest x outermost, z innermost). For linear index i: z = i mod size\_z, y = (i ÷ size\_z) mod size\_y, x = i ÷ (size\_z × size\_y), using integer division. This is the reverse of .cvox, which has x fastest, so the importer must transpose. In numpy: reshape the decoded voxels to (size\_x, size\_y, size\_z), then transpose to (size\_z, size\_y, size\_x).

**X mirror:** the Unity loader mirrors the X axis. Each voxel's x becomes size\_x − 1 − x, and the matrix position becomes −position\_x − size\_x. Because .cvox uses Unity's coordinate convention, the importer applies this mirror once during conversion, and .cvox files need no flip afterwards.

### Conversion to the in-memory model

- **Name:** kept as is.
- **Parent:** read and discarded.
- **Position:** apply the X mirror, then convert from 1/10 units to 1/128 units: stored\_cvox = round(value\_in\_voxels × 128). This can introduce up to 1/256 voxel of rounding.
- **Rotation and pivot:** identity rotation, pivot at the matrix center, which is size × 64 per axis in 1/128 units.
- **Colors:** build one shared palette from all distinct RGB values across all matrices. Alpha 0 means no voxel and maps to index 0, regardless of its RGB values. Any other alpha (the old editor wrote 1) only meant "solid": it is dropped, the color is stored with alpha 255, and colors that differ only in alpha are one color. If a model has more than 65,535 distinct colors, report an error instead of silently quantizing.

### Name characters

Matrix names are ASCII only. C#'s `ReadChars` counts characters rather than bytes, but for ASCII the two are the same, so the importer reads n bytes and decodes them as ASCII.

## Design decisions

Each decision below was agreed for version 1; change one only with Nick's sign-off and a new entry here.

| Decision | Choice | Rationale |
| --- | --- | --- |
| Language | Python | Stable (no manual memory management), quick to start, easy to modify |
| Format identity | New format .cvox, magic `CVOX` | Nearly every field changes layout; a new extension avoids confusion and protects the old editor from misreading the new layout |
| Legacy support | Import .csm v4, read-only | Existing models carry over without maintaining the old format |
| Position precision | 1/128 voxel, int32 | Power of two, so values are exact in floats and round-trip without drift; 0.01 is not exactly representable. int16 would cap range at ±256 voxels. |
| Rotation | Quaternion, float32 × 4 | No gimbal lock, no rotation-order ambiguity, maps directly to Unity's Quaternion. The editor may show Euler angles in degrees. |
| Pivot | Explicit per matrix | Small rotations need a sensible center; without a pivot they rotate around a corner. Imported and new matrices default to their center. |
| Parent field | Removed | Bones and hierarchy are set up later, outside this tool |
| Palette | Shared per file, up to 65,535 colors, index 0 = empty | Large color range while keeping indices compact |
| Palette alpha | Always 255 in the editor; any other alpha is reset on import and on opening a file (Oct 2026, Nick) | The old editor's alpha 1 only marked a voxel as solid. The format still stores RGBA, but nothing uses alpha. |
| Unused palette colors | Kept while editing, left out of saved files (Oct 2026, Nick) | A color you erased by accident stays available; files only carry colors some voxel uses. Unused colors are marked in the palette. |
| Matrix order | The order in the editor's list is the order in the file (Oct 2026, Nick) | The list can be reordered by drag and drop or Ctrl+Up / Ctrl+Down |
| Resolution | Per matrix, 1 to 8: a cell is 1 / 2^(resolution − 1) units across, so a higher number means finer cells. Stored as the first byte of the matrix block, right after the block length; format version stays 1 because no files had been saved yet (Oct 2026, Nick) | Lets a big area such as a tree's leaves use coarse cells (resolution 1) while branches use fine ones. The editor draws matrices at their real size. Cells scale about the pivot, which is in cells. The limit of 8 is where a cell is 1/128 unit, the finest a position can address. |
| Merging matrices | Needs the same rotation, the same resolution and a whole-cell offset (Oct 2026, Nick) | Cells must line up exactly; otherwise merging would have to resample. The selected matrix is the target and the chosen one is merged into it. |
| Index width | 1 byte if ≤ 255 colors, else 2 | Small models stay as compact as possible |
| Run counts | Unsigned LEB128 varint | Removes the 255-per-run cap; long empty stretches become one run |
| Extra compression | Optional zlib per matrix | RLE handles repetition within rows; zlib catches patterns across rows. Built into Python and .NET. |
| Block length per matrix | uint32 | Readers can skip matrices or future unknown data, a lesson from .vox's chunk design |
| Iteration order | x fastest, then y, then z | Removes the ambiguity the .csm spec has |
| Coordinates | Y-up, left-handed | Matches Unity, so import needs no axis conversion |

## Milestone plan

The first milestone is file I/O, so real models can be tested from day one; each later milestone ends with something visibly working.

1. **.cvox reader and writer.** Python module for read and write, plus round-trip tests (write → read → identical data), including both index widths and compression on and off.
2. **.csm v4 importer.** Load legacy files into the in-memory model, using the confirmed v4 layout. Convert a real model to .cvox and compare file sizes.
3. **Viewer.** A window with an orbit camera that renders a loaded model, with face culling.
4. **Basic editing.** Click to place and remove voxels, a color picker, save to .cvox.
5. **Undo and redo.**
6. **Matrix tools.** Add, rename, delete, and select matrices; edit position, pivot, and rotation (shown in degrees).
7. **Palette panel.** View, edit, and pick colors from the shared palette.
8. **Performance pass.** Greedy meshing and numba where profiling shows a need.
9. **Unity importer.** A C# reader for .cvox in Aetheria.

## Open questions and agent guidance

All open questions are resolved; the list below records each answer.

**Open questions**

- [x] Matrix names: ASCII only.
- [x] .csm versions: v4 only; no v3 files need importing.
- [x] Pivot default: the matrix center.
- [x] Model sizes: vary widely; some, such as trees, exceed 200 voxels per axis.
- [x] Rendering library: moderngl.
- [x] UI library: imgui.
- [x] Target platform: Windows.
- [x] .vox and .obj export: non-goals.

**Guidance for agents**

- Follow the .cvox spec exactly; do not add fields or reinterpret existing ones without Nick's approval.
- Keep file I/O in its own module with no rendering or UI imports, so it can be tested and reused alone.
- Store each matrix's voxels as a numpy array of shape (size\_z, size\_y, size\_x) holding palette indices.
- Write round-trip tests for every format change.
- When a detail is unknown, say so and ask rather than guessing, especially for .csm import.
