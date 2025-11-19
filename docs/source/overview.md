# Overview

gsply is a high-performance Python library for Gaussian Splatting PLY file I/O.
It combines zero-copy memory management with JIT-compiled compression pipelines
to deliver industry-leading performance with a clean, Pythonic API.

## Core Design Philosophy

gsply focuses on three complementary workflows:

- **Ultra-fast I/O** — Read and write uncompressed or PlayCanvas-compressed PLY files using
  vectorized NumPy kernels and Numba JIT pipelines. Zero-copy views into a shared `_base` buffer
  eliminate unnecessary memory allocations.

- **Optimal GPU Transfer** — Data loaded from `plyread()` uses the `_base` tensor for **11x faster**
  GPU transfers. Single tensor transfer eliminates CPU-side memory copies, achieving 1.99ms vs
  22.78ms for 400K Gaussians.

- **Flexible Containers** — `GSData` (CPU) and `GSTensor` (GPU) provide helpers for concatenation,
  masking, contiguity optimization, and CPU↔GPU transfers. Format state is tracked automatically
  via `_format` dictionary for seamless in-place conversions.

## Key Capabilities

### Performance First

Benchmarked at **93M Gaussians/sec** peak read and **57M Gaussians/sec** peak write (zero-copy).
Uses fused unpack/pack kernels and pre-computed lookup tables. Zero-copy reads eliminate
memory overhead; auto-consolidation optimizes writes automatically.

### Format Flexibility

`detect_format()` auto-detects PLY layout. `plywrite()` selects compressed output when
`compressed=True` or when the file extension is `.compressed.ply`. Supports uncompressed PLY,
PlayCanvas compressed PLY, and SOG (Splat Ordering Grid) formats.

### Object-Oriented API

Convenient save/load methods for cleaner code:
- `data.save(file_path, compressed=False)` - Instance method for saving
- `GSData.load(file_path)` - Classmethod for loading (auto-detects format)
- `gstensor.save(file_path, compressed=True)` - GPU compression by default
- `GSTensor.load(file_path, device='cuda')` - Direct GPU loading

### Format Conversion with In-Place Tracking

Convert between linear and PLY formats seamlessly with automatic format state tracking:
- `normalize()` / `denormalize()` - Convert scales/opacities between linear and PLY formats
- `to_rgb()` / `to_sh()` - Convert sh0 between SH and RGB color formats
- Available for both `GSData` (CPU) and `GSTensor` (GPU)
- In-place operations by default (`inplace=True`) for efficiency
- Format state tracked automatically via `_format` dictionary (scales, opacities, sh0, sh_order)
- Conversion methods update `_format` to reflect current data state

### Advanced Mask Management

`GSData` and `GSTensor` support multiple boolean mask layers with named entries.
Use `add_mask_layer()`, `combine_masks()`, and `apply_masks()` for filtering.
Masks persist through slicing, concatenation, and CPU↔GPU transfers.

### Memory Layout Optimization

`GSData.make_contiguous()` recompacts `_base` views into contiguous arrays once you cross
the ≈8 operations break-even point, dramatically accelerating reductions (`sum`, `max`, etc.)
and point-wise transforms. Up to **45x faster** for certain operations.

### Optimal GPU Transfer

`GSTensor.from_gsdata()` uses the `_base` tensor optimization for **11x faster** GPU transfers:
- **With `_base`** (from `plyread()`): Single tensor transfer, zero CPU copy overhead (1.99ms for 400K Gaussians)
- **Without `_base`**: Falls back to stacking arrays on CPU then transferring (22.78ms for 400K Gaussians)
- Mask layers persist through transfers, and GPU operations leverage PyTorch's parallelism for 100-1000x speedups over CPU
- Format state (`_format` dict) is preserved during GPU transfers for seamless conversion tracking

## Data Layout

Each Gaussian is represented by the following NumPy arrays:

| Attribute   | Shape      | Description                          |
|-------------|------------|--------------------------------------|
| `means`     | `(N, 3)`   | XYZ world-space coordinates          |
| `scales`    | `(N, 3)`   | Log-scale parameters                 |
| `quats`     | `(N, 4)`   | WXYZ quaternion rotations            |
| `opacities` | `(N,)`     | Logit opacity values                 |
| `sh0`       | `(N, 3)`   | DC spherical harmonics (RGB)         |
| `shN`       | `(N, K, 3)`| Higher-order SH bands (optional)     |
| `masks`     | `(N,)` or `(N, L)` | Boolean mask layers (optional) |

**Zero-copy optimization**: When reading from PLY files, these properties are arranged as
column slices of a single `_base` array. Each view shares storage with the base array,
and Python's reference counting keeps the base alive automatically.

## API Selection Guide

Choose the right API for your use case:

| Scenario                                    | Recommended API                                    |
|---------------------------------------------|-----------------------------------------------------|
| Load a PLY file (any format)                | `GSData.load()` — Object-oriented, auto-detects format |
| Write back to disk (auto-optimized)        | `data.save()` — Object-oriented, automatic optimization |
| Load SOG format files                      | `sogread()` — Returns GSData (same API)            |
| Convert linear ↔ PLY format                | `normalize()` / `denormalize()` — In-place conversion |
| Convert SH ↔ RGB colors                    | `to_rgb()` / `to_sh()` — In-place color conversion |
| Stream compressed bytes over network        | `compress_to_bytes()` / `decompress_from_bytes()`  |
| Batch merge hundreds of shards              | `GSData.concatenate()` — Bulk merge (5.74x faster) |
| GPU training / rendering loops              | `GSTensor.load()` — Direct GPU loading             |
| GPU compression                            | `gstensor.save()` — GPU compression (default)      |
| Filter data with multiple conditions        | `add_mask_layer()` + `combine_masks()`             |
| Optimize for many array operations         | `make_contiguous()` — Up to 45x speedup           |

## Next Steps

- **New users**: Start with the :doc:`usage` guide for installation and basic examples
- **API reference**: Browse the :doc:`api/index` for complete function and class documentation
- **Performance tuning**: See the performance notes in individual function docstrings
