# Overview

gsply is a high-performance Python library for Gaussian Splatting PLY file I/O.
It combines zero-copy memory management with JIT-compiled compression pipelines
to deliver industry-leading performance with a clean, Pythonic API.

## Core Design Philosophy

gsply focuses on two complementary workflows:

- **Ultra-fast I/O** — Read and write uncompressed or PlayCanvas-compressed PLY files using
  vectorized NumPy kernels and Numba JIT pipelines. Zero-copy views into a shared `_base` buffer
  eliminate unnecessary memory allocations.

- **Flexible Containers** — `GSData` (CPU) and `GSTensor` (GPU) provide helpers for concatenation,
  masking, contiguity optimization, and CPU↔GPU transfers.

## Key Capabilities

### Performance First

Benchmarked at **93M Gaussians/sec** peak read and **57M Gaussians/sec** peak write (zero-copy).
Uses fused unpack/pack kernels and pre-computed lookup tables. Zero-copy reads eliminate
memory overhead; auto-consolidation optimizes writes automatically.

### Format Flexibility

`detect_format()` auto-detects PLY layout. `plywrite()` selects compressed output when
`compressed=True` or when the file extension is `.compressed.ply`.

### Advanced Mask Management

`GSData` and `GSTensor` support multiple boolean mask layers with named entries.
Use `add_mask_layer()`, `combine_masks()`, and `apply_masks()` for filtering.
Masks persist through slicing, concatenation, and CPU↔GPU transfers.

### Memory Layout Optimization

`GSData.make_contiguous()` recompacts `_base` views into contiguous arrays once you cross
the ≈8 operations break-even point, dramatically accelerating reductions (`sum`, `max`, etc.)
and point-wise transforms. Up to **45x faster** for certain operations.

### Seamless GPU Integration

`GSTensor.from_gsdata()` transfers the shared base tensor to the requested device
(CUDA or CPU) in a single operation. Mask layers persist through transfers, and
GPU operations leverage PyTorch's parallelism for 100-1000x speedups over CPU.

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
| Load a PLY file (any format)                | `plyread()` — Auto-detects format                  |
| Write back to disk (auto-optimized)        | `plywrite()` — Automatic zero-copy optimization    |
| Stream compressed bytes over network        | `compress_to_bytes()` / `decompress_from_bytes()`  |
| Batch merge hundreds of shards              | `GSData.concatenate()` — Bulk merge (5.74x faster) |
| GPU training / rendering loops              | `GSTensor.from_gsdata()` + `apply_masks()`         |
| Filter data with multiple conditions        | `add_mask_layer()` + `combine_masks()`             |
| Optimize for many array operations         | `make_contiguous()` — Up to 45x speedup           |

## Next Steps

- **New users**: Start with the :doc:`usage` guide for installation and basic examples
- **API reference**: Browse the :doc:`api/index` for complete function and class documentation
- **Performance tuning**: See the performance notes in individual function docstrings
