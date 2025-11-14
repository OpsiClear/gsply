<div align="center">

# gsply

### Ultra-Fast Gaussian Splatting PLY I/O Library

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

**93M Gaussians/sec read | 87M Gaussians/sec write | Pure Python with NumPy and Numba**

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Performance](#performance) | [Documentation](#documentation)

</div>

---

## Overview

**gsply** is a pure Python library for ultra-fast reading and writing of Gaussian Splatting PLY files. Built specifically for performance-critical applications, gsply achieves read speeds up to 93M Gaussians/sec and write speeds up to 87M Gaussians/sec. Core functionality requires only NumPy and Numba, with optional PyTorch integration for GPU acceleration.

**Why gsply?**
- **Blazing Fast**: Zero-copy reads by default, Numba JIT-accelerated I/O
- **Pure Python**: NumPy + Numba required, PyTorch optional for GPU
- **Format Support**: Native Gaussian Splatting PLY + PlayCanvas compressed format
- **Auto-Detection**: Automatically detects format and SH degree
- **GPU Acceleration**: Optional PyTorch integration with GSTensor for training/inference

---

## Features

- **Fastest Gaussian PLY I/O**: Peak performance of 93M Gaussians/sec read, 87M Gaussians/sec write
  - **100K Gaussians, SH0 (uncompressed)**: Read 1.1ms (89M/s), Write 1.15ms (87M/s optimized)
  - **400K Gaussians, SH0 (uncompressed)**: Read 5.7ms (70M/s), Write 7.5ms (53M/s optimized)
  - **1M Gaussians, SH0 (uncompressed)**: Read 11.2ms (90M/s), Write 32ms (31M/s optimized)
  - **100K Gaussians, SH3 (uncompressed)**: Read 4.8ms (21M/s), Write 53ms (1.9M/s)
  - **1M Gaussians, SH3 (uncompressed)**: Read 59.8ms (17M/s), Write 697ms (1.4M/s)
  - **400K Gaussians, SH0 (compressed)**: Read 4.4ms (91M/s) - **71% smaller**
  - **Real-world data (390K Gaussians)**: Read average 75M/s (peak 93M/s)
  - **Write optimization**: 1.9-2.9x faster with consolidated data (read->write workflows)
  - **Verified equivalent**: Output files match plyfile exactly (byte-for-byte validation)
  - **Optimizations**: Zero-copy reads, zero-copy writes (with _base), Numba JIT parallel processing, LRU header caching
- **Zero-copy optimization**: Enabled by default for maximum performance
- **Pure Python**: NumPy + Numba required for JIT acceleration (no C++ compilation needed)
- **GPU Acceleration**: Optional PyTorch integration with `GSTensor` for training/inference
  - **11x faster GPU transfer** when using consolidated data
  - **Zero-copy GPU slicing** for memory efficiency
  - **Gradient tracking** support for training workflows
- **Multiple SH degrees**: Supports SH degrees 0-3 (14, 23, 38, 59 properties)
- **Auto-format detection**: Automatically detects uncompressed vs compressed formats
- **In-memory compression/decompression**: APIs for compressing and decompressing bytes without disk I/O
- **Type-safe**: Full type hints for Python 3.10+

---

## Installation

### Basic Installation

```bash
pip install gsply
```

**Required dependencies**: NumPy and Numba are always installed for optimal performance.

### Optional: PyTorch for GPU Acceleration

PyTorch features (`GSTensor`) are included in gsply but only available when PyTorch is installed. Install PyTorch separately if you need GPU acceleration:

```bash
# Install PyTorch with CUDA support (adjust version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Or with uv
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Or CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

gsply will automatically detect PyTorch and enable `GSTensor` if available.

### From Source

```bash
git clone https://github.com/OpsiClear/gsply.git
cd gsply
pip install -e .

# Optional: Install PyTorch for GPU features
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Quick Start

```python
from gsply import plyread, plywrite, detect_format

# Read PLY file (auto-detects format) - returns GSData container
# Zero-copy optimization for maximum speed (up to 78M Gaussians/sec)
data = plyread("model.ply")
print(f"Loaded {data.means.shape[0]} Gaussians")

# Access via attributes
positions = data.means
colors = data.sh0

# Unpack for standard GS workflows (NEW!)
means, scales, quats, opacities, sh0, shN = data.unpack()
# Now compatible with any rendering function expecting individual arrays
render(means, scales, quats, opacities, sh0)

# Or get as dictionary
props = data.to_dict()
render(**props)  # Unpack dict as kwargs

# Write uncompressed PLY file
plywrite("output.ply", data.means, data.scales, data.quats,
         data.opacities, data.sh0, data.shN)

# Write compressed PLY file (saves as "output.compressed.ply", 14.5x smaller)
plywrite("output.ply", data.means, data.scales, data.quats,
         data.opacities, data.sh0, data.shN, compressed=True)

# NEW: Compress/decompress without disk I/O (clean API!)
from gsply import compress_to_bytes, decompress_from_bytes
compressed = compress_to_bytes(data)  # Compress to bytes
data_restored = decompress_from_bytes(compressed)  # Decompress from bytes
# Perfect for network transfer, database storage, streaming, etc.

# Detect format before reading
is_compressed, sh_degree = detect_format("model.ply")
print(f"Compressed: {is_compressed}, SH degree: {sh_degree}")

# Optional: GPU acceleration with PyTorch (if torch is installed)
try:
    from gsply import GSTensor

    # Transfer to GPU (11x faster with consolidated data)
    gstensor = GSTensor.from_gsdata(data, device='cuda')

    # GPU operations
    positions_gpu = gstensor.means  # torch.Tensor on GPU
    subset_gpu = gstensor[::10]      # GPU slicing (zero-copy)

    # Back to CPU
    data_cpu = gstensor.to_gsdata()
except ImportError:
    print("PyTorch not installed, skipping GPU features")
```

---

## API Reference

**Quick Navigation:**
- [Core I/O](#core-io)
  - [`plyread()`](#plyreadfile_path) - Read PLY files
  - [`plywrite()`](#plywritefile_path-means-scales-quats-opacities-sh0-shn-compressedfalse) - Write PLY files
  - [`detect_format()`](#detect_formatfile_path) - Detect format and SH degree
- [GSData](#gsdata) - CPU dataclass container
  - [`data.unpack()`](#dataunpackinclude_shntrue) - Unpack to tuple
  - [`data.to_dict()`](#datato_dict) - Convert to dictionary
  - [`data.copy()`](#datacopy) - Deep copy
  - [`data.consolidate()`](#dataconsolidate) - Optimize for slicing
  - [`data[index]`](#dataindex) - Indexing and slicing
  - [`len(data)`](#lendata) - Get number of Gaussians
- [Compression APIs](#compression-apis)
  - [`compress_to_bytes()`](#compress_to_bytesdata) - Compress to bytes
  - [`compress_to_arrays()`](#compress_to_arraysdata) - Compress to arrays
  - [`decompress_from_bytes()`](#decompress_from_bytescompressed_bytes) - Decompress bytes
- [Utility Functions](#utility-functions)
  - [`sh2rgb()`](#sh2rgbsh) - SH to RGB conversion
  - [`rgb2sh()`](#rgb2shrgb) - RGB to SH conversion
  - [`SH_C0`](#sh_c0) - SH normalization constant
- [GSTensor (GPU)](#gstensor---gpu-accelerated-dataclass) - PyTorch integration
  - [`GSTensor.from_gsdata()`](#gstensorfrom_gsdatadata-devicecuda-dtypetorchfloat32-requires_gradfalse) - Convert to GPU
  - [`gstensor.to_gsdata()`](#gstensorto_gsdata) - Convert to CPU
  - [`gstensor.to()`](#gstensortodevicenonedtypenone) - Device/dtype transfer
  - [`gstensor.cpu()` / `cuda()`](#gstensorcpu) - Device shortcuts
  - [`gstensor.half()` / `float()` / `double()`](#gstensorhalf-gstensorfloat-gstensordouble) - Precision conversion
  - [`gstensor.consolidate()`](#gstensorconsolidate) - Optimize for slicing
  - [`gstensor.clone()`](#gstensorclone) - Deep copy
  - [`gstensor.unpack()`](#gstensorunpackinclude_shntrue) - Unpack to tuple
  - [`gstensor.to_dict()`](#gstensorto_dict) - Convert to dictionary
  - [`gstensor[index]`](#gstensorindex) - Indexing and slicing
  - [`len(gstensor)`](#lengstensor) - Get number of Gaussians
  - [Properties & Helpers](#gstensordevice-property) - `device`, `dtype`, `get_sh_degree()`, `has_high_order_sh()`

---

## Core I/O

### `plyread(file_path)`

Read Gaussian Splatting PLY file (auto-detects format).

Always uses zero-copy optimization for maximum performance.

**Parameters:**
- `file_path` (str | Path): Path to PLY file

**Returns:**
`GSData` dataclass with Gaussian parameters:
- `means`: (N, 3) - Gaussian centers
- `scales`: (N, 3) - Log scales
- `quats`: (N, 4) - Rotations as quaternions (wxyz)
- `opacities`: (N,) - Logit opacities
- `sh0`: (N, 3) - DC spherical harmonics
- `shN`: (N, K, 3) - Higher-order SH coefficients (K=0 for degree 0, K=9 for degree 1, etc.)
- `masks`: (N,) - Boolean mask for filtering Gaussians
- `_base`: (N, P) - Internal array for zero-copy views (private)

**Performance:**
- Uncompressed: 5.7ms for 400K Gaussians (70M/sec), 12.8ms for 1M (78M/sec peak)
- Compressed: 8.5ms for 400K Gaussians (47M/sec), 16.7ms for 1M (60M/sec)
- Scales linearly with data size

**Example:**
```python
from gsply import plyread

# Zero-copy reading - up to 78M Gaussians/sec
data = plyread("model.ply")
print(f"Loaded {data.means.shape[0]} Gaussians with SH degree {data.shN.shape[1]}")

# Access via attributes
positions = data.means
colors = data.sh0

# Unpack for standard GS workflows
means, scales, quats, opacities, sh0, shN = data.unpack()

# Or exclude shN for SH0 data
means, scales, quats, opacities, sh0 = data.unpack(include_shN=False)

# Or get as dictionary
props = data.to_dict()
```

---

### `plywrite(file_path, means, scales, quats, opacities, sh0, shN=None, compressed=False)`

Write Gaussian Splatting PLY file.

**Parameters:**
- `file_path` (str | Path): Output PLY file path (auto-adjusted to `.compressed.ply` if `compressed=True`)
- `means` (np.ndarray): Shape (N, 3) - Gaussian centers
- `scales` (np.ndarray): Shape (N, 3) - Log scales
- `quats` (np.ndarray): Shape (N, 4) - Rotations as quaternions (wxyz)
- `opacities` (np.ndarray): Shape (N,) - Logit opacities
- `sh0` (np.ndarray): Shape (N, 3) - DC spherical harmonics
- `shN` (np.ndarray, optional): Shape (N, K, 3) or (N, K*3) - Higher-order SH
- `compressed` (bool): If True, write compressed format and auto-adjust extension

**Format Selection:**
- `compressed=False` or `.ply` extension -> Uncompressed format (fast)
- `compressed=True` -> Compressed format, saves as `.compressed.ply` automatically
- `.compressed.ply` or `.ply_compressed` extension -> Compressed format

**Performance:**
- Uncompressed SH0: 1.15ms for 100K (87M/s), 7.5ms for 400K (53M/s), 32ms for 1M (31M/s) - optimized path
- Uncompressed SH0 (fallback): 3.3ms for 100K (30M/s), 21ms for 400K (19M/s), 61ms for 1M (16M/s) - without optimization
- Uncompressed SH3: 53ms for 100K (1.9M/s), 697ms for 1M (1.4M/s)
- Compressed: ~63ms for 400K Gaussians (6.3M Gaussians/sec), 3.4x smaller files
- 1.9-2.9x faster when writing data loaded from PLY (zero-copy optimization)

**Example:**
```python
from gsply import plywrite

# Write uncompressed (fast, ~8ms for 400K Gaussians)
plywrite("output.ply", means, scales, quats, opacities, sh0, shN)

# Write compressed (saves as "output.compressed.ply", ~63ms, 3.4x smaller)
plywrite("output.ply", means, scales, quats, opacities, sh0, shN, compressed=True)
```

---

### `detect_format(file_path)`

Detect PLY format type and SH degree.

**Parameters:**
- `file_path` (str | Path): Path to PLY file

**Returns:**
Tuple of (is_compressed, sh_degree):
- `is_compressed` (bool): True if compressed format
- `sh_degree` (int | None): 0-3 for uncompressed, None for compressed/unknown

**Example:**
```python
from gsply import detect_format

is_compressed, sh_degree = detect_format("model.ply")
if is_compressed:
    print("Compressed PlayCanvas format")
else:
    print(f"Uncompressed format with SH degree {sh_degree}")
```

---

## GSData

Container dataclass for Gaussian Splatting data with zero-copy optimization.

`GSData` is returned by `plyread()` and provides efficient access to Gaussian parameters through both direct attributes and convenience methods. All arrays are mutable and can be modified in-place. Arrays can be views into a shared `_base` array for maximum performance (zero memory overhead).

**Attributes:**

- `means` (np.ndarray): Shape (N, 3) - Gaussian centers (xyz positions)
- `scales` (np.ndarray): Shape (N, 3) - Log scales for each axis
- `quats` (np.ndarray): Shape (N, 4) - Rotations as quaternions (wxyz order)
- `opacities` (np.ndarray): Shape (N,) - Logit opacities (before sigmoid)
- `sh0` (np.ndarray): Shape (N, 3) - DC spherical harmonics (RGB color basis)
- `shN` (np.ndarray | None): Shape (N, K, 3) - Higher-order SH coefficients
  - K=0 for SH degree 0 (no higher-order)
  - K=9 for SH degree 1
  - K=24 for SH degree 2
  - K=45 for SH degree 3
- `masks` (np.ndarray): Shape (N,) boolean - Mask for filtering (initialized to all True)
- `_base` (np.ndarray | None): Shape (N, P) - Private base array (auto-managed, do not modify)

**Example:**
```python
from gsply import plyread

data = plyread("scene.ply")
print(f"Loaded {len(data)} Gaussians")

# Direct attribute access
positions = data.means
colors = data.sh0

# Mutable - modify in place
data.means[0] = [1, 2, 3]
data.sh0 *= 1.5  # Make brighter
```

---

### `data.unpack(include_shN=True)`

Unpack Gaussian data into tuple of individual arrays.

Most useful for passing data to rendering functions that expect separate arrays rather than a container object.

**Parameters:**
- `include_shN` (bool): If True, include shN in output (default: True)

**Returns:**
- If `include_shN=True`: `(means, scales, quats, opacities, sh0, shN)`
- If `include_shN=False`: `(means, scales, quats, opacities, sh0)`

**Example:**
```python
data = plyread("scene.ply")

# Full unpacking (recommended for SH1-3)
means, scales, quats, opacities, sh0, shN = data.unpack()
render(means, scales, quats, opacities, sh0, shN)

# Without higher-order SH (recommended for SH0)
means, scales, quats, opacities, sh0 = data.unpack(include_shN=False)
render(means, scales, quats, opacities, sh0)

# Tuple unpacking for plywrite
plywrite("output.ply", *data.unpack())
```

---

### `data.to_dict()`

Convert Gaussian data to dictionary for keyword argument unpacking.

Useful when calling functions that accept keyword arguments matching the Gaussian parameter names.

**Returns:**
- Dictionary with keys: `means`, `scales`, `quats`, `opacities`, `sh0`, `shN`

**Example:**
```python
data = plyread("scene.ply")

# Dictionary unpacking
props = data.to_dict()
render(**props)  # Unpack as kwargs

# Access by key
positions = props['means']
colors = props['sh0']
```

---

### `data.copy()`

Create deep copy of GSData with independent arrays.

Modifications to the copy will not affect the original data. Optimized to use `_base` array when available (faster than copying individual arrays).

**Returns:**
- `GSData`: New GSData object with copied arrays

**Example:**
```python
data = plyread("scene.ply")

# Create independent copy
data_copy = data.copy()
data_copy.means[0] = 0  # Doesn't affect original

# Use for creating variations
bright = data.copy()
bright.sh0 *= 1.5  # Make brighter
```

---

### `data.consolidate()`

Consolidate separate arrays into single base array for faster slicing operations.

Creates a `_base` array from separate arrays, which improves performance for boolean masking operations (1.5x faster). Only beneficial if you plan to perform many boolean mask operations on the same data.

**Returns:**
- `GSData`: New GSData with `_base` array, or self if already consolidated

**Performance:**
- One-time cost: ~2ms per 100K Gaussians
- Benefit: 1.5x faster boolean masking
- Most useful before multiple filter operations

**Example:**
```python
data = plyread("scene.ply")

# Consolidate for faster filtering
data_consolidated = data.consolidate()

# Now boolean masking is 1.5x faster
high_opacity = data_consolidated[data_consolidated.opacities > 0.5]
low_opacity = data_consolidated[data_consolidated.opacities <= 0.5]
```

---

### `data[index]`

Slice GSData using standard Python indexing.

Supports integers, slices, boolean masks, and fancy indexing. Returns views when possible (zero-copy).

**Indexing Modes:**
- Integer: `data[0]` - Returns tuple of (means, scales, quats, opacities, sh0, shN, masks)
- Slice: `data[100:200]` - Returns new GSData with subset
- Step: `data[::10]` - Returns every 10th Gaussian
- Boolean mask: `data[mask]` - Filter by boolean array
- Fancy: `data[[0, 10, 20]]` - Select specific indices

**Example:**
```python
data = plyread("scene.ply")

# Single Gaussian (returns tuple)
means, scales, quats, opacities, sh0, shN, masks = data[0]

# Slice (returns GSData)
subset = data[100:200]

# Boolean mask (returns GSData)
high_opacity = data[data.opacities > 0.5]

# Step slicing (returns GSData)
every_10th = data[::10]
```

---

### `len(data)`

Get number of Gaussians in the dataset.

**Returns:**
- `int`: Number of Gaussians (equivalent to `data.means.shape[0]`)

**Example:**
```python
data = plyread("scene.ply")
print(f"Loaded {len(data)} Gaussians")
```

---

## Compression APIs

### `compress_to_bytes(data)`

Compress Gaussian splatting data to bytes (PlayCanvas format) without writing to disk.

Useful for network transfer, streaming, or custom storage solutions.

**Parameters:**
- `data` (GSData): Gaussian data from `plyread()` or created manually
  - Alternative: Pass individual arrays for backward compatibility

**Returns:**
`bytes`: Complete compressed PLY file as bytes

**Example:**
```python
from gsply import plyread, compress_to_bytes

# Method 1: Clean API with GSData (recommended)
data = plyread("model.ply")
compressed_bytes = compress_to_bytes(data)  # Simple!

# Method 2: Individual arrays (backward compatible)
compressed_bytes = compress_to_bytes(
    means, scales, quats, opacities, sh0, shN
)

# Send over network or store in database
with open("output.compressed.ply", "wb") as f:
    f.write(compressed_bytes)
```

---

### `compress_to_arrays(data)`

Compress Gaussian splatting data to component arrays (PlayCanvas format).

Returns separate components for custom processing or partial updates.

**Parameters:**
- `data` (GSData): Gaussian data from `plyread()` or created manually
  - Alternative: Pass individual arrays for backward compatibility

**Returns:**
Tuple containing:
- `header_bytes` (bytes): PLY header as bytes
- `chunk_bounds` (np.ndarray): Shape (num_chunks, 18) float32 - Chunk boundary array
- `packed_data` (np.ndarray): Shape (N, 4) uint32 - Main compressed data
- `packed_sh` (np.ndarray | None): Shape varies, uint8 - Compressed SH data if present

**Example:**
```python
from gsply import plyread, compress_to_arrays
from io import BytesIO

# Method 1: Clean API with GSData (recommended)
data = plyread("model.ply")
header, chunks, packed, sh = compress_to_arrays(data)  # Simple!

# Method 2: Individual arrays (backward compatible)
header, chunks, packed, sh = compress_to_arrays(
    means, scales, quats, opacities, sh0, shN
)

# Process components individually
print(f"Header size: {len(header)} bytes")
print(f"Chunks: {chunks.shape[0]} chunks")
print(f"Packed data: {packed.nbytes} bytes")

# Manually assemble if needed
buffer = BytesIO()
buffer.write(header)
buffer.write(chunks.tobytes())
buffer.write(packed.tobytes())
if sh is not None:
    buffer.write(sh.tobytes())

compressed_bytes = buffer.getvalue()
```

---

### `decompress_from_bytes(compressed_bytes)`

Decompress Gaussian splatting data from bytes (PlayCanvas format) without reading from disk.

Symmetric with `compress_to_bytes()` - perfect for network transfer, streaming, or custom storage.

**Parameters:**
- `compressed_bytes` (bytes): Complete compressed PLY file as bytes

**Returns:**
`GSData` dataclass with decompressed Gaussian parameters:
- `means`: (N, 3) - Gaussian centers
- `scales`: (N, 3) - Log scales
- `quats`: (N, 4) - Rotations as quaternions (wxyz)
- `opacities`: (N,) - Logit opacities
- `sh0`: (N, 3) - DC spherical harmonics
- `shN`: (N, K, 3) - Higher-order SH coefficients
- `masks`: (N,) - Boolean mask (all True for decompressed data)
- `_base`: None (not applicable for decompressed data)

**Example:**
```python
from gsply import compress_to_bytes, decompress_from_bytes, plyread

# Example 1: Round-trip without disk I/O
data = plyread("model.ply")
compressed = compress_to_bytes(data)
data_restored = decompress_from_bytes(compressed)
# data_restored is ready to use!

# Example 2: Network transfer
# Sender side
compressed_bytes = compress_to_bytes(data)
# send compressed_bytes over network...

# Receiver side
# ...receive compressed_bytes from network
data = decompress_from_bytes(compressed_bytes)
# No temporary files needed!

# Example 3: Database storage
import sqlite3
conn = sqlite3.connect('gaussians.db')
conn.execute('CREATE TABLE IF NOT EXISTS models (id INTEGER, data BLOB)')
# Store
compressed = compress_to_bytes(data)
conn.execute('INSERT INTO models VALUES (?, ?)', (1, compressed))
# Retrieve
row = conn.execute('SELECT data FROM models WHERE id = 1').fetchone()
data_restored = decompress_from_bytes(row[0])
```

**Note:** PlayCanvas compression is lossy (quantization). Decompressed data will be very close to but not exactly identical to the original.

---

## Utility Functions

### `sh2rgb(sh)`

Convert spherical harmonic DC coefficients to RGB colors.

Converts the DC component (sh0) of spherical harmonics to standard RGB color values in the range [0, 1]. Useful for visualization and color manipulation.

**Parameters:**
- `sh` (np.ndarray | float): SH DC coefficients - Shape (N, 3) or scalar

**Returns:**
- `np.ndarray | float`: RGB colors in [0, 1] range

**Example:**
```python
from gsply import plyread, sh2rgb

data = plyread("scene.ply")

# Convert SH to RGB for visualization
rgb_colors = sh2rgb(data.sh0)
print(f"First color: RGB({rgb_colors[0, 0]:.3f}, {rgb_colors[0, 1]:.3f}, {rgb_colors[0, 2]:.3f})")

# Modify colors in RGB space
rgb_colors *= 1.5  # Make brighter
data.sh0 = rgb2sh(np.clip(rgb_colors, 0, 1))  # Convert back
```

---

### `rgb2sh(rgb)`

Convert RGB colors to spherical harmonic DC coefficients.

Converts standard RGB color values in the range [0, 1] to the DC component (sh0) of spherical harmonics. Inverse of `sh2rgb()`.

**Parameters:**
- `rgb` (np.ndarray | float): RGB colors in [0, 1] range - Shape (N, 3) or scalar

**Returns:**
- `np.ndarray | float`: SH DC coefficients

**Example:**
```python
from gsply import rgb2sh, plywrite
import numpy as np

# Create Gaussians with specific RGB colors
n = 1000
means = np.random.randn(n, 3).astype(np.float32)
scales = np.ones((n, 3), dtype=np.float32) * 0.01
quats = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
opacities = np.ones(n, dtype=np.float32)

# Set colors in RGB space
rgb_colors = np.random.rand(n, 3).astype(np.float32)  # Random colors
sh0 = rgb2sh(rgb_colors)  # Convert to SH

plywrite("colored.ply", means, scales, quats, opacities, sh0, None)
```

---

### `SH_C0`

Constant for spherical harmonic DC coefficient normalization.

This constant (0.28209479177387814) is used in the conversion between SH coefficients and RGB colors. It represents the normalization factor for the 0th order spherical harmonic.

**Type:** `float`

**Value:** `0.28209479177387814`

**Example:**
```python
from gsply import SH_C0

# Manual conversion (equivalent to sh2rgb/rgb2sh)
rgb = sh * SH_C0 + 0.5  # SH to RGB
sh = (rgb - 0.5) / SH_C0  # RGB to SH
```

---

## GPU Support (PyTorch)

**Optional GPU acceleration** with PyTorch tensors for training and inference workflows.

### Installation

PyTorch is **optional**. `GSTensor` features are always included in gsply but only work when PyTorch is installed.

```bash
# Install gsply first
pip install gsply

# Then install PyTorch if you need GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

gsply will automatically detect PyTorch and enable `GSTensor` if available. Without PyTorch, gsply works normally for CPU-only workflows.

### GSTensor - GPU-Accelerated Dataclass

`GSTensor` is a PyTorch-backed version of `GSData` that enables GPU-accelerated operations:

```python
from gsply import plyread, GSTensor

# Load data from disk (CPU NumPy)
data = plyread("model.ply")

# Convert to GPU tensors (11x faster with _base optimization)
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Access GPU tensors
positions_gpu = gstensor.means  # torch.Tensor on GPU
colors_gpu = gstensor.sh0       # torch.Tensor on GPU

# Unpack for rendering functions (NEW!)
means, scales, quats, opacities, sh0, shN = gstensor.unpack()
rendered = render_gaussians(means, scales, quats, opacities, sh0)

# Or use dict unpacking
rendered = render_gaussians(**gstensor.to_dict())

# Slice on GPU (zero-cost views)
subset = gstensor[100:200]      # Returns GSTensor view

# Training workflow
gstensor_trainable = GSTensor.from_gsdata(data, device='cuda', requires_grad=True)
loss = render_loss(gstensor_trainable.means, ...)
loss.backward()

# Convert back to CPU NumPy
data_cpu = gstensor.to_gsdata()
```

### Key Features

- **11x Faster GPU Transfer**: When data has `_base` (from `plyread()` or `consolidate()`), GPU transfer is 11x faster than manual stacking
- **Zero-Copy Views**: GPU slicing creates views (no memory overhead)
- **Device Management**: Seamless transfer between CPU/GPU with `.to()`, `.cpu()`, `.cuda()`
- **Training Support**: Optional gradient tracking with `requires_grad=True`
- **Type Conversions**: `half()`, `float()`, `double()` for precision control
- **Optimized Slicing**: 25x faster boolean masking with `consolidate()`

### Performance

**GPU Transfer (400K Gaussians, SH0, RTX 3090 Ti):**
- **With `_base` optimization**: 1.99 ms (zero CPU copy overhead)
- **Without `_base` (fallback)**: 22.78 ms (requires CPU stacking)
- **Speedup**: 11.4x faster with `_base`

**Memory Efficiency:**
- Single tensor transfer vs 5 separate transfers
- 50% less I/O (no CPU copy when using `_base`)
- GPU views are free (zero additional memory)

### API Reference

#### `GSTensor.from_gsdata(data, device='cuda', dtype=torch.float32, requires_grad=False)`

Convert `GSData` to `GSTensor`.

**Parameters:**
- `data` (GSData): Input Gaussian data
- `device` (str | torch.device): Target device ('cuda', 'cpu', or torch.device)
- `dtype` (torch.dtype): Target dtype (default: torch.float32)
- `requires_grad` (bool): Enable gradient tracking (default: False)

**Returns:**
- `GSTensor`: GPU-accelerated tensor container

**Example:**
```python
# Fast path (uses _base if available)
gstensor = GSTensor.from_gsdata(data, device='cuda')

# For training
gstensor = GSTensor.from_gsdata(data, device='cuda', requires_grad=True)

# Half precision for memory savings
gstensor = GSTensor.from_gsdata(data, device='cuda', dtype=torch.float16)
```

---

#### `gstensor.to_gsdata()`

Convert `GSTensor` back to `GSData` (CPU NumPy).

**Returns:**
- `GSData`: CPU NumPy container

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')
# ... GPU operations ...
data_cpu = gstensor.to_gsdata()  # Back to NumPy
```

---

#### `gstensor.to(device=None, dtype=None)`

Move tensors to different device and/or dtype.

**Parameters:**
- `device` (str | torch.device, optional): Target device
- `dtype` (torch.dtype, optional): Target dtype

**Returns:**
- `GSTensor`: New GSTensor on target device/dtype

**Example:**
```python
gstensor_gpu = gstensor.to('cuda')
gstensor_half = gstensor.to(dtype=torch.float16)
gstensor_gpu_half = gstensor.to('cuda', dtype=torch.float16)
```

---

#### `gstensor.consolidate()`

Create `_base` tensor for 25x faster slicing.

**Returns:**
- `GSTensor`: New GSTensor with `_base` tensor

**Example:**
```python
# Consolidate for faster slicing
gstensor = gstensor.consolidate()

# Boolean masking is now 25x faster
mask = gstensor.opacities > 0.5
subset = gstensor[mask]  # Fast with _base
```

---

#### `gstensor.clone()`

Create independent deep copy.

**Returns:**
- `GSTensor`: Cloned GSTensor

**Example:**
```python
gstensor_copy = gstensor.clone()
gstensor_copy.means[0] = 0  # Doesn't affect original
```

---

#### `gstensor.cpu()`

Move tensors to CPU.

Shorthand for `gstensor.to('cpu')`.

**Returns:**
- `GSTensor`: GSTensor on CPU

**Example:**
```python
gstensor_gpu = GSTensor.from_gsdata(data, device='cuda')
gstensor_cpu = gstensor_gpu.cpu()  # Now on CPU
```

---

#### `gstensor.cuda(device=None)`

Move tensors to GPU.

Shorthand for `gstensor.to('cuda')`.

**Parameters:**
- `device` (int | None): GPU device index (default: None = cuda:0)

**Returns:**
- `GSTensor`: GSTensor on GPU

**Example:**
```python
gstensor_gpu = gstensor.cuda()  # Move to cuda:0
gstensor_gpu1 = gstensor.cuda(1)  # Move to cuda:1
```

---

#### `gstensor.half()`, `gstensor.float()`, `gstensor.double()`

Convert tensor precision.

Convenience methods for dtype conversion:
- `half()` - Convert to `torch.float16`
- `float()` - Convert to `torch.float32`
- `double()` - Convert to `torch.float64`

**Returns:**
- `GSTensor`: GSTensor with new dtype

**Example:**
```python
# Half precision for memory savings (2x less VRAM)
gstensor_fp16 = gstensor.half()

# Back to full precision
gstensor_fp32 = gstensor_fp16.float()

# Double precision for high accuracy
gstensor_fp64 = gstensor.double()
```

---

#### `gstensor.unpack(include_shN=True)`

Unpack GSTensor into tuple of individual tensors.

Identical to `GSData.unpack()` but returns PyTorch tensors instead of NumPy arrays.

**Parameters:**
- `include_shN` (bool): If True, include shN in output (default: True)

**Returns:**
- If `include_shN=True`: `(means, scales, quats, opacities, sh0, shN)`
- If `include_shN=False`: `(means, scales, quats, opacities, sh0)`

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Full unpacking for rendering
means, scales, quats, opacities, sh0, shN = gstensor.unpack()
rendered = render_gaussians(means, scales, quats, opacities, sh0, shN)

# Without higher-order SH
means, scales, quats, opacities, sh0 = gstensor.unpack(include_shN=False)
```

---

#### `gstensor.to_dict()`

Convert GSTensor to dictionary for keyword argument unpacking.

Identical to `GSData.to_dict()` but returns PyTorch tensors instead of NumPy arrays.

**Returns:**
- Dictionary with keys: `means`, `scales`, `quats`, `opacities`, `sh0`, `shN`

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Dictionary unpacking
props = gstensor.to_dict()
rendered = render_gaussians(**props)
```

---

#### `gstensor[index]`

Slice GSTensor using standard Python indexing.

Supports integers, slices, boolean masks, and fancy indexing. Returns views when possible (zero-copy on GPU).

**Indexing Modes:**
- Integer: `gstensor[0]` - Returns tuple of tensors
- Slice: `gstensor[100:200]` - Returns new GSTensor with subset
- Step: `gstensor[::10]` - Returns every 10th Gaussian
- Boolean mask: `gstensor[mask]` - Filter by boolean tensor
- Fancy: `gstensor[[0, 10, 20]]` - Select specific indices

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Single Gaussian (returns tuple)
means, scales, quats, opacities, sh0, shN, masks = gstensor[0]

# Slice (returns GSTensor view - zero memory cost)
subset = gstensor[100:200]

# Boolean mask (returns GSTensor)
high_opacity = gstensor[gstensor.opacities > 0.5]

# Step slicing (returns GSTensor)
every_10th = gstensor[::10]
```

---

#### `len(gstensor)`

Get number of Gaussians.

**Returns:**
- `int`: Number of Gaussians (equivalent to `gstensor.means.shape[0]`)

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')
print(f"Processing {len(gstensor)} Gaussians on GPU")
```

---

#### `gstensor.device` (property)

Get current device of tensors.

**Returns:**
- `torch.device`: Current device (e.g., `torch.device('cuda:0')` or `torch.device('cpu')`)

**Example:**
```python
print(f"Tensors are on {gstensor.device}")
if gstensor.device.type == 'cuda':
    print(f"Using GPU {gstensor.device.index}")
```

---

#### `gstensor.dtype` (property)

Get current dtype of tensors.

**Returns:**
- `torch.dtype`: Current dtype (e.g., `torch.float32`, `torch.float16`)

**Example:**
```python
print(f"Using precision: {gstensor.dtype}")
```

---

#### `gstensor.get_sh_degree()`

Get spherical harmonic degree from data shape.

**Returns:**
- `int`: SH degree (0-3)

**Example:**
```python
sh_degree = gstensor.get_sh_degree()
print(f"Data has SH degree {sh_degree}")
```

---

#### `gstensor.has_high_order_sh()`

Check if data has higher-order spherical harmonics.

**Returns:**
- `bool`: True if SH degree > 0

**Example:**
```python
if gstensor.has_high_order_sh():
    print("Has higher-order SH coefficients")
else:
    print("Only DC component (SH0)")
```

---

### Complete Workflow Examples

#### Training Workflow

```python
import gsply
from gsply import GSTensor
import torch

# Load from disk
data = gsply.plyread("scene.ply")  # Has _base -> fast GPU transfer

# Transfer to GPU (11x faster with _base)
gstensor = GSTensor.from_gsdata(data, device='cuda', requires_grad=True)

# Training loop
optimizer = torch.optim.Adam([gstensor.means, gstensor.scales], lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()

    # Unpack for rendering (cleaner API)
    means, scales, quats, opacities, sh0, shN = gstensor.unpack()
    loss = render_gaussians(means, scales, quats, opacities, sh0)

    loss.backward()
    optimizer.step()

# Save optimized results
optimized_data = gstensor.to_gsdata()
gsply.plywrite("optimized.ply", optimized_data.means, optimized_data.scales,
               optimized_data.quats, optimized_data.opacities,
               optimized_data.sh0, optimized_data.shN)
```

#### Inference Workflow

```python
import gsply
from gsply import GSTensor
import torch

# Load scene
data = gsply.plyread("scene.ply")

# Transfer to GPU (inference mode, no gradients)
gstensor = GSTensor.from_gsdata(data, device='cuda', requires_grad=False)

# Filter Gaussians by opacity threshold
high_opacity_mask = gstensor.opacities > 0.5
filtered = gstensor[high_opacity_mask]

# Render filtered scene with unpacking
with torch.no_grad():
    means, scales, quats, opacities, sh0, shN = filtered.unpack()
    rendered = render_gaussians(means, scales, quats, opacities, sh0)

# Save filtered version
filtered_data = filtered.to_gsdata()
gsply.plywrite("filtered.ply", filtered_data.means, filtered_data.scales,
               filtered_data.quats, filtered_data.opacities,
               filtered_data.sh0, filtered_data.shN)
```

---

## Performance

### Benchmark Results

Comprehensive performance benchmarks across different file sizes and formats (latest results with dataclass implementation):

**Uncompressed Format Performance**

| Gaussians | SH | Read (ms) | Write (ms) | Write Opt (ms) | Read (M/s) | Write (M/s) | Write Opt (M/s) |
|-----------|----|---------:|-----------:|---------------:|-----------:|------------:|----------------:|
| 100K | 0 | 1.2 | 3.3 | 1.15 | **82.7** | 30.0 | **87.0** |
| 400K | 0 | 5.7 | 21.3 | 7.5 | 70.2 | 18.8 | **53.5** |
| 1M | 0 | 12.2 | 61.5 | 32.2 | **82.1** | 16.3 | **31.1** |
| 100K | 3 | 4.8 | 53.0 | - | 20.9 | 1.9 | - |
| 1M | 3 | 59.8 | 697 | - | 16.7 | 1.4 | - |

**Note**: "Write Opt" is when writing data that was loaded from PLY (has internal _base array for zero-copy)

**Compressed Format Performance**

| Gaussians | SH | Read (ms) | Write (ms) | Read (M/s) | Write (M/s) | Size Reduction |
|-----------|----|---------:|-----------:|-----------:|------------:|---------------:|
| 100K | 0 | - | - | - | - | 71% |
| 100K | 3 | - | 33.0 | - | 3.0 | 74% |
| 400K | 0 | 5.1 | - | **77.9** | - | 71% |
| 1M | 0 | - | 56.6 | - | **17.7** | 71% |

**Real-World Performance (D:/4D/all_plys - 390K Gaussians, SH0)**

| File | Read (ms) | Peak (ms) | Avg (M/s) | Peak (M/s) |
|------|----------:|----------:|----------:|-----------:|
| frame_0.ply | 5.4 | 4.9 | 71.9 | 79.0 |
| frame_10.ply | 5.7 | 4.5 | 68.8 | 87.1 |
| frame_50.ply | 4.6 | 4.3 | 85.7 | **93.0** |
| **Average** | **5.2** | **4.6** | **75.5** | **86.4** |

### Key Performance Highlights

- **Peak Read Speed**: 93M Gaussians/sec (real-world data, 390K Gaussians)
- **Average Real-World**: 75.5M Gaussians/sec (tested on 90 production files)
- **Peak Write Speed**: 87M Gaussians/sec (100K Gaussians, optimized path)
- **Optimized Write (SH0)**: 87M/s (100K), 53M/s (400K), 31M/s (1M)
- **Write Optimization**: 1.9-2.9x speedup when writing data loaded from PLY (zero-copy)
- **SH3 Write**: 1.9M/s (100K), 1.4M/s (1M) - no optimization (copies needed)
- **Compression Benefits**: 71-74% file size reduction with excellent performance
- **Scalability**: Linear scaling proven up to 1M Gaussians

### Optimization Details

- **Zero-copy reads**: Direct memory views without data duplication
- **Zero-copy writes**: When data has _base array (from plyread), use directly without copying
- **Parallel processing**: Numba JIT compilation with parallel chunk operations
- **Smart caching**: LRU cache for frequently used headers
- **Lookup tables**: Eliminate branching for SH degree detection
- **Fast-path checks**: Skip unnecessary dtype conversions
- **Single file handle**: Reduce file open/close syscall overhead

### Why gsply is Faster

**Read Performance (4.3-8x speedup):**
- **gsply**: Optimized bulk header read + `np.fromfile()` + zero-copy views
  - **Bulk header reading**: Single 8KB read + decode (vs. N readline() calls)
  - Reads entire binary data as contiguous block in one system call
  - Creates memory views directly into the data array (no copies)
  - Base array kept alive via GSData container's reference counting
  - **Consistent performance**: Works equally well on real-world and random data
- **plyfile**: Line-by-line header + individual property accesses per element
  - Multiple readline() + decode operations for header parsing
  - Accesses each property separately through PLY structure
  - Stacks columns together requiring multiple memory allocations and copies
  - Generic PLY parser handles arbitrary formats with overhead
  - **Data-dependent performance**: 10x slower on random/synthetic data vs real-world structured data

**Write Performance (1.9-2.9x speedup with optimization):**
- **gsply (optimized path)**: Zero-copy when data has _base array
  - **Zero-copy write**: When data from plyread() has _base array, use directly (no copying)
  - **66% less memory I/O**: Skip data preparation step entirely
  - Typical after read->write workflows (data already has _base from plyread)
  - Performance (SH0): 87M Gaussians/sec (100K), 53M Gaussians/sec (400K), 31M Gaussians/sec (1M)
  - Only works for SH0 (14 properties) - _base optimization not implemented for SH1-3 yet
- **gsply (fallback path)**: Pre-computed templates + pre-allocated array + buffered I/O
  - **Pre-computed header templates**: Avoids dynamic string building in loops
  - **Buffered I/O**: 2MB buffer for large files reduces system call overhead
  - Allocates single contiguous array with exact dtype needed
  - Fills array via direct slice assignment (no intermediate structures)
  - Used when data created from scratch (no _base array) or for SH1-3
  - Performance (SH0): 30M Gaussians/sec (100K), 19M Gaussians/sec (400K), 16M Gaussians/sec (1M)
  - Performance (SH3): 1.9M Gaussians/sec (100K), 1.4M Gaussians/sec (1M)
- **plyfile**: Dynamic header + per-property assignments + PLY construction
  - Builds header dynamically with loop + f-string formatting
  - Creates PLY element structure with per-property descriptors
  - Assigns each property individually through PLY abstraction layer
  - Additional overhead from generic format handling

**Key Insight**: gsply's performance comes from recognizing that Gaussian Splatting PLY files follow a fixed format, allowing bulk operations and zero-copy views instead of generic PLY parsing.

---

## Format Support

### Uncompressed PLY

Standard binary little-endian PLY format with Gaussian Splatting properties:

| SH Degree | Properties | Description |
|-----------|-----------|-------------|
| 0 | 14 | xyz, f_dc(3), opacity, scales(3), quats(4) |
| 1 | 23 | + 9 f_rest coefficients |
| 2 | 38 | + 24 f_rest coefficients |
| 3 | 59 | + 45 f_rest coefficients |

### Compressed PLY (PlayCanvas)

Chunk-based quantized format with automatic extension handling:
- **File extension**: Automatically saves as `.compressed.ply` when `compressed=True`
- **Compression ratio**: 3.4x for SH0 (3.8-14.5x depending on SH degree)
- **Chunk size**: 256 Gaussians per chunk
- **Bit-packed data**: 11-10-11 bits (position/scale), 2+10-10-10 bits (quaternion)
- **Parallel decompression**: 14.74ms for 400K Gaussians (27M Gaussians/sec)
- **Parallel compression**: 63ms for 400K Gaussians (6.3M Gaussians/sec) with radix sort
- **Compatible with**: PlayCanvas, SuperSplat, other WebGL viewers

For format details, see [docs/COMPRESSED_FORMAT.md](docs/COMPRESSED_FORMAT.md).

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/OpsiClear/gsply.git
cd gsply

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=gsply --cov-report=html
```

### Project Structure

```
gsply/
├── src/
│   └── gsply/
│       ├── __init__.py        # Public API
│       ├── gsdata.py          # GSData dataclass
│       ├── reader.py          # PLY reading (uncompressed + compressed)
│       ├── writer.py          # PLY writing (uncompressed + compressed)
│       ├── formats.py         # Format detection and specs
│       ├── torch/             # Optional PyTorch integration
│       │   ├── __init__.py
│       │   └── gstensor.py    # GSTensor GPU dataclass
│       └── py.typed           # PEP 561 type marker
├── tests/                     # Unit tests (169 tests)
├── benchmarks/                # Performance benchmarks
├── docs/                      # Documentation
│   ├── CHANGELOG.md           # Version changelog
│   └── archive/               # Historical documentation
├── .github/                   # CI/CD workflows
├── pyproject.toml             # Package configuration
└── README.md                  # This file
```

---

## Benchmarking

Compare gsply performance against other PLY libraries:

```bash
# Install benchmark dependencies
pip install -e .[benchmark]

# Run benchmark with default settings
python benchmarks/benchmark.py

# Custom test file and iterations
python benchmarks/benchmark.py --config.file path/to/model.ply --config.iterations 20

# Skip write benchmarks
python benchmarks/benchmark.py --config.skip-write
```

The benchmark measures:
- **Read performance**: Time to load PLY file into numpy arrays
- **Write performance**: Time to write numpy arrays to PLY file
- **File sizes**: Comparison of output file sizes
- **Verification**: Output equivalence between libraries

Example output:
```
READ PERFORMANCE (50K Gaussians, SH degree 3)
Library         Time            Speedup
gsply (fast)    2.89ms          baseline (FASTEST)
gsply (safe)    4.75ms          0.61x (1.6x slower than fast)
plyfile         18.23ms         0.16x (6.3x SLOWER)
Open3D          43.10ms         0.07x (14.9x slower)

WRITE PERFORMANCE
Library         Time            Speedup         File Size
gsply           8.72ms          baseline (FASTEST)    11.34MB
plyfile         12.18ms         0.72x (1.4x slower)   11.34MB
Open3D          35.69ms         0.24x (4.1x slower)   1.15MB (XYZ only)
```

---

## Testing

gsply has comprehensive test coverage with 169 passing tests:

```bash
# Run all tests (NumPy/Numba core)
pytest tests/ -v

# Run PyTorch tests (requires torch installed)
pytest tests/ -v -k "torch or gstensor"

# Run specific test file
pytest tests/test_reader.py -v

# Run with coverage report
pytest tests/ -v --cov=gsply --cov-report=html
```

Test categories:
- Core I/O: Format detection, reading, writing, round-trip consistency
- GSData: Dataclass operations, slicing, masking, consolidation
- Compressed format: PlayCanvas compression/decompression
- GSTensor (PyTorch): GPU transfer, slicing, device management, conversions
- Performance: Optimization verification, benchmark validation
- Error handling: Invalid files, malformed data, edge cases

---

## Documentation

gsply includes comprehensive documentation:

- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - Version changelog and release notes
- **[benchmarks/TRANSFER_OPTIMIZATION_ANALYSIS.md](benchmarks/TRANSFER_OPTIMIZATION_ANALYSIS.md)** - GPU transfer optimization analysis
- **[benchmarks/QUICK_REFERENCE.md](benchmarks/QUICK_REFERENCE.md)** - Performance quick reference
- **[docs/archive/](docs/archive/)** - Historical documentation from development phases

---

## CI/CD

gsply includes a complete GitHub Actions CI/CD pipeline:

- **Multi-platform testing**: Ubuntu, Windows, macOS
- **Multi-version testing**: Python 3.10, 3.11, 3.12, 3.13
- **Core + PyTorch testing**: Separate test jobs for NumPy/Numba core and PyTorch integration
- **Automated benchmarking**: Performance tracking on PRs
- **Build verification**: Wheel building and installation testing
- **PyPI publishing**: Trusted publishing on GitHub Release
- **Pip caching**: Fast CI runs with dependency caching

---

## Contributing

Contributions are welcome! Please see [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

**Quick start:**
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run tests and benchmarks
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

If you use gsply in your research, please cite:

```bibtex
@software{gsply2024,
  author = {OpsiClear},
  title = {gsply: Ultra-Fast Gaussian Splatting PLY I/O},
  year = {2024},
  url = {https://github.com/OpsiClear/gsply}
}
```

---

## Related Projects

- **gsplat**: CUDA-accelerated Gaussian Splatting rasterizer
- **nerfstudio**: NeRF training framework with Gaussian Splatting support
- **PlayCanvas SuperSplat**: Web-based Gaussian Splatting viewer
- **3D Gaussian Splatting**: Original paper and implementation

---

<div align="center">

**Made with Python and numpy**

[Report Bug](https://github.com/OpsiClear/gsply/issues) | [Request Feature](https://github.com/OpsiClear/gsply/issues) | [Documentation](docs/PERFORMANCE.md)

</div>
