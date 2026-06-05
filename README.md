<div align="center">

# gsply

### Ultra-Fast Gaussian Splatting PLY + SPZ I/O Library

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/gsply/badge/?version=latest)](https://gsply.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-438%20passed%2C%2022%20skipped-brightgreen.svg)](#testing)

**93M Gaussians/sec read | 57M Gaussians/sec write | Auto-optimized**

[Quick Start](#quick-start) • [Installation](#installation) • [Examples](#examples) • [API Reference](docs/API_REFERENCE.md) • [Performance](#performance)

</div>

---

## Quick Start

```python
from gsply import plyread, GSData, GSTensor

# Read PLY file (auto-detects format, zero-copy)
data = plyread("model.ply")  # Functional API

# Or use object-oriented API
data = GSData.load("model.ply")  # Classmethod

# Access fields
positions = data.means    # (N, 3) xyz coordinates
colors = data.sh0         # (N, 3) RGB colors
scales = data.scales      # (N, 3) scale parameters
rotations = data.quats    # (N, 4) quaternions

# Save PLY file
data.save("output.ply")  # Uncompressed
data.save("output.ply", compressed=True)  # Compressed (71-74% smaller)

# GPU acceleration (optional)
gstensor = GSTensor.load("model.ply", device='cuda')
gstensor.save("output.compressed.ply")  # GPU compression
```

**Performance:** 93M Gaussians/sec read, 57M Gaussians/sec write (400K Gaussians in 6-7ms)

---

## Overview

Ultra-fast Gaussian Splatting PLY and SPZ I/O for Python. Zero-copy reads, auto-optimized writes, optional GPU acceleration.

### Key Features

- **Ultra-Fast**: 93M Gaussians/sec read, 57M Gaussians/sec write
- **Zero-Copy**: Reads use memory views for maximum performance
- **Auto-Optimized**: Writes are 2.6-2.8x faster automatically
- **Format Support**: Uncompressed PLY + PlayCanvas compressed (71-74% smaller) + SOG + Niantic SPZ (v1–v4)
- **GPU Ready**: Optional PyTorch integration with GSTensor (11x faster transfers)
- **Python-first**: NumPy + Numba core with bundled C++ acceleration in v0.4.2+ wheels
- **Object-Oriented API**: `data.save()`, `GSData.load()`, `gstensor.save()`, `GSTensor.load()`
- **Format Conversion**: `normalize()`, `denormalize()` with fused kernels (~8-15x faster)
- **Color Conversion**: `to_rgb()`, `to_sh()` for SH ↔ RGB conversion
- **Comprehensive**: 438 passing tests, 22 skipped optional-environment tests, full type hints, extensive documentation

---

## Installation

### Basic Installation

```bash
pip install gsply
```

**Core dependencies:** NumPy and Numba (automatically installed)

**Current PyPI wheel status (v0.4.2+):** `pip install gsply` installs the
bundled `gsply_cpp` extension automatically on supported platform wheels:
Linux x86_64, Windows AMD64, and macOS arm64 for CPython 3.10-3.13. The Python
backend remains the default at runtime; opt into C++ with
`gsply.use_backend("cpp")` or `gsply.use_backend("auto")`.

### Optional Features

**GPU Acceleration (PyTorch):**
```bash
pip install torch
```
Enables `GSTensor`, `plyread_gpu()`, `plywrite_gpu()`, `read_spz_gpu()`, and
GPU-accelerated format conversions.

**SOG Format Support:**
```bash
pip install gsply[sogs]
```
Enables `sogread()` for reading SOG (Spatially Ordered Gaussians) format files.

**C++ acceleration backend (bundled wheels):**

`pip install gsply` installs the `gsply_cpp` backend automatically from the
published v0.4.2+ platform wheels on supported platforms. No `gsply[cpp]` extra
and no separate `gsply-cpp` package are needed.

```bash
pip install gsply
```

If pip uses the sdist, or if the platform has no matching wheel, the install can
remain Python-only by default. To build the extension locally from the
repository root, opt in explicitly:

```bash
pip install . -Cwheel.cmake=true -Ccmake.define.GSPLY_BUILD_CPP=ON
```

Then opt in at runtime (the Python backend remains the default):

```python
import gsply
gsply.use_backend("cpp")        # or "auto" (use C++ if importable), or "python"
# equivalently: set the env var GSPLY_BACKEND=cpp before importing gsply
print(gsply.active_backend())   # -> "cpp" when gsply_cpp is installed

data = gsply.read_spz("scene.spz")      # routed through C++ when enabled
gsply.write_spz("out.spz", data, version=4)
```

When enabled, `plyread`/`plywrite`/`read_spz`/`write_spz` use `gsply_cpp` where it
applies and fall back to pure Python otherwise (e.g. compressed PLY). Reads are
float32-ULP-identical to the Python path; writes may differ by <=1 LSB per
quantized value and use a different compressor (decoded data matches; compressed
bytes differ).

For non-Python C++ applications, use the standalone CMake target in `cpp/`:
`add_subdirectory(path/to/gsply/cpp)` then link `gsplycpp::core` and include
`gsplycpp.hpp`. See [cpp/README.md](cpp/README.md).

**Full Installation:**
```bash
pip install "gsply[sogs,spz]" torch  # GPU + SOG + SPZ v4 support
```

**Development:**
```bash
git clone https://github.com/OpsiClear/gsply.git
cd gsply
pip install -e .[dev]  # Includes pytest, ruff, mypy
```

---

## Examples

### Basic I/O

```python
from gsply import plyread, plywrite, GSData

# Read PLY file
data = plyread("model.ply")
print(f"Loaded {len(data)} Gaussians")

# Object-oriented API
data = GSData.load("model.ply")  # Auto-detects format
data.save("output.ply")  # Uncompressed
data.save("output.ply", compressed=True)  # Compressed

# Unpack to individual arrays
means, scales, quats, opacities, sh0, shN = data.unpack()

# Write with individual arrays
plywrite("output.ply", means, scales, quats, opacities, sh0, shN)
```

### Format Conversion

```python
from gsply import GSData
import numpy as np

# Load PLY file (log-scales, logit-opacities)
data = GSData.load("scene.ply")

# Convert to linear format for easier manipulation
data.denormalize()  # Uses fused kernel (~8-15x faster)

# Modify in linear space
data.opacities = np.clip(data.opacities * 1.2, 0, 1)
data.scales *= 1.1

# Convert back to PLY format before saving
data.normalize()  # Uses fused kernel (~8-15x faster)
data.save("modified.ply")
```

### GPU Acceleration

```python
from gsply import GSTensor, plyread_gpu, plywrite_gpu, read_spz_gpu

# Direct GPU I/O (4-5x faster than CPU decompress + GPU transfer)
gstensor = plyread_gpu("model.compressed.ply", device='cuda')

# Direct SPZ decode to tensors. Container decompression happens on CPU, then
# packed SPZ sections are uploaded once and unpacked with PyTorch ops.
spz_tensor = read_spz_gpu("model.spz", device='cuda')

# Or convert from CPU
data = GSData.load("model.ply")
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Access GPU tensors
positions_gpu = gstensor.means  # torch.Tensor on GPU
colors_gpu = gstensor.sh0        # torch.Tensor on GPU

# Filter on GPU
high_opacity = gstensor[gstensor.opacities > 0.5]

# GPU format conversion
gstensor.denormalize()  # GPU-accelerated

# Write back (GPU compression)
gstensor.save("output.compressed.ply")  # GPU compression by default
```

### Creating from External Data

```python
from gsply import GSData, GSTensor
import numpy as np
import torch

# Create GSData from arrays with format preset
data = GSData.from_arrays(
    means=np.random.randn(1000, 3),
    scales=np.random.rand(1000, 3),
    quats=np.random.randn(1000, 4),
    opacities=np.random.rand(1000),
    sh0=np.random.randn(1000, 3),
    format="linear"  # or "auto", "ply", "rasterizer"
)

# Create GSTensor from tensors
gstensor = GSTensor.from_arrays(
    means=torch.randn(1000, 3),
    scales=torch.rand(1000, 3),
    quats=torch.randn(1000, 4),
    opacities=torch.rand(1000),
    sh0=torch.randn(1000, 3),
    format="linear",
    device="cuda"
)
```

### Data Manipulation

```python
from gsply import GSData

# Slicing and indexing
subset = data[100:200]                    # Slice
first = data[0]                          # Single Gaussian
filtered = data[data.opacities > 0.5]    # Boolean mask

# Concatenation
combined = data1 + data2                 # Pairwise (1.9x faster)
merged = GSData.concatenate([data1, data2, data3])  # Bulk (6.15x faster)

# Optimize for faster operations
data = data.make_contiguous()            # 2-45x speedup for operations

# Copy and modify
bright = data.copy()
bright.sh0 *= 1.5  # Make brighter
```

### In-Memory Compression

```python
from gsply import compress_to_bytes, decompress_from_bytes

# Compress for network transfer or storage
compressed_bytes = compress_to_bytes(data)

# Decompress from bytes
data_restored = decompress_from_bytes(compressed_bytes)
```

### SPZ Format (Niantic)

```python
from gsply import read_spz, read_spz_gpu, write_spz

# Read any SPZ — the container is auto-detected (legacy gzip v1/v2/v3 or NGSP v4
# zstd). Returns a GSData in PLY format, same API as plyread().
data = read_spz("scene.spz")

# PyTorch path: returns GSTensor on the requested device.
# This avoids materializing the decoded float arrays on CPU before upload.
gstensor = read_spz_gpu("scene.spz", device="cuda")

# Write v3 (gzip, smallest-three quats) — the default
write_spz("out.spz", data)

# Write v4 (NGSP container, per-attribute zstd) — Niantic's modern format, ~3% smaller
write_spz("out_v4.spz", data, version=4)               # zstd_level=12 (default)
write_spz("fast.spz",   data, version=4, zstd_level=9) # ~2.3x faster write, +0.8% size
```

> v4 needs the zstd extra (`pip install gsply[spz]`); v1–v3 work without it
> (large v3 writes use a core parallel gzip fallback when `isal` is not present;
> `isal` remains preferred when installed).

### SOG Format (Optional)

```python
from gsply import sogread

# Read SOG file - returns GSData (same API as plyread)
data = sogread("model.sog")
positions = data.means
colors = data.sh0

# Read unbundled splat-transform output
data = sogread("model/meta.json")

# Read from bytes (in-memory, no disk I/O)
with open("model.sog", "rb") as f:
    sog_bytes = f.read()
data = sogread(sog_bytes)  # Fully in-memory extraction

# SOG returns PLY-format values: log-scales and logit-opacities.
linear = data.denormalize(inplace=False)
```

---

## Performance

### Benchmark Summary

**Uncompressed Format (400K Gaussians, SH0):**
- Read: 5.7ms (70M Gaussians/sec)
- Write: 19.3ms (21M Gaussians/sec)

**Compressed Format (400K Gaussians, SH0):**
- Read: 8.5ms (47M Gaussians/sec)
- Write: 15.0ms (27M Gaussians/sec)
- Size reduction: 71-74%

**Peak Performance:**
- Read: 78M Gaussians/sec (1M Gaussians, SH0, uncompressed)
- Write: 29M Gaussians/sec (100K Gaussians, SH0, compressed)

**GPU Transfer (400K Gaussians, RTX 3090 Ti):**
- With `_base` optimization: 1.99ms (11x faster, single tensor transfer)
- Without `_base`: 22.78ms (CPU copy + transfer)

**Format Conversion (Fused Kernels):**
- `normalize()` / `denormalize()`: ~8-15x faster with parallel Numba kernels
- Single-pass processing reduces memory overhead

See detailed [performance benchmarks](docs/API_REFERENCE.md#performance) for more information.

---

## Format Support

### Uncompressed PLY

Standard binary little-endian PLY format:

| SH Degree | Properties | Description |
|-----------|-----------|-------------|
| 0 | 14 | xyz, f_dc(3), opacity, scales(3), quats(4) |
| 1 | 23 | + 9 f_rest coefficients |
| 2 | 38 | + 24 f_rest coefficients |
| 3 | 59 | + 45 f_rest coefficients |

### Compressed PLY (PlayCanvas)

Chunk-based quantized format:
- Automatically saves as `.compressed.ply` when `compressed=True`
- Compression ratio: 71-74% size reduction
- Compatible with PlayCanvas, SuperSplat, other WebGL viewers
- Parallel compression/decompression with Numba JIT

### SOG Format (Spatially Ordered Gaussians) - Optional

WebP-based texture format for web deployment:
- Requires `gsply[sogs]` installation
- Uses WebP images for efficient storage
- Codebook-based compression for scales and colors
- Compatible with PlayCanvas splat-transform
- Supports current `version: 2` SOG and legacy V1 assets without a version field
- Supports `.sog` ZIP bundles, folders, direct `meta.json` paths, and ZIP bytes
- Returns `GSData` container (same API as `plyread()`)
- Returns PLY-format log-scales and logit-opacities; call `denormalize()` for
  linear scales/opacities
- Invalid quaternion tags fall back to the canonical `wxyz` identity quaternion
  `[1, 0, 0, 0]`, matching splat-transform
- In-memory ZIP extraction: can read bundled SOG directly from bytes

### SPZ (Niantic) - Optional

[Niantic SPZ](https://github.com/nianticlabs/spz), read + write, both containers:
- **Legacy gzip (v1/v2/v3)**: one gzip stream; v3 uses smallest-three quaternions
- **NGSP v4**: per-attribute **zstd** streams behind a 32-byte header + table of
  contents — Niantic's current format, ~3% smaller than gzip
- Container is auto-detected on read; `write_spz(..., version=3)` (default, gzip) or
  `version=4` (NGSP/zstd, `zstd_level` tunable). `read_spz`/`write_spz` use the
  PLY-format `GSData` API, round-trip with the Niantic reference both directions.
- Requires `gsply[spz]` for v4 (zstd); v1–v3 work with the core package. Large
  v3 writes use a parallel single-member gzip fallback unless `isal` is present.

---

## API Reference

Complete API documentation: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

### Core I/O
- `plyread(file_path)` - Read PLY files (auto-detects format)
- `plywrite(file_path, ...)` - Write PLY files
- `read_spz(file_path)` - Read Niantic SPZ (gzip v1/v2/v3 or NGSP v4 zstd)
- `read_spz_gpu(file_path, device="cuda")` - Read SPZ into `GSTensor` with tensor decode
- `write_spz(file_path, data, *, version=3, ...)` - Write SPZ (v3 gzip / v4 zstd)
- `detect_format(file_path)` - Detect format and SH degree
- `sogread(file_path | meta_json_path | bytes)` - Read SOG files (optional)

### GSData Container
- `GSData.load(file_path)` - Load from PLY (classmethod)
- `data.save(file_path, compressed=False)` - Save to PLY
- `GSData.from_arrays(...)` - Create from arrays with format preset
- `GSData.from_dict(...)` - Create from dictionary with format preset
- `data.normalize()` / `data.denormalize()` - Format conversion (fused kernels, ~8-15x faster)
- `data.to_rgb()` / `data.to_sh()` - Color conversion
- Format query: `is_scales_ply`, `is_scales_linear`, `is_opacities_ply`, `is_opacities_linear`, `is_sh0_sh`, `is_sh0_rgb`, `is_sh_order_0/1/2/3`
- `data[index]` - Indexing and slicing
- `data.unpack()` - Unpack to tuple
- `data.copy()` - Deep copy

### Compression APIs
- `compress_to_bytes(data)` - Compress to bytes
- `compress_to_arrays(data)` - Compress to arrays
- `decompress_from_bytes(bytes)` - Decompress from bytes

### Utility Functions
- `sh2rgb(sh)` / `rgb2sh(rgb)` - Color conversion
- `logit(x)` / `sigmoid(x)` - Optimized CPU functions
- `apply_pre_activations(data, ...)` - Fused activation kernel (~8-15x faster)
- `apply_pre_deactivations(data, ...)` - Fused deactivation kernel (~8-15x faster)
- `SH_C0` - Normalization constant

### GPU Support (PyTorch)
- `GSTensor.load(file_path, device='cuda')` - Load from PLY (classmethod)
- `gstensor.save(file_path, compressed=True)` - Save with GPU compression
- `GSTensor.from_arrays(...)` - Create from tensors with format preset
- `GSTensor.from_gsdata(data, device='cuda')` - Convert to GPU
- `gstensor.to_gsdata()` - Convert to CPU
- `gstensor.normalize()` / `gstensor.denormalize()` - GPU format conversion
- `gstensor.to_rgb()` / `gstensor.to_sh()` - GPU color conversion
- Format query: `is_scales_ply`, `is_scales_linear`, `is_opacities_ply`, `is_opacities_linear`, `is_sh0_sh`, `is_sh0_rgb`, `is_sh_order_0/1/2/3`
- Device management: `.to()`, `.cpu()`, `.cuda()`
- Precision: `.half()`, `.float()`, `.double()`

---

## What's New

### v0.4.5 - C++ Backend Performance Routing

- **C++ PLY parity without slow paths**: C++ reads now preserve an aligned
  canonical `_base` row buffer, so read-backed writes keep the zero-copy path.
- **Smarter backend dispatch**: `gsply.use_backend("cpp")` now routes PLY writes
  through C++ only for generated higher-order SH data where it wins; SH0 and
  read-backed `_base` writes stay on the faster Python paths.
- **SPZ routing tuned**: SPZ v4 reads use the Python zstd path when available,
  while C++ remains the fallback for environments without `zstandard`.

### v0.4.4 - SOG Parity and Normalization Correctness

- **SOG parity**: `sogread()` now follows the current PlayCanvas
  splat-transform reader for `version: 2`, legacy V1 assets, direct
  `meta.json` paths, and unknown-version errors.
- **Normalization correctness**: SOG data remains in PLY-format log-scales and
  logit-opacities, and activated `exp`/`sigmoid` values match splat-transform
  round trips.
- **Quaternion fallback**: Invalid SOG tags and zero-norm activation fallback
  now use canonical `wxyz` identity `[1, 0, 0, 0]`.

### v0.4.3 - SPZ v3 Python Write Performance

- **Faster pure-Python SPZ v3 writes**: Large `write_spz(..., version=3)`
  payloads now use a parallel single-member gzip fallback when `isal` is not
  installed.
- **Compatibility preserved**: The optimized path still emits one strict gzip
  member and round-trips through the same SPZ reader and Niantic-compatible
  payload format.

### v0.4.2 - Bundled Platform Wheels

- **Bundled C++ wheels**: `pip install gsply` installs `gsply_cpp`
  automatically on supported CPython 3.10-3.13 platform wheels.
- **No separate C++ extra**: No `gsply[cpp]` extra or separate `gsply-cpp`
  package is needed.

### v0.2.11 - GPU Compression Optimization

- **torch.compile() Auto-Optimization**: GPU compression now uses `torch.compile()` when available
  - ~25% faster GPU compression (5.0ms → 4.0ms for 365K Gaussians)
  - Automatic fallback to eager mode if compilation fails
  - Zero configuration required - works transparently
- **GPU Rounding Fix**: Fixed quaternion quantization to match CPU behavior exactly
- **Platform Support**: Triton backend on Linux (standard), Windows (requires `triton-windows` package)

### v0.2.9 - Protocol Interfaces & Performance Optimization

- **Protocol Interfaces**: Type-safe interfaces for format management
  - `FormatAware` - Protocol for objects that track format state
  - `Normalizable` - Protocol for objects that support format conversion
  - `GaussianContainer` - Protocol for objects containing Gaussian splat data
  - Enables structural typing across GSData and GSTensor
- **Format Management API**: Advanced format control methods
  - `format_state` property - Returns current format as immutable dictionary
  - `copy_format_from(other)` - Copy format state from another object
  - `with_format(**kwargs)` - Create shallow copy with modified format
- **Performance Optimizations**: Improved efficiency for common operations
  - Removed auto-consolidate overhead in plywrite() - users can call `make_contiguous()` manually when needed
  - In-place format conversion now default (`inplace=True`) - reduces memory allocations
  - Better performance for already-contiguous data

### v0.2.8 - Format Query Properties

- **Format Query Properties**: Convenient boolean properties to check current data format
  - `is_scales_ply`, `is_scales_linear` - Check scale format
  - `is_opacities_ply`, `is_opacities_linear` - Check opacity format
  - `is_sh0_sh`, `is_sh0_rgb` - Check color format
  - `is_sh_order_0/1/2/3` - Check SH degree
  - Available on both `GSData` and `GSTensor`

### v0.2.7 - Fused Activation Kernels & Performance Optimization

- **Fused Activation Kernels**: Ultra-fast format conversion with parallel Numba kernels
  - `apply_pre_activations()` - Fused kernel for activating scales, opacities, and quaternions (~8-15x faster)
  - `apply_pre_deactivations()` - Fused kernel for deactivating scales and opacities (~8-15x faster)
  - `normalize()` and `denormalize()` now use optimized fused kernels internally
  - Single-pass processing reduces memory overhead and improves cache locality

### v0.2.6 - Format Safety & Auto-detection

- **Convenience Factory Methods**: `GSData.from_arrays()`, `GSData.from_dict()`, `GSTensor.from_arrays()`, `GSTensor.from_dict()`
- **Auto-Format Detection**: Smart heuristics automatically detect PLY vs Linear format
- **Format Safety**: Strict validation prevents mixing incompatible formats
- **Format Helpers**: `create_ply_format()`, `create_rasterizer_format()`

### v0.2.5 - SOG Format Support & API Improvements

- **SOG Format Support**: `sogread()` - Read SOG (Spatially Ordered Gaussians) format files
- **Object-Oriented I/O API**: `data.save()`, `GSData.load()`, `gstensor.save()`, `GSTensor.load()`
- **Format Conversion API**: `normalize()`, `denormalize()` for linear ↔ PLY format conversion
- **Color Conversion API**: `to_rgb()`, `to_sh()` for SH ↔ RGB conversion

[Full Changelog](docs/CHANGELOG.md) • [API Reference](docs/API_REFERENCE.md)

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
├── src/gsply/          # Source code
│   ├── __init__.py     # Public API exports and lazy optional imports
│   ├── _backend.py     # gsply_cpp backend selection
│   ├── gsdata.py       # GSData dataclass
│   ├── reader.py       # PLY reading
│   ├── writer.py       # PLY writing
│   ├── spz.py          # Niantic SPZ reading/writing
│   ├── sog_reader.py   # SOG reading
│   ├── formats.py      # Format detection
│   ├── utils.py        # Utility functions (fused kernels)
│   └── torch/          # PyTorch integration
│       ├── gstensor.py # GSTensor GPU dataclass
│       ├── compression.py
│       ├── spz.py      # SPZ tensor decode
│       └── io.py       # GPU I/O
├── cpp/                # Bundled C++ acceleration backend source
├── tests/              # Unit tests (460 collected)
├── benchmarks/         # Performance benchmarks
├── docs/               # Documentation
└── pyproject.toml      # Package configuration
```

### Testing

gsply has comprehensive test coverage with **438 passed, 22 skipped, 460 collected** in the latest local verification:

```bash
# Run all tests
pytest tests/ -v

# Run PyTorch tests (requires torch)
pytest tests/ -v -k "torch or gstensor"

# Run with coverage
pytest tests/ -v --cov=gsply --cov-report=html
```

### Benchmarking

```bash
# Install benchmark dependencies
pip install -e .[benchmark]

# Run benchmark
python benchmarks/benchmark.py
```

---

## Documentation

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Changelog](docs/CHANGELOG.md)** - Version history and release notes
- **[Contributing](docs/CONTRIBUTING.md)** - Contribution guidelines

---

## Contributing

Contributions are welcome! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

**Quick start:**
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run tests: `pytest tests/ -v`
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

[Report Bug](https://github.com/OpsiClear/gsply/issues) • [Request Feature](https://github.com/OpsiClear/gsply/issues) • [Documentation](docs/API_REFERENCE.md)

</div>
