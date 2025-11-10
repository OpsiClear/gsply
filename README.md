<div align="center">

# gsply

### Ultra-Fast Gaussian Splatting PLY I/O Library

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-coming%20soon-orange.svg)](https://pypi.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**6.3x faster reads | 1.4x faster writes | Zero dependencies | Pure Python**

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Performance](#performance) | [Documentation](#documentation)

</div>

---

## Overview

**gsply** is a pure Python library for ultra-fast reading and writing of Gaussian Splatting PLY files. Built specifically for performance-critical applications, gsply achieves 6.3x faster reads and 1.4x faster writes compared to plyfile, with zero external dependencies beyond numpy.

**Why gsply?**
- **Blazing Fast**: Zero-copy reads by default, single bulk operations
- **Zero Dependencies**: Pure Python + numpy (no C++ compilation needed)
- **Format Support**: Uncompressed PLY + PlayCanvas compressed format
- **Auto-Detection**: Automatically detects format and SH degree
- **Production Ready**: 65 passing tests, comprehensive benchmarks, CI/CD pipeline

---

## Features

- **Fastest Gaussian PLY I/O**: Read 6.3x faster, write 1.4x faster than plyfile
  - Read (default zero-copy): 8.09ms (400K Gaussians, SH0) - 49M Gaussians/sec
  - Write (uncompressed): 8.72ms vs 12.18ms (50K Gaussians, SH3)
  - Compressed read: 14.74ms (400K Gaussians) - 27M Gaussians/sec
  - Compressed write: 63ms (400K Gaussians) - 6.3M Gaussians/sec with parallel processing
- **Zero-copy optimization**: Enabled by default for maximum performance
- **Zero dependencies**: Pure Python + numpy (no compilation required)
- **Multiple SH degrees**: Supports SH degrees 0-3 (14, 23, 38, 59 properties)
- **Compressed format**: PlayCanvas-compatible chunk-based compression (3.4x smaller for SH0)
- **Auto-format detection**: Automatically detects uncompressed vs compressed formats
- **Type-safe**: Full type hints for Python 3.10+
- **GSData container**: Named tuple for clean attribute access and tuple unpacking

---

## Installation

### From PyPI (coming soon)

```bash
pip install gsply
```

### From Source

```bash
git clone https://github.com/OpsiClear/gsply.git
cd gsply
pip install -e .
```

---

## Quick Start

```python
import gsply

# Read PLY file (auto-detects format) - returns GSData container
# Uses fast zero-copy mode by default (6.3x faster than plyfile)
data = gsply.plyread("model.ply")
print(f"Loaded {data.means.shape[0]} Gaussians")

# Access via attributes
positions = data.means
colors = data.sh0

# Or unpack if needed (for compatibility)
means, scales, quats, opacities, sh0, shN = data[:6]

# Explicit zero-copy reading (default, 6.3x faster than plyfile)
data = gsply.plyread("model.ply", fast=True)

# Safe copy reading (if you need independent arrays)
data = gsply.plyread("model.ply", fast=False)

# Write uncompressed PLY file
gsply.plywrite("output.ply", data.means, data.scales, data.quats,
               data.opacities, data.sh0, data.shN)

# Write compressed PLY file (saves as "output.compressed.ply", 14.5x smaller)
gsply.plywrite("output.ply", data.means, data.scales, data.quats,
               data.opacities, data.sh0, data.shN, compressed=True)

# Detect format before reading
is_compressed, sh_degree = gsply.detect_format("model.ply")
print(f"Compressed: {is_compressed}, SH degree: {sh_degree}")
```

---

## API Reference

### `plyread(file_path, fast=True)`

Read Gaussian Splatting PLY file (auto-detects format).

**Parameters:**
- `file_path` (str | Path): Path to PLY file
- `fast` (bool, optional): Use zero-copy optimization for uncompressed files. Default: True.
  - `fast=True`: 1.65x faster, uses array views (default)
  - `fast=False`: Safe independent copies of all arrays

**Returns:**
`GSData` namedtuple with Gaussian parameters:
- `means`: (N, 3) - Gaussian centers
- `scales`: (N, 3) - Log scales
- `quats`: (N, 4) - Rotations as quaternions (wxyz)
- `opacities`: (N,) - Logit opacities
- `sh0`: (N, 3) - DC spherical harmonics
- `shN`: (N, K, 3) - Higher-order SH coefficients (K=0 for degree 0, K=9 for degree 1, etc.)
- `base`: Base array (None for safe copies, array for zero-copy reads)

**Performance:**
- `fast=True`: ~2.89ms for 50K Gaussians (zero-copy views)
- `fast=False`: ~4.75ms for 50K Gaussians (safe copies)
- Speedup: 1.65x with zero-copy optimization

**Example:**
```python
# Default: fast zero-copy reading (1.65x faster)
data = gsply.plyread("model.ply")
print(f"Loaded {data.means.shape[0]} Gaussians with SH degree {data.shN.shape[1]}")

# Access via attributes
positions = data.means
colors = data.sh0

# Or unpack if needed
means, scales, quats, opacities, sh0, shN = data[:6]

# Safe copy reading (if you need independent arrays)
data = gsply.plyread("model.ply", fast=False)
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
- Uncompressed: ~8ms for 400K Gaussians (49M Gaussians/sec)
- Compressed: ~63ms for 400K Gaussians (6.3M Gaussians/sec), 3.4x smaller files

**Example:**
```python
# Write uncompressed (fast, ~8ms for 400K Gaussians)
gsply.plywrite("output.ply", means, scales, quats, opacities, sh0, shN)

# Write compressed (saves as "output.compressed.ply", ~63ms, 3.4x smaller)
gsply.plywrite("output.ply", means, scales, quats, opacities, sh0, shN, compressed=True)
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
is_compressed, sh_degree = gsply.detect_format("model.ply")
if is_compressed:
    print("Compressed PlayCanvas format")
else:
    print(f"Uncompressed format with SH degree {sh_degree}")
```

---

## Performance

### Benchmark Results

**Real-World Dataset** (90 files, 36M Gaussians total, SH degree 0):

| Operation | gsply | Throughput | Details |
|-----------|-------|------------|---------|
| **Read** (uncompressed, zero-copy) | **8.09ms** | **49M Gaussians/sec** | 400K Gaussians avg |
| **Write** (compressed) | **63ms** | **6.3M Gaussians/sec** | 400K Gaussians, 3.4x compression |
| **Read** (compressed) | **14.74ms** | **27M Gaussians/sec** | Parallel decompression |

**Comparison vs Other Libraries** (50K Gaussians, SH degree 3):

| Operation | gsply | plyfile | Open3D | Winner |
|-----------|-------|---------|--------|--------|
| **Read** (zero-copy, default) | **2.89ms** | 18.23ms | 43.10ms | **gsply 6.3x faster** |
| **Read** (safe copies) | **4.75ms** | 18.23ms | 43.10ms | **gsply 3.8x faster** |
| **Write** (uncompressed) | **8.72ms** | 12.18ms | 35.69ms | **gsply 1.4x faster** |

### Why gsply is Faster

**Read Performance (6.3x speedup with zero-copy, default):**
- gsply: Single `np.fromfile()` + zero-copy array views (keeps base array alive)
- plyfile: 59 individual property accesses + column stacking + multiple copies
- Result: No memory copies for shN coefficients, 1.65x faster than safe copies
- Alternative: `fast=False` still achieves 3.8x speedup with safe independent arrays

**Write Performance (1.4x speedup):**
- gsply: Pre-allocated arrays + dtype optimization + direct binary write
- plyfile: 59 individual assignments + PLY structure overhead
- Result: Eliminates unnecessary copies and type conversions

**Compressed Format (with optimizations):**
- Read: 14.74ms for 400K Gaussians (27M Gaussians/sec) via parallel JIT decompression
- Write: 63ms for 400K Gaussians (6.3M Gaussians/sec) via radix sort + parallel JIT packing
- Optimizations: O(n) radix sort for chunk sorting, numba parallel processing for bit packing/unpacking
- Size: 3.4x smaller files with PlayCanvas-compatible chunk quantization

**Key Insight**: gsply's performance comes from recognizing that Gaussian Splatting PLY files follow a fixed format, allowing bulk operations and zero-copy views instead of generic PLY parsing. For details, see [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

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
│       ├── reader.py          # PLY reading (uncompressed + compressed)
│       ├── writer.py          # PLY writing (uncompressed + compressed)
│       └── py.typed           # PEP 561 type marker
├── tests/                     # Unit tests (56 tests)
├── benchmarks/                # Performance benchmarks
├── docs/                      # Documentation
│   ├── PERFORMANCE.md         # Performance benchmarks and optimization history
│   ├── COMPRESSED_FORMAT.md   # Compressed PLY format specification
│   ├── VECTORIZATION_EXPLAINED.md  # Vectorization deep-dive
│   ├── CI_CD_SETUP.md         # CI/CD pipeline documentation
│   ├── BUILD.md               # Build and distribution guide
│   ├── RELEASE_NOTES.md       # Release notes and version history
│   ├── COMPATIBILITY_FIXES.md # Format compatibility details
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

gsply has comprehensive test coverage with 65 passing tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_reader.py -v

# Run with coverage report
pytest tests/ -v --cov=gsply --cov-report=html
```

Test categories:
- Format detection (compressed/uncompressed, SH degrees)
- Reading (various SH degrees, edge cases, compressed format)
- Writing (various SH degrees, format preservation, compressed format)
- Round-trip (read-write-read consistency)
- Error handling (invalid files, malformed data)

---

## Documentation

gsply includes comprehensive documentation covering all aspects of the library:

- **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** - Performance benchmarks, optimization history, and comparison with other libraries
- **[docs/COMPRESSED_FORMAT.md](docs/COMPRESSED_FORMAT.md)** - Complete compressed PLY format specification with examples
- **[docs/VECTORIZATION_EXPLAINED.md](docs/VECTORIZATION_EXPLAINED.md)** - Deep-dive into vectorization techniques for 38.5x speedup
- **[docs/BUILD.md](docs/BUILD.md)** - Build system, distribution, and packaging guide
- **[docs/CI_CD_SETUP.md](docs/CI_CD_SETUP.md)** - CI/CD pipeline reference and GitHub Actions workflows
- **[docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)** - Release notes and version history
- **[docs/COMPATIBILITY_FIXES.md](docs/COMPATIBILITY_FIXES.md)** - Format compatibility details and fixes
- **[docs/archive/](docs/archive/)** - Historical documentation from development phases

---

## CI/CD

gsply includes a complete GitHub Actions CI/CD pipeline:

- **Multi-platform testing**: Ubuntu, Windows, macOS
- **Multi-version testing**: Python 3.10, 3.11, 3.12, 3.13
- **Automated benchmarking**: Performance tracking on PRs
- **Build verification**: Wheel building and installation testing
- **PyPI publishing**: Automated release on GitHub Release

See [docs/CI_CD_SETUP.md](docs/CI_CD_SETUP.md) for details.

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
