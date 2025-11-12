<div align="center">

# gsply

### Ultra-Fast Gaussian Splatting PLY I/O Library

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

**4.3x faster reads | 1.5x faster writes | Zero dependencies | Pure Python**

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Performance](#performance) | [Documentation](#documentation)

</div>

---

## Overview

**gsply** is a pure Python library for ultra-fast reading and writing of Gaussian Splatting PLY files. Built specifically for performance-critical applications, gsply achieves 4.3x faster reads and 1.5x faster writes compared to plyfile, with zero external dependencies beyond numpy.

**Why gsply?**
- **Blazing Fast**: Zero-copy reads by default, single bulk operations
- **Zero Dependencies**: Pure Python + numpy + numba (no C++ compilation needed)
- **Format Support**: native Gaussian Splatting ply + PlayCanvas compressed ply format
- **Auto-Detection**: Automatically detects format and SH degree

---

## Features

- **Fastest Gaussian PLY I/O**: Read 4.3x faster, write 1.5x faster than plyfile
  - Read (zero-copy): 7.46ms vs 31.73ms (388K Gaussians, SH3) - 52M Gaussians/sec
  - Write (uncompressed): 28.24ms vs 41.78ms (388K Gaussians, SH3) - 14M Gaussians/sec
  - **Verified equivalent**: Output files match plyfile exactly (byte-for-byte validation)
  - **Optimized**: Bulk header reading, pre-computed templates, buffered I/O
- **Zero-copy optimization**: Enabled by default for maximum performance
- **Zero dependencies**: Pure Python + numpy (no compilation required)
- **Multiple SH degrees**: Supports SH degrees 0-3 (14, 23, 38, 59 properties)
- **Auto-format detection**: Automatically detects uncompressed vs compressed formats
- **Type-safe**: Full type hints for Python 3.10+

---

## Installation

### From PyPI

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
# Always uses zero-copy optimization (6.3x faster than plyfile)
data = gsply.plyread("model.ply")
print(f"Loaded {data.means.shape[0]} Gaussians")

# Access via attributes
positions = data.means
colors = data.sh0

# Or unpack if needed (for compatibility)
means, scales, quats, opacities, sh0, shN = data[:6]

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

### `plyread(file_path)`

Read Gaussian Splatting PLY file (auto-detects format).

Always uses zero-copy optimization for maximum performance.

**Parameters:**
- `file_path` (str | Path): Path to PLY file

**Returns:**
`GSData` namedtuple with Gaussian parameters:
- `means`: (N, 3) - Gaussian centers
- `scales`: (N, 3) - Log scales
- `quats`: (N, 4) - Rotations as quaternions (wxyz)
- `opacities`: (N,) - Logit opacities
- `sh0`: (N, 3) - DC spherical harmonics
- `shN`: (N, K, 3) - Higher-order SH coefficients (K=0 for degree 0, K=9 for degree 1, etc.)
- `base`: Base array (kept alive for zero-copy views)

**Performance:**
- Uncompressed: ~6ms for 400K Gaussians (zero-copy views)
- Compressed (with JIT): ~15ms for 400K Gaussians
- Compressed (no JIT): ~90ms for 400K Gaussians

**Example:**
```python
# Zero-copy reading (6.3x faster than plyfile)
data = gsply.plyread("model.ply")
print(f"Loaded {data.means.shape[0]} Gaussians with SH degree {data.shN.shape[1]}")

# Access via attributes
positions = data.means
colors = data.sh0

# Or unpack if needed
means, scales, quats, opacities, sh0, shN = data[:6]
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

**Real-World Dataset** (388K Gaussians, SH degree 3, 30 iterations):

| Operation | gsply | plyfile | Speedup |
|-----------|-------|---------|---------|
| **Read** (uncompressed) | **7.46ms** (52M/sec) | 31.73ms (12M/sec) | **4.3x faster** |
| **Write** (uncompressed) | **28.24ms** (14M/sec) | 41.78ms (9M/sec) | **1.5x faster** |

- **Input/Output Equivalence**: Verified - gsply and plyfile produce identical results
- **File size**: 20.76 MB (59 properties per Gaussian)
- **Test file**: frame_0.ply from production dataset
- **Optimizations**: Bulk header reading (7.6% read improvement), pre-computed templates + buffered I/O (4.9% write improvement)

### Why gsply is Faster

**Read Performance (4.3x speedup):**
- **gsply**: Optimized bulk header read + `np.fromfile()` + zero-copy views (7.46ms for 388K Gaussians)
  - **Bulk header reading**: Single 8KB read + decode (vs. N readline() calls)
  - Reads entire binary data as contiguous block in one system call
  - Creates memory views directly into the data array (no copies)
  - Base array kept alive via GSData container's reference counting
- **plyfile**: Line-by-line header + 59 individual property accesses (31.73ms)
  - Multiple readline() + decode operations for header parsing
  - Accesses each property separately through PLY structure
  - Stacks columns together requiring multiple memory allocations and copies
  - Generic PLY parser handles arbitrary formats with overhead

**Write Performance (1.5x speedup):**
- **gsply**: Pre-computed templates + pre-allocated array + buffered I/O (28.24ms for 388K Gaussians)
  - **Pre-computed header templates**: Avoids dynamic string building in loops
  - **Buffered I/O**: 2MB buffer for large files reduces system call overhead
  - Allocates single contiguous array with exact dtype needed
  - Fills array via direct slice assignment (no intermediate structures)
  - Single `tobytes()` + buffered file write operation
- **plyfile**: Dynamic header + 59 property assignments + PLY construction (41.78ms)
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
