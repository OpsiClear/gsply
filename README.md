<div align="center">

# gsply

### Ultra-Fast Gaussian Splatting PLY I/O Library

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

**78M Gaussians/sec read | 29M Gaussians/sec write | Pure Python with NumPy and Numba**

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Performance](#performance) | [Documentation](#documentation)

</div>

---

## Overview

**gsply** is a pure Python library for ultra-fast reading and writing of Gaussian Splatting PLY files. Built specifically for performance-critical applications, gsply achieves read speeds up to 78M Gaussians/sec and write speeds up to 29M Gaussians/sec, with zero external dependencies beyond numpy.

**Why gsply?**
- **Blazing Fast**: Zero-copy reads by default, JIT-accelerated compressed I/O
- **Pure Python**: NumPy + Numba (no C++ compilation needed)
- **Format Support**: native Gaussian Splatting ply + PlayCanvas compressed ply format
- **Auto-Detection**: Automatically detects format and SH degree

---

## Features

- **Fastest Gaussian PLY I/O**: Peak performance of 78M Gaussians/sec read, 29M Gaussians/sec write
  - **400K Gaussians, SH0 (uncompressed)**: Read 5.7ms (70M/s), Write 19.3ms (21M/s)
  - **400K Gaussians, SH3 (uncompressed)**: Read 25.3ms (16M/s), Write 98ms (4.1M/s)
  - **100K Gaussians, SH0 (compressed)**: Read 2.8ms (35M/s), Write 3.4ms (29M/s) - **71% smaller**
  - **1M Gaussians, SH0**: Peak read 12.8ms (78M/s), compressed write 35.5ms (28M/s)
  - **Verified equivalent**: Output files match plyfile exactly (byte-for-byte validation)
  - **Optimizations**: Zero-copy reads, parallel JIT processing, LRU header caching, lookup tables
- **Zero-copy optimization**: Enabled by default for maximum performance
- **Pure Python**: NumPy + Numba JIT (no C++ compilation required)
- **Multiple SH degrees**: Supports SH degrees 0-3 (14, 23, 38, 59 properties)
- **Auto-format detection**: Automatically detects uncompressed vs compressed formats
- **In-memory compression/decompression**: New APIs for compressing and decompressing bytes without disk I/O
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
from gsply import plyread, plywrite, detect_format

# Read PLY file (auto-detects format) - returns GSData container
# Zero-copy optimization for maximum speed (up to 78M Gaussians/sec)
data = plyread("model.ply")
print(f"Loaded {data.means.shape[0]} Gaussians")

# Access via attributes
positions = data.means
colors = data.sh0

# Or unpack if needed (for compatibility)
means, scales, quats, opacities, sh0, shN = data[:6]

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
`GSData` namedtuple with decompressed Gaussian parameters:
- `means`: (N, 3) - Gaussian centers
- `scales`: (N, 3) - Log scales
- `quats`: (N, 4) - Rotations as quaternions (wxyz)
- `opacities`: (N,) - Logit opacities
- `sh0`: (N, 3) - DC spherical harmonics
- `shN`: (N, K, 3) - Higher-order SH coefficients
- `base`: None (not applicable for decompressed data)

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

## Performance

### Benchmark Results

Comprehensive performance benchmarks across different file sizes and formats (median of multiple runs):

**Uncompressed Format Performance**

| Gaussians | SH | Read (ms) | Write (ms) | Read (M/s) | Write (M/s) |
|-----------|----|---------:|-----------:|-----------:|------------:|
| 100K | 0 | 1.4 | 5.5 | **70.7** | 18.3 |
| 100K | 3 | 6.1 | 18.4 | 16.4 | 5.4 |
| 400K | 0 | 5.7 | 19.3 | **69.6** | 20.7 |
| 400K | 3 | 25.3 | 98.0 | 15.8 | 4.1 |
| 1M | 0 | 12.8 | 62.2 | **78.0** | 16.1 |
| 1M | 3 | 71.3 | 256.1 | 14.0 | 3.9 |

**Compressed Format Performance**

| Gaussians | SH | Read (ms) | Write (ms) | Read (M/s) | Write (M/s) | Size Reduction |
|-----------|----|---------:|-----------:|-----------:|------------:|---------------:|
| 100K | 0 | 2.8 | 3.4 | 35.4 | **29.4** | 71% |
| 100K | 3 | 30.5 | 22.5 | 3.3 | 4.5 | 74% |
| 400K | 0 | 8.5 | 15.0 | 47.0 | 26.6 | 71% |
| 400K | 3 | 118.2 | 91.7 | 3.4 | 4.4 | 74% |
| 1M | 0 | 16.7 | 35.5 | **60.0** | **28.2** | 71% |
| 1M | 3 | 256.4 | 210.0 | 3.9 | 4.8 | 74% |

### Key Performance Highlights

- **Peak Read Speed**: 78M Gaussians/sec (1M Gaussians, SH0, uncompressed)
- **Peak Write Speed**: 29.4M Gaussians/sec (100K Gaussians, SH0, compressed)
- **Compression Benefits**: 71-74% file size reduction with excellent performance
- **Scalability**: Linear scaling proven up to 1M Gaussians
- **Format Flexibility**: Compressed format can be *faster* than uncompressed for writes

### Optimization Details

- **Zero-copy reads**: Direct memory views without data duplication
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

**Write Performance (1.4-2.3x speedup):**
- **gsply**: Pre-computed templates + pre-allocated array + buffered I/O
  - **Pre-computed header templates**: Avoids dynamic string building in loops
  - **Buffered I/O**: 2MB buffer for large files reduces system call overhead
  - Allocates single contiguous array with exact dtype needed
  - Fills array via direct slice assignment (no intermediate structures)
  - Single `tobytes()` + buffered file write operation
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
