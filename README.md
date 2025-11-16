<div align="center">

# gsply

### Ultra-Fast Gaussian Splatting PLY I/O Library

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

**93M Gaussians/sec read | 57M Gaussians/sec write | Auto-optimized**

</div>

---

## What's New in v0.2.1

- Data Concatenation: `GSData.concatenate()` - Bulk merge (6.15x faster), `GSData.add()` - Optimized pairwise (1.9x faster)
- GPU Concatenation: `GSTensor.add()` - 18x faster than CPU
- Performance Optimization: `make_contiguous()` - Fix cache locality (2-45x speedup)
- Mask Management: Multi-layer boolean masks with GPU-optimized operations (100-1000x faster)

[Full API Reference](docs/API_REFERENCE.md) | [Changelog](docs/CHANGELOG.md)

---

## Quick Start

```python
from gsply import plyread, plywrite

# Read PLY file (auto-detects format, zero-copy)
data = plyread("model.ply")

# Access fields
positions = data.means    # (N, 3) xyz coordinates
colors = data.sh0         # (N, 3) RGB colors
scales = data.scales      # (N, 3) scale parameters
rotations = data.quats    # (N, 4) quaternions

# Write PLY file (automatically optimized)
plywrite("output.ply", data)

# Write compressed (71-74% smaller)
plywrite("output.ply", data, compressed=True)
```

**Performance:** 93M Gaussians/sec read, 57M Gaussians/sec write (400K Gaussians in 6-7ms)

[Installation](#installation) | [Examples](#examples) | [API Reference](docs/API_REFERENCE.md) | [Performance](#performance)

---

## Overview

Ultra-fast Gaussian Splatting PLY I/O for Python. Zero-copy reads, auto-optimized writes, optional GPU acceleration.

**Key Features:**
- Fast: 93M Gaussians/sec read, 57M Gaussians/sec write
- Auto-optimized: Writes are 2.6-2.8x faster automatically
- Pure Python: NumPy + Numba (no C++ compilation)
- Format support: Uncompressed PLY + PlayCanvas compressed (71-74% smaller)
- GPU ready: Optional PyTorch integration with GSTensor

---

## Installation

```bash
pip install gsply
```

**Dependencies:** NumPy and Numba (auto-installed)

**Optional GPU acceleration:**
```bash
pip install torch  # For GSTensor GPU features
```

---

## Examples

### Basic I/O

```python
from gsply import plyread, plywrite

# Read and access data
data = plyread("model.ply")
print(f"Loaded {len(data)} Gaussians")

# Unpack to individual arrays
means, scales, quats, opacities, sh0, shN = data.unpack()

# Write with individual arrays
plywrite("output.ply", means, scales, quats, opacities, sh0, shN)

# Or write directly from GSData
plywrite("output.ply", data)
```

### Format Detection

```python
from gsply import detect_format

is_compressed, sh_degree = detect_format("model.ply")
if is_compressed:
    print("Compressed PlayCanvas format")
else:
    print(f"Uncompressed format with SH degree {sh_degree}")
```

### In-Memory Compression

```python
from gsply import compress_to_bytes, decompress_from_bytes

# Compress for network transfer or storage
compressed_bytes = compress_to_bytes(data)

# Decompress from bytes
data_restored = decompress_from_bytes(compressed_bytes)
```

### GPU Acceleration

```python
from gsply import GSTensor

# Convert to GPU tensors (requires PyTorch)
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Access GPU tensors
positions_gpu = gstensor.means  # torch.Tensor on GPU
colors_gpu = gstensor.sh0       # torch.Tensor on GPU

# Filter on GPU
high_opacity = gstensor[gstensor.opacities > 0.5]

# Convert back to CPU
data_cpu = gstensor.to_gsdata()
```

### Data Manipulation

```python
# Slicing and indexing
subset = data[100:200]          # Slice
first = data[0]                 # Single Gaussian
filtered = data[data.opacities > 0.5]  # Boolean mask

# Concatenation
combined = data1.add(data2)     # Pairwise (1.9x faster)
combined = data1 + data2        # Or use + operator
merged = GSData.concatenate([data1, data2, data3])  # Bulk (6.15x faster)

# Optimize for faster operations
data = data.make_contiguous()   # 2-45x speedup for operations

# Copy and modify
bright = data.copy()
bright.sh0 *= 1.5  # Make brighter
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
- With optimization: 1.99ms (11.4x faster)
- Without optimization: 22.78ms

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
- Parallel compression/decompression

---

## API Reference

Complete API documentation is available in [docs/API_REFERENCE.md](docs/API_REFERENCE.md):

**Core I/O:**
- `plyread(file_path)` - Read PLY files
- `plywrite(file_path, ...)` - Write PLY files
- `detect_format(file_path)` - Detect format and SH degree

**GSData Container:**
- `data.unpack()` - Unpack to tuple
- `data.to_dict()` - Convert to dictionary
- `data.copy()` - Deep copy
- `data.consolidate()` - Optimize for slicing
- `data[index]` - Indexing and slicing

**Compression:**
- `compress_to_bytes(data)` - Compress to bytes
- `compress_to_arrays(data)` - Compress to arrays
- `decompress_from_bytes(bytes)` - Decompress from bytes

**Utilities:**
- `sh2rgb(sh)` - SH to RGB conversion
- `rgb2sh(rgb)` - RGB to SH conversion
- `SH_C0` - Normalization constant

**GPU Support (PyTorch):**
- `GSTensor.from_gsdata(data, device='cuda')` - Convert to GPU
- `gstensor.to_gsdata()` - Convert to CPU
- Device management: `.to()`, `.cpu()`, `.cuda()`
- Precision: `.half()`, `.float()`, `.double()`
- Full slicing and manipulation support

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
│   ├── gsdata.py       # GSData dataclass
│   ├── reader.py       # PLY reading
│   ├── writer.py       # PLY writing
│   ├── formats.py      # Format detection
│   └── torch/          # PyTorch integration
│       └── gstensor.py # GSTensor GPU dataclass
├── tests/              # Unit tests (169 tests)
├── benchmarks/         # Performance benchmarks
├── docs/               # Documentation
└── pyproject.toml      # Package configuration
```

---

## Testing

gsply has comprehensive test coverage with 169 passing tests:

```bash
# Run all tests
pytest tests/ -v

# Run PyTorch tests (requires torch)
pytest tests/ -v -k "torch or gstensor"

# Run with coverage
pytest tests/ -v --cov=gsply --cov-report=html
```

---

## Benchmarking

Compare gsply performance against other PLY libraries:

```bash
# Install benchmark dependencies
pip install -e .[benchmark]

# Run benchmark
python benchmarks/benchmark.py

# Custom settings
python benchmarks/benchmark.py --config.file path/to/model.ply --config.iterations 20
```

---

## Documentation

- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Changelog](docs/CHANGELOG.md) - Version history and release notes
- [Contributing](.github/CONTRIBUTING.md) - Contribution guidelines

---

## CI/CD

Complete GitHub Actions pipeline:
- Multi-platform: Ubuntu, Windows, macOS
- Multi-version: Python 3.10, 3.11, 3.12, 3.13
- Core + PyTorch testing
- Automated benchmarking
- PyPI publishing on release

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

[Report Bug](https://github.com/OpsiClear/gsply/issues) | [Request Feature](https://github.com/OpsiClear/gsply/issues) | [Documentation](docs/API_REFERENCE.md)

</div>
