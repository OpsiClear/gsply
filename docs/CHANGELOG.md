# Release Notes

## v0.2.0 (Current - Breaking Changes & New Features)

### Breaking Changes
- **GSData is now a regular dataclass** (was NamedTuple)
- **Removed tuple unpacking compatibility** - Use direct attribute access only
  - Before: `means, scales, quats, opacities, sh0, shN = data[:6]`
  - After: Access via `data.means`, `data.scales`, etc.
- **GSData constructor requires keyword arguments**
  - Before: `GSData(means, scales, quats, opacities, sh0, shN, _base)`
  - After: `GSData(means=means, scales=scales, ...)`

### New Features
- **Mutable fields**: All GSData fields can now be modified after creation
  - `data.means[0, 0] = 999.0`  # Now works!
  - `data.scales *= 2.0`  # In-place operations supported
  - `data.means = new_array`  # Complete array replacement
- **New `masks` attribute**: Boolean mask for filtering Gaussians
  - Initialized to all `True` when reading files
  - `data.masks[100:200] = False`  # Mark Gaussians for filtering
  - Use for filtering: `filtered_means = data.means[data.masks]`
  - Not persisted to PLY files (runtime-only)
- **`len(data)` returns number of Gaussians**: More intuitive API
  - `print(f"Loaded {len(data)} Gaussians")`  # Natural usage
  - Returns `data.means.shape[0]`
- **Efficient slicing with `data[slice]`**: Pythonic data access
  - `data[0]` - Single Gaussian
  - `data[100:200]` - Range of Gaussians
  - `data[::10]` - Every 10th Gaussian
  - `data[mask]` - Boolean mask selection
  - Optimized with `_base` array slicing (up to 25x faster for masks)

### Performance Improvements
- **Peak read performance**: 93M Gaussians/sec (up from 78M)
- **Peak write performance**: 57M Gaussians/sec (zero-copy)
- **Real-world average**: 75.5M Gaussians/sec on 90 test files
- **Automatic write optimization**: All writes are automatically optimized
  - Auto-consolidation: 2.6-2.8x faster writes via automatic `_base` construction
  - Zero-copy path: Additional 2.8x speedup for data from `plyread()` (total 7-8x vs baseline)
  - Works transparently - no user code changes required!
  - 400K SH0: 18-22ms (auto-optimized) or 7ms (zero-copy from file)
  - 400K SH3: 96ms (auto-optimized) or 35ms (zero-copy from file)
- Dataclass implementation faster than NamedTuple

### Migration Guide
```python
# Old (v0.1.x)
means, scales, quats, opacities, sh0, shN = data[:6]
data = GSData(means, scales, quats, opacities, sh0, shN, _base=None)

# New (v0.2.0)
means = data.means
scales = data.scales
# ... or use attributes directly

# Write operations - automatically optimized!
data = gsply.plyread("input.ply")
gsply.plywrite("output.ply", data)  # RECOMMENDED - zero-copy (7-8x faster)!

# Creating new data - automatically optimized via consolidation (2.6-2.8x faster)
data = GSData(means=means, scales=scales, ...)
gsply.plywrite("output.ply", data)  # Automatically consolidated internally

# Or unpack - still automatically optimized
gsply.plywrite("output.ply", *data.unpack())  # Auto-consolidated too!

# Creating new GSData
data = GSData(
    means=means,
    scales=scales,
    quats=quats,
    opacities=opacities,
    sh0=sh0,
    shN=shN,
    masks=None,  # Optional
    _base=None
)
```

## v0.1.1 (Performance & Code Quality)

### Performance Improvements
- **Peak Performance**: 78M Gaussians/sec read, 29.4M Gaussians/sec write
- **Zero-copy reads**: Always enabled for maximum performance
- **Fast-path dtype checks**: Skip unnecessary float32 conversions
- **LRU header caching**: Cache frequently generated PLY headers
- **Single file handle**: Reduce file open/close syscalls
- **Lookup tables**: Eliminate SH degree branching
- **Direct array operations**: Optimize opacity column assignment

### Code Quality Improvements
- **Type Safety**: Complete type hints with mypy configuration
- **Documentation**: Enhanced docstrings with algorithm details and bit-packing format
- **Error Messages**: Improved validation errors with actionable context
- **Code Organization**: Extracted magic numbers to named constants, eliminated code duplication
- **Development Tools**: Configured ruff linter and mypy for CI/CD integration
- **Testing**: Added edge case tests (92 tests passing)

### Benchmarks (1M Gaussians, SH0)
- Uncompressed Read: 12.8ms (78M/sec)
- Compressed Write: 35.5ms (28.2M/sec)
- Compression: 71% file size reduction

### Testing
- All 92 tests passing
- Zero regressions
- Full backward compatibility

## v0.1.0 (Initial - Optimized Release)

### Features
- Ultra-fast Gaussian Splatting PLY I/O library
- Pure Python + numpy + numba (no C++ compilation required)
- Numba JIT for parallel processing and fast compressed I/O
- Support for SH degrees 0-3 (14, 23, 38, 59 properties)
- Auto-format detection (uncompressed vs compressed)
- Full compressed format support (PlayCanvas compatible) - read AND write
- Zero-copy reads for maximum performance
- GSData namedtuple container for clean API

### Performance (Real-World Benchmarks)

Tested on 90 files, 36M Gaussians total, SH degree 0:

**Uncompressed I/O**:
- **Read** (zero-copy): 8.09ms for 400K Gaussians (49M Gaussians/sec)
- **Write**: 8.72ms vs 12.18ms (plyfile) - 1.4x faster

**Compressed I/O**:
- **Read**: 14.74ms for 400K Gaussians (27M Gaussians/sec)
- **Write**: 63ms for 400K Gaussians (6.3M Gaussians/sec)
- **Compression ratio**: 3.44x (1.92 GB → 558 MB)

**Comparison vs Other Libraries** (50K Gaussians, SH3):
- Read: 2.89ms vs 18.23ms (plyfile) - **6.3x faster**
- Write: 8.72ms vs 12.18ms (plyfile) - **1.4x faster**

### Optimizations Implemented

#### Phase 1: Vectorization
- Vectorized quaternion extraction (eliminated Python loops)
- Performance: 21% improvement

#### Phase 2: Algorithmic Optimization
- O(n log n) chunk bounds computation (replaced O(n*m) boolean masking)
- Performance: 73% improvement

#### Phase 3: Radix Sort + Parallel Processing
- O(n) radix sort for chunk sorting (vs O(n log n) comparison sort)
- Parallel JIT processing with Numba for bit packing/unpacking
- Performance: 10.4x write, 15.3x read vs baseline

**Combined Impact**: Production-ready performance with 60+ FPS capability

### API

```python
import gsply

# Read PLY file (auto-detects format, returns GSData dataclass)
data = gsply.plyread("scene.ply")  # Uses fast zero-copy by default
data = gsply.plyread("scene.ply", fast=False)  # Safe copies

# Access via attributes
positions = data.means
colors = data.sh0

# Or unpack if needed
means, scales, quats, opacities, sh0, shN = data.unpack()

# Write uncompressed PLY
gsply.plywrite("output.ply", data.means, data.scales, data.quats,
               data.opacities, data.sh0, data.shN)

# Write compressed PLY (auto-adjusts extension to .compressed.ply)
gsply.plywrite("output.ply", data.means, data.scales, data.quats,
               data.opacities, data.sh0, data.shN, compressed=True)

# Detect format
is_compressed, sh_degree = gsply.detect_format("scene.ply")
```

### Documentation
- Comprehensive README with performance benchmarks
- OPTIMIZATION_SUMMARY.md: Detailed optimization history and analysis
- GUIDE.md: User guide with examples
- CONTRIBUTING.md: Contribution guidelines
- Extensive inline code documentation

### Testing
- 65 passing tests
- Full coverage of read/write operations
- Compressed format tests (read + write)
- Format detection tests
- Round-trip verification
- Edge case handling

### Distribution
- Universal wheel (`py3-none-any`)
- Works on Linux, macOS, Windows
- No platform-specific compilation required
- PEP 561 type hints marker included

### Known Limitations
- Requires Python 3.10+
- ASCII PLY format not supported (binary little-endian only)
- Compressed format is lossy (chunk-based quantization)

### Installation

```bash
# Basic installation
pip install gsply

# Development installation
pip install -e .[dev]

# All optional dependencies (dev + benchmark)
pip install -e .[all]
```

### Dependencies
- **Required**: numpy>=1.20.0, numba>=0.59.0
- **Dev**: pytest, pytest-cov, build, twine
- **Benchmark**: open3d, plyfile

### Contributors
- OpsiClear

### License
MIT License

---

## Previous Development Phases

### Phase 1: Initial Implementation
- Basic uncompressed PLY read/write
- Format detection
- SH degree support (0-3)

### Phase 2: Compressed Format Support
- PlayCanvas compressed PLY reading
- Vectorized bit unpacking
- 38.5x speedup over naive Python loops

### Phase 3: Compressed Writing
- Compressed PLY writing implementation
- Vectorized quaternion extraction
- O(n log n) chunk bounds computation
- 78.6% speedup vs initial implementation

### Phase 4: Parallel Optimization (Current)
- O(n) radix sort for chunk sorting
- Parallel JIT processing with Numba
- 10.4x write, 15.3x read speedup vs baseline
- Production-ready performance (60+ FPS capable)
