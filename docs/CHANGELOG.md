# Release Notes

## v0.2.9 (Protocol Interfaces & Performance Optimization)

### New Features
- **Protocol Interfaces**: Type-safe interfaces for format management and data operations
  - `FormatAware` - Protocol for objects that track format state (scales, opacities, sh0, sh_order)
  - `Normalizable` - Protocol for objects that support format conversion (normalize/denormalize)
  - `GaussianContainer` - Protocol for objects containing Gaussian splat data
  - Enables type checking across GSData and GSTensor with structural typing
  - Improves IDE autocomplete and static analysis
- **Format Management API**: Advanced methods for format state control
  - `format_state` property - Returns current format as immutable FormatState TypedDict
  - `copy_format_from(other)` - Copy format state from another FormatAware object
  - `with_format(**kwargs)` - Create shallow copy with modified format (functional style)
  - Simplifies format handling in complex data pipelines
- **In-Place Format Conversion**: All format conversion methods now modify data in-place by default
  - `normalize(inplace=True)` - Default behavior optimized for performance
  - `denormalize(inplace=True)` - Default behavior optimized for performance
  - `to_rgb(inplace=True)` - Default behavior optimized for performance
  - `to_sh(inplace=True)` - Default behavior optimized for performance
  - Set `inplace=False` to create a copy (previous default behavior)

### Performance Improvements
- **Removed Auto-Consolidate Overhead**: Eliminated automatic `_base` array construction in plywrite()
  - Previous behavior: plywrite() automatically called consolidate() for 2.6-2.8x speedup
  - New behavior: Users can manually call `data.make_contiguous()` or `data.consolidate()` when needed
  - Reason: Auto-consolidate added unnecessary overhead for already-contiguous data
  - Maintains backward compatibility - zero-copy path still works automatically
- **In-Place Operations Default**: Format conversions now modify data in-place by default
  - Reduces memory allocations and copies
  - Better performance for typical use cases where copy is not needed
  - Previous behavior available via `inplace=False` parameter

### API Changes
- **Format Conversion Default Changed**: `inplace=True` is now the default for all conversion methods
  - Previous default: `inplace=False` (created copies)
  - New default: `inplace=True` (modifies in-place)
  - Migration: Add `inplace=False` if you need a copy instead of in-place modification
- **Example Migration**:
  ```python
  # Old code (v0.2.8 and earlier)
  data_normalized = data.normalize()  # Created a copy

  # New code (v0.2.9+)
  data_normalized = data.normalize(inplace=False)  # Explicitly request copy
  # or
  data.normalize()  # Modifies in-place (new default)
  ```

### Implementation Details
- Protocol interfaces use structural typing (no inheritance required)
- FormatState uses TypedDict with Required/NotRequired for partial updates
- All existing GSData and GSTensor methods support new protocols
- Format copying preserves format state across operations
- `with_format()` enables functional-style format updates without mutation

### Testing
- Added comprehensive test coverage for protocol interfaces (11 new tests)
  - `tests/test_protocols.py` - Tests for FormatAware, Normalizable, GaussianContainer protocols
  - Tests cover: protocol compliance, format_state property, copy_format_from(), with_format()
  - Integration tests verify GSData and GSTensor implement all protocols correctly
- Total test count: **406 tests** (365 + 41 new)

---

## v0.2.8 (Format Query Properties)

### New Features
- **Format Query Properties**: Convenient boolean properties to check current data format
  - Scale format: `is_scales_ply`, `is_scales_linear`
  - Opacity format: `is_opacities_ply`, `is_opacities_linear`
  - Color format: `is_sh0_sh`, `is_sh0_rgb`
  - SH degree: `is_sh_order_0`, `is_sh_order_1`, `is_sh_order_2`, `is_sh_order_3`
  - Available on both `GSData` and `GSTensor` classes
  - Properties update automatically during format conversions (`normalize()`, `denormalize()`, `to_rgb()`, `to_sh()`)

### Usage Example
```python
data = gsply.plyread("scene.ply")

# Check format before operations
if data.is_scales_ply:
    data.denormalize()  # Convert to linear

if data.is_sh0_sh:
    data.to_rgb()  # Convert to RGB colors

# Check SH degree
if data.is_sh_order_3:
    print("High-quality SH3 data")
```

### Implementation Details
- Properties use safe `.get()` access on `_format` dict
- All conversion methods properly update format tracking
- Format is preserved through copy, slice, concatenate, and device transfer operations
- Format validation in `add()` and `concatenate()` raises clear errors for mismatches

---

## v0.2.7 (Fused Activation Kernels & Performance Optimization)

### New Features
- **Fused Activation Kernels**: Ultra-fast format conversion with parallel Numba kernels
  - `apply_pre_activations(data, min_scale=1e-4, max_scale=100.0, min_quat_norm=1e-8, inplace=True)` - Fused kernel for activating scales, opacities, and quaternions
    - Converts log-scales → linear scales (exp + clamp) in single pass
    - Converts logit-opacities → linear opacities (sigmoid) in single pass
    - Normalizes quaternions with safety floor
    - **Performance**: ~8-15x faster than individual operations
  - `apply_pre_deactivations(data, min_scale=1e-9, min_opacity=1e-4, max_opacity=0.9999, inplace=True)` - Fused kernel for deactivating scales and opacities
    - Converts linear scales → log-scales (log + clamp) in single pass
    - Converts linear opacities → logit-opacities (logit + clamp) in single pass
    - **Performance**: ~8-15x faster than individual operations
  - Both functions use parallel Numba JIT compilation for optimal performance
  - Single-pass processing reduces memory overhead and improves cache locality

### Performance Improvements
- **Format Conversion Optimization**: `normalize()` and `denormalize()` now use fused kernels internally
  - `GSData.normalize()` uses `apply_pre_deactivations()` for ~8-15x speedup
  - `GSData.denormalize()` uses `apply_pre_activations()` for ~8-15x speedup
  - Quaternion normalization included in activation kernel (denormalize only)
  - Scales and opacities processed together in single parallel pass
- **Memory Efficiency**: Fused kernels reduce intermediate allocations
  - Single-pass processing improves cache locality
  - Lower memory overhead compared to sequential operations

### Improvements
- **Internal Refactoring**: Format conversion methods now use optimized fused kernels
  - `normalize()` replaced manual `np.log()` and `logit()` calls with `apply_pre_deactivations()`
  - `denormalize()` replaced manual `np.exp()` and `sigmoid()` calls with `apply_pre_activations()`
  - Maintains backward compatibility - same API, better performance
- **Code Quality**: Centralized activation/deactivation logic in reusable functions
  - Consistent behavior across all format conversion operations
  - Easier to maintain and optimize

### Testing
- Added comprehensive test coverage for new activation functions (17 new tests)
  - `tests/test_pre_activations.py` - Full test suite for `apply_pre_activations()` and `apply_pre_deactivations()`
  - Tests cover: basic functionality, in-place vs copy, custom bounds, edge cases, validation errors, roundtrips
  - Integration tests verify `normalize()` and `denormalize()` use optimized kernels correctly
- Total test count: **365 tests** (348 + 17 new)

## v0.2.6 (Format Safety & Auto-detection)

### New Features
- **Convenience Factory Methods**: Create GSData/GSTensor from external data with format presets
  - `GSData.from_arrays(means, scales, quats, opacities, sh0, shN=None, format='auto')` - Create from arrays with format preset
  - `GSData.from_dict(data_dict, format='auto')` - Create from dictionary with format preset
  - `GSTensor.from_arrays(means, scales, quats, opacities, sh0, shN=None, format='auto', device='cuda')` - Create from tensors with format preset
  - `GSTensor.from_dict(data_dict, format='auto', device='cuda')` - Create from dictionary with format preset
  - Format presets: `"auto"` (detect), `"ply"` (log/logit), `"linear"` or `"rasterizer"` (linear)
  - Auto-detects SH degree from `shN` shape when not specified
- **Automatic Format Detection**: Smart heuristics to detect PLY format vs Linear format
  - Automatically detects if data uses log-scales/logit-opacities (PLY format) or linear values
  - `_detect_format_from_values()` uses statistical analysis of data ranges
  - Works when creating `GSData` or `GSTensor` from raw arrays
  - Ensures correct format handling without manual flag setting
- **Format Helper Functions**: Clearer API for creating format dictionaries
  - `create_ply_format(sh_degree)` - For data matching PLY file spec
  - `create_rasterizer_format(sh_degree)` - For data matching renderer spec
  - `create_linear_format(sh_degree)` - Alias for rasterizer format
- **Strict Format Validation**:
  - `GSData` and `GSTensor` now enforce format consistency during concatenation
  - Prevents accidental merging of mixed formats (e.g. linear + log-space)
  - Raises clear `ValueError` with helpful instructions

### Improvements
- **Enhanced `GSData` / `GSTensor`**:
  - `_format` field is now always present (never None), auto-populated if missing
  - `__post_init__` automatically detects format from data values if not specified
  - All format conversion methods (`normalize`, `denormalize`, `to_rgb`, `to_sh`) correctly update format tracking
- **Writer Safety**:
  - `plywrite()` now auto-detects format when passed raw arrays
  - `plywrite()` ensures data is in PLY format before writing (auto-converts linear -> PLY if needed)
  - Prevents writing linear data as if it were log-space (which would cause invalid scale/opacity values)

### Testing
- Added `tests/test_format_management.py` covering all new format utilities and safety checks
- Added comprehensive edge case tests for `from_arrays()` and `from_dict()` methods (26 new tests)
  - Empty data, single Gaussian, shape mismatches, format boundary values
  - Missing/extra dictionary keys, device/dtype handling, format preset edge cases

## v0.2.5 (SOG Format Support & API Improvements)

### New Features
- **SOG Format Reader**: Read SOG (Splat Ordering Grid) format files
  - `sogread(file_path | bytes)` - Read SOG files from path or bytes (requires `gsply[sogs]`)
  - Returns `GSData` container (same as `plyread()`) for consistent API
  - Supports `.sog` ZIP bundles and folder formats
  - **In-memory ZIP extraction**: Can read directly from bytes without disk I/O
  - Uses `imagecodecs` (fastest WebP decoder) for optimal performance
  - Compatible with PlayCanvas splat-transform format
- **Object-Oriented I/O API**: Convenient save/load methods for GSData and GSTensor
  - `data.save(file_path, compressed=False)` - Instance method wrapping `plywrite()` for object-oriented API
  - `GSData.load(file_path)` - Classmethod wrapping `plyread()` (auto-detects format)
  - `gstensor.save(file_path, compressed=True)` - Instance method for saving GSTensor (GPU compression by default)
  - `gstensor.save_compressed(file_path)` - Convenience alias for compressed saves
  - `GSTensor.load(file_path, device='cuda')` - Classmethod for loading GSTensor (auto-detects format, uses GPU decompression for compressed files)
  - Provides cleaner object-oriented API while maintaining backward compatibility with module-level functions
- **Format Conversion API**: Elegant in-place operations for PLY format conversion
  - `GSData.normalize(inplace=True)` - Convert linear scales/opacities to PLY-compatible log/logit format
  - `GSData.denormalize(inplace=True)` - Convert PLY format back to linear scales/opacities
  - `GSTensor.normalize(inplace=True)` - GPU version of normalize
  - `GSTensor.denormalize(inplace=True)` - GPU version of denormalize
  - Supports both in-place modification and copy creation
  - Uses optimized Numba-accelerated functions (CPU) and PyTorch CUDA kernels (GPU)
- **Color Conversion API**: In-place SH ↔ RGB conversion methods
  - `data.to_rgb(inplace=True)` - Convert sh0 from SH format to RGB colors (Numba-optimized CPU)
  - `data.to_sh(inplace=True)` - Convert sh0 from RGB format to SH coefficients (Numba-optimized CPU)
  - `gstensor.to_rgb(inplace=True)` - GPU version of to_rgb
  - `gstensor.to_sh(inplace=True)` - GPU version of to_sh
  - True in-place operations (modifies arrays/tensors directly without intermediate copies)

### API Improvements
- **Object-Oriented I/O**: Added save/load methods to GSData and GSTensor for cleaner API
  - Module-level functions (`plyread`, `plywrite`) remain available for functional style
  - Lazy imports prevent circular dependencies with `writer.py` and `reader.py`
  - `GSTensor.save()` uses GPU compression by default for optimal performance
- **Refactored conversion methods**: `to_ply_data()` and `from_ply_data()` now use `normalize()`/`denormalize()` internally
  - More consistent API design
  - Better support for in-place operations
  - Clearer method names (`normalize`/`denormalize` vs `to_ply_data`/`from_ply_data`)
- **Format tracking**: Internal `_format` dictionary tracks data format state
  - Uses `TypedDict` for type safety and IDE autocomplete
  - Tracks scales (PLY/linear), opacities (PLY/linear), sh0 (SH/RGB), and SH order
  - Automatically set during I/O operations and format conversions
- **Simplified dependencies**: SOG support now requires only `imagecodecs` (removed fallback libraries)
- **API consistency**: SOG reader returns `GSData` container matching `plyread()` behavior

### Performance Improvements
- **SOG reading**: In-memory reading from bytes is ~6x faster than file path reading
- **CPU utilities**: Enhanced `logit()` and `sigmoid()` functions with Numba parallel JIT compilation

### Code Cleanup
- **Removed redundant code**: Eliminated `torch/utils.py` wrapper module
  - GPU operations now use PyTorch functions directly (`torch.logit`, `torch.sigmoid`)
  - Reduced code duplication
  - Simpler import structure
- **Optimized CPU utilities**: Enhanced `logit()` and `sigmoid()` functions
  - Numba parallel JIT compilation for better performance
  - Both functions are now part of the public API

### Documentation
- Added comprehensive documentation for `save()` and `load()` methods in README and API reference
- Added documentation for `normalize()` and `denormalize()` methods
- Added documentation for `to_rgb()` and `to_sh()` color conversion methods
- Added documentation for `logit()` and `sigmoid()` utility functions
- Updated API reference with examples for object-oriented I/O
- Updated AGENTS.md with implementation details for save/load methods

### Dependencies
- Added optional `sogs` dependency group: `pip install gsply[sogs]`
  - Installs `imagecodecs>=2024.0.0` for WebP decoding

---

## v0.2.4 (GPU I/O API & Performance Optimizations)

### New Features
- **GPU I/O API**: Direct GPU compression/decompression functions
  - `plyread_gpu(file_path, device='cuda')` - Read compressed PLY directly to GPU
    - 4-5x faster than CPU decompression + GPU transfer
    - Direct GPU memory allocation (no intermediate CPU copies)
    - Optimized batch memory transfer (1.71x speedup)
    - ~19ms for 365K Gaussians (19 M/s throughput)
  - `plywrite_gpu(file_path, gstensor, compressed=True)` - Write GSTensor using GPU compression
    - 4-5x faster compression than CPU Numba
    - GPU reduction for chunk bounds (instant)
    - Minimal CPU-GPU data transfer
    - ~18ms for 365K Gaussians (20 M/s throughput)
  - Lazy import pattern - PyTorch only loaded when functions are accessed
  - Consistent API style matching `plyread()`/`plywrite()`

### Performance Improvements
- **GPU Compression**: Full GPU-accelerated compression pipeline
  - Optimized memory transfers (batch transfer reduces DMA overhead)
  - Pre-computed ranges for quantization
  - Vectorized chunk bounds computation
- **CPU Compression**: Pre-compute ranges optimization
  - 1.44x speedup by computing ranges once per chunk instead of per-vertex
  - Eliminates redundant calculations in packing loops

### API Changes
- New top-level functions: `gsply.plyread_gpu()` and `gsply.plywrite_gpu()`
  - Available via lazy import (PyTorch not required unless used)
  - Returns `GSTensor` instead of `GSData` for GPU operations
  - Only supports compressed format (GPU path optimized for this)

### Documentation
- Added GPU I/O API documentation to Sphinx docs
- Updated API reference with performance metrics
- Added examples for GPU workflows

---

## v0.2.2 (Data Concatenation & Performance)

### New Features
- **Data Concatenation**: Bulk merge operations
  - `GSData.concatenate([data1, data2, data3])` - 6.15x faster than loops
  - `GSData.add(other)` - Optimized pairwise (1.9x faster)
  - `GSTensor.add(other)` - GPU concatenation (18x faster than CPU)
- **Performance Optimization**:
  - `make_contiguous()` - Fix cache locality (2-45x speedup for operations)
  - `is_contiguous()` - Check array layout
  - Direct masked GPU transfer (no intermediate CPU copies)
- **Mask Management**:
  - Multi-layer boolean masks with named layers
  - GPU-optimized mask operations (100-1000x faster)
  - Automatic mask merging during concatenation

---

## v0.2.0 (Breaking Changes & New Features)

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

# New (v0.2.0+)
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
