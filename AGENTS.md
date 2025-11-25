# AGENTS.md

This file provides context and instructions for AI coding agents working on the gsply project.

## Project Overview

**gsply** is an ultra-fast Gaussian Splatting PLY I/O library for Python.

- **Language**: Pure Python (3.10+)
- **Core Dependencies**: NumPy, Numba (JIT acceleration)
- **Optional Dependencies**: PyTorch (GPU acceleration via GSTensor)
- **Performance**: 93M Gaussians/sec read, 57M Gaussians/sec write
- **Key Features**: Zero-copy optimization, compressed format support, GPU integration, SOG format support
- **Current Version**: 0.2.7

## Development Environment Setup

### Installation

```bash
# Clone repository
git clone https://github.com/OpsiClear/gsply.git
cd gsply

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: Install PyTorch for GPU features (testing GSTensor)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Using uv (preferred by maintainer)

Per CLAUDE.md instructions, the maintainer prefers `uv` for running scripts:

```bash
# Run tests
uv run pytest

# Run scripts
uv run python script.py

# Install pre-commit (for git hooks)
uv pip install pre-commit
pre-commit install
```

### Pre-commit Hooks

The project uses pre-commit to ensure code quality before commits:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files

# Run mypy manually (not run automatically due to pre-existing issues)
pre-commit run --hook-stage manual mypy --all-files

# Update hook versions
pre-commit autoupdate
```

**Hooks configured:**
- Trailing whitespace removal
- End of file fixing
- YAML/TOML validation
- Large file detection
- Merge conflict detection
- Line ending normalization (LF)
- Ruff linter and formatter
- MyPy type checking (manual stage)

### Project Structure

```
gsply/
├── src/gsply/           # Main package
│   ├── __init__.py      # Public API exports
│   ├── reader.py        # PLY reading (plyread, decompress)
│   ├── writer.py        # PLY writing (plywrite, compress)
│   ├── gsdata.py        # GSData dataclass (CPU container)
│   ├── formats.py       # Format detection and constants
│   ├── utils.py         # Utility functions (sh2rgb, rgb2sh, logit, sigmoid, apply_pre_activations, apply_pre_deactivations)
│   ├── sog_reader.py    # SOG format reading (sogread, optional dependency)
│   ├── py.typed         # Type checking marker (PEP 561)
│   └── torch/           # Optional PyTorch integration
│       ├── __init__.py  # Conditional import (checks torch availability)
│       ├── gstensor.py  # GSTensor GPU dataclass
│       ├── compression.py  # GPU compression/decompression
│       └── io.py        # GPU I/O (plyread_gpu, plywrite_gpu)
├── tests/               # Test suite (365 tests)
├── benchmarks/          # Performance benchmarks
├── docs/                # Documentation
└── .github/workflows/   # CI/CD pipelines
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gsply --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run tests matching pattern
pytest -k "test_plyread"

# Run PyTorch tests only (requires torch installed)
pytest -k "torch or gstensor"
```

### Test Data Generation

Tests automatically generate synthetic data. Some tests use real PLY files:
- Set `GSPLY_TEST_DATA_DIR` environment variable to point to real PLY files (optional)
- Tests will create `export_with_edits/` directory with synthetic data

### CI/CD

- **Location**: `.github/workflows/test.yml`
- **Matrix Testing**: Ubuntu, Windows, macOS × Python 3.10, 3.11, 3.12, 3.13
- **Separate Torch Tests**: `.github/workflows/test.yml` has dedicated `test-torch` job
- **Benchmarks**: `.github/workflows/benchmark.yml` runs performance tests
- **All tests must pass** before merging

### Test Count

Current test count: **365 tests** (documented in README.md)
- Update this count in README if adding/removing tests
- Includes 17 new tests for fused activation kernels (tests/test_pre_activations.py)

## Code Style and Conventions

### Type Hints

- **Required**: Use Python 3.12+ type hint syntax
- **Tool**: `tyro` for type management and CLI (per CLAUDE.md)
- **Example**: `list[int]` not `List[int]`, `dict[str, Any]` not `Dict[str, Any]`

### CLI Argument Parsing

- **Use argparse** for this project (per CLAUDE.md)
- Not tyro for CLI despite being preferred for type management

### Logging

- **Use logging module** instead of print statements for info (per CLAUDE.md)
- Example: `logging.info()` not `print()`

### Unicode and Emojis

- **Never use emojis or Unicode characters** in code or output (per CLAUDE.md)
- Use ASCII only: `[OK]`, `[FAIL]`, `->` instead of ✓, ✅, ❌, →

### Code Formatting

- **Linter**: Ruff (`ruff check .`)
- **Type Checker**: MyPy (`mypy src/`)
- **Line Length**: 100 characters (conventional Python)
- Follow PEP 8 conventions

### Naming Conventions

- **Variables**: `snake_case`
- **Classes**: `PascalCase` (GSData, GSTensor)
- **Constants**: `UPPER_SNAKE_CASE` (SH_C0)
- **Private**: Prefix with `_` (_base, _recreate_from_base)

### Documentation Style

- **Docstrings**: Google style with type hints
- **Example format**:
  ```python
  def plyread(file_path: str | Path) -> GSData:
      """Read Gaussian Splatting PLY file (auto-detects format).

      Args:
          file_path: Path to PLY file

      Returns:
          GSData dataclass with Gaussian parameters

      Example:
          >>> data = plyread("model.ply")
          >>> print(len(data))
      """
  ```

## Important Implementation Notes

### Zero-Copy Optimization

All PLY reads use zero-copy views via `_base` array:
- `_base` is a private field (N, P) that holds all properties in single array
- Individual fields (means, scales, etc.) are views into `_base`
- **Never modify `_base` directly**
- When creating GSData manually, set `_base=None`

### NumPy Array Layouts

Standard layout for Gaussian properties:
```
means:     (N, 3) - xyz positions
scales:    (N, 3) - log scales (PLY format) or linear scales
quats:     (N, 4) - quaternions (wxyz order)
opacities: (N,)   - logit opacities (PLY format) or linear opacities
sh0:       (N, 3) - DC spherical harmonics
shN:       (N, K, 3) - Higher-order SH (K=0/9/24/45 for degree 0/1/2/3)
masks:     (N,)   - boolean mask (initialized to all True)
```

**Format Conversion:**
- PLY files store scales in log-space and opacities in logit-space
- Use `normalize()` to convert linear → PLY format (uses fused kernel, ~8-15x faster)
- Use `denormalize()` to convert PLY → linear format (uses fused kernel, ~8-15x faster)
- Both `GSData` and `GSTensor` have identical format conversion APIs
- Default is `inplace=True` for efficiency
- Direct access: `apply_pre_activations()` / `apply_pre_deactivations()` for fine-grained control (v0.2.7+)

### PyTorch Integration

- PyTorch is **optional** (soft dependency via lazy import)
- **GSTensor uses lazy import** via `__getattr__` in `__init__.py`
  - This prevents torch from loading when just importing gsply
  - Avoids torch-related errors in CI when torch isn't needed
  - Only imports torch when `gsply.GSTensor`, `gsply.plyread_gpu`, or `gsply.plywrite_gpu` is accessed
- Use `try/except ImportError` for torch imports in modules
- Tests use `pytest.importorskip("torch")` to skip when unavailable
- GSTensor features only work if user installs PyTorch separately
- **Never add PyTorch to dependencies** in pyproject.toml
- **Python 3.13 not supported** by PyTorch yet - exclude from torch test matrix

### GPU I/O API (v0.2.4+)

**New Functions: `plyread_gpu()` and `plywrite_gpu()`**
- Located in `src/gsply/torch/io.py`
- Exported via lazy import in `src/gsply/__init__.py` (matches GSTensor pattern)
- Provides direct GPU I/O for compressed PLY files
- **Performance**: 4-5x faster than CPU decompress + GPU transfer
- **API Style**: Matches `plyread()`/`plywrite()` for consistency
- **Tests**: `tests/test_gpu_io_api.py` (5 tests covering API, roundtrip, lazy import)
- **Implementation**: Uses `read_compressed_gpu()` and `write_compressed_gpu()` from `compression.py`

### SOG Format Support (v0.2.5+)

**SOG Reader: `sogread()`**
- Located in `src/gsply/sog_reader.py`
- Exported via lazy import in `src/gsply/__init__.py` (optional dependency)
- Reads SOG (Splat Ordering Grid) format files (PlayCanvas splat-transform compatible)
- **Returns**: `GSData` container (same as `plyread()`) for consistent API
- **Supports**: Both `.sog` ZIP bundles and folder formats
- **In-memory**: Can read directly from bytes without disk I/O
- **Dependencies**: Requires `gsply[sogs]` which installs `imagecodecs` (fastest WebP decoder)
- **Performance**: ~6x faster when reading from bytes vs file path
- **API**: `sogread(file_path | bytes) -> GSData`

### Numba JIT Acceleration

- Numba is **required** dependency for performance
- Used for parallel bit packing/unpacking in compressed format
- Functions decorated with `@numba.jit` should be pure functions
- Also used for SOG format decoding (means, scales, quats, colors, SHN)

### Format Conversion: Linear ↔ PLY Format (v0.2.5+)

**GSData and GSTensor Format Conversion Methods:**
- `normalize(inplace=True)`: Convert linear → PLY format
  - Linear scales → log-scales: `log(scale)` with clamping
  - Linear opacities → logit-opacities: `logit(opacity, eps=1e-4)` with clamping
  - **Uses fused kernel**: `apply_pre_deactivations()` internally (~8-15x faster)
- `denormalize(inplace=True)`: Convert PLY → linear format
  - Log-scales → linear scales: `exp(log_scale)` with clamping
  - Logit-opacities → linear opacities: `sigmoid(logit)`
  - Quaternions → normalized quaternions (with safety floor)
  - **Uses fused kernel**: `apply_pre_activations()` internally (~8-15x faster)

**Auto-Detection (v0.2.6+):**
- `__post_init__` automatically detects format from data values if `_format` is not provided
- Uses heuristics (negative values for log-scales, range for opacities) via `_detect_format_from_values()`
- `plywrite()` validates/converts to PLY format automatically before writing

**Implementation Details (v0.2.7+):**
- **GSData**: Uses fused Numba kernels (`apply_pre_activations()`, `apply_pre_deactivations()`) from `utils.py`
  - Parallel Numba JIT compilation with `@njit(parallel=True, fastmath=True, cache=True, nogil=True)`
  - Single-pass processing reduces memory overhead and improves cache locality
  - Scales and opacities processed together in single fused pass
  - Quaternion normalization included in activation kernel (denormalize only)
- **GSTensor**: Uses PyTorch operations (`torch.logit()`, `torch.sigmoid()`, `torch.exp()`) for GPU acceleration
- **Consistency**: Both use `eps=1e-4` for logit operations to ensure identical behavior
- **Default**: `inplace=True` for efficiency (modifies object in-place)
- **Constants**: `min_scale=1e-9`, `min_opacity=1e-4`, `max_opacity=1.0-1e-4` (same for both)
- **Performance**: ~8-15x faster than sequential operations due to fused kernels

**Direct Fused Kernel Access (v0.2.7+):**
- `apply_pre_activations(data, min_scale=1e-4, max_scale=100.0, min_quat_norm=1e-8, inplace=True)`: Direct access to activation kernel
  - Processes scales, opacities, and quaternions in single fused pass
  - Useful for fine-grained control over activation parameters
- `apply_pre_deactivations(data, min_scale=1e-9, min_opacity=1e-4, max_opacity=0.9999, inplace=True)`: Direct access to deactivation kernel
  - Processes scales and opacities in single fused pass
  - Useful for fine-grained control over deactivation parameters

**When to use:**
- `normalize()`: Before saving to PLY format if you have linear data (uses fused kernel automatically)
- `denormalize()`: After loading PLY files if you need linear values for computation/visualization (uses fused kernel automatically)
- `apply_pre_activations()` / `apply_pre_deactivations()`: For direct access when you need custom bounds or fine-grained control

### Object-Oriented I/O Methods (v0.2.5+)

**GSData and GSTensor File I/O:**
- `data.save(file_path, compressed=False)`: Instance method wrapping `plywrite()` for object-oriented API
- `GSData.load(file_path)`: Classmethod wrapping `plyread()` for object-oriented API (auto-detects format)
- `gstensor.save(file_path, compressed=True)`: Instance method for saving GSTensor (GPU compression by default)
- `gstensor.save_compressed(file_path)`: Convenience alias for compressed saves
- `GSTensor.load(file_path, device='cuda')`: Classmethod for loading GSTensor (auto-detects format, uses GPU decompression for compressed files)

**Implementation Details:**
- **GSData.save()**: Wraps `plywrite()` with module-level import to avoid circular dependencies
- **GSData.load()**: Wraps `plyread()` with module-level import to avoid circular dependencies
- **GSTensor.save()**: Uses `write_compressed_gpu()` for compressed saves, converts to GSData for uncompressed saves
- **GSTensor.load()**: Uses `read_compressed_gpu()` for compressed files, `plyread()` + `from_gsdata()` for uncompressed files
- **Lazy imports**: `plyread` and `plywrite` are imported inside methods to prevent circular dependencies with `writer.py` and `reader.py`

**When to use:**
- Use `data.save()` / `GSData.load()` for cleaner object-oriented API
- Use `gstensor.save()` / `GSTensor.load()` for GPU-accelerated I/O
- Module-level functions (`plyread`, `plywrite`) still available for functional style

### Factory Methods for External Data (v0.2.6+)

**GSData and GSTensor Factory Methods:**
- `GSData.from_arrays(means, scales, quats, opacities, sh0, shN=None, format='auto', sh_degree=None, sh0_format=SH0_SH)`: Create GSData from NumPy arrays with format preset
- `GSData.from_dict(data_dict, format='auto', sh_degree=None, sh0_format=SH0_SH)`: Create GSData from dictionary with format preset
- `GSTensor.from_arrays(means, scales, quats, opacities, sh0, shN=None, format='auto', sh_degree=None, sh0_format=SH0_SH, device=None, dtype=None)`: Create GSTensor from PyTorch tensors with format preset
- `GSTensor.from_dict(data_dict, format='auto', sh_degree=None, sh0_format=SH0_SH, device='cuda', dtype=None)`: Create GSTensor from dictionary with format preset

**Format Presets:**
- `"auto"` (default): Automatically detects format from data values using heuristics
  - Detects PLY format (log-scales/logit-opacities) vs Linear format (linear scales/opacities)
  - Uses `_detect_format_from_values()` for statistical analysis
- `"ply"`: Explicitly sets PLY format (log-scales/logit-opacities)
- `"linear"` or `"rasterizer"`: Explicitly sets linear format (linear scales/opacities)

**SH Degree Inference:**
- Automatically infers SH degree from `shN.shape[1]` if `sh_degree` is not provided
- Uses `SH_BANDS_TO_DEGREE` mapping: 0→0, 9→1, 24→2, 45→3
- Raises `ValueError` if `shN` shape doesn't match a valid SH degree

**Implementation Details:**
- **GSData.from_arrays()**: Validates array shapes, detects/infers format and SH degree, performs format conversion if needed
- **GSData.from_dict()**: Extracts arrays from dictionary, validates required keys (`means`, `scales`, `quats`, `opacities`, `sh0`), optional `shN`
- **GSTensor.from_arrays()**: Same as GSData but handles PyTorch tensors, device/dtype conversion, and GPU format conversion
- **GSTensor.from_dict()**: Same as GSData but extracts PyTorch tensors from dictionary
- **Format Conversion**: If format preset differs from auto-detected format, automatically converts data using `normalize()` or `denormalize()`
- **Shape Validation**: Raises `ValueError` if input arrays/tensors have inconsistent lengths (N mismatch)

**When to use:**
- `from_arrays()`: When you have individual arrays/tensors and want convenient initialization with format handling
- `from_dict()`: When you have data in dictionary format (e.g., from JSON, other libraries, or custom loaders)
- Use format presets when you know the data format and want to avoid auto-detection overhead
- Use `"auto"` when format is uncertain - heuristics handle most cases correctly

**Examples:**
```python
# Auto-detect format from arrays
data = GSData.from_arrays(means, scales, quats, opacities, sh0)

# Explicit PLY format (log/logit)
data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="ply")

# From dictionary with auto-detection
data = GSData.from_dict({"means": means, "scales": scales, ...})

# GPU tensor with device specification
gstensor = GSTensor.from_arrays(means_tensor, scales_tensor, ..., device="cuda")
```

### Color Conversion: SH ↔ RGB Format

**GSData and GSTensor Color Conversion Methods:**
- `to_rgb(inplace=True)`: Convert sh0 from SH format to RGB colors
  - SH DC coefficients → RGB colors: `rgb = sh0 * SH_C0 + 0.5`
- `to_sh(inplace=True)`: Convert sh0 from RGB format to SH coefficients
  - RGB colors → SH DC coefficients: `sh0 = (rgb - 0.5) / SH_C0`

**Implementation Details:**
- **GSData**: Uses Numba JIT-compiled parallel functions (`_sh2rgb_inplace_jit`, `_rgb2sh_inplace_jit`) from `utils.py`
  - Always uses Numba for optimal performance (optimized for >100K Gaussians)
  - True in-place operations (modifies array directly without intermediate copies)
- **GSTensor**: Uses PyTorch in-place operations (`mul_()`, `add_()`, `sub_()`, `div_()`) for GPU acceleration
  - True in-place operations (modifies tensor directly without intermediate copies)
- **Default**: `inplace=True` for efficiency (modifies object in-place)
- **Format Tracking**: Updates `_format` dict with `SH0_SH` or `SH0_RGB` enum values

**When to use:**
- `to_rgb()`: When you need RGB colors in [0, 1] range for visualization or color manipulation
- `to_sh()`: When you have RGB colors and need to convert back to SH format for PLY file compatibility

### Compression APIs (In-Memory)

**Functions: `compress_to_bytes()`, `compress_to_arrays()`, `decompress_from_bytes()`**
- Located in `src/gsply/writer.py` and `src/gsply/reader.py`
- Exported in `src/gsply/__init__.py` (in `__all__`)
- Provides in-memory compression/decompression without disk I/O
- **Use cases**: Network transfer, streaming, custom storage solutions
- **Performance**: Same as file-based compression (uses same JIT-accelerated code)
- **API**:
  - `compress_to_bytes(data: GSData) -> bytes`: Compress GSData to bytes
  - `compress_to_arrays(data: GSData) -> tuple`: Returns (header, chunks, packed, sh) arrays
  - `decompress_from_bytes(bytes) -> GSData`: Decompress bytes to GSData
- **Backward compatibility**: Also accepts individual arrays (not just GSData)

### Utility Functions

**Functions: `sh2rgb()`, `rgb2sh()`, `logit()`, `sigmoid()`, `apply_pre_activations()`, `apply_pre_deactivations()`, `SH_C0`**
- Located in `src/gsply/utils.py`
- Exported in `src/gsply/__init__.py` (in `__all__`)
- **sh2rgb()**: Convert spherical harmonics to RGB colors (creates new array)
- **rgb2sh()**: Convert RGB colors to spherical harmonics (creates new array)
- **logit()**: Compute logit function (inverse sigmoid) with numerical stability (Numba-optimized)
  - Default `eps=1e-6` for numerical stability
  - Used internally by `apply_pre_deactivations()` with `eps=1e-4` (via `normalize()`)
- **sigmoid()**: Compute sigmoid function (inverse logit) with numerical stability (Numba-optimized)
  - Used internally by `apply_pre_activations()` (via `denormalize()`)
- **apply_pre_activations(data, min_scale=1e-4, max_scale=100.0, min_quat_norm=1e-8, inplace=True)**: Fused activation kernel (v0.2.7+)
  - Processes scales, opacities, and quaternions in single parallel pass
  - Uses Numba `@njit(parallel=True, fastmath=True, cache=True, nogil=True)`
  - ~8-15x faster than sequential operations
  - Used internally by `GSData.denormalize()` for optimal performance
  - Direct access available for fine-grained control over activation parameters
- **apply_pre_deactivations(data, min_scale=1e-9, min_opacity=1e-4, max_opacity=0.9999, inplace=True)**: Fused deactivation kernel (v0.2.7+)
  - Processes scales and opacities in single parallel pass
  - Uses Numba `@njit(parallel=True, fastmath=True, cache=True, nogil=True)`
  - ~8-15x faster than sequential operations
  - Used internally by `GSData.normalize()` for optimal performance
  - Direct access available for fine-grained control over deactivation parameters
- **SH_C0**: Constant for spherical harmonic DC coefficient normalization (0.28209479177387814)

**Internal Numba Functions (for in-place operations):**
- `_activate_gaussians_numba()`: Fused Numba kernel for activating scales, opacities, and quaternions (used by `apply_pre_activations()`)
- `_deactivate_gaussians_numba()`: Fused Numba kernel for deactivating scales and opacities (used by `apply_pre_deactivations()`)
- `_sh2rgb_inplace_jit()`: Numba JIT-compiled in-place SH to RGB conversion (used by `GSData.to_rgb()`)
- `_rgb2sh_inplace_jit()`: Numba JIT-compiled in-place RGB to SH conversion (used by `GSData.to_sh()`)
- All utility functions are CPU-only (Numba JIT-compiled)

### Concatenation Optimizations (v0.2.2)

**GSData.add() and GSData.concatenate()**
- `add()`: Pairwise concatenation using pre-allocation + direct assignment
  - 1.9x faster than np.concatenate for 500K Gaussians (99 M/s)
  - Handles mask layer merging automatically
  - Validates SH degree compatibility via `get_sh_degree()`
- `+` operator: Pythonic concatenation syntax
  - `data1 + data2` delegates to `data1.add(data2)` (same performance)
  - `__radd__` enables `sum([data1, data2, data3])` to work
  - Note: For >= 3 arrays, use `GSData.concatenate()` for 5.74x speedup
- `concatenate()`: Bulk concatenation for multiple arrays
  - 6.15x faster than repeated `add()` calls
  - Single allocation vs N-1 intermediate allocations
  - Use for >= 2 arrays
- **Implementation details**:
  - Pre-allocate output arrays with `np.empty()`
  - Use slice assignment: `arr[:n1] = data1.arr`
  - Preserve dtype, handle mask layer merging
  - _base optimization path when both have compatible _base

**GSTensor.add() - GPU Concatenation**
- GPU-optimized using `torch.cat()`
- 18.23x faster than CPU for 500K Gaussians
- Automatic device/dtype handling
- Preserves `requires_grad` flag
- Handles mask layer concatenation with GPU tensors
- `+` operator: Same as GSData, `gstensor1 + gstensor2` delegates to `add()`
- `sum()` works with GSTensor objects via `__radd__`

**Mask Layer Management**
- Both GSData and GSTensor support multi-layer boolean masks
- Methods: `add_mask_layer()`, `get_mask_layer()`, `remove_mask_layer()`, `combine_masks()`, `apply_masks()`
- GSTensor uses `torch.all()`/`torch.any()` (100-1000x faster than CPU Numba)
- Masks persist through slicing and device transfers
- `mask_names` field tracks layer names (list[str])

### Contiguity Optimizations (v0.2.2)

**Problem: Non-Contiguous Arrays from plyread()**
- PLY files load into interleaved `_base` array (zero-copy)
- Creates non-contiguous views with stride=56 instead of 12
- 2-45x performance penalty for array operations

**GSData.make_contiguous()**
- Converts non-contiguous arrays to contiguous layout
- **Break-even**: ~8 operations (measured threshold)
- **Speedups (100K Gaussians)**:
  - argmax(): 45.5x faster
  - max/min(): 18-19x faster
  - sum/mean(): 6-7x faster
  - Element-wise: 2-4x faster
- **Conversion cost**:
  - 10K: 0.14 ms
  - 100K: 2.2 ms
  - 1M: 25 ms
- **Decision rule**: Convert if >= 8 operations
- **Memory**: Zero overhead (same total, reorganized)
- **Usage**: `inplace=True` modifies object, `inplace=False` returns copy

**GSData.is_contiguous()**
- Checks if all arrays are C-contiguous
- Returns boolean
- Useful for conditional optimization

**Direct Masked GPU Transfer**
- `GSTensor.from_gsdata(data, mask=mask, device="cuda")`
- Filters data during GPU transfer (no intermediate CPU copy)
- More efficient than `data[mask]` then `from_gsdata()`

**Implementation Considerations**:
- Always check contiguity before expensive operations
- Document contiguity assumptions in function docstrings
- Preserve contiguity through operations when possible
- Test both contiguous and non-contiguous paths

## Pull Request Guidelines

### Before Creating PR

1. **Run pre-commit hooks**: `pre-commit run --all-files` (automatically checks formatting, linting, etc.)
2. **Run full test suite**: `pytest` (all 365 tests must pass)
3. **Type check** (optional): `mypy src/` or `pre-commit run --hook-stage manual mypy --all-files`
4. **Update test count** in README.md if you added/removed tests
5. **Update CHANGELOG.md** with your changes
6. **Run benchmark** if performance-critical code changed

Note: Pre-commit hooks will run automatically on commit if installed via `pre-commit install`

### PR Title Format

Follow conventional commits style:
- `feat: Add new feature`
- `fix: Fix bug in plyread`
- `perf: Optimize GPU transfer`
- `docs: Update API reference`
- `test: Add tests for GSTensor`
- `refactor: Simplify reader logic`

### Code Review Checklist

- [ ] All tests pass (365/365)
- [ ] No new linter warnings
- [ ] Type hints added for new functions
- [ ] Docstrings added for public APIs
- [ ] Examples updated if API changed
- [ ] Performance not regressed (if applicable)
- [ ] Cross-platform compatible (Windows, macOS, Linux)
- [ ] Python 3.10-3.13 compatible
- [ ] Format conversion consistency: GSData and GSTensor use same constants/eps values

### Commits

- **Do not commit** unless explicitly requested by maintainer (per CLAUDE.md)
- This is very important - wait for explicit permission
- Maintainer prefers to review changes before committing

## Build and Release Process

### Local Build

```bash
# Build package
python -m build

# Check dist files
ls dist/
# gsply-0.2.7-py3-none-any.whl
# gsply-0.2.7.tar.gz
```

### Publishing (Maintainer Only)

```bash
# Test PyPI
twine upload --repository testpypi dist/*

# Production PyPI
twine upload dist/*
```

### Version Bumping

1. Update version in `pyproject.toml`
2. Update `__version__` in `src/gsply/__init__.py`
3. Update CHANGELOG.md with release notes
4. Update test count in README.md and AGENTS.md if tests changed
5. Create git tag: `git tag v0.2.X`

## API Design Principles

### Consistency

- All read operations return `GSData` dataclass (`plyread()`, `sogread()`)
- All write operations accept individual arrays OR GSData
- Use `unpack()` pattern for tuple unpacking, not indexing
- GPU operations return `GSTensor`, CPU operations return `GSData`
- Format conversion methods (`normalize()`, `denormalize()`) are identical between GSData and GSTensor
- GSData uses fused Numba kernels (`apply_pre_activations()`, `apply_pre_deactivations()`) for ~8-15x faster format conversion (v0.2.7+)
- GSTensor uses PyTorch operations (`torch.logit()`, `torch.sigmoid()`, `torch.exp()`) for GPU acceleration
- Both use same constants (`min_scale=1e-9`, `min_opacity=1e-4`, `max_opacity=1.0-1e-4`, `eps=1e-4` for logit)
- Compression APIs (`compress_to_bytes()`, `compress_to_arrays()`, `decompress_from_bytes()`) work with GSData or individual arrays
- Utility functions (`sh2rgb()`, `rgb2sh()`, `logit()`, `sigmoid()`, `apply_pre_activations()`, `apply_pre_deactivations()`, `SH_C0`) are always available (CPU-only)

### Backward Compatibility

- Maintain support for individual array arguments (not just GSData)
- Example: `plywrite()` accepts both patterns:
  ```python
  # Pattern 1: Individual arrays
  plywrite("out.ply", means, scales, quats, opacities, sh0, shN)

  # Pattern 2: Unpacked GSData
  plywrite("out.ply", *data.unpack())
  ```

### Performance First

- Always prefer zero-copy when possible
- Document performance characteristics in docstrings
- Benchmark any performance-critical changes
- Current benchmarks are in README.md "Performance" section

## Common Workflows

### Adding a New Feature

1. Add implementation in appropriate module (`reader.py`, `writer.py`, `utils.py`, etc.)
2. Export in `src/gsply/__init__.py` if public API
3. Add to `__all__` list (or use lazy import via `__getattr__` for optional dependencies)
4. Write tests in `tests/test_*.py`
5. Add documentation to README.md API Reference section
6. Update CHANGELOG.md
7. Update AGENTS.md if adding new major features or APIs

### Fixing a Bug

1. Add regression test that fails
2. Fix the bug
3. Verify test passes
4. Update CHANGELOG.md under "Bug Fixes"

### Performance Optimization

1. Benchmark current performance
2. Implement optimization
3. Benchmark new performance
4. Document improvement in CHANGELOG.md
5. Update README.md performance metrics if significant

## Known Constraints

- **Windows Path Handling**: Use `Path` from pathlib for cross-platform compatibility
- **Numba Compatibility**: Some NumPy operations not supported in JIT functions
- **PyTorch Versions**: Test with PyTorch 2.0+ if modifying GSTensor
- **Python 3.13**: PyTorch not yet supported - exclude from torch test matrix
- **Memory**: Large files (>1M Gaussians) need testing for memory efficiency
- **SH Degrees**: Support degrees 0-3 only (14, 23, 38, 59 properties)
- **Optional Dependencies**:
  - PyTorch: Required only for `GSTensor`, `plyread_gpu()`, `plywrite_gpu()`
  - `gsply[sogs]`: Required only for `sogread()` (installs `imagecodecs`)
- **Type Checking**: `py.typed` marker file exists for PEP 561 compliance

## Contact

- **Maintainer**: OpsiClear (yehe@opsiclear.com)
- **Repository**: https://github.com/OpsiClear/gsply
- **Issues**: Use GitHub Issues for bug reports and feature requests
