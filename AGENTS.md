# AGENTS.md

This file provides context and instructions for AI coding agents working on the gsply project.

## Project Overview

**gsply** is an ultra-fast Gaussian Splatting PLY I/O library for Python.

- **Language**: Pure Python (3.10+)
- **Core Dependencies**: NumPy, Numba (JIT acceleration)
- **Optional Dependencies**: PyTorch (GPU acceleration via GSTensor)
- **Performance**: 93M Gaussians/sec read, 87M Gaussians/sec write
- **Key Features**: Zero-copy optimization, compressed format support, GPU integration

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
```

### Project Structure

```
gsply/
├── src/gsply/           # Main package
│   ├── __init__.py      # Public API exports
│   ├── reader.py        # PLY reading (plyread, decompress)
│   ├── writer.py        # PLY writing (plywrite, compress)
│   ├── gsdata.py        # GSData dataclass (CPU container)
│   ├── formats.py       # Format detection and constants
│   ├── utils.py         # Utility functions (sh2rgb, rgb2sh)
│   └── torch/           # Optional PyTorch integration
│       ├── __init__.py  # Conditional import (checks torch availability)
│       └── gstensor.py  # GSTensor GPU dataclass
├── tests/               # Test suite (169 tests)
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

Current test count: **169 tests** (documented in README.md)
- Update this count in README if adding/removing tests

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
scales:    (N, 3) - log scales
quats:     (N, 4) - quaternions (wxyz order)
opacities: (N,)   - logit opacities
sh0:       (N, 3) - DC spherical harmonics
shN:       (N, K, 3) - Higher-order SH (K=0/9/24/45 for degree 0/1/2/3)
masks:     (N,)   - boolean mask (initialized to all True)
```

### PyTorch Integration

- PyTorch is **optional** (soft dependency)
- Use `try/except ImportError` for torch imports
- Tests use `pytest.importorskip("torch")` to skip when unavailable
- GSTensor features only work if user installs PyTorch separately
- **Never add PyTorch to dependencies** in pyproject.toml

### Numba JIT Acceleration

- Numba is **required** dependency for performance
- Used for parallel bit packing/unpacking in compressed format
- Functions decorated with `@numba.jit` should be pure functions

## Pull Request Guidelines

### Before Creating PR

1. **Run full test suite**: `pytest` (all 169 tests must pass)
2. **Type check**: `mypy src/`
3. **Lint**: `ruff check .`
4. **Update test count** in README.md if you added/removed tests
5. **Update CHANGELOG.md** with your changes
6. **Run benchmark** if performance-critical code changed

### PR Title Format

Follow conventional commits style:
- `feat: Add new feature`
- `fix: Fix bug in plyread`
- `perf: Optimize GPU transfer`
- `docs: Update API reference`
- `test: Add tests for GSTensor`
- `refactor: Simplify reader logic`

### Code Review Checklist

- [ ] All tests pass (169/169)
- [ ] No new linter warnings
- [ ] Type hints added for new functions
- [ ] Docstrings added for public APIs
- [ ] Examples updated if API changed
- [ ] Performance not regressed (if applicable)
- [ ] Cross-platform compatible (Windows, macOS, Linux)
- [ ] Python 3.10-3.13 compatible

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
# gsply-0.2.0-py3-none-any.whl
# gsply-0.2.0.tar.gz
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
4. Create git tag: `git tag v0.2.0`

## API Design Principles

### Consistency

- All read operations return `GSData` dataclass
- All write operations accept individual arrays OR GSData
- Use `unpack()` pattern for tuple unpacking, not indexing
- GPU operations return `GSTensor`, CPU operations return `GSData`

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

1. Add implementation in appropriate module (`reader.py`, `writer.py`, etc.)
2. Export in `src/gsply/__init__.py` if public API
3. Add to `__all__` list
4. Write tests in `tests/test_*.py`
5. Add documentation to README.md API Reference section
6. Update CHANGELOG.md

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
- **Memory**: Large files (>1M Gaussians) need testing for memory efficiency
- **SH Degrees**: Support degrees 0-3 only (14, 23, 38, 59 properties)

## Contact

- **Maintainer**: OpsiClear (yehe@opsiclear.com)
- **Repository**: https://github.com/OpsiClear/gsply
- **Issues**: Use GitHub Issues for bug reports and feature requests
