# Contributing to gsply

Complete guide for developers and contributors to gsply - the ultra-fast Gaussian Splatting PLY I/O library.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Building and Testing](#building-and-testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Release Process](#release-process)
- [Technical Implementation](#technical-implementation)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

### How to Contribute

We welcome contributions! Here's how to get started:

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your changes
4. **Make your changes** with appropriate tests
5. **Run tests and benchmarks** to verify
6. **Submit a pull request** with a clear description

### Code of Conduct

By contributing to gsply, you agree to be respectful and constructive in all interactions. We aim to maintain a welcoming environment for all contributors.

### Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Check existing issues and pull requests
- **Documentation**: Read README.md and docs/ files
- **Repository**: https://github.com/OpsiClear/gsply

---

## Development Setup

### Prerequisites

**Required:**
- Python 3.10 or higher (tested on 3.10, 3.11, 3.12, 3.13)
- pip or uv for package management
- git for version control

**Optional (for benchmarks):**
- open3d >= 0.17.0
- plyfile >= 0.9.0

### Installation from Source

```bash
# Clone the repository
git clone https://github.com/OpsiClear/gsply.git
cd gsply

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Or install all dependencies (including benchmarks)
pip install -e ".[all]"
```

### Virtual Environment

Always use a virtual environment for development:

```bash
# Create virtual environment
python -m venv .venv

# Activate (Unix/Linux/Mac)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Deactivate when done
deactivate
```

### Dependencies

gsply has minimal runtime dependencies:
- **numpy** >= 1.20.0 (only runtime dependency)

Development dependencies:
- **pytest** >= 7.0 (testing)
- **pytest-cov** (coverage reporting)
- **build** (package building)
- **twine** (PyPI publishing)

Benchmark dependencies:
- **open3d** >= 0.17.0 (comparison)
- **plyfile** >= 0.9.0 (comparison)
- **tyro** >= 0.8.0 (CLI parsing)

---

## Building and Testing

### Quick Build

Build gsply wheel and source distribution:

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build wheel and source distribution
python -m build

# Or use convenience scripts
./build.sh          # Unix/Linux/Mac
.\build.ps1         # Windows PowerShell
```

This creates:
- `dist/gsply-0.1.0.tar.gz` - Source distribution (~26KB)
- `dist/gsply-0.1.0-py3-none-any.whl` - Universal wheel (~15KB)

### Running Tests

gsply has 56 comprehensive unit tests:

```bash
# Run all tests (from gsply root directory)
pytest tests/ -v

# Run specific test file
pytest tests/test_reader.py -v
pytest tests/test_writer.py -v

# Run with coverage report
pytest tests/ -v --cov=gsply --cov-report=html

# Run with coverage and view report
pytest tests/ -v --cov=gsply --cov-report=html
# Open htmlcov/index.html in browser
```

### Test Coverage

Coverage goals:
- **Overall coverage**: Aim for >90%
- **Core functionality**: 100% coverage for reader.py and writer.py
- **Edge cases**: Test error conditions and boundary cases

Run coverage locally:
```bash
pytest tests/ --cov=gsply --cov-report=term --cov-report=html
```

### Package Verification

Before submitting a PR, verify the package:

```bash
# 1. Build the package
python -m build

# 2. Test installation in clean environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate
pip install dist/gsply-0.1.0-py3-none-any.whl

# 3. Test import
python -c "import gsply; print(f'gsply v{gsply.__version__}')"

# 4. Run tests
pytest tests/ -v

# 5. Deactivate and clean up
deactivate
rm -rf test_env
```

Verification checklist:
- [OK] 56 tests passing
- [OK] Wheel installs successfully
- [OK] Package imports correctly
- [OK] All APIs accessible
- [OK] Type hints included (py.typed)
- [OK] Metadata validated
- [OK] Universal wheel created (py3-none-any)

### Running Benchmarks

Compare gsply performance against other libraries:

```bash
# Install benchmark dependencies
pip install -e ".[benchmark]"

# Run benchmark with default settings
python benchmarks/benchmark.py

# Custom test file and iterations
python benchmarks/benchmark.py --config.file path/to/model.ply --config.iterations 20

# Skip write benchmarks
python benchmarks/benchmark.py --config.skip-write
```

Example output:
```
READ PERFORMANCE
Library         Time            Speedup
gsply           5.56ms          baseline (FASTEST)
plyfile         18.23ms         0.31x (3.3x SLOWER)
Open3D          43.10ms         0.13x (7.7x slower)

WRITE PERFORMANCE
Library         Time            Speedup         File Size
gsply           8.72ms          baseline        11.34MB
plyfile         12.18ms         0.72x          11.34MB
Open3D          35.69ms         0.24x          1.15MB (XYZ only)
```

---

## CI/CD Pipeline

### Overview

gsply has a complete GitHub Actions CI/CD pipeline for automated testing, building, and publishing.

### GitHub Actions Workflows

Located in `.github/workflows/`:

**1. test.yml - Automated Testing**
- **Triggers**: Every push and PR to main/develop
- **Platforms**: Ubuntu, Windows, macOS
- **Python versions**: 3.10, 3.11, 3.12, 3.13
- **Features**:
  - Unit tests with pytest
  - Code coverage with Codecov
  - Linting with ruff
  - Type checking with mypy

**2. build.yml - Build & Distribution**
- **Triggers**: Push, PR, and tags
- **Builds**: Universal wheel + source distribution
- **Verifies**: Installation on all platforms
- **Artifacts**: Uploaded for download

**3. publish.yml - PyPI Publishing**
- **Triggers**: GitHub Release creation
- **Publishes to**: PyPI (production) and TestPyPI
- **Security**: Trusted publishing (no API tokens)
- **Features**: Sigstore artifact signing

**4. benchmark.yml - Performance Testing**
- **Triggers**: Push to main, PRs, manual trigger
- **Benchmarks**: Read/write performance
- **Compares**: Against plyfile and Open3D
- **Reports**: Results as PR comments

**5. docs.yml - Documentation Validation**
- **Triggers**: Push and PR
- **Checks**: README, docs/BUILD.md, LICENSE
- **Validates**: Markdown formatting
- **Verifies**: API docstrings

### Automated Testing

The CI/CD pipeline automatically runs:

```yaml
# Multi-platform testing matrix
platforms: [ubuntu-latest, windows-latest, macos-latest]
python-versions: ['3.10', '3.11', '3.12', '3.13']

# Test steps
- Install dependencies
- Run pytest with coverage
- Upload coverage to Codecov
- Run linting (ruff)
- Run type checking (mypy)
```

### Workflow Triggers

| Workflow | Push | PR | Tag | Release | Manual |
|----------|------|-----|-----|---------|--------|
| test.yml | YES | YES | - | - | - |
| build.yml | YES | YES | YES | - | - |
| publish.yml | - | - | - | YES | - |
| benchmark.yml | YES | YES | - | - | YES |
| docs.yml | YES | YES | - | - | - |

### Local Testing with act

Test workflows locally before pushing:

```bash
# Install act (GitHub Actions local runner)
# https://github.com/nektos/act

# Run test workflow locally
act -j test

# Run build workflow locally
act -j build
```

---

## Release Process

### Version Management

Version is defined in two places and must be kept in sync:

1. **pyproject.toml**:
```toml
[project]
version = "0.1.0"
```

2. **src/gsply/__init__.py**:
```python
__version__ = "0.1.0"
```

### Creating a Release

**Step 1: Update Version Numbers**

```bash
# Edit pyproject.toml
version = "0.2.6"

# Edit src/gsply/__init__.py
__version__ = "0.2.6"
```

**Step 2: Update Documentation**

```bash
# Edit docs/CHANGELOG.md - Add new version section
# Edit README.md if API changes
# Update test count if tests changed
```

**Step 3: Commit and Tag**

```bash
# Commit changes
git add .
git commit -m "Release v0.2.6"

# Create annotated tag
git tag -a v0.2.6 -m "Release v0.2.6"

# Push commits and tags
git push && git push --tags
```

**Step 4: Create GitHub Release**

1. Go to GitHub > Releases > Create new release
2. Choose tag: v0.2.6
3. Write release notes (copy from docs/CHANGELOG.md)
4. Publish release

**Step 5: Automated Publishing**

The CI/CD pipeline will automatically:
- Build wheels and source distribution
- Run full test suite on all platforms
- Publish to TestPyPI (for verification)
- Publish to PyPI (production)
- Upload artifacts to GitHub Release
- Sign artifacts with Sigstore

### Publishing to PyPI

#### Manual Publishing (if needed)

```bash
# Install twine
pip install twine

# Build distributions
python -m build

# Upload to Test PyPI (recommended first)
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ gsply

# Upload to Production PyPI
python -m twine upload dist/*
```

#### Setting Up PyPI Trusted Publishing

**Step 1: PyPI Configuration**

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new publisher"
3. Fill in:
   - PyPI Project Name: `gsply`
   - Owner: `OpsiClear`
   - Repository: `gsply`
   - Workflow: `publish.yml`
   - Environment: `pypi`

**Step 2: GitHub Environment**

1. Go to GitHub > Settings > Environments
2. Create environment: `pypi`
3. Add protection rules:
   - [OK] Required reviewers (optional but recommended)
   - [OK] Wait timer (optional)

**Step 3: TestPyPI (Optional)**

Repeat Step 1 for TestPyPI:
- Go to https://test.pypi.org/manage/account/publishing/
- Same settings as above

### Setting Up Codecov

1. Go to https://codecov.io
2. Sign in with GitHub
3. Add repository: `OpsiClear/gsply`
4. Copy the token (if needed)
5. Add to GitHub Secrets:
   - Settings > Secrets > Actions
   - New secret: `CODECOV_TOKEN`

### Release Checklist

Before creating a release:

- [ ] All tests pass locally (`pytest tests/ -v`)
- [ ] Benchmarks run successfully (`python benchmarks/benchmark.py`)
- [ ] Version updated in `pyproject.toml`
- [ ] Version updated in `src/gsply/__init__.py`
- [ ] `docs/RELEASE_NOTES.md` updated with changes
- [ ] `README.md` updated if needed (API changes, new features)
- [ ] All changes committed
- [ ] Git tag created (`git tag -a vX.Y.Z -m "Release vX.Y.Z"`)
- [ ] Tag pushed (`git push --tags`)
- [ ] GitHub Release created with release notes
- [ ] CI/CD pipeline passes (check Actions tab)
- [ ] Package published to PyPI (automatic via CI/CD)
- [ ] Installation verified (`pip install gsply`)

### Post-Release Verification

```bash
# Verify PyPI installation
pip install --upgrade gsply
python -c "import gsply; print(gsply.__version__)"

# Run tests
pytest tests/ -v

# Check package metadata
pip show gsply
```

---

## Technical Implementation

### Format Compatibility

gsply is fully compatible with the PlayCanvas splat-transform specification for compressed PLY files. All implementations match the reference bit-for-bit.

#### Compressed Format Specification

**Chunk-based encoding:**
- **Chunk size**: 256 Gaussians per chunk
- **Compression ratio**: 3.8-14.5x (depending on SH degree)
- **Format**: Binary little-endian PLY

**Bit packing layout:**

| Component | Bits | Layout | Description |
|-----------|------|--------|-------------|
| Position | 32 | 11-10-11 (x, y, z) | Quantized to chunk bounds |
| Rotation | 32 | 2+10+10+10 (which, a, b, c) | Smallest-three quaternion encoding |
| Scale | 32 | 11-10-11 (sx, sy, sz) | Log-space scales, quantized |
| Color | 32 | 8-8-8-8 (r, g, b, opacity) | RGBA with opacity |
| SH coeffs | 8/coeff | uint8 per coefficient | Quantized spherical harmonics |

**Chunk metadata (18 floats per chunk):**
- Position bounds: min_x, min_y, min_z, max_x, max_y, max_z (6 floats)
- Scale bounds: min_sx, min_sy, min_sz, max_sx, max_sy, max_sz (6 floats, clamped)
- Color bounds: min_r, min_g, min_b, max_r, max_g, max_b (6 floats)

### Quaternion Encoding

**Critical implementation detail:** gsply uses the LARGEST quaternion component (not smallest) for the "smallest-three" encoding format. This matches the PlayCanvas reference implementation.

**File**: `src/gsply/writer.py:422-470`

```python
# Find LARGEST component (PlayCanvas convention)
quats_normalized = quats / np.linalg.norm(quats, axis=1, keepdims=True)
abs_quats = np.abs(quats_normalized)
largest_idx = np.argmax(abs_quats, axis=1)  # Find LARGEST

# Sign flipping (matches splat-transform compressed-chunk.ts:133-138)
for i in range(num_gaussians):
    if quats_normalized[i, largest_idx[i]] < 0:
        quats_normalized[i] = -quats_normalized[i]

# Extract three components (excluding largest)
three_components = []
for i in range(num_gaussians):
    components = []
    for j in range(4):
        if j != largest_idx[i]:
            components.append(quats_normalized[i, j])
    three_components.append(components)
three_components = np.array(three_components)

# Normalize with sqrt(2)/2 factor
norm = np.sqrt(2.0) * 0.5
qa_norm = three_components[:, 0] * norm + 0.5
qb_norm = three_components[:, 1] * norm + 0.5
qc_norm = three_components[:, 2] * norm + 0.5
```

**Reference**: PlayCanvas splat-transform `compressed-chunk.ts:133-140`

### Scale Bounds Clamping

**File**: `src/gsply/writer.py:305-311`

Scale bounds are clamped to [-20, 20] to handle infinity values:

```python
# Clamp scale bounds (matches splat-transform)
chunk_bounds[chunk_idx, 6] = np.clip(np.min(chunk_scales[:, 0]), -20, 20)
chunk_bounds[chunk_idx, 7] = np.clip(np.min(chunk_scales[:, 1]), -20, 20)
chunk_bounds[chunk_idx, 8] = np.clip(np.min(chunk_scales[:, 2]), -20, 20)
chunk_bounds[chunk_idx, 9] = np.clip(np.max(chunk_scales[:, 0]), -20, 20)
chunk_bounds[chunk_idx, 10] = np.clip(np.max(chunk_scales[:, 1]), -20, 20)
chunk_bounds[chunk_idx, 11] = np.clip(np.max(chunk_scales[:, 2]), -20, 20)
```

**Reference**: PlayCanvas splat-transform `compressed-chunk.ts:88-95`

**Impact**: Prevents crashes when scale values are at infinity.

### SH Coefficient Quantization

**File**: `src/gsply/writer.py:476-486`

Simple truncation-based quantization:

```python
# Quantize SH coefficients (matches splat-transform)
sh_normalized = (sh - sh_min[:, np.newaxis, :]) / (sh_range[:, np.newaxis, :] + 1e-8)
packed_sh = np.clip(np.trunc(sh_normalized * 256.0), 0, 255).astype(np.uint8)
```

**Reference**: PlayCanvas splat-transform `write-compressed-ply.ts:85-86`

### Vectorization for Performance

gsply achieves 38.5x speedup over naive Python loops through vectorization:

**Before (Python loop):**
```python
for i in range(num_gaussians):
    # Process each Gaussian individually
    unpack_bits(...)
```

**After (Vectorized NumPy):**
```python
# Process all Gaussians in parallel
packed = np.frombuffer(data, dtype=np.uint32).reshape(num_gaussians, 4)
positions = unpack_positions_vectorized(packed[:, 0])
rotations = unpack_rotations_vectorized(packed[:, 1])
scales = unpack_scales_vectorized(packed[:, 2])
colors = unpack_colors_vectorized(packed[:, 3])
```

**Key techniques:**
- SIMD parallelization via NumPy
- Memory-efficient bulk operations
- Elimination of Python loops
- Zero-copy slicing where possible

For detailed explanation, see [docs/VECTORIZATION_EXPLAINED.md](VECTORIZATION_EXPLAINED.md).

### Testing Requirements

**Compatibility verification:**

```bash
# Run compatibility tests
python verify_compatibility.py
```

All checks must pass:
- [OK] Quaternion encoding correct (max error: 0.001256)
- [OK] Scale bounds properly clamped to [-20, 20]
- [OK] SH coefficient quantization correct (max error: 0.015625)
- [OK] File structure matches specification

**Format interoperability:**
- Files written by gsply must be readable by PlayCanvas splat-transform
- Files written by splat-transform must be readable by gsply
- Bit-for-bit compatibility with the reference implementation

### Performance Considerations

**Read optimization:**
- Single `np.fromfile()` instead of per-property reads
- Zero-copy slicing for data extraction
- Pre-allocated arrays for decompression
- Vectorized bit unpacking (SIMD)

**Write optimization:**
- Pre-allocated output arrays
- Bulk data type conversions
- Direct binary writes without intermediate buffers
- Optimized chunk boundary calculations

**Memory efficiency:**
- Avoid unnecessary copies
- Use views where possible
- Reuse buffers for repeated operations
- Stream large files without full memory load (future)

---

## Coding Standards

### Style Guidelines

gsply follows PEP 8 with the following conventions:

**Type Hints:**
- Use Python 3.10+ type hints for all functions
- Use `from __future__ import annotations` for forward references
- Use `typing` module types (List, Dict, Optional, etc.)
- Type hint return values explicitly

```python
from __future__ import annotations
from pathlib import Path
import numpy as np

def plyread(file_path: str | Path) -> tuple[
    np.ndarray,  # means
    np.ndarray,  # scales
    np.ndarray,  # quats
    np.ndarray,  # opacities
    np.ndarray,  # sh0
    np.ndarray | None,  # shN
]:
    """Read Gaussian Splatting PLY file."""
    pass
```

**Naming Conventions:**
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private functions: `_leading_underscore`

**Docstrings:**
- Use Google-style docstrings
- Include Args, Returns, Raises sections
- Provide examples for public API

```python
def detect_format(file_path: str | Path) -> tuple[bool, int | None]:
    """Detect PLY format type and SH degree.

    Args:
        file_path: Path to PLY file

    Returns:
        Tuple of (is_compressed, sh_degree):
        - is_compressed: True if compressed format
        - sh_degree: 0-3 for uncompressed, None for compressed

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is not a valid PLY

    Example:
        >>> is_compressed, sh_degree = detect_format("model.ply")
        >>> print(f"Compressed: {is_compressed}, SH: {sh_degree}")
    """
    pass
```

**Code Organization:**
- Keep functions focused and concise (max 50 lines)
- Use descriptive variable names
- Avoid magic numbers (use named constants)
- Comment complex algorithms

### Linting and Formatting

gsply uses **ruff** for linting:

```bash
# Lint code
ruff check src/

# Format code
ruff format src/

# Check and fix
ruff check --fix src/
```

Configuration in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py310"
```

### Type Checking

gsply uses **mypy** for static type checking:

```bash
# Run type checker
mypy src/gsply/

# Strict mode
mypy --strict src/gsply/
```

### Test Writing

**Test structure:**
- One test file per module (`test_reader.py`, `test_writer.py`)
- Use descriptive test names (`test_read_sh0_format`, `test_write_compressed_format`)
- Group related tests in classes
- Use fixtures for shared setup

**Test example:**
```python
import pytest
import numpy as np
from gsply import plyread, plywrite

def test_read_sh0_format():
    """Test reading SH degree 0 format."""
    means, scales, quats, opacities, sh0, shN = plyread("test_sh0.ply")

    assert means.shape[0] > 0
    assert means.shape[1] == 3
    assert shN is None  # SH0 has no higher-order coefficients

def test_write_and_read_roundtrip():
    """Test write-read consistency."""
    # Create test data
    means = np.random.randn(100, 3)
    scales = np.random.randn(100, 3)
    quats = np.random.randn(100, 4)
    opacities = np.random.randn(100)
    sh0 = np.random.randn(100, 3)

    # Write and read
    plywrite("temp.ply", means, scales, quats, opacities, sh0)
    means2, scales2, quats2, opacities2, sh02, shN2 = plyread("temp.ply")

    # Verify
    np.testing.assert_allclose(means, means2, rtol=1e-6)
```

---

## Pull Request Process

### Before Submitting

1. **Run all tests locally:**
```bash
pytest tests/ -v
```

2. **Run benchmarks if performance-related:**
```bash
python benchmarks/benchmark.py
```

3. **Check code style:**
```bash
ruff check src/
mypy src/gsply/
```

4. **Update documentation:**
- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update docs/RELEASE_NOTES.md

5. **Commit changes with clear messages:**
```bash
git commit -m "Add feature: compressed PLY writing"
```

### Pull Request Template

Fill out the PR template completely:

```markdown
## Description
[Describe what changes you made and why]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Breaking change

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for changes
- [ ] Benchmarks run (if performance-related)

## Documentation
- [ ] Updated README.md
- [ ] Updated docstrings
- [ ] Updated RELEASE_NOTES.md

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] No new warnings generated
```

### Review Process

1. **Automated checks must pass:**
   - All tests pass on all platforms
   - Linting passes (ruff)
   - Type checking passes (mypy)
   - Coverage maintained or improved

2. **Code review:**
   - Maintainer will review code
   - Address feedback promptly
   - Make changes in new commits (don't force-push)

3. **Approval and merge:**
   - Maintainer approves PR
   - Squash commits if requested
   - PR merged to main branch

### Issue Reporting

**Bug reports:** Use the bug report template
- Describe the bug clearly
- Provide minimal reproduction steps
- Include system information (OS, Python version)
- Attach relevant files if possible

**Feature requests:** Use the feature request template
- Describe the feature and use case
- Explain why it would be useful
- Suggest implementation approach if possible

---

## Troubleshooting

### Build Issues

**"No module named 'setuptools'"**
```bash
pip install --upgrade setuptools wheel
```

**Wheel is not pure Python**
- Check `pyproject.toml` has `zip-safe = true`
- Ensure no C extensions are included
- Verify wheel tag is `py3-none-any`

**Version mismatch warnings**
- Ensure version in `pyproject.toml` matches `__init__.py`
- Check both files are committed

### Test Issues

**Tests fail on Windows**
- Check for path separator issues (`/` vs `\`)
- Use `pathlib.Path` for cross-platform paths
- Verify line endings (CRLF vs LF)

**Coverage not uploading**
- Check `CODECOV_TOKEN` secret is set in GitHub
- Verify repository is added to Codecov
- Check workflow has codecov action configured

### CI/CD Issues

**Workflow fails with "No module named 'gsply'"**
- Check `pip install -e .` is in workflow
- Verify `pyproject.toml` is correct
- Ensure all dependencies are listed

**PyPI publishing fails**
- Verify trusted publishing is configured
- Check GitHub environment name matches (`pypi`)
- Ensure version number is updated and unique
- Verify you created a GitHub Release (not just a tag)

**Tests pass locally but fail in CI**
- Check Python version compatibility
- Verify all dependencies are in pyproject.toml
- Check for platform-specific code
- Review CI logs for specific error messages

### Runtime Issues

**"Invalid PLY format" error**
- Verify file is a valid PLY file
- Check file is binary (not ASCII)
- Ensure file is little-endian
- Try `detect_format()` to diagnose

**Performance slower than expected**
- Verify NumPy is using BLAS/LAPACK
- Check file is on fast storage (not network drive)
- Profile with `python -m cProfile` to identify bottlenecks
- Ensure running in optimized mode (not debug)

**Memory issues with large files**
- Current implementation loads entire file
- Use streaming for files >1GB (future feature)
- Process in chunks if possible
- Check available system memory

### Getting Help

If you encounter issues not covered here:

1. **Search existing issues**: https://github.com/OpsiClear/gsply/issues
2. **Check documentation**: README.md and docs/ files
3. **Open a new issue**: Provide detailed information and reproduction steps
4. **Ask in discussions**: For questions and general help

---

## Platform-Specific Notes

### Windows

- Use PowerShell for scripts (`.\build.ps1`)
- Virtual environment activation: `.venv\Scripts\Activate.ps1`
- Path separators: Use `pathlib.Path` for compatibility
- Line endings: Git should auto-convert (check `.gitattributes`)

### Linux/Unix

- Use bash scripts (`./build.sh`)
- Virtual environment activation: `source .venv/bin/activate`
- May need `chmod +x` for scripts
- Install build tools: `sudo apt-get install build-essential` (if needed)

### macOS

- Same as Linux/Unix for most operations
- May need Xcode command line tools: `xcode-select --install`
- Use Homebrew for dependencies: `brew install python@3.10`
- Apple Silicon (M1/M2): Ensure native ARM64 Python

---

## Additional Resources

### Documentation

- **README.md**: User guide and API reference
- **docs/PERFORMANCE.md**: Performance benchmarks and optimization history
- **docs/COMPRESSED_FORMAT.md**: Compressed PLY format specification
- **docs/VECTORIZATION_EXPLAINED.md**: Deep-dive into vectorization techniques
- **docs/BUILD.md**: Build and distribution guide (detailed)
- **docs/CI_CD_SETUP.md**: CI/CD pipeline reference (detailed)
- **docs/RELEASE_NOTES.md**: Release notes and version history
- **docs/COMPATIBILITY_FIXES.md**: Format compatibility details

### External References

- **PlayCanvas splat-transform**: https://github.com/playcanvas/splat-transform
- **Gaussian Splatting**: Original paper and implementation
- **Python Packaging**: https://packaging.python.org/
- **pytest**: https://docs.pytest.org/
- **NumPy**: https://numpy.org/doc/

---

## License

By contributing to gsply, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to gsply!** Your efforts help make gsply faster, more reliable, and more useful for the community.
