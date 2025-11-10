# Building gsply

This document describes how to build and distribute gsply packages.

## Overview

gsply is a fully buildable Python package with proper distribution tools. It builds as a **universal wheel** (`py3-none-any`) that works on all platforms without compilation:
- All Python 3.10+ versions
- All operating systems (Linux, macOS, Windows)
- All architectures (x86_64, ARM64)

## Quick Start

### Prerequisites

```bash
# Install build dependencies
pip install build twine

# Or with uv
uv pip install build twine
```

### Build Wheels

```bash
# Clean and build
cd gsply
rm -rf build dist *.egg-info
python -m build

# Or with uv
uv run python -m build
```

Or use the convenience scripts:
```bash
./build.sh          # Unix/Linux/Mac
.\build.ps1         # Windows PowerShell
```

This creates:
- `dist/gsply-0.1.0.tar.gz` - Source distribution (~26KB)
- `dist/gsply-0.1.0-py3-none-any.whl` - Universal wheel (~15KB)

### Test Installation

```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# Install from wheel
pip install dist/gsply-0.1.0-py3-none-any.whl

# Test import
python -c "import gsply; print(f'gsply v{gsply.__version__}')"

# Run tests
pytest tests/ -v
```

## Detailed Build Process

### Step 1: Clean previous builds

```bash
# Remove old build artifacts
rm -rf build/ dist/ *.egg-info
```

### Step 2: Build source distribution and wheel

```bash
# Build both sdist and wheel
python -m build

# Or with uv
uv run python -m build
```

### Step 3: Verify the build

```bash
# Check package contents
tar -tzf dist/gsply-0.1.0.tar.gz

# List wheel contents
unzip -l dist/gsply-0.1.0-py3-none-any.whl

# Check package metadata
python -m build --check
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=gsply --cov-report=html
```

### Installation Tests

```bash
# Test wheel installation in clean environment
python -m venv test_env
source test_env/bin/activate
pip install dist/gsply-0.1.0-py3-none-any.whl
python -c "import gsply; print(gsply.__version__)"
pytest tests/
```

### Package Verification

All checks should pass:
- 53 tests passing
- Wheel installs successfully
- Package imports correctly
- All APIs accessible
- Type hints included (py.typed)
- Metadata validated
- Universal wheel created

## Distribution

### Package Contents

The wheel includes:
- `gsply/__init__.py` - Public API
- `gsply/reader.py` - PLY reading (optimized)
- `gsply/writer.py` - PLY writing
- `gsply/formats.py` - Format detection
- `gsply/py.typed` - Type hints marker
- `LICENSE` - MIT license
- Full metadata

The source distribution additionally includes:
- Full test suite
- Documentation files
- Build configuration
- `README.md` - Package documentation
- `BUILD.md` - Build instructions
- `RELEASE_NOTES.md` - Version history

### Publishing to PyPI

#### Test PyPI (recommended first)

```bash
# Install twine if not already installed
pip install twine

# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ gsply
```

#### Production PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Install from PyPI
pip install gsply
```

## Version Management

Version is defined in `pyproject.toml` and should match `src/gsply/__init__.py`:

```python
__version__ = "0.1.0"
```

### Release Process

To release a new version:

1. Update version in `pyproject.toml`
2. Update version in `src/gsply/__init__.py`
3. Update `RELEASE_NOTES.md` with changes
4. Update `README.md` if needed
5. Commit changes: `git commit -am "Release v0.1.0"`
6. Tag release: `git tag -a v0.1.0 -m "Release v0.1.0"`
7. Push: `git push && git push --tags`
8. Build: `python -m build`
9. Test locally: `pip install dist/*.whl && pytest tests/`
10. Publish: `python -m twine upload dist/*`

## Build System Configuration

### Files Created

1. **LICENSE** - MIT license
2. **MANIFEST.in** - Controls what files are included in distributions
3. **.gitignore** - Excludes build artifacts from version control
4. **src/gsply/py.typed** - PEP 561 marker for type checking support
5. **build.sh** - Unix/Linux/Mac build script
6. **build.ps1** - Windows PowerShell build script
7. **RELEASE_NOTES.md** - Version history and features

### Files Modified

1. **pyproject.toml** - Enhanced with:
   - Modern SPDX license format
   - Comprehensive metadata
   - Build dependencies
   - Test configuration
   - Coverage settings
   - Proper classifiers

## Platform Support

gsply builds as a universal wheel (`py3-none-any`), meaning:
- No platform-specific compilation needed
- Works on Linux (x86_64, ARM64)
- Works on macOS (Intel, Apple Silicon)
- Works on Windows (x86_64, ARM64)

Pure Python implementation with zero C++ dependencies!

## Automated Build Scripts

### Unix/Linux/Mac (build.sh)

```bash
# Make executable
chmod +x build.sh

# Run build
./build.sh
```

### Windows PowerShell (build.ps1)

```bash
# Run PowerShell script
.\build.ps1
```

These scripts automate the clean, build, and verification steps.

## Continuous Integration

For automated builds, use GitHub Actions or similar CI/CD platforms:

```yaml
# .github/workflows/build.yml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install build twine pytest
      - name: Build wheel
        run: python -m build
      - name: Install and test
        run: |
          pip install dist/*.whl
          pytest tests/
```

## Package Features

gsply v0.1.0 provides:
- Ultra-fast PLY I/O (6.66ms read, 9.19ms write for 50K Gaussians)
- 2.7x faster than plyfile for reading
- Support for SH degrees 0-3
- Auto-format detection
- Compressed format reading (PlayCanvas compatible)
- Pure Python implementation
- Comprehensive test suite (53 tests)
- Full type hints support (py.typed)
- Cross-platform compatibility
- Zero dependencies (except numpy)

## Troubleshooting

### Build fails with "No module named 'setuptools'"

```bash
pip install --upgrade setuptools wheel
```

### Wheel is not pure Python

Check `pyproject.toml` has:
```toml
[tool.setuptools]
zip-safe = true
```

And no C extensions are included.

### Version mismatch warnings

Ensure version in `pyproject.toml` matches `__init__.py`.

### Package size is large

Check MANIFEST.in excludes unnecessary files:
- Build artifacts (*.pyc, __pycache__)
- Test data
- Documentation source files
- .git directory

### twine upload fails

Ensure you have:
- A PyPI account (https://pypi.org/account/register/)
- Created a token (https://pypi.org/manage/account/tokens/)
- Created `~/.pypirc` with proper credentials:
```ini
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi_...

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi_...
```

## Next Steps

You can now:
1. Publish to PyPI: `twine upload dist/*`
2. Share wheels with others
3. Install via `pip install gsply`
4. Use in other projects without C++ dependencies
5. Distribute as a standalone library

## Support

For issues or questions:
- GitHub: https://github.com/OpsiClear/gsply
- Documentation: See README.md and docs/RELEASE_NOTES.md
