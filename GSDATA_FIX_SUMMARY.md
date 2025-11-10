# GSData Return Type Fix + Tyro Removal

**Date**: 2025-11-10
**Issue**: Benchmark script failed due to GSData namedtuple unpacking + tyro dependency

## Problem

### 1. GSData Unpacking Error
```
ValueError: too many values to unpack (expected 6)
```

**Root Cause**: `gsply.plyread()` now returns a `GSData` namedtuple (7 fields) instead of a plain tuple (6 values).

GSData structure:
```python
GSData(means, scales, quats, opacities, sh0, shN, base)
```

The benchmark script was trying to unpack it as:
```python
means, scales, quats, opacities, sh0, shN = result  # ERROR: 7 values, not 6
```

### 2. Tyro Dependency
The benchmark script used `tyro` for CLI argument parsing, but this adds an unnecessary external dependency.

## Solutions Implemented

### 1. Fixed GSData Unpacking (3 locations)

**Location 1**: `benchmark.py:101` - `benchmark_gsply_read()`
```python
# Before:
means, scales, quats, opacities, sh0, shN = result
num_gaussians = means.shape[0]

# After:
# GSData namedtuple - access via attributes
num_gaussians = result.means.shape[0]
```

**Location 2**: `benchmark.py:192` - `benchmark_gsply_write()`
```python
# Before:
means, scales, quats, opacities, sh0, shN = data

# After:
# GSData namedtuple - unpack first 6 elements
means, scales, quats, opacities, sh0, shN = data[:6]
```

**Location 3**: `benchmark.py:235` - `benchmark_plyfile_write()`
```python
# Before:
means, scales, quats, opacities, sh0, shN = data

# After:
# GSData namedtuple - unpack first 6 elements
means, scales, quats, opacities, sh0, shN = data[:6]
```

### 2. Replaced Tyro with Argparse

**Removed**: `import tyro`
**Added**: `import argparse`

**Changed main entry point** from:
```python
if __name__ == "__main__":
    exit(tyro.cli(main))
```

To:
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gsply performance")
    parser.add_argument("--file", type=str, default="../export_with_edits/frame_00000.ply",
                        help="Path to test PLY file")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--skip-write", action="store_true",
                        help="Skip write benchmarks")

    args = parser.parse_args()
    config = BenchmarkConfig(
        file=args.file,
        warmup=args.warmup,
        iterations=args.iterations,
        skip_write=args.skip_write
    )
    exit(main(config))
```

### 3. Updated Dependencies

**pyproject.toml** - Removed tyro from:
- `benchmark` extra
- `all` extra

```toml
# Before:
benchmark = [
    "open3d>=0.17.0",
    "plyfile>=0.9.0",
    "tyro>=0.8.0",  # REMOVED
]

# After:
benchmark = [
    "open3d>=0.17.0",
    "plyfile>=0.9.0",
]
```

### 4. Updated CI/CD Workflow

**benchmark.yml** - Changed command syntax from tyro to argparse:

```yaml
# Before:
python benchmarks/benchmark.py --config.file test_data.ply --config.iterations 10

# After:
python benchmarks/benchmark.py --file test_data.ply --iterations 10
```

### 5. Updated Documentation

**docs/CHANGELOG.md** - Removed tyro from benchmark dependencies list.

## Files Modified

1. `benchmarks/benchmark.py` - Fixed unpacking, replaced tyro with argparse
2. `pyproject.toml` - Removed tyro dependency
3. `.github/workflows/benchmark.yml` - Updated command syntax
4. `docs/CHANGELOG.md` - Updated dependencies list

## Verification

### Command-Line Interface
```bash
# New usage (argparse):
python benchmarks/benchmark.py --help
python benchmarks/benchmark.py --file test.ply --iterations 20
python benchmarks/benchmark.py --skip-write

# Old usage (tyro) NO LONGER SUPPORTED:
# python benchmarks/benchmark.py --config.file test.ply --config.iterations 20
```

### GSData Usage
Users can now access GSData in multiple ways:

```python
import gsply

# Method 1: Access via attributes (recommended)
data = gsply.plyread("model.ply")
positions = data.means
colors = data.sh0

# Method 2: Unpack first 6 elements
means, scales, quats, opacities, sh0, shN = data[:6]

# Method 3: Full unpacking (7 elements)
means, scales, quats, opacities, sh0, shN, base = data
```

## Benefits

### 1. Removed External Dependency
- **Before**: Required `tyro` (additional external dependency)
- **After**: Uses standard library `argparse` (no extra dependencies)
- **Impact**: Simpler installation, fewer dependency conflicts

### 2. Standard Python CLI
- Uses familiar argparse syntax
- Better compatibility with shell scripts
- Easier to understand and maintain

### 3. Consistent with GSData Design
- Properly handles namedtuple return type
- Supports both attribute access and unpacking
- Clear comments explain the pattern

## Testing

### Local Testing
```bash
cd gsply

# Test help
python benchmarks/benchmark.py --help

# Test with custom file
python benchmarks/benchmark.py --file test_data.ply --iterations 5

# Test skip-write
python benchmarks/benchmark.py --skip-write
```

### CI/CD Testing
The benchmark workflow will automatically:
1. Create synthetic test data (50K Gaussians)
2. Run standard benchmark with 10 iterations
3. Run compressed benchmark
4. Upload results as artifacts

## Migration Guide

If you were using the old tyro-based CLI:

**Old command**:
```bash
python benchmarks/benchmark.py --config.file path/to/file.ply --config.iterations 20
```

**New command**:
```bash
python benchmarks/benchmark.py --file path/to/file.ply --iterations 20
```

**Changes**:
- `--config.file` → `--file`
- `--config.iterations` → `--iterations`
- `--config.warmup` → `--warmup`
- `--config.skip_write` → `--skip-write`

## Summary

✓ Fixed GSData unpacking in 3 locations
✓ Replaced tyro with standard argparse
✓ Updated pyproject.toml (removed tyro)
✓ Updated CI/CD workflow (new command syntax)
✓ Updated documentation
✓ Verified CLI works correctly

**Status**: READY TO COMMIT

All benchmark scripts now work correctly with GSData namedtuple and use standard Python argparse instead of tyro.
