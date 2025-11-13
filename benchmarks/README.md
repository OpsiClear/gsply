# Benchmarks

Performance benchmarking suite for gsply.

## Available Benchmarks

### Core Benchmarks

**benchmark_optimizations.py** - Multi-size, multi-degree performance testing
- Tests 10K, 100K, 400K Gaussians
- Tests SH degrees 0, 1, 3
- Comprehensive throughput measurements
- Referenced in BENCHMARK_SUMMARY.md

**benchmark_extended.py** - Extended testing with large datasets
- Tests up to 1M Gaussians
- File size analysis
- Memory usage profiling
- Extended performance characteristics

**benchmark_compressed.py** - Compressed format focused testing
- Detailed compressed vs uncompressed comparison
- Compression ratio analysis
- Format-specific optimizations

### Utilities

**verify_benchmarks.py** - Validation and sanity checks
- Verifies benchmark data integrity
- Quick performance sanity tests
- Regression detection

**generate_test_data.py** - Test data generator
- Generates synthetic Gaussian data
- Creates various sizes and SH degrees
- Output: benchmarks/test_data/ (~538MB)

## Running Benchmarks

### Quick Start

```bash
# Generate test data (first time only)
uv run python benchmarks/generate_test_data.py

# Run core benchmarks
uv run python benchmarks/benchmark_optimizations.py

# Run extended benchmarks
uv run python benchmarks/benchmark_extended.py

# Run compressed format benchmarks
uv run python benchmarks/benchmark_compressed.py

# Verify benchmark integrity
uv run python benchmarks/verify_benchmarks.py
```

## Test Data

Test data is generated locally and stored in `benchmarks/test_data/`:
- ~538MB total (9 PLY files)
- Various sizes: 10K, 100K, 400K, 1M Gaussians
- Various SH degrees: 0, 1, 3
- Both compressed and uncompressed formats

**Note**: Test data is gitignored and must be regenerated locally:
```bash
uv run python benchmarks/generate_test_data.py
```

## Benchmark Results

See the following documentation for comprehensive results:
- **BENCHMARK_SUMMARY.md** - Comprehensive results with analysis
- **QUICK_REFERENCE.md** - Quick lookup and decision matrix

## Current Performance (v0.1.1+)

### Compressed Format
- 100K SH0: ~2.6ms (38M Gaussians/sec)
- 100K SH3: ~22ms (4.5M Gaussians/sec)
- 400K SH0: ~7ms (57M Gaussians/sec)
- 400K SH3: ~23ms (17M Gaussians/sec)

### Key Optimizations
- JIT-compiled decompression (4.2x faster)
- Vectorized operations
- Zero-copy where possible
- Numba parallel processing
