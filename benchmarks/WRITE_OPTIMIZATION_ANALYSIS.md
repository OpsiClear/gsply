# Write Performance Optimization Analysis

**Date:** January 14, 2025
**Analysis:** Uncompressed PLY write performance

## Current Performance

Based on verified benchmarks:

| Gaussians | SH | Write Time | Throughput | Expected Optimal |
|-----------|----|-----------:|-----------:|------------------:|
| 100K | 0 | 3.9ms | 26M/s | 40-50M/s |
| 400K | 0 | 19.3ms | 21M/s | 35-45M/s |
| 100K | 3 | 24.6ms | 4.1M/s | 8-12M/s |
| 400K | 3 | 121.5ms | 3.3M/s | 6-10M/s |
| 1M | 3 | 316.5ms | 3.2M/s | 5-8M/s |

**Analysis:** SH3 writes are significantly slower than expected, especially at scale.

## Root Cause Analysis

### Issue 1: NO ZERO-COPY WRITE PATH (CRITICAL)

**Location:** `src/gsply/writer.py:814-831`

**Current Implementation:**
```python
# ALWAYS creates a new array and copies ALL data
data = np.empty((num_gaussians, total_props), dtype="<f4")
data[:, 0:3] = means        # COPY 1: means (N, 3)
data[:, 3:6] = sh0          # COPY 2: sh0 (N, 3)
data[:, 6:6+sh_coeffs] = shN  # COPY 3: shN (N, K*3) - LARGE for SH3!
data[:, 6+sh_coeffs] = opacities  # COPY 4: opacities (N,)
data[:, 7+sh_coeffs:10+sh_coeffs] = scales  # COPY 5: scales (N, 3)
data[:, 10+sh_coeffs:14+sh_coeffs] = quats  # COPY 6: quats (N, 4)
```

**Problem:**
- For 400K Gaussians SH3 (59 properties):
  - Array size: 400K × 59 × 4 bytes = 94.4 MB
  - 6 separate memory copy operations totaling 94.4 MB copied
  - For SH3, the shN copy alone is ~73 MB (45 properties out of 59)

**README Claims Zero-Copy But It's Not Implemented:**
The README says:
> "Zero-copy writes: When data has _base array (from plyread), use directly without copying"

This is **FALSE**. The code has NO implementation of this optimization!

### Issue 2: Multiple Memory Copies

**Problem:** Each slice assignment (`data[:, X:Y] = array`) triggers a separate memcpy:
- SH0: 5 copies (means, sh0, opacities, scales, quats)
- SH3: 6 copies (+ shN which is the largest)

**Impact:**
- For 400K SH3: ~95 MB copied in 6 separate operations
- Memory bandwidth limited: ~30-50 GB/s typical DRAM
- Theoretical minimum copy time: 95 MB / 40 GB/s = 2.4ms
- Actual copy time: ~5-10ms due to overhead

### Issue 3: No GSData Fast Path

**Current:** Only accepts individual np.ndarray arguments
**Missing:** Direct GSData support that could detect `_base` array

When data comes from `plyread()`:
- All fields (means, scales, etc.) are **views** into a single contiguous `_base` array
- The `_base` array is already in PLY format order
- We could write `_base` directly to disk (after header)
- **Zero data movement, only I/O time!**

## Proposed Optimizations

### Optimization 1: Zero-Copy Write Path (HIGH PRIORITY)

**Implementation:**
```python
def write_uncompressed(...):
    # NEW: Check if inputs are views of a common base array
    if _can_use_zero_copy(means, scales, quats, opacities, sh0, shN):
        # Fast path: write base array directly
        base = means.base  # All views share same base
        header_bytes = _build_header_fast(num_gaussians, num_sh_rest)

        with open(file_path, "wb", buffering=buffer_size) as f:
            f.write(header_bytes)
            base.tofile(f)  # Direct write, NO COPYING
        return

    # Fallback: existing slow path
    # ... current implementation ...
```

**Detection Logic:**
```python
def _can_use_zero_copy(means, scales, quats, opacities, sh0, shN):
    """Check if all arrays are views of a contiguous base array in PLY order."""
    # Check if means has a base (from plyread)
    if means.base is None:
        return False

    base = means.base

    # Verify all arrays share the same base
    if scales.base is not base:
        return False
    if quats.base is not base:
        return False
    if opacities.base is not base:
        return False
    if sh0.base is not base:
        return False
    if shN is not None and shN.base is not base:
        return False

    # Verify base is contiguous
    if not base.flags.c_contiguous:
        return False

    # Verify correct shape (N, P) where P is total properties
    expected_props = 14 if shN is None else (14 + shN.shape[1])
    if base.shape != (means.shape[0], expected_props):
        return False

    return True
```

**Expected Speedup:**
- 400K SH3: From 121.5ms → ~15-20ms (6-8x faster)
  - Eliminates ~95MB of memory copying
  - Only header creation + I/O time remains

### Optimization 2: Vectorized Array Assembly (MEDIUM PRIORITY)

For cases where zero-copy doesn't apply (data created from scratch):

**Current:** 6 separate slice assignments
**Proposed:** Use `np.concatenate` or structured arrays

**Option A: Pre-allocate and use memoryviews**
```python
# Create views before copying to reduce overhead
data = np.empty((num_gaussians, total_props), dtype="<f4")
data_view = memoryview(data)

# Single-pass copy using flat indexing (faster than slice assignments)
np.copyto(data[:, 0:3], means, casting='no')
# ... etc
```

**Option B: Use structured dtype (if beneficial)**
```python
# Define structured dtype matching PLY format
dtype_sh3 = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('f_dc', '<f4', (3,)),
    ('f_rest', '<f4', (45,)),
    ('opacity', '<f4'),
    ('scales', '<f4', (3,)),
    ('quats', '<f4', (4,)),
])
```

**Expected Speedup:**
- 400K SH3: From 121.5ms → ~80-100ms (1.2-1.5x faster)
  - Reduces copy overhead through better memory access patterns

### Optimization 3: GSData-Aware API

**Add overload to accept GSData directly:**
```python
def write_uncompressed(
    file_path: str | Path,
    means: np.ndarray | GSData,
    scales: np.ndarray | None = None,
    ...
):
    # Detect GSData input
    if isinstance(means, GSData):
        data = means
        # Extract arrays from GSData
        means, scales, quats, opacities, sh0, shN = data.unpack()

    # ... rest of function
```

**Benefits:**
- Cleaner API: `plywrite("out.ply", data)` instead of `plywrite("out.ply", *data.unpack())`
- Enables automatic zero-copy detection

## Performance Impact Estimate

### With Zero-Copy Optimization

| Case | Current | With Zero-Copy | Speedup | Notes |
|------|---------|----------------|---------|-------|
| 100K SH0 (from plyread) | 3.9ms | ~1.5ms | 2.6x | Header + 5MB I/O |
| 400K SH0 (from plyread) | 19.3ms | ~4-5ms | 4-5x | Header + 21MB I/O |
| 100K SH3 (from plyread) | 24.6ms | ~3-4ms | 6-8x | Header + 22MB I/O |
| 400K SH3 (from plyread) | 121.5ms | ~15-20ms | 6-8x | Header + 90MB I/O |
| 1M SH3 (from plyread) | 316.5ms | ~40-50ms | 6-8x | Header + 225MB I/O |

### Without Zero-Copy (Data Created from Scratch)

With vectorized assembly optimization:

| Case | Current | Optimized | Speedup |
|------|---------|-----------|---------|
| 400K SH3 | 121.5ms | ~80-100ms | 1.2-1.5x |

## Recommendation

**Priority 1: Implement Zero-Copy Write Path**
- HIGH IMPACT: 6-8x speedup for the common case (writing data loaded from plyread)
- MEDIUM EFFORT: ~50-100 lines of code
- NO RISK: Falls back to current implementation if zero-copy not possible

**Priority 2: Fix Documentation**
- Remove false zero-copy claim from README
- Document actual optimizations

**Priority 3: Consider Vectorized Assembly**
- MEDIUM IMPACT: 1.2-1.5x speedup for non-zero-copy cases
- MEDIUM EFFORT: Refactor existing code
- MEDIUM RISK: Requires careful testing

## Implementation Notes

### Zero-Copy Safety Checks

Must verify:
1. All arrays share same base (are views)
2. Base is contiguous in memory
3. Base has correct shape (N, P)
4. Arrays are in correct order (PLY property order)
5. No data modifications needed (already float32 little-endian)

### Compatibility

- Zero-copy ONLY works when writing data loaded from `plyread()`
- Data created from scratch still uses current path
- No API changes required (transparent optimization)

### Testing

Must test:
1. Zero-copy path with plyread → plywrite round-trip
2. Non-zero-copy path with manually created arrays
3. Mixed cases (some views, some copies)
4. Byte-for-byte output equivalence
5. All SH degrees (0-3)
6. Edge cases (empty arrays, single Gaussian, etc.)

## Conclusion

The missing zero-copy write optimization is a **critical performance bottleneck**. Implementing it would:

1. **6-8x speedup** for the common workflow (read → modify → write)
2. **Match read performance** (~70M/s for SH0, ~12M/s for SH3)
3. **Fulfill README claims** that are currently false
4. **Minimal code complexity** with clear fallback path

This is a **high-priority fix** that should be implemented immediately.
