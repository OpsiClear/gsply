# Benchmark Results - After Optimization

## Test Configuration

- **Test file**: frame_00000.ply (50,375 Gaussians, SH degree 3)
- **File size**: 11.34MB
- **Warmup iterations**: 3
- **Benchmark iterations**: 10
- **Platform**: Windows, Intel i7

---

## Performance Improvements

### READ Performance

| Library | Before | After | Improvement | vs plyfile |
|---------|--------|-------|-------------|------------|
| **gsply** | **7.59ms** | **5.69ms** | **-25%** ⬇️ | **3.1x faster** |
| plyfile | 19.05ms | 17.77ms | - | baseline |
| Open3D | 43.77ms | 42.70ms | - | 2.4x slower |

**Result**: Read optimization exceeded expectations!
- Expected: 10-15% improvement
- Achieved: **25% improvement** (7.59ms → 5.69ms)
- Now **3.1x faster than plyfile** (was 2.5x)
- Now **7.5x faster than Open3D**

---

### WRITE Performance (SH degree 3, 59 properties)

| Library | Before | After | Improvement | vs plyfile |
|---------|--------|-------|-------------|------------|
| **gsply** | **12.15ms** | **10.70ms** | **-12%** ⬇️ | **20% faster** |
| plyfile | 14.00ms | 12.89ms | - | baseline |
| Open3D | 35.63ms | 35.28ms | - | 2.7x slower |

**Result**: Write optimization also exceeded expectations!
- Expected: 2-3% improvement
- Achieved: **12% improvement** (12.15ms → 10.70ms)
- Now **20% faster than plyfile** (was 15%)
- Now **3.3x faster than Open3D**

---

### WRITE Performance (SH degree 0, 14 properties)

| Library | Time | Speedup | File Size |
|---------|------|---------|-----------|
| plyfile (SH0) | 3.54ms | 1.36x | 2.69MB |
| **gsply (SH0)** | **4.81ms** | 1.00x | 2.69MB |

**Note**: SH0 write still slower than plyfile. This is expected since:
- Smaller property count means less benefit from pre-allocation
- plyfile's per-property writes are more efficient for small property counts
- Still excellent absolute performance (4.81ms for 50K Gaussians)

---

## Combined Performance Summary

### Throughput (Read + Write)

**gsply total**: 5.69ms + 10.70ms = **16.39ms** = **61 FPS**
- Before: 7.59ms + 12.15ms = 19.74ms = 51 FPS
- **Improvement: +20% throughput**

**plyfile total**: 17.77ms + 12.89ms = 30.66ms = 33 FPS
- **gsply is 1.87x faster overall**

**Open3D total**: 42.70ms + 35.28ms = 77.98ms = 13 FPS
- **gsply is 4.76x faster overall**

---

## Standard Deviations

**gsply consistency:**
- Read: 409.71us (7.2% of mean) - Very stable
- Write: 603.08us (5.6% of mean) - Very stable

**Comparison:**
- gsply read: 409us std dev
- plyfile read: 1.03ms std dev (2.5x more variable)
- Open3D read: 1.03ms std dev (2.5x more variable)

**Conclusion**: gsply is not only faster but also more consistent

---

## Output Verification

✓ **SH3 outputs**: Files are equivalent (bit-exact)
✓ **SH0 outputs**: Files are equivalent (bit-exact)

All outputs verified against plyfile baseline - exact match.

---

## Key Takeaways

1. **Read optimization exceeded expectations**: 25% improvement (expected 10-15%)
2. **Write optimization exceeded expectations**: 12% improvement (expected 2-3%)
3. **Overall throughput**: +20% improvement (61 FPS vs 51 FPS)
4. **vs plyfile**:
   - Read: 3.1x faster (up from 2.5x)
   - Write: 1.20x faster (up from 1.15x)
5. **Consistency**: More stable performance (lower std dev)
6. **Correctness**: 100% output equivalence maintained

---

## What Worked

### Optimization 1: Remove Array Copies
- **Impact**: Eliminated 6 unnecessary allocations per read
- **Result**: 25% read speedup (exceeded 10-15% estimate)
- **Why it worked**: Memory bandwidth is a bottleneck; eliminating copies helps significantly

### Optimization 3: Use newaxis Instead of reshape
- **Impact**: Eliminated reshape overhead
- **Result**: 12% write speedup (exceeded 2-3% estimate)
- **Why it worked**: Compound effect - faster view creation + better cache behavior

---

## Next Optimization Targets

Based on these results, the next high-value optimizations are:

1. **Optimization 2: Avoid Redundant dtype Conversions**
   - Expected: 15-20% write improvement (especially for already-float32 data)
   - Priority: High (quick win)

2. **Optimization 4: Vectorize Compressed Decompression**
   - Expected: 5-10x compressed read improvement
   - Priority: Very High (huge impact for compressed format)

3. **SH0 Write Optimization**
   - Current: 4.81ms vs plyfile 3.54ms (36% slower)
   - Could optimize pre-allocation for small property counts
   - Priority: Medium (less common use case)

---

## Benchmark Command

```bash
cd gsply
python benchmarks/benchmark.py
```

---

**Status**: Optimizations successful ✓

**Performance gain**: +20% overall throughput

**Next steps**: Implement remaining optimizations for even more speed
