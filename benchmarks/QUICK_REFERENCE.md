# GSPLY Performance Quick Reference

## At a Glance

### Peak Performance
- **Fastest Read:** 78.0 M Gaussians/sec (1M, SH0, uncompressed)
- **Fastest Write:** 29.4 M Gaussians/sec (100K, SH0, compressed)
- **Best Compression:** 74% reduction (SH3, 3.8x smaller)

### Typical Performance (400K Gaussians)

| Format | SH0 Write | SH0 Read | SH3 Write | SH3 Read |
|--------|-----------|----------|-----------|----------|
| **Uncompressed** | 19ms (21M/s) | 6ms (70M/s) | 98ms (4.1M/s) | 25ms (16M/s) |
| **Compressed** | 15ms (27M/s) | 8.5ms (47M/s) | 92ms (4.4M/s) | 118ms (3.4M/s) |

---

## Decision Matrix

### Choose Uncompressed When:
- [X] Need maximum read speed (up to 78 M/s)
- [X] Training/development with frequent iterations
- [X] Disk space not a concern
- [X] Using SH3 with read-heavy workload

### Choose Compressed When:
- [X] Storage space limited (71-74% savings)
- [X] Network transfer required
- [X] Using SH0 with large files (>100K)
- [X] Archival or distribution

---

## Performance by Use Case

### Training Pipeline (Frequent Read/Write)
**Recommendation:** Uncompressed SH0 or SH3
```
100K Gaussians, SH3, Uncompressed:
  Write: 18ms   Read: 6ms
  Total round-trip: 24ms
  File size: 22.5 MB
```

### Deployment/Distribution (Space Priority)
**Recommendation:** Compressed SH0 or SH3
```
400K Gaussians, SH3, Compressed:
  Write: 92ms   Read: 118ms
  File size: 23.4 MB (vs 90MB uncompressed)
  Savings: 74% reduction
```

### Large Scene (1M Gaussians)
**Recommendation:** Mixed approach
```
SH0 Compressed:
  Write: 36ms (28M/s)  Read: 17ms (60M/s)
  File: 15.5MB (vs 53MB uncompressed, 71% savings)

SH3 Uncompressed:
  Write: 256ms (3.9M/s)  Read: 71ms (14M/s)
  File: 225MB (vs 58MB compressed)
```

---

## Throughput Reference Table

### Uncompressed Format (M Gaussians/sec)

| Size | SH0 Write | SH0 Read | SH3 Write | SH3 Read |
|------|-----------|----------|-----------|----------|
| 10K | 16.5 | 40.7 | 5.6 | 16.3 |
| 100K | 18.3 | 70.7 | 5.4 | 16.4 |
| 400K | 20.7 | 69.6 | 4.1 | 15.8 |
| 1M | 16.1 | **78.0** | 3.9 | 14.0 |

### Compressed Format (M Gaussians/sec)

| Size | SH0 Write | SH0 Read | SH3 Write | SH3 Read |
|------|-----------|----------|-----------|----------|
| 10K | 10.4 | 14.9 | 3.8 | 3.0 |
| 100K | **29.4** | 35.4 | 4.5 | 3.3 |
| 400K | 26.6 | 47.0 | 4.4 | 3.4 |
| 1M | 28.2 | 60.0 | 4.8 | 3.9 |

---

## File Size Reference

### Bytes per Gaussian

| Format | SH0 | SH3 | Ratio |
|--------|-----|-----|-------|
| Uncompressed | ~53 bytes | ~225 bytes | 1.0x |
| Compressed | ~15 bytes | ~58 bytes | 3.5-3.9x smaller |

### Example File Sizes

| Gaussians | UC SH0 | C SH0 | UC SH3 | C SH3 |
|-----------|--------|-------|--------|-------|
| 100K | 5.3 MB | 1.6 MB | 22.5 MB | 5.8 MB |
| 400K | 21.4 MB | 6.2 MB | 90.0 MB | 23.4 MB |
| 1M | 53.4 MB | 15.5 MB | 225.1 MB | 58.4 MB |

---

## Key Insights

### 1. Format Trade-offs

**Uncompressed Advantages:**
- 1.3-4.7x faster reads (especially SH3)
- Lower latency (< 100ms for most operations)
- Predictable performance

**Compressed Advantages:**
- 71-74% smaller files
- Often faster writes at scale
- Better for I/O-bound scenarios

### 2. SH Degree Impact

- **SH3 vs SH0:** 4.2x more properties
- **Performance:** ~4-5x slower (linear scaling)
- **Compression:** Better ratio for SH3 (74% vs 71%)

### 3. Scaling Behavior

**Excellent scaling characteristics:**
- Uncompressed reads improve with size (better cache)
- Compressed operations benefit from JIT parallelization
- Both formats practical up to 1M+ Gaussians

---

## Quick Start Examples

### Maximum Speed
```python
# Fastest read: Uncompressed SH0
data = gsply.plyread("model.ply")  # 70-78 M/s

# Fastest write: Compressed SH0 (100K+)
gsply.plywrite("out.ply", *data, compressed=True)  # 26-29 M/s
```

### Space Efficient
```python
# Best compression with good performance
gsply.plywrite("model.ply", *data, compressed=True)
# -> Saves as "model.compressed.ply"
# -> 71-74% smaller, 4-60 M/s throughput
```

### Balanced
```python
# 400K Gaussians, SH0: Both formats perform well
# Uncompressed: 21 M/s write, 70 M/s read, 21 MB
# Compressed:   27 M/s write, 47 M/s read,  6 MB
```

---

## When Performance Matters Most

### Critical Path Operations

**Read-Heavy Workloads:**
- Use uncompressed format
- Expected: 40-78 M/s (SH0)
- Latency: 1-13ms for up to 1M Gaussians

**Write-Heavy Workloads:**
- Use compressed format (SH0)
- Expected: 17-29 M/s
- File size benefit: 3.4x smaller

**Balanced Workloads:**
- SH0: Either format performs well
- SH3: Uncompressed for reads, consider compression for storage

---

## Validation Status

- [OK] All benchmark data verified (9/9 test files)
- [OK] Performance claims validated vs. README
- [OK] Scalability tested up to 1M Gaussians
- [OK] Both formats tested across all configurations
- [OK] 92 tests passing - Zero failures
- [OK] Code quality improvements with zero performance impact
- [OK] Ready for production use

## Version Info

**gsply v0.1.1**
- Performance: 78M Gaussians/sec read, 29M/sec write
- Tests: 92 passing in 2.91s
- Zero performance regressions from code quality improvements

---

**Last Updated:** 2025
**See Also:** BENCHMARK_SUMMARY.md for full details
