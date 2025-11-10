# gsply Write Performance Investigation

## Problem Statement

Initial benchmarking revealed that gsply is **slower** than plyfile for SH degree 0 (14 properties) writes:
- plyfile: 2.41ms
- gsply: 4.07ms (69% slower)

However, gsply is **faster** for SH degree 3 (59 properties):
- gsply: 11.89ms
- plyfile: 13.73ms (15% faster)

## Investigation Process

### Step 1: Profiling Current Implementation

Created `profile_write.py` to measure time spent in each operation:

**gsply SH0 write breakdown:**
- concatenate: 1.291ms (38%)
- astype: 0.504ms (15%)
- header + I/O: 1.595ms (47%)
- **Data prep total: 1.795ms**

**plyfile SH0 write breakdown:**
- assign_data: 0.887ms (45%)
- write: 0.918ms (47%)
- overhead: 0.164ms (8%)
- **Data prep total: 0.887ms**

**Key Finding:** gsply's data preparation (concatenate + astype) is **2x slower** than plyfile's structured array assignment approach.

### Step 2: Testing Alternative Approaches

Created `test_optimizations.py` to compare 6 different concatenation methods:

| Method | Time (SH0) | Speedup |
|--------|-----------|---------|
| **preallocate + assign** | **1.301ms** | **baseline** |
| pre-astype + concatenate | 1.343ms | 0.97x |
| concatenate + conditional astype | 1.405ms | 0.93x |
| column_stack + astype | 1.682ms | 0.77x |
| hstack + astype | 1.827ms | 0.71x |
| **concatenate + astype (current)** | **1.896ms** | **0.69x** |

**Winner:** Preallocate + assign is **31.4% faster** than current approach!

### Step 3: Testing on Large Files (SH3)

Created `test_sh3_optimization.py` to verify the optimization also helps for SH degree 3:

| Method | Time (SH3) |
|--------|-----------|
| **preallocate + assign** | **3.998ms** |
| concatenate + astype (current) | 6.203ms |

**Improvement:** **35.5% faster** for SH3!

## Root Cause Analysis

### Why is Concatenate Slow?

`np.concatenate()` performs these operations:
1. Calculates output shape
2. Allocates new contiguous array
3. Copies data from each input array sequentially
4. Returns the new array

Then `astype('<f4')` performs:
5. Checks if conversion is needed
6. Allocates another new array
7. Copies/converts data again

**Total: 2 allocations + 2 copy operations**

### Why is Preallocate + Assign Fast?

```python
data = np.empty((num_gaussians, 14), dtype='<f4')  # 1 allocation, correct dtype
data[:, 0:3] = means      # Direct memory copy
data[:, 3:6] = sh0        # Direct memory copy
data[:, 6] = opacities    # Direct memory copy
data[:, 7:10] = scales    # Direct memory copy
data[:, 10:14] = quats    # Direct memory copy
```

**Total: 1 allocation + 5 direct copy operations (no temporary arrays)**

### Why Does the Difference Matter More for Small Files?

For SH0 (14 properties):
- concatenate overhead: ~0.6ms
- Data size: 50K × 14 × 4 bytes = 2.8MB

For SH3 (59 properties):
- concatenate overhead: ~2.2ms
- Data size: 50K × 59 × 4 bytes = 11.8MB

The overhead percentage is similar (31-35%), but the absolute time savings is larger for bigger files.

## Recommended Optimization

Replace current implementation in `src/gsply/writer.py`:

```python
# BEFORE (current)
if shN is not None:
    data = np.concatenate([means, sh0, shN, opacities, scales, quats], axis=1)
else:
    data = np.concatenate([means, sh0, opacities, scales, quats], axis=1)
data = data.astype('<f4')
```

```python
# AFTER (optimized)
num_gaussians = means.shape[0]

if shN is not None:
    sh_coeffs = shN.size // (num_gaussians * 3)
    total_props = 3 + 3 + sh_coeffs*3 + 1 + 3 + 4
    data = np.empty((num_gaussians, total_props), dtype='<f4')
    data[:, 0:3] = means
    data[:, 3:6] = sh0
    data[:, 6:6+sh_coeffs*3] = shN.reshape(num_gaussians, -1)
    data[:, 6+sh_coeffs*3] = opacities[:, None]
    data[:, 7+sh_coeffs*3:10+sh_coeffs*3] = scales
    data[:, 10+sh_coeffs*3:14+sh_coeffs*3] = quats
else:
    data = np.empty((num_gaussians, 14), dtype='<f4')
    data[:, 0:3] = means
    data[:, 3:6] = sh0
    data[:, 6] = opacities
    data[:, 7:10] = scales
    data[:, 10:14] = quats
```

## Expected Performance Improvements

### SH0 (14 properties):
- Current: ~4.07ms
- Optimized: ~2.72ms (predicted)
- **Improvement: 33% faster**
- **Competitive with plyfile's 2.41ms**

### SH3 (59 properties):
- Current: ~11.89ms
- Optimized: ~9.68ms (predicted)
- **Improvement: 19% faster**
- **Maintain lead over plyfile's 13.73ms**

## Conclusion

The investigation identified that gsply's `concatenate + astype` approach is inefficient due to:
1. Multiple temporary array allocations
2. Separate dtype conversion step

Switching to **preallocate + assign** provides:
- **31-35% faster data preparation**
- **Universal improvement** across all SH degrees
- **Makes gsply competitive with plyfile even for small files**
- **Maintains performance advantage for large files**

This optimization should be implemented to improve gsply's write performance across the board.
