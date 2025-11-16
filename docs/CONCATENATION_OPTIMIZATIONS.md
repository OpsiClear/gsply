# Concatenation and Contiguity Optimizations

## New Features (v0.2.2)

This release adds powerful data concatenation and contiguity optimizations to both `GSData` (CPU) and `GSTensor` (GPU) classes.

### 1. Data Concatenation Methods

#### GSData.add() - Pairwise Concatenation
**CPU-optimized with pre-allocation + direct assignment**

```python
data1 = gsply.plyread("scene1.ply")
data2 = gsply.plyread("scene2.ply")

# Concatenate two datasets
combined = data1.add(data2)
```

**Performance (measured):**
- 10K Gaussians:   412 M/s  (1.10x faster than np.concatenate)
- 100K Gaussians:  106 M/s  (1.56x faster)
- 500K Gaussians:   99 M/s  (1.90x faster)

#### GSData.concatenate() - Bulk Concatenation
**For concatenating multiple arrays - 6.15x faster than repeated `add()`**

```python
scenes = [gsply.plyread(f"scene{i}.ply") for i in range(10)]

# Slow: Pairwise add() (6.35 ms for 10 arrays)
result = scenes[0]
for scene in scenes[1:]:
    result = result.add(scene)

# Fast: Bulk concatenate (1.03 ms - 6.15x faster!)
result = GSData.concatenate(scenes)
```

**Why so much faster?**
- Single allocation vs N-1 intermediate allocations
- Reduces total memory copies
- Break-even at 2+ arrays

####GSTensor.add() - GPU Concatenation
**GPU-optimized with torch.cat() - 18x faster than CPU**

```python
from gsply.torch import GSTensor

tensor1 = GSTensor.from_gsdata(data1, device="cuda")
tensor2 = GSTensor.from_gsdata(data2, device="cuda")

# GPU concatenation
combined = tensor1.add(tensor2)
```

**Performance (measured):**
- 10K:    1.10x faster than CPU
- 100K:   2.44x faster than CPU
- 500K:  18.23x faster than CPU

### 2. Contiguity Optimization

#### Problem: Non-Contiguous Arrays from PLY Files

When loading PLY files via `plyread()`, data is stored in an interleaved `_base` array for zero-copy slicing. This creates **non-contiguous views** with poor cache locality:

```python
data = gsply.plyread("scene.ply")

# Non-contiguous arrays (stride=56 instead of 12)
print(data.means.flags["C_CONTIGUOUS"])  # False
print(data.means.strides)                # (56, 4) - skips 42 bytes!
```

**Performance impact (100K Gaussians):**
- `sum()`:    7.3x slower
- `max()`:   18.9x slower
- `argmax()`: 45.5x slower
- Element-wise ops: 2-4x slower

####GSData.make_contiguous() - Reorganize for Speed

Convert non-contiguous arrays to contiguous layout:

```python
data = gsply.plyread("scene.ply")

# For few operations (< 8) - don't convert
total = data.means.sum()  # Just use as-is

# For many operations (>= 8) - convert first!
data.make_contiguous()  # Pays for itself after ~8 operations
for i in range(100):
    result = data.means.sum() + data.means.max()  # 7.9x faster!
```

**Conversion cost (measured):**
- 1K Gaussians:   0.02 ms
- 10K Gaussians:  0.14 ms
- 100K Gaussians: 2.2 ms
- 1M Gaussians:   25 ms

**Break-even analysis:**
- < 8 operations:   DON'T convert (overhead not justified)
- >= 8 operations:  CONVERT (speedup outweighs cost)
- >= 100 operations: CRITICAL (7.9x total speedup)

**Real-world scenarios (100K Gaussians):**
- Light processing (3 ops):   2.4x slower (don't convert)
- Iterative processing (10x): 2.1x faster (convert!)
- Heavy computation (100x):   7.9x faster (convert!)

**Memory:** Zero overhead (same total memory, just reorganized)

#### GSData.is_contiguous() - Check Status

```python
data = gsply.plyread("scene.ply")
print(data.is_contiguous())  # False

data.make_contiguous()
print(data.is_contiguous())  # True
```

### 3. Mask Layer Management

Both `GSData` and `GSTensor` now support multi-layer boolean masks for complex filtering:

```python
data = gsply.plyread("scene.ply")

# Add named mask layers
data.add_mask_layer("high_opacity", data.opacities > 0.5)
data.add_mask_layer("foreground", data.means[:, 2] < 0)
data.add_mask_layer("small_scale", data.scales.max(axis=1) < 2.0)

# Combine masks with AND/OR logic
combined = data.combine_masks(mode="and")  # All conditions
combined = data.combine_masks(mode="or")   # Any condition

# Apply masks to filter data
filtered = data.apply_masks(mode="and", inplace=False)

# Get specific mask layer
opacity_mask = data.get_mask_layer("high_opacity")

# Remove mask layer
data.remove_mask_layer("small_scale")
```

**GSTensor GPU optimization:**
- Uses `torch.all()`/`torch.any()` instead of Numba
- 100-1000x faster on GPU for mask combination
- 22.6x faster than CPU for masked add() operations

### 4. New Utility Methods

#### GSData.get_sh_degree()
```python
data = gsply.plyread("scene.ply")
degree = data.get_sh_degree()  # Returns 0-3
```

### 5. Direct Masked GPU Transfer

Transfer filtered data to GPU without intermediate CPU copies:

```python
# OLD: Inefficient (CPU copy, then GPU transfer)
filtered = data[mask]
tensor = GSTensor.from_gsdata(filtered, device="cuda")

# NEW: Efficient (filter during GPU transfer)
tensor = GSTensor.from_gsdata(data, mask=mask, device="cuda")
```

## Performance Summary

### Concatenation
| Operation | Dataset | Performance |
|-----------|---------|-------------|
| GSData.add() | 500K | 99 M/s (1.9x faster) |
| GSData.concatenate() | 10x10K | 6.15x faster than pairwise |
| GSTensor.add() GPU | 500K | 18.23x faster than CPU |

### Contiguity
| Operation | Non-contiguous | Contiguous | Speedup |
|-----------|---------------|------------|---------|
| argmax() | 0.780 ms | 0.017 ms | 45.5x |
| max() | 0.318 ms | 0.017 ms | 18.9x |
| sum() | 0.370 ms | 0.051 ms | 7.3x |
| Element ops | 0.498 ms | 0.192 ms | 2.6x |

### Decision Matrix

**When to concatenate:**
- 2+ arrays: Use `GSData.concatenate()` (always faster)
- GPU workload: Use `GSTensor.add()` (18x faster for large data)

**When to make contiguous:**
- < 8 operations: Don't convert
- >= 8 operations: Convert (2-45x speedup per operation)
- Heavy workloads (100+ ops): Critical (7.9x total speedup)

## Examples

### Example 1: Merge Multiple Scenes

```python
import gsply

# Load multiple scene files
scenes = [gsply.plyread(f"scene{i}.ply") for i in range(10)]

# Bulk concatenate (6.15x faster than loop)
combined = gsply.GSData.concatenate(scenes)

# Save merged scene
gsply.plywrite("merged.ply", combined)
```

### Example 2: Iterative Processing

```python
data = gsply.plyread("scene.ply")

# Make contiguous for iterative processing
if not data.is_contiguous():
    data.make_contiguous()  # Pay 2ms once

# Now 7.9x faster for 100 iterations
for i in range(100):
    centroid = data.means.mean(axis=0)
    max_dist = np.linalg.norm(data.means - centroid, axis=1).max()
    # ... more operations
```

### Example 3: GPU Acceleration

```python
from gsply.torch import GSTensor

# Load multiple scenes
data1 = gsply.plyread("scene1.ply")
data2 = gsply.plyread("scene2.ply")

# Transfer to GPU
tensor1 = GSTensor.from_gsdata(data1, device="cuda")
tensor2 = GSTensor.from_gsdata(data2, device="cuda")

# GPU concatenation (18x faster)
combined = tensor1.add(tensor2)

# Convert back to CPU for saving
result = combined.to_gsdata()
gsply.plywrite("output.ply", result)
```

### Example 4: Complex Mask Management

```python
data = gsply.plyread("large_scene.ply")

# Define multiple filtering criteria
data.add_mask_layer("visible", data.opacities > 0.1)
data.add_mask_layer("near", np.linalg.norm(data.means, axis=1) < 10.0)
data.add_mask_layer("clean", data.scales.max(axis=1) < 2.0)

# Try different combinations
strict = data.combine_masks(mode="and", layers=["visible", "clean"])
permissive = data.combine_masks(mode="or")

# Apply best filter
filtered = data.apply_masks(mode="and", layers=["visible", "near"], inplace=False)
gsply.plywrite("filtered.ply", filtered)
```

## API Reference

### GSData Methods

```python
# Concatenation
data.add(other: GSData) -> GSData
GSData.concatenate(arrays: list[GSData]) -> GSData

# Contiguity
data.make_contiguous(inplace: bool = True) -> GSData
data.is_contiguous() -> bool

# Mask management
data.add_mask_layer(name: str, mask: np.ndarray) -> None
data.get_mask_layer(name: str) -> np.ndarray
data.remove_mask_layer(name: str) -> None
data.combine_masks(mode: str = "and", layers: list[str] | None = None) -> np.ndarray
data.apply_masks(mode: str = "and", layers: list[str] | None = None, inplace: bool = False) -> GSData

# Utilities
data.get_sh_degree() -> int
```

### GSTensor Methods

```python
# Concatenation (GPU-optimized)
tensor.add(other: GSTensor) -> GSTensor

# Mask management (GPU-optimized)
tensor.add_mask_layer(name: str, mask: torch.Tensor) -> None
tensor.get_mask_layer(name: str) -> torch.Tensor
tensor.remove_mask_layer(name: str) -> None
tensor.combine_masks(mode: str = "and", layers: list[str] | None = None) -> torch.Tensor
tensor.apply_masks(mode: str = "and", layers: list[str] | None = None, inplace: bool = False) -> GSTensor

# Direct masked transfer
GSTensor.from_gsdata(data: GSData, mask: np.ndarray | None = None, device: str = "cuda") -> GSTensor
```

## Upgrade Guide

Existing code continues to work without changes. New features are opt-in.

**For bulk concatenation:**
```python
# OLD (still works)
result = data1
for d in data_list:
    result = result.add(d)

# NEW (6.15x faster)
result = GSData.concatenate([data1] + data_list)
```

**For iterative processing:**
```python
# OLD (still works, but slower for non-contiguous data)
for i in range(100):
    result = process(data.means)

# NEW (7.9x faster for 100+ operations)
data.make_contiguous()
for i in range(100):
    result = process(data.means)
```

## Compatibility

- Python >= 3.10
- NumPy >= 1.24
- PyTorch >= 2.0 (optional, for GPU features)
- All existing code remains compatible
- New methods are optional enhancements

---

**For complete documentation, see:**
- [README.md](../README.md) - Main documentation
- [AGENTS.md](../AGENTS.md) - Implementation details
- [Benchmark Results](../benchmarks/README.md) - Detailed performance data
