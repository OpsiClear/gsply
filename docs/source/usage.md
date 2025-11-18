# Usage Guide

This guide covers the most common workflows: installation, reading/writing Gaussian splats,
and moving between CPU and GPU containers.

## Installation

Install gsply from PyPI:

```bash
pip install gsply
```

For development work:

```bash
# Editable install with development extras
pip install -e ".[dev]"

# Install documentation extras for Sphinx
pip install -e ".[docs]"
```

**Requirements**: Python 3.10+ with NumPy and Numba (auto-installed).

**Optional**: PyTorch for `GSTensor` GPU workflows.

## Reading Gaussian Splats

The `plyread()` function automatically detects and reads both compressed and uncompressed PLY formats:

```python
from gsply import plyread

# Auto-detects format (compressed or uncompressed)
data = plyread("scene.ply")

print(f"Loaded {len(data):,} Gaussians")
print(f"SH degree: {data.get_sh_degree()}")
print(f"Contiguous: {data.is_contiguous()}")
```

The returned `GSData` object exposes vector-friendly NumPy arrays. All reads use zero-copy
views into a shared `_base` buffer, so slicing and masking operations don't duplicate memory.

### Mask Management

Create named mask layers and combine them with boolean logic:

```python
# Add named mask layers
data.add_mask_layer("high_opacity", data.opacities > 0.25)
data.add_mask_layer("foreground", data.means[:, 2] < 0.0)

# Combine with AND logic (both conditions must pass)
filtered = data.apply_masks(mode="and")

# Or combine with OR logic (either condition passes)
visible = data.apply_masks(mode="or", layers=["high_opacity", "foreground"])

# Combine specific layers programmatically
combined_mask = data.combine_masks(mode="and", layers=["high_opacity", "foreground"])
```

Mask layers persist through slicing, concatenation, and CPU↔GPU transfers.

## Writing Data

The `plywrite()` function automatically optimizes writes based on the data structure:

```python
from gsply import plywrite

# Write uncompressed (auto-optimized)
plywrite("output.ply", data)

# Write compressed format
plywrite("output.ply", data, compressed=True)

# Or use file extension to indicate compression
plywrite("output.compressed.ply", data)
```

**Automatic optimizations**:

- **Zero-copy writes**: When `data._base` exists (from `plyread()`), the buffer is streamed directly
- **Auto-consolidation**: Without `_base`, arrays are automatically consolidated for 2.4x faster writes
- **Format detection**: Compression is selected when `compressed=True` or when the file extension
  is `.compressed.ply` / `.ply_compressed`

## In-Memory Compression

Compress and decompress data without disk I/O:

```python
from gsply import compress_to_bytes, decompress_from_bytes

# Compress to bytes (no disk I/O)
payload = compress_to_bytes(data)

# Decompress from bytes
round_trip = decompress_from_bytes(payload)
assert round_trip.means.shape == data.means.shape
```

Ideal for network transport, streaming, or custom storage backends.
Achieves 71-74% size reduction with PlayCanvas format.

## GPU Acceleration with PyTorch

Transfer data to GPU for training or rendering:

```python
from gsply import GSTensor

# Transfer to GPU (11x faster with zero-copy base tensor)
gstensor = GSTensor.from_gsdata(data, device="cuda", requires_grad=False)

# GPU-optimized mask operations (100-1000x faster than CPU)
mask = gstensor.combine_masks(mode="and")
subset = gstensor[mask]

# Enable gradients for training
gstensor_train = GSTensor.from_gsdata(data, device="cuda", requires_grad=True)
```

`GSTensor` mirrors `GSData` ergonomics: `.add()`, `.concatenate()`, mask helpers,
and `apply_masks()`. When `_base` is present, transfers use a single operation for efficiency.

## Performance Tips

### Contiguity Optimization

For workloads with many array operations, convert to contiguous layout:

```python
# Check if arrays are contiguous
if not data.is_contiguous():
    # Convert (one-time cost, but 2-45x faster per operation)
    data.make_contiguous(inplace=True)

# Now array operations are much faster
result = data.means.sum() + data.means.max()  # Up to 45x faster!
```

**Break-even**: Convert if you will perform ≥8 operations on the data.

### Bulk Concatenation

For merging multiple datasets, use bulk concatenation:

```python
# Fast: Single allocation (5.74x faster)
combined = GSData.concatenate([data1, data2, data3, ...])

# Slower: Repeated pairwise operations
result = data1
for d in [data2, data3, ...]:
    result = result.add(d)  # Creates intermediate allocations
```

### GPU Mask Operations

GPU mask operations are 100-1000x faster than CPU:

```python
# CPU: ~1.43ms for 100K Gaussians, 5 layers
mask = data.combine_masks(mode="and")

# GPU: ~0.001ms for 100K Gaussians, 5 layers (1000x faster!)
gstensor = GSTensor.from_gsdata(data, device="cuda")
mask = gstensor.combine_masks(mode="and")
```
