# Usage Guide

This guide covers the most common workflows: installation, reading/writing Gaussian splats,
format conversion, color conversion, and moving between CPU and GPU containers.

## Installation

**Basic installation:**
```bash
pip install gsply
```

**Optional features:**

GPU acceleration (PyTorch):
```bash
pip install torch
```
Enables `GSTensor`, `plyread_gpu()`, `plywrite_gpu()`, and GPU-accelerated format conversions.

SOG format support:
```bash
pip install gsply[sogs]
```
Enables `sogread()` for reading SOG format files.

**Full installation:**
```bash
pip install gsply[sogs] torch  # GPU + SOG support
```

**Development:**
```bash
# Editable install with development extras
pip install -e ".[dev]"

# Install documentation extras for Sphinx
pip install -e ".[docs]"
```

**Requirements**: Python 3.10+ with NumPy and Numba (auto-installed).

## Reading Gaussian Splats

**Object-Oriented API (Recommended):**
```python
from gsply import GSData

# Auto-detects format (compressed or uncompressed)
data = GSData.load("scene.ply")

print(f"Loaded {len(data):,} Gaussians")
print(f"SH degree: {data.get_sh_degree()}")
print(f"Contiguous: {data.is_contiguous()}")
```

**Functional API:**
```python
from gsply import plyread

# Auto-detects format (compressed or uncompressed)
data = plyread("scene.ply")
```

The returned `GSData` object exposes vector-friendly NumPy arrays. All reads use zero-copy
views into a shared `_base` buffer, so slicing and masking operations don't duplicate memory.

**SOG Format Support:**
```python
from gsply import sogread

# Read SOG format (requires gsply[sogs])
data = sogread("model.sog")  # Returns GSData (same API as plyread)

# In-memory reading from bytes
with open("model.sog", "rb") as f:
    sog_bytes = f.read()
data = sogread(sog_bytes)  # No disk I/O
```

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

## Creating Data from External Sources

**From Arrays (Recommended):**
```python
from gsply import GSData, GSTensor
import numpy as np

# Create from NumPy arrays with auto-format detection
data = GSData.from_arrays(
    means=means,      # (N, 3) xyz positions
    scales=scales,    # (N, 3) scales (auto-detects log vs linear)
    quats=quats,      # (N, 4) quaternions (wxyz order)
    opacities=opacities,  # (N,) opacities (auto-detects logit vs linear)
    sh0=sh0,          # (N, 3) DC spherical harmonics
    shN=shN,          # (N, K, 3) higher-order SH (optional, auto-detects degree)
    format="auto"     # "auto", "ply", or "linear"
)

# Explicit format specification (faster, skips detection)
data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="ply")

# From dictionary
data = GSData.from_dict({
    "means": means,
    "scales": scales,
    "quats": quats,
    "opacities": opacities,
    "sh0": sh0,
    "shN": shN  # optional
}, format="auto")
```

**GPU Tensors:**
```python
import torch

# Create GSTensor from PyTorch tensors
gstensor = GSTensor.from_arrays(
    means=means_tensor,      # torch.Tensor (N, 3)
    scales=scales_tensor,    # torch.Tensor (N, 3)
    quats=quats_tensor,      # torch.Tensor (N, 4)
    opacities=opacities_tensor,  # torch.Tensor (N,)
    sh0=sh0_tensor,          # torch.Tensor (N, 3)
    shN=shN_tensor,          # torch.Tensor (N, K, 3) optional
    format="auto",
    device="cuda",           # Auto-converts to target device
    dtype=torch.float32      # Auto-converts to target dtype
)

# From dictionary
gstensor = GSTensor.from_dict({
    "means": means_tensor,
    "scales": scales_tensor,
    "quats": quats_tensor,
    "opacities": opacities_tensor,
    "sh0": sh0_tensor,
    "shN": shN_tensor
}, format="ply", device="cuda")
```

**Format Presets:**
- `"auto"` (default): Automatically detects PLY format (log-scales/logit-opacities) vs Linear format
- `"ply"`: Explicitly sets PLY format (log-scales/logit-opacities) - use when data matches PLY file spec
- `"linear"` or `"rasterizer"`: Explicitly sets linear format (linear scales/opacities) - use for renderer compatibility

**SH Degree Inference:**
- Automatically infers SH degree from `shN.shape[1]` if not specified
- Valid degrees: 0 (no shN), 1 (9 bands), 2 (24 bands), 3 (45 bands)
- Raises `ValueError` if shape doesn't match a valid degree

## Writing Data

**Object-Oriented API (Recommended):**
```python
from gsply import GSData, GSTensor

# Save uncompressed (auto-optimized)
data.save("output.ply")

# Save compressed format
data.save("output.ply", compressed=True)

# GPU acceleration
gstensor = GSTensor.load("model.ply", device='cuda')
gstensor.save("output.compressed.ply")  # GPU compression (default)
```

**Functional API:**
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
- **Format conversion**: Automatically converts to PLY format (log-scales, logit-opacities) before writing

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

## Format Conversion

PLY files store scales in log-space and opacities in logit-space. Convert between formats as needed:

```python
from gsply import GSData

# Load PLY file (contains log-scales and logit-opacities)
data = GSData.load("scene.ply")

# Convert to linear format for computation/visualization
data.denormalize()  # Converts log-scales → linear, logit-opacities → linear
print(f"Linear opacity range: [{data.opacities.min():.3f}, {data.opacities.max():.3f}]")

# Modify in linear space
data.opacities = np.clip(data.opacities * 1.2, 0, 1)

# Convert back to PLY format before saving
data.normalize()  # Converts linear → log-scales, linear → logit-opacities
data.save("modified.ply")
```

**Color Conversion (SH ↔ RGB):**
```python
# Convert sh0 from SH format to RGB colors
data.to_rgb()  # sh0 now contains RGB colors [0, 1]
data.sh0 *= 1.5  # Make brighter (RGB space)
data.to_sh()  # Convert back to SH format for PLY compatibility
```

## GPU Acceleration with PyTorch

**Object-Oriented API (Recommended):**
```python
from gsply import GSTensor

# Direct GPU loading (auto-detects format)
gstensor = GSTensor.load("model.ply", device="cuda")

# Save with GPU compression
gstensor.save("output.compressed.ply")  # GPU compression (default)

# GPU format conversion
gstensor.denormalize()  # GPU-accelerated (uses torch.exp, torch.sigmoid)
gstensor.to_rgb()  # GPU-accelerated SH → RGB conversion
```

**Functional API:**
```python
from gsply import GSTensor, plyread_gpu, plywrite_gpu

# Transfer to GPU (11x faster with zero-copy base tensor)
gstensor = GSTensor.from_gsdata(data, device="cuda", requires_grad=False)

# Direct GPU I/O
gstensor = plyread_gpu("model.compressed.ply", device="cuda")
plywrite_gpu("output.compressed.ply", gstensor)

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
