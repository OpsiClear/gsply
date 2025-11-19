# gsply API Reference

Complete API reference for gsply - Ultra-Fast Gaussian Splatting PLY I/O Library

**Version:** 0.2.7

**New in v0.2.7:**
- Fused Activation Kernels (`apply_pre_activations`, `apply_pre_deactivations`) - Ultra-fast format conversion (~8-15x faster)
- Optimized Format Conversion - `normalize()` and `denormalize()` now use parallel Numba kernels internally

**New in v0.2.6:**
- Automatic Format Detection (auto-detects log vs linear values)
- Format Helper Functions (`create_ply_format`, `create_rasterizer_format`)
- Strict Format Validation (prevents mixing incompatible formats)

**New in v0.2.5:**
- Object-Oriented I/O API (`data.save()`, `GSData.load()`, `gstensor.save()`, `GSTensor.load()`)
- Format Conversion API (`normalize()`, `denormalize()`) - Convert between linear and PLY formats
- Color Conversion API (`to_rgb()`, `to_sh()`) - Convert between SH and RGB color formats
- SOG Format Support (`sogread()`) - Read SOG (Splat Ordering Grid) format files

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

**Quick Navigation:**
- [gsply API Reference](#gsply-api-reference)
  - [Installation](#installation)
  - [Core I/O](#core-io)
    - [`plyread(file_path)`](#plyreadfile_path)
    - [`plywrite(file_path, means, scales, quats, opacities, sh0, shN=None, compressed=False)`](#plywritefile_path-means-scales-quats-opacities-sh0-shnnone-compressedfalse)
    - [`detect_format(file_path)`](#detect_formatfile_path)
    - [`create_ply_format(sh_degree=0)`](#create_ply_formatsh_degree0)
    - [`create_rasterizer_format(sh_degree=0)`](#create_rasterizer_formatsh_degree0)
    - [`create_linear_format(sh_degree=0)`](#create_linear_formatsh_degree0)
    - [`sogread(file_path | bytes)`](#sogreadfile_path--bytes)
  - [GSData](#gsdata)
    - [`data.save(file_path, compressed=False)`](#datasavefile_path-compressedfalse)
    - [`GSData.load(file_path)`](#gsdataloadfile_path)
    - [`GSData.from_arrays(means, scales, quats, opacities, sh0, shN=None, format='auto', sh_degree=None, sh0_format=SH0_SH)`](#gsdatafrom_arraysmeans-scales-quats-opacities-sh0-shnnone-formatauto-sh_degreenone-sh0_formatsh0_sh)
    - [`GSData.from_dict(data_dict, format='auto', sh_degree=None, sh0_format=SH0_SH)`](#gsdatafrom_dictdata_dict-formatauto-sh_degreenone-sh0_formatsh0_sh)
    - [`data.unpack(include_shN=True)`](#dataunpackinclude_shntrue)
    - [`data.to_dict()`](#datato_dict)
    - [`data.copy()`](#datacopy)
    - [`data.consolidate()`](#dataconsolidate)
    - [`data[index]`](#dataindex)
    - [Format Conversion: Linear ↔ PLY Format](#format-conversion-linear--ply-format)
    - [`data.normalize(inplace=True)` / `data.to_ply_format(inplace=True)`](#datanormalizeinplacetrue--datato_ply_formatinplacetrue)
    - [`data.denormalize(inplace=True)` / `data.from_ply_format(inplace=True)` / `data.to_linear(inplace=True)`](#datadenormalizeinplacetrue--datafrom_ply_formatinplacetrue--datato_linearinplacetrue)
    - [`data.to_rgb(inplace=True)`](#datato_rgbinplacetrue)
    - [`data.to_sh(inplace=True)`](#datato_shinplacetrue)
    - [`len(data)`](#lendata)
    - [`plyread_gpu(file_path, device='cuda')`](#plyread_gpufile_path-devicecuda)
    - [`plywrite_gpu(file_path, gstensor, compressed=True)`](#plywrite_gpufile_path-gstensor-compressedtrue)
  - [Compression APIs](#compression-apis)
    - [`compress_to_bytes(data)`](#compress_to_bytesdata)
    - [`compress_to_arrays(data)`](#compress_to_arraysdata)
    - [`decompress_from_bytes(compressed_bytes)`](#decompress_from_bytescompressed_bytes)
  - [Utility Functions](#utility-functions)
    - [`sh2rgb(sh)`](#sh2rgbsh)
    - [`rgb2sh(rgb)`](#rgb2shrgb)
    - [`logit(x, eps=1e-6)`](#logitx-eps1e-6)
    - [`sigmoid(x)`](#sigmoidx)
    - [`apply_pre_activations(data, min_scale=1e-4, max_scale=100.0, min_quat_norm=1e-8, inplace=True)`](#apply_pre_activationsdata-min_scale1e-4-max_scale1000-min_quat_norm1e-8-inplacetrue)
    - [`apply_pre_deactivations(data, min_scale=1e-9, min_opacity=1e-4, max_opacity=0.9999, inplace=True)`](#apply_pre_deactivationsdata-min_scale1e-9-min_opacity1e-4-max_opacity09999-inplacetrue)
    - [`SH_C0`](#sh_c0)
  - [GSTensor - GPU-Accelerated Dataclass](#gstensor---gpu-accelerated-dataclass)
    - [Key Features](#key-features)
    - [Performance](#performance)
    - [`GSTensor.load(file_path, device='cuda')`](#gstensorloadfile_path-devicecuda)
    - [`GSTensor.from_arrays(means, scales, quats, opacities, sh0, shN=None, format='auto', sh_degree=None, sh0_format=SH0_SH, device=None, dtype=None)`](#gstensorfrom_arraysmeans-scales-quats-opacities-sh0-shnnone-formatauto-sh_degreenone-sh0_formatsh0_sh-devicenone-dtypenone)
    - [`GSTensor.from_dict(data_dict, format='auto', sh_degree=None, sh0_format=SH0_SH, device='cuda', dtype=None)`](#gstensorfrom_dictdata_dict-formatauto-sh_degreenone-sh0_formatsh0_sh-devicecuda-dtypenone)
    - [`gstensor.save(file_path, compressed=True)`](#gstensorsavefile_path-compressedtrue)
    - [`gstensor.save_compressed(file_path)`](#gstensorsave_compressedfile_path)
    - [`GSTensor.from_gsdata(data, device='cuda', dtype=torch.float32, requires_grad=False)`](#gstensorfrom_gsdatadata-devicecuda-dtypetorchfloat32-requires_gradfalse)
    - [`gstensor.to_gsdata()`](#gstensorto_gsdata)
    - [Format Conversion: Linear ↔ PLY Format (GSTensor)](#format-conversion-linear--ply-format-gstensor)
    - [`gstensor.normalize(inplace=True)` / `gstensor.to_ply_format(inplace=True)`](#gstensornormalizeinplacetrue--gstensorto_ply_formatinplacetrue)
    - [`gstensor.denormalize(inplace=True)` / `gstensor.from_ply_format(inplace=True)` / `gstensor.to_linear(inplace=True)`](#gstensordenormalizeinplacetrue--gstensorfrom_ply_formatinplacetrue--gstensorto_linearinplacetrue)
    - [`gstensor.to_rgb(inplace=True)`](#gstensorto_rgbinplacetrue)
    - [`gstensor.to_sh(inplace=True)`](#gstensorto_shinplacetrue)
    - [`gstensor.to(device=None, dtype=None)`](#gstensortodevicenone-dtypenone)
    - [`gstensor.consolidate()`](#gstensorconsolidate)
    - [`gstensor.clone()`](#gstensorclone)
    - [`gstensor.cpu()`](#gstensorcpu)
    - [`gstensor.cuda(device=None)`](#gstensorcudadevicenone)
    - [`gstensor.half()`, `gstensor.float()`, `gstensor.double()`](#gstensorhalf-gstensorfloat-gstensordouble)
    - [`gstensor.unpack(include_shN=True)`](#gstensorunpackinclude_shntrue)
    - [`gstensor.to_dict()`](#gstensorto_dict)
    - [`gstensor[index]`](#gstensorindex)
    - [`len(gstensor)`](#lengstensor)
    - [`gstensor.device` (property)](#gstensordevice-property)
    - [`gstensor.dtype` (property)](#gstensordtype-property)
    - [`gstensor.get_sh_degree()`](#gstensorget_sh_degree)
    - [`gstensor.has_high_order_sh()`](#gstensorhas_high_order_sh)
  - [Complete Workflow Examples](#complete-workflow-examples)
    - [Training Workflow](#training-workflow)
    - [Inference Workflow](#inference-workflow)

---

## Core I/O

### `plyread(file_path)`

Read Gaussian Splatting PLY file (auto-detects format).

Always uses zero-copy optimization for maximum performance.

**Parameters:**
- `file_path` (str | Path): Path to PLY file

**Returns:**
`GSData` dataclass with Gaussian parameters:
- `means`: (N, 3) - Gaussian centers
- `scales`: (N, 3) - Log scales
- `quats`: (N, 4) - Rotations as quaternions (wxyz)
- `opacities`: (N,) - Logit opacities
- `sh0`: (N, 3) - DC spherical harmonics
- `shN`: (N, K, 3) - Higher-order SH coefficients (K=0 for degree 0, K=9 for degree 1, etc.)
- `masks`: (N,) - Boolean mask for filtering Gaussians
- `_base`: (N, P) - Internal array for zero-copy views (private)

**Performance:**
- Uncompressed: 5.7ms for 400K Gaussians (70M/sec), 12.8ms for 1M (78M/sec peak)
- Compressed: 8.5ms for 400K Gaussians (47M/sec), 16.7ms for 1M (60M/sec)
- Scales linearly with data size

**Example:**
```python
from gsply import plyread

# Zero-copy reading - up to 78M Gaussians/sec
data = plyread("model.ply")
print(f"Loaded {data.means.shape[0]} Gaussians with SH degree {data.shN.shape[1]}")

# Access via attributes
positions = data.means
colors = data.sh0

# Unpack for standard GS workflows
means, scales, quats, opacities, sh0, shN = data.unpack()

# Or exclude shN for SH0 data
means, scales, quats, opacities, sh0 = data.unpack(include_shN=False)

# Or get as dictionary
props = data.to_dict()
```

---

### `plywrite(file_path, means, scales, quats, opacities, sh0, shN=None, compressed=False)`

Write Gaussian Splatting PLY file.

**Parameters:**
- `file_path` (str | Path): Output PLY file path (auto-adjusted to `.compressed.ply` if `compressed=True`)
- `means` (np.ndarray): Shape (N, 3) - Gaussian centers
- `scales` (np.ndarray): Shape (N, 3) - Log scales
- `quats` (np.ndarray): Shape (N, 4) - Rotations as quaternions (wxyz)
- `opacities` (np.ndarray): Shape (N,) - Logit opacities
- `sh0` (np.ndarray): Shape (N, 3) - DC spherical harmonics
- `shN` (np.ndarray, optional): Shape (N, K, 3) or (N, K*3) - Higher-order SH
- `compressed` (bool): If True, write compressed format and auto-adjust extension

**Format Selection:**
- `compressed=False` or `.ply` extension -> Uncompressed format (fast)
- `compressed=True` -> Compressed format, saves as `.compressed.ply` automatically
- `.compressed.ply` or `.ply_compressed` extension -> Compressed format

**Performance:**
- Uncompressed SH0: 3.9ms for 100K (26M/s), 19.3ms for 400K (21M/s), 62.2ms for 1M (16M/s)
- Uncompressed SH3: 24.6ms for 100K (4.1M/s), 121.5ms for 400K (3.3M/s), 316.5ms for 1M (3.2M/s)
- Compressed SH0: 3.4ms for 100K (29M/s), 15.0ms for 400K (27M/s), 35.5ms for 1M (28M/s) - 71% smaller
- Compressed SH3: 22.5ms for 100K (4.5M/s), 110.5ms for 400K (3.6M/s), 210ms for 1M (4.8M/s) - 74% smaller
- Up to 2.9x faster when writing data loaded from PLY (zero-copy optimization)

**Example:**
```python
from gsply import plywrite

# Write uncompressed (fast, ~8ms for 400K Gaussians)
plywrite("output.ply", means, scales, quats, opacities, sh0, shN)

# Write compressed (saves as "output.compressed.ply", ~63ms, 3.4x smaller)
plywrite("output.ply", means, scales, quats, opacities, sh0, shN, compressed=True)
```

---

### `detect_format(file_path)`

Detect PLY format type and SH degree.

**Parameters:**
- `file_path` (str | Path): Path to PLY file

**Returns:**
Tuple of (is_compressed, sh_degree):
- `is_compressed` (bool): True if compressed format
- `sh_degree` (int | None): 0-3 for uncompressed, None for compressed/unknown

**Example:**
```python
from gsply import detect_format

is_compressed, sh_degree = detect_format("model.ply")
if is_compressed:
    print("Compressed PlayCanvas format")
else:
    print(f"Uncompressed format with SH degree {sh_degree}")
```

---

### `create_ply_format(sh_degree=0)`

Create format dictionary for PLY file format (log-scales, logit-opacities).

Use this when creating GSData from data that matches PLY file format or when you want to ensure compatibility.

**Parameters:**
- `sh_degree` (int): Spherical harmonics degree (0-3)

**Returns:**
- `FormatDict`: Format dictionary with PLY settings

---

### `create_rasterizer_format(sh_degree=0)`

Create format dictionary for rasterizer format (linear scales, linear opacities).

Use this when creating GSData for rasterization or when you have linear values. Alias: `create_linear_format`.

**Parameters:**
- `sh_degree` (int): Spherical harmonics degree (0-3)

**Returns:**
- `FormatDict`: Format dictionary with linear settings

---

### `create_linear_format(sh_degree=0)`

Alias for `create_rasterizer_format`.

---

### `sogread(file_path | bytes)`

Read SOG (Splat Ordering Grid) format file.

Returns `GSData` container (same as `plyread()`) for consistent API across all formats.
Supports both `.sog` ZIP bundles and folders with separate files.
Can also accept bytes directly for in-memory ZIP extraction.

**Parameters:**
- `file_path` (str | Path | bytes): Path to `.sog` file, folder containing SOG files, or bytes (ZIP data)

**Returns:**
`GSData` dataclass with Gaussian parameters (same structure as `plyread()`):
- `means`: (N, 3) - Gaussian centers
- `scales`: (N, 3) - Log scales
- `quats`: (N, 4) - Rotations as quaternions (wxyz)
- `opacities`: (N,) - Logit opacities
- `sh0`: (N, 3) - DC spherical harmonics
- `shN`: (N, K, 3) - Higher-order SH coefficients (K=0 for degree 0, K=9 for degree 1, etc.)
- `masks`: (N,) - Boolean mask for filtering Gaussians

**Requirements:**
- Requires `gsply[sogs]` installation: `pip install gsply[sogs]`
- Installs `imagecodecs` (fastest WebP decoder) for optimal performance

**Performance:**
- In-memory reading from bytes: ~6x faster than file path reading
- Uses `imagecodecs` for fastest WebP decoding

**Example:**
```python
from gsply import sogread

# Read from file path - returns GSData (same as plyread)
data = sogread("model.sog")
print(f"Loaded {len(data)} Gaussians")
positions = data.means  # Same API as GSData from plyread
colors = data.sh0

# Read from bytes (in-memory, no disk I/O)
with open("model.sog", "rb") as f:
    sog_bytes = f.read()
data = sogread(sog_bytes)  # Returns GSData - fully in-memory extraction and decoding

# Compatible with all GSData operations
means, scales, quats, opacities, sh0, shN = data.unpack()
```

**Note:** SOG format is compatible with PlayCanvas splat-transform format.

---

## GSData

Container dataclass for Gaussian Splatting data with zero-copy optimization.

`GSData` is returned by `plyread()` and provides efficient access to Gaussian parameters through both direct attributes and convenience methods. All arrays are mutable and can be modified in-place. Arrays can be views into a shared `_base` array for maximum performance (zero memory overhead).

**Attributes:**

- `means` (np.ndarray): Shape (N, 3) - Gaussian centers (xyz positions)
- `scales` (np.ndarray): Shape (N, 3) - Log scales for each axis
- `quats` (np.ndarray): Shape (N, 4) - Rotations as quaternions (wxyz order)
- `opacities` (np.ndarray): Shape (N,) - Logit opacities (before sigmoid)
- `sh0` (np.ndarray): Shape (N, 3) - DC spherical harmonics (RGB color basis)
- `shN` (np.ndarray | None): Shape (N, K, 3) - Higher-order SH coefficients
  - K=0 for SH degree 0 (no higher-order)
  - K=9 for SH degree 1
  - K=24 for SH degree 2
  - K=45 for SH degree 3
- `masks` (np.ndarray): Shape (N,) boolean - Mask for filtering (initialized to all True)
- `_base` (np.ndarray | None): Shape (N, P) - Private base array (auto-managed, do not modify)

**Example:**
```python
from gsply import plyread

data = plyread("scene.ply")
print(f"Loaded {len(data)} Gaussians")

# Direct attribute access
positions = data.means
colors = data.sh0

# Mutable - modify in place
data.means[0] = [1, 2, 3]
data.sh0 *= 1.5  # Make brighter
```

---

### `data.save(file_path, compressed=False)`

Save GSData to PLY file.

Convenience method that wraps `plywrite()` for object-oriented API.

**Parameters:**
- `file_path` (str | Path): Output PLY file path
- `compressed` (bool): If True, write compressed format (default False)

**Example:**
```python
from gsply import plyread

# Load data
data = plyread("input.ply")

# Save uncompressed
data.save("output.ply")

# Save compressed
data.save("output.ply", compressed=True)
```

---

### `GSData.load(file_path)`

Load GSData from PLY file.

Convenience classmethod that wraps `plyread()` for object-oriented API. Auto-detects compressed and uncompressed formats.

**Parameters:**
- `file_path` (str | Path): Path to PLY file

**Returns:**
- `GSData`: Container with loaded data

**Example:**
```python
from gsply import GSData

# Load using classmethod (auto-detects format)
data = GSData.load("scene.ply")
print(f"Loaded {len(data)} Gaussians")

# Same as plyread()
data2 = plyread("scene.ply")  # Equivalent
```

---

### `GSData.from_arrays(means, scales, quats, opacities, sh0, shN=None, format='auto', sh_degree=None, sh0_format=SH0_SH)`

Create GSData from individual arrays with format preset.

Convenient factory method for creating GSData from external arrays with automatic format detection or explicit format presets.

**Parameters:**
- `means` (np.ndarray): (N, 3) - Gaussian centers
- `scales` (np.ndarray): (N, 3) - Scale parameters
- `quats` (np.ndarray): (N, 4) - Rotation quaternions
- `opacities` (np.ndarray): (N,) - Opacity values
- `sh0` (np.ndarray): (N, 3) - DC spherical harmonics
- `shN` (np.ndarray, optional): (N, K, 3) - Higher-order SH coefficients
- `format` (str): Format preset - "auto" (detect), "ply" (log/logit), "linear" or "rasterizer" (linear)
- `sh_degree` (int, optional): SH degree (0-3) - auto-detected from shN if None
- `sh0_format` (DataFormat): Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH

**Returns:**
- `GSData`: Object with specified format

**Example:**
```python
from gsply import GSData
import numpy as np

# Auto-detect format from values
data = GSData.from_arrays(means, scales, quats, opacities, sh0)

# Explicit PLY format (log-scales, logit-opacities)
data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="ply")

# Explicit linear format (for rasterizer)
data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="linear")
```

---

### `GSData.from_dict(data_dict, format='auto', sh_degree=None, sh0_format=SH0_SH)`

Create GSData from dictionary with format preset.

Convenient factory method for creating GSData from a dictionary with automatic format detection or explicit format presets.

**Parameters:**
- `data_dict` (dict): Dictionary with keys: means, scales, quats, opacities, sh0, shN (optional)
- `format` (str): Format preset - "auto" (detect), "ply" (log/logit), "linear" or "rasterizer" (linear)
- `sh_degree` (int, optional): SH degree (0-3) - auto-detected from shN if None
- `sh0_format` (DataFormat): Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH

**Returns:**
- `GSData`: Object with specified format

**Example:**
```python
from gsply import GSData

# From dictionary with auto-detection
data = GSData.from_dict({
    "means": means, "scales": scales, "quats": quats,
    "opacities": opacities, "sh0": sh0, "shN": shN
})

# Explicit PLY format
data = GSData.from_dict(data_dict, format="ply")

# Explicit linear format
data = GSData.from_dict(data_dict, format="linear")
```

---

### `data.unpack(include_shN=True)`

Unpack Gaussian data into tuple of individual arrays.

Most useful for passing data to rendering functions that expect separate arrays rather than a container object.

**Parameters:**
- `include_shN` (bool): If True, include shN in output (default: True)

**Returns:**
- If `include_shN=True`: `(means, scales, quats, opacities, sh0, shN)`
- If `include_shN=False`: `(means, scales, quats, opacities, sh0)`

**Example:**
```python
data = plyread("scene.ply")

# Full unpacking (recommended for SH1-3)
means, scales, quats, opacities, sh0, shN = data.unpack()
render(means, scales, quats, opacities, sh0, shN)

# Without higher-order SH (recommended for SH0)
means, scales, quats, opacities, sh0 = data.unpack(include_shN=False)
render(means, scales, quats, opacities, sh0)

# Tuple unpacking for plywrite
plywrite("output.ply", *data.unpack())
```

---

### `data.to_dict()`

Convert Gaussian data to dictionary for keyword argument unpacking.

Useful when calling functions that accept keyword arguments matching the Gaussian parameter names.

**Returns:**
- Dictionary with keys: `means`, `scales`, `quats`, `opacities`, `sh0`, `shN`

**Example:**
```python
data = plyread("scene.ply")

# Dictionary unpacking
props = data.to_dict()
render(**props)  # Unpack as kwargs

# Access by key
positions = props['means']
colors = props['sh0']
```

---

### `data.copy()`

Create deep copy of GSData with independent arrays.

Modifications to the copy will not affect the original data. Optimized to use `_base` array when available (faster than copying individual arrays).

**Returns:**
- `GSData`: New GSData object with copied arrays

**Example:**
```python
data = plyread("scene.ply")

# Create independent copy
data_copy = data.copy()
data_copy.means[0] = 0  # Doesn't affect original

# Use for creating variations
bright = data.copy()
bright.sh0 *= 1.5  # Make brighter
```

---

### `data.consolidate()`

Consolidate separate arrays into single base array for faster slicing operations.

Creates a `_base` array from separate arrays, which improves performance for boolean masking operations (1.5x faster). Only beneficial if you plan to perform many boolean mask operations on the same data.

**Returns:**
- `GSData`: New GSData with `_base` array, or self if already consolidated

**Performance:**
- One-time cost: ~2ms per 100K Gaussians
- Benefit: 1.5x faster boolean masking
- Most useful before multiple filter operations

**Example:**
```python
data = plyread("scene.ply")

# Consolidate for faster filtering
data_consolidated = data.consolidate()

# Now boolean masking is 1.5x faster
high_opacity = data_consolidated[data_consolidated.opacities > 0.5]
low_opacity = data_consolidated[data_consolidated.opacities <= 0.5]
```

---

### `data[index]`

Slice GSData using standard Python indexing.

Supports integers, slices, boolean masks, and fancy indexing. Returns views when possible (zero-copy).

**Indexing Modes:**
- Integer: `data[0]` - Returns tuple of (means, scales, quats, opacities, sh0, shN, masks)
- Slice: `data[100:200]` - Returns new GSData with subset
- Step: `data[::10]` - Returns every 10th Gaussian
- Boolean mask: `data[mask]` - Filter by boolean array
- Fancy: `data[[0, 10, 20]]` - Select specific indices

**Example:**
```python
data = plyread("scene.ply")

# Single Gaussian (returns tuple)
means, scales, quats, opacities, sh0, shN, masks = data[0]

# Slice (returns GSData)
subset = data[100:200]

# Boolean mask (returns GSData)
high_opacity = data[data.opacities > 0.5]

# Step slicing (returns GSData)
every_10th = data[::10]
```

---

### Format Conversion: Linear ↔ PLY Format

GSData provides methods to convert between linear and PLY formats:

**PLY Format** (used in `.ply` files):
- Scales: log-space (`log(scale)`)
- Opacities: logit-space (`logit(opacity)`)

**Linear Format** (easier for computation/visualization):
- Scales: linear values (e.g., `0.1`, `0.5`, `1.0`)
- Opacities: linear values in range `[0, 1]`

---

### `data.normalize(inplace=True)` / `data.to_ply_format(inplace=True)`

Convert **linear scales/opacities → PLY format** (log-scales, logit-opacities).

**Converts:**
- Linear scales → log-scales: `log(scale)`
- Linear opacities → logit-opacities: `logit(opacity)` (uses `gsply.logit()` with `eps=1e-4`)

**When to use:** When you have linear data and need to save to PLY format.

**Note:** Uses `apply_pre_deactivations()` internally with fused Numba kernel for optimal performance (~8-15x faster). Uses `gsply.logit()` from `utils.py` (Numba-optimized CPU implementation). Behavior matches `GSTensor.normalize()` for consistency.

**Parameters:**
- `inplace` (bool): If True, modify this object in-place (default). If False, return new object

**Returns:**
- `GSData`: GSData object (self if inplace=True, new object otherwise)

**Example:**
```python
from gsply import GSData, plywrite
import numpy as np

# Create GSData with linear scales and opacities
data = GSData(
    means=np.random.randn(100, 3),
    scales=np.random.rand(100, 3) * 0.1 + 0.01,  # Linear scales: [0.01, 0.11]
    quats=np.random.randn(100, 4),
    opacities=np.random.rand(100) * 0.8 + 0.1,  # Linear opacities: [0.1, 0.9]
    sh0=np.random.randn(100, 3),
    shN=None
)

# Convert to PLY format in-place (modifies data)
data.normalize()  # or: data.normalize(inplace=True) or data.to_ply_format()
# Now ready to save
plywrite("output.ply", data)

# Or create a copy if you need to keep original
ply_data = data.normalize(inplace=False)
```

---

### `data.denormalize(inplace=True)` / `data.from_ply_format(inplace=True)` / `data.to_linear(inplace=True)`

Convert **PLY format → linear scales/opacities** (log-scales → linear, logit-opacities → linear).

**Converts:**
- Log-scales → linear scales: `exp(log_scale)`
- Logit-opacities → linear opacities: `sigmoid(logit)` (uses `gsply.sigmoid()`)

**When to use:** When you load PLY files and need linear values for computations or visualization.

**Note:** Uses `apply_pre_activations()` internally with fused Numba kernel for optimal performance (~8-15x faster). Uses `gsply.sigmoid()` from `utils.py` (Numba-optimized CPU implementation). Behavior matches `GSTensor.denormalize()` for consistency.

**Parameters:**
- `inplace` (bool): If True, modify this object in-place (default). If False, return new object

**Returns:**
- `GSData`: GSData object (self if inplace=True, new object otherwise)

**Example:**
```python
from gsply import plyread

# Load PLY file (contains log-scales and logit-opacities)
data = plyread("scene.ply")

# Convert to linear format in-place (modifies data)
data.denormalize()  # or: data.denormalize(inplace=True) or data.from_ply_format() or data.to_linear()

# Now scales and opacities are in linear space
print(f"Linear opacity range: [{data.opacities.min():.3f}, {data.opacities.max():.3f}]")
# Output: Linear opacity range: [0.000, 1.000]

# Easier to work with linear values
high_opacity = data[data.opacities > 0.5]  # Filter by linear opacity

# Or create a copy if you need to keep PLY format
linear_data = data.denormalize(inplace=False)
```

---

### `data.to_rgb(inplace=True)`

Convert sh0 from spherical harmonics (SH) format to RGB color format.

**Converts:**
- SH DC coefficients → RGB colors: `rgb = sh0 * SH_C0 + 0.5`

**When to use:** When you need RGB colors in [0, 1] range for visualization or color manipulation.

**Note:** Uses Numba JIT-compiled parallel function for optimal performance. Modifies array in-place for efficiency.

**Parameters:**
- `inplace` (bool): If True, modify this object in-place (default). If False, return new object

**Returns:**
- `GSData`: GSData object (self if inplace=True, new object otherwise)

**Example:**
```python
from gsply import plyread

# Load PLY file (sh0 is in SH format)
data = plyread("scene.ply")

# Convert to RGB format in-place
data.to_rgb()  # or: data.to_rgb(inplace=True)

# Now sh0 contains RGB colors [0, 1]
print(f"RGB color range: [{data.sh0.min():.3f}, {data.sh0.max():.3f}]")
# Output: RGB color range: [0.000, 1.000]

# Modify colors in RGB space
data.sh0 *= 1.5  # Make brighter
data.sh0 = np.clip(data.sh0, 0, 1)  # Clamp to valid range

# Convert back to SH format if needed
data.to_sh()

# Or create a copy if you need to keep SH format
rgb_data = data.to_rgb(inplace=False)
```

---

### `data.to_sh(inplace=True)`

Convert sh0 from RGB color format to spherical harmonics (SH) format.

**Converts:**
- RGB colors → SH DC coefficients: `sh0 = (rgb - 0.5) / SH_C0`

**When to use:** When you have RGB colors and need to convert back to SH format for PLY file compatibility.

**Note:** Uses Numba JIT-compiled parallel function for optimal performance. Modifies array in-place for efficiency.

**Parameters:**
- `inplace` (bool): If True, modify this object in-place (default). If False, return new object

**Returns:**
- `GSData`: GSData object (self if inplace=True, new object otherwise)

**Example:**
```python
from gsply import GSData, plywrite
import numpy as np

# Create GSData with RGB colors
rgb_colors = np.random.rand(1000, 3).astype(np.float32)
data = GSData(
    means=np.random.randn(1000, 3).astype(np.float32),
    scales=np.ones((1000, 3), dtype=np.float32) * 0.01,
    quats=np.tile([1, 0, 0, 0], (1000, 1)).astype(np.float32),
    opacities=np.ones(1000, dtype=np.float32) * 0.5,
    sh0=rgb_colors,  # RGB colors
    shN=None,
)

# Convert to SH format in-place
data.to_sh()  # or: data.to_sh(inplace=True)

# Now sh0 contains SH DC coefficients
# Ready to save to PLY file
plywrite("output.ply", data)

# Or create a copy if you need to keep RGB format
sh_data = data.to_sh(inplace=False)
```

---

### `len(data)`

Get number of Gaussians in the dataset.

**Returns:**
- `int`: Number of Gaussians (equivalent to `data.means.shape[0]`)

**Example:**
```python
data = plyread("scene.ply")
print(f"Loaded {len(data)} Gaussians")
```

---

### `plyread_gpu(file_path, device='cuda')`

Read compressed PLY file directly to GPU using GPU-accelerated decompression.

**Parameters:**
- `file_path` (str | Path): Path to compressed PLY file
- `device` (str): Target GPU device (default "cuda")

**Returns:**
`GSTensor` with decompressed data on GPU

**Performance:**
- 4-5x faster than CPU decompression + GPU transfer
- Direct GPU memory allocation (no intermediate CPU copies)
- ~19ms for 365K Gaussians (19 M/s throughput)

**Example:**
```python
from gsply import plyread_gpu

# Read compressed PLY directly to GPU
gstensor = plyread_gpu("scene.compressed.ply", device="cuda")
print(f"Loaded {len(gstensor):,} Gaussians on GPU")

# Access GPU tensors directly
positions_gpu = gstensor.means  # Already on GPU
colors_gpu = gstensor.sh0
```

**Note:** Only supports compressed PLY format. Requires PyTorch to be installed.

---

### `plywrite_gpu(file_path, gstensor, compressed=True)`

Write GSTensor to compressed PLY file using GPU compression.

**Parameters:**
- `file_path` (str | Path): Output file path
- `gstensor` (GSTensor): GSTensor on GPU to compress and save
- `compressed` (bool): Must be True (default True, required for GPU path)

**Returns:**
None (writes to file)

**Performance:**
- 4-5x faster compression than CPU Numba
- GPU reduction for chunk bounds (instant)
- Minimal CPU-GPU data transfer
- ~18ms for 365K Gaussians (20 M/s throughput)

**Example:**
```python
from gsply import plyread_gpu, plywrite_gpu

# Read to GPU
gstensor = plyread_gpu("input.compressed.ply", device="cuda")

# ... GPU operations ...

# Write back to compressed file
plywrite_gpu("output.compressed.ply", gstensor)
```

**Note:** Only supports compressed format. GSTensor must be on GPU (use `gstensor.to("cuda")` if needed).

---

## Compression APIs

### `compress_to_bytes(data)`

Compress Gaussian splatting data to bytes (PlayCanvas format) without writing to disk.

Useful for network transfer, streaming, or custom storage solutions.

**Parameters:**
- `data` (GSData): Gaussian data from `plyread()` or created manually
  - Alternative: Pass individual arrays for backward compatibility

**Returns:**
`bytes`: Complete compressed PLY file as bytes

**Example:**
```python
from gsply import plyread, compress_to_bytes

# Method 1: Clean API with GSData (recommended)
data = plyread("model.ply")
compressed_bytes = compress_to_bytes(data)  # Simple!

# Method 2: Individual arrays (backward compatible)
compressed_bytes = compress_to_bytes(
    means, scales, quats, opacities, sh0, shN
)

# Send over network or store in database
with open("output.compressed.ply", "wb") as f:
    f.write(compressed_bytes)
```

---

### `compress_to_arrays(data)`

Compress Gaussian splatting data to component arrays (PlayCanvas format).

Returns separate components for custom processing or partial updates.

**Parameters:**
- `data` (GSData): Gaussian data from `plyread()` or created manually
  - Alternative: Pass individual arrays for backward compatibility

**Returns:**
Tuple containing:
- `header_bytes` (bytes): PLY header as bytes
- `chunk_bounds` (np.ndarray): Shape (num_chunks, 18) float32 - Chunk boundary array
- `packed_data` (np.ndarray): Shape (N, 4) uint32 - Main compressed data
- `packed_sh` (np.ndarray | None): Shape varies, uint8 - Compressed SH data if present

**Example:**
```python
from gsply import plyread, compress_to_arrays
from io import BytesIO

# Method 1: Clean API with GSData (recommended)
data = plyread("model.ply")
header, chunks, packed, sh = compress_to_arrays(data)  # Simple!

# Method 2: Individual arrays (backward compatible)
header, chunks, packed, sh = compress_to_arrays(
    means, scales, quats, opacities, sh0, shN
)

# Process components individually
print(f"Header size: {len(header)} bytes")
print(f"Chunks: {chunks.shape[0]} chunks")
print(f"Packed data: {packed.nbytes} bytes")

# Manually assemble if needed
buffer = BytesIO()
buffer.write(header)
buffer.write(chunks.tobytes())
buffer.write(packed.tobytes())
if sh is not None:
    buffer.write(sh.tobytes())

compressed_bytes = buffer.getvalue()
```

---

### `decompress_from_bytes(compressed_bytes)`

Decompress Gaussian splatting data from bytes (PlayCanvas format) without reading from disk.

Symmetric with `compress_to_bytes()` - perfect for network transfer, streaming, or custom storage.

**Parameters:**
- `compressed_bytes` (bytes): Complete compressed PLY file as bytes

**Returns:**
`GSData` dataclass with decompressed Gaussian parameters:
- `means`: (N, 3) - Gaussian centers
- `scales`: (N, 3) - Log scales
- `quats`: (N, 4) - Rotations as quaternions (wxyz)
- `opacities`: (N,) - Logit opacities
- `sh0`: (N, 3) - DC spherical harmonics
- `shN`: (N, K, 3) - Higher-order SH coefficients
- `masks`: (N,) - Boolean mask (all True for decompressed data)
- `_base`: None (not applicable for decompressed data)

**Example:**
```python
from gsply import compress_to_bytes, decompress_from_bytes, plyread

# Example 1: Round-trip without disk I/O
data = plyread("model.ply")
compressed = compress_to_bytes(data)
data_restored = decompress_from_bytes(compressed)
# data_restored is ready to use!

# Example 2: Network transfer
# Sender side
compressed_bytes = compress_to_bytes(data)
# send compressed_bytes over network...

# Receiver side
# ...receive compressed_bytes from network
data = decompress_from_bytes(compressed_bytes)
# No temporary files needed!

# Example 3: Database storage
import sqlite3
conn = sqlite3.connect('gaussians.db')
conn.execute('CREATE TABLE IF NOT EXISTS models (id INTEGER, data BLOB)')
# Store
compressed = compress_to_bytes(data)
conn.execute('INSERT INTO models VALUES (?, ?)', (1, compressed))
# Retrieve
row = conn.execute('SELECT data FROM models WHERE id = 1').fetchone()
data_restored = decompress_from_bytes(row[0])
```

**Note:** PlayCanvas compression is lossy (quantization). Decompressed data will be very close to but not exactly identical to the original.

---

## Utility Functions

### `sh2rgb(sh)`

Convert spherical harmonic DC coefficients to RGB colors.

Converts the DC component (sh0) of spherical harmonics to standard RGB color values in the range [0, 1]. Useful for visualization and color manipulation.

**Parameters:**
- `sh` (np.ndarray | float): SH DC coefficients - Shape (N, 3) or scalar

**Returns:**
- `np.ndarray | float`: RGB colors in [0, 1] range

**Example:**
```python
from gsply import plyread, sh2rgb

data = plyread("scene.ply")

# Convert SH to RGB for visualization
rgb_colors = sh2rgb(data.sh0)
print(f"First color: RGB({rgb_colors[0, 0]:.3f}, {rgb_colors[0, 1]:.3f}, {rgb_colors[0, 2]:.3f})")

# Modify colors in RGB space
rgb_colors *= 1.5  # Make brighter
data.sh0 = rgb2sh(np.clip(rgb_colors, 0, 1))  # Convert back
```

---

### `rgb2sh(rgb)`

Convert RGB colors to spherical harmonic DC coefficients.

Converts standard RGB color values in the range [0, 1] to the DC component (sh0) of spherical harmonics. Inverse of `sh2rgb()`.

**Parameters:**
- `rgb` (np.ndarray | float): RGB colors in [0, 1] range - Shape (N, 3) or scalar

**Returns:**
- `np.ndarray | float`: SH DC coefficients

**Example:**
```python
from gsply import rgb2sh, plywrite
import numpy as np

# Create Gaussians with specific RGB colors
n = 1000
means = np.random.randn(n, 3).astype(np.float32)
scales = np.ones((n, 3), dtype=np.float32) * 0.01
quats = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
opacities = np.ones(n, dtype=np.float32)

# Set colors in RGB space
rgb_colors = np.random.rand(n, 3).astype(np.float32)  # Random colors
sh0 = rgb2sh(rgb_colors)  # Convert to SH

plywrite("colored.ply", means, scales, quats, opacities, sh0, None)
```

---

### `logit(x, eps=1e-6)`

Compute logit function (inverse sigmoid) with numerical stability.

Optimized CPU implementation using Numba JIT compilation with parallel execution. Converts probabilities in [0, 1] range to log-odds space. Formula: `log(x / (1 - x))`

**Parameters:**
- `x` (np.ndarray | float): Input values in [0, 1] range (probabilities) - Shape (N,) or scalar
- `eps` (float): Epsilon for numerical stability (clamping) - Default: 1e-6

**Returns:**
- `np.ndarray | float`: Logit values

**Performance:**
- Optimized with Numba parallel JIT compilation
- Supports both scalar and array inputs
- Automatically clamps values to [eps, 1-eps] for numerical stability

**Example:**
```python
from gsply import logit
import numpy as np

# Scalar
prob = 0.5
logit_val = logit(prob)  # 0.0

# Array
probs = np.array([0.1, 0.5, 0.9])
logit_vals = logit(probs)

# Edge cases are handled automatically
probs_edge = np.array([0.0, 1.0])
logit_vals = logit(probs_edge)  # Clamped to finite values
```

---

### `sigmoid(x)`

Compute sigmoid function (inverse logit) with numerical stability.

Optimized CPU implementation using Numba JIT compilation with parallel execution. Converts log-odds space to probabilities in [0, 1] range. Formula: `1 / (1 + exp(-x))`

**Parameters:**
- `x` (np.ndarray | float): Input values (logits) - Shape (N,) or scalar

**Returns:**
- `np.ndarray | float`: Values in [0, 1] range (probabilities)

**Performance:**
- Optimized with Numba parallel JIT compilation
- Supports both scalar and array inputs
- Uses stable sigmoid implementation (avoids overflow)

**Example:**
```python
from gsply import sigmoid, logit
import numpy as np

# Scalar
logit_val = 0.0
prob = sigmoid(logit_val)  # 0.5

# Array
logit_vals = np.array([-10.0, 0.0, 10.0])
probs = sigmoid(logit_vals)  # [~0.0, 0.5, ~1.0]

# Round-trip
probs = np.array([0.1, 0.5, 0.9])
logit_vals = logit(probs)
probs_restored = sigmoid(logit_vals)  # Should match original
assert np.allclose(probs, probs_restored)
```

---

### `apply_pre_activations(data, min_scale=1e-4, max_scale=100.0, min_quat_norm=1e-8, inplace=True)`

Activate GSData attributes (scales, opacities, quaternions) in a single fused pass.

This function uses a fused Numba kernel that processes all three attributes together for optimal performance (~8-15x faster than individual operations). Converts PLY format (log-scales, logit-opacities) to linear format.

**Converts:**
- Log-scales → linear scales: `exp(log_scale)` with clamping
- Logit-opacities → linear opacities: `sigmoid(logit)`
- Quaternions → normalized quaternions (with safety floor)

**Parameters:**
- `data` (GSData): GSData instance to process
- `min_scale` (float): Minimum allowed scale value after exponentiation - Default: 1e-4
- `max_scale` (float): Maximum allowed scale value after exponentiation - Default: 100.0
- `min_quat_norm` (float): Norm floor for normalizing quaternions (avoids NaNs) - Default: 1e-8
- `inplace` (bool): If False, returns a copy before activation - Default: True

**Returns:**
- `GSData`: GSData with activated attributes (either modified in-place or copy)

**Performance:**
- **~8-15x faster** than individual operations
- Uses parallel Numba JIT compilation
- Single-pass processing reduces memory overhead
- Improves cache locality

**Example:**
```python
from gsply import plyread, apply_pre_activations

# Load PLY file (contains log-scales and logit-opacities)
data = plyread("scene.ply")

# Activate in-place (modifies data)
apply_pre_activations(data, inplace=True)

# Now scales and opacities are in linear space
print(f"Linear opacity range: [{data.opacities.min():.3f}, {data.opacities.max():.3f}]")
# Output: Linear opacity range: [0.000, 1.000]

# Quaternions are normalized
quat_norms = np.linalg.norm(data.quats, axis=1)
assert np.allclose(quat_norms, 1.0)  # All normalized

# Or create a copy
activated = apply_pre_activations(data, inplace=False)
```

**Note:** This function is used internally by `denormalize()` for optimal performance. You can use it directly for fine-grained control over activation parameters.

---

### `apply_pre_deactivations(data, min_scale=1e-9, min_opacity=1e-4, max_opacity=0.9999, inplace=True)`

Deactivate GSData attributes (scales, opacities) in a single fused pass.

This function uses a fused Numba kernel that processes scales and opacities together for optimal performance (~8-15x faster than individual operations). Converts linear format to PLY format (log-scales, logit-opacities).

**Converts:**
- Linear scales → log-scales: `log(scale)` with clamping
- Linear opacities → logit-opacities: `logit(opacity)` with clamping

**Parameters:**
- `data` (GSData): GSData instance to process
- `min_scale` (float): Minimum allowed scale value before logarithm - Default: 1e-9
- `min_opacity` (float): Minimum allowed opacity value before logit - Default: 1e-4
- `max_opacity` (float): Maximum allowed opacity value before logit - Default: 0.9999
- `inplace` (bool): If False, returns a copy before deactivation - Default: True

**Returns:**
- `GSData`: GSData with deactivated attributes (either modified in-place or copy)

**Performance:**
- **~8-15x faster** than individual operations
- Uses parallel Numba JIT compilation
- Single-pass processing reduces memory overhead
- Improves cache locality

**Example:**
```python
from gsply import GSData, apply_pre_deactivations
import numpy as np

# Create GSData with linear scales and opacities
data = GSData(
    means=np.random.randn(100, 3).astype(np.float32),
    scales=np.random.rand(100, 3).astype(np.float32) * 0.1 + 0.01,
    quats=np.random.randn(100, 4).astype(np.float32),
    opacities=np.random.rand(100).astype(np.float32) * 0.8 + 0.1,
    sh0=np.random.randn(100, 3).astype(np.float32),
    shN=None,
)

# Deactivate in-place (modifies data)
apply_pre_deactivations(data, inplace=True)

# Now scales and opacities are in PLY format (log/logit)
print(f"Log-scale range: [{data.scales.min():.3f}, {data.scales.max():.3f}]")
# Scales are now in log-space

# Or create a copy
deactivated = apply_pre_deactivations(data, inplace=False)
```

**Note:** This function is used internally by `normalize()` for optimal performance. You can use it directly for fine-grained control over deactivation parameters.

---

### `SH_C0`

Constant for spherical harmonic DC coefficient normalization.

This constant (0.28209479177387814) is used in the conversion between SH coefficients and RGB colors. It represents the normalization factor for the 0th order spherical harmonic.

**Type:** `float`

**Value:** `0.28209479177387814`

**Example:**
```python
from gsply import SH_C0

# Manual conversion (equivalent to sh2rgb/rgb2sh)
rgb = sh * SH_C0 + 0.5  # SH to RGB
sh = (rgb - 0.5) / SH_C0  # RGB to SH
```

---

## GSTensor - GPU-Accelerated Dataclass

`GSTensor` is a PyTorch-backed version of `GSData` that enables GPU-accelerated operations.

**Note:** GSTensor requires PyTorch to be installed. Install with: `pip install torch`

**Example:**
```python
from gsply import plyread, GSTensor

# Load data from disk (CPU NumPy)
data = plyread("model.ply")

# Convert to GPU tensors (11x faster with _base optimization)
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Access GPU tensors
positions_gpu = gstensor.means  # torch.Tensor on GPU
colors_gpu = gstensor.sh0       # torch.Tensor on GPU

# Unpack for rendering functions
means, scales, quats, opacities, sh0, shN = gstensor.unpack()
rendered = render_gaussians(means, scales, quats, opacities, sh0)

# Or use dict unpacking
rendered = render_gaussians(**gstensor.to_dict())

# Slice on GPU (zero-cost views)
subset = gstensor[100:200]      # Returns GSTensor view

# Training workflow
gstensor_trainable = GSTensor.from_gsdata(data, device='cuda', requires_grad=True)
loss = render_loss(gstensor_trainable.means, ...)
loss.backward()

# Convert back to CPU NumPy
data_cpu = gstensor.to_gsdata()
```

### Key Features

- **11x Faster GPU Transfer**: When data has `_base` (from `plyread()` or `consolidate()`), GPU transfer is 11x faster than manual stacking
- **Zero-Copy Views**: GPU slicing creates views (no memory overhead)
- **Device Management**: Seamless transfer between CPU/GPU with `.to()`, `.cpu()`, `.cuda()`
- **Training Support**: Optional gradient tracking with `requires_grad=True`
- **Type Conversions**: `half()`, `float()`, `double()` for precision control
- **Optimized Slicing**: 25x faster boolean masking with `consolidate()`

### Performance

**GPU Transfer (400K Gaussians, SH0, RTX 3090 Ti):**
- **With `_base` optimization**: 1.99 ms (zero CPU copy overhead)
- **Without `_base` (fallback)**: 22.78 ms (requires CPU stacking)
- **Speedup**: 11.4x faster with `_base`

**Memory Efficiency:**
- Single tensor transfer vs 5 separate transfers
- 50% less I/O (no CPU copy when using `_base`)
- GPU views are free (zero additional memory)

---

### `GSTensor.load(file_path, device='cuda')`

Load GSTensor from PLY file.

Convenience classmethod that auto-detects format and loads to GPU. Uses GPU decompression for compressed files, CPU read + GPU transfer for uncompressed.

**Parameters:**
- `file_path` (str | Path): Path to PLY file
- `device` (str | torch.device): Target device (default "cuda")

**Returns:**
- `GSTensor`: Container with loaded data on GPU

**Performance:**
- Compressed: GPU decompression (4-5x faster than CPU)
- Uncompressed: CPU read + GPU transfer

**Example:**
```python
from gsply import GSTensor

# Load compressed PLY directly to GPU
gstensor = GSTensor.load("scene.compressed.ply", device="cuda")

# Load uncompressed PLY (auto-detects format)
gstensor = GSTensor.load("scene.ply", device="cuda")
print(f"Loaded {len(gstensor):,} Gaussians on GPU")
```

---

### `GSTensor.from_arrays(means, scales, quats, opacities, sh0, shN=None, format='auto', sh_degree=None, sh0_format=SH0_SH, device=None, dtype=None)`

Create GSTensor from individual tensors with format preset.

Convenient factory method for creating GSTensor from external tensors with automatic format detection or explicit format presets.

**Parameters:**
- `means` (torch.Tensor): (N, 3) - Gaussian centers
- `scales` (torch.Tensor): (N, 3) - Scale parameters
- `quats` (torch.Tensor): (N, 4) - Rotation quaternions
- `opacities` (torch.Tensor): (N,) - Opacity values
- `sh0` (torch.Tensor): (N, 3) - DC spherical harmonics
- `shN` (torch.Tensor, optional): (N, K, 3) - Higher-order SH coefficients
- `format` (str): Format preset - "auto" (detect), "ply" (log/logit), "linear" or "rasterizer" (linear)
- `sh_degree` (int, optional): SH degree (0-3) - auto-detected from shN if None
- `sh0_format` (DataFormat): Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH
- `device` (str | torch.device, optional): Target device - inferred from tensors if None
- `dtype` (torch.dtype, optional): Target dtype - inferred from tensors if None

**Returns:**
- `GSTensor`: Object with specified format

**Example:**
```python
from gsply import GSTensor
import torch

# Auto-detect format from values
gstensor = GSTensor.from_arrays(means, scales, quats, opacities, sh0, device="cuda")

# Explicit PLY format (log-scales, logit-opacities)
gstensor = GSTensor.from_arrays(means, scales, quats, opacities, sh0, format="ply", device="cuda")

# Explicit linear format (for rasterizer)
gstensor = GSTensor.from_arrays(means, scales, quats, opacities, sh0, format="linear", device="cuda")
```

---

### `GSTensor.from_dict(data_dict, format='auto', sh_degree=None, sh0_format=SH0_SH, device='cuda', dtype=None)`

Create GSTensor from dictionary with format preset.

Convenient factory method for creating GSTensor from a dictionary with automatic format detection or explicit format presets.

**Parameters:**
- `data_dict` (dict): Dictionary with keys: means, scales, quats, opacities, sh0, shN (optional)
- `format` (str): Format preset - "auto" (detect), "ply" (log/logit), "linear" or "rasterizer" (linear)
- `sh_degree` (int, optional): SH degree (0-3) - auto-detected from shN if None
- `sh0_format` (DataFormat): Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH
- `device` (str | torch.device): Target device (default "cuda")
- `dtype` (torch.dtype, optional): Target dtype - inferred from tensors if None

**Returns:**
- `GSTensor`: Object with specified format

**Example:**
```python
from gsply import GSTensor

# From dictionary with auto-detection
gstensor = GSTensor.from_dict({
    "means": means, "scales": scales, "quats": quats,
    "opacities": opacities, "sh0": sh0, "shN": shN
}, device="cuda")

# Explicit PLY format
gstensor = GSTensor.from_dict(data_dict, format="ply", device="cuda")

# Explicit linear format
gstensor = GSTensor.from_dict(data_dict, format="linear", device="cuda")
```

---

### `gstensor.save(file_path, compressed=True)`

Save GSTensor to PLY file.

Convenience method for saving GSTensor. Uses GPU compression when compressed=True, otherwise converts to GSData and saves uncompressed.

**Parameters:**
- `file_path` (str | Path): Output PLY file path
- `compressed` (bool): If True, use GPU compression (default True). If False, convert to GSData and save uncompressed.

**Performance:**
- Compressed: 5-20x faster compression than CPU Numba
- Uncompressed: Converts to GSData first (CPU transfer)

**Example:**
```python
from gsply import GSTensor

gstensor = GSTensor.from_gsdata(data, device="cuda")

# Save compressed (default, uses GPU compression)
gstensor.save("output.compressed.ply")

# Save uncompressed (converts to GSData first)
gstensor.save("output.ply", compressed=False)
```

---

### `gstensor.save_compressed(file_path)`

Save GSTensor to compressed PLY file using GPU compression.

Convenience alias for `gstensor.save(file_path, compressed=True)`.

**Parameters:**
- `file_path` (str | Path): Output file path

**Performance:**
- 5-20x faster compression than CPU Numba
- GPU reduction for chunk bounds (instant)
- Minimal CPU-GPU data transfer

**Format:**
- PlayCanvas compressed PLY format
- 3.8-14.5x compression ratio
- 256 Gaussians per chunk with quantization

**Example:**
```python
from gsply import GSTensor

gstensor = GSTensor.from_gsdata(data, device="cuda")
gstensor.save_compressed("output.ply_compressed")
# File is ~14x smaller than uncompressed
```

---

### `GSTensor.from_gsdata(data, device='cuda', dtype=torch.float32, requires_grad=False)`

Convert `GSData` to `GSTensor`.

**Parameters:**
- `data` (GSData): Input Gaussian data
- `device` (str | torch.device): Target device ('cuda', 'cpu', or torch.device)
- `dtype` (torch.dtype): Target dtype (default: torch.float32)
- `requires_grad` (bool): Enable gradient tracking (default: False)

**Returns:**
- `GSTensor`: GPU-accelerated tensor container

**Example:**
```python
# Fast path (uses _base if available)
gstensor = GSTensor.from_gsdata(data, device='cuda')

# For training
gstensor = GSTensor.from_gsdata(data, device='cuda', requires_grad=True)

# Half precision for memory savings
gstensor = GSTensor.from_gsdata(data, device='cuda', dtype=torch.float16)
```

---

### `gstensor.to_gsdata()`

Convert `GSTensor` back to `GSData` (CPU NumPy).

**Returns:**
- `GSData`: CPU NumPy container

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')
# ... GPU operations ...
data_cpu = gstensor.to_gsdata()  # Back to NumPy
```

---

### Format Conversion: Linear ↔ PLY Format (GSTensor)

GSTensor provides the same format conversion methods as GSData for consistency:

**PLY Format** (used in `.ply` files):
- Scales: log-space (`log(scale)`)
- Opacities: logit-space (`logit(opacity)`)

**Linear Format** (easier for computation/visualization):
- Scales: linear values (e.g., `0.1`, `0.5`, `1.0`)
- Opacities: linear values in range `[0, 1]`

---

### `gstensor.normalize(inplace=True)` / `gstensor.to_ply_format(inplace=True)`

Convert **linear scales/opacities → PLY format** (log-scales, logit-opacities).

**Converts:**
- Linear scales → log-scales: `log(scale)`
- Linear opacities → logit-opacities: `logit(opacity)` (uses `torch.logit()` with `eps=1e-4`)

**When to use:** When you have linear data and need to save to PLY format.

**Note:** Uses PyTorch's `torch.logit()` and `torch.sigmoid()` internally for GPU acceleration. Behavior matches `GSData.normalize()` for consistency.

**Parameters:**
- `inplace` (bool): If True, modify this object in-place (default). If False, return new object

**Returns:**
- `GSTensor`: GSTensor object (self if inplace=True, new object otherwise)

**Example:**
```python
from gsply import GSTensor, plywrite_gpu
import torch

# Create GSTensor with linear scales and opacities
gstensor = GSTensor(
    means=torch.randn(100, 3),
    scales=torch.rand(100, 3) * 0.1 + 0.01,  # Linear scales: [0.01, 0.11]
    quats=torch.randn(100, 4),
    opacities=torch.rand(100) * 0.8 + 0.1,  # Linear opacities: [0.1, 0.9]
    sh0=torch.randn(100, 3),
    shN=None
)

# Convert to PLY format in-place (modifies gstensor)
gstensor.normalize()  # or: gstensor.normalize(inplace=True) or gstensor.to_ply_format()
# Now ready to save
plywrite_gpu("output.ply", gstensor)

# Or create a copy if you need to keep original
ply_tensor = gstensor.normalize(inplace=False)
```

---

### `gstensor.denormalize(inplace=True)` / `gstensor.from_ply_format(inplace=True)` / `gstensor.to_linear(inplace=True)`

Convert **PLY format → linear scales/opacities** (log-scales → linear, logit-opacities → linear).

**Converts:**
- Log-scales → linear scales: `exp(log_scale)`
- Logit-opacities → linear opacities: `sigmoid(logit)` (uses `torch.sigmoid()`)

**When to use:** When you load PLY files and need linear values for computations or visualization.

**Note:** Uses PyTorch's `torch.sigmoid()` internally for GPU acceleration. Behavior matches `GSData.denormalize()` for consistency.

**Parameters:**
- `inplace` (bool): If True, modify this object in-place (default). If False, return new object

**Returns:**
- `GSTensor`: GSTensor object (self if inplace=True, new object otherwise)

**Example:**
```python
from gsply import plyread_gpu

# Load PLY file directly to GPU (contains log-scales and logit-opacities)
gstensor = plyread_gpu("scene.ply", device='cuda')

# Convert to linear format in-place (modifies gstensor)
gstensor.denormalize()  # or: gstensor.denormalize(inplace=True) or gstensor.from_ply_format() or gstensor.to_linear()

# Now scales and opacities are in linear space
print(f"Linear opacity range: [{gstensor.opacities.min():.3f}, {gstensor.opacities.max():.3f}]")
# Output: Linear opacity range: [0.000, 1.000]

# Easier to work with linear values
high_opacity = gstensor[gstensor.opacities > 0.5]  # Filter by linear opacity

# Or create a copy if you need to keep PLY format
linear_tensor = gstensor.denormalize(inplace=False)
```

---

### `gstensor.to_rgb(inplace=True)`

Convert sh0 from spherical harmonics (SH) format to RGB color format (GPU-accelerated).

**Converts:**
- SH DC coefficients → RGB colors: `rgb = sh0 * SH_C0 + 0.5`

**When to use:** When you need RGB colors in [0, 1] range for visualization or color manipulation on GPU.

**Note:** Uses PyTorch in-place operations (`mul_()`, `add_()`) for GPU acceleration. Modifies tensor in-place for efficiency.

**Parameters:**
- `inplace` (bool): If True, modify this object in-place (default). If False, return new object

**Returns:**
- `GSTensor`: GSTensor object (self if inplace=True, new object otherwise)

**Example:**
```python
from gsply import plyread_gpu

# Load PLY file to GPU (sh0 is in SH format)
gstensor = plyread_gpu("scene.ply")

# Convert to RGB format in-place
gstensor.to_rgb()  # or: gstensor.to_rgb(inplace=True)

# Now sh0 contains RGB colors [0, 1]
print(f"RGB color range: [{gstensor.sh0.min():.3f}, {gstensor.sh0.max():.3f}]")

# Modify colors in RGB space (GPU-accelerated)
gstensor.sh0.mul_(1.5).clamp_(0, 1)  # Make brighter, clamp to valid range

# Convert back to SH format if needed
gstensor.to_sh()

# Or create a copy if you need to keep SH format
rgb_tensor = gstensor.to_rgb(inplace=False)
```

---

### `gstensor.to_sh(inplace=True)`

Convert sh0 from RGB color format to spherical harmonics (SH) format (GPU-accelerated).

**Converts:**
- RGB colors → SH DC coefficients: `sh0 = (rgb - 0.5) / SH_C0`

**When to use:** When you have RGB colors and need to convert back to SH format for PLY file compatibility.

**Note:** Uses PyTorch in-place operations (`sub_()`, `div_()`) for GPU acceleration. Modifies tensor in-place for efficiency.

**Parameters:**
- `inplace` (bool): If True, modify this object in-place (default). If False, return new object

**Returns:**
- `GSTensor`: GSTensor object (self if inplace=True, new object otherwise)

**Example:**
```python
from gsply import GSTensor, plywrite_gpu
import torch

# Create GSTensor with RGB colors
rgb_colors = torch.rand(1000, 3, device="cuda")
gstensor = GSTensor(
    means=torch.randn(1000, 3, device="cuda"),
    scales=torch.ones(1000, 3, device="cuda") * 0.01,
    quats=torch.tile(torch.tensor([1, 0, 0, 0], device="cuda"), (1000, 1)),
    opacities=torch.ones(1000, device="cuda") * 0.5,
    sh0=rgb_colors,  # RGB colors
    shN=None,
)

# Convert to SH format in-place
gstensor.to_sh()  # or: gstensor.to_sh(inplace=True)

# Now sh0 contains SH DC coefficients
# Ready to save to PLY file
plywrite_gpu("output.ply", gstensor)

# Or create a copy if you need to keep RGB format
sh_tensor = gstensor.to_sh(inplace=False)
```

---

### `gstensor.to(device=None, dtype=None)`

Move tensors to different device and/or dtype.

**Parameters:**
- `device` (str | torch.device, optional): Target device
- `dtype` (torch.dtype, optional): Target dtype

**Returns:**
- `GSTensor`: New GSTensor on target device/dtype

**Example:**
```python
gstensor_gpu = gstensor.to('cuda')
gstensor_half = gstensor.to(dtype=torch.float16)
gstensor_gpu_half = gstensor.to('cuda', dtype=torch.float16)
```

---

### `gstensor.consolidate()`

Create `_base` tensor for 25x faster slicing.

**Returns:**
- `GSTensor`: New GSTensor with `_base` tensor

**Example:**
```python
# Consolidate for faster slicing
gstensor = gstensor.consolidate()

# Boolean masking is now 25x faster
mask = gstensor.opacities > 0.5
subset = gstensor[mask]  # Fast with _base
```

---

### `gstensor.clone()`

Create independent deep copy.

**Returns:**
- `GSTensor`: Cloned GSTensor

**Example:**
```python
gstensor_copy = gstensor.clone()
gstensor_copy.means[0] = 0  # Doesn't affect original
```

---

### `gstensor.cpu()`

Move tensors to CPU.

Shorthand for `gstensor.to('cpu')`.

**Returns:**
- `GSTensor`: GSTensor on CPU

**Example:**
```python
gstensor_gpu = GSTensor.from_gsdata(data, device='cuda')
gstensor_cpu = gstensor_gpu.cpu()  # Now on CPU
```

---

### `gstensor.cuda(device=None)`

Move tensors to GPU.

Shorthand for `gstensor.to('cuda')`.

**Parameters:**
- `device` (int | None): GPU device index (default: None = cuda:0)

**Returns:**
- `GSTensor`: GSTensor on GPU

**Example:**
```python
gstensor_gpu = gstensor.cuda()  # Move to cuda:0
gstensor_gpu1 = gstensor.cuda(1)  # Move to cuda:1
```

---

### `gstensor.half()`, `gstensor.float()`, `gstensor.double()`

Convert tensor precision.

Convenience methods for dtype conversion:
- `half()` - Convert to `torch.float16`
- `float()` - Convert to `torch.float32`
- `double()` - Convert to `torch.float64`

**Returns:**
- `GSTensor`: GSTensor with new dtype

**Example:**
```python
# Half precision for memory savings (2x less VRAM)
gstensor_fp16 = gstensor.half()

# Back to full precision
gstensor_fp32 = gstensor_fp16.float()

# Double precision for high accuracy
gstensor_fp64 = gstensor.double()
```

---

### `gstensor.unpack(include_shN=True)`

Unpack GSTensor into tuple of individual tensors.

Identical to `GSData.unpack()` but returns PyTorch tensors instead of NumPy arrays.

**Parameters:**
- `include_shN` (bool): If True, include shN in output (default: True)

**Returns:**
- If `include_shN=True`: `(means, scales, quats, opacities, sh0, shN)`
- If `include_shN=False`: `(means, scales, quats, opacities, sh0)`

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Full unpacking for rendering
means, scales, quats, opacities, sh0, shN = gstensor.unpack()
rendered = render_gaussians(means, scales, quats, opacities, sh0, shN)

# Without higher-order SH
means, scales, quats, opacities, sh0 = gstensor.unpack(include_shN=False)
```

---

### `gstensor.to_dict()`

Convert GSTensor to dictionary for keyword argument unpacking.

Identical to `GSData.to_dict()` but returns PyTorch tensors instead of NumPy arrays.

**Returns:**
- Dictionary with keys: `means`, `scales`, `quats`, `opacities`, `sh0`, `shN`

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Dictionary unpacking
props = gstensor.to_dict()
rendered = render_gaussians(**props)
```

---

### `gstensor[index]`

Slice GSTensor using standard Python indexing.

Supports integers, slices, boolean masks, and fancy indexing. Returns views when possible (zero-copy on GPU).

**Indexing Modes:**
- Integer: `gstensor[0]` - Returns tuple of tensors
- Slice: `gstensor[100:200]` - Returns new GSTensor with subset
- Step: `gstensor[::10]` - Returns every 10th Gaussian
- Boolean mask: `gstensor[mask]` - Filter by boolean tensor
- Fancy: `gstensor[[0, 10, 20]]` - Select specific indices

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')

# Single Gaussian (returns tuple)
means, scales, quats, opacities, sh0, shN, masks = gstensor[0]

# Slice (returns GSTensor view - zero memory cost)
subset = gstensor[100:200]

# Boolean mask (returns GSTensor)
high_opacity = gstensor[gstensor.opacities > 0.5]

# Step slicing (returns GSTensor)
every_10th = gstensor[::10]
```

---

### `len(gstensor)`

Get number of Gaussians.

**Returns:**
- `int`: Number of Gaussians (equivalent to `gstensor.means.shape[0]`)

**Example:**
```python
gstensor = GSTensor.from_gsdata(data, device='cuda')
print(f"Processing {len(gstensor)} Gaussians on GPU")
```

---

### `gstensor.device` (property)

Get current device of tensors.

**Returns:**
- `torch.device`: Current device (e.g., `torch.device('cuda:0')` or `torch.device('cpu')`)

**Example:**
```python
print(f"Tensors are on {gstensor.device}")
if gstensor.device.type == 'cuda':
    print(f"Using GPU {gstensor.device.index}")
```

---

### `gstensor.dtype` (property)

Get current dtype of tensors.

**Returns:**
- `torch.dtype`: Current dtype (e.g., `torch.float32`, `torch.float16`)

**Example:**
```python
print(f"Using precision: {gstensor.dtype}")
```

---

### `gstensor.get_sh_degree()`

Get spherical harmonic degree from data shape.

**Returns:**
- `int`: SH degree (0-3)

**Example:**
```python
sh_degree = gstensor.get_sh_degree()
print(f"Data has SH degree {sh_degree}")
```

---

### `gstensor.has_high_order_sh()`

Check if data has higher-order spherical harmonics.

**Returns:**
- `bool`: True if SH degree > 0

**Example:**
```python
if gstensor.has_high_order_sh():
    print("Has higher-order SH coefficients")
else:
    print("Only DC component (SH0)")
```

---

## Complete Workflow Examples

### Training Workflow

```python
import gsply
from gsply import GSTensor
import torch

# Load from disk
data = gsply.plyread("scene.ply")  # Has _base -> fast GPU transfer

# Transfer to GPU (11x faster with _base)
gstensor = GSTensor.from_gsdata(data, device='cuda', requires_grad=True)

# Training loop
optimizer = torch.optim.Adam([gstensor.means, gstensor.scales], lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()

    # Unpack for rendering (cleaner API)
    means, scales, quats, opacities, sh0, shN = gstensor.unpack()
    loss = render_gaussians(means, scales, quats, opacities, sh0)

    loss.backward()
    optimizer.step()

# Save optimized results
optimized_data = gstensor.to_gsdata()
gsply.plywrite("optimized.ply", optimized_data.means, optimized_data.scales,
               optimized_data.quats, optimized_data.opacities,
               optimized_data.sh0, optimized_data.shN)
```

### Inference Workflow

```python
import gsply
from gsply import GSTensor
import torch

# Load scene
data = gsply.plyread("scene.ply")

# Transfer to GPU (inference mode, no gradients)
gstensor = GSTensor.from_gsdata(data, device='cuda', requires_grad=False)

# Filter Gaussians by opacity threshold
high_opacity_mask = gstensor.opacities > 0.5
filtered = gstensor[high_opacity_mask]

# Render filtered scene with unpacking
with torch.no_grad():
    means, scales, quats, opacities, sh0, shN = filtered.unpack()
    rendered = render_gaussians(means, scales, quats, opacities, sh0)

# Save filtered version
filtered_data = filtered.to_gsdata()
gsply.plywrite("filtered.ply", filtered_data.means, filtered_data.scales,
               filtered_data.quats, filtered_data.opacities,
               filtered_data.sh0, filtered_data.shN)
```
