# gsply User Guide

Complete guide to gsply - Ultra-fast Gaussian Splatting PLY I/O library for Python

**Version:** 0.2.9
**Last Updated:** 2025-11-24

---

## Table of Contents

- [Quick Start](#quick-start)
- [Performance](#performance)
  - [Current Benchmarks](#current-benchmarks)
  - [Key Results](#key-results)
  - [Library Comparisons](#library-comparisons)
  - [Real-World Use Cases](#real-world-use-cases)
- [Format Specification](#format-specification)
  - [Uncompressed PLY Format](#uncompressed-ply-format)
  - [Compressed PLY Format](#compressed-ply-format)
  - [Format Comparison](#format-comparison)
  - [Reading Compressed PLY](#reading-compressed-ply)
  - [Writing Compressed PLY](#writing-compressed-ply)
  - [Use Cases](#use-cases)
- [Advanced Topics](#advanced-topics)
  - [Vectorization Explained](#vectorization-explained)
  - [Optimization Techniques](#optimization-techniques)
  - [Performance Tuning](#performance-tuning)
- [Compatibility](#compatibility)
  - [PlayCanvas splat-transform](#playcanvas-splat-transform)
  - [Format Validation](#format-validation)
  - [Known Limitations](#known-limitations)
- [Appendix](#appendix)
  - [Historical Benchmarks](#historical-benchmarks)
  - [Future Optimizations](#future-optimizations)
  - [Related Documentation](#related-documentation)

---

## Quick Start

### Installation

```bash
# From PyPI (coming soon)
pip install gsply

# From source
git clone https://github.com/OpsiClear/gsply.git
cd gsply
pip install -e .
```

### Basic Usage

**Object-Oriented API (Recommended):**
```python
from gsply import GSData, GSTensor

# Read PLY file (auto-detects format)
data = GSData.load("model.ply")

# Access fields
positions = data.means    # (N, 3) xyz coordinates
colors = data.sh0         # (N, 3) RGB colors
scales = data.scales      # (N, 3) scale parameters

# Save PLY file
data.save("output.ply")  # Uncompressed
data.save("output.ply", compressed=True)  # Compressed (71-74% smaller)

# GPU acceleration (optional)
gstensor = GSTensor.load("model.ply", device='cuda')
gstensor.save("output.compressed.ply")  # GPU compression
```

**Functional API (Alternative):**
```python
from gsply import plyread, plywrite

# Read PLY file (auto-detects format)
data = plyread("model.ply")

# Write uncompressed PLY file
plywrite("output.ply", data)

# Write compressed PLY file (3.8-14.5x smaller)
plywrite("output.ply", data, compressed=True)

# Detect format before reading
from gsply import detect_format
is_compressed, sh_degree = detect_format("model.ply")
print(f"Compressed: {is_compressed}, SH degree: {sh_degree}")
```

### Data Format

All arrays are returned as numpy arrays with the following shapes:

- `means`: (N, 3) - Gaussian centers (x, y, z)
- `scales`: (N, 3) - Log-space scales (PLY format) or linear scales
- `quats`: (N, 4) - Rotations as quaternions (w, x, y, z)
- `opacities`: (N,) - Logit-space opacities (PLY format) or linear opacities
- `sh0`: (N, 3) - DC spherical harmonic coefficients (SH format) or RGB colors
- `shN`: (N, K, 3) or None - Higher-order SH coefficients (K=0 for degree 0)

**Format Conversion:**
- PLY files store scales in log-space and opacities in logit-space
- Use `data.normalize()` to convert linear → PLY format before saving (uses fused kernel, ~8-15x faster)
- Use `data.denormalize()` to convert PLY → linear format after loading (uses fused kernel, ~8-15x faster)
- Use `data.to_rgb()` / `data.to_sh()` to convert between SH and RGB color formats
- Advanced: Use `apply_pre_activations()` / `apply_pre_deactivations()` for direct access to fused kernels

---

## Performance

### Current Benchmarks

All benchmarks measured on Intel i7-10700K with 50K Gaussians (SH degree 3).

#### Uncompressed PLY Read Performance

```
gsply:   5.56ms   [FASTEST - baseline]
plyfile: 18.23ms  [3.3x slower]
Open3D:  43.10ms  [7.7x slower]
```

#### Uncompressed PLY Write Performance (SH3)

```
gsply:   8.72ms   [FASTEST - baseline]
plyfile: 12.18ms  [1.4x slower]
Open3D:  35.69ms  [4.1x slower]
```

#### Uncompressed PLY Write Performance (SH0)

```
gsply:   3.42ms   [FASTEST - baseline]
plyfile: 3.53ms   [essentially tied]
```

#### Compressed PLY Decompression (Vectorized)

```
Original (Python loop): 65.42ms
Vectorized (NumPy):     1.70ms
Speedup: 38.5x faster
```

### Key Results

**Performance Achievements:**
- **Total read speedup from baseline**: 27% faster (7.59ms -> 5.56ms)
- **Total write speedup from baseline**: 34% faster (12.15ms -> 8.72ms for SH3)
- **Compressed decompression speedup**: 38.5x faster
- **vs plyfile (industry standard)**: 3.3x faster reads, 1.4x faster writes
- **vs Open3D**: 7.7x faster reads, 4.1x faster writes
- **Memory efficiency**: Zero additional memory overhead from optimizations

**Quality Metrics:**
- 56/56 tests passing
- 100% API compatibility
- Zero memory overhead
- Comprehensive documentation

### Library Comparisons

#### vs plyfile (Industry Standard)

**Advantages:**
- 3.3x faster reads
- 1.4x faster writes (SH3)
- More consistent performance (lower std dev)
- Compressed format support with real-time performance

**Trade-offs:**
- SH0 writes essentially tied (3.42ms vs 3.53ms)
- Slightly more complex code (but well-documented)

#### vs Open3D

**Advantages:**
- 7.7x faster reads
- 4.1x faster writes (SH3)
- 4.76x faster overall throughput
- Specialized for Gaussian Splatting format

**Trade-offs:**
- More focused scope (Gaussian Splatting PLY only)
- Open3D provides broader 3D processing capabilities

### Real-World Use Cases

#### Use Case 1: Animation Frame Loading (100 frames)

```python
# Loading 100 frames for playback
frames = [gsply.plyread(f"frame_{i:05d}.ply") for i in range(100)]
```

**Performance:**
```
Before: 7.59ms x 100 = 759ms
After:  5.56ms x 100 = 556ms
Improvement: 203ms saved (27% faster)
```

#### Use Case 2: Batch Processing Pipeline (1000 files)

```python
# Read, process, write pipeline
for file in ply_files:
    data = gsply.plyread(file)
    processed = process(data)
    gsply.plywrite(output_file, *processed)
```

**Performance:**
```
Before: (7.59 + 12.15) x 1000 = 19.74 seconds
After:  (5.56 + 8.72) x 1000 = 14.28 seconds
Improvement: 5.46 seconds saved (28% faster)
```

#### Use Case 3: Compressed Streaming (Real-time 4D Gaussian Splatting)

```python
# Real-time decompression and rendering
while streaming:
    compressed_frame = fetch_next_frame()
    gaussians = gsply.plyread(compressed_frame)
    render(gaussians)  # 60+ FPS target
```

**Performance:**
```
Before: 65ms decompression + 10ms render = 75ms = 13 FPS (too slow)
After:  2ms decompression + 10ms render = 12ms = 83 FPS (smooth!)
Improvement: Compressed format now viable for real-time!
```

---

## Format Specification

### Uncompressed PLY Format

Standard binary little-endian PLY format with Gaussian Splatting properties.

#### Format Details

| SH Degree | Properties | Description |
|-----------|-----------|-------------|
| 0 | 14 | xyz, f_dc(3), opacity, scales(3), quats(4) |
| 1 | 23 | + 9 f_rest coefficients |
| 2 | 38 | + 24 f_rest coefficients |
| 3 | 59 | + 45 f_rest coefficients |

**File Size:** ~232 bytes per Gaussian (SH3)
**Example:** 50K Gaussians = 11.34 MB

**Properties:**
- Position (xyz): 3 float32 values
- SH coefficients (f_dc, f_rest): (3 + K*3) float32 values
- Opacity: 1 float32 value
- Scales: 3 float32 values
- Quaternions: 4 float32 values

### Compressed PLY Format

Chunk-based quantized format created by PlayCanvas for web-based 3D Gaussian Splatting viewers.

#### Format Overview

**Compression Ratio:** 3.8x to 14.5x (depends on SH degree)
- SH0 (DC only): ~14-15x compression
- SH3 (15 bands): ~3.8-4x compression
- Higher SH degrees benefit more from compression

**File Size:** ~16 bytes per Gaussian vertex + overhead
**Example:** 50K Gaussians (SH3) = ~3.0 MB (vs 11.34 MB uncompressed)

#### File Structure

```
PLY Header (text)
    - element chunk <N_chunks>
    - element vertex <N_vertices>
    - element sh <N_vertices> (optional)
    - end_header

Binary Data:
    Chunk Metadata: N_chunks x 72 bytes (18 floats per chunk)
    Vertex Data:    N_vertices x 16 bytes (4 uint32 per vertex)
    SH Data:        N_vertices x K bytes (optional, K uint8 per vertex)
```

#### Chunk-Based Quantization

Data is organized into **chunks of 256 Gaussians**:

```
50,375 Gaussians = 197 chunks of 256 (+ 1 partial chunk of 119)
```

Each chunk stores min/max bounds for per-chunk quantization:
- **Position** (x, y, z): min_x, max_x, min_y, max_y, min_z, max_z
- **Scale** (x, y, z): min_scale_x, max_scale_x, min_scale_y, max_scale_y, min_scale_z, max_scale_z
- **Color** (r, g, b): min_r, max_r, min_g, max_g, min_b, max_b

**Chunk metadata:** 18 floats (72 bytes per chunk)

**Per-splat overhead:** 72 / 256 = 0.28 bytes

#### Bit Packing Specification

Each vertex is encoded as 4 uint32 values (16 bytes total):

**Position/Scale Packing (11-10-11 bits)**

```
32-bit integer breakdown:
Bits 21-31 (11 bits): X coordinate [0, 2047] -> [min_x, max_x]
Bits 11-20 (10 bits): Y coordinate [0, 1023] -> [min_y, max_y]
Bits  0-10 (11 bits): Z coordinate [0, 2047] -> [min_z, max_z]
```

**Quantization:**
```python
# Normalize to [0, 1] within chunk bounds
norm_x = (x - min_x) / (max_x - min_x)
# Quantize to 11 bits
px = (norm_x * 2047.0).astype(np.uint32)
# Pack into 32-bit integer
packed_position = (px << 21) | (py << 11) | pz
```

**Dequantization:**
```python
px = ((packed >> 21) & 0x7FF) / 2047.0  # Normalize to [0, 1]
py = ((packed >> 11) & 0x3FF) / 1023.0
pz = (packed & 0x7FF) / 2047.0

# Map to actual range
x = min_x + px * (max_x - min_x)
y = min_y + py * (max_y - min_y)
z = min_z + pz * (max_z - min_z)
```

**Rotation Packing (Smallest-Three Encoding)**

Quaternions have 4 components but unit norm constraint (only 3 degrees of freedom).

```
32-bit integer breakdown:
Bits 30-31 (2 bits): Which component is largest (to reconstruct)
Bits 20-29 (10 bits): Component A
Bits 10-19 (10 bits): Component B
Bits  0-9  (10 bits): Component C
```

**Encoding Algorithm:**
1. Find the largest quaternion component by absolute value
2. Flip all components if the largest is negative (ensures positive dominant component)
3. Store the three non-dominant components (in original order, excluding the largest)
4. Record which component was largest (2 bits)

**Decoding:**
```python
# Extract which component was largest
which = (packed >> 30) & 0x3
# Extract three stored components
a = ((packed >> 20) & 0x3FF) / 511.5 - 1.0
b = ((packed >> 10) & 0x3FF) / 511.5 - 1.0
c = (packed & 0x3FF) / 511.5 - 1.0

# Reconstruct the largest component from unit norm constraint
largest = sqrt(1.0 - (a*a + b*b + c*c))

# Reassemble quaternion (insert largest at correct position)
quat = [a, b, c]
quat.insert(which, largest)
```

**Note:** Despite the name "smallest-three encoding", this actually stores the three components OTHER than the largest component. The range for each stored component is [-1/sqrt(2), 1/sqrt(2)] = [-0.707, 0.707].

**Color Packing (8-8-8-8 bits)**

```
32-bit integer breakdown:
Bits 24-31 (8 bits): Red [0, 255] -> [min_r, max_r]
Bits 16-23 (8 bits): Green [0, 255] -> [min_g, max_g]
Bits  8-15 (8 bits): Blue [0, 255] -> [min_b, max_b]
Bits  0-7  (8 bits): Opacity [0, 255] -> [0, 1] (sigmoid/logit space)
```

**Encoding:**
```python
# Convert SH0 to RGB
color = sh0 * 0.28209 + 0.5
# Normalize within chunk bounds
norm_r = (color[:, 0] - min_r) / (max_r - min_r)
# Quantize to 8 bits
cr = (norm_r * 255.0).astype(np.uint32)
# Convert opacity to sigmoid space
co = 1.0 / (1.0 + np.exp(-opacity))
co_quantized = (co * 255.0).astype(np.uint32)
# Pack
packed_color = (cr << 24) | (cg << 16) | (cb << 8) | co_quantized
```

**SH Coefficient Packing (8 bits per coefficient)**

Higher-order spherical harmonics (beyond DC) are stored separately:

```python
# Normalize to [0, 1]: assumes shN values in [-8, 8] range
normalized = (shN / 8.0) + 0.5
# Quantize to 8 bits using truncation (matches PlayCanvas)
quantized = np.trunc(normalized * 256.0).astype(np.uint8)
# Clamp to [0, 255]
quantized = np.clip(quantized, 0, 255)
```

### Format Comparison

| Format | Size (50K, SH3) | Precision | Read Speed | Write Speed | Use Case |
|--------|-----------------|-----------|------------|-------------|----------|
| **Uncompressed PLY** | 11.34 MB | Full float32 | Fast (5-10ms) | Fast (6-8ms) | Offline, editing |
| **Compressed PLY** | 2.95 MB | Quantized | Medium (30-50ms) | Slow (200ms) | Web, streaming |
| **Binary .splat** | ~11 MB | Full float32 | Fast | Fast | Native apps |
| **PNG compression** | ~2-5 MB | Lossless | Slow (50-100ms) | Very slow | Archival |

### Reading Compressed PLY

#### Decompression Algorithm

gsply implements fully vectorized decompression using NumPy operations:

1. **Read chunk metadata**: Load all chunk bounds (18 floats x N_chunks)
2. **Read packed vertices**: Load all 4 uint32 values per vertex
3. **Assign vertices to chunks**: `chunk_indices = vertex_idx // 256`
4. **Vectorized dequantization**:
   - Extract bit fields using bit shifts and masks
   - Normalize to [0, 1] range
   - Scale to chunk bounds: `value = min_val + normalized * (max_val - min_val)`
5. **Quaternion reconstruction**: Compute largest component from unit norm constraint
6. **SH coefficient decompression** (if present): Scale from [0, 255] back to [-8, 8]

#### Usage Example

```python
import gsply

# Automatically detects compressed format
means, scales, quats, opacities, sh0, shN = gsply.plyread("scene.ply")

print(f"Loaded {means.shape[0]} Gaussians")
# Output: Loaded 50375 Gaussians

# Data is already decompressed and ready to use
# Same format as uncompressed PLY
```

#### Performance Characteristics

**Current Performance** (vectorized NumPy):
```
Read compressed: 30-50ms for 50K Gaussians
~1000-1600 splats/ms
38.5x faster than Python loop implementation
```

**Read Performance vs Uncompressed:**
```
Uncompressed: 5-10ms for 50K Gaussians
Compressed: 30-50ms for 50K Gaussians
Overhead: 3-10x slower (but file is 3.8-14.5x smaller)
```

### Writing Compressed PLY

#### Implementation Overview

gsply implements fast compressed PLY writing using vectorized NumPy operations.

#### Compression Algorithm

The compression algorithm consists of these steps:

1. **Chunk Organization**
   - Divide Gaussians into chunks of 256
   - Last chunk may contain fewer than 256 elements

2. **Chunk Bounds Computation**
   - For each chunk, compute min/max for position, scale, color
   - Scale bounds clamped to [-20, 20] to handle infinity values
   - Store as chunk metadata (18 floats per chunk)

3. **Position Quantization** (11-10-11 bits)
   ```python
   # Normalize to [0, 1] within chunk bounds
   norm_x = (means[:, 0] - min_x[chunk_indices]) / range_x
   # Quantize to 11 bits
   px = (norm_x * 2047.0).astype(np.uint32)
   # Pack into 32-bit integer
   packed_position = (px << 21) | (py << 11) | pz
   ```

4. **Scale Quantization** (11-10-11 bits)
   - Same pattern as position
   - Per-chunk normalization

5. **Color Quantization** (8-8-8-8 bits)
   - Convert SH0 to RGB: `color = sh0 * 0.28209 + 0.5`
   - Normalize within chunk bounds
   - 8 bits per channel
   - Opacity in sigmoid space: `co = 1.0 / (1.0 + exp(-opacity))`

6. **Quaternion Quantization** (smallest-three encoding: 2+10+10+10 bits)
   - Find largest component by absolute value
   - Flip all components if largest is negative
   - Extract three non-largest components (maintaining order)
   - Quantize each to 10 bits
   - Pack with 2-bit flag indicating which component was largest

7. **SH Coefficient Compression** (8-bit per coefficient, optional)
   - Normalize to [0, 1]: `(shN / 8.0) + 0.5`
   - Quantize to 8 bits using truncation: `trunc(normalized * 256)`
   - Clamp to [0, 255] for edge cases

8. **File Writing**
   - Write PLY header with chunk and vertex elements
   - Write chunk metadata (18 floats per chunk)
   - Write packed vertices (4 uint32 per vertex)
   - Write SH coefficients if present (K uint8 per vertex)

#### Performance Benchmarks

**Benchmark Results (50,375 Gaussians, SH degree 3):**

| Format | Write Time | File Size | Relative Speed | Relative Size |
|--------|------------|-----------|----------------|---------------|
| **Uncompressed** | 6.82ms | 11.34 MB | baseline | baseline |
| **Compressed** | 204.50ms | 2.95 MB | 30x slower | 3.8x smaller |

**Key Metrics:**
- **Compression ratio**: 3.8x (SH3) to 14.5x (SH0)
- **Write overhead**: 30x slower than uncompressed
- **Decompression speed**: 38.5x faster than Python loop
- **Net benefit**: Faster overall for streaming/distribution use cases

#### Usage Examples

```python
from gsply import plywrite

# Write compressed PLY
plywrite("output.ply", means, scales, quats, opacities, sh0, shN, compressed=True)

# Or use the direct function
from gsply.writer import write_compressed
write_compressed("output.ply", means, scales, quats, opacities, sh0, shN)
```

### Use Cases

#### When to Use Compressed Format

**Good for:**
- Network streaming (3.8-14.5x less bandwidth)
- Storage-constrained environments
- Distribution of 4D Gaussian Splatting data
- Real-time applications (fast decompression)
- Web/mobile viewers
- Large datasets (1M Gaussians: 232 MB -> 16-60 MB)

**Not ideal for:**
- Single-file workflows (30x write overhead)
- Scenarios requiring perfect round-trip accuracy
- Data with extreme quaternion distributions
- High-precision applications (scientific visualization, measurement, medical imaging)
- Offline processing (training data, editing, quality-critical rendering)
- Fine detail preservation (< 1cm features, high-frequency patterns)

#### Quality vs Size Tradeoff

**Quantization Precision:**

**Position** (11 bits): 2048 levels
- Typical chunk size: 10 meters
- Precision: 10m / 2048 = ~5mm per level
- Good for: Large scenes, architectural visualization

**Rotation** (10 bits per component): 1024 levels
- Quaternion component precision: 1/511.5 (normalized to [-1, 1])
- Good for: Most rotations, minor artifacts on rapid spins

**Color** (8 bits per channel): 256 levels
- Standard RGB precision
- Good for: Most scenes

**Scale** (11 bits): 2048 levels
- Logarithmic scale space
- Good for: Most Gaussian sizes

**SH Coefficients** (8 bits per coefficient): 256 levels
- Range: [-8, 8]
- Precision: ~0.063 per level
- Good for: Most lighting conditions

#### Quality Loss Characteristics

**Minimal quality loss** for:
- Static scenes
- Large objects
- Smooth surfaces
- Moderate color gradients

**Noticeable quality loss** for:
- Fine details (< 5mm)
- Rapid rotations
- High-frequency textures
- Extreme dynamic range

---

## Advanced Topics

### Vectorization Explained

#### The Problem

The original compressed PLY decompression used a **Python loop** that processed vertices **one at a time**:

```python
# Current implementation (reader.py lines 360-393)
for i in range(num_vertices):  # Loop through 50,375 vertices!
    chunk_idx = i // CHUNK_SIZE

    # Unpack one vertex
    px, py, pz = _unpack_111011(int(packed_position[i]))
    qx, qy, qz, qw = _unpack_rotation(int(packed_rotation[i]))
    sx, sy, sz = _unpack_111011(int(packed_scale[i]))
    cr, cg, cb, co = _unpack_8888(int(packed_color[i]))

    # Dequantize one vertex
    means[i, 0] = min_x[chunk_idx] + px * (max_x[chunk_idx] - min_x[chunk_idx])
    means[i, 1] = min_y[chunk_idx] + py * (max_y[chunk_idx] - min_y[chunk_idx])
    means[i, 2] = min_z[chunk_idx] + pz * (max_z[chunk_idx] - min_z[chunk_idx])

    # ... more assignments
```

**Why this is slow:**
- Python loops are ~100x slower than NumPy operations
- 50,375 iterations × Python interpreter overhead
- Function call overhead for each unpack operation
- No CPU vectorization (SIMD)

**Current performance:** 30-50ms for 50K Gaussians

#### The Solution: Vectorization

**Vectorization** means replacing the Python loop with NumPy array operations that process **all vertices at once**:

```python
# Vectorized implementation

# 1. Pre-compute chunk indices for ALL vertices at once
chunk_indices = np.arange(num_vertices, dtype=np.int32) // CHUNK_SIZE

# 2. Unpack ALL positions at once using bitwise operations
px = ((packed_position >> 21) & 0x7FF).astype(np.float32) / 2047.0
py = ((packed_position >> 11) & 0x3FF).astype(np.float32) / 1023.0
pz = (packed_position & 0x7FF).astype(np.float32) / 2047.0

# 3. Dequantize ALL positions at once using advanced indexing
means[:, 0] = min_x[chunk_indices] + px * (max_x[chunk_indices] - min_x[chunk_indices])
means[:, 1] = min_y[chunk_indices] + py * (max_y[chunk_indices] - min_y[chunk_indices])
means[:, 2] = min_z[chunk_indices] + pz * (max_z[chunk_indices] - min_z[chunk_indices])

# Same for scales, colors, etc.
```

**Why this is fast:**
- NumPy operations run in compiled C code
- Single operation on entire arrays (no loop overhead)
- CPU SIMD instructions (process 4-8 values simultaneously)
- Cache-friendly memory access patterns

**Achieved performance:** 1.70ms for 50K Gaussians
**Speedup:** 38.5x faster!

#### Performance Breakdown

**Original Implementation (Python Loop):**

| Operation | Time | % of Total |
|-----------|------|------------|
| Loop overhead | 15ms | 37% |
| Unpack functions | 12ms | 30% |
| Array indexing | 8ms | 20% |
| Dequantization | 5ms | 13% |
| **Total** | **40ms** | **100%** |

**Vectorized Implementation:**

| Operation | Time | % of Total |
|-----------|------|------------|
| Bit unpacking | 1.5ms | 30% |
| Advanced indexing | 2.0ms | 40% |
| Dequantization | 1.0ms | 20% |
| Color processing | 0.5ms | 10% |
| **Total** | **5ms** | **100%** |

**Speedup:** 40ms -> 5ms = **8x faster**

#### CPU SIMD (Bonus Speed)

Modern CPUs have **SIMD** (Single Instruction, Multiple Data) instructions:

**Without SIMD** (scalar):
```
Process 1 value per instruction
4 values = 4 instructions
```

**With SIMD** (vectorized):
```
Process 4-8 values per instruction (AVX2/AVX-512)
4 values = 1 instruction
```

NumPy automatically uses SIMD when possible. Python loops **cannot** use SIMD.

#### Visual Analogy

**Python Loop (Original):**
```
Processing Gaussians:
[G1] -> [process] -> done
[G2] -> [process] -> done
[G3] -> [process] -> done
...
[G50375] -> [process] -> done

Time: ~40ms (one at a time)
```

**Vectorized (Current):**
```
Processing Gaussians:
[G1, G2, G3, ..., G50375] -> [process all] -> done

Time: ~5ms (all at once)
```

#### Real-World Impact

**Use Case: Streaming 4D Gaussian Splatting**

**Before Vectorization:**
```
Decompression: 40ms
Rendering: 10ms
Total: 50ms
FPS: 20 FPS

Result: Stuttery, not real-time
```

**After Vectorization:**
```
Decompression: 5ms
Rendering: 10ms
Total: 15ms
FPS: 66 FPS

Result: Smooth, real-time!
```

### Optimization Techniques

gsply underwent three major optimization phases, progressing from "fast" to "blazing fast" through careful profiling, strategic refactoring, and aggressive vectorization.

#### Phase 1: Memory Allocation Optimization

**Focus:** Eliminate unnecessary memory allocations

**What Was Optimized:**

1. **Removed unnecessary array copies in reader** (`reader.py` lines 130-159)
   - Eliminated 6 `.copy()` calls that were creating redundant allocations
   - Exception: Kept copy for reshape operation where required
   - Rationale: Sliced arrays are returned immediately, parent data goes out of scope

2. **Pre-allocation instead of concatenate in writer** (`writer.py` lines 140-157)
   - Replaced `np.concatenate()` + `astype()` with single pre-allocation
   - Changed from 2 allocations + 2 copies to 1 allocation + direct assignments
   - Reduced intermediate array creation

3. **Use newaxis instead of reshape** (`writer.py` line 95-96)
   - Changed `opacities.reshape(-1, 1)` to `opacities[:, np.newaxis]`
   - Creates view directly without reshape validation overhead
   - More Pythonic and explicit about intent

**Results Achieved:**

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Read (SH3) | 7.59ms | 5.69ms | **25% faster** |
| Write (SH3) | 12.15ms | 10.70ms | **12% faster** |
| Write (SH0) | 4.81ms | 4.81ms | No change |

**Memory Impact:** Reduced allocations by 6 per read, 2 per write

#### Phase 2: Validation & Type Checking

**Focus:** Eliminate redundant operations in hot paths

**What Was Optimized:**

1. **dtype checks before conversion** (`writer.py` lines 78-90)
   - Added conditional checks: only convert if dtype != float32
   - Used `copy=False` flag to avoid allocation when possible
   - Common case optimization: data already float32 from reading or GPU

2. **Batch header validation** (`reader.py` lines 114-117)
   - Replaced loop with direct list comparison
   - Cleaner code, microseconds faster
   - More Pythonic

3. **Optional validation flag** (`writer.py`)
   - Added `validate=True` parameter to skip assertions for trusted data
   - 5-10% speedup when disabled
   - Power user feature for internal pipelines

**Results Achieved:**

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Read (SH3) | 5.69ms | 5.54ms | **+2% faster** |
| Write (SH3) | 10.70ms | 7.98ms | **+24% faster** |
| Write (SH0) | 4.81ms | 3.19ms | **+26% faster** |

**Why Such Big Write Improvement?**
- dtype optimization: Eliminates 6 unnecessary `astype()` calls (~1.8ms saved)
- Validation overhead: 5 shape assertions removed (~0.4ms saved)
- Compound effects: Better cache behavior from fewer allocations

**Common Case Scenario:** Read PLY (data is float32) -> Write immediately = zero dtype conversions needed

#### Phase 3: Vectorized Decompression

**Focus:** Replace Python loops with NumPy array operations

**What Was Optimized:**

Replaced entire Python loop (lines 360-413 in `reader.py`) with vectorized NumPy operations for compressed PLY decompression:

1. **Position unpacking & dequantization** (11-10-11 bit encoding)
   - Vectorized bitwise operations: `>>`, `&`, bit masking
   - Advanced indexing for chunk-based dequantization
   - 55x speedup for position operations alone

2. **Scale unpacking & dequantization** (11-10-11 bit encoding)
   - Same pattern as position
   - Parallel processing of all 50K vertices

3. **Color unpacking & dequantization** (8-8-8-8 bit encoding)
   - 4-channel color extraction
   - Conversion to SH0 coefficients
   - Vectorized linear interpolation

4. **Opacity conversion** (logit space)
   - Replaced conditional logic with `np.where()`
   - Handles edge cases: co = 0, co = 1, 0 < co < 1
   - Nested `np.where` for multi-case branching

5. **Quaternion unpacking** (smallest-three encoding)
   - Most complex operation: 2-bit flag determines which component to reconstruct
   - Used nested `np.where()` to handle 4 cases
   - Vectorized normalization and fourth component calculation

6. **SH coefficient decompression**
   - Vectorized special case handling (val=0, val=255)
   - Flattened array processing with reshape
   - 50x speedup over nested loops

**Vectorization Techniques:**

**NumPy Operations Used:**
- Bitwise operations: `>>`, `&` for unpacking
- Type casting: `.astype(np.float32)` for precision
- Advanced indexing: `array[indices]` for chunk mapping
- Boolean operations: `&`, `|` for element-wise logic
- Conditional masking: `np.where()` for if/else branching
- Mathematical operations: `np.sqrt()`, `np.log()`, `np.maximum()`

**CPU SIMD Utilization:**
- AVX2 (256-bit): Process 8 float32 values per instruction
- AVX-512 (512-bit): Process 16 float32 values per instruction
- NumPy automatically uses SIMD when possible
- Result: Not only eliminate Python overhead, but also get 8-16x parallelism from SIMD

**Results Achieved:**

**Micro-Benchmark (Position Operations Only):**
```
Python Loop:       62.97ms
Vectorized NumPy:   1.14ms
Speedup: 55.3x faster
```

**Full Decompression Performance:**
```
Original (Python loop): 65.42ms
Vectorized (NumPy):     1.70ms
Speedup: 38.5x faster
```

**Real-World Impact - Streaming 4D Gaussian Splatting:**
```
Before Vectorization:
  Decompression: 40ms
  Rendering:     10ms
  Total:         50ms (20 FPS - too slow for real-time)

After Vectorization:
  Decompression: 1-3ms
  Rendering:     10ms
  Total:         11-13ms (75-90 FPS - smooth real-time!)
```

**File Size Benefits:**
- Compressed format: 14.5x smaller files (11.34 MB -> 0.8 MB)
- With vectorization: Smaller + faster to download + fast to decompress
- **Conclusion:** Compressed format now viable for real-time streaming

#### Cross-Cutting Principles

- Profile before optimizing
- Maintain test coverage (56/56 tests passing throughout)
- Preserve API compatibility
- Document rationale for each change
- Verify output equivalence with plyfile

### Performance Tuning

#### Optimization Approach

The key insight was that the entire decompression pipeline could be expressed as a series of vectorized transformations:

1. **Pre-compute chunk indices once:** `chunk_indices = np.arange(num_vertices) // CHUNK_SIZE`
2. **Parallel unpacking:** Use bitwise operations on entire arrays
3. **Advanced indexing:** Map per-chunk parameters to all vertices simultaneously
4. **Conditional logic:** Use `np.where()` for branching without loops

#### Memory Management

**Phase 1 Techniques:**
- Eliminate redundant `.copy()` calls
- Pre-allocate arrays with correct dtype
- Use views instead of reshape where possible
- Direct slice assignment over concatenation

#### Conditional Optimization

**Phase 2 Techniques:**
- Check before convert (dtype checks)
- Batch validation instead of loops
- Optional validation for trusted data
- Avoid redundant operations in hot paths

#### Vectorization Patterns

**Phase 3 Techniques:**
- Replace Python loops with NumPy operations
- Bitwise operations for unpacking
- Advanced indexing for broadcasting
- Conditional masking for branching logic
- SIMD-friendly operation patterns

#### Fused Kernels for Format Conversion (v0.2.7+)

**Phase 4: Format Conversion Optimization**

Format conversion (`normalize()` and `denormalize()`) now uses fused Numba kernels for optimal performance:

**Before (Sequential Operations):**
```python
# Separate operations - multiple passes through data
scales = np.log(np.clip(scales, min_scale, None))
opacities = logit(np.clip(opacities, min_opacity, max_opacity))
# Quaternion normalization in separate pass
quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
```

**After (Fused Kernel):**
```python
# Single fused kernel - one pass through data
apply_pre_deactivations(data, min_scale=1e-9, min_opacity=1e-4, max_opacity=0.9999)
# Processes scales and opacities together in parallel
```

**Performance Benefits:**
- **~8-15x faster** than sequential operations
- Single-pass processing reduces memory overhead
- Parallel Numba JIT compilation with `prange`
- Improved cache locality (processes related data together)
- Quaternion normalization included in activation kernel

**Implementation Details:**
- Uses Numba `@njit(parallel=True, fastmath=True, cache=True, nogil=True)`
- Processes all Gaussians in parallel using `prange`
- Clamping and transformations done in single kernel
- Zero intermediate allocations

**When to Use:**
- `normalize()` / `denormalize()` - Automatic (uses fused kernels internally)
- `apply_pre_activations()` / `apply_pre_deactivations()` - Direct access for fine-grained control

---

## Compatibility

### PlayCanvas splat-transform

The implementation has been verified to match the PlayCanvas splat-transform reference implementation exactly. Key compatibility features:

#### Quaternion Encoding

**Critical Fix:** Uses "largest component" encoding (despite the historical name "smallest-three")

- Finds the largest quaternion component by absolute value
- Flips all components if the largest is negative (ensures positive dominant component)
- Stores the three non-dominant components
- Range: Each stored component in [-1/sqrt(2), 1/sqrt(2)]

**Implementation:**
```python
# Correct (matches splat-transform):
quats_normalized = quats / np.linalg.norm(quats, axis=1, keepdims=True)
abs_quats = np.abs(quats_normalized)
largest_idx = np.argmax(abs_quats, axis=1)  # Find LARGEST

# Added sign flipping (matches splat-transform compressed-chunk.ts:133-138):
for i in range(num_gaussians):
    if quats_normalized[i, largest_idx[i]] < 0:
        quats_normalized[i] = -quats_normalized[i]
```

#### Scale Clamping

Chunk bounds for scales are clamped to [-20, 20] to handle infinity values:

```python
chunk_bounds[chunk_idx, 6] = np.clip(np.min(chunk_scales[:, 0]), -20, 20)
```

**Reference:** splat-transform `compressed-chunk.ts:88-95`

#### SH Quantization

Uses `trunc(normalized * 256)` quantization matching the reference:

```python
packed_sh = np.clip(np.trunc(sh_normalized * 256.0), 0, 255).astype(np.uint8)
```

**Reference:** splat-transform `write-compressed-ply.ts:85-86`

#### Bit Packing

Exact match with splat-transform specification:

| Component | Bits | Layout | Match |
|-----------|------|--------|-------|
| Position | 32 | 11-10-11 (x, y, z) | [OK] |
| Rotation | 32 | 2+10+10+10 (which, a, b, c) | [OK] |
| Scale | 32 | 11-10-11 (sx, sy, sz) | [OK] |
| Color | 32 | 8-8-8-8 (r, g, b, opacity) | [OK] |
| SH coeffs | 8/coeff | uint8 per coefficient | [OK] |

#### Chunk Metadata

18 floats per chunk (256 Gaussians):
- Position bounds: min_x, min_y, min_z, max_x, max_y, max_z (6 floats)
- Scale bounds: min_sx, min_sy, min_sz, max_sx, max_sy, max_sz (6 floats, clamped to [-20, 20])
- Color bounds: min_r, min_g, min_b, max_r, max_g, max_b (6 floats)

### Format Validation

Run `python verify_compatibility.py` in the gsply directory to test format compatibility.

**All checks should pass:**
- [OK] Quaternion encoding correct (max error: 0.001256)
- [OK] Scale bounds properly clamped to [-20, 20]
- [OK] SH coefficient quantization correct (max error: 0.015625)
- [OK] File structure matches specification

### Known Limitations

#### Quaternion Encoding

The smallest-three encoding has inherent limitations:

- Can only represent quaternion components in range [-0.707, 0.707] = [-1/sqrt(2), 1/sqrt(2)]
- Values outside this range are clamped, causing rotation errors
- Occurs when one quaternion component is very small and others are large
- This is a limitation of the format specification, not the implementation

**Example:**
```python
# This quaternion will have clamping errors:
q = [0.95, 0.05, 0.1, 0.2]  # Component 0 exceeds 0.707

# Well-behaved quaternions work fine:
q = [0.5, 0.5, 0.5, 0.5]  # All components within range
```

#### Compression Efficiency

Compression ratio depends on SH degree:
- **SH0 (DC only):** ~14.5x compression (no separate SH element)
- **SH1 (4 bands):** ~8-10x compression
- **SH3 (15 bands):** ~3.8-4x compression
- Higher SH degrees have proportionally more uncompressed SH data

#### Compatibility Status

**Status:** FULLY COMPATIBLE

gsply compressed PLY format is now 100% compatible with PlayCanvas splat-transform. Files can be:
- Written by gsply, read by splat-transform [OK]
- Written by splat-transform, read by gsply [OK]
- Interchanged with any tool using the splat-transform specification [OK]

---

## Appendix

### Historical Benchmarks

#### Baseline (Before Optimizations)

**Read Performance:**
```
gsply:   7.59ms
plyfile: 19.05ms  (2.5x slower than gsply)
Open3D:  43.77ms  (5.8x slower than gsply)
```

**Write Performance (SH3):**
```
gsply:   12.15ms
plyfile: 14.00ms  (1.15x slower than gsply)
Open3D:  35.63ms  (2.9x slower than gsply)
```

#### After Phase 1 (Memory Optimizations)

**Read Performance:**
```
gsply:   5.69ms  (25% faster than baseline)
plyfile: 17.77ms (3.1x slower than gsply)
```

**Write Performance (SH3):**
```
gsply:   10.70ms (12% faster than baseline)
plyfile: 12.89ms (1.20x slower than gsply)
```

#### After Phase 2 (Validation Optimizations)

**Read Performance:**
```
gsply:   5.54ms  (27% faster than baseline, +2% from Phase 1)
plyfile: 17.96ms (3.2x slower than gsply)
```

**Write Performance (SH3):**
```
gsply:   7.98ms  (34% faster than baseline, +24% from Phase 1)
plyfile: 12.57ms (1.57x slower than gsply)
```

#### After Phase 3 (Vectorized Decompression)

**Uncompressed Performance:** See "Current Benchmarks" section (minor refinements)

**Compressed Performance:**
```
Before vectorization: 65.42ms
After vectorization:  1.70ms
Improvement: 38.5x faster
```

#### After Phase 4 (Fused Format Conversion Kernels - v0.2.7+)

**Format Conversion Performance:**
```
Before (sequential): normalize() ~2.5ms, denormalize() ~3.2ms
After (fused kernels): normalize() ~0.2ms, denormalize() ~0.3ms
Improvement: ~8-15x faster
```

**Key Benefits:**
- Single-pass processing reduces memory overhead
- Parallel Numba JIT compilation
- Improved cache locality
- Quaternion normalization included in activation kernel

### Future Optimizations

The following optimizations have been identified but not yet implemented. They are listed in order of estimated value/effort ratio.

#### 1. Optimize Header Reading (Medium Priority)

**Expected Impact:** 5-10% read improvement
**Effort:** 30 minutes
**Current bottleneck:** Line-by-line reading with many I/O syscalls

**Approach:** Read header in single chunk instead of line-by-line
```python
# Read header in one chunk (typical headers are <5KB)
header_chunk = f.read(8192)
header_end = header_chunk.find(b'end_header\n')

if header_end != -1:
    # Fast path: header fits in one read
    header_lines = header_chunk[:header_end].decode('ascii').split('\n')
    data_offset = header_end + 11  # len('end_header\n')
```

**Benefits:** Reduces I/O syscalls from ~30 to 1, better for network filesystems

#### 2. Cache Header Templates (Low Priority)

**Expected Impact:** 2-3% write improvement
**Effort:** 1 hour
**Current bottleneck:** String concatenation on every write

**Approach:** Pre-compute header strings for each SH degree
```python
# Pre-computed header templates (module level)
_HEADER_TEMPLATES = {
    0: "ply\nformat binary_little_endian 1.0\nelement vertex {}\n" + ...,
    1: ...,  # 23 properties
    2: ...,  # 38 properties
    3: ...,  # 59 properties
}

def write_uncompressed(...):
    sh_degree = 0 if shN is None else shN.shape[1] // 3
    header = _HEADER_TEMPLATES[sh_degree].format(num_gaussians).encode('ascii')
```

**Benefits:** Eliminates header string construction overhead

#### 3. Memory-Mapped Reading (Specialized)

**Expected Impact:** 20-30% for large files >100MB
**Effort:** 1-2 hours
**Target use case:** Very large datasets

**Approach:** Use `np.memmap` for zero-copy reading
```python
def read_uncompressed(file_path, use_mmap=False):
    if use_mmap and file_size > 100_000_000:
        # Memory-mapped reading (zero-copy)
        data = np.memmap(file_path, dtype=np.float32, mode='r',
                        offset=data_offset,
                        shape=(vertex_count, property_count))
        # Return views directly (no copy needed)
        means = data[:, 0:3]
```

**Benefits:** Near-zero memory overhead, enables working with huge datasets

#### 4. Parallel Batch Reading (Advanced)

**Expected Impact:** 4x throughput for batches
**Effort:** 1 hour
**Target use case:** Loading animation sequences

**Approach:** ThreadPoolExecutor for loading multiple files
```python
def plyread_batch(file_paths, num_workers=4):
    """Read multiple PLY files in parallel."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(plyread, file_paths))
    return results
```

**Benefits:** Near-linear scaling with CPU cores for batch workflows

#### 5. Numba JIT Compilation (Advanced)

**Expected Impact:** 2-3x additional for compressed
**Effort:** 2-3 hours
**Requires:** numba dependency
**Target use case:** Compressed format heavy users

**Approach:** JIT-compile hot paths
```python
from numba import jit

@jit(nopython=True, fastmath=True)
def _unpack_and_dequantize_vectorized(packed_data, chunk_indices, min_vals, max_vals):
    # Vectorized unpacking with JIT compilation
    # 10-20x faster than pure Python
```

**Benefits:** Significant speedup for compressed format, especially quaternion unpacking

### Related Documentation

**Developer Documentation:**
- [CONTRIBUTING.md](../.github/CONTRIBUTING.md) - Contribution guidelines
- [BUILD.md](./BUILD.md) - Build system and distribution guide
- [CI_CD_SETUP.md](./CI_CD_SETUP.md) - CI/CD pipeline documentation
- [RELEASE_NOTES.md](./RELEASE_NOTES.md) - Release notes and version history

**User Documentation:**
- [README.md](../README.md) - Library overview and quick start guide
- [docs/archive/](./archive/) - Historical documentation from development phases

**External References:**
- [PlayCanvas SuperSplat](https://github.com/playcanvas/super-splat)
- [PlayCanvas splat-transform](https://github.com/playcanvas/splat-transform)
- [Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [PLY Format Specification](http://paulbourke.net/dataformats/ply/)

---

## Summary

gsply is the **fastest Gaussian Splatting PLY I/O library** for Python, achieving:

**Performance:**
- 3.3x faster reads than plyfile
- 1.4x faster writes than plyfile
- 7.7x faster reads than Open3D
- 38.5x faster compressed decompression
- Enables real-time streaming (75-90 FPS)

**Quality:**
- 56/56 tests passing
- Zero memory overhead
- 100% API compatibility
- Comprehensive documentation

**Format Support:**
- Uncompressed PLY (standard binary little-endian)
- Compressed PLY (PlayCanvas-compatible, 3.8-14.5x smaller)
- Auto-detection of format and SH degree
- Full compatibility with PlayCanvas splat-transform

**Use Cases:**
- Animation frame loading (27% faster)
- Batch processing pipelines (28% faster)
- Real-time compressed streaming (enables 60+ FPS)
- Web/mobile distribution (3.8-14.5x smaller files)

The optimization journey demonstrates that careful profiling, strategic refactoring, and aggressive vectorization can deliver massive performance improvements while maintaining code quality and backward compatibility.

**Status:** Ready for v0.1.0 release
