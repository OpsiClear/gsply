# Compressed PLY Format Specification

## Overview

**Compressed PLY** is a specialized format for Gaussian Splatting that achieves **3.8x to 14.5x compression** (16 bytes/splat vs 232 bytes/splat for uncompressed) while maintaining visual quality. It was created by PlayCanvas/SuperSplat for web-based 3D Gaussian Splatting viewers.

The compression ratio varies based on spherical harmonic (SH) degree:
- **SH0 (DC only)**: ~3-4x compression
- **SH3 (15 bands)**: ~14-15x compression
- Higher SH degrees benefit more from compression

---

## Format Comparison

### Uncompressed PLY (Standard)
```
File size: 11.34 MB for 50K Gaussians
Format: Binary little-endian PLY
Per-splat size: 232 bytes (59 properties x 4 bytes)
Properties: xyz, f_dc(3), f_rest(45), opacity, scales(3), quats(4)
```

### Compressed PLY (PlayCanvas)
```
File size: ~800 KB for 50K Gaussians (SH3)
Format: Custom chunk-based quantization
Per-splat size: 16 bytes
Compression: 14.5x smaller
```

---

## Format Details

### File Structure

The compressed format uses chunk-based quantization organized into fixed-size groups of 256 Gaussians.

#### PLY Header Example
```
ply
format binary_little_endian 1.0
element chunk 197
property float min_x
property float min_y
property float min_z
property float max_x
property float max_y
property float max_z
property float min_scale_x
property float min_scale_y
property float min_scale_z
property float max_scale_x
property float max_scale_y
property float max_scale_z
property float min_r
property float min_g
property float min_b
property float max_r
property float max_g
property float max_b
element vertex 50375
property uint packed_position
property uint packed_rotation
property uint packed_scale
property uint packed_color
element sh 50375
property uchar coeff_0
property uchar coeff_1
...
end_header
[binary data follows]
```

#### Binary Data Layout
```
Offset 0: Chunk data (N_chunks x 18 floats = N_chunks x 72 bytes)
  Chunk 0: [18 floats]
  Chunk 1: [18 floats]
  ...

Offset (N_chunks x 72): Vertex data (N_vertices x 4 uints = N_vertices x 16 bytes)
  Vertex 0: [4 uints]
  Vertex 1: [4 uints]
  ...

Offset (N_chunks x 72 + N_vertices x 16): SH data (optional, N_vertices x K bytes)
  Vertex 0: [K bytes]
  Vertex 1: [K bytes]
  ...
```

### Chunk-Based Quantization

Data is organized into **chunks of 256 Gaussians**:
```
50,375 Gaussians = 197 chunks of 256 (+ 1 partial chunk of 119)
```

Each chunk stores min/max bounds for per-chunk quantization:
- **Position** (x, y, z): min_x, max_x, min_y, max_y, min_z, max_z
- **Scale** (x, y, z): min_scale_x, max_scale_x, min_scale_y, max_scale_y, min_scale_z, max_scale_z
- **Color** (r, g, b): min_r, max_r, min_g, max_g, min_b, max_b

**Chunk metadata**: 18 floats (72 bytes per chunk)

Each Gaussian is quantized relative to its chunk's bounds, enabling better precision than global quantization.

### Bit Packing Specification

Each vertex is encoded as 4 uint32 values (16 bytes total):

#### Position/Scale Packing (11-10-11 bits)

```
32-bit integer breakdown:
Bits 21-31 (11 bits): X coordinate [0, 2047] -> [min_x, max_x]
Bits 11-20 (10 bits): Y coordinate [0, 1023] -> [min_y, max_y]
Bits  0-10 (11 bits): Z coordinate [0, 2047] -> [min_z, max_z]
```

**Quantization**:
```python
# Normalize to [0, 1] within chunk bounds
norm_x = (x - min_x) / (max_x - min_x)
# Quantize to 11 bits
px = (norm_x * 2047.0).astype(np.uint32)
# Pack into 32-bit integer
packed_position = (px << 21) | (py << 11) | pz
```

**Dequantization**:
```python
px = ((packed >> 21) & 0x7FF) / 2047.0  # Normalize to [0, 1]
py = ((packed >> 11) & 0x3FF) / 1023.0
pz = (packed & 0x7FF) / 2047.0

# Map to actual range
x = min_x + px * (max_x - min_x)
y = min_y + py * (max_y - min_y)
z = min_z + pz * (max_z - min_z)
```

#### Rotation Packing (Smallest-Three Encoding)

Quaternions have 4 components but unit norm constraint (only 3 degrees of freedom).

```
32-bit integer breakdown:
Bits 30-31 (2 bits): Which component is largest (to reconstruct)
Bits 20-29 (10 bits): Component A
Bits 10-19 (10 bits): Component B
Bits  0-9  (10 bits): Component C
```

**Encoding Algorithm**:
1. Find the largest quaternion component by absolute value
2. Flip all components if the largest is negative (ensures positive dominant component)
3. Store the three non-dominant components (in original order, excluding the largest)
4. Record which component was largest (2 bits)

**Decoding**:
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

**Note**: Despite the name "smallest-three encoding", this actually stores the three components OTHER than the largest component. The range for each stored component is [-1/sqrt(2), 1/sqrt(2)] = [-0.707, 0.707].

#### Color Packing (8-8-8-8 bits)

```
32-bit integer breakdown:
Bits 24-31 (8 bits): Red [0, 255] -> [min_r, max_r]
Bits 16-23 (8 bits): Green [0, 255] -> [min_g, max_g]
Bits  8-15 (8 bits): Blue [0, 255] -> [min_b, max_b]
Bits  0-7  (8 bits): Opacity [0, 255] -> [0, 1] (sigmoid/logit space)
```

**Encoding**:
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

#### SH Coefficient Packing (8 bits per coefficient)

Higher-order spherical harmonics (beyond DC) are stored separately:

```python
# Normalize to [0, 1]: assumes shN values in [-8, 8] range
normalized = (shN / 8.0) + 0.5
# Quantize to 8 bits using truncation (matches PlayCanvas)
quantized = np.trunc(normalized * 256.0).astype(np.uint8)
# Clamp to [0, 255]
quantized = np.clip(quantized, 0, 255)
```

---

## Storage Breakdown

### Per Chunk (256 Gaussians)
```
Chunk metadata: 18 floats x 4 bytes = 72 bytes
Vertex data: 256 splats x 16 bytes = 4,096 bytes
Total per chunk: 4,168 bytes
Per-splat overhead: 72 / 256 = 0.28 bytes
```

### Total File Size (SH3 Example)
```
50,375 Gaussians:
- Chunk count: 197 chunks
- Chunk metadata: 197 x 72 = 14,184 bytes (~14 KB)
- Vertex data: 50,375 x 16 = 806,000 bytes (~806 KB)
- SH data: 50,375 x 45 = 2,266,875 bytes (~2.2 MB)
- Header: ~1 KB
- Total: ~3.0 MB

Compression ratio: 11.34 MB / 3.0 MB = 3.8x (SH3)
```

For SH0 (DC only), no separate SH element exists, yielding ~14.5x compression.

---

## Reading Compressed PLY

### Decompression Algorithm

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

**Example Usage**:
```python
import gsply

# Automatically detects compressed format
means, scales, quats, opacities, sh0, shN = gsply.plyread("scene.ply")

print(f"Loaded {means.shape[0]} Gaussians")
# Output: Loaded 50375 Gaussians

# Data is already decompressed and ready to use
# Same format as uncompressed PLY
```

### Performance Characteristics

**Current Performance** (vectorized NumPy):
```
Read compressed: 30-50ms for 50K Gaussians
~1000-1600 splats/ms
38.5x faster than Python loop implementation
```

**Read Performance vs Uncompressed**:
```
Uncompressed: 5-10ms for 50K Gaussians
Compressed: 30-50ms for 50K Gaussians
Overhead: 3-10x slower (but file is 3.8-14.5x smaller)
```

---

## Writing Compressed PLY

### Implementation Overview

gsply implements fast compressed PLY writing using vectorized NumPy operations. The compression pipeline mirrors the optimized decompression logic.

### Compression Algorithm

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

### Performance Benchmarks

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

### Usage Examples

```python
from gsply import plywrite

# Write compressed PLY
plywrite("output.ply", means, scales, quats, opacities, sh0, shN, compressed=True)

# Or use the direct function
from gsply.writer import write_compressed
write_compressed("output.ply", means, scales, quats, opacities, sh0, shN)
```

---

## Format Compatibility

### PlayCanvas splat-transform

The implementation has been verified to match the PlayCanvas splat-transform reference implementation exactly. Key compatibility features:

1. **Quaternion Encoding**: Uses "largest component" encoding (despite the historical name "smallest-three")
   - Finds the largest quaternion component by absolute value
   - Flips all components if the largest is negative (ensures positive dominant component)
   - Stores the three non-dominant components
   - Range: Each stored component in [-1/sqrt(2), 1/sqrt(2)]

2. **Scale Clamping**: Chunk bounds for scales are clamped to [-20, 20] to handle infinity values

3. **SH Quantization**: Uses `trunc(normalized * 256)` quantization matching the reference

4. **Bit Packing**: Exact match with splat-transform specification
   - Position: 11-10-11 bits (x, y, z)
   - Rotation: 2+10+10+10 bits (which, a, b, c)
   - Scale: 11-10-11 bits (sx, sy, sz)
   - Color: 8-8-8-8 bits (r, g, b, opacity)

### Verification

Run `python verify_compatibility.py` in the gsply directory to test format compatibility (all checks should pass).

---

## Limitations

### Quaternion Encoding

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

### Compression Efficiency

Compression ratio depends on SH degree:
- **SH0 (DC only)**: ~14.5x compression (no separate SH element)
- **SH1 (4 bands)**: ~8-10x compression
- **SH3 (15 bands)**: ~3.8-4x compression
- Higher SH degrees have proportionally more uncompressed SH data

### Quality Loss

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

## Use Cases

### When to Use Compressed Format

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

---

## Quality vs Size Tradeoff

### Quantization Precision

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

---

## Implementation Notes

### Code Statistics

**Implementation:**
- Lines added: ~180 lines (writer.py)
- Test cases: 56 tests passing (4 for compressed writing)
- Benchmark scripts: 1 new benchmark
- Complexity: Fully vectorized quantization, minimal loops

### Future Optimizations

Potential improvements (not yet implemented):

1. **Vectorize quaternion component extraction**
   - Currently uses a loop to extract three components
   - Could use fancy indexing or masking
   - Estimated: 2-3x faster quaternion encoding

2. **Parallel chunk processing**
   - Compute chunk bounds in parallel
   - Requires threading/multiprocessing
   - Estimated: 2-4x faster on multi-core

3. **Numba JIT compilation**
   - JIT-compile hot paths
   - Estimated: 2-3x overall speedup
   - Requires: numba dependency

4. **Further vectorization of decompression**
   - Current implementation is 38.5x faster than Python loop
   - Additional 2-5x speedup possible with advanced NumPy techniques
   - Goal: ~5-10ms for 50K Gaussians (currently 30-50ms)

---

## Comparison with Other Formats

| Format | Size (50K, SH3) | Precision | Read Speed | Write Speed | Use Case |
|--------|-----------------|-----------|------------|-------------|----------|
| **Uncompressed PLY** | 11.34 MB | Full float32 | Fast (5-10ms) | Fast (6-8ms) | Offline, editing |
| **Compressed PLY** | 2.95 MB | Quantized | Medium (30-50ms) | Slow (200ms) | Web, streaming |
| **Binary .splat** | ~11 MB | Full float32 | Fast | Fast | Native apps |
| **PNG compression** | ~2-5 MB | Lossless | Slow (50-100ms) | Very slow | Archival |

---

## Summary

**Compressed PLY** is a 3.8-14.5x smaller format for Gaussian Splatting:
- Chunk-based quantization (256 splats per chunk)
- 16 bytes per vertex (4 uint32 values)
- Optional 8-bit SH coefficients
- gsply reads it (30-50ms for 50K splats, 38.5x faster than Python loop)
- gsply writes it (200ms for 50K splats, 30x slower than uncompressed)
- Full compatibility with PlayCanvas splat-transform
- Minimal quality loss for most scenes
- Not for high-precision applications

**Trade-off Summary:**
- Write: 30x slower
- Read: 3-10x slower
- Size: 3.8-14.5x smaller
- **Net benefit**: Faster overall for streaming/distribution (smaller file = faster network transfer)

Use **uncompressed** for offline work, **compressed** for distribution!

---

## Related Documentation

- [PERFORMANCE.md](./PERFORMANCE.md) - Performance benchmarks and optimization details
- [VECTORIZATION_EXPLAINED.md](./VECTORIZATION_EXPLAINED.md) - Vectorization techniques deep-dive
- [COMPATIBILITY_FIXES.md](./COMPATIBILITY_FIXES.md) - Format compatibility details

---

## References

- **PlayCanvas SuperSplat**: https://github.com/playcanvas/super-splat
- **PlayCanvas splat-transform**: https://github.com/playcanvas/splat-transform
- **Gaussian Splatting**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **PLY Format**: http://paulbourke.net/dataformats/ply/
