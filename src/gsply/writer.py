"""Writing functions for Gaussian splatting PLY files.

This module provides ultra-fast writing of Gaussian splatting PLY files
in uncompressed format, with compressed format support planned.

API Examples:
    >>> from gsply import plywrite
    >>> plywrite("output.ply", means, scales, quats, opacities, sh0, shN)

    >>> # Or use format-specific writers
    >>> from gsply.writer import write_uncompressed
    >>> write_uncompressed("output.ply", means, scales, quats, opacities, sh0, shN)

Performance:
    - Write uncompressed: 3-7ms for 50K Gaussians (7-17M Gaussians/sec)
    - Write compressed: 2-11ms for 50K Gaussians (4-25M Gaussians/sec)
"""

import logging
from functools import lru_cache
from pathlib import Path

import numba
import numpy as np

# Import numba for JIT optimization
from numba import jit

from gsply.formats import CHUNK_SIZE, CHUNK_SIZE_SHIFT, SH_C0
from gsply.gsdata import GSData

logger = logging.getLogger(__name__)


# ======================================================================================
# I/O BUFFER SIZE CONSTANTS
# ======================================================================================

# Buffer sizes for optimized I/O performance
_LARGE_BUFFER_SIZE = 2 * 1024 * 1024  # 2MB buffer for large files
_SMALL_BUFFER_SIZE = 1 * 1024 * 1024  # 1MB buffer for small files
_LARGE_FILE_THRESHOLD = 10_000_000  # 10MB threshold for buffer size selection


# ======================================================================================
# PRE-COMPUTED HEADER TEMPLATES (Optimization)
# ======================================================================================

# Pre-computed header template for SH degree 0 (14 properties)
_HEADER_TEMPLATE_SH0 = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {num_gaussians}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property float f_dc_0\n"
    "property float f_dc_1\n"
    "property float f_dc_2\n"
    "property float opacity\n"
    "property float scale_0\n"
    "property float scale_1\n"
    "property float scale_2\n"
    "property float rot_0\n"
    "property float rot_1\n"
    "property float rot_2\n"
    "property float rot_3\n"
    "end_header\n"
)

# Pre-computed f_rest property lines for SH degrees 1-3
_F_REST_PROPERTIES = {
    9: "\n".join(f"property float f_rest_{i}" for i in range(9)) + "\n",
    24: "\n".join(f"property float f_rest_{i}" for i in range(24)) + "\n",
    45: "\n".join(f"property float f_rest_{i}" for i in range(45)) + "\n",
}


@lru_cache(maxsize=32)
def _build_header_fast(num_gaussians: int, num_sh_rest: int | None) -> bytes:
    """Generate PLY header using pre-computed templates (with LRU cache).

    This optimization pre-computes header strings for common SH degrees (0-3),
    avoiding dynamic string building in loops. Provides 3-5% speedup for writes.

    Args:
        num_gaussians: Number of Gaussians
        num_sh_rest: Number of higher-order SH coefficients (None for SH0)

    Returns:
        Header bytes ready to write
    """
    if num_sh_rest is None:
        # SH degree 0: use pre-computed template
        return _HEADER_TEMPLATE_SH0.format(num_gaussians=num_gaussians).encode("ascii")

    if num_sh_rest in _F_REST_PROPERTIES:
        # SH degrees 1-3: use pre-computed f_rest properties
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_gaussians}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float f_dc_0\n"
            "property float f_dc_1\n"
            "property float f_dc_2\n" + _F_REST_PROPERTIES[num_sh_rest] + "property float opacity\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "property float rot_0\n"
            "property float rot_1\n"
            "property float rot_2\n"
            "property float rot_3\n"
            "end_header\n"
        )
        return header.encode("ascii")

    # Fallback for arbitrary SH degrees (rare)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_gaussians}",
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(num_sh_rest):
        header_lines.append(f"property float f_rest_{i}")
    header_lines.extend(
        [
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )
    return ("\n".join(header_lines) + "\n").encode("ascii")


# ======================================================================================
# JIT-COMPILED COMPRESSION FUNCTIONS
# ======================================================================================


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _pack_positions_jit(sorted_means, chunk_indices, min_x, min_y, min_z, max_x, max_y, max_z):
    """JIT-compiled position quantization and packing (11-10-11 bits) with parallel processing.

    Args:
        sorted_means: (N, 3) float32 array of positions
        chunk_indices: int32 array of chunk indices for each vertex
        min_x, min_y, min_z: chunk minimum bounds
        max_x, max_y, max_z: chunk maximum bounds

    Returns:
        packed: (N,) uint32 array of packed positions
    """
    n = len(sorted_means)
    packed = np.zeros(n, dtype=np.uint32)

    for i in numba.prange(n):
        chunk_idx = chunk_indices[i]

        # Compute ranges (handle zero range)
        range_x = max_x[chunk_idx] - min_x[chunk_idx]
        range_y = max_y[chunk_idx] - min_y[chunk_idx]
        range_z = max_z[chunk_idx] - min_z[chunk_idx]

        if range_x == 0.0:
            range_x = 1.0
        if range_y == 0.0:
            range_y = 1.0
        if range_z == 0.0:
            range_z = 1.0

        # Normalize to [0, 1]
        norm_x = (sorted_means[i, 0] - min_x[chunk_idx]) / range_x
        norm_y = (sorted_means[i, 1] - min_y[chunk_idx]) / range_y
        norm_z = (sorted_means[i, 2] - min_z[chunk_idx]) / range_z

        # Clamp
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        norm_z = max(0.0, min(1.0, norm_z))

        # Quantize
        px = np.uint32(norm_x * 2047.0)
        py = np.uint32(norm_y * 1023.0)
        pz = np.uint32(norm_z * 2047.0)

        # Pack (11-10-11 bits)
        packed[i] = (px << 21) | (py << 11) | pz

    return packed


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _pack_scales_jit(sorted_scales, chunk_indices, min_sx, min_sy, min_sz, max_sx, max_sy, max_sz):
    """JIT-compiled scale quantization and packing (11-10-11 bits) with parallel processing.

    Args:
        sorted_scales: (N, 3) float32 array of scales
        chunk_indices: int32 array of chunk indices for each vertex
        min_sx, min_sy, min_sz: chunk minimum scale bounds
        max_sx, max_sy, max_sz: chunk maximum scale bounds

    Returns:
        packed: (N,) uint32 array of packed scales
    """
    n = len(sorted_scales)
    packed = np.zeros(n, dtype=np.uint32)

    for i in numba.prange(n):
        chunk_idx = chunk_indices[i]

        # Compute ranges (handle zero range)
        range_sx = max_sx[chunk_idx] - min_sx[chunk_idx]
        range_sy = max_sy[chunk_idx] - min_sy[chunk_idx]
        range_sz = max_sz[chunk_idx] - min_sz[chunk_idx]

        if range_sx == 0.0:
            range_sx = 1.0
        if range_sy == 0.0:
            range_sy = 1.0
        if range_sz == 0.0:
            range_sz = 1.0

        # Normalize to [0, 1]
        norm_sx = (sorted_scales[i, 0] - min_sx[chunk_idx]) / range_sx
        norm_sy = (sorted_scales[i, 1] - min_sy[chunk_idx]) / range_sy
        norm_sz = (sorted_scales[i, 2] - min_sz[chunk_idx]) / range_sz

        # Clamp
        norm_sx = max(0.0, min(1.0, norm_sx))
        norm_sy = max(0.0, min(1.0, norm_sy))
        norm_sz = max(0.0, min(1.0, norm_sz))

        # Quantize
        sx = np.uint32(norm_sx * 2047.0)
        sy = np.uint32(norm_sy * 1023.0)
        sz = np.uint32(norm_sz * 2047.0)

        # Pack (11-10-11 bits)
        packed[i] = (sx << 21) | (sy << 11) | sz

    return packed


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _pack_colors_jit(
    sorted_color_rgb,
    sorted_opacities,
    chunk_indices,
    min_r,
    min_g,
    min_b,
    max_r,
    max_g,
    max_b,
):
    """JIT-compiled color and opacity quantization and packing (8-8-8-8 bits) with parallel processing.

    Args:
        sorted_color_rgb: (N, 3) float32 array of pre-computed RGB colors (SH0 * SH_C0 + 0.5)
        sorted_opacities: (N,) float32 array of opacities (logit space)
        chunk_indices: int32 array of chunk indices for each vertex
        min_r, min_g, min_b: chunk minimum color bounds
        max_r, max_g, max_b: chunk maximum color bounds

    Returns:
        packed: (N,) uint32 array of packed colors
    """
    n = len(sorted_color_rgb)
    packed = np.zeros(n, dtype=np.uint32)

    for i in numba.prange(n):
        chunk_idx = chunk_indices[i]

        # Use pre-computed RGB colors
        color_r = sorted_color_rgb[i, 0]
        color_g = sorted_color_rgb[i, 1]
        color_b = sorted_color_rgb[i, 2]

        # Compute ranges (handle zero range)
        range_r = max_r[chunk_idx] - min_r[chunk_idx]
        range_g = max_g[chunk_idx] - min_g[chunk_idx]
        range_b = max_b[chunk_idx] - min_b[chunk_idx]

        if range_r == 0.0:
            range_r = 1.0
        if range_g == 0.0:
            range_g = 1.0
        if range_b == 0.0:
            range_b = 1.0

        # Normalize to [0, 1]
        norm_r = (color_r - min_r[chunk_idx]) / range_r
        norm_g = (color_g - min_g[chunk_idx]) / range_g
        norm_b = (color_b - min_b[chunk_idx]) / range_b

        # Clamp
        norm_r = max(0.0, min(1.0, norm_r))
        norm_g = max(0.0, min(1.0, norm_g))
        norm_b = max(0.0, min(1.0, norm_b))

        # Quantize colors
        cr = np.uint32(norm_r * 255.0)
        cg = np.uint32(norm_g * 255.0)
        cb = np.uint32(norm_b * 255.0)

        # Opacity: logit to linear
        opacity_linear = 1.0 / (1.0 + np.exp(-sorted_opacities[i]))
        opacity_linear = max(0.0, min(1.0, opacity_linear))
        co = np.uint32(opacity_linear * 255.0)

        # Pack (8-8-8-8 bits)
        packed[i] = (cr << 24) | (cg << 16) | (cb << 8) | co

    return packed


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _pack_quaternions_jit(sorted_quats):
    """JIT-compiled quaternion normalization and packing (2+10-10-10 bits, smallest-three) with parallel processing.

    Args:
        sorted_quats: (N, 4) float32 array of quaternions

    Returns:
        packed: (N,) uint32 array of packed quaternions
    """
    n = len(sorted_quats)
    packed = np.zeros(n, dtype=np.uint32)
    norm_factor = np.sqrt(2.0) * 0.5

    for i in numba.prange(n):
        # Normalize quaternion
        quat = sorted_quats[i]
        norm = np.sqrt(
            quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]
        )
        if norm > 0:
            quat = quat / norm

        # Find largest component by absolute value
        abs_vals = np.abs(quat)
        largest_idx = 0
        largest_val = abs_vals[0]
        for j in range(1, 4):
            if abs_vals[j] > largest_val:
                largest_val = abs_vals[j]
                largest_idx = j

        # Flip quaternion if largest component is negative
        if quat[largest_idx] < 0:
            quat = -quat

        # Extract three smaller components
        three_components = np.zeros(3, dtype=np.float32)
        idx = 0
        for j in range(4):
            if j != largest_idx:
                three_components[idx] = quat[j]
                idx += 1

        # Normalize to [0, 1] for quantization
        qa_norm = three_components[0] * norm_factor + 0.5
        qb_norm = three_components[1] * norm_factor + 0.5
        qc_norm = three_components[2] * norm_factor + 0.5

        # Clamp
        qa_norm = max(0.0, min(1.0, qa_norm))
        qb_norm = max(0.0, min(1.0, qb_norm))
        qc_norm = max(0.0, min(1.0, qc_norm))

        # Quantize
        qa_int = np.uint32(qa_norm * 1023.0)
        qb_int = np.uint32(qb_norm * 1023.0)
        qc_int = np.uint32(qc_norm * 1023.0)

        # Pack (2 bits for which + 10+10+10 bits)
        packed[i] = (np.uint32(largest_idx) << 30) | (qa_int << 20) | (qb_int << 10) | qc_int

    return packed


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _compute_chunk_bounds_jit(
    sorted_means, sorted_scales, sorted_color_rgb, chunk_starts, chunk_ends
):
    """JIT-compiled chunk bounds computation (9x faster than Python loop).

    Computes min/max bounds for positions, scales, and colors for each chunk.
    This is the main bottleneck in compressed write (~90ms -> ~10ms).

    Args:
        sorted_means: (N, 3) float32 array of positions
        sorted_scales: (N, 3) float32 array of scales
        sorted_color_rgb: (N, 3) float32 array of pre-computed RGB colors (SH0 * SH_C0 + 0.5)
        chunk_starts: (num_chunks,) int array of chunk start indices
        chunk_ends: (num_chunks,) int array of chunk end indices

    Returns:
        bounds: (num_chunks, 18) float32 array with layout:
            [0:6]   - min_x, min_y, min_z, max_x, max_y, max_z
            [6:12]  - min_scale_x/y/z, max_scale_x/y/z (clamped to [-20,20])
            [12:18] - min_r, min_g, min_b, max_r, max_g, max_b
    """
    num_chunks = len(chunk_starts)
    bounds = np.zeros((num_chunks, 18), dtype=np.float32)

    for chunk_idx in numba.prange(num_chunks):
        start = chunk_starts[chunk_idx]
        end = chunk_ends[chunk_idx]

        if start >= end:  # Empty chunk
            continue

        # Initialize with first element
        bounds[chunk_idx, 0] = sorted_means[start, 0]  # min_x
        bounds[chunk_idx, 1] = sorted_means[start, 1]  # min_y
        bounds[chunk_idx, 2] = sorted_means[start, 2]  # min_z
        bounds[chunk_idx, 3] = sorted_means[start, 0]  # max_x
        bounds[chunk_idx, 4] = sorted_means[start, 1]  # max_y
        bounds[chunk_idx, 5] = sorted_means[start, 2]  # max_z

        bounds[chunk_idx, 6] = sorted_scales[start, 0]  # min_scale_x
        bounds[chunk_idx, 7] = sorted_scales[start, 1]  # min_scale_y
        bounds[chunk_idx, 8] = sorted_scales[start, 2]  # min_scale_z
        bounds[chunk_idx, 9] = sorted_scales[start, 0]  # max_scale_x
        bounds[chunk_idx, 10] = sorted_scales[start, 1]  # max_scale_y
        bounds[chunk_idx, 11] = sorted_scales[start, 2]  # max_scale_z

        # Use pre-computed RGB for first element
        color_r = sorted_color_rgb[start, 0]
        color_g = sorted_color_rgb[start, 1]
        color_b = sorted_color_rgb[start, 2]

        bounds[chunk_idx, 12] = color_r  # min_r
        bounds[chunk_idx, 13] = color_g  # min_g
        bounds[chunk_idx, 14] = color_b  # min_b
        bounds[chunk_idx, 15] = color_r  # max_r
        bounds[chunk_idx, 16] = color_g  # max_g
        bounds[chunk_idx, 17] = color_b  # max_b

        # Process remaining elements in chunk
        for i in range(start + 1, end):
            # Position bounds
            bounds[chunk_idx, 0] = min(bounds[chunk_idx, 0], sorted_means[i, 0])
            bounds[chunk_idx, 1] = min(bounds[chunk_idx, 1], sorted_means[i, 1])
            bounds[chunk_idx, 2] = min(bounds[chunk_idx, 2], sorted_means[i, 2])
            bounds[chunk_idx, 3] = max(bounds[chunk_idx, 3], sorted_means[i, 0])
            bounds[chunk_idx, 4] = max(bounds[chunk_idx, 4], sorted_means[i, 1])
            bounds[chunk_idx, 5] = max(bounds[chunk_idx, 5], sorted_means[i, 2])

            # Scale bounds
            bounds[chunk_idx, 6] = min(bounds[chunk_idx, 6], sorted_scales[i, 0])
            bounds[chunk_idx, 7] = min(bounds[chunk_idx, 7], sorted_scales[i, 1])
            bounds[chunk_idx, 8] = min(bounds[chunk_idx, 8], sorted_scales[i, 2])
            bounds[chunk_idx, 9] = max(bounds[chunk_idx, 9], sorted_scales[i, 0])
            bounds[chunk_idx, 10] = max(bounds[chunk_idx, 10], sorted_scales[i, 1])
            bounds[chunk_idx, 11] = max(bounds[chunk_idx, 11], sorted_scales[i, 2])

            # Color bounds (already converted to RGB)
            color_r = sorted_color_rgb[i, 0]
            color_g = sorted_color_rgb[i, 1]
            color_b = sorted_color_rgb[i, 2]

            bounds[chunk_idx, 12] = min(bounds[chunk_idx, 12], color_r)
            bounds[chunk_idx, 13] = min(bounds[chunk_idx, 13], color_g)
            bounds[chunk_idx, 14] = min(bounds[chunk_idx, 14], color_b)
            bounds[chunk_idx, 15] = max(bounds[chunk_idx, 15], color_r)
            bounds[chunk_idx, 16] = max(bounds[chunk_idx, 16], color_g)
            bounds[chunk_idx, 17] = max(bounds[chunk_idx, 17], color_b)

        # Clamp scale bounds to [-20, 20] (matches splat-transform)
        for j in range(6, 12):
            bounds[chunk_idx, j] = max(-20.0, min(20.0, bounds[chunk_idx, j]))

    return bounds


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _validate_and_normalize_inputs(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    shN: np.ndarray | None,  # noqa: N803
    validate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and normalize input arrays to float32 format.

    Args:
        means: Gaussian centers, shape (N, 3)
        scales: Log scales, shape (N, 3)
        quats: Rotations as quaternions (wxyz), shape (N, 4)
        opacities: Logit opacities, shape (N,)
        sh0: DC spherical harmonics, shape (N, 3)
        shN: Higher-order SH coefficients, shape (N, K, 3) or None
        validate: Whether to validate shapes

    Returns:
        Tuple of normalized arrays (all float32)
    """
    # Ensure all arrays are numpy arrays
    if not isinstance(means, np.ndarray):
        means = np.asarray(means, dtype=np.float32)
    if not isinstance(scales, np.ndarray):
        scales = np.asarray(scales, dtype=np.float32)
    if not isinstance(quats, np.ndarray):
        quats = np.asarray(quats, dtype=np.float32)
    if not isinstance(opacities, np.ndarray):
        opacities = np.asarray(opacities, dtype=np.float32)
    if not isinstance(sh0, np.ndarray):
        sh0 = np.asarray(sh0, dtype=np.float32)
    if shN is not None and not isinstance(shN, np.ndarray):
        shN = np.asarray(shN, dtype=np.float32)  # noqa: N806

    # Fast path: check if all arrays are already float32
    all_float32 = (
        means.dtype == np.float32
        and scales.dtype == np.float32
        and quats.dtype == np.float32
        and opacities.dtype == np.float32
        and sh0.dtype == np.float32
        and (shN is None or shN.dtype == np.float32)
    )

    # Only convert dtype if needed (avoids copy when already float32)
    if not all_float32:
        if means.dtype != np.float32:
            means = means.astype(np.float32, copy=False)
        if scales.dtype != np.float32:
            scales = scales.astype(np.float32, copy=False)
        if quats.dtype != np.float32:
            quats = quats.astype(np.float32, copy=False)
        if opacities.dtype != np.float32:
            opacities = opacities.astype(np.float32, copy=False)
        if sh0.dtype != np.float32:
            sh0 = sh0.astype(np.float32, copy=False)
        if shN is not None and shN.dtype != np.float32:
            shN = shN.astype(np.float32, copy=False)  # noqa: N806

    num_gaussians = means.shape[0]

    # Validate shapes if requested
    if validate:
        assert means.shape == (num_gaussians, 3), (
            f"means array has incorrect shape: expected ({num_gaussians}, 3), "
            f"got {means.shape}. Ensure all arrays have the same number of Gaussians (N)."
        )
        assert scales.shape == (num_gaussians, 3), (
            f"scales array has incorrect shape: expected ({num_gaussians}, 3), "
            f"got {scales.shape}. Ensure all arrays have the same number of Gaussians (N)."
        )
        assert quats.shape == (num_gaussians, 4), (
            f"quats array has incorrect shape: expected ({num_gaussians}, 4), "
            f"got {quats.shape}. Quaternions must have 4 components (w, x, y, z)."
        )
        assert opacities.shape == (num_gaussians,), (
            f"opacities array has incorrect shape: expected ({num_gaussians},), "
            f"got {opacities.shape}. Opacities should be a 1D array with one value per Gaussian."
        )
        assert sh0.shape == (num_gaussians, 3), (
            f"sh0 array has incorrect shape: expected ({num_gaussians}, 3), "
            f"got {sh0.shape}. SH DC coefficients must have 3 components (RGB)."
        )

    # Flatten shN if needed (from (N, K, 3) to (N, K*3))
    if shN is not None and shN.ndim == 3:
        N, K, C = shN.shape  # noqa: N806
        if validate:
            assert C == 3, f"shN must have shape (N, K, 3), got {shN.shape}"
        shN = shN.reshape(N, K * 3)  # noqa: N806

    return means, scales, quats, opacities, sh0, shN


def _compress_data_internal(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    shN: np.ndarray | None,  # noqa: N803
) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    """Internal function to compress Gaussian data (shared compression logic).

    This function contains the core compression logic extracted from write_compressed().
    All inputs must be pre-validated and normalized to float32.

    Args:
        means: (N, 3) float32 - xyz positions
        scales: (N, 3) float32 - scale parameters
        quats: (N, 4) float32 - rotation quaternions
        opacities: (N,) float32 - opacity values
        sh0: (N, 3) float32 - DC spherical harmonics
        shN: (N, K*3) float32 or None - flattened SH coefficients

    Returns:
        Tuple of (header_bytes, chunk_bounds, packed_data, packed_sh, num_gaussians, num_chunks)
    """
    num_gaussians = means.shape[0]
    num_chunks = (num_gaussians + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Pre-compute chunk indices for all vertices (vectorized)
    # Use bit shift instead of division (256 = 2^8, so >> 8 is faster)
    chunk_indices = np.arange(num_gaussians, dtype=np.int32) >> CHUNK_SIZE_SHIFT

    # OPTIMIZATION: chunk_indices are ALWAYS already sorted!
    # Since we compute chunk_indices = np.arange(num_gaussians) >> CHUNK_SIZE_SHIFT,
    # the indices are sequential [0,0,0..., 1,1,1..., 2,2,2...] which is already sorted.
    sorted_chunk_indices = chunk_indices
    sorted_means = means
    sorted_scales = scales
    sorted_sh0 = sh0
    sorted_quats = quats
    sorted_opacities = opacities
    sorted_shN = shN  # noqa: N806

    # Pre-compute SH0 to RGB conversion (used in chunk bounds and packing)
    sorted_color_rgb = sorted_sh0 * SH_C0 + 0.5

    # Compute chunk boundaries (start/end indices for each chunk)
    chunk_starts = np.arange(num_chunks, dtype=np.int32) * CHUNK_SIZE
    chunk_ends = np.minimum(chunk_starts + CHUNK_SIZE, num_gaussians)

    # Allocate chunk bounds arrays
    chunk_bounds = np.zeros((num_chunks, 18), dtype=np.float32)

    # Compute chunk bounds using JIT-compiled function
    chunk_bounds = _compute_chunk_bounds_jit(
        sorted_means, sorted_scales, sorted_color_rgb, chunk_starts, chunk_ends
    )

    # Extract individual min/max values for packing (views into chunk_bounds)
    min_x, min_y, min_z = chunk_bounds[:, 0], chunk_bounds[:, 1], chunk_bounds[:, 2]
    max_x, max_y, max_z = chunk_bounds[:, 3], chunk_bounds[:, 4], chunk_bounds[:, 5]
    min_scale_x, min_scale_y, min_scale_z = (
        chunk_bounds[:, 6],
        chunk_bounds[:, 7],
        chunk_bounds[:, 8],
    )
    max_scale_x, max_scale_y, max_scale_z = (
        chunk_bounds[:, 9],
        chunk_bounds[:, 10],
        chunk_bounds[:, 11],
    )
    min_r, min_g, min_b = chunk_bounds[:, 12], chunk_bounds[:, 13], chunk_bounds[:, 14]
    max_r, max_g, max_b = chunk_bounds[:, 15], chunk_bounds[:, 16], chunk_bounds[:, 17]

    # Allocate packed vertex data (4 uint32 per vertex)
    packed_data = np.zeros((num_gaussians, 4), dtype=np.uint32)

    # Use JIT-compiled functions for parallel compression (5-6x faster)
    packed_data[:, 0] = _pack_positions_jit(
        sorted_means, sorted_chunk_indices, min_x, min_y, min_z, max_x, max_y, max_z
    )
    packed_data[:, 2] = _pack_scales_jit(
        sorted_scales,
        sorted_chunk_indices,
        min_scale_x,
        min_scale_y,
        min_scale_z,
        max_scale_x,
        max_scale_y,
        max_scale_z,
    )
    packed_data[:, 3] = _pack_colors_jit(
        sorted_color_rgb,
        sorted_opacities,
        sorted_chunk_indices,
        min_r,
        min_g,
        min_b,
        max_r,
        max_g,
        max_b,
    )
    packed_data[:, 1] = _pack_quaternions_jit(sorted_quats)

    # SH coefficient compression (8-bit quantization)
    packed_sh = None
    if sorted_shN is not None and sorted_shN.shape[1] > 0:
        # Quantize to uint8: ((shN / 8 + 0.5) * 256), clamped to [0, 255]
        # Simplified to: shN * 32 + 128, clamped to [0, 255]
        packed_sh = np.clip(sorted_shN * 32.0 + 128.0, 0, 255).astype(np.uint8)

    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element chunk {num_chunks}",
    ]

    # Add chunk properties (18 floats)
    chunk_props = [
        "min_x",
        "min_y",
        "min_z",
        "max_x",
        "max_y",
        "max_z",
        "min_scale_x",
        "min_scale_y",
        "min_scale_z",
        "max_scale_x",
        "max_scale_y",
        "max_scale_z",
        "min_r",
        "min_g",
        "min_b",
        "max_r",
        "max_g",
        "max_b",
    ]
    for prop in chunk_props:
        header_lines.append(f"property float {prop}")

    # Add vertex element
    header_lines.append(f"element vertex {num_gaussians}")
    header_lines.append("property uint packed_position")
    header_lines.append("property uint packed_rotation")
    header_lines.append("property uint packed_scale")
    header_lines.append("property uint packed_color")

    # Add SH element if present
    if packed_sh is not None:
        num_sh_coeffs = packed_sh.shape[1]
        header_lines.append(f"element sh {num_gaussians}")
        for i in range(num_sh_coeffs):
            header_lines.append(f"property uchar coeff_{i}")

    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"
    header_bytes = header.encode("ascii")

    return header_bytes, chunk_bounds, packed_data, packed_sh, num_gaussians, num_chunks


# ======================================================================================
# UNCOMPRESSED PLY WRITER
# ======================================================================================


def write_uncompressed(
    file_path: str | Path,
    data: "GSData",  # noqa: F821
    validate: bool = True,
) -> None:
    """Write uncompressed Gaussian splatting PLY file with zero-copy optimization.

    Always operates on GSData objects. Automatically uses zero-copy when data has
    a _base array (from plyread), achieving 6-8x speedup.

    Performance:
        - Zero-copy path (data with _base): Header + I/O only, no memory copying
          * 400K SH3: ~15-20ms (vs 121ms without optimization) - 6-8x faster!
        - Standard path (data without _base): ~20-120ms depending on size and SH degree
        - Peak: 70M Gaussians/sec for 400K Gaussians, SH0 (zero-copy)

    Args:
        file_path: Output PLY file path
        data: GSData object containing Gaussian parameters
        validate: If True, validate input shapes (default True)

    Example:
        >>> # RECOMMENDED: Pass GSData directly (automatic zero-copy)
        >>> data = plyread("input.ply")
        >>> write_uncompressed("output.ply", data)  # 6-8x faster!
        >>>
        >>> # Create GSData from scratch
        >>> data = GSData(means, scales, quats, opacities, sh0, shN)
        >>> write_uncompressed("output.ply", data)
    """
    file_path = Path(file_path)

    # ZERO-COPY FAST PATH: Write _base array directly if it exists
    if data._base is not None:
        num_gaussians = len(data)
        # shN.shape = (N, K, 3) where K is number of bands
        # Header needs total coefficients = K * 3
        num_sh_rest = (
            data.shN.shape[1] * 3 if (data.shN is not None and data.shN.size > 0) else None
        )
        header_bytes = _build_header_fast(num_gaussians, num_sh_rest)

        buffer_size = (
            _LARGE_BUFFER_SIZE if data._base.nbytes > _LARGE_FILE_THRESHOLD else _SMALL_BUFFER_SIZE
        )
        with open(file_path, "wb", buffering=buffer_size) as f:
            f.write(header_bytes)
            data._base.tofile(f)

        logger.debug(
            f"[Gaussian PLY] Wrote uncompressed (zero-copy): {num_gaussians} Gaussians to {file_path.name}"
        )
        return

    # STANDARD PATH: Construct array from GSData fields
    means, scales, quats, opacities, sh0, shN = data.unpack()  # noqa: N806

    # Validate and normalize inputs using shared helper
    means, scales, quats, opacities, sh0, shN = _validate_and_normalize_inputs(  # noqa: N806
        means, scales, quats, opacities, sh0, shN, validate
    )

    num_gaussians = means.shape[0]

    # Build header using pre-computed templates (3-5% faster)
    num_sh_rest = shN.shape[1] if shN is not None else None
    header_bytes = _build_header_fast(num_gaussians, num_sh_rest)

    # STANDARD PATH: Construct array from individual components
    # Used when data was created from scratch (not from plyread)
    if shN is not None:
        sh_coeffs = shN.shape[1]  # Number of SH coefficients (already reshaped to N x K*3)
        total_props = 3 + 3 + sh_coeffs + 1 + 3 + 4  # means, sh0, shN, opacity, scales, quats
        output_array = np.empty((num_gaussians, total_props), dtype="<f4")
        output_array[:, 0:3] = means
        output_array[:, 3:6] = sh0
        output_array[:, 6 : 6 + sh_coeffs] = shN
        output_array[:, 6 + sh_coeffs] = opacities  # Direct 1D assignment, no need for slicing
        output_array[:, 7 + sh_coeffs : 10 + sh_coeffs] = scales
        output_array[:, 10 + sh_coeffs : 14 + sh_coeffs] = quats
    else:
        output_array = np.empty((num_gaussians, 14), dtype="<f4")
        output_array[:, 0:3] = means
        output_array[:, 3:6] = sh0
        output_array[:, 6] = opacities  # Direct 1D assignment to single column
        output_array[:, 7:10] = scales
        output_array[:, 10:14] = quats

    # Write with optimized buffering (1-3% faster for large files)
    buffer_size = (
        _LARGE_BUFFER_SIZE if output_array.nbytes > _LARGE_FILE_THRESHOLD else _SMALL_BUFFER_SIZE
    )
    with open(file_path, "wb", buffering=buffer_size) as f:
        f.write(header_bytes)
        output_array.tofile(f)

    logger.debug(
        f"[Gaussian PLY] Wrote uncompressed: {num_gaussians} Gaussians to {file_path.name}"
    )


# ======================================================================================
# COMPRESSED PLY WRITER (VECTORIZED)
# ======================================================================================


def write_compressed(
    file_path: str | Path,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    shN: np.ndarray | None = None,  # noqa: N803
    validate: bool = True,
) -> None:
    """Write compressed Gaussian splatting PLY file (PlayCanvas format).

    Compresses data using chunk-based quantization (256 Gaussians per chunk).
    Achieves 3.8-14.5x compression ratio using highly optimized vectorized operations.

    Uses Numba JIT compilation for fast parallel compression (3.8x faster than pure NumPy).

    Args:
        file_path: Output PLY file path
        means: (N, 3) - xyz positions
        scales: (N, 3) - scale parameters
        quats: (N, 4) - rotation quaternions (must be normalized)
        opacities: (N,) - opacity values
        sh0: (N, 3) - DC spherical harmonics
        shN: (N, K, 3) or (N, K*3) - Higher-order SH coefficients (optional)
        validate: If True, validate input shapes (default True)

    Performance:
        - With JIT: ~15ms for 400K Gaussians, SH0 (27M Gaussians/sec)
        - With JIT: ~92ms for 400K Gaussians, SH3 (4.4M Gaussians/sec)

    Format:
        Compressed PLY with chunk-based quantization:
        - 256 Gaussians per chunk
        - Position: 11-10-11 bit quantization
        - Scale: 11-10-11 bit quantization
        - Color: 8-8-8-8 bit quantization
        - Quaternion: smallest-three encoding (2+10+10+10 bits)
        - SH coefficients: 8-bit quantization (optional)

    Example:
        >>> write_compressed("output.ply", means, scales, quats, opacities, sh0, shN)
        >>> # File is 14.5x smaller than uncompressed
    """
    file_path = Path(file_path)

    # Validate and normalize inputs using shared helper
    means, scales, quats, opacities, sh0, shN = _validate_and_normalize_inputs(  # noqa: N806
        means, scales, quats, opacities, sh0, shN, validate
    )

    # Use internal compression function
    header_bytes, chunk_bounds, packed_data, packed_sh, num_gaussians, num_chunks = (
        _compress_data_internal(means, scales, quats, opacities, sh0, shN)
    )

    # Write to file
    with open(file_path, "wb") as f:
        f.write(header_bytes)
        chunk_bounds.tofile(f)
        packed_data.tofile(f)
        if packed_sh is not None:
            packed_sh.tofile(f)

    logger.debug(
        f"[Gaussian PLY] Wrote compressed: {num_gaussians} Gaussians to {file_path.name} "
        f"({num_chunks} chunks, {len(header_bytes) + chunk_bounds.nbytes + packed_data.nbytes + (packed_sh.nbytes if packed_sh is not None else 0)} bytes)"
    )


def compress_to_bytes(
    data_or_means: GSData | np.ndarray,
    scales: np.ndarray | None = None,
    quats: np.ndarray | None = None,
    opacities: np.ndarray | None = None,
    sh0: np.ndarray | None = None,
    shN: np.ndarray | None = None,  # noqa: N803
    validate: bool = True,
) -> bytes:
    """Compress Gaussian splatting data to bytes (PlayCanvas format).

    Compresses Gaussian data into PlayCanvas format and returns as bytes,
    without writing to disk. Useful for network transfer or custom storage.

    Args:
        data_or_means: Either a GSData object or means array (N, 3) float32
        scales: Gaussian scales (N, 3) float32 (required if first arg is means)
        quats: Gaussian quaternions (N, 4) float32 (required if first arg is means)
        opacities: Gaussian opacities (N,) float32 (required if first arg is means)
        sh0: Degree 0 SH coefficients RGB (N, 3) float32 (required if first arg is means)
        shN: Optional higher degree SH coefficients (N, K, 3) float32
        validate: Whether to validate inputs

    Returns:
        bytes: Complete compressed PLY file as bytes

    Example:
        >>> from gsply import plyread, compress_to_bytes
        >>> # Method 1: Using GSData (recommended)
        >>> data = plyread("model.ply")
        >>> compressed_bytes = compress_to_bytes(data)
        >>>
        >>> # Method 2: Using individual arrays (backward compatible)
        >>> compressed_bytes = compress_to_bytes(
        ...     means, scales, quats, opacities, sh0, shN
        ... )
        >>>
        >>> # Save or transmit
        >>> with open("output.compressed.ply", "wb") as f:
        ...     f.write(compressed_bytes)
    """
    # Handle GSData input
    if isinstance(data_or_means, GSData):
        means = data_or_means.means
        scales = data_or_means.scales
        quats = data_or_means.quats
        opacities = data_or_means.opacities
        sh0 = data_or_means.sh0
        shN = data_or_means.shN  # noqa: N806
    else:
        # Use individual arrays
        means = data_or_means
        if scales is None or quats is None or opacities is None or sh0 is None:
            raise ValueError(
                "When passing individual arrays, scales, quats, opacities, and sh0 are required. "
                "Consider using GSData for cleaner API: compress_to_bytes(data)"
            )

    # Validate and normalize inputs
    means, scales, quats, opacities, sh0, shN = _validate_and_normalize_inputs(  # noqa: N806
        means, scales, quats, opacities, sh0, shN, validate
    )

    # Compress data using internal helper
    header_bytes, chunk_bounds, packed_data, packed_sh, num_gaussians, num_chunks = (
        _compress_data_internal(means, scales, quats, opacities, sh0, shN)
    )

    # Assemble complete file bytes (use bytes.join for ~4% speed improvement)
    parts = [header_bytes, chunk_bounds.tobytes(), packed_data.tobytes()]
    if packed_sh is not None:
        parts.append(packed_sh.tobytes())
    total_bytes = b"".join(parts)

    logger.debug(
        f"[Gaussian PLY] Compressed to bytes: {num_gaussians} Gaussians "
        f"({num_chunks} chunks, {len(total_bytes)} bytes)"
    )

    return total_bytes


def compress_to_arrays(
    data_or_means: GSData | np.ndarray,
    scales: np.ndarray | None = None,
    quats: np.ndarray | None = None,
    opacities: np.ndarray | None = None,
    sh0: np.ndarray | None = None,
    shN: np.ndarray | None = None,  # noqa: N803
    validate: bool = True,
) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compress Gaussian splatting data to component arrays (PlayCanvas format).

    Compresses Gaussian data into PlayCanvas format and returns as separate
    components (header, chunks, data, SH), without writing to disk.
    Useful for custom processing or partial updates.

    Args:
        data_or_means: Either a GSData object or means array (N, 3) float32
        scales: Gaussian scales (N, 3) float32 (required if first arg is means)
        quats: Gaussian quaternions (N, 4) float32 (required if first arg is means)
        opacities: Gaussian opacities (N,) float32 (required if first arg is means)
        sh0: Degree 0 SH coefficients RGB (N, 3) float32 (required if first arg is means)
        shN: Optional higher degree SH coefficients (N, K, 3) float32
        validate: Whether to validate inputs

    Returns:
        Tuple containing:
        - header_bytes: PLY header as bytes
        - chunk_bounds: Chunk boundary array (num_chunks, 18) float32
        - packed_data: Main compressed data array (N, 4) uint32
        - packed_sh: Optional compressed SH data array uint8

    Example:
        >>> from gsply import plyread, compress_to_arrays
        >>> # Method 1: Using GSData (recommended)
        >>> data = plyread("model.ply")
        >>> header, chunks, packed, sh = compress_to_arrays(data)
        >>>
        >>> # Method 2: Using individual arrays (backward compatible)
        >>> header, chunks, packed, sh = compress_to_arrays(
        ...     means, scales, quats, opacities, sh0, shN
        ... )
        >>>
        >>> # Process components individually
        >>> print(f"Header size: {len(header)} bytes")
        >>> print(f"Chunks shape: {chunks.shape}")
        >>> print(f"Packed data: {packed.nbytes} bytes")
    """
    # Handle GSData input
    if isinstance(data_or_means, GSData):
        means = data_or_means.means
        scales = data_or_means.scales
        quats = data_or_means.quats
        opacities = data_or_means.opacities
        sh0 = data_or_means.sh0
        shN = data_or_means.shN  # noqa: N806
    else:
        # Use individual arrays
        means = data_or_means
        if scales is None or quats is None or opacities is None or sh0 is None:
            raise ValueError(
                "When passing individual arrays, scales, quats, opacities, and sh0 are required. "
                "Consider using GSData for cleaner API: compress_to_arrays(data)"
            )

    # Validate and normalize inputs
    means, scales, quats, opacities, sh0, shN = _validate_and_normalize_inputs(  # noqa: N806
        means, scales, quats, opacities, sh0, shN, validate
    )

    # Compress data using internal helper
    header_bytes, chunk_bounds, packed_data, packed_sh, num_gaussians, num_chunks = (
        _compress_data_internal(means, scales, quats, opacities, sh0, shN)
    )

    logger.debug(
        f"[Gaussian PLY] Compressed to arrays: {num_gaussians} Gaussians "
        f"({num_chunks} chunks, header={len(header_bytes)} bytes, "
        f"bounds={chunk_bounds.nbytes} bytes, data={packed_data.nbytes} bytes, "
        f"sh={packed_sh.nbytes if packed_sh is not None else 0} bytes)"
    )

    return header_bytes, chunk_bounds, packed_data, packed_sh


# ======================================================================================
# UNIFIED WRITING API
# ======================================================================================


def plywrite(
    file_path: str | Path,
    data: "GSData | np.ndarray",  # noqa: F821
    scales: np.ndarray | None = None,
    quats: np.ndarray | None = None,
    opacities: np.ndarray | None = None,
    sh0: np.ndarray | None = None,
    shN: np.ndarray | None = None,  # noqa: N803
    compressed: bool = False,
    validate: bool = True,
) -> None:
    """Write Gaussian splatting PLY file with automatic optimization.

    Supports two input patterns:
    1. GSData object (RECOMMENDED): Automatic zero-copy optimization
    2. Individual arrays: Converted to GSData and auto-consolidated

    Automatic optimizations:
    - Zero-copy writes: GSData with _base (from plyread) writes 2.9x faster
    - Auto-consolidation: GSData without _base is automatically consolidated
      for 2.4x faster writes (one-time 10-35ms cost, faster even for single write!)

    Format selection (automatic based on compressed parameter or extension):
    - compressed=False or .ply -> uncompressed (fast, zero-copy optimized)
    - compressed=True -> automatically saves as .compressed.ply
    - .compressed.ply or .ply_compressed extension -> compressed format

    Args:
        file_path: Output PLY file path (extension auto-adjusted if compressed=True)
        data: GSData object OR (N, 3) xyz positions array
        scales: (N, 3) scale parameters (required if data is array)
        quats: (N, 4) rotation quaternions (required if data is array)
        opacities: (N,) opacity values (required if data is array)
        sh0: (N, 3) DC spherical harmonics (required if data is array)
        shN: (N, K, 3) or (N, K*3) - Higher-order SH coefficients (optional)
        compressed: If True, write compressed format and auto-adjust extension
        validate: If True, validate input shapes (default True)

    Performance:
        - GSData from plyread: ~7ms for 400K Gaussians (zero-copy, 53 M/s)
        - GSData created manually: ~19ms for 400K Gaussians (auto-consolidated, 49 M/s)
        - Individual arrays: ~19ms for 400K Gaussians (converted + consolidated)
        - All methods produce identical output

    Example:
        >>> # RECOMMENDED: Pass GSData from file (automatic zero-copy)
        >>> data = plyread("input.ply")
        >>> plywrite("output.ply", data)  # ~7ms for 400K, zero-copy!
        >>>
        >>> # GSData created manually (auto-consolidated)
        >>> data = GSData(means=means, scales=scales, ...)
        >>> plywrite("output.ply", data)  # ~19ms for 400K, auto-optimized!
        >>>
        >>> # Individual arrays (converted + auto-consolidated)
        >>> plywrite("output.ply", means, scales, quats, opacities, sh0, shN)
        >>>
        >>> # Write compressed format
        >>> plywrite("output.ply", data, compressed=True)
    """
    from gsply.gsdata import GSData  # noqa: PLC0415

    file_path = Path(file_path)

    # Convert individual arrays to GSData
    if not isinstance(data, GSData):
        # data is actually means array
        if any(x is None for x in [scales, quats, opacities, sh0]):
            raise ValueError(
                "When passing individual arrays, all of data (means), scales, quats, "
                "opacities, and sh0 must be provided"
            )
        # Create GSData without _base (will auto-consolidate below)
        data = GSData(
            means=data,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN if shN is not None else np.empty((data.shape[0], 0, 3), dtype=np.float32),
            _base=None,  # No _base for manually created data
        )

    # Auto-consolidate for uncompressed writes if no _base exists
    # This provides 2.4x faster writes even for a single write!
    # Break-even point: exactly 1 write (faster from the first write)
    if (
        data._base is None
        and not compressed
        and not file_path.name.endswith((".ply_compressed", ".compressed.ply"))
    ):
        logger.debug(
            f"[Gaussian PLY] Auto-consolidating {len(data):,} Gaussians for optimized write "
            "(2.4x faster, one-time 10-35ms cost)"
        )
        data = data.consolidate()

    # Auto-detect compression from extension
    is_compressed_ext = file_path.name.endswith((".ply_compressed", ".compressed.ply"))

    # Check if compressed format requested
    if compressed or is_compressed_ext:
        # If compressed=True but no compressed extension, add .compressed.ply
        if compressed and not is_compressed_ext:
            # Replace .ply with .compressed.ply, or just append if no .ply
            if file_path.suffix == ".ply":
                file_path = file_path.with_suffix(".compressed.ply")
            else:
                file_path = Path(str(file_path) + ".compressed.ply")

        # Extract arrays for compressed write (compressed write doesn't use GSData yet)
        means, scales, quats, opacities, sh0, shN = data.unpack()  # noqa: N806
        write_compressed(file_path, means, scales, quats, opacities, sh0, shN)
    else:
        write_uncompressed(file_path, data, validate=validate)


__all__ = [
    "plywrite",
    "write_uncompressed",
    "write_compressed",
    "compress_to_bytes",
    "compress_to_arrays",
]
