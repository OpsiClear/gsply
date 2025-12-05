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
from typing import TYPE_CHECKING

import numba
import numpy as np

# Import numba for JIT optimization
from numba import jit

from gsply.formats import CHUNK_SIZE, SH_C0
from gsply.gsdata import (
    DataFormat,
    GSData,
    _create_format_dict,
    _interleave_sh0_jit,
    _interleave_shn_jit,
)

if TYPE_CHECKING:
    from gsply.torch.gstensor import GSTensor  # noqa: F401

logger = logging.getLogger(__name__)


# ======================================================================================
# I/O BUFFER SIZE CONSTANTS
# ======================================================================================

# Buffer sizes for optimized I/O performance
_LARGE_BUFFER_SIZE = 2 * 1024 * 1024  # 2MB buffer for large files
_SMALL_BUFFER_SIZE = 1 * 1024 * 1024  # 1MB buffer for small files
_LARGE_FILE_THRESHOLD = 10_000_000  # 10MB threshold for buffer size selection


# ======================================================================================
# BIT-PACKING QUANTIZATION CONSTANTS
# ======================================================================================

# Quantization maxima for bit-packing (used in compression)
# Position and Scale: 11-10-11 bit scheme
_QUANTIZE_11_BIT_MAX = 2047.0  # 2^11 - 1 = 2047 (X and Z coordinates)
_QUANTIZE_10_BIT_MAX = 1023.0  # 2^10 - 1 = 1023 (Y coordinate and quaternions)
_QUANTIZE_8_BIT_MAX = 255.0  # 2^8 - 1 = 255 (RGB and opacity)

# Rounding offset for proper quantization (avoids truncation bias)
_ROUNDING_OFFSET = 0.5

# Bit shift positions for 32-bit packing
# Position/Scale packing: (X:11 bits)(Y:10 bits)(Z:11 bits)
_POSITION_X_SHIFT = 21  # bits 31-21: X coordinate (11 bits)
_POSITION_Y_SHIFT = 11  # bits 20-11: Y coordinate (10 bits)
_POSITION_Z_SHIFT = 0  # bits 10-0: Z coordinate (11 bits)

# Quaternion packing: (largest_idx:2 bits)(qa:10 bits)(qb:10 bits)(qc:10 bits)
_QUAT_INDEX_SHIFT = 30  # bits 31-30: largest component index (2 bits)
_QUAT_A_SHIFT = 20  # bits 29-20: first remaining component (10 bits)
_QUAT_B_SHIFT = 10  # bits 19-10: second remaining component (10 bits)
_QUAT_C_SHIFT = 0  # bits 9-0: third remaining component (10 bits)

# Color packing: (R:8 bits)(G:8 bits)(B:8 bits)(Opacity:8 bits)
_COLOR_R_SHIFT = 24  # bits 31-24: red channel (8 bits)
_COLOR_G_SHIFT = 16  # bits 23-16: green channel (8 bits)
_COLOR_B_SHIFT = 8  # bits 15-8: blue channel (8 bits)
_COLOR_O_SHIFT = 0  # bits 7-0: opacity (8 bits)


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

    :param num_gaussians: Number of Gaussians
    :param num_sh_rest: Number of higher-order SH coefficients (None for SH0)
    :returns: Header bytes ready to write
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
def _pack_positions_jit(
    sorted_means, chunk_indices, min_x, min_y, min_z, range_x, range_y, range_z
):
    """JIT-compiled position quantization and packing (11-10-11 bits) with parallel processing.

    Optimized: Pre-computed ranges (1.44x speedup) - ranges computed once per chunk instead of every vertex.

    :param sorted_means: (N, 3) float32 array of positions
    :param chunk_indices: int32 array of chunk indices for each vertex
    :param min_x: chunk minimum x bounds
    :param min_y: chunk minimum y bounds
    :param min_z: chunk minimum z bounds
    :param range_x: chunk x range (max - min, pre-computed)
    :param range_y: chunk y range (max - min, pre-computed)
    :param range_z: chunk z range (max - min, pre-computed)
    :returns: (N,) uint32 array of packed positions
    """
    n = len(sorted_means)
    packed = np.zeros(n, dtype=np.uint32)

    for i in numba.prange(n):
        chunk_idx = chunk_indices[i]

        # Normalize to [0, 1] using pre-computed ranges
        norm_x = (sorted_means[i, 0] - min_x[chunk_idx]) / range_x[chunk_idx]
        norm_y = (sorted_means[i, 1] - min_y[chunk_idx]) / range_y[chunk_idx]
        norm_z = (sorted_means[i, 2] - min_z[chunk_idx]) / range_z[chunk_idx]

        # Clamp
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        norm_z = max(0.0, min(1.0, norm_z))

        # Quantize to integer range
        px = np.uint32(norm_x * _QUANTIZE_11_BIT_MAX)
        py = np.uint32(norm_y * _QUANTIZE_10_BIT_MAX)
        pz = np.uint32(norm_z * _QUANTIZE_11_BIT_MAX)

        # Pack into 32-bit integer: (X:11 bits)(Y:10 bits)(Z:11 bits)
        packed[i] = (px << _POSITION_X_SHIFT) | (py << _POSITION_Y_SHIFT) | pz

    return packed


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _pack_scales_jit(
    sorted_scales, chunk_indices, min_sx, min_sy, min_sz, range_sx, range_sy, range_sz
):
    """JIT-compiled scale quantization and packing (11-10-11 bits) with parallel processing.

    Optimized: Pre-computed ranges (1.44x speedup) - ranges computed once per chunk instead of every vertex.

    :param sorted_scales: (N, 3) float32 array of scales
    :param chunk_indices: int32 array of chunk indices for each vertex
    :param min_sx: chunk minimum scale x bounds
    :param min_sy: chunk minimum scale y bounds
    :param min_sz: chunk minimum scale z bounds
    :param range_sx: chunk scale x range (max - min, pre-computed)
    :param range_sy: chunk scale y range (max - min, pre-computed)
    :param range_sz: chunk scale z range (max - min, pre-computed)
    :returns: (N,) uint32 array of packed scales
    """
    n = len(sorted_scales)
    packed = np.zeros(n, dtype=np.uint32)

    for i in numba.prange(n):
        chunk_idx = chunk_indices[i]

        # Normalize to [0, 1] using pre-computed ranges
        norm_sx = (sorted_scales[i, 0] - min_sx[chunk_idx]) / range_sx[chunk_idx]
        norm_sy = (sorted_scales[i, 1] - min_sy[chunk_idx]) / range_sy[chunk_idx]
        norm_sz = (sorted_scales[i, 2] - min_sz[chunk_idx]) / range_sz[chunk_idx]

        # Clamp
        norm_sx = max(0.0, min(1.0, norm_sx))
        norm_sy = max(0.0, min(1.0, norm_sy))
        norm_sz = max(0.0, min(1.0, norm_sz))

        # Quantize to integer range
        sx = np.uint32(norm_sx * _QUANTIZE_11_BIT_MAX)
        sy = np.uint32(norm_sy * _QUANTIZE_10_BIT_MAX)
        sz = np.uint32(norm_sz * _QUANTIZE_11_BIT_MAX)

        # Pack into 32-bit integer: (X:11 bits)(Y:10 bits)(Z:11 bits)
        packed[i] = (sx << _POSITION_X_SHIFT) | (sy << _POSITION_Y_SHIFT) | sz

    return packed


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _pack_colors_jit(
    sorted_color_rgb,
    sorted_opacities,
    chunk_indices,
    min_r,
    min_g,
    min_b,
    range_r,
    range_g,
    range_b,
):
    """JIT-compiled color and opacity quantization and packing (8-8-8-8 bits) with parallel processing.

    Optimized: Pre-computed ranges (1.44x speedup) - ranges computed once per chunk instead of every vertex.

    :param sorted_color_rgb: (N, 3) float32 array of pre-computed RGB colors (SH0 * SH_C0 + 0.5)
    :param sorted_opacities: (N,) float32 array of opacities (logit space)
    :param chunk_indices: int32 array of chunk indices for each vertex
    :param min_r: chunk minimum color r bounds
    :param min_g: chunk minimum color g bounds
    :param min_b: chunk minimum color b bounds
    :param range_r: chunk color r range (max - min, pre-computed)
    :param range_g: chunk color g range (max - min, pre-computed)
    :param range_b: chunk color b range (max - min, pre-computed)
    :returns: (N,) uint32 array of packed colors
    """
    n = len(sorted_color_rgb)
    packed = np.zeros(n, dtype=np.uint32)

    for i in numba.prange(n):
        chunk_idx = chunk_indices[i]

        # Use pre-computed RGB colors
        color_r = sorted_color_rgb[i, 0]
        color_g = sorted_color_rgb[i, 1]
        color_b = sorted_color_rgb[i, 2]

        # Normalize to [0, 1] using pre-computed ranges
        norm_r = (color_r - min_r[chunk_idx]) / range_r[chunk_idx]
        norm_g = (color_g - min_g[chunk_idx]) / range_g[chunk_idx]
        norm_b = (color_b - min_b[chunk_idx]) / range_b[chunk_idx]

        # Clamp
        norm_r = max(0.0, min(1.0, norm_r))
        norm_g = max(0.0, min(1.0, norm_g))
        norm_b = max(0.0, min(1.0, norm_b))

        # Quantize colors to 8-bit range
        cr = np.uint32(norm_r * _QUANTIZE_8_BIT_MAX)
        cg = np.uint32(norm_g * _QUANTIZE_8_BIT_MAX)
        cb = np.uint32(norm_b * _QUANTIZE_8_BIT_MAX)

        # Opacity: logit to linear
        opacity_linear = 1.0 / (1.0 + np.exp(-sorted_opacities[i]))
        opacity_linear = max(0.0, min(1.0, opacity_linear))
        co = np.uint32(opacity_linear * _QUANTIZE_8_BIT_MAX)

        # Pack into 32-bit integer: (R:8)(G:8)(B:8)(O:8)
        packed[i] = (cr << _COLOR_R_SHIFT) | (cg << _COLOR_G_SHIFT) | (cb << _COLOR_B_SHIFT) | co

    return packed


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _pack_quaternions_jit(sorted_quats):
    """JIT-compiled quaternion normalization and packing (2+10-10-10 bits, smallest-three) with parallel processing.

    :param sorted_quats: (N, 4) float32 array of quaternions
    :returns: (N,) uint32 array of packed quaternions
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

        # Quantize to 10-bit range
        qa_int = np.uint32(qa_norm * _QUANTIZE_10_BIT_MAX)
        qb_int = np.uint32(qb_norm * _QUANTIZE_10_BIT_MAX)
        qc_int = np.uint32(qc_norm * _QUANTIZE_10_BIT_MAX)

        # Pack into 32-bit integer: (index:2)(qa:10)(qb:10)(qc:10)
        packed[i] = (
            (np.uint32(largest_idx) << _QUAT_INDEX_SHIFT)
            | (qa_int << _QUAT_A_SHIFT)
            | (qb_int << _QUAT_B_SHIFT)
            | qc_int
        )

    return packed


# Chunk size shift constant (256 = 2^8)
_CHUNK_SIZE_SHIFT_PACK = 8


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _pack_all_jit(
    sorted_means,
    sorted_scales,
    sorted_color_rgb,
    sorted_opacities,
    sorted_quats,
    min_x,
    min_y,
    min_z,
    range_x,
    range_y,
    range_z,
    min_sx,
    min_sy,
    min_sz,
    range_sx,
    range_sy,
    range_sz,
    min_r,
    min_g,
    min_b,
    range_r,
    range_g,
    range_b,
):
    """Fused JIT-compiled packing of all vertex data in single parallel pass.

    Combines position, scale, color, and quaternion packing into one loop for:
    - Better cache locality (single pass over all data)
    - Reduced parallel overhead (1 loop instead of 4)
    - Chunk index computed inline (avoids redundant lookups)

    :param sorted_means: (N, 3) float32 array of positions
    :param sorted_scales: (N, 3) float32 array of scales
    :param sorted_color_rgb: (N, 3) float32 array of pre-computed RGB colors
    :param sorted_opacities: (N,) float32 array of opacities (logit space)
    :param sorted_quats: (N, 4) float32 array of quaternions
    :param min_x, min_y, min_z: chunk minimum position bounds
    :param range_x, range_y, range_z: chunk position ranges
    :param min_sx, min_sy, min_sz: chunk minimum scale bounds
    :param range_sx, range_sy, range_sz: chunk scale ranges
    :param min_r, min_g, min_b: chunk minimum color bounds
    :param range_r, range_g, range_b: chunk color ranges
    :returns: (N, 4) uint32 array with packed [position, quaternion, scale, color]
    """
    n = len(sorted_means)
    packed = np.zeros((n, 4), dtype=np.uint32)
    norm_factor = np.sqrt(2.0) * 0.5

    for i in numba.prange(n):
        # Compute chunk index inline (256 Gaussians per chunk)
        chunk_idx = i >> _CHUNK_SIZE_SHIFT_PACK

        # ======================================================================
        # SECTION 1: Pack positions (11-10-11 bits)
        # ======================================================================
        norm_x = (sorted_means[i, 0] - min_x[chunk_idx]) / range_x[chunk_idx]
        norm_y = (sorted_means[i, 1] - min_y[chunk_idx]) / range_y[chunk_idx]
        norm_z = (sorted_means[i, 2] - min_z[chunk_idx]) / range_z[chunk_idx]
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        norm_z = max(0.0, min(1.0, norm_z))
        px = np.uint32(norm_x * _QUANTIZE_11_BIT_MAX + _ROUNDING_OFFSET)
        py = np.uint32(norm_y * _QUANTIZE_10_BIT_MAX + _ROUNDING_OFFSET)
        pz = np.uint32(norm_z * _QUANTIZE_11_BIT_MAX + _ROUNDING_OFFSET)
        packed[i, 0] = (px << _POSITION_X_SHIFT) | (py << _POSITION_Y_SHIFT) | pz

        # ======================================================================
        # SECTION 2: Pack scales (11-10-11 bits)
        # ======================================================================
        norm_sx = (sorted_scales[i, 0] - min_sx[chunk_idx]) / range_sx[chunk_idx]
        norm_sy = (sorted_scales[i, 1] - min_sy[chunk_idx]) / range_sy[chunk_idx]
        norm_sz = (sorted_scales[i, 2] - min_sz[chunk_idx]) / range_sz[chunk_idx]
        norm_sx = max(0.0, min(1.0, norm_sx))
        norm_sy = max(0.0, min(1.0, norm_sy))
        norm_sz = max(0.0, min(1.0, norm_sz))
        sx = np.uint32(norm_sx * _QUANTIZE_11_BIT_MAX + _ROUNDING_OFFSET)
        sy = np.uint32(norm_sy * _QUANTIZE_10_BIT_MAX + _ROUNDING_OFFSET)
        sz = np.uint32(norm_sz * _QUANTIZE_11_BIT_MAX + _ROUNDING_OFFSET)
        packed[i, 2] = (sx << _POSITION_X_SHIFT) | (sy << _POSITION_Y_SHIFT) | sz

        # ======================================================================
        # SECTION 3: Pack colors (8-8-8-8 bits)
        # ======================================================================
        color_r = sorted_color_rgb[i, 0]
        color_g = sorted_color_rgb[i, 1]
        color_b = sorted_color_rgb[i, 2]
        norm_r = (color_r - min_r[chunk_idx]) / range_r[chunk_idx]
        norm_g = (color_g - min_g[chunk_idx]) / range_g[chunk_idx]
        norm_b = (color_b - min_b[chunk_idx]) / range_b[chunk_idx]
        norm_r = max(0.0, min(1.0, norm_r))
        norm_g = max(0.0, min(1.0, norm_g))
        norm_b = max(0.0, min(1.0, norm_b))
        cr = np.uint32(norm_r * _QUANTIZE_8_BIT_MAX + _ROUNDING_OFFSET)
        cg = np.uint32(norm_g * _QUANTIZE_8_BIT_MAX + _ROUNDING_OFFSET)
        cb = np.uint32(norm_b * _QUANTIZE_8_BIT_MAX + _ROUNDING_OFFSET)
        # Opacity: logit to linear (use rounding for better precision)
        opacity_linear = 1.0 / (1.0 + np.exp(-sorted_opacities[i]))
        opacity_linear = max(0.0, min(1.0, opacity_linear))
        co = np.uint32(opacity_linear * _QUANTIZE_8_BIT_MAX + _ROUNDING_OFFSET)
        packed[i, 3] = (cr << _COLOR_R_SHIFT) | (cg << _COLOR_G_SHIFT) | (cb << _COLOR_B_SHIFT) | co

        # ======================================================================
        # SECTION 4: Pack quaternions (2+10-10-10 bits, smallest-three)
        # ======================================================================
        qw = sorted_quats[i, 0]
        qx = sorted_quats[i, 1]
        qy = sorted_quats[i, 2]
        qz = sorted_quats[i, 3]

        # --- Step 4.1: Normalize quaternion ---
        qnorm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if qnorm > 0:
            inv_norm = 1.0 / qnorm
            qw *= inv_norm
            qx *= inv_norm
            qy *= inv_norm
            qz *= inv_norm

        # --- Step 4.2: Find largest component by absolute value ---
        abs_w, abs_x, abs_y, abs_z = abs(qw), abs(qx), abs(qy), abs(qz)
        largest_idx = 0
        largest_val = abs_w
        if abs_x > largest_val:
            largest_idx = 1
            largest_val = abs_x
        if abs_y > largest_val:
            largest_idx = 2
            largest_val = abs_y
        if abs_z > largest_val:
            largest_idx = 3

        # --- Step 4.3: Get components in order, flip sign if largest is negative ---
        if largest_idx == 0:
            if qw < 0:
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
            qa, qb, qc = qx, qy, qz
        elif largest_idx == 1:
            if qx < 0:
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
            qa, qb, qc = qw, qy, qz
        elif largest_idx == 2:
            if qy < 0:
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
            qa, qb, qc = qw, qx, qz
        else:
            if qz < 0:
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
            qa, qb, qc = qw, qx, qy

        # --- Step 4.4: Normalize to [0, 1] for quantization ---
        qa_norm = qa * norm_factor + 0.5
        qb_norm = qb * norm_factor + 0.5
        qc_norm = qc * norm_factor + 0.5
        qa_norm = max(0.0, min(1.0, qa_norm))
        qb_norm = max(0.0, min(1.0, qb_norm))
        qc_norm = max(0.0, min(1.0, qc_norm))

        # --- Step 4.5: Quantize and pack ---
        qa_int = np.uint32(qa_norm * _QUANTIZE_10_BIT_MAX + _ROUNDING_OFFSET)
        qb_int = np.uint32(qb_norm * _QUANTIZE_10_BIT_MAX + _ROUNDING_OFFSET)
        qc_int = np.uint32(qc_norm * _QUANTIZE_10_BIT_MAX + _ROUNDING_OFFSET)
        packed[i, 1] = (
            (np.uint32(largest_idx) << _QUAT_INDEX_SHIFT)
            | (qa_int << _QUAT_A_SHIFT)
            | (qb_int << _QUAT_B_SHIFT)
            | qc_int
        )

    return packed


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _compute_chunk_bounds_jit(
    sorted_means, sorted_scales, sorted_color_rgb, chunk_starts, chunk_ends
):
    """JIT-compiled chunk bounds computation (9x faster than Python loop).

    Computes min/max bounds for positions, scales, and colors for each chunk.
    This is the main bottleneck in compressed write (~90ms -> ~10ms).

    :param sorted_means: (N, 3) float32 array of positions
    :param sorted_scales: (N, 3) float32 array of scales
    :param sorted_color_rgb: (N, 3) float32 array of pre-computed RGB colors (SH0 * SH_C0 + 0.5)
    :param chunk_starts: (num_chunks,) int array of chunk start indices
    :param chunk_ends: (num_chunks,) int array of chunk end indices
    :returns: (num_chunks, 18) float32 array with layout [0:6] min_x, min_y, min_z, max_x, max_y, max_z,
              [6:12] min_scale_x/y/z, max_scale_x/y/z (clamped to [-20,20]), [12:18] min_r, min_g, min_b, max_r, max_g, max_b
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


def _ensure_numpy_arrays(
    means, scales, quats, opacities, sh0, shn
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Convert inputs to numpy arrays if they aren't already.

    :param means: Gaussian centers (any array-like)
    :param scales: Log scales (any array-like)
    :param quats: Rotations as quaternions (any array-like)
    :param opacities: Logit opacities (any array-like)
    :param sh0: DC spherical harmonics (any array-like)
    :param shn: Higher-order SH coefficients or None (any array-like or None)
    :return: Tuple of numpy arrays (may be converted to float32 if not already numpy arrays)
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
    """
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
    if shn is not None and not isinstance(shn, np.ndarray):
        shn = np.asarray(shn, dtype=np.float32)
    return means, scales, quats, opacities, sh0, shn


def _convert_to_float32(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    shn: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Convert arrays to float32 dtype if needed (avoids copy when already float32).

    :param means: Gaussian centers array
    :type means: np.ndarray
    :param scales: Log scales array
    :type scales: np.ndarray
    :param quats: Rotations as quaternions array
    :type quats: np.ndarray
    :param opacities: Logit opacities array
    :type opacities: np.ndarray
    :param sh0: DC spherical harmonics array
    :type sh0: np.ndarray
    :param shn: Higher-order SH coefficients or None
    :type shn: np.ndarray | None
    :return: Tuple of float32 arrays
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
    """
    # Fast path: check if all arrays are already float32
    all_float32 = (
        means.dtype == np.float32
        and scales.dtype == np.float32
        and quats.dtype == np.float32
        and opacities.dtype == np.float32
        and sh0.dtype == np.float32
        and (shn is None or shn.dtype == np.float32)
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
        if shn is not None and shn.dtype != np.float32:
            shn = shn.astype(np.float32, copy=False)

    return means, scales, quats, opacities, sh0, shn


def _validate_array_shapes(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    num_gaussians: int,
) -> None:
    """Validate that all arrays have the expected shapes.

    :param means: Gaussian centers array, shape (N, 3)
    :type means: np.ndarray
    :param scales: Log scales array, shape (N, 3)
    :type scales: np.ndarray
    :param quats: Rotations as quaternions array, shape (N, 4)
    :type quats: np.ndarray
    :param opacities: Logit opacities array, shape (N,)
    :type opacities: np.ndarray
    :param sh0: DC spherical harmonics array, shape (N, 3)
    :type sh0: np.ndarray
    :param num_gaussians: Expected number of Gaussians (N)
    :type num_gaussians: int
    :raises AssertionError: If any array has incorrect shape
    """
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


def _flatten_shn(shn: np.ndarray | None, validate: bool) -> np.ndarray | None:
    """Flatten shN array from (N, K, 3) to (N, 3*K) with channel-grouped ordering.

    Original 3DGS PLY format stores f_rest coefficients as channel-grouped:
    [R0,R1,...,Rk, G0,G1,...,Gk, B0,B1,...,Bk]

    This matches the original 3DGS save_ply which does:
    f_rest = features_rest.transpose(1, 2).flatten(start_dim=1)

    :param shn: Higher-order SH coefficients [N, K, 3] or None
    :type shn: np.ndarray | None
    :param validate: Whether to validate the shape
    :type validate: bool
    :return: Flattened shN array [N, 3*K] in channel-grouped order, or None
    :rtype: np.ndarray | None
    """
    if shn is not None and shn.ndim == 3:
        n_gaussians, n_bands, n_components = shn.shape
        if validate:
            assert n_components == 3, f"shN must have shape (N, K, 3), got {shn.shape}"
        # Transpose [N, K, 3] -> [N, 3, K] then flatten to [N, 3*K]
        # This gives channel-grouped order matching original 3DGS PLY format
        shn = shn.transpose(0, 2, 1).reshape(n_gaussians, n_bands * n_components)
    return shn


def _compute_chunk_boundaries(num_chunks: int, num_gaussians: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute chunk start and end indices for chunked processing.

    Each chunk contains CHUNK_SIZE Gaussians, except possibly the last chunk
    which may be smaller if num_gaussians is not a multiple of CHUNK_SIZE.

    :param num_chunks: Number of chunks
    :type num_chunks: int
    :param num_gaussians: Total number of Gaussians
    :type num_gaussians: int
    :return: Tuple of (chunk_starts, chunk_ends) arrays of shape (num_chunks,)
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    chunk_starts = np.arange(num_chunks, dtype=np.int32) * CHUNK_SIZE
    chunk_ends = np.minimum(chunk_starts + CHUNK_SIZE, num_gaussians)
    return chunk_starts, chunk_ends


def _validate_and_normalize_inputs(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    shn: np.ndarray | None,
    validate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and normalize input arrays to float32 format.

    :param means: Gaussian centers, shape (N, 3)
    :type means: np.ndarray
    :param scales: Log scales, shape (N, 3)
    :type scales: np.ndarray
    :param quats: Rotations as quaternions (wxyz), shape (N, 4)
    :type quats: np.ndarray
    :param opacities: Logit opacities, shape (N,)
    :type opacities: np.ndarray
    :param sh0: DC spherical harmonics, shape (N, 3)
    :type sh0: np.ndarray
    :param shn: Higher-order SH coefficients, shape (N, K, 3) or None
    :type shn: np.ndarray | None
    :param validate: Whether to validate shapes
    :type validate: bool
    :return: Tuple of normalized arrays (all float32)
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
    """
    # Step 1: Ensure all inputs are numpy arrays
    means, scales, quats, opacities, sh0, shn = _ensure_numpy_arrays(
        means, scales, quats, opacities, sh0, shn
    )

    # Step 2: Convert all arrays to float32 dtype
    means, scales, quats, opacities, sh0, shn = _convert_to_float32(
        means, scales, quats, opacities, sh0, shn
    )

    num_gaussians = means.shape[0]

    # Step 3: Validate shapes if requested
    if validate:
        _validate_array_shapes(means, scales, quats, opacities, sh0, num_gaussians)

    # Step 4: Flatten shN if needed (from (N, K, 3) to (N, K*3))
    shn = _flatten_shn(shn, validate)

    return means, scales, quats, opacities, sh0, shn


def _compress_data_internal(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    shn: np.ndarray | None,
) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    """Internal function to compress Gaussian data (shared compression logic).

    This function contains the core compression logic extracted from write_compressed().
    All inputs must be pre-validated and normalized to float32.

    :param means: (N, 3) float32 - xyz positions
    :param scales: (N, 3) float32 - scale parameters
    :param quats: (N, 4) float32 - rotation quaternions
    :param opacities: (N,) float32 - opacity values
    :param sh0: (N, 3) float32 - DC spherical harmonics
    :param shn: (N, K*3) float32 or None - flattened SH coefficients
    :returns: Tuple of (header_bytes, chunk_bounds, packed_data, packed_sh, num_gaussians, num_chunks)
    """
    num_gaussians = means.shape[0]
    num_chunks = (num_gaussians + CHUNK_SIZE - 1) // CHUNK_SIZE

    # OPTIMIZATION: Chunk indices are ALWAYS already sorted!
    # If we computed chunk_indices = np.arange(num_gaussians) >> CHUNK_SIZE_SHIFT,
    # the indices would be sequential [0,0,0..., 1,1,1..., 2,2,2...] which is already sorted.
    # Since we don't need to sort, we can skip computing chunk_indices entirely.
    sorted_means = means
    sorted_scales = scales
    sorted_sh0 = sh0
    sorted_quats = quats
    sorted_opacities = opacities
    sorted_shn = shn

    # Pre-compute SH0 to RGB conversion (used in chunk bounds and packing)
    sorted_color_rgb = sorted_sh0 * SH_C0 + 0.5

    # Compute chunk boundaries (start/end indices for each chunk)
    chunk_starts, chunk_ends = _compute_chunk_boundaries(num_chunks, num_gaussians)

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

    # Pre-compute ranges using vectorized NumPy operations
    # Uses np.maximum to handle zero-range case (replaces conditional: r if r > 0 else 1.0)
    # This is faster than Python loop for large num_chunks
    min_range_epsilon = np.float32(1e-10)  # Small epsilon to avoid division by zero

    # Position ranges (vectorized subtraction + max with epsilon)
    range_x = np.maximum(max_x - min_x, min_range_epsilon)
    range_y = np.maximum(max_y - min_y, min_range_epsilon)
    range_z = np.maximum(max_z - min_z, min_range_epsilon)

    # Scale ranges
    range_scale_x = np.maximum(max_scale_x - min_scale_x, min_range_epsilon)
    range_scale_y = np.maximum(max_scale_y - min_scale_y, min_range_epsilon)
    range_scale_z = np.maximum(max_scale_z - min_scale_z, min_range_epsilon)

    # Color ranges
    range_r = np.maximum(max_r - min_r, min_range_epsilon)
    range_g = np.maximum(max_g - min_g, min_range_epsilon)
    range_b = np.maximum(max_b - min_b, min_range_epsilon)

    # Use fused JIT-compiled function for parallel compression
    # Single pass over all data for better cache locality and reduced overhead
    packed_data = _pack_all_jit(
        sorted_means,
        sorted_scales,
        sorted_color_rgb,
        sorted_opacities,
        sorted_quats,
        min_x,
        min_y,
        min_z,
        range_x,
        range_y,
        range_z,
        min_scale_x,
        min_scale_y,
        min_scale_z,
        range_scale_x,
        range_scale_y,
        range_scale_z,
        min_r,
        min_g,
        min_b,
        range_r,
        range_g,
        range_b,
    )

    # SH coefficient compression (8-bit quantization)
    packed_sh = None
    if sorted_shn is not None and sorted_shn.shape[1] > 0:
        # Quantize to uint8: ((shN / 8 + 0.5) * 256), clamped to [0, 255]
        # Simplified to: shN * 32 + 128, clamped to [0, 255]
        packed_sh = np.clip(sorted_shn * 32.0 + 128.0, 0, 255).astype(np.uint8)

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

    :param file_path: Output PLY file path
    :param data: GSData object containing Gaussian parameters
    :param validate: If True, validate input shapes (default True)

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
    means, scales, quats, opacities, sh0, shn = data.unpack()

    # Validate and normalize inputs using shared helper
    means, scales, quats, opacities, sh0, shn = _validate_and_normalize_inputs(
        means, scales, quats, opacities, sh0, shn, validate
    )

    num_gaussians = means.shape[0]

    # Build header using pre-computed templates (3-5% faster)
    num_sh_rest = shn.shape[1] if shn is not None else None
    header_bytes = _build_header_fast(num_gaussians, num_sh_rest)

    # STANDARD PATH: Construct array using JIT-compiled interleaving (2.8-5x faster)
    # Ensure arrays are contiguous float32 for JIT kernels
    means = np.ascontiguousarray(means, dtype=np.float32)
    sh0 = np.ascontiguousarray(sh0, dtype=np.float32)
    opacities = np.ascontiguousarray(opacities.ravel(), dtype=np.float32)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    quats = np.ascontiguousarray(quats, dtype=np.float32)

    if shn is not None:
        sh_coeffs = shn.shape[1]  # Number of SH coefficients (already reshaped to N x K*3)
        total_props = 14 + sh_coeffs
        shn_flat = np.ascontiguousarray(shn, dtype=np.float32)
        output_array = np.empty((num_gaussians, total_props), dtype=np.float32)
        _interleave_shn_jit(means, sh0, shn_flat, opacities, scales, quats, output_array, sh_coeffs)
    else:
        output_array = np.empty((num_gaussians, 14), dtype=np.float32)
        _interleave_sh0_jit(means, sh0, opacities, scales, quats, output_array)

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

    :param file_path: Output PLY file path
    :param means: (N, 3) - xyz positions
    :param scales: (N, 3) - scale parameters
    :param quats: (N, 4) - rotation quaternions (must be normalized)
    :param opacities: (N,) - opacity values
    :param sh0: (N, 3) - DC spherical harmonics
    :param shN: (N, K, 3) or (N, K*3) - Higher-order SH coefficients (optional)
    :param validate: If True, validate input shapes (default True)

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

    :param data_or_means: Either a GSData object or means array (N, 3) float32
    :param scales: Gaussian scales (N, 3) float32 (required if first arg is means)
    :param quats: Gaussian quaternions (N, 4) float32 (required if first arg is means)
    :param opacities: Gaussian opacities (N,) float32 (required if first arg is means)
    :param sh0: Degree 0 SH coefficients RGB (N, 3) float32 (required if first arg is means)
    :param shN: Optional higher degree SH coefficients (N, K, 3) float32
    :param validate: Whether to validate inputs
    :returns: Complete compressed PLY file as bytes

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

    :param data_or_means: Either a GSData object or means array (N, 3) float32
    :param scales: Gaussian scales (N, 3) float32 (required if first arg is means)
    :param quats: Gaussian quaternions (N, 4) float32 (required if first arg is means)
    :param opacities: Gaussian opacities (N,) float32 (required if first arg is means)
    :param sh0: Degree 0 SH coefficients RGB (N, 3) float32 (required if first arg is means)
    :param shN: Optional higher degree SH coefficients (N, K, 3) float32
    :param validate: Whether to validate inputs
    :returns: Tuple containing header_bytes (PLY header as bytes), chunk_bounds (Chunk boundary array (num_chunks, 18) float32),
              packed_data (Main compressed data array (N, 4) uint32), packed_sh (Optional compressed SH data array uint8)

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
    data: "GSData | GSTensor | np.ndarray",  # noqa: F821
    scales: np.ndarray | None = None,
    quats: np.ndarray | None = None,
    opacities: np.ndarray | None = None,
    sh0: np.ndarray | None = None,
    shN: np.ndarray | None = None,  # noqa: N803
    compressed: bool = False,
    validate: bool = True,
) -> None:
    """Write Gaussian splatting PLY file with automatic optimization.

    The helper accepts either a :class:`gsply.GSData` instance (recommended),
    a :class:`gsply.GSTensor` instance (converted to GSData automatically),
    or the individual Gaussian arrays.  When `_base` is available the writer
    streams the consolidated buffer directly to disk; otherwise it performs a
    one-time consolidation before writing.  File format selection happens
    automatically: the compressed path is chosen when `compressed=True` or when
    the destination filename already ends with `.compressed.ply` /
    `.ply_compressed`.

    :param file_path: Output PLY file path (extension auto-adjusted if compressed=True)
    :param data: GSData object, GSTensor object, OR (N, 3) xyz positions array
    :param scales: (N, 3) scale parameters (required if data is array)
    :param quats: (N, 4) rotation quaternions (required if data is array)
    :param opacities: (N,) opacity values (required if data is array)
    :param sh0: (N, 3) DC spherical harmonics (required if data is array)
    :param shN: (N, K, 3) or (N, K*3) - Higher-order SH coefficients (optional)
    :param compressed: If True, write compressed format and auto-adjust extension
    :param validate: If True, validate input shapes (default True)

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
        >>> # GSTensor (converted to GSData automatically)
        >>> gstensor = plyread_gpu("input.compressed.ply", device="cuda")
        >>> plywrite("output.ply", gstensor, compressed=False)  # Uncompressed PLY
        >>>
        >>> # Individual arrays (converted + auto-consolidated)
        >>> plywrite("output.ply", means, scales, quats, opacities, sh0, shN)
        >>>
        >>> # Write compressed format
        >>> plywrite("output.ply", data, compressed=True)
    """
    from gsply.gsdata import GSData  # noqa: PLC0415

    file_path = Path(file_path)

    # Convert GSTensor to GSData if needed (lazy import to avoid torch import issues)
    try:
        from gsply.torch.gstensor import GSTensor  # noqa: PLC0415

        if isinstance(data, GSTensor):
            # Convert GSTensor to GSData (transfers to CPU)
            data = data.to_gsdata()
    except (ImportError, RuntimeError):
        # PyTorch not available or has import issues, skip GSTensor check
        pass

    # Convert individual arrays to GSData
    if not isinstance(data, GSData):
        # data is actually means array
        if any(x is None for x in [scales, quats, opacities, sh0]):
            raise ValueError(
                "When passing individual arrays, all of data (means), scales, quats, "
                "opacities, and sh0 must be provided"
            )
        # Create GSData without _base (will auto-consolidate below)
        # Automatically detect format from values (always returns valid format)
        from gsply.gsdata import _detect_format_from_values, _get_sh_order_format

        scales_format, opacities_format = _detect_format_from_values(scales, opacities)

        # Determine SH degree for format dict
        if shN is not None and shN.shape[1] > 0:
            if shN.ndim == 2:
                sh_bands = shN.shape[1] // 3
            else:
                sh_bands = shN.shape[1]
            from gsply.formats import SH_BANDS_TO_DEGREE

            sh_degree = SH_BANDS_TO_DEGREE.get(sh_bands, 0)
        else:
            sh_degree = 0

        # Create format dict (always provided)
        format_dict = _create_format_dict(
            scales=scales_format,
            opacities=opacities_format,
            sh0=DataFormat.SH0_SH,  # Assume SH format
            sh_order=_get_sh_order_format(sh_degree),
            means=DataFormat.MEANS_RAW,
            quats=DataFormat.QUATS_RAW,
        )

        data = GSData(
            means=data,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN if shN is not None else np.empty((data.shape[0], 0, 3), dtype=np.float32),
            _base=None,  # No _base for manually created data
            _format=format_dict,  # Auto-detected format (always provided)
        )

    # Note: Auto-consolidate was removed for better performance
    # The standard path already uses optimized JIT interleaving kernels,
    # so pre-consolidating adds overhead without benefit (71ms for 400K Gaussians)

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

        # Ensure data is in PLY format before writing compressed (log-scales, logit-opacities)
        # Check format flags and convert if needed
        scales_format = data._format.get("scales")
        opacities_format = data._format.get("opacities")

        # Convert to PLY format if not already in PLY format
        if scales_format != DataFormat.SCALES_PLY or opacities_format != DataFormat.OPACITIES_PLY:
            logger.debug(
                f"[PLY Write] Converting from {scales_format}/{opacities_format} to PLY format before writing"
            )
            # Use inplace=True for better performance (avoids 22MB copy for 400K Gaussians)
            # Safe since we're just writing the file and don't need to preserve original format
            data = data.normalize(inplace=True)

        # Extract arrays for compressed write (compressed write doesn't use GSData yet)
        means, scales, quats, opacities, sh0, shN = data.unpack()  # noqa: N806
        write_compressed(file_path, means, scales, quats, opacities, sh0, shN)
    else:
        # Ensure data is in PLY format before writing uncompressed (log-scales, logit-opacities)
        # Check format flags and convert if needed
        scales_format = data._format.get("scales")
        opacities_format = data._format.get("opacities")

        # Convert to PLY format if not already in PLY format
        if scales_format != DataFormat.SCALES_PLY or opacities_format != DataFormat.OPACITIES_PLY:
            logger.debug(
                f"[PLY Write] Converting from {scales_format}/{opacities_format} to PLY format before writing"
            )
            # Use inplace=True for better performance (avoids 22MB copy for 400K Gaussians)
            # Safe since we're just writing the file and don't need to preserve original format
            data = data.normalize(inplace=True)

        write_uncompressed(file_path, data, validate=validate)


__all__ = [
    "plywrite",
    "write_uncompressed",
    "write_compressed",
    "compress_to_bytes",
    "compress_to_arrays",
]
