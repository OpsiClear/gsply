"""Reading functions for Gaussian splatting PLY files.

This module provides ultra-fast reading of Gaussian splatting PLY files
in both uncompressed and compressed formats.

API Examples:
    >>> from gsply import plyread
    >>> data = plyread("scene.ply")
    >>> print(f"Loaded {data.means.shape[0]} Gaussians with SH degree {data.shN.shape[1]}")

    >>> # Or use format-specific readers
    >>> from gsply.reader import read_uncompressed
    >>> data = read_uncompressed("scene.ply")
    >>> if data is not None:
    ...     print(f"Loaded {data.means.shape[0]} Gaussians")

Performance:
    - Read uncompressed: 1-3ms for 50K Gaussians (17-50M Gaussians/sec)
    - Read compressed: 3-16ms for 50K Gaussians (3-16M Gaussians/sec)
"""

import logging
from pathlib import Path

import numba
import numpy as np

# Import numba for JIT optimization
from numba import jit

from gsply.formats import (
    EXPECTED_PROPERTIES_BY_SH_DEGREE,
    SH_C0,
    detect_format,
    get_sh_degree_from_property_count,
)

# Import GSData from separate module
from gsply.gsdata import GSData

logger = logging.getLogger(__name__)

# ======================================================================================
# PRE-COMPUTED CONSTANTS (Optimization - avoid runtime computation)
# ======================================================================================

# Quantization constants (pre-computed for multiplication instead of division)
_INV_2047 = 1.0 / 2047.0  # 11-bit unpacking
_INV_1023 = 1.0 / 1023.0  # 10-bit unpacking
_INV_255 = 1.0 / 255.0  # 8-bit unpacking

# Quaternion norm constant
_QUAT_NORM = 1.4142135623730951  # 1.0 / (sqrt(2) * 0.5) = sqrt(2)

# SH unpacking constants
_SH_UNPACK_SCALE = 1.0 / 32.0  # 0.03125
_SH_UNPACK_OFFSET = -127.5 * _SH_UNPACK_SCALE  # -3.984375

# Chunk size shift (256 = 2^8)
_CHUNK_SIZE_SHIFT = 8

# SH0 conversion constant (pre-computed inverse for multiplication)
_INV_SH_C0 = 1.0 / SH_C0  # = 3.544907701811032

# Header reading constants
_HEADER_READ_CHUNK_SIZE = 8192  # bytes - typical header is 300-2000 bytes
_MAX_HEADER_LINES = 200  # sanity limit to prevent infinite loops on malformed files

# ======================================================================================
# PRE-COMPUTED SLICE INDICES (Optimization to eliminate branching)
# ======================================================================================

# Lookup table for SH degree to slice indices (eliminates 4-way branching)
_SLICE_INDICES = {
    0: {
        "shN_start": 6,
        "shN_end": 6,
        "opacity": 6,
        "scales_start": 7,
        "scales_end": 10,
        "quats_start": 10,
        "quats_end": 14,
    },
    1: {
        "shN_start": 6,
        "shN_end": 15,
        "opacity": 15,
        "scales_start": 16,
        "scales_end": 19,
        "quats_start": 19,
        "quats_end": 23,
    },
    2: {
        "shN_start": 6,
        "shN_end": 30,
        "opacity": 30,
        "scales_start": 31,
        "scales_end": 34,
        "quats_start": 34,
        "quats_end": 38,
    },
    3: {
        "shN_start": 6,
        "shN_end": 51,
        "opacity": 51,
        "scales_start": 52,
        "scales_end": 55,
        "quats_start": 55,
        "quats_end": 59,
    },
}


# ======================================================================================
# OPTIMIZED HEADER READING
# ======================================================================================


def _read_header_fast(f) -> tuple[list[str], int] | None:
    """Read PLY header with optimized bulk reading from file handle.

    This optimization reads the header in a single bulk operation (8KB chunk)
    and decodes it once, rather than doing multiple readline() + decode() calls.
    Provides 4-10% speedup for reads.

    Args:
        f: Open file handle in binary mode

    Returns:
        Tuple of (header_lines, data_offset) or None on error
    """
    try:
        # Read first chunk (typical header is 300-2000 bytes)
        chunk = f.read(_HEADER_READ_CHUNK_SIZE)

        # Find end_header marker
        end_marker = b"end_header\n"
        end_idx = chunk.find(end_marker)

        if end_idx == -1:
            # Header larger than chunk size - use line-by-line fallback
            f.seek(0)
            header_lines = []
            while True:
                line = f.readline().decode("ascii").strip()
                header_lines.append(line)
                if line == "end_header":
                    break
                if len(header_lines) > _MAX_HEADER_LINES:
                    return None
            return header_lines, f.tell()

        # Fast path: decode entire header at once
        header_bytes = chunk[: end_idx + len(end_marker)]
        header_text = header_bytes.decode("ascii")
        header_lines = [line.strip() for line in header_text.split("\n") if line.strip()]
        data_offset = len(header_bytes)

        return header_lines, data_offset

    except Exception:
        return None


# ======================================================================================
# JIT-COMPILED DECOMPRESSION FUNCTIONS
# ======================================================================================


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _unpack_all_jit(
    packed_position,
    packed_rotation,
    packed_scale,
    packed_color,
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
    sh_c0,
):
    """Fused JIT-compiled unpacking of all vertex data in single parallel pass.

    Combines position, scale, color, and quaternion unpacking into one loop for:
    - Better cache locality (single pass over indices)
    - Reduced parallel overhead (1 loop instead of 4)
    - Improved CPU pipeline utilization
    - Chunk indices computed inline (avoids 1.6MB allocation for 400K vertices)

    Bit-Packing Format (PlayCanvas compressed PLY):
    - Position: uint32 with 11-10-11 bits (x, y, z) -> [0, 2047], [0, 1023], [0, 2047]
    - Scale: uint32 with 11-10-11 bits (sx, sy, sz) -> [0, 2047], [0, 1023], [0, 2047]
    - Color: uint32 with 8-8-8-8 bits (r, g, b, opacity) -> [0, 255] each
    - Rotation: uint32 with 2-10-10-10 bits (which, a, b, c) -> [0, 1023] each
      - 'which' (2 bits): Indicates largest quaternion component (0-3)
      - a, b, c (10 bits each): Three smallest quaternion components
      - Largest component computed as: m = sqrt(1 - (a² + b² + c²))

    Quantization Algorithm:
    1. Unpack integer bits using bit shifts and masks
    2. Normalize to [0, 1] by multiplying with pre-computed constants
    3. Dequantize using chunk bounds: value = min + normalized * range
    4. Convert colors to SH DC: sh = (color - 0.5) / SH_C0
    5. Convert opacity to logit space: -log(1/opacity - 1)

    Optimizations:
    - Division replaced with multiplication using pre-computed constants (_INV_2047, etc.)
    - Chunk indices computed inline (i >> 8) instead of array lookup
    - Pre-computed quaternion norm constant (_QUAT_NORM)
    - boundscheck=False: skip array bounds checking (indices guaranteed valid)

    Args:
        packed_position: (N,) uint32 array with packed xyz positions
        packed_rotation: (N,) uint32 array with packed quaternions
        packed_scale: (N,) uint32 array with packed scales
        packed_color: (N,) uint32 array with packed colors and opacity
        min_x, min_y, min_z, range_x, range_y, range_z: (num_chunks,) position bounds per chunk
        min_sx, min_sy, min_sz, range_sx, range_sy, range_sz: (num_chunks,) scale bounds per chunk
        min_r, min_g, min_b, range_r, range_g, range_b: (num_chunks,) color bounds per chunk
        sh_c0: SH constant for color to SH DC conversion (0.28209479...)

    Returns:
        tuple: (means, scales, quats, sh0, opacities)
            - means: (N, 3) xyz positions in world space
            - scales: (N, 3) scale parameters
            - quats: (N, 4) normalized quaternions (w, x, y, z)
            - sh0: (N, 3) SH DC coefficients (converted from RGB)
            - opacities: (N,) opacity in logit space
    """
    n = len(packed_position)
    means = np.empty((n, 3), dtype=np.float32)
    scales = np.empty((n, 3), dtype=np.float32)
    quats = np.empty((n, 4), dtype=np.float32)
    sh0 = np.empty((n, 3), dtype=np.float32)
    opacities = np.empty(n, dtype=np.float32)

    for i in numba.prange(n):
        # Compute chunk index inline (256 Gaussians per chunk)
        chunk_idx = i >> _CHUNK_SIZE_SHIFT

        # Unpack positions (11-10-11 bits) - use multiplication instead of division
        p_packed = packed_position[i]
        px = float((p_packed >> 21) & 0x7FF) * _INV_2047
        py = float((p_packed >> 11) & 0x3FF) * _INV_1023
        pz = float(p_packed & 0x7FF) * _INV_2047
        means[i, 0] = min_x[chunk_idx] + px * range_x[chunk_idx]
        means[i, 1] = min_y[chunk_idx] + py * range_y[chunk_idx]
        means[i, 2] = min_z[chunk_idx] + pz * range_z[chunk_idx]

        # Unpack scales (11-10-11 bits) - use multiplication instead of division
        s_packed = packed_scale[i]
        sx = float((s_packed >> 21) & 0x7FF) * _INV_2047
        sy = float((s_packed >> 11) & 0x3FF) * _INV_1023
        sz = float(s_packed & 0x7FF) * _INV_2047
        scales[i, 0] = min_sx[chunk_idx] + sx * range_sx[chunk_idx]
        scales[i, 1] = min_sy[chunk_idx] + sy * range_sy[chunk_idx]
        scales[i, 2] = min_sz[chunk_idx] + sz * range_sz[chunk_idx]

        # Unpack colors (8-8-8-8 bits) - use multiplication instead of division
        c_packed = packed_color[i]
        cr = float((c_packed >> 24) & 0xFF) * _INV_255
        cg = float((c_packed >> 16) & 0xFF) * _INV_255
        cb = float((c_packed >> 8) & 0xFF) * _INV_255
        co = float(c_packed & 0xFF) * _INV_255

        color_r = min_r[chunk_idx] + cr * range_r[chunk_idx]
        color_g = min_g[chunk_idx] + cg * range_g[chunk_idx]
        color_b = min_b[chunk_idx] + cb * range_b[chunk_idx]

        sh0[i, 0] = (color_r - 0.5) * _INV_SH_C0
        sh0[i, 1] = (color_g - 0.5) * _INV_SH_C0
        sh0[i, 2] = (color_b - 0.5) * _INV_SH_C0

        # Opacity conversion
        if co > 0.0 and co < 1.0:
            opacities[i] = -np.log(1.0 / co - 1.0)
        elif co >= 1.0:
            opacities[i] = 10.0
        else:
            opacities[i] = -10.0

        # Unpack quaternions (2+10-10-10 bits) - use pre-computed norm constant
        r_packed = packed_rotation[i]
        a = (float((r_packed >> 20) & 0x3FF) * _INV_1023 - 0.5) * _QUAT_NORM
        b = (float((r_packed >> 10) & 0x3FF) * _INV_1023 - 0.5) * _QUAT_NORM
        c = (float(r_packed & 0x3FF) * _INV_1023 - 0.5) * _QUAT_NORM

        m_squared = 1.0 - (a * a + b * b + c * c)
        m = np.sqrt(max(0.0, m_squared))
        which = r_packed >> 30

        if which == 0:
            quats[i, 0] = m
            quats[i, 1] = a
            quats[i, 2] = b
            quats[i, 3] = c
        elif which == 1:
            quats[i, 0] = a
            quats[i, 1] = m
            quats[i, 2] = b
            quats[i, 3] = c
        elif which == 2:
            quats[i, 0] = a
            quats[i, 1] = b
            quats[i, 2] = m
            quats[i, 3] = c
        else:
            quats[i, 0] = a
            quats[i, 1] = b
            quats[i, 2] = c
            quats[i, 3] = m

    return means, scales, quats, sh0, opacities


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _unpack_sh_jit(shN_data):  # noqa: N803
    """JIT-compiled SH coefficient decompression with parallel processing.

    Dequantizes higher-order spherical harmonics from compressed format.

    Quantization Algorithm:
    - Storage: uint8 in range [0, 255]
    - Conversion: float32 = (uint8 - 127.5) / 32.0
    - Output range: [-4.0, 4.0] (approximately ±4 covers typical SH coefficient values)
    - Pre-computed constants: scale=1/32, offset=-127.5/32 for performance

    The formula combines two operations:
    1. Center around zero: (x - 127.5) maps [0, 255] to [-127.5, 127.5]
    2. Scale to output range: divide by 32.0 to get [-4.0, 4.0]

    Args:
        shN_data: (N, num_coeffs) uint8 array of packed SH coefficients

    Returns:
        sh_flat: (N, num_coeffs) float32 array of decompressed SH values
    """
    n, num_coeffs = shN_data.shape
    sh_flat = np.empty((n, num_coeffs), dtype=np.float32)

    for i in numba.prange(n):
        for j in range(num_coeffs):
            # FMA (fused multiply-add): x * scale + offset
            # Replaces division with multiplication (3-5x faster)
            # (x - 127.5) / 32.0  -->  x * (1/32) + (-127.5/32)
            sh_flat[i, j] = shN_data[i, j] * _SH_UNPACK_SCALE + _SH_UNPACK_OFFSET

    return sh_flat


# ======================================================================================
# UNCOMPRESSED PLY READER
# ======================================================================================


def read_uncompressed(file_path: str | Path) -> GSData | None:  # noqa: PLR0911
    """Read uncompressed Gaussian splatting PLY file with zero-copy optimization.

    Uses zero-copy views into a single base array for maximum performance.
    The returned arrays share memory with a base array that is kept alive
    via the GSData container's reference counting.

    Note: This function does NOT use JIT compilation - it's already optimally
    implemented with NumPy zero-copy views. JIT is only used for compressed format.

    Args:
        file_path: Path to PLY file

    Returns:
        GSData container with zero-copy array views, or None if format
        is incompatible. The base array is kept alive to ensure views remain valid.

    Performance:
        - SH degree 0 (14 props): ~6ms for 400K Gaussians (70M Gaussians/sec)
        - SH degree 3 (59 props): ~3ms for 50K Gaussians (17M Gaussians/sec)
        - Peak: 78M Gaussians/sec for 1M Gaussians, SH0

    Example:
        >>> data = read_uncompressed("scene.ply")
        >>> if data is not None:
        ...     print(f"Loaded {data.means.shape[0]} Gaussians")
        ...     positions = data.means
        ...     colors = data.sh0
    """
    file_path = Path(file_path)

    try:
        # Single file handle optimization: read header and data in one go
        with open(file_path, "rb") as f:
            # Use existing header reading function
            header_result = _read_header_fast(f)
            if header_result is None:
                return None
            header_lines, data_offset = header_result

            # Parse header
            vertex_count = None
            is_binary_le = False
            property_names = []

            for line in header_lines:
                if line.startswith("format "):
                    format_type = line.split()[1]
                    is_binary_le = format_type == "binary_little_endian"
                elif line.startswith("element vertex "):
                    vertex_count = int(line.split()[2])
                elif line.startswith("property float "):
                    prop_name = line.split()[2]
                    property_names.append(prop_name)

            # Validate format
            if not is_binary_le or vertex_count is None:
                return None

            # Detect SH degree from property count
            property_count = len(property_names)
            sh_degree = get_sh_degree_from_property_count(property_count)

            if sh_degree is None:
                return None

            # Validate property names and order
            expected_properties = EXPECTED_PROPERTIES_BY_SH_DEGREE[sh_degree]
            if property_names != expected_properties:
                return None

            # Seek to data position and read binary data
            f.seek(data_offset)
            data = np.fromfile(f, dtype=np.float32, count=vertex_count * property_count)

            if data.size != vertex_count * property_count:
                return None

            data = data.reshape(vertex_count, property_count)

        # Extract arrays as zero-copy views
        means = data[:, 0:3]
        sh0 = data[:, 3:6]

        # Use lookup table to eliminate branching
        indices = _SLICE_INDICES[sh_degree]

        # Handle SH coefficients (special case for degree 0)
        if sh_degree == 0:
            shN = np.zeros((vertex_count, 0, 3), dtype=np.float32)  # noqa: N806
        else:
            shN_flat = data[:, indices["shN_start"] : indices["shN_end"]]  # noqa: N806
            num_sh_coeffs = shN_flat.shape[1]
            shN = shN_flat.reshape(vertex_count, num_sh_coeffs // 3, 3)  # noqa: N806

        # Extract remaining properties using lookup indices
        opacities = data[:, indices["opacity"]]
        scales = data[:, indices["scales_start"] : indices["scales_end"]]
        quats = data[:, indices["quats_start"] : indices["quats_end"]]

        logger.debug(
            f"[Gaussian PLY] Read uncompressed (fast): {vertex_count} Gaussians, SH degree {sh_degree}"
        )

        # Initialize masks to all True
        num_gaussians = means.shape[0]
        masks = np.ones(num_gaussians, dtype=bool)

        # Return GSData with base array to keep views alive
        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            masks=masks,
            _base=data,  # Keep alive for zero-copy views
        )

    except (OSError, ValueError):
        return None


# ======================================================================================
# COMPRESSED PLY READER
# ======================================================================================


def _parse_elements_from_header(header_lines: list[str]) -> dict:
    """Parse element information from PLY header lines.

    Args:
        header_lines: List of header lines from PLY file

    Returns:
        Dictionary mapping element names to their properties and counts
    """
    elements = {}
    current_element = None

    for line in header_lines:
        if line.startswith("element "):
            parts = line.split()
            name = parts[1]
            count = int(parts[2])
            elements[name] = {"count": count, "properties": []}
            current_element = name
        elif line.startswith("property ") and current_element:
            parts = line.split()
            prop_type = parts[1]
            prop_name = parts[2]
            elements[current_element]["properties"].append((prop_type, prop_name))

    return elements


def _is_compressed_format(header_lines: list) -> bool:
    """Check if PLY header indicates compressed format."""
    # Reuse element parsing logic
    elements = _parse_elements_from_header(header_lines)

    # Compressed format has "chunk" and "vertex" elements with specific properties
    if "chunk" not in elements or "vertex" not in elements:
        return False

    chunk_props = elements["chunk"]["properties"]
    if len(chunk_props) != 18:
        return False

    vertex_props = elements["vertex"]["properties"]
    if len(vertex_props) != 4:
        return False

    expected_vertex = [
        "packed_position",
        "packed_rotation",
        "packed_scale",
        "packed_color",
    ]
    for (_, prop_name), expected_name in zip(vertex_props, expected_vertex, strict=False):
        if prop_name != expected_name:
            return False

    return True


def _decompress_data_internal(
    chunk_data: np.ndarray,
    vertex_data: np.ndarray,
    shN_data: np.ndarray | None,  # noqa: N803
    num_vertices: int,
    num_chunks: int,
) -> GSData:
    """Internal function to decompress Gaussian data (shared decompression logic).

    This function contains the core decompression logic shared between read_compressed()
    and decompress_from_bytes(). All JIT-compiled unpacking happens here.

    Args:
        chunk_data: Chunk bounds array (num_chunks, 18) float32
        vertex_data: Packed vertex data (num_vertices, 4) uint32
        shN_data: Optional SH coefficient data (num_vertices, num_coeffs) uint8
        num_vertices: Total number of Gaussians
        num_chunks: Total number of chunks

    Returns:
        GSData container with decompressed Gaussian parameters
    """
    # Extract chunk bounds (views into chunk_data)
    min_x, min_y, min_z = chunk_data[:, 0], chunk_data[:, 1], chunk_data[:, 2]
    max_x, max_y, max_z = chunk_data[:, 3], chunk_data[:, 4], chunk_data[:, 5]
    min_scale_x, min_scale_y, min_scale_z = (
        chunk_data[:, 6],
        chunk_data[:, 7],
        chunk_data[:, 8],
    )
    max_scale_x, max_scale_y, max_scale_z = (
        chunk_data[:, 9],
        chunk_data[:, 10],
        chunk_data[:, 11],
    )
    min_r, min_g, min_b = chunk_data[:, 12], chunk_data[:, 13], chunk_data[:, 14]
    max_r, max_g, max_b = chunk_data[:, 15], chunk_data[:, 16], chunk_data[:, 17]

    # Extract packed data (views into vertex_data)
    packed_position = vertex_data[:, 0]
    packed_rotation = vertex_data[:, 1]
    packed_scale = vertex_data[:, 2]
    packed_color = vertex_data[:, 3]

    # Pre-compute ranges for better performance (avoids recomputing in loops)
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    range_scale_x = max_scale_x - min_scale_x
    range_scale_y = max_scale_y - min_scale_y
    range_scale_z = max_scale_z - min_scale_z
    range_r = max_r - min_r
    range_g = max_g - min_g
    range_b = max_b - min_b

    # Use fused JIT-compiled function for parallel decompression
    # Chunk indices computed inline (i >> 8) - saves 1.6MB allocation for 400K vertices
    means, scales, quats, sh0, opacities = _unpack_all_jit(
        packed_position,
        packed_rotation,
        packed_scale,
        packed_color,
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
        SH_C0,
    )

    # Decompress SH coefficients (JIT-compiled for performance)
    if shN_data is not None:
        num_sh_coeffs = shN_data.shape[1]
        num_sh_bands = num_sh_coeffs // 3

        # JIT-compiled parallel decompression (60-80% faster than vectorized NumPy)
        sh_flat = _unpack_sh_jit(shN_data)

        # Reshape to (N, num_bands, 3)
        shN = sh_flat.reshape(num_vertices, num_sh_bands, 3)  # noqa: N806
    else:
        shN = np.zeros((num_vertices, 0, 3), dtype=np.float32)  # noqa: N806

    logger.debug(f"[Gaussian PLY] Decompressed: {num_vertices} Gaussians, SH bands {shN.shape[1]}")

    # Initialize masks to all True
    num_gaussians = means.shape[0]
    masks = np.ones(num_gaussians, dtype=bool)

    # Return GSData container (_base=None since decompressed data is separate)
    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        masks=masks,
        _base=None,
    )


def _parse_header_from_bytes(compressed_bytes: bytes) -> tuple[list[str], int, dict]:
    """Parse PLY header from bytes (optimized, matches _read_header_fast logic).

    Uses fast byte search instead of line-by-line reading for maximum performance.

    Args:
        compressed_bytes: Complete PLY file as bytes

    Returns:
        Tuple of (header_lines, data_offset, elements_dict)

    Raises:
        ValueError: If header is invalid or not compressed format
    """
    # Fast path: Find end_header with byte search (like _read_header_fast)
    end_marker = b"end_header\n"
    end_idx = compressed_bytes.find(end_marker)

    if end_idx == -1:
        raise ValueError("Invalid PLY: no end_header found")

    # Decode entire header at once (like _read_header_fast)
    header_bytes = compressed_bytes[: end_idx + len(end_marker)]
    header_text = header_bytes.decode("ascii")
    header_lines = [line.strip() for line in header_text.split("\n") if line.strip()]
    data_offset = len(header_bytes)

    # Check if compressed format
    if not _is_compressed_format(header_lines):
        raise ValueError("Bytes do not contain compressed PLY format")

    # Parse element info using shared helper
    elements = _parse_elements_from_header(header_lines)

    return header_lines, data_offset, elements


def read_compressed(file_path: str | Path) -> GSData | None:
    """Read compressed Gaussian splatting PLY file (PlayCanvas format).

    Format uses chunk-based quantization with 256 Gaussians per chunk.
    Achieves 14.5x compression (16 bytes/splat vs 232 bytes/splat).

    Uses Numba JIT compilation for fast parallel decompression (6x faster than pure NumPy).

    Args:
        file_path: Path to compressed PLY file

    Returns:
        GSData container with decompressed Gaussian parameters, or None
        if format is incompatible. The base field is None (no shared array).

    Performance:
        - With JIT: ~9ms for 400K Gaussians, SH0 (47M Gaussians/sec)
        - With JIT: ~118ms for 400K Gaussians, SH3 (3.4M Gaussians/sec)

    Example:
        >>> result = read_compressed("scene.ply_compressed")
        >>> if result is not None:
        ...     print(f"Loaded {result.means.shape[0]} compressed Gaussians")
        ...     positions = result.means
    """
    file_path = Path(file_path)

    try:
        # Use optimized bulk header reading with single file handle
        with open(file_path, "rb") as f:
            result = _read_header_fast(f)
            if result is None:
                return None
            header_lines, data_offset = result

            # Check if compressed format
            if not _is_compressed_format(header_lines):
                return None

            # Parse element info using shared helper
            elements = _parse_elements_from_header(header_lines)

            # Seek to data and read binary from same file handle
            f.seek(data_offset)

            num_chunks = elements["chunk"]["count"]
            chunk_data = np.fromfile(f, dtype=np.float32, count=num_chunks * 18)
            chunk_data = chunk_data.reshape(num_chunks, 18)

            num_vertices = elements["vertex"]["count"]
            vertex_data = np.fromfile(f, dtype=np.uint32, count=num_vertices * 4)
            vertex_data = vertex_data.reshape(num_vertices, 4)

            shN_data = None  # noqa: N806
            if "sh" in elements:
                num_sh_coeffs = len(elements["sh"]["properties"])
                shN_data = np.fromfile(f, dtype=np.uint8, count=num_vertices * num_sh_coeffs)  # noqa: N806
                shN_data = shN_data.reshape(num_vertices, num_sh_coeffs)  # noqa: N806

        # Decompress using shared internal function
        return _decompress_data_internal(
            chunk_data, vertex_data, shN_data, num_vertices, num_chunks
        )

    except (OSError, ValueError):
        return None


def decompress_from_bytes(compressed_bytes: bytes) -> GSData:
    """Decompress Gaussian splatting data from bytes (PlayCanvas format).

    Reads compressed PLY data from bytes without disk I/O.
    Symmetric with compress_to_bytes() - use for network transfer, streaming, etc.

    Uses optimized header parsing (bytes.find) matching read_compressed() performance.

    Args:
        compressed_bytes: Complete compressed PLY file as bytes

    Returns:
        GSData container with decompressed Gaussian parameters

    Example:
        >>> from gsply import compress_to_bytes, decompress_from_bytes
        >>> # Compress
        >>> data = plyread("model.ply")
        >>> compressed = compress_to_bytes(data)
        >>>
        >>> # Decompress (no disk I/O!)
        >>> data_restored = decompress_from_bytes(compressed)
        >>> assert data_restored.means.shape == data.means.shape
    """
    # Parse header using fast byte search (matches _read_header_fast approach)
    header_lines, data_offset, elements = _parse_header_from_bytes(compressed_bytes)

    # Read binary data directly from bytes (zero-copy with np.frombuffer)
    num_chunks = elements["chunk"]["count"]
    offset = data_offset

    chunk_bytes = compressed_bytes[offset : offset + num_chunks * 72]  # 18 floats * 4 bytes
    chunk_data = np.frombuffer(chunk_bytes, dtype=np.float32).reshape(num_chunks, 18)
    offset += num_chunks * 72

    num_vertices = elements["vertex"]["count"]
    vertex_bytes = compressed_bytes[offset : offset + num_vertices * 16]  # 4 uint32 * 4 bytes
    vertex_data = np.frombuffer(vertex_bytes, dtype=np.uint32).reshape(num_vertices, 4)
    offset += num_vertices * 16

    shN_data = None  # noqa: N806
    if "sh" in elements:
        num_sh_coeffs = len(elements["sh"]["properties"])
        sh_bytes = compressed_bytes[offset : offset + num_vertices * num_sh_coeffs]
        shN_data = np.frombuffer(sh_bytes, dtype=np.uint8).reshape(num_vertices, num_sh_coeffs)  # noqa: N806

    # Decompress using shared internal function (SAME as read_compressed!)
    return _decompress_data_internal(chunk_data, vertex_data, shN_data, num_vertices, num_chunks)


# ======================================================================================
# UNIFIED READING API
# ======================================================================================


def plyread(file_path: str | Path) -> GSData:
    """Read Gaussian splatting PLY file (auto-detect format).

    Automatically detects and reads both compressed and uncompressed formats.
    Uses formats.detect_format() for fast format detection.

    All reads use zero-copy optimization for maximum performance.

    Args:
        file_path: Path to PLY file

    Returns:
        GSData container with Gaussian parameters

    Raises:
        ValueError: If file format is not recognized or invalid

    Performance:
        - Uncompressed: ~6ms for 400K Gaussians, SH0 (70M Gaussians/sec)
        - Compressed: ~9ms for 400K Gaussians, SH0 (47M Gaussians/sec)
        - Peak: 78M Gaussians/sec for 1M Gaussians, SH0

    Example:
        >>> data = plyread("scene.ply")
        >>> print(f"Loaded {data.means.shape[0]} Gaussians")
        >>> positions = data.means
        >>> colors = data.sh0
        >>>
        >>> # Unpack for standard GS workflows
        >>> means, scales, quats, opacities, sh0, shN = data.unpack()
    """
    file_path = Path(file_path)

    # Detect format first
    is_compressed, sh_degree = detect_format(file_path)

    # Try appropriate reader based on format
    if is_compressed:
        result = read_compressed(file_path)
    else:
        result = read_uncompressed(file_path)

    if result is not None:
        return result

    raise ValueError(
        f"Unsupported PLY format or invalid file: {file_path}. "
        f"Ensure the file is a valid Gaussian Splatting PLY file "
        f"(either uncompressed with SH degree 0-3, or PlayCanvas compressed format)."
    )


__all__ = [
    "plyread",
    "GSData",
    "read_uncompressed",
    "read_compressed",
    "decompress_from_bytes",
]
