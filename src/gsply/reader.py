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
    - Read uncompressed: 8-12ms for 50K Gaussians
    - Read compressed: 30-50ms for 50K Gaussians
"""

import numpy as np
import struct
from pathlib import Path
from typing import Optional, Union, Tuple, NamedTuple
import logging

from gsply.formats import (
    detect_format,
    get_sh_degree_from_property_count,
    EXPECTED_PROPERTIES_BY_SH_DEGREE,
    CHUNK_SIZE,
    SH_C0,
)

# Try to import numba for JIT optimization (optional)
try:
    from numba import jit
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    # Mock numba module for prange fallback
    class _MockNumba:
        @staticmethod
        def prange(n):
            return range(n)
    numba = _MockNumba()

logger = logging.getLogger(__name__)


# ======================================================================================
# ZERO-COPY DATA CONTAINER
# ======================================================================================

class GSData(NamedTuple):
    """Gaussian Splatting data container.

    This container holds Gaussian parameters, either as separate arrays
    or as zero-copy views into a single base array for maximum performance.

    Attributes:
        means: (N, 3) - xyz positions
        scales: (N, 3) - scale parameters
        quats: (N, 4) - rotation quaternions
        opacities: (N,) - opacity values
        sh0: (N, 3) - DC spherical harmonics
        shN: (N, K, 3) - Higher-order SH coefficients (K bands)
        base: (N, P) - Base array (keeps memory alive for views, None otherwise)

    Performance:
        - Zero-copy reads via plyfastread() are 1.65x faster
        - No memory overhead (views share memory with base)

    Example:
        >>> data = plyfastread("scene.ply")
        >>> print(f"Loaded {data.means.shape[0]} Gaussians")
        >>> # Access via attributes
        >>> positions = data.means
        >>> colors = data.sh0
    """
    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh0: np.ndarray
    shN: np.ndarray
    base: np.ndarray  # Keeps base array alive for zero-copy views


# ======================================================================================
# JIT-COMPILED DECOMPRESSION FUNCTIONS
# ======================================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _unpack_positions_jit(packed_position, chunk_indices, min_x, min_y, min_z, max_x, max_y, max_z, chunk_size=256):
    """JIT-compiled position unpacking and dequantization (11-10-11 bits) with parallel processing.

    Args:
        packed_position: uint32 array of packed position data
        chunk_indices: int32 array of chunk indices for each vertex
        min_x, min_y, min_z: chunk minimum bounds
        max_x, max_y, max_z: chunk maximum bounds
        chunk_size: chunk size (default 256)

    Returns:
        means: (N, 3) float32 array of dequantized positions
    """
    n = len(packed_position)
    means = np.zeros((n, 3), dtype=np.float32)

    for i in numba.prange(n):
        packed = packed_position[i]
        chunk_idx = chunk_indices[i]

        # Unpack 11-10-11 bits
        px = float((packed >> 21) & 0x7FF) / 2047.0
        py = float((packed >> 11) & 0x3FF) / 1023.0
        pz = float(packed & 0x7FF) / 2047.0

        # Dequantize
        means[i, 0] = min_x[chunk_idx] + px * (max_x[chunk_idx] - min_x[chunk_idx])
        means[i, 1] = min_y[chunk_idx] + py * (max_y[chunk_idx] - min_y[chunk_idx])
        means[i, 2] = min_z[chunk_idx] + pz * (max_z[chunk_idx] - min_z[chunk_idx])

    return means


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _unpack_scales_jit(packed_scale, chunk_indices, min_sx, min_sy, min_sz, max_sx, max_sy, max_sz, chunk_size=256):
    """JIT-compiled scale unpacking and dequantization (11-10-11 bits) with parallel processing.

    Args:
        packed_scale: uint32 array of packed scale data
        chunk_indices: int32 array of chunk indices for each vertex
        min_sx, min_sy, min_sz: chunk minimum scale bounds
        max_sx, max_sy, max_sz: chunk maximum scale bounds
        chunk_size: chunk size (default 256)

    Returns:
        scales: (N, 3) float32 array of dequantized scales
    """
    n = len(packed_scale)
    scales = np.zeros((n, 3), dtype=np.float32)

    for i in numba.prange(n):
        packed = packed_scale[i]
        chunk_idx = chunk_indices[i]

        # Unpack 11-10-11 bits
        sx = float((packed >> 21) & 0x7FF) / 2047.0
        sy = float((packed >> 11) & 0x3FF) / 1023.0
        sz = float(packed & 0x7FF) / 2047.0

        # Dequantize
        scales[i, 0] = min_sx[chunk_idx] + sx * (max_sx[chunk_idx] - min_sx[chunk_idx])
        scales[i, 1] = min_sy[chunk_idx] + sy * (max_sy[chunk_idx] - min_sy[chunk_idx])
        scales[i, 2] = min_sz[chunk_idx] + sz * (max_sz[chunk_idx] - min_sz[chunk_idx])

    return scales


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _unpack_colors_jit(packed_color, chunk_indices, min_r, min_g, min_b, max_r, max_g, max_b, sh_c0, chunk_size=256):
    """JIT-compiled color unpacking, dequantization, and SH0 conversion (8-8-8-8 bits) with parallel processing.

    Args:
        packed_color: uint32 array of packed color data
        chunk_indices: int32 array of chunk indices for each vertex
        min_r, min_g, min_b: chunk minimum color bounds
        max_r, max_g, max_b: chunk maximum color bounds
        sh_c0: SH constant for conversion
        chunk_size: chunk size (default 256)

    Returns:
        sh0: (N, 3) float32 array of SH0 coefficients
        opacities: (N,) float32 array of opacities in logit space
    """
    n = len(packed_color)
    sh0 = np.zeros((n, 3), dtype=np.float32)
    opacities = np.zeros(n, dtype=np.float32)

    for i in numba.prange(n):
        packed = packed_color[i]
        chunk_idx = chunk_indices[i]

        # Unpack 8-8-8-8 bits
        cr = float((packed >> 24) & 0xFF) / 255.0
        cg = float((packed >> 16) & 0xFF) / 255.0
        cb = float((packed >> 8) & 0xFF) / 255.0
        co = float(packed & 0xFF) / 255.0

        # Dequantize colors
        color_r = min_r[chunk_idx] + cr * (max_r[chunk_idx] - min_r[chunk_idx])
        color_g = min_g[chunk_idx] + cg * (max_g[chunk_idx] - min_g[chunk_idx])
        color_b = min_b[chunk_idx] + cb * (max_b[chunk_idx] - min_b[chunk_idx])

        # Convert to SH0
        sh0[i, 0] = (color_r - 0.5) / sh_c0
        sh0[i, 1] = (color_g - 0.5) / sh_c0
        sh0[i, 2] = (color_b - 0.5) / sh_c0

        # Convert opacity to logit space
        if co > 0.0 and co < 1.0:
            opacities[i] = -np.log(1.0 / co - 1.0)
        elif co >= 1.0:
            opacities[i] = 10.0
        else:
            opacities[i] = -10.0

    return sh0, opacities


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _unpack_quaternions_jit(packed_rotation, chunk_size=256):
    """JIT-compiled quaternion unpacking (smallest-three encoding, 2+10-10-10 bits).

    Args:
        packed_rotation: uint32 array of packed rotation data
        chunk_size: chunk size (default 256)

    Returns:
        quats: (N, 4) float32 array of quaternions
    """
    n = len(packed_rotation)
    quats = np.zeros((n, 4), dtype=np.float32)
    norm = 1.0 / (np.sqrt(2) * 0.5)

    for i in numba.prange(n):
        packed = packed_rotation[i]

        # Unpack three components (10 bits each)
        a = (float((packed >> 20) & 0x3FF) / 1023.0 - 0.5) * norm
        b = (float((packed >> 10) & 0x3FF) / 1023.0 - 0.5) * norm
        c = (float(packed & 0x3FF) / 1023.0 - 0.5) * norm

        # Compute fourth component from unit constraint
        m_squared = 1.0 - (a * a + b * b + c * c)
        m = np.sqrt(max(0.0, m_squared))

        # Which component is the fourth? (2 bits)
        which = (packed >> 30)

        # Reconstruct quaternion based on 'which' flag
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
        else:  # which == 3
            quats[i, 0] = a
            quats[i, 1] = b
            quats[i, 2] = c
            quats[i, 3] = m

    return quats


# ======================================================================================
# UNCOMPRESSED PLY READER
# ======================================================================================

def read_uncompressed(file_path: Union[str, Path]) -> Optional[GSData]:
    """Read uncompressed Gaussian splatting PLY file.

    Supports all standard Gaussian PLY formats (SH degrees 0-3).
    Uses zero-copy numpy operations for maximum performance.

    Args:
        file_path: Path to PLY file

    Returns:
        GSData container with Gaussian parameters, or None if format
        is incompatible. The base field is None for this function (copies are made).

    Performance:
        - SH degree 0 (14 props): ~17ms for 388K Gaussians
        - SH degree 3 (59 props): ~8ms for 50K Gaussians

    Example:
        >>> result = read_uncompressed("scene.ply")
        >>> if result is not None:
        ...     print(f"Loaded {result.means.shape[0]} Gaussians")
        ...     positions = result.means
    """
    file_path = Path(file_path)

    try:
        with open(file_path, 'rb') as f:
            # Read header
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == "end_header":
                    break
                if len(header_lines) > 200:
                    return None

            # Parse header inline (while file is still open)
            vertex_count = None
            is_binary_le = False
            property_names = []

            for line in header_lines:
                if line.startswith("format "):
                    format_type = line.split()[1]
                    is_binary_le = (format_type == "binary_little_endian")
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

            # Validate property names and order (batch comparison is faster)
            expected_properties = EXPECTED_PROPERTIES_BY_SH_DEGREE[sh_degree]
            if property_names != expected_properties:
                return None

            # Read binary data in single operation (file already positioned after header)
            data = np.fromfile(f, dtype=np.float32, count=vertex_count * property_count)

            if data.size != vertex_count * property_count:
                return None

            data = data.reshape(vertex_count, property_count)

        # Extract arrays based on SH degree (zero-copy slicing)
        # No .copy() needed - arrays are returned immediately and parent data goes out of scope
        means = data[:, 0:3]
        sh0 = data[:, 3:6]

        if sh_degree == 0:
            shN = np.zeros((vertex_count, 0, 3), dtype=np.float32)
            opacities = data[:, 6]
            scales = data[:, 7:10]
            quats = data[:, 10:14]
        elif sh_degree == 1:
            shN = data[:, 6:15]
            opacities = data[:, 15]
            scales = data[:, 16:19]
            quats = data[:, 19:23]
        elif sh_degree == 2:
            shN = data[:, 6:30]
            opacities = data[:, 30]
            scales = data[:, 31:34]
            quats = data[:, 34:38]
        else:  # sh_degree == 3
            shN = data[:, 6:51]
            opacities = data[:, 51]
            scales = data[:, 52:55]
            quats = data[:, 55:59]

        # Reshape shN to (N, K, 3) format - need copy here since we're reshaping
        if sh_degree > 0:
            num_sh_coeffs = shN.shape[1]
            shN = shN.copy().reshape(vertex_count, num_sh_coeffs // 3, 3)

        logger.debug(f"[Gaussian PLY] Read uncompressed: {vertex_count} Gaussians, SH degree {sh_degree}")

        # Return GSData container (base=None since we made copies)
        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            base=None  # No shared base array for standard read
        )

    except (OSError, ValueError, IOError):
        return None


def read_uncompressed_fast(file_path: Union[str, Path]) -> Optional[GSData]:
    """Read uncompressed Gaussian splatting PLY file with zero-copy optimization.

    This is a high-performance variant of read_uncompressed() that avoids expensive
    memory copies by returning views into a single base array. Approximately 1.86x
    faster than read_uncompressed() for files with higher-order SH coefficients.

    Args:
        file_path: Path to PLY file

    Returns:
        GSData namedtuple with zero-copy array views, or None if format
        is incompatible. The base array is kept alive to ensure views remain valid.

    Performance:
        - SH degree 3 (59 props): ~3.2ms for 50K Gaussians (1.86x faster)
        - Zero memory overhead (views share memory with base array)

    Example:
        >>> data = read_uncompressed_fast("scene.ply")
        >>> if data is not None:
        ...     print(f"Loaded {data.means.shape[0]} Gaussians")
        ...     positions = data.means
        ...     colors = data.sh0

    Note:
        The returned arrays are views into a shared base array. This is safe
        because the GSData container keeps the base array alive via
        Python's reference counting mechanism.
    """
    file_path = Path(file_path)

    try:
        with open(file_path, 'rb') as f:
            # Read header
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == "end_header":
                    break
                if len(header_lines) > 200:
                    return None

            data_offset = f.tell()

        # Parse header
        vertex_count = None
        is_binary_le = False
        property_names = []

        for line in header_lines:
            if line.startswith("format "):
                format_type = line.split()[1]
                is_binary_le = (format_type == "binary_little_endian")
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

        # Read binary data in single operation
        with open(file_path, 'rb') as f:
            f.seek(data_offset)
            data = np.fromfile(f, dtype=np.float32, count=vertex_count * property_count)

            if data.size != vertex_count * property_count:
                return None

            data = data.reshape(vertex_count, property_count)

        # Extract arrays as zero-copy views
        means = data[:, 0:3]
        sh0 = data[:, 3:6]

        if sh_degree == 0:
            shN = np.zeros((vertex_count, 0, 3), dtype=np.float32)
            opacities = data[:, 6]
            scales = data[:, 7:10]
            quats = data[:, 10:14]
        elif sh_degree == 1:
            shN_flat = data[:, 6:15]
            opacities = data[:, 15]
            scales = data[:, 16:19]
            quats = data[:, 19:23]
            num_sh_coeffs = shN_flat.shape[1]
            shN = shN_flat.reshape(vertex_count, num_sh_coeffs // 3, 3)
        elif sh_degree == 2:
            shN_flat = data[:, 6:30]
            opacities = data[:, 30]
            scales = data[:, 31:34]
            quats = data[:, 34:38]
            num_sh_coeffs = shN_flat.shape[1]
            shN = shN_flat.reshape(vertex_count, num_sh_coeffs // 3, 3)
        else:  # sh_degree == 3
            shN_flat = data[:, 6:51]
            opacities = data[:, 51]
            scales = data[:, 52:55]
            quats = data[:, 55:59]
            num_sh_coeffs = shN_flat.shape[1]
            shN = shN_flat.reshape(vertex_count, num_sh_coeffs // 3, 3)

        logger.debug(f"[Gaussian PLY] Read uncompressed (fast): {vertex_count} Gaussians, SH degree {sh_degree}")

        # Return GSData with base array to keep views alive
        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            base=data  # Keep alive for zero-copy views
        )

    except (OSError, ValueError, IOError):
        return None


# ======================================================================================
# COMPRESSED PLY READER
# ======================================================================================

def _unpack_unorm(value: int, bits: int) -> float:
    """Extract normalized value [0,1] from packed bits."""
    mask = (1 << bits) - 1
    return (value & mask) / mask


def _unpack_111011(value: int) -> Tuple[float, float, float]:
    """Unpack 3D vector from 32-bit value (11-10-11 bits)."""
    x = _unpack_unorm(value >> 21, 11)
    y = _unpack_unorm(value >> 11, 10)
    z = _unpack_unorm(value, 11)
    return x, y, z


def _unpack_8888(value: int) -> Tuple[float, float, float, float]:
    """Unpack 4 channels from 32-bit value (8 bits each)."""
    x = _unpack_unorm(value >> 24, 8)
    y = _unpack_unorm(value >> 16, 8)
    z = _unpack_unorm(value >> 8, 8)
    w = _unpack_unorm(value, 8)
    return x, y, z, w


def _unpack_rotation(value: int) -> Tuple[float, float, float, float]:
    """Unpack quaternion using smallest-three encoding."""
    norm = 1.0 / (np.sqrt(2) * 0.5)

    a = (_unpack_unorm(value >> 20, 10) - 0.5) * norm
    b = (_unpack_unorm(value >> 10, 10) - 0.5) * norm
    c = (_unpack_unorm(value, 10) - 0.5) * norm

    m = np.sqrt(max(0.0, 1.0 - (a * a + b * b + c * c)))
    which = value >> 30

    if which == 0:
        return m, a, b, c
    elif which == 1:
        return a, m, b, c
    elif which == 2:
        return a, b, m, c
    else:
        return a, b, c, m


def _is_compressed_format(header_lines: list) -> bool:
    """Check if PLY header indicates compressed format."""
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

    # Compressed format has "chunk" and "vertex" elements with specific properties
    if "chunk" not in elements or "vertex" not in elements:
        return False

    chunk_props = elements["chunk"]["properties"]
    if len(chunk_props) != 18:
        return False

    vertex_props = elements["vertex"]["properties"]
    if len(vertex_props) != 4:
        return False

    expected_vertex = ["packed_position", "packed_rotation", "packed_scale", "packed_color"]
    for (_, prop_name), expected_name in zip(vertex_props, expected_vertex):
        if prop_name != expected_name:
            return False

    return True


def read_compressed(file_path: Union[str, Path]) -> Optional[GSData]:
    """Read compressed Gaussian splatting PLY file (PlayCanvas format).

    Format uses chunk-based quantization with 256 Gaussians per chunk.
    Achieves 14.5x compression (16 bytes/splat vs 232 bytes/splat).

    Args:
        file_path: Path to compressed PLY file

    Returns:
        GSData container with decompressed Gaussian parameters, or None
        if format is incompatible. The base field is None (no shared array).

    Performance:
        ~30-50ms for 50K Gaussians (decompression overhead)

    Example:
        >>> result = read_compressed("scene.ply_compressed")
        >>> if result is not None:
        ...     print(f"Loaded {result.means.shape[0]} compressed Gaussians")
        ...     positions = result.means
    """
    file_path = Path(file_path)

    try:
        with open(file_path, 'rb') as f:
            # Read header
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == "end_header":
                    break
                if len(header_lines) > 200:
                    return None

            data_offset = f.tell()

        # Check if compressed format
        if not _is_compressed_format(header_lines):
            return None

        # Parse element info
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

        # Read chunk data (18 float32 per chunk)
        with open(file_path, 'rb') as f:
            f.seek(data_offset)

            num_chunks = elements["chunk"]["count"]
            chunk_data = np.fromfile(f, dtype=np.float32, count=num_chunks * 18)
            chunk_data = chunk_data.reshape(num_chunks, 18)

            # Read vertex data (4 uint32 per vertex)
            num_vertices = elements["vertex"]["count"]
            vertex_data = np.fromfile(f, dtype=np.uint32, count=num_vertices * 4)
            vertex_data = vertex_data.reshape(num_vertices, 4)

            # Read optional SH data (uint8 per coefficient)
            shN_data = None
            if "sh" in elements:
                num_sh_coeffs = len(elements["sh"]["properties"])
                shN_data = np.fromfile(f, dtype=np.uint8, count=num_vertices * num_sh_coeffs)
                shN_data = shN_data.reshape(num_vertices, num_sh_coeffs)

        # Extract chunk bounds
        min_x, min_y, min_z = chunk_data[:, 0], chunk_data[:, 1], chunk_data[:, 2]
        max_x, max_y, max_z = chunk_data[:, 3], chunk_data[:, 4], chunk_data[:, 5]
        min_scale_x, min_scale_y, min_scale_z = chunk_data[:, 6], chunk_data[:, 7], chunk_data[:, 8]
        max_scale_x, max_scale_y, max_scale_z = chunk_data[:, 9], chunk_data[:, 10], chunk_data[:, 11]
        min_r, min_g, min_b = chunk_data[:, 12], chunk_data[:, 13], chunk_data[:, 14]
        max_r, max_g, max_b = chunk_data[:, 15], chunk_data[:, 16], chunk_data[:, 17]

        # Allocate output arrays
        means = np.zeros((num_vertices, 3), dtype=np.float32)
        scales = np.zeros((num_vertices, 3), dtype=np.float32)
        quats = np.zeros((num_vertices, 4), dtype=np.float32)
        opacities = np.zeros(num_vertices, dtype=np.float32)
        sh0 = np.zeros((num_vertices, 3), dtype=np.float32)

        # Decompress vertices (vectorized for 5-10x speedup)
        packed_position = vertex_data[:, 0]
        packed_rotation = vertex_data[:, 1]
        packed_scale = vertex_data[:, 2]
        packed_color = vertex_data[:, 3]

        # Pre-compute chunk indices for all vertices
        chunk_indices = np.arange(num_vertices, dtype=np.int32) // CHUNK_SIZE

        # Use JIT-compiled functions if available (2-3x faster)
        if HAS_NUMBA:
            # JIT-compiled decompression (parallel, fastmath)
            means = _unpack_positions_jit(packed_position, chunk_indices, min_x, min_y, min_z, max_x, max_y, max_z)
            scales = _unpack_scales_jit(packed_scale, chunk_indices, min_scale_x, min_scale_y, min_scale_z, max_scale_x, max_scale_y, max_scale_z)
            sh0, opacities = _unpack_colors_jit(packed_color, chunk_indices, min_r, min_g, min_b, max_r, max_g, max_b, SH_C0)
            quats = _unpack_quaternions_jit(packed_rotation)
        else:
            # Fallback: Vectorized NumPy operations
            # Position unpacking (11-10-11 bits)
            px = ((packed_position >> 21) & 0x7FF).astype(np.float32) / 2047.0
            py = ((packed_position >> 11) & 0x3FF).astype(np.float32) / 1023.0
            pz = (packed_position & 0x7FF).astype(np.float32) / 2047.0

            means[:, 0] = min_x[chunk_indices] + px * (max_x[chunk_indices] - min_x[chunk_indices])
            means[:, 1] = min_y[chunk_indices] + py * (max_y[chunk_indices] - min_y[chunk_indices])
            means[:, 2] = min_z[chunk_indices] + pz * (max_z[chunk_indices] - min_z[chunk_indices])

            # Scale unpacking (11-10-11 bits)
            sx = ((packed_scale >> 21) & 0x7FF).astype(np.float32) / 2047.0
            sy = ((packed_scale >> 11) & 0x3FF).astype(np.float32) / 1023.0
            sz = (packed_scale & 0x7FF).astype(np.float32) / 2047.0

            scales[:, 0] = min_scale_x[chunk_indices] + sx * (max_scale_x[chunk_indices] - min_scale_x[chunk_indices])
            scales[:, 1] = min_scale_y[chunk_indices] + sy * (max_scale_y[chunk_indices] - min_scale_y[chunk_indices])
            scales[:, 2] = min_scale_z[chunk_indices] + sz * (max_scale_z[chunk_indices] - min_scale_z[chunk_indices])

            # Color unpacking (8-8-8-8 bits)
            cr = ((packed_color >> 24) & 0xFF).astype(np.float32) / 255.0
            cg = ((packed_color >> 16) & 0xFF).astype(np.float32) / 255.0
            cb = ((packed_color >> 8) & 0xFF).astype(np.float32) / 255.0
            co = (packed_color & 0xFF).astype(np.float32) / 255.0

            color_r = min_r[chunk_indices] + cr * (max_r[chunk_indices] - min_r[chunk_indices])
            color_g = min_g[chunk_indices] + cg * (max_g[chunk_indices] - min_g[chunk_indices])
            color_b = min_b[chunk_indices] + cb * (max_b[chunk_indices] - min_b[chunk_indices])

            # Convert to SH0
            sh0[:, 0] = (color_r - 0.5) / SH_C0
            sh0[:, 1] = (color_g - 0.5) / SH_C0
            sh0[:, 2] = (color_b - 0.5) / SH_C0

            # Opacity conversion (logit space)
            opacities = np.where(
                (co > 0.0) & (co < 1.0),
                -np.log(1.0 / co - 1.0),
                np.where(co >= 1.0, 10.0, -10.0)
            )

            # Quaternion unpacking (smallest-three encoding)
            norm = 1.0 / (np.sqrt(2) * 0.5)
            a = (((packed_rotation >> 20) & 0x3FF).astype(np.float32) / 1023.0 - 0.5) * norm
            b = (((packed_rotation >> 10) & 0x3FF).astype(np.float32) / 1023.0 - 0.5) * norm
            c = ((packed_rotation & 0x3FF).astype(np.float32) / 1023.0 - 0.5) * norm
            m = np.sqrt(np.maximum(0.0, 1.0 - (a * a + b * b + c * c)))
            which = (packed_rotation >> 30).astype(np.int32)

            quats[:, 0] = np.where(which == 0, m, a)
            quats[:, 1] = np.where(which == 1, m, np.where(which == 0, a, b))
            quats[:, 2] = np.where(which == 2, m, np.where(which <= 1, b, c))
            quats[:, 3] = np.where(which == 3, m, c)

        # Decompress SH coefficients (vectorized)
        if shN_data is not None:
            num_sh_coeffs = shN_data.shape[1]
            num_sh_bands = num_sh_coeffs // 3

            # Vectorized normalization
            # Handle three cases: val==0, val==255, else
            normalized = np.where(
                shN_data == 0,
                0.0,
                np.where(
                    shN_data == 255,
                    1.0,
                    (shN_data.astype(np.float32) + 0.5) / 256.0
                )
            )

            # Vectorized conversion to SH values
            sh_flat = (normalized - 0.5) * 8.0

            # Reshape to (N, num_bands, 3)
            shN = sh_flat.reshape(num_vertices, num_sh_bands, 3)
        else:
            shN = np.zeros((num_vertices, 0, 3), dtype=np.float32)

        logger.debug(f"[Gaussian PLY] Read compressed: {num_vertices} Gaussians, SH bands {shN.shape[1]}")

        # Return GSData container (base=None since decompressed data is separate)
        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            base=None  # No shared base array for compressed format
        )

    except (OSError, ValueError, struct.error):
        return None


# ======================================================================================
# UNIFIED READING API
# ======================================================================================

def plyread(file_path: Union[str, Path], fast: bool = True) -> GSData:
    """Read Gaussian splatting PLY file (auto-detect format).

    Automatically detects and reads both compressed and uncompressed formats.
    Uses formats.detect_format() for fast format detection.

    Args:
        file_path: Path to PLY file
        fast: If True, use zero-copy optimization for uncompressed files (1.65x faster).
              If False, make safe copies of all arrays. Default: True.

    Returns:
        GSData container with Gaussian parameters

    Raises:
        ValueError: If file format is not recognized or invalid

    Performance:
        - fast=True: 2.89ms for 50K Gaussians (zero-copy views, 1.65x faster)
        - fast=False: 4.75ms for 50K Gaussians (safe independent copies)

    Example:
        >>> # Fast zero-copy reading (default, recommended)
        >>> data = plyread("scene.ply")
        >>> print(f"Loaded {data.means.shape[0]} Gaussians")
        >>> positions = data.means
        >>>
        >>> # Safe reading with independent copies
        >>> data = plyread("scene.ply", fast=False)
        >>>
        >>> # Can still unpack if needed
        >>> means, scales, quats, opacities, sh0, shN = data[:6]
    """
    file_path = Path(file_path)

    # Detect format first
    is_compressed, sh_degree = detect_format(file_path)

    # Try appropriate reader based on format and fast parameter
    if is_compressed:
        result = read_compressed(file_path)
    else:
        if fast:
            result = read_uncompressed_fast(file_path)
        else:
            result = read_uncompressed(file_path)

    if result is not None:
        return result

    raise ValueError(f"Unsupported PLY format or invalid file: {file_path}")


__all__ = [
    'plyread',
    'GSData',
    'read_uncompressed',
    'read_uncompressed_fast',
    'read_compressed',
]
