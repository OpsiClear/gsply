"""GPU-accelerated compression/decompression for Gaussian Splatting PLY files.

This module provides PyTorch-based GPU implementations of the PlayCanvas compressed
format for ultra-fast compression and decompression operations.

Key Optimization: Chunk Broadcasting
    Instead of per-vertex chunk lookups (slow on GPU), we reshape data to
    (NumChunks, 256, ...) and use broadcasting for vectorized operations.

Performance:
    - GPU decompression: 10-50x faster than CPU Numba JIT
    - GPU compression: 5-20x faster than CPU Numba JIT
    - Zero intermediate CPU transfers (direct GPU->File or File->GPU)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from gsply.formats import CHUNK_SIZE, SH_C0

if TYPE_CHECKING:
    from gsply.torch.gstensor import GSTensor

logger = logging.getLogger(__name__)

# ======================================================================================
# PRE-COMPUTED CONSTANTS (Phase 1 Optimization)
# ======================================================================================

# Quantization constants (avoid runtime division)
_INV_2047 = 1.0 / 2047.0  # 11-bit unpacking
_INV_1023 = 1.0 / 1023.0  # 10-bit unpacking
_INV_255 = 1.0 / 255.0  # 8-bit unpacking

# Quaternion normalization constant
_NORM_FACTOR = 1.4142135623730951 * 0.5  # sqrt(2) * 0.5, pre-computed

# Rounding offset for proper quantization (matches CPU's _ROUNDING_OFFSET)
# Using + 0.5 before int conversion is faster than torch.round() and matches CPU behavior
_ROUNDING_OFFSET = 0.5

# SH conversion constant
_INV_SH_C0 = 1.0 / SH_C0  # 1.0 / 0.28209479177387814 = 3.544907701811032

# Quaternion permutation table (Phase 2B: cached for gather-based unpacking)
# Shape: (4, 4) - for each 'which' value [0-3], the permutation of [m, a, b, c]
# Cache keyed by device to avoid global state issues
_QUAT_PERM_TABLE_CACHE: dict[str, torch.Tensor] = {}

# ======================================================================================
# TORCH.COMPILE SUPPORT (Optional optimization with automatic fallback)
# ======================================================================================

# Check if torch.compile with Triton backend is available
_TORCH_COMPILE_AVAILABLE = False
try:
    # Test if torch.compile works (requires triton on Linux, triton-windows on Windows)
    import torch._dynamo  # noqa: F401

    _TORCH_COMPILE_AVAILABLE = True
    logger.debug("torch.compile available for GPU compression acceleration")
except ImportError:
    logger.debug("torch.compile not available, using standard PyTorch operations")

# Cache for compiled functions (lazily populated on first use)
_COMPILED_FUNCTIONS: dict[str, object] = {}

# Flag to disable torch.compile after runtime failures
_TORCH_COMPILE_DISABLED = False


def _get_compiled_fn(name: str, fn: object) -> object:
    """Get compiled version of function, with lazy compilation and caching.

    Falls back to original function if compilation fails at runtime
    (e.g., missing C++ compiler on Windows).

    :param name: Name of the function (for caching)
    :param fn: The function to compile
    :returns: Compiled function if available, otherwise original function
    """
    if not _TORCH_COMPILE_AVAILABLE or _TORCH_COMPILE_DISABLED:
        return fn

    if name not in _COMPILED_FUNCTIONS:
        try:
            # Use default mode (reduce-overhead has issues with int32 bit ops on Windows)
            _COMPILED_FUNCTIONS[name] = torch.compile(fn, mode="default")
            logger.debug("Compiled %s with torch.compile", name)
        except Exception as e:
            logger.debug("Failed to compile %s: %s, using original", name, e)
            _COMPILED_FUNCTIONS[name] = fn

    return _COMPILED_FUNCTIONS[name]


def _call_with_fallback(compiled_fn: object, original_fn: object, *args: object) -> object:
    """Call compiled function with automatic fallback to original on runtime errors.

    :param compiled_fn: The compiled function to try first
    :param original_fn: The original function to fall back to
    :param args: Arguments to pass to the function
    :returns: Result of the function call
    """
    global _TORCH_COMPILE_DISABLED  # noqa: PLW0603

    if _TORCH_COMPILE_DISABLED or compiled_fn is original_fn:
        return original_fn(*args)

    try:
        return compiled_fn(*args)
    except Exception as e:
        # Runtime compilation failed (e.g., missing cl.exe on Windows)
        logger.debug("torch.compile runtime error: %s, falling back to standard ops", e)
        _TORCH_COMPILE_DISABLED = True
        # Clear cached compiled functions
        _COMPILED_FUNCTIONS.clear()
        return original_fn(*args)


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _compute_chunk_boundaries_gpu(
    num_chunks: int, num_gaussians: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute chunk start and end indices for chunked GPU processing.

    Each chunk contains CHUNK_SIZE Gaussians, except possibly the last chunk
    which may be smaller if num_gaussians is not a multiple of CHUNK_SIZE.

    :param num_chunks: Number of chunks
    :type num_chunks: int
    :param num_gaussians: Total number of Gaussians
    :type num_gaussians: int
    :param device: Device to create tensors on
    :type device: torch.device
    :return: Tuple of (chunk_starts, chunk_ends) tensors of shape (num_chunks,)
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    chunk_indices = torch.arange(num_chunks, device=device)
    chunk_starts = chunk_indices * CHUNK_SIZE
    chunk_ends = torch.minimum(
        chunk_starts + CHUNK_SIZE, torch.tensor(num_gaussians, device=device)
    )
    return chunk_starts, chunk_ends


# ======================================================================================
# GPU DECOMPRESSION (READ)
# ======================================================================================


def _unpack_positions_gpu(
    packed: torch.Tensor,
    min_bounds: torch.Tensor,
    range_bounds: torch.Tensor,
) -> torch.Tensor:
    """GPU-accelerated position unpacking using broadcasting (11-10-11 bits).

    :param packed: (NumChunks, 256) int32 packed positions
    :param min_bounds: (NumChunks, 1, 3) minimum xyz bounds per chunk
    :param range_bounds: (NumChunks, 1, 3) range (max - min) per chunk (pre-computed)
    :returns: (NumChunks, 256, 3) float32 unpacked positions
    """
    # Unpack bits using vectorized operations with pre-computed constants
    px = ((packed >> 21) & 0x7FF).float() * _INV_2047  # 11 bits
    py = ((packed >> 11) & 0x3FF).float() * _INV_1023  # 10 bits
    pz = (packed & 0x7FF).float() * _INV_2047  # 11 bits

    # Stack to (NumChunks, 256, 3)
    normalized = torch.stack([px, py, pz], dim=-1)

    # Dequantize using broadcasting (range_bounds is pre-computed)
    return min_bounds + normalized * range_bounds


def _unpack_scales_gpu(
    packed: torch.Tensor,
    min_bounds: torch.Tensor,
    range_bounds: torch.Tensor,
) -> torch.Tensor:
    """GPU-accelerated scale unpacking using broadcasting (11-10-11 bits).

    :param packed: (NumChunks, 256) uint32 packed scales
    :param min_bounds: (NumChunks, 1, 3) minimum scale bounds per chunk
    :param range_bounds: (NumChunks, 1, 3) range (max - min) per chunk (pre-computed)
    :returns: (NumChunks, 256, 3) float32 unpacked scales
    """
    # Unpack bits (same layout as positions)
    sx = ((packed >> 21) & 0x7FF).float() * _INV_2047
    sy = ((packed >> 11) & 0x3FF).float() * _INV_1023
    sz = (packed & 0x7FF).float() * _INV_2047

    normalized = torch.stack([sx, sy, sz], dim=-1)

    # Dequantize (range_bounds is pre-computed)
    return min_bounds + normalized * range_bounds


def _unpack_colors_and_opacities_gpu(
    packed: torch.Tensor,
    min_color: torch.Tensor,
    range_color: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-accelerated color and opacity unpacking (8-8-8-8 bits).

    :param packed: (NumChunks, 256) uint32 packed colors
    :param min_color: (NumChunks, 1, 3) minimum RGB bounds per chunk
    :param range_color: (NumChunks, 1, 3) range (max - min) per chunk (pre-computed)
    :returns: Tuple of sh0 (NumChunks, 256, 3) and opacities (NumChunks, 256)
    """
    # Unpack 8-bit channels with pre-computed constants
    cr = ((packed >> 24) & 0xFF).float() * _INV_255
    cg = ((packed >> 16) & 0xFF).float() * _INV_255
    cb = ((packed >> 8) & 0xFF).float() * _INV_255
    co = (packed & 0xFF).float() * _INV_255

    # Stack colors
    normalized_color = torch.stack([cr, cg, cb], dim=-1)

    # Dequantize colors (range_color is pre-computed)
    color_rgb = min_color + normalized_color * range_color

    # Convert RGB to SH0: sh0 = (color - 0.5) / SH_C0 (use pre-computed constant)
    sh0 = (color_rgb - 0.5) * _INV_SH_C0

    # Convert opacity from linear to logit space
    # opacity = -log(1/x - 1)
    epsilon = 1e-7
    opacities = torch.logit(co, eps=epsilon)

    return sh0, opacities


def _unpack_quaternions_gpu(packed: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated quaternion unpacking (2+10-10-10 bits, smallest-three).

    :param packed: (NumChunks, 256) int32 packed quaternions
    :returns: (NumChunks, 256, 4) float32 normalized quaternions (wxyz)
    """
    # Convert to unsigned for proper bit extraction
    packed_uint = packed.to(torch.int64) & 0xFFFFFFFF

    # Extract which component is largest (2 bits) - must mask after shift
    which = (packed_uint >> 30) & 0x3  # (NumChunks, 256)

    # Extract three components (10 bits each) with pre-computed constant
    a = ((packed_uint >> 20) & 0x3FF).float() * _INV_1023
    b = ((packed_uint >> 10) & 0x3FF).float() * _INV_1023
    c = (packed_uint & 0x3FF).float() * _INV_1023

    # Normalize to [-sqrt(2)/2, sqrt(2)/2] using pre-computed constant
    a = (a - 0.5) * _NORM_FACTOR * 2.0
    b = (b - 0.5) * _NORM_FACTOR * 2.0
    c = (c - 0.5) * _NORM_FACTOR * 2.0

    # Compute largest component: m = sqrt(1 - (a^2 + b^2 + c^2))
    m_squared = 1.0 - (a * a + b * b + c * c)
    m = torch.sqrt(torch.clamp(m_squared, min=0.0))

    # Phase 2B: Branchless quaternion unpacking using gather (3.35x speedup)
    # Stack all components: (NumChunks, 256, 4) = [m, a, b, c]
    components = torch.stack([m, a, b, c], dim=-1)

    # Use cached permutation table (create once per device)
    device_str = str(packed.device)
    if device_str not in _QUAT_PERM_TABLE_CACHE:
        _QUAT_PERM_TABLE_CACHE[device_str] = torch.tensor(
            [
                [0, 1, 2, 3],  # which=0: [m, a, b, c]
                [1, 0, 2, 3],  # which=1: [a, m, b, c]
                [1, 2, 0, 3],  # which=2: [a, b, m, c]
                [1, 2, 3, 0],  # which=3: [a, b, c, m]
            ],
            device=packed.device,
            dtype=torch.int64,
        )

    # Gather using which index: (NumChunks, 256) → (NumChunks, 256, 4)
    perm_table = _QUAT_PERM_TABLE_CACHE[device_str]
    indices = perm_table[which.long()]  # (NumChunks, 256, 4)
    return torch.gather(components, -1, indices)


def _unpack_sh_gpu(sh_data: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated SH coefficient unpacking.

    :param sh_data: (N, num_coeffs) uint8 packed SH coefficients
    :returns: (N, num_coeffs) float32 unpacked SH coefficients
    """
    # Dequantize: (x - 127.5) / 32.0
    return (sh_data.float() - 127.5) / 32.0


def decompress_gpu(
    chunk_data: np.ndarray,
    vertex_data: np.ndarray,
    sh_data: np.ndarray | None,
    device: str = "cuda",
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
]:
    """Decompress Gaussian data on GPU using broadcasting optimization.

    :param chunk_data: (num_chunks, 18) float32 chunk bounds (CPU numpy)
    :param vertex_data: (num_vertices, 4) uint32 packed vertex data (CPU numpy)
    :param sh_data: (num_vertices, num_coeffs) uint8 SH data or None (CPU numpy)
    :param device: Device to decompress on (default "cuda")
    :returns: Tuple of (means, scales, quats, opacities, sh0, shN) as GPU tensors
    """
    num_vertices = vertex_data.shape[0]
    num_chunks = chunk_data.shape[0]

    # Calculate padding needed
    padded_vertices = num_chunks * CHUNK_SIZE
    padding_needed = padded_vertices - num_vertices

    # Phase 3A: Pre-pad and cast on CPU (cheaper than GPU operations)
    if padding_needed > 0:
        pad_zeros = np.zeros((padding_needed, 4), dtype=np.uint32)
        vertex_data = np.concatenate([vertex_data, pad_zeros], axis=0)

    # Phase 4B: Batch memory transfer using byte-level concatenation (1.71x speedup)
    # Combine chunk_bounds and vertex_data into single transfer to reduce DMA overhead
    # Use byte-level concatenation to preserve uint32 bit patterns
    if device != "cpu":
        # Convert to bytes and concatenate (preserves bit patterns)
        chunk_bytes = chunk_data.tobytes()
        vertex_bytes = vertex_data.tobytes()
        combined_bytes = chunk_bytes + vertex_bytes

        # Transfer as uint8 (byte array), then reinterpret on GPU
        # Use copy() to make array writable (PyTorch requirement)
        combined_uint8 = np.frombuffer(combined_bytes, dtype=np.uint8).copy()
        combined_tensor = torch.from_numpy(combined_uint8).to(device)

        # Reinterpret bytes as float32 for chunk_bounds and int32 for vertex_data
        # Chunk bounds: first num_chunks*18*4 bytes as float32
        chunk_bounds_bytes = combined_tensor[: num_chunks * 18 * 4]
        chunk_bounds = chunk_bounds_bytes.view(torch.float32).reshape(num_chunks, 18)

        # Vertex data: remaining bytes as int32 (uint32 reinterpreted as int32)
        vertex_bytes_tensor = combined_tensor[num_chunks * 18 * 4 :]
        vertex_packed = vertex_bytes_tensor.view(torch.int32).reshape(padded_vertices, 4)
    else:
        # CPU path: separate transfers (no benefit from batching)
        chunk_bounds = torch.from_numpy(chunk_data).to(device)
        vertex_packed = torch.from_numpy(vertex_data.astype(np.int32)).to(device)

    # Phase 4A: Use views instead of reshape to avoid redundant operations
    # Extract packed components directly using view (more efficient than reshape)
    packed_position = vertex_packed.view(num_chunks, CHUNK_SIZE, 4)[:, :, 0]  # (NumChunks, 256)
    packed_rotation = vertex_packed.view(num_chunks, CHUNK_SIZE, 4)[:, :, 1]
    packed_scale = vertex_packed.view(num_chunks, CHUNK_SIZE, 4)[:, :, 2]
    packed_color = vertex_packed.view(num_chunks, CHUNK_SIZE, 4)[:, :, 3]

    # Phase 2C: Vectorize chunk bounds extraction (1.19x speedup)
    # Reshape to (NumChunks, 6, 3) for [min/max] × [pos, scale, color]
    num_chunks = chunk_bounds.shape[0]
    bounds_reshaped = chunk_bounds.reshape(num_chunks, 6, 3)

    # Extract using views (zero copy) - (NumChunks, 1, 3)
    min_pos = bounds_reshaped[:, 0:1, :]
    max_pos = bounds_reshaped[:, 1:2, :]
    min_scale = bounds_reshaped[:, 2:3, :]
    max_scale = bounds_reshaped[:, 3:4, :]
    min_color = bounds_reshaped[:, 4:5, :]
    max_color = bounds_reshaped[:, 5:6, :]

    # Phase 4A: Pre-compute ranges (21.50x micro-benchmark speedup)
    # Compute ranges once instead of in each unpack function
    range_pos = max_pos - min_pos
    range_scale = max_scale - min_scale
    range_color = max_color - min_color

    # Unpack all components using GPU kernels (pass pre-computed ranges)
    positions = _unpack_positions_gpu(packed_position, min_pos, range_pos)
    scales = _unpack_scales_gpu(packed_scale, min_scale, range_scale)
    quats = _unpack_quaternions_gpu(packed_rotation)
    sh0, opacities = _unpack_colors_and_opacities_gpu(packed_color, min_color, range_color)

    # Flatten back to (N, ...)
    means = positions.reshape(-1, 3)[:num_vertices]
    scales = scales.reshape(-1, 3)[:num_vertices]
    quats = quats.reshape(-1, 4)[:num_vertices]
    opacities = opacities.reshape(-1)[:num_vertices]
    sh0 = sh0.reshape(-1, 3)[:num_vertices]

    # Handle SH coefficients with optimized transfer
    shN = None  # noqa: N806
    if sh_data is not None:
        sh_tensor = torch.from_numpy(sh_data)
        if device != "cpu":
            sh_tensor = sh_tensor.pin_memory().to(device, non_blocking=True)
        else:
            sh_tensor = sh_tensor.to(device)
        sh_flat = _unpack_sh_gpu(sh_tensor)

        # PLY stores SH coefficients channel-grouped: [R0..Rk, G0..Gk, B0..Bk]
        # Reshape to [N, 3, K] then transpose to [N, K, 3] for gsplat convention
        num_coeffs = sh_flat.shape[1]
        num_bands = num_coeffs // 3
        shN = sh_flat.reshape(num_vertices, 3, num_bands).transpose(1, 2)  # noqa: N806

    logger.debug(f"[GPU Decompression] Decompressed {num_vertices:,} Gaussians on {device}")

    return means, scales, quats, opacities, sh0, shN


# ======================================================================================
# GPU COMPRESSION (WRITE)
# ======================================================================================


def _compute_chunk_bounds_gpu(
    data: torch.Tensor, num_chunks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute min/max bounds for each chunk using GPU reduction.

    Computes bounds on actual data only (without padding), matching CPU behavior.
    This ensures accurate quantization bounds that don't include padded values.

    Uses vectorized operations with masking to handle partial last chunk efficiently.

    :param data: (N, D) tensor to compute bounds for
    :param num_chunks: Number of chunks
    :returns: Tuple of (min_bounds, max_bounds) with shape (num_chunks, D)
    """
    num_gaussians = data.shape[0]
    padded_size = num_chunks * CHUNK_SIZE
    padding_needed = padded_size - num_gaussians

    # Pad data for reshaping (pad with last value, but we'll mask it out)
    if padding_needed > 0:
        pad_values = data[-1:].expand(padding_needed, -1)
        data_padded = torch.cat([data, pad_values], dim=0)
    else:
        data_padded = data

    # Reshape to (NumChunks, 256, D)
    data_reshaped = data_padded.reshape(num_chunks, CHUNK_SIZE, -1)

    # Create mask to exclude padded values in last chunk
    # For last chunk, mask out positions >= num_gaussians
    chunk_starts, chunk_ends = _compute_chunk_boundaries_gpu(num_chunks, num_gaussians, data.device)

    # Create mask: (NumChunks, 256) - True for valid data, False for padding
    positions = torch.arange(CHUNK_SIZE, device=data.device).unsqueeze(0)  # (1, 256)
    valid_mask = positions < (chunk_ends - chunk_starts).unsqueeze(1)  # (NumChunks, 256)

    # Compute bounds with masking: set padded values to inf/-inf so they're excluded
    data_masked = data_reshaped.clone()
    # Set padded values to very large/small values so min/max ignores them
    data_masked[~valid_mask] = float("inf")
    min_bounds = data_masked.min(dim=1).values  # (num_chunks, D)

    data_masked = data_reshaped.clone()
    data_masked[~valid_mask] = float("-inf")
    max_bounds = data_masked.max(dim=1).values  # (num_chunks, D)

    return min_bounds, max_bounds


def _pack_positions_gpu(
    positions: torch.Tensor,
    min_bounds: torch.Tensor,
    max_bounds: torch.Tensor,
) -> torch.Tensor:
    """GPU-accelerated position packing (11-10-11 bits).

    :param positions: (NumChunks, 256, 3) float32 positions
    :param min_bounds: (NumChunks, 1, 3) minimum bounds
    :param max_bounds: (NumChunks, 1, 3) maximum bounds
    :returns: (NumChunks, 256) uint32 packed positions
    """
    # Handle zero range
    range_bounds = max_bounds - min_bounds
    range_bounds = torch.where(range_bounds == 0, torch.ones_like(range_bounds), range_bounds)

    # Normalize to [0, 1]
    normalized = (positions - min_bounds) / range_bounds
    normalized = torch.clamp(normalized, 0.0, 1.0)

    # Quantize with rounding (+ 0.5 matches CPU behavior and is faster than torch.round())
    px = (normalized[:, :, 0] * 2047.0 + _ROUNDING_OFFSET).to(torch.int32)
    py = (normalized[:, :, 1] * 1023.0 + _ROUNDING_OFFSET).to(torch.int32)
    pz = (normalized[:, :, 2] * 2047.0 + _ROUNDING_OFFSET).to(torch.int32)

    # Pack bits
    return (px << 21) | (py << 11) | pz


def _pack_scales_gpu(
    scales: torch.Tensor,
    min_bounds: torch.Tensor,
    max_bounds: torch.Tensor,
) -> torch.Tensor:
    """GPU-accelerated scale packing (11-10-11 bits).

    :param scales: (NumChunks, 256, 3) float32 scales
    :param min_bounds: (NumChunks, 1, 3) minimum bounds (clamped to [-20, 20])
    :param max_bounds: (NumChunks, 1, 3) maximum bounds (clamped to [-20, 20])
    :returns: (NumChunks, 256) uint32 packed scales
    """
    # Clamp bounds to [-20, 20] (matches CPU implementation)
    min_bounds = torch.clamp(min_bounds, -20.0, 20.0)
    max_bounds = torch.clamp(max_bounds, -20.0, 20.0)

    range_bounds = max_bounds - min_bounds
    range_bounds = torch.where(range_bounds == 0, torch.ones_like(range_bounds), range_bounds)

    normalized = (scales - min_bounds) / range_bounds
    normalized = torch.clamp(normalized, 0.0, 1.0)

    # Quantize with rounding (+ 0.5 matches CPU behavior and is faster than torch.round())
    sx = (normalized[:, :, 0] * 2047.0 + _ROUNDING_OFFSET).to(torch.int32)
    sy = (normalized[:, :, 1] * 1023.0 + _ROUNDING_OFFSET).to(torch.int32)
    sz = (normalized[:, :, 2] * 2047.0 + _ROUNDING_OFFSET).to(torch.int32)

    return (sx << 21) | (sy << 11) | sz


def _pack_colors_and_opacities_gpu(
    sh0: torch.Tensor,
    opacities: torch.Tensor,
    min_color: torch.Tensor,
    max_color: torch.Tensor,
) -> torch.Tensor:
    """GPU-accelerated color and opacity packing (8-8-8-8 bits).

    :param sh0: (NumChunks, 256, 3) float32 SH0 coefficients
    :param opacities: (NumChunks, 256) float32 opacities in logit space
    :param min_color: (NumChunks, 1, 3) minimum color bounds
    :param max_color: (NumChunks, 1, 3) maximum color bounds
    :returns: (NumChunks, 256) uint32 packed colors
    """
    # Convert SH0 to RGB: color = sh0 * SH_C0 + 0.5
    color_rgb = sh0 * SH_C0 + 0.5

    # Normalize colors
    range_color = max_color - min_color
    range_color = torch.where(range_color == 0, torch.ones_like(range_color), range_color)

    normalized = (color_rgb - min_color) / range_color
    normalized = torch.clamp(normalized, 0.0, 1.0)

    # Quantize with rounding (+ 0.5 matches CPU behavior and is faster than torch.round())
    cr = (normalized[:, :, 0] * 255.0 + _ROUNDING_OFFSET).to(torch.int32)
    cg = (normalized[:, :, 1] * 255.0 + _ROUNDING_OFFSET).to(torch.int32)
    cb = (normalized[:, :, 2] * 255.0 + _ROUNDING_OFFSET).to(torch.int32)

    # Convert opacity from logit to linear: opacity = 1 / (1 + exp(-x))
    opacity_linear = torch.sigmoid(opacities)
    opacity_linear = torch.clamp(opacity_linear, 0.0, 1.0)
    co = (opacity_linear * 255.0 + _ROUNDING_OFFSET).to(torch.int32)

    # Pack bits
    return (cr << 24) | (cg << 16) | (cb << 8) | co


def _pack_quaternions_gpu(quats: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated quaternion packing (2+10-10-10 bits, smallest-three).

    :param quats: (NumChunks, 256, 4) float32 quaternions (wxyz)
    :returns: (NumChunks, 256) uint32 packed quaternions
    """
    # Normalize quaternions
    norms = torch.norm(quats, dim=-1, keepdim=True)
    quats_norm = quats / torch.clamp(norms, min=1e-7)

    # Find largest component by absolute value
    abs_quats = torch.abs(quats_norm)
    which = torch.argmax(abs_quats, dim=-1)  # (NumChunks, 256)

    # Flip if largest component is negative
    num_chunks, chunk_size = which.shape
    largest_vals = torch.gather(quats_norm, 2, which.unsqueeze(-1)).squeeze(-1)
    flip_mask = largest_vals < 0
    quats_norm = torch.where(flip_mask.unsqueeze(-1), -quats_norm, quats_norm)

    # Extract three smallest components
    # Create a mask for components to keep (all except largest)
    mask = torch.ones((num_chunks, chunk_size, 4), dtype=torch.bool, device=quats.device)
    mask.scatter_(2, which.unsqueeze(-1), False)

    # Extract three components
    three_components = quats_norm[mask].reshape(num_chunks, chunk_size, 3)

    # Normalize to [0, 1] for quantization (Phase 2A: use pre-computed constant for 2.94x speedup)
    qa = three_components[:, :, 0] / (_NORM_FACTOR * 2.0) + 0.5
    qb = three_components[:, :, 1] / (_NORM_FACTOR * 2.0) + 0.5
    qc = three_components[:, :, 2] / (_NORM_FACTOR * 2.0) + 0.5

    # Clamp and quantize with rounding (+ 0.5 matches CPU behavior and is faster than torch.round())
    stacked = torch.stack([qa, qb, qc], dim=-1)
    stacked = torch.clamp(stacked, 0.0, 1.0)
    qa_clamped, qb_clamped, qc_clamped = stacked[..., 0], stacked[..., 1], stacked[..., 2]

    qa_int = (qa_clamped * 1023.0 + _ROUNDING_OFFSET).to(torch.int32)
    qb_int = (qb_clamped * 1023.0 + _ROUNDING_OFFSET).to(torch.int32)
    qc_int = (qc_clamped * 1023.0 + _ROUNDING_OFFSET).to(torch.int32)

    # Pack bits
    return (which.to(torch.int32) << 30) | (qa_int << 20) | (qb_int << 10) | qc_int


def _pack_sh_gpu(shN: torch.Tensor) -> torch.Tensor:  # noqa: N803
    """GPU-accelerated SH coefficient packing.

    Original 3DGS PLY format stores f_rest coefficients as channel-grouped:
    [R0,R1,...,Rk, G0,G1,...,Gk, B0,B1,...,Bk]

    This matches the original 3DGS save_ply which does:
    f_rest = features_rest.transpose(1, 2).flatten(start_dim=1)

    :param shN: (N, num_bands, 3) float32 SH coefficients
    :returns: (N, num_coeffs) uint8 packed SH coefficients in channel-grouped order
    """
    # Transpose [N, K, 3] -> [N, 3, K] then flatten to [N, 3*K]
    # This gives channel-grouped order matching original 3DGS PLY format
    sh_flat = shN.transpose(1, 2).reshape(shN.shape[0], -1)

    # Quantize: shN * 32 + 128, clamped to [0, 255]
    return torch.clamp(sh_flat * 32.0 + 128.0, 0.0, 255.0).to(torch.uint8)


def compress_gpu(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor | None = None,  # noqa: N803
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compress Gaussian data on GPU using reduction optimization.

    :param means: (N, 3) xyz positions on GPU
    :param scales: (N, 3) scale parameters on GPU
    :param quats: (N, 4) rotation quaternions on GPU
    :param opacities: (N,) opacity values on GPU
    :param sh0: (N, 3) DC spherical harmonics on GPU
    :param shN: (N, K, 3) higher-order SH or None on GPU
    :returns: Tuple of (chunk_bounds, packed_vertex_data, packed_sh_data) as CPU numpy arrays
    """
    num_gaussians = means.shape[0]
    num_chunks = (num_gaussians + CHUNK_SIZE - 1) // CHUNK_SIZE
    padded_size = num_chunks * CHUNK_SIZE

    # Compute chunk bounds for positions, scales, and colors
    min_pos, max_pos = _compute_chunk_bounds_gpu(means, num_chunks)
    min_scale, max_scale = _compute_chunk_bounds_gpu(scales, num_chunks)

    # Clamp scale bounds to [-20, 20]
    min_scale = torch.clamp(min_scale, -20.0, 20.0)
    max_scale = torch.clamp(max_scale, -20.0, 20.0)

    # Convert SH0 to color for bounds computation
    color_rgb = sh0 * SH_C0 + 0.5
    min_color, max_color = _compute_chunk_bounds_gpu(color_rgb, num_chunks)

    # Pad data to chunk boundaries
    padding_needed = padded_size - num_gaussians
    if padding_needed > 0:
        means_padded = torch.cat([means, means[-1:].expand(padding_needed, -1)], dim=0)
        scales_padded = torch.cat([scales, scales[-1:].expand(padding_needed, -1)], dim=0)
        quats_padded = torch.cat([quats, quats[-1:].expand(padding_needed, -1)], dim=0)
        opacities_padded = torch.cat([opacities, opacities[-1:].expand(padding_needed)], dim=0)
        sh0_padded = torch.cat([sh0, sh0[-1:].expand(padding_needed, -1)], dim=0)
    else:
        means_padded = means
        scales_padded = scales
        quats_padded = quats
        opacities_padded = opacities
        sh0_padded = sh0

    # Reshape to (NumChunks, 256, ...)
    means_reshaped = means_padded.reshape(num_chunks, CHUNK_SIZE, 3)
    scales_reshaped = scales_padded.reshape(num_chunks, CHUNK_SIZE, 3)
    quats_reshaped = quats_padded.reshape(num_chunks, CHUNK_SIZE, 4)
    opacities_reshaped = opacities_padded.reshape(num_chunks, CHUNK_SIZE)
    sh0_reshaped = sh0_padded.reshape(num_chunks, CHUNK_SIZE, 3)

    # Reshape bounds for broadcasting
    min_pos_bc = min_pos.unsqueeze(1)  # (NumChunks, 1, 3)
    max_pos_bc = max_pos.unsqueeze(1)
    min_scale_bc = min_scale.unsqueeze(1)
    max_scale_bc = max_scale.unsqueeze(1)
    min_color_bc = min_color.unsqueeze(1)
    max_color_bc = max_color.unsqueeze(1)

    # Get compiled packing functions (uses torch.compile if available, else original)
    pack_positions_fn = _get_compiled_fn("pack_positions", _pack_positions_gpu)
    pack_scales_fn = _get_compiled_fn("pack_scales", _pack_scales_gpu)
    pack_colors_fn = _get_compiled_fn("pack_colors", _pack_colors_and_opacities_gpu)
    pack_quats_fn = _get_compiled_fn("pack_quaternions", _pack_quaternions_gpu)

    # Pack all components (with automatic fallback on runtime errors)
    packed_position = _call_with_fallback(
        pack_positions_fn, _pack_positions_gpu, means_reshaped, min_pos_bc, max_pos_bc
    )
    packed_scale = _call_with_fallback(
        pack_scales_fn, _pack_scales_gpu, scales_reshaped, min_scale_bc, max_scale_bc
    )
    packed_color = _call_with_fallback(
        pack_colors_fn,
        _pack_colors_and_opacities_gpu,
        sh0_reshaped,
        opacities_reshaped,
        min_color_bc,
        max_color_bc,
    )
    packed_rotation = _call_with_fallback(pack_quats_fn, _pack_quaternions_gpu, quats_reshaped)

    # Stack into vertex data array (N, 4)
    packed_vertex = torch.stack(
        [packed_position, packed_rotation, packed_scale, packed_color], dim=-1
    )
    packed_vertex = packed_vertex.reshape(-1, 4)[:num_gaussians]

    # Assemble chunk bounds (num_chunks, 18)
    chunk_bounds = torch.cat([min_pos, max_pos, min_scale, max_scale, min_color, max_color], dim=1)

    # Pack SH coefficients if present
    packed_sh = None
    if shN is not None and shN.shape[1] > 0:
        packed_sh = _pack_sh_gpu(shN)
        packed_sh = packed_sh.cpu().numpy()

    # Transfer to CPU
    chunk_bounds_np = chunk_bounds.cpu().numpy().astype(np.float32)
    packed_vertex_np = packed_vertex.cpu().numpy().astype(np.uint32)

    logger.debug(f"[GPU Compression] Compressed {num_gaussians:,} Gaussians ({num_chunks} chunks)")

    return chunk_bounds_np, packed_vertex_np, packed_sh


# ======================================================================================
# HIGH-LEVEL API
# ======================================================================================


def read_compressed_gpu(file_path: str | Path, device: str = "cuda") -> GSTensor:
    """Read compressed PLY file directly to GPU.

    Uses CPU for file I/O and header parsing, then performs decompression on GPU.
    Faster than CPU decompression + GPU upload for large files.

    :param file_path: Path to compressed PLY file
    :param device: Target GPU device (default "cuda")
    :returns: GSTensor with decompressed data on GPU
    """
    from gsply.gsdata import DataFormat, _create_format_dict, _get_sh_order_format
    from gsply.reader import _parse_header_from_bytes
    from gsply.torch.gstensor import GSTensor

    file_path = Path(file_path)

    # Optimized file reading: read header only, then seek + np.fromfile for data (5.48x faster)
    with open(file_path, "rb") as f:
        # Read header (typically <2KB)
        header_bytes = b""
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            header_bytes += chunk
            if b"end_header\n" in header_bytes:
                break

        # Find header end
        end_idx = header_bytes.find(b"end_header\n")
        if end_idx == -1:
            raise ValueError("Invalid PLY: no end_header found")

        header_bytes = header_bytes[: end_idx + len(b"end_header\n")]
        data_offset = len(header_bytes)

        # Parse header
        header_lines, _, elements = _parse_header_from_bytes(header_bytes)

        num_chunks = elements["chunk"]["count"]
        num_vertices = elements["vertex"]["count"]

        # Seek to data section and read directly into numpy arrays (faster than read + frombuffer)
        f.seek(data_offset)
        chunk_data = np.fromfile(f, dtype=np.float32, count=num_chunks * 18).reshape(num_chunks, 18)
        vertex_data = np.fromfile(f, dtype=np.uint32, count=num_vertices * 4).reshape(
            num_vertices, 4
        )

        # Read SH data if present
        sh_data = None
        if "sh" in elements:
            num_sh_coeffs = len(elements["sh"]["properties"])
            sh_data = np.fromfile(f, dtype=np.uint8, count=num_vertices * num_sh_coeffs).reshape(
                num_vertices, num_sh_coeffs
            )

    # Decompress on GPU
    means, scales, quats, opacities, sh0, shN = decompress_gpu(  # noqa: N806
        chunk_data, vertex_data, sh_data, device
    )

    # Determine SH degree from shN shape
    if shN is not None and shN.shape[1] > 0:
        sh_bands = shN.shape[1]
        from gsply.formats import SH_BANDS_TO_DEGREE

        sh_degree = SH_BANDS_TO_DEGREE.get(int(sh_bands), 0)
    else:
        sh_degree = 0

    logger.info(f"[GPU Read] Loaded {num_vertices:,} Gaussians from {file_path.name} to {device}")

    # Create GSTensor with format flags (PLY format)
    return GSTensor(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN if shN is not None else torch.zeros((num_vertices, 0, 3), device=device),
        masks=None,
        mask_names=None,
        _base=None,
        _format=_create_format_dict(
            scales=DataFormat.SCALES_PLY,  # PLY files use log-scales
            opacities=DataFormat.OPACITIES_PLY,  # PLY files use logit-opacities
            sh0=DataFormat.SH0_SH,  # SH format
            sh_order=_get_sh_order_format(sh_degree),
            means=DataFormat.MEANS_RAW,
            quats=DataFormat.QUATS_RAW,
        ),
    )


def write_compressed_gpu(file_path: str | Path, gstensor: GSTensor) -> None:
    """Write GSTensor to compressed PLY file using GPU compression.

    Performs compression on GPU, then transfers to CPU for file I/O.
    Faster than CPU download + CPU compression for large datasets.

    :param file_path: Output file path
    :param gstensor: GSTensor on GPU to compress and save
    """
    file_path = Path(file_path)

    # Compress on GPU
    chunk_bounds, packed_vertex, packed_sh = compress_gpu(
        gstensor.means,
        gstensor.scales,
        gstensor.quats,
        gstensor.opacities,
        gstensor.sh0,
        gstensor.shN if gstensor.shN is not None and gstensor.shN.shape[1] > 0 else None,
    )

    num_gaussians = packed_vertex.shape[0]
    num_chunks = chunk_bounds.shape[0]

    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element chunk {num_chunks}",
    ]

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

    header_lines.append(f"element vertex {num_gaussians}")
    header_lines.append("property uint packed_position")
    header_lines.append("property uint packed_rotation")
    header_lines.append("property uint packed_scale")
    header_lines.append("property uint packed_color")

    if packed_sh is not None:
        num_sh_coeffs = packed_sh.shape[1]
        header_lines.append(f"element sh {num_gaussians}")
        for i in range(num_sh_coeffs):
            header_lines.append(f"property uchar coeff_{i}")

    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"
    header_bytes = header.encode("ascii")

    # Write to file
    with open(file_path, "wb") as f:
        f.write(header_bytes)
        chunk_bounds.tofile(f)
        packed_vertex.tofile(f)
        if packed_sh is not None:
            packed_sh.tofile(f)

    logger.info(
        f"[GPU Write] Saved {num_gaussians:,} Gaussians to {file_path.name} "
        f"({num_chunks} chunks, {len(header_bytes) + chunk_bounds.nbytes + packed_vertex.nbytes + (packed_sh.nbytes if packed_sh is not None else 0)} bytes)"
    )


__all__ = [
    "decompress_gpu",
    "compress_gpu",
    "read_compressed_gpu",
    "write_compressed_gpu",
]
