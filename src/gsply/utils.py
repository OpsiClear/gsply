"""Utility functions for Gaussian Splatting operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import numpy as np
from numba import jit, njit, prange
from numpy.typing import NDArray

from gsply.formats import SH_C0

if TYPE_CHECKING:
    from gsply.gsdata import GSData

logger = logging.getLogger(__name__)

Float32Array = NDArray[np.float32]

# Default clamp values recommended by rendering pipeline
_DEFAULT_MIN_SCALE: Final[np.float32] = np.float32(1e-4)
_DEFAULT_MAX_SCALE: Final[np.float32] = np.float32(100.0)
_DEFAULT_MIN_NORM: Final[np.float32] = np.float32(1e-8)


def sh2rgb(sh: np.ndarray | float) -> np.ndarray | float:
    """Convert SH DC coefficients to RGB colors.

    :param sh: SH DC coefficients (N, 3) or scalar
    :returns: RGB colors in [0, 1] range

    Example:
        >>> import gsply
        >>> sh = np.array([[0.0, 0.5, -0.5]])
        >>> rgb = gsply.sh2rgb(sh)
        >>> print(rgb)  # [[0.5, 0.641, 0.359]]
    """
    return sh * SH_C0 + 0.5


def rgb2sh(rgb: np.ndarray | float) -> np.ndarray | float:
    """Convert RGB colors to SH DC coefficients.

    :param rgb: RGB colors in [0, 1] range (N, 3) or scalar
    :returns: SH DC coefficients

    Example:
        >>> import gsply
        >>> rgb = np.array([[1.0, 0.5, 0.0]])
        >>> sh = gsply.rgb2sh(rgb)
    """
    return (rgb - 0.5) / SH_C0


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _logit_impl(x: np.ndarray, out: np.ndarray, eps: float):
    for i in prange(x.size):
        val = x.flat[i]
        if val < eps:
            val = eps
        elif val > 1.0 - eps:
            val = 1.0 - eps
        out.flat[i] = np.log(val / (1.0 - val))


def logit(x: np.ndarray | float, eps: float = 1e-6) -> np.ndarray | float:
    """Compute logit function (inverse sigmoid) with numerical stability.

    Optimized for both scalar and array inputs using Numba.
    Formula: log(x / (1 - x))

    :param x: Input values in [0, 1] range (probabilities)
    :param eps: Epsilon for numerical stability (clamping)
    :returns: Logit values
    """
    if np.isscalar(x):
        val = float(x)
        val = max(eps, min(val, 1.0 - eps))
        return np.log(val / (1.0 - val))

    out = np.empty_like(x)
    _logit_impl(x, out, eps)
    return out


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _sigmoid_impl(x: np.ndarray, out: np.ndarray):
    for i in prange(x.size):
        val = x.flat[i]
        # Stable sigmoid
        if val >= 0:
            out.flat[i] = 1.0 / (1.0 + np.exp(-val))
        else:
            z = np.exp(val)
            out.flat[i] = z / (1.0 + z)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Compute sigmoid function (inverse logit) with numerical stability.

    Optimized for both scalar and array inputs using Numba.
    Formula: 1 / (1 + exp(-x))

    :param x: Input values (logits)
    :returns: Values in [0, 1] range (probabilities)
    """
    if np.isscalar(x):
        val = float(x)
        if val >= 0:
            return 1.0 / (1.0 + np.exp(-val))
        z = np.exp(val)
        return z / (1.0 + z)

    out = np.empty_like(x)
    _sigmoid_impl(x, out)
    return out


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _sh2rgb_inplace_jit(sh: np.ndarray, sh_c0: float):
    """Numba-accelerated in-place SH to RGB conversion.

    :param sh: (N, 3) float32 array - modified in-place
    :param sh_c0: SH constant (0.28209479177387814)
    """
    n = sh.shape[0]
    for i in prange(n):
        for j in range(3):
            sh[i, j] = sh[i, j] * sh_c0 + 0.5


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _rgb2sh_inplace_jit(rgb: np.ndarray, inv_sh_c0: float):
    """Numba-accelerated in-place RGB to SH conversion.

    :param rgb: (N, 3) float32 array - modified in-place
    :param inv_sh_c0: Inverse SH constant (1.0 / 0.28209479177387814)
    """
    n = rgb.shape[0]
    for i in prange(n):
        for j in range(3):
            rgb[i, j] = (rgb[i, j] - 0.5) * inv_sh_c0


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _activate_gaussians_numba(
    scales: Float32Array,
    opacities: Float32Array,
    quats: Float32Array,
    min_scale: np.float32,
    max_scale: np.float32,
    min_quat_norm: np.float32,
) -> None:
    """
    Fused attribute activation kernel.

    :param scales: Log-scale values, shape [N, 3]
    :param opacities: Logit opacities, shape [N]
    :param quats: Raw quaternions, shape [N, 4]
    :param min_scale: Minimum clamp value post-exp
    :param max_scale: Maximum clamp value post-exp
    :param min_quat_norm: Minimum allowable quaternion norm (safety floor)
    """
    count = scales.shape[0]

    for i in prange(count):
        # Scale activation: exp + clamp
        sx = np.exp(scales[i, 0])
        sy = np.exp(scales[i, 1])
        sz = np.exp(scales[i, 2])

        sx = min(max(sx, min_scale), max_scale)
        sy = min(max(sy, min_scale), max_scale)
        sz = min(max(sz, min_scale), max_scale)

        scales[i, 0] = sx
        scales[i, 1] = sy
        scales[i, 2] = sz

        # Opacity activation: numerically-stable sigmoid
        logit = opacities[i]
        if logit >= 0.0:
            exp_term = np.exp(-logit)
            sigmoid = 1.0 / (1.0 + exp_term)
        else:
            exp_term = np.exp(logit)
            sigmoid = exp_term / (1.0 + exp_term)
        opacities[i] = sigmoid

        # Quaternion activation: normalize with safety floor
        qx = quats[i, 0]
        qy = quats[i, 1]
        qz = quats[i, 2]
        qw = quats[i, 3]

        norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < min_quat_norm:
            quats[i, 0] = np.float32(0.0)
            quats[i, 1] = np.float32(0.0)
            quats[i, 2] = np.float32(0.0)
            quats[i, 3] = np.float32(1.0)
        else:
            inv = 1.0 / norm
            quats[i, 0] = qx * inv
            quats[i, 1] = qy * inv
            quats[i, 2] = qz * inv
            quats[i, 3] = qw * inv


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _deactivate_gaussians_numba(
    scales: Float32Array,
    opacities: Float32Array,
    min_scale: np.float32,
    min_opacity: np.float32,
    max_opacity: np.float32,
) -> None:
    """
    Fused attribute deactivation kernel (reverse of activation).

    :param scales: Linear scale values, shape [N, 3]
    :param opacities: Linear opacities, shape [N]
    :param min_scale: Minimum clamp value before log
    :param min_opacity: Minimum clamp value before logit
    :param max_opacity: Maximum clamp value before logit
    """
    count = scales.shape[0]

    for i in prange(count):
        # Scale deactivation: clamp + log
        sx = max(scales[i, 0], min_scale)
        sy = max(scales[i, 1], min_scale)
        sz = max(scales[i, 2], min_scale)

        scales[i, 0] = np.log(sx)
        scales[i, 1] = np.log(sy)
        scales[i, 2] = np.log(sz)

        # Opacity deactivation: clamp + logit
        opacity = opacities[i]
        if opacity < min_opacity:
            opacity = min_opacity
        elif opacity > max_opacity:
            opacity = max_opacity

        opacities[i] = np.log(opacity / (1.0 - opacity))


def _ensure_float32_contiguous(array: Float32Array | None, name: str) -> Float32Array:
    """
    Ensure arrays passed to kernels are float32 and C-contiguous.

    :param array: Array to validate
    :param name: Attribute name (for error messages)
    :return: Array guaranteed to be float32 and contiguous
    """
    if array is None:
        raise ValueError(f"GSData.{name} is required.")

    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)

    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    return array


def apply_pre_activations(
    data: GSData,
    *,
    min_scale: float = float(_DEFAULT_MIN_SCALE),
    max_scale: float = float(_DEFAULT_MAX_SCALE),
    min_quat_norm: float = float(_DEFAULT_MIN_NORM),
    inplace: bool = True,
) -> GSData:
    """
    Activate GSData attributes (scales, opacities, quaternions) in a single fused pass.

    This function uses a fused Numba kernel that processes all three attributes together
    for optimal performance (~8-15x faster than individual operations).

    :param data: GSData instance to process
    :param min_scale: Minimum allowed scale value after exponentiation
    :param max_scale: Maximum allowed scale value after exponentiation
    :param min_quat_norm: Norm floor for normalizing quaternions (avoids NaNs)
    :param inplace: If False, returns a copy before activation
    :return: GSData with activated attributes (either modified in-place or copy)

    Example:
        >>> import gsply
        >>> data = gsply.plyread("scene_logits.ply")
        >>> gsply.apply_pre_activations(data, inplace=True)
    """
    # Lazy import to avoid circular dependency (cached in function attribute)
    if not hasattr(apply_pre_activations, "_GSData"):
        from gsply.gsdata import GSData as _GSData

        apply_pre_activations._GSData = _GSData

    if min_scale <= 0:
        raise ValueError("min_scale must be positive to avoid degenerate exponentiation results.")
    if max_scale <= 0 or max_scale < min_scale:
        raise ValueError("max_scale must be positive and >= min_scale.")
    if min_quat_norm <= 0:
        raise ValueError("min_quat_norm must be positive.")

    if not inplace:
        data = data.copy()

    scales = _ensure_float32_contiguous(data.scales, "scales")
    opacities = _ensure_float32_contiguous(data.opacities, "opacities")
    quats = _ensure_float32_contiguous(data.quats, "quats")

    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must have shape [N, 3].")
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError("quats must have shape [N, 4].")

    if opacities.ndim == 2 and opacities.shape[1] == 1:
        opacity_view = opacities.reshape(opacities.shape[0])
    elif opacities.ndim == 1:
        opacity_view = opacities
    else:
        raise ValueError("opacities must be 1D or have shape [N, 1].")

    n_gaussians = scales.shape[0]
    if quats.shape[0] != n_gaussians or opacity_view.shape[0] != n_gaussians:
        raise ValueError("scales, opacities, and quats must have matching lengths.")

    _activate_gaussians_numba(
        scales,
        opacity_view,
        quats,
        np.float32(min_scale),
        np.float32(max_scale),
        np.float32(min_quat_norm),
    )

    data.scales = scales
    data.opacities = opacities
    data.quats = quats

    logger.debug(
        "[PreActivation] Activated %d Gaussians (min_scale=%.2e, max_scale=%.2f, min_quat_norm=%.2e)",
        scales.shape[0],
        min_scale,
        max_scale,
        min_quat_norm,
    )

    return data


def apply_pre_deactivations(
    data: GSData,
    *,
    min_scale: float = 1e-9,
    min_opacity: float = 1e-4,
    max_opacity: float = 1.0 - 1e-4,
    inplace: bool = True,
) -> GSData:
    """
    Deactivate GSData attributes (scales, opacities) in a single fused pass.

    This function uses a fused Numba kernel that processes scales and opacities together
    for optimal performance (~8-15x faster than individual operations).

    :param data: GSData instance to process
    :param min_scale: Minimum allowed scale value before logarithm
    :param min_opacity: Minimum allowed opacity value before logit
    :param max_opacity: Maximum allowed opacity value before logit
    :param inplace: If False, returns a copy before deactivation
    :return: GSData with deactivated attributes (either modified in-place or copy)

    Example:
        >>> import gsply
        >>> data = gsply.GSData(...)  # Linear format
        >>> gsply.apply_pre_deactivations(data, inplace=True)
    """
    # Lazy import to avoid circular dependency (cached in function attribute)
    if not hasattr(apply_pre_deactivations, "_GSData"):
        from gsply.gsdata import GSData as _GSData

        apply_pre_deactivations._GSData = _GSData

    if min_scale <= 0:
        raise ValueError("min_scale must be positive to avoid degenerate logarithm results.")
    if min_opacity <= 0 or max_opacity >= 1.0:
        raise ValueError("min_opacity must be positive and max_opacity must be < 1.0.")
    if max_opacity <= min_opacity:
        raise ValueError("max_opacity must be > min_opacity.")

    if not inplace:
        data = data.copy()

    scales = _ensure_float32_contiguous(data.scales, "scales")
    opacities = _ensure_float32_contiguous(data.opacities, "opacities")

    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must have shape [N, 3].")

    if opacities.ndim == 2 and opacities.shape[1] == 1:
        opacity_view = opacities.reshape(opacities.shape[0])
    elif opacities.ndim == 1:
        opacity_view = opacities
    else:
        raise ValueError("opacities must be 1D or have shape [N, 1].")

    n_gaussians = scales.shape[0]
    if opacity_view.shape[0] != n_gaussians:
        raise ValueError("scales and opacities must have matching lengths.")

    _deactivate_gaussians_numba(
        scales,
        opacity_view,
        np.float32(min_scale),
        np.float32(min_opacity),
        np.float32(max_opacity),
    )

    data.scales = scales
    data.opacities = opacities

    logger.debug(
        "[PreDeactivation] Deactivated %d Gaussians (min_scale=%.2e, min_opacity=%.2e, max_opacity=%.2e)",
        scales.shape[0],
        min_scale,
        min_opacity,
        max_opacity,
    )

    return data


__all__ = [
    "sh2rgb",
    "rgb2sh",
    "SH_C0",
    "sigmoid",
    "logit",
    "apply_pre_activations",
    "apply_pre_deactivations",
]
