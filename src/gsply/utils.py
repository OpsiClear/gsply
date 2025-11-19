"""Utility functions for Gaussian Splatting operations."""

import numpy as np
from numba import jit, prange

from gsply.formats import SH_C0


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


__all__ = ["sh2rgb", "rgb2sh", "SH_C0", "sigmoid", "logit"]
