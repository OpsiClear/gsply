"""Utility functions for Gaussian Splatting operations."""

import numpy as np

from gsply.formats import SH_C0


def sh2rgb(sh: np.ndarray | float) -> np.ndarray | float:
    """Convert SH DC coefficients to RGB colors.

    Args:
        sh: SH DC coefficients (N, 3) or scalar

    Returns:
        RGB colors in [0, 1] range

    Example:
        >>> import gsply
        >>> sh = np.array([[0.0, 0.5, -0.5]])
        >>> rgb = gsply.sh2rgb(sh)
        >>> print(rgb)  # [[0.5, 0.641, 0.359]]
    """
    return sh * SH_C0 + 0.5


def rgb2sh(rgb: np.ndarray | float) -> np.ndarray | float:
    """Convert RGB colors to SH DC coefficients.

    Args:
        rgb: RGB colors in [0, 1] range (N, 3) or scalar

    Returns:
        SH DC coefficients

    Example:
        >>> import gsply
        >>> rgb = np.array([[1.0, 0.5, 0.0]])
        >>> sh = gsply.rgb2sh(rgb)
    """
    return (rgb - 0.5) / SH_C0


__all__ = ["sh2rgb", "rgb2sh", "SH_C0"]
