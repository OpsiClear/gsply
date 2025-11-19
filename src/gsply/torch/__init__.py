"""PyTorch integration for gsply (optional dependency).

This submodule provides GPU-accelerated tensor operations for Gaussian Splatting data.
Requires PyTorch to be installed separately.

Install PyTorch with: pip install torch
"""

# Check if PyTorch is available
try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, RuntimeError):
    # Catch both ImportError (not installed) and RuntimeError (import issues)
    # RuntimeError can occur with PyTorch version incompatibilities or bugs
    TORCH_AVAILABLE = False
    torch = None

# Import GSTensor and GPU I/O functions only if PyTorch is available
if TORCH_AVAILABLE:
    from .gstensor import GSTensor
    from .io import plyread_gpu, plywrite_gpu

    __all__ = ["GSTensor", "plyread_gpu", "plywrite_gpu", "TORCH_AVAILABLE"]
else:
    __all__ = ["TORCH_AVAILABLE"]


def _check_torch_available():
    """Raise helpful error if PyTorch is not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. GSTensor requires PyTorch.\nInstall with: pip install torch"
        )
