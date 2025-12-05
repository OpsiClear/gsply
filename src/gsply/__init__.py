"""gsply - Fast Gaussian Splatting PLY I/O Library

A pure Python library for ultra-fast reading and writing of Gaussian splatting
PLY files in both uncompressed and compressed formats.

Basic Usage:
    >>> import gsply
    >>>
    >>> # Read PLY file (auto-detect format) - returns GSData
    >>> data = gsply.plyread("model.ply")
    >>> print(f"Loaded {data.means.shape[0]} Gaussians")
    >>> positions = data.means
    >>> colors = data.sh0
    >>>
    >>>
    >>> # Write uncompressed PLY file
    >>> gsply.plywrite("output.ply", data.means, data.scales, data.quats,
    ...                data.opacities, data.sh0, data.shN)
    >>>
    >>> # Write compressed PLY file (saves as "output.compressed.ply")
    >>> gsply.plywrite("output.ply", data.means, data.scales, data.quats,
    ...                data.opacities, data.sh0, data.shN, compressed=True)
    >>>
    >>> # GPU-accelerated I/O (requires PyTorch)
    >>> gstensor = gsply.plyread_gpu("model.compressed.ply", device="cuda")
    >>> gsply.plywrite_gpu("output.compressed.ply", gstensor)
    >>>
    >>> # Compress/decompress without disk I/O
    >>> compressed = gsply.compress_to_bytes(data)  # For network transfer, etc.
    >>> data_restored = gsply.decompress_from_bytes(compressed)
    >>>
    >>> # Detect format
    >>> is_compressed, sh_degree = gsply.detect_format("model.ply")
    >>>
    >>> # SOG format support (requires gsply[sogs])
    >>> data = gsply.sogread("model.sog")

Features:
    - Fast with NumPy and Numba JIT acceleration
    - SH degrees 0-3 support (14, 23, 38, 59 properties)
    - Compressed format (PlayCanvas compatible)
    - In-memory compression/decompression (no disk I/O)
    - Ultra-fast (~3-6ms read, ~5-10ms write)
    - Zero-copy optimization (all reads use views)
    - Auto-format detection
    - Numba JIT acceleration (3.8-6x faster compressed I/O)

Performance (400K Gaussians, SH0):
    - Read uncompressed: ~6ms (70M/sec, zero-copy views)
    - Read compressed: ~8.5ms (47M/sec, JIT-accelerated)
    - Write uncompressed: ~19ms (21M/sec)
    - Write compressed: ~15ms (27M/sec, JIT-accelerated, 71% smaller)
"""

from gsply.formats import detect_format
from gsply.gsdata import GSData, create_ply_format, create_rasterizer_format
from gsply.reader import decompress_from_bytes, plyread
from gsply.utils import (
    SH_C0,
    apply_pre_activations,
    apply_pre_deactivations,
    logit,
    rgb2sh,
    sh2rgb,
    sigmoid,
)
from gsply.writer import compress_to_arrays, compress_to_bytes, plywrite

__version__ = "0.2.12"
__all__ = [
    "plyread",
    "GSData",
    "plywrite",
    "compress_to_bytes",
    "compress_to_arrays",
    "decompress_from_bytes",
    "detect_format",
    "create_ply_format",
    "create_rasterizer_format",
    "sh2rgb",
    "rgb2sh",
    "logit",
    "sigmoid",
    "apply_pre_activations",
    "apply_pre_deactivations",
    "SH_C0",
    "__version__",
]
# Note: GSTensor and sogread are available via lazy import but not in __all__ (they're optional)


def __getattr__(name):
    """Lazy import for optional PyTorch integration.

    This avoids importing torch when just importing gsply, which prevents
    torch-related errors in CI environments where torch may have issues.
    """
    if name == "GSTensor":
        try:
            from gsply.torch import GSTensor

            return GSTensor
        except ImportError as e:
            raise ImportError(
                "GSTensor requires PyTorch to be installed.\nInstall with: pip install torch"
            ) from e
    elif name == "plyread_gpu":
        try:
            from gsply.torch.io import plyread_gpu

            return plyread_gpu
        except ImportError as e:
            raise ImportError(
                "plyread_gpu requires PyTorch to be installed.\nInstall with: pip install torch"
            ) from e
    elif name == "plywrite_gpu":
        try:
            from gsply.torch.io import plywrite_gpu

            return plywrite_gpu
        except ImportError as e:
            raise ImportError(
                "plywrite_gpu requires PyTorch to be installed.\nInstall with: pip install torch"
            ) from e
    elif name == "sogread":
        try:
            from gsply.sog_reader import sogread

            return sogread
        except ImportError as e:
            raise ImportError(
                "sogread requires SOG format support.\n"
                "Install with: pip install gsply[sogs]\n"
                "This installs imagecodecs (fastest WebP decoder)."
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
