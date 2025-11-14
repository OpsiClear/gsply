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
    >>> # Compress/decompress without disk I/O
    >>> compressed = gsply.compress_to_bytes(data)  # For network transfer, etc.
    >>> data_restored = gsply.decompress_from_bytes(compressed)
    >>>
    >>> # Detect format
    >>> is_compressed, sh_degree = gsply.detect_format("model.ply")

Features:
    - Fast with NumPy and Numba JIT acceleration
    - SH degrees 0-3 support (14, 23, 38, 59 properties)
    - Compressed format (PlayCanvas compatible)
    - In-memory compression/decompression (no disk I/O)
    - Ultra-fast (~3-6ms read, ~5-10ms write)
    - Zero-copy optimization (all reads use views)
    - Auto-format detection
    - Numba JIT acceleration (3.8-6x faster compressed I/O)

Performance (400K Gaussians):
    - Read uncompressed: ~6ms (zero-copy views)
    - Read compressed: ~15ms (JIT-accelerated)
    - Write uncompressed: ~23ms
    - Write compressed: ~63ms (JIT-accelerated)
"""

from gsply.formats import detect_format
from gsply.gsdata import GSData
from gsply.reader import decompress_from_bytes, plyread
from gsply.utils import SH_C0, rgb2sh, sh2rgb
from gsply.writer import compress_to_arrays, compress_to_bytes, plywrite

__version__ = "0.2.0"
__all__ = ["plyread", "GSData", "plywrite", "compress_to_bytes",
           "compress_to_arrays", "decompress_from_bytes", "detect_format",
           "sh2rgb", "rgb2sh", "SH_C0", "__version__"]

# Optional PyTorch integration (only available if torch is installed)
try:
    from gsply.torch import GSTensor
    __all__.append("GSTensor")
except ImportError:
    # PyTorch not installed, GSTensor not available
    pass
