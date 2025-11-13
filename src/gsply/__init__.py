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
    >>> # Or unpack if needed
    >>> means, scales, quats, opacities, sh0, shN = data[:6]
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

from gsply.reader import plyread, GSData, decompress_from_bytes
from gsply.writer import plywrite, compress_to_bytes, compress_to_arrays
from gsply.formats import detect_format
from gsply.utils import sh2rgb, rgb2sh, SH_C0

__version__ = "0.1.1"
__all__ = ["plyread", "GSData", "plywrite", "compress_to_bytes",
           "compress_to_arrays", "decompress_from_bytes", "detect_format",
           "sh2rgb", "rgb2sh", "SH_C0", "__version__"]
