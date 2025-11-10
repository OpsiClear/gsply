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
    >>> # Zero-copy reading (1.65x faster, default behavior)
    >>> data = gsply.plyread("model.ply", fast=True)
    >>>
    >>> # Safe copy reading (if you need independent arrays)
    >>> data = gsply.plyread("model.ply", fast=False)
    >>>
    >>> # Write uncompressed PLY file
    >>> gsply.plywrite("output.ply", data.means, data.scales, data.quats,
    ...                data.opacities, data.sh0, data.shN)
    >>>
    >>> # Write compressed PLY file (saves as "output.compressed.ply")
    >>> gsply.plywrite("output.ply", data.means, data.scales, data.quats,
    ...                data.opacities, data.sh0, data.shN, compressed=True)
    >>>
    >>> # Detect format
    >>> is_compressed, sh_degree = gsply.detect_format("model.ply")

Features:
    - Zero dependencies (pure Python + numpy)
    - SH degrees 0-3 support (14, 23, 38, 59 properties)
    - Compressed format (PlayCanvas compatible)
    - Ultra-fast (~3-5ms read, ~5-10ms write)
    - Zero-copy optimization (1.65x faster reads)
    - Auto-format detection

Performance (50K Gaussians):
    - Read uncompressed (fast=True): ~3ms (SH degree 3, zero-copy)
    - Read uncompressed (fast=False): ~5ms (SH degree 3, safe copies)
    - Read compressed: ~30-50ms (with decompression)
    - Write uncompressed: ~5-10ms
"""

from gsply.reader import plyread, GSData
from gsply.writer import plywrite
from gsply.formats import detect_format

__version__ = "0.1.0"
__all__ = ["plyread", "GSData", "plywrite", "detect_format", "__version__"]
