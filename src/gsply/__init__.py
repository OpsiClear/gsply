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
    >>> # Detect format
    >>> is_compressed, sh_degree = gsply.detect_format("model.ply")

Features:
    - Zero dependencies (pure Python + numpy)
    - SH degrees 0-3 support (14, 23, 38, 59 properties)
    - Compressed format (PlayCanvas compatible)
    - Ultra-fast (~3-6ms read, ~5-10ms write)
    - Zero-copy optimization (all reads use views)
    - Auto-format detection
    - Optional JIT acceleration (3.8-6x faster compressed I/O)

Performance (400K Gaussians):
    - Read uncompressed: ~6ms (zero-copy views)
    - Read compressed (with JIT): ~15ms
    - Read compressed (no JIT): ~90ms
    - Write uncompressed: ~23ms
    - Write compressed (with JIT): ~63ms
    - Write compressed (no JIT): ~240ms
"""

from gsply.reader import plyread, GSData
from gsply.writer import plywrite
from gsply.formats import detect_format
from gsply.utils import sh2rgb, rgb2sh, SH_C0

__version__ = "0.1.1"
__all__ = ["plyread", "GSData", "plywrite", "detect_format",
           "sh2rgb", "rgb2sh", "SH_C0", "__version__"]
