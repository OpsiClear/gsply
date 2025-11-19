"""GPU-accelerated PLY I/O functions.

This module provides direct GPU read/write functions for compressed PLY files,
matching the API style of the main gsply module.
"""

import logging
from pathlib import Path

from gsply.gsdata import DataFormat
from gsply.torch.compression import read_compressed_gpu, write_compressed_gpu
from gsply.torch.gstensor import GSTensor

logger = logging.getLogger(__name__)


def plyread_gpu(file_path: str | Path, device: str = "cuda") -> GSTensor:
    """Read compressed PLY file directly to GPU.

    Uses GPU-accelerated decompression for maximum performance. Reads compressed
    binary data from disk and decompresses directly on GPU, avoiding CPU
    decompression overhead.

    :param file_path: Path to compressed PLY file
    :param device: Target GPU device (default "cuda")
    :returns: GSTensor with decompressed data on GPU

    Performance:
        - 4-5x faster than CPU decompression + GPU transfer
        - Direct GPU memory allocation (no intermediate CPU copies)
        - Optimized batch memory transfer (1.71x speedup)
        - ~19ms for 365K Gaussians (19 M/s throughput)

    Example:
        >>> import gsply
        >>> gstensor = gsply.plyread_gpu("scene.compressed.ply", device="cuda")
        >>> print(f"Loaded {len(gstensor):,} Gaussians on GPU")
        >>> positions_gpu = gstensor.means  # Already on GPU
        >>> colors_gpu = gstensor.sh0

    Note:
        - Only supports compressed PLY format (auto-detected)
        - Requires PyTorch to be installed
        - Returns GSTensor (not GSData) for GPU operations
    """
    file_path = Path(file_path)
    return read_compressed_gpu(file_path, device)


def plywrite_gpu(
    file_path: str | Path,
    gstensor: GSTensor,
    compressed: bool = True,
) -> None:
    """Write GSTensor to compressed PLY file using GPU compression.

    Performs compression on GPU for maximum performance. Faster than downloading
    to CPU and using CPU compression.

    :param file_path: Output file path
    :param gstensor: GSTensor on GPU to compress and save
    :param compressed: If True, write compressed format (default True, required for GPU path)

    Performance:
        - 4-5x faster compression than CPU Numba
        - GPU reduction for chunk bounds (instant)
        - Minimal CPU-GPU data transfer
        - ~18ms for 365K Gaussians (20 M/s throughput)

    Format:
        - PlayCanvas compressed PLY format
        - 3.8-14.5x compression ratio
        - 256 Gaussians per chunk with quantization

    Example:
        >>> import gsply
        >>> # Read to GPU
        >>> gstensor = gsply.plyread_gpu("input.compressed.ply", device="cuda")
        >>> # ... GPU operations ...
        >>> # Write back to compressed file
        >>> gsply.plywrite_gpu("output.compressed.ply", gstensor)

    Note:
        - Only supports compressed format (GPU compression is optimized for this)
        - Requires PyTorch to be installed
        - GSTensor must be on GPU (use gstensor.to("cuda") if needed)
    """
    if not compressed:
        raise ValueError(
            "plywrite_gpu only supports compressed format. "
            "Use plywrite() for uncompressed format."
        )

    # Ensure data is in PLY format before writing (log-scales, logit-opacities)
    # Check format flags and convert if needed
    scales_format = gstensor._format.get("scales")
    opacities_format = gstensor._format.get("opacities")

    # Convert to PLY format if not already in PLY format
    if scales_format != DataFormat.SCALES_PLY or opacities_format != DataFormat.OPACITIES_PLY:
        logger.debug(
            f"[GPU Write] Converting from {scales_format}/{opacities_format} to PLY format before writing"
        )
        gstensor = gstensor.normalize(inplace=False)  # Create copy with PLY format

    file_path = Path(file_path)
    write_compressed_gpu(file_path, gstensor)


__all__ = ["plyread_gpu", "plywrite_gpu"]
