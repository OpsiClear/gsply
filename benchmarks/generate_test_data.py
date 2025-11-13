"""Generate synthetic Gaussian splatting data for benchmarking.

Creates test PLY files in various formats:
- Uncompressed: SH degree 0, 1, 2, 3
- Compressed: SH degree 0, 1, 2, 3

Allows benchmarking all optimization paths.
"""

import numpy as np
from pathlib import Path
import gsply
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_gaussian_data(num_gaussians: int, sh_degree: int = 3):
    """Generate synthetic Gaussian splatting data.

    Args:
        num_gaussians: Number of Gaussians to generate
        sh_degree: Spherical harmonics degree (0-3)

    Returns:
        Tuple of (means, scales, quats, opacities, sh0, shN)
    """
    logger.info(f"Generating {num_gaussians:,} Gaussians with SH degree {sh_degree}")

    # Random seed for reproducibility
    np.random.seed(42)

    # Positions: centered around origin with some spread
    means = np.random.randn(num_gaussians, 3).astype(np.float32) * 2.0

    # Scales: log-space (typical for 3DGS)
    scales = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.5 - 1.0

    # Quaternions: random rotations (normalize after)
    quats = np.random.randn(num_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    # Opacities: logit space
    opacities = np.random.randn(num_gaussians).astype(np.float32)

    # SH degree 0 (DC component) - RGB-ish colors
    sh0 = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.3

    # Higher-order SH coefficients
    if sh_degree > 0:
        # Number of additional SH coefficients per degree
        # These are the total f_rest coefficients (already accounting for 3 channels)
        sh_counts = {
            0: 0,   # DC only (no f_rest)
            1: 9,   # 9 f_rest coefficients
            2: 24,  # 24 f_rest coefficients
            3: 45,  # 45 f_rest coefficients
        }
        num_sh_coeffs = sh_counts[sh_degree]
        # shN should be (N, num_sh_coeffs) - flattened across channels
        shN = np.random.randn(num_gaussians, num_sh_coeffs).astype(np.float32) * 0.1
    else:
        shN = None

    return means, scales, quats, opacities, sh0, shN


def main():
    """Generate test data files."""
    output_dir = Path("benchmarks/test_data")
    output_dir.mkdir(exist_ok=True)

    # Test configurations
    configs = [
        # (num_gaussians, sh_degree, compressed)
        (100_000, 0, False),   # Small, SH0, uncompressed
        (100_000, 3, False),   # Small, SH3, uncompressed
        (400_000, 0, False),   # Medium, SH0, uncompressed
        (400_000, 3, False),   # Medium, SH3, uncompressed
        (400_000, 0, True),    # Medium, SH0, compressed
        (400_000, 3, True),    # Medium, SH3, compressed
        (1_000_000, 0, False), # Large, SH0, uncompressed
        (1_000_000, 3, False), # Large, SH3, uncompressed
    ]

    for num_gaussians, sh_degree, compressed in configs:
        # Generate data
        means, scales, quats, opacities, sh0, shN = generate_gaussian_data(
            num_gaussians, sh_degree
        )

        # Construct filename
        format_str = "compressed" if compressed else "uncompressed"
        filename = f"synthetic_{num_gaussians//1000}k_sh{sh_degree}_{format_str}.ply"
        output_file = output_dir / filename

        # Write file
        logger.info(f"Writing {filename}...")
        try:
            gsply.plywrite(
                output_file,
                means, scales, quats, opacities, sh0, shN,
                compressed=compressed
            )
        except Exception as e:
            logger.error(f"  Failed to write {filename}: {e}")
            continue

        # Check what file was actually created
        if compressed:
            # Compressed writes may add .compressed to the filename
            if not output_file.exists():
                # Check for .compressed.ply variant
                compressed_variant = output_file.with_suffix('.compressed.ply')
                if compressed_variant.exists():
                    output_file = compressed_variant
                    filename = compressed_variant.name
                else:
                    logger.error(f"  File not found after write: {output_file}")
                    continue

        # Verify and report size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"  Created: {filename} ({file_size_mb:.2f} MB)")

        # Verify readable
        data = gsply.plyread(output_file)
        assert data.means.shape[0] == num_gaussians
        logger.info(f"  Verified: {num_gaussians:,} Gaussians read back successfully")

    logger.info(f"\nTest data created in: {output_dir.absolute()}")
    logger.info("Use these files with benchmark.py:")
    logger.info(f"  uv run python benchmarks/benchmark.py --file benchmarks/test_data/synthetic_400k_sh3_uncompressed.ply")


if __name__ == "__main__":
    main()
