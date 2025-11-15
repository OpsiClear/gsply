"""Verify benchmark data integrity and test a quick sanity check."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

import gsply

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_file(file_path: Path):
    """Verify a single file can be read correctly."""
    logger.info(f"Verifying: {file_path.name}")

    try:
        # Detect format
        is_compressed, sh_degree = gsply.detect_format(file_path)
        logger.info(f"  Format: {'Compressed' if is_compressed else 'Uncompressed'}, SH degree: {sh_degree}")

        # Read file
        start = time.perf_counter()
        data = gsply.plyread(file_path)
        elapsed = (time.perf_counter() - start) * 1000

        # Check data
        num_gaussians = data.means.shape[0]
        file_size = file_path.stat().st_size / (1024 * 1024)

        logger.info(f"  Gaussians: {num_gaussians:,}")
        logger.info(f"  File size: {file_size:.2f} MB")
        logger.info(f"  Read time: {elapsed:.2f} ms")
        logger.info(f"  Throughput: {num_gaussians / (elapsed/1000) / 1e6:.1f} M/s")

        # Verify shapes
        assert data.means.shape == (num_gaussians, 3), "Invalid means shape"
        assert data.scales.shape == (num_gaussians, 3), "Invalid scales shape"
        assert data.quats.shape == (num_gaussians, 4), "Invalid quats shape"
        assert data.opacities.shape == (num_gaussians,), "Invalid opacities shape"
        assert data.sh0.shape == (num_gaussians, 3), "Invalid sh0 shape"

        logger.info("  [OK] All shapes valid")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"  [FAIL] {e}")
        logger.info("")
        return False


def main():
    """Run verification on test data files."""
    logger.info("=" * 80)
    logger.info("BENCHMARK DATA VERIFICATION")
    logger.info("=" * 80)
    logger.info("")

    test_data_dir = Path(__file__).parent / "test_data"

    if not test_data_dir.exists():
        logger.error(f"Test data directory not found: {test_data_dir}")
        return

    # Find test files
    test_files = sorted(test_data_dir.glob("*.ply"))

    if not test_files:
        logger.warning("No test files found")
        return

    logger.info(f"Found {len(test_files)} test files")
    logger.info("")

    # Verify each file
    results = []
    for file_path in test_files:
        result = verify_file(file_path)
        results.append((file_path.name, result))

    # Summary
    logger.info("=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        logger.info(f"{status} {name}")

    logger.info("")
    logger.info(f"Passed: {passed}/{total}")

    if passed == total:
        logger.info("")
        logger.info("All verification tests passed!")
    else:
        logger.info("")
        logger.info(f"WARNING: {total - passed} files failed verification")


if __name__ == "__main__":
    main()
