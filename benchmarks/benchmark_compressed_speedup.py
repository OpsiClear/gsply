"""Benchmark compressed format speedup for CI workflow.

This script benchmarks compressed read/write operations and compares
against uncompressed format to show speedup and compression ratio.
"""

import numpy as np
import time
from pathlib import Path
import gsply
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def benchmark_compressed_speedup(num_gaussians: int = 50000, iterations: int = 10):
    """Benchmark compressed vs uncompressed format.

    Args:
        num_gaussians: Number of Gaussians to test
        iterations: Number of iterations to run
    """
    logger.info("=" * 70)
    logger.info("Compressed Format Speedup Benchmark")
    logger.info("=" * 70)
    logger.info(f"\nTest configuration:")
    logger.info(f"  Gaussians: {num_gaussians:,}")
    logger.info(f"  Iterations: {iterations}")
    logger.info(f"  SH degree: 3")

    # Generate test data
    np.random.seed(42)
    means = np.random.randn(num_gaussians, 3).astype(np.float32) * 2.0
    scales = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.5
    quats = np.random.randn(num_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.randn(num_gaussians).astype(np.float32)
    sh0 = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.3
    shN = np.random.randn(num_gaussians, 15, 3).astype(np.float32) * 0.1

    test_dir = Path("benchmarks/test_data")
    test_dir.mkdir(parents=True, exist_ok=True)

    unc_file = test_dir / "speedup_uncompressed.ply"
    cmp_file = test_dir / "speedup_compressed.compressed.ply"

    # Warmup
    gsply.plywrite(unc_file, means, scales, quats, opacities, sh0, shN, compressed=False)
    gsply.plywrite(cmp_file, means, scales, quats, opacities, sh0, shN, compressed=True)
    _ = gsply.plyread(str(unc_file))
    _ = gsply.plyread(str(cmp_file))

    # Benchmark uncompressed write
    logger.info("\n[1/4] Benchmarking uncompressed write...")
    unc_write_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        gsply.plywrite(unc_file, means, scales, quats, opacities, sh0, shN, compressed=False)
        elapsed = (time.perf_counter() - start) * 1000
        unc_write_times.append(elapsed)

    unc_write_mean = np.mean(unc_write_times)
    logger.info(f"  Mean: {unc_write_mean:.2f} ms")
    logger.info(f"  Throughput: {num_gaussians / unc_write_mean * 1000 / 1e6:.1f} M Gaussians/sec")

    # Benchmark compressed write
    logger.info("\n[2/4] Benchmarking compressed write...")
    cmp_write_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        gsply.plywrite(cmp_file, means, scales, quats, opacities, sh0, shN, compressed=True)
        elapsed = (time.perf_counter() - start) * 1000
        cmp_write_times.append(elapsed)

    cmp_write_mean = np.mean(cmp_write_times)
    logger.info(f"  Mean: {cmp_write_mean:.2f} ms")
    logger.info(f"  Throughput: {num_gaussians / cmp_write_mean * 1000 / 1e6:.1f} M Gaussians/sec")

    # Benchmark uncompressed read
    logger.info("\n[3/4] Benchmarking uncompressed read...")
    unc_read_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = gsply.plyread(str(unc_file))
        elapsed = (time.perf_counter() - start) * 1000
        unc_read_times.append(elapsed)

    unc_read_mean = np.mean(unc_read_times)
    logger.info(f"  Mean: {unc_read_mean:.2f} ms")
    logger.info(f"  Throughput: {num_gaussians / unc_read_mean * 1000 / 1e6:.1f} M Gaussians/sec")

    # Benchmark compressed read
    logger.info("\n[4/4] Benchmarking compressed read...")
    cmp_read_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = gsply.plyread(str(cmp_file))
        elapsed = (time.perf_counter() - start) * 1000
        cmp_read_times.append(elapsed)

    cmp_read_mean = np.mean(cmp_read_times)
    logger.info(f"  Mean: {cmp_read_mean:.2f} ms")
    logger.info(f"  Throughput: {num_gaussians / cmp_read_mean * 1000 / 1e6:.1f} M Gaussians/sec")

    # File size comparison
    unc_size = unc_file.stat().st_size / 1024 / 1024
    cmp_size = cmp_file.stat().st_size / 1024 / 1024
    compression_ratio = unc_size / cmp_size

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\nFile sizes:")
    logger.info(f"  Uncompressed: {unc_size:.2f} MB")
    logger.info(f"  Compressed:   {cmp_size:.2f} MB")
    logger.info(f"  Compression ratio: {compression_ratio:.1f}x")

    logger.info(f"\nWrite performance:")
    logger.info(f"  Uncompressed: {unc_write_mean:.2f} ms")
    logger.info(f"  Compressed:   {cmp_write_mean:.2f} ms")
    if cmp_write_mean < unc_write_mean:
        speedup = unc_write_mean / cmp_write_mean
        logger.info(f"  Compressed is {speedup:.2f}x faster")
    else:
        slowdown = cmp_write_mean / unc_write_mean
        logger.info(f"  Compressed is {slowdown:.2f}x slower")

    logger.info(f"\nRead performance:")
    logger.info(f"  Uncompressed: {unc_read_mean:.2f} ms")
    logger.info(f"  Compressed:   {cmp_read_mean:.2f} ms")
    if cmp_read_mean < unc_read_mean:
        speedup = unc_read_mean / cmp_read_mean
        logger.info(f"  Compressed is {speedup:.2f}x faster")
    else:
        slowdown = cmp_read_mean / unc_read_mean
        logger.info(f"  Compressed is {slowdown:.2f}x slower")

    logger.info("\n" + "=" * 70)

    # Cleanup
    if unc_file.exists():
        unc_file.unlink()
    if cmp_file.exists():
        cmp_file.unlink()


def main():
    benchmark_compressed_speedup()


if __name__ == '__main__':
    main()
