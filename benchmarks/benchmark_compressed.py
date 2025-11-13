"""Benchmark compressed format specifically to measure optimization impact."""

import numpy as np
import time
from pathlib import Path
import gsply
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_compressed_write(num_gaussians: int, sh_degree: int, iterations: int = 20):
    """Benchmark compressed write performance."""
    logger.info(f"Benchmarking compressed write: {num_gaussians:,} Gaussians, SH degree {sh_degree}")

    # Generate test data
    np.random.seed(42)
    means = np.random.randn(num_gaussians, 3).astype(np.float32) * 2.0
    scales = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.5
    quats = np.random.randn(num_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.randn(num_gaussians).astype(np.float32)
    sh0 = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.3

    if sh_degree > 0:
        sh_counts = {1: 9, 2: 24, 3: 45}
        shN = np.random.randn(num_gaussians, sh_counts[sh_degree]).astype(np.float32) * 0.1
    else:
        shN = None

    output_file = Path(f"benchmarks/test_data/benchmark_compressed_{num_gaussians}_{sh_degree}.ply")

    # Warmup
    for _ in range(3):
        gsply.plywrite(output_file, means, scales, quats, opacities, sh0, shN, compressed=True)

    # Benchmark
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        gsply.plywrite(output_file, means, scales, quats, opacities, sh0, shN, compressed=True)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)

    # Check file exists (compressed adds .compressed)
    if not output_file.exists():
        output_file = output_file.with_suffix('.compressed.ply')

    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    logger.info(f"  Mean: {mean_time:.2f}ms +/- {std_time:.2f}ms")
    logger.info(f"  Min:  {min_time:.2f}ms")
    logger.info(f"  File: {file_size_mb:.2f} MB")
    logger.info(f"  Throughput: {num_gaussians / (mean_time/1000) / 1e6:.2f}M Gaussians/sec")

    # Cleanup
    output_file.unlink()

    return mean_time, std_time


def benchmark_compressed_read(num_gaussians: int, sh_degree: int, iterations: int = 20):
    """Benchmark compressed read performance."""
    logger.info(f"Benchmarking compressed read: {num_gaussians:,} Gaussians, SH degree {sh_degree}")

    # Generate and write test data
    np.random.seed(42)
    means = np.random.randn(num_gaussians, 3).astype(np.float32) * 2.0
    scales = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.5
    quats = np.random.randn(num_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.randn(num_gaussians).astype(np.float32)
    sh0 = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.3

    if sh_degree > 0:
        sh_counts = {1: 9, 2: 24, 3: 45}
        shN = np.random.randn(num_gaussians, sh_counts[sh_degree]).astype(np.float32) * 0.1
    else:
        shN = None

    output_file = Path(f"benchmarks/test_data/benchmark_compressed_read_{num_gaussians}_{sh_degree}.ply")
    gsply.plywrite(output_file, means, scales, quats, opacities, sh0, shN, compressed=True)

    # Check actual filename
    if not output_file.exists():
        output_file = output_file.with_suffix('.compressed.ply')

    # Warmup
    for _ in range(3):
        data = gsply.plyread(output_file)

    # Benchmark
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        data = gsply.plyread(output_file)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)

    logger.info(f"  Mean: {mean_time:.2f}ms +/- {std_time:.2f}ms")
    logger.info(f"  Min:  {min_time:.2f}ms")
    logger.info(f"  Throughput: {num_gaussians / (mean_time/1000) / 1e6:.2f}M Gaussians/sec")

    # Cleanup
    output_file.unlink()

    return mean_time, std_time


def main():
    """Run comprehensive compressed format benchmarks."""
    logger.info("=" * 80)
    logger.info("COMPRESSED FORMAT BENCHMARK")
    logger.info("=" * 80)
    logger.info("")

    # Test configurations
    configs = [
        (100_000, 0),
        (100_000, 3),
        (400_000, 0),
        (400_000, 3),
    ]

    logger.info("COMPRESSED WRITE PERFORMANCE")
    logger.info("-" * 80)
    for num_gaussians, sh_degree in configs:
        benchmark_compressed_write(num_gaussians, sh_degree)
        logger.info("")

    logger.info("COMPRESSED READ PERFORMANCE")
    logger.info("-" * 80)
    for num_gaussians, sh_degree in configs:
        benchmark_compressed_read(num_gaussians, sh_degree)
        logger.info("")

    logger.info("=" * 80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
