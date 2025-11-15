"""Standard benchmark script for CI workflow.

This script runs basic read/write benchmarks on a test file.
Designed to be simple and fast for CI execution.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

import gsply

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def benchmark_file(file_path: str, iterations: int = 10):
    """Benchmark read and write operations on a file.

    Args:
        file_path: Path to PLY file to benchmark
        iterations: Number of iterations to run
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info("=" * 70)
    logger.info(f"Benchmarking: {file_path.name}")
    logger.info("=" * 70)

    # Read the file
    logger.info("\n[1/3] Reading file...")
    data = gsply.plyread(str(file_path))
    num_gaussians = data.means.shape[0]
    sh_degree = data.shN.shape[1] // 3 if data.shN.shape[1] > 0 else 0

    logger.info(f"  Gaussians: {num_gaussians:,}")
    logger.info(f"  SH degree: {sh_degree}")
    logger.info(f"  File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Benchmark read
    logger.info(f"\n[2/3] Benchmarking read ({iterations} iterations)...")
    read_times = []
    for i in range(iterations):
        start = time.perf_counter()
        _ = gsply.plyread(str(file_path))
        elapsed = (time.perf_counter() - start) * 1000
        read_times.append(elapsed)

    read_mean = np.mean(read_times)
    read_std = np.std(read_times)
    read_min = np.min(read_times)

    logger.info(f"  Mean: {read_mean:.2f} ms")
    logger.info(f"  Std:  {read_std:.2f} ms")
    logger.info(f"  Min:  {read_min:.2f} ms")
    logger.info(f"  Throughput: {num_gaussians / read_mean * 1000 / 1e6:.1f} M Gaussians/sec")

    # Benchmark write
    logger.info(f"\n[3/3] Benchmarking write ({iterations} iterations)...")
    temp_file = file_path.parent / "temp_benchmark.ply"

    # Warmup
    gsply.plywrite(temp_file, data.means, data.scales, data.quats,
                   data.opacities, data.sh0, data.shN)

    write_times = []
    for i in range(iterations):
        start = time.perf_counter()
        gsply.plywrite(temp_file, data.means, data.scales, data.quats,
                       data.opacities, data.sh0, data.shN)
        elapsed = (time.perf_counter() - start) * 1000
        write_times.append(elapsed)

    write_mean = np.mean(write_times)
    write_std = np.std(write_times)
    write_min = np.min(write_times)

    logger.info(f"  Mean: {write_mean:.2f} ms")
    logger.info(f"  Std:  {write_std:.2f} ms")
    logger.info(f"  Min:  {write_min:.2f} ms")
    logger.info(f"  Throughput: {num_gaussians / write_mean * 1000 / 1e6:.1f} M Gaussians/sec")

    # Cleanup
    if temp_file.exists():
        temp_file.unlink()

    logger.info("\n" + "=" * 70)
    logger.info("Benchmark complete")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Benchmark gsply read/write operations')
    parser.add_argument('--file', type=str, required=True, help='Path to PLY file')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')

    args = parser.parse_args()

    benchmark_file(args.file, args.iterations)


if __name__ == '__main__':
    main()
