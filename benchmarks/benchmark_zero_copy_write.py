"""Comprehensive benchmark for zero-copy write optimization.

Compares GSData direct write (zero-copy) vs unpacked array write (standard path).
Tests on real PLY files which have _base arrays for zero-copy optimization.
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

import gsply

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def benchmark_zero_copy(file_path: str, iterations: int = 20):
    """Benchmark zero-copy vs standard write paths.

    Args:
        file_path: Path to PLY file (must be real file for zero-copy to work)
        iterations: Number of iterations for each test
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info("=" * 80)
    logger.info("ZERO-COPY WRITE OPTIMIZATION BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"File: {file_path.name}")
    logger.info(f"Iterations: {iterations}")
    logger.info("=" * 80)

    # Read file (creates GSData with _base)
    data = gsply.plyread(str(file_path))
    num_gaussians = len(data)
    sh_degree = data.shN.shape[1] // 3 if data.shN.size > 0 else 0

    logger.info(f"\n[File Info]")
    logger.info(f"  Gaussians:  {num_gaussians:,}")
    logger.info(f"  SH degree:  {sh_degree}")
    logger.info(f"  File size:  {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"  Has _base:  {data._base is not None}")
    if data._base is not None:
        logger.info(f"  _base shape: {data._base.shape}")
        logger.info(f"  _base size:  {data._base.nbytes / 1024 / 1024:.2f} MB")

    # Test 1: Direct GSData write (zero-copy path)
    logger.info(f"\n[Test 1] Direct GSData Write (Zero-Copy Path)")
    logger.info(f"  Method: gsply.plywrite(path, data)")

    output_file = Path(tempfile.mktemp(suffix=".ply"))
    times_direct = []

    for i in range(iterations):
        start = time.perf_counter()
        gsply.plywrite(str(output_file), data)
        elapsed = (time.perf_counter() - start) * 1000
        times_direct.append(elapsed)

        if i == 0:
            file_size = output_file.stat().st_size / 1024 / 1024
            logger.info(f"  Output size: {file_size:.2f} MB")

        output_file.unlink()

    mean_direct = np.mean(times_direct)
    std_direct = np.std(times_direct)
    min_direct = np.min(times_direct)
    max_direct = np.max(times_direct)

    logger.info(f"  Mean:  {mean_direct:.2f} ms")
    logger.info(f"  Std:   {std_direct:.2f} ms")
    logger.info(f"  Min:   {min_direct:.2f} ms")
    logger.info(f"  Max:   {max_direct:.2f} ms")
    logger.info(f"  Throughput: {num_gaussians / mean_direct * 1000 / 1e6:.1f} M Gaussians/sec")

    # Test 2: Unpacked array write (standard path)
    logger.info(f"\n[Test 2] Unpacked Array Write (Standard Path)")
    logger.info(f"  Method: gsply.plywrite(path, *data.unpack())")

    output_file2 = Path(tempfile.mktemp(suffix=".ply"))
    times_unpacked = []

    for i in range(iterations):
        start = time.perf_counter()
        gsply.plywrite(str(output_file2), *data.unpack())
        elapsed = (time.perf_counter() - start) * 1000
        times_unpacked.append(elapsed)
        output_file2.unlink()

    mean_unpacked = np.mean(times_unpacked)
    std_unpacked = np.std(times_unpacked)
    min_unpacked = np.min(times_unpacked)
    max_unpacked = np.max(times_unpacked)

    logger.info(f"  Mean:  {mean_unpacked:.2f} ms")
    logger.info(f"  Std:   {std_unpacked:.2f} ms")
    logger.info(f"  Min:   {min_unpacked:.2f} ms")
    logger.info(f"  Max:   {max_unpacked:.2f} ms")
    logger.info(f"  Throughput: {num_gaussians / mean_unpacked * 1000 / 1e6:.1f} M Gaussians/sec")

    # Test 3: Verify byte-for-byte equivalence
    logger.info(f"\n[Test 3] Output Verification")
    gsply.plywrite(str(output_file), data)
    gsply.plywrite(str(output_file2), *data.unpack())

    with open(output_file, "rb") as f1, open(output_file2, "rb") as f2:
        bytes1 = f1.read()
        bytes2 = f2.read()

    if bytes1 == bytes2:
        logger.info(f"  [OK] Both methods produce identical output")
    else:
        logger.info(f"  [FAIL] Outputs differ! {len(bytes1)} vs {len(bytes2)} bytes")

    output_file.unlink()
    output_file2.unlink()

    # Summary
    logger.info(f"\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)

    speedup = mean_unpacked / mean_direct
    logger.info(f"Zero-Copy Speedup: {speedup:.2f}x")
    logger.info(f"  Direct:   {mean_direct:.2f} ms ({num_gaussians / mean_direct * 1000 / 1e6:.1f} M/s)")
    logger.info(f"  Unpacked: {mean_unpacked:.2f} ms ({num_gaussians / mean_unpacked * 1000 / 1e6:.1f} M/s)")

    time_saved = mean_unpacked - mean_direct
    logger.info(f"Time Saved: {time_saved:.2f} ms per write ({time_saved / mean_unpacked * 100:.1f}%)")

    if data._base is not None:
        memory_saved = data._base.nbytes / 1024 / 1024
        logger.info(f"Memory Copied Avoided: {memory_saved:.2f} MB")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark zero-copy write optimization'
    )
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Path to PLY file (must be real file, not generated)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=20,
        help='Number of iterations per test (default: 20)'
    )

    args = parser.parse_args()
    benchmark_zero_copy(args.file, args.iterations)


if __name__ == '__main__':
    main()
