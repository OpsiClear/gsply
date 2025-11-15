"""Benchmark: _base construction + zero-copy write vs direct write.

Compares three strategies for writing data without _base:
1. Direct write (standard path - constructs array during write)
2. Convert to _base first, then zero-copy write
3. Zero-copy from existing _base (baseline - data from plyread)

Tests whether it's worth constructing _base for synthetic data.
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

import gsply
from gsply.gsdata import GSData

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_data(num_gaussians: int, sh_degree: int = 0) -> tuple:
    """Generate synthetic data as individual arrays (no _base)."""
    np.random.seed(42)

    means = np.random.randn(num_gaussians, 3).astype(np.float32)
    scales = np.random.randn(num_gaussians, 3).astype(np.float32)
    quats = np.random.randn(num_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.rand(num_gaussians).astype(np.float32)
    sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

    if sh_degree > 0:
        sh_coeffs = [0, 9, 24, 45]
        num_coeffs = sh_coeffs[sh_degree]
        shN = np.random.randn(num_gaussians, num_coeffs // 3, 3).astype(np.float32)
    else:
        shN = np.empty((num_gaussians, 0, 3), dtype=np.float32)

    return means, scales, quats, opacities, sh0, shN


def construct_base_array(means, scales, quats, opacities, sh0, shN) -> np.ndarray:
    """Manually construct a _base array from individual arrays.

    This mimics what write_uncompressed does in the standard path,
    but we do it once upfront to reuse for multiple writes.
    """
    num_gaussians = means.shape[0]
    num_sh_rest = shN.shape[1] if shN.size > 0 else 0
    num_props = 14 + num_sh_rest * 3

    # Construct the array (same as in writer.py)
    base = np.empty((num_gaussians, num_props), dtype=np.float32)
    base[:, 0:3] = means
    base[:, 3:6] = sh0
    base[:, 6] = opacities
    base[:, 7:10] = scales
    base[:, 10:14] = quats

    if num_sh_rest > 0:
        base[:, 14:] = shN.reshape(num_gaussians, -1)

    return base


def benchmark_strategy(
    label: str,
    write_func,
    setup_func=None,
    iterations: int = 20
) -> dict:
    """Benchmark a write strategy.

    Args:
        label: Description of the strategy
        write_func: Function that performs the write
        setup_func: Optional function to run once before timing
        iterations: Number of iterations
    """
    output_file = Path(tempfile.mktemp(suffix=".ply"))

    # Setup phase (not timed)
    setup_time = 0
    if setup_func:
        start = time.perf_counter()
        setup_result = setup_func()
        setup_time = (time.perf_counter() - start) * 1000
    else:
        setup_result = None

    # Write benchmark
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        write_func(output_file, setup_result)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        output_file.unlink()

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)

    return {
        'label': label,
        'setup_ms': setup_time,
        'mean_ms': mean_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'total_ms': setup_time + mean_time
    }


def run_benchmark(num_gaussians: int = 400_000, sh_degree: int = 0, iterations: int = 20):
    """Run comprehensive benchmark comparing write strategies."""
    logger.info("=" * 80)
    logger.info("BASE CONSTRUCTION STRATEGY BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Gaussians: {num_gaussians:,}")
    logger.info(f"SH degree: {sh_degree}")
    logger.info(f"Iterations: {iterations}")
    logger.info("=" * 80)

    # Generate synthetic data (no _base)
    means, scales, quats, opacities, sh0, shN = generate_synthetic_data(num_gaussians, sh_degree)

    logger.info("\n[Test Data]")
    logger.info(f"  Arrays: means, scales, quats, opacities, sh0, shN")
    logger.info(f"  Total memory: {(means.nbytes + scales.nbytes + quats.nbytes + opacities.nbytes + sh0.nbytes + shN.nbytes) / 1024 / 1024:.2f} MB")

    results = []

    # Strategy 1: Direct write (standard path - current implementation)
    logger.info("\n[Strategy 1] Direct Write (Standard Path)")
    logger.info("  Method: Pass individual arrays to plywrite()")
    logger.info("  Constructs array during each write call")

    def write_direct(path, _):
        gsply.plywrite(str(path), means, scales, quats, opacities, sh0, shN)

    result1 = benchmark_strategy("Direct (Standard)", write_direct, iterations=iterations)
    results.append(result1)

    logger.info(f"  Setup:  {result1['setup_ms']:.2f} ms")
    logger.info(f"  Mean:   {result1['mean_ms']:.2f} ms")
    logger.info(f"  Std:    {result1['std_ms']:.2f} ms")
    logger.info(f"  Total:  {result1['total_ms']:.2f} ms (per write)")
    logger.info(f"  Throughput: {num_gaussians / result1['mean_ms'] * 1000 / 1e6:.1f} M/s")

    # Strategy 2: Construct _base once, then use zero-copy writes
    logger.info("\n[Strategy 2] Construct _base + Zero-Copy Write")
    logger.info("  Method: Build _base array once, then pass GSData with _base")
    logger.info("  One-time construction cost + fast zero-copy writes")

    def setup_base_once():
        base = construct_base_array(means, scales, quats, opacities, sh0, shN)
        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            _base=base
        )

    def write_with_base(path, data):
        gsply.plywrite(str(path), data)

    result2 = benchmark_strategy("Construct + ZeroCopy", write_with_base, setup_base_once, iterations)
    results.append(result2)

    logger.info(f"  Setup:  {result2['setup_ms']:.2f} ms (one-time _base construction)")
    logger.info(f"  Mean:   {result2['mean_ms']:.2f} ms (per write)")
    logger.info(f"  Std:    {result2['std_ms']:.2f} ms")
    logger.info(f"  Total:  {result2['total_ms']:.2f} ms (first write)")
    logger.info(f"  Throughput: {num_gaussians / result2['mean_ms'] * 1000 / 1e6:.1f} M/s (subsequent writes)")

    # Strategy 3: Zero-copy from plyread (baseline - best case)
    logger.info("\n[Strategy 3] Zero-Copy from File (Baseline)")
    logger.info("  Method: Data loaded from file with _base already present")
    logger.info("  Best-case scenario for comparison")

    # Write a file first to read from
    temp_input = Path(tempfile.mktemp(suffix=".ply"))
    gsply.plywrite(str(temp_input), means, scales, quats, opacities, sh0, shN)

    def setup_from_file():
        return gsply.plyread(str(temp_input))

    def write_from_file(path, data):
        gsply.plywrite(str(path), data)

    result3 = benchmark_strategy("ZeroCopy (File)", write_from_file, setup_from_file, iterations)
    results.append(result3)

    temp_input.unlink()

    logger.info(f"  Setup:  {result3['setup_ms']:.2f} ms (file read)")
    logger.info(f"  Mean:   {result3['mean_ms']:.2f} ms")
    logger.info(f"  Std:    {result3['std_ms']:.2f} ms")
    logger.info(f"  Throughput: {num_gaussians / result3['mean_ms'] * 1000 / 1e6:.1f} M/s")

    # Summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n{'Strategy':<30} {'Setup':>10} {'Write':>10} {'Total':>10} {'Speedup':>10}")
    logger.info("-" * 80)

    for r in results:
        speedup = result1['mean_ms'] / r['mean_ms']
        logger.info(
            f"{r['label']:<30} {r['setup_ms']:>9.2f}ms {r['mean_ms']:>9.2f}ms "
            f"{r['total_ms']:>9.2f}ms {speedup:>9.2f}x"
        )

    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS")
    logger.info("=" * 80)

    # Break-even point for Strategy 2
    setup_overhead = result2['setup_ms']
    write_savings = result1['mean_ms'] - result2['mean_ms']

    if write_savings > 0:
        breakeven = setup_overhead / write_savings
        logger.info(f"\n[Break-Even Analysis for Strategy 2]")
        logger.info(f"  Setup cost: {setup_overhead:.2f} ms")
        logger.info(f"  Savings per write: {write_savings:.2f} ms")
        logger.info(f"  Break-even: {breakeven:.1f} writes")
        logger.info(f"  Recommendation: Use Strategy 2 if writing {int(np.ceil(breakeven))}+ times")
    else:
        logger.info(f"\n[Strategy 2 Not Recommended]")
        logger.info(f"  Setup cost: {setup_overhead:.2f} ms")
        logger.info(f"  No write time savings (Strategy 1 is faster)")
        logger.info(f"  Recommendation: Always use Strategy 1 (direct write)")

    # Single write comparison
    logger.info(f"\n[Single Write (Most Common Use Case)]")
    logger.info(f"  Strategy 1 (Direct):            {result1['mean_ms']:.2f} ms")
    logger.info(f"  Strategy 2 (Construct + Zero):  {result2['total_ms']:.2f} ms")
    logger.info(f"  Winner: Strategy {1 if result1['mean_ms'] < result2['total_ms'] else 2}")

    # Multiple writes comparison
    num_writes = 10
    total1 = result1['mean_ms'] * num_writes
    total2 = result2['setup_ms'] + result2['mean_ms'] * num_writes

    logger.info(f"\n[{num_writes} Writes (Batch Processing)]")
    logger.info(f"  Strategy 1 (Direct):            {total1:.2f} ms total")
    logger.info(f"  Strategy 2 (Construct + Zero):  {total2:.2f} ms total")
    logger.info(f"  Winner: Strategy {1 if total1 < total2 else 2}")
    logger.info(f"  Savings: {abs(total1 - total2):.2f} ms ({abs(total1 - total2) / max(total1, total2) * 100:.1f}%)")

    logger.info("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark _base construction strategies'
    )
    parser.add_argument(
        '--gaussians',
        type=int,
        default=400_000,
        help='Number of Gaussians (default: 400,000)'
    )
    parser.add_argument(
        '--sh-degree',
        type=int,
        default=0,
        choices=[0, 3],
        help='SH degree (default: 0)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=20,
        help='Number of iterations per test (default: 20)'
    )

    args = parser.parse_args()
    run_benchmark(args.gaussians, args.sh_degree, args.iterations)


if __name__ == '__main__':
    main()
