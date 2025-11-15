"""Comprehensive write benchmark comparing zero-copy vs standard paths.

Tests both:
1. Real PLY files (with _base for zero-copy)
2. Synthetic data (without _base, standard path)

Shows speedup for zero-copy and baseline for standard path.
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


def generate_synthetic_data(num_gaussians: int, sh_degree: int = 0) -> GSData:
    """Generate synthetic GSData without _base (tests standard path)."""
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

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        _base=None  # No _base = standard path
    )


def benchmark_write(data: GSData, label: str, iterations: int = 20) -> dict:
    """Benchmark write performance for given data.

    Returns dict with timing statistics.
    """
    output_file = Path(tempfile.mktemp(suffix=".ply"))
    times = []

    for i in range(iterations):
        start = time.perf_counter()
        gsply.plywrite(str(output_file), data)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        output_file.unlink()

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)

    num_gaussians = len(data)
    throughput = num_gaussians / mean_time * 1000 / 1e6

    return {
        'label': label,
        'num_gaussians': num_gaussians,
        'has_base': data._base is not None,
        'mean_ms': mean_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'throughput_m': throughput
    }


def run_comprehensive_benchmark(file_path: str | None = None, iterations: int = 20):
    """Run comprehensive write benchmarks."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE WRITE BENCHMARK")
    logger.info("=" * 80)
    logger.info("")

    results = []

    # Test 1: Real file (if provided) - Zero-copy path
    if file_path and Path(file_path).exists():
        logger.info("[Test 1] Real PLY File (Zero-Copy Path)")
        logger.info(f"  File: {Path(file_path).name}")

        data = gsply.plyread(file_path)
        sh_degree = data.shN.shape[1] // 3 if data.shN.size > 0 else 0

        logger.info(f"  Gaussians: {len(data):,}")
        logger.info(f"  SH degree: {sh_degree}")
        logger.info(f"  Has _base: {data._base is not None}")

        result = benchmark_write(data, f"Real SH{sh_degree}", iterations)
        results.append(result)

        logger.info(f"  Mean:  {result['mean_ms']:.2f} ms")
        logger.info(f"  Std:   {result['std_ms']:.2f} ms")
        logger.info(f"  Throughput: {result['throughput_m']:.1f} M Gaussians/sec")
        logger.info("")

    # Test 2: Synthetic data - Standard path
    logger.info("[Test 2] Synthetic Data (Standard Path)")

    test_configs = [
        (100_000, 0, "100K SH0"),
        (400_000, 0, "400K SH0"),
        (400_000, 3, "400K SH3"),
    ]

    for num_gaussians, sh_degree, label in test_configs:
        logger.info(f"  Config: {label}")
        data = generate_synthetic_data(num_gaussians, sh_degree)

        result = benchmark_write(data, label, iterations)
        results.append(result)

        logger.info(f"    Mean:  {result['mean_ms']:.2f} ms")
        logger.info(f"    Throughput: {result['throughput_m']:.1f} M Gaussians/sec")

    logger.info("")

    # Summary table
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Config':<15} {'Gaussians':>10} {'ZeroCopy':>10} {'Mean':>10} {'Throughput':>12}")
    logger.info("-" * 80)

    for r in results:
        zero_copy = "Yes" if r['has_base'] else "No"
        logger.info(
            f"{r['label']:<15} {r['num_gaussians']:>10,} {zero_copy:>10} "
            f"{r['mean_ms']:>9.2f}ms {r['throughput_m']:>10.1f} M/s"
        )

    logger.info("")

    # Calculate speedup if we have both real and synthetic data
    real_results = [r for r in results if r['has_base']]
    synthetic_similar = [r for r in results if not r['has_base'] and 'SH0' in r['label'] and r['num_gaussians'] > 300_000]

    if real_results and synthetic_similar:
        real = real_results[0]
        synth = synthetic_similar[0]

        # Normalize to same number of Gaussians for fair comparison
        real_normalized = real['mean_ms'] * (synth['num_gaussians'] / real['num_gaussians'])
        speedup = synth['mean_ms'] / real_normalized

        logger.info("Zero-Copy Optimization Impact:")
        logger.info(f"  Real file (zero-copy):     {real_normalized:.2f} ms (normalized to {synth['num_gaussians']:,})")
        logger.info(f"  Synthetic (standard):      {synth['mean_ms']:.2f} ms")
        logger.info(f"  Speedup from zero-copy:    {speedup:.2f}x")
        logger.info("")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive write benchmark'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Optional: Path to real PLY file for zero-copy testing'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=20,
        help='Number of iterations per test (default: 20)'
    )

    args = parser.parse_args()
    run_comprehensive_benchmark(args.file, args.iterations)


if __name__ == '__main__':
    main()
