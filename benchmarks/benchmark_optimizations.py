"""
Benchmark for measuring optimization improvements in gsply library.
Tests both uncompressed and compressed read/write operations.
"""

import os
import sys
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import gsply

def generate_test_data(num_gaussians: int, sh_degree: int = 3) -> Tuple:
    """Generate synthetic Gaussian data for testing."""
    np.random.seed(42)

    means = np.random.randn(num_gaussians, 3).astype(np.float32)
    scales = np.random.randn(num_gaussians, 3).astype(np.float32)
    quats = np.random.randn(num_gaussians, 4).astype(np.float32)
    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.rand(num_gaussians).astype(np.float32)
    sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

    # Generate SH coefficients based on degree
    sh_coeffs_per_degree = [0, 9, 24, 45]  # Cumulative counts
    num_coeffs = sh_coeffs_per_degree[sh_degree]
    if num_coeffs > 0:
        shN = np.random.randn(num_gaussians, num_coeffs // 3, 3).astype(np.float32)
    else:
        shN = None

    return means, scales, quats, opacities, sh0, shN

def benchmark_uncompressed_write(data: Tuple, num_runs: int = 10) -> float:
    """Benchmark uncompressed write performance."""
    means, scales, quats, opacities, sh0, shN = data

    times = []
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.ply"

        for _ in range(num_runs):
            start = time.perf_counter()
            gsply.plywrite(str(output_path), means, scales, quats, opacities, sh0, shN, compressed=False)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            output_path.unlink()  # Delete file for next iteration

    return np.median(times) * 1000  # Return median time in ms

def benchmark_compressed_write(data: Tuple, num_runs: int = 10) -> float:
    """Benchmark compressed write performance."""
    means, scales, quats, opacities, sh0, shN = data

    times = []
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.ply"

        for _ in range(num_runs):
            start = time.perf_counter()
            gsply.plywrite(str(output_path), means, scales, quats, opacities, sh0, shN, compressed=True)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            # Compressed write creates .compressed.ply
            Path(str(output_path).replace('.ply', '.compressed.ply')).unlink()

    return np.median(times) * 1000  # Return median time in ms

def benchmark_uncompressed_read(data: Tuple, num_runs: int = 10) -> float:
    """Benchmark uncompressed read performance."""
    means, scales, quats, opacities, sh0, shN = data

    times = []
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.ply"
        # Write once
        gsply.plywrite(str(test_path), means, scales, quats, opacities, sh0, shN, compressed=False)

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = gsply.plyread(str(test_path))
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return np.median(times) * 1000  # Return median time in ms

def benchmark_compressed_read(data: Tuple, num_runs: int = 10) -> float:
    """Benchmark compressed read performance."""
    means, scales, quats, opacities, sh0, shN = data

    times = []
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.ply"
        # Write once (compressed)
        gsply.plywrite(str(test_path), means, scales, quats, opacities, sh0, shN, compressed=True)
        compressed_path = str(test_path).replace('.ply', '.compressed.ply')

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = gsply.plyread(compressed_path)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return np.median(times) * 1000  # Return median time in ms

def run_comprehensive_benchmark():
    """Run comprehensive benchmarks for different sizes and formats."""
    print("=" * 80)
    print("GSPLY OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print()

    # Test different sizes
    test_sizes = [10_000, 50_000, 100_000, 400_000]
    sh_degrees = [0, 3]

    results = []

    for num_gaussians in test_sizes:
        for sh_degree in sh_degrees:
            print(f"Testing {num_gaussians:,} Gaussians, SH degree {sh_degree}:")
            print("-" * 60)

            # Generate test data
            data = generate_test_data(num_gaussians, sh_degree)

            # Run benchmarks
            uncompressed_write = benchmark_uncompressed_write(data, num_runs=5)
            uncompressed_read = benchmark_uncompressed_read(data, num_runs=5)
            compressed_write = benchmark_compressed_write(data, num_runs=5)
            compressed_read = benchmark_compressed_read(data, num_runs=5)

            print(f"  Uncompressed Write: {uncompressed_write:8.2f} ms")
            print(f"  Uncompressed Read:  {uncompressed_read:8.2f} ms")
            print(f"  Compressed Write:   {compressed_write:8.2f} ms")
            print(f"  Compressed Read:    {compressed_read:8.2f} ms")
            print()

            results.append({
                'gaussians': num_gaussians,
                'sh_degree': sh_degree,
                'uncompressed_write': uncompressed_write,
                'uncompressed_read': uncompressed_read,
                'compressed_write': compressed_write,
                'compressed_read': compressed_read
            })

    # Print summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'Gaussians':>10} | {'SH':>3} | {'UC Write':>10} | {'UC Read':>10} | {'C Write':>10} | {'C Read':>10}")
    print("-" * 75)

    for r in results:
        print(f"{r['gaussians']:>10,} | {r['sh_degree']:>3} | "
              f"{r['uncompressed_write']:>9.2f}ms | {r['uncompressed_read']:>9.2f}ms | "
              f"{r['compressed_write']:>9.2f}ms | {r['compressed_read']:>9.2f}ms")

    print()
    print("Legend: UC = Uncompressed, C = Compressed")
    print()

    # Calculate throughput for 400K Gaussians
    for r in results:
        if r['gaussians'] == 400_000:
            print(f"Throughput for 400K Gaussians, SH degree {r['sh_degree']}:")
            print(f"  Uncompressed Write: {400_000 / (r['uncompressed_write'] / 1000) / 1e6:.1f} M Gaussians/sec")
            print(f"  Uncompressed Read:  {400_000 / (r['uncompressed_read'] / 1000) / 1e6:.1f} M Gaussians/sec")
            print(f"  Compressed Write:   {400_000 / (r['compressed_write'] / 1000) / 1e6:.1f} M Gaussians/sec")
            print(f"  Compressed Read:    {400_000 / (r['compressed_read'] / 1000) / 1e6:.1f} M Gaussians/sec")
            print()

if __name__ == "__main__":
    run_comprehensive_benchmark()
