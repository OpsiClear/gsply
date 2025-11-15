"""Extended benchmark including 1M Gaussians and file size comparisons."""

import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import logging

import gsply

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def benchmark_with_filesize(num_gaussians: int, sh_degree: int, iterations: int = 10):
    """Benchmark with file size comparison."""
    logger.info(f"Testing {num_gaussians:,} Gaussians, SH degree {sh_degree}:")
    logger.info("-" * 70)

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

    # Test uncompressed
    unc_file = Path("benchmarks/test_data") / f"extended_unc_{num_gaussians}_{sh_degree}.ply"

    # Warmup
    gsply.plywrite(unc_file, means, scales, quats, opacities, sh0, shN, compressed=False)
    _ = gsply.plyread(unc_file)

    # Benchmark write
    write_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        gsply.plywrite(unc_file, means, scales, quats, opacities, sh0, shN, compressed=False)
        write_times.append((time.perf_counter() - start) * 1000)

    # Benchmark read
    read_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = gsply.plyread(unc_file)
        read_times.append((time.perf_counter() - start) * 1000)

    unc_size = unc_file.stat().st_size / (1024 * 1024)
    unc_write = np.median(write_times)
    unc_read = np.median(read_times)

    logger.info("  Uncompressed:")
    logger.info(f"    Write:      {unc_write:8.2f} ms ({num_gaussians / (unc_write/1000) / 1e6:5.1f}M/s)")
    logger.info(f"    Read:       {unc_read:8.2f} ms ({num_gaussians / (unc_read/1000) / 1e6:5.1f}M/s)")
    logger.info(f"    File size:  {unc_size:8.2f} MB")

    # Test compressed
    comp_base = Path("benchmarks/test_data") / f"extended_comp_{num_gaussians}_{sh_degree}.ply"

    # Warmup
    gsply.plywrite(comp_base, means, scales, quats, opacities, sh0, shN, compressed=True)
    comp_file = comp_base.with_suffix('.compressed.ply')
    _ = gsply.plyread(comp_file)

    # Benchmark write
    write_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        gsply.plywrite(comp_base, means, scales, quats, opacities, sh0, shN, compressed=True)
        write_times.append((time.perf_counter() - start) * 1000)

    # Benchmark read
    read_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = gsply.plyread(comp_file)
        read_times.append((time.perf_counter() - start) * 1000)

    comp_size = comp_file.stat().st_size / (1024 * 1024)
    comp_write = np.median(write_times)
    comp_read = np.median(read_times)

    logger.info("  Compressed:")
    logger.info(f"    Write:      {comp_write:8.2f} ms ({num_gaussians / (comp_write/1000) / 1e6:5.1f}M/s)")
    logger.info(f"    Read:       {comp_read:8.2f} ms ({num_gaussians / (comp_read/1000) / 1e6:5.1f}M/s)")
    logger.info(f"    File size:  {comp_size:8.2f} MB")
    logger.info(f"    Compression: {(1 - comp_size/unc_size)*100:.1f}% reduction")
    logger.info("")

    # Cleanup
    unc_file.unlink(missing_ok=True)
    comp_file.unlink(missing_ok=True)

    return {
        'gaussians': num_gaussians,
        'sh_degree': sh_degree,
        'unc_write': unc_write,
        'unc_read': unc_read,
        'unc_size': unc_size,
        'comp_write': comp_write,
        'comp_read': comp_read,
        'comp_size': comp_size
    }


def main():
    """Run extended benchmarks."""
    logger.info("=" * 80)
    logger.info("GSPLY EXTENDED BENCHMARK - INCLUDING 1M GAUSSIANS")
    logger.info("=" * 80)
    logger.info("")

    # Test configurations
    configs = [
        (10_000, 0),
        (10_000, 3),
        (50_000, 0),
        (50_000, 3),
        (100_000, 0),
        (100_000, 3),
        (400_000, 0),
        (400_000, 3),
        (1_000_000, 0),
        (1_000_000, 3),
    ]

    results = []
    for num_gaussians, sh_degree in configs:
        result = benchmark_with_filesize(num_gaussians, sh_degree, iterations=5)
        results.append(result)

    # Print summary
    logger.info("=" * 80)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"{'Gaussians':>10} | {'SH':>2} | {'UC Write':>8} | {'UC Read':>8} | {'C Write':>8} | {'C Read':>8} | {'UC Size':>8} | {'C Size':>7} | {'Comp%':>5}")
    logger.info("-" * 115)

    for r in results:
        comp_pct = (1 - r['comp_size']/r['unc_size'])*100
        logger.info(
            f"{r['gaussians']:>10,} | {r['sh_degree']:>2} | "
            f"{r['unc_write']:>7.1f}ms | {r['unc_read']:>7.1f}ms | "
            f"{r['comp_write']:>7.1f}ms | {r['comp_read']:>7.1f}ms | "
            f"{r['unc_size']:>7.1f}MB | {r['comp_size']:>6.1f}MB | {comp_pct:>4.0f}%"
        )

    logger.info("")
    logger.info("Legend: UC = Uncompressed, C = Compressed, Comp% = Compression ratio")
    logger.info("")

    # Highlight best performance
    logger.info("=" * 80)
    logger.info("KEY PERFORMANCE HIGHLIGHTS")
    logger.info("=" * 80)
    logger.info("")

    for r in results:
        if r['gaussians'] in [400_000, 1_000_000]:
            logger.info(f"{r['gaussians']:,} Gaussians, SH degree {r['sh_degree']}:")
            logger.info(f"  Best write throughput: {max(r['gaussians']/(r['unc_write']/1000), r['gaussians']/(r['comp_write']/1000))/1e6:.1f}M/s")
            logger.info(f"  Best read throughput:  {max(r['gaussians']/(r['unc_read']/1000), r['gaussians']/(r['comp_read']/1000))/1e6:.1f}M/s")
            logger.info(f"  Compression savings:   {(1 - r['comp_size']/r['unc_size'])*100:.0f}%")
            logger.info("")


if __name__ == "__main__":
    main()
