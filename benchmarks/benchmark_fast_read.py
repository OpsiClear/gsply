"""Benchmark plyread fast=True vs fast=False performance."""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gsply

def benchmark_read_methods(file_path, iterations=10):
    """Benchmark plyread fast=True vs fast=False."""

    print("=" * 80)
    print("PLY READ PERFORMANCE BENCHMARK (fast=True vs fast=False)")
    print("=" * 80)
    print()

    # Get file info
    file_size = Path(file_path).stat().st_size / (1024 * 1024)
    is_compressed, sh_degree = gsply.detect_format(file_path)

    # Quick read to get Gaussian count
    data = gsply.plyread(file_path)
    num_gaussians = data[0].shape[0]

    print("Test configuration:")
    print(f"  File: {Path(file_path).name}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Gaussians: {num_gaussians:,}")
    print(f"  SH degree: {sh_degree}")
    print(f"  Compressed: {is_compressed}")
    print(f"  Iterations: {iterations}")
    print()

    # Benchmark plyread(fast=False)
    print("Benchmarking plyread(fast=False)...")
    times_standard = []
    for i in range(iterations):
        start = time.perf_counter()
        result = gsply.plyread(file_path, fast=False)
        end = time.perf_counter()
        times_standard.append((end - start) * 1000)

    mean_standard = np.mean(times_standard)
    std_standard = np.std(times_standard)

    # Benchmark plyread(fast=True)
    print("Benchmarking plyread(fast=True)...")
    times_fast = []
    for i in range(iterations):
        start = time.perf_counter()
        result = gsply.plyread(file_path, fast=True)
        end = time.perf_counter()
        times_fast.append((end - start) * 1000)

    mean_fast = np.mean(times_fast)
    std_fast = np.std(times_fast)

    # Calculate improvement
    speedup = mean_standard / mean_fast
    time_saved = mean_standard - mean_fast
    percent_faster = ((mean_standard - mean_fast) / mean_standard) * 100

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"{'Method':<20} {'Time (ms)':<20} {'Speedup':<15} {'Status'}")
    print("-" * 80)
    print(f"{'fast=False':<20} {mean_standard:>8.2f} +/- {std_standard:<6.2f}  {'baseline':<15}")
    print(f"{'fast=True':<20} {mean_fast:>8.2f} +/- {std_fast:<6.2f}  {speedup:>6.2f}x faster")
    print()
    print(f"Performance improvement: {percent_faster:.1f}% faster")
    print(f"Time saved per read: {time_saved:.2f}ms")
    print()

    # Verify correctness
    print("Verifying correctness...")
    standard_data = gsply.plyread(file_path, fast=False)
    fast_data = gsply.plyread(file_path, fast=True)

    # Compare arrays
    means_match = np.allclose(standard_data.means, fast_data.means, rtol=1e-6, atol=1e-6)
    scales_match = np.allclose(standard_data.scales, fast_data.scales, rtol=1e-6, atol=1e-6)
    quats_match = np.allclose(standard_data.quats, fast_data.quats, rtol=1e-6, atol=1e-6)
    opacities_match = np.allclose(standard_data.opacities, fast_data.opacities, rtol=1e-6, atol=1e-6)
    sh0_match = np.allclose(standard_data.sh0, fast_data.sh0, rtol=1e-6, atol=1e-6)
    shN_match = np.allclose(standard_data.shN, fast_data.shN, rtol=1e-6, atol=1e-6)

    all_match = means_match and scales_match and quats_match and opacities_match and sh0_match and shN_match

    if all_match:
        print("[OK] All arrays match exactly")
    else:
        print("[FAIL] Data mismatch detected!")
        if not means_match: print("  - means mismatch")
        if not scales_match: print("  - scales mismatch")
        if not quats_match: print("  - quats mismatch")
        if not opacities_match: print("  - opacities mismatch")
        if not sh0_match: print("  - sh0 mismatch")
        if not shN_match: print("  - shN mismatch")

    print()
    print("Memory characteristics:")
    print("  fast=False copies: YES (safe, no shared memory)")
    print("  fast=True copies: NO (zero-copy views into base array)")
    print(f"  Base array kept alive: {fast_data.base is not None}")
    print()

if __name__ == "__main__":
    # Test with uncompressed SH3 file
    test_file = Path("../export_with_edits/frame_00000.ply")

    if not test_file.exists():
        print(f"[ERROR] Test file not found: {test_file}")
        print("Please run from the benchmarks directory or adjust the path.")
        sys.exit(1)

    benchmark_read_methods(test_file, iterations=20)
