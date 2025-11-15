"""Test and benchmark zero-copy write optimization."""

import tempfile
import time
from pathlib import Path

import numpy as np

import gsply


def test_zero_copy_correctness():
    """Test that zero-copy write produces identical output."""
    print("=" * 80)
    print("TEST: Zero-Copy Write Correctness")
    print("=" * 80)

    # Test with real PLY file from disk
    test_file = Path("D:/4D/all_plys/frame_0.ply")
    if not test_file.exists():
        print(f"[SKIP] Test file not found: {test_file}")
        # Fallback to synthetic data
        print("Generating synthetic test data...")
        n = 10000
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.rand(n, 3).astype(np.float32)
        shN = np.random.rand(n, 45).astype(np.float32)

        # Write test file
        test_file = Path(tempfile.mktemp(suffix=".ply"))
        gsply.plywrite(str(test_file), means, scales, quats, opacities, sh0, shN)

    # Read data (creates _base array with views)
    print(f"\n[1/4] Reading test file: {test_file.name}")
    data = gsply.plyread(str(test_file))
    print(f"  Loaded {len(data):,} Gaussians")
    print(f"  Has _base: {data._base is not None}")
    print(f"  means.base is _base: {data.means.base is data._base}")

    # Write using zero-copy path
    output_zero_copy = Path(tempfile.mktemp(suffix="_zero_copy.ply"))
    print(f"\n[2/4] Writing with zero-copy optimization...")
    gsply.plywrite(str(output_zero_copy), *data.unpack())

    # Write using standard path (copy data to break _base)
    output_standard = Path(tempfile.mktemp(suffix="_standard.ply"))
    print(f"\n[3/4] Writing with standard path (copied data)...")
    means_copy = data.means.copy()
    scales_copy = data.scales.copy()
    quats_copy = data.quats.copy()
    opacities_copy = data.opacities.copy()
    sh0_copy = data.sh0.copy()
    shN_copy = data.shN.copy() if data.shN is not None else None
    gsply.plywrite(str(output_standard), means_copy, scales_copy, quats_copy,
                   opacities_copy, sh0_copy, shN_copy)

    # Verify outputs are identical
    print(f"\n[4/4] Verifying byte-for-byte equivalence...")
    with open(output_zero_copy, "rb") as f1, open(output_standard, "rb") as f2:
        bytes1 = f1.read()
        bytes2 = f2.read()

    if bytes1 == bytes2:
        print("  [OK] Files are byte-for-byte identical")
    else:
        print(f"  [FAIL] Files differ!")
        print(f"    Zero-copy: {len(bytes1):,} bytes")
        print(f"    Standard:  {len(bytes2):,} bytes")
        return False

    # Verify round-trip
    data_zero = gsply.plyread(str(output_zero_copy))
    data_std = gsply.plyread(str(output_standard))

    assert np.allclose(data_zero.means, data.means), "means mismatch"
    assert np.allclose(data_zero.scales, data.scales), "scales mismatch"
    assert np.allclose(data_zero.quats, data.quats), "quats mismatch"
    assert np.allclose(data_zero.opacities, data.opacities), "opacities mismatch"
    assert np.allclose(data_zero.sh0, data.sh0), "sh0 mismatch"
    if data.shN is not None:
        assert np.allclose(data_zero.shN, data.shN), "shN mismatch"

    print("  [OK] Round-trip verification passed")

    # Cleanup
    output_zero_copy.unlink()
    output_standard.unlink()

    print("\n" + "=" * 80)
    print("RESULT: Zero-Copy Write Correctness - PASSED")
    print("=" * 80)
    return True


def benchmark_zero_copy_speedup():
    """Benchmark zero-copy vs standard write performance."""
    print("\n" * 2)
    print("=" * 80)
    print("BENCHMARK: Zero-Copy Write Speedup")
    print("=" * 80)

    # Generate test files of different sizes
    configs = [
        (100_000, 3, "100K SH3"),
        (400_000, 3, "400K SH3"),
    ]

    results = []

    for num_gaussians, sh_degree, label in configs:
        print(f"\n{'-' * 80}")
        print(f"Test: {label} ({num_gaussians:,} Gaussians)")
        print(f"{'-' * 80}")

        # Generate data
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.random.rand(num_gaussians, 3).astype(np.float32) * 0.1
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        opacities = np.random.rand(num_gaussians).astype(np.float32)
        sh0 = np.random.rand(num_gaussians, 3).astype(np.float32)

        if sh_degree > 0:
            sh_coeffs = {1: 9, 2: 24, 3: 45}[sh_degree]
            shN = np.random.rand(num_gaussians, sh_coeffs).astype(np.float32) * 0.1
        else:
            shN = None

        # Create test file
        test_file = Path(tempfile.mktemp(suffix=".ply"))
        gsply.plywrite(str(test_file), means, scales, quats, opacities, sh0, shN)

        # Read back (creates _base with views)
        data = gsply.plyread(str(test_file))

        # Benchmark zero-copy path
        print("\n[1/2] Benchmarking ZERO-COPY path (from plyread)...")
        output_file = Path(tempfile.mktemp(suffix=".ply"))
        iterations = 20

        times_zero_copy = []
        for _ in range(iterations):
            start = time.perf_counter()
            gsply.plywrite(str(output_file), *data.unpack())
            elapsed = (time.perf_counter() - start) * 1000
            times_zero_copy.append(elapsed)
            output_file.unlink()

        mean_zero = np.mean(times_zero_copy)
        min_zero = np.min(times_zero_copy)

        print(f"  Mean: {mean_zero:.2f} ms")
        print(f"  Min:  {min_zero:.2f} ms")
        print(f"  Throughput: {num_gaussians / mean_zero * 1000 / 1e6:.1f} M Gaussians/sec")

        # Benchmark standard path (copy data first)
        print("\n[2/2] Benchmarking STANDARD path (copied data)...")
        means_copy = data.means.copy()
        scales_copy = data.scales.copy()
        quats_copy = data.quats.copy()
        opacities_copy = data.opacities.copy()
        sh0_copy = data.sh0.copy()
        shN_copy = data.shN.copy() if data.shN is not None else None

        times_standard = []
        for _ in range(iterations):
            start = time.perf_counter()
            gsply.plywrite(str(output_file), means_copy, scales_copy, quats_copy,
                           opacities_copy, sh0_copy, shN_copy)
            elapsed = (time.perf_counter() - start) * 1000
            times_standard.append(elapsed)
            output_file.unlink()

        mean_std = np.mean(times_standard)
        min_std = np.min(times_standard)

        print(f"  Mean: {mean_std:.2f} ms")
        print(f"  Min:  {min_std:.2f} ms")
        print(f"  Throughput: {num_gaussians / mean_std * 1000 / 1e6:.1f} M Gaussians/sec")

        # Calculate speedup
        speedup = mean_std / mean_zero

        print(f"\n[SPEEDUP] Zero-copy is {speedup:.1f}x faster ({mean_std:.1f}ms -> {mean_zero:.1f}ms)")

        results.append({
            'label': label,
            'num_gaussians': num_gaussians,
            'zero_copy_ms': mean_zero,
            'standard_ms': mean_std,
            'speedup': speedup,
        })

        # Cleanup
        test_file.unlink()

    # Summary
    print("\n" * 2)
    print("=" * 80)
    print("SUMMARY: Zero-Copy Write Speedup")
    print("=" * 80)
    print()
    print(f"{'Test':<12} | {'Zero-Copy':<12} | {'Standard':<12} | {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:<12} | {r['zero_copy_ms']:>8.1f} ms | {r['standard_ms']:>8.1f} ms | {r['speedup']:>6.1f}x")

    print()
    print("=" * 80)


if __name__ == "__main__":
    # Run correctness test
    if test_zero_copy_correctness():
        # Run benchmarks
        benchmark_zero_copy_speedup()
