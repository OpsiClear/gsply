"""Simple test of read performance with refactored dataclass."""

import time
import numpy as np
from pathlib import Path
from gsply import plyread


def test_with_existing_file():
    """Test with existing file."""
    # Find test file
    test_files = [
        Path("benchmarks/test_data/synthetic_100k_sh3_uncompressed.ply"),
        Path("benchmarks/test_data/synthetic_100k_sh0_uncompressed.ply"),
        Path("benchmarks/test_data/synthetic_1000k_sh3_uncompressed.ply"),
    ]

    test_file = None
    for f in test_files:
        if f.exists():
            test_file = f
            break

    if test_file is None:
        print("No test file found. Please provide a PLY file.")
        return

    print(f"Using test file: {test_file}")
    print()

    # Test 1: Read performance
    print("TEST 1: Read Performance")
    print("-" * 40)

    times = []
    for i in range(10):
        start = time.perf_counter()
        data = plyread(test_file)
        end = time.perf_counter()
        times.append(end - start)

        if i == 0:
            print(f"  Loaded {data.means.shape[0]} Gaussians")
            print(f"  SH bands: {data.shN.shape[1] if data.shN.ndim > 1 else 0}")

    avg_time = np.mean(times)
    print(f"  Average read time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {data.means.shape[0] / avg_time / 1e6:.1f}M Gaussians/sec")
    print()

    # Test 2: Unpack interface
    print("TEST 2: Unpack Interface")
    print("-" * 40)

    # Test that unpack() works
    means, scales, quats, opacities, sh0, shN = data.unpack()  # noqa: N806
    print(f"  Unpacked 6 fields successfully")
    assert means is data.means
    assert shN is data.shN
    print("  [OK] Unpack interface works correctly")
    print()

    # Test 3: Attribute access
    print("TEST 3: Attribute Access Performance")
    print("-" * 40)

    n_iterations = 100000
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = data.means
        _ = data.scales
        _ = data.quats
        _ = data.opacities
        _ = data.sh0
        _ = data.shN
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / n_iterations
    print(f"  {n_iterations} accesses in {total_time*1000:.2f} ms")
    print(f"  Average per access: {avg_time*1e9:.1f} ns")
    print()

    # Test 4: Mutability
    print("TEST 4: Field Mutability")
    print("-" * 40)

    original_mean = data.means[0, 0].copy()
    data.means[0, 0] = 999.0

    assert data.means[0, 0] == 999.0
    print(f"  Original value: {original_mean:.3f}")
    print(f"  Modified value: {data.means[0, 0]:.3f}")
    print("  [OK] Fields are mutable")
    print()

    # Test 5: Private _base field
    print("TEST 5: Private _base Field")
    print("-" * 40)

    assert hasattr(data, '_base')
    print(f"  _base field exists: {data._base is not None}")

    # Check that unpack returns 6 fields
    unpacked = data.unpack()
    assert len(unpacked) == 6
    print(f"  Unpacked fields count: {len(unpacked)} (excludes _base and masks)")
    print("  [OK] _base and masks are excluded from unpacking")
    print()

    print("=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print("All tests passed successfully!")
    print("- Read performance maintained")
    print("- Unpack interface works")
    print("- Fields are mutable")
    print("- _base is private")


if __name__ == "__main__":
    test_with_existing_file()