"""Test real-world read performance with the new dataclass implementation."""

import time
from pathlib import Path
import tempfile
import numpy as np
from gsply import plyread, plywrite


def create_test_file(n_gaussians=100000, sh_degree=3):
    """Create a test PLY file."""
    # Create test data
    np.random.seed(42)
    means = np.random.randn(n_gaussians, 3).astype(np.float32)
    scales = np.random.rand(n_gaussians, 3).astype(np.float32) * 0.1
    quats = np.random.randn(n_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.rand(n_gaussians).astype(np.float32)
    sh0 = np.random.rand(n_gaussians, 3).astype(np.float32)

    if sh_degree > 0:
        n_sh_coeffs = {1: 9, 2: 24, 3: 45}[sh_degree]
        # Note: plywrite expects shN in shape (N, K, 3) or (N, K*3)
        shN = np.random.rand(n_gaussians, n_sh_coeffs, 3).astype(np.float32)  # noqa: N806
    else:
        shN = None  # noqa: N806

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    # Write the file (shN can be None or shape (N, K, 3))
    plywrite(temp_path, means, scales, quats, opacities, sh0, shN, compressed=False)

    return temp_path


def benchmark_read_performance(file_path, n_iterations=100):
    """Benchmark read performance."""
    times = []

    for i in range(n_iterations):
        start = time.perf_counter()
        data = plyread(file_path)
        end = time.perf_counter()
        times.append(end - start)

        # Verify data loaded correctly
        if i == 0:
            print(f"  Loaded {data.means.shape[0]} Gaussians")
            print(f"  SH bands: {data.shN.shape[1]}")

    return times


def test_unpack_performance(file_path, n_iterations=1000):
    """Test unpack() performance."""
    # Read once
    data = plyread(file_path)

    # Benchmark unpacking
    start = time.perf_counter()
    for _ in range(n_iterations):
        means, scales, quats, opacities, sh0, shN = data.unpack()  # noqa: N806
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / n_iterations

    return total_time, avg_time


def test_attribute_access_performance(file_path, n_iterations=10000):
    """Test attribute access performance."""
    # Read once
    data = plyread(file_path)

    # Benchmark attribute access
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

    return total_time, avg_time


def test_mutability(file_path):
    """Test that fields are mutable."""
    data = plyread(file_path)

    # Test mutability
    original_mean = data.means[0, 0].copy()
    data.means[0, 0] = 999.0

    assert data.means[0, 0] == 999.0, "Field mutation failed"
    print(f"  Original value: {original_mean:.3f}")
    print(f"  Modified value: {data.means[0, 0]:.3f}")
    print("  [OK] Fields are mutable")


def main():
    """Run performance benchmarks."""
    print("=" * 80)
    print("Real-World Performance Test with Refactored GSData (Dataclass)")
    print("=" * 80)
    print()

    # Create test files
    print("Creating test files...")
    file_100k = create_test_file(100000, sh_degree=3)
    file_10k = create_test_file(10000, sh_degree=0)
    print()

    # Test 1: Read performance
    print("-" * 80)
    print("TEST 1: Read Performance (100K Gaussians, SH3)")
    print("-" * 80)
    times_100k = benchmark_read_performance(file_100k, n_iterations=20)
    avg_time = np.mean(times_100k)
    std_time = np.std(times_100k)

    print(f"  Average read time: {avg_time*1000:.2f} ms")
    print(f"  Std deviation: {std_time*1000:.2f} ms")
    print(f"  Min/Max: {min(times_100k)*1000:.2f} / {max(times_100k)*1000:.2f} ms")
    print(f"  Throughput: {100000 / avg_time / 1e6:.1f}M Gaussians/sec")
    print()

    # Test 2: Unpack performance
    print("-" * 80)
    print("TEST 2: Unpack Performance")
    print("-" * 80)
    total_time, avg_time = test_unpack_performance(file_100k, n_iterations=10000)
    print(f"  Total time (10K unpacks): {total_time*1000:.2f} ms")
    print(f"  Average per unpack: {avg_time*1e9:.1f} ns")
    print()

    # Test 3: Attribute access performance
    print("-" * 80)
    print("TEST 3: Attribute Access Performance")
    print("-" * 80)
    total_time, avg_time = test_attribute_access_performance(file_100k, n_iterations=100000)
    print(f"  Total time (100K accesses): {total_time*1000:.2f} ms")
    print(f"  Average per access: {avg_time*1e9:.1f} ns")
    print()

    # Test 4: Mutability
    print("-" * 80)
    print("TEST 4: Field Mutability")
    print("-" * 80)
    test_mutability(file_100k)
    print()

    # Test 5: Small file performance
    print("-" * 80)
    print("TEST 5: Small File Performance (10K Gaussians, SH0)")
    print("-" * 80)
    times_10k = benchmark_read_performance(file_10k, n_iterations=100)
    avg_time = np.mean(times_10k)

    print(f"  Average read time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {10000 / avg_time / 1e6:.1f}M Gaussians/sec")
    print()

    # Cleanup
    file_100k.unlink()
    file_10k.unlink()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("The refactored dataclass implementation:")
    print("  - Maintains excellent read performance")
    print("  - Supports unpack() interface for compatibility")
    print("  - Fields are now mutable (can modify after loading)")
    print("  - Private _base field is not exposed in unpacking")
    print()


if __name__ == "__main__":
    main()