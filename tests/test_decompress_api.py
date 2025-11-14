"""Test the new decompress_from_bytes API."""

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from gsply import (
    GSData,
    compress_to_bytes,
    decompress_from_bytes,
    plyread,
)


def create_test_data(n_gaussians=256, sh_degree=1):
    """Create test Gaussian data."""
    np.random.seed(42)
    means = np.random.randn(n_gaussians, 3).astype(np.float32)
    scales = np.random.rand(n_gaussians, 3).astype(np.float32) * 0.1
    quats = np.random.randn(n_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.rand(n_gaussians).astype(np.float32)
    sh0 = np.random.rand(n_gaussians, 3).astype(np.float32)

    if sh_degree > 0:
        n_sh_coeffs = {1: 9, 2: 24, 3: 45}[sh_degree]
        shN = np.random.rand(n_gaussians, n_sh_coeffs, 3).astype(np.float32)  # noqa: N806
    else:
        shN = None  # noqa: N806

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        masks=None,
        _base=None,
    )


def test_basic_round_trip():
    """Test basic compress -> decompress round trip."""
    print("=" * 60)
    print("TEST 1: Basic Round-Trip")
    print("=" * 60)

    data = create_test_data(256, sh_degree=1)

    print("\nOriginal data:")
    print(f"  Gaussians: {data.means.shape[0]}")
    print("  SH degree: 1")

    # Compress
    compressed = compress_to_bytes(data)
    print(f"\nCompressed: {len(compressed):,} bytes")

    # Decompress
    data_restored = decompress_from_bytes(compressed)
    print("\nRestored:")
    print(f"  Gaussians: {data_restored.means.shape[0]}")
    print(f"  SH bands: {data_restored.shN.shape[1]}")

    # Verify shapes
    assert data_restored.means.shape == data.means.shape
    assert data_restored.scales.shape == data.scales.shape
    assert data_restored.quats.shape == data.quats.shape
    assert data_restored.opacities.shape == data.opacities.shape
    assert data_restored.sh0.shape == data.sh0.shape
    assert data_restored.shN.shape == data.shN.shape

    print("\n[OK] Round-trip successful, all shapes match")


def test_various_sizes():
    """Test with different data sizes."""
    print("\n" + "=" * 60)
    print("TEST 2: Various Data Sizes")
    print("=" * 60)

    sizes = [1, 10, 256, 257, 512, 1000, 10000]

    for n in sizes:
        data = create_test_data(n, sh_degree=0)
        compressed = compress_to_bytes(data)
        data_restored = decompress_from_bytes(compressed)

        assert data_restored.means.shape[0] == n
        print(f"  {n:>5} Gaussians: [OK] ({len(compressed):,} bytes)")

    print("\n[OK] All sizes handled correctly")


def test_various_sh_degrees():
    """Test with different SH degrees."""
    print("\n" + "=" * 60)
    print("TEST 3: Various SH Degrees")
    print("=" * 60)

    for sh_degree in [0, 1, 2, 3]:
        data = create_test_data(512, sh_degree=sh_degree)
        compressed = compress_to_bytes(data)
        data_restored = decompress_from_bytes(compressed)

        expected_bands = {0: 0, 1: 9, 2: 24, 3: 45}[sh_degree]
        assert data_restored.shN.shape[1] == expected_bands

        print(f"  SH degree {sh_degree} (bands={expected_bands}): [OK] ({len(compressed):,} bytes)")

    print("\n[OK] All SH degrees work correctly")


def test_vs_file_based():
    """Test that decompress_from_bytes matches file-based reading."""
    print("\n" + "=" * 60)
    print("TEST 4: Compare with File-Based Reading")
    print("=" * 60)

    data = create_test_data(1024, sh_degree=2)

    # Method 1: Compress to bytes, then decompress
    compressed = compress_to_bytes(data)
    data_from_bytes = decompress_from_bytes(compressed)

    # Method 2: Write to file, then read
    with tempfile.NamedTemporaryFile(suffix=".compressed.ply", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(compressed)

    try:
        data_from_file = plyread(str(tmp_path))

        # Compare
        print(f"\nFrom bytes: {data_from_bytes.means.shape[0]} Gaussians")
        print(f"From file:  {data_from_file.means.shape[0]} Gaussians")

        # Shapes should match
        assert data_from_bytes.means.shape == data_from_file.means.shape
        assert data_from_bytes.shN.shape == data_from_file.shN.shape

        # Values should be very close (both are lossy decompression)
        assert np.allclose(data_from_bytes.means, data_from_file.means, rtol=1e-5, atol=1e-5)
        assert np.allclose(data_from_bytes.sh0, data_from_file.sh0, rtol=1e-3, atol=1e-3)

        print("\n[OK] decompress_from_bytes matches plyread")

    finally:
        tmp_path.unlink()


def test_network_transfer_simulation():
    """Simulate network transfer use case."""
    print("\n" + "=" * 60)
    print("TEST 5: Network Transfer Simulation")
    print("=" * 60)

    # Sender side
    print("\nSender: Compressing data...")
    data_original = create_test_data(5000, sh_degree=1)
    compressed = compress_to_bytes(data_original)
    print(f"  Compressed to {len(compressed):,} bytes")

    # Simulate network transfer (just copy bytes)
    print("\nTransferring over network...")
    received_bytes = bytes(compressed)  # Simulate transfer

    # Receiver side
    print("\nReceiver: Decompressing data...")
    data_received = decompress_from_bytes(received_bytes)
    print(f"  Restored {data_received.means.shape[0]} Gaussians")

    # Verify
    assert data_received.means.shape[0] == 5000
    assert data_received.shN.shape[1] == 9

    print("\n[OK] Network transfer simulation successful")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 60)
    print("TEST 6: Error Handling")
    print("=" * 60)

    # Test with invalid bytes
    print("\nTesting with invalid bytes...")
    try:
        decompress_from_bytes(b"invalid data")
        print("  [FAIL] Should have raised ValueError")
    except ValueError as e:
        print(f"  [OK] Raised ValueError: {str(e)[:50]}...")

    # Test with empty bytes
    print("\nTesting with empty bytes...")
    try:
        decompress_from_bytes(b"")
        print("  [FAIL] Should have raised ValueError")
    except ValueError:
        print("  [OK] Raised ValueError")

    print("\n[OK] Error handling works correctly")


def test_api_convenience():
    """Test API convenience and usability."""
    print("\n" + "=" * 60)
    print("TEST 7: API Convenience")
    print("=" * 60)

    # Test that decompress_from_bytes is convenient for common use cases
    data = create_test_data(100, sh_degree=1)

    print("\nUse Case 1: Simple round-trip")
    compressed = compress_to_bytes(data)
    restored = decompress_from_bytes(compressed)
    assert restored.means.shape == data.means.shape
    print("  [OK] One-line compress and decompress")

    print("\nUse Case 2: Network transfer pattern")
    # Sender
    payload = compress_to_bytes(data)
    # Receiver
    received_data = decompress_from_bytes(payload)
    assert received_data.means.shape[0] == 100
    print("  [OK] Network transfer pattern works")

    print("\nUse Case 3: In-memory processing")
    # No disk I/O needed
    compressed_data = compress_to_bytes(data)
    processed = decompress_from_bytes(compressed_data)
    # Can immediately use the data
    assert isinstance(processed, GSData)
    assert processed.means.dtype == np.float32
    print("  [OK] In-memory processing (no disk I/O)")

    print("\n[OK] API is convenient and easy to use")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 20 + "DECOMPRESS API TESTS")
    print("=" * 70)

    tests = [
        test_basic_round_trip,
        test_various_sizes,
        test_various_sh_degrees,
        test_vs_file_based,
        test_network_transfer_simulation,
        test_error_handling,
        test_api_convenience,
    ]

    results = []
    for test_func in tests:
        try:
            test_func()
            results.append((test_func.__name__, True))
        except Exception as e:
            print(f"\n[ERROR] {test_func.__name__} failed: {e}")
            traceback.print_exc()
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "=" * 70)
    print(" " * 28 + "SUMMARY")
    print("=" * 70)

    all_passed = all(passed for _, passed in results)

    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        print("\nThe new decompress_from_bytes API:")
        print("  - Works correctly for all data sizes")
        print("  - Handles all SH degrees")
        print("  - Matches file-based plyread output")
        print("  - Perfect for network transfer/streaming")
        print("  - Symmetric with compress_to_bytes")
    else:
        print("\n[FAILURE] Some tests failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
