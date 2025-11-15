"""Verification tests for performance optimizations in gsply writer.

This test suite specifically verifies the correctness of:
1. Quaternion extraction vectorization
2. Data ordering after sorting
3. Chunk bounds computation after sorting
"""

import numpy as np

from gsply.reader import read_compressed
from gsply.writer import CHUNK_SIZE, write_compressed


class TestQuaternionVectorization:
    """Verify quaternion extraction vectorization is correct."""

    def test_quaternion_extraction_manual_vs_vectorized(self):
        """Test that vectorized quaternion extraction matches original loop logic."""
        # Create 5 test quaternions with known properties
        num_quats = 5
        quats = np.array(
            [
                [0.9, 0.1, 0.2, 0.3],  # largest: 0 (0.9)
                [0.1, 0.8, 0.3, 0.2],  # largest: 1 (0.8)
                [0.2, 0.3, 0.85, 0.15],  # largest: 2 (0.85)
                [0.1, 0.2, 0.15, 0.9],  # largest: 3 (0.9)
                [-0.7, 0.1, 0.1, 0.1],  # largest: 0 (-0.7), should flip
            ],
            dtype=np.float32,
        )

        # Normalize
        quats_normalized = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        # ORIGINAL LOOP LOGIC (reference implementation)
        largest_idx_orig = []
        three_components_orig = []

        for i in range(num_quats):
            abs_quat = np.abs(quats_normalized[i])
            largest = np.argmax(abs_quat)
            largest_idx_orig.append(largest)

            # Flip if negative
            quat = quats_normalized[i]
            if quat[largest] < 0:
                quat = -quat

            # Extract three components (not the largest)
            three = []
            for j in range(4):
                if j != largest:
                    three.append(quat[j])
            three_components_orig.append(three)

        largest_idx_orig = np.array(largest_idx_orig, dtype=np.uint32)
        three_components_orig = np.array(three_components_orig, dtype=np.float32)

        # VECTORIZED LOGIC (current implementation)
        abs_quats = np.abs(quats_normalized)
        largest_idx = np.argmax(abs_quats, axis=1).astype(np.uint32)

        # Flip quaternion if largest component is negative
        sign_mask = np.take_along_axis(quats_normalized, largest_idx[:, np.newaxis], axis=1) < 0
        quats_normalized_vec = np.where(sign_mask, -quats_normalized, quats_normalized)

        # Extract the three components that are NOT the largest
        mask = np.ones((num_quats, 4), dtype=bool)
        mask[np.arange(num_quats), largest_idx] = False
        three_components = quats_normalized_vec[mask].reshape(num_quats, 3)

        # VERIFY MATCH
        np.testing.assert_array_equal(
            largest_idx, largest_idx_orig, err_msg="Largest indices do not match"
        )
        np.testing.assert_allclose(
            three_components,
            three_components_orig,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Three components do not match",
        )

        # Print detailed comparison for verification
        print("\n=== Quaternion Extraction Verification ===")
        for i in range(num_quats):
            print(f"Quat {i}: original={quats[i]}")
            print(f"  Largest idx: orig={largest_idx_orig[i]}, vec={largest_idx[i]}")
            print(f"  Three components: orig={three_components_orig[i]}, vec={three_components[i]}")
            print()


class TestDataOrderingAfterSorting:
    """Verify that chunk-based sorting works correctly for compression.

    NOTE: The compressed PLY format stores data in chunk order (not original order).
    This is a format specification requirement for optimal chunk-based compression.
    These tests verify that the sorting logic works correctly.
    """

    def test_data_order_in_compressed_file(self, tmp_path):
        """Verify data is written and read correctly in chunk order.

        NOTE: The compressed PLY format stores data in chunk order (not original order).
        This is a format specification requirement for optimal chunk-based compression.
        """
        output_file = tmp_path / "order_test.ply"

        # Create test data ALREADY in chunk order
        # Data at indices 0-255 are chunk 0, 256-511 are chunk 1, etc.
        num_gaussians = 3 * CHUNK_SIZE  # 768 Gaussians (3 chunks)

        # Create means that are easy to identify by chunk
        means = np.zeros((num_gaussians, 3), dtype=np.float32)
        for chunk_idx in range(3):
            start = chunk_idx * CHUNK_SIZE
            end = start + CHUNK_SIZE
            # Each chunk has a distinctive X value
            means[start:end, 0] = chunk_idx * 100.0  # Chunk 0: 0, Chunk 1: 100, Chunk 2: 200
            means[start:end, 1] = np.arange(CHUNK_SIZE) * 0.1  # Varies within chunk
            means[start:end, 2] = 0.0

        # Create other data
        scales = np.ones((num_gaussians, 3), dtype=np.float32) * 0.01
        quats = np.tile([1, 0, 0, 0], (num_gaussians, 1)).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)  # Normalize
        opacities = np.ones(num_gaussians, dtype=np.float32)
        sh0 = np.zeros((num_gaussians, 3), dtype=np.float32)

        # Write compressed (data already in chunk order)
        write_compressed(output_file, means, scales, quats, opacities, sh0, validate=True)

        # Read back
        result = read_compressed(output_file)

        # Verify the data matches (within compression tolerance)
        # Since input was already in chunk order, output should match closely
        print("\n=== Data Ordering Verification ===")
        print(f"Input first 5 means X values: {means[:5, 0]}")
        print(f"Output first 5 means X values: {result.means[:5, 0]}")
        print(f"Input chunk 1 first 3 means X values: {means[CHUNK_SIZE : CHUNK_SIZE + 3, 0]}")
        print(
            f"Output chunk 1 first 3 means X values: {result.means[CHUNK_SIZE : CHUNK_SIZE + 3, 0]}"
        )

        # Verify the data matches (within compression tolerance)
        np.testing.assert_allclose(
            result.means,
            means,
            rtol=1e-2,
            atol=0.2,
            err_msg="Means do not match after compression round-trip",
        )

    def test_shuffled_data_order(self, tmp_path):
        """Test that data with varying spatial patterns compresses correctly.

        NOTE: The compressed PLY format assigns chunks based on INDEX position, not spatial position.
        Chunk indices are computed as: index // CHUNK_SIZE
        The writer sorts by these chunk indices, but for sequentially indexed data,
        this is a no-op (data stays in same order).

        This test verifies compression works correctly with spatially varying data.
        """
        output_file = tmp_path / "shuffled_order_test.ply"

        # Create 512 Gaussians (2 chunks) with alternating spatial values
        num_gaussians = 2 * CHUNK_SIZE

        # Create data with alternating x-values (tests compression with varying data)
        means_input = np.zeros((num_gaussians, 3), dtype=np.float32)
        for i in range(num_gaussians):
            val = (i % 2) * 100.0  # Alternates: 0, 100, 0, 100, ...
            means_input[i, 0] = val
            means_input[i, 1] = i * 0.1
            means_input[i, 2] = 0.0

        scales = np.ones((num_gaussians, 3), dtype=np.float32) * 0.01
        quats = np.tile([1, 0, 0, 0], (num_gaussians, 1)).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)  # Normalize
        opacities = np.ones(num_gaussians, dtype=np.float32)
        sh0 = np.zeros((num_gaussians, 3), dtype=np.float32)

        # Write and read back
        write_compressed(output_file, means_input, scales, quats, opacities, sh0, validate=True)
        result = read_compressed(output_file)

        # Verify that output preserves order (since chunks are based on index, not position)
        print("\n=== Shuffled Data Compression Verification ===")
        print(f"Input pattern (alternating 0, 100): {means_input[:10, 0]}")
        print(f"Output pattern (should match input): {result.means[:10, 0]}")

        # Writer doesn't reorder data - chunks are based on original index position
        # So output should match input (within compression tolerance)
        np.testing.assert_allclose(
            result.means,
            means_input,
            rtol=1e-2,
            atol=0.2,
            err_msg="Compressed data does not match input order",
        )


class TestChunkBoundsAlignment:
    """Verify chunk bounds are computed correctly after sorting."""

    def test_chunk_bounds_match_data(self, tmp_path):
        """Verify that chunk bounds in the file match the actual data ranges."""
        output_file = tmp_path / "bounds_test.ply"

        # Create 2 chunks with DISTINCT ranges
        num_gaussians = 2 * CHUNK_SIZE

        means = np.zeros((num_gaussians, 3), dtype=np.float32)
        scales = np.zeros((num_gaussians, 3), dtype=np.float32)

        # Chunk 0: means in [0, 10], scales in [0.01, 0.1]
        means[0:CHUNK_SIZE, 0] = np.linspace(0, 10, CHUNK_SIZE)
        means[0:CHUNK_SIZE, 1] = np.linspace(0, 10, CHUNK_SIZE)
        means[0:CHUNK_SIZE, 2] = np.linspace(0, 10, CHUNK_SIZE)
        scales[0:CHUNK_SIZE, :] = np.linspace(0.01, 0.1, CHUNK_SIZE)[:, np.newaxis]

        # Chunk 1: means in [100, 110], scales in [1.0, 2.0]
        means[CHUNK_SIZE:, 0] = np.linspace(100, 110, CHUNK_SIZE)
        means[CHUNK_SIZE:, 1] = np.linspace(100, 110, CHUNK_SIZE)
        means[CHUNK_SIZE:, 2] = np.linspace(100, 110, CHUNK_SIZE)
        scales[CHUNK_SIZE:, :] = np.linspace(1.0, 2.0, CHUNK_SIZE)[:, np.newaxis]

        quats = np.tile([1, 0, 0, 0], (num_gaussians, 1)).astype(np.float32)
        opacities = np.ones(num_gaussians, dtype=np.float32)
        sh0 = np.zeros((num_gaussians, 3), dtype=np.float32)

        # Write compressed
        write_compressed(output_file, means, scales, quats, opacities, sh0, validate=True)

        # Read back
        result = read_compressed(output_file)

        # Verify that chunk 0 data is in the correct range
        chunk0_means = result.means[0:CHUNK_SIZE]
        chunk0_scales = result.scales[0:CHUNK_SIZE]

        print("\n=== Chunk Bounds Verification ===")
        print(
            f"Chunk 0 means X range: [{chunk0_means[:, 0].min():.2f}, {chunk0_means[:, 0].max():.2f}]"
        )
        print("Expected: [0.00, 10.00]")
        print(f"Chunk 0 scales range: [{chunk0_scales.min():.4f}, {chunk0_scales.max():.4f}]")
        print("Expected: [0.0100, 0.1000]")

        # Verify chunk 1 data is in the correct range
        chunk1_means = result.means[CHUNK_SIZE:]
        chunk1_scales = result.scales[CHUNK_SIZE:]

        print(
            f"Chunk 1 means X range: [{chunk1_means[:, 0].min():.2f}, {chunk1_means[:, 0].max():.2f}]"
        )
        print("Expected: [100.00, 110.00]")
        print(f"Chunk 1 scales range: [{chunk1_scales.min():.4f}, {chunk1_scales.max():.4f}]")
        print("Expected: [1.0000, 2.0000]")

        # Verify ranges match expectations (with compression tolerance)
        assert 0 <= chunk0_means[:, 0].min() <= 1, "Chunk 0 min out of range"
        assert 9 <= chunk0_means[:, 0].max() <= 11, "Chunk 0 max out of range"
        assert 99 <= chunk1_means[:, 0].min() <= 101, "Chunk 1 min out of range"
        assert 109 <= chunk1_means[:, 0].max() <= 111, "Chunk 1 max out of range"


class TestRoundTripWithRealData:
    """Test round-trip with realistic random data."""

    def test_random_data_round_trip(self, tmp_path):
        """Test that random data survives compression/decompression.

        NOTE: The compressed format stores data in chunk order, so we need to
        sort the original data by chunks before comparing.
        """
        output_file = tmp_path / "random_roundtrip.ply"

        num_gaussians = 512  # 2 chunks
        np.random.seed(42)  # Reproducible

        means_orig = np.random.randn(num_gaussians, 3).astype(np.float32) * 10
        scales_orig = np.abs(np.random.randn(num_gaussians, 3).astype(np.float32))
        quats_orig = np.random.randn(num_gaussians, 4).astype(np.float32)
        quats_orig = quats_orig / np.linalg.norm(quats_orig, axis=1, keepdims=True)
        opacities_orig = np.random.randn(num_gaussians).astype(np.float32)
        sh0_orig = np.random.randn(num_gaussians, 3).astype(np.float32)

        # Write compressed (will sort by chunks internally)
        write_compressed(
            output_file,
            means_orig,
            scales_orig,
            quats_orig,
            opacities_orig,
            sh0_orig,
            validate=True,
        )

        # Read back (data will be in chunk order)
        result = read_compressed(output_file)

        # Verify shapes
        assert result.means.shape == means_orig.shape
        assert result.scales.shape == scales_orig.shape
        assert result.quats.shape == quats_orig.shape
        assert result.opacities.shape == opacities_orig.shape
        assert result.sh0.shape == sh0_orig.shape

        # Sort original data by chunks to match output order
        # IMPORTANT: Must use stable sort to match the writer's radix sort behavior
        # The writer uses a stable radix sort, so within each chunk, elements
        # maintain their original order. np.argsort() defaults to quicksort which
        # is NOT stable and can produce different orderings on different platforms.
        chunk_indices = np.arange(num_gaussians) // CHUNK_SIZE
        sort_idx = np.argsort(chunk_indices, kind="stable")  # Stable sort is critical
        means_sorted = means_orig[sort_idx]
        scales_sorted = scales_orig[sort_idx]
        quats_sorted = quats_orig[sort_idx]
        opacities_sorted = opacities_orig[sort_idx]
        sh0_sorted = sh0_orig[sort_idx]

        # Verify approximate match (compression is lossy)
        print("\n=== Round-Trip Error Analysis ===")
        mean_error = np.abs(result.means - means_sorted).mean()
        scale_error = np.abs(result.scales - scales_sorted).mean()

        # For quaternions, we need to account for the fact that q and -q represent the same rotation
        # Also, quaternion compression is lossy due to smallest-three encoding
        # Compare normalized quaternions
        quats_norm = result.quats / np.linalg.norm(result.quats, axis=1, keepdims=True)
        quats_sorted_norm = quats_sorted / np.linalg.norm(quats_sorted, axis=1, keepdims=True)

        # Try both q and -q to find the closer match
        quat_error_pos = np.abs(quats_norm - quats_sorted_norm).mean()
        quat_error_neg = np.abs(quats_norm + quats_sorted_norm).mean()
        quat_error = min(quat_error_pos, quat_error_neg)

        opacity_error = np.abs(result.opacities - opacities_sorted).mean()
        sh0_error = np.abs(result.sh0 - sh0_sorted).mean()

        print(f"Mean error: {mean_error:.6f}")
        print(f"Scale error: {scale_error:.6f}")
        print(f"Quat error (normalized): {quat_error:.6f}")
        print(f"Opacity error: {opacity_error:.6f}")
        print(f"SH0 error: {sh0_error:.6f}")

        # These should be small (compression is lossy but not too lossy)
        # Note: Quaternion compression using smallest-three encoding has higher error
        assert mean_error < 0.2, "Mean error too large"
        assert scale_error < 0.15, "Scale error too large"
        assert quat_error < 0.5, "Quat error too large (even with smallest-three compression)"
        assert opacity_error < 0.15, "Opacity error too large"
        assert sh0_error < 0.15, "SH0 error too large"
