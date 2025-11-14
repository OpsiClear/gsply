"""Test for new compressed data APIs."""

import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from gsply import (
    GSData,
    compress_to_arrays,
    compress_to_bytes,
    plyread,
    plywrite,
)


def create_test_data(n_gaussians=100, sh_degree=0):
    """Create test Gaussian data."""
    np.random.seed(42)

    means = np.random.randn(n_gaussians, 3).astype(np.float32)
    scales = np.random.rand(n_gaussians, 3).astype(np.float32) * 0.1
    quats = np.random.randn(n_gaussians, 4).astype(np.float32)
    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.rand(n_gaussians).astype(np.float32)  # Shape (N,), not (N, 1)
    sh0 = np.random.rand(n_gaussians, 3).astype(np.float32)

    # Create higher degree SH if needed
    if sh_degree > 0:
        n_sh_coeffs = {1: 9, 2: 24, 3: 45}[sh_degree]
        shN = np.random.rand(n_gaussians, n_sh_coeffs, 3).astype(np.float32)  # noqa: N806
    else:
        shN = None  # noqa: N806

    return means, scales, quats, opacities, sh0, shN


class TestCompressToBytes:
    """Test compress_to_bytes function."""

    def test_basic_compression(self):
        """Test basic compression to bytes."""
        means, scales, quats, opacities, sh0, shN = create_test_data(100)  # noqa: N806

        # Compress to bytes
        compressed_bytes = compress_to_bytes(means, scales, quats, opacities, sh0, shN)

        assert isinstance(compressed_bytes, bytes)
        assert len(compressed_bytes) > 0

    def test_roundtrip_with_file(self):
        """Test that compressed bytes match file output."""
        means, scales, quats, opacities, sh0, shN = create_test_data(256)  # noqa: N806

        # Get compressed bytes
        compressed_bytes = compress_to_bytes(means, scales, quats, opacities, sh0, shN)

        # Write to file using normal API
        with tempfile.NamedTemporaryFile(suffix=".compressed.ply", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            plywrite(str(tmp_path), means, scales, quats, opacities, sh0, shN, compressed=True)

            # Read file bytes
            with open(tmp_path, "rb") as f:
                file_bytes = f.read()

            # Should be identical
            assert compressed_bytes == file_bytes

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_can_write_and_read_bytes(self):
        """Test that compressed bytes can be written and read back."""
        means, scales, quats, opacities, sh0, shN = create_test_data(512, sh_degree=1)  # noqa: N806

        # Compress to bytes
        compressed_bytes = compress_to_bytes(means, scales, quats, opacities, sh0, shN)

        # Write bytes to file
        with tempfile.NamedTemporaryFile(suffix=".compressed.ply", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(compressed_bytes)

        try:
            # Read back using plyread
            data = plyread(str(tmp_path))

            # Check that data is similar (compressed is lossy)
            assert data.means.shape == means.shape
            assert data.scales.shape == scales.shape
            assert data.quats.shape == quats.shape
            assert data.opacities.shape == opacities.shape
            assert data.sh0.shape == sh0.shape
            if shN is not None:
                assert data.shN.shape == shN.shape

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_various_sizes(self):
        """Test compression with various data sizes."""
        for n_gaussians in [1, 10, 100, 256, 257, 512, 1000]:
            means, scales, quats, opacities, sh0, shN = create_test_data(n_gaussians)  # noqa: N806

            compressed_bytes = compress_to_bytes(means, scales, quats, opacities, sh0, shN)

            assert isinstance(compressed_bytes, bytes)
            assert len(compressed_bytes) > 0

    def test_with_sh_degrees(self):
        """Test compression with different SH degrees."""
        for sh_degree in [0, 1, 2, 3]:
            means, scales, quats, opacities, sh0, shN = create_test_data(256, sh_degree)  # noqa: N806

            compressed_bytes = compress_to_bytes(means, scales, quats, opacities, sh0, shN)

            assert isinstance(compressed_bytes, bytes)
            assert len(compressed_bytes) > 0


class TestCompressToArrays:
    """Test compress_to_arrays function."""

    def test_basic_compression(self):
        """Test basic compression to arrays."""
        means, scales, quats, opacities, sh0, shN = create_test_data(100)  # noqa: N806

        # Compress to arrays
        header_bytes, chunk_bounds, packed_data, packed_sh = compress_to_arrays(
            means, scales, quats, opacities, sh0, shN
        )

        assert isinstance(header_bytes, bytes)
        assert isinstance(chunk_bounds, np.ndarray)
        assert isinstance(packed_data, np.ndarray)
        assert packed_sh is None  # SH degree 0

        assert chunk_bounds.dtype == np.float32
        assert packed_data.dtype == np.uint32

    def test_with_sh_data(self):
        """Test compression with SH data."""
        means, scales, quats, opacities, sh0, shN = create_test_data(256, sh_degree=2)  # noqa: N806

        # Compress to arrays
        header_bytes, chunk_bounds, packed_data, packed_sh = compress_to_arrays(
            means, scales, quats, opacities, sh0, shN
        )

        assert isinstance(header_bytes, bytes)
        assert isinstance(chunk_bounds, np.ndarray)
        assert isinstance(packed_data, np.ndarray)
        assert isinstance(packed_sh, np.ndarray)

        assert packed_sh.dtype == np.uint8

    def test_arrays_match_bytes(self):
        """Test that arrays can be assembled to match bytes output."""
        means, scales, quats, opacities, sh0, shN = create_test_data(  # noqa: N806
            512, sh_degree=1
        )

        # Get both outputs
        compressed_bytes = compress_to_bytes(means, scales, quats, opacities, sh0, shN)

        header_bytes, chunk_bounds, packed_data, packed_sh = compress_to_arrays(
            means, scales, quats, opacities, sh0, shN
        )

        # Assemble arrays into bytes
        buffer = BytesIO()
        buffer.write(header_bytes)
        buffer.write(chunk_bounds.tobytes())
        buffer.write(packed_data.tobytes())
        if packed_sh is not None:
            buffer.write(packed_sh.tobytes())

        assembled_bytes = buffer.getvalue()

        # Should be identical
        assert assembled_bytes == compressed_bytes

    def test_chunk_bounds_shape(self):
        """Test that chunk bounds have correct shape."""
        for n_gaussians in [100, 256, 257, 512, 1000]:
            means, scales, quats, opacities, sh0, shN = create_test_data(n_gaussians)  # noqa: N806

            header_bytes, chunk_bounds, packed_data, packed_sh = compress_to_arrays(
                means, scales, quats, opacities, sh0, shN
            )

            # Calculate expected number of chunks
            expected_chunks = (n_gaussians + 255) // 256

            assert chunk_bounds.shape == (expected_chunks, 18)  # 18 floats per chunk
            assert chunk_bounds.dtype == np.float32

    def test_packed_data_size(self):
        """Test that packed data has correct size."""
        n_gaussians = 512
        means, scales, quats, opacities, sh0, shN = create_test_data(n_gaussians)  # noqa: N806

        header_bytes, chunk_bounds, packed_data, packed_sh = compress_to_arrays(
            means, scales, quats, opacities, sh0, shN
        )

        # Each Gaussian takes 4 uint32 values in compressed format
        assert packed_data.shape == (n_gaussians, 4)
        assert packed_data.dtype == np.uint32


class TestIntegration:
    """Integration tests for compress APIs."""

    def test_gsdata_input(self):
        """Test that GSData can be used directly."""
        # Create test data and wrap in GSData
        means, scales, quats, opacities, sh0, shN = create_test_data(256)  # noqa: N806
        data = GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            masks=None,
            _base=None,
        )

        # Test new clean API: pass GSData directly
        compressed_bytes = compress_to_bytes(data)
        assert isinstance(compressed_bytes, bytes)

        header_bytes, chunk_bounds, packed_data, packed_sh = compress_to_arrays(data)
        assert isinstance(header_bytes, bytes)
        assert isinstance(chunk_bounds, np.ndarray)

        # Also test backward compatibility with extracted fields
        compressed_bytes2 = compress_to_bytes(
            data.means, data.scales, data.quats, data.opacities, data.sh0, data.shN
        )
        assert compressed_bytes == compressed_bytes2  # Should be identical

    def test_validation(self):
        """Test input validation."""
        means, scales, quats, opacities, sh0, shN = create_test_data(100)  # noqa: N806

        # Should work with validation enabled (default)
        compressed_bytes = compress_to_bytes(means, scales, quats, opacities, sh0, shN)
        assert isinstance(compressed_bytes, bytes)

        # Should work with validation disabled
        compressed_bytes = compress_to_bytes(
            means, scales, quats, opacities, sh0, shN, validate=False
        )
        assert isinstance(compressed_bytes, bytes)

    def test_network_transfer_workflow(self):
        """Test simulated network transfer workflow."""
        # Create data on "sender" side
        means, scales, quats, opacities, sh0, shN = create_test_data(512, sh_degree=2)  # noqa: N806

        # Compress to bytes for transfer
        compressed_bytes = compress_to_bytes(means, scales, quats, opacities, sh0, shN)

        # Simulate network transfer (just copy bytes)
        received_bytes = bytes(compressed_bytes)

        # On "receiver" side, save and read
        with tempfile.NamedTemporaryFile(suffix=".compressed.ply", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(received_bytes)

        try:
            # Read received data
            data = plyread(str(tmp_path))

            # Verify shape
            assert data.means.shape == means.shape
            assert data.scales.shape == scales.shape
            assert data.quats.shape == quats.shape
            assert data.opacities.shape == opacities.shape
            assert data.sh0.shape == sh0.shape
            assert data.shN.shape == shN.shape

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_custom_processing_workflow(self):
        """Test workflow for custom processing of compressed components."""
        means, scales, quats, opacities, sh0, shN = create_test_data(768, sh_degree=1)  # noqa: N806

        # Get compressed components
        header_bytes, chunk_bounds, packed_data, packed_sh = compress_to_arrays(
            means, scales, quats, opacities, sh0, shN
        )

        # Example: analyze chunk distribution
        num_chunks = chunk_bounds.shape[0]
        assert num_chunks == (768 + 255) // 256  # 3 chunks

        # Example: check compression ratio
        original_size = (
            means.nbytes + scales.nbytes + quats.nbytes + opacities.nbytes + sh0.nbytes + shN.nbytes
        )
        compressed_size = (
            len(header_bytes) + chunk_bounds.nbytes + packed_data.nbytes + packed_sh.nbytes
        )
        compression_ratio = original_size / compressed_size

        # Should have decent compression
        assert compression_ratio > 2.0

        # Example: modify header (e.g., for custom metadata)
        # This is just to show the components can be processed separately
        assert b"ply" in header_bytes
        assert b"format binary_little_endian" in header_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
