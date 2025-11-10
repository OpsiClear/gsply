"""Tests for gsply.writer module."""

import pytest
import numpy as np
from pathlib import Path
from gsply.writer import plywrite, write_uncompressed, write_compressed
from gsply.reader import plyread


class TestWriteUncompressed:
    """Test write_uncompressed function."""

    def test_write_basic(self, tmp_path):
        """Test basic write operation."""
        output_file = tmp_path / "test.ply"

        # Create test data
        num_gaussians = 100
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.random.randn(num_gaussians, 3).astype(np.float32)
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)
        shN = np.random.randn(num_gaussians, 15, 3).astype(np.float32)

        # Write
        write_uncompressed(output_file, means, scales, quats, opacities, sh0, shN)

        # Check file exists and has content
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_write_sh0(self, tmp_path):
        """Test writing SH degree 0 (no shN)."""
        output_file = tmp_path / "sh0.ply"

        num_gaussians = 50
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.random.randn(num_gaussians, 3).astype(np.float32)
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

        # Write without shN
        write_uncompressed(output_file, means, scales, quats, opacities, sh0, shN=None)

        assert output_file.exists()

    def test_write_with_flattened_shn(self, tmp_path):
        """Test writing with flattened shN array."""
        output_file = tmp_path / "flattened.ply"

        num_gaussians = 100
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.random.randn(num_gaussians, 3).astype(np.float32)
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

        # Flattened shN (N, K*3)
        shN_flat = np.random.randn(num_gaussians, 45).astype(np.float32)

        write_uncompressed(output_file, means, scales, quats, opacities, sh0, shN_flat)

        assert output_file.exists()

    def test_write_invalid_shapes_raises_error(self, tmp_path):
        """Test that invalid shapes raise errors."""
        output_file = tmp_path / "invalid.ply"

        means = np.random.randn(100, 3).astype(np.float32)
        scales = np.random.randn(50, 3).astype(np.float32)  # Wrong count
        quats = np.random.randn(100, 4).astype(np.float32)
        opacities = np.random.randn(100).astype(np.float32)
        sh0 = np.random.randn(100, 3).astype(np.float32)

        with pytest.raises(AssertionError):
            write_uncompressed(output_file, means, scales, quats, opacities, sh0)


class TestWriteCompressed:
    """Test write_compressed function."""

    def test_write_compressed_basic(self, tmp_path):
        """Test basic compressed writing."""
        output_file = tmp_path / "compressed.ply"

        num_gaussians = 512  # 2 chunks
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.abs(np.random.randn(num_gaussians, 3).astype(np.float32))

        # Normalized quaternions
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

        write_compressed(output_file, means, scales, quats, opacities, sh0)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_write_compressed_with_sh(self, tmp_path):
        """Test compressed writing with higher-order SH."""
        output_file = tmp_path / "compressed_sh3.ply"

        num_gaussians = 256  # 1 chunk
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.abs(np.random.randn(num_gaussians, 3).astype(np.float32))

        # Normalized quaternions
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)
        shN = np.random.randn(num_gaussians, 15, 3).astype(np.float32)

        write_compressed(output_file, means, scales, quats, opacities, sh0, shN)

        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestPlywrite:
    """Test plywrite function (main API)."""

    def test_plywrite_uncompressed(self, tmp_path):
        """Test plywrite with compressed=False."""
        output_file = tmp_path / "test.ply"

        num_gaussians = 100
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.random.randn(num_gaussians, 3).astype(np.float32)
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)
        shN = np.random.randn(num_gaussians, 15, 3).astype(np.float32)

        plywrite(output_file, means, scales, quats, opacities, sh0, shN, compressed=False)

        assert output_file.exists()

    def test_plywrite_compressed(self, tmp_path):
        """Test plywrite with compressed=True."""
        output_file = tmp_path / "compressed.ply"

        num_gaussians = 256
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.abs(np.random.randn(num_gaussians, 3).astype(np.float32))

        # Normalized quaternions
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)
        shN = np.random.randn(num_gaussians, 15, 3).astype(np.float32)

        plywrite(output_file, means, scales, quats, opacities, sh0, shN, compressed=True)

        # Extension is automatically changed to .compressed.ply
        expected_file = tmp_path / "compressed.compressed.ply"
        assert expected_file.exists()
        assert expected_file.stat().st_size > 0

    def test_plywrite_accepts_string_path(self, tmp_path):
        """Test that plywrite accepts string paths."""
        output_file = str(tmp_path / "test.ply")

        num_gaussians = 50
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.random.randn(num_gaussians, 3).astype(np.float32)
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

        plywrite(output_file, means, scales, quats, opacities, sh0)

        assert Path(output_file).exists()


class TestRoundTrip:
    """Test read/write round-trip."""

    def test_roundtrip_sh3(self, tmp_path):
        """Test round-trip for SH degree 3."""
        output_file = tmp_path / "roundtrip.ply"

        # Create test data
        num_gaussians = 100
        means_orig = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales_orig = np.random.randn(num_gaussians, 3).astype(np.float32)
        quats_orig = np.random.randn(num_gaussians, 4).astype(np.float32)
        opacities_orig = np.random.randn(num_gaussians).astype(np.float32)
        sh0_orig = np.random.randn(num_gaussians, 3).astype(np.float32)
        shN_orig = np.random.randn(num_gaussians, 15, 3).astype(np.float32)

        # Write
        plywrite(output_file, means_orig, scales_orig, quats_orig, opacities_orig, sh0_orig, shN_orig)

        # Read back
        result = plyread(output_file)

        # Verify
        np.testing.assert_allclose(result.means, means_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.scales, scales_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.quats, quats_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.opacities, opacities_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.sh0, sh0_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.shN, shN_orig, rtol=1e-6, atol=1e-6)

    def test_roundtrip_sh0(self, tmp_path):
        """Test round-trip for SH degree 0."""
        output_file = tmp_path / "roundtrip_sh0.ply"

        num_gaussians = 50
        means_orig = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales_orig = np.random.randn(num_gaussians, 3).astype(np.float32)
        quats_orig = np.random.randn(num_gaussians, 4).astype(np.float32)
        opacities_orig = np.random.randn(num_gaussians).astype(np.float32)
        sh0_orig = np.random.randn(num_gaussians, 3).astype(np.float32)

        # Write without shN
        plywrite(output_file, means_orig, scales_orig, quats_orig, opacities_orig, sh0_orig)

        # Read back
        result = plyread(output_file)

        # Verify
        np.testing.assert_allclose(result.means, means_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.scales, scales_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.quats, quats_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.opacities, opacities_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.sh0, sh0_orig, rtol=1e-6, atol=1e-6)
        assert result.shN.shape == (num_gaussians, 0, 3)  # Empty for SH degree 0

    def test_compressed_file_size(self, tmp_path):
        """Test that compressed files are significantly smaller."""
        uncompressed_file = tmp_path / "uncompressed.ply"
        compressed_file = tmp_path / "compressed.ply"

        num_gaussians = 512
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.abs(np.random.randn(num_gaussians, 3).astype(np.float32))
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

        # Write both formats
        write_uncompressed(uncompressed_file, means, scales, quats, opacities, sh0)
        write_compressed(compressed_file, means, scales, quats, opacities, sh0)

        # Verify compressed is significantly smaller (at least 3x for SH0)
        uncompressed_size = uncompressed_file.stat().st_size
        compressed_size = compressed_file.stat().st_size
        compression_ratio = uncompressed_size / compressed_size

        assert compression_ratio > 3, f"Compression ratio {compression_ratio:.1f}x is too low"
        # Note: Compression ratio is higher with SH coefficients (~14x for SH3)

        # Verify compressed file can be read
        from gsply.reader import read_compressed
        result = read_compressed(compressed_file)
        assert result is not None, "Failed to read compressed file"

        # Basic sanity checks
        assert result.means.shape == (num_gaussians, 3)
        assert result.scales.shape == (num_gaussians, 3)
        assert result.quats.shape == (num_gaussians, 4)
        assert result.opacities.shape == (num_gaussians,)
        assert result.sh0.shape == (num_gaussians, 3)

    def test_compressed_with_sh_coefficients(self, tmp_path):
        """Test compressed writing with higher-order SH coefficients."""
        output_file = tmp_path / "compressed_sh3.ply"

        num_gaussians = 256
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.abs(np.random.randn(num_gaussians, 3).astype(np.float32))
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)
        shN = np.random.randn(num_gaussians, 15, 3).astype(np.float32)

        # Write compressed with SH
        write_compressed(output_file, means, scales, quats, opacities, sh0, shN)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify can be read back
        from gsply.reader import read_compressed
        result = read_compressed(output_file)
        assert result is not None

        assert result.shN.shape == (num_gaussians, 15, 3)

    def test_roundtrip_with_actual_file(self, tmp_path):
        """Test round-trip with real file."""
        test_file = Path("../export_with_edits/frame_00000.ply")
        if not test_file.exists():
            pytest.skip("Test file not found")

        output_file = tmp_path / "roundtrip_real.ply"

        # Read original
        data_orig = plyread(test_file)

        # Write
        plywrite(output_file, data_orig.means, data_orig.scales, data_orig.quats, data_orig.opacities, data_orig.sh0, data_orig.shN)

        # Read back
        result = plyread(output_file)

        # Verify exact match
        np.testing.assert_allclose(result.means, data_orig.means, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.scales, data_orig.scales, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.quats, data_orig.quats, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.opacities, data_orig.opacities, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.sh0, data_orig.sh0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.shN, data_orig.shN, rtol=1e-6, atol=1e-6)
