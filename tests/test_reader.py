"""Tests for gsply.reader module."""

import numpy as np
import pytest

from gsply.reader import plyread, read_compressed, read_uncompressed


class TestReadUncompressed:
    """Test read_uncompressed function."""

    def test_read_sh3_file(self, test_ply_file):
        """Test reading PLY file."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        result = read_uncompressed(test_ply_file)
        assert result is not None

        # Check shapes
        num_gaussians = result.means.shape[0]
        assert result.means.shape == (num_gaussians, 3)
        assert result.scales.shape == (num_gaussians, 3)
        assert result.quats.shape == (num_gaussians, 4)
        assert result.opacities.shape == (num_gaussians,)
        assert result.sh0.shape == (num_gaussians, 3)
        assert result.shN.shape[0] == num_gaussians
        assert result.shN.shape[2] == 3

        # Check data types
        assert result.means.dtype == np.float32
        assert result.scales.dtype == np.float32
        assert result.quats.dtype == np.float32
        assert result.opacities.dtype == np.float32
        assert result.sh0.dtype == np.float32
        assert result.shN.dtype == np.float32

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading non-existent file returns None."""
        nonexistent = tmp_path / "nonexistent.ply"
        result = read_uncompressed(nonexistent)
        assert result is None

    def test_read_invalid_format(self, tmp_path):
        """Test reading invalid format returns None."""
        invalid_file = tmp_path / "invalid.ply"
        invalid_file.write_text("not a ply file")

        result = read_uncompressed(invalid_file)
        assert result is None

    def test_read_sh0_file(self, tmp_path):
        """Test reading SH degree 0 file."""
        # Create minimal SH degree 0 PLY
        ply_header = """ply
format binary_little_endian 1.0
element vertex 100
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        test_file = tmp_path / "sh0.ply"
        test_file.write_text(ply_header)

        # Add binary data (100 vertices, 14 floats each)
        data = np.random.randn(100, 14).astype(np.float32)
        with open(test_file, "ab") as f:
            f.write(data.tobytes())

        result = read_uncompressed(test_file)
        assert result is not None

        assert result.means.shape == (100, 3)
        assert result.shN.shape == (100, 0, 3)  # No higher-order SH for degree 0

    def test_read_ascii_format_returns_none(self, tmp_path):
        """Test that ASCII format is not supported."""
        ascii_ply = """ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
end_header
0.0 0.0 0.0
"""
        test_file = tmp_path / "ascii.ply"
        test_file.write_text(ascii_ply)

        result = read_uncompressed(test_file)
        assert result is None


class TestReadCompressed:
    """Test read_compressed function."""

    def test_read_uncompressed_returns_none(self, test_ply_file):
        """Test that uncompressed files return None from compressed reader."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        result = read_compressed(test_ply_file)
        assert result is None  # Uncompressed file should return None

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading non-existent file returns None."""
        nonexistent = tmp_path / "nonexistent.ply"
        result = read_compressed(nonexistent)
        assert result is None


class TestPlyread:
    """Test plyread function (main API)."""

    def test_plyread_uncompressed(self, test_ply_file):
        """Test plyread with uncompressed file."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        result = plyread(test_ply_file)

        # Basic shape checks
        num_gaussians = result.means.shape[0]
        assert result.means.shape == (num_gaussians, 3)
        assert result.scales.shape == (num_gaussians, 3)
        assert result.quats.shape == (num_gaussians, 4)
        assert result.opacities.shape == (num_gaussians,)
        assert result.sh0.shape == (num_gaussians, 3)
        assert result.shN.shape[0] == num_gaussians
        assert result.shN.shape[2] == 3  # Last dimension is always 3

    def test_plyread_nonexistent_file(self, tmp_path):
        """Test plyread with non-existent file."""
        nonexistent = tmp_path / "nonexistent.ply"

        with pytest.raises(ValueError, match="Unsupported PLY format or invalid file"):
            plyread(nonexistent)

    def test_plyread_data_consistency(self, test_ply_file):
        """Test that plyread returns consistent data."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        # Read twice
        result1 = plyread(test_ply_file)
        result2 = plyread(test_ply_file)

        # Should be identical - compare fields explicitly (skip base as it may differ)
        np.testing.assert_array_equal(result1.means, result2.means)
        np.testing.assert_array_equal(result1.scales, result2.scales)
        np.testing.assert_array_equal(result1.quats, result2.quats)
        np.testing.assert_array_equal(result1.opacities, result2.opacities)
        np.testing.assert_array_equal(result1.sh0, result2.sh0)
        np.testing.assert_array_equal(result1.shN, result2.shN)

    def test_plyread_accepts_string_path(self, test_ply_file):
        """Test that plyread accepts string paths."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        result = plyread(str(test_ply_file))
        assert result.means.shape[0] > 0  # Should load some Gaussians

    def test_plyread_accepts_path_object(self, test_ply_file):
        """Test that plyread accepts Path objects."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        result = plyread(test_ply_file)
        assert result.means.shape[0] > 0


class TestDataIntegrity:
    """Test data integrity checks."""

    def test_no_nan_values(self, test_ply_file):
        """Test that loaded data contains no NaN values."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        result = plyread(test_ply_file)

        assert not np.any(np.isnan(result.means))
        assert not np.any(np.isnan(result.scales))
        assert not np.any(np.isnan(result.quats))
        assert not np.any(np.isnan(result.opacities))
        assert not np.any(np.isnan(result.sh0))
        if result.shN.size > 0:
            assert not np.any(np.isnan(result.shN))

    def test_no_inf_values(self, test_ply_file):
        """Test that loaded data contains no infinite values."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        result = plyread(test_ply_file)

        assert not np.any(np.isinf(result.means))
        assert not np.any(np.isinf(result.scales))
        assert not np.any(np.isinf(result.quats))
        assert not np.any(np.isinf(result.opacities))
        assert not np.any(np.isinf(result.sh0))
        if result.shN.size > 0:
            assert not np.any(np.isinf(result.shN))

    def test_finite_values(self, test_ply_file):
        """Test that all loaded data is finite."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        result = plyread(test_ply_file)

        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.scales))
        assert np.all(np.isfinite(result.quats))
        assert np.all(np.isfinite(result.opacities))
        assert np.all(np.isfinite(result.sh0))
        if result.shN.size > 0:
            assert np.all(np.isfinite(result.shN))
