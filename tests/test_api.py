"""Tests for gsply main API (__init__.py)."""

import pytest
import numpy as np
from pathlib import Path
import gsply


class TestAPIExports:
    """Test that main API exports are available."""

    def test_plyread_exported(self):
        """Test that plyread is exported."""
        assert hasattr(gsply, 'plyread')
        assert callable(gsply.plyread)

    def test_plywrite_exported(self):
        """Test that plywrite is exported."""
        assert hasattr(gsply, 'plywrite')
        assert callable(gsply.plywrite)

    def test_detect_format_exported(self):
        """Test that detect_format is exported."""
        assert hasattr(gsply, 'detect_format')
        assert callable(gsply.detect_format)

    def test_version_exported(self):
        """Test that __version__ is exported."""
        assert hasattr(gsply, '__version__')
        assert isinstance(gsply.__version__, str)
        assert gsply.__version__ == "0.1.1"

    def test_all_contains_expected_exports(self):
        """Test that __all__ contains expected exports."""
        expected = ['plyread', 'GSData', 'plywrite', 'detect_format', '__version__']
        assert all(name in gsply.__all__ for name in expected)

    def test_gsdata_exported(self):
        """Test that GSData is exported."""
        assert hasattr(gsply, 'GSData')
        # It's a class/type, not a callable function
        assert isinstance(gsply.GSData, type)


class TestAPIFunctionality:
    """Test that API functions work correctly."""

    def test_plyread_works(self, test_ply_file):
        """Test that plyread works through main API."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        result = gsply.plyread(test_ply_file)
        assert isinstance(result, gsply.GSData)

        assert isinstance(result.means, np.ndarray)
        assert result.means.ndim == 2
        assert result.means.shape[1] == 3

    def test_detect_format_works(self, test_ply_file):
        """Test that detect_format works through main API."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        is_compressed, sh_degree = gsply.detect_format(test_ply_file)
        assert isinstance(is_compressed, bool)
        assert sh_degree in [0, 1, 2, 3, None]

    def test_plywrite_works(self, tmp_path):
        """Test that plywrite works through main API."""
        output_file = tmp_path / "api_test.ply"

        num_gaussians = 50
        means = np.random.randn(num_gaussians, 3).astype(np.float32)
        scales = np.random.randn(num_gaussians, 3).astype(np.float32)
        quats = np.random.randn(num_gaussians, 4).astype(np.float32)
        opacities = np.random.randn(num_gaussians).astype(np.float32)
        sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

        gsply.plywrite(output_file, means, scales, quats, opacities, sh0)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_gsdata_can_unpack(self, test_ply_file):
        """Test that GSData result can be unpacked like a tuple."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        data = gsply.plyread(test_ply_file)

        # Can unpack first 6 elements (excluding base)
        means, scales, quats, opacities, sh0, shN = data[:6]

        assert isinstance(means, np.ndarray)
        assert isinstance(scales, np.ndarray)
        assert isinstance(quats, np.ndarray)
        assert isinstance(opacities, np.ndarray)
        assert isinstance(sh0, np.ndarray)
        assert isinstance(shN, np.ndarray)


class TestEndToEnd:
    """Test end-to-end workflows using main API."""

    def test_complete_workflow(self, test_ply_file, tmp_path):
        """Test complete read -> modify -> write workflow."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        output_file = tmp_path / "workflow_output.ply"

        # 1. Detect format
        is_compressed, sh_degree = gsply.detect_format(test_ply_file)
        assert isinstance(is_compressed, bool)
        assert sh_degree in [0, 1, 2, 3, None]

        # 2. Read
        data = gsply.plyread(test_ply_file)

        num_gaussians = data.means.shape[0]
        assert num_gaussians > 0

        # 3. Modify (translate all Gaussians)
        means_modified = data.means + np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # 4. Write
        gsply.plywrite(output_file, means_modified, data.scales, data.quats, data.opacities, data.sh0, data.shN)

        # 5. Read back and verify
        data_read = gsply.plyread(output_file)

        np.testing.assert_allclose(data_read.means, means_modified, rtol=1e-6, atol=1e-6)

    def test_conversion_workflow(self, test_ply_file, tmp_path):
        """Test conversion from one SH degree to another."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        output_file = tmp_path / "converted_sh0.ply"

        # Read
        data = gsply.plyread(test_ply_file)

        # Write as SH degree 0 (drop higher-order SH)
        gsply.plywrite(output_file, data.means, data.scales, data.quats, data.opacities, data.sh0, shN=None)

        # Read back
        data_read = gsply.plyread(output_file)

        # Should have empty shN (degree 0)
        assert data_read.shN.shape[1] == 0

    def test_batch_processing(self, test_ply_file, tmp_path):
        """Test processing multiple files."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        # Read once
        data = gsply.plyread(test_ply_file)

        # Write 5 modified versions
        for i in range(5):
            output_file = tmp_path / f"batch_{i}.ply"

            # Apply different translation to each
            translation = np.array([i * 0.1, i * 0.2, i * 0.3], dtype=np.float32)
            means_modified = data.means + translation

            gsply.plywrite(output_file, means_modified, data.scales, data.quats, data.opacities, data.sh0, data.shN)

            # Verify
            data_read = gsply.plyread(output_file)
            np.testing.assert_allclose(data_read.means, means_modified, rtol=1e-6, atol=1e-6)


class TestCompressionAPIs:
    """Test the new compression APIs."""

    def test_compress_to_bytes_in_exports(self):
        """Test that compress_to_bytes is exported."""
        assert hasattr(gsply, 'compress_to_bytes')
        assert callable(gsply.compress_to_bytes)

    def test_compress_to_arrays_in_exports(self):
        """Test that compress_to_arrays is exported."""
        assert hasattr(gsply, 'compress_to_arrays')
        assert callable(gsply.compress_to_arrays)

    def test_compress_to_bytes_usage(self, tmp_path):
        """Test compress_to_bytes basic usage."""
        # Create test data
        n_gaussians = 256
        means = np.random.randn(n_gaussians, 3).astype(np.float32)
        scales = np.random.rand(n_gaussians, 3).astype(np.float32) * 0.1
        quats = np.random.randn(n_gaussians, 4).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        opacities = np.random.rand(n_gaussians).astype(np.float32)
        sh0 = np.random.rand(n_gaussians, 3).astype(np.float32)

        # Compress to bytes
        compressed_bytes = gsply.compress_to_bytes(
            means, scales, quats, opacities, sh0
        )

        assert isinstance(compressed_bytes, bytes)
        assert len(compressed_bytes) > 0

        # Should be able to save and read
        temp_file = tmp_path / "test.compressed.ply"
        with open(temp_file, 'wb') as f:
            f.write(compressed_bytes)

        data = gsply.plyread(str(temp_file))
        assert data.means.shape == means.shape

    def test_compress_to_arrays_usage(self):
        """Test compress_to_arrays basic usage."""
        # Create test data
        n_gaussians = 512
        means = np.random.randn(n_gaussians, 3).astype(np.float32)
        scales = np.random.rand(n_gaussians, 3).astype(np.float32) * 0.1
        quats = np.random.randn(n_gaussians, 4).astype(np.float32)
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        opacities = np.random.rand(n_gaussians).astype(np.float32)
        sh0 = np.random.rand(n_gaussians, 3).astype(np.float32)

        # Compress to arrays
        header, chunks, packed, sh = gsply.compress_to_arrays(
            means, scales, quats, opacities, sh0
        )

        assert isinstance(header, bytes)
        assert isinstance(chunks, np.ndarray)
        assert isinstance(packed, np.ndarray)
        assert chunks.dtype == np.float32  # Chunk bounds are float32
        assert packed.dtype == np.uint32

    def test_compress_apis_with_gsdata(self, sample_gaussian_data):
        """Test that compression APIs work with GSData."""
        means = sample_gaussian_data['means']
        scales = sample_gaussian_data['scales']
        quats = sample_gaussian_data['quats']
        opacities = sample_gaussian_data['opacities']
        sh0 = sample_gaussian_data['sh0']
        shN = sample_gaussian_data['shN']

        # Create GSData
        data = gsply.GSData(means, scales, quats, opacities, sh0, shN, base=None)

        # Test clean GSData API
        compressed_bytes = gsply.compress_to_bytes(data)
        assert isinstance(compressed_bytes, bytes)

        header, chunks, packed, sh = gsply.compress_to_arrays(data)
        assert isinstance(header, bytes)
        assert isinstance(chunks, np.ndarray)

        # Also verify backward compatibility works
        compressed_bytes2 = gsply.compress_to_bytes(
            data.means, data.scales, data.quats,
            data.opacities, data.sh0, data.shN
        )
        assert compressed_bytes == compressed_bytes2


class TestDocstrings:
    """Test that functions have proper docstrings."""

    def test_plyread_has_docstring(self):
        """Test that plyread has docstring."""
        assert gsply.plyread.__doc__ is not None
        assert len(gsply.plyread.__doc__) > 50

    def test_plywrite_has_docstring(self):
        """Test that plywrite has docstring."""
        assert gsply.plywrite.__doc__ is not None
        assert len(gsply.plywrite.__doc__) > 50

    def test_detect_format_has_docstring(self):
        """Test that detect_format has docstring."""
        assert gsply.detect_format.__doc__ is not None
        assert len(gsply.detect_format.__doc__) > 50

    def test_module_has_docstring(self):
        """Test that module has docstring."""
        assert gsply.__doc__ is not None
        assert 'gsply' in gsply.__doc__


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_gaussian(self, tmp_path):
        """Test reading and writing a single Gaussian."""
        output_file = tmp_path / "single_gaussian.ply"

        # Create single Gaussian
        means = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scales = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        opacities = np.array([1.0], dtype=np.float32)
        sh0 = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

        # Write
        gsply.plywrite(output_file, means, scales, quats, opacities, sh0)

        # Read back
        data = gsply.plyread(output_file)

        assert data.means.shape[0] == 1
        assert data.scales.shape[0] == 1
        assert data.quats.shape[0] == 1
        assert data.opacities.shape[0] == 1
        assert data.sh0.shape[0] == 1

    def test_gsdata_attribute_access(self, test_ply_file):
        """Test GSData attribute access methods."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        data = gsply.plyread(test_ply_file)

        # Test attribute access
        assert hasattr(data, 'means')
        assert hasattr(data, 'scales')
        assert hasattr(data, 'quats')
        assert hasattr(data, 'opacities')
        assert hasattr(data, 'sh0')
        assert hasattr(data, 'shN')
        assert hasattr(data, 'base')

        # Test indexing
        assert data[0] is data.means
        assert data[1] is data.scales
        assert data[2] is data.quats
        assert data[3] is data.opacities
        assert data[4] is data.sh0
        assert data[5] is data.shN

    def test_gsdata_slicing(self, test_ply_file):
        """Test GSData slicing behavior."""
        if test_ply_file is None:
            pytest.skip("Test file not found")

        data = gsply.plyread(test_ply_file)

        # Test tuple unpacking via slicing
        first_six = data[:6]
        assert len(first_six) == 6
        assert first_six[0] is data.means
        assert first_six[5] is data.shN

        # Test full unpacking
        means, scales, quats, opacities, sh0, shN = data[:6]
        assert means is data.means
        assert shN is data.shN
