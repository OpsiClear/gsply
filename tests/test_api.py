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
        assert gsply.__version__ == "0.1.0"

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

    def test_plyread_works(self):
        """Test that plyread works through main API."""
        test_file = Path("../export_with_edits/frame_00000.ply")
        if not test_file.exists():
            pytest.skip("Test file not found")

        result = gsply.plyread(test_file)
        assert isinstance(result, gsply.GSData)

        assert isinstance(result.means, np.ndarray)
        assert result.means.ndim == 2
        assert result.means.shape[1] == 3

    def test_detect_format_works(self):
        """Test that detect_format works through main API."""
        test_file = Path("../export_with_edits/frame_00000.ply")
        if not test_file.exists():
            pytest.skip("Test file not found")

        is_compressed, sh_degree = gsply.detect_format(test_file)
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

    def test_plyread_fast_parameter(self):
        """Test that plyread fast parameter works correctly."""
        test_file = Path("../export_with_edits/frame_00000.ply")
        if not test_file.exists():
            pytest.skip("Test file not found")

        # Test with fast=True (zero-copy)
        result = gsply.plyread(test_file, fast=True)
        assert isinstance(result, gsply.GSData)

        # Check all attributes exist
        assert hasattr(result, 'means')
        assert hasattr(result, 'scales')
        assert hasattr(result, 'quats')
        assert hasattr(result, 'opacities')
        assert hasattr(result, 'sh0')
        assert hasattr(result, 'shN')
        assert hasattr(result, 'base')

        # Check data types
        assert isinstance(result.means, np.ndarray)
        assert result.means.ndim == 2
        assert result.means.shape[1] == 3

        # Verify zero-copy behavior (base should exist for uncompressed fast reads)
        assert result.base is not None

    def test_fast_vs_standard_equivalence(self):
        """Test that plyread fast=True returns same data as fast=False."""
        test_file = Path("../export_with_edits/frame_00000.ply")
        if not test_file.exists():
            pytest.skip("Test file not found")

        # Read with both methods
        standard = gsply.plyread(test_file, fast=False)
        fast = gsply.plyread(test_file, fast=True)

        # Both should be GSData
        assert isinstance(standard, gsply.GSData)
        assert isinstance(fast, gsply.GSData)

        # Compare all arrays
        np.testing.assert_allclose(fast.means, standard.means, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fast.scales, standard.scales, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fast.quats, standard.quats, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fast.opacities, standard.opacities, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fast.sh0, standard.sh0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fast.shN, standard.shN, rtol=1e-6, atol=1e-6)

    def test_gsdata_can_unpack(self):
        """Test that GSData result can be unpacked like a tuple."""
        test_file = Path("../export_with_edits/frame_00000.ply")
        if not test_file.exists():
            pytest.skip("Test file not found")

        data = gsply.plyread(test_file)

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

    def test_complete_workflow(self, tmp_path):
        """Test complete read -> modify -> write workflow."""
        test_file = Path("../export_with_edits/frame_00000.ply")
        if not test_file.exists():
            pytest.skip("Test file not found")

        output_file = tmp_path / "workflow_output.ply"

        # 1. Detect format
        is_compressed, sh_degree = gsply.detect_format(test_file)
        assert is_compressed is False
        assert sh_degree == 3

        # 2. Read
        data = gsply.plyread(test_file)

        num_gaussians = data.means.shape[0]
        assert num_gaussians > 0

        # 3. Modify (translate all Gaussians)
        means_modified = data.means + np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # 4. Write
        gsply.plywrite(output_file, means_modified, data.scales, data.quats, data.opacities, data.sh0, data.shN)

        # 5. Read back and verify
        data_read = gsply.plyread(output_file)

        np.testing.assert_allclose(data_read.means, means_modified, rtol=1e-6, atol=1e-6)

    def test_conversion_workflow(self, tmp_path):
        """Test conversion from one SH degree to another."""
        test_file = Path("../export_with_edits/frame_00000.ply")
        if not test_file.exists():
            pytest.skip("Test file not found")

        output_file = tmp_path / "converted_sh0.ply"

        # Read SH degree 3
        data = gsply.plyread(test_file)

        # Write as SH degree 0 (drop higher-order SH)
        gsply.plywrite(output_file, data.means, data.scales, data.quats, data.opacities, data.sh0, shN=None)

        # Read back
        data_read = gsply.plyread(output_file)

        # Should have empty shN (degree 0)
        assert data_read.shN.shape[1] == 0

    def test_batch_processing(self, tmp_path):
        """Test processing multiple files."""
        test_file = Path("../export_with_edits/frame_00000.ply")
        if not test_file.exists():
            pytest.skip("Test file not found")

        # Read once
        data = gsply.plyread(test_file)

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
