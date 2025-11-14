"""Tests for gsply.formats module."""

from pathlib import Path

import numpy as np

from gsply.formats import (
    CHUNK_SIZE,
    EXPECTED_PROPERTIES_BY_SH_DEGREE,
    PROPERTY_COUNTS_BY_SH_DEGREE,
    SH_C0,
    detect_format,
    get_sh_degree_from_property_count,
)


class TestConstants:
    """Test module constants."""

    def test_property_counts_mapping(self):
        """Test SH degree to property count mapping."""
        assert PROPERTY_COUNTS_BY_SH_DEGREE[0] == 14
        assert PROPERTY_COUNTS_BY_SH_DEGREE[1] == 23
        assert PROPERTY_COUNTS_BY_SH_DEGREE[2] == 38
        assert PROPERTY_COUNTS_BY_SH_DEGREE[3] == 59

    def test_expected_properties_lengths(self):
        """Test that expected property lists have correct lengths."""
        for sh_degree, prop_count in PROPERTY_COUNTS_BY_SH_DEGREE.items():
            assert len(EXPECTED_PROPERTIES_BY_SH_DEGREE[sh_degree]) == prop_count

    def test_chunk_size(self):
        """Test chunk size constant."""
        assert CHUNK_SIZE == 256

    def test_sh_c0_constant(self):
        """Test SH C0 constant value."""
        expected_sh_c0 = 0.28209479177387814
        assert abs(SH_C0 - expected_sh_c0) < 1e-10


class TestGetSHDegreeFromPropertyCount:
    """Test get_sh_degree_from_property_count function."""

    def test_valid_property_counts(self):
        """Test valid property counts return correct SH degree."""
        assert get_sh_degree_from_property_count(14) == 0
        assert get_sh_degree_from_property_count(23) == 1
        assert get_sh_degree_from_property_count(38) == 2
        assert get_sh_degree_from_property_count(59) == 3

    def test_invalid_property_counts(self):
        """Test invalid property counts return None."""
        assert get_sh_degree_from_property_count(10) is None
        assert get_sh_degree_from_property_count(15) is None
        assert get_sh_degree_from_property_count(100) is None
        assert get_sh_degree_from_property_count(0) is None


class TestDetectFormat:
    """Test detect_format function."""

    def test_detect_uncompressed_sh3(self, tmp_path):
        """Test detection of uncompressed SH degree 3 format."""
        # Use existing test file
        test_file = Path("../export_with_edits/frame_00000.ply")
        if test_file.exists():
            is_compressed, sh_degree = detect_format(test_file)
            assert is_compressed is False
            assert sh_degree == 3

    def test_detect_nonexistent_file(self, tmp_path):
        """Test detection of non-existent file."""
        nonexistent = tmp_path / "nonexistent.ply"
        is_compressed, sh_degree = detect_format(nonexistent)
        assert is_compressed is False
        assert sh_degree is None

    def test_detect_invalid_ply(self, tmp_path):
        """Test detection of invalid PLY file."""
        invalid_file = tmp_path / "invalid.ply"
        invalid_file.write_text("not a valid ply file")

        is_compressed, sh_degree = detect_format(invalid_file)
        assert is_compressed is False
        assert sh_degree is None

    def test_detect_uncompressed_sh0(self, tmp_path):
        """Test detection of SH degree 0 format."""
        # Create minimal SH degree 0 PLY file
        ply_content = """ply
format binary_little_endian 1.0
element vertex 1
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
        test_file.write_text(ply_content)

        # Add binary data (1 vertex, 14 floats)
        data = np.random.randn(14).astype(np.float32)
        with open(test_file, "ab") as f:
            f.write(data.tobytes())

        is_compressed, sh_degree = detect_format(test_file)
        assert is_compressed is False
        assert sh_degree == 0


class TestPropertyNameValidation:
    """Test property name validation in expected properties."""

    def test_sh0_property_names(self):
        """Test SH degree 0 property names."""
        props = EXPECTED_PROPERTIES_BY_SH_DEGREE[0]

        assert props[0:3] == ["x", "y", "z"]
        assert props[3:6] == ["f_dc_0", "f_dc_1", "f_dc_2"]
        assert props[6] == "opacity"
        assert props[7:10] == ["scale_0", "scale_1", "scale_2"]
        assert props[10:14] == ["rot_0", "rot_1", "rot_2", "rot_3"]

    def test_sh3_property_names(self):
        """Test SH degree 3 property names."""
        props = EXPECTED_PROPERTIES_BY_SH_DEGREE[3]

        # Check xyz
        assert props[0:3] == ["x", "y", "z"]

        # Check f_dc
        assert props[3:6] == ["f_dc_0", "f_dc_1", "f_dc_2"]

        # Check f_rest (45 coefficients for degree 3)
        f_rest_props = [f"f_rest_{i}" for i in range(45)]
        assert props[6:51] == f_rest_props

        # Check opacity
        assert props[51] == "opacity"

        # Check scales and quats
        assert props[52:55] == ["scale_0", "scale_1", "scale_2"]
        assert props[55:59] == ["rot_0", "rot_1", "rot_2", "rot_3"]
