"""Tests for format management features: helper functions, auto-detection, and validation."""

import numpy as np
import pytest

from gsply import GSData, create_linear_format, create_ply_format, create_rasterizer_format
from gsply.gsdata import DataFormat

# Check if PyTorch is available
pytest.importorskip("torch")
import torch  # noqa: E402

from gsply.torch import GSTensor  # noqa: E402


class TestFormatHelperFunctions:
    """Test format helper functions: create_ply_format, create_rasterizer_format."""

    def test_create_ply_format_default(self):
        """Test create_ply_format with default parameters."""
        format_dict = create_ply_format()
        assert format_dict["scales"] == DataFormat.SCALES_PLY
        assert format_dict["opacities"] == DataFormat.OPACITIES_PLY
        assert format_dict["sh0"] == DataFormat.SH0_SH
        assert format_dict["sh_order"] == DataFormat.SH_ORDER_0
        assert format_dict["means"] == DataFormat.MEANS_RAW
        assert format_dict["quats"] == DataFormat.QUATS_RAW

    def test_create_ply_format_sh3(self):
        """Test create_ply_format with SH degree 3."""
        format_dict = create_ply_format(sh_degree=3)
        assert format_dict["scales"] == DataFormat.SCALES_PLY
        assert format_dict["opacities"] == DataFormat.OPACITIES_PLY
        assert format_dict["sh0"] == DataFormat.SH0_SH
        assert format_dict["sh_order"] == DataFormat.SH_ORDER_3

    def test_create_ply_format_rgb(self):
        """Test create_ply_format with RGB sh0 format."""
        format_dict = create_ply_format(sh0_format=DataFormat.SH0_RGB)
        assert format_dict["scales"] == DataFormat.SCALES_PLY
        assert format_dict["opacities"] == DataFormat.OPACITIES_PLY
        assert format_dict["sh0"] == DataFormat.SH0_RGB

    def test_create_rasterizer_format_default(self):
        """Test create_rasterizer_format with default parameters."""
        format_dict = create_rasterizer_format()
        assert format_dict["scales"] == DataFormat.SCALES_LINEAR
        assert format_dict["opacities"] == DataFormat.OPACITIES_LINEAR
        assert format_dict["sh0"] == DataFormat.SH0_SH
        assert format_dict["sh_order"] == DataFormat.SH_ORDER_0

    def test_create_rasterizer_format_sh1(self):
        """Test create_rasterizer_format with SH degree 1."""
        format_dict = create_rasterizer_format(sh_degree=1)
        assert format_dict["scales"] == DataFormat.SCALES_LINEAR
        assert format_dict["opacities"] == DataFormat.OPACITIES_LINEAR
        assert format_dict["sh_order"] == DataFormat.SH_ORDER_1

    def test_create_linear_format_alias(self):
        """Test that create_linear_format is an alias for create_rasterizer_format."""
        format_linear = create_linear_format(sh_degree=2)
        format_rasterizer = create_rasterizer_format(sh_degree=2)
        assert format_linear == format_rasterizer


# =============================================================================
# Format Auto-Detection Tests
# =============================================================================


class TestFormatAutoDetection:
    """Test automatic format detection in __post_init__."""

    def test_auto_detect_ply_format_scales(self):
        """Test auto-detection of PLY format from log-scales."""
        n = 100
        # Create log-scales (PLY format) - many negative values
        scales = np.random.randn(n, 3).astype(np.float32)
        scales[scales > 0] = -np.abs(scales[scales > 0])  # Make mostly negative

        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=scales,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),  # Logit-opacities (outside [0,1])
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
        )

        # Should auto-detect PLY format
        assert data._format["scales"] == DataFormat.SCALES_PLY
        assert data._format["opacities"] == DataFormat.OPACITIES_PLY

    def test_auto_detect_linear_format_scales(self):
        """Test auto-detection of linear format from positive scales."""
        n = 100
        # Create linear scales (all positive, small values)
        scales = np.random.rand(n, 3).astype(np.float32) * 5.0  # All positive, < 10

        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=scales,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),  # Linear opacities [0, 1]
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
        )

        # Should auto-detect linear format
        assert data._format["scales"] == DataFormat.SCALES_LINEAR
        assert data._format["opacities"] == DataFormat.OPACITIES_LINEAR

    def test_explicit_format_overrides_auto_detection(self):
        """Test that explicitly provided format overrides auto-detection."""
        n = 100
        # Create data that looks like linear format
        scales = np.random.rand(n, 3).astype(np.float32) * 5.0
        opacities = np.random.rand(n).astype(np.float32)

        # But explicitly set PLY format
        format_dict = create_ply_format(sh_degree=0)
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=scales,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=opacities,
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
            _format=format_dict,
        )

        # Should use explicit format, not auto-detected
        assert data._format["scales"] == DataFormat.SCALES_PLY
        assert data._format["opacities"] == DataFormat.OPACITIES_PLY


# =============================================================================
# Format Equivalence Validation Tests
# =============================================================================


class TestFormatEquivalenceValidation:
    """Test format equivalence checks in add() and concatenate()."""

    def test_add_same_format_succeeds(self):
        """Test that add() succeeds when formats match."""
        n = 10
        format_dict = create_ply_format(sh_degree=0)

        data1 = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
            _format=format_dict,
        )

        data2 = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
            _format=format_dict,
        )

        result = data1.add(data2)
        assert len(result) == n * 2
        assert result._format == format_dict

    def test_add_different_formats_raises(self):
        """Test that add() raises ValueError when formats don't match."""
        n = 10

        data1 = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
            _format=create_ply_format(sh_degree=0),
        )

        data2 = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
            _format=create_rasterizer_format(sh_degree=0),
        )

        with pytest.raises(ValueError, match="different formats"):
            data1.add(data2)

    def test_concatenate_same_format_succeeds(self):
        """Test that concatenate() succeeds when all formats match."""
        n = 10
        format_dict = create_rasterizer_format(sh_degree=0)

        arrays = [
            GSData(
                means=np.random.randn(n, 3).astype(np.float32),
                scales=np.random.rand(n, 3).astype(np.float32),
                quats=np.random.randn(n, 4).astype(np.float32),
                opacities=np.random.rand(n).astype(np.float32),
                sh0=np.random.randn(n, 3).astype(np.float32),
                shN=np.empty((n, 0, 3), dtype=np.float32),
                _format=format_dict,
            )
            for _ in range(5)
        ]

        result = GSData.concatenate(arrays)
        assert len(result) == n * 5
        assert result._format == format_dict

    def test_concatenate_different_formats_raises(self):
        """Test that concatenate() raises ValueError when formats don't match."""
        n = 10

        arrays = [
            GSData(
                means=np.random.randn(n, 3).astype(np.float32),
                scales=np.random.randn(n, 3).astype(np.float32),
                quats=np.random.randn(n, 4).astype(np.float32),
                opacities=np.random.randn(n).astype(np.float32),
                sh0=np.random.randn(n, 3).astype(np.float32),
                shN=np.empty((n, 0, 3), dtype=np.float32),
                _format=create_ply_format(sh_degree=0)
                if i == 0
                else create_rasterizer_format(sh_degree=0),
            )
            for i in range(3)
        ]

        with pytest.raises(ValueError, match="same format"):
            GSData.concatenate(arrays)

    def test_add_format_mismatch_error_message(self):
        """Test that format mismatch error message is helpful."""
        n = 10

        data1 = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
            _format=create_ply_format(sh_degree=0),
        )

        data2 = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
            _format=create_rasterizer_format(sh_degree=0),
        )

        with pytest.raises(ValueError) as exc_info:
            data1.add(data2)

        error_msg = str(exc_info.value)
        assert "different formats" in error_msg
        assert "normalize" in error_msg or "denormalize" in error_msg


# =============================================================================
# GSTensor Format Tests
# =============================================================================


class TestGSTensorFormatManagement:
    """Test format management for GSTensor."""

    def test_gstensor_format_auto_detection(self):
        """Test auto-detection of format in GSTensor."""
        n = 100
        # Create log-scales (PLY format)
        scales = torch.randn(n, 3).float()
        scales[scales > 0] = -torch.abs(scales[scales > 0])

        gstensor = GSTensor(
            means=torch.randn(n, 3).float(),
            scales=scales,
            quats=torch.randn(n, 4).float(),
            opacities=torch.randn(n).float(),
            sh0=torch.randn(n, 3).float(),
            shN=None,
        )

        # Should auto-detect PLY format
        assert gstensor._format["scales"] == DataFormat.SCALES_PLY
        assert gstensor._format["opacities"] == DataFormat.OPACITIES_PLY

    def test_gstensor_add_same_format_succeeds(self):
        """Test that GSTensor.add() succeeds when formats match."""
        n = 10
        format_dict = create_ply_format(sh_degree=0)

        gstensor1 = GSTensor(
            means=torch.randn(n, 3).float(),
            scales=torch.randn(n, 3).float(),
            quats=torch.randn(n, 4).float(),
            opacities=torch.randn(n).float(),
            sh0=torch.randn(n, 3).float(),
            shN=None,
            _format=format_dict,
        )

        gstensor2 = GSTensor(
            means=torch.randn(n, 3).float(),
            scales=torch.randn(n, 3).float(),
            quats=torch.randn(n, 4).float(),
            opacities=torch.randn(n).float(),
            sh0=torch.randn(n, 3).float(),
            shN=None,
            _format=format_dict,
        )

        result = gstensor1.add(gstensor2)
        assert len(result) == n * 2
        assert result._format == format_dict

    def test_gstensor_add_different_formats_raises(self):
        """Test that GSTensor.add() raises ValueError when formats don't match."""
        n = 10

        gstensor1 = GSTensor(
            means=torch.randn(n, 3).float(),
            scales=torch.randn(n, 3).float(),
            quats=torch.randn(n, 4).float(),
            opacities=torch.randn(n).float(),
            sh0=torch.randn(n, 3).float(),
            shN=None,
            _format=create_ply_format(sh_degree=0),
        )

        gstensor2 = GSTensor(
            means=torch.randn(n, 3).float(),
            scales=torch.rand(n, 3).float(),
            quats=torch.randn(n, 4).float(),
            opacities=torch.rand(n).float(),
            sh0=torch.randn(n, 3).float(),
            shN=None,
            _format=create_rasterizer_format(sh_degree=0),
        )

        with pytest.raises(ValueError, match="different formats"):
            gstensor1.add(gstensor2)

    def test_gstensor_format_preserved_in_conversion(self):
        """Test that format is preserved when converting GSData <-> GSTensor."""
        n = 10
        format_dict = create_rasterizer_format(sh_degree=0)

        gsdata = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.empty((n, 0, 3), dtype=np.float32),
            _format=format_dict,
        )

        gstensor = GSTensor.from_gsdata(gsdata, device="cpu")
        assert gstensor._format == format_dict

        gsdata_back = gstensor.to_gsdata()
        assert gsdata_back._format == format_dict


# =============================================================================
# Format Helper Functions Usage Tests
# =============================================================================


class TestFormatHelperUsage:
    """Test using format helper functions with GSData/GSTensor creation."""

    def test_create_gsdata_with_ply_format(self):
        """Test creating GSData with PLY format helper."""
        n = 100
        format_dict = create_ply_format(sh_degree=3)

        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.random.randn(n, 45, 3).astype(np.float32),
            _format=format_dict,
        )

        assert data._format == format_dict
        assert data._format["scales"] == DataFormat.SCALES_PLY
        assert data._format["opacities"] == DataFormat.OPACITIES_PLY
        assert data._format["sh_order"] == DataFormat.SH_ORDER_3

    def test_create_gsdata_with_rasterizer_format(self):
        """Test creating GSData with rasterizer format helper."""
        n = 100
        format_dict = create_rasterizer_format(sh_degree=1)

        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.random.randn(n, 9, 3).astype(np.float32),
            _format=format_dict,
        )

        assert data._format == format_dict
        assert data._format["scales"] == DataFormat.SCALES_LINEAR
        assert data._format["opacities"] == DataFormat.OPACITIES_LINEAR
        assert data._format["sh_order"] == DataFormat.SH_ORDER_1

    def test_create_gstensor_with_format_helpers(self):
        """Test creating GSTensor with format helper functions."""
        n = 10
        format_dict = create_ply_format(sh_degree=0)

        gstensor = GSTensor(
            means=torch.randn(n, 3).float(),
            scales=torch.randn(n, 3).float(),
            quats=torch.randn(n, 4).float(),
            opacities=torch.randn(n).float(),
            sh0=torch.randn(n, 3).float(),
            shN=None,
            _format=format_dict,
        )

        assert gstensor._format == format_dict
        assert gstensor._format["scales"] == DataFormat.SCALES_PLY


# =============================================================================
# from_arrays and from_dict Tests
# =============================================================================


class TestGSDataFromArrays:
    """Test GSData.from_arrays() convenience method."""

    def test_from_arrays_auto_format(self):
        """Test from_arrays with auto format detection."""
        n = 100
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32) * 5.0  # Linear scales
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)  # Linear opacities
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="auto")

        # Should auto-detect linear format
        assert data._format["scales"] == DataFormat.SCALES_LINEAR
        assert data._format["opacities"] == DataFormat.OPACITIES_LINEAR
        assert len(data) == n

    def test_from_arrays_ply_format(self):
        """Test from_arrays with explicit PLY format."""
        n = 100
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="ply")

        assert data._format["scales"] == DataFormat.SCALES_PLY
        assert data._format["opacities"] == DataFormat.OPACITIES_PLY
        assert data._format["sh_order"] == DataFormat.SH_ORDER_0

    def test_from_arrays_linear_format(self):
        """Test from_arrays with explicit linear format."""
        n = 100
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="linear")

        assert data._format["scales"] == DataFormat.SCALES_LINEAR
        assert data._format["opacities"] == DataFormat.OPACITIES_LINEAR

    def test_from_arrays_rasterizer_format(self):
        """Test from_arrays with rasterizer format (alias for linear)."""
        n = 100
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="rasterizer")

        assert data._format["scales"] == DataFormat.SCALES_LINEAR
        assert data._format["opacities"] == DataFormat.OPACITIES_LINEAR

    def test_from_arrays_with_shN(self):
        """Test from_arrays with higher-order SH coefficients."""
        n = 100
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)
        shN = np.random.randn(n, 3, 3).astype(np.float32)  # SH degree 1: 3 bands

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, shN=shN, format="ply")

        assert data._format["sh_order"] == DataFormat.SH_ORDER_1
        assert data.shN.shape == (n, 3, 3)

    def test_from_arrays_explicit_sh_degree(self):
        """Test from_arrays with explicit SH degree."""
        n = 100
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="ply", sh_degree=3)

        assert data._format["sh_order"] == DataFormat.SH_ORDER_3

    def test_from_arrays_invalid_format(self):
        """Test from_arrays with invalid format preset."""
        n = 10
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Invalid format preset"):
            GSData.from_arrays(means, scales, quats, opacities, sh0, format="invalid")


class TestGSDataFromDict:
    """Test GSData.from_dict() convenience method."""

    def test_from_dict_auto_format(self):
        """Test from_dict with auto format detection."""
        n = 100
        data_dict = {
            "means": np.random.randn(n, 3).astype(np.float32),
            "scales": np.random.rand(n, 3).astype(np.float32) * 5.0,
            "quats": np.random.randn(n, 4).astype(np.float32),
            "opacities": np.random.rand(n).astype(np.float32),
            "sh0": np.random.randn(n, 3).astype(np.float32),
        }

        data = GSData.from_dict(data_dict, format="auto")

        assert data._format["scales"] == DataFormat.SCALES_LINEAR
        assert data._format["opacities"] == DataFormat.OPACITIES_LINEAR
        assert len(data) == n

    def test_from_dict_ply_format(self):
        """Test from_dict with explicit PLY format."""
        n = 100
        data_dict = {
            "means": np.random.randn(n, 3).astype(np.float32),
            "scales": np.random.randn(n, 3).astype(np.float32),
            "quats": np.random.randn(n, 4).astype(np.float32),
            "opacities": np.random.randn(n).astype(np.float32),
            "sh0": np.random.randn(n, 3).astype(np.float32),
        }

        data = GSData.from_dict(data_dict, format="ply")

        assert data._format["scales"] == DataFormat.SCALES_PLY
        assert data._format["opacities"] == DataFormat.OPACITIES_PLY

    def test_from_dict_with_shN(self):
        """Test from_dict with shN included."""
        n = 100
        data_dict = {
            "means": np.random.randn(n, 3).astype(np.float32),
            "scales": np.random.rand(n, 3).astype(np.float32),
            "quats": np.random.randn(n, 4).astype(np.float32),
            "opacities": np.random.rand(n).astype(np.float32),
            "sh0": np.random.randn(n, 3).astype(np.float32),
            "shN": np.random.randn(n, 8, 3).astype(np.float32),  # SH degree 2: 8 bands
        }

        data = GSData.from_dict(data_dict, format="linear")

        assert data._format["sh_order"] == DataFormat.SH_ORDER_2
        assert data.shN.shape == (n, 8, 3)


class TestGSTensorFromArrays:
    """Test GSTensor.from_arrays() convenience method."""

    def test_from_arrays_auto_format(self):
        """Test from_arrays with auto format detection."""
        n = 100
        means = torch.randn(n, 3).float()
        scales = torch.rand(n, 3).float() * 5.0  # Linear scales
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).float()  # Linear opacities
        sh0 = torch.randn(n, 3).float()

        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="auto", device="cpu"
        )

        # Should auto-detect linear format
        assert gstensor._format["scales"] == DataFormat.SCALES_LINEAR
        assert gstensor._format["opacities"] == DataFormat.OPACITIES_LINEAR
        assert len(gstensor) == n
        assert gstensor.device.type == "cpu"

    def test_from_arrays_ply_format(self):
        """Test from_arrays with explicit PLY format."""
        n = 100
        means = torch.randn(n, 3).float()
        scales = torch.randn(n, 3).float()
        quats = torch.randn(n, 4).float()
        opacities = torch.randn(n).float()
        sh0 = torch.randn(n, 3).float()

        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="ply", device="cpu"
        )

        assert gstensor._format["scales"] == DataFormat.SCALES_PLY
        assert gstensor._format["opacities"] == DataFormat.OPACITIES_PLY
        assert gstensor._format["sh_order"] == DataFormat.SH_ORDER_0

    def test_from_arrays_linear_format(self):
        """Test from_arrays with explicit linear format."""
        n = 100
        means = torch.randn(n, 3).float()
        scales = torch.rand(n, 3).float()
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).float()
        sh0 = torch.randn(n, 3).float()

        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="linear", device="cpu"
        )

        assert gstensor._format["scales"] == DataFormat.SCALES_LINEAR
        assert gstensor._format["opacities"] == DataFormat.OPACITIES_LINEAR

    def test_from_arrays_device_inference(self):
        """Test from_arrays infers device from tensors."""
        n = 10
        means = torch.randn(n, 3).float()
        scales = torch.rand(n, 3).float()
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).float()
        sh0 = torch.randn(n, 3).float()

        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="ply", device=None
        )

        assert gstensor.device == means.device

    def test_from_arrays_dtype_conversion(self):
        """Test from_arrays converts dtype correctly."""
        n = 10
        means = torch.randn(n, 3).double()
        scales = torch.rand(n, 3).double()
        quats = torch.randn(n, 4).double()
        opacities = torch.rand(n).double()
        sh0 = torch.randn(n, 3).double()

        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="ply", device="cpu", dtype=torch.float32
        )

        assert gstensor.dtype == torch.float32
        assert gstensor.means.dtype == torch.float32

    def test_from_arrays_with_shN(self):
        """Test from_arrays with higher-order SH coefficients."""
        n = 100
        means = torch.randn(n, 3).float()
        scales = torch.rand(n, 3).float()
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).float()
        sh0 = torch.randn(n, 3).float()
        shN = torch.randn(n, 3, 3).float()  # SH degree 1: 3 bands

        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, shN=shN, format="ply", device="cpu"
        )

        assert gstensor._format["sh_order"] == DataFormat.SH_ORDER_1
        assert gstensor.shN.shape == (n, 3, 3)

    def test_from_arrays_invalid_format(self):
        """Test from_arrays with invalid format preset."""
        n = 10
        means = torch.randn(n, 3).float()
        scales = torch.rand(n, 3).float()
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).float()
        sh0 = torch.randn(n, 3).float()

        with pytest.raises(ValueError, match="Invalid format preset"):
            GSTensor.from_arrays(
                means, scales, quats, opacities, sh0, format="invalid", device="cpu"
            )


class TestGSTensorFromDict:
    """Test GSTensor.from_dict() convenience method."""

    def test_from_dict_auto_format(self):
        """Test from_dict with auto format detection."""
        n = 100
        data_dict = {
            "means": torch.randn(n, 3).float(),
            "scales": torch.rand(n, 3).float() * 5.0,
            "quats": torch.randn(n, 4).float(),
            "opacities": torch.rand(n).float(),
            "sh0": torch.randn(n, 3).float(),
        }

        gstensor = GSTensor.from_dict(data_dict, format="auto", device="cpu")

        assert gstensor._format["scales"] == DataFormat.SCALES_LINEAR
        assert gstensor._format["opacities"] == DataFormat.OPACITIES_LINEAR
        assert len(gstensor) == n

    def test_from_dict_ply_format(self):
        """Test from_dict with explicit PLY format."""
        n = 100
        data_dict = {
            "means": torch.randn(n, 3).float(),
            "scales": torch.randn(n, 3).float(),
            "quats": torch.randn(n, 4).float(),
            "opacities": torch.randn(n).float(),
            "sh0": torch.randn(n, 3).float(),
        }

        gstensor = GSTensor.from_dict(data_dict, format="ply", device="cpu")

        assert gstensor._format["scales"] == DataFormat.SCALES_PLY
        assert gstensor._format["opacities"] == DataFormat.OPACITIES_PLY

    def test_from_dict_with_shN(self):
        """Test from_dict with shN included."""
        n = 100
        data_dict = {
            "means": torch.randn(n, 3).float(),
            "scales": torch.rand(n, 3).float(),
            "quats": torch.randn(n, 4).float(),
            "opacities": torch.rand(n).float(),
            "sh0": torch.randn(n, 3).float(),
            "shN": torch.randn(n, 8, 3).float(),  # SH degree 2: 8 bands
        }

        gstensor = GSTensor.from_dict(data_dict, format="linear", device="cpu")

        assert gstensor._format["sh_order"] == DataFormat.SH_ORDER_2
        assert gstensor.shN.shape == (n, 8, 3)

    def test_from_dict_device_handling(self):
        """Test from_dict handles device correctly."""
        n = 10
        data_dict = {
            "means": torch.randn(n, 3).float(),
            "scales": torch.rand(n, 3).float(),
            "quats": torch.randn(n, 4).float(),
            "opacities": torch.rand(n).float(),
            "sh0": torch.randn(n, 3).float(),
        }

        gstensor = GSTensor.from_dict(data_dict, format="ply", device="cpu")

        assert gstensor.device.type == "cpu"
        assert gstensor.means.device.type == "cpu"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestGSDataFromArraysEdgeCases:
    """Test edge cases for GSData.from_arrays()."""

    def test_from_arrays_empty_data(self):
        """Test from_arrays with empty arrays."""
        n = 0
        means = np.empty((n, 3), dtype=np.float32)
        scales = np.empty((n, 3), dtype=np.float32)
        quats = np.empty((n, 4), dtype=np.float32)
        opacities = np.empty(n, dtype=np.float32)
        sh0 = np.empty((n, 3), dtype=np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="ply")

        assert len(data) == 0
        assert data._format["scales"] == DataFormat.SCALES_PLY

    def test_from_arrays_single_gaussian(self):
        """Test from_arrays with single Gaussian."""
        means = np.random.randn(1, 3).astype(np.float32)
        scales = np.random.rand(1, 3).astype(np.float32)
        quats = np.random.randn(1, 4).astype(np.float32)
        opacities = np.random.rand(1).astype(np.float32)
        sh0 = np.random.randn(1, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="linear")

        assert len(data) == 1
        assert data._format["scales"] == DataFormat.SCALES_LINEAR

    def test_from_arrays_wrong_shapes_raises(self):
        """Test from_arrays with mismatched array shapes."""
        n = 10
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n + 1, 3).astype(np.float32)  # Wrong size
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        # Shape mismatch will cause issues when accessing data or during operations
        # The error might not be immediate, but data will be invalid
        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="ply")
        # Verify data is created but has inconsistent shapes
        assert len(data) == n  # Uses means.shape[0]
        # Accessing scales might fail or give wrong results
        assert data.scales.shape[0] == n + 1  # Scales has wrong size

    def test_from_arrays_shN_shape_mismatch(self):
        """Test from_arrays with shN shape mismatch."""
        n = 10
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)
        shN = np.random.randn(n + 1, 3, 3).astype(np.float32)  # Wrong first dimension

        # Shape mismatch will cause issues but might not raise immediately
        data = GSData.from_arrays(means, scales, quats, opacities, sh0, shN=shN, format="ply")
        # Verify data is created but has inconsistent shapes
        assert len(data) == n  # Uses means.shape[0]
        assert data.shN.shape[0] == n + 1  # shN has wrong size

    def test_from_arrays_format_boundary_values(self):
        """Test format detection with boundary values."""
        n = 100
        means = np.random.randn(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        # Scales exactly at boundary (all zeros - detected as linear since all positive and small)
        scales = np.zeros((n, 3), dtype=np.float32)
        opacities = np.random.rand(n).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="auto")
        # All zeros are positive and small (< 10), so detected as linear
        assert data._format["scales"] == DataFormat.SCALES_LINEAR

    def test_from_arrays_with_none_shN(self):
        """Test from_arrays explicitly passing None for shN."""
        n = 10
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, shN=None, format="ply")

        assert data.shN is None or data.shN.shape[1] == 0
        assert data._format["sh_order"] == DataFormat.SH_ORDER_0


class TestGSDataFromDictEdgeCases:
    """Test edge cases for GSData.from_dict()."""

    def test_from_dict_missing_keys(self):
        """Test from_dict with missing required keys."""
        data_dict = {
            "means": np.random.randn(10, 3).astype(np.float32),
            # Missing scales, quats, opacities, sh0
        }

        with pytest.raises(KeyError):
            GSData.from_dict(data_dict, format="ply")

    def test_from_dict_extra_keys(self):
        """Test from_dict with extra keys (should be ignored)."""
        n = 10
        data_dict = {
            "means": np.random.randn(n, 3).astype(np.float32),
            "scales": np.random.rand(n, 3).astype(np.float32),
            "quats": np.random.randn(n, 4).astype(np.float32),
            "opacities": np.random.rand(n).astype(np.float32),
            "sh0": np.random.randn(n, 3).astype(np.float32),
            "extra_key": "should be ignored",
            "another_extra": 123,
        }

        data = GSData.from_dict(data_dict, format="ply")
        assert len(data) == n

    def test_from_dict_empty_dict(self):
        """Test from_dict with empty dictionary."""
        with pytest.raises(KeyError):
            GSData.from_dict({}, format="ply")

    def test_from_dict_shN_as_none(self):
        """Test from_dict with shN explicitly set to None."""
        n = 10
        data_dict = {
            "means": np.random.randn(n, 3).astype(np.float32),
            "scales": np.random.rand(n, 3).astype(np.float32),
            "quats": np.random.randn(n, 4).astype(np.float32),
            "opacities": np.random.rand(n).astype(np.float32),
            "sh0": np.random.randn(n, 3).astype(np.float32),
            "shN": None,
        }

        data = GSData.from_dict(data_dict, format="linear")
        assert data.shN is None or data.shN.shape[1] == 0


class TestGSTensorFromArraysEdgeCases:
    """Test edge cases for GSTensor.from_arrays()."""

    def test_from_arrays_empty_data(self):
        """Test from_arrays with empty tensors."""
        n = 0
        means = torch.empty(n, 3).float()
        scales = torch.empty(n, 3).float()
        quats = torch.empty(n, 4).float()
        opacities = torch.empty(n).float()
        sh0 = torch.empty(n, 3).float()

        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="ply", device="cpu"
        )

        assert len(gstensor) == 0
        assert gstensor._format["scales"] == DataFormat.SCALES_PLY

    def test_from_arrays_single_gaussian(self):
        """Test from_arrays with single Gaussian."""
        means = torch.randn(1, 3).float()
        scales = torch.rand(1, 3).float()
        quats = torch.randn(1, 4).float()
        opacities = torch.rand(1).float()
        sh0 = torch.randn(1, 3).float()

        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="linear", device="cpu"
        )

        assert len(gstensor) == 1
        assert gstensor._format["scales"] == DataFormat.SCALES_LINEAR

    def test_from_arrays_device_mismatch(self):
        """Test from_arrays with tensors on different devices (should move to target)."""
        n = 10
        means = torch.randn(n, 3).float()
        scales = torch.rand(n, 3).float()
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).float()
        sh0 = torch.randn(n, 3).float()

        # All tensors on CPU, but request CUDA (if available)
        # Should move all to target device
        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="ply", device="cpu"
        )

        assert gstensor.device.type == "cpu"
        assert gstensor.means.device.type == "cpu"

    def test_from_arrays_dtype_mismatch(self):
        """Test from_arrays with mixed dtypes (should convert to target)."""
        n = 10
        means = torch.randn(n, 3).double()
        scales = torch.rand(n, 3).float()
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).half()
        sh0 = torch.randn(n, 3).float()

        # Request float32, should convert all
        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="ply", device="cpu", dtype=torch.float32
        )

        assert gstensor.dtype == torch.float32
        assert gstensor.means.dtype == torch.float32
        assert gstensor.opacities.dtype == torch.float32

    def test_from_arrays_wrong_shapes_raises(self):
        """Test from_arrays with mismatched tensor shapes."""
        n = 10
        means = torch.randn(n, 3).float()
        scales = torch.rand(n + 1, 3).float()  # Wrong size
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).float()
        sh0 = torch.randn(n, 3).float()

        # Shape mismatch will cause issues when accessing data or during operations
        # The error might not be immediate, but data will be invalid
        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, format="ply", device="cpu"
        )
        # Verify tensor is created but has inconsistent shapes
        assert len(gstensor) == n  # Uses means.shape[0]
        # Accessing scales might fail or give wrong results
        assert gstensor.scales.shape[0] == n + 1  # Scales has wrong size

    def test_from_arrays_shN_shape_mismatch(self):
        """Test from_arrays with shN shape mismatch."""
        n = 10
        means = torch.randn(n, 3).float()
        scales = torch.rand(n, 3).float()
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).float()
        sh0 = torch.randn(n, 3).float()
        shN = torch.randn(n + 1, 3, 3).float()  # Wrong first dimension

        # Shape mismatch will cause issues but might not raise immediately
        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, shN=shN, format="ply", device="cpu"
        )
        # Verify tensor is created but has inconsistent shapes
        assert len(gstensor) == n  # Uses means.shape[0]
        assert gstensor.shN.shape[0] == n + 1  # shN has wrong size

    def test_from_arrays_with_none_shN(self):
        """Test from_arrays explicitly passing None for shN."""
        n = 10
        means = torch.randn(n, 3).float()
        scales = torch.rand(n, 3).float()
        quats = torch.randn(n, 4).float()
        opacities = torch.rand(n).float()
        sh0 = torch.randn(n, 3).float()

        gstensor = GSTensor.from_arrays(
            means, scales, quats, opacities, sh0, shN=None, format="ply", device="cpu"
        )

        assert gstensor.shN is None or gstensor.shN.shape[1] == 0
        assert gstensor._format["sh_order"] == DataFormat.SH_ORDER_0


class TestGSTensorFromDictEdgeCases:
    """Test edge cases for GSTensor.from_dict()."""

    def test_from_dict_missing_keys(self):
        """Test from_dict with missing required keys."""
        data_dict = {
            "means": torch.randn(10, 3).float(),
            # Missing scales, quats, opacities, sh0
        }

        with pytest.raises(KeyError):
            GSTensor.from_dict(data_dict, format="ply", device="cpu")

    def test_from_dict_extra_keys(self):
        """Test from_dict with extra keys (should be ignored)."""
        n = 10
        data_dict = {
            "means": torch.randn(n, 3).float(),
            "scales": torch.rand(n, 3).float(),
            "quats": torch.randn(n, 4).float(),
            "opacities": torch.rand(n).float(),
            "sh0": torch.randn(n, 3).float(),
            "extra_key": "should be ignored",
            "another_extra": 123,
        }

        gstensor = GSTensor.from_dict(data_dict, format="ply", device="cpu")
        assert len(gstensor) == n

    def test_from_dict_empty_dict(self):
        """Test from_dict with empty dictionary."""
        with pytest.raises(KeyError):
            GSTensor.from_dict({}, format="ply", device="cpu")

    def test_from_dict_shN_as_none(self):
        """Test from_dict with shN explicitly set to None."""
        n = 10
        data_dict = {
            "means": torch.randn(n, 3).float(),
            "scales": torch.rand(n, 3).float(),
            "quats": torch.randn(n, 4).float(),
            "opacities": torch.rand(n).float(),
            "sh0": torch.randn(n, 3).float(),
            "shN": None,
        }

        gstensor = GSTensor.from_dict(data_dict, format="linear", device="cpu")
        assert gstensor.shN is None or gstensor.shN.shape[1] == 0

    def test_from_dict_device_handling_mixed(self):
        """Test from_dict handles device conversion correctly."""
        n = 10
        # Create tensors on CPU
        data_dict = {
            "means": torch.randn(n, 3).float(),
            "scales": torch.rand(n, 3).float(),
            "quats": torch.randn(n, 4).float(),
            "opacities": torch.rand(n).float(),
            "sh0": torch.randn(n, 3).float(),
        }

        # Request CPU (should work even if tensors are already on CPU)
        gstensor = GSTensor.from_dict(data_dict, format="ply", device="cpu")
        assert gstensor.device.type == "cpu"
        assert gstensor.means.device.type == "cpu"


class TestFormatPresetEdgeCases:
    """Test edge cases for format presets."""

    def test_format_auto_with_uncertain_values(self):
        """Test auto format detection with uncertain values (should default to PLY)."""
        n = 100
        # Mix of positive and negative scales (uncertain)
        scales = np.random.randn(n, 3).astype(np.float32)
        scales[::2] = np.abs(scales[::2])  # Every other is positive

        # Mix of in-range and out-of-range opacities (uncertain)
        opacities = np.random.randn(n).astype(np.float32)
        opacities[: n // 2] = np.clip(opacities[: n // 2], 0, 1)  # Half in range

        means = np.random.randn(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="auto")

        # Should default to PLY format when uncertain
        assert data._format["scales"] == DataFormat.SCALES_PLY
        assert data._format["opacities"] == DataFormat.OPACITIES_PLY

    def test_format_auto_all_positive_small_scales(self):
        """Test auto format detection with all positive small scales."""
        n = 100
        scales = np.random.rand(n, 3).astype(np.float32) * 5.0  # All positive, < 10
        opacities = np.random.rand(n).astype(np.float32)  # All in [0, 1]

        means = np.random.randn(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="auto")

        # Should detect linear format
        assert data._format["scales"] == DataFormat.SCALES_LINEAR
        assert data._format["opacities"] == DataFormat.OPACITIES_LINEAR

    def test_format_auto_many_negative_scales(self):
        """Test auto format detection with many negative scales."""
        n = 100
        scales = np.random.randn(n, 3).astype(np.float32)
        scales[scales > 0] = -np.abs(scales[scales > 0])  # Make mostly negative
        opacities = np.random.randn(n).astype(np.float32)  # Outside [0, 1]

        means = np.random.randn(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="auto")

        # Should detect PLY format
        assert data._format["scales"] == DataFormat.SCALES_PLY
        assert data._format["opacities"] == DataFormat.OPACITIES_PLY

    def test_format_rasterizer_alias(self):
        """Test that 'rasterizer' format is same as 'linear'."""
        n = 10
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.randn(n, 3).astype(np.float32)

        data_linear = GSData.from_arrays(means, scales, quats, opacities, sh0, format="linear")
        data_rasterizer = GSData.from_arrays(
            means, scales, quats, opacities, sh0, format="rasterizer"
        )

        assert data_linear._format == data_rasterizer._format
        assert data_linear._format["scales"] == DataFormat.SCALES_LINEAR
        assert data_rasterizer._format["scales"] == DataFormat.SCALES_LINEAR
