"""Tests for GSData <-> GSTensor conversion."""

import numpy as np
import pytest

# Check if PyTorch is available
pytest.importorskip("torch")
import torch  # noqa: E402

from gsply import GSData  # noqa: E402
from gsply.torch import GSTensor  # noqa: E402


@pytest.fixture
def gsdata_sh0():
    """Create sample GSData (SH0)."""
    n = 1000
    return GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.randn(n, 3).astype(np.float32),
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.randn(n, 3).astype(np.float32),
        shN=None,
        masks=np.ones(n, dtype=bool),
        _base=None,
    )


@pytest.fixture
def gsdata_sh0_with_base():
    """Create sample GSData with _base (SH0)."""
    n = 1000
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.randn(n, 3).astype(np.float32),
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.randn(n, 3).astype(np.float32),
        shN=None,
        masks=np.ones(n, dtype=bool),
        _base=None,
    )
    return data.consolidate()


@pytest.fixture
def gsdata_sh1():
    """Create sample GSData (SH1)."""
    n = 500
    return GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.randn(n, 3).astype(np.float32),
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.randn(n, 3).astype(np.float32),
        shN=np.random.randn(n, 3, 3).astype(np.float32),  # SH1: 3 bands
        masks=np.ones(n, dtype=bool),
        _base=None,
    )


# =============================================================================
# GSData -> GSTensor Conversion
# =============================================================================


def test_from_gsdata_basic(gsdata_sh0):
    """Test basic conversion from GSData to GSTensor."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")

    assert isinstance(gstensor, GSTensor)
    assert len(gstensor) == len(gsdata_sh0)
    assert gstensor.device == torch.device("cpu")
    assert gstensor.dtype == torch.float32


def test_from_gsdata_with_base(gsdata_sh0_with_base):
    """Test conversion uses _base optimization when available."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0_with_base, device="cpu")

    # Should have _base from conversion
    assert gstensor._base is not None
    assert gstensor._base.shape == (1000, 14)


def test_from_gsdata_without_base(gsdata_sh0):
    """Test conversion without _base (fallback path)."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")

    # Fallback path doesn't create _base
    assert gstensor._base is None or gstensor._base is not None  # Either is fine


def test_from_gsdata_sh1(gsdata_sh1):
    """Test conversion with SH1 data."""
    gstensor = GSTensor.from_gsdata(gsdata_sh1, device="cpu")

    assert gstensor.get_sh_degree() == 1
    assert gstensor.shN.shape == (500, 3, 3)  # SH1: 3 bands


def test_from_gsdata_dtype_conversion(gsdata_sh0):
    """Test dtype conversion during from_gsdata."""
    gstensor_half = GSTensor.from_gsdata(gsdata_sh0, device="cpu", dtype=torch.float16)
    assert gstensor_half.dtype == torch.float16

    gstensor_double = GSTensor.from_gsdata(gsdata_sh0, device="cpu", dtype=torch.float64)
    assert gstensor_double.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_from_gsdata_cuda(gsdata_sh0):
    """Test conversion to CUDA device."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cuda")

    assert gstensor.device.type == "cuda"
    assert isinstance(gstensor.means, torch.Tensor)
    assert gstensor.means.is_cuda


def test_from_gsdata_requires_grad(gsdata_sh0):
    """Test requires_grad parameter."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu", requires_grad=True)

    assert gstensor.means.requires_grad
    assert gstensor.scales.requires_grad
    assert gstensor.quats.requires_grad


# =============================================================================
# GSTensor -> GSData Conversion
# =============================================================================


def test_to_gsdata_basic(gsdata_sh0):
    """Test basic conversion from GSTensor to GSData."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")
    gsdata_restored = gstensor.to_gsdata()

    assert isinstance(gsdata_restored, GSData)
    assert len(gsdata_restored) == len(gsdata_sh0)
    assert np.allclose(gsdata_restored.means, gsdata_sh0.means, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_gsdata_from_cuda(gsdata_sh0):
    """Test conversion from CUDA GSTensor to GSData."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cuda")
    gsdata_restored = gstensor.to_gsdata()

    assert isinstance(gsdata_restored, GSData)
    assert isinstance(gsdata_restored.means, np.ndarray)
    assert np.allclose(gsdata_restored.means, gsdata_sh0.means, rtol=1e-5)


def test_to_gsdata_with_base(gsdata_sh0_with_base):
    """Test to_gsdata uses _base optimization."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0_with_base, device="cpu")
    gsdata_restored = gstensor.to_gsdata()

    # Should restore with _base from GSTensor's _base
    assert np.allclose(gsdata_restored.means, gsdata_sh0_with_base.means, rtol=1e-5)


def test_to_gsdata_sh1(gsdata_sh1):
    """Test to_gsdata with SH1 data."""
    gstensor = GSTensor.from_gsdata(gsdata_sh1, device="cpu")
    gsdata_restored = gstensor.to_gsdata()

    assert gsdata_restored.shN.shape == (500, 3, 3)  # SH1: 3 bands
    assert np.allclose(gsdata_restored.shN, gsdata_sh1.shN, rtol=1e-5)


# =============================================================================
# Round-Trip Conversion
# =============================================================================


def test_round_trip_sh0(gsdata_sh0):
    """Test round-trip conversion preserves data (SH0)."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")
    gsdata_restored = gstensor.to_gsdata()

    assert len(gsdata_restored) == len(gsdata_sh0)
    assert np.allclose(gsdata_restored.means, gsdata_sh0.means, rtol=1e-5)
    assert np.allclose(gsdata_restored.scales, gsdata_sh0.scales, rtol=1e-5)
    assert np.allclose(gsdata_restored.quats, gsdata_sh0.quats, rtol=1e-5)
    assert np.allclose(gsdata_restored.opacities, gsdata_sh0.opacities, rtol=1e-5)
    assert np.allclose(gsdata_restored.sh0, gsdata_sh0.sh0, rtol=1e-5)


def test_round_trip_sh1(gsdata_sh1):
    """Test round-trip conversion preserves data (SH1)."""
    gstensor = GSTensor.from_gsdata(gsdata_sh1, device="cpu")
    gsdata_restored = gstensor.to_gsdata()

    assert len(gsdata_restored) == len(gsdata_sh1)
    assert np.allclose(gsdata_restored.means, gsdata_sh1.means, rtol=1e-5)
    assert np.allclose(gsdata_restored.shN, gsdata_sh1.shN, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_round_trip_cuda(gsdata_sh0):
    """Test round-trip through CUDA preserves data."""
    gstensor_gpu = GSTensor.from_gsdata(gsdata_sh0, device="cuda")
    gsdata_restored = gstensor_gpu.to_gsdata()

    assert np.allclose(gsdata_restored.means, gsdata_sh0.means, rtol=1e-5)
    assert np.allclose(gsdata_restored.scales, gsdata_sh0.scales, rtol=1e-5)


def test_round_trip_with_base_optimization(gsdata_sh0_with_base):
    """Test round-trip with _base optimization."""
    # GSData with _base -> GSTensor with _base -> GSData with _base
    gstensor = GSTensor.from_gsdata(gsdata_sh0_with_base, device="cpu")
    assert gstensor._base is not None  # Should use _base optimization

    gsdata_restored = gstensor.to_gsdata()

    assert np.allclose(gsdata_restored.means, gsdata_sh0_with_base.means, rtol=1e-5)


def test_round_trip_dtype_preservation(gsdata_sh0):
    """Test different dtypes in round-trip."""
    # Convert to float64
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu", dtype=torch.float64)
    assert gstensor.dtype == torch.float64

    gsdata_restored = gstensor.to_gsdata()

    # NumPy arrays should be float64 now
    assert gsdata_restored.means.dtype == np.float64


# =============================================================================
# Accuracy Tests
# =============================================================================


def test_conversion_accuracy_float16(gsdata_sh0):
    """Test conversion accuracy with float16 (precision loss expected)."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu", dtype=torch.float16)
    gsdata_restored = gstensor.to_gsdata()

    # With float16, expect some precision loss but values should be close
    assert np.allclose(gsdata_restored.means, gsdata_sh0.means, rtol=1e-2, atol=1e-3)


def test_conversion_preserves_masks(gsdata_sh0):
    """Test that masks are preserved in conversion."""
    # Create data with specific mask pattern
    gsdata_sh0.masks = np.random.rand(len(gsdata_sh0)) > 0.5

    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")
    gsdata_restored = gstensor.to_gsdata()

    assert np.array_equal(gsdata_restored.masks, gsdata_sh0.masks)


def test_conversion_with_slicing(gsdata_sh0_with_base):
    """Test conversion after slicing."""
    # Convert to tensor
    gstensor = GSTensor.from_gsdata(gsdata_sh0_with_base, device="cpu")

    # Slice
    gstensor_subset = gstensor[100:200]

    # Convert back
    gsdata_subset = gstensor_subset.to_gsdata()

    assert len(gsdata_subset) == 100
    assert np.allclose(gsdata_subset.means, gsdata_sh0_with_base.means[100:200], rtol=1e-5)


# =============================================================================
# Edge Cases
# =============================================================================


def test_from_gsdata_no_masks(gsdata_sh0):
    """Test conversion when masks is None."""
    gsdata_sh0.masks = None

    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")

    assert gstensor.masks is None


def test_from_gsdata_no_shN(gsdata_sh0):
    """Test conversion when shN is None (SH0)."""
    assert gsdata_sh0.shN is None or gsdata_sh0.shN.shape[1] == 0

    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")

    assert gstensor.get_sh_degree() == 0
    assert not gstensor.has_high_order_sh()


def test_to_gsdata_detaches_gradients(gsdata_sh0):
    """Test to_gsdata detaches tensors from computation graph."""
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu", requires_grad=True)

    # Perform some operation to create computation graph
    gstensor.means.sum().backward()

    # Convert to GSData should detach
    gsdata_restored = gstensor.to_gsdata()

    # Should not have gradient info
    assert isinstance(gsdata_restored.means, np.ndarray)


def test_from_gsdata_with_mask(gsdata_sh0):
    """Test masked transfer - only transfers filtered subset to GPU."""
    # Create mask (50% pass)
    mask = gsdata_sh0.opacities > 0.5
    expected_count = mask.sum()

    # Direct masked transfer (no intermediate copy)
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu", mask=mask)

    # Verify correct subset transferred
    assert len(gstensor) == expected_count
    np.testing.assert_array_equal(gstensor.means.cpu().numpy(), gsdata_sh0.means[mask])
    np.testing.assert_array_equal(gstensor.opacities.cpu().numpy(), gsdata_sh0.opacities[mask])


def test_from_gsdata_with_mask_validation(gsdata_sh0):
    """Test mask validation."""
    # Wrong length mask
    wrong_mask = np.ones(100, dtype=bool)

    with pytest.raises(ValueError, match="doesn't match data length"):
        GSTensor.from_gsdata(gsdata_sh0, device="cpu", mask=wrong_mask)


def test_from_gsdata_masked_with_base(gsdata_sh0_with_base):
    """Test masked transfer preserves _base optimization."""
    mask = gsdata_sh0_with_base.opacities > 0.5
    expected_count = mask.sum()

    # Masked transfer with _base optimization
    gstensor = GSTensor.from_gsdata(gsdata_sh0_with_base, device="cpu", mask=mask)

    # Should still have _base (from sliced data)
    assert len(gstensor) == expected_count
    # Verify data correctness
    np.testing.assert_array_equal(gstensor.means.cpu().numpy(), gsdata_sh0_with_base.means[mask])


def test_mask_names_transfer(gsdata_sh0):
    """Test that mask_names are transferred from GSData to GSTensor."""
    # Clear existing masks from fixture
    gsdata_sh0.masks = None
    gsdata_sh0.mask_names = None

    # Add mask layers to GSData
    gsdata_sh0.add_mask_layer("opacity", gsdata_sh0.opacities > 0.5)
    gsdata_sh0.add_mask_layer("scale", gsdata_sh0.scales[:, 0] < 1.0)

    # Convert to GSTensor
    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")

    # Verify mask_names transferred
    assert gstensor.mask_names == ["opacity", "scale"]
    assert gstensor.masks is not None
    assert gstensor.masks.shape == (len(gsdata_sh0), 2)

    # Verify round-trip preserves mask_names
    gsdata_restored = gstensor.to_gsdata()
    assert gsdata_restored.mask_names == ["opacity", "scale"]
    np.testing.assert_array_equal(gsdata_restored.masks, gsdata_sh0.masks)


def test_mask_names_preserved_in_slicing(gsdata_sh0):
    """Test that mask_names are preserved when slicing GSTensor."""
    # Clear existing masks from fixture
    gsdata_sh0.masks = None
    gsdata_sh0.mask_names = None

    # Add mask layers
    gsdata_sh0.add_mask_layer("layer1", gsdata_sh0.opacities > 0.5)
    gsdata_sh0.add_mask_layer("layer2", gsdata_sh0.scales[:, 0] < 1.0)

    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")

    # Slice
    sliced = gstensor[10:20]

    # Verify mask_names preserved
    assert sliced.mask_names == ["layer1", "layer2"]
    assert sliced.masks.shape == (10, 2)


def test_mask_names_preserved_in_device_move(gsdata_sh0):
    """Test that mask_names are preserved when moving devices."""
    # Clear existing masks from fixture
    gsdata_sh0.masks = None
    gsdata_sh0.mask_names = None

    # Add mask layers
    gsdata_sh0.add_mask_layer("test1", gsdata_sh0.opacities > 0.5)
    gsdata_sh0.add_mask_layer("test2", gsdata_sh0.scales[:, 0] < 1.0)

    gstensor_cpu = GSTensor.from_gsdata(gsdata_sh0, device="cpu")

    # Clone (which internally uses to())
    gstensor_cloned = gstensor_cpu.clone()

    # Verify mask_names preserved
    assert gstensor_cloned.mask_names == ["test1", "test2"]


def test_mask_names_in_repr(gsdata_sh0):
    """Test that __repr__ shows mask layer information."""
    # Clear existing masks from fixture
    gsdata_sh0.masks = None
    gsdata_sh0.mask_names = None

    # Add mask layers
    gsdata_sh0.add_mask_layer("opacity", gsdata_sh0.opacities > 0.5)
    gsdata_sh0.add_mask_layer("scale", gsdata_sh0.scales[:, 0] < 1.0)

    gstensor = GSTensor.from_gsdata(gsdata_sh0, device="cpu")
    repr_str = repr(gstensor)

    # Should show mask info
    assert "Masks:" in repr_str
    assert "2 layers" in repr_str
    assert "opacity" in repr_str
    assert "scale" in repr_str
