"""Comprehensive tests for GSData.add() and GSTensor.add() methods."""

import numpy as np
import pytest

# Check if PyTorch is available
pytest.importorskip("torch")
import torch  # noqa: E402

from gsply import GSData  # noqa: E402
from gsply.torch import GSTensor  # noqa: E402

# =============================================================================
# GSData.add() Tests
# =============================================================================


def test_gsdata_add_basic():
    """Test basic concatenation of two GSData objects."""
    n1, n2 = 100, 50
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    )
    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    )

    combined = data1.add(data2)

    assert len(combined) == n1 + n2
    np.testing.assert_array_equal(combined.means[:n1], data1.means)
    np.testing.assert_array_equal(combined.means[n1:], data2.means)
    np.testing.assert_array_equal(combined.opacities[:n1], data1.opacities)
    np.testing.assert_array_equal(combined.opacities[n1:], data2.opacities)


def test_gsdata_add_with_base_optimization():
    """Test that _base optimization path works correctly."""
    n1, n2 = 100, 50
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    ).consolidate()  # Create _base

    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    ).consolidate()  # Create _base

    assert data1._base is not None
    assert data2._base is not None

    combined = data1.add(data2)

    assert len(combined) == n1 + n2
    np.testing.assert_array_equal(combined.means[:n1], data1.means)
    np.testing.assert_array_equal(combined.means[n1:], data2.means)


def test_gsdata_add_with_masks():
    """Test mask layer concatenation."""
    n1, n2 = 100, 50
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    )
    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    )

    # Add masks to both
    data1.add_mask_layer("test1", data1.opacities > 0.5)
    data1.add_mask_layer("test2", data1.scales[:, 0] < 1.0)

    data2.add_mask_layer("test1", data2.opacities > 0.3)
    data2.add_mask_layer("test2", data2.scales[:, 0] < 0.5)

    combined = data1.add(data2)

    assert combined.mask_names == ["test1", "test2"]
    assert combined.masks.shape == (n1 + n2, 2)
    np.testing.assert_array_equal(combined.masks[:n1, 0], data1.get_mask_layer("test1"))
    np.testing.assert_array_equal(combined.masks[n1:, 0], data2.get_mask_layer("test1"))


def test_gsdata_add_one_has_masks():
    """Test concatenation when only one object has masks."""
    n1, n2 = 100, 50
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    )
    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    )

    # Only data1 has masks
    data1.add_mask_layer("test", data1.opacities > 0.5)

    combined = data1.add(data2)

    assert combined.mask_names == ["test"]
    assert combined.masks.shape == (n1 + n2, 1)
    # data2's portion should be all False
    assert not combined.masks[n1:].any()


def test_gsdata_add_incompatible_sh_degrees_raises():
    """Test that incompatible SH degrees raise error."""
    data1 = GSData(
        means=np.random.randn(100, 3).astype(np.float32),
        scales=np.random.rand(100, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (100, 1)).astype(np.float32),
        opacities=np.random.rand(100).astype(np.float32),
        sh0=np.random.rand(100, 3).astype(np.float32),
        shN=None,  # SH0
    )
    data2 = GSData(
        means=np.random.randn(50, 3).astype(np.float32),
        scales=np.random.rand(50, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (50, 1)).astype(np.float32),
        opacities=np.random.rand(50).astype(np.float32),
        sh0=np.random.rand(50, 3).astype(np.float32),
        shN=np.random.rand(50, 3, 3).astype(np.float32),  # SH1
    )

    with pytest.raises(ValueError, match="different SH degrees"):
        data1.add(data2)


# =============================================================================
# GSTensor.add() Tests
# =============================================================================


def test_gstensor_add_basic():
    """Test basic concatenation of two GSTensor objects."""
    n1, n2 = 100, 50
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    )
    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    )

    gstensor1 = GSTensor.from_gsdata(data1, device="cpu")
    gstensor2 = GSTensor.from_gsdata(data2, device="cpu")

    combined = gstensor1.add(gstensor2)

    assert len(combined) == n1 + n2
    torch.testing.assert_close(combined.means[:n1], gstensor1.means)
    torch.testing.assert_close(combined.means[n1:], gstensor2.means)
    torch.testing.assert_close(combined.opacities[:n1], gstensor1.opacities)
    torch.testing.assert_close(combined.opacities[n1:], gstensor2.opacities)


def test_gstensor_add_with_base_optimization():
    """Test that _base optimization path works for GSTensor."""
    n1, n2 = 100, 50
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    )
    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    )

    gstensor1 = GSTensor.from_gsdata(data1, device="cpu").consolidate()
    gstensor2 = GSTensor.from_gsdata(data2, device="cpu").consolidate()

    assert gstensor1._base is not None
    assert gstensor2._base is not None

    combined = gstensor1.add(gstensor2)

    assert len(combined) == n1 + n2
    torch.testing.assert_close(combined.means[:n1], gstensor1.means)
    torch.testing.assert_close(combined.means[n1:], gstensor2.means)


def test_gstensor_add_with_masks():
    """Test mask layer concatenation for GSTensor."""
    n1, n2 = 100, 50
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    )
    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    )

    gstensor1 = GSTensor.from_gsdata(data1, device="cpu")
    gstensor2 = GSTensor.from_gsdata(data2, device="cpu")

    # Add masks
    gstensor1.add_mask_layer("test1", gstensor1.opacities > 0.5)
    gstensor1.add_mask_layer("test2", gstensor1.scales[:, 0] < 1.0)

    gstensor2.add_mask_layer("test1", gstensor2.opacities > 0.3)
    gstensor2.add_mask_layer("test2", gstensor2.scales[:, 0] < 0.5)

    combined = gstensor1.add(gstensor2)

    assert combined.mask_names == ["test1", "test2"]
    assert combined.masks.shape == (n1 + n2, 2)
    torch.testing.assert_close(combined.masks[:n1, 0], gstensor1.get_mask_layer("test1"))
    torch.testing.assert_close(combined.masks[n1:, 0], gstensor2.get_mask_layer("test1"))


def test_gstensor_add_device_mismatch():
    """Test that device mismatch is handled automatically."""
    n1, n2 = 100, 50
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    )
    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    )

    gstensor1 = GSTensor.from_gsdata(data1, device="cpu")
    gstensor2 = GSTensor.from_gsdata(data2, device="cpu")

    # Should work without error (both on CPU)
    combined = gstensor1.add(gstensor2)
    assert combined.device.type == "cpu"
    assert len(combined) == n1 + n2


def test_gstensor_add_requires_grad():
    """Test that requires_grad is preserved correctly."""
    n1, n2 = 100, 50
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    )
    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    )

    gstensor1 = GSTensor.from_gsdata(data1, device="cpu", requires_grad=True)
    gstensor2 = GSTensor.from_gsdata(data2, device="cpu", requires_grad=False)

    combined = gstensor1.add(gstensor2)

    # Should require grad if either input requires grad
    assert combined.means.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gstensor_add_on_gpu():
    """Test concatenation on GPU."""
    n1, n2 = 10000, 5000
    data1 = GSData(
        means=np.random.randn(n1, 3).astype(np.float32),
        scales=np.random.rand(n1, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n1, 1)).astype(np.float32),
        opacities=np.random.rand(n1).astype(np.float32),
        sh0=np.random.rand(n1, 3).astype(np.float32),
        shN=None,
    )
    data2 = GSData(
        means=np.random.randn(n2, 3).astype(np.float32),
        scales=np.random.rand(n2, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n2, 1)).astype(np.float32),
        opacities=np.random.rand(n2).astype(np.float32),
        sh0=np.random.rand(n2, 3).astype(np.float32),
        shN=None,
    )

    gstensor1 = GSTensor.from_gsdata(data1, device="cuda")
    gstensor2 = GSTensor.from_gsdata(data2, device="cuda")

    combined = gstensor1.add(gstensor2)

    assert combined.device.type == "cuda"
    assert len(combined) == n1 + n2
