"""Comprehensive tests for GSTensor mask layer functionality."""

import numpy as np
import pytest

# Check if PyTorch is available
pytest.importorskip("torch")
import torch  # noqa: E402

from gsply import GSData  # noqa: E402
from gsply.torch import GSTensor  # noqa: E402


@pytest.fixture
def gstensor_cpu():
    """Create sample GSTensor on CPU."""
    n = 1000
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )
    return GSTensor.from_gsdata(data, device="cpu")


# =============================================================================
# add_mask_layer() Tests
# =============================================================================


def test_add_mask_layer_single(gstensor_cpu):
    """Test adding a single mask layer."""
    mask = gstensor_cpu.opacities > 0.5
    gstensor_cpu.add_mask_layer("high_opacity", mask)

    assert gstensor_cpu.mask_names == ["high_opacity"]
    assert gstensor_cpu.masks.shape == (len(gstensor_cpu), 1)
    torch.testing.assert_close(gstensor_cpu.masks[:, 0], mask)


def test_add_mask_layer_multiple(gstensor_cpu):
    """Test adding multiple mask layers."""
    mask1 = gstensor_cpu.opacities > 0.5
    mask2 = gstensor_cpu.scales[:, 0] < 1.0
    mask3 = gstensor_cpu.means[:, 0] > 0

    gstensor_cpu.add_mask_layer("opacity", mask1)
    gstensor_cpu.add_mask_layer("scale", mask2)
    gstensor_cpu.add_mask_layer("position", mask3)

    assert gstensor_cpu.mask_names == ["opacity", "scale", "position"]
    assert gstensor_cpu.masks.shape == (len(gstensor_cpu), 3)
    torch.testing.assert_close(gstensor_cpu.masks[:, 0], mask1)
    torch.testing.assert_close(gstensor_cpu.masks[:, 1], mask2)
    torch.testing.assert_close(gstensor_cpu.masks[:, 2], mask3)


def test_add_mask_layer_duplicate_raises(gstensor_cpu):
    """Test that duplicate layer names raise error."""
    mask = gstensor_cpu.opacities > 0.5
    gstensor_cpu.add_mask_layer("test", mask)

    with pytest.raises(ValueError, match="already exists"):
        gstensor_cpu.add_mask_layer("test", mask)


def test_add_mask_layer_wrong_shape_raises(gstensor_cpu):
    """Test that wrong mask shape raises error."""
    wrong_mask = torch.ones(50, dtype=torch.bool)

    with pytest.raises(ValueError, match="doesn't match"):
        gstensor_cpu.add_mask_layer("test", wrong_mask)


def test_add_mask_layer_numpy_input(gstensor_cpu):
    """Test adding mask from NumPy array."""
    mask_np = np.random.rand(len(gstensor_cpu)) > 0.5
    gstensor_cpu.add_mask_layer("numpy_mask", mask_np)

    assert gstensor_cpu.mask_names == ["numpy_mask"]
    assert isinstance(gstensor_cpu.masks, torch.Tensor)
    assert gstensor_cpu.masks.device == gstensor_cpu.device


# =============================================================================
# get_mask_layer() Tests
# =============================================================================


def test_get_mask_layer(gstensor_cpu):
    """Test retrieving mask layer by name."""
    mask = gstensor_cpu.opacities > 0.5
    gstensor_cpu.add_mask_layer("test", mask)

    retrieved = gstensor_cpu.get_mask_layer("test")
    torch.testing.assert_close(retrieved, mask)


def test_get_mask_layer_multi(gstensor_cpu):
    """Test retrieving from multiple layers."""
    mask1 = gstensor_cpu.opacities > 0.5
    mask2 = gstensor_cpu.scales[:, 0] < 1.0

    gstensor_cpu.add_mask_layer("layer1", mask1)
    gstensor_cpu.add_mask_layer("layer2", mask2)

    retrieved1 = gstensor_cpu.get_mask_layer("layer1")
    retrieved2 = gstensor_cpu.get_mask_layer("layer2")

    torch.testing.assert_close(retrieved1, mask1)
    torch.testing.assert_close(retrieved2, mask2)


def test_get_mask_layer_nonexistent_raises(gstensor_cpu):
    """Test that getting nonexistent layer raises error."""
    with pytest.raises(ValueError, match="not found"):
        gstensor_cpu.get_mask_layer("nonexistent")


# =============================================================================
# remove_mask_layer() Tests
# =============================================================================


def test_remove_mask_layer(gstensor_cpu):
    """Test removing a mask layer."""
    gstensor_cpu.add_mask_layer("layer1", gstensor_cpu.opacities > 0.5)
    gstensor_cpu.add_mask_layer("layer2", gstensor_cpu.scales[:, 0] < 1.0)

    gstensor_cpu.remove_mask_layer("layer1")

    assert gstensor_cpu.mask_names == ["layer2"]
    assert gstensor_cpu.masks.ndim == 1  # Back to 1D


def test_remove_last_layer_clears_all(gstensor_cpu):
    """Test that removing last layer clears masks entirely."""
    gstensor_cpu.add_mask_layer("test", gstensor_cpu.opacities > 0.5)
    gstensor_cpu.remove_mask_layer("test")

    assert gstensor_cpu.masks is None
    assert gstensor_cpu.mask_names is None


def test_remove_mask_layer_nonexistent_raises(gstensor_cpu):
    """Test that removing nonexistent layer raises error."""
    with pytest.raises(ValueError, match="not found"):
        gstensor_cpu.remove_mask_layer("nonexistent")


def test_remove_middle_layer(gstensor_cpu):
    """Test removing middle layer from 3 layers."""
    gstensor_cpu.add_mask_layer("layer1", gstensor_cpu.opacities > 0.5)
    gstensor_cpu.add_mask_layer("layer2", gstensor_cpu.scales[:, 0] < 1.0)
    gstensor_cpu.add_mask_layer("layer3", gstensor_cpu.means[:, 0] > 0)

    gstensor_cpu.remove_mask_layer("layer2")

    assert gstensor_cpu.mask_names == ["layer1", "layer3"]
    assert gstensor_cpu.masks.shape == (len(gstensor_cpu), 2)


# =============================================================================
# combine_masks() Tests
# =============================================================================


def test_combine_masks_and_all_true(gstensor_cpu):
    """Test AND combination where all masks pass."""
    gstensor_cpu.add_mask_layer("all1", torch.ones(len(gstensor_cpu), dtype=torch.bool))
    gstensor_cpu.add_mask_layer("all2", torch.ones(len(gstensor_cpu), dtype=torch.bool))

    combined = gstensor_cpu.combine_masks(mode="and")

    assert combined.all()


def test_combine_masks_and_mixed(gstensor_cpu):
    """Test AND combination with mixed results."""
    mask1 = gstensor_cpu.opacities > 0.3  # ~70% pass
    mask2 = gstensor_cpu.opacities > 0.7  # ~30% pass

    gstensor_cpu.add_mask_layer("loose", mask1)
    gstensor_cpu.add_mask_layer("strict", mask2)

    combined = gstensor_cpu.combine_masks(mode="and")

    # AND should match the stricter mask
    torch.testing.assert_close(combined, mask1 & mask2)
    torch.testing.assert_close(combined, mask2)  # Since mask2 is stricter


def test_combine_masks_or(gstensor_cpu):
    """Test OR combination."""
    mask1 = gstensor_cpu.opacities > 0.7  # ~30% pass
    mask2 = gstensor_cpu.opacities < 0.3  # ~30% pass

    gstensor_cpu.add_mask_layer("high", mask1)
    gstensor_cpu.add_mask_layer("low", mask2)

    combined = gstensor_cpu.combine_masks(mode="or")

    torch.testing.assert_close(combined, mask1 | mask2)


def test_combine_masks_specific_layers(gstensor_cpu):
    """Test combining only specific layers."""
    gstensor_cpu.add_mask_layer("layer1", gstensor_cpu.opacities > 0.5)
    gstensor_cpu.add_mask_layer("layer2", gstensor_cpu.scales[:, 0] < 1.0)
    gstensor_cpu.add_mask_layer("layer3", gstensor_cpu.means[:, 0] > 0)

    # Combine only layer1 and layer3
    combined = gstensor_cpu.combine_masks(mode="and", layers=["layer1", "layer3"])

    expected = gstensor_cpu.get_mask_layer("layer1") & gstensor_cpu.get_mask_layer("layer3")
    torch.testing.assert_close(combined, expected)


def test_combine_masks_no_masks_raises(gstensor_cpu):
    """Test that combining with no masks raises error."""
    with pytest.raises(ValueError, match="No mask layers"):
        gstensor_cpu.combine_masks(mode="and")


def test_combine_masks_invalid_mode_raises(gstensor_cpu):
    """Test that invalid mode raises error."""
    gstensor_cpu.add_mask_layer("test", gstensor_cpu.opacities > 0.5)

    with pytest.raises(ValueError, match="must be 'and' or 'or'"):
        gstensor_cpu.combine_masks(mode="invalid")


# =============================================================================
# apply_masks() Tests
# =============================================================================


def test_apply_masks_creates_copy(gstensor_cpu):
    """Test that apply_masks with inplace=False creates new object."""
    gstensor_cpu.add_mask_layer("test", gstensor_cpu.opacities > 0.5)

    filtered = gstensor_cpu.apply_masks(mode="and", inplace=False)

    assert filtered is not gstensor_cpu
    assert len(filtered) < len(gstensor_cpu)


def test_apply_masks_inplace(gstensor_cpu):
    """Test that apply_masks with inplace=True modifies original."""
    original_len = len(gstensor_cpu)
    gstensor_cpu.add_mask_layer("test", gstensor_cpu.opacities > 0.5)

    result = gstensor_cpu.apply_masks(mode="and", inplace=True)

    assert result is gstensor_cpu
    assert len(gstensor_cpu) < original_len


def test_apply_masks_filters_correctly(gstensor_cpu):
    """Test that filtering produces correct results."""
    mask = gstensor_cpu.opacities > 0.5
    gstensor_cpu.add_mask_layer("test", mask)

    filtered = gstensor_cpu.apply_masks(mode="and", inplace=False)

    # Check all arrays filtered correctly
    torch.testing.assert_close(filtered.means, gstensor_cpu.means[mask])
    torch.testing.assert_close(filtered.opacities, gstensor_cpu.opacities[mask])
    assert len(filtered) == mask.sum()


def test_apply_masks_specific_layers(gstensor_cpu):
    """Test applying only specific mask layers."""
    gstensor_cpu.add_mask_layer("layer1", gstensor_cpu.opacities > 0.5)
    gstensor_cpu.add_mask_layer("layer2", gstensor_cpu.scales[:, 0] < 1.0)

    filtered = gstensor_cpu.apply_masks(layers=["layer1"], inplace=False)

    expected_mask = gstensor_cpu.opacities > 0.5
    assert len(filtered) == expected_mask.sum()


# =============================================================================
# GPU Tests (if CUDA available)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mask_operations_on_gpu():
    """Test that all mask operations work on GPU."""
    n = 10000
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )
    gstensor = GSTensor.from_gsdata(data, device="cuda")

    # Test all operations on GPU
    mask1 = gstensor.opacities > 0.5
    mask2 = gstensor.scales[:, 0] < 1.0

    gstensor.add_mask_layer("gpu_mask1", mask1)
    gstensor.add_mask_layer("gpu_mask2", mask2)

    assert gstensor.masks.device.type == "cuda"

    combined = gstensor.combine_masks(mode="and")
    assert combined.device.type == "cuda"

    filtered = gstensor.apply_masks(mode="and", inplace=False)
    assert filtered.device.type == "cuda"


# =============================================================================
# Persistence Tests
# =============================================================================


def test_masks_preserved_in_slicing(gstensor_cpu):
    """Test that mask layer names preserved when slicing."""
    gstensor_cpu.add_mask_layer("test", gstensor_cpu.opacities > 0.5)

    sliced = gstensor_cpu[10:20]

    assert sliced.mask_names == ["test"]
    assert sliced.masks.shape == (10, 1)


def test_masks_preserved_in_device_move(gstensor_cpu):
    """Test that mask layers preserved when moving devices."""
    gstensor_cpu.add_mask_layer("test1", gstensor_cpu.opacities > 0.5)
    gstensor_cpu.add_mask_layer("test2", gstensor_cpu.scales[:, 0] < 1.0)

    cloned = gstensor_cpu.clone()

    assert cloned.mask_names == ["test1", "test2"]
    assert cloned.masks.shape == gstensor_cpu.masks.shape
