"""Comprehensive tests for GSData mask layer functionality."""

import numpy as np
import pytest

from gsply import GSData


@pytest.fixture
def sample_data():
    """Create sample GSData for testing."""
    n = 100
    return GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )


class TestMaskLayerManagement:
    """Test mask layer add/get/remove operations."""

    def test_add_single_layer(self, sample_data):
        """Test adding a single mask layer."""
        mask = sample_data.opacities > 0.5
        sample_data.add_mask_layer("high_opacity", mask)

        assert sample_data.mask_names == ["high_opacity"]
        assert sample_data.masks.shape == (len(sample_data), 1)
        np.testing.assert_array_equal(sample_data.masks[:, 0], mask)

    def test_add_multiple_layers(self, sample_data):
        """Test adding multiple mask layers."""
        mask1 = sample_data.opacities > 0.5
        mask2 = sample_data.scales[:, 0] < 1.0
        mask3 = sample_data.means[:, 0] > 0

        sample_data.add_mask_layer("opacity", mask1)
        sample_data.add_mask_layer("scale", mask2)
        sample_data.add_mask_layer("position", mask3)

        assert sample_data.mask_names == ["opacity", "scale", "position"]
        assert sample_data.masks.shape == (len(sample_data), 3)
        np.testing.assert_array_equal(sample_data.masks[:, 0], mask1)
        np.testing.assert_array_equal(sample_data.masks[:, 1], mask2)
        np.testing.assert_array_equal(sample_data.masks[:, 2], mask3)

    def test_add_duplicate_name_raises(self, sample_data):
        """Test that duplicate layer names raise error."""
        mask = sample_data.opacities > 0.5
        sample_data.add_mask_layer("test", mask)

        with pytest.raises(ValueError, match="already exists"):
            sample_data.add_mask_layer("test", mask)

    def test_add_wrong_shape_raises(self, sample_data):
        """Test that wrong mask shape raises error."""
        wrong_mask = np.ones(50, dtype=bool)  # Wrong length

        with pytest.raises(ValueError, match="doesn't match"):
            sample_data.add_mask_layer("test", wrong_mask)

    def test_get_mask_layer(self, sample_data):
        """Test retrieving mask layer by name."""
        mask = sample_data.opacities > 0.5
        sample_data.add_mask_layer("test", mask)

        retrieved = sample_data.get_mask_layer("test")
        np.testing.assert_array_equal(retrieved, mask)

    def test_get_nonexistent_layer_raises(self, sample_data):
        """Test that getting nonexistent layer raises error."""
        with pytest.raises(ValueError, match="not found"):
            sample_data.get_mask_layer("nonexistent")

    def test_remove_mask_layer(self, sample_data):
        """Test removing a mask layer."""
        sample_data.add_mask_layer("layer1", sample_data.opacities > 0.5)
        sample_data.add_mask_layer("layer2", sample_data.scales[:, 0] < 1.0)

        sample_data.remove_mask_layer("layer1")

        assert sample_data.mask_names == ["layer2"]
        assert sample_data.masks.ndim == 1  # Back to 1D

    def test_remove_last_layer_clears_all(self, sample_data):
        """Test that removing last layer clears masks entirely."""
        sample_data.add_mask_layer("test", sample_data.opacities > 0.5)
        sample_data.remove_mask_layer("test")

        assert sample_data.masks is None
        assert sample_data.mask_names is None

    def test_remove_nonexistent_layer_raises(self, sample_data):
        """Test that removing nonexistent layer raises error."""
        with pytest.raises(ValueError, match="not found"):
            sample_data.remove_mask_layer("nonexistent")


class TestMaskCombination:
    """Test mask combination operations."""

    def test_combine_and_all_true(self, sample_data):
        """Test AND combination where all masks pass."""
        sample_data.add_mask_layer("all1", np.ones(len(sample_data), dtype=bool))
        sample_data.add_mask_layer("all2", np.ones(len(sample_data), dtype=bool))

        combined = sample_data.combine_masks(mode="and")

        assert combined.all()

    def test_combine_and_mixed(self, sample_data):
        """Test AND combination with mixed results."""
        mask1 = sample_data.opacities > 0.3  # ~70% pass
        mask2 = sample_data.opacities > 0.7  # ~30% pass

        sample_data.add_mask_layer("loose", mask1)
        sample_data.add_mask_layer("strict", mask2)

        combined = sample_data.combine_masks(mode="and")

        # AND should match the stricter mask
        np.testing.assert_array_equal(combined, mask1 & mask2)
        np.testing.assert_array_equal(combined, mask2)  # Since mask2 is stricter

    def test_combine_or(self, sample_data):
        """Test OR combination."""
        mask1 = sample_data.opacities > 0.7  # ~30% pass
        mask2 = sample_data.opacities < 0.3  # ~30% pass

        sample_data.add_mask_layer("high", mask1)
        sample_data.add_mask_layer("low", mask2)

        combined = sample_data.combine_masks(mode="or")

        np.testing.assert_array_equal(combined, mask1 | mask2)

    def test_combine_specific_layers(self, sample_data):
        """Test combining only specific layers."""
        sample_data.add_mask_layer("layer1", sample_data.opacities > 0.5)
        sample_data.add_mask_layer("layer2", sample_data.scales[:, 0] < 1.0)
        sample_data.add_mask_layer("layer3", sample_data.means[:, 0] > 0)

        # Combine only layer1 and layer3
        combined = sample_data.combine_masks(mode="and", layers=["layer1", "layer3"])

        expected = sample_data.get_mask_layer("layer1") & sample_data.get_mask_layer("layer3")
        np.testing.assert_array_equal(combined, expected)

    def test_combine_no_masks_raises(self, sample_data):
        """Test that combining with no masks raises error."""
        with pytest.raises(ValueError, match="No mask layers"):
            sample_data.combine_masks(mode="and")

    def test_combine_invalid_mode_raises(self, sample_data):
        """Test that invalid mode raises error."""
        sample_data.add_mask_layer("test", sample_data.opacities > 0.5)

        with pytest.raises(ValueError, match="must be 'and' or 'or'"):
            sample_data.combine_masks(mode="invalid")


class TestMaskApplication:
    """Test applying masks to filter GSData."""

    def test_apply_masks_creates_copy(self, sample_data):
        """Test that apply_masks with inplace=False creates new object."""
        sample_data.add_mask_layer("test", sample_data.opacities > 0.5)

        filtered = sample_data.apply_masks(mode="and", inplace=False)

        assert filtered is not sample_data
        assert len(filtered) < len(sample_data)

    def test_apply_masks_inplace(self, sample_data):
        """Test that apply_masks with inplace=True modifies original."""
        original_len = len(sample_data)
        sample_data.add_mask_layer("test", sample_data.opacities > 0.5)

        result = sample_data.apply_masks(mode="and", inplace=True)

        assert result is sample_data
        assert len(sample_data) < original_len

    def test_apply_masks_filters_correctly(self, sample_data):
        """Test that filtering produces correct results."""
        mask = sample_data.opacities > 0.5
        sample_data.add_mask_layer("test", mask)

        filtered = sample_data.apply_masks(mode="and", inplace=False)

        # Check all arrays filtered correctly
        np.testing.assert_array_equal(filtered.means, sample_data.means[mask])
        np.testing.assert_array_equal(filtered.opacities, sample_data.opacities[mask])
        assert len(filtered) == mask.sum()

    def test_apply_specific_layers(self, sample_data):
        """Test applying only specific mask layers."""
        sample_data.add_mask_layer("layer1", sample_data.opacities > 0.5)
        sample_data.add_mask_layer("layer2", sample_data.scales[:, 0] < 1.0)

        filtered = sample_data.apply_masks(layers=["layer1"], inplace=False)

        expected_mask = sample_data.opacities > 0.5
        assert len(filtered) == expected_mask.sum()


class TestMaskPersistence:
    """Test that masks persist through copy/slice operations."""

    def test_masks_preserved_in_copy(self, sample_data):
        """Test that mask layers are preserved when copying."""
        sample_data.add_mask_layer("test1", sample_data.opacities > 0.5)
        sample_data.add_mask_layer("test2", sample_data.scales[:, 0] < 1.0)

        copied = sample_data.copy()

        assert copied.mask_names == sample_data.mask_names
        np.testing.assert_array_equal(copied.masks, sample_data.masks)

    def test_masks_preserved_in_slice(self, sample_data):
        """Test that mask layer names preserved when slicing."""
        sample_data.add_mask_layer("test", sample_data.opacities > 0.5)

        sliced = sample_data[10:20]

        assert sliced.mask_names == ["test"]
        assert sliced.masks.shape == (10, 1)  # 10 Gaussians, 1 layer

    def test_masks_preserved_in_boolean_indexing(self, sample_data):
        """Test that mask layer names preserved with boolean indexing."""
        sample_data.add_mask_layer("layer1", sample_data.opacities > 0.5)
        sample_data.add_mask_layer("layer2", sample_data.scales[:, 0] < 1.0)

        filter_mask = sample_data.opacities > 0.7
        filtered = sample_data[filter_mask]

        assert filtered.mask_names == ["layer1", "layer2"]
        assert filtered.masks.shape[1] == 2  # Still 2 layers


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_data(self):
        """Test mask operations on empty GSData."""
        data = GSData(
            means=np.empty((0, 3), dtype=np.float32),
            scales=np.empty((0, 3), dtype=np.float32),
            quats=np.empty((0, 4), dtype=np.float32),
            opacities=np.empty(0, dtype=np.float32),
            sh0=np.empty((0, 3), dtype=np.float32),
            shN=None,
        )

        empty_mask = np.empty(0, dtype=bool)
        data.add_mask_layer("test", empty_mask)

        combined = data.combine_masks(mode="and")
        assert len(combined) == 0

    def test_all_rejected(self, sample_data):
        """Test filtering where all Gaussians are rejected."""
        sample_data.add_mask_layer("reject_all", np.zeros(len(sample_data), dtype=bool))

        filtered = sample_data.apply_masks(mode="and", inplace=False)

        assert len(filtered) == 0

    def test_all_accepted(self, sample_data):
        """Test filtering where all Gaussians are accepted."""
        sample_data.add_mask_layer("accept_all", np.ones(len(sample_data), dtype=bool))

        filtered = sample_data.apply_masks(mode="and", inplace=False)

        assert len(filtered) == len(sample_data)

    def test_single_layer_combination(self, sample_data):
        """Test combining a single layer (should return same mask)."""
        mask = sample_data.opacities > 0.5
        sample_data.add_mask_layer("test", mask)

        combined = sample_data.combine_masks(mode="and")

        np.testing.assert_array_equal(combined, mask)
