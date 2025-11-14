"""Test dataclass functionality."""

import numpy as np

from gsply import GSData


class TestDataclassRefactoring:
    """Test that the dataclass works correctly."""

    def test_dataclass_creation(self):
        """Test that we can create a GSData instance."""
        # Create test data
        n = 100
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.rand(n, 3).astype(np.float32)
        shN = np.random.rand(n, 15, 3).astype(np.float32)  # noqa: N806

        # Create GSData instance
        data = GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            _base=None,
        )

        # Test attribute access
        assert data.means is means
        assert data.scales is scales
        assert data.quats is quats
        assert data.opacities is opacities
        assert data.sh0 is sh0
        assert data.shN is shN
        assert data._base is None

    def test_len_returns_gaussians(self):
        """Test that len() returns the number of Gaussians."""
        # Create test data
        n = 100
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.rand(n, 3).astype(np.float32),
            shN=np.random.rand(n, 15, 3).astype(np.float32),
            masks=None,
            _base=None,
        )

        # len() should return number of Gaussians
        assert len(data) == n
        assert len(data) == data.means.shape[0]

    def test_masks_attribute(self):
        """Test that masks attribute works correctly."""
        # Create test data
        n = 100
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.rand(n, 3).astype(np.float32)
        shN = np.random.rand(n, 15, 3).astype(np.float32)  # noqa: N806

        # Test with default masks (None)
        data = GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            _base=None,
        )
        assert data.masks is None  # Default is None

        # Test with custom masks
        custom_masks = np.random.rand(n) > 0.5
        data.masks = custom_masks
        assert np.array_equal(data.masks, custom_masks)

    def test_mutability(self):
        """Test that dataclass fields are mutable."""
        n = 25
        means = np.random.randn(n, 3).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        quats = np.random.randn(n, 4).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        sh0 = np.random.rand(n, 3).astype(np.float32)
        shN = np.random.rand(n, 15, 3).astype(np.float32)  # noqa: N806

        data = GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            _base=None,
        )

        # Test that we can mutate fields
        new_means = np.random.randn(n, 3).astype(np.float32)
        data.means = new_means
        assert data.means is new_means

        # Test that we can mutate array elements
        data.scales[0, 0] = 999.0
        assert data.scales[0, 0] == 999.0

    def test_private_base_field(self):
        """Test that _base field is private."""
        n = 20
        base_array = np.random.randn(n, 59).astype(np.float32)

        data = GSData(
            means=base_array[:, 0:3],
            scales=base_array[:, 6:9],
            quats=base_array[:, 9:13],
            opacities=base_array[:, 13],
            sh0=base_array[:, 3:6],
            shN=base_array[:, 14:59].reshape(n, 15, 3),
            _base=base_array,  # Private field
        )

        # _base should be accessible
        assert data._base is base_array

        # Verify that the views share memory with base
        assert np.shares_memory(data.means, base_array)
        assert np.shares_memory(data.scales, base_array)

    def test_zero_copy_views_with_base(self):
        """Test that zero-copy views work correctly with _base."""
        n = 100
        base = np.random.randn(n, 59).astype(np.float32)

        # Create zero-copy views
        means_view = base[:, 0:3]
        scales_view = base[:, 6:9]
        quats_view = base[:, 9:13]
        opacities_view = base[:, 13]
        sh0_view = base[:, 3:6]
        shN_view = base[:, 14:59].reshape(n, 15, 3)  # noqa: N806

        data = GSData(
            means=means_view,
            scales=scales_view,
            quats=quats_view,
            opacities=opacities_view,
            sh0=sh0_view,
            shN=shN_view,
            _base=base,
        )

        # Verify that the views share memory with base
        assert np.shares_memory(data.means, base)
        assert np.shares_memory(data.scales, base)
        assert np.shares_memory(data.quats, base)
        assert np.shares_memory(data.opacities, base)
        assert np.shares_memory(data.sh0, base)
        assert np.shares_memory(data.shN, base)

        # Test that modifying base affects the views
        base[0, 0] = 999.0
        assert data.means[0, 0] == 999.0
