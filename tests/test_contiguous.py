"""Tests for GSData contiguity methods."""

import numpy as np
import pytest

from gsply import GSData


@pytest.fixture
def data_noncontiguous():
    """Create non-contiguous GSData (from _base)."""
    n = 1000
    base_array = np.random.randn(n, 14).astype(np.float32)
    return GSData._recreate_from_base(base_array)


@pytest.fixture
def data_contiguous():
    """Create contiguous GSData (direct construction)."""
    n = 1000
    return GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )


def test_is_contiguous_true(data_contiguous):
    """Test is_contiguous returns True for contiguous data."""
    assert data_contiguous.is_contiguous() is True


def test_is_contiguous_false(data_noncontiguous):
    """Test is_contiguous returns False for non-contiguous data."""
    assert data_noncontiguous.is_contiguous() is False


def test_make_contiguous_inplace(data_noncontiguous):
    """Test make_contiguous with inplace=True."""
    # Verify starts non-contiguous
    assert data_noncontiguous.is_contiguous() is False
    assert data_noncontiguous._base is not None

    # Make contiguous
    result = data_noncontiguous.make_contiguous(inplace=True)

    # Should return self
    assert result is data_noncontiguous

    # Should now be contiguous
    assert data_noncontiguous.is_contiguous() is True
    assert data_noncontiguous._base is None

    # All arrays should be contiguous
    assert data_noncontiguous.means.flags["C_CONTIGUOUS"]
    assert data_noncontiguous.scales.flags["C_CONTIGUOUS"]
    assert data_noncontiguous.quats.flags["C_CONTIGUOUS"]
    assert data_noncontiguous.opacities.flags["C_CONTIGUOUS"]
    assert data_noncontiguous.sh0.flags["C_CONTIGUOUS"]


def test_make_contiguous_not_inplace(data_noncontiguous):
    """Test make_contiguous with inplace=False."""
    original_means = data_noncontiguous.means.copy()

    # Make contiguous (new object)
    result = data_noncontiguous.make_contiguous(inplace=False)

    # Should return new object
    assert result is not data_noncontiguous

    # Original should be unchanged
    assert data_noncontiguous.is_contiguous() is False
    assert data_noncontiguous._base is not None

    # Result should be contiguous
    assert result.is_contiguous() is True
    assert result._base is None

    # Data should match
    np.testing.assert_array_equal(result.means, original_means)


def test_make_contiguous_already_contiguous(data_contiguous):
    """Test make_contiguous on already contiguous data (no-op)."""
    original_means_id = id(data_contiguous.means)

    result = data_contiguous.make_contiguous(inplace=True)

    # Should return self
    assert result is data_contiguous

    # Should still be contiguous
    assert data_contiguous.is_contiguous() is True

    # Arrays should not be copied (already contiguous)
    assert id(data_contiguous.means) == original_means_id


def test_make_contiguous_preserves_data(data_noncontiguous):
    """Test that make_contiguous preserves all data correctly."""
    # Save original data
    original_means = data_noncontiguous.means.copy()
    original_scales = data_noncontiguous.scales.copy()
    original_quats = data_noncontiguous.quats.copy()
    original_opacities = data_noncontiguous.opacities.copy()
    original_sh0 = data_noncontiguous.sh0.copy()

    # Make contiguous
    data_noncontiguous.make_contiguous()

    # Verify data unchanged
    np.testing.assert_array_equal(data_noncontiguous.means, original_means)
    np.testing.assert_array_equal(data_noncontiguous.scales, original_scales)
    np.testing.assert_array_equal(data_noncontiguous.quats, original_quats)
    np.testing.assert_array_equal(data_noncontiguous.opacities, original_opacities)
    np.testing.assert_array_equal(data_noncontiguous.sh0, original_sh0)


def test_make_contiguous_with_masks(data_noncontiguous):
    """Test make_contiguous with mask layers."""
    # Add mask layers
    data_noncontiguous.add_mask_layer("test1", np.random.rand(len(data_noncontiguous)) > 0.5)
    data_noncontiguous.add_mask_layer("test2", np.random.rand(len(data_noncontiguous)) > 0.5)

    # Make contiguous
    data_noncontiguous.make_contiguous()

    # Should be contiguous including masks
    assert data_noncontiguous.is_contiguous() is True
    assert data_noncontiguous.masks.flags["C_CONTIGUOUS"]


def test_make_contiguous_with_shN():
    """Test make_contiguous with higher-order SH coefficients."""
    n = 1000
    base_array = np.random.randn(n, 23).astype(np.float32)  # SH1: 23 properties
    data = GSData._recreate_from_base(base_array)

    # Should start non-contiguous
    assert data.is_contiguous() is False
    assert not data.shN.flags["C_CONTIGUOUS"]

    # Make contiguous
    data.make_contiguous()

    # Should be contiguous including shN
    assert data.is_contiguous() is True
    assert data.shN.flags["C_CONTIGUOUS"]


def test_contiguous_performance_benefit():
    """Verify that contiguous arrays are actually faster."""
    import time

    n = 100_000

    # Create non-contiguous data
    base_array = np.random.randn(n, 14).astype(np.float32)
    data_noncontig = GSData._recreate_from_base(base_array)

    # Create contiguous version
    data_contig = data_noncontig.make_contiguous(inplace=False)

    # Benchmark sum() operation (should be 6x faster on contiguous)
    # Non-contiguous
    times_noncontig = []
    for _ in range(20):
        start = time.perf_counter()
        _ = data_noncontig.means.sum()
        end = time.perf_counter()
        times_noncontig.append(end - start)

    # Contiguous
    times_contig = []
    for _ in range(20):
        start = time.perf_counter()
        _ = data_contig.means.sum()
        end = time.perf_counter()
        times_contig.append(end - start)

    noncontig_time = np.mean(times_noncontig)
    contig_time = np.mean(times_contig)
    speedup = noncontig_time / contig_time

    # Contiguous should be at least 2x faster (conservative check)
    assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"
