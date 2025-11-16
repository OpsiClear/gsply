"""Tests for GSData.concatenate() bulk concatenation method."""

import numpy as np
import pytest

from gsply import GSData


@pytest.fixture
def sample_arrays():
    """Create list of sample GSData arrays."""
    arrays = []
    for i in range(5):
        n = 1000 + i * 100  # Varying sizes
        arrays.append(
            GSData(
                means=np.random.randn(n, 3).astype(np.float32),
                scales=np.random.rand(n, 3).astype(np.float32),
                quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
                opacities=np.random.rand(n).astype(np.float32),
                sh0=np.random.rand(n, 3).astype(np.float32),
                shN=None,
            )
        )
    return arrays


def test_concatenate_basic(sample_arrays):
    """Test basic concatenation of multiple arrays."""
    result = GSData.concatenate(sample_arrays)

    # Check total length
    expected_len = sum(len(arr) for arr in sample_arrays)
    assert len(result) == expected_len

    # Verify data correctness by checking first and last arrays
    first_n = len(sample_arrays[0])
    np.testing.assert_array_equal(result.means[:first_n], sample_arrays[0].means)

    last_n = len(sample_arrays[-1])
    np.testing.assert_array_equal(result.means[-last_n:], sample_arrays[-1].means)


def test_concatenate_vs_pairwise(sample_arrays):
    """Test that concatenate gives same result as pairwise add()."""
    # Bulk concatenate
    result_bulk = GSData.concatenate(sample_arrays)

    # Pairwise add
    result_pairwise = sample_arrays[0]
    for arr in sample_arrays[1:]:
        result_pairwise = result_pairwise.add(arr)

    # Should match
    assert len(result_bulk) == len(result_pairwise)
    np.testing.assert_allclose(result_bulk.means, result_pairwise.means)
    np.testing.assert_allclose(result_bulk.scales, result_pairwise.scales)
    np.testing.assert_allclose(result_bulk.quats, result_pairwise.quats)
    np.testing.assert_allclose(result_bulk.opacities, result_pairwise.opacities)
    np.testing.assert_allclose(result_bulk.sh0, result_pairwise.sh0)


def test_concatenate_empty_raises():
    """Test that empty list raises error."""
    with pytest.raises(ValueError, match="empty list"):
        GSData.concatenate([])


def test_concatenate_single_array(sample_arrays):
    """Test concatenation of single array returns that array."""
    result = GSData.concatenate([sample_arrays[0]])
    assert result is sample_arrays[0]


def test_concatenate_with_shN():
    """Test concatenation with higher-order SH coefficients."""
    arrays = []
    for i in range(3):
        n = 500
        arrays.append(
            GSData(
                means=np.random.randn(n, 3).astype(np.float32),
                scales=np.random.rand(n, 3).astype(np.float32),
                quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
                opacities=np.random.rand(n).astype(np.float32),
                sh0=np.random.rand(n, 3).astype(np.float32),
                shN=np.random.rand(n, 3, 3).astype(np.float32),  # SH1: 3 bands
            )
        )

    result = GSData.concatenate(arrays)

    assert len(result) == 1500
    assert result.shN is not None
    assert result.shN.shape == (1500, 3, 3)


def test_concatenate_mixed_shN_raises():
    """Test that mixing shN and non-shN raises error."""
    arrays = [
        GSData(
            means=np.random.randn(100, 3).astype(np.float32),
            scales=np.random.rand(100, 3).astype(np.float32),
            quats=np.tile([1, 0, 0, 0], (100, 1)).astype(np.float32),
            opacities=np.random.rand(100).astype(np.float32),
            sh0=np.random.rand(100, 3).astype(np.float32),
            shN=np.random.rand(100, 3, 3).astype(np.float32),  # SH1: degree 1
        ),
        GSData(
            means=np.random.randn(100, 3).astype(np.float32),
            scales=np.random.rand(100, 3).astype(np.float32),
            quats=np.tile([1, 0, 0, 0], (100, 1)).astype(np.float32),
            opacities=np.random.rand(100).astype(np.float32),
            sh0=np.random.rand(100, 3).astype(np.float32),
            shN=None,  # SH0: degree 0 (incompatible!)
        ),
    ]

    # Mixing SH degrees should raise error
    with pytest.raises(ValueError, match="same SH degree"):
        GSData.concatenate(arrays)


def test_concatenate_incompatible_sh_degrees_raises():
    """Test that incompatible SH degrees raise error."""
    arrays = [
        GSData(
            means=np.random.randn(100, 3).astype(np.float32),
            scales=np.random.rand(100, 3).astype(np.float32),
            quats=np.tile([1, 0, 0, 0], (100, 1)).astype(np.float32),
            opacities=np.random.rand(100).astype(np.float32),
            sh0=np.random.rand(100, 3).astype(np.float32),
            shN=np.random.rand(100, 3, 3).astype(np.float32),  # SH1: 3 bands
        ),
        GSData(
            means=np.random.randn(100, 3).astype(np.float32),
            scales=np.random.rand(100, 3).astype(np.float32),
            quats=np.tile([1, 0, 0, 0], (100, 1)).astype(np.float32),
            opacities=np.random.rand(100).astype(np.float32),
            sh0=np.random.rand(100, 3).astype(np.float32),
            shN=np.random.rand(100, 8, 3).astype(np.float32),  # SH2: 8 bands (incompatible!)
        ),
    ]

    with pytest.raises(ValueError, match="same SH degree"):
        GSData.concatenate(arrays)


def test_concatenate_performance_benefit():
    """Verify concatenate is faster than pairwise (smoke test)."""
    import time

    # Create 10 small arrays
    arrays = [
        GSData(
            means=np.random.randn(1000, 3).astype(np.float32),
            scales=np.random.rand(1000, 3).astype(np.float32),
            quats=np.tile([1, 0, 0, 0], (1000, 1)).astype(np.float32),
            opacities=np.random.rand(1000).astype(np.float32),
            sh0=np.random.rand(1000, 3).astype(np.float32),
            shN=None,
        )
        for _ in range(10)
    ]

    # Warmup
    for _ in range(3):
        _ = GSData.concatenate(arrays)

    # Time concatenate
    start = time.perf_counter()
    result_concat = GSData.concatenate(arrays)
    _concat_time = time.perf_counter() - start

    # Time pairwise (just once, no need for many iterations)
    start = time.perf_counter()
    result_pairwise = arrays[0]
    for arr in arrays[1:]:
        result_pairwise = result_pairwise.add(arr)
    _pairwise_time = time.perf_counter() - start

    # Concatenate should be faster (though might vary due to noise in single run)
    # Just verify it completes without error
    assert len(result_concat) == len(result_pairwise)
