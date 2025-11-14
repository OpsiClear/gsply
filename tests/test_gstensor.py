"""Unit tests for GSTensor (PyTorch GPU-accelerated dataclass)."""

import pytest

# Check if PyTorch is available
pytest.importorskip("torch")
import torch  # noqa: E402

from gsply.torch import GSTensor  # noqa: E402


@pytest.fixture
def sample_data_sh0():
    """Create sample Gaussian data (SH0)."""
    n = 1000
    return {
        "means": torch.randn(n, 3, dtype=torch.float32),
        "scales": torch.randn(n, 3, dtype=torch.float32),
        "quats": torch.randn(n, 4, dtype=torch.float32),
        "opacities": torch.rand(n, dtype=torch.float32),
        "sh0": torch.randn(n, 3, dtype=torch.float32),
        "shN": None,
        "masks": torch.ones(n, dtype=torch.bool),
        "_base": None,
    }


@pytest.fixture
def sample_data_sh1():
    """Create sample Gaussian data (SH1)."""
    n = 500
    return {
        "means": torch.randn(n, 3, dtype=torch.float32),
        "scales": torch.randn(n, 3, dtype=torch.float32),
        "quats": torch.randn(n, 4, dtype=torch.float32),
        "opacities": torch.rand(n, dtype=torch.float32),
        "sh0": torch.randn(n, 3, dtype=torch.float32),
        "shN": torch.randn(n, 3, 3, dtype=torch.float32),  # SH1: 3 bands
        "masks": torch.ones(n, dtype=torch.bool),
        "_base": None,
    }


# =============================================================================
# Basic Construction and Properties
# =============================================================================


def test_gstensor_creation_sh0(sample_data_sh0):
    """Test creating GSTensor with SH0 data."""
    gstensor = GSTensor(**sample_data_sh0)

    assert len(gstensor) == 1000
    assert gstensor.means.shape == (1000, 3)
    assert gstensor.scales.shape == (1000, 3)
    assert gstensor.quats.shape == (1000, 4)
    assert gstensor.opacities.shape == (1000,)
    assert gstensor.sh0.shape == (1000, 3)
    assert gstensor.shN is None
    assert gstensor.get_sh_degree() == 0
    assert not gstensor.has_high_order_sh()


def test_gstensor_creation_sh1(sample_data_sh1):
    """Test creating GSTensor with SH1 data."""
    gstensor = GSTensor(**sample_data_sh1)

    assert len(gstensor) == 500
    assert gstensor.shN.shape == (500, 3, 3)  # SH1: 3 bands
    assert gstensor.get_sh_degree() == 1
    assert gstensor.has_high_order_sh()


def test_gstensor_device_dtype_properties(sample_data_sh0):
    """Test device and dtype properties."""
    gstensor = GSTensor(**sample_data_sh0)

    assert gstensor.device == torch.device("cpu")
    assert gstensor.dtype == torch.float32


#  =============================================================================
# Device Management
# =============================================================================


def test_gstensor_cpu(sample_data_sh0):
    """Test cpu() method."""
    gstensor = GSTensor(**sample_data_sh0)
    gstensor_cpu = gstensor.cpu()

    assert gstensor_cpu.device == torch.device("cpu")
    assert torch.allclose(gstensor_cpu.means, gstensor.means)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gstensor_cuda(sample_data_sh0):
    """Test cuda() method."""
    gstensor = GSTensor(**sample_data_sh0)
    gstensor_gpu = gstensor.cuda()

    assert gstensor_gpu.device.type == "cuda"
    assert torch.allclose(gstensor_gpu.means.cpu(), gstensor.means)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gstensor_to_device(sample_data_sh0):
    """Test to(device) method."""
    gstensor = GSTensor(**sample_data_sh0)

    # Move to CUDA
    gstensor_gpu = gstensor.to("cuda")
    assert gstensor_gpu.device.type == "cuda"

    # Move back to CPU
    gstensor_cpu = gstensor_gpu.to("cpu")
    assert gstensor_cpu.device == torch.device("cpu")
    assert torch.allclose(gstensor_cpu.means, gstensor.means)


def test_gstensor_to_dtype(sample_data_sh0):
    """Test to(dtype) method."""
    gstensor = GSTensor(**sample_data_sh0)

    # Convert to float16
    gstensor_half = gstensor.to(dtype=torch.float16)
    assert gstensor_half.dtype == torch.float16

    # Convert to float64
    gstensor_double = gstensor.to(dtype=torch.float64)
    assert gstensor_double.dtype == torch.float64


def test_gstensor_dtype_conversions(sample_data_sh0):
    """Test half(), float(), double() shortcuts."""
    gstensor = GSTensor(**sample_data_sh0)

    # Half precision
    gstensor_half = gstensor.half()
    assert gstensor_half.dtype == torch.float16

    # Single precision
    gstensor_float = gstensor_half.float()
    assert gstensor_float.dtype == torch.float32

    # Double precision
    gstensor_double = gstensor.double()
    assert gstensor_double.dtype == torch.float64


# =============================================================================
# Slicing and Indexing
# =============================================================================


def test_gstensor_single_index(sample_data_sh0):
    """Test single index returns tuple."""
    gstensor = GSTensor(**sample_data_sh0)
    result = gstensor[0]

    assert isinstance(result, tuple)
    assert len(result) == 7  # means, scales, quats, opacities, sh0, shN, masks


def test_gstensor_slice(sample_data_sh0):
    """Test continuous slicing."""
    gstensor = GSTensor(**sample_data_sh0)
    subset = gstensor[100:200]

    assert isinstance(subset, GSTensor)
    assert len(subset) == 100
    assert subset.means.shape == (100, 3)
    assert torch.allclose(subset.means, gstensor.means[100:200])


def test_gstensor_step_slice(sample_data_sh0):
    """Test step slicing."""
    gstensor = GSTensor(**sample_data_sh0)
    subset = gstensor[::10]

    assert isinstance(subset, GSTensor)
    assert len(subset) == 100
    assert torch.allclose(subset.means, gstensor.means[::10])


def test_gstensor_boolean_mask(sample_data_sh0):
    """Test boolean masking."""
    gstensor = GSTensor(**sample_data_sh0)
    mask = gstensor.opacities > 0.5

    subset = gstensor[mask]

    assert isinstance(subset, GSTensor)
    assert len(subset) == mask.sum().item()
    assert torch.allclose(subset.opacities, gstensor.opacities[mask])


def test_gstensor_fancy_indexing(sample_data_sh0):
    """Test fancy indexing with list."""
    gstensor = GSTensor(**sample_data_sh0)
    indices = [0, 10, 20, 30, 40]

    subset = gstensor[indices]

    assert isinstance(subset, GSTensor)
    assert len(subset) == 5
    assert torch.allclose(subset.means, gstensor.means[indices])


def test_gstensor_negative_index(sample_data_sh0):
    """Test negative indexing."""
    gstensor = GSTensor(**sample_data_sh0)
    last = gstensor[-1]
    assert isinstance(last, tuple)


def test_gstensor_get_gaussian(sample_data_sh0):
    """Test get_gaussian() method."""
    gstensor = GSTensor(**sample_data_sh0)
    gaussian = gstensor.get_gaussian(0)

    assert isinstance(gaussian, GSTensor)
    assert len(gaussian) == 1
    assert gaussian.means.shape == (1, 3)


# =============================================================================
# Clone and Copy Operations
# =============================================================================


def test_gstensor_clone(sample_data_sh0):
    """Test clone() creates independent copy."""
    gstensor = GSTensor(**sample_data_sh0)
    gstensor_copy = gstensor.clone()

    # Verify copy is independent
    gstensor_copy.means[0] = 999.0
    assert not torch.allclose(gstensor.means, gstensor_copy.means)


# =============================================================================
# _base Optimization
# =============================================================================


def test_gstensor_consolidate_sh0(sample_data_sh0):
    """Test consolidate() creates _base tensor (SH0)."""
    gstensor = GSTensor(**sample_data_sh0)
    assert gstensor._base is None

    gstensor_consolidated = gstensor.consolidate()

    assert gstensor_consolidated._base is not None
    assert gstensor_consolidated._base.shape == (1000, 14)
    assert torch.allclose(gstensor_consolidated.means, gstensor.means)


def test_gstensor_consolidate_sh1(sample_data_sh1):
    """Test consolidate() creates _base tensor (SH1)."""
    gstensor = GSTensor(**sample_data_sh1)
    gstensor_consolidated = gstensor.consolidate()

    assert gstensor_consolidated._base is not None
    assert gstensor_consolidated._base.shape == (
        500,
        14 + 3 * 3,
    )  # SH1 layout: 23 props (K=3 bands)
    assert torch.allclose(gstensor_consolidated.means, gstensor.means)


def test_gstensor_consolidate_idempotent(sample_data_sh0):
    """Test consolidate() is idempotent."""
    gstensor = GSTensor(**sample_data_sh0)
    gstensor_1 = gstensor.consolidate()
    gstensor_2 = gstensor_1.consolidate()

    # Should return self if already consolidated
    assert gstensor_1 is gstensor_2


def test_gstensor_slice_with_base(sample_data_sh0):
    """Test slicing is faster with _base."""
    gstensor = GSTensor(**sample_data_sh0)
    gstensor_consolidated = gstensor.consolidate()

    # Boolean mask slicing
    mask = gstensor_consolidated.opacities > 0.5
    subset = gstensor_consolidated[mask]

    assert isinstance(subset, GSTensor)
    assert len(subset) == mask.sum().item()


def test_gstensor_clone_with_base(sample_data_sh0):
    """Test clone() with _base is faster."""
    gstensor = GSTensor(**sample_data_sh0)
    gstensor_consolidated = gstensor.consolidate()

    gstensor_copy = gstensor_consolidated.clone()

    assert gstensor_copy._base is not None
    assert torch.allclose(gstensor_copy.means, gstensor.means)


# =============================================================================
# Edge Cases
# =============================================================================


def test_gstensor_empty_slice(sample_data_sh0):
    """Test slicing that returns empty result."""
    gstensor = GSTensor(**sample_data_sh0)
    mask = torch.zeros(1000, dtype=torch.bool)

    subset = gstensor[mask]

    assert isinstance(subset, GSTensor)
    assert len(subset) == 0


def test_gstensor_out_of_bounds_index(sample_data_sh0):
    """Test out of bounds indexing raises error."""
    gstensor = GSTensor(**sample_data_sh0)

    with pytest.raises(IndexError):
        _ = gstensor[10000]


def test_gstensor_invalid_mask_length(sample_data_sh0):
    """Test boolean mask with wrong length raises error."""
    gstensor = GSTensor(**sample_data_sh0)
    mask = torch.ones(500, dtype=torch.bool)  # Wrong length

    with pytest.raises(ValueError):
        _ = gstensor[mask]


# =============================================================================
# Repr and String Representation
# =============================================================================


def test_gstensor_repr(sample_data_sh0):
    """Test __repr__ returns useful string."""
    gstensor = GSTensor(**sample_data_sh0)
    repr_str = repr(gstensor)

    assert "GSTensor" in repr_str
    assert "1000" in repr_str or "1,000" in repr_str  # Gaussians count
    assert "SH degree: 0" in repr_str
    assert "cpu" in repr_str
    assert "float32" in repr_str


# =============================================================================
# Unpack Interface Tests
# =============================================================================


def test_gstensor_unpack_with_shN(sample_data_sh1):
    """Test unpacking with shN included."""
    gstensor = GSTensor(**sample_data_sh1)
    means, scales, quats, opacities, sh0, shN = gstensor.unpack()

    # Verify unpacked tensors match original
    assert torch.equal(means, gstensor.means)
    assert torch.equal(scales, gstensor.scales)
    assert torch.equal(quats, gstensor.quats)
    assert torch.equal(opacities, gstensor.opacities)
    assert torch.equal(sh0, gstensor.sh0)
    assert torch.equal(shN, gstensor.shN)


def test_gstensor_unpack_without_shN(sample_data_sh0):
    """Test unpacking without shN."""
    gstensor = GSTensor(**sample_data_sh0)
    means, scales, quats, opacities, sh0 = gstensor.unpack(include_shN=False)

    # Verify unpacked tensors match original
    assert torch.equal(means, gstensor.means)
    assert torch.equal(scales, gstensor.scales)
    assert torch.equal(quats, gstensor.quats)
    assert torch.equal(opacities, gstensor.opacities)
    assert torch.equal(sh0, gstensor.sh0)


def test_gstensor_unpack_returns_tuple(sample_data_sh0):
    """Test that unpack returns a tuple."""
    gstensor = GSTensor(**sample_data_sh0)
    result = gstensor.unpack()

    assert isinstance(result, tuple)
    assert len(result) == 6

    result_no_shN = gstensor.unpack(include_shN=False)
    assert isinstance(result_no_shN, tuple)
    assert len(result_no_shN) == 5


def test_gstensor_to_dict(sample_data_sh1):
    """Test converting to dictionary."""
    gstensor = GSTensor(**sample_data_sh1)
    props = gstensor.to_dict()

    # Verify dict has expected keys
    assert isinstance(props, dict)
    expected_keys = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    assert set(props.keys()) == expected_keys

    # Verify values match original
    assert torch.equal(props["means"], gstensor.means)
    assert torch.equal(props["scales"], gstensor.scales)
    assert torch.equal(props["quats"], gstensor.quats)
    assert torch.equal(props["opacities"], gstensor.opacities)
    assert torch.equal(props["sh0"], gstensor.sh0)
    assert torch.equal(props["shN"], gstensor.shN)


def test_gstensor_unpack_gpu(sample_data_sh0):
    """Test unpack works on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    gstensor = GSTensor(**sample_data_sh0).to("cuda")
    means, scales, quats, opacities, sh0, shN = gstensor.unpack()

    # Verify unpacked tensors are on GPU
    assert means.device.type == "cuda"
    assert scales.device.type == "cuda"
    assert quats.device.type == "cuda"
    assert opacities.device.type == "cuda"
    assert sh0.device.type == "cuda"


def test_gstensor_to_dict_preserves_device(sample_data_sh0):
    """Test to_dict preserves device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    gstensor = GSTensor(**sample_data_sh0).to("cuda")
    props = gstensor.to_dict()

    # All tensors should be on GPU
    for key, value in props.items():
        if value is not None and isinstance(value, torch.Tensor):
            assert value.device.type == "cuda"
