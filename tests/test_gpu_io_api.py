"""Test GPU I/O API functions (plyread_gpu, plywrite_gpu)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from gsply import GSData, plyread_gpu, plywrite_gpu  # noqa: E402
from gsply.torch import GSTensor  # noqa: E402


@pytest.fixture
def sample_gsdata_sh0():
    """Create sample GSData with SH0."""
    n = 512  # Use chunk-aligned size for easier testing
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.randn(n, 3).astype(np.float32)
    quats = np.random.randn(n, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)  # Normalize
    opacities = np.random.randn(n).astype(np.float32)
    sh0 = np.random.randn(n, 3).astype(np.float32) * 0.1  # Keep small for SH0
    shN = np.zeros((n, 0, 3), dtype=np.float32)

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        masks=np.ones(n, dtype=bool),
        _base=None,
    )


def test_plyread_gpu_api(sample_gsdata_sh0):
    """Test plyread_gpu API matches plyread style."""
    # Compress sample data
    from gsply import compress_to_bytes

    compressed_bytes = compress_to_bytes(sample_gsdata_sh0)

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp.write(compressed_bytes)
        tmp_path = tmp.name

    try:
        # Test API: plyread_gpu (should match plyread style)
        gstensor = plyread_gpu(tmp_path, device="cpu")  # Use CPU for testing

        assert isinstance(gstensor, GSTensor)
        assert len(gstensor) == len(sample_gsdata_sh0)
        assert gstensor.means.shape == sample_gsdata_sh0.means.shape
        assert gstensor.scales.shape == sample_gsdata_sh0.scales.shape
        assert gstensor.quats.shape == sample_gsdata_sh0.quats.shape
        assert gstensor.opacities.shape == sample_gsdata_sh0.opacities.shape
        assert gstensor.sh0.shape == sample_gsdata_sh0.sh0.shape

    finally:
        Path(tmp_path).unlink()


def test_plywrite_gpu_api(sample_gsdata_sh0):
    """Test plywrite_gpu API matches plywrite style."""
    # Convert to GSTensor
    gstensor = GSTensor.from_gsdata(sample_gsdata_sh0, device="cpu")

    # Test API: plywrite_gpu (should match plywrite style)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = tmp.name

    try:
        plywrite_gpu(tmp_path, gstensor)

        # Verify file was created
        assert Path(tmp_path).exists()
        assert Path(tmp_path).stat().st_size > 0

        # Verify we can read it back
        gstensor_read = plyread_gpu(tmp_path, device="cpu")
        assert len(gstensor_read) == len(sample_gsdata_sh0)

    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def test_plywrite_gpu_requires_compressed(sample_gsdata_sh0):
    """Test that plywrite_gpu requires compressed=True (or defaults to compressed)."""
    gstensor = GSTensor.from_gsdata(sample_gsdata_sh0, device="cpu")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = tmp.name

    try:
        # Should work with compressed=True (default)
        plywrite_gpu(tmp_path, gstensor, compressed=True)

        # Should fail with compressed=False
        with pytest.raises(ValueError, match="only supports compressed format"):
            plywrite_gpu(tmp_path, gstensor, compressed=False)

    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def test_gpu_io_roundtrip(sample_gsdata_sh0):
    """Test round-trip: plyread_gpu -> plywrite_gpu -> plyread_gpu."""
    # Compress sample data
    from gsply import compress_to_bytes

    compressed_bytes = compress_to_bytes(sample_gsdata_sh0)

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp.write(compressed_bytes)
        tmp_path = tmp.name

    try:
        # Read with GPU API
        gstensor1 = plyread_gpu(tmp_path, device="cpu")

        # Write with GPU API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp2:
            tmp_path2 = tmp2.name

        try:
            plywrite_gpu(tmp_path2, gstensor1)

            # Read back
            gstensor2 = plyread_gpu(tmp_path2, device="cpu")

            # Verify equivalence (within quantization tolerance)
            # Note: Roundtrip compression/decompression accumulates quantization errors
            np.testing.assert_allclose(
                gstensor1.means.cpu().numpy(),
                gstensor2.means.cpu().numpy(),
                rtol=5e-3,
                atol=1e-2,
            )
            np.testing.assert_allclose(
                gstensor1.scales.cpu().numpy(),
                gstensor2.scales.cpu().numpy(),
                rtol=5e-3,
                atol=1e-2,
            )
            np.testing.assert_allclose(
                gstensor1.sh0.cpu().numpy(),
                gstensor2.sh0.cpu().numpy(),
                rtol=5e-3,
                atol=1e-2,
            )

        finally:
            if Path(tmp_path2).exists():
                Path(tmp_path2).unlink()

    finally:
        Path(tmp_path).unlink()


def test_gpu_io_lazy_import():
    """Test that plyread_gpu and plywrite_gpu are available via lazy import."""
    import gsply

    # Should be available via lazy import
    assert hasattr(gsply, "plyread_gpu")
    assert hasattr(gsply, "plywrite_gpu")

    # Should be callable
    assert callable(gsply.plyread_gpu)
    assert callable(gsply.plywrite_gpu)
