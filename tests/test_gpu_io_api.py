"""Test GPU I/O API functions (plyread_gpu, plywrite_gpu)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import gsply.spz as spz_module  # noqa: E402
from gsply import GSData, plyread_gpu, plywrite_gpu, read_spz, read_spz_gpu, write_spz  # noqa: E402
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


@pytest.fixture
def sample_spz_gsdata():
    """Create deterministic PLY-format GSData for SPZ GPU tests."""
    rng = np.random.default_rng(123)
    n = 257
    means = rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)
    scales = rng.uniform(-8.0, -2.0, (n, 3)).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.uniform(-3.0, 4.0, n).astype(np.float32)
    sh0 = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    shN = rng.uniform(-0.4, 0.4, (n, 8, 3)).astype(np.float32)  # noqa: N806
    return GSData.from_arrays(means, scales, quats, opacities, sh0, shN, format="ply")


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
    assert hasattr(gsply, "read_spz_gpu")

    # Should be callable
    assert callable(gsply.plyread_gpu)
    assert callable(gsply.plywrite_gpu)
    assert callable(gsply.read_spz_gpu)


@pytest.mark.parametrize("version", [3, 4])
def test_read_spz_gpu_matches_cpu_reader(sample_spz_gsdata, tmp_path, version):
    """read_spz_gpu returns GSTensor values matching the CPU SPZ reader."""
    if version == 4:
        pytest.importorskip("zstandard")

    path = tmp_path / f"scene_v{version}.spz"
    write_spz(path, sample_spz_gsdata, version=version)

    expected = read_spz(path)
    actual = read_spz_gpu(path, device="cpu")

    assert isinstance(actual, GSTensor)
    assert actual.device.type == "cpu"
    assert actual.is_scales_ply
    assert actual.is_opacities_ply
    assert actual.is_sh0_sh
    assert actual.get_sh_degree() == expected.get_sh_degree()

    np.testing.assert_allclose(actual.means.numpy(), expected.means, atol=1e-6)
    np.testing.assert_allclose(actual.scales.numpy(), expected.scales, atol=1e-6)
    np.testing.assert_allclose(actual.opacities.numpy(), expected.opacities, atol=1e-5)
    np.testing.assert_allclose(actual.sh0.numpy(), expected.sh0, atol=1e-6)
    np.testing.assert_allclose(actual.shN.numpy(), expected.shN, atol=1e-6)
    np.testing.assert_allclose(actual.quats.numpy(), expected.quats, atol=1e-6)


def test_read_spz_gpu_uses_packed_tensor_decode(sample_spz_gsdata, tmp_path, monkeypatch):
    """read_spz_gpu must not route through the CPU GSData decode path."""
    path = tmp_path / "scene.spz"
    write_spz(path, sample_spz_gsdata, version=3)

    def fail_cpu_decode(*_args, **_kwargs):
        raise AssertionError("read_spz_gpu should not materialize CPU GSData first")

    monkeypatch.setattr(spz_module, "_decode_to_gsdata", fail_cpu_decode)
    actual = read_spz_gpu(path, device="cpu")

    assert isinstance(actual, GSTensor)
    assert len(actual) == len(sample_spz_gsdata)
