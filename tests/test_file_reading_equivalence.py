"""Test equivalence between optimized file reading and CPU decompression."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from gsply import GSData  # noqa: E402
from gsply.reader import read_compressed  # noqa: E402
from gsply.torch.compression import read_compressed_gpu  # noqa: E402
from gsply.writer import compress_to_bytes  # noqa: E402


@pytest.fixture
def sample_gsdata_sh0():
    """Create sample GSData with SH0."""
    n = 512
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.randn(n, 3).astype(np.float32)
    quats = np.random.randn(n, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.randn(n).astype(np.float32)
    sh0 = np.random.randn(n, 3).astype(np.float32) * 0.1
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
def sample_gsdata_sh3():
    """Create sample GSData with SH3."""
    n = 512
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.randn(n, 3).astype(np.float32)
    quats = np.random.randn(n, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.randn(n).astype(np.float32)
    sh0 = np.random.randn(n, 3).astype(np.float32) * 0.1
    shN = np.random.randn(n, 15, 3).astype(np.float32) * 0.05

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


def test_file_reading_equivalence_sh0(sample_gsdata_sh0):
    """Verify optimized GPU file reading produces same results as CPU decompression."""
    # Compress data using CPU
    compressed_bytes = compress_to_bytes(sample_gsdata_sh0)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(compressed_bytes)

    try:
        # Method 1: CPU decompression (original)
        data_cpu = read_compressed(tmp_path)

        # Method 2: GPU decompression with optimized file reading
        gstensor = read_compressed_gpu(tmp_path, device="cpu")  # Use CPU for equivalence test
        data_gpu = gstensor.to_gsdata()

        # Verify equivalence
        np.testing.assert_allclose(data_cpu.means, data_gpu.means, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(data_cpu.scales, data_gpu.scales, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(data_cpu.quats, data_gpu.quats, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(data_cpu.opacities, data_gpu.opacities, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(data_cpu.sh0, data_gpu.sh0, rtol=1e-6, atol=1e-6)

        # Verify SHN if present
        if data_cpu.shN is not None and data_gpu.shN is not None:
            np.testing.assert_allclose(data_cpu.shN, data_gpu.shN, rtol=1e-6, atol=1e-6)
        elif data_cpu.shN is None and data_gpu.shN is None:
            pass  # Both None, OK
        else:
            pytest.fail("SHN mismatch: one is None, other is not")

        # Verify masks (GPU decompression may not preserve masks, which is OK)
        # Both methods should produce same data regardless of mask handling

    finally:
        tmp_path.unlink(missing_ok=True)


def test_file_reading_equivalence_sh3(sample_gsdata_sh3):
    """Verify optimized GPU file reading produces same results as CPU decompression (SH3)."""
    # Compress data using CPU
    compressed_bytes = compress_to_bytes(sample_gsdata_sh3)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(compressed_bytes)

    try:
        # Method 1: CPU decompression (original)
        data_cpu = read_compressed(tmp_path)

        # Method 2: GPU decompression with optimized file reading
        gstensor = read_compressed_gpu(tmp_path, device="cpu")  # Use CPU for equivalence test
        data_gpu = gstensor.to_gsdata()

        # Verify equivalence
        np.testing.assert_allclose(data_cpu.means, data_gpu.means, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(data_cpu.scales, data_gpu.scales, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(data_cpu.quats, data_gpu.quats, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(data_cpu.opacities, data_gpu.opacities, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(data_cpu.sh0, data_gpu.sh0, rtol=1e-6, atol=1e-6)

        # Verify SHN
        assert data_cpu.shN is not None, "CPU data should have SHN"
        assert data_gpu.shN is not None, "GPU data should have SHN"
        np.testing.assert_allclose(data_cpu.shN, data_gpu.shN, rtol=1e-6, atol=1e-6)

    finally:
        tmp_path.unlink(missing_ok=True)


def test_file_reading_equivalence_roundtrip(sample_gsdata_sh0):
    """Test roundtrip: CPU compress -> optimized GPU read -> verify equivalence.

    Note: Compression introduces quantization errors, so we use relaxed tolerances.
    """
    # Compress using CPU
    compressed_bytes = compress_to_bytes(sample_gsdata_sh0)

    # Write to file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(compressed_bytes)

    try:
        # Read back using optimized GPU method
        gstensor = read_compressed_gpu(tmp_path, device="cpu")
        data_read = gstensor.to_gsdata()

        # Compare with original (compression introduces quantization, use relaxed tolerances)
        np.testing.assert_allclose(sample_gsdata_sh0.means, data_read.means, rtol=5e-3, atol=1e-2)
        np.testing.assert_allclose(sample_gsdata_sh0.scales, data_read.scales, rtol=5e-3, atol=1e-2)

        # Quaternions: handle q vs -q equivalence
        quat_diff1 = np.abs(sample_gsdata_sh0.quats - data_read.quats)
        quat_diff2 = np.abs(sample_gsdata_sh0.quats + data_read.quats)
        quat_diff = np.minimum(quat_diff1, quat_diff2)
        assert np.all(quat_diff < 1e-2), "Quaternion mismatch exceeds tolerance"

        # Opacities: stored in logit space, quantization errors are larger
        np.testing.assert_allclose(
            sample_gsdata_sh0.opacities, data_read.opacities, rtol=1e-1, atol=1e-1
        )
        np.testing.assert_allclose(sample_gsdata_sh0.sh0, data_read.sh0, rtol=5e-3, atol=1e-2)

    finally:
        tmp_path.unlink(missing_ok=True)


def test_file_reading_equivalence_non_chunk_aligned():
    """Test equivalence with non-chunk-aligned data."""
    from gsply.gsdata import GSData

    # Create data that's not aligned to 256 (chunk size)
    num_gaussians = 500  # Not a multiple of 256

    means = np.random.randn(num_gaussians, 3).astype(np.float32)
    scales = np.random.randn(num_gaussians, 3).astype(np.float32)
    quats = np.random.randn(num_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.randn(num_gaussians).astype(np.float32)
    sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

    data = GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=None,
        masks=None,
        mask_names=None,
        _base=None,
    )

    # Compress
    compressed_bytes = compress_to_bytes(data)

    # Write to file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(compressed_bytes)

    try:
        # Read using CPU
        data_cpu = read_compressed(tmp_path)

        # Read using optimized GPU method
        gstensor = read_compressed_gpu(tmp_path, device="cpu")
        data_gpu = gstensor.to_gsdata()

        # Verify equivalence
        np.testing.assert_allclose(data_cpu.means, data_gpu.means, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(data_cpu.scales, data_gpu.scales, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(data_cpu.quats, data_gpu.quats, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(data_cpu.opacities, data_gpu.opacities, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(data_cpu.sh0, data_gpu.sh0, rtol=1e-5, atol=1e-5)

    finally:
        tmp_path.unlink(missing_ok=True)
