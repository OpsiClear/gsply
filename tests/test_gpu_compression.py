"""Unit tests for GPU compression/decompression functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from gsply import GSData, compress_to_bytes, decompress_from_bytes, plyread  # noqa: E402
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
def sample_gsdata_sh3():
    """Create sample GSData with SH3."""
    n = 512
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.randn(n, 3).astype(np.float32)
    quats = np.random.randn(n, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.randn(n).astype(np.float32)
    sh0 = np.random.randn(n, 3).astype(np.float32) * 0.1
    shN = np.random.randn(n, 15, 3).astype(np.float32) * 0.05  # SH3: 15 bands

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


# =============================================================================
# GPU Decompression Tests
# =============================================================================


def test_gpu_decompression_sh0(sample_gsdata_sh0):
    """Test GPU decompression matches CPU decompression for SH0."""
    # Compress on CPU
    compressed_bytes = compress_to_bytes(sample_gsdata_sh0)

    # Decompress on CPU
    cpu_data = decompress_from_bytes(compressed_bytes)

    # Decompress on GPU
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp.write(compressed_bytes)
        tmp_path = tmp.name

    try:
        gpu_tensor = GSTensor.from_compressed(tmp_path, device="cpu")  # Use CPU for testing

        # Compare results (should match exactly - both CPU and GPU use same decompression)
        np.testing.assert_allclose(gpu_tensor.means.numpy(), cpu_data.means, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(gpu_tensor.scales.numpy(), cpu_data.scales, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(gpu_tensor.quats.numpy(), cpu_data.quats, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            gpu_tensor.opacities.numpy(), cpu_data.opacities, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(gpu_tensor.sh0.numpy(), cpu_data.sh0, rtol=1e-5, atol=1e-6)

    finally:
        Path(tmp_path).unlink()


def test_gpu_decompression_sh3(sample_gsdata_sh3):
    """Test GPU decompression matches CPU decompression for SH3."""
    compressed_bytes = compress_to_bytes(sample_gsdata_sh3)

    cpu_data = decompress_from_bytes(compressed_bytes)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp.write(compressed_bytes)
        tmp_path = tmp.name

    try:
        gpu_tensor = GSTensor.from_compressed(tmp_path, device="cpu")

        # Compare main attributes
        np.testing.assert_allclose(gpu_tensor.means.numpy(), cpu_data.means, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(gpu_tensor.scales.numpy(), cpu_data.scales, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(gpu_tensor.quats.numpy(), cpu_data.quats, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(
            gpu_tensor.opacities.numpy(), cpu_data.opacities, rtol=1e-2, atol=1e-2
        )
        np.testing.assert_allclose(gpu_tensor.sh0.numpy(), cpu_data.sh0, rtol=1e-3, atol=1e-3)

        # Compare SH coefficients (larger tolerance due to 8-bit quantization)
        np.testing.assert_allclose(gpu_tensor.shN.numpy(), cpu_data.shN, rtol=2e-2, atol=1e-1)

    finally:
        Path(tmp_path).unlink()


# =============================================================================
# GPU Compression Tests
# =============================================================================


def test_gpu_compression_sh0(sample_gsdata_sh0):
    """Test GPU compression produces valid compressed files."""
    # Upload to GPU
    gstensor = GSTensor.from_gsdata(sample_gsdata_sh0, device="cpu")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = tmp.name

    try:
        # Compress on GPU and save
        gstensor.save_compressed(tmp_path)

        # Verify file was created
        assert Path(tmp_path).exists()
        assert Path(tmp_path).stat().st_size > 0

        # Read back and verify (using CPU decompression)
        data_restored = plyread(tmp_path)

        # Should match original within quantization tolerance (looser for compression artifacts)
        np.testing.assert_allclose(
            data_restored.means, sample_gsdata_sh0.means, rtol=5e-3, atol=1e-2
        )
        np.testing.assert_allclose(
            data_restored.scales, sample_gsdata_sh0.scales, rtol=5e-3, atol=1e-2
        )
        # Quaternions can be flipped (q and -q represent same rotation)
        # Check if q1 ≈ q2 OR q1 ≈ -q2
        quat_diff1 = np.abs(data_restored.quats - sample_gsdata_sh0.quats)
        quat_diff2 = np.abs(data_restored.quats + sample_gsdata_sh0.quats)
        quat_match = np.minimum(quat_diff1, quat_diff2)
        assert np.all(quat_match < 0.05), f"Max quat mismatch: {quat_match.max()}"

    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def test_gpu_compression_sh3(sample_gsdata_sh3):
    """Test GPU compression with SH3."""
    gstensor = GSTensor.from_gsdata(sample_gsdata_sh3, device="cpu")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = tmp.name

    try:
        gstensor.save_compressed(tmp_path)

        data_restored = plyread(tmp_path)

        # Verify SH degree preserved
        assert data_restored.get_sh_degree() == 3

        # Check SH coefficients (larger tolerance for 8-bit quantization)
        np.testing.assert_allclose(data_restored.shN, sample_gsdata_sh3.shN, rtol=2e-2, atol=1e-1)

    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


# =============================================================================
# Round-Trip Tests (CPU -> GPU -> CPU)
# =============================================================================


def test_roundtrip_cpu_gpu_cpu_sh0(sample_gsdata_sh0):
    """Test data consistency through CPU -> GPU compression -> CPU decompression."""
    # CPU compress
    compressed_cpu = compress_to_bytes(sample_gsdata_sh0)

    # Upload to GPU and compress
    gstensor = GSTensor.from_gsdata(sample_gsdata_sh0, device="cpu")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = tmp.name

    try:
        gstensor.save_compressed(tmp_path)

        # Read compressed file
        with open(tmp_path, "rb") as f:
            compressed_gpu = f.read()

        # Both should decompress to similar results
        # Note: GPU and CPU compression may have slight quantization differences
        data_cpu = decompress_from_bytes(compressed_cpu)
        data_gpu = decompress_from_bytes(compressed_gpu)

        np.testing.assert_allclose(data_cpu.means, data_gpu.means, rtol=5e-3, atol=1e-2)
        np.testing.assert_allclose(data_cpu.scales, data_gpu.scales, rtol=5e-3, atol=1e-2)
        np.testing.assert_allclose(data_cpu.quats, data_gpu.quats, rtol=5e-3, atol=1e-2)

    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def test_roundtrip_gpu_decompress_compress_sh0(sample_gsdata_sh0):
    """Test GPU decompress -> GPU compress produces consistent results."""
    # Compress on CPU first
    compressed_bytes = compress_to_bytes(sample_gsdata_sh0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp1:
        tmp1.write(compressed_bytes)
        tmp1_path = tmp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp2:
        tmp2_path = tmp2.name

    try:
        # GPU decompress
        gstensor = GSTensor.from_compressed(tmp1_path, device="cpu")

        # GPU compress
        gstensor.save_compressed(tmp2_path)

        # Read both back
        data1 = plyread(tmp1_path)
        data2 = plyread(tmp2_path)

        # Should be very similar (allowing for double quantization)
        np.testing.assert_allclose(data1.means, data2.means, rtol=5e-3, atol=1e-2)
        np.testing.assert_allclose(data1.scales, data2.scales, rtol=5e-3, atol=1e-2)

    finally:
        if Path(tmp1_path).exists():
            Path(tmp1_path).unlink()
        if Path(tmp2_path).exists():
            Path(tmp2_path).unlink()


# =============================================================================
# Edge Cases
# =============================================================================


def test_gpu_compression_non_chunk_aligned():
    """Test GPU compression with non-256-aligned vertex count."""
    n = 500  # Not divisible by 256
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.randn(n, 3).astype(np.float32)
    quats = np.random.randn(n, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.randn(n).astype(np.float32)
    sh0 = np.random.randn(n, 3).astype(np.float32) * 0.1
    shN = np.zeros((n, 0, 3), dtype=np.float32)

    data = GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        masks=np.ones(n, dtype=bool),
        _base=None,
    )

    gstensor = GSTensor.from_gsdata(data, device="cpu")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = tmp.name

    try:
        gstensor.save_compressed(tmp_path)

        data_restored = plyread(tmp_path)

        # Verify correct number of vertices
        assert len(data_restored) == n

        np.testing.assert_allclose(data_restored.means, means, rtol=5e-3, atol=1e-2)

    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def test_gpu_compression_small_dataset():
    """Test GPU compression with very small dataset (< 256 vertices)."""
    n = 100
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.randn(n, 3).astype(np.float32)
    quats = np.random.randn(n, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.randn(n).astype(np.float32)
    sh0 = np.random.randn(n, 3).astype(np.float32) * 0.1
    shN = np.zeros((n, 0, 3), dtype=np.float32)

    data = GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        masks=np.ones(n, dtype=bool),
        _base=None,
    )

    gstensor = GSTensor.from_gsdata(data, device="cpu")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply_compressed") as tmp:
        tmp_path = tmp.name

    try:
        gstensor.save_compressed(tmp_path)

        data_restored = plyread(tmp_path)
        assert len(data_restored) == n

    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
