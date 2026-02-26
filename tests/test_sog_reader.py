"""Tests for gsply.sog_reader module - both v2 (codebook) and v3 (linear) formats."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gsply.sog_reader import (
    _decode_colors_linear_jit,
    _decode_scales_linear_jit,
    _decode_shn_linear_jit,
    sogread,
)

# imagecodecs is optional (needed for WebP encoding in tests)
try:
    import imagecodecs
except ImportError:
    imagecodecs = None

requires_imagecodecs = pytest.mark.skipif(imagecodecs is None, reason="imagecodecs not installed")


def _make_rgba_webp(height: int, width: int, rgba: np.ndarray) -> bytes:
    """Encode an RGBA array (H, W, 4) to lossless WebP bytes."""
    assert rgba.shape == (height, width, 4) and rgba.dtype == np.uint8
    return imagecodecs.webp_encode(rgba, lossless=True)


def _build_v3_sog_folder(
    tmp_dir: Path,
    count: int = 16,
    *,
    include_shn: bool = True,
) -> Path:
    """Build a minimal v3 SOG folder with synthetic WebP images.

    Creates a deterministic dataset so we can verify the dequantized values.
    """
    rng = np.random.RandomState(42)

    # Texture dimensions: smallest square that fits count pixels
    side = int(np.ceil(np.sqrt(count)))

    # --- means ---
    means_lo = rng.randint(0, 256, (side, side, 4), dtype=np.uint8)
    means_hi = rng.randint(0, 256, (side, side, 4), dtype=np.uint8)

    means_mins = [-1.0, -1.0, -1.0]
    means_maxs = [1.0, 1.0, 1.0]

    # --- scales ---
    scales_img = rng.randint(0, 256, (side, side, 4), dtype=np.uint8)
    scales_mins = [-10.0, -10.0, -10.0]
    scales_maxs = [-2.0, -2.0, -2.0]

    # --- quats ---
    quats_img = np.zeros((side, side, 4), dtype=np.uint8)
    # Set alpha to valid tag (252-255) for valid quaternions
    quats_img[:, :, 3] = 254
    quats_img[:, :, :3] = rng.randint(0, 256, (side, side, 3), dtype=np.uint8)

    # --- sh0 ---
    sh0_img = rng.randint(0, 256, (side, side, 4), dtype=np.uint8)
    sh0_mins = [-2.0, -2.0, -2.0, -4.0]
    sh0_maxs = [2.0, 2.0, 2.0, 6.0]

    # --- meta.json ---
    meta = {
        "means": {
            "shape": [count, 3],
            "dtype": "float32",
            "mins": means_mins,
            "maxs": means_maxs,
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {
            "shape": [count, 3],
            "dtype": "float32",
            "mins": scales_mins,
            "maxs": scales_maxs,
            "files": ["scales.webp"],
        },
        "quats": {
            "shape": [count, 4],
            "dtype": "uint8",
            "encoding": "quaternion_packed",
            "files": ["quats.webp"],
        },
        "sh0": {
            "shape": [count, 1, 4],
            "dtype": "float32",
            "mins": sh0_mins,
            "maxs": sh0_maxs,
            "files": ["sh0.webp"],
        },
    }

    if include_shn:
        sh_coeffs = 15  # SH degree 3
        palette_count = 64  # 1 row of 64 entries
        centroids_w = 64 * sh_coeffs  # 960
        centroids_h = 1  # palette_count / 64
        centroids_img = rng.randint(0, 256, (centroids_h, centroids_w, 4), dtype=np.uint8)

        # Labels: assign each Gaussian to a random palette entry
        labels_img = np.zeros((side, side, 4), dtype=np.uint8)
        for i in range(count):
            r, c = divmod(i, side)
            label = i % palette_count
            labels_img[r, c, 0] = label & 0xFF
            labels_img[r, c, 1] = (label >> 8) & 0xFF

        meta["shN"] = {
            "shape": [count, sh_coeffs],
            "dtype": "float32",
            "mins": -0.5,
            "maxs": 0.5,
            "quantization": 8,
            "files": ["shN_centroids.webp", "shN_labels.webp"],
        }
    else:
        centroids_img = labels_img = None

    sog_dir = tmp_dir / "sog_v3"
    sog_dir.mkdir(exist_ok=True)

    (sog_dir / "meta.json").write_text(json.dumps(meta))
    (sog_dir / "means_l.webp").write_bytes(_make_rgba_webp(side, side, means_lo))
    (sog_dir / "means_u.webp").write_bytes(_make_rgba_webp(side, side, means_hi))
    (sog_dir / "scales.webp").write_bytes(_make_rgba_webp(side, side, scales_img))
    (sog_dir / "quats.webp").write_bytes(_make_rgba_webp(side, side, quats_img))
    (sog_dir / "sh0.webp").write_bytes(_make_rgba_webp(side, side, sh0_img))

    if include_shn:
        (sog_dir / "shN_centroids.webp").write_bytes(
            _make_rgba_webp(centroids_h, centroids_w, centroids_img)
        )
        (sog_dir / "shN_labels.webp").write_bytes(_make_rgba_webp(side, side, labels_img))

    return sog_dir


def _build_v2_sog_folder(
    tmp_dir: Path,
    count: int = 16,
) -> Path:
    """Build a minimal v2 SOG folder (codebook-based) with synthetic data."""
    rng = np.random.RandomState(123)
    side = int(np.ceil(np.sqrt(count)))

    # Generate codebooks (256 entries each)
    scales_codebook = np.linspace(-10.0, -2.0, 256).tolist()
    sh0_codebook = np.linspace(-2.0, 2.0, 256).tolist()
    shn_codebook = np.linspace(-0.5, 0.5, 256).tolist()

    # --- textures ---
    means_lo = rng.randint(0, 256, (side, side, 4), dtype=np.uint8)
    means_hi = rng.randint(0, 256, (side, side, 4), dtype=np.uint8)
    scales_img = rng.randint(0, 256, (side, side, 4), dtype=np.uint8)
    quats_img = np.zeros((side, side, 4), dtype=np.uint8)
    quats_img[:, :, 3] = 254
    quats_img[:, :, :3] = rng.randint(0, 256, (side, side, 3), dtype=np.uint8)
    sh0_img = rng.randint(0, 256, (side, side, 4), dtype=np.uint8)

    # SHN
    sh_coeffs = 15
    palette_count = 64
    centroids_w = 64 * sh_coeffs
    centroids_h = 1
    centroids_img = rng.randint(0, 256, (centroids_h, centroids_w, 4), dtype=np.uint8)
    labels_img = np.zeros((side, side, 4), dtype=np.uint8)
    for i in range(count):
        r, c = divmod(i, side)
        label = i % palette_count
        labels_img[r, c, 0] = label & 0xFF

    meta = {
        "version": 2,
        "count": count,
        "means": {
            "mins": [-1.0, -1.0, -1.0],
            "maxs": [1.0, 1.0, 1.0],
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {
            "codebook": scales_codebook,
            "files": ["scales.webp"],
        },
        "quats": {
            "files": ["quats.webp"],
        },
        "sh0": {
            "codebook": sh0_codebook,
            "files": ["sh0.webp"],
        },
        "shN": {
            "bands": 3,
            "count": palette_count,
            "codebook": shn_codebook,
            "files": ["shN_centroids.webp", "shN_labels.webp"],
        },
    }

    sog_dir = tmp_dir / "sog_v2"
    sog_dir.mkdir(exist_ok=True)

    (sog_dir / "meta.json").write_text(json.dumps(meta))
    (sog_dir / "means_l.webp").write_bytes(_make_rgba_webp(side, side, means_lo))
    (sog_dir / "means_u.webp").write_bytes(_make_rgba_webp(side, side, means_hi))
    (sog_dir / "scales.webp").write_bytes(_make_rgba_webp(side, side, scales_img))
    (sog_dir / "quats.webp").write_bytes(_make_rgba_webp(side, side, quats_img))
    (sog_dir / "sh0.webp").write_bytes(_make_rgba_webp(side, side, sh0_img))
    (sog_dir / "shN_centroids.webp").write_bytes(
        _make_rgba_webp(centroids_h, centroids_w, centroids_img)
    )
    (sog_dir / "shN_labels.webp").write_bytes(_make_rgba_webp(side, side, labels_img))

    return sog_dir


# ===========================================================================
# Unit tests for v3 JIT dequantization functions
# ===========================================================================


class TestDecodeScalesLinear:
    """Test _decode_scales_linear_jit."""

    def test_basic_dequantization(self):
        """Min/max bounds map pixel 0 → min, pixel 255 → max."""
        count = 3
        # 3 pixels at RGBA positions: (0, 128, 255, unused)
        rgba = np.array([0, 0, 0, 0, 128, 128, 128, 0, 255, 255, 255, 0], dtype=np.uint8)
        mins = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
        maxs = np.array([-2.0, -2.0, -2.0], dtype=np.float32)

        sx, sy, sz = _decode_scales_linear_jit(rgba, mins, maxs, count)

        np.testing.assert_allclose(sx[0], -10.0, atol=1e-5)
        np.testing.assert_allclose(sx[2], -2.0, atol=1e-5)
        # midpoint
        expected_mid = -10.0 + (128.0 / 255.0) * 8.0
        np.testing.assert_allclose(sx[1], expected_mid, atol=1e-5)

    def test_per_channel_ranges(self):
        """Each channel uses its own min/max."""
        count = 1
        rgba = np.array([100, 200, 50, 0], dtype=np.uint8)
        mins = np.array([0.0, -5.0, 10.0], dtype=np.float32)
        maxs = np.array([1.0, 5.0, 20.0], dtype=np.float32)

        sx, sy, sz = _decode_scales_linear_jit(rgba, mins, maxs, count)

        np.testing.assert_allclose(sx[0], 100.0 / 255.0, atol=1e-5)
        np.testing.assert_allclose(sy[0], -5.0 + (200.0 / 255.0) * 10.0, atol=1e-5)
        np.testing.assert_allclose(sz[0], 10.0 + (50.0 / 255.0) * 10.0, atol=1e-5)


class TestDecodeColorsLinear:
    """Test _decode_colors_linear_jit."""

    def test_basic_dequantization(self):
        """All four channels (RGB + opacity) are dequantized correctly."""
        count = 1
        rgba = np.array([0, 128, 255, 200], dtype=np.uint8)
        mins = np.array([-2.0, -2.0, -2.0, -4.0], dtype=np.float32)
        maxs = np.array([2.0, 2.0, 2.0, 6.0], dtype=np.float32)

        r, g, b, o = _decode_colors_linear_jit(rgba, mins, maxs, count)

        np.testing.assert_allclose(r[0], -2.0, atol=1e-5)
        np.testing.assert_allclose(b[0], 2.0, atol=1e-5)
        expected_o = -4.0 + (200.0 / 255.0) * 10.0
        np.testing.assert_allclose(o[0], expected_o, atol=1e-4)

    def test_opacity_is_logit_space(self):
        """Verify opacity min/max range can represent logit-space values."""
        count = 2
        # pixel=0 → min, pixel=255 → max
        rgba = np.array([128, 128, 128, 0, 128, 128, 128, 255], dtype=np.uint8)
        mins = np.array([-1.0, -1.0, -1.0, -5.0], dtype=np.float32)
        maxs = np.array([1.0, 1.0, 1.0, 5.0], dtype=np.float32)

        _, _, _, o = _decode_colors_linear_jit(rgba, mins, maxs, count)

        np.testing.assert_allclose(o[0], -5.0, atol=1e-5)
        np.testing.assert_allclose(o[1], 5.0, atol=1e-5)


class TestDecodeShnLinear:
    """Test _decode_shn_linear_jit."""

    def test_basic_dequantization(self):
        """Centroids pixel values are dequantized with scalar min/max."""
        count = 1
        sh_coeffs = 3  # SH degree 1
        palette_count = 64
        centroids_width = 64 * sh_coeffs  # 192

        # Create a centroid image (1 row, centroids_width columns)
        centroids_rgba = np.zeros(1 * centroids_width * 4, dtype=np.uint8)
        # Set centroid 0, coeff 0: R=0, G=128, B=255
        centroids_rgba[0] = 0
        centroids_rgba[1] = 128
        centroids_rgba[2] = 255

        # Label for gaussian 0 → centroid 0
        labels_rgba = np.array([0, 0, 0, 0], dtype=np.uint8)

        shn = _decode_shn_linear_jit(
            labels_rgba, centroids_rgba, -0.5, 0.5, count, sh_coeffs, palette_count, centroids_width
        )

        assert shn.shape == (1, 3, 3)
        np.testing.assert_allclose(shn[0, 0, 0], -0.5, atol=1e-5)  # R=0 → min
        np.testing.assert_allclose(shn[0, 0, 1], -0.5 + (128.0 / 255.0), atol=1e-3)
        np.testing.assert_allclose(shn[0, 0, 2], 0.5, atol=1e-5)  # B=255 → max


# ===========================================================================
# Integration tests for sogread
# ===========================================================================


@requires_imagecodecs
class TestSogreadV3:
    """Test sogread with v3 (linear min/max) format."""

    def test_loads_correct_count(self, tmp_path):
        """sogread returns correct number of Gaussians."""
        sog_dir = _build_v3_sog_folder(tmp_path, count=16)
        data = sogread(str(sog_dir))
        assert len(data) == 16

    def test_output_shapes(self, tmp_path):
        """All output arrays have expected shapes."""
        count = 25
        sog_dir = _build_v3_sog_folder(tmp_path, count=count)
        data = sogread(str(sog_dir))

        assert data.means.shape == (count, 3)
        assert data.scales.shape == (count, 3)
        assert data.quats.shape == (count, 4)
        assert data.opacities.shape == (count,)
        assert data.sh0.shape == (count, 3)
        assert data.shN.shape == (count, 15, 3)

    def test_output_dtypes(self, tmp_path):
        """All output arrays are float32."""
        sog_dir = _build_v3_sog_folder(tmp_path, count=16)
        data = sogread(str(sog_dir))

        assert data.means.dtype == np.float32
        assert data.scales.dtype == np.float32
        assert data.quats.dtype == np.float32
        assert data.opacities.dtype == np.float32
        assert data.sh0.dtype == np.float32
        assert data.shN.dtype == np.float32

    def test_scales_in_range(self, tmp_path):
        """Scales values fall within the configured min/max bounds."""
        sog_dir = _build_v3_sog_folder(tmp_path, count=16)
        data = sogread(str(sog_dir))

        assert data.scales.min() >= -10.0 - 1e-5
        assert data.scales.max() <= -2.0 + 1e-5

    def test_opacities_in_range(self, tmp_path):
        """Opacities (logit-space) fall within configured bounds."""
        sog_dir = _build_v3_sog_folder(tmp_path, count=16)
        data = sogread(str(sog_dir))

        assert data.opacities.min() >= -4.0 - 1e-5
        assert data.opacities.max() <= 6.0 + 1e-5

    def test_shn_in_range(self, tmp_path):
        """SHN values fall within scalar min/max bounds."""
        sog_dir = _build_v3_sog_folder(tmp_path, count=16)
        data = sogread(str(sog_dir))

        assert data.shN.min() >= -0.5 - 1e-5
        assert data.shN.max() <= 0.5 + 1e-5

    def test_sh_degree_3(self, tmp_path):
        """SH degree is correctly detected as 3 when shN has 15 bands."""
        sog_dir = _build_v3_sog_folder(tmp_path, count=16, include_shn=True)
        data = sogread(str(sog_dir))

        assert data.get_sh_degree() == 3
        assert data.shN.shape[1] == 15

    def test_no_shn(self, tmp_path):
        """Works correctly without higher-order SH data."""
        sog_dir = _build_v3_sog_folder(tmp_path, count=16, include_shn=False)
        data = sogread(str(sog_dir))

        assert data.get_sh_degree() == 0
        assert data.shN.shape == (16, 0, 3)

    def test_save_roundtrip(self, tmp_path):
        """Verify v3 SOG data survives save → load roundtrip as compressed PLY."""
        sog_dir = _build_v3_sog_folder(tmp_path, count=64)
        data = sogread(str(sog_dir))

        out_path = str(tmp_path / "out")
        data.save(out_path, compressed=True)

        from gsply import plyread

        data2 = plyread(str(tmp_path / "out.compressed.ply"))

        assert len(data2) == 64
        assert data2.get_sh_degree() == 3


@requires_imagecodecs
class TestSogreadV2:
    """Test sogread still works with v2 (codebook) format."""

    def test_loads_correct_count(self, tmp_path):
        """sogread returns correct number of Gaussians."""
        sog_dir = _build_v2_sog_folder(tmp_path, count=16)
        data = sogread(str(sog_dir))
        assert len(data) == 16

    def test_output_shapes(self, tmp_path):
        """All output arrays have expected shapes."""
        count = 16
        sog_dir = _build_v2_sog_folder(tmp_path, count=count)
        data = sogread(str(sog_dir))

        assert data.means.shape == (count, 3)
        assert data.scales.shape == (count, 3)
        assert data.quats.shape == (count, 4)
        assert data.opacities.shape == (count,)
        assert data.sh0.shape == (count, 3)
        assert data.shN.shape == (count, 15, 3)

    def test_sh_degree_3(self, tmp_path):
        """SH degree is correctly detected as 3."""
        sog_dir = _build_v2_sog_folder(tmp_path, count=16)
        data = sogread(str(sog_dir))

        assert data.get_sh_degree() == 3


@requires_imagecodecs
class TestSogreadFormatDetection:
    """Test format detection between v2 and v3."""

    def test_v2_detected_by_count_key(self, tmp_path):
        """v2 format is detected when meta has top-level 'count'."""
        sog_dir = _build_v2_sog_folder(tmp_path, count=16)
        data = sogread(str(sog_dir))
        # v2 should work without error
        assert len(data) == 16

    def test_v3_detected_by_shape_key(self, tmp_path):
        """v3 format is detected when meta['means'] has 'shape'."""
        sog_dir = _build_v3_sog_folder(tmp_path, count=16)
        data = sogread(str(sog_dir))
        # v3 should work without error
        assert len(data) == 16

    def test_both_produce_same_structure(self, tmp_path):
        """Both formats produce GSData with identical field names and shapes."""
        v2_dir = _build_v2_sog_folder(tmp_path, count=16)
        v3_dir = _build_v3_sog_folder(tmp_path, count=16)

        d2 = sogread(str(v2_dir))
        d3 = sogread(str(v3_dir))

        assert d2.means.shape == d3.means.shape
        assert d2.scales.shape == d3.scales.shape
        assert d2.quats.shape == d3.quats.shape
        assert d2.opacities.shape == d3.opacities.shape
        assert d2.sh0.shape == d3.sh0.shape
        assert d2.shN.shape == d3.shN.shape


@requires_imagecodecs
class TestSogreadRealData:
    """Test with real SOG data from NAS (skipped if unavailable)."""

    V3_SCENE = os.path.join(tempfile.gettempdir(), "test_sog_v3")
    V2_SCENE = os.path.join(tempfile.gettempdir(), "test_sog_v2")

    @pytest.mark.skipif(
        not Path(V3_SCENE).exists(),
        reason="Real v3 test data not available",
    )
    def test_real_v3_scene(self):
        """Load real v3 scene (047ec389 - Maltese on Bean Bag)."""
        data = sogread(self.V3_SCENE)
        assert len(data) == 137827
        assert data.means.shape == (137827, 3)
        assert data.shN.shape == (137827, 15, 3)
        assert data.get_sh_degree() == 3

    @pytest.mark.skipif(
        not Path(V2_SCENE).exists(),
        reason="Real v2 test data not available",
    )
    def test_real_v2_scene(self):
        """Load real v2 scene (fd1f012f)."""
        data = sogread(self.V2_SCENE)
        assert len(data) == 1729652
        assert data.means.shape == (1729652, 3)
        assert data.shN.shape == (1729652, 15, 3)
        assert data.get_sh_degree() == 3
