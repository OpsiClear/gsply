"""Tests for SPZ read/write: round-trip, independent-decoder parity, and (opt-in)
parity against the upstream Niantic ``spz`` bindings."""

from __future__ import annotations

import gzip
import struct

import numpy as np
import pytest

import gsply
from gsply.spz import COLOR_SCALE, NGSP_MAGIC, read_spz, write_spz


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture
def gs(tmp_path):
    """A PLY-format GSData with SH3 + the arrays it was built from."""
    rng = np.random.default_rng(42)
    n = 256
    means = rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)
    scales = rng.uniform(-8.0, -2.5, (n, 3)).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)  # wxyz unit
    opacities = rng.uniform(-3.0, 4.0, n).astype(np.float32)
    sh0 = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    shN = rng.uniform(-0.4, 0.4, (n, 15, 3)).astype(np.float32)
    data = gsply.GSData.from_arrays(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
    )
    return data, {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }


def _reference_decode(raw: bytes) -> dict:
    """Independent pure-numpy SPZ decoder mirroring nianticlabs/spz load-spz.cc.

    Deliberately shares no code with the Numba kernel so it can validate it.
    """
    magic, version, n, sh_deg, frac, _fl, _r = struct.unpack("<IIIBBBB", raw[:16])
    assert magic == NGSP_MAGIC
    sh_dim = {0: 0, 1: 3, 2: 8, 3: 15}[sh_deg]
    rs = 4 if version >= 3 else 3
    pay = raw[16:]
    o = 0
    pos = np.frombuffer(pay, np.uint8, 9 * n, o).reshape(n, 3, 3).astype(np.int32)
    o += 9 * n
    al = np.frombuffer(pay, np.uint8, n, o).astype(np.float32)
    o += n
    col = np.frombuffer(pay, np.uint8, 3 * n, o).reshape(n, 3).astype(np.float32)
    o += 3 * n
    scl = np.frombuffer(pay, np.uint8, 3 * n, o).reshape(n, 3).astype(np.float32)
    o += 3 * n
    rot = np.frombuffer(pay, np.uint8, rs * n, o)
    o += rs * n
    sh = np.frombuffer(pay, np.uint8, sh_dim * 3 * n, o).reshape(n, sh_dim, 3).astype(np.float32)

    pi = pos[..., 0] | (pos[..., 1] << 8) | (pos[..., 2] << 16)
    pi -= (pi & 0x800000) << 1
    means = pi.astype(np.float32) / (1 << frac)
    scales = scl / 16.0 - 10.0
    p = np.clip(al / 255.0, 1e-6, 1 - 1e-6)
    opac = np.log(p / (1 - p))
    sh0 = (col / 255.0 - 0.5) / COLOR_SCALE
    shN = (sh - 128.0) / 128.0 if sh_dim else None

    comp = rot.reshape(n, 4).astype(np.uint32)
    packed = comp[:, 0] | (comp[:, 1] << 8) | (comp[:, 2] << 16) | (comp[:, 3] << 24)
    ilg = (packed >> 30) & 3
    work = packed.copy()
    q = np.zeros((n, 4), np.float32)
    for ax in (3, 2, 1, 0):
        wm = ilg != ax
        mag = (work & 511).astype(np.float32)
        nb = (work >> 9) & 1
        val = 0.7071067811865476 * mag / 511.0
        val = np.where(nb == 1, -val, val)
        q[wm, ax] = val[wm]
        work = np.where(wm, work >> 10, work)
    q[np.arange(n), ilg] = np.sqrt(np.maximum(0.0, 1.0 - (q * q).sum(1)))
    quats = np.roll(q, 1, axis=1)  # xyzw -> wxyz
    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opac,
        "sh0": sh0,
        "shN": shN,
    }


class TestRoundTrip:
    def test_write_read_within_quant(self, gs, tmp_path):
        data, arr = gs
        path = tmp_path / "rt.spz"
        write_spz(path, data)
        out = read_spz(path)

        np.testing.assert_allclose(out.means, arr["means"], atol=2e-4)  # 12-bit fixed point
        np.testing.assert_allclose(out.scales, arr["scales"], atol=1.0 / 16 + 1e-6)
        np.testing.assert_allclose(
            _sigmoid(np.asarray(out.opacities).reshape(-1)),
            _sigmoid(arr["opacities"]),
            atol=1.0 / 255,
        )
        np.testing.assert_allclose(out.sh0, arr["sh0"], atol=1.0 / (COLOR_SCALE * 255))
        np.testing.assert_allclose(out.shN, arr["shN"], atol=2.0 / 128)
        # quats: sign- and order-aware (|q.q'| == 1 for equal rotations)
        dots = np.abs((np.asarray(out.quats) * arr["quats"]).sum(1))
        assert dots.min() > 0.99, f"min |dot|={dots.min():.4f}"

    def test_sh0_only_roundtrip(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 64
        data = gsply.GSData.from_arrays(
            means=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
            scales=rng.uniform(-8, -3, (n, 3)).astype(np.float32),
            quats=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
            opacities=rng.uniform(-2, 3, n).astype(np.float32),
            sh0=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
            shN=None,
            format="ply",
        )
        path = tmp_path / "sh0.spz"
        write_spz(path, data)
        out = read_spz(path)
        assert out.means.shape == (n, 3)
        assert out.shN is None or np.asarray(out.shN).size == 0


class TestKernelParity:
    def test_kernel_matches_independent_reference(self, gs, tmp_path):
        """The Numba decode must agree with an independent numpy spec decoder."""
        data, _ = gs
        path = tmp_path / "p.spz"
        write_spz(path, data)
        ref = _reference_decode(gzip.decompress(path.read_bytes()))
        out = read_spz(path)
        np.testing.assert_allclose(out.means, ref["means"], atol=1e-6)
        np.testing.assert_allclose(out.scales, ref["scales"], atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(out.opacities).reshape(-1), ref["opacities"], atol=1e-4
        )
        np.testing.assert_allclose(out.sh0, ref["sh0"], atol=1e-4)
        np.testing.assert_allclose(out.shN, ref["shN"], atol=1e-6)
        np.testing.assert_allclose(np.asarray(out.quats), ref["quats"], atol=1e-4)


class TestReferenceBindings:
    """Opt-in parity against the upstream Niantic spz bindings (skips if absent)."""

    def test_gsply_spz_readable_by_niantic(self, gs, tmp_path):
        spz = pytest.importorskip("spz")
        data, arr = gs
        path = tmp_path / "ref.spz"
        write_spz(path, data)

        cloud = spz.load_spz(str(path))
        assert cloud.num_points == arr["means"].shape[0]
        assert cloud.sh_degree == 3
        # Compare coordinate-invariant channels (positions/rotations may be flipped
        # by the loader's coordinate conversion; scales/colors are frame-agnostic).
        scales = np.asarray(cloud.scales, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(cloud.colors, dtype=np.float32).reshape(-1, 3)
        np.testing.assert_allclose(scales, arr["scales"], atol=1.0 / 16 + 1e-6)
        # GaussianCloud.colors are already decoded to the SH DC convention (== sh0),
        # not the 0..1 wide-RGB byte values, so compare directly.
        np.testing.assert_allclose(colors, arr["sh0"], atol=1.0 / (COLOR_SCALE * 255))


class TestV4:
    """NGSP v4 (zstd) container: round-trip, parity with the gzip path, interop."""

    @pytest.fixture(autouse=True)
    def _require_zstd(self):
        """v4 needs the optional 'zstandard' dependency (gsply[spz]); skip without it."""
        pytest.importorskip("zstandard")

    def test_v4_roundtrip(self, gs, tmp_path):
        data, arr = gs
        path = tmp_path / "v4.spz"
        write_spz(path, data, version=4)
        raw = path.read_bytes()
        assert struct.unpack_from("<I", raw, 0)[0] == NGSP_MAGIC  # uncompressed NGSP header
        assert struct.unpack_from("<I", raw, 4)[0] == 4  # version field
        assert raw[:2] != b"\x1f\x8b"  # not gzip
        out = read_spz(path)
        np.testing.assert_allclose(np.asarray(out.means), arr["means"], atol=2e-4)
        np.testing.assert_allclose(np.asarray(out.scales), arr["scales"], atol=1.0 / 16 + 1e-6)
        np.testing.assert_allclose(np.asarray(out.sh0), arr["sh0"], atol=1.0 / (COLOR_SCALE * 255))
        np.testing.assert_allclose(np.asarray(out.shN), arr["shN"], atol=2.0 / 128)
        dots = np.abs((np.asarray(out.quats) * arr["quats"]).sum(1))
        assert dots.min() > 0.99

    def test_v4_decodes_identically_to_v3(self, gs, tmp_path):
        """v3 and v4 share identical packed sections -> identical decoded arrays."""
        data, _ = gs
        p3, p4 = tmp_path / "a.spz", tmp_path / "b.spz"
        write_spz(p3, data, version=3)
        write_spz(p4, data, version=4)
        a, b = read_spz(p3), read_spz(p4)
        for f in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            np.testing.assert_array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)))

    def test_v4_sh0_only(self, tmp_path):
        """SH degree 0 -> 5 streams (no sh stream)."""
        rng = np.random.default_rng(1)
        n = 64
        data = gsply.GSData.from_arrays(
            means=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
            scales=rng.uniform(-6, -3, (n, 3)).astype(np.float32),
            quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
            opacities=rng.uniform(-2, 2, n).astype(np.float32),
            sh0=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
            shN=None,
            format="ply",
        )
        path = tmp_path / "sh0.spz"
        write_spz(path, data, version=4)
        assert path.read_bytes()[15] == 5  # num_streams (no sh)
        out = read_spz(path)
        assert out.shN is None
        assert np.asarray(out.means).shape == (n, 3)

    def test_bad_version(self, gs, tmp_path):
        data, _ = gs
        with pytest.raises(ValueError, match="version"):
            write_spz(tmp_path / "x.spz", data, version=2)

    def test_v4_niantic_interop(self, gs, tmp_path):
        """gsply writes v4 -> Niantic reads; Niantic writes v4 -> gsply reads."""
        spz = pytest.importorskip("spz")
        data, arr = gs
        gp = tmp_path / "g_v4.spz"
        write_spz(gp, data, version=4)
        cloud = spz.load_spz(str(gp))
        assert cloud.num_points == arr["means"].shape[0]
        assert cloud.sh_degree == 3
        scales = np.asarray(cloud.scales, np.float32).reshape(-1, 3)
        np.testing.assert_allclose(scales, arr["scales"], atol=1.0 / 16 + 1e-6)
        # Niantic's own v4 (default PackOptions) must read back in gsply.
        np_path = tmp_path / "n_v4.spz"
        spz.save_spz(cloud, spz.PackOptions(), str(np_path))
        out = read_spz(np_path)
        nm = np.asarray(cloud.positions, np.float32).reshape(-1, 3)
        np.testing.assert_allclose(np.asarray(out.means), nm, atol=2e-4)


class TestErrors:
    def test_bad_magic(self, tmp_path):
        path = tmp_path / "bad.spz"
        path.write_bytes(gzip.compress(b"\x00" * 32))
        with pytest.raises(ValueError, match="Not an SPZ file"):
            read_spz(path)

    def test_not_gzip(self, tmp_path):
        path = tmp_path / "x.spz"
        path.write_bytes(b"not a gzip stream")
        # Neither gzip (1f 8b) nor NGSP magic -> unrecognized container.
        with pytest.raises(ValueError, match="Not an SPZ file"):
            read_spz(path)
