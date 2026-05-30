"""Parity tests: gsply_cpp (C++) must match gsply (Python) for PLY I/O."""

from __future__ import annotations

import numpy as np
import pytest

gsply = pytest.importorskip("gsply")
gsply_cpp = pytest.importorskip("gsply_cpp")


@pytest.fixture
def arrays():
    rng = np.random.default_rng(7)
    n = 200
    means = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
    scales = rng.uniform(-8, -2, (n, 3)).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.uniform(-3, 4, n).astype(np.float32)
    sh0 = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    shN = rng.uniform(-0.4, 0.4, (n, 15, 3)).astype(np.float32)
    return means, scales, quats, opacities, sh0, shN


def test_cpp_read_matches_gsply(arrays, tmp_path):
    """C++ read of a gsply-written PLY == gsply's own read (lossless float)."""
    means, scales, quats, opacities, sh0, shN = arrays
    path = str(tmp_path / "scene.ply")
    gsply.plywrite(path, means, scales, quats, opacities, sh0, shN)

    py = gsply.plyread(path)
    cpp = gsply_cpp.read_ply(path)
    np.testing.assert_array_equal(cpp["means"], np.asarray(py.means))
    np.testing.assert_array_equal(cpp["scales"], np.asarray(py.scales))
    np.testing.assert_array_equal(cpp["quats"], np.asarray(py.quats))
    np.testing.assert_array_equal(
        cpp["opacities"].reshape(-1), np.asarray(py.opacities).reshape(-1)
    )
    np.testing.assert_array_equal(cpp["sh0"], np.asarray(py.sh0))
    assert cpp["shN"].shape == (200, 15, 3)
    np.testing.assert_array_equal(cpp["shN"], np.asarray(py.shN))


def test_cpp_write_readable_by_gsply(arrays, tmp_path):
    """gsply can read a C++-written PLY and recover the original arrays exactly."""
    means, scales, quats, opacities, sh0, shN = arrays
    path = str(tmp_path / "cpp.ply")
    gsply_cpp.write_ply(path, means, scales, quats, opacities, sh0, shN)

    py = gsply.plyread(path)
    np.testing.assert_array_equal(np.asarray(py.means), means)
    np.testing.assert_array_equal(np.asarray(py.scales), scales)
    np.testing.assert_array_equal(np.asarray(py.quats), quats)
    np.testing.assert_array_equal(np.asarray(py.opacities).reshape(-1), opacities)
    np.testing.assert_array_equal(np.asarray(py.sh0), sh0)
    np.testing.assert_array_equal(np.asarray(py.shN), shN)


def test_cpp_roundtrip(arrays, tmp_path):
    """C++ write -> C++ read recovers the input exactly."""
    means, scales, quats, opacities, sh0, shN = arrays
    path = str(tmp_path / "rt.ply")
    gsply_cpp.write_ply(path, means, scales, quats, opacities, sh0, shN)
    d = gsply_cpp.read_ply(path)
    np.testing.assert_array_equal(d["means"], means)
    np.testing.assert_array_equal(d["shN"], shN)


def test_sh0_only(arrays, tmp_path):
    means, scales, quats, opacities, sh0, _ = arrays
    path = str(tmp_path / "sh0.ply")
    gsply_cpp.write_ply(path, means, scales, quats, opacities, sh0, None)
    d = gsply_cpp.read_ply(path)
    assert d["means"].shape == (200, 3)
    assert d["shN"] is None


def _gsdata(arrays):
    means, scales, quats, opacities, sh0, shN = arrays
    return gsply.GSData.from_arrays(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
    )


class TestSpz:
    def test_cpp_read_matches_gsply(self, arrays, tmp_path):
        """Reading the same SPZ bytes: C++ == Python, exactly."""
        path = str(tmp_path / "s.spz")
        gsply.write_spz(path, _gsdata(arrays))
        py = gsply.read_spz(path)
        cpp = gsply_cpp.read_spz(path)
        # Same bytes, same decode math — but numba's fastmath fuses/reorders float
        # ops, so multi-op channels match plain C++ only to float32 ULP (~1e-7).
        np.testing.assert_allclose(cpp["means"], np.asarray(py.means), atol=1e-6)
        np.testing.assert_allclose(cpp["scales"], np.asarray(py.scales), atol=1e-6)
        np.testing.assert_allclose(cpp["sh0"], np.asarray(py.sh0), atol=1e-6)
        np.testing.assert_allclose(cpp["shN"], np.asarray(py.shN), atol=1e-6)
        np.testing.assert_allclose(cpp["quats"], np.asarray(py.quats), atol=1e-6)
        np.testing.assert_allclose(
            cpp["opacities"].reshape(-1), np.asarray(py.opacities).reshape(-1), atol=1e-5
        )

    def test_writer_payload_parity(self, arrays, tmp_path):
        """C++ and Python writers must emit identical SPZ payloads (gzip body may
        differ by compressor; the decompressed sections must match byte-for-byte)."""
        import gzip

        py_path = str(tmp_path / "py.spz")
        cpp_path = str(tmp_path / "cpp.spz")
        means, scales, quats, opacities, sh0, shN = arrays
        gsply.write_spz(py_path, _gsdata(arrays))
        gsply_cpp.write_spz(cpp_path, means, scales, quats, opacities, sh0, shN)
        py_payload = gzip.decompress(open(py_path, "rb").read())
        cpp_payload = gzip.decompress(open(cpp_path, "rb").read())
        assert py_payload == cpp_payload, "SPZ encode payloads differ"

    def test_cpp_write_readable_by_gsply(self, arrays, tmp_path):
        """gsply reads a C++-written SPZ and recovers the input within quantization."""
        means, scales, quats, opacities, sh0, shN = arrays
        path = str(tmp_path / "cw.spz")
        gsply_cpp.write_spz(path, means, scales, quats, opacities, sh0, shN)
        d = gsply.read_spz(path)
        np.testing.assert_allclose(np.asarray(d.means), means, atol=2e-4)
        np.testing.assert_allclose(np.asarray(d.scales), scales, atol=1.0 / 16 + 1e-6)
        np.testing.assert_allclose(np.asarray(d.sh0), sh0, atol=1.0 / (0.15 * 255))
        np.testing.assert_allclose(np.asarray(d.shN), shN, atol=2.0 / 128)
        dots = np.abs((np.asarray(d.quats) * quats).sum(1))
        assert dots.min() > 0.99

    def test_real_nas_file(self):
        """C++ and Python read of a real Niantic SPZ agree exactly (skips if absent)."""
        from pathlib import Path

        p = Path(r"\\OpsiClearNAS2\Data_NAS2\datasets\4DSL\SEQUENCES\2001take4\frame_000001.spz")
        if not p.exists():
            pytest.skip("NAS sample SPZ not available")
        py = gsply.read_spz(str(p))
        cpp = gsply_cpp.read_spz(str(p))
        np.testing.assert_allclose(cpp["means"], np.asarray(py.means), atol=1e-6)
        np.testing.assert_allclose(cpp["shN"], np.asarray(py.shN), atol=1e-6)
        np.testing.assert_allclose(cpp["quats"], np.asarray(py.quats), atol=1e-6)

    def test_parallel_write_is_single_member(self, tmp_path):
        """The parallel (pigz-style) writer must emit ONE gzip member: a strict
        single-shot inflate (as Niantic's loader does — stop at first Z_STREAM_END)
        must recover the entire payload, with nothing left over."""
        import gzip
        import zlib

        rng = np.random.default_rng(3)
        n = 300_000  # SH3 payload ~19.5 MB -> many 1 MB blocks (exercises stitching)
        means = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
        scales = rng.uniform(-8, -2, (n, 3)).astype(np.float32)
        quats = rng.standard_normal((n, 4)).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        opac = rng.uniform(-3, 4, n).astype(np.float32)
        sh0 = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        shN = rng.uniform(-0.4, 0.4, (n, 15, 3)).astype(np.float32)  # noqa: N806

        path = str(tmp_path / "big.spz")
        gsply_cpp.write_spz(path, means, scales, quats, opac, sh0, shN)
        raw = open(path, "rb").read()

        # Strict single-member decode (decompressobj stops at the first member end).
        dobj = zlib.decompressobj(16 + 15)  # 16 => gzip framing
        strict = dobj.decompress(raw) + dobj.flush()
        assert dobj.eof, "stream did not terminate in a single member"
        assert dobj.unused_data == b"", "trailing data after first member (multi-member!)"
        # And it must equal the full gzip decode (which loops members if any).
        assert strict == gzip.decompress(raw), "single-shot decode lost data"

        # Round-trips through both readers within quantization.
        d = gsply.read_spz(path)
        cpp = gsply_cpp.read_spz(path)
        np.testing.assert_allclose(cpp["means"], np.asarray(d.means), atol=1e-6)
        np.testing.assert_allclose(cpp["shN"], np.asarray(d.shN), atol=1e-6)
        np.testing.assert_allclose(np.asarray(d.means), means, atol=2e-4)
        dots = np.abs((np.asarray(d.quats) * quats).sum(1))
        assert dots.min() > 0.99

    def test_write_level_param(self, arrays, tmp_path):
        """level= is accepted and all levels round-trip to the same payload."""
        import gzip

        means, scales, quats, opacities, sh0, shN = arrays  # noqa: N806
        payloads = []
        for lvl in (1, 6, 12):
            p = str(tmp_path / f"l{lvl}.spz")
            gsply_cpp.write_spz(p, means, scales, quats, opacities, sh0, shN, level=lvl)
            payloads.append(gzip.decompress(open(p, "rb").read()))
        assert payloads[0] == payloads[1] == payloads[2], "level changed the payload"


class TestSpzV4:
    """NGSP v4 (zstd) container in C++: round-trip, parity with v3, cross-interop."""

    def _arrays(self, n=2000):
        rng = np.random.default_rng(11)
        means = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
        scales = rng.uniform(-8, -2, (n, 3)).astype(np.float32)
        quats = rng.standard_normal((n, 4)).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        opac = rng.uniform(-3, 4, n).astype(np.float32)
        sh0 = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        shN = rng.uniform(-0.4, 0.4, (n, 15, 3)).astype(np.float32)  # noqa: N806
        return means, scales, quats, opac, sh0, shN

    def test_v4_header_and_roundtrip(self, tmp_path):
        import struct

        me, sc, q, op, s0, sN = self._arrays()  # noqa: N806
        p = str(tmp_path / "v4.spz")
        gsply_cpp.write_spz(p, me, sc, q, op, s0, sN, version=4)
        raw = open(p, "rb").read()
        assert struct.unpack_from("<I", raw, 0)[0] == 0x5053474E  # "NGSP"
        assert struct.unpack_from("<I", raw, 4)[0] == 4  # version
        assert raw[:2] != b"\x1f\x8b"  # not gzip
        d = gsply_cpp.read_spz(p)
        np.testing.assert_allclose(d["means"], me, atol=2e-4)
        np.testing.assert_allclose(d["scales"], sc, atol=1.0 / 16 + 1e-6)
        np.testing.assert_allclose(d["sh0"], s0, atol=1.0 / (0.15 * 255))
        np.testing.assert_allclose(d["shN"], sN, atol=2.0 / 128)

    def test_v4_decodes_identically_to_v3(self, tmp_path):
        me, sc, q, op, s0, sN = self._arrays()  # noqa: N806
        p3, p4 = str(tmp_path / "a.spz"), str(tmp_path / "b.spz")
        gsply_cpp.write_spz(p3, me, sc, q, op, s0, sN, version=3)
        gsply_cpp.write_spz(p4, me, sc, q, op, s0, sN, version=4)
        a, b = gsply_cpp.read_spz(p3), gsply_cpp.read_spz(p4)
        for k in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            np.testing.assert_array_equal(a[k], b[k])

    def test_v4_cpp_python_cross_read(self, tmp_path):
        """C++ reads Python's v4 and vice versa; both decoders agree on each file.

        Per-file (not cross-file): the Python and C++ *packers* round half-way
        values differently (np.round half-to-even vs std::lround half-away), so
        independently-written files can differ by 1 LSB. Each file is internally
        consistent, and the two decoders agree on it to float32 ULP.
        """
        me, sc, q, op, s0, sN = self._arrays()  # noqa: N806
        cp = str(tmp_path / "cpp_v4.spz")
        pp = str(tmp_path / "py_v4.spz")
        gsply_cpp.write_spz(cp, me, sc, q, op, s0, sN, version=4)
        d = gsply.GSData.from_arrays(
            means=me, scales=sc, quats=q, opacities=op, sh0=s0, shN=sN, format="ply"
        )
        gsply.write_spz(pp, d, version=4)
        for src in (cp, pp):  # C++ reads py-v4 (and py reads cpp-v4); decoders agree
            x = gsply_cpp.read_spz(src)
            y = gsply.read_spz(src)
            for k in ("means", "scales", "quats", "sh0", "shN"):
                np.testing.assert_allclose(x[k], np.asarray(getattr(y, k)), atol=1e-6)

    def test_v4_interop_niantic(self, tmp_path):
        spz = pytest.importorskip("spz")
        me, sc, q, op, s0, sN = self._arrays()  # noqa: N806
        p = str(tmp_path / "g.spz")
        gsply_cpp.write_spz(p, me, sc, q, op, s0, sN, version=4)
        cloud = spz.load_spz(p)  # Niantic reads our v4
        assert cloud.num_points == me.shape[0] and cloud.sh_degree == 3
        np.testing.assert_allclose(
            np.asarray(cloud.scales, np.float32).reshape(-1, 3), sc, atol=1.0 / 16 + 1e-6
        )
        np_path = str(tmp_path / "n.spz")
        spz.save_spz(cloud, spz.PackOptions(), np_path)  # Niantic writes v4
        d = gsply_cpp.read_spz(np_path)  # we read Niantic's v4
        np.testing.assert_allclose(
            d["means"], np.asarray(cloud.positions, np.float32).reshape(-1, 3), atol=2e-4
        )
