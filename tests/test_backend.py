"""Tests for the optional C++ acceleration backend: selection + dispatch parity."""

from __future__ import annotations

import gc
import importlib.util

import numpy as np
import pytest

import gsply
from gsply._backend import selected_backend, use_backend


@pytest.fixture(autouse=True)
def _restore_backend():
    """Restore the process-wide backend selection after each test."""
    prev = selected_backend()
    yield
    use_backend(prev)


@pytest.fixture
def data():
    rng = np.random.default_rng(9)
    n = 1000
    means = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
    scales = rng.uniform(-8, -2, (n, 3)).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.uniform(-3, 4, n).astype(np.float32)
    sh0 = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    shN = rng.uniform(-0.4, 0.4, (n, 15, 3)).astype(np.float32)
    return gsply.GSData.from_arrays(
        means=means, scales=scales, quats=quats, opacities=opacities, sh0=sh0, shN=shN, format="ply"
    )


class TestBackendSelection:
    def test_python_is_always_python(self):
        use_backend("python")
        assert gsply.active_backend() == "python"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            use_backend("rust")

    def test_auto_resolves_to_availability(self):
        use_backend("auto")
        expect = "cpp" if importlib.util.find_spec("gsply_cpp") else "python"
        assert gsply.active_backend() == expect

    def test_cpp_without_module_falls_back_to_python(self):
        use_backend("cpp")
        # active_backend() is "cpp" only if gsply_cpp is importable.
        if importlib.util.find_spec("gsply_cpp") is None:
            assert gsply.active_backend() == "python"
        else:
            assert gsply.active_backend() == "cpp"


class TestCppDispatch:
    def test_cpp_decode_matches_python(self, data, tmp_path):
        """C++ backend reading Python-written files decodes identically (ULP)."""
        pytest.importorskip("gsply_cpp")
        use_backend("python")
        gsply.write_spz(str(tmp_path / "py.spz"), data, version=4)
        gsply.plywrite(str(tmp_path / "py.ply"), data)
        py_spz = gsply.read_spz(str(tmp_path / "py.spz"))
        py_ply = gsply.plyread(str(tmp_path / "py.ply"))

        use_backend("cpp")
        assert gsply.active_backend() == "cpp"
        cpp_spz = gsply.read_spz(str(tmp_path / "py.spz"))
        cpp_ply = gsply.plyread(str(tmp_path / "py.ply"))
        assert isinstance(cpp_spz, gsply.GSData)
        assert isinstance(cpp_ply, gsply.GSData)
        for cpp_out, py_out in ((cpp_spz, py_spz), (cpp_ply, py_ply)):
            for f in ("means", "scales", "quats", "sh0", "shN"):
                np.testing.assert_allclose(
                    np.asarray(getattr(cpp_out, f)), np.asarray(getattr(py_out, f)), atol=1e-6
                )

    def test_cpp_plyread_preserves_base(self, data, tmp_path):
        """C++ PLY read should keep the canonical row buffer for zero-copy writes."""
        pytest.importorskip("gsply_cpp")
        use_backend("python")
        gsply.plywrite(str(tmp_path / "py.ply"), data)

        use_backend("cpp")
        cpp_ply = gsply.plyread(str(tmp_path / "py.ply"))

        assert cpp_ply._base is not None
        assert cpp_ply._base.shape == (len(cpp_ply), 59)
        np.testing.assert_array_equal(np.asarray(cpp_ply.means), np.asarray(data.means))
        gc.collect()
        rewrite_path = tmp_path / "cpp_base_rewrite.ply"
        gsply.plywrite(str(rewrite_path), cpp_ply)
        rewritten = gsply.plyread(str(rewrite_path))
        np.testing.assert_array_equal(np.asarray(rewritten.means), np.asarray(data.means))

    def test_cpp_write_roundtrips(self, data, tmp_path):
        """C++ backend writes (SPZ v3/v4 + PLY) round-trip within quantization."""
        pytest.importorskip("gsply_cpp")
        use_backend("cpp")
        for name, kw in (("v3.spz", {"version": 3}), ("v4.spz", {"version": 4})):
            gsply.write_spz(str(tmp_path / name), data, **kw)
            rt = gsply.read_spz(str(tmp_path / name))
            np.testing.assert_allclose(np.asarray(rt.means), np.asarray(data.means), atol=2e-4)
        gsply.plywrite(str(tmp_path / "c.ply"), data)
        rt_ply = gsply.plyread(str(tmp_path / "c.ply"))
        np.testing.assert_array_equal(np.asarray(rt_ply.means), np.asarray(data.means))

    def test_cpp_plywrite_uses_python_fast_paths(self, data, tmp_path, monkeypatch):
        """C++ mode must not bypass faster Python _base and SH0 PLY write paths."""
        pytest.importorskip("gsply_cpp")

        class FakeCpp:
            calls = 0

            @classmethod
            def write_ply(cls, path, *args, **kwargs):
                cls.calls += 1
                with open(path, "wb") as f:
                    f.write(b"cpp")

        use_backend("python")
        source = tmp_path / "source.ply"
        gsply.plywrite(str(source), data)
        data_with_base = gsply.plyread(str(source))
        data_sh0 = gsply.GSData.from_arrays(
            means=np.asarray(data.means),
            scales=np.asarray(data.scales),
            quats=np.asarray(data.quats),
            opacities=np.asarray(data.opacities),
            sh0=np.asarray(data.sh0),
            format="ply",
        )

        from gsply import _backend

        monkeypatch.setattr(_backend, "cpp", lambda: FakeCpp)
        use_backend("cpp")
        base_out = tmp_path / "base_out.ply"
        sh0_out = tmp_path / "sh0_out.ply"

        gsply.plywrite(str(base_out), data_with_base)
        gsply.plywrite(str(sh0_out), data_sh0)

        assert FakeCpp.calls == 0
        assert base_out.exists()
        assert sh0_out.exists()

        shn_out = tmp_path / "shn_out.ply"
        gsply.plywrite(str(shn_out), data)

        assert FakeCpp.calls == 1
        assert shn_out.read_bytes() == b"cpp"

    def test_cpp_mode_uses_python_spz_v4_read_when_zstd_available(
        self, data, tmp_path, monkeypatch
    ):
        """Python zstd is faster for v4 reads; C++ remains fallback when unavailable."""
        pytest.importorskip("gsply_cpp")
        pytest.importorskip("zstandard")

        use_backend("python")
        path = tmp_path / "v4.spz"
        gsply.write_spz(path, data, version=4)

        class FakeCpp:
            @staticmethod
            def read_spz(*args, **kwargs):
                raise AssertionError("v4 read should use the Python zstd path")

        from gsply import _backend

        monkeypatch.setattr(_backend, "cpp", lambda: FakeCpp)
        use_backend("cpp")
        out = gsply.read_spz(path)

        assert len(out) == len(data)

    def test_cpp_plyread_falls_back_for_compressed(self, data, tmp_path):
        """Compressed PLY isn't supported by gsply_cpp -> dispatch falls back to Python."""
        pytest.importorskip("gsply_cpp")
        path = str(tmp_path / "scene.compressed.ply")
        gsply.plywrite(path, data, compressed=True)  # Python writes it
        use_backend("cpp")
        out = gsply.plyread(path)  # must not raise; falls back to pure Python
        assert np.asarray(out.means).shape == (np.asarray(data.means).shape[0], 3)
