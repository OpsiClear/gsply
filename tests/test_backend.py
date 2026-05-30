"""Tests for the optional C++ acceleration backend: selection + dispatch parity."""

from __future__ import annotations

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

    def test_cpp_plyread_falls_back_for_compressed(self, data, tmp_path):
        """Compressed PLY isn't supported by gsply_cpp -> dispatch falls back to Python."""
        pytest.importorskip("gsply_cpp")
        path = str(tmp_path / "scene.compressed.ply")
        gsply.plywrite(path, data, compressed=True)  # Python writes it
        use_backend("cpp")
        out = gsply.plyread(path)  # must not raise; falls back to pure Python
        assert np.asarray(out.means).shape == (np.asarray(data.means).shape[0], 3)
