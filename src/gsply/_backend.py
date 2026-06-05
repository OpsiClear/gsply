"""Bundled C++ acceleration backend (``gsply_cpp``), opt-in.

gsply uses the Python backend by default (NumPy + Numba). Published platform
wheels include ``gsply_cpp`` on supported platforms. If the backend is set to
``"cpp"`` -- via ``gsply.use_backend("cpp")`` or the ``GSPLY_BACKEND=cpp``
environment variable -- then the PLY/SPZ read/write entry points route through
the C++ backend, falling back to pure Python for anything it doesn't support
(e.g. compressed PLY).

Install with ``pip install gsply``. Source installs can remain Python-only, or
build the extension explicitly with
``-Cwheel.cmake=true -Ccmake.define.GSPLY_BUILD_CPP=ON``. Numerical notes: C++
reads are float32-ULP-identical to the Python path; C++ writes may differ by
<=1 LSB per quantized value (NumPy ``round`` half-to-even vs C++ ``lround``
half-away) and use a different compressor, so the compressed bytes differ while
the decoded data matches.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from gsply.gsdata import GSData

_VALID_BACKENDS = ("python", "cpp", "auto")
# "python": always pure Python. "cpp": use gsply_cpp (error if unusable paths).
# "auto": use gsply_cpp when importable, else pure Python.
_state = {"backend": (os.environ.get("GSPLY_BACKEND") or "python").strip().lower()}
if _state["backend"] not in _VALID_BACKENDS:
    _state["backend"] = "python"


@functools.lru_cache(maxsize=1)
def _import_cpp() -> Any | None:
    try:
        import gsply_cpp  # type: ignore[import-not-found]

        return gsply_cpp
    except ImportError:
        return None


def use_backend(name: str) -> None:
    """Select the I/O backend for PLY/SPZ read/write.

    Args:
        name: ``"python"`` (default, pure Python), ``"cpp"`` (use the
            ``gsply_cpp`` C++ backend; requires it to be installed), or
            ``"auto"`` (use ``gsply_cpp`` if importable, else pure Python).

    Raises:
        ValueError: For an unknown backend name.
    """
    key = name.strip().lower()
    if key not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {name!r}")
    _state["backend"] = key


def selected_backend() -> str:
    """Return the requested backend setting (``python`` / ``cpp`` / ``auto``)."""
    return _state["backend"]


def active_backend() -> str:
    """Return the backend that will actually be used: ``"cpp"`` or ``"python"``.

    ``"cpp"`` only when it was requested (``cpp``/``auto``) *and* ``gsply_cpp``
    is importable; otherwise ``"python"``.
    """
    if _state["backend"] in ("cpp", "auto") and _import_cpp() is not None:
        return "cpp"
    return "python"


def cpp() -> Any | None:
    """Return the imported ``gsply_cpp`` module, or ``None`` if unavailable."""
    return _import_cpp()


def dict_to_gsdata(d: dict[str, Any]) -> GSData:
    """Wrap a ``gsply_cpp`` read result (dict of arrays) into a PLY-format GSData."""
    import numpy as np

    from gsply.gsdata import GSData

    base = _canonical_base_from_cpp_dict(d)
    if base is not None:
        from gsply.gsdata import DataFormat, _create_format_dict, _get_sh_order_format

        sh_coeffs = 0 if d.get("shN") is None else np.asarray(d["shN"]).shape[1]
        recreated = GSData._recreate_from_base(
            base,
            _create_format_dict(
                scales=DataFormat.SCALES_PLY,
                opacities=DataFormat.OPACITIES_PLY,
                sh0=DataFormat.SH0_SH,
                sh_order=_get_sh_order_format({0: 0, 3: 1, 8: 2, 15: 3}[sh_coeffs]),
                means=DataFormat.MEANS_RAW,
                quats=DataFormat.QUATS_RAW,
            ),
        )
        if recreated is not None:
            return recreated

    return GSData.from_arrays(
        means=d["means"],
        scales=d["scales"],
        quats=d["quats"],
        opacities=d["opacities"],
        sh0=d["sh0"],
        shN=d["shN"],
        format="ply",
    )


def _canonical_base_from_cpp_dict(d: dict[str, Any]) -> np.ndarray | None:
    """Recover the canonical PLY base from ``gsply_cpp.read_ply`` views.

    The C++ reader exposes zero-copy field views into one canonical row buffer.
    ``GSData.from_arrays()`` would discard that shared base and force later writes
    to gather/interleave again, so reconstruct the full ``(N, P)`` array when
    the field strides and offsets prove this is the canonical PLY layout.
    """
    import numpy as np

    means = np.asarray(d.get("means"))
    if means.ndim != 2 or means.shape[1] != 3 or means.dtype != np.float32:
        return None
    if len(means.strides) != 2 or means.strides[1] != means.dtype.itemsize:
        return None

    shn = d.get("shN")
    sh_coeffs = 0 if shn is None else np.asarray(shn).shape[1]
    if sh_coeffs not in (0, 3, 8, 15):
        return None

    n_props = 14 + sh_coeffs * 3
    row_stride = means.strides[0]
    itemsize = means.dtype.itemsize
    if row_stride != n_props * itemsize:
        return None

    ptr0 = means.__array_interface__["data"][0]
    base_from_cpp = None
    if d.get("_base") is not None:
        candidate = np.asarray(d["_base"])
        if (
            candidate.dtype == np.float32
            and candidate.shape == (means.shape[0], n_props)
            and candidate.strides == (row_stride, itemsize)
            and candidate.__array_interface__["data"][0] == ptr0
        ):
            base_from_cpp = candidate

    def _field_offset(name: str, shape: tuple[int, ...], strides: tuple[int, ...]) -> int | None:
        arr = np.asarray(d.get(name))
        if arr.dtype != np.float32 or arr.shape != shape or arr.strides != strides:
            return None
        delta = arr.__array_interface__["data"][0] - ptr0
        if delta < 0 or delta % itemsize != 0:
            return None
        return delta // itemsize

    opacity_idx = 6 + sh_coeffs * 3
    n = means.shape[0]
    expected_fields = {
        "sh0": ((n, 3), (row_stride, itemsize), 3),
        "opacities": ((n,), (row_stride,), opacity_idx),
        "scales": ((n, 3), (row_stride, itemsize), opacity_idx + 1),
        "quats": ((n, 4), (row_stride, itemsize), opacity_idx + 4),
    }
    for name, (shape, strides, expected) in expected_fields.items():
        offset = _field_offset(name, shape, strides)
        if offset != expected:
            return None

    if sh_coeffs > 0:
        shn_arr = np.asarray(shn)
        if shn_arr.ndim != 3 or shn_arr.shape[2] != 3:
            return None
        if shn_arr.__array_interface__["data"][0] - ptr0 != 6 * itemsize:
            return None
        if shn_arr.strides != (row_stride, itemsize, sh_coeffs * itemsize):
            return None

    if base_from_cpp is not None:
        return base_from_cpp

    return np.lib.stride_tricks.as_strided(
        means,
        shape=(means.shape[0], n_props),
        strides=(row_stride, itemsize),
        writeable=means.flags.writeable,
    )


def gsdata_ply_arrays(
    data: GSData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract contiguous float32 PLY-format arrays from a GSData for a C++ write.

    Mirrors the assumptions of the pure-Python writers (data is interpreted in
    PLY format). Returns ``(means, scales, quats, opacities, sh0, shN)`` with
    ``shN`` either an ``(N, K, 3)`` array or ``None``.
    """
    import numpy as np

    means = np.ascontiguousarray(data.means, dtype=np.float32)
    scales = np.ascontiguousarray(data.scales, dtype=np.float32)
    quats = np.ascontiguousarray(data.quats, dtype=np.float32)
    opacities = np.ascontiguousarray(data.opacities, dtype=np.float32).reshape(-1)
    sh0 = np.ascontiguousarray(data.sh0, dtype=np.float32)
    shn = data.shN
    if shn is None or np.asarray(shn).size == 0:
        shn_arr = None
    else:
        shn_arr = np.ascontiguousarray(shn, dtype=np.float32)
    return means, scales, quats, opacities, sh0, shn_arr
