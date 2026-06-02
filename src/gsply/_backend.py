"""Optional C++ acceleration backend (``gsply_cpp``), opt-in.

gsply is pure Python by default (NumPy + Numba). If the optional
``gsply_cpp`` package is installed *and* the backend is set to ``"cpp"`` -- via
``gsply.use_backend("cpp")`` or the ``GSPLY_BACKEND=cpp`` environment variable --
then the PLY/SPZ read/write entry points route through the C++ backend, falling
back to pure Python for anything it doesn't support (e.g. compressed PLY).

Install it with ``pip install "gsply[cpp]"``. Numerical notes: C++ reads are
float32-ULP-identical to the Python path; C++ writes may differ by <=1 LSB per
quantized value (NumPy ``round`` half-to-even vs C++ ``lround`` half-away) and
use a different compressor, so the compressed bytes differ while the decoded
data matches.
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
    from gsply.gsdata import GSData

    return GSData.from_arrays(
        means=d["means"],
        scales=d["scales"],
        quats=d["quats"],
        opacities=d["opacities"],
        sh0=d["sh0"],
        shN=d["shN"],
        format="ply",
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
