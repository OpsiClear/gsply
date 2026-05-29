"""
Read and write Niantic SPZ Gaussian-splat files (legacy gzip v1/v2/v3).

SPZ is a gzip-compressed binary container. Layout after gunzip:

    Header (16B): magic u32 ("NGSP") | version u32 | num_points u32 |
                  sh_degree u8 | fractional_bits u8 | flags u8 | reserved u8
    Payload (contiguous sections):
        positions  9*N bytes   (xyz as 24-bit signed fixed point, little-endian)
        alphas     N   bytes   (sigmoid(opacity) * 255)
        colors     3*N bytes   (sh0 wide-RGB:  byte = sh0 * 0.15 * 255 + 127.5)
        scales     3*N bytes   (byte = (log_scale + 10) * 16)
        rotations  R*N bytes   (v<=2: 3 bytes xyz; v>=3: 4 bytes smallest-three)
        sh         K*3*N bytes (per coeff: byte = sh * 128 + 128)

Reading uses a single fused Numba kernel (one parallel pass over the payload, no
intermediate allocations) and ISA-L's igzip when available (falls back to stdlib
gzip). Writing mirrors the Niantic ``packGaussians`` spec (v3 smallest-three quats).

Reference: github.com/nianticlabs/spz (load-spz.cc).
"""

from __future__ import annotations

import gzip
import math
import struct
import threading
from pathlib import Path

import numba
import numpy as np
from numba import jit

from gsply.gsdata import GSData

try:  # ISA-L igzip: gzip-compatible, markedly faster (de)compress where wheels exist.
    from isal import igzip as _igzip

    def _gunzip(raw: bytes) -> bytes:
        out: bytes = _igzip.decompress(raw)
        return out

    def _gzip_compress(raw: bytes) -> bytes:
        out: bytes = _igzip.compress(raw)
        return out

except ImportError:  # pragma: no cover - environment-dependent

    def _gunzip(raw: bytes) -> bytes:
        return gzip.decompress(raw)

    def _gzip_compress(raw: bytes) -> bytes:
        return gzip.compress(raw)


NGSP_MAGIC = 0x5053474E
COLOR_SCALE = 0.15
SH_MAX_DEGREE = 3
MAX_FRACTIONAL_BITS = 24  # positions are 24-bit fixed point; more would overflow
SH_DIM_FOR_DEGREE = {0: 0, 1: 3, 2: 8, 3: 15}
_DEGREE_FOR_SH_DIM = {0: 0, 3: 1, 8: 2, 15: 3}

_INV_SQRT2 = np.float32(0.7071067811865476)
_C_MASK = np.uint32((1 << 9) - 1)  # 9-bit magnitude mask for smallest-three quats
_EPS = np.float32(1e-6)


# ======================================================================================
# Reading
# ======================================================================================


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _decode_spz_kernel(
    buf: np.ndarray,  # [payload] uint8, full SPZ payload (read-only)
    n: int,
    sh_dim: int,
    frac_bits: int,
    uses_st: bool,
    rot_stride: int,
    means: np.ndarray,  # [N, 3] out
    scales: np.ndarray,  # [N, 3] out
    quats: np.ndarray,  # [N, 4] out, wxyz
    opac: np.ndarray,  # [N] out
    sh0: np.ndarray,  # [N, 3] out
    shN: np.ndarray,  # noqa: N803  # [N, sh_dim, 3] out
) -> None:
    """Fused per-Gaussian SPZ decode (single parallel pass, no intermediates)."""
    alpha_ofs = 9 * n
    color_ofs = 10 * n
    scale_ofs = 13 * n
    rot_ofs = 16 * n
    sh_ofs = 16 * n + rot_stride * n

    inv_frac = np.float32(1.0) / np.float32(1 << frac_bits)

    for i in numba.prange(n):
        # --- positions: 24-bit signed fixed point ---
        p = i * 9
        for j in range(3):
            v = (
                np.int32(buf[p + j * 3])
                | (np.int32(buf[p + j * 3 + 1]) << 8)
                | (np.int32(buf[p + j * 3 + 2]) << 16)
            )
            if v >= 8388608:  # 0x800000: sign bit set
                v -= 16777216  # 0x1000000
            means[i, j] = np.float32(v) * inv_frac

        # --- scales: byte / 16 - 10 (log space) ---
        s = scale_ofs + i * 3
        for j in range(3):
            scales[i, j] = np.float32(buf[s + j]) / np.float32(16.0) - np.float32(10.0)

        # --- rotation -> unit quaternion in wxyz ---
        qx = np.float32(0.0)
        qy = np.float32(0.0)
        qz = np.float32(0.0)
        qw = np.float32(0.0)
        if uses_st:
            r = rot_ofs + i * 4
            packed = (
                np.uint32(buf[r])
                | (np.uint32(buf[r + 1]) << 8)
                | (np.uint32(buf[r + 2]) << 16)
                | (np.uint32(buf[r + 3]) << 24)
            )
            i_largest = (packed >> 30) & np.uint32(3)
            work = packed
            ss = np.float32(0.0)
            for axis in range(3, -1, -1):
                if np.uint32(axis) != i_largest:
                    mag = work & _C_MASK
                    negbit = (work >> 9) & np.uint32(1)
                    work = work >> 10
                    val = _INV_SQRT2 * (np.float32(mag) / np.float32(_C_MASK))
                    if negbit == np.uint32(1):
                        val = -val
                    if axis == 0:
                        qx = val
                    elif axis == 1:
                        qy = val
                    elif axis == 2:
                        qz = val
                    else:
                        qw = val
                    ss += val * val
            large = np.float32(math.sqrt(max(np.float32(0.0), np.float32(1.0) - ss)))
            if i_largest == np.uint32(0):
                qx = large
            elif i_largest == np.uint32(1):
                qy = large
            elif i_largest == np.uint32(2):
                qz = large
            else:
                qw = large
        else:
            r = rot_ofs + i * 3
            qx = np.float32(buf[r]) / np.float32(127.5) - np.float32(1.0)
            qy = np.float32(buf[r + 1]) / np.float32(127.5) - np.float32(1.0)
            qz = np.float32(buf[r + 2]) / np.float32(127.5) - np.float32(1.0)
            qw = np.float32(
                math.sqrt(max(np.float32(0.0), np.float32(1.0) - qx * qx - qy * qy - qz * qz))
            )
        quats[i, 0] = qw  # SPZ stores xyzw; gsplat/PLY use wxyz
        quats[i, 1] = qx
        quats[i, 2] = qy
        quats[i, 3] = qz

        # --- alpha -> logit (inverse sigmoid, edge-clamped) ---
        a = np.float32(buf[alpha_ofs + i]) / np.float32(255.0)
        if a < _EPS:
            a = _EPS
        elif a > np.float32(1.0) - _EPS:
            a = np.float32(1.0) - _EPS
        opac[i] = np.float32(math.log(a / (np.float32(1.0) - a)))

        # --- color -> SH0 (wide RGB) ---
        c = color_ofs + i * 3
        for j in range(3):
            sh0[i, j] = (np.float32(buf[c + j]) / np.float32(255.0) - np.float32(0.5)) / np.float32(
                COLOR_SCALE
            )

        # --- higher-order SH: (byte - 128) / 128 ---
        if sh_dim > 0:
            sb = sh_ofs + i * sh_dim * 3
            for kk in range(sh_dim):
                for ch in range(3):
                    shN[i, kk, ch] = (
                        np.float32(buf[sb + kk * 3 + ch]) - np.float32(128.0)
                    ) / np.float32(128.0)


_warm_lock = threading.Lock()
_warm_state = {"done": False}


def _ensure_kernel_compiled() -> None:
    """Compile the decode kernel once, single-threaded (avoids cold-cache JIT races)."""
    if _warm_state["done"]:
        return
    with _warm_lock:
        if _warm_state["done"]:
            return
        dummy = np.zeros(9 + 1 + 3 + 3 + 4, dtype=np.uint8)
        _decode_spz_kernel(
            dummy,
            1,
            0,
            12,
            True,
            4,
            np.empty((1, 3), np.float32),
            np.empty((1, 3), np.float32),
            np.empty((1, 4), np.float32),
            np.empty(1, np.float32),
            np.empty((1, 3), np.float32),
            np.empty((1, 1, 3), np.float32),
        )
        _warm_state["done"] = True


def read_spz(file_path: str | Path) -> GSData:
    """Read a Niantic SPZ file into a :class:`GSData` in PLY format.

    Output conventions (matching :func:`gsply.plyread`):
        means linear, scales log-space, quats unit wxyz, opacities logit-space,
        sh0 SH DC coefficients, shN higher-order SH ``[N, K, 3]``.

    Args:
        file_path: Path to the ``.spz`` file.

    Returns:
        GSData populated with the decoded Gaussians (PLY format).

    Raises:
        ValueError: If the file is not a valid legacy-gzip SPZ container.
    """
    file_path = Path(file_path)
    compressed = file_path.read_bytes()  # file errors propagate as OSError
    try:
        raw = _gunzip(compressed)
    except Exception as exc:
        raise ValueError(
            f"Could not gunzip SPZ (corrupt, or an unsupported ZSTD/NGSP v4 container?): "
            f"{file_path}"
        ) from exc
    if len(raw) < 16:
        raise ValueError(f"SPZ file too small ({len(raw)} bytes): {file_path}")

    magic, version, num_points, sh_degree, fractional_bits, _flags, _reserved = struct.unpack(
        "<IIIBBBB", raw[:16]
    )
    if magic != NGSP_MAGIC:
        raise ValueError(f"Not an SPZ file (magic 0x{magic:08x}): {file_path}")
    if version not in (1, 2, 3):
        raise ValueError(
            f"Unsupported SPZ version {version} (only legacy gzip v1-3 supported): {file_path}"
        )
    if sh_degree > SH_MAX_DEGREE:
        raise ValueError(f"Unsupported SH degree {sh_degree}: {file_path}")
    if not 1 <= fractional_bits <= MAX_FRACTIONAL_BITS:
        raise ValueError(
            f"Invalid SPZ fractional_bits {fractional_bits} "
            f"(expected 1-{MAX_FRACTIONAL_BITS}): {file_path}"
        )

    n = int(num_points)
    sh_dim = SH_DIM_FOR_DEGREE[int(sh_degree)]
    uses_smallest_three = version >= 3
    rot_stride = 4 if uses_smallest_three else 3

    expected = (9 + 1 + 3 + 3 + rot_stride + sh_dim * 3) * n
    available = len(raw) - 16
    if available < expected:
        raise ValueError(
            f"SPZ payload too small: have {available} need {expected} "
            f"(v{version}, N={n}, sh_dim={sh_dim}): {file_path}"
        )
    # Tolerate trailing extension bytes (header flag 0x2) — read only the sections.
    payload = np.frombuffer(raw, dtype=np.uint8, count=expected, offset=16)

    _ensure_kernel_compiled()
    means = np.empty((n, 3), dtype=np.float32)
    scales = np.empty((n, 3), dtype=np.float32)
    quats = np.empty((n, 4), dtype=np.float32)
    opac = np.empty(n, dtype=np.float32)
    sh0 = np.empty((n, 3), dtype=np.float32)
    shN = np.empty((n, max(sh_dim, 1), 3), dtype=np.float32)  # noqa: N806

    _decode_spz_kernel(
        payload,
        n,
        sh_dim,
        int(fractional_bits),
        uses_smallest_three,
        rot_stride,
        means,
        scales,
        quats,
        opac,
        sh0,
        shN,
    )

    return GSData.from_arrays(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opac,
        sh0=sh0,
        shN=shN if sh_dim > 0 else None,
        format="ply",
    )


# ======================================================================================
# Writing
# ======================================================================================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _pack_quats_smallest_three(quats_xyzw: np.ndarray) -> np.ndarray:
    """Pack unit quaternions (xyzw) into SPZ v3 smallest-three uint8[N, 4].

    Stores the index of the largest component (2 bits) and the other three as
    9-bit magnitude + sign, with the largest made positive (sign is implied).
    """
    n = quats_xyzw.shape[0]
    q = quats_xyzw / np.linalg.norm(quats_xyzw, axis=1, keepdims=True)
    i_largest = np.argmax(np.abs(q), axis=1)
    sign = np.sign(q[np.arange(n), i_largest])
    sign[sign == 0] = 1.0
    q = q * sign[:, None]  # make the largest component positive

    c_mask = (1 << 9) - 1
    sqrt1_2 = 1.0 / np.sqrt(2.0)
    packed = np.zeros(n, dtype=np.uint32)
    for axis in range(4):
        keep = i_largest != axis
        val = q[:, axis]
        mag = np.clip(np.round(c_mask * np.abs(val) / sqrt1_2), 0, c_mask).astype(np.uint32)
        negbit = (val < 0).astype(np.uint32)
        packed = np.where(keep, (packed << 10) | (negbit << 9) | mag, packed)
    packed = packed | (i_largest.astype(np.uint32) << 30)

    out = np.empty((n, 4), dtype=np.uint8)
    out[:, 0] = packed & 0xFF
    out[:, 1] = (packed >> 8) & 0xFF
    out[:, 2] = (packed >> 16) & 0xFF
    out[:, 3] = (packed >> 24) & 0xFF
    return out


def write_spz(file_path: str | Path, data: GSData, *, fractional_bits: int = 12) -> None:
    """Write a :class:`GSData` to a Niantic SPZ file (gzip v3, smallest-three quats).

    The input is interpreted in PLY format (as produced by :func:`plyread` /
    :func:`read_spz`): means linear, scales log-space, quats unit wxyz, opacities
    logit-space, sh0 SH DC coefficients, shN ``[N, K, 3]``.

    Args:
        file_path: Output ``.spz`` path.
        data: Gaussians to encode.
        fractional_bits: Position fixed-point precision (default 12 = ~0.24mm).
    """
    if not 1 <= fractional_bits <= MAX_FRACTIONAL_BITS:
        raise ValueError(f"fractional_bits must be 1-{MAX_FRACTIONAL_BITS}, got {fractional_bits}")

    means = np.ascontiguousarray(data.means, dtype=np.float32)
    scales = np.ascontiguousarray(data.scales, dtype=np.float32)
    quats = np.ascontiguousarray(data.quats, dtype=np.float32)  # wxyz
    opac = np.ascontiguousarray(data.opacities, dtype=np.float32).reshape(-1)
    sh0 = np.ascontiguousarray(data.sh0, dtype=np.float32)
    n = means.shape[0]

    shN = data.shN  # noqa: N806
    if shN is None or np.asarray(shN).size == 0:
        sh_dim = 0
    else:
        sh_dim = int(np.asarray(shN).shape[1])
        if sh_dim not in _DEGREE_FOR_SH_DIM:
            raise ValueError(f"Unsupported shN coefficient count {sh_dim} (need 3/8/15)")
    sh_degree = _DEGREE_FOR_SH_DIM[sh_dim]

    # positions: signed 24-bit fixed point
    scale = float(1 << fractional_bits)
    fixed = np.round(means * scale).astype(np.int64)
    fixed = np.clip(fixed, -(1 << 23), (1 << 23) - 1).astype(np.int32) & 0xFFFFFF
    pos_bytes = np.empty((n, 3, 3), dtype=np.uint8)
    pos_bytes[..., 0] = fixed & 0xFF
    pos_bytes[..., 1] = (fixed >> 8) & 0xFF
    pos_bytes[..., 2] = (fixed >> 16) & 0xFF

    alpha = np.clip(np.round(_sigmoid(opac) * 255.0), 0, 255).astype(np.uint8)
    color = np.clip(np.round(sh0 * (COLOR_SCALE * 255.0) + 127.5), 0, 255).astype(np.uint8)
    scl = np.clip(np.round((scales + 10.0) * 16.0), 0, 255).astype(np.uint8)
    rot = _pack_quats_smallest_three(np.roll(quats, shift=-1, axis=1))  # wxyz -> xyzw

    sh_blob = b""
    if sh_dim > 0:
        # shN[i] is [K, 3]; row-major flatten gives the kk*3+ch byte order the reader expects.
        sh_flat = np.ascontiguousarray(shN, dtype=np.float32).reshape(n, sh_dim * 3)
        sh_bytes = np.clip(np.round(sh_flat * 128.0 + 128.0), 0, 255).astype(np.uint8)
        sh_blob = sh_bytes.tobytes()

    header = struct.pack("<IIIBBBB", NGSP_MAGIC, 3, n, sh_degree, fractional_bits, 0, 0)
    payload = (
        header
        + pos_bytes.tobytes()
        + alpha.tobytes()
        + color.tobytes()
        + scl.tobytes()
        + rot.tobytes()
        + sh_blob
    )
    Path(file_path).write_bytes(_gzip_compress(payload))
