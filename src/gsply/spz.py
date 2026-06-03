"""
Read and write Niantic SPZ Gaussian-splat files.

Two on-disk containers share the same per-attribute quantization ("packed"
sections); only the framing differs:

* **Legacy gzip (v1/v2/v3)** — the whole payload is one gzip stream:

      Header (16B): magic u32 ("NGSP") | version u32 | num_points u32 |
                    sh_degree u8 | fractional_bits u8 | flags u8 | reserved u8
      Payload (contiguous sections): positions | alphas | colors | scales |
                                     rotations | sh

* **NGSP v4** — an uncompressed 32B header + optional extensions + a TOC, then
  each attribute section as its own **zstd** stream (no gzip):

      Header (32B): magic u32 | version u32 | num_points u32 | sh_degree u8 |
                    fractional_bits u8 | flags u8 | num_streams u8 |
                    toc_byte_offset u32 | reserved[12]
      TOC: num_streams * (compressed_size u64, uncompressed_size u64)
      Streams: zstd(positions), zstd(alphas), ... in that order (empty skipped)

Sections (identical in both containers):

    positions  9*N bytes   (xyz as 24-bit signed fixed point, little-endian)
    alphas     N   bytes   (sigmoid(opacity) * 255)
    colors     3*N bytes   (sh0 wide-RGB:  byte = sh0 * 0.15 * 255 + 127.5)
    scales     3*N bytes   (byte = (log_scale + 10) * 16)
    rotations  R*N bytes   (v<=2: 3 bytes xyz; v>=3: 4 bytes smallest-three)
    sh         K*3*N bytes (per coeff: byte = sh * 128 + 128)

Reading uses a single fused Numba kernel (one parallel pass over the payload, no
intermediate allocations) and ISA-L's igzip when available (falls back to stdlib
gzip). v4 needs the ``zstandard`` package (``gsply[spz]``). Writing defaults to
gzip v3 (smallest-three quats); pass ``version=4`` for the NGSP zstd container.

Reference: github.com/nianticlabs/spz (load-spz.cc).
"""

from __future__ import annotations

import gzip
import math
import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numba
import numpy as np
from numba import jit

from gsply import _backend
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
        if len(raw) < _GZIP_PARALLEL_MIN_BYTES:
            return gzip.compress(raw, compresslevel=_GZIP_COMPRESSION_LEVEL)
        return _gzip_compress_parallel(raw)


try:  # zstd is required only for the NGSP v4 container.
    import zstandard as _zstd

    _HAS_ZSTD = True

    def _zstd_compress(data: bytes, level: int, threads: int = -1) -> bytes:
        # threads=-1 => all logical CPUs (intra-frame MT); 0 => single-threaded.
        out: bytes = _zstd.ZstdCompressor(level=level, threads=threads).compress(data)
        return out

    def _zstd_decompress(data: bytes, size: int) -> bytes:
        out: bytes = _zstd.ZstdDecompressor().decompress(data, max_output_size=size)
        return out

except ImportError:  # pragma: no cover - environment-dependent
    _HAS_ZSTD = False


NGSP_MAGIC = 0x5053474E
NGSP_HEADER_SIZE = 32  # v4 uncompressed header
COLOR_SCALE = 0.15
SH_MAX_DEGREE = 3
MAX_FRACTIONAL_BITS = 24  # positions are 24-bit fixed point; more would overflow
LATEST_SPZ_VERSION = 4
MIN_ZSTD_VERSION = 4  # versions >= this use the NGSP zstd container
DEFAULT_ZSTD_LEVEL = 12  # matches the Niantic reference
SH_DIM_FOR_DEGREE = {0: 0, 1: 3, 2: 8, 3: 15}
_DEGREE_FOR_SH_DIM = {0: 0, 3: 1, 8: 2, 15: 3}

_INV_SQRT2 = np.float32(0.7071067811865476)
_C_MASK = np.uint32((1 << 9) - 1)  # 9-bit magnitude mask for smallest-three quats
_EPS = np.float32(1e-6)
_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
_GZIP_PARALLEL_MIN_BYTES = 1 << 20
_GZIP_PARALLEL_BLOCK_BYTES = 1 << 18
_GZIP_COMPRESSION_LEVEL = 6  # matches the C++ backend default more closely than gzip's level 9


class _SpzPayload(NamedTuple):
    """Validated uncompressed SPZ section payload and decode metadata."""

    payload: np.ndarray
    n: int
    sh_degree: int
    sh_dim: int
    fractional_bits: int
    uses_smallest_three: bool
    rot_stride: int


def _gzip_compress_parallel(
    raw: bytes,
    *,
    level: int = _GZIP_COMPRESSION_LEVEL,
    block_size: int = _GZIP_PARALLEL_BLOCK_BYTES,
) -> bytes:
    """Compress as one gzip member while deflating independent blocks in parallel.

    Each non-final block is flushed at a byte boundary so concatenating the raw
    deflate outputs stays a single valid deflate stream. The gzip wrapper is
    assembled once around the combined body, preserving strict single-member
    compatibility with Niantic's loader.
    """
    chunk_count = (len(raw) + block_size - 1) // block_size
    if chunk_count <= 1:
        return gzip.compress(raw, compresslevel=level)

    view = memoryview(raw)

    def compress_chunk(idx: int) -> bytes:
        start = idx * block_size
        chunk = view[start : min(start + block_size, len(raw))]
        compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
        flush_mode = zlib.Z_FINISH if idx == chunk_count - 1 else zlib.Z_SYNC_FLUSH
        return compressor.compress(chunk) + compressor.flush(flush_mode)

    workers = min(chunk_count, max(1, min(16, os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        body = b"".join(ex.map(compress_chunk, range(chunk_count)))

    trailer = struct.pack("<II", zlib.crc32(raw) & 0xFFFFFFFF, len(raw) & 0xFFFFFFFF)
    return _GZIP_HEADER + body + trailer


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


def _section_layout(n: int, sh_dim: int, rot_stride: int) -> list[tuple[str, int]]:
    """Per-section uncompressed byte sizes, in the canonical stream order."""
    return [
        ("positions", 9 * n),
        ("alphas", 1 * n),
        ("colors", 3 * n),
        ("scales", 3 * n),
        ("rotations", rot_stride * n),
        ("sh", sh_dim * 3 * n),
    ]


def _decode_to_gsdata(
    payload: np.ndarray, n: int, sh_dim: int, frac_bits: int, uses_st: bool, rot_stride: int
) -> GSData:
    """Run the fused decode kernel over a contiguous sections buffer -> GSData."""
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
        int(frac_bits),
        uses_st,
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


def _read_legacy_gzip_payload(compressed: bytes, file_path: Path) -> _SpzPayload:
    """Parse a legacy gzip-container SPZ into packed sections."""
    try:
        raw = _gunzip(compressed)
    except Exception as exc:
        raise ValueError(f"Could not gunzip SPZ (corrupt container?): {file_path}") from exc
    if len(raw) < 16:
        raise ValueError(f"SPZ file too small ({len(raw)} bytes): {file_path}")

    magic, version, num_points, sh_degree, fractional_bits, _flags, _reserved = struct.unpack(
        "<IIIBBBB", raw[:16]
    )
    if magic != NGSP_MAGIC:
        raise ValueError(f"Not an SPZ file (magic 0x{magic:08x}): {file_path}")
    if version not in (1, 2, 3):
        raise ValueError(f"Unexpected legacy SPZ version {version}: {file_path}")
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
    return _SpzPayload(
        payload=payload,
        n=n,
        sh_degree=int(sh_degree),
        sh_dim=sh_dim,
        fractional_bits=int(fractional_bits),
        uses_smallest_three=uses_smallest_three,
        rot_stride=rot_stride,
    )


def _read_legacy_gzip(compressed: bytes, file_path: Path) -> GSData:
    """Read a legacy gzip-container SPZ (v1/v2/v3)."""
    parsed = _read_legacy_gzip_payload(compressed, file_path)
    return _decode_to_gsdata(
        parsed.payload,
        parsed.n,
        parsed.sh_dim,
        parsed.fractional_bits,
        parsed.uses_smallest_three,
        parsed.rot_stride,
    )


def _read_ngsp_v4_payload(raw_file: bytes, file_path: Path) -> _SpzPayload:
    """Parse an NGSP v4 container into packed sections."""
    if not _HAS_ZSTD:
        raise ValueError(
            f"Reading SPZ v4 (NGSP/zstd) requires the 'zstandard' package "
            f"(install 'gsply[spz]'): {file_path}"
        )
    if len(raw_file) < NGSP_HEADER_SIZE:
        raise ValueError(f"NGSP file too small ({len(raw_file)} bytes): {file_path}")

    magic, version, num_points, sh_degree, fractional_bits, _flags, num_streams, toc_off = (
        struct.unpack_from("<IIIBBBBI", raw_file, 0)
    )
    if magic != NGSP_MAGIC:
        raise ValueError(f"Not an SPZ file (magic 0x{magic:08x}): {file_path}")
    if not MIN_ZSTD_VERSION <= version <= LATEST_SPZ_VERSION:
        raise ValueError(f"Unsupported NGSP version {version}: {file_path}")
    if sh_degree > SH_MAX_DEGREE:
        raise ValueError(f"Unsupported SH degree {sh_degree}: {file_path}")
    if not 1 <= fractional_bits <= MAX_FRACTIONAL_BITS:
        raise ValueError(
            f"Invalid SPZ fractional_bits {fractional_bits} "
            f"(expected 1-{MAX_FRACTIONAL_BITS}): {file_path}"
        )

    n = int(num_points)
    sh_dim = SH_DIM_FOR_DEGREE[int(sh_degree)]
    rot_stride = 4  # v4 always uses smallest-three quaternions
    sections = [(name, sz) for name, sz in _section_layout(n, sh_dim, rot_stride) if sz > 0]
    if num_streams != len(sections):
        raise ValueError(
            f"NGSP stream count {num_streams} != expected {len(sections)} "
            f"(N={n}, sh_dim={sh_dim}): {file_path}"
        )

    toc_end = toc_off + num_streams * 16
    if toc_off < NGSP_HEADER_SIZE or toc_end > len(raw_file):
        raise ValueError(f"NGSP TOC out of bounds: {file_path}")

    # Compressed offsets are cumulative, so resolve each stream's slice serially,
    # then zstd-decompress them concurrently (zstandard releases the GIL).
    jobs: list[tuple[str, int, int, int]] = []  # (name, offset, csize, usize)
    offset = toc_end
    for i, (name, usize_expected) in enumerate(sections):
        csize, usize = struct.unpack_from("<QQ", raw_file, toc_off + i * 16)
        if usize != usize_expected:
            raise ValueError(
                f"NGSP stream '{name}' size {usize} != expected {usize_expected}: {file_path}"
            )
        if offset + csize > len(raw_file):
            raise ValueError(f"NGSP stream '{name}' overruns file: {file_path}")
        jobs.append((name, offset, csize, usize))
        offset += csize

    def _decode_stream(job: tuple[str, int, int, int]) -> np.ndarray:
        name, off, csize, usize = job
        chunk = _zstd_decompress(raw_file[off : off + csize], usize)
        if len(chunk) != usize:
            raise ValueError(f"NGSP stream '{name}' decompressed size mismatch: {file_path}")
        return np.frombuffer(chunk, dtype=np.uint8)

    with ThreadPoolExecutor(max(1, len(jobs))) as ex:
        parts = list(ex.map(_decode_stream, jobs))

    payload = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint8)
    return _SpzPayload(
        payload=payload,
        n=n,
        sh_degree=int(sh_degree),
        sh_dim=sh_dim,
        fractional_bits=int(fractional_bits),
        uses_smallest_three=True,
        rot_stride=rot_stride,
    )


def _read_ngsp_v4(raw_file: bytes, file_path: Path) -> GSData:
    """Read an NGSP v4 container (uncompressed header + TOC + per-section zstd streams)."""
    parsed = _read_ngsp_v4_payload(raw_file, file_path)
    return _decode_to_gsdata(
        parsed.payload,
        parsed.n,
        parsed.sh_dim,
        parsed.fractional_bits,
        parsed.uses_smallest_three,
        parsed.rot_stride,
    )


def _read_spz_payload(file_path: str | Path) -> _SpzPayload:
    """Read and validate an SPZ container, returning packed sections for decoding."""
    file_path = Path(file_path)
    raw_file = file_path.read_bytes()  # file errors propagate as OSError
    if len(raw_file) >= 2 and raw_file[0] == 0x1F and raw_file[1] == 0x8B:
        return _read_legacy_gzip_payload(raw_file, file_path)
    if len(raw_file) >= 4 and struct.unpack_from("<I", raw_file, 0)[0] == NGSP_MAGIC:
        return _read_ngsp_v4_payload(raw_file, file_path)
    raise ValueError(f"Not an SPZ file (unrecognized container): {file_path}")


def read_spz(file_path: str | Path) -> GSData:
    """Read a Niantic SPZ file into a :class:`GSData` in PLY format.

    Supports both containers: legacy gzip (v1/v2/v3) and NGSP v4 (zstd). Output
    conventions (matching :func:`gsply.plyread`): means linear, scales log-space,
    quats unit wxyz, opacities logit-space, sh0 SH DC, shN higher-order ``[N,K,3]``.

    Args:
        file_path: Path to the ``.spz`` file.

    Returns:
        GSData populated with the decoded Gaussians (PLY format).

    Raises:
        ValueError: If the file is not a recognized SPZ container (or v4 is
            requested without the ``zstandard`` package).
    """
    if _backend.active_backend() == "cpp":  # opt-in C++ backend (full SPZ parity)
        return _backend.dict_to_gsdata(_backend.cpp().read_spz(str(file_path)))
    parsed = _read_spz_payload(file_path)
    return _decode_to_gsdata(
        parsed.payload,
        parsed.n,
        parsed.sh_dim,
        parsed.fractional_bits,
        parsed.uses_smallest_three,
        parsed.rot_stride,
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


def _pack_sections(
    data: GSData, fractional_bits: int
) -> tuple[int, int, int, list[tuple[str, bytes]]]:
    """Quantize a GSData into the canonical SPZ sections (shared by both containers).

    Returns ``(sh_degree, sh_dim, n, sections)`` where ``sections`` is an ordered
    list of ``(name, bytes)`` for the non-empty attribute streams.
    """
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

    sections = [
        ("positions", pos_bytes.tobytes()),
        ("alphas", alpha.tobytes()),
        ("colors", color.tobytes()),
        ("scales", scl.tobytes()),
        ("rotations", rot.tobytes()),
    ]
    if sh_dim > 0:
        # shN[i] is [K, 3]; row-major flatten gives the kk*3+ch byte order the reader expects.
        sh_flat = np.ascontiguousarray(shN, dtype=np.float32).reshape(n, sh_dim * 3)
        sh_bytes = np.clip(np.round(sh_flat * 128.0 + 128.0), 0, 255).astype(np.uint8)
        sections.append(("sh", sh_bytes.tobytes()))

    return sh_degree, sh_dim, n, sections


def write_spz(
    file_path: str | Path,
    data: GSData,
    *,
    version: int = 3,
    fractional_bits: int = 12,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
) -> None:
    """Write a :class:`GSData` to a Niantic SPZ file.

    The input is interpreted in PLY format (as produced by :func:`plyread` /
    :func:`read_spz`): means linear, scales log-space, quats unit wxyz, opacities
    logit-space, sh0 SH DC coefficients, shN ``[N, K, 3]``.

    Args:
        file_path: Output ``.spz`` path.
        data: Gaussians to encode.
        version: ``3`` (default) writes the legacy gzip container with
            smallest-three quaternions; ``4`` writes the NGSP zstd container.
        fractional_bits: Position fixed-point precision (default 12 = ~0.24mm).
        zstd_level: zstd compression level for ``version=4`` (1..22, default 12).

    Raises:
        ValueError: For an unsupported ``version``, or if ``version=4`` is
            requested without the ``zstandard`` package installed.
    """
    if not 1 <= fractional_bits <= MAX_FRACTIONAL_BITS:
        raise ValueError(f"fractional_bits must be 1-{MAX_FRACTIONAL_BITS}, got {fractional_bits}")
    if version not in (3, 4):
        raise ValueError(f"write_spz supports version 3 (gzip) or 4 (zstd), got {version}")

    if _backend.active_backend() == "cpp":  # opt-in C++ backend (full SPZ parity)
        m, s, q, o, c0, cn = _backend.gsdata_ply_arrays(data)
        # gsply_cpp: level<0 => per-codec default (gzip 6 / zstd 12); v4 uses zstd_level.
        _backend.cpp().write_spz(
            str(file_path),
            m,
            s,
            q,
            o,
            c0,
            cn,
            fractional_bits,
            version,
            zstd_level if version == 4 else -1,
        )
        return

    sh_degree, _sh_dim, n, sections = _pack_sections(data, fractional_bits)

    if version == 3:
        header = struct.pack("<IIIBBBB", NGSP_MAGIC, 3, n, sh_degree, fractional_bits, 0, 0)
        payload = header + b"".join(body for _, body in sections)
        Path(file_path).write_bytes(_gzip_compress(payload))
        return

    # version == 4: NGSP zstd container
    if not _HAS_ZSTD:
        raise ValueError(
            "Writing SPZ v4 (NGSP/zstd) requires the 'zstandard' package (install 'gsply[spz]')"
        )
    # Hybrid parallelism: compress streams concurrently (zstandard releases the GIL);
    # the largest stream (SH) gets intra-frame workers so it isn't the lone long pole.
    bodies = [body for _, body in sections]
    big = max(range(len(bodies)), key=lambda i: len(bodies[i]))
    with ThreadPoolExecutor(max(1, len(bodies))) as ex:
        chunks = list(
            ex.map(
                lambda i: _zstd_compress(bodies[i], zstd_level, threads=-1 if i == big else 0),
                range(len(bodies)),
            )
        )
    num_streams = len(sections)
    toc_off = NGSP_HEADER_SIZE  # no extensions
    header = (
        struct.pack(
            "<IIIBBBBI", NGSP_MAGIC, 4, n, sh_degree, fractional_bits, 0, num_streams, toc_off
        )
        + b"\x00" * 12  # reserved
    )
    toc = b"".join(
        struct.pack("<QQ", len(chunks[i]), len(sections[i][1])) for i in range(num_streams)
    )
    Path(file_path).write_bytes(header + toc + b"".join(chunks))
