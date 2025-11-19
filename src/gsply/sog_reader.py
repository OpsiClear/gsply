"""SOG (Splat Ordering Grid) format reader - optimized implementation.

SOG format uses WebP images to store quantized Gaussian splatting data with
codebook-based compression for scales and colors.

Returns GSData container (same as plyread) for consistent API across all formats.

Format:
    - meta.json: Metadata (bounds, codebooks, file references)
    - means_l.webp, means_u.webp: Position data (16-bit split into low/high bytes)
    - quats.webp: Quaternion data (packed smallest-three encoding)
    - scales.webp: Scale labels + codebook (256 centroids from k-means)
    - sh0.webp: Color labels + codebook (256 centroids) + opacity
    - shN_centroids.webp, shN_labels.webp: Optional higher-order SH (if present)

Can be:
    - .sog ZIP bundle (all files in one archive)
    - Folder with separate files
    - Bytes (in-memory ZIP extraction)
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import numba
import numpy as np
from numba import jit

from gsply.gsdata import DataFormat, GSData, _create_format_dict, _get_sh_order_format

# Use imagecodecs (fastest WebP decoder)
try:
    import imagecodecs  # noqa: PLC0415
except ImportError:
    imagecodecs = None

logger = logging.getLogger(__name__)

# Constants
SH_C0 = 0.28209479177387814
SQRT2 = np.sqrt(2.0)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _decode_means_jit(
    lo_rgba: np.ndarray, hi_rgba: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled means decoding from low/high byte textures.

    :param lo_rgba: (N*4,) uint8 RGBA data from means_l.webp
    :param hi_rgba: (N*4,) uint8 RGBA data from means_u.webp
    :param count: Number of Gaussians
    :returns: Tuple of (xs, ys, zs) as uint16 arrays
    """
    xs = np.empty(count, dtype=np.uint16)
    ys = np.empty(count, dtype=np.uint16)
    zs = np.empty(count, dtype=np.uint16)

    for i in numba.prange(count):
        o = i * 4
        xs[i] = lo_rgba[o + 0] | (hi_rgba[o + 0] << 8)
        ys[i] = lo_rgba[o + 1] | (hi_rgba[o + 1] << 8)
        zs[i] = lo_rgba[o + 2] | (hi_rgba[o + 2] << 8)

    return xs, ys, zs


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _inv_log_transform_jit(values: np.ndarray) -> np.ndarray:
    """JIT-compiled inverse log transform.

    Inverse of logTransform(x) = sign(x) * ln(|x| + 1)

    :param values: Input values in log space
    :returns: Values in original space
    """
    result = np.empty_like(values)
    for i in numba.prange(len(values)):
        v = values[i]
        a = abs(v)
        e = np.exp(a) - 1.0
        result[i] = -e if v < 0 else e
    return result


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _unpack_quats_jit(
    rgba: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled quaternion unpacking from packed format.

    :param rgba: (N*4,) uint8 RGBA data from quats.webp
    :param count: Number of Gaussians
    :returns: Tuple of (r0, r1, r2, r3) quaternion components
    """
    r0 = np.empty(count, dtype=np.float32)
    r1 = np.empty(count, dtype=np.float32)
    r2 = np.empty(count, dtype=np.float32)
    r3 = np.empty(count, dtype=np.float32)

    sqrt2_inv = 1.0 / SQRT2

    for i in numba.prange(count):
        o = i * 4
        tag = rgba[o + 3]

        if tag < 252 or tag > 255:
            r0[i] = 0.0
            r1[i] = 0.0
            r2[i] = 0.0
            r3[i] = 1.0
            continue

        max_comp = tag - 252
        a = (rgba[o + 0] / 255.0) * 2.0 - 1.0
        b = (rgba[o + 1] / 255.0) * 2.0 - 1.0
        c = (rgba[o + 2] / 255.0) * 2.0 - 1.0

        comps = np.zeros(4, dtype=np.float32)
        if max_comp == 0:
            comps[1] = a * sqrt2_inv
            comps[2] = b * sqrt2_inv
            comps[3] = c * sqrt2_inv
        elif max_comp == 1:
            comps[0] = a * sqrt2_inv
            comps[2] = b * sqrt2_inv
            comps[3] = c * sqrt2_inv
        elif max_comp == 2:
            comps[0] = a * sqrt2_inv
            comps[1] = b * sqrt2_inv
            comps[3] = c * sqrt2_inv
        else:
            comps[0] = a * sqrt2_inv
            comps[1] = b * sqrt2_inv
            comps[2] = c * sqrt2_inv

        t = 1.0 - (
            comps[0] * comps[0] + comps[1] * comps[1] + comps[2] * comps[2] + comps[3] * comps[3]
        )
        comps[max_comp] = np.sqrt(max(0.0, t))

        r0[i] = comps[0]
        r1[i] = comps[1]
        r2[i] = comps[2]
        r3[i] = comps[3]

    return r0, r1, r2, r3


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _decode_scales_jit(
    rgba: np.ndarray, codebook: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled scale decoding using codebook lookup.

    :param rgba: (N*4,) uint8 RGBA data from scales.webp
    :param codebook: (256,) float32 codebook values
    :param count: Number of Gaussians
    :returns: Tuple of (sx, sy, sz) scale components
    """
    sx = np.empty(count, dtype=np.float32)
    sy = np.empty(count, dtype=np.float32)
    sz = np.empty(count, dtype=np.float32)

    for i in numba.prange(count):
        o = i * 4
        sx[i] = codebook[rgba[o + 0]]
        sy[i] = codebook[rgba[o + 1]]
        sz[i] = codebook[rgba[o + 2]]

    return sx, sy, sz


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _decode_colors_jit(
    rgba: np.ndarray, codebook: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled color and opacity decoding.

    :param rgba: (N*4,) uint8 RGBA data from sh0.webp
    :param codebook: (256,) float32 codebook values for colors
    :param count: Number of Gaussians
    :returns: Tuple of (sh0_r, sh0_g, sh0_b, opacities)
    """
    sh0_r = np.empty(count, dtype=np.float32)
    sh0_g = np.empty(count, dtype=np.float32)
    sh0_b = np.empty(count, dtype=np.float32)
    opacities = np.empty(count, dtype=np.float32)

    epsilon = 1e-6
    for i in numba.prange(count):
        o = i * 4
        sh0_r[i] = codebook[rgba[o + 0]]
        sh0_g[i] = codebook[rgba[o + 1]]
        sh0_b[i] = codebook[rgba[o + 2]]

        y = rgba[o + 3] / 255.0
        e = min(1.0 - epsilon, max(epsilon, y))
        opacities[i] = np.log(e / (1.0 - e))

    return sh0_r, sh0_g, sh0_b, opacities


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _decode_shn_jit(
    labels_rgba: np.ndarray,
    centroids_rgba: np.ndarray,
    codebook: np.ndarray,
    count: int,
    sh_coeffs: int,
    palette_count: int,
    centroids_width: int,
) -> np.ndarray:
    """JIT-compiled SHN decoding from labels and centroids textures.

    :param labels_rgba: (N*4,) uint8 RGBA data from shN_labels.webp
    :param centroids_rgba: (W*H*4,) uint8 RGBA data from shN_centroids.webp
    :param codebook: (256,) float32 codebook values
    :param count: Number of Gaussians
    :param sh_coeffs: Number of SH coefficients per Gaussian
    :param palette_count: Number of palette entries
    :param centroids_width: Width of centroids texture
    :returns: (N, sh_coeffs, 3) float32 SHN coefficients
    """
    shn = np.zeros((count, sh_coeffs, 3), dtype=np.float32)
    codebook_len = len(codebook)

    for i in numba.prange(count):
        o = i * 4
        label = labels_rgba[o] | (labels_rgba[o + 1] << 8)

        if label >= palette_count:
            continue

        # Get centroid pixel for this label and coefficient
        for j in range(sh_coeffs):
            cx = (label % 64) * sh_coeffs + j
            cy = label // 64

            idx = (cy * centroids_width + cx) * 4
            if idx < len(centroids_rgba):
                lr = centroids_rgba[idx]
                lg = centroids_rgba[idx + 1]
                lb = centroids_rgba[idx + 2]

                shn[i, j, 0] = codebook[lr] if lr < codebook_len else 0.0
                shn[i, j, 1] = codebook[lg] if lg < codebook_len else 0.0
                shn[i, j, 2] = codebook[lb] if lb < codebook_len else 0.0

    return shn


def _load_webp_image(data: bytes) -> tuple[np.ndarray, int, int]:
    """Load WebP image and return RGBA data using imagecodecs.

    :param data: WebP image bytes
    :returns: Tuple of (rgba_data, width, height)
    :raises ImportError: If imagecodecs is not installed
    """
    if imagecodecs is None:
        raise ImportError(
            "SOG format requires imagecodecs for WebP decoding.\n"
            "Install with: pip install gsply[sogs]\n"
            "Or directly: pip install imagecodecs"
        )

    rgba = imagecodecs.webp_decode(data)
    # Ensure RGBA format
    if rgba.shape[2] == 3:
        rgba = np.concatenate(
            [rgba, np.full((rgba.shape[0], rgba.shape[1], 1), 255, dtype=np.uint8)], axis=2
        )
    height, width = rgba.shape[:2]
    rgba_flat = rgba.reshape(-1)
    return rgba_flat, width, height


def sogread(file_path: str | Path | bytes) -> GSData:
    """Read SOG (Splat Ordering Grid) format file.

    Returns GSData container (same as plyread) for consistent API.
    Supports both .sog ZIP bundles and folders with separate files.
    Can also accept bytes directly for in-memory ZIP extraction.

    :param file_path: Path to .sog file, folder containing SOG files, or bytes (ZIP data)
    :returns: GSData container with Gaussian parameters (same container as plyread)
    :raises ImportError: If imagecodecs is not installed
    :raises ValueError: If file format is invalid or missing required files

    Example:
        >>> # From file path - returns GSData (same as plyread)
        >>> data = sogread("model.sog")
        >>> print(f"Loaded {len(data)} Gaussians")
        >>> positions = data.means  # Same API as GSData from plyread
        >>>
        >>> # From bytes (in-memory)
        >>> with open("model.sog", "rb") as f:
        ...     sog_bytes = f.read()
        >>> data = sogread(sog_bytes)  # Returns GSData
    """
    entries: dict[str, bytes] | None = None
    path_obj: Path | None = None

    # Handle bytes input (in-memory ZIP)
    if isinstance(file_path, bytes):
        with zipfile.ZipFile(io.BytesIO(file_path), "r") as zf:
            entries = {name: zf.read(name) for name in zf.namelist()}
    else:
        # Handle file path
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise ValueError(f"SOG file or folder not found: {path_obj}")

        if path_obj.is_file() and path_obj.suffix.lower() == ".sog":
            with zipfile.ZipFile(path_obj, "r") as zf:
                entries = {name: zf.read(name) for name in zf.namelist()}

    def load(name: str) -> bytes:
        if entries is not None:
            # ZIP bundle (from file or bytes)
            if name not in entries:
                raise ValueError(f"Missing entry '{name}' in SOG bundle")
            return entries[name]
        # Folder mode (requires path_obj)
        if path_obj is None:
            raise ValueError("Cannot load from folder: file_path must be a Path, not bytes")
        file_full_path = path_obj / name
        if not file_full_path.exists():
            raise ValueError(f"Missing file '{name}' in SOG folder")
        return file_full_path.read_bytes()

    meta_bytes = load("meta.json")
    meta = json.loads(meta_bytes.decode("utf-8"))
    count = meta["count"]

    means_mins = np.array(meta["means"]["mins"], dtype=np.float32)
    means_maxs = np.array(meta["means"]["maxs"], dtype=np.float32)
    means_ranges = means_maxs - means_mins
    means_ranges = np.where(means_ranges == 0, np.ones_like(means_ranges), means_ranges)

    means_lo_data = load(meta["means"]["files"][0])
    means_hi_data = load(meta["means"]["files"][1])
    means_lo_rgba, width_lo, height_lo = _load_webp_image(means_lo_data)
    means_hi_rgba, width_hi, height_hi = _load_webp_image(means_hi_data)

    if width_lo * height_lo < count or width_hi * height_hi < count:
        raise ValueError("SOG means textures too small for count")

    xs, ys, zs = _decode_means_jit(means_lo_rgba, means_hi_rgba, count)

    # Optimized: avoid intermediate astype by using float32 division directly
    means_log = np.empty((count, 3), dtype=np.float32)
    inv_65535 = 1.0 / 65535.0
    means_log[:, 0] = means_mins[0] + means_ranges[0] * (xs.astype(np.float32) * inv_65535)
    means_log[:, 1] = means_mins[1] + means_ranges[1] * (ys.astype(np.float32) * inv_65535)
    means_log[:, 2] = means_mins[2] + means_ranges[2] * (zs.astype(np.float32) * inv_65535)

    means = _inv_log_transform_jit(means_log.flatten()).reshape(count, 3)

    quats_data = load(meta["quats"]["files"][0])
    quats_rgba, qw, qh = _load_webp_image(quats_data)
    if qw * qh < count:
        raise ValueError("SOG quats texture too small for count")

    quats = np.empty((count, 4), dtype=np.float32)
    r0, r1, r2, r3 = _unpack_quats_jit(quats_rgba, count)
    quats[:, 0] = r0
    quats[:, 1] = r1
    quats[:, 2] = r2
    quats[:, 3] = r3

    scales_data = load(meta["scales"]["files"][0])
    scales_rgba, sw, sh = _load_webp_image(scales_data)
    if sw * sh < count:
        raise ValueError("SOG scales texture too small for count")

    scales_codebook = np.array(meta["scales"]["codebook"], dtype=np.float32)
    scales = np.empty((count, 3), dtype=np.float32)
    sx, sy, sz = _decode_scales_jit(scales_rgba, scales_codebook, count)
    scales[:, 0] = sx
    scales[:, 1] = sy
    scales[:, 2] = sz

    sh0_data = load(meta["sh0"]["files"][0])
    sh0_rgba, cw, ch = _load_webp_image(sh0_data)
    if cw * ch < count:
        raise ValueError("SOG sh0 texture too small for count")

    sh0_codebook = np.array(meta["sh0"]["codebook"], dtype=np.float32)
    sh0_r, sh0_g, sh0_b, opacities = _decode_colors_jit(sh0_rgba, sh0_codebook, count)

    sh0 = np.empty((count, 3), dtype=np.float32)
    sh0[:, 0] = sh0_r
    sh0[:, 1] = sh0_g
    sh0[:, 2] = sh0_b

    shn = None  # noqa: N806
    sh_degree = 0  # Default to SH0
    if "shN" in meta:
        shn_meta = meta["shN"]  # noqa: N806
        bands = shn_meta["bands"]
        sh_degree = bands  # bands directly maps to SH degree (0, 1, 2, 3)
        sh_coeffs = [0, 3, 8, 15][bands]
        palette_count = shn_meta["count"]

        if sh_coeffs > 0:
            centroids_data = load(shn_meta["files"][0])
            labels_data = load(shn_meta["files"][1])
            centroids_rgba, cw, ch = _load_webp_image(centroids_data)
            labels_rgba, lw, lh = _load_webp_image(labels_data)

            if lw * lh < count:
                raise ValueError("SOG shN labels texture too small for count")

            codebook = np.array(shn_meta["codebook"], dtype=np.float32)

            # Optimized JIT-compiled SHN decoding
            centroids_width = 64 * sh_coeffs
            shn = _decode_shn_jit(  # noqa: N806
                labels_rgba,
                centroids_rgba,
                codebook,
                count,
                sh_coeffs,
                palette_count,
                centroids_width,
            )  # noqa: N806
    else:
        shn = np.zeros((count, 0, 3), dtype=np.float32)  # noqa: N806

    source_name = (
        path_obj.name
        if path_obj is not None
        else f"<{len(file_path)} bytes>"
        if isinstance(file_path, bytes)
        else str(file_path)
    )
    logger.debug(f"[SOG Read] Loaded {count:,} Gaussians from {source_name}")

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shn,  # noqa: N806
        masks=np.ones(count, dtype=bool),
        _base=None,
        _format=_create_format_dict(
            scales=DataFormat.SCALES_LINEAR,
            opacities=DataFormat.OPACITIES_PLY,
            sh0=DataFormat.SH0_SH,
            sh_order=_get_sh_order_format(sh_degree),
            means=DataFormat.MEANS_RAW,
            quats=DataFormat.QUATS_RAW,
        ),  # SOG uses mixed format: linear scales but logit opacities
    )


__all__ = ["sogread"]
