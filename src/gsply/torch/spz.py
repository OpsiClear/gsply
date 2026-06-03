"""GPU-oriented SPZ reading into :class:`GSTensor`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gsply.gsdata import create_ply_format
from gsply.spz import _C_MASK, _EPS, _INV_SQRT2, COLOR_SCALE, _read_spz_payload
from gsply.torch.gstensor import GSTensor


def _payload_to_tensor(payload: np.ndarray, device: torch.device) -> torch.Tensor:
    """Move packed SPZ bytes to the target device as a single tensor."""
    payload = np.ascontiguousarray(payload)
    if not payload.flags.writeable:
        payload = payload.copy()

    tensor = torch.from_numpy(payload)
    if device.type == "cuda":
        tensor = tensor.pin_memory().to(device=device, non_blocking=True)
    else:
        tensor = tensor.to(device=device)
    return tensor


def _decode_positions(payload: torch.Tensor, n: int, fractional_bits: int) -> torch.Tensor:
    pos = payload[: 9 * n].view(n, 3, 3).to(torch.int64)
    fixed = pos[:, :, 0] | (pos[:, :, 1] << 8) | (pos[:, :, 2] << 16)
    fixed = torch.where(fixed >= 8_388_608, fixed - 16_777_216, fixed)
    return fixed.to(torch.float32) * (1.0 / float(1 << fractional_bits))


def _decode_smallest_three_quats(rot: torch.Tensor) -> torch.Tensor:
    bytes_i64 = rot.to(torch.int64)
    packed = (
        bytes_i64[:, 0] | (bytes_i64[:, 1] << 8) | (bytes_i64[:, 2] << 16) | (bytes_i64[:, 3] << 24)
    )
    largest = (packed >> 30) & 3
    work = packed.clone()
    xyzw = torch.zeros((rot.shape[0], 4), device=rot.device, dtype=torch.float32)
    ss = torch.zeros(rot.shape[0], device=rot.device, dtype=torch.float32)

    for axis in range(3, -1, -1):
        keep = largest != axis
        mag = (work & int(_C_MASK)).to(torch.float32)
        neg = ((work >> 9) & 1).to(torch.bool)
        val = float(_INV_SQRT2) * (mag / float(_C_MASK))
        val = torch.where(neg, -val, val)
        xyzw[:, axis] = torch.where(keep, val, xyzw[:, axis])
        ss = ss + torch.where(keep, val * val, torch.zeros_like(val))
        work = torch.where(keep, work >> 10, work)

    large = torch.sqrt(torch.clamp(1.0 - ss, min=0.0))
    xyzw.scatter_(1, largest.view(-1, 1), large.view(-1, 1))
    return xyzw[:, [3, 0, 1, 2]]


def _decode_first_three_quats(rot: torch.Tensor) -> torch.Tensor:
    xyz = rot.to(torch.float32) / 127.5 - 1.0
    w = torch.sqrt(torch.clamp(1.0 - torch.sum(xyz * xyz, dim=1), min=0.0))
    return torch.cat([w[:, None], xyz], dim=1)


def _decode_spz_tensor(
    payload: torch.Tensor,
    *,
    n: int,
    sh_degree: int,
    sh_dim: int,
    fractional_bits: int,
    uses_smallest_three: bool,
    rot_stride: int,
) -> GSTensor:
    alpha_ofs = 9 * n
    color_ofs = 10 * n
    scale_ofs = 13 * n
    rot_ofs = 16 * n
    sh_ofs = 16 * n + rot_stride * n

    means = _decode_positions(payload, n, fractional_bits)
    alphas = payload[alpha_ofs : alpha_ofs + n].to(torch.float32) / 255.0
    alphas = torch.clamp(alphas, float(_EPS), 1.0 - float(_EPS))
    opacities = torch.log(alphas / (1.0 - alphas))
    sh0 = (
        payload[color_ofs : color_ofs + 3 * n].view(n, 3).to(torch.float32) / 255.0 - 0.5
    ) / float(COLOR_SCALE)
    scales = payload[scale_ofs : scale_ofs + 3 * n].view(n, 3).to(torch.float32) / 16.0 - 10.0

    rot = payload[rot_ofs : rot_ofs + rot_stride * n].view(n, rot_stride)
    quats = (
        _decode_smallest_three_quats(rot) if uses_smallest_three else _decode_first_three_quats(rot)
    )

    if sh_dim > 0:
        sh_n = payload[sh_ofs : sh_ofs + sh_dim * 3 * n].view(n, sh_dim, 3).to(torch.float32)
        sh_n = (sh_n - 128.0) / 128.0
    else:
        sh_n = torch.zeros((n, 0, 3), device=payload.device, dtype=torch.float32)

    return GSTensor(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=sh_n,
        masks=None,
        mask_names=None,
        _base=None,
        _format=create_ply_format(sh_degree),
    )


def read_spz_gpu(file_path: str | Path, device: str | torch.device = "cuda") -> GSTensor:
    """Read a Niantic SPZ file into a :class:`GSTensor` on ``device``.

    Container decompression and validation happen on CPU. The packed SPZ sections
    are then transferred once to the target device and unpacked with PyTorch
    tensor operations, avoiding full CPU float materialization before VRAM upload.

    Args:
        file_path: Path to a legacy gzip SPZ v1/v2/v3 or NGSP/zstd v4 file.
        device: Target device, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        GSTensor in PLY format on the requested device.
    """
    parsed = _read_spz_payload(file_path)
    device_obj = torch.device(device)
    payload = _payload_to_tensor(parsed.payload, device_obj)
    return _decode_spz_tensor(
        payload,
        n=parsed.n,
        sh_degree=parsed.sh_degree,
        sh_dim=parsed.sh_dim,
        fractional_bits=parsed.fractional_bits,
        uses_smallest_three=parsed.uses_smallest_three,
        rot_stride=parsed.rot_stride,
    )


__all__ = ["read_spz_gpu"]
