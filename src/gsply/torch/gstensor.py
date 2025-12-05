"""PyTorch GPU-accelerated Gaussian Splatting data container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gsply.formats import SH_BANDS_TO_DEGREE, SH_C0

# Import DataFormat, FormatDict, GSData, and helpers from gsdata
# No circular dependency - gsdata.py doesn't import GSTensor
from gsply.gsdata import (
    DataFormat,
    FormatDict,
    GSData,
    _create_format_dict,
    _detect_format_from_values,
    _get_sh_order_format,
    create_ply_format,
    create_rasterizer_format,
)

# Import compression functions (no circular dependency - compression.py only imports GSTensor for type hints)
from gsply.torch.compression import read_compressed_gpu, write_compressed_gpu

# Lazy imports to avoid circular dependencies with writer.py and reader.py
# (writer.py and reader.py import GSData at module level, creating circular imports)


@dataclass
class GSTensor:
    """GPU-accelerated Gaussian Splatting data container using PyTorch tensors.

    This container holds Gaussian parameters as PyTorch tensors, supporting both
    CPU and GPU devices. Designed for efficient GPU operations and training workflows.

    Attributes:
        means: (N, 3) - xyz positions [torch.Tensor]
        scales: (N, 3) - scale parameters [torch.Tensor]
            - PLY format: log-scales (log(scale))
            - LINEAR format: linear scales (scale)
        quats: (N, 4) - rotation quaternions [torch.Tensor]
        opacities: (N,) - opacity values [torch.Tensor]
            - PLY format: logit-opacities (logit(opacity))
            - LINEAR format: linear opacities (opacity in [0, 1])
        sh0: (N, 3) - DC spherical harmonics [torch.Tensor] (always SH format)
        shN: (N, K, 3) - Higher-order SH coefficients (K bands) [torch.Tensor or None] (always SH format)
        masks: (N,) or (N, L) - Boolean mask layers [torch.Tensor or None]
        mask_names: List of mask layer names [list[str] or None]
        _base: (N, P) - Private base tensor (keeps memory alive for views) [torch.Tensor or None]
        _format: DataFormat enum - Format tracking for scales and opacities
            - DataFormat.PLY: scales are log-scales, opacities are logit-opacities
            - DataFormat.LINEAR: scales are linear scales, opacities are linear opacities
            - DataFormat.UNKNOWN: format unknown (manually created or mixed formats)

    Performance:
        - Zero-copy GPU transfers when using _base (11x faster)
        - GPU slicing is free (views have zero memory cost)
        - Single tensor transfer vs multiple separate transfers

    Example:
        >>> import gsply
        >>> data = gsply.plyread("scene.ply")  # GSData on CPU
        >>> gstensor = data.to_tensor(device='cuda')  # GSTensor on GPU
        >>> positions_gpu = gstensor.means  # (N, 3) tensor on GPU
        >>> subset = gstensor[100:200]  # Slice (returns view)
        >>> data_cpu = gstensor.to_gsdata()  # Convert back to GSData
    """

    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    sh0: torch.Tensor
    shN: torch.Tensor | None = None
    _format: FormatDict = field(
        default_factory=lambda: {}
    )  # Format tracking - auto-detected in __post_init__ if empty
    masks: torch.Tensor | None = None
    mask_names: list[str] | None = None
    _base: torch.Tensor | None = None

    def __post_init__(self):
        """Auto-detect format if not provided."""
        # Copy format dict to avoid sharing mutable state between instances
        self._format = dict(self._format)

        # If _format is empty dict, auto-detect from values
        if not self._format:
            # Convert tensors to numpy for format detection
            scales_np = self.scales.detach().cpu().numpy()
            opacities_np = self.opacities.detach().cpu().numpy()
            scales_format, opacities_format = _detect_format_from_values(scales_np, opacities_np)
            self._format = _create_format_dict(
                scales=scales_format,
                opacities=opacities_format,
                sh0=DataFormat.SH0_SH,
                sh_order=_get_sh_order_format(self.get_sh_degree()),
                means=DataFormat.MEANS_RAW,
                quats=DataFormat.QUATS_RAW,
            )

    def __len__(self) -> int:
        """Return the number of Gaussians."""
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        """Return device of the tensors."""
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype of the tensors."""
        return self.means.dtype

    def get_sh_degree(self) -> int:
        """Get SH degree from shN shape.

        :returns: SH degree (0-3)
        """
        if self.shN is None or self.shN.shape[1] == 0:
            return 0
        # shN.shape[1] is number of bands (K)
        sh_bands = self.shN.shape[1]
        return SH_BANDS_TO_DEGREE.get(int(sh_bands), 0)

    def has_high_order_sh(self) -> bool:
        """Check if data has higher-order SH coefficients.

        :returns: True if SH degree > 0
        """
        return self.shN is not None and self.shN.shape[1] > 0

    # ==========================================================================
    # Format Query Properties
    # ==========================================================================

    @property
    def is_scales_ply(self) -> bool:
        """Check if scales are in PLY format (log-scales).

        :returns: True if scales are log-scales
        """
        return self._format.get("scales") == DataFormat.SCALES_PLY

    @property
    def is_scales_linear(self) -> bool:
        """Check if scales are in linear format.

        :returns: True if scales are linear
        """
        return self._format.get("scales") == DataFormat.SCALES_LINEAR

    @property
    def is_opacities_ply(self) -> bool:
        """Check if opacities are in PLY format (logit-opacities).

        :returns: True if opacities are logit-opacities
        """
        return self._format.get("opacities") == DataFormat.OPACITIES_PLY

    @property
    def is_opacities_linear(self) -> bool:
        """Check if opacities are in linear format [0, 1].

        :returns: True if opacities are linear
        """
        return self._format.get("opacities") == DataFormat.OPACITIES_LINEAR

    @property
    def is_sh0_sh(self) -> bool:
        """Check if sh0 is in spherical harmonics format.

        :returns: True if sh0 is in SH format
        """
        return self._format.get("sh0") == DataFormat.SH0_SH

    @property
    def is_sh0_rgb(self) -> bool:
        """Check if sh0 is in RGB color format.

        :returns: True if sh0 is in RGB format
        """
        return self._format.get("sh0") == DataFormat.SH0_RGB

    @property
    def is_sh_order_0(self) -> bool:
        """Check if SH degree is 0 (only sh0, no shN).

        :returns: True if SH degree is 0
        """
        return self._format.get("sh_order") == DataFormat.SH_ORDER_0

    @property
    def is_sh_order_1(self) -> bool:
        """Check if SH degree is 1 (3 bands).

        :returns: True if SH degree is 1
        """
        return self._format.get("sh_order") == DataFormat.SH_ORDER_1

    @property
    def is_sh_order_2(self) -> bool:
        """Check if SH degree is 2 (8 bands).

        :returns: True if SH degree is 2
        """
        return self._format.get("sh_order") == DataFormat.SH_ORDER_2

    @property
    def is_sh_order_3(self) -> bool:
        """Check if SH degree is 3 (15 bands).

        :returns: True if SH degree is 3
        """
        return self._format.get("sh_order") == DataFormat.SH_ORDER_3

    # ==========================================================================
    # Format Management API (Public)
    # ==========================================================================

    @property
    def format_state(self) -> FormatDict:
        """Get a read-only copy of the format state.

        Returns a copy of the internal format dict for inspection.
        Use copy_format_from() to copy format between objects.

        :returns: Copy of the format dict (modifications won't affect original)

        Example:
            >>> gstensor = GSTensor.from_gsdata(data, device='cuda')
            >>> fmt = gstensor.format_state
            >>> print(fmt)  # {'scales': DataFormat.SCALES_PLY, ...}
        """
        return dict(self._format)

    def copy_format_from(self, other: GSTensor | GSData) -> None:
        """Copy format tracking from another GSTensor or GSData object.

        This is the public API for copying format state between objects.
        Use this instead of directly accessing _format dict.

        :param other: Source GSTensor or GSData to copy format from

        Example:
            >>> # After processing that might lose format
            >>> processed.copy_format_from(original)
        """
        self._format = dict(other._format)

    def with_format(self, **updates) -> GSTensor:
        """Create a copy with updated format settings.

        Returns a new GSTensor with the same data but updated format dict.
        This is useful for explicitly setting format after operations.

        :param updates: Format updates (keys: scales, opacities, sh0, sh_order)
        :returns: New GSTensor with updated format

        Example:
            >>> # Mark data as having linear opacities after conversion
            >>> linear_tensor = tensor.with_format(opacities=DataFormat.OPACITIES_LINEAR)
        """
        new_format = dict(self._format)
        for key, value in updates.items():
            if key in ("scales", "opacities", "sh0", "sh_order", "means", "quats"):
                new_format[key] = value
            else:
                raise ValueError(f"Unknown format key: {key}")

        return GSTensor(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh0=self.sh0,
            shN=self.shN,
            masks=self.masks,
            mask_names=self.mask_names,
            _base=self._base,
            _format=new_format,
        )

    # ==========================================================================
    # Conversion Methods (GSData <-> GSTensor)
    # ==========================================================================

    @classmethod
    def from_compressed(
        cls, file_path: str | Path, device: str | torch.device = "cuda"
    ) -> GSTensor:
        """Read compressed PLY file directly to GPU.

        Uses GPU-accelerated decompression for faster loading compared to
        CPU decompression + upload. Ideal for large compressed datasets.

        :param file_path: Path to compressed PLY file
        :param device: Target device (default "cuda")
        :returns: GSTensor with decompressed data on GPU

        Performance:
            - 10-50x faster decompression than CPU Numba
            - Zero intermediate CPU transfers
            - Direct GPU memory allocation

        Example:
            >>> gstensor = GSTensor.from_compressed("scene.ply_compressed", device="cuda")
            >>> print(f"Loaded {len(gstensor):,} Gaussians on GPU")
        """
        return read_compressed_gpu(Path(file_path), str(device))

    @classmethod
    def from_gsdata(
        cls,
        data: GSData,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
        mask: np.ndarray | None = None,
    ) -> GSTensor:
        """Convert GSData to GSTensor efficiently.

        Uses _base optimization when available for 11x faster transfer:
        - With _base: Single tensor transfer (zero CPU copy overhead)
        - Without _base: Stack arrays then transfer (one CPU copy + transfer)

        When mask is provided, only the masked subset is transferred to GPU,
        avoiding intermediate CPU copies and unnecessary GPU memory usage.

        :param data: GSData object to convert
        :param device: Target device ('cuda', 'cpu', or torch.device)
        :param dtype: Target dtype (default: float32)
        :param requires_grad: Enable gradient tracking (default: False)
        :param mask: Optional boolean mask to filter data before transfer (default: None)
        :returns: GSTensor on specified device

        Example:
            >>> data = gsply.plyread("scene.ply")
            >>> # Fast path (data has _base from plyread)
            >>> gstensor = GSTensor.from_gsdata(data, device='cuda')
            >>> # Or with gradients for training
            >>> gstensor = GSTensor.from_gsdata(data, device='cuda', requires_grad=True)
            >>> # Direct masked transfer (no intermediate CPU copy)
            >>> mask = data.opacities > 0.5
            >>> gstensor = GSTensor.from_gsdata(data, device='cuda', mask=mask)
        """
        if dtype is None:
            dtype = torch.float32

        device_obj = torch.device(device)

        # Apply mask if provided
        if mask is not None:
            if len(mask) != len(data):
                raise ValueError(f"Mask length {len(mask)} doesn't match data length {len(data)}")
            # Slice data with mask (creates views where possible, avoiding copies)
            data = data[mask]

        # Fast path: Use _base if available (11x faster)
        if data._base is not None:
            # Ensure array is contiguous (handles sliced data edge case)
            base_array = np.ascontiguousarray(data._base)

            # Single tensor transfer (zero CPU copy if already contiguous)
            base_tensor = torch.from_numpy(base_array).to(device=device_obj, dtype=dtype)
            base_tensor.requires_grad_(requires_grad)

            # Transfer masks separately if present
            masks_tensor = None
            if data.masks is not None:
                masks_array = np.ascontiguousarray(data.masks)
                masks_tensor = torch.from_numpy(masks_array).to(device=device_obj)

            # Recreate GSTensor from base tensor (preserve format from GSData)
            return cls._recreate_from_base(
                base_tensor,
                format_flag=data._format,
                masks_tensor=masks_tensor,
                mask_names=data.mask_names,
            )

        # Fallback: Stack arrays on CPU then transfer (2x faster than separate transfers)
        # Convert to Python int to avoid numpy _NoValueType issues
        n = int(len(data))

        # Determine property count based on SH degree
        # Layout: means(3) + sh0(3) + shN(K*3) + opacity(1) + scales(3) + quats(4)
        # Total: 14 + K*3 where K=0/9/24/45
        if data.shN is not None and data.shN.shape[1] > 0:
            # Convert to Python int to avoid numpy _NoValueType issues
            sh_coeffs = int(data.shN.shape[1])  # K = 9, 24, 45
            n_props = 14 + sh_coeffs * 3  # Total properties
        else:
            sh_coeffs = 0
            n_props = 14  # SH0

        # Stack all arrays into single base array on CPU
        base_cpu = np.empty((n, n_props), dtype=np.float32)
        base_cpu[:, 0:3] = data.means
        base_cpu[:, 3:6] = data.sh0

        if sh_coeffs > 0:
            # Store shN in channel-grouped order [R0..Rk, G0..Gk, B0..Bk] to match PLY convention
            # data.shN is [N, K, 3], transpose to [N, 3, K] then flatten to [N, 3*K]
            shN_flat = data.shN.transpose(0, 2, 1).reshape(n, sh_coeffs * 3)
            base_cpu[:, 6 : 6 + sh_coeffs * 3] = shN_flat
            opacity_idx = 6 + sh_coeffs * 3
        else:
            opacity_idx = 6

        base_cpu[:, opacity_idx] = data.opacities
        base_cpu[:, opacity_idx + 1 : opacity_idx + 4] = data.scales
        base_cpu[:, opacity_idx + 4 : opacity_idx + 8] = data.quats

        # Single GPU transfer (2x faster than 5 separate transfers)
        base_tensor = torch.from_numpy(base_cpu).to(device=device_obj, dtype=dtype)
        base_tensor.requires_grad_(requires_grad)

        # Transfer masks separately
        masks_tensor = None
        if data.masks is not None:
            masks_array = np.ascontiguousarray(data.masks)
            masks_tensor = torch.from_numpy(masks_array).to(device=device_obj)

        # Recreate GSTensor from base tensor (preserve format from GSData)
        return cls._recreate_from_base(
            base_tensor,
            format_flag=data._format,
            masks_tensor=masks_tensor,
            mask_names=data.mask_names,
        )

    def to_gsdata(self) -> GSData:
        """Convert GSTensor back to GSData (CPU NumPy arrays).

        Transfers all tensors to CPU and converts to NumPy arrays.

        :returns: GSData object with NumPy arrays on CPU

        Example:
            >>> gstensor = data.to_tensor(device='cuda')
            >>> # ... GPU operations ...
            >>> data_cpu = gstensor.to_gsdata()  # Back to NumPy on CPU
        """
        # Transfer to CPU first
        cpu_tensor = self.to("cpu")

        # Fast path: Use _base if available
        if cpu_tensor._base is not None:
            base_numpy = cpu_tensor._base.detach().numpy()
            masks_numpy = (
                cpu_tensor.masks.detach().numpy() if cpu_tensor.masks is not None else None
            )

            return GSData._recreate_from_base(
                base_numpy,
                format_flag=cpu_tensor._format,
                masks_array=masks_numpy,
                mask_names=cpu_tensor.mask_names,
            )

        # Fallback: Convert each tensor
        shN_numpy = None
        if cpu_tensor.shN is not None:
            shN_numpy = cpu_tensor.shN.detach().numpy()

        masks_numpy = None
        if cpu_tensor.masks is not None:
            masks_numpy = cpu_tensor.masks.detach().numpy()

        return GSData(
            means=cpu_tensor.means.detach().numpy(),
            scales=cpu_tensor.scales.detach().numpy(),
            quats=cpu_tensor.quats.detach().numpy(),
            opacities=cpu_tensor.opacities.detach().numpy(),
            sh0=cpu_tensor.sh0.detach().numpy(),
            shN=shN_numpy,
            masks=masks_numpy,
            mask_names=cpu_tensor.mask_names,
            _base=None,
            _format=cpu_tensor._format,  # Preserve format flag
        )

    def to_ply_tensor(self) -> GSTensor:
        """Convert to GSTensor with PLY-compatible log/logit scaling.

        Returns a new GSTensor with log-scales and logit-opacities.
        Useful for GPU-accelerated writing.
        """
        # Constants for numerical stability
        _MIN_SCALE = 1e-9
        _MIN_OPACITY = 1e-4
        _MAX_OPACITY = 1.0 - 1e-4

        # Clone to avoid modifying original
        scales = torch.log(torch.clamp(self.scales, min=_MIN_SCALE))

        # Logit opacities: log(p / (1 - p))
        opacities = torch.logit(torch.clamp(self.opacities, min=_MIN_OPACITY, max=_MAX_OPACITY))

        # Preserve format and update to PLY format
        format_flag = self._format.copy()
        format_flag["scales"] = DataFormat.SCALES_PLY
        format_flag["opacities"] = DataFormat.OPACITIES_PLY
        if "sh_order" not in format_flag:
            format_flag["sh_order"] = _get_sh_order_format(self.get_sh_degree())
        if "sh0" not in format_flag:
            format_flag["sh0"] = DataFormat.SH0_SH
        if "means" not in format_flag:
            format_flag["means"] = DataFormat.MEANS_RAW
        if "quats" not in format_flag:
            format_flag["quats"] = DataFormat.QUATS_RAW

        return GSTensor(
            means=self.means,
            scales=scales,
            quats=self.quats,
            opacities=opacities,
            sh0=self.sh0,
            shN=self.shN,
            masks=self.masks,
            mask_names=self.mask_names,
            _base=None,
            _format=format_flag,  # Set to PLY format
        )

    def to_ply_data(self) -> GSData:
        """Convert to GSData with PLY-compatible log/logit scaling.

        Converts linear scales to log-scales and linear opacities to logit-opacities,
        which is the standard format for Gaussian Splatting PLY files.

        :returns: GSData object ready for plywrite()
        """
        # Perform conversion using PyTorch (handles logit correctly)
        return self.to_ply_tensor().to_gsdata()

    def normalize(self, inplace: bool = True) -> GSTensor:
        """Convert linear scales/opacities to PLY format (log-scales, logit-opacities).

        Converts:
        - Linear scales → log-scales: log(scale)
        - Linear opacities → logit-opacities: logit(opacity)

        This is the standard format used in Gaussian Splatting PLY files.
        Use this when you have linear data and need to save to PLY format.

        :param inplace: If True, modify this object in-place (default). If False, return new object.
        :returns: GSTensor object (self if inplace=True, new object otherwise)

        Example:
            >>> # Data with linear scales and opacities
            >>> gstensor = GSTensor(scales=torch.tensor([[0.1, 0.2, 0.3]]), opacities=torch.tensor([0.5]), ...)
            >>> # Convert to PLY format in-place (modifies gstensor)
            >>> gstensor.normalize()  # or: gstensor.normalize(inplace=True)
            >>> # Now ready to save with plywrite_gpu()
            >>> plywrite_gpu("output.ply", gstensor)
            >>>
            >>> # Or create a copy if you need to keep original
            >>> ply_tensor = gstensor.normalize(inplace=False)
        """
        # Constants for numerical stability
        min_scale = 1e-9
        min_opacity = 1e-4
        max_opacity = 1.0 - 1e-4

        # Convert linear scales to log-scales: log(scale)
        scales = torch.log(torch.clamp(self.scales, min=min_scale))

        # Convert linear opacities to logit-opacities: logit(opacity)
        # Use eps=min_opacity to match GSData behavior (1e-4)
        opacities = torch.logit(
            torch.clamp(self.opacities, min=min_opacity, max=max_opacity), eps=min_opacity
        )

        if inplace:
            self.scales = scales
            self.opacities = opacities
            self._base = None  # Invalidate _base since we modified tensors
            # Update format dict: scales and opacities are now in PLY format
            self._format["scales"] = DataFormat.SCALES_PLY
            self._format["opacities"] = DataFormat.OPACITIES_PLY
            return self

        return GSTensor(
            means=self.means,
            scales=scales,
            quats=self.quats,
            opacities=opacities,
            sh0=self.sh0,
            shN=self.shN,
            masks=self.masks,
            mask_names=self.mask_names,
            _base=None,
            _format={
                **self._format,
                "scales": DataFormat.SCALES_PLY,
                "opacities": DataFormat.OPACITIES_PLY,
                "sh_order": _get_sh_order_format(self.get_sh_degree()),
            },
        )

    def denormalize(self, inplace: bool = True) -> GSTensor:
        """Convert PLY format (log-scales, logit-opacities) to linear format.

        Converts:
        - Log-scales → linear scales: exp(log_scale)
        - Logit-opacities → linear opacities: sigmoid(logit)

        Use this when you load PLY files (which use log/logit format) and need
        linear values for computations or visualization.

        :param inplace: If True, modify this object in-place (default). If False, return new object.
        :returns: GSTensor object (self if inplace=True, new object otherwise)

        Example:
            >>> # Load PLY file (contains log-scales and logit-opacities)
            >>> gstensor = plyread_gpu("scene.ply")
            >>> # Convert to linear format in-place (modifies gstensor)
            >>> gstensor.denormalize()  # or: gstensor.denormalize(inplace=True)
            >>> # Now scales and opacities are in linear space [0, 1] for opacities
            >>> print(f"Linear opacity range: [{gstensor.opacities.min():.3f}, {gstensor.opacities.max():.3f}]")
            >>>
            >>> # Or create a copy if you need to keep PLY format
            >>> linear_tensor = gstensor.denormalize(inplace=False)
        """
        # Convert log-scales back to linear: exp(log_scale)
        scales = torch.exp(self.scales)

        # Convert logit-opacities back to linear: sigmoid(logit)
        opacities = torch.sigmoid(self.opacities)

        if inplace:
            self.scales = scales
            self.opacities = opacities
            self._base = None  # Invalidate _base since we modified tensors
            # Update format dict: scales and opacities are now in linear format
            self._format["scales"] = DataFormat.SCALES_LINEAR
            self._format["opacities"] = DataFormat.OPACITIES_LINEAR
            return self

        return GSTensor(
            means=self.means,
            scales=scales,
            quats=self.quats,
            opacities=opacities,
            sh0=self.sh0,
            shN=self.shN,
            masks=self.masks,
            mask_names=self.mask_names,
            _base=None,
            _format={
                **self._format,
                "scales": DataFormat.SCALES_LINEAR,
                "opacities": DataFormat.OPACITIES_LINEAR,
                "sh_order": _get_sh_order_format(self.get_sh_degree()),
            },
        )

    def to_rgb(self, inplace: bool = True) -> GSTensor:
        """Convert sh0 from spherical harmonics (SH) format to RGB color format.

        Converts SH DC coefficients to RGB colors in [0, 1] range.
        Formula: rgb = sh0 * SH_C0 + 0.5

        :param inplace: If True, modify this object in-place (default). If False, return new object.
        :returns: GSTensor object (self if inplace=True, new object otherwise)

        Example:
            >>> # Load PLY file to GPU (sh0 is in SH format)
            >>> gstensor = gsply.plyread_gpu("scene.ply")
            >>> # Convert to RGB format in-place
            >>> gstensor.to_rgb()  # or: gstensor.to_rgb(inplace=True)
            >>> # Now sh0 contains RGB colors [0, 1]
            >>> print(f"RGB color range: [{gstensor.sh0.min():.3f}, {gstensor.sh0.max():.3f}]")
            >>>
            >>> # Or create a copy if you need to keep SH format
            >>> rgb_tensor = gstensor.to_rgb(inplace=False)
        """
        if inplace:
            # True in-place: modify self.sh0 directly using PyTorch in-place operations
            self.sh0.mul_(SH_C0).add_(0.5)
            self._base = None  # Invalidate _base since we modified tensors
            # Update format dict: sh0 is now in RGB format
            self._format["sh0"] = DataFormat.SH0_RGB
            return self

        # Create copy for non-inplace operation
        rgb = self.sh0 * SH_C0 + 0.5
        return GSTensor(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh0=rgb,
            shN=self.shN,
            masks=self.masks,
            mask_names=self.mask_names,
            _base=None,
            _format={**self._format, "sh0": DataFormat.SH0_RGB},
        )

    def to_sh(self, inplace: bool = True) -> GSTensor:
        """Convert sh0 from RGB color format to spherical harmonics (SH) format.

        Converts RGB colors in [0, 1] range to SH DC coefficients.
        Formula: sh0 = (rgb - 0.5) / SH_C0

        :param inplace: If True, modify this object in-place (default). If False, return new object.
        :returns: GSTensor object (self if inplace=True, new object otherwise)

        Example:
            >>> # Create GSTensor with RGB colors
            >>> rgb_colors = torch.rand(1000, 3, device="cuda")
            >>> gstensor = GSTensor(means=..., scales=..., sh0=rgb_colors, ...)
            >>> # Convert to SH format in-place
            >>> gstensor.to_sh()  # or: gstensor.to_sh(inplace=True)
            >>> # Now sh0 contains SH DC coefficients
            >>>
            >>> # Or create a copy if you need to keep RGB format
            >>> sh_tensor = gstensor.to_sh(inplace=False)
        """
        if inplace:
            # True in-place: modify self.sh0 directly using PyTorch in-place operations
            self.sh0.sub_(0.5).div_(SH_C0)
            self._base = None  # Invalidate _base since we modified tensors
            # Update format dict: sh0 is now in SH format
            self._format["sh0"] = DataFormat.SH0_SH
            return self

        # Create copy for non-inplace operation
        sh = (self.sh0 - 0.5) / SH_C0
        return GSTensor(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh0=sh,
            shN=self.shN,
            masks=self.masks,
            mask_names=self.mask_names,
            _base=None,
            _format={**self._format, "sh0": DataFormat.SH0_SH},
        )

    # ==========================================================================
    # Device Management
    # ==========================================================================

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> GSTensor:
        """Move tensors to specified device and/or dtype.

        :param device: Target device ('cuda', 'cpu', or torch.device)
        :param dtype: Target dtype
        :param non_blocking: If True, asynchronous transfer (default: False)
        :returns: New GSTensor on target device/dtype

        Example:
            >>> gstensor_gpu = gstensor.to('cuda')
            >>> gstensor_half = gstensor.to(dtype=torch.float16)
            >>> gstensor_gpu_half = gstensor.to('cuda', dtype=torch.float16)
        """
        # If no changes requested, return self
        if device is None and dtype is None:
            return self

        # Determine target device and dtype
        target_device = torch.device(device) if device is not None else self.device
        target_dtype = dtype if dtype is not None else self.dtype

        # If already on target device and dtype, return self
        if target_device == self.device and target_dtype == self.dtype:
            return self

        # Fast path: Use _base if available
        if self._base is not None:
            new_base = self._base.to(
                device=target_device, dtype=target_dtype, non_blocking=non_blocking
            )
            new_masks = None
            if self.masks is not None:
                new_masks = self.masks.to(device=target_device, non_blocking=non_blocking)

            return self._recreate_from_base(
                new_base,
                format_flag=self._format,
                masks_tensor=new_masks,
                mask_names=self.mask_names,
            )

        # Fallback: Move each tensor
        new_shN = None
        if self.shN is not None:
            new_shN = self.shN.to(
                device=target_device, dtype=target_dtype, non_blocking=non_blocking
            )

        new_masks = None
        if self.masks is not None:
            new_masks = self.masks.to(device=target_device, non_blocking=non_blocking)

        return GSTensor(
            means=self.means.to(
                device=target_device, dtype=target_dtype, non_blocking=non_blocking
            ),
            scales=self.scales.to(
                device=target_device, dtype=target_dtype, non_blocking=non_blocking
            ),
            quats=self.quats.to(
                device=target_device, dtype=target_dtype, non_blocking=non_blocking
            ),
            opacities=self.opacities.to(
                device=target_device, dtype=target_dtype, non_blocking=non_blocking
            ),
            sh0=self.sh0.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking),
            shN=new_shN,
            masks=new_masks,
            mask_names=self.mask_names,
            _base=None,
            _format=self._format,  # Preserve format flag
        )

    def cpu(self) -> GSTensor:
        """Move tensors to CPU.

        :returns: New GSTensor on CPU
        """
        return self.to("cpu")

    def cuda(self, device: int | None = None) -> GSTensor:
        """Move tensors to CUDA device.

        :param device: CUDA device index (default: current device)
        :returns: New GSTensor on CUDA
        """
        if device is None:
            return self.to("cuda")
        return self.to(f"cuda:{device}")

    # ==========================================================================
    # _base Optimization (25x Faster Slicing)
    # ==========================================================================

    def consolidate(self) -> GSTensor:
        """Consolidate separate tensors into a single base tensor.

        Creates a _base tensor from separate tensors, improving performance for
        slicing operations (25x faster boolean masking on GPU).

        :returns: New GSTensor with _base tensor, or self if already consolidated

        Example:
            >>> gstensor = gstensor.consolidate()  # Create _base
            >>> subset = gstensor[mask]  # 25x faster with _base
        """
        if self._base is not None:
            return self  # Already consolidated

        # Determine property count based on SH degree
        n_gaussians = len(self)

        # Layout: means(3) + sh0(3) + shN(K*3) + opacity(1) + scales(3) + quats(4)
        # Total: 14 + K*3 where K=0/9/24/45
        if self.shN is not None and self.shN.shape[1] > 0:
            sh_coeffs = self.shN.shape[1]
            n_props = 14 + sh_coeffs * 3  # SH1: 41, SH2: 86, SH3: 149
        else:
            n_props = 14  # SH0

        # Create base tensor
        new_base = torch.empty((n_gaussians, n_props), dtype=self.dtype, device=self.device)
        new_base[:, 0:3] = self.means
        new_base[:, 3:6] = self.sh0

        # Handle shN if present
        if self.shN is not None and self.shN.shape[1] > 0:
            sh_coeffs = self.shN.shape[1]
            shN_flat = self.shN.reshape(n_gaussians, sh_coeffs * 3)
            new_base[:, 6 : 6 + sh_coeffs * 3] = shN_flat
            opacity_idx = 6 + sh_coeffs * 3
        else:
            opacity_idx = 6

        new_base[:, opacity_idx] = self.opacities
        new_base[:, opacity_idx + 1 : opacity_idx + 4] = self.scales
        new_base[:, opacity_idx + 4 : opacity_idx + 8] = self.quats

        # Copy masks if present
        new_masks = self.masks.clone() if self.masks is not None else None

        # Recreate GSTensor with new base (preserve format)
        return self._recreate_from_base(
            new_base, format_flag=self._format, masks_tensor=new_masks, mask_names=self.mask_names
        )

    @classmethod
    def _recreate_from_base(
        cls,
        base_tensor: torch.Tensor,
        format_flag: FormatDict,
        masks_tensor: torch.Tensor | None = None,
        mask_names: list[str] | None = None,
    ) -> GSTensor | None:
        """Helper to recreate GSTensor from a base tensor.

        :param base_tensor: Base tensor (N, P) where P is property count
        :param format_flag: Format dict (required)
        :param masks_tensor: Optional masks tensor (N,) or (N, L)
        :param mask_names: Optional mask layer names
        :returns: New GSTensor with views into base_tensor, or None if unknown format
        """
        # Convert to Python int to avoid numpy _NoValueType issues
        n_gaussians = int(base_tensor.shape[0])
        n_props = int(base_tensor.shape[1])

        # Map property count to SH degree
        # Layout: means(3) + sh0(3) + shN(K*3) + opacity(1) + scales(3) + quats(4)
        # Total: 14 + K*3 where K is number of bands
        # Note: shN.shape = (N, K, 3) where K is the number of bands
        if n_props == 14:  # SH0: no shN
            sh_coeffs = 0
        elif n_props == 23:  # SH1: 14 + 3*3, K=3 bands
            sh_coeffs = 3
        elif n_props == 38:  # SH2: 14 + 8*3, K=8 bands
            sh_coeffs = 8
        elif n_props == 59:  # SH3: 14 + 15*3, K=15 bands
            sh_coeffs = 15
        else:
            return None  # Unknown format

        # Create views into the base tensor
        means = base_tensor[:, 0:3]
        sh0 = base_tensor[:, 3:6]

        if sh_coeffs > 0:
            shN_flat = base_tensor[:, 6 : 6 + sh_coeffs * 3]
            # PLY stores SH coefficients channel-grouped: [R0..Rk, G0..Gk, B0..Bk]
            # Reshape to [N, 3, K] then transpose to [N, K, 3] for gsplat convention
            # contiguous() is required for correct memory layout after transpose
            shN = shN_flat.reshape(n_gaussians, 3, sh_coeffs).transpose(1, 2).contiguous()
            opacity_idx = 6 + sh_coeffs * 3
        else:
            shN = None
            opacity_idx = 6

        opacities = base_tensor[:, opacity_idx]
        scales = base_tensor[:, opacity_idx + 1 : opacity_idx + 4]
        quats = base_tensor[:, opacity_idx + 4 : opacity_idx + 8]

        return cls(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            masks=masks_tensor,
            mask_names=mask_names,
            _base=base_tensor,
            _format=format_flag,  # Format dict (always provided)
        )

    def _slice_from_base(self, indices_or_mask):
        """Efficiently slice data when _base tensor exists.

        :param indices_or_mask: Slice, boolean mask, or integer indices
        :returns: New GSTensor with sliced data, or None if no _base
        """
        if self._base is None:
            return None

        # Slice the base tensor
        base_subset = self._base[indices_or_mask]

        # Handle masks if present
        if self.masks is not None:
            masks_subset = self.masks[indices_or_mask]
        else:
            masks_subset = None

        # Recreate from sliced base (preserve mask_names and format)
        return self._recreate_from_base(
            base_subset,
            format_flag=self._format,
            masks_tensor=masks_subset,
            mask_names=self.mask_names,
        )

    # ==========================================================================
    # Slicing and Indexing
    # ==========================================================================

    def __getitem__(self, key):
        """Support efficient slicing and indexing.

        Following PyTorch conventions:
        - Continuous slice: Returns GSTensor view (shares memory)
        - Boolean mask: Returns GSTensor copy (independent data)
        - Fancy indexing: Returns GSTensor copy
        - Single index: Returns tuple of values

        When _base exists, slicing is up to 25x faster for boolean masks.

        Examples:
            >>> gstensor[0]         # Single Gaussian (tuple)
            >>> gstensor[10:20]     # Slice (VIEW)
            >>> gstensor[::10]      # Step slice (VIEW)
            >>> gstensor[mask]      # Boolean mask (COPY)
            >>> gstensor[[0,1,2]]   # Fancy indexing (COPY)

        :param key: Slice, index, boolean mask, or index array
        :returns: Single Gaussian (tuple) or new GSTensor
        """
        # Handle single index - return tuple
        if isinstance(key, int):
            # Convert negative index
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} out of range for {len(self)} Gaussians")

            # Return tuple of values
            return (
                self.means[key],
                self.scales[key],
                self.quats[key],
                self.opacities[key],
                self.sh0[key],
                self.shN[key] if self.shN is not None else None,
                self.masks[key] if self.masks is not None else None,
            )

        # Handle slice
        if isinstance(key, slice):
            # Try fast path with _base
            if self._base is not None:
                result = self._slice_from_base(key)
                if result is not None:
                    return result

            # Fallback: Slice individual tensors
            return GSTensor(
                means=self.means[key],
                scales=self.scales[key],
                quats=self.quats[key],
                opacities=self.opacities[key],
                sh0=self.sh0[key],
                shN=self.shN[key] if self.shN is not None else None,
                masks=self.masks[key] if self.masks is not None else None,
                mask_names=self.mask_names,
                _base=None,
                _format=self._format,  # Preserve format flag
            )

        # Handle boolean tensor masking
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            if len(key) != len(self):
                raise ValueError(
                    f"Boolean mask length {len(key)} doesn't match data length {len(self)}"
                )

            # Try fast path with _base
            if self._base is not None:
                result = self._slice_from_base(key)
                if result is not None:
                    return result

            # Fallback: Use boolean indexing on each tensor
            return GSTensor(
                means=self.means[key],
                scales=self.scales[key],
                quats=self.quats[key],
                opacities=self.opacities[key],
                sh0=self.sh0[key],
                shN=self.shN[key] if self.shN is not None else None,
                masks=self.masks[key] if self.masks is not None else None,
                mask_names=self.mask_names,
                _base=None,
                _format=self._format,  # Preserve format flag
            )

        # Handle integer tensor/array indexing
        if isinstance(key, (torch.Tensor, list, np.ndarray)):
            if isinstance(key, (list, np.ndarray)):
                key = torch.as_tensor(key, dtype=torch.long, device=self.device)

            # Try fast path with _base
            if self._base is not None:
                result = self._slice_from_base(key)
                if result is not None:
                    return result

            # Fallback: Use indexing on each tensor
            return GSTensor(
                means=self.means[key],
                scales=self.scales[key],
                quats=self.quats[key],
                opacities=self.opacities[key],
                sh0=self.sh0[key],
                shN=self.shN[key] if self.shN is not None else None,
                masks=self.masks[key] if self.masks is not None else None,
                mask_names=self.mask_names,
                _base=None,
                _format=self._format,  # Preserve format flag
            )

        raise TypeError(f"Invalid index type: {type(key)}")

    def get_gaussian(self, index: int) -> GSTensor:
        """Get a single Gaussian as a GSTensor object.

        :param index: Index of the Gaussian
        :returns: GSTensor with single Gaussian

        Example:
            >>> gaussian = gstensor.get_gaussian(0)  # Returns GSTensor
            >>> values = gstensor[0]  # Returns tuple
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for {len(self)} Gaussians")

        # Use slice to get GSTensor
        return self[index : index + 1]

    # ==========================================================================
    # Clone and Copy Operations
    # ==========================================================================

    def clone(self) -> GSTensor:
        """Create a deep copy of the GSTensor.

        :returns: New GSTensor with cloned tensors (independent data)

        Example:
            >>> gstensor_copy = gstensor.clone()
            >>> gstensor_copy.means[0] = 0  # Doesn't affect original
        """
        # Optimize: If we have _base, clone it and recreate views (2-3x faster)
        if self._base is not None:
            new_base = self._base.clone()
            masks_clone = self.masks.clone() if self.masks is not None else None
            mask_names_copy = self.mask_names.copy() if self.mask_names is not None else None

            result = self._recreate_from_base(
                new_base,
                format_flag=self._format,
                masks_tensor=masks_clone,
                mask_names=mask_names_copy,
            )
            if result is not None:
                return result

        # Fallback: Clone individual tensors
        return GSTensor(
            means=self.means.clone(),
            scales=self.scales.clone(),
            quats=self.quats.clone(),
            opacities=self.opacities.clone(),
            sh0=self.sh0.clone(),
            shN=self.shN.clone() if self.shN is not None else None,
            masks=self.masks.clone() if self.masks is not None else None,
            mask_names=self.mask_names.copy() if self.mask_names is not None else None,
            _base=None,
            _format=self._format,  # Preserve format flag
        )

    def __add__(self, other: GSTensor) -> GSTensor:
        """Support + operator for concatenation.

        Allows Pythonic concatenation using the + operator.

        :param other: Another GSTensor object to concatenate
        :returns: New GSTensor object with combined Gaussians

        Example:
            >>> combined = gstensor1 + gstensor2  # Same as gstensor1.add(gstensor2)
        """
        return self.add(other)

    def __radd__(self, other):
        """Support reverse addition (rarely used but completes the interface)."""
        if other == 0:
            # Allow sum([gstensor1, gstensor2, gstensor3]) to work
            return self
        return self.add(other)

    def add(self, other: GSTensor) -> GSTensor:
        """Concatenate two GSTensor objects along the Gaussian dimension (GPU-optimized).

        Combines two GSTensor objects by stacking all Gaussians. Automatically
        handles device and dtype compatibility, validates SH degrees, and merges
        mask layers.

        Performance: Uses torch.cat() which is massively parallel on GPU,
        achieving 10-100x speedup over CPU for large datasets. _base optimization
        provides additional 2-3x speedup when both tensors have consolidated bases.

        :param other: Another GSTensor object to concatenate
        :returns: New GSTensor object with combined Gaussians
        :raises ValueError: If SH degrees don't match or formats don't match

        Example:
            >>> gstensor1 = GSTensor.from_gsdata(data1, device='cuda')  # 100K Gaussians
            >>> gstensor2 = GSTensor.from_gsdata(data2, device='cuda')  # 50K Gaussians
            >>> combined = gstensor1.add(gstensor2)  # 150K Gaussians
            >>> # Or use + operator
            >>> combined = gstensor1 + gstensor2  # Same result
            >>> print(len(combined))  # 150000
        """
        # Validate compatibility
        if self.get_sh_degree() != other.get_sh_degree():
            raise ValueError(
                f"Cannot concatenate GSTensor with different SH degrees: "
                f"{self.get_sh_degree()} vs {other.get_sh_degree()}"
            )

        # Validate format equivalence
        if self._format != other._format:
            raise ValueError(
                f"Cannot concatenate GSTensor with different formats. "
                f"self: {self._format}, other: {other._format}. "
                f"Use normalize() or denormalize() to convert formats before concatenating."
            )

        # Ensure same device and dtype
        if other.device != self.device or other.dtype != self.dtype:
            other = other.to(device=self.device, dtype=self.dtype)

        # Preserve requires_grad (if either requires grad, result should too)
        requires_grad = self.means.requires_grad or other.means.requires_grad

        # Fast path: If both have _base with same format, concatenate base tensors
        if (
            self._base is not None
            and other._base is not None
            and self._base.shape[1] == other._base.shape[1]
        ):
            # Concatenate base tensors (GPU-optimized)
            combined_base = torch.cat([self._base, other._base], dim=0)
            if requires_grad:
                combined_base.requires_grad_(True)

            # Handle masks
            combined_masks = None
            combined_mask_names = None

            if self.masks is not None or other.masks is not None:
                self_masks = self.masks if self.masks is not None else None
                other_masks = other.masks if other.masks is not None else None

                if self_masks is not None and other_masks is not None:
                    # Both have masks - concatenate
                    # Ensure 2D
                    if self_masks.ndim == 1:
                        self_masks = self_masks.unsqueeze(-1)
                    if other_masks.ndim == 1:
                        other_masks = other_masks.unsqueeze(-1)

                    # Check layer count compatibility
                    if self_masks.shape[1] == other_masks.shape[1]:
                        combined_masks = torch.cat([self_masks, other_masks], dim=0)
                        # Merge names (prefer self names)
                        if self.mask_names is not None:
                            combined_mask_names = self.mask_names.copy()
                        elif other.mask_names is not None:
                            combined_mask_names = other.mask_names.copy()
                    else:
                        # Incompatible mask layers - skip masks
                        combined_masks = None
                        combined_mask_names = None
                elif self_masks is not None:
                    # Only self has masks - create False masks for other
                    if self_masks.ndim == 1:
                        other_masks_filled = torch.zeros(
                            len(other), dtype=torch.bool, device=self.device
                        )
                    else:
                        other_masks_filled = torch.zeros(
                            (len(other), self_masks.shape[1]), dtype=torch.bool, device=self.device
                        )
                    combined_masks = torch.cat([self_masks, other_masks_filled], dim=0)
                    combined_mask_names = self.mask_names.copy() if self.mask_names else None
                else:  # other_masks is not None
                    # Only other has masks - create False masks for self
                    if other_masks.ndim == 1:
                        self_masks_filled = torch.zeros(
                            len(self), dtype=torch.bool, device=self.device
                        )
                    else:
                        self_masks_filled = torch.zeros(
                            (len(self), other_masks.shape[1]), dtype=torch.bool, device=self.device
                        )
                    combined_masks = torch.cat([self_masks_filled, other_masks], dim=0)
                    combined_mask_names = other.mask_names.copy() if other.mask_names else None

            # Format already validated above, use self's format
            format_flag = self._format
            return self._recreate_from_base(
                combined_base,
                format_flag=format_flag,
                masks_tensor=combined_masks,
                mask_names=combined_mask_names,
            )

        # Fallback: Concatenate individual tensors (still GPU-optimized)
        combined_shN = None
        if self.shN is not None or other.shN is not None:
            # Ensure both have shN (use zeros if missing)
            self_shN = (
                self.shN
                if self.shN is not None
                else torch.zeros((len(self), 0, 3), dtype=self.dtype, device=self.device)
            )
            other_shN = (
                other.shN
                if other.shN is not None
                else torch.zeros((len(other), 0, 3), dtype=self.dtype, device=self.device)
            )

            if self_shN.shape[1] == other_shN.shape[1]:
                combined_shN = torch.cat([self_shN, other_shN], dim=0)
            else:
                raise ValueError(
                    f"Cannot concatenate shN with different band counts: "
                    f"{self_shN.shape[1]} vs {other_shN.shape[1]}"
                )

        # Handle masks (same logic as above)
        combined_masks = None
        combined_mask_names = None

        if self.masks is not None or other.masks is not None:
            self_masks = self.masks if self.masks is not None else None
            other_masks = other.masks if other.masks is not None else None

            if self_masks is not None and other_masks is not None:
                if self_masks.ndim == 1:
                    self_masks = self_masks.unsqueeze(-1)
                if other_masks.ndim == 1:
                    other_masks = other_masks.unsqueeze(-1)

                if self_masks.shape[1] == other_masks.shape[1]:
                    combined_masks = torch.cat([self_masks, other_masks], dim=0)
                    if self.mask_names is not None:
                        combined_mask_names = self.mask_names.copy()
                    elif other.mask_names is not None:
                        combined_mask_names = other.mask_names.copy()
            elif self_masks is not None:
                if self_masks.ndim == 1:
                    other_masks_filled = torch.zeros(
                        len(other), dtype=torch.bool, device=self.device
                    )
                else:
                    other_masks_filled = torch.zeros(
                        (len(other), self_masks.shape[1]), dtype=torch.bool, device=self.device
                    )
                combined_masks = torch.cat([self_masks, other_masks_filled], dim=0)
                combined_mask_names = self.mask_names.copy() if self.mask_names else None
            else:
                if other_masks.ndim == 1:
                    self_masks_filled = torch.zeros(len(self), dtype=torch.bool, device=self.device)
                else:
                    self_masks_filled = torch.zeros(
                        (len(self), other_masks.shape[1]), dtype=torch.bool, device=self.device
                    )
                combined_masks = torch.cat([self_masks_filled, other_masks], dim=0)
                combined_mask_names = other.mask_names.copy() if other.mask_names else None

        # Create combined GSTensor
        combined_means = torch.cat([self.means, other.means], dim=0)
        if requires_grad:
            combined_means.requires_grad_(True)

        # Format already validated above, use self's format
        format_flag = self._format

        return GSTensor(
            means=combined_means,
            scales=torch.cat([self.scales, other.scales], dim=0),
            quats=torch.cat([self.quats, other.quats], dim=0),
            opacities=torch.cat([self.opacities, other.opacities], dim=0),
            sh0=torch.cat([self.sh0, other.sh0], dim=0),
            shN=combined_shN,
            masks=combined_masks,
            mask_names=combined_mask_names,
            _base=None,  # Clear _base since we created new tensors
            _format=format_flag,  # Preserve format flag
        )

    def unpack(self, include_shN: bool = True) -> tuple:
        """Unpack Gaussian data into tuple of tensors.

        Convenient for standard Gaussian Splatting workflows that expect
        individual tensors rather than a container object.

        :param include_shN: If True, include shN in output (default True)
        :returns: If include_shN=True: (means, scales, quats, opacities, sh0, shN),
                  If include_shN=False: (means, scales, quats, opacities, sh0)

        Example:
            >>> data = plyread("scene.ply")
            >>> gstensor = GSTensor.from_gsdata(data, device='cuda')
            >>> means, scales, quats, opacities, sh0, shN = gstensor.unpack()
            >>> # Use with rendering functions
            >>> render(means, scales, quats, opacities, sh0)
            >>>
            >>> # For SH0 data, exclude shN
            >>> means, scales, quats, opacities, sh0 = gstensor.unpack(include_shN=False)
        """
        if include_shN:
            return (self.means, self.scales, self.quats, self.opacities, self.sh0, self.shN)
        return (self.means, self.scales, self.quats, self.opacities, self.sh0)

    def to_dict(self) -> dict:
        """Convert Gaussian data to dictionary.

        :returns: Dictionary with keys: means, scales, quats, opacities, sh0, shN

        Example:
            >>> gstensor = GSTensor.from_gsdata(data, device='cuda')
            >>> props = gstensor.to_dict()
            >>> # Access by key
            >>> positions = props['means']
            >>> # Unpack dict values
            >>> render(**props)
        """
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "sh0": self.sh0,
            "shN": self.shN,
        }

    def save(self, file_path: str | Path, compressed: bool = True) -> None:
        """Save GSTensor to PLY file.

        Convenience method for saving GSTensor. Uses GPU compression when
        compressed=True, otherwise converts to GSData and saves uncompressed.

        :param file_path: Output PLY file path
        :param compressed: If True, use GPU compression (default True).
                           If False, convert to GSData and save uncompressed.

        Performance:
            - Compressed: 5-20x faster compression than CPU Numba
            - Uncompressed: Converts to GSData first (CPU transfer)

        Example:
            >>> gstensor = GSTensor.from_gsdata(data, device="cuda")
            >>> gstensor.save("output.compressed.ply")  # GPU compressed (default)
            >>> gstensor.save("output.ply", compressed=False)  # Uncompressed
        """
        file_path = Path(file_path)

        if compressed:
            write_compressed_gpu(file_path, self)
        else:
            # Convert to GSData and save uncompressed
            from gsply.writer import plywrite

            gsdata = self.to_gsdata()
            plywrite(file_path, gsdata, compressed=False)

    def save_compressed(self, file_path: str | Path) -> None:
        """Save GSTensor to compressed PLY file using GPU compression.

        Performs compression on GPU for maximum performance. Faster than
        downloading to CPU and using CPU compression.

        :param file_path: Output file path

        Performance:
            - 5-20x faster compression than CPU Numba
            - GPU reduction for chunk bounds (instant)
            - Minimal CPU-GPU data transfer

        Format:
            - PlayCanvas compressed PLY format
            - 3.8-14.5x compression ratio
            - 256 Gaussians per chunk with quantization

        Example:
            >>> gstensor = GSTensor.from_gsdata(data, device="cuda")
            >>> gstensor.save_compressed("output.ply_compressed")
            >>> # File is ~14x smaller than uncompressed

        Note:
            This is a convenience alias for save(file_path, compressed=True).
            Use save() for more flexibility.
        """
        self.save(file_path, compressed=True)

    @classmethod
    def load(cls, file_path: str | Path, device: str | torch.device = "cuda") -> GSTensor:
        """Load GSTensor from PLY file.

        Convenience classmethod that auto-detects format and loads to GPU.
        Uses GPU decompression for compressed files, CPU read + GPU transfer for uncompressed.

        :param file_path: Path to PLY file
        :param device: Target device (default "cuda")
        :returns: GSTensor with loaded data on GPU

        Performance:
            - Compressed: GPU decompression (4-5x faster than CPU)
            - Uncompressed: CPU read + GPU transfer

        Example:
            >>> gstensor = GSTensor.load("scene.ply", device="cuda")
            >>> print(f"Loaded {len(gstensor):,} Gaussians on GPU")
        """
        file_path = Path(file_path)

        # Auto-detect format from extension
        is_compressed = file_path.name.endswith((".ply_compressed", ".compressed.ply"))

        if is_compressed:
            # Use GPU decompression
            return cls.from_compressed(file_path, device=device)

        # Read uncompressed on CPU, then transfer to GPU
        from gsply.reader import plyread

        gsdata = plyread(file_path)
        return cls.from_gsdata(gsdata, device=device)

    @classmethod
    def from_arrays(
        cls,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor | None = None,
        format: str = "auto",
        sh_degree: int | None = None,
        sh0_format: DataFormat = DataFormat.SH0_SH,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> GSTensor:
        """Create GSTensor from individual tensors with format preset.

        Convenient factory method for creating GSTensor from external tensors
        with automatic format detection or explicit format presets.

        :param means: (N, 3) tensor - Gaussian centers
        :param scales: (N, 3) tensor - Scale parameters
        :param quats: (N, 4) tensor - Rotation quaternions
        :param opacities: (N,) tensor - Opacity values
        :param sh0: (N, 3) tensor - DC spherical harmonics
        :param shN: (N, K, 3) tensor or None - Higher-order SH coefficients
        :param format: Format preset - "auto" (detect), "ply" (log/logit), "linear" or "rasterizer" (linear)
        :param sh_degree: SH degree (0-3) - auto-detected from shN if None
        :param sh0_format: Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH
        :param device: Target device - inferred from tensors if None
        :param dtype: Target dtype - inferred from tensors if None
        :returns: GSTensor object with specified format

        Example:
            >>> # Auto-detect format from values
            >>> gstensor = GSTensor.from_arrays(means, scales, quats, opacities, sh0, device="cuda")
            >>>
            >>> # Explicit PLY format (log-scales, logit-opacities)
            >>> gstensor = GSTensor.from_arrays(means, scales, quats, opacities, sh0, format="ply", device="cuda")
            >>>
            >>> # Explicit linear format (for rasterizer)
            >>> gstensor = GSTensor.from_arrays(means, scales, quats, opacities, sh0, format="linear", device="cuda")
        """
        # Infer device and dtype from first tensor if not provided
        if device is None:
            device = means.device
        if dtype is None:
            dtype = means.dtype

        # Ensure all tensors are on correct device and dtype
        device_obj = torch.device(device)
        means = means.to(device=device_obj, dtype=dtype)
        scales = scales.to(device=device_obj, dtype=dtype)
        quats = quats.to(device=device_obj, dtype=dtype)
        opacities = opacities.to(device=device_obj, dtype=dtype)
        sh0 = sh0.to(device=device_obj, dtype=dtype)
        if shN is not None:
            shN = shN.to(device=device_obj, dtype=dtype)

        # Determine SH degree
        if sh_degree is None:
            if shN is not None and shN.shape[1] > 0:
                sh_bands = int(shN.shape[1])
                sh_degree = SH_BANDS_TO_DEGREE.get(sh_bands, 0)
            else:
                sh_degree = 0

        # Create format dict based on preset
        if format == "auto":
            # Auto-detect format from values (convert to numpy for detection)
            scales_np = scales.detach().cpu().numpy()
            opacities_np = opacities.detach().cpu().numpy()
            scales_format, opacities_format = _detect_format_from_values(scales_np, opacities_np)
            format_dict = _create_format_dict(
                scales=scales_format,
                opacities=opacities_format,
                sh0=sh0_format,
                sh_order=_get_sh_order_format(sh_degree),
                means=DataFormat.MEANS_RAW,
                quats=DataFormat.QUATS_RAW,
            )
        elif format == "ply":
            # PLY format (log-scales, logit-opacities)
            format_dict = create_ply_format(sh_degree=sh_degree, sh0_format=sh0_format)
        elif format in ("linear", "rasterizer"):
            # Linear/rasterizer format (linear scales, linear opacities)
            format_dict = create_rasterizer_format(sh_degree=sh_degree, sh0_format=sh0_format)
        else:
            raise ValueError(
                f"Invalid format preset: {format}. Must be 'auto', 'ply', 'linear', or 'rasterizer'"
            )

        return cls(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            masks=None,
            mask_names=None,
            _base=None,
            _format=format_dict,
        )

    @classmethod
    def from_dict(
        cls,
        data_dict: dict,
        format: str = "auto",
        sh_degree: int | None = None,
        sh0_format: DataFormat = DataFormat.SH0_SH,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> GSTensor:
        """Create GSTensor from dictionary with format preset.

        Convenient factory method for creating GSTensor from a dictionary
        with automatic format detection or explicit format presets.

        :param data_dict: Dictionary with keys: means, scales, quats, opacities, sh0, shN (optional)
        :param format: Format preset - "auto" (detect), "ply" (log/logit), "linear" or "rasterizer" (linear)
        :param sh_degree: SH degree (0-3) - auto-detected from shN if None
        :param sh0_format: Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH
        :param device: Target device (default "cuda")
        :param dtype: Target dtype - inferred from tensors if None
        :returns: GSTensor object with specified format

        Example:
            >>> # From dictionary with auto-detection
            >>> gstensor = GSTensor.from_dict({
            ...     "means": means, "scales": scales, "quats": quats,
            ...     "opacities": opacities, "sh0": sh0, "shN": shN
            ... }, device="cuda")
            >>>
            >>> # Explicit PLY format
            >>> gstensor = GSTensor.from_dict(data_dict, format="ply", device="cuda")
            >>>
            >>> # Explicit linear format
            >>> gstensor = GSTensor.from_dict(data_dict, format="linear", device="cuda")
        """
        return cls.from_arrays(
            means=data_dict["means"],
            scales=data_dict["scales"],
            quats=data_dict["quats"],
            opacities=data_dict["opacities"],
            sh0=data_dict["sh0"],
            shN=data_dict.get("shN"),
            format=format,
            sh_degree=sh_degree,
            sh0_format=sh0_format,
            device=device,
            dtype=dtype,
        )

    # ==========================================================================
    # Type Conversions
    # ==========================================================================

    def to_dtype(self, dtype: torch.dtype) -> GSTensor:
        """Convert tensors to specified dtype.

        :param dtype: Target dtype (e.g., torch.float16, torch.float32, torch.float64)
        :returns: New GSTensor with converted dtype

        Example:
            >>> gstensor_half = gstensor.to_dtype(torch.float16)
        """
        return self.to(dtype=dtype)

    def half(self) -> GSTensor:
        """Convert to float16 (half precision).

        :returns: New GSTensor with float16 dtype
        """
        return self.to_dtype(torch.float16)

    def float(self) -> GSTensor:
        """Convert to float32 (single precision).

        :returns: New GSTensor with float32 dtype
        """
        return self.to_dtype(torch.float32)

    def double(self) -> GSTensor:
        """Convert to float64 (double precision).

        :returns: New GSTensor with float64 dtype
        """
        return self.to_dtype(torch.float64)

    # ==========================================================================
    # Mask Layer Management (GPU-Optimized)
    # ==========================================================================

    def add_mask_layer(self, name: str, mask: torch.Tensor) -> None:
        """Add a named boolean mask layer.

        :param name: Name for this mask layer
        :param mask: Boolean tensor of shape (N,) where N is number of Gaussians
        :raises ValueError: If mask shape doesn't match data length or name already exists

        Example:
            >>> gstensor.add_mask_layer("high_opacity", gstensor.opacities > 0.5)
            >>> gstensor.add_mask_layer("foreground", gstensor.means[:, 2] < 0)
            >>> print(gstensor.mask_names)  # ['high_opacity', 'foreground']
        """
        # Convert to tensor if needed and ensure boolean type
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(dtype=torch.bool, device=self.device)

        if mask.shape != (len(self),):
            raise ValueError(f"Mask shape {mask.shape} doesn't match data length ({len(self)},)")

        # Check for duplicate names
        if self.mask_names is not None and name in self.mask_names:
            raise ValueError(f"Mask layer '{name}' already exists")

        # Initialize or append to masks
        if self.masks is None:
            self.masks = mask.unsqueeze(-1)  # Shape (N, 1)
            self.mask_names = [name]
        else:
            # Ensure masks is 2D
            if self.masks.ndim == 1:
                self.masks = self.masks.unsqueeze(-1)
            self.masks = torch.cat([self.masks, mask.unsqueeze(-1)], dim=1)
            if self.mask_names is None:
                self.mask_names = [f"layer_{i}" for i in range(self.masks.shape[1] - 1)]
            self.mask_names.append(name)

    def get_mask_layer(self, name: str) -> torch.Tensor:
        """Get a mask layer by name.

        :param name: Name of the mask layer
        :returns: Boolean tensor of shape (N,)
        :raises ValueError: If layer name not found

        Example:
            >>> opacity_mask = gstensor.get_mask_layer("high_opacity")
        """
        if self.mask_names is None or name not in self.mask_names:
            raise ValueError(f"Mask layer '{name}' not found")

        layer_idx = self.mask_names.index(name)
        if self.masks.ndim == 1:
            return self.masks
        return self.masks[:, layer_idx]

    def remove_mask_layer(self, name: str) -> None:
        """Remove a mask layer by name.

        :param name: Name of the mask layer to remove
        :raises ValueError: If layer name not found

        Example:
            >>> gstensor.remove_mask_layer("foreground")
        """
        if self.mask_names is None or name not in self.mask_names:
            raise ValueError(f"Mask layer '{name}' not found")

        layer_idx = self.mask_names.index(name)

        # Remove from masks
        if self.masks.ndim == 1:
            # Single layer - clear everything
            self.masks = None
            self.mask_names = None
        else:
            # Multi-layer - remove one column efficiently using indexing
            n_layers = self.masks.shape[1]
            if n_layers == 1:
                # Only one layer - clear everything
                self.masks = None
                self.mask_names = None
            elif n_layers == 2:
                # Two layers - keep the other one as 1D
                other_idx = 1 - layer_idx
                self.masks = self.masks[:, other_idx]
                self.mask_names = [n for n in self.mask_names if n != name]
            else:
                # 3+ layers - use boolean indexing to remove column
                indices = torch.tensor(
                    [i for i in range(n_layers) if i != layer_idx],
                    dtype=torch.long,
                    device=self.device,
                )
                self.masks = self.masks[:, indices]
                self.mask_names = [n for n in self.mask_names if n != name]

    def combine_masks(self, mode: str = "and", layers: list[str] | None = None) -> torch.Tensor:
        """Combine mask layers using boolean logic (GPU-optimized).

        Uses PyTorch's native GPU operations for massive speedup:
        - torch.all() for AND: 100-1000x faster than CPU Numba
        - torch.any() for OR: 100-1000x faster than CPU Numba

        :param mode: Combination mode - "and" (all must pass) or "or" (any must pass)
        :param layers: List of layer names to combine (None = use all layers)
        :returns: Combined boolean tensor of shape (N,)
        :raises ValueError: If no masks exist or invalid mode

        Example:
            >>> # Combine all layers with AND
            >>> mask = gstensor.combine_masks(mode="and")
            >>> filtered = gstensor[mask]
            >>>
            >>> # Combine specific layers with OR
            >>> mask = gstensor.combine_masks(mode="or", layers=["opacity", "foreground"])
        """
        if self.masks is None:
            raise ValueError("No mask layers exist")

        if mode not in ("and", "or"):
            raise ValueError(f"Mode must be 'and' or 'or', got '{mode}'")

        # Get mask tensor
        if layers is None:
            # Use all layers
            if self.masks.ndim == 1:
                return self.masks
            masks_to_combine = self.masks
        else:
            # Select specific layers
            if self.mask_names is None:
                raise ValueError("Cannot select layers by name - no layer names set")
            indices = [self.mask_names.index(name) for name in layers]
            if self.masks.ndim == 1:
                if len(indices) != 1 or indices[0] != 0:
                    raise ValueError(f"Invalid layer selection: {layers}")
                return self.masks
            indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
            masks_to_combine = self.masks[:, indices_tensor]

        # GPU-optimized combination using PyTorch native operations
        # These are massively parallel and 100-1000x faster than CPU Numba
        if masks_to_combine.ndim == 1:
            # Single layer - return as-is
            return masks_to_combine

        n_layers = masks_to_combine.shape[1]

        if n_layers == 1:
            # Technically 2D but only 1 layer - flatten
            return masks_to_combine[:, 0]

        # 2+ layers: Use PyTorch native operations (GPU-optimized)
        if mode == "and":
            return torch.all(masks_to_combine, dim=1)
        # mode == "or"
        return torch.any(masks_to_combine, dim=1)

    def apply_masks(
        self, mode: str = "and", layers: list[str] | None = None, inplace: bool = False
    ) -> GSTensor:
        """Apply mask layers to filter Gaussians.

        :param mode: Combination mode - "and" or "or"
        :param layers: List of layer names to apply (None = all layers)
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Filtered GSTensor (self if inplace=True, new object if inplace=False)

        Example:
            >>> # Filter using all mask layers (AND logic)
            >>> filtered = gstensor.apply_masks(mode="and")
            >>>
            >>> # Filter in-place using specific layers (OR logic)
            >>> gstensor.apply_masks(mode="or", layers=["opacity", "scale"], inplace=True)
        """
        combined_mask = self.combine_masks(mode=mode, layers=layers)

        if inplace:
            # Filter tensors in-place (replace with filtered versions)
            self.means = self.means[combined_mask]
            self.scales = self.scales[combined_mask]
            self.quats = self.quats[combined_mask]
            self.opacities = self.opacities[combined_mask]
            self.sh0 = self.sh0[combined_mask]
            if self.shN is not None:
                self.shN = self.shN[combined_mask]
            if self.masks is not None:
                if self.masks.ndim == 1:
                    self.masks = self.masks[combined_mask]
                else:
                    self.masks = self.masks[combined_mask, :]
            if self._base is not None:
                self._base = self._base[combined_mask]
            return self
        # Return filtered copy (leverages existing __getitem__ with _base optimization)
        return self[combined_mask]

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def __repr__(self) -> str:
        """String representation of GSTensor."""
        sh_degree = self.get_sh_degree()
        mask_info = "None"
        if self.masks is not None:
            if self.masks.ndim == 1:
                mask_info = "1 layer"
            else:
                mask_info = f"{self.masks.shape[1]} layers"
            if self.mask_names is not None:
                mask_info += f" ({', '.join(self.mask_names)})"

        return (
            f"GSTensor(\n"
            f"  Gaussians: {len(self):,}\n"
            f"  SH degree: {sh_degree}\n"
            f"  Device: {self.device}\n"
            f"  Dtype: {self.dtype}\n"
            f"  Has _base: {self._base is not None}\n"
            f"  Masks: {mask_info}\n"
            f")"
        )
