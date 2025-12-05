"""Gaussian Splatting data container."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypedDict

import numba
import numpy as np
from numba import jit, prange

from gsply.formats import SH_BANDS_TO_DEGREE

# Lazy imports to avoid circular dependencies
# These are imported inside methods to break circular import cycles
# (writer.py and reader.py import GSData, so we can't import them at module level)


# ======================================================================================
# JIT-COMPILED INTERLEAVING KERNELS (Optimization for consolidate/write)
# ======================================================================================


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _interleave_sh0_jit(
    means: np.ndarray,
    sh0: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    output: np.ndarray,
) -> None:
    """JIT-compiled interleaving for SH0 data (14 properties).

    Fused parallel kernel for optimal cache utilization.
    5x faster than slice assignment for 400K Gaussians.

    :param means: (N, 3) float32 positions
    :param sh0: (N, 3) float32 DC spherical harmonics
    :param opacities: (N,) float32 opacity values
    :param scales: (N, 3) float32 scale parameters
    :param quats: (N, 4) float32 rotation quaternions
    :param output: (N, 14) float32 output array (pre-allocated)
    """
    n = len(means)
    for i in prange(n):
        # Means (0-2)
        output[i, 0] = means[i, 0]
        output[i, 1] = means[i, 1]
        output[i, 2] = means[i, 2]
        # SH0 (3-5)
        output[i, 3] = sh0[i, 0]
        output[i, 4] = sh0[i, 1]
        output[i, 5] = sh0[i, 2]
        # Opacity (6)
        output[i, 6] = opacities[i]
        # Scales (7-9)
        output[i, 7] = scales[i, 0]
        output[i, 8] = scales[i, 1]
        output[i, 9] = scales[i, 2]
        # Quats (10-13)
        output[i, 10] = quats[i, 0]
        output[i, 11] = quats[i, 1]
        output[i, 12] = quats[i, 2]
        output[i, 13] = quats[i, 3]


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True, boundscheck=False)
def _interleave_shn_jit(
    means: np.ndarray,
    sh0: np.ndarray,
    shn_flat: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    output: np.ndarray,
    sh_coeffs: int,
) -> None:
    """JIT-compiled interleaving for SH1-3 data (variable properties).

    Fused parallel kernel for optimal cache utilization.
    2.8x faster than slice assignment for 400K SH3 Gaussians.

    :param means: (N, 3) float32 positions
    :param sh0: (N, 3) float32 DC spherical harmonics
    :param shn_flat: (N, sh_coeffs) float32 flattened higher-order SH
    :param opacities: (N,) float32 opacity values
    :param scales: (N, 3) float32 scale parameters
    :param quats: (N, 4) float32 rotation quaternions
    :param output: (N, 14 + sh_coeffs) float32 output array (pre-allocated)
    :param sh_coeffs: Number of SH coefficients (9, 24, or 45)
    """
    n = len(means)
    opacity_idx = 6 + sh_coeffs

    for i in prange(n):
        # Means (0-2)
        output[i, 0] = means[i, 0]
        output[i, 1] = means[i, 1]
        output[i, 2] = means[i, 2]
        # SH0 (3-5)
        output[i, 3] = sh0[i, 0]
        output[i, 4] = sh0[i, 1]
        output[i, 5] = sh0[i, 2]
        # ShN (6 to 6+sh_coeffs-1)
        for j in range(sh_coeffs):
            output[i, 6 + j] = shn_flat[i, j]
        # Opacity
        output[i, opacity_idx] = opacities[i]
        # Scales
        output[i, opacity_idx + 1] = scales[i, 0]
        output[i, opacity_idx + 2] = scales[i, 1]
        output[i, opacity_idx + 3] = scales[i, 2]
        # Quats
        output[i, opacity_idx + 4] = quats[i, 0]
        output[i, opacity_idx + 5] = quats[i, 1]
        output[i, opacity_idx + 6] = quats[i, 2]
        output[i, opacity_idx + 7] = quats[i, 3]


class DataFormat(Enum):
    """Format tracking for individual attributes - each value specifies attribute and format."""

    # Scales formats
    SCALES_PLY = "scales_ply"  # log-scales (log(scale))
    SCALES_LINEAR = "scales_linear"  # linear scales (scale)

    # Opacities formats
    OPACITIES_PLY = "opacities_ply"  # logit-opacities (logit(opacity))
    OPACITIES_LINEAR = "opacities_linear"  # linear opacities (opacity in [0, 1])

    # Spherical harmonics formats (for colors)
    SH0_SH = "sh0_sh"  # spherical harmonics format (mathematical representation)
    SH0_RGB = "sh0_rgb"  # RGB color format (visual representation, converted from SH)

    # Spherical harmonics order (for shN)
    SH_ORDER_0 = "sh_order_0"  # SH degree 0 (no shN, only sh0)
    SH_ORDER_1 = "sh_order_1"  # SH degree 1 (3 bands)
    SH_ORDER_2 = "sh_order_2"  # SH degree 2 (8 bands)
    SH_ORDER_3 = "sh_order_3"  # SH degree 3 (15 bands)

    # Raw formats (no conversion)
    MEANS_RAW = "means_raw"  # raw format (no conversion)
    QUATS_RAW = "quats_raw"  # raw format (no conversion)


# Type-safe format dictionary (TypedDict for IDE autocomplete and type checking)
class FormatDict(TypedDict, total=False):
    """Type-safe format dictionary - provides IDE autocomplete and type checking.

    All fields are optional (total=False) to allow partial format tracking.
    """

    scales: DataFormat
    opacities: DataFormat
    sh0: DataFormat
    sh_order: DataFormat
    means: DataFormat
    quats: DataFormat


# Mapping from SH degree to format enum (module-level constant for performance)
_SH_DEGREE_TO_FORMAT: dict[int, DataFormat] = {
    0: DataFormat.SH_ORDER_0,
    1: DataFormat.SH_ORDER_1,
    2: DataFormat.SH_ORDER_2,
    3: DataFormat.SH_ORDER_3,
}


def _create_format_dict(
    scales: DataFormat | None = None,
    opacities: DataFormat | None = None,
    sh0: DataFormat | None = None,
    sh_order: DataFormat | None = None,
    means: DataFormat | None = None,
    quats: DataFormat | None = None,
) -> FormatDict:
    """Create format dict for GSData attributes.

    :param scales: Format for scales (DataFormat.SCALES_PLY or DataFormat.SCALES_LINEAR)
    :param opacities: Format for opacities (DataFormat.OPACITIES_PLY or DataFormat.OPACITIES_LINEAR)
    :param sh0: Format for sh0 (DataFormat.SH0_SH or DataFormat.SH0_RGB)
    :param sh_order: SH order/degree for shN (DataFormat.SH_ORDER_0/1/2/3)
    :param means: Format for means (DataFormat.MEANS_RAW)
    :param quats: Format for quats (DataFormat.QUATS_RAW)
    :returns: Format dict with all non-None attributes
    """
    format_mapping = {
        "scales": scales,
        "opacities": opacities,
        "sh0": sh0,
        "sh_order": sh_order,
        "means": means,
        "quats": quats,
    }
    return {key: value for key, value in format_mapping.items() if value is not None}


def _get_sh_order_format(sh_degree: int) -> DataFormat:
    """Get SH order format enum from SH degree.

    :param sh_degree: SH degree (0-3)
    :returns: DataFormat enum for SH order
    :raises ValueError: If sh_degree is not in range 0-3
    """
    if sh_degree not in _SH_DEGREE_TO_FORMAT:
        raise ValueError(f"Invalid SH degree: {sh_degree}, must be 0-3")
    return _SH_DEGREE_TO_FORMAT[sh_degree]


def create_ply_format(sh_degree: int = 0, sh0_format: DataFormat = DataFormat.SH0_SH) -> FormatDict:
    """Create format dict for PLY file format (log-scales, logit-opacities).

    This is the standard format used when loading from raw PLY files.
    Use this when creating GSData from data that matches PLY file format
    or when you want to ensure compatibility with PLY file format.

    Format details:
    - Scales: log-scales (log(scale)) - PLY format
    - Opacities: logit-opacities (logit(opacity)) - PLY format
    - Colors: SH format (spherical harmonics)

    :param sh_degree: Spherical harmonics degree (0-3), default 0
    :param sh0_format: Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH
    :returns: Format dict with PLY format settings

    Example:
        >>> # Create GSData matching PLY file format (loaded from raw PLY)
        >>> format_dict = create_ply_format(sh_degree=3)
        >>> data = GSData(means=..., scales=..., _format=format_dict)
    """
    return _create_format_dict(
        scales=DataFormat.SCALES_PLY,
        opacities=DataFormat.OPACITIES_PLY,
        sh0=sh0_format,
        sh_order=_get_sh_order_format(sh_degree),
        means=DataFormat.MEANS_RAW,
        quats=DataFormat.QUATS_RAW,
    )


def create_rasterizer_format(
    sh_degree: int = 0, sh0_format: DataFormat = DataFormat.SH0_SH
) -> FormatDict:
    """Create format dict for rasterizer format (linear scales, linear opacities).

    This is the format expected by gsplat rasterizer and other rendering pipelines.
    Use this when creating GSData for rasterization or when you need linear values
    for computation and visualization.

    Format details:
    - Scales: linear scales (scale) - rasterizer format
    - Opacities: linear opacities (opacity in [0, 1]) - rasterizer format
    - Colors: SH format (spherical harmonics)

    :param sh_degree: Spherical harmonics degree (0-3), default 0
    :param sh0_format: Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH
    :returns: Format dict with rasterizer format settings

    Example:
        >>> # Create GSData for gsplat rasterizer (linear format)
        >>> format_dict = create_rasterizer_format(sh_degree=3)
        >>> data = GSData(means=..., scales=..., _format=format_dict)
        >>> # Data is ready to pass to rasterizer
    """
    return _create_format_dict(
        scales=DataFormat.SCALES_LINEAR,
        opacities=DataFormat.OPACITIES_LINEAR,
        sh0=sh0_format,
        sh_order=_get_sh_order_format(sh_degree),
        means=DataFormat.MEANS_RAW,
        quats=DataFormat.QUATS_RAW,
    )


def _detect_format_from_values(
    scales: np.ndarray, opacities: np.ndarray
) -> tuple[DataFormat, DataFormat]:
    """Detect format from scale and opacity values (heuristic).

    Uses heuristics to detect if data is in PLY format (log-scales, logit-opacities)
    or linear format. Defaults to PLY format if uncertain (backward compatibility).

    Heuristics:
    - Scales: PLY format (log-scales) typically has many negative values
    - Opacities: PLY format (logit-opacities) typically has values outside [0, 1]
    - Linear scales are typically positive and small (< 10)
    - Linear opacities are typically in [0, 1] range

    :param scales: Scale array (N, 3)
    :param opacities: Opacity array (N,)
    :returns: Tuple of (scales_format, opacities_format) - always returns valid formats
    """
    # Handle empty arrays - default to PLY format
    if scales.size == 0 or opacities.size == 0:
        return DataFormat.SCALES_PLY, DataFormat.OPACITIES_PLY

    # Check scales: PLY format (log-scales) often has negative values
    # Linear scales are typically positive
    scales_flat = scales.flatten()
    negative_ratio = np.sum(scales_flat < 0) / scales_flat.size
    max_scale = np.max(np.abs(scales_flat))

    # If many negative values or very large values, likely PLY format (log-scales)
    if negative_ratio > 0.1 or max_scale > 10.0:
        scales_format = DataFormat.SCALES_PLY
    # If all positive and small, likely linear
    elif negative_ratio == 0.0 and max_scale < 10.0:
        scales_format = DataFormat.SCALES_LINEAR
    else:
        # Uncertain: default to PLY format (backward compatibility)
        scales_format = DataFormat.SCALES_PLY

    # Check opacities: PLY format (logit-opacities) often outside [0, 1]
    # Linear opacities are typically in [0, 1]
    in_range_ratio = np.sum((opacities >= 0) & (opacities <= 1)) / opacities.size

    # If mostly outside [0, 1], likely PLY format (logit-opacities)
    if in_range_ratio < 0.9:
        opacities_format = DataFormat.OPACITIES_PLY
    # If mostly in [0, 1], likely linear
    elif in_range_ratio > 0.95:
        opacities_format = DataFormat.OPACITIES_LINEAR
    else:
        # Uncertain: default to PLY format (backward compatibility)
        opacities_format = DataFormat.OPACITIES_PLY

    return scales_format, opacities_format


# Numba-optimized mask combination (37-68x faster than numpy.all())
@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _combine_masks_numba_and(masks):
    """Combine masks with AND logic using parallel Numba.

    Benchmarks (100K Gaussians, 5 layers):
    - numpy.all(): 1.43ms (72M/sec)
    - numba parallel: 0.039ms (2,550M/sec) - 37x faster!

    :param masks: Boolean array of shape (N, L) where L >= 2
    :returns: Boolean array of shape (N,) - result of AND across layers
    """
    n, m = masks.shape
    result = np.empty(n, dtype=np.bool_)

    for i in numba.prange(n):
        val = True
        for j in range(m):
            if not masks[i, j]:
                val = False
                break  # Short-circuit
        result[i] = val

    return result


@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _combine_masks_numba_or(masks):
    """Combine masks with OR logic using parallel Numba.

    :param masks: Boolean array of shape (N, L) where L >= 2
    :returns: Boolean array of shape (N,) - result of OR across layers
    """
    n, m = masks.shape
    result = np.empty(n, dtype=np.bool_)

    for i in numba.prange(n):
        val = False
        for j in range(m):
            if masks[i, j]:
                val = True
                break  # Short-circuit
        result[i] = val

    return result


@dataclass
class GSData:
    """Gaussian Splatting data container.

    This container holds Gaussian parameters, either as separate arrays
    or as zero-copy views into a single base array for maximum performance.
    Implemented as a mutable dataclass with direct attribute access.

    Attributes:
        means: (N, 3) - xyz positions
        scales: (N, 3) - scale parameters
            - PLY format: log-scales (log(scale))
            - LINEAR format: linear scales (scale)
        quats: (N, 4) - rotation quaternions
        opacities: (N,) - opacity values
            - PLY format: logit-opacities (logit(opacity))
            - LINEAR format: linear opacities (opacity in [0, 1])
        sh0: (N, 3) - DC spherical harmonics (always SH format)
        shN: (N, K, 3) - Higher-order SH coefficients (K bands) (always SH format)
        masks: (N,) or (N, L) - Boolean mask layers for filtering (None = no masks)
        mask_names: list[str] - Names for each mask layer (None = unnamed layers)
        _base: (N, P) - Private base array (keeps memory alive for views, None otherwise)
        _format: FormatDict - Format tracking per attribute (type-safe TypedDict)
            - Format: {"scales": DataFormat.SCALES_PLY, "opacities": DataFormat.OPACITIES_PLY, ...}
            - Scales: DataFormat.SCALES_PLY (log-scales) or DataFormat.SCALES_LINEAR (linear scales)
            - Opacities: DataFormat.OPACITIES_PLY (logit-opacities) or DataFormat.OPACITIES_LINEAR (linear opacities)
            - Colors: DataFormat.SH0_SH (sh0 as SH) or DataFormat.SH0_RGB (sh0 as RGB)
            - SH Order: DataFormat.SH_ORDER_0/1/2/3 (spherical harmonics degree for shN)
            - Positions/Rotations: DataFormat.MEANS_RAW (means) and DataFormat.QUATS_RAW (quats) - raw format
            - Always provided when creating GSData (auto-detected if not specified)

    Mask Layers:
        - Single layer: masks shape (N,), mask_names = None or ["name"]
        - Multi-layer: masks shape (N, L), mask_names = ["name1", "name2", ...]
        - Use add_mask_layer() to add named layers
        - Use combine_masks() to merge layers with AND/OR logic
        - Use apply_masks() to filter data using mask layers

    Performance:
        - Zero-copy reads provide maximum performance
        - No memory overhead (views share memory with base)

    Example:
        >>> data = plyread("scene.ply")
        >>> print(f"Loaded {len(data)} Gaussians")
        >>> # Add named mask layers
        >>> data.add_mask_layer("high_opacity", data.opacities > 0.5)
        >>> data.add_mask_layer("foreground", data.means[:, 2] < 0)
        >>> # Combine and apply
        >>> filtered = data.apply_masks(mode="and")
    """

    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh0: np.ndarray
    shN: np.ndarray  # noqa: N815
    _format: FormatDict = field(
        default_factory=lambda: {}
    )  # Format tracking - auto-detected in __post_init__ if empty
    masks: np.ndarray | None = None  # Boolean mask layers (N,) or (N, L)
    mask_names: list[str] | None = None  # Names for each mask layer
    _base: np.ndarray | None = None  # Private field for zero-copy views

    def __post_init__(self):
        """Auto-detect format if not provided."""
        # Copy format dict to avoid sharing mutable state between instances
        self._format = dict(self._format)

        # If _format is empty dict, auto-detect from values
        if not self._format:
            scales_format, opacities_format = _detect_format_from_values(
                self.scales, self.opacities
            )
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

    def get_sh_degree(self) -> int:
        """Get SH degree from shN shape.

        :returns: SH degree (0-3)
        """
        if self.shN is None or self.shN.shape[1] == 0:
            return 0
        # shN.shape[1] is number of bands (K)
        sh_bands = self.shN.shape[1]
        return SH_BANDS_TO_DEGREE.get(sh_bands, 0)

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
            >>> data = gsply.plyread("scene.ply")
            >>> fmt = data.format_state
            >>> print(fmt)  # {'scales': DataFormat.SCALES_PLY, ...}
        """
        return dict(self._format)

    def copy_format_from(self, other: "GSData") -> None:
        """Copy format tracking from another GSData object.

        This is the public API for copying format state between objects.
        Use this instead of directly accessing _format dict.

        :param other: Source GSData to copy format from

        Example:
            >>> # After processing that might lose format
            >>> processed.copy_format_from(original)
        """
        self._format = dict(other._format)

    def with_format(self, **updates) -> "GSData":
        """Create a copy with updated format settings.

        Returns a new GSData with the same data but updated format dict.
        This is useful for explicitly setting format after operations.

        :param updates: Format updates (keys: scales, opacities, sh0, sh_order)
        :returns: New GSData with updated format

        Example:
            >>> # Mark data as having linear opacities after conversion
            >>> linear_data = data.with_format(opacities=DataFormat.OPACITIES_LINEAR)
        """
        new_format = dict(self._format)
        for key, value in updates.items():
            if key in ("scales", "opacities", "sh0", "sh_order", "means", "quats"):
                new_format[key] = value
            else:
                raise ValueError(f"Unknown format key: {key}")

        return GSData(
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

    def add_mask_layer(self, name: str, mask: np.ndarray) -> None:
        """Add a named boolean mask layer.

        :param name: Name for this mask layer
        :param mask: Boolean array of shape (N,) where N is number of Gaussians
        :raises ValueError: If mask shape doesn't match data length or name already exists

        Example:
            >>> data.add_mask_layer("high_opacity", data.opacities > 0.5)
            >>> data.add_mask_layer("foreground", data.means[:, 2] < 0)
            >>> print(data.mask_names)  # ['high_opacity', 'foreground']
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError(f"Mask shape {mask.shape} doesn't match data length ({len(self)},)")

        # Check for duplicate names
        if self.mask_names is not None and name in self.mask_names:
            raise ValueError(f"Mask layer '{name}' already exists")

        # Initialize or append to masks
        if self.masks is None:
            self.masks = mask[:, None]  # Shape (N, 1)
            self.mask_names = [name]
        else:
            # Ensure masks is 2D
            if self.masks.ndim == 1:
                self.masks = self.masks[:, None]
            self.masks = np.column_stack([self.masks, mask])
            if self.mask_names is None:
                self.mask_names = [f"layer_{i}" for i in range(self.masks.shape[1] - 1)]
            self.mask_names.append(name)

    def get_mask_layer(self, name: str) -> np.ndarray:
        """Get a mask layer by name.

        :param name: Name of the mask layer
        :returns: Boolean array of shape (N,)
        :raises ValueError: If layer name not found

        Example:
            >>> opacity_mask = data.get_mask_layer("high_opacity")
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
            >>> data.remove_mask_layer("foreground")
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
            # Multi-layer - remove one column
            mask_list = [self.masks[:, i] for i in range(self.masks.shape[1]) if i != layer_idx]
            if len(mask_list) == 0:
                self.masks = None
                self.mask_names = None
            else:
                if len(mask_list) == 1:
                    self.masks = mask_list[0]
                else:
                    self.masks = np.column_stack(mask_list)
                self.mask_names = [n for n in self.mask_names if n != name]

    def combine_masks(self, mode: str = "and", layers: list[str] | None = None) -> np.ndarray:
        """Combine mask layers using boolean logic.

        :param mode: Combination mode - "and" (all must pass) or "or" (any must pass)
        :param layers: List of layer names to combine (None = use all layers)
        :returns: Combined boolean mask of shape (N,)
        :raises ValueError: If no masks exist or invalid mode

        Example:
            >>> # Combine all layers with AND
            >>> mask = data.combine_masks(mode="and")
            >>> filtered = data[mask]
            >>>
            >>> # Combine specific layers with OR
            >>> mask = data.combine_masks(mode="or", layers=["opacity", "foreground"])
        """
        if self.masks is None:
            raise ValueError("No mask layers exist")

        if mode not in ("and", "or"):
            raise ValueError(f"Mode must be 'and' or 'or', got '{mode}'")

        # Get mask array
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
            masks_to_combine = self.masks[:, indices]

        # Combine using specified mode with adaptive optimization strategy
        # Benchmarks show:
        # - 1 layer: numpy is fastest (no Numba overhead)
        # - 2+ layers: Numba is 37-68x faster than numpy

        if masks_to_combine.ndim == 1:
            # Single layer - return as-is
            return masks_to_combine

        # Multi-layer combination
        n_layers = masks_to_combine.shape[1]

        if n_layers == 1:
            # Technically 2D but only 1 layer - flatten
            return masks_to_combine[:, 0]

        # 2+ layers: Use Numba (37-68x faster!)
        if mode == "and":
            return _combine_masks_numba_and(masks_to_combine)
        # mode == "or"
        return _combine_masks_numba_or(masks_to_combine)

    def apply_masks(
        self, mode: str = "and", layers: list[str] | None = None, inplace: bool = False
    ) -> "GSData":
        """Apply mask layers to filter Gaussians.

        :param mode: Combination mode - "and" or "or"
        :param layers: List of layer names to apply (None = all layers)
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Filtered GSData (self if inplace=True, new object if inplace=False)

        Example:
            >>> # Filter using all mask layers (AND logic)
            >>> filtered = data.apply_masks(mode="and")
            >>>
            >>> # Filter in-place using specific layers (OR logic)
            >>> data.apply_masks(mode="or", layers=["opacity", "scale"], inplace=True)
        """
        combined_mask = self.combine_masks(mode=mode, layers=layers)

        if inplace:
            # Filter arrays in-place (replace with filtered versions)
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
        # Return filtered copy
        return self[combined_mask]

    def consolidate(self) -> "GSData":
        """Consolidate separate arrays into a single base array.

        This creates a _base array from separate arrays, which can improve
        performance for boolean masking operations and file writes.

        Uses JIT-compiled parallel kernels for 2.8-5x faster interleaving
        compared to slice assignment.

        :returns: New GSData with _base array, or self if already consolidated

        Note:
            - One-time cost: ~3ms per 400K Gaussians (JIT-optimized)
            - Benefit: 1.5x faster boolean masking, 36% faster writes
            - No benefit for slicing (actually slightly slower)
            - Use when doing many boolean mask operations or file writes
        """
        if self._base is not None:
            return self  # Already consolidated

        # Create base array with standard layout
        n_gaussians = len(self)

        # Ensure arrays are contiguous float32 for JIT
        means = np.ascontiguousarray(self.means, dtype=np.float32)
        sh0 = np.ascontiguousarray(self.sh0, dtype=np.float32)
        opacities = np.ascontiguousarray(self.opacities.ravel(), dtype=np.float32)
        scales = np.ascontiguousarray(self.scales, dtype=np.float32)
        quats = np.ascontiguousarray(self.quats, dtype=np.float32)

        # Determine property count and use appropriate JIT kernel
        # Layout: means(3) + sh0(3) + shN(K*3) + opacity(1) + scales(3) + quats(4)
        if self.shN is not None and self.shN.shape[1] > 0:
            # SH1-3: use general kernel with variable SH coefficients
            sh_bands = self.shN.shape[1]
            sh_coeffs = sh_bands * 3  # Total coefficients (9, 24, or 45)
            n_props = 14 + sh_coeffs

            # Flatten shN from (N, bands, 3) to (N, bands*3)
            shn_flat = np.ascontiguousarray(
                self.shN.reshape(n_gaussians, sh_coeffs), dtype=np.float32
            )

            # Allocate and populate using JIT kernel
            new_base = np.empty((n_gaussians, n_props), dtype=np.float32)
            _interleave_shn_jit(means, sh0, shn_flat, opacities, scales, quats, new_base, sh_coeffs)
        else:
            # SH0: use optimized kernel (14 properties)
            n_props = 14
            new_base = np.empty((n_gaussians, n_props), dtype=np.float32)
            _interleave_sh0_jit(means, sh0, opacities, scales, quats, new_base)

        # Recreate GSData with new base
        return GSData._recreate_from_base(
            new_base,
            format_flag=self._format,
            masks_array=self.masks.copy() if self.masks is not None else None,
            mask_names=self.mask_names.copy() if self.mask_names is not None else None,
        )

    def copy(self) -> "GSData":
        """Return a deep copy of the GSData.

        Creates independent copies of all arrays, ensuring modifications
        to the copy won't affect the original data.

        :returns: A new GSData object with copied arrays
        """
        # Optimize: If we have _base, copy it and recreate views (2-3x faster)
        if self._base is not None:
            new_base = self._base.copy()
            masks_copy = self.masks.copy() if self.masks is not None else None
            mask_names_copy = self.mask_names.copy() if self.mask_names is not None else None

            result = GSData._recreate_from_base(
                new_base,
                format_flag=self._format,
                masks_array=masks_copy,
                mask_names=mask_names_copy,
            )
            if result is not None:
                return result

        # Fallback: No base array or unknown format, copy individual arrays
        return GSData(
            means=self.means.copy(),
            scales=self.scales.copy(),
            quats=self.quats.copy(),
            opacities=self.opacities.copy(),
            sh0=self.sh0.copy(),
            shN=self.shN.copy() if self.shN is not None else None,
            masks=self.masks.copy() if self.masks is not None else None,
            mask_names=self.mask_names.copy() if self.mask_names is not None else None,
            _base=None,
            _format=self._format,  # Preserve format flag
        )

    def __add__(self, other: "GSData") -> "GSData":
        """Support + operator for concatenation.

        Allows Pythonic concatenation using the + operator.

        :param other: Another GSData object to concatenate
        :returns: New GSData object with combined Gaussians

        Example:
            >>> combined = data1 + data2  # Same as data1.add(data2)
        """
        return self.add(other)

    def __radd__(self, other):
        """Support reverse addition (rarely used but completes the interface)."""
        if other == 0:
            # Allow sum([data1, data2, data3]) to work
            return self
        return self.add(other)

    def add(self, other: "GSData") -> "GSData":
        """Concatenate two GSData objects along the Gaussian dimension.

        Combines two GSData objects by stacking all Gaussians. Validates
        compatibility (same SH degree) and handles mask layer merging.

        Performance: Highly optimized using pre-allocation + direct assignment
        - 1.10x faster for 10K Gaussians (412 M/s)
        - 1.56x faster for 100K Gaussians (106 M/s)
        - 1.90x faster for 500K Gaussians (99 M/s)

        For GPU operations, use GSTensor.add() which is 18x faster on large datasets.

        Note: For concatenating multiple arrays, use GSData.concatenate() which is
        5.74x faster than repeated add() calls due to single allocation.

        :param other: Another GSData object to concatenate
        :returns: New GSData object with combined Gaussians
        :raises ValueError: If SH degrees don't match or formats don't match

        Example:
            >>> data1 = gsply.plyread("scene1.ply")  # 100K Gaussians
            >>> data2 = gsply.plyread("scene2.ply")  # 50K Gaussians
            >>> combined = data1.add(data2)  # 150K Gaussians
            >>> # Or use + operator
            >>> combined = data1 + data2  # Same result
            >>> print(len(combined))  # 150000

        See Also:
            concatenate: Bulk concatenation of multiple arrays (5.74x faster)
        """
        # Validate compatibility
        if self.get_sh_degree() != other.get_sh_degree():
            raise ValueError(
                f"Cannot concatenate GSData with different SH degrees: "
                f"{self.get_sh_degree()} vs {other.get_sh_degree()}"
            )

        # Validate format equivalence
        if self._format != other._format:
            raise ValueError(
                f"Cannot concatenate GSData with different formats. "
                f"self: {self._format}, other: {other._format}. "
                f"Use normalize() or denormalize() to convert formats before concatenating."
            )

        # Fast path: If both have _base with same format, concatenate base arrays
        if (
            self._base is not None
            and other._base is not None
            and self._base.shape[1] == other._base.shape[1]
        ):
            # Optimized: Pre-allocate and use direct assignment
            n1 = len(self)
            n2 = len(other)
            combined_base = np.empty((n1 + n2, self._base.shape[1]), dtype=self._base.dtype)
            combined_base[:n1] = self._base
            combined_base[n1:] = other._base

            # Handle masks
            combined_masks = None
            combined_mask_names = None

            if self.masks is not None or other.masks is not None:
                # Ensure both have same number of mask layers
                self_masks = self.masks if self.masks is not None else None
                other_masks = other.masks if other.masks is not None else None

                if self_masks is not None and other_masks is not None:
                    # Both have masks - concatenate
                    # Ensure 2D
                    if self_masks.ndim == 1:
                        self_masks = self_masks[:, None]
                    if other_masks.ndim == 1:
                        other_masks = other_masks[:, None]

                    # Check layer count compatibility
                    if self_masks.shape[1] == other_masks.shape[1]:
                        combined_masks = np.concatenate([self_masks, other_masks], axis=0)
                        # Merge names (prefer self names, use other as fallback)
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
                        other_masks_filled = np.zeros(len(other), dtype=bool)
                    else:
                        other_masks_filled = np.zeros((len(other), self_masks.shape[1]), dtype=bool)
                    combined_masks = np.concatenate([self_masks, other_masks_filled], axis=0)
                    combined_mask_names = self.mask_names.copy() if self.mask_names else None
                else:  # other_masks is not None
                    # Only other has masks - create False masks for self
                    if other_masks.ndim == 1:
                        self_masks_filled = np.zeros(len(self), dtype=bool)
                    else:
                        self_masks_filled = np.zeros((len(self), other_masks.shape[1]), dtype=bool)
                    combined_masks = np.concatenate([self_masks_filled, other_masks], axis=0)
                    combined_mask_names = other.mask_names.copy() if other.mask_names else None

            # Format already validated above, use self's format
            format_flag = self._format
            return GSData._recreate_from_base(
                combined_base,
                format_flag=format_flag,
                masks_array=combined_masks,
                mask_names=combined_mask_names,
            )

        # Fallback: Concatenate individual arrays
        combined_shN = None  # noqa: N806
        if self.shN is not None or other.shN is not None:
            # Ensure both have shN (use zeros if missing)
            self_shN = (  # noqa: N806
                self.shN if self.shN is not None else np.zeros((len(self), 0, 3), dtype=np.float32)
            )
            other_shN = (  # noqa: N806
                other.shN
                if other.shN is not None
                else np.zeros((len(other), 0, 3), dtype=np.float32)
            )

            if self_shN.shape[1] == other_shN.shape[1]:
                combined_shN = np.concatenate([self_shN, other_shN], axis=0)  # noqa: N806
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
                    self_masks = self_masks[:, None]
                if other_masks.ndim == 1:
                    other_masks = other_masks[:, None]

                if self_masks.shape[1] == other_masks.shape[1]:
                    combined_masks = np.concatenate([self_masks, other_masks], axis=0)
                    if self.mask_names is not None:
                        combined_mask_names = self.mask_names.copy()
                    elif other.mask_names is not None:
                        combined_mask_names = other.mask_names.copy()
            elif self_masks is not None:
                if self_masks.ndim == 1:
                    other_masks_filled = np.zeros(len(other), dtype=bool)
                else:
                    other_masks_filled = np.zeros((len(other), self_masks.shape[1]), dtype=bool)
                combined_masks = np.concatenate([self_masks, other_masks_filled], axis=0)
                combined_mask_names = self.mask_names.copy() if self.mask_names else None
            else:
                if other_masks.ndim == 1:
                    self_masks_filled = np.zeros(len(self), dtype=bool)
                else:
                    self_masks_filled = np.zeros((len(self), other_masks.shape[1]), dtype=bool)
                combined_masks = np.concatenate([self_masks_filled, other_masks], axis=0)
                combined_mask_names = other.mask_names.copy() if other.mask_names else None

        # Optimized path: Pre-allocate and use direct assignment (4.5x faster for small arrays)
        n1 = len(self)
        n2 = len(other)
        total = n1 + n2

        # Pre-allocate output arrays
        means = np.empty((total, 3), dtype=self.means.dtype)
        scales = np.empty((total, 3), dtype=self.scales.dtype)
        quats = np.empty((total, 4), dtype=self.quats.dtype)
        opacities = np.empty(total, dtype=self.opacities.dtype)
        sh0 = np.empty((total, 3), dtype=self.sh0.dtype)

        # Direct assignment (faster than concatenate)
        means[:n1] = self.means
        means[n1:] = other.means
        scales[:n1] = self.scales
        scales[n1:] = other.scales
        quats[:n1] = self.quats
        quats[n1:] = other.quats
        opacities[:n1] = self.opacities
        opacities[n1:] = other.opacities
        sh0[:n1] = self.sh0
        sh0[n1:] = other.sh0

        # Format already validated above, use self's format
        format_flag = self._format

        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=combined_shN,
            masks=combined_masks,
            mask_names=combined_mask_names,
            _base=None,  # Clear _base since we created new arrays
            _format=format_flag,  # Preserve format if both are same
        )

    @staticmethod
    def concatenate(arrays: list["GSData"]) -> "GSData":
        """Bulk concatenate multiple GSData objects.

        Significantly more efficient than repeated add() calls:
        - Single allocation instead of N-1 intermediate allocations
        - 5.74x faster for concatenating 10 arrays
        - Reduces total memory copies

        :param arrays: List of GSData objects to concatenate
        :returns: New GSData object with all Gaussians combined
        :raises ValueError: If list is empty, SH degrees don't match, or formats don't match

        Example:
            >>> scenes = [gsply.plyread(f"scene{i}.ply") for i in range(10)]
            >>> combined = GSData.concatenate(scenes)  # 5.74x faster than loop!

        Performance Comparison (10 arrays of 10K Gaussians):
            >>> # Slow: Pairwise add() - 5.990 ms
            >>> result = scenes[0]
            >>> for scene in scenes[1:]:
            ...     result = result.add(scene)
            >>>
            >>> # Fast: Bulk concatenate - 1.044 ms (5.74x faster!)
            >>> result = GSData.concatenate(scenes)
        """
        if not arrays:
            raise ValueError("Cannot concatenate empty list")
        if len(arrays) == 1:
            return arrays[0]

        # Validate all have same SH degree
        sh_degree = arrays[0].get_sh_degree()
        for arr in arrays[1:]:
            if arr.get_sh_degree() != sh_degree:
                raise ValueError(
                    f"All arrays must have same SH degree, got {sh_degree} and {arr.get_sh_degree()}"
                )

        # Validate all have same format
        format_ref = arrays[0]._format
        for i, arr in enumerate(arrays[1:], start=1):
            if arr._format != format_ref:
                raise ValueError(
                    f"All arrays must have same format. "
                    f"Array 0: {format_ref}, Array {i}: {arr._format}. "
                    f"Use normalize() or denormalize() to convert formats before concatenating."
                )

        # Calculate total size
        total = sum(len(arr) for arr in arrays)

        # Pre-allocate output arrays (single allocation for efficiency)
        means = np.empty((total, 3), dtype=arrays[0].means.dtype)
        scales = np.empty((total, 3), dtype=arrays[0].scales.dtype)
        quats = np.empty((total, 4), dtype=arrays[0].quats.dtype)
        opacities = np.empty(total, dtype=arrays[0].opacities.dtype)
        sh0 = np.empty((total, 3), dtype=arrays[0].sh0.dtype)

        # Handle shN
        combined_shN = None  # noqa: N806
        if any(arr.shN is not None for arr in arrays):
            # Get shN shape from first array that has it
            sh_bands = next(arr.shN.shape[1] for arr in arrays if arr.shN is not None)
            combined_shN = np.empty((total, sh_bands, 3), dtype=arrays[0].sh0.dtype)  # noqa: N806

        # Copy data in one pass
        offset = 0
        for arr in arrays:
            n = len(arr)
            means[offset : offset + n] = arr.means
            scales[offset : offset + n] = arr.scales
            quats[offset : offset + n] = arr.quats
            opacities[offset : offset + n] = arr.opacities
            sh0[offset : offset + n] = arr.sh0

            if combined_shN is not None:
                if arr.shN is not None:
                    combined_shN[offset : offset + n] = arr.shN
                else:
                    # Fill with zeros for arrays without shN
                    combined_shN[offset : offset + n] = 0

            offset += n

        # Format already validated above, use first array's format
        format_flag = arrays[0]._format

        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=combined_shN,
            masks=None,  # Don't concatenate masks for bulk operation
            mask_names=None,
            _base=None,
            _format=format_flag,  # Preserve format if all are same
        )

    def make_contiguous(self, inplace: bool = True) -> "GSData":
        """Convert all arrays to contiguous memory layout for better performance.

        When data is loaded from PLY files via _base arrays, all field arrays
        (means, scales, etc.) are non-contiguous views with poor cache locality,
        causing 1.5-45x performance overhead for operations.

        Conversion Cost (measured):
        - 1K Gaussians:   0.02 ms
        - 10K Gaussians:  0.14 ms
        - 100K Gaussians: 2.2 ms
        - 1M Gaussians:   25 ms

        Per-Operation Speedup (100K Gaussians):
        - argmax():       45.5x faster
        - max/min():      18-19x faster
        - sum/mean():     6-7x faster
        - std():          2.7x faster
        - element-wise:   2-4x faster

        Break-Even Analysis:
        - < 8 operations:    DON'T convert (overhead not justified)
        - >= 8 operations:   CONVERT (speedup outweighs cost)
        - >= 100 operations: CRITICAL (7.9x total speedup)

        Real-World Scenarios (100K Gaussians):
        - Light processing (3 ops):    2.4x slower (DON'T convert)
        - Iterative processing (10x):  2.1x faster (CONVERT!)
        - Heavy computation (100x):    7.9x faster (CONVERT!)

        Memory: Zero overhead (same total memory, just reorganized)

        :param inplace: If True, modify arrays in-place and clear _base (default).
                        If False, return new GSData with contiguous arrays.
        :returns: Self if inplace=True, new GSData if inplace=False

        Example:
            >>> data = gsply.plyread("scene.ply")  # Non-contiguous from _base
            >>>
            >>> # For few operations (< 8) - don't convert
            >>> total = data.means.sum()  # Just use as-is
            >>>
            >>> # For many operations (>= 8) - convert first!
            >>> data.make_contiguous()  # Up to 45x faster per operation
            >>> for i in range(100):
            ...     result = data.means.sum() + data.means.max()  # 7.9x faster!

        See Also:
            is_contiguous: Check if arrays are already contiguous
        """
        # Check if already contiguous
        if self._base is None:
            # No _base means separate arrays, likely already contiguous
            all_contiguous = all(
                arr.flags["C_CONTIGUOUS"]
                for arr in [self.means, self.scales, self.quats, self.opacities, self.sh0]
                if arr is not None
            )
            if all_contiguous and (self.shN is None or self.shN.flags["C_CONTIGUOUS"]):
                return self  # Already contiguous, nothing to do

        # Convert to contiguous arrays
        means = np.ascontiguousarray(self.means)
        scales = np.ascontiguousarray(self.scales)
        quats = np.ascontiguousarray(self.quats)
        opacities = np.ascontiguousarray(self.opacities)
        sh0 = np.ascontiguousarray(self.sh0)
        shN = np.ascontiguousarray(self.shN) if self.shN is not None else None  # noqa: N806
        masks = np.ascontiguousarray(self.masks) if self.masks is not None else None

        if inplace:
            # Modify in-place
            self.means = means
            self.scales = scales
            self.quats = quats
            self.opacities = opacities
            self.sh0 = sh0
            self.shN = shN
            self.masks = masks
            self._base = None  # Clear _base reference
            return self
        # Return new object
        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            masks=masks,
            mask_names=self.mask_names.copy() if self.mask_names else None,
            _base=None,
            _format=self._format,  # Preserve format flag
        )

    def is_contiguous(self) -> bool:
        """Check if all arrays are C-contiguous.

        :returns: True if all arrays are contiguous, False otherwise

        Example:
            >>> data = gsply.plyread("scene.ply")
            >>> print(data.is_contiguous())  # False (from _base)
            >>> data.make_contiguous()
            >>> print(data.is_contiguous())  # True
        """
        arrays_to_check = [self.means, self.scales, self.quats, self.opacities, self.sh0]
        if self.shN is not None:
            arrays_to_check.append(self.shN)
        if self.masks is not None:
            arrays_to_check.append(self.masks)

        return all(arr.flags["C_CONTIGUOUS"] for arr in arrays_to_check)

    def unpack(self, include_shN: bool = True) -> tuple:
        """Unpack Gaussian data into tuple of arrays.

        Convenient for standard Gaussian Splatting workflows that expect
        individual arrays rather than a container object.

        :param include_shN: If True, include shN in output (default True)
        :returns: If include_shN=True: (means, scales, quats, opacities, sh0, shN),
                  If include_shN=False: (means, scales, quats, opacities, sh0)

        Example:
            >>> data = plyread("scene.ply")
            >>> means, scales, quats, opacities, sh0, shN = data.unpack()
            >>> # Use with rendering functions
            >>> render(means, scales, quats, opacities, sh0)
            >>>
            >>> # For SH0 data, exclude shN
            >>> means, scales, quats, opacities, sh0 = data.unpack(include_shN=False)
        """
        if include_shN:
            return (self.means, self.scales, self.quats, self.opacities, self.sh0, self.shN)
        return (self.means, self.scales, self.quats, self.opacities, self.sh0)

    def to_dict(self) -> dict:
        """Convert Gaussian data to dictionary.

        :returns: Dictionary with keys: means, scales, quats, opacities, sh0, shN

        Example:
            >>> data = plyread("scene.ply")
            >>> props = data.to_dict()
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

    def normalize(self, inplace: bool = True) -> "GSData":
        """Convert linear scales/opacities to PLY format (log-scales, logit-opacities).

        Converts:
        - Linear scales → log-scales: log(scale) with clamping
        - Linear opacities → logit-opacities: logit(opacity) with clamping

        This is the standard format used in Gaussian Splatting PLY files.
        Use this when you have linear data and need to save to PLY format.

        :param inplace: If True, modify this object in-place (default). If False, return new object.
        :returns: GSData object (self if inplace=True, new object otherwise)

        Example:
            >>> # Data with linear scales and opacities
            >>> data = GSData(scales=[0.1, 0.2, 0.3], opacities=[0.5, 0.7, 0.9], ...)
            >>> # Convert to PLY format in-place (modifies data)
            >>> data.normalize()  # or: data.normalize(inplace=True)
            >>> # Now ready to save with plywrite()
            >>> plywrite("output.ply", data)
            >>>
            >>> # Or create a copy if you need to keep original
            >>> ply_data = data.normalize(inplace=False)
        """
        from gsply.utils import apply_pre_deactivations

        # Constants for numerical stability (matching GSTensor)
        min_scale = 1e-9
        min_opacity = 1e-4
        max_opacity = 1.0 - 1e-4

        # Use fused deactivation kernel for optimal performance (~8-15x faster)
        result = apply_pre_deactivations(
            self,
            min_scale=min_scale,
            min_opacity=min_opacity,
            max_opacity=max_opacity,
            inplace=inplace,
        )

        # Update format dict: scales and opacities are now in PLY format
        if inplace:
            self._format["scales"] = DataFormat.SCALES_PLY
            self._format["opacities"] = DataFormat.OPACITIES_PLY
            return self

        # For non-inplace, update format dict in returned object
        result._format = {
            **result._format,
            "scales": DataFormat.SCALES_PLY,
            "opacities": DataFormat.OPACITIES_PLY,
            "sh_order": _get_sh_order_format(result.get_sh_degree()),
        }
        return result

    def denormalize(self, inplace: bool = True) -> "GSData":
        """Convert PLY format (log-scales, logit-opacities) to linear format.

        Converts:
        - Log-scales → linear scales: exp(log_scale) with clamping
        - Logit-opacities → linear opacities: sigmoid(logit)
        - Quaternions → normalized quaternions

        Use this when you load PLY files (which use log/logit format) and need
        linear values for computations or visualization.

        :param inplace: If True, modify this object in-place (default). If False, return new object.
        :returns: GSData object (self if inplace=True, new object otherwise)

        Example:
            >>> # Load PLY file (contains log-scales and logit-opacities)
            >>> data = plyread("scene.ply")
            >>> # Convert to linear format in-place (modifies data)
            >>> data.denormalize()  # or: data.denormalize(inplace=True)
            >>> # Now scales and opacities are in linear space [0, 1] for opacities
            >>> print(f"Linear opacity range: [{data.opacities.min():.3f}, {data.opacities.max():.3f}]")
            >>>
            >>> # Or create a copy if you need to keep PLY format
            >>> linear_data = data.denormalize(inplace=False)
        """
        from gsply.utils import apply_pre_activations

        # Use fused activation kernel for optimal performance (~8-15x faster)
        result = apply_pre_activations(self, inplace=inplace)

        # Update format dict: scales and opacities are now in linear format
        if inplace:
            self._format["scales"] = DataFormat.SCALES_LINEAR
            self._format["opacities"] = DataFormat.OPACITIES_LINEAR
            return self

        # For non-inplace, update format dict in returned object
        result._format = {
            **result._format,
            "scales": DataFormat.SCALES_LINEAR,
            "opacities": DataFormat.OPACITIES_LINEAR,
            "sh_order": _get_sh_order_format(result.get_sh_degree()),
        }
        return result

    def to_rgb(self, inplace: bool = True) -> "GSData":
        """Convert sh0 from spherical harmonics (SH) format to RGB color format.

        Converts SH DC coefficients to RGB colors in [0, 1] range.
        Formula: rgb = sh0 * SH_C0 + 0.5

        :param inplace: If True, modify this object in-place (default). If False, return new object.
        :returns: GSData object (self if inplace=True, new object otherwise)

        Example:
            >>> # Load PLY file (sh0 is in SH format)
            >>> data = gsply.plyread("scene.ply")
            >>> # Convert to RGB format in-place
            >>> data.to_rgb()  # or: data.to_rgb(inplace=True)
            >>> # Now sh0 contains RGB colors [0, 1]
            >>> print(f"RGB color range: [{data.sh0.min():.3f}, {data.sh0.max():.3f}]")
            >>>
            >>> # Or create a copy if you need to keep SH format
            >>> rgb_data = data.to_rgb(inplace=False)
        """
        from gsply.formats import SH_C0
        from gsply.utils import _sh2rgb_inplace_jit

        if inplace:
            # True in-place: modify self.sh0 directly using Numba JIT
            _sh2rgb_inplace_jit(self.sh0, SH_C0)
            self._base = None  # Invalidate _base since we modified arrays
            # Update format dict: sh0 is now in RGB format
            self._format["sh0"] = DataFormat.SH0_RGB
            return self

        # Create copy for non-inplace operation
        rgb = self.sh0 * SH_C0 + 0.5
        return GSData(
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

    def to_sh(self, inplace: bool = True) -> "GSData":
        """Convert sh0 from RGB color format to spherical harmonics (SH) format.

        Converts RGB colors in [0, 1] range to SH DC coefficients.
        Formula: sh0 = (rgb - 0.5) / SH_C0

        :param inplace: If True, modify this object in-place (default). If False, return new object.
        :returns: GSData object (self if inplace=True, new object otherwise)

        Example:
            >>> # Create GSData with RGB colors
            >>> rgb_colors = np.random.rand(1000, 3).astype(np.float32)
            >>> data = GSData(means=..., scales=..., sh0=rgb_colors, ...)
            >>> # Convert to SH format in-place
            >>> data.to_sh()  # or: data.to_sh(inplace=True)
            >>> # Now sh0 contains SH DC coefficients
            >>>
            >>> # Or create a copy if you need to keep RGB format
            >>> sh_data = data.to_sh(inplace=False)
        """
        from gsply.formats import SH_C0
        from gsply.utils import _rgb2sh_inplace_jit

        if inplace:
            # True in-place: modify self.sh0 directly using Numba JIT
            _rgb2sh_inplace_jit(self.sh0, 1.0 / SH_C0)
            self._base = None  # Invalidate _base since we modified arrays
            # Update format dict: sh0 is now in SH format
            self._format["sh0"] = DataFormat.SH0_SH
            return self

        # Create copy for non-inplace operation
        sh = (self.sh0 - 0.5) / SH_C0
        return GSData(
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

    def copy_slice(self, key) -> "GSData":
        """Efficiently slice and copy in one operation.

        For slices that return views, this is more efficient than data[key].copy()
        as it avoids creating intermediate view objects.

        For boolean masks and fancy indexing, this simply delegates to __getitem__
        since those already return copies.

        :param key: Slice key (slice, int, array, or boolean mask)
        :returns: A new GSData object with copied sliced data

        Examples:
            data.copy_slice(100:200)    # Copy of elements 100-199 (avoids view)
            data.copy_slice(::10)        # Copy of every 10th element (avoids view)
            data.copy_slice(mask)        # Same as data[mask] (already a copy)
        """
        # For boolean masking and fancy indexing, __getitem__ already returns copies
        # So just delegate to it - no need to do redundant work
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                # Boolean mask - __getitem__ uses np.compress which returns copy
                return self[key]
            # Fancy indexing - __getitem__ already returns copy
            return self[key]
        if isinstance(key, list):
            # List indexing - __getitem__ already returns copy
            return self[key]

        # For single index, create single-element GSData copy
        if isinstance(key, int):
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} out of range for {len(self)} Gaussians")

            # Create single-element copies
            return GSData(
                means=self.means[key : key + 1].copy(),
                scales=self.scales[key : key + 1].copy(),
                quats=self.quats[key : key + 1].copy(),
                opacities=self.opacities[key : key + 1].copy(),
                sh0=self.sh0[key : key + 1].copy(),
                shN=self.shN[key : key + 1].copy() if self.shN is not None else None,
                masks=self.masks[key : key + 1].copy() if self.masks is not None else None,
                mask_names=self.mask_names.copy() if self.mask_names is not None else None,
                _base=None,
                _format=self._format,  # Preserve format flag
            )

        # For slicing, optimize using base array when available
        if isinstance(key, slice):
            # Optimize: Use base array copy if available (2-3x faster)
            if self._base is not None:
                base_copy = self._base[key].copy()
                masks_copy = self.masks[key].copy() if self.masks is not None else None
                mask_names_copy = self.mask_names.copy() if self.mask_names is not None else None

                result = GSData._recreate_from_base(
                    base_copy,
                    format_flag=self._format,
                    masks_array=masks_copy,
                    mask_names=mask_names_copy,
                )
                if result is not None:
                    return result

            # Fallback: Copy individual arrays
            return GSData(
                means=self.means[key].copy(),
                scales=self.scales[key].copy(),
                quats=self.quats[key].copy(),
                opacities=self.opacities[key].copy(),
                sh0=self.sh0[key].copy(),
                shN=self.shN[key].copy() if self.shN is not None else None,
                masks=self.masks[key].copy() if self.masks is not None else None,
                mask_names=self.mask_names.copy() if self.mask_names is not None else None,
                _base=None,
                _format=self._format,  # Preserve format flag
            )

        raise TypeError(f"Invalid index type: {type(key)}")

    def __iter__(self):
        """Iterate over Gaussians, yielding tuples."""
        for i in range(len(self)):
            yield self[i]

    def get_gaussian(self, index: int) -> "GSData":
        """Get a single Gaussian as a GSData object.

        Unlike direct indexing which returns a tuple for efficiency,
        this method returns a GSData object containing a single Gaussian.

        :param index: Index of the Gaussian to retrieve
        :returns: GSData object with a single Gaussian
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for {len(self)} Gaussians")

        # Use slice to get GSData with single element
        return self[index : index + 1]

    @staticmethod
    def _recreate_from_base(
        base_array,
        format_flag: FormatDict,
        masks_array=None,
        mask_names=None,
    ) -> "GSData | None":
        """Helper method to recreate GSData from a base array.

        This centralizes the view recreation logic that was duplicated
        across multiple methods.

        :param base_array: The base array to create views from
        :param format_flag: Format dict (required)
        :param masks_array: Optional masks array
        :param mask_names: Optional list of mask layer names
        :returns: New GSData object with views into base_array, or None if unknown format
        """
        n_gaussians = base_array.shape[0]
        n_props = base_array.shape[1]

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

        # Create views into the base array
        means = base_array[:, 0:3]
        sh0 = base_array[:, 3:6]

        if sh_coeffs > 0:
            shN_flat = base_array[:, 6 : 6 + sh_coeffs * 3]  # noqa: N806
            # PLY stores SH coefficients channel-grouped: [R0..Rk, G0..Gk, B0..Bk]
            # Reshape to [N, 3, K] then transpose to [N, K, 3] for gsplat convention
            shN = shN_flat.reshape(n_gaussians, 3, sh_coeffs).transpose(0, 2, 1)  # noqa: N806
            opacity_idx = 6 + sh_coeffs * 3
        else:
            shN = None  # noqa: N806
            opacity_idx = 6

        opacities = base_array[:, opacity_idx]
        scales = base_array[:, opacity_idx + 1 : opacity_idx + 4]
        quats = base_array[:, opacity_idx + 4 : opacity_idx + 8]

        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            masks=masks_array,
            mask_names=mask_names,
            _base=base_array,
            _format=format_flag,  # Format dict (always provided)
        )

    def _slice_from_base(self, indices_or_mask):
        """Efficiently slice data when _base array exists.

        This method slices the base array once and recreates views,
        which is much faster than slicing individual arrays.
        """
        if self._base is None:
            return None

        # Slice the base array
        if isinstance(indices_or_mask, np.ndarray) and indices_or_mask.dtype == bool:
            # Boolean mask - use compress for efficiency
            base_subset = np.compress(indices_or_mask, self._base, axis=0)
        elif isinstance(indices_or_mask, slice):
            # Direct slice - most efficient
            base_subset = self._base[indices_or_mask]
        else:
            # Integer indices or array
            base_subset = self._base[indices_or_mask]

        # Handle masks if present
        if self.masks is not None:
            if isinstance(indices_or_mask, np.ndarray) and indices_or_mask.dtype == bool:
                masks_subset = np.compress(indices_or_mask, self.masks, axis=0)
            else:
                masks_subset = self.masks[indices_or_mask]
        else:
            masks_subset = None

        # Preserve mask_names when slicing (layer structure stays same, just fewer Gaussians)
        mask_names_copy = self.mask_names.copy() if self.mask_names is not None else None

        # Use helper to recreate views from sliced base
        return GSData._recreate_from_base(
            base_subset,
            format_flag=self._format,
            masks_array=masks_subset,
            mask_names=mask_names_copy,
        )

    def __getitem__(self, key):
        """Support efficient slicing of Gaussians.

        Different return types for optimal performance:
        - Single index: Returns tuple of values for that Gaussian
        - Slice/mask: Returns new GSData object with sliced data

        When _base array exists, slices it directly for maximum performance
        (up to 25x faster for boolean masks).

        IMPORTANT: Following NumPy conventions:
        - Continuous/step slicing returns VIEWS (shares memory with original)
        - Boolean/fancy indexing returns COPIES (independent data)
        - Use .copy() method if you need an independent copy

        Examples:
            data[0]         # Single Gaussian (returns tuple)
            data[10:20]     # Gaussians 10-19 (returns GSData VIEW)
            data[::10]      # Every 10th Gaussian (returns GSData VIEW)
            data[-100:]     # Last 100 Gaussians (returns GSData VIEW)
            data[:1000]     # First 1000 Gaussians (returns GSData VIEW)
            data[mask]      # Boolean mask selection (returns GSData COPY)
            data[[0,1,2]]   # Fancy indexing (returns GSData COPY)
            data[10:20].copy()  # Explicit copy of slice
        """
        # Handle single index - return tuple for efficiency
        if isinstance(key, int):
            # Convert negative index
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} out of range for {len(self)} Gaussians")

            # Return tuple of values for single Gaussian
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
            # Get the actual indices
            start, stop, step = key.indices(len(self))

            # Try fast path with _base array first (for all slicing)
            if self._base is not None:
                result = self._slice_from_base(key)
                if result is not None:
                    return result

            # Fallback: Slice individual arrays (no _base or unknown format)
            return GSData(
                means=self.means[key],
                scales=self.scales[key],
                quats=self.quats[key],
                opacities=self.opacities[key],
                sh0=self.sh0[key],
                shN=self.shN[key] if self.shN is not None else None,
                masks=self.masks[key] if self.masks is not None else None,
                mask_names=self.mask_names.copy() if self.mask_names is not None else None,
                _base=None,
                _format=self._format,  # Preserve format flag
            )

        # Handle boolean array masking
        if isinstance(key, np.ndarray) and key.dtype == bool:
            if len(key) != len(self):
                raise ValueError(
                    f"Boolean mask length {len(key)} doesn't match data length {len(self)}"
                )

            # Try fast path with _base array first
            result = self._slice_from_base(key)
            if result is not None:
                return result

            # Fallback: Use np.compress for better performance with boolean masks
            return GSData(
                means=np.compress(key, self.means, axis=0),
                scales=np.compress(key, self.scales, axis=0),
                quats=np.compress(key, self.quats, axis=0),
                opacities=np.compress(key, self.opacities, axis=0),
                sh0=np.compress(key, self.sh0, axis=0),
                shN=np.compress(key, self.shN, axis=0) if self.shN is not None else None,
                masks=np.compress(key, self.masks, axis=0) if self.masks is not None else None,
                mask_names=self.mask_names.copy() if self.mask_names is not None else None,
                _base=None,
                _format=self._format,  # Preserve format flag
            )

        # Handle integer array indexing
        if isinstance(key, (np.ndarray, list)):
            indices = np.asarray(key, dtype=np.intp)
            # Check bounds
            if np.any(indices < -len(self)) or np.any(indices >= len(self)):
                raise IndexError("Index out of bounds")

            # Convert negative indices
            indices = np.where(indices < 0, indices + len(self), indices)

            # Try fast path with _base array first
            result = self._slice_from_base(indices)
            if result is not None:
                return result

            # Fallback to individual array indexing
            return GSData(
                means=self.means[indices],
                scales=self.scales[indices],
                quats=self.quats[indices],
                opacities=self.opacities[indices],
                sh0=self.sh0[indices],
                shN=self.shN[indices] if self.shN is not None else None,
                masks=self.masks[indices] if self.masks is not None else None,
                mask_names=self.mask_names.copy() if self.mask_names is not None else None,
                _base=None,
                _format=self._format,  # Preserve format flag
            )

        raise TypeError(f"Invalid index type: {type(key)}")

    # ==========================================================================
    # File I/O Methods
    # ==========================================================================

    def save(self, file_path: str | Path, compressed: bool = False) -> None:
        """Save GSData to PLY file.

        Convenience method that wraps plywrite() for object-oriented API.

        :param file_path: Output PLY file path
        :param compressed: If True, write compressed format (default False)

        Example:
            >>> data = gsply.plyread("input.ply")
            >>> data.save("output.ply")  # Uncompressed
            >>> data.save("output.ply", compressed=True)  # Compressed
        """
        from gsply.writer import plywrite

        plywrite(file_path, self, compressed=compressed)

    @classmethod
    def load(cls, file_path: str | Path) -> "GSData":
        """Load GSData from PLY file.

        Convenience classmethod that wraps plyread() for object-oriented API.
        Auto-detects compressed and uncompressed formats.

        :param file_path: Path to PLY file
        :returns: GSData container with loaded data

        Example:
            >>> data = GSData.load("scene.ply")  # Auto-detect format
            >>> print(f"Loaded {len(data)} Gaussians")
        """
        from gsply.reader import plyread

        return plyread(file_path)

    @classmethod
    def from_arrays(
        cls,
        means: np.ndarray,
        scales: np.ndarray,
        quats: np.ndarray,
        opacities: np.ndarray,
        sh0: np.ndarray,
        shN: np.ndarray | None = None,
        format: str = "auto",
        sh_degree: int | None = None,
        sh0_format: DataFormat = DataFormat.SH0_SH,
    ) -> "GSData":
        """Create GSData from individual arrays with format preset.

        Convenient factory method for creating GSData from external arrays
        with automatic format detection or explicit format presets.

        :param means: (N, 3) array - Gaussian centers
        :param scales: (N, 3) array - Scale parameters
        :param quats: (N, 4) array - Rotation quaternions
        :param opacities: (N,) array - Opacity values
        :param sh0: (N, 3) array - DC spherical harmonics
        :param shN: (N, K, 3) array or None - Higher-order SH coefficients
        :param format: Format preset - "auto" (detect), "ply" (log/logit), "linear" or "rasterizer" (linear)
        :param sh_degree: SH degree (0-3) - auto-detected from shN if None
        :param sh0_format: Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH
        :returns: GSData object with specified format

        Example:
            >>> # Auto-detect format from values
            >>> data = GSData.from_arrays(means, scales, quats, opacities, sh0)
            >>>
            >>> # Explicit PLY format (log-scales, logit-opacities)
            >>> data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="ply")
            >>>
            >>> # Explicit linear format (for rasterizer)
            >>> data = GSData.from_arrays(means, scales, quats, opacities, sh0, format="linear")
        """
        # Determine SH degree
        if sh_degree is None:
            if shN is not None and shN.shape[1] > 0:
                sh_bands = shN.shape[1]
                sh_degree = SH_BANDS_TO_DEGREE.get(sh_bands, 0)
            else:
                sh_degree = 0

        # Create format dict based on preset
        if format == "auto":
            # Auto-detect format from values
            scales_format, opacities_format = _detect_format_from_values(scales, opacities)
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
    ) -> "GSData":
        """Create GSData from dictionary with format preset.

        Convenient factory method for creating GSData from a dictionary
        with automatic format detection or explicit format presets.

        :param data_dict: Dictionary with keys: means, scales, quats, opacities, sh0, shN (optional)
        :param format: Format preset - "auto" (detect), "ply" (log/logit), "linear" or "rasterizer" (linear)
        :param sh_degree: SH degree (0-3) - auto-detected from shN if None
        :param sh0_format: Format for sh0 (SH0_SH or SH0_RGB), default SH0_SH
        :returns: GSData object with specified format

        Example:
            >>> # From dictionary with auto-detection
            >>> data = GSData.from_dict({
            ...     "means": means, "scales": scales, "quats": quats,
            ...     "opacities": opacities, "sh0": sh0, "shN": shN
            ... })
            >>>
            >>> # Explicit PLY format
            >>> data = GSData.from_dict(data_dict, format="ply")
            >>>
            >>> # Explicit linear format
            >>> data = GSData.from_dict(data_dict, format="linear")
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
        )
