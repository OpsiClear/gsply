"""Gaussian Splatting data container."""

from dataclasses import dataclass

import numba
import numpy as np


# Numba-optimized mask combination (37-68x faster than numpy.all())
@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _combine_masks_numba_and(masks):
    """Combine masks with AND logic using parallel Numba.

    Benchmarks (100K Gaussians, 5 layers):
    - numpy.all(): 1.43ms (72M/sec)
    - numba parallel: 0.039ms (2,550M/sec) - 37x faster!

    Args:
        masks: Boolean array of shape (N, L) where L >= 2

    Returns:
        Boolean array of shape (N,) - result of AND across layers
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

    Args:
        masks: Boolean array of shape (N, L) where L >= 2

    Returns:
        Boolean array of shape (N,) - result of OR across layers
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
        quats: (N, 4) - rotation quaternions
        opacities: (N,) - opacity values
        sh0: (N, 3) - DC spherical harmonics
        shN: (N, K, 3) - Higher-order SH coefficients (K bands)
        masks: (N,) or (N, L) - Boolean mask layers for filtering (None = no masks)
        mask_names: list[str] - Names for each mask layer (None = unnamed layers)
        _base: (N, P) - Private base array (keeps memory alive for views, None otherwise)

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
    masks: np.ndarray | None = None  # Boolean mask layers (N,) or (N, L)
    mask_names: list[str] | None = None  # Names for each mask layer
    _base: np.ndarray | None = None  # Private field for zero-copy views

    def __len__(self) -> int:
        """Return the number of Gaussians."""
        return self.means.shape[0]

    def get_sh_degree(self) -> int:
        """Get SH degree from shN shape.

        Returns:
            SH degree (0-3)
        """
        if self.shN is None or self.shN.shape[1] == 0:
            return 0
        # shN.shape[1] is number of bands (K)
        sh_bands = self.shN.shape[1]
        if sh_bands == 3:  # SH1: 3 bands
            return 1
        if sh_bands == 8:  # SH2: 8 bands
            return 2
        if sh_bands == 15:  # SH3: 15 bands
            return 3
        return 0

    def add_mask_layer(self, name: str, mask: np.ndarray) -> None:
        """Add a named boolean mask layer.

        Args:
            name: Name for this mask layer
            mask: Boolean array of shape (N,) where N is number of Gaussians

        Raises:
            ValueError: If mask shape doesn't match data length or name already exists

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

        Args:
            name: Name of the mask layer

        Returns:
            Boolean array of shape (N,)

        Raises:
            ValueError: If layer name not found

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

        Args:
            name: Name of the mask layer to remove

        Raises:
            ValueError: If layer name not found

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
            elif len(mask_list) == 1:
                self.masks = mask_list[0]
                self.mask_names = [n for n in self.mask_names if n != name]
            else:
                self.masks = np.column_stack(mask_list)
                self.mask_names = [n for n in self.mask_names if n != name]

    def combine_masks(self, mode: str = "and", layers: list[str] | None = None) -> np.ndarray:
        """Combine mask layers using boolean logic.

        Args:
            mode: Combination mode - "and" (all must pass) or "or" (any must pass)
            layers: List of layer names to combine (None = use all layers)

        Returns:
            Combined boolean mask of shape (N,)

        Raises:
            ValueError: If no masks exist or invalid mode

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

        Args:
            mode: Combination mode - "and" or "or"
            layers: List of layer names to apply (None = all layers)
            inplace: If True, modify self; if False, return filtered copy

        Returns:
            Filtered GSData (self if inplace=True, new object if inplace=False)

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
        performance for boolean masking operations. Only beneficial if you
        plan to perform many boolean mask operations.

        Returns:
            New GSData with _base array, or self if already consolidated

        Note:
            - One-time cost: ~2ms per 100K Gaussians
            - Benefit: 1.5x faster boolean masking
            - No benefit for slicing (actually slightly slower)
            - Use when doing many boolean mask operations
        """
        if self._base is not None:
            return self  # Already consolidated

        # Create base array with standard layout
        n_gaussians = len(self)

        # Determine property count based on SH degree
        # Layout: means(3) + sh0(3) + shN(K*3) + opacity(1) + scales(3) + quats(4)
        # Total: 14 + K*3 where K=0/9/24/45
        if self.shN is not None and self.shN.shape[1] > 0:
            sh_coeffs = self.shN.shape[1]
            n_props = 14 + sh_coeffs * 3  # SH1: 41, SH2: 86, SH3: 149
        else:
            n_props = 14  # SH0

        # Create and populate base array
        new_base = np.empty((n_gaussians, n_props), dtype=np.float32)
        new_base[:, 0:3] = self.means
        new_base[:, 3:6] = self.sh0

        if self.shN is not None and self.shN.shape[1] > 0:
            sh_coeffs = self.shN.shape[1]
            new_base[:, 6 : 6 + sh_coeffs * 3] = self.shN.reshape(n_gaussians, sh_coeffs * 3)
            opacity_idx = 6 + sh_coeffs * 3
        else:
            opacity_idx = 6

        new_base[:, opacity_idx] = self.opacities
        new_base[:, opacity_idx + 1 : opacity_idx + 4] = self.scales
        new_base[:, opacity_idx + 4 : opacity_idx + 8] = self.quats

        # Recreate GSData with new base
        return GSData._recreate_from_base(
            new_base,
            masks_array=self.masks.copy() if self.masks is not None else None,
            mask_names=self.mask_names.copy() if self.mask_names is not None else None,
        )

    def copy(self) -> "GSData":
        """Return a deep copy of the GSData.

        Creates independent copies of all arrays, ensuring modifications
        to the copy won't affect the original data.

        Returns:
            GSData: A new GSData object with copied arrays
        """
        # Optimize: If we have _base, copy it and recreate views (2-3x faster)
        if self._base is not None:
            new_base = self._base.copy()
            masks_copy = self.masks.copy() if self.masks is not None else None
            mask_names_copy = self.mask_names.copy() if self.mask_names is not None else None

            result = GSData._recreate_from_base(new_base, masks_copy, mask_names_copy)
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
        )

    def __add__(self, other: "GSData") -> "GSData":
        """Support + operator for concatenation.

        Allows Pythonic concatenation using the + operator.

        Args:
            other: Another GSData object to concatenate

        Returns:
            New GSData object with combined Gaussians

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

        Args:
            other: Another GSData object to concatenate

        Returns:
            New GSData object with combined Gaussians

        Raises:
            ValueError: If SH degrees don't match

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
                        if (
                            self.mask_names is not None
                            and other.mask_names is not None
                            or self.mask_names is not None
                        ):
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

            return GSData._recreate_from_base(combined_base, combined_masks, combined_mask_names)

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
                    if (
                        self.mask_names is not None
                        and other.mask_names is not None
                        or self.mask_names is not None
                    ):
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
        )

    @staticmethod
    def concatenate(arrays: list["GSData"]) -> "GSData":
        """Bulk concatenate multiple GSData objects.

        Significantly more efficient than repeated add() calls:
        - Single allocation instead of N-1 intermediate allocations
        - 5.74x faster for concatenating 10 arrays
        - Reduces total memory copies

        Args:
            arrays: List of GSData objects to concatenate

        Returns:
            New GSData object with all Gaussians combined

        Raises:
            ValueError: If list is empty or SH degrees don't match

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

        Args:
            inplace: If True, modify arrays in-place and clear _base (default).
                     If False, return new GSData with contiguous arrays.

        Returns:
            Self if inplace=True, new GSData if inplace=False

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
        )

    def is_contiguous(self) -> bool:
        """Check if all arrays are C-contiguous.

        Returns:
            True if all arrays are contiguous, False otherwise

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

        Args:
            include_shN: If True, include shN in output (default True)

        Returns:
            If include_shN=True: (means, scales, quats, opacities, sh0, shN)
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

        Returns:
            Dictionary with keys: means, scales, quats, opacities, sh0, shN

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

    def copy_slice(self, key) -> "GSData":
        """Efficiently slice and copy in one operation.

        For slices that return views, this is more efficient than data[key].copy()
        as it avoids creating intermediate view objects.

        For boolean masks and fancy indexing, this simply delegates to __getitem__
        since those already return copies.

        Args:
            key: Slice key (slice, int, array, or boolean mask)

        Returns:
            GSData: A new GSData object with copied sliced data

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
            )

        # For slicing, optimize using base array when available
        if isinstance(key, slice):
            # Optimize: Use base array copy if available (2-3x faster)
            if self._base is not None:
                base_copy = self._base[key].copy()
                masks_copy = self.masks[key].copy() if self.masks is not None else None
                mask_names_copy = self.mask_names.copy() if self.mask_names is not None else None

                result = GSData._recreate_from_base(base_copy, masks_copy, mask_names_copy)
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

        Args:
            index: Index of the Gaussian to retrieve

        Returns:
            GSData object with a single Gaussian
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for {len(self)} Gaussians")

        # Use slice to get GSData with single element
        return self[index : index + 1]

    @staticmethod
    def _recreate_from_base(base_array, masks_array=None, mask_names=None) -> "GSData":
        """Helper method to recreate GSData from a base array.

        This centralizes the view recreation logic that was duplicated
        across multiple methods.

        Args:
            base_array: The base array to create views from
            masks_array: Optional masks array
            mask_names: Optional list of mask layer names

        Returns:
            New GSData object with views into base_array, or None if unknown format
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
            shN = shN_flat.reshape(n_gaussians, sh_coeffs, 3)  # noqa: N806
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
        return GSData._recreate_from_base(base_subset, masks_subset, mask_names_copy)

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
            )

        raise TypeError(f"Invalid index type: {type(key)}")
