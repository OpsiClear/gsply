"""Gaussian Splatting data container."""

from dataclasses import dataclass

import numpy as np


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
        masks: (N,) - Boolean mask for filtering/selecting Gaussians (default all True)
        _base: (N, P) - Private base array (keeps memory alive for views, None otherwise)

    Performance:
        - Zero-copy reads provide maximum performance
        - No memory overhead (views share memory with base)

    Example:
        >>> data = plyread("scene.ply")
        >>> print(f"Loaded {len(data)} Gaussians")  # len() returns number of Gaussians
        >>> # Access via attributes
        >>> positions = data.means
        >>> colors = data.sh0
        >>> masks = data.masks  # Boolean mask for filtering
    """

    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh0: np.ndarray
    shN: np.ndarray  # noqa: N815
    masks: np.ndarray | None = None  # Boolean mask for filtering Gaussians
    _base: np.ndarray | None = None  # Private field for zero-copy views

    def __len__(self) -> int:
        """Return the number of Gaussians."""
        return self.means.shape[0]

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
            new_base[:, 6:6 + sh_coeffs * 3] = self.shN.reshape(n_gaussians, sh_coeffs * 3)
            opacity_idx = 6 + sh_coeffs * 3
        else:
            opacity_idx = 6

        new_base[:, opacity_idx] = self.opacities
        new_base[:, opacity_idx + 1:opacity_idx + 4] = self.scales
        new_base[:, opacity_idx + 4:opacity_idx + 8] = self.quats

        # Recreate GSData with new base
        return GSData._recreate_from_base(
            new_base,
            masks_array=self.masks.copy() if self.masks is not None else None
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

            result = GSData._recreate_from_base(new_base, masks_copy)
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
            _base=None
        )

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
        else:
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
            'means': self.means,
            'scales': self.scales,
            'quats': self.quats,
            'opacities': self.opacities,
            'sh0': self.sh0,
            'shN': self.shN
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
            else:
                # Fancy indexing - __getitem__ already returns copy
                return self[key]
        elif isinstance(key, list):
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
                means=self.means[key:key + 1].copy(),
                scales=self.scales[key:key + 1].copy(),
                quats=self.quats[key:key + 1].copy(),
                opacities=self.opacities[key:key + 1].copy(),
                sh0=self.sh0[key:key + 1].copy(),
                shN=self.shN[key:key + 1].copy() if self.shN is not None else None,
                masks=self.masks[key:key + 1].copy() if self.masks is not None else None,
                _base=None
            )

        # For slicing, optimize using base array when available
        if isinstance(key, slice):
            # Optimize: Use base array copy if available (2-3x faster)
            if self._base is not None:
                base_copy = self._base[key].copy()
                masks_copy = self.masks[key].copy() if self.masks is not None else None

                result = GSData._recreate_from_base(base_copy, masks_copy)
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
                _base=None
            )

        else:
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
        return self[index:index + 1]

    @staticmethod
    def _recreate_from_base(base_array, masks_array=None) -> "GSData":
        """Helper method to recreate GSData from a base array.

        This centralizes the view recreation logic that was duplicated
        across multiple methods.

        Args:
            base_array: The base array to create views from
            masks_array: Optional masks array

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
            shN_flat = base_array[:, 6:6 + sh_coeffs * 3]  # noqa: N806
            shN = shN_flat.reshape(n_gaussians, sh_coeffs, 3)  # noqa: N806
            opacity_idx = 6 + sh_coeffs * 3
        else:
            shN = None  # noqa: N806
            opacity_idx = 6

        opacities = base_array[:, opacity_idx]
        scales = base_array[:, opacity_idx + 1:opacity_idx + 4]
        quats = base_array[:, opacity_idx + 4:opacity_idx + 8]

        return GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            masks=masks_array,
            _base=base_array
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

        # Use helper to recreate views from sliced base
        return GSData._recreate_from_base(base_subset, masks_subset)

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
                self.masks[key] if self.masks is not None else None
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
                _base=None
            )

        # Handle boolean array masking
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            if len(key) != len(self):
                raise ValueError(f"Boolean mask length {len(key)} doesn't match data length {len(self)}")

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
                _base=None
            )

        # Handle integer array indexing
        elif isinstance(key, (np.ndarray, list)):
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
                _base=None
            )

        else:
            raise TypeError(f"Invalid index type: {type(key)}")