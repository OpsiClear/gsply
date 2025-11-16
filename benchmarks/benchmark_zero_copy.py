"""Explore zero-copy concatenation strategies for CPU.

Investigates:
1. Lazy concatenation view (zero-copy reads, copy-on-write)
2. Pre-allocated buffer pool (amortized allocation cost)
3. NumPy view tricks (np.lib.stride_tricks)
4. When zero-copy is actually possible
5. Fundamental limitations of NumPy memory model
"""

import time
from dataclasses import dataclass

import numpy as np

from gsply import GSData


# Strategy 1: Lazy Concatenation View
class LazyGSData:
    """Lazy concatenation view that defers actual copying.

    Provides zero-copy read access to concatenated data by maintaining
    references to source arrays and dispatching access.

    Trade-offs:
    - Zero allocation/copy overhead
    - Slower element access due to indirection
    - Can't pass to NumPy functions expecting contiguous arrays
    - Most operations trigger materialization
    """

    def __init__(self, arrays: list[GSData]):
        self.arrays = arrays
        self.lengths = [len(a) for a in arrays]
        self.cumsum = np.cumsum([0] + self.lengths)
        self.total_len = sum(self.lengths)

    def __len__(self):
        return self.total_len

    def _find_array(self, idx: int) -> tuple[int, int]:
        """Find which array contains index and local offset."""
        if idx < 0:
            idx = self.total_len + idx
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range for length {self.total_len}")

        # Binary search for array
        array_idx = np.searchsorted(self.cumsum[1:], idx, side="right")
        local_idx = idx - self.cumsum[array_idx]
        return array_idx, local_idx

    def get_means(self, idx: int) -> np.ndarray:
        """Get means for a single Gaussian (zero-copy)."""
        array_idx, local_idx = self._find_array(idx)
        return self.arrays[array_idx].means[local_idx]

    def materialize(self) -> GSData:
        """Materialize lazy view into actual GSData (triggers copy)."""
        return GSData.concatenate(self.arrays)

    @property
    def means(self) -> np.ndarray:
        """Access means array (triggers materialization)."""
        return self.materialize().means


# Strategy 2: Pre-allocated Buffer Pool
class BufferPool:
    """Maintain pool of pre-allocated buffers to amortize allocation cost.

    Trade-offs:
    - Reduces allocation overhead (not zero-copy, but faster)
    - Memory overhead from pre-allocated buffers
    - Only helps if you reuse buffers
    """

    def __init__(self):
        self.buffers = {}  # size -> list of buffers

    def get_buffer(self, shape, dtype):
        """Get buffer from pool or allocate new one."""
        key = (shape, dtype)
        if key in self.buffers and self.buffers[key]:
            return self.buffers[key].pop()
        return np.empty(shape, dtype=dtype)

    def return_buffer(self, buf):
        """Return buffer to pool for reuse."""
        key = (buf.shape, buf.dtype)
        if key not in self.buffers:
            self.buffers[key] = []
        self.buffers[key].append(buf)


def concatenate_with_pool(arrays: list[GSData], pool: BufferPool) -> GSData:
    """Concatenate using pre-allocated buffer pool."""
    total = sum(len(arr) for arr in arrays)

    # Get buffers from pool (or allocate if not available)
    means = pool.get_buffer((total, 3), np.float32)
    scales = pool.get_buffer((total, 3), np.float32)
    quats = pool.get_buffer((total, 4), np.float32)
    opacities = pool.get_buffer(total, np.float32)
    sh0 = pool.get_buffer((total, 3), np.float32)

    # Copy data
    offset = 0
    for arr in arrays:
        n = len(arr)
        means[offset : offset + n] = arr.means
        scales[offset : offset + n] = arr.scales
        quats[offset : offset + n] = arr.quats
        opacities[offset : offset + n] = arr.opacities
        sh0[offset : offset + n] = arr.sh0
        offset += n

    return GSData(means=means, scales=scales, quats=quats, opacities=opacities, sh0=sh0, shN=None)


# Strategy 3: When is zero-copy actually possible?
def analyze_zero_copy_scenarios():
    """Analyze when zero-copy is theoretically possible."""
    print("=" * 80)
    print("Zero-Copy Analysis: When Is It Possible?")
    print("=" * 80)

    scenarios = {
        "Slicing existing array": "ZERO-COPY",
        "View into existing memory": "ZERO-COPY",
        "Concatenating contiguous arrays": "REQUIRES COPY (fundamental limitation)",
        "Reshaping without copy": "ZERO-COPY (if compatible strides)",
        "Transpose": "ZERO-COPY (just changes strides)",
        "Multiple arrays -> single array": "REQUIRES COPY (must be contiguous)",
    }

    print("\nNumPy Memory Model Limitations:")
    print("-" * 80)
    for scenario, result in scenarios.items():
        print(f"  {scenario:40s}: {result}")

    print("\nWhy concatenation requires copy:")
    print("-" * 80)
    print("  1. NumPy arrays must be contiguous in memory (or strided views)")
    print("  2. Multiple separate memory blocks cannot appear as one")
    print("  3. Virtual memory remapping (mremap) is platform-specific")
    print("  4. Object arrays have indirection overhead")

    print("\nTrue zero-copy alternatives:")
    print("-" * 80)
    print("  1. Don't concatenate - process in batches")
    print("  2. Lazy view for read-only access")
    print("  3. Keep arrays separate, track offsets")
    print("  4. Use database/out-of-core approach")


def benchmark_lazy_view():
    """Benchmark lazy concatenation view."""
    print("\n" + "=" * 80)
    print("Lazy Concatenation View Benchmark")
    print("=" * 80)

    # Create test data
    arrays = [
        GSData(
            means=np.random.randn(10000, 3).astype(np.float32),
            scales=np.random.rand(10000, 3).astype(np.float32),
            quats=np.tile([1, 0, 0, 0], (10000, 1)).astype(np.float32),
            opacities=np.random.rand(10000).astype(np.float32),
            sh0=np.random.rand(10000, 3).astype(np.float32),
            shN=None,
        )
        for _ in range(10)
    ]

    print(f"\nConcatenating 10 arrays of 10,000 Gaussians")
    print("-" * 80)

    # Test 1: Lazy view creation (zero-copy)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        lazy = LazyGSData(arrays)
        end = time.perf_counter()
        times.append(end - start)
    lazy_time = np.mean(times) * 1000
    print(f"Lazy view creation: {lazy_time:.6f} ms (zero allocation/copy)")

    # Test 2: Regular concatenate (with copy)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        result = GSData.concatenate(arrays)
        end = time.perf_counter()
        times.append(end - start)
    concat_time = np.mean(times) * 1000
    print(f"Regular concatenate: {concat_time:.3f} ms (full copy)")

    print(f"\nLazy view overhead: ~{lazy_time:.6f} ms (negligible)")
    print(f"But: accessing data triggers materialization")

    # Test 3: Access pattern comparison
    lazy = LazyGSData(arrays)
    materialized = GSData.concatenate(arrays)

    # Single element access
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = lazy.get_means(50000)  # Middle element
        end = time.perf_counter()
        times.append(end - start)
    lazy_access = np.mean(times) * 1000000  # microseconds
    print(f"\nSingle element access (lazy):       {lazy_access:.3f} us")

    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = materialized.means[50000]
        end = time.perf_counter()
        times.append(end - start)
    direct_access = np.mean(times) * 1000000
    print(f"Single element access (materialized): {direct_access:.3f} us")
    print(f"Lazy overhead: {lazy_access / direct_access:.1f}x slower")


def benchmark_buffer_pool():
    """Benchmark buffer pool approach."""
    print("\n" + "=" * 80)
    print("Buffer Pool Benchmark")
    print("=" * 80)

    arrays = [
        GSData(
            means=np.random.randn(10000, 3).astype(np.float32),
            scales=np.random.rand(10000, 3).astype(np.float32),
            quats=np.tile([1, 0, 0, 0], (10000, 1)).astype(np.float32),
            opacities=np.random.rand(10000).astype(np.float32),
            sh0=np.random.rand(10000, 3).astype(np.float32),
            shN=None,
        )
        for _ in range(10)
    ]

    print(f"\nConcatenating 10 arrays, repeated 20 times")
    print("-" * 80)

    # Test 1: Regular concatenate (allocates every time)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        result = GSData.concatenate(arrays)
        end = time.perf_counter()
        times.append(end - start)
    regular_time = np.mean(times) * 1000
    print(f"Regular concatenate: {regular_time:.3f} ms (allocate every time)")

    # Test 2: Buffer pool (amortized allocation)
    pool = BufferPool()
    times = []
    for i in range(20):
        start = time.perf_counter()
        result = concatenate_with_pool(arrays, pool)
        end = time.perf_counter()
        times.append(end - start)

        # Return buffers to pool for reuse
        if i < 19:  # Don't return on last iteration
            pool.return_buffer(result.means)
            pool.return_buffer(result.scales)
            pool.return_buffer(result.quats)
            pool.return_buffer(result.opacities)
            pool.return_buffer(result.sh0)

    pool_time = np.mean(times) * 1000
    print(f"Buffer pool:         {pool_time:.3f} ms (reuse allocations)")
    print(f"Speedup:             {regular_time / pool_time:.2f}x")


def demonstrate_true_zero_copy():
    """Demonstrate when zero-copy is actually achieved."""
    print("\n" + "=" * 80)
    print("True Zero-Copy Examples")
    print("=" * 80)

    data = GSData(
        means=np.random.randn(100000, 3).astype(np.float32),
        scales=np.random.rand(100000, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (100000, 1)).astype(np.float32),
        opacities=np.random.rand(100000).astype(np.float32),
        sh0=np.random.rand(100000, 3).astype(np.float32),
        shN=None,
    )

    print("\n1. Slicing (ZERO-COPY):")
    print("-" * 80)
    subset = data[:1000]
    print(f"  Original data: {data.means.shape}")
    print(f"  Sliced data:   {subset.means.shape}")
    print(f"  Shares memory: {np.shares_memory(data.means, subset.means)}")
    print(f"  Subset owns data: {subset.means.flags['OWNDATA']}")

    print("\n2. View creation with _base (ZERO-COPY):")
    print("-" * 80)
    # Create data with _base
    base_array = np.random.randn(100000, 14).astype(np.float32)
    data_with_base = GSData._recreate_from_base(base_array)
    print(f"  _base array: {data_with_base._base.shape}")
    print(f"  means view:  {data_with_base.means.shape}")
    print(f"  Shares memory: {np.shares_memory(data_with_base._base, data_with_base.means)}")
    print(f"  Means owns data: {data_with_base.means.flags['OWNDATA']}")

    print("\n3. Array transpose (ZERO-COPY):")
    print("-" * 80)
    means_T = data.means.T
    print(f"  Original strides: {data.means.strides}")
    print(f"  Transposed strides: {means_T.strides}")
    print(f"  Shares memory: {np.shares_memory(data.means, means_T)}")

    print("\n4. Reshape (ZERO-COPY when possible):")
    print("-" * 80)
    flat = data.means.reshape(-1)
    print(f"  Original shape: {data.means.shape}")
    print(f"  Reshaped shape: {flat.shape}")
    print(f"  Shares memory: {np.shares_memory(data.means, flat)}")
    print(f"  Is contiguous: {flat.flags['C_CONTIGUOUS']}")


def main():
    """Run all zero-copy analyses."""
    analyze_zero_copy_scenarios()
    demonstrate_true_zero_copy()
    benchmark_lazy_view()
    benchmark_buffer_pool()

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("1. True zero-copy concatenation is IMPOSSIBLE in NumPy")
    print("2. Lazy views provide zero creation overhead but:")
    print("   - Slower element access (indirection)")
    print("   - Most operations trigger materialization")
    print("3. Buffer pools reduce allocation overhead slightly")
    print("4. Best 'zero-copy' approach: DON'T concatenate")
    print("   - Process arrays in batches")
    print("   - Only concatenate final results if needed")
    print("=" * 80)


if __name__ == "__main__":
    main()
