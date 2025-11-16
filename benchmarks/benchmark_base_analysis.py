"""Deep analysis of _base array structure and optimization opportunities.

Analyzes:
1. Memory layout of _base (row-major interleaved)
2. Cache locality and access patterns
3. Concatenation strategies for _base path
4. Comparison: pairwise add() vs bulk concatenate
"""

import time

import numpy as np

from gsply import GSData


def create_gsdata(n: int) -> GSData:
    """Create test GSData."""
    return GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )


def create_gsdata_with_base(n: int) -> GSData:
    """Create GSData with _base array."""
    base_array = np.random.randn(n, 14).astype(np.float32)
    return GSData._recreate_from_base(base_array)


def analyze_memory_layout():
    """Analyze _base memory layout and access patterns."""
    print("=" * 80)
    print("_base Memory Layout Analysis")
    print("=" * 80)

    data = create_gsdata_with_base(1000)

    print(f"\n_base shape: {data._base.shape}")
    print(f"_base dtype: {data._base.dtype}")
    print(f"_base strides: {data._base.strides}")
    print(f"_base is C-contiguous: {data._base.flags['C_CONTIGUOUS']}")
    print(f"_base is F-contiguous: {data._base.flags['F_CONTIGUOUS']}")

    print(f"\nLayout for SH0 (14 properties per Gaussian):")
    print(f"  [0:3]   means    - shape {data.means.shape}, strides {data.means.strides}")
    print(f"  [3:6]   sh0      - shape {data.sh0.shape}, strides {data.sh0.strides}")
    print(f"  [6]     opacity  - shape {data.opacities.shape}, strides {data.opacities.strides}")
    print(f"  [7:10]  scales   - shape {data.scales.shape}, strides {data.scales.strides}")
    print(f"  [10:14] quats    - shape {data.quats.shape}, strides {data.quats.strides}")

    print(f"\nMemory sharing:")
    print(f"  means shares memory with _base: {np.shares_memory(data.means, data._base)}")
    print(f"  means is C-contiguous: {data.means.flags['C_CONTIGUOUS']}")
    print(f"  means is view (not owner): {not data.means.flags['OWNDATA']}")

    # Calculate memory overhead
    total_elements = 1000 * 14
    base_bytes = total_elements * 4  # float32
    print(f"\nMemory efficiency:")
    print(f"  Total elements: {total_elements:,}")
    print(f"  _base memory: {base_bytes:,} bytes ({base_bytes/1024:.1f} KB)")
    print(f"  All views share this memory (zero overhead!)")


def benchmark_base_concat_strategies(n: int = 100_000, iterations: int = 50):
    """Compare concatenation strategies specifically for _base arrays."""
    print("\n" + "=" * 80)
    print(f"_base Concatenation Strategies ({n:,} Gaussians)")
    print("=" * 80)

    data1 = create_gsdata_with_base(n)
    data2 = create_gsdata_with_base(n)

    strategies = {
        "Current (pre-allocate)": lambda: concat_preallocate(data1._base, data2._base),
        "np.concatenate": lambda: np.concatenate([data1._base, data2._base], axis=0),
        "np.vstack": lambda: np.vstack([data1._base, data2._base]),
        "np.r_": lambda: np.r_[data1._base, data2._base],
    }

    for name, func in strategies.items():
        # Warmup
        for _ in range(5):
            _ = func()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = func()
            end = time.perf_counter()
            times.append(end - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        throughput = (2 * n) / (avg_time / 1000) / 1e6
        print(f"{name:30s}: {avg_time:7.3f} ms (+/- {std_time:.3f}) | {throughput:6.1f} M/s")


def concat_preallocate(base1, base2):
    """Pre-allocate strategy."""
    n1, n2 = len(base1), len(base2)
    result = np.empty((n1 + n2, base1.shape[1]), dtype=base1.dtype)
    result[:n1] = base1
    result[n1:] = base2
    return result


def benchmark_pairwise_vs_bulk():
    """Compare pairwise add() vs bulk concatenation."""
    print("\n" + "=" * 80)
    print("Pairwise add() vs Bulk Concatenation")
    print("=" * 80)

    n_arrays = 10
    n_gaussians = 10_000

    print(f"\nConcatenating {n_arrays} arrays of {n_gaussians:,} Gaussians each")
    print("-" * 80)

    # Create test data
    arrays = [create_gsdata(n_gaussians) for _ in range(n_arrays)]

    # Strategy 1: Pairwise add()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        result = arrays[0]
        for arr in arrays[1:]:
            result = result.add(arr)
        end = time.perf_counter()
        times.append(end - start)

    pairwise_time = np.mean(times) * 1000
    print(f"Pairwise add():     {pairwise_time:7.3f} ms")

    # Strategy 2: Bulk concatenate (manual implementation)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        result = concatenate_bulk(arrays)
        end = time.perf_counter()
        times.append(end - start)

    bulk_time = np.mean(times) * 1000
    print(f"Bulk concatenate:   {bulk_time:7.3f} ms")
    print(f"Speedup:            {pairwise_time / bulk_time:.2f}x")

    # Verify correctness
    result_pairwise = arrays[0]
    for arr in arrays[1:]:
        result_pairwise = result_pairwise.add(arr)
    result_bulk = concatenate_bulk(arrays)

    assert len(result_pairwise) == len(result_bulk)
    np.testing.assert_allclose(result_pairwise.means, result_bulk.means)
    print("\nCorrectness verified [OK]")


def concatenate_bulk(arrays: list[GSData]) -> GSData:
    """Bulk concatenate multiple GSData objects at once.

    More efficient than pairwise add() because:
    - Single allocation instead of N-1 allocations
    - Reduces total memory copies
    """
    if not arrays:
        raise ValueError("Cannot concatenate empty list")
    if len(arrays) == 1:
        return arrays[0]

    # Validate all have same SH degree
    sh_degree = arrays[0].get_sh_degree()
    for arr in arrays[1:]:
        if arr.get_sh_degree() != sh_degree:
            raise ValueError("All arrays must have same SH degree")

    # Calculate total size
    total = sum(len(arr) for arr in arrays)

    # Pre-allocate output arrays
    means = np.empty((total, 3), dtype=arrays[0].means.dtype)
    scales = np.empty((total, 3), dtype=arrays[0].scales.dtype)
    quats = np.empty((total, 4), dtype=arrays[0].quats.dtype)
    opacities = np.empty(total, dtype=arrays[0].opacities.dtype)
    sh0 = np.empty((total, 3), dtype=arrays[0].sh0.dtype)

    # Copy data in one pass
    offset = 0
    for arr in arrays:
        n = len(arr)
        means[offset : offset + n] = arr.means
        scales[offset : offset + n] = arr.scales
        quats[offset : offset + n] = arr.quats
        opacities[offset : offset + n] = arr.opacities
        sh0[offset : offset + n] = arr.sh0
        offset += n

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


def analyze_cache_locality():
    """Analyze cache locality of different access patterns."""
    print("\n" + "=" * 80)
    print("Cache Locality Analysis")
    print("=" * 80)

    n = 1_000_000
    data = create_gsdata_with_base(n)

    print(f"\nTesting with {n:,} Gaussians")
    print("-" * 80)

    # Test 1: Access via views (current approach)
    iterations = 100
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = data.means.sum()
        end = time.perf_counter()
        times.append(end - start)
    view_time = np.mean(times) * 1000
    print(f"Access via view (data.means.sum()):  {view_time:.3f} ms")

    # Test 2: Access via _base slicing
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = data._base[:, 0:3].sum()
        end = time.perf_counter()
        times.append(end - start)
    base_time = np.mean(times) * 1000
    print(f"Access via _base slice:               {base_time:.3f} ms")

    # Test 3: Access via contiguous copy
    means_copy = data.means.copy()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = means_copy.sum()
        end = time.perf_counter()
        times.append(end - start)
    copy_time = np.mean(times) * 1000
    print(f"Access via contiguous copy:           {copy_time:.3f} ms")

    print(f"\nView overhead: {(view_time / copy_time):.2f}x")


def main():
    """Run all analyses."""
    analyze_memory_layout()
    benchmark_base_concat_strategies()
    benchmark_pairwise_vs_bulk()
    analyze_cache_locality()

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
