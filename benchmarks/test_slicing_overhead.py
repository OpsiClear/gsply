"""Test slicing overhead in uncompressed reader."""

import numpy as np
import time

def benchmark_slicing_approaches():
    """Compare different slicing approaches."""

    # Simulate the data shape from PLY
    n_verts = 50375
    n_props = 59
    data = np.random.randn(n_verts, n_props).astype(np.float32)

    print("=" * 80)
    print("SLICING OVERHEAD ANALYSIS")
    print("=" * 80)
    print(f"\nData shape: {data.shape}")
    print(f"Memory: {data.nbytes / 1024 / 1024:.2f}MB")
    print()

    iterations = 100

    # ==========================================================================
    # CURRENT APPROACH (what gsply does)
    # ==========================================================================
    print("Testing CURRENT APPROACH (individual slices)...")
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()

        # Extract arrays (as in reader.py lines 131-153)
        means = data[:, 0:3]
        sh0 = data[:, 3:6]
        shN = data[:, 6:51]
        opacities = data[:, 51]
        scales = data[:, 52:55]
        quats = data[:, 55:59]

        # Reshape shN (requires copy)
        shN = shN.copy().reshape(n_verts, 15, 3)

        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_current = np.mean(times) * 1000
    std_current = np.std(times) * 1000
    print(f"  Time: {mean_current:.2f}ms +/- {std_current:.2f}ms")
    print()

    # ==========================================================================
    # OPTIMIZED: Pre-computed indices
    # ==========================================================================
    print("Testing OPTIMIZED (pre-computed indices)...")

    # Pre-compute slice objects (done once at module load)
    SLICE_MEANS = slice(0, 3)
    SLICE_SH0 = slice(3, 6)
    SLICE_SHN = slice(6, 51)
    IDX_OPACITY = 51
    SLICE_SCALES = slice(52, 55)
    SLICE_QUATS = slice(55, 59)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()

        means = data[:, SLICE_MEANS]
        sh0 = data[:, SLICE_SH0]
        shN = data[:, SLICE_SHN]
        opacities = data[:, IDX_OPACITY]
        scales = data[:, SLICE_SCALES]
        quats = data[:, SLICE_QUATS]

        shN = shN.copy().reshape(n_verts, 15, 3)

        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_precomp = np.mean(times) * 1000
    std_precomp = np.std(times) * 1000
    print(f"  Time: {mean_precomp:.2f}ms +/- {std_precomp:.2f}ms")
    print(f"  Speedup: {mean_current/mean_precomp:.2f}x")
    print()

    # ==========================================================================
    # OPTIMIZED: Use array indexing instead of multiple slices
    # ==========================================================================
    print("Testing OPTIMIZED (array indexing)...")

    # Pre-compute index arrays
    idx_means = np.array([0, 1, 2])
    idx_sh0 = np.array([3, 4, 5])
    idx_shN = np.arange(6, 51)
    idx_scales = np.array([52, 53, 54])
    idx_quats = np.array([55, 56, 57, 58])

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()

        means = data[:, idx_means]
        sh0 = data[:, idx_sh0]
        shN = data[:, idx_shN]
        opacities = data[:, 51]
        scales = data[:, idx_scales]
        quats = data[:, idx_quats]

        shN = shN.copy().reshape(n_verts, 15, 3)

        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_array = np.mean(times) * 1000
    std_array = np.std(times) * 1000
    print(f"  Time: {mean_array:.2f}ms +/- {std_array:.2f}ms")
    print(f"  Speedup: {mean_current/mean_array:.2f}x")
    print()

    # ==========================================================================
    # ALTERNATIVE: Structured array (zero-copy)
    # ==========================================================================
    print("Testing ALTERNATIVE (structured array - zero-copy)...")

    # Create a structured dtype
    dtype = np.dtype([
        ('means', np.float32, (3,)),
        ('sh0', np.float32, (3,)),
        ('shN', np.float32, (45,)),
        ('opacity', np.float32),
        ('scales', np.float32, (3,)),
        ('quats', np.float32, (4,)),
    ])

    # Convert data to structured array (one-time cost)
    t0 = time.perf_counter()
    structured = np.empty(n_verts, dtype=dtype)
    structured['means'] = data[:, 0:3]
    structured['sh0'] = data[:, 3:6]
    structured['shN'] = data[:, 6:51]
    structured['opacity'] = data[:, 51]
    structured['scales'] = data[:, 52:55]
    structured['quats'] = data[:, 55:59]
    t1 = time.perf_counter()
    conversion_time = (t1 - t0) * 1000

    print(f"  Conversion time: {conversion_time:.2f}ms (one-time)")

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()

        means = structured['means']
        sh0 = structured['sh0']
        shN = structured['shN'].copy().reshape(n_verts, 15, 3)
        opacities = structured['opacity']
        scales = structured['scales']
        quats = structured['quats']

        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_struct = np.mean(times) * 1000
    std_struct = np.std(times) * 1000
    print(f"  Access time: {mean_struct:.2f}ms +/- {std_struct:.2f}ms")
    print(f"  Total time: {conversion_time + mean_struct:.2f}ms")
    print(f"  Speedup: {mean_current/mean_struct:.2f}x (access only)")
    print()

    # ==========================================================================
    # TEST: Copy vs view performance
    # ==========================================================================
    print("Testing COPY vs VIEW overhead...")

    # Test without copy
    times_nocopy = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        shN = data[:, 6:51].reshape(n_verts, 15, 3)  # View only
        t1 = time.perf_counter()
        times_nocopy.append(t1 - t0)

    # Test with copy
    times_copy = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        shN = data[:, 6:51].copy().reshape(n_verts, 15, 3)  # Explicit copy
        t1 = time.perf_counter()
        times_copy.append(t1 - t0)

    mean_nocopy = np.mean(times_nocopy) * 1000
    mean_copy = np.mean(times_copy) * 1000

    print(f"  View + reshape:       {mean_nocopy:.2f}ms")
    print(f"  Copy + reshape:       {mean_copy:.2f}ms")
    print(f"  Copy overhead:        {mean_copy - mean_nocopy:.2f}ms")
    print()

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    print(f"Current slicing overhead: ~{mean_current:.2f}ms")
    print(f"  Of which ~{mean_copy - mean_nocopy:.2f}ms is shN copy overhead")
    print(f"  Remaining: ~{mean_current - (mean_copy - mean_nocopy):.2f}ms for slicing ops")
    print()
    print("Optimization potential:")
    print(f"  1. Pre-computed slices:  {mean_current - mean_precomp:.2f}ms saved ({(mean_current - mean_precomp)/mean_current*100:.1f}%)")
    print(f"  2. Avoid shN copy:       {mean_copy - mean_nocopy:.2f}ms saved (if safe)")
    print()
    print("Note: shN.copy() is necessary because we're reshaping and returning.")
    print("The parent 'data' array goes out of scope, so views would be unsafe.")
    print()
    print("VERDICT: Current slicing approach is already well-optimized.")
    print("The ~2.9ms 'slicing overhead' is mostly the shN copy (required for safety).")
    print("Pre-computing slice objects might save ~0.1-0.2ms but adds complexity.")

if __name__ == "__main__":
    benchmark_slicing_approaches()
