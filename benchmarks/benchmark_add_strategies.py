"""Compare different CPU concatenation strategies for add() method.

Strategies to test:
1. np.concatenate (current approach)
2. Pre-allocation + direct assignment
3. np.vstack
4. Numba JIT-compiled concatenation
"""

import time

import numba
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


# Strategy 1: Current np.concatenate approach
def add_concatenate(data1: GSData, data2: GSData) -> GSData:
    """Current approach using np.concatenate."""
    return GSData(
        means=np.concatenate([data1.means, data2.means], axis=0),
        scales=np.concatenate([data1.scales, data2.scales], axis=0),
        quats=np.concatenate([data1.quats, data2.quats], axis=0),
        opacities=np.concatenate([data1.opacities, data2.opacities]),
        sh0=np.concatenate([data1.sh0, data2.sh0], axis=0),
        shN=None,
    )


# Strategy 2: Pre-allocation + direct assignment
def add_preallocate(data1: GSData, data2: GSData) -> GSData:
    """Pre-allocate arrays and use direct assignment."""
    n1 = len(data1)
    n2 = len(data2)
    total = n1 + n2

    # Pre-allocate all arrays
    means = np.empty((total, 3), dtype=np.float32)
    scales = np.empty((total, 3), dtype=np.float32)
    quats = np.empty((total, 4), dtype=np.float32)
    opacities = np.empty(total, dtype=np.float32)
    sh0 = np.empty((total, 3), dtype=np.float32)

    # Direct assignment (should be faster than concatenate)
    means[:n1] = data1.means
    means[n1:] = data2.means
    scales[:n1] = data1.scales
    scales[n1:] = data2.scales
    quats[:n1] = data1.quats
    quats[n1:] = data2.quats
    opacities[:n1] = data1.opacities
    opacities[n1:] = data2.opacities
    sh0[:n1] = data1.sh0
    sh0[n1:] = data2.sh0

    return GSData(means=means, scales=scales, quats=quats, opacities=opacities, sh0=sh0, shN=None)


# Strategy 3: np.vstack (for 2D arrays)
def add_vstack(data1: GSData, data2: GSData) -> GSData:
    """Use np.vstack for 2D arrays."""
    return GSData(
        means=np.vstack([data1.means, data2.means]),
        scales=np.vstack([data1.scales, data2.scales]),
        quats=np.vstack([data1.quats, data2.quats]),
        opacities=np.concatenate([data1.opacities, data2.opacities]),
        sh0=np.vstack([data1.sh0, data2.sh0]),
        shN=None,
    )


# Strategy 4: Numba JIT-compiled copy
@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_copy_2d(src, dst, dst_start):
    """Copy 2D array using Numba parallel loops."""
    n, m = src.shape
    for i in numba.prange(n):
        for j in range(m):
            dst[dst_start + i, j] = src[i, j]


@numba.jit(nopython=True, parallel=True, cache=True)
def _numba_copy_1d(src, dst, dst_start):
    """Copy 1D array using Numba parallel loops."""
    n = src.shape[0]
    for i in numba.prange(n):
        dst[dst_start + i] = src[i]


def add_numba(data1: GSData, data2: GSData) -> GSData:
    """Use Numba JIT-compiled parallel copy."""
    n1 = len(data1)
    n2 = len(data2)
    total = n1 + n2

    # Pre-allocate
    means = np.empty((total, 3), dtype=np.float32)
    scales = np.empty((total, 3), dtype=np.float32)
    quats = np.empty((total, 4), dtype=np.float32)
    opacities = np.empty(total, dtype=np.float32)
    sh0 = np.empty((total, 3), dtype=np.float32)

    # Numba parallel copy
    _numba_copy_2d(data1.means, means, 0)
    _numba_copy_2d(data2.means, means, n1)
    _numba_copy_2d(data1.scales, scales, 0)
    _numba_copy_2d(data2.scales, scales, n1)
    _numba_copy_2d(data1.quats, quats, 0)
    _numba_copy_2d(data2.quats, quats, n1)
    _numba_copy_1d(data1.opacities, opacities, 0)
    _numba_copy_1d(data2.opacities, opacities, n1)
    _numba_copy_2d(data1.sh0, sh0, 0)
    _numba_copy_2d(data2.sh0, sh0, n1)

    return GSData(means=means, scales=scales, quats=quats, opacities=opacities, sh0=sh0, shN=None)


# Strategy 5: Single large concatenate (pack everything first)
def add_packed(data1: GSData, data2: GSData) -> GSData:
    """Pack arrays into single array, concatenate once, then unpack."""
    n1 = len(data1)
    n2 = len(data2)

    # Pack into single arrays (14 properties for SH0)
    packed1 = np.empty((n1, 14), dtype=np.float32)
    packed1[:, 0:3] = data1.means
    packed1[:, 3:6] = data1.sh0
    packed1[:, 6] = data1.opacities
    packed1[:, 7:10] = data1.scales
    packed1[:, 10:14] = data1.quats

    packed2 = np.empty((n2, 14), dtype=np.float32)
    packed2[:, 0:3] = data2.means
    packed2[:, 3:6] = data2.sh0
    packed2[:, 6] = data2.opacities
    packed2[:, 7:10] = data2.scales
    packed2[:, 10:14] = data2.quats

    # Single concatenate
    packed = np.concatenate([packed1, packed2], axis=0)

    # Unpack
    return GSData(
        means=packed[:, 0:3],
        sh0=packed[:, 3:6],
        opacities=packed[:, 6],
        scales=packed[:, 7:10],
        quats=packed[:, 10:14],
        shN=None,
    )


def benchmark_strategy(func, data1, data2, warmup=5, iterations=50):
    """Benchmark a concatenation strategy."""
    # Warmup
    for _ in range(warmup):
        _ = func(data1, data2)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(data1, data2)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000  # ms
    throughput = (len(data1) + len(data2)) / (avg_time / 1000) / 1e6  # M/s

    return {"time_ms": avg_time, "std_ms": std_time, "throughput_M/s": throughput}


def main():
    """Run strategy comparison."""
    print("=" * 80)
    print("CPU Concatenation Strategy Comparison")
    print("=" * 80)

    strategies = [
        ("np.concatenate (current)", add_concatenate),
        ("Pre-allocate + assign", add_preallocate),
        ("np.vstack", add_vstack),
        ("Numba parallel copy", add_numba),
        ("Pack + single concat", add_packed),
    ]

    for n in [10_000, 100_000, 500_000]:
        print(f"\nSize: {n:,} Gaussians per array ({2*n:,} total)")
        print("-" * 80)

        data1 = create_gsdata(n)
        data2 = create_gsdata(n)

        results = []
        for name, func in strategies:
            result = benchmark_strategy(func, data1, data2)
            results.append((name, result))
            print(f"{name:30s}: {result['time_ms']:7.3f} ms "
                  f"(+/- {result['std_ms']:.3f}) | {result['throughput_M/s']:6.1f} M/s")

        # Find fastest
        fastest = min(results, key=lambda x: x[1]['time_ms'])
        baseline = results[0][1]['time_ms']  # np.concatenate is baseline

        print(f"\nFastest: {fastest[0]} ({fastest[1]['time_ms']:.3f} ms)")
        print(f"Speedup vs baseline: {baseline / fastest[1]['time_ms']:.2f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
