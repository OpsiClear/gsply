"""Benchmark color conversion methods (SH ↔ RGB) performance.

Compares NumPy in-place operations vs Numba JIT for different array sizes.
"""

import time

import numpy as np

from gsply import GSData
from gsply.formats import SH_C0
from gsply.utils import _rgb2sh_inplace_jit, _sh2rgb_inplace_jit


def benchmark_numpy_inplace(sh0: np.ndarray, num_iterations: int = 10) -> float:
    """Benchmark NumPy in-place operations."""
    times = []
    for _ in range(num_iterations):
        # Create a copy for each iteration
        test_sh0 = sh0.copy()

        start = time.perf_counter()
        test_sh0 *= SH_C0
        test_sh0 += 0.5
        test_sh0 -= 0.5
        test_sh0 /= SH_C0
        end = time.perf_counter()

        times.append(end - start)

    return np.mean(times[1:])  # Skip first iteration (warmup)


def benchmark_numba_jit(sh0: np.ndarray, num_iterations: int = 10) -> float:
    """Benchmark Numba JIT operations."""
    times = []
    for _ in range(num_iterations):
        # Create a copy for each iteration
        test_sh0 = sh0.copy()

        start = time.perf_counter()
        _sh2rgb_inplace_jit(test_sh0, SH_C0)
        _rgb2sh_inplace_jit(test_sh0, 1.0 / SH_C0)
        end = time.perf_counter()

        times.append(end - start)

    return np.mean(times[1:])  # Skip first iteration (warmup)


def benchmark_gsdata_method(data: GSData, num_iterations: int = 10) -> float:
    """Benchmark GSData.to_rgb() and to_sh() methods."""
    times = []
    for _ in range(num_iterations):
        # Create a copy for each iteration
        test_data = data.copy()

        start = time.perf_counter()
        test_data.to_rgb(inplace=True)
        test_data.to_sh(inplace=True)
        end = time.perf_counter()

        times.append(end - start)

    return np.mean(times[1:])  # Skip first iteration (warmup)


def main():
    """Run benchmarks for different array sizes."""
    print("=" * 80)
    print("Color Conversion Performance Benchmark (SH <-> RGB)")
    print("=" * 80)
    print()

    # Test different array sizes
    sizes = [
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (500_000, "500K"),
        (1_000_000, "1M"),
    ]

    results = []

    for size, label in sizes:
        print(f"Testing {label} Gaussians ({size:,} elements)...")

        # Create test data
        sh0 = np.random.randn(size, 3).astype(np.float32)
        means = np.random.randn(size, 3).astype(np.float32)
        scales = np.ones((size, 3), dtype=np.float32) * 0.01
        quats = np.tile([1, 0, 0, 0], (size, 1)).astype(np.float32)
        opacities = np.ones(size, dtype=np.float32) * 0.5

        data = GSData(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=None,
        )

        # Warmup (first iteration)
        _sh2rgb_inplace_jit(sh0.copy(), SH_C0)

        # Benchmark NumPy in-place
        numpy_time = benchmark_numpy_inplace(sh0, num_iterations=10)

        # Benchmark Numba JIT
        numba_time = benchmark_numba_jit(sh0, num_iterations=10)

        # Benchmark GSData method (uses hybrid approach)
        gsdata_time = benchmark_gsdata_method(data, num_iterations=10)

        # Calculate throughput (Gaussians/sec)
        numpy_throughput = size / numpy_time
        numba_throughput = size / numba_time
        gsdata_throughput = size / gsdata_time

        results.append({
            "size": size,
            "label": label,
            "numpy_time": numpy_time,
            "numba_time": numba_time,
            "gsdata_time": gsdata_time,
            "numpy_throughput": numpy_throughput,
            "numba_throughput": numba_throughput,
            "gsdata_throughput": gsdata_throughput,
        })

        print(f"  NumPy in-place:  {numpy_time*1000:.3f} ms ({numpy_throughput/1e6:.2f} M Gaussians/s)")
        print(f"  Numba JIT:       {numba_time*1000:.3f} ms ({numba_throughput/1e6:.2f} M Gaussians/s)")
        print(f"  GSData method:  {gsdata_time*1000:.3f} ms ({gsdata_throughput/1e6:.2f} M Gaussians/s)")

        if numba_time < numpy_time:
            speedup = numpy_time / numba_time
            print(f"  -> Numba is {speedup:.2f}x faster")
        else:
            speedup = numba_time / numpy_time
            print(f"  -> NumPy is {speedup:.2f}x faster")
        print()

    # Summary table
    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Size':<10} {'NumPy (ms)':<12} {'Numba (ms)':<12} {'GSData (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for r in results:
        speedup = r["numpy_time"] / r["numba_time"] if r["numba_time"] < r["numpy_time"] else r["numba_time"] / r["numpy_time"]
        faster = "Numba" if r["numba_time"] < r["numpy_time"] else "NumPy"
        print(
            f"{r['label']:<10} "
            f"{r['numpy_time']*1000:>10.3f}  "
            f"{r['numba_time']*1000:>10.3f}  "
            f"{r['gsdata_time']*1000:>12.3f}  "
            f"{faster} {speedup:.2f}x"
        )

    print()
    print("=" * 80)
    print("Conclusion:")
    print("=" * 80)

    # Find crossover point
    for i, r in enumerate(results):
        if r["numba_time"] < r["numpy_time"]:
            print(f"Numba becomes faster at {r['label']} Gaussians")
            break
    else:
        print("NumPy remains faster for all tested sizes")

    print()
    print("Note: GSData method always uses Numba JIT for optimal performance")


if __name__ == "__main__":
    main()
