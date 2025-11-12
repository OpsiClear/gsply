"""Test if we can achieve zero-copy by keeping data alive."""

import numpy as np
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from test_utils import get_test_file

def benchmark_return_strategies():
    """Compare different return strategies for zero-copy."""

    test_file = get_test_file()

    print("=" * 80)
    print("ZERO-COPY RETURN STRATEGY ANALYSIS")
    print("=" * 80)
    print()

    iterations = 50

    # ==========================================================================
    # CURRENT APPROACH (returns copies/views, data goes out of scope)
    # ==========================================================================
    print("Testing CURRENT APPROACH (standard return)...")

    def read_current(file_path):
        """Current implementation."""
        with open(file_path, 'rb') as f:
            # Skip header
            while True:
                line = f.readline()
                if b'end_header' in line:
                    break

            # Read data
            data = np.fromfile(f, dtype=np.float32, count=50375 * 59)
            data = data.reshape(50375, 59)

        # Extract arrays (data goes out of scope after return)
        means = data[:, 0:3]
        sh0 = data[:, 3:6]
        shN = data[:, 6:51].copy().reshape(50375, 15, 3)  # Must copy
        opacities = data[:, 51]
        scales = data[:, 52:55]
        quats = data[:, 55:59]

        return means, scales, quats, opacities, sh0, shN

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = read_current(test_file)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_current = np.mean(times) * 1000
    std_current = np.std(times) * 1000
    print(f"  Time: {mean_current:.2f}ms +/- {std_current:.2f}ms")
    print()

    # ==========================================================================
    # ALTERNATIVE 1: Return data array + views (keep data alive)
    # ==========================================================================
    print("Testing ALTERNATIVE 1 (return base array + views)...")

    def read_with_base(file_path):
        """Return base array to keep it alive."""
        with open(file_path, 'rb') as f:
            # Skip header
            while True:
                line = f.readline()
                if b'end_header' in line:
                    break

            # Read data
            data = np.fromfile(f, dtype=np.float32, count=50375 * 59)
            data = data.reshape(50375, 59)

        # Return views (no copies needed!)
        means = data[:, 0:3]
        sh0 = data[:, 3:6]
        shN = data[:, 6:51].reshape(50375, 15, 3)  # View only!
        opacities = data[:, 51]
        scales = data[:, 52:55]
        quats = data[:, 55:59]

        # Return data too, to keep it alive
        return data, means, scales, quats, opacities, sh0, shN

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = read_with_base(test_file)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_base = np.mean(times) * 1000
    std_base = np.std(times) * 1000
    print(f"  Time: {mean_base:.2f}ms +/- {std_base:.2f}ms")
    print(f"  Speedup: {mean_current/mean_base:.2f}x")
    print(f"  Savings: {mean_current - mean_base:.2f}ms")
    print()

    # Verify views work correctly
    data, means, scales, quats, opacities, sh0, shN = read_with_base(test_file)
    print("  Verification:")
    print(f"    means.base is data:   {means.base is data}")
    print(f"    sh0.base is data:     {sh0.base is data}")
    print(f"    shN.base is data:     {shN.base is data.ravel()}")  # reshaped view
    print(f"    opacities.base is data: {opacities.base is data}")
    print()

    # ==========================================================================
    # ALTERNATIVE 2: Return namedtuple (cleaner API)
    # ==========================================================================
    print("Testing ALTERNATIVE 2 (namedtuple with base)...")

    from collections import namedtuple
    GaussianData = namedtuple('GaussianData', [
        'base', 'means', 'scales', 'quats', 'opacities', 'sh0', 'shN'
    ])

    def read_namedtuple(file_path):
        """Return namedtuple to keep base alive."""
        with open(file_path, 'rb') as f:
            # Skip header
            while True:
                line = f.readline()
                if b'end_header' in line:
                    break

            # Read data
            data = np.fromfile(f, dtype=np.float32, count=50375 * 59)
            data = data.reshape(50375, 59)

        return GaussianData(
            base=data,  # Keep alive
            means=data[:, 0:3],
            scales=data[:, 52:55],
            quats=data[:, 55:59],
            opacities=data[:, 51],
            sh0=data[:, 3:6],
            shN=data[:, 6:51].reshape(50375, 15, 3)
        )

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = read_namedtuple(test_file)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_tuple = np.mean(times) * 1000
    std_tuple = np.std(times) * 1000
    print(f"  Time: {mean_tuple:.2f}ms +/- {std_tuple:.2f}ms")
    print(f"  Speedup: {mean_current/mean_tuple:.2f}x")
    print(f"  Savings: {mean_current - mean_tuple:.2f}ms")
    print()

    # ==========================================================================
    # ALTERNATIVE 3: Dataclass (modern approach)
    # ==========================================================================
    print("Testing ALTERNATIVE 3 (dataclass with base)...")

    from dataclasses import dataclass

    @dataclass
    class GaussianArrays:
        base: np.ndarray
        means: np.ndarray
        scales: np.ndarray
        quats: np.ndarray
        opacities: np.ndarray
        sh0: np.ndarray
        shN: np.ndarray

    def read_dataclass(file_path):
        """Return dataclass to keep base alive."""
        with open(file_path, 'rb') as f:
            # Skip header
            while True:
                line = f.readline()
                if b'end_header' in line:
                    break

            # Read data
            data = np.fromfile(f, dtype=np.float32, count=50375 * 59)
            data = data.reshape(50375, 59)

        return GaussianArrays(
            base=data,
            means=data[:, 0:3],
            scales=data[:, 52:55],
            quats=data[:, 55:59],
            opacities=data[:, 51],
            sh0=data[:, 3:6],
            shN=data[:, 6:51].reshape(50375, 15, 3)
        )

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = read_dataclass(test_file)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_dataclass = np.mean(times) * 1000
    std_dataclass = np.std(times) * 1000
    print(f"  Time: {mean_dataclass:.2f}ms +/- {std_dataclass:.2f}ms")
    print(f"  Speedup: {mean_current/mean_dataclass:.2f}x")
    print(f"  Savings: {mean_current - mean_dataclass:.2f}ms")
    print()

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    print(f"Current approach:   {mean_current:.2f}ms (with shN copy)")
    print(f"Zero-copy tuple:    {mean_tuple:.2f}ms (no copies)")
    print(f"Zero-copy dataclass: {mean_dataclass:.2f}ms (no copies)")
    print()
    print(f"Performance gain from zero-copy: ~{mean_current - mean_tuple:.2f}ms ({(mean_current - mean_tuple)/mean_current*100:.1f}%)")
    print()
    print("RECOMMENDATION:")
    print("  If API change is acceptable, use namedtuple or dataclass return")
    print("  to enable zero-copy views. This would bring read performance")
    print(f"  from ~{mean_current:.1f}ms to ~{mean_tuple:.1f}ms (~{mean_current/mean_tuple:.1f}x speedup).")
    print()
    print("  Trade-off: Users must now use attribute access or slicing:")
    print("    OLD: means, scales, quats, opacities, sh0, shN = plyread(file)  # No longer works")
    print("    NEW: data = plyread(file); means = data.means; scales = data.scales; ...")
    print("    OR:  means, scales, ... = plyread(file)[:6]  # Slice first 6 fields")
    print()
    print("  This is a BREAKING API change but provides significant speedup.")

if __name__ == "__main__":
    benchmark_return_strategies()
