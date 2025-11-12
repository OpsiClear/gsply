"""Benchmark: Can we make gsply faster with miniply's C++ backend?

This script compares:
1. Pure gsply (current): Pure Python + numpy
2. Hybrid approach: pyminiply C++ for reading + gsply data extraction
3. pyminiply.gaussian: Pure Python implementation in pyminiply

The goal is to determine if using C++ miniply for file I/O provides any
performance benefit over numpy's fromfile().
"""

import time
import numpy as np
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "pyminiply" / "src"))

import gsply
from test_utils import get_test_file

def timer(func, warmup=3, iterations=10):
    """Time a function with warmup and multiple iterations."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    return mean_time, std_time, result


def benchmark_gsply_read(file_path):
    """Benchmark pure gsply reading."""
    def read_fn():
        return gsply.plyread(file_path)

    mean_time, std_time, result = timer(read_fn)
    return {
        'name': 'gsply (pure Python)',
        'mean_time': mean_time,
        'std_time': std_time,
        'result': result
    }


def benchmark_pyminiply_gaussian(file_path):
    """Benchmark pyminiply.gaussian reading."""
    try:
        import pyminiply.gaussian as gply

        def read_fn():
            return gply.read(file_path)

        mean_time, std_time, result = timer(read_fn)
        return {
            'name': 'pyminiply.gaussian',
            'mean_time': mean_time,
            'std_time': std_time,
            'result': result
        }
    except ImportError as e:
        return {
            'name': 'pyminiply.gaussian',
            'error': f'Not available: {e}'
        }


def benchmark_hybrid_approach(file_path):
    """Benchmark hybrid: pyminiply C++ read + gsply data extraction."""
    try:
        import pyminiply

        # Define property list for Gaussian PLY (SH degree 3)
        property_list = (
            ['x', 'y', 'z'] +
            ['f_dc_0', 'f_dc_1', 'f_dc_2'] +
            [f'f_rest_{i}' for i in range(45)] +
            ['opacity'] +
            ['scale_0', 'scale_1', 'scale_2'] +
            ['rot_0', 'rot_1', 'rot_2', 'rot_3']
        )

        def read_fn():
            # Use C++ miniply to read properties
            props = pyminiply.read_properties(str(file_path), property_list)

            # Extract using gsply's efficient slicing
            means = props[:, 0:3]
            sh0 = props[:, 3:6]
            shN = props[:, 6:51]
            opacities = props[:, 51]
            scales = props[:, 52:55]
            quats = props[:, 55:59]

            # Reshape shN to (N, 15, 3)
            num_gaussians = means.shape[0]
            shN = shN.reshape(num_gaussians, 15, 3)

            return means, scales, quats, opacities, sh0, shN

        mean_time, std_time, result = timer(read_fn)
        return {
            'name': 'hybrid (miniply C++ + gsply)',
            'mean_time': mean_time,
            'std_time': std_time,
            'result': result
        }
    except (ImportError, AttributeError) as e:
        return {
            'name': 'hybrid (miniply C++ + gsply)',
            'error': f'Not available: {e}'
        }


def main():
    try:
        test_file = get_test_file()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("=" * 80)
    print("GSPLY + MINIPLY HYBRID BENCHMARK")
    print("=" * 80)
    print(f"\nTest file: {test_file}")
    print(f"File size: {test_file.stat().st_size / 1024 / 1024:.2f}MB")
    print()

    results = []

    print("Benchmarking gsply (pure Python)...")
    result = benchmark_gsply_read(test_file)
    results.append(result)

    print("Benchmarking pyminiply.gaussian...")
    result = benchmark_pyminiply_gaussian(test_file)
    results.append(result)

    print("Benchmarking hybrid approach...")
    result = benchmark_hybrid_approach(test_file)
    results.append(result)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Filter successful results
    successful = [r for r in results if 'mean_time' in r]
    failed = [r for r in results if 'error' in r]

    if successful:
        # Sort by speed
        successful.sort(key=lambda x: x['mean_time'])

        print(f"{'Approach':<35} {'Time':<20} {'Speedup':<10}")
        print("-" * 80)

        baseline = successful[0]['mean_time']
        for r in successful:
            speedup = baseline / r['mean_time']
            speedup_str = "baseline" if speedup == 1.0 else f"{speedup:.2f}x"
            print(f"{r['name']:<35} {r['mean_time']:.2f}ms +/- {r['std_time']:.2f}ms   {speedup_str}")

    if failed:
        print("\n" + "=" * 80)
        print("UNAVAILABLE APPROACHES")
        print("=" * 80)
        for r in failed:
            print(f"{r['name']:<35} {r['error']}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    if len(successful) >= 2:
        gsply_result = next((r for r in results if r['name'] == 'gsply (pure Python)'), None)
        hybrid_result = next((r for r in results if r['name'] == 'hybrid (miniply C++ + gsply)'), None)

        if gsply_result and hybrid_result and 'mean_time' in hybrid_result:
            improvement = gsply_result['mean_time'] - hybrid_result['mean_time']
            improvement_pct = (improvement / gsply_result['mean_time']) * 100

            print(f"Current gsply:     {gsply_result['mean_time']:.2f}ms")
            print(f"Hybrid approach:   {hybrid_result['mean_time']:.2f}ms")
            print(f"Difference:        {improvement:+.2f}ms ({improvement_pct:+.1f}%)")
            print()

            if improvement > 0:
                print(f"[OK] Hybrid approach is {improvement:.2f}ms faster!")
                print("     Consider integrating miniply C++ backend into gsply.")
            else:
                print(f"[INFO] Pure Python gsply is already {-improvement:.2f}ms faster.")
                print("       No benefit from C++ miniply for this use case.")
        else:
            print("Hybrid approach not available - pyminiply C++ extension not built.")
            print()
            print("To build pyminiply:")
            print("  cd third_party/pyminiply")
            print("  pip install -e .")
    else:
        print("Insufficient data for comparison.")


if __name__ == "__main__":
    main()
