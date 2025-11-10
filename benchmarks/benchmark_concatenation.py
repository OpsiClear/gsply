"""Test potential optimizations for gsply SH0 write performance."""

import time
import numpy as np
from pathlib import Path

import gsply


def test_concatenate_methods(means, scales, quats, opacities, sh0, iterations=100):
    """Compare different concatenation methods."""

    results = {}

    # Method 1: Current approach (concatenate + astype)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        data = np.concatenate([means, sh0, opacities[:, None], scales, quats], axis=1)
        data = data.astype('<f4')
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['concatenate + astype'] = (np.mean(times) * 1000, np.std(times) * 1000)

    # Method 2: Ensure dtype before concatenate
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        data = np.concatenate([
            means.astype('<f4', copy=False),
            sh0.astype('<f4', copy=False),
            opacities[:, None].astype('<f4', copy=False),
            scales.astype('<f4', copy=False),
            quats.astype('<f4', copy=False)
        ], axis=1)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['pre-astype + concatenate'] = (np.mean(times) * 1000, np.std(times) * 1000)

    # Method 3: column_stack
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        data = np.column_stack([
            means[:, 0], means[:, 1], means[:, 2],
            sh0[:, 0], sh0[:, 1], sh0[:, 2],
            opacities,
            scales[:, 0], scales[:, 1], scales[:, 2],
            quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        ]).astype('<f4')
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['column_stack + astype'] = (np.mean(times) * 1000, np.std(times) * 1000)

    # Method 4: Pre-allocate and assign (like plyfile but for regular array)
    times = []
    num_gaussians = means.shape[0]
    for _ in range(iterations):
        t0 = time.perf_counter()
        data = np.empty((num_gaussians, 14), dtype='<f4')
        data[:, 0:3] = means
        data[:, 3:6] = sh0
        data[:, 6] = opacities
        data[:, 7:10] = scales
        data[:, 10:14] = quats
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['preallocate + assign'] = (np.mean(times) * 1000, np.std(times) * 1000)

    # Method 5: Using hstack
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        data = np.hstack([means, sh0, opacities[:, None], scales, quats]).astype('<f4')
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['hstack + astype'] = (np.mean(times) * 1000, np.std(times) * 1000)

    # Method 6: Check if data is already correct dtype (no-copy path)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        data = np.concatenate([means, sh0, opacities[:, None], scales, quats], axis=1)
        if data.dtype != np.dtype('<f4'):
            data = data.astype('<f4')
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['concatenate + conditional astype'] = (np.mean(times) * 1000, np.std(times) * 1000)

    return results


def main():
    # Load test data
    test_file = Path("../export_with_edits/frame_00000.ply")
    means, scales, quats, opacities, sh0, shN = gsply.plyread(test_file)

    print("=" * 80)
    print("CONCATENATION METHOD COMPARISON (SH0)")
    print("=" * 80)
    print(f"\nTest data: {means.shape[0]} Gaussians")
    print(f"Input dtypes: means={means.dtype}, sh0={sh0.dtype}, opacities={opacities.dtype}")
    print()

    results = test_concatenate_methods(means, scales, quats, opacities, sh0)

    # Sort by speed
    sorted_results = sorted(results.items(), key=lambda x: x[1][0])

    print(f"{'Method':<40} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    baseline = sorted_results[0][1][0]
    for method, (mean_time, std_time) in sorted_results:
        speedup = baseline / mean_time
        speedup_str = "baseline" if speedup == 1.0 else f"{speedup:.2f}x"
        print(f"{method:<40} {mean_time:.3f} +/- {std_time:.3f}   {speedup_str:<10}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best_method = sorted_results[0][0]
    improvement = (results['concatenate + astype'][0] - sorted_results[0][1][0])
    improvement_pct = (improvement / results['concatenate + astype'][0]) * 100

    print(f"\nBest method: {best_method}")
    print(f"Improvement over current: {improvement:.3f}ms ({improvement_pct:.1f}% faster)")


if __name__ == "__main__":
    main()
