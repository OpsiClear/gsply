"""Test if preallocate+assign optimization also helps for SH3 (large files)."""

import time
import numpy as np
from pathlib import Path

import gsply


def compare_methods_sh3(means, scales, quats, opacities, sh0, shN, iterations=100):
    """Compare concatenate vs preallocate for SH3."""

    num_gaussians = means.shape[0]
    sh_coeffs = shN.shape[1]  # Number of higher-order SH coefficients
    total_props = 3 + 3 + sh_coeffs*3 + 1 + 3 + 4  # means, sh0, shN, opacity, scales, quats

    results = {}

    # Current method: concatenate
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        data = np.concatenate([means, sh0, shN.reshape(num_gaussians, -1), opacities[:, None], scales, quats], axis=1)
        data = data.astype('<f4')
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['concatenate + astype'] = (np.mean(times) * 1000, np.std(times) * 1000)

    # Optimized method: preallocate
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        data = np.empty((num_gaussians, total_props), dtype='<f4')
        data[:, 0:3] = means
        data[:, 3:6] = sh0
        data[:, 6:6+sh_coeffs*3] = shN.reshape(num_gaussians, -1)
        data[:, 6+sh_coeffs*3] = opacities
        data[:, 7+sh_coeffs*3:10+sh_coeffs*3] = scales
        data[:, 10+sh_coeffs*3:14+sh_coeffs*3] = quats
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['preallocate + assign'] = (np.mean(times) * 1000, np.std(times) * 1000)

    return results


def main():
    # Load test data
    test_file = Path("../export_with_edits/frame_00000.ply")
    means, scales, quats, opacities, sh0, shN = gsply.plyread(test_file)

    print("=" * 80)
    print("SH3 (59 properties) OPTIMIZATION TEST")
    print("=" * 80)
    print(f"\nTest data: {means.shape[0]} Gaussians")
    print(f"SH coefficients: {shN.shape[1]} (SH degree 3)")
    print(f"Total properties: {3 + 3 + shN.shape[1]*3 + 1 + 3 + 4}")
    print()

    results = compare_methods_sh3(means, scales, quats, opacities, sh0, shN)

    for method, (mean_time, std_time) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"{method:<30} {mean_time:.3f}ms +/- {std_time:.3f}ms")

    print()
    improvement = results['concatenate + astype'][0] - results['preallocate + assign'][0]
    improvement_pct = (improvement / results['concatenate + astype'][0]) * 100
    print(f"Improvement: {improvement:.3f}ms ({improvement_pct:.1f}% faster)")


if __name__ == "__main__":
    main()
