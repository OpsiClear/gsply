"""Benchmark comparison of different container types for GSData.

Compares:
1. Current implementation (regular dataclass)
2. Frozen dataclass
3. NamedTuple

Tests both creation time and attribute access performance.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


# =============================================================================
# Container Definitions
# =============================================================================

@dataclass
class GSDataRegular:
    """Regular dataclass (current implementation)."""
    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh0: np.ndarray
    shN: np.ndarray  # noqa: N815
    base: np.ndarray


@dataclass(frozen=True)
class GSDataFrozen:
    """Frozen dataclass (immutable)."""
    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh0: np.ndarray
    shN: np.ndarray  # noqa: N815
    base: np.ndarray


class GSDataNamedTuple(NamedTuple):
    """NamedTuple implementation."""
    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh0: np.ndarray
    shN: np.ndarray
    base: np.ndarray


# =============================================================================
# Benchmark Functions
# =============================================================================

def create_test_data(n_gaussians=100000):
    """Create test numpy arrays."""
    means = np.random.randn(n_gaussians, 3).astype(np.float32)
    scales = np.random.rand(n_gaussians, 3).astype(np.float32) * 0.1
    quats = np.random.randn(n_gaussians, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.rand(n_gaussians).astype(np.float32)
    sh0 = np.random.rand(n_gaussians, 3).astype(np.float32)
    shN = np.random.rand(n_gaussians, 15, 3).astype(np.float32)  # noqa: N806
    base = np.random.rand(n_gaussians, 59).astype(np.float32)

    return means, scales, quats, opacities, sh0, shN, base


def benchmark_creation(container_class, data, n_iterations=10000):
    """Benchmark container creation time."""
    means, scales, quats, opacities, sh0, shN, base = data  # noqa: N806

    start = time.perf_counter()
    for _ in range(n_iterations):
        obj = container_class(means, scales, quats, opacities, sh0, shN, base)
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / n_iterations

    return total_time, avg_time


def benchmark_attribute_access(container_class, data, n_iterations=1000000):
    """Benchmark attribute access time."""
    means, scales, quats, opacities, sh0, shN, base = data  # noqa: N806
    obj = container_class(means, scales, quats, opacities, sh0, shN, base)

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = obj.means
        _ = obj.scales
        _ = obj.quats
        _ = obj.opacities
        _ = obj.sh0
        _ = obj.shN
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / n_iterations

    return total_time, avg_time


def benchmark_indexing(container_class, data, n_iterations=100000):
    """Benchmark indexing/unpacking time."""
    means, scales, quats, opacities, sh0, shN, base = data  # noqa: N806
    obj = container_class(means, scales, quats, opacities, sh0, shN, base)

    start = time.perf_counter()
    for _ in range(n_iterations):
        if hasattr(obj, '__getitem__'):
            _ = obj[0]
            _ = obj[1]
            _ = obj[2]
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / n_iterations

    return total_time, avg_time


def measure_memory_size(container_class, data):
    """Measure approximate memory overhead of container."""
    import sys
    means, scales, quats, opacities, sh0, shN, base = data  # noqa: N806
    obj = container_class(means, scales, quats, opacities, sh0, shN, base)

    # Size of container object itself (not the arrays)
    container_size = sys.getsizeof(obj)

    # Size of arrays
    array_sizes = sum([
        sys.getsizeof(means),
        sys.getsizeof(scales),
        sys.getsizeof(quats),
        sys.getsizeof(opacities),
        sys.getsizeof(sh0),
        sys.getsizeof(shN),
        sys.getsizeof(base),
    ])

    return container_size, array_sizes


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("GSData Container Type Performance Benchmark")
    print("=" * 80)
    print()

    # Create test data
    print("Creating test data (100K Gaussians)...")
    data = create_test_data(n_gaussians=100000)
    print()

    containers = [
        ("Regular Dataclass (current)", GSDataRegular),
        ("Frozen Dataclass", GSDataFrozen),
        ("NamedTuple", GSDataNamedTuple),
    ]

    # Benchmark 1: Creation time
    print("-" * 80)
    print("BENCHMARK 1: Container Creation Time")
    print("-" * 80)
    print(f"{'Container Type':<30} {'Total (ms)':<15} {'Avg (ns)':<15} {'Speedup':<10}")
    print("-" * 80)

    creation_results = []
    for name, container_class in containers:
        total_time, avg_time = benchmark_creation(container_class, data, n_iterations=10000)
        creation_results.append((name, total_time, avg_time))
        print(f"{name:<30} {total_time*1000:>14.3f} {avg_time*1e9:>14.1f} {'-':<10}")

    # Calculate speedups relative to regular dataclass
    baseline_time = creation_results[0][1]
    print()
    print("Speedup relative to Regular Dataclass:")
    for name, total_time, avg_time in creation_results:
        speedup = baseline_time / total_time
        print(f"  {name:<30} {speedup:>6.2f}x")
    print()

    # Benchmark 2: Attribute access
    print("-" * 80)
    print("BENCHMARK 2: Attribute Access Time")
    print("-" * 80)
    print(f"{'Container Type':<30} {'Total (ms)':<15} {'Avg (ns)':<15} {'Speedup':<10}")
    print("-" * 80)

    access_results = []
    for name, container_class in containers:
        total_time, avg_time = benchmark_attribute_access(container_class, data, n_iterations=1000000)
        access_results.append((name, total_time, avg_time))
        print(f"{name:<30} {total_time*1000:>14.3f} {avg_time*1e9:>14.1f} {'-':<10}")

    # Calculate speedups
    baseline_time = access_results[0][1]
    print()
    print("Speedup relative to Regular Dataclass:")
    for name, total_time, avg_time in access_results:
        speedup = baseline_time / total_time
        print(f"  {name:<30} {speedup:>6.2f}x")
    print()

    # Benchmark 3: Indexing (for NamedTuple)
    print("-" * 80)
    print("BENCHMARK 3: Indexing/Unpacking Performance")
    print("-" * 80)
    print(f"{'Container Type':<30} {'Total (ms)':<15} {'Avg (ns)':<15}")
    print("-" * 80)

    for name, container_class in containers:
        total_time, avg_time = benchmark_indexing(container_class, data, n_iterations=100000)
        if total_time > 0:
            print(f"{name:<30} {total_time*1000:>14.3f} {avg_time*1e9:>14.1f}")
        else:
            print(f"{name:<30} {'N/A (no indexing)':<30}")
    print()

    # Benchmark 4: Memory overhead
    print("-" * 80)
    print("BENCHMARK 4: Memory Overhead")
    print("-" * 80)
    print(f"{'Container Type':<30} {'Container (bytes)':<20} {'Arrays (MB)':<15}")
    print("-" * 80)

    for name, container_class in containers:
        container_size, array_sizes = measure_memory_size(container_class, data)
        print(f"{name:<30} {container_size:>19} {array_sizes/1e6:>14.2f}")
    print()

    # Summary and Recommendations
    print("=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print()

    print("Creation Speed:")
    fastest_creation = min(creation_results, key=lambda x: x[1])
    print(f"  Fastest: {fastest_creation[0]} ({baseline_time/fastest_creation[1]:.2f}x speedup)")
    print()

    print("Attribute Access Speed:")
    fastest_access = min(access_results, key=lambda x: x[1])
    baseline_access = access_results[0][1]
    print(f"  Fastest: {fastest_access[0]} ({baseline_access/fastest_access[1]:.2f}x speedup)")
    print()

    print("Trade-offs:")
    print("  - Regular Dataclass: Mutable, good balance, current implementation")
    print("  - Frozen Dataclass: Immutable (safer), similar performance")
    print("  - NamedTuple: Immutable, supports indexing, lightest weight")
    print()

    print("Recommendation:")
    if fastest_creation[0] == fastest_access[0]:
        print(f"  {fastest_creation[0]} offers best overall performance")
    else:
        print(f"  Creation: {fastest_creation[0]}")
        print(f"  Access: {fastest_access[0]}")
    print()


if __name__ == "__main__":
    main()
