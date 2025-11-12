"""Benchmark compressed decompression vectorization speedup.

This script measures the performance improvement from vectorizing the
compressed PLY decompression compared to the original Python loop implementation.
"""

import numpy as np
import time

# Note: We don't have a compressed PLY file readily available, so this benchmark
# demonstrates the theoretical speedup by measuring the core operations.

def benchmark_array_operations():
    """Benchmark the key vectorized operations."""

    num_vertices = 50375
    num_chunks = (num_vertices + 255) // 256
    iterations = 20

    print("=" * 80)
    print("VECTORIZATION SPEEDUP BENCHMARK")
    print("=" * 80)
    print("\nTest configuration:")
    print(f"  Vertices: {num_vertices:,}")
    print(f"  Chunks: {num_chunks}")
    print(f"  Iterations: {iterations}")
    print()

    # Simulate packed data
    # Use np.random.randint with proper uint32 range or use random bytes
    # randint upper bound is exclusive, so 2**32 would overflow
    packed_position = np.random.randint(0, 2**31, size=num_vertices, dtype=np.uint32)
    packed_rotation = np.random.randint(0, 2**31, size=num_vertices, dtype=np.uint32)
    packed_scale = np.random.randint(0, 2**31, size=num_vertices, dtype=np.uint32)
    packed_color = np.random.randint(0, 2**31, size=num_vertices, dtype=np.uint32)

    # Simulate chunk bounds
    min_x = np.random.rand(num_chunks).astype(np.float32)
    max_x = min_x + np.random.rand(num_chunks).astype(np.float32)
    min_y = np.random.rand(num_chunks).astype(np.float32)
    max_y = min_y + np.random.rand(num_chunks).astype(np.float32)
    min_z = np.random.rand(num_chunks).astype(np.float32)
    max_z = min_z + np.random.rand(num_chunks).astype(np.float32)

    min_scale_x = np.random.rand(num_chunks).astype(np.float32) * 0.01
    max_scale_x = min_scale_x + np.random.rand(num_chunks).astype(np.float32) * 0.1
    min_scale_y = np.random.rand(num_chunks).astype(np.float32) * 0.01
    max_scale_y = min_scale_y + np.random.rand(num_chunks).astype(np.float32) * 0.1
    min_scale_z = np.random.rand(num_chunks).astype(np.float32) * 0.01
    max_scale_z = min_scale_z + np.random.rand(num_chunks).astype(np.float32) * 0.1

    min_r = np.random.rand(num_chunks).astype(np.float32)
    max_r = np.random.rand(num_chunks).astype(np.float32)
    min_g = np.random.rand(num_chunks).astype(np.float32)
    max_g = np.random.rand(num_chunks).astype(np.float32)
    min_b = np.random.rand(num_chunks).astype(np.float32)
    max_b = np.random.rand(num_chunks).astype(np.float32)

    CHUNK_SIZE = 256
    SH_C0 = 0.28209479177387814

    # ========================================================================
    # ORIGINAL PYTHON LOOP (Simulated)
    # ========================================================================

    print("Testing ORIGINAL (Python loop)...")
    times_original = []

    for _ in range(iterations):
        means = np.zeros((num_vertices, 3), dtype=np.float32)
        scales = np.zeros((num_vertices, 3), dtype=np.float32)
        sh0 = np.zeros((num_vertices, 3), dtype=np.float32)
        opacities = np.zeros(num_vertices, dtype=np.float32)

        t0 = time.perf_counter()

        # Simulate the Python loop overhead (just unpacking, not full processing)
        for i in range(num_vertices):
            chunk_idx = i // CHUNK_SIZE

            # Unpack position
            packed = int(packed_position[i])
            px = ((packed >> 21) & 0x7FF) / 2047.0
            py = ((packed >> 11) & 0x3FF) / 1023.0
            pz = (packed & 0x7FF) / 2047.0

            # Dequantize
            means[i, 0] = min_x[chunk_idx] + px * (max_x[chunk_idx] - min_x[chunk_idx])
            means[i, 1] = min_y[chunk_idx] + py * (max_y[chunk_idx] - min_y[chunk_idx])
            means[i, 2] = min_z[chunk_idx] + pz * (max_z[chunk_idx] - min_z[chunk_idx])

        t1 = time.perf_counter()
        times_original.append((t1 - t0) * 1000)

    mean_original = np.mean(times_original)
    std_original = np.std(times_original)

    print(f"  Time: {mean_original:.2f}ms ± {std_original:.2f}ms")
    print()

    # ========================================================================
    # VECTORIZED NUMPY
    # ========================================================================

    print("Testing VECTORIZED (NumPy)...")
    times_vectorized = []

    for _ in range(iterations):
        means = np.zeros((num_vertices, 3), dtype=np.float32)
        scales = np.zeros((num_vertices, 3), dtype=np.float32)
        sh0 = np.zeros((num_vertices, 3), dtype=np.float32)
        opacities = np.zeros(num_vertices, dtype=np.float32)

        t0 = time.perf_counter()

        # Pre-compute chunk indices (vectorized)
        chunk_indices = np.arange(num_vertices, dtype=np.int32) // CHUNK_SIZE

        # Vectorized position unpacking
        px = ((packed_position >> 21) & 0x7FF).astype(np.float32) / 2047.0
        py = ((packed_position >> 11) & 0x3FF).astype(np.float32) / 1023.0
        pz = (packed_position & 0x7FF).astype(np.float32) / 2047.0

        # Vectorized position dequantization
        means[:, 0] = min_x[chunk_indices] + px * (max_x[chunk_indices] - min_x[chunk_indices])
        means[:, 1] = min_y[chunk_indices] + py * (max_y[chunk_indices] - min_y[chunk_indices])
        means[:, 2] = min_z[chunk_indices] + pz * (max_z[chunk_indices] - min_z[chunk_indices])

        t1 = time.perf_counter()
        times_vectorized.append((t1 - t0) * 1000)

    mean_vectorized = np.mean(times_vectorized)
    std_vectorized = np.std(times_vectorized)

    print(f"  Time: {mean_vectorized:.2f}ms ± {std_vectorized:.2f}ms")
    print()

    # ========================================================================
    # RESULTS
    # ========================================================================

    speedup = mean_original / mean_vectorized
    improvement_pct = (mean_original - mean_vectorized) / mean_original * 100

    print("=" * 80)
    print("RESULTS (Position unpacking + dequantization only)")
    print("=" * 80)
    print(f"\nOriginal (Python loop):  {mean_original:.2f}ms")
    print(f"Vectorized (NumPy):      {mean_vectorized:.2f}ms")
    print(f"\nSpeedup:                 {speedup:.1f}x faster")
    print(f"Improvement:             {improvement_pct:.1f}% faster")
    print()
    print("Note: This benchmarks only position operations.")
    print("Full decompression includes scales, colors, quaternions, and SH coefficients.")
    print("Expected full decompression speedup: 5-10x")
    print()

if __name__ == "__main__":
    benchmark_array_operations()
