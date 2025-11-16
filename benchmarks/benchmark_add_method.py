"""Benchmark for GSData and GSTensor add() method performance.

Tests:
1. GSData.add() - regular vs _base optimization
2. GSTensor.add() - CPU vs GPU
3. Scaling with data size
4. Impact of mask layers
"""

import time

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from gsply import GSData

if TORCH_AVAILABLE:
    from gsply.torch import GSTensor


def create_gsdata(n: int, sh_degree: int = 0) -> GSData:
    """Create GSData with specified size and SH degree."""
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None if sh_degree == 0 else np.random.rand(n, 3 * sh_degree, 3).astype(np.float32),
    )
    return data


def create_gsdata_with_base(n: int, sh_degree: int = 0) -> GSData:
    """Create GSData with _base array for zero-copy slicing.

    Creates data in the packed format that plyread uses, enabling _base optimization.
    """
    # Determine number of SH coefficients
    if sh_degree == 0:
        sh_coeffs = 0
        n_props = 14  # means(3) + sh0(3) + opacity(1) + scales(3) + quats(4)
    elif sh_degree == 1:
        sh_coeffs = 3
        n_props = 23  # 14 + 3*3
    elif sh_degree == 2:
        sh_coeffs = 8
        n_props = 38  # 14 + 8*3
    elif sh_degree == 3:
        sh_coeffs = 15
        n_props = 59  # 14 + 15*3
    else:
        raise ValueError(f"Invalid SH degree: {sh_degree}")

    # Create packed base array
    # Layout: means(3) + sh0(3) + shN(K*3) + opacity(1) + scales(3) + quats(4)
    base_array = np.random.randn(n, n_props).astype(np.float32)

    # Use _recreate_from_base to create GSData with proper _base reference
    return GSData._recreate_from_base(base_array)


def benchmark_gsdata_add(n: int, warmup: int = 3, iterations: int = 20) -> dict:
    """Benchmark GSData.add() with and without _base optimization."""
    results = {}

    # Test 1: Regular concatenation (no _base)
    data1 = create_gsdata(n)
    data2 = create_gsdata(n)

    # Warmup
    for _ in range(warmup):
        _ = data1.add(data2)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = data1.add(data2)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    throughput = (2 * n) / (avg_time / 1000) / 1e6  # Million Gaussians/sec
    results["regular"] = {"time_ms": avg_time, "throughput_M/s": throughput}

    # Test 2: With _base optimization
    data1_base = create_gsdata_with_base(n)
    data2_base = create_gsdata_with_base(n)

    # Verify _base exists
    assert data1_base._base is not None
    assert data2_base._base is not None

    # Warmup
    for _ in range(warmup):
        _ = data1_base.add(data2_base)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = data1_base.add(data2_base)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000
    throughput = (2 * n) / (avg_time / 1000) / 1e6
    results["with_base"] = {"time_ms": avg_time, "throughput_M/s": throughput}

    # Calculate speedup
    speedup = results["regular"]["time_ms"] / results["with_base"]["time_ms"]
    results["speedup"] = speedup

    return results


def benchmark_gstensor_add_cpu(n: int, warmup: int = 3, iterations: int = 20) -> dict:
    """Benchmark GSTensor.add() on CPU."""
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    data1 = create_gsdata(n)
    data2 = create_gsdata(n)

    gs1 = GSTensor.from_gsdata(data1, device="cpu")
    gs2 = GSTensor.from_gsdata(data2, device="cpu")

    # Warmup
    for _ in range(warmup):
        _ = gs1.add(gs2)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = gs1.add(gs2)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000
    throughput = (2 * n) / (avg_time / 1000) / 1e6
    return {"time_ms": avg_time, "throughput_M/s": throughput}


def benchmark_gstensor_add_gpu(n: int, warmup: int = 3, iterations: int = 20) -> dict:
    """Benchmark GSTensor.add() on GPU."""
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    if not CUDA_AVAILABLE:
        return {"error": "CUDA not available"}

    data1 = create_gsdata(n)
    data2 = create_gsdata(n)

    gs1 = GSTensor.from_gsdata(data1, device="cuda")
    gs2 = GSTensor.from_gsdata(data2, device="cuda")

    # Warmup
    for _ in range(warmup):
        _ = gs1.add(gs2)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = gs1.add(gs2)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000
    throughput = (2 * n) / (avg_time / 1000) / 1e6
    return {"time_ms": avg_time, "throughput_M/s": throughput}


def benchmark_with_masks(n: int, num_layers: int = 3) -> dict:
    """Benchmark add() with mask layers."""
    results = {}

    # GSData with masks
    data1 = create_gsdata(n)
    data2 = create_gsdata(n)

    for i in range(num_layers):
        data1.add_mask_layer(f"layer{i}", np.random.rand(n) > 0.5)
        data2.add_mask_layer(f"layer{i}", np.random.rand(n) > 0.5)

    times = []
    for _ in range(20):
        start = time.perf_counter()
        result = data1.add(data2)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000
    throughput = (2 * n) / (avg_time / 1000) / 1e6
    results["gsdata"] = {"time_ms": avg_time, "throughput_M/s": throughput}

    # GSTensor GPU with masks
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        gs1 = GSTensor.from_gsdata(create_gsdata(n), device="cuda")
        gs2 = GSTensor.from_gsdata(create_gsdata(n), device="cuda")

        for i in range(num_layers):
            gs1.add_mask_layer(f"layer{i}", torch.rand(n, device="cuda") > 0.5)
            gs2.add_mask_layer(f"layer{i}", torch.rand(n, device="cuda") > 0.5)

        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = gs1.add(gs2)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        avg_time = np.mean(times) * 1000
        throughput = (2 * n) / (avg_time / 1000) / 1e6
        results["gstensor_gpu"] = {"time_ms": avg_time, "throughput_M/s": throughput}

    return results


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("GSData and GSTensor add() Method Benchmarks")
    print("=" * 80)

    # Test 1: GSData add() - regular vs _base optimization
    print("\n[Test 1] GSData.add() - _base optimization impact")
    print("-" * 80)
    for n in [10_000, 100_000, 500_000]:
        print(f"\nSize: {n:,} Gaussians")
        results = benchmark_gsdata_add(n)
        print(f"  Regular:   {results['regular']['time_ms']:.3f} ms "
              f"({results['regular']['throughput_M/s']:.1f} M/s)")
        print(f"  With _base: {results['with_base']['time_ms']:.3f} ms "
              f"({results['with_base']['throughput_M/s']:.1f} M/s)")
        print(f"  Speedup:   {results['speedup']:.2f}x")

    # Test 2: GSTensor CPU vs GPU
    if TORCH_AVAILABLE:
        print("\n[Test 2] GSTensor.add() - CPU vs GPU")
        print("-" * 80)
        for n in [10_000, 100_000, 500_000]:
            print(f"\nSize: {n:,} Gaussians")

            cpu_results = benchmark_gstensor_add_cpu(n)
            print(f"  CPU: {cpu_results['time_ms']:.3f} ms "
                  f"({cpu_results['throughput_M/s']:.1f} M/s)")

            if CUDA_AVAILABLE:
                gpu_results = benchmark_gstensor_add_gpu(n)
                print(f"  GPU: {gpu_results['time_ms']:.3f} ms "
                      f"({gpu_results['throughput_M/s']:.1f} M/s)")
                speedup = cpu_results['time_ms'] / gpu_results['time_ms']
                print(f"  Speedup: {speedup:.2f}x")
            else:
                print("  GPU: CUDA not available")

    # Test 3: Impact of mask layers
    print("\n[Test 3] add() with mask layers (3 layers)")
    print("-" * 80)
    n = 100_000
    print(f"Size: {n:,} Gaussians")
    results = benchmark_with_masks(n)
    print(f"  GSData: {results['gsdata']['time_ms']:.3f} ms "
          f"({results['gsdata']['throughput_M/s']:.1f} M/s)")
    if "gstensor_gpu" in results:
        print(f"  GSTensor GPU: {results['gstensor_gpu']['time_ms']:.3f} ms "
              f"({results['gstensor_gpu']['throughput_M/s']:.1f} M/s)")
        speedup = results['gsdata']['time_ms'] / results['gstensor_gpu']['time_ms']
        print(f"  Speedup: {speedup:.2f}x")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
