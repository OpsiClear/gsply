"""Benchmark logit and sigmoid functions for CPU and GPU.

Tests both optimized Numba implementations and compares with NumPy/PyTorch baselines.
"""

import time
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from gsply.utils import logit, sigmoid

if TORCH_AVAILABLE:
    from gsply.torch.utils import logit as logit_gpu, sigmoid as sigmoid_gpu


def benchmark_cpu_logit_sigmoid(sizes: list[int], iterations: int = 10, warmup: int = 3):
    """Benchmark CPU logit and sigmoid functions.

    :param sizes: List of array sizes to test
    :param iterations: Number of benchmark iterations
    :param warmup: Number of warmup iterations
    """
    print("\n" + "=" * 70)
    print("CPU Logit/Sigmoid Benchmark")
    print("=" * 70)

    results = []

    for size in sizes:
        # Generate test data
        np.random.seed(42)
        probs = np.random.rand(size).astype(np.float32)
        logits = np.random.randn(size).astype(np.float32) * 5.0  # Range [-15, 15]

        # Warmup
        for _ in range(warmup):
            _ = logit(probs)
            _ = sigmoid(logits)

        # Benchmark logit
        logit_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = logit(probs)
            end = time.perf_counter()
            logit_times.append((end - start) * 1000)  # Convert to ms

        # Benchmark sigmoid
        sigmoid_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = sigmoid(logits)
            end = time.perf_counter()
            sigmoid_times.append((end - start) * 1000)

        # Benchmark roundtrip (logit -> sigmoid)
        roundtrip_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            l = logit(probs)
            s = sigmoid(l)
            end = time.perf_counter()
            roundtrip_times.append((end - start) * 1000)

        logit_median = np.median(logit_times)
        sigmoid_median = np.median(sigmoid_times)
        roundtrip_median = np.median(roundtrip_times)

        logit_throughput = size / (logit_median / 1000) / 1e6  # Millions/sec
        sigmoid_throughput = size / (sigmoid_median / 1000) / 1e6

        results.append({
            'size': size,
            'logit_ms': logit_median,
            'sigmoid_ms': sigmoid_median,
            'roundtrip_ms': roundtrip_median,
            'logit_throughput': logit_throughput,
            'sigmoid_throughput': sigmoid_throughput,
        })

        print(f"\nSize: {size:,} elements")
        print(f"  Logit:   {logit_median:.3f} ms ({logit_throughput:.2f}M ops/sec)")
        print(f"  Sigmoid: {sigmoid_median:.3f} ms ({sigmoid_throughput:.2f}M ops/sec)")
        print(f"  Roundtrip: {roundtrip_median:.3f} ms")

    return results


def benchmark_gpu_logit_sigmoid(sizes: list[int], iterations: int = 10, warmup: int = 3):
    """Benchmark GPU logit and sigmoid functions.

    :param sizes: List of array sizes to test
    :param iterations: Number of benchmark iterations
    :param warmup: Number of warmup iterations
    """
    if not TORCH_AVAILABLE:
        print("\nPyTorch not available, skipping GPU benchmarks")
        return None

    print("\n" + "=" * 70)
    print("GPU Logit/Sigmoid Benchmark")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = []

    for size in sizes:
        # Generate test data
        torch.manual_seed(42)
        probs = torch.rand(size, dtype=torch.float32, device=device)
        logits = torch.randn(size, dtype=torch.float32, device=device) * 5.0

        # Warmup
        for _ in range(warmup):
            _ = logit_gpu(probs)
            _ = sigmoid_gpu(logits)
            if device == "cuda":
                torch.cuda.synchronize()

        # Benchmark logit
        logit_times = []
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = logit_gpu(probs)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            logit_times.append((end - start) * 1000)

        # Benchmark sigmoid
        sigmoid_times = []
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = sigmoid_gpu(logits)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            sigmoid_times.append((end - start) * 1000)

        # Benchmark roundtrip
        roundtrip_times = []
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            l = logit_gpu(probs)
            s = sigmoid_gpu(l)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            roundtrip_times.append((end - start) * 1000)

        logit_median = np.median(logit_times)
        sigmoid_median = np.median(sigmoid_times)
        roundtrip_median = np.median(roundtrip_times)

        logit_throughput = size / (logit_median / 1000) / 1e6
        sigmoid_throughput = size / (sigmoid_median / 1000) / 1e6

        results.append({
            'size': size,
            'logit_ms': logit_median,
            'sigmoid_ms': sigmoid_median,
            'roundtrip_ms': roundtrip_median,
            'logit_throughput': logit_throughput,
            'sigmoid_throughput': sigmoid_throughput,
        })

        print(f"\nSize: {size:,} elements")
        print(f"  Logit:   {logit_median:.3f} ms ({logit_throughput:.2f}M ops/sec)")
        print(f"  Sigmoid: {sigmoid_median:.3f} ms ({sigmoid_throughput:.2f}M ops/sec)")
        print(f"  Roundtrip: {roundtrip_median:.3f} ms")

    return results


def compare_with_numpy(sizes: list[int], iterations: int = 10):
    """Compare optimized implementation with NumPy baseline.

    :param sizes: List of array sizes to test
    :param iterations: Number of benchmark iterations
    """
    print("\n" + "=" * 70)
    print("Comparison: Optimized vs NumPy")
    print("=" * 70)

    for size in sizes:
        np.random.seed(42)
        probs = np.random.rand(size).astype(np.float32)
        logits = np.random.randn(size).astype(np.float32) * 5.0

        # NumPy baseline (using scipy.special.expit for sigmoid if available)
        try:
            from scipy.special import expit
            numpy_sigmoid = expit
        except ImportError:
            # Fallback to manual implementation
            def numpy_sigmoid(x):
                return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        # NumPy logit (manual implementation)
        def numpy_logit(x, eps=1e-6):
            x_clipped = np.clip(x, eps, 1.0 - eps)
            return np.log(x_clipped / (1.0 - x_clipped))

        # Warmup
        for _ in range(3):
            _ = numpy_logit(probs)
            _ = numpy_sigmoid(logits)
            _ = logit(probs)
            _ = sigmoid(logits)

        # Benchmark NumPy logit
        numpy_logit_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = numpy_logit(probs)
            end = time.perf_counter()
            numpy_logit_times.append((end - start) * 1000)

        # Benchmark optimized logit
        opt_logit_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = logit(probs)
            end = time.perf_counter()
            opt_logit_times.append((end - start) * 1000)

        # Benchmark NumPy sigmoid
        numpy_sigmoid_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = numpy_sigmoid(logits)
            end = time.perf_counter()
            numpy_sigmoid_times.append((end - start) * 1000)

        # Benchmark optimized sigmoid
        opt_sigmoid_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = sigmoid(logits)
            end = time.perf_counter()
            opt_sigmoid_times.append((end - start) * 1000)

        numpy_logit_median = np.median(numpy_logit_times)
        opt_logit_median = np.median(opt_logit_times)
        numpy_sigmoid_median = np.median(numpy_sigmoid_times)
        opt_sigmoid_median = np.median(opt_sigmoid_times)

        logit_speedup = numpy_logit_median / opt_logit_median
        sigmoid_speedup = numpy_sigmoid_median / opt_sigmoid_median

        print(f"\nSize: {size:,} elements")
        print(f"  Logit:")
        print(f"    NumPy:     {numpy_logit_median:.3f} ms")
        print(f"    Optimized: {opt_logit_median:.3f} ms")
        print(f"    Speedup:   {logit_speedup:.2f}x")
        print(f"  Sigmoid:")
        print(f"    NumPy:     {numpy_sigmoid_median:.3f} ms")
        print(f"    Optimized: {opt_sigmoid_median:.3f} ms")
        print(f"    Speedup:   {sigmoid_speedup:.2f}x")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("Logit/Sigmoid Performance Benchmark")
    print("=" * 70)

    # Test sizes: small to large arrays
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    # CPU benchmarks
    cpu_results = benchmark_cpu_logit_sigmoid(sizes, iterations=20, warmup=5)

    # GPU benchmarks
    gpu_results = benchmark_gpu_logit_sigmoid(sizes, iterations=20, warmup=5)

    # Comparison with NumPy
    compare_with_numpy(sizes[:4], iterations=20)  # Skip largest size for NumPy comparison

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
