"""Benchmark GSData to GSTensor conversion performance.

Compares:
1. Current implementation (standard transfer)
2. Pinned memory approach
3. Different dataset sizes
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from gsply import GSData, plyread


def median_time_ms(func, n_trials=50, warmup=5):
    """Run function multiple times and return median time in ms."""
    # Warmup
    for _ in range(warmup):
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = func()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        # Cleanup
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.median(times), np.std(times), np.min(times), np.max(times)


def standard_transfer(data: GSData, device: str, dtype: torch.dtype):
    """Current implementation - standard transfer."""
    device_obj = torch.device(device)

    if data._base is not None:
        # Fast path
        base_tensor = torch.from_numpy(data._base).to(device=device_obj, dtype=dtype)
        return base_tensor
    else:
        # Fallback path
        n = len(data)
        if data.shN is not None and data.shN.shape[1] > 0:
            sh_coeffs = data.shN.shape[1]
            n_props = 14 + sh_coeffs * 3
        else:
            n_props = 14

        base_cpu = np.empty((n, n_props), dtype=np.float32)
        base_cpu[:, 0:3] = data.means
        base_cpu[:, 3:6] = data.sh0

        if data.shN is not None and sh_coeffs > 0:
            shN_flat = data.shN.reshape(n, sh_coeffs * 3)
            base_cpu[:, 6:6 + sh_coeffs * 3] = shN_flat
            opacity_idx = 6 + sh_coeffs * 3
        else:
            opacity_idx = 6

        base_cpu[:, opacity_idx] = data.opacities
        base_cpu[:, opacity_idx + 1:opacity_idx + 4] = data.scales
        base_cpu[:, opacity_idx + 4:opacity_idx + 8] = data.quats

        base_tensor = torch.from_numpy(base_cpu).to(device=device_obj, dtype=dtype)
        return base_tensor


def pinned_transfer(data: GSData, device: str, dtype: torch.dtype):
    """Proposed optimization - pinned memory transfer."""
    device_obj = torch.device(device)

    if data._base is not None:
        # Fast path with pinning
        if device_obj.type == 'cuda':
            base_cpu = torch.from_numpy(data._base)
            if base_cpu.dtype != dtype:
                base_cpu = base_cpu.to(dtype=dtype)
            base_tensor = base_cpu.pin_memory().to(device=device_obj, non_blocking=True)
        else:
            base_tensor = torch.from_numpy(data._base).to(device=device_obj, dtype=dtype)
        return base_tensor
    else:
        # Fallback path with pinning
        n = len(data)
        if data.shN is not None and data.shN.shape[1] > 0:
            sh_coeffs = data.shN.shape[1]
            n_props = 14 + sh_coeffs * 3
        else:
            n_props = 14

        base_cpu = np.empty((n, n_props), dtype=np.float32)
        base_cpu[:, 0:3] = data.means
        base_cpu[:, 3:6] = data.sh0

        if data.shN is not None and sh_coeffs > 0:
            shN_flat = data.shN.reshape(n, sh_coeffs * 3)
            base_cpu[:, 6:6 + sh_coeffs * 3] = shN_flat
            opacity_idx = 6 + sh_coeffs * 3
        else:
            opacity_idx = 6

        base_cpu[:, opacity_idx] = data.opacities
        base_cpu[:, opacity_idx + 1:opacity_idx + 4] = data.scales
        base_cpu[:, opacity_idx + 4:opacity_idx + 8] = data.quats

        if device_obj.type == 'cuda':
            base_cpu_tensor = torch.from_numpy(base_cpu)
            if base_cpu_tensor.dtype != dtype:
                base_cpu_tensor = base_cpu_tensor.to(dtype=dtype)
            base_tensor = base_cpu_tensor.pin_memory().to(device=device_obj, non_blocking=True)
        else:
            base_tensor = torch.from_numpy(base_cpu).to(device=device_obj, dtype=dtype)

        return base_tensor


def benchmark_file(file_path: Path, device: str = 'cuda', n_trials: int = 50):
    """Benchmark a specific PLY file."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {file_path.name}")
    print(f"{'='*80}")

    # Load data
    print("Loading data...")
    data = plyread(str(file_path))
    n_gaussians = len(data)
    has_base = data._base is not None

    if data.shN is not None and data.shN.shape[1] > 0:
        sh_degree = 3 if data.shN.shape[1] == 15 else (2 if data.shN.shape[1] == 8 else 1)
    else:
        sh_degree = 0

    print(f"  Gaussians: {n_gaussians:,}")
    print(f"  SH degree: {sh_degree}")
    print(f"  Has _base: {has_base}")

    if data._base is not None:
        mem_mb = data._base.nbytes / 1024 / 1024
        print(f"  Memory size: {mem_mb:.2f} MB")

    dtype = torch.float32

    # Benchmark fast path (with _base)
    if has_base:
        print(f"\n--- FAST PATH (with _base) ---")

        print(f"Running standard transfer ({n_trials} trials)...")
        median_std, std_std, min_std, max_std = median_time_ms(
            lambda: standard_transfer(data, device, dtype),
            n_trials=n_trials
        )

        print(f"Running pinned memory transfer ({n_trials} trials)...")
        median_pin, std_pin, min_pin, max_pin = median_time_ms(
            lambda: pinned_transfer(data, device, dtype),
            n_trials=n_trials
        )

        print(f"\nResults:")
        print(f"  Standard:     {median_std:6.2f} ms  (std: {std_std:5.2f}, min: {min_std:6.2f}, max: {max_std:6.2f})")
        print(f"  Pinned:       {median_pin:6.2f} ms  (std: {std_pin:5.2f}, min: {min_pin:6.2f}, max: {max_pin:6.2f})")

        if median_pin < median_std:
            speedup = median_std / median_pin
            print(f"  Speedup:      {speedup:.2f}x FASTER with pinned memory")
        else:
            slowdown = median_pin / median_std
            print(f"  Slowdown:     {slowdown:.2f}x SLOWER with pinned memory")

        throughput_std = n_gaussians / median_std * 1000 / 1e6
        throughput_pin = n_gaussians / median_pin * 1000 / 1e6
        print(f"  Throughput (standard): {throughput_std:.2f} M Gaussians/sec")
        print(f"  Throughput (pinned):   {throughput_pin:.2f} M Gaussians/sec")

    # Benchmark fallback path (without _base)
    print(f"\n--- FALLBACK PATH (without _base) ---")

    # Create data without _base
    data_no_base = GSData(
        means=data.means.copy(),
        scales=data.scales.copy(),
        quats=data.quats.copy(),
        opacities=data.opacities.copy(),
        sh0=data.sh0.copy(),
        shN=data.shN.copy() if data.shN is not None else None,
        masks=data.masks.copy() if data.masks is not None else None,
        _base=None
    )

    print(f"Running standard transfer ({n_trials} trials)...")
    median_std, std_std, min_std, max_std = median_time_ms(
        lambda: standard_transfer(data_no_base, device, dtype),
        n_trials=n_trials
    )

    print(f"Running pinned memory transfer ({n_trials} trials)...")
    median_pin, std_pin, min_pin, max_pin = median_time_ms(
        lambda: pinned_transfer(data_no_base, device, dtype),
        n_trials=n_trials
    )

    print(f"\nResults:")
    print(f"  Standard:     {median_std:6.2f} ms  (std: {std_std:5.2f}, min: {min_std:6.2f}, max: {max_std:6.2f})")
    print(f"  Pinned:       {median_pin:6.2f} ms  (std: {std_pin:5.2f}, min: {min_pin:6.2f}, max: {max_pin:6.2f})")

    if median_pin < median_std:
        speedup = median_std / median_pin
        print(f"  Speedup:      {speedup:.2f}x FASTER with pinned memory")
    else:
        slowdown = median_pin / median_std
        print(f"  Slowdown:     {slowdown:.2f}x SLOWER with pinned memory")

    throughput_std = n_gaussians / median_std * 1000 / 1e6
    throughput_pin = n_gaussians / median_pin * 1000 / 1e6
    print(f"  Throughput (standard): {throughput_std:.2f} M Gaussians/sec")
    print(f"  Throughput (pinned):   {throughput_pin:.2f} M Gaussians/sec")


def benchmark_synthetic(sizes: list[int], device: str = 'cuda', n_trials: int = 50):
    """Benchmark with synthetic data of various sizes."""
    print(f"\n{'='*80}")
    print("Benchmarking Synthetic Data (SH3)")
    print(f"{'='*80}")

    dtype = torch.float32

    results = []

    for n in sizes:
        print(f"\n--- Dataset size: {n:,} Gaussians ---")

        # Create synthetic SH3 data
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=np.random.randn(n, 15, 3).astype(np.float32),  # SH3
            masks=np.ones(n, dtype=bool),
            _base=None
        )

        # Consolidate to create _base
        data = data.consolidate()

        mem_mb = data._base.nbytes / 1024 / 1024
        print(f"  Memory size: {mem_mb:.2f} MB")

        print(f"  Standard transfer...")
        median_std, _, _, _ = median_time_ms(
            lambda: standard_transfer(data, device, dtype),
            n_trials=n_trials,
            warmup=3
        )

        print(f"  Pinned transfer...")
        median_pin, _, _, _ = median_time_ms(
            lambda: pinned_transfer(data, device, dtype),
            n_trials=n_trials,
            warmup=3
        )

        ratio = median_pin / median_std

        print(f"  Standard: {median_std:6.2f} ms")
        print(f"  Pinned:   {median_pin:6.2f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        results.append({
            'size': n,
            'mem_mb': mem_mb,
            'standard': median_std,
            'pinned': median_pin,
            'ratio': ratio
        })

        # Cleanup
        del data
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Size':>10} | {'Memory':>10} | {'Standard':>10} | {'Pinned':>10} | {'Ratio':>8}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for r in results:
        size_str = f"{r['size']:,}"
        mem_str = f"{r['mem_mb']:.1f} MB"
        std_str = f"{r['standard']:.2f} ms"
        pin_str = f"{r['pinned']:.2f} ms"
        ratio_str = f"{r['ratio']:.2f}x"

        print(f"{size_str:>10} | {mem_str:>10} | {std_str:>10} | {pin_str:>10} | {ratio_str:>8}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark GSData to GSTensor conversion')
    parser.add_argument('--file', type=str, help='Path to PLY file')
    parser.add_argument('--synthetic', action='store_true', help='Run synthetic benchmarks')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            return
        benchmark_file(file_path, args.device, args.trials)

    if args.synthetic:
        sizes = [10000, 50000, 100000, 200000, 500000, 1000000]
        benchmark_synthetic(sizes, args.device, args.trials)

    if not args.file and not args.synthetic:
        print("ERROR: Specify --file or --synthetic")


if __name__ == '__main__':
    main()
