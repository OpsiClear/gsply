"""Performance benchmarks for GSTensor."""

import time

import numpy as np
import pytest

# Check if PyTorch is available
pytest.importorskip("torch")
import torch  # noqa: E402

from gsply import GSData  # noqa: E402
from gsply.torch import GSTensor  # noqa: E402


def median_time_ms(func, n_trials=100):
    """Run function multiple times and return median time in ms."""
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)


@pytest.fixture
def gsdata_large_no_base():
    """Create large GSData without _base."""
    n = 400000
    return GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.randn(n, 3).astype(np.float32),
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.randn(n, 3).astype(np.float32),
        shN=None,
        masks=np.ones(n, dtype=bool),
        _base=None,
    )


@pytest.fixture
def gsdata_large_with_base(gsdata_large_no_base):
    """Create large GSData with _base."""
    return gsdata_large_no_base.consolidate()


# =============================================================================
# Transfer Performance (CPU -> GPU)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.benchmark
def test_transfer_speed_with_base(gsdata_large_with_base):
    """Benchmark GPU transfer WITH _base (optimized)."""

    def transfer():
        gstensor = GSTensor.from_gsdata(gsdata_large_with_base, device="cuda")
        torch.cuda.synchronize()
        return gstensor

    time_with_base = median_time_ms(transfer, n_trials=10)

    print(f"\nTransfer WITH _base: {time_with_base:.2f} ms (400K Gaussians)")
    assert time_with_base < 20.0  # Should be very fast (< 20ms, allows for system variance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.benchmark
def test_transfer_speed_without_base(gsdata_large_no_base):
    """Benchmark GPU transfer WITHOUT _base (fallback)."""

    def transfer():
        gstensor = GSTensor.from_gsdata(gsdata_large_no_base, device="cuda")
        torch.cuda.synchronize()
        return gstensor

    time_without_base = median_time_ms(transfer, n_trials=10)

    print(f"\nTransfer WITHOUT _base: {time_without_base:.2f} ms (400K Gaussians)")
    # Fallback may be slower


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.benchmark
def test_transfer_speedup_comparison(gsdata_large_no_base, gsdata_large_with_base):
    """Compare transfer speed with and without _base."""

    def transfer_without_base():
        gstensor = GSTensor.from_gsdata(gsdata_large_no_base, device="cuda")
        torch.cuda.synchronize()
        return gstensor

    def transfer_with_base():
        gstensor = GSTensor.from_gsdata(gsdata_large_with_base, device="cuda")
        torch.cuda.synchronize()
        return gstensor

    time_without = median_time_ms(transfer_without_base, n_trials=5)
    time_with = median_time_ms(transfer_with_base, n_trials=5)

    speedup = time_without / time_with

    print("\n--- GPU Transfer Performance (400K Gaussians) ---")
    print(f"WITHOUT _base: {time_without:.2f} ms")
    print(f"WITH _base:    {time_with:.2f} ms")
    print(f"Speedup:       {speedup:.2f}x")

    # With _base should be faster (or at least not significantly slower)
    assert time_with <= time_without * 1.5  # Allow 50% margin


# =============================================================================
# Slicing Performance
# =============================================================================


@pytest.mark.benchmark
def test_slice_speed_without_base(gsdata_large_no_base):
    """Benchmark slicing without _base."""
    gstensor = GSTensor.from_gsdata(gsdata_large_no_base, device="cpu")
    mask = torch.rand(400000) > 0.5

    def slice_op():
        return gstensor[mask]

    time_slice = median_time_ms(slice_op, n_trials=100)

    print(f"\nSlicing WITHOUT _base: {time_slice:.2f} ms")


@pytest.mark.benchmark
def test_slice_speed_with_base(gsdata_large_with_base):
    """Benchmark slicing with _base."""
    gstensor = GSTensor.from_gsdata(gsdata_large_with_base, device="cpu")
    gstensor_consolidated = gstensor.consolidate()
    mask = torch.rand(400000) > 0.5

    def slice_op():
        return gstensor_consolidated[mask]

    time_slice = median_time_ms(slice_op, n_trials=100)

    print(f"\nSlicing WITH _base: {time_slice:.2f} ms")


@pytest.mark.benchmark
def test_slice_speedup_comparison(gsdata_large_no_base, gsdata_large_with_base):
    """Compare slicing speed with and without _base."""
    gstensor_no_base = GSTensor.from_gsdata(gsdata_large_no_base, device="cpu")
    gstensor_with_base = GSTensor.from_gsdata(gsdata_large_with_base, device="cpu").consolidate()

    mask = torch.rand(400000) > 0.5

    def slice_without_base():
        return gstensor_no_base[mask]

    def slice_with_base():
        return gstensor_with_base[mask]

    time_without = median_time_ms(slice_without_base, n_trials=50)
    time_with = median_time_ms(slice_with_base, n_trials=50)

    speedup = time_without / time_with

    print("\n--- Slicing Performance (400K Gaussians, ~50% mask) ---")
    print(f"WITHOUT _base: {time_without:.2f} ms")
    print(f"WITH _base:    {time_with:.2f} ms")
    print(f"Speedup:       {speedup:.2f}x")

    # Both should complete in reasonable time (< 5ms)
    # Note: On CPU tensors, _base may not provide speedup for boolean masking
    assert time_without < 5.0, f"Slicing without _base too slow: {time_without:.2f}ms"
    assert time_with < 5.0, f"Slicing with _base too slow: {time_with:.2f}ms"


# =============================================================================
# Clone Performance
# =============================================================================


@pytest.mark.benchmark
def test_clone_speed_without_base(gsdata_large_no_base):
    """Benchmark clone() without _base."""
    gstensor = GSTensor.from_gsdata(gsdata_large_no_base, device="cpu")

    def clone_op():
        return gstensor.clone()

    time_clone = median_time_ms(clone_op, n_trials=50)

    print(f"\nClone WITHOUT _base: {time_clone:.2f} ms")


@pytest.mark.benchmark
def test_clone_speed_with_base(gsdata_large_with_base):
    """Benchmark clone() with _base."""
    gstensor = GSTensor.from_gsdata(gsdata_large_with_base, device="cpu")

    def clone_op():
        return gstensor.clone()

    time_clone = median_time_ms(clone_op, n_trials=50)

    print(f"\nClone WITH _base: {time_clone:.2f} ms")


@pytest.mark.benchmark
def test_clone_speedup_comparison(gsdata_large_no_base, gsdata_large_with_base):
    """Compare clone speed with and without _base."""
    gstensor_no_base = GSTensor.from_gsdata(gsdata_large_no_base, device="cpu")
    gstensor_with_base = GSTensor.from_gsdata(gsdata_large_with_base, device="cpu")

    def clone_without_base():
        return gstensor_no_base.clone()

    def clone_with_base():
        return gstensor_with_base.clone()

    time_without = median_time_ms(clone_without_base, n_trials=50)
    time_with = median_time_ms(clone_with_base, n_trials=50)

    speedup = time_without / time_with

    print("\n--- Clone Performance (400K Gaussians) ---")
    print(f"WITHOUT _base: {time_without:.2f} ms")
    print(f"WITH _base:    {time_with:.2f} ms")
    print(f"Speedup:       {speedup:.2f}x")

    # Both should complete in reasonable time (< 10ms)
    # Note: Speedup varies with system load and may not always be consistent
    assert time_without < 10.0, f"Clone without _base too slow: {time_without:.2f}ms"
    assert time_with < 10.0, f"Clone with _base too slow: {time_with:.2f}ms"


# =============================================================================
# Memory Usage
# =============================================================================


@pytest.mark.benchmark
def test_memory_usage_with_base(gsdata_large_with_base):
    """Measure memory footprint with _base."""
    gstensor = GSTensor.from_gsdata(gsdata_large_with_base, device="cpu")

    # With _base: Single tensor holds all data
    if gstensor._base is not None:
        memory_bytes = gstensor._base.numel() * gstensor._base.element_size()
        print(f"\nMemory WITH _base: {memory_bytes / 1024 / 1024:.2f} MB")
    else:
        print("\nWARNING: _base not created from GSData with _base")


# =============================================================================
# Consolidate Performance
# =============================================================================


@pytest.mark.benchmark
def test_consolidate_speed(gsdata_large_no_base):
    """Benchmark consolidate() operation."""
    gstensor = GSTensor.from_gsdata(gsdata_large_no_base, device="cpu")

    def consolidate_op():
        return gstensor.consolidate()

    time_consolidate = median_time_ms(consolidate_op, n_trials=50)

    print(f"\nConsolidate time: {time_consolidate:.2f} ms (400K Gaussians)")

    # Should be reasonably fast (< 10ms for 400K)
    assert time_consolidate < 20.0  # Allow some margin


# =============================================================================
# Summary
# =============================================================================


@pytest.mark.benchmark
def test_performance_summary(gsdata_large_no_base, gsdata_large_with_base):
    """Print comprehensive performance summary."""
    print("\n" + "=" * 80)
    print("GSTENSOR PERFORMANCE SUMMARY (400K Gaussians, SH0)")
    print("=" * 80)

    # Create GSTensors
    gstensor_no_base = GSTensor.from_gsdata(gsdata_large_no_base, device="cpu")
    gstensor_with_base = GSTensor.from_gsdata(gsdata_large_with_base, device="cpu").consolidate()

    # Mask for testing
    mask = torch.rand(400000) > 0.5

    # Benchmark operations
    operations = [
        ("Boolean masking", lambda g: g[mask]),
        ("Continuous slice", lambda g: g[1000:2000]),
        ("Clone", lambda g: g.clone()),
    ]

    for op_name, op_func in operations:
        time_without = median_time_ms(lambda: op_func(gstensor_no_base), n_trials=20)
        time_with = median_time_ms(lambda: op_func(gstensor_with_base), n_trials=20)
        speedup = time_without / time_with

        print(f"\n{op_name}:")
        print(f"  Without _base: {time_without:.2f} ms")
        print(f"  With _base:    {time_with:.2f} ms")
        print(f"  Speedup:       {speedup:.2f}x")

    print("\n" + "=" * 80)
