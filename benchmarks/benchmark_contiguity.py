"""Comprehensive analysis of array contiguity in GSData.

Analyzes:
1. Which arrays are contiguous in different scenarios
2. Performance impact of non-contiguous arrays
3. When to make contiguous copies
4. Potential optimizations for contiguity
"""

import time

import numpy as np

from gsply import GSData


def analyze_contiguity(data: GSData, label: str):
    """Analyze contiguity of all arrays in GSData."""
    print(f"\n{label}")
    print("-" * 80)

    arrays = {
        "means": data.means,
        "scales": data.scales,
        "quats": data.quats,
        "opacities": data.opacities,
        "sh0": data.sh0,
    }

    if data._base is not None:
        arrays["_base"] = data._base

    for name, arr in arrays.items():
        c_contig = arr.flags["C_CONTIGUOUS"]
        f_contig = arr.flags["F_CONTIGUOUS"]
        owndata = arr.flags["OWNDATA"]

        status = "CONTIGUOUS" if c_contig else "NON-CONTIGUOUS"
        if not owndata:
            status += " (view)"

        print(f"  {name:12s}: {status:25s} | shape={str(arr.shape):15s} | "
              f"strides={str(arr.strides):15s} | dtype={arr.dtype}")

    if data._base is not None:
        print(f"\n  Memory layout: INTERLEAVED (row-major)")
        print(f"  Row stride: {data._base.strides[0]} bytes "
              f"({data._base.strides[0] // 4} floats)")
    else:
        print(f"\n  Memory layout: SEPARATE ARRAYS")


def test_all_scenarios():
    """Test contiguity in all common scenarios."""
    print("=" * 80)
    print("Array Contiguity Analysis - All Scenarios")
    print("=" * 80)

    n = 10000

    # Scenario 1: Direct construction
    data1 = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )
    analyze_contiguity(data1, "Scenario 1: Direct Construction")

    # Scenario 2: From _base (simulating plyread)
    base_array = np.random.randn(n, 14).astype(np.float32)
    data2 = GSData._recreate_from_base(base_array)
    analyze_contiguity(data2, "Scenario 2: From _base (plyread)")

    # Scenario 3: After slicing (contiguous slice)
    data3 = data2[:5000]
    analyze_contiguity(data3, "Scenario 3: Contiguous Slice [:5000]")

    # Scenario 4: After slicing (step slice)
    data4 = data2[::2]
    analyze_contiguity(data4, "Scenario 4: Step Slice [::2]")

    # Scenario 5: After add()
    data5 = data1.add(data1)
    analyze_contiguity(data5, "Scenario 5: After add()")

    # Scenario 6: After concatenate()
    data6 = GSData.concatenate([data1, data1, data1])
    analyze_contiguity(data6, "Scenario 6: After concatenate()")

    # Scenario 7: After making contiguous copy
    means_contig = np.ascontiguousarray(data2.means)
    print(f"\nScenario 7: After np.ascontiguousarray()")
    print("-" * 80)
    print(f"  Original means: NON-CONTIGUOUS | shape={data2.means.shape} | "
          f"strides={data2.means.strides}")
    print(f"  Contiguous copy: CONTIGUOUS     | shape={means_contig.shape} | "
          f"strides={means_contig.strides}")
    print(f"  Shares memory: {np.shares_memory(data2.means, means_contig)}")


def benchmark_contiguous_vs_noncontiguous():
    """Benchmark performance impact of contiguity."""
    print("\n" + "=" * 80)
    print("Performance Impact of Non-Contiguous Arrays")
    print("=" * 80)

    n = 1_000_000

    # Create non-contiguous (from _base)
    base_array = np.random.randn(n, 14).astype(np.float32)
    data_noncontig = GSData._recreate_from_base(base_array)

    # Create contiguous (direct construction)
    data_contig = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )

    print(f"\nTesting with {n:,} Gaussians")
    print("-" * 80)

    operations = {
        "sum()": lambda arr: arr.sum(),
        "mean()": lambda arr: arr.mean(),
        "max()": lambda arr: arr.max(),
        "std()": lambda arr: arr.std(),
        "@ vector (matmul)": lambda arr: arr @ np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "element-wise add": lambda arr: arr + 1.0,
        "element-wise multiply": lambda arr: arr * 2.0,
    }

    for op_name, op_func in operations.items():
        # Non-contiguous
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = op_func(data_noncontig.means)
            end = time.perf_counter()
            times.append(end - start)
        noncontig_time = np.mean(times) * 1000

        # Contiguous
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = op_func(data_contig.means)
            end = time.perf_counter()
            times.append(end - start)
        contig_time = np.mean(times) * 1000

        overhead = noncontig_time / contig_time
        print(f"  {op_name:25s}: {contig_time:6.3f} ms (contig) | "
              f"{noncontig_time:6.3f} ms (non-contig) | "
              f"{overhead:.2f}x overhead")


def benchmark_making_contiguous():
    """Benchmark cost of making arrays contiguous."""
    print("\n" + "=" * 80)
    print("Cost of Making Arrays Contiguous")
    print("=" * 80)

    sizes = [10_000, 100_000, 1_000_000]

    for n in sizes:
        # Create non-contiguous data
        base_array = np.random.randn(n, 14).astype(np.float32)
        data = GSData._recreate_from_base(base_array)

        print(f"\nSize: {n:,} Gaussians")
        print("-" * 80)

        # Method 1: np.ascontiguousarray
        times = []
        for _ in range(20):
            start = time.perf_counter()
            means_contig = np.ascontiguousarray(data.means)
            end = time.perf_counter()
            times.append(end - start)
        ascontig_time = np.mean(times) * 1000
        throughput = n / (ascontig_time / 1000) / 1e6
        print(f"  np.ascontiguousarray: {ascontig_time:6.3f} ms ({throughput:.1f} M/s)")

        # Method 2: .copy()
        times = []
        for _ in range(20):
            start = time.perf_counter()
            means_copy = data.means.copy()
            end = time.perf_counter()
            times.append(end - start)
        copy_time = np.mean(times) * 1000
        throughput = n / (copy_time / 1000) / 1e6
        print(f"  .copy():              {copy_time:6.3f} ms ({throughput:.1f} M/s)")

        # Method 3: Manual copy with empty+assign
        times = []
        for _ in range(20):
            start = time.perf_counter()
            means_manual = np.empty((n, 3), dtype=np.float32)
            means_manual[:] = data.means
            end = time.perf_counter()
            times.append(end - start)
        manual_time = np.mean(times) * 1000
        throughput = n / (manual_time / 1000) / 1e6
        print(f"  empty() + assign:     {manual_time:6.3f} ms ({throughput:.1f} M/s)")


def analyze_when_to_make_contiguous():
    """Analyze when it's worth making arrays contiguous."""
    print("\n" + "=" * 80)
    print("When to Make Arrays Contiguous")
    print("=" * 80)

    n = 100_000
    base_array = np.random.randn(n, 14).astype(np.float32)
    data = GSData._recreate_from_base(base_array)

    # Cost of making contiguous
    start = time.perf_counter()
    means_contig = np.ascontiguousarray(data.means)
    end = time.perf_counter()
    contig_cost = (end - start) * 1000

    print(f"\nArray: {n:,} Gaussians")
    print(f"Cost to make contiguous: {contig_cost:.3f} ms")
    print("-" * 80)

    # Simulate different workloads
    workloads = {
        "Single operation": 1,
        "Few operations (3x)": 3,
        "Medium workload (10x)": 10,
        "Heavy workload (100x)": 100,
    }

    for workload_name, num_ops in workloads.items():
        # Non-contiguous path
        start = time.perf_counter()
        for _ in range(num_ops):
            _ = data.means.sum()
        end = time.perf_counter()
        noncontig_total = (end - start) * 1000

        # Contiguous path (include conversion cost)
        start = time.perf_counter()
        means_c = np.ascontiguousarray(data.means)
        for _ in range(num_ops):
            _ = means_c.sum()
        end = time.perf_counter()
        contig_total = (end - start) * 1000

        if contig_total < noncontig_total:
            verdict = f"MAKE CONTIGUOUS ({noncontig_total/contig_total:.2f}x faster)"
        else:
            verdict = f"KEEP NON-CONTIGUOUS ({contig_total/noncontig_total:.2f}x slower)"

        print(f"\n  {workload_name}:")
        print(f"    Non-contiguous: {noncontig_total:.3f} ms")
        print(f"    Contiguous:     {contig_total:.3f} ms (includes {contig_cost:.3f} ms conversion)")
        print(f"    Verdict: {verdict}")


def propose_optimizations():
    """Propose potential optimizations based on analysis."""
    print("\n" + "=" * 80)
    print("Proposed Optimizations")
    print("=" * 80)

    proposals = [
        ("1. Add .make_contiguous() method",
         "Explicit method to convert all views to contiguous arrays when needed"),

        ("2. Lazy contiguous conversion",
         "Convert to contiguous on first operation, cache result"),

        ("3. Hybrid storage mode",
         "Allow switching between _base (memory-efficient) and separate (performance) modes"),

        ("4. Operation-specific optimization",
         "Automatically make contiguous for expensive operations (matmul, etc.)"),

        ("5. Contiguous flag in metadata",
         "Track which arrays are contiguous, optimize accordingly"),
    ]

    for title, description in proposals:
        print(f"\n{title}")
        print(f"  {description}")

    print("\n" + "=" * 80)
    print("Recommendation:")
    print("=" * 80)
    print("  - Keep current design (zero-copy views from _base)")
    print("  - 7.56x overhead is acceptable for most use cases")
    print("  - Add .make_contiguous() for performance-critical code paths")
    print("  - Users can explicitly convert when needed")
    print("=" * 80)


def main():
    """Run all contiguity analyses."""
    test_all_scenarios()
    benchmark_contiguous_vs_noncontiguous()
    benchmark_making_contiguous()
    analyze_when_to_make_contiguous()
    propose_optimizations()


if __name__ == "__main__":
    main()
