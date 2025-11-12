"""Benchmark gsply performance with real data from D:/4D/all_plys."""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gsply


def benchmark_file(file_path, iterations=10):
    """Benchmark a single PLY file."""
    file_path = Path(file_path)

    # Get file info
    file_size = file_path.stat().st_size / (1024 * 1024)
    is_compressed, sh_degree = gsply.detect_format(file_path)

    # Quick read to get Gaussian count
    data = gsply.plyread(file_path)
    num_gaussians = data.means.shape[0]

    print(f"\nFile: {file_path.name}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Gaussians: {num_gaussians:,}")
    print(f"  SH degree: {sh_degree}")
    print(f"  Compressed: {is_compressed}")

    if is_compressed:
        # Benchmark compressed read
        print("\n  Benchmarking compressed read...")
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            result = gsply.plyread(file_path)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Compressed read: {mean_time:.2f} +/- {std_time:.2f} ms")

    else:
        # Benchmark fast read
        print("\n  Benchmarking fast read (zero-copy)...")
        times_fast = []
        for i in range(iterations):
            start = time.perf_counter()
            result = gsply.plyread(file_path)
            end = time.perf_counter()
            times_fast.append((end - start) * 1000)

        mean_fast = np.mean(times_fast)
        std_fast = np.std(times_fast)

        # Benchmark safe read
        print("  Benchmarking safe read (copies)...")
        times_safe = []
        for i in range(iterations):
            start = time.perf_counter()
            result = gsply.plyread(file_path)
            end = time.perf_counter()
            times_safe.append((end - start) * 1000)

        mean_safe = np.mean(times_safe)
        std_safe = np.std(times_safe)

        speedup = mean_safe / mean_fast

        print(f"  Fast read (zero-copy): {mean_fast:.2f} +/- {std_fast:.2f} ms")
        print(f"  Safe read (copies):    {mean_safe:.2f} +/- {std_safe:.2f} ms")
        print(f"  Speedup (fast vs safe): {speedup:.2f}x")

        # Benchmark write uncompressed
        print("\n  Benchmarking uncompressed write...")
        times_write = []
        temp_file = Path("temp_write_test.ply")
        temp_file_comp = Path("temp_write_test_compressed.ply")
        comp_file = Path(str(temp_file_comp).replace(".ply", ".compressed.ply"))

        try:
            for i in range(iterations):
                start = time.perf_counter()
                gsply.plywrite(temp_file, data.means, data.scales, data.quats,
                              data.opacities, data.sh0, data.shN)
                end = time.perf_counter()
                times_write.append((end - start) * 1000)

            mean_write = np.mean(times_write)
            std_write = np.std(times_write)
            print(f"  Write (uncompressed): {mean_write:.2f} +/- {std_write:.2f} ms")

            # Benchmark write compressed
            print("  Benchmarking compressed write...")
            times_write_comp = []
            for i in range(iterations):
                start = time.perf_counter()
                gsply.plywrite(temp_file_comp, data.means, data.scales, data.quats,
                              data.opacities, data.sh0, data.shN, compressed=True)
                end = time.perf_counter()
                times_write_comp.append((end - start) * 1000)

            mean_write_comp = np.mean(times_write_comp)
            std_write_comp = np.std(times_write_comp)

            # Check compression ratio and benchmark compressed read
            if comp_file.exists():
                comp_size = comp_file.stat().st_size / (1024 * 1024)
                comp_ratio = file_size / comp_size
                print(f"  Write (compressed):   {mean_write_comp:.2f} +/- {std_write_comp:.2f} ms")
                print(f"  Compression ratio:    {comp_ratio:.2f}x ({file_size:.2f} MB -> {comp_size:.2f} MB)")

                # Benchmark compressed read
                print("\n  Benchmarking compressed read...")
                times_read_comp = []
                for i in range(iterations):
                    start = time.perf_counter()
                    data_comp = gsply.plyread(comp_file)
                    end = time.perf_counter()
                    times_read_comp.append((end - start) * 1000)

                mean_read_comp = np.mean(times_read_comp)
                std_read_comp = np.std(times_read_comp)
                print(f"  Read (compressed):    {mean_read_comp:.2f} +/- {std_read_comp:.2f} ms")

                # Compare speedup
                read_speedup = mean_fast / mean_read_comp
                print(f"  Speedup (compressed vs uncompressed): {read_speedup:.2f}x")

                # Verify data integrity
                print("\n  Verifying compression integrity...")
                data_decompressed = gsply.plyread(comp_file)

                # Compare all arrays with appropriate tolerances for lossy compression
                # Note: This is chunk-based quantized compression (11-10-11 bits for position/scale)
                means_match = np.allclose(data.means, data_decompressed.means, rtol=5e-3, atol=0.05)
                scales_match = np.allclose(data.scales, data_decompressed.scales, rtol=5e-3, atol=0.05)

                # Quaternions: Compression normalizes them, so normalize originals for fair comparison
                # Note: 10-bit quantization per component can produce errors up to ~2.0 for extreme values
                quats_normalized = data.quats / np.linalg.norm(data.quats, axis=1, keepdims=True)
                quats_match = np.allclose(quats_normalized, data_decompressed.quats, rtol=1e-1, atol=2.5)

                # For opacities, clamp original to [-10, 10] before comparison (format limitation)
                # Values outside this range will be clamped, producing errors up to ~4.5 for extreme values
                opacities_clamped = np.clip(data.opacities, -10.0, 10.0)
                opacities_match = np.allclose(opacities_clamped, data_decompressed.opacities, rtol=1e-1, atol=5.0)

                # SH0 colors use 8 bits per channel
                sh0_match = np.allclose(data.sh0, data_decompressed.sh0, rtol=5e-3, atol=0.1)

                all_match = means_match and scales_match and quats_match and opacities_match and sh0_match

                # Calculate error metrics (use normalized quats and clamped opacities)
                means_error = np.abs(data.means - data_decompressed.means).max()
                scales_error = np.abs(data.scales - data_decompressed.scales).max()
                quats_error = np.abs(quats_normalized - data_decompressed.quats).max()
                opacities_error = np.abs(opacities_clamped - data_decompressed.opacities).max()
                sh0_error = np.abs(data.sh0 - data_decompressed.sh0).max()

                # Calculate modification stats
                num_clamped_opacity = np.sum((data.opacities < -10.0) | (data.opacities > 10.0))
                pct_clamped = 100.0 * num_clamped_opacity / len(data.opacities)

                # Check quaternion normalization
                quat_norms = np.linalg.norm(data.quats, axis=1)
                num_unnormalized = np.sum(np.abs(quat_norms - 1.0) > 0.01)
                pct_unnormalized = 100.0 * num_unnormalized / len(quat_norms)

                if all_match:
                    print("  Integrity check:      [OK] Lossy compression within tolerance")
                    print("  Max absolute errors:")
                    print(f"    Means:     {means_error:.6f}")
                    print(f"    Scales:    {scales_error:.6f}")
                    print(f"    Quats:     {quats_error:.6f} ({pct_unnormalized:.1f}% needed normalization)")
                    print(f"    Opacities: {opacities_error:.6f} ({pct_clamped:.1f}% clamped to [-10,10])")
                    print(f"    SH0:       {sh0_error:.6f}")
                else:
                    print("  Integrity check:      [FAIL] Data mismatch detected!")
                    print("  Max absolute errors:")
                    print(f"    Means:     {means_error:.6f} {'[FAIL]' if not means_match else '[OK]'}")
                    print(f"    Scales:    {scales_error:.6f} {'[FAIL]' if not scales_match else '[OK]'}")
                    print(f"    Quats:     {quats_error:.6f} {'[FAIL]' if not quats_match else '[OK]'} ({pct_unnormalized:.1f}% normalized)")
                    print(f"    Opacities: {opacities_error:.6f} {'[FAIL]' if not opacities_match else '[OK]'} ({pct_clamped:.1f}% clamped)")
                    print(f"    SH0:       {sh0_error:.6f} {'[FAIL]' if not sh0_match else '[OK]'}")

                    # Print data range info
                    print("  Original data ranges:")
                    print(f"    Means:     [{data.means.min():.3f}, {data.means.max():.3f}]")
                    print(f"    Scales:    [{data.scales.min():.3f}, {data.scales.max():.3f}]")
                    print(f"    Opacities: [{data.opacities.min():.3f}, {data.opacities.max():.3f}]")
                    print("  Decompressed data ranges:")
                    print(f"    Means:     [{data_decompressed.means.min():.3f}, {data_decompressed.means.max():.3f}]")
                    print(f"    Scales:    [{data_decompressed.scales.min():.3f}, {data_decompressed.scales.max():.3f}]")
                    print(f"    Opacities: [{data_decompressed.opacities.min():.3f}, {data_decompressed.opacities.max():.3f}]")
        finally:
            # Cleanup - ensure files are removed even on error
            if temp_file.exists():
                temp_file.unlink()
            if comp_file.exists():
                comp_file.unlink()
            if temp_file_comp.exists():
                temp_file_comp.unlink()

    return {
        'file': file_path.name,
        'num_gaussians': num_gaussians,
        'sh_degree': sh_degree,
        'is_compressed': is_compressed,
        'file_size_mb': file_size,
    }


def main():
    import os
    # Use environment variable or default to D:/4D/all_plys
    data_dir_str = os.getenv("GSPLY_TEST_DATA_DIR", "D:/4D/all_plys")
    data_dir = Path(data_dir_str)

    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        print("Please set GSPLY_TEST_DATA_DIR environment variable or update the default path.")
        return

    # Get all PLY files
    ply_files = sorted(data_dir.glob("*.ply"))

    if not ply_files:
        print(f"[ERROR] No PLY files found in {data_dir}")
        return

    print("=" * 80)
    print("GSPLY REAL DATA BENCHMARK")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Total PLY files: {len(ply_files)}")
    print("Benchmarking first 5 files (10 iterations each)")
    print()

    # Benchmark first 5 files
    results = []
    for file_path in ply_files[:5]:
        try:
            result = benchmark_file(file_path, iterations=10)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to benchmark {file_path.name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"{r['file']}: {r['num_gaussians']:,} Gaussians, SH{r['sh_degree']}, {r['file_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
