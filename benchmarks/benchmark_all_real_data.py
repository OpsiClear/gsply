"""Benchmark gsply performance with all files from D:/4D/all_plys."""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gsply


def benchmark_file(file_path, iterations=3):
    """Benchmark a single PLY file with fewer iterations for speed."""
    file_path = Path(file_path)

    # Get file info
    file_size = file_path.stat().st_size / (1024 * 1024)
    is_compressed, sh_degree = gsply.detect_format(file_path)

    # Quick read to get Gaussian count
    data = gsply.plyread(file_path)
    num_gaussians = data.means.shape[0]

    results = {
        'file': file_path.name,
        'num_gaussians': num_gaussians,
        'sh_degree': sh_degree,
        'is_compressed': is_compressed,
        'file_size_mb': file_size,
    }

    if is_compressed:
        # Benchmark compressed read
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            result = gsply.plyread(file_path)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        results['read_time_ms'] = np.mean(times)
        results['read_std_ms'] = np.std(times)

    else:
        # Benchmark fast read
        times_fast = []
        for i in range(iterations):
            start = time.perf_counter()
            result = gsply.plyread(file_path, fast=True)
            end = time.perf_counter()
            times_fast.append((end - start) * 1000)

        results['read_time_ms'] = np.mean(times_fast)
        results['read_std_ms'] = np.std(times_fast)

        # Benchmark write compressed
        times_write_comp = []
        temp_file_comp = Path("temp_write_test_compressed.ply")
        for i in range(iterations):
            start = time.perf_counter()
            gsply.plywrite(temp_file_comp, data.means, data.scales, data.quats,
                          data.opacities, data.sh0, data.shN, compressed=True)
            end = time.perf_counter()
            times_write_comp.append((end - start) * 1000)

        results['write_time_ms'] = np.mean(times_write_comp)
        results['write_std_ms'] = np.std(times_write_comp)

        # Check compression ratio
        comp_file = Path(str(temp_file_comp).replace(".ply", ".compressed.ply"))
        if comp_file.exists():
            comp_size = comp_file.stat().st_size / (1024 * 1024)
            results['comp_size_mb'] = comp_size
            results['comp_ratio'] = file_size / comp_size

            # Benchmark compressed read
            times_read_comp = []
            for i in range(iterations):
                start = time.perf_counter()
                data_comp = gsply.plyread(comp_file)
                end = time.perf_counter()
                times_read_comp.append((end - start) * 1000)

            results['comp_read_time_ms'] = np.mean(times_read_comp)
            results['comp_read_std_ms'] = np.std(times_read_comp)

            # Cleanup
            comp_file.unlink()

    return results


def main():
    data_dir = Path("D:/4D/all_plys")

    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        print("Please update the path in the script.")
        return

    # Get all PLY files
    ply_files = sorted(data_dir.glob("*.ply"))

    if not ply_files:
        print(f"[ERROR] No PLY files found in {data_dir}")
        return

    print("=" * 80)
    print("GSPLY COMPREHENSIVE REAL DATA BENCHMARK")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Total PLY files: {len(ply_files)}")
    print("Benchmarking ALL files (3 iterations each for speed)")
    print(f"This will take approximately {len(ply_files) * 3 * 0.3:.0f} seconds...")
    print()

    # Benchmark all files
    results = []
    for idx, file_path in enumerate(ply_files, 1):
        print(f"[{idx}/{len(ply_files)}] Processing {file_path.name}...", end='', flush=True)
        try:
            result = benchmark_file(file_path, iterations=3)
            results.append(result)
            print(f" {result['num_gaussians']:,} Gaussians, {result['read_time_ms']:.1f}ms read", end='')
            if 'write_time_ms' in result:
                print(f", {result['write_time_ms']:.1f}ms write")
            else:
                print()
        except Exception as e:
            print(f" [FAIL] {e}")
            continue

    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    total_gaussians = sum(r['num_gaussians'] for r in results)
    total_size_mb = sum(r['file_size_mb'] for r in results)

    read_times = [r['read_time_ms'] for r in results]
    write_times = [r['write_time_ms'] for r in results if 'write_time_ms' in r]
    comp_read_times = [r['comp_read_time_ms'] for r in results if 'comp_read_time_ms' in r]
    comp_ratios = [r['comp_ratio'] for r in results if 'comp_ratio' in r]

    print(f"\nTotal files:         {len(results)}")
    print(f"Total Gaussians:     {total_gaussians:,}")
    print(f"Total size:          {total_size_mb:.2f} MB")
    print(f"Avg Gaussians/file:  {total_gaussians / len(results):,.0f}")
    print(f"Avg file size:       {total_size_mb / len(results):.2f} MB")

    print("\nRead Performance (uncompressed):")
    print(f"  Mean:   {np.mean(read_times):.2f} ms")
    print(f"  Median: {np.median(read_times):.2f} ms")
    print(f"  Min:    {np.min(read_times):.2f} ms")
    print(f"  Max:    {np.max(read_times):.2f} ms")
    print(f"  Std:    {np.std(read_times):.2f} ms")

    if write_times:
        print("\nWrite Performance (compressed):")
        print(f"  Mean:   {np.mean(write_times):.2f} ms")
        print(f"  Median: {np.median(write_times):.2f} ms")
        print(f"  Min:    {np.min(write_times):.2f} ms")
        print(f"  Max:    {np.max(write_times):.2f} ms")
        print(f"  Std:    {np.std(write_times):.2f} ms")

    if comp_read_times:
        print("\nRead Performance (compressed):")
        print(f"  Mean:   {np.mean(comp_read_times):.2f} ms")
        print(f"  Median: {np.median(comp_read_times):.2f} ms")
        print(f"  Min:    {np.min(comp_read_times):.2f} ms")
        print(f"  Max:    {np.max(comp_read_times):.2f} ms")
        print(f"  Std:    {np.std(comp_read_times):.2f} ms")

    if comp_ratios:
        print("\nCompression Ratio:")
        print(f"  Mean:   {np.mean(comp_ratios):.2f}x")
        print(f"  Median: {np.median(comp_ratios):.2f}x")
        print(f"  Min:    {np.min(comp_ratios):.2f}x")
        print(f"  Max:    {np.max(comp_ratios):.2f}x")

        total_comp_size = sum(r['comp_size_mb'] for r in results if 'comp_size_mb' in r)
        print(f"\nTotal compressed size: {total_comp_size:.2f} MB")
        print(f"Overall compression:   {total_size_mb / total_comp_size:.2f}x ({total_size_mb:.2f} MB -> {total_comp_size:.2f} MB)")

    # Performance summary
    if write_times and comp_read_times:
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        avg_gaussians = total_gaussians / len(results)
        print(f"\nFor {avg_gaussians:,.0f} Gaussians (average):")
        print(f"  Read (uncompressed):  {np.mean(read_times):.2f} ms")
        print(f"  Write (compressed):   {np.mean(write_times):.2f} ms")
        print(f"  Read (compressed):    {np.mean(comp_read_times):.2f} ms")
        print(f"  Compression ratio:    {np.mean(comp_ratios):.2f}x")

        throughput_read = (avg_gaussians / 1000) / (np.mean(read_times) / 1000)
        throughput_write = (avg_gaussians / 1000) / (np.mean(write_times) / 1000)
        throughput_comp_read = (avg_gaussians / 1000) / (np.mean(comp_read_times) / 1000)

        print("\nThroughput:")
        print(f"  Read (uncompressed):  {throughput_read:.1f}K Gaussians/sec")
        print(f"  Write (compressed):   {throughput_write:.1f}K Gaussians/sec")
        print(f"  Read (compressed):    {throughput_comp_read:.1f}K Gaussians/sec")


if __name__ == "__main__":
    main()
