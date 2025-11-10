"""Benchmark gsply against other PLY libraries.

Compares performance of gsply, Open3D, and plyfile for reading and writing
Gaussian Splatting PLY files.

Usage:
    uv run python benchmark.py [--file PATH] [--warmup N] [--iterations N]

Example:
    uv run python benchmark.py --file ../export_with_edits/frame_00000.ply --iterations 10
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass
import tempfile
import gc

import numpy as np
import argparse

# Import libraries
try:
    import gsply
    GSPLY_AVAILABLE = True
except ImportError:
    GSPLY_AVAILABLE = False
    logging.warning("gsply not available")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D not available - install with: uv pip install open3d")

try:
    from plyfile import PlyData, PlyElement
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False
    logging.warning("plyfile not available - install with: uv pip install plyfile")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    file: str = "../export_with_edits/frame_00000.ply"
    """Path to test PLY file"""

    warmup: int = 3
    """Number of warmup iterations"""

    iterations: int = 10
    """Number of benchmark iterations"""

    skip_write: bool = False
    """Skip write benchmarks"""


def timer(func, *args, warmup: int = 3, iterations: int = 10, **kwargs):
    """Time a function with warmup and multiple iterations.

    Args:
        func: Function to time
        warmup: Number of warmup runs
        iterations: Number of timed runs
        *args, **kwargs: Arguments to pass to func

    Returns:
        Tuple of (mean_time, std_time, results_from_last_run)
    """
    # Warmup
    for _ in range(warmup):
        result = func(*args, **kwargs)
        gc.collect()

    # Timed runs
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time, result


def benchmark_gsply_read(file_path: Path, **timer_kwargs):
    """Benchmark gsply read performance."""
    def read_fn():
        return gsply.plyread(file_path)

    mean_time, std_time, result = timer(read_fn, **timer_kwargs)
    # GSData namedtuple - access via attributes
    num_gaussians = result.means.shape[0]

    return {
        'name': 'gsply',
        'operation': 'read',
        'mean_time': mean_time,
        'std_time': std_time,
        'num_gaussians': num_gaussians,
    }


def benchmark_open3d_read(file_path: Path, **timer_kwargs):
    """Benchmark Open3D read performance."""
    def read_fn():
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        return points

    mean_time, std_time, result = timer(read_fn, **timer_kwargs)
    num_points = result.shape[0]

    return {
        'name': 'Open3D',
        'operation': 'read',
        'mean_time': mean_time,
        'std_time': std_time,
        'num_gaussians': num_points,
    }


def benchmark_plyfile_read(file_path: Path, **timer_kwargs):
    """Benchmark plyfile read performance (full Gaussian properties)."""
    def read_fn():
        plydata = PlyData.read(file_path)
        vertex = plydata['vertex']

        # Extract ALL Gaussian properties (same as gsply)
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        means = np.column_stack((x, y, z))

        # Scales
        scale_0 = vertex['scale_0']
        scale_1 = vertex['scale_1']
        scale_2 = vertex['scale_2']
        scales = np.column_stack((scale_0, scale_1, scale_2))

        # Quaternions
        rot_0 = vertex['rot_0']
        rot_1 = vertex['rot_1']
        rot_2 = vertex['rot_2']
        rot_3 = vertex['rot_3']
        quats = np.column_stack((rot_0, rot_1, rot_2, rot_3))

        # Opacities
        opacities = np.array(vertex['opacity'])

        # SH0
        f_dc_0 = vertex['f_dc_0']
        f_dc_1 = vertex['f_dc_1']
        f_dc_2 = vertex['f_dc_2']
        sh0 = np.column_stack((f_dc_0, f_dc_1, f_dc_2))

        # SH higher-order (if exists)
        prop_names = [p.name for p in vertex.properties]
        sh_rest_props = [p for p in prop_names if p.startswith('f_rest_')]
        if sh_rest_props:
            sh_rest_data = [np.array(vertex[p]) for p in sh_rest_props]
            shN = np.column_stack(sh_rest_data).reshape(len(x), -1, 3)
        else:
            shN = np.empty((len(x), 0, 3), dtype=np.float32)

        return means, scales, quats, opacities, sh0, shN

    mean_time, std_time, result = timer(read_fn, **timer_kwargs)
    means = result[0]
    num_points = means.shape[0]

    return {
        'name': 'plyfile',
        'operation': 'read',
        'mean_time': mean_time,
        'std_time': std_time,
        'num_gaussians': num_points,
    }


def benchmark_gsply_write(data, output_path: Path, **timer_kwargs):
    """Benchmark gsply write performance."""
    # GSData namedtuple - unpack first 6 elements
    means, scales, quats, opacities, sh0, shN = data[:6]

    def write_fn():
        gsply.plywrite(output_path, means, scales, quats, opacities, sh0, shN)

    mean_time, std_time, _ = timer(write_fn, **timer_kwargs)

    file_size = output_path.stat().st_size if output_path.exists() else 0

    return {
        'name': 'gsply',
        'operation': 'write',
        'mean_time': mean_time,
        'std_time': std_time,
        'file_size': file_size,
    }


def benchmark_open3d_write(data, output_path: Path, **timer_kwargs):
    """Benchmark Open3D write performance."""
    means, _, _, _, _, _ = data

    def write_fn():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means)
        o3d.io.write_point_cloud(str(output_path), pcd)

    mean_time, std_time, _ = timer(write_fn, **timer_kwargs)

    file_size = output_path.stat().st_size if output_path.exists() else 0

    return {
        'name': 'Open3D',
        'operation': 'write',
        'mean_time': mean_time,
        'std_time': std_time,
        'file_size': file_size,
    }


def benchmark_plyfile_write(data, output_path: Path, **timer_kwargs):
    """Benchmark plyfile write performance (full SH degree 3)."""
    # GSData namedtuple - unpack first 6 elements
    means, scales, quats, opacities, sh0, shN = data[:6]
    num_gaussians = means.shape[0]

    # Flatten shN for writing
    if shN.ndim == 3:
        shN_flat = shN.reshape(num_gaussians, -1)
    else:
        shN_flat = shN

    def write_fn():
        # Create dtype with ALL properties (SH degree 3 = 59 properties)
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]

        # Add f_rest properties (45 for SH degree 3)
        for i in range(shN_flat.shape[1]):
            dtype_list.append((f'f_rest_{i}', 'f4'))

        dtype_list.extend([
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ])

        vertex_data = np.empty(num_gaussians, dtype=dtype_list)

        # Fill basic properties
        vertex_data['x'] = means[:, 0]
        vertex_data['y'] = means[:, 1]
        vertex_data['z'] = means[:, 2]
        vertex_data['f_dc_0'] = sh0[:, 0]
        vertex_data['f_dc_1'] = sh0[:, 1]
        vertex_data['f_dc_2'] = sh0[:, 2]

        # Fill f_rest properties
        for i in range(shN_flat.shape[1]):
            vertex_data[f'f_rest_{i}'] = shN_flat[:, i]

        vertex_data['opacity'] = opacities
        vertex_data['scale_0'] = scales[:, 0]
        vertex_data['scale_1'] = scales[:, 1]
        vertex_data['scale_2'] = scales[:, 2]
        vertex_data['rot_0'] = quats[:, 0]
        vertex_data['rot_1'] = quats[:, 1]
        vertex_data['rot_2'] = quats[:, 2]
        vertex_data['rot_3'] = quats[:, 3]

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        plydata = PlyData([vertex_element], text=False)
        plydata.write(str(output_path))

    mean_time, std_time, _ = timer(write_fn, **timer_kwargs)

    file_size = output_path.stat().st_size if output_path.exists() else 0

    return {
        'name': 'plyfile',
        'operation': 'write',
        'mean_time': mean_time,
        'std_time': std_time,
        'file_size': file_size,
    }


def benchmark_gsply_write_sh0(data, output_path: Path, **timer_kwargs):
    """Benchmark gsply write performance (SH degree 0 only)."""
    means, scales, quats, opacities, sh0, _ = data

    def write_fn():
        gsply.plywrite(output_path, means, scales, quats, opacities, sh0, shN=None)

    mean_time, std_time, _ = timer(write_fn, **timer_kwargs)

    file_size = output_path.stat().st_size if output_path.exists() else 0

    return {
        'name': 'gsply (SH0)',
        'operation': 'write_sh0',
        'mean_time': mean_time,
        'std_time': std_time,
        'file_size': file_size,
    }


def benchmark_plyfile_write_sh0(data, output_path: Path, **timer_kwargs):
    """Benchmark plyfile write performance (SH degree 0 only)."""
    means, scales, quats, opacities, sh0, _ = data
    num_gaussians = means.shape[0]

    def write_fn():
        # Create dtype with only SH0 properties (14 properties total)
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ]

        vertex_data = np.empty(num_gaussians, dtype=dtype_list)

        # Fill basic properties
        vertex_data['x'] = means[:, 0]
        vertex_data['y'] = means[:, 1]
        vertex_data['z'] = means[:, 2]
        vertex_data['f_dc_0'] = sh0[:, 0]
        vertex_data['f_dc_1'] = sh0[:, 1]
        vertex_data['f_dc_2'] = sh0[:, 2]
        vertex_data['opacity'] = opacities
        vertex_data['scale_0'] = scales[:, 0]
        vertex_data['scale_1'] = scales[:, 1]
        vertex_data['scale_2'] = scales[:, 2]
        vertex_data['rot_0'] = quats[:, 0]
        vertex_data['rot_1'] = quats[:, 1]
        vertex_data['rot_2'] = quats[:, 2]
        vertex_data['rot_3'] = quats[:, 3]

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        plydata = PlyData([vertex_element], text=False)
        plydata.write(str(output_path))

    mean_time, std_time, _ = timer(write_fn, **timer_kwargs)

    file_size = output_path.stat().st_size if output_path.exists() else 0

    return {
        'name': 'plyfile (SH0)',
        'operation': 'write_sh0',
        'mean_time': mean_time,
        'std_time': std_time,
        'file_size': file_size,
    }


def verify_outputs_equivalent(gsply_path: Path, plyfile_path: Path) -> tuple[bool, str]:
    """Verify that gsply and plyfile outputs are equivalent.

    Returns:
        Tuple of (is_equivalent, message)
    """
    try:
        # Read both files
        gsply_data = gsply.plyread(gsply_path)
        plyfile_plydata = PlyData.read(plyfile_path)
        vertex = plyfile_plydata['vertex']

        # Extract plyfile data
        plyfile_means = np.column_stack((vertex['x'], vertex['y'], vertex['z']))
        plyfile_sh0 = np.column_stack((vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']))
        plyfile_opacities = np.array(vertex['opacity'])
        plyfile_scales = np.column_stack((vertex['scale_0'], vertex['scale_1'], vertex['scale_2']))
        plyfile_quats = np.column_stack((vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']))

        # Check if higher-order SH exists
        prop_names = [p.name for p in vertex.properties]
        sh_rest_props = [p for p in prop_names if p.startswith('f_rest_')]
        if sh_rest_props:
            sh_rest_data = [np.array(vertex[p]) for p in sh_rest_props]
            plyfile_shN = np.column_stack(sh_rest_data).reshape(len(vertex['x']), -1, 3)
        else:
            plyfile_shN = np.empty((len(vertex['x']), 0, 3), dtype=np.float32)

        # Unpack gsply data
        gsply_means, gsply_scales, gsply_quats, gsply_opacities, gsply_sh0, gsply_shN = gsply_data

        # Compare shapes
        if gsply_means.shape != plyfile_means.shape:
            return False, f"Means shape mismatch: {gsply_means.shape} vs {plyfile_means.shape}"

        # Compare values (use allclose for floating point comparison)
        if not np.allclose(gsply_means, plyfile_means, rtol=1e-6, atol=1e-8):
            return False, "Means values differ"
        if not np.allclose(gsply_scales, plyfile_scales, rtol=1e-6, atol=1e-8):
            return False, "Scales values differ"
        if not np.allclose(gsply_quats, plyfile_quats, rtol=1e-6, atol=1e-8):
            return False, "Quaternions values differ"
        if not np.allclose(gsply_opacities, plyfile_opacities, rtol=1e-6, atol=1e-8):
            return False, "Opacities values differ"
        if not np.allclose(gsply_sh0, plyfile_sh0, rtol=1e-6, atol=1e-8):
            return False, "SH0 values differ"
        if not np.allclose(gsply_shN, plyfile_shN, rtol=1e-6, atol=1e-8):
            return False, "Higher-order SH values differ"

        return True, "Files are equivalent"

    except Exception as e:
        return False, f"Error during verification: {e}"


def format_time(seconds: float) -> str:
    """Format time in ms with appropriate precision."""
    ms = seconds * 1000
    if ms < 1:
        return f"{ms*1000:.2f}us"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{seconds:.3f}s"


def format_size(bytes: int) -> str:
    """Format file size."""
    if bytes < 1024:
        return f"{bytes}B"
    elif bytes < 1024**2:
        return f"{bytes/1024:.2f}KB"
    else:
        return f"{bytes/(1024**2):.2f}MB"


def print_results(results: list, baseline_name: str = 'gsply'):
    """Print benchmark results in a nice table."""
    if not results:
        return

    # Separate read and write results
    read_results = [r for r in results if r['operation'] == 'read']
    write_results = [r for r in results if r['operation'] == 'write']
    write_sh0_results = [r for r in results if r['operation'] == 'write_sh0']

    # Print read results
    if read_results:
        print("\n" + "="*80)
        print("READ PERFORMANCE")
        print("="*80)

        baseline = next((r for r in read_results if r['name'] == baseline_name), read_results[0])
        baseline_time = baseline['mean_time']

        print(f"\n{'Library':<15} {'Time':<15} {'Std Dev':<15} {'Speedup':<10} {'Gaussians':<12}")
        print("-"*80)

        for result in sorted(read_results, key=lambda x: x['mean_time']):
            speedup = baseline_time / result['mean_time']
            speedup_str = f"{speedup:.2f}x" if result['name'] != baseline_name else "baseline"

            print(f"{result['name']:<15} "
                  f"{format_time(result['mean_time']):<15} "
                  f"{format_time(result['std_time']):<15} "
                  f"{speedup_str:<10} "
                  f"{result['num_gaussians']:<12,}")

    # Print write results (SH degree 3)
    if write_results:
        print("\n" + "="*80)
        print("WRITE PERFORMANCE (SH degree 3, 59 properties)")
        print("="*80)

        baseline = next((r for r in write_results if r['name'] == baseline_name), write_results[0])
        baseline_time = baseline['mean_time']

        print(f"\n{'Library':<15} {'Time':<15} {'Std Dev':<15} {'Speedup':<10} {'File Size':<12}")
        print("-"*80)

        for result in sorted(write_results, key=lambda x: x['mean_time']):
            speedup = baseline_time / result['mean_time']
            speedup_str = f"{speedup:.2f}x" if result['name'] != baseline_name else "baseline"

            print(f"{result['name']:<15} "
                  f"{format_time(result['mean_time']):<15} "
                  f"{format_time(result['std_time']):<15} "
                  f"{speedup_str:<10} "
                  f"{format_size(result['file_size']):<12}")

    # Print write SH0 results
    if write_sh0_results:
        print("\n" + "="*80)
        print("WRITE PERFORMANCE (SH degree 0, 14 properties)")
        print("="*80)

        baseline = next((r for r in write_sh0_results if r['name'] == 'gsply (SH0)'), write_sh0_results[0])
        baseline_time = baseline['mean_time']

        print(f"\n{'Library':<15} {'Time':<15} {'Std Dev':<15} {'Speedup':<10} {'File Size':<12}")
        print("-"*80)

        for result in sorted(write_sh0_results, key=lambda x: x['mean_time']):
            speedup = baseline_time / result['mean_time']
            speedup_str = f"{speedup:.2f}x" if result['name'] == 'gsply (SH0)' else f"{speedup:.2f}x"

            print(f"{result['name']:<15} "
                  f"{format_time(result['mean_time']):<15} "
                  f"{format_time(result['std_time']):<15} "
                  f"{speedup_str:<10} "
                  f"{format_size(result['file_size']):<12}")


def main(config: BenchmarkConfig):
    """Run benchmarks."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    file_path = Path(config.file)
    if not file_path.exists():
        logging.error(f"Test file not found: {file_path}")
        return 1

    print("\n" + "="*80)
    print("GSPLY BENCHMARK")
    print("="*80)
    print(f"\nTest file: {file_path}")
    print(f"Warmup iterations: {config.warmup}")
    print(f"Benchmark iterations: {config.iterations}")
    print(f"\nFile size: {format_size(file_path.stat().st_size)}")

    # Check library availability
    print("\nLibrary availability:")
    print(f"  gsply:   {'[OK]' if GSPLY_AVAILABLE else '[FAIL]'}")
    print(f"  Open3D:  {'[OK]' if OPEN3D_AVAILABLE else '[FAIL]'}")
    print(f"  plyfile: {'[OK]' if PLYFILE_AVAILABLE else '[FAIL]'}")

    if not any([GSPLY_AVAILABLE, OPEN3D_AVAILABLE, PLYFILE_AVAILABLE]):
        logging.error("\nNo libraries available for benchmarking!")
        return 1

    timer_kwargs = {
        'warmup': config.warmup,
        'iterations': config.iterations,
    }

    results = []

    # Read benchmarks
    print("\nRunning READ benchmarks...")

    if GSPLY_AVAILABLE:
        print("  Benchmarking gsply read...")
        result = benchmark_gsply_read(file_path, **timer_kwargs)
        results.append(result)
        gsply_data = gsply.plyread(file_path)

    if OPEN3D_AVAILABLE:
        print("  Benchmarking Open3D read...")
        result = benchmark_open3d_read(file_path, **timer_kwargs)
        results.append(result)

    if PLYFILE_AVAILABLE:
        print("  Benchmarking plyfile read...")
        result = benchmark_plyfile_read(file_path, **timer_kwargs)
        results.append(result)

    # Write benchmarks
    if not config.skip_write and GSPLY_AVAILABLE:
        print("\nRunning WRITE benchmarks...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            if GSPLY_AVAILABLE:
                print("  Benchmarking gsply write...")
                output_path = tmpdir / "gsply_output.ply"
                result = benchmark_gsply_write(gsply_data, output_path, **timer_kwargs)
                results.append(result)

            if OPEN3D_AVAILABLE:
                print("  Benchmarking Open3D write...")
                output_path = tmpdir / "open3d_output.ply"
                result = benchmark_open3d_write(gsply_data, output_path, **timer_kwargs)
                results.append(result)

            if PLYFILE_AVAILABLE:
                print("  Benchmarking plyfile write...")
                output_path = tmpdir / "plyfile_output.ply"
                result = benchmark_plyfile_write(gsply_data, output_path, **timer_kwargs)
                results.append(result)

            # Write benchmarks (SH degree 0)
            print("\nRunning WRITE (SH0) benchmarks...")

            gsply_sh0_path = tmpdir / "gsply_sh0.ply"
            plyfile_sh0_path = tmpdir / "plyfile_sh0.ply"

            if GSPLY_AVAILABLE:
                print("  Benchmarking gsply write (SH0)...")
                result = benchmark_gsply_write_sh0(gsply_data, gsply_sh0_path, **timer_kwargs)
                results.append(result)

            if PLYFILE_AVAILABLE:
                print("  Benchmarking plyfile write (SH0)...")
                result = benchmark_plyfile_write_sh0(gsply_data, plyfile_sh0_path, **timer_kwargs)
                results.append(result)

            # Verify output equivalence
            if GSPLY_AVAILABLE and PLYFILE_AVAILABLE:
                print("\nVerifying output equivalence...")

                # Verify SH3 outputs
                gsply_sh3_path = tmpdir / "gsply_output.ply"
                plyfile_sh3_path = tmpdir / "plyfile_output.ply"
                if gsply_sh3_path.exists() and plyfile_sh3_path.exists():
                    is_eq, msg = verify_outputs_equivalent(gsply_sh3_path, plyfile_sh3_path)
                    status = "[OK]" if is_eq else "[FAIL]"
                    print(f"  SH3 outputs: {status} {msg}")

                # Verify SH0 outputs
                if gsply_sh0_path.exists() and plyfile_sh0_path.exists():
                    is_eq, msg = verify_outputs_equivalent(gsply_sh0_path, plyfile_sh0_path)
                    status = "[OK]" if is_eq else "[FAIL]"
                    print(f"  SH0 outputs: {status} {msg}")

    # Print results
    print_results(results)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gsply performance")
    parser.add_argument(
        "--file",
        type=str,
        default="../export_with_edits/frame_00000.ply",
        help="Path to test PLY file"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Skip write benchmarks"
    )

    args = parser.parse_args()
    config = BenchmarkConfig(
        file=args.file,
        warmup=args.warmup,
        iterations=args.iterations,
        skip_write=args.skip_write
    )
    exit(main(config))
