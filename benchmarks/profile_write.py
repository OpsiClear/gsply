"""Profile write performance to identify bottlenecks."""

import time
import numpy as np
from pathlib import Path
import tempfile

import gsply
from plyfile import PlyData, PlyElement
from test_utils import get_test_file


def profile_gsply_sh0(means, scales, quats, opacities, sh0, iterations=50):
    """Profile gsply SH0 write with detailed timing."""
    times = {
        'concatenate': [],
        'astype': [],
        'write': [],
        'total': []
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.ply"

        for _ in range(iterations):
            # Simulate the operations in gsply
            start_total = time.perf_counter()

            # Concatenate
            t0 = time.perf_counter()
            data = np.concatenate([means, sh0, opacities[:, None], scales, quats], axis=1)
            t1 = time.perf_counter()
            times['concatenate'].append(t1 - t0)

            # Astype
            t0 = time.perf_counter()
            data = data.astype('<f4')
            t1 = time.perf_counter()
            times['astype'].append(t1 - t0)

            # Write (including header)
            t0 = time.perf_counter()
            gsply.plywrite(output_path, means, scales, quats, opacities, sh0, shN=None)
            t1 = time.perf_counter()
            times['write'].append(t1 - t0)

            end_total = time.perf_counter()
            times['total'].append(end_total - start_total)

    return {k: (np.mean(v) * 1000, np.std(v) * 1000) for k, v in times.items()}


def profile_plyfile_sh0(means, scales, quats, opacities, sh0, iterations=50):
    """Profile plyfile SH0 write with detailed timing."""
    num_gaussians = means.shape[0]

    times = {
        'create_dtype': [],
        'allocate_array': [],
        'assign_data': [],
        'create_element': [],
        'write': [],
        'total': []
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.ply"

        for _ in range(iterations):
            start_total = time.perf_counter()

            # Create dtype
            t0 = time.perf_counter()
            dtype_list = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                ('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ]
            t1 = time.perf_counter()
            times['create_dtype'].append(t1 - t0)

            # Allocate array
            t0 = time.perf_counter()
            vertex_data = np.empty(num_gaussians, dtype=dtype_list)
            t1 = time.perf_counter()
            times['allocate_array'].append(t1 - t0)

            # Assign data (14 assignments)
            t0 = time.perf_counter()
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
            t1 = time.perf_counter()
            times['assign_data'].append(t1 - t0)

            # Create element
            t0 = time.perf_counter()
            vertex_element = PlyElement.describe(vertex_data, 'vertex')
            plydata = PlyData([vertex_element], text=False)
            t1 = time.perf_counter()
            times['create_element'].append(t1 - t0)

            # Write
            t0 = time.perf_counter()
            plydata.write(str(output_path))
            t1 = time.perf_counter()
            times['write'].append(t1 - t0)

            end_total = time.perf_counter()
            times['total'].append(end_total - start_total)

    return {k: (np.mean(v) * 1000, np.std(v) * 1000) for k, v in times.items()}


def main():
    # Load test data
    test_file = get_test_file()
    data = gsply.plyread(test_file)
    means, scales, quats, opacities, sh0, shN = data.means, data.scales, data.quats, data.opacities, data.sh0, data.shN

    print("=" * 80)
    print("WRITE PERFORMANCE PROFILING (SH0)")
    print("=" * 80)
    print(f"\nTest data: {means.shape[0]} Gaussians\n")

    # Profile gsply
    print("Profiling gsply...")
    gsply_times = profile_gsply_sh0(means, scales, quats, opacities, sh0)

    print("\ngsply breakdown:")
    print(f"  concatenate:  {gsply_times['concatenate'][0]:.3f}ms +/- {gsply_times['concatenate'][1]:.3f}ms")
    print(f"  astype:       {gsply_times['astype'][0]:.3f}ms +/- {gsply_times['astype'][1]:.3f}ms")
    print(f"  write (full): {gsply_times['write'][0]:.3f}ms +/- {gsply_times['write'][1]:.3f}ms")
    print(f"  TOTAL:        {gsply_times['total'][0]:.3f}ms +/- {gsply_times['total'][1]:.3f}ms")

    # Profile plyfile
    print("\nProfiling plyfile...")
    plyfile_times = profile_plyfile_sh0(means, scales, quats, opacities, sh0)

    print("\nplyfile breakdown:")
    print(f"  create_dtype:    {plyfile_times['create_dtype'][0]:.3f}ms +/- {plyfile_times['create_dtype'][1]:.3f}ms")
    print(f"  allocate_array:  {plyfile_times['allocate_array'][0]:.3f}ms +/- {plyfile_times['allocate_array'][1]:.3f}ms")
    print(f"  assign_data:     {plyfile_times['assign_data'][0]:.3f}ms +/- {plyfile_times['assign_data'][1]:.3f}ms")
    print(f"  create_element:  {plyfile_times['create_element'][0]:.3f}ms +/- {plyfile_times['create_element'][1]:.3f}ms")
    print(f"  write:           {plyfile_times['write'][0]:.3f}ms +/- {plyfile_times['write'][1]:.3f}ms")
    print(f"  TOTAL:           {plyfile_times['total'][0]:.3f}ms +/- {plyfile_times['total'][1]:.3f}ms")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Calculate overhead
    gsply_overhead = gsply_times['write'][0] - (gsply_times['concatenate'][0] + gsply_times['astype'][0])
    plyfile_overhead = plyfile_times['total'][0] - (plyfile_times['assign_data'][0] + plyfile_times['write'][0])

    print(f"\ngsply overhead (header + I/O): {gsply_overhead:.3f}ms")
    print(f"plyfile overhead (setup + element creation): {plyfile_overhead:.3f}ms")
    print(f"\ngsply data prep (concatenate + astype): {gsply_times['concatenate'][0] + gsply_times['astype'][0]:.3f}ms")
    print(f"plyfile data prep (assign): {plyfile_times['assign_data'][0]:.3f}ms")


if __name__ == "__main__":
    main()
