"""Test theoretical I/O performance limits."""

import numpy as np
import time
from pathlib import Path
from test_utils import get_test_file

def benchmark_raw_io():
    """Benchmark raw file I/O to understand limits."""
    test_file = get_test_file()
    file_size = test_file.stat().st_size

    print("=" * 80)
    print("RAW I/O PERFORMANCE LIMITS")
    print("=" * 80)
    print(f"\nFile: {test_file.name}")
    print(f"Size: {file_size / 1024 / 1024:.2f}MB\n")

    # Test 1: Read entire file as bytes
    iterations = 50
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with open(test_file, 'rb') as f:
            data = f.read()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_read = np.mean(times) * 1000
    std_read = np.std(times) * 1000
    throughput = file_size / np.mean(times) / 1024 / 1024

    print("Read entire file (bytes):")
    print(f"  Time: {mean_read:.2f}ms +/- {std_read:.2f}ms")
    print(f"  Throughput: {throughput:.0f}MB/s")
    print()

    # Test 2: np.fromfile from offset (simulating PLY read)
    # First find header size
    header_size = 0
    with open(test_file, 'rb') as f:
        while True:
            line = f.readline()
            header_size += len(line)
            if b'end_header' in line:
                break

    data_size = file_size - header_size
    num_floats = data_size // 4

    print(f"Header size: {header_size} bytes")
    print(f"Data size: {data_size / 1024 / 1024:.2f}MB ({num_floats:,} floats)")
    print()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with open(test_file, 'rb') as f:
            f.seek(header_size)
            data = np.fromfile(f, dtype=np.float32)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_fromfile = np.mean(times) * 1000
    std_fromfile = np.std(times) * 1000
    throughput = data_size / np.mean(times) / 1024 / 1024

    print("np.fromfile (from offset):")
    print(f"  Time: {mean_fromfile:.2f}ms +/- {std_fromfile:.2f}ms")
    print(f"  Throughput: {throughput:.0f}MB/s")
    print()

    # Test 3: Read + reshape (what gsply does)
    n_verts = 50375
    n_props = 59

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with open(test_file, 'rb') as f:
            f.seek(header_size)
            data = np.fromfile(f, dtype=np.float32, count=n_verts * n_props)
            data = data.reshape(n_verts, n_props)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_reshape = np.mean(times) * 1000
    std_reshape = np.std(times) * 1000

    print("np.fromfile + reshape:")
    print(f"  Time: {mean_reshape:.2f}ms +/- {std_reshape:.2f}ms")
    print()

    # Test 4: Header parsing overhead
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with open(test_file, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == "end_header":
                    break
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_header = np.mean(times) * 1000
    std_header = np.std(times) * 1000

    print("Header parsing (readline + decode):")
    print(f"  Time: {mean_header:.2f}ms +/- {std_header:.2f}ms")
    print()

    # Test 5: Property name validation
    expected = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + \
               [f"f_rest_{i}" for i in range(45)] + \
               ["opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"]

    times_list = []
    times_tuple = []

    for _ in range(10000):
        t0 = time.perf_counter()
        result = expected == expected  # List comparison
        t1 = time.perf_counter()
        times_list.append(t1 - t0)

        t0 = time.perf_counter()
        result = tuple(expected) == tuple(expected)  # Tuple comparison
        t1 = time.perf_counter()
        times_tuple.append(t1 - t0)

    print("Property name validation (10k iterations):")
    print(f"  List comparison:  {np.mean(times_list)*1e6:.2f}us")
    print(f"  Tuple comparison: {np.mean(times_tuple)*1e6:.2f}us")
    print()

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    print("Current gsply read: ~5.96ms (from benchmark)")
    print(f"  Header parsing:   ~{mean_header:.2f}ms ({mean_header/5.96*100:.1f}%)")
    print(f"  Data read+reshape: ~{mean_reshape:.2f}ms ({mean_reshape/5.96*100:.1f}%)")
    print(f"  Slicing overhead:  ~{5.96-mean_header-mean_reshape:.2f}ms")
    print()
    print(f"Theoretical minimum: {mean_header + mean_reshape:.2f}ms")
    print("Current performance: 5.96ms")
    print(f"Potential speedup:   {5.96/(mean_header + mean_reshape):.2f}x")

if __name__ == "__main__":
    benchmark_raw_io()
