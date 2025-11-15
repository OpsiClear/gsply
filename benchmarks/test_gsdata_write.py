"""Test GSData-based write API with zero-copy."""

import tempfile
import time
from pathlib import Path

import gsply

# Read real file (creates GSData with _base)
print("=" * 80)
print("TEST: GSData Write API with Zero-Copy")
print("=" * 80)

data = gsply.plyread("D:/4D/all_plys/frame_0.ply")
print(f"\n[1] Loaded {len(data):,} Gaussians")
print(f"    Has _base: {data._base is not None}")
if data._base is not None:
    print(f"    _base shape: {data._base.shape}")

# Test 1: Write GSData directly (should use zero-copy)
print("\n[2] Testing GSData direct write (RECOMMENDED)...")
output_file = Path(tempfile.mktemp(suffix=".ply"))

times = []
for i in range(20):
    start = time.perf_counter()
    gsply.plywrite(str(output_file), data)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    if i == 0:
        file_size = output_file.stat().st_size / 1024 / 1024
        print(f"    First write: {elapsed:.2f}ms, file size: {file_size:.2f} MB")
    output_file.unlink()

import numpy as np
mean_time = np.mean(times)
min_time = np.min(times)
print(f"    Mean time: {mean_time:.2f} ms")
print(f"    Min time:  {min_time:.2f} ms")
print(f"    Throughput: {len(data) / mean_time * 1000 / 1e6:.1f} M Gaussians/sec")

# Test 2: Write with unpacked arrays (old way)
print("\n[3] Testing unpacked array write (old way)...")
output_file2 = Path(tempfile.mktemp(suffix=".ply"))

times2 = []
for i in range(20):
    start = time.perf_counter()
    gsply.plywrite(str(output_file2), *data.unpack())
    elapsed = (time.perf_counter() - start) * 1000
    times2.append(elapsed)
    output_file2.unlink()

mean_time2 = np.mean(times2)
print(f"    Mean time: {mean_time2:.2f} ms")
print(f"    Throughput: {len(data) / mean_time2 * 1000 / 1e6:.1f} M Gaussians/sec")

# Test 3: Verify outputs are identical
print("\n[4] Verifying byte-for-byte equivalence...")
gsply.plywrite(str(output_file), data)
gsply.plywrite(str(output_file2), *data.unpack())

with open(output_file, "rb") as f1, open(output_file2, "rb") as f2:
    bytes1 = f1.read()
    bytes2 = f2.read()

if bytes1 == bytes2:
    print("    [OK] Both methods produce identical output")
else:
    print(f"    [FAIL] Outputs differ! {len(bytes1)} vs {len(bytes2)} bytes")

# Cleanup
output_file.unlink()
output_file2.unlink()

print("\n" + "=" * 80)
print(f"RESULT: GSData API is {mean_time2 / mean_time:.1f}x speed (unpacked / direct)")
if mean_time < mean_time2:
    print(f"        Direct GSData write is FASTER ({mean_time:.1f}ms vs {mean_time2:.1f}ms)")
else:
    print(f"        Both methods have similar performance")
print("=" * 80)
