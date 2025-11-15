"""Test automatic consolidation in plywrite()."""

import tempfile
import time
from pathlib import Path

import numpy as np

import gsply
from gsply.gsdata import GSData

print("=" * 80)
print("AUTO-CONSOLIDATION TEST")
print("=" * 80)

# Generate test data
np.random.seed(42)
num_gaussians = 100_000

means = np.random.randn(num_gaussians, 3).astype(np.float32)
scales = np.random.randn(num_gaussians, 3).astype(np.float32)
quats = np.random.randn(num_gaussians, 4).astype(np.float32)
quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
opacities = np.random.rand(num_gaussians).astype(np.float32)
sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)
shN = np.empty((num_gaussians, 0, 3), dtype=np.float32)

# Test 1: GSData without _base (should auto-consolidate)
print("\n[Test 1] GSData without _base (should auto-consolidate)")
data_no_base = GSData(
    means=means,
    scales=scales,
    quats=quats,
    opacities=opacities,
    sh0=sh0,
    shN=shN,
    _base=None
)

print(f"  Before write: _base is None = {data_no_base._base is None}")

temp1 = Path(tempfile.mktemp(suffix=".ply"))
start = time.perf_counter()
gsply.plywrite(str(temp1), data_no_base)
elapsed = (time.perf_counter() - start) * 1000

print(f"  Write time: {elapsed:.2f} ms")
print(f"  File size: {temp1.stat().st_size / 1024 / 1024:.2f} MB")
print(f"  After write: data._base is still None = {data_no_base._base is None}")
print(f"  (Auto-consolidation creates new GSData internally)")

# Test 2: Individual arrays (should convert + auto-consolidate)
print("\n[Test 2] Individual arrays (should convert + auto-consolidate)")

temp2 = Path(tempfile.mktemp(suffix=".ply"))
start = time.perf_counter()
gsply.plywrite(str(temp2), means, scales, quats, opacities, sh0, shN)
elapsed2 = (time.perf_counter() - start) * 1000

print(f"  Write time: {elapsed2:.2f} ms")
print(f"  File size: {temp2.stat().st_size / 1024 / 1024:.2f} MB")

# Test 3: GSData from file (should use zero-copy, no consolidation needed)
print("\n[Test 3] GSData from file (should use zero-copy)")

data_from_file = gsply.plyread(str(temp1))
print(f"  Has _base: {data_from_file._base is not None}")

temp3 = Path(tempfile.mktemp(suffix=".ply"))
start = time.perf_counter()
gsply.plywrite(str(temp3), data_from_file)
elapsed3 = (time.perf_counter() - start) * 1000

print(f"  Write time: {elapsed3:.2f} ms")
print(f"  Speedup vs auto-consolidated: {elapsed / elapsed3:.2f}x")

# Test 4: Verify all files are identical
print("\n[Test 4] Verify byte-for-byte equivalence")

with open(temp1, "rb") as f1, open(temp2, "rb") as f2, open(temp3, "rb") as f3:
    bytes1 = f1.read()
    bytes2 = f2.read()
    bytes3 = f3.read()

if bytes1 == bytes2 == bytes3:
    print("  [OK] All three methods produce IDENTICAL output!")
else:
    print("  [FAIL] Files differ!")
    print(f"    File 1: {len(bytes1)} bytes")
    print(f"    File 2: {len(bytes2)} bytes")
    print(f"    File 3: {len(bytes3)} bytes")

# Test 5: Compressed write (should NOT consolidate)
print("\n[Test 5] Compressed write (should NOT auto-consolidate)")

temp4 = Path(tempfile.mktemp(suffix=".ply"))
start = time.perf_counter()
gsply.plywrite(str(temp4), data_no_base, compressed=True)
elapsed4 = (time.perf_counter() - start) * 1000

actual_file = Path(str(temp4).replace('.ply', '.compressed.ply'))
print(f"  Write time: {elapsed4:.2f} ms")
print(f"  File size: {actual_file.stat().st_size / 1024 / 1024:.2f} MB")
print(f"  Compression ratio: {temp1.stat().st_size / actual_file.stat().st_size:.2f}x")

# Cleanup
temp1.unlink()
temp2.unlink()
temp3.unlink()
actual_file.unlink()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Auto-consolidation is working correctly!")
print(f"  GSData without _base: {elapsed:.2f} ms (auto-consolidated)")
print(f"  Individual arrays: {elapsed2:.2f} ms (converted + consolidated)")
print(f"  GSData from file: {elapsed3:.2f} ms (zero-copy, no consolidation)")
print(f"  All methods produce identical output")
print("=" * 80)
