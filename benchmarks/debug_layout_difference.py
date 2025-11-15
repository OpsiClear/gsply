"""Debug the layout difference between consolidate() and PLY writer."""

import tempfile
from pathlib import Path

import numpy as np

import gsply
from gsply.gsdata import GSData

# Small test case for easier debugging
np.random.seed(42)
num_gaussians = 10
sh_degree = 3  # Test SH3

means = np.random.randn(num_gaussians, 3).astype(np.float32)
scales = np.random.randn(num_gaussians, 3).astype(np.float32)
quats = np.random.randn(num_gaussians, 4).astype(np.float32)
quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
opacities = np.random.rand(num_gaussians).astype(np.float32)
sh0 = np.random.randn(num_gaussians, 3).astype(np.float32)

if sh_degree > 0:
    sh_coeffs = [0, 9, 24, 45]
    num_coeffs = sh_coeffs[sh_degree]
    shN = np.random.randn(num_gaussians, num_coeffs // 3, 3).astype(np.float32)
else:
    shN = np.empty((num_gaussians, 0, 3), dtype=np.float32)

data = GSData(means=means, scales=scales, quats=quats, opacities=opacities, sh0=sh0, shN=shN, _base=None)

# Write without _base (standard path)
temp1 = Path(tempfile.mktemp(suffix=".ply"))
gsply.plywrite(str(temp1), data)

# Consolidate and write with _base
data_consolidated = data.consolidate()
temp2 = Path(tempfile.mktemp(suffix=".ply"))
gsply.plywrite(str(temp2), data_consolidated)

# Read both files
with open(temp1, "rb") as f:
    file1_data = f.read()

with open(temp2, "rb") as f:
    file2_data = f.read()

print(f"File 1 (standard path) size: {len(file1_data)}")
print(f"File 2 (consolidated) size: {len(file2_data)}")
print(f"Difference: {abs(len(file1_data) - len(file2_data))} bytes")

# Find header end (should be "end_header\n")
header1_end = file1_data.find(b'end_header\n') + len(b'end_header\n')
header2_end = file2_data.find(b'end_header\n') + len(b'end_header\n')

print(f"\nHeader 1 ends at byte: {header1_end}")
print(f"Header 2 ends at byte: {header2_end}")

# Compare headers
header1 = file1_data[:header1_end].decode('ascii')
header2 = file2_data[:header2_end].decode('ascii')

if header1 == header2:
    print("\n[OK] Headers are IDENTICAL")
else:
    print("\n[FAIL] Headers DIFFER:")
    print("\nHeader 1 (standard):")
    print(header1)
    print("\nHeader 2 (consolidated):")
    print(header2)

# Compare binary data
binary1 = file1_data[header1_end:]
binary2 = file2_data[header2_end:]

print(f"\nBinary data 1 size: {len(binary1)} bytes")
print(f"Binary data 2 size: {len(binary2)} bytes")

if binary1 == binary2:
    print("[OK] Binary data is IDENTICAL")
else:
    print("[FAIL] Binary data DIFFERS")

    # Show first few floats from each file
    num_props = 14 if sh_degree == 0 else (14 + [0, 9, 24, 45][sh_degree])
    arr1 = np.frombuffer(binary1, dtype=np.float32).reshape(num_gaussians, num_props)
    arr2 = np.frombuffer(binary2, dtype=np.float32).reshape(num_gaussians, num_props)

    shN_size = [0, 9, 24, 45][sh_degree]
    opacity_idx = 6 + shN_size

    print(f"\nFirst Gaussian from file 1 (standard path):")
    print(f"  means:     {arr1[0, 0:3]}")
    print(f"  sh0:       {arr1[0, 3:6]}")
    if shN_size > 0:
        print(f"  shN:       {arr1[0, 6:6+shN_size][:3]}...")  # First 3 coeffs
    print(f"  opacity:   {arr1[0, opacity_idx]}")
    print(f"  scales:    {arr1[0, opacity_idx+1:opacity_idx+4]}")
    print(f"  quats:     {arr1[0, opacity_idx+4:opacity_idx+8]}")

    print(f"\nFirst Gaussian from file 2 (consolidated):")
    print(f"  [0:3]:     {arr2[0, 0:3]}")
    print(f"  [3:6]:     {arr2[0, 3:6]}")
    if shN_size > 0:
        print(f"  [6:{6+shN_size}]:   {arr2[0, 6:6+shN_size][:3]}...")  # First 3 coeffs
    print(f"  [{opacity_idx}]:     {arr2[0, opacity_idx]}")
    print(f"  [{opacity_idx+1}:{opacity_idx+4}]: {arr2[0, opacity_idx+1:opacity_idx+4]}")
    print(f"  [{opacity_idx+4}:{opacity_idx+8}]: {arr2[0, opacity_idx+4:opacity_idx+8]}")

    print(f"\nOriginal data (Python):")
    print(f"  means:     {means[0]}")
    print(f"  sh0:       {sh0[0]}")
    print(f"  opacity:   {opacities[0]}")
    print(f"  scales:    {scales[0]}")
    print(f"  quats:     {quats[0]}")

    # Show what consolidate() created
    print(f"\nconsolidate() _base[0]:")
    print(f"  [0:3]:     {data_consolidated._base[0, 0:3]}")
    print(f"  [3:6]:     {data_consolidated._base[0, 3:6]}")
    if shN_size > 0:
        print(f"  [6:{6+shN_size}]:   {data_consolidated._base[0, 6:6+shN_size][:3]}...")
    print(f"  [{opacity_idx}]:     {data_consolidated._base[0, opacity_idx]}")
    print(f"  [{opacity_idx+1}:{opacity_idx+4}]: {data_consolidated._base[0, opacity_idx+1:opacity_idx+4]}")
    print(f"  [{opacity_idx+4}:{opacity_idx+8}]: {data_consolidated._base[0, opacity_idx+4:opacity_idx+8]}")

temp1.unlink()
temp2.unlink()
