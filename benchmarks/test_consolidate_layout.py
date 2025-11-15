"""Test if GSData.consolidate() creates correct _base layout for writing."""

import tempfile
from pathlib import Path

import numpy as np

import gsply
from gsply.gsdata import GSData

# Generate synthetic data
np.random.seed(42)
num_gaussians = 1000
sh_degree = 3

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

# Create GSData without _base
data_no_base = GSData(
    means=means,
    scales=scales,
    quats=quats,
    opacities=opacities,
    sh0=sh0,
    shN=shN,
    _base=None
)

print(f"Testing consolidate() layout with {num_gaussians} Gaussians, SH degree {sh_degree}")
print(f"shN shape: {shN.shape}")

# Test 1: Write without _base (standard path)
temp1 = Path(tempfile.mktemp(suffix=".ply"))
gsply.plywrite(str(temp1), data_no_base)
print(f"\n[1] Written without _base: {temp1.stat().st_size} bytes")

# Test 2: Consolidate then write (should use zero-copy)
data_with_base = data_no_base.consolidate()
print(f"\n[2] After consolidate():")
print(f"    _base is not None: {data_with_base._base is not None}")
print(f"    _base shape: {data_with_base._base.shape if data_with_base._base is not None else None}")

temp2 = Path(tempfile.mktemp(suffix=".ply"))
gsply.plywrite(str(temp2), data_with_base)
print(f"    Written with _base: {temp2.stat().st_size} bytes")

# Test 3: Compare byte-for-byte
with open(temp1, "rb") as f1, open(temp2, "rb") as f2:
    bytes1 = f1.read()
    bytes2 = f2.read()

if bytes1 == bytes2:
    print(f"\n[3] [OK] Files are IDENTICAL")
    print(f"    consolidate() creates correct _base layout for PLY writes!")
else:
    print(f"\n[3] [FAIL] Files DIFFER")
    print(f"    File 1: {len(bytes1)} bytes")
    print(f"    File 2: {len(bytes2)} bytes")

    # Find first difference
    for i, (b1, b2) in enumerate(zip(bytes1, bytes2)):
        if b1 != b2:
            print(f"    First difference at byte {i}: {b1} vs {b2}")
            break

# Test 4: Verify roundtrip
data_roundtrip = gsply.plyread(str(temp2))
print(f"\n[4] Roundtrip verification:")
print(f"    Original means[0]: {means[0]}")
print(f"    Roundtrip means[0]: {data_roundtrip.means[0]}")
print(f"    Match: {np.allclose(means, data_roundtrip.means)}")

# Cleanup
temp1.unlink()
temp2.unlink()
