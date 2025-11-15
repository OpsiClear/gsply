"""Test zero-copy detection with verbose output."""

import numpy as np
import gsply

# Read real file
data = gsply.plyread("D:/4D/all_plys/frame_0.ply")

print("Testing zero-copy detection (verbose)...")
means, scales, quats, opacities, sh0, shN = data.unpack()

print("\n[Step 1] Check if arrays share memory")
print(f"  shares_memory(means, scales): {np.shares_memory(means, scales)}")
print(f"  shares_memory(means, quats): {np.shares_memory(means, quats)}")
print(f"  shares_memory(means, opacities): {np.shares_memory(means, opacities)}")
print(f"  shares_memory(means, sh0): {np.shares_memory(means, sh0)}")
if shN is not None and shN.size > 0:
    print(f"  shares_memory(means, shN): {np.shares_memory(means, shN)}")
else:
    print(f"  shN is None or empty (size={shN.size if shN is not None else 'None'})")

print("\n[Step 2] Find base array")
base = means
depth = 0
while base.base is not None:
    depth += 1
    base = base.base
    print(f"  Depth {depth}: shape={base.shape}, dtype={base.dtype}, ndim={base.ndim}")

print(f"\n[Step 3] Base array properties")
print(f"  base.ndim: {base.ndim}")
print(f"  base.shape: {base.shape}")
print(f"  base.dtype: {base.dtype}")
print(f"  base.flags.c_contiguous: {base.flags.c_contiguous}")

print(f"\n[Step 4] Shape verification")
num_gaussians = means.shape[0]
expected_props = 14 if shN is None or shN.size == 0 else (14 + shN.shape[1])
print(f"  num_gaussians: {num_gaussians}")
print(f"  expected_props: {expected_props}")
print(f"  base.shape: {base.shape}")
print(f"  Match: {base.shape == (num_gaussians, expected_props)}")

print(f"\n[Step 5] Memory address verification")
print(f"  means.__array_interface__['data'][0]: {means.__array_interface__['data'][0]}")
print(f"  base.__array_interface__['data'][0]: {base.__array_interface__['data'][0]}")
print(f"  Match: {means.__array_interface__['data'][0] == base.__array_interface__['data'][0]}")

# Check if base is 2D
if base.ndim == 2:
    print("\n[PASS] Base is 2D")
else:
    print(f"\n[FAIL] Base is not 2D (ndim={base.ndim})")

# Check if we can get to data._base
print(f"\n[Comparison with data._base]")
print(f"  data._base shape: {data._base.shape}")
print(f"  data._base dtype: {data._base.dtype}")
print(f"  base is data._base: {base is data._base}")
print(f"  shares_memory(base, data._base): {np.shares_memory(base, data._base)}")
