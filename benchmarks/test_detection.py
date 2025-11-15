"""Test zero-copy detection with updated logic."""

import gsply
from gsply.writer import _can_use_zero_copy_write

# Read real file
data = gsply.plyread("D:/4D/all_plys/frame_0.ply")

print("Testing zero-copy detection...")
means, scales, quats, opacities, sh0, shN = data.unpack()

can_zero_copy, base = _can_use_zero_copy_write(means, scales, quats, opacities, sh0, shN)

print(f"Can use zero-copy: {can_zero_copy}")
if base is not None:
    print(f"Base shape: {base.shape}")
    print(f"Base dtype: {base.dtype}")
    print(f"Base is data._base: {base is data._base}")
else:
    print("No base array found")
