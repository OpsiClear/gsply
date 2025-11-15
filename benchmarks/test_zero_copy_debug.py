"""Debug zero-copy detection."""

import gsply
import numpy as np

# Read real file
data = gsply.plyread("D:/4D/all_plys/frame_0.ply")

print("=" * 80)
print("DEBUGGING ZERO-COPY DETECTION")
print("=" * 80)

print("\n[Data Structure]")
print(f"  data._base shape: {data._base.shape if data._base is not None else None}")
print(f"  data._base dtype: {data._base.dtype if data._base is not None else None}")
print(f"  data._base id: {id(data._base) if data._base is not None else None}")

print("\n[Field: means]")
print(f"  means shape: {data.means.shape}")
print(f"  means dtype: {data.means.dtype}")
print(f"  means.base is not None: {data.means.base is not None}")
print(f"  means.base id: {id(data.means.base) if data.means.base is not None else None}")
print(f"  means.base is data._base: {data.means.base is data._base if data.means.base is not None else False}")
print(f"  means.flags.c_contiguous: {data.means.flags.c_contiguous}")

print("\n[Field: scales]")
print(f"  scales.base is data._base: {data.scales.base is data._base if data.scales.base is not None else False}")

print("\n[Field: quats]")
print(f"  quats.base is data._base: {data.quats.base is data._base if data.quats.base is not None else False}")

print("\n[Field: opacities]")
print(f"  opacities shape: {data.opacities.shape}")
print(f"  opacities.base is not None: {data.opacities.base is not None}")
print(f"  opacities.base is data._base: {data.opacities.base is data._base if data.opacities.base is not None else False}")

print("\n[Field: sh0]")
print(f"  sh0.base is data._base: {data.sh0.base is data._base if data.sh0.base is not None else False}")

print("\n[Field: shN]")
if data.shN is not None:
    print(f"  shN.base is not None: {data.shN.base is not None}")
    print(f"  shN.base is data._base: {data.shN.base is data._base if data.shN.base is not None else False}")
else:
    print(f"  shN is None")

# Test after unpacking
print("\n" + "=" * 80)
print("AFTER UNPACKING")
print("=" * 80)

means, scales, quats, opacities, sh0, shN = data.unpack()

print("\n[Unpacked: means]")
print(f"  means.base is not None: {means.base is not None}")
if means.base is not None:
    print(f"  means.base is data._base: {means.base is data._base}")
    print(f"  means.base shape: {means.base.shape}")

print("\n[Unpacked: opacities]")
print(f"  opacities shape: {opacities.shape}")
print(f"  opacities.base is not None: {opacities.base is not None}")
if opacities.base is not None:
    print(f"  opacities.base is data._base: {opacities.base is data._base}")

# Check what validate_and_normalize does
print("\n" + "=" * 80)
print("AFTER VALIDATE_AND_NORMALIZE")
print("=" * 80)

from gsply.writer import _validate_and_normalize_inputs

means2, scales2, quats2, opacities2, sh02, shN2 = _validate_and_normalize_inputs(
    means, scales, quats, opacities, sh0, shN, validate=False
)

print("\n[After normalize: means]")
print(f"  means2 is means: {means2 is means}")
print(f"  means2.base is not None: {means2.base is not None}")
if means2.base is not None:
    print(f"  means2.base is data._base: {means2.base is data._base}")

print("\n[After normalize: opacities]")
print(f"  opacities2 is opacities: {opacities2 is opacities}")
print(f"  opacities2 shape: {opacities2.shape}")
print(f"  opacities2.base is not None: {opacities2.base is not None}")
if opacities2.base is not None:
    print(f"  opacities2.base is data._base: {opacities2.base is data._base}")

print("\n[After normalize: shN]")
if shN2 is not None:
    print(f"  shN2 is shN: {shN2 is shN}")
    print(f"  shN2 shape: {shN2.shape}")
    print(f"  shN2.base is not None: {shN2.base is not None}")
    if shN2.base is not None:
        print(f"  shN2.base is data._base: {shN2.base is data._base}")
        print(f"  shN2.base shape: {shN2.base.shape}")
        print(f"  data._base shape: {data._base.shape}")
