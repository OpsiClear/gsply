# gsply-cpp

A C++ parity reimplementation of [gsply](../)'s Gaussian-splat I/O, exposed as a
Python module (`gsply_cpp`) via nanobind. The goal is byte/value parity with the
pure-Python `gsply` so it can serve as an accelerated, compiler-built backend.

## Status

| Path | Status |
|------|--------|
| Uncompressed PLY read/write | ✅ implemented (`read_ply` / `write_ply`) |
| SPZ read/write | ⏳ planned (zlib) |
| Compressed PLY (PlayCanvas) | ⏳ planned |

## Build

```bash
pip install .            # scikit-build-core + nanobind, needs a C++17 compiler
# or: pip install ./cpp  from the gsply repo root
```

## Use

```python
import gsply_cpp
d = gsply_cpp.read_ply("scene.ply")   # -> {"means","scales","quats","opacities","sh0","shN"}
gsply_cpp.write_ply("out.ply", d["means"], d["scales"], d["quats"],
                    d["opacities"], d["sh0"], d["shN"])
```

Parity is verified against `gsply` in `tests/test_parity.py`.
