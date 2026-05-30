# gsply-cpp

A C++ parity reimplementation of [gsply](../)'s Gaussian-splat I/O, exposed as a
Python module (`gsply_cpp`) via nanobind. The goal is byte/value parity with the
pure-Python `gsply` so it can serve as an accelerated, compiler-built backend.

## Status

| Path | Status |
|------|--------|
| Uncompressed PLY read/write | ✅ implemented (`read_ply` / `write_ply`), zero-copy reads |
| SPZ read/write | ✅ implemented (`read_spz` / `write_spz`); gzip v1–v3 + NGSP v4 (zstd) |
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

## Prebuilt wheels & releasing

`.github/workflows/publish-cpp.yml` builds **abi3 (cp312) wheels** for Linux,
Windows and macOS (x86_64 + arm64) with cibuildwheel and publishes to PyPI via
trusted publishing. One abi3 wheel per platform covers CPython 3.12+; 3.10/3.11
fall back to the sdist (source build, needs a compiler + CMake).

To cut a release:

1. Bump `version` in `cpp/pyproject.toml`.
2. **One-time PyPI setup**: add a Trusted Publisher for the `gsply-cpp` project at
   <https://pypi.org/manage/account/publishing/> — Owner `OpsiClear`, Repository
   `gsply`, Workflow `publish-cpp.yml`, Environment `pypi` (use a *pending
   publisher* if the project doesn't exist yet; the first run creates it).
3. Tag and push: `git tag cpp-v<X.Y.Z> && git push origin cpp-v<X.Y.Z>`.

`workflow_dispatch` builds + smoke-tests the wheels **without** publishing (for
verification). Once `gsply-cpp` is on PyPI, `pip install gsply[cpp]` installs the
prebuilt backend; enable it with `gsply.use_backend("cpp")`.
