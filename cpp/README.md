# Bundled gsply C++ Backend

This directory contains the C++ parity implementation exposed as the top-level
Python module `gsply_cpp` via nanobind. It is built into the root `gsply`
platform wheels; it is not a separate PyPI project.

## Status

| Path | Status |
|------|--------|
| Uncompressed PLY read/write | implemented (`read_ply` / `write_ply`), zero-copy reads |
| SPZ read/write | implemented (`read_spz` / `write_spz`), gzip v1-v3 + NGSP v4 (zstd) |
| Compressed PLY (PlayCanvas) | planned |

## Build

Default local/source builds are Python-only:

```bash
pip install .
```

To compile the bundled C++ extension from the repository root, opt in with
scikit-build-core config settings:

```bash
pip install . -Cwheel.cmake=true -Ccmake.define.GSPLY_BUILD_CPP=ON
```

Release wheels are built by `.github/workflows/publish.yml` with the same CMake
option enabled through cibuildwheel.

## Use

The public runtime path is the root package backend selector:

```python
import gsply

gsply.use_backend("cpp")
data = gsply.read_spz("scene.spz")
gsply.write_spz("out.spz", data, version=4)
```

The low-level module can also be imported directly for parity tests and
benchmarks:

```python
import gsply_cpp

d = gsply_cpp.read_ply("scene.ply")
gsply_cpp.write_ply(
    "out.ply",
    d["means"],
    d["scales"],
    d["quats"],
    d["opacities"],
    d["sh0"],
    d["shN"],
)
```

Parity is verified against `gsply` in `cpp/tests/test_parity.py`.

## Releasing

Root `gsply` releases build and publish:

- an sdist, whose default build remains Python-only
- CPython 3.10-3.13 platform wheels for Linux x86_64, Windows AMD64, and macOS
  arm64, each containing `gsply_cpp`

The release workflow verifies the built wheel artifacts on Ubuntu and Windows
with `pip install gsply` before publishing. After publication, run
`.github/workflows/verify-pip.yml` to verify the public PyPI install path.
