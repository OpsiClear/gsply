"""Benchmark gsply (Python: numpy + numba) vs gsply_cpp (C++/nanobind)."""

import time
import tempfile
from pathlib import Path

import numpy as np

import gsply
import gsply_cpp


def make(n):
    rng = np.random.default_rng(0)
    means = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
    scales = rng.uniform(-8, -2, (n, 3)).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opac = rng.uniform(-3, 4, n).astype(np.float32)
    sh0 = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    shN = rng.uniform(-0.4, 0.4, (n, 15, 3)).astype(np.float32)
    return means, scales, quats, opac, sh0, shN


def bench(fn, iters):
    fn()  # warmup (JIT, caches)
    best = float("inf")
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    n = 400_000
    iters = 15
    means, scales, quats, opac, sh0, shN = make(n)
    data = gsply.GSData.from_arrays(
        means=means, scales=scales, quats=quats, opacities=opac, sh0=sh0, shN=shN, format="ply"
    )

    # Which gzip backend does the Python SPZ path use?
    try:
        import isal  # noqa: F401

        gz = "isal"
    except ImportError:
        gz = "gzip(stdlib)"

    tmp = Path(tempfile.mkdtemp())
    ply = str(tmp / "b.ply")
    spz = str(tmp / "b.spz")
    gsply.plywrite(ply, means, scales, quats, opac, sh0, shN)
    gsply.write_spz(spz, data)
    print(f"N={n:,} Gaussians (SH3)   iters={iters} (best-of)   python gzip backend: {gz}")
    print(
        f"  PLY file: {Path(ply).stat().st_size/1e6:.0f}MB   SPZ file: {Path(spz).stat().st_size/1e6:.0f}MB\n"
    )

    rows = []

    def add(name, py_fn, cpp_fn):
        py = bench(py_fn, iters)
        cpp = bench(cpp_fn, iters)
        rows.append((name, py * 1e3, cpp * 1e3, py / cpp, n / py / 1e6, n / cpp / 1e6))

    add("PLY read", lambda: gsply.plyread(ply), lambda: gsply_cpp.read_ply(ply))
    add("SPZ read", lambda: gsply.read_spz(spz), lambda: gsply_cpp.read_spz(spz))
    add(
        "PLY write",
        lambda: gsply.plywrite(str(tmp / "pw.ply"), means, scales, quats, opac, sh0, shN),
        lambda: gsply_cpp.write_ply(str(tmp / "cw.ply"), means, scales, quats, opac, sh0, shN),
    )
    add(
        "SPZ write",
        lambda: gsply.write_spz(str(tmp / "pw.spz"), data),
        lambda: gsply_cpp.write_spz(str(tmp / "cw.spz"), means, scales, quats, opac, sh0, shN),
    )

    print(f"{'op':<11}{'python ms':>11}{'cpp ms':>10}{'speedup':>9}{'py M/s':>9}{'cpp M/s':>9}")
    print("-" * 59)
    for name, pms, cms, sp, pm, cm in rows:
        tag = f"{sp:.2f}x" if sp >= 1 else f"{1/sp:.2f}x slower"
        print(f"{name:<11}{pms:>11.2f}{cms:>10.2f}{tag:>9}{pm:>9.1f}{cm:>9.1f}")


if __name__ == "__main__":
    main()
