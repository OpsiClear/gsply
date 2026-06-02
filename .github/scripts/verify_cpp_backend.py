import tempfile
from pathlib import Path

import gsply_cpp
import numpy as np

import gsply


def _sample_data(n: int = 64) -> gsply.GSData:
    rng = np.random.default_rng(1234)
    means = rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)
    scales = rng.uniform(-8.0, -2.0, (n, 3)).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.uniform(-3.0, 4.0, n).astype(np.float32)
    sh0 = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    shn = rng.uniform(-0.4, 0.4, (n, 15, 3)).astype(np.float32)
    return gsply.GSData.from_arrays(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shn,
        format="ply",
    )


def main() -> None:
    assert gsply_cpp is not None
    gsply.use_backend("cpp")
    assert gsply.active_backend() == "cpp"

    data = _sample_data()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        ply_path = tmp_path / "roundtrip.ply"
        gsply.plywrite(ply_path, data)
        ply_out = gsply.plyread(ply_path)
        assert len(ply_out) == len(data)
        np.testing.assert_array_equal(np.asarray(ply_out.means), np.asarray(data.means))

        spz_path = tmp_path / "roundtrip.spz"
        gsply.write_spz(spz_path, data, version=4)
        spz_out = gsply.read_spz(spz_path)
        assert len(spz_out) == len(data)
        np.testing.assert_allclose(np.asarray(spz_out.means), np.asarray(data.means), atol=2e-4)

    print("pip install gsply C++ backend e2e OK")


if __name__ == "__main__":
    main()
