#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstring>
#include <vector>

#include "ply.hpp"
#include "spz.hpp"

namespace nb = nanobind;
using namespace gsplycpp;

namespace {

// numpy float32 array, C-contiguous, CPU (input).
using InArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

// Build an owning numpy array from a vector (copies; freed when Python drops it).
nb::ndarray<nb::numpy, float> to_numpy(const std::vector<float>& v,
                                       std::initializer_list<size_t> shape) {
  float* data = new float[v.size() ? v.size() : 1];
  if (!v.empty()) std::memcpy(data, v.data(), v.size() * sizeof(float));
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
  std::vector<size_t> s(shape);
  return nb::ndarray<nb::numpy, float>(data, s.size(), s.data(), owner);
}

void copy_in(const InArray& a, std::vector<float>& out) {
  size_t total = 1;
  for (size_t i = 0; i < a.ndim(); ++i) total *= a.shape(i);
  out.resize(total);
  if (total) std::memcpy(out.data(), a.data(), total * sizeof(float));
}

nb::dict gsdata_to_dict(const GSData& d) {
  const size_t n = static_cast<size_t>(d.n);
  nb::dict out;
  out["means"] = to_numpy(d.means, {n, 3});
  out["scales"] = to_numpy(d.scales, {n, 3});
  out["quats"] = to_numpy(d.quats, {n, 4});
  out["opacities"] = to_numpy(d.opacities, {n});
  out["sh0"] = to_numpy(d.sh0, {n, 3});
  if (d.sh_dim > 0) {
    out["shN"] = to_numpy(d.shN, {n, static_cast<size_t>(d.sh_dim), 3});
  } else {
    out["shN"] = nb::none();
  }
  return out;
}

GSData args_to_gsdata(InArray means, InArray scales, InArray quats, InArray opacities, InArray sh0,
                      nb::object shN) {
  GSData d;
  d.n = static_cast<int64_t>(means.shape(0));
  copy_in(means, d.means);
  copy_in(scales, d.scales);
  copy_in(quats, d.quats);
  copy_in(opacities, d.opacities);
  copy_in(sh0, d.sh0);
  if (shN.is_none()) {
    d.sh_dim = 0;
  } else {
    InArray a = nb::cast<InArray>(shN);
    d.sh_dim = static_cast<int>(a.shape(1));  // [n, sh_dim, 3]
    copy_in(a, d.shN);
  }
  return d;
}

}  // namespace

NB_MODULE(gsply_cpp, m) {
  m.doc() = "C++ parity implementation of gsply (Gaussian-splat PLY/SPZ I/O).";

  m.def("read_ply", [](const std::string& p) { return gsdata_to_dict(read_ply(p)); },
        nb::arg("path"),
        "Read an uncompressed GS PLY -> dict (means, scales, quats, opacities, sh0, shN).");
  m.def(
      "write_ply",
      [](const std::string& p, InArray me, InArray sc, InArray q, InArray op, InArray s0,
         nb::object sN) { write_ply(p, args_to_gsdata(me, sc, q, op, s0, sN)); },
      nb::arg("path"), nb::arg("means"), nb::arg("scales"), nb::arg("quats"), nb::arg("opacities"),
      nb::arg("sh0"), nb::arg("shN") = nb::none(),
      "Write an uncompressed GS PLY (canonical INRIA property order).");

  m.def("read_spz", [](const std::string& p) { return gsdata_to_dict(read_spz(p)); },
        nb::arg("path"),
        "Read a Niantic SPZ (gzip v1/v2/v3) -> dict (means, scales, quats, opacities, sh0, shN).");
  m.def(
      "write_spz",
      [](const std::string& p, InArray me, InArray sc, InArray q, InArray op, InArray s0,
         nb::object sN, int fractional_bits) {
        write_spz(p, args_to_gsdata(me, sc, q, op, s0, sN), fractional_bits);
      },
      nb::arg("path"), nb::arg("means"), nb::arg("scales"), nb::arg("quats"), nb::arg("opacities"),
      nb::arg("sh0"), nb::arg("shN") = nb::none(), nb::arg("fractional_bits") = 12,
      "Write a Niantic SPZ (gzip v3, smallest-three quaternions).");
}
