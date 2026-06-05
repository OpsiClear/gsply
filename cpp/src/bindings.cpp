#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
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

// Zero-copy strided view into an externally-owned float buffer. `owner` keeps
// the buffer alive for the array's lifetime; strides are in float elements.
nb::ndarray<nb::numpy, float> strided_view(float* base, std::vector<size_t> shape,
                                           std::vector<int64_t> strides, nb::handle owner) {
  return nb::ndarray<nb::numpy, float>(base, shape.size(), shape.data(), owner, strides.data());
}

// Adopt a new[]-allocated buffer into a numpy array (no copy); the capsule
// delete[]s it when Python drops the last reference.
nb::ndarray<nb::numpy, float> adopt_numpy(float* data, std::vector<size_t> shape) {
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
  return nb::ndarray<nb::numpy, float>(data, shape.size(), shape.data(), owner);
}

// Build a non-owning view over the contiguous numpy inputs (no copy). The
// caller must keep the source arrays (including shN) alive across the write.
GSView make_view(const InArray& means, const InArray& scales, const InArray& quats,
                 const InArray& opacities, const InArray& sh0, const float* shN, int sh_dim) {
  GSView v;
  v.n = static_cast<int64_t>(means.shape(0));
  v.sh_dim = sh_dim;
  v.means = means.data();
  v.scales = scales.data();
  v.quats = quats.data();
  v.opacities = opacities.data();
  v.sh0 = sh0.data();
  v.shN = shN;
  return v;
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

// Read a GS PLY into a numpy dict. The float payload is read once into an
// aligned heap buffer; for the canonical layout (each attribute's columns
// contiguous and in order) the returned arrays are zero-copy strided views into
// that buffer, matching gsply.plyread. Non-contiguous layouts fall back to a
// gather copy.
nb::dict read_ply_dict(const std::string& path) {
  // fread (not std::ifstream) — MSVC's stream read is ~3x slower for bulk loads.
  // Read only enough bytes to parse the header, then stream the float payload
  // directly into an aligned float[] buffer. PLY headers have arbitrary byte
  // length, so pointing NumPy at raw + data_offset often creates unaligned
  // arrays that make later ndarray.tofile()/numeric operations slower.
  std::error_code ec;
  const std::uintmax_t fsz = std::filesystem::file_size(path, ec);
  if (ec) throw std::runtime_error("Cannot open PLY: " + path);
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error("Cannot open PLY: " + path);
  const size_t fsize = static_cast<size_t>(fsz);
  float* P = nullptr;
  try {
    static constexpr char marker[] = "end_header";
    static constexpr size_t marker_len = sizeof(marker) - 1;
    static constexpr size_t chunk = 4096;
    std::vector<char> header_buf;
    header_buf.reserve(chunk);
    bool found = false;
    while (!found) {
      const size_t old = header_buf.size();
      if (old >= fsize) throw std::runtime_error("PLY header end marker not found: " + path);
      const size_t want = std::min(chunk, fsize - old);
      header_buf.resize(old + want);
      const size_t got = std::fread(header_buf.data() + old, 1, want, fp);
      if (got != want) throw std::runtime_error("Short read on PLY: " + path);
      const size_t start = old > marker_len ? old - marker_len : 0;
      for (size_t i = start; i + marker_len <= header_buf.size(); ++i) {
        if (std::memcmp(header_buf.data() + i, marker, marker_len) == 0) {
          found = true;
          break;
        }
      }
    }

    PlyHeader h = parse_ply_header(header_buf.data(), header_buf.size(), path);
    PlyLayout L = compute_ply_layout(h, path);
    const int64_t n = h.n;
    const int rs = L.n_props;  // row stride (float elements)
    const size_t payload_bytes = static_cast<size_t>(n) * rs * sizeof(float);
    const size_t need = h.data_offset + payload_bytes;
    if (fsize < need) throw std::runtime_error("PLY truncated: " + path);
    P = new float[payload_bytes ? payload_bytes / sizeof(float) : 1];
    const size_t preloaded =
        header_buf.size() > h.data_offset
            ? std::min(header_buf.size() - h.data_offset, payload_bytes)
            : 0;
    if (preloaded > 0) {
      std::memcpy(reinterpret_cast<char*>(P), header_buf.data() + h.data_offset, preloaded);
    }
    const size_t remaining = payload_bytes - preloaded;
    if (remaining > 0) {
      const size_t got = std::fread(reinterpret_cast<char*>(P) + preloaded, 1, remaining, fp);
      if (got != remaining) throw std::runtime_error("Short read on PLY: " + path);
    }
    std::fclose(fp);
    fp = nullptr;

    const auto c = [&](const char* k) { return L.col.at(k); };
    const auto adj3 = [&](const char* a, const char* b, const char* cc) {
      return c(b) == c(a) + 1 && c(cc) == c(a) + 2;
    };
    bool viewable = adj3("x", "y", "z") && adj3("f_dc_0", "f_dc_1", "f_dc_2") &&
                    adj3("scale_0", "scale_1", "scale_2") &&
                    (c("rot_1") == c("rot_0") + 1 && c("rot_2") == c("rot_0") + 2 &&
                     c("rot_3") == c("rot_0") + 3);
    int r0 = L.sh_dim > 0 ? c("f_rest_0") : 0;
    for (int j = 0; viewable && j < L.sh_dim * 3; ++j) {
      if (c(("f_rest_" + std::to_string(j)).c_str()) != r0 + j) viewable = false;
    }

    if (viewable) {
      float* payload = P;
      nb::capsule owner(payload, [](void* p) noexcept { delete[] static_cast<float*>(p); });
      P = nullptr;  // capsule owns it now; keep catch's delete[] from double-freeing
      const size_t un = static_cast<size_t>(n);
      nb::dict out;
      out["_base"] = strided_view(payload, {un, static_cast<size_t>(rs)}, {rs, 1}, owner);
      out["means"] = strided_view(payload + c("x"), {un, 3}, {rs, 1}, owner);
      out["scales"] = strided_view(payload + c("scale_0"), {un, 3}, {rs, 1}, owner);
      out["quats"] = strided_view(payload + c("rot_0"), {un, 4}, {rs, 1}, owner);
      out["opacities"] = strided_view(payload + c("opacity"), {un}, {rs}, owner);
      out["sh0"] = strided_view(payload + c("f_dc_0"), {un, 3}, {rs, 1}, owner);
      // f_rest is channel-major [3,K] per row; present as [N,K,3] via strides
      // (coeff stride 1 element, channel stride sh_dim elements) — no transpose.
      if (L.sh_dim > 0) {
        out["shN"] = strided_view(payload + r0, {un, static_cast<size_t>(L.sh_dim), 3},
                                  {rs, 1, static_cast<int64_t>(L.sh_dim)}, owner);
      } else {
        out["shN"] = nb::none();
      }
      return out;  // payload ownership transferred to the capsule
    }

    // Rare non-canonical column order: gather-decode then copy out.
    GSData d = decode_ply_payload(h, L, P, path);
    delete[] P;
    P = nullptr;
    return gsdata_to_dict(d);
  } catch (...) {
    if (fp) std::fclose(fp);
    delete[] P;
    throw;
  }
}

// Read an SPZ into a numpy dict, decoding directly into adopted output buffers
// (no GSData->numpy copy). Handles both gzip (v1-3) and NGSP v4 containers; the
// decoded section buffer is freed after decode.
nb::dict read_spz_dict(const std::string& path) {
  const SpzLoaded loaded = load_spz(path);
  const SpzInfo& info = loaded.info;
  const uint8_t* sections = loaded.buffer.data() + loaded.base;
  const size_t n = static_cast<size_t>(info.n);
  const int K = info.sh_dim;
  float* means = new float[n * 3];
  float* scales = new float[n * 3];
  float* quats = new float[n * 4];
  float* opac = new float[n];
  float* sh0 = new float[n * 3];
  float* shN = K > 0 ? new float[n * static_cast<size_t>(K) * 3] : nullptr;
  try {
    decode_spz_body(sections, info, means, scales, quats, opac, sh0, shN);
  } catch (...) {
    delete[] means;
    delete[] scales;
    delete[] quats;
    delete[] opac;
    delete[] sh0;
    delete[] shN;
    throw;
  }
  nb::dict out;
  out["means"] = adopt_numpy(means, {n, 3});
  out["scales"] = adopt_numpy(scales, {n, 3});
  out["quats"] = adopt_numpy(quats, {n, 4});
  out["opacities"] = adopt_numpy(opac, {n});
  out["sh0"] = adopt_numpy(sh0, {n, 3});
  if (K > 0) {
    out["shN"] = adopt_numpy(shN, {n, static_cast<size_t>(K), 3});
  } else {
    out["shN"] = nb::none();
  }
  return out;
}

}  // namespace

NB_MODULE(gsply_cpp, m) {
  m.doc() = "C++ parity implementation of gsply (Gaussian-splat PLY/SPZ I/O).";

  m.def("read_ply", &read_ply_dict, nb::arg("path"),
        "Read an uncompressed GS PLY -> dict (means, scales, quats, opacities, sh0, shN). "
        "Zero-copy strided views into the file buffer for canonical layouts.");
  m.def(
      "write_ply",
      [](const std::string& p, InArray me, InArray sc, InArray q, InArray op, InArray s0,
         nb::object sN) {
        const float* shN = nullptr;
        int sh_dim = 0;
        InArray a;  // keep the shN array alive across the write
        if (!sN.is_none()) {
          a = nb::cast<InArray>(sN);
          sh_dim = static_cast<int>(a.shape(1));  // [n, sh_dim, 3]
          shN = a.data();
        }
        write_ply(p, make_view(me, sc, q, op, s0, shN, sh_dim));
      },
      nb::arg("path"), nb::arg("means"), nb::arg("scales"), nb::arg("quats"), nb::arg("opacities"),
      nb::arg("sh0"), nb::arg("shN") = nb::none(),
      "Write an uncompressed GS PLY (canonical INRIA property order).");

  m.def("read_spz", &read_spz_dict, nb::arg("path"),
        "Read a Niantic SPZ (gzip v1/v2/v3 or NGSP v4 zstd) -> dict (means, scales, quats, "
        "opacities, sh0, shN). Decodes directly into the output arrays (no intermediate copy).");
  m.def(
      "write_spz",
      [](const std::string& p, InArray me, InArray sc, InArray q, InArray op, InArray s0,
         nb::object sN, int fractional_bits, int version, int level) {
        const float* shN = nullptr;
        int sh_dim = 0;
        InArray a;  // keep the shN array alive across the write
        if (!sN.is_none()) {
          a = nb::cast<InArray>(sN);
          sh_dim = static_cast<int>(a.shape(1));  // [n, sh_dim, 3]
          shN = a.data();
        }
        write_spz(p, make_view(me, sc, q, op, s0, shN, sh_dim), fractional_bits, version, level);
      },
      nb::arg("path"), nb::arg("means"), nb::arg("scales"), nb::arg("quats"), nb::arg("opacities"),
      nb::arg("sh0"), nb::arg("shN") = nb::none(), nb::arg("fractional_bits") = 12,
      nb::arg("version") = 3, nb::arg("level") = -1,
      "Write a Niantic SPZ. version=3 gzip (smallest-three quats) or 4 NGSP/zstd. "
      "level<0 picks the per-codec default (gzip 6, zstd 12).");
}
