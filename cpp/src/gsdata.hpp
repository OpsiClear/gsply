#pragma once

#include <cstdint>
#include <vector>

namespace gsplycpp {

// Parity with gsply's GSData in "ply" format: means linear, scales log-space,
// quats unit wxyz, opacities logit-space, sh0 SH DC, shN higher-order SH.
// Arrays are row-major flat; shapes are implied by `n` and `sh_dim`.
struct GSData {
  int64_t n = 0;       // number of Gaussians
  int sh_dim = 0;      // higher-order SH coeffs per channel: 0, 3, 8, or 15
  std::vector<float> means;      // [n, 3]
  std::vector<float> scales;     // [n, 3]
  std::vector<float> quats;      // [n, 4]  (w, x, y, z)
  std::vector<float> opacities;  // [n]
  std::vector<float> sh0;        // [n, 3]
  std::vector<float> shN;        // [n, sh_dim, 3]  (coeff-major, channel-inner)
};

// Non-owning view over GS arrays (same layout as GSData) for write paths that
// consume caller-owned buffers (e.g. numpy inputs) without copying. shN may be
// null when sh_dim == 0.
struct GSView {
  int64_t n = 0;
  int sh_dim = 0;
  const float* means = nullptr;
  const float* scales = nullptr;
  const float* quats = nullptr;
  const float* opacities = nullptr;
  const float* sh0 = nullptr;
  const float* shN = nullptr;

  GSView() = default;
  // Implicit view over an owning GSData (exact-match overloads still win, so
  // no call ambiguity between the GSData and GSView write overloads).
  GSView(const GSData& d)
      : n(d.n),
        sh_dim(d.sh_dim),
        means(d.means.data()),
        scales(d.scales.data()),
        quats(d.quats.data()),
        opacities(d.opacities.data()),
        sh0(d.sh0.data()),
        shN(d.sh_dim > 0 ? d.shN.data() : nullptr) {}
};

// sh_dim (coeffs/channel) for an SH degree, and the inverse.
inline int sh_dim_for_degree(int degree) {
  switch (degree) {
    case 0: return 0;
    case 1: return 3;
    case 2: return 8;
    case 3: return 15;
    default: return -1;
  }
}

inline int degree_for_sh_dim(int sh_dim) {
  switch (sh_dim) {
    case 0: return 0;
    case 3: return 1;
    case 8: return 2;
    case 15: return 3;
    default: return -1;
  }
}

}  // namespace gsplycpp
