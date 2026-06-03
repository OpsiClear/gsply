#include "gsplycpp.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

void require_exact(const std::vector<float>& a, const std::vector<float>& b,
                   const std::string& name) {
  require(a.size() == b.size(), name + " size mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      throw std::runtime_error(name + " exact mismatch at " + std::to_string(i));
    }
  }
}

void require_close(const std::vector<float>& a, const std::vector<float>& b, float atol,
                   const std::string& name) {
  require(a.size() == b.size(), name + " size mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > atol) {
      throw std::runtime_error(name + " close mismatch at " + std::to_string(i));
    }
  }
}

gsplycpp::GSData sample_data() {
  gsplycpp::GSData d;
  d.n = 64;
  d.sh_dim = 3;
  d.means.resize(static_cast<size_t>(d.n) * 3);
  d.scales.resize(static_cast<size_t>(d.n) * 3);
  d.quats.resize(static_cast<size_t>(d.n) * 4);
  d.opacities.resize(static_cast<size_t>(d.n));
  d.sh0.resize(static_cast<size_t>(d.n) * 3);
  d.shN.resize(static_cast<size_t>(d.n) * d.sh_dim * 3);

  for (int64_t i = 0; i < d.n; ++i) {
    const float t = static_cast<float>(i);
    d.means[i * 3 + 0] = 0.01f * t;
    d.means[i * 3 + 1] = -0.02f * t;
    d.means[i * 3 + 2] = 0.03f * t;
    d.scales[i * 3 + 0] = -6.0f + 0.001f * t;
    d.scales[i * 3 + 1] = -5.0f + 0.002f * t;
    d.scales[i * 3 + 2] = -4.0f + 0.003f * t;
    d.quats[i * 4 + 0] = 1.0f;
    d.quats[i * 4 + 1] = 0.0f;
    d.quats[i * 4 + 2] = 0.0f;
    d.quats[i * 4 + 3] = 0.0f;
    d.opacities[i] = -2.0f + 0.03f * t;
    d.sh0[i * 3 + 0] = -0.2f + 0.001f * t;
    d.sh0[i * 3 + 1] = 0.1f - 0.001f * t;
    d.sh0[i * 3 + 2] = 0.05f + 0.0005f * t;
    for (int k = 0; k < d.sh_dim; ++k) {
      for (int ch = 0; ch < 3; ++ch) {
        d.shN[(i * d.sh_dim + k) * 3 + ch] =
            -0.25f + 0.01f * static_cast<float>((i + k + ch) % 17);
      }
    }
  }
  return d;
}

}  // namespace

int main() {
  const gsplycpp::GSData input = sample_data();
  const std::string ply_path = "gsplycpp_standalone_test.ply";
  const std::string spz3_path = "gsplycpp_standalone_test_v3.spz";
  const std::string spz4_path = "gsplycpp_standalone_test_v4.spz";

  gsplycpp::write_ply(ply_path, input);
  const gsplycpp::GSData ply = gsplycpp::read_ply(ply_path);
  require(ply.n == input.n, "PLY n mismatch");
  require(ply.sh_dim == input.sh_dim, "PLY sh_dim mismatch");
  require_exact(ply.means, input.means, "PLY means");
  require_exact(ply.scales, input.scales, "PLY scales");
  require_exact(ply.quats, input.quats, "PLY quats");
  require_exact(ply.opacities, input.opacities, "PLY opacities");
  require_exact(ply.sh0, input.sh0, "PLY sh0");
  require_exact(ply.shN, input.shN, "PLY shN");

  gsplycpp::write_spz(spz3_path, input, 12, 3);
  gsplycpp::write_spz(spz4_path, input, 12, 4);
  const gsplycpp::GSData spz3 = gsplycpp::read_spz(spz3_path);
  const gsplycpp::GSData spz4 = gsplycpp::read_spz(spz4_path);
  require(spz3.n == input.n && spz4.n == input.n, "SPZ n mismatch");
  require(spz3.sh_dim == input.sh_dim && spz4.sh_dim == input.sh_dim, "SPZ sh_dim mismatch");
  require_close(spz3.means, input.means, 3e-4f, "SPZ v3 means");
  require_close(spz3.scales, input.scales, 0.07f, "SPZ v3 scales");
  require_close(spz3.sh0, input.sh0, 0.027f, "SPZ v3 sh0");
  require_close(spz3.shN, input.shN, 0.016f, "SPZ v3 shN");
  require_close(spz4.means, spz3.means, 1e-7f, "SPZ v4 means");
  require_close(spz4.scales, spz3.scales, 1e-7f, "SPZ v4 scales");
  require_close(spz4.quats, spz3.quats, 1e-7f, "SPZ v4 quats");
  require_close(spz4.opacities, spz3.opacities, 1e-7f, "SPZ v4 opacities");
  require_close(spz4.sh0, spz3.sh0, 1e-7f, "SPZ v4 sh0");
  require_close(spz4.shN, spz3.shN, 1e-7f, "SPZ v4 shN");

  std::remove(ply_path.c_str());
  std::remove(spz3_path.c_str());
  std::remove(spz4_path.c_str());
  return 0;
}
