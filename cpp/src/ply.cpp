#include "ply.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace gsplycpp {

namespace {

struct Header {
  int64_t n = 0;
  std::vector<std::string> props;  // ordered "property float" names
  size_t data_offset = 0;          // byte offset of binary payload
};

Header parse_header(const std::vector<char>& buf, const std::string& path) {
  static const char marker[] = "end_header";
  const char* found = nullptr;
  if (buf.size() >= sizeof(marker) - 1) {
    for (size_t i = 0; i + (sizeof(marker) - 1) <= buf.size(); ++i) {
      if (std::memcmp(buf.data() + i, marker, sizeof(marker) - 1) == 0) {
        found = buf.data() + i;
        break;
      }
    }
  }
  if (!found) throw std::runtime_error("PLY header end marker not found: " + path);

  size_t end = static_cast<size_t>(found - buf.data()) + (sizeof(marker) - 1);
  // Skip the newline after end_header (\n or \r\n).
  if (end < buf.size() && buf[end] == '\r') ++end;
  if (end < buf.size() && buf[end] == '\n') ++end;

  std::string header_text(buf.data(), found - buf.data());
  std::istringstream ss(header_text);
  std::string line;
  bool binary_le = false;
  Header h;
  while (std::getline(ss, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("format ", 0) == 0) {
      binary_le = line.find("binary_little_endian") != std::string::npos;
    } else if (line.rfind("element vertex ", 0) == 0) {
      h.n = std::stoll(line.substr(std::strlen("element vertex ")));
    } else if (line.rfind("property ", 0) == 0) {
      std::istringstream ps(line);
      std::string kw, type, name;
      ps >> kw >> type >> name;
      if (type != "float" && type != "float32") {
        throw std::runtime_error("Unsupported PLY property type in: " + path);
      }
      h.props.push_back(name);
    }
  }
  if (!binary_le) throw std::runtime_error("Only binary_little_endian PLY supported: " + path);
  h.data_offset = end;
  return h;
}

}  // namespace

GSData read_ply(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open PLY: " + path);
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  Header h = parse_header(buf, path);
  const int64_t n = h.n;
  const int n_props = static_cast<int>(h.props.size());

  std::unordered_map<std::string, int> col;
  int n_rest = 0;
  for (int i = 0; i < n_props; ++i) {
    col[h.props[i]] = i;
    if (h.props[i].rfind("f_rest_", 0) == 0) ++n_rest;
  }
  const int sh_dim = degree_for_sh_dim(n_rest / 3) >= 0 && n_rest % 3 == 0 ? n_rest / 3 : -1;
  if (sh_dim < 0) throw std::runtime_error("Unexpected f_rest count in PLY: " + path);

  static const char* required[] = {"x",       "y",       "z",     "scale_0", "scale_1",
                                    "scale_2", "opacity", "rot_0", "rot_1",   "rot_2",
                                    "rot_3",   "f_dc_0",  "f_dc_1", "f_dc_2"};
  for (const char* r : required) {
    if (col.find(r) == col.end()) {
      throw std::runtime_error(std::string("PLY missing GS property '") + r + "': " + path);
    }
  }

  const size_t need = h.data_offset + static_cast<size_t>(n) * n_props * sizeof(float);
  if (buf.size() < need) throw std::runtime_error("PLY truncated: " + path);
  const float* data = reinterpret_cast<const float*>(buf.data() + h.data_offset);

  GSData d;
  d.n = n;
  d.sh_dim = sh_dim;
  d.means.resize(static_cast<size_t>(n) * 3);
  d.scales.resize(static_cast<size_t>(n) * 3);
  d.quats.resize(static_cast<size_t>(n) * 4);
  d.opacities.resize(static_cast<size_t>(n));
  d.sh0.resize(static_cast<size_t>(n) * 3);
  d.shN.resize(static_cast<size_t>(n) * sh_dim * 3);

  const int ix = col["x"], iy = col["y"], iz = col["z"];
  const int is0 = col["scale_0"], is1 = col["scale_1"], is2 = col["scale_2"];
  const int io = col["opacity"];
  const int ir0 = col["rot_0"], ir1 = col["rot_1"], ir2 = col["rot_2"], ir3 = col["rot_3"];
  const int id0 = col["f_dc_0"], id1 = col["f_dc_1"], id2 = col["f_dc_2"];
  // f_rest is channel-major ([N,3,K]); store shN coeff-major ([N,K,3]).
  std::vector<int> rest_cols(static_cast<size_t>(n_rest));
  for (int j = 0; j < n_rest; ++j) rest_cols[j] = col["f_rest_" + std::to_string(j)];

  for (int64_t i = 0; i < n; ++i) {
    const float* row = data + i * n_props;
    d.means[i * 3 + 0] = row[ix];
    d.means[i * 3 + 1] = row[iy];
    d.means[i * 3 + 2] = row[iz];
    d.scales[i * 3 + 0] = row[is0];
    d.scales[i * 3 + 1] = row[is1];
    d.scales[i * 3 + 2] = row[is2];
    d.quats[i * 4 + 0] = row[ir0];
    d.quats[i * 4 + 1] = row[ir1];
    d.quats[i * 4 + 2] = row[ir2];
    d.quats[i * 4 + 3] = row[ir3];
    d.opacities[i] = row[io];
    d.sh0[i * 3 + 0] = row[id0];
    d.sh0[i * 3 + 1] = row[id1];
    d.sh0[i * 3 + 2] = row[id2];
    for (int ch = 0; ch < 3 && sh_dim > 0; ++ch) {
      for (int k = 0; k < sh_dim; ++k) {
        d.shN[(i * sh_dim + k) * 3 + ch] = row[rest_cols[ch * sh_dim + k]];
      }
    }
  }
  return d;
}

void write_ply(const std::string& path, const GSData& d) {
  const int sh_dim = d.sh_dim;
  if (degree_for_sh_dim(sh_dim) < 0) {
    throw std::runtime_error("write_ply: invalid sh_dim " + std::to_string(sh_dim));
  }
  const int64_t n = d.n;
  const int n_props = 14 + sh_dim * 3;

  std::ostringstream hs;
  hs << "ply\nformat binary_little_endian 1.0\nelement vertex " << n << "\n";
  hs << "property float x\nproperty float y\nproperty float z\n";
  hs << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
  for (int j = 0; j < sh_dim * 3; ++j) hs << "property float f_rest_" << j << "\n";
  hs << "property float opacity\n";
  hs << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n";
  hs << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
  hs << "end_header\n";
  const std::string header = hs.str();

  std::vector<float> rows(static_cast<size_t>(n) * n_props);
  for (int64_t i = 0; i < n; ++i) {
    float* r = rows.data() + i * n_props;
    int o = 0;
    r[o++] = d.means[i * 3 + 0];
    r[o++] = d.means[i * 3 + 1];
    r[o++] = d.means[i * 3 + 2];
    r[o++] = d.sh0[i * 3 + 0];
    r[o++] = d.sh0[i * 3 + 1];
    r[o++] = d.sh0[i * 3 + 2];
    for (int ch = 0; ch < 3; ++ch) {  // f_rest channel-major
      for (int k = 0; k < sh_dim; ++k) {
        r[o++] = d.shN[(i * sh_dim + k) * 3 + ch];
      }
    }
    r[o++] = d.opacities[i];
    r[o++] = d.scales[i * 3 + 0];
    r[o++] = d.scales[i * 3 + 1];
    r[o++] = d.scales[i * 3 + 2];
    r[o++] = d.quats[i * 4 + 0];
    r[o++] = d.quats[i * 4 + 1];
    r[o++] = d.quats[i * 4 + 2];
    r[o++] = d.quats[i * 4 + 3];
  }

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot write PLY: " + path);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char*>(rows.data()),
          static_cast<std::streamsize>(rows.size() * sizeof(float)));
}

}  // namespace gsplycpp
