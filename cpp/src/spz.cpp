#include "spz.hpp"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(GSPLY_OPENMP)
#define GSPLY_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define GSPLY_PARALLEL_FOR
#endif

namespace gsplycpp {

namespace {

constexpr uint32_t NGSP_MAGIC = 0x5053474Eu;  // "NGSP"
constexpr float COLOR_SCALE = 0.15f;
constexpr int MAX_FRACTIONAL_BITS = 24;
constexpr float INV_SQRT2 = 0.70710678118654752440f;
constexpr uint32_t C_MASK = (1u << 9) - 1u;  // 9-bit magnitude

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open SPZ: " + path);
  f.seekg(0, std::ios::end);
  const std::streamsize fsize = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(fsize));
  f.read(reinterpret_cast<char*>(buf.data()), fsize);  // bulk read, not byte-by-byte
  return buf;
}

std::vector<uint8_t> gunzip(const std::vector<uint8_t>& src) {
  z_stream zs{};
  if (inflateInit2(&zs, 15 + 16) != Z_OK) throw std::runtime_error("inflateInit2 failed");
  zs.next_in = const_cast<Bytef*>(src.data());
  zs.avail_in = static_cast<uInt>(src.size());
  std::vector<uint8_t> out;
  out.reserve(src.size() * 3 + 64);
  std::vector<uint8_t> chunk(1 << 16);
  int ret;
  do {
    zs.next_out = chunk.data();
    zs.avail_out = static_cast<uInt>(chunk.size());
    ret = inflate(&zs, Z_NO_FLUSH);
    if (ret != Z_OK && ret != Z_STREAM_END && ret != Z_BUF_ERROR) {
      inflateEnd(&zs);
      throw std::runtime_error("gzip inflate failed (corrupt or non-gzip)");
    }
    out.insert(out.end(), chunk.data(), chunk.data() + (chunk.size() - zs.avail_out));
    if (ret == Z_BUF_ERROR && zs.avail_in == 0) break;
  } while (ret != Z_STREAM_END);
  inflateEnd(&zs);
  return out;
}

std::vector<uint8_t> gzip_compress(const std::vector<uint8_t>& src) {
  z_stream zs{};
  if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
    throw std::runtime_error("deflateInit2 failed");
  zs.next_in = const_cast<Bytef*>(src.data());
  zs.avail_in = static_cast<uInt>(src.size());
  std::vector<uint8_t> out;
  out.reserve(src.size() / 2 + 64);
  std::vector<uint8_t> chunk(1 << 16);
  int ret;
  do {
    zs.next_out = chunk.data();
    zs.avail_out = static_cast<uInt>(chunk.size());
    ret = deflate(&zs, Z_FINISH);
    out.insert(out.end(), chunk.data(), chunk.data() + (chunk.size() - zs.avail_out));
  } while (ret != Z_STREAM_END);
  deflateEnd(&zs);
  return out;
}

template <typename T>
T read_le(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;  // x86/ARM little-endian
}

}  // namespace

GSData read_spz(const std::string& path) {
  std::vector<uint8_t> raw;
  try {
    raw = gunzip(read_file(path));
  } catch (const std::runtime_error&) {
    throw std::runtime_error("Could not gunzip SPZ (corrupt, or unsupported v4 container?): " + path);
  }
  if (raw.size() < 16) throw std::runtime_error("SPZ file too small: " + path);

  const uint32_t magic = read_le<uint32_t>(raw.data());
  const uint32_t version = read_le<uint32_t>(raw.data() + 4);
  const uint32_t num_points = read_le<uint32_t>(raw.data() + 8);
  const uint8_t sh_degree = raw[12];
  const uint8_t frac_bits = raw[13];
  if (magic != NGSP_MAGIC) throw std::runtime_error("Not an SPZ file: " + path);
  if (version < 1 || version > 3) throw std::runtime_error("Unsupported SPZ version: " + path);
  if (sh_degree > 3) throw std::runtime_error("Unsupported SH degree: " + path);
  if (frac_bits < 1 || frac_bits > MAX_FRACTIONAL_BITS)
    throw std::runtime_error("Invalid SPZ fractional_bits: " + path);

  const int64_t n = num_points;
  const int sh_dim = sh_dim_for_degree(sh_degree);
  const bool uses_st = version >= 3;
  const int rot_stride = uses_st ? 4 : 3;

  const size_t expected = static_cast<size_t>(9 + 1 + 3 + 3 + rot_stride + sh_dim * 3) * n;
  if (raw.size() - 16 < expected) throw std::runtime_error("SPZ payload too small: " + path);
  const uint8_t* buf = raw.data() + 16;  // tolerate trailing extension bytes

  const size_t alpha_ofs = static_cast<size_t>(9) * n;
  const size_t color_ofs = static_cast<size_t>(10) * n;
  const size_t scale_ofs = static_cast<size_t>(13) * n;
  const size_t rot_ofs = static_cast<size_t>(16) * n;
  const size_t sh_ofs = rot_ofs + static_cast<size_t>(rot_stride) * n;
  const float inv_frac = 1.0f / static_cast<float>(1 << frac_bits);

  GSData d;
  d.n = n;
  d.sh_dim = sh_dim;
  d.means.resize(static_cast<size_t>(n) * 3);
  d.scales.resize(static_cast<size_t>(n) * 3);
  d.quats.resize(static_cast<size_t>(n) * 4);
  d.opacities.resize(static_cast<size_t>(n));
  d.sh0.resize(static_cast<size_t>(n) * 3);
  d.shN.resize(static_cast<size_t>(n) * sh_dim * 3);

  GSPLY_PARALLEL_FOR
  for (int64_t i = 0; i < n; ++i) {
    // positions: 24-bit signed fixed point
    const size_t p = static_cast<size_t>(i) * 9;
    for (int j = 0; j < 3; ++j) {
      int32_t v = static_cast<int32_t>(buf[p + j * 3]) | (static_cast<int32_t>(buf[p + j * 3 + 1]) << 8) |
                  (static_cast<int32_t>(buf[p + j * 3 + 2]) << 16);
      if (v >= 0x800000) v -= 0x1000000;
      d.means[i * 3 + j] = static_cast<float>(v) * inv_frac;
    }
    // scales
    const size_t s = scale_ofs + static_cast<size_t>(i) * 3;
    for (int j = 0; j < 3; ++j) d.scales[i * 3 + j] = static_cast<float>(buf[s + j]) / 16.0f - 10.0f;

    // rotation -> wxyz unit quaternion
    float qx = 0, qy = 0, qz = 0, qw = 0;
    if (uses_st) {
      const size_t r = rot_ofs + static_cast<size_t>(i) * 4;
      uint32_t packed = static_cast<uint32_t>(buf[r]) | (static_cast<uint32_t>(buf[r + 1]) << 8) |
                        (static_cast<uint32_t>(buf[r + 2]) << 16) | (static_cast<uint32_t>(buf[r + 3]) << 24);
      const uint32_t i_largest = (packed >> 30) & 3u;
      uint32_t work = packed;
      float ss = 0.0f;
      for (int axis = 3; axis >= 0; --axis) {
        if (static_cast<uint32_t>(axis) != i_largest) {
          const uint32_t mag = work & C_MASK;
          const uint32_t negbit = (work >> 9) & 1u;
          work >>= 10;
          float val = INV_SQRT2 * (static_cast<float>(mag) / static_cast<float>(C_MASK));
          if (negbit) val = -val;
          if (axis == 0) qx = val;
          else if (axis == 1) qy = val;
          else if (axis == 2) qz = val;
          else qw = val;
          ss += val * val;
        }
      }
      const float large = std::sqrt(std::max(0.0f, 1.0f - ss));
      if (i_largest == 0) qx = large;
      else if (i_largest == 1) qy = large;
      else if (i_largest == 2) qz = large;
      else qw = large;
    } else {
      const size_t r = rot_ofs + static_cast<size_t>(i) * 3;
      qx = static_cast<float>(buf[r]) / 127.5f - 1.0f;
      qy = static_cast<float>(buf[r + 1]) / 127.5f - 1.0f;
      qz = static_cast<float>(buf[r + 2]) / 127.5f - 1.0f;
      qw = std::sqrt(std::max(0.0f, 1.0f - qx * qx - qy * qy - qz * qz));
    }
    d.quats[i * 4 + 0] = qw;
    d.quats[i * 4 + 1] = qx;
    d.quats[i * 4 + 2] = qy;
    d.quats[i * 4 + 3] = qz;

    // alpha -> logit (edge-clamped inverse sigmoid)
    float a = static_cast<float>(buf[alpha_ofs + i]) / 255.0f;
    a = std::min(std::max(a, 1e-6f), 1.0f - 1e-6f);
    d.opacities[i] = std::log(a / (1.0f - a));

    // color -> sh0 (wide RGB)
    const size_t c = color_ofs + static_cast<size_t>(i) * 3;
    for (int j = 0; j < 3; ++j)
      d.sh0[i * 3 + j] = (static_cast<float>(buf[c + j]) / 255.0f - 0.5f) / COLOR_SCALE;

    // higher-order SH
    if (sh_dim > 0) {
      const size_t sb = sh_ofs + static_cast<size_t>(i) * sh_dim * 3;
      for (int k = 0; k < sh_dim; ++k)
        for (int ch = 0; ch < 3; ++ch)
          d.shN[(i * sh_dim + k) * 3 + ch] =
              (static_cast<float>(buf[sb + k * 3 + ch]) - 128.0f) / 128.0f;
    }
  }
  return d;
}

void write_spz(const std::string& path, const GSData& d, int fractional_bits) {
  if (fractional_bits < 1 || fractional_bits > MAX_FRACTIONAL_BITS)
    throw std::runtime_error("write_spz: fractional_bits out of range");
  const int sh_dim = d.sh_dim;
  const int sh_degree = degree_for_sh_dim(sh_dim);
  if (sh_degree < 0) throw std::runtime_error("write_spz: invalid sh_dim");
  const int64_t n = d.n;
  const int rot_stride = 4;

  std::vector<uint8_t> payload(16 + static_cast<size_t>(9 + 1 + 3 + 3 + rot_stride + sh_dim * 3) * n);
  // header
  uint32_t magic = NGSP_MAGIC, version = 3, np = static_cast<uint32_t>(n);
  std::memcpy(payload.data() + 0, &magic, 4);
  std::memcpy(payload.data() + 4, &version, 4);
  std::memcpy(payload.data() + 8, &np, 4);
  payload[12] = static_cast<uint8_t>(sh_degree);
  payload[13] = static_cast<uint8_t>(fractional_bits);
  payload[14] = 0;
  payload[15] = 0;

  uint8_t* buf = payload.data() + 16;
  const size_t alpha_ofs = static_cast<size_t>(9) * n;
  const size_t color_ofs = static_cast<size_t>(10) * n;
  const size_t scale_ofs = static_cast<size_t>(13) * n;
  const size_t rot_ofs = static_cast<size_t>(16) * n;
  const size_t sh_ofs = rot_ofs + static_cast<size_t>(rot_stride) * n;
  const float scale = static_cast<float>(1 << fractional_bits);

  auto clamp_u8 = [](float x) -> uint8_t {
    long v = std::lround(x);
    return static_cast<uint8_t>(std::min<long>(255, std::max<long>(0, v)));
  };

  GSPLY_PARALLEL_FOR
  for (int64_t i = 0; i < n; ++i) {
    // positions
    const size_t p = static_cast<size_t>(i) * 9;
    for (int j = 0; j < 3; ++j) {
      long fx = std::lround(d.means[i * 3 + j] * scale);
      fx = std::min<long>((1 << 23) - 1, std::max<long>(-(1 << 23), fx));
      uint32_t u = static_cast<uint32_t>(fx) & 0xFFFFFFu;
      buf[p + j * 3 + 0] = u & 0xFF;
      buf[p + j * 3 + 1] = (u >> 8) & 0xFF;
      buf[p + j * 3 + 2] = (u >> 16) & 0xFF;
    }
    // scales
    const size_t s = scale_ofs + static_cast<size_t>(i) * 3;
    for (int j = 0; j < 3; ++j) buf[s + j] = clamp_u8((d.scales[i * 3 + j] + 10.0f) * 16.0f);
    // alpha
    const float sig = 1.0f / (1.0f + std::exp(-d.opacities[i]));
    buf[alpha_ofs + i] = clamp_u8(sig * 255.0f);
    // color
    const size_t c = color_ofs + static_cast<size_t>(i) * 3;
    for (int j = 0; j < 3; ++j)
      buf[c + j] = clamp_u8(d.sh0[i * 3 + j] * (COLOR_SCALE * 255.0f) + 127.5f);

    // rotation: wxyz -> xyzw -> smallest-three pack
    float q[4] = {d.quats[i * 4 + 1], d.quats[i * 4 + 2], d.quats[i * 4 + 3], d.quats[i * 4 + 0]};
    float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norm > 0) for (float& v : q) v /= norm;
    int i_largest = 0;
    for (int a = 1; a < 4; ++a)
      if (std::fabs(q[a]) > std::fabs(q[i_largest])) i_largest = a;
    const float sgn = q[i_largest] < 0 ? -1.0f : 1.0f;
    for (float& v : q) v *= sgn;
    uint32_t packed = 0;
    const float sqrt1_2 = 1.0f / std::sqrt(2.0f);
    for (int a = 0; a < 4; ++a) {
      if (a != i_largest) {
        long mag = std::lround(static_cast<float>(C_MASK) * std::fabs(q[a]) / sqrt1_2);
        mag = std::min<long>(C_MASK, std::max<long>(0, mag));
        const uint32_t negbit = q[a] < 0 ? 1u : 0u;
        packed = (packed << 10) | (negbit << 9) | static_cast<uint32_t>(mag);
      }
    }
    packed |= static_cast<uint32_t>(i_largest) << 30;
    const size_t r = rot_ofs + static_cast<size_t>(i) * 4;
    buf[r + 0] = packed & 0xFF;
    buf[r + 1] = (packed >> 8) & 0xFF;
    buf[r + 2] = (packed >> 16) & 0xFF;
    buf[r + 3] = (packed >> 24) & 0xFF;

    // higher-order SH
    if (sh_dim > 0) {
      const size_t sb = sh_ofs + static_cast<size_t>(i) * sh_dim * 3;
      for (int k = 0; k < sh_dim; ++k)
        for (int ch = 0; ch < 3; ++ch)
          buf[sb + k * 3 + ch] = clamp_u8(d.shN[(i * sh_dim + k) * 3 + ch] * 128.0f + 128.0f);
    }
  }

  std::vector<uint8_t> gz = gzip_compress(payload);
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot write SPZ: " + path);
  f.write(reinterpret_cast<const char*>(gz.data()), static_cast<std::streamsize>(gz.size()));
}

}  // namespace gsplycpp
