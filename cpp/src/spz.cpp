#include "spz.hpp"

#include <libdeflate.h>
#include <zlib.h>   // SPZ write: pigz-style parallel deflate (needs Z_SYNC_FLUSH)
#include <zstd.h>   // SPZ v4: per-attribute zstd streams

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(GSPLY_OPENMP)
#include <omp.h>
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
constexpr size_t NGSP_HEADER_SIZE = 32;       // v4 uncompressed header
constexpr uint32_t LATEST_SPZ_VERSION = 4;
constexpr uint32_t MIN_ZSTD_VERSION = 4;  // versions >= this use the NGSP zstd container

template <typename T>
T read_le(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;  // x86/ARM little-endian
}

template <typename T>
void write_le(uint8_t* p, T v) {
  std::memcpy(p, &v, sizeof(T));  // x86/ARM little-endian
}

// zstd whole-buffer compress (used for the NGSP v4 per-attribute streams).
// nbWorkers parallelizes one frame internally (matches Python's threads=-1) so
// the dominant SH stream uses all cores; output is still a single zstd frame.
std::vector<uint8_t> zstd_compress(const uint8_t* src, size_t n, int level, int nbWorkers) {
  std::vector<uint8_t> out(ZSTD_compressBound(n));
  ZSTD_CCtx* c = ZSTD_createCCtx();
  if (!c) throw std::runtime_error("ZSTD_createCCtx failed");
  ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, level);
  if (nbWorkers > 0) ZSTD_CCtx_setParameter(c, ZSTD_c_nbWorkers, nbWorkers);  // no-op if non-MT build
  const size_t r = ZSTD_compress2(c, out.data(), out.size(), src, n);
  ZSTD_freeCCtx(c);
  if (ZSTD_isError(r)) throw std::runtime_error("zstd compress failed");
  out.resize(r);
  return out;
}

void zstd_decompress(uint8_t* dst, size_t dst_size, const uint8_t* src, size_t src_size) {
  const size_t r = ZSTD_decompress(dst, dst_size, src, src_size);
  if (ZSTD_isError(r) || r != dst_size)
    throw std::runtime_error("zstd decompress failed (corrupt NGSP stream?)");
}

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
  if (src.size() < 18) throw std::runtime_error("gzip stream too small");
  // gzip trailer's last 4 bytes are ISIZE (uncompressed size mod 2^32); SPZ files
  // are far below 4 GB so this is exact. Grow-and-retry guards the rare wrap.
  size_t out_size = read_le<uint32_t>(src.data() + src.size() - 4);
  if (out_size == 0) out_size = src.size() * 4 + 64;
  libdeflate_decompressor* d = libdeflate_alloc_decompressor();
  if (!d) throw std::runtime_error("libdeflate alloc failed");
  std::vector<uint8_t> out;
  for (int attempt = 0; attempt < 4; ++attempt) {
    out.resize(out_size);
    size_t actual = 0;
    libdeflate_result r =
        libdeflate_gzip_decompress(d, src.data(), src.size(), out.data(), out.size(), &actual);
    if (r == LIBDEFLATE_SUCCESS) {
      out.resize(actual);
      libdeflate_free_decompressor(d);
      return out;
    }
    if (r != LIBDEFLATE_INSUFFICIENT_SPACE) break;
    out_size *= 2;  // ISIZE wrapped (>4 GB); retry larger
  }
  libdeflate_free_decompressor(d);
  throw std::runtime_error("gzip inflate failed (corrupt or non-gzip)");
}

// Raw-deflate one block. Non-final blocks end byte-aligned via Z_SYNC_FLUSH (an
// empty stored block, BFINAL=0) so independently-compressed blocks concatenate
// into ONE valid deflate stream; the final block uses Z_FINISH (BFINAL=1).
// Blocks compress without a cross-block dictionary, so each block's back-refs
// stay within itself and decode correctly after concatenation (pigz -i style).
std::vector<uint8_t> deflate_block_raw(const uint8_t* data, size_t len, int zlevel, bool last) {
  z_stream s{};
  if (deflateInit2(&s, zlevel, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK)
    throw std::runtime_error("deflateInit2 failed");
  s.next_in = const_cast<Bytef*>(data);
  s.avail_in = static_cast<uInt>(len);
  std::vector<uint8_t> out(deflateBound(&s, len) + 16);
  s.next_out = out.data();
  s.avail_out = static_cast<uInt>(out.size());
  const int rc = deflate(&s, last ? Z_FINISH : Z_SYNC_FLUSH);
  const size_t produced = out.size() - s.avail_out;
  const bool ok = last ? (rc == Z_STREAM_END) : (rc == Z_OK && s.avail_in == 0);
  deflateEnd(&s);
  if (!ok) throw std::runtime_error("deflate block failed");
  out.resize(produced);
  return out;
}

// Parallel single-member gzip: split the payload into ~1 MB blocks, deflate them
// concurrently, and stitch into one standard gzip stream (10-byte header + body
// + CRC32 + ISIZE). Single-member, so any single-shot gzip reader (incl. the
// Niantic SPZ loader) decodes it fully. `level` is libdeflate-scale; mapped to
// zlib 1..9.
std::vector<uint8_t> gzip_compress(const std::vector<uint8_t>& src, int level) {
  const size_t n = src.size();
  const int zlevel = std::min(std::max(level, 1), 9);

  const size_t TARGET_BLOCK = size_t(1) << 20;  // 1 MB
  size_t nblocks = n ? (n + TARGET_BLOCK - 1) / TARGET_BLOCK : 1;
#if defined(GSPLY_OPENMP)
  const size_t threads = static_cast<size_t>(std::max(1, omp_get_max_threads()));
  if (nblocks > threads) nblocks = threads;  // ~one block per core; bigger blocks compress better
#else
  nblocks = 1;
#endif
  if (nblocks < 1) nblocks = 1;

  std::vector<std::pair<size_t, size_t>> ranges(nblocks);  // (offset, len)
  const size_t base = n / nblocks, rem = n % nblocks;
  for (size_t b = 0, off = 0; b < nblocks; ++b) {
    const size_t len = base + (b < rem ? 1 : 0);
    ranges[b] = {off, len};
    off += len;
  }

  std::vector<std::vector<uint8_t>> parts(nblocks);
  GSPLY_PARALLEL_FOR
  for (int64_t b = 0; b < static_cast<int64_t>(nblocks); ++b) {
    parts[b] = deflate_block_raw(src.data() + ranges[b].first, ranges[b].second, zlevel,
                                 b == static_cast<int64_t>(nblocks) - 1);
  }

  size_t body = 0;
  for (const auto& p : parts) body += p.size();
  std::vector<uint8_t> out;
  out.reserve(10 + body + 8);
  const uint8_t header[10] = {0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff};
  out.insert(out.end(), header, header + 10);
  for (const auto& p : parts) out.insert(out.end(), p.begin(), p.end());

  const uint32_t crc = libdeflate_crc32(0, src.data(), n);
  const uint32_t isize = static_cast<uint32_t>(n & 0xffffffffu);
  for (const uint32_t v : {crc, isize})
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xffu));
  return out;
}

// Parse an NGSP v4 container and zstd-decompress its streams into one contiguous
// packed-sections buffer (canonical order positions|alphas|colors|scales|
// rotations|sh, base 0), filling `info` with offsets relative to that buffer.
std::vector<uint8_t> decompress_ngsp(const std::vector<uint8_t>& file, const std::string& path,
                                     SpzInfo& info) {
  if (file.size() < NGSP_HEADER_SIZE) throw std::runtime_error("NGSP file too small: " + path);
  const uint32_t magic = read_le<uint32_t>(file.data());
  const uint32_t version = read_le<uint32_t>(file.data() + 4);
  const uint32_t num_points = read_le<uint32_t>(file.data() + 8);
  const uint8_t sh_degree = file[12];
  const uint8_t frac_bits = file[13];
  const uint8_t num_streams = file[15];
  const uint32_t toc_off = read_le<uint32_t>(file.data() + 16);
  if (magic != NGSP_MAGIC) throw std::runtime_error("Not an SPZ file: " + path);
  if (version < MIN_ZSTD_VERSION || version > LATEST_SPZ_VERSION)
    throw std::runtime_error("Unsupported NGSP version: " + path);
  if (sh_degree > 3) throw std::runtime_error("Unsupported SH degree: " + path);
  if (frac_bits < 1 || frac_bits > MAX_FRACTIONAL_BITS)
    throw std::runtime_error("Invalid SPZ fractional_bits: " + path);

  const int64_t n = num_points;
  info.n = n;
  info.sh_dim = sh_dim_for_degree(sh_degree);
  info.uses_st = true;
  info.rot_stride = 4;
  info.alpha_ofs = static_cast<size_t>(9) * n;
  info.color_ofs = static_cast<size_t>(10) * n;
  info.scale_ofs = static_cast<size_t>(13) * n;
  info.rot_ofs = static_cast<size_t>(16) * n;
  info.sh_ofs = static_cast<size_t>(20) * n;  // rot_stride == 4
  info.inv_frac = 1.0f / static_cast<float>(1 << frac_bits);

  // Section sizes + their offsets within the assembled buffer, canonical order.
  const size_t sec_sizes[6] = {
      static_cast<size_t>(9) * n, static_cast<size_t>(1) * n,
      static_cast<size_t>(3) * n, static_cast<size_t>(3) * n,
      static_cast<size_t>(4) * n, static_cast<size_t>(info.sh_dim) * 3 * n};
  const size_t sec_offsets[6] = {0, info.alpha_ofs, info.color_ofs,
                                 info.scale_ofs, info.rot_ofs, info.sh_ofs};
  int expected_streams = 0;
  size_t total = 0;
  for (int k = 0; k < 6; ++k) {
    total += sec_sizes[k];
    if (sec_sizes[k]) ++expected_streams;
  }
  if (static_cast<int>(num_streams) != expected_streams)
    throw std::runtime_error("NGSP stream count mismatch: " + path);

  const size_t toc_end = static_cast<size_t>(toc_off) + static_cast<size_t>(num_streams) * 16;
  if (toc_off < NGSP_HEADER_SIZE || toc_end > file.size())
    throw std::runtime_error("NGSP TOC out of bounds: " + path);

  std::vector<uint8_t> sections(total);
  // Resolve each stream's (compressed src, dst) first — compressed offsets are
  // cumulative, so this prefix pass is serial — then decompress them in parallel.
  struct Job {
    const uint8_t* src;
    size_t csize;
    uint8_t* dst;
    size_t dsize;
  };
  std::vector<Job> jobs;
  size_t comp_off = toc_end;
  int si = 0;
  for (int k = 0; k < 6; ++k) {
    if (sec_sizes[k] == 0) continue;
    const size_t e = static_cast<size_t>(toc_off) + static_cast<size_t>(si) * 16;
    const uint64_t csize = read_le<uint64_t>(file.data() + e);
    const uint64_t usize = read_le<uint64_t>(file.data() + e + 8);
    if (usize != sec_sizes[k]) throw std::runtime_error("NGSP stream size mismatch: " + path);
    if (comp_off + csize > file.size())
      throw std::runtime_error("NGSP stream overruns file: " + path);
    jobs.push_back({file.data() + comp_off, static_cast<size_t>(csize),
                    sections.data() + sec_offsets[k], sec_sizes[k]});
    comp_off += csize;
    ++si;
  }
  std::vector<char> ok(jobs.size(), 1);  // exceptions can't cross an OpenMP region
  GSPLY_PARALLEL_FOR
  for (int64_t j = 0; j < static_cast<int64_t>(jobs.size()); ++j) {
    const size_t r = ZSTD_decompress(jobs[j].dst, jobs[j].dsize, jobs[j].src, jobs[j].csize);
    if (ZSTD_isError(r) || r != jobs[j].dsize) ok[j] = 0;
  }
  for (char c : ok)
    if (!c) throw std::runtime_error("zstd decompress failed (corrupt NGSP stream?): " + path);
  return sections;
}

}  // namespace

std::vector<uint8_t> read_spz_payload(const std::string& path) {
  try {
    return gunzip(read_file(path));
  } catch (const std::runtime_error&) {
    throw std::runtime_error("Could not gunzip SPZ (corrupt, or unsupported v4 container?): " +
                             path);
  }
}

SpzInfo parse_spz(const std::vector<uint8_t>& raw, const std::string& path) {
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

  SpzInfo info;
  info.n = num_points;
  info.sh_dim = sh_dim_for_degree(sh_degree);
  info.uses_st = version >= 3;
  info.rot_stride = info.uses_st ? 4 : 3;

  const size_t expected =
      static_cast<size_t>(9 + 1 + 3 + 3 + info.rot_stride + info.sh_dim * 3) * info.n;
  if (raw.size() - 16 < expected) throw std::runtime_error("SPZ payload too small: " + path);

  info.alpha_ofs = static_cast<size_t>(9) * info.n;
  info.color_ofs = static_cast<size_t>(10) * info.n;
  info.scale_ofs = static_cast<size_t>(13) * info.n;
  info.rot_ofs = static_cast<size_t>(16) * info.n;
  info.sh_ofs = info.rot_ofs + static_cast<size_t>(info.rot_stride) * info.n;
  info.inv_frac = 1.0f / static_cast<float>(1 << frac_bits);
  return info;
}

SpzLoaded load_spz(const std::string& path) {
  std::vector<uint8_t> file = read_file(path);
  SpzLoaded out;
  // Legacy SPZ is a gzip stream (1f 8b); NGSP v4 starts with "NGSP" in the clear.
  if (file.size() >= 2 && file[0] == 0x1f && file[1] == 0x8b) {
    try {
      out.buffer = gunzip(file);
    } catch (const std::runtime_error&) {
      throw std::runtime_error("Could not gunzip SPZ (corrupt container?): " + path);
    }
    out.info = parse_spz(out.buffer, path);
    out.base = 16;
  } else if (file.size() >= 4 && read_le<uint32_t>(file.data()) == NGSP_MAGIC) {
    out.buffer = decompress_ngsp(file, path, out.info);
    out.base = 0;
  } else {
    throw std::runtime_error("Not an SPZ file (unrecognized container): " + path);
  }
  return out;
}

void decode_spz_body(const uint8_t* buf, const SpzInfo& info, float* means, float* scales,
                     float* quats, float* opacities, float* sh0, float* shN) {
  const int64_t n = info.n;
  const int sh_dim = info.sh_dim;
  const bool uses_st = info.uses_st;
  const float inv_frac = info.inv_frac;

  GSPLY_PARALLEL_FOR
  for (int64_t i = 0; i < n; ++i) {
    // positions: 24-bit signed fixed point
    const size_t p = static_cast<size_t>(i) * 9;
    for (int j = 0; j < 3; ++j) {
      int32_t v = static_cast<int32_t>(buf[p + j * 3]) |
                  (static_cast<int32_t>(buf[p + j * 3 + 1]) << 8) |
                  (static_cast<int32_t>(buf[p + j * 3 + 2]) << 16);
      if (v >= 0x800000) v -= 0x1000000;
      means[i * 3 + j] = static_cast<float>(v) * inv_frac;
    }
    // scales
    const size_t s = info.scale_ofs + static_cast<size_t>(i) * 3;
    for (int j = 0; j < 3; ++j) scales[i * 3 + j] = static_cast<float>(buf[s + j]) / 16.0f - 10.0f;

    // rotation -> wxyz unit quaternion
    float qx = 0, qy = 0, qz = 0, qw = 0;
    if (uses_st) {
      const size_t r = info.rot_ofs + static_cast<size_t>(i) * 4;
      uint32_t packed = static_cast<uint32_t>(buf[r]) | (static_cast<uint32_t>(buf[r + 1]) << 8) |
                        (static_cast<uint32_t>(buf[r + 2]) << 16) |
                        (static_cast<uint32_t>(buf[r + 3]) << 24);
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
      const size_t r = info.rot_ofs + static_cast<size_t>(i) * 3;
      qx = static_cast<float>(buf[r]) / 127.5f - 1.0f;
      qy = static_cast<float>(buf[r + 1]) / 127.5f - 1.0f;
      qz = static_cast<float>(buf[r + 2]) / 127.5f - 1.0f;
      qw = std::sqrt(std::max(0.0f, 1.0f - qx * qx - qy * qy - qz * qz));
    }
    quats[i * 4 + 0] = qw;
    quats[i * 4 + 1] = qx;
    quats[i * 4 + 2] = qy;
    quats[i * 4 + 3] = qz;

    // alpha -> logit (edge-clamped inverse sigmoid)
    float a = static_cast<float>(buf[info.alpha_ofs + i]) / 255.0f;
    a = std::min(std::max(a, 1e-6f), 1.0f - 1e-6f);
    opacities[i] = std::log(a / (1.0f - a));

    // color -> sh0 (wide RGB)
    const size_t c = info.color_ofs + static_cast<size_t>(i) * 3;
    for (int j = 0; j < 3; ++j)
      sh0[i * 3 + j] = (static_cast<float>(buf[c + j]) / 255.0f - 0.5f) / COLOR_SCALE;

    // higher-order SH
    if (sh_dim > 0) {
      const size_t sb = info.sh_ofs + static_cast<size_t>(i) * sh_dim * 3;
      for (int k = 0; k < sh_dim; ++k)
        for (int ch = 0; ch < 3; ++ch)
          shN[(i * sh_dim + k) * 3 + ch] =
              (static_cast<float>(buf[sb + k * 3 + ch]) - 128.0f) / 128.0f;
    }
  }
}

GSData read_spz(const std::string& path) {
  const SpzLoaded loaded = load_spz(path);  // gzip v1-3 or NGSP v4
  const SpzInfo& info = loaded.info;
  GSData d;
  d.n = info.n;
  d.sh_dim = info.sh_dim;
  d.means.resize(static_cast<size_t>(info.n) * 3);
  d.scales.resize(static_cast<size_t>(info.n) * 3);
  d.quats.resize(static_cast<size_t>(info.n) * 4);
  d.opacities.resize(static_cast<size_t>(info.n));
  d.sh0.resize(static_cast<size_t>(info.n) * 3);
  d.shN.resize(static_cast<size_t>(info.n) * info.sh_dim * 3);
  decode_spz_body(loaded.buffer.data() + loaded.base, info, d.means.data(), d.scales.data(),
                  d.quats.data(), d.opacities.data(), d.sh0.data(), d.shN.data());
  return d;
}

void write_spz(const std::string& path, const GSView& d, int fractional_bits, int version,
               int level) {
  if (fractional_bits < 1 || fractional_bits > MAX_FRACTIONAL_BITS)
    throw std::runtime_error("write_spz: fractional_bits out of range");
  if (version != 3 && version != 4)
    throw std::runtime_error("write_spz: version must be 3 (gzip) or 4 (zstd)");
  const int sh_dim = d.sh_dim;
  const int sh_degree = degree_for_sh_dim(sh_dim);
  if (sh_degree < 0) throw std::runtime_error("write_spz: invalid sh_dim");
  const int64_t n = d.n;
  const int rot_stride = 4;

  std::vector<uint8_t> payload(16 + static_cast<size_t>(9 + 1 + 3 + 3 + rot_stride + sh_dim * 3) * n);
  // Legacy 16B header (used by the v3 gzip path; ignored by v4 which builds its own).
  uint32_t magic = NGSP_MAGIC, ver3 = 3, np = static_cast<uint32_t>(n);
  std::memcpy(payload.data() + 0, &magic, 4);
  std::memcpy(payload.data() + 4, &ver3, 4);
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

  auto write_file = [&path](const uint8_t* data, size_t size) {
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Cannot write SPZ: " + path);
    const bool ok = std::fwrite(data, 1, size, fp) == size;
    std::fclose(fp);
    if (!ok) throw std::runtime_error("Short write on SPZ: " + path);
  };

  if (version == 3) {
    // Legacy gzip container: 16B header + concatenated sections (already in payload).
    const std::vector<uint8_t> gz = gzip_compress(payload, level < 0 ? 6 : level);
    write_file(gz.data(), gz.size());
    return;
  }

  // version == 4: NGSP container — each non-empty section as its own zstd stream.
  const int zlevel = level < 0 ? 12 : level;
  const int workers = std::max(1u, std::thread::hardware_concurrency());
  // Non-empty sections within `buf`, canonical order (offset, size). sh is empty at deg 0.
  struct Sec { size_t off, size; };
  const Sec all[6] = {{0, static_cast<size_t>(9) * n},
                      {alpha_ofs, static_cast<size_t>(1) * n},
                      {color_ofs, static_cast<size_t>(3) * n},
                      {scale_ofs, static_cast<size_t>(3) * n},
                      {rot_ofs, static_cast<size_t>(4) * n},
                      {sh_ofs, static_cast<size_t>(sh_dim) * 3 * n}};
  std::vector<Sec> nz;
  for (const Sec& s : all) {
    if (s.size) nz.push_back(s);
  }
  // Hybrid parallelism: compress streams concurrently, and give the largest (SH)
  // intra-frame zstd workers so it isn't the lone long pole on one core. The
  // small streams run single-threaded alongside it. ~1.4x over per-stream MT.
  size_t big = 0;
  for (size_t i = 1; i < nz.size(); ++i)
    if (nz[i].size > nz[big].size) big = i;
  std::vector<std::vector<uint8_t>> chunks(nz.size());
  std::vector<char> ok(nz.size(), 1);  // exceptions can't cross an OpenMP region
  GSPLY_PARALLEL_FOR
  for (int64_t i = 0; i < static_cast<int64_t>(nz.size()); ++i) {
    const int nw = (static_cast<size_t>(i) == big) ? workers : 0;
    try {
      chunks[i] = zstd_compress(buf + nz[i].off, nz[i].size, zlevel, nw);
    } catch (...) {
      ok[i] = 0;
    }
  }
  for (char c : ok)
    if (!c) throw std::runtime_error("zstd compress failed");

  const uint8_t num_streams = static_cast<uint8_t>(nz.size());
  const uint32_t toc_off = static_cast<uint32_t>(NGSP_HEADER_SIZE);
  std::vector<uint8_t> out(NGSP_HEADER_SIZE + static_cast<size_t>(num_streams) * 16);
  uint32_t magic4 = NGSP_MAGIC, version4 = 4, np4 = static_cast<uint32_t>(n);
  write_le<uint32_t>(out.data() + 0, magic4);
  write_le<uint32_t>(out.data() + 4, version4);
  write_le<uint32_t>(out.data() + 8, np4);
  out[12] = static_cast<uint8_t>(sh_degree);
  out[13] = static_cast<uint8_t>(fractional_bits);
  out[14] = 0;  // flags (no antialiased / extensions)
  out[15] = num_streams;
  write_le<uint32_t>(out.data() + 16, toc_off);
  // out[20..32) reserved, already zero.
  for (size_t i = 0; i < nz.size(); ++i) {
    write_le<uint64_t>(out.data() + toc_off + i * 16, static_cast<uint64_t>(chunks[i].size()));
    write_le<uint64_t>(out.data() + toc_off + i * 16 + 8, static_cast<uint64_t>(nz[i].size));
  }
  for (const auto& chunk : chunks) out.insert(out.end(), chunk.begin(), chunk.end());
  write_file(out.data(), out.size());
}

}  // namespace gsplycpp
