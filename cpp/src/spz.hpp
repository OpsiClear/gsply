#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gsdata.hpp"

namespace gsplycpp {

// Parsed SPZ header + section byte offsets (relative to payload base = data+16).
struct SpzInfo {
  int64_t n = 0;
  int sh_dim = 0;
  bool uses_st = false;  // version >= 3 packs rotations smallest-three (4 bytes)
  int rot_stride = 3;
  size_t alpha_ofs = 0, color_ofs = 0, scale_ofs = 0, rot_ofs = 0, sh_ofs = 0;
  float inv_frac = 0.0f;
};

// Read + gunzip an SPZ file into its decompressed payload (section base = data+16).
std::vector<uint8_t> read_spz_payload(const std::string& path);

// Validate the SPZ header and resolve section offsets (throws on bad magic /
// version / SH degree / fractional_bits / truncated payload).
SpzInfo parse_spz(const std::vector<uint8_t>& payload, const std::string& path);

// A loaded SPZ container, normalized so the packed sections start at
// (buffer.data() + base) and `info`'s offsets are relative to that base.
// Covers both the legacy gzip (v1/v2/v3) and NGSP v4 (zstd) containers.
struct SpzLoaded {
  std::vector<uint8_t> buffer;  // owns the decoded section bytes
  size_t base = 0;              // byte offset of the positions section in buffer
  SpzInfo info;
};

// Read any SPZ container (gzip v1-3 or NGSP v4) to a flat packed-sections buffer.
SpzLoaded load_spz(const std::string& path);

// Decode the SPZ body into caller-owned buffers (sizes: means/scales 3n, quats
// 4n, opacities n, sh0 3n, shN sh_dim*3n; shN may be null when sh_dim == 0).
// `buf` is the section base (payload.data() + 16).
void decode_spz_body(const uint8_t* buf, const SpzInfo& info, float* means, float* scales,
                     float* quats, float* opacities, float* sh0, float* shN);

// Read a Niantic SPZ file (legacy gzip v1/v2/v3) into PLY-format GSData.
// Parity with gsply.read_spz: raw frame (no coordinate conversion).
GSData read_spz(const std::string& path);

// Write to SPZ. `version` 3 => legacy gzip container (smallest-three quats);
// `version` 4 => NGSP container with per-attribute zstd streams. Accepts a
// non-owning GSView (a GSData converts implicitly) so callers can write numpy
// inputs copy-free. `level` is the codec level (gzip 1..12 / zstd 1..22);
// `level < 0` picks the per-codec default (gzip 6, zstd 12). Parity with
// gsply.write_spz.
void write_spz(const std::string& path, const GSView& data, int fractional_bits = 12,
               int version = 3, int level = -1);

}  // namespace gsplycpp
