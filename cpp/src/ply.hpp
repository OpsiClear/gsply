#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gsdata.hpp"

namespace gsplycpp {

// Parsed PLY header: vertex count, ordered "property float" names, and the byte
// offset of the binary payload.
struct PlyHeader {
  int64_t n = 0;
  std::vector<std::string> props;
  size_t data_offset = 0;
};

// Resolved Gaussian-splat layout: column index of each property within a row,
// the per-row float count, and the SH coefficient count (sh_dim).
struct PlyLayout {
  int n_props = 0;
  int sh_dim = 0;
  std::unordered_map<std::string, int> col;  // property name -> column index
};

// Parse a binary_little_endian PLY header from an in-memory buffer (throws on a
// missing end_header marker, non-float property, or non-little-endian format).
PlyHeader parse_ply_header(const char* buf, size_t size, const std::string& path);

// Validate the required GS properties and resolve sh_dim + column indices.
PlyLayout compute_ply_layout(const PlyHeader& h, const std::string& path);

// Decode the interleaved float payload into a GSData (copies / reorders f_rest).
GSData decode_ply_payload(const PlyHeader& h, const PlyLayout& layout, const float* payload,
                          const std::string& path);

// Read an uncompressed binary-little-endian Gaussian-splat PLY, mapping
// properties by name (any group order; extra properties like normals ignored).
// Mirrors gsply.plyread output conventions.
GSData read_ply(const std::string& path);

// Write an uncompressed binary-little-endian PLY in canonical INRIA property
// order (x,y,z, f_dc, f_rest, opacity, scale, rot). Accepts a non-owning GSView
// (a GSData converts implicitly) so callers can write numpy inputs copy-free.
void write_ply(const std::string& path, const GSView& data);

}  // namespace gsplycpp
