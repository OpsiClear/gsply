#pragma once

#include <string>

#include "gsdata.hpp"

namespace gsplycpp {

// Read an uncompressed binary-little-endian Gaussian-splat PLY, mapping
// properties by name (any group order; extra properties like normals ignored).
// Mirrors gsply.plyread output conventions.
GSData read_ply(const std::string& path);

// Write an uncompressed binary-little-endian PLY in canonical INRIA property
// order (x,y,z, f_dc, f_rest, opacity, scale, rot).
void write_ply(const std::string& path, const GSData& data);

}  // namespace gsplycpp
