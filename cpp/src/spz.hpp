#pragma once

#include <string>

#include "gsdata.hpp"

namespace gsplycpp {

// Read a Niantic SPZ file (legacy gzip v1/v2/v3) into PLY-format GSData.
// Parity with gsply.read_spz: raw frame (no coordinate conversion).
GSData read_spz(const std::string& path);

// Write GSData to SPZ (gzip v3, smallest-three quaternions). Parity with
// gsply.write_spz.
void write_spz(const std::string& path, const GSData& data, int fractional_bits = 12);

}  // namespace gsplycpp
