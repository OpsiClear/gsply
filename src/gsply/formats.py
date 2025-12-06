"""Format detection and constants for Gaussian splatting PLY files."""

from pathlib import Path
from typing import Any

__all__ = [
    "detect_format",
    "get_sh_degree_from_property_count",
    "SH_C0",
    "CHUNK_SIZE",
    "CHUNK_SIZE_SHIFT",
    "PROPERTY_COUNTS_BY_SH_DEGREE",
    "PROPERTY_COUNT_TO_SH_DEGREE",
    "SH_BANDS_TO_DEGREE",
    "EXPECTED_PROPERTIES_BY_SH_DEGREE",
    "SH_DEGREE_TO_COEFFS",
    # Quantization constants
    "INV_2047",
    "INV_1023",
    "INV_255",
    "QUANTIZE_11_BIT_MAX",
    "QUANTIZE_10_BIT_MAX",
    "QUANTIZE_8_BIT_MAX",
    "MASK_11_BIT",
    "MASK_10_BIT",
    "MASK_8_BIT",
    "MASK_2_BIT",
    "POSITION_X_SHIFT",
    "POSITION_Y_SHIFT",
    "POSITION_Z_SHIFT",
    "QUAT_INDEX_SHIFT",
    "QUAT_A_SHIFT",
    "QUAT_B_SHIFT",
    "QUAT_C_SHIFT",
    "COLOR_R_SHIFT",
    "COLOR_G_SHIFT",
    "COLOR_B_SHIFT",
    "COLOR_O_SHIFT",
    "QUAT_NORM",
]


# Property counts by SH degree
PROPERTY_COUNTS_BY_SH_DEGREE = {
    0: 14,  # xyz(3) + f_dc(3) + opacity(1) + scales(3) + quats(4)
    1: 23,  # +9 f_rest
    2: 38,  # +24 f_rest
    3: 59,  # +45 f_rest
}

# Reverse lookup: property count -> SH degree (for fast lookup)
PROPERTY_COUNT_TO_SH_DEGREE = {
    14: 0,
    23: 1,
    38: 2,
    59: 3,
}

# SH bands to degree mapping (for shN.shape[1] -> degree conversion)
SH_BANDS_TO_DEGREE = {
    3: 1,  # SH1: 3 bands
    8: 2,  # SH2: 8 bands
    15: 3,  # SH3: 15 bands
}

# Mapping from SH degree to number of f_rest bands
_SH_DEGREE_TO_REST_BANDS = {
    0: 0,
    1: 9,
    2: 24,
    3: 45,
}


def _build_property_list(sh_degree: int) -> list[str]:
    """Build property list for given SH degree.

    :param sh_degree: SH degree (0-3)
    :returns: List of property names in order
    """
    base_properties = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    ]
    rest_bands = _SH_DEGREE_TO_REST_BANDS[sh_degree]
    rest_properties = [f"f_rest_{i}" for i in range(rest_bands)]
    suffix_properties = [
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    return base_properties + rest_properties + suffix_properties


# Expected property names in order for each SH degree
EXPECTED_PROPERTIES_BY_SH_DEGREE: dict[int, list[str]] = {
    degree: _build_property_list(degree) for degree in range(4)
}

# Compressed format constants
CHUNK_SIZE = 256
CHUNK_SIZE_SHIFT = 8  # log2(256) - for fast division using bit shift
COMPRESSED_CHUNK_PROPERTIES = 18  # min/max bounds (6*3)
COMPRESSED_VERTEX_PROPERTIES = 4  # packed position, rotation, scale, color

# SH coefficient for color conversion
SH_C0 = 0.28209479177387814  # sqrt(1/(4*pi))

# ======================================================================================
# QUANTIZATION CONSTANTS (shared between reader.py and writer.py)
# ======================================================================================

# Quantization inverses for unpacking (pre-computed for multiplication instead of division)
INV_2047 = 1.0 / 2047.0  # 11-bit unpacking
INV_1023 = 1.0 / 1023.0  # 10-bit unpacking
INV_255 = 1.0 / 255.0  # 8-bit unpacking

# Quantization maxima for packing
QUANTIZE_11_BIT_MAX = 2047.0  # 2^11 - 1 (X and Z coordinates)
QUANTIZE_10_BIT_MAX = 1023.0  # 2^10 - 1 (Y coordinate and quaternions)
QUANTIZE_8_BIT_MAX = 255.0  # 2^8 - 1 (RGB and opacity)

# Bit masks for extracting packed values
MASK_11_BIT = 0x7FF  # 2^11 - 1 = 2047 (11-bit mask for X and Z coordinates)
MASK_10_BIT = 0x3FF  # 2^10 - 1 = 1023 (10-bit mask for Y coordinate and quaternions)
MASK_8_BIT = 0xFF  # 2^8 - 1 = 255 (8-bit mask for RGB and opacity)
MASK_2_BIT = 0x3  # 2^2 - 1 = 3 (2-bit mask for quaternion index)

# Bit shift positions for unpacking/packing 32-bit integers
# Position/Scale: (X:11 bits)(Y:10 bits)(Z:11 bits)
POSITION_X_SHIFT = 21  # bits 31-21: X coordinate (11 bits)
POSITION_Y_SHIFT = 11  # bits 20-11: Y coordinate (10 bits)
POSITION_Z_SHIFT = 0  # bits 10-0: Z coordinate (11 bits)

# Quaternion: (largest_idx:2 bits)(qa:10 bits)(qb:10 bits)(qc:10 bits)
QUAT_INDEX_SHIFT = 30  # bits 31-30: largest component index (2 bits)
QUAT_A_SHIFT = 20  # bits 29-20: first remaining component (10 bits)
QUAT_B_SHIFT = 10  # bits 19-10: second remaining component (10 bits)
QUAT_C_SHIFT = 0  # bits 9-0: third remaining component (10 bits)

# Color: (R:8 bits)(G:8 bits)(B:8 bits)(Opacity:8 bits)
COLOR_R_SHIFT = 24  # bits 31-24: red channel (8 bits)
COLOR_G_SHIFT = 16  # bits 23-16: green channel (8 bits)
COLOR_B_SHIFT = 8  # bits 15-8: blue channel (8 bits)
COLOR_O_SHIFT = 0  # bits 7-0: opacity (8 bits)

# Quaternion norm constant
QUAT_NORM = 1.4142135623730951  # 1.0 / (sqrt(2) * 0.5) = sqrt(2)

# SH degree to f_rest coefficients mapping (for PLY format)
SH_DEGREE_TO_COEFFS = {0: 0, 1: 3, 2: 8, 3: 15}


def _parse_ply_header(file_path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Parse PLY header to extract element definitions.

    :param file_path: Path to PLY file
    :returns: Tuple of (elements_dict, header_size_bytes)
    :raises ValueError: If PLY format is invalid
    """
    elements = {}
    current_element = None
    header_size = 0

    with open(file_path, "rb") as f:
        line = f.readline()
        header_size += len(line)

        if line.strip() != b"ply":
            raise ValueError("Not a valid PLY file")

        while True:
            line = f.readline()
            header_size += len(line)
            line_str = line.decode("ascii").strip()

            if line_str == "end_header":
                break

            parts = line_str.split()
            if not parts:
                continue

            if parts[0] == "element":
                element_name = parts[1]
                element_count = int(parts[2])
                elements[element_name] = {"count": element_count, "properties": []}
                current_element = element_name

            elif parts[0] == "property" and current_element:
                prop_type = parts[1]
                prop_name = parts[2]
                elements[current_element]["properties"].append((prop_type, prop_name))

    return elements, header_size


def detect_format(file_path: str | Path) -> tuple[bool, int | None]:
    """Detect PLY format type and SH degree.

    :param file_path: Path to PLY file
    :returns: Tuple of (is_compressed, sh_degree) where is_compressed is True if compressed format,
              False if uncompressed, and sh_degree is 0-3 for uncompressed, None for compressed or unknown

    Example:
        >>> is_compressed, sh_degree = detect_format("model.ply")
        >>> if is_compressed:
        ...     print("Compressed format")
        ... else:
        ...     print(f"Uncompressed SH degree {sh_degree}")
    """
    file_path = Path(file_path)

    # Check file existence first
    if not file_path.exists():
        return False, None

    try:
        elements, _ = _parse_ply_header(file_path)
    except Exception:
        return False, None

    # Check for compressed format
    if _is_compressed_format(elements):
        return True, None

    # Check for uncompressed format
    if "vertex" not in elements:
        return False, None

    vertex_props = elements["vertex"]["properties"]
    property_count = len(vertex_props)

    # Try to match against known SH degrees
    for sh_degree, expected_count in PROPERTY_COUNTS_BY_SH_DEGREE.items():
        if property_count == expected_count:
            # Verify property names match
            prop_names = [prop[1] for prop in vertex_props]
            expected_names = EXPECTED_PROPERTIES_BY_SH_DEGREE[sh_degree]
            if prop_names == expected_names:
                return False, sh_degree

    # Unknown format
    return False, None


# Expected chunk property names for compressed format
_EXPECTED_CHUNK_NAMES = [
    "min_x",
    "min_y",
    "min_z",
    "max_x",
    "max_y",
    "max_z",
    "min_scale_x",
    "min_scale_y",
    "min_scale_z",
    "max_scale_x",
    "max_scale_y",
    "max_scale_z",
    "min_r",
    "min_g",
    "min_b",
    "max_r",
    "max_g",
    "max_b",
]

# Expected vertex property names for compressed format
_EXPECTED_VERTEX_NAMES = [
    "packed_position",
    "packed_rotation",
    "packed_scale",
    "packed_color",
]


def _is_compressed_format(elements: dict[str, dict[str, Any]]) -> bool:
    """Check if elements dict represents compressed format.

    :param elements: Parsed PLY header elements
    :returns: True if compressed format detected
    """
    # Must have chunk and vertex elements
    required_elements = {"chunk", "vertex"}
    if not required_elements.issubset(elements):
        return False

    chunk_elem = elements["chunk"]
    vertex_elem = elements["vertex"]

    # Check chunk element (18 float properties)
    chunk_props = chunk_elem["properties"]
    if len(chunk_props) != COMPRESSED_CHUNK_PROPERTIES:
        return False

    # Verify chunk property types and names match expected format
    if not all(
        prop_type == "float" and prop_name == expected_name
        for (prop_type, prop_name), expected_name in zip(
            chunk_props, _EXPECTED_CHUNK_NAMES, strict=False
        )
    ):
        return False

    # Check vertex element (4 uint properties)
    vertex_props = vertex_elem["properties"]
    if len(vertex_props) < COMPRESSED_VERTEX_PROPERTIES:
        return False

    # Verify vertex property types and names match expected format
    if not all(
        prop_type == "uint" and prop_name == expected_name
        for (prop_type, prop_name), expected_name in zip(
            vertex_props[:COMPRESSED_VERTEX_PROPERTIES], _EXPECTED_VERTEX_NAMES, strict=False
        )
    ):
        return False

    # Check chunk count matches splat count
    num_chunks = chunk_elem["count"]
    num_vertices = vertex_elem["count"]
    expected_chunks = (num_vertices + CHUNK_SIZE - 1) // CHUNK_SIZE

    return num_chunks == expected_chunks


def get_sh_degree_from_property_count(property_count: int) -> int | None:
    """Get SH degree from property count.

    :param property_count: Number of properties in vertex element
    :returns: SH degree (0-3) or None if unknown
    """
    return PROPERTY_COUNT_TO_SH_DEGREE.get(property_count)
