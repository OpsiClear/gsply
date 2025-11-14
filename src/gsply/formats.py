"""Format detection and constants for Gaussian splatting PLY files."""

from pathlib import Path
from typing import Any

__all__ = [
    "detect_format",
    "get_sh_degree_from_property_count",
    "SH_C0",
    "CHUNK_SIZE",
    "PROPERTY_COUNTS_BY_SH_DEGREE",
    "PROPERTY_COUNT_TO_SH_DEGREE",
    "EXPECTED_PROPERTIES_BY_SH_DEGREE",
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

# Expected property names in order for each SH degree
EXPECTED_PROPERTIES_BY_SH_DEGREE: dict[int, list[str]] = {
    0: [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ],
    1: [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *[f"f_rest_{i}" for i in range(9)],
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ],
    2: [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *[f"f_rest_{i}" for i in range(24)],
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ],
    3: [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *[f"f_rest_{i}" for i in range(45)],
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ],
}

# Compressed format constants
CHUNK_SIZE = 256
CHUNK_SIZE_SHIFT = 8  # log2(256) - for fast division using bit shift
COMPRESSED_CHUNK_PROPERTIES = 18  # min/max bounds (6*3)
COMPRESSED_VERTEX_PROPERTIES = 4  # packed position, rotation, scale, color

# SH coefficient for color conversion
SH_C0 = 0.28209479177387814  # sqrt(1/(4*pi))


def _parse_ply_header(file_path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Parse PLY header to extract element definitions.

    Args:
        file_path: Path to PLY file

    Returns:
        Tuple of (elements_dict, header_size_bytes)

    Raises:
        ValueError: If PLY format is invalid
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

    Args:
        file_path: Path to PLY file

    Returns:
        Tuple of (is_compressed, sh_degree)
        - is_compressed: True if compressed format, False if uncompressed
        - sh_degree: 0-3 for uncompressed, None for compressed or unknown

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
    if "vertex" in elements:
        vertex_props = elements["vertex"]["properties"]
        property_count = len(vertex_props)

        # Try to match against known SH degrees
        for sh_degree, expected_count in PROPERTY_COUNTS_BY_SH_DEGREE.items():
            if property_count == expected_count:
                # Verify property names match
                prop_names = [p[1] for p in vertex_props]
                expected_names = EXPECTED_PROPERTIES_BY_SH_DEGREE[sh_degree]
                if prop_names == expected_names:
                    return False, sh_degree

    # Unknown format
    return False, None


def _is_compressed_format(elements: dict[str, dict[str, Any]]) -> bool:
    """Check if elements dict represents compressed format.

    Args:
        elements: Parsed PLY header elements

    Returns:
        True if compressed format detected
    """
    # Must have chunk and vertex elements
    if "chunk" not in elements or "vertex" not in elements:
        return False

    chunk_elem = elements["chunk"]
    vertex_elem = elements["vertex"]

    # Check chunk element (18 float properties)
    chunk_props = chunk_elem["properties"]
    if len(chunk_props) != COMPRESSED_CHUNK_PROPERTIES:
        return False

    expected_chunk_names = [
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

    for i, (prop_type, prop_name) in enumerate(chunk_props):
        if prop_type != "float" or prop_name != expected_chunk_names[i]:
            return False

    # Check vertex element (4 uint properties)
    vertex_props = vertex_elem["properties"]
    if len(vertex_props) < COMPRESSED_VERTEX_PROPERTIES:
        return False

    expected_vertex_names = [
        "packed_position",
        "packed_rotation",
        "packed_scale",
        "packed_color",
    ]

    for i in range(COMPRESSED_VERTEX_PROPERTIES):
        prop_type, prop_name = vertex_props[i]
        if prop_type != "uint" or prop_name != expected_vertex_names[i]:
            return False

    # Check chunk count matches splat count
    num_chunks = chunk_elem["count"]
    num_vertices = vertex_elem["count"]
    expected_chunks = (num_vertices + CHUNK_SIZE - 1) // CHUNK_SIZE

    return num_chunks == expected_chunks


def get_sh_degree_from_property_count(property_count: int) -> int | None:
    """Get SH degree from property count.

    Args:
        property_count: Number of properties in vertex element

    Returns:
        SH degree (0-3) or None if unknown
    """
    return PROPERTY_COUNT_TO_SH_DEGREE.get(property_count)
