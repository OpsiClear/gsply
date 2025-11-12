"""Shared utilities for benchmarks and tests."""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the project root directory (parent of benchmarks/)
    """
    return Path(__file__).parent.parent


def get_test_file(name: str = "frame_00000.ply") -> Path:
    """Get path to test PLY file.

    Args:
        name: Name of the test file (default: frame_00000.ply)

    Returns:
        Absolute path to the test file

    Raises:
        FileNotFoundError: If test file does not exist
    """
    test_file = get_project_root() / "export_with_edits" / name
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}\n"
            f"Please ensure test data is available in export_with_edits/"
        )
    return test_file


def get_test_data_dir() -> Path:
    """Get path to test data directory (for bulk testing).

    Uses GSPLY_TEST_DATA_DIR environment variable if set,
    otherwise defaults to D:/4D/all_plys

    Returns:
        Path to test data directory

    Raises:
        FileNotFoundError: If directory does not exist
    """
    data_dir_str = os.getenv("GSPLY_TEST_DATA_DIR", "D:/4D/all_plys")
    data_dir = Path(data_dir_str)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Test data directory not found: {data_dir}\n"
            f"Set GSPLY_TEST_DATA_DIR environment variable or ensure "
            f"default directory exists"
        )

    return data_dir
