"""Pytest configuration for gsply tests."""

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_gaussian_data():
    """Create sample Gaussian data for testing."""
    np.random.seed(42)  # For reproducibility

    num_gaussians = 100
    return {
        "means": np.random.randn(num_gaussians, 3).astype(np.float32),
        "scales": np.random.randn(num_gaussians, 3).astype(np.float32),
        "quats": np.random.randn(num_gaussians, 4).astype(np.float32),
        "opacities": np.random.randn(num_gaussians).astype(np.float32),
        "sh0": np.random.randn(num_gaussians, 3).astype(np.float32),
        "shN": np.random.randn(num_gaussians, 15, 3).astype(np.float32),
    }


@pytest.fixture
def sample_sh0_data():
    """Create sample SH degree 0 data (no higher-order SH)."""
    np.random.seed(42)

    num_gaussians = 50
    return {
        "means": np.random.randn(num_gaussians, 3).astype(np.float32),
        "scales": np.random.randn(num_gaussians, 3).astype(np.float32),
        "quats": np.random.randn(num_gaussians, 4).astype(np.float32),
        "opacities": np.random.randn(num_gaussians).astype(np.float32),
        "sh0": np.random.randn(num_gaussians, 3).astype(np.float32),
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory from environment or default."""
    data_dir_str = os.getenv("GSPLY_TEST_DATA_DIR", r"D:\4D\all_plys")
    return Path(data_dir_str)


@pytest.fixture
def test_ply_file(test_data_dir):
    """Path to test PLY file (if available)."""
    if not test_data_dir.exists():
        return None

    # Try frame_0.ply first
    test_file = test_data_dir / "frame_0.ply"
    if test_file.exists():
        return test_file

    # Fall back to first available .ply file
    ply_files = list(test_data_dir.glob("*.ply"))
    if ply_files:
        return ply_files[0]

    return None


@pytest.fixture
def test_compressed_ply_file(test_data_dir):
    """Get a compressed PLY file if available."""
    if not test_data_dir.exists():
        return None

    # Check in compressed subdirectory
    compressed_dir = test_data_dir / "20251112_compressed-ply"
    if compressed_dir.exists():
        compressed_files = list(compressed_dir.glob("*.compressed.ply"))
        if not compressed_files:
            compressed_files = list(compressed_dir.glob("*.ply"))
        if compressed_files:
            return compressed_files[0]

    # Fall back to any .compressed.ply in main directory
    compressed_files = list(test_data_dir.glob("*.compressed.ply"))
    if compressed_files:
        return compressed_files[0]

    return None


@pytest.fixture(scope="session")
def multiple_test_files(test_data_dir):
    """Get multiple test files for batch testing."""
    if not test_data_dir.exists():
        return []

    ply_files = sorted(test_data_dir.glob("*.ply"))[:5]  # Get first 5
    return list(ply_files)


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "requires_test_file: marks tests that require test PLY file")
