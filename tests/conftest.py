"""Pytest configuration for gsply tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_gaussian_data():
    """Create sample Gaussian data for testing."""
    np.random.seed(42)  # For reproducibility

    num_gaussians = 100
    data = {
        'means': np.random.randn(num_gaussians, 3).astype(np.float32),
        'scales': np.random.randn(num_gaussians, 3).astype(np.float32),
        'quats': np.random.randn(num_gaussians, 4).astype(np.float32),
        'opacities': np.random.randn(num_gaussians).astype(np.float32),
        'sh0': np.random.randn(num_gaussians, 3).astype(np.float32),
        'shN': np.random.randn(num_gaussians, 15, 3).astype(np.float32),
    }

    return data


@pytest.fixture
def sample_sh0_data():
    """Create sample SH degree 0 data (no higher-order SH)."""
    np.random.seed(42)

    num_gaussians = 50
    data = {
        'means': np.random.randn(num_gaussians, 3).astype(np.float32),
        'scales': np.random.randn(num_gaussians, 3).astype(np.float32),
        'quats': np.random.randn(num_gaussians, 4).astype(np.float32),
        'opacities': np.random.randn(num_gaussians).astype(np.float32),
        'sh0': np.random.randn(num_gaussians, 3).astype(np.float32),
    }

    return data


@pytest.fixture
def test_ply_file():
    """Path to test PLY file (if available)."""
    test_file = Path("../export_with_edits/frame_00000.ply")
    if test_file.exists():
        return test_file
    return None


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_test_file: marks tests that require test PLY file"
    )
