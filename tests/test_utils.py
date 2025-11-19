import numpy as np
import pytest

from gsply.utils import logit, sigmoid


def test_cpu_logit_sigmoid():
    """Test CPU-based logit and sigmoid functions."""
    # Test scalar
    p = 0.5
    logit_val = logit(p)
    assert np.isclose(logit_val, 0.0)
    s = sigmoid(logit_val)
    assert np.isclose(s, p)

    # Test array
    p = np.array([0.1, 0.5, 0.9])
    logit_val = logit(p)
    s = sigmoid(logit_val)
    assert np.allclose(s, p)

    # Test edge cases (clamping)
    p = np.array([0.0, 1.0])
    logit_val = logit(p)
    assert np.isfinite(logit_val).all()

    # Check values for edge cases
    # logit(0) should be log(eps/(1-eps)) approx log(1e-6) = -13.8
    assert logit_val[0] < -10.0
    assert logit_val[1] > 10.0

    # Test sigmoid stability
    x = np.array([-100.0, 100.0])
    s = sigmoid(x)
    assert np.isclose(s[0], 0.0)
    assert np.isclose(s[1], 1.0)


def test_gpu_logit_sigmoid():
    """Test GPU-based logit and sigmoid functions (requires PyTorch)."""
    torch = pytest.importorskip("torch")

    # Test on CPU tensor first to ensure logic works without CUDA
    device = "cpu"

    # Test scalar tensor
    p = torch.tensor(0.5, device=device)
    logit_val = torch.logit(p)
    assert torch.isclose(logit_val, torch.tensor(0.0, device=device))
    s = torch.sigmoid(logit_val)
    assert torch.isclose(s, p)

    # Test array tensor
    p = torch.tensor([0.1, 0.5, 0.9], device=device)
    logit_val = torch.logit(p)
    s = torch.sigmoid(logit_val)
    assert torch.allclose(s, p)

    # Test edge cases (with eps for clamping)
    p = torch.tensor([0.0, 1.0], device=device)
    logit_val = torch.logit(p, eps=1e-6)
    assert torch.isfinite(logit_val).all()

    # If CUDA is available, test on GPU
    if torch.cuda.is_available():
        device = "cuda"
        p = torch.tensor([0.1, 0.5, 0.9], device=device)
        logit_val = torch.logit(p)
        s = torch.sigmoid(logit_val)
        assert torch.allclose(s, p)
