"""Tests for pre-activation and pre-deactivation functions."""

import numpy as np
import pytest

import gsply
from gsply import GSData, apply_pre_activations, apply_pre_deactivations
from gsply.gsdata import DataFormat


class TestApplyPreActivations:
    """Test apply_pre_activations function."""

    def test_basic_activation(self):
        """Test basic activation of log-scales and logit-opacities."""
        n = 100
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0,  # Log-scales
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),  # Logit-opacities
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        original_scales = data.scales.copy()
        original_opacities = data.opacities.copy()

        result = apply_pre_activations(data, inplace=True)

        # Should modify in-place
        assert result is data

        # Scales should be activated (exp + clamp)
        assert np.all(data.scales > 0)
        assert np.all(data.scales >= 1e-4)  # min_scale default
        assert np.all(data.scales <= 100.0)  # max_scale default

        # Opacities should be activated (sigmoid)
        assert np.all(data.opacities >= 0.0)
        assert np.all(data.opacities <= 1.0)

        # Quaternions should be normalized
        quat_norms = np.linalg.norm(data.quats, axis=1)
        np.testing.assert_allclose(quat_norms, 1.0, rtol=1e-5)

        # Verify activation worked correctly
        expected_scales = np.clip(np.exp(original_scales), 1e-4, 100.0)
        np.testing.assert_allclose(data.scales, expected_scales, rtol=1e-5)

        # Verify sigmoid activation
        from gsply.utils import sigmoid

        expected_opacities = sigmoid(original_opacities)
        np.testing.assert_allclose(data.opacities, expected_opacities, rtol=1e-5)

    def test_activation_not_inplace(self):
        """Test activation with inplace=False creates a copy."""
        n = 50
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        original_scales = data.scales.copy()
        original_opacities = data.opacities.copy()

        result = apply_pre_activations(data, inplace=False)

        # Should return a new object
        assert result is not data

        # Original should be unchanged
        np.testing.assert_array_equal(data.scales, original_scales)
        np.testing.assert_array_equal(data.opacities, original_opacities)

        # Result should be activated
        assert np.all(result.scales > 0)
        assert np.all(result.opacities >= 0.0)
        assert np.all(result.opacities <= 1.0)

    def test_activation_custom_bounds(self):
        """Test activation with custom scale bounds."""
        n = 50
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        result = apply_pre_activations(
            data, min_scale=0.01, max_scale=10.0, min_quat_norm=1e-6, inplace=False
        )

        # Should respect custom bounds
        assert np.all(result.scales >= 0.01)
        assert np.all(result.scales <= 10.0)

    def test_activation_quaternion_normalization(self):
        """Test that quaternions are properly normalized."""
        n = 50
        # Create unnormalized quaternions
        quats = np.random.randn(n, 4).astype(np.float32) * 2.0

        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0,
            quats=quats,
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        result = apply_pre_activations(data, inplace=False)

        # Quaternions should be normalized
        quat_norms = np.linalg.norm(result.quats, axis=1)
        np.testing.assert_allclose(quat_norms, 1.0, rtol=1e-5)

    def test_activation_small_quaternion_norm(self):
        """Test handling of very small quaternion norms."""
        n = 10
        # Create quaternions with very small norm
        quats = np.random.randn(n, 4).astype(np.float32) * 1e-10

        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0,
            quats=quats,
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        result = apply_pre_activations(data, inplace=False)

        # Should handle gracefully (set to [0,0,0,1] for very small norms)
        quat_norms = np.linalg.norm(result.quats, axis=1)
        assert np.all(quat_norms >= 1e-8)  # min_quat_norm default

    def test_activation_validation_errors(self):
        """Test validation errors for invalid inputs."""
        n = 10
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        # Invalid min_scale
        with pytest.raises(ValueError, match="min_scale must be positive"):
            apply_pre_activations(data, min_scale=-1.0, inplace=False)

        # Invalid max_scale
        with pytest.raises(ValueError, match="max_scale"):
            apply_pre_activations(data, max_scale=-1.0, inplace=False)

        # max_scale < min_scale
        with pytest.raises(ValueError, match="max_scale"):
            apply_pre_activations(data, min_scale=10.0, max_scale=5.0, inplace=False)

        # Invalid min_quat_norm
        with pytest.raises(ValueError, match="min_quat_norm"):
            apply_pre_activations(data, min_quat_norm=-1.0, inplace=False)

    def test_activation_roundtrip_with_denormalize(self):
        """Test that activation works correctly with denormalize."""
        n = 100
        # Start with PLY format (log-scales, logit-opacities)
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        original_scales = data.scales.copy()
        original_opacities = data.opacities.copy()

        # Activate (same as denormalize)
        activated = apply_pre_activations(data, inplace=False)

        # Verify activation
        assert np.all(activated.scales > 0)
        assert np.all(activated.opacities >= 0.0)
        assert np.all(activated.opacities <= 1.0)

        # Verify roundtrip: activation should match denormalize
        data_copy = GSData(
            means=data.means.copy(),
            scales=original_scales.copy(),
            quats=data.quats.copy(),
            opacities=original_opacities.copy(),
            sh0=data.sh0.copy(),
            shN=None,
        )
        denormalized = data_copy.denormalize(inplace=False)

        np.testing.assert_allclose(activated.scales, denormalized.scales, rtol=1e-5)
        np.testing.assert_allclose(activated.opacities, denormalized.opacities, rtol=1e-5)


class TestApplyPreDeactivations:
    """Test apply_pre_deactivations function."""

    def test_basic_deactivation(self):
        """Test basic deactivation of linear scales and opacities."""
        n = 100
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32) * 0.1 + 0.01,  # Linear scales
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32) * 0.8 + 0.1,  # Linear opacities
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        original_scales = data.scales.copy()
        original_opacities = data.opacities.copy()

        result = apply_pre_deactivations(data, inplace=True)

        # Should modify in-place
        assert result is data

        # Scales should be deactivated (log)
        assert np.all(np.isfinite(data.scales))

        # Opacities should be deactivated (logit)
        assert np.all(np.isfinite(data.opacities))

        # Verify deactivation worked correctly
        expected_scales = np.log(np.clip(original_scales, 1e-9, None))
        np.testing.assert_allclose(data.scales, expected_scales, rtol=1e-5)

        # Verify logit deactivation
        from gsply.utils import logit

        expected_opacities = logit(np.clip(original_opacities, 1e-4, 1.0 - 1e-4), eps=1e-4)
        np.testing.assert_allclose(data.opacities, expected_opacities, rtol=1e-5)

    def test_deactivation_not_inplace(self):
        """Test deactivation with inplace=False creates a copy."""
        n = 50
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32) * 0.1 + 0.01,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32) * 0.8 + 0.1,
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        original_scales = data.scales.copy()
        original_opacities = data.opacities.copy()

        result = apply_pre_deactivations(data, inplace=False)

        # Should return a new object
        assert result is not data

        # Original should be unchanged
        np.testing.assert_array_equal(data.scales, original_scales)
        np.testing.assert_array_equal(data.opacities, original_opacities)

        # Result should be deactivated
        assert np.all(np.isfinite(result.scales))
        assert np.all(np.isfinite(result.opacities))

    def test_deactivation_custom_bounds(self):
        """Test deactivation with custom opacity bounds."""
        n = 50
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32) * 0.1 + 0.01,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32) * 0.8 + 0.1,
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        result = apply_pre_deactivations(
            data, min_scale=1e-8, min_opacity=1e-5, max_opacity=0.999, inplace=False
        )

        # Should respect custom bounds
        assert np.all(np.isfinite(result.scales))
        assert np.all(np.isfinite(result.opacities))

    def test_deactivation_clamping(self):
        """Test that clamping works correctly for edge cases."""
        n = 50
        # Create data with values outside normal range
        scales = np.random.rand(n, 3).astype(np.float32) * 0.1 + 0.01
        scales[0, :] = 1e-10  # Very small scale
        scales[1, :] = 100.0  # Large scale

        opacities = np.random.rand(n).astype(np.float32) * 0.8 + 0.1
        opacities[0] = 0.0  # Edge case
        opacities[1] = 1.0  # Edge case

        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=scales,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=opacities,
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        result = apply_pre_deactivations(data, inplace=False)

        # Should handle edge cases gracefully
        assert np.all(np.isfinite(result.scales))
        assert np.all(np.isfinite(result.opacities))

    def test_deactivation_validation_errors(self):
        """Test validation errors for invalid inputs."""
        n = 10
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32) * 0.1 + 0.01,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32) * 0.8 + 0.1,
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        # Invalid min_scale
        with pytest.raises(ValueError, match="min_scale must be positive"):
            apply_pre_deactivations(data, min_scale=-1.0, inplace=False)

        # Invalid min_opacity
        with pytest.raises(ValueError, match="min_opacity"):
            apply_pre_deactivations(data, min_opacity=-1.0, inplace=False)

        # Invalid max_opacity
        with pytest.raises(ValueError, match="max_opacity"):
            apply_pre_deactivations(data, max_opacity=1.5, inplace=False)

        # max_opacity <= min_opacity
        with pytest.raises(ValueError, match="max_opacity"):
            apply_pre_deactivations(data, min_opacity=0.5, max_opacity=0.3, inplace=False)

    def test_deactivation_roundtrip_with_normalize(self):
        """Test that deactivation works correctly with normalize."""
        n = 100
        # Start with linear format
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32) * 0.1 + 0.01,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32) * 0.8 + 0.1,
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        original_scales = data.scales.copy()
        original_opacities = data.opacities.copy()

        # Deactivate (same as normalize)
        deactivated = apply_pre_deactivations(data, inplace=False)

        # Verify deactivation
        assert np.all(np.isfinite(deactivated.scales))
        assert np.all(np.isfinite(deactivated.opacities))

        # Verify roundtrip: deactivation should match normalize
        data_copy = GSData(
            means=data.means.copy(),
            scales=original_scales.copy(),
            quats=data.quats.copy(),
            opacities=original_opacities.copy(),
            sh0=data.sh0.copy(),
            shN=None,
        )
        normalized = data_copy.normalize(inplace=False)

        np.testing.assert_allclose(deactivated.scales, normalized.scales, rtol=1e-5)
        np.testing.assert_allclose(deactivated.opacities, normalized.opacities, rtol=1e-5)


class TestIntegrationWithFormatConversion:
    """Test integration of pre-activation/deactivation with format conversion methods."""

    def test_denormalize_uses_apply_pre_activations(self):
        """Test that denormalize() uses apply_pre_activations internally."""
        n = 100
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0,  # Log-scales
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.randn(n).astype(np.float32),  # Logit-opacities
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        # Denormalize should activate scales, opacities, and normalize quaternions
        denormalized = data.denormalize(inplace=False)

        assert np.all(denormalized.scales > 0)
        assert np.all(denormalized.opacities >= 0.0)
        assert np.all(denormalized.opacities <= 1.0)

        # Quaternions should be normalized
        quat_norms = np.linalg.norm(denormalized.quats, axis=1)
        np.testing.assert_allclose(quat_norms, 1.0, rtol=1e-5)

        # Format should be updated
        assert denormalized._format["scales"] == DataFormat.SCALES_LINEAR
        assert denormalized._format["opacities"] == DataFormat.OPACITIES_LINEAR

    def test_normalize_uses_apply_pre_deactivations(self):
        """Test that normalize() uses apply_pre_deactivations internally."""
        n = 100
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32) * 0.1 + 0.01,  # Linear scales
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32) * 0.8 + 0.1,  # Linear opacities
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        # Normalize should deactivate scales and opacities
        normalized = data.normalize(inplace=False)

        assert np.all(np.isfinite(normalized.scales))
        assert np.all(np.isfinite(normalized.opacities))

        # Format should be updated
        assert normalized._format["scales"] == DataFormat.SCALES_PLY
        assert normalized._format["opacities"] == DataFormat.OPACITIES_PLY

    def test_full_roundtrip(self):
        """Test full roundtrip: normalize -> denormalize."""
        n = 100
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32) * 0.1 + 0.01,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32) * 0.8 + 0.1,
            sh0=np.random.randn(n, 3).astype(np.float32),
            shN=None,
        )

        original_scales = data.scales.copy()
        original_opacities = data.opacities.copy()

        # Normalize (linear -> PLY)
        normalized = data.normalize(inplace=False)

        # Denormalize (PLY -> linear)
        denormalized = normalized.denormalize(inplace=False)

        # Should recover original values (within numerical precision)
        np.testing.assert_allclose(denormalized.scales, original_scales, rtol=1e-4)
        np.testing.assert_allclose(denormalized.opacities, original_opacities, rtol=1e-4)

    def test_api_export(self):
        """Test that functions are exported in the main API."""
        assert hasattr(gsply, "apply_pre_activations")
        assert hasattr(gsply, "apply_pre_deactivations")
        assert callable(gsply.apply_pre_activations)
        assert callable(gsply.apply_pre_deactivations)
