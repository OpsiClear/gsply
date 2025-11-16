"""Tests for + operator on GSData and GSTensor."""

import numpy as np
import pytest

import gsply


class TestGSDataAddOperator:
    """Test + operator for GSData."""

    def test_add_operator_basic(self, sample_gaussian_data):
        """Test that + operator works for GSData."""
        means = sample_gaussian_data["means"]
        scales = sample_gaussian_data["scales"]
        quats = sample_gaussian_data["quats"]
        opacities = sample_gaussian_data["opacities"]
        sh0 = sample_gaussian_data["sh0"]
        shN = sample_gaussian_data["shN"]

        # Create two GSData objects
        data1 = gsply.GSData(
            means=means[:50],
            scales=scales[:50],
            quats=quats[:50],
            opacities=opacities[:50],
            sh0=sh0[:50],
            shN=shN[:50] if shN is not None else None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        data2 = gsply.GSData(
            means=means[50:],
            scales=scales[50:],
            quats=quats[50:],
            opacities=opacities[50:],
            sh0=sh0[50:],
            shN=shN[50:] if shN is not None else None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        # Test + operator
        combined = data1 + data2

        # Verify result
        assert len(combined) == len(data1) + len(data2)
        assert combined.means.shape[0] == 100
        np.testing.assert_array_equal(combined.means[:50], data1.means)
        np.testing.assert_array_equal(combined.means[50:], data2.means)

    def test_add_operator_equals_add_method(self, sample_gaussian_data):
        """Test that + operator produces same result as add() method."""
        means = sample_gaussian_data["means"]
        scales = sample_gaussian_data["scales"]
        quats = sample_gaussian_data["quats"]
        opacities = sample_gaussian_data["opacities"]
        sh0 = sample_gaussian_data["sh0"]
        shN = sample_gaussian_data["shN"]

        data1 = gsply.GSData(
            means=means[:50],
            scales=scales[:50],
            quats=quats[:50],
            opacities=opacities[:50],
            sh0=sh0[:50],
            shN=shN[:50] if shN is not None else None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        data2 = gsply.GSData(
            means=means[50:],
            scales=scales[50:],
            quats=quats[50:],
            opacities=opacities[50:],
            sh0=sh0[50:],
            shN=shN[50:] if shN is not None else None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        # Compare results
        result_add = data1.add(data2)
        result_operator = data1 + data2

        # Should be identical
        assert len(result_add) == len(result_operator)
        np.testing.assert_array_equal(result_add.means, result_operator.means)
        np.testing.assert_array_equal(result_add.scales, result_operator.scales)
        np.testing.assert_array_equal(result_add.quats, result_operator.quats)
        np.testing.assert_array_equal(result_add.opacities, result_operator.opacities)
        np.testing.assert_array_equal(result_add.sh0, result_operator.sh0)

    def test_sum_with_add_operator(self, sample_gaussian_data):
        """Test that sum() works with GSData objects."""
        means = sample_gaussian_data["means"]
        scales = sample_gaussian_data["scales"]
        quats = sample_gaussian_data["quats"]
        opacities = sample_gaussian_data["opacities"]
        sh0 = sample_gaussian_data["sh0"]
        shN = sample_gaussian_data["shN"]

        # Create three GSData objects
        data_list = []
        for i in range(3):
            start = i * 33
            end = (i + 1) * 33
            data_list.append(
                gsply.GSData(
                    means=means[start:end],
                    scales=scales[start:end],
                    quats=quats[start:end],
                    opacities=opacities[start:end],
                    sh0=sh0[start:end],
                    shN=shN[start:end] if shN is not None else None,
                    masks=None,
                    mask_names=None,
                    _base=None,
                )
            )

        # Test sum()
        combined = sum(data_list)

        # Verify result
        assert len(combined) == 99  # 33 * 3
        assert combined.means.shape[0] == 99

    def test_add_operator_chains(self, sample_gaussian_data):
        """Test chaining multiple + operations."""
        means = sample_gaussian_data["means"]
        scales = sample_gaussian_data["scales"]
        quats = sample_gaussian_data["quats"]
        opacities = sample_gaussian_data["opacities"]
        sh0 = sample_gaussian_data["sh0"]

        # Create three GSData objects (without shN for simplicity)
        data1 = gsply.GSData(
            means=means[:25],
            scales=scales[:25],
            quats=quats[:25],
            opacities=opacities[:25],
            sh0=sh0[:25],
            shN=None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        data2 = gsply.GSData(
            means=means[25:50],
            scales=scales[25:50],
            quats=quats[25:50],
            opacities=opacities[25:50],
            sh0=sh0[25:50],
            shN=None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        data3 = gsply.GSData(
            means=means[50:75],
            scales=scales[50:75],
            quats=quats[50:75],
            opacities=opacities[50:75],
            sh0=sh0[50:75],
            shN=None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        # Chain operations
        combined = data1 + data2 + data3

        # Verify result
        assert len(combined) == 75
        np.testing.assert_array_equal(combined.means[:25], data1.means)
        np.testing.assert_array_equal(combined.means[25:50], data2.means)
        np.testing.assert_array_equal(combined.means[50:75], data3.means)


class TestGSTensorAddOperator:
    """Test + operator for GSTensor."""

    def test_add_operator_basic(self, sample_gaussian_data):
        """Test that + operator works for GSTensor."""
        pytest.importorskip("torch")
        from gsply.torch import GSTensor

        means = sample_gaussian_data["means"]
        scales = sample_gaussian_data["scales"]
        quats = sample_gaussian_data["quats"]
        opacities = sample_gaussian_data["opacities"]
        sh0 = sample_gaussian_data["sh0"]
        shN = sample_gaussian_data["shN"]

        # Create two GSData objects
        data1 = gsply.GSData(
            means=means[:50],
            scales=scales[:50],
            quats=quats[:50],
            opacities=opacities[:50],
            sh0=sh0[:50],
            shN=shN[:50] if shN is not None else None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        data2 = gsply.GSData(
            means=means[50:],
            scales=scales[50:],
            quats=quats[50:],
            opacities=opacities[50:],
            sh0=sh0[50:],
            shN=shN[50:] if shN is not None else None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        # Convert to GSTensor
        gstensor1 = GSTensor.from_gsdata(data1, device="cpu")
        gstensor2 = GSTensor.from_gsdata(data2, device="cpu")

        # Test + operator
        combined = gstensor1 + gstensor2

        # Verify result
        assert len(combined) == len(gstensor1) + len(gstensor2)
        assert combined.means.shape[0] == 100

    def test_add_operator_equals_add_method(self, sample_gaussian_data):
        """Test that + operator produces same result as add() method for GSTensor."""
        pytest.importorskip("torch")
        from gsply.torch import GSTensor

        means = sample_gaussian_data["means"]
        scales = sample_gaussian_data["scales"]
        quats = sample_gaussian_data["quats"]
        opacities = sample_gaussian_data["opacities"]
        sh0 = sample_gaussian_data["sh0"]
        shN = sample_gaussian_data["shN"]

        data1 = gsply.GSData(
            means=means[:50],
            scales=scales[:50],
            quats=quats[:50],
            opacities=opacities[:50],
            sh0=sh0[:50],
            shN=shN[:50] if shN is not None else None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        data2 = gsply.GSData(
            means=means[50:],
            scales=scales[50:],
            quats=quats[50:],
            opacities=opacities[50:],
            sh0=sh0[50:],
            shN=shN[50:] if shN is not None else None,
            masks=None,
            mask_names=None,
            _base=None,
        )

        gstensor1 = GSTensor.from_gsdata(data1, device="cpu")
        gstensor2 = GSTensor.from_gsdata(data2, device="cpu")

        # Compare results
        result_add = gstensor1.add(gstensor2)
        result_operator = gstensor1 + gstensor2

        # Should be identical
        assert len(result_add) == len(result_operator)
        assert result_add.means.shape == result_operator.means.shape
