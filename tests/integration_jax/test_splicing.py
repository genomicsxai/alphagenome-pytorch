"""Tests for the splicing heads: PyTorch vs. JAX comparison.

Splice heads predict splicing-related outputs from DNA sequences:
- Classification: 5-class probabilities for splice site type
- Usage: Per-track splice site usage values (sigmoid activated)
- Junction: Pairwise junction counts between positions (softplus activated)
"""

import pytest
import numpy as np

from .comparison_utils import compare_arrays


@pytest.mark.integration_jax
@pytest.mark.jax
class TestSpliceSitesClassification:
    """SpliceSitesClassificationHead comparison tests."""

    NUM_CLASSES = 5
    JAX_OUTPUT_NAME = "splice_sites_classification"
    PT_OUTPUT_NAME = "splice_sites_classification"

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_classification_comparison(
        self,
        cached_predictions,
        tolerances,
        organism,
    ):
        """Test splice site classification predictions match between JAX and PyTorch."""
        jax_out = cached_predictions[organism]["jax"]
        pt_out = cached_predictions[organism]["pytorch"]

        jax_arr = jax_out[self.JAX_OUTPUT_NAME]["predictions"]
        pt_arr = pt_out[self.PT_OUTPUT_NAME]["probs"]

        result = compare_arrays(
            f"splice_classification_{organism}",
            pt_arr,
            jax_arr,
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
        )

        assert result.passed, result.message

    def test_classification_output_shape(self, cached_predictions, sequence_length, batch_size):
        """Verify splice classification output shape is (B, S, 5)."""
        pt_out = cached_predictions["human"]["pytorch"]
        arr = pt_out[self.PT_OUTPUT_NAME]["probs"]

        expected = (batch_size, sequence_length, self.NUM_CLASSES)
        assert arr.shape == expected, f"Shape mismatch: {arr.shape} vs {expected}"

    def test_classification_valid_probabilities(self, cached_predictions):
        """Verify classification outputs are valid probabilities.

        - Values in [0, 1]
        - Sum to 1 across classes (last dimension)
        """
        pt_out = cached_predictions["human"]["pytorch"]
        arr = pt_out[self.PT_OUTPUT_NAME]["probs"]

        # Check range [0, 1]
        assert arr.min() >= 0, f"Probabilities have negative values: min={arr.min()}"
        assert arr.max() <= 1, f"Probabilities exceed 1: max={arr.max()}"

        # Check sum to 1 (softmax property)
        sums = arr.sum(axis=-1)
        np.testing.assert_allclose(
            sums,
            np.ones_like(sums),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Probabilities don't sum to 1 across classes",
        )


@pytest.mark.integration_jax
@pytest.mark.jax
class TestSpliceSitesUsage:
    """SpliceSitesUsageHead comparison tests."""

    NUM_TRACKS = 734
    JAX_OUTPUT_NAME = "splice_sites_usage"
    PT_OUTPUT_NAME = "splice_sites_usage"

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_usage_comparison(
        self,
        cached_predictions,
        tolerances,
        organism,
    ):
        """Test splice site usage predictions match between JAX and PyTorch."""
        jax_out = cached_predictions[organism]["jax"]
        pt_out = cached_predictions[organism]["pytorch"]

        jax_arr = jax_out[self.JAX_OUTPUT_NAME]["predictions"]
        pt_arr = pt_out[self.PT_OUTPUT_NAME]["predictions"]

        result = compare_arrays(
            f"splice_usage_{organism}",
            pt_arr,
            jax_arr,
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
        )

        assert result.passed, result.message

    def test_usage_output_shape(self, cached_predictions, sequence_length, batch_size):
        """Verify splice usage output shape is (B, S, 734)."""
        pt_out = cached_predictions["human"]["pytorch"]
        arr = pt_out[self.PT_OUTPUT_NAME]["predictions"]

        expected = (batch_size, sequence_length, self.NUM_TRACKS)
        assert arr.shape == expected, f"Shape mismatch: {arr.shape} vs {expected}"

    def test_usage_valid_range(self, cached_predictions):
        """Verify usage outputs are in valid sigmoid range [0, 1]."""
        pt_out = cached_predictions["human"]["pytorch"]
        arr = pt_out[self.PT_OUTPUT_NAME]["predictions"]

        assert arr.min() >= 0, f"Usage has negative values: min={arr.min()}"
        assert arr.max() <= 1, f"Usage exceeds 1: max={arr.max()}"


@pytest.mark.integration_jax
@pytest.mark.jax
class TestSpliceSitesJunction:
    """SpliceSitesJunctionHead comparison tests."""

    NUM_TRACKS = 734
    JAX_OUTPUT_NAME = "splice_sites_junction"
    PT_OUTPUT_NAME = "splice_sites_junction"

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_junction_comparison(
        self,
        cached_junction_with_shared_positions,
        tolerances,
        organism,
    ):
        """Test splice junction predictions match between JAX and PyTorch.

        Uses shared splice_site_positions from JAX to ensure both frameworks
        compute junction outputs at identical positions. This is necessary
        because small numerical differences in classification probabilities
        can cause different top-K position selections.
        """
        jax_out = cached_junction_with_shared_positions[organism]["jax"]
        pt_out = cached_junction_with_shared_positions[organism]["pytorch"]

        jax_arr = jax_out[self.JAX_OUTPUT_NAME]["predictions"]
        pt_arr = pt_out[self.PT_OUTPUT_NAME]["pred_counts"]

        result = compare_arrays(
            f"splice_junction_{organism}",
            pt_arr,
            jax_arr,
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
        )

        assert result.passed, result.message

    def test_junction_output_shape(self, cached_predictions, batch_size):
        """Verify splice junction output shape is (B, P, P, 734) - 4D square tensor."""
        pt_out = cached_predictions["human"]["pytorch"]
        arr = pt_out[self.PT_OUTPUT_NAME]["pred_counts"]

        assert len(arr.shape) == 4, f"Expected 4D tensor, got {len(arr.shape)}D"
        assert arr.shape[0] == batch_size, f"Batch size mismatch: {arr.shape[0]} vs {batch_size}"
        assert arr.shape[1] == 512, (
            f"Expected seq_len {512}, got {arr.shape[1]}"
        )
        assert arr.shape[1] == arr.shape[2], (
            f"Junction map should be square: {arr.shape[1]} vs {arr.shape[2]}"
        )
        assert arr.shape[3] == self.NUM_TRACKS, (
            f"Expected {self.NUM_TRACKS} tracks, got {arr.shape[3]}"
        )

    def test_junction_non_negative(self, cached_predictions):
        """Verify junction outputs are non-negative (softplus property)."""
        pt_out = cached_predictions["human"]["pytorch"]
        arr = pt_out[self.PT_OUTPUT_NAME]["pred_counts"]

        assert arr.min() >= 0, f"Junction counts have negative values: min={arr.min()}"

    def test_junction_mask_shape(self, cached_predictions, batch_size):
        """Verify junction mask exists and has matching shape."""
        pt_out = cached_predictions["human"]["pytorch"]

        # Check mask exists
        assert "splice_junction_mask" in pt_out[self.PT_OUTPUT_NAME], (
            "Junction output should include 'splice_junction_mask' key"
        )

        mask = pt_out[self.PT_OUTPUT_NAME]["splice_junction_mask"]
        pred = pt_out[self.PT_OUTPUT_NAME]["pred_counts"]

        # Mask should be (B, P, P, 2*T) - same dims as predictions
        assert mask.shape == pred.shape, (
            f"Mask shape {mask.shape} should match prediction shape {pred.shape}"
        )
