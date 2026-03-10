"""Tests for Contact Maps head: JAX vs PyTorch comparison.

Contact Maps use pair-based predictions from pair embeddings.
- 28 tracks
- Output shape: (B, S, S, 28) where S = seq_len / 128
"""

import numpy as np
import pytest

from .comparison_utils import compare_arrays


@pytest.mark.integration_jax
@pytest.mark.jax
class TestContactMaps:
    """Contact Maps head comparison tests."""

    HEAD_NAME = "contact_maps"
    PYTORCH_KEY = "pair_activations"  # PyTorch uses this key
    NUM_TRACKS = 28

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_contact_maps_comparison(
        self,
        cached_predictions,
        tolerances,
        organism,
    ):
        """Test Contact Maps predictions with threshold-based tolerance.

        Contact maps have many near-zero values where small absolute differences
        cause huge relative differences. We use a two-tier check:
        - For values > threshold: check relative tolerance
        - For values <= threshold: check relaxed absolute tolerance
        """
        jax_out = cached_predictions[organism]["jax"]
        pt_out = cached_predictions[organism]["pytorch"]

        jax_arr = jax_out[self.HEAD_NAME]["predictions"]
        pt_arr = pt_out[self.PYTORCH_KEY]

        # Threshold to separate "significant" values from near-zero values
        threshold = 0.1
        rtol = tolerances["rtol"]  # 1% relative tolerance for large values
        atol_small = 0.003  # Relaxed absolute tolerance for near-zero values

        abs_diff = np.abs(pt_arr - jax_arr)
        jax_abs = np.abs(jax_arr)

        # Split into large and small value regions
        large_mask = jax_abs > threshold
        small_mask = ~large_mask

        # Check 1: Large values should match within relative tolerance
        if np.any(large_mask):
            large_rel_diff = abs_diff[large_mask] / jax_abs[large_mask]
            large_max_rel = float(large_rel_diff.max())
            large_mean_rel = float(large_rel_diff.mean())
            large_passed = large_max_rel <= rtol
        else:
            large_max_rel = 0.0
            large_mean_rel = 0.0
            large_passed = True

        # Check 2: Small values should match within absolute tolerance
        if np.any(small_mask):
            small_abs_diff = abs_diff[small_mask]
            small_max_abs = float(small_abs_diff.max())
            small_mean_abs = float(small_abs_diff.mean())
            small_passed = small_max_abs <= atol_small
        else:
            small_max_abs = 0.0
            small_mean_abs = 0.0
            small_passed = True

        # Build detailed message
        n_large = int(large_mask.sum())
        n_small = int(small_mask.sum())
        total = pt_arr.size

        msg = (
            f"contact_maps_{organism}: "
            f"Large values (>{threshold}): {n_large}/{total} "
            f"[max_rel={large_max_rel:.4%}, mean_rel={large_mean_rel:.4%}] "
            f"{'PASS' if large_passed else 'FAIL'}; "
            f"Small values: {n_small}/{total} "
            f"[max_abs={small_max_abs:.6f}, mean_abs={small_mean_abs:.6f}] "
            f"{'PASS' if small_passed else 'FAIL'}"
        )

        passed = large_passed and small_passed
        assert passed, msg

    def test_contact_maps_shape(self, cached_predictions, sequence_length):
        """Verify contact maps have correct shape (B, S, S, tracks)."""
        pt_out = cached_predictions["human"]["pytorch"]
        arr = pt_out[self.PYTORCH_KEY]

        expected_seq_len = sequence_length // 2048  # 64 (includes 16x pooling)

        assert len(arr.shape) == 4, f"Expected 4D tensor, got {len(arr.shape)}D"
        assert (
            arr.shape[1] == expected_seq_len
        ), f"Expected seq_len {expected_seq_len}, got {arr.shape[1]}"
        assert (
            arr.shape[1] == arr.shape[2]
        ), "Contact map should be square in spatial dims"
        assert arr.shape[3] == self.NUM_TRACKS, f"Expected {self.NUM_TRACKS} tracks"

    def test_contact_maps_symmetry(self, cached_predictions):
        """Contact maps should be symmetric in spatial dimensions."""
        pt_out = cached_predictions["human"]["pytorch"]
        arr = pt_out[self.PYTORCH_KEY]

        # arr shape: (B, S, S, Tracks) - check S x S symmetry
        arr_transposed = np.transpose(arr, (0, 2, 1, 3))  # swap spatial dims

        np.testing.assert_allclose(
            arr, arr_transposed, rtol=1e-5, atol=1e-6,
            err_msg="Contact maps should be symmetric in spatial dimensions"
        )

    def test_contact_maps_finite_and_bounded(self, cached_predictions):
        """Contact map values should be finite and within a reasonable range.

        This test catches NaN/Inf from numerical issues
        and exploding values from bad weights.
        """
        max_reasonable = 1000.0  # Raw outputs shouldn't be astronomically large

        for organism in ["human", "mouse"]:
            pt_out = cached_predictions[organism]["pytorch"]
            arr = pt_out[self.PYTORCH_KEY]

            assert np.all(np.isfinite(arr)), (
                f"Contact maps for {organism} contain non-finite values: "
                f"NaN count={np.isnan(arr).sum()}, "
                f"Inf count={np.isinf(arr).sum()}"
            )
            assert np.all(np.abs(arr) < max_reasonable), (
                f"Contact maps for {organism} contain unreasonably large values: "
                f"max abs={np.abs(arr).max():.2f}"
            )
