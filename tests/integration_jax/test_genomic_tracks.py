"""Consolidated genomic track prediction tests.

Tests all genomic track heads (ATAC, DNase, PRO-cap, CAGE, RNA-seq, ChIP-TF, ChIP-Histone)
with parametrized test patterns. This consolidates the following test files:
- test_atacseq.py
- test_accessible_chromatin.py
- test_rnaseq.py
- test_chipseq.py

Each head is tested for:
- JAX vs PyTorch prediction parity
- Output shape validation
- Track count verification
- Head-specific validation (e.g., RNA-seq non-negative values)
"""

import pytest
import numpy as np

from .comparison_utils import compare_arrays


@pytest.mark.integration_jax
@pytest.mark.jax
class TestGenomicTracks:
    """Test all genomic track heads with parametrization."""

    # (head_name, num_tracks, resolutions, apply_squashing)
    HEADS = [
        ("atac", 256, [1, 128], False),
        ("dnase", 384, [1, 128], False),
        ("procap", 128, [1, 128], False),
        ("cage", 640, [1, 128], False),
        ("rna_seq", 768, [1, 128], True),  # apply_squashing=True
        ("chip_tf", 1664, [128], False),
        ("chip_histone", 1152, [128], False),
    ]

    @pytest.mark.parametrize(
        "head_name,num_tracks,resolutions,apply_squashing",
        HEADS,
        ids=[h[0] for h in HEADS]
    )
    @pytest.mark.parametrize("organism", ["human", "mouse"])
    @pytest.mark.parametrize("resolution", [1, 128])
    def test_head_prediction_parity(
        self,
        cached_predictions,
        tolerances,
        organism,
        resolution,
        head_name,
        num_tracks,
        resolutions,
        apply_squashing,
    ):
        """Test JAX vs PyTorch prediction parity for genomic track heads.

        Args:
            cached_predictions: Session-scoped fixture with cached JAX and PyTorch predictions
            tolerances: Relative and absolute tolerance settings
            organism: "human" or "mouse"
            resolution: 1 or 128 (bp)
            head_name: Name of the genomic track head
            num_tracks: Expected number of tracks for this head
            resolutions: List of supported resolutions for this head
            apply_squashing: Whether this head uses squashing (power law expansion)
        """
        if resolution not in resolutions:
            pytest.skip(f"{head_name} doesn't support {resolution}bp resolution")

        jax_out = cached_predictions[organism]["jax"]
        pt_out = cached_predictions[organism]["pytorch"]

        # Extract predictions
        jax_key = f"predictions_{resolution}bp"
        jax_arr = jax_out[head_name][jax_key]
        pt_arr = pt_out[head_name][resolution]

        # Compare
        result = compare_arrays(
            f"{head_name}_{resolution}bp_{organism}",
            pt_arr,
            jax_arr,
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
        )

        assert result.passed, result.message

    @pytest.mark.parametrize(
        "head_name,num_tracks,resolutions,apply_squashing",
        HEADS,
        ids=[h[0] for h in HEADS]
    )
    def test_head_output_shape(
        self,
        cached_predictions,
        expected_output_shapes,
        head_name,
        num_tracks,
        resolutions,
        apply_squashing,
    ):
        """Verify output shapes for each head match expected values."""
        pt_out = cached_predictions["human"]["pytorch"]
        head_out = pt_out[head_name]

        for res in resolutions:
            expected = expected_output_shapes[head_name][res]
            actual = head_out[res].shape
            assert (
                actual == expected
            ), f"{head_name} shape mismatch at {res}bp: {actual} vs {expected}"

    @pytest.mark.parametrize(
        "head_name,num_tracks,resolutions,apply_squashing",
        HEADS,
        ids=[h[0] for h in HEADS]
    )
    def test_head_track_count(
        self,
        cached_predictions,
        head_name,
        num_tracks,
        resolutions,
        apply_squashing,
    ):
        """Verify each head outputs the correct number of tracks."""
        pt_out = cached_predictions["human"]["pytorch"]
        head_out = pt_out[head_name]

        for res in resolutions:
            actual_tracks = head_out[res].shape[-1]
            assert (
                actual_tracks == num_tracks
            ), f"{head_name} expected {num_tracks} tracks at {res}bp, got {actual_tracks}"

    def test_rnaseq_non_negative(self, cached_predictions):
        """Verify RNA-seq outputs are non-negative after unscaling.

        RNA-seq uses apply_squashing=True (power law expansion), so after
        softplus + unscaling, values should be non-negative.
        """
        pt_out = cached_predictions["human"]["pytorch"]

        for res in [1, 128]:
            arr = pt_out["rna_seq"][res]
            # Mask out NaN values (tracks with NaN means)
            valid_mask = ~np.isnan(arr)
            if valid_mask.sum() > 0:
                min_val = arr[valid_mask].min()
                # After softplus + unscaling, values should be non-negative
                assert min_val >= 0, f"RNA-seq {res}bp has negative values: min={min_val}"
