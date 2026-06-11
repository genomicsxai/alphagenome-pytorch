"""Unit tests for saturation ISM attribution.

Uses a tiny fake model where mutating position ``p`` to base ``b`` changes the
score by a known delta, so we can verify values, NaN placement, N-position
handling, and batching.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from alphagenome_pytorch.extensions.attribution.ism import saturation_ism
from alphagenome_pytorch.extensions.attribution.types import AttributionResult


# ---------------------------------------------------------------------------
# Fake model where output = sum_over_L(onehot * base_weights)
# ---------------------------------------------------------------------------


class _BaseWeightedModel(nn.Module):
    """Model whose output at each bin is ``(onehot * base_weights).sum()``.

    ``base_weights`` = [1, 2, 3, 4] for A, C, G, T.  Mutating position p
    from ref to alt changes the single-position contribution by
    ``base_weights[alt] - base_weights[ref]``. With ``reduction='sum'``
    over the target window the delta is the same magnitude because only one
    position is flipped at a time.
    """

    BASE_WEIGHTS = torch.tensor([1.0, 2.0, 3.0, 4.0])

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))


def _base_weighted_head(
    model: nn.Module,
    onehot: torch.Tensor,
    organism_index: torch.Tensor,
    *,
    output_type: str,
    resolution: int,
) -> torch.Tensor:
    """HeadSelector for ``_BaseWeightedModel``: 1 track, 1:1 resolution."""
    assert isinstance(model, _BaseWeightedModel)
    bw = _BaseWeightedModel.BASE_WEIGHTS.to(device=onehot.device)
    # (B, L, 4) * (4,) → sum → (B, L)
    val = (onehot * bw).sum(dim=-1)  # (B, L)
    return val.unsqueeze(-1)  # (B, L, 1) — single track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_onehot(seq: str, device: str = "cpu") -> torch.Tensor:
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": -1}
    L = len(seq)
    oh = torch.zeros(1, L, 4, dtype=torch.float32, device=device)
    for i, ch in enumerate(seq):
        idx = base_to_idx[ch]
        if idx >= 0:
            oh[0, i, idx] = 1.0
    return oh


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaturationISM:
    """Saturation ISM unit tests."""

    def test_known_delta(self):
        """Mutating A(=1) → C(=2) at any position should give delta = +1."""
        model = _BaseWeightedModel()
        seq = "AAAA"
        onehot = _make_onehot(seq)
        target_slice = slice(0, 4)

        result = saturation_ism(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=target_slice,
            track_indices=[0],
            reduction="sum",
            batch_size=8,
            head_selector=_base_weighted_head,
        )

        assert isinstance(result, AttributionResult)
        assert result.method == "saturation_ism"
        assert result.kind == "base_matrix"
        assert result.values.shape == (4, 4, 1)

        bw = _BaseWeightedModel.BASE_WEIGHTS.numpy()
        for w in range(4):
            ref_b = 0  # A
            for b in range(4):
                if b == ref_b:
                    # Reference-base column should be NaN.
                    assert np.isnan(result.values[w, b, 0]), (
                        f"Reference cell ({w}, {b}) should be NaN"
                    )
                else:
                    expected_delta = bw[b] - bw[ref_b]
                    np.testing.assert_allclose(
                        result.values[w, b, 0],
                        expected_delta,
                        atol=1e-5,
                        err_msg=f"Delta mismatch at w={w}, alt={b}",
                    )

    def test_reference_column_is_nan(self):
        """The reference-base column must always be NaN."""
        model = _BaseWeightedModel()
        onehot = _make_onehot("ACGT")
        result = saturation_ism(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=slice(0, 4),
            track_indices=[0],
            head_selector=_base_weighted_head,
        )
        ref_bases = [0, 1, 2, 3]  # A, C, G, T
        for w, ref_b in enumerate(ref_bases):
            assert np.isnan(result.values[w, ref_b, 0]), (
                f"Ref cell ({w}, {ref_b}) should be NaN"
            )

    def test_n_positions_all_nan(self):
        """Positions with N should have all-NaN rows."""
        model = _BaseWeightedModel()
        onehot = _make_onehot("ANNG")
        result = saturation_ism(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=slice(0, 4),
            track_indices=[0],
            head_selector=_base_weighted_head,
        )
        # Positions 1 and 2 are N → entire row should be NaN.
        for w in [1, 2]:
            assert np.all(np.isnan(result.values[w, :, 0])), (
                f"N position w={w} should be all NaN"
            )

    def test_batching_matches_unbatched(self):
        """batch_size=2 over 8 positions must give same result as batch_size=100."""
        model = _BaseWeightedModel()
        seq = "ACGTACGT"
        onehot = _make_onehot(seq)
        target_slice = slice(0, 8)

        r_small = saturation_ism(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=target_slice,
            track_indices=[0],
            batch_size=2,
            head_selector=_base_weighted_head,
        )
        r_big = saturation_ism(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=target_slice,
            track_indices=[0],
            batch_size=100,
            head_selector=_base_weighted_head,
        )
        np.testing.assert_allclose(
            np.nan_to_num(r_small.values),
            np.nan_to_num(r_big.values),
            atol=1e-5,
        )

    def test_strand_averaged(self):
        """Strand averaging should produce a finite result without crashing."""
        model = _BaseWeightedModel()
        onehot = _make_onehot("ACGT")
        result = saturation_ism(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=slice(0, 4),
            track_indices=[0],
            strand_averaged=True,
            head_selector=_base_weighted_head,
        )
        # Should succeed and return the same shape.
        assert result.values.shape == (4, 4, 1)
        assert result.metadata.get("strand_averaged") is True

    def test_raw_gradient_is_none(self):
        """ISM results never include raw_gradient."""
        model = _BaseWeightedModel()
        onehot = _make_onehot("ACGT")
        result = saturation_ism(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=slice(0, 4),
            track_indices=[0],
            head_selector=_base_weighted_head,
        )
        assert result.raw_gradient is None

    def test_invalid_batch_size_raises(self):
        model = _BaseWeightedModel()
        onehot = _make_onehot("ACGT")
        with pytest.raises(ValueError, match="batch_size"):
            saturation_ism(
                model,
                onehot=onehot,
                organism_index=0,
                output_type="x",
                resolution=1,
                target_slice=slice(0, 4),
                track_indices=[0],
                batch_size=0,
                head_selector=_base_weighted_head,
            )
