"""Unit tests for variant scoring aggregations (no JAX dependency)."""

import pytest
import torch

from alphagenome_pytorch.variant_scoring.aggregations import create_center_mask


@pytest.mark.unit
class TestCreateCenterMask:
    """Regression tests for the upstream center-mask off-by-one fix.

    Upstream switched from 1-based ``variant.position`` to 0-based
    ``variant.start`` to center the mask. Our implementation accepts a 1-based
    VCF ``variant_position`` and immediately converts via ``variant_position - 1``,
    so the centered bin should match upstream's 0-based behavior.
    """

    def test_width_one_centers_on_variant_position(self):
        # variant_position=3 (1-based) → 0-based index 2 → only bin 2 set.
        mask = create_center_mask(
            variant_position=3,
            interval_start=0,
            width=1,
            seq_length=5,
        )
        expected = torch.tensor([False, False, True, False, False])
        assert torch.equal(mask, expected)

    def test_width_three_spans_three_bins(self):
        # width=3 around 0-based center 2 → bins 1, 2, 3.
        mask = create_center_mask(
            variant_position=3,
            interval_start=0,
            width=3,
            seq_length=5,
        )
        expected = torch.tensor([False, True, True, True, False])
        assert torch.equal(mask, expected)

    def test_interval_start_offset(self):
        # interval_start shifts the relative position, so a variant at 1-based
        # position 13 with interval_start=10 maps to 0-based rel_position 2.
        mask = create_center_mask(
            variant_position=13,
            interval_start=10,
            width=1,
            seq_length=5,
        )
        expected = torch.tensor([False, False, True, False, False])
        assert torch.equal(mask, expected)

    def test_none_width_returns_all_true(self):
        mask = create_center_mask(
            variant_position=3,
            interval_start=0,
            width=None,
            seq_length=4,
        )
        assert torch.equal(mask, torch.ones(4, dtype=torch.bool))
