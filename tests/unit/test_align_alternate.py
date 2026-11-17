"""Tests for align_alternate, including 3D (batched) support added for B2.3."""

from __future__ import annotations

import pytest
import torch

from alphagenome_pytorch.variant_scoring.aggregations import align_alternate


@pytest.mark.unit
def test_align_alternate_snv_is_identity_2d():
    alt = torch.randn(16, 4)
    out = align_alternate(alt, variant_start=10, ref_length=1, alt_length=1, interval_start=0)
    assert out.shape == alt.shape
    assert torch.equal(out, alt)


@pytest.mark.unit
def test_align_alternate_snv_is_identity_3d():
    alt = torch.randn(2, 16, 4)
    out = align_alternate(alt, variant_start=10, ref_length=1, alt_length=1, interval_start=0)
    assert out.shape == alt.shape
    assert torch.equal(out, alt)


@pytest.mark.unit
def test_align_alternate_2d_and_3d_match_for_insertion():
    """3D path with B=1 must give the same answer as the 2D path."""
    torch.manual_seed(0)
    alt2d = torch.randn(16, 4)
    alt3d = alt2d.unsqueeze(0).clone()  # (1, 16, 4)

    kw = dict(variant_start=8, ref_length=1, alt_length=3, interval_start=0)
    out2d = align_alternate(alt2d, **kw)
    out3d = align_alternate(alt3d, **kw)

    assert out2d.shape == (16, 4)
    assert out3d.shape == (1, 16, 4)
    torch.testing.assert_close(out2d, out3d.squeeze(0))


@pytest.mark.unit
def test_align_alternate_2d_and_3d_match_for_deletion():
    torch.manual_seed(1)
    alt2d = torch.randn(16, 4)
    alt3d = alt2d.unsqueeze(0).clone()

    kw = dict(variant_start=5, ref_length=3, alt_length=1, interval_start=0)
    out2d = align_alternate(alt2d, **kw)
    out3d = align_alternate(alt3d, **kw)
    torch.testing.assert_close(out2d, out3d.squeeze(0))


@pytest.mark.unit
def test_align_alternate_insertion_pools_with_max_3d():
    """For an insertion of length 2, the inserted region collapses to max."""
    # Create a tensor where positions 8, 9, 10 have known patterns
    alt = torch.zeros(2, 16, 1)  # B=2, S=16, T=1
    # Make the variant region distinguishable
    alt[0, 8, 0] = 1.0
    alt[0, 9, 0] = 5.0  # max in this batch row
    alt[0, 10, 0] = 3.0
    alt[1, 8, 0] = 2.0
    alt[1, 9, 0] = 4.0
    alt[1, 10, 0] = 7.0  # max in this batch row

    # variant_start=8 (interval_start=0), ref_length=1, alt_length=3
    # variant_start_in_vector = 8 + min(1,3)-1 = 8
    # insertion pool spans [8, 11)
    out = align_alternate(alt, variant_start=8, ref_length=1, alt_length=3, interval_start=0)

    assert out.shape == (2, 16, 1)
    # At position 8 we expect the max of [8,9,10] for each batch row
    assert out[0, 8, 0].item() == pytest.approx(5.0)
    assert out[1, 8, 0].item() == pytest.approx(7.0)
    # Last 2 positions should be zero-padded
    assert torch.all(out[:, -2:, :] == 0.0)


@pytest.mark.unit
def test_align_alternate_deletion_inserts_zeros_3d():
    """For a deletion of length 2, two zero rows are inserted after the variant."""
    alt = torch.ones(1, 16, 3)
    # variant_start=5, ref_length=3, alt_length=1
    # variant_start_in_vector = 5 + min(3,1)-1 = 5
    # zeros inserted starting at index 6 (length 2)
    out = align_alternate(alt, variant_start=5, ref_length=3, alt_length=1, interval_start=0)
    assert out.shape == (1, 16, 3)
    # Positions 6, 7 should be zero
    assert torch.all(out[0, 6:8, :] == 0.0)
    # Position 5 should be unchanged (= 1.0)
    assert torch.all(out[0, 5, :] == 1.0)


@pytest.mark.unit
def test_align_alternate_rejects_invalid_shape():
    with pytest.raises(ValueError):
        align_alternate(torch.zeros(4), variant_start=0, ref_length=1, alt_length=1, interval_start=0)
    with pytest.raises(ValueError):
        align_alternate(
            torch.zeros(2, 1, 16, 4),
            variant_start=0, ref_length=1, alt_length=1, interval_start=0,
        )
