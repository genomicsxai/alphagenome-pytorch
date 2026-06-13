"""Unit tests for gradient x input attribution.

Uses a tiny differentiable fake model with known linear relationships so the
analytic answer is trivially verifiable.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from alphagenome_pytorch.extensions.attribution.gradient import gradient_x_input
from alphagenome_pytorch.extensions.attribution.heads import (
    HeadSelector,
    default_head_selector,
)
from alphagenome_pytorch.extensions.attribution.types import AttributionResult


# ---------------------------------------------------------------------------
# Fake model whose per-track output is a known linear function of the input.
# ---------------------------------------------------------------------------


class _LinearFakeModel(nn.Module):
    """A dummy model whose output is a per-track linear projection of the input.

    For track `t`, the output at resolution-bin `b` is:
        output[b, t] = weight[t] * input[b, :].sum()

    This means gradient w.r.t. input[b, c] is exactly `weight[t]` for every
    channel `c`, making the analytic gradient x input trivially:
        (grad * input)[b] = weight[t] * (input one-hot value at b)
    summed over channels = weight[t] * 1.0 = weight[t].
    """

    def __init__(self, n_tracks: int = 3, weights: list[float] | None = None):
        super().__init__()
        if weights is None:
            weights = list(range(1, n_tracks + 1))
        self._w = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    @property
    def n_tracks(self) -> int:
        return len(self._w)


def _fake_head_selector(
    model: nn.Module,
    onehot: torch.Tensor,
    organism_index: torch.Tensor,
    *,
    output_type: str,
    resolution: int,
) -> torch.Tensor:
    """HeadSelector for ``_LinearFakeModel``."""
    assert isinstance(model, _LinearFakeModel)
    # onehot: (B, L, 4)
    x = onehot.sum(dim=-1)  # (B, L) — effectively 1 at each position for a valid one-hot
    # (B, L, T)
    return x.unsqueeze(-1) * model._w.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Asymmetric model for strand-averaging tests
# ---------------------------------------------------------------------------


class _AsymmetricModel(nn.Module):
    """Model whose output is position-dependent so forward != RC."""

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, onehot, org):
        raise NotImplementedError


def _asymmetric_head(
    model: nn.Module,
    onehot: torch.Tensor,
    organism_index: torch.Tensor,
    *,
    output_type: str,
    resolution: int,
) -> torch.Tensor:
    # Output is a weighted sum with position-dependent weights.
    # (B, L, 4) → (B, L, 1) with weight = position index.
    B, L, _ = onehot.shape
    pos = torch.arange(L, dtype=onehot.dtype, device=onehot.device)
    weighted = onehot.sum(-1) * pos  # (B, L)
    return weighted.unsqueeze(-1)  # (B, L, 1) — single track


class _AlphaGenomeLikeAdapterModel(nn.Module):
    """Fake adapter-backed/fine-tuned model with the public AlphaGenome contract."""

    def __init__(self):
        super().__init__()
        self.adapter_weight = nn.Parameter(torch.tensor([2.0]))

    def forward(
        self,
        onehot,
        organism_index,
        *,
        heads=None,
        resolutions=None,
        channels_last=True,
        return_scaled_predictions=False,
    ):
        assert heads == ("custom_dnase",)
        assert resolutions == (1,)
        assert channels_last is True
        assert return_scaled_predictions is False
        assert organism_index.shape == (onehot.shape[0],)
        values = onehot.sum(dim=-1, keepdim=True) * self.adapter_weight
        return {"custom_dnase": {1: values}}


# ---------------------------------------------------------------------------
# Tests
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


class TestGradientXInput:
    """Core gradient x input tests."""

    def test_analytic_answer(self):
        """For a linear model the gradient projection equals the weight."""
        model = _LinearFakeModel(n_tracks=2, weights=[3.0, 5.0])
        seq = "ACGT" * 4  # 16 bp
        onehot = _make_onehot(seq)
        target_slice = slice(4, 12)  # middle 8 bp
        W = (target_slice.stop - target_slice.start)

        result = gradient_x_input(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="dnase",
            resolution=1,
            target_slice=target_slice,
            track_indices=[0, 1],
            reduction="sum",
            head_selector=_fake_head_selector,
        )

        assert isinstance(result, AttributionResult)
        assert result.method == "input_x_gradient"
        assert result.kind == "base_matrix"
        assert result.values.shape == (W, 4, 2)

        # For each position, only the reference-base cell should be filled.
        # The projected value = weight[t] * 1.0 * W_bins (sum reduction sums
        # W_bins bins, each contributing weight[t]).
        # Actually: gradient of sum over window w.r.t. input at position p
        # equals weight[t] regardless of the other positions.
        # So grad * input at position p, channel c:
        #   weight[t] * onehot[p, c]
        # Projection = sum over c → weight[t] * 1.0 = weight[t].
        for w in range(W):
            ref_base = onehot[0, target_slice.start + w].argmax().item()
            for b in range(4):
                if b == ref_base:
                    np.testing.assert_allclose(
                        result.values[w, b, 0], 3.0, atol=1e-5,
                        err_msg=f"Track 0 mismatch at w={w}, base={b}",
                    )
                    np.testing.assert_allclose(
                        result.values[w, b, 1], 5.0, atol=1e-5,
                        err_msg=f"Track 1 mismatch at w={w}, base={b}",
                    )
                else:
                    assert np.isnan(result.values[w, b, 0])
                    assert np.isnan(result.values[w, b, 1])

    def test_include_raw_gradient_shape(self):
        model = _LinearFakeModel(n_tracks=1, weights=[2.0])
        onehot = _make_onehot("ACGTACGT")
        target_slice = slice(2, 6)
        W = 4

        result = gradient_x_input(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=target_slice,
            track_indices=[0],
            reduction="sum",
            include_raw_gradient=True,
            head_selector=_fake_head_selector,
        )
        assert result.raw_gradient is not None
        assert result.raw_gradient.shape == (W, 4, 1)

    def test_default_selector_uses_alphagenome_forward_contract(self):
        """Adapter-backed/fine-tuned models should not need a special selector."""
        model = _AlphaGenomeLikeAdapterModel()
        onehot = _make_onehot("ACGTACGT")

        selected = default_head_selector(
            model,
            onehot,
            torch.zeros(1, dtype=torch.long),
            output_type="custom_dnase",
            resolution=1,
        )

        assert selected.shape == (1, 8, 1)
        np.testing.assert_allclose(
            selected.detach().numpy(),
            np.full((1, 8, 1), 2.0, dtype=np.float32),
        )

        result = gradient_x_input(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="custom_dnase",
            resolution=1,
            target_slice=slice(0, 8),
            track_indices=[0],
        )
        finite_values = result.values[np.isfinite(result.values)]
        np.testing.assert_allclose(
            finite_values,
            np.full_like(finite_values, 2.0),
        )

    def test_raw_gradient_not_returned_by_default(self):
        model = _LinearFakeModel(n_tracks=1)
        onehot = _make_onehot("ACGT")
        result = gradient_x_input(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=slice(0, 4),
            track_indices=[0],
            head_selector=_fake_head_selector,
        )
        assert result.raw_gradient is None

    def test_n_positions_are_nan(self):
        model = _LinearFakeModel(n_tracks=1, weights=[1.0])
        onehot = _make_onehot("ANNC")
        result = gradient_x_input(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=slice(0, 4),
            track_indices=[0],
            head_selector=_fake_head_selector,
        )
        # Positions 1 and 2 are N → all four base cells should be NaN.
        for w in [1, 2]:
            assert np.all(np.isnan(result.values[w, :, 0])), (
                f"N position w={w} should be all NaN"
            )

    def test_strand_averaged_differs_from_forward_only(self):
        """With an asymmetric model, strand-averaging must differ from fwd-only."""
        model = _AsymmetricModel()
        onehot = _make_onehot("ACGTACGT")
        target_slice = slice(2, 6)

        fwd = gradient_x_input(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=target_slice,
            track_indices=[0],
            reduction="sum",
            strand_averaged=False,
            head_selector=_asymmetric_head,
        )
        avg = gradient_x_input(
            model,
            onehot=onehot,
            organism_index=0,
            output_type="x",
            resolution=1,
            target_slice=target_slice,
            track_indices=[0],
            reduction="sum",
            strand_averaged=True,
            head_selector=_asymmetric_head,
        )
        # They should NOT be identical because the model is asymmetric.
        assert not np.allclose(
            np.nan_to_num(fwd.values), np.nan_to_num(avg.values)
        ), "Strand-averaged should differ from forward-only for an asymmetric model."

    def test_invalid_onehot_shape_raises(self):
        model = _LinearFakeModel(n_tracks=1)
        bad = torch.zeros(2, 8, 4)  # batch != 1
        with pytest.raises(ValueError, match="shape"):
            gradient_x_input(
                model,
                onehot=bad,
                organism_index=0,
                output_type="x",
                resolution=1,
                target_slice=slice(0, 4),
                track_indices=[0],
                head_selector=_fake_head_selector,
            )

    def test_invalid_reduction_raises(self):
        model = _LinearFakeModel(n_tracks=1)
        onehot = _make_onehot("ACGT")
        with pytest.raises(ValueError, match="reduction"):
            gradient_x_input(
                model,
                onehot=onehot,
                organism_index=0,
                output_type="x",
                resolution=1,
                target_slice=slice(0, 4),
                track_indices=[0],
                reduction="max",
                head_selector=_fake_head_selector,
            )
