"""Input-gradient nucleotide attribution.

Lifted (and generalized) from ``scripts/ism_locus.py:_gradient_attribution``.
Always runs in fp32: the per-track loss surface is small, gradient stability
matters, and the original script's comment explicitly notes "gradient
deliberately uses fp32".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from .heads import HeadSelector, default_head_selector
from .types import BASES, AttributionResult
from .window import reduce_window


# Order: A, C, G, T -> complement T, G, C, A -> indices [3, 2, 1, 0].
_RC_BASE_PERM = (3, 2, 1, 0)


def _reverse_complement_onehot(onehot: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a one-hot tensor of shape ``(B, L, 4)``.

    Reverses the position axis and swaps base channels A<->T, C<->G.
    """
    return onehot.flip(dims=(1,))[..., list(_RC_BASE_PERM)]


def _align_rc_to_forward(values: np.ndarray) -> np.ndarray:
    """Flip RC-frame ``(W, 4, T)`` attribution back to forward coordinates.

    Reverses the position axis and inverts the base-channel permutation
    (A<->T, C<->G).
    """
    return values[::-1, list(_RC_BASE_PERM), :].copy()


def _gradient_pass(
    model: nn.Module,
    onehot: torch.Tensor,
    organism_index: torch.Tensor,
    *,
    head_selector: HeadSelector,
    output_type: str,
    resolution: int,
    target_slice: slice,
    track_indices: Sequence[int],
    reduction: str,
) -> np.ndarray:
    """Single-direction (no RC) gradient pass.

    Returns ``(W, 4, T)`` raw input gradient ``dL/dx`` evaluated over the
    target window, where W = ``target_slice.stop - target_slice.start`` in
    *input* positions.
    """
    L = onehot.shape[1]
    T = len(track_indices)
    target_lo_bp = target_slice.start * resolution
    target_hi_bp = target_slice.stop * resolution
    W = target_hi_bp - target_lo_bp

    raw_grad = np.zeros((W, 4, T), dtype=np.float32)

    for t_i, track_idx in enumerate(track_indices):
        x = onehot.clone().detach().float().requires_grad_(True)
        pred = head_selector(
            model, x, organism_index,
            output_type=output_type, resolution=resolution,
        )  # (1, T_bins, n_tracks)
        window_pred = pred[:, target_slice, track_idx]  # (1, W_bins)
        scalar = reduce_window(window_pred.unsqueeze(-1), reduction).squeeze()
        if scalar.dim() != 0:
            raise RuntimeError(
                f"Reduction returned non-scalar tensor of shape {tuple(scalar.shape)}; "
                "gradient attribution expects a single scalar per track."
            )
        scalar.backward()
        if x.grad is None:
            raise RuntimeError(
                "Backward did not populate input gradients. The model may have "
                "detached the input or run under torch.no_grad()."
            )
        g = x.grad.detach().float().cpu().numpy()[0]  # (L, 4)
        raw_grad[:, :, t_i] = g[target_lo_bp:target_hi_bp]
        model.zero_grad(set_to_none=True)

    _ = L  # silence unused-warning in tooling that flags it; L is implicit above.
    return raw_grad


def gradient_x_input(
    model: nn.Module,
    *,
    onehot: torch.Tensor,
    organism_index: int,
    output_type: str,
    resolution: int,
    target_slice: slice,
    track_indices: Sequence[int],
    reduction: str = "sum",
    include_raw_gradient: bool = False,
    strand_averaged: bool = False,
    head_selector: HeadSelector = default_head_selector,
    sequence: str = "",
    target_start: int = 0,
    target_end: int = 0,
) -> AttributionResult:
    """Gradient x input attribution.

    Args:
        model: The model. Must be on the same device as ``onehot``.
        onehot: Reference one-hot tensor of shape ``(1, L, 4)``.
        organism_index: 0 (human) or 1 (mouse).
        output_type: Head name (e.g. ``"dnase"``).
        resolution: Output resolution in bp (1 or 128).
        target_slice: Resolution-bin slice covering the target window.
        track_indices: Tracks to attribute.
        reduction: Window reduction passed to :func:`reduce_window`.
        include_raw_gradient: If True, also include ``raw_gradient`` of shape
            ``(W, 4, T)`` on the result.
        strand_averaged: If True, run a second forward+backward on the
            reverse-complement input and average the projected (and raw)
            attributions. Doubles compute. Recommended only for unstranded heads.
        head_selector: Strategy for forward + head extraction. Defaults to the
            AlphaGenome-compatible forward path used by base, adapter-backed,
            and fine-tuned models.
        sequence: Optional reference sequence for the result; passed through to
            the dataclass for downstream serialization.
        target_start, target_end: Absolute genomic coordinates of the target
            window; passed through.

    Returns:
        AttributionResult with ``kind == "base_matrix"``. ``values[w, b, t]``
        is the projected attribution at position ``w`` for the reference base;
        all non-reference base cells are NaN. If ``include_raw_gradient``, the
        full ``(W, 4, T)`` gradient (with all four base columns filled) is
        attached as ``raw_gradient``.
    """
    if onehot.dim() != 3 or onehot.shape[0] != 1 or onehot.shape[2] != 4:
        raise ValueError(
            f"onehot must have shape (1, L, 4); got {tuple(onehot.shape)}."
        )
    if reduction not in ("sum", "mean", "peak"):
        raise ValueError(f"Unknown reduction {reduction!r}.")

    L = onehot.shape[1]
    device = onehot.device
    organism_t = torch.tensor([int(organism_index)], dtype=torch.long, device=device)

    raw_fwd = _gradient_pass(
        model, onehot.float(), organism_t,
        head_selector=head_selector,
        output_type=output_type, resolution=resolution,
        target_slice=target_slice, track_indices=track_indices,
        reduction=reduction,
    )

    if strand_averaged:
        rc_onehot = _reverse_complement_onehot(onehot.float())
        # Flip target window into RC coordinates (in resolution bins).
        L_bins = L // resolution
        rc_target_slice = slice(L_bins - target_slice.stop, L_bins - target_slice.start)
        raw_rc = _gradient_pass(
            model, rc_onehot, organism_t,
            head_selector=head_selector,
            output_type=output_type, resolution=resolution,
            target_slice=rc_target_slice, track_indices=track_indices,
            reduction=reduction,
        )
        raw_grad = 0.5 * (raw_fwd + _align_rc_to_forward(raw_rc))
    else:
        raw_grad = raw_fwd

    # Project onto reference base: (raw_grad * input).sum(base) gives a per-position
    # scalar per track. To stay shape-compatible with the (W, 4, T) "base_matrix"
    # convention used by ISM, we encode the projection into the reference-base cell
    # and leave the others as NaN.
    onehot_window = onehot[0, target_slice.start * resolution:target_slice.stop * resolution].cpu().numpy()  # (W, 4)
    W, _, T = raw_grad.shape
    values = np.full((W, 4, T), np.nan, dtype=np.float32)
    ref_base_idx = onehot_window.argmax(axis=1)  # (W,)
    has_ref = onehot_window.sum(axis=1) > 0.5  # False at N positions
    proj = (raw_grad * onehot_window[:, :, None]).sum(axis=1)  # (W, T)
    for w in range(W):
        if has_ref[w]:
            values[w, ref_base_idx[w], :] = proj[w]

    return AttributionResult(
        method="input_x_gradient",
        kind="base_matrix",
        bases=BASES,
        values=values,
        sequence=sequence,
        target_start=target_start,
        target_end=target_end,
        resolution=resolution,
        track_indices=tuple(int(i) for i in track_indices),
        reduction=reduction,
        raw_gradient=raw_grad if include_raw_gradient else None,
        metadata={"strand_averaged": bool(strand_averaged)},
    )
