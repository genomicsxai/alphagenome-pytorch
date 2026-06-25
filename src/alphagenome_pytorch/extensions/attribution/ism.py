"""Saturation in-silico mutagenesis (ISM) attribution.

Lifted from ``scripts/ism_locus.py:_saturation_ism``, generalized to any
``HeadSelector``. Returns a JSON-friendly ``(W, 4, T)`` matrix in which the
reference-base column is NaN and all other cells hold the score delta vs. the
unmutated reference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from .gradient import strand_average
from .heads import HeadSelector, default_head_selector
from .types import BASES, AttributionResult
from .window import reduce_window


def _ism_pass(
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
    batch_size: int,
    autocast_dtype: torch.dtype | None,
) -> np.ndarray:
    """Single-direction saturation ISM. Returns ``(W, 4, T)`` with NaN at
    reference-base cells and at N positions.
    """
    target_lo_bp = target_slice.start * resolution
    target_hi_bp = target_slice.stop * resolution
    W = target_hi_bp - target_lo_bp
    T = len(track_indices)
    device = onehot.device

    values = np.full((W, 4, T), np.nan, dtype=np.float32)

    onehot_np = onehot[0].detach().float().cpu().numpy()  # (L, 4)
    is_ref = onehot_np.sum(axis=1) > 0.5
    ref_idx = onehot_np.argmax(axis=1)  # 0..3 (meaningless when not is_ref)

    # Reference scalar per track.
    with torch.no_grad():
        if autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                ref_pred = head_selector(
                    model, onehot, organism_index,
                    output_type=output_type, resolution=resolution,
                )
        else:
            ref_pred = head_selector(
                model, onehot, organism_index,
                output_type=output_type, resolution=resolution,
            )
    ref_window = ref_pred[:, target_slice, :][:, :, list(track_indices)]
    ref_scalar = reduce_window(ref_window, reduction).float().cpu().numpy()[0]  # (T,)

    # Build mutation plan: (pos_in_window, alt_base) for each non-ref non-N cell.
    plan: list[tuple[int, int]] = []
    for p in range(W):
        L_pos = target_lo_bp + p
        if not is_ref[L_pos]:
            continue  # N position: leave row as NaN.
        ref_b = int(ref_idx[L_pos])
        for alt_b in range(4):
            if alt_b == ref_b:
                continue
            plan.append((p, alt_b))

    if not plan:
        return values

    track_idx_list = list(track_indices)

    for bstart in range(0, len(plan), batch_size):
        chunk = plan[bstart:bstart + batch_size]
        B = len(chunk)
        batch = onehot.repeat(B, 1, 1).contiguous().float()
        for i, (p, alt_b) in enumerate(chunk):
            L_pos = target_lo_bp + p
            batch[i, L_pos, :] = 0.0
            batch[i, L_pos, alt_b] = 1.0
        batch_org = organism_index.expand(B).contiguous()

        with torch.no_grad():
            if autocast_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    pred = head_selector(
                        model, batch, batch_org,
                        output_type=output_type, resolution=resolution,
                    )
            else:
                pred = head_selector(
                    model, batch, batch_org,
                    output_type=output_type, resolution=resolution,
                )
        alt_window = pred[:, target_slice, :][:, :, track_idx_list]
        alt_scalar = reduce_window(alt_window, reduction).float().cpu().numpy()  # (B, T)

        for i, (p, alt_b) in enumerate(chunk):
            values[p, alt_b, :] = alt_scalar[i] - ref_scalar

    return values


def saturation_ism(
    model: nn.Module,
    *,
    onehot: torch.Tensor,
    organism_index: int,
    output_type: str,
    resolution: int,
    target_slice: slice,
    track_indices: Sequence[int],
    reduction: str = "sum",
    batch_size: int = 8,
    strand_averaged: bool = False,
    autocast_dtype: torch.dtype | None = None,
    head_selector: HeadSelector = default_head_selector,
    sequence: str = "",
    target_start: int = 0,
    target_end: int = 0,
) -> AttributionResult:
    """Saturation ISM. Returns ``(W, 4, T)`` deltas vs. reference."""
    if onehot.dim() != 3 or onehot.shape[0] != 1 or onehot.shape[2] != 4:
        raise ValueError(
            f"onehot must have shape (1, L, 4); got {tuple(onehot.shape)}."
        )
    if reduction not in ("sum", "mean", "max"):
        raise ValueError(f"Unknown reduction {reduction!r}.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    device = onehot.device
    organism_t = torch.tensor([int(organism_index)], dtype=torch.long, device=device)

    def run_pass(input_onehot: torch.Tensor, slice_: slice) -> np.ndarray:
        return _ism_pass(
            model, input_onehot, organism_t,
            head_selector=head_selector,
            output_type=output_type, resolution=resolution,
            target_slice=slice_, track_indices=track_indices,
            reduction=reduction, batch_size=batch_size,
            autocast_dtype=autocast_dtype,
        )

    fwd = run_pass(onehot, target_slice)
    if strand_averaged:
        values = strand_average(
            fwd, run_pass,
            onehot=onehot, target_slice=target_slice, resolution=resolution,
        )
    else:
        values = fwd

    return AttributionResult(
        method="saturation_ism",
        kind="base_matrix",
        bases=BASES,
        values=values,
        sequence=sequence,
        target_start=target_start,
        target_end=target_end,
        resolution=resolution,
        track_indices=tuple(int(i) for i in track_indices),
        reduction=reduction,
        raw_gradient=None,
        metadata={"strand_averaged": bool(strand_averaged)},
    )
