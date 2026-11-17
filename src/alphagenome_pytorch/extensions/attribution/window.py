"""Target-window helpers for attribution methods.

Lifted from ``scripts/ism_locus.py`` so the attribution library does not depend
on the script.
"""

from __future__ import annotations

import torch


_REDUCTIONS = ("sum", "mean", "max")


def reduce_window(window_pred: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce ``(B, W_bins, n_tracks)`` -> ``(B, n_tracks)``.

    ``max`` returns the per-track maximum across the window bins.
    It can be used e.g. for gradient saliency. 
    """
    if reduction == "sum":
        return window_pred.sum(dim=1)
    if reduction == "mean":
        return window_pred.mean(dim=1)
    if reduction == "max":
        return window_pred.amax(dim=1)
    raise ValueError(
        f"Unknown reduction {reduction!r}. Expected one of {_REDUCTIONS}."
    )


def target_slice_for_resolution(
    interval_start: int,
    target_start: int,
    target_end: int,
    resolution: int,
) -> slice:
    """Resolution-bin slice covering ``[target_start, target_end)``.

    Bin alignment is the caller's responsibility — pass already-aligned offsets.
    """
    lo = (target_start - interval_start) // resolution
    hi = (target_end - interval_start) // resolution
    return slice(lo, hi)
