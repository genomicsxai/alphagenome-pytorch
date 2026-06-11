"""Shared dataclasses for attribution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


BASES: tuple[str, ...] = ("A", "C", "G", "T")


@dataclass
class AttributionResult:
    """Output of an attribution method.

    Attributes:
        method: Method name from the registry (e.g. ``"input_x_gradient"``).
        kind: Score kind. ``"base_matrix"`` means ``values`` has shape
            ``(W, 4, T)``. Future methods may emit other kinds.
        bases: Base ordering for the second axis when ``kind == "base_matrix"``.
        values: ``(W, 4, T)`` numpy array. For gradient methods, only the
            reference-base column is filled (others are NaN). For ISM, only
            the mutated cells are filled (the reference-base column is NaN).
        sequence: One-letter DNA sequence covering the whole input interval.
        target_start: Absolute genomic coordinate of the first reported base.
        target_end: Absolute genomic coordinate just past the last reported base.
        resolution: Resolution of the head used to compute the score.
        track_indices: Indices selected from the head's track axis.
        reduction: Reduction applied across the target window.
        raw_gradient: Optional ``(W, 4, T)`` raw input gradient. Only set when
            the caller requested it for a gradient-based method.
    """

    method: str
    kind: str
    bases: tuple[str, ...]
    values: np.ndarray
    sequence: str
    target_start: int
    target_end: int
    resolution: int
    track_indices: tuple[int, ...]
    reduction: str
    raw_gradient: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
