"""Head selection strategy for attribution.

Attribution methods need a thin abstraction over "run the model and give me
``(B, T_bins, n_tracks)`` for the requested head + resolution." Both the base
pretrained AlphaGenome and a fine-tuned variant share this signature, so a
single default selector works for both — it just calls ``model.forward`` with
``heads=`` and ``resolutions=`` filters and indexes the result.

Callers can pass a custom ``HeadSelector`` for non-standard wrapping.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class HeadSelector(Protocol):
    """Strategy: forward pass + extract per-resolution prediction tensor."""

    def __call__(
        self,
        model: nn.Module,
        onehot: torch.Tensor,
        organism_index: torch.Tensor,
        *,
        output_type: str,
        resolution: int,
    ) -> torch.Tensor:  # (B, T_bins, n_tracks), channels_last
        ...


def default_head_selector(
    model: nn.Module,
    onehot: torch.Tensor,
    organism_index: torch.Tensor,
    *,
    output_type: str,
    resolution: int,
) -> torch.Tensor:
    """Run AlphaGenome.forward filtered to one head and resolution.

    Returns predictions in experimental space (``return_scaled_predictions=False``)
    in NLC layout, matching what ``GenomeTracksHead`` produces by default.
    """
    outputs = model(
        onehot,
        organism_index,
        heads=(output_type,),
        resolutions=(resolution,),
        channels_last=True,
        return_scaled_predictions=False,
    )
    if output_type not in outputs:
        raise KeyError(
            f"Model did not produce head {output_type!r}. "
            f"Available heads: {sorted(outputs.keys())}."
        )
    head_outputs = outputs[output_type]
    if resolution not in head_outputs:
        raise KeyError(
            f"Head {output_type!r} did not produce resolution {resolution}. "
            f"Available: {sorted(head_outputs.keys())}."
        )
    return head_outputs[resolution]
