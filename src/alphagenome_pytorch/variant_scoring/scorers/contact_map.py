"""ContactMapScorer for 3D chromatin contact disruption scoring.

This module implements the Orca method (Zhou et al. 2022) for scoring
variant effects on 3D chromatin contacts.
"""

from __future__ import annotations

from typing import Any

import torch

from ..types import Interval, OutputType, Variant, VariantScore
from .base import BaseVariantScorer

# 1MB window in base pairs
WINDOW_SIZE_BP = 1_000_000
# Resolution of contact maps (128bp per bin)
RESOLUTION = 128


class ContactMapScorer(BaseVariantScorer):
    """Variant scorer for 3D chromatin contact disruption.

    Implements the Orca scoring method (Zhou et al. 2022) for quantifying
    local contact disruption between ALT and REF alleles.

    From the AlphaGenome paper: "For variants affecting 3D chromatin contacts,
    a method similar to that used by Orca is employed for SNVs. This calculates
    the mean absolute difference between REF and ALT contact map predictions
    for all interactions involving the single genomic bin containing the variant
    and other bins within a defined local window (e.g., 1 Mb)."

    The scoring algorithm (matches JAX reference implementation):
    1. Identify the genomic bin (at 128bp resolution) containing the variant
    2. Compute absolute difference |ALT - REF| for the entire contact map
    3. Average over ALL rows to get mean disruption for each column
    4. Select the variant bin column to get per-track scores

    This measures the average disruption of contacts TO/FROM the variant position
    across all other positions in the contact map.

    This scorer is always non-directional (scores are positive).

    Reference:
        Zhou et al. 2022: https://doi.org/10.1038/s41588-022-01065-4

    Example:
        >>> scorer = ContactMapScorer()
        >>> scorer.is_signed
        False
        >>> scorer.requested_output
        <OutputType.CONTACT_MAPS: 'pair_activations'>
    """

    @property
    def name(self) -> str:
        return "ContactMapScorer()"

    @property
    def requested_output(self) -> OutputType:
        return OutputType.CONTACT_MAPS

    @property
    def is_signed(self) -> bool:
        return False

    def score(
        self,
        ref_outputs: dict[str, Any],
        alt_outputs: dict[str, Any],
        variant: Variant,
        interval: Interval,
        organism_index: int,
        **kwargs,
    ) -> VariantScore:
        """Compute contact map disruption score.

        Args:
            ref_outputs: Model outputs for reference sequence
            alt_outputs: Model outputs for alternate sequence
            variant: The variant being scored
            interval: Genomic interval of the input sequence
            organism_index: 0 for human, 1 for mouse

        Returns:
            VariantScore with per-track contact disruption scores
        """
        # Get contact map predictions
        # Shape: (B, S, S, T) where S = sequence bins, T = tracks
        ref_contacts = self._get_predictions(ref_outputs)
        alt_contacts = self._get_predictions(alt_outputs)

        # Ensure batch dimension
        if ref_contacts.dim() == 3:
            ref_contacts = ref_contacts.unsqueeze(0)
            alt_contacts = alt_contacts.unsqueeze(0)

        B, S, _, T = ref_contacts.shape

        # Find variant bin (128bp resolution)
        variant_pos_in_seq = variant.start - interval.start  # 0-based position
        variant_bin = variant_pos_in_seq // RESOLUTION

        # Clamp to valid range
        variant_bin = max(0, min(variant_bin, S - 1))

        # Experiment: LFC
        # log2(alt+1) - log2(ref+1)
        # Shape: (B, S, S, T)
        diff = torch.log2(alt_contacts + 1) - torch.log2(ref_contacts + 1)
        
        # Mean absolute LFC
        abs_diff = torch.abs(diff)

        # JAX order: average over ALL rows first, then select variant bin
        # Shape: (B, S, T)
        avg_over_rows = abs_diff.mean(dim=1)

        # Select the variant bin column (matches JAX: abs_diff.mean(axis=0)[variant_bin])
        # Shape: (B, T)
        scores = avg_over_rows[:, variant_bin, :]

        # Remove batch dimension if single sample
        if scores.shape[0] == 1:
            scores = scores.squeeze(0)  # (T,)

        return VariantScore(
            variant=variant,
            interval=interval,
            scorer=self,
            scores=scores,
        )
