"""Gene-level variant scorers using gene masks.

This module provides scorers that aggregate predictions at the gene level,
using either exon-only masks or full gene body masks.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from ..aggregations import align_alternate
from ..types import Interval, OutputType, Variant, VariantScore
from .base import BaseVariantScorer

if TYPE_CHECKING:
    from ..annotations import GeneAnnotation


class GeneMaskMode(Enum):
    """Gene mask mode for scoring.

    Controls which genomic regions are included in the gene mask:
    - EXONS: Only exonic regions (coding + UTR). Best for expression-level
             analysis where intronic signal is less relevant.
    - BODY: Full gene body (exons + introns). Best for analyses where
            intronic regulation (e.g., enhancers, splicing) matters.
    """

    EXONS = 'exons'
    BODY = 'body'


# Supported output types for gene mask scorers
SUPPORTED_OUTPUT_TYPES = frozenset([
    OutputType.ATAC,
    OutputType.CAGE,
    OutputType.DNASE,
    OutputType.PROCAP,
    OutputType.RNA_SEQ,
    OutputType.SPLICE_SITES,
    OutputType.SPLICE_SITE_USAGE,
])

# Supported resolutions
SUPPORTED_RESOLUTIONS = frozenset([1, 128])


class GeneMaskLFCScorer(BaseVariantScorer):
    """Gene-level log fold change scorer using gene masks.

    Quantifies the impact on overall gene transcript abundance by calculating
    the log fold change between ALT and REF alleles using a gene mask.

    Requires gene annotations to create masks.

    Args:
        requested_output: The model output type to score
        mask_mode: Whether to use exon-only mask or full gene body mask.
            Default is EXONS for expression-level analysis.
        resolution: Output resolution to use (1 for 1bp, 128 for 128bp).
            Default is 128bp as gene bodies are typically >1kb.

    Example:
        >>> scorer = GeneMaskLFCScorer(requested_output=OutputType.RNA_SEQ)
        >>> scorer.is_signed
        True

        >>> # Use gene body mask instead of exons
        >>> scorer = GeneMaskLFCScorer(
        ...     requested_output=OutputType.RNA_SEQ,
        ...     mask_mode=GeneMaskMode.BODY,
        ... )
    """

    def __init__(
        self,
        requested_output: OutputType,
        mask_mode: GeneMaskMode = GeneMaskMode.EXONS,
        resolution: int = 1,
    ):
        if requested_output not in SUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                f"Unsupported output type: {requested_output}. "
                f"Supported: {sorted(o.value for o in SUPPORTED_OUTPUT_TYPES)}"
            )
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Resolution must be 1 or 128, got {resolution}"
            )
        self._requested_output = requested_output
        self._mask_mode = mask_mode
        self._resolution = resolution

    @property
    def requested_output(self) -> OutputType:
        return self._requested_output

    @property
    def mask_mode(self) -> GeneMaskMode:
        """Gene mask mode (EXONS or BODY)."""
        return self._mask_mode

    @property
    def resolution(self) -> int:
        """Output resolution (1 for 1bp, 128 for 128bp)."""
        return self._resolution

    @property
    def name(self) -> str:
        return (
            f"GeneMaskLFCScorer("
            f"output={self._requested_output.value}, "
            f"mode={self._mask_mode.value}, "
            f"res={self._resolution}bp)"
        )

    @property
    def is_signed(self) -> bool:
        return True

    def score(
        self,
        ref_outputs: dict[str, Any],
        alt_outputs: dict[str, Any],
        variant: Variant,
        interval: Interval,
        organism_index: int,
        gene_annotation: 'GeneAnnotation | None' = None,
        gene_ids: list[str] | None = None,
        **kwargs,
    ) -> list[VariantScore]:
        """Compute gene-level log fold change scores.

        Args:
            ref_outputs: Model outputs for reference sequence
            alt_outputs: Model outputs for alternate sequence
            variant: The variant being scored
            interval: Genomic interval of the input sequence
            organism_index: 0 for human, 1 for mouse
            gene_annotation: GeneAnnotation object for creating exon masks
            gene_ids: Optional list of gene IDs to score. If None, scores
                all genes overlapping the interval.

        Returns:
            List of VariantScore, one per gene
        """
        if gene_annotation is None:
            raise ValueError(
                "GeneMaskLFCScorer requires gene_annotation parameter. "
                "Create with: GeneAnnotation(gtf_path)"
            )

        # Get predictions at configured resolution
        ref_preds = self._get_predictions(ref_outputs, resolution=self._resolution)
        alt_preds = self._get_predictions(alt_outputs, resolution=self._resolution)

        # Handle splice outputs
        if self._requested_output == OutputType.SPLICE_SITES:
            if isinstance(ref_preds, dict) and 'probs' in ref_preds:
                ref_preds = ref_preds['probs']
                alt_preds = alt_preds['probs']

        # Ensure batch dimension
        if ref_preds.dim() == 2:
            ref_preds = ref_preds.unsqueeze(0)
            alt_preds = alt_preds.unsqueeze(0)

        B, S, T = ref_preds.shape

        # Apply indel alignment if needed
        if variant.is_indel:
            alt_preds_aligned = align_alternate(
                alt_preds.squeeze(0),
                variant.start,
                len(variant.reference_bases),
                len(variant.alternate_bases),
                interval.start,
            ).unsqueeze(0)
        else:
            alt_preds_aligned = alt_preds

        # Get genes to score
        if gene_ids is None:
            gene_ids = gene_annotation.get_genes_in_interval(interval)

        if not gene_ids:
            return []

        scores_list = []
        eps = 1e-3  # Match JAX

        for gene_id in gene_ids:
            # Get mask for this gene based on mask mode
            if self._mask_mode == GeneMaskMode.EXONS:
                gene_mask = gene_annotation.get_exon_mask(
                    gene_id=gene_id,
                    interval=interval,
                    resolution=self._resolution,
                    seq_length=S,
                    device=ref_preds.device,
                )
            else:  # BODY mode
                gene_mask = gene_annotation.get_gene_mask(
                    gene_id=gene_id,
                    interval=interval,
                    resolution=self._resolution,
                    seq_length=S,
                    device=ref_preds.device,
                )

            if gene_mask.sum() == 0:
                # No coverage in this interval for this gene
                continue

            # Apply mask and sum
            mask = gene_mask.unsqueeze(0).unsqueeze(-1)  # (1, S, 1)
            ref_masked = (ref_preds * mask).sum(dim=1)  # (B, T)
            alt_masked = (alt_preds_aligned * mask).sum(dim=1)  # (B, T)
            
            # Normalize by mask sum to get mean (matches JAX)
            gene_mask_sum = gene_mask.sum()
            ref_mean = ref_masked / gene_mask_sum
            alt_mean = alt_masked / gene_mask_sum

            # Log fold change: ln(alt) - ln(ref) using natural log
            lfc = torch.log(alt_mean + eps) - torch.log(ref_mean + eps)

            # Remove batch dimension if single sample
            if lfc.shape[0] == 1:
                lfc = lfc.squeeze(0)

            gene_info = gene_annotation.get_gene_info(gene_id)

            # Use gene coordinates as fallback for junction start/end if valid
            j_start = gene_info.get('start') if gene_info else None
            j_end = gene_info.get('end') if gene_info else None

            scores_list.append(VariantScore(
                variant=variant,
                interval=interval,
                scorer=self,
                scores=lfc,
                gene_id=gene_id,
                gene_name=gene_info.get('gene_name') if gene_info else None,
                gene_type=gene_info.get('gene_type') if gene_info else None,
                gene_strand=gene_info.get('strand') if gene_info else None,
                junction_start=j_start,
                junction_end=j_end,
            ))

        return scores_list


class GeneMaskActiveScorer(BaseVariantScorer):
    """Gene-level active allele scorer using gene masks.

    Captures the absolute activity level associated with one of the alleles,
    rather than the difference. Active allele gene scores are calculated by
    taking the maximum of the aggregated ALT and REF signals across the gene.

    This is non-directional (scores are always positive).

    Args:
        requested_output: The model output type to score
        mask_mode: Whether to use exon-only mask or full gene body mask.
            Default is EXONS for expression-level analysis.
        resolution: Output resolution to use (1 for 1bp, 128 for 128bp).
            Default is 128bp as gene bodies are typically >1kb.

    Example:
        >>> scorer = GeneMaskActiveScorer(requested_output=OutputType.RNA_SEQ)
        >>> scorer.is_signed
        False

        >>> # Use gene body mask instead of exons
        >>> scorer = GeneMaskActiveScorer(
        ...     requested_output=OutputType.RNA_SEQ,
        ...     mask_mode=GeneMaskMode.BODY,
        ... )
    """

    def __init__(
        self,
        requested_output: OutputType,
        mask_mode: GeneMaskMode = GeneMaskMode.EXONS,
        resolution: int = 1,
    ):
        if requested_output not in SUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                f"Unsupported output type: {requested_output}. "
                f"Supported: {sorted(o.value for o in SUPPORTED_OUTPUT_TYPES)}"
            )
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Resolution must be 1 or 128, got {resolution}"
            )
        self._requested_output = requested_output
        self._mask_mode = mask_mode
        self._resolution = resolution

    @property
    def requested_output(self) -> OutputType:
        return self._requested_output

    @property
    def mask_mode(self) -> GeneMaskMode:
        """Gene mask mode (EXONS or BODY)."""
        return self._mask_mode

    @property
    def resolution(self) -> int:
        """Output resolution (1 for 1bp, 128 for 128bp)."""
        return self._resolution

    @property
    def name(self) -> str:
        return (
            f"GeneMaskActiveScorer("
            f"output={self._requested_output.value}, "
            f"mode={self._mask_mode.value}, "
            f"res={self._resolution}bp)"
        )

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
        gene_annotation: 'GeneAnnotation | None' = None,
        gene_ids: list[str] | None = None,
        **kwargs,
    ) -> list[VariantScore]:
        """Compute gene-level active allele scores.

        Args:
            ref_outputs: Model outputs for reference sequence
            alt_outputs: Model outputs for alternate sequence
            variant: The variant being scored
            interval: Genomic interval of the input sequence
            organism_index: 0 for human, 1 for mouse
            gene_annotation: GeneAnnotation object for creating exon masks
            gene_ids: Optional list of gene IDs to score. If None, scores
                all genes overlapping the interval.

        Returns:
            List of VariantScore, one per gene
        """
        if gene_annotation is None:
            raise ValueError(
                "GeneMaskActiveScorer requires gene_annotation parameter. "
                "Create with: GeneAnnotation(gtf_path)"
            )

        # Get predictions at configured resolution
        ref_preds = self._get_predictions(ref_outputs, resolution=self._resolution)
        alt_preds = self._get_predictions(alt_outputs, resolution=self._resolution)

        # Handle splice outputs
        if self._requested_output == OutputType.SPLICE_SITES:
            if isinstance(ref_preds, dict) and 'probs' in ref_preds:
                ref_preds = ref_preds['probs']
                alt_preds = alt_preds['probs']

        # Ensure batch dimension
        if ref_preds.dim() == 2:
            ref_preds = ref_preds.unsqueeze(0)
            alt_preds = alt_preds.unsqueeze(0)

        B, S, T = ref_preds.shape

        # Apply indel alignment if needed
        if variant.is_indel:
            alt_preds_aligned = align_alternate(
                alt_preds.squeeze(0),
                variant.start,
                len(variant.reference_bases),
                len(variant.alternate_bases),
                interval.start,
            ).unsqueeze(0)
        else:
            alt_preds_aligned = alt_preds

        # Get genes to score
        if gene_ids is None:
            gene_ids = gene_annotation.get_genes_in_interval(interval)

        if not gene_ids:
            return []

        scores_list = []

        for gene_id in gene_ids:
            # Get mask for this gene based on mask mode
            if self._mask_mode == GeneMaskMode.EXONS:
                gene_mask = gene_annotation.get_exon_mask(
                    gene_id=gene_id,
                    interval=interval,
                    resolution=self._resolution,
                    seq_length=S,
                    device=ref_preds.device,
                )
            else:  # BODY mode
                gene_mask = gene_annotation.get_gene_mask(
                    gene_id=gene_id,
                    interval=interval,
                    resolution=self._resolution,
                    seq_length=S,
                    device=ref_preds.device,
                )

            gene_mask_sum = gene_mask.sum()
            if gene_mask_sum == 0:
                # No coverage in this interval for this gene
                continue

            # Apply mask and compute mean (matching JAX implementation)
            mask = gene_mask.unsqueeze(0).unsqueeze(-1)  # (1, S, 1)
            ref_mean = (ref_preds * mask).sum(dim=1) / gene_mask_sum  # (B, T)
            alt_mean = (alt_preds_aligned * mask).sum(dim=1) / gene_mask_sum  # (B, T)

            # Active score: max(mean(ref), mean(alt))
            active_scores = torch.maximum(ref_mean, alt_mean)

            # Remove batch dimension if single sample
            if active_scores.shape[0] == 1:
                active_scores = active_scores.squeeze(0)

            gene_info = gene_annotation.get_gene_info(gene_id)

            # Use gene coordinates as fallback for junction start/end if valid
            j_start = gene_info.get('start') if gene_info else None
            j_end = gene_info.get('end') if gene_info else None

            scores_list.append(VariantScore(
                variant=variant,
                interval=interval,
                scorer=self,
                scores=active_scores,
                gene_id=gene_id,
                gene_name=gene_info.get('gene_name') if gene_info else None,
                gene_type=gene_info.get('gene_type') if gene_info else None,
                gene_strand=gene_info.get('strand') if gene_info else None,
                junction_start=j_start,
                junction_end=j_end,
            ))

        return scores_list
