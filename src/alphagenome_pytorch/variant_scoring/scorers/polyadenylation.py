"""PolyadenylationScorer for paQTL scoring.

This scorer quantifies a variant's impact on RNA isoform production by
analyzing proximal vs distal polyadenylation site (PAS) usage.

The scoring method is analogous to the Borzoi paQTL approach, as described
in the AlphaGenome paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..aggregations import align_alternate
from ..types import Interval, OutputType, Variant, VariantScore
from .base import BaseVariantScorer

if TYPE_CHECKING:
    from ..annotations import GeneAnnotation, PolyAAnnotation

# Constants matching JAX implementation
MAX_GENES = 22
MAX_PAS = 136
_PAS_MASK_WIDTH = 400


@dataclass
class PolyadenylationMasks:
    """Container for polyadenylation scoring masks.

    Attributes:
        pas_mask: Boolean mask of shape (S, G, P) where S=sequence length,
            G=MAX_GENES, P=MAX_PAS. Each PAS has a 400bp window.
        gene_mask: Boolean mask of shape (G,) indicating valid genes.
    """
    pas_mask: torch.Tensor  # (S, G, P)
    gene_mask: torch.Tensor  # (G,)


def _aggregate_maximum_ratio_coverage_fc(
    ref: torch.Tensor,
    alt: torch.Tensor,
    gene_pas_mask: torch.Tensor,
) -> torch.Tensor:
    """Implements the Borzoi statistic for paQTL variant scoring.

    This computes the maximum absolute log2 fold change between proximal
    and distal polyadenylation site coverage ratios across all possible
    proximal/distal split points.

    Args:
        ref: Reference predictions of shape (S, T) where S=sequence, T=tracks
        alt: Alternate predictions of shape (S, T)
        gene_pas_mask: Boolean mask of shape (S, G, P)

    Returns:
        Scores of shape (G, T) - one score per gene and track
    """
    # Aggregate RNA signal over PAS windows
    # einsum: 'st,sgp->gpt' (sequence×tracks, sequence×genes×pas → genes×pas×tracks)
    # Use same dtype as input to support bfloat16
    mask_dtype = ref.dtype
    ref_aggregation = torch.einsum('st,sgp->gpt', ref, gene_pas_mask.to(mask_dtype))
    alt_aggregation = torch.einsum('st,sgp->gpt', alt, gene_pas_mask.to(mask_dtype))

    # Coverage ratio: alt / ref
    # Let division by zero produce inf, then nan_to_num converts to 0 - matches JAX
    covr_ratio = alt_aggregation / ref_aggregation
    covr_ratio = torch.nan_to_num(covr_ratio, posinf=0.0, neginf=0.0, nan=0.0)
    # Shape: (G, P, T) - genes × pas × tracks

    # Create mask for all possible proximal/distal splits
    # k_interval: [0, 1, 2, ..., MAX_PAS-1]
    k_interval = torch.arange(MAX_PAS, device=ref.device)
    # proximal_sites[k, a] = True if a <= k (PAS 'a' is proximal when split at 'k')
    proximal_sites = k_interval.unsqueeze(1) >= k_interval.unsqueeze(0)  # (K, K)

    # Get total number of PAS sites per gene
    # gene_pas_mask.max(dim=0) gives (G, P) - which PAS exist for each gene
    gene_pas_exists = gene_pas_mask.max(dim=0).values  # (G, P)
    k_total = gene_pas_exists.sum(dim=-1, keepdim=True).to(mask_dtype)  # (G, 1)

    # Cumulative PAS count per gene (how many PAS are included up to position k)
    k_gene = gene_pas_exists.to(mask_dtype).cumsum(dim=-1)  # (G, P)

    # k_scaling = (total - cumulative) / cumulative
    # This balances the ratio based on how many PAS are proximal vs distal
    # Division by zero at k=0 produces inf, handled by nan_to_num later - matches JAX
    k_scaling = ((k_total - k_gene) / k_gene).T.unsqueeze(-1)  # (P, G, 1)

    # Compute proximal and distal counts for each possible split
    # covr_ratio: (G, P, T), proximal_sites: (K, P)
    # einsum: 'gpt,kp->kgt'
    proximal_counts = torch.einsum('gpt,kp->kgt', covr_ratio, proximal_sites.to(mask_dtype))
    distal_counts = torch.einsum('gpt,kp->kgt', covr_ratio, (~proximal_sites).to(mask_dtype))

    # Score = |log2(k_scaling * proximal / distal)|
    # Let division by zero produce inf, then nan_to_num converts to 0 - matches JAX
    scores = torch.abs(torch.log2(k_scaling * proximal_counts / distal_counts))
    scores = torch.nan_to_num(scores, posinf=0.0, neginf=0.0, nan=0.0)
    # Shape: (K, G, T)

    # Take maximum over all possible split points
    return scores.max(dim=0).values  # (G, T)


class PolyadenylationScorer(BaseVariantScorer):
    """Variant scorer for polyadenylation QTLs (paQTLs).

    Captures a variant's impact on RNA isoform production by quantifying the
    maximum log2 fold change in expression between the set of proximal
    polyadenylation sites (PASs) vs. the set of distal PASs.

    This implements the Borzoi paQTL scoring method as described in the
    AlphaGenome paper:

    1. Creates 400bp upstream windows around each PAS site (strand-aware)
    2. Computes coverage ratio: alt_coverage / ref_coverage per PAS
    3. For each possible proximal/distal split point k:
       - Aggregates coverage ratios for proximal (first k) and distal PAS
       - Applies k-scaling: (total_PAS - k) / k
       - Computes |log2(k_scaling * proximal / distal)|
    4. Returns maximum score across all split points

    Requires both gene annotation and polyA annotation files. The polyA
    annotation links PAS sites to genes via spatial overlap.

    Note: This scorer is only supported for human (organism_index=0).

    Args:
        min_pas_count: Minimum number of PAS sites required per gene.
            Default is 2 (need at least proximal + distal).
        min_pas_coverage: Minimum fraction of gene's PAS sites that must
            be within the interval. Default is 0.8.

    Example:
        >>> scorer = PolyadenylationScorer()
        >>> scorer.is_signed
        False
        >>> scorer.requested_output
        <OutputType.RNA_SEQ: 'rna_seq'>
    """

    def __init__(
        self,
        min_pas_count: int = 2,
        min_pas_coverage: float = 0.8,
    ):
        self._min_pas_count = min_pas_count
        self._min_pas_coverage = min_pas_coverage

    @property
    def name(self) -> str:
        return "PolyadenylationScorer()"

    @property
    def requested_output(self) -> OutputType:
        return OutputType.RNA_SEQ

    @property
    def is_signed(self) -> bool:
        return False

    def _create_pas_masks(
        self,
        gene_annotation: 'GeneAnnotation',
        polya_annotation: 'PolyAAnnotation',
        interval: Interval,
        gene_ids: list[str],
        device: torch.device,
    ) -> tuple[PolyadenylationMasks, list[dict]]:
        """Create PAS masks and metadata for scoring.

        Args:
            gene_annotation: Gene annotation object
            polya_annotation: PolyA annotation object
            interval: Genomic interval
            gene_ids: List of gene IDs to process
            device: Torch device

        Returns:
            Tuple of (PolyadenylationMasks, list of gene metadata dicts)
        """
        # Initialize masks
        pas_mask = torch.zeros(
            (interval.width, MAX_GENES, MAX_PAS),
            dtype=torch.bool,
            device=device,
        )
        gene_padding_mask = torch.zeros(MAX_GENES, dtype=torch.bool, device=device)
        gene_metadata_list = []

        valid_gene_idx = 0
        for gene_id in gene_ids:
            if valid_gene_idx >= MAX_GENES:
                break

            gene_info = gene_annotation.get_gene_info(gene_id)
            if gene_info is None:
                continue

            # Get PAS positions for this gene in interval
            pas_positions = polya_annotation.get_pas_for_gene(gene_info, interval)

            # Check coverage requirement (JAX: at least 80% of gene's total PAS must be in interval)
            gene_strand = gene_info.get('strand', '+')
            
            # Use gene_id-based total count if available (linked parquet), else use spatial fallback
            if hasattr(polya_annotation, 'has_gene_id') and polya_annotation.has_gene_id:
                total_pas = polya_annotation.get_total_pas_count_for_gene(gene_id, gene_strand)
            else:
                # Fallback: query a large interval (spatial method)
                full_interval = Interval(interval.chromosome, 0, 1_000_000_000)
                all_pas_positions = polya_annotation.get_pas_for_gene(gene_info, full_interval)
                total_pas = len(all_pas_positions)
            
            if total_pas > 0:
                coverage_ratio = len(pas_positions) / total_pas
                if coverage_ratio < self._min_pas_coverage:
                    continue
            
            # Filter by minimum count
            if len(pas_positions) < self._min_pas_count:
                continue

            # Sort PAS positions
            pas_positions = sorted(pas_positions)

            # Create 400bp windows for each PAS
            for pas_idx, pas_pos in enumerate(pas_positions):
                if pas_idx >= MAX_PAS:
                    break

                # Strand-aware window: upstream for +, downstream for -
                if gene_strand == '+':
                    bin_end = min(pas_pos + 1, interval.width)
                    bin_start = max(bin_end - _PAS_MASK_WIDTH, 0)
                else:
                    bin_start = max(pas_pos, 0)
                    bin_end = min(bin_start + _PAS_MASK_WIDTH, interval.width)

                if bin_start < bin_end:
                    pas_mask[bin_start:bin_end, valid_gene_idx, pas_idx] = True

            # Mark gene as valid
            gene_padding_mask[valid_gene_idx] = True
            gene_metadata_list.append({
                'gene_id': gene_id,
                'gene_name': gene_info.get('gene_name'),
                'gene_type': gene_info.get('gene_type'),
                'gene_strand': gene_strand,
                'num_pas': len(pas_positions),
            })
            valid_gene_idx += 1

        masks = PolyadenylationMasks(
            pas_mask=pas_mask,
            gene_mask=gene_padding_mask,
        )
        return masks, gene_metadata_list

    def score(
        self,
        ref_outputs: dict[str, Any],
        alt_outputs: dict[str, Any],
        variant: Variant,
        interval: Interval,
        organism_index: int,
        gene_annotation: 'GeneAnnotation | None' = None,
        polya_annotation: 'PolyAAnnotation | None' = None,
        gene_ids: list[str] | None = None,
        **kwargs,
    ) -> list[VariantScore]:
        """Compute polyadenylation QTL scores.

        The paQTL score measures the maximum absolute log2 fold change between
        proximal and distal polyadenylation site usage for each gene.

        Args:
            ref_outputs: Model outputs for reference sequence
            alt_outputs: Model outputs for alternate sequence
            variant: The variant being scored
            interval: Genomic interval of the input sequence
            organism_index: 0 for human, 1 for mouse (human only supported)
            gene_annotation: GeneAnnotation object for gene info
            polya_annotation: PolyAAnnotation object for PAS positions
            gene_ids: Optional list of gene IDs to score

        Returns:
            List of VariantScore, one per gene with sufficient PAS coverage

        Raises:
            ValueError: If organism_index is not 0 (human)
        """
        if organism_index != 0:
            raise ValueError(
                "PolyadenylationScorer is only supported for human (organism_index=0)"
            )

        if polya_annotation is None:
            # Fall back to simple peak detection if no annotation provided
            return self._score_without_annotation(
                ref_outputs, alt_outputs, variant, interval
            )

        if gene_annotation is None:
            raise ValueError(
                "PolyadenylationScorer requires gene_annotation when "
                "polya_annotation is provided"
            )

        # Get RNA-seq predictions at 1bp resolution
        ref_preds = self._get_predictions(ref_outputs, resolution=1)
        alt_preds = self._get_predictions(alt_outputs, resolution=1)

        # Ensure no batch dimension for scoring
        if ref_preds.dim() == 3:
            ref_preds = ref_preds.squeeze(0)
            alt_preds = alt_preds.squeeze(0)

        S, T = ref_preds.shape

        # Apply indel alignment if needed
        if variant.is_indel:
            alt_preds = align_alternate(
                alt_preds,
                variant.start,
                len(variant.reference_bases),
                len(variant.alternate_bases),
                interval.start,
            )

        # Get genes to score
        if gene_ids is None:
            gene_ids = gene_annotation.get_genes_in_interval(interval)

        if not gene_ids:
            return []

        # Create PAS masks
        masks, gene_metadata_list = self._create_pas_masks(
            gene_annotation=gene_annotation,
            polya_annotation=polya_annotation,
            interval=interval,
            gene_ids=gene_ids,
            device=ref_preds.device,
        )

        if not gene_metadata_list:
            return []

        # Compute Borzoi scores
        scores = _aggregate_maximum_ratio_coverage_fc(
            ref_preds, alt_preds, masks.pas_mask
        )
        # Shape: (MAX_GENES, T)

        # Extract scores for valid genes
        scores_list = []
        for gene_idx, gene_meta in enumerate(gene_metadata_list):
            gene_scores = scores[gene_idx]  # (T,)

            scores_list.append(VariantScore(
                variant=variant,
                interval=interval,
                scorer=self,
                scores=gene_scores,
                gene_id=gene_meta['gene_id'],
                gene_name=gene_meta['gene_name'],
                gene_type=gene_meta['gene_type'],
                gene_strand=gene_meta['gene_strand'],
            ))

        return scores_list

    def _score_without_annotation(
        self,
        ref_outputs: dict[str, Any],
        alt_outputs: dict[str, Any],
        variant: Variant,
        interval: Interval,
    ) -> list[VariantScore]:
        """Fallback scoring using peak detection when no annotation available.

        This is a simplified fallback that detects RNA signal peaks and uses
        them as pseudo-PAS sites. Results will differ from the full Borzoi
        method.
        """
        # Get RNA-seq predictions at 1bp resolution
        ref_preds = self._get_predictions(ref_outputs, resolution=1)
        alt_preds = self._get_predictions(alt_outputs, resolution=1)

        # Ensure batch dimension
        if ref_preds.dim() == 2:
            ref_preds = ref_preds.unsqueeze(0)
            alt_preds = alt_preds.unsqueeze(0)

        B, S, T = ref_preds.shape
        eps = 1e-8

        # Detect PAS positions from RNA-seq signal peaks
        avg_ref = ref_preds.mean(dim=-1)  # (B, S)
        avg_alt = alt_preds.mean(dim=-1)  # (B, S)
        avg_signal = (avg_ref + avg_alt) / 2

        # Find local maxima
        padded = torch.nn.functional.pad(avg_signal, (1, 1), value=0)
        is_peak = (avg_signal > padded[:, :-2]) & (avg_signal > padded[:, 2:])
        is_peak = is_peak & (avg_signal > avg_signal.mean(dim=-1, keepdim=True))

        # Get peak positions for first batch element
        pas_positions = is_peak[0].nonzero(as_tuple=True)[0].tolist()

        if len(pas_positions) < 2:
            scores = torch.zeros(T, device=ref_preds.device, dtype=ref_preds.dtype)
            return [VariantScore(
                variant=variant,
                interval=interval,
                scorer=self,
                scores=scores,
            )]

        # Use simplified scoring for fallback
        pas_tensor = torch.tensor(pas_positions, device=ref_preds.device)
        ref_pas_signal = ref_preds[:, pas_tensor, :]
        alt_pas_signal = alt_preds[:, pas_tensor, :]

        num_pas = len(pas_positions)
        max_score = torch.zeros(B, T, device=ref_preds.device, dtype=ref_preds.dtype)

        for i in range(1, num_pas):
            ref_proximal = ref_pas_signal[:, :i, :].sum(dim=1)
            alt_proximal = alt_pas_signal[:, :i, :].sum(dim=1)
            ref_distal = ref_pas_signal[:, i:, :].sum(dim=1)
            alt_distal = alt_pas_signal[:, i:, :].sum(dim=1)

            # k-scaling
            k_scaling = (num_pas - i) / i

            ref_ratio = ref_proximal / (ref_distal + eps)
            alt_ratio = alt_proximal / (alt_distal + eps)

            score = torch.abs(torch.log2(
                k_scaling * alt_ratio / (ref_ratio + eps) + eps
            ))
            score = torch.nan_to_num(score, posinf=0.0, neginf=0.0, nan=0.0)
            max_score = torch.maximum(max_score, score)

        scores = max_score
        if scores.shape[0] == 1:
            scores = scores.squeeze(0)

        return [VariantScore(
            variant=variant,
            interval=interval,
            scorer=self,
            scores=scores,
        )]
