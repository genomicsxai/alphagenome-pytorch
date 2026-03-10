"""Splicing variant scorers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..aggregations import align_alternate, create_center_mask
from ..types import Interval, OutputType, Variant, VariantScore
from .base import BaseVariantScorer
from ...utils.splicing import unstack_junction_predictions

if TYPE_CHECKING:
    from ..annotations import GeneAnnotation

# Supported output types for splicing scorer
SUPPORTED_SPLICING_OUTPUT_TYPES = frozenset([
    OutputType.SPLICE_SITES,
    OutputType.SPLICE_SITE_USAGE,
])

# Supported widths for splicing scorer
SUPPORTED_SPLICING_WIDTHS = frozenset([None, 101, 1_001, 10_001])

# Maximum splice sites to consider (matches JAX _MAX_SPLICE_SITES)
_MAX_SPLICE_SITES = 256


class GeneMaskSplicingScorer(BaseVariantScorer):
    """Variant scorer for splicing impact using gene masks.

    Quantifies changes in splice site assignment probabilities or splice site
    usage between ALT and REF alleles using a gene exon mask.

    Args:
        requested_output: OutputType.SPLICE_SITES or OutputType.SPLICE_SITE_USAGE
        width: Width of spatial mask around variant. If None, uses gene mask.
            Supported: None, 101, 1001, 10001

    Example:
        >>> scorer = GeneMaskSplicingScorer(
        ...     requested_output=OutputType.SPLICE_SITES,
        ...     width=None,
        ... )
        >>> scorer.is_signed
        False
    """

    def __init__(self, requested_output: OutputType, width: int | None):
        if requested_output not in SUPPORTED_SPLICING_OUTPUT_TYPES:
            raise ValueError(
                f"Unsupported output type: {requested_output}. "
                f"Supported: {sorted(o.value for o in SUPPORTED_SPLICING_OUTPUT_TYPES)}"
            )
        if width not in SUPPORTED_SPLICING_WIDTHS:
            raise ValueError(
                f"Unsupported width: {width}. "
                f"Supported: {sorted(w for w in SUPPORTED_SPLICING_WIDTHS if w is not None)}"
            )
        self._requested_output = requested_output
        self._width = width

    @property
    def requested_output(self) -> OutputType:
        return self._requested_output

    @property
    def width(self) -> int | None:
        return self._width

    @property
    def name(self) -> str:
        width_str = str(self._width) if self._width else 'gene_mask'
        return (
            f"GeneMaskSplicingScorer("
            f"output={self._requested_output.value}, "
            f"width={width_str})"
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
        """Compute splicing impact scores.

        Args:
            ref_outputs: Model outputs for reference sequence
            alt_outputs: Model outputs for alternate sequence
            variant: The variant being scored
            interval: Genomic interval of the input sequence
            organism_index: 0 for human, 1 for mouse
            gene_annotation: GeneAnnotation object (required if width is None)
            gene_ids: Optional list of gene IDs to score

        Returns:
            List of VariantScore (one per gene if using gene mask, or single if using width)
        """
        # Get predictions
        # Note: Splicing heads return dicts with 'logits'/'probs'/'predictions' keys,
        # but _get_predictions assumes dicts are int->tensor (resolution map) and defaults to first key.
        # So we extract manually here.

        output_key = self.requested_output.value
        ref_out = ref_outputs[output_key]
        alt_out = alt_outputs[output_key]

        if self._requested_output == OutputType.SPLICE_SITES:
            # SpliceSitesClassificationHead returns {'logits': ..., 'probs': ...}
            if isinstance(ref_out, dict) and 'probs' in ref_out:
                ref_preds = ref_out['probs']
                alt_preds = alt_out['probs']
            else:
                # Fallback if just tensor
                ref_preds = ref_out
                alt_preds = alt_out

        elif self._requested_output == OutputType.SPLICE_SITE_USAGE:
            # SpliceSitesUsageHead returns {'logits': ..., 'predictions': ...}
            if isinstance(ref_out, dict) and 'predictions' in ref_out:
                ref_preds = ref_out['predictions']
                alt_preds = alt_out['predictions']
            else:
                ref_preds = ref_out
                alt_preds = alt_out
        else:
            # Fallback for unexpected types
            ref_preds = self._get_predictions(ref_outputs)
            alt_preds = self._get_predictions(alt_outputs)

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

        if self._width is not None:
            # Use center mask around variant
            mask = create_center_mask(
                variant_position=variant.position,
                interval_start=interval.start,
                width=self._width,
                seq_length=S,
                resolution=1,  # Splicing is at 1bp resolution
                device=ref_preds.device,
            )
            mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, S, 1)

            # Max absolute difference within mask (matches JAX)
            diff = torch.abs(alt_preds_aligned - ref_preds) * mask
            scores = diff.max(dim=1)[0]  # [B, T]

            if scores.shape[0] == 1:
                scores = scores.squeeze(0)

            return [VariantScore(
                variant=variant,
                interval=interval,
                scorer=self,
                scores=scores,
            )]
        else:
            # Use gene mask
            if gene_annotation is None:
                raise ValueError(
                    "GeneMaskSplicingScorer with width=None requires gene_annotation"
                )

            if gene_ids is None:
                gene_ids = gene_annotation.get_genes_in_interval(interval)

            if not gene_ids:
                return []

            scores_list = []

            for gene_id in gene_ids:
                exon_mask = gene_annotation.get_exon_mask(
                    gene_id=gene_id,
                    interval=interval,
                    resolution=1,
                    seq_length=S,
                    device=ref_preds.device,
                )

                if exon_mask.sum() == 0:
                    continue

                mask = exon_mask.unsqueeze(0).unsqueeze(-1)  # (1, S, 1)

                # Max absolute difference within gene mask (matches JAX)
                diff = torch.abs(alt_preds_aligned - ref_preds) * mask
                gene_scores = diff.max(dim=1)[0]  # [B, T]

                if gene_scores.shape[0] == 1:
                    gene_scores = gene_scores.squeeze(0)

                gene_info = gene_annotation.get_gene_info(gene_id)

                # Use gene coordinates as fallback for junction start/end if valid
                j_start = gene_info.get('start') if gene_info else None
                j_end = gene_info.get('end') if gene_info else None

                scores_list.append(VariantScore(
                    variant=variant,
                    interval=interval,
                    scorer=self,
                    scores=gene_scores,
                    gene_id=gene_id,
                    gene_name=gene_info.get('gene_name') if gene_info else None,
                    gene_type=gene_info.get('gene_type') if gene_info else None,
                    gene_strand=gene_info.get('strand') if gene_info else None,
                    junction_start=j_start,
                    junction_end=j_end,
                ))

            return scores_list


class SpliceJunctionScorer(BaseVariantScorer):
    """Variant scorer for splice junction impact.

    Scores the impact of a variant on splice junction predictions.

    For each gene overlapping the variant, this scorer:
    1. Computes log fold change for all junctions within the gene body
    2. Finds the junction with maximum absolute change per tissue
    3. Returns one score per gene with the max junction's coordinates

    This matches the JAX SpliceJunctionVariantScorer behavior.

    Example:
        >>> scorer = SpliceJunctionScorer()
        >>> scorer.is_signed
        False
    """

    def __init__(self, filter_protein_coding: bool = True):
        """Initialize SpliceJunctionScorer.

        Args:
            filter_protein_coding: If True, only score protein-coding genes.
                Default True to match JAX behavior.
        """
        self._filter_protein_coding = filter_protein_coding

    @property
    def name(self) -> str:
        return "SpliceJunctionScorer()"

    @property
    def requested_output(self) -> OutputType:
        return OutputType.SPLICE_JUNCTIONS

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
        """Compute splice junction impact score.

        For each gene, returns the maximum absolute log fold change across all
        junctions within the gene body, along with the coordinates of the
        max-scoring junction.

        Args:
            ref_outputs: Model outputs for reference sequence
            alt_outputs: Model outputs for alternate sequence
            variant: The variant being scored
            interval: Genomic interval of the input sequence
            organism_index: 0 for human, 1 for mouse
            gene_annotation: GeneAnnotation object for gene body filtering.
                If None, returns all junctions without gene grouping.
            gene_ids: Optional list of gene IDs to score

        Returns:
            List of VariantScore, one per gene with the max junction's scores
        """
        # Get junction predictions
        # These come as a dict with 'pred_counts', 'splice_site_positions'
        output_key = self.requested_output.value
        ref_pred_counts = ref_outputs[output_key]['pred_counts']
        ref_positions = ref_outputs[output_key]['splice_site_positions']

        alt_pred_counts = alt_outputs[output_key]['pred_counts']
        alt_positions = alt_outputs[output_key]['splice_site_positions']

        # Limit to MAX_SPLICE_SITES (matches JAX)
        if ref_pred_counts.shape[1] > _MAX_SPLICE_SITES:
            ref_pred_counts = ref_pred_counts[:, :_MAX_SPLICE_SITES, :_MAX_SPLICE_SITES]
            alt_pred_counts = alt_pred_counts[:, :_MAX_SPLICE_SITES, :_MAX_SPLICE_SITES]
            ref_positions = ref_positions[:, :, :_MAX_SPLICE_SITES]
            alt_positions = alt_positions[:, :, :_MAX_SPLICE_SITES]

        # Unstack to get list of junctions
        # scores: [B, N, T], starts: [B, N], ends: [B, N]
        ref_scores, ref_starts, ref_ends, ref_strands, ref_mask = unstack_junction_predictions(
            ref_pred_counts, ref_positions, interval_start=interval.start
        )
        alt_scores, alt_starts, alt_ends, alt_strands, alt_mask = unstack_junction_predictions(
            alt_pred_counts, alt_positions, interval_start=interval.start
        )

        # Filter to valid junctions in both
        valid_mask = ref_mask & alt_mask  # [B, N]

        if not valid_mask.any():
            return []

        # Log transform with epsilon for stability (matches JAX: log(x + 1e-7))
        eps = 1e-7
        ref_log = torch.log(ref_scores + eps)
        alt_log = torch.log(alt_scores + eps)

        # Compute absolute log fold change
        delta = torch.abs(alt_log - ref_log)  # [B, N, T]

        # Zero out invalid junctions
        delta = delta * valid_mask.unsqueeze(-1).float()

        # If no gene annotation, return all valid junctions
        if gene_annotation is None:
            return self._score_all_junctions(
                delta, ref_starts, ref_ends, ref_strands, valid_mask,
                variant, interval
            )

        # Get genes overlapping the variant
        # JAX uses VARIANT_OVERLAPPING query type and filters to protein_coding
        if gene_ids is None:
            gene_ids = gene_annotation.get_genes_in_interval(interval)

        if not gene_ids:
            return []

        # Filter to protein-coding if requested
        if self._filter_protein_coding:
            filtered_ids = []
            for gid in gene_ids:
                info = gene_annotation.get_gene_info(gid)
                if info and info.get('gene_type') == 'protein_coding':
                    filtered_ids.append(gid)
            gene_ids = filtered_ids

        if not gene_ids:
            return []

        B = delta.shape[0]
        num_tracks = delta.shape[2]
        scores_list = []

        for gene_id in gene_ids:
            gene_info = gene_annotation.get_gene_info(gene_id)
            if gene_info is None:
                continue

            gene_start = gene_info['start']
            gene_end = gene_info['end']
            gene_strand = gene_info.get('strand', '+')

            # Find junctions within gene body
            # JAX: (Start > gene_start) & (End < gene_end)
            # We use genomic coordinates (ref_starts, ref_ends already have interval_start added)
            for b in range(B):
                valid_indices = valid_mask[b].nonzero(as_tuple=True)[0]

                gene_junction_mask = torch.zeros(delta.shape[1], dtype=torch.bool, device=delta.device)

                for idx in valid_indices:
                    idx = idx.item()
                    j_start = ref_starts[b, idx].item()
                    j_end = ref_ends[b, idx].item()
                    j_strand = '+' if ref_strands[b, idx].item() == 0 else '-'

                    # Junction must be fully within gene body and match strand
                    if (j_start > gene_start and j_end < gene_end and j_strand == gene_strand):
                        gene_junction_mask[idx] = True

                if not gene_junction_mask.any():
                    continue

                # Get delta scores for junctions in this gene
                gene_delta = delta[b] * gene_junction_mask.unsqueeze(-1).float()  # [N, T]

                # Find max-scoring junction per track
                # For each track, find which junction has max score
                max_scores, max_indices = gene_delta.max(dim=0)  # [T], [T]

                # Get the junction that has max across all tracks (for coords)
                # This gives us a single junction to report
                overall_max_track = max_scores.argmax().item()
                best_junction_idx = max_indices[overall_max_track].item()

                best_start = int(ref_starts[b, best_junction_idx].item())
                best_end = int(ref_ends[b, best_junction_idx].item())

                scores_list.append(VariantScore(
                    variant=variant,
                    interval=interval,
                    scorer=self,
                    scores=max_scores,  # [T] - max per track
                    gene_id=gene_id,
                    gene_name=gene_info.get('gene_name'),
                    gene_type=gene_info.get('gene_type'),
                    gene_strand=gene_strand,
                    junction_start=best_start,
                    junction_end=best_end,
                ))

        return scores_list

    def _score_all_junctions(
        self,
        delta: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        strands: torch.Tensor,
        valid_mask: torch.Tensor,
        variant: Variant,
        interval: Interval,
    ) -> list[VariantScore]:
        """Return scores for all valid junctions when no gene annotation is provided.

        This is a fallback mode that returns one VariantScore per junction.
        """
        B = delta.shape[0]
        scores_list = []

        for b in range(B):
            valid_indices = valid_mask[b].nonzero(as_tuple=True)[0]

            for idx in valid_indices:
                idx = idx.item()
                j_start = int(starts[b, idx].item())
                j_end = int(ends[b, idx].item())
                j_scores = delta[b, idx]  # [T]

                scores_list.append(VariantScore(
                    variant=variant,
                    interval=interval,
                    scorer=self,
                    scores=j_scores,
                    junction_start=j_start,
                    junction_end=j_end,
                ))

        return scores_list
