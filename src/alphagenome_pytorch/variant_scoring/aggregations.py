"""Aggregation functions for variant scoring.

This module implements the 8 aggregation methods for comparing reference
and alternate predictions in variant scoring, plus the align_alternate
function for indel handling.
"""

from __future__ import annotations

import torch

from .types import AggregationType


def align_alternate(
    alt: torch.Tensor,
    variant_start: int,
    ref_length: int,
    alt_length: int,
    interval_start: int,
) -> torch.Tensor:
    """Align ALT predictions to REF coordinate space for indels.

    This function adjusts the `alt` prediction tensor to account for indels
    (insertions or deletions) in the variant.

    For insertions: summarize inserted region with max, pad end with zeros.
    For deletions: insert zeros at deletion locations.

    Args:
        alt: ALT allele predictions of shape (S, T) where S=sequence, T=tracks.
        variant_start: 0-based start position of the variant in genomic coords.
        ref_length: Length of reference allele.
        alt_length: Length of alternate allele.
        interval_start: 0-based start of the sequence interval.

    Returns:
        Aligned ALT predictions of shape (S, T), matching REF coordinate space.
    """
    insertion_length = alt_length - ref_length
    deletion_length = -insertion_length
    variant_start_in_vector = variant_start - interval_start
    # Assume left-aligned variants; adjustments occur at end of variant
    variant_start_in_vector += min(ref_length, alt_length) - 1
    original_length = alt.shape[0]

    if insertion_length > 0:
        # Summarize insertion by computing max score across alternate bases
        pool_range = slice(
            variant_start_in_vector,
            variant_start_in_vector + insertion_length + 1
        )
        pool_alt = alt[pool_range].max(dim=0, keepdim=True)[0]
        alt = torch.cat([
            alt[:variant_start_in_vector],
            pool_alt,
            alt[variant_start_in_vector + insertion_length + 1:],
            torch.zeros(insertion_length, alt.shape[1], device=alt.device, dtype=alt.dtype),
        ], dim=0)
        # Truncate to original length
        alt = alt[:original_length]
    elif deletion_length > 0:
        # Insert zero signal at deletion locations
        alt = torch.cat([
            alt[:variant_start_in_vector + 1],
            torch.zeros(deletion_length, alt.shape[1], device=alt.device, dtype=alt.dtype),
            alt[variant_start_in_vector + 1:],
        ], dim=0)
        alt = alt[:original_length]

    return alt


def compute_aggregation(
    ref_preds: torch.Tensor,
    alt_preds: torch.Tensor,
    aggregation_type: AggregationType,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute aggregated score from reference and alternate predictions.

    Args:
        ref_preds: Reference predictions of shape (B, S, T) or (B, S)
            where B=batch, S=sequence length, T=tracks
        alt_preds: Alternate predictions, same shape as ref_preds
        aggregation_type: Type of aggregation to apply
        mask: Optional spatial mask of shape (B, S) where True = include.
            If None, all positions are included.
        eps: Small constant for numerical stability in log operations

    Returns:
        Aggregated scores of shape (B, T) or (B,) depending on input shape
    """
    if ref_preds.shape != alt_preds.shape:
        raise ValueError(
            f"ref_preds shape {ref_preds.shape} != alt_preds shape {alt_preds.shape}"
        )

    # Ensure 3D for consistent handling: (B, S, T)
    squeeze_output = False
    if ref_preds.dim() == 2:
        ref_preds = ref_preds.unsqueeze(-1)
        alt_preds = alt_preds.unsqueeze(-1)
        squeeze_output = True

    B, S, T = ref_preds.shape

    # Apply mask if provided
    if mask is not None:
        if mask.shape != (B, S):
            raise ValueError(f"mask shape {mask.shape} != expected ({B}, {S})")
        # Expand mask to match predictions: (B, S) -> (B, S, 1)
        mask = mask.unsqueeze(-1).expand_as(ref_preds)
        ref_masked = ref_preds.masked_fill(~mask, 0.0)
        alt_masked = alt_preds.masked_fill(~mask, 0.0)
        # Count valid positions per batch/track
        valid_counts = mask.float().sum(dim=1)  # (B, T)
        valid_counts = valid_counts.clamp(min=1.0)  # Avoid division by zero
    else:
        ref_masked = ref_preds
        alt_masked = alt_preds
        valid_counts = torch.full((B, T), S, device=ref_preds.device, dtype=ref_preds.dtype)

    if aggregation_type == AggregationType.DIFF_MEAN:
        # mean(ALT) - mean(REF)
        ref_mean = ref_masked.sum(dim=1) / valid_counts
        alt_mean = alt_masked.sum(dim=1) / valid_counts
        result = alt_mean - ref_mean

    elif aggregation_type == AggregationType.DIFF_SUM:
        # sum(ALT) - sum(REF)
        # Use float64 for accumulation to matching ACTIVE_SUM precision
        ref_sum = ref_masked.to(torch.float64).sum(dim=1).to(ref_preds.dtype)
        alt_sum = alt_masked.to(torch.float64).sum(dim=1).to(ref_preds.dtype)
        result = alt_sum - ref_sum

    elif aggregation_type == AggregationType.DIFF_SUM_LOG2:
        # sum(log2(ALT + 1)) - sum(log2(REF + 1))
        # Use log2(x + 1) for stability to handle zeros (matches JAX)
        ref_log = torch.log2(ref_preds + 1)
        alt_log = torch.log2(alt_preds + 1)
        if mask is not None:
            ref_log = ref_log.masked_fill(~mask, 0.0)
            alt_log = alt_log.masked_fill(~mask, 0.0)
        # Use float64 for accumulation
        result = alt_log.to(torch.float64).sum(dim=1).to(ref_preds.dtype) - \
                 ref_log.to(torch.float64).sum(dim=1).to(ref_preds.dtype)

    elif aggregation_type == AggregationType.DIFF_LOG2_SUM:
        # log2(1 + sum(ALT)) - log2(1 + sum(REF))
        # Use log2(1 + sum) for stability (matches JAX)
        # Use float64 for accumulation matching JAX/ACTIVE_SUM precision
        ref_sum = ref_masked.to(torch.float64).sum(dim=1).to(ref_preds.dtype)
        alt_sum = alt_masked.to(torch.float64).sum(dim=1).to(ref_preds.dtype)
        result = torch.log2(1 + alt_sum) - torch.log2(1 + ref_sum)

    elif aggregation_type == AggregationType.L2_DIFF:
        # ||ALT - REF||_2 (L2 norm of difference)
        diff = alt_preds - ref_preds
        if mask is not None:
            diff = diff.masked_fill(~mask, 0.0)
        # L2 norm over spatial dimension
        result = torch.sqrt((diff ** 2).sum(dim=1) + eps)

    elif aggregation_type == AggregationType.L2_DIFF_LOG1P:
        # ||log1p(ALT) - log1p(REF)||_2
        ref_log = torch.log1p(ref_preds)
        alt_log = torch.log1p(alt_preds)
        diff = alt_log - ref_log
        if mask is not None:
            diff = diff.masked_fill(~mask, 0.0)
        result = torch.sqrt((diff ** 2).sum(dim=1) + eps)

    elif aggregation_type == AggregationType.ACTIVE_MEAN:
        # max(mean(ALT), mean(REF))
        ref_mean = ref_masked.sum(dim=1) / valid_counts
        alt_mean = alt_masked.sum(dim=1) / valid_counts
        result = torch.maximum(ref_mean, alt_mean)

    elif aggregation_type == AggregationType.ACTIVE_SUM:
        # max(sum(ALT), sum(REF))
        # Use float64 for accumulation to reduce precision loss over large windows
        ref_sum = ref_masked.to(torch.float64).sum(dim=1).to(ref_preds.dtype)
        alt_sum = alt_masked.to(torch.float64).sum(dim=1).to(ref_preds.dtype)
        result = torch.maximum(ref_sum, alt_sum)

    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")

    if squeeze_output:
        result = result.squeeze(-1)

    return result


def create_center_mask(
    variant_position: int,
    interval_start: int,
    width: int | None,
    seq_length: int,
    resolution: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a spatial mask centered on a variant position.

    Args:
        variant_position: 1-based variant position (VCF convention)
        interval_start: 0-based start of the sequence interval
        width: Width of the mask in base pairs. If None, returns all-True mask.
        seq_length: Length of the sequence in bins (at given resolution)
        resolution: Bin size in base pairs (1 for 1bp, 128 for 128bp)
        device: Device for the mask tensor

    Returns:
        Boolean mask tensor of shape (seq_length,) where True = include
    """
    if width is None:
        return torch.ones(seq_length, dtype=torch.bool, device=device)

    # Convert variant position to 0-based position relative to interval
    variant_0based = variant_position - 1  # Convert from 1-based to 0-based
    rel_position = variant_0based - interval_start

    # Create mask
    mask = torch.zeros(seq_length, dtype=torch.bool, device=device)

    # Use 0-based bp coordinates relative to interval start
    start_bp = rel_position - (width // 2)
    end_bp = rel_position + (width // 2) + (width % 2) # Ensure total width matches input width

    # Convert to bin coordinates (inclusive of any bin with overlap)
    # Start bin: covers start_bp
    start_bin = max(0, start_bp // resolution)
    # End bin: covers end_bp-1 (exclusive of end_bp)
    end_bin = min(seq_length, (end_bp - 1) // resolution + 1)

    mask[start_bin:end_bin] = True

    return mask
