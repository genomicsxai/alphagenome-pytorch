"""CenterMaskScorer for spatial masking around variant position."""

from __future__ import annotations

from typing import Any

import torch

from ..aggregations import compute_aggregation, create_center_mask
from ..types import AggregationType, Interval, OutputType, Variant, VariantScore
from .base import BaseVariantScorer

# Supported output types for CenterMaskScorer
SUPPORTED_OUTPUT_TYPES = frozenset([
    OutputType.ATAC,
    OutputType.CAGE,
    OutputType.DNASE,
    OutputType.PROCAP,
    OutputType.RNA_SEQ,
    OutputType.CHIP_HISTONE,
    OutputType.CHIP_TF,
    OutputType.SPLICE_SITES,
    OutputType.SPLICE_SITE_USAGE,
])

# Supported mask widths (in base pairs)
SUPPORTED_WIDTHS = frozenset([None, 501, 2001, 10_001, 100_001, 200_001])

# All aggregation types are supported
SUPPORTED_AGGREGATIONS = frozenset(AggregationType)


class CenterMaskScorer(BaseVariantScorer):
    """Variant scorer using spatial masking around the variant position.

    Aggregates ALT and REF predictions using a spatial mask centered on the
    variant before computing the difference. Returns one score per output track.

    This is the most versatile scorer and works with most track-based outputs
    (ATAC, DNASE, CAGE, ChIP, etc.).

    Args:
        requested_output: The model output type to score (e.g., OutputType.ATAC)
        width: Width of the spatial mask in base pairs. If None, uses full sequence.
            Supported: None, 501, 2001, 10001, 100001, 200001
        aggregation_type: How to compare ref and alt predictions.
            See AggregationType for options.
        resolution: Output resolution to use (1 for 1bp, 128 for 128bp).
            If None (default), auto-selects based on width:
            - width <= 2001: uses 1bp for fine-grained spatial detail
            - width > 2001 or None: uses 128bp for efficiency
            Note: CHIP_TF and CHIP_HISTONE only support 128bp resolution.

    Example:
        >>> scorer = CenterMaskScorer(
        ...     requested_output=OutputType.ATAC,
        ...     width=501,
        ...     aggregation_type=AggregationType.DIFF_LOG2_SUM,
        ... )
        >>> scorer.is_signed
        True

        >>> # Explicit resolution control
        >>> scorer = CenterMaskScorer(
        ...     requested_output=OutputType.ATAC,
        ...     width=501,
        ...     aggregation_type=AggregationType.DIFF_LOG2_SUM,
        ...     resolution=128,  # Force 128bp for efficiency
        ... )
    """

    def __init__(
        self,
        requested_output: OutputType,
        width: int | None,
        aggregation_type: AggregationType,
        resolution: int | None = None,
    ):
        if requested_output not in SUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                f"Unsupported output type: {requested_output}. "
                f"Supported: {sorted(o.value for o in SUPPORTED_OUTPUT_TYPES)}"
            )
        if width not in SUPPORTED_WIDTHS:
            raise ValueError(
                f"Unsupported width: {width}. "
                f"Supported: {sorted(w for w in SUPPORTED_WIDTHS if w is not None)}"
            )
        if aggregation_type not in SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported aggregation: {aggregation_type}. "
                f"Supported: {sorted(a.value for a in SUPPORTED_AGGREGATIONS)}"
            )
        if resolution is not None and resolution not in (1, 128):
            raise ValueError(
                f"Resolution must be 1, 128, or None (auto), got {resolution}"
            )

        self._requested_output = requested_output
        self._width = width
        self._aggregation_type = aggregation_type

        # Determine resolution: explicit or auto-select based on width
        # - 1bp resolution gives finer spatial detail but higher computational cost
        # - 128bp resolution is faster and sufficient for wider windows
        if resolution is not None:
            self._resolution = resolution
        elif requested_output in (OutputType.CHIP_TF, OutputType.CHIP_HISTONE):
            # ChIP outputs only have 128bp resolution
            self._resolution = 128
        elif width is not None and width <= 2001:
            # Narrow windows benefit from 1bp precision
            self._resolution = 1
        else:
            # Wide windows or full sequence use 128bp for efficiency
            self._resolution = 128

    @property
    def requested_output(self) -> OutputType:
        return self._requested_output

    @property
    def width(self) -> int | None:
        return self._width

    @property
    def aggregation_type(self) -> AggregationType:
        return self._aggregation_type

    @property
    def resolution(self) -> int:
        """Output resolution (1 for 1bp, 128 for 128bp)."""
        return self._resolution

    @property
    def name(self) -> str:
        width_str = str(self._width) if self._width else 'full'
        return (
            f"CenterMaskScorer("
            f"output={self._requested_output.value}, "
            f"width={width_str}, "
            f"agg={self._aggregation_type.value}, "
            f"res={self._resolution}bp)"
        )

    @property
    def is_signed(self) -> bool:
        return self._aggregation_type.is_signed()

    def score(
        self,
        ref_outputs: dict[str, Any],
        alt_outputs: dict[str, Any],
        variant: Variant,
        interval: Interval,
        organism_index: int,
        **kwargs,
    ) -> VariantScore:
        """Compute variant score using center mask aggregation.

        Args:
            ref_outputs: Model outputs for reference sequence
            alt_outputs: Model outputs for alternate sequence
            variant: The variant being scored
            interval: Genomic interval of the input sequence
            organism_index: 0 for human, 1 for mouse

        Returns:
            VariantScore with per-track scores
        """
        # Use stored resolution (determined at init based on width or explicit setting)
        output_key = self._requested_output.value
        resolution = self._resolution

        # Fall back to 128bp if requested 1bp is not available
        if isinstance(ref_outputs.get(output_key), dict):
            if resolution == 1 and 1 not in ref_outputs[output_key]:
                resolution = 128

        # Get predictions
        ref_preds = self._get_predictions(ref_outputs, resolution)
        alt_preds = self._get_predictions(alt_outputs, resolution)

        # Handle splice outputs which have special structure
        if self._requested_output == OutputType.SPLICE_SITES:
            # splice_sites_classification has 'probs' key
            if isinstance(ref_preds, dict) and 'probs' in ref_preds:
                ref_preds = ref_preds['probs']
                alt_preds = alt_preds['probs']

        # Ensure batch dimension
        if ref_preds.dim() == 2:
            ref_preds = ref_preds.unsqueeze(0)
            alt_preds = alt_preds.unsqueeze(0)

        B, S, T = ref_preds.shape

        # Create center mask
        mask = create_center_mask(
            variant_position=variant.position,
            interval_start=interval.start,
            width=self._width,
            seq_length=S,
            resolution=resolution,
            device=ref_preds.device,
        )
        mask = mask.unsqueeze(0).expand(B, -1)  # (B, S)

        # Compute aggregated scores
        scores = compute_aggregation(
            ref_preds,
            alt_preds,
            self._aggregation_type,
            mask=mask,
        )

        # Remove batch dimension if single sample
        if scores.shape[0] == 1:
            scores = scores.squeeze(0)

        return VariantScore(
            variant=variant,
            interval=interval,
            scorer=self,
            scores=scores,
        )
