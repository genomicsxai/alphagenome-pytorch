"""Base class for variant scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from ..types import Interval, OutputType, Variant, VariantScore

if TYPE_CHECKING:
    from ..annotations import GeneAnnotation


class BaseVariantScorer(ABC):
    """Abstract base class for variant scorers.

    All variant scorers must implement:
        - name: Unique identifier for the scorer configuration
        - requested_output: Which model output this scorer uses
        - is_signed: Whether scores can be negative (directional)
        - score(): Compute score from ref/alt model outputs
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this scorer configuration."""
        pass

    @property
    @abstractmethod
    def requested_output(self) -> OutputType:
        """The model output type this scorer uses."""
        pass

    @property
    @abstractmethod
    def is_signed(self) -> bool:
        """Whether scores are directional (can be negative)."""
        pass

    @abstractmethod
    def score(
        self,
        ref_outputs: dict[str, Any],
        alt_outputs: dict[str, Any],
        variant: Variant,
        interval: Interval,
        organism_index: int,
        **kwargs,
    ) -> VariantScore | list[VariantScore]:
        """Compute variant score from model outputs.

        Args:
            ref_outputs: Model outputs for reference sequence
            alt_outputs: Model outputs for alternate sequence
            variant: The variant being scored
            interval: Genomic interval of the input sequence
            organism_index: 0 for human, 1 for mouse
            **kwargs: Additional arguments (e.g., gene annotations)

        Returns:
            VariantScore or list of VariantScore (for gene-centric scorers)
        """
        pass

    def _get_predictions(
        self,
        outputs: dict[str, Any],
        resolution: int | None = None,
    ) -> torch.Tensor:
        """Extract predictions for this scorer's output type.

        Args:
            outputs: Model output dictionary
            resolution: Specific resolution to use (1 or 128).
                If None, uses default for the output type.

        Returns:
            Prediction tensor
        """
        output_key = self.requested_output.value

        if output_key not in outputs:
            raise KeyError(
                f"Output type '{output_key}' not found in model outputs. "
                f"Available keys: {list(outputs.keys())}"
            )

        output = outputs[output_key]

        # Handle dict outputs (multi-resolution heads)
        if isinstance(output, dict):
            if resolution is not None:
                if resolution not in output:
                    raise KeyError(
                        f"Resolution {resolution} not found for output '{output_key}'. "
                        f"Available: {list(output.keys())}"
                    )
                return output[resolution]
            else:
                # Default to highest resolution available
                if 1 in output:
                    return output[1]
                elif 128 in output:
                    return output[128]
                else:
                    return output[list(output.keys())[0]]

        return output

    def __repr__(self) -> str:
        return self.name
