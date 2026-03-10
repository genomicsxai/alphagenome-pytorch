"""Data transformations for AlphaGenome fine-tuning.

Provides functions to normalize and transform genomic signals following
the AlphaGenome data processing pipeline:

- ATAC-seq: mean normalization + smooth clipping (no power transform)
- RNA-seq: mean normalization + power transform (x^0.75) + smooth clipping

These transforms should be applied in the Dataset during data loading.

Example:
    >>> from alphagenome_pytorch.extensions.finetuning.data_transforms import (
    ...     apply_atac_transforms,
    ...     apply_rnaseq_transforms,
    ... )
    >>> atac_signal = apply_atac_transforms(raw_atac_signal)
    >>> rnaseq_signal = apply_rnaseq_transforms(raw_rnaseq_signal)
"""

from __future__ import annotations

import numpy as np
import torch


# Default thresholds from AlphaGenome paper
DEFAULT_SOFT_CLIP_THRESHOLD = 384.0
DEFAULT_TOTAL_COUNT = 100_000_000  # 100M insertions for ATAC


def normalize_to_total(
    x: np.ndarray | torch.Tensor,
    total: float = DEFAULT_TOTAL_COUNT,
) -> np.ndarray | torch.Tensor:
    """Scale tracks to a fixed total count.

    Used for ATAC-seq normalization to 100M insertions per track.

    Args:
        x: Input signal array.
        total: Target total count (default: 100M).

    Returns:
        Normalized signal scaled to the target total.
    """
    current_total = x.sum()
    if current_total == 0:
        return x
    return x * (total / current_total)


def mean_normalize(
    x: np.ndarray | torch.Tensor,
    epsilon: float = 1e-8,
) -> np.ndarray | torch.Tensor:
    """Divide by the mean of non-zero values.

    Args:
        x: Input signal array.
        epsilon: Small value to prevent division by zero.

    Returns:
        Signal divided by mean of non-zero values.
    """
    if isinstance(x, torch.Tensor):
        nonzero_mask = x > 0
        if nonzero_mask.sum() == 0:
            return x
        mean_val = x[nonzero_mask].mean()
    else:
        nonzero_mask = x > 0
        if nonzero_mask.sum() == 0:
            return x
        mean_val = x[nonzero_mask].mean()

    return x / (mean_val + epsilon)


def power_transform(
    x: np.ndarray | torch.Tensor,
    power: float = 0.75,
) -> np.ndarray | torch.Tensor:
    """Apply power law compression.

    Used for RNA-seq "squashing" to compress dynamic range.

    Args:
        x: Input signal array (must be non-negative).
        power: Exponent for power transform (default: 0.75).

    Returns:
        Power-transformed signal.
    """
    if isinstance(x, torch.Tensor):
        return torch.pow(x, power)
    return np.power(x, power)


def power_transform_inverse(
    x: np.ndarray | torch.Tensor,
    power: float = 0.75,
) -> np.ndarray | torch.Tensor:
    """Inverse of power_transform (for predictions).

    Args:
        x: Power-transformed signal.
        power: Original power used (default: 0.75).

    Returns:
        Signal in original scale.
    """
    if isinstance(x, torch.Tensor):
        return torch.pow(x, 1.0 / power)
    return np.power(x, 1.0 / power)


def smooth_clip(
    x: np.ndarray | torch.Tensor,
    threshold: float = DEFAULT_SOFT_CLIP_THRESHOLD,
) -> np.ndarray | torch.Tensor:
    """Dampen extreme values using square-root transformation.

    For x > threshold: x' = threshold + 2 * sqrt(x - threshold)

    This provides a smooth transition that dampens large values
    while preserving the ranking of signals.

    Args:
        x: Input signal array.
        threshold: Threshold above which to apply dampening.

    Returns:
        Signal with extreme values dampened.
    """
    if isinstance(x, torch.Tensor):
        return torch.where(
            x > threshold,
            threshold + 2.0 * torch.sqrt(x - threshold),
            x,
        )
    return np.where(
        x > threshold,
        threshold + 2.0 * np.sqrt(x - threshold),
        x,
    )


def smooth_clip_inverse(
    x: np.ndarray | torch.Tensor,
    threshold: float = DEFAULT_SOFT_CLIP_THRESHOLD,
) -> np.ndarray | torch.Tensor:
    """Inverse of smooth_clip (for predictions).

    Args:
        x: Clipped signal.
        threshold: Threshold used in smooth_clip.

    Returns:
        Signal in original scale.
    """
    if isinstance(x, torch.Tensor):
        return torch.where(
            x > threshold,
            threshold + ((x - threshold) / 2.0) ** 2,
            x,
        )
    return np.where(
        x > threshold,
        threshold + ((x - threshold) / 2.0) ** 2,
        x,
    )


def apply_atac_transforms(
    x: np.ndarray | torch.Tensor,
    total: float = DEFAULT_TOTAL_COUNT,
    clip_threshold: float = DEFAULT_SOFT_CLIP_THRESHOLD,
) -> np.ndarray | torch.Tensor:
    """Apply ATAC-seq transforms: normalize to total, mean normalize, clip.

    ATAC-seq does NOT use power transform (preserves linear count relationship).

    Args:
        x: Raw ATAC-seq signal.
        total: Target total count for normalization.
        clip_threshold: Threshold for smooth clipping.

    Returns:
        Transformed ATAC-seq signal.
    """
    x = normalize_to_total(x, total=total)
    x = mean_normalize(x)
    x = smooth_clip(x, threshold=clip_threshold)
    return x


def apply_rnaseq_transforms(
    x: np.ndarray | torch.Tensor,
    power: float = 0.75,
    clip_threshold: float = DEFAULT_SOFT_CLIP_THRESHOLD,
) -> np.ndarray | torch.Tensor:
    """Apply RNA-seq transforms: mean normalize, power transform, clip.

    RNA-seq uses the "squashing" power transform to compress dynamic range.

    Args:
        x: Raw RNA-seq signal.
        power: Power for compression transform.
        clip_threshold: Threshold for smooth clipping.

    Returns:
        Transformed RNA-seq signal.
    """
    x = mean_normalize(x)
    x = power_transform(x, power=power)
    x = smooth_clip(x, threshold=clip_threshold)
    return x


__all__ = [
    "normalize_to_total",
    "mean_normalize",
    "power_transform",
    "power_transform_inverse",
    "smooth_clip",
    "smooth_clip_inverse",
    "apply_atac_transforms",
    "apply_rnaseq_transforms",
    "DEFAULT_SOFT_CLIP_THRESHOLD",
    "DEFAULT_TOTAL_COUNT",
]
