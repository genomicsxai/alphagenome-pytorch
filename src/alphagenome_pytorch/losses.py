"""
Losses for AlphaGenome PyTorch.

Ported from alphagenome_research.model.losses (JAX implementation).
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _safe_masked_mean(
    x: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Safe mean that handles completely masked arrays.
    
    Args:
        x: Input tensor of any shape.
        mask: Boolean mask tensor, broadcastable to x.shape.
        
    Returns:
        Scalar tensor with the masked mean.
    """
    if mask is None:
        masked = x
        mask = torch.ones_like(x)
    else:
        # Broadcast mask to compute correct mean
        mask = mask.expand_as(x).float()
        masked = x * mask
    
    return masked.sum() / torch.clamp(mask.sum(), min=1.0)


def poisson_loss(
    *,
    y_true: Tensor,
    y_pred: Tensor,
    mask: Tensor,
) -> Tensor:
    """Poisson loss.
    
    Args:
        y_true: Target values.
        y_pred: Model predictions.
        mask: Boolean mask tensor.
        
    Returns:
        Scalar loss value.
    """
    y_true = y_true.abs().float()
    y_pred = y_pred.float()
    y_pred_logits = torch.log(y_pred + 1e-7)
    
    # Subtract the minimum value such that loss is zero at optimal prediction
    min_value = y_true - y_true * torch.log(y_true + 1e-7)
    loss = (y_pred - y_true * y_pred_logits) - min_value
    
    return _safe_masked_mean(loss, mask)


def multinomial_loss(
    *,
    y_true: Tensor,
    y_pred: Tensor,
    mask: Tensor,
    multinomial_resolution: int,
    positional_weight: float,
    count_weight: float = 1.0,
    channels_last: bool = True,
) -> Dict[str, Tensor]:
    """Returns sum of multinomial losses and Poisson loss on total count.
    
    Args:
        y_true: Target values with shape (..., S, C) if channels_last else (..., C, S).
        y_pred: Model predictions with shape (..., S, C) if channels_last else (..., C, S).
        mask: Boolean mask with shape (..., 1, C) if channels_last else (..., C, 1).
        multinomial_resolution: Splits input into sub-sequences and computes
            separate multinomial loss over each sub-sequence.
        positional_weight: Weight of the positional loss.
        count_weight: Weight of the count (Poisson) loss. Default 1.0.
        channels_last: If True, assumes (..., S, C). If False, assumes (..., C, S).

    Returns:
        Dictionary with:
            - 'loss': Combined loss
            - 'loss_total': Poisson loss on total counts
            - 'loss_positional': Positional (multinomial) loss
            - 'max_sum_preds': Maximum sum of predictions
            - 'max_preds': Maximum prediction value
            - 'max_targets': Maximum target value
    """
    assert y_true.shape == y_pred.shape, f"{y_true.shape=} != {y_pred.shape=}"
    
    # Normalize to channels-last for a single implementation path.
    if channels_last:
        if mask.shape[-2] != 1:
            raise ValueError(
                f"For channels_last=True, expected mask shape (..., 1, C), got {mask.shape}."
            )
    else:
        if mask.shape[-1] != 1:
            raise ValueError(
                f"For channels_last=False, expected mask shape (..., C, 1), got {mask.shape}."
            )
        y_true = y_true.transpose(-1, -2).contiguous()
        y_pred = y_pred.transpose(-1, -2).contiguous()
        mask = mask.transpose(-1, -2).contiguous()
    
    seq_len = y_pred.shape[-2]
    if seq_len % multinomial_resolution != 0:
        raise ValueError(
            f'{seq_len=} must be divisible by {multinomial_resolution=}.'
        )
    
    num_segments = seq_len // multinomial_resolution
    
    # Remove the masked out bins from the totals sum
    mask_float = mask.float()
    y_true = torch.clamp(y_true, min=0) * mask_float
    y_pred = y_pred * mask_float
    
    # Split sequence into n sub-sequences of size multinomial_resolution
    # Shape: (..., S, C) -> (..., num_segments, multinomial_resolution, C)
    batch_dims = y_true.shape[:-2]
    channels = y_true.shape[-1]
    
    y_true = y_true.reshape(*batch_dims, num_segments, multinomial_resolution, channels)
    y_pred = y_pred.reshape(*batch_dims, num_segments, multinomial_resolution, channels)
    
    # Sum over the resolution dimension to get totals per segment
    # Accummulate in float32
    total_pred = y_pred.sum(dim=-2, keepdim=True, dtype=torch.float32)  # (..., n, 1, C)
    total_true = y_true.sum(dim=-2, keepdim=True, dtype=torch.float32)  # (..., n, 1, C)
    
    # Broadcast mask over segments
    mask_expanded = mask.unsqueeze(-2)  # (..., 1, 1, C)
    
    loss_total_count = poisson_loss(
        y_true=total_true,
        y_pred=total_pred,
        mask=mask_expanded,
    )
    
    # Normalize by resolution to keep loss magnitude invariant
    loss_total_count = loss_total_count / multinomial_resolution
    
    # Positional loss (cross-entropy style)
    prob_predictions = y_pred.float() / (total_pred + 1e-7)
    loss_positional = -y_true * torch.log(prob_predictions + 1e-7)
    loss_positional = _safe_masked_mean(loss_positional, mask=mask_expanded)
    
    return {
        'loss': count_weight * loss_total_count + positional_weight * loss_positional,
        'loss_total': loss_total_count,
        'loss_positional': loss_positional,
        'max_sum_preds': total_pred.max(),
        'max_preds': y_pred.max(),
        'max_targets': y_true.max().float(),
    }


def mse(
    y_pred: Tensor,
    y_true: Tensor,
    mask: Tensor,
) -> Tensor:
    """Mean squared error.
    
    Args:
        y_pred: Predictions.
        y_true: Targets.
        mask: Boolean mask.
        
    Returns:
        Scalar MSE loss.
    """
    return _safe_masked_mean(torch.square(y_pred - y_true), mask)


def cross_entropy_loss_from_logits(
    *,
    y_pred_logits: Tensor,
    y_true: Tensor,
    mask: Tensor,
    axis: int,
) -> Tensor:
    """Cross-entropy loss from logits.
    
    Args:
        y_pred_logits: Prediction logits.
        y_true: Target probabilities or one-hot.
        mask: Boolean mask.
        axis: Axis for softmax.
        
    Returns:
        Scalar loss.
    """
    log_softmax_preds = F.log_softmax(y_pred_logits.float(), dim=axis)
    loss = -(y_true.float() * log_softmax_preds).sum(dim=axis)
    mask_reduced = mask.any(dim=axis)
    return _safe_masked_mean(loss, mask_reduced)


def binary_crossentropy_from_logits(
    *,
    y_pred: Tensor,
    y_true: Tensor,
    mask: Tensor,
) -> Tensor:
    """Binary cross-entropy loss from sigmoid logits.
    
    Uses the numerically stable formulation.
    
    Args:
        y_pred: Prediction logits (pre-sigmoid).
        y_true: Binary targets.
        mask: Boolean mask.
        
    Returns:
        Scalar loss.
    """
    # Numerically stable BCE: max(x, 0) - x*y + log(1 + exp(-|x|))
    loss = (
        torch.clamp(y_pred, min=0)
        - y_pred * y_true
        + torch.log1p(torch.exp(-torch.abs(y_pred)))
    )
    return _safe_masked_mean(loss, mask)


def cross_entropy_loss(
    *,
    y_true: Tensor,
    y_pred: Tensor,
    mask: Tensor,
    axis: int,
    eps: float = 1e-7,
) -> Tensor:
    """Cross entropy loss on counts.
    
    Args:
        y_true: Target counts.
        y_pred: Predicted counts.
        mask: Boolean mask.
        axis: Axis for normalization.
        eps: Small epsilon for numerical stability.
        
    Returns:
        Scalar loss.
    """
    mask = mask.expand_as(y_true)
    assert y_true.shape == y_pred.shape == mask.shape
    
    y_true = torch.where(mask, y_true.float(), torch.zeros_like(y_true.float()))
    p_true = y_true / torch.clamp(y_true.sum(dim=axis, keepdim=True), min=eps)
    
    masked_pred = torch.where(mask, y_pred, torch.zeros_like(y_pred))
    log_normalizer = torch.log((masked_pred + eps).sum(dim=axis))
    log_likelihood = (p_true * torch.log(y_pred + eps)).sum(dim=axis)
    
    log_loss = log_normalizer - log_likelihood
    return _safe_masked_mean(log_loss, mask.any(dim=axis))
