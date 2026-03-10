"""
Metrics for AlphaGenome PyTorch.

Provides evaluation metrics for genomic predictions.
"""

from typing import Dict, Any, Optional
import torch
from torch import Tensor


def pearson_r(
    pred: Tensor,
    true: Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Compute Pearson correlation coefficient.

    Args:
        pred: Predicted values.
        true: True values.
        dim: Dimension to compute correlation over.
        eps: Small epsilon for numerical stability.

    Returns:
        Pearson correlation coefficient.
    """
    pred = pred.float()
    true = true.float()

    pred_centered = pred - pred.mean(dim=dim, keepdim=True)
    true_centered = true - true.mean(dim=dim, keepdim=True)

    numerator = (pred_centered * true_centered).sum(dim=dim)
    denominator = (
        pred_centered.pow(2).sum(dim=dim).sqrt() *
        true_centered.pow(2).sum(dim=dim).sqrt()
    )

    return numerator / (denominator + eps)


def profile_pearson_r(
    pred: Tensor,
    true: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute Pearson R for profile shape within each region.

    For each (region, track) pair, computes correlation between predicted
    and observed values across all genomic positions within that region.
    This measures how well the predicted signal shape matches the true shape.

    Following the AlphaGenome paper:
    "For a given track and a held-out test interval, Pearson r was calculated
    between the vector of predicted values and the vector of observed values
    across all corresponding genomic bins within that interval."

    Args:
        pred: Predictions with shape (N_regions, seq_len, tracks).
        true: Targets with shape (N_regions, seq_len, tracks).
        eps: Small epsilon for numerical stability.

    Returns:
        Per-region, per-track correlation with shape (N_regions, tracks).
        To get mean profile Pearson R: result.mean()
        To get per-track mean: result.mean(dim=0)
    """
    # Correlation over positions (dim=1) for each region and track
    return pearson_r(pred, true, dim=1, eps=eps)


def count_pearson_r(
    pred: Tensor,
    true: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute Pearson R for total counts across all regions.

    For each track, computes ONE correlation between predicted and observed
    total counts using all N regions as data points:
        corr([sum(pred_1), ..., sum(pred_N)], [sum(true_1), ..., sum(true_N)])

    This measures whether regions with high observed signal also have
    high predicted signal.

    Args:
        pred: Predictions with shape (N_regions, seq_len, tracks).
        true: Targets with shape (N_regions, seq_len, tracks).
        eps: Small epsilon for numerical stability.

    Returns:
        Per-track correlation with shape (tracks,).
        Each value is a single Pearson R computed across all N regions.
    """
    # Sum over positions to get total counts per region
    pred_counts = pred.sum(dim=1)  # (N_regions, tracks)
    true_counts = true.sum(dim=1)  # (N_regions, tracks)

    # For each track, correlate counts across all regions
    return pearson_r(pred_counts, true_counts, dim=0, eps=eps)


def compute_metrics(
    pred: Tensor,
    true: Tensor,
    track_names: Optional[list] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute comprehensive metrics for genomic predictions.

    Args:
        pred: Predictions with shape (batch, seq_len, tracks).
        true: Targets with shape (batch, seq_len, tracks).
        track_names: Optional list of track names for labeling.
        eps: Small epsilon for numerical stability.

    Returns:
        Dictionary with metrics:
            - profile_pearson_r: Mean profile correlation (across batch and tracks)
            - profile_pearson_r_per_track: Per-track mean profile correlation
            - count_pearson_r: Mean count correlation (across tracks)
            - count_pearson_r_per_track: Per-track count correlation
    """
    results = {}

    # Profile Pearson R: correlation over positions
    profile_r = profile_pearson_r(pred, true, eps=eps)  # (batch, tracks)
    results["profile_pearson_r"] = profile_r.mean().item()

    # Per-track profile Pearson R (averaged over batch)
    profile_r_per_track = profile_r.mean(dim=0)  # (tracks,)

    # Count Pearson R: correlation over samples after summing positions
    if pred.shape[0] > 1:  # Need at least 2 samples for meaningful correlation
        count_r = count_pearson_r(pred, true, eps=eps)  # (tracks,)
        results["count_pearson_r"] = count_r.mean().item()
    else:
        count_r = None
        results["count_pearson_r"] = float("nan")

    # Add per-track metrics if track names provided
    if track_names is not None:
        for i, name in enumerate(track_names):
            results[f"profile_pearson_r_{name}"] = profile_r_per_track[i].item()
            if count_r is not None:
                results[f"count_pearson_r_{name}"] = count_r[i].item()

    return results


def spearman_r(
    pred: Tensor,
    true: Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Compute Spearman rank correlation coefficient.
    
    Args:
        pred: Predicted values.
        true: True values.
        dim: Dimension to compute correlation over.
        eps: Small epsilon for numerical stability.
        
    Returns:
        Spearman correlation coefficient.
    """
    # Convert to ranks
    def to_ranks(x: Tensor, dim: int) -> Tensor:
        return x.argsort(dim=dim).argsort(dim=dim).float()
    
    pred_ranks = to_ranks(pred, dim)
    true_ranks = to_ranks(true, dim)
    
    return pearson_r(pred_ranks, true_ranks, dim=dim, eps=eps)


class AlphaGenomeMetrics:
    """Compute validation metrics for AlphaGenome.
    
    Computes per-head Pearson correlation and optionally other metrics.
    Extensible via custom metric functions.
    
    Args:
        heads: List of head names to compute metrics for.
        additional_metrics: Dict of name -> callable(pred, true) for extra metrics.
        
    Example:
        >>> metrics = AlphaGenomeMetrics()
        >>> results = metrics(outputs, targets)
        >>> print(results)
        {'atac_pearson_r': 0.85, 'dnase_pearson_r': 0.82, ...}
    """
    
    def __init__(
        self,
        heads: Optional[list] = None,
        additional_metrics: Optional[Dict[str, callable]] = None,
    ):
        self.heads = heads
        self.additional_metrics = additional_metrics or {}
    
    def __call__(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """Compute metrics for all heads.
        
        Args:
            outputs: Model outputs dict.
            targets: Target values dict.
            
        Returns:
            Dict of metric names to values.
        """
        results = {}
        heads = self.heads or list(outputs.keys())
        
        for head in heads:
            if head not in outputs or head not in targets:
                continue
                
            pred = self._extract_tensor(outputs[head])
            true = self._extract_tensor(targets[head])
            
            if pred is None or true is None:
                continue
            
            # Compute Pearson R (primary metric)
            r = pearson_r(pred.flatten(), true.flatten()).item()
            results[f'{head}_pearson_r'] = r
            
            # Compute additional metrics
            for metric_name, metric_fn in self.additional_metrics.items():
                try:
                    value = metric_fn(pred, true)
                    if isinstance(value, Tensor):
                        value = value.item()
                    results[f'{head}_{metric_name}'] = value
                except Exception:
                    pass  # Skip failed metrics
        
        # Compute average Pearson R across heads
        pearson_values = [v for k, v in results.items() if k.endswith('_pearson_r')]
        if pearson_values:
            results['avg_pearson_r'] = sum(pearson_values) / len(pearson_values)
        
        return results
    
    def _extract_tensor(self, x: Any) -> Optional[Tensor]:
        """Extract tensor from possibly nested structure."""
        if isinstance(x, Tensor):
            return x
        if isinstance(x, dict):
            # Use highest resolution
            res_keys = [k for k in x.keys() if isinstance(k, int)]
            if res_keys:
                return x[min(res_keys)]
            # Try first value
            for v in x.values():
                if isinstance(v, Tensor):
                    return v
        return None


__all__ = [
    'pearson_r',
    'profile_pearson_r',
    'count_pearson_r',
    'compute_metrics',
    'spearman_r',
    'AlphaGenomeMetrics',
]
