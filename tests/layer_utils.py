
import pytest
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# METRICS AND COMPARISON UTILITIES
# =============================================================================


@dataclass
class LayerComparisonResult:
    """Results of comparing two layer outputs."""

    name: str
    shape: tuple
    passed: bool

    # Core metrics
    pearson_corr: float
    cosine_sim: float
    scale_ratio: float  # std(pytorch) / std(jax)

    # Absolute differences
    max_diff: float
    mean_diff: float

    # Statistics
    pt_std: float
    jax_std: float
    pt_mean: float
    jax_mean: float


def compute_metrics(
    name: str,
    pt_arr: np.ndarray,
    jax_arr: np.ndarray,
    corr_threshold: float = 0.9999,
) -> LayerComparisonResult:
    """Compute comparison metrics between two arrays.

    Args:
        name: Layer name for reporting
        pt_arr: PyTorch output (float32)
        jax_arr: JAX output (float32)
        corr_threshold: Minimum correlation to pass (default 0.9999)

    Returns:
        LayerComparisonResult with all metrics
    """
    assert pt_arr.shape == jax_arr.shape, f"Shape mismatch: {pt_arr.shape} vs {jax_arr.shape}"

    pt_flat = pt_arr.flatten().astype(np.float64)
    jax_flat = jax_arr.flatten().astype(np.float64)

    # Statistics
    pt_std = float(np.std(pt_flat))
    jax_std = float(np.std(jax_flat))
    pt_mean = float(np.mean(pt_flat))
    jax_mean = float(np.mean(jax_flat))

    # Pearson correlation
    if pt_std > 1e-10 and jax_std > 1e-10:
        corr = float(np.corrcoef(pt_flat, jax_flat)[0, 1])
    else:
        # Both nearly constant - check if they're the same constant
        corr = 1.0 if np.allclose(pt_flat, jax_flat, rtol=1e-5, atol=1e-6) else 0.0

    # Cosine similarity
    pt_norm = np.linalg.norm(pt_flat)
    jax_norm = np.linalg.norm(jax_flat)
    if pt_norm > 1e-10 and jax_norm > 1e-10:
        cosine_sim = float(np.dot(pt_flat, jax_flat) / (pt_norm * jax_norm))
    else:
        cosine_sim = 1.0 if np.allclose(pt_flat, jax_flat, rtol=1e-5, atol=1e-6) else 0.0

    # Scale ratio: std(pytorch) / std(jax)
    # If JAX std is near zero, use 1.0 (avoid division by zero)
    scale_ratio = pt_std / jax_std if jax_std > 1e-10 else 1.0

    # Absolute differences
    diff = np.abs(pt_flat - jax_flat)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    # Pass criteria: correlation above threshold
    passed = corr >= corr_threshold

    return LayerComparisonResult(
        name=name,
        shape=pt_arr.shape,
        passed=passed,
        pearson_corr=corr,
        cosine_sim=cosine_sim,
        scale_ratio=scale_ratio,
        max_diff=max_diff,
        mean_diff=mean_diff,
        pt_std=pt_std,
        jax_std=jax_std,
        pt_mean=pt_mean,
        jax_mean=jax_mean,
    )


def print_layer_result(result: LayerComparisonResult, verbose: bool = True):
    """Print formatted comparison result."""
    status = "PASS" if result.passed else "FAIL"
    status_marker = " " if result.passed else "***"

    if verbose:
        print(f"\n{'='*80}")
        print(f"{status_marker} LAYER: {result.name} [{status}]")
        print(f"{'='*80}")
        print(f"Shape: {result.shape}")
        print(f"-"*80)
        print(f"{'Metric':<25} {'Value':>15} {'Threshold':>15} {'Status':>10}")
        print(f"-"*80)

        corr_status = "OK" if result.pearson_corr >= 0.9999 else "FAIL"
        print(f"{'Pearson Correlation':<25} {result.pearson_corr:>15.6f} {'>= 0.9999':>15} {corr_status:>10}")

        cos_status = "OK" if result.cosine_sim >= 0.9999 else "WARN"
        print(f"{'Cosine Similarity':<25} {result.cosine_sim:>15.6f} {'>= 0.9999':>15} {cos_status:>10}")

        # Scale ratio: should be close to 1.0 (within 1%)
        scale_status = "OK" if 0.99 <= result.scale_ratio <= 1.01 else "WARN"
        print(f"{'Scale Ratio (PT/JAX)':<25} {result.scale_ratio:>15.6f} {'~1.0':>15} {scale_status:>10}")

        print(f"-"*80)
        print(f"{'Max Absolute Diff':<25} {result.max_diff:>15.6f}")
        print(f"{'Mean Absolute Diff':<25} {result.mean_diff:>15.6f}")
        print(f"-"*80)
        print(f"{'PyTorch std':<25} {result.pt_std:>15.6f}")
        print(f"{'JAX std':<25} {result.jax_std:>15.6f}")
        print(f"{'PyTorch mean':<25} {result.pt_mean:>15.6f}")
        print(f"{'JAX mean':<25} {result.jax_mean:>15.6f}")
        print(f"{'='*80}\n")
    else:
        # One-line summary
        print(f"{status_marker} {result.name:<35} corr={result.pearson_corr:.6f} "
              f"scale={result.scale_ratio:.4f} max_diff={result.max_diff:.4f} [{status}]")
