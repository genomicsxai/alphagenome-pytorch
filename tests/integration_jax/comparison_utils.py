"""Reusable comparison utilities for JAX vs PyTorch testing.

Extracted and adapted from scripts/simple_compare.py.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GradientComparisonResult:
    """Detailed gradient comparison result with multiple metrics."""

    name: str
    passed: bool
    cosine_sim: float
    rel_l2_diff: float
    max_abs_diff: float
    mean_abs_diff: float
    pt_norm: float
    jax_norm: float
    pt_shape: Tuple[int, ...]
    jax_shape: Tuple[int, ...]
    shape_match: bool
    message: str


def compare_gradients(
    name: str,
    pt_grad: np.ndarray,
    jax_grad: np.ndarray,
    rtol: float = 0.01,
    cosine_threshold: float = 0.99,
) -> GradientComparisonResult:
    """Compare gradients with multiple metrics for detailed analysis.

    Args:
        name: Parameter name for reporting.
        pt_grad: PyTorch gradient array.
        jax_grad: JAX gradient array (should already be aligned to PT shape).
        rtol: Relative tolerance for rel_l2 comparison.
        cosine_threshold: Minimum cosine similarity to pass.

    Returns:
        GradientComparisonResult with detailed comparison metrics.
    """
    shape_match = pt_grad.shape == jax_grad.shape

    if not shape_match:
        return GradientComparisonResult(
            name=name,
            passed=False,
            cosine_sim=0.0,
            rel_l2_diff=float("inf"),
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            pt_norm=float(np.linalg.norm(pt_grad)),
            jax_norm=float(np.linalg.norm(jax_grad)),
            pt_shape=pt_grad.shape,
            jax_shape=jax_grad.shape,
            shape_match=False,
            message=f"Shape mismatch: PT {pt_grad.shape} vs JAX {jax_grad.shape}",
        )

    pt_flat = pt_grad.flatten()
    jax_flat = jax_grad.flatten()

    # Handle NaN values
    valid_mask = ~(np.isnan(pt_flat) | np.isnan(jax_flat))
    if not valid_mask.any():
        return GradientComparisonResult(
            name=name,
            passed=False,
            cosine_sim=float("nan"),
            rel_l2_diff=float("nan"),
            max_abs_diff=float("nan"),
            mean_abs_diff=float("nan"),
            pt_norm=float("nan"),
            jax_norm=float("nan"),
            pt_shape=pt_grad.shape,
            jax_shape=jax_grad.shape,
            shape_match=True,
            message="No valid values (all NaN)",
        )

    pt_valid = pt_flat[valid_mask]
    jax_valid = jax_flat[valid_mask]

    # Norms
    pt_norm = np.linalg.norm(pt_valid)
    jax_norm = np.linalg.norm(jax_valid)

    # Cosine similarity
    if pt_norm > 0 and jax_norm > 0:
        cosine_sim = np.dot(pt_valid, jax_valid) / (pt_norm * jax_norm)
    else:
        cosine_sim = 1.0 if pt_norm == jax_norm == 0 else 0.0

    # Absolute differences
    abs_diff = np.abs(pt_valid - jax_valid)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())

    # Relative L2 difference
    diff_norm = np.linalg.norm(pt_valid - jax_valid)
    rel_l2_diff = diff_norm / (pt_norm + 1e-8)

    # Determine pass/fail
    passed = cosine_sim >= cosine_threshold and rel_l2_diff <= rtol

    if passed:
        message = f"PASS: cosine={cosine_sim:.4f}, rel_l2={rel_l2_diff:.4%}"
    else:
        message = f"FAIL: cosine={cosine_sim:.4f}, rel_l2={rel_l2_diff:.4%}"

    return GradientComparisonResult(
        name=name,
        passed=passed,
        cosine_sim=float(cosine_sim),
        rel_l2_diff=float(rel_l2_diff),
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        pt_norm=float(pt_norm),
        jax_norm=float(jax_norm),
        pt_shape=pt_grad.shape,
        jax_shape=jax_grad.shape,
        shape_match=True,
        message=message,
    )


def report_top_offenders(
    results: List[GradientComparisonResult],
    k: int = 10,
) -> str:
    """Generate report of top-K worst gradient mismatches.

    Args:
        results: List of gradient comparison results.
        k: Number of worst offenders to report.

    Returns:
        Formatted string report.
    """
    # Sort by cosine similarity (ascending) to get worst first
    sorted_results = sorted(results, key=lambda r: r.cosine_sim)

    lines = [f"Top {k} Gradient Mismatches (by cosine similarity):"]
    lines.append("-" * 70)

    for i, r in enumerate(sorted_results[:k]):
        lines.append(
            f"{i+1}. {r.name}\n"
            f"   cosine={r.cosine_sim:.4f}, rel_l2={r.rel_l2_diff:.4%}, "
            f"max_diff={r.max_abs_diff:.6f}\n"
            f"   PT norm={r.pt_norm:.4f}, JAX norm={r.jax_norm:.4f}"
        )

    return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Result of comparing two arrays."""

    name: str
    passed: bool
    max_diff: float
    mean_diff: float
    rel_diff_mean: float
    rel_diff_max: float
    pt_shape: Tuple[int, ...]
    jax_shape: Tuple[int, ...]
    pt_nan_count: int
    jax_nan_count: int
    valid_count: int
    message: str


def compare_arrays(
    name: str,
    pytorch_arr: np.ndarray,
    jax_arr: np.ndarray,
    rtol: float = 0.01,
    atol: float = 1e-4,
) -> ComparisonResult:
    """Compare PyTorch and JAX arrays with detailed statistics.

    Args:
        name: Name for the comparison (for reporting)
        pytorch_arr: PyTorch output array
        jax_arr: JAX output array
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        ComparisonResult with detailed comparison statistics
    """
    # Handle shape mismatch
    if pytorch_arr.shape != jax_arr.shape:
        return ComparisonResult(
            name=name,
            passed=False,
            max_diff=float("inf"),
            mean_diff=float("inf"),
            rel_diff_mean=float("inf"),
            rel_diff_max=float("inf"),
            pt_shape=pytorch_arr.shape,
            jax_shape=jax_arr.shape,
            pt_nan_count=int(np.isnan(pytorch_arr).sum()),
            jax_nan_count=int(np.isnan(jax_arr).sum()),
            valid_count=0,
            message=f"Shape mismatch: PT {pytorch_arr.shape} vs JAX {jax_arr.shape}",
        )

    # Count NaN values
    pt_nan = int(np.isnan(pytorch_arr).sum())
    jax_nan = int(np.isnan(jax_arr).sum())

    # Create valid mask (exclude NaN from both)
    valid_mask = ~(np.isnan(pytorch_arr) | np.isnan(jax_arr))
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        return ComparisonResult(
            name=name,
            passed=False,
            max_diff=float("nan"),
            mean_diff=float("nan"),
            rel_diff_mean=float("nan"),
            rel_diff_max=float("nan"),
            pt_shape=pytorch_arr.shape,
            jax_shape=jax_arr.shape,
            pt_nan_count=pt_nan,
            jax_nan_count=jax_nan,
            valid_count=0,
            message="No valid values to compare (all NaN)",
        )

    # Compute differences on valid values
    pt_valid = pytorch_arr[valid_mask]
    jax_valid = jax_arr[valid_mask]

    abs_diff = np.abs(pt_valid - jax_valid)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    # Relative difference (avoid division by zero)
    rel_diff = abs_diff / (np.abs(jax_valid) + 1e-8)
    rel_diff_mean = float(rel_diff.mean())
    rel_diff_max = float(rel_diff.max())

    # Check if within tolerance
    passed = np.allclose(pt_valid, jax_valid, rtol=rtol, atol=atol)

    if passed:
        message = f"PASS: max_diff={max_diff:.6f}, rel_diff={rel_diff_mean:.4%}"
    else:
        message = f"FAIL: max_diff={max_diff:.6f}, rel_diff={rel_diff_mean:.4%}"

    return ComparisonResult(
        name=name,
        passed=passed,
        max_diff=max_diff,
        mean_diff=mean_diff,
        rel_diff_mean=rel_diff_mean,
        rel_diff_max=rel_diff_max,
        pt_shape=pytorch_arr.shape,
        jax_shape=jax_arr.shape,
        pt_nan_count=pt_nan,
        jax_nan_count=jax_nan,
        valid_count=valid_count,
        message=message,
    )
