"""Centralized gradient alignment utilities for JAX vs PyTorch comparison.

This module provides gradient alignment using the unified transform specification
from alphagenome_pytorch.transforms. This ensures that gradient alignment uses
the EXACT SAME transforms as weight conversion.

For transform details, see: src/alphagenome_pytorch/transforms.py
"""

from typing import Tuple
import numpy as np

from alphagenome_pytorch.jax_compat.transforms import apply_transform


def align_jax_gradient_to_pytorch(
    pt_name: str,
    jax_grad: np.ndarray,
    pt_shape: Tuple[int, ...],
) -> np.ndarray:
    """Align JAX gradient to match PyTorch parameter shape.

    This function uses the unified transform specification to ensure
    gradient alignment is consistent with weight conversion.

    Args:
        pt_name: PyTorch parameter name (used to determine transform type).
        jax_grad: JAX gradient array.
        pt_shape: Expected PyTorch parameter shape.

    Returns:
        Aligned gradient with shape matching pt_shape.

    Raises:
        ValueError: If alignment fails (no matching pattern or shape mismatch).
    """
    return apply_transform(pt_name, jax_grad, pt_shape)


def compute_gradient_metrics(
    pt_grad: np.ndarray,
    jax_grad: np.ndarray,
) -> dict:
    """Compute comparison metrics between two gradient arrays.

    Args:
        pt_grad: PyTorch gradient (reference).
        jax_grad: JAX gradient (already aligned to same shape).

    Returns:
        Dictionary with comparison metrics:
        - cosine_sim: Cosine similarity (1.0 = identical direction)
        - rel_l2: Relative L2 norm difference
        - max_abs_diff: Maximum absolute element-wise difference
        - mean_abs_diff: Mean absolute element-wise difference
        - pt_norm: L2 norm of PyTorch gradient
        - jax_norm: L2 norm of JAX gradient
    """
    pt_flat = pt_grad.flatten()
    jax_flat = jax_grad.flatten()

    # Handle NaN values
    valid_mask = ~(np.isnan(pt_flat) | np.isnan(jax_flat))
    if not valid_mask.any():
        return {
            "cosine_sim": float("nan"),
            "rel_l2": float("nan"),
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
            "pt_norm": float("nan"),
            "jax_norm": float("nan"),
        }

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
    rel_l2 = diff_norm / (pt_norm + 1e-8)

    return {
        "cosine_sim": float(cosine_sim),
        "rel_l2": float(rel_l2),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "pt_norm": float(pt_norm),
        "jax_norm": float(jax_norm),
    }
