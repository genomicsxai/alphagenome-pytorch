"""Shared utilities for test fixtures.

This module contains extracted utility functions to eliminate code duplication
across test fixtures in conftest.py. These utilities handle common patterns:
- JAX model setup with precision policies
- Memory management between frameworks
- Gradient flattening and conversion
- NumPy conversions
"""

import gc
from contextlib import contextmanager
from typing import Dict, Any, Callable, Tuple
import numpy as np


# =============================================================================
# JAX Model Setup Utilities
# =============================================================================

def create_jax_forward_fn(jax_model, use_float32: bool = True):
    """Create JAX forward function with standardized precision policy.

    Args:
        jax_model: JAX model instance with _metadata, _params, and _state
        use_float32: If True, use float32 precision. If False, use bfloat16.
                     Default True to match PyTorch precision and reduce noise.

    Returns:
        tuple: (transformed_forward, apply_fn) where apply_fn can be called
               with (params, state, dna_sequence, organism_index)
    """
    import haiku as hk
    import jmp
    from alphagenome_research.model import model

    # Use float32 compute to match PyTorch and eliminate precision noise
    # Original JAX uses bfloat16, which causes ~2-3% differences
    if use_float32:
        jmp_policy = jmp.get_policy('params=float32,compute=float32,output=float32')
    else:
        jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=float32')

    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy):
            return model.AlphaGenome(jax_model._metadata)(dna_sequence, organism_index)

    def _apply_fn(params, state, dna_sequence, organism_index):
        (predictions, _), _ = _forward.apply(
            params, state, None, dna_sequence, organism_index
        )
        return predictions

    return _forward, _apply_fn


def create_jax_loss_fn(jax_model, loss_type: str = 'combined', use_float32: bool = True):
    """Create JAX loss function for gradient computation.

    Args:
        jax_model: JAX model instance
        loss_type: Type of loss - 'combined', 'per_head', or 'training'
        use_float32: Use float32 precision (default True)

    Returns:
        Callable loss function that takes (params, state, dna_sequence, organism_index)
        and returns loss value
    """
    import jax.numpy as jnp
    import haiku as hk
    import jmp
    from alphagenome_research.model import model

    if use_float32:
        jmp_policy = jmp.get_policy('params=float32,compute=float32,output=float32')
    else:
        jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=float32')

    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy):
            return model.AlphaGenome(jax_model._metadata)(dna_sequence, organism_index)

    if loss_type == 'combined':
        def _loss_fn(params, state, dna_sequence, organism_index):
            """Compute combined loss (sum of all head outputs)."""
            (predictions, _), _ = _forward.apply(
                params, state, None, dna_sequence, organism_index
            )
            loss = jnp.array(0.0)

            # Genomic tracks
            for head_name in ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']:
                if head_name in predictions:
                    head_out = predictions[head_name]
                    for key, val in head_out.items():
                        # Only sum final predictions (exclude scaled_predictions_*)
                        if key.startswith('predictions_'):
                            loss = loss + jnp.nanmean(val)

            # Contact maps
            if 'contact_maps' in predictions:
                cm_out = predictions['contact_maps']
                if isinstance(cm_out, dict) and 'predictions' in cm_out:
                    loss = loss + cm_out['predictions'].mean()
            elif 'pair_activations' in predictions:
                loss = loss + predictions['pair_activations'].mean()

            # Splice heads
            if 'splice_sites_classification' in predictions:
                if 'logits' in predictions['splice_sites_classification']:
                    loss = loss + predictions['splice_sites_classification']['logits'].mean()
            if 'splice_sites_usage' in predictions:
                if 'logits' in predictions['splice_sites_usage']:
                    loss = loss + predictions['splice_sites_usage']['logits'].mean()

            return loss

    elif loss_type == 'training':
        def _loss_fn(params, state, dna_sequence, organism_index, targets):
            """Compute training loss using scaled predictions and targets."""
            (predictions, _), _ = _forward.apply(
                params, state, None, dna_sequence, organism_index
            )
            from alphagenome_research.losses import poisson_multinomial_loss

            loss = jnp.array(0.0)

            # Use scaled predictions for training
            for head_name in ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']:
                if head_name in predictions:
                    head_out = predictions[head_name]
                    for key, pred in head_out.items():
                        if key.startswith('scaled_predictions_'):
                            res_str = key.replace('scaled_predictions_', '').replace('bp', '')
                            res = int(res_str)
                            if head_name in targets and res in targets[head_name]:
                                target = targets[head_name][res]
                                head_loss = poisson_multinomial_loss(
                                    target, pred, aggregate_tracks=True, aggregate_positions=True
                                )
                                loss = loss + head_loss

            return loss

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return _loss_fn


# =============================================================================
# Memory Management
# =============================================================================

@contextmanager
def clear_gpu_memory_between_frameworks():
    """Context manager to clear GPU memory when switching between PyTorch and JAX.

    This is critical to avoid OOM errors when both frameworks try to allocate
    memory simultaneously. Use this after PyTorch operations and before JAX operations.

    Example:
        # Run PyTorch
        pt_output = run_pytorch_model()

        # Clear memory before JAX
        with clear_gpu_memory_between_frameworks():
            pass

        # Run JAX
        jax_output = run_jax_model()
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    yield

    # Cleanup after context
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_gpu_memory():
    """Clear GPU memory (PyTorch) and run garbage collection.

    Call this explicitly when you need to free GPU memory between operations.
    For switching between frameworks, prefer clear_gpu_memory_between_frameworks().
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# =============================================================================
# Gradient Utilities
# =============================================================================

def flatten_jax_gradients(grads: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Flatten nested JAX gradient dictionary to flat dict with path names.

    Converts JAX's nested gradient tree structure to a flat dictionary where
    keys are parameter paths (e.g., "encoder/conv1/w") and values are numpy arrays.

    Args:
        grads: Nested dictionary of JAX gradients (from jax.grad)

    Returns:
        Flat dictionary mapping parameter paths to gradient arrays

    Example:
        >>> nested = {"encoder": {"conv1": {"w": jnp.array([1,2,3])}}}
        >>> flat = flatten_jax_gradients(nested)
        >>> flat
        {"encoder/conv1/w": array([1,2,3])}
    """
    flat_grads = {}

    def _flatten(d, prefix=''):
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                # Convert to numpy float32 for compatibility
                flat_grads[key] = np.array(v).astype(np.float32)

    _flatten(grads)
    return flat_grads


# =============================================================================
# Conversion Utilities
# =============================================================================

def jax_to_numpy(x):
    """Convert JAX array to numpy float32, handling bfloat16.

    Args:
        x: JAX array or pytree of JAX arrays

    Returns:
        Numpy array(s) in float32
    """
    if x is None:
        return None

    arr = np.array(x)

    # Handle bfloat16 (JAX's default precision)
    dtype_str = str(arr.dtype)
    if "bfloat16" in dtype_str:
        arr = arr.astype(np.float32)

    return arr.astype(np.float32)


def pytorch_to_numpy(x):
    """Convert PyTorch tensor to numpy float32.

    Args:
        x: PyTorch tensor, dict of tensors, or None

    Returns:
        Numpy array(s) in float32
    """
    import torch

    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy().astype(np.float32)
    elif isinstance(x, dict):
        return {k: pytorch_to_numpy(v) for k, v in x.items()}
    return x


# =============================================================================
# Organism Iteration Helpers
# =============================================================================

def iterate_organisms():
    """Generator for organism names and indices.

    Yields:
        tuple: (organism_name, organism_index)

    Example:
        >>> for org_name, org_idx in iterate_organisms():
        ...     print(f"Processing {org_name} (index {org_idx})")
        Processing human (index 0)
        Processing mouse (index 1)
    """
    yield ("human", 0)
    yield ("mouse", 1)


# =============================================================================
# Caching Patterns
# =============================================================================

def cache_both_frameworks(
    pytorch_fn: Callable,
    jax_fn: Callable,
    organisms: list = None
) -> Dict[str, Dict[str, Any]]:
    """Run both PyTorch and JAX functions for all organisms with proper memory management.

    This implements the standard caching pattern:
    1. Run PyTorch for all organisms
    2. Clear GPU memory
    3. Run JAX for all organisms

    Args:
        pytorch_fn: Function that takes (organism_index) and returns results
        jax_fn: Function that takes (organism_index) and returns results
        organisms: List of (name, index) tuples. Defaults to [(human, 0), (mouse, 1)]

    Returns:
        Dict with structure: {
            "human": {"pytorch": ..., "jax": ...},
            "mouse": {"pytorch": ..., "jax": ...},
        }

    Example:
        >>> cache = cache_both_frameworks(
        ...     lambda org_idx: run_pytorch_model(input, org_idx),
        ...     lambda org_idx: run_jax_model(input, org_idx)
        ... )
    """
    if organisms is None:
        organisms = list(iterate_organisms())

    cache = {}

    # Run PyTorch first for all organisms
    for org_name, org_idx in organisms:
        cache[org_name] = {"pytorch": pytorch_fn(org_idx)}

    # Clear GPU memory before JAX
    clear_gpu_memory()

    # Run JAX for all organisms
    for org_name, org_idx in organisms:
        cache[org_name]["jax"] = jax_fn(org_idx)

    return cache
