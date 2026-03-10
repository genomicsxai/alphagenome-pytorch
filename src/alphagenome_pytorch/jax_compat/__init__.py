"""JAX/PyTorch weight and gradient conversion utilities.

This subpackage provides utilities for converting weights and aligning gradients
between JAX (AlphaGenome reference) and PyTorch implementations.

Main components:
- transforms: Shape transformations for weight/gradient conversion
- weight_mapping: Parameter name mapping between JAX and PyTorch
"""

from alphagenome_pytorch.jax_compat.transforms import (
    TransformType,
    ParamPattern,
    get_transform_for_param,
    apply_transform,
    describe_transform,
)
from alphagenome_pytorch.jax_compat.weight_mapping import map_pytorch_to_jax


__all__ = [
    # transforms
    "TransformType",
    "ParamPattern",
    "get_transform_for_param",
    "apply_transform",
    "describe_transform",
    # weight_mapping
    "map_pytorch_to_jax",
]
