"""Unified transforms for JAX <-> PyTorch weight/gradient conversion.

This module provides a single source of truth for all shape transformations
needed when converting weights or aligning gradients between JAX and PyTorch.

Weight conversion (JAX -> PyTorch) transforms:
- Conv1d: (K, In, Out) -> (Out, In, K) via transpose(2, 1, 0)
- Linear: (In, Out) -> (Out, In) via transpose(1, 0)
- MultiOrganismLinear: (NumOrg, In, Out) -> same (no transform)
- Organism embed: (Dim, NumOrg) -> (NumOrg, Dim) via .T
- Norm params (3D): (1, 1, C) or (C, 1, 1) -> (C,) via squeeze
- Conv scale: (1, 1, C) -> (C, 1, 1) via transpose(2, 0, 1)
- Residual scale: scalar () -> (1,) via reshape
- Splice RoPE: (NumOrg, flat) -> (NumOrg, 2, T, H) via reshape

Gradient alignment uses the SAME transforms since gradients flow in the
same shape as weights.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple
import re
import numpy as np


class TransformType(Enum):
    """Enumeration of all transform types."""
    NONE = auto()
    CONV1D_WEIGHT = auto()       # (K, In, Out) -> (Out, In, K)
    LINEAR_WEIGHT = auto()       # (In, Out) -> (Out, In)
    LINEAR_TO_CONV1D = auto()    # (In, Out) -> (Out, In, 1) for Linear->Conv1d(k=1)
    MULTI_ORG_CONV1D = auto()    # (NumOrg, In, Out) -> (NumOrg, Out, In)
    CONV_SCALE = auto()          # (1, 1, C) -> (C, 1, 1)
    NORM_PARAM = auto()          # (1, 1, C) or (1, 1, 1, C) -> (C,)
    ORGANISM_EMBED = auto()      # (Dim, NumOrg) -> (NumOrg, Dim)
    RESIDUAL_SCALE = auto()      # () -> (1,)
    SPLICE_ROPE = auto()         # (NumOrg, flat) -> (NumOrg, 2, T, H)


@dataclass
class ParamPattern:
    """Pattern for matching parameter names to transform types.

    Attributes:
        pattern: Regex pattern to match against PyTorch parameter names.
        transform_type: The transform type to apply when pattern matches.
        condition: Optional callable that takes (name, jax_shape) and returns
            True if the transform should be applied. Used for additional
            shape-based validation.
        description: Human-readable description for debugging.
    """
    pattern: str
    transform_type: TransformType
    condition: Optional[Callable[[str, Tuple[int, ...]], bool]] = None
    description: str = ""


# Ordered list of patterns - FIRST MATCH WINS
# Order matters: more specific patterns must come before catch-all patterns
PARAM_PATTERNS: List[ParamPattern] = [
    # === OutputEmbedder project_in/project_skip: Linear -> Conv1d(k=1) ===
    # Must come before catch-all LINEAR_WEIGHT pattern
    ParamPattern(
        r"^embedder_\d+bp\.project_(in|skip)\.weight$",
        TransformType.LINEAR_TO_CONV1D,
        description="OutputEmbedder project_in/skip weights"
    ),

    # === GenomeTracksHead: MultiOrganismLinear -> MultiOrganismConv1d (NCL) ===
    ParamPattern(
        r"^heads\..+\.convs\.\d+\.weight$",
        TransformType.MULTI_ORG_CONV1D,
        lambda n, s: len(s) == 3,
        "GenomeTracksHead MultiOrganismConv1d (NCL format)"
    ),

    # === SpliceSites heads: MultiOrganismLinear -> MultiOrganismConv1d (NCL) ===
    ParamPattern(
        r"^splice_sites_\w+_head\.conv\.weight$",
        TransformType.MULTI_ORG_CONV1D,
        lambda n, s: len(s) == 3,
        "SpliceSites MultiOrganismConv1d (NCL format)"
    ),

    # === ContactMapsHead: stays as MultiOrganismLinear (NLC pair activations) ===
    ParamPattern(
        r"^contact_maps_head\.linear\.weight$",
        TransformType.NONE,
        lambda n, s: len(s) == 3,
        "ContactMaps MultiOrganismLinear (NLC pair activations)"
    ),

    # === Splice RoPE params ===
    ParamPattern(
        r"^splice_sites_junction_head\.rope_params\.(pos|neg)_(donor|acceptor)$",
        TransformType.SPLICE_ROPE,
        description="Splice junction RoPE embeddings"
    ),

    # === Organism embeddings ===
    ParamPattern(
        r".*organism_embed\.weight$",
        TransformType.ORGANISM_EMBED,
        description="Organism embedding weights"
    ),

    # === Residual scales ===
    ParamPattern(
        r".*residual_scale.*$",
        TransformType.RESIDUAL_SCALE,
        description="Residual scaling parameters"
    ),

    # === Conv scales ===
    ParamPattern(
        r".*\.conv\.scale$",
        TransformType.CONV_SCALE,
        description="StandardizedConv1d scale"
    ),

    # === Norm parameters ===
    # Covers RMSBatchNorm, LayerNorm, etc.
    ParamPattern(
        r".*norm.*\.(weight|bias|running_var)$",
        TransformType.NORM_PARAM,
        description="Normalization layer parameters"
    ),

    # === EDGE CASE: Pointwise ConvBlock uses Conv1d(kernel_size=1) ===
    # NCL refactor: proj -> conv, and now uses Conv1d instead of Linear
    # This must come BEFORE the StandardizedConv1d patterns
    ParamPattern(
        r".*\.pointwise\.conv\.weight$",
        TransformType.LINEAR_TO_CONV1D,
        description="Pointwise ConvBlock Conv1d (kernel_size=1)"
    ),

    # === StandardizedConv1d weights (3D) ===
    ParamPattern(
        r"^encoder\.dna_embedder\.(conv1|block\.conv)\.weight$",
        TransformType.CONV1D_WEIGHT,
        description="DnaEmbedder conv weights"
    ),
    ParamPattern(
        r"^encoder\.down_blocks\.\d+\.block\d\.conv\.weight$",
        TransformType.CONV1D_WEIGHT,
        description="DownResBlock conv weights"
    ),
    ParamPattern(
        r"^decoder\.up_blocks\.\d+\.(conv_in|conv_out)\.conv\.weight$",
        TransformType.CONV1D_WEIGHT,
        description="UpResBlock conv weights"
    ),

    # === Regular Linear weights (catch-all for .weight) ===
    # Only matches 2D weights to avoid misclassifying 3D weights
    ParamPattern(
        r".*\.weight$",
        TransformType.LINEAR_WEIGHT,
        lambda n, s: len(s) == 2,
        "Linear layer weights (2D)"
    ),

    # === Bias and other parameters (no transform) ===
    ParamPattern(
        r".*\.(bias|q_r_bias|k_r_bias)$",
        TransformType.NONE,
        description="Bias parameters"
    ),
]


def get_transform_for_param(
    pt_name: str,
    jax_shape: Tuple[int, ...],
) -> TransformType:
    """Determine transform type from name pattern and JAX shape.

    Args:
        pt_name: PyTorch parameter name (e.g., "tower.blocks.0.mha.q_proj.weight")
        jax_shape: Shape of the JAX parameter/gradient array

    Returns:
        TransformType enum value indicating which transform to apply

    Raises:
        ValueError: If no matching pattern found (indicates missing pattern
            that should be added to PARAM_PATTERNS)
    """
    for pattern in PARAM_PATTERNS:
        if re.match(pattern.pattern, pt_name):
            # Check additional shape condition if present
            if pattern.condition is None or pattern.condition(pt_name, jax_shape):
                return pattern.transform_type

    # No pattern matched - this is an error, not a silent fallback
    raise ValueError(
        f"No matching pattern for '{pt_name}' with shape {jax_shape}. "
        f"Add a pattern to PARAM_PATTERNS in transforms.py."
    )


def apply_transform(
    pt_name: str,
    jax_array: np.ndarray,
    pt_shape: Tuple[int, ...],
) -> np.ndarray:
    """Apply appropriate transform to convert JAX array to PyTorch shape.

    This function applies the same transforms used in weight conversion,
    making it suitable for both weight loading and gradient alignment.

    IMPORTANT: This function uses STRICT validation and will raise errors
    rather than silently producing incorrect results. There are NO fallback
    reshapes that could mask alignment errors.

    Args:
        pt_name: PyTorch parameter name (used to determine transform type)
        jax_array: JAX parameter or gradient array
        pt_shape: Expected PyTorch parameter shape

    Returns:
        Transformed array with shape matching pt_shape

    Raises:
        ValueError: If transform fails to produce correct shape, or if
            no matching pattern exists for the parameter name
    """
    # Get transform type first (even if shapes match, we may need to transform)
    transform_type = get_transform_for_param(pt_name, jax_array.shape)

    # Early return only if shapes match AND transform is NONE
    # For LINEAR_WEIGHT with square matrices, we still need to transpose
    if jax_array.shape == pt_shape and transform_type == TransformType.NONE:
        return jax_array.copy()
    result = jax_array.copy()

    if transform_type == TransformType.NONE:
        # No transform, but shapes don't match - this is an error
        pass  # Will be caught by validation below

    elif transform_type == TransformType.CONV1D_WEIGHT:
        # JAX Conv1d: (K, In, Out) -> PyTorch: (Out, In, K)
        if result.ndim != 3:
            raise ValueError(
                f"{pt_name}: CONV1D_WEIGHT expects 3D array, got {result.ndim}D"
            )
        result = result.transpose(2, 1, 0)

    elif transform_type == TransformType.LINEAR_WEIGHT:
        # JAX Linear: (In, Out) -> PyTorch: (Out, In)
        if result.ndim != 2:
            raise ValueError(
                f"{pt_name}: LINEAR_WEIGHT expects 2D array, got {result.ndim}D"
            )
        result = result.transpose(1, 0)

    elif transform_type == TransformType.LINEAR_TO_CONV1D:
        # JAX Linear: (In, Out) -> PyTorch Conv1d: (Out, In, 1)
        if result.ndim != 2:
            raise ValueError(
                f"{pt_name}: LINEAR_TO_CONV1D expects 2D array, got {result.ndim}D"
            )
        result = result.transpose(1, 0)
        result = np.expand_dims(result, axis=2)

    elif transform_type == TransformType.MULTI_ORG_CONV1D:
        # JAX MultiOrgLinear: (NumOrg, In, Out) -> PyTorch MultiOrgConv1d: (NumOrg, Out, In)
        if result.ndim != 3:
            raise ValueError(
                f"{pt_name}: MULTI_ORG_CONV1D expects 3D array, got {result.ndim}D"
            )
        result = result.transpose(0, 2, 1)  # Swap last two dims, keep first

    elif transform_type == TransformType.CONV_SCALE:
        # JAX: (1, 1, C) -> PyTorch: (C, 1, 1)
        if result.ndim == 3 and result.shape[0] == 1 and result.shape[1] == 1:
            result = result.transpose(2, 0, 1)
        elif result.ndim == 1 and len(pt_shape) == 3:
            # Handle case where JAX stores as 1D
            result = result.reshape(pt_shape)

    elif transform_type == TransformType.NORM_PARAM:
        # JAX: (1, 1, C), (C, 1, 1), or (1, 1, 1, C) -> PyTorch: (C,)
        if result.ndim > 1:
            result = result.squeeze()

    elif transform_type == TransformType.ORGANISM_EMBED:
        # JAX: (Dim, NumOrg) -> PyTorch: (NumOrg, Dim)
        # Only transpose if the shapes indicate this transform is needed
        if result.ndim == 2 and result.shape[1] == pt_shape[0]:
            result = result.T

    elif transform_type == TransformType.RESIDUAL_SCALE:
        # JAX: scalar () -> PyTorch: (1,)
        if result.ndim == 0:
            result = result.reshape((1,))

    elif transform_type == TransformType.SPLICE_ROPE:
        # JAX: (NumOrg, flattened) -> PyTorch: (NumOrg, 2, T, H)
        if result.ndim == 2 and len(pt_shape) == 4:
            result = result.reshape(pt_shape)

    # STRICT validation - fail loudly rather than silently producing garbage
    if result.shape != pt_shape:
        raise ValueError(
            f"{pt_name}: Transform {transform_type.name} failed. "
            f"JAX shape {jax_array.shape} -> result shape {result.shape}, "
            f"expected PyTorch shape {pt_shape}"
        )

    return result


def describe_transform(transform_type: TransformType) -> str:
    """Get human-readable description of a transform type."""
    descriptions = {
        TransformType.NONE: "No transform (shapes match)",
        TransformType.CONV1D_WEIGHT: "Conv1d weight: (K, In, Out) -> (Out, In, K)",
        TransformType.LINEAR_WEIGHT: "Linear weight: (In, Out) -> (Out, In)",
        TransformType.LINEAR_TO_CONV1D: "Linear to Conv1d: (In, Out) -> (Out, In, 1)",
        TransformType.MULTI_ORG_CONV1D: "MultiOrg Linear to Conv1d: (N, In, Out) -> (N, Out, In)",
        TransformType.CONV_SCALE: "Conv scale: (1, 1, C) -> (C, 1, 1)",
        TransformType.NORM_PARAM: "Norm param: squeeze to 1D",
        TransformType.ORGANISM_EMBED: "Organism embed: (Dim, NumOrg) -> (NumOrg, Dim)",
        TransformType.RESIDUAL_SCALE: "Residual scale: () -> (1,)",
        TransformType.SPLICE_ROPE: "Splice RoPE: reshape flat to 4D",
    }
    return descriptions.get(transform_type, f"Unknown: {transform_type}")
