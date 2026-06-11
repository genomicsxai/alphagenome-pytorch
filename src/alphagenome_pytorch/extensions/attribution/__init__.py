"""Nucleotide attribution methods for AlphaGenome models.

Pure-Torch attribution primitives (gradient x input, saturation ISM) and a
small method registry for wiring them into serving / scripts. No dependency
on variant scoring, FASTA, or REST.

Example:
    >>> from alphagenome_pytorch.extensions.attribution import (
    ...     gradient_x_input, saturation_ism, default_head_selector,
    ... )
    >>> scores = gradient_x_input(
    ...     model, onehot=onehot, organism_index=0,
    ...     head_selector=default_head_selector,
    ...     output_type='dnase', resolution=1,
    ...     target_slice=slice(1000, 2000),
    ...     track_indices=[0, 5, 17], reduction='sum',
    ... )
"""

from .gradient import gradient_x_input
from .heads import HeadSelector, default_head_selector
from .ism import saturation_ism
from .registry import METHODS, MethodSpec, UnsupportedMethodError, get_method
from .types import AttributionResult
from .window import reduce_window, target_slice_for_resolution

__all__ = [
    "AttributionResult",
    "HeadSelector",
    "METHODS",
    "MethodSpec",
    "UnsupportedMethodError",
    "default_head_selector",
    "get_method",
    "gradient_x_input",
    "reduce_window",
    "saturation_ism",
    "target_slice_for_resolution",
]
