"""Utility functions for AlphaGenome fine-tuning.

Re-exports DNA sequence ↔ one-hot encoding conversions from
:mod:`alphagenome_pytorch.utils.sequence`.
"""

from alphagenome_pytorch.utils.sequence import (
    onehot_to_sequence,
    sequence_to_onehot,
)

__all__ = [
    "sequence_to_onehot",
    "onehot_to_sequence",
]
