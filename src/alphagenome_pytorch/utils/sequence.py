"""DNA sequence ↔ one-hot encoding conversions.

Provides canonical numpy implementations (``sequence_to_onehot``,
``onehot_to_sequence``) and thin torch wrappers (``sequence_to_onehot_tensor``,
``onehot_tensor_to_sequence``) for use throughout the package.

Encoding mapping::

    A → [1, 0, 0, 0]
    C → [0, 1, 0, 0]
    G → [0, 0, 1, 0]
    T → [0, 0, 0, 1]
    N / other → [0, 0, 0, 0]   (all-zeros, matching JAX reference)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# A=0, C=1, G=2, T=3
_BASES = "ACGT"

# Build lookup table once (128 entries covers ASCII)
_ENCODE_LOOKUP = np.full(128, -1, dtype=np.int8)
for _i, _ch in enumerate(_BASES):
    _ENCODE_LOOKUP[ord(_ch)] = _i
    _ENCODE_LOOKUP[ord(_ch.lower())] = _i


def sequence_to_onehot(sequence: str) -> np.ndarray:
    """Convert a DNA sequence string to a one-hot encoded numpy array.

    Handles both upper- and lower-case nucleotides.
    Ambiguous / unknown bases (e.g. ``N``) are encoded as all-zeros.

    Args:
        sequence: DNA sequence string (``ACGTN``).

    Returns:
        One-hot encoded ``uint8`` array of shape ``(len(sequence), 4)``.
    """
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    onehot = np.zeros((len(seq_bytes), 4), dtype=np.uint8)
    # clip(0, 127) prevents crash on non-ASCII if present
    indices = _ENCODE_LOOKUP[seq_bytes.clip(0, 127)]
    mask = indices >= 0
    onehot[np.where(mask)[0], indices[mask]] = 1
    return onehot


def onehot_to_sequence(onehot: np.ndarray) -> str:
    """Convert a one-hot encoded array back to a DNA sequence string.

    All-zero rows (ambiguous bases) are decoded as ``N``.

    Args:
        onehot: Array of shape ``(L, 4)`` with one-hot encoding.

    Returns:
        DNA sequence string of length ``L``.
    """
    if onehot.ndim != 2 or onehot.shape[1] != 4:
        raise ValueError(f"Expected shape (L, 4), got {onehot.shape}")

    bases = np.array(list(_BASES + "N"))  # index 4 → 'N'
    # All-zero rows → argmax returns 0, but we want 'N'
    is_valid = onehot.any(axis=1)
    indices = onehot.argmax(axis=1)
    indices = np.where(is_valid, indices, 4)
    return "".join(bases[indices])


# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------


def sequence_to_onehot_tensor(
    sequence: str,
    dtype: "torch.dtype | None" = None,
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """Convert DNA sequence string to a one-hot encoded torch tensor.

    Thin wrapper around :func:`sequence_to_onehot` that converts the result
    to a :class:`torch.Tensor` with the requested dtype and device.

    Args:
        sequence: DNA sequence string (``ACGTN``).
        dtype: Output tensor dtype. Defaults to ``torch.float32``.
        device: Output tensor device.

    Returns:
        One-hot encoded tensor of shape ``(len(sequence), 4)``.
    """
    import torch as _torch

    np_onehot = sequence_to_onehot(sequence)
    tensor = _torch.from_numpy(np_onehot.astype(np.float32))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def onehot_tensor_to_sequence(onehot: "torch.Tensor") -> str:
    """Convert a one-hot encoded torch tensor back to a DNA sequence string.

    Accepts tensors of shape ``(L, 4)`` or ``(B, L, 4)`` (takes first batch
    element).

    Args:
        onehot: One-hot tensor of shape ``(L, 4)`` or ``(B, L, 4)``.

    Returns:
        DNA sequence string of length ``L``.
    """
    if onehot.dim() == 3:
        onehot = onehot[0]
    return onehot_to_sequence(onehot.detach().cpu().numpy())


__all__ = [
    "sequence_to_onehot",
    "onehot_to_sequence",
    "sequence_to_onehot_tensor",
    "onehot_tensor_to_sequence",
]
