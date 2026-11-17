"""Strict, model-bound organism normalization.

Public, strict counterpart to :func:`alphagenome_pytorch.named_outputs._resolve_organism_index`
(which silently maps unknown input to a default). Values are validated against the target model's
``num_organisms`` so a malformed organism fails loudly instead of defaulting to human.

Intended to be the single shared home for organism string/int normalization; finetuning,
serving, prediction, and variant scoring can converge on it over time.
"""
from __future__ import annotations

import operator
from collections.abc import Iterable

__all__ = ["ORGANISM_ALIASES", "normalize_organism_index", "normalize_organism_indices"]


# Canonical organism name/alias -> backbone slot index. Index is a model-local slot,
# not a biological identifier; a future registry may add taxonomy ids alongside these.
ORGANISM_ALIASES: dict[str, int] = {
    "human": 0,
    "homo_sapiens": 0,
    "mouse": 1,
    "mus_musculus": 1,
}


def normalize_organism_index(value, *, num_organisms: int) -> int | None:
    """Normalize a single organism value to a bounded index, or ``None`` when missing.

    Accepts ``None`` (missing), an alias string (``"human"``/``"mouse"``/...), a decimal
    string (``"0"``/``"1"`` as produced by CSV/TSV metadata), or an integer-like value
    (including numpy ints via ``operator.index``). Rejects ``bool`` (``True``/``False`` are
    ``int`` subclasses but must not become organisms 1/0), unknown strings, non-integers,
    and any index outside ``[0, num_organisms)``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"organism must not be a bool, got {value!r}")

    if isinstance(value, str):
        text = value.strip().lower()
        if text in ORGANISM_ALIASES:
            index = ORGANISM_ALIASES[text]
        elif text.isdecimal():
            index = int(text)
        else:
            raise ValueError(
                f"Unknown organism {value!r}; expected one of {sorted(ORGANISM_ALIASES)} "
                "or an integer index"
            )
    else:
        try:
            index = operator.index(value)
        except TypeError:
            raise ValueError(
                f"organism must be an int or known string, got {type(value).__name__}"
            ) from None

    if not 0 <= index < num_organisms:
        raise ValueError(
            f"organism index {index} out of range for num_organisms={num_organisms} "
            f"(valid: 0..{num_organisms - 1})"
        )
    return index


def normalize_organism_indices(value, *, num_organisms: int) -> tuple[int, ...] | None:
    """Normalize a scalar or collection of organisms to a sorted, unique tuple.

    ``None`` -> ``None`` (missing information). A string is treated as a scalar. A
    ``list``/``tuple``/``set`` (or other non-string iterable) is plural: each member is
    normalized, deduplicated, and sorted. An **empty** collection is invalid — use ``None``
    for "missing"; an empty declared set is almost certainly corrupt metadata.
    """
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Iterable):
        return (normalize_organism_index(value, num_organisms=num_organisms),)

    members = list(value)
    if not members:
        raise ValueError("organism_indices must contain at least one organism")

    normalized: set[int] = set()
    for member in members:
        index = normalize_organism_index(member, num_organisms=num_organisms)
        if index is None:
            raise ValueError("organism_indices members must not be None")
        normalized.add(index)
    return tuple(sorted(normalized))
