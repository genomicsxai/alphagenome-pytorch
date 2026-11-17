"""BigWig depth normalization shared by preprocessing and fine-tuning."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

# Limit per-base reads to one million bases at a time.
WINDOW = 1_000_000


def parse_target(s: str) -> float:
    """Parse human-readable target strings like '100M', '50k' to numbers."""
    text = s.strip()
    suffixes = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9, "b": 1e9}
    match = re.fullmatch(r"([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*([a-z]?)", text, re.IGNORECASE)
    if match is None or match[2].lower() not in suffixes:
        raise ValueError(f"Invalid target {s!r}. Use a number such as 100M or 50k.")
    target = float(match[1]) * suffixes[match[2].lower()]
    if not math.isfinite(target):
        raise ValueError("Target must be finite.")
    return target


def iter_bigwig_windows(bw, chrom: str, size: int):
    """Yield nonempty (offset, values) windows. Gaps are NaN."""
    for start in range(0, size, WINDOW):
        end = min(start + WINDOW, size)
        values = bw.values(chrom, start, end, numpy=True)
        if values is not None and len(values):
            yield start, values


def bigwig_total_signal(
    path: str,
    *,
    absolute: bool = False,
    require_nonnegative: bool = False,
) -> float:
    """Sum stored signal across all bases, with gaps contributing zero.

    Paired offline scaling uses magnitudes to support signed tracks used for
    plotting. Fine-tuning checks the sign in the same pass as the sum.
    """
    import numpy as np
    import pyBigWig

    total = 0.0
    with pyBigWig.open(path) as bw:
        if require_nonnegative and bw.header()["nBasesCovered"] == 0:
            raise ValueError(f"{path} contains no signal intervals.")
        for chrom, size in bw.chroms().items():
            for _, values in iter_bigwig_windows(bw, chrom, size):
                if require_nonnegative and np.any(values < 0):
                    raise ValueError(
                        f"{path} contains negative targets. If a minus-strand "
                        "track stores negative coverage, flip its sign before fine-tuning."
                    )
                signal = np.abs(values) if absolute else values
                total += float(np.nansum(signal, dtype=np.float64))
    if not math.isfinite(total):
        raise ValueError(f"{path} has non-finite total signal.")
    return total


def track_scale_factors(
    totals: Sequence[float],
    target_total: float,
    pair_groups: Sequence[tuple[int, int]] | None = None,
) -> list[float]:
    """Scale each singleton or pair to the target using one factor per group."""
    groups = []
    paired: set[int] = set()
    for pair in pair_groups or ():
        plus, minus = pair
        if (plus == minus or min(pair) < 0 or max(pair) >= len(totals)
                or not paired.isdisjoint(pair)):
            raise ValueError(f"Invalid strand pair {pair}: expected distinct, unused track indices.")
        paired.update(pair)
        groups.append(pair)
    groups.extend((i,) for i in range(len(totals)) if i not in paired)

    factors = [1.0] * len(totals)
    for group in groups:
        group_total = sum(totals[i] for i in group)
        if group_total <= 0:
            raise ValueError(f"Tracks {group} have zero total signal and cannot be depth-normalized.")
        for i in group:
            factors[i] = target_total / group_total
    return factors
