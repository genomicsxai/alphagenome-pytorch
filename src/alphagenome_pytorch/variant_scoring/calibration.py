"""Variant-score quantile calibration.

Transforms raw variant scores into *quantile scores* by comparing each raw
per-track score against a fixed null distribution of scores computed over a
background set of common variants (348,126 MAF>0.01 chr22 gnomAD v3 SNPs).

The quantile score answers "how extreme is this variant's effect relative to
common variation?" For signed scorers it lies in roughly [-1, +1] (sign = effect
direction); for unsigned scorers in [0, 1]. Variants with |quantile| near 1 are
high-confidence functional candidates (see Avsec et al. 2026, Extended Data Fig. 5).

The null distribution itself is precomputed by DeepMind and shipped as a bundled
parquet (``data/variant_quantile_calibration_human.parquet``), converted once from
the upstream ``calibration_scores.pb`` via
``scripts/convert_calibration_to_parquet.py``. Only human calibration exists upstream.

The transform mirrors ``alphagenome_research`` commit ``dad09dd11``: for each track,
``searchsorted`` the raw score into that track's sorted quantile breakpoints and map
the resulting index to a quantile probability, breaking ties uniformly at random on
tracks with duplicated breakpoints.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep in practice
    torch = None


@dataclasses.dataclass(frozen=True)
class ScorerCalibration:
    """Per-scorer calibration table.

    Attributes:
        quantiles: ``(T, Q)`` float32 array of sorted quantile breakpoints, one row
            per calibrated track. Tracks are aligned to the *first* ``T`` output
            tracks of the scorer's head (the named human tracks).
        probabilities: ``(Q,)`` float32 array of quantile probabilities mapped to by
            each breakpoint (signed scorers span ~[-1, 1], unsigned ~[0, 1]).
        dup_mask: ``(T,)`` bool array; True where a track has duplicated breakpoints
            (requires random tie-breaking).
        is_signed: Whether this scorer's quantiles are directional.
        output_type: The scorer's output type value (e.g. ``"atac"``), for reference.
    """

    quantiles: np.ndarray
    probabilities: np.ndarray
    dup_mask: np.ndarray
    is_signed: bool
    output_type: str

    @property
    def num_tracks(self) -> int:
        return int(self.quantiles.shape[0])


class Calibration:
    """Collection of per-scorer calibration tables, keyed by scorer name.

    Keys are the upstream canonical scorer strings (e.g.
    ``"CenterMaskScorer(requested_output=ATAC, width=501, aggregation_type=DIFF_LOG2_SUM)"``),
    matching ``BaseVariantScorer.calibration_key``.
    """

    def __init__(
        self,
        scorers: dict[str, ScorerCalibration],
        *,
        seed: int = 42,
    ):
        self._scorers = scorers
        self._seed = seed

    def has_scorer(self, scorer_key: str | None) -> bool:
        """Whether calibration data exists for the given scorer key."""
        return scorer_key is not None and scorer_key in self._scorers

    def scorer_keys(self) -> list[str]:
        return list(self._scorers.keys())

    def __len__(self) -> int:
        return len(self._scorers)

    def __contains__(self, key: str) -> bool:
        return key in self._scorers

    def quantile_scores(
        self,
        scorer_key: str,
        raw_scores,
        *,
        break_quantile_ties: bool = True,
        seed: int | None = None,
    ):
        """Compute quantile scores for a single variant's per-track raw scores.

        Args:
            scorer_key: Upstream canonical scorer key (see ``has_scorer``).
            raw_scores: 1-D tensor/array of per-track raw scores, length = the
                scorer's output track count. Tracks beyond the calibrated set (and
                NaN inputs) yield NaN quantile scores.
            break_quantile_ties: When True (default, matching the API), tracks with
                duplicated quantile breakpoints have ties broken uniformly at random
                within the tied run. Set False for a fully deterministic mapping
                (uses the left edge of the run), independent of ``seed``.
            seed: Optional override for the random tie-break RNG (defaults to the
                instance seed for reproducibility). Ignored when
                ``break_quantile_ties`` is False.

        Returns:
            Quantile scores matching the input type (torch tensor or numpy array)
            and shape.
        """
        cal = self._scorers.get(scorer_key)
        if cal is None:
            raise KeyError(f"No calibration data for scorer: {scorer_key!r}")

        is_torch = torch is not None and torch.is_tensor(raw_scores)
        if is_torch:
            raw_np = raw_scores.detach().float().cpu().numpy()
        else:
            raw_np = np.asarray(raw_scores, dtype=np.float32)

        flat = raw_np.reshape(-1)
        out = np.full(flat.shape[0], np.nan, dtype=np.float32)

        rng = np.random.default_rng(self._seed if seed is None else seed)
        probs = cal.probabilities
        last = probs.shape[0] - 1
        n = min(flat.shape[0], cal.num_tracks)

        for i in range(n):
            value = flat[i]
            if np.isnan(value):
                continue
            breakpoints = cal.quantiles[i]
            idx = int(np.searchsorted(breakpoints, value, side="left"))
            if break_quantile_ties and cal.dup_mask[i]:
                end = int(np.searchsorted(breakpoints, value, side="right"))
                # Uniformly pick within the tied run [idx, end]; equals idx when no tie.
                idx = int(rng.integers(idx, end, endpoint=True))
            if idx > last:
                idx = last
            out[i] = probs[idx]

        result = out.reshape(raw_np.shape)
        if is_torch:
            return torch.from_numpy(result).to(raw_scores.device)
        return result

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 42) -> "Calibration":
        """Load calibration from a parquet file produced by the converter."""
        import pandas as pd

        df = pd.read_parquet(path)
        scorers: dict[str, ScorerCalibration] = {}
        for scorer_key, group in df.groupby("scorer_key", sort=False):
            prob_rows = group[group["row_type"] == "probabilities"]
            quant_rows = group[group["row_type"] == "quantiles"].sort_values(
                "track_index"
            )
            if prob_rows.empty or quant_rows.empty:
                raise ValueError(
                    f"Malformed calibration for scorer {scorer_key!r}: "
                    "missing probabilities or quantiles rows."
                )
            probabilities = np.asarray(
                prob_rows.iloc[0]["values"], dtype=np.float32
            )
            quantiles = np.stack(
                [np.asarray(v, dtype=np.float32) for v in quant_rows["values"]],
                axis=0,
            )
            dup_mask = np.any(np.diff(quantiles, axis=1) == 0, axis=1)
            scorers[str(scorer_key)] = ScorerCalibration(
                quantiles=quantiles,
                probabilities=probabilities,
                dup_mask=dup_mask,
                is_signed=bool(quant_rows.iloc[0]["is_signed"]),
                output_type=str(quant_rows.iloc[0]["output_type"]),
            )
        return cls(scorers, seed=seed)

    @classmethod
    def from_package(
        cls,
        organism: str = "human",
        *,
        seed: int = 42,
    ) -> "Calibration":
        """Load the calibration bundled in the package data directory.

        Only ``human`` calibration exists upstream.
        """
        filename = f"variant_quantile_calibration_{organism}.parquet"

        # Prefer importlib.resources (mirrors named_outputs.OutputMetadata loading).
        try:
            import importlib.resources as resources

            files = resources.files("alphagenome_pytorch.data")
            resource = files.joinpath(filename)
            if hasattr(resource, "is_file") and resource.is_file():
                return cls.load(str(resource), seed=seed)
        except (TypeError, AttributeError, ModuleNotFoundError):
            pass

        fallback = Path(__file__).resolve().parent.parent / "data" / filename
        if fallback.exists():
            return cls.load(fallback, seed=seed)

        raise FileNotFoundError(
            f"Bundled calibration not found: {filename}. "
            "Run scripts/convert_calibration_to_parquet.py to generate it."
        )
