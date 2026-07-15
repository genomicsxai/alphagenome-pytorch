"""Shared utilities for API-based variant scoring tests.

This module provides utilities for fetching variant scores from the
AlphaGenome API, caching results, and comparing PyTorch vs API outputs.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import anndata

# No default API key - must be set via environment variable

# Test variant configuration
DEFAULT_VARIANT_CHROMOSOME = "chr22"
DEFAULT_VARIANT_POSITION = 36201698
DEFAULT_VARIANT_REFERENCE_BASES = "A"
DEFAULT_VARIANT_ALTERNATE_BASES = "C"
DEFAULT_SEQUENCE_LENGTH = "100KB"  # 131072bp


def get_api_key() -> str | None:
    """Get AlphaGenome API key from environment.

    Returns:
        API key string, or None if ALPHAGENOME_API_KEY not set
    """
    return os.environ.get("ALPHAGENOME_API_KEY")


def get_cache_path(cache_dir: Path | str | None = None) -> Path:
    """Get path to API cache file.

    Args:
        cache_dir: Optional cache directory. If None, uses default locations.

    Returns:
        Path to cache pickle file
    """
    if cache_dir:
        return Path(cache_dir) / "variant_scores.pkl"

    # Check common locations
    possible_paths = [
        Path("data/api_cache/variant_scores.pkl"),
        Path(__file__).parent.parent.parent / "data" / "api_cache" / "variant_scores.pkl",
    ]
    for p in possible_paths:
        if p.exists():
            return p.absolute()

    # Default to first path for writing
    return possible_paths[0].absolute()


def load_cached_scores(cache_path: Path | str | None = None) -> list['anndata.AnnData'] | None:
    """Load cached API scores if available.

    Args:
        cache_path: Path to cache file. If None, searches default locations.

    Returns:
        List of AnnData objects, or None if cache not found
    """
    if cache_path is None:
        cache_path = get_cache_path()
    else:
        cache_path = Path(cache_path)

    if not cache_path.exists():
        return None

    with open(cache_path, "rb") as f:
        return pickle.load(f)


def fetch_api_scores(
    variant_chromosome: str = DEFAULT_VARIANT_CHROMOSOME,
    variant_position: int = DEFAULT_VARIANT_POSITION,
    variant_reference_bases: str = DEFAULT_VARIANT_REFERENCE_BASES,
    variant_alternate_bases: str = DEFAULT_VARIANT_ALTERNATE_BASES,
    sequence_length: str = DEFAULT_SEQUENCE_LENGTH,
    api_key: str | None = None,
) -> list['anndata.AnnData']:
    """Fetch variant scores from the AlphaGenome API.

    Args:
        variant_chromosome: Chromosome of the variant
        variant_position: 1-based position of the variant
        variant_reference_bases: Reference allele
        variant_alternate_bases: Alternate allele
        sequence_length: Sequence length key (e.g., "100KB")
        api_key: API key. If None, uses get_api_key()

    Returns:
        List of AnnData objects from API

    Raises:
        ImportError: If alphagenome package is not installed
        Exception: If API call fails
    """
    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers

    if api_key is None:
        api_key = get_api_key()

    if not api_key:
        raise ValueError("No API key available")

    # Create API client
    client = dna_client.create(api_key=api_key)

    # Create variant and interval
    variant = genome.Variant(
        chromosome=variant_chromosome,
        position=variant_position,
        reference_bases=variant_reference_bases,
        alternate_bases=variant_alternate_bases,
    )

    sequence_length_value = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f"SEQUENCE_LENGTH_{sequence_length}"
    ]
    interval = variant.reference_interval.resize(sequence_length_value)

    # Get recommended scorers
    recommended_scorers = list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.values())

    # Fetch scores
    api_scores = client.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=recommended_scorers,
    )

    return api_scores


def save_api_scores(
    api_scores: list['anndata.AnnData'],
    cache_dir: Path | str | None = None,
    variant_str: str | None = None,
    cache_path: Path | str | None = None,
) -> Path:
    """Save API scores to cache.

    Args:
        api_scores: List of AnnData objects from API
        cache_dir: Directory to save cache. If None, uses default.
        variant_str: Variant string for metadata
        cache_path: Exact file to write. Takes precedence over ``cache_dir``.
            Callers that read from a specific file must write back to that same
            file — deriving a directory from it is not enough, since the
            ``cache_dir`` form would rename it to ``variant_scores.pkl`` and the
            next read would miss.

    Returns:
        Path to saved pickle file
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_dir = cache_path.parent
    elif cache_dir is None:
        cache_path = get_cache_path()
        cache_dir = cache_path.parent
    else:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / "variant_scores.pkl"

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save pickle
    with open(cache_path, "wb") as f:
        pickle.dump(api_scores, f)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "variant": variant_str or "unknown",
        "num_scorers": len(api_scores),
    }
    metadata_path = cache_dir / "api_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return cache_path


def get_or_fetch_api_scores(
    cache_path: Path | str | None = None,
    force_refresh: bool = False,
    verbose: bool = True,
) -> list['anndata.AnnData']:
    """Get API scores from cache, or fetch from API if not cached.

    This is the main entry point for tests. It:
    1. Tries to load from cache
    2. If cache doesn't exist or force_refresh=True, fetches from API
    3. Saves fetched results to cache for future use

    Args:
        cache_path: Path to cache file. If None, uses default locations.
        force_refresh: If True, fetch from API even if cache exists.
        verbose: Print status messages

    Returns:
        List of AnnData objects

    Raises:
        RuntimeError: If neither cache nor API is available
    """
    # Try to load from cache first
    if not force_refresh:
        cached = load_cached_scores(cache_path)
        if cached is not None:
            if verbose:
                print(f"Loaded {len(cached)} scorer results from cache")
            return cached

    # No cache, try API
    if verbose:
        print("Cache not found, fetching from API...")

    try:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("No API key available")

        api_scores = fetch_api_scores(api_key=api_key)

        # Save to cache for future use — back to the file we read from, or the
        # caller misses and refetches from the API on every single call.
        variant_str = f"{DEFAULT_VARIANT_CHROMOSOME}:{DEFAULT_VARIANT_POSITION}:{DEFAULT_VARIANT_REFERENCE_BASES}>{DEFAULT_VARIANT_ALTERNATE_BASES}"
        save_path = save_api_scores(api_scores, variant_str=variant_str, cache_path=cache_path)
        if verbose:
            print(f"Fetched {len(api_scores)} scorer results from API")
            print(f"Saved to cache: {save_path}")

        return api_scores

    except ImportError as e:
        raise RuntimeError(
            f"Cannot load cache and alphagenome package not installed: {e}\n"
            "Either provide cached results or install alphagenome package."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Cannot load cache and API call failed: {e}\n"
            "Check your API key and network connection."
        ) from e


# =============================================================================
# Raw prediction parity (localizing diagnostic: do PREDICTIONS diverge from the
# API, upstream of scoring?). Full predictions are huge, so we cache a center
# window around the variant per output type.
# =============================================================================

# Output types compared for prediction parity (the CenterMask track outputs).
PREDICTION_OUTPUT_TYPES = (
    "ATAC",
    "DNASE",
    "CAGE",
    "PROCAP",
    "CHIP_TF",
    "CHIP_HISTONE",
)

# Width (bp) of the center window kept per output type. Large enough to cover
# the variant's neighborhood; small enough to cache. Windowed per resolution.
DEFAULT_PREDICTION_WINDOW_BP = 8192


def get_predictions_cache_path(cache_dir: Path | str | None = None) -> Path:
    """Path to the cached API predictions pickle (mirrors get_cache_path)."""
    if cache_dir:
        return Path(cache_dir) / "variant_predictions.pkl"
    possible_paths = [
        Path("data/api_cache/variant_predictions.pkl"),
        Path(__file__).parent.parent.parent
        / "data"
        / "api_cache"
        / "variant_predictions.pkl",
    ]
    for p in possible_paths:
        if p.exists():
            return p.absolute()
    return possible_paths[0].absolute()


def window_trackdata(ref_td, alt_td, variant, window_bp: int) -> dict:
    """Slice a center window (around the variant) out of ref/alt TrackData.

    Returns a plain, picklable dict with per-track names/strands, the windowed
    ref (and alt) values ``(window_bins, num_tracks)``, the resolution, and the
    slice bounds so a local model's prediction can be windowed identically.
    """
    resolution = int(ref_td.resolution)
    ref_values = np.asarray(ref_td.values)
    n_bins = ref_values.shape[0]

    # 0-based variant position relative to the interval, in bins. Fall back to
    # the interval centre (resize() centres the interval on the variant).
    interval = getattr(ref_td, "interval", None)
    if interval is not None:
        center_bin = (variant.start - interval.start) // resolution
    else:
        center_bin = n_bins // 2

    half = max(1, (window_bp // resolution) // 2)
    start = int(max(0, center_bin - half))
    end = int(min(n_bins, center_bin + half))

    alt_values = np.asarray(alt_td.values) if alt_td is not None else None
    return {
        "resolution": resolution,
        "interval_start": int(interval.start) if interval is not None else None,
        "start_bin": start,
        "end_bin": end,
        "names": [str(n) for n in ref_td.metadata["name"].tolist()],
        "strands": [str(s) for s in ref_td.metadata["strand"].tolist()],
        "ref": ref_values[start:end].astype(np.float32),
        "alt": (
            alt_values[start:end].astype(np.float32)
            if alt_values is not None
            else None
        ),
    }


def fetch_api_predictions(
    variant_chromosome: str = DEFAULT_VARIANT_CHROMOSOME,
    variant_position: int = DEFAULT_VARIANT_POSITION,
    variant_reference_bases: str = DEFAULT_VARIANT_REFERENCE_BASES,
    variant_alternate_bases: str = DEFAULT_VARIANT_ALTERNATE_BASES,
    sequence_length: str = DEFAULT_SEQUENCE_LENGTH,
    window_bp: int = DEFAULT_PREDICTION_WINDOW_BP,
    api_key: str | None = None,
) -> dict:
    """Fetch raw ref/alt predictions from the API, windowed around the variant.

    Returns ``{output_type_name: window_trackdata(...) dict}`` for the
    ``PREDICTION_OUTPUT_TYPES`` (uses the same variant/interval as the scores
    cache).
    """
    from alphagenome.data import genome
    from alphagenome.models import dna_client, dna_output

    if api_key is None:
        api_key = get_api_key()
    if not api_key:
        raise ValueError("No API key available")

    client = dna_client.create(api_key=api_key)

    variant = genome.Variant(
        chromosome=variant_chromosome,
        position=variant_position,
        reference_bases=variant_reference_bases,
        alternate_bases=variant_alternate_bases,
    )
    sequence_length_value = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f"SEQUENCE_LENGTH_{sequence_length}"
    ]
    interval = variant.reference_interval.resize(sequence_length_value)

    requested = [getattr(dna_output.OutputType, n) for n in PREDICTION_OUTPUT_TYPES]
    variant_output = client.predict_variant(
        interval=interval,
        variant=variant,
        requested_outputs=requested,
        ontology_terms=None,
    )

    result: dict = {}
    for output_type in requested:
        ref_td = variant_output.reference.get(output_type)
        alt_td = variant_output.alternate.get(output_type)
        if ref_td is None:
            continue
        result[output_type.name] = window_trackdata(ref_td, alt_td, variant, window_bp)
    return result


def save_api_predictions(
    predictions: dict,
    cache_dir: Path | str | None = None,
    variant_str: str | None = None,
    cache_path: Path | str | None = None,
) -> Path:
    """Save windowed API predictions to cache (mirrors save_api_scores).

    Args:
        predictions: Windowed predictions keyed by output type.
        cache_dir: Directory to write into; the filename is chosen for you.
        variant_str: Variant string for the metadata sidecar.
        cache_path: Exact file to write. Takes precedence over ``cache_dir``.
            Callers that read from a specific file must write back to that same
            file — deriving a directory from it is not enough, since the
            ``cache_dir`` form would rename it to ``variant_predictions.pkl``
            and the next read would miss.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_dir = cache_path.parent
    elif cache_dir is None:
        cache_path = get_predictions_cache_path()
        cache_dir = cache_path.parent
    else:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / "variant_predictions.pkl"

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(predictions, f)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "variant": variant_str or "unknown",
        "output_types": list(predictions.keys()),
        "window_bp": DEFAULT_PREDICTION_WINDOW_BP,
    }
    with open(cache_dir / "api_predictions_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return cache_path


def get_or_fetch_api_predictions(
    cache_path: Path | str | None = None,
    force_refresh: bool = False,
    verbose: bool = True,
) -> dict:
    """Get windowed API predictions from cache, or fetch and cache them.

    Mirrors get_or_fetch_api_scores but for raw predictions.
    """
    if not force_refresh:
        path = Path(cache_path) if cache_path else get_predictions_cache_path()
        if path.exists():
            with open(path, "rb") as f:
                cached = pickle.load(f)
            if verbose:
                print(f"Loaded predictions for {len(cached)} output types from cache")
            return cached

    if verbose:
        print("Prediction cache not found, fetching from API...")
    try:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("No API key available")
        predictions = fetch_api_predictions(api_key=api_key)
        variant_str = (
            f"{DEFAULT_VARIANT_CHROMOSOME}:{DEFAULT_VARIANT_POSITION}:"
            f"{DEFAULT_VARIANT_REFERENCE_BASES}>{DEFAULT_VARIANT_ALTERNATE_BASES}"
        )
        # Write back to the file we read from, or the caller misses and refetches
        # from the API on every single call.
        save_path = save_api_predictions(
            predictions, variant_str=variant_str, cache_path=cache_path
        )
        if verbose:
            print(f"Fetched predictions for {len(predictions)} output types from API")
            print(f"Saved to cache: {save_path}")
        return predictions
    except ImportError as e:
        raise RuntimeError(
            f"Cannot load cache and alphagenome package not installed: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Cannot load prediction cache and API call failed: {e}\n"
            "Check your API key and network connection."
        ) from e
