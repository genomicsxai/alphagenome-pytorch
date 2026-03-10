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
) -> Path:
    """Save API scores to cache.

    Args:
        api_scores: List of AnnData objects from API
        cache_dir: Directory to save cache. If None, uses default.
        variant_str: Variant string for metadata

    Returns:
        Path to saved pickle file
    """
    if cache_dir is None:
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

        # Save to cache for future use
        variant_str = f"{DEFAULT_VARIANT_CHROMOSOME}:{DEFAULT_VARIANT_POSITION}:{DEFAULT_VARIANT_REFERENCE_BASES}>{DEFAULT_VARIANT_ALTERNATE_BASES}"
        save_path = save_api_scores(api_scores, variant_str=variant_str)
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
