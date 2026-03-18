"""Metadata-aware wrappers for AlphaGenome outputs.

This module adds opt-in wrappers around model output dictionaries so users can
filter tracks by metadata while keeping raw tensors fully torch-compatible.
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch


_ORGANISM_ALIASES = {
    "human": 0,
    "homo_sapiens": 0,
    "mouse": 1,
    "mus_musculus": 1,
}

_OUTPUT_ALIASES = {
    "contact_maps": "pair_activations",
    "splice_sites": "splice_sites_classification",
    "splice_site_usage": "splice_sites_usage",
    "splice_junctions": "splice_sites_junction",
}


def _normalize_output_name(output_name: str) -> str:
    key = output_name.strip().lower()
    return _OUTPUT_ALIASES.get(key, key)


def _resolve_organism_index(value: int | str | torch.Tensor | None, default: int = 0) -> int:
    if value is None:
        return default

    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        flat = value.detach().view(-1)
        first = int(flat[0].item())
        # Named outputs require a single metadata catalog per call.
        # If a mixed-organism batch is passed, use the first element.
        return first

    if isinstance(value, int):
        return value

    text = str(value).strip().lower()
    if text in _ORGANISM_ALIASES:
        return _ORGANISM_ALIASES[text]
    if text.isdigit():
        return int(text)
    return default


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _placeholder_tracks(
    output_name: str,
    organism: int,
    count: int,
    start_index: int = 0,
) -> tuple["TrackMetadata", ...]:
    return tuple(
        TrackMetadata(
            track_index=start_index + i,
            output_name=output_name,
            organism=organism,
            track_name=f"track_{start_index + i}",
        )
        for i in range(count)
    )


@dataclass(frozen=True)
class TrackMetadata:
    """Metadata for one output track (one channel in a prediction tensor).

    Attributes:
        track_index: Channel index in the prediction tensor.
        output_name: Canonical output head name (e.g., "atac", "rna_seq").
        organism: Organism index (0=human, 1=mouse).
        track_name: Human-readable track identifier.
        extras: All other metadata fields (ontology_curie, biosample_name, etc.).
            These can be accessed directly as attributes (e.g., ``track.ontology_curie``)
            or via ``.get()`` for safe access with defaults.

    Example:
        >>> track.ontology_curie  # Direct attribute access
        'UBERON:0002107'
        >>> track.get('biosample_type', 'unknown')  # Safe access with default
        'tissue'
        >>> track.has('genetically_modified')  # Check if field exists and is not None
        False
    """

    track_index: int
    output_name: str
    organism: int
    track_name: str
    extras: dict[str, Any] = field(default_factory=dict)

    # Core attribute names that should not be looked up in extras
    _CORE_FIELDS: frozenset[str] = frozenset({
        "track_index", "output_name", "organism", "track_name", "extras"
    })

    def __getattr__(self, name: str) -> Any:
        """Access extras fields as attributes.

        This allows ``track.ontology_curie`` instead of ``track.extras['ontology_curie']``.

        Raises:
            AttributeError: If the field is not found in extras.
        """
        # Avoid infinite recursion during dataclass init
        if name.startswith("_") or name in TrackMetadata._CORE_FIELDS:
            raise AttributeError(name)
        extras = object.__getattribute__(self, "extras")
        if name in extras:
            return extras[name]
        raise AttributeError(
            f"TrackMetadata has no field '{name}'. "
            f"Available extras: {list(extras.keys())}"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a field by name, checking core attributes then extras.

        Args:
            key: Field name to look up.
            default: Value to return if field is not found.

        Returns:
            The field value, or default if not found.

        Example:
            >>> track.get('ontology_curie')
            'UBERON:0002107'
            >>> track.get('nonexistent_field', 'fallback')
            'fallback'
        """
        if key in TrackMetadata._CORE_FIELDS and key != "extras":
            return getattr(self, key)
        return self.extras.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a field exists and is not None.

        Args:
            key: Field name to check.

        Returns:
            True if the field exists and has a non-None value.

        Example:
            >>> track.has('ontology_curie')
            True
            >>> track.has('genetically_modified')  # Field is None or missing
            False
        """
        return self.get(key) is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a plain dictionary."""
        result = {
            "track_index": self.track_index,
            "output_name": self.output_name,
            "organism": self.organism,
            "track_name": self.track_name,
        }
        result.update(self.extras)
        return result


class TrackMetadataCatalog:
    """Track metadata indexed by organism and output name."""

    def __init__(
        self,
        tracks_by_organism: Mapping[int, Mapping[str, tuple[TrackMetadata, ...]]] | None = None,
    ):
        self._tracks_by_organism: dict[int, dict[str, tuple[TrackMetadata, ...]]] = {}
        if tracks_by_organism is None:
            return

        for organism, per_output in tracks_by_organism.items():
            org_idx = int(organism)
            self._tracks_by_organism[org_idx] = {}
            for output_name, tracks in per_output.items():
                canonical_name = _normalize_output_name(output_name)
                ordered = sorted(tracks, key=lambda t: t.track_index)
                self._tracks_by_organism[org_idx][canonical_name] = tuple(
                    dataclasses.replace(track, track_index=i, output_name=canonical_name, organism=org_idx)
                    for i, track in enumerate(ordered)
                )

    @classmethod
    def from_file(
        cls,
        metadata_path: str | Path,
        *,
        default_organism: int = 0,
        default_output_name: str | None = None,
    ) -> "TrackMetadataCatalog":
        """Load metadata from parquet/csv/tsv."""
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Track metadata file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError(
                    "Reading parquet metadata requires pandas. "
                    "Install optional dependency: pip install alphagenome-pytorch[finetuning]"
                ) from exc
            rows = pd.read_parquet(path).to_dict(orient="records")
        elif suffix in {".csv", ".tsv", ".txt"}:
            delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                rows = list(reader)
        else:
            raise ValueError(
                f"Unsupported metadata file extension '{suffix}'. "
                "Expected .parquet, .csv, .tsv, or .txt."
            )

        return cls.from_rows(
            rows,
            default_organism=default_organism,
            default_output_name=default_output_name,
        )

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        default_organism: int = 0,
        default_output_name: str | None = None,
    ) -> "TrackMetadataCatalog":
        """Build a catalog from row dictionaries."""
        grouped: dict[tuple[int, str], list[TrackMetadata]] = {}

        for row in rows:
            data = dict(row)
            output_raw = data.get("output_type") or data.get("output_name") or default_output_name
            if output_raw is None:
                raise ValueError(
                    "Metadata row is missing output type. "
                    "Expected 'output_type' or pass default_output_name."
                )

            output_name = _normalize_output_name(str(output_raw))
            organism = _resolve_organism_index(data.get("organism"), default=default_organism)

            key = (organism, output_name)
            current = grouped.setdefault(key, [])

            track_index_raw = data.get("track_index", data.get("index"))
            if track_index_raw in (None, ""):
                track_index = len(current)
            else:
                try:
                    track_index = int(track_index_raw)
                except (TypeError, ValueError):
                    track_index = len(current)

            track_name = str(
                data.get("track_name")
                or data.get("name")
                or f"track_{track_index}"
            )

            # Core fields handled separately; everything else goes to extras
            core_keys = {
                "track_index",
                "index",
                "output_type",
                "output_name",
                "organism",
                "track_name",
                "name",
            }

            extras = {}
            for k, v in data.items():
                if k in core_keys:
                    continue
                cleaned = _clean_optional(v)
                if cleaned is not None:
                    extras[k] = cleaned

            current.append(
                TrackMetadata(
                    track_index=track_index,
                    output_name=output_name,
                    organism=organism,
                    track_name=track_name,
                    extras=extras,
                )
            )

        nested: dict[int, dict[str, tuple[TrackMetadata, ...]]] = {}
        for (organism, output_name), tracks in grouped.items():
            ordered = sorted(tracks, key=lambda t: t.track_index)
            nested.setdefault(organism, {})[output_name] = tuple(
                dataclasses.replace(track, track_index=i)
                for i, track in enumerate(ordered)
            )

        return cls(nested)

    def get_tracks(
        self,
        output_name: str,
        *,
        organism: int | str | torch.Tensor | None = 0,
        num_tracks: int | None = None,
        strict: bool = False,
    ) -> tuple[TrackMetadata, ...]:
        """Return metadata tracks for an output, optionally matched to a tensor width."""
        canonical_name = _normalize_output_name(output_name)
        organism_idx = _resolve_organism_index(organism)

        tracks = self._tracks_by_organism.get(organism_idx, {}).get(canonical_name)

        # Fallback to the first organism with metadata for this output.
        if tracks is None and not strict:
            for per_output in self._tracks_by_organism.values():
                if canonical_name in per_output:
                    tracks = per_output[canonical_name]
                    break

        if tracks is None:
            if strict:
                raise KeyError(
                    f"No metadata found for output '{canonical_name}' and organism {organism_idx}."
                )
            if num_tracks is None:
                return tuple()
            return _placeholder_tracks(canonical_name, organism_idx, num_tracks)

        if num_tracks is None:
            return tracks

        if len(tracks) == num_tracks:
            return tracks

        if len(tracks) > num_tracks:
            if strict:
                raise ValueError(
                    f"Metadata for output '{canonical_name}' has {len(tracks)} tracks, "
                    f"but tensor has {num_tracks}."
                )
            return tuple(dataclasses.replace(track, track_index=i) for i, track in enumerate(tracks[:num_tracks]))

        # len(tracks) < num_tracks
        if strict:
            raise ValueError(
                f"Metadata for output '{canonical_name}' has {len(tracks)} tracks, "
                f"but tensor has {num_tracks}."
            )

        padding = _placeholder_tracks(
            canonical_name,
            organism_idx,
            count=num_tracks - len(tracks),
            start_index=len(tracks),
        )
        merged = tuple(list(tracks) + list(padding))
        return tuple(dataclasses.replace(track, track_index=i) for i, track in enumerate(merged))

    def add_tracks(
        self,
        output_name: str,
        tracks: Iterable[TrackMetadata],
        *,
        organism: int | str | torch.Tensor | None = 0,
    ) -> None:
        """Add or replace metadata for a single output and organism."""
        org_idx = _resolve_organism_index(organism)
        canonical_name = _normalize_output_name(output_name)
        ordered = sorted(tracks, key=lambda t: t.track_index)
        normalized = tuple(
            dataclasses.replace(track, track_index=i, output_name=canonical_name, organism=org_idx)
            for i, track in enumerate(ordered)
        )
        if org_idx not in self._tracks_by_organism:
            self._tracks_by_organism[org_idx] = {}
        self._tracks_by_organism[org_idx][canonical_name] = normalized

    def has_tracks(
        self,
        output_name: str,
        *,
        organism: int | str | torch.Tensor | None = 0,
    ) -> bool:
        """Check whether metadata exists for one output and organism."""
        canonical_name = _normalize_output_name(output_name)
        organism_idx = _resolve_organism_index(organism)
        return canonical_name in self._tracks_by_organism.get(organism_idx, {})

    def outputs(self, organism: int | str | torch.Tensor | None = 0) -> list[str]:
        """List output names with metadata for one organism."""
        organism_idx = _resolve_organism_index(organism)
        return sorted(self._tracks_by_organism.get(organism_idx, {}).keys())

    @property
    def organisms(self) -> list[int]:
        """List organism indices present in this catalog."""
        return sorted(self._tracks_by_organism.keys())

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        *,
        default_organism: int = 0,
        default_output_name: str | None = None,
    ) -> "TrackMetadataCatalog":
        """Build a catalog from a pandas DataFrame.

        Args:
            df: DataFrame with columns like 'track_name', 'output_type', 'organism', etc.
            default_organism: Organism index to use if 'organism' column is missing.
            default_output_name: Output name to use if 'output_type' column is missing.

        Returns:
            A TrackMetadataCatalog.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'track_name': ['track_0', 'track_1'],
            ...     'output_type': ['atac', 'atac'],
            ...     'organism': [0, 0],
            ...     'biosample_type': ['tissue', 'cell_line'],
            ... })
            >>> catalog = TrackMetadataCatalog.from_dataframe(df)
        """
        rows = df.to_dict(orient="records")
        return cls.from_rows(
            rows,
            default_organism=default_organism,
            default_output_name=default_output_name,
        )

    @classmethod
    def load_builtin(cls, organism: str | int = "human") -> "TrackMetadataCatalog":
        """Load built-in metadata extracted from AlphaGenome.

        Args:
            organism: "human" (or 0) or "mouse" (or 1).

        Returns:
            A TrackMetadataCatalog with metadata for the specified organism.

        Raises:
            FileNotFoundError: If the built-in metadata file is not found.
                Run `python scripts/extract_track_metadata.py` to generate it.
        """
        org_idx = _resolve_organism_index(organism)
        org_name = "human" if org_idx == 0 else "mouse"

        # Try package data first
        try:
            import importlib.resources as resources

            try:
                # Python 3.9+
                files = resources.files("alphagenome_pytorch.data")
                parquet_path = files.joinpath(f"track_metadata_{org_name}.parquet")
                if hasattr(parquet_path, "is_file") and parquet_path.is_file():
                    return cls.from_file(str(parquet_path))
            except (TypeError, AttributeError):
                pass
        except ImportError:
            pass

        # Fallback to file path relative to this module
        module_dir = Path(__file__).parent
        parquet_path = module_dir / "data" / f"track_metadata_{org_name}.parquet"
        if parquet_path.exists():
            return cls.from_file(parquet_path)

        raise FileNotFoundError(
            f"Built-in metadata for '{org_name}' not found. "
            "Run `python scripts/extract_track_metadata.py` to generate it, "
            "or load your own metadata with TrackMetadataCatalog.from_file()."
        )


@dataclass(frozen=True)
class NamedTrackTensor:
    """A tensor plus aligned track metadata."""

    tensor: torch.Tensor
    tracks: tuple[TrackMetadata, ...]
    output_name: str
    resolution: int
    track_axis: int = -1

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    def _match_track(self, track: TrackMetadata, criteria: dict[str, Any]) -> bool:
        """Check if a track matches the given criteria.

        Special handling:
            - ``field=None`` matches tracks where the field is missing or None.
            - ``field=[val1, val2]`` matches tracks where field is in the list.
        """
        for key, expected in criteria.items():
            # Use track.get() for unified access to core fields and extras
            actual = track.get(key)

            if expected is None:
                # Special case: None means "field is missing or None"
                if actual is not None:
                    return False
            elif isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def indices(
        self,
        *,
        predicate: Callable[[TrackMetadata], bool] | None = None,
        **criteria: Any,
    ) -> list[int]:
        """Return channel indices for tracks matching filters.

        Args:
            predicate: Optional callable that takes a TrackMetadata and returns bool.
            **criteria: Field=value filters. Supports several matching modes:

                - ``field="value"`` - exact match
                - ``field=["a", "b"]`` - match any value in list
                - ``field=None`` - match tracks where field is missing or None

        Returns:
            List of matching channel indices.

        Example:
            >>> idx = tracks.indices(biosample_type="tissue")
            >>> tensor[..., idx]  # Manual slicing
        """
        result = []
        for axis_index, track in enumerate(self.tracks):
            if not self._match_track(track, criteria):
                continue
            if predicate is not None and not predicate(track):
                continue
            result.append(axis_index)
        return result

    def mask(
        self,
        *,
        predicate: Callable[[TrackMetadata], bool] | None = None,
        device: torch.device | str | None = None,
        **criteria: Any,
    ) -> torch.Tensor:
        """Return a boolean channel mask for tracks matching filters.

        Args:
            predicate: Optional callable that takes a TrackMetadata and returns bool.
            device: Device for the mask tensor. Defaults to self.tensor.device.
            **criteria: Field=value filters. Supports several matching modes:

                - ``field="value"`` - exact match
                - ``field=["a", "b"]`` - match any value in list
                - ``field=None`` - match tracks where field is missing or None

        Returns:
            Boolean tensor of shape (num_tracks,) where True indicates a match.

        Example:
            >>> mask = tracks.mask(biosample_type="tissue")
            >>> loss = (predictions * mask).mean()  # Mask loss to specific tracks
        """
        matched = set(self.indices(predicate=predicate, **criteria))
        if device is None:
            device = self.tensor.device
        return torch.tensor(
            [i in matched for i in range(self.num_tracks)],
            dtype=torch.bool,
            device=device,
        )

    def to_dataframe(self):
        """Convert track metadata to a pandas DataFrame.

        Returns:
            pandas.DataFrame with one row per track.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_dataframe() requires pandas. "
                "Install with: pip install pandas"
            ) from exc
        return pd.DataFrame([track.to_dict() for track in self.tracks])

    def select(
        self,
        *,
        predicate: Callable[[TrackMetadata], bool] | None = None,
        allow_empty: bool = False,
        **criteria: Any,
    ) -> "NamedTrackTensor":
        """Filter tracks by metadata values.

        Args:
            predicate: Optional callable that takes a TrackMetadata and returns bool.
            allow_empty: If True, return empty tensor when no tracks match.
                If False (default), raise ValueError.
            **criteria: Field=value filters. Supports several matching modes:

                - ``field="value"`` - exact match
                - ``field=["a", "b"]`` - match any value in list
                - ``field=None`` - match tracks where field is missing or None

        Returns:
            A new NamedTrackTensor with only the matching tracks.

        Example:
            >>> tracks.select(ontology_curie="UBERON:0002107")
            >>> tracks.select(biosample_type=["tissue", "primary cell"])
            >>> tracks.select(genetically_modified=None)  # Where field is missing
            >>> tracks.select(predicate=lambda t: "liver" in t.track_name.lower())
        """
        matched_indices: list[int] = []
        matched_tracks: list[TrackMetadata] = []

        for axis_index, track in enumerate(self.tracks):
            if not self._match_track(track, criteria):
                continue
            if predicate is not None and not predicate(track):
                continue
            matched_indices.append(axis_index)
            matched_tracks.append(track)

        if not matched_indices:
            if not allow_empty:
                raise ValueError(
                    f"No tracks matched filters for output '{self.output_name}' at {self.resolution}bp."
                )
            # Return empty tensor
            axis = self.track_axis if self.track_axis >= 0 else self.tensor.ndim + self.track_axis
            slices = [slice(None)] * self.tensor.ndim
            slices[axis] = slice(0, 0)
            empty_tensor = self.tensor[tuple(slices)]
            return NamedTrackTensor(
                tensor=empty_tensor,
                tracks=(),
                output_name=self.output_name,
                resolution=self.resolution,
                track_axis=self.track_axis,
            )

        axis = self.track_axis if self.track_axis >= 0 else self.tensor.ndim + self.track_axis
        index = torch.tensor(matched_indices, device=self.tensor.device, dtype=torch.long)
        filtered_tensor = torch.index_select(self.tensor, dim=axis, index=index)

        reindexed_tracks = tuple(
            dataclasses.replace(track, track_index=i)
            for i, track in enumerate(matched_tracks)
        )

        return NamedTrackTensor(
            tensor=filtered_tensor,
            tracks=reindexed_tracks,
            output_name=self.output_name,
            resolution=self.resolution,
            track_axis=self.track_axis,
        )

    # -------------------------------------------------------------------------
    # Arithmetic operators
    # -------------------------------------------------------------------------

    def _binary_op(
        self, other: "NamedTrackTensor | torch.Tensor | float", op: Callable
    ) -> "NamedTrackTensor":
        """Apply a binary operation, preserving metadata from self."""
        if isinstance(other, NamedTrackTensor):
            other_tensor = other.tensor
        else:
            other_tensor = other
        return NamedTrackTensor(
            tensor=op(self.tensor, other_tensor),
            tracks=self.tracks,
            output_name=self.output_name,
            resolution=self.resolution,
            track_axis=self.track_axis,
        )

    def __sub__(self, other: "NamedTrackTensor | torch.Tensor | float") -> "NamedTrackTensor":
        """Subtract: self - other."""
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: "torch.Tensor | float") -> "NamedTrackTensor":
        """Reverse subtract: other - self."""
        return self._binary_op(other, lambda a, b: b - a)

    def __add__(self, other: "NamedTrackTensor | torch.Tensor | float") -> "NamedTrackTensor":
        """Add: self + other."""
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: "torch.Tensor | float") -> "NamedTrackTensor":
        """Reverse add: other + self."""
        return self._binary_op(other, lambda a, b: b + a)

    def __mul__(self, other: "NamedTrackTensor | torch.Tensor | float") -> "NamedTrackTensor":
        """Multiply: self * other."""
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: "torch.Tensor | float") -> "NamedTrackTensor":
        """Reverse multiply: other * self."""
        return self._binary_op(other, lambda a, b: b * a)

    def __truediv__(self, other: "NamedTrackTensor | torch.Tensor | float") -> "NamedTrackTensor":
        """Divide: self / other."""
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other: "torch.Tensor | float") -> "NamedTrackTensor":
        """Reverse divide: other / self."""
        return self._binary_op(other, lambda a, b: b / a)

    def __neg__(self) -> "NamedTrackTensor":
        """Negate: -self."""
        return NamedTrackTensor(
            tensor=-self.tensor,
            tracks=self.tracks,
            output_name=self.output_name,
            resolution=self.resolution,
            track_axis=self.track_axis,
        )

    def __abs__(self) -> "NamedTrackTensor":
        """Absolute value: abs(self)."""
        return NamedTrackTensor(
            tensor=torch.abs(self.tensor),
            tracks=self.tracks,
            output_name=self.output_name,
            resolution=self.resolution,
            track_axis=self.track_axis,
        )


class NamedOutputHead:
    """Named views of one model output head across resolutions.

    Metadata (track names, ontology, biosample, etc.) is shared across
    resolutions — only the tensor shapes differ.  Metadata-level operations
    like ``select``, ``tracks``, ``indices``, and ``mask`` are available
    directly on this object without choosing a resolution first.

    Example:
        >>> head = named.rna_seq
        >>> head.tracks                          # shared metadata
        >>> head.select(strand='+')              # filtered NamedOutputHead
        >>> head.select(strand='+')[128].tensor  # tensor at 128bp
        >>> head[128].select(strand='+').tensor  # equivalent
    """

    def __init__(self, output_name: str, by_resolution: Mapping[int, NamedTrackTensor]):
        if not by_resolution:
            raise ValueError("NamedOutputHead requires at least one resolution.")
        self.output_name = output_name
        self._by_resolution = dict(sorted(by_resolution.items()))

    # -----------------------------------------------------------------
    # Resolution access
    # -----------------------------------------------------------------

    def __getitem__(self, resolution: int) -> NamedTrackTensor:
        return self._by_resolution[resolution]

    def __contains__(self, resolution: int) -> bool:
        return resolution in self._by_resolution

    def resolutions(self) -> tuple[int, ...]:
        return tuple(self._by_resolution.keys())

    def items(self):
        return self._by_resolution.items()

    def __iter__(self):
        return iter(self._by_resolution)

    # -----------------------------------------------------------------
    # Shared metadata (resolution-independent)
    # -----------------------------------------------------------------

    def _any_resolution(self) -> NamedTrackTensor:
        """Return the NamedTrackTensor for any resolution (metadata is shared)."""
        return next(iter(self._by_resolution.values()))

    @property
    def tracks(self) -> tuple["TrackMetadata", ...]:
        """Track metadata shared across all resolutions."""
        return self._any_resolution().tracks

    @property
    def num_tracks(self) -> int:
        """Number of tracks (same at every resolution)."""
        return len(self.tracks)

    def to_dataframe(self):
        """Convert track metadata to a pandas DataFrame.

        Returns:
            pandas.DataFrame with one row per track.

        Raises:
            ImportError: If pandas is not installed.
        """
        return self._any_resolution().to_dataframe()

    # -----------------------------------------------------------------
    # Filtering / selection (resolution-independent)
    # -----------------------------------------------------------------

    def indices(
        self,
        *,
        predicate: Callable[[TrackMetadata], bool] | None = None,
        **criteria: Any,
    ) -> list[int]:
        """Return channel indices for tracks matching filters.

        Args:
            predicate: Optional callable that takes a TrackMetadata and returns bool.
            **criteria: Field=value filters.

        Returns:
            List of matching channel indices.
        """
        return self._any_resolution().indices(predicate=predicate, **criteria)

    def mask(
        self,
        *,
        predicate: Callable[[TrackMetadata], bool] | None = None,
        device: "torch.device | str | None" = None,
        **criteria: Any,
    ) -> "torch.Tensor":
        """Return a boolean channel mask for tracks matching filters.

        Args:
            predicate: Optional callable that takes a TrackMetadata and returns bool.
            device: Device for the mask tensor.
            **criteria: Field=value filters.

        Returns:
            Boolean tensor of shape (num_tracks,).
        """
        return self._any_resolution().mask(predicate=predicate, device=device, **criteria)

    def select(
        self,
        *,
        predicate: Callable[[TrackMetadata], bool] | None = None,
        allow_empty: bool = False,
        **criteria: Any,
    ) -> "NamedOutputHead":
        """Filter tracks across all resolutions.

        Returns a new NamedOutputHead with only matching tracks at every
        resolution.  The result can then be indexed by resolution to get
        the filtered tensor.

        Args:
            predicate: Optional callable that takes a TrackMetadata and returns bool.
            allow_empty: If True, allow empty results.
            **criteria: Field=value filters.

        Returns:
            A new NamedOutputHead with filtered tracks at all resolutions.

        Example:
            >>> head.select(strand='+')[128].tensor
        """
        filtered = {
            res: ntt.select(predicate=predicate, allow_empty=allow_empty, **criteria)
            for res, ntt in self._by_resolution.items()
        }
        return NamedOutputHead(self.output_name, filtered)

    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        resolutions = ", ".join(str(r) for r in self._by_resolution)
        return f"NamedOutputHead(output_name='{self.output_name}', resolutions=[{resolutions}])"


class NamedOutputs:
    """Metadata-aware wrapper around raw model outputs."""

    def __init__(
        self,
        raw_outputs: Mapping[str, Any],
        named_heads: Mapping[str, NamedOutputHead],
    ):
        self._raw_outputs = dict(raw_outputs)
        self._named_heads = dict(named_heads)

    @classmethod
    def from_raw(
        cls,
        outputs: Mapping[str, Any],
        *,
        organism: int | str | torch.Tensor | None = None,
        catalog: TrackMetadataCatalog | None = None,
        strict_metadata: bool = False,
        channels_last: bool = True,
    ) -> "NamedOutputs":
        """Build named views from a raw model output dict."""
        organism_idx = _resolve_organism_index(organism, default=0)
        named_heads: dict[str, NamedOutputHead] = {}

        for output_name, output_value in outputs.items():
            if not isinstance(output_value, Mapping):
                continue

            resolution_tensors = {
                res: tensor
                for res, tensor in output_value.items()
                if isinstance(res, int) and torch.is_tensor(tensor)
            }
            if not resolution_tensors:
                continue

            by_resolution: dict[int, NamedTrackTensor] = {}
            for resolution, tensor in resolution_tensors.items():
                if tensor.ndim == 0:
                    continue
                track_axis = -1 if channels_last else (1 if tensor.ndim >= 2 else -1)
                axis = track_axis if track_axis >= 0 else tensor.ndim + track_axis
                num_tracks = int(tensor.shape[axis])

                if catalog is None:
                    tracks = _placeholder_tracks(
                        _normalize_output_name(output_name),
                        organism_idx,
                        num_tracks,
                    )
                else:
                    tracks = catalog.get_tracks(
                        output_name,
                        organism=organism_idx,
                        num_tracks=num_tracks,
                        strict=strict_metadata,
                    )

                by_resolution[resolution] = NamedTrackTensor(
                    tensor=tensor,
                    tracks=tracks,
                    output_name=_normalize_output_name(output_name),
                    resolution=resolution,
                    track_axis=track_axis,
                )

            if by_resolution:
                named_heads[output_name] = NamedOutputHead(output_name, by_resolution)

        return cls(outputs, named_heads)

    def heads(self) -> list[str]:
        """Return output head names that support named/resolution views."""
        return list(self._named_heads.keys())

    def head(self, name: str) -> NamedOutputHead:
        return self._named_heads[name]

    def as_dict(self) -> dict[str, Any]:
        """Return the original raw output dictionary."""
        return self._raw_outputs

    def __contains__(self, key: str) -> bool:
        return key in self._raw_outputs

    def __getitem__(self, key: str) -> Any:
        if key in self._named_heads:
            return self._named_heads[key]
        return self._raw_outputs[key]

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        if key in self._named_heads:
            return self._named_heads[key]
        if key in self._raw_outputs:
            return self._raw_outputs[key]
        raise AttributeError(
            f"NamedOutputs has no attribute '{key}'. "
            f"Available heads: {self.heads()}"
        )

    def keys(self):
        return self._raw_outputs.keys()

    def items(self):
        return self._raw_outputs.items()

    def values(self):
        return self._raw_outputs.values()

    def __iter__(self):
        return iter(self._raw_outputs)

    def __len__(self) -> int:
        return len(self._raw_outputs)

    def select(
        self,
        *,
        predicate: Callable[[TrackMetadata], bool] | None = None,
        allow_empty: bool = False,
        **criteria: Any,
    ) -> dict[tuple[str, int], NamedTrackTensor]:
        """Filter tracks across all heads and resolutions.

        Args:
            predicate: Optional callable that takes a TrackMetadata and returns bool.
            allow_empty: If True, include heads with no matching tracks.
            **criteria: Field=value filters applied to all heads.

        Returns:
            Dict mapping (output_name, resolution) to filtered NamedTrackTensor.

        Raises:
            ValueError: If no tracks match filters (unless allow_empty=True).

        Example:
            >>> # Get all tissue tracks across all outputs
            >>> tissue_tracks = named.select(biosample_type='tissue')
            >>> for (output, res), tracks in tissue_tracks.items():
            ...     print(f"{output}@{res}bp: {tracks.num_tracks} tracks")
        """
        result: dict[tuple[str, int], NamedTrackTensor] = {}

        for output_name, head in self._named_heads.items():
            for resolution, track_tensor in head.items():
                filtered = track_tensor.select(
                    predicate=predicate,
                    allow_empty=True,
                    **criteria,
                )
                if filtered.num_tracks > 0 or allow_empty:
                    result[(output_name, resolution)] = filtered

        if not result and not allow_empty:
            raise ValueError("No tracks matched filters across any outputs.")

        return result

    def __repr__(self) -> str:
        return (
            "NamedOutputs("
            f"heads={self.heads()}, "
            f"raw_keys={list(self._raw_outputs.keys())}"
            ")"
        )


__all__ = [
    "NamedOutputs",
    "NamedOutputHead",
    "NamedTrackTensor",
    "TrackMetadata",
    "TrackMetadataCatalog",
]
