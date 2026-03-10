"""Core data types for variant scoring."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from .scorers.base import BaseVariantScorer


class OutputType(enum.Enum):
    """Model output types for variant scoring.

    Maps to the output heads available in the AlphaGenome model.
    """
    ATAC = 'atac'
    CAGE = 'cage'
    DNASE = 'dnase'
    PROCAP = 'procap'
    RNA_SEQ = 'rna_seq'
    CHIP_HISTONE = 'chip_histone'
    CHIP_TF = 'chip_tf'
    SPLICE_SITES = 'splice_sites_classification'
    SPLICE_SITE_USAGE = 'splice_sites_usage'
    SPLICE_JUNCTIONS = 'splice_sites_junction'
    CONTACT_MAPS = 'pair_activations'

import enum


class Width(enum.IntEnum):
    W_2KB   = 2 * 1024
    W_4KB   = 4 * 1024
    W_8KB   = 8 * 1024
    W_16KB  = 16 * 1024
    W_100KB = 128 * 1024
    W_300KB = 256 * 1024
    W_500KB = 512 * 1024
    W_1MB   = 1024 * 1024

    @classmethod
    def normalize(cls, value: Union["Width", int, str]) -> "Width":
        # Already a Width
        if isinstance(value, cls):
            return value

        # Raw integer (base pairs)
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"{value} is not a supported width. "
                    f"Choose from: {', '.join(w.name[2:] for w in cls)}"
                )

        # String like "2KB", "100KB", "1MB"
        if isinstance(value, str):
            key = f"W_{value.strip().upper()}"
            try:
                return cls[key]
            except KeyError:
                raise ValueError(
                    f"Invalid width '{value}'. "
                    f"Choose from: {', '.join(w.name[2:] for w in cls)}"
                )

        raise TypeError(f"Unsupported width type: {type(value)}")


class AggregationType(enum.Enum):
    """Variant score aggregation methods.

    Defines how reference and alternate predictions are compared.

    Signed/directional (scores can be positive or negative):
        - DIFF_MEAN: mean(ALT) - mean(REF)
        - DIFF_SUM: sum(ALT) - sum(REF)
        - DIFF_SUM_LOG2: sum(log2(ALT)) - sum(log2(REF))
        - DIFF_LOG2_SUM: log2(sum(ALT)) - log2(sum(REF))

    Unsigned/non-directional (scores are always positive):
        - L2_DIFF: ||ALT - REF||_2 (Euclidean norm of difference)
        - L2_DIFF_LOG1P: ||log1p(ALT) - log1p(REF)||_2
        - ACTIVE_MEAN: max(mean(ALT), mean(REF))
        - ACTIVE_SUM: max(sum(ALT), sum(REF))
    """
    DIFF_MEAN = 'diff_mean'
    DIFF_SUM = 'diff_sum'
    DIFF_SUM_LOG2 = 'diff_sum_log2'
    DIFF_LOG2_SUM = 'diff_log2_sum'
    L2_DIFF = 'l2_diff'
    L2_DIFF_LOG1P = 'l2_diff_log1p'
    ACTIVE_MEAN = 'active_mean'
    ACTIVE_SUM = 'active_sum'

    def is_signed(self) -> bool:
        """Whether this aggregation produces signed (directional) scores."""
        return self in (
            AggregationType.DIFF_MEAN,
            AggregationType.DIFF_SUM,
            AggregationType.DIFF_SUM_LOG2,
            AggregationType.DIFF_LOG2_SUM,
        )


@dataclass(frozen=True)
class Interval:
    """Genomic interval (0-based, half-open coordinates).

    Args:
        chromosome: Chromosome name (e.g., 'chr1', 'chrX')
        start: 0-based start position (inclusive)
        end: 0-based end position (exclusive)
        strand: Strand ('+', '-', or '.' for unstranded)
        name: Optional name/identifier

    Example:
        >>> interval = Interval('chr22', 36136162, 36267234)
        >>> interval.width
        "1KB"
        >>> str(interval)
        'chr22:36136162-36267234'
    """
    chromosome: str
    start: int
    end: int
    strand: str = '.'
    name: str = ''

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f"Start position must be non-negative, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"End ({self.end}) must be greater than start ({self.start})")
        if self.strand not in ('+', '-', '.'):
            raise ValueError(f"Strand must be '+', '-', or '.', got '{self.strand}'")

    @property
    def width(self) -> int:
        """Length of the interval in base pairs."""
        return self.end - self.start

    @property
    def center(self) -> int:
        """Center position of the interval (0-based)."""
        return (self.start + self.end) // 2

    def contains(self, position: int) -> bool:
        """Check if a position falls within this interval."""
        return self.start <= position < self.end

    def __str__(self) -> str:
        if self.strand == '.':
            return f"{self.chromosome}:{self.start}-{self.end}"
        return f"{self.chromosome}:{self.start}-{self.end}:{self.strand}"

    @classmethod
    def from_str(cls, s: str) -> Interval:
        """Parse interval from string format.

        Supports formats:
            - 'chr1:100-200' (without strand)
            - 'chr1:100-200:+' (with strand)
        """
        # Try with strand first
        match = re.match(r'^([^:]+):(\d+)-(\d+):([+-\.])$', s)
        if match:
            return cls(
                chromosome=match.group(1),
                start=int(match.group(2)),
                end=int(match.group(3)),
                strand=match.group(4),
            )

        # Without strand
        match = re.match(r'^([^:]+):(\d+)-(\d+)$', s)
        if match:
            return cls(
                chromosome=match.group(1),
                start=int(match.group(2)),
                end=int(match.group(3)),
            )

        raise ValueError(f"Could not parse interval string: '{s}'")

    @classmethod
    def centered_on(cls, chromosome: str, position: int, width: Union[Width, int, str] = Width.W_100KB) -> Interval:
        """Create an interval centered on a genomic position.

        Args:
            chromosome: Chromosome name
            position: Center position (0-based)
            width: Total width of the interval (default: 131072 for AlphaGenome)
        """
        width = Width.normalize(width)
        half_width = width // 2
        return cls(
            chromosome=chromosome,
            start=max(0, position - half_width),
            end=position + half_width + (width % 2),
        )


@dataclass(frozen=True)
class Variant:
    """Genomic variant (VCF-style, 1-based position).

    Args:
        chromosome: Chromosome name (e.g., 'chr1', 'chrX')
        position: 1-based position (VCF convention)
        reference_bases: Reference allele sequence
        alternate_bases: Alternate allele sequence
        name: Optional variant identifier (e.g., rsID)

    Example:
        >>> v = Variant('chr22', 36201698, 'A', 'C')
        >>> v.is_snv
        True
        >>> str(v)
        'chr22:36201698:A>C'
    """
    chromosome: str
    position: int  # 1-based (VCF convention)
    reference_bases: str
    alternate_bases: str
    name: str = ''

    def __post_init__(self):
        if self.position < 1:
            raise ValueError(f"Position must be >= 1 (VCF convention), got {self.position}")
        if not self.reference_bases:
            raise ValueError("Reference bases cannot be empty")
        if not self.alternate_bases:
            raise ValueError("Alternate bases cannot be empty")

    @property
    def start(self) -> int:
        """0-based start position."""
        return self.position - 1

    @property
    def end(self) -> int:
        """0-based end position (exclusive)."""
        return self.start + len(self.reference_bases)

    @property
    def is_snv(self) -> bool:
        """Whether this is a single nucleotide variant."""
        return len(self.reference_bases) == 1 and len(self.alternate_bases) == 1

    @property
    def is_insertion(self) -> bool:
        """Whether this is an insertion."""
        return len(self.alternate_bases) > len(self.reference_bases)

    @property
    def is_deletion(self) -> bool:
        """Whether this is a deletion."""
        return len(self.alternate_bases) < len(self.reference_bases)

    @property
    def is_indel(self) -> bool:
        """Whether this is an insertion or deletion."""
        return self.is_insertion or self.is_deletion

    def __str__(self) -> str:
        return f"{self.chromosome}:{self.position}:{self.reference_bases}>{self.alternate_bases}"

    @classmethod
    def from_str(cls, s: str, format: str = 'default') -> Variant:
        """Parse variant from string format.

        Args:
            s: Variant string
            format: Format type:
                - 'default': 'chr:pos:ref>alt' (e.g., 'chr22:36201698:A>C')
                - 'gtex': 'chr_pos_ref_alt_build' (e.g., 'chr22_36201698_A_C_b38')
                - 'gnomad': 'chr-pos-ref-alt' (e.g., 'chr22-36201698-A-C')
        """
        if format == 'default':
            match = re.match(r'^([^:]+):(\d+):([ACGTN]+)>([ACGTN]+)$', s, re.IGNORECASE)
            if match:
                return cls(
                    chromosome=match.group(1),
                    position=int(match.group(2)),
                    reference_bases=match.group(3).upper(),
                    alternate_bases=match.group(4).upper(),
                )
            raise ValueError(f"Could not parse variant string: '{s}'")

        elif format == 'gtex':
            # Format: chr_pos_ref_alt_build (e.g., chr22_36201698_A_C_b38)
            parts = s.split('_')
            if len(parts) >= 4:
                return cls(
                    chromosome=parts[0],
                    position=int(parts[1]),
                    reference_bases=parts[2].upper(),
                    alternate_bases=parts[3].upper(),
                )
            raise ValueError(f"Could not parse GTEx variant string: '{s}'")

        elif format == 'gnomad':
            # Format: chr-pos-ref-alt (e.g., chr22-36201698-A-C)
            parts = s.split('-')
            if len(parts) >= 4:
                return cls(
                    chromosome=parts[0],
                    position=int(parts[1]),
                    reference_bases=parts[2].upper(),
                    alternate_bases=parts[3].upper(),
                )
            raise ValueError(f"Could not parse gnomAD variant string: '{s}'")

        else:
            raise ValueError(f"Unknown format: '{format}'")


@dataclass
class VariantScore:
    """Score result for a single variant-scorer combination.

    Args:
        variant: The variant that was scored
        interval: The genomic interval used for scoring
        scorer: The scorer configuration used
        scores: Per-track scores tensor of shape (num_tracks,)
        gene_id: Gene ID (for gene-centric scorers)
        gene_name: Gene name/symbol (for gene-centric scorers)
        gene_type: Gene biotype (for gene-centric scorers, e.g., 'protein_coding')
        gene_strand: Gene strand (for gene-centric scorers, '+', '-', or '.')
    """
    variant: Variant
    interval: Interval
    scorer: BaseVariantScorer
    scores: torch.Tensor  # (num_tracks,)
    gene_id: str | None = None
    gene_name: str | None = None
    gene_type: str | None = None
    gene_strand: str | None = None
    junction_start: int | None = None
    junction_end: int | None = None

    @property
    def scorer_name(self) -> str:
        """Name of the scorer used."""
        return self.scorer.name

    @property
    def output_type(self) -> OutputType:
        """Output type of the scorer."""
        return self.scorer.requested_output

    @property
    def is_signed(self) -> bool:
        """Whether the scores are directional (can be negative)."""
        return self.scorer.is_signed

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            'variant': str(self.variant),
            'interval': str(self.interval),
            'scorer': self.scorer_name,
            'output_type': self.output_type.value,
            'is_signed': self.is_signed,
            'gene_id': self.gene_id,
            'gene_name': self.gene_name,
            'gene_type': self.gene_type,
            'gene_strand': self.gene_strand,
            'junction_Start': self.junction_start,
            'junction_End': self.junction_end,
            'scores': self.scores.float().cpu().numpy(),
        }


def scores_to_dataframe(
    scores: list[VariantScore] | list[list[VariantScore]],
    expand_tracks: bool = True,
) -> 'pd.DataFrame':
    """Convert variant scores to a pandas DataFrame.

    Args:
        scores: List of VariantScore objects (or nested list from batch scoring)
        expand_tracks: If True, creates one row per track. If False, keeps
            scores as a single array column.

    Returns:
        DataFrame with columns: variant, interval, scorer, output_type,
        is_signed, gene_id, gene_name, and either raw_score (if expand_tracks)
        or scores (if not expand_tracks).
    """
    import pandas as pd

    # Flatten nested lists
    flat_scores = []
    for item in scores:
        if isinstance(item, list):
            flat_scores.extend(item)
        else:
            flat_scores.append(item)

    if not expand_tracks:
        return pd.DataFrame([s.to_dict() for s in flat_scores])

    # Expand to one row per track
    rows = []
    for score in flat_scores:
        base = score.to_dict()
        track_scores = base.pop('scores')

        for i, s in enumerate(track_scores):
            row = {**base, 'track_index': i, 'raw_score': float(s)}
            rows.append(row)

    return pd.DataFrame(rows)


@dataclass
class TrackMetadata:
    """Metadata for a single output track.

    This class holds information about individual tracks that can be used
    to enrich variant scores with descriptive information.

    Args:
        track_index: Index of the track within its output type
        track_name: Human-readable track name
        track_strand: Strand of the track ('+', '-', or '.')
        output_type: The output type this track belongs to
        ontology_curie: Ontology term for cell type/tissue (e.g., 'UBERON:0036149')
        gtex_tissue: GTEx tissue name (e.g., 'Liver')
        assay_title: Assay subtype (e.g., 'total RNA-seq')
        biosample_name: Biosample name (e.g., 'liver')
        biosample_type: Biosample type (e.g., 'tissue', 'primary cell')
        transcription_factor: TF name for ChIP-TF tracks (e.g., 'CTCF')
        histone_mark: Histone mark for ChIP-Histone tracks (e.g., 'H3K4ME3')
    """
    track_index: int
    track_name: str
    track_strand: str = '.'
    output_type: OutputType | None = None
    ontology_curie: str | None = None
    gtex_tissue: str | None = None
    assay_title: str | None = None
    biosample_name: str | None = None
    biosample_type: str | None = None
    transcription_factor: str | None = None
    histone_mark: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'track_index': self.track_index,
            'track_name': self.track_name,
            'track_strand': self.track_strand,
            'output_type': self.output_type.value if self.output_type else None,
            'ontology_curie': self.ontology_curie,
            'gtex_tissue': self.gtex_tissue,
            'assay_title': self.assay_title,
            'biosample_name': self.biosample_name,
            'biosample_type': self.biosample_type,
            'transcription_factor': self.transcription_factor,
            'histone_mark': self.histone_mark,
        }


def tidy_scores(
    scores: list[VariantScore] | list[list[VariantScore]],
    track_metadata: dict[OutputType, list[TrackMetadata]] | None = None,
    match_gene_strand: bool = True,
    include_extended_metadata: bool = True,
) -> 'pd.DataFrame | None':
    """Format variant scores into a tidy (long) pandas DataFrame.

    This function converts variant scores into a readable DataFrame with one
    score per row. It handles both variant-centric scoring (one score row per
    variant-track pair) and gene-centric scoring (one score row per
    variant-gene-track combination).

    The function accepts these score input types:
    - A list of VariantScore objects (e.g., output of `score_variant`)
    - A nested list of VariantScore objects (e.g., output of `score_variants`)

    Scores from multiple scorers or variants are concatenated into a single
    DataFrame containing all applicable columns.

    Args:
        scores: Scoring output as either a list of VariantScore objects, or a
            nested list of VariantScore objects.
        track_metadata: Optional dictionary mapping OutputType to list of
            TrackMetadata objects for enriching output with track information.
        match_gene_strand: If True (and using gene-centric scoring), rows with
            mismatched gene and track strands are removed.
        include_extended_metadata: If True, include additional metadata columns
            (ontology_curie, gtex_tissue, etc.) when track_metadata is provided.

    Returns:
        pd.DataFrame with columns:
            - variant_id: Variant of interest (e.g., chr22:36201698:A>C)
            - scored_interval: Genomic interval scored (e.g., chr22:36100000-36300000)
            - gene_id: ENSEMBL gene ID without version (e.g., ENSG00000100342),
                or None if not applicable
            - gene_name: HGNC gene symbol (e.g., APOL1), or None if not applicable
            - gene_type: Gene biotype (e.g., protein_coding), or None if not applicable
            - gene_strand: Strand of the gene ('+', '-', '.'), or None if not applicable
            - junction_Start: Splice junction start position (if applicable)
            - junction_End: Splice junction end position (if applicable)
            - output_type: Type of model output (e.g., rna_seq, dnase)
            - variant_scorer: Name of the variant scorer used
            - track_index: Index of the track within its output type
            - track_name: Name of the track (if track_metadata provided)
            - track_strand: Strand of the track ('+', '-', or '.')
            - ontology_curie: Ontology term (if track_metadata provided)
            - gtex_tissue: GTEx tissue name (if track_metadata provided)
            - Assay title: Assay subtype (if track_metadata provided)
            - biosample_name: Biosample name (if track_metadata provided)
            - biosample_type: Biosample type (if track_metadata provided)
            - transcription_factor: TF name (if track_metadata provided)
            - histone_mark: Histone mark name (if track_metadata provided)
            - raw_score: Raw variant score

        Returns None if no scores are provided.

    Example:
        >>> from alphagenome_pytorch.variant_scoring import (
        ...     VariantScoringModel, Variant, Interval,
        ...     CenterMaskScorer, OutputType, AggregationType, tidy_scores,
        ... )
        >>> scores = scoring_model.score_variant(
        ...     interval, variant,
        ...     scorers=[CenterMaskScorer(OutputType.ATAC, 501, AggregationType.DIFF_LOG2_SUM)],
        ... )
        >>> df = tidy_scores(scores)
        >>> print(df.head())

    Raises:
        ValueError: If the input is not a valid type.
    """
    import pandas as pd

    # Flatten nested lists
    flat_scores: list[VariantScore] = []
    for item in scores:
        if isinstance(item, list):
            flat_scores.extend(item)
        else:
            flat_scores.append(item)

    if not flat_scores:
        return None

    # Build rows
    rows = []
    for score in flat_scores:
        num_tracks = score.scores.shape[0]
        output_type = score.output_type
        track_scores = score.scores.float().cpu().numpy()

        # Get track metadata for this output type if available
        output_track_meta = None
        if track_metadata is not None and output_type in track_metadata:
            output_track_meta = track_metadata[output_type]

        for track_idx in range(num_tracks):
            row = {
                'variant_id': str(score.variant),
                'scored_interval': str(score.interval),
                'gene_id': score.gene_id,
                'gene_name': score.gene_name,
                'gene_type': score.gene_type,
                'gene_strand': score.gene_strand,
                'junction_Start': score.junction_start,
                'junction_End': score.junction_end,
                'output_type': output_type.value,
                'variant_scorer': score.scorer_name,
                'track_index': track_idx,
                'raw_score': float(track_scores[track_idx]),
            }

            # Add track metadata if available
            if output_track_meta is not None and track_idx < len(output_track_meta):
                meta = output_track_meta[track_idx]
                row['track_name'] = meta.track_name
                row['track_strand'] = meta.track_strand

                if include_extended_metadata:
                    row['ontology_curie'] = meta.ontology_curie
                    row['gtex_tissue'] = meta.gtex_tissue
                    row['Assay title'] = meta.assay_title  # Using 'Assay title' as requested
                    row['biosample_name'] = meta.biosample_name
                    row['biosample_type'] = meta.biosample_type
                    row['transcription_factor'] = meta.transcription_factor
                    row['histone_mark'] = meta.histone_mark
            else:
                row['track_name'] = f'track_{track_idx}'
                row['track_strand'] = '.'
                if include_extended_metadata:
                    row['Assay title'] = None

            rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all requested columns exist (even if all None)
    # This matches the user's requested schema
    requested_cols = [
        'gene_id', 'gene_name', 'gene_type', 'gene_strand', 
        'junction_Start', 'junction_End', 
        'output_type', 'variant_scorer', 
        'track_name', 'track_strand', 
        'Assay title', 'ontology_curie', 
        'biosample_name', 'biosample_type', 
        'transcription_factor', 'histone_mark', 'gtex_tissue', 
        'raw_score'
    ]
    for col in requested_cols:
        if col not in df.columns:
            df[col] = None

    # Reorder columns to match the requested order (with variant_id/interval first)
    first_cols = ['variant_id', 'scored_interval']
    # Filter requested_cols to only those present in df (to avoid errors if logic changes)
    ordered_cols = first_cols + [c for c in requested_cols if c in df.columns]
    # Add any remaining columns at the end
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    return df


def load_track_metadata(
    metadata_path: str,
    output_type: OutputType,
) -> list[TrackMetadata]:
    """Load track metadata from a CSV or parquet file.

    Args:
        metadata_path: Path to metadata file (CSV or parquet)
        output_type: OutputType these tracks belong to

    Returns:
        List of TrackMetadata objects

    The file should have columns:
        - track_name (required)
        - track_strand (optional, default '.')
        - ontology_curie (optional)
        - gtex_tissue (optional)
        - assay_title (optional)
        - biosample_name (optional)
        - biosample_type (optional)
        - transcription_factor (optional)
        - histone_mark (optional)
    """
    import pandas as pd
    from pathlib import Path

    path = Path(metadata_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    tracks = []
    for idx, row in df.iterrows():
        meta = TrackMetadata(
            track_index=idx,
            track_name=row.get('track_name', f'track_{idx}'),
            track_strand=row.get('track_strand', '.'),
            output_type=output_type,
            ontology_curie=row.get('ontology_curie'),
            gtex_tissue=row.get('gtex_tissue'),
            assay_title=row.get('assay_title'),
            biosample_name=row.get('biosample_name'),
            biosample_type=row.get('biosample_type'),
            transcription_factor=row.get('transcription_factor'),
            histone_mark=row.get('histone_mark'),
        )
        tracks.append(meta)

    return tracks


def scores_to_anndata(
    scores: list[VariantScore] | list[list[VariantScore]],
    track_metadata: dict[OutputType, list[TrackMetadata]] | None = None,
) -> 'anndata.AnnData':
    """Convert variant scores to AnnData format for compatibility with official API.

    This function converts VariantScore objects to an AnnData object, which is
    the output format used by the official AlphaGenome gRPC API. This enables
    interoperability with tools designed for the official API.

    Args:
        scores: Scoring output as either a list of VariantScore objects, or a
            nested list of VariantScore objects.
        track_metadata: Optional dictionary mapping OutputType to list of
            TrackMetadata objects for enriching var (track) metadata.

    Returns:
        anndata.AnnData with:
            - X: Score matrix (n_obs x n_vars) where obs are variants/genes
                 and vars are tracks
            - obs: Observation metadata (variant_id, gene_id, gene_name, etc.)
            - var: Variable metadata (track_name, track_strand, etc.)
            - uns: Unstructured metadata (scorer info)

    Raises:
        ImportError: If anndata is not installed.

    Example:
        >>> from alphagenome_pytorch.variant_scoring import scores_to_anndata
        >>> adata = scores_to_anndata(scores, track_metadata=metadata)
        >>> print(adata.X.shape)  # (n_variants, n_tracks)
        >>> print(adata.obs)  # variant/gene metadata
        >>> print(adata.var)  # track metadata
    """
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "anndata is required for AnnData output. "
            "Install with: pip install anndata"
        )

    import numpy as np
    import pandas as pd
    import torch

    # Flatten nested lists
    flat_scores: list[VariantScore] = []
    for item in scores:
        if isinstance(item, list):
            flat_scores.extend(item)
        else:
            flat_scores.append(item)

    if not flat_scores:
        return anndata.AnnData()

    # Build observation (variant/gene) metadata and score matrix
    obs_data = []
    X_rows = []

    for score in flat_scores:
        # Extract scores as numpy array
        if torch.is_tensor(score.scores):
            scores_np = score.scores.float().cpu().numpy()
        else:
            scores_np = np.asarray(score.scores)

        X_rows.append(scores_np)

        # Build obs row
        obs_row = {
            'variant_id': str(score.variant),
            'interval': str(score.interval),
            'scorer': score.scorer.name if hasattr(score.scorer, 'name') else str(score.scorer),
            'output_type': score.scorer.requested_output.value if hasattr(score.scorer, 'requested_output') else None,
            'is_signed': score.scorer.is_signed if hasattr(score.scorer, 'is_signed') else None,
        }

        # Add gene metadata if present
        obs_row['gene_id'] = score.gene_id
        obs_row['gene_name'] = score.gene_name
        obs_row['gene_type'] = score.gene_type
        obs_row['gene_strand'] = score.gene_strand

        # Add junction info if present
        obs_row['junction_start'] = score.junction_start
        obs_row['junction_end'] = score.junction_end

        obs_data.append(obs_row)

    # Create observation DataFrame
    obs = pd.DataFrame(obs_data)

    # Stack score rows into matrix
    X = np.stack(X_rows, axis=0)

    # Build variable (track) metadata if available
    var = None
    if track_metadata and flat_scores:
        # Get the output type from the first score
        first_score = flat_scores[0]
        if hasattr(first_score.scorer, 'requested_output'):
            output_type = first_score.scorer.requested_output
            if output_type in track_metadata:
                var_data = []
                for meta in track_metadata[output_type]:
                    var_data.append(meta.to_dict())
                var = pd.DataFrame(var_data)

    # If no track metadata, create minimal var DataFrame
    if var is None:
        n_tracks = X.shape[1] if len(X.shape) > 1 else 1
        var = pd.DataFrame({
            'track_index': range(n_tracks),
            'track_name': [f'track_{i}' for i in range(n_tracks)],
        })

    # Ensure var index matches X columns
    if len(var) != X.shape[1]:
        # Resize var to match X columns
        n_tracks = X.shape[1]
        var = pd.DataFrame({
            'track_index': range(n_tracks),
            'track_name': [f'track_{i}' for i in range(n_tracks)],
        })

    # Create AnnData object
    adata = anndata.AnnData(
        X=X.astype(np.float32),
        obs=obs,
        var=var,
    )

    # Add scorer info to uns
    if flat_scores:
        scorer = flat_scores[0].scorer
        adata.uns['scorer'] = scorer.name if hasattr(scorer, 'name') else str(scorer)

    return adata
