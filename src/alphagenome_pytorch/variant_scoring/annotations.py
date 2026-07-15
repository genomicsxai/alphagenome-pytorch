"""Gene annotation utilities for variant scoring.

This module provides classes for loading and querying gene annotations
from GTF/GFF or Parquet files for use with gene-centric variant scorers.

Convert GTF files to Parquet with:
    python scripts/convert_gtf_to_parquet.py --input annotation.gtf --output annotation.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

if TYPE_CHECKING:
    from .types import Interval, Variant


@dataclass
class GeneInfo:
    """Basic gene information."""
    gene_id: str
    gene_name: str | None
    gene_type: str | None
    chromosome: str
    start: int  # 0-based
    end: int  # exclusive
    strand: str


_REQUIRED_COLUMNS = ('Feature', 'Chromosome', 'Start', 'End', 'gene_id')


def _validate_annotation_frame(df: pd.DataFrame, source: str) -> None:
    """Reject frames GeneAnnotation cannot index, at the point of supply.

    Gene lookups are driven entirely by ``Feature == 'gene'`` rows, so a frame
    without them answers every query with an empty result instead of raising.
    Filtering a GTF on a transcript-level attribute (``tag``, ``transcript_type``)
    drops those rows — fail loudly instead.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Annotation from {source} is missing required column(s): {missing}. "
            f"Expected at least {list(_REQUIRED_COLUMNS)}."
        )
    if not (df['Feature'] == 'gene').any():
        raise ValueError(
            f"Annotation from {source} contains no `Feature == 'gene'` rows, so "
            "every gene lookup would return empty. Filtering a GTF on a "
            "transcript-level attribute (e.g. tag == 'MANE_Select') drops them — "
            "keep the gene rows alongside the filtered transcript/exon rows."
        )


class GeneAnnotation:
    """Load and query gene/exon annotations from a GTF, Parquet file, or DataFrame.

    Accepts GTF/GFF files (via pyranges), Parquet files, or an in-memory
    DataFrame in the same layout — one row per GTF feature, with at least
    ``Feature``, ``Chromosome``, ``Start``, ``End``, and ``gene_id`` columns,
    and coordinates 0-based half-open.

    Passing a DataFrame is the supported way to restrict the annotation: filter
    it with pandas first. Keep the ``Feature == 'gene'`` rows — gene lookups are
    built from those, and a frame without them is rejected at construction.

    To convert GTF to Parquet:
        python scripts/convert_gtf_to_parquet.py --input annotation.gtf --output annotation.parquet

    Example:
        >>> annotation = GeneAnnotation('/path/to/gencode.parquet')
        >>> genes = annotation.get_genes_in_interval(interval)
        >>>
        >>> annotation = GeneAnnotation('/path/to/gencode.gtf')
        >>>
        >>> # Pre-filtered DataFrame (e.g. protein-coding only). `gene_type` is
        >>> # present on gene and exon rows alike, so one predicate covers both.
        >>> import pandas as pd
        >>> gtf = pd.read_parquet('/path/to/gencode.parquet')
        >>> annotation = GeneAnnotation(gtf[gtf.gene_type == 'protein_coding'])
    """

    def __init__(self, annotation: str | Path | pd.DataFrame):
        """Initialize from an annotation path or a DataFrame.

        Args:
            annotation: One of:
                - Parquet file path (.parquet)
                - GTF/GFF file path (.gtf, .gff, .gff3), requires pyranges
                - A pandas DataFrame in GTF layout (see the class docstring).
                  Held by reference and not copied, so do not mutate it
                  afterwards; the indices built from it would go stale.

        Raises:
            ValueError: If a DataFrame is missing required columns or has no
                ``Feature == 'gene'`` rows.
        """
        self._df: pd.DataFrame | None = None
        self._gene_index: dict[str, GeneInfo] = {}
        self._gene_index_built: bool = False
        # Exon coordinates by (versionless) gene id. Built once, lazily, in a
        # single groupby pass — see `_get_exons_for_gene`.
        self._exon_cache: dict[str, list[tuple[int, int]]] = {}
        self._exon_index_built: bool = False
        # Per-chromosome start-sorted gene index for fast interval overlap
        # queries; built lazily in `get_genes_in_interval`.
        self._interval_index: dict[str, dict[str, Any]] | None = None
        self._max_gene_span: int = 0

        if isinstance(annotation, pd.DataFrame):
            # Validate eagerly: there is no IO to defer, and raising here points
            # at the caller's filter rather than at some later gene lookup.
            _validate_annotation_frame(annotation, 'DataFrame')
            self.annotation_path = None
            self._file_format = 'dataframe'
            self._df = annotation
            return

        self.annotation_path = Path(annotation)

        # Detect file format
        suffix = self.annotation_path.suffix.lower()
        if suffix == '.parquet':
            self._file_format = 'parquet'
        elif suffix in ('.gtf', '.gff', '.gff3'):
            self._file_format = 'gtf'
            # Only check for pyranges when GTF is used
            try:
                import pyranges as _pr  # noqa: F401
                del _pr
            except ImportError:
                raise ImportError(
                    "pyranges is required to read GTF/GFF files. "
                    "Install with: pip install pyranges\n"
                    "Or pass a .parquet file or a DataFrame, neither of which "
                    "needs it. (scripts/convert_gtf_to_parquet.py converts a GTF "
                    "once, but itself runs on pyranges.)"
                )
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Expected .parquet, .gtf, .gff, or .gff3"
            )

    # Keep gtf_path as alias for backward compatibility
    @property
    def gtf_path(self) -> Path | None:
        """Alias for annotation_path (backward compatibility).

        ``None`` when built from a DataFrame.
        """
        return self.annotation_path

    @property
    def df(self) -> pd.DataFrame:
        """Annotation DataFrame; loaded and indexed on first access."""
        if self._df is None:
            if self._file_format == 'parquet':
                self._load_from_parquet()
            else:
                self._load_from_gtf()
            _validate_annotation_frame(self._df, str(self.annotation_path))
        if not self._gene_index_built:
            self._build_gene_index()
        return self._df

    # Keep gtf property for backward compatibility (returns DataFrame now)
    @property
    def gtf(self) -> pd.DataFrame:
        """Alias for df property (backward compatibility)."""
        return self.df

    def _load_from_parquet(self) -> None:
        """Load annotations from Parquet file."""
        self._df = pd.read_parquet(self.annotation_path)

    def _load_from_gtf(self) -> None:
        """Load annotations from GTF file using pyranges."""
        import pyranges
        pr_obj = pyranges.read_gtf(str(self.annotation_path))
        self._df = pr_obj.df

    def _build_gene_index(self) -> None:
        """Build index of gene information."""
        # Filter for gene features
        genes_df = self._df[self._df['Feature'] == 'gene']
        self._gene_index_built = True

        for _, row in genes_df.iterrows():
            gene_id = row.get('gene_id', '')
            # Remove version suffix if present (e.g., ENSG00000123456.1 -> ENSG00000123456)
            gene_id_base = gene_id.split('.')[0] if gene_id else ''

            # Coordinates are 0-based (from pyranges or Parquet)
            self._gene_index[gene_id_base] = GeneInfo(
                gene_id=gene_id_base,
                gene_name=row.get('gene_name'),
                gene_type=row.get('gene_type') or row.get('gene_biotype'),
                chromosome=row['Chromosome'],
                start=int(row['Start']),
                end=int(row['End']),
                strand=row.get('Strand', '.'),
            )

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Sort and merge overlapping/adjacent (start, end) intervals."""
        if not intervals:
            return []
        intervals = sorted(intervals)
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _build_exon_index(self) -> None:
        """Index merged exon coordinates by versionless gene id in one pass.

        Replaces per-gene full-frame filtering: a single ``groupby`` over the
        exon rows populates ``_exon_cache`` for every gene at once, so
        ``_get_exons_for_gene`` is O(1) even on a cold cache. Built lazily the
        first time exons are requested, so gene-only consumers never pay for it.
        """
        if 'Feature' not in self.df.columns:
            self._exon_index_built = True
            return
        exons_df = self.df[self.df['Feature'] == 'exon']
        if not exons_df.empty:
            # Group exon (start, end) pairs by versionless gene id, then merge.
            base_ids = exons_df['gene_id'].str.split('.').str[0].to_numpy()
            starts = exons_df['Start'].astype(int).to_numpy()
            ends = exons_df['End'].astype(int).to_numpy()
            by_gene: dict[str, list[tuple[int, int]]] = {}
            for gid, s, e in zip(base_ids, starts, ends):
                by_gene.setdefault(gid, []).append((int(s), int(e)))
            for gid, ivals in by_gene.items():
                self._exon_cache[gid] = self._merge_intervals(ivals)
        self._exon_index_built = True

    def _get_exons_for_gene(self, gene_id: str) -> list[tuple[int, int]]:
        """Get merged exon coordinates for a gene (0-based (start, end) tuples)."""
        if not self._exon_index_built:
            self._build_exon_index()
        return self._exon_cache.get(gene_id, [])

    def has_exon_annotations(self) -> bool:
        """True if the annotation contains any exon rows (needed for exon masks)."""
        if not self._exon_index_built:
            self._build_exon_index()
        return bool(self._exon_cache)

    def get_gene_info(self, gene_id: str) -> dict[str, Any] | None:
        """Get information for a gene.

        Args:
            gene_id: Gene ID (with or without version)

        Returns:
            Dictionary with gene information, or None if not found
        """
        # Ensure index is built by accessing df
        _ = self.df

        gene_id_base = gene_id.split('.')[0]
        gene_info = self._gene_index.get(gene_id_base)

        if gene_info is None:
            return None

        return {
            'gene_id': gene_info.gene_id,
            'gene_name': gene_info.gene_name,
            'gene_type': gene_info.gene_type,
            'chromosome': gene_info.chromosome,
            'start': gene_info.start,
            'end': gene_info.end,
            'strand': gene_info.strand,
        }

    def get_genes_in_interval(
        self,
        interval: 'Interval',
        gene_types: list[str] | None = None,
    ) -> list[str]:
        """Get gene IDs overlapping an interval.

        Args:
            interval: Genomic interval
            gene_types: Optional list of gene types to include
                (e.g., ['protein_coding', 'lncRNA'])

        Returns:
            List of gene IDs (without version)
        """
        import bisect

        if self._interval_index is None:
            self._build_interval_index()

        genes: list[tuple[int, str]] = []  # (insertion_rank, gene_id)
        for key in self._matching_chrom_keys(interval.chromosome):
            entry = self._interval_index.get(key)
            if entry is None:
                continue
            starts = entry['starts']
            ends = entry['ends']
            ids = entry['ids']
            ranks = entry['ranks']
            # An overlapping gene has start < interval.end and start >=
            # interval.start - max_gene_span (any gene starting earlier than that
            # cannot reach into the interval). Bisect both bounds on the
            # start-sorted array, then filter end > interval.start.
            lo = bisect.bisect_left(starts, interval.start - self._max_gene_span)
            hi = bisect.bisect_left(starts, interval.end)
            for i in range(lo, hi):
                if ends[i] <= interval.start:
                    continue  # start < interval.end already guaranteed
                if gene_types is not None:
                    if self._gene_index[ids[i]].gene_type not in gene_types:
                        continue
                genes.append((ranks[i], ids[i]))

        # Preserve the original _gene_index insertion order of the results.
        genes.sort()
        return [gid for _, gid in genes]

    def _matching_chrom_keys(self, chrom: str) -> list[str]:
        """Index keys matching a query chromosome, tolerating 'chr' prefix diffs.

        Mirrors the original overlap check: a gene on chromosome ``K`` matches a
        query ``Q`` when ``K == Q``, ``K == 'chr'+Q``, or ``Q == 'chr'+K``.
        """
        keys = {chrom, 'chr' + chrom}
        if chrom.startswith('chr'):
            keys.add(chrom[3:])
        return list(keys)

    def _build_interval_index(self) -> None:
        """Per-chromosome, start-sorted gene arrays for O(log G + k) overlap."""
        _ = self.df  # ensure gene index is built
        by_chrom: dict[str, list[tuple[int, int, str, int]]] = {}
        max_span = 0
        for rank, (gene_id, info) in enumerate(self._gene_index.items()):
            by_chrom.setdefault(info.chromosome, []).append(
                (info.start, info.end, gene_id, rank)
            )
            span = info.end - info.start
            if span > max_span:
                max_span = span

        index: dict[str, dict[str, Any]] = {}
        for chrom, entries in by_chrom.items():
            entries.sort(key=lambda e: e[0])  # by start
            index[chrom] = {
                'starts': [e[0] for e in entries],
                'ends': [e[1] for e in entries],
                'ids': [e[2] for e in entries],
                'ranks': [e[3] for e in entries],
            }
        self._interval_index = index
        self._max_gene_span = max_span

    def get_genes_overlapping_variant(
        self,
        variant: 'Variant',
        gene_types: list[str] | None = None,
    ) -> list[str]:
        """Get gene IDs whose body overlaps the variant position.

        Mirrors the upstream JAX behavior: only genes whose body contains
        the variant's 0-based start position are returned. Used by splicing
        scorers to restrict scoring to variant-overlapping genes.

        Args:
            variant: Variant whose 0-based start defines the overlap query
            gene_types: Optional list of gene types to include
                (e.g., ['protein_coding', 'lncRNA'])

        Returns:
            List of gene IDs (without version)
        """
        # Ensure index is built by accessing df
        _ = self.df

        chrom = variant.chromosome
        pos = variant.start  # 0-based
        genes = []

        for gene_id, info in self._gene_index.items():
            # Check chromosome match (handle chr prefix differences)
            if info.chromosome != chrom:
                if info.chromosome == 'chr' + chrom or chrom == 'chr' + info.chromosome:
                    pass
                else:
                    continue

            # Variant 0-based position must lie within [start, end)
            if not (info.start <= pos < info.end):
                continue

            if gene_types is not None and info.gene_type not in gene_types:
                continue

            genes.append(gene_id)

        return genes

    def get_exon_mask(
        self,
        gene_id: str,
        interval: 'Interval',
        resolution: int,
        seq_length: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Create an exon mask for a gene within an interval.

        Args:
            gene_id: Gene ID (with or without version)
            interval: Genomic interval the mask applies to
            resolution: Bin size in base pairs (1 for 1bp, 128 for 128bp)
            seq_length: Number of bins in the sequence
            device: Device for the mask tensor

        Returns:
            Boolean mask tensor of shape (seq_length,) where True = exonic
        """
        mask = torch.zeros(seq_length, dtype=torch.bool, device=device)
        for bin_start, bin_end in self.get_exon_bin_ranges(
            gene_id, interval, resolution, seq_length
        ):
            mask[bin_start:bin_end] = True
        return mask

    def get_exon_bin_ranges(
        self,
        gene_id: str,
        interval: 'Interval',
        resolution: int,
        seq_length: int,
    ) -> list[tuple[int, int]]:
        """Exonic ``[bin_start, bin_end)`` ranges for a gene within an interval.

        The compact form underlying :meth:`get_exon_mask` — a few int pairs per
        gene instead of a dense ``[seq_length]`` tensor. Cheap to cache and to
        rebuild a mask from (see ``aggregation._ExonWindow``).
        """
        exons = self._get_exons_for_gene(gene_id.split('.')[0])
        ranges: list[tuple[int, int]] = []
        for exon_start, exon_end in exons:
            # Interval-relative coordinates
            rel_start = max(0, exon_start - interval.start)
            rel_end = min(interval.width, exon_end - interval.start)
            if rel_start >= interval.width or rel_end <= 0:
                continue
            # Bin coordinates (ceiling division on the end), clamped.
            bin_start = max(0, min(rel_start // resolution, seq_length))
            bin_end = max(0, min((rel_end + resolution - 1) // resolution, seq_length))
            if bin_start < bin_end:
                ranges.append((bin_start, bin_end))
        return ranges

    def get_gene_mask(
        self,
        gene_id: str,
        interval: 'Interval',
        resolution: int,
        seq_length: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Create a full gene body mask (not just exons).

        Args:
            gene_id: Gene ID (with or without version)
            interval: Genomic interval the mask applies to
            resolution: Bin size in base pairs
            seq_length: Number of bins in the sequence
            device: Device for the mask tensor

        Returns:
            Boolean mask tensor of shape (seq_length,) where True = within gene
        """
        gene_info = self.get_gene_info(gene_id)
        if gene_info is None:
            return torch.zeros(seq_length, dtype=torch.bool, device=device)

        mask = torch.zeros(seq_length, dtype=torch.bool, device=device)

        # Convert to interval-relative coordinates
        rel_start = max(0, gene_info['start'] - interval.start)
        rel_end = min(interval.width, gene_info['end'] - interval.start)

        if rel_start >= interval.width or rel_end <= 0:
            return mask

        # Convert to bin coordinates
        bin_start = rel_start // resolution
        bin_end = (rel_end + resolution - 1) // resolution

        bin_start = max(0, min(bin_start, seq_length))
        bin_end = max(0, min(bin_end, seq_length))

        if bin_start < bin_end:
            mask[bin_start:bin_end] = True

        return mask


class PolyAAnnotation:
    """PolyA site annotations from GENCODE polyAs GTF or linked parquet.

    This class loads and queries polyadenylation site annotations from
    GENCODE polyAs files.

    A *linked* parquet (from scripts/preprocess_polya.py) carries Ensembl gene
    IDs, which changes what this class can do rather than just how fast it is:
    PAS are matched to a gene by ID, mirroring the JAX reference. Without them,
    :meth:`get_pas_for_gene` falls back to spatial overlap and
    :meth:`get_total_pas_count_for_gene` returns 0.

    Features read: polyA_site, polyA_signal, pseudo_polyA

    Example:
        >>> polya = PolyAAnnotation('/path/to/gencode.v49.polyAs.linked.parquet')
        >>> pas_positions = polya.get_pas_for_gene(gene_info, interval)
    """

    def __init__(self, polya_path: str | Path):
        """Initialize with path to polyA annotation file.

        Args:
            polya_path: Path to annotation file. Supports:
                - Linked parquet from preprocess_polya.py (carries Ensembl gene
                  IDs; see the class docstring for what they enable)
                - Raw parquet files (.parquet)
                - GTF files (.gtf), requires pyranges
        """
        self.polya_path = Path(polya_path)
        self._df: pd.DataFrame | None = None
        self._has_gene_id: bool | None = None  # Detect on load
        self._gene_id_index: dict[str, pd.DataFrame] | None = None

        # Detect file format
        suffix = self.polya_path.suffix.lower()
        if suffix == '.parquet':
            self._file_format = 'parquet'
        elif suffix in ('.gtf', '.gff', '.gff3'):
            self._file_format = 'gtf'
            try:
                import pyranges as _pr  # noqa: F401
                del _pr
            except ImportError:
                raise ImportError(
                    "pyranges is required to read GTF/GFF files. "
                    "Install with: pip install pyranges\n"
                    "Or pass a .parquet file, which does not need it. "
                    "(scripts/preprocess_polya.py converts a GTF once, but "
                    "itself runs on pyranges.)"
                )
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Expected .parquet or .gtf"
            )

    @property
    def df(self) -> pd.DataFrame:
        """Lazy-loaded polyA annotation DataFrame."""
        if self._df is None:
            if self._file_format == 'parquet':
                self._df = pd.read_parquet(self.polya_path)
            else:
                import pyranges
                pr_obj = pyranges.read_gtf(str(self.polya_path))
                self._df = pr_obj.df
            # Normalize chromosome naming (ensure 'Chromosome' column exists)
            if 'Chromosome' not in self._df.columns and 'chr' in self._df.columns:
                self._df['Chromosome'] = self._df['chr']
            
            # Detect if this is a linked parquet with proper Ensembl gene_ids
            if 'gene_id' in self._df.columns:
                sample_id = str(self._df['gene_id'].iloc[0])
                self._has_gene_id = sample_id.startswith('ENSG')
            else:
                self._has_gene_id = False
            
            # Build gene_id index for fast lookup if available
            if self._has_gene_id:
                self._build_gene_index()
                
        return self._df
    
    def _build_gene_index(self) -> None:
        """Build index mapping gene_id to PAS rows for fast lookup."""
        self._gene_id_index = {}
        # Group by gene_id for fast lookup
        for gene_id, group in self._df.groupby('gene_id'):
            self._gene_id_index[gene_id] = group
    
    @property
    def has_gene_id(self) -> bool:
        """Whether this parquet has linked Ensembl gene IDs."""
        _ = self.df  # Ensure loaded
        return self._has_gene_id or False
    
    def get_total_pas_count_for_gene(self, gene_id: str, strand: str | None = None) -> int:
        """Get total number of PAS sites for a gene (for coverage calculation).
        
        Args:
            gene_id: Gene ID (with or without version)
            strand: Optional strand filter
            
        Returns:
            Total PAS count for the gene
        """
        _ = self.df  # Ensure loaded
        gene_id_base = gene_id.split('.')[0]
        
        if self._has_gene_id and self._gene_id_index:
            if gene_id_base not in self._gene_id_index:
                return 0
            gene_pas = self._gene_id_index[gene_id_base]
            if strand is not None:
                strand_col = 'pas_strand' if 'pas_strand' in gene_pas.columns else 'Strand'
                gene_pas = gene_pas[gene_pas[strand_col] == strand]
            return len(gene_pas)
        return 0  # Cannot count without gene_id linkage

    def get_pas_for_gene(
        self,
        gene_info: dict[str, Any],
        interval: 'Interval',
        downstream_extension: int = 1000,
    ) -> list[int]:
        """Get PAS positions for a gene within an interval.

        If the parquet has linked gene IDs (created by preprocess_polya.py),
        filters by gene_id directly. Otherwise falls back to spatial overlap.

        Args:
            gene_info: Dictionary with gene information (gene_id, start, end, strand).
            interval: Genomic interval to search within.
            downstream_extension: Extension downstream of gene 3' end in bp.
                Default 1000bp.

        Returns:
            List of 0-based PAS positions relative to the interval.
        """
        _ = self.df  # Ensure loaded
        
        gene_strand = gene_info.get('strand', '+')
        gene_id = gene_info.get('gene_id', '').split('.')[0]
        
        # Use gene_id-based filtering if available (matches JAX behavior)
        if self._has_gene_id and self._gene_id_index and gene_id:
            return self._get_pas_by_gene_id(gene_id, gene_strand, interval)
        
        # Fallback to spatial overlap
        return self._get_pas_by_spatial(gene_info, interval, downstream_extension)
    
    def _get_pas_by_gene_id(
        self,
        gene_id: str,
        strand: str,
        interval: 'Interval',
    ) -> list[int]:
        """Get PAS positions by gene_id filtering (JAX-compatible)."""
        if gene_id not in self._gene_id_index:
            return []
        
        gene_pas = self._gene_id_index[gene_id]
        
        # Filter by strand
        strand_col = 'pas_strand' if 'pas_strand' in gene_pas.columns else 'Strand'
        gene_pas = gene_pas[gene_pas[strand_col] == strand]
        
        # Filter by interval
        chrom = interval.chromosome
        chrom_col = gene_pas['Chromosome'].astype(str)
        chrom_match = (
            (chrom_col == chrom) | 
            (chrom_col == 'chr' + chrom) |
            ('chr' + chrom_col == chrom)
        )
        
        mask = (
            chrom_match &
            (gene_pas['Start'] >= interval.start) &
            (gene_pas['Start'] < interval.end)
        )
        
        positions = gene_pas.loc[mask, 'Start'].values
        relative_positions = [int(pos - interval.start) for pos in positions]
        
        return sorted(relative_positions)
    
    def _get_pas_by_spatial(
        self,
        gene_info: dict[str, Any],
        interval: 'Interval',
        downstream_extension: int,
    ) -> list[int]:
        """Get PAS positions by spatial overlap (fallback method)."""
        gene_strand = gene_info.get('strand', '+')
        gene_start = gene_info['start']
        gene_end = gene_info['end']

        # Expand search region downstream of gene 3' end
        if gene_strand == '+':
            search_end = gene_end + downstream_extension
            search_start = gene_start
        else:
            search_start = gene_start - downstream_extension
            search_end = gene_end

        # Filter polyA sites
        df = self.df
        chrom = interval.chromosome

        # Handle chromosome prefix differences
        chrom_col = df['Chromosome'].astype(str)
        chrom_match = (
            (chrom_col == chrom) | 
            (chrom_col == 'chr' + chrom) |
            ('chr' + chrom_col == chrom)
        )
        strand_col = 'pas_strand' if 'pas_strand' in df.columns else 'Strand'
        mask = (
            chrom_match &
            (df[strand_col] == gene_strand) &
            (df['Start'] >= max(search_start, interval.start)) &
            (df['Start'] < min(search_end, interval.end))
        )

        positions = df.loc[mask, 'Start'].values
        relative_positions = [int(pos - interval.start) for pos in positions]

        return sorted(relative_positions)

    def get_pas_in_interval(
        self,
        interval: 'Interval',
        strand: str | None = None,
    ) -> list[tuple[int, str]]:
        """Get all PAS positions within an interval.

        Args:
            interval: Genomic interval to search within.
            strand: Optional strand filter ('+' or '-').

        Returns:
            List of (position, strand) tuples where position is 0-based
            relative to interval start.
        """
        df = self.df
        chrom = interval.chromosome

        # Handle chromosome prefix differences
        chrom_col = df['Chromosome'].astype(str)
        chrom_match = (
            (chrom_col == chrom) | 
            (chrom_col == 'chr' + chrom) |
            ('chr' + chrom_col == chrom)
        )
        mask = (
            chrom_match &
            (df['Start'] >= interval.start) &
            (df['Start'] < interval.end)
        )

        if strand is not None:
            strand_col = 'pas_strand' if 'pas_strand' in df.columns else 'Strand'
            mask &= (df[strand_col] == strand)

        result = []
        strand_col = 'pas_strand' if 'pas_strand' in df.columns else 'Strand'
        for _, row in df.loc[mask].iterrows():
            rel_pos = int(row['Start'] - interval.start)
            pas_strand = row.get(strand_col, '.')
            result.append((rel_pos, pas_strand))

        return sorted(result, key=lambda x: x[0])
