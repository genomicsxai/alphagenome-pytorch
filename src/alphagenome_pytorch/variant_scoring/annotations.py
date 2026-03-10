"""Gene annotation utilities for variant scoring.

This module provides classes for loading and querying gene annotations
from GTF/GFF or Parquet files for use with gene-centric variant scorers.

For best performance, convert GTF files to Parquet format using:
    python scripts/convert_gtf_to_parquet.py --input annotation.gtf --output annotation.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

if TYPE_CHECKING:
    from .types import Interval


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


class GeneAnnotation:
    """Load and query gene/exon annotations from GTF or Parquet files.

    Supports both GTF/GFF files (using pyranges) and pre-converted Parquet files.
    Parquet files load ~50-100x faster than GTF files.

    To convert GTF to Parquet:
        python scripts/convert_gtf_to_parquet.py --input annotation.gtf --output annotation.parquet

    Example:
        >>> # Fast loading from Parquet (recommended)
        >>> annotation = GeneAnnotation('/path/to/gencode.parquet')
        >>> genes = annotation.get_genes_in_interval(interval)
        >>>
        >>> # Traditional GTF loading (slower)
        >>> annotation = GeneAnnotation('/path/to/gencode.gtf')
    """

    def __init__(self, annotation_path: str | Path):
        """Initialize with path to annotation file.

        Args:
            annotation_path: Path to annotation file. Supports:
                - Parquet files (.parquet) - fast loading, recommended
                - GTF/GFF files (.gtf, .gff, .gff3) - slow, requires pyranges
        """
        self.annotation_path = Path(annotation_path)
        self._df: pd.DataFrame | None = None
        self._gene_index: dict[str, GeneInfo] = {}
        self._exon_cache: dict[str, list[tuple[int, int]]] = {}

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
                    "pyranges is required for GTF files. "
                    "Install with: pip install pyranges\n"
                    "Or convert to Parquet for faster loading: "
                    "python scripts/convert_gtf_to_parquet.py"
                )
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Expected .parquet, .gtf, .gff, or .gff3"
            )

    # Keep gtf_path as alias for backward compatibility
    @property
    def gtf_path(self) -> Path:
        """Alias for annotation_path (backward compatibility)."""
        return self.annotation_path

    @property
    def df(self) -> pd.DataFrame:
        """Lazy-loaded annotation DataFrame."""
        if self._df is None:
            if self._file_format == 'parquet':
                self._load_from_parquet()
            else:
                self._load_from_gtf()
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

    def _get_exons_for_gene(self, gene_id: str) -> list[tuple[int, int]]:
        """Get exon coordinates for a gene.

        Args:
            gene_id: Gene ID (without version)

        Returns:
            List of (start, end) tuples for exons (0-based coordinates)
        """
        if gene_id in self._exon_cache:
            return self._exon_cache[gene_id]

        # Filter for exons of this gene
        # Match both versioned and unversioned gene IDs
        exons_df = self.df[self.df['Feature'] == 'exon']
        exons_df = exons_df[exons_df['gene_id'].str.split('.').str[0] == gene_id]

        exons = []
        for _, row in exons_df.iterrows():
            # Coordinates are 0-based
            start = int(row['Start'])
            end = int(row['End'])
            exons.append((start, end))

        # Merge overlapping exons
        if exons:
            exons.sort()
            merged = [exons[0]]
            for start, end in exons[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            exons = merged

        self._exon_cache[gene_id] = exons
        return exons

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
        # Ensure index is built by accessing df
        _ = self.df

        genes = []
        chrom = interval.chromosome

        for gene_id, info in self._gene_index.items():
            # Check chromosome match (handle chr prefix)
            if info.chromosome != chrom:
                if info.chromosome == 'chr' + chrom or chrom == 'chr' + info.chromosome:
                    pass  # Match with chr prefix difference
                else:
                    continue

            # Check overlap
            if info.end <= interval.start or info.start >= interval.end:
                continue

            # Check gene type if specified
            if gene_types is not None:
                if info.gene_type not in gene_types:
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
        gene_id_base = gene_id.split('.')[0]
        exons = self._get_exons_for_gene(gene_id_base)

        mask = torch.zeros(seq_length, dtype=torch.bool, device=device)

        for exon_start, exon_end in exons:
            # Convert to interval-relative coordinates
            rel_start = max(0, exon_start - interval.start)
            rel_end = min(interval.width, exon_end - interval.start)

            if rel_start >= interval.width or rel_end <= 0:
                continue

            # Convert to bin coordinates
            bin_start = rel_start // resolution
            bin_end = (rel_end + resolution - 1) // resolution  # Ceiling division

            # Clamp to sequence length
            bin_start = max(0, min(bin_start, seq_length))
            bin_end = max(0, min(bin_end, seq_length))

            if bin_start < bin_end:
                mask[bin_start:bin_end] = True

        return mask

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
    GENCODE polyAs files. For best results, use a linked parquet created
    by scripts/preprocess_polya.py which contains proper Ensembl gene IDs.

    Features read: polyA_site, polyA_signal, pseudo_polyA

    Example:
        >>> polya = PolyAAnnotation('/path/to/gencode.v49.polyAs.linked.parquet')
        >>> pas_positions = polya.get_pas_for_gene(gene_info, interval)
    """

    def __init__(self, polya_path: str | Path):
        """Initialize with path to polyA annotation file.

        Args:
            polya_path: Path to annotation file. Supports:
                - Linked parquet files (recommended, created by preprocess_polya.py)
                - Raw parquet files (.parquet)
                - GTF files (.gtf) - slower, requires pyranges
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
                    "pyranges is required for GTF files. "
                    "Install with: pip install pyranges\n"
                    "Or convert to Parquet for faster loading."
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
