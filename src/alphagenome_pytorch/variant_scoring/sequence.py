"""DNA sequence utilities for variant scoring.

This module provides functions for:
- Converting DNA sequences to one-hot encoding
- Applying variants to sequences
- Extracting sequences from FASTA files

The generic implementations now live in :mod:`alphagenome_pytorch.genome`.
This module remains as a compatibility layer for existing imports.
"""

from __future__ import annotations

from pathlib import Path
from alphagenome_pytorch.utils.sequence import (
    sequence_to_onehot_tensor as sequence_to_onehot,
    onehot_tensor_to_sequence as onehot_to_sequence,
)
from alphagenome_pytorch.genome import (
    GenomeSequenceSource,
    apply_variant_to_onehot,
    apply_variant_to_sequence,
    extract_sequence_from_fasta,
)
from .types import Interval, Variant


class FastaExtractor:
    """Extract sequences from a FASTA file.

    Uses pyfaidx for efficient indexed access.

    Example:
        >>> extractor = FastaExtractor('/path/to/genome.fa')
        >>> interval = Interval('chr22', 36136162, 36267234)
        >>> seq = extractor.extract(interval)
        >>> len(seq)
        131072
    """

    def __init__(self, fasta_path: str | Path):
        """Initialize with path to FASTA file.

        Args:
            fasta_path: Path to FASTA file. Will create .fai index if not present.
        """
        self.source = GenomeSequenceSource(fasta_path)
        self.fasta_path = str(fasta_path)

    @property
    def fasta(self):
        """Lazy-loaded FASTA file handle."""
        return self.source.fasta

    def extract(self, interval: Interval) -> str:
        """Extract sequence for a genomic interval.

        Args:
            interval: Genomic interval to extract

        Returns:
            DNA sequence string (uppercase)
        """
        return self.source.fetch_sequence(interval)

    def extract_with_variant(
        self,
        interval: Interval,
        variant: Variant,
    ) -> tuple[str, str]:
        """Extract both reference and alternate sequences.

        Args:
            interval: Genomic interval to extract
            variant: Variant to apply

        Returns:
            Tuple of (reference_sequence, alternate_sequence)
        """
        ref_seq = self.extract(interval)
        alt_seq = apply_variant_to_sequence(ref_seq, variant, interval)
        return ref_seq, alt_seq

    def close(self):
        """Close the FASTA file handle."""
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
