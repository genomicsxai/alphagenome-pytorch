"""DNA sequence utilities for variant scoring.

This module provides functions for:
- Converting DNA sequences to one-hot encoding
- Applying variants to sequences
- Extracting sequences from FASTA files
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from alphagenome_pytorch.utils.sequence import (
    sequence_to_onehot_tensor as sequence_to_onehot,
    onehot_tensor_to_sequence as onehot_to_sequence,
)
from .types import Interval, Variant

# DNA nucleotide to index mapping (used by apply_variant_to_onehot for SNVs)
_NUCLEOTIDE_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

if TYPE_CHECKING:
    import pyfaidx


def apply_variant_to_sequence(
    sequence: str,
    variant: Variant,
    interval: Interval,
) -> str:
    """Apply a variant to a DNA sequence.

    Args:
        sequence: Reference DNA sequence string
        variant: Variant to apply
        interval: The genomic interval that the sequence corresponds to

    Returns:
        Alternate sequence with the variant applied

    Raises:
        ValueError: If variant position is outside the interval, or if
            the reference allele doesn't match the sequence

    Example:
        >>> seq = 'AAACCCGGG'
        >>> interval = Interval('chr1', 0, 9)
        >>> variant = Variant('chr1', 4, 'C', 'T')  # Position 4 (1-based) = index 3
        >>> apply_variant_to_sequence(seq, variant, interval)
        'AAATCCGGG'
    """
    # Validate variant is within interval
    if variant.chromosome != interval.chromosome:
        raise ValueError(
            f"Variant chromosome ({variant.chromosome}) doesn't match "
            f"interval chromosome ({interval.chromosome})"
        )

    # Convert variant position to sequence coordinates (0-based)
    var_start = variant.start - interval.start  # variant.start is already 0-based
    var_end = var_start + len(variant.reference_bases)

    if var_start < 0 or var_end > len(sequence):
        raise ValueError(
            f"Variant position {variant.position} (ref length {len(variant.reference_bases)}) "
            f"is outside interval {interval}"
        )

    # Verify reference matches
    seq_ref = sequence[var_start:var_end].upper()
    if seq_ref != variant.reference_bases.upper():
        raise ValueError(
            f"Reference allele mismatch at {variant.chromosome}:{variant.position}. "
            f"Expected '{variant.reference_bases}', found '{seq_ref}' in sequence"
        )

    # Apply the variant
    alt_sequence = sequence[:var_start] + variant.alternate_bases + sequence[var_end:]

    return alt_sequence


def apply_variant_to_onehot(
    onehot: torch.Tensor,
    variant: Variant,
    interval: Interval,
) -> torch.Tensor:
    """Apply a variant to a one-hot encoded sequence.

    For SNVs, this is efficient and doesn't require converting to/from string.
    For indels, the sequence is converted to string, modified, and re-encoded.

    Args:
        onehot: One-hot encoded reference sequence of shape (L, 4)
        variant: Variant to apply
        interval: The genomic interval that the sequence corresponds to

    Returns:
        One-hot encoded alternate sequence. For SNVs, shape is (L, 4).
        For indels, shape may differ due to insertion/deletion.
    """
    if variant.is_snv:
        # Efficient in-place modification for SNVs
        var_pos = variant.start - interval.start  # 0-based position in sequence

        if var_pos < 0 or var_pos >= onehot.shape[0]:
            raise ValueError(
                f"Variant position {variant.position} is outside interval {interval}"
            )

        alt_onehot = onehot.clone()
        alt_onehot[var_pos] = 0.0
        alt_onehot[var_pos, _NUCLEOTIDE_TO_INDEX[variant.alternate_bases.upper()]] = 1.0

        return alt_onehot
    else:
        # For indels, convert to string, apply, and re-encode
        ref_seq = onehot_to_sequence(onehot)
        alt_seq = apply_variant_to_sequence(ref_seq, variant, interval)
        return sequence_to_onehot(alt_seq, dtype=onehot.dtype, device=onehot.device)


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

    def __init__(self, fasta_path: str):
        """Initialize with path to FASTA file.

        Args:
            fasta_path: Path to FASTA file. Will create .fai index if not present.
        """
        try:
            import pyfaidx
        except ImportError:
            raise ImportError(
                "pyfaidx is required for FASTA extraction. "
                "Install with: pip install pyfaidx"
            )

        self.fasta_path = fasta_path
        self._fasta: pyfaidx.Fasta | None = None

    @property
    def fasta(self) -> 'pyfaidx.Fasta':
        """Lazy-loaded FASTA file handle."""
        if self._fasta is None:
            import pyfaidx
            self._fasta = pyfaidx.Fasta(self.fasta_path)
        return self._fasta

    def extract(self, interval: Interval) -> str:
        """Extract sequence for a genomic interval.

        Args:
            interval: Genomic interval to extract

        Returns:
            DNA sequence string (uppercase)
        """
        chrom = interval.chromosome
        if chrom not in self.fasta:
            # Try with/without 'chr' prefix
            if chrom.startswith('chr'):
                chrom = chrom[3:]
            else:
                chrom = 'chr' + chrom

            if chrom not in self.fasta:
                raise ValueError(
                    f"Chromosome {interval.chromosome} not found in FASTA file"
                )

        seq = str(self.fasta[chrom][interval.start:interval.end])
        return seq.upper()

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
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def extract_sequence_from_fasta(
    fasta_path: str,
    interval: Interval,
) -> str:
    """Convenience function to extract a sequence from a FASTA file.

    Args:
        fasta_path: Path to FASTA file
        interval: Genomic interval to extract

    Returns:
        DNA sequence string (uppercase)
    """
    with FastaExtractor(fasta_path) as extractor:
        return extractor.extract(interval)
