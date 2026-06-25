"""Genome-coordinate types and FASTA-backed sequence access."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from alphagenome_pytorch.utils.sequence import (
    onehot_tensor_to_sequence,
    sequence_to_onehot,
    sequence_to_onehot_tensor,
)

if TYPE_CHECKING:
    import pyfaidx


class Width(IntEnum):
    """Supported AlphaGenome interval widths.

    IntEnum so it works anywhere a raw base-pair integer is expected.
    """

    W_2KB = 2 * 1024
    W_4KB = 4 * 1024
    W_8KB = 8 * 1024
    W_16KB = 16 * 1024
    W_100KB = 128 * 1024
    W_300KB = 256 * 1024
    W_500KB = 512 * 1024
    W_1MB = 1024 * 1024

    @classmethod
    def normalize(cls, value: Union["Width", int, str]) -> int:
        names = {
            "2KB": cls.W_2KB,
            "4KB": cls.W_4KB,
            "8KB": cls.W_8KB,
            "16KB": cls.W_16KB,
            "100KB": cls.W_100KB,
            "300KB": cls.W_300KB,
            "500KB": cls.W_500KB,
            "1MB": cls.W_1MB,
        }
        if isinstance(value, int):
            if value in [int(v) for v in names.values()]:
                return int(value)
            raise ValueError(
                f"{value} is not a supported width. "
                f"Choose from: {', '.join(names)}"
            )
        if isinstance(value, str):
            key = value.strip().upper()
            if key in names:
                return int(names[key])
            prefixed = key.removeprefix("W_")
            if prefixed in names:
                return int(names[prefixed])
            raise ValueError(
                f"Invalid width {value!r}. Choose from: {', '.join(names)}"
            )
        raise TypeError(f"Unsupported width type: {type(value)}")


@dataclass(frozen=True)
class Interval:
    """Genomic interval using 0-based, half-open coordinates."""

    chromosome: str
    start: int
    end: int
    strand: str = "."
    name: str = ""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Start position must be non-negative, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"End ({self.end}) must be greater than start ({self.start})")
        if self.strand not in ("+", "-", "."):
            raise ValueError(f"Strand must be '+', '-', or '.', got {self.strand!r}")

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    def contains(self, position: int) -> bool:
        return self.start <= position < self.end

    def __str__(self) -> str:
        if self.strand == ".":
            return f"{self.chromosome}:{self.start}-{self.end}"
        return f"{self.chromosome}:{self.start}-{self.end}:{self.strand}"

    @classmethod
    def from_str(cls, value: str) -> "Interval":
        match = re.match(r"^([^:]+):(\d+)-(\d+):([+\-.])$", value)
        if match:
            return cls(match.group(1), int(match.group(2)), int(match.group(3)), match.group(4))
        match = re.match(r"^([^:]+):(\d+)-(\d+)$", value)
        if match:
            return cls(match.group(1), int(match.group(2)), int(match.group(3)))
        raise ValueError(f"Could not parse interval string: {value!r}")

    @classmethod
    def centered_on(
        cls,
        chromosome: str,
        position: int,
        width: Union[Width, int, str] = Width.W_100KB,
    ) -> "Interval":
        normalized = Width.normalize(width)
        half_width = normalized // 2
        return cls(
            chromosome=chromosome,
            start=max(0, position - half_width),
            end=position + half_width + (normalized % 2),
        )


@dataclass(frozen=True)
class Variant:
    """Genomic variant using VCF-style 1-based position."""

    chromosome: str
    position: int
    reference_bases: str
    alternate_bases: str
    name: str = ""

    def __post_init__(self) -> None:
        if self.position < 1:
            raise ValueError(f"Position must be >= 1 (VCF convention), got {self.position}")
        if not self.reference_bases:
            raise ValueError("Reference bases cannot be empty")
        if not self.alternate_bases:
            raise ValueError("Alternate bases cannot be empty")
        object.__setattr__(self, "reference_bases", self.reference_bases.upper())
        object.__setattr__(self, "alternate_bases", self.alternate_bases.upper())

    @property
    def start(self) -> int:
        return self.position - 1

    @property
    def end(self) -> int:
        return self.start + len(self.reference_bases)

    @property
    def is_snv(self) -> bool:
        return len(self.reference_bases) == 1 and len(self.alternate_bases) == 1

    @property
    def is_insertion(self) -> bool:
        return len(self.alternate_bases) > len(self.reference_bases)

    @property
    def is_deletion(self) -> bool:
        return len(self.alternate_bases) < len(self.reference_bases)

    @property
    def is_indel(self) -> bool:
        return self.is_insertion or self.is_deletion

    def __str__(self) -> str:
        return f"{self.chromosome}:{self.position}:{self.reference_bases}>{self.alternate_bases}"

    @classmethod
    def from_str(cls, value: str, format: str = "default") -> "Variant":
        if format == "default":
            match = re.match(r"^([^:]+):(\d+):([ACGTN]+)>([ACGTN]+)$", value, re.IGNORECASE)
            if match:
                return cls(
                    chromosome=match.group(1),
                    position=int(match.group(2)),
                    reference_bases=match.group(3),
                    alternate_bases=match.group(4),
                )
            raise ValueError(f"Could not parse variant string: {value!r}")
        if format == "gtex":
            parts = value.split("_")
            if len(parts) >= 4:
                return cls(parts[0], int(parts[1]), parts[2], parts[3])
            raise ValueError(f"Could not parse GTEx variant string: {value!r}")
        if format == "gnomad":
            parts = value.split("-")
            if len(parts) >= 4:
                return cls(parts[0], int(parts[1]), parts[2], parts[3])
            raise ValueError(f"Could not parse gnomAD variant string: {value!r}")
        raise ValueError(f"Unknown format: {format!r}")


def _coerce_interval(interval: object) -> Interval:
    return Interval(
        chromosome=getattr(interval, "chromosome"),
        start=int(getattr(interval, "start")),
        end=int(getattr(interval, "end")),
        strand=getattr(interval, "strand", "."),
        name=getattr(interval, "name", ""),
    )


def _coerce_variant(variant: object) -> Variant:
    return Variant(
        chromosome=getattr(variant, "chromosome"),
        position=int(getattr(variant, "position")),
        reference_bases=getattr(variant, "reference_bases"),
        alternate_bases=getattr(variant, "alternate_bases"),
        name=getattr(variant, "name", ""),
    )


def apply_variant_to_sequence(sequence: str, variant: object, interval: object) -> str:
    """Apply a variant to a reference sequence."""
    v = _coerce_variant(variant)
    i = _coerce_interval(interval)
    if v.chromosome != i.chromosome:
        raise ValueError(
            f"Variant chromosome ({v.chromosome}) doesn't match interval chromosome ({i.chromosome})"
        )

    var_start = v.start - i.start
    var_end = var_start + len(v.reference_bases)
    if var_start < 0 or var_end > len(sequence):
        raise ValueError(
            f"Variant position {v.position} (ref length {len(v.reference_bases)}) "
            f"is outside interval {i}"
        )

    seq_ref = sequence[var_start:var_end].upper()
    if seq_ref != v.reference_bases:
        raise ValueError(
            f"Reference allele mismatch at {v.chromosome}:{v.position}. "
            f"Expected {v.reference_bases!r}, found {seq_ref!r} in sequence"
        )
    return sequence[:var_start] + v.alternate_bases + sequence[var_end:]


def apply_variant_to_onehot(onehot: torch.Tensor, variant: object, interval: object) -> torch.Tensor:
    """Apply a variant to a one-hot encoded sequence."""
    v = _coerce_variant(variant)
    i = _coerce_interval(interval)
    if v.is_snv:
        base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        var_pos = v.start - i.start
        if var_pos < 0 or var_pos >= onehot.shape[0]:
            raise ValueError(f"Variant position {v.position} is outside interval {i}")
        alt_onehot = onehot.clone()
        alt_onehot[var_pos] = 0.0
        alt_onehot[var_pos, base_to_idx[v.alternate_bases]] = 1.0
        return alt_onehot

    ref_seq = onehot_tensor_to_sequence(onehot)
    alt_seq = apply_variant_to_sequence(ref_seq, v, i)
    return sequence_to_onehot_tensor(alt_seq, dtype=onehot.dtype, device=onehot.device)


class GenomeSequenceSource:
    """FASTA-backed sequence source with optional chromosome-level cache."""

    def __init__(
        self,
        fasta_path: str | Path,
        *,
        chromosomes: set[str] | None = None,
        cache: bool = False,
        ambiguous: str = "zero",
        verbose: bool = False,
    ):
        try:
            import pyfaidx
        except ImportError as exc:
            raise ImportError(
                "pyfaidx is required for FASTA extraction. Install with: pip install pyfaidx"
            ) from exc

        self.fasta_path = str(fasta_path)
        self.ambiguous = ambiguous
        self._pyfaidx = pyfaidx
        self._fasta: pyfaidx.Fasta | None = None
        self._owner_pid: int | None = None
        self._cache: dict[str, np.ndarray] = {}
        self.chrom_sizes: dict[str, int] = {}

        fasta = pyfaidx.Fasta(self.fasta_path)
        try:
            for ref in fasta.keys():
                self.chrom_sizes[ref] = len(fasta[ref])
            if cache:
                refs_to_load = chromosomes if chromosomes else set(fasta.keys())
                for ref in refs_to_load:
                    if ref in self.chrom_sizes:
                        self._cache[ref] = sequence_to_onehot(
                            str(fasta[ref][:]), ambiguous=ambiguous
                        )
        finally:
            fasta.close()

        if verbose and cache:
            cached_mb = sum(arr.nbytes for arr in self._cache.values()) / 1e6
            print(f"Cached genome: loaded {len(self._cache)} chromosomes ({cached_mb:.1f} MB)")

    @property
    def fasta(self) -> "pyfaidx.Fasta":
        current_pid = os.getpid()
        if self._fasta is None or self._owner_pid != current_pid:
            if self._fasta is not None:
                try:
                    self._fasta.close()
                except Exception:
                    pass
            self._fasta = self._pyfaidx.Fasta(self.fasta_path)
            self._owner_pid = current_pid
        return self._fasta

    def _resolve_chrom(self, chrom: str) -> str:
        if chrom in self.chrom_sizes:
            return chrom
        alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alt in self.chrom_sizes:
            return alt
        raise ValueError(f"Chromosome {chrom} not found in FASTA file")

    def fetch_sequence(self, interval: object, *, variant: object | None = None) -> str:
        i = _coerce_interval(interval)
        chrom = self._resolve_chrom(i.chromosome)
        seq = str(self.fasta[chrom][i.start:i.end]).upper()
        if variant is not None:
            seq = apply_variant_to_sequence(seq, variant, i)
        return seq

    def fetch_onehot(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        ambiguous: str | None = None,
        pad: bool = False,
        copy: bool = True,
    ) -> np.ndarray:
        policy = ambiguous or self.ambiguous
        seq_len = end - start
        resolved = self._resolve_chrom(chrom)
        chrom_len = self.chrom_sizes[resolved]

        cached = resolved in self._cache

        def _encode(lo: int, hi: int) -> np.ndarray:
            if cached:
                return self._cache[resolved][lo:hi]
            return sequence_to_onehot(
                str(self.fasta[resolved][lo:hi]), ambiguous=policy
            )

        if start >= 0 and end <= chrom_len:
            seq = _encode(start, end)
            return seq.copy() if (copy and cached) else seq

        if not pad:
            raise ValueError(f"Requested interval {chrom}:{start}-{end} is out of bounds")

        # Out-of-bounds positions are padded per the ``ambiguous`` policy:
        # ``"uniform"`` fills 0.25 (float32), ``"zero"`` fills zeros (uint8),
        # matching the encoding of in-bounds ambiguous bases.
        if policy == "uniform":
            result = np.full((seq_len, 4), 0.25, dtype=np.float32)
        else:
            result = np.zeros((seq_len, 4), dtype=np.uint8)
        valid_start = max(0, start)
        valid_end = min(chrom_len, end)
        if valid_start < valid_end:
            seq = _encode(valid_start, valid_end)
            dest_start = valid_start - start
            result[dest_start:dest_start + (valid_end - valid_start)] = seq
        return result

    def close(self) -> None:
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None
            self._owner_pid = None

    def __enter__(self) -> "GenomeSequenceSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def extract_sequence_from_fasta(fasta_path: str | Path, interval: object) -> str:
    with GenomeSequenceSource(fasta_path) as source:
        return source.fetch_sequence(interval)


__all__ = [
    "Width",
    "Interval",
    "Variant",
    "GenomeSequenceSource",
    "apply_variant_to_sequence",
    "apply_variant_to_onehot",
    "extract_sequence_from_fasta",
]
