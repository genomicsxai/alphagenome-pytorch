"""Datasets for AlphaGenome fine-tuning.

Provides Dataset classes for loading genomic data from pre-split BED files.
Each split (train/val) should be provided as a separate BED file.

Example:
    >>> from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset
    >>> train_ds = GenomicDataset(
    ...     genome_fasta='hg38.fa',
    ...     bigwig_files=['signal.bw'],
    ...     bed_file='train_positions.bed',
    ... )

For faster data loading (recommended for training), use caching:
    >>> train_ds = GenomicDataset(
    ...     genome_fasta='hg38.fa',
    ...     bigwig_files=['signal.bw'],
    ...     bed_file='train_positions.bed',
    ...     cache_genome=True,   # Prefetch genome into memory
    ...     cache_signals=True,  # Prefetch bigwig signals into memory
    ... )

Multi-head example:
    >>> from alphagenome_pytorch.extensions.finetuning.datasets import (
    ...     MultiHeadGenomicDataset, HeadDataConfig
    ... )
    >>> head_configs = [
    ...     HeadDataConfig(name='atac', bigwig_files=['atac.bw']),
    ...     HeadDataConfig(name='rna_seq', bigwig_files=['rna.bw']),
    ... ]
    >>> train_ds = MultiHeadGenomicDataset(
    ...     genome_fasta='hg38.fa',
    ...     bed_file='positions.bed',
    ...     head_configs=head_configs,
    ... )
"""

from __future__ import annotations
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from alphagenome_pytorch.utils.sequence import sequence_to_onehot

# Lazy imports for pyfaidx/pyBigWig to avoid import errors when not needed
pyfaidx = None
pyBigWig = None


def _ensure_genomic_deps():
    """Lazily import pyfaidx and pyBigWig."""
    global pyfaidx, pyBigWig
    if pyfaidx is None:
        import pyfaidx as _pyfaidx
        pyfaidx = _pyfaidx
    if pyBigWig is None:
        import pyBigWig as _pyBigWig
        pyBigWig = _pyBigWig


# Default max threads for parallel bigwig I/O (actual = min(n_files, this))
DEFAULT_MAX_IO_WORKERS = 16




class CachedGenome:
    """Memory-cached genome for fast sequence retrieval.

    Loads and caches entire chromosomes as one-hot encoded arrays.
    This provides ~2-10x speedup over pyfaidx for repeated access patterns.

    Memory usage: ~4 bytes per base pair (uint8 one-hot, 4 columns).
    For hg38 (~3.1 billion bp), this uses ~12 GB of RAM.

    Args:
        fasta_path: Path to the genome FASTA file.
        chromosomes: Optional set of chromosomes to load. If None, loads all.
    """

    def __init__(self, fasta_path: str, chromosomes: set[str] | None = None):
        _ensure_genomic_deps()
        self.fasta_path = fasta_path
        self._cache: dict[str, np.ndarray] = {}
        self.chrom_sizes: dict[str, int] = {}

        print(f"CachedGenome: Loading genome from {fasta_path}...")
        fasta = pyfaidx.Fasta(fasta_path)
        try:
            refs_to_load = chromosomes if chromosomes else set(fasta.keys())

            for ref in fasta.keys():
                length = len(fasta[ref])
                self.chrom_sizes[ref] = length

                if ref in refs_to_load:
                    # Fetch and encode the entire chromosome
                    seq_str = str(fasta[ref][:])
                    self._cache[ref] = sequence_to_onehot(seq_str)
        finally:
            fasta.close()

        cached_size_mb = sum(arr.nbytes for arr in self._cache.values()) / 1e6
        print(f"CachedGenome: Loaded {len(self._cache)} chromosomes ({cached_size_mb:.1f} MB)")

    def fetch(self, chrom: str, start: int, end: int, copy: bool = True) -> np.ndarray:
        """Fetch one-hot encoded sequence for a region.

        Args:
            chrom: Chromosome name.
            start: Start position (0-based, inclusive).
            end: End position (0-based, exclusive).
            copy: If return the copy of the underlying array (default: True).

        Returns:
            One-hot encoded array of shape (end - start, 4).
        """
        if chrom in self._cache:
            return self._cache[chrom][start:end].copy() if copy else self._cache[chrom][start:end]
        else:
            raise KeyError(f"Chromosome {chrom} not in cache. "
                          f"Available: {list(self._cache.keys())}")


class CachedBigWig:
    """Memory-cached BigWig for fast signal retrieval.

    Loads and caches signal data for specified chromosomes.

    Args:
        bigwig_path: Path to the BigWig file.
        chromosomes: Set of chromosomes to load.
        chrom_sizes: Dict mapping chromosome names to sizes.
    """

    def __init__(
        self,
        bigwig_path: str,
        chromosomes: set[str],
        chrom_sizes: dict[str, int],
    ):
        _ensure_genomic_deps()
        self.bigwig_path = bigwig_path
        self._cache: dict[str, np.ndarray] = {}

        with pyBigWig.open(bigwig_path) as bw:
            bw_chroms = set(bw.chroms().keys())

            for chrom in chromosomes:
                if chrom not in bw_chroms:
                    # Store zeros for missing chromosomes
                    self._cache[chrom] = np.zeros(chrom_sizes[chrom], dtype=np.float32)
                    continue

                size = chrom_sizes[chrom]
                try:
                    values = bw.values(chrom, 0, size, numpy=True)
                    if values is None:
                        values = np.zeros(size, dtype=np.float32)
                    else:
                        values = np.asarray(values, dtype=np.float32)
                        values = np.nan_to_num(values, nan=0.0)
                    self._cache[chrom] = values
                except Exception:
                    self._cache[chrom] = np.zeros(size, dtype=np.float32)

    def values(self, chrom: str, start: int, end: int, copy: bool = True) -> np.ndarray:
        """Get signal values for a region.

        Args:
            chrom: Chromosome name.
            start: Start position (0-based, inclusive).
            end: End position (0-based, exclusive).
            copy: If return the copy of the underlying array (default: True).

        Returns:
            Signal values array of shape (end - start,).
        """
        if chrom in self._cache:
            return self._cache[chrom][start:end].copy() if copy else self._cache[chrom][start:end]
        else:
            raise KeyError(f"Chromosome {chrom} not in cache")


class MmapBigWig:
    """Memory-mapped signal storage for fast retrieval.

    Reads pre-converted signal data from memory-mapped numpy files.
    Use `convert_bigwigs_to_mmap()` to create these files from BigWig.

    Args:
        mmap_dir: Directory containing .npy files and metadata.json.
    """

    def __init__(self, mmap_dir: str | Path):
        mmap_dir = Path(mmap_dir)
        if not mmap_dir.exists():
            raise FileNotFoundError(f"Mmap directory not found: {mmap_dir}")

        # Load metadata
        metadata_path = mmap_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {mmap_dir}. "
                "Use convert_bigwigs_to_mmap() to create mmap files."
            )

        with open(metadata_path) as f:
            self._metadata = json.load(f)

        self._mmap_dir = mmap_dir
        self._mmaps: dict[str, np.ndarray] = {}

        # Load memory-mapped arrays for each chromosome
        for chrom, info in self._metadata["chromosomes"].items():
            npy_path = mmap_dir / info["file"]
            if npy_path.exists():
                self._mmaps[chrom] = np.load(npy_path, mmap_mode="r")

    def values(self, chrom: str, start: int, end: int, copy: bool = True) -> np.ndarray:
        """Get signal values for a region.

        Args:
            chrom: Chromosome name.
            start: Start position (0-based, inclusive).
            end: End position (0-based, exclusive).
            copy: If return the copy of the underlying array (default: True).

        Returns:
            Signal values array of shape (end - start,).
        """
        if chrom in self._mmaps:
            # Copy from mmap to avoid holding reference
            return self._mmaps[chrom][start:end].copy() if copy else self._mmaps[chrom][start:end]
        else:
            # Return zeros for missing chromosomes
            return np.zeros(end - start, dtype=np.float32)

    @property
    def chromosomes(self) -> set[str]:
        """Available chromosomes."""
        return set(self._mmaps.keys())


def convert_bigwig_to_mmap(
    bigwig_path: str | Path,
    output_dir: str | Path,
    chromosomes: list[str] | None = None,
    dtype: np.dtype = np.float32,
) -> Path:
    """Convert a BigWig file to memory-mapped numpy arrays.

    Creates a directory with per-chromosome .npy files and metadata.json.

    Args:
        bigwig_path: Path to input BigWig file.
        output_dir: Directory to write output files.
        chromosomes: Optional list of chromosomes to convert. If None, converts all.
        dtype: Data type for output arrays (default: float32).

    Returns:
        Path to output directory.
    """
    _ensure_genomic_deps()

    bigwig_path = Path(bigwig_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": str(bigwig_path),
        "dtype": str(dtype),
        "chromosomes": {},
    }

    with pyBigWig.open(str(bigwig_path)) as bw:
        bw_chroms = bw.chroms()
        chroms_to_convert = chromosomes if chromosomes else list(bw_chroms.keys())

        for chrom in chroms_to_convert:
            if chrom not in bw_chroms:
                continue

            size = bw_chroms[chrom]
            values = bw.values(chrom, 0, size, numpy=True)

            if values is None:
                values = np.zeros(size, dtype=dtype)
            else:
                values = np.nan_to_num(np.asarray(values, dtype=dtype), nan=0.0)

            # Save as .npy file
            npy_filename = f"{chrom}.npy"
            np.save(output_dir / npy_filename, values)

            metadata["chromosomes"][chrom] = {
                "file": npy_filename,
                "size": size,
            }

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


def load_bigwigs_parallel(
    bigwig_paths: list[str],
    chromosomes: set[str],
    chrom_sizes: dict[str, int],
    max_workers: int = DEFAULT_MAX_IO_WORKERS,
) -> list[CachedBigWig]:
    """Load multiple BigWig files in parallel using ThreadPoolExecutor.

    Args:
        bigwig_paths: List of BigWig file paths.
        chromosomes: Set of chromosomes to load.
        chrom_sizes: Dict mapping chromosome names to sizes.
        max_workers: Maximum number of parallel workers.

    Returns:
        List of CachedBigWig instances (same order as input paths).
    """
    n_workers = min(len(bigwig_paths), max_workers)

    if n_workers <= 1:
        # Sequential loading for single file
        return [CachedBigWig(p, chromosomes, chrom_sizes) for p in bigwig_paths]

    print(f"Loading {len(bigwig_paths)} bigwig files with {n_workers} workers...")

    results: dict[int, CachedBigWig] = {}

    def load_one(idx_path: tuple[int, str]) -> tuple[int, CachedBigWig]:
        idx, path = idx_path
        return idx, CachedBigWig(path, chromosomes, chrom_sizes)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(load_one, (i, p))
            for i, p in enumerate(bigwig_paths)
        ]
        for future in futures:
            idx, cached = future.result()
            results[idx] = cached

    # Return in original order
    return [results[i] for i in range(len(bigwig_paths))]


def _load_intervals_from_bed(
    bed_path: str,
) -> tuple[list[tuple[str, int, int]], set[str]]:
    """Load genomic intervals from a BED file.

    Args:
        bed_path: Path to BED file with chrom, start, end columns.

    Returns:
        Tuple of (intervals_list, chromosomes_set) where intervals_list
        contains (chrom, start, end) tuples.
    """
    intervals: list[tuple[str, int, int]] = []
    chromosomes: set[str] = set()

    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            intervals.append((chrom, start, end))
            chromosomes.add(chrom)

    return intervals, chromosomes


class GenomicDataset(Dataset):
    """Dataset for genomic fine-tuning (ATAC, RNA-seq, etc).

    Loads genomic intervals from a pre-split BED file. If the BED intervals
    are smaller than sequence_length, they are expanded from their center.
    If intervals are larger than sequence_length, they are centered and
    truncated, which may lose flanking regions (a warning will be issued).

    Args:
        genome_fasta: Path to genome FASTA file, or a pre-built
            :class:`CachedGenome` instance.  Passing a ``CachedGenome``
            lets you share a single cache across train/val datasets to
            halve memory usage.
        bigwig_files: List of BigWig files (one per track). Can also be
            paths to mmap directories created by ``convert_bigwig_to_mmap()``.
        bed_file: BED file with intervals (chrom, start, end).
        resolutions: Output resolutions (default: (1, 128)).
        sequence_length: Input sequence length (default: 2**20 == 1,048,576).
            If BED intervals are smaller, they are centered and expanded.
            If BED intervals are larger, they are centered and truncated.
        cache_genome: If True **and** ``genome_fasta`` is a string path,
            prefetch the entire genome into memory for faster access.
            Ignored when a ``CachedGenome`` is passed directly.
        cache_signals: If True, prefetch all bigwig signals into memory.
            Recommended for training. Memory usage depends on track count.
            Uses parallel loading with ThreadPoolExecutor.
        max_io_workers: Maximum number of threads for parallel I/O operations.
            Used for both cache initialization and per-sample reads (when not
            caching). Actual workers = min(n_tracks, max_io_workers).
            Default: 16.
        use_mmap: If True, treat bigwig_files as paths to mmap directories
            created by ``convert_bigwig_to_mmap()``. Provides fastest reads
            with minimal memory overhead.

    Example:
        >>> from alphagenome_pytorch.extensions.finetuning.datasets import (
        ...     GenomicDataset, CachedGenome,
        ... )
        >>> # Shared cache between train and val (recommended):
        >>> genome = CachedGenome('hg38.fa')
        >>> train_ds = GenomicDataset(
        ...     genome_fasta=genome,
        ...     bigwig_files=['signal.bw'],
        ...     bed_file='train_positions.bed',
        ... )
        >>> val_ds = GenomicDataset(
        ...     genome_fasta=genome,
        ...     bigwig_files=['signal.bw'],
        ...     bed_file='val_positions.bed',
        ... )
        >>> # Or pass a path string with cache_genome=True:
        >>> ds = GenomicDataset(
        ...     genome_fasta='hg38.fa',
        ...     bigwig_files=['signal.bw'],
        ...     bed_file='positions.bed',
        ...     cache_genome=True,
        ... )
        >>> # Using pre-converted mmap files (fastest):
        >>> ds = GenomicDataset(
        ...     genome_fasta='hg38.fa',
        ...     bigwig_files=['signal_mmap/'],  # Created by convert_bigwig_to_mmap
        ...     bed_file='positions.bed',
        ...     use_mmap=True,
        ... )
    """

    def __init__(
        self,
        genome_fasta: str | CachedGenome,
        bigwig_files: list[str],
        bed_file: str,
        resolutions: tuple[int, ...] = (1, 128),
        sequence_length: int = 131_072,
        cache_genome: bool = False,
        cache_signals: bool = False,
        max_io_workers: int = DEFAULT_MAX_IO_WORKERS,
        use_mmap: bool = False,
    ):
        _ensure_genomic_deps()

        self.sequence_length = sequence_length
        self.resolutions = sorted(resolutions)
        self.bigwig_files = bigwig_files
        self.cache_signals = cache_signals
        self.max_io_workers = max_io_workers
        self.use_mmap = use_mmap

        # Normalise genome input: str path or pre-built CachedGenome
        if isinstance(genome_fasta, CachedGenome):
            self._cached_genome = genome_fasta
            self.genome_fasta = genome_fasta.fasta_path
        else:
            self._cached_genome = None
            self.genome_fasta = genome_fasta

        # Calculate output lengths for each resolution
        self.output_lengths = {res: sequence_length // res for res in self.resolutions}

        # Load intervals from BED
        all_intervals, self._chromosomes = _load_intervals_from_bed(bed_file)

        # Get chromosome sizes
        if self._cached_genome is not None:
            self._chrom_sizes = {
                ref: size
                for ref, size in self._cached_genome.chrom_sizes.items()
                if ref in self._chromosomes
            }
        else:
            fasta = pyfaidx.Fasta(self.genome_fasta)
            try:
                self._chrom_sizes = {
                    ref: len(fasta[ref])
                    for ref in fasta.keys()
                    if ref in self._chromosomes
                }
            finally:
                fasta.close()

        # Process intervals: expand from center if needed
        half_len = self.sequence_length // 2
        self._positions_list: list[tuple[str, int, int]] = []
        n_skipped = 0
        n_truncated = 0

        for chrom, start, end in all_intervals:
            if chrom not in self._chrom_sizes:
                n_skipped += 1
                continue

            interval_size = end - start
            if interval_size == self.sequence_length:
                # Use interval directly
                final_start = start
                final_end = end
            elif interval_size < self.sequence_length:
                # Expand from center
                center = (start + end) // 2
                final_start = center - half_len
                final_end = center + half_len
            else:
                # interval_size > sequence_length: truncate from center
                n_truncated += 1
                center = (start + end) // 2
                final_start = center - half_len
                final_end = center + half_len

            # Validate bounds
            if final_start < 0 or final_end > self._chrom_sizes[chrom]:
                n_skipped += 1
                continue

            self._positions_list.append((chrom, final_start, final_end))

        if n_skipped > 0:
            warnings.warn(
                f"{n_skipped} intervals were skipped because they would exceed "
                f"chromosome boundaries when expanded to sequence_length={sequence_length}."
            )

        if n_truncated > 0:
            warnings.warn(
                f"{n_truncated} intervals were larger than sequence_length={sequence_length} "
                f"and were centered and truncated, which may lose important flanking regions."
            )

        self.n_tracks = len(bigwig_files)

        # Initialize remaining caches / lazy handles
        self._cached_bigwigs: list[CachedBigWig] | None = None
        self._mmap_bigwigs: list[MmapBigWig] | None = None
        self._fasta = None
        self._bigwigs = None
        self._io_executor: ThreadPoolExecutor | None = None

        # Prefetch genome if requested and not already provided
        if cache_genome and self._cached_genome is None:
            self._cached_genome = CachedGenome(self.genome_fasta, self._chromosomes)

        # Initialize signal sources
        if use_mmap:
            # Load from pre-converted mmap directories
            print(f"MmapBigWig: Loading {len(bigwig_files)} mmap directories...")
            self._mmap_bigwigs = [MmapBigWig(p) for p in bigwig_files]
            print(f"MmapBigWig: Loaded {len(bigwig_files)} mmap sources")
        elif cache_signals:
            # Load bigwigs into memory (parallel)
            print(f"CachedBigWig: Loading {len(bigwig_files)} bigwig file(s)...")
            self._cached_bigwigs = load_bigwigs_parallel(
                bigwig_files,
                self._chromosomes,
                self._chrom_sizes,
                max_workers=max_io_workers,
            )
            cached_size_mb = sum(
                sum(arr.nbytes for arr in bw._cache.values())
                for bw in self._cached_bigwigs
            ) / 1e6
            print(f"CachedBigWig: Loaded {len(bigwig_files)} files ({cached_size_mb:.1f} MB)")
        # Note: ThreadPoolExecutor for lazy reads is created lazily in _ensure_handles()
        # to be fork-safe with DataLoader workers

    def _ensure_handles(self):
        """Ensure file handles are open for the current process.

        File handles (pyfaidx, pyBigWig) cannot be shared across forked processes,
        so we track the PID and re-open handles if we're in a new process
        (e.g., a DataLoader worker).
        """
        import os
        current_pid = os.getpid()

        # Check if we need to reinitialize (new process or first call)
        if not hasattr(self, "_owner_pid") or self._owner_pid != current_pid:
            self._owner_pid = current_pid
            # Close any stale handles from parent process
            if hasattr(self, "_fasta") and self._fasta is not None:
                try:
                    self._fasta.close()
                except Exception:
                    pass
            if hasattr(self, "_bigwigs") and self._bigwigs is not None:
                for bw in self._bigwigs:
                    try:
                        bw.close()
                    except Exception:
                        pass
            if hasattr(self, "_io_executor") and self._io_executor is not None:
                try:
                    self._io_executor.shutdown(wait=False)
                except Exception:
                    pass
            # Reset handles to be reopened
            self._fasta = None
            self._bigwigs = None
            self._io_executor = None

        # Open genome handle if needed (lazy mode only)
        if self._cached_genome is None and self._fasta is None:
            _ensure_genomic_deps()
            self._fasta = pyfaidx.Fasta(self.genome_fasta)

        # Open bigwig handles if needed (lazy mode only)
        if self._cached_bigwigs is None and self._mmap_bigwigs is None:
            _ensure_genomic_deps()
            if self._bigwigs is None:
                self._bigwigs = [pyBigWig.open(bw) for bw in self.bigwig_files]
            # Create thread pool for parallel reads if multiple tracks
            if self._io_executor is None and self.n_tracks > 1:
                n_workers = min(self.n_tracks, self.max_io_workers)
                self._io_executor = ThreadPoolExecutor(max_workers=n_workers)

    def _get_sequence(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Get one-hot encoded sequence for a region."""
        if self._cached_genome is not None:
            return self._cached_genome.fetch(chrom, start, end)
        else:
            sequence_str = str(self._fasta[chrom][start:end])
            return sequence_to_onehot(sequence_str)

    def _get_signal(self, track_idx: int, chrom: str, start: int, end: int) -> np.ndarray:
        """Get signal values for a region from a specific track."""
        if self._mmap_bigwigs is not None:
            return self._mmap_bigwigs[track_idx].values(chrom, start, end)
        elif self._cached_bigwigs is not None:
            return self._cached_bigwigs[track_idx].values(chrom, start, end)
        else:
            bw = self._bigwigs[track_idx]
            try:
                sig = bw.values(chrom, start, end, numpy=True)
                if sig is None:
                    raise ValueError(f"No values found for {chrom}:{start}-{end}")
                sig = np.asarray(sig, dtype=np.float32)
                return np.nan_to_num(sig, nan=0.0)
            except Exception as e:
                bw_path = self.bigwig_files[track_idx]
                bw_info = f"Tracks in BW: {bw.chroms()}"
                raise RuntimeError(
                    f"Error fetching values from {bw_path} at {chrom}:{start}-{end}. "
                    f"Error: {e}. {bw_info}"
                ) from e

    def _get_all_signals_parallel(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Get signal values for all tracks in parallel using ThreadPoolExecutor."""
        def fetch_one(idx: int) -> tuple[int, np.ndarray]:
            return idx, self._get_signal(idx, chrom, start, end)

        results: dict[int, np.ndarray] = {}
        futures = [
            self._io_executor.submit(fetch_one, i)
            for i in range(self.n_tracks)
        ]
        for future in futures:
            idx, sig = future.result()
            results[idx] = sig

        # Stack in order
        return np.stack([results[i] for i in range(self.n_tracks)], axis=-1)

    def __len__(self) -> int:
        return len(self._positions_list)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Get a single sample.

        Returns:
            Tuple of (sequence, targets_dict):
                - sequence: One-hot encoded DNA (seq_len, 4)
                - targets_dict: Dict mapping resolution to signals
                    {res: tensor of shape (output_len, n_tracks)}
        """
        self._ensure_handles()
        chrom, start, end = self._positions_list[idx]

        # Get one-hot sequence
        sequence = self._get_sequence(chrom, start, end)

        # Get raw signals (1bp resolution)
        if self._io_executor is not None:
            # Parallel reads for lazy mode with multiple tracks
            raw_signals = self._get_all_signals_parallel(chrom, start, end)
        else:
            # Sequential reads (cached/mmap modes are already fast)
            raw_signals = []
            for i in range(self.n_tracks):
                sig = self._get_signal(i, chrom, start, end)
                raw_signals.append(sig)
            # Stack tracks at 1bp: (seq_len, n_tracks)
            raw_signals = np.stack(raw_signals, axis=-1)

        # Create targets at each resolution
        targets_dict = {}
        for res in self.resolutions:
            if res == 1:
                targets_dict[res] = torch.from_numpy(raw_signals).float()
            else:
                # Bin to resolution
                if self.sequence_length % res != 0:
                    raise ValueError(
                        f"Resolution {res} does not evenly divide sequence_length "
                        f"{self.sequence_length}. Ensure that all resolutions in "
                        "GenomicDataset.resolutions evenly divide sequence_length."
                    )
                output_len = self.sequence_length // res
                binned = raw_signals.reshape(output_len, res, self.n_tracks).sum(axis=1)
                targets_dict[res] = torch.from_numpy(binned).float()

        return (
            torch.from_numpy(sequence).float(),
            targets_dict,
        )

    def __del__(self) -> None:
        """Clean up file handles and thread pool (only if we own them)."""
        import os
        # Only cleanup if we're in the process that created the handles
        if not hasattr(self, "_owner_pid") or self._owner_pid != os.getpid():
            return
        if hasattr(self, "_fasta") and self._fasta is not None:
            try:
                self._fasta.close()
            except Exception:
                pass
        if hasattr(self, "_bigwigs") and self._bigwigs is not None:
            for bw in self._bigwigs:
                if bw is not None:
                    try:
                        bw.close()
                    except Exception:
                        pass
        if hasattr(self, "_io_executor") and self._io_executor is not None:
            try:
                self._io_executor.shutdown(wait=False)
            except Exception:
                pass


def compute_track_means(
    bigwig_files: list[str],
    bed_file: str,
    sequence_length: int = 1_048_576,
    resolution: int = 1,
    max_samples: int | None = None,
) -> torch.Tensor:
    """Compute nonzero_mean signal per track from training data.

    Computes the mean signal value over NON-ZERO positions only, matching
    the `nonzero_mean` statistic used in AlphaGenome's track metadata.
    This is critical for proper scaling since genomic signals are sparse.

    Args:
        bigwig_files: List of BigWig files (one per track).
        bed_file: BED file with training positions.
        sequence_length: Sequence length for each position (default: 1M).
        resolution: Resolution for computing means (1 or 128). Use 1 for
            most accurate means, 128 for faster computation.
        max_samples: Maximum number of samples to use for computing means.
            If None, uses all samples. Using a subset (e.g., 1000) speeds
            up computation while giving a good estimate.

    Returns:
        Track means tensor of shape (1, n_tracks) suitable for passing
        to create_finetuning_head(track_means=...).

    Example:
        >>> track_means = compute_track_means(
        ...     bigwig_files=['cell1.bw', 'cell2.bw'],
        ...     bed_file='train_positions.bed',
        ...     max_samples=1000,  # Use subset for speed
        ... )
        >>> head = create_finetuning_head('atac', n_tracks=2, track_means=track_means)
    """
    _ensure_genomic_deps()

    # Load intervals
    all_intervals, chromosomes = _load_intervals_from_bed(bed_file)

    # Open bigwigs to get chrom sizes
    bws = [pyBigWig.open(bw) for bw in bigwig_files]
    n_tracks = len(bws)

    try:
        # Get chromosome sizes from first bigwig
        chrom_sizes = dict(bws[0].chroms())

        # Process intervals (expand from center if needed)
        half_len = sequence_length // 2
        valid_positions: list[tuple[str, int, int]] = []

        for chrom, start, end in all_intervals:
            if chrom not in chrom_sizes:
                continue

            # Expand/contract to sequence_length
            center = (start + end) // 2
            final_start = center - half_len
            final_end = center + half_len

            if final_start < 0 or final_end > chrom_sizes[chrom]:
                continue

            valid_positions.append((chrom, final_start, final_end))

        # Optionally limit samples
        if max_samples is not None and len(valid_positions) > max_samples:
            # Use deterministic subset (every Nth sample)
            step = len(valid_positions) // max_samples
            valid_positions = valid_positions[::step][:max_samples]

        if not valid_positions:
            raise ValueError("No valid positions found for computing track means")

        # Compute nonzero_mean: sum of nonzero values / count of nonzero values
        # This matches AlphaGenome's track metadata nonzero_mean statistic
        track_nonzero_sums = np.zeros(n_tracks, dtype=np.float64)
        track_nonzero_counts = np.zeros(n_tracks, dtype=np.int64)

        for chrom, start, end in valid_positions:
            for i, bw in enumerate(bws):
                values = bw.values(chrom, start, end, numpy=True)
                values = np.asarray(values, dtype=np.float32)
                values = np.nan_to_num(values, nan=0.0)

                if resolution > 1:
                    # Bin and sum
                    output_len = sequence_length // resolution
                    values = values.reshape(output_len, resolution).sum(axis=1)

                # Only count non-zero values for nonzero_mean
                nonzero_vals = values[values != 0]
                track_nonzero_sums[i] += nonzero_vals.sum()
                track_nonzero_counts[i] += len(nonzero_vals)
    finally:
        # Ensure bigwigs are closed even if an exception occurs
        for bw in bws:
            bw.close()

    # Compute nonzero_mean per track
    # Use 1.0 as fallback if no nonzero values found (shouldn't happen normally)
    track_means = np.where(
        track_nonzero_counts > 0,
        track_nonzero_sums / track_nonzero_counts,
        1.0,
    )

    print(f"Computed nonzero_mean per track: {track_means}")

    # Return as (1, n_tracks) for num_organisms=1
    return torch.tensor(track_means, dtype=torch.float32).unsqueeze(0)


class MultimodalDataset(Dataset):
    """Dataset wrapper for multi-modality training.

    Wraps multiple GenomicDatasets (one per modality) that share the same
    genomic positions. Returns targets for all modalities per sample.

    All wrapped datasets must have the same length (same BED file).

    Args:
        datasets: Dict mapping modality name to GenomicDataset.

    Example:
        >>> atac_ds = GenomicDataset(genome, atac_bigwigs, bed_file, ...)
        >>> rna_ds = GenomicDataset(genome, rna_bigwigs, bed_file, ...)
        >>> multi_ds = MultimodalDataset({
        ...     "atac": atac_ds,
        ...     "rna_seq": rna_ds,
        ... })
        >>> seq, modality_targets = multi_ds[0]
        >>> # modality_targets = {"atac": {1: tensor, 128: tensor}, "rna_seq": {...}}
    """

    def __init__(self, datasets: dict[str, GenomicDataset]):
        self.datasets = datasets
        self.modalities = list(datasets.keys())

        # Verify all datasets have the same length
        lengths = {name: len(ds) for name, ds in datasets.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"All datasets must have the same length (same BED file). Got: {lengths}"
            )

        self._length = next(iter(lengths.values()))

        # Use first dataset for sequence access
        self._primary_dataset = next(iter(datasets.values()))

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]]]:
        """Get sequence and targets for all modalities.

        Returns:
            Tuple of (sequence, modality_targets) where:
                - sequence: One-hot encoded DNA (seq_len, 4)
                - modality_targets: Dict mapping modality name to targets_dict
                    {modality: {resolution: tensor}}
        """
        # Get sequence from primary dataset
        sequence, _ = self._primary_dataset[idx]

        # Get targets from all datasets
        modality_targets = {}
        for modality, dataset in self.datasets.items():
            _, targets_dict = dataset[idx]
            modality_targets[modality] = targets_dict

        return sequence, modality_targets


def collate_multimodal(
    batch: list[tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]]]],
) -> tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]]]:
    """Collate function for MultimodalDataset.

    Args:
        batch: List of (sequence, modality_targets) tuples.

    Returns:
        Tuple of (sequences, modality_targets) where:
            - sequences: Stacked sequences (batch, seq_len, 4)
            - modality_targets: Dict of modality -> {resolution -> (batch, out_len, n_tracks)}
    """
    sequences = torch.stack([item[0] for item in batch])

    # Aggregate targets by modality and resolution
    modality_targets: dict[str, dict[int, torch.Tensor]] = {}

    first_item = batch[0][1]
    for modality in first_item.keys():
        modality_targets[modality] = {}
        for res in first_item[modality].keys():
            modality_targets[modality][res] = torch.stack([
                item[1][modality][res] for item in batch
            ])

    return sequences, modality_targets


# Backward-compatible aliases
ATACDataset = GenomicDataset
RNASeqDataset = GenomicDataset


__all__ = [
    "GenomicDataset",
    "MultimodalDataset",
    "collate_multimodal",
    "ATACDataset",
    "RNASeqDataset",
    "CachedGenome",
    "CachedBigWig",
    "MmapBigWig",
    "convert_bigwig_to_mmap",
    "load_bigwigs_parallel",
    "compute_track_means",
    "DEFAULT_MAX_IO_WORKERS",
]
