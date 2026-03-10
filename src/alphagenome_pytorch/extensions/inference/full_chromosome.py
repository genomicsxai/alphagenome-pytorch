"""Full chromosome prediction with tiling and BigWig output.

Generates genome-wide predictions by tiling across chromosomes and stitching
results into BigWig files.

Example:
    >>> from alphagenome_pytorch import AlphaGenome
    >>> from alphagenome_pytorch.extensions.inference import (
    ...     TilingConfig,
    ...     predict_full_chromosome,
    ...     predict_full_chromosomes_to_bigwig,
    ... )
    >>>
    >>> model = AlphaGenome.from_pretrained('model.pth', device='cuda')
    >>> config = TilingConfig(crop_bp=0, resolution=128)
    >>>
    >>> # Single chromosome -> numpy array
    >>> preds = predict_full_chromosome(
    ...     model, genome, chrom='chr1', head='atac', config=config
    ... )
    >>>
    >>> # Multiple chromosomes -> BigWig files
    >>> predict_full_chromosomes_to_bigwig(
    ...     model, fasta_path='hg38.fa', output_dir='./preds',
    ...     head='atac', chromosomes=['chr1', 'chr2'], config=config
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

# Lazy imports
pyBigWig = None
pysam = None


def _ensure_deps():
    """Lazily import pyBigWig and pysam."""
    global pyBigWig, pysam
    if pyBigWig is None:
        import pyBigWig as _pyBigWig
        pyBigWig = _pyBigWig
    if pysam is None:
        import pysam as _pysam
        pysam = _pysam


# Head configurations: name -> (num_tracks, supported_resolutions)
HEAD_CONFIGS = {
    'atac': {'num_tracks': 256, 'resolutions': [1, 128]},
    'dnase': {'num_tracks': 384, 'resolutions': [1, 128]},
    'procap': {'num_tracks': 128, 'resolutions': [1, 128]},
    'cage': {'num_tracks': 640, 'resolutions': [1, 128]},
    'rna_seq': {'num_tracks': 768, 'resolutions': [1, 128]},
    'chip_tf': {'num_tracks': 1664, 'resolutions': [128]},
    'chip_histone': {'num_tracks': 1152, 'resolutions': [128]},
}


@dataclass
class TilingConfig:
    """Configuration for genome tiling.

    Args:
        window_size: Model input window size in bp. Default: 131072 (AlphaGenome native).
        crop_bp: Base pairs to crop from each edge. Default: 0 (no overlap).
            Set to e.g. 32768 to keep only center ~50% of each window.
        resolution: Output resolution in bp. Default: 128.
            Use 1 for base-pair resolution (slower, requires decoder).
            Use 128 for bin-level resolution (faster).
        batch_size: Number of windows to process per batch. Default: 4.
    """
    window_size: int = 131072
    crop_bp: int = 0
    resolution: int = 128
    batch_size: int = 4

    def __post_init__(self):
        if self.crop_bp < 0:
            raise ValueError(f"crop_bp must be >= 0, got {self.crop_bp}")
        if self.crop_bp * 2 >= self.window_size:
            raise ValueError(
                f"crop_bp ({self.crop_bp}) too large for window_size ({self.window_size}). "
                f"Must be less than window_size / 2."
            )
        if self.resolution not in (1, 128):
            raise ValueError(f"resolution must be 1 or 128, got {self.resolution}")
        if self.crop_bp % self.resolution != 0:
            raise ValueError(
                f"crop_bp ({self.crop_bp}) must be divisible by resolution ({self.resolution})"
            )

    @property
    def effective_size(self) -> int:
        """Size of the kept region per window (in bp)."""
        return self.window_size - 2 * self.crop_bp

    @property
    def step_size(self) -> int:
        """Step between window starts (equals effective_size for seamless tiling)."""
        return self.effective_size

    @property
    def crop_start(self) -> int:
        """Start index of kept region within window (in bp)."""
        return self.crop_bp

    @property
    def crop_end(self) -> int:
        """End index of kept region within window (in bp)."""
        return self.window_size - self.crop_bp


class GenomeSequenceProvider:
    """Provides one-hot encoded sequences with padding for out-of-bounds regions.

    Can use either a CachedGenome instance or load directly from FASTA.
    """

    def __init__(
        self,
        source: str | Path,
        chromosomes: set[str] | None = None,
        cache: bool = True,
    ):
        """Initialize sequence provider.

        Args:
            source: Path to FASTA file or existing CachedGenome.
            chromosomes: Optional set of chromosomes to load. If None, loads all.
            cache: Whether to cache chromosomes in memory. Default: True.
        """
        _ensure_deps()

        self.chrom_sizes: dict[str, int] = {}
        self._cache: dict[str, np.ndarray] = {}
        self._fasta_path = str(source)
        self._cache_enabled = cache

        # Load chromosome sizes and optionally cache sequences
        print(f"Loading genome from {source}...")
        with pysam.FastaFile(self._fasta_path) as fasta:
            for ref in fasta.references:
                self.chrom_sizes[ref] = fasta.get_reference_length(ref)

            if cache:
                refs_to_load = chromosomes if chromosomes else set(fasta.references)
                for ref in refs_to_load:
                    if ref in self.chrom_sizes:
                        seq_str = fasta.fetch(ref)
                        self._cache[ref] = _sequence_to_onehot(seq_str)

                cached_mb = sum(arr.nbytes for arr in self._cache.values()) / 1e6
                print(f"Cached {len(self._cache)} chromosomes ({cached_mb:.1f} MB)")

    def fetch(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Fetch one-hot encoded sequence, padding out-of-bounds with N (0.25).

        Args:
            chrom: Chromosome name.
            start: Start position (can be negative for padding).
            end: End position (can exceed chromosome length).

        Returns:
            One-hot encoded array of shape (end - start, 4).
        """
        seq_len = end - start
        chrom_len = self.chrom_sizes.get(chrom, 0)

        # Fast path: fully within chromosome and cached
        if start >= 0 and end <= chrom_len:
            if chrom in self._cache:
                return self._cache[chrom][start:end].copy()
            else:
                return self._fetch_from_fasta(chrom, start, end)

        # Need padding for out-of-bounds regions
        result = np.full((seq_len, 4), 0.25, dtype=np.float32)  # N = uniform

        valid_start = max(0, start)
        valid_end = min(chrom_len, end)

        if valid_start < valid_end:
            if chrom in self._cache:
                seq = self._cache[chrom][valid_start:valid_end]
            else:
                seq = self._fetch_from_fasta(chrom, valid_start, valid_end)

            dest_start = valid_start - start
            dest_end = dest_start + (valid_end - valid_start)
            result[dest_start:dest_end] = seq

        return result

    def _fetch_from_fasta(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Fetch and encode sequence directly from FASTA."""
        with pysam.FastaFile(self._fasta_path) as fasta:
            seq_str = fasta.fetch(chrom, start, end)
            return _sequence_to_onehot(seq_str)


def _sequence_to_onehot(seq: str) -> np.ndarray:
    """Convert DNA sequence string to one-hot encoding.

    Args:
        seq: DNA sequence string (ACGT, case-insensitive).

    Returns:
        One-hot encoded array of shape (len(seq), 4) with columns [A, C, G, T].
        Unknown bases (N, etc.) are encoded as [0.25, 0.25, 0.25, 0.25].
    """
    seq = seq.upper()
    n = len(seq)
    onehot = np.full((n, 4), 0.25, dtype=np.float32)

    seq_array = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

    onehot[seq_array == ord('A')] = [1, 0, 0, 0]
    onehot[seq_array == ord('C')] = [0, 1, 0, 0]
    onehot[seq_array == ord('G')] = [0, 0, 1, 0]
    onehot[seq_array == ord('T')] = [0, 0, 0, 1]

    return onehot


def _generate_tiles(
    chrom_length: int,
    config: TilingConfig,
) -> list[tuple[int, int, int, int]]:
    """Generate tiling coordinates for a chromosome.

    Args:
        chrom_length: Length of chromosome in bp.
        config: Tiling configuration.

    Returns:
        List of (window_start, window_end, keep_start, keep_end) tuples.
        keep_start/keep_end are indices within the window of the region to keep.
    """
    tiles = []
    step = config.step_size
    keep_start = config.crop_start
    keep_end = config.crop_end

    # Start so first kept region begins at position 0
    # window_start + keep_start = 0 => window_start = -keep_start
    window_start = -keep_start

    while window_start < chrom_length:
        window_end = window_start + config.window_size

        # Genomic coordinates this tile's kept region covers
        genome_keep_start = window_start + keep_start
        genome_keep_end = window_start + keep_end

        # Only include if kept region overlaps chromosome
        if genome_keep_end > 0 and genome_keep_start < chrom_length:
            tiles.append((window_start, window_end, keep_start, keep_end))

        window_start += step

    return tiles


def predict_full_chromosome(
    model,
    genome: GenomeSequenceProvider | str | Path,
    chrom: str,
    head: str,
    config: TilingConfig | None = None,
    track_indices: list[int] | None = None,
    organism_index: int = 0,
    device: str | torch.device = "cuda",
    show_progress: bool = True,
) -> np.ndarray:
    """Generate predictions for an entire chromosome.

    Args:
        model: Loaded AlphaGenome model.
        genome: GenomeSequenceProvider instance or path to FASTA file.
        chrom: Chromosome name (e.g., 'chr1').
        head: Prediction head name ('atac', 'dnase', 'cage', 'rna_seq',
            'chip_tf', 'chip_histone', 'procap').
        config: Tiling configuration. Default: TilingConfig().
        track_indices: Which track indices to output. Default: all tracks.
        organism_index: Organism index (0=human, 1=mouse). Default: 0.
        device: PyTorch device. Default: 'cuda'.
        show_progress: Show progress bar. Default: True.

    Returns:
        Predictions array of shape (chrom_length // resolution, n_tracks).
    """
    config = config or TilingConfig()

    # Validate head
    if head not in HEAD_CONFIGS:
        raise ValueError(
            f"Unknown head: {head}. "
            f"Available: {list(HEAD_CONFIGS.keys())}"
        )

    head_config = HEAD_CONFIGS[head]
    if config.resolution not in head_config['resolutions']:
        raise ValueError(
            f"Head '{head}' does not support resolution {config.resolution}. "
            f"Supported: {head_config['resolutions']}"
        )

    # Setup genome provider
    if isinstance(genome, (str, Path)):
        genome = GenomeSequenceProvider(genome, chromosomes={chrom})

    if chrom not in genome.chrom_sizes:
        raise ValueError(f"Chromosome {chrom} not found in genome")

    chrom_length = genome.chrom_sizes[chrom]
    output_length = chrom_length // config.resolution

    # Determine output tracks
    n_head_tracks = head_config['num_tracks']
    if track_indices is None:
        track_indices = list(range(n_head_tracks))
    n_output_tracks = len(track_indices)

    # Initialize output array
    predictions = np.zeros((output_length, n_output_tracks), dtype=np.float32)

    # Generate tiles
    tiles = _generate_tiles(chrom_length, config)

    if len(tiles) == 0:
        return predictions

    # Process in batches
    model.eval()
    device = torch.device(device)

    iterator = range(0, len(tiles), config.batch_size)
    if show_progress:
        n_batches = (len(tiles) + config.batch_size - 1) // config.batch_size
        iterator = tqdm(iterator, total=n_batches, desc=f"Predicting {chrom}")

    for batch_start in iterator:
        batch_tiles = tiles[batch_start:batch_start + config.batch_size]

        # Extract sequences
        sequences = []
        for window_start, window_end, _, _ in batch_tiles:
            seq = genome.fetch(chrom, window_start, window_end)
            sequences.append(seq)

        # Stack and predict
        batch_seq = torch.tensor(np.stack(sequences), device=device)
        batch_org = torch.tensor(
            [organism_index] * len(batch_tiles),
            device=device,
            dtype=torch.long,
        )

        with torch.no_grad():
            preds = model.predict(
                batch_seq,
                batch_org,
                resolutions=(config.resolution,),
            )

        # Extract predictions for the requested head
        # Output shape: (batch, seq_len_at_res, n_tracks)
        head_preds = preds[head][config.resolution]
        head_preds = head_preds[:, :, track_indices].cpu().numpy()

        # Place kept regions into output
        for i, (window_start, window_end, keep_start, keep_end) in enumerate(batch_tiles):
            # Convert to output resolution
            keep_start_res = keep_start // config.resolution
            keep_end_res = keep_end // config.resolution

            # Genomic position of kept region start (at output resolution)
            genome_pos = (window_start + keep_start) // config.resolution

            # Handle edges
            out_start = max(0, genome_pos)
            out_end = min(output_length, genome_pos + (keep_end_res - keep_start_res))

            # Corresponding indices in the prediction
            pred_start = keep_start_res + (out_start - genome_pos)
            pred_end = pred_start + (out_end - out_start)

            if out_start < out_end:
                predictions[out_start:out_end] = head_preds[i, pred_start:pred_end]

    return predictions


def write_bigwig(
    predictions: np.ndarray,
    output_path: str | Path,
    chrom: str,
    chrom_sizes: dict[str, int],
    resolution: int = 128,
    track_names: list[str] | None = None,
) -> list[Path]:
    """Write predictions to BigWig file(s).

    Args:
        predictions: Array of shape (length, n_tracks).
        output_path: Output path. If multiple tracks, will append track name.
        chrom: Chromosome name.
        chrom_sizes: Dict mapping chromosome names to sizes.
        resolution: Base pair resolution. Default: 128.
        track_names: Optional names for each track.

    Returns:
        List of written BigWig file paths.
    """
    _ensure_deps()

    output_path = Path(output_path)
    n_tracks = predictions.shape[1]

    if track_names is None:
        track_names = [f"track_{i}" for i in range(n_tracks)]

    written_paths = []

    for i, track_name in enumerate(track_names):
        if n_tracks > 1:
            bw_path = output_path.parent / f"{output_path.stem}_{track_name}{output_path.suffix}"
        else:
            bw_path = output_path

        bw = pyBigWig.open(str(bw_path), "w")

        # Add header with all chromosome sizes
        header = [(k, v) for k, v in chrom_sizes.items()]
        bw.addHeader(header)

        # Get track data
        track_data = predictions[:, i].astype(np.float64)
        chrom_len = chrom_sizes[chrom]

        # Build coordinate arrays
        n_bins = len(track_data)
        starts = np.arange(n_bins, dtype=np.int64) * resolution
        ends = np.minimum(starts + resolution, chrom_len)

        # Filter to valid range
        valid_mask = starts < chrom_len
        starts = starts[valid_mask]
        ends = ends[valid_mask]
        values = track_data[valid_mask]

        # Write entries
        bw.addEntries(
            [chrom] * len(starts),
            starts.tolist(),
            ends=ends.tolist(),
            values=values.tolist(),
        )

        bw.close()
        written_paths.append(bw_path)

    return written_paths


def predict_full_chromosomes_to_bigwig(
    model,
    fasta_path: str | Path,
    output_dir: str | Path,
    head: str,
    chromosomes: list[str] | None = None,
    config: TilingConfig | None = None,
    track_indices: list[int] | None = None,
    track_names: list[str] | None = None,
    organism_index: int = 0,
    device: str | torch.device = "cuda",
    show_progress: bool = True,
) -> dict[str, list[Path]]:
    """Generate chromosome-wide predictions and save as BigWig files.

    Args:
        model: Loaded AlphaGenome model.
        fasta_path: Path to reference genome FASTA.
        output_dir: Directory for output BigWig files.
        head: Prediction head name.
        chromosomes: List of chromosomes. Default: chr1-22, chrX.
        config: Tiling configuration. Default: TilingConfig().
        track_indices: Which tracks to output. Default: all.
        track_names: Names for output tracks. Default: track_0, track_1, ...
        organism_index: Organism index (0=human, 1=mouse). Default: 0.
        device: PyTorch device. Default: 'cuda'.
        show_progress: Show progress bars. Default: True.

    Returns:
        Dict mapping chromosome names to lists of written BigWig paths.
    """
    config = config or TilingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load genome
    if chromosomes is None:
        chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

    genome = GenomeSequenceProvider(
        fasta_path,
        chromosomes=set(chromosomes),
        cache=True,
    )

    # Filter to available chromosomes
    chromosomes = [c for c in chromosomes if c in genome.chrom_sizes]

    if not chromosomes:
        raise ValueError("No valid chromosomes found in genome")

    print(f"Will predict {len(chromosomes)} chromosomes: {chromosomes}")

    # Predict and write each chromosome
    results: dict[str, list[Path]] = {}

    for chrom in chromosomes:
        print(f"\nProcessing {chrom}...")

        predictions = predict_full_chromosome(
            model=model,
            genome=genome,
            chrom=chrom,
            head=head,
            config=config,
            track_indices=track_indices,
            organism_index=organism_index,
            device=device,
            show_progress=show_progress,
        )

        # Write to BigWig
        output_path = output_dir / f"{head}_{chrom}.bw"
        written = write_bigwig(
            predictions=predictions,
            output_path=output_path,
            chrom=chrom,
            chrom_sizes=genome.chrom_sizes,
            resolution=config.resolution,
            track_names=track_names,
        )

        results[chrom] = written
        print(f"  Wrote {len(written)} file(s): {[p.name for p in written]}")

    return results
