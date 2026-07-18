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

from alphagenome_pytorch.genome import GenomeSequenceSource
from alphagenome_pytorch.utils.sequence import sequence_to_onehot

#: Chromosomes predicted when none are named: the main assembly, chr1..22 + chrX.
#: Excludes chrY, chrM and scaffolds. Names not present in the FASTA are dropped.
DEFAULT_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

# Lazy imports
pyBigWig = None
pyfaidx = None


def _ensure_deps():
    """Lazily import pyBigWig and pyfaidx."""
    global pyBigWig, pyfaidx
    if pyBigWig is None:
        import pyBigWig as _pyBigWig
        pyBigWig = _pyBigWig
    if pyfaidx is None:
        import pyfaidx as _pyfaidx
        pyfaidx = _pyfaidx


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
    Uses pyfaidx for efficient indexed FASTA access.
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
        self.chrom_sizes: dict[str, int] = {}
        print(f"Loading genome from {source}...")
        self._source = GenomeSequenceSource(
            source,
            chromosomes=chromosomes,
            cache=cache,
            verbose=True,
        )
        self.chrom_sizes = self._source.chrom_sizes

    def fetch(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Fetch one-hot encoded sequence, padding out-of-bounds with zeros.

        Args:
            chrom: Chromosome name.
            start: Start position (can be negative for padding).
            end: End position (can exceed chromosome length).

        Returns:
            One-hot encoded array of shape (end - start, 4).
        """
        return self._source.fetch_onehot(chrom, start, end, pad=True)

    def close(self):
        """Close the FASTA file handle."""
        if hasattr(self, "_source"):
            self._source.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _fetch_from_fasta(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Fetch and encode sequence directly from FASTA."""
        return self._source.fetch_onehot(chrom, start, end)


def _sequence_to_onehot(seq: str) -> np.ndarray:
    """Convert DNA sequence string to one-hot encoding.

    Args:
        seq: DNA sequence string (ACGT, case-insensitive).

    Returns:
        One-hot encoded array of shape (len(seq), 4) with columns [A, C, G, T].
        Unknown bases (N, etc.) are encoded as [0, 0, 0, 0].
    """
    return sequence_to_onehot(seq)


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


def _resolve_head_config(model, head: str, resolution: int) -> dict:
    """Resolve a head's ``{num_tracks, resolutions}`` and validate ``resolution``.

    Introspects the model's live heads first (finetuned/custom heads), falling
    back to the hardcoded ``HEAD_CONFIGS`` for stock pretrained heads.
    """
    _inner = getattr(model, '_orig_mod', model)  # unwrap torch.compile
    heads = getattr(_inner, 'heads', None)
    head_module = heads[head] if heads is not None and head in heads else None

    if head_module is not None:
        head_config = {
            'num_tracks': head_module.num_tracks,
            'resolutions': list(head_module.resolutions),
        }
    elif head in HEAD_CONFIGS:
        head_config = HEAD_CONFIGS[head]
    else:
        available = list(heads.keys()) if heads is not None else list(HEAD_CONFIGS.keys())
        raise ValueError(f"Unknown head: {head}. Available: {available}")

    if resolution not in head_config['resolutions']:
        raise ValueError(
            f"Head '{head}' does not support resolution {resolution}. "
            f"Supported: {head_config['resolutions']}"
        )
    return head_config


def _iter_tile_predictions(
    model,
    genome: "GenomeSequenceProvider",
    chrom: str,
    head: str,
    config: TilingConfig,
    track_indices: list[int],
    output_length: int,
    *,
    organism_index: int = 0,
    device: str | torch.device = "cuda",
    show_progress: bool = True,
):
    """Yield ``(out_start_res, kept_preds)`` for each tile's kept central region.

    ``kept_preds`` is a ``[n_bins, n_tracks]`` float32 ndarray for output-resolution
    positions ``[out_start_res, out_start_res + n_bins)``, clamped to the chromosome
    and with the ``config.crop_bp`` edges removed. Shared by the BigWig and
    gene-count output paths.
    """
    tiles = _generate_tiles(genome.chrom_sizes[chrom], config)
    if not tiles:
        return
    model.eval()
    device = torch.device(device)

    n_batches = (len(tiles) + config.batch_size - 1) // config.batch_size
    iterator = range(0, len(tiles), config.batch_size)
    if show_progress:
        # Local import so this module stays importable with only core deps
        # (e.g. for HEAD_CONFIGS during CLI parser build). tqdm ships with the
        # inference extra, which is present whenever predictions actually run.
        from tqdm import tqdm
        iterator = tqdm(iterator, total=n_batches, desc=f"Predicting {chrom}")

    for batch_start in iterator:
        batch_tiles = tiles[batch_start:batch_start + config.batch_size]

        sequences = [genome.fetch(chrom, ws, we) for ws, we, _, _ in batch_tiles]
        batch_seq = torch.tensor(np.stack(sequences), device=device)
        batch_org = torch.tensor(
            [organism_index] * len(batch_tiles), device=device, dtype=torch.long
        )

        with torch.no_grad():
            preds = model.predict(
                batch_seq, batch_org,
                resolutions=(config.resolution,), heads=(head,),
            )

        # (batch, seq_len_at_res, n_tracks)
        head_preds = preds[head][config.resolution][:, :, track_indices].cpu().numpy()
        del preds, batch_seq, batch_org

        for i, (window_start, window_end, keep_start, keep_end) in enumerate(batch_tiles):
            keep_start_res = keep_start // config.resolution
            keep_end_res = keep_end // config.resolution
            genome_pos = (window_start + keep_start) // config.resolution

            out_start = max(0, genome_pos)
            out_end = min(output_length, genome_pos + (keep_end_res - keep_start_res))
            pred_start = keep_start_res + (out_start - genome_pos)
            pred_end = pred_start + (out_end - out_start)

            if out_start < out_end:
                yield out_start, head_preds[i, pred_start:pred_end]


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

    head_config = _resolve_head_config(model, head, config.resolution)

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

    tiles = _generate_tiles(chrom_length, config)
    if len(tiles) == 0:
        return predictions

    if show_progress:
        n_batches = (len(tiles) + config.batch_size - 1) // config.batch_size
        print(f"  Tiles: {len(tiles)}, Batches: {n_batches}")
        print(f"  Output array: {predictions.nbytes / 1e6:.1f} MB "
              f"({output_length:,} x {n_output_tracks} float32)")

    # Stitch each tile's kept region into the full-chromosome array.
    for out_start, kept in _iter_tile_predictions(
        model, genome, chrom, head, config, track_indices, output_length,
        organism_index=organism_index, device=device, show_progress=show_progress,
    ):
        predictions[out_start:out_start + kept.shape[0]] = kept

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

        # Filter to valid range
        n_valid = min(len(track_data), chrom_len // resolution)

        # Write in chunks using fixed-step format to avoid
        # materializing huge Python lists (critical at 1bp resolution
        # where n_valid can be ~46.7M for chr21)
        CHUNK_SIZE = 1_000_000
        for chunk_start in range(0, n_valid, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_valid)
            bw.addEntries(
                chrom, chunk_start * resolution,
                values=track_data[chunk_start:chunk_end].tolist(),
                span=resolution, step=resolution,
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
        chromosomes = list(DEFAULT_CHROMOSOMES)

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


def _build_track_frame(track_indices, track_names=None, track_strands=None):
    """A ``[C]``-row track-metadata DataFrame for the AnnData ``obs`` table."""
    import pandas as pd

    n = len(track_indices)
    data: dict = {"track_index": list(track_indices)}
    if track_names is not None:
        if len(track_names) != n:
            raise ValueError(f"track_names has {len(track_names)} entries but "
                             f"{n} tracks are being aggregated.")
        data["track_name"] = list(track_names)
    if track_strands is not None:
        if len(track_strands) != n:
            raise ValueError(f"track_strands has {len(track_strands)} entries but "
                             f"{n} tracks are being aggregated.")
        from ...aggregation import _validate_track_strands
        _validate_track_strands(track_strands)
        data["strand"] = [str(s) for s in track_strands]
    return pd.DataFrame(data)


def predict_full_chromosomes_to_anndata(
    model,
    fasta_path: str | Path,
    annotation_path: str | Path,
    head: str,
    *,
    output_path: str | Path | None = None,
    chromosomes: list[str] | None = None,
    config: TilingConfig | None = None,
    track_indices: list[int] | None = None,
    track_names: list[str] | None = None,
    track_strands: list[str] | None = None,
    over: str = "exons",
    reduce: str = "sum",
    log: bool = False,
    strand: str | None = None,
    organism_index: int = 0,
    device: str | torch.device = "cuda",
    show_progress: bool = True,
):
    """Aggregate whole-chromosome predictions into a per-gene × per-track table.

    Tiles each chromosome, streams every tile's predictions through a
    :class:`~alphagenome_pytorch.aggregation.GeneCountAccumulator`, and returns a
    single :class:`~alphagenome_pytorch.aggregation.GeneCounts` (``.to_anndata()``
    for an AnnData).

    Args:
        model: loaded AlphaGenome model.
        fasta_path: reference genome FASTA (or a prebuilt provider).
        annotation_path: GTF/parquet gene annotation (or a prebuilt
            ``GeneAnnotation``); needs exon rows when ``over="exons"``.
        head: prediction head (e.g. ``"rna_seq"``).
        output_path: optional ``.h5ad`` to write (via ``GeneCounts.to_anndata``).
        chromosomes: default ``chr1..22, chrX``.
        config: :class:`TilingConfig` (use ``crop_bp`` to trim edge artifacts).
        track_indices / track_names / track_strands: track subset + labels; strands
            are required for ``strand="match"``.
        over: ``"exons"`` (default) or ``"gene_body"``.
        reduce: ``"sum"`` (default, count-like) or ``"mean"``.
        log: if True, apply ``log1p`` after the reduce.
        strand: ``None``/``"match"``/``"merge"`` post-processing.

    Returns:
        A :class:`GeneCounts` with ``B == 1`` (whole run collapsed to one table).
    """
    from ...aggregation import GeneCountAccumulator
    from ...variant_scoring.annotations import GeneAnnotation

    config = config or TilingConfig()
    if chromosomes is None:
        chromosomes = list(DEFAULT_CHROMOSOMES)

    # `fasta_path` / `annotation_path` accept prebuilt objects too (handy for tests
    # and for reusing an already-loaded genome / annotation).
    if isinstance(fasta_path, GenomeSequenceProvider):
        genome = fasta_path
    else:
        genome = GenomeSequenceProvider(fasta_path, chromosomes=set(chromosomes), cache=True)
    chromosomes = [c for c in chromosomes if c in genome.chrom_sizes]
    if not chromosomes:
        raise ValueError("No valid chromosomes found in genome")

    head_config = _resolve_head_config(model, head, config.resolution)
    if track_indices is None:
        track_indices = list(range(head_config['num_tracks']))

    # Build (and length-validate) the track-metadata frame up front, so a
    # track_names / track_strands mismatch fails now instead of after inference.
    track_frame = _build_track_frame(track_indices, track_names, track_strands)

    annotation = (
        annotation_path if isinstance(annotation_path, GeneAnnotation)
        else GeneAnnotation(annotation_path)
    )
    if over == "exons" and not annotation.has_exon_annotations():
        raise ValueError(
            "over='exons' needs an annotation with exon rows, but none were found "
            f"in {annotation_path!r}. Provide a GTF/parquet that includes exon "
            "features, or use over='gene_body'."
        )
    accumulator = GeneCountAccumulator(
        annotation, resolution=config.resolution, over=over, reduce=reduce,
    )

    print(f"Aggregating {head} over {over} for {len(chromosomes)} chromosomes: {chromosomes}")
    for chrom in chromosomes:
        print(f"\nProcessing {chrom}...")
        output_length = genome.chrom_sizes[chrom] // config.resolution
        for out_start, kept in _iter_tile_predictions(
            model, genome, chrom, head, config, track_indices, output_length,
            organism_index=organism_index, device=device, show_progress=show_progress,
        ):
            start_bp = out_start * config.resolution
            end_bp = start_bp + kept.shape[0] * config.resolution
            accumulator.add_tile(kept, chrom, start_bp, end_bp)
        print(f"  Genes so far: {accumulator.n_genes}")

    gene_counts = accumulator.to_gene_counts(track_metadata=track_frame, log=log, strand=strand)

    if output_path is not None:
        adata = gene_counts.to_anndata()
        adata.write_h5ad(str(output_path))
        print(f"\nWrote AnnData ({adata.shape[0]} tracks x {adata.shape[1]} genes) to {output_path}")

    return gene_counts
