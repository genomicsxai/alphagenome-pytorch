"""Region-level prediction: locus, BED, and raw FASTA sequences.

Complements :mod:`full_chromosome` by offering per-region inference with
explicit handling for inputs shorter or longer than the model window:

- **pad**: short inputs are centered in a full window (reference flanks
  fetched from the genome for locus/BED; N/A for raw FASTA).
- **cut**: long inputs are center-cropped to a single window.
- **tile**: long inputs are tiled with overlap and stitched (same logic as
  full-chromosome inference).

Example:
    >>> from alphagenome_pytorch.extensions.inference import (
    ...     TilingConfig, GenomeSequenceProvider, predict_region,
    ... )
    >>> genome = GenomeSequenceProvider('hg38.fa', chromosomes={'chr1'})
    >>> config = TilingConfig(resolution=128, crop_bp=16384)
    >>> preds, info = predict_region(
    ...     model, genome, chrom='chr1', start=10_000, end=2_000_000,
    ...     head='atac', config=config, tile=True,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import full_chromosome as _fc
from .full_chromosome import (
    HEAD_CONFIGS,
    GenomeSequenceProvider,
    TilingConfig,
    _generate_tiles,
    _sequence_to_onehot,
    _ensure_deps,
)


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class RegionInfo:
    """Metadata describing how a region was processed.

    Attributes:
        name: Region name (from BED col 4 or synthesized from coordinates).
        chrom: Chromosome (None for raw FASTA sequences).
        start: Genomic start coordinate (0 for raw FASTA; shifts to 0 after cut).
        end: Genomic end coordinate (sequence length for raw FASTA).
        length_bp: Original input length.
        handling: One of "single", "padded", "cut", "tiled".
        tile_count: Number of tiles used (1 for non-tiled).
        warnings: List of warning messages produced during processing.
    """
    name: str
    chrom: str | None
    start: int
    end: int
    length_bp: int
    handling: str
    tile_count: int
    warnings: list[str]


@dataclass
class BedRegion:
    """A parsed BED row."""
    chrom: str
    start: int
    end: int
    name: str


# ============================================================================
# Parsing helpers
# ============================================================================


def parse_locus(locus: str) -> tuple[str, int, int]:
    """Parse a locus string like 'chr1:1000-2000' (commas allowed in numbers)."""
    if ":" not in locus or "-" not in locus:
        raise ValueError(
            f"Invalid locus {locus!r}. Expected format 'chrom:start-end' "
            "(e.g. 'chr1:10000-141072')."
        )
    chrom, _, rest = locus.partition(":")
    start_str, _, end_str = rest.partition("-")
    start = int(start_str.replace(",", ""))
    end = int(end_str.replace(",", ""))
    if start < 0:
        raise ValueError(f"Invalid locus {locus!r}: start ({start}) must be ≥ 0")
    if end <= start:
        raise ValueError(f"Invalid locus {locus!r}: end ({end}) must be > start ({start})")
    return chrom, start, end


def parse_bed(path: str | Path) -> list[BedRegion]:
    """Parse a BED file (cols: chrom, start, end, optional name).

    Lines beginning with '#', 'track', or 'browser' are skipped.
    """
    path = Path(path)
    regions: list[BedRegion] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) < 3:
                raise ValueError(
                    f"{path}:{lineno}: BED line needs ≥ 3 fields (chrom start end), "
                    f"got {len(fields)}: {line!r}"
                )
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: bad start/end: {e}")
            if start < 0:
                raise ValueError(f"{path}:{lineno}: start ({start}) must be ≥ 0")
            if end <= start:
                raise ValueError(
                    f"{path}:{lineno}: end ({end}) must be > start ({start})"
                )
            name = fields[3] if len(fields) >= 4 else f"{chrom}_{start}_{end}"
            regions.append(BedRegion(chrom=chrom, start=start, end=end, name=name))
    if not regions:
        raise ValueError(f"No regions parsed from {path}")
    return regions


def read_fasta_sequences(path: str | Path) -> list[tuple[str, np.ndarray]]:
    """Read a FASTA file as a list of (name, one_hot) pairs.

    Uses pyfaidx for indexed access. Each sequence is one-hot encoded with
    the same scheme as :class:`GenomeSequenceProvider` (unknown bases → 0.25).
    """
    _ensure_deps()
    fasta = _fc.pyfaidx.Fasta(str(path))
    try:
        return [(name, _sequence_to_onehot(str(fasta[name][:]))) for name in fasta.keys()]
    finally:
        fasta.close()


# ============================================================================
# Sequence reshaping
# ============================================================================


def pad_to_window(seq: np.ndarray, window_size: int) -> tuple[np.ndarray, int]:
    """Center ``seq`` in a window of ``window_size`` bp, padding with N (0.25).

    Returns (padded, left_pad) where ``left_pad`` is the bp offset at which
    the original sequence starts in the padded window.
    """
    if seq.ndim != 2 or seq.shape[1] != 4:
        raise ValueError(f"Expected seq shape (L, 4), got {seq.shape}")
    L = seq.shape[0]
    if L > window_size:
        raise ValueError(f"Sequence ({L}bp) longer than window ({window_size}bp)")
    if L == window_size:
        return seq.copy(), 0
    padded = np.full((window_size, 4), 0.25, dtype=np.float32)
    left_pad = (window_size - L) // 2
    padded[left_pad:left_pad + L] = seq
    return padded, left_pad


def center_crop(seq: np.ndarray, window_size: int) -> tuple[np.ndarray, int]:
    """Center-crop ``seq`` to ``window_size`` bp.

    Returns (cropped, crop_start) where ``crop_start`` is the bp offset at
    which the crop began within the original sequence.
    """
    if seq.ndim != 2 or seq.shape[1] != 4:
        raise ValueError(f"Expected seq shape (L, 4), got {seq.shape}")
    L = seq.shape[0]
    if L < window_size:
        raise ValueError(f"Sequence ({L}bp) shorter than window ({window_size}bp)")
    if L == window_size:
        return seq.copy(), 0
    crop_start = (L - window_size) // 2
    return seq[crop_start:crop_start + window_size].copy(), crop_start


# ============================================================================
# Track-extraction helpers (copied pattern from predict_full_chromosome)
# ============================================================================


def _resolve_head_config(model, head: str) -> dict:
    inner = getattr(model, "_orig_mod", model)
    heads = getattr(inner, "heads", None)
    head_module = heads[head] if heads is not None and head in heads else None
    if head_module is not None:
        return {
            "num_tracks": head_module.num_tracks,
            "resolutions": list(head_module.resolutions),
        }
    if head in HEAD_CONFIGS:
        return HEAD_CONFIGS[head]
    available = list(heads.keys()) if heads is not None else list(HEAD_CONFIGS.keys())
    raise ValueError(f"Unknown head: {head}. Available: {available}")


@torch.no_grad()
def _batch_predict(
    model,
    sequences: list[np.ndarray],
    *,
    head: str,
    resolution: int,
    organism_index: int,
    device: torch.device,
) -> np.ndarray:
    """Stack, forward, and return head predictions as (batch, seq_len, n_tracks)."""
    batch_seq = torch.tensor(np.stack(sequences), device=device)
    batch_org = torch.tensor(
        [organism_index] * len(sequences), device=device, dtype=torch.long,
    )
    preds = model.predict(
        batch_seq, batch_org,
        resolutions=(resolution,),
        heads=(head,),
    )
    return preds[head][resolution].cpu().numpy()


# ============================================================================
# Core region prediction
# ============================================================================


def predict_single_window(
    model,
    window_seq: np.ndarray,
    *,
    head: str,
    config: TilingConfig,
    track_indices: list[int] | None = None,
    organism_index: int = 0,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Run one forward pass on a single pre-fetched window.

    Args:
        window_seq: One-hot sequence of shape (window_size, 4).
        track_indices: Optional subset of tracks to return.

    Returns:
        Predictions of shape (window_size // resolution, n_tracks).
    """
    if window_seq.shape != (config.window_size, 4):
        raise ValueError(
            f"Expected window shape ({config.window_size}, 4), got {window_seq.shape}"
        )
    head_config = _resolve_head_config(model, head)
    if config.resolution not in head_config["resolutions"]:
        raise ValueError(
            f"Head '{head}' does not support resolution {config.resolution}. "
            f"Supported: {head_config['resolutions']}"
        )
    device = torch.device(device)
    head_preds = _batch_predict(
        model, [window_seq],
        head=head, resolution=config.resolution,
        organism_index=organism_index, device=device,
    )  # (1, window_size//res, n_tracks)
    out = head_preds[0]
    if track_indices is not None:
        out = out[:, track_indices]
    return out.astype(np.float32)


def predict_region(
    model,
    genome: GenomeSequenceProvider,
    *,
    chrom: str,
    start: int,
    end: int,
    head: str,
    config: TilingConfig,
    track_indices: list[int] | None = None,
    organism_index: int = 0,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Run tiled inference over an arbitrary genomic region and stitch.

    The region may be longer than the window; overlapping tiles with
    ``config.crop_bp`` edge cropping are stitched into a single array.

    Returns predictions of shape ((end-start) // resolution, n_tracks).
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start})")

    head_config = _resolve_head_config(model, head)
    if config.resolution not in head_config["resolutions"]:
        raise ValueError(
            f"Head '{head}' does not support resolution {config.resolution}. "
            f"Supported: {head_config['resolutions']}"
        )

    region_len = end - start
    region_len_res = region_len // config.resolution
    # Translate locus-local tile plan into genomic coords
    local_tiles = _generate_tiles(region_len, config)
    if not local_tiles:
        raise ValueError(f"No tiles generated for region {chrom}:{start}-{end}")
    tiles = [(ws + start, we + start, ks, ke) for (ws, we, ks, ke) in local_tiles]

    n_out_tracks = (
        len(track_indices) if track_indices is not None else head_config["num_tracks"]
    )
    output = np.zeros((region_len_res, n_out_tracks), dtype=np.float32)

    device = torch.device(device)
    for bstart in range(0, len(tiles), config.batch_size):
        batch = tiles[bstart:bstart + config.batch_size]
        seqs = [genome.fetch(chrom, ws, we) for (ws, we, _, _) in batch]
        head_preds = _batch_predict(
            model, seqs,
            head=head, resolution=config.resolution,
            organism_index=organism_index, device=device,
        )
        if track_indices is not None:
            head_preds = head_preds[:, :, track_indices]

        for i, (window_start, _window_end, keep_start, keep_end) in enumerate(batch):
            keep_start_res = keep_start // config.resolution
            keep_end_res = keep_end // config.resolution
            genome_pos_res = (window_start - start) // config.resolution + keep_start_res

            out_start = max(0, genome_pos_res)
            out_end = min(region_len_res, genome_pos_res + (keep_end_res - keep_start_res))
            if out_start >= out_end:
                continue
            pred_start = keep_start_res + (out_start - genome_pos_res)
            pred_end = pred_start + (out_end - out_start)
            output[out_start:out_end] = head_preds[i, pred_start:pred_end]

    return output


# ============================================================================
# High-level dispatch for CLI
# ============================================================================


def predict_region_auto(
    model,
    genome: GenomeSequenceProvider,
    *,
    chrom: str,
    start: int,
    end: int,
    head: str,
    config: TilingConfig,
    tile: bool = False,
    name: str | None = None,
    track_indices: list[int] | None = None,
    organism_index: int = 0,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, RegionInfo]:
    """Predict over a genomic region, dispatching on length and ``tile``.

    * len < window: pad with real reference flanks (single forward pass).
    * len == window: single forward pass.
    * len > window:
        - ``tile=True``:  tiled/stitched inference.
        - ``tile=False``: center-cut to one window (with a warning).

    Returns (predictions, info). Predictions have shape
    ``((effective_end - effective_start) // resolution, n_tracks)``. The
    effective range matches ``info.start``/``info.end`` (which may be a
    cut center-window rather than the original ``[start, end)``).
    """
    if start < 0:
        raise ValueError(
            f"{chrom}:{start}-{end}: start ({start}) must be ≥ 0 (chromosome "
            "coordinates are non-negative)"
        )
    if chrom not in genome.chrom_sizes:
        raise ValueError(f"Chromosome {chrom!r} not found in genome")
    chrom_len = genome.chrom_sizes[chrom]
    if end > chrom_len:
        raise ValueError(
            f"{chrom}:{start}-{end}: end ({end}) exceeds chromosome length ({chrom_len})"
        )

    region_len = end - start
    W = config.window_size
    warnings: list[str] = []
    name = name or f"{chrom}_{start}_{end}"

    if region_len == W:
        window = genome.fetch(chrom, start, end)
        preds = predict_single_window(
            model, window,
            head=head, config=config,
            track_indices=track_indices, organism_index=organism_index, device=device,
        )
        return preds, RegionInfo(
            name=name, chrom=chrom, start=start, end=end,
            length_bp=region_len, handling="single", tile_count=1, warnings=warnings,
        )

    if region_len < W:
        # Fit a W-bp window centered on the region midpoint; shift inward so
        # the window stays within chromosome bounds (no negative coords).
        midpoint = (start + end) // 2
        w_start = midpoint - W // 2
        w_end = w_start + W

        if w_start < 0:
            w_start = 0
            w_end = min(W, chrom_len)
            warnings.append(
                f"{chrom}:{start}-{end} ({region_len}bp) padded with reference flanks; "
                f"window shifted to [0, {w_end}) because region sits near chromosome start."
            )
        elif w_end > chrom_len:
            w_end = chrom_len
            w_start = max(0, w_end - W)
            warnings.append(
                f"{chrom}:{start}-{end} ({region_len}bp) padded with reference flanks; "
                f"window shifted to [{w_start}, {w_end}) because region sits near chromosome end."
            )
        else:
            warnings.append(
                f"{chrom}:{start}-{end} ({region_len}bp) padded with reference flanks to "
                f"a {W}bp window [{w_start}, {w_end}); output covers only the region."
            )

        # If chromosome is shorter than the window, N-pad the tail (fetch handles it).
        window = genome.fetch(chrom, w_start, w_start + W)
        full_preds = predict_single_window(
            model, window,
            head=head, config=config,
            track_indices=track_indices, organism_index=organism_index, device=device,
        )
        # Slice out just the [start, end) region (coords are always ≥ w_start).
        left_bp = start - w_start
        right_bp = left_bp + region_len
        res = config.resolution
        preds = full_preds[left_bp // res:right_bp // res]
        return preds, RegionInfo(
            name=name, chrom=chrom, start=start, end=end,
            length_bp=region_len, handling="padded", tile_count=1, warnings=warnings,
        )

    # region_len > W
    if tile:
        preds = predict_region(
            model, genome,
            chrom=chrom, start=start, end=end,
            head=head, config=config,
            track_indices=track_indices,
            organism_index=organism_index, device=device,
        )
        tile_count = len(_generate_tiles(region_len, config))
        return preds, RegionInfo(
            name=name, chrom=chrom, start=start, end=end,
            length_bp=region_len, handling="tiled", tile_count=tile_count,
            warnings=warnings,
        )

    # Cut: center a single W-bp window on the region midpoint.
    # Since region_len > W and the region is within chromosome bounds, the
    # centered window is always inside [start, end) ⊆ [0, chrom_len).
    midpoint = (start + end) // 2
    w_start = midpoint - W // 2
    w_end = w_start + W
    warnings.append(
        f"{chrom}:{start}-{end} ({region_len}bp) center-cut to "
        f"{chrom}:{w_start}-{w_end} ({W}bp); pass --tile to predict the full region."
    )
    window = genome.fetch(chrom, w_start, w_end)
    preds = predict_single_window(
        model, window,
        head=head, config=config,
        track_indices=track_indices, organism_index=organism_index, device=device,
    )
    return preds, RegionInfo(
        name=name, chrom=chrom, start=w_start, end=w_end,
        length_bp=region_len, handling="cut", tile_count=1, warnings=warnings,
    )


def predict_sequence_auto(
    model,
    seq: np.ndarray,
    *,
    name: str,
    head: str,
    config: TilingConfig,
    tile: bool = False,
    track_indices: list[int] | None = None,
    organism_index: int = 0,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, RegionInfo]:
    """Predict over a raw FASTA sequence (no genomic coordinates).

    * len < window: ERROR (cannot meaningfully pad without reference context).
    * len == window: single forward pass.
    * len > window: requires ``tile=True``; stitched prediction.
    """
    L = seq.shape[0]
    W = config.window_size

    if L < W:
        raise ValueError(
            f"Sequence {name!r} ({L}bp) is shorter than the model window "
            f"({W}bp); not supported for --sequences. Pad the sequence before "
            "running, or use --locus/--bed with a reference genome."
        )

    if L == W:
        preds = predict_single_window(
            model, seq,
            head=head, config=config,
            track_indices=track_indices, organism_index=organism_index, device=device,
        )
        return preds, RegionInfo(
            name=name, chrom=None, start=0, end=L,
            length_bp=L, handling="single", tile_count=1, warnings=[],
        )

    # L > W
    if not tile:
        raise ValueError(
            f"Sequence {name!r} ({L}bp) is longer than the model window "
            f"({W}bp); pass --tile to enable tiling."
        )

    preds = _predict_raw_sequence_tiled(
        model, seq,
        head=head, config=config,
        track_indices=track_indices,
        organism_index=organism_index, device=device,
    )
    tile_count = len(_generate_tiles(L, config))
    return preds, RegionInfo(
        name=name, chrom=None, start=0, end=L,
        length_bp=L, handling="tiled", tile_count=tile_count, warnings=[],
    )


def _predict_raw_sequence_tiled(
    model,
    seq: np.ndarray,
    *,
    head: str,
    config: TilingConfig,
    track_indices: list[int] | None,
    organism_index: int,
    device: str | torch.device,
) -> np.ndarray:
    """Tiled prediction over a raw one-hot sequence.

    Mirrors :func:`predict_region` but pulls sub-sequences from the in-memory
    array rather than a genome provider. Tiles that extend past the sequence
    end are right-padded with N (0.25) — same convention as
    GenomeSequenceProvider for out-of-bounds fetches.
    """
    L = seq.shape[0]
    W = config.window_size
    res = config.resolution

    head_config = _resolve_head_config(model, head)
    if config.resolution not in head_config["resolutions"]:
        raise ValueError(
            f"Head '{head}' does not support resolution {config.resolution}. "
            f"Supported: {head_config['resolutions']}"
        )

    tiles = _generate_tiles(L, config)
    if not tiles:
        raise ValueError(f"No tiles generated for sequence of length {L}")

    out_len_res = L // res
    n_out_tracks = (
        len(track_indices) if track_indices is not None else head_config["num_tracks"]
    )
    output = np.zeros((out_len_res, n_out_tracks), dtype=np.float32)

    device = torch.device(device)
    for bstart in range(0, len(tiles), config.batch_size):
        batch = tiles[bstart:bstart + config.batch_size]
        sequences = []
        for ws, we, _, _ in batch:
            # Slice with N-padding for out-of-bounds ends
            window = np.full((W, 4), 0.25, dtype=np.float32)
            valid_start = max(0, ws)
            valid_end = min(L, we)
            if valid_start < valid_end:
                dest_start = valid_start - ws
                window[dest_start:dest_start + (valid_end - valid_start)] = seq[
                    valid_start:valid_end
                ]
            sequences.append(window)

        head_preds = _batch_predict(
            model, sequences,
            head=head, resolution=res,
            organism_index=organism_index, device=device,
        )
        if track_indices is not None:
            head_preds = head_preds[:, :, track_indices]

        for i, (ws, _we, keep_start, keep_end) in enumerate(batch):
            keep_start_res = keep_start // res
            keep_end_res = keep_end // res
            out_pos_res = ws // res + keep_start_res

            out_start = max(0, out_pos_res)
            out_end = min(out_len_res, out_pos_res + (keep_end_res - keep_start_res))
            if out_start >= out_end:
                continue
            pred_start = keep_start_res + (out_start - out_pos_res)
            pred_end = pred_start + (out_end - out_start)
            output[out_start:out_end] = head_preds[i, pred_start:pred_end]

    return output


# ============================================================================
# Output writers
# ============================================================================


def write_region_bigwig(
    predictions: np.ndarray,
    output_path: str | Path,
    chrom: str,
    start: int,
    chrom_sizes: dict[str, int],
    resolution: int = 128,
    track_names: list[str] | None = None,
) -> list[Path]:
    """Write per-region predictions to BigWig file(s), one per track.

    Writes entries at genomic coordinates ``[start, start + len(predictions)*resolution)``.
    Unlike :func:`write_bigwig` (full-chromosome), this function places the
    predictions at an arbitrary genomic offset, so gaps between regions in a
    BED file are preserved as missing data.
    """
    _ensure_deps()

    output_path = Path(output_path)
    n_tracks = predictions.shape[1]
    if track_names is None:
        track_names = [f"track_{i}" for i in range(n_tracks)]

    written: list[Path] = []
    for i, tname in enumerate(track_names):
        if n_tracks > 1:
            bw_path = output_path.parent / f"{output_path.stem}_{tname}{output_path.suffix}"
        else:
            bw_path = output_path
        bw = _fc.pyBigWig.open(str(bw_path), "w")
        bw.addHeader([(k, v) for k, v in chrom_sizes.items()])

        track_data = predictions[:, i].astype(np.float64)
        CHUNK = 1_000_000
        for cstart in range(0, len(track_data), CHUNK):
            cend = min(cstart + CHUNK, len(track_data))
            bw.addEntries(
                chrom, start + cstart * resolution,
                values=track_data[cstart:cend].tolist(),
                span=resolution, step=resolution,
            )
        bw.close()
        written.append(bw_path)
    return written


def write_regions_merged_bigwig(
    entries: list[tuple[np.ndarray, str, int]],
    output_path: str | Path,
    chrom_sizes: dict[str, int],
    resolution: int = 128,
    track_names: list[str] | None = None,
) -> list[Path]:
    """Write many (predictions, chrom, start) entries into one BigWig per track.

    ``entries`` must be sorted by (chrom order in ``chrom_sizes``, start).
    BigWig requires sorted, non-overlapping entries.
    """
    _ensure_deps()

    if not entries:
        raise ValueError("No entries to write")

    output_path = Path(output_path)
    n_tracks = entries[0][0].shape[1]
    if track_names is None:
        track_names = [f"track_{i}" for i in range(n_tracks)]

    chrom_order = list(chrom_sizes.keys())
    chrom_idx = {c: i for i, c in enumerate(chrom_order)}
    sorted_entries = sorted(entries, key=lambda e: (chrom_idx.get(e[1], 1 << 30), e[2]))

    # BigWig requires non-overlapping entries within a chromosome. Detect overlaps
    # now and raise a clear error rather than letting pyBigWig fail cryptically.
    prev_chrom: str | None = None
    prev_start: int = 0
    prev_end: int = 0
    for preds, chrom, start in sorted_entries:
        end = start + preds.shape[0] * resolution
        if chrom == prev_chrom and start < prev_end:
            raise ValueError(
                f"Overlapping regions on {chrom}: "
                f"[{prev_start}, {prev_end}) overlaps [{start}, {end}). "
                f"BigWig requires non-overlapping entries — deduplicate or merge "
                f"input regions before writing."
            )
        prev_chrom, prev_start, prev_end = chrom, start, end

    written: list[Path] = []
    for i, tname in enumerate(track_names):
        if n_tracks > 1:
            bw_path = output_path.parent / f"{output_path.stem}_{tname}{output_path.suffix}"
        else:
            bw_path = output_path
        bw = _fc.pyBigWig.open(str(bw_path), "w")
        bw.addHeader([(k, v) for k, v in chrom_sizes.items()])

        for preds, chrom, start in sorted_entries:
            track_data = preds[:, i].astype(np.float64)
            CHUNK = 1_000_000
            for cstart in range(0, len(track_data), CHUNK):
                cend = min(cstart + CHUNK, len(track_data))
                bw.addEntries(
                    chrom, start + cstart * resolution,
                    values=track_data[cstart:cend].tolist(),
                    span=resolution, step=resolution,
                )
        bw.close()
        written.append(bw_path)
    return written


def write_sequence_npz(
    predictions: np.ndarray,
    output_path: str | Path,
    *,
    info: RegionInfo,
    head: str,
    resolution: int,
    track_names: list[str] | None = None,
) -> Path:
    """Write one sequence's predictions to an NPZ bundle.

    Stored keys:
        * ``predictions``: float32 array (seq_len//res, n_tracks)
        * ``track_names``: object array of per-track names
        * ``metadata``: dict with name/head/resolution/handling/tile_count/length_bp
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = np.array(track_names or [f"track_{i}" for i in range(predictions.shape[1])])
    metadata = np.array(
        {
            "name": info.name,
            "chrom": info.chrom,
            "start": info.start,
            "end": info.end,
            "length_bp": info.length_bp,
            "head": head,
            "resolution_bp": resolution,
            "handling": info.handling,
            "tile_count": info.tile_count,
        },
        dtype=object,
    )
    np.savez_compressed(
        output_path,
        predictions=predictions.astype(np.float32),
        track_names=names,
        metadata=metadata,
    )
    return output_path
