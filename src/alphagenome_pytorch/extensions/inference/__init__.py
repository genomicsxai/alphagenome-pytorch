"""Inference utilities for AlphaGenome.

This module provides tools for running inference across full chromosomes
and generating BigWig output files.

Example:
    >>> from alphagenome_pytorch import AlphaGenome
    >>> from alphagenome_pytorch.extensions.inference import (
    ...     TilingConfig,
    ...     predict_full_chromosome,
    ...     predict_full_chromosomes_to_bigwig,
    ... )
    >>>
    >>> model = AlphaGenome.from_pretrained('model.pth', device='cuda')
    >>>
    >>> # Configure tiling (default: non-overlapping)
    >>> config = TilingConfig(resolution=128, crop_bp=0)
    >>>
    >>> # Predict single chromosome
    >>> preds = predict_full_chromosome(
    ...     model, 'hg38.fa', chrom='chr1', head='atac', config=config
    ... )
    >>>
    >>> # Predict multiple chromosomes to BigWig
    >>> predict_full_chromosomes_to_bigwig(
    ...     model, 'hg38.fa', './predictions/', head='atac', config=config
    ... )
"""

from .full_chromosome import (
    TilingConfig,
    GenomeSequenceProvider,
    predict_full_chromosome,
    predict_full_chromosomes_to_bigwig,
    write_bigwig,
    HEAD_CONFIGS,
)
from .regions import (
    BedRegion,
    RegionInfo,
    center_crop,
    pad_to_window,
    parse_bed,
    parse_locus,
    predict_region,
    predict_region_auto,
    predict_sequence_auto,
    predict_single_window,
    read_fasta_sequences,
    write_region_bigwig,
    write_regions_merged_bigwig,
    write_sequence_npz,
)

__all__ = [
    'TilingConfig',
    'GenomeSequenceProvider',
    'predict_full_chromosome',
    'predict_full_chromosomes_to_bigwig',
    'write_bigwig',
    'HEAD_CONFIGS',
    # Region / locus / sequence prediction
    'BedRegion',
    'RegionInfo',
    'center_crop',
    'pad_to_window',
    'parse_bed',
    'parse_locus',
    'predict_region',
    'predict_region_auto',
    'predict_sequence_auto',
    'predict_single_window',
    'read_fasta_sequences',
    'write_region_bigwig',
    'write_regions_merged_bigwig',
    'write_sequence_npz',
]
