#!/usr/bin/env python
"""
Script to create mock data for AlphaGenome fine-tuning tests.

This script documents how the mock data in tests/fixtures/mock_data/ was generated.
The mock data includes:
- mock_genome.fa: FASTA with chr1 and chr22 (300kb each)
- mock_rnaseq_track1.bw, mock_rnaseq_track2.bw: RNA-seq BigWig files
- mock_atac_track1.bw, mock_atac_track2.bw: ATAC-seq BigWig files
- mock_positions.bed: Positions BED file (training positions)

Usage:
    python create_mock_data.py --output_dir tests/fixtures/mock_data
"""

import argparse
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create mock data for AlphaGenome fine-tuning tests"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/fixtures/mock_data",
        help="Output directory for mock data",
    )
    parser.add_argument(
        "--chrom_size",
        type=int,
        default=300000,
        help="Size of each chromosome (default: 300000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def create_mock_genome(output_path: Path, chrom_size: int, seed: int):
    """Create mock genome FASTA file.

    Args:
        output_path: Path to output FASTA file.
        chrom_size: Size of each chromosome.
        seed: Random seed.
    """
    random.seed(seed)
    bases = ["A", "C", "G", "T"]
    chromosomes = ["chr1", "chr22"]

    with open(output_path, "w") as f:
        for chrom in chromosomes:
            f.write(f">{chrom}\n")
            # Generate sequence in chunks of 80 characters (standard FASTA line length)
            for i in range(0, chrom_size, 80):
                chunk_size = min(80, chrom_size - i)
                seq = "".join(random.choice(bases) for _ in range(chunk_size))
                f.write(f"{seq}\n")

    print(f"Created mock genome: {output_path}")


def create_mock_bigwig(
    output_path: Path,
    chrom_sizes: dict[str, int],
    signal_type: str,
    seed: int,
):
    """Create mock BigWig file with random signal values.

    Args:
        output_path: Path to output BigWig file.
        chrom_sizes: Dict mapping chromosome names to sizes.
        signal_type: Type of signal ('rnaseq' or 'atac').
        seed: Random seed.
    """
    try:
        import pyBigWig
    except ImportError:
        raise ImportError(
            "pyBigWig is required to create BigWig files. "
            "Install with: pip install pyBigWig"
        )

    np.random.seed(seed)

    bw = pyBigWig.open(str(output_path), "w")
    bw.addHeader(list(chrom_sizes.items()))

    for chrom, size in chrom_sizes.items():
        # Generate random signal values
        if signal_type == "rnaseq":
            # RNA-seq: sparse signal with occasional peaks
            signal = np.zeros(size, dtype=np.float32)
            n_peaks = size // 1000
            peak_positions = np.random.randint(0, size, n_peaks)
            for pos in peak_positions:
                peak_width = np.random.randint(10, 100)
                peak_height = np.random.exponential(10)
                start = max(0, pos - peak_width // 2)
                end = min(size, pos + peak_width // 2)
                signal[start:end] += peak_height * np.exp(
                    -0.5 * ((np.arange(end - start) - (end - start) / 2) / (peak_width / 4)) ** 2
                )
        else:
            # ATAC: broader accessibility regions
            signal = np.zeros(size, dtype=np.float32)
            n_peaks = size // 2000
            peak_positions = np.random.randint(0, size, n_peaks)
            for pos in peak_positions:
                peak_width = np.random.randint(100, 500)
                peak_height = np.random.exponential(5)
                start = max(0, pos - peak_width // 2)
                end = min(size, pos + peak_width // 2)
                signal[start:end] += peak_height

        # Add values to BigWig
        # pyBigWig expects 0-indexed, half-open intervals
        starts = np.arange(0, size, dtype=np.int64)
        ends = starts + 1
        values = signal.astype(np.float64)

        # Filter out zeros for smaller file
        nonzero = values > 0.01
        if nonzero.any():
            bw.addEntries(
                [chrom] * nonzero.sum(),
                starts[nonzero].tolist(),
                ends=ends[nonzero].tolist(),
                values=values[nonzero].tolist(),
            )

    bw.close()
    print(f"Created mock BigWig: {output_path}")


def create_mock_bed(
    output_path: Path,
    chrom_sizes: dict[str, int],
    n_positions: int,
    seed: int,
    min_distance_from_edge: int = 70000,
    include_fold_labels: bool = True,
):
    """Create mock BED file with random positions.

    Args:
        output_path: Path to output BED file.
        chrom_sizes: Dict mapping chromosome names to sizes.
        n_positions: Number of positions to generate.
        seed: Random seed.
        min_distance_from_edge: Minimum distance from chromosome edges.
        include_fold_labels: Whether to add fold labels (fold0-fold7) in 4th column.
    """
    random.seed(seed)
    positions = []

    for chrom, size in chrom_sizes.items():
        n_chrom = n_positions // len(chrom_sizes)
        valid_start = min_distance_from_edge
        valid_end = size - min_distance_from_edge

        if valid_end <= valid_start:
            print(f"Warning: Chromosome {chrom} too small for positions, skipping")
            continue

        for i in range(n_chrom):
            pos = random.randint(valid_start, valid_end)
            # Assign fold labels cyclically (fold0-fold7)
            fold_label = f"fold{i % 8}" if include_fold_labels else None
            positions.append((chrom, pos, pos + 1, fold_label))

    # Sort by chromosome and position
    positions.sort(key=lambda x: (x[0], x[1]))

    with open(output_path, "w") as f:
        for chrom, start, end, fold in positions:
            if fold:
                f.write(f"{chrom}\t{start}\t{end}\t{fold}\n")
            else:
                f.write(f"{chrom}\t{start}\t{end}\n")

    print(f"Created mock BED: {output_path} ({len(positions)} positions)")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chrom_sizes = {"chr1": args.chrom_size, "chr22": args.chrom_size}

    # Create mock genome
    create_mock_genome(
        output_dir / "mock_genome.fa",
        args.chrom_size,
        args.seed,
    )

    # Create mock BigWig files
    for i, signal_type in enumerate(["rnaseq", "atac"]):
        for track_num in [1, 2]:
            create_mock_bigwig(
                output_dir / f"mock_{signal_type}_track{track_num}.bw",
                chrom_sizes,
                signal_type,
                args.seed + i * 10 + track_num,
            )

    # Create mock BED file
    create_mock_bed(
        output_dir / "mock_positions.bed",
        chrom_sizes,
        n_positions=20,
        seed=args.seed + 100,
    )



    print(f"\nMock data created in: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
