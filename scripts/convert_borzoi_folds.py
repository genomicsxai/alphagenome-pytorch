#!/usr/bin/env python3
"""Convert Borzoi sequence folds to AlphaGenome folds.

This script reads Borzoi sequence folds (~196kb regions) and converts them to
AlphaGenome format (1Mb regions) with proper train/valid/test splits.

To prevent data leakage, validation and test regions are excluded if their
1Mb windows overlap with any training region's 1Mb window within the same fold.

Input format (sequences_human.bed.gz):
    chr4    82524421    82721029    fold0
    chr13   18604798    18801406    fold0
    ...

Output structure:
    output_dir/
        FOLD_0/
            train.bed
            valid.bed
            test.bed
        FOLD_1/
            train.bed
            valid.bed
            test.bed
        ...

Usage:
    python scripts/convert_borzoi_folds.py \\
        --input sequences_human.bed.gz \\
        --output-dir data/alphagenome_folds

References:
    AlphaGenome uses 1Mb (2^20 = 1,048,576 bp) input/output windows.
    Borzoi uses ~196kb (196,608 bp) target intervals.
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

# AlphaGenome sequence length (1Mb)
ALPHAGENOME_SEQ_LENGTH = 2**20  # 1,048,576 bp

# From https://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes
HG38_CHROMOSOME_LENGTHS = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chrX": 156040895,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr11": 135086622,
    "chr10": 133797422,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr20": 64444167,
    "chr19": 58617616,
    "chrY": 57227415,
    "chr22": 50818468,
    "chr21": 46709983,
}

# From https://hgdownload.cse.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes
MM10_CHROMOSOME_LENGTHS = {
    "chr1": 195471971,
    "chr2": 182113224,
    "chrX": 171031299,
    "chr3": 160039680,
    "chr4": 156508116,
    "chr5": 151834684,
    "chr6": 149736546,
    "chr7": 145441459,
    "chr10": 130694993,
    "chr8": 129401213,
    "chr14": 124902244,
    "chr9": 124595110,
    "chr11": 122082543,
    "chr13": 120421639,
    "chr12": 120129022,
    "chr15": 104043685,
    "chr16": 98207768,
    "chr17": 94987271,
    "chrY": 91744698,
    "chr18": 90702639,
    "chr19": 61431566,
}

# AlphaGenome fold configurations (from datasets.py)
# Maps model fold to (valid_fold, test_fold)
ALPHAGENOME_FOLDS = {
    "FOLD_0": {"valid": "fold0", "test": "fold1"},
    "FOLD_1": {"valid": "fold3", "test": "fold4"},
    "FOLD_2": {"valid": "fold2", "test": "fold5"},
    "FOLD_3": {"valid": "fold6", "test": "fold7"},
}

ALL_REGION_FOLDS = [f"fold{i}" for i in range(8)]


class Region(NamedTuple):
    """A genomic region."""

    chrom: str
    start: int
    end: int
    fold: str

    @property
    def midpoint(self) -> int:
        return (self.start + self.end) // 2

    def to_alphagenome(self, chrom_len: int | None = None) -> Region | None:
        """Extend region to 1Mb centered on midpoint, with clamping to boundaries.

        Args:
            chrom_len: Optional chromosome length for boundary clamping.

        Returns:
            Region extended to 1Mb, or None if the chromosome is too short.
        """
        if chrom_len is not None and chrom_len < ALPHAGENOME_SEQ_LENGTH:
            return None

        mid = self.midpoint
        half_len = ALPHAGENOME_SEQ_LENGTH // 2
        start = mid - half_len
        end = mid + half_len

        # Clamp to boundaries
        if start < 0:
            start = 0
            end = ALPHAGENOME_SEQ_LENGTH
        elif chrom_len is not None and end > chrom_len:
            end = chrom_len
            start = chrom_len - ALPHAGENOME_SEQ_LENGTH

        # Safety check (should not happen if chrom_len >= SEQ_LENGTH)
        if start < 0 or (chrom_len is not None and end > chrom_len):
            return None

        return Region(
            chrom=self.chrom,
            start=int(start),
            end=int(end),
            fold=self.fold,
        )

    def to_bed_line(self) -> str:
        """Convert to BED format line (chrom, start, end)."""
        return f"{self.chrom}\t{self.start}\t{self.end}"


def load_borzoi_regions(bed_path: str | Path) -> list[Region]:
    """Load regions from Borzoi BED file.

    Args:
        bed_path: Path to BED file (can be gzipped).

    Returns:
        List of Region objects.
    """
    regions = []
    path = Path(bed_path)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            chrom, start, end, fold = parts[0], int(parts[1]), int(parts[2]), parts[3]
            regions.append(Region(chrom, start, end, fold))

    return regions


def build_interval_index(
    regions: list[Region],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build sorted interval arrays for efficient overlap queries.

    Args:
        regions: List of regions (already extended to 1Mb).

    Returns:
        Dict mapping chromosome to (starts_array, ends_array).
    """
    by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for region in regions:
        by_chrom[region.chrom].append((region.start, region.end))

    result = {}
    for chrom, intervals in by_chrom.items():
        intervals.sort()  # Sort by start position
        starts = np.array([i[0] for i in intervals], dtype=np.int64)
        ends = np.array([i[1] for i in intervals], dtype=np.int64)
        result[chrom] = (starts, ends)

    return result


def has_overlap(
    region: Region,
    interval_index: dict[str, tuple[np.ndarray, np.ndarray]],
) -> bool:
    """Check if a region overlaps with any interval in the index.

    Two intervals [a1, a2) and [b1, b2) overlap if a1 < b2 and b1 < a2.

    Args:
        region: Region to check.
        interval_index: Dict mapping chromosome to (starts, ends) arrays.

    Returns:
        True if the region overlaps with any indexed interval.
    """
    if region.chrom not in interval_index:
        return False

    starts, ends = interval_index[region.chrom]

    # Binary search to find potential overlapping intervals
    # An interval [s, e) overlaps with [region.start, region.end) if:
    # s < region.end AND e > region.start

    # Find intervals where start < region.end
    idx_end = np.searchsorted(starts, region.end, side="left")
    if idx_end == 0:
        return False

    # Check if any of these have end > region.start
    candidate_ends = ends[:idx_end]
    return np.any(candidate_ends > region.start)


def filter_overlapping_regions(
    eval_regions: list[Region],
    train_index: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[list[Region], int]:
    """Filter out eval regions that overlap with training regions.

    Args:
        eval_regions: List of validation or test regions.
        train_index: Interval index for training regions.

    Returns:
        Tuple of (filtered_regions, n_removed).
    """
    filtered = []
    n_removed = 0

    for region in eval_regions:
        if has_overlap(region, train_index):
            n_removed += 1
        else:
            filtered.append(region)

    return filtered, n_removed


def write_bed_file(regions: list[Region], path: Path) -> None:
    """Write regions to BED file.

    Args:
        regions: List of regions to write.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for region in regions:
            f.write(region.to_bed_line() + "\n")


def convert_borzoi_to_alphagenome(
    input_path: str | Path,
    output_dir: str | Path,
    organism: str = "human",
    verbose: bool = True,
) -> dict[str, dict[str, int]]:
    """Convert Borzoi folds to AlphaGenome format.

    Args:
        input_path: Path to Borzoi sequences BED file.
        output_dir: Output directory for fold files.
        organism: Organism name ('human' or 'mouse').
        verbose: Print progress information.

    Returns:
        Statistics dict with counts per fold and split.
    """
    if organism.lower() == "human":
        chrom_lengths = HG38_CHROMOSOME_LENGTHS
    elif organism.lower() == "mouse":
        chrom_lengths = MM10_CHROMOSOME_LENGTHS
    else:
        raise ValueError(f"Unknown organism: {organism}")

    output_dir = Path(output_dir)
    stats: dict[str, dict[str, int]] = {}

    # Load all Borzoi regions
    if verbose:
        print(f"Loading regions from {input_path}...")
    borzoi_regions = load_borzoi_regions(input_path)
    if verbose:
        print(f"  Loaded {len(borzoi_regions)} Borzoi regions")

    # Group by fold
    regions_by_fold: dict[str, list[Region]] = defaultdict(list)
    for region in borzoi_regions:
        regions_by_fold[region.fold].append(region)

    if verbose:
        print("  Regions per fold:")
        for fold in sorted(regions_by_fold.keys()):
            print(f"    {fold}: {len(regions_by_fold[fold])}")

    # Convert each region to 1Mb with clamping
    alphagenome_regions_by_fold: dict[str, list[Region]] = {}
    for fold, regions in regions_by_fold.items():
        ag_regions = []
        for r in regions:
            chrom_len = chrom_lengths.get(r.chrom)
            ag_r = r.to_alphagenome(chrom_len=chrom_len)
            if ag_r:
                ag_regions.append(ag_r)
        alphagenome_regions_by_fold[fold] = ag_regions

    # Process each model fold
    for model_fold, config in ALPHAGENOME_FOLDS.items():
        if verbose:
            print(f"\nProcessing {model_fold}...")

        fold_dir = output_dir / model_fold
        stats[model_fold] = {}

        valid_fold = config["valid"]
        test_fold = config["test"]
        train_folds = [f for f in ALL_REGION_FOLDS if f not in {valid_fold, test_fold}]

        # Collect training regions
        train_regions = []
        for fold in train_folds:
            if fold in alphagenome_regions_by_fold:
                train_regions.extend(alphagenome_regions_by_fold[fold])

        if verbose:
            print(f"  Train folds: {train_folds}")
            print(f"  Valid fold: {valid_fold}")
            print(f"  Test fold: {test_fold}")
            print(f"  Training regions: {len(train_regions)}")

        # Build interval index for training regions
        train_index = build_interval_index(train_regions)

        # Write training regions (no filtering needed)
        write_bed_file(train_regions, fold_dir / "train.bed")
        stats[model_fold]["train"] = len(train_regions)

        # Process validation regions
        valid_regions = alphagenome_regions_by_fold.get(valid_fold, [])
        valid_filtered, valid_removed = filter_overlapping_regions(
            valid_regions, train_index
        )
        write_bed_file(valid_filtered, fold_dir / "valid.bed")
        stats[model_fold]["valid"] = len(valid_filtered)
        stats[model_fold]["valid_removed"] = valid_removed

        if verbose:
            print(
                f"  Valid: {len(valid_filtered)} kept, {valid_removed} removed "
                f"({valid_removed / len(valid_regions) * 100:.1f}% overlap)"
                if valid_regions
                else "  Valid: 0 regions"
            )

        # Process test regions
        test_regions = alphagenome_regions_by_fold.get(test_fold, [])
        test_filtered, test_removed = filter_overlapping_regions(
            test_regions, train_index
        )
        write_bed_file(test_filtered, fold_dir / "test.bed")
        stats[model_fold]["test"] = len(test_filtered)
        stats[model_fold]["test_removed"] = test_removed

        if verbose:
            print(
                f"  Test: {len(test_filtered)} kept, {test_removed} removed "
                f"({test_removed / len(test_regions) * 100:.1f}% overlap)"
                if test_regions
                else "  Test: 0 regions"
            )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert Borzoi sequence folds to AlphaGenome format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to Borzoi sequences BED file (e.g., sequences_human.bed.gz)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory for fold files",
    )
    parser.add_argument(
        "--organism",
        "-r",
        choices=["human", "mouse"],
        default="human",
        help="Organism name (default: human)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    stats = convert_borzoi_to_alphagenome(
        input_path=args.input,
        output_dir=args.output_dir,
        organism=args.organism,
        verbose=not args.quiet,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Fold':<10} {'Train':>10} {'Valid':>10} {'Test':>10} {'V.Rem':>8} {'T.Rem':>8}")
    print("-" * 60)
    for model_fold in ALPHAGENOME_FOLDS:
        s = stats[model_fold]
        print(
            f"{model_fold:<10} {s['train']:>10} {s['valid']:>10} {s['test']:>10} "
            f"{s['valid_removed']:>8} {s['test_removed']:>8}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
