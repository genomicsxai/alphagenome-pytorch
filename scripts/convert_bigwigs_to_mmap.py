#!/usr/bin/env python
"""Convert BigWig files to memory-mapped numpy arrays for fast loading.

This script converts BigWig signal files to a format optimized for random
access during training. Memory-mapped files provide near-instant reads
without loading entire chromosomes into memory.

Usage:
    # Convert a single BigWig file
    python scripts/convert_bigwigs_to_mmap.py \
        --bigwig signal.bw \
        --output-dir mmap_signals/signal

    # Convert multiple BigWig files (parallel)
    python scripts/convert_bigwigs_to_mmap.py \
        --bigwig *.bw \
        --output-dir mmap_signals/ \
        --workers 8

    # Convert only specific chromosomes
    python scripts/convert_bigwigs_to_mmap.py \
        --bigwig signal.bw \
        --output-dir mmap_signals/signal \
        --chromosomes chr1 chr2 chr3

Then use in training:
    train_ds = GenomicDataset(
        genome_fasta='hg38.fa',
        bigwig_files=['mmap_signals/signal1', 'mmap_signals/signal2'],
        bed_file='positions.bed',
        use_mmap=True,  # Enable mmap mode
    )
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def convert_single_bigwig(
    bigwig_path: str | Path,
    output_dir: str | Path,
    chromosomes: list[str] | None = None,
    dtype: np.dtype = np.float32,
) -> tuple[Path, float, float]:
    """Convert a single BigWig file to mmap format.

    Returns:
        (output_path, time_seconds, size_mb)
    """
    import json
    import pyBigWig

    bigwig_path = Path(bigwig_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    metadata = {
        "source": str(bigwig_path),
        "dtype": str(dtype),
        "chromosomes": {},
    }

    total_size = 0

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
            npy_path = output_dir / npy_filename
            np.save(npy_path, values)

            total_size += values.nbytes

            metadata["chromosomes"][chrom] = {
                "file": npy_filename,
                "size": size,
            }

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.perf_counter() - t0
    size_mb = total_size / 1e6

    return output_dir, elapsed, size_mb


def main():
    parser = argparse.ArgumentParser(
        description="Convert BigWig files to memory-mapped numpy arrays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bigwig",
        type=str,
        nargs="+",
        required=True,
        help="BigWig file(s) to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory. For multiple files, subdirs are created per file.",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="*",
        default=None,
        help="Chromosomes to convert (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for multiple files",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Data type for output arrays",
    )
    args = parser.parse_args()

    bigwig_files = args.bigwig
    output_base = Path(args.output_dir)
    dtype = np.float32 if args.dtype == "float32" else np.float16

    print("=" * 70)
    print("BigWig to Mmap Converter")
    print("=" * 70)
    print(f"Input files: {len(bigwig_files)}")
    print(f"Output dir: {output_base}")
    print(f"Chromosomes: {args.chromosomes or 'all'}")
    print(f"Dtype: {dtype}")
    print(f"Workers: {args.workers}")
    print()

    # Determine output paths
    if len(bigwig_files) == 1:
        # Single file: output directly to output_dir
        outputs = [(bigwig_files[0], output_base)]
    else:
        # Multiple files: create subdirs per file
        outputs = [
            (bw, output_base / Path(bw).stem)
            for bw in bigwig_files
        ]

    # Convert files
    total_time = 0.0
    total_size = 0.0

    if len(outputs) == 1 or args.workers == 1:
        # Sequential
        for bw_path, out_path in outputs:
            print(f"Converting: {Path(bw_path).name}")
            _, elapsed, size_mb = convert_single_bigwig(
                bw_path, out_path, args.chromosomes, dtype
            )
            total_time += elapsed
            total_size += size_mb
            print(f"  -> {out_path} ({elapsed:.1f}s, {size_mb:.1f} MB)")
    else:
        # Parallel
        n_workers = min(len(outputs), args.workers)
        print(f"Converting {len(outputs)} files with {n_workers} workers...")
        print()

        def convert_one(item: tuple[str, Path]) -> tuple[str, Path, float, float]:
            bw_path, out_path = item
            _, elapsed, size_mb = convert_single_bigwig(
                bw_path, out_path, args.chromosomes, dtype
            )
            return bw_path, out_path, elapsed, size_mb

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(convert_one, item): item
                for item in outputs
            }

            for future in as_completed(futures):
                bw_path, out_path, elapsed, size_mb = future.result()
                total_time += elapsed
                total_size += size_mb
                print(f"  {Path(bw_path).name} -> {out_path.name}/ ({elapsed:.1f}s, {size_mb:.1f} MB)")

    print()
    print("-" * 70)
    print(f"Done! Converted {len(outputs)} file(s)")
    print(f"Total time: {total_time:.1f}s")
    print(f"Total size: {total_size:.1f} MB")
    print()
    print("Usage in training:")
    print("  train_ds = GenomicDataset(")
    print(f"      bigwig_files=['{outputs[0][1]}', ...],")
    print("      use_mmap=True,")
    print("  )")


if __name__ == "__main__":
    main()
