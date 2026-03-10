#!/usr/bin/env python
"""Convert GTF annotation file to optimized Parquet format.

This script converts a GTF/GFF file (e.g., GENCODE annotation) to Parquet format
for fast loading with the GeneAnnotation class. Loading a ~1.5GB GTF takes several
minutes, while loading the equivalent Parquet takes seconds.

Usage:
    python scripts/convert_gtf_to_parquet.py \
        --input /path/to/gencode.v49.annotation.gtf \
        --output /path/to/gencode.v49.parquet

The output Parquet file preserves all features (genes, transcripts, exons, etc.)
and all columns from the original GTF, with optimized column types for fast loading.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def convert_gtf_to_parquet(
    input_path: str | Path,
    output_path: str | Path,
    compression: str = "snappy",
) -> None:
    """Convert GTF file to Parquet format.

    Args:
        input_path: Path to input GTF/GFF file
        output_path: Path to output Parquet file
        compression: Parquet compression codec ('snappy', 'gzip', 'zstd', 'none')
    """
    import pandas as pd
    import pyranges

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading GTF file: {input_path}")
    print("This may take a few minutes for large files...")

    start = time.perf_counter()
    gtf = pyranges.read_gtf(str(input_path))
    read_time = time.perf_counter() - start
    print(f"  Read GTF in {read_time:.1f}s")

    # Convert to DataFrame
    df = gtf.df

    print(f"  Total features: {len(df):,}")
    print(f"  Feature types: {df['Feature'].value_counts().to_dict()}")

    # Optimize column types for smaller file and faster loading
    print("Optimizing column types...")

    # Convert string columns with low cardinality to categorical
    categorical_columns = [
        'Chromosome', 'Feature', 'Strand', 'gene_type', 'gene_biotype',
        'transcript_type', 'transcript_biotype', 'tag', 'Source'
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Ensure integer columns are proper ints
    int_columns = ['Start', 'End', 'exon_number', 'level']
    for col in int_columns:
        if col in df.columns:
            # Handle NaN values by converting to nullable Int64
            if df[col].isna().any():
                df[col] = df[col].astype('Int64')
            else:
                df[col] = df[col].astype('int64')

    # Write to Parquet
    print(f"Writing Parquet file: {output_path}")
    start = time.perf_counter()

    compression_arg = None if compression == 'none' else compression
    df.to_parquet(output_path, compression=compression_arg, index=False)

    write_time = time.perf_counter() - start
    print(f"  Wrote Parquet in {write_time:.1f}s")

    # Report file sizes
    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    output_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\nConversion complete!")
    print(f"  Input size:  {input_size_mb:.1f} MB")
    print(f"  Output size: {output_size_mb:.1f} MB")
    print(f"  Compression ratio: {input_size_mb / output_size_mb:.1f}x")

    # Test loading speed
    print("\nTesting Parquet load speed...")
    start = time.perf_counter()
    _ = pd.read_parquet(output_path)
    load_time = time.perf_counter() - start
    print(f"  Parquet load time: {load_time:.2f}s (vs {read_time:.1f}s for GTF)")
    print(f"  Speedup: {read_time / load_time:.0f}x faster")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GTF annotation file to optimized Parquet format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert GENCODE GTF to Parquet
    python scripts/convert_gtf_to_parquet.py \\
        --input /path/to/gencode.v49.annotation.gtf \\
        --output /path/to/gencode.v49.parquet

    # Use zstd compression for smaller file size
    python scripts/convert_gtf_to_parquet.py \\
        --input annotation.gtf \\
        --output annotation.parquet \\
        --compression zstd
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input GTF/GFF file",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output Parquet file",
    )
    parser.add_argument(
        "--compression", "-c",
        choices=["snappy", "gzip", "zstd", "none"],
        default="snappy",
        help="Parquet compression codec (default: snappy)",
    )

    args = parser.parse_args()

    try:
        convert_gtf_to_parquet(
            input_path=args.input,
            output_path=args.output,
            compression=args.compression,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
