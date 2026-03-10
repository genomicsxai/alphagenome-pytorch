#!/usr/bin/env python
"""Convert GENCODE polyA metadata to Parquet with proper gene_id linking.

The GENCODE polyAs.gtf uses internal numeric IDs for gene_id. This script
uses the metadata.PolyA_feature file which contains transcript IDs, then
links them to gene_ids via the main annotation GTF.

Usage:
    python scripts/preprocess_polya.py \\
        --metadata /path/to/gencode.v46.metadata.PolyA_feature \\
        --gtf /path/to/gencode.v46.annotation.parquet \\
        --output /path/to/gencode.v46.polyAs.linked.parquet

The output parquet will have columns:
    - Chromosome, Start, End, Strand, Feature
    - transcript_id (from metadata)
    - gene_id (linked from GTF)
    - gene_name, gene_type (optional, from GTF)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def load_polya_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load GENCODE polyA metadata file.
    
    Format: transcript_id, tx_start, tx_end, chrom, genomic_start, genomic_end, strand, feature
    
    Args:
        metadata_path: Path to gencode.vX.metadata.PolyA_feature file
        
    Returns:
        DataFrame with polyA site information
    """
    print(f"Reading polyA metadata: {metadata_path}")
    
    # The metadata file is tab-separated with no header
    df = pd.read_csv(
        metadata_path,
        sep='\t',
        header=None,
        names=[
            'transcript_id',
            'tx_start',  # transcript-relative start
            'tx_end',    # transcript-relative end
            'Chromosome',
            'Start',     # genomic start (0-based in GENCODE metadata)
            'End',       # genomic end
            'Strand',
            'Feature',
        ],
    )
    
    print(f"  Loaded {len(df):,} polyA features")
    print(f"  Feature types: {df['Feature'].value_counts().to_dict()}")
    
    return df


def load_transcript_gene_mapping(gtf_path: str | Path) -> dict[str, dict]:
    """Load transcript->gene mapping from GTF/Parquet.
    
    Args:
        gtf_path: Path to annotation file (GTF or Parquet)
        
    Returns:
        Dict mapping transcript_id (no version) to gene info dict
    """
    print(f"Reading GTF for transcript mapping: {gtf_path}")
    
    path = Path(gtf_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        import pyranges
        df = pyranges.read_gtf(str(path)).df
    
    # Filter for transcripts to get transcript->gene mapping
    transcripts = df[df['Feature'] == 'transcript'].copy()
    print(f"  Found {len(transcripts):,} transcripts")
    
    # Build mapping: transcript_id -> {gene_id, gene_name, gene_type}
    mapping = {}
    for _, row in transcripts.iterrows():
        # Strip version from both transcript and gene IDs
        tx_id = row['transcript_id'].split('.')[0]
        gene_id = row['gene_id'].split('.')[0]
        
        mapping[tx_id] = {
            'gene_id': gene_id,
            'gene_id_full': row['gene_id'],  # Keep versioned for reference
            'gene_name': row.get('gene_name', ''),
            'gene_type': row.get('gene_type') or row.get('gene_biotype', ''),
        }
    
    print(f"  Built mapping for {len(mapping):,} unique transcripts")
    return mapping


def link_polya_to_genes(
    polya_df: pd.DataFrame,
    tx_gene_map: dict[str, dict],
) -> pd.DataFrame:
    """Link polyA features to genes via transcript IDs.
    
    Args:
        polya_df: DataFrame from load_polya_metadata
        tx_gene_map: Dict from load_transcript_gene_mapping
        
    Returns:
        DataFrame with gene_id column added
    """
    print("Linking polyA features to genes...")
    
    # Strip version from transcript_id for matching
    polya_df['transcript_id_base'] = polya_df['transcript_id'].str.split('.').str[0]
    
    # Map to gene info
    gene_ids = []
    gene_names = []
    gene_types = []
    unmatched = 0
    
    for tx_id in polya_df['transcript_id_base']:
        if tx_id in tx_gene_map:
            info = tx_gene_map[tx_id]
            gene_ids.append(info['gene_id'])
            gene_names.append(info['gene_name'])
            gene_types.append(info['gene_type'])
        else:
            gene_ids.append(None)
            gene_names.append(None)
            gene_types.append(None)
            unmatched += 1
    
    polya_df['gene_id'] = gene_ids
    polya_df['gene_name'] = gene_names
    polya_df['gene_type'] = gene_types
    
    # Clean up temp column
    polya_df.drop('transcript_id_base', axis=1, inplace=True)
    
    # Report matching stats
    matched = len(polya_df) - unmatched
    print(f"  Matched: {matched:,} / {len(polya_df):,} ({100*matched/len(polya_df):.1f}%)")
    if unmatched > 0:
        print(f"  Unmatched: {unmatched:,} (will be dropped)")
        polya_df = polya_df[polya_df['gene_id'].notna()].copy()
    
    return polya_df


def preprocess_polya(
    metadata_path: str | Path,
    gtf_path: str | Path,
    output_path: str | Path,
    compression: str = "snappy",
) -> None:
    """Create linked polyA parquet from GENCODE metadata.
    
    Args:
        metadata_path: Path to gencode.vX.metadata.PolyA_feature
        gtf_path: Path to annotation GTF or Parquet
        output_path: Path for output Parquet file
        compression: Parquet compression ('snappy', 'gzip', 'zstd', 'none')
    """
    start_time = time.perf_counter()
    
    # Load data
    polya_df = load_polya_metadata(metadata_path)
    tx_gene_map = load_transcript_gene_mapping(gtf_path)
    
    # Link polyA to genes
    linked_df = link_polya_to_genes(polya_df, tx_gene_map)
    
    # Add pas_strand column for JAX compatibility
    linked_df['pas_strand'] = linked_df['Strand']
    
    # Reorder columns for convenience
    cols = [
        'Chromosome', 'Start', 'End', 'Strand', 'Feature',
        'gene_id', 'gene_name', 'gene_type',
        'transcript_id', 'pas_strand',
    ]
    linked_df = linked_df[[c for c in cols if c in linked_df.columns]]
    
    # Optimize types
    linked_df['Chromosome'] = linked_df['Chromosome'].astype('category')
    linked_df['Strand'] = linked_df['Strand'].astype('category')
    linked_df['Feature'] = linked_df['Feature'].astype('category')
    linked_df['pas_strand'] = linked_df['pas_strand'].astype('category')
    
    # Write output
    print(f"\nWriting output: {output_path}")
    compression_arg = None if compression == 'none' else compression
    linked_df.to_parquet(output_path, compression=compression_arg, index=False)
    
    elapsed = time.perf_counter() - start_time
    output_size = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"\nDone in {elapsed:.1f}s!")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_size:.1f} MB")
    print(f"  Features: {len(linked_df):,}")
    print(f"  Unique genes: {linked_df['gene_id'].nunique():,}")
    
    # Sample output for verification
    print("\nSample rows:")
    print(linked_df.head(3).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Convert GENCODE polyA metadata to linked Parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using parquet annotation (faster)
    python scripts/preprocess_polya.py \\
        --metadata gencode.v46.metadata.PolyA_feature \\
        --gtf gencode.v46.annotation.parquet \\
        --output gencode.v46.polyAs.linked.parquet
        
    # Using GTF annotation (slower, needs pyranges)
    python scripts/preprocess_polya.py \\
        --metadata gencode.v46.metadata.PolyA_feature \\
        --gtf gencode.v46.annotation.gtf \\
        --output gencode.v46.polyAs.linked.parquet
        """,
    )
    parser.add_argument(
        "--metadata", "-m",
        required=True,
        help="Path to gencode.vX.metadata.PolyA_feature file",
    )
    parser.add_argument(
        "--gtf", "-g",
        required=True,
        help="Path to annotation GTF or Parquet file",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for output Parquet file",
    )
    parser.add_argument(
        "--compression", "-c",
        choices=["snappy", "gzip", "zstd", "none"],
        default="snappy",
        help="Parquet compression (default: snappy)",
    )
    
    args = parser.parse_args()
    
    try:
        preprocess_polya(
            metadata_path=args.metadata,
            gtf_path=args.gtf,
            output_path=args.output,
            compression=args.compression,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
