"""Extract track metadata from JAX model and save as parquet for PyTorch.

This script extracts track metadata from the alphagenome and alphagenome_research
packages and saves them as parquet files for use with alphagenome-pytorch.

Usage:
    # Extract to default location (src/alphagenome_pytorch/data/)
    python scripts/extract_track_metadata.py

    # Extract to custom location
    python scripts/extract_track_metadata.py --output-dir /path/to/output

    # Extract single organism
    python scripts/extract_track_metadata.py --organisms human

Requirements:
    - alphagenome (pip install alphagenome)
    - alphagenome_research (from source)
    - pandas
    - pyarrow (for parquet support)
"""

import argparse
from pathlib import Path
import pandas as pd

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_module


# Head configurations matching JAX heads.py and extract_track_means.py
HEAD_CONFIGS = {
    'atac': {'num_tracks': 256},
    'dnase': {'num_tracks': 384},
    'procap': {'num_tracks': 128},
    'cage': {'num_tracks': 640},
    'rna_seq': {'num_tracks': 768},
    'chip_tf': {'num_tracks': 1664},
    'chip_histone': {'num_tracks': 1152},
    'contact_maps': {'num_tracks': 28},
    'splice_sites_classification': {'num_tracks': 5},
    'splice_sites_usage': {'num_tracks': 734},
    'splice_sites_junction': {'num_tracks': 367},
}

# Map JAX/Original metadata columns to PyTorch TrackMetadata fields
COLUMN_MAPPING = {
    'name': 'track_name',
    'strand': 'strand',
    'ontology_curie': 'ontology_curie',
    'gtex_tissue': 'gtex_tissue',
    'Assay title': 'assay_title',
    'biosample_name': 'biosample_name',
    'biosample_type': 'biosample_type',
    'biosample_life_stage': 'biosample_life_stage',
    'transcription_factor': 'transcription_factor',
    'histone_mark': 'histone_mark',
    'data_source': 'data_source',
    'endedness': 'endedness',
    'genetically_modified': 'genetically_modified',
    'nonzero_mean': 'nonzero_mean',
}

def get_metadata_for_head(metadata_obj, head_name, num_tracks):
    """Extract and format metadata for a specific head."""
    # Map head names to output types
    output_type_map = {
        'atac': 'ATAC',
        'dnase': 'DNASE',
        'procap': 'PROCAP',
        'cage': 'CAGE',
        'rna_seq': 'RNA_SEQ',
        'chip_tf': 'CHIP_TF',
        'chip_histone': 'CHIP_HISTONE',
        'contact_maps': 'CONTACT_MAPS',
        'splice_sites_classification': 'SPLICE_SITES',
        'splice_sites_usage': 'SPLICE_SITE_USAGE',
        'splice_sites_junction': 'SPLICE_JUNCTIONS',
    }

    output_type_enum = getattr(dna_output.OutputType, output_type_map[head_name])
    head_metadata = metadata_obj.get(output_type_enum)

    if head_metadata is None:
        print(f"  {head_name}: No metadata found, creating placeholders")
        df = pd.DataFrame({
            'track_index': list(range(num_tracks)),
            'track_name': [f'{head_name}_{i}' for i in range(num_tracks)],
            'strand': ['.'] * num_tracks,
        })
        return df

    # Select and rename columns
    available_cols = [c for c in COLUMN_MAPPING.keys() if c in head_metadata.columns]
    df = head_metadata[available_cols].rename(columns=COLUMN_MAPPING).copy()

    # Ensure required columns exist
    if 'track_name' not in df.columns:
        df['track_name'] = [f'{head_name}_{i}' for i in range(len(df))]
    if 'strand' not in df.columns:
        df['strand'] = '.'

    # Handle length mismatch
    if len(df) < num_tracks:
        print(f"  {head_name}: Padding metadata from {len(df)} to {num_tracks}")
        padding = pd.DataFrame({
            'track_name': [f'{head_name}_{i}' for i in range(len(df), num_tracks)],
            'strand': ['.'] * (num_tracks - len(df))
        })
        df = pd.concat([df, padding], ignore_index=True)
    elif len(df) > num_tracks:
        print(f"  {head_name}: Truncating metadata from {len(df)} to {num_tracks}")
        df = df.iloc[:num_tracks]

    # Add track_index column
    df['track_index'] = list(range(len(df)))

    return df

def extract_metadata_for_organism(org_name: str, output_dir: Path) -> pd.DataFrame:
    """Extract metadata for one organism and save to parquet."""
    organisms = {
        'human': dna_model.Organism.HOMO_SAPIENS,
        'mouse': dna_model.Organism.MUS_MUSCULUS
    }
    org_enum = organisms[org_name]
    org_index = 0 if org_name == 'human' else 1

    print(f"\nProcessing {org_name.upper()}...")
    org_metadata = metadata_module.load(org_enum)

    all_dfs = []
    for head_name, config in HEAD_CONFIGS.items():
        num_tracks = config['num_tracks']
        df = get_metadata_for_head(org_metadata, head_name, num_tracks)

        # Add identifying columns
        df['organism'] = org_index

        # Map head_name to PyTorch output_type string value
        if head_name == 'contact_maps':
            df['output_type'] = 'pair_activations'
        else:
            df['output_type'] = head_name

        print(f"  {head_name}: {len(df)} tracks")
        all_dfs.append(df)

    if not all_dfs:
        print(f"No metadata extracted for {org_name}.")
        return pd.DataFrame()

    # Concatenate all heads
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns with standard columns first
    standard_columns = [
        'track_index',
        'track_name',
        'output_type',
        'organism',
        'strand',
        'ontology_curie',
        'biosample_name',
        'biosample_type',
        'biosample_life_stage',
        'assay_title',
        'data_source',
        'gtex_tissue',
        'histone_mark',
        'transcription_factor',
        'endedness',
        'genetically_modified',
        'nonzero_mean',
    ]
    columns = [c for c in standard_columns if c in final_df.columns]
    extra_columns = [c for c in final_df.columns if c not in standard_columns]
    columns.extend(sorted(extra_columns))
    final_df = final_df[columns]

    # Save to parquet
    output_path = output_dir / f"track_metadata_{org_name}.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"Saved {len(final_df)} tracks to {output_path}")

    return final_df


def main():
    default_output_dir = Path(__file__).parent.parent / "src" / "alphagenome_pytorch" / "data"

    parser = argparse.ArgumentParser(
        description="Extract track metadata from AlphaGenome JAX packages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=default_output_dir,
        help=f'Output directory for parquet files (default: {default_output_dir})',
    )
    parser.add_argument(
        '--organisms',
        nargs='+',
        choices=['human', 'mouse'],
        default=['human', 'mouse'],
        help='Organisms to extract (default: human mouse)',
    )
    args = parser.parse_args()

    for organism in args.organisms:
        extract_metadata_for_organism(organism, args.output_dir)

    print("\nDone! Metadata files can be loaded with:")
    print("  from alphagenome_pytorch.named_outputs import TrackMetadataCatalog")
    print("  catalog = TrackMetadataCatalog.load_builtin('human')")


if __name__ == "__main__":
    main()
