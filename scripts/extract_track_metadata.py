"""Extract track metadata from JAX model and save as CSV for PyTorch.

Usage:
    python scripts/extract_track_metadata.py --output-file track_metadata.parquet
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

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
    'strand': 'track_strand',
    'ontology_curie': 'ontology_curie',
    'gtex_tissue': 'gtex_tissue',
    'Assay title': 'assay_title',
    'biosample_name': 'biosample_name',
    'biosample_type': 'biosample_type',
    'transcription_factor': 'transcription_factor',
    'histone_mark': 'histone_mark',
}

def get_metadata_for_head(metadata_df, head_name, num_tracks):
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
    head_metadata = metadata_df.get(output_type_enum)
    
    if head_metadata is None:
        print(f"  {head_name}: No metadata found, creating empty placeholders")
        df = pd.DataFrame({'track_name': [f'{head_name}_{i}' for i in range(num_tracks)]})
        df['track_strand'] = '.'
        return df

    # Select and rename columns
    available_cols = [c for c in COLUMN_MAPPING.keys() if c in head_metadata.columns]
    df = head_metadata[available_cols].rename(columns=COLUMN_MAPPING)
    
    # Ensure required columns exist
    if 'track_name' not in df.columns:
        df['track_name'] = [f'{head_name}_{i}' for i in range(len(df))]
    if 'track_strand' not in df.columns:
        df['track_strand'] = '.'
        
    # Handle length mismatch
    if len(df) < num_tracks:
        print(f"  {head_name}: Padding metadata from {len(df)} to {num_tracks}")
        # Pad with empty rows/default values
        padding = pd.DataFrame({
            'track_name': [f'{head_name}_{i}' for i in range(len(df), num_tracks)],
            'track_strand': ['.'] * (num_tracks - len(df))
        })
        df = pd.concat([df, padding], ignore_index=True)
    elif len(df) > num_tracks:
        print(f"  {head_name}: Truncating metadata from {len(df)} to {num_tracks}")
        df = df.iloc[:num_tracks]
        
    return df

def extract_metadata(output_path):
    """Extract metadata for all organisms and heads and save to single file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    organisms = {
        'human': dna_model.Organism.HOMO_SAPIENS,
        'mouse': dna_model.Organism.MUS_MUSCULUS
    }
    
    all_dfs = []

    for org_name, org_enum in organisms.items():
        print(f"\nProcessing {org_name.upper()}...")
        org_metadata = metadata_module.load(org_enum)
        
        for head_name, config in HEAD_CONFIGS.items():
            num_tracks = config['num_tracks']
            df = get_metadata_for_head(org_metadata, head_name, num_tracks)
            
            # Add identifying columns
            df['organism'] = org_name
            
            # Map head_name to PyTorch OutputType string value
            if head_name == 'contact_maps':
                df['output_type'] = 'pair_activations'
            else:
                df['output_type'] = head_name
            
            all_dfs.append(df)

    if not all_dfs:
        print("No metadata extracted.")
        return

    # Concatenate all
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to Parquet
    final_df.to_parquet(output_path, index=False)
    print(f"\nSaved combined metadata ({len(final_df)} rows) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract track metadata from JAX model")
    parser.add_argument('--output-file', default='track_metadata.parquet', help='Output parquet file')
    args = parser.parse_args()
    
    extract_metadata(args.output_file)
