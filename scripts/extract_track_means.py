"""Extract track_means from JAX model metadata and save for PyTorch.

Usage:
    python scripts/extract_track_means.py --output track_means.pt
"""

import argparse
import sys
import torch
import numpy as np

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_module


# Head configurations matching JAX heads.py
HEAD_CONFIGS = {
    'atac': {'num_tracks': 256},
    'dnase': {'num_tracks': 384},
    'procap': {'num_tracks': 128},
    'cage': {'num_tracks': 640},
    'rna_seq': {'num_tracks': 768},
    'chip_tf': {'num_tracks': 1664},
    'chip_histone': {'num_tracks': 1152},
    'contact_maps': {'num_tracks': 28},
}


def get_track_means_for_head(metadata, head_name, num_tracks):
    """Extract track means for a head, matching JAX logic."""
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
    }

    output_type = getattr(dna_output.OutputType, output_type_map[head_name])
    track_metadata = metadata.get(output_type)

    if track_metadata is None or 'nonzero_mean' not in track_metadata.columns:
        print(f"  {head_name}: No nonzero_mean column found, using ones")
        return np.ones(num_tracks, dtype=np.float32)

    means = track_metadata['nonzero_mean'].values.astype(np.float32)

    # Keep NaN values to match JAX behavior (NaN tracks produce NaN outputs)
    nan_count = np.isnan(means).sum()
    if nan_count > 0:
        print(f"  {head_name}: {nan_count} tracks have NaN means (will produce NaN outputs)")

    # Pad to num_tracks if needed
    if len(means) < num_tracks:
        means = np.pad(means, (0, num_tracks - len(means)), constant_values=1.0)
    return means[:num_tracks]


def extract_track_means(output_path):
    """Extract track means from JAX metadata for all heads."""
    print("Loading metadata for HOMO_SAPIENS...")
    metadata_human = metadata_module.load(dna_model.Organism.HOMO_SAPIENS)

    print("Loading metadata for MUS_MUSCULUS...")
    metadata_mouse = metadata_module.load(dna_model.Organism.MUS_MUSCULUS)

    track_means_dict = {}

    for head_name, config in HEAD_CONFIGS.items():
        num_tracks = config['num_tracks']

        # Get means for each organism
        means_human = get_track_means_for_head(metadata_human, head_name, num_tracks)
        means_mouse = get_track_means_for_head(metadata_mouse, head_name, num_tracks)

        # Stack: (num_organisms=2, num_tracks)
        track_means = np.stack([means_human, means_mouse], axis=0)
        track_means_dict[head_name] = torch.from_numpy(track_means)

        print(f"  {head_name}: shape={track_means.shape}, "
              f"human_mean={means_human.mean():.4f}, mouse_mean={means_mouse.mean():.4f}")

    # Save
    torch.save(track_means_dict, output_path)
    print(f"\nSaved track_means to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract track means from JAX metadata")
    parser.add_argument('--output', default='track_means.pt', help='Output path for track means')
    args = parser.parse_args()

    extract_track_means(args.output)
