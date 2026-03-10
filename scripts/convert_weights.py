"""Convert JAX AlphaGenome checkpoint to PyTorch format.

Usage:
    python scripts/convert_weights.py /path/to/jax/checkpoint --output model.pth

This script:
1. Converts JAX model weights to PyTorch format
2. Extracts track_means from JAX metadata and bundles them in the output file
3. Converts weights to native NCL (channels-first) format for Conv1d operations
"""

import argparse
import sys
import torch
import numpy as np
import jax

from alphagenome_pytorch.model import AlphaGenome
from alphagenome_pytorch.jax_compat.weight_mapping import map_pytorch_to_jax
from alphagenome_pytorch.jax_compat.transforms import apply_transform

# Head configurations matching JAX heads.py
HEAD_CONFIGS = {
    'atac': {'num_tracks': 256},
    'dnase': {'num_tracks': 384},
    'procap': {'num_tracks': 128},
    'cage': {'num_tracks': 640},
    'rna_seq': {'num_tracks': 768},
    'chip_tf': {'num_tracks': 1664},
    'chip_histone': {'num_tracks': 1152},
}


def get_track_means_for_head(metadata, head_name, num_tracks):
    """Extract track means for a head, matching JAX logic."""
    from alphagenome.models import dna_output

    # Map head names to output types
    output_type_map = {
        'atac': 'ATAC',
        'dnase': 'DNASE',
        'procap': 'PROCAP',
        'cage': 'CAGE',
        'rna_seq': 'RNA_SEQ',
        'chip_tf': 'CHIP_TF',
        'chip_histone': 'CHIP_HISTONE',
    }

    output_type = getattr(dna_output.OutputType, output_type_map[head_name])
    track_metadata = metadata.get(output_type)

    if track_metadata is None or 'nonzero_mean' not in track_metadata.columns:
        print(f"  {head_name}: No nonzero_mean column found, using ones")
        return np.ones(num_tracks, dtype=np.float32)

    means = track_metadata['nonzero_mean'].values.astype(np.float32)

    # Keep NaN values - they will be sanitized when loading into the model
    nan_count = np.isnan(means).sum()
    if nan_count > 0:
        print(f"  {head_name}: {nan_count} tracks have NaN means (will be sanitized on load)")

    # Pad to num_tracks if needed
    if len(means) < num_tracks:
        means = np.pad(means, (0, num_tracks - len(means)), constant_values=1.0)
    return means[:num_tracks]


def extract_track_means():
    """Extract track means from JAX metadata for all heads."""
    from alphagenome_research.model import dna_model
    from alphagenome_research.model.metadata import metadata as metadata_module

    print("\nExtracting track_means from JAX metadata...")
    print("  Loading metadata for HOMO_SAPIENS...")
    metadata_human = metadata_module.load(dna_model.Organism.HOMO_SAPIENS)

    print("  Loading metadata for MUS_MUSCULUS...")
    metadata_mouse = metadata_module.load(dna_model.Organism.MUS_MUSCULUS)

    track_means_dict = {}

    for head_name, config in HEAD_CONFIGS.items():
        num_tracks = config['num_tracks']

        # Get means for each organism
        means_human = get_track_means_for_head(metadata_human, head_name, num_tracks)
        means_mouse = get_track_means_for_head(metadata_mouse, head_name, num_tracks)

        # Stack: (num_organisms=2, num_tracks)
        track_means = np.stack([means_human, means_mouse], axis=0)

        # Sanitize NaN values to 0 (matching GenomeTracksHead behavior)
        track_means = np.nan_to_num(track_means, nan=0.0)

        track_means_dict[head_name] = torch.from_numpy(track_means).contiguous()

        print(f"  {head_name}: shape={track_means.shape}, "
              f"human_mean={means_human[~np.isnan(means_human)].mean():.4f}, "
              f"mouse_mean={means_mouse[~np.isnan(means_mouse)].mean():.4f}")

    return track_means_dict


def convert(jax_checkpoint_path):
    """Convert JAX checkpoint to PyTorch state dict."""
    print(f"Loading JAX checkpoint from {jax_checkpoint_path}...")

    import orbax.checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    params, state = checkpointer.restore(jax_checkpoint_path)
    print("JAX params loaded.")

    print("Initializing PyTorch model...")
    pt_model = AlphaGenome(num_organisms=2)
    pt_state_dict = pt_model.state_dict()

    new_state_dict = {}

    def transform_param(name, param, pt_shape):
        """Transform JAX param to PyTorch format using unified transforms."""
        param = np.array(param)

        # Handle bfloat16
        is_bf16 = False
        if param.dtype == 'bfloat16' or str(param.dtype) == 'bfloat16':
            is_bf16 = True
            param = param.astype(np.float32)

        # Use unified transform from transforms.py
        transformed = apply_transform(name, param, pt_shape)
        tensor = torch.tensor(transformed).contiguous()

        if is_bf16:
            tensor = tensor.to(torch.bfloat16)

        return tensor

    # Flatten JAX params to a single dict
    flat_jax = {}

    def flatten_haiku(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten_haiku(v, f"{prefix}/{k}" if prefix else k)
            else:
                flat_jax[f"{prefix}/{k}" if prefix else k] = v

    flatten_haiku(params)
    if state:
        flatten_haiku(state)

    print("WARNING: Automatic mapping is heuristic. Verify manually.")

    # Iterate over PyTorch state dict and find corresponding JAX keys
    # Skip track_means buffers - they come from metadata, not JAX checkpoint
    count_mapped = 0
    count_track_means = 0

    for pt_name, pt_param in pt_state_dict.items():
        # Skip track_means - will be handled separately from JAX metadata
        if pt_name.endswith('.track_means'):
            count_track_means += 1
            continue

        pt_shape = tuple(pt_param.shape)
        jax_key = map_pytorch_to_jax(pt_name)

        if jax_key and jax_key in flat_jax:
            print(f"Mapping {pt_name} <- {jax_key}")
            new_state_dict[pt_name] = transform_param(pt_name, flat_jax[jax_key], pt_shape)
            count_mapped += 1
        elif jax_key:
            print(f"MISSING JAX KEY: {jax_key} for {pt_name}")
        else:
            print(f"Skipping {pt_name} (no map rule)")

    print(f"Mapped {count_mapped} / {len(pt_state_dict) - count_track_means} parameters.")

    # Extract track_means from JAX metadata and add to state dict
    track_means_dict = extract_track_means()
    for head_name, means in track_means_dict.items():
        key = f"heads.{head_name}.track_means"
        new_state_dict[key] = means
        print(f"Added {key} (from JAX metadata)")

    print(f"Total parameters in output: {len(new_state_dict)}")

    return new_state_dict


def convert_weights_to_ncl(state_dict: dict) -> dict:
    """Convert NLC-format weights to NCL format.

    Key conversions:
    - heads.*.linears (MultiOrganismLinear → MultiOrganismConv1d):
        weight: (num_org, in, out) → (num_org, out, in), rename to .convs
    - splice_sites_*_head.linear (MultiOrganismLinear → MultiOrganismConv1d):
        weight: (num_org, in, out) → (num_org, out, in), rename to .conv
    - OutputEmbedder.project_in: Linear → Conv1d
        weight: (out, in) → (out, in, 1) for Conv1d
    - ConvBlock.proj: Linear → Conv1d
        weight: (out, in) → (out, in, 1) for Conv1d

    NOT converted (stays as MultiOrganismLinear):
    - contact_maps_head.linear - operates on pair activations (NLC format)

    Note: RMSBatchNorm parameters stay as (C,) - no conversion needed.
    Note: Regular Conv1d weights are NOT transposed - only multi-organism weights in heads.

    Args:
        state_dict: Original state dict with NLC-format weights.

    Returns:
        New state dict with NCL-format weights.
    """
    print("\nConverting weights to NCL format...")
    new_state = {}

    for key, value in state_dict.items():
        new_key = key
        new_value = value

        # === GenomeTracksHead: MultiOrganismLinear → MultiOrganismConv1d ===
        # heads.*.linears.* need transpose and rename to .convs.*
        if key.startswith('heads.') and '.linears.' in key:
            if '.weight' in key and value.dim() == 3:
                # Transpose: (num_org, in, out) → (num_org, out, in)
                new_value = value.transpose(1, 2).contiguous()
                print(f"Transposed {key}: {value.shape} → {new_value.shape}")
            # Rename linears → convs
            new_key = key.replace('.linears.', '.convs.')
            print(f"Renamed {key} → {new_key}")

        # === SpliceSites heads: MultiOrganismLinear → MultiOrganismConv1d ===
        # splice_sites_*_head.linear.* need transpose and rename to .conv.*
        elif 'splice_sites' in key and '.linear.' in key:
            if '.weight' in key and value.dim() == 3:
                # Transpose: (num_org, in, out) → (num_org, out, in)
                new_value = value.transpose(1, 2).contiguous()
                print(f"Transposed {key}: {value.shape} → {new_value.shape}")
            # Rename linear → conv
            new_key = key.replace('.linear.', '.conv.')
            print(f"Renamed {key} → {new_key}")

        # === contact_maps_head.linear: NO CHANGE ===
        # Stays as MultiOrganismLinear, operates on NLC pair activations
        elif key.startswith('contact_maps_head.linear.'):
            # No transpose, no rename - keep as-is
            print(f"Keeping unchanged (MultiOrganismLinear): {key}")

        # === OutputEmbedder.project_in: Linear → Conv1d ===
        # weight: (out, in) → (out, in, 1)
        elif 'embedder_' in key and 'project_in.weight' in key:
            if value.dim() == 2:
                new_value = value.unsqueeze(2)
                print(f"Expanded {key}: {value.shape} → {new_value.shape}")

        # === OutputEmbedder.project_skip: Linear → Conv1d ===
        elif 'project_skip.weight' in key:
            if value.dim() == 2:
                new_value = value.unsqueeze(2)
                print(f"Expanded {key}: {value.shape} → {new_value.shape}")

        # === ConvBlock.proj → ConvBlock.conv: Linear → Conv1d (kernel_size=1) ===
        # Matches block.*.proj and pointwise.proj patterns
        # But NOT attn_bias.proj which stays as nn.Linear
        elif '.proj.' in key and 'attn_bias' not in key:
            if '.weight' in key and value.dim() == 2:
                new_value = value.unsqueeze(2)
                print(f"Expanded {key}: {value.shape} → {new_value.shape}")
            # Rename proj → conv
            new_key = new_key.replace('.proj.', '.conv.')
            print(f"Renamed {key} → {new_key}")

        new_state[new_key] = new_value

    return new_state


def save_weights(state_dict, output_path, use_safetensors=False):
    """Save state dict to file."""
    if use_safetensors:
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError(
                "safetensors is required for --safetensors flag. "
                "Install with: pip install safetensors"
            )
        save_file(state_dict, output_path)
        print(f"Saved model to {output_path} (safetensors format)")
    else:
        torch.save(state_dict, output_path)
        print(f"Saved model to {output_path} (torch format)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help='Path to JAX checkpoint directory')
    parser.add_argument('--output', default=None, help='Output file path')
    parser.add_argument('--safetensors', action='store_true', help='Save as safetensors format')
    args = parser.parse_args()

    # Default output path based on format
    output_path = args.output
    if output_path is None:
        output_path = 'alphagenome_pt.safetensors' if args.safetensors else 'alphagenome_pt.pth'

    state_dict = convert(args.checkpoint_path)
    # Note: transforms.py now handles NCL format conversion directly during weight mapping
    save_weights(state_dict, output_path, use_safetensors=args.safetensors)
