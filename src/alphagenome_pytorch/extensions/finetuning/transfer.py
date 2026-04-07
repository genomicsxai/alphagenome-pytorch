"""
Transfer learning utilities for AlphaGenome.

Provides configuration and functions to prepare AlphaGenome for fine-tuning
on custom genomic tracks.

Example:
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.extensions.finetuning.transfer import (
        TransferConfig,
        prepare_for_transfer,
        load_trunk,
    )

    # Configure transfer
    config = TransferConfig(
        mode='lora',
        lora_rank=8,
        new_heads={'my_atac': {'modality': 'atac', 'num_tracks': 4}},
        remove_heads=['atac', 'dnase'],
    )
    
    # Load pretrained and prepare
    model = AlphaGenome()
    model = load_trunk(model, 'pretrained.pt')
    model = prepare_for_transfer(model, config)
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.heads import GenomeTracksHead
from alphagenome_pytorch.extensions.finetuning.adapters import (
    apply_lora,
    apply_locon,
    apply_ia3,
    apply_houlsby,
    apply_houlsby_baskerville,
    unfreeze_norm_layers,
)
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head


@dataclass
class TransferConfig:
    """Configuration for AlphaGenome transfer learning.
    
    Args:
        mode: Transfer mode(s). A single string or list of strings.
            Adapter modes can be combined (e.g. ``['lora', 'locon']``).
            Options:
            - 'full': Train all weights (lower LR recommended). Cannot combine.
            - 'linear': Freeze trunk, train only heads (linear probing)
            - 'encoder-only': Freeze trunk, train heads on raw CNN encoder
              output at 128bp. Cannot combine. Use with ``encoder_only=True``
              in head configs and ``model.forward(encoder_only=True)``.
            - 'lora': Apply LoRA adapters to attention
            - 'locon': Apply Locon adapters to conv layers
            - 'ia3': Apply IA3 scaling adapters
            - 'houlsby': Apply Houlsby bottleneck adapters
        lora_rank: LoRA rank (default: 8).
        lora_alpha: LoRA alpha scaling factor (default: 16).
        lora_targets: Module name substrings to apply LoRA to.
        locon_rank: Locon rank (default: 4).
        locon_alpha: Locon alpha scaling factor (default: 1).
        locon_targets: Module name substrings to apply Locon to.
        ia3_targets: Module name substrings for IA3 output-scaling.
        ia3_ff_targets: Module name substrings for IA3 input-scaling
            (feed-forward down-projections).
        houlsby_latent_dim: Houlsby bottleneck dimension (default: 8).
        houlsby_placement: Where to insert Houlsby adapters:
            - 'block': Baskerville-style, at transformer block boundaries
              (after MHA/MLP, before residual add). This is the default and
              matches the original Houlsby paper placement.
            - 'linear': Wrap individual Linear layers (like LoRA targeting).
        houlsby_targets: Which components to adapt:
            - For 'block' placement: ['mha', 'mlp'] (default), ['mha'], or ['mlp']
            - For 'linear' placement: Module name substrings like ['q_proj', 'v_proj']
        unfreeze_norm: Whether to unfreeze normalization layers (LayerNorm,
            RMSBatchNorm) when using adapters. Default False. Set to True to
            also train norm layers (adds ~227k params). Note: Baskerville only
            unfreezes norms for Houlsby adapters, not LoRA/IA3/Locon.
        new_heads: Dict mapping head name to config. Each config requires:
            - 'modality': Assay type ('atac' or 'rna_seq'). Required.
            - 'num_tracks': Number of output tracks. Required.
            - 'resolutions': Output resolutions (default: [1, 128]). Optional.
            - 'track_means': Track means tensor or path to .pt file. Optional.
            - 'num_organisms': Number of organisms (default: 1). Optional.
            - 'init_scheme': Weight init ('truncated_normal' or 'uniform'). Optional.
            Example: {'my_atac': {'modality': 'atac', 'num_tracks': 4}}
        remove_heads: List of original head names to remove.
        keep_heads: Alternative to remove_heads - specify which to keep.
            If both are empty, all original heads are kept.
        learning_rate: Suggested learning rate (for reference).
    """
    mode: str | list[str] = 'linear'
    
    # LoRA settings
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_targets: list[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    
    # Locon settings
    locon_rank: int = 4
    locon_alpha: int = 1
    locon_targets: list[str] = field(default_factory=lambda: ['conv_tower'])
    
    # IA3 settings
    ia3_targets: list[str] = field(default_factory=lambda: ['to_k', 'to_v'])
    ia3_ff_targets: list[str] = field(default_factory=list)
    
    # Houlsby settings
    houlsby_latent_dim: int = 8
    houlsby_placement: str = 'block'  # 'block' (Baskerville-style) or 'linear'
    houlsby_targets: list[str] = field(default_factory=lambda: ['mha', 'mlp'])
    unfreeze_norm: bool = False  # Manually enable to unfreeze normalization layers

    # Head configuration
    new_heads: dict[str, dict[str, Any]] = field(default_factory=dict)
    remove_heads: list[str] = field(default_factory=list)
    keep_heads: list[str] = field(default_factory=list)
    
    # Training (reference)
    learning_rate: float = 1e-4


def load_trunk(
    model: nn.Module,
    weights_path: str,
    exclude_heads: bool = True,
    strict: bool = False,
) -> nn.Module:
    """Load pretrained weights into model, optionally excluding heads.
    
    This function loads a pretrained checkpoint while allowing for
    mismatches in head configurations (useful for transfer learning).
    
    Args:
        model: AlphaGenome model instance.
        weights_path: Path to pretrained weights (.pt or .pth).
        exclude_heads: If True, skip loading head-related weights.
            This is useful when you want to train new heads.
        strict: If False (default), allows missing/unexpected keys.
        
    Returns:
        Model with loaded trunk weights.
        
    Example:
        >>> model = AlphaGenome()
        >>> model = load_trunk(model, 'alphagenome_pretrained.pt')
    """
    if Path(weights_path).suffix == '.safetensors':
        from safetensors.torch import load_file
        state_dict = load_file(weights_path, device="cpu")
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    
    # All head-related prefixes
    head_prefixes = (
        'heads.',
        'contact_maps_head.',
        'splice_sites_classification_head.',
        'splice_sites_usage_head.',
        'splice_sites_junction_head.',
    )

    if exclude_heads:
        # Filter out head parameters
        filtered = {}
        for k, v in state_dict.items():
            if k.startswith(head_prefixes):
                continue
            filtered[k] = v
        state_dict = filtered
    
    # Load with strict=False to handle mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    
    if exclude_heads:
        # Filter out expected missing keys (heads)
        missing = [k for k in missing if not k.startswith(head_prefixes)]
    
    if missing:
        print(f"Warning: Missing keys (not loaded): {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected keys (ignored): {unexpected[:5]}...")
    
    return model


def remove_all_heads(model: AlphaGenome) -> AlphaGenome:
    """Remove all heads from an AlphaGenome model."""
    model.heads = nn.ModuleDict()
    model.contact_maps_head = None
    model.splice_sites_classification_head = None
    model.splice_sites_usage_head = None
    model.splice_sites_junction_head = None
    return model


def add_head(
    model: AlphaGenome,
    name: str,
    head: GenomeTracksHead,
    replace: bool = False,
) -> None:
    """Register a new head on the model.

    Args:
        model: AlphaGenome model instance.
        name: Name for the head (e.g., 'my_atac').
        head: GenomeTracksHead instance to add.
        replace: If True, overwrite existing head with same name.
            If False (default), raises ValueError if head exists.

    Raises:
        ValueError: If head with same name exists and replace=False.

    Example:
        >>> head = create_finetuning_head('atac', n_tracks=4)
        >>> add_head(model, 'my_atac', head)
    """
    if name in model.heads and not replace:
        raise ValueError(
            f"Head '{name}' already exists. Use replace=True to overwrite."
        )
    model.heads[name] = head


def _freeze_trunk(
    model: nn.Module,
) -> nn.Module:
    """Freeze the trunk of the AlphaGenome model."""
    for name, param in model.named_parameters():
        if not name.startswith('heads.') and \
           not name.startswith('contact_maps_head.') and \
           not name.startswith('splice_sites_'):
            param.requires_grad = False
    return model


def prepare_for_transfer(
    model: nn.Module,
    config: TransferConfig,
) -> nn.Module:
    """Prepare AlphaGenome model for transfer learning.
    
    This function:
    1. Removes unwanted original heads
    2. Adds new heads for custom tracks
    3. Applies adapters based on transfer mode
    4. Freezes trunk if needed (linear probing)
    
    Args:
        model: AlphaGenome model (optionally with loaded trunk).
        config: Transfer configuration.
        
    Returns:
        Model ready for fine-tuning.
        
    Example:
        >>> config = TransferConfig(
        ...     mode='lora',
        ...     new_heads={'my_atac': {'num_tracks': 4}},
        ...     remove_heads=['atac', 'dnase'],
        ... )
        >>> model = prepare_for_transfer(model, config)
    """
    # 1. Handle head removal
    if config.keep_heads:
        # Remove all except keep_heads
        to_remove = [name for name in model.heads.keys() 
                     if name not in config.keep_heads]
        for name in to_remove:
            del model.heads[name]
    elif config.remove_heads:
        # Remove specified heads
        for name in config.remove_heads:
            if name in model.heads:
                del model.heads[name]
    
    # 2. Add new heads
    for head_name, head_config in config.new_heads.items():
        # Validate required fields
        if 'modality' not in head_config:
            raise ValueError(f"Head '{head_name}' missing required 'modality' field")
        if 'num_tracks' not in head_config:
            raise ValueError(f"Head '{head_name}' missing required 'num_tracks' field")

        # Handle track_means: can be None, tensor, or path
        track_means = head_config.get('track_means')
        if isinstance(track_means, (str, Path)):
            track_means = torch.load(track_means, weights_only=True)

        # Determine encoder_only flag: infer from mode, validate consistency
        is_encoder_only_mode = (
            config.mode == 'encoder-only'
            or (isinstance(config.mode, list) and 'encoder-only' in config.mode)
        )
        head_encoder_only = head_config.get('encoder_only', None)
        if is_encoder_only_mode:
            if head_encoder_only is not None and not head_encoder_only:
                raise ValueError(
                    f"Head '{head_name}' sets encoder_only=False, but "
                    f"mode is 'encoder-only'. All heads must use "
                    f"encoder_only=True in encoder-only mode."
                )
            head_encoder_only = True
        else:
            if head_encoder_only is None:
                head_encoder_only = False

        # Let create_finetuning_head pick the default resolution when
        # the caller didn't specify one (encoder_only defaults to 128bp).
        resolutions = head_config.get('resolutions')
        if resolutions is None and not head_encoder_only:
            resolutions = [1, 128]

        head = create_finetuning_head(
            assay_type=head_config['modality'],
            n_tracks=head_config['num_tracks'],
            resolutions=resolutions,
            num_organisms=head_config.get('num_organisms', 1),
            track_means=track_means,
            init_scheme=head_config.get('init_scheme', 'truncated_normal'),
            encoder_only=head_encoder_only,
        )
        add_head(model, head_name, head, replace=True)
    
    # 3. Apply adapter/freezing based on mode
    modes = config.mode if isinstance(config.mode, list) else [config.mode]
    
    # Validate modes
    valid_modes = {'full', 'linear', 'encoder-only', 'lora', 'locon', 'ia3', 'houlsby'}
    for m in modes:
        if m not in valid_modes:
            raise ValueError(
                f"Invalid mode '{m}'. Must be one of: {sorted(valid_modes)}"
            )
    
    if 'full' in modes and len(modes) > 1:
        raise ValueError(
            "'full' mode cannot be combined with other modes. "
            f"Got: {modes}"
        )

    if 'encoder-only' in modes and len(modes) > 1:
        raise ValueError(
            "'encoder-only' mode cannot be combined with other modes. "
            f"Got: {modes}"
        )

    if 'full' in modes:
        # Train everything - no freezing
        return model
    
    # Freeze trunk for linear probing and/or adapter modes
    model = _freeze_trunk(model)
    
    # Apply adapters in sequence
    if 'lora' in modes:
        model = apply_lora(
            model,
            config.lora_targets,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
        )
    
    if 'locon' in modes:
        model = apply_locon(
            model,
            config.locon_targets,
            rank=config.locon_rank,
            alpha=config.locon_alpha,
        )
    
    if 'ia3' in modes:
        model = apply_ia3(
            model,
            config.ia3_targets,
            ff_modules=config.ia3_ff_targets or None,
        )
    
    if 'houlsby' in modes:
        valid_placements = {'block', 'linear'}
        if config.houlsby_placement not in valid_placements:
            raise ValueError(
                f"Invalid houlsby_placement: '{config.houlsby_placement}'. "
                f"Must be one of: {valid_placements}"
            )
        if config.houlsby_placement == 'block':
            # Baskerville-style: adapters at transformer block boundaries
            model = apply_houlsby_baskerville(
                model,
                latent_dim=config.houlsby_latent_dim,
                target_blocks=config.houlsby_targets,
            )
        else:
            # Linear-style: wrap individual Linear layers
            model = apply_houlsby(
                model,
                config.houlsby_targets,
                latent_dim=config.houlsby_latent_dim,
            )

    # Optionally unfreeze normalization layers (Baskerville default)
    if config.unfreeze_norm and modes != ['linear']:
        # Only unfreeze norms when using adapters, not for pure linear probing
        adapter_modes = {'lora', 'locon', 'ia3', 'houlsby'}
        if any(m in adapter_modes for m in modes):
            unfreeze_norm_layers(model)

    return model


def count_trainable_params(model: nn.Module) -> dict[str, int]:
    """Count trainable parameters in model, grouped by component.

    Args:
        model: The model to analyze.

    Returns:
        Dict with 'total', 'heads', 'adapters', 'norm', 'other' counts.
        'norm' tracks unfrozen normalization layer parameters.
    """
    total = 0
    heads_count = 0
    adapters_count = 0
    norm_count = 0
    other_count = 0

    # Patterns for normalization layers
    norm_patterns = ['norm', 'layernorm', 'batchnorm', 'rmsnorm', 'rmsbatchnorm']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        total += n

        name_lower = name.lower()

        if name.startswith('heads.') or name.startswith('contact_maps_head.'):
            heads_count += n
        elif 'lora_' in name or 'locon_' in name or 'ia3' in name_lower or 'adapter.' in name or 'houlsby' in name_lower:
            adapters_count += n
        elif any(pattern in name_lower for pattern in norm_patterns):
            norm_count += n
        else:
            other_count += n

    return {
        'total': total,
        'heads': heads_count,
        'adapters': adapters_count,
        'norm': norm_count,
        'other': other_count,
    }


def transfer_config_to_dict(config: TransferConfig) -> dict[str, Any]:
    """Serialize TransferConfig to a JSON-compatible dict.

    Non-serializable values in ``new_heads`` (e.g. tensors, Path objects)
    are dropped so the result is safe for ``json.dumps`` and safetensors
    metadata.

    Args:
        config: TransferConfig instance.

    Returns:
        Dict representation suitable for JSON serialization and checkpoints.

    Example:
        >>> config = TransferConfig(mode='lora', lora_rank=16)
        >>> data = transfer_config_to_dict(config)
        >>> data['lora_rank']
        16
    """
    import torch

    data = asdict(config)

    # Sanitize new_heads: drop values that are not JSON-serializable
    # (e.g. track_means tensors, Path objects)
    for head_config in data.get("new_heads", {}).values():
        for key in list(head_config.keys()):
            val = head_config[key]
            if isinstance(val, (torch.Tensor, Path)):
                del head_config[key]

    return data


def transfer_config_from_dict(data: dict[str, Any]) -> TransferConfig:
    """Reconstruct TransferConfig from a dict.

    Filters to known fields for forward compatibility (ignores unknown keys
    from future versions).

    Args:
        data: Dict from checkpoint or config file.

    Returns:
        TransferConfig instance.

    Example:
        >>> data = {'mode': 'lora', 'lora_rank': 16, 'future_field': 'ignored'}
        >>> config = transfer_config_from_dict(data)
        >>> config.mode
        'lora'
    """
    known_fields = {f.name for f in fields(TransferConfig)}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return TransferConfig(**filtered)


__all__ = [
    'TransferConfig',
    'load_trunk',
    'add_head',
    'remove_all_heads',
    'prepare_for_transfer',
    'count_trainable_params',
    'transfer_config_to_dict',
    'transfer_config_from_dict',
]
