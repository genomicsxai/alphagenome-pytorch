"""Fine-tuning heads for AlphaGenome.

Provides a factory function to create GenomeTracksHead instances
configured for fine-tuning on specific assay types.
"""
from __future__ import annotations

import torch
from typing import Literal

from alphagenome_pytorch.heads import GenomeTracksHead


# All supported assay types and their squashing behavior
# Only RNA-seq uses squashing (power law expansion)
ASSAY_TYPES = {
    'rna_seq': {'apply_squashing': True, 'default_resolutions': (1, 128)},
    'atac': {'apply_squashing': False, 'default_resolutions': (1, 128)},
    'dnase': {'apply_squashing': False, 'default_resolutions': (1, 128)},
    'procap': {'apply_squashing': False, 'default_resolutions': (1, 128)},
    'cage': {'apply_squashing': False, 'default_resolutions': (1, 128)},
    'chip_tf': {'apply_squashing': False, 'default_resolutions': (128,)},
    'chip_histone': {'apply_squashing': False, 'default_resolutions': (128,)},
}


def create_finetuning_head(
    assay_type: Literal['rna_seq', 'atac', 'dnase', 'procap', 'cage', 'chip_tf', 'chip_histone'],
    n_tracks: int,
    resolutions: list[int] | tuple[int, ...] | None = None,
    num_organisms: int = 1,
    track_means: torch.Tensor | None = None,
    init_scheme: Literal['truncated_normal', 'uniform'] = 'truncated_normal',
    encoder_only: bool = False,
) -> GenomeTracksHead:
    """Create a GenomeTracksHead configured for fine-tuning.

    Args:
        assay_type: Type of assay. Controls whether squashing is applied.
            'rna_seq' applies power law expansion.
            All other types do not apply squashing.
        n_tracks: Number of output tracks (e.g., number of cell types).
        resolutions: Output resolutions. Valid values are 1 and/or 128.
            If None, uses default resolutions for the assay type:
            - (1, 128) for atac, dnase, procap, cage, rna_seq
            - (128,) for chip_tf, chip_histone
        num_organisms: Number of organisms. Default: 1 for fine-tuning.
        track_means: Optional track means tensor for scaling.
            Shape: (num_organisms, n_tracks). Defaults to ones.
        init_scheme: Weight initialization scheme for head parameters.
            'truncated_normal' (default): Match JAX - truncated normal for
                weights (std=1/sqrt(fan_in)), zeros for biases.
            'uniform': Legacy PyTorch-style uniform initialization for both
                weights and biases.
        encoder_only: If True, create a head that accepts raw CNN encoder output
            (B, S//128, 1536) instead of full transformer embeddings. Automatically
            restricts resolutions to (128,). Use with ``model.forward(encoder_only=True)``
            for short-sequence fine-tuning (e.g. MPRA assays).

    Returns:
        Configured GenomeTracksHead instance.

    Example:
        >>> head = create_finetuning_head('atac', n_tracks=10)
        >>> head = create_finetuning_head('rna_seq', n_tracks=5, resolutions=(1, 128))
        >>> head = create_finetuning_head('chip_tf', n_tracks=100, resolutions=(128,))
        >>> # Encoder-only head for short sequences
        >>> head = create_finetuning_head('atac', n_tracks=10, encoder_only=True)

    Raises:
        ValueError: If an invalid assay type or resolution is provided.
    """
    if assay_type not in ASSAY_TYPES:
        valid_types = ', '.join(sorted(ASSAY_TYPES.keys()))
        raise ValueError(f"Invalid assay type '{assay_type}'. Must be one of: {valid_types}")

    assay_config = ASSAY_TYPES[assay_type]

    if encoder_only:
        # Encoder output is at 128bp resolution only; the decoder is not run.
        if resolutions is None:
            resolutions = (128,)
        for res in resolutions:
            if res != 128:
                raise ValueError(
                    f"encoder_only heads only support resolution 128 "
                    f"(got {res}). The CNN encoder produces features at 128bp; "
                    f"the decoder is not run in encoder-only mode."
                )
        return GenomeTracksHead(
            in_channels=1536,  # raw encoder output dim (ENCODER_EMBEDDING_DIM)
            num_tracks=n_tracks,
            resolutions=list(resolutions),
            num_organisms=num_organisms,
            apply_squashing=assay_config['apply_squashing'],
            track_means=track_means,
            init_scheme=init_scheme,
        )

    # Use default resolutions for assay type if not specified
    if resolutions is None:
        resolutions = assay_config['default_resolutions']

    valid_resolutions = {1, 128}
    for res in resolutions:
        if res not in valid_resolutions:
            raise ValueError(f"Invalid resolution {res}. Must be one of {valid_resolutions}")

    apply_squashing = assay_config['apply_squashing']

    return GenomeTracksHead(
        in_channels=None,
        num_tracks=n_tracks,
        resolutions=list(resolutions),
        num_organisms=num_organisms,
        apply_squashing=apply_squashing,
        track_means=track_means,
        init_scheme=init_scheme,
    )


# Embedding dimension of the raw CNN encoder output (before transformer/decoder).
ENCODER_EMBEDDING_DIM = 1536

__all__ = ['ASSAY_TYPES', 'ENCODER_EMBEDDING_DIM', 'create_finetuning_head']
