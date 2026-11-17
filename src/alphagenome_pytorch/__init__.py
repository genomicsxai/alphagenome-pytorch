"""AlphaGenome PyTorch implementation.

A PyTorch port of the JAX AlphaGenome model for genomic sequence analysis.

Example usage:
    import torch
    from alphagenome_pytorch import AlphaGenome

    # Load pretrained model:
    model = AlphaGenome.from_pretrained('model.pth', device='cuda')

    # Run inference
    sequence = np.random.randint(0, 4, size=(1, 131072))
    dna_seq = torch.tensor(np.eye(4)[sequence], dtype=torch.float32)
    organism_idx = 0  # 0=human, 1=mouse
    outputs = model.predict(dna_seq, organism_idx)
"""

from typing import TYPE_CHECKING

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    # Fallback for editable installs without build
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")

from .model import AlphaGenome

if TYPE_CHECKING:
    from .extensions.finetuning.transfer import (
        TransferConfig,
        load_trunk,
        prepare_for_transfer,
    )


def __getattr__(name):
    """Lazily expose fine-tuning helpers without loading optional dependencies."""
    if name in {'TransferConfig', 'load_trunk', 'prepare_for_transfer'}:
        from .extensions.finetuning import transfer

        return getattr(transfer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    '__version__',
    'AlphaGenome',
    'TransferConfig',
    'load_trunk',
    'prepare_for_transfer',
]
