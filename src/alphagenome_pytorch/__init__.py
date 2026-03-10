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

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    # Fallback for editable installs without build
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")

from .model import AlphaGenome

__all__ = [
    '__version__',
    'AlphaGenome',
]
