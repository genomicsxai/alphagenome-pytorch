"""Fine-tuning modality registry: the assay types and their resolutions.

Deliberately dependency-light — dataclasses only, no torch — so the flag layer
(:mod:`alphagenome_pytorch.extensions.finetuning.args`) can read the registry to
build ``--modality``'s choices without importing the training code. ``training``
re-exports these names, so importing them from there still works.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModalityConfig:
    """Configuration for a fine-tuning modality.

    Attributes:
        name: Modality name ('rnaseq' or 'atac').
        resolutions: Tuple of output resolutions (e.g., (1, 128) or (128,)).
        default_resolution_weights: Default weights for each resolution.
        embedding_dim: Embedding dimension for ATAC (None for RNA-seq).
        positions_arg: CLI argument name for positions ('positions' or 'peaks').
    """

    name: str
    resolutions: tuple[int, ...]
    default_resolution_weights: dict[int, float]
    embedding_dim: int | None
    positions_arg: str


# Registry of modality configurations
MODALITY_CONFIGS: dict[str, ModalityConfig] = {
    "rna_seq": ModalityConfig(
        name="rna_seq",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "atac": ModalityConfig(
        name="atac",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "dnase": ModalityConfig(
        name="dnase",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "procap": ModalityConfig(
        name="procap",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "cage": ModalityConfig(
        name="cage",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "chip_tf": ModalityConfig(
        name="chip_tf",
        resolutions=(128,),
        default_resolution_weights={128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "chip_histone": ModalityConfig(
        name="chip_histone",
        resolutions=(128,),
        default_resolution_weights={128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
}
