"""Fine-tuning extensions for AlphaGenome PyTorch."""

from typing import TYPE_CHECKING

from alphagenome_pytorch.extensions.finetuning.adapters import (
    LoRA, Locon, IA3, IA3_FF, AdapterHoulsby
)
from alphagenome_pytorch.extensions.finetuning.training import (
    ModalityConfig, collate_genomic, MODALITY_CONFIGS,
    create_lr_scheduler, compute_finetuning_loss,
    train_epoch, validate,
    ProfilingStats, train_epoch_ddp, validate_ddp,
    train_epoch_multihead, validate_multihead,
)
from alphagenome_pytorch.extensions.finetuning.transfer import TransferConfig

# Distributed utilities
from alphagenome_pytorch.extensions.finetuning.distributed import (
    setup_distributed, cleanup_distributed, is_main_process,
    print_rank0, reduce_tensor, gather_tensors, barrier, broadcast_object,
)

# Logging utilities
from alphagenome_pytorch.extensions.finetuning.logging import TrainingLogger

# Checkpointing utilities
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    PreemptionHandler, setup_preemption_handler,
)

if TYPE_CHECKING:
    # Import for static type checkers only - actual imports are lazy-loaded
    from alphagenome_pytorch.extensions.finetuning.datasets import (
        ATACDataset,
        CachedGenome,
        GenomicDataset,
        MultimodalDataset,
        RNASeqDataset,
        collate_multimodal,
        compute_track_means,
    )


def __getattr__(name):
    """Lazy import for datasets to avoid pyfaidx/pyBigWig dependency at import time."""
    if name in ("ATACDataset", "RNASeqDataset", "GenomicDataset", "CachedGenome",
                "compute_track_means", "MultimodalDataset", "collate_multimodal"):
        from alphagenome_pytorch.extensions.finetuning import datasets
        return getattr(datasets, name)
    if name in (
        "apply_atac_transforms",
        "apply_rnaseq_transforms",
        "normalize_to_total",
        "mean_normalize",
        "power_transform",
        "smooth_clip",
    ):
        from alphagenome_pytorch.extensions.finetuning import data_transforms
        return getattr(data_transforms, name)
    if name in ("sequence_to_onehot", "onehot_to_sequence"):
        from alphagenome_pytorch.utils import sequence
        return getattr(sequence, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Utilities (lazy-loaded)
    "sequence_to_onehot",
    "onehot_to_sequence",
    # Datasets (lazy-loaded)
    "ATACDataset",
    "RNASeqDataset",
    "GenomicDataset",
    "MultimodalDataset",
    "collate_multimodal",
    "CachedGenome",
    "compute_track_means",
    # Adapters
    "LoRA",
    "Locon",
    "IA3",
    "IA3_FF",
    "AdapterHoulsby",
    # Training
    "ModalityConfig",
    "MODALITY_CONFIGS",
    "TransferConfig",
    "collate_genomic",
    "create_lr_scheduler",
    "compute_finetuning_loss",
    "train_epoch",
    "validate",
    "save_checkpoint",
    # Enhanced training with DDP
    "ProfilingStats",
    "train_epoch_ddp",
    "validate_ddp",
    # Multi-head training
    "train_epoch_multihead",
    "validate_multihead",
    # Distributed utilities
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "print_rank0",
    "reduce_tensor",
    "gather_tensors",
    "barrier",
    "broadcast_object",
    # Logging
    "TrainingLogger",
    # Checkpointing
    "find_latest_checkpoint",
    "load_checkpoint",
    "PreemptionHandler",
    "setup_preemption_handler",
    # Data transforms (lazy-loaded)
    "apply_atac_transforms",
    "apply_rnaseq_transforms",
    "normalize_to_total",
    "mean_normalize",
    "power_transform",
    "smooth_clip",
]
