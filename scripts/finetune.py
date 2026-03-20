#!/usr/bin/env python
"""Unified AlphaGenome training script.

Supports: linear probing, LoRA, full finetuning, encoder-only.
Features: DDP, resume, preemption handling, W&B, profiling, multi-modality.

Usage:
    # Linear probing (frozen backbone, single modality)
    python scripts/finetune.py --mode linear-probe \\
        --genome hg38.fa \\
        --modality atac --bigwig *.bw \\
        --train-bed train.bed --val-bed val.bed \\
        --pretrained-weights model.pth \\
        --resolutions 1

    # LoRA finetuning (single modality)
    python scripts/finetune.py --mode lora \\
        --lora-rank 8 --lora-alpha 16 \\
        --genome hg38.fa \\
        --modality atac --bigwig *.bw \\
        --train-bed train.bed --val-bed val.bed \\
        --pretrained-weights model.pth \\
        --resolutions 1

    # Encoder-only (CNN encoder only, no transformer)
    python scripts/finetune.py --mode encoder-only \\
        --genome hg38.fa \\
        --modality atac --bigwig *.bw \\
        --train-bed train.bed --val-bed val.bed \\
        --pretrained-weights model.pth \\
        --sequence-length 500 --resolutions 128

    # Multi-modality training (multiple --modality --bigwig pairs)
    python scripts/finetune.py --mode lora \\
        --genome hg38.fa \\
        --modality atac --bigwig atac1.bw atac2.bw \\
        --modality rna_seq --bigwig rna1.bw rna2.bw \\
        --modality-weights atac:1.0,rna_seq:0.5 \\
        --train-bed train.bed --val-bed val.bed \\
        --pretrained-weights model.pth

    # Full finetuning (all parameters)
    python scripts/finetune.py --mode full \\
        --genome hg38.fa \\
        --modality atac --bigwig *.bw \\
        --train-bed train.bed --val-bed val.bed \\
        --pretrained-weights model.pth

    # Multi-GPU with DDP
    torchrun --nproc_per_node=4 scripts/finetune.py --mode lora ...

    # Resume from checkpoint
    python scripts/finetune.py ... --resume auto
    python scripts/finetune.py ... --resume path/to/checkpoint.pth

    # Graceful shutdown (saves checkpoint_preempt.pth)
    kill -USR1 <pid>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Workaround for torch.compile bug in quantization pattern matcher
import torch._inductor.config
torch._inductor.config.post_grad_fusion_options = {}

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# AlphaGenome imports
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning import (
    # Data
    CachedGenome,
    GenomicDataset,
    MultimodalDataset,
    compute_track_means,
    collate_genomic,
    collate_multimodal,
    # Model
    MODALITY_CONFIGS,
    TransferConfig,
    # Training
    create_lr_scheduler,
    train_epoch_ddp,
    validate_ddp,
    train_epoch_multihead,
    validate_multihead,
    # Distributed
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    print_rank0,
    barrier,
    broadcast_object,
    # Logging
    TrainingLogger,
    # Checkpointing
    find_latest_checkpoint,
    setup_preemption_handler,
)
from alphagenome_pytorch.extensions.finetuning.adapters import get_adapter_params
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    save_checkpoint, load_checkpoint, save_delta_checkpoint,
    load_delta_checkpoint, is_delta_checkpoint,
)
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.transfer import (
    load_trunk,
    remove_all_heads,
    add_head,
    prepare_for_transfer,
)


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULTS = {
    # Data
    "sequence_length": 131072,
    "resolutions": "1",
    # Model
    "modality": "atac",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_targets": "q_proj,v_proj",
    # Training
    "epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "lr": 1e-4,
    "weight_decay": 0.1,
    "warmup_steps": 500,
    "lr_schedule": "cosine",
    "positional_weight": 5.0,
    "count_weight": 1.0,
    "num_workers": 4,
    "max_grad_norm": 1.0,
    "num_segments": 8,
    "min_segment_size": None,
    # Logging
    "wandb_project": "alphagenome-finetune",
    "log_every": 50,
    "save_every": 1,
    # Output
    "output_dir": "finetuning_output",
}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified AlphaGenome training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file (CLI flags override config values)",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["linear-probe", "lora", "full", "encoder-only"],
        default="lora",
        help=(
            "Training mode: "
            "'linear-probe' (frozen backbone, train heads on full transformer embeddings), "
            "'lora' (LoRA adapters + heads), "
            "'full' (all parameters), "
            "'encoder-only' (frozen backbone, train heads on raw CNN encoder output at 128bp; "
            "supports short sequences such as MPRA; forces --resolutions 128)"
        ),
    )

    # Data arguments
    data = parser.add_argument_group("Data")
    data.add_argument("--genome", type=str, required=False, help="Reference genome FASTA")
    data.add_argument(
        "--bigwig",
        type=str,
        nargs="+",
        action="append",
        dest="bigwigs",
        help="BigWig signal file(s). Repeat --bigwig for each modality when using multi-modality.",
    )
    data.add_argument("--train-bed", type=str, required=False, help="Training positions BED")
    data.add_argument("--val-bed", type=str, required=False, help="Validation positions BED")
    data.add_argument("--sequence-length", type=int, default=DEFAULTS["sequence_length"])
    data.add_argument(
        "--resolutions",
        type=str,
        default=DEFAULTS["resolutions"],
        help="Comma-separated output resolutions (e.g., '1' or '1,128')",
    )
    data.add_argument(
        "--cache-genome",
        action="store_true",
        help="Cache genome in memory (~12GB for hg38)",
    )
    data.add_argument(
        "--cache-signals",
        action="store_true",
        help="Cache BigWig signals in memory (parallel init)",
    )
    data.add_argument(
        "--max-io-workers",
        type=int,
        default=16,
        help="Max threads for parallel BigWig I/O (default: 16)",
    )

    # Model arguments
    model = parser.add_argument_group("Model")
    model.add_argument("--pretrained-weights", type=str, required=False, help="Pretrained weights .pth")
    model.add_argument(
        "--modality",
        type=str,
        action="append",
        dest="modalities",
        choices=list(MODALITY_CONFIGS.keys()),
        help="Assay modality type. Repeat --modality for each --bigwig group in multi-modality mode.",
    )
    model.add_argument(
        "--modality-weights",
        type=str,
        default=None,
        help="Optional per-modality loss weights, e.g. 'atac:1.0,rna_seq:0.5'",
    )
    model.add_argument("--lora-rank", type=int, default=DEFAULTS["lora_rank"], help="LoRA rank (0 to disable)")
    model.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"], help="LoRA alpha scaling")
    model.add_argument(
        "--lora-targets",
        type=str,
        default=DEFAULTS["lora_targets"],
        help="Comma-separated modules for LoRA",
    )
    model.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Model dtype",
    )
    model.add_argument(
        "--head-init-scheme",
        type=str,
        default="truncated_normal",
        choices=["truncated_normal", "uniform"],
        help="Head weight initialization",
    )
    model.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")

    # Training arguments
    train = parser.add_argument_group("Training")
    train.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    train.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    train.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULTS["gradient_accumulation_steps"],
        help="Accumulate gradients over N batches",
    )
    train.add_argument("--lr", type=float, default=DEFAULTS["lr"], help="Learning rate")
    train.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    train.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    train.add_argument(
        "--lr-schedule",
        type=str,
        default=DEFAULTS["lr_schedule"],
        choices=["cosine", "constant"],
    )
    train.add_argument("--positional-weight", type=float, default=DEFAULTS["positional_weight"])
    train.add_argument("--count-weight", type=float, default=DEFAULTS["count_weight"])
    train.add_argument("--max-grad-norm", type=float, default=DEFAULTS["max_grad_norm"])
    train.add_argument("--num-segments", type=int, default=DEFAULTS["num_segments"])
    train.add_argument("--min-segment-size", type=int, default=DEFAULTS["min_segment_size"])
    train.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    train.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    train.add_argument("--track-means-samples", type=int, default=None, help="Samples for track means (default: all)")
    train.add_argument("--profile-batches", type=int, default=0, help="Profile first N batches")
    train.add_argument("--compile", action="store_true", help="Use torch.compile")
    train.add_argument("--seed", type=int, default=None, help="Random seed")

    # Logging arguments
    log = parser.add_argument_group("Logging")
    log.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    log.add_argument("--wandb-project", type=str, default=DEFAULTS["wandb_project"])
    log.add_argument("--wandb-entity", type=str, default=None)
    log.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])

    # Output arguments
    out = parser.add_argument_group("Output")
    out.add_argument("--output-dir", type=str, default=DEFAULTS["output_dir"])
    out.add_argument("--run-name", type=str, default=None)
    out.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])

    # Resume arguments
    resume = parser.add_argument_group("Resume / Checkpointing")
    resume.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path or 'auto' to find latest",
    )
    resume.add_argument(
        "--save-delta",
        action="store_true",
        help="Save delta checkpoints (adapter + head weights only, much smaller). "
             "Delta files saved as best_model.delta.pth",
    )

    args = parser.parse_args()
    cli_flags = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }

    def _load_yaml_config(path: str) -> dict[str, Any]:
        try:
            import yaml
        except ImportError:
            parser.error("YAML config support requires PyYAML (`pip install pyyaml`).")
        config_path = Path(path)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")
        with config_path.open() as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            parser.error("YAML config root must be a mapping/dictionary")
        return data

    def _parse_resolutions_value(value: Any, context: str) -> tuple[int, ...]:
        if isinstance(value, int):
            parsed = (int(value),)
        elif isinstance(value, str):
            parsed = tuple(int(r.strip()) for r in value.split(",") if r.strip())
        elif isinstance(value, (list, tuple)):
            parsed = tuple(int(r) for r in value)
        else:
            parser.error(f"Invalid resolutions for {context}: {value!r}")
        if not parsed:
            parser.error(f"Empty resolutions for {context}")
        return parsed

    def _apply_config_scalar(attr: str, config: dict[str, Any], key: str | None = None) -> None:
        flag = f"--{attr.replace('_', '-')}"
        if flag in cli_flags:
            return
        config_key = key or attr
        if config_key in config and config[config_key] is not None:
            setattr(args, attr, config[config_key])

    def _parse_weight_overrides(raw: Any) -> dict[str, float]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {str(k): float(v) for k, v in raw.items()}
        if isinstance(raw, str):
            out: dict[str, float] = {}
            for item in raw.split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" not in item:
                    parser.error("Weights must be specified as modality:weight pairs")
                mod, weight = item.split(":", 1)
                out[mod.strip()] = float(weight.strip())
            return out
        parser.error("Task weights in config must be a dict or comma-separated string")

    config_data = _load_yaml_config(args.config) if args.config else {}
    args.config_data = config_data
    args.config_path = args.config

    # Scalar config values (CLI wins)
    for attr in (
        "mode",
        "genome",
        "train_bed",
        "val_bed",
        "sequence_length",
        "resolutions",
        "cache_genome",
        "cache_signals",
        "max_io_workers",
        "pretrained_weights",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
        "dtype",
        "head_init_scheme",
        "gradient_checkpointing",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "lr",
        "weight_decay",
        "warmup_steps",
        "lr_schedule",
        "positional_weight",
        "count_weight",
        "max_grad_norm",
        "num_segments",
        "min_segment_size",
        "num_workers",
        "track_means_samples",
        "profile_batches",
        "compile",
        "seed",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "log_every",
        "output_dir",
        "run_name",
        "save_every",
        "resume",
        "modality_weights",
    ):
        _apply_config_scalar(attr, config_data)

    # Boolean aliases / migration-friendly keys
    if "--no-amp" not in cli_flags:
        if "use_amp" in config_data:
            args.no_amp = not bool(config_data["use_amp"])
        elif "no_amp" in config_data:
            args.no_amp = bool(config_data["no_amp"])

    if "--cache-genome" not in cli_flags and "--cache-signals" not in cli_flags:
        if bool(config_data.get("no_cache", False)):
            args.cache_genome = False
            args.cache_signals = False

    args.global_resolutions = _parse_resolutions_value(args.resolutions, "global resolutions")
    args.resolutions = ",".join(str(r) for r in args.global_resolutions)

    raw_modalities = config_data.get("modalities", {}) or {}
    if not isinstance(raw_modalities, dict):
        parser.error("Config key 'modalities' must be a mapping of modality -> settings")

    modality_specs: dict[str, dict[str, Any]] = {}
    for modality, mod_cfg in raw_modalities.items():
        if modality not in MODALITY_CONFIGS:
            parser.error(f"Unknown modality in config: {modality}")
        if not isinstance(mod_cfg, dict):
            parser.error(f"modalities.{modality} must be a mapping")
        spec: dict[str, Any] = {}
        if "bigwig" in mod_cfg and mod_cfg["bigwig"] is not None:
            bigwigs = mod_cfg["bigwig"]
            if isinstance(bigwigs, str):
                bigwigs = [bigwigs]
            if not isinstance(bigwigs, list):
                parser.error(f"modalities.{modality}.bigwig must be a string or list")
            spec["bigwig"] = [str(p) for p in bigwigs]
        if "resolutions" in mod_cfg and mod_cfg["resolutions"] is not None:
            spec["resolutions"] = _parse_resolutions_value(
                mod_cfg["resolutions"], f"modalities.{modality}.resolutions"
            )
        if "task_weight" in mod_cfg and mod_cfg["task_weight"] is not None:
            spec["task_weight"] = float(mod_cfg["task_weight"])
        modality_specs[modality] = spec

    cli_modality_to_bigwigs: dict[str, list[str]] = {}
    if args.bigwigs is not None:
        if args.modalities is None:
            args.modalities = [DEFAULTS["modality"]]
        if len(args.modalities) != len(args.bigwigs):
            parser.error(
                f"Number of --modality ({len(args.modalities)}) must match number of --bigwig groups ({len(args.bigwigs)}). "
                "Each --modality should be followed by --bigwig FILES."
            )
        for modality, bigwigs in zip(args.modalities, args.bigwigs):
            if modality in cli_modality_to_bigwigs:
                parser.error(f"Duplicate modality: {modality}")
            cli_modality_to_bigwigs[modality] = bigwigs
    elif args.modalities is not None and "--modality" in cli_flags:
        parser.error("--modality requires matching --bigwig entries")

    for modality, bigwigs in cli_modality_to_bigwigs.items():
        merged = dict(modality_specs.get(modality, {}))
        merged["bigwig"] = list(bigwigs)
        modality_specs[modality] = merged

    args.modalities = list(modality_specs.keys())
    if not args.modalities:
        parser.error("--bigwig is required (or provide modalities in --config)")

    # Required scalar args after config merge
    for flag, value in (
        ("--genome", args.genome),
        ("--train-bed", args.train_bed),
        ("--val-bed", args.val_bed),
        ("--pretrained-weights", args.pretrained_weights),
    ):
        if not value:
            parser.error(f"{flag} is required (or provide it in --config)")

    args.modality_to_bigwigs = {}
    args.modality_resolutions = {}
    args.modality_weight_dict = {}
    for modality in args.modalities:
        spec = modality_specs.get(modality, {})
        if "bigwig" not in spec or not spec["bigwig"]:
            parser.error(f"No bigwig files specified for modality '{modality}'")
        args.modality_to_bigwigs[modality] = list(spec["bigwig"])
        args.modality_resolutions[modality] = spec.get("resolutions", args.global_resolutions)
        args.modality_weight_dict[modality] = float(spec.get("task_weight", 1.0))

    if "--modality-weights" not in cli_flags:
        for mod, weight in _parse_weight_overrides(
            config_data.get("modality_weights", config_data.get("task_weights"))
        ).items():
            if mod not in args.modality_to_bigwigs:
                parser.error(f"Unknown modality in config task weights: {mod}")
            args.modality_weight_dict[mod] = weight

    if args.modality_weights:
        for item in args.modality_weights.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                parser.error("Each --modality-weights item must be 'modality:weight'")
            mod, weight = item.split(":", 1)
            mod = mod.strip()
            if mod not in args.modality_to_bigwigs:
                parser.error(f"Unknown modality in --modality-weights: {mod}")
            args.modality_weight_dict[mod] = float(weight.strip())

    args.is_multimodal = len(args.modalities) > 1

    return args


# =============================================================================
# Data Loading
# =============================================================================


def create_datasets(
    args: argparse.Namespace,
    rank: int,
) -> tuple:
    """Create training and validation datasets.

    Returns:
        For single-modality:
            (train_dataset, val_dataset, {"modality": track_names}, {"modality": resolutions})
        For multi-modality:
            (train_dataset, val_dataset, {"mod1": names1, "mod2": names2, ...}, modality_resolutions)
    """
    cache_genome = args.cache_genome
    cache_signals = args.cache_signals
    max_io_workers = args.max_io_workers

    print_rank0(f"Global resolutions: {args.global_resolutions}", rank)
    print_rank0(f"Modalities: {list(args.modality_to_bigwigs.keys())}", rank)
    print_rank0(f"Caching: genome={cache_genome}, signals={cache_signals}", rank)
    print_rank0(f"Parallel I/O workers: {max_io_workers}", rank)

    # Shared genome cache for train + val
    genome = CachedGenome(args.genome) if cache_genome else args.genome

    # Build per-modality track names
    modality_track_names: dict[str, list[str]] = {}
    for modality, bigwigs in args.modality_to_bigwigs.items():
        modality_track_names[modality] = [Path(bw).stem for bw in bigwigs]
        print_rank0(
            f"  {modality}: {len(bigwigs)} tracks, resolutions={args.modality_resolutions[modality]} - "
            f"{modality_track_names[modality]}",
            rank,
        )

    if args.is_multimodal:
        # Multi-modality: create per-modality datasets and wrap in MultimodalDataset
        print_rank0("Creating multi-modality train datasets...", rank)
        train_datasets = {}
        val_datasets = {}

        for modality, bigwigs in args.modality_to_bigwigs.items():
            resolutions = args.modality_resolutions[modality]
            train_datasets[modality] = GenomicDataset(
                genome_fasta=genome,
                bigwig_files=bigwigs,
                bed_file=args.train_bed,
                resolutions=resolutions,
                sequence_length=args.sequence_length,
                cache_genome=cache_genome,
                cache_signals=cache_signals,
                max_io_workers=max_io_workers,
            )
            val_datasets[modality] = GenomicDataset(
                genome_fasta=genome,
                bigwig_files=bigwigs,
                bed_file=args.val_bed,
                resolutions=resolutions,
                sequence_length=args.sequence_length,
                cache_genome=cache_genome,
                cache_signals=cache_signals,
                max_io_workers=max_io_workers,
            )

        train_dataset = MultimodalDataset(train_datasets)
        val_dataset = MultimodalDataset(val_datasets)
    else:
        # Single-modality: create simple GenomicDataset
        modality = args.modalities[0]
        bigwigs = args.modality_to_bigwigs[modality]
        resolutions = args.modality_resolutions[modality]

        print_rank0("Creating train dataset...", rank)
        train_dataset = GenomicDataset(
            genome_fasta=genome,
            bigwig_files=bigwigs,
            bed_file=args.train_bed,
            resolutions=resolutions,
            sequence_length=args.sequence_length,
            cache_genome=cache_genome,
            cache_signals=cache_signals,
            max_io_workers=max_io_workers,
        )

        print_rank0("Creating validation dataset...", rank)
        val_dataset = GenomicDataset(
            genome_fasta=genome,
            bigwig_files=bigwigs,
            bed_file=args.val_bed,
            resolutions=resolutions,
            sequence_length=args.sequence_length,
            cache_genome=cache_genome,
            cache_signals=cache_signals,
            max_io_workers=max_io_workers,
        )

    print_rank0(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}", rank)

    return train_dataset, val_dataset, modality_track_names, args.modality_resolutions


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int,
    is_multimodal: bool = False,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
    """Create data loaders with optional distributed samplers."""
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    collate_fn = collate_multimodal if is_multimodal else collate_genomic

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, train_sampler, val_sampler


# =============================================================================
# Model Setup
# =============================================================================


def create_model(
    args: argparse.Namespace,
    modality_track_names: dict[str, list[str]],
    modality_track_means: dict[str, torch.Tensor | None],
    modality_resolutions: dict[str, tuple[int, ...]],
    device: torch.device,
    rank: int,
    world_size: int,
    local_rank: int,
) -> tuple[nn.Module, dict[str, nn.Module], list[torch.nn.Parameter]]:
    """Create and configure the model based on training mode.

    Args:
        args: Command line arguments.
        modality_track_names: Dict mapping modality to list of track names.
        modality_track_means: Dict mapping modality to track means tensor (or None).
        modality_resolutions: Per-modality output resolutions.
        device: Torch device.
        rank: Process rank.
        world_size: Number of processes.
        local_rank: Local rank for GPU assignment.

    Returns:
        Tuple of (model, heads_dict, trainable_params).
    """
    print_rank0(f"Loading pretrained model from {args.pretrained_weights}", rank)

    # Dtype policy
    dtype_policy = (
        DtypePolicy.full_float32() if args.dtype == "float32" else DtypePolicy.mixed_precision()
    )
    print_rank0(f"Dtype policy: {dtype_policy}", rank)

    model = AlphaGenome(
        gradient_checkpointing=args.gradient_checkpointing,
        dtype_policy=dtype_policy,
    )
    model = load_trunk(model, args.pretrained_weights, exclude_heads=True)

    # Freeze base model first (for non-full modes)
    # This way, newly created heads will have requires_grad=True by default
    if args.mode != "full":
        for param in model.parameters():
            param.requires_grad = False

    # Remove original heads
    model = remove_all_heads(model)

    # encoder-only mode forces 128bp resolution for all heads
    is_encoder_only = args.mode == "encoder-only"

    # Create heads for each modality (after freeze, so they have requires_grad=True)
    heads: dict[str, nn.Module] = {}
    for modality, track_names in modality_track_names.items():
        n_tracks = len(track_names)
        track_means = modality_track_means.get(modality)
        resolutions = modality_resolutions[modality]

        head = create_finetuning_head(
            assay_type=modality,
            n_tracks=n_tracks,
            resolutions=resolutions if not is_encoder_only else (128,),
            num_organisms=1,
            track_means=track_means,
            init_scheme=args.head_init_scheme,
            encoder_only=is_encoder_only,
        )
        add_head(model, modality, head)
        heads[modality] = head
        head_resolutions = (128,) if is_encoder_only else resolutions
        print_rank0(f"Created {modality} head with {n_tracks} tracks at resolutions {head_resolutions}", rank)

    # Build new_heads dict for TransferConfig (used for delta checkpoints)
    new_heads_config: dict[str, dict] = {}
    for modality in heads:
        head_res = (128,) if is_encoder_only else modality_resolutions[modality]
        new_heads_config[modality] = {
            "modality": modality,
            "num_tracks": len(modality_track_names[modality]),
            "resolutions": list(head_res),
            "encoder_only": is_encoder_only,
        }

    # Configure trainable params based on mode
    trainable_params: list[torch.nn.Parameter] = []
    transfer_config: TransferConfig | None = None  # For delta checkpoints

    if args.mode == "linear-probe":
        # Heads already have requires_grad=True (created after freeze)
        for head in heads.values():
            trainable_params.extend(list(head.parameters()))
        transfer_config = TransferConfig(mode="linear", new_heads=new_heads_config)
        print_rank0("Mode: linear-probe (frozen backbone)", rank)

    elif args.mode == "encoder-only":
        # Frozen backbone; head receives raw encoder output (B, S//128, 1536) at 128bp.
        # Useful for short sequences (MPRA, ~100-500 bp) that cannot pass through the
        # transformer, or when global attention context is not needed.
        for head in heads.values():
            trainable_params.extend(list(head.parameters()))
        transfer_config = TransferConfig(mode="encoder-only", new_heads=new_heads_config)
        print_rank0("Mode: encoder-only (frozen backbone, raw CNN encoder output to head)", rank)

    elif args.mode == "lora":
        if args.lora_rank > 0:
            lora_targets = [t.strip() for t in args.lora_targets.split(",")]
            print_rank0(f"Applying LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}", rank)
            print_rank0(f"  Target modules: {lora_targets}", rank)

            transfer_config = TransferConfig(
                mode="lora",
                lora_targets=lora_targets,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                new_heads=new_heads_config,
            )
            model = prepare_for_transfer(model, transfer_config)
            # LoRA adapters + heads (heads already have requires_grad=True)
            trainable_params = get_adapter_params(model)
            for head in heads.values():
                trainable_params.extend(list(head.parameters()))
        else:
            # LoRA rank 0 means just train heads
            for head in heads.values():
                trainable_params.extend(list(head.parameters()))
            transfer_config = TransferConfig(mode="linear", new_heads=new_heads_config)
            print_rank0("Mode: lora (rank=0, heads only)", rank)

    elif args.mode == "full":
        # All parameters trainable (model was not frozen above)
        trainable_params = list(model.parameters())
        # Delta checkpoints don't make sense for full mode (all weights change)
        if args.save_delta:
            print_rank0("Warning: --save-delta ignored for --mode full (all weights trained)", rank)
        print_rank0("Mode: full (all parameters trainable)", rank)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Move to device
    model = model.to(device)

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print_rank0("Model wrapped with DistributedDataParallel", rank)

    # Get head references from GPU model
    model_module = model.module if isinstance(model, DDP) else model
    heads = {modality: model_module.heads[modality] for modality in heads}

    # Optionally compile
    if args.compile:
        print_rank0("Compiling model with torch.compile...", rank)
        import torch._inductor.config as inductor_config
        inductor_config.group_fusion = False
        model = torch.compile(model)

    # Count parameters
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model_module.parameters())
    print_rank0(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)", rank)

    return model, heads, trainable_params


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)
        print_rank0(f"Random seed: {args.seed} (+ rank offset)", rank)

    if world_size > 1:
        print_rank0(f"Distributed training with {world_size} GPUs", rank)
    print_rank0(f"Device: {device}", rank)

    # Output directory
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")
    barrier()

    # Resolve resume checkpoint
    resume_path = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(output_dir)
        if is_main_process(rank):
            if resume_path:
                print(f"Auto-resume: found {resume_path}")
            else:
                print("Auto-resume: no checkpoint found, starting fresh")
    elif args.resume:
        resume_path = Path(args.resume)

    # Create datasets
    train_dataset, val_dataset, modality_track_names, modality_resolutions = create_datasets(args, rank)

    # Build resolution weights per modality.
    # encoder-only mode always operates at 128bp (encoder output resolution).
    resolution_weights_per_modality: dict[str, dict[int, float]] = {}
    for modality in args.modalities:
        if args.mode == "encoder-only":
            resolution_weights_per_modality[modality] = {128: 1.0}
        else:
            resolution_weights_per_modality[modality] = {
                res: 1.0 for res in modality_resolutions[modality]
            }

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        args.num_workers,
        world_size,
        rank,
        is_multimodal=args.is_multimodal,
    )
    print_rank0(f"Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}", rank)

    # Compute track means for each modality (rank 0 computes, then broadcast)
    modality_track_means: dict[str, torch.Tensor | None] = {}
    if is_main_process(rank):
        print("Computing track means...")
        for modality, bigwigs in args.modality_to_bigwigs.items():
            modality_track_means[modality] = compute_track_means(
                bigwigs,
                args.train_bed,
                sequence_length=args.sequence_length,
                max_samples=args.track_means_samples,
            )
            print(f"  {modality}: mean={modality_track_means[modality].mean():.4f}")
    modality_track_means = broadcast_object(modality_track_means, src=0)

    # Create model
    model, heads, trainable_params = create_model(
        args,
        modality_track_names,
        modality_track_means,
        modality_resolutions,
        device,
        rank,
        world_size,
        local_rank,
    )
    model_module = model.module if isinstance(model, DDP) else model

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    total_steps = (args.epochs * len(train_loader)) // args.gradient_accumulation_steps
    scheduler = create_lr_scheduler(optimizer, args.warmup_steps, total_steps, schedule=args.lr_schedule)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    print_rank0(f"Gradient accumulation: {args.gradient_accumulation_steps}", rank)
    print_rank0(f"Effective batch size: {effective_batch_size}", rank)
    print_rank0(f"Total optimizer steps: {total_steps:,}", rank)
    print_rank0(f"LR schedule: {args.lr_schedule} (warmup: {args.warmup_steps} steps)", rank)

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    wandb_run_id = None

    if resume_path and resume_path.exists():
        print_rank0(f"Resuming from: {resume_path}", rank)

        # Check if it's a delta checkpoint
        if is_delta_checkpoint(resume_path):
            # Delta checkpoint - load adapter + head weights only
            # skip_prepare=True because create_model already set up adapters/heads
            _, metadata = load_delta_checkpoint(
                resume_path,
                model=model_module,
                optimizer=optimizer,
                scheduler=scheduler,
                skip_prepare=True,
            )
            start_epoch = metadata.get("epoch", 0) + 1
            best_val_loss = metadata.get("best_val_loss", metadata.get("val_loss", float("inf")))
            wandb_run_id = metadata.get("wandb_run_id")
            print_rank0(f"  Resumed from delta checkpoint at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}", rank)
        else:
            # Full checkpoint
            ckpt = load_checkpoint(
                resume_path,
                model=model_module,
                optimizer=optimizer,
                scheduler=scheduler,
                device="cpu",
            )
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            wandb_run_id = ckpt.get("wandb_run_id")
            print_rank0(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}", rank)

    # Config for logging
    config = {
        "mode": args.mode,
        "genome": args.genome,
        "modalities": args.modalities,
        "modality_to_bigwigs": {k: list(v) for k, v in args.modality_to_bigwigs.items()},
        "modality_weights": args.modality_weight_dict,
        "train_bed": args.train_bed,
        "val_bed": args.val_bed,
        "sequence_length": args.sequence_length,
        "resolutions": list(args.global_resolutions),
        "modality_resolutions": {m: list(r) for m, r in modality_resolutions.items()},
        "track_names": modality_track_names,
        "pretrained_weights": args.pretrained_weights,
        "lora_rank": args.lora_rank if args.mode == "lora" else None,
        "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
        "lora_targets": args.lora_targets if args.mode == "lora" else None,
        "head_init_scheme": args.head_init_scheme,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_schedule": args.lr_schedule,
        "positional_weight": args.positional_weight,
        "count_weight": args.count_weight,
        "max_grad_norm": args.max_grad_norm,
        "num_segments": args.num_segments,
        "min_segment_size": args.min_segment_size,
        "total_steps": total_steps,
        "n_trainable_params": sum(p.numel() for p in trainable_params),
        "n_total_params": sum(p.numel() for p in model_module.parameters()),
        "use_amp": not args.no_amp,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dtype": args.dtype,
        "world_size": world_size,
        "seed": args.seed,
        "resumed_from": str(resume_path) if resume_path else None,
    }

    # Logger (rank 0 only)
    logger = TrainingLogger(
        output_dir=output_dir,
        rank=rank,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=run_name,
        config=config,
        resume_id=wandb_run_id if resume_path else None,
    )

    use_amp = not args.no_amp

    # Preemption handler state
    current_epoch = start_epoch

    def _save_preempt():
        """Save preemption checkpoint."""
        if is_main_process(rank):
            save_checkpoint(
                path=output_dir / "checkpoint_preempt.pth",
                epoch=max(0, current_epoch - 1),  # Last completed epoch
                model=model_module,
                optimizer=optimizer,
                val_loss=best_val_loss,
                track_names=modality_track_names,
                modality=args.modalities,
                resolutions=modality_resolutions,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                wandb_run_id=logger.wandb_run_id,
            )
            print(f"Preemption checkpoint saved to {output_dir / 'checkpoint_preempt.pth'}")

    handler = setup_preemption_handler(_save_preempt, rank, world_size)

    # Training loop
    print_rank0("\n" + "=" * 60, rank)
    print_rank0(f"Starting training (epoch {start_epoch} to {args.epochs})", rank)
    print_rank0("=" * 60, rank)

    # Freeze backbone (use torch.no_grad) when no backbone params need gradients.
    # - linear-probe: only heads train
    # - encoder-only: only heads train (backbone always frozen; uses encoder_only forward)
    # - lora with rank=0: only heads train (no LoRA adapters)
    # - lora with rank>0: LoRA adapters need gradients, can't freeze
    # - full: all params need gradients
    frozen_backbone = args.mode in ("linear-probe", "encoder-only") or (
        args.mode == "lora" and args.lora_rank == 0
    )
    encoder_only = args.mode == "encoder-only"

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if handler.preempted:
                print_rank0("Preemption flag set - saving and exiting.", rank)
                handler.save_and_exit()
                break

            current_epoch = epoch

            # Clear GPU cache between epochs for robustness
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Training
            if args.is_multimodal:
                train_loss, per_modality_train_loss = train_epoch_multihead(
                    model=model,
                    heads=heads,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    modality_weights=args.modality_weight_dict,
                    resolution_weights=resolution_weights_per_modality,
                    positional_weight=args.positional_weight,
                    count_weight=args.count_weight,
                    epoch=epoch,
                    log_every=args.log_every,
                    use_amp=use_amp,
                    accumulation_steps=args.gradient_accumulation_steps,
                    frozen_backbone=frozen_backbone,
                    train_sampler=train_sampler,
                    rank=rank,
                    world_size=world_size,
                    max_grad_norm=args.max_grad_norm,
                    num_segments=args.num_segments,
                    min_segment_size=args.min_segment_size,
                    profile_batches=args.profile_batches if epoch == start_epoch else 0,
                    log_fn=logger.log_step if is_main_process(rank) else None,
                    encoder_only=encoder_only,
                )
            else:
                # Single modality: use the standard train_epoch_ddp
                primary_modality = args.modalities[0]
                train_loss = train_epoch_ddp(
                    model=model,
                    head=heads[primary_modality],
                    train_loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    resolution_weights=resolution_weights_per_modality[primary_modality],
                    positional_weight=args.positional_weight,
                    count_weight=args.count_weight,
                    epoch=epoch,
                    log_every=args.log_every,
                    use_amp=use_amp,
                    accumulation_steps=args.gradient_accumulation_steps,
                    frozen_backbone=frozen_backbone,
                    train_sampler=train_sampler,
                    rank=rank,
                    world_size=world_size,
                    max_grad_norm=args.max_grad_norm,
                    num_segments=args.num_segments,
                    min_segment_size=args.min_segment_size,
                    profile_batches=args.profile_batches if epoch == start_epoch else 0,
                    log_fn=logger.log_step if is_main_process(rank) else None,
                    encoder_only=encoder_only,
                )

            if handler.preempted:
                print_rank0("Preemption flag set - saving and exiting.", rank)
                handler.save_and_exit()
                break

            # Validation
            if args.is_multimodal:
                val_loss, val_metrics = validate_multihead(
                    model=model,
                    heads=heads,
                    val_loader=val_loader,
                    device=device,
                    modality_weights=args.modality_weight_dict,
                    resolution_weights=resolution_weights_per_modality,
                    positional_weight=args.positional_weight,
                    count_weight=args.count_weight,
                    use_amp=use_amp,
                    num_segments=args.num_segments,
                    min_segment_size=args.min_segment_size,
                    compute_pearson=True,
                    rank=rank,
                    world_size=world_size,
                    encoder_only=encoder_only,
                )
            else:
                primary_modality = args.modalities[0]
                val_loss, val_metrics = validate_ddp(
                    model=model,
                    head=heads[primary_modality],
                    val_loader=val_loader,
                    device=device,
                    resolution_weights=resolution_weights_per_modality[primary_modality],
                    positional_weight=args.positional_weight,
                    count_weight=args.count_weight,
                    use_amp=use_amp,
                    num_segments=args.num_segments,
                    min_segment_size=args.min_segment_size,
                    compute_pearson=True,
                    rank=rank,
                    world_size=world_size,
                    encoder_only=encoder_only,
                )

            # Synchronize CUDA to ensure all validation ops complete before next epoch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            current_lr = scheduler.get_last_lr()[0]
            is_best = val_loss < best_val_loss

            # Print epoch summary
            if is_main_process(rank):
                summary = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                if args.is_multimodal:
                    for mod, mod_loss in per_modality_train_loss.items():
                        summary += f", {mod}_train={mod_loss:.4f}"
                for key, val in val_metrics.items():
                    if key.endswith("_values") or key.endswith("_std"):
                        continue
                    if "pearson" in key or "_loss" in key:
                        summary += f", {key}={val:.4f}"
                print(summary)

            # Log epoch
            extra = {}
            histograms = {}
            for key, val in val_metrics.items():
                if key.endswith("_values"):
                    histograms[key] = val
                elif "pearson" in key:
                    extra[key] = val
                else:
                    extra[f"val_loss_{key}"] = val

            logger.log_epoch(epoch, train_loss, val_loss, current_lr, is_best, extra, histograms)

            # Save checkpoints
            if is_main_process(rank):
                if is_best:
                    best_val_loss = val_loss
                    save_checkpoint(
                        path=output_dir / "best_model.pth",
                        epoch=epoch,
                        model=model_module,
                        optimizer=optimizer,
                        val_loss=val_loss,
                        track_names=modality_track_names,
                        modality=args.modalities,
                        resolutions=modality_resolutions,
                        scheduler=scheduler,
                        best_val_loss=best_val_loss,
                        wandb_run_id=logger.wandb_run_id,
                    )
                    print(f"  Saved best model (val_loss={val_loss:.4f})")

                    # Save delta checkpoint (adapter + head weights only)
                    if args.save_delta and transfer_config is not None:
                        save_delta_checkpoint(
                            path=output_dir / "best_model.delta.pth",
                            model=model_module,
                            config=transfer_config,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            val_loss=val_loss,
                            best_val_loss=best_val_loss,
                            track_names=modality_track_names,
                            modality=args.modalities,
                            resolutions=modality_resolutions,
                        )
                        print(f"  Saved delta checkpoint (adapter + head weights)")

                if epoch % args.save_every == 0:
                    save_checkpoint(
                        path=output_dir / f"checkpoint_epoch{epoch}.pth",
                        epoch=epoch,
                        model=model_module,
                        optimizer=optimizer,
                        val_loss=val_loss,
                        track_names=modality_track_names,
                        modality=args.modalities,
                        resolutions=modality_resolutions,
                        scheduler=scheduler,
                        best_val_loss=best_val_loss,
                        wandb_run_id=logger.wandb_run_id,
                    )

            barrier()

    except KeyboardInterrupt:
        print_rank0("\nTraining interrupted by user", rank)
    finally:
        logger.finish()
        handler.unregister()
        cleanup_distributed()

    print_rank0(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}", rank)
    print_rank0(f"Output: {output_dir}", rank)


if __name__ == "__main__":
    main()
