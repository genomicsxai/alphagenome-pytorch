"""Unified AlphaGenome training script.

Supports: linear probing, LoRA, Locon, LoRA+Locon, full finetuning, encoder-only.
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

    # LoRA + Locon finetuning (Baskerville-style Locon parity)
    python scripts/finetune.py --mode lora+locon \\
        --lora-rank 8 --lora-alpha 16 \\
        --locon-rank 4 --locon-alpha 1 \\
        --locon-targets down_blocks.4,down_blocks.5 \\
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
from collections.abc import Mapping, Sequence
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
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
from alphagenome_pytorch.sequence_parallel import SequenceParallelism
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
    train_epoch_multihead,
    validate_multihead,
    train_epoch_sequence_parallel,
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
    validate_locon_targets,
    transfer_config_to_dict,
)

from alphagenome_pytorch.extensions.finetuning.args import parse_args

# =============================================================================
# Utilities
# =============================================================================


def unwrap_training_model(model: nn.Module) -> nn.Module:
    """Unwrap the exact wrapper stack used in this training script.

    Wrapping order in finetune.py is deterministic:
    1. base model
    2. optional DDP
    3. optional torch.compile
    """
    inner = getattr(model, "_orig_mod", model)
    if isinstance(inner, DDP):
        return inner.module
    return inner


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

    # Optional gene-mask extractor for the gene LFC training loss (B3.2).
    # Only attached to the rna_seq dataset; gene_mask is sample-level so
    # MultimodalDataset will propagate it to the batch.
    gene_mask_extractor = None
    g_max = None
    if args.gene_loss_weight > 0:
        from alphagenome_pytorch.extensions.finetuning.gene_annotation import (
            GeneMaskExtractor,
            cached_load_gene_table,
            derive_g_max,
        )
        from alphagenome_pytorch.extensions.finetuning.datasets import (
            _load_intervals_from_bed,
        )

        print_rank0(f"Loading GTF for gene LFC loss: {args.gtf}", rank)
        gene_table = cached_load_gene_table(args.gtf, filter_protein_coding=True)
        gene_mask_extractor = GeneMaskExtractor(gene_table)

        # Project the BED windows used by the dataset (after centering/expansion
        # to args.sequence_length) so derive_g_max sees the same intervals
        # GenomicDataset will request at __getitem__.
        all_intervals: list[tuple[str, int, int]] = []
        for bed in (args.train_bed, args.val_bed):
            raw_intervals, _ = _load_intervals_from_bed(bed)
            half_len = args.sequence_length // 2
            for chrom, s, e in raw_intervals:
                center = (s + e) // 2
                all_intervals.append((chrom, center - half_len, center + half_len))
        g_max = derive_g_max(gene_mask_extractor, all_intervals)
        print_rank0(
            f"Gene LFC: scanned {len(all_intervals)} intervals, g_max={g_max}",
            rank,
        )

    # Always create MultimodalDataset (even for single modality) to have a unified interface
    # This is required by train_epoch_sequence_parallel
    print_rank0("Creating datasets...", rank)
    train_datasets = {}
    val_datasets = {}

    for modality, bigwigs in args.modality_to_bigwigs.items():
        resolutions = args.modality_resolutions[modality]
        # Attach the gene-mask extractor only to the modality that consumes
        # the gene LFC loss (rna_seq today).
        attach_gene_mask = (
            gene_mask_extractor is not None
            and modality == "rna_seq"
            and args.gene_loss_weight > 0
        )
        gme = gene_mask_extractor if attach_gene_mask else None
        gme_g_max = g_max if attach_gene_mask else None
        train_datasets[modality] = GenomicDataset(
            genome_fasta=genome,
            bigwig_files=bigwigs,
            bed_file=args.train_bed,
            resolutions=resolutions,
            sequence_length=args.sequence_length,
            cache_genome=cache_genome,
            cache_signals=cache_signals,
            max_io_workers=max_io_workers,
            gene_mask_extractor=gme,
            g_max=gme_g_max,
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
            gene_mask_extractor=gme,
            g_max=gme_g_max,
        )

    train_dataset = MultimodalDataset(train_datasets)
    val_dataset = MultimodalDataset(val_datasets)

    print_rank0(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}", rank)

    return train_dataset, val_dataset, modality_track_names, args.modality_resolutions


ORGANISM_NAME_TO_INDEX = {"human": 0, "mouse": 1}


def organism_index_from_args(args: argparse.Namespace) -> int:
    """Biological organism index (0=human, 1=mouse) this fine-tune trains.

    Drives the forward pass: mouse data is forwarded at index 1 so it uses the
    mouse organism embedding. Heads stay ``num_organisms=1`` (organism-agnostic)
    and ignore the index, so no head capacity is wasted.
    """
    return ORGANISM_NAME_TO_INDEX.get(getattr(args, "organism", None) or "human", 0)


def load_track_metadata_for_finetune(
    path: str | None,
    modality_track_names: Mapping[str, Sequence[str]],
    rank: int,
    organism: str | None = None,
) -> tuple[dict[str, list[str]], list[dict[str, Any]] | None]:
    """Load and validate user-supplied track metadata for fine-tuning.

    Returns ``(track_names, metadata_rows)``:

    * ``track_names`` — possibly updated track-name dict (rows in the parquet
      override BigWig stems so checkpoint and embedded catalog agree).
    * ``metadata_rows`` — list-of-dicts ready for embedding into checkpoints,
      or ``None`` when ``path`` is ``None``.

    A fine-tune trains a single organism (the forward uses one organism
    embedding, selected by ``--organism``), so the metadata must describe that
    organism. ``organism`` (``"human"``/``"mouse"``, default human) is the
    organism the heads are trained for; it fills rows whose parquet ``organism``
    column is absent. If the parquet declares any *other* organism, that is a
    mistake (mixed human+mouse training is not supported yet) and raises — this
    also stops mouse data from being embedded while the trainer forwards at the
    human embedding.

    Validates that every fine-tuning head has a matching ``output_type`` in the
    catalog with the right number of tracks. Head name == ``--modality`` by
    convention in this script.
    """
    if path is None:
        return modality_track_names, None

    organism_index = ORGANISM_NAME_TO_INDEX.get(organism or "human", 0)
    organism_name = "mouse" if organism_index == 1 else "human"
    # The default fills rows whose 'organism' value is absent; a per-track
    # 'organism' column wins (and must agree with --organism, checked below).
    catalog = TrackMetadataCatalog.from_file(path, default_organism=organism_index)
    print_rank0(f"Loaded track metadata from {path}", rank)

    present = set(catalog.organisms)
    if present - {organism_index}:
        raise ValueError(
            f"--track-metadata declares organism(s) {sorted(present)}, but this "
            f"fine-tune trains organism {organism_index} ({organism_name}). "
            "Fine-tuning is single-organism: set every row's 'organism' to match "
            "--organism (use --organism mouse for mouse tracks). Mixed "
            "human+mouse training is not supported yet."
        )

    updated_names: dict[str, list[str]] = {}
    for head_name, bigwig_names in modality_track_names.items():
        tracks = catalog.get_tracks(head_name, organism=organism_index)
        if not tracks:
            available = catalog.outputs(organism=organism_index)
            raise ValueError(
                f"--track-metadata has no entries for head/output '{head_name}'. "
                f"Available outputs: {available}. The 'output_type' column "
                "must match the head name (= --modality)."
            )
        if len(tracks) != len(bigwig_names):
            raise ValueError(
                f"--track-metadata has {len(tracks)} tracks for '{head_name}', "
                f"but {len(bigwig_names)} BigWig file(s) were provided. "
                "Counts must match."
            )
        updated_names[head_name] = [t.track_name for t in tracks]

    # Embed all rows (all organism `organism_index` after the check above) so the
    # served catalog labels the tracks with the correct organism.
    metadata_rows = catalog.to_rows()
    return updated_names, metadata_rows


def apply_training_strands(
    metadata_rows: list[dict[str, Any]] | None,
    modality_strands: Mapping[str, Sequence[str]],
    modality_track_names: Mapping[str, Sequence[str]],
    organism: str | None = None,
    rank: int = 0,
) -> list[dict[str, Any]]:
    """Record the strands a fine-tune used into the embedded track metadata.

    Keeps the catalog **complete** — one row per track of *every* head — so it
    never shadows the per-head ``track_names`` fallback (a partial catalog is
    treated as authoritative by serving, blanking heads it omits). Then overlays
    the per-track strand for heads trained with ``--track-strands``.

    Base rows are the rich ``--track-metadata`` catalog when present, else a
    skeleton built from ``track_names``. Training strands fill missing values and
    override any that disagree (with a warning), since they are what the run
    actually used. Heads without a ``--track-strands`` entry keep their metadata
    but get no strand — their strand was never specified. Rows are keyed as
    ``TrackMetadataCatalog.to_rows`` emits (``output_name``).
    """
    organism_index = ORGANISM_NAME_TO_INDEX.get(organism or "human", 0)

    if metadata_rows is None:
        rows: list[dict[str, Any]] = [
            {"output_name": head, "track_index": i,
             "organism": organism_index, "track_name": name}
            for head, names in modality_track_names.items()
            for i, name in enumerate(names)
        ]
    else:
        rows = [dict(r) for r in metadata_rows]

    def _head(row: dict[str, Any]) -> str:
        return str(row.get("output_name") or row.get("output_type") or "").lower()

    by_track = {(_head(r), r.get("track_index")): r for r in rows}

    for head, strands in modality_strands.items():
        for i, strand in enumerate(strands):
            row = by_track.get((head.lower(), i))
            if row is None:
                # A track present in --track-strands but absent from the base
                # rows (e.g. rich metadata missing a track): add it so the
                # strand is still recorded.
                row = {"output_name": head, "track_index": i, "organism": organism_index}
                names = modality_track_names.get(head)
                if names is not None and i < len(names):
                    row["track_name"] = names[i]
                rows.append(row)
                by_track[(head.lower(), i)] = row
            existing = row.get("strand")
            if existing is not None and str(existing) != str(strand):
                print_rank0(
                    f"--track-strands overrides embedded strand for {head} "
                    f"track {i}: metadata {existing!r} -> training {strand!r}.",
                    rank,
                )
            row["strand"] = str(strand)
    return rows


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int,
    is_multimodal: bool = False,
    sequence_parallel_mode: bool = False,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
    """Create data loaders with optional distributed samplers.

    Args:
        sequence_parallel_mode: If True, use non-distributed sampler (all ranks see same data).
    """
    # In sequence-parallel mode, all ranks must process the same sequence (shards of it)
    if sequence_parallel_mode:
        train_sampler = None
        val_sampler = None
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    # Always use collate_multimodal since we now always use MultimodalDataset
    collate_fn = collate_multimodal

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
) -> tuple[nn.Module, dict[str, nn.Module], list[torch.nn.Parameter], TransferConfig | None]:
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
        Tuple of (model, heads_dict, trainable_params, transfer_config).
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

    # Build new_heads dict for TransferConfig (used for delta checkpoints).
    # Heads stay single-organism (organism-agnostic): the organism only selects
    # the trunk embedding in the forward, not a head weight slot.
    new_heads_config: dict[str, dict] = {}
    for modality, track_names in modality_track_names.items():
        head_res = (128,) if is_encoder_only else modality_resolutions[modality]
        new_heads_config[modality] = {
            "modality": modality,
            "num_tracks": len(track_names),
            "resolutions": list(head_res),
            "encoder_only": is_encoder_only,
            "track_means": modality_track_means.get(modality),
            "num_organisms": 1,
            "init_scheme": args.head_init_scheme,
        }

    # Create heads directly except in active adapter modes, where
    # prepare_for_transfer() constructs the actual trainable heads we want the
    # optimizer to own.
    heads: dict[str, nn.Module] = {}
    has_active_adapters = (
        (args.mode in {"lora", "lora+locon"} and args.lora_rank > 0)
        or (args.mode in {"locon", "lora+locon"} and args.locon_rank > 0)
    )
    create_heads_directly = not has_active_adapters
    if create_heads_directly:
        for modality, track_names in modality_track_names.items():
            head = create_finetuning_head(
                assay_type=modality,
                n_tracks=len(track_names),
                resolutions=tuple(new_heads_config[modality]["resolutions"]),
                num_organisms=1,
                track_means=modality_track_means.get(modality),
                init_scheme=args.head_init_scheme,
                encoder_only=is_encoder_only,
            )
            add_head(model, modality, head)
            heads[modality] = head
            print_rank0(
                f"Created {modality} head with {len(track_names)} tracks "
                f"at resolutions {tuple(new_heads_config[modality]['resolutions'])}",
                rank,
            )

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

    elif args.mode in {"lora", "locon", "lora+locon"}:
        lora_enabled = args.mode in {"lora", "lora+locon"} and args.lora_rank > 0
        locon_enabled = args.mode in {"locon", "lora+locon"} and args.locon_rank > 0

        lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        locon_targets = [t.strip() for t in args.locon_targets.split(",") if t.strip()]

        adapter_modes: list[str] = []
        if lora_enabled:
            adapter_modes.append("lora")
            print_rank0(f"Applying LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}", rank)
            print_rank0(f"  Target modules: {lora_targets}", rank)
        if locon_enabled:
            validate_locon_targets(model, locon_targets)
            adapter_modes.append("locon")
            print_rank0(f"Applying Locon: rank={args.locon_rank}, alpha={args.locon_alpha}", rank)
            print_rank0(f"  Target modules: {locon_targets}", rank)

        if adapter_modes:
            transfer_mode: str | list[str]
            transfer_mode = adapter_modes[0] if len(adapter_modes) == 1 else adapter_modes

            transfer_config = TransferConfig(
                mode=transfer_mode,
                lora_targets=lora_targets,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                locon_targets=locon_targets,
                locon_rank=args.locon_rank,
                locon_alpha=args.locon_alpha,
                new_heads=new_heads_config,
            )
            model = prepare_for_transfer(model, transfer_config)
            heads = {
                modality: model.heads[modality]
                for modality in modality_track_names
            }
            for modality, track_names in modality_track_names.items():
                print_rank0(
                    f"Created {modality} head with {len(track_names)} tracks "
                    f"at resolutions {tuple(new_heads_config[modality]['resolutions'])}",
                    rank,
                )
            # Adapter weights + the freshly registered heads.
            trainable_params = get_adapter_params(model)
            for head in heads.values():
                trainable_params.extend(list(head.parameters()))
        else:
            # Adapter rank 0 means just train heads
            for head in heads.values():
                trainable_params.extend(list(head.parameters()))
            transfer_config = TransferConfig(mode="linear", new_heads=new_heads_config)
            print_rank0(f"Mode: {args.mode} (adapter rank=0, heads only)", rank)

    elif args.mode == "full":
        # All parameters trainable (model was not frozen above)
        trainable_params = list(model.parameters())
        # Embed TransferConfig so checkpoints are self-describing at load time
        # (head names, modalities, resolutions). --save-delta is rejected at
        # parse time because delta checkpoints cannot capture trunk updates.
        transfer_config = TransferConfig(mode="full", new_heads=new_heads_config)
        print_rank0("Mode: full (all parameters trainable)", rank)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Move to device
    model = model.to(device)

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print_rank0("Model wrapped with DistributedDataParallel", rank)

    # Get head references from the underlying model before optional compile.
    model_module = unwrap_training_model(model)
    heads = {modality: model_module.heads[modality] for modality in heads}

    # Optionally compile
    if args.compile:
        print_rank0("Compiling model with torch.compile...", rank)
        import torch._inductor.config as inductor_config
        inductor_config.group_fusion = False
        model = torch.compile(model)
        model_module = unwrap_training_model(model)

    # Count parameters
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model_module.parameters())
    print_rank0(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)", rank)

    return model, heads, trainable_params, transfer_config


# =============================================================================
# Main
# =============================================================================


def main(args: argparse.Namespace | None = None) -> None:
    """Main training function."""
    if args is None:
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

    # Optional gene-expression validation metric for rna_seq.
    # Load a GeneAnnotation WITH exon rows (decoupled from --gene-loss-weight,
    # which only needs a gene-only body table) and stream per-window coords to
    # the val loop so it can build exon masks.
    gene_expr_annotation = None
    gene_expr_track_strands = None
    # Per-window exon-mask cache, created once and reused every val epoch so the
    # pandas-heavy gene lookup runs once per window for the whole run.
    gene_expr_window_cache: dict = {}
    if getattr(args, "gene_expr_eval", False):
        from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation

        ann_path = args.gene_expr_annotation or args.gtf
        print_rank0(f"Loading annotation for gene-expression eval: {ann_path}", rank)
        gene_expr_annotation = GeneAnnotation(ann_path)
        # The metric aggregates over exons, but --gtf may legitimately be a
        # gene-only annotation (the gene-LFC loss only needs gene rows). Without
        # exons the metric finds no genes and reports NaN every epoch, silently,
        # for the whole run — so fail here, as the AnnData export path does.
        if not gene_expr_annotation.has_exon_annotations():
            raise ValueError(
                f"--gene-expr-eval needs an annotation with exon rows, but none were "
                f"found in {ann_path}. Pass --gene-expr-annotation pointing at a "
                f"GTF/parquet that includes exon features."
            )
        gene_expr_track_strands = list(args.modality_strands["rna_seq"])
        val_dataset.return_coords = True

    # Optional rich track metadata (overrides BigWig stems with parquet names
    # and embeds the catalog into checkpoints / exported delta weights).
    modality_track_names, track_metadata_rows = load_track_metadata_for_finetune(
        args.track_metadata, modality_track_names, rank, organism=args.organism,
    )
    # Record the strands training used so the checkpoint is self-describing for
    # downstream strand-matched aggregation (agt predict --gene-strand match).
    # Overlays onto the rich catalog when present, or a complete skeleton built
    # from track_names otherwise -- never a partial catalog that would blank the
    # metadata of heads it omits.
    if getattr(args, "modality_strands", None):
        track_metadata_rows = apply_training_strands(
            track_metadata_rows, args.modality_strands, modality_track_names,
            organism=args.organism, rank=rank,
        )

    # The organism this fine-tune trains, resolved once here and reused for both
    # checkpoint metadata and the training/validation forward pass so producer
    # metadata can never drift from actual training routing. Note: args.organism
    # is None for a default-human run, so we persist the *resolved* name/index
    # (not args.organism) — otherwise a new human checkpoint records "unknown".
    organism_index = organism_index_from_args(args)
    organism_name = "mouse" if organism_index == 1 else "human"

    # Track identity embedded into every checkpoint/delta save below. Defined
    # once and spread as **metadata_kwargs so a new save site cannot silently
    # drop the embedded metadata. ``organism`` is the compatibility scalar;
    # ``organism_indices`` is the forward-facing plural form.
    metadata_kwargs = dict(
        track_names=modality_track_names,
        modality=args.modalities,
        resolutions=modality_resolutions,
        track_metadata=track_metadata_rows,
        organism=organism_name,
        organism_indices=[organism_index],
    )

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
        is_multimodal=True,  # Always multimodal now
        sequence_parallel_mode=args.sequence_parallel,
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
                strand_pair_groups=args.modality_strand_pairs.get(modality),
            )
            print(f"  {modality}: mean={modality_track_means[modality].mean():.4f}")
    modality_track_means = broadcast_object(modality_track_means, src=0)

    # Create model
    model, heads, trainable_params, transfer_config = create_model(
        args,
        modality_track_names,
        modality_track_means,
        modality_resolutions,
        device,
        rank,
        world_size,
        local_rank,
    )
    model_module = unwrap_training_model(model)

    # Build per-modality strand-channel masks for the gene LFC loss (B3.2).
    # Empty dict when gene_loss_weight is 0; populated only for modalities
    # whose strand info was supplied (today: rna_seq via --track-strands or
    # the YAML strand field). Each mask is `[2, 1, C]` and lives on `device`.
    gene_strand_channel_masks: dict[str, torch.Tensor] = {}
    if args.gene_loss_weight > 0:
        from alphagenome_pytorch.training import _build_strand_channel_mask
        for modality, strands in args.modality_strands.items():
            gene_strand_channel_masks[modality] = (
                _build_strand_channel_mask(strands).to(device)
            )

    # Per-modality gene_loss_weights dict. Today only rna_seq receives a
    # non-zero entry; other modalities are absent from the dict and the
    # training loop's `gene_loss_weights.get(modality, 0.0)` returns 0.0.
    gene_loss_weights: dict[str, float] = {}
    if args.gene_loss_weight > 0:
        gene_loss_weights["rna_seq"] = args.gene_loss_weight

    # Sequence parallelism setup
    sequence_parallel = None
    if args.sequence_parallel:
        if world_size == 1:
            print_rank0(
                "Warning: --sequence-parallel requires multiple GPUs. Running with single GPU.",
                rank,
            )
        else:
            sequence_parallel = SequenceParallelism(
                overlap_highres=args.overlap_highres,
                overlap_lowres=args.overlap_highres // 128,
            )
            overlap_lowres = args.overlap_highres // 128
            print_rank0(
                f"Sequence parallelism enabled: overlap_highres={args.overlap_highres}, "
                f"overlap_lowres={overlap_lowres}",
                rank,
            )

    # Only include transfer_config in checkpoints when one exists, so loaders
    # can cleanly distinguish "no config saved" from "config was None".
    transfer_config_kwargs = (
        {"transfer_config": transfer_config_to_dict(transfer_config)}
        if transfer_config is not None
        else {}
    )

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
        "lora_rank": args.lora_rank if args.mode in ("lora", "lora+locon") else None,
        "lora_alpha": args.lora_alpha if args.mode in ("lora", "lora+locon") else None,
        "lora_targets": args.lora_targets if args.mode in ("lora", "lora+locon") else None,
        "locon_rank": args.locon_rank if args.mode in ("locon", "lora+locon") else None,
        "locon_alpha": args.locon_alpha if args.mode in ("locon", "lora+locon") else None,
        "locon_targets": args.locon_targets if args.mode in ("locon", "lora+locon") else None,
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
        """Save preemption checkpoint, honoring --save-delta / --no-full-checkpoint."""
        if not (is_main_process(rank) and not args.no_save_checkpoints):
            return
        last_completed = max(0, current_epoch - 1)
        if not args.no_full_checkpoint:
            save_checkpoint(
                path=output_dir / "checkpoint_preempt.pth",
                epoch=last_completed,
                model=model_module,
                optimizer=optimizer,
                val_loss=best_val_loss,
                **metadata_kwargs,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                wandb_run_id=logger.wandb_run_id,
                **transfer_config_kwargs,
            )
            print(f"Preemption checkpoint saved to {output_dir / 'checkpoint_preempt.pth'}")
        if args.save_delta and transfer_config is not None:
            save_delta_checkpoint(
                path=output_dir / "checkpoint_preempt.delta.pth",
                model=model_module,
                config=transfer_config,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=last_completed,
                val_loss=best_val_loss,
                best_val_loss=best_val_loss,
                **metadata_kwargs,
                wandb_run_id=logger.wandb_run_id,
            )
            print(f"Preemption delta checkpoint saved to {output_dir / 'checkpoint_preempt.delta.pth'}")

    handler = setup_preemption_handler(_save_preempt, rank, world_size)

    # Training loop
    print_rank0("\n" + "=" * 60, rank)
    print_rank0(f"Starting training (epoch {start_epoch} to {args.epochs})", rank)
    print_rank0("=" * 60, rank)

    # Freeze backbone (use torch.no_grad) when no backbone params need gradients.
    # - linear-probe / encoder-only: only heads train
    # - lora / locon / lora+locon with all adapter ranks == 0: only heads train
    # - adapter modes with active adapters: adapters need gradients, can't freeze
    # - full: all params need gradients
    has_active_adapters = (
        (args.mode in ("lora", "lora+locon") and args.lora_rank > 0)
        or (args.mode in ("locon", "lora+locon") and args.locon_rank > 0)
    )
    frozen_backbone = args.mode in ("linear-probe", "encoder-only") or (
        args.mode in ("lora", "locon", "lora+locon") and not has_active_adapters
    )
    encoder_only = args.mode == "encoder-only"
    # ``organism_index`` was resolved once above (near metadata_kwargs) and is
    # reused here: mouse data forwards the trunk at index 1 so it uses the mouse
    # trunk embedding. Fine-tuned heads are single-organism (num_organisms=1) and
    # organism-agnostic — they map any index to slot 0 — so the organism index
    # only selects the trunk embedding, not a head weight slot.

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
            if args.sequence_parallel and sequence_parallel is not None:
                # Sequence parallel training (distributes sequence across GPUs)
                train_loss, per_modality_train_loss = train_epoch_sequence_parallel(
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
                    sequence_parallel=sequence_parallel,
                    epoch=epoch,
                    log_every=args.log_every,
                    use_amp=use_amp,
                    accumulation_steps=args.gradient_accumulation_steps,
                    frozen_backbone=frozen_backbone,
                    num_segments=args.num_segments,
                    min_segment_size=args.min_segment_size,
                    train_sampler=train_sampler,
                    rank=rank,
                    world_size=world_size,
                    max_grad_norm=args.max_grad_norm,
                    profile_batches=args.profile_batches if epoch == start_epoch else 0,
                    log_fn=logger.log_step if is_main_process(rank) else None,
                    encoder_only=encoder_only,
                    gene_loss_weights=gene_loss_weights,
                    gene_cross_track_weight=args.gene_cross_track_weight,
                    strand_channel_masks=gene_strand_channel_masks,
                    organism=organism_index,
                )
            else:
                # Standard multimodal training (uses multihead functions)
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
                    gene_loss_weights=gene_loss_weights,
                    gene_cross_track_weight=args.gene_cross_track_weight,
                    strand_channel_masks=gene_strand_channel_masks,
                    organism=organism_index,
                )

            if handler.preempted:
                print_rank0("Preemption flag set - saving and exiting.", rank)
                handler.save_and_exit()
                break

            # Validation (always use multihead since we always have multimodal dataset format now)
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
                organism=organism_index,
                gene_annotation=gene_expr_annotation,
                gene_expr_track_strands=gene_expr_track_strands,
                gene_expr_window_cache=gene_expr_window_cache,
            )

            # Synchronize CUDA to ensure all validation ops complete before next epoch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            current_lr = scheduler.get_last_lr()[0]
            is_best = val_loss < best_val_loss

            # Print epoch summary
            if is_main_process(rank):
                summary = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                # Always print per-modality losses (we always have multimodal dataset format now)
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
            if is_main_process(rank) and not args.no_save_checkpoints:
                # Delta saves require a transfer_config (full mode now also
                # builds one, so this only skips the delta write in legacy
                # paths where transfer_config is None).
                write_delta = args.save_delta and transfer_config is not None
                write_full = not args.no_full_checkpoint

                if is_best:
                    best_val_loss = val_loss
                    if write_full:
                        save_checkpoint(
                            path=output_dir / "best_model.pth",
                            epoch=epoch,
                            model=model_module,
                            optimizer=optimizer,
                            val_loss=val_loss,
                            **metadata_kwargs,
                            scheduler=scheduler,
                            best_val_loss=best_val_loss,
                            wandb_run_id=logger.wandb_run_id,
                            **transfer_config_kwargs,
                        )
                        print(f"  Saved best model (val_loss={val_loss:.4f})")
                    if write_delta:
                        save_delta_checkpoint(
                            path=output_dir / "best_model.delta.pth",
                            model=model_module,
                            config=transfer_config,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            val_loss=val_loss,
                            best_val_loss=best_val_loss,
                            **metadata_kwargs,
                            wandb_run_id=logger.wandb_run_id,
                        )
                        print(f"  Saved best delta checkpoint (val_loss={val_loss:.4f})")

                if epoch % args.save_every == 0:
                    if write_full:
                        save_checkpoint(
                            path=output_dir / f"checkpoint_epoch{epoch}.pth",
                            epoch=epoch,
                            model=model_module,
                            optimizer=optimizer,
                            val_loss=val_loss,
                            **metadata_kwargs,
                            scheduler=scheduler,
                            best_val_loss=best_val_loss,
                            wandb_run_id=logger.wandb_run_id,
                            **transfer_config_kwargs,
                        )
                    if write_delta:
                        save_delta_checkpoint(
                            path=output_dir / f"checkpoint_epoch{epoch}.delta.pth",
                            model=model_module,
                            config=transfer_config,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            val_loss=val_loss,
                            best_val_loss=best_val_loss,
                            **metadata_kwargs,
                            wandb_run_id=logger.wandb_run_id,
                        )

            barrier()

    except KeyboardInterrupt:
        print_rank0("\nTraining interrupted by user", rank)
    finally:
        logger.finish()
        handler.unregister()
        cleanup_distributed()

    # Export transfer config if requested
    if args.export_transfer_config and transfer_config is not None and is_main_process(rank):
        import json
        config_path = Path(args.export_transfer_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(transfer_config_to_dict(transfer_config), f, indent=2)
        print(f"Exported TransferConfig to {config_path}")

    print_rank0(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}", rank)
    print_rank0(f"Output: {output_dir}", rank)
