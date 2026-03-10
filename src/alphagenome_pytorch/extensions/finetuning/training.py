"""Unified training utilities for AlphaGenome fine-tuning.

Provides common training functions for both RNA-seq and ATAC-seq modalities.
Includes enhanced versions with DDP support, profiling, and Pearson R metrics.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from alphagenome_pytorch.losses import multinomial_loss

# Number of segments for multinomial loss computation.
# AlphaGenome divides sequences into 8 equal segments for numerical stability.
NUM_SEGMENTS = 8

if TYPE_CHECKING:
    from torch.optim import Optimizer


def collate_genomic(
    batch: list[tuple[Tensor, dict[int, Tensor]]],
) -> tuple[Tensor, dict[int, Tensor]]:
    """Collate function for genomic fine-tuning datasets.

    Args:
        batch: List of (sequence, targets_dict) tuples from dataset.

    Returns:
        Tuple of (sequences, targets_dict) where:
            - sequences: Stacked sequences tensor (batch, seq_len, 4)
            - targets_dict: Dict mapping resolution to targets (batch, out_len, n_tracks)

    Example:
        >>> batch = [(seq1, {1: t1_1bp, 128: t1_128bp}), (seq2, {1: t2_1bp, 128: t2_128bp})]
        >>> sequences, targets = collate_genomic(batch)
        >>> targets[1].shape, targets[128].shape
    """
    sequences = torch.stack([item[0] for item in batch])

    # Targets are always a dict
    first_targets = batch[0][1]
    targets_dict: dict[int, Tensor] = {}
    for res in first_targets.keys():
        targets_dict[res] = torch.stack([item[1][res] for item in batch])

    return sequences, targets_dict


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


def create_lr_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
) -> LambdaLR:
    """Create learning rate scheduler with optional warmup.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Number of warmup steps (linear ramp from 0 to lr).
        total_steps: Total number of training steps.
        schedule: Schedule type after warmup. Options:
            - "cosine": Cosine decay to 0 (default)
            - "constant": Constant learning rate

    Returns:
        LambdaLR scheduler.

    Examples:
        # Warmup + cosine decay (default)
        scheduler = create_lr_scheduler(opt, warmup_steps=500, total_steps=10000)

        # Constant learning rate (no warmup, no decay)
        scheduler = create_lr_scheduler(opt, warmup_steps=0, total_steps=10000, schedule="constant")

        # Warmup then constant
        scheduler = create_lr_scheduler(opt, warmup_steps=500, total_steps=10000, schedule="constant")
    """
    if schedule not in ("cosine", "constant"):
        raise ValueError(f"Unknown schedule: {schedule}. Must be 'cosine' or 'constant'.")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule == "constant":
            return 1.0
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_finetuning_loss(
    predictions: dict[int, Tensor],
    targets: dict[int, Tensor],
    resolution_weights: dict[int, float],
    positional_weight: float,
    device: torch.device,
    channels_last: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute combined loss across resolutions.

    Uses dynamic multinomial_resolution = seq_len // 8 for consistent loss
    granularity across different sequence lengths.

    Args:
        predictions: Dict mapping resolution to prediction tensors.
        targets: Dict mapping resolution to target tensors.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        device: Torch device.
        channels_last: If True, assumes (B, S, C). If False, assumes (B, C, S).

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains per-resolution
        losses and other metrics.
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_dict: dict[str, Tensor] = {}

    for res, weight in resolution_weights.items():
        if res not in predictions:
            continue

        pred = predictions[res]
        target = targets[res]

        # Detect dimensions based on format
        if channels_last:
            # (B, S, C)
            current_seq_len = pred.shape[-2]
            num_channels = pred.shape[-1]
            mask_shape = (pred.shape[0], 1, num_channels)
        else:
            # (B, C, S)
            current_seq_len = pred.shape[-1]
            num_channels = pred.shape[-2]
            mask_shape = (pred.shape[0], num_channels, 1)

        # Use multinomial_resolution matching JAX for 1Mb sequences (2^17 // res),
        # but allow for fewer segments if the sequence is shorter.
        # This ensures segments are at least 131k bp (at 1bp) and that
        # multinomial_resolution always divides current_seq_len.
        num_segments = max(1, min(8, current_seq_len // (131072 // res)))
        multinomial_resolution = current_seq_len // num_segments

        # Create mask (all True)
        mask = torch.ones(*mask_shape, dtype=torch.bool, device=device)

        res_loss_dict = multinomial_loss(
            y_pred=pred,
            y_true=target,
            mask=mask,
            multinomial_resolution=multinomial_resolution,
            positional_weight=positional_weight,
            channels_last=channels_last,
        )

        total_loss = total_loss + weight * res_loss_dict["loss"]
        loss_dict[f"loss_{res}bp"] = res_loss_dict["loss"]

    loss_dict["loss"] = total_loss
    return total_loss, loss_dict


def train_epoch(
    model: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    resolutions: tuple[int, ...] | None = None,
) -> float:
    """Train for one epoch.

    Args:
        model: AlphaGenome trunk model.
        head: Output head module.
        train_loader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision (default: True).
        accumulation_steps: Number of batches to accumulate gradients over
            before performing an optimizer step. Useful for simulating larger
            batch sizes when GPU memory is limited (default: 1, no accumulation).
        resolutions: Tuple of resolutions to train on (e.g., (1,), (128,), or (1, 128)).
            If None, inferred from resolution_weights keys. Training on 1bp resolution
            requires significantly more memory than 128bp.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    head.train()

    total_loss = 0.0
    n_batches = 0

    # Determine which resolutions to use
    if resolutions is None:
        resolutions = tuple(resolution_weights.keys())
    if invalid := (set(resolutions) - {1, 128}):
        raise ValueError(f"Invalid resolutions {invalid}, must be 1 or 128")

    # Set up autocast context (bfloat16 on CUDA, no-op on CPU)
    if use_amp and device.type == "cuda":
        amp_context = autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (sequences, targets_dict) in enumerate(pbar):
        sequences = sequences.to(device)
        targets_dict = {k: v.to(device) for k, v in targets_dict.items() if k in resolutions}

        # Organism index (assume human)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        with amp_context:
            # Forward through trunk
            outputs = model(sequences, organism_idx, return_embeddings=True, channels_last=False)

            # Only get embeddings for requested resolutions (1bp is 128x larger than 128bp)
            embeddings_dict = {}
            if 1 in resolutions:
                emb = outputs.get("embeddings_1bp")
                if emb is not None:
                    embeddings_dict[1] = emb
            if 128 in resolutions:
                emb = outputs.get("embeddings_128bp")
                if emb is not None:
                    embeddings_dict[128] = emb

            # Forward through head
            predictions = head(embeddings_dict, organism_idx)

            # Compute loss
            loss, _ = compute_finetuning_loss(
                predictions=predictions,
                targets=targets_dict,
                resolution_weights=resolution_weights,
                positional_weight=positional_weight,
                device=device,
                channels_last=True,
            )

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        # Optimizer step every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % log_every == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

    # Handle remaining gradients if dataset size is not divisible by accumulation_steps
    if n_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    head: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    use_amp: bool = True,
    resolutions: tuple[int, ...] | None = None,
) -> float:
    """Validate the model.

    Args:
        model: AlphaGenome trunk model.
        head: Output head module.
        val_loader: Validation data loader.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        use_amp: Whether to use automatic mixed precision (default: True).
        resolutions: Tuple of resolutions to validate on (e.g., (1,), (128,), or (1, 128)).
            If None, inferred from resolution_weights keys.

    Returns:
        Average validation loss.
    """
    model.eval()
    head.eval()

    total_loss = 0.0
    n_batches = 0

    # Determine which resolutions to use
    if resolutions is None:
        resolutions = tuple(resolution_weights.keys())
    if invalid := (set(resolutions) - {1, 128}):
        raise ValueError(f"Invalid resolutions {invalid}, must be 1 or 128")

    # Set up autocast context (bfloat16 on CUDA, no-op on CPU)
    if use_amp and device.type == "cuda":
        amp_context = autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    for sequences, targets_dict in tqdm(val_loader, desc="Validation"):
        sequences = sequences.to(device)
        targets_dict = {k: v.to(device) for k, v in targets_dict.items() if k in resolutions}
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        with amp_context:
            outputs = model(sequences, organism_idx, return_embeddings=True, channels_last=False)

            # Only get embeddings for requested resolutions
            embeddings_dict = {}
            if 1 in resolutions:
                emb = outputs.get("embeddings_1bp")
                if emb is not None:
                    embeddings_dict[1] = emb
            if 128 in resolutions:
                emb = outputs.get("embeddings_128bp")
                if emb is not None:
                    embeddings_dict[128] = emb

            predictions = head(embeddings_dict, organism_idx)

            loss, _ = compute_finetuning_loss(
                predictions=predictions,
                targets=targets_dict,
                resolution_weights=resolution_weights,
                positional_weight=positional_weight,
                device=device,
                channels_last=True,
            )

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# Re-export save_checkpoint from checkpointing module for backward compatibility
from alphagenome_pytorch.extensions.finetuning.checkpointing import save_checkpoint


# =============================================================================
# Profiling utilities
# =============================================================================


class ProfilingStats:
    """Collect timing statistics for profiling training batches.

    Example:
        >>> stats = ProfilingStats()
        >>> t0 = time.perf_counter()
        >>> # ... some operation ...
        >>> stats.add("forward", time.perf_counter() - t0)
        >>> print(stats.report(n_batches=10))
    """

    def __init__(self) -> None:
        self.times: dict[str, list[float]] = defaultdict(list)

    def add(self, name: str, elapsed: float) -> None:
        """Add a timing measurement.

        Args:
            name: Name of the operation (e.g., "forward", "backward").
            elapsed: Elapsed time in seconds.
        """
        self.times[name].append(elapsed)

    def report(self, n_batches: int) -> str:
        """Generate a profiling report.

        Args:
            n_batches: Number of batches profiled.

        Returns:
            Formatted report string with timing breakdowns.
        """
        import numpy as np

        lines = ["\n" + "=" * 70, "PROFILING REPORT", "=" * 70]
        total_time = 0.0

        for name, times in sorted(self.times.items()):
            arr = np.array(times)
            total_time += arr.sum()
            lines.append(
                f"\n{name}:\n"
                f"  Mean:  {arr.mean()*1000:.2f} ms (+/- {arr.std()*1000:.2f} ms)\n"
                f"  Total: {arr.sum():.2f} s ({len(times)} samples)"
            )

        lines.append(f"\n{'=' * 70}")
        lines.append(f"TOTAL TIME: {total_time:.2f} s for {n_batches} batches")
        lines.append(f"AVG TIME PER BATCH: {total_time/n_batches*1000:.2f} ms")

        # Breakdown percentages
        lines.append(f"\nBREAKDOWN:")
        for name, times in sorted(self.times.items()):
            pct = np.sum(times) / total_time * 100
            lines.append(f"  {name}: {pct:.1f}%")

        lines.append("=" * 70)
        return "\n".join(lines)

    def estimated_epoch_time(self, total_batches: int) -> float:
        """Estimate total epoch time based on profiled batches.

        Args:
            total_batches: Total number of batches in the epoch.

        Returns:
            Estimated epoch time in seconds.
        """
        import numpy as np

        n_profiled = len(next(iter(self.times.values()))) if self.times else 0
        if n_profiled == 0:
            return 0.0

        total_profiled_time = sum(np.sum(t) for t in self.times.values())
        avg_batch_time = total_profiled_time / n_profiled
        return avg_batch_time * total_batches


# =============================================================================
# Enhanced training functions with DDP and profiling support
# =============================================================================


def _cuda_sync(device: torch.device) -> None:
    """Synchronize CUDA if on GPU (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _compute_multinomial_resolution(
    seq_len: int,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
) -> int:
    """Compute positions per segment for multinomial loss.

    Args:
        seq_len: Total sequence length (number of positions).
        num_segments: Target number of segments.
        min_segment_size: Minimum positions per segment (optional).

    Returns:
        Resolution (positions per segment).
    """
    resolution = max(1, seq_len // num_segments)

    if min_segment_size is not None:
        resolution = max(resolution, min_segment_size)

    return resolution


def train_epoch_ddp(
    model: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    count_weight: float,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    frozen_backbone: bool = False,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    train_sampler: DistributedSampler | None = None,
    rank: int = 0,
    world_size: int = 1,
    max_grad_norm: float = 1.0,
    profile_batches: int = 0,
    log_fn: Any | None = None,
    encoder_only: bool = False,
) -> float:
    """Train for one epoch with DDP and profiling support.

    This is the enhanced version of train_epoch() with:
    - Distributed Data Parallel (DDP) support
    - Optional profiling of first N batches
    - Gradient accumulation
    - Frozen backbone optimization (memory saving when no LoRA)

    Args:
        model: AlphaGenome trunk model (may be DDP-wrapped).
        head: Output head module.
        train_loader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision.
        accumulation_steps: Number of batches to accumulate gradients over.
        frozen_backbone: If True, use torch.no_grad() for backbone (memory optimization).
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        train_sampler: DistributedSampler for DDP (set epoch for shuffling).
        rank: Process rank for DDP.
        world_size: Total number of processes.
        max_grad_norm: Maximum gradient norm for clipping.
        profile_batches: Number of batches to profile (0 to disable).
        log_fn: Optional function to call for step logging: log_fn(metrics_dict).
        encoder_only: If True, run only the CNN encoder (skip transformer and decoder)
            and pass the raw encoder output (B, S//128, 1536) to the head as resolution
            128. The backbone is always frozen in encoder-only mode. Requires the head
            to have been created with ``create_finetuning_head(..., encoder_only=True)``.

    Returns:
        Average training loss for the epoch (synchronized across ranks).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        is_main_process,
        reduce_tensor,
    )

    model.train()
    head.train()

    # Set epoch for distributed sampler (important for shuffling)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    total_loss = 0.0
    n_batches = 0

    # Profiling (only on rank 0)
    do_profile = profile_batches > 0 and is_main_process(rank)
    profile_stats = ProfilingStats() if do_profile else None

    if do_profile:
        print(f"\n*** PROFILING ENABLED for first {profile_batches} batches ***\n")

    # Only show progress bar on rank 0
    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    t_batch_start = time.perf_counter()
    running_loss = 0.0
    accumulated_batches = 0

    for batch_idx, (sequences, targets_dict) in enumerate(pbar):
        is_profiling = do_profile and batch_idx < profile_batches

        # --- Data loading time (time since last batch ended) ---
        if is_profiling and batch_idx > 0:
            _cuda_sync(device)
            t_data_load = time.perf_counter() - t_batch_start
            profile_stats.add("1_data_loading", t_data_load)

        # --- Transfer to GPU ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("2_to_device", time.perf_counter() - t0)

        # --- Forward pass ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        # When backbone is frozen (no LoRA), we can save memory by not building
        # the computation graph for the backbone forward pass.
        resolutions = tuple(resolution_weights.keys())

        if encoder_only:
            # Run only the CNN encoder; skip transformer, decoder, OutputEmbedders.
            # Backbone is always frozen in encoder-only mode.
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"].detach()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                predictions = head(
                    embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                )
        elif frozen_backbone:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

            # Detach embeddings to ensure no gradients flow back to backbone
            embeddings_dict = {}
            for res in resolution_weights:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    embeddings_dict[res] = outputs[emb_key].detach()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                predictions = head(
                    embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                )
        else:
            # LoRA enabled: gradients need to flow through backbone
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

                embeddings_dict = {}
                for res in resolution_weights:
                    emb_key = f"embeddings_{res}bp"
                    if emb_key in outputs:
                        embeddings_dict[res] = outputs[emb_key]

                predictions = head(
                    embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                )

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("3_forward", time.perf_counter() - t0)

        # --- Loss computation ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        loss = torch.tensor(0.0, device=device)
        loss_components: dict[str, float] = {}

        for res, weight in resolution_weights.items():
            if res not in predictions or res not in targets_dict:
                continue

            pred = predictions[res]
            targets = targets_dict[res].to(device)

            # Scale targets from experimental space to model space
            head_module = head.module if hasattr(head, "module") else head
            targets = head_module.scale(targets, organism_idx, resolution=res, channels_last=True)
            mask = torch.ones(pred.shape[0], 1, pred.shape[-1], dtype=torch.bool, device=device)

            # Compute multinomial loss
            current_seq_len = pred.shape[-2]
            multinomial_res = _compute_multinomial_resolution(
                current_seq_len, num_segments, min_segment_size
            )

            loss_dict = multinomial_loss(
                y_pred=pred,
                y_true=targets,
                mask=mask,
                multinomial_resolution=multinomial_res,
                positional_weight=positional_weight,
                count_weight=count_weight,
                channels_last=True,
            )

            res_loss = loss_dict["loss"] * weight
            loss = loss + res_loss
            loss_components[f"loss_{res}bp"] = res_loss.item()
            # Log raw (unweighted) losses for comparability across runs
            loss_components[f"loss_{res}bp_count"] = loss_dict["loss_total"].item()
            loss_components[f"loss_{res}bp_positional"] = loss_dict["loss_positional"].item()

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("4_loss", time.perf_counter() - t0)

        # --- Backward pass ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        scaled_loss.backward()

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("5_backward", time.perf_counter() - t0)

        # --- Optimizer step (only every accumulation_steps batches) ---
        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = batch_idx == len(train_loader) - 1

        if is_accumulation_step or is_last_batch:
            if is_profiling:
                _cuda_sync(device)
                t0 = time.perf_counter()

            # Get trainable parameters for gradient clipping
            trainable_params = [p for p in head.parameters() if p.requires_grad]
            trainable_params += [p for p in model.parameters() if p.requires_grad]

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if is_profiling:
                _cuda_sync(device)
                profile_stats.add("6_optimizer", time.perf_counter() - t0)

        # Update totals
        raw_loss = loss.item()
        total_loss += raw_loss
        n_batches += 1

        # Update running loss
        running_loss += raw_loss
        accumulated_batches += 1

        current_lr = scheduler.get_last_lr()[0]

        # Logging (only on rank 0)
        if is_main_process(rank) and batch_idx % log_every == 0:
            avg_running_loss = running_loss / accumulated_batches
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{raw_loss:.4f}",
                    "run_loss": f"{avg_running_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            if log_fn is not None:
                step_metrics = {
                    "batch": batch_idx,
                    "epoch": epoch,
                    "loss": raw_loss,
                    "running_loss": avg_running_loss,
                    "learning_rate": current_lr,
                    **loss_components,
                }
                log_fn(step_metrics)

            # Reset running loss after logging
            running_loss = 0.0
            accumulated_batches = 0

        # Print profiling report after profiling is done
        if do_profile and batch_idx == profile_batches - 1:
            print(profile_stats.report(profile_batches))

            # Estimate epoch time
            estimated_time = profile_stats.estimated_epoch_time(len(train_loader))
            print(f"\nESTIMATED EPOCH TIME: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")
            print(f"  Based on {profile_batches} profiled batches, {len(train_loader)} total batches")
            print()

        # Mark end of batch for next iteration's data loading measurement
        if is_profiling:
            _cuda_sync(device)
        t_batch_start = time.perf_counter()

    # Reduce loss across all processes
    avg_loss = total_loss / max(1, n_batches)
    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

    return avg_loss


@torch.no_grad()
def validate_ddp(
    model: nn.Module,
    head: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    count_weight: float,
    use_amp: bool = True,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    compute_pearson: bool = True,
    rank: int = 0,
    world_size: int = 1,
    encoder_only: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Validate the model with DDP support and Pearson R metrics.

    This is the enhanced version of validate() with:
    - Distributed Data Parallel (DDP) support with proper tensor gathering
    - Optional Pearson R computation (profile and count correlations)

    Args:
        model: AlphaGenome trunk model (may be DDP-wrapped).
        head: Output head module.
        val_loader: Validation data loader.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        use_amp: Whether to use automatic mixed precision.
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        compute_pearson: Whether to compute Pearson R metrics.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        encoder_only: If True, run only the CNN encoder and pass raw encoder output
            (B, S//128, 1536) to the head as resolution 128. Must match the setting
            used during training.

    Returns:
        Tuple of (avg_loss, metrics_dict) where metrics_dict contains:
        - Per-resolution losses (e.g., "1bp", "128bp")
        - Pearson R metrics if compute_pearson=True (profile_pearson_r_mean, count_pearson_r, etc.)
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        gather_tensors,
        is_main_process,
        reduce_tensor,
    )
    from alphagenome_pytorch.metrics import pearson_r, profile_pearson_r

    model.eval()
    head.eval()

    total_loss = 0.0
    n_batches = 0
    loss_by_resolution: dict[str, float] = defaultdict(float)

    # For Pearson R computation - accumulate across ALL batches
    accumulated_profile_r: dict[int, list[Tensor]] = defaultdict(list)
    accumulated_pred_counts: dict[int, list[Tensor]] = defaultdict(list)
    accumulated_true_counts: dict[int, list[Tensor]] = defaultdict(list)

    # Only show progress bar on rank 0
    if is_main_process(rank):
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader

    for sequences, targets_dict in pbar:
        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        resolutions = tuple(resolution_weights.keys())

        if encoder_only:
            outputs = model(sequences, organism_idx, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"]}
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = {}
            for res in resolution_weights:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    embeddings_dict[res] = outputs[emb_key]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # Get predictions in MODEL space for loss computation
            head_module = head.module if hasattr(head, "module") else head
            predictions_scaled = head(
                embeddings_dict, organism_idx, return_scaled=True, channels_last=True
            )

            # Get predictions in EXPERIMENTAL space for Pearson R
            if compute_pearson:
                predictions_unscaled = head(
                    embeddings_dict, organism_idx, return_scaled=False, channels_last=True
                )

        loss = torch.tensor(0.0, device=device)

        for res, weight in resolution_weights.items():
            if res not in predictions_scaled or res not in targets_dict:
                continue

            pred_scaled = predictions_scaled[res]
            targets = targets_dict[res].to(device)

            # Scale targets from experimental space to model space for loss
            targets_scaled = head_module.scale(
                targets, organism_idx, resolution=res, channels_last=True
            )
            mask = torch.ones(
                pred_scaled.shape[0], 1, pred_scaled.shape[-1], dtype=torch.bool, device=device
            )

            # Compute multinomial loss
            current_seq_len = pred_scaled.shape[-2]
            multinomial_res = _compute_multinomial_resolution(
                current_seq_len, num_segments, min_segment_size
            )

            loss_dict = multinomial_loss(
                y_pred=pred_scaled,
                y_true=targets_scaled,
                mask=mask,
                multinomial_resolution=multinomial_res,
                positional_weight=positional_weight,
                count_weight=count_weight,
                channels_last=True,
            )

            res_loss = loss_dict["loss"] * weight
            loss = loss + res_loss
            loss_by_resolution[f"{res}bp"] += res_loss.item()
            # Log raw (unweighted) losses for comparability across runs
            loss_by_resolution[f"{res}bp_count"] += loss_dict["loss_total"].item()
            loss_by_resolution[f"{res}bp_positional"] += loss_dict["loss_positional"].item()

            # Accumulate for Pearson R (in experimental space)
            if compute_pearson:
                pred_unscaled = predictions_unscaled[res]

                # Profile Pearson R: compute per-region correlation on-the-fly, store scalars
                batch_profile_r = profile_pearson_r(pred_unscaled, targets)  # (batch, tracks)
                accumulated_profile_r[res].append(batch_profile_r.float().cpu())

                # Count Pearson R: store total counts per region (tiny memory)
                accumulated_pred_counts[res].append(pred_unscaled.sum(dim=1).float().cpu())  # (batch, tracks)
                accumulated_true_counts[res].append(targets.sum(dim=1).float().cpu())

        total_loss += loss.item()
        n_batches += 1

    # Reduce across all processes
    avg_loss = total_loss / max(1, n_batches)
    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

    # Compute per-resolution loss metrics (synchronized across ranks)
    metrics: dict[str, Any] = {}
    for k, v in loss_by_resolution.items():
        res_avg = v / max(1, n_batches)
        if world_size > 1:
            res_tensor = torch.tensor(res_avg, device=device)
            res_tensor = reduce_tensor(res_tensor, world_size)
            metrics[k] = res_tensor.item()
        else:
            metrics[k] = res_avg

    # Compute Pearson R metrics (must gather across all DDP ranks)
    if compute_pearson:
        for res in resolution_weights.keys():
            # Profile Pearson R (from accumulated per-region correlations)
            if res in accumulated_profile_r and accumulated_profile_r[res]:
                all_profile_r = torch.cat(accumulated_profile_r[res], dim=0)  # (N_local, tracks)

                # Gather profile correlations from all ranks
                if world_size > 1:
                    all_profile_r = gather_tensors(all_profile_r, world_size, device)

                metrics[f"{res}bp_profile_pearson_r_mean"] = all_profile_r.mean().item()
                metrics[f"{res}bp_profile_pearson_r_std"] = all_profile_r.std().item()
                # Store full distribution for wandb histogram
                metrics[f"{res}bp_profile_pearson_r_values"] = all_profile_r.flatten().tolist()

            # Count Pearson R (from accumulated counts)
            if res in accumulated_pred_counts and accumulated_pred_counts[res]:
                all_pred_counts = torch.cat(accumulated_pred_counts[res], dim=0)  # (N_local, tracks)
                all_true_counts = torch.cat(accumulated_true_counts[res], dim=0)

                # Gather counts from all ranks
                if world_size > 1:
                    all_pred_counts = gather_tensors(all_pred_counts, world_size, device)
                    all_true_counts = gather_tensors(all_true_counts, world_size, device)

                if all_pred_counts.shape[0] > 1:
                    count_r = pearson_r(all_pred_counts, all_true_counts, dim=0)  # (tracks,)
                    metrics[f"{res}bp_count_pearson_r"] = count_r.mean().item()
                else:
                    metrics[f"{res}bp_count_pearson_r"] = float("nan")

    return avg_loss, metrics


def train_epoch_multihead(
    model: nn.Module,
    heads: dict[str, nn.Module],
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    modality_weights: dict[str, float],
    resolution_weights: dict[str, dict[int, float]],
    positional_weight: float,
    count_weight: float,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    frozen_backbone: bool = False,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    train_sampler: DistributedSampler | None = None,
    rank: int = 0,
    world_size: int = 1,
    max_grad_norm: float = 1.0,
    profile_batches: int = 0,
    log_fn: Any | None = None,
    encoder_only: bool = False,
) -> tuple[float, dict[str, float]]:
    """Train for one epoch with multiple modality heads.

    This extends train_epoch_ddp to support multi-modality training where
    each modality has its own head and weights.

    Args:
        model: AlphaGenome trunk model (may be DDP-wrapped).
        heads: Dict mapping modality name to output head module.
        train_loader: Training data loader (yields sequences, modality_targets dict).
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        modality_weights: Weight for each modality's loss (e.g., {"atac": 1.0, "rna_seq": 0.5}).
        resolution_weights: Per-modality resolution weights (e.g., {"atac": {1: 1.0, 128: 1.0}}).
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision.
        accumulation_steps: Number of batches to accumulate gradients over.
        frozen_backbone: If True, use torch.no_grad() for backbone.
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        train_sampler: DistributedSampler for DDP.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        max_grad_norm: Maximum gradient norm for clipping.
        profile_batches: Number of batches to profile.
        log_fn: Optional function for step logging.
        encoder_only: If True, run only the CNN encoder and pass raw encoder output
            (B, S//128, 1536) to all heads as resolution 128. Backbone is always
            frozen in encoder-only mode.

    Returns:
        Tuple of (avg_total_loss, per_modality_losses).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        is_main_process,
        reduce_tensor,
    )

    model.train()
    for head in heads.values():
        head.train()

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    total_loss_accum = 0.0
    modality_loss_accum: dict[str, float] = {m: 0.0 for m in heads}
    n_batches = 0

    # Profiling (only on rank 0)
    do_profile = profile_batches > 0 and is_main_process(rank)
    profile_stats = ProfilingStats() if do_profile else None

    if do_profile:
        print(f"\n*** PROFILING ENABLED for first {profile_batches} batches ***\n")

    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    t_batch_start = time.perf_counter()
    running_loss = 0.0
    accumulated_batches = 0

    for batch_idx, (sequences, modality_targets) in enumerate(pbar):
        is_profiling = do_profile and batch_idx < profile_batches

        if is_profiling and batch_idx > 0:
            _cuda_sync(device)
            t_data_load = time.perf_counter() - t_batch_start
            profile_stats.add("1_data_loading", t_data_load)

        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("2_to_device", time.perf_counter() - t0)

        # Forward through backbone
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        # Collect all needed resolutions across all modalities
        all_resolutions = set()
        for modality in heads:
            all_resolutions.update(resolution_weights.get(modality, {}).keys())
        resolutions = tuple(all_resolutions)

        if encoder_only:
            # Run only the CNN encoder; backbone is always frozen in encoder-only mode.
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"].detach()}
        elif frozen_backbone:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = {}
            for res in resolutions:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    embeddings_dict[res] = outputs[emb_key].detach()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = {}
            for res in resolutions:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    embeddings_dict[res] = outputs[emb_key]

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("3_forward_backbone", time.perf_counter() - t0)

        # Forward through each head and compute losses
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        loss = torch.tensor(0.0, device=device)
        loss_components: dict[str, float] = {}

        for modality, head in heads.items():
            if modality not in modality_targets:
                continue

            modality_weight = modality_weights.get(modality, 1.0)
            res_weights = resolution_weights.get(modality, {})
            targets_dict = modality_targets[modality]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                predictions = head(
                    embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                )

            modality_loss = torch.tensor(0.0, device=device)

            for res, weight in res_weights.items():
                if res not in predictions or res not in targets_dict:
                    continue

                pred = predictions[res]
                targets = targets_dict[res].to(device)

                head_module = head.module if hasattr(head, "module") else head
                targets = head_module.scale(
                    targets, organism_idx, resolution=res, channels_last=True
                )
                mask = torch.ones(pred.shape[0], 1, pred.shape[-1], dtype=torch.bool, device=device)

                current_seq_len = pred.shape[-2]
                multinomial_res = _compute_multinomial_resolution(
                    current_seq_len, num_segments, min_segment_size
                )

                loss_dict = multinomial_loss(
                    y_pred=pred,
                    y_true=targets,
                    mask=mask,
                    multinomial_resolution=multinomial_res,
                    positional_weight=positional_weight,
                    count_weight=count_weight,
                    channels_last=True,
                )

                res_loss = loss_dict["loss"] * weight
                modality_loss = modality_loss + res_loss
                loss_components[f"{modality}_loss_{res}bp"] = res_loss.item()
                loss_components[f"{modality}_loss_{res}bp_count"] = loss_dict["loss_total"].item()
                loss_components[f"{modality}_loss_{res}bp_positional"] = loss_dict["loss_positional"].item()

            weighted_modality_loss = modality_loss * modality_weight
            loss = loss + weighted_modality_loss
            loss_components[f"{modality}_loss"] = modality_loss.item()
            modality_loss_accum[modality] += modality_loss.item()

        scaled_loss = loss / accumulation_steps

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("4_heads_and_loss", time.perf_counter() - t0)

        # Backward
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        scaled_loss.backward()

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("5_backward", time.perf_counter() - t0)

        # Optimizer step
        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = batch_idx == len(train_loader) - 1

        if is_accumulation_step or is_last_batch:
            if is_profiling:
                _cuda_sync(device)
                t0 = time.perf_counter()

            trainable_params = []
            for head in heads.values():
                trainable_params.extend([p for p in head.parameters() if p.requires_grad])
            trainable_params.extend([p for p in model.parameters() if p.requires_grad])

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if is_profiling:
                _cuda_sync(device)
                profile_stats.add("6_optimizer", time.perf_counter() - t0)

        raw_loss = loss.item()
        total_loss_accum += raw_loss
        n_batches += 1

        running_loss += raw_loss
        accumulated_batches += 1

        current_lr = scheduler.get_last_lr()[0]

        if is_main_process(rank) and batch_idx % log_every == 0:
            avg_running_loss = running_loss / accumulated_batches
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{raw_loss:.4f}",
                    "run_loss": f"{avg_running_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            if log_fn is not None:
                step_metrics = {
                    "batch": batch_idx,
                    "epoch": epoch,
                    "loss": raw_loss,
                    "running_loss": avg_running_loss,
                    "learning_rate": current_lr,
                    **loss_components,
                }
                log_fn(step_metrics)

            running_loss = 0.0
            accumulated_batches = 0

        if do_profile and batch_idx == profile_batches - 1:
            print(profile_stats.report(profile_batches))
            estimated_time = profile_stats.estimated_epoch_time(len(train_loader))
            print(f"\nESTIMATED EPOCH TIME: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")
            print()

        if is_profiling:
            _cuda_sync(device)
        t_batch_start = time.perf_counter()

    # Reduce across processes
    avg_loss = total_loss_accum / max(1, n_batches)
    per_modality_loss = {m: v / max(1, n_batches) for m, v in modality_loss_accum.items()}

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

        for m in per_modality_loss:
            m_tensor = torch.tensor(per_modality_loss[m], device=device)
            m_tensor = reduce_tensor(m_tensor, world_size)
            per_modality_loss[m] = m_tensor.item()

    return avg_loss, per_modality_loss


@torch.no_grad()
def validate_multihead(
    model: nn.Module,
    heads: dict[str, nn.Module],
    val_loader: DataLoader,
    device: torch.device,
    modality_weights: dict[str, float],
    resolution_weights: dict[str, dict[int, float]],
    positional_weight: float,
    count_weight: float,
    use_amp: bool = True,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    compute_pearson: bool = True,
    rank: int = 0,
    world_size: int = 1,
    encoder_only: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Validate model with multiple modality heads.

    Args:
        model: AlphaGenome trunk model.
        heads: Dict mapping modality name to output head module.
        val_loader: Validation data loader.
        device: Torch device.
        modality_weights: Weight for each modality's loss.
        resolution_weights: Per-modality resolution weights.
        positional_weight: Weight for positional component.
        count_weight: Weight for count component.
        use_amp: Whether to use automatic mixed precision.
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        compute_pearson: Whether to compute Pearson R metrics.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        encoder_only: If True, run only the CNN encoder and pass raw encoder output
            (B, S//128, 1536) to all heads as resolution 128.

    Returns:
        Tuple of (avg_total_loss, metrics_dict).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        gather_tensors,
        is_main_process,
        reduce_tensor,
    )
    from alphagenome_pytorch.metrics import pearson_r, profile_pearson_r

    model.eval()
    for head in heads.values():
        head.eval()

    total_loss_accum = 0.0
    modality_loss_accum: dict[str, float] = {m: 0.0 for m in heads}
    n_batches = 0

    # For Pearson R - per modality and resolution
    accumulated_profile_r: dict[str, dict[int, list[Tensor]]] = {m: defaultdict(list) for m in heads}
    accumulated_pred_counts: dict[str, dict[int, list[Tensor]]] = {m: defaultdict(list) for m in heads}
    accumulated_true_counts: dict[str, dict[int, list[Tensor]]] = {m: defaultdict(list) for m in heads}

    if is_main_process(rank):
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader

    for sequences, modality_targets in pbar:
        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        # Collect all resolutions
        all_resolutions = set()
        for modality in heads:
            all_resolutions.update(resolution_weights.get(modality, {}).keys())
        resolutions = tuple(all_resolutions)

        if encoder_only:
            outputs = model(sequences, organism_idx, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"]}
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = {}
            for res in resolutions:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    embeddings_dict[res] = outputs[emb_key]

        loss = torch.tensor(0.0, device=device)

        for modality, head in heads.items():
            if modality not in modality_targets:
                continue

            modality_weight = modality_weights.get(modality, 1.0)
            res_weights = resolution_weights.get(modality, {})
            targets_dict = modality_targets[modality]

            head_module = head.module if hasattr(head, "module") else head

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                predictions_scaled = head(
                    embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                )
                if compute_pearson:
                    predictions_unscaled = head(
                        embeddings_dict, organism_idx, return_scaled=False, channels_last=True
                    )

            modality_loss = torch.tensor(0.0, device=device)

            for res, weight in res_weights.items():
                if res not in predictions_scaled or res not in targets_dict:
                    continue

                pred_scaled = predictions_scaled[res]
                targets = targets_dict[res].to(device)
                targets_scaled = head_module.scale(
                    targets, organism_idx, resolution=res, channels_last=True
                )
                mask = torch.ones(
                    pred_scaled.shape[0], 1, pred_scaled.shape[-1], dtype=torch.bool, device=device
                )

                current_seq_len = pred_scaled.shape[-2]
                multinomial_res = _compute_multinomial_resolution(
                    current_seq_len, num_segments, min_segment_size
                )

                loss_dict = multinomial_loss(
                    y_pred=pred_scaled,
                    y_true=targets_scaled,
                    mask=mask,
                    multinomial_resolution=multinomial_res,
                    positional_weight=positional_weight,
                    count_weight=count_weight,
                    channels_last=True,
                )

                res_loss = loss_dict["loss"] * weight
                modality_loss = modality_loss + res_loss

                # Accumulate for Pearson R
                if compute_pearson:
                    pred_unscaled = predictions_unscaled[res]
                    batch_profile_r = profile_pearson_r(pred_unscaled, targets)
                    accumulated_profile_r[modality][res].append(batch_profile_r.float().cpu())
                    accumulated_pred_counts[modality][res].append(pred_unscaled.sum(dim=1).float().cpu())
                    accumulated_true_counts[modality][res].append(targets.sum(dim=1).float().cpu())

            weighted_modality_loss = modality_loss * modality_weight
            loss = loss + weighted_modality_loss
            modality_loss_accum[modality] += modality_loss.item()

        total_loss_accum += loss.item()
        n_batches += 1

    # Reduce across processes
    avg_loss = total_loss_accum / max(1, n_batches)
    per_modality_loss = {m: v / max(1, n_batches) for m, v in modality_loss_accum.items()}

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

        for m in per_modality_loss:
            m_tensor = torch.tensor(per_modality_loss[m], device=device)
            m_tensor = reduce_tensor(m_tensor, world_size)
            per_modality_loss[m] = m_tensor.item()

    # Build metrics dict
    metrics: dict[str, Any] = {}
    for m, v in per_modality_loss.items():
        metrics[f"{m}_loss"] = v

    # Compute Pearson R
    if compute_pearson:
        for modality in heads:
            for res in resolution_weights.get(modality, {}).keys():
                if res in accumulated_profile_r[modality] and accumulated_profile_r[modality][res]:
                    all_profile_r = torch.cat(accumulated_profile_r[modality][res], dim=0)
                    if world_size > 1:
                        all_profile_r = gather_tensors(all_profile_r, world_size, device)
                    metrics[f"{modality}_{res}bp_profile_pearson_r_mean"] = all_profile_r.mean().item()
                    metrics[f"{modality}_{res}bp_profile_pearson_r_std"] = all_profile_r.std().item()
                    metrics[f"{modality}_{res}bp_profile_pearson_r_values"] = all_profile_r.flatten().tolist()

                if res in accumulated_pred_counts[modality] and accumulated_pred_counts[modality][res]:
                    all_pred_counts = torch.cat(accumulated_pred_counts[modality][res], dim=0)
                    all_true_counts = torch.cat(accumulated_true_counts[modality][res], dim=0)
                    if world_size > 1:
                        all_pred_counts = gather_tensors(all_pred_counts, world_size, device)
                        all_true_counts = gather_tensors(all_true_counts, world_size, device)
                    if all_pred_counts.shape[0] > 1:
                        count_r = pearson_r(all_pred_counts, all_true_counts, dim=0)
                        metrics[f"{modality}_{res}bp_count_pearson_r"] = count_r.mean().item()
                    else:
                        metrics[f"{modality}_{res}bp_count_pearson_r"] = float("nan")

    return avg_loss, metrics


__all__ = [
    "collate_genomic",
    "ModalityConfig",
    "MODALITY_CONFIGS",
    "create_lr_scheduler",
    "compute_finetuning_loss",
    "train_epoch",
    "validate",
    "save_checkpoint",
    # Enhanced versions with DDP support
    "ProfilingStats",
    "train_epoch_ddp",
    "validate_ddp",
    # Multi-head training
    "train_epoch_multihead",
    "validate_multihead",
]
