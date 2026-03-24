"""Checkpointing utilities for AlphaGenome fine-tuning.

Provides checkpoint save/load, discovery, and preemption handling for resumable training.
"""

from __future__ import annotations

import hashlib
import os
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from alphagenome_pytorch.extensions.finetuning.distributed import is_main_process

if TYPE_CHECKING:
    from alphagenome_pytorch.extensions.finetuning.transfer import TransferConfig


def atomic_torch_save(obj: Any, path: Path | str) -> None:
    """Save a PyTorch object atomically.

    Uses temp file + rename to prevent corrupted files from crashes or
    power failures during save. Rename is atomic on POSIX systems.

    Args:
        obj: Object to save (checkpoint dict, model state, etc.).
        path: Destination path.
    """
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(fd)  # Close the file descriptor, torch.save will open it
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)  # Atomic rename (works cross-platform)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_checkpoint(
    path: Path | str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    val_loss: float,
    track_names: list[str] | dict[str, list[str]],
    modality: str | list[str],
    resolutions: tuple[int, ...] | dict[str, tuple[int, ...]],
    scheduler: LRScheduler | None = None,
    best_val_loss: float | None = None,
    wandb_run_id: str | None = None,
    **extra_metadata: Any,
) -> None:
    """Save a training checkpoint atomically.

    The entire model state (trunk + all heads) is saved in model_state_dict.
    This works for all training modes (linear-probe, lora, full) and supports
    both single and multi-modality training.

    Uses atomic writes (temp file + rename) to prevent corrupted checkpoints
    from crashes or power failures during save.

    Args:
        path: Path to save checkpoint.
        epoch: Current epoch number.
        model: AlphaGenome model (trunk + heads).
        optimizer: Optimizer.
        val_loss: Validation loss at this checkpoint.
        track_names: Track names - either list (single modality) or dict (multi-modality).
        modality: Modality name(s) - either str or list of str.
        resolutions: Output resolutions - either tuple or dict mapping modality to resolutions.
        scheduler: Learning rate scheduler (optional).
        best_val_loss: Best validation loss seen so far (optional).
        wandb_run_id: W&B run ID for resume support (optional).
        **extra_metadata: Additional metadata to save.

    Example:
        >>> # Single modality
        >>> save_checkpoint(
        ...     path="checkpoint.pth",
        ...     epoch=5,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     val_loss=0.123,
        ...     track_names=["track1", "track2"],
        ...     modality="atac",
        ...     resolutions=(1, 128),
        ... )
        >>> # Multi-modality
        >>> save_checkpoint(
        ...     path="checkpoint.pth",
        ...     epoch=5,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     val_loss=0.123,
        ...     track_names={"atac": ["t1"], "rna_seq": ["t2"]},
        ...     modality=["atac", "rna_seq"],
        ...     resolutions={"atac": (1,), "rna_seq": (128,)},
        ... )
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "track_names": track_names,
        "modality": modality,
        "resolutions": resolutions,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss

    if wandb_run_id is not None:
        checkpoint["wandb_run_id"] = wandb_run_id

    checkpoint.update(extra_metadata)

    atomic_torch_save(checkpoint, path)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the most recent checkpoint in output_dir.

    Prefers ``checkpoint_preempt.pth`` (saved mid-epoch by the signal
    handler) over ``checkpoint_epoch*.pth`` when it is newer.

    Args:
        output_dir: Directory to search for checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.

    Example:
        >>> ckpt_path = find_latest_checkpoint(Path("output/run_001"))
        >>> if ckpt_path:
        ...     checkpoint = torch.load(ckpt_path)
    """
    preempt = output_dir / "checkpoint_preempt.pth"

    def _epoch_num(p: Path) -> int:
        return int(p.stem.replace("checkpoint_epoch", ""))

    epoch_ckpts = list(output_dir.glob("checkpoint_epoch*.pth"))
    epoch_ckpts.sort(key=_epoch_num)

    if not epoch_ckpts and not preempt.exists():
        return None

    # If preempt checkpoint exists, prefer it when it's newer than the
    # latest epoch checkpoint (it was saved *after* the last completed epoch).
    if preempt.exists():
        if not epoch_ckpts or preempt.stat().st_mtime >= epoch_ckpts[-1].stat().st_mtime:
            return preempt

    return epoch_ckpts[-1] if epoch_ckpts else None


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint.

    Loads the entire model state (trunk + all heads) from model_state_dict.
    This works for all training modes (linear-probe, lora, full) and supports
    both single and multi-modality training.

    Args:
        path: Path to checkpoint file.
        model: AlphaGenome model to load state into.
        optimizer: Optimizer to load state into.
        scheduler: Learning rate scheduler to load state into (optional).
        device: Device to map checkpoint to.

    Returns:
        Checkpoint dict with metadata (epoch, val_loss, best_val_loss, wandb_run_id, etc.).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load entire model state (trunk + heads)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class PreemptionHandler:
    """Handler for graceful preemption via signals (e.g., SIGUSR1 from SLURM).

    When the signal is received, sets a flag. The training loop should check
    `handler.preempted` and call `handler.save_and_exit()` to save a checkpoint
    before exiting gracefully.

    Note:
        The save function is NOT called inside the signal handler to avoid
        deadlocks. Signal handlers can interrupt the main thread at arbitrary
        points (including during I/O), so calling torch.save() there is unsafe.
        Instead, the training loop should check the flag and save explicitly.

    Attributes:
        preempted: Whether preemption signal was received.

    Example:
        >>> handler = PreemptionHandler(
        ...     save_fn=lambda: save_checkpoint(...),
        ...     rank=0,
        ...     world_size=1,
        ... )
        >>> handler.register()
        >>> for epoch in range(100):
        ...     if handler.preempted:
        ...         handler.save_and_exit()
        ...         break
        ...     # ... training loop ...
    """

    def __init__(
        self,
        save_fn: Callable[[], None] | None = None,
        rank: int = 0,
        world_size: int = 1,
        signal_num: int = signal.SIGUSR1,
    ) -> None:
        """Initialize the preemption handler.

        Args:
            save_fn: Function to call to save checkpoint (called from training loop, not signal handler).
            rank: Process rank for distributed training.
            world_size: Total number of processes.
            signal_num: Signal to handle (default: SIGUSR1, used by SLURM).
        """
        self.save_fn = save_fn
        self.rank = rank
        self.world_size = world_size
        self.signal_num = signal_num
        self.preempted = False
        self._original_handler = None

    def _handler(self, signum: int, frame) -> None:
        """Signal handler that sets preempted flag.

        Note: Does NOT perform I/O to avoid deadlocks. The training loop
        should check `self.preempted` and call `save_and_exit()` explicitly.
        """
        self.preempted = True

        if is_main_process(self.rank):
            print(f"\n{'='*60}")
            print(f"SIGNAL {signum} received — will save checkpoint and exit.")
            print(f"{'='*60}")

    def save_and_exit(self) -> None:
        """Save checkpoint and synchronize processes.

        Call this from the training loop when `preempted` is True.
        This is safe to call because it runs in the main thread, not
        inside a signal handler.
        """
        if is_main_process(self.rank):
            if self.save_fn is not None:
                print("Saving preemption checkpoint...")
                self.save_fn()
                print("Preemption checkpoint saved.")
            print("Training will exit.")

        # Synchronize all processes
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

    def register(self) -> None:
        """Register the signal handler."""
        self._original_handler = signal.signal(self.signal_num, self._handler)

    def unregister(self) -> None:
        """Restore the original signal handler."""
        if self._original_handler is not None:
            signal.signal(self.signal_num, self._original_handler)
            self._original_handler = None


def setup_preemption_handler(
    save_fn: Callable[[], None] | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> PreemptionHandler:
    """Set up and register a preemption handler.

    Convenience function that creates a PreemptionHandler and registers it.

    Args:
        save_fn: Function to call when signal is received (should save checkpoint).
        rank: Process rank for distributed training.
        world_size: Total number of processes.

    Returns:
        The registered PreemptionHandler instance.

    Example:
        >>> def save():
        ...     save_checkpoint(output_dir / "checkpoint_preempt.pth", ...)
        >>> handler = setup_preemption_handler(save, rank=0, world_size=1)
        >>> # In training loop:
        >>> if handler.preempted:
        ...     break
    """
    handler = PreemptionHandler(save_fn=save_fn, rank=rank, world_size=world_size)
    handler.register()
    return handler


# =============================================================================
# Delta Checkpoints - save only adapters + new heads
# =============================================================================

# Current delta checkpoint format version
DELTA_CHECKPOINT_VERSION = 1


_HEAD_PREFIXES = (
    "heads.",
    "contact_maps_head.",
    "splice_sites_classification_head.",
    "splice_sites_usage_head.",
    "splice_sites_junction_head.",
)


def _get_adapter_module_names(model: nn.Module) -> set[str]:
    """Return the set of module paths that are adapter wrappers."""
    from alphagenome_pytorch.extensions.finetuning.adapters import (
        HoulsbyBlockWrapper,
        HoulsbyWrapper,
        IA3,
        IA3_FF,
        Locon,
        LoRA,
    )

    adapter_types = (LoRA, Locon, IA3, IA3_FF, HoulsbyWrapper, HoulsbyBlockWrapper)
    return {name for name, module in model.named_modules() if isinstance(module, adapter_types)}


def _normalize_trunk_key(key: str, adapter_module_names: set[str]) -> str:
    """Normalize a state dict key by removing adapter wrapper path segments.

    Adapter wrappers store the original module as ``self.original_layer``
    (LoRA, Locon, IA3, HoulsbyWrapper) or ``self.block`` (HoulsbyBlockWrapper),
    which inserts an extra path segment into the state dict key. This function
    strips that segment so the key matches the bare-model form.

    Example:
        ``tower.blocks.0.mha.q_proj.original_layer.weight``
        → ``tower.blocks.0.mha.q_proj.weight``
    """
    for adapter_name in adapter_module_names:
        for segment in (".original_layer.", ".block."):
            prefix = adapter_name + segment
            if key.startswith(prefix):
                return adapter_name + "." + key[len(prefix):]
    return key


def _identify_adapter_params(model: nn.Module) -> set[str]:
    """Identify all adapter parameter names in model.

    Walks through model.named_modules() to find adapter wrappers and
    returns the set of parameter names belonging to adapters.

    Args:
        model: Model with adapters applied.

    Returns:
        Set of parameter names (e.g., "tower.blocks.0.mha.q_proj.lora_A.weight").
    """
    # Import adapter classes here to avoid circular imports
    from alphagenome_pytorch.extensions.finetuning.adapters import (
        HoulsbyBlockWrapper,
        HoulsbyWrapper,
        IA3,
        IA3_FF,
        Locon,
        LoRA,
    )

    adapter_params: set[str] = set()

    for name, module in model.named_modules():
        if isinstance(module, LoRA):
            adapter_params.add(f"{name}.lora_A.weight")
            adapter_params.add(f"{name}.lora_B.weight")
        elif isinstance(module, Locon):
            adapter_params.add(f"{name}.locon_down.weight")
            adapter_params.add(f"{name}.locon_up.weight")
        elif isinstance(module, (IA3, IA3_FF)):
            adapter_params.add(f"{name}.scale")
        elif isinstance(module, (HoulsbyWrapper, HoulsbyBlockWrapper)):
            adapter_params.add(f"{name}.adapter.down_project.weight")
            adapter_params.add(f"{name}.adapter.down_project.bias")
            adapter_params.add(f"{name}.adapter.up_project.weight")
            adapter_params.add(f"{name}.adapter.up_project.bias")

    return adapter_params


def split_model_state_dict(
    model: nn.Module,
    new_head_names: list[str] | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Split model state dict into trunk, adapter, and head parts.

    Purely structural classification — does not depend on ``requires_grad``
    or other training state, so the trunk dict (and its hash) is identical
    whether the model is bare or adapted.

    Trunk keys are normalized so that adapter wrapper paths
    (e.g. ``.original_layer.``) are stripped.

    Args:
        model: AlphaGenome model (bare or with adapters/heads).
        new_head_names: Head names to extract. If None, all ``heads.*``
            params go into the heads dict.

    Returns:
        ``(trunk, adapters, heads)`` — three dicts of name → tensor.
        Norm layer params are included in *trunk*.

    Raises:
        ValueError: If a name in *new_head_names* has no matching params.
    """
    adapter_param_names = _identify_adapter_params(model)
    adapter_module_names = _get_adapter_module_names(model)

    if new_head_names is not None:
        head_prefixes = tuple(f"heads.{n}." for n in new_head_names)
    else:
        head_prefixes = _HEAD_PREFIXES

    state_dict = model.state_dict()

    trunk: dict[str, torch.Tensor] = {}
    adapters: dict[str, torch.Tensor] = {}
    heads: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if key in adapter_param_names:
            adapters[key] = tensor
        elif key.startswith(head_prefixes):
            heads[key] = tensor
        elif key.startswith(_HEAD_PREFIXES):
            # Head params not in new_head_names — skip (not trunk)
            continue
        else:
            normalized = _normalize_trunk_key(key, adapter_module_names)
            trunk[normalized] = tensor

    # Validate requested heads were found
    if new_head_names is not None:
        for head_name in new_head_names:
            prefix = f"heads.{head_name}."
            if not any(k.startswith(prefix) for k in heads):
                raise ValueError(
                    f"Head '{head_name}' from config.new_heads not found in model.heads. "
                    f"Available heads: {list(model.heads.keys())}"
                )

    return trunk, adapters, heads


# ---------------------------------------------------------------------------
# Convenience accessors (thin wrappers over split_model_state_dict)
# ---------------------------------------------------------------------------


def get_trunk_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract trunk parameters with normalized keys.

    See :func:`split_model_state_dict` for details.
    """
    trunk, _, _ = split_model_state_dict(model)
    return trunk


def get_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only adapter parameters from model state_dict.

    Example:
        >>> model = prepare_for_transfer(model, TransferConfig(mode='lora'))
        >>> adapter_weights = get_adapter_state_dict(model)
        >>> len(adapter_weights)  # Only LoRA weights
        36
    """
    _, adapters, _ = split_model_state_dict(model)
    return adapters


def get_new_head_state_dict(
    model: nn.Module,
    new_head_names: list[str],
) -> dict[str, torch.Tensor]:
    """Extract parameters for newly added heads.

    Raises:
        ValueError: If a head name is not found in model.heads.

    Example:
        >>> head_weights = get_new_head_state_dict(model, ['my_atac'])
    """
    _, _, heads = split_model_state_dict(model, new_head_names=new_head_names)
    return heads


def get_norm_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract trainable normalization layer parameters from model state_dict.

    Only returns norm params with ``requires_grad=True`` that are not
    already classified as adapter or head params. This is independent of
    :func:`split_model_state_dict` (which is purely structural).
    """
    norm_class_patterns = ("norm", "layernorm", "batchnorm", "rmsnorm", "rmsbatchnorm")
    adapter_param_names = _identify_adapter_params(model)
    state_dict = model.state_dict()
    norm_params: dict[str, torch.Tensor] = {}

    for mod_name, module in model.named_modules():
        class_name = type(module).__name__.lower()
        is_norm = isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm))
        if not is_norm:
            is_norm = any(p in class_name for p in norm_class_patterns)
        if not is_norm:
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{mod_name}.{param_name}" if mod_name else param_name
            if full_name in adapter_param_names:
                continue
            if full_name.startswith(_HEAD_PREFIXES):
                continue
            if full_name in state_dict:
                norm_params[full_name] = state_dict[full_name]

    return norm_params


def _hash_state_dict_structure(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash the structural info (sorted keys, shapes, dtypes) of a state dict."""
    items = [
        (key, tuple(state_dict[key].shape), str(state_dict[key].dtype))
        for key in sorted(state_dict.keys())
    ]
    content = str(items).encode("utf-8")
    return f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"


def compute_base_model_hash(model: nn.Module) -> str:
    """Compute a hash of trunk structure for compatibility verification.

    Hashes the sorted list of (key, shape, dtype) tuples from the trunk
    state dict (excluding adapters and heads, with normalized key paths).
    Produces the same hash whether called on a bare or adapted model.

    Args:
        model: AlphaGenome model (with or without adapters/heads applied).

    Returns:
        A string like ``"sha256:abc123..."``.

    Example:
        >>> model = AlphaGenome()
        >>> hash1 = compute_base_model_hash(model)
        >>> model = prepare_for_transfer(model, config)
        >>> hash2 = compute_base_model_hash(model)
        >>> assert hash1 == hash2  # same trunk → same hash
    """
    return _hash_state_dict_structure(get_trunk_state_dict(model))


def save_delta_checkpoint(
    path: Path | str,
    model: nn.Module,
    config: "TransferConfig",
    base_model_hash: str | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    **metadata: Any,
) -> None:
    """Save a delta checkpoint containing only adapter and new head weights.

    Delta checkpoints are much smaller than full checkpoints (~5-10MB vs ~1GB)
    because they only store the finetuning-specific weights. They are
    self-contained with the serialized TransferConfig, so you only need the
    base model weights + delta file to reconstruct the finetuned model.

    IMPORTANT: Call this BEFORE merge_adapters() - adapters must still be
    separate modules to identify their parameters.

    Args:
        path: Path to save checkpoint (recommended: use .delta.pth extension).
        model: Model with adapters and new heads (NOT merged).
        config: TransferConfig used to prepare the model.
        base_model_hash: Optional pre-computed hash identifying the base model.
            If None, will be computed from the trunk-only state dict (excluding
            adapters and new heads, with normalized key paths).
        optimizer: Optional optimizer to save state for training resume.
        scheduler: Optional LR scheduler to save state for training resume.
        **metadata: Additional metadata (epoch, val_loss, track_names, etc.)

    Raises:
        ValueError: If adapters appear to be merged (no adapter params found
            when adapter modes are in config).

    Example:
        >>> config = TransferConfig(mode='lora', new_heads={'my_atac': {...}})
        >>> model = prepare_for_transfer(model, config)
        >>> # Train...
        >>> save_delta_checkpoint(
        ...     "best_model.delta.pth",
        ...     model,
        ...     config,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     epoch=10,
        ...     val_loss=0.05,
        ... )
    """
    from alphagenome_pytorch.extensions.finetuning.transfer import (
        transfer_config_to_dict,
    )

    # Split state dict structurally (trunk / adapters / heads)
    new_head_names = list(config.new_heads.keys())
    trunk_state_dict, adapter_state_dict, head_state_dict = (
        split_model_state_dict(model, new_head_names=new_head_names)
    )

    # Validate adapters are present when expected
    modes = config.mode if isinstance(config.mode, list) else [config.mode]
    adapter_modes = {"lora", "locon", "ia3", "houlsby"}
    expects_adapters = any(m in adapter_modes for m in modes)

    if expects_adapters and not adapter_state_dict:
        raise ValueError(
            "No adapter parameters found but config specifies adapter mode(s): "
            f"{[m for m in modes if m in adapter_modes]}. "
            "Did you call merge_adapters()? Delta checkpoints must be saved "
            "BEFORE merging adapters."
        )

    # Get trainable norm params separately (requires_grad-based, not structural)
    norm_state_dict = get_norm_state_dict(model)

    # Compute hash from the already-extracted trunk if not provided
    if base_model_hash is None:
        base_model_hash = _hash_state_dict_structure(trunk_state_dict)

    # Build checkpoint
    checkpoint = {
        "delta_checkpoint_version": DELTA_CHECKPOINT_VERSION,
        "transfer_config": transfer_config_to_dict(config),
        "adapter_state_dict": adapter_state_dict,
        "head_state_dict": head_state_dict,
        "norm_state_dict": norm_state_dict,
        "base_model_hash": base_model_hash,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        },
    }

    # Optional training state for resume
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    atomic_torch_save(checkpoint, path)


def is_delta_checkpoint(path: Path | str) -> bool:
    """Check if a checkpoint file is a delta checkpoint.

    Args:
        path: Path to checkpoint file.

    Returns:
        True if the checkpoint is a delta checkpoint, False otherwise.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return "delta_checkpoint_version" in checkpoint


def load_delta_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    verify_hash: bool = True,
    strict: bool = True,
    skip_prepare: bool = False,
) -> tuple["TransferConfig", dict[str, Any]]:
    """Load a delta checkpoint and apply to base model.

    This function:
    1. Loads the delta checkpoint
    2. Verifies base model compatibility (if verify_hash=True)
    3. Applies the TransferConfig to the model (calls prepare_for_transfer)
       unless skip_prepare=True
    4. Loads adapter and head weights
    5. Optionally loads optimizer and scheduler state for training resume

    Args:
        path: Path to delta checkpoint.
        model: Base AlphaGenome model (without adapters/new heads).
            Should have trunk weights loaded but heads can be present or absent.
        optimizer: Optional optimizer to load state into for training resume.
        scheduler: Optional LR scheduler to load state into for training resume.
        verify_hash: If True, verify base model structure matches checkpoint's
            hash. Set to False to skip verification (useful when model structure
            changed slightly but weights are compatible).
        strict: If True, raise error on missing keys. If False, warn only.
        skip_prepare: If True, skip calling prepare_for_transfer. Use this when
            resuming training where the model already has adapters/heads set up.

    Returns:
        Tuple of (TransferConfig, metadata_dict). The metadata dict includes
        'has_optimizer_state' and 'has_scheduler_state' booleans.

    Raises:
        ValueError: If verify_hash=True and model structure doesn't match.
        ValueError: If checkpoint version is unsupported.
        RuntimeError: If strict=True and required keys are missing.

    Example:
        >>> # Load base model
        >>> model = AlphaGenome()
        >>> model = load_trunk(model, "pretrained.pth", exclude_heads=True)
        >>>
        >>> # Apply delta checkpoint (inference only)
        >>> config, metadata = load_delta_checkpoint("finetuned.delta.pth", model)
        >>> print(f"Loaded from epoch {metadata.get('epoch')}")
        >>>
        >>> # Or with optimizer for training resume
        >>> config, metadata = load_delta_checkpoint(
        ...     "finetuned.delta.pth", model, optimizer=optimizer, scheduler=scheduler
        ... )
    """
    from alphagenome_pytorch.extensions.finetuning.transfer import (
        prepare_for_transfer,
        transfer_config_from_dict,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Version check
    version = checkpoint.get("delta_checkpoint_version", 0)
    if version > DELTA_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported delta checkpoint version {version}. "
            f"This version of alphagenome-pytorch supports up to version "
            f"{DELTA_CHECKPOINT_VERSION}. Please upgrade."
        )

    # Verify trunk hash before modifying the model
    if verify_hash:
        saved_hash = checkpoint.get("base_model_hash")
        if saved_hash is not None:
            current_hash = compute_base_model_hash(model)
            if current_hash != saved_hash:
                raise ValueError(
                    f"Base model structure mismatch. "
                    f"Expected hash '{saved_hash}', got '{current_hash}'. "
                    "Ensure you're using the correct pretrained weights, or "
                    "set verify_hash=False to skip this check."
                )

    # Reconstruct config
    config = transfer_config_from_dict(checkpoint["transfer_config"])

    # Apply config to model (adds adapters and new heads)
    if not skip_prepare:
        model = prepare_for_transfer(model, config)

    # Load adapter, head, and norm weights in one shot
    adapter_state_dict = checkpoint.get("adapter_state_dict", {})
    head_state_dict = checkpoint.get("head_state_dict", {})
    norm_state_dict = checkpoint.get("norm_state_dict", {})

    delta_state = {**adapter_state_dict, **head_state_dict, **norm_state_dict}
    if delta_state:
        result = model.load_state_dict(delta_state, strict=False)
        # unexpected_keys = delta keys not found in the model
        if result.unexpected_keys:
            msg = f"Delta keys not found in model: {result.unexpected_keys[:5]}..."
            if strict:
                raise RuntimeError(msg + " Set strict=False to ignore.")
            else:
                print(f"Warning: {msg}")

    # Load optimizer state if provided
    has_optimizer_state = "optimizer_state_dict" in checkpoint
    if optimizer is not None and has_optimizer_state:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    has_scheduler_state = "scheduler_state_dict" in checkpoint
    if scheduler is not None and has_scheduler_state:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    metadata = checkpoint.get("metadata", {})
    metadata["has_optimizer_state"] = has_optimizer_state
    metadata["has_scheduler_state"] = has_scheduler_state
    return config, metadata


# =============================================================================
# Export Functions - clean weights for sharing/distribution
# =============================================================================


def export_model_weights(
    model: nn.Module,
    path: Path | str,
    format: str = "safetensors",
) -> None:
    """Export full model weights for inference/sharing.

    Exports the complete model state_dict (trunk + heads + merged adapters).
    Call merge_adapters() first if you want adapters folded into base weights.

    Args:
        model: Model to export.
        path: Output path. Extension is added based on format if not present.
        format: 'safetensors' or 'pth'.

    Example:
        >>> from alphagenome_pytorch.extensions.finetuning import merge_adapters
        >>> model = merge_adapters(model)  # Fold LoRA into base weights
        >>> export_model_weights(model, "model_weights.safetensors")
    """
    path = Path(path)
    state_dict = model.state_dict()

    if format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )
        if not path.suffix:
            path = path.with_suffix(".safetensors")
        save_file({k: v.cpu() for k, v in state_dict.items()}, path)
    elif format == "pth":
        if not path.suffix:
            path = path.with_suffix(".pth")
        torch.save(state_dict, path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'safetensors' or 'pth'.")


def export_delta_weights(
    model: nn.Module,
    config: "TransferConfig",
    path: Path | str,
    format: str = "safetensors",
) -> None:
    """Export only delta weights (adapters + new heads) for sharing.

    Exports a minimal weight file containing only the finetuning-specific
    weights plus the TransferConfig (so recipients can reconstruct the model).
    Does not include optimizer state or other training metadata.

    Args:
        model: Model with adapters and new heads (NOT merged).
        config: TransferConfig used for training (to identify new heads).
        path: Output path. Extension is added based on format if not present.
        format: 'safetensors' or 'pth'.

    Example:
        >>> # Export just the LoRA weights + head for sharing
        >>> export_delta_weights(model, config, "my_lora_weights.safetensors")
        >>>
        >>> # Recipient can load with:
        >>> config = load_delta_config("my_lora_weights.safetensors")
        >>> model = prepare_for_transfer(base_model, config)
        >>> load_delta_weights(model, "my_lora_weights.safetensors")
        >>>
        >>> # Or use the high-level API:
        >>> model = AlphaGenome.from_delta("my_lora_weights.safetensors", "base.pth")
    """
    from alphagenome_pytorch.extensions.finetuning.transfer import transfer_config_to_dict
    import json

    path = Path(path)

    # Get adapter, head, and trainable norm weights
    adapter_weights = get_adapter_state_dict(model)
    head_weights = get_new_head_state_dict(model, list(config.new_heads.keys()))
    norm_weights = get_norm_state_dict(model)
    weights = {**adapter_weights, **head_weights, **norm_weights}

    # Serialize config
    config_dict = transfer_config_to_dict(config)

    if format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )
        if not path.suffix:
            path = path.with_suffix(".safetensors")
        # safetensors metadata must be str -> str
        metadata = {"transfer_config": json.dumps(config_dict)}
        save_file({k: v.cpu() for k, v in weights.items()}, path, metadata=metadata)
    elif format == "pth":
        if not path.suffix:
            path = path.with_suffix(".pth")
        # For pth, we wrap in a dict with config
        torch.save({"weights": weights, "transfer_config": config_dict}, path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'safetensors' or 'pth'.")


def load_delta_config(
    path: Path | str,
) -> "TransferConfig":
    """Read the TransferConfig from a delta weights file without loading weights.

    Use this to inspect how a delta weights file was trained (adapter type,
    heads, etc.) before setting up your model.

    Args:
        path: Path to delta weights file (.safetensors or .pth).

    Returns:
        The TransferConfig embedded in the file.

    Example:
        >>> config = load_delta_config("colleague_lora.safetensors")
        >>> print(config.mode, config.lora_rank, list(config.new_heads.keys()))
        >>> model = prepare_for_transfer(base_model, config)
        >>> load_delta_weights(model, "colleague_lora.safetensors")
    """
    from alphagenome_pytorch.extensions.finetuning.transfer import transfer_config_from_dict
    import json

    path = Path(path)

    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
            if not metadata or "transfer_config" not in metadata:
                raise ValueError(
                    f"Delta weights file {path} is missing transfer_config metadata"
                )
            config_dict = json.loads(metadata["transfer_config"])
    else:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "transfer_config" not in data:
            raise ValueError(
                f"Delta weights file {path} is missing transfer_config key"
            )
        config_dict = data["transfer_config"]

    return transfer_config_from_dict(config_dict)


def load_delta_weights(
    model: nn.Module,
    path: Path | str,
    strict: bool = True,
) -> "TransferConfig":
    """Load delta weights into a model that already has adapters/heads set up.

    The model must already be configured with the correct adapters and heads
    (typically via ``prepare_for_transfer()``). Use ``load_delta_config()``
    first to read the config if needed.

    Args:
        model: Model with adapters and new heads already configured.
        path: Path to delta weights file (.safetensors or .pth).
        strict: If True, raise error on missing keys. If False, warn only.

    Returns:
        The TransferConfig from the exported file.

    Example:
        >>> config = load_delta_config("colleague_lora.safetensors")
        >>> model = prepare_for_transfer(base_model, config)
        >>> load_delta_weights(model, "colleague_lora.safetensors")
    """
    import json

    path = Path(path)

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )
        weights = load_file(path)
    else:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "weights" not in data:
            raise ValueError(
                f"Delta weights file {path} is missing 'weights' key"
            )
        weights = data["weights"]

    # Read config via load_delta_config (avoids duplicating parsing logic)
    config = load_delta_config(path)

    # Load weights into model
    current_state = model.state_dict()
    missing = []
    for key, value in weights.items():
        if key in current_state:
            current_state[key] = value
        else:
            missing.append(key)

    model.load_state_dict(current_state, strict=False)

    if missing:
        msg = f"Missing keys when loading delta weights: {missing[:5]}..."
        if strict:
            raise RuntimeError(msg + " Set strict=False to ignore.")
        else:
            print(f"Warning: {msg}")

    return config


__all__ = [
    # Full checkpoints
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    # Preemption
    "PreemptionHandler",
    "setup_preemption_handler",
    # Delta checkpoints (with training state)
    "save_delta_checkpoint",
    "load_delta_checkpoint",
    "is_delta_checkpoint",
    "DELTA_CHECKPOINT_VERSION",
    # Export (clean weights for sharing)
    "export_model_weights",
    "export_delta_weights",
    "load_delta_config",
    "load_delta_weights",
    # Utilities
    "split_model_state_dict",
    "get_trunk_state_dict",
    "get_adapter_state_dict",
    "get_new_head_state_dict",
    "get_norm_state_dict",
    "compute_base_model_hash",
]
