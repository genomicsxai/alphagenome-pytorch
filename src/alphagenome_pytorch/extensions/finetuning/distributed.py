"""Distributed training utilities for AlphaGenome fine-tuning.

Provides helpers for DDP setup, tensor reduction/gathering, and rank-aware operations.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import Tensor


def resolve_device(
    requested: str, *, local_rank: int = 0, requires_cuda: bool = False
) -> torch.device:
    """Resolve a fine-tuning device, using the rank for unindexed CUDA requests."""
    if requested == "auto":
        if not requires_cuda and not torch.cuda.is_available():
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        requested = "cuda"

    invalid_device = f"Invalid fine-tuning device {requested!r}. Use auto, cpu, mps, cuda, or cuda:N."
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(invalid_device) from exc
    if device.type not in ("cpu", "cuda", "mps") or (
        device.type != "cuda" and device.index is not None
    ):
        raise ValueError(invalid_device)

    if requires_cuda and device != torch.device("cuda"):
        raise ValueError(
            "NCCL fine-tuning requires --device auto or cuda. "
            "LOCAL_RANK selects the CUDA device for each process."
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable for fine-tuning.")
        index = local_rank if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device cuda:{index} is not visible to PyTorch.")
        return torch.device("cuda", index)

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Choose --device cpu or an available accelerator.")
    return device


def setup_distributed(requested_device: str = "auto") -> tuple[int, int, int, torch.device]:
    """Resolve the device and initialize NCCL when launched via torchrun."""
    initialized = dist.is_initialized()
    launch_nccl = not initialized and "RANK" in os.environ
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif launch_nccl:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0

    device = resolve_device(
        requested_device,
        local_rank=local_rank,
        requires_cuda=launch_nccl or (initialized and dist.get_backend() == "nccl"),
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if launch_nccl:
        dist.init_process_group(backend="nccl")

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def print_rank0(msg: str, rank: int) -> None:
    """Print only on rank 0."""
    if is_main_process(rank):
        print(msg)


def reduce_tensor(tensor: Tensor, world_size: int) -> Tensor:
    """All-reduce tensor across all processes and average.

    Args:
        tensor: Tensor to reduce (will be cloned, not modified in-place).
        world_size: Number of processes.

    Returns:
        Averaged tensor (same on all ranks after all_reduce).
    """
    if world_size == 1:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_tensors(tensor: Tensor, world_size: int, device: torch.device) -> Tensor:
    """Gather tensors from all processes along the first dimension.

    Handles uneven splits (different tensor sizes per rank) by padding
    to the maximum size and then trimming.

    Args:
        tensor: Local tensor to gather (must be on the correct device).
        world_size: Number of processes.
        device: Device tensors are on.

    Returns:
        Concatenated tensor from all ranks along dim=0.
        All ranks receive the same gathered result.

    Example:
        >>> # On rank 0: tensor shape (10, 5)
        >>> # On rank 1: tensor shape (8, 5)
        >>> gathered = gather_tensors(tensor, world_size=2, device=device)
        >>> gathered.shape  # (18, 5) on both ranks
    """
    if world_size == 1:
        return tensor

    # Move to device if needed
    tensor = tensor.to(device)

    # Get local sizes from all ranks (they may differ due to uneven data splits)
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)

    # Pad tensor to max size for uniform all_gather
    if tensor.shape[0] < max_size:
        pad_shape = (max_size - tensor.shape[0],) + tensor.shape[1:]
        tensor = torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=device)], dim=0)

    # Gather from all ranks
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    # Remove padding and concatenate
    result = []
    for i, size in enumerate(all_sizes):
        result.append(gathered[i][:size])

    return torch.cat(result, dim=0)


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj, src: int = 0):
    """Broadcast an object from src rank to all ranks.

    Args:
        obj: Object to broadcast (only used on src rank).
        src: Source rank.

    Returns:
        The broadcasted object (same on all ranks).
    """
    if not dist.is_initialized():
        return obj

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


__all__ = [
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "print_rank0",
    "reduce_tensor",
    "gather_tensors",
    "barrier",
    "broadcast_object",
]
