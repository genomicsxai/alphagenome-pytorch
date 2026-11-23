"""Distributed training utilities for AlphaGenome fine-tuning.

Provides helpers for DDP setup, tensor reduction/gathering, gradient
synchronization, and rank-aware operations.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import Tensor, nn


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize distributed training environment.

    Auto-detects if launched via torchrun/torch.distributed.run.

    Returns:
        Tuple of (rank, world_size, local_rank, device).
        Returns (0, 1, 0, cuda:0 or cpu) if not in distributed mode.

    Example:
        >>> rank, world_size, local_rank, device = setup_distributed()
        >>> if world_size > 1:
        ...     model = DDP(model, device_ids=[local_rank])
    """
    if not dist.is_initialized():
        if "RANK" in os.environ:
            # Launched with torchrun or torch.distributed.run
            dist.init_process_group(backend="nccl")
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            # Single GPU mode
            rank = 0
            world_size = 1
            local_rank = 0
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

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


def unique_trainable_parameters(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    """Keep trainable parameters once by identity, preserving iteration order.

    A head can be referenced both directly and as a registered model child.
    Passing those aliases to in-place gradient clipping is not safe.
    """
    seen: set[int] = set()
    result: list[nn.Parameter] = []
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


@torch.no_grad()
def average_gradients_across_ranks(parameters: Iterable[nn.Parameter]) -> None:
    """Average dense gradients when the frozen backbone bypasses DDP backward.

    All ranks must supply the same parameters in the same order. Missing local
    gradients count as zero; globally unused parameters retain ``grad=None``
    so AdamW neither decays them nor advances their optimizer state. Call once
    at the optimizer boundary, after accumulation and before gradient clipping.
    This is data-parallel averaging, not sequence-parallel gradient summation.
    """
    parameters = unique_trainable_parameters(parameters)
    if not parameters or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    active = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.int32,
        device=parameters[0].device,
    )
    dist.all_reduce(active, op=dist.ReduceOp.SUM)
    for parameter, count in zip(parameters, active.tolist()):
        if not count:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradient = parameter.grad.contiguous()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(dist.get_world_size())
        if gradient.data_ptr() != parameter.grad.data_ptr():
            parameter.grad.copy_(gradient)
