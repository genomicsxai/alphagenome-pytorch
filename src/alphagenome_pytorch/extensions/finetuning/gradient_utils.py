"""Gradient handling for externally evaluated fine-tuning heads."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn


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
