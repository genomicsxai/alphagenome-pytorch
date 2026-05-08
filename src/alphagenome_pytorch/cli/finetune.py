"""agt finetune — training and finetuning wrapper.

Forwards arguments to the finetune script's argparse.
"""

from __future__ import annotations

import argparse
import sys

from alphagenome_pytorch.cli._deps import require_extra


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "finetune",
        help="Training and finetuning (linear-probe, LoRA, full, encoder-only)",
        description="Training and finetuning — supports linear probing, LoRA, full finetuning, "
        "and encoder-only modes. Forwards all arguments to the finetune training script.",
    )

    # Accept all remaining args to forward to the finetune script
    p.add_argument(
        "finetune_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the finetune training script",
    )


def run(args: argparse.Namespace) -> int:
    require_extra("finetuning", "finetune")

    # Forward to the finetune script
    from scripts.finetune import parse_args, main as finetune_main

    # Parse only the forwarded args; pass an empty list when none were
    # provided so the finetune parser does not fall back to the outer
    # CLI's sys.argv (which would contain "finetune" and fail).
    ft_args = parse_args(args.finetune_args if args.finetune_args else [])
    return finetune_main(ft_args) or 0
