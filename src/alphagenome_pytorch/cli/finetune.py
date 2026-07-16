"""agt finetune — training and finetuning.

The flags are declared by
:func:`alphagenome_pytorch.extensions.finetuning.args.add_finetune_arguments` —
the same function ``scripts/finetune.py`` builds its parser from — so
``agt finetune`` and ``python scripts/finetune.py`` accept identical options.
"""

from __future__ import annotations

import argparse
import sys

from alphagenome_pytorch.cli._deps import require_extra
from alphagenome_pytorch.extensions.finetuning.args import add_finetune_arguments


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "finetune",
        help="Training and finetuning (linear-probe, LoRA, full, encoder-only)",
        description="Training and finetuning — supports linear probing, LoRA, Locon, "
        "full finetuning, and encoder-only modes. Equivalent to running "
        "scripts/finetune.py directly. For multi-GPU, use "
        "'torchrun -m alphagenome_pytorch.cli finetune ...'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_finetune_arguments(p)
    # postprocess_args reports failures against the parser the user invoked, so
    # errors read as 'agt finetune: error: ...' rather than naming the script.
    p.set_defaults(_finetune_parser=p)


def run(args: argparse.Namespace) -> int:
    require_extra("finetuning", "finetune")

    from alphagenome_pytorch.extensions.finetuning.args import build_parser, postprocess_args
    from alphagenome_pytorch.extensions.finetuning.runner import main as finetune_main

    parser = getattr(args, "_finetune_parser", None) or build_parser()
    # Which flags were passed explicitly decides what overrides --config, so the
    # raw tokens are needed here. _argv is recorded by the root parser and falls
    # back to sys.argv for direct calls; the leading subcommand name is ignored
    # either way, since only '--'-prefixed tokens are inspected.
    tokens = getattr(args, "_argv", None)
    if tokens is None:
        tokens = sys.argv[1:]

    postprocess_args(args, parser, list(tokens))
    return finetune_main(args) or 0
