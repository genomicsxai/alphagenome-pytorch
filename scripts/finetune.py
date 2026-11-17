#!/usr/bin/env python
"""Unified AlphaGenome training script — thin shim over the packaged implementation.

The implementation now lives in the package, so it works from a wheel install
too (no repo checkout, no sys.path games):

    alphagenome_pytorch.extensions.finetuning.args    — the flags
    alphagenome_pytorch.extensions.finetuning.runner  — the training code

This script stays as the entry point existing commands and docs already use.
Each invocation below is equivalent to its ``agt`` form:

    python scripts/finetune.py --mode lora ...
    agt finetune --mode lora ...

    torchrun --nproc_per_node=4 scripts/finetune.py --mode lora ...
    torchrun --nproc_per_node=4 -m alphagenome_pytorch.cli finetune --mode lora ...

Both parse the same flags and run the same code; see ``agt finetune --help``.
"""

from alphagenome_pytorch.extensions.finetuning.runner import main

if __name__ == "__main__":
    main()
