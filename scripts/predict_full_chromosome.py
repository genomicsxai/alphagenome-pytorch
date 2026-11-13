#!/usr/bin/env python3
"""Generate full-chromosome predictions as BigWig files — thin shim.

The implementation lives in the package, on the same code path as ``agt``:

    alphagenome_pytorch.cli.predict                            — flags + dispatch
    alphagenome_pytorch.extensions.inference.full_chromosome   — tiling + writers

This script stays as the entry point existing docs and commands already use.
Every flag it accepted, ``agt predict`` accepts, so the two are equivalent:

    python scripts/predict_full_chromosome.py --model m.pth --fasta hg38.fa \\
        --output preds/ --head atac --chromosomes chr1
    agt predict --model m.pth --fasta hg38.fa --output preds/ --head atac \\
        --chromosomes chr1

One wrinkle is preserved below. Omitting --chromosomes has always meant
"chr1-22,chrX" for this script, while ``agt predict`` insists on an explicit
input selector so that a genome-scale run is never kicked off by an incomplete
command. The historical default is therefore passed through explicitly, keeping
this script's behaviour unchanged.
"""

from __future__ import annotations

import sys

from alphagenome_pytorch.cli._main import main
from alphagenome_pytorch.extensions.inference.full_chromosome import DEFAULT_CHROMOSOMES

#: Flags by which ``agt predict`` chooses an input mode.
_INPUT_SELECTORS = ("--chromosomes", "--locus", "--bed", "--sequences")


def to_predict_argv(argv: list[str]) -> list[str]:
    """Map this script's arguments onto ``agt predict``.

    Restores the historical chromosome default when the caller named no input,
    so ``--model ... --fasta ... --head atac`` keeps meaning chr1-22,chrX.
    """
    has_selector = any(
        arg == selector or arg.startswith(f"{selector}=")
        for arg in argv
        for selector in _INPUT_SELECTORS
    )
    if not has_selector:
        argv = [*argv, "--chromosomes", ",".join(DEFAULT_CHROMOSOMES)]
    return ["predict", *argv]


if __name__ == "__main__":
    sys.exit(main(to_predict_argv(sys.argv[1:])))
