#!/usr/bin/env python
"""Convert BigWig files to memory-mapped numpy arrays — thin shim.

The implementation lives in the package, so it also works from a wheel install:

    alphagenome_pytorch.extensions.finetuning.preprocessing

This script stays as the entry point existing docs and commands already use.
Each invocation below is equivalent to its ``agt`` form:

    python scripts/convert_bigwigs_to_mmap.py --bigwig *.bw --output-dir mmap/ --workers 8
    agt preprocess bigwig-to-mmap --input *.bw --output mmap/ --workers 8

(The flags are spelled differently: --bigwig/--output-dir here, --input/--output
under agt. Both run the same code.)
"""

from alphagenome_pytorch.extensions.finetuning.preprocessing import main

if __name__ == "__main__":
    main()
