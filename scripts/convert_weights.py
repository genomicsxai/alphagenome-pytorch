"""Convert JAX AlphaGenome checkpoint to PyTorch format — thin shim.

The implementation lives in the package, so it also works from a wheel install:

    alphagenome_pytorch.jax_compat.convert

This script stays as the entry point existing docs and commands already use:

    python scripts/convert_weights.py /path/to/jax/checkpoint --output model.pth
    agt convert --input /path/to/jax/checkpoint --output model.pth

Note the two spell the checkpoint differently: this script takes it as a
positional argument, ``agt convert`` takes ``--input``. Both run the same code.
"""

from alphagenome_pytorch.jax_compat.convert import main

if __name__ == "__main__":
    main()
