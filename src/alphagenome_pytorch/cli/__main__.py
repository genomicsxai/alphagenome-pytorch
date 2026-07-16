"""``python -m alphagenome_pytorch.cli`` — same entry point as ``agt``.

torchrun needs a module or script target, so multi-GPU runs go through here:

    torchrun --nproc_per_node=2 -m alphagenome_pytorch.cli finetune \\
        --sequence-parallel --mode lora ...
"""

from __future__ import annotations

import sys

from alphagenome_pytorch.cli._main import main

if __name__ == "__main__":
    sys.exit(main())
