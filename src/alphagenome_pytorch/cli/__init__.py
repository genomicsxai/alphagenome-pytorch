"""AlphaGenome Torch CLI (``agt``).

Entry point:
    ``agt [--json] <command> [options]``
"""

from __future__ import annotations

from alphagenome_pytorch.cli._main import main

__all__ = ["main"]
