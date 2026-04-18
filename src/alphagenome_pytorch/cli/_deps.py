"""Dependency gating for CLI commands.

Each optional extra maps to a set of probe modules. If any probe fails to
import, the user gets an actionable ``pip install`` message and the process
exits with code 1.
"""

from __future__ import annotations

import importlib
import sys

# Mapping: extra name -> list of modules to probe
_EXTRA_PROBES: dict[str, list[str]] = {
    "inference": ["pyBigWig", "pyfaidx", "tqdm"],
    "finetuning": ["pyBigWig", "pandas", "tqdm", "pyfaidx"],
    "scoring": ["pyfaidx", "pandas", "tqdm"],
    "serving": ["grpc", "pandas", "pyfaidx", "alphagenome"],
    "jax": ["jax", "orbax.checkpoint"],
}


def require_extra(extra_name: str, command_name: str) -> None:
    """Check that *extra_name* dependencies are importable.

    If not, print an actionable message and ``sys.exit(1)``.
    """
    probes = _EXTRA_PROBES.get(extra_name, [])
    missing: list[str] = []
    for mod in probes:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print(
            f"Error: 'agt {command_name}' requires additional dependencies.\n"
            f"Install them with: pip install alphagenome-pytorch[{extra_name}]",
            file=sys.stderr,
        )
        sys.exit(1)
