"""Dependency gating for CLI commands.

Each optional extra maps to a set of probe modules. If any probe fails to
import, ``require_extra`` raises ``MissingExtraError`` with an actionable
message; the CLI's top-level handler formats it (text or JSON) via
``emit_error``.
"""

from __future__ import annotations

import importlib

# Mapping: extra name -> list of modules to probe
_EXTRA_PROBES: dict[str, list[str]] = {
    "inference": ["pyBigWig", "pyfaidx", "tqdm"],
    "finetuning": ["pyBigWig", "pandas", "tqdm", "pyfaidx"],
    "scoring": ["pyfaidx", "pandas", "tqdm"],
    "serving": ["grpc", "pandas", "pyfaidx", "alphagenome"],
    "jax": ["jax", "orbax.checkpoint"],
}


class MissingExtraError(Exception):
    """Raised when an optional extra's dependencies are not installed."""

    def __init__(self, extra_name: str, command_name: str, missing: list[str]) -> None:
        self.extra_name = extra_name
        self.command_name = command_name
        self.missing = missing
        super().__init__(
            f"'agt {command_name}' requires additional dependencies "
            f"({', '.join(missing)}). "
            f"Install them with: pip install alphagenome-pytorch[{extra_name}]"
        )


def require_extra(extra_name: str, command_name: str) -> None:
    """Check that *extra_name* dependencies are importable.

    Raises ``MissingExtraError`` if any probe module fails to import.
    """
    probes = _EXTRA_PROBES.get(extra_name, [])
    missing: list[str] = []
    for mod in probes:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        raise MissingExtraError(extra_name, command_name, missing)
