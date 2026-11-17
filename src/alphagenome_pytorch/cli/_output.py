"""Output formatting for the agt CLI.

Enforces the JSON contract: --json produces structured JSON on stdout,
errors produce JSON on stderr with nonzero exit code.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO


def emit_json(data: Any, *, file: TextIO = sys.stdout) -> None:
    """Emit pretty-printed JSON to *file* and flush."""
    json.dump(data, file, indent=2, default=str)
    file.write("\n")
    file.flush()


def emit_jsonl(data: Any, *, file: TextIO = sys.stdout) -> None:
    """Emit a single-line JSON object (JSONL) and flush."""
    json.dump(data, file, default=str, separators=(",", ":"))
    file.write("\n")
    file.flush()


def emit_error(exc: BaseException, *, json_mode: bool) -> None:
    """Emit an error. JSON mode writes structured JSON to stderr."""
    if json_mode:
        payload = {
            "error": type(exc).__name__,
            "message": str(exc),
        }
        json.dump(payload, sys.stderr, default=str)
        sys.stderr.write("\n")
        sys.stderr.flush()
    else:
        print(f"Error: {exc}", file=sys.stderr)


def emit_text(text: str, *, file: TextIO = sys.stdout) -> None:
    """Emit plain text and flush."""
    file.write(text)
    if not text.endswith("\n"):
        file.write("\n")
    file.flush()
