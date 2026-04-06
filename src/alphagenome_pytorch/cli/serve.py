"""agt serve — reserved for future release."""

from __future__ import annotations

import argparse
import sys


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``serve`` subcommand."""
    subparsers.add_parser(
        "serve",
        help="Serve the model via REST or gRPC (not yet implemented)",
        description="Serve the model via REST or gRPC.",
    )


def run(args: argparse.Namespace) -> int:
    """Run the ``serve`` command — currently a stub."""
    print(
        "Error: 'agt serve' is not yet implemented.\n"
        "Follow https://github.com/google-deepmind/alphagenome for updates.",
        file=sys.stderr,
    )
    return 1
