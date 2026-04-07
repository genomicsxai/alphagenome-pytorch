"""Root parser, dispatch, and error handling for the ``agt`` CLI."""

from __future__ import annotations

import argparse
import sys

from alphagenome_pytorch.cli._output import emit_error


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="agt",
        description="AlphaGenome Torch CLI — model inspection, inference, and training.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        default=False,
        help="Machine-readable JSON output on stdout",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Register subcommands — each module has register(subparsers)
    from alphagenome_pytorch.cli import info, predict, score, convert, preprocess, serve, finetune

    info.register(subparsers)
    predict.register(subparsers)
    finetune.register(subparsers)
    score.register(subparsers)
    convert.register(subparsers)
    preprocess.register(subparsers)
    serve.register(subparsers)

    return parser


# Mapping from command name to module.run function (filled lazily)
_COMMAND_RUNNERS: dict[str, object] | None = None


def _get_runners() -> dict[str, object]:
    global _COMMAND_RUNNERS
    if _COMMAND_RUNNERS is None:
        from alphagenome_pytorch.cli import (
            info, predict, score, convert, preprocess, serve, finetune,
        )
        _COMMAND_RUNNERS = {
            "info": info.run,
            "predict": predict.run,
            "finetune": finetune.run,
            "score": score.run,
            "convert": convert.run,
            "preprocess": preprocess.run,
            "serve": serve.run,
        }
    return _COMMAND_RUNNERS


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``agt`` CLI.

    Returns an integer exit code (0 = success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    json_mode = getattr(args, "json_output", False)

    try:
        runners = _get_runners()
        runner = runners.get(args.command)
        if runner is None:
            parser.print_help()
            return 1
        return runner(args)
    except SystemExit as exc:
        # Re-raise sys.exit() from dep gating etc.
        raise
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        emit_error(exc, json_mode=json_mode)
        return 1
