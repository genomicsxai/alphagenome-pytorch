"""agt convert — convert JAX checkpoint to PyTorch format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alphagenome_pytorch.cli._deps import require_extra
from alphagenome_pytorch.cli._output import emit_json, emit_text


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "convert",
        help="Convert JAX checkpoint to PyTorch format",
        description="Convert JAX AlphaGenome checkpoint to PyTorch format.",
    )

    p.add_argument("--input", required=True, help="Path to JAX checkpoint directory")
    p.add_argument("--output", default=None, help="Output file path (.pth or .safetensors)")
    p.add_argument("--safetensors", action="store_true",
                    help="Save as safetensors format")


def run(args: argparse.Namespace) -> int:
    require_extra("jax", "convert")

    json_mode = getattr(args, "json_output", False)

    if not Path(args.input).exists():
        raise FileNotFoundError(f"JAX checkpoint not found: {args.input}")

    # Determine output path
    output_path = args.output
    use_safetensors = args.safetensors
    if output_path is None:
        output_path = "alphagenome_pt.safetensors" if use_safetensors else "alphagenome_pt.pth"
    elif output_path.endswith(".safetensors"):
        use_safetensors = True

    # Import from script module
    from scripts.convert_weights import convert, save_weights

    state_dict = convert(args.input)
    save_weights(state_dict, output_path, use_safetensors=use_safetensors)

    if json_mode:
        # Count heads
        head_names = set()
        has_track_means = False
        for key in state_dict:
            if key.startswith("heads."):
                parts = key.split(".")
                if len(parts) >= 2:
                    head_names.add(parts[1])
            if key.endswith(".track_means"):
                has_track_means = True

        emit_json({
            "output": output_path,
            "format": "safetensors" if use_safetensors else "pth",
            "params_mapped": len(state_dict),
            "params_total": len(state_dict),
            "heads": sorted(head_names),
            "track_means_included": has_track_means,
        })
    else:
        print(f"Done! Saved to {output_path}")

    return 0
