"""agt convert — convert JAX checkpoint to PyTorch format.

Handles both the pretrained JAX AlphaGenome checkpoint (orbax dir -> full state dict)
and alphagenome_ft finetuned checkpoint dirs (``config.json`` + ``checkpoint/``,
auto-detected -> ``.delta.pth`` for ``load_finetuned_model``).
"""

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
        description=(
            "Convert JAX AlphaGenome checkpoint to PyTorch format. An alphagenome_ft "
            "finetuned checkpoint directory (config.json + checkpoint/) is detected "
            "automatically and converted to a delta checkpoint (.delta.pth) that "
            "loads on top of the converted pretrained weights of the same fold."
        ),
    )

    p.add_argument("--input", required=True, help="Path to JAX checkpoint directory")
    p.add_argument("--output", default=None, help="Output file path (.pth or .safetensors)")
    p.add_argument("--safetensors", action="store_true",
                    help="Save as safetensors format")

    ft = p.add_argument_group("alphagenome_ft finetuned checkpoints")
    ft.add_argument("--organism", default=None,
                    help="Organism to extract (human|mouse) when a head has several "
                         "organism slots (default: the checkpoint's single organism)")
    ft.add_argument("--targets-config", default=None,
                    help="alphagenome_ft targets YAML/JSON with per-target nonzero_mean; "
                         "for checkpoints saved without head_metadata")
    ft.add_argument("--track-means", action="append", default=None, metavar="HEAD=V1,V2,...",
                    help="Per-track nonzero means for a track head (repeatable); "
                         "overrides head_metadata/--targets-config")
    ft.add_argument("--base-weights", default=None,
                    help="Converted pretrained PyTorch weights to record the exact base "
                         "weights hash in the delta (optional)")


def run(args: argparse.Namespace) -> int:
    require_extra("jax", "convert")

    json_mode = getattr(args, "json_output", False)

    if not Path(args.input).exists():
        raise FileNotFoundError(f"JAX checkpoint not found: {args.input}")

    from alphagenome_pytorch.jax_compat.finetuned import is_alphagenome_ft_checkpoint

    if is_alphagenome_ft_checkpoint(args.input):
        return _run_finetuned(args, json_mode)

    # Determine output path
    output_path = args.output
    use_safetensors = args.safetensors
    if output_path is None:
        output_path = "alphagenome_pt.safetensors" if use_safetensors else "alphagenome_pt.pth"
    elif output_path.endswith(".safetensors"):
        use_safetensors = True

    from alphagenome_pytorch.jax_compat.convert import convert, save_weights

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


def _run_finetuned(args: argparse.Namespace, json_mode: bool) -> int:
    """alphagenome_ft finetuned checkpoint dir -> PyTorch delta checkpoint."""
    from alphagenome_pytorch.jax_compat.finetuned import convert_finetuned_checkpoint

    output_path = args.output or "alphagenome_ft_converted.delta.pth"
    if args.safetensors or output_path.endswith(".safetensors"):
        raise ValueError(
            "alphagenome_ft checkpoints convert to a delta checkpoint; use a .delta.pth output."
        )
    conversion = convert_finetuned_checkpoint(
        args.input,
        output_path,
        organism=args.organism,
        targets_config=args.targets_config,
        track_means=args.track_means,
        base_weights=args.base_weights,
    )
    config = conversion.transfer_config
    if json_mode:
        emit_json({
            "output": output_path,
            "format": "delta",
            "mode": config.mode,
            "heads": list(config.new_heads),
            "organism": conversion.metadata["organism"],
            "params_mapped": len(conversion.state_dict),
        })
    else:
        print(
            f"Done! Saved {config.mode} delta checkpoint ({', '.join(config.new_heads)}; "
            f"organism {conversion.metadata['organism']}) to {output_path}"
        )
    return 0
