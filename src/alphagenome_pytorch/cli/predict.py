"""agt predict — full-chromosome inference writing BigWig files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alphagenome_pytorch.cli._deps import require_extra
from alphagenome_pytorch.cli._output import emit_json, emit_text


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "predict",
        help="Full-chromosome inference to BigWig",
        description="Run the model across chromosomes and write predictions to BigWig files.",
    )

    p.add_argument("--model", required=True, help="Path to model weights (.pth)")
    p.add_argument("--fasta", required=True, help="Path to reference genome FASTA")
    p.add_argument("--output", required=True, help="Output directory for BigWig files")
    p.add_argument("--head", required=True, help="Prediction head (e.g. atac, dnase)")

    p.add_argument("--tracks", type=str, default=None,
                    help="Track indices (comma-separated). Default: all")
    p.add_argument("--track-names", type=str, default=None,
                    help="Names for output tracks (comma-separated)")
    p.add_argument("--resolution", type=int, default=128, choices=[1, 128],
                    help="Output resolution in bp (default: 128)")
    p.add_argument("--crop-bp", type=int, default=0,
                    help="Base pairs to crop from each edge")
    p.add_argument("--batch-size", type=int, default=4,
                    help="Batch size for inference")
    p.add_argument("--window-size", type=int, default=131072,
                    help="Model input window size (default: 131072)")
    p.add_argument("--chromosomes", type=str, default=None,
                    help="Chromosomes (comma-separated, e.g. 'chr1,chr2')")
    p.add_argument("--organism", type=int, default=0, choices=[0, 1],
                    help="Organism: 0=human, 1=mouse")
    p.add_argument("--device", type=str, default="cuda", help="PyTorch device")
    p.add_argument("--dtype-policy", type=str, default="full_float32",
                    choices=["full_float32", "mixed_precision"],
                    help="Dtype policy")
    p.add_argument("--compile", action="store_true",
                    help="Use torch.compile for faster inference")

    # Finetuned model options
    ft = p.add_argument_group("Finetuned model (optional)")
    ft.add_argument("--checkpoint", type=str, default=None,
                    help="Path to finetuned checkpoint")
    ft.add_argument("--transfer-config", type=str, default=None,
                    help="Path to TransferConfig JSON file")
    ft.add_argument("--no-merge-adapters", action="store_true",
                    help="Keep adapter modules separate instead of merging")

    p.add_argument("--quiet", action="store_true", help="Suppress progress bars")


def run(args: argparse.Namespace) -> int:
    require_extra("inference", "predict")

    json_mode = getattr(args, "json_output", False)

    import torch
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.extensions.inference import (
        TilingConfig,
        predict_full_chromosomes_to_bigwig,
    )

    # Validate paths
    for label, path in [("Model", args.model), ("FASTA", args.fasta)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    # Parse track indices
    track_indices = None
    if args.tracks:
        track_indices = [int(t.strip()) for t in args.tracks.split(",")]

    track_names = None
    if args.track_names:
        track_names = [t.strip() for t in args.track_names.split(",")]

    chromosomes = None
    if args.chromosomes:
        chromosomes = [c.strip() for c in args.chromosomes.split(",")]

    # Configure dtype
    dtype_policy = (
        DtypePolicy.mixed_precision()
        if args.dtype_policy == "mixed_precision"
        else DtypePolicy.full_float32()
    )

    # Load model
    if args.checkpoint:
        from alphagenome_pytorch.extensions.finetuning.checkpointing import load_finetuned_model

        ext_config = None
        if args.transfer_config:
            import json as json_mod
            from alphagenome_pytorch.extensions.finetuning.transfer import transfer_config_from_dict
            with open(args.transfer_config) as f:
                ext_config = transfer_config_from_dict(json_mod.load(f))

        if not json_mode:
            print(f"Loading finetuned model...")
            print(f"  Base: {args.model}")
            print(f"  Checkpoint: {args.checkpoint}")

        model, meta = load_finetuned_model(
            checkpoint_path=args.checkpoint,
            pretrained_weights=args.model,
            device=args.device,
            dtype_policy=dtype_policy,
            transfer_config=ext_config,
            merge=not args.no_merge_adapters,
        )

        if track_names is None and meta.get("track_names"):
            ckpt_names = meta["track_names"]
            track_names = ckpt_names.get(args.head) if isinstance(ckpt_names, dict) else ckpt_names
    else:
        if not json_mode:
            print(f"Loading model from {args.model}...")
        model = AlphaGenome.from_pretrained(args.model, device=args.device, dtype_policy=dtype_policy)

    model.eval()

    if args.head not in model.heads:
        available = list(model.heads.keys())
        raise ValueError(f"Head '{args.head}' not found. Available: {available}")

    if args.compile:
        if not json_mode:
            print("Compiling model...")
        model = torch.compile(model)

    config = TilingConfig(
        window_size=args.window_size,
        crop_bp=args.crop_bp,
        resolution=args.resolution,
        batch_size=args.batch_size,
    )

    results = predict_full_chromosomes_to_bigwig(
        model=model,
        fasta_path=args.fasta,
        output_dir=args.output,
        head=args.head,
        chromosomes=chromosomes,
        config=config,
        track_indices=track_indices,
        track_names=track_names,
        organism_index=args.organism,
        device=args.device,
        show_progress=not args.quiet and not json_mode,
    )

    if json_mode:
        output_files = []
        for chrom, paths in results.items():
            for p in paths:
                output_files.append({
                    "path": str(p),
                    "head": args.head,
                    "chromosome": chrom,
                    "resolution_bp": args.resolution,
                })
        emit_json({"output_files": output_files, "warnings": []})
    else:
        total = sum(len(ps) for ps in results.values())
        print(f"\nDone! Wrote {total} BigWig file(s) to {args.output}")

    return 0
