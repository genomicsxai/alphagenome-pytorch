#!/usr/bin/env python3
"""Generate full-chromosome predictions as BigWig files.

This script tiles across chromosomes, runs model inference, and writes
predictions to BigWig format.

Examples:
    # Basic usage - predict ATAC track 0 for chr1
    python scripts/predict_full_chromosome.py \\
        --model model.pth \\
        --fasta hg38.fa \\
        --output predictions/ \\
        --head atac \\
        --tracks 0 \\
        --chromosomes chr1

    # Full genome at 128bp resolution (faster)
    python scripts/predict_full_chromosome.py \\
        --model model.pth \\
        --fasta hg38.fa \\
        --output predictions/ \\
        --head atac \\
        --resolution 128 \\
        --batch-size 8

    # With center cropping to reduce edge artifacts
    python scripts/predict_full_chromosome.py \\
        --model model.pth \\
        --fasta hg38.fa \\
        --output predictions/ \\
        --head atac \\
        --crop-bp 32768 \\
        --resolution 128

    # 1bp resolution (slower, requires decoder)
    python scripts/predict_full_chromosome.py \\
        --model model.pth \\
        --fasta hg38.fa \\
        --output predictions/ \\
        --head atac \\
        --resolution 1 \\
        --batch-size 2

    # Finetuned model with delta checkpoint
    python scripts/predict_full_chromosome.py \\
        --model pretrained.pth \\
        --checkpoint best_model.delta.pth \\
        --fasta hg38.fa \\
        --output predictions/ \\
        --head my_atac \\
        --chromosomes chr21

    # Finetuned model with full checkpoint + external transfer config
    python scripts/predict_full_chromosome.py \\
        --model pretrained.pth \\
        --checkpoint best_model.pth \\
        --transfer-config transfer_config.json \\
        --fasta hg38.fa \\
        --output predictions/ \\
        --head my_atac \\
        --chromosomes chr21
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate full-chromosome predictions as BigWig files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model weights (.pth file)",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to reference genome FASTA file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for BigWig files",
    )
    parser.add_argument(
        "--head",
        required=True,
        help="Prediction head to use (e.g., 'atac', 'dnase', or a custom finetuned head name)",
    )

    # Track selection
    parser.add_argument(
        "--tracks",
        type=str,
        default=None,
        help="Track indices to output (comma-separated, e.g., '0,1,2'). Default: all tracks",
    )
    parser.add_argument(
        "--track-names",
        type=str,
        default=None,
        help="Names for output tracks (comma-separated). Default: track_0, track_1, ...",
    )

    # Tiling configuration
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        choices=[1, 128],
        help="Output resolution in bp. 128 is faster (default), 1 requires decoder",
    )
    parser.add_argument(
        "--crop-bp",
        type=int,
        default=0,
        help="Base pairs to crop from each edge (default: 0, no overlap). "
             "Set to e.g. 32768 to keep only center ~50%% of each window",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4). Increase for faster processing",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=131072,
        help="Model input window size (default: 131072)",
    )

    # Region selection
    parser.add_argument(
        "--chromosomes",
        type=str,
        default=None,
        help="Chromosomes to predict (comma-separated, e.g., 'chr1,chr2'). "
             "Default: chr1-22,chrX",
    )

    # Model configuration
    parser.add_argument(
        "--organism",
        type=int,
        default=0,
        choices=[0, 1],
        help="Organism index: 0=human (default), 1=mouse",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (default: cuda)",
    )
    parser.add_argument(
        "--dtype-policy",
        type=str,
        default="full_float32",
        choices=["full_float32", "mixed_precision"],
        help="Dtype policy: full_float32 (default, works everywhere) or "
             "mixed_precision (bfloat16 compute, ~halves GPU memory, requires Ampere+ GPU)",
    )

    # Performance options
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster inference (first batch is slower due to compilation)",
    )

    # Finetuned model options
    finetune = parser.add_argument_group("Finetuned model (optional)")
    finetune.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to finetuned checkpoint (.pth or .delta.pth). "
             "When set, --model is used as the base pretrained weights.",
    )
    finetune.add_argument(
        "--transfer-config",
        type=str,
        default=None,
        help="Path to TransferConfig JSON file. Required for full checkpoints "
             "that don't embed their config (e.g., older Locon/Houlsby checkpoints).",
    )
    finetune.add_argument(
        "--no-merge-adapters",
        action="store_true",
        help="Keep adapter modules separate instead of merging into base weights. "
             "By default, mergeable adapters (LoRA, IA3) are merged for faster inference.",
    )

    # Output options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars",
    )

    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"Error: FASTA file not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)

    if args.checkpoint and not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    if args.transfer_config and not Path(args.transfer_config).exists():
        print(f"Error: Transfer config not found: {args.transfer_config}", file=sys.stderr)
        sys.exit(1)

    # Parse track indices
    track_indices = None
    if args.tracks:
        track_indices = [int(t.strip()) for t in args.tracks.split(",")]

    # Parse track names
    track_names = None
    if args.track_names:
        track_names = [t.strip() for t in args.track_names.split(",")]

    # Parse chromosomes
    chromosomes = None
    if args.chromosomes:
        chromosomes = [c.strip() for c in args.chromosomes.split(",")]

    # Import here to avoid slow imports when just showing help
    import torch
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.extensions.inference import (
        TilingConfig,
        predict_full_chromosomes_to_bigwig,
    )

    # Configure dtype policy
    if args.dtype_policy == "mixed_precision":
        dtype_policy = DtypePolicy.mixed_precision()
    else:
        dtype_policy = DtypePolicy.full_float32()

    # Load model
    if args.checkpoint:
        # Finetuned model: --model is base weights, --checkpoint is finetuned
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            load_finetuned_model,
        )

        ext_config = None
        if args.transfer_config:
            import json
            from alphagenome_pytorch.extensions.finetuning.transfer import (
                transfer_config_from_dict,
            )
            with open(args.transfer_config) as f:
                ext_config = transfer_config_from_dict(json.load(f))

        print(f"Loading finetuned model...")
        print(f"  Base weights: {model_path}")
        print(f"  Checkpoint: {args.checkpoint}")
        model, meta = load_finetuned_model(
            checkpoint_path=args.checkpoint,
            pretrained_weights=str(model_path),
            device=args.device,
            dtype_policy=dtype_policy,
            transfer_config=ext_config,
            merge=not args.no_merge_adapters,
        )
        print(f"  Epoch: {meta.get('epoch')}, val_loss: {meta.get('val_loss')}")
        print(f"  Available heads: {meta.get('head_names')}")

        # Auto-populate track names from checkpoint metadata
        if track_names is None and meta.get("track_names"):
            ckpt_track_names = meta["track_names"]
            if isinstance(ckpt_track_names, dict):
                track_names = ckpt_track_names.get(args.head)
            else:
                track_names = ckpt_track_names
    else:
        # Standard pretrained model (existing behavior)
        print(f"Loading model from {model_path}...")
        model = AlphaGenome.from_pretrained(
            model_path,
            device=args.device,
            dtype_policy=dtype_policy,
        )

    model.eval()

    # Validate head exists
    if args.head not in model.heads:
        available = list(model.heads.keys())
        print(
            f"Error: Head '{args.head}' not found. "
            f"Available heads: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Configure tiling
    config = TilingConfig(
        window_size=args.window_size,
        crop_bp=args.crop_bp,
        resolution=args.resolution,
        batch_size=args.batch_size,
    )

    print(f"\nTiling configuration:")
    print(f"  Window size: {config.window_size:,} bp")
    print(f"  Crop: {config.crop_bp:,} bp from each edge")
    print(f"  Effective size: {config.effective_size:,} bp per window")
    print(f"  Step size: {config.step_size:,} bp")
    print(f"  Resolution: {config.resolution} bp")
    print(f"  Batch size: {config.batch_size}")

    # Run prediction
    print(f"\nPredicting head '{args.head}'...")

    results = predict_full_chromosomes_to_bigwig(
        model=model,
        fasta_path=str(fasta_path),
        output_dir=args.output,
        head=args.head,
        chromosomes=chromosomes,
        config=config,
        track_indices=track_indices,
        track_names=track_names,
        organism_index=args.organism,
        device=args.device,
        show_progress=not args.quiet,
    )

    # Summary
    total_files = sum(len(paths) for paths in results.values())
    print(f"\nDone! Wrote {total_files} BigWig file(s) to {args.output}")


if __name__ == "__main__":
    main()
