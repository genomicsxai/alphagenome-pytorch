"""agt predict — inference over chromosomes, genomic regions, or raw sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alphagenome_pytorch.cli._deps import require_extra
from alphagenome_pytorch.cli._output import emit_json, emit_text


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "predict",
        help="Inference over chromosomes, regions, or sequences to BigWig/NPZ",
        description=(
            "Run the model and write predictions to disk. Four input modes:\n"
            "  --chromosomes CHR,CHR  full chromosomes, tiled  (BigWig)\n"
            "  --locus CHR:S-E        one genomic interval    (BigWig)\n"
            "  --bed FILE             multiple genomic regions (BigWig, merged)\n"
            "  --sequences FILE       raw FASTA sequences      (NPZ per seq)\n"
            "Short locus/bed regions are padded with real reference flanks; long\n"
            "regions are center-cut unless --tile is passed. FASTA sequences\n"
            "must match the window size exactly, or require --tile for long ones."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--model", required=True, help="Path to model weights (.pth)")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--head", required=True, help="Prediction head (e.g. atac, dnase)")

    # Input mode. --locus / --bed / --sequences are mutually exclusive; if none
    # are given, --chromosomes selects full-chromosome mode. When --bed is used,
    # --chromosomes acts as a chromosome filter over the BED regions.
    inp = p.add_argument_group("Input")
    grp = inp.add_mutually_exclusive_group()
    grp.add_argument("--locus", type=str, default=None,
                     help="Single genomic interval, e.g. chr1:10000000-10131072")
    grp.add_argument("--bed", type=str, default=None,
                     help="BED file of regions (columns: chrom, start, end, [name])")
    grp.add_argument("--sequences", type=str, default=None,
                     help="FASTA file of raw sequences (no genomic coordinates)")

    inp.add_argument("--chromosomes", type=str, default=None,
                     help="Chromosome list (comma-separated). Without --bed/--locus/--sequences "
                          "this selects full-chromosome tiling. With --bed, acts as a filter on BED rows.")
    inp.add_argument("--fasta", type=str, default=None,
                     help="Reference genome FASTA (required for --chromosomes/--locus/--bed)")
    inp.add_argument("--tile", action="store_true",
                     help="Enable tiled/stitched inference for inputs longer than the window. "
                          "Off by default for --locus/--bed (long regions are center-cut). "
                          "Required for --sequences with inputs longer than the window.")

    # Tracks / resolution / tiling knobs
    p.add_argument("--tracks", type=str, default=None,
                   help="Track indices (comma-separated). Default: all")
    p.add_argument("--track-names", type=str, default=None,
                   help="Names for output tracks (comma-separated)")
    p.add_argument("--resolution", type=int, default=128, choices=[1, 128],
                   help="Output resolution in bp (default: 128)")
    p.add_argument("--crop-bp", type=int, default=0,
                   help="Base pairs to crop from each edge when tiling")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for inference")
    p.add_argument("--window-size", type=int, default=131072,
                   help="Model input window size (default: 131072)")
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

    p.add_argument("--quiet", action="store_true", help="Suppress progress bars and per-region logs")


def _parse_csv_ints(s: str | None) -> list[int] | None:
    return [int(t.strip()) for t in s.split(",")] if s else None


def _parse_csv_strs(s: str | None) -> list[str] | None:
    return [t.strip() for t in s.split(",")] if s else None


def _load_model(args, dtype_policy, json_mode):
    import torch
    from alphagenome_pytorch import AlphaGenome

    if args.checkpoint:
        from alphagenome_pytorch.extensions.finetuning.checkpointing import load_finetuned_model

        ext_config = None
        if args.transfer_config:
            import json as json_mod
            from alphagenome_pytorch.extensions.finetuning.transfer import transfer_config_from_dict
            with open(args.transfer_config) as f:
                ext_config = transfer_config_from_dict(json_mod.load(f))

        if not json_mode:
            print("Loading finetuned model...")
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
        track_names_from_ckpt = None
        if meta.get("track_names"):
            ckpt_names = meta["track_names"]
            track_names_from_ckpt = (
                ckpt_names.get(args.head) if isinstance(ckpt_names, dict) else ckpt_names
            )
        return model, track_names_from_ckpt

    if not json_mode:
        print(f"Loading model from {args.model}...")
    model = AlphaGenome.from_pretrained(
        args.model, device=args.device, dtype_policy=dtype_policy,
    )
    return model, None


def _describe_handling(info, json_mode: bool, quiet: bool) -> None:
    """Print a per-region status line and any warnings.

    The status line is suppressed under --quiet/--json, but warnings for
    ``padded`` and ``cut`` handling are always printed to stderr so the user
    knows the region was reshaped.
    """
    if not (quiet or json_mode):
        coord = (
            f"{info.chrom}:{info.start}-{info.end}"
            if info.chrom is not None else info.name
        )
        line = f"  {coord} ({info.length_bp}bp) → {info.handling}"
        if info.tile_count > 1:
            line += f" ({info.tile_count} tiles)"
        print(line)
    if not json_mode:
        for w in info.warnings:
            print(f"    WARNING: {w}", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    require_extra("inference", "predict")

    json_mode = getattr(args, "json_output", False)
    show_progress = not args.quiet and not json_mode

    import torch
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.extensions.inference import (
        GenomeSequenceProvider,
        TilingConfig,
        parse_bed,
        parse_locus,
        predict_full_chromosomes_to_bigwig,
        predict_region_auto,
        predict_sequence_auto,
        read_fasta_sequences,
        write_region_bigwig,
        write_regions_merged_bigwig,
        write_sequence_npz,
    )

    # Validate paths
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # Determine the effective input mode.
    mode_flags = {
        "--locus": args.locus is not None,
        "--bed": args.bed is not None,
        "--sequences": args.sequences is not None,
    }
    chosen = [k for k, v in mode_flags.items() if v]
    if len(chosen) == 0:
        # Fall back to full-chromosome mode (requires --chromosomes).
        if args.chromosomes is None:
            raise ValueError(
                "No input specified. Pass one of: --chromosomes, --locus, --bed, --sequences."
            )
        effective_mode = "chromosomes"
    else:
        effective_mode = chosen[0].lstrip("-")

    # --chromosomes is only a filter when combined with --bed; reject otherwise.
    if args.chromosomes is not None and effective_mode in ("locus", "sequences"):
        raise ValueError(
            f"--chromosomes cannot be combined with --{effective_mode}. "
            "Use it alone for full-chromosome mode, or with --bed to filter regions."
        )

    needs_fasta = effective_mode in ("chromosomes", "locus", "bed")
    if needs_fasta:
        if not args.fasta:
            raise ValueError(
                "--fasta is required when using --chromosomes, --locus, or --bed."
            )
        if not Path(args.fasta).exists():
            raise FileNotFoundError(f"FASTA file not found: {args.fasta}")

    track_indices = _parse_csv_ints(args.tracks)
    track_names = _parse_csv_strs(args.track_names)

    dtype_policy = (
        DtypePolicy.mixed_precision()
        if args.dtype_policy == "mixed_precision"
        else DtypePolicy.full_float32()
    )

    model, track_names_from_ckpt = _load_model(args, dtype_policy, json_mode)
    if track_names is None and track_names_from_ckpt is not None:
        track_names = track_names_from_ckpt
    model.eval()

    inner_model = getattr(model, "_orig_mod", model)
    if args.head not in inner_model.heads:
        available = list(inner_model.heads.keys())
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

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ dispatch
    if effective_mode == "chromosomes":
        chromosomes = _parse_csv_strs(args.chromosomes)
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
            show_progress=show_progress,
        )
        if json_mode:
            entries = []
            for chrom, paths in results.items():
                for pth in paths:
                    entries.append({
                        "path": str(pth),
                        "head": args.head,
                        "chromosome": chrom,
                        "resolution_bp": args.resolution,
                        "handling": "tiled",
                    })
            emit_json({"output_files": entries, "warnings": []})
        else:
            total = sum(len(ps) for ps in results.values())
            print(f"\nDone! Wrote {total} BigWig file(s) to {args.output}")
        return 0

    if effective_mode == "locus":
        chrom, start, end = parse_locus(args.locus)
        genome = GenomeSequenceProvider(args.fasta, chromosomes={chrom})
        if chrom not in genome.chrom_sizes:
            raise ValueError(f"Chromosome {chrom!r} not found in {args.fasta}")

        preds, info = predict_region_auto(
            model, genome,
            chrom=chrom, start=start, end=end,
            head=args.head, config=config,
            tile=args.tile,
            track_indices=track_indices,
            organism_index=args.organism,
            device=args.device,
        )
        _describe_handling(info, json_mode, args.quiet)

        out_path = output_dir / f"{args.head}_{info.chrom}_{info.start}_{info.end}.bw"
        written = write_region_bigwig(
            predictions=preds,
            output_path=out_path,
            chrom=info.chrom,
            start=info.start,
            chrom_sizes=genome.chrom_sizes,
            resolution=config.resolution,
            track_names=track_names,
        )
        _emit_region_results(
            [(info, written)], args, json_mode,
            extra={"resolution_bp": args.resolution},
        )
        return 0

    if effective_mode == "bed":
        regions = parse_bed(args.bed)
        # Optional --chromosomes filter.
        if args.chromosomes is not None:
            keep = set(_parse_csv_strs(args.chromosomes))
            before = len(regions)
            regions = [r for r in regions if r.chrom in keep]
            if not regions:
                raise ValueError(
                    f"--chromosomes filter removed all BED regions "
                    f"(filter: {sorted(keep)}, BED had {before} rows)."
                )
            if not args.quiet and not json_mode:
                print(f"Filtered {before} BED rows → {len(regions)} "
                      f"(kept chromosomes: {sorted(keep)})")
        needed_chroms = {r.chrom for r in regions}
        genome = GenomeSequenceProvider(args.fasta, chromosomes=needed_chroms)
        missing = needed_chroms - set(genome.chrom_sizes)
        if missing:
            raise ValueError(
                f"BED references chromosomes not in FASTA: {sorted(missing)}"
            )

        entries: list[tuple] = []  # for merged BigWig
        region_meta: list = []
        if not args.quiet and not json_mode:
            print(f"Predicting {len(regions)} regions from {args.bed}...")
        for r in regions:
            preds, info = predict_region_auto(
                model, genome,
                chrom=r.chrom, start=r.start, end=r.end,
                head=args.head, config=config,
                tile=args.tile,
                name=r.name,
                track_indices=track_indices,
                organism_index=args.organism,
                device=args.device,
            )
            _describe_handling(info, json_mode, args.quiet)
            entries.append((preds, info.chrom, info.start))
            region_meta.append(info)

        out_path = output_dir / f"{args.head}.bw"
        written = write_regions_merged_bigwig(
            entries=entries,
            output_path=out_path,
            chrom_sizes=genome.chrom_sizes,
            resolution=config.resolution,
            track_names=track_names,
        )

        if json_mode:
            emit_json({
                "output_files": [
                    {
                        "path": str(p),
                        "head": args.head,
                        "resolution_bp": args.resolution,
                        "handling": "merged_regions",
                    }
                    for p in written
                ],
                "regions": [
                    {
                        "name": info.name,
                        "chrom": info.chrom,
                        "start": info.start,
                        "end": info.end,
                        "length_bp": info.length_bp,
                        "handling": info.handling,
                        "tile_count": info.tile_count,
                        "warnings": info.warnings,
                    }
                    for info in region_meta
                ],
                "warnings": [w for info in region_meta for w in info.warnings],
            })
        else:
            print(f"\nDone! Wrote {len(written)} BigWig file(s) to {args.output}")
        return 0

    if effective_mode == "sequences":
        if not Path(args.sequences).exists():
            raise FileNotFoundError(f"FASTA file not found: {args.sequences}")
        pairs = read_fasta_sequences(args.sequences)
        if not args.quiet and not json_mode:
            print(f"Predicting {len(pairs)} sequence(s) from {args.sequences}...")

        output_files = []
        for seq_name, seq in pairs:
            preds, info = predict_sequence_auto(
                model, seq,
                name=seq_name,
                head=args.head, config=config,
                tile=args.tile,
                track_indices=track_indices,
                organism_index=args.organism,
                device=args.device,
            )
            _describe_handling(info, json_mode, args.quiet)

            # Sanitize sequence name for filesystem
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in seq_name)
            npz_path = output_dir / f"{args.head}_{safe_name}.npz"
            write_sequence_npz(
                predictions=preds,
                output_path=npz_path,
                info=info,
                head=args.head,
                resolution=config.resolution,
                track_names=track_names,
            )
            output_files.append({
                "path": str(npz_path),
                "head": args.head,
                "sequence": seq_name,
                "length_bp": info.length_bp,
                "resolution_bp": args.resolution,
                "handling": info.handling,
                "tile_count": info.tile_count,
            })

        if json_mode:
            emit_json({"output_files": output_files, "warnings": []})
        else:
            print(f"\nDone! Wrote {len(output_files)} NPZ file(s) to {args.output}")
        return 0

    # Should be unreachable thanks to mutually_exclusive_group(required=True).
    raise ValueError("No input mode selected (use --chromosomes / --locus / --bed / --sequences).")


def _emit_region_results(items, args, json_mode, *, extra=None):
    """Emit CLI output for single-region / single-locus predictions."""
    all_warnings = []
    if json_mode:
        entries = []
        for info, written in items:
            for p in written:
                entry = {
                    "path": str(p),
                    "head": args.head,
                    "chromosome": info.chrom,
                    "start": info.start,
                    "end": info.end,
                    "length_bp": info.length_bp,
                    "handling": info.handling,
                    "tile_count": info.tile_count,
                }
                if extra:
                    entry.update(extra)
                entries.append(entry)
            all_warnings.extend(info.warnings)
        emit_json({"output_files": entries, "warnings": all_warnings})
    else:
        total = sum(len(w) for _, w in items)
        print(f"\nDone! Wrote {total} BigWig file(s) to {args.output}")
