"""agt preprocess — data preprocessing utilities.

Subcommands:
    bigwig-to-mmap    Convert BigWig files to memory-mapped format
    scale-bigwig      Normalize BigWig signal to a target total
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from alphagenome_pytorch.cli._deps import require_extra
from alphagenome_pytorch.cli._output import emit_json, emit_text


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "preprocess",
        help="Data preprocessing utilities",
        description="Data preprocessing utilities (bigwig-to-mmap, scale-bigwig).",
    )

    sub = p.add_subparsers(dest="preprocess_command")

    # ---- bigwig-to-mmap ----
    bw = sub.add_parser(
        "bigwig-to-mmap",
        help="Convert BigWig files to memory-mapped format",
    )
    bw.add_argument("--input", nargs="+", required=True, help="BigWig file(s)")
    bw.add_argument("--output", required=True, help="Output directory")
    bw.add_argument("--genome", type=str, default=None, help="Reference FASTA (unused, for compat)")
    bw.add_argument("--resolution", type=int, default=128, help="Resolution (unused, for compat)")
    bw.add_argument("--chromosomes", nargs="*", default=None,
                     help="Chromosomes to convert")
    bw.add_argument("--workers", type=int, default=4, help="Parallel workers")
    bw.add_argument("--dtype", choices=["float32", "float16"], default="float32")

    # ---- scale-bigwig ----
    sb = sub.add_parser(
        "scale-bigwig",
        help="Normalize BigWig signal to a target total",
    )
    sb.add_argument("--input", nargs="+", required=True, help="BigWig file(s)")
    sb.add_argument("--output", default=None, help="Output path or directory")
    sb.add_argument("--target", required=True,
                     help="Target total signal (e.g. 100M, 50k)")
    sb.add_argument("--dry-run", action="store_true",
                     help="Compute scale factor without writing")


def parse_target(s: str) -> float:
    """Parse human-readable target strings like '100M', '50k' to numbers."""
    s = s.strip()
    suffixes = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "g": 1e9, "G": 1e9, "b": 1e9, "B": 1e9}
    m = re.match(r'^([0-9]*\.?[0-9]+)\s*([a-zA-Z]?)$', s)
    if m is None:
        raise ValueError(f"Cannot parse target: '{s}'. Use format like '100M', '50k', or a plain number.")
    number = float(m.group(1))
    suffix = m.group(2)
    if suffix:
        if suffix not in suffixes:
            raise ValueError(f"Unknown suffix '{suffix}' in target '{s}'. Use k, M, G.")
        number *= suffixes[suffix]
    return number


def _run_bigwig_to_mmap(args: argparse.Namespace) -> int:
    """Convert BigWig files to mmap format."""
    require_extra("inference", "preprocess bigwig-to-mmap")

    json_mode = getattr(args, "json_output", False)

    import numpy as np
    from scripts.convert_bigwigs_to_mmap import convert_single_bigwig

    bigwig_files = args.input
    output_base = Path(args.output)
    dtype = np.float32 if args.dtype == "float32" else np.float16

    if len(bigwig_files) == 1:
        outputs = [(bigwig_files[0], output_base)]
    else:
        outputs = [(bw, output_base / Path(bw).stem) for bw in bigwig_files]

    results = []
    for bw_path, out_path in outputs:
        if not json_mode:
            print(f"Converting: {Path(bw_path).name}")
        _, elapsed, size_mb = convert_single_bigwig(bw_path, out_path, args.chromosomes, dtype)
        results.append({
            "path": str(out_path),
            "tracks": 1,
            "size_mb": round(size_mb, 1),
        })
        if not json_mode:
            print(f"  -> {out_path} ({elapsed:.1f}s, {size_mb:.1f} MB)")

    if json_mode:
        emit_json({
            "output_files": results,
            "records_processed": len(results),
        })
    else:
        print(f"\nDone! Converted {len(results)} file(s)")

    return 0


def _run_scale_bigwig(args: argparse.Namespace) -> int:
    """Scale BigWig files to target total signal."""
    require_extra("inference", "preprocess scale-bigwig")

    json_mode = getattr(args, "json_output", False)

    import numpy as np

    target = parse_target(args.target)
    bigwig_files = args.input
    dry_run = args.dry_run

    if not dry_run and not args.output:
        print(
            "Error: --output is required unless --dry-run is set "
            "(otherwise scale-bigwig would compute scale factors but write nothing).",
            file=sys.stderr,
        )
        return 1

    results = []
    for bw_path in bigwig_files:
        bw_path = str(bw_path)
        if not Path(bw_path).exists():
            raise FileNotFoundError(f"BigWig file not found: {bw_path}")

        total_signal = _compute_bigwig_total(bw_path)
        scale_factor = target / total_signal if total_signal > 0 else 1.0

        # Determine output path
        if args.output and len(bigwig_files) == 1:
            out_path = args.output
        elif args.output:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir / Path(bw_path).name)
        else:
            out_path = None

        if not dry_run and out_path:
            _write_scaled_bigwig(bw_path, out_path, scale_factor)

        results.append({
            "input": bw_path,
            "output": out_path,
            "original_total": round(total_signal, 1),
            "target_total": target,
            "scale_factor": round(scale_factor, 4),
        })

        if not json_mode:
            mode = " (dry run)" if dry_run else ""
            print(f"{Path(bw_path).name}: total={total_signal:.0f}, scale_factor={scale_factor:.4f}{mode}")

    if json_mode:
        emit_json({"files": results})

    return 0


def _compute_bigwig_total(path: str) -> float:
    """Compute total signal across all chromosomes in a BigWig file."""
    import pyBigWig
    import numpy as np

    total = 0.0
    with pyBigWig.open(path) as bw:
        for chrom, size in bw.chroms().items():
            values = bw.values(chrom, 0, size, numpy=True)
            if values is not None:
                total += float(np.nansum(values))
    return total


def _write_scaled_bigwig(input_path: str, output_path: str, scale_factor: float) -> None:
    """Write a scaled BigWig file chromosome-by-chromosome."""
    import pyBigWig
    import numpy as np

    with pyBigWig.open(input_path) as bw_in:
        chroms = bw_in.chroms()
        header = list(chroms.items())

        bw_out = pyBigWig.open(output_path, "w")
        bw_out.addHeader(header)

        for chrom, size in chroms.items():
            values = bw_in.values(chrom, 0, size, numpy=True)
            if values is None:
                continue
            values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
            values *= scale_factor

            # Write in chunks
            CHUNK = 1_000_000
            for start in range(0, len(values), CHUNK):
                end = min(start + CHUNK, len(values))
                bw_out.addEntries(chrom, start, values=values[start:end].tolist(), span=1, step=1)

        bw_out.close()


def run(args: argparse.Namespace) -> int:
    if not hasattr(args, "preprocess_command") or args.preprocess_command is None:
        print("Error: specify a subcommand: bigwig-to-mmap, scale-bigwig", file=sys.stderr)
        return 1

    if args.preprocess_command == "bigwig-to-mmap":
        return _run_bigwig_to_mmap(args)
    elif args.preprocess_command == "scale-bigwig":
        return _run_scale_bigwig(args)
    else:
        print(f"Error: unknown preprocess command: {args.preprocess_command}", file=sys.stderr)
        return 1
