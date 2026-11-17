"""agt preprocess — data preprocessing utilities.

Subcommands:
    bigwig-to-mmap    Convert BigWig files to memory-mapped format
    scale-bigwig      Normalize BigWig signal to a target total
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from alphagenome_pytorch.cli._deps import require_extra
from alphagenome_pytorch.cli._output import emit_json, emit_text
from alphagenome_pytorch.extensions.finetuning.signal_scaling import (
    bigwig_total_signal,
    iter_bigwig_windows,
    parse_target,
)


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
    sb.add_argument("--together", action="store_true",
                     help="Scale all inputs by one factor so their combined total "
                          "equals --target (e.g. a stranded RNA-seq +/- pair). "
                          "Files stay separate.")


def _run_bigwig_to_mmap(args: argparse.Namespace) -> int:
    """Convert BigWig files to mmap format."""
    require_extra("inference", "preprocess bigwig-to-mmap")

    json_mode = getattr(args, "json_output", False)

    import numpy as np
    from alphagenome_pytorch.extensions.finetuning.preprocessing import (
        convert_bigwigs_to_mmap,
    )

    dtype = np.float32 if args.dtype == "float32" else np.float16

    def _report(record: dict) -> None:
        print(f"  {Path(record['input']).name} -> {record['output']} "
              f"({record['elapsed_s']:.1f}s, {record['size_mb']:.1f} MB)")

    records = convert_bigwigs_to_mmap(
        args.input,
        args.output,
        chromosomes=args.chromosomes,
        dtype=dtype,
        workers=args.workers,
        on_result=None if json_mode else _report,
    )

    results = [
        {"path": record["output"], "tracks": 1, "size_mb": round(record["size_mb"], 1)}
        for record in records
    ]

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

    bigwig_files = [str(p) for p in bigwig_files]
    for bw_path in bigwig_files:
        if not Path(bw_path).exists():
            raise FileNotFoundError(f"BigWig file not found: {bw_path}")

    if not args.output:
        outputs = [None] * len(bigwig_files)
    elif len(bigwig_files) == 1:
        outputs = [args.output]
    else:
        outputs = [str(Path(args.output) / Path(p).name) for p in bigwig_files]
        if not dry_run:
            destinations = {Path(p).resolve() for p in outputs}
            sources = {Path(p).resolve() for p in bigwig_files}
            if destinations & sources or len(destinations) != len(outputs):
                raise ValueError("Output files must be distinct and separate from the input files.")

    totals = [bigwig_total_signal(p, absolute=args.together) for p in bigwig_files]
    combined_total = sum(totals)
    if args.together and (combined_total <= 0 or target <= 0):
        raise ValueError("--together needs a positive combined input total and target.")

    if not dry_run and len(bigwig_files) > 1:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    results = []
    for bw_path, out_path, total_signal in zip(bigwig_files, outputs, totals):
        denom = combined_total if args.together else total_signal
        scale_factor = target / denom if denom > 0 else 1.0

        if not dry_run and out_path:
            _write_scaled_bigwig(bw_path, out_path, scale_factor)

        results.append({
            "input": bw_path,
            "output": out_path,
            "original_total": round(total_signal, 1),
            # With --together, target applies to the sum, so report
            # this file's share of it.
            "target_total": round(total_signal * scale_factor, 1) if args.together else target,
            "scale_factor": round(scale_factor, 4),
        })

        if not json_mode:
            mode = " (dry run)" if dry_run else ""
            print(f"{Path(bw_path).name}: total={total_signal:.0f}, scale_factor={scale_factor:.4f}{mode}")

    if json_mode:
        emit_json({"files": results})
    elif args.together:
        print(f"combined: total={combined_total:.0f}, target={target:.0f}")

    return 0


def _runs(values):
    """Run-length encode *values*, dropping NaN gaps.

    Returns (starts, ends, run_values) relative to the window start.
    """
    import numpy as np

    valid = ~np.isnan(values)
    # A run starts where validity or value changes.
    change = np.empty(len(values), dtype=bool)
    change[0] = True
    change[1:] = (valid[1:] != valid[:-1]) | (valid[1:] & (values[1:] != values[:-1]))
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(values))
    keep = valid[starts]
    return starts[keep], ends[keep], values[starts[keep]]


def _write_scaled_bigwig(input_path: str, output_path: str, scale_factor: float) -> None:
    """Write a scaled BigWig file. NaN gaps stay gaps; equal-value runs stay runs."""
    import numpy as np
    import pyBigWig

    output = Path(output_path)
    # Write beside the destination so a failed write leaves it intact.
    with tempfile.TemporaryDirectory(dir=output.parent) as temp_dir:
        staged = Path(temp_dir) / output.name
        with pyBigWig.open(input_path) as bw_in, pyBigWig.open(str(staged), "w") as bw_out:
            chroms = bw_in.chroms()
            bw_out.addHeader(list(chroms.items()))

            for chrom, size in chroms.items():
                for offset, values in iter_bigwig_windows(bw_in, chrom, size):
                    starts, ends, run_values = _runs(values)
                    if len(starts) == 0:
                        continue
                    if len(starts) > len(values) // 2 and not np.isnan(values).any():
                        # Fixed-step output is cheaper for dense windows.
                        bw_out.addEntries(
                            chrom, offset,
                            values=values.astype(np.float64) * scale_factor,
                            span=1, step=1,
                        )
                    else:
                        bw_out.addEntries(
                            [chrom] * len(starts),
                            starts + offset,
                            ends=ends + offset,
                            values=run_values.astype(np.float64) * scale_factor,
                        )
        staged.replace(output)


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
