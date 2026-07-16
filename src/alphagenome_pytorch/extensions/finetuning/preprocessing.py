"""Convert BigWig files to memory-mapped numpy arrays for fast loading.

The implementation behind ``agt preprocess bigwig-to-mmap`` and
``scripts/convert_bigwigs_to_mmap.py``. Lives here, next to the
:class:`~alphagenome_pytorch.extensions.finetuning.datasets.MmapDataset` that
reads the layout it writes, so it resolves for wheel installs too.

Memory-mapped files give near-instant random reads during training without
loading whole chromosomes into memory. Produce them with
:func:`convert_bigwigs_to_mmap`, then point a dataset at them:

    train_ds = GenomicDataset(
        genome_fasta='hg38.fa',
        bigwig_files=['mmap_signals/signal1', 'mmap_signals/signal2'],
        bed_file='positions.bed',
        use_mmap=True,
    )
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def convert_single_bigwig(
    bigwig_path: str | Path,
    output_dir: str | Path,
    chromosomes: list[str] | None = None,
    dtype: np.dtype = np.float32,
) -> tuple[Path, float, float]:
    """Convert a single BigWig file to mmap format.

    Returns:
        (output_path, time_seconds, size_mb)
    """
    import json
    import pyBigWig

    bigwig_path = Path(bigwig_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    metadata = {
        "source": str(bigwig_path),
        "dtype": str(dtype),
        "chromosomes": {},
    }

    total_size = 0

    with pyBigWig.open(str(bigwig_path)) as bw:
        bw_chroms = bw.chroms()
        chroms_to_convert = chromosomes if chromosomes else list(bw_chroms.keys())

        for chrom in chroms_to_convert:
            if chrom not in bw_chroms:
                continue

            size = bw_chroms[chrom]
            values = bw.values(chrom, 0, size, numpy=True)

            if values is None:
                values = np.zeros(size, dtype=dtype)
            else:
                values = np.nan_to_num(np.asarray(values, dtype=dtype), nan=0.0)

            # Save as .npy file
            npy_filename = f"{chrom}.npy"
            npy_path = output_dir / npy_filename
            np.save(npy_path, values)

            total_size += values.nbytes

            metadata["chromosomes"][chrom] = {
                "file": npy_filename,
                "size": size,
            }

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.perf_counter() - t0
    size_mb = total_size / 1e6

    return output_dir, elapsed, size_mb


def convert_bigwigs_to_mmap(
    bigwig_files: Sequence[str | Path],
    output_dir: str | Path,
    *,
    chromosomes: list[str] | None = None,
    dtype: np.dtype = np.float32,
    workers: int = 4,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Convert one or more BigWig files to memory-mapped .npy directories.

    A single input writes straight into *output_dir*; multiple inputs each get a
    subdirectory named after the file stem — the layout ``MmapDataset`` expects.

    Args:
        bigwig_files: BigWig path(s) to convert.
        output_dir: destination directory.
        chromosomes: chromosomes to convert (default: all in the file).
        dtype: output array dtype.
        workers: threads to use for multiple files; 1 converts sequentially.
        on_result: called with each record as it finishes, for progress output.
            Fires in completion order, which is why the return value is
            re-sorted: callers formatting machine-readable output need a stable
            order regardless of which conversion finishes first.

    Returns:
        One record per input, **in input order**, each with keys
        ``input``, ``output``, ``elapsed_s``, ``size_mb``.
    """
    paths = [str(b) for b in bigwig_files]
    output_base = Path(output_dir)

    if len(paths) == 1:
        outputs = [(paths[0], output_base)]
    else:
        outputs = [(bw, output_base / Path(bw).stem) for bw in paths]

    def _convert(index: int, bw_path: str, out_path: Path) -> tuple[int, dict]:
        _, elapsed, size_mb = convert_single_bigwig(bw_path, out_path, chromosomes, dtype)
        return index, {
            "input": bw_path,
            "output": str(out_path),
            "elapsed_s": elapsed,
            "size_mb": size_mb,
        }

    records: list[dict | None] = [None] * len(outputs)
    n_workers = min(len(outputs), max(1, workers))

    if n_workers == 1:
        for index, (bw_path, out_path) in enumerate(outputs):
            _, record = _convert(index, bw_path, out_path)
            records[index] = record
            if on_result is not None:
                on_result(record)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_convert, index, bw_path, out_path)
                for index, (bw_path, out_path) in enumerate(outputs)
            ]
            for future in as_completed(futures):
                index, record = future.result()
                records[index] = record
                if on_result is not None:
                    on_result(record)

    return [r for r in records if r is not None]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for BigWig -> mmap conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BigWig files to memory-mapped numpy arrays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bigwig",
        type=str,
        nargs="+",
        required=True,
        help="BigWig file(s) to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory. For multiple files, subdirs are created per file.",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="*",
        default=None,
        help="Chromosomes to convert (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for multiple files",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Data type for output arrays",
    )
    args = parser.parse_args(argv)

    dtype = np.float32 if args.dtype == "float32" else np.float16

    print("=" * 70)
    print("BigWig to Mmap Converter")
    print("=" * 70)
    print(f"Input files: {len(args.bigwig)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Chromosomes: {args.chromosomes or 'all'}")
    print(f"Dtype: {dtype}")
    print(f"Workers: {args.workers}")
    print()

    def _report(record: dict) -> None:
        name = Path(record["input"]).name
        print(f"  {name} -> {record['output']} "
              f"({record['elapsed_s']:.1f}s, {record['size_mb']:.1f} MB)")

    records = convert_bigwigs_to_mmap(
        args.bigwig,
        args.output_dir,
        chromosomes=args.chromosomes,
        dtype=dtype,
        workers=args.workers,
        on_result=_report,
    )

    print()
    print("-" * 70)
    print(f"Done! Converted {len(records)} file(s)")
    print(f"Total time: {sum(r['elapsed_s'] for r in records):.1f}s")
    print(f"Total size: {sum(r['size_mb'] for r in records):.1f} MB")
    print()
    print("Usage in training:")
    print("  train_ds = GenomicDataset(")
    print(f"      bigwig_files=['{records[0]['output']}', ...],")
    print("      use_mmap=True,")
    print("  )")


if __name__ == "__main__":
    main()
