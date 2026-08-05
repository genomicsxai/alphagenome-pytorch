#!/usr/bin/env python3
"""Extract splice junctions from a STAR-aligned BAM in SJ.out.tab format.

Uses regtools junctions extract (fast C++ streaming) for total read counts
and pysam for unique (NH=1) read counts. Both passes run in parallel.

Junction strand is inferred from the XS tag that STAR writes based on
splice-site sequence — works for both stranded and unstranded libraries.

Output columns match STAR SJ.out.tab exactly:
  chrom  intron_start  intron_end  strand_code  intron_motif  annotated
  n_uniquely_mapped_reads  n_multi_mapped_reads  max_overhang

intron_motif and annotated are set to 0 (no genome sequence required).

Usage:
    python scripts/get_star_junctions.py \\
        --bam sample.bam \\
        --output sample.SJ.out.tab

    # restrict to specific chromosomes
    python scripts/get_star_junctions.py \\
        --bam sample.bam \\
        --output sample.SJ.out.tab \\
        --chroms chr1 chr2
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit(
        "Error: pandas is required.\n"
        "Install with:  pip install pandas"
    )

try:
    import pysam  # noqa: F401 — checked here so the error surfaces early
except ImportError:
    sys.exit(
        "Error: pysam is required.\n"
        "Install with:  pip install pysam\n"
        "  or (conda):  conda install -c bioconda pysam"
    )

if shutil.which("regtools") is None:
    sys.exit(
        "Error: regtools is required but was not found on PATH.\n"
        "Install with:  conda install -c bioconda regtools\n"
        "  or see:      https://regtools.readthedocs.io/en/latest/install/"
    )


STRAND_CODE = {"+": 1, "-": 2, ".": 0}

_BED12_NAMES = [
    "chrom", "chromStart", "chromEnd", "name", "score", "strand",
    "thickStart", "thickEnd", "rgb", "blockCount", "blockSizes", "blockStarts",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    io_g = p.add_argument_group("Input / output")
    io_g.add_argument("--bam", "-b", required=True,
                      help="Coordinate-sorted, indexed BAM")
    io_g.add_argument("--output", "-o", required=True,
                      help="Output SJ.out.tab path")

    filt = p.add_argument_group("Filtering")
    filt.add_argument("--chroms", "-C", nargs="*", default=None, metavar="CHROM",
                      help="Chromosomes to process (default: all)")
    filt.add_argument("--min-overhang", type=int, default=8, metavar="INT",
                      help="Minimum junction anchor for regtools (default: 8)")
    filt.add_argument("--strand", default="XS", choices=["XS", "RF", "FR"],
                      help="Strand protocol: XS=use STAR XS tag (works for stranded "
                           "and unstranded), RF=first-strand, FR=second-strand (default: XS)")
    return p.parse_args()


# ------------------------------------------------------------------ #
# regtools: total junction counts
# ------------------------------------------------------------------ #

def _run_regtools_chrom(
    bam: str, chrom: str | None, strand: str, min_overhang: int
) -> bytes:
    """Run regtools on one chromosome (or whole BAM if chrom=None)."""
    cmd = [
        "regtools", "junctions", "extract",
        "-s", strand,
        "-a", str(min_overhang),
        "-o", "/dev/stdout",
    ]
    if chrom:
        cmd += ["-r", chrom]
    cmd.append(bam)

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"regtools failed (exit {result.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _regtools_total(
    bam: str,
    chroms: list[str] | None,
    strand: str,
    min_overhang: int,
) -> bytes:
    """Run regtools across all requested chromosomes, in parallel if multiple."""
    if not chroms:
        return _run_regtools_chrom(bam, None, strand, min_overhang)

    with ThreadPoolExecutor(max_workers=len(chroms)) as pool:
        futures = [
            pool.submit(_run_regtools_chrom, bam, c, strand, min_overhang)
            for c in chroms
        ]
        return b"".join(f.result() for f in futures)


def _parse_bed(bed_bytes: bytes) -> pd.DataFrame:
    """Parse regtools BED12 bytes into a junction DataFrame.

    BED coords (0-based) → STAR coords (1-based):
      intron_start = chromStart + left_anchor + 1
      intron_end   = chromEnd   - right_anchor       (inclusive)
    """
    if not bed_bytes.strip():
        return pd.DataFrame(
            columns=["chrom", "intron_start", "intron_end", "strand", "n_total", "max_overhang"]
        )

    df = pd.read_csv(
        io.BytesIO(bed_bytes),
        sep="\t",
        header=None,
        names=_BED12_NAMES,
        dtype={"chrom": str, "chromStart": int, "chromEnd": int,
               "score": int, "strand": str},
    )

    anchors     = df["blockSizes"].str.rstrip(",").str.split(",", expand=True).astype(int)
    left_anchor = anchors[0]
    right_anchor = anchors[1]

    df["intron_start"] = df["chromStart"] + left_anchor  + 1
    df["intron_end"]   = df["chromEnd"]   - right_anchor
    df["n_total"]      = df["score"]
    df["max_overhang"] = anchors.max(axis=1)

    return df[["chrom", "intron_start", "intron_end", "strand", "n_total", "max_overhang"]].copy()


# ------------------------------------------------------------------ #
# pysam: unique (NH=1) junction counts
# ------------------------------------------------------------------ #

def _pysam_unique_counts(
    bam_path: str,
    chroms: list[str] | None,
) -> dict[tuple[str, int, int], int]:
    """Count reads with NH=1 per intron using pysam.find_introns().

    Returns {(chrom, intron_start_1based, intron_end_1based): count}.
    """
    bam = pysam.AlignmentFile(bam_path, "rb")

    if not chroms:
        chroms = [bam.get_reference_name(i) for i in range(bam.nreferences)]

    counts: dict[tuple[str, int, int], int] = {}
    for chrom in chroms:
        unique_reads = (
            r for r in bam.fetch(chrom)
            if not r.is_unmapped
            and not r.is_secondary
            and not r.is_supplementary
            and r.has_tag("NH")
            and r.get_tag("NH") == 1
        )
        # find_introns returns {(iv_s, iv_e): count} in 0-based half-open coords
        for (iv_s, iv_e), n in bam.find_introns(unique_reads).items():
            counts[(chrom, iv_s + 1, iv_e)] = n  # → 1-based STAR coords

    bam.close()
    return counts


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()

    print(f"Extracting junctions from {args.bam!r} …")

    # Run regtools (total counts) and pysam (unique counts) in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_total  = pool.submit(
            _regtools_total, args.bam, args.chroms, args.strand, args.min_overhang
        )
        fut_unique = pool.submit(
            _pysam_unique_counts, args.bam, args.chroms
        )
        total_bed  = fut_total.result()
        unique_dict = fut_unique.result()

    total_df = _parse_bed(total_bed)
    print(f"  regtools total:  {len(total_df):,} junctions")
    print(f"  pysam unique:    {len(unique_dict):,} junctions")

    # Map unique counts onto the total DataFrame
    key = ["chrom", "intron_start", "intron_end"]
    total_df["n_unique"] = total_df.apply(
        lambda r: unique_dict.get((r["chrom"], r["intron_start"], r["intron_end"]), 0),
        axis=1,
    )
    total_df["n_multi"] = (total_df["n_total"] - total_df["n_unique"]).clip(lower=0)

    out = pd.DataFrame({
        "chrom":                   total_df["chrom"],
        "intron_start":            total_df["intron_start"],
        "intron_end":              total_df["intron_end"],
        "strand_code":             total_df["strand"].map(STRAND_CODE).fillna(0).astype(int),
        "intron_motif":            0,
        "annotated":               0,
        "n_uniquely_mapped_reads": total_df["n_unique"],
        "n_multi_mapped_reads":    total_df["n_multi"],
        "max_overhang":            total_df["max_overhang"],
    })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", header=False, index=False)
    print(f"Wrote {len(out):,} junctions → {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
