#!/usr/bin/env python
"""Compute splice site usage (SSU) from STAR junction data, optionally with a BAM.

For each splice site in a SJ.out.tab file, computes:

  SSU approx  = α / (α + β2)         [junction-only, no BAM needed]
  SSU full    = α / (α + β1 + β2)    [α/β2 from junctions, β1 from BAM]
  SSU spliser = α / (α + β1 + β2)    [all counts from BAM, equivalent to SpliSER]

where:
  α  = split reads using this site
  β1 = reads spanning the site continuously without splicing (no N CIGAR)
  β2 = reads using a competing site for the same partner

Coordinates are 1-based.  Each splice site is reported with both its exonic
coordinate (last exon base for donor, first exon base for acceptor) and its
intronic coordinate (first intron base for donor, last intron base for acceptor).

Usage:
    # Junction-only approximation
    python scripts/compute_ssu.py \\
        --junctions second_pass.SJ.out.tab \\
        --output ssu.parquet

    # Full SSU with BAM ground truth (3 metrics)
    python scripts/compute_ssu.py \\
        --junctions second_pass.SJ.out.tab \\
        --bam second_pass.Aligned.sortedByCoord.out.filtered.bam \\
        --output ssu.parquet
"""

from __future__ import annotations

import argparse
import bisect
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute splice site usage (SSU) from STAR junction data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io = p.add_argument_group("Input / output")
    io.add_argument("--junctions", "-j", required=True,
                    help="STAR SJ.out.tab file")
    io.add_argument("--bam", "-b", default=None,
                    help="Coordinate-sorted, indexed BAM (enables SSU full)")
    io.add_argument("--output", "-o", required=True,
                    help="Output Parquet file path")

    filt = p.add_argument_group("Filtering")
    filt.add_argument("--min-unique-reads", type=int, default=1,
                      help="Minimum uniquely mapped reads to retain a junction (default: 1)")
    filt.add_argument("--mapq", type=int, default=30,
                      help="Minimum MAPQ for all BAM reads (α, β1, β2 counting and "
                           "whiteset construction). Use 0 to match SpliSER's "
                           "unfiltered behaviour (default: 30)")
    filt.add_argument("--chroms", "-C", nargs="*", default=None, metavar="CHROM",
                      help="Chromosomes to process (default: all). Space-separated list.")

    out = p.add_argument_group("Output")
    out.add_argument("--compression", "-c",
                     choices=["snappy", "gzip", "zstd", "none"],
                     default="zstd",
                     help="Parquet compression codec (default: zstd)")

    return p.parse_args()


# ------------------------------------------------------------------ #
# Step 1: load and filter junctions
# ------------------------------------------------------------------ #

_STRAND_MAP = {"0": ".", "1": "+", "2": "-"}

def read_star_junctions(path: str) -> pd.DataFrame:
    """Read a STAR SJ.out.tab file into a DataFrame.

    Args:
        path: Path to SJ.out.tab file.

    Returns:
        DataFrame with columns: chrom, intron_start, intron_end, strand,
        intron_motif, annotated, n_uniquely_mapped_reads, n_multi_mapped_reads,
        max_overhang.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "chrom",
            "intron_start",
            "intron_end",
            "strand_code",
            "intron_motif",
            "annotated",
            "n_uniquely_mapped_reads",
            "n_multi_mapped_reads",
            "max_overhang",
        ],
        dtype={
            "chrom": str,
            "intron_start": np.int64,
            "intron_end": np.int64,
            "strand_code": str,
            "intron_motif": np.int64,
            "annotated": np.int64,
            "n_uniquely_mapped_reads": np.int64,
            "n_multi_mapped_reads": np.int64,
            "max_overhang": np.int64,
        },
    )
    df["strand"] = df["strand_code"].map(_STRAND_MAP).fillna(".")
    df = df.drop(columns=["strand_code"])
    return df

def load_junctions(
    path: str | Path,
    min_unique_reads: int,
    chroms: "list[str] | None" = None,
) -> "pd.DataFrame":
    """Read and quality-filter a STAR SJ.out.tab file.

    Args:
        path: Path to SJ.out.tab.
        min_unique_reads: Minimum n_uniquely_mapped_reads threshold.
        chroms: Optional list of chromosomes to retain (default: all).

    Returns:
        DataFrame with added columns exon_start, exon_end, count.
    """

    junctions = read_star_junctions(str(path))
    mask = (
        (junctions["n_uniquely_mapped_reads"] >= min_unique_reads)
        & (junctions["strand"].isin(["+", "-"]))
    )
    if chroms is not None:
        mask &= junctions["chrom"].isin(chroms)
    junctions = junctions.loc[mask].copy()
    junctions["exon_start"] = junctions["intron_start"] - 1  # last exon base (1-based)
    junctions["exon_end"]   = junctions["intron_end"]   + 1  # first exon base (1-based)
    junctions["count"]      = junctions["n_uniquely_mapped_reads"]
    return junctions.reset_index(drop=True)


# ------------------------------------------------------------------ #
# Step 2: α and β2 from junction data
# ------------------------------------------------------------------ #

def compute_alpha_beta2(
    junctions: "pd.DataFrame",
) -> "tuple[pd.Series, pd.Series, pd.Series, pd.Series]":
    """Compute per-site α and β2 from junction counts.

    β2(D) = Σ_{A: D→A} acceptor_total(A) − α(D)
    β2(A) = Σ_{D: D→A} donor_total(D) − α(A)

    Fully vectorized (no iterrows).

    Args:
        junctions: DataFrame with exon_start, exon_end, strand, chrom, count.

    Returns:
        (donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2) as Series
        indexed by (chrom, 1-based position, strand).
    """
    donor_alpha = (
        junctions.groupby(["chrom", "exon_start", "strand"])["count"].sum()
        .rename("donor_total")
    )
    acceptor_alpha = (
        junctions.groupby(["chrom", "exon_end", "strand"])["count"].sum()
        .rename("acceptor_total")
    )

    j = junctions.join(acceptor_alpha, on=["chrom", "exon_end", "strand"])
    j = j.join(donor_alpha,            on=["chrom", "exon_start", "strand"])

    donor_beta2 = (
        j.groupby(["chrom", "exon_start", "strand"])["acceptor_total"].sum()
        - donor_alpha
    ).rename("donor_beta2")
    acceptor_beta2 = (
        j.groupby(["chrom", "exon_end", "strand"])["donor_total"].sum()
        - acceptor_alpha
    ).rename("acceptor_beta2")

    return donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2


# ------------------------------------------------------------------ #
# Step 3: β1 from BAM (optional)
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# Step 3: SpliSER-equivalent counts from BAM (all α, β1, β2 from BAM)
# ------------------------------------------------------------------ #

def _check_strand_from_flag(flag: int, strandedType: str = "rf") -> str | None:
    """Determine transcript strand from SAM flag bits (mirrors SpliSER check_strand)."""
    is_paired   = bool(flag & 0x1)
    is_reverse  = bool(flag & 0x10)
    is_read1    = bool(flag & 0x40)

    if not is_paired:
        mate = 1
    elif is_read1:
        mate = 1
    else:
        mate = 2

    if strandedType == "rf":
        if mate == 1:
            return "+" if is_reverse else "-"
        else:
            return "-" if is_reverse else "+"
    elif strandedType == "fr":
        if mate == 1:
            return "-" if is_reverse else "+"
        else:
            return "+" if is_reverse else "-"
    return None


def compute_spliser_counts(
    bam_path: str | Path,
    junctions: "pd.DataFrame",
    mapq_min: int = 30,
    strandedType: str = "rf",
) -> "pd.DataFrame":
    """Compute SpliSER-equivalent α, β1, β2 for all splice sites from BAM.

    Single streaming pass per chromosome — no read buffering.  The target
    index is pre-built from the junction file so alpha and beta can be
    accumulated simultaneously as each read is consumed.

    Coordinate note: donor scan pos = exon_start (= iv_s, 0-based intron
    start); acceptor scan pos = exon_end - 2 (= iv_e - 1, 0-based last
    intron base).  Both equal the corresponding pysam find_introns keys
    numerically, so results are identical to the find_introns approach.

    Returns DataFrame with columns:
    chrom, position, strand, role, alpha_bam, beta1_bam, beta2_bam, ssu_spliser.
    """
    try:
        import pysam
    except ImportError as e:
        raise ImportError("pysam is required for SpliSER computation") from e

    bam = pysam.AlignmentFile(str(bam_path), "rb")

    donor_alpha_bam:    defaultdict = defaultdict(int)
    acceptor_alpha_bam: defaultdict = defaultdict(int)
    beta1_bam: defaultdict = defaultdict(int)
    beta2_bam: defaultdict = defaultdict(int)

    # strand-agnostic accumulators (AlphaGenome-style: "regardless of strand")
    donor_alpha_ag_bam:    defaultdict = defaultdict(int)
    acceptor_alpha_ag_bam: defaultdict = defaultdict(int)
    beta1_ag_bam:          defaultdict = defaultdict(int)

    # truly unstranded α: find_introns on ALL reads before majority-rule, so
    # reads with ambiguous strand flags (excluded by _check_strand_from_flag)
    # also contribute.  β1 reuses beta1_ag_bam (already all-reads).
    donor_alpha_nostr_bam:    defaultdict = defaultdict(int)
    acceptor_alpha_nostr_bam: defaultdict = defaultdict(int)

    for chrom, chrom_junc in junctions.groupby("chrom"):
        # ── Target index ───────────────────────────────────────────────
        # donor scan pos    = exon_start      (= iv_s,   0-based intron start)
        # acceptor scan pos = exon_end - 2    (= iv_e-1, 0-based last intron base)
        # Sets deduplicate roles so a site shared by N junctions is only
        # scanned once per read.
        chrom_target_roles: dict[str, dict[int, set[str]]] = {"+": {}, "-": {}}

        # Partner/competitor maps for SpliSER-compatible beta2 classification.
        # donor_to_acc[strand][d_pos]  → {a_pos, ...}   (a_pos = iv_e - 1)
        # acc_to_donor[strand][a_pos]  → {d_pos, ...}
        donor_to_acc: dict[str, dict] = {"+": defaultdict(set), "-": defaultdict(set)}
        acc_to_donor: dict[str, dict] = {"+": defaultdict(set), "-": defaultdict(set)}

        for row in chrom_junc[["strand", "exon_start", "exon_end"]].itertuples(index=False):
            d_pos = int(row.exon_start)
            a_pos = int(row.exon_end) - 1
            s     = row.strand
            chrom_target_roles[s].setdefault(d_pos, set()).add("donor")
            chrom_target_roles[s].setdefault(a_pos, set()).add("acceptor")
            donor_to_acc[s][d_pos].add(a_pos)
            acc_to_donor[s][a_pos].add(d_pos)

        # Competitor maps: sites sharing a partner with this site.
        # donor_comps[strand][d_pos]  → {competing d_pos', ...}
        # acc_comps[strand][a_pos]    → {competing a_pos', ...}
        donor_comps: dict[str, dict] = {"+": defaultdict(set), "-": defaultdict(set)}
        acc_comps:   dict[str, dict] = {"+": defaultdict(set), "-": defaultdict(set)}
        # Whiteset + partner map extension: bam.find_introns includes junctions
        # absent from SJ.out.tab (e.g. non-canonical or low-overhang junctions).
        # Extending donor_to_acc / acc_to_donor with these gives complete
        # competitor maps so compSplicing is detected even for STAR-missed partners.
        # Majority-rule strand collapse (mirrors SpliSER's collapse_duplicate_introns):
        # for introns seen on both strands, keep only the dominant strand to prevent
        # antisense-noise reads from polluting the whiteset of the sense strand.
        whiteset: dict[str, set[int]] = {"+": set(), "-": set()}

        # Unstranded introns: all filtered reads, no strand check.
        # Differs from alpha_ag (which sums majority-rule strand counts) only for
        # reads whose strand flag is ambiguous — those are excluded by
        # _check_strand_from_flag and therefore absent from _introns_by_strand.
        _gen_nostr = (
            r for r in bam.fetch(chrom)
            if not r.is_unmapped and not r.is_secondary and not r.is_supplementary
            and r.mapping_quality >= mapq_min
        )
        _introns_nostr: dict = dict(bam.find_introns(_gen_nostr))

        _gen_p = (
            r for r in bam.fetch(chrom)
            if not r.is_unmapped and not r.is_secondary and not r.is_supplementary
            and r.mapping_quality >= mapq_min
            and _check_strand_from_flag(r.flag, strandedType) == "+"
        )
        _gen_m = (
            r for r in bam.fetch(chrom)
            if not r.is_unmapped and not r.is_secondary and not r.is_supplementary
            and r.mapping_quality >= mapq_min
            and _check_strand_from_flag(r.flag, strandedType) == "-"
        )
        _introns_by_strand: dict[str, dict] = {
            "+": dict(bam.find_introns(_gen_p)),
            "-": dict(bam.find_introns(_gen_m)),
        }
        for _iv in set(_introns_by_strand["+"]) & set(_introns_by_strand["-"]):
            if _introns_by_strand["+"][_iv] >= _introns_by_strand["-"][_iv]:
                _introns_by_strand["+"][_iv] += _introns_by_strand["-"][_iv]
                del _introns_by_strand["-"][_iv]
            else:
                _introns_by_strand["-"][_iv] += _introns_by_strand["+"][_iv]
                del _introns_by_strand["+"][_iv]
        for s, _introns_s in _introns_by_strand.items():
            for iv_s_w, iv_e_w in _introns_s:
                whiteset[s].add(iv_s_w)
                whiteset[s].add(iv_e_w)
                donor_to_acc[s][iv_s_w].add(iv_e_w)
                acc_to_donor[s][iv_e_w].add(iv_s_w)

        for s in ("+", "-"):
            for d, acc_set in donor_to_acc[s].items():
                for a in acc_set:
                    for other_d in acc_to_donor[s][a]:
                        if other_d != d:
                            donor_comps[s][d].add(other_d)
            for a, don_set in acc_to_donor[s].items():
                for d in don_set:
                    for other_a in donor_to_acc[s][d]:
                        if other_a != a:
                            acc_comps[s][a].add(other_a)

        chrom_targets: dict[str, list[int]] = {
            strand: sorted(pos_roles)
            for strand, pos_roles in chrom_target_roles.items()
        }

        # Strand-agnostic target index: union of all positions from both strands.
        chrom_targets_ag: dict[int, set[str]] = {}
        for s in ("+", "-"):
            for pos, roles in chrom_target_roles[s].items():
                if pos not in chrom_targets_ag:
                    chrom_targets_ag[pos] = set()
                chrom_targets_ag[pos].update(roles)
        targets_ag_sorted = sorted(chrom_targets_ag)

        # Alpha from the collapsed intron dict (same majority-rule as whiteset/SpliSER).
        chrom_donor_alpha:    defaultdict = defaultdict(int)
        chrom_acceptor_alpha: defaultdict = defaultdict(int)
        for s, _introns_s in _introns_by_strand.items():
            for (iv_s, iv_e), count in _introns_s.items():
                chrom_donor_alpha[(iv_s, s)]    += count
                chrom_acceptor_alpha[(iv_e, s)] += count

        # Unstranded alpha: from _introns_nostr, no strand assignment.
        chrom_donor_alpha_nostr:    defaultdict = defaultdict(int)
        chrom_acceptor_alpha_nostr: defaultdict = defaultdict(int)
        for (iv_s, iv_e), count in _introns_nostr.items():
            chrom_donor_alpha_nostr[iv_s]    += count
            chrom_acceptor_alpha_nostr[iv_e] += count

        # Single streaming pass for beta counting.
        for read in bam.fetch(chrom):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < mapq_min:
                continue

            read_strand = _check_strand_from_flag(read.flag, strandedType)
            if read_strand not in ("+", "-"):
                continue

            # get_blocks() returns contiguous mapped intervals [(start, end), ...].
            # Introns are the gaps between consecutive blocks.
            blocks = read.get_blocks()
            if not blocks:
                continue
            introns = [(blocks[i][1], blocks[i + 1][0]) for i in range(len(blocks) - 1)]

            # Beta: check junction-file target sites in this read's span.
            targets = chrom_targets.get(read_strand, [])
            lo = bisect.bisect_left(targets, read.reference_start)
            hi = bisect.bisect_right(targets, read.reference_end - 1)

            for target_pos in targets[lo:hi]:
                for role in chrom_target_roles[read_strand][target_pos]:
                    key = (chrom, target_pos, read_strand, role)

                    is_in_block = any(bs < target_pos < be for bs, be in blocks)

                    if role == "donor":
                        # Alpha: intron starts exactly at target_pos.
                        # Naturally excluded from is_in_block (block ends at iv_s).
                        is_alpha_r = any(iv_s_r == target_pos for iv_s_r, _ in introns)
                        is_in_gap  = any(iv_s_r < target_pos < iv_e_r
                                         for iv_s_r, iv_e_r in introns)
                        d_partners = donor_to_acc[read_strand].get(target_pos, set())
                        d_comps    = donor_comps[read_strand].get(target_pos, set())
                        # compSplicing: read uses a junction (competing_donor → partner_acceptor).
                        comp_splicing = any(
                            iv_s_r in d_comps and iv_e_r in d_partners
                            for iv_s_r, iv_e_r in introns
                        )
                    else:  # acceptor
                        # Alpha: intron ends at target_pos + 1 (= iv_e).
                        is_alpha_r = any(iv_e_r == target_pos + 1 for _, iv_e_r in introns)
                        is_in_gap  = any(iv_s_r < target_pos < iv_e_r
                                         for iv_s_r, iv_e_r in introns)
                        a_partners = acc_to_donor[read_strand].get(target_pos, set())
                        a_comps    = acc_comps[read_strand].get(target_pos, set())
                        comp_splicing = any(
                            iv_s_r in a_partners and iv_e_r in a_comps
                            for iv_s_r, iv_e_r in introns
                        )

                    if not is_alpha_r:
                        if (is_in_block or is_in_gap) and comp_splicing:
                            # SimpleBeta2_flanking or SimpleBeta2_beta1type:
                            # read uses a competing junction while spanning target.
                            beta2_bam[key] += 1
                        elif is_in_gap:
                            # SimpleBeta2_mutuallyExclusive: spanning intron whose
                            # endpoints are both known sites (SpliSER Whiteset).
                            ws = whiteset[read_strand]
                            if any(
                                iv_s_r < target_pos < iv_e_r
                                and iv_s_r in ws and iv_e_r in ws
                                for iv_s_r, iv_e_r in introns
                            ):
                                beta2_bam[key] += 1
                        elif is_in_block:
                            beta1_bam[key] += 1

            # Strand-agnostic beta1: check ALL reads against the merged target set,
            # regardless of which strand identified the splice site.
            lo_ag = bisect.bisect_left(targets_ag_sorted, read.reference_start)
            hi_ag = bisect.bisect_right(targets_ag_sorted, read.reference_end - 1)
            for target_pos in targets_ag_sorted[lo_ag:hi_ag]:
                for role in chrom_targets_ag[target_pos]:
                    if role == "donor":
                        is_alpha_ag  = any(iv_s == target_pos for iv_s, _ in introns)
                        is_in_gap_ag = any(iv_s < target_pos < iv_e for iv_s, iv_e in introns)
                    else:
                        is_alpha_ag  = any(iv_e == target_pos + 1 for _, iv_e in introns)
                        is_in_gap_ag = any(iv_s < target_pos < iv_e for iv_s, iv_e in introns)
                    is_in_block_ag = any(bs < target_pos < be for bs, be in blocks)
                    if not is_alpha_ag and not is_in_gap_ag and is_in_block_ag:
                        beta1_ag_bam[(chrom, target_pos, role)] += 1

        for (pos, strand), count in chrom_donor_alpha.items():
            donor_alpha_bam[(chrom, pos, strand)] = count
            donor_alpha_ag_bam[(chrom, pos)] += count
        for (pos, strand), count in chrom_acceptor_alpha.items():
            acceptor_alpha_bam[(chrom, pos, strand)] = count
            acceptor_alpha_ag_bam[(chrom, pos)] += count
        for pos, count in chrom_donor_alpha_nostr.items():
            donor_alpha_nostr_bam[(chrom, pos)] += count
        for pos, count in chrom_acceptor_alpha_nostr.items():
            acceptor_alpha_nostr_bam[(chrom, pos)] += count

    bam.close()

    rows = []
    for (chrom, pos, strand), alpha in donor_alpha_bam.items():
        b1              = beta1_bam.get((chrom, pos, strand, "donor"), 0)
        b2              = beta2_bam.get((chrom, pos, strand, "donor"), 0)
        denom           = alpha + b1 + b2
        denom_b1        = alpha + b1
        a_ag            = donor_alpha_ag_bam.get((chrom, pos), 0)
        b1_ag           = beta1_ag_bam.get((chrom, pos, "donor"), 0)
        denom_ag        = a_ag + b1_ag
        a_nostr         = donor_alpha_nostr_bam.get((chrom, pos), 0)
        denom_nostr     = a_nostr + b1_ag
        denom_str_nostr = alpha + b1_ag
        rows.append({
            "chrom":              chrom,
            "position":           pos,
            "strand":             strand,
            "role":               "donor",
            "alpha_bam":          int(alpha),
            "beta1_bam":          int(b1),
            "beta2_bam":          int(b2),
            "ssu_spliser":        alpha   / denom           if denom           > 0 else float("nan"),
            "ssu_b1only":         alpha   / denom_b1        if denom_b1        > 0 else float("nan"),
            "alpha_ag":           int(a_ag),
            "beta1_ag":           int(b1_ag),
            "ssu_ag":             a_ag    / denom_ag        if denom_ag        > 0 else float("nan"),
            "ssu_str_a_nostr_b1": alpha   / denom_str_nostr if denom_str_nostr > 0 else float("nan"),
            "alpha_nostr":        int(a_nostr),
            "ssu_nostr":          a_nostr / denom_nostr     if denom_nostr     > 0 else float("nan"),
        })
    for (chrom, pos, strand), alpha in acceptor_alpha_bam.items():
        b1              = beta1_bam.get((chrom, pos, strand, "acceptor"), 0)
        b2              = beta2_bam.get((chrom, pos, strand, "acceptor"), 0)
        denom           = alpha + b1 + b2
        denom_b1        = alpha + b1
        a_ag            = acceptor_alpha_ag_bam.get((chrom, pos), 0)
        b1_ag           = beta1_ag_bam.get((chrom, pos, "acceptor"), 0)
        denom_ag        = a_ag + b1_ag
        a_nostr         = acceptor_alpha_nostr_bam.get((chrom, pos), 0)
        denom_nostr     = a_nostr + b1_ag
        denom_str_nostr = alpha + b1_ag
        rows.append({
            "chrom":              chrom,
            "position":           pos + 1,
            "strand":             strand,
            "role":               "acceptor",
            "alpha_bam":          int(alpha),
            "beta1_bam":          int(b1),
            "beta2_bam":          int(b2),
            "ssu_spliser":        alpha   / denom           if denom           > 0 else float("nan"),
            "ssu_b1only":         alpha   / denom_b1        if denom_b1        > 0 else float("nan"),
            "alpha_ag":           int(a_ag),
            "beta1_ag":           int(b1_ag),
            "ssu_ag":             a_ag    / denom_ag        if denom_ag        > 0 else float("nan"),
            "ssu_str_a_nostr_b1": alpha   / denom_str_nostr if denom_str_nostr > 0 else float("nan"),
            "alpha_nostr":        int(a_nostr),
            "ssu_nostr":          a_nostr / denom_nostr     if denom_nostr     > 0 else float("nan"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["chrom", "position", "strand", "role"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
# Step 4: assemble site table
# ------------------------------------------------------------------ #

def assemble_site_table(
    junctions: "pd.DataFrame",
    donor_alpha: "pd.Series",
    acceptor_alpha: "pd.Series",
    donor_beta2: "pd.Series",
    acceptor_beta2: "pd.Series",
) -> "pd.DataFrame":
    """Build one row per splice site with SSU scores.

    Args:
        junctions: Filtered junctions DataFrame (for intron coordinates).
        donor_alpha: Series indexed by (chrom, exon_start, strand).
        acceptor_alpha: Series indexed by (chrom, exon_end, strand).
        donor_beta2: Series indexed by (chrom, exon_start, strand).
        acceptor_beta2: Series indexed by (chrom, exon_end, strand).

    Returns:
        DataFrame with columns: chrom, strand, role, exon_pos, intron_pos,
        alpha_juncs, beta2_juncs, ssu_approx.
    """
    
    # Build a lookup from exon coord → intron coord
    donor_intron   = junctions.groupby(["chrom", "exon_start", "strand"])["intron_start"].first()
    acceptor_intron = junctions.groupby(["chrom", "exon_end",   "strand"])["intron_end"].first()

    rows: list[dict] = []

    for (chrom, exon_pos, strand), alpha in donor_alpha.items():
        intron_pos = int(donor_intron.get((chrom, exon_pos, strand), exon_pos + 1))
        b2 = int(donor_beta2.get((chrom, exon_pos, strand), 0))
        d_approx = alpha + b2
        rows.append({
            "chrom":        chrom,
            "strand":       strand,
            "role":         "donor",
            "exon_pos":     int(exon_pos),
            "intron_pos":   intron_pos,
            "alpha_juncs":  int(alpha),
            "beta2_juncs":  b2,
            "ssu_approx":   alpha / d_approx if d_approx > 0 else float("nan"),
        })

    for (chrom, exon_pos, strand), alpha in acceptor_alpha.items():
        intron_pos = int(acceptor_intron.get((chrom, exon_pos, strand), exon_pos - 1))
        b2 = int(acceptor_beta2.get((chrom, exon_pos, strand), 0))
        d_approx = alpha + b2
        rows.append({
            "chrom":        chrom,
            "strand":       strand,
            "role":         "acceptor",
            "exon_pos":     int(exon_pos),
            "intron_pos":   intron_pos,
            "alpha_juncs":  int(alpha),
            "beta2_juncs":  b2,
            "ssu_approx":   alpha / d_approx if d_approx > 0 else float("nan"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(
        subset=["chrom", "strand", "role", "exon_pos"]
    ).reset_index(drop=True)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.chroms:
        print(f"Restricting to chromosomes: {' '.join(args.chroms)}")
    print(f"Loading junctions from {args.junctions!r} …")
    junctions = load_junctions(args.junctions, args.min_unique_reads, args.chroms)
    print(f"  {len(junctions):,} junctions after quality filtering")

    if junctions.empty:
        print("No junctions remain after filtering. Check --min-unique-reads.", file=sys.stderr)
        sys.exit(1)

    print("Computing α and β2 …")
    donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2 = compute_alpha_beta2(junctions)
    print(f"  {len(donor_alpha):,} donor sites, {len(acceptor_alpha):,} acceptor sites")

    df_spliser = None
    if args.bam is not None:
        print(f"Computing SpliSER-equivalent (all counts from BAM) …")
        df_spliser = compute_spliser_counts(args.bam, junctions, args.mapq)
        print(f"  {len(df_spliser):,} sites with BAM counts")

    print("Assembling site table …")
    df = assemble_site_table(
        junctions,
        donor_alpha, acceptor_alpha,
        donor_beta2, acceptor_beta2,
    )
    print(f"  {len(df):,} unique splice sites")

    if df_spliser is not None and not df_spliser.empty:
        df = df.merge(
            df_spliser[["chrom", "position", "strand", "role",
                        "alpha_bam", "beta1_bam", "beta2_bam",
                        "ssu_spliser", "ssu_b1only",
                        "alpha_ag", "beta1_ag", "ssu_ag",
                        "ssu_str_a_nostr_b1",
                        "alpha_nostr", "ssu_nostr"]],
            left_on=["chrom", "exon_pos", "strand", "role"],
            right_on=["chrom", "position", "strand", "role"],
            how="left",
        ).drop(columns=["position"])
        denom = df["alpha_juncs"] + df["beta1_bam"].fillna(0) + df["beta2_juncs"]
        df["ssu_full"] = (df["alpha_juncs"] / denom).where(denom > 0)
    elif args.bam is not None:
        for col in ("alpha_bam", "beta1_bam", "beta2_bam",
                    "ssu_spliser", "ssu_b1only", "ssu_full",
                    "alpha_ag", "beta1_ag", "ssu_ag",
                    "ssu_str_a_nostr_b1",
                    "alpha_nostr", "ssu_nostr"):
            df[col] = float("nan")

    n_b2_zero = int((df["beta2_juncs"] == 0).sum())
    print(f"  sites with β2=0 (uncontested, ssu_approx=1.0): "
          f"{n_b2_zero} ({100 * n_b2_zero / len(df):.1f}%)")

    compression_arg = None if args.compression == "none" else args.compression
    df.to_parquet(out_path, index=False, compression=compression_arg)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
