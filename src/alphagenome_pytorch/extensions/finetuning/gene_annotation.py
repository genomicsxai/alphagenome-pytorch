"""Gene annotation extractor for fine-tuning datasets.

Builds per-interval gene-body boolean masks from a user-supplied GTF, for use
in the cross-track gene LFC training loss (mirrors AlphaGenome's upstream
`_GeneBodyAnnotationExtractor`).

The user MUST supply their own GTF — this module never downloads or relies on
hosted reference data. Reads GTFs via `pyranges.read_gtf`, which produces a
DataFrame with the columns the extractor expects (`Chromosome`, `Start`, `End`,
`Strand`, `Feature`, `gene_id`, `gene_type`, ...).
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd


# Maximum genes per training window. Mirrors upstream's
# `GeneVariantScorer.pad_num_genes=256` ceiling. If a window exceeds this we
# raise rather than silently truncate — same behavior as upstream.
PAD_NUM_GENES_CEILING = 256


def load_gene_table(
    gtf_path: str,
    *,
    filter_protein_coding: bool = True,
) -> pd.DataFrame:
    """Load a GTF as a gene-body DataFrame.

    Returns rows with `Feature == "gene"` and the seven columns
    `_GeneBodyAnnotationExtractor` requires.

    Args:
        gtf_path: Path to a GTF file readable by `pyranges.read_gtf`.
        filter_protein_coding: If True (default), keep only rows with
            `gene_type == "protein_coding"`. Set False for assemblies / GTFs
            that don't have or use that biotype.
    """
    import pyranges  # local import: heavy dep, only needed when GTF is supplied

    pr = pyranges.read_gtf(gtf_path)
    df = pr.df if hasattr(pr, "df") else pr  # pyranges 0.x → .df, 1.x → DataFrame

    required = {"Chromosome", "Start", "End", "Strand", "Feature", "gene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"GTF at {gtf_path} is missing required columns: {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}"
        )

    gene_rows = df[df["Feature"] == "gene"]
    if gene_rows.empty:
        raise ValueError(
            f"GTF at {gtf_path} contains no `Feature == 'gene'` rows. "
            "AlphaGenome gene LFC loss requires gene-level features; "
            "transcript-only or exon-only GTFs are not supported."
        )

    if filter_protein_coding:
        if "gene_type" not in gene_rows.columns:
            raise ValueError(
                f"filter_protein_coding=True requires a 'gene_type' column "
                f"in the GTF, but it was not found. Pass "
                f"filter_protein_coding=False to disable, or use a GTF with "
                f"biotype annotations."
            )
        gene_rows = gene_rows[gene_rows["gene_type"] == "protein_coding"]
        if gene_rows.empty:
            raise ValueError(
                f"No protein_coding genes found in {gtf_path}. "
                "Pass filter_protein_coding=False if your GTF uses different "
                "biotype labels."
            )

    keep_cols = ["Chromosome", "Start", "End", "Strand", "gene_id"]
    if "gene_name" in gene_rows.columns:
        keep_cols.append("gene_name")
    if "gene_type" in gene_rows.columns:
        keep_cols.append("gene_type")

    return gene_rows[keep_cols].reset_index(drop=True)


class GeneMaskExtractor:
    """Per-interval gene-body mask extractor.

    Mirrors upstream `_GeneBodyAnnotationExtractor` but:
      - sources the gene table from a user-supplied GTF (via `load_gene_table`),
        never from upstream's hosted feathers;
      - returns a strand-bucketed `[S, 2, G]` mask directly (upstream returns
        `[S, G]` and lets the caller bucket by strand). Bucketing here keeps
        the loss-side einsum simple.

    The mask uses 0-based, half-open coordinates (consistent with `pyranges`
    output and the rest of this codebase).

    Args:
        gene_table: DataFrame from `load_gene_table` (or an equivalent).
        cache_size: LRU cache size on (chromosome, start, end). Larger is
            useful at training time when intervals repeat across epochs.
            Defaults to 1024.
    """

    def __init__(self, gene_table: pd.DataFrame, *, cache_size: int = 1024):
        self._gene_table = gene_table
        self._cache_size = cache_size
        self._cache: "OrderedDict[Tuple[str, int, int], Tuple[np.ndarray, pd.DataFrame]]" = (
            OrderedDict()
        )
        # Pre-group by chromosome for fast filtering.
        self._by_chrom: dict[str, pd.DataFrame] = {
            str(chrom): grp
            for chrom, grp in gene_table.groupby("Chromosome", observed=True)
        }

    def extract(
        self,
        chromosome: str,
        start: int,
        end: int,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Extract `[S, 2, G]` gene-body mask for a half-open interval.

        Args:
            chromosome: Chromosome name (must match GTF naming, e.g. "chr1").
            start: 0-based inclusive start.
            end: 0-based exclusive end.

        Returns:
            Tuple `(mask, metadata)`:
              - mask: bool array of shape `(end - start, 2, G)`. Axis 1 is
                `[plus_strand_genes, minus_strand_genes]`. `G` is the number
                of genes contained in the interval (variable; may be 0).
              - metadata: DataFrame slice for the contained genes (one row
                per column of axis 2, in the same order).
        """
        key = (chromosome, int(start), int(end))
        if key in self._cache:
            self._cache.move_to_end(key)
            mask, metadata = self._cache[key]
            return mask, metadata

        width = end - start
        chrom_df = self._by_chrom.get(chromosome)
        if chrom_df is None or chrom_df.empty:
            mask = np.zeros((width, 2, 0), dtype=bool)
            metadata = self._gene_table.iloc[:0].copy()
            self._cache_set(key, mask, metadata)
            return mask, metadata

        # Genes fully contained in the interval (matches upstream's
        # INTERVAL_CONTAINED query type). Genes that overhang are skipped
        # since their per-gene total would be biased.
        contained = chrom_df[
            (chrom_df["Start"].values >= start) & (chrom_df["End"].values <= end)
        ]

        num_genes = len(contained)
        if num_genes > PAD_NUM_GENES_CEILING:
            raise ValueError(
                f"Window {chromosome}:{start}-{end} contains {num_genes} genes, "
                f"exceeding the ceiling of {PAD_NUM_GENES_CEILING}. "
                "Either reduce the window size or relax the ceiling in "
                "alphagenome_pytorch.extensions.finetuning.gene_annotation."
            )

        mask = np.zeros((width, 2, num_genes), dtype=bool)
        for i, row in enumerate(contained.itertuples(index=False)):
            rel_start = max(int(row.Start) - start, 0)
            rel_end = min(int(row.End) - start, width)
            strand = str(row.Strand)
            # Match upstream's strand-channel semantics: "+" → bucket 0 only,
            # "-" → bucket 1 only, "." (unstranded) → BOTH buckets, so an
            # unstranded gene aggregates from + and - tracks alike.
            if strand in ("+", "."):
                mask[rel_start:rel_end, 0, i] = True
            if strand in ("-", "."):
                mask[rel_start:rel_end, 1, i] = True
            if strand not in ("+", "-", "."):
                raise ValueError(
                    f"Unknown Strand value {strand!r} for gene "
                    f"{getattr(row, 'gene_id', '<unknown>')!r}. "
                    "Strand must be one of '+', '-', '.'."
                )

        metadata = contained.reset_index(drop=True)
        self._cache_set(key, mask, metadata)
        return mask, metadata

    def _cache_set(
        self,
        key: Tuple[str, int, int],
        mask: np.ndarray,
        metadata: pd.DataFrame,
    ) -> None:
        self._cache[key] = (mask, metadata)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def derive_g_max(
    extractor: GeneMaskExtractor,
    intervals: list[Tuple[str, int, int]],
    *,
    headroom: int = 16,
) -> int:
    """Scan training intervals once to pick a fixed `G_max`.

    Returns `observed_max + headroom`. Any window exceeding
    `PAD_NUM_GENES_CEILING` raises during the scan via the extractor's own
    bound check; this function therefore needs no additional cap.

    Reuses the extractor's LRU cache so masks built here are still warm
    when training begins.
    """
    observed_max = 0
    for chrom, start, end in intervals:
        mask, _ = extractor.extract(chrom, start, end)
        observed_max = max(observed_max, mask.shape[-1])
    return observed_max + headroom


@functools.lru_cache(maxsize=4)
def cached_load_gene_table(gtf_path: str, filter_protein_coding: bool = True) -> pd.DataFrame:
    """Module-level cache for `load_gene_table`.

    GTF parsing is expensive (~seconds for hg38). When the same script
    constructs train + val datasets back-to-back from the same GTF, this
    avoids re-parsing.
    """
    return load_gene_table(gtf_path, filter_protein_coding=filter_protein_coding)


__all__ = [
    "GeneMaskExtractor",
    "load_gene_table",
    "cached_load_gene_table",
    "derive_g_max",
    "PAD_NUM_GENES_CEILING",
]
