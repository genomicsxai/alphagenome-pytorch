"""Tests for GeneAnnotation's performance indices (exon index + interval index).

These guard the optimizations behind the gene-expression metric:
  - `_get_exons_for_gene` builds a merged-exon index in one pass (behavior must
    match the old per-gene filter, including exon merging).
  - `get_genes_in_interval` uses a start-sorted per-chromosome index but must
    preserve the original `_gene_index` insertion order and overlap/chr-prefix
    semantics (it's shared with variant scoring).
  - `get_exon_bin_ranges` must be consistent with `get_exon_mask`.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from alphagenome_pytorch.variant_scoring import Interval
from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation


def _annotation(rows: list[dict]) -> GeneAnnotation:
    ann = GeneAnnotation(pd.DataFrame(rows))
    return ann


def _gene(gene_id, start, end, *, chrom="chr1", strand="+", gtype="protein_coding"):
    return dict(Feature="gene", Chromosome=chrom, Start=start, End=end, Strand=strand,
                gene_id=gene_id, gene_name=gene_id, gene_type=gtype)


def _exon(gene_id, start, end, *, chrom="chr1", strand="+", gtype="protein_coding"):
    return dict(Feature="exon", Chromosome=chrom, Start=start, End=end, Strand=strand,
                gene_id=gene_id, gene_name=gene_id, gene_type=gtype)


def _brute_overlap(ann, interval, gene_types=None):
    """Reference: original O(G) insertion-order overlap scan."""
    out = []
    for gid, info in ann._gene_index.items():
        chrom = interval.chromosome
        if info.chromosome != chrom and not (
            info.chromosome == "chr" + chrom or chrom == "chr" + info.chromosome
        ):
            continue
        if info.end <= interval.start or info.start >= interval.end:
            continue
        if gene_types is not None and info.gene_type not in gene_types:
            continue
        out.append(gid)
    return out


def test_get_genes_in_interval_preserves_insertion_order():
    # Insertion order deliberately differs from start-sorted order.
    ann = _annotation([
        _gene("LATE", 5000, 5200),
        _gene("EARLY", 100, 200),
        _gene("BIG", 0, 9000),   # long gene starting far before the query window
    ])
    iv = Interval("chr1", 4000, 6000)
    got = ann.get_genes_in_interval(iv)
    # BIG must be found despite starting at 0 (the max-gene-span lower bound),
    # and results come back in _gene_index insertion order: LATE before BIG.
    assert got == ["LATE", "BIG"]
    assert got == _brute_overlap(ann, iv)


def test_get_genes_in_interval_matches_bruteforce_over_many_windows():
    rows = [_gene(f"G{i}", start=i * 137 % 9000, end=i * 137 % 9000 + (i % 5) * 300 + 50)
            for i in range(60)]
    ann = _annotation(rows)
    for qs in range(0, 9000, 250):
        iv = Interval("chr1", qs, qs + 500)
        assert ann.get_genes_in_interval(iv) == _brute_overlap(ann, iv)


def test_get_genes_in_interval_chr_prefix_and_gene_types():
    ann = _annotation([
        _gene("A", 100, 200, chrom="1"),               # stored WITHOUT chr prefix
        _gene("B", 100, 200, chrom="chr1", gtype="lncRNA"),
    ])
    # Query 'chr1' still finds the '1'-stored gene, and vice versa.
    assert set(ann.get_genes_in_interval(Interval("chr1", 50, 300))) == {"A", "B"}
    assert set(ann.get_genes_in_interval(Interval("1", 50, 300))) == {"A", "B"}
    # gene_types filter preserved.
    assert ann.get_genes_in_interval(Interval("chr1", 50, 300), gene_types=["lncRNA"]) == ["B"]


def test_exon_index_merges_overlapping_exons():
    ann = _annotation([
        _gene("A", 100, 400),
        _exon("A", 100, 150),
        _exon("A", 140, 200),   # overlaps previous -> merge to (100, 200)
        _exon("A", 300, 350),
    ])
    assert ann._get_exons_for_gene("A") == [(100, 200), (300, 350)]
    # Versioned lookups collapse to the same base id.
    ann2 = _annotation([_gene("ENSG1.7", 100, 200), _exon("ENSG1.7", 100, 200)])
    assert ann2._get_exons_for_gene("ENSG1") == [(100, 200)]


def test_exon_bin_ranges_consistent_with_mask():
    ann = _annotation([
        _gene("A", 100, 400),
        _exon("A", 110, 130),
        _exon("A", 200, 264),
    ])
    iv = Interval("chr1", 100, 356)  # width 256
    for resolution, seq_length in ((1, 256), (128, 2)):
        ranges = ann.get_exon_bin_ranges("A", iv, resolution, seq_length)
        mask = ann.get_exon_mask("A", iv, resolution, seq_length)
        rebuilt = torch.zeros(seq_length, dtype=torch.bool)
        for b0, b1 in ranges:
            rebuilt[b0:b1] = True
        assert torch.equal(rebuilt, mask)


class TestDataFrameConstructor:
    """GeneAnnotation accepts a pre-filtered DataFrame, and rejects unusable ones.

    Filtering with pandas is the supported way to restrict an annotation, so the
    constructor has to validate what it is handed — a frame that lost its gene
    rows answers every lookup with an empty result rather than raising, which is
    the failure these tests pin down.
    """

    def test_dataframe_is_indexed_without_a_path(self):
        ann = GeneAnnotation(pd.DataFrame([_gene("ENSG1", 100, 200), _exon("ENSG1", 100, 150)]))
        assert ann.get_genes_in_interval(Interval("chr1", 0, 300)) == ["ENSG1"]
        assert ann._get_exons_for_gene("ENSG1") == [(100, 150)]
        # No file backs this annotation; the path aliases report that honestly.
        assert ann.annotation_path is None
        assert ann.gtf_path is None

    def test_pre_filtering_restricts_the_gene_set(self):
        rows = [
            _gene("ENSG1", 100, 200), _exon("ENSG1", 100, 150),
            _gene("ENSG2", 300, 400, gtype="lncRNA"), _exon("ENSG2", 300, 350, gtype="lncRNA"),
        ]
        df = pd.DataFrame(rows)
        # gene_type sits on gene and exon rows alike, so one predicate covers both.
        ann = GeneAnnotation(df[df.gene_type == "protein_coding"])
        assert ann.get_genes_in_interval(Interval("chr1", 0, 500)) == ["ENSG1"]
        assert ann._get_exons_for_gene("ENSG1") == [(100, 150)]
        assert ann._get_exons_for_gene("ENSG2") == []

    def test_filtering_away_gene_rows_raises(self):
        """The MANE_Select footgun: transcript-level tags never match gene rows."""
        df = pd.DataFrame([
            dict(_gene("ENSG1", 100, 200), tag=None),
            dict(_exon("ENSG1", 100, 150), tag="MANE_Select"),
        ])
        kept = df[df["tag"].astype(str).str.contains("MANE_Select")]
        assert len(kept) == 1 and (kept.Feature == "gene").sum() == 0  # gene row dropped
        with pytest.raises(ValueError, match="no `Feature == 'gene'` rows"):
            GeneAnnotation(kept)

    def test_missing_required_columns_raise(self):
        df = pd.DataFrame([_gene("ENSG1", 100, 200)]).drop(columns=["gene_id"])
        with pytest.raises(ValueError, match="missing required column"):
            GeneAnnotation(df)

    def test_gene_only_annotation_is_allowed(self):
        """Exon rows are optional — gene-body aggregation does not need them."""
        ann = GeneAnnotation(pd.DataFrame([_gene("ENSG1", 100, 200)]))
        assert ann.get_genes_in_interval(Interval("chr1", 0, 300)) == ["ENSG1"]
        assert not ann.has_exon_annotations()
