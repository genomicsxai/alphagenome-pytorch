"""Unit tests for the gene annotation extractor used by the gene LFC loss.

Covers `load_gene_table`, `GeneMaskExtractor.extract`, the LRU cache,
strand bucketing, the PAD_NUM_GENES_CEILING guard, and `derive_g_max`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphagenome_pytorch.extensions.finetuning.gene_annotation import (
    GeneMaskExtractor,
    PAD_NUM_GENES_CEILING,
    derive_g_max,
    load_gene_table,
)


def _toy_gene_table() -> pd.DataFrame:
    """Three protein-coding genes on chr1, one on chr2.

    Designed so a window chr1:1000-2000 fully contains GENE-A (+) and GENE-B (+),
    overlaps but does not contain GENE-C (- strand, extends past 2000), and
    excludes the chr2 gene entirely.
    """
    return pd.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr1", "chr2"],
            "Start": [1100, 1300, 1700, 5000],
            "End": [1200, 1500, 2500, 5500],  # GENE-C overhangs the test window
            "Strand": ["+", "+", "-", "+"],
            "gene_id": ["GENE-A", "GENE-B", "GENE-C", "GENE-D"],
            "gene_name": ["A", "B", "C", "D"],
            "gene_type": ["protein_coding"] * 4,
        }
    )


@pytest.mark.unit
class TestGeneMaskExtractor:

    def test_extract_returns_correct_shape(self):
        ex = GeneMaskExtractor(_toy_gene_table())
        mask, meta = ex.extract("chr1", 1000, 2000)
        assert mask.shape == (1000, 2, 2)  # GENE-A + GENE-B contained; GENE-C overhangs
        assert mask.dtype == bool
        assert list(meta["gene_id"]) == ["GENE-A", "GENE-B"]

    def test_strand_bucketing(self):
        """Plus-strand genes go in axis 1 = 0; minus-strand in axis 1 = 1."""
        # Use a window that fully contains a minus-strand gene by widening.
        table = _toy_gene_table().copy()
        table.loc[2, "End"] = 1900  # shrink GENE-C so it's contained in [1000, 2000)
        ex = GeneMaskExtractor(table)
        mask, meta = ex.extract("chr1", 1000, 2000)

        assert list(meta["gene_id"]) == ["GENE-A", "GENE-B", "GENE-C"]
        # GENE-A is plus → bucket 0
        gene_a_col = list(meta["gene_id"]).index("GENE-A")
        assert mask[:, 0, gene_a_col].any() and not mask[:, 1, gene_a_col].any()
        # GENE-C is minus → bucket 1
        gene_c_col = list(meta["gene_id"]).index("GENE-C")
        assert mask[:, 1, gene_c_col].any() and not mask[:, 0, gene_c_col].any()

    def test_unstranded_dot_appears_in_both_buckets(self):
        """A '.' (unstranded) gene must appear in BOTH strand buckets, matching
        upstream's strand-channel semantics."""
        table = pd.DataFrame(
            {
                "Chromosome": ["chr1"],
                "Start": [1100],
                "End": [1200],
                "Strand": ["."],
                "gene_id": ["GENE-DOT"],
                "gene_name": ["dot"],
                "gene_type": ["protein_coding"],
            }
        )
        ex = GeneMaskExtractor(table)
        mask, _ = ex.extract("chr1", 1000, 2000)
        assert mask.shape == (1000, 2, 1)
        # Both strand buckets should be populated identically.
        assert mask[:, 0, 0].any()
        assert mask[:, 1, 0].any()
        np.testing.assert_array_equal(mask[:, 0, 0], mask[:, 1, 0])

    def test_invalid_strand_raises(self):
        table = pd.DataFrame(
            {
                "Chromosome": ["chr1"],
                "Start": [1100],
                "End": [1200],
                "Strand": ["?"],
                "gene_id": ["GENE-Q"],
                "gene_name": ["q"],
                "gene_type": ["protein_coding"],
            }
        )
        ex = GeneMaskExtractor(table)
        with pytest.raises(ValueError, match="Strand must be one of"):
            ex.extract("chr1", 1000, 2000)

    def test_mask_positions_exact(self):
        """GENE-A spans [1100, 1200) genomic → [100, 200) relative to window 1000."""
        ex = GeneMaskExtractor(_toy_gene_table())
        mask, meta = ex.extract("chr1", 1000, 2000)
        gene_a_col = list(meta["gene_id"]).index("GENE-A")
        assert mask[:100, 0, gene_a_col].sum() == 0
        assert mask[100:200, 0, gene_a_col].all()
        assert mask[200:, 0, gene_a_col].sum() == 0

    def test_unknown_chromosome_returns_empty(self):
        ex = GeneMaskExtractor(_toy_gene_table())
        mask, meta = ex.extract("chrFAKE", 0, 1000)
        assert mask.shape == (1000, 2, 0)
        assert meta.empty

    def test_window_with_no_contained_genes(self):
        ex = GeneMaskExtractor(_toy_gene_table())
        mask, meta = ex.extract("chr1", 0, 500)  # before any gene
        assert mask.shape == (500, 2, 0)
        assert meta.empty

    def test_overhanging_gene_excluded(self):
        """GENE-C extends to 2500, window ends at 2000 → not contained → excluded."""
        ex = GeneMaskExtractor(_toy_gene_table())
        mask, meta = ex.extract("chr1", 1000, 2000)
        assert "GENE-C" not in list(meta["gene_id"])

    def test_lru_cache_hits(self):
        ex = GeneMaskExtractor(_toy_gene_table(), cache_size=2)
        m1, _ = ex.extract("chr1", 1000, 2000)
        m2, _ = ex.extract("chr1", 1000, 2000)  # cache hit
        # Same array object → cache returned the cached value, not recomputed.
        assert m1 is m2

    def test_cache_eviction(self):
        ex = GeneMaskExtractor(_toy_gene_table(), cache_size=1)
        ex.extract("chr1", 1000, 2000)
        ex.extract("chr1", 0, 500)  # evicts the first
        assert ("chr1", 1000, 2000) not in ex._cache
        assert ("chr1", 0, 500) in ex._cache

    def test_pad_ceiling_raises(self):
        """A window exceeding PAD_NUM_GENES_CEILING raises rather than truncating."""
        n = PAD_NUM_GENES_CEILING + 1
        table = pd.DataFrame(
            {
                "Chromosome": ["chr1"] * n,
                "Start": list(range(0, n * 10, 10)),
                "End": list(range(5, n * 10 + 5, 10)),
                "Strand": ["+"] * n,
                "gene_id": [f"G{i}" for i in range(n)],
                "gene_name": [f"g{i}" for i in range(n)],
                "gene_type": ["protein_coding"] * n,
            }
        )
        ex = GeneMaskExtractor(table)
        with pytest.raises(ValueError, match="exceeding the ceiling"):
            ex.extract("chr1", 0, n * 10 + 100)


@pytest.mark.unit
class TestDeriveGMax:

    def test_returns_exact_observed_max(self):
        """3 genes in window 1, 0 in window 2 → max=2 (GENE-A + GENE-B contained)."""
        ex = GeneMaskExtractor(_toy_gene_table())
        intervals = [("chr1", 1000, 2000), ("chr1", 0, 500)]
        g_max = derive_g_max(ex, intervals)
        assert g_max == 2

    def test_propagates_extractor_ceiling_error(self):
        """If a scanned window blows past the extractor's PAD ceiling,
        the error fires during the scan — derive_g_max needs no own cap."""
        n = PAD_NUM_GENES_CEILING + 1
        table = pd.DataFrame(
            {
                "Chromosome": ["chr1"] * n,
                "Start": list(range(0, n * 10, 10)),
                "End": list(range(5, n * 10 + 5, 10)),
                "Strand": ["+"] * n,
                "gene_id": [f"G{i}" for i in range(n)],
                "gene_name": [f"g{i}" for i in range(n)],
                "gene_type": ["protein_coding"] * n,
            }
        )
        ex = GeneMaskExtractor(table)
        with pytest.raises(ValueError, match="exceeding the ceiling"):
            derive_g_max(ex, [("chr1", 0, n * 10 + 100)])


@pytest.mark.unit
class TestLoadGeneTable:
    """Tests against a minimal synthetic GTF written to a tempfile."""

    @pytest.fixture
    def gtf_path(self):
        # GTF is 1-based inclusive in source format; pyranges converts to
        # 0-based half-open on load. We just write valid GTF text and let
        # pyranges do the conversion.
        gtf_text = (
            'chr1\tHAVANA\tgene\t1101\t1200\t.\t+\t.\t'
            'gene_id "GENE-A"; gene_name "A"; gene_type "protein_coding";\n'
            'chr1\tHAVANA\tgene\t1301\t1500\t.\t+\t.\t'
            'gene_id "GENE-B"; gene_name "B"; gene_type "lncRNA";\n'
            'chr1\tHAVANA\ttranscript\t1101\t1200\t.\t+\t.\t'
            'gene_id "GENE-A"; transcript_id "TX-A";\n'
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".gtf", mode="w", delete=False)
        tmp.write(gtf_text)
        tmp.close()
        try:
            yield tmp.name
        finally:
            os.unlink(tmp.name)

    def test_load_filters_to_protein_coding(self, gtf_path):
        df = load_gene_table(gtf_path, filter_protein_coding=True)
        assert list(df["gene_id"]) == ["GENE-A"]
        assert (df["gene_type"] == "protein_coding").all()

    def test_load_without_filter_keeps_all_genes(self, gtf_path):
        df = load_gene_table(gtf_path, filter_protein_coding=False)
        assert sorted(df["gene_id"]) == ["GENE-A", "GENE-B"]

    def test_load_drops_non_gene_rows(self, gtf_path):
        df = load_gene_table(gtf_path, filter_protein_coding=False)
        # Transcript row should not appear; we filter to Feature == 'gene'.
        # gene_id is unique here (one row per gene), so 2 genes → 2 rows.
        assert len(df) == 2

    def test_load_coordinates_zero_based(self, gtf_path):
        """pyranges converts 1-based inclusive (1101) → 0-based half-open (1100)."""
        df = load_gene_table(gtf_path, filter_protein_coding=True)
        assert df.iloc[0]["Start"] == 1100
        assert df.iloc[0]["End"] == 1200
