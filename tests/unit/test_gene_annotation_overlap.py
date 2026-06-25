"""Unit tests for GeneAnnotation.get_genes_overlapping_variant (B2.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from alphagenome_pytorch.variant_scoring import Interval, Variant
from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation


def _make_annotation(tmp_path, rows):
    df = pd.DataFrame(rows)
    out = tmp_path / "anno.parquet"
    df.to_parquet(out)
    return GeneAnnotation(out)


@pytest.mark.unit
def test_get_genes_overlapping_variant_returns_only_overlapping(tmp_path):
    rows = [
        # Gene A spans [100, 200), Gene B spans [300, 400)
        {
            'Feature': 'gene', 'gene_id': 'A', 'gene_name': 'A',
            'gene_type': 'protein_coding', 'Chromosome': 'chr1',
            'Start': 100, 'End': 200, 'Strand': '+',
        },
        {
            'Feature': 'gene', 'gene_id': 'B', 'gene_name': 'B',
            'gene_type': 'protein_coding', 'Chromosome': 'chr1',
            'Start': 300, 'End': 400, 'Strand': '+',
        },
    ]
    anno = _make_annotation(tmp_path, rows)

    # Variant inside A only (1-based 150 -> 0-based 149)
    variant_in_a = Variant('chr1', 150, 'A', 'C')
    overlap = anno.get_genes_overlapping_variant(variant_in_a)
    assert overlap == ['A']

    # Compare against interval-spanning query: returns both
    interval_spanning = Interval('chr1', 50, 500)
    in_interval = anno.get_genes_in_interval(interval_spanning)
    assert set(in_interval) == {'A', 'B'}


@pytest.mark.unit
def test_get_genes_overlapping_variant_at_boundaries(tmp_path):
    rows = [
        {
            'Feature': 'gene', 'gene_id': 'G', 'gene_name': 'G',
            'gene_type': 'protein_coding', 'Chromosome': 'chr1',
            'Start': 100, 'End': 200, 'Strand': '+',
        },
    ]
    anno = _make_annotation(tmp_path, rows)

    # Variant at 0-based 100 (inclusive start) => overlaps
    v_start = Variant('chr1', 101, 'A', 'C')
    assert anno.get_genes_overlapping_variant(v_start) == ['G']

    # Variant at 0-based 199 (last included pos) => overlaps
    v_last = Variant('chr1', 200, 'A', 'C')
    assert anno.get_genes_overlapping_variant(v_last) == ['G']

    # Variant at 0-based 200 (exclusive end) => does NOT overlap
    v_end = Variant('chr1', 201, 'A', 'C')
    assert anno.get_genes_overlapping_variant(v_end) == []


@pytest.mark.unit
def test_get_genes_overlapping_variant_filters_gene_types(tmp_path):
    rows = [
        {
            'Feature': 'gene', 'gene_id': 'A', 'gene_name': 'A',
            'gene_type': 'protein_coding', 'Chromosome': 'chr1',
            'Start': 100, 'End': 200, 'Strand': '+',
        },
        {
            'Feature': 'gene', 'gene_id': 'B', 'gene_name': 'B',
            'gene_type': 'lncRNA', 'Chromosome': 'chr1',
            'Start': 100, 'End': 200, 'Strand': '+',
        },
    ]
    anno = _make_annotation(tmp_path, rows)

    v = Variant('chr1', 150, 'A', 'C')
    assert anno.get_genes_overlapping_variant(v, gene_types=['protein_coding']) == ['A']
    assert set(anno.get_genes_overlapping_variant(v)) == {'A', 'B'}
