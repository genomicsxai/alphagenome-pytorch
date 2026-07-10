"""Unit tests for the gene / interval aggregation module.

Covers the shared primitive, the Fig. 2d correlation helpers, and the two
serving helpers (`aggregate_genes` gene-body, `gene_expression` exon-based),
plus the `GeneCounts` converters. Uses dependency-free toy fixtures (no pyranges,
no bigwigs).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from alphagenome_pytorch.aggregation import (
    GeneCounts,
    aggregate_genes,
    aggregate_intervals,
    combine_gene_expression,
    gene_expression,
    gene_expression_correlations,
    gene_expression_values,
    normalize_expression,
)
from alphagenome_pytorch.named_outputs import TrackMetadata
from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
INTERVAL = ("chr1", 100, 120)  # width 20; use 1bp resolution (seq_len == 20)


def _position_preds(n_tracks=3, seq_len=20):
    """[1, S, C] where channel c = position index * (c + 1)."""
    pos = torch.arange(seq_len, dtype=torch.float32)
    return torch.stack([pos * (c + 1) for c in range(n_tracks)], dim=-1).unsqueeze(0)


def _gene_table():
    # geneA + on [102,108); geneB - on [110,116). Both fully inside [100,120).
    return pd.DataFrame({
        "Chromosome": ["chr1", "chr1"],
        "Start": [102, 110],
        "End": [108, 116],
        "Strand": ["+", "-"],
        "gene_id": ["ENSGA", "ENSGB"],
        "gene_name": ["A", "B"],
        "gene_type": ["protein_coding", "protein_coding"],
    })


def _make_annotation():
    """GeneAnnotation from an in-memory GTF-like frame (no file / pyranges)."""
    rows = [
        # geneA + : exons [102,104) and [106,108) (intron 104-106)
        dict(Feature="gene", Chromosome="chr1", Start=102, End=108, Strand="+",
             gene_id="ENSGA", gene_name="A", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=102, End=104, Strand="+",
             gene_id="ENSGA", gene_name="A", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=106, End=108, Strand="+",
             gene_id="ENSGA", gene_name="A", gene_type="protein_coding"),
        # geneB - : exon [110,116)
        dict(Feature="gene", Chromosome="chr1", Start=110, End=116, Strand="-",
             gene_id="ENSGB", gene_name="B", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=110, End=116, Strand="-",
             gene_id="ENSGB", gene_name="B", gene_type="protein_coding"),
        # geneC : mostly outside interval — total exon 20bp, only 2bp inside -> dropped by >=50% rule
        dict(Feature="gene", Chromosome="chr1", Start=118, End=200, Strand="+",
             gene_id="ENSGC", gene_name="C", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=118, End=128, Strand="+",
             gene_id="ENSGC", gene_name="C", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=190, End=200, Strand="+",
             gene_id="ENSGC", gene_name="C", gene_type="protein_coding"),
    ]
    df = pd.DataFrame(rows)
    ann = GeneAnnotation("/tmp/does_not_exist.parquet")  # suffix only; file never read
    ann._df = df
    ann._build_gene_index()
    return ann


def _tracks():
    return [
        TrackMetadata(0, "rna_seq", 0, "t0", {"strand": "+", "biosample": "liver"}),
        TrackMetadata(1, "rna_seq", 0, "t1", {"strand": "-", "biosample": "liver"}),
        TrackMetadata(2, "rna_seq", 0, "t2", {"strand": ".", "biosample": "brain"}),
    ]


# --------------------------------------------------------------------------- #
# primitive
# --------------------------------------------------------------------------- #
def test_aggregate_intervals_sum_mean():
    pred = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    mask = torch.zeros(6, 2)
    mask[0:3, 0] = 1.0
    mask[3:6, 1] = 1.0
    s = aggregate_intervals(pred, mask, "sum")
    m = aggregate_intervals(pred, mask, "mean")
    assert s.shape == (1, 2, 2)
    assert s[0, 0, 0] == 6 and m[0, 0, 0] == 2
    assert s[0, 1, 1] == 27 and m[0, 1, 1] == 9


def test_aggregate_intervals_empty_mask_and_2d():
    pred = torch.ones(1, 6, 1)
    empty = torch.zeros(6, 1)
    assert aggregate_intervals(pred, empty, "mean").abs().sum() == 0  # clamp, no NaN
    # 2D input gets a batch axis
    assert aggregate_intervals(pred[0], torch.ones(6, 1), "sum").shape == (1, 1, 1)


def test_aggregate_intervals_validation():
    with pytest.raises(ValueError):
        aggregate_intervals(torch.ones(1, 6, 1), torch.ones(5, 1))  # length mismatch
    with pytest.raises(ValueError):
        aggregate_intervals(torch.ones(1, 6, 1), torch.ones(6, 1), reduce="bogus")


# --------------------------------------------------------------------------- #
# correlation helpers
# --------------------------------------------------------------------------- #
def test_normalize_expression_gene_centered():
    torch.manual_seed(0)
    m = torch.randn(20, 4)
    n = normalize_expression(m)
    assert n.shape == (20, 4)
    assert n.mean(dim=1).abs().max() < 1e-5  # each gene's row mean is ~0


def test_gene_expression_correlations_three_flavors():
    torch.manual_seed(1)
    truth = torch.randn(30, 5)
    pred = truth + 0.05 * torch.randn(30, 5)
    d = gene_expression_correlations(pred, truth)
    assert set(d) == {"across_genes", "across_genes_norm", "across_tracks_norm"}
    assert d["across_genes"] > 0.9
    assert d["across_genes_norm"] > 0.8


def test_gene_expression_correlations_handles_nan():
    torch.manual_seed(2)
    truth = torch.randn(15, 3)
    pred = truth.clone()
    pred[0, 1] = float("nan")  # strand-incompatible cell
    d = gene_expression_correlations(pred, truth)
    assert not math.isnan(d["across_genes"])


# --------------------------------------------------------------------------- #
# aggregate_genes (gene-body)
# --------------------------------------------------------------------------- #
def test_aggregate_genes_body_mean():
    pred = _position_preds()
    gc = aggregate_genes(pred, _gene_table(), INTERVAL, track_metadata=_tracks())
    assert gc.space == "linear"
    assert gc.counts.shape == (1, 2, 3)  # 2 genes, 3 tracks
    # geneA body rel positions 2..7 -> channel0 mean = mean(2..7) = 4.5
    assert gc.counts[0, 0, 0].item() == pytest.approx(4.5)
    # channel1 doubles the position values -> 9.0
    assert gc.counts[0, 0, 1].item() == pytest.approx(9.0)
    # geneB body rel positions 10..15 -> channel0 mean = mean(10..15) = 12.5
    assert gc.counts[0, 1, 0].item() == pytest.approx(12.5)


def test_aggregate_genes_strand_modes():
    pred = _position_preds()
    tracks = _tracks()
    # default: no strand logic, full 3 columns, no NaN
    gc = aggregate_genes(pred, _gene_table(), INTERVAL, track_metadata=tracks)
    assert gc.counts.shape[-1] == 3 and not torch.isnan(gc.counts).any()

    # match: geneA(+) NaN on t1(-); geneB(-) NaN on t0(+); t2(.) always kept
    gm = aggregate_genes(pred, _gene_table(), INTERVAL, track_metadata=tracks, strand="match")
    assert torch.isnan(gm.counts[0, 0, 1])   # geneA x t1(-)
    assert not torch.isnan(gm.counts[0, 0, 0])  # geneA x t0(+)
    assert not torch.isnan(gm.counts[0, 0, 2])  # geneA x t2(.)
    assert torch.isnan(gm.counts[0, 1, 0])   # geneB x t0(+)

    # merge: liver +/- pair collapses; brain '.' stays -> 2 columns
    gmerge = aggregate_genes(pred, _gene_table(), INTERVAL, track_metadata=tracks, strand="merge")
    assert gmerge.counts.shape[-1] == 2


# --------------------------------------------------------------------------- #
# gene_expression (exon-based, log)
# --------------------------------------------------------------------------- #
def test_gene_expression_exon_log_and_50pct_rule():
    pred = _position_preds()
    ann = _make_annotation()
    ge = gene_expression(pred, ann, INTERVAL, track_metadata=_tracks())
    assert ge.space == "log"
    # geneC dropped by the >=50%-exon rule -> only geneA, geneB remain
    assert ge.counts.shape[1] == 2
    assert set(ge.gene_metadata["gene_id"]) == {"ENSGA", "ENSGB"}
    # geneA exon positions rel 2,3 (102-104) and 6,7 (106-108) -> channel0 mean = (2+3+6+7)/4 = 4.5
    row = ge.gene_metadata.index[ge.gene_metadata["gene_id"] == "ENSGA"][0]
    assert ge.counts[0, row, 0].item() == pytest.approx(math.log1p(4.5), abs=1e-5)


def test_gene_expression_linear_and_strand_default_match():
    pred = _position_preds()
    ann = _make_annotation()
    ge = gene_expression(pred, ann, INTERVAL, track_metadata=_tracks(), log=None)
    assert ge.space == "linear"
    # default strand="match": geneA(+) is NaN on the '-' track
    a_row = ge.gene_metadata.index[ge.gene_metadata["gene_id"] == "ENSGA"][0]
    minus_track = 1
    assert torch.isnan(ge.counts[0, a_row, minus_track])


def test_gene_expression_excludes_introns():
    # constant-1 preds: exon mean == 1 exactly; the intron positions (104-106)
    # must not change the mean (they're excluded from the mask).
    pred = torch.ones(1, 20, 1)
    ann = _make_annotation()
    ge = gene_expression(pred, ann, INTERVAL, track_metadata=None, log=None, strand=None)
    assert torch.allclose(ge.counts, torch.ones_like(ge.counts))


# --------------------------------------------------------------------------- #
# GeneCounts converters
# --------------------------------------------------------------------------- #
def test_gene_counts_to_tables_and_long():
    pred = _position_preds()
    gc = aggregate_genes(pred, _gene_table(), INTERVAL, track_metadata=_tracks())
    x, obs, var = gc.to_tables()
    assert x.shape == (3, 2)  # [tracks, genes]
    assert len(obs) == 3 and len(var) == 2
    df = gc.to_dataframe(long=True)
    assert len(df) == 2 * 3  # gene x track rows
    assert "count" in df.columns


def test_gene_counts_to_tables_requires_single_interval():
    pred = _position_preds().repeat(2, 1, 1)  # B=2
    gc = aggregate_genes(pred, _gene_table(), INTERVAL, track_metadata=_tracks())
    with pytest.raises(ValueError):
        gc.to_tables()


def test_gene_counts_to_anndata_optional():
    import warnings

    pred = _position_preds()
    gc = aggregate_genes(pred, _gene_table(), INTERVAL, track_metadata=_tracks())
    anndata = pytest.importorskip("anndata")
    with warnings.catch_warnings():
        warnings.simplefilter("error", anndata.ImplicitModificationWarning)
        adata = gc.to_anndata()  # must not warn about coercing the index to str
    assert adata.X.shape == (3, 2)
    assert adata.obs.shape[0] == 3 and adata.var.shape[0] == 2
    # idiomatic names: var_names = gene ids, obs_names = track names.
    assert list(adata.var_names) == ["ENSGA", "ENSGB"]
    assert list(adata.obs_names) == ["t0", "t1", "t2"]


# --------------------------------------------------------------------------- #
# validation-metric helpers (single-window values + cross-window combine)
# --------------------------------------------------------------------------- #
def test_gene_expression_values_shape_ids_and_strand():
    pred = _position_preds()
    ann = _make_annotation()
    values, gene_ids, gene_strands = gene_expression_values(
        pred, ann, INTERVAL, track_strands=["+", "-", "."]
    )
    # geneC dropped by the >=50%-exon rule -> geneA, geneB kept, in index order.
    assert gene_ids == ["ENSGA", "ENSGB"]
    assert gene_strands == ["+", "-"]
    assert values.shape == (2, 3)  # [G, C], log space
    # geneA(+): '+' track kept, '-' track NaN, '.' track kept.
    assert values[0, 0].item() == pytest.approx(math.log1p(4.5), abs=1e-5)
    assert torch.isnan(values[0, 1])           # geneA x '-' track
    assert not torch.isnan(values[0, 2])       # geneA x '.' track
    # geneB(-): '+' track NaN, '-' track kept.
    assert torch.isnan(values[1, 0])           # geneB x '+' track
    assert not torch.isnan(values[1, 1])


def test_gene_expression_values_no_strand_matching():
    pred = _position_preds()
    ann = _make_annotation()
    values, _, _ = gene_expression_values(pred, ann, INTERVAL, track_strands=None)
    assert not torch.isnan(values).any()       # no strand logic -> no NaN
    # linear-space check via log=None
    lin, _, _ = gene_expression_values(pred, ann, INTERVAL, log=None, track_strands=None)
    assert lin[0, 0].item() == pytest.approx(4.5, abs=1e-5)


def test_combine_gene_expression_dedup_and_corr():
    torch.manual_seed(3)
    truth = torch.randn(5, 4)
    pred = truth + 0.02 * torch.randn(5, 4)
    # window 1: genes g0..g2 ; window 2: genes g2..g4 (g2 overlaps -> deduped)
    w1 = (["g0", "g1", "g2"], pred[:3], truth[:3])
    w2 = (["g2", "g3", "g4"], pred[2:], truth[2:])
    out = combine_gene_expression([w1, w2])
    assert out["n_genes"] == 5                 # g2 counted once
    assert out["across_genes"] > 0.9
    assert set(out) == {"across_genes", "across_genes_norm", "across_tracks_norm", "n_genes"}


def test_combine_gene_expression_too_few_genes():
    out = combine_gene_expression([(["g0"], torch.zeros(1, 3), torch.zeros(1, 3))])
    assert out["n_genes"] == 1
    assert math.isnan(out["across_genes"])


def test_gene_expression_values_window_cache_reuse():
    pred = _position_preds()
    ann = _make_annotation()
    cache: dict = {}
    v1, ids1, s1 = gene_expression_values(pred, ann, INTERVAL, window_cache=cache)
    # Same window a second time (e.g. the obs pass, or a later epoch): one entry.
    v2, ids2, s2 = gene_expression_values(pred, ann, INTERVAL, window_cache=cache)
    assert len(cache) == 1
    assert ids1 == ids2 and s1 == s2
    # Cached result is identical to the uncached path (NaNs compare equal here).
    v_nocache, _, _ = gene_expression_values(pred, ann, INTERVAL)
    assert torch.equal(torch.nan_to_num(v1, nan=-1.0), torch.nan_to_num(v_nocache, nan=-1.0))
    # A different window adds a distinct cache entry.
    gene_expression_values(pred, ann, ("chr1", 100_000, 100_020), window_cache=cache)
    assert len(cache) == 2
