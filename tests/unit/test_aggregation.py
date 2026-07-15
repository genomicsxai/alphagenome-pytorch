"""Unit tests for the gene / interval aggregation module.

Covers the shared primitive, the gene-expression correlation helpers, and the two
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
    GeneCountAccumulator,
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
        # geneC : 0 of its 2 exons fall fully within [100,120) -> dropped by the >=50% count rule
        dict(Feature="gene", Chromosome="chr1", Start=118, End=200, Strand="+",
             gene_id="ENSGC", gene_name="C", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=118, End=128, Strand="+",
             gene_id="ENSGC", gene_name="C", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=190, End=200, Strand="+",
             gene_id="ENSGC", gene_name="C", gene_type="protein_coding"),
    ]
    df = pd.DataFrame(rows)
    ann = GeneAnnotation(df)
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


def test_gene_expression_50pct_rule_counts_contained_exons_not_basepairs():
    """The >=50% rule counts whole exons *contained* in the interval, not bases.

    ENSKEEP: 1 tiny in-window exon + 1 large out-of-window exon -> base-pair
    fraction ~2% (the old rule dropped it) but 1/2 = 50% of exons are contained,
    so the count rule keeps it. ENSDROP: 1 contained exon + a boundary-straddling
    exon + 1 fully-outside exon -> only 1/3 contained (< 50%), so it is dropped;
    the straddling exon is contained in neither side and does not count.
    """
    interval = ("chr1", 100, 120)
    rows = [
        dict(Feature="gene", Chromosome="chr1", Start=101, End=230, Strand="+",
             gene_id="ENSKEEP", gene_name="K", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=101, End=103, Strand="+",
             gene_id="ENSKEEP", gene_name="K", gene_type="protein_coding"),   # contained
        dict(Feature="exon", Chromosome="chr1", Start=130, End=230, Strand="+",
             gene_id="ENSKEEP", gene_name="K", gene_type="protein_coding"),   # outside
        dict(Feature="gene", Chromosome="chr1", Start=104, End=127, Strand="+",
             gene_id="ENSDROP", gene_name="D", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=104, End=106, Strand="+",
             gene_id="ENSDROP", gene_name="D", gene_type="protein_coding"),   # contained
        dict(Feature="exon", Chromosome="chr1", Start=118, End=122, Strand="+",
             gene_id="ENSDROP", gene_name="D", gene_type="protein_coding"),   # straddles 120
        dict(Feature="exon", Chromosome="chr1", Start=125, End=127, Strand="+",
             gene_id="ENSDROP", gene_name="D", gene_type="protein_coding"),   # outside
    ]
    ann = GeneAnnotation(pd.DataFrame(rows))

    _, gene_ids, _ = gene_expression_values(
        torch.ones(1, 20, 1), ann, interval, track_strands=None
    )
    assert gene_ids == ["ENSKEEP"]


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


# --------------------------------------------------------------------------- #
# GeneCountAccumulator (whole-chromosome streaming aggregation)
# --------------------------------------------------------------------------- #
def _two_tile_annotation():
    """geneG spans two tiles; geneH lives only in the second tile.

    geneG(+): exons [2,4) (tile A) and [12,15) (tile B); body [2,15).
    geneH(-): exon [16,18) (tile B only); body [16,18).
    """
    rows = [
        dict(Feature="gene", Chromosome="chr1", Start=2, End=15, Strand="+",
             gene_id="ENSG", gene_name="G", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=2, End=4, Strand="+",
             gene_id="ENSG", gene_name="G", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=12, End=15, Strand="+",
             gene_id="ENSG", gene_name="G", gene_type="protein_coding"),
        dict(Feature="gene", Chromosome="chr1", Start=16, End=18, Strand="-",
             gene_id="ENSH", gene_name="H", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=16, End=18, Strand="-",
             gene_id="ENSH", gene_name="H", gene_type="protein_coding"),
    ]
    ann = GeneAnnotation(pd.DataFrame(rows))
    return ann


def _coord_preds(start, end, n_tracks=1):
    """[n_bins, n_tracks] where every track's value == the genomic position."""
    pos = torch.arange(start, end, dtype=torch.float32)
    return torch.stack([pos for _ in range(n_tracks)], dim=-1)


def test_accumulator_sum_reconstructs_gene_across_tiles():
    ann = _two_tile_annotation()
    acc = GeneCountAccumulator(ann, resolution=1, over="exons", reduce="sum")
    # Tile A = genomic [0,10), tile B = [10,20); values == genomic position.
    acc.add_tile(_coord_preds(0, 10), "chr1", 0, 10)
    acc.add_tile(_coord_preds(10, 20), "chr1", 10, 20)

    gc = acc.to_gene_counts()
    assert list(gc.gene_metadata["gene_id"]) == ["ENSG", "ENSH"]  # first-seen order
    g = gc.gene_metadata.index[gc.gene_metadata["gene_id"] == "ENSG"][0]
    # geneG exon signal: tile A [2,4)->2+3=5 ; tile B [12,15)->12+13+14=39 ; total 44.
    assert gc.counts[0, g, 0].item() == pytest.approx(44.0)
    h = gc.gene_metadata.index[gc.gene_metadata["gene_id"] == "ENSH"][0]
    assert gc.counts[0, h, 0].item() == pytest.approx(16.0 + 17.0)


def test_accumulator_mean_and_log():
    ann = _two_tile_annotation()
    acc = GeneCountAccumulator(ann, resolution=1, over="exons", reduce="mean")
    acc.add_tile(_coord_preds(0, 10), "chr1", 0, 10)
    acc.add_tile(_coord_preds(10, 20), "chr1", 10, 20)
    gc = acc.to_gene_counts(log=True)
    g = gc.gene_metadata.index[gc.gene_metadata["gene_id"] == "ENSG"][0]
    # mean over 5 exon bases = 44/5 = 8.8 ; then log1p.
    assert gc.space == "log"
    assert gc.counts[0, g, 0].item() == pytest.approx(math.log1p(44.0 / 5.0), abs=1e-5)


def test_accumulator_gene_body_includes_introns():
    ann = _two_tile_annotation()
    acc = GeneCountAccumulator(ann, resolution=1, over="gene_body", reduce="sum")
    acc.add_tile(_coord_preds(0, 10), "chr1", 0, 10)
    acc.add_tile(_coord_preds(10, 20), "chr1", 10, 20)
    gc = acc.to_gene_counts()
    g = gc.gene_metadata.index[gc.gene_metadata["gene_id"] == "ENSG"][0]
    # body [2,15): sum of positions 2..14 == 104 (includes the [4,12) intron).
    assert gc.counts[0, g, 0].item() == pytest.approx(sum(range(2, 15)))


def test_accumulator_merges_bins_shared_by_exons_at_128bp():
    """Two exons landing in the same 128bp bin must not double-count that bin."""
    rows = [
        dict(Feature="gene", Chromosome="chr1", Start=10, End=150, Strand="+",
             gene_id="ENSG", gene_name="G", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=10, End=20, Strand="+",
             gene_id="ENSG", gene_name="G", gene_type="protein_coding"),   # -> bin 0
        dict(Feature="exon", Chromosome="chr1", Start=30, End=40, Strand="+",
             gene_id="ENSG", gene_name="G", gene_type="protein_coding"),   # -> bin 0 (shared)
        dict(Feature="exon", Chromosome="chr1", Start=140, End=150, Strand="+",
             gene_id="ENSG", gene_name="G", gene_type="protein_coding"),   # -> bin 1
    ]
    ann = GeneAnnotation(pd.DataFrame(rows))
    preds = torch.tensor([[5.0], [7.0]])  # [2 bins, 1 track]: bin0=5, bin1=7

    acc = GeneCountAccumulator(ann, resolution=128, over="exons", reduce="sum")
    acc.add_tile(preds, "chr1", 0, 256)  # 2 bins x 128bp
    # bin0 counted once despite two exons in it: 5 + 7 = 12 (not 5 + 5 + 7).
    assert acc.to_gene_counts().counts[0, 0, 0].item() == pytest.approx(12.0)

    acc_mean = GeneCountAccumulator(ann, resolution=128, over="exons", reduce="mean")
    acc_mean.add_tile(preds, "chr1", 0, 256)
    # Per-base mean over 2 distinct bins: 12 / (2 bins * 128 bases). Counting bin0
    # twice would instead give 17 / (3 * 128), so this still pins the dedup.
    assert acc_mean.to_gene_counts().counts[0, 0, 0].item() == pytest.approx(12.0 / (2 * 128))


def test_accumulator_strand_and_track_metadata():
    ann = _two_tile_annotation()
    acc = GeneCountAccumulator(ann, resolution=1, over="exons", reduce="sum")
    acc.add_tile(_coord_preds(0, 10, n_tracks=2), "chr1", 0, 10)
    acc.add_tile(_coord_preds(10, 20, n_tracks=2), "chr1", 10, 20)
    track_frame = pd.DataFrame({"track_index": [0, 1], "strand": ["+", "-"]})
    gc = acc.to_gene_counts(track_metadata=track_frame, strand="match")
    g = gc.gene_metadata.index[gc.gene_metadata["gene_id"] == "ENSG"][0]  # '+' gene
    assert not torch.isnan(gc.counts[0, g, 0])       # '+' track kept
    assert torch.isnan(gc.counts[0, g, 1])           # '-' track NaN'd


# --------------------------------------------------------------------------- #
# strand-label validation (API must agree with the CLI)
# --------------------------------------------------------------------------- #
def test_gene_expression_values_rejects_invalid_strand_labels():
    ann = _make_annotation()
    pred = _position_preds()  # 3 tracks
    with pytest.raises(ValueError, match="track strands must be"):
        gene_expression_values(pred, ann, INTERVAL, track_strands=["plus", "-", "."])


def test_gene_expression_match_rejects_invalid_track_metadata_strand():
    ann = _make_annotation()
    pred = _position_preds()
    bad_tracks = [
        TrackMetadata(0, "rna_seq", 0, "t0", {"strand": "+"}),
        TrackMetadata(1, "rna_seq", 0, "t1", {"strand": "?"}),   # invalid
        TrackMetadata(2, "rna_seq", 0, "t2", {"strand": "."}),
    ]
    with pytest.raises(ValueError, match="track strands must be"):
        gene_expression(pred, ann, INTERVAL, track_metadata=bad_tracks, strand="match")


def test_to_gene_counts_rejects_mismatched_dataframe_track_metadata():
    """A DataFrame must describe the accumulated tracks, like a TrackMetadata list does.

    Without the check the mismatch survives into GeneCounts and only surfaces
    later as a dimension error from inside anndata, about X vs obs rather than
    about the metadata the caller passed.
    """
    import pandas as pd

    ann = GeneAnnotation(pd.DataFrame([
        dict(Feature="gene", Chromosome="chr1", Start=0, End=10, Strand="+",
             gene_id="ENSG1", gene_name="G1", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=0, End=10, Strand="+",
             gene_id="ENSG1", gene_name="G1", gene_type="protein_coding"),
    ]))
    acc = GeneCountAccumulator(ann, resolution=1, over="exons", reduce="sum")
    acc.add_tile(torch.ones(10, 2), "chr1", 0, 10)  # 2 tracks

    bad = pd.DataFrame({"track_index": [0, 1, 2], "track_name": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="3 rows but predictions have 2 tracks"):
        acc.to_gene_counts(track_metadata=bad)

    ok = pd.DataFrame({"track_index": [0, 1], "track_name": ["a", "b"]})
    gc = acc.to_gene_counts(track_metadata=ok)
    assert gc.counts.shape[-1] == len(gc.track_metadata) == 2


# --------------------------------------------------------------------------- #
# Unit consistency between 1bp and 128bp (bin-sum inputs)
# --------------------------------------------------------------------------- #
def _per_base_signal(n_bases: int, seed: int = 0):
    """Nonconstant per-base coverage. Constant signal hides divisor errors."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_bases, 1, generator=g) * 10.0


def _to_bin_sums(per_base: torch.Tensor, resolution: int) -> torch.Tensor:
    """Bin the 1bp signal the way the pipeline does: sum the bases in each bin.

    Mirrors datasets.py (`reshape(output_len, res, n_tracks).sum(axis=1)`) and
    heads.predictions_scaling, which multiplies by `resolution` to reach
    experimental space. This is what makes 128bp values bin sums.
    """
    n, c = per_base.shape
    return per_base.reshape(n // resolution, resolution, c).sum(axis=1)


def _one_gene_annotation(exon_start: int, exon_end: int, gene_end: int | None = None):
    import pandas as pd

    end = gene_end if gene_end is not None else exon_end
    return GeneAnnotation(pd.DataFrame([
        dict(Feature="gene", Chromosome="chr1", Start=exon_start, End=end, Strand="+",
             gene_id="ENSG1", gene_name="G1", gene_type="protein_coding"),
        dict(Feature="exon", Chromosome="chr1", Start=exon_start, End=exon_end, Strand="+",
             gene_id="ENSG1", gene_name="G1", gene_type="protein_coding"),
    ]))


def test_aggregate_intervals_bin_size_normalizes_to_bases():
    per_base = _per_base_signal(1280)
    binned = _to_bin_sums(per_base, 128)          # [10, 1] bin sums

    mask_1bp = torch.ones(1280, 1)
    mask_128 = torch.ones(10, 1)

    # Sums agree regardless of resolution: a sum of sums is the same total.
    s1 = aggregate_intervals(per_base, mask_1bp, "sum")[0, 0, 0]
    s128 = aggregate_intervals(binned, mask_128, "sum")[0, 0, 0]
    assert torch.isclose(s1, s128, rtol=1e-5)

    # Per-base means agree only once bin_size converts elements -> bases.
    m1 = aggregate_intervals(per_base, mask_1bp, "mean", bin_size=1)[0, 0, 0]
    m128 = aggregate_intervals(binned, mask_128, "mean", bin_size=128)[0, 0, 0]
    assert torch.isclose(m1, m128, rtol=1e-5)
    assert torch.isclose(m1, per_base.mean(), rtol=1e-5)

    # Without bin_size the 128bp mean is a per-bin mean: 128x too large.
    wrong = aggregate_intervals(binned, mask_128, "mean")[0, 0, 0]
    assert torch.isclose(wrong, m1 * 128, rtol=1e-5)


def test_aggregate_intervals_rejects_nonpositive_bin_size():
    pred, mask = torch.ones(1, 4, 1), torch.ones(4, 1)
    for bad in (0, -1, -128):
        with pytest.raises(ValueError, match="bin_size must be positive"):
            aggregate_intervals(pred, mask, "mean", bin_size=bad)


def test_gene_expression_unit_consistent_for_bin_aligned_exons():
    """Exon on 128bp boundaries -> 1bp and 128bp agree, in linear and log space."""
    per_base = _per_base_signal(2560)
    binned = _to_bin_sums(per_base, 128)
    ann = _one_gene_annotation(0, 1280)           # exon [0,1280): bins [0,10)
    tracks = [TrackMetadata(0, "rna_seq", 0, "t+", {"strand": "+"})]
    iv = ("chr1", 0, 2560)
    truth = per_base[0:1280].mean()

    lin1 = gene_expression(per_base, ann, iv, track_metadata=tracks, log=None)
    lin128 = gene_expression(binned, ann, iv, track_metadata=tracks, log=None)
    assert torch.isclose(lin1.counts[0, 0, 0], truth, rtol=1e-5)
    assert torch.isclose(lin128.counts[0, 0, 0], truth, rtol=1e-5)

    # log1p(mean) must be taken after normalization, or the 128x lands inside the log.
    log1 = gene_expression(per_base, ann, iv, track_metadata=tracks)
    log128 = gene_expression(binned, ann, iv, track_metadata=tracks)
    assert torch.isclose(log1.counts[0, 0, 0], torch.log1p(truth), rtol=1e-5)
    assert torch.isclose(log128.counts[0, 0, 0], torch.log1p(truth), rtol=1e-5)


def test_gene_expression_128bp_is_approximate_for_unaligned_exons():
    """An exon off the bin grid cannot be recovered exactly from summed bins.

    get_exon_bin_ranges includes every bin the exon *touches*, and such a bin is
    an already-summed total mixing exonic and non-exonic bases. This pins the
    approximation as expected behavior rather than a regression.
    """
    per_base = _per_base_signal(2560, seed=1)
    binned = _to_bin_sums(per_base, 128)
    ann = _one_gene_annotation(100, 1200)         # straddles bins 0 and 9
    tracks = [TrackMetadata(0, "rna_seq", 0, "t+", {"strand": "+"})]
    iv = ("chr1", 0, 2560)

    exact = gene_expression(per_base, ann, iv, track_metadata=tracks, log=None)
    approx = gene_expression(binned, ann, iv, track_metadata=tracks, log=None)
    e, a = exact.counts[0, 0, 0], approx.counts[0, 0, 0]

    # Same order of magnitude (units are right) but not equal (mask is coarse).
    assert not torch.isclose(e, a, rtol=1e-4)
    assert 0.5 < (a / e).item() < 2.0


def test_accumulator_mean_is_per_base_at_128bp():
    per_base = _per_base_signal(1280, seed=2)
    binned = _to_bin_sums(per_base, 128)
    ann = _one_gene_annotation(0, 1280)
    truth = per_base[0:1280].mean()

    acc1 = GeneCountAccumulator(ann, resolution=1, over="exons", reduce="mean")
    acc1.add_tile(per_base, "chr1", 0, 1280)
    acc128 = GeneCountAccumulator(ann, resolution=128, over="exons", reduce="mean")
    acc128.add_tile(binned, "chr1", 0, 1280)

    v1 = acc1.to_gene_counts().counts[0, 0, 0]
    v128 = acc128.to_gene_counts().counts[0, 0, 0]
    assert torch.isclose(v1, truth, rtol=1e-5)
    assert torch.isclose(v128, truth, rtol=1e-5)


def test_gene_expression_values_unit_consistent_at_both_resolutions():
    per_base = _per_base_signal(2560, seed=3)
    binned = _to_bin_sums(per_base, 128)
    ann = _one_gene_annotation(0, 1280)
    iv = ("chr1", 0, 2560)
    truth = torch.log1p(per_base[0:1280].mean())

    v1, ids1, _ = gene_expression_values(per_base, ann, iv, track_strands=None)
    v128, ids128, _ = gene_expression_values(binned, ann, iv, track_strands=None)
    assert ids1 == ids128 == ["ENSG1"]
    assert torch.isclose(v1[0, 0], truth, rtol=1e-5)
    assert torch.isclose(v128[0, 0], truth, rtol=1e-5)
