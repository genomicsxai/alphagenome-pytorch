"""Tests for the exon-based gene-expression validation metric glue.

The correlation math, exon masking, >=50% rule, and strand matching are covered
in ``test_aggregation.py``. Here we test the training-side glue that connects
``validate_multihead`` to those helpers: ``_accumulate_gene_expr_windows`` builds
per-window ``(gene_ids, pred, obs)`` triples from batched predictions + coords,
which ``combine_gene_expression`` then dedups and correlates.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from alphagenome_pytorch.aggregation import combine_gene_expression
from alphagenome_pytorch.extensions.finetuning.training import (
    _accumulate_gene_expr_windows,
    _gene_expr_metrics,
)
from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation

INTERVAL = ("chr1", 100, 120)  # width 20, 1bp resolution


def _make_annotation() -> GeneAnnotation:
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
    ]
    ann = GeneAnnotation(pd.DataFrame(rows))
    return ann


def _position_preds(batch=2, seq_len=20, n_tracks=3):
    """[B, S, C] where channel c = position index * (c + 1)."""
    pos = torch.arange(seq_len, dtype=torch.float32)
    one = torch.stack([pos * (c + 1) for c in range(n_tracks)], dim=-1)
    return one.unsqueeze(0).repeat(batch, 1, 1)


def test_accumulate_builds_windows_and_dedups():
    ann = _make_annotation()
    pred = _position_preds(batch=2)
    obs = pred.clone()  # perfect prediction -> correlation 1
    coords = [INTERVAL, INTERVAL]  # same window twice -> genes dedup across windows

    windows: list = []
    _accumulate_gene_expr_windows(
        windows,
        pred_unscaled=pred,
        targets=obs,
        coords=coords,
        annotation=ann,
        track_strands=None,
    )
    # One triple per batch window; each holds both genes (geneA, geneB).
    assert len(windows) == 2
    for gene_ids, p, o in windows:
        assert gene_ids == ["ENSGA", "ENSGB"]
        assert p.shape == (2, 3) and o.shape == (2, 3)
        # log1p of the exon-mean; geneA channel0 mean over exon positions = 4.5
        assert p[0, 0].item() == pytest.approx(math.log1p(4.5), abs=1e-5)

    out = combine_gene_expression(windows)
    assert out["n_genes"] == 2  # deduped across the two identical windows
    assert out["across_genes"] == pytest.approx(1.0, abs=1e-4)
    assert set(out) == {"across_genes", "across_genes_norm", "across_tracks_norm", "n_genes"}


def test_accumulate_strand_matching_nans_incompatible_cells():
    ann = _make_annotation()
    pred = _position_preds(batch=1)
    windows: list = []
    _accumulate_gene_expr_windows(
        windows,
        pred_unscaled=pred,
        targets=pred.clone(),
        coords=[INTERVAL],
        annotation=ann,
        track_strands=["+", "-", "."],
    )
    gene_ids, p, _ = windows[0]
    # geneA(+): '+' kept, '-' NaN, '.' kept.
    assert not torch.isnan(p[0, 0]) and torch.isnan(p[0, 1]) and not torch.isnan(p[0, 2])
    # geneB(-): '+' NaN, '-' kept.
    assert torch.isnan(p[1, 0]) and not torch.isnan(p[1, 1])


def test_accumulate_skips_windows_without_genes():
    ann = _make_annotation()
    pred = _position_preds(batch=2)
    # Second window is far from any annotated gene -> no qualifying genes.
    coords = [INTERVAL, ("chr1", 100_000, 100_020)]
    windows: list = []
    _accumulate_gene_expr_windows(
        windows,
        pred_unscaled=pred,
        targets=pred.clone(),
        coords=coords,
        annotation=ann,
        track_strands=None,
    )
    assert len(windows) == 1  # empty window skipped
    assert windows[0][0] == ["ENSGA", "ENSGB"]


def test_accumulate_window_cache_builds_once_per_window():
    ann = _make_annotation()
    pred = _position_preds(batch=2)  # both items are the same INTERVAL window
    coords = [INTERVAL, INTERVAL]
    cache: dict = {}
    windows: list = []
    _accumulate_gene_expr_windows(
        windows,
        pred_unscaled=pred,
        targets=pred.clone(),
        coords=coords,
        annotation=ann,
        track_strands=["+", "-", "."],
        window_cache=cache,
    )
    # Two batch items × (pred + obs) = 4 lookups, but a single unique window.
    assert len(cache) == 1
    assert len(windows) == 2
    # Reusing the same cache on a later "epoch" adds nothing new.
    _accumulate_gene_expr_windows(
        windows, pred_unscaled=pred, targets=pred.clone(), coords=coords,
        annotation=ann, track_strands=["+", "-", "."], window_cache=cache,
    )
    assert len(cache) == 1


# --------------------------------------------------------------------------- #
# _gene_expr_metrics: the reduce/emit step (single-rank path)
# --------------------------------------------------------------------------- #
def test_gene_expr_metrics_emits_named_keys():
    torch.manual_seed(4)
    truth = torch.randn(6, 4)
    pred = truth + 0.02 * torch.randn(6, 4)
    windows = [
        (["g0", "g1", "g2"], pred[:3], truth[:3]),
        (["g3", "g4", "g5"], pred[3:], truth[3:]),
    ]
    m = _gene_expr_metrics(windows, modality="rna_seq", world_size=1)
    assert set(m) == {
        "rna_seq_gene_log_expr_pearson_across_genes",
        "rna_seq_gene_log_expr_pearson_across_genes_norm",
        "rna_seq_gene_log_expr_pearson_across_tracks_norm",
        "rna_seq_gene_log_expr_n_genes",
    }
    assert m["rna_seq_gene_log_expr_n_genes"] == 6
    assert m["rna_seq_gene_log_expr_pearson_across_genes"] > 0.9


def test_gene_expr_metrics_modality_prefix_and_empty():
    m = _gene_expr_metrics([], modality="cage", world_size=1)
    assert m["cage_gene_log_expr_n_genes"] == 0
    assert math.isnan(m["cage_gene_log_expr_pearson_across_genes"])
