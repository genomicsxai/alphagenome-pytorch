"""Unit tests for the one-vs-rest log2 fold-change gene-expression metric.

Uses one-vs-rest ratios inspired by Borzoi Prime: per track,
the ratio of that track to the mean of the remaining tracks, formed within
predictions and within observations, then correlated across genes.
"""

from __future__ import annotations

import math

import pytest
import torch

from alphagenome_pytorch.aggregation import (
    combine_gene_expression,
    gene_expression_fold_change_correlations,
)


def _log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x)


@pytest.mark.unit
class TestOneVsRestFoldChange:

    def test_identical_inputs_correlate_perfectly(self):
        torch.manual_seed(0)
        x = _log1p(torch.rand(60, 3) * 20)
        out = gene_expression_fold_change_correlations(x, x)
        assert out["one_vs_rest_lfc"] == pytest.approx(1.0, abs=1e-5)
        assert all(v == pytest.approx(1.0, abs=1e-5) for v in out["one_vs_rest_lfc_per_track"])

    def test_inverted_specificity_anticorrelates(self):
        """A model that assigns the specificity to the wrong cell type should
        read negative, not merely lower. With two tracks each one's 'rest' is
        the other, so swapping the columns negates every fold change exactly."""
        torch.manual_seed(1)
        obs_lin = torch.rand(80, 2) * 10 + 1
        pred_lin = obs_lin.flip(dims=[1])
        out = gene_expression_fold_change_correlations(
            _log1p(pred_lin), _log1p(obs_lin)
        )
        assert out["one_vs_rest_lfc"] == pytest.approx(-1.0, abs=1e-5)

    def test_invariant_to_per_gene_scaling(self):
        """The ratio cancels anything shared by a gene's tracks -- sequencing
        depth, gene length, annotation span. That invariance is the reason this
        metric is unaffected by the sum-vs-mean aggregation choice."""
        torch.manual_seed(2)
        obs_lin = torch.rand(70, 4) * 10 + 1
        pred_lin = obs_lin * (0.5 + torch.rand(70, 4))
        base = gene_expression_fold_change_correlations(
            _log1p(pred_lin), _log1p(obs_lin)
        )
        # Scale each gene (all of its tracks) by a different constant.
        scale = (1.0 + torch.rand(70, 1) * 9.0)
        scaled = gene_expression_fold_change_correlations(
            _log1p(pred_lin * scale), _log1p(obs_lin * scale)
        )
        assert scaled["one_vs_rest_lfc"] == pytest.approx(
            base["one_vs_rest_lfc"], abs=1e-4
        )

    def test_strand_nans_split_the_gene_sets(self):
        """Sense matching NaNs the antisense cells, so a plus-strand gene's
        'rest' must contain only the other plus tracks -- no strand-pair
        bookkeeping required."""
        torch.manual_seed(3)
        x = _log1p(torch.rand(40, 4) * 10 + 1)
        x[:20, 1] = float("nan")   # plus genes: minus tracks absent
        x[:20, 3] = float("nan")
        x[20:, 0] = float("nan")   # minus genes: plus tracks absent
        x[20:, 2] = float("nan")
        out = gene_expression_fold_change_correlations(x, x)
        per_track = out["one_vs_rest_lfc_per_track"]
        assert len(per_track) == 4
        assert all(v == pytest.approx(1.0, abs=1e-5) for v in per_track)

    def test_single_track_is_nan(self):
        out = gene_expression_fold_change_correlations(
            torch.rand(10, 1), torch.rand(10, 1)
        )
        assert math.isnan(out["one_vs_rest_lfc"])
        assert len(out["one_vs_rest_lfc_per_track"]) == 1

    def test_too_few_genes_is_nan(self):
        out = gene_expression_fold_change_correlations(
            torch.rand(1, 3), torch.rand(1, 3)
        )
        assert math.isnan(out["one_vs_rest_lfc"])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match"):
            gene_expression_fold_change_correlations(
                torch.rand(10, 3), torch.rand(10, 2)
            )

    def test_zero_coverage_genes_drop_out(self):
        """Prime filters genes with zero coverage on either side before taking
        the ratio; a zero here must yield NaN for that gene, not an inf."""
        x = _log1p(torch.tensor([[0.0, 5.0, 7.0], [3.0, 4.0, 6.0], [2.0, 8.0, 1.0]]))
        out = gene_expression_fold_change_correlations(x, x)
        assert all(
            math.isnan(v) or -1.0 <= v <= 1.0
            for v in out["one_vs_rest_lfc_per_track"]
        )


@pytest.mark.unit
class TestCombineGeneExpressionSurfacesFoldChange:

    def test_keys_present(self):
        torch.manual_seed(4)
        pred = _log1p(torch.rand(6, 3) * 10)
        obs = _log1p(torch.rand(6, 3) * 10)
        gene_ids = [f"g{i}" for i in range(6)]
        out = combine_gene_expression([(gene_ids, pred, obs)])
        assert "one_vs_rest_lfc" in out
        assert len(out["one_vs_rest_lfc_per_track"]) == 3
        assert out["n_genes"] == 6

    def test_degenerate_window_still_reports_keys(self):
        """One gene means no correlation is defined, but the metric keys must
        still exist or the logger's column set changes between epochs."""
        out = combine_gene_expression([(["g0"], torch.rand(1, 3), torch.rand(1, 3))])
        assert math.isnan(out["one_vs_rest_lfc"])
        assert len(out["one_vs_rest_lfc_per_track"]) == 3
