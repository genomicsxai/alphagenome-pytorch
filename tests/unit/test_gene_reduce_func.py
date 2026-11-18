"""Unit tests for the within-gene reduction (``gene_reduce_func``).

One setting decides how signal is reduced inside a gene body: ``mean`` divides
by gene length (full-length libraries), ``sum`` does not (3'-biased libraries
such as 10x pseudobulk). It has to reach two places, or a run optimizes one
quantity and is scored on another.

Covers:
  - `gene_lfc_loss(reduce_func=...)`: 'mean' divides by gene length, while
    'sum' computes the loss from gene totals and applies a fixed scale.
  - Resolution in the CLI layer: default 'mean', per-head config, CLI
    override.
  - The same value reaching the gene-expression validation metric, which
    reports both reductions and uses the head's own for the unsuffixed keys.
"""

from __future__ import annotations

import math

import pytest
import torch

from alphagenome_pytorch.losses import gene_lfc_loss
from alphagenome_pytorch.training import _build_strand_channel_mask


def _inputs(gene_spans, B=1, S=128, C=4, seed=0):
    """Build (preds, targets, track_mask, gene_mask) with the given gene spans.

    `gene_spans` is a list of (strand_bucket, start, end); each becomes one
    column of the `[B, S, 2, G]` gene mask.
    """
    torch.manual_seed(seed)
    preds = torch.rand(B, S, C) + 0.5
    targets = torch.rand(B, S, C) + 0.5
    track_mask = torch.ones(B, 1, C, dtype=torch.bool)
    gene_mask = torch.zeros(B, S, 2, len(gene_spans), dtype=torch.bool)
    for g, (bucket, start, end) in enumerate(gene_spans):
        gene_mask[:, start:end, bucket, g] = True
    return preds, targets, track_mask, gene_mask


@pytest.mark.unit
class TestGeneLossReduceFunc:

    def _loss(self, gene_spans, reduce_func, **kw):
        preds, targets, track_mask, gene_mask = _inputs(gene_spans, **kw)
        strand_mask = _build_strand_channel_mask(["+", "-", "+", "-"])
        loss, aux = gene_lfc_loss(
            predictions=preds,
            targets=targets,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_mask,
            reduce_func=reduce_func,
        )
        return loss, aux

    def test_unequal_length_genes_differ(self):
        """Once lengths differ, 'mean' down-weights the long gene and 'sum'
        does not -- the whole point of the setting."""
        spans = [(0, 10, 30), (1, 40, 120)]  # 20bp vs 80bp
        mean_loss, _ = self._loss(spans, "mean")
        sum_loss, _ = self._loss(spans, "sum")
        assert not torch.isclose(mean_loss, sum_loss), (
            "reduce_func made no difference despite a 4x gene-length spread"
        )

    def test_mean_is_the_default(self):
        """Unspecified must stay on the upstream-JAX path, or every existing
        run's loss silently changes value."""
        spans = [(0, 10, 30), (1, 40, 120)]
        preds, targets, track_mask, gene_mask = _inputs(spans)
        strand_mask = _build_strand_channel_mask(["+", "-", "+", "-"])
        common = dict(
            predictions=preds,
            targets=targets,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_mask,
        )
        torch.testing.assert_close(
            gene_lfc_loss(**common)[0], gene_lfc_loss(**common, reduce_func="mean")[0]
        )

    def test_sum_stays_finite_and_differentiable(self):
        spans = [(0, 10, 30), (1, 40, 120)]
        preds, targets, track_mask, gene_mask = _inputs(spans)
        preds = preds.clone().requires_grad_(True)
        loss, _ = gene_lfc_loss(
            predictions=preds,
            targets=targets,
            targets_mask=track_mask,
            gene_mask=gene_mask,
            strand_channel_mask=_build_strand_channel_mask(["+", "-", "+", "-"]),
            reduce_func="sum",
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(preds.grad).all()
        assert preds.grad.abs().sum() > 0

    def test_sum_uses_a_fixed_scale_across_windows(self):
        preds = torch.zeros(1, 10_000, 2)
        targets = torch.zeros_like(preds)
        preds[0, 0] = torch.tensor([2.0, 3.0])
        targets[0, 0] = torch.tensor([1.0, 5.0])
        preds.requires_grad_()

        def loss_and_gradient(length, reduce_func):
            mask = torch.zeros(1, 10_000, 2, 1, dtype=torch.bool)
            mask[0, :length, 0, 0] = True
            loss, _ = gene_lfc_loss(
                predictions=preds,
                targets=targets,
                targets_mask=None,
                gene_mask=mask,
                strand_channel_mask=_build_strand_channel_mask(["+", "+"]),
                reduce_func=reduce_func,
            )
            gradient, = torch.autograd.grad(loss, preds)
            return loss, gradient[0, 0]

        # A mean over one base feeds raw totals into both loss terms.
        raw_loss, raw_gradient = loss_and_gradient(1, "mean")
        for length in (100, 10_000):
            loss, gradient = loss_and_gradient(length, "sum")
            torch.testing.assert_close(loss, raw_loss / 10_000)
            torch.testing.assert_close(gradient, raw_gradient / 10_000)

    def test_no_genes_in_window_is_safe(self):
        """Empty masks happen at window edges; the scalar divisor must not
        become 0/0."""
        preds, targets, track_mask, _ = _inputs([(0, 10, 30)])
        empty = torch.zeros(1, 128, 2, 1, dtype=torch.bool)
        loss, _ = gene_lfc_loss(
            predictions=preds,
            targets=targets,
            targets_mask=track_mask,
            gene_mask=empty,
            strand_channel_mask=_build_strand_channel_mask(["+", "-", "+", "-"]),
            reduce_func="sum",
        )
        assert torch.isfinite(loss)

    def test_invalid_reduce_func_raises(self):
        spans = [(0, 10, 30)]
        with pytest.raises(ValueError, match="reduce_func must be"):
            self._loss(spans, "median")


def _parse(argv, tmp_path, config: str | None = None):
    """Run the finetune parser over a minimal valid argument set."""
    from alphagenome_pytorch.extensions.finetuning.args import parse_args

    base = [
        "--genome", "g.fa", "--train-bed", "t.bed", "--val-bed", "v.bed",
        "--pretrained-weights", "m.pth",
    ]
    if config is not None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(config)
        base += ["--config", str(cfg_path)]
    return parse_args(base + argv)


@pytest.mark.unit
class TestGeneReduceFuncResolution:
    """Precedence: --gene-reduce-func > per-head config > 'mean'."""

    CLI = ["--modality", "rna_seq", "--bigwig", "a+.bw", "a-.bw"]

    def test_defaults_to_mean(self, tmp_path):
        args = _parse(self.CLI, tmp_path)
        assert args.modality_gene_reduce_funcs == {"rna_seq": "mean"}

    def test_per_head_config_selects_sum(self, tmp_path):
        args = _parse(
            [], tmp_path,
            config=(
                "modalities:\n"
                "  rna_seq:\n"
                "    bigwig: [a+.bw, a-.bw]\n"
                "    gene_reduce_func: sum\n"
            ),
        )
        assert args.modality_gene_reduce_funcs == {"rna_seq": "sum"}

    def test_cli_flag_overrides_config(self, tmp_path):
        args = _parse(
            ["--gene-reduce-func", "mean"], tmp_path,
            config=(
                "modalities:\n"
                "  rna_seq:\n"
                "    bigwig: [a+.bw, a-.bw]\n"
                "    gene_reduce_func: sum\n"
            ),
        )
        assert args.modality_gene_reduce_funcs == {"rna_seq": "mean"}

    def test_cli_flag_applies_to_every_head(self, tmp_path):
        args = _parse(
            self.CLI + ["--modality", "atac", "--bigwig", "x.bw",
                        "--gene-reduce-func", "sum"],
            tmp_path,
        )
        assert args.modality_gene_reduce_funcs == {"rna_seq": "sum", "atac": "sum"}

    def test_invalid_top_level_config_value_is_rejected(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _parse(self.CLI, tmp_path, config="gene_reduce_func: sums\n")
        assert "--gene-reduce-func must be 'auto', 'mean', or 'sum'" in capsys.readouterr().err

    def test_cli_flag_overrides_invalid_top_level_config_value(self, tmp_path):
        args = _parse(
            self.CLI + ["--gene-reduce-func", "sum"], tmp_path,
            config="gene_reduce_func: sums\n",
        )
        assert args.modality_gene_reduce_funcs == {"rna_seq": "sum"}

    def test_only_explicit_strand_pairs_preselect_gene_expression_partners(self, tmp_path):
        auto = _parse(self.CLI + ["--strand-pairs", "rna_seq:auto"], tmp_path)
        explicit = _parse(self.CLI + ["--strand-pairs", "rna_seq:0,1"], tmp_path)
        assert auto.modality_strand_pairs == explicit.modality_strand_pairs
        assert auto.modality_explicit_strand_pairs == {}
        assert explicit.modality_explicit_strand_pairs == {"rna_seq": [(0, 1)]}

    def test_invalid_config_value_is_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            _parse(
                [], tmp_path,
                config=(
                    "modalities:\n"
                    "  rna_seq:\n"
                    "    bigwig: [a.bw]\n"
                    "    gene_reduce_func: median\n"
                ),
            )


@pytest.mark.unit
class TestResolveGeneLossSettings:
    """The reduction that actually reaches the loss, per head."""

    def _args(self, **kw):
        import argparse

        base = dict(
            gene_loss_weight=0.1,
            modality_to_bigwigs={"rna_seq": ["a+", "a-"]},
            modality_strands={"rna_seq": "+-"},
            modality_gene_reduce_funcs={"rna_seq": "sum"},
        )
        base.update(kw)
        return argparse.Namespace(**base)

    def _resolve(self, args):
        from alphagenome_pytorch.extensions.finetuning.runner import (
            resolve_gene_loss_settings,
        )

        return resolve_gene_loss_settings(args)

    def test_reads_the_resolved_per_head_value(self):
        weights, reduce_funcs = self._resolve(self._args())
        assert weights == {"rna_seq": 0.1}
        assert reduce_funcs == {"rna_seq": "sum"}

    def test_falls_back_to_mean(self):
        _, reduce_funcs = self._resolve(self._args(modality_gene_reduce_funcs={}))
        assert reduce_funcs == {"rna_seq": "mean"}

    def test_non_rna_modalities_excluded(self):
        weights, reduce_funcs = self._resolve(
            self._args(
                modality_to_bigwigs={"atac": ["x"], "rna_seq": ["c+", "c-"]},
                modality_strands={"rna_seq": "+-"},
            )
        )
        assert "atac" not in weights and "atac" not in reduce_funcs

    def test_unstranded_rna_head_excluded(self):
        """No strand info means no strand-channel mask, so the gene term cannot
        run for that head -- it must not get a weight either."""
        weights, _ = self._resolve(self._args(modality_strands={}))
        assert weights == {}

    def test_disabled_gene_loss_yields_empty(self):
        weights, reduce_funcs = self._resolve(self._args(gene_loss_weight=0.0))
        assert weights == {} and reduce_funcs == {}


@pytest.mark.unit
class TestGeneExprMetricReduction:
    """The validation metric has the same length question as the loss."""

    def _windows(self, scale: float = 1.0):
        """Two windows of (gene_ids, pred[G, C], obs[G, C]) in log1p space."""
        torch.manual_seed(0)
        out = []
        for w, ids in enumerate([["g1", "g2", "g3"], ["g4", "g5", "g6"]]):
            obs = torch.rand(len(ids), 4) * 3.0 + 0.5
            pred = obs + torch.randn(len(ids), 4) * 0.2
            out.append((ids, (pred * scale).abs(), (obs * scale).abs()))
        return out

    def _metrics(self, primary: str):
        from alphagenome_pytorch.extensions.finetuning.training import (
            GENE_EXPR_REDUCE_FUNCS,
            _gene_expr_metrics,
        )

        windows = {r: self._windows() for r in GENE_EXPR_REDUCE_FUNCS}
        return _gene_expr_metrics(windows, modality="rna_seq", primary=primary)

    def test_both_reductions_are_reported(self):
        m = self._metrics("mean")
        prefix = "rna_seq_gene_log_expr_pearson"
        for reduce_func in ("mean", "sum"):
            for key in ("across_genes", "across_genes_norm", "across_tracks_norm"):
                assert f"{prefix}_{reduce_func}_{key}" in m

    @pytest.mark.parametrize("primary", ["mean", "sum"])
    def test_unsuffixed_keys_follow_the_head(self, primary):
        m = self._metrics(primary)
        prefix = "rna_seq_gene_log_expr_pearson"
        assert m[f"{prefix}_across_genes"] == m[f"{prefix}_{primary}_across_genes"]

    def test_one_vs_rest_is_emitted_once(self):
        """It is invariant under the reduction, so a second copy would be a
        duplicate series in every dashboard."""
        m = self._metrics("sum")
        prefix = "rna_seq_gene_log_expr_pearson"
        assert f"{prefix}_one_vs_rest_lfc" in m
        assert f"{prefix}_mean_one_vs_rest_lfc" not in m
        assert f"{prefix}_sum_one_vs_rest_lfc" not in m

    def test_one_vs_rest_invariance_holds(self):
        """Why the single copy is correct: every track of a gene is divided by
        the same gene length, so it cancels in the one-vs-rest ratio."""
        from alphagenome_pytorch.aggregation import (
            gene_expression_fold_change_correlations,
        )

        torch.manual_seed(0)
        G, C = 40, 5
        lengths = (torch.rand(G, 1) * 4.0 + 1.0)  # per-gene divisor
        obs_sum = torch.rand(G, C) * 50.0 + 1.0
        pred_sum = obs_sum + torch.randn(G, C).abs()

        as_sum = gene_expression_fold_change_correlations(
            torch.log1p(pred_sum), torch.log1p(obs_sum)
        )
        as_mean = gene_expression_fold_change_correlations(
            torch.log1p(pred_sum / lengths), torch.log1p(obs_sum / lengths)
        )
        assert math.isclose(
            as_sum["one_vs_rest_lfc"], as_mean["one_vs_rest_lfc"], rel_tol=1e-5
        )

    def test_gene_expression_values_honors_reduce(self):
        """`sum` must be `mean` times the exon length, or the parameter is not
        actually reaching the aggregation."""
        pytest.importorskip("pandas")
        from alphagenome_pytorch.aggregation import gene_expression_values

        annotation = _one_gene_annotation(start=100, end=200)
        preds = torch.ones(1, 1000, 2)
        common = dict(annotation=annotation, interval=("chr1", 0, 1000), log=None)
        mean_vals, _, _ = gene_expression_values(preds, reduce="mean", **common)
        sum_vals, _, _ = gene_expression_values(preds, reduce="sum", **common)
        assert mean_vals.numel() > 0
        torch.testing.assert_close(sum_vals, mean_vals * 100.0)


def _one_gene_annotation(start: int, end: int):
    """A GeneAnnotation with a single one-exon gene on chr1."""
    import pandas as pd

    from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation

    rows = [
        dict(Chromosome="chr1", Start=start, End=end, Strand="+",
             Feature=feature, gene_id="g1", gene_name="g1",
             gene_type="protein_coding", transcript_id="t1")
        for feature in ("gene", "exon")
    ]
    return GeneAnnotation(pd.DataFrame(rows))


class _StubTrunk(torch.nn.Module):
    """Returns fixed-shape embeddings; stands in for AlphaGenome in the loops."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, sequences, organism_idx, **kwargs):
        B, S = sequences.shape[0], sequences.shape[1]
        return {
            "embeddings_1bp": torch.randn(B, 1536, S) + 0 * self.anchor,
            "embeddings_128bp": torch.randn(B, 3072, S // 128) + 0 * self.anchor,
        }


@pytest.mark.unit
class TestMultiheadLoopsThreadReduceFunc:
    """Run a real batch through the loops `agt finetune` uses with the gene loss
    on. Unit tests of `gene_lfc_loss` alone cannot catch a mis-threaded keyword
    at the call site -- this is the path that runs in production."""

    S, C, MODALITY = 256, 4, "rna_seq"

    def _batch(self):
        torch.manual_seed(0)
        gene_mask = torch.zeros(1, self.S, 2, 2, dtype=torch.bool)
        gene_mask[:, 10:60, 0, 0] = True
        gene_mask[:, 100:220, 1, 1] = True
        targets = {
            1: torch.rand(1, self.S, self.C) + 0.1,
            128: torch.rand(1, self.S // 128, self.C) + 0.1,
        }
        return (
            torch.rand(1, self.S, 4),
            {self.MODALITY: targets},
            {"gene_mask": gene_mask},
        )

    def _common(self):
        from alphagenome_pytorch.extensions.finetuning.heads import (
            create_finetuning_head,
        )

        head = create_finetuning_head(self.MODALITY, n_tracks=self.C, resolutions=(1, 128))
        return dict(
            model=_StubTrunk(),
            heads={self.MODALITY: head},
            device="cpu",
            modality_weights={self.MODALITY: 1.0},
            resolution_weights={self.MODALITY: {1: 1.0, 128: 1.0}},
            positional_weight=5.0,
            count_weight=1.0,
            use_amp=False,
            gene_loss_weights={self.MODALITY: 0.1},
            gene_reduce_funcs={self.MODALITY: "sum"},
            strand_channel_masks={
                self.MODALITY: _build_strand_channel_mask(["+", "-", "+", "-"])
            },
        )

    def _spy(self, monkeypatch):
        import alphagenome_pytorch.losses as losses

        seen = []
        real = losses.gene_lfc_loss

        def spy(**kwargs):
            seen.append(kwargs.get("reduce_func"))
            return real(**kwargs)

        monkeypatch.setattr(losses, "gene_lfc_loss", spy)
        return seen

    def test_train_loop_passes_resolved_reduce_func(self, monkeypatch):
        from alphagenome_pytorch.extensions.finetuning.training import (
            train_epoch_multihead,
        )

        seen = self._spy(monkeypatch)
        common = self._common()
        head = common["heads"][self.MODALITY]
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        avg_loss, _ = train_epoch_multihead(
            train_loader=[self._batch()],
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            log_every=1,
            **common,
        )
        assert seen == ["sum"], f"loss saw reduce_func={seen}"
        assert math.isfinite(avg_loss)

    def test_every_gene_lfc_call_site_uses_real_parameters(self):
        """The sequence-parallel train loop calls `gene_lfc_loss` directly too,
        and needs multi-rank machinery to execute. Check every call site's
        keywords against the real signature so a renamed or misspelled argument
        fails here rather than on a cluster at step one."""
        import ast
        import inspect

        import alphagenome_pytorch.extensions.finetuning.training as training

        valid = set(inspect.signature(gene_lfc_loss).parameters)
        tree = ast.parse(inspect.getsource(training))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None)) == "gene_lfc_loss"
        ]
        assert len(calls) >= 3, "expected the single-head, multihead and SP call sites"
        for call in calls:
            unknown = {kw.arg for kw in call.keywords} - valid
            assert not unknown, f"line {call.lineno}: unknown keyword(s) {sorted(unknown)}"


@pytest.mark.unit
class TestRunnerThreadsEvalReduction:
    """The loss and the metric must not disagree about a head. That bug is
    invisible in a unit test of either one, so check the wiring itself."""

    def test_validate_multihead_receives_the_reduce_func(self):
        import ast
        import inspect

        import alphagenome_pytorch.extensions.finetuning.runner as runner

        tree = ast.parse(inspect.getsource(runner))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None))
            == "validate_multihead"
        ]
        assert calls, "no validate_multihead call site found in the runner"
        for call in calls:
            passed = {kw.arg for kw in call.keywords}
            assert "gene_expr_reduce_func" in passed, (
                f"line {call.lineno}: validation would silently fall back to "
                "'mean' for a 3'-biased head"
            )


@pytest.mark.unit
class TestValidationLoopThreadsReduceFunc:
    """Run a real batch through `validate_multihead`: both reductions must be
    accumulated, and the head's own must be the one reported unsuffixed. This
    is the path that used to hardcode 'mean' for every head."""

    S, C = 256, 4

    def test_accumulates_both_and_marks_the_head_reduction_primary(self, monkeypatch):
        import alphagenome_pytorch.extensions.finetuning.training as training
        from alphagenome_pytorch.extensions.finetuning.heads import (
            create_finetuning_head,
        )

        seen_reduce: list = []
        seen_primary: list = []
        real_metrics = training._gene_expr_metrics

        def accumulate_spy(windows, **kwargs):
            seen_reduce.append(kwargs.get("reduce"))
            windows.append((["g1"], torch.zeros(1, self.C), torch.zeros(1, self.C)))

        def metrics_spy(windows, **kwargs):
            seen_primary.append(kwargs.get("primary"))
            return real_metrics(windows, **kwargs)

        monkeypatch.setattr(training, "_accumulate_gene_expr_windows", accumulate_spy)
        monkeypatch.setattr(training, "_gene_expr_metrics", metrics_spy)

        torch.manual_seed(0)
        head = create_finetuning_head("rna_seq", n_tracks=self.C, resolutions=(1, 128))
        batch = (
            torch.rand(1, self.S, 4),
            {"rna_seq": {
                1: torch.rand(1, self.S, self.C) + 0.1,
                128: torch.rand(1, self.S // 128, self.C) + 0.1,
            }},
            {"coords": [("chr1", 0, self.S)]},
        )
        _, metrics = training.validate_multihead(
            model=_StubTrunk(),
            heads={"rna_seq": head},
            val_loader=[batch],
            device="cpu",
            modality_weights={"rna_seq": 1.0},
            resolution_weights={"rna_seq": {1: 1.0, 128: 1.0}},
            positional_weight=5.0,
            count_weight=1.0,
            use_amp=False,
            gene_annotation=object(),  # read only by the accumulator, which is spied
            gene_expr_track_strands=["+", "-", "+", "-"],
            gene_expr_strand_pairs=[(0, 1), (2, 3)],
            gene_expr_reduce_func="sum",
        )
        assert sorted(seen_reduce) == ["mean", "sum"]
        assert seen_primary == ["sum"]
        assert "rna_seq_gene_log_expr_pearson_sum_across_genes" in metrics
        assert "rna_seq_gene_log_expr_pearson_mean_across_genes" in metrics
