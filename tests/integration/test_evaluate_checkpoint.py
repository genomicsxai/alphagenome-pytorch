"""Integration tests for scripts/evaluate_checkpoint.py.

Tests the two inference entry points (`evaluate_split` and
`evaluate_native_split`) against a real AlphaGenome model with random weights
and mock BigWig data. These tests guard against two historical bugs:

1. `evaluate_split` passed NLC embeddings to `GenomeTracksHead`, which
   expects NCL, causing a Conv1d shape mismatch crash.
2. `evaluate_native_split` read flat keys like `"atac_128bp"` from model
   outputs, but `AlphaGenome.forward()` returns nested
   `outputs[head_name][resolution]`, so every lookup missed and the
   function silently returned `{}`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning.datasets import ATACDataset
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.training import collate_genomic


SEQUENCE_LENGTH = 16384
N_TRACKS = 2
RESOLUTIONS = (1, 128)
MODALITY = "atac"


def _load_script_module():
    """Load scripts/evaluate_checkpoint.py as a module (it's not a package)."""
    script_path = (
        Path(__file__).parent.parent.parent / "scripts" / "evaluate_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_checkpoint_script", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model(device):
    m = AlphaGenome(num_organisms=1, dtype_policy=DtypePolicy.full_float32())
    # Replace default ATAC head (256 tracks) with a finetuning-sized head
    # so head output shape matches the mock dataset (N_TRACKS).
    m.heads[MODALITY] = create_finetuning_head(
        assay_type=MODALITY,
        n_tracks=N_TRACKS,
        resolutions=RESOLUTIONS,
        num_organisms=1,
    )
    m.eval()
    m.to(device)
    return m


@pytest.fixture(scope="module")
def loader(mock_data_dir):
    dataset = ATACDataset(
        genome_fasta=str(mock_data_dir / "mock_genome.fa"),
        bigwig_files=[
            str(mock_data_dir / f"mock_atac_track{i}.bw") for i in (1, 2)
        ],
        bed_file=str(mock_data_dir / "mock_positions.bed"),
        resolutions=list(RESOLUTIONS),
        sequence_length=SEQUENCE_LENGTH,
    )
    # Two batches is enough to catch crashes and empty-dict bugs.
    from torch.utils.data import Subset
    dataset = Subset(dataset, list(range(min(2, len(dataset)))))
    return DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_genomic,
    )


@pytest.mark.finetuning
def test_evaluate_split_returns_nonempty_predictions(
    script_module, model, loader, device,
):
    """Regression: embeddings must be NCL when fed into GenomeTracksHead.

    Before the fix, `model(..., embeddings_only=True)` returned NLC embeddings
    by default, which caused Conv1d in the head to fail with a shape mismatch.
    """
    preds, targets, avg_loss = script_module.evaluate_split(
        model=model,
        modality=MODALITY,
        loader=loader,
        device=device,
        resolutions=RESOLUTIONS,
        organism_index=0,
    )

    # Both resolutions should be populated with batch-first arrays.
    for res in RESOLUTIONS:
        assert res in preds, f"missing preds for resolution {res}"
        assert res in targets, f"missing targets for resolution {res}"
        assert preds[res].shape[0] > 0, f"empty preds for resolution {res}"
        assert preds[res].shape[-1] == N_TRACKS, (
            f"expected {N_TRACKS} tracks, got preds[{res}].shape={preds[res].shape}"
        )
        expected_seq_len = SEQUENCE_LENGTH // res
        assert preds[res].shape[1] == expected_seq_len, (
            f"expected seq_len={expected_seq_len} at res={res}, "
            f"got {preds[res].shape[1]}"
        )

    assert torch.isfinite(torch.tensor(avg_loss)), (
        f"non-finite loss: {avg_loss}"
    )


@pytest.mark.finetuning
def test_evaluate_native_split_reads_nested_outputs(
    script_module, model, loader, device,
):
    """Regression: outputs[modality] is `dict[int, Tensor]`, not flat keys.

    Before the fix, `evaluate_native_split` looked for flat keys like
    "atac_128bp" which never existed, so the function silently returned {}.
    """
    preds = script_module.evaluate_native_split(
        model=model,
        modality=MODALITY,
        track_index=0,
        loader=loader,
        device=device,
        resolutions=RESOLUTIONS,
        organism_index=0,
    )

    for res in RESOLUTIONS:
        assert res in preds, (
            f"missing preds for resolution {res} — "
            "likely the nested-output lookup is wrong"
        )
        # Single track was requested via track_index=0.
        assert preds[res].shape[-1] == 1, (
            f"expected 1 track, got preds[{res}].shape={preds[res].shape}"
        )
        expected_seq_len = SEQUENCE_LENGTH // res
        assert preds[res].shape[1] == expected_seq_len


@pytest.mark.finetuning
def test_evaluate_native_split_skips_missing_resolution(
    script_module, model, loader, device,
):
    """Asking for a resolution the head doesn't expose must skip cleanly.

    Guards against the `len(head_outputs) == 1` fallback that would
    silently return the wrong-resolution tensor.
    """
    # ATAC head has resolutions (1, 128). Request 1bp only, plus a bogus
    # resolution (e.g. 64) that doesn't exist — it should be absent from
    # the returned dict, not silently substituted.
    preds = script_module.evaluate_native_split(
        model=model,
        modality=MODALITY,
        track_index=0,
        loader=loader,
        device=device,
        resolutions=(1, 64),
        organism_index=0,
    )
    assert 1 in preds
    assert 64 not in preds


def test_parse_resolutions_defaults_to_128(script_module):
    assert script_module.parse_resolutions(None) == (128,)
    assert script_module.parse_resolutions("1,128") == (1, 128)


def test_center_crop_and_sum_pool_metric_views(script_module):
    values = np.arange(16, dtype=np.float64).reshape(1, 16, 1)
    pred_views, target_views = script_module.build_metric_views(
        values,
        values,
        source_resolution=1,
        bin_sizes=(1, 4),
        score_window_bp=8,
        reduction="sum",
    )

    np.testing.assert_array_equal(pred_views[1][0, :, 0], np.arange(4, 12))
    np.testing.assert_array_equal(pred_views[4][0, :, 0], [22, 38])
    np.testing.assert_array_equal(pred_views[4], target_views[4])


def test_score_window_larger_than_predictions_rejected(script_module):
    values = np.zeros((1, 131_072, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="cover only 131,072 bp"):
        script_module.center_crop_profiles(values, values, 196_608, 1)


def test_comparison_metrics_expose_paper_and_requested_variants(script_module):
    targets = np.asarray([
        [[0.0], [1.0], [2.0], [3.0]],
        [[1.0], [2.0], [3.0], [4.0]],
    ])
    metrics = script_module.compute_comparison_metrics(targets, targets)

    assert metrics["profile_pearson_r_mean"] == pytest.approx(1.0)
    assert metrics["track_pearson_r_accumulated_mean"] == pytest.approx(1.0)
    assert metrics["count_pearson_r_raw"] == pytest.approx(1.0)
    assert metrics["count_pearson_r_log1p"] == pytest.approx(1.0)
    assert metrics["js_divergence_mean"] == pytest.approx(0.0)
    assert metrics["js_distance_mean"] == pytest.approx(0.0)


def test_profile_accumulator_matches_alphagenome_research_formula(script_module):
    x = np.asarray([
        [[1.0, 4.0], [2.0, 4.0]],
        [[3.0, 4.0], [5.0, 4.0]],
    ], dtype=np.float32)
    y = np.asarray([
        [[2.0, 7.0], [1.0, 7.0]],
        [[4.0, 7.0], [8.0, 7.0]],
    ], dtype=np.float32)

    state = script_module._pearsonr_initialize()
    state += script_module._pearsonr_update(x[:1], y[:1], axis=(-2, -3))
    state += script_module._pearsonr_update(x[1:], y[1:], axis=(-2, -3))
    actual = script_module._pearsonr_result(state)

    # Direct NumPy transcription of alphagenome_research/evals/
    # regression_metrics.py::_pearsonr_update/_pearsonr_result.
    axes = (0, 1)
    count = np.sum(np.ones_like(x), axis=axes, dtype=np.float32)
    x_sum = np.sum(x, axis=axes, dtype=np.float32)
    y_sum = np.sum(y, axis=axes, dtype=np.float32)
    x_mean = x_sum / count
    y_mean = y_sum / count
    covariance = (
        np.sum(x * y, axis=axes, dtype=np.float32)
        - count * x_mean * y_mean
    )
    x_variance = (
        np.sum(np.square(x), axis=axes, dtype=np.float32)
        - count * np.square(x_mean)
    )
    y_variance = (
        np.sum(np.square(y), axis=axes, dtype=np.float32)
        - count * np.square(y_mean)
    )
    denominator = np.sqrt(x_variance) * np.sqrt(y_variance)
    expected = covariance / (denominator + np.finfo(denominator.dtype).eps)

    np.testing.assert_array_equal(actual, expected)
    assert actual[1] == 0.0  # Reference epsilon behavior for constant tracks.


def test_parse_metric_bin_sizes(script_module):
    assert script_module.parse_metric_bin_sizes(None) is None
    assert script_module.parse_metric_bin_sizes("1,32") == (1, 32)
    with pytest.raises(ValueError, match="duplicates"):
        script_module.parse_metric_bin_sizes("1,1")


def test_format_summary_table_supports_native_only(script_module):
    native_metrics = {
        128: {
            "profile_pearson_r_mean": 0.25,
            "profile_pearson_r_median": 0.20,
            "count_pearson_r": 0.50,
            "jsd_mean": 0.10,
            "jsd_median": 0.08,
            "mse": 1.25,
            "spearman_global": 0.40,
            "n_regions": 3,
        }
    }

    summary = script_module.format_summary_table(
        ft_metrics=None,
        native_metrics=native_metrics,
        native_display_name="K562",
        resolutions=(128,),
    )

    assert "Native(K562)" in summary
    assert "Finetuned" not in summary
    assert "Profile r (mean)" in summary
    assert "0.2500" in summary


def test_native_only_requires_explicit_metrics(script_module, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_checkpoint.py",
            "--native-only",
            "--pretrained-weights",
            "model.safetensors",
            "--output-dir",
            str(tmp_path),
            "--modality",
            "atac",
            "--native-biosample",
            "K562",
        ],
    )

    with pytest.raises(SystemExit, match="--native-only currently supports metrics"):
        script_module.main()


def test_native_only_requires_native_track_selector(script_module, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_checkpoint.py",
            "--native-only",
            "--metrics",
            "--pretrained-weights",
            "model.safetensors",
            "--output-dir",
            str(tmp_path),
            "--modality",
            "atac",
        ],
    )

    with pytest.raises(SystemExit, match="--native-biosample or --native-track-index"):
        script_module.main()


@pytest.mark.parametrize(
    ("metadata", "override", "expected"),
    [
        ({"gene_reduce_funcs": {"rna_seq": "sum"}}, [], "sum"),
        ({"gene_reduce_funcs": {"rna_seq": "sum"}}, ["--gene-reduce-func", "mean"], "mean"),
        ({}, [], "mean"),
        ({"gene_reduce_funcs": None}, [], "mean"),
        ({"gene_reduce_funcs": {"atac": "sum"}}, [], "mean"),
    ],
)
def test_evaluator_gene_reduction_precedence(script_module, monkeypatch, metadata, override, expected):
    monkeypatch.setattr(sys, "argv", [
        "evaluate_checkpoint.py", "--pretrained-weights", "base.pth", "--output-dir", "out",
        *override,
    ])
    args = script_module.parse_args()
    assert script_module.resolve_gene_reduce_func(args.gene_reduce_func, metadata, "rna_seq") == expected


@pytest.mark.parametrize(
    ("flags", "expected_pairs"),
    [
        ([], [(0, 3), (1, 2)]),
        (["--strand-pairs", "0,2;1,3"], [(0, 2), (1, 3)]),
    ],
)
def test_evaluator_reuses_saved_strand_pairs(
    script_module, monkeypatch, tmp_path, flags, expected_pairs,
):
    from types import SimpleNamespace

    names = ["a_plus", "b_plus", "b_minus", "a_minus"]
    metadata = {
        "modality": "rna_seq", "track_names": names, "resolutions": (1,),
        "epoch": 1, "val_loss": 0.0, "strand_pairs": {"rna_seq": [(0, 3), (1, 2)]},
        "track_metadata": [
            {"output_name": "rna_seq", "track_index": index, "strand": strand}
            for index, strand in enumerate("++--")
        ],
    }
    model = SimpleNamespace(finetuned_organism_context=None, num_organisms=1)
    monkeypatch.setattr(script_module, "load_finetuned_model", lambda *_: (model, metadata))
    monkeypatch.setattr(script_module, "select_organism_index", lambda *_, **__: 0)
    monkeypatch.setattr(
        script_module, "GenomicDataset",
        lambda **_: SimpleNamespace(_positions_list=[("chr1", 0, 128)]),
    )
    monkeypatch.setattr(script_module, "DataLoader", lambda *_, **__: None)
    monkeypatch.setattr(script_module, "evaluate_split", lambda *_, **__: ({}, {}, 0.0))
    captured = {}

    def gene_metrics(*_, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(script_module, "compute_gene_expression_metrics", gene_metrics)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_checkpoint.py", "--checkpoint", "model.pth", "--pretrained-weights", "base.pth",
        "--output-dir", str(tmp_path), "--test-bed", "test.bed", "--max-regions", "0",
        "--gene-annotation", "genes.parquet", "--bigwig", *names, *flags,
    ])
    script_module.main()
    assert captured["strand_pairs"] == expected_pairs


def test_gene_expression_metrics_match_validation_metric(tmp_path):
    """--gene-annotation reports the same numbers as --gene-expr-eval."""
    import json

    import pandas as pd

    from alphagenome_pytorch.aggregation import (
        combine_gene_expression,
        gene_expression_values,
        resolve_strand_pairs,
    )
    from alphagenome_pytorch.annotations import GeneAnnotation

    script = _load_script_module()
    rows = []
    for g in range(8):
        strand = "+" if g % 2 == 0 else "-"
        start = 10 + 30 * g
        for feature in ("gene", "exon"):
            rows.append(dict(
                Chromosome="chr1", Start=start, End=start + 20, Strand=strand,
                Feature=feature, gene_id=f"G{g}", gene_name=f"g{g}",
                gene_type="protein_coding",
            ))
    annotation_path = tmp_path / "genes.parquet"
    pd.DataFrame(rows).to_parquet(annotation_path)

    rng = np.random.default_rng(0)
    positions = [("chr1", 0, 256)]
    preds = {1: rng.gamma(1.0, 2.0, size=(1, 256, 5)).astype(np.float32)}
    targets = {1: (preds[1] * rng.lognormal(0, 0.3, size=preds[1].shape)).astype(np.float32)}
    strands = script.resolve_track_strands("++--.", {}, "rna_seq", 5)
    pairs = [(1, 2), (0, 3)]
    pair_ids = ["A", "B", "B", "A", None]
    track_names = ["A+", "B+", "B-", "A-", "C"]

    metrics = script.compute_gene_expression_metrics(
        preds, targets, positions, modality="rna_seq",
        annotation_path=str(annotation_path), track_strands=strands, strand_pairs=pairs,
        track_names=track_names, strand_pair_ids=pair_ids,
    )

    annotation = GeneAnnotation(pd.DataFrame(rows))
    pred, gene_ids, _ = gene_expression_values(
        torch.from_numpy(preds[1][0]), annotation, positions[0], track_strands=strands,
    )
    obs, _, _ = gene_expression_values(
        torch.from_numpy(targets[1][0]), annotation, positions[0], track_strands=strands,
    )
    expected = combine_gene_expression(
        [(gene_ids, pred, obs)], track_strands=strands, strand_pairs=pairs,
    )
    assert metrics["rna_seq_gene_log_expr_n_genes"] == expected["n_genes"] == 8
    prefix = "rna_seq_gene_log_expr_pearson_one_vs_rest_lfc"
    values = metrics[f"{prefix}_per_track"]
    assert values == pytest.approx(expected["one_vs_rest_lfc_per_track"])
    path = tmp_path / "summary.json"
    script.save_summary_json(None, None, {}, None, None, path, gene_expression=metrics)
    rows = json.loads(path.read_text())["gene_expression"][f"{prefix}_by_sample"]
    assert rows == [
        {"sample_id": "B", "track_indices": [1, 2], "value": values[0]},
        {"sample_id": "A", "track_indices": [0, 3], "value": values[1]},
        {"sample_id": "C", "track_indices": [4], "value": values[2]},
    ]

    metadata_pairs = resolve_strand_pairs(strands, strand_pair_ids=pair_ids)
    assert metadata_pairs == [(0, 3), (1, 2)]
    from_metadata = script.compute_gene_expression_metrics(
        preds, targets, positions, modality="rna_seq",
        annotation_path=str(annotation_path), track_strands=strands,
        strand_pairs=metadata_pairs, track_names=track_names, strand_pair_ids=pair_ids,
        reduce_func=script.resolve_gene_reduce_func(
            None, {"gene_reduce_funcs": {"rna_seq": "sum"}}, "rna_seq",
        ),
    )
    summary_prefix = "rna_seq_gene_log_expr_pearson"
    assert from_metadata["gene_reduce_func"] == "sum"
    assert from_metadata[f"{summary_prefix}_across_genes"] == from_metadata[f"{summary_prefix}_sum_across_genes"]
    reordered_rows = from_metadata[f"{prefix}_by_sample"]
    for actual, expected_row in zip(reordered_rows, [rows[1], rows[0], rows[2]], strict=True):
        assert actual["sample_id"] == expected_row["sample_id"]
        assert actual["track_indices"] == expected_row["track_indices"]
        assert actual["value"] == pytest.approx(expected_row["value"])
    assert "rna_seq_gene_log_expr_pearson_sum_across_genes" in metrics
    for key in ("across_genes", "across_genes_norm", "across_tracks_norm"):
        assert metrics[f"rna_seq_gene_log_expr_pearson_{key}"] == pytest.approx(expected[key])
        assert not np.isnan(expected[key])


def test_track_strands_fall_back_to_checkpoint_metadata():
    script = _load_script_module()
    meta = {"track_metadata": [
        {"output_name": "rna_seq", "track_index": 1, "strand": "-"},
        {"output_name": "rna_seq", "track_index": 0, "strand": "+"},
        {"output_name": "atac", "track_index": 0, "strand": "."},
    ]}
    assert script.resolve_track_strands(None, meta, "rna_seq", 2) == ["+", "-"]
    with pytest.raises(SystemExit):
        script.resolve_track_strands(None, {}, "rna_seq", 2)
    with pytest.raises(SystemExit):
        script.resolve_track_strands("+-+", meta, "rna_seq", 2)

