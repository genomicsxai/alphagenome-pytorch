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


def test_borzoi_window_rejected_for_131kb_predictions(script_module):
    values = np.zeros((1, 131_072, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot be scored over the 196,608-bp"):
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
