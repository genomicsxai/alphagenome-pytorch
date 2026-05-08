"""Integration tests for scripts/evaluate_finetuned.py.

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
from pathlib import Path

import pytest
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
    """Load scripts/evaluate_finetuned.py as a module (it's not a package)."""
    script_path = (
        Path(__file__).parent.parent.parent / "scripts" / "evaluate_finetuned.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_finetuned_script", script_path
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
