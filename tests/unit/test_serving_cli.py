"""Adapter-construction tests for ``agt serve`` CLI helpers.

These tests pin the two adapter-construction paths in
``alphagenome_pytorch.extensions.serving.cli``:

* ``--checkpoint`` → fine-tuned :class:`LocalDnaModelAdapter` with a configured
  :class:`VariantScorer`.
* ``--weights``   → pretrained :class:`LocalDnaModelAdapter` with a configured
  :class:`VariantScorer`.

They do **not** spin up the gRPC/REST servers — the heavy I/O dependencies
(``load_finetuned_model``, ``AlphaGenome``, ``VariantScoringModel`` and
``torch.load``) are patched out so the test stays unit-scope. End-to-end
serving coverage lives in the existing REST/gRPC test files.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from alphagenome_pytorch.extensions.serving.adapter import LocalDnaModelAdapter
from alphagenome_pytorch.extensions.serving.cli import _build_adapter
from alphagenome_pytorch.extensions.serving.scorer import VariantScorer


def _checkpoint_args(**overrides):
    base = dict(
        checkpoint='/fake/checkpoint.pt',
        weights=None,
        fasta=None,
        gtf=None,
        polya=None,
        track_metadata=None,
        device='cpu',
        transfer_config=None,
        no_merge_adapters=False,
        log_level='INFO',
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _weights_args(**overrides):
    base = dict(
        checkpoint=None,
        weights='/fake/model.pth',
        fasta=None,
        gtf=None,
        polya=None,
        device='cpu',
        track_metadata=None,
        log_level='INFO',
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _stub_model():
    """Tiny ``torch.nn.Module`` that satisfies the runtime's introspection."""
    model = torch.nn.Module()
    model.num_organisms = 2
    return model


def test_checkpoint_path_creates_scoring_adapter():
    """``--checkpoint`` builds a ``LocalDnaModelAdapter`` with scoring enabled."""
    fake_meta = {'track_names': {'dnase': ['t0', 't1']}}

    with (
        patch(
            'alphagenome_pytorch.extensions.finetuning.checkpointing.load_finetuned_model',
            return_value=(_stub_model(), fake_meta),
        ) as load_mock,
        patch(
            'alphagenome_pytorch.extensions.serving.cli.VariantScoringModel',
            return_value=MagicMock(spec=[]),
        ) as scoring_mock,
    ):
        adapter = _build_adapter(_checkpoint_args())

    assert isinstance(adapter, LocalDnaModelAdapter)
    assert isinstance(adapter.scorer, VariantScorer)
    load_mock.assert_called_once()
    scoring_mock.assert_called_once()
    # The runtime was wired up with the track names from the checkpoint metadata.
    assert adapter.runtime._track_names == fake_meta['track_names']


def test_checkpoint_uses_embedded_track_metadata():
    """When the checkpoint embeds track_metadata, the runtime gets a populated catalog."""
    embedded_rows = [
        {"output_name": "dnase", "track_name": "t0", "biosample_name": "K562"},
        {"output_name": "dnase", "track_name": "t1", "biosample_name": "GM12878"},
    ]
    fake_meta = {
        'track_names': {'dnase': ['t0', 't1']},
        'track_metadata': embedded_rows,
    }

    with (
        patch(
            'alphagenome_pytorch.extensions.finetuning.checkpointing.load_finetuned_model',
            return_value=(_stub_model(), fake_meta),
        ),
        patch(
            'alphagenome_pytorch.extensions.serving.cli.VariantScoringModel',
            return_value=MagicMock(),
        ),
    ):
        adapter = _build_adapter(_checkpoint_args())

    catalog = adapter.runtime.metadata_catalog
    assert catalog is not None
    tracks = catalog.get_tracks('dnase', organism=0)
    assert [t.track_name for t in tracks] == ['t0', 't1']
    assert tracks[0].extras.get('biosample_name') == 'K562'


def test_checkpoint_cli_track_metadata_overrides_embedded(tmp_path, caplog):
    """An explicit --track-metadata flag wins over the embedded catalog and warns."""
    import logging
    import pandas as pd

    cli_metadata_path = tmp_path / "cli_metadata.parquet"
    pd.DataFrame([
        {"output_type": "dnase", "track_name": "from_cli_a", "biosample_name": "HepG2"},
        {"output_type": "dnase", "track_name": "from_cli_b", "biosample_name": "A549"},
    ]).to_parquet(cli_metadata_path)

    fake_meta = {
        'track_names': {'dnase': ['t0', 't1']},
        'track_metadata': [
            {"output_name": "dnase", "track_name": "from_embedded", "biosample_name": "K562"},
        ],
    }

    with (
        patch(
            'alphagenome_pytorch.extensions.finetuning.checkpointing.load_finetuned_model',
            return_value=(_stub_model(), fake_meta),
        ),
        patch(
            'alphagenome_pytorch.extensions.serving.cli.VariantScoringModel',
            return_value=MagicMock(),
        ),
        caplog.at_level(logging.WARNING, logger='alphagenome_pytorch.extensions.serving.cli'),
    ):
        adapter = _build_adapter(
            _checkpoint_args(track_metadata=str(cli_metadata_path))
        )

    catalog = adapter.runtime.metadata_catalog
    assert catalog is not None
    tracks = catalog.get_tracks('dnase', organism=0)
    assert [t.track_name for t in tracks] == ['from_cli_a', 'from_cli_b']
    assert any(
        'Both --track-metadata and an embedded metadata catalog' in r.message
        for r in caplog.records
    )


def test_checkpoint_without_metadata_yields_no_catalog():
    """Bare checkpoints (no embedded metadata, no --track-metadata) leave the catalog unset."""
    fake_meta = {'track_names': {'dnase': ['t0']}, 'track_metadata': None}

    with (
        patch(
            'alphagenome_pytorch.extensions.finetuning.checkpointing.load_finetuned_model',
            return_value=(_stub_model(), fake_meta),
        ),
        patch(
            'alphagenome_pytorch.extensions.serving.cli.VariantScoringModel',
            return_value=MagicMock(),
        ),
    ):
        adapter = _build_adapter(_checkpoint_args())

    assert adapter.runtime.metadata_catalog is None


def test_weights_path_creates_scoring_adapter():
    """``--weights`` builds a ``LocalDnaModelAdapter`` with a ``VariantScorer``."""
    stub_model = _stub_model()

    # Patch in the order construction visits these symbols.
    with (
        patch(
            'alphagenome_pytorch.extensions.serving.cli.AlphaGenome',
            return_value=stub_model,
        ),
        patch(
            'alphagenome_pytorch.extensions.serving.cli.torch.load',
            return_value={},
        ),
        patch(
            'alphagenome_pytorch.extensions.serving.cli.VariantScoringModel',
            return_value=MagicMock(spec=[]),
        ) as scoring_mock,
        # No bundled metadata; keep this path off disk.
        patch(
            'alphagenome_pytorch.extensions.serving.cli._resolve_bundled_metadata_paths',
            return_value=[],
        ),
    ):
        adapter = _build_adapter(_weights_args())

    assert isinstance(adapter, LocalDnaModelAdapter)
    assert isinstance(adapter.scorer, VariantScorer)
    scoring_mock.assert_called_once()
