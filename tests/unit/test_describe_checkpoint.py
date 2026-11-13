"""Tests for artifact classification and the base-weights requirement.

``describe_checkpoint`` is the single classifier shared by
``load_finetuned_model``'s dispatch, ``agt info`` and the documentation's
routing table, so these tests pin the vocabulary the docs promise.

The base-weights tests assert the *fast-fail* contract: a delta-shaped artifact
given no base weights must raise before any tensor work, which is what lets the
error arrive instantly instead of after a multi-second load.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    CHECKPOINT_KINDS,
    DELTA_SHAPED_KINDS,
    describe_checkpoint,
    export_delta_weights,
    export_model_weights,
    load_finetuned_model,
    save_checkpoint,
    save_delta_checkpoint,
)
from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    prepare_for_transfer,
)


class _TinyModel(nn.Module):
    """Minimal stand-in: classification only inspects file structure."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)


def _lora_model_and_config():
    config = TransferConfig(mode=["lora"], lora_rank=2, lora_targets=["q_proj"])
    model = prepare_for_transfer(_TinyModel(), config)
    return model, config


def _save_full_checkpoint(path, **extra):
    model = _TinyModel()
    save_checkpoint(
        path=path,
        epoch=1,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        val_loss=0.5,
        track_names={"atac": ["t1"]},
        modality=["atac"],
        resolutions={"atac": (128,)},
        **extra,
    )


class TestDescribeCheckpoint:
    def test_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.pth"
            _save_full_checkpoint(path)
            info = describe_checkpoint(path)
            assert info.kind == "full_checkpoint"
            assert info.requires_base_weights is False

    def test_delta_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.delta.pth"
            model, config = _lora_model_and_config()
            save_delta_checkpoint(path=path, model=model, config=config, epoch=1)
            info = describe_checkpoint(path)
            assert info.kind == "delta_checkpoint"
            assert info.requires_base_weights is True

    def test_delta_export_safetensors(self):
        pytest.importorskip("safetensors")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.safetensors"
            model, config = _lora_model_and_config()
            export_delta_weights(model, config, path, format="safetensors")
            info = describe_checkpoint(path)
            assert info.kind == "delta_export"
            assert info.requires_base_weights is True

    def test_delta_export_pth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.pth"
            model, config = _lora_model_and_config()
            export_delta_weights(model, config, path, format="pth")
            info = describe_checkpoint(path)
            assert info.kind == "delta_export"
            assert info.requires_base_weights is True

    def test_full_export_safetensors(self):
        pytest.importorskip("safetensors")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.safetensors"
            export_model_weights(_TinyModel(), path, format="safetensors")
            info = describe_checkpoint(path)
            assert info.kind == "full_export"
            assert info.requires_base_weights is False

    def test_full_export_pth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pth"
            export_model_weights(_TinyModel(), path, format="pth")
            info = describe_checkpoint(path)
            assert info.kind == "full_export"
            assert info.requires_base_weights is False

    def test_bundle_directory(self):
        pytest.importorskip("safetensors")
        from alphagenome_pytorch.extensions.serving.bundle import MANIFEST_FILENAME

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "my-bundle"
            bundle.mkdir()
            model, config = _lora_model_and_config()
            export_delta_weights(
                model, config, bundle / "adapter.safetensors", format="safetensors"
            )
            (bundle / MANIFEST_FILENAME).write_text(
                json.dumps(
                    {
                        "id": "my-bundle",
                        "schema_version": 1,
                        "base_model_hash": "abc123",
                        "base_model_weights_hash": "def456",
                    }
                )
            )
            info = describe_checkpoint(bundle)
            assert info.kind == "bundle"
            assert info.requires_base_weights is True
            assert info.base_model_weights_hash == "def456"

    def test_every_kind_is_declared(self):
        """The documented vocabulary and the code must not drift apart."""
        assert set(DELTA_SHAPED_KINDS) <= set(CHECKPOINT_KINDS)

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            describe_checkpoint("/nonexistent/nope.pth")

    def test_directory_without_manifest_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="not an adapter bundle"):
                describe_checkpoint(tmp)

    def test_records_base_weights_hash_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.delta.pth"
            model, config = _lora_model_and_config()
            save_delta_checkpoint(
                path=path,
                model=model,
                config=config,
                epoch=1,
                base_model_weights_hash="cafebabe",
            )
            assert describe_checkpoint(path).base_model_weights_hash == "cafebabe"


class TestBaseWeightsRequired:
    """Delta-shaped artifacts must fail fast and informatively without a base."""

    def test_delta_checkpoint_without_base_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.delta.pth"
            model, config = _lora_model_and_config()
            save_delta_checkpoint(path=path, model=model, config=config, epoch=1)
            with pytest.raises(ValueError, match="cannot be loaded on its own"):
                load_finetuned_model(path)

    def test_delta_export_without_base_weights(self):
        pytest.importorskip("safetensors")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.safetensors"
            model, config = _lora_model_and_config()
            export_delta_weights(model, config, path, format="safetensors")
            with pytest.raises(ValueError, match="cannot be loaded on its own"):
                load_finetuned_model(path)

    def test_error_names_the_expected_base_weights(self):
        """Knowing *which* base file is needed is the difference between an
        actionable error and a scavenger hunt."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.delta.pth"
            model, config = _lora_model_and_config()
            save_delta_checkpoint(
                path=path,
                model=model,
                config=config,
                epoch=1,
                base_model_weights_hash="deadbeef",
            )
            with pytest.raises(ValueError, match="deadbeef"):
                load_finetuned_model(path)

    def test_error_suggests_both_interfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.delta.pth"
            model, config = _lora_model_and_config()
            save_delta_checkpoint(path=path, model=model, config=config, epoch=1)
            with pytest.raises(ValueError) as exc:
                load_finetuned_model(path)
            message = str(exc.value)
            assert "pretrained_weights=" in message
            assert "agt predict --model" in message

    def test_fast_fail_does_not_touch_base_weights_file(self):
        """The check must run before any base-weights I/O.

        Passing a nonexistent base path must not change the outcome for a
        *self-contained* artifact, and for a delta artifact the requirement
        error must arrive without the loader ever opening the base file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.delta.pth"
            model, config = _lora_model_and_config()
            save_delta_checkpoint(path=path, model=model, config=config, epoch=1)

            # No base weights: fails on the requirement, not on file access.
            with pytest.raises(ValueError, match="cannot be loaded on its own"):
                load_finetuned_model(path)
