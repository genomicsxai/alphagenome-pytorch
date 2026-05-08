"""
Tests for load_finetuned_model() checkpoint-format detection.

Covers:
  - Path B, lora: full checkpoint with embedded lora TransferConfig.
  - Path B, full: full checkpoint with embedded TransferConfig(mode="full"),
    exercising custom head names decoupled from assay type.
  - Path C guard: adapter keys without a TransferConfig raises ValueError.
  - Path C legacy: linear-probe-style full checkpoint with head name
    equal to modality (the legacy save convention).
  - Non-dict payload raises ValueError (not TypeError).
"""

import gc
import tempfile
from pathlib import Path

import pytest
import torch

from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    load_finetuned_model,
)
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    prepare_for_transfer,
    remove_all_heads,
    transfer_config_to_dict,
)
from alphagenome_pytorch.model import AlphaGenome


def _make_model(**kwargs):
    # num_organisms defaults to 2 to match what load_finetuned_model
    # constructs internally.
    model = AlphaGenome(
        dtype_policy=DtypePolicy.full_float32(),
        **kwargs,
    )
    model.eval()
    return model


@pytest.fixture
def base_weights_path():
    """Save a bare AlphaGenome's state_dict to use as pretrained weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _make_model()
        path = Path(tmpdir) / "base.pth"
        torch.save(base.state_dict(), path)
        del base
        gc.collect()
        yield path


def _lora_config(base_model):
    return TransferConfig(
        mode="lora",
        lora_rank=4,
        lora_alpha=8,
        lora_targets=["q_proj", "v_proj"],
        remove_heads=list(base_model.heads.keys()),
        new_heads={
            "my_atac": {
                "modality": "atac",
                "num_tracks": 4,
                "resolutions": [128],
            },
        },
    )


def _full_mode_config(base_model):
    return TransferConfig(
        mode="full",
        remove_heads=list(base_model.heads.keys()),
        new_heads={
            "my_atac": {
                "modality": "atac",
                "num_tracks": 4,
                "resolutions": [128],
            },
        },
    )


@pytest.mark.integration
class TestLoadFinetunedModel:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        gc.collect()

    def test_lora_checkpoint_with_embedded_config(self, base_weights_path):
        """Path B: embedded lora TransferConfig reconstructs adapters and heads."""
        base_model = _make_model()
        config = _lora_config(base_model)
        adapted = _make_model()
        adapted.load_state_dict(
            torch.load(base_weights_path, weights_only=True), strict=False
        )
        adapted = prepare_for_transfer(adapted, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "finetuned.pth"
            torch.save(
                {
                    "model_state_dict": adapted.state_dict(),
                    "transfer_config": transfer_config_to_dict(config),
                    "modality": "my_atac",
                    "track_names": ["t1", "t2", "t3", "t4"],
                    "resolutions": (128,),
                    "epoch": 3,
                    "val_loss": 0.1,
                },
                ckpt_path,
            )

            # merge=False so we can verify adapters were reconstructed
            model, meta = load_finetuned_model(
                ckpt_path, base_weights_path, merge=False,
            )

        assert "my_atac" in model.heads
        lora_names = [n for n, _ in model.named_parameters() if "lora_" in n]
        assert len(lora_names) > 0
        assert meta["head_names"] == ["my_atac"]
        assert meta["epoch"] == 3

    def test_full_mode_checkpoint_with_custom_head_name(self, base_weights_path):
        """Path B: mode="full" TransferConfig round-trips a head whose name
        differs from its assay type (e.g. head "my_atac", modality "atac")."""
        base_model = _make_model()
        config = _full_mode_config(base_model)
        prepared = _make_model()
        prepared.load_state_dict(
            torch.load(base_weights_path, weights_only=True), strict=False
        )
        prepared = prepare_for_transfer(prepared, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "full_finetuned.pth"
            torch.save(
                {
                    "model_state_dict": prepared.state_dict(),
                    "transfer_config": transfer_config_to_dict(config),
                    "modality": "atac",
                    "track_names": ["t1", "t2", "t3", "t4"],
                    "resolutions": (128,),
                    "epoch": 2,
                    "val_loss": 0.2,
                },
                ckpt_path,
            )

            model, meta = load_finetuned_model(ckpt_path, base_weights_path)

        assert "my_atac" in model.heads
        assert "atac" not in model.heads
        assert meta["head_names"] == ["my_atac"]
        # No LoRA adapters in full mode
        assert not [n for n, _ in model.named_parameters() if "lora_" in n]

    def test_adapter_keys_without_config_raises(self, base_weights_path):
        """Path C guard: adapter-shaped keys without a TransferConfig must
        produce a clear ValueError, not a silent partial load."""
        base_model = _make_model()
        config = _lora_config(base_model)
        adapted = _make_model()
        adapted.load_state_dict(
            torch.load(base_weights_path, weights_only=True), strict=False
        )
        adapted = prepare_for_transfer(adapted, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "finetuned.pth"
            # Intentionally omit transfer_config
            torch.save(
                {
                    "model_state_dict": adapted.state_dict(),
                    "modality": "my_atac",
                    "track_names": ["t1", "t2", "t3", "t4"],
                    "resolutions": (128,),
                },
                ckpt_path,
            )

            with pytest.raises(ValueError, match="adapter parameters"):
                load_finetuned_model(ckpt_path, base_weights_path)

    def test_legacy_linear_probe_checkpoint_without_config(self, base_weights_path):
        """Path C backward-compat: legacy checkpoints have no TransferConfig
        and rely on the head-name==modality convention from finetune.py."""
        model = _make_model()
        model.load_state_dict(
            torch.load(base_weights_path, weights_only=True), strict=False
        )
        model = remove_all_heads(model)
        model.heads["atac"] = create_finetuning_head(
            assay_type="atac", n_tracks=4, resolutions=(128,),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "linprobe.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "modality": "atac",
                    "track_names": ["t1", "t2", "t3", "t4"],
                    "resolutions": (128,),
                    "epoch": 1,
                    "val_loss": 0.5,
                },
                ckpt_path,
            )

            loaded, meta = load_finetuned_model(ckpt_path, base_weights_path)

        assert "atac" in loaded.heads
        assert not [n for n, _ in loaded.named_parameters() if "lora_" in n]
        assert meta["head_names"] == ["atac"]

    def test_non_dict_checkpoint_raises_value_error(self, base_weights_path):
        """Non-dict payload (e.g. a raw tensor) must raise ValueError, not TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "raw_tensor.pth"
            torch.save(torch.zeros(3), ckpt_path)

            with pytest.raises(ValueError, match="Expected dict"):
                load_finetuned_model(ckpt_path, base_weights_path)
