"""
Tests for load_finetuned_model() checkpoint-format detection.

Covers the three checkpoint-format branches and the non-dict guard:
  - Full checkpoint with embedded TransferConfig (adapter reconstruction)
  - Full checkpoint with adapter keys but no config (error)
  - Linear-probe full checkpoint without config (heads reconstructed from metadata)
  - Non-dict checkpoint payload (clean ValueError, not TypeError)
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
    model = AlphaGenome(
        num_organisms=1,
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


@pytest.mark.integration
class TestLoadFinetunedModel:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        gc.collect()

    def test_full_checkpoint_with_embedded_config(self, base_weights_path):
        """Path B: full checkpoint with embedded TransferConfig reconstructs adapters."""
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
        # LoRA adapters should have been rebuilt
        lora_names = [n for n, _ in model.named_parameters() if "lora_" in n]
        assert len(lora_names) > 0
        assert meta["head_names"] == ["my_atac"]
        assert meta["epoch"] == 3

    def test_full_checkpoint_adapter_keys_without_config_raises(
        self, base_weights_path,
    ):
        """Path C guard: adapter keys without a TransferConfig must error clearly."""
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

    def test_linear_probe_full_checkpoint_without_config(self, base_weights_path):
        """Path C success: linear-probe checkpoint has no adapter keys and
        heads are reconstructed from modality/track_names/resolutions metadata."""
        model = _make_model()
        model.load_state_dict(
            torch.load(base_weights_path, weights_only=True), strict=False
        )
        model = remove_all_heads(model)
        model.heads["my_atac"] = create_finetuning_head(
            assay_type="atac", n_tracks=4, resolutions=(128,),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "linprobe.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "modality": "my_atac",
                    "track_names": ["t1", "t2", "t3", "t4"],
                    "resolutions": (128,),
                    "epoch": 1,
                    "val_loss": 0.5,
                },
                ckpt_path,
            )

            loaded, meta = load_finetuned_model(ckpt_path, base_weights_path)

        assert "my_atac" in loaded.heads
        # No adapters should have been created
        assert not [n for n, _ in loaded.named_parameters() if "lora_" in n]
        assert meta["head_names"] == ["my_atac"]

    def test_non_dict_checkpoint_raises_value_error(self, base_weights_path):
        """Guard: torch.save of a raw state_dict must produce ValueError, not TypeError."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "raw_state.pth"
            # A raw state_dict is an OrderedDict subclass of dict — also try a
            # genuinely non-dict payload (a single tensor) to exercise the guard.
            torch.save(torch.zeros(3), ckpt_path)

            with pytest.raises(ValueError, match="Expected dict"):
                load_finetuned_model(ckpt_path, base_weights_path)
