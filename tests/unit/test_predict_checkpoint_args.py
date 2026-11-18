"""``agt predict`` checkpoint argument handling.

Two behaviours are pinned here:

- ``--checkpoint`` accepts the same forms as ``agt serve`` — a file, a bundle
  directory, or a ``local:``/``file:``/``hf://`` URI — rather than only a path.
- ``--model`` is required exactly when the checkpoint is delta-shaped, and the
  error says which base weights are wanted.

Only argument validation is exercised, so no AlphaGenome is constructed.
"""

import argparse
import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch.cli.predict import _validate_checkpoint_arg
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    export_delta_weights,
    save_checkpoint,
    save_delta_checkpoint,
)
from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    prepare_for_transfer,
)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)


def _args(**kw) -> argparse.Namespace:
    kw.setdefault("model", None)
    return argparse.Namespace(**kw)


@pytest.fixture
def artifacts():
    """A full checkpoint, a delta checkpoint, a bundle, and base weights."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = _Tiny()
        full = tmp / "best_model.pth"
        save_checkpoint(
            path=full,
            epoch=1,
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            val_loss=0.1,
            track_names={"atac": ["t"]},
            modality=["atac"],
            resolutions={"atac": (128,)},
        )

        config = TransferConfig(mode=["lora"], lora_rank=2, lora_targets=["q_proj"])
        lora = prepare_for_transfer(_Tiny(), config)
        delta = tmp / "best_model.delta.pth"
        save_delta_checkpoint(
            path=delta,
            model=lora,
            config=config,
            epoch=1,
            base_model_weights_hash="feedface",
        )

        bundle = None
        if pytest.importorskip("safetensors", reason="bundles need safetensors"):
            from alphagenome_pytorch.extensions.serving.bundle import (
                MANIFEST_FILENAME,
            )

            bundle = tmp / "bundle"
            bundle.mkdir()
            export_delta_weights(
                lora, config, bundle / "adapter.safetensors", format="safetensors"
            )
            (bundle / MANIFEST_FILENAME).write_text(
                json.dumps(
                    {
                        "id": "b",
                        "schema_version": 1,
                        "base_model_hash": "h",
                        "base_model_weights_hash": "feedface",
                    }
                )
            )

        base = tmp / "base.pth"
        torch.save(_Tiny().state_dict(), base)

        yield {"full": full, "delta": delta, "bundle": bundle, "base": base, "tmp": tmp}


class TestModelRequirement:
    def test_full_checkpoint_needs_no_base_weights(self, artifacts):
        """A full checkpoint carries the whole model, so --model is optional."""
        _validate_checkpoint_arg(_args(checkpoint=str(artifacts["full"])))

    def test_delta_checkpoint_requires_base_weights(self, artifacts):
        with pytest.raises(ValueError, match="--model is required"):
            _validate_checkpoint_arg(_args(checkpoint=str(artifacts["delta"])))

    def test_error_names_the_expected_base_weights(self, artifacts):
        with pytest.raises(ValueError, match="feedface"):
            _validate_checkpoint_arg(_args(checkpoint=str(artifacts["delta"])))

    def test_delta_checkpoint_accepted_with_base_weights(self, artifacts):
        _validate_checkpoint_arg(
            _args(checkpoint=str(artifacts["delta"]), model=str(artifacts["base"]))
        )


class TestAcceptedCheckpointForms:
    def test_bundle_directory_is_accepted(self, artifacts):
        _validate_checkpoint_arg(
            _args(checkpoint=str(artifacts["bundle"]), model=str(artifacts["base"]))
        )

    def test_bundle_directory_still_requires_base_weights(self, artifacts):
        with pytest.raises(ValueError, match="--model is required"):
            _validate_checkpoint_arg(_args(checkpoint=str(artifacts["bundle"])))

    def test_hf_uri_is_not_rejected_up_front(self, artifacts):
        """Remote bundles are classified after download, not before."""
        _validate_checkpoint_arg(
            _args(checkpoint="hf://org/repo", model=str(artifacts["base"]))
        )

    def test_missing_path_still_raises(self, artifacts):
        missing = artifacts["tmp"] / "nope.pth"
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            _validate_checkpoint_arg(_args(checkpoint=str(missing)))
