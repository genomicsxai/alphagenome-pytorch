"""``--save-every 0`` turns per-epoch checkpoints off.

The epoch loop used to run ``epoch % args.save_every`` unguarded, so
``--save-every 0`` raised ZeroDivisionError right after the first best-model
save and killed the run.
"""

import pytest

from alphagenome_pytorch.extensions.finetuning.args import parse_args
from alphagenome_pytorch.extensions.finetuning.runner import (
    should_save_epoch_checkpoint,
)

REQUIRED_ARGS = [
    "--modality", "atac", "--bigwig", "a.bw",
    "--genome", "g.fa", "--train-bed", "t.bed", "--val-bed", "v.bed",
    "--pretrained-weights", "w.pth",
]
EPOCHS = range(1, 11)


class TestShouldSaveEpochCheckpoint:
    def test_zero_saves_no_epoch(self):
        assert not any(should_save_epoch_checkpoint(e, 0) for e in EPOCHS)

    def test_default_saves_every_epoch(self):
        assert all(should_save_epoch_checkpoint(e, 1) for e in EPOCHS)

    def test_saves_every_nth_epoch(self):
        saved = [e for e in EPOCHS if should_save_epoch_checkpoint(e, 3)]
        assert saved == [3, 6, 9]


class TestSaveEveryValidation:
    def test_zero_is_accepted(self):
        assert parse_args([*REQUIRED_ARGS, "--save-every", "0"]).save_every == 0

    def test_negative_is_rejected(self):
        with pytest.raises(SystemExit):
            parse_args([*REQUIRED_ARGS, "--save-every", "-1"])

    def test_negative_from_config_is_rejected(self, tmp_path):
        """The check runs after the YAML merge, so config values are caught too."""
        config = tmp_path / "config.yaml"
        config.write_text("save_every: -1\n")
        with pytest.raises(SystemExit):
            parse_args([*REQUIRED_ARGS, "--config", str(config)])
