"""Unit tests for multimodal support in scripts/finetune.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Import script module symbols directly (tests run from repository root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from finetune import (  # noqa: E402
    MultimodalDataset,
    collate_multimodal,
    parse_args,
)
from alphagenome_pytorch.extensions.finetuning.training import (  # noqa: E402
    _compute_multinomial_resolution,
)


def _required_cli_args() -> list[str]:
    """Return the minimal required CLI args for parse_args tests."""
    return [
        "finetune.py",
        "--genome",
        "hg38.fa",
        "--train-bed",
        "train.bed",
        "--val-bed",
        "val.bed",
        "--pretrained-weights",
        "model.pth",
    ]


@pytest.mark.unit
class TestComputeMultinomialResolution:
    """Tests for _compute_multinomial_resolution utility."""

    def test_default_8_segments(self):
        assert _compute_multinomial_resolution(256) == 32
        assert _compute_multinomial_resolution(1024) == 128
        assert _compute_multinomial_resolution(64) == 8

    def test_custom_segments(self):
        assert _compute_multinomial_resolution(256, num_segments=4) == 64
        assert _compute_multinomial_resolution(256, num_segments=16) == 16

    def test_min_segment_size(self):
        assert _compute_multinomial_resolution(64, min_segment_size=16) == 16
        assert _compute_multinomial_resolution(256, min_segment_size=16) == 32

    def test_small_sequence(self):
        assert _compute_multinomial_resolution(8) == 1
        assert _compute_multinomial_resolution(4) == 1


@pytest.mark.unit
class TestParseArgsMultimodal:
    """Tests for multimodal/task-weight parsing in scripts/finetune.py."""

    def test_parse_args_with_two_modalities_and_weights(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            _required_cli_args()
            + [
                "--modality",
                "atac",
                "--bigwig",
                "atac1.bw",
                "atac2.bw",
                "--modality",
                "rna_seq",
                "--bigwig",
                "rna1.bw",
                "--modality-weights",
                "atac:1.0,rna_seq:0.5",
                "--resolutions",
                "1,128",
            ],
        )

        args = parse_args()

        assert args.is_multimodal is True
        assert args.modalities == ["atac", "rna_seq"]
        assert args.modality_to_bigwigs["atac"] == ["atac1.bw", "atac2.bw"]
        assert args.modality_to_bigwigs["rna_seq"] == ["rna1.bw"]
        assert args.global_resolutions == (1, 128)
        assert args.modality_resolutions["atac"] == (1, 128)
        assert args.modality_resolutions["rna_seq"] == (1, 128)
        assert args.modality_weight_dict["atac"] == pytest.approx(1.0)
        assert args.modality_weight_dict["rna_seq"] == pytest.approx(0.5)

    def test_parse_args_missing_modality_weight_defaults_to_one(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            _required_cli_args()
            + [
                "--modality",
                "atac",
                "--bigwig",
                "atac1.bw",
                "--modality",
                "rna_seq",
                "--bigwig",
                "rna1.bw",
                "--modality-weights",
                "atac:2.0",
            ],
        )

        args = parse_args()
        assert args.modality_weight_dict["atac"] == pytest.approx(2.0)
        assert args.modality_weight_dict["rna_seq"] == pytest.approx(1.0)

    def test_parse_args_rejects_mismatched_modality_and_bigwig_groups(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            _required_cli_args()
            + [
                "--modality",
                "atac",
                "--modality",
                "rna_seq",
                "--bigwig",
                "atac1.bw",
            ],
        )

        with pytest.raises(SystemExit):
            parse_args()


@pytest.mark.unit
class TestMultimodalDataset:
    """Tests for MultimodalDataset wrapper."""

    def test_length(self):
        mock_ds1 = MagicMock()
        mock_ds1.__len__ = MagicMock(return_value=100)
        mock_ds1.__getitem__ = MagicMock(return_value=(torch.randn(256, 4), {128: torch.randn(256, 5)}))

        mock_ds2 = MagicMock()
        mock_ds2.__len__ = MagicMock(return_value=100)
        mock_ds2.__getitem__ = MagicMock(return_value=(torch.randn(256, 4), {128: torch.randn(256, 3)}))

        dataset = MultimodalDataset({"atac": mock_ds1, "rna_seq": mock_ds2})
        assert len(dataset) == 100

    def test_length_mismatch_raises(self):
        mock_ds1 = MagicMock()
        mock_ds1.__len__ = MagicMock(return_value=100)

        mock_ds2 = MagicMock()
        mock_ds2.__len__ = MagicMock(return_value=50)

        with pytest.raises(ValueError, match="same length"):
            MultimodalDataset({"atac": mock_ds1, "rna_seq": mock_ds2})

    def test_getitem_returns_all_modalities(self):
        seq = torch.randn(256, 4)
        targets1 = {128: torch.randn(256, 5)}
        targets2 = {128: torch.randn(256, 3)}

        mock_ds1 = MagicMock()
        mock_ds1.__len__ = MagicMock(return_value=10)
        mock_ds1.__getitem__ = MagicMock(return_value=(seq, targets1))

        mock_ds2 = MagicMock()
        mock_ds2.__len__ = MagicMock(return_value=10)
        mock_ds2.__getitem__ = MagicMock(return_value=(seq, targets2))

        dataset = MultimodalDataset({"atac": mock_ds1, "rna_seq": mock_ds2})
        result_seq, result_targets = dataset[0]

        assert torch.equal(result_seq, seq)
        assert "atac" in result_targets
        assert "rna_seq" in result_targets
        assert 128 in result_targets["atac"]
        assert 128 in result_targets["rna_seq"]


@pytest.mark.unit
class TestCollateMultimodal:
    """Tests for collate_multimodal function."""

    def test_collate_single_modality(self):
        batch = [
            (torch.randn(256, 4), {"atac": {128: torch.randn(256, 5)}}),
            (torch.randn(256, 4), {"atac": {128: torch.randn(256, 5)}}),
        ]

        sequences, modality_targets = collate_multimodal(batch)

        assert sequences.shape == (2, 256, 4)
        assert "atac" in modality_targets
        assert 128 in modality_targets["atac"]
        assert modality_targets["atac"][128].shape == (2, 256, 5)

    def test_collate_multiple_modalities(self):
        batch = [
            (
                torch.randn(256, 4),
                {
                    "atac": {128: torch.randn(256, 5)},
                    "rna_seq": {1: torch.randn(256, 3), 128: torch.randn(256, 3)},
                },
            ),
            (
                torch.randn(256, 4),
                {
                    "atac": {128: torch.randn(256, 5)},
                    "rna_seq": {1: torch.randn(256, 3), 128: torch.randn(256, 3)},
                },
            ),
        ]

        sequences, modality_targets = collate_multimodal(batch)

        assert sequences.shape == (2, 256, 4)
        assert "atac" in modality_targets
        assert "rna_seq" in modality_targets
        assert modality_targets["atac"][128].shape == (2, 256, 5)
        assert modality_targets["rna_seq"][1].shape == (2, 256, 3)
        assert modality_targets["rna_seq"][128].shape == (2, 256, 3)

    def test_collate_preserves_batch_order(self):
        seq1 = torch.ones(256, 4)
        seq2 = torch.zeros(256, 4)
        targets1 = torch.ones(256, 3)
        targets2 = torch.zeros(256, 3)

        batch = [
            (seq1, {"atac": {128: targets1}}),
            (seq2, {"atac": {128: targets2}}),
        ]

        sequences, modality_targets = collate_multimodal(batch)

        assert torch.equal(sequences[0], seq1)
        assert torch.equal(sequences[1], seq2)
        assert torch.equal(modality_targets["atac"][128][0], targets1)
        assert torch.equal(modality_targets["atac"][128][1], targets2)
