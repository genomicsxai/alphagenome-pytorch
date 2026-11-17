"""Tests for on-the-fly BigWig depth normalization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pyBigWig = pytest.importorskip("pyBigWig")

from alphagenome_pytorch.extensions.finetuning.args import parse_args
from alphagenome_pytorch.extensions.finetuning.datasets import (
    GenomicDataset,
    compute_track_means,
)
from alphagenome_pytorch.extensions.finetuning.runner import (
    compute_modality_track_scales,
    create_datasets,
)
from alphagenome_pytorch.extensions.finetuning.signal_scaling import (
    bigwig_total_signal,
    track_scale_factors,
)
from alphagenome_pytorch.heads import targets_scaling

CHROM = "chr1"
CHROM_SIZE = 8_192


def _write_bw(path: Path, starts, ends, values) -> str:
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(CHROM, CHROM_SIZE)])
    bw.addEntries([CHROM] * len(starts), list(starts), ends=list(ends), values=list(values))
    bw.close()
    return str(path)


@pytest.fixture
def data(tmp_path):
    """Plus track (total 300), minus-strand track (total 100), FASTA, BED."""
    plus = _write_bw(tmp_path / "plus.bw", [4000], [4100], [3.0])
    minus = _write_bw(tmp_path / "minus.bw", [4000], [4100], [1.0])
    fasta = tmp_path / "genome.fa"
    fasta.write_text(f">{CHROM}\n{'A' * CHROM_SIZE}\n")
    bed = tmp_path / "regions.bed"
    bed.write_text(f"{CHROM}\t3072\t5120\n")
    return [plus, minus], str(fasta), str(bed)


@pytest.mark.unit
class TestTrackScaleFactors:
    def test_unpaired_tracks_scale_on_their_own(self):
        assert track_scale_factors([50.0, 200.0], 100.0) == [2.0, 0.5]

    def test_pair_shares_one_factor(self):
        factors = track_scale_factors([300.0, 100.0, 50.0], 100.0, [(0, 1)])
        assert factors == [0.25, 0.25, 2.0]

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="zero total"):
            track_scale_factors([0.0], 100.0)


    @pytest.mark.parametrize("pairs", [[(0, 0)], [(0, 3)], [(0, 1), (1, 2)]])
    def test_invalid_pairs_are_rejected(self, pairs):
        with pytest.raises(ValueError, match="Invalid strand pair"):
            track_scale_factors([300.0, 100.0, 50.0], 100.0, pairs)


@pytest.mark.unit
class TestParseTargetTotal:
    BASE = [
        "finetune.py", "--genome", "hg38.fa", "--train-bed", "train.bed",
        "--val-bed", "val.bed", "--pretrained-weights", "model.pth",
        "--modality", "rna_seq", "--bigwig", "p.bw", "m.bw",
    ]

    def test_suffix_is_parsed(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self.BASE + ["--target-total", "100M"])
        assert parse_args().target_total == pytest.approx(1e8)

    def test_default_is_none(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self.BASE)
        assert parse_args().target_total is None

    @pytest.mark.parametrize("bad", ["lots", "0", "1e999"])
    def test_bad_value_errors(self, monkeypatch, bad):
        monkeypatch.setattr(sys, "argv", self.BASE + ["--target-total", bad])
        with pytest.raises(SystemExit):
            parse_args()


@pytest.mark.unit
class TestModalityTrackScales:
    def test_rna_pairs_share_other_modalities_do_not(self, data):
        (plus, minus), _, _ = data
        args = argparse.Namespace(
            target_total=100.0,
            modality_to_bigwigs={"rna_seq": [plus, minus], "cage": [plus, minus]},
            modality_strand_pairs={"rna_seq": [(0, 1)], "cage": [(0, 1)]},
        )
        scales = compute_modality_track_scales(args)
        # RNA-seq: a 400-signal pair shares one factor.
        assert scales["rna_seq"] == pytest.approx([0.25, 0.25])
        # cage: per strand, as in the paper, despite strand_pairs.
        assert scales["cage"] == pytest.approx([100 / 300, 1.0])

    @pytest.mark.parametrize("modality", ["chip_tf", "chip_histone"])
    def test_chip_is_left_unscaled(self, data, modality):
        """ChIP tracks are fold change over control; a depth factor would
        break their units, so they get no entry (and are not even read)."""
        (plus, _), _, _ = data
        args = argparse.Namespace(
            target_total=100.0,
            modality_to_bigwigs={modality: ["unopened-chip.bw"], "atac": [plus]},
            modality_strand_pairs={},
        )
        scales = compute_modality_track_scales(args)
        assert modality not in scales
        assert scales["atac"] == pytest.approx([100 / 300])


@pytest.mark.unit
class TestScaledTargets:
    def test_dataset_scales_targets_at_every_resolution(self, data):
        bigwigs, fasta, bed = data
        kw = dict(genome_fasta=fasta, bigwig_files=bigwigs, bed_file=bed,
                  resolutions=(1, 128), sequence_length=2048)
        _, raw = GenomicDataset(**kw)[0]
        _, scaled = GenomicDataset(**kw, track_scales=[0.5, 2.0])[0]
        for res in (1, 128):
            expected = raw[res] * torch.tensor([0.5, 2.0])
            assert torch.allclose(scaled[res], expected)

    def test_dataset_rejects_wrong_length(self, data):
        bigwigs, fasta, bed = data
        with pytest.raises(ValueError, match="track_scales"):
            GenomicDataset(genome_fasta=fasta, bigwig_files=bigwigs, bed_file=bed,
                           sequence_length=2048, track_scales=[1.0])

    def test_model_space_targets_are_unchanged(self, data):
        """Targets and means move together, so the loss sees the same values."""
        bigwigs, fasta, bed = data
        plus = bigwigs[:1]
        c = 100.0 / bigwig_total_signal(plus[0])
        kw = dict(genome_fasta=fasta, bigwig_files=plus, bed_file=bed,
                  resolutions=(1,), sequence_length=2048)
        _, raw = GenomicDataset(**kw)[0]
        _, scaled = GenomicDataset(**kw, track_scales=[c])[0]
        means = compute_track_means(plus, bed, sequence_length=2048)
        scaled_means = compute_track_means(plus, bed, sequence_length=2048, track_scales=[c])
        assert torch.allclose(scaled_means, means * c)

        x = targets_scaling(raw[1][None], means, resolution=1, apply_squashing=True)
        y = targets_scaling(scaled[1][None], scaled_means, resolution=1, apply_squashing=True)
        assert torch.allclose(x, y, rtol=1e-5)


@pytest.mark.unit
def test_normalized_track_rejects_fractional_negative_signal(tmp_path):
    path = _write_bw(tmp_path / "minus.bw", [10, 500], [20, 510], [5.0, -0.3])
    args = argparse.Namespace(
        target_total=100.0,
        modality_to_bigwigs={"rna_seq": [path]},
        modality_strand_pairs={},
    )
    with pytest.raises(ValueError, match="minus.bw contains negative targets"):
        compute_modality_track_scales(args)


@pytest.mark.unit
def test_normalized_pair_reports_an_empty_bigwig(data, tmp_path):
    (plus, _), _, _ = data
    empty = str(tmp_path / "empty.bw")
    with pyBigWig.open(empty, "w") as bw:
        bw.addHeader([(CHROM, CHROM_SIZE)])
    args = argparse.Namespace(
        target_total=100.0,
        modality_to_bigwigs={"rna_seq": [plus, empty]},
        modality_strand_pairs={"rna_seq": [(0, 1)]},
    )
    with pytest.raises(ValueError, match="empty.bw contains no signal intervals"):
        compute_modality_track_scales(args)


@pytest.mark.unit
def test_train_and_validation_share_the_same_factors(data, monkeypatch):
    bigwigs, fasta, bed = data
    monkeypatch.setattr(sys, "argv", [
        "agt finetune", "--genome", fasta, "--train-bed", bed, "--val-bed", bed,
        "--pretrained-weights", "model.pth", "--modality", "rna_seq",
        "--bigwig", *bigwigs, "--sequence-length", "2048",
        "--target-total", "100", "--strand-pairs", "rna_seq:auto",
    ])
    args = parse_args()
    factors = compute_modality_track_scales(args)
    raw_train, _, _, _ = create_datasets(args, 0)
    train, val, _, _ = create_datasets(args, 0, modality_track_scales=factors)
    raw = raw_train[0][1]["rna_seq"][1]
    torch.testing.assert_close(train[0][1]["rna_seq"][1], raw * 0.25)
    torch.testing.assert_close(val[0][1]["rna_seq"][1], raw * 0.25)
