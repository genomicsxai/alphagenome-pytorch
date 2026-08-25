"""
Unit tests for fine-tuning datasets.

Tests the fine-tuning datasets with mock data.
"""

import pytest
import torch


class TestGenomicDataset:
    """Tests for GenomicDataset with mock data."""

    def test_dataset_getitem_shapes(self, mock_data_dir):
        """Test __getitem__ returns correct shapes."""
        from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset

        dataset = GenomicDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[
                str(mock_data_dir / "mock_rnaseq_track1.bw"),
                str(mock_data_dir / "mock_rnaseq_track2.bw"),
            ],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(128,),
        )

        seq, targets = dataset[0]

        assert seq.shape == (131072, 4)
        assert seq.dtype == torch.float32
        assert 128 in targets
        assert targets[128].shape == (1024, 2)  # 2 tracks

    def test_dataset_dual_resolution(self, mock_data_dir):
        """Test dataset with both 1bp and 128bp resolutions."""
        from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset

        dataset = GenomicDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[
                str(mock_data_dir / "mock_rnaseq_track1.bw"),
            ],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(1, 128),
        )

        seq, targets = dataset[0]

        assert seq.shape == (131072, 4)
        assert 1 in targets
        assert 128 in targets
        assert targets[1].shape == (131072, 1)
        assert targets[128].shape == (1024, 1)

    def test_dataset_length(self, mock_data_dir):
        """Test dataset length matches BED file entries."""
        from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset

        dataset = GenomicDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[str(mock_data_dir / "mock_rnaseq_track1.bw")],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(128,),
        )

        # Mock BED has 20 positions, some may be filtered at boundaries
        assert len(dataset) > 0
        assert len(dataset) <= 20


class TestRNASeqDataset:
    """Tests for RNASeqDataset alias."""

    def test_dataset_getitem_shapes(self, mock_data_dir):
        """Test __getitem__ returns correct shapes."""
        from alphagenome_pytorch.extensions.finetuning.datasets import RNASeqDataset

        dataset = RNASeqDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[
                str(mock_data_dir / "mock_rnaseq_track1.bw"),
                str(mock_data_dir / "mock_rnaseq_track2.bw"),
            ],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(128,),
        )

        seq, targets = dataset[0]

        assert seq.shape == (131072, 4)
        assert seq.dtype == torch.float32
        assert 128 in targets
        assert targets[128].shape == (1024, 2)  # 2 tracks

    def test_dataset_dual_resolution(self, mock_data_dir):
        """Test dataset with both 1bp and 128bp resolutions."""
        from alphagenome_pytorch.extensions.finetuning.datasets import RNASeqDataset

        dataset = RNASeqDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[
                str(mock_data_dir / "mock_rnaseq_track1.bw"),
            ],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(1, 128),
        )

        seq, targets = dataset[0]

        assert seq.shape == (131072, 4)
        assert 1 in targets
        assert 128 in targets
        assert targets[1].shape == (131072, 1)
        assert targets[128].shape == (1024, 1)


class TestATACDataset:
    """Tests for ATACDataset alias."""

    def test_dataset_getitem_shapes(self, mock_data_dir):
        """Test __getitem__ returns correct shapes."""
        from alphagenome_pytorch.extensions.finetuning.datasets import ATACDataset

        dataset = ATACDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[
                str(mock_data_dir / "mock_atac_track1.bw"),
                str(mock_data_dir / "mock_atac_track2.bw"),
            ],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(128,),
        )

        seq, targets = dataset[0]

        assert seq.shape == (131072, 4)
        assert seq.dtype == torch.float32
        assert 128 in targets
        assert targets[128].shape == (1024, 2)  # 2 tracks at 128bp

    def test_dataset_dual_resolution(self, mock_data_dir):
        """Test dataset with both 1bp and 128bp resolutions."""
        from alphagenome_pytorch.extensions.finetuning.datasets import ATACDataset

        dataset = ATACDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[
                str(mock_data_dir / "mock_atac_track1.bw"),
            ],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(1, 128),
        )

        seq, targets = dataset[0]

        assert seq.shape == (131072, 4)
        assert 1 in targets
        assert 128 in targets
        assert targets[1].shape == (131072, 1)
        assert targets[128].shape == (1024, 1)


class TestCollateWithDatasets:
    """Tests for collate_genomic with actual datasets."""

    def test_collate_with_rnaseq_dataset(self, mock_data_dir):
        """Test collate function works with RNASeqDataset."""
        from torch.utils.data import DataLoader
        from alphagenome_pytorch.extensions.finetuning import (
            RNASeqDataset,
            collate_genomic,
        )

        dataset = RNASeqDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[str(mock_data_dir / "mock_rnaseq_track1.bw")],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(128,),
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_genomic,
        )

        sequences, targets_dict = next(iter(loader))

        assert sequences.shape[0] == 2  # batch size
        assert sequences.shape[1] == 131072
        assert sequences.shape[2] == 4
        assert 128 in targets_dict
        assert targets_dict[128].shape[0] == 2

    def test_collate_with_atac_dataset(self, mock_data_dir):
        """Test collate function works with ATACDataset."""
        from torch.utils.data import DataLoader
        from alphagenome_pytorch.extensions.finetuning import (
            ATACDataset,
            collate_genomic,
        )

        dataset = ATACDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[str(mock_data_dir / "mock_atac_track1.bw")],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(128,),
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_genomic,
        )

        sequences, targets_dict = next(iter(loader))

        assert sequences.shape[0] == 2  # batch size
        assert sequences.shape[1] == 131072
        assert sequences.shape[2] == 4
        assert 128 in targets_dict
        assert targets_dict[128].shape[0] == 2


class TestGenomicDatasetMultiprocessing:
    """Tests for GenomicDataset with multi-process DataLoader."""

    def test_dataloader_multiprocessing(self, mock_data_dir):
        """Test that DataLoader with num_workers > 0 works correctly."""
        from torch.utils.data import DataLoader
        from alphagenome_pytorch.extensions.finetuning import (
            GenomicDataset,
            collate_genomic,
        )

        dataset = GenomicDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[str(mock_data_dir / "mock_atac_track1.bw")],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(128,),
        )

        # Use 2 workers to test multiprocessing safety
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=2,
            collate_fn=collate_genomic,
            shuffle=True,
        )

        # Iterate through a few batches
        batch_count = 0
        for sequences, targets_dict in loader:
            assert sequences.shape[0] <= 4
            assert sequences.shape[1] == 131072
            assert 128 in targets_dict
            batch_count += 1
            if batch_count >= 2:
                break

        assert batch_count > 0


@pytest.mark.unit
class TestAugmentation:
    """Train-time reverse-complement (strand-aware) and random-shift augmentation."""

    class _FakeRng:
        """Deterministic stand-in for the per-worker RNG so tests can force rc/shift."""
        def __init__(self, do_rc: bool, shift: int = 0):
            self._rc = do_rc
            self._shift = shift

        def random(self):
            return 0.0 if self._rc else 0.9          # < 0.5 -> rc applied

        def integers(self, lo, hi):                  # __getitem__ calls integers(lo, hi+1)
            return max(lo, min(self._shift, hi - 1))

    def _dataset(self, mock_data_dir, tracks, **kw):
        from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset
        return GenomicDataset(
            genome_fasta=str(mock_data_dir / "mock_genome.fa"),
            bigwig_files=[str(mock_data_dir / t) for t in tracks],
            bed_file=str(mock_data_dir / "mock_positions.bed"),
            resolutions=(1,),
            **kw,
        )

    def test_no_augment_matches_plain(self, mock_data_dir):
        """augment off (val/test path) returns the un-transformed sample."""
        ds = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"])
        aug = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"],
                            augment_rc=True, augment_shift_bp=100)
        aug._aug_rng = self._FakeRng(do_rc=False, shift=0)   # no rc, no shift this draw
        s0, t0 = ds[0]
        s1, t1 = aug[0]
        assert torch.equal(s0, s1)
        assert torch.equal(t0[1], t1[1])

    def test_rc_unstranded_reverses_seq_and_target(self, mock_data_dir):
        """Unstranded RC: sequence reverse-complemented, target reversed along length only."""
        ds = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"])
        aug = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"], augment_rc=True)
        aug._aug_rng = self._FakeRng(do_rc=True, shift=0)
        s0, t0 = ds[0]
        s1, t1 = aug[0]
        assert torch.equal(s1, torch.flip(s0, dims=[0, 1]))          # reverse-complement
        assert torch.equal(t1[1], torch.flip(t0[1], dims=[0]))       # length reverse, no swap

    def test_rc_stranded_swaps_pairs(self, mock_data_dir):
        """Stranded RC: target reversed along length AND +/- channels swapped."""
        tracks = ["mock_rnaseq_track1.bw", "mock_rnaseq_track2.bw"]
        ds = self._dataset(mock_data_dir, tracks)
        aug = self._dataset(mock_data_dir, tracks, augment_rc=True, strand_pairs=[(0, 1)])
        aug._aug_rng = self._FakeRng(do_rc=True, shift=0)
        _, t0 = ds[0]
        _, t1 = aug[0]
        expected = torch.flip(t0[1], dims=[0])[:, [1, 0]]            # reverse + swap channels
        assert torch.equal(t1[1], expected)

    def test_shift_changes_crop_and_stays_in_bounds(self, mock_data_dir):
        """A forced shift produces a different crop; large shifts are clamped, never erroring."""
        base = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"],
                             augment_shift_bp=100)
        base._aug_rng = self._FakeRng(do_rc=False, shift=0)
        shifted = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"],
                               augment_shift_bp=100)
        shifted._aug_rng = self._FakeRng(do_rc=False, shift=100)
        s_base, _ = base[0]
        s_shift, _ = shifted[0]
        assert s_base.shape == s_shift.shape
        assert not torch.equal(s_base, s_shift)                     # crop moved

        huge = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"],
                            augment_shift_bp=10_000_000)
        huge._aug_rng = self._FakeRng(do_rc=False, shift=10_000_000)
        s_huge, _ = huge[0]                                          # clamped, no crash
        assert s_huge.shape == s_base.shape
