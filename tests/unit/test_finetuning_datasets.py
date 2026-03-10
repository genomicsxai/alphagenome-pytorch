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
