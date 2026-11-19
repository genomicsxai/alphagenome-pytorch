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
        """get() with no shift/rc on an augmenting dataset returns the plain sample."""
        ds = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"])
        aug = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"],
                            augment_rc=True, augment_shift_bp=100)
        s0, t0 = ds[0]
        s1, t1 = aug.get(0)
        assert torch.equal(s0, s1)
        assert torch.equal(t0[1], t1[1])

    def test_rc_unstranded_reverses_seq_and_target(self, mock_data_dir):
        """Unstranded RC: sequence reverse-complemented, target reversed along length only."""
        ds = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"])
        aug = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"], augment_rc=True)
        s0, t0 = ds[0]
        s1, t1 = aug.get(0, do_rc=True)
        assert torch.equal(s1, torch.flip(s0, dims=[0, 1]))          # reverse-complement
        assert torch.equal(t1[1], torch.flip(t0[1], dims=[0]))       # length reverse, no swap

    def test_rc_stranded_swaps_pairs(self, mock_data_dir):
        """Stranded RC: target reversed along length AND +/- channels swapped."""
        tracks = ["mock_rnaseq_track1.bw", "mock_rnaseq_track2.bw"]
        ds = self._dataset(mock_data_dir, tracks)
        aug = self._dataset(mock_data_dir, tracks, augment_rc=True, strand_pairs=[(0, 1)])
        _, t0 = ds[0]
        _, t1 = aug.get(0, do_rc=True)
        expected = torch.flip(t0[1], dims=[0])[:, [1, 0]]            # reverse + swap channels
        assert torch.equal(t1[1], expected)

    def test_shift_slides_the_window(self, mock_data_dir):
        """A shift of +k moves the crop k bp downstream for sequence and targets alike."""
        aug = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"], augment_shift_bp=100)
        s_base, t_base = aug.get(0)
        s_shift, t_shift = aug.get(0, shift=100)
        assert torch.equal(s_shift[:-100], s_base[100:])
        assert torch.equal(t_shift[1][:-100], t_base[1][100:])

    def test_sample_augmentation_stays_in_bounds(self, mock_data_dir):
        """Draws are clamped to the chromosome even for a shift larger than it."""
        import numpy as np
        huge = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"],
                             augment_rc=True, augment_shift_bp=10_000_000)
        rng = np.random.default_rng(0)
        rc_seen = set()
        for idx in range(len(huge)):
            chrom, start, end = huge._positions_list[idx]
            for _ in range(20):
                shift, do_rc = huge.sample_augmentation(idx, rng)
                assert 0 <= start + shift and end + shift <= huge._chrom_sizes[chrom]
                rc_seen.add(do_rc)
        assert rc_seen == {True, False}
        seq, _ = huge.get(idx, shift=shift, do_rc=do_rc)            # a clamped draw reads fine
        assert seq.shape[0] == huge.sequence_length

    def test_get_rejects_out_of_bounds_shift(self, mock_data_dir):
        aug = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"], augment_shift_bp=100)
        with pytest.raises(ValueError, match="outside the chromosome"):
            aug.get(0, shift=10_000_000)

    def test_no_draw_when_augmentation_off(self, mock_data_dir):
        import numpy as np
        ds = self._dataset(mock_data_dir, ["mock_rnaseq_track1.bw"])
        assert not ds.augments
        assert ds.sample_augmentation(0, np.random.default_rng(0)) == (0, False)


@pytest.fixture
def a_indicator_data(tmp_path):
    """Random genome plus a bigwig whose signal is 1 exactly where the base is 'A'.

    An aligned sample has target == the sequence's A channel, or its T channel
    after reverse-complement. Any sequence/target misalignment, including a
    shift or RC drawn separately for each, breaks that equality.
    """
    import numpy as np
    pyBigWig = pytest.importorskip("pyBigWig")
    pytest.importorskip("pyfaidx")
    rng = np.random.default_rng(0)
    chrom_len = 40_000
    bases = rng.choice(list("ACGT"), chrom_len)
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "".join(bases) + "\n")
    bw_path = tmp_path / "a.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", chrom_len)])
    bw.addEntries("chr1", 0, values=(bases == "A").astype(float).tolist(), span=1, step=1)
    bw.close()
    bed = tmp_path / "train.bed"
    bed.write_text("".join(f"chr1\t{s}\t{s + 4096}\n" for s in range(2_000, 34_000, 4_000)))
    return {"fasta": str(fasta), "bigwig": str(bw_path), "bed": str(bed), "bases": bases}


@pytest.mark.unit
class TestMultimodalAugmentation:
    """MultimodalDataset draws augmentation once per sample and shares it (PR #58 review)."""

    SEQ_LEN = 4096
    SHIFT = 500

    def _ds(self, data, **kw):
        from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset
        return GenomicDataset(
            genome_fasta=data["fasta"], bigwig_files=[data["bigwig"]], bed_file=data["bed"],
            resolutions=(1,), sequence_length=self.SEQ_LEN, **kw,
        )

    def _mm(self, data, modalities=("rna_seq",), **kw):
        from alphagenome_pytorch.extensions.finetuning.datasets import MultimodalDataset
        return MultimodalDataset({m: self._ds(data, **kw) for m in modalities})

    @staticmethod
    def _rc_state(seq, target):
        """'fwd' or 'rc' when target lines up with the sequence, None when misaligned."""
        if torch.equal(target, seq[:, 0]):
            return "fwd"
        if torch.equal(target, seq[:, 3]):
            return "rc"
        return None

    def test_sequence_and_targets_come_from_one_draw(self, a_indicator_data):
        torch.manual_seed(0)
        mm = self._mm(a_indicator_data, augment_rc=True, augment_shift_bp=self.SHIFT)
        states = []
        for i in range(48):
            seq, targets = mm[i % len(mm)]
            state = self._rc_state(seq, targets["rna_seq"][1][:, 0])
            assert state is not None, f"sample {i}: targets misaligned with sequence"
            states.append(state)
        assert set(states) == {"fwd", "rc"}
        # The crop really moves between draws of the same window.
        assert len({mm[0][0].numpy().tobytes() for _ in range(10)}) > 2

    def test_every_modality_gets_the_same_draw(self, a_indicator_data):
        torch.manual_seed(0)
        mm = self._mm(a_indicator_data, modalities=("atac", "rna_seq"),
                      augment_rc=True, augment_shift_bp=self.SHIFT)
        for i in range(24):
            seq, targets = mm[i % len(mm)]
            atac, rna = targets["atac"][1][:, 0], targets["rna_seq"][1][:, 0]
            assert torch.equal(atac, rna)
            assert self._rc_state(seq, rna) is not None

    def test_return_coords_follow_the_shift(self, a_indicator_data):
        torch.manual_seed(0)
        mm = self._mm(a_indicator_data, augment_shift_bp=self.SHIFT)
        mm.return_coords = True
        bases = a_indicator_data["bases"]
        for i in range(12):
            seq, _, (chrom, start, end) = mm[i % len(mm)]
            expected_a = torch.from_numpy((bases[start:end] == "A").astype("float32"))
            assert torch.equal(seq[:, 0], expected_a)

    def test_mismatched_augment_configs_rejected(self, a_indicator_data):
        from alphagenome_pytorch.extensions.finetuning.datasets import MultimodalDataset
        with pytest.raises(ValueError, match="share augment_rc"):
            MultimodalDataset({
                "atac": self._ds(a_indicator_data, augment_rc=True),
                "rna_seq": self._ds(a_indicator_data),
            })

    def test_seeded_draws_are_reproducible(self, a_indicator_data):
        def draws(seed):
            torch.manual_seed(seed)
            mm = self._mm(a_indicator_data, augment_rc=True, augment_shift_bp=self.SHIFT)
            return [mm[i][0] for i in range(len(mm))]

        a, b, c = draws(123), draws(123), draws(456)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert not all(torch.equal(x, y) for x, y in zip(a, c))

    def test_rng_rebuilt_in_a_new_process(self, a_indicator_data):
        from alphagenome_pytorch.extensions.finetuning.datasets import _process_aug_rng
        mm = self._mm(a_indicator_data, augment_shift_bp=self.SHIFT)
        rng = _process_aug_rng(mm)
        assert _process_aug_rng(mm) is rng
        mm._aug_rng_pid = -1                       # as if forked into a DataLoader worker
        assert _process_aug_rng(mm) is not rng

    def test_aligned_through_dataloader_workers(self, a_indicator_data):
        from torch.utils.data import DataLoader
        from alphagenome_pytorch.extensions.finetuning.datasets import collate_multimodal
        mm = self._mm(a_indicator_data, augment_rc=True, augment_shift_bp=self.SHIFT)
        loader = DataLoader(mm, batch_size=2, num_workers=2, collate_fn=collate_multimodal)
        n = 0
        for seqs, targets in loader:
            for seq, tgt in zip(seqs, targets["rna_seq"][1][..., 0]):
                assert self._rc_state(seq, tgt) is not None
                n += 1
        assert n == len(mm)


@pytest.mark.unit
class TestResolveRcStrandPairs:
    """Strand pairs swapped under RC: explicit --strand-pairs, else derived from strands."""

    def _resolve(self, *a, **kw):
        from alphagenome_pytorch.extensions.finetuning.datasets import resolve_rc_strand_pairs
        return resolve_rc_strand_pairs(*a, **kw)

    def test_unstranded_needs_no_pairs(self):
        assert self._resolve(3) is None
        assert self._resolve(3, strands="...") is None

    def test_derived_from_strands_by_order(self):
        assert self._resolve(4, strands="+-+-") == [(0, 1), (2, 3)]
        assert self._resolve(4, strands="++--") == [(0, 2), (1, 3)]
        assert self._resolve(3, strands="+.-") == [(0, 2)]

    def test_unbalanced_strands_rejected(self):
        with pytest.raises(ValueError, match="cannot derive"):
            self._resolve(3, strands="++-")

    def test_explicit_pairs_win(self):
        assert self._resolve(4, strands="+-+-", strand_pairs=[(0, 3), (2, 1)]) == [(0, 3), (2, 1)]
        assert self._resolve(2, strand_pairs=[(0, 1)]) == [(0, 1)]

    def test_explicit_pair_must_join_plus_and_minus(self):
        with pytest.raises(ValueError, match="one '\\+' and one '-'"):
            self._resolve(4, strands="+-+-", strand_pairs=[(0, 2), (1, 3)])

    def test_explicit_pairs_must_cover_stranded_tracks(self):
        with pytest.raises(ValueError, match="not in any strand pair"):
            self._resolve(4, strands="+-+-", strand_pairs=[(0, 1)])

    def test_bad_indices_rejected(self):
        with pytest.raises(ValueError, match="distinct track indices"):
            self._resolve(2, strand_pairs=[(0, 2)])
        with pytest.raises(ValueError, match="distinct track indices"):
            self._resolve(3, strand_pairs=[(0, 1), (1, 2)])

    def test_strand_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="2 entries but there are 3 tracks"):
            self._resolve(3, strands="+-")
