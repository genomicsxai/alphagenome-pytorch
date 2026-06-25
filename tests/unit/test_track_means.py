"""Tests for compute_track_means strand-pair averaging (B3.3).

Verifies the optional `strand_pair_groups` parameter averages per-track
nonzero means within each (plus, minus) pair so the two strands share a
scaling factor, matching upstream AlphaGenome's official scaling semantics.
The default (None) is asserted to keep existing behavior bit-equal.
"""

import pytest
import torch

pyBigWig = pytest.importorskip("pyBigWig")

from alphagenome_pytorch.extensions.finetuning.datasets import compute_track_means


@pytest.fixture
def synthetic_bigwig_pair(tmp_path):
    """Create two tiny bigwigs whose nonzero means are exactly (2.0, 4.0)."""
    chrom = "chrFake"
    chrom_size = 4096

    bw1_path = str(tmp_path / "track_plus.bw")
    bw2_path = str(tmp_path / "track_minus.bw")

    # Track 1: ten positions all = 2.0 inside the central window → nonzero_mean = 2.0
    bw1 = pyBigWig.open(bw1_path, "w")
    bw1.addHeader([(chrom, chrom_size)])
    bw1.addEntries(
        [chrom] * 10,
        list(range(2000, 2010)),
        ends=list(range(2001, 2011)),
        values=[2.0] * 10,
    )
    bw1.close()

    # Track 2: ten positions all = 4.0 → nonzero_mean = 4.0
    bw2 = pyBigWig.open(bw2_path, "w")
    bw2.addHeader([(chrom, chrom_size)])
    bw2.addEntries(
        [chrom] * 10,
        list(range(2000, 2010)),
        ends=list(range(2001, 2011)),
        values=[4.0] * 10,
    )
    bw2.close()

    bed_path = str(tmp_path / "regions.bed")
    # One window centered at 2048 with width 2048 covers [1024, 3072) — includes
    # all our nonzero positions (2000..2009).
    with open(bed_path, "w") as f:
        f.write(f"{chrom}\t1024\t3072\n")

    return [bw1_path, bw2_path], bed_path


@pytest.mark.unit
class TestComputeTrackMeansStrandPair:

    def test_default_no_pairing_preserves_per_track_means(self, synthetic_bigwig_pair):
        """Without strand_pair_groups, each track keeps its own nonzero mean."""
        bigwigs, bed = synthetic_bigwig_pair
        means = compute_track_means(
            bigwig_files=bigwigs,
            bed_file=bed,
            sequence_length=2048,
            resolution=1,
        )
        assert means.shape == (1, 2)
        assert torch.isclose(means[0, 0], torch.tensor(2.0))
        assert torch.isclose(means[0, 1], torch.tensor(4.0))

    def test_pairing_averages_paired_strands(self, synthetic_bigwig_pair):
        """With strand_pair_groups=[(0, 1)], both indices receive the average (3.0)."""
        bigwigs, bed = synthetic_bigwig_pair
        means = compute_track_means(
            bigwig_files=bigwigs,
            bed_file=bed,
            sequence_length=2048,
            resolution=1,
            strand_pair_groups=[(0, 1)],
        )
        assert means.shape == (1, 2)
        assert torch.isclose(means[0, 0], torch.tensor(3.0))
        assert torch.isclose(means[0, 1], torch.tensor(3.0))

    @pytest.mark.parametrize(
        "bad_groups, match",
        [
            ([(0, 2)], "out of range"),        # index >= n_tracks
            ([(-1, 0)], "out of range"),       # negative would silently wrap
            ([(0, 0)], "two distinct tracks"),  # self-pair
            ([(0, 1), (1, 0)], "more than one pair"),  # reused index
        ],
    )
    def test_invalid_strand_pair_groups_raise(self, synthetic_bigwig_pair, bad_groups, match):
        bigwigs, bed = synthetic_bigwig_pair
        with pytest.raises(ValueError, match=match):
            compute_track_means(
                bigwig_files=bigwigs,
                bed_file=bed,
                sequence_length=2048,
                resolution=1,
                strand_pair_groups=bad_groups,
            )
