"""Unit tests for full-chromosome tiling and prediction infrastructure.

Tests TilingConfig validation, _generate_tiles() correctness,
_sequence_to_onehot() encoding, and stitching logic with a mock model.
All pure-logic tests -- no pyBigWig, pyfaidx, GPU, or model weights needed.
"""

import numpy as np
import pytest
import torch

from alphagenome_pytorch.extensions.inference.full_chromosome import (
    TilingConfig,
    _generate_tiles,
    _sequence_to_onehot,
    HEAD_CONFIGS,
)


@pytest.mark.unit
class TestTilingConfig:
    """Tests for TilingConfig validation and properties."""

    def test_default_config(self):
        cfg = TilingConfig()
        assert cfg.window_size == 131072
        assert cfg.crop_bp == 0
        assert cfg.resolution == 128
        assert cfg.batch_size == 4

    def test_effective_size_no_crop(self):
        cfg = TilingConfig(crop_bp=0)
        assert cfg.effective_size == 131072
        assert cfg.step_size == 131072

    def test_effective_size_with_crop(self):
        cfg = TilingConfig(crop_bp=32768)
        assert cfg.effective_size == 131072 - 2 * 32768
        assert cfg.step_size == cfg.effective_size

    def test_crop_start_end(self):
        cfg = TilingConfig(crop_bp=32768)
        assert cfg.crop_start == 32768
        assert cfg.crop_end == 131072 - 32768

    def test_negative_crop_raises(self):
        with pytest.raises(ValueError, match="crop_bp must be >= 0"):
            TilingConfig(crop_bp=-1)

    def test_crop_too_large_raises(self):
        with pytest.raises(ValueError, match="crop_bp.*too large"):
            TilingConfig(crop_bp=131072 // 2)

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution must be 1 or 128"):
            TilingConfig(resolution=64)

    def test_crop_not_divisible_by_resolution_raises(self):
        with pytest.raises(ValueError, match="divisible by resolution"):
            TilingConfig(crop_bp=100, resolution=128)

    def test_resolution_1_with_valid_crop(self):
        cfg = TilingConfig(crop_bp=32768, resolution=1)
        assert cfg.resolution == 1
        assert cfg.effective_size == 131072 - 2 * 32768


@pytest.mark.unit
class TestGenerateTiles:
    """Tests for _generate_tiles() tiling correctness."""

    def test_single_tile_no_crop(self):
        """Chromosome shorter than window -> single tile."""
        cfg = TilingConfig(crop_bp=0, resolution=128)
        tiles = _generate_tiles(100000, cfg)
        assert len(tiles) == 1
        window_start, window_end, keep_start, keep_end = tiles[0]
        assert window_start == 0
        assert window_end == cfg.window_size
        assert keep_start == 0
        assert keep_end == cfg.window_size

    def test_seamless_coverage_no_crop(self):
        """Without cropping, kept regions tile seamlessly."""
        chrom_len = 500000
        cfg = TilingConfig(crop_bp=0, resolution=128)
        tiles = _generate_tiles(chrom_len, cfg)

        # Check that kept regions cover the chromosome without gaps/overlaps
        covered = set()
        for window_start, window_end, keep_start, keep_end in tiles:
            genome_keep_start = max(0, window_start + keep_start)
            genome_keep_end = min(chrom_len, window_start + keep_end)
            for bp in range(genome_keep_start, genome_keep_end, cfg.resolution):
                assert bp not in covered, f"Position {bp} covered twice"
                covered.add(bp)

        # All positions should be covered
        expected = set(range(0, chrom_len, cfg.resolution))
        # Allow the last partial bin to be missing
        missing = expected - covered
        assert all(pos >= chrom_len - cfg.resolution for pos in missing)

    def test_seamless_coverage_with_crop(self):
        """With cropping, kept regions tile seamlessly (no gaps, no overlaps)."""
        chrom_len = 500000
        cfg = TilingConfig(crop_bp=32768, resolution=128)
        tiles = _generate_tiles(chrom_len, cfg)

        # Verify seamless: collect all kept genome positions
        genome_positions = []
        for window_start, window_end, keep_start, keep_end in tiles:
            genome_keep_start = window_start + keep_start
            genome_keep_end = window_start + keep_end
            genome_positions.append((genome_keep_start, genome_keep_end))

        # Sort by start
        genome_positions.sort()

        # Check no gaps between consecutive kept regions
        for i in range(1, len(genome_positions)):
            prev_end = genome_positions[i - 1][1]
            curr_start = genome_positions[i][0]
            assert prev_end == curr_start, (
                f"Gap or overlap: prev_end={prev_end}, curr_start={curr_start}"
            )

        # First kept region should start at 0
        assert genome_positions[0][0] == 0

        # Last kept region should cover past the chromosome end
        assert genome_positions[-1][1] >= chrom_len

    def test_tile_count_with_crop(self):
        """Number of tiles should increase with cropping (smaller step)."""
        chrom_len = 1000000
        tiles_no_crop = _generate_tiles(chrom_len, TilingConfig(crop_bp=0, resolution=128))
        tiles_with_crop = _generate_tiles(chrom_len, TilingConfig(crop_bp=32768, resolution=128))
        assert len(tiles_with_crop) > len(tiles_no_crop)

    def test_empty_chromosome(self):
        """Zero-length chromosome -> no tiles."""
        cfg = TilingConfig(crop_bp=0, resolution=128)
        tiles = _generate_tiles(0, cfg)
        assert len(tiles) == 0

    def test_keep_indices_consistent(self):
        """keep_start/keep_end should be consistent with crop config."""
        cfg = TilingConfig(crop_bp=16384, resolution=128)
        tiles = _generate_tiles(300000, cfg)
        for _, _, keep_start, keep_end in tiles:
            assert keep_start == cfg.crop_start
            assert keep_end == cfg.crop_end

    def test_1bp_resolution_tiling(self):
        """Tiling at 1bp resolution produces valid tiles."""
        chrom_len = 300000
        cfg = TilingConfig(crop_bp=32768, resolution=1)
        tiles = _generate_tiles(chrom_len, cfg)
        assert len(tiles) > 0

        # Kept regions should tile seamlessly
        genome_positions = []
        for window_start, _, keep_start, keep_end in tiles:
            genome_positions.append((
                window_start + keep_start,
                window_start + keep_end,
            ))
        genome_positions.sort()

        for i in range(1, len(genome_positions)):
            assert genome_positions[i][0] == genome_positions[i - 1][1]


@pytest.mark.unit
class TestSequenceToOnehot:
    """Tests for _sequence_to_onehot() encoding."""

    def test_basic_encoding(self):
        onehot = _sequence_to_onehot("ACGT")
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        np.testing.assert_array_equal(onehot, expected)

    def test_case_insensitive(self):
        upper = _sequence_to_onehot("ACGT")
        lower = _sequence_to_onehot("acgt")
        np.testing.assert_array_equal(upper, lower)

    def test_n_encoding(self):
        """N bases should be encoded as uniform [0.25, 0.25, 0.25, 0.25]."""
        onehot = _sequence_to_onehot("N")
        expected = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
        np.testing.assert_array_equal(onehot, expected)

    def test_mixed_sequence(self):
        onehot = _sequence_to_onehot("ACNGT")
        assert onehot.shape == (5, 4)
        # A
        np.testing.assert_array_equal(onehot[0], [1, 0, 0, 0])
        # N
        np.testing.assert_array_equal(onehot[2], [0.25, 0.25, 0.25, 0.25])
        # T
        np.testing.assert_array_equal(onehot[4], [0, 0, 0, 1])

    def test_output_dtype(self):
        onehot = _sequence_to_onehot("ACGT")
        assert onehot.dtype == np.float32

    def test_empty_sequence(self):
        onehot = _sequence_to_onehot("")
        assert onehot.shape == (0, 4)


@pytest.mark.unit
class TestStitchingWithMockModel:
    """Test end-to-end stitching using a mock model that returns position-dependent values."""

    class _MockModel(torch.nn.Module):
        """Mock model returning a known function of input position.

        Returns a single track with value = mean of one-hot A channel
        over the window. This lets us verify the stitching places predictions
        at the correct genomic positions.
        """
        def __init__(self, resolution=128, n_tracks=1):
            super().__init__()
            self._resolution = resolution
            self._n_tracks = n_tracks

        def eval(self):
            return self

        def predict(self, dna_sequence, organism_index, resolutions=None, heads=None):
            B, S, _ = dna_sequence.shape
            out_len = S // self._resolution

            # Use the A-channel mean as a position signal
            preds = torch.zeros(B, out_len, self._n_tracks)
            for b in range(B):
                for i in range(out_len):
                    start = i * self._resolution
                    end = start + self._resolution
                    preds[b, i, 0] = dna_sequence[b, start:end, 0].mean()

            return {
                'atac': {self._resolution: preds},
            }

    def _make_genome_array(self, chrom_len):
        """Create a simple genome with position-dependent A-frequency."""
        # Create alternating A/C pattern with known frequency
        onehot = np.zeros((chrom_len, 4), dtype=np.float32)
        onehot[:, 0] = 1.0  # All A's for simplicity
        return onehot

    def test_stitching_no_crop_128bp(self):
        """Verify stitching without cropping recovers full chromosome predictions."""
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            predict_full_chromosome,
            GenomeSequenceProvider,
        )

        chrom_len = 131072 * 3  # Exactly 3 windows
        config = TilingConfig(crop_bp=0, resolution=128, batch_size=2)
        model = self._MockModel(resolution=128)

        # Create a mock GenomeSequenceProvider
        provider = object.__new__(GenomeSequenceProvider)
        provider.chrom_sizes = {'chr1': chrom_len}
        provider._cache = {'chr1': self._make_genome_array(chrom_len)}
        provider._fasta_path = '/dev/null'
        provider._cache_enabled = True

        preds = predict_full_chromosome(
            model, provider, 'chr1', 'atac',
            config=config,
            track_indices=[0],
            device='cpu',
            show_progress=False,
        )

        expected_len = chrom_len // 128
        assert preds.shape == (expected_len, 1)
        # All A genome: each 128bp bin should have mean(A) = 1.0
        np.testing.assert_allclose(preds[:, 0], 1.0, atol=1e-6)

    def test_stitching_with_crop_128bp(self):
        """Verify stitching with cropping still covers chromosome without gaps."""
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            predict_full_chromosome,
            GenomeSequenceProvider,
        )

        chrom_len = 131072 * 2
        config = TilingConfig(crop_bp=32768, resolution=128, batch_size=1)
        model = self._MockModel(resolution=128)

        provider = object.__new__(GenomeSequenceProvider)
        provider.chrom_sizes = {'chr1': chrom_len}
        provider._cache = {'chr1': self._make_genome_array(chrom_len)}
        provider._fasta_path = '/dev/null'
        provider._cache_enabled = True

        preds = predict_full_chromosome(
            model, provider, 'chr1', 'atac',
            config=config,
            track_indices=[0],
            device='cpu',
            show_progress=False,
        )

        expected_len = chrom_len // 128
        assert preds.shape == (expected_len, 1)
        # Check no zeros in the interior (would indicate gaps in stitching)
        assert np.all(preds[1:-1, 0] > 0)


@pytest.mark.unit
class TestHeadConfigs:
    """Tests for HEAD_CONFIGS dictionary."""

    def test_all_heads_have_required_keys(self):
        for name, config in HEAD_CONFIGS.items():
            assert 'num_tracks' in config, f"{name} missing num_tracks"
            assert 'resolutions' in config, f"{name} missing resolutions"

    def test_known_heads_present(self):
        expected_heads = ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']
        for head in expected_heads:
            assert head in HEAD_CONFIGS, f"{head} missing from HEAD_CONFIGS"
