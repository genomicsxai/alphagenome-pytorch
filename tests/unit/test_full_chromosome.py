"""Unit tests for full-chromosome tiling and prediction infrastructure.

Tests TilingConfig validation, _generate_tiles() correctness,
_sequence_to_onehot() encoding, and stitching logic with a mock model.
All pure-logic tests -- no pyBigWig, pyfaidx, GPU, or model weights needed.
"""

from pathlib import Path

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
        """N bases should be encoded as all-zeros (matching the JAX reference)."""
        onehot = _sequence_to_onehot("N")
        expected = np.array([[0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(onehot, expected)

    def test_mixed_sequence(self):
        onehot = _sequence_to_onehot("ACNGT")
        assert onehot.shape == (5, 4)
        # A
        np.testing.assert_array_equal(onehot[0], [1, 0, 0, 0])
        # N
        np.testing.assert_array_equal(onehot[2], [0, 0, 0, 0])
        # T
        np.testing.assert_array_equal(onehot[4], [0, 0, 0, 1])

    def test_output_dtype(self):
        onehot = _sequence_to_onehot("ACGT")
        assert onehot.dtype == np.uint8

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

    def _build_in_memory_provider(self, GenomeSequenceProvider, chrom_len):
        """Construct a ``GenomeSequenceProvider`` backed by an in-memory genome.

        Bypasses ``__init__`` (which requires a real FASTA) by injecting a
        fake source object that mirrors the
        :class:`~alphagenome_pytorch.genome.GenomeSequenceSource` API the
        provider delegates to: ``fetch_onehot(chrom, start, end, ...)`` plus
        a ``chrom_sizes`` dict.
        """
        genome_array = self._make_genome_array(chrom_len)

        class _FakeSource:
            chrom_sizes = {'chr1': chrom_len}

            def fetch_onehot(self, chrom, start, end, *, pad=True):
                length = end - start
                if pad:
                    result = np.zeros((length, 4), dtype=genome_array.dtype)
                    valid_start = max(0, start)
                    valid_end = min(self.chrom_sizes.get(chrom, 0), end)
                    if valid_start < valid_end:
                        dest = valid_start - start
                        result[dest:dest + (valid_end - valid_start)] = (
                            genome_array[valid_start:valid_end]
                        )
                    return result
                return genome_array[start:end].copy()

        provider = object.__new__(GenomeSequenceProvider)
        provider._source = _FakeSource()
        provider.chrom_sizes = provider._source.chrom_sizes
        return provider

    def test_stitching_no_crop_128bp(self):
        """Verify stitching without cropping recovers full chromosome predictions."""
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            predict_full_chromosome,
            GenomeSequenceProvider,
        )

        chrom_len = 131072 * 3  # Exactly 3 windows
        config = TilingConfig(crop_bp=0, resolution=128, batch_size=2)
        model = self._MockModel(resolution=128)

        provider = self._build_in_memory_provider(GenomeSequenceProvider, chrom_len)

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

        provider = self._build_in_memory_provider(GenomeSequenceProvider, chrom_len)

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

    def test_gene_counts_anndata_across_tiles(self):
        """Whole-chromosome gene counts reconstruct a gene that spans two tiles."""
        import pandas as pd
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            predict_full_chromosomes_to_anndata,
            GenomeSequenceProvider,
        )
        from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation

        chrom_len = 131072 * 2  # two 131072bp tiles at crop_bp=0
        config = TilingConfig(crop_bp=0, resolution=128, batch_size=1)
        model = self._MockModel(resolution=128, n_tracks=1)  # all-A genome -> every bin == 1.0
        provider = self._build_in_memory_provider(GenomeSequenceProvider, chrom_len)

        # geneX: exon [256,640) -> bins [2,5) in tile 0 (3 bins).
        # geneY: exon [130944,131328) straddles the tile-0/tile-1 boundary at 131072
        #        -> 1 bin in tile 0 + 2 bins in tile 1 (3 bins total).
        rows = [
            dict(Feature="gene", Chromosome="chr1", Start=256, End=640, Strand="+",
                 gene_id="ENSX", gene_name="X", gene_type="protein_coding"),
            dict(Feature="exon", Chromosome="chr1", Start=256, End=640, Strand="+",
                 gene_id="ENSX", gene_name="X", gene_type="protein_coding"),
            dict(Feature="gene", Chromosome="chr1", Start=130944, End=131328, Strand="+",
                 gene_id="ENSY", gene_name="Y", gene_type="protein_coding"),
            dict(Feature="exon", Chromosome="chr1", Start=130944, End=131328, Strand="+",
                 gene_id="ENSY", gene_name="Y", gene_type="protein_coding"),
        ]
        ann = GeneAnnotation(pd.DataFrame(rows))

        gc = predict_full_chromosomes_to_anndata(
            model, provider, ann, "atac",
            chromosomes=["chr1"], config=config, track_indices=[0],
            over="exons", reduce="sum", device="cpu", show_progress=False,
        )

        counts = {gid: gc.counts[0, i, 0].item()
                  for i, gid in enumerate(gc.gene_metadata["gene_id"])}
        assert counts["ENSX"] == pytest.approx(3.0)  # 3 exon bins, all-A -> 1.0 each
        assert counts["ENSY"] == pytest.approx(3.0)  # reconstructed across the tile seam

    def test_gene_counts_anndata_requires_exon_rows(self):
        """over='exons' with a gene-only annotation errors up front, not after inference."""
        import pandas as pd
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            predict_full_chromosomes_to_anndata,
            GenomeSequenceProvider,
        )
        from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation

        chrom_len = 131072
        config = TilingConfig(crop_bp=0, resolution=128, batch_size=1)
        model = self._MockModel(resolution=128, n_tracks=1)
        provider = self._build_in_memory_provider(GenomeSequenceProvider, chrom_len)
        # Gene-only annotation: no exon rows.
        ann = GeneAnnotation(pd.DataFrame([
            dict(Feature="gene", Chromosome="chr1", Start=256, End=640, Strand="+",
                 gene_id="ENSX", gene_name="X", gene_type="protein_coding"),
        ]))

        kwargs = dict(chromosomes=["chr1"], config=config, track_indices=[0],
                      device="cpu", show_progress=False)
        with pytest.raises(ValueError, match="exon rows"):
            predict_full_chromosomes_to_anndata(model, provider, ann, "atac",
                                                over="exons", **kwargs)
        # gene_body works with the same gene-only annotation.
        gc = predict_full_chromosomes_to_anndata(model, provider, ann, "atac",
                                                 over="gene_body", **kwargs)
        assert list(gc.gene_metadata["gene_id"]) == ["ENSX"]

    def test_gene_counts_anndata_write_path(self, tmp_path):
        """End-to-end: write .h5ad and read it back; check orientation + metadata."""
        import pandas as pd
        anndata = pytest.importorskip("anndata")
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            predict_full_chromosomes_to_anndata,
            GenomeSequenceProvider,
        )
        from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation

        chrom_len = 131072
        config = TilingConfig(crop_bp=0, resolution=128, batch_size=1)
        model = self._MockModel(resolution=128, n_tracks=2)  # track 0 == 1.0, track 1 == 0.0
        provider = self._build_in_memory_provider(GenomeSequenceProvider, chrom_len)
        rows = [
            dict(Feature="gene", Chromosome="chr1", Start=256, End=640, Strand="+",
                 gene_id="ENSX", gene_name="X", gene_type="protein_coding"),
            dict(Feature="exon", Chromosome="chr1", Start=256, End=640, Strand="+",
                 gene_id="ENSX", gene_name="X", gene_type="protein_coding"),  # bins [2,5)
        ]
        ann = GeneAnnotation(pd.DataFrame(rows))

        out = tmp_path / "gene_counts.h5ad"
        predict_full_chromosomes_to_anndata(
            model, provider, ann, "atac",
            output_path=str(out), chromosomes=["chr1"], config=config,
            track_indices=[0, 1], track_names=["t0", "t1"],
            over="exons", reduce="sum", device="cpu", show_progress=False,
        )

        assert out.exists()
        adata = anndata.read_h5ad(str(out))
        assert adata.shape == (2, 1)                       # obs=tracks, var=genes
        assert list(adata.var_names) == ["ENSX"]
        assert list(adata.obs["track_name"]) == ["t0", "t1"]
        assert float(adata.X[0, 0]) == pytest.approx(3.0)  # track 0: 3 exon bins x 1.0
        assert float(adata.X[1, 0]) == pytest.approx(0.0)  # track 1: mock is 0

    def test_gene_counts_anndata_progress_never_reaches_stdout(self, capsys):
        """`agt predict --json` reads stdout as JSON, so prose must not land there.

        show_progress=False must be silent on both streams; show_progress=True
        must report on stderr only.
        """
        import pandas as pd
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            predict_full_chromosomes_to_anndata,
            GenomeSequenceProvider,
        )
        from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation

        chrom_len = 131072
        config = TilingConfig(crop_bp=0, resolution=128, batch_size=1)
        model = self._MockModel(resolution=128, n_tracks=1)
        ann = GeneAnnotation(pd.DataFrame([
            dict(Feature="gene", Chromosome="chr1", Start=256, End=640, Strand="+",
                 gene_id="ENSX", gene_name="X", gene_type="protein_coding"),
            dict(Feature="exon", Chromosome="chr1", Start=256, End=640, Strand="+",
                 gene_id="ENSX", gene_name="X", gene_type="protein_coding"),
        ]))

        def aggregate(show_progress):
            provider = self._build_in_memory_provider(GenomeSequenceProvider, chrom_len)
            predict_full_chromosomes_to_anndata(
                model, provider, ann, "atac",
                chromosomes=["chr1"], config=config, track_indices=[0],
                over="exons", reduce="sum", device="cpu",
                show_progress=show_progress,
            )
            return capsys.readouterr()

        quiet = aggregate(False)
        assert quiet.out == ""
        assert quiet.err == ""

        loud = aggregate(True)
        assert loud.out == ""
        assert "Aggregating" in loud.err

    def test_gene_counts_anndata_log_flag_survives_the_progress_helper(self):
        """The progress helper must not shadow this function's `log` (log1p) flag."""
        import pandas as pd
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            predict_full_chromosomes_to_anndata,
            GenomeSequenceProvider,
        )
        from alphagenome_pytorch.variant_scoring.annotations import GeneAnnotation

        chrom_len = 131072
        config = TilingConfig(crop_bp=0, resolution=128, batch_size=1)
        model = self._MockModel(resolution=128, n_tracks=1)
        ann = GeneAnnotation(pd.DataFrame([
            dict(Feature="gene", Chromosome="chr1", Start=256, End=640, Strand="+",
                 gene_id="ENSX", gene_name="X", gene_type="protein_coding"),
            dict(Feature="exon", Chromosome="chr1", Start=256, End=640, Strand="+",
                 gene_id="ENSX", gene_name="X", gene_type="protein_coding"),
        ]))

        def aggregate(log):
            provider = self._build_in_memory_provider(GenomeSequenceProvider, chrom_len)
            gc = predict_full_chromosomes_to_anndata(
                model, provider, ann, "atac",
                chromosomes=["chr1"], config=config, track_indices=[0],
                over="exons", reduce="sum", device="cpu", show_progress=False,
                log=log,
            )
            return gc.counts[0, 0, 0].item()

        raw = aggregate(False)
        assert raw == pytest.approx(3.0)             # 3 exon bins x 1.0
        assert aggregate(True) == pytest.approx(np.log1p(raw))


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


class TestAnnDataTrackMetadata:
    """obs should carry full catalog metadata, and padding should be dropped.

    Before this, `predict_full_chromosomes_to_anndata` built obs from three
    parallel lists and emitted `track_index` only, including the 101 padding
    channels of the rna_seq head.
    """

    @staticmethod
    def _catalog_tracks():
        from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
        catalog = TrackMetadataCatalog.load_builtin(0)
        return list(catalog.get_tracks("rna_seq", organism=0, strict=True))

    def test_is_padding_accepts_objects_and_row_dicts(self):
        from alphagenome_pytorch.extensions.inference.full_chromosome import _is_padding

        tracks = self._catalog_tracks()
        pad = next(t for t in tracks if t.track_name.lower() == "padding")
        real = next(t for t in tracks if t.track_name.lower() != "padding")
        assert _is_padding(pad) and not _is_padding(real)
        # checkpoints embed flat dicts rather than TrackMetadata
        assert _is_padding({"track_name": "padding"})
        assert _is_padding({"track_name": "PADDING"})
        assert not _is_padding({"track_name": "CL:0000047 polyA plus RNA-seq"})
        assert not _is_padding({})

    def test_metadata_populates_obs(self):
        from alphagenome_pytorch.extensions.inference.full_chromosome import _build_track_frame

        tracks = self._catalog_tracks()[:4]
        frame = _build_track_frame(list(range(4)), metadata=tracks)
        assert len(frame) == 4
        for col in ("track_index", "track_name", "biosample_name", "assay_title", "strand"):
            assert col in frame.columns, col
        assert list(frame["track_index"]) == [0, 1, 2, 3]

    def test_without_metadata_falls_back_to_track_index(self):
        from alphagenome_pytorch.extensions.inference.full_chromosome import _build_track_frame

        frame = _build_track_frame([0, 1, 2])
        assert list(frame.columns) == ["track_index"]

    def test_track_index_follows_the_subset(self):
        """obs.track_index must record the head channel, not the row position."""
        from alphagenome_pytorch.extensions.inference.full_chromosome import _build_track_frame

        tracks = self._catalog_tracks()
        picked = [5, 9, 30]
        frame = _build_track_frame(picked, metadata=[tracks[i] for i in picked])
        assert list(frame["track_index"]) == picked

    def test_metadata_length_mismatch_rejected(self):
        from alphagenome_pytorch.extensions.inference.full_chromosome import _build_track_frame

        tracks = self._catalog_tracks()[:3]
        with pytest.raises(ValueError, match="metadata has 3 entries"):
            _build_track_frame([0, 1], metadata=tracks)

    def test_rna_seq_head_has_padding_to_strip(self):
        """Guards the premise: the human rna_seq head really is padded 667 -> 768."""
        from alphagenome_pytorch.extensions.inference.full_chromosome import _is_padding

        tracks = self._catalog_tracks()
        assert len(tracks) == 768
        assert sum(1 for t in tracks if _is_padding(t)) == 101


@pytest.mark.unit
class TestWriteChromosomesBigwig:
    """One BigWig per track, streamed across chromosomes.

    A BigWig holds one signal for a whole genome, so chromosomes are appended
    to a single open handle per track instead of each getting its own file.
    """

    CHROM_SIZES = {"chr1": 1280, "chr2": 640}  # 10 and 5 bins at 128bp

    @staticmethod
    def _write(*args, **kwargs):
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            write_chromosomes_bigwig,
        )
        return write_chromosomes_bigwig(*args, **kwargs)

    def test_single_track_writes_one_file_for_all_chromosomes(self, tmp_path):
        pyBigWig = pytest.importorskip("pyBigWig")
        out = tmp_path / "dnase.bw"

        written = self._write(
            [(np.arange(1, 11, dtype=float).reshape(10, 1), "chr1"),
             (np.arange(1, 6, dtype=float).reshape(5, 1), "chr2")],
            output_path=out,
            chrom_sizes=self.CHROM_SIZES,
            resolution=128,
            chromosome_order=["chr1", "chr2"],
        )

        # One track means no track suffix, and one file rather than one per chromosome.
        assert written == [out]
        assert [p.name for p in tmp_path.iterdir()] == ["dnase.bw"]

        bw = pyBigWig.open(str(out))
        try:
            assert bw.chroms() == self.CHROM_SIZES
            assert len(bw.intervals("chr1")) == 10
            assert len(bw.intervals("chr2")) == 5
        finally:
            bw.close()

    def test_values_cover_resolution_sized_spans(self, tmp_path):
        """Each value spans one bin -- ranges, not one entry per base."""
        pyBigWig = pytest.importorskip("pyBigWig")
        out = tmp_path / "atac.bw"

        # Distinct values, so adjacent bins cannot merge into a single interval.
        self._write(
            [(np.arange(1, 11, dtype=float).reshape(10, 1), "chr1")],
            output_path=out,
            chrom_sizes=self.CHROM_SIZES,
            resolution=128,
            chromosome_order=["chr1"],
        )

        bw = pyBigWig.open(str(out))
        try:
            intervals = bw.intervals("chr1")
        finally:
            bw.close()

        assert list(intervals) == [
            (i * 128, (i + 1) * 128, float(i + 1)) for i in range(10)
        ]

    def test_multiple_tracks_write_one_file_each(self, tmp_path):
        pyBigWig = pytest.importorskip("pyBigWig")
        out = tmp_path / "dnase.bw"

        written = self._write(
            [(np.ones((10, 3)), "chr1"), (np.ones((5, 3)), "chr2")],
            output_path=out,
            chrom_sizes=self.CHROM_SIZES,
            resolution=128,
            track_names=["k562", "gm12878", "hepg2"],
            chromosome_order=["chr1", "chr2"],
        )

        assert [p.name for p in written] == [
            "dnase_k562.bw", "dnase_gm12878.bw", "dnase_hepg2.bw",
        ]
        # Each track file spans both chromosomes, so the output is one file per
        # track -- not one per (track, chromosome) as the per-chromosome writer gives.
        for path in written:
            bw = pyBigWig.open(str(path))
            try:
                assert bw.intervals("chr1") and bw.intervals("chr2")
            finally:
                bw.close()

    def test_header_declares_only_the_chromosomes_written(self, tmp_path):
        pyBigWig = pytest.importorskip("pyBigWig")
        out = tmp_path / "atac.bw"

        self._write(
            [(np.ones((10, 1)), "chr1")],
            output_path=out,
            chrom_sizes=self.CHROM_SIZES,
            resolution=128,
            chromosome_order=["chr1"],
        )

        bw = pyBigWig.open(str(out))
        try:
            assert bw.chroms() == {"chr1": 1280}
        finally:
            bw.close()

    def test_trailing_partial_bin_is_dropped(self, tmp_path):
        """chr2 holds 5 whole bins, so a 6th value has nowhere valid to go."""
        pyBigWig = pytest.importorskip("pyBigWig")
        out = tmp_path / "atac.bw"

        self._write(
            [(np.arange(1, 7, dtype=float).reshape(6, 1), "chr2")],
            output_path=out,
            chrom_sizes=self.CHROM_SIZES,
            resolution=128,
            chromosome_order=["chr2"],
        )

        bw = pyBigWig.open(str(out))
        try:
            intervals = bw.intervals("chr2")
        finally:
            bw.close()

        assert len(intervals) == 5
        assert intervals[-1] == (512, 640, 5.0)

    def test_out_of_order_chromosome_rejected(self, tmp_path):
        pytest.importorskip("pyBigWig")

        with pytest.raises(ValueError, match="header order"):
            self._write(
                [(np.ones((5, 1)), "chr2"), (np.ones((10, 1)), "chr1")],
                output_path=tmp_path / "atac.bw",
                chrom_sizes=self.CHROM_SIZES,
                resolution=128,
                chromosome_order=["chr1", "chr2"],
            )

    def test_chromosomes_are_flushed_before_the_next_arrives(self, tmp_path):
        """Laziness: an eager writer would drain the input before opening a file."""
        pytest.importorskip("pyBigWig")
        out = tmp_path / "atac.bw"

        def chrom_arrays():
            yield np.ones((10, 1)), "chr1"
            raise RuntimeError("second chromosome failed")

        with pytest.raises(RuntimeError, match="second chromosome failed"):
            self._write(
                chrom_arrays(),
                output_path=out,
                chrom_sizes=self.CHROM_SIZES,
                resolution=128,
                chromosome_order=["chr1", "chr2"],
            )

        # chr1 reached an open handle, which the finally block then closed.
        assert out.exists()


@pytest.mark.unit
class TestFullChromosomeBigwigApi:
    """The genome-wide default, and the flag that restores per-chromosome files."""

    def test_split_by_chrom_defaults_to_one_file_per_track(self):
        import inspect
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            predict_full_chromosomes_to_bigwig,
        )

        parameter = inspect.signature(predict_full_chromosomes_to_bigwig).parameters
        assert parameter["split_by_chrom"].default is False

    def test_bigwig_output_records_the_chromosomes_covered(self):
        from alphagenome_pytorch.extensions.inference.full_chromosome import BigwigOutput

        out = BigwigOutput(Path("dnase.bw"), ["chr20", "chr21"])
        assert out.path.name == "dnase.bw"
        assert out.chromosomes == ["chr20", "chr21"]


@pytest.mark.unit
class TestProgressStreamDiscipline:
    """Progress prose must never reach stdout, which carries the JSON payload."""

    @staticmethod
    def _tiny_fasta(tmp_path):
        fa = tmp_path / "tiny.fa"
        fa.write_text(">chr1\n" + "ACGT" * 16 + "\n>chr2\n" + "ACGT" * 8 + "\n")
        return fa

    def test_provider_is_silent_when_progress_is_off(self, tmp_path, capsys):
        pytest.importorskip("pyfaidx")
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            GenomeSequenceProvider,
        )

        provider = GenomeSequenceProvider(
            self._tiny_fasta(tmp_path), chromosomes={"chr1"}, show_progress=False,
        )
        try:
            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err == ""
        finally:
            provider.close()

    def test_provider_reports_on_stderr_not_stdout(self, tmp_path, capsys):
        """Both messages -- the provider's own and GenomeSequenceSource's."""
        pytest.importorskip("pyfaidx")
        from alphagenome_pytorch.extensions.inference.full_chromosome import (
            GenomeSequenceProvider,
        )

        provider = GenomeSequenceProvider(
            self._tiny_fasta(tmp_path), chromosomes={"chr1"}, show_progress=True,
        )
        try:
            captured = capsys.readouterr()
            assert captured.out == ""
            assert "Loading genome from" in captured.err
            assert "Cached genome" in captured.err
        finally:
            provider.close()

