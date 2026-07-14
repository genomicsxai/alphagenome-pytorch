"""Tests for scripts/compute_gene_metrics.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_script_module():
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "compute_gene_metrics_script",
        scripts_dir / "compute_gene_metrics.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


def test_gene_expression_rows_use_exon_means_and_matching_strand(script_module):
    columns = [
        "Chromosome", "Start", "End", "Strand", "Feature", "gene_id",
        "gene_name", "gene_type", "transcript_id",
    ]
    gtf = pd.DataFrame([
        ["chr1", 0, 4, "+", "gene", "plus", "PLUS", "pc", None],
        ["chr1", 0, 4, "+", "transcript", "plus", "PLUS", "pc", "tx_plus"],
        ["chr1", 0, 4, "+", "exon", "plus", "PLUS", "pc", "tx_plus"],
        ["chr1", 4, 8, "-", "gene", "minus", "MINUS", "pc", None],
        ["chr1", 4, 7, "-", "transcript", "minus", "MINUS", "pc", "tx_minus"],
        ["chr1", 4, 8, "-", "exon", "minus", "MINUS", "pc", "tx_minus"],
    ], columns=columns)
    intervals = pd.DataFrame([{
        "original_interval_idx": 7,
        "chrom": "chr1",
        "start": 0,
        "end": 8,
        "window_start": 0,
        "window_end": 8,
    }])
    one_bp = np.asarray([[
        [1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0],
        [5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0],
    ]])
    two_bp = one_bp.reshape(1, 4, 2, 2).mean(axis=2)

    rows = script_module.collect_gene_expression_rows(
        gtf=gtf,
        test_intervals=intervals,
        positions=[("chr1", 0, 8)],
        prediction_views={1: one_bp, 2: two_bp},
        target_views={1: one_bp, 2: two_bp},
        bin_sizes=(1, 2),
        sample_names=["sample"],
    )

    assert len(rows) == 4
    plus_1bp = rows[(rows["gene_id"] == "plus") & (rows["bin_size_bp"] == 1)].iloc[0]
    minus_1bp = rows[(rows["gene_id"] == "minus") & (rows["bin_size_bp"] == 1)].iloc[0]
    assert plus_1bp["track_index"] == 0
    assert plus_1bp["pred_mean"] == pytest.approx(2.5)
    assert minus_1bp["track_index"] == 1
    assert minus_1bp["pred_mean"] == pytest.approx(65.0)
    assert plus_1bp["pred_log1p_mean"] == pytest.approx(np.log1p(2.5))

    summary, per_track = script_module.summarize_gene_expression(rows, (1, 2))
    assert summary[1]["pearson_r_log1p_exon_mean"] == pytest.approx(1.0)
    assert summary[1]["n_unique_genes"] == 2
    assert len(per_track) == 4


def test_gene_assignment_uses_nearest_interval_center_once(script_module):
    columns = [
        "Chromosome", "Start", "End", "Strand", "Feature", "gene_id",
        "gene_name", "gene_type", "transcript_id",
    ]
    gtf = pd.DataFrame([
        ["chr1", 6, 8, "+", "transcript", "shared", "SHARED", "pc", "tx1"],
        ["chr1", 10, 11, "+", "transcript", "second", "SECOND", "pc", "tx2"],
    ], columns=columns)
    positions = [("chr1", 0, 10), ("chr1", 2, 12)]

    assignments = script_module.assign_genes_to_nearest_center(gtf, positions)
    # The first gene is eligible in both windows. Its center distance ties, so
    # BED order deterministically selects the first window.
    assert assignments["shared"] == 0
    # Half-open containment excludes TSS=10 from [0, 10).
    assert assignments["second"] == 1


def test_center_crop_genomic_positions_matches_profile_crop(script_module):
    positions = [("chr1", 100, 1100), ("chr2", 200, 1200)]

    assert script_module.center_crop_genomic_positions(
        positions, 200,
    ) == [("chr1", 500, 700), ("chr2", 600, 800)]
    assert script_module.center_crop_genomic_positions(positions, None) is positions

    # Match center_crop_profiles when an odd number of source bins is removed.
    assert script_module.center_crop_genomic_positions(
        [("chr1", 0, 10)], 4, source_resolution=2,
    ) == [("chr1", 2, 6)]

    with pytest.raises(ValueError, match="spans only"):
        script_module.center_crop_genomic_positions(positions, 1001)


def test_official_gene_selection_uses_transcript_tss(script_module):
    columns = [
        "Chromosome", "Start", "End", "Strand", "Feature", "gene_id",
        "gene_name", "gene_type", "transcript_id",
    ]
    gtf = pd.DataFrame([
        ["chr1", 0, 20, "+", "gene", "inside", "INSIDE", "pc", None],
        ["chr1", 2, 12, "+", "transcript", "inside", "INSIDE", "pc", "tx1"],
        ["chr1", 2, 5, "+", "exon", "inside", "INSIDE", "pc", "tx1"],
        ["chr1", 4, 7, "+", "exon", "inside", "INSIDE", "pc", "tx1"],
        ["chr1", 0, 20, "+", "gene", "outside", "OUTSIDE", "pc", None],
        ["chr1", 12, 18, "+", "transcript", "outside", "OUTSIDE", "pc", "tx2"],
        ["chr1", 6, 8, "+", "exon", "outside", "OUTSIDE", "pc", "tx2"],
    ], columns=columns)

    extractor = script_module._OfficialGeneExonMaskExtractor(gtf)
    mask, metadata = extractor.extract("chr1", 0, 8)
    assert metadata["gene_id"].tolist() == ["inside"]

    # Overlapping exon rows are combined as a boolean union: positions 2..6
    # are counted once, matching the official GeneIntervalScorer mask.
    profile = np.arange(8, dtype=np.float64)[:, None]
    mean = script_module.score_gene_interval_mean(profile, mask)
    assert mean[0, 0] == pytest.approx(np.mean(np.arange(2, 7)))

    # AlphaGenome max-pools a gene mask when the output resolution is coarse.
    pooled_mask = script_module.downsample_gene_mask(mask, 2)
    np.testing.assert_array_equal(
        pooled_mask[:, 0], np.asarray([False, True, True, True]),
    )
