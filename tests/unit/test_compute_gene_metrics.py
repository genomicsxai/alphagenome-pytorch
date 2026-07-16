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

    summary, per_track, per_gene = script_module.summarize_gene_expression(
        rows, (1, 2),
    )
    assert summary[1]["raw_pearson_r_mean_across_tracks"] == pytest.approx(1.0)
    assert summary[1]["n_genes"] == 2
    assert len(per_track) == 2
    assert len(per_gene) == 4


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


def test_exon_base_weights_preserve_partial_coarse_bins(script_module):
    mask = np.zeros((8, 1), dtype=bool)
    mask[2:7, 0] = True
    profile = np.arange(8, dtype=np.float64)[:, None]
    mean = script_module.score_gene_interval_mean(profile, mask)
    assert mean[0, 0] == pytest.approx(np.mean(np.arange(2, 7)))

    weighted_mask = script_module.downsample_gene_mask_weights(mask, 2)
    np.testing.assert_array_equal(
        weighted_mask[:, 0], np.asarray([0, 2, 2, 1]),
    )


def test_paper_gene_selection_uses_unique_exon_bases(script_module):
    columns = [
        "Chromosome", "Start", "End", "Strand", "Feature", "gene_id",
        "gene_name", "gene_type", "transcript_id",
    ]
    gtf = pd.DataFrame([
        # The two overlapping exon annotations have a 15-bp union. The [5, 10)
        # interval contains only 5/15 unique exon bases and must not qualify;
        # counting the overlapping annotations twice would incorrectly give 50%.
        ["chr1", 0, 10, "+", "exon", "overlap", "OVERLAP", "pc", "tx1"],
        ["chr1", 5, 15, "+", "exon", "overlap", "OVERLAP", "pc", "tx2"],
        ["chr1", 20, 30, "+", "exon", "assigned", "ASSIGNED", "pc", "tx3"],
    ], columns=columns)
    annotations = script_module.build_paper_gene_annotations(gtf)
    assert annotations.loc["overlap", "total_unique_exon_bp"] == 15

    positions = [("chr1", 5, 10), ("chr1", 20, 26), ("chr1", 22, 30)]
    assignments = script_module.assign_paper_genes_to_intervals(
        annotations, positions,
    )
    assert "overlap" not in assignments
    assert assignments["assigned"]["interval_index"] == 2
    assert assignments["assigned"]["n_qualifying_intervals"] == 2

    mask, metadata = script_module.build_interval_gene_masks(
        annotations, assignments, 2, "chr1", 22, 30,
    )
    assert metadata["gene_id"].tolist() == ["assigned"]
    assert mask[:, 0].sum() == 8
    assert metadata.iloc[0]["exon_fraction_in_interval"] == pytest.approx(0.8)


def test_paper_correlations_combine_strands_by_sample(script_module):
    values = {
        "gene1": (0.0, 3.0),
        "gene2": (1.0, 1.0),
        "gene3": (3.0, 0.0),
    }
    rows = []
    for gene_index, (gene_id, sample_values) in enumerate(values.items()):
        strand = "+" if gene_index % 2 == 0 else "-"
        for sample_index, value in enumerate(sample_values):
            rows.append({
                "bin_size_bp": 1,
                "gene_id": gene_id,
                "sample": f"sample{sample_index}",
                # Plus and minus rows originate from different physical tracks,
                # but are one strand-matched biological track in the paper.
                "track_index": 2 * sample_index + (strand == "-"),
                "pred_log1p_mean": value,
                "obs_log1p_mean": value,
            })

    summary, per_track, per_gene = script_module.summarize_gene_expression(
        pd.DataFrame(rows), (1,),
    )
    metrics = summary[1]
    assert metrics["raw_pearson_r_mean_across_tracks"] == pytest.approx(1.0)
    assert metrics[
        "normalized_pearson_r_mean_across_genes_by_track"
    ] == pytest.approx(1.0)
    assert metrics[
        "normalized_pearson_r_mean_across_tracks_by_gene"
    ] == pytest.approx(1.0)
    assert len(per_track) == 2
    assert metrics["n_normalized_gene_correlations"] == 2
    assert len(per_gene) == 3


def test_quantile_normalization_averages_tied_ranks(script_module):
    values = np.asarray([
        [0.0, 2.0],
        [0.0, 1.0],
        [2.0, 0.0],
    ])
    normalized = script_module.quantile_normalize_columns(values)
    # Reference quantiles are [0, 0.5, 2]. The tied zeros in column 0 occupy
    # the first two ranks and both receive their mean, 0.25.
    np.testing.assert_allclose(normalized[:, 0], [0.25, 0.25, 2.0])
    np.testing.assert_allclose(normalized[:, 1], [2.0, 0.5, 0.0])
