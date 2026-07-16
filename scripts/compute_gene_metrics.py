#!/usr/bin/env python
"""Evaluate an AlphaGenome checkpoint on RNA profile and gene metrics.

RNA profile Pearson follows the AlphaGenome paper by applying log(1 + x).
Jensen-Shannon metrics use raw, non-negative profiles. Regional total-count
metrics are computed from the cropped 1-bp arrays and are therefore identical
for every reported display resolution.

Gene expression follows the AlphaGenome paper protocol: log1p of mean coverage
over unioned GENCODE v46 exon bases, strand matching, one assignment per gene
when at least 50% of its unique exon bases fall in a test interval, and raw plus
quantile-normalized correlations across genes and across tracks.

Example:

    python scripts/compute_gene_metrics.py \
        --checkpoint best_model.pth \
        --pretrained-weights model_fold1.pth \
        --genome GRCh38.fa \
        --bigwig sample_forward.bw sample_reverse.bw \
        --test-bed fold_1/test.bed \
        --gtf-parquet gencode.v46.annotation.gtf.parquet \
        --samples sample \
        --sequence-length 1048576 \
        --score-window-bp 196608 \
        --gene-score-window-bp 196608 \
        --bin-sizes 1,32 \
        --output-dir evaluation/rna_seq
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


log = logging.getLogger(__name__)


def _pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with the same finite/constant guards as ATAC eval."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) <= 1e-10 or np.std(y) <= 1e-10:
        return float("nan")
    return float(stats.pearsonr(x, y)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an AlphaGenome checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument(
        "--bigwig", nargs="+", required=True,
        help="RNA-seq coverage BigWigs in checkpoint track order",
    )
    parser.add_argument("--test-bed", required=True, help="fold_1 test BED")
    parser.add_argument(
        "--gtf-parquet", required=True,
        help=(
            "GENCODE v46 GTF parquet containing exon coordinates and gene "
            "metadata; the path must identify v46 or release_46"
        ),
    )
    parser.add_argument(
        "--samples", nargs="+", default=None,
        help=(
            "Sample names for interleaved forward/reverse track pairs. If omitted, "
            "sample_0, sample_1, ... are used."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=131_072)
    parser.add_argument(
        "--score-window-bp", type=int, default=None,
        help="Centered post-inference width for regional profile metrics",
    )
    parser.add_argument(
        "--gene-score-window-bp", type=int, default=None,
        help=(
            "Centered post-inference test width for the paper's >=50%% exon "
            "selection and exon-mean metrics; omit to use the full inference "
            "interval"
        ),
    )
    parser.add_argument("--source-resolution", type=int, default=1)
    parser.add_argument("--bin-sizes", default="1,32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def _clean_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def validate_gencode_v46_path(path: str) -> None:
    """Require annotation provenance identifying GENCODE release 46."""
    normalized_path = str(Path(path)).lower().replace("-", "_")
    if "v46" not in normalized_path and "release_46" not in normalized_path:
        raise ValueError(
            "AlphaGenome paper gene metrics require GENCODE v46; "
            f"the annotation path does not identify v46: {path}"
        )


def load_gtf(gtf_parquet: str) -> pd.DataFrame:
    """Load the GENCODE v46 columns needed for paper gene-expression metrics."""
    validate_gencode_v46_path(gtf_parquet)
    return pd.read_parquet(
        gtf_parquet,
        columns=[
            "Chromosome", "Start", "End", "Strand", "Feature",
            "gene_id", "gene_name", "gene_type", "transcript_id",
        ],
    )


def downsample_gene_mask_weights(mask: np.ndarray, resolution: int) -> np.ndarray:
    """Count exact exonic bases per output bin for paper exon means.

    Weighting a mean-pooled coarse prediction by this count is equivalent to
    upsampling the prediction and averaging it over the original 1-bp mask.
    """
    if resolution == 1:
        return mask.astype(np.int64)
    if mask.shape[0] % resolution:
        raise ValueError(
            f"Gene mask length {mask.shape[0]} is not divisible by {resolution}"
        )
    return mask.reshape(
        mask.shape[0] // resolution, resolution, -1,
    ).sum(axis=1)


def score_gene_interval_mean(tracks: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Mean coverage using boolean masks or per-bin exon-base weights."""
    return np.einsum("lt,lg->gt", tracks, masks) / np.expand_dims(
        masks.sum(axis=0), axis=-1,
    )


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return the genomic union of half-open intervals."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def build_paper_gene_annotations(gtf: pd.DataFrame) -> pd.DataFrame:
    """Build one GENCODE gene record with a union of all annotated exons.

    GENCODE can contain overlapping exon records from multiple transcripts.
    Treating the exon annotation as a genomic mask, as AlphaGenome's public
    GeneIntervalScorer does, ensures every exonic base contributes once.
    """
    exons = gtf[gtf["Feature"] == "exon"].copy()
    if exons.empty:
        raise ValueError("GENCODE annotation contains no exon rows")

    records: list[dict] = []
    for gene_id, gene_exons in exons.groupby("gene_id", sort=False):
        chromosomes = gene_exons["Chromosome"].dropna().unique()
        strands = gene_exons["Strand"].dropna().unique()
        if len(chromosomes) != 1 or len(strands) != 1:
            raise ValueError(
                f"Gene {gene_id!r} has inconsistent chromosome or strand "
                "annotations"
            )
        intervals = _merge_intervals(list(zip(
            gene_exons["Start"].astype(int),
            gene_exons["End"].astype(int),
        )))
        exon_bp = sum(end - start for start, end in intervals)
        if exon_bp == 0:
            continue
        first = gene_exons.iloc[0]
        records.append({
            "gene_id": gene_id,
            "gene_name": first.get("gene_name", gene_id),
            "gene_type": first.get("gene_type", None),
            "chrom": chromosomes[0],
            "strand": strands[0],
            "exon_intervals": intervals,
            "total_unique_exon_bp": exon_bp,
            "exon_start": intervals[0][0],
            "exon_end": intervals[-1][1],
        })
    return pd.DataFrame.from_records(records).set_index("gene_id", drop=False)


def _exon_overlap_bp(
    intervals: tuple[tuple[int, int], ...], start: int, end: int,
) -> int:
    return sum(
        max(0, min(exon_end, end) - max(exon_start, start))
        for exon_start, exon_end in intervals
    )


def assign_paper_genes_to_intervals(
    gene_annotations: pd.DataFrame,
    positions: list[tuple[str, int, int]],
    minimum_exon_fraction: float = 0.5,
) -> dict[str, dict]:
    """Assign genes satisfying the paper's >=50% exon criterion exactly once.

    The percentage is calculated over unique annotated exon bases. If
    overlapping test intervals both qualify, use the interval containing the
    greatest number of exon bases, then the closest interval center, then BED
    order. The tie-break normally has no effect but makes deduplication stable.
    """
    if not 0 < minimum_exon_fraction <= 1:
        raise ValueError("minimum_exon_fraction must be in (0, 1]")

    candidates: dict[str, list[dict]] = {}
    for interval_index, (chrom, start, end) in enumerate(positions):
        chrom_genes = gene_annotations[
            (gene_annotations["chrom"] == chrom)
            & (gene_annotations["exon_start"] < end)
            & (gene_annotations["exon_end"] > start)
        ]
        center = (start + end) / 2
        for gene in chrom_genes.itertuples():
            overlap_bp = _exon_overlap_bp(gene.exon_intervals, start, end)
            fraction = overlap_bp / gene.total_unique_exon_bp
            if fraction < minimum_exon_fraction:
                continue
            exon_center = (gene.exon_start + gene.exon_end) / 2
            candidates.setdefault(gene.gene_id, []).append({
                "interval_index": interval_index,
                "exon_overlap_bp": overlap_bp,
                "exon_fraction_in_interval": fraction,
                "center_distance_bp": abs(exon_center - center),
            })

    assignments: dict[str, dict] = {}
    for gene_id, gene_candidates in candidates.items():
        best = min(gene_candidates, key=lambda item: (
            -item["exon_overlap_bp"],
            item["center_distance_bp"],
            item["interval_index"],
        ))
        assignments[gene_id] = {
            **best,
            "n_qualifying_intervals": len(gene_candidates),
        }
    return assignments


def build_interval_gene_masks(
    gene_annotations: pd.DataFrame,
    assignments: dict[str, dict],
    interval_index: int,
    chrom: str,
    start: int,
    end: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build union-exon masks for genes assigned to one scored interval."""
    gene_ids = [
        gene_id for gene_id, assignment in assignments.items()
        if assignment["interval_index"] == interval_index
    ]
    mask = np.zeros((end - start, len(gene_ids)), dtype=bool)
    metadata_rows: list[dict] = []
    for gene_index, gene_id in enumerate(gene_ids):
        gene = gene_annotations.loc[gene_id]
        if gene["chrom"] != chrom:
            raise ValueError(f"Assigned gene {gene_id!r} is on the wrong chromosome")
        for exon_start, exon_end in gene["exon_intervals"]:
            relative_start = max(exon_start, start) - start
            relative_end = min(exon_end, end) - start
            if relative_end > relative_start:
                mask[relative_start:relative_end, gene_index] = True
        assignment = assignments[gene_id]
        metadata_rows.append({
            "gene_id": gene_id,
            "gene_name": gene["gene_name"],
            "gene_type": gene["gene_type"],
            "strand": gene["strand"],
            "total_unique_exon_bp": int(gene["total_unique_exon_bp"]),
            "scored_unique_exon_bp": int(mask[:, gene_index].sum()),
            "exon_fraction_in_interval": float(
                assignment["exon_fraction_in_interval"]
            ),
            "n_qualifying_intervals": int(
                assignment["n_qualifying_intervals"]
            ),
        })
    return mask, pd.DataFrame(metadata_rows)


def center_crop_genomic_positions(
    positions: list[tuple[str, int, int]],
    score_window_bp: int | None,
    source_resolution: int = 1,
) -> list[tuple[str, int, int]]:
    """Return genomic coordinates matching a centered profile-array crop."""
    if score_window_bp is None:
        return positions
    if (
        score_window_bp <= 0
        or source_resolution <= 0
        or score_window_bp % source_resolution
    ):
        raise ValueError(
            "gene_score_window_bp must be positive and divisible by "
            "source_resolution"
        )

    cropped = []
    for chromosome, start, end in positions:
        available_bp = end - start
        if score_window_bp > available_bp:
            raise ValueError(
                f"Requested {score_window_bp:,} bp gene scoring window, but "
                f"interval {chromosome}:{start}-{end} spans only "
                f"{available_bp:,} bp"
            )
        if available_bp % source_resolution:
            raise ValueError(
                f"Interval width {available_bp} is not divisible by source "
                f"resolution {source_resolution}"
            )
        available_bins = available_bp // source_resolution
        score_bins = score_window_bp // source_resolution
        left_bins = (available_bins - score_bins) // 2
        crop_start = start + left_bins * source_resolution
        cropped.append((chromosome, crop_start, crop_start + score_window_bp))
    return cropped


def align_test_intervals_to_dataset(
    test_bed: str,
    dataset: GenomicDataset,
) -> pd.DataFrame:
    """Match original BED intervals to the windows retained by GenomicDataset."""
    intervals = pd.read_csv(
        test_bed,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    retained: list[dict] = []
    position_index = 0
    half_length = dataset.sequence_length // 2

    for original_index, row in intervals.iterrows():
        chrom = row["chrom"]
        start, end = int(row["start"]), int(row["end"])
        if chrom not in dataset._chrom_sizes:
            continue
        if end - start == dataset.sequence_length:
            window_start, window_end = start, end
        else:
            center = (start + end) // 2
            window_start = center - half_length
            window_end = window_start + dataset.sequence_length
        if window_start < 0 or window_end > dataset._chrom_sizes[chrom]:
            continue

        expected = (chrom, window_start, window_end)
        if position_index >= len(dataset._positions_list):
            raise ValueError("More valid BED intervals than GenomicDataset positions")
        actual = dataset._positions_list[position_index]
        if expected != actual:
            raise ValueError(
                f"BED/dataset interval mismatch at retained index {position_index}: "
                f"expected {expected}, found {actual}"
            )
        retained.append({
            "original_interval_idx": int(original_index),
            "chrom": chrom,
            "start": start,
            "end": end,
            "window_start": window_start,
            "window_end": window_end,
        })
        position_index += 1

    if position_index != len(dataset._positions_list):
        raise ValueError(
            f"Matched {position_index} intervals but dataset retained "
            f"{len(dataset._positions_list)}"
        )
    return pd.DataFrame(retained)


def collect_gene_expression_rows(
    gtf: pd.DataFrame,
    test_intervals: pd.DataFrame,
    positions: list[tuple[str, int, int]],
    prediction_views: dict[int, np.ndarray],
    target_views: dict[int, np.ndarray],
    bin_sizes: tuple[int, ...],
    sample_names: list[str],
) -> pd.DataFrame:
    """Collect paper-protocol exon-mean expression per gene and sample."""
    columns = [
        "bin_size_bp", "gene_id", "gene_name", "chrom", "strand",
        "interval_index", "original_interval_index", "track_index", "sample",
        "total_unique_exon_bp", "scored_unique_exon_bp",
        "exon_fraction_in_interval", "n_qualifying_intervals",
        "pred_mean", "obs_mean", "pred_log1p_mean", "obs_log1p_mean",
    ]
    rows: list[dict] = []
    gene_annotations = build_paper_gene_annotations(gtf)
    assignments = assign_paper_genes_to_intervals(gene_annotations, positions)
    for interval_index, (chrom, window_start, window_end) in enumerate(positions):
        one_bp_masks, metadata = build_interval_gene_masks(
            gene_annotations=gene_annotations,
            assignments=assignments,
            interval_index=interval_index,
            chrom=chrom,
            start=window_start,
            end=window_end,
        )
        if metadata.empty:
            continue
        for bin_size in bin_sizes:
            masks = downsample_gene_mask_weights(one_bp_masks, bin_size)
            pred_means = score_gene_interval_mean(
                prediction_views[bin_size][interval_index], masks,
            )
            target_means = score_gene_interval_mean(
                target_views[bin_size][interval_index], masks,
            )
            for gene_index, gene in metadata.iterrows():
                strand = gene["strand"]
                track_indices = range(
                    0 if strand == "+" else 1, len(sample_names) * 2, 2,
                )
                for track_index in track_indices:
                    if track_index >= pred_means.shape[1]:
                        continue
                    rows.append({
                        "bin_size_bp": bin_size,
                        "gene_id": gene["gene_id"],
                        "gene_name": gene["gene_name"],
                        "chrom": chrom,
                        "strand": strand,
                        "interval_index": interval_index,
                        "original_interval_index": int(
                            test_intervals.iloc[interval_index][
                                "original_interval_idx"
                            ]
                        ),
                        "track_index": track_index,
                        "sample": sample_names[track_index // 2],
                        "total_unique_exon_bp": int(
                            gene["total_unique_exon_bp"]
                        ),
                        "scored_unique_exon_bp": int(
                            gene["scored_unique_exon_bp"]
                        ),
                        "exon_fraction_in_interval": float(
                            gene["exon_fraction_in_interval"]
                        ),
                        "n_qualifying_intervals": int(
                            gene["n_qualifying_intervals"]
                        ),
                        "pred_mean": float(pred_means[gene_index, track_index]),
                        "obs_mean": float(target_means[gene_index, track_index]),
                        "pred_log1p_mean": float(np.log1p(
                            pred_means[gene_index, track_index]
                        )),
                        "obs_log1p_mean": float(np.log1p(
                            target_means[gene_index, track_index]
                        )),
                    })
    return pd.DataFrame(rows, columns=columns)


def quantile_normalize_columns(values: np.ndarray) -> np.ndarray:
    """Quantile-normalize a genes-by-tracks matrix across its track columns.

    Predicted and observed matrices are normalized independently. Tied values
    receive the mean reference quantile across all ranks occupied by the tie.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("quantile normalization expects a 2D matrix")
    if not np.isfinite(values).all():
        raise ValueError("quantile normalization requires finite values")
    if values.size == 0:
        return values.copy()

    sorted_values = np.sort(values, axis=0, kind="mergesort")
    reference_quantiles = sorted_values.mean(axis=1)
    normalized = np.empty_like(values)
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        order = np.argsort(column, kind="mergesort")
        sorted_column = column[order]
        group_start = 0
        while group_start < len(order):
            group_end = group_start + 1
            while (
                group_end < len(order)
                and sorted_column[group_end] == sorted_column[group_start]
            ):
                group_end += 1
            normalized[order[group_start:group_end], column_index] = (
                reference_quantiles[group_start:group_end].mean()
            )
            group_start = group_end
    return normalized


def _paper_expression_matrices(
    subset: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Return aligned complete gene-by-sample log-expression matrices."""
    if subset.duplicated(["gene_id", "sample"]).any():
        duplicates = subset.loc[
            subset.duplicated(["gene_id", "sample"], keep=False),
            ["gene_id", "sample"],
        ]
        raise ValueError(
            "Expected one strand-matched value per gene and sample; found "
            f"duplicates including {duplicates.iloc[0].to_dict()}"
        )

    sample_order = list(dict.fromkeys(subset["sample"].tolist()))
    pred = subset.pivot(index="gene_id", columns="sample", values="pred_log1p_mean")
    obs = subset.pivot(index="gene_id", columns="sample", values="obs_log1p_mean")
    gene_order = sorted(set(pred.index).intersection(obs.index))
    pred = pred.reindex(index=gene_order, columns=sample_order)
    obs = obs.reindex(index=gene_order, columns=sample_order)
    complete = pred.notna().all(axis=1) & obs.notna().all(axis=1)
    pred = pred.loc[complete]
    obs = obs.loc[complete]
    return (
        pred.index.tolist(),
        sample_order,
        pred.to_numpy(dtype=np.float64),
        obs.to_numpy(dtype=np.float64),
    )


def summarize_gene_expression(
    rows: pd.DataFrame,
    bin_sizes: tuple[int, ...],
) -> tuple[dict[int, dict], pd.DataFrame, pd.DataFrame]:
    """Compute the three AlphaGenome paper gene-expression correlations."""
    summary: dict[int, dict] = {}
    per_track_rows: list[dict] = []
    per_gene_rows: list[dict] = []
    for bin_size in bin_sizes:
        subset = rows[rows["bin_size_bp"] == bin_size]
        gene_ids, samples, pred, obs = _paper_expression_matrices(subset)

        pred_quantile = quantile_normalize_columns(pred)
        obs_quantile = quantile_normalize_columns(obs)
        pred_normalized = pred_quantile - pred_quantile.mean(axis=1, keepdims=True)
        obs_normalized = obs_quantile - obs_quantile.mean(axis=1, keepdims=True)

        raw_across_genes: list[float] = []
        normalized_across_genes: list[float] = []
        for track_index, sample in enumerate(samples):
            raw_pearson = _pearson_or_nan(pred[:, track_index], obs[:, track_index])
            normalized_pearson = _pearson_or_nan(
                pred_normalized[:, track_index],
                obs_normalized[:, track_index],
            )
            raw_across_genes.append(raw_pearson)
            normalized_across_genes.append(normalized_pearson)
            per_track_rows.append({
                "bin_size_bp": bin_size,
                "sample": sample,
                "raw_pearson_r_across_genes": raw_pearson,
                "normalized_pearson_r_across_genes": normalized_pearson,
                "n_genes": len(gene_ids),
            })

        normalized_across_tracks: list[float] = []
        for gene_index, gene_id in enumerate(gene_ids):
            pearson = _pearson_or_nan(
                pred_normalized[gene_index], obs_normalized[gene_index],
            )
            normalized_across_tracks.append(pearson)
            per_gene_rows.append({
                "bin_size_bp": bin_size,
                "gene_id": gene_id,
                "normalized_pearson_r_across_tracks": pearson,
                "n_tracks": len(samples),
            })

        raw_finite = np.asarray(raw_across_genes, dtype=np.float64)
        raw_finite = raw_finite[np.isfinite(raw_finite)]
        normalized_gene_finite = np.asarray(
            normalized_across_genes, dtype=np.float64,
        )
        normalized_gene_finite = normalized_gene_finite[
            np.isfinite(normalized_gene_finite)
        ]
        normalized_track_finite = np.asarray(
            normalized_across_tracks, dtype=np.float64,
        )
        normalized_track_finite = normalized_track_finite[
            np.isfinite(normalized_track_finite)
        ]
        summary[bin_size] = {
            "raw_pearson_r_mean_across_tracks": (
                float(raw_finite.mean()) if raw_finite.size else float("nan")
            ),
            "normalized_pearson_r_mean_across_genes_by_track": (
                float(normalized_gene_finite.mean())
                if normalized_gene_finite.size else float("nan")
            ),
            "normalized_pearson_r_mean_across_tracks_by_gene": (
                float(normalized_track_finite.mean())
                if normalized_track_finite.size else float("nan")
            ),
            "n_genes": len(gene_ids),
            "n_tracks": len(samples),
            "n_raw_track_correlations": int(raw_finite.size),
            "n_normalized_track_correlations": int(
                normalized_gene_finite.size
            ),
            "n_normalized_gene_correlations": int(
                normalized_track_finite.size
            ),
        }
    return (
        summary,
        pd.DataFrame(per_track_rows),
        pd.DataFrame(per_gene_rows),
    )


def _write_per_region_csv(
    path: Path,
    positions: list[tuple[str, int, int]],
    metrics_by_bin: dict[int, dict],
    pred_totals: np.ndarray,
    target_totals: np.ndarray,
) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bin_size_bp", "region_index", "chrom", "start", "end",
                "profile_pearson_r", "js_divergence", "js_distance",
                "pred_total", "target_total",
            ],
        )
        writer.writeheader()
        for bin_size, metrics in metrics_by_bin.items():
            for index, (chrom, start, end) in enumerate(positions):
                writer.writerow({
                    "bin_size_bp": bin_size,
                    "region_index": index,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "profile_pearson_r": metrics["profile_pearson_r_all"][index],
                    "js_divergence": metrics["js_divergence_all"][index],
                    "js_distance": metrics["js_distance_all"][index],
                    # One total per region is expected for the usual single
                    # RNA coverage track. Multiple tracks are summed here and
                    # remain available individually in saved predictions.
                    "pred_total": pred_totals[index].sum(),
                    "target_total": target_totals[index].sum(),
                })


def main() -> None:
    # Keep model-framework imports out of module scope. Borzoi reuses the
    # NumPy/Pandas gene-metric helpers from this file and must not load
    # PyTorch's CUDA runtime into its TensorFlow process.
    import torch
    from torch.utils.data import DataLoader, Subset

    from alphagenome_pytorch.extensions.finetuning.datasets import GenomicDataset
    from alphagenome_pytorch.extensions.finetuning.evaluation import evaluate_split
    from alphagenome_pytorch.extensions.finetuning.training import collate_genomic
    from evaluate_checkpoint import (
        build_metric_views,
        center_crop_profiles,
        compute_comparison_metrics,
        load_finetuned_model,
        parse_metric_bin_sizes,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        bin_sizes = parse_metric_bin_sizes(args.bin_sizes)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    assert bin_sizes is not None

    if args.score_window_bp is not None and args.score_window_bp > args.sequence_length:
        sys.exit(
            "Error: --score-window-bp cannot exceed --sequence-length. "
            "Do not request the Borzoi window for a 131-kbp checkpoint."
        )
    if (
        args.gene_score_window_bp is not None
        and args.gene_score_window_bp > args.sequence_length
    ):
        sys.exit(
            "Error: --gene-score-window-bp cannot exceed --sequence-length."
        )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint_meta = load_finetuned_model(
        args.checkpoint, args.pretrained_weights, device,
    )
    modality = checkpoint_meta["modality"]
    if isinstance(modality, list):
        if len(modality) != 1:
            sys.exit(f"Error: expected one RNA modality, found {modality}")
        modality = modality[0]
    if modality not in {"rna_seq", "rna-seq", "rnaseq"}:
        sys.exit(f"Error: checkpoint modality is {modality!r}, not RNA-seq")

    resolutions = checkpoint_meta["resolutions"]
    if isinstance(resolutions, dict):
        resolutions = resolutions.get(modality, (args.source_resolution,))
    resolutions = tuple(resolutions)
    if args.source_resolution not in resolutions:
        sys.exit(
            f"Error: source resolution {args.source_resolution} is not in "
            f"checkpoint resolutions {resolutions}"
        )

    dataset = GenomicDataset(
        genome_fasta=args.genome,
        bigwig_files=args.bigwig,
        bed_file=args.test_bed,
        resolutions=resolutions,
        sequence_length=args.sequence_length,
    )
    test_intervals = align_test_intervals_to_dataset(args.test_bed, dataset)
    if args.max_regions > 0 and len(dataset) > args.max_regions:
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(dataset), args.max_regions, replace=False,
        ).tolist()
        positions = [dataset._positions_list[index] for index in indices]
        test_intervals = test_intervals.iloc[indices].reset_index(drop=True)
        evaluation_dataset = Subset(dataset, indices)
    else:
        positions = dataset._positions_list
        test_intervals = test_intervals.reset_index(drop=True)
        evaluation_dataset = dataset

    if len(args.bigwig) % 2:
        sys.exit(
            "Error: gene-expression evaluation expects interleaved "
            "forward/reverse BigWig track pairs"
        )
    n_samples = len(args.bigwig) // 2
    sample_names = args.samples or [f"sample_{index}" for index in range(n_samples)]
    if len(sample_names) != n_samples:
        sys.exit(
            f"Error: expected {n_samples} --samples values for {len(args.bigwig)} "
            f"interleaved tracks, received {len(sample_names)}"
        )

    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_genomic,
    )
    predictions, targets, loss = evaluate_split(
        model, modality, loader, device, resolutions,
    )
    source = args.source_resolution
    cropped_predictions, cropped_targets = center_crop_profiles(
        predictions[source], targets[source], args.score_window_bp, source,
    )
    prediction_views, target_views = build_metric_views(
        predictions[source],
        targets[source],
        source_resolution=source,
        bin_sizes=bin_sizes,
        score_window_bp=args.score_window_bp,
        reduction="sum",
    )
    gene_prediction_views, gene_target_views = build_metric_views(
        predictions[source],
        targets[source],
        source_resolution=source,
        bin_sizes=bin_sizes,
        score_window_bp=args.gene_score_window_bp,
        reduction="mean",
    )
    gene_positions = center_crop_genomic_positions(
        positions, args.gene_score_window_bp, source,
    )

    # Regional counts are defined from the cropped 1-bp signal, not from the
    # mean-pooled display profiles.
    count_metrics = compute_comparison_metrics(
        cropped_predictions, cropped_targets, profile_transform="log1p",
    )
    pred_totals = cropped_predictions.sum(axis=1)
    target_totals = cropped_targets.sum(axis=1)

    metrics_by_bin: dict[int, dict] = {}
    for bin_size in bin_sizes:
        metrics = compute_comparison_metrics(
            prediction_views[bin_size],
            target_views[bin_size],
            profile_transform="log1p",
        )
        for key in ("count_pearson_r", "count_pearson_r_raw", "count_pearson_r_log1p"):
            metrics[key] = count_metrics[key]
        metrics_by_bin[bin_size] = metrics
        log.info(
            "%dbp: profile r=%.4f, accumulated r=%.4f, log-count r=%.4f, "
            "JS distance=%.4f",
            bin_size,
            metrics["profile_pearson_r_mean"],
            metrics["track_pearson_r_accumulated_mean"],
            metrics["count_pearson_r_log1p"],
            metrics["js_distance_mean"],
        )

    gtf = load_gtf(args.gtf_parquet)
    gene_rows = collect_gene_expression_rows(
        gtf=gtf,
        test_intervals=test_intervals,
        positions=gene_positions,
        prediction_views=gene_prediction_views,
        target_views=gene_target_views,
        bin_sizes=bin_sizes,
        sample_names=sample_names,
    )
    (
        gene_metrics,
        gene_metrics_per_track,
        gene_metrics_per_gene,
    ) = summarize_gene_expression(gene_rows, bin_sizes)
    for bin_size, metrics in gene_metrics.items():
        log.info(
            "%dbp gene expression: raw across-genes r=%.4f, normalized "
            "across-genes r=%.4f, normalized across-tracks r=%.4f "
            "(%d genes, %d tracks)",
            bin_size,
            metrics["raw_pearson_r_mean_across_tracks"],
            metrics["normalized_pearson_r_mean_across_genes_by_track"],
            metrics["normalized_pearson_r_mean_across_tracks_by_gene"],
            metrics["n_genes"],
            metrics["n_tracks"],
        )

    summary = {
        "checkpoint": checkpoint_meta,
        "evaluation": {
            "modality": modality,
            "test_bed": args.test_bed,
            "fold": "fold_1 expected; determined by --test-bed",
            "sequence_length_bp": args.sequence_length,
            "score_window_bp": args.score_window_bp or args.sequence_length,
            "source_resolution_bp": source,
            "bin_sizes_bp": list(bin_sizes),
            "profile_transform": "log1p",
            "profile_pooling": "sum",
            "count_source": "cropped_source_resolution_profile",
            "gene_score_window_bp": (
                args.gene_score_window_bp or args.sequence_length
            ),
            "gene_expression_window": (
                "center_crop" if args.gene_score_window_bp is not None
                else "full_inference_test_interval"
            ),
            "gene_annotation": "GENCODE_v46_required",
            "gene_selection": "at_least_50pct_unique_exon_bp_in_test_interval",
            "gene_interval_assignment": (
                "maximum_unique_exon_overlap_then_center_distance_then_bed_order"
            ),
            "gene_mask_aggregation": "union_all_annotated_exons_then_mean",
            "gene_expression_transform": "log1p_mean_exonic_coverage",
            "gene_strand_matching": (
                "forward_for_plus_reverse_for_minus_combined_per_sample"
            ),
            "gene_correlations": [
                "raw_across_genes_per_track",
                "quantile_normalized_gene_centered_across_genes_per_track",
                "quantile_normalized_gene_centered_across_tracks_per_gene",
            ],
            "quantile_normalization": (
                "predicted_and_observed_gene_by_track_matrices_independently_"
                "normalized_across_track_columns"
            ),
            "gtf_parquet": args.gtf_parquet,
            "n_regions": len(positions),
            "loss": loss,
        },
        "metrics": metrics_by_bin,
        "gene_expression_metrics": gene_metrics,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(_clean_json(summary), handle, indent=2, default=str)
    _write_per_region_csv(
        output_dir / "metrics_per_region.csv",
        positions,
        metrics_by_bin,
        pred_totals,
        target_totals,
    )
    gene_rows.to_csv(output_dir / "gene_expression_per_gene.csv", index=False)
    gene_metrics_per_track.to_csv(
        output_dir / "gene_expression_metrics_per_track.csv", index=False,
    )
    gene_metrics_per_gene.to_csv(
        output_dir / "gene_expression_metrics_per_gene.csv", index=False,
    )

    if args.save_predictions:
        prediction_dir = output_dir / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        np.save(
            prediction_dir / "predictions_scored_1bp.npy",
            cropped_predictions.astype(np.float16),
        )
        np.save(
            prediction_dir / "targets_scored_1bp.npy",
            cropped_targets.astype(np.float16),
        )
        for bin_size in bin_sizes:
            if bin_size == source:
                continue
            np.save(
                prediction_dir / f"predictions_scored_{bin_size}bp.npy",
                prediction_views[bin_size].astype(np.float16),
            )
            np.save(
                prediction_dir / f"targets_scored_{bin_size}bp.npy",
                target_views[bin_size].astype(np.float16),
            )

    log.info("Wrote RNA-seq evaluation to %s", output_dir)


if __name__ == "__main__":
    main()
